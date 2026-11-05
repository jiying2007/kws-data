#!/usr/bin/env python3
"""Replay additive context-v2 native/ASR evidence without changing v1 packages.

The independent context plan and its explicit job owner are authenticated before
the unchanged whole-transcript weak policy is applied. Context position and
repeat counts come only from observed transcripts, never requested prompts.
Legacy consumers retain their vocabulary meaning; full2601 is a separate pinned
consumer projection of this package's complete observed rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_weak_view as weak

ROOT = Path(__file__).resolve().parents[1]
BULK = "research/auto-tts-voice-clone-20261102/"
sys.path.insert(0, str(ROOT / BULK))
import context_supplement_v2 as pool

SCHEMA = "kws-active-context-v2-research-weak-view-v1"
SOURCE_ADAPTER = "kws-active-context-v2-native-source-v1"
PRODUCER_FILES = ("automation/active_context_v2_weak_view.py", "automation/active_asr_weak_view.py",
                  BULK + "context_supplement_v2.py", BULK + "context_supplement.py",
                  BULK + "d2_pool.py", BULK + "bulk_pool.py",
                  BULK + "bulk_runner.py", BULK + "runner.py", BULK + "campaign.py",
                  "kws_data_contract/__init__.py", "kws_data_contract/identity.py",
                  "kws_data_contract/reference_text.py", "kws_data_contract/source_registry.py")
CONSUMERS = ("rnn-four-token", "fsmn-fulltext")
SPLITS = ("train", "development")
CONSUMER_CONTRACTS = {
    "rnn-four-token": {"legacy": True, "full_transcript_required": True, "oov_policy": "reject-whole-transcript"},
    "fsmn-fulltext": {"legacy": True, "output_count": 2599, "full_transcript_required": True,
                      "oov_policy": "reject-whole-transcript", "full2601_projection": False},
}
require, read_bytes, sha, loads, canonical = weak.require, weak.read_bytes, weak.sha, weak.loads, weak.canonical


def _plan(raw, expected):
    value = loads(raw)
    require(type(expected) is str and weak.SHA.fullmatch(expected) and
            sha(canonical(value)) == expected, "CONTEXT_V2_FROZEN_PLAN_CHANGED")
    require(value.get("schema") == "kws-context-supplement-plan-v2", "CONTEXT_V2_PLAN_SCHEMA_REQUIRED")
    pool.verify_plan(value)
    return value, {row["reference_id"]: row for row in value["references"]}, {
        row["cell_id"]: row for row in value["cells"]}


def _native_call(raw, cell, ref):
    call = loads(raw)
    checked = ref["reference_text_status"] in ("verified", "dataset-provided-content-checked")
    expected = {"schema": "active-tts-generation-call-v1", "cell_id": cell["cell_id"],
                "reference_id": ref["reference_id"], "source_family_id": ref["source_family_id"],
                "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
                "reference_text_status": ref["reference_text_status"],
                "reference_mode": "ICL" if checked else "x-vector-only", "reference_text_passed": checked,
                "seed": cell["seed"], "intended_text": cell["prompt"], "attempt": 1}
    require(set(call) == set(expected) | {"generation", "reference_prompt_seconds"} and
            all(call[key] == value for key, value in expected.items()) and
            type(call["generation"]) is dict and bool(call["generation"]) and
            type(call["reference_prompt_seconds"]) in (int, float) and call["reference_prompt_seconds"] >= 0,
            "CONTEXT_V2_NATIVE_CALL_REFERENCE_MODE_JOIN")
    return call


def _owner(plan, plan_raw, observed, generation, recipe_raw, expected_recipe_sha, stage_raw):
    arguments = (recipe_raw, expected_recipe_sha, stage_raw)
    require(all(v is not None for v in arguments),
            "CONTEXT_V2_JOB_RECIPE_PIN_AND_STAGE_REQUIRED")
    require(type(expected_recipe_sha) is str and weak.SHA.fullmatch(expected_recipe_sha) and
            sha(recipe_raw) == expected_recipe_sha, "CONTEXT_V2_JOB_RECIPE_PIN_CHANGED")
    recipe, stage = loads(recipe_raw), loads(stage_raw)
    require(recipe["schema"] == "kws-context-v2-generation-job-v1" and
            stage["schema"] == "kws-context-v2-generation-stage-v1" and
            {k: recipe["stage"][k] for k in ("bytes", "sha256")} ==
                {"bytes": len(stage_raw), "sha256": sha(stage_raw)} and
            stage["plan"] == {"bytes": len(plan_raw), "sha256": sha(plan_raw)}, "CONTEXT_V2_JOB_STAGE_PLAN_JOIN")
    require(type(recipe["chunk_index"]) is int and 0 <= recipe["chunk_index"] < len(stage["chunks"]) and
            [c for chunk in stage["chunks"] for c in chunk] == [c["cell_id"] for c in plan["cells"]] and
            stage["max_new_calls"] == len(plan["cells"]) and stage["automatic_retries"] == 0 and
            stage["d1_cells_generated_again"] == stage["d2_cells_generated_again"] == 0 and
            stage["context_v1_cells_generated_again"] == 0 and
            stage["model_concurrency"] == 1 and
            stage["human_gold"] is False, "CONTEXT_V2_STAGE_REQUEST_BOUNDARY")
    selected = recipe["selected_cells"]
    require(selected == stage["chunks"][recipe["chunk_index"]] and
            type(recipe["max_calls"]) is int and recipe["max_calls"] == len(selected) and
            recipe["job_id"] == f"generate-{recipe['chunk_index'] + 1:03d}" and
            1 <= len(selected) <= 16 and len(set(selected)) == len(selected) and
            len(set(observed)) == len(observed) and set(observed) <= set(selected) and
            recipe["generation"] == stage["generation"] == generation == plan["generation"],
            "CONTEXT_V2_JOB_OWNER_SCOPE")
    return {"job_recipe_sha256": expected_recipe_sha, "stage_sha256": sha(stage_raw),
            "job_id": recipe["job_id"], "selected_cell_ids": selected,
            "observed_cell_ids": observed,
            "unobserved_selected_cells": [cell for cell in selected if cell not in set(observed)]}


def source_rows(archive_raw, inputs, plan_raw, expected_plan_sha, generation_root, native_root, *,
                job_recipe_raw, expected_job_recipe_sha256, stage_raw):
    """Authenticate actual native returns, not an intended text export.

    generation_root is the shared bulk directory containing plan-lock.json and
    cells/. native_root is one job's outputs/native directory. Retained copies
    use the same layout, so the reader needs neither original mutable root.
    """
    plan, refs, cells = _plan(plan_raw, expected_plan_sha)
    generation_root, native_root = Path(generation_root), Path(native_root)
    evidence = {"context-v2/plan.json": plan_raw}

    def take(root, name, prefix, limit=16 * 1024 * 1024):
        raw = read_bytes(root / weak.relative(name), limit)
        evidence["context-v2/" + prefix + "/" + name] = raw
        return raw

    weak._d1_lock(take(generation_root, "plan-lock.json", "generation"), plan, expected_plan_sha)
    sources, generation = {}, None
    for clip, cell, raw, pcm in weak._d1_audio_rows(archive_raw, inputs, cells):
        name, ref = cell["cell_id"], refs[cell["reference_id"]]
        folder, native_folder = "cells/" + name + "/", name + "/"
        files = {p.name for p in (generation_root / folder).iterdir() if p.is_file()}
        native_files = {p.name for p in (native_root / name).iterdir() if p.is_file()}
        success = "completed.json" in files
        terminal = "completed.json" if success else "failed.json"
        expected_files = {"started.json", terminal} | ({"native-f32.wav", "audio-16k.wav"} if success else set())
        require(files == expected_files and native_files ==
                {"call.json", "returned.json", "native.json", "returned-array.npy", "returned-native.wav"},
                "CONTEXT_V2_NATIVE_FILE_ROSTER")
        started_raw = take(generation_root, folder + "started.json", "generation")
        weak._d1_started(started_raw, cell, ref, expected_plan_sha)
        call_raw = take(native_root, native_folder + "call.json", "native")
        call = _native_call(call_raw, cell, ref)
        generation = call["generation"] if generation is None else generation
        require(call["generation"] == generation, "CONTEXT_V2_MIXED_GENERATION_CONFIGURATION")
        native_raw = take(native_root, native_folder + "native.json", "native")
        native = loads(native_raw)
        array = take(native_root, native_folder + "returned-array.npy", "native")
        returned_wav = take(native_root, native_folder + "returned-native.wav", "native")
        require(native["schema"] == "active-tts-native-return-v1" and
                native["array"] == {"bytes": len(array), "sha256": sha(array)} and
                native["wav"] == {"bytes": len(returned_wav), "sha256": sha(returned_wav)} and
                native["technical_gate_applied"] is False and native["all_finite"] is True and
                native["sample_rate_hz"] == 24000 and native.get("wav_unavailable_reason") is None and
                returned_wav == raw, "CONTEXT_V2_NATIVE_RETURN_JOIN")
        weak._d1_native_array(array, native, pcm)
        returned = loads(take(native_root, native_folder + "returned.json", "native"))
        require(set(returned) == {"generation_seconds", "total_seconds", "native_stop_reason"} and
                returned["native_stop_reason"] == "not_exposed_by_voice_clone_API" and
                all(type(returned[key]) in (int, float) and returned[key] >= 0
                    for key in ("generation_seconds", "total_seconds")), "CONTEXT_V2_NATIVE_RETURN_RECEIPT")
        terminal_raw = take(generation_root, folder + terminal, "generation")
        reason = None
        if success:
            require(take(generation_root, folder + "native-f32.wav", "generation") == raw,
                    "CONTEXT_V2_ARCHIVE_NATIVE_CHANGED")
            weak._d1_completed(terminal_raw, cell, clip, raw, pcm,
                take(generation_root, folder + "audio-16k.wav", "generation"))
        else:
            failure = loads(terminal_raw)
            require(set(failure) == {"schema", "cell_id", "status", "error_type", "training_admitted"} and
                    failure["schema"] == "kws-bulk-clone-failure-v1" and failure["cell_id"] == name and
                    failure["status"] == "FAILED_NO_RETRY" and failure["training_admitted"] is False and
                    type(failure["error_type"]) is str and 1 <= len(failure["error_type"]) <= 128,
                    "CONTEXT_V2_FAILED_NATIVE_RECEIPT")
            reason = "saved_bulk_terminal:" + failure["error_type"]
        source = weak._d1_source(cell, ref, plan, expected_plan_sha, started_raw, terminal_raw,
                                  "context-v2/generation/" + folder, terminal)
        source.update(generation_outcome="completed" if success else "failed-native",
                      technical_failure_reason=reason, outcome_record_sha256=sha(terminal_raw),
                      generation_call_receipt_sha256=sha(call_raw), generation_native_receipt_sha256=sha(native_raw),
                      context_stage=plan["stage"], d2_parent_plan_sha256=plan["d2_parent"]["plan_sha256"],
                      context_parent_plan_sha256=plan["context_parent"]["plan_sha256"],
                      requested_vocabulary_sha256=plan["vocabulary"]["canonical_sha256"],
                      requested_text_preflight=cell["requested_text_preflight"],
                      explicit_additive_revision=cell["explicit_additive_revision"],
                      same_requested_text_as_parent=cell["same_requested_text_as_parent"],
                      requested_context_class=cell["context_class"],
                      requested_keyword_occurrences=cell["requested_keyword_occurrences"],
                      source_manifest_sha256=plan["source_manifest_sha256"],
                      speaker_registry_sha256=plan["speaker_registry"]["sha256"],
                      dataset_id=ref["dataset_id"], source_terms_sha256=ref["source_terms_sha256"],
                      requested_label_authority="intended_only")
        sources[clip["id"]] = source
    observed_ids = [source["source_id"] for source in sources.values()]
    owner = _owner(plan, plan_raw, observed_ids, generation, job_recipe_raw, expected_job_recipe_sha256, stage_raw)
    evidence["context-v2/job-recipe.json"] = job_recipe_raw
    evidence["context-v2/stage.json"] = stage_raw
    return sources, evidence, {"schema": SOURCE_ADAPTER, "plan_sha256": expected_plan_sha,
        "plan_file_sha256": sha(plan_raw), "stage": plan["stage"],
        "d2_parent_plan_sha256": plan["d2_parent"]["plan_sha256"],
        "context_parent_plan_sha256": plan["context_parent"]["plan_sha256"],
        "vocabulary_sha256": plan["vocabulary"]["canonical_sha256"],
        "vocabulary_file_sha256": plan["vocabulary"]["file_sha256"],
        "vocabulary_preflight_scope": "requested_text_only",
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "speaker_registry_sha256": plan["speaker_registry"]["sha256"],
        "generation": generation, "observed_cell_ids": observed_ids, "job_owner": owner,
        "source_files_sha256": {name: sha(raw) for name, raw in evidence.items()},
        "inherited_d1_cells_reissued": 0, "inherited_d2_cells_reissued": 0,
        "inherited_context_v1_cells_reissued": 0,
        "missing_returns_are_not_training_rows": True}


def _join(inputs_path, expected_inputs_sha, archive_path, *, plan_path, plan_sha256, generation_root, native_root,
          job_recipe_path, job_recipe_sha256, stage_path):
    inputs_path = Path(inputs_path)
    inputs_raw = read_bytes(inputs_path)
    require(sha(inputs_raw) == expected_inputs_sha, "ACTIVE_INPUT_MANIFEST_CHANGED")
    inputs = loads(inputs_raw)
    weak.verify_inputs(inputs, inputs_path.parent)
    archive = read_bytes(archive_path, 256 * 1024 * 1024)
    sources, evidence, adapter = source_rows(archive, inputs, read_bytes(plan_path, pool.MAX_PLAN_BYTES),
        plan_sha256, generation_root, native_root,
        job_recipe_raw=read_bytes(job_recipe_path),
        expected_job_recipe_sha256=job_recipe_sha256,
        stage_raw=read_bytes(stage_path))
    return inputs_raw, inputs, archive, sources, evidence, adapter


def build_source_map(inputs_path, expected_inputs_sha, archive_path, **source_options):
    """Audio/source-only routing map, computed before observing ASR output."""
    inputs_raw, _, _, sources, _, adapter = _join(inputs_path, expected_inputs_sha, archive_path, **source_options)
    return weak._source_map(inputs_raw, Path(inputs_path).parent, sources, adapter["plan_sha256"])


def observed_context(text):
    """Character locations in a saved full transcript, never acoustic times."""
    require(type(text) is str, "CONTEXT_V2_OBSERVED_TEXT_REQUIRED")
    normalized = weak.normalize(text)
    keywords = []
    for index, keyword in enumerate(weak.KEYWORDS, 1):
        offsets, start = [], 0
        while True:
            start = normalized.find(keyword, start)
            if start < 0:
                break
            offsets.append(start)
            start += len(keyword)
        if not offsets:
            position = "absent"
        elif len(offsets) > 1:
            position = "repeated-call"
        elif offsets[0] == 0:
            position = "isolated" if len(keyword) == len(normalized) else "prefix"
        else:
            position = "suffix" if offsets[0] + len(keyword) == len(normalized) else "middle"
        keywords.append({"keyword_id": index, "keyword": keyword, "occurrences": len(offsets),
                         "start_offsets": offsets, "observed_context_class": position})
    return {"normalized_text": normalized, "keywords": keywords}


def context_observation(row):
    source, decision = row["source"], row["decision"]
    agreed = observed_context(decision["full_text"]) if decision["full_text"] is not None else None
    actual = agreed["keywords"][source["intended_keyword_id"] - 1] if agreed else None
    confirmed = (decision["status"] == "machine_candidate" and decision["ctc_mask"] is True and
                 decision["keyword_targets"][source["intended_keyword_id"] - 1] == 1)
    return {"basis": "retained-observed-full-transcripts-only", "offset_unit": "normalized-Unicode-codepoints",
            "occurrence_rule": "left-to-right-non-overlapping", "event_times": None,
            "observations": [{"model": item["model"], "usable": item["usable"],
                              **observed_context(item["text"])} for item in row["asr"]
                             if item["status"] == "complete" and type(item.get("text")) is str],
            "agreed_full_transcript": agreed, "confirmed_positive": confirmed,
            "confirmed_context_class": actual["observed_context_class"] if confirmed else None,
            "confirmed_keyword_occurrences": actual["occurrences"] if confirmed else None,
            "requested_context_confirmed": bool(confirmed and
                actual["observed_context_class"] == source["requested_context_class"] and
                actual["occurrences"] == source["requested_keyword_occurrences"])}


def _statistics(rows):
    return {"source_items": len(rows), "source_families": len({r["source"]["source_family_id"] for r in rows}),
            "completed_asr_results": sum(o["status"] == "complete" for r in rows for o in r["asr"]),
            "not_requested_asr_results": sum(o["status"] == "not_requested" for r in rows for o in r["asr"]),
            "machine_candidates": sum(r["decision"]["status"] == "machine_candidate" for r in rows),
            "uncertain": sum(r["decision"]["status"] == "uncertain" for r in rows),
            "disagreement": sum(r["decision"]["disagreement"] for r in rows),
            "train": sum(r["source"]["split"] == "train" for r in rows),
            "development": sum(r["source"]["split"] == "development" for r in rows),
            "observed_confirmed_context_classes": dict(sorted(Counter(
                r["context_observation"]["confirmed_context_class"] for r in rows
                if r["context_observation"]["confirmed_positive"]).items())),
            "requested_context_confirmed": sum(r["context_observation"]["requested_context_confirmed"] for r in rows)}


def _exports(rows):
    result = {"all-observed.jsonl": rows,
              "weak-fulltext.jsonl": [row for row in rows if row["decision"]["status"] == "machine_candidate"],
              "uncertain.jsonl": [row for row in rows if row["decision"]["status"] != "machine_candidate"],
              "disagreement.jsonl": [row for row in rows if row["decision"]["disagreement"]]}
    for consumer in CONSUMERS:
        for split in SPLITS:
            result[consumer + "-" + split + ".jsonl"] = [weak.training_row(row, consumer) for row in rows
                if row["source"]["split"] == split and row["decision"]["ctc_mask"] and
                row["consumers"][consumer]["supported"]]
    return result


def create_view(inputs_path, expected_inputs_sha, archive_path, asr_root, output, *, fsmn_tokens=None,
                selective_source_map, selective_policy, selective_audit_sha256, **source_options):
    inputs_path, asr_root, output = Path(inputs_path), Path(asr_root), Path(output).absolute()
    require(not output.exists() and output.parent.is_dir() and ROOT not in output.parents and
            not any(path.is_symlink() for path in (output, *output.parents)), "NEW_PRIVATE_VIEW_REQUIRED")
    inputs_raw, inputs, archive, sources, evidence, adapter = _join(
        inputs_path, expected_inputs_sha, archive_path, **source_options)
    retained = {}
    observed, asr_adapter = weak.selective_evidence(asr_root, inputs_raw, inputs_path.parent, sources,
        adapter["plan_sha256"], read_bytes(selective_source_map), read_bytes(selective_policy),
        selective_audit_sha256, retained)
    token_raw = read_bytes(fsmn_tokens, 65536) if fsmn_tokens is not None else None
    vocab = weak.fsmn_vocabulary(token_raw) if token_raw is not None else None
    rows, audio, roles = [], {}, weak.SourceRoles()
    for clip in inputs["clips"]:
        row, wav, original, derived = weak.make_row(clip, sources[clip["id"]], inputs_path.parent, asr_root,
            expected_inputs_sha, vocab, retained, observations=observed[clip["id"]])
        row["context_observation"] = context_observation(row)
        weak.add_source_role(roles, sources[clip["id"]], clip, row["audio_derivation"])
        rows.append(row); audio[row["audio"]] = wav
        for name, body in ((clip["derived_pcm"]["path"], derived), (clip["original_wav"]["path"], original)):
            path = "evidence/inputs/" + name
            require(path not in retained or retained[path] == body, "RETAINED_EVIDENCE_CONFLICT")
            retained[path] = body
    producer = {name: read_bytes(ROOT / name) for name in PRODUCER_FILES}
    report = {"schema": SCHEMA, "policy": weak.POLICY_V2, "inputs_sha256": expected_inputs_sha,
        "producer": {"python": sys.version.split()[0], "source_files_sha256": {n: sha(b) for n, b in producer.items()}},
        "source_archive_sha256": inputs["source_archive_sha256"], "items": rows,
        "source_adapter": adapter, "asr_adapter": asr_adapter,
        "split_scope": "source-group-train-and-development-machine-weak-no-independent-test",
        "new_model_calls": 0, "training_runs": 0, "human_gold": False,
        "consumer_contracts": CONSUMER_CONTRACTS,
        "requested_text_is_ground_truth": False, "statistics": _statistics(rows)}
    # Validate every input before creating a package. Writes are exclusive and private.
    output.mkdir(mode=0o700)
    weak.save(output, "evidence/inputs.json", inputs_raw)
    weak.save(output, "evidence/source-original.zip", archive)
    for name, body in evidence.items():
        weak.save(output, "evidence/source/" + name, body)
    for name, body in producer.items():
        weak.save(output, "evidence/view-code/" + name, body)
    if token_raw is not None:
        weak.save(output, "evidence/fsmn-tokens.txt", token_raw)
    for name, body in {**retained, **audio}.items():
        weak.save(output, name, body)
    weak.save_json(output, "report.json", report)
    for name, subset in _exports(rows).items():
        weak.save_rows(output, name, subset)
    weak.save_json(output, "MANIFEST.json", {p.relative_to(output).as_posix(): sha(p.read_bytes())
        for p in sorted(output.rglob("*")) if p.is_file()})
    return report


def verify_view(package, *, expected_manifest_sha256=None, other_split_sources=()):
    """Replay retained source/native/ASR bytes, decisions, audio and every export."""
    package = Path(package).absolute()
    manifest_raw = read_bytes(package / "MANIFEST.json")
    require(expected_manifest_sha256 is None or sha(manifest_raw) == expected_manifest_sha256,
            "CONTEXT_V2_WEAK_MANIFEST_PIN_CHANGED")
    manifest = loads(manifest_raw)
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*")
              if p.is_file() and p != package / "MANIFEST.json"}
    require(type(manifest) is dict and actual == set(manifest), "WEAK_PACKAGE_ROSTER_CHANGED")
    for name, expected in manifest.items():
        path = package / weak.relative(name)
        raw = weak.empty_evidence(path) if expected == sha(b"") else read_bytes(path, 256 * 1024 * 1024)
        require(sha(raw) == expected, "WEAK_PACKAGE_CHANGED")
    report = loads(read_bytes(package / "report.json", 32 * 1024 * 1024))
    require(report["schema"] == SCHEMA and report["policy"] == weak.POLICY_V2 and
            report["human_gold"] is False and report["requested_text_is_ground_truth"] is False and
            report["new_model_calls"] == report["training_runs"] == 0 and
            report["split_scope"] == "source-group-train-and-development-machine-weak-no-independent-test",
            "CONTEXT_V2_WEAK_VIEW_SCHEMA")
    require(report["consumer_contracts"] == CONSUMER_CONTRACTS, "CONTEXT_V2_LEGACY_CONSUMER_CONTRACT_CHANGED")
    require(set(report["producer"]["source_files_sha256"]) == set(PRODUCER_FILES), "CONTEXT_V2_PRODUCER_ROSTER_CHANGED")
    for name, expected in report["producer"]["source_files_sha256"].items():
        require(sha(read_bytes(package / "evidence/view-code" / name)) == expected, "CONTEXT_V2_PRODUCER_SOURCE_CHANGED")
    adapter = report["source_adapter"]
    require(adapter["schema"] == SOURCE_ADAPTER, "CONTEXT_V2_SOURCE_ADAPTER_SCHEMA")
    owner_options = {"job_recipe_path": package / "evidence/source/context-v2/job-recipe.json",
                      "job_recipe_sha256": adapter["job_owner"]["job_recipe_sha256"],
                      "stage_path": package / "evidence/source/context-v2/stage.json"}
    inputs_raw, inputs, _, sources, _, rebuilt = _join(package / "evidence/inputs/inputs.json",
        report["inputs_sha256"], package / "evidence/source-original.zip",
        plan_path=package / "evidence/source/context-v2/plan.json", plan_sha256=adapter["plan_sha256"],
        generation_root=package / "evidence/source/context-v2/generation", native_root=package / "evidence/source/context-v2/native",
        **owner_options)
    require(read_bytes(package / "evidence/inputs.json") == inputs_raw and adapter == rebuilt and
            report["source_archive_sha256"] == inputs["source_archive_sha256"], "CONTEXT_V2_SOURCE_ADAPTER_CHANGED")
    observed, asr_adapter = weak.selective_evidence(package / "evidence/asr", inputs_raw,
        package / "evidence/inputs", sources, adapter["plan_sha256"],
        read_bytes(package / "evidence/selective/source-map.json"), read_bytes(package / "evidence/selective/policy.json"),
        report["asr_adapter"]["final_audit_sha256"], {})
    require(report["asr_adapter"] == asr_adapter, "SELECTIVE_ASR_ADAPTER_CHANGED")
    tokens = package / "evidence/fsmn-tokens.txt"
    vocab = weak.fsmn_vocabulary(read_bytes(tokens, 65536)) if tokens.exists() else None
    require([r["item_id"] for r in report["items"]] == [c["id"] for c in inputs["clips"]], "WEAK_INPUT_ROSTER_CHANGED")
    roles = weak.SourceRoles()
    for source in other_split_sources:
        roles.add(source["role"], family=source.get("source_family_id"), speaker=source.get("speaker_id"),
                  pcm_sha256=source.get("pcm_sha256"), wav_sha256=source.get("wav_sha256"))
    for row, clip in zip(report["items"], inputs["clips"]):
        expected, wav, _, _ = weak.make_row(clip, sources[clip["id"]], package / "evidence/inputs",
            package / "evidence/asr", report["inputs_sha256"], vocab, {}, observations=observed[clip["id"]])
        expected["context_observation"] = context_observation(expected)
        require(row == expected, "WEAK_VIEW_DIFFERS_FROM_RETAINED_EVIDENCE")
        require(read_bytes(package / weak.relative(row["audio"]), 2 * 1024 * 1024) == wav,
                "WEAK_QUANTIZED_AUDIO_CHANGED")
        weak.add_source_role(roles, row["source"], clip, row["audio_derivation"])
    require(report["statistics"] == _statistics(report["items"]), "CONTEXT_V2_WEAK_STATISTICS_CHANGED")
    for name, subset in _exports(report["items"]).items():
        expected = b"".join(canonical(row) + b"\n" for row in subset)
        path = package / name
        raw = read_bytes(path, 32 * 1024 * 1024) if expected else weak.empty_evidence(path)
        require(raw == expected, "CONTEXT_V2_WEAK_EXPORT_CHANGED")
    return report


def _load_rows(package, consumer, split, other_split_sources, expected_manifest_sha256):
    require(consumer in CONSUMERS, "UNKNOWN_WEAK_CONSUMER")
    package = Path(package).absolute()
    report = verify_view(package, expected_manifest_sha256=expected_manifest_sha256,
                         other_split_sources=other_split_sources)
    require(consumer != "fsmn-fulltext" or (package / "evidence/fsmn-tokens.txt").exists(),
            "FSMN_VOCABULARY_NOT_SUPPLIED")
    for source_row in _exports(report["items"])[consumer + "-" + split + ".jsonl"]:
        row = dict(source_row)
        row["audio"] = row["audio_path"] = str(package / row["audio"])
        if consumer == "fsmn-fulltext":
            row.update(recording=row["item_id"], ctc_ids=list(row["target_ids"]),
                       ctc_text=row["full_text"], audio_sha256=row["file_sha256"])
        yield row


def load_training_rows(package, consumer="fsmn-fulltext", *, other_split_sources=(), expected_manifest_sha256=None):
    yield from _load_rows(package, consumer, "train", other_split_sources, expected_manifest_sha256)


def load_development_rows(package, consumer="fsmn-fulltext", *, other_split_sources=(), expected_manifest_sha256=None):
    yield from _load_rows(package, consumer, "development", other_split_sources, expected_manifest_sha256)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for action in ("source-map", "create"):
        command = commands.add_parser(action)
        for name in ("inputs", "inputs-sha256", "archive", "plan", "plan-sha256", "generation-root", "native-root", "output"):
            command.add_argument("--" + name, required=True)
        for name in ("job-recipe", "job-recipe-sha256", "stage"):
            command.add_argument("--" + name, required=True)
        if action == "create":
            for name in ("asr-root", "selective-source-map", "selective-policy", "selective-audit-sha256"):
                command.add_argument("--" + name, required=True)
            command.add_argument("--fsmn-tokens")
    check = commands.add_parser("verify")
    check.add_argument("--package", required=True)
    check.add_argument("--manifest-sha256")
    args = cli.parse_args(argv)
    if args.command == "verify":
        result = verify_view(args.package, expected_manifest_sha256=args.manifest_sha256)
    else:
        options = {"plan_path": args.plan, "plan_sha256": args.plan_sha256,
                   "generation_root": args.generation_root, "native_root": args.native_root,
                   "job_recipe_path": args.job_recipe, "job_recipe_sha256": args.job_recipe_sha256,
                   "stage_path": args.stage}
        if args.command == "source-map":
            result = build_source_map(args.inputs, args.inputs_sha256, args.archive, **options)
            weak.save_json(Path(args.output).absolute().parent, Path(args.output).name, result)
        else:
            result = create_view(args.inputs, args.inputs_sha256, args.archive, args.asr_root, args.output,
                fsmn_tokens=args.fsmn_tokens, selective_source_map=args.selective_source_map,
                selective_policy=args.selective_policy, selective_audit_sha256=args.selective_audit_sha256, **options)
    print(canonical({"schema": result["schema"], "statistics": result.get("statistics"), "new_model_calls": 0}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
