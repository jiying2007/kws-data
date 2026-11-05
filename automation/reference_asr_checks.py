#!/usr/bin/env python3
"""Bind a completed native ASR matrix to new reference content observations.

All three planned recognizers must have complete usable observations before a
reference contributes checks. Every available transcript remains in the private
review. Matching uses the existing full-sentence reference rule, never voting.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_weak_view as weak
import reference_asr_inputs as inputs_adapter
from kws_data_contract import reference_text

WHISPER_SMALL_SHA = "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"


def model_fields(model, assets, locks, whisper_lock, whisper_lock_raw):
    if model == "whisper":
        weak.require(assets == {"model": "openai/whisper-small", "sha256": WHISPER_SMALL_SHA,
                                "source_sha256": weak.sha(whisper_lock_raw)} and
                     whisper_lock["source_repository"] == "https://github.com/openai/whisper",
                     "REFERENCE_WHISPER_MODEL_IDENTITY")
        return {"model_id": assets["model"], "model_revision_kind": "native-checkpoint-sha256",
                "model_revision": WHISPER_SMALL_SHA,
                "implementation_source_commit": whisper_lock["source_revision"],
                "model_assets_sha256": weak.sha(weak.canonical(assets))}
    spec = locks[model]
    weak.require(assets == spec["asset_lock"] and weak.sha(weak.canonical(assets)) == spec["asset_lock_sha256"],
                 "REFERENCE_REPOSITORY_MODEL_IDENTITY")
    return {"model_id": assets["model_id"], "model_revision_kind": "model-repository-commit",
            "model_revision": assets["revision"], "implementation_source_commit": None,
            "model_assets_sha256": spec["asset_lock_sha256"]}


def verify_audit(asr_root, input_root, inputs_sha, count):
    audit_raw = weak.read_bytes(asr_root / "audit/final-summary.json")
    audit = weak.loads(audit_raw)
    raw = weak.read_bytes(asr_root / "audit/file-hashes.json", 8 * 1024 * 1024)
    inventory = weak.loads(raw)
    weak.require(audit["status"] == "complete" and audit["errors"] == [] and
                 audit["input_manifest_sha256"] == inputs_sha and
                 audit["verified_matrix_cells"] == audit["original_recognition_calls"] == count * 3 and
                 audit["duplicate_model_clip_pairs"] == 0 and
                 audit["target_context_audit"]["passed"] is True and
                 audit["target_context_audit"]["intended_text_and_peer_results_passed_to_native_recognizers"] is False and
                 audit["inventory_sha256"] == weak.sha(raw) and
                 inventory["files_sha256"] == weak.sha(weak.canonical(inventory["files"])),
                 "REFERENCE_ASR_AUDIT_INCOMPLETE")
    for name, record in inventory["files"].items():
        path = input_root / weak.relative(name[7:]) if name.startswith("inputs/") else asr_root / weak.relative(name)
        weak.require(not any(p.is_symlink() for p in (path, *path.parents)), "REFERENCE_ASR_EVIDENCE_ALIAS")
        info = path.stat()
        weak.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                     info.st_size == record["bytes"] <= 256 * 1024 * 1024,
                     "REFERENCE_ASR_EVIDENCE_SIZE")
        weak.require(weak.sha(path.read_bytes()) == record["sha256"], "REFERENCE_ASR_EVIDENCE_CHANGED")
    return audit, audit_raw


def export(input_root, source_manifest, expected_source_sha, asr_root, model_locks,
           whisper_source_lock, output):
    input_root, asr_root, output = Path(input_root), Path(asr_root), Path(output).absolute()
    inputs, bindings = inputs_adapter.verify(input_root)
    source_raw = weak.read_bytes(source_manifest)
    weak.require(weak.sha(source_raw) == expected_source_sha == bindings["source_manifest_sha256"],
                 "REFERENCE_SOURCE_MANIFEST_CHANGED")
    source = weak.loads(source_raw)
    refs = {row["reference_id"]: row for row in source["references"]}
    weak.require(len(refs) == len(source["references"]) == len(bindings["rows"]) and
                 set(refs) == {row["reference_id"] for row in bindings["rows"]}, "REFERENCE_CHECK_SOURCE_ROSTER")
    audit, audit_raw = verify_audit(asr_root, input_root, bindings["inputs_manifest_sha256"], len(refs))
    locks_raw = weak.read_bytes(model_locks, 8 * 1024 * 1024)
    locks = weak.loads(locks_raw)
    whisper_raw = weak.read_bytes(whisper_source_lock)
    whisper = weak.loads(whisper_raw)
    checks, reviews = [], []
    for clip, binding in zip(inputs["clips"], bindings["rows"]):
        ref = refs[binding["reference_id"]]
        reference_text.validate_reference(ref)
        weak.require(all(binding[field] == ref[field] for field in ("source_family_id", "speaker_group_id", "role")) and
                     binding["normalized_wav"]["sha256"] == ref["wav_sha256"] and
                     binding["normalized_pcm"]["sha256"] == ref["pcm_sha256"] and
                     binding["normalized_pcm"]["frames"] == ref["frames"] and
                     binding["normalized_pcm"]["sample_rate"] == ref["sample_rate_hz"] and
                     binding["original_wav_sha256"] == ref["parent_source_sha256"], "REFERENCE_CHECK_SOURCE_JOIN")
        observed = [weak.observation(asr_root, model, clip, bindings["inputs_manifest_sha256"], {})
                    for model in weak.MODELS]
        eligible = all(row["status"] == "complete" and row["usable"] is True for row in observed)
        observations = []
        if eligible:
            for row in observed:
                item = {"schema": reference_text.OBSERVATION_SCHEMA_V2, "engine": row["model"],
                        **model_fields(row["model"], row["model_identity"]["model_assets"], locks, whisper, whisper_raw),
                        "wav_sha256": ref["wav_sha256"], "pcm_sha256": ref["pcm_sha256"],
                        "sample_rate_hz": ref["sample_rate_hz"], "audio_frames": ref["frames"],
                        "actual_text": row["text"], "input_mode": "audio-only", "decoder_completed": True}
                reference_text.validate_observation(item)
                observations.append(item)
            checks.extend(observations)
        matched = reference_text._checked(ref["reference_text"], observations, ref)
        reviews.append({"clip_id": clip["id"], "reference_id": ref["reference_id"],
                        "source_family_id": ref["source_family_id"], "role": ref["role"],
                        "normalized_wav_sha256": ref["wav_sha256"], "publisher_text": ref["reference_text"],
                        "asr": observed, "all_three_usable": eligible,
                        "each_matches_publisher": [row.get("status") == "complete" and
                                                   reference_text.normalize(row["text"]) == reference_text.normalize(ref["reference_text"])
                                                   for row in observed],
                        "reference_text_status": reference_text.CHECKED if matched else reference_text.PROVIDED,
                        "prompt_mode": "icl" if matched else "x-vector-only", "human_gold": False})
    content = {"schema": reference_text.CHECK_SCHEMA_V2, "observations": checks}
    review = {"schema": "kws-reference-native-asr-content-review-v1",
              "source_manifest_sha256": expected_source_sha, "inputs_manifest_sha256": bindings["inputs_manifest_sha256"],
              "asr_audit_sha256": weak.sha(audit_raw), "decoder_source_sha256": audit["decoder_source_sha256"],
              "export_source_sha256": weak.sha(Path(__file__).read_bytes()),
              "content_checks_canonical_sha256": weak.sha(weak.canonical(content)),
              "model_lock_sha256": weak.sha(locks_raw), "whisper_source_lock_sha256": weak.sha(whisper_raw),
              "planned_recognizers_per_reference": 3, "actual_recognition_calls": audit["original_recognition_calls"],
              "new_model_calls_during_export": 0, "human_gold": False, "references": reviews,
              "counts": {"content_checked": sum(row["prompt_mode"] == "icl" for row in reviews),
                         "dataset_provided": sum(row["prompt_mode"] == "x-vector-only" for row in reviews),
                         "exported_observations": len(checks)}}
    weak.require(not output.exists() and output.parent.is_dir() and weak.ROOT not in output.parents and
                 not any(p.is_symlink() for p in (output, *output.parents)), "NEW_PRIVATE_REFERENCE_REVIEW_REQUIRED")
    output.mkdir(mode=0o700)
    for name, raw in (("source-manifest.dataset-provided.json", source_raw), ("model-locks.json", locks_raw),
                      ("whisper-source-lock.json", whisper_raw), ("export-source.py", Path(__file__).read_bytes()),
                      ("reference-text-source.py", Path(reference_text.__file__).read_bytes()),
                      ("asr-final-audit.json", audit_raw)):
        weak.save(output, name, raw)
    weak.save_json(output, "reference-content-checks.json", content)
    weak.save_json(output, "content-review.json", review)
    reference_text.content_checks(output / "reference-content-checks.json")
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "sources", "asr-results", "model-locks", "whisper-source-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sources-sha256", required=True)
    args = parser.parse_args(argv)
    review = export(args.input_root, args.sources, args.sources_sha256, args.asr_results,
                    args.model_locks, args.whisper_source_lock, args.output)
    print(weak.canonical(review["counts"]).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
