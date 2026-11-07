#!/usr/bin/env python3
"""Replay sealed natural-window ASR observations into an explicit weak-text view.

Standard library only. No recognizer, queue, audit sealing or training runs.
The view preserves whole-window text and parent intervals, not event times.
"""
from __future__ import annotations

import argparse
import array
from collections import Counter
import io
from pathlib import Path
import re
import struct
import sys
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automation import active_asr_weak_view as weak  # noqa: E402

REQUEST_SCHEMA = "kws-natural-asr-evidence-request-v1"
SCHEMA = "kws-natural-asr-evidence-view-v1"
MANIFEST_SCHEMA = "kws-natural-asr-evidence-manifest-v1"
POLICY = "natural-three-clean-fulltext-or-two-fulltext-with-unanimous-keywords-v1"
RATE = 16000
MAX_BATCHES, MAX_ROWS = 64, 1024
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
PRODUCER_FILES = (
    "automation/active_natural_asr_weak_view.py",
    "automation/active_asr_weak_view.py",
    "kws_data_contract/__init__.py",
    "kws_data_contract/identity.py",
    "kws_data_contract/reference_text.py",
    "kws_data_contract/source_registry.py",
)
BOUNDARY = {
    "human_gold": False, "training_admitted": False,
    "independent_test": False, "natural_far_available": False,
    "accepted_negative_seconds": 0, "acoustic_target_presence": "unknown",
    "acoustic_verbatim_completeness": "unknown", "event_times": None,
    "new_model_calls": 0, "new_training_runs": 0,
    "speaker_independence": "not_established",
}
INPUT_BOUNDARY = {
    "evidence_class": "unlabelled-natural-audio-windows",
    "target_presence": "unknown", "human_gold": False,
    "training_admitted": False, "natural_far_available": False,
    "source_independence": "not_established", "new_model_calls": 0,
    "duration_is_negative_exposure": False,
}
DERIVATION = {
    "source": "declared sample slices of mono PCM16 16000Hz parent WAVs",
    "operation": "signed-int16 / 32768 exactly represented as little-endian float32",
    "window_rule": "integer start-inclusive end-exclusive sample ranges",
    "window_wav": "new PCM16 container; exact parent sample bytes retained",
    "resampling": False, "clipping": False, "normalization": False,
    "padding": False, "acoustic_trimming": False,
    "rms_used_as_admission_gate": False,
    "output": "mono16000Hz little-endian float32 PCM",
}


def require(ok, reason):
    weak.require(ok, "NATURAL_VIEW_" + reason)


def is_sha(value):
    return type(value) is str and weak.SHA.fullmatch(value) is not None


def safe_dir(value):
    path = Path(value).absolute()
    require(path.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)),
            "REAL_DIRECTORY_REQUIRED")
    return path


def read(path, limit=32 * 1024 * 1024):
    return weak.read_bytes(path, limit)


def record(raw):
    return {"bytes": len(raw), "sha256": weak.sha(raw)}


def validate_request(value):
    require(type(value) is dict and set(value) == {"schema", "batches"} and
            value["schema"] == REQUEST_SCHEMA, "REQUEST_SCHEMA")
    require(type(value["batches"]) is list and 1 <= len(value["batches"]) <= MAX_BATCHES,
            "BOUNDED_BATCHES")
    ids = set()
    for row in value["batches"]:
        require(type(row) is dict and set(row) == {
            "batch_id", "bundle", "declaration_sha256", "bindings_sha256",
            "asr_output", "asr_request_sha256", "final_audit_sha256", "policy"},
            "BATCH_FIELDS")
        require(type(row["batch_id"]) is str and ID.fullmatch(row["batch_id"]) and
                row["batch_id"] not in ids, "BATCH_IDENTITY")
        ids.add(row["batch_id"])
        for name in ("declaration_sha256", "bindings_sha256", "asr_request_sha256",
                     "final_audit_sha256"):
            require(is_sha(row[name]), "EXTERNAL_PIN_REQUIRED")
        for name in ("bundle", "asr_output"):
            require(type(row[name]) is str and Path(row[name]).is_absolute(), "ABSOLUTE_INPUT_PATH")
        pin = row["policy"]
        require(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"} and
                type(pin["path"]) is str and Path(pin["path"]).is_absolute() and
                type(pin["bytes"]) is int and 0 < pin["bytes"] <= 65536 and
                pin["sha256"] == weak.SELECTIVE_POLICY_SHA, "POLICY_PIN")


def interval_union(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def validate_declaration(value):
    require(type(value) is dict and set(value) == {
        "schema", "status", "selection_reason", "parents", "windows"} and
        value["schema"] == "kws-natural-asr-window-declaration-v1" and
        value["status"] == "declared_not_materialized" and
        type(value["selection_reason"]) is str and 1 <= len(value["selection_reason"]) <= 1024,
        "DECLARATION")
    require(type(value["parents"]) is list and 1 <= len(value["parents"]) <= 8 and
            type(value["windows"]) is list and 1 <= len(value["windows"]) <= 64,
            "DECLARATION_ROSTER")
    parents, pcms, sessions, paths, families = {}, set(), set(), set(), {}
    for parent in value["parents"]:
        require(type(parent) is dict and set(parent) == {
            "parent_id", "audio_path", "wav_sha256", "pcm_sha256", "sample_rate_hz",
            "sample_count", "dataset_id", "source_partition", "session_id",
            "source_family_id", "split", "provenance"}, "PARENT_FIELDS")
        for name in ("parent_id", "dataset_id", "source_partition", "session_id", "source_family_id"):
            require(type(parent[name]) is str and ID.fullmatch(parent[name]), "PARENT_IDENTITY")
        weak.relative(parent["audio_path"])
        require(is_sha(parent["wav_sha256"]) and is_sha(parent["pcm_sha256"]) and
                type(parent["sample_rate_hz"]) is int and parent["sample_rate_hz"] == RATE and
                type(parent["sample_count"]) is int and 0 < parent["sample_count"] <= (256 * 1024**2 - 44) // 2,
                "PARENT_GEOMETRY_OR_HASH")
        require(parent["split"] in ("train", "development") and
                parent["source_partition"].casefold() not in ("test", "heldout", "holdout"),
                "NO_TEST_OR_HELDOUT")
        if parent["source_partition"].casefold() in ("eval", "evaluation", "dev", "development", "validation"):
            require(parent["split"] == "development", "DEVELOPMENT_PARTITION_NOT_TRAIN")
        session = (parent["dataset_id"], parent["session_id"])
        require(parent["parent_id"] not in parents and parent["pcm_sha256"] not in pcms and
                session not in sessions and parent["audio_path"] not in paths, "PARENT_ALIAS")
        require(families.setdefault(parent["source_family_id"], parent["split"]) == parent["split"],
                "FAMILY_SPLIT_OVERLAP")
        provenance = parent["provenance"]
        require(type(provenance) is dict and set(provenance) == {
            "source_archive_sha256", "source_member", "source_wav_sha256",
            "channel_index", "source_audit_sha256", "annotation_sha256"}, "PARENT_PROVENANCE")
        for key in ("source_archive_sha256", "source_wav_sha256", "source_audit_sha256", "annotation_sha256"):
            require(provenance[key] is None or is_sha(provenance[key]), "PROVENANCE_HASH")
        if provenance["source_member"] is not None:
            weak.relative(provenance["source_member"])
        require(provenance["channel_index"] is None or (
            type(provenance["channel_index"]) is int and 0 <= provenance["channel_index"] < 64),
            "PROVENANCE_CHANNEL")
        parents[parent["parent_id"]] = parent
        pcms.add(parent["pcm_sha256"]); sessions.add(session); paths.add(parent["audio_path"])
    ids, ranges, used = set(), set(), set()
    for window in value["windows"]:
        require(type(window) is dict and set(window) == {
            "window_id", "parent_id", "start_sample", "end_sample"}, "WINDOW_FIELDS")
        require(type(window["window_id"]) is str and ID.fullmatch(window["window_id"]) and
                window["window_id"] not in ids and window["parent_id"] in parents, "WINDOW_IDENTITY")
        start, end = window["start_sample"], window["end_sample"]
        require(type(start) is int and type(end) is int and
                0 <= start < end <= parents[window["parent_id"]]["sample_count"] and
                end - start <= RATE * 30, "WINDOW_BOUNDS")
        key = (window["parent_id"], start, end)
        require(key not in ranges, "DUPLICATE_WINDOW")
        ids.add(window["window_id"]); ranges.add(key); used.add(window["parent_id"])
    require(used == set(parents), "UNUSED_PARENT")
    return parents


def declared_coverage(value):
    rows = []
    for parent in value["parents"]:
        spans = [(w["start_sample"], w["end_sample"]) for w in value["windows"]
                 if w["parent_id"] == parent["parent_id"]]
        union = interval_union(spans)
        processed, unique = sum(b-a for a, b in spans), sum(b-a for a, b in union)
        rows.append({key: parent[key] for key in (
            "parent_id", "dataset_id", "session_id", "source_family_id", "split")} | {
            "union_sample_ranges": union, "window_input_samples": processed,
            "unique_parent_samples": unique, "overlap_samples": processed-unique})
    processed = sum(r["window_input_samples"] for r in rows)
    unique = sum(r["unique_parent_samples"] for r in rows)
    return {"sample_rate_hz": RATE, "window_input_samples": processed,
            "unique_parent_samples": unique, "overlap_samples": processed-unique,
            "window_input_seconds": processed / RATE, "unique_parent_seconds": unique / RATE,
            "overlap_seconds": (processed-unique) / RATE, "by_parent": rows,
            "semantics": "union within each parent PCM; not independent or verified negative exposure"}


def pcm16_window(raw):
    require(len(raw) >= 46 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and
            struct.unpack_from("<I", raw, 4)[0] + 8 == len(raw), "WINDOW_RIFF")
    with wave.open(io.BytesIO(raw), "rb") as reader:
        require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(),
                 reader.getcomptype()) == (1, 2, RATE, "NONE"), "WINDOW_PCM16_REQUIRED")
        pcm = reader.readframes(reader.getnframes())
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(RATE)
        writer.writeframes(pcm)
    require(output.getvalue() == raw and 0 < len(pcm) <= RATE * 30 * 2, "EXACT_WINDOW_CONTAINER")
    integers = array.array("h"); integers.frombytes(pcm)
    if sys.byteorder != "little":
        integers.byteswap()
    floats = array.array("f", (value / 32768 for value in integers))
    if sys.byteorder != "little":
        floats.byteswap()
    return pcm, floats.tobytes()


def verify_bundle(bundle, entry):
    """Verify declared window bytes/ancestry, without re-reading full parents.

    The full-parent check belongs to the pinned upstream preparation. Its exact
    declaration and preparation-source identities are retained, not re-certified.
    """
    raw = read(bundle / "declaration.json", 1024**2)
    require(weak.sha(raw) == entry["declaration_sha256"], "DECLARATION_PIN")
    declaration = weak.loads(raw)
    parents = validate_declaration(declaration)
    binding_raw = read(bundle / "natural-window-bindings.json")
    require(weak.sha(binding_raw) == entry["bindings_sha256"], "BINDINGS_PIN")
    binding = weak.loads(binding_raw)
    inputs_raw, mapping_raw = read(bundle / "inputs.json"), read(bundle / "source-map.json")
    inputs, mapping = weak.loads(inputs_raw), weak.loads(mapping_raw)
    archive_raw = read(bundle / "window-audio.zip", 64 * 1024**2)
    require(type(inputs) is dict and set(inputs) == {
        "schema", "source_archive_sha256", "created_at", "preparation_source_sha256", "derivation", "clips"} and
        inputs["schema"] == "active-asr-inputs-v1" and inputs["derivation"] == DERIVATION and
        type(inputs["created_at"]) is str and
        inputs["source_archive_sha256"] == weak.sha(archive_raw) and
        len(inputs["clips"]) == len(declaration["windows"]), "NATURAL_INPUTS")
    prep_raw = read(bundle / "preparation-source.py")
    require(inputs["preparation_source_sha256"] == weak.sha(prep_raw), "PREPARATION_SOURCE")
    expected_rows, map_rows, metadata = [], [], []
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        expected_members = [f"original/clip-{i:03d}.wav" for i in range(len(inputs["clips"]))]
        require(archive.namelist() == expected_members, "WINDOW_ARCHIVE_ROSTER")
        for index, (clip, window) in enumerate(zip(inputs["clips"], declaration["windows"])):
            parent, clip_id = parents[window["parent_id"]], f"clip-{index:03d}"
            frames = window["end_sample"] - window["start_sample"]
            original, derived = f"original/{clip_id}.wav", f"pcm16k/{clip_id}.f32"
            wav = read(bundle / original, 2 * 1024**2)
            pcm, floating = pcm16_window(wav)
            require(len(pcm) == frames * 2 and read(bundle / derived, 2 * 1024**2) == floating and
                    archive.read(original) == wav, "WINDOW_AUDIO_BINDING")
            expected_clip = {"id": clip_id, "original_member": original,
                "original_wav": {"path": original, **record(wav)},
                "original_pcm": {"sha256": weak.sha(pcm), "sample_rate": RATE, "channels": 1,
                                 "frames": frames, "dtype": "<i2"},
                "derived_pcm": {"path": derived, "sha256": weak.sha(floating), "sample_rate": RATE,
                                "channels": 1, "frames": frames, "dtype": "<f4"}}
            require(clip == expected_clip, "INPUT_CLIP_IDENTITY")
            # The upstream source ID hashes canonical JSON with one trailing LF.
            source_id = "natural-window-" + weak.sha(weak.canonical({
                "parent_pcm_sha256": parent["pcm_sha256"], "sample_rate_hz": RATE,
                "start_sample": window["start_sample"], "end_sample": window["end_sample"]}) + b"\n")
            map_rows.append({"clip_id": clip_id, "source_id": source_id,
                "original_wav_sha256": weak.sha(wav), "original_pcm_sha256": weak.sha(pcm),
                "derived_pcm_sha256": weak.sha(floating), "quality_uncertain": True,
                "quality_reasons": ["unreviewed-natural-audio-window"]})
            expected_rows.append({"clip_id": clip_id, "source_id": source_id, **window,
                **{key: parent[key] for key in ("dataset_id", "session_id", "source_family_id", "split")},
                "parent_wav_sha256": parent["wav_sha256"], "parent_pcm_sha256": parent["pcm_sha256"],
                "window_wav_sha256": weak.sha(wav), "window_pcm_sha256": weak.sha(pcm),
                "derived_pcm_sha256": weak.sha(floating)})
            metadata.append({**expected_rows[-1], "sample_rate_hz": RATE,
                "parent_sample_count": parent["sample_count"], "source_partition": parent["source_partition"],
                "parent_provenance": parent["provenance"]})
    require(mapping == {"schema": "active-asr-source-map-v2",
        "input_manifest_sha256": weak.sha(inputs_raw), "source_archive_sha256": weak.sha(archive_raw),
        "plan_sha256": entry["declaration_sha256"], "rows": map_rows}, "NATURAL_SOURCE_MAP")
    sources = binding.get("preparation_sources")
    require(type(sources) is dict and sources.get("tools/prepare_natural_asr_windows.py") == record(prep_raw),
            "PREPARATION_SOURCE_BINDING")
    for name, pin in sources.items():
        weak.relative(name)
        require(type(pin) is dict and set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int and
                pin["bytes"] >= 0 and is_sha(pin["sha256"]), "PREPARATION_SOURCE_PIN")
    require(binding == {
        "schema": "kws-natural-asr-window-bindings-v1", "declaration_sha256": entry["declaration_sha256"],
        "inputs_manifest_sha256": weak.sha(inputs_raw), "source_map_sha256": weak.sha(mapping_raw),
        "window_archive_sha256": weak.sha(archive_raw), "preparation_sources": sources,
        "not_decoder_input": True, "target_text_included": False,
        "parent_identity_scope": "full parent WAV and decoded PCM checked against declaration",
        "upstream_provenance_scope": "retained declaration only; upstream archives and annotations not read",
        "parents": declaration["parents"], "rows": expected_rows,
        "coverage": declared_coverage(declaration), "boundary": INPUT_BOUNDARY}, "WINDOW_BINDINGS")
    summary = weak.loads(read(bundle / "summary.json"))
    require(summary == {"schema": "kws-natural-asr-window-summary-v1",
        "status": "materialized_unlabelled_inputs", "declaration_sha256": entry["declaration_sha256"],
        "parents": len(parents), "windows": len(inputs["clips"]), "coverage": declared_coverage(declaration),
        "bindings_sha256": weak.sha(binding_raw), "boundary": INPUT_BOUNDARY}, "WINDOW_SUMMARY")
    return inputs_raw, mapping_raw, metadata


def keyword_evidence(observations):
    rows = []
    for keyword in weak.KEYWORDS:
        items = []
        for observed in observations:
            text = weak.normalize(observed["text"])
            literal = keyword in text
            ordered = weak.contains_ordered(text, keyword)
            items.append({"model": observed["model"], "usable": observed["usable"],
                "literal_present": literal, "ordered_path_present": ordered,
                "ordered_nonliteral_path": ordered and not literal})
        rows.append({"keyword": keyword, "observations": items,
            "three_usable_literal_presence_agree": all(x["usable"] for x in items) and
                len({x["literal_present"] for x in items}) == 1,
            "any_ordered_nonliteral_path": any(x["ordered_nonliteral_path"] for x in items)})
    return rows


def decide(observations):
    """Apply the frozen natural full-text quorum; never vote on absence alone."""
    require(len(observations) == 3 and [r["model"] for r in observations] == list(weak.MODELS),
            "THREE_MODEL_ROSTER")
    evidence = keyword_evidence(observations)
    decision = {"status": "uncertain", "reason": "three_clean_nonempty_observations_required",
        "full_text_quorum": False, "full_text": None, "selected_raw_text": None, "agreeing_models": [],
        "ctc_mask": False, "ctc_weight": 0.0, "keyword_targets": [None, None],
        "keyword_mask": [False, False], "keyword_weight": [0.0, 0.0],
        "label_tier": "machine-weak-uncertain", "human_gold": False,
        "acoustic_verbatim_completeness": "unknown", "event_times": None}
    if not all(r.get("status") == "complete" and r.get("usable") is True and
               r.get("decoder_terminated") is True and r.get("quality_flags") == [] and
               weak.normalize(r["text"]) for r in observations):
        return decision, evidence
    normalized = [weak.normalize(r["text"]) for r in observations]
    votes = Counter(normalized)
    text, count = votes.most_common(1)[0]
    if count == 3:
        weight, reason = 1.0, "three_clean_full_transcripts_agree"
    elif count == 2:
        if not all(row["three_usable_literal_presence_agree"] for row in evidence):
            decision["reason"] = "two_full_transcripts_but_keyword_presence_disagrees"
            return decision, evidence
        if any(row["any_ordered_nonliteral_path"] for row in evidence):
            decision["reason"] = "two_full_transcripts_with_ordered_nonliteral_keyword_ambiguity"
            return decision, evidence
        weight, reason = 0.5, "two_clean_full_transcripts_and_three_keyword_presence_vectors_agree"
    else:
        decision["reason"] = "no_full_transcript_quorum"
        return decision, evidence
    supporters = [r for r in observations if weak.normalize(r["text"]) == text]
    decision.update(status="machine_candidate", reason=reason, full_text_quorum=True, full_text=text,
        selected_raw_text=supporters[0]["text"], agreeing_models=[r["model"] for r in supporters],
        ctc_mask=True, ctc_weight=weight, keyword_targets=[int(k in text) for k in weak.KEYWORDS],
        keyword_mask=[True, True], keyword_weight=[weight, weight],
        label_tier="machine-weak-three" if count == 3 else "machine-weak-two")
    return decision, evidence


def batch_view(entry, bundle, asr_root, policy_path):
    bundle, asr_root = safe_dir(bundle), safe_dir(asr_root)
    inputs_raw, mapping_raw, metadata = verify_bundle(bundle, entry)
    require(weak.sha(read(asr_root / "request.json")) == entry["asr_request_sha256"],
            "ASR_RECIPE_PIN")
    # read_bytes fails if the seal does not exist. No audit/worker callback exists here.
    require(weak.sha(read(asr_root / "audit/final-summary.json")) == entry["final_audit_sha256"],
            "EXISTING_FINAL_SEAL_REQUIRED")
    policy_raw = read(policy_path, 65536)
    require(record(policy_raw) == {key: entry["policy"][key] for key in ("bytes", "sha256")},
            "POLICY_CHANGED")
    retained = {}
    observed, adapter = weak.replay_selective_evidence(asr_root, inputs_raw, bundle,
        entry["declaration_sha256"], mapping_raw, policy_raw, entry["final_audit_sha256"], retained)
    rows = []
    for source in metadata:
        observations = observed[source["clip_id"]]
        require([r["model"] for r in observations] == list(weak.MODELS) and
                all(r["status"] == "complete" for r in observations), "ALL_NATURAL_REQUESTS_REQUIRED")
        decision, keywords = decide(observations)
        rows.append({**source, "batch_id": entry["batch_id"],
            "observations": [{**r, "requested": True} for r in observations],
            "keyword_evidence": keywords, "decision": decision})
    batch = {"batch_id": entry["batch_id"], "status": "complete",
        "declaration_sha256": entry["declaration_sha256"], "bindings_sha256": entry["bindings_sha256"],
        "input_manifest_sha256": weak.sha(inputs_raw), "asr_request_sha256": adapter["request_sha256"],
        "final_audit_sha256": adapter["final_audit_sha256"], "inventory_sha256": adapter["inventory_sha256"],
        "requested_native_results": len(rows) * 3, "windows": len(rows)}
    return batch, rows, retained


def summarize(request_raw, batches, rows, producer):
    require(1 <= len(rows) <= MAX_ROWS, "TOTAL_WINDOW_BOUND")
    ids, parents, sessions, roles = set(), {}, {}, weak.SourceRoles()
    for row in rows:
        require(row["source_id"] not in ids, "DUPLICATE_GLOBAL_SOURCE")
        ids.add(row["source_id"])
        parent_identity = {key: row[key] for key in ("dataset_id", "source_partition", "session_id",
            "source_family_id", "split", "parent_wav_sha256", "parent_sample_count", "parent_provenance")}
        require(parents.setdefault(row["parent_pcm_sha256"], parent_identity) == parent_identity,
                "CROSS_BATCH_PARENT_IDENTITY")
        session = (row["dataset_id"], row["session_id"])
        require(sessions.setdefault(session, row["parent_pcm_sha256"]) == row["parent_pcm_sha256"],
                "CROSS_BATCH_SESSION_CHANNEL_ALIAS")
        role = "train-candidate" if row["split"] == "train" else "development-only"
        roles.add(role, family=row["source_family_id"], pcm_sha256=row["parent_pcm_sha256"],
                  wav_sha256=row["parent_wav_sha256"])
        roles.add(role, pcm_sha256=row["window_pcm_sha256"], wav_sha256=row["window_wav_sha256"])
    counts = Counter(r["decision"]["label_tier"] for r in rows)
    return {"schema": SCHEMA, "policy": POLICY, "request_sha256": weak.sha(request_raw),
        "producer": {"source_files": producer}, "batches": batches, "rows": rows,
        "statistics": {"windows": len(rows), "parents": len(parents),
            "verified_native_observations": len(rows) * 3,
            "usable_native_observations": sum(o["usable"] for r in rows for o in r["observations"]),
            "machine_weak_three": counts["machine-weak-three"], "machine_weak_two": counts["machine-weak-two"],
            "uncertain": counts["machine-weak-uncertain"],
            "training_role_windows": sum(r["split"] == "train" for r in rows),
            "development_role_windows": sum(r["split"] == "development" for r in rows)},
        "source_verification_scope": "declared window bytes, PCM derivation and sealed native evidence replayed; full parents and upstream archives are not reloaded",
        "boundary": BOUNDARY}


def producer_files():
    return {name: record(read(ROOT / name)) for name in PRODUCER_FILES}


def create_view(request_path, expected_request_sha256, output):
    request_raw = read(request_path, 1024**2)
    require(is_sha(expected_request_sha256) and weak.sha(request_raw) == expected_request_sha256, "REQUEST_PIN")
    request = weak.loads(request_raw); validate_request(request)
    output = Path(output).absolute()
    require(not output.exists() and output.parent.is_dir() and ROOT not in output.parents and
            not any(p.is_symlink() for p in (output, *output.parents)), "NEW_PRIVATE_OUTPUT")
    # Check externally pinned seals/recipes exist for every batch before creating output.
    for entry in request["batches"]:
        root = safe_dir(entry["asr_output"])
        require(weak.sha(read(root / "audit/final-summary.json")) == entry["final_audit_sha256"] and
                weak.sha(read(root / "request.json")) == entry["asr_request_sha256"], "EXISTING_SEAL_AND_RECIPE")
    output.mkdir(mode=0o700)
    weak.save(output, "request.json", request_raw)
    producer = producer_files()
    for name in PRODUCER_FILES:
        weak.save(output, "evidence/view-code/" + name, read(ROOT / name))
    batches, rows = [], []
    for entry in request["batches"]:
        batch, current, retained = batch_view(entry, entry["bundle"], entry["asr_output"], entry["policy"]["path"])
        batches.append(batch); rows.extend(current)
        for name, raw in retained.items():
            weak.save(output, "batches/" + entry["batch_id"] + "/" + name, raw)
    report = summarize(request_raw, batches, rows, producer)
    weak.save_json(output, "report.json", report)
    weak.save_rows(output, "rows.jsonl", rows)
    files = {p.relative_to(output).as_posix(): record(p.read_bytes())
             for p in sorted(output.rglob("*")) if p.is_file()}
    weak.save_json(output, "MANIFEST.json", {"schema": MANIFEST_SCHEMA, "files": files,
        "file_count": len(files), "files_sha256": weak.sha(weak.canonical(files))})
    return report


def verify_view(package, *, expected_manifest_sha256):
    """Read-only exact replay. It never invokes upstream audit sealing."""
    package = safe_dir(package)
    manifest_raw = read(package / "MANIFEST.json", 8 * 1024**2)
    require(is_sha(expected_manifest_sha256) and weak.sha(manifest_raw) == expected_manifest_sha256,
            "MANIFEST_PIN")
    manifest = weak.loads(manifest_raw)
    require(set(manifest) == {"schema", "files", "file_count", "files_sha256"} and
            manifest["schema"] == MANIFEST_SCHEMA and type(manifest["files"]) is dict and
            0 < manifest["file_count"] == len(manifest["files"]) <= 65536 and
            manifest["files_sha256"] == weak.sha(weak.canonical(manifest["files"])), "MANIFEST")
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    require(actual == set(manifest["files"]) | {"MANIFEST.json"}, "PACKAGE_FILE_ROSTER")
    for name, expected in manifest["files"].items():
        weak.relative(name)
        require(type(expected) is dict and set(expected) == {"bytes", "sha256"}, "FILE_PIN")
        body = weak.empty_evidence(package / name) if expected["bytes"] == 0 else read(package / name, 64 * 1024**2)
        require(record(body) == expected, "RETAINED_FILE_CHANGED")
    original = weak.loads(read(package / "report.json", 32 * 1024**2))
    require(original["schema"] == SCHEMA and original["producer"]["source_files"] == producer_files(),
            "VIEW_CODE_CHANGED_USE_RETAINED_SOURCE")
    for name, pin in original["producer"]["source_files"].items():
        require(record(read(package / "evidence/view-code" / name)) == pin, "RETAINED_VIEW_CODE_CHANGED")
    request_raw = read(package / "request.json", 1024**2)
    require(weak.sha(request_raw) == original["request_sha256"], "RETAINED_REQUEST_CHANGED")
    request = weak.loads(request_raw); validate_request(request)
    batches, rows = [], []
    for entry in request["batches"]:
        root = package / "batches" / entry["batch_id"] / "evidence"
        batch, current, _retained = batch_view(entry, root / "inputs", root / "asr", root / "selective/policy.json")
        batches.append(batch); rows.extend(current)
    rebuilt = summarize(request_raw, batches, rows, original["producer"]["source_files"])
    require(original == rebuilt and read(package / "rows.jsonl", 32 * 1024**2) ==
            b"".join(weak.canonical(row) + b"\n" for row in rows), "REPORT_DIFFERS_FROM_NATIVE_EVIDENCE")
    return rebuilt


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--request", type=Path, required=True)
    create.add_argument("--request-sha256", required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--manifest-sha256", required=True)
    args = cli.parse_args(argv)
    try:
        report = (create_view(args.request, args.request_sha256, args.output) if args.command == "create"
                  else verify_view(args.package, expected_manifest_sha256=args.manifest_sha256))
        print(weak.canonical({"schema": SCHEMA, "status": "verified" if args.command == "verify" else "created",
            "statistics": report["statistics"], "boundary": report["boundary"]}).decode())
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, wave.Error) as error:
        print("NATURAL_ASR_VIEW_REJECTED: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
