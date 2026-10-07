#!/usr/bin/env python3
"""Read-only verification of frozen evidence; Python standard library only.

The two source arguments point to research/melo6_tts directories from the
matching prior and recovery pipeline source snapshots. Source is read as data.
This script never imports it, constructs a model, or performs inference.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import zipfile


AUDIT_SHA = "076cee1e6d89228570a9c30e98d384a3957c04cb1eee53a6514f616883494a0c"
PRIOR_PLAN_SHA = "d3cd5e4a1c5c816e02c42b915b7930868ebe686ce4b8d57d9c721225af37cc21"
RECOVERY_PLAN_SHA = "a7a78e618152167c05dd8c65eeda546100db66634e08ce1c5cae63fd9467a5db"
ARCHIVE_PATH = "research/2026-10-05-melo6-source-screen"
MANIFEST_NAME = "content-manifest.json"
EXPERIMENT = "melo-native-six-phrase-pilot-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def safe_file(root, relative):
    part = PurePosixPath(relative)
    require(not part.is_absolute() and ".." not in part.parts and part.as_posix() == relative,
            "Unsafe or noncanonical declared relative path")
    require(part.parts and "\\" not in relative, "Invalid declared path")
    current = root
    for piece in part.parts:
        current = current / piece
        require(not current.is_symlink(), "Symlinks are not evidence files")
    require(current.is_file(), "Declared file missing")
    return current


def read_json(path):
    return json.loads(path.read_bytes())


def identity(path, relative, with_blob=False):
    raw = path.read_bytes()
    result = {"path": relative, "bytes": len(raw), "sha256": digest(raw)}
    if with_blob:
        result["git_blob_sha1"] = blob(raw)
    return result


def tree_files(root):
    paths = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Symlinks in evidence tree")
    return sorted(p.relative_to(root).as_posix() for p in paths if p.is_file())


def check_rows(root, rows, with_blob=False):
    require(len(rows) == len({row["path"] for row in rows}), "Duplicate declared file")
    for row in rows:
        require(identity(safe_file(root, row["path"]), row["path"], with_blob) == row,
                "Declared file identity mismatch: " + row["path"])


def check_generation_freeze(root, plan_sha, expected_count):
    freeze = read_json(safe_file(root, "generation-freeze.json"))
    require(freeze["schema"] == "melo-six-generation-freeze-v1", "Generation freeze schema")
    require(freeze["experiment_id"] == EXPERIMENT and freeze["plan_sha256"] == plan_sha,
            "Generation freeze plan mismatch")
    require(len(freeze["files"]) == expected_count, "Generation freeze file count")
    check_rows(root, freeze["files"])
    require(sorted(row["path"] for row in freeze["files"]) ==
            [p for p in tree_files(root) if p != "generation-freeze.json"],
            "Generation freeze does not cover exact directory")


def check_source(root, expected_plan, expected_runner):
    require(digest(safe_file(root, "execution-plan.json").read_bytes()) == expected_plan,
            "Source execution plan identity mismatch")
    plan = read_json(root / "execution-plan.json")
    require(plan["experiment_id"] == EXPERIMENT, "Source experiment mismatch")
    check_rows(root, plan["bindings"])
    freeze = read_json(safe_file(root, "adapter-freeze.json"))
    for name, expected in freeze["source_sha256"].items():
        require(digest(safe_file(root, name).read_bytes()) == expected,
                "Declared source identity mismatch: " + name)
    require(freeze["source_sha256"]["run_melo6.py"] == expected_runner, "Runner identity mismatch")
    return plan, freeze


def wav_chunks(raw):
    require(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "Not RIFF WAVE")
    require(struct.unpack("<I", raw[4:8])[0] == len(raw) - 8, "RIFF size mismatch")
    cursor, chunks = 12, []
    while cursor < len(raw):
        require(cursor + 8 <= len(raw), "Truncated WAV chunk header")
        name, size = struct.unpack("<4sI", raw[cursor:cursor + 8])
        require(cursor + 8 + size <= len(raw), "Truncated WAV chunk")
        chunks.append((name, raw[cursor + 8:cursor + 8 + size]))
        cursor += 8 + size + size % 2
    require(cursor == len(raw), "WAV trailing bytes")
    return chunks


def tensor_boundary(spec):
    code = {"int64": "q", "float32": "f"}[spec["dtype"]]
    def convert(value):
        if isinstance(value, list):
            return [convert(v) for v in value]
        return struct.unpack("<" + code, struct.pack("<" + code, value))[0]
    def flatten(value):
        if isinstance(value, list):
            return [n for child in value for n in flatten(child)]
        return [value]
    def shape(value):
        if not isinstance(value, list):
            return []
        children = [shape(v) for v in value]
        require(not children or all(s == children[0] for s in children), "Ragged tensor")
        return [len(value)] + (children[0] if children else [])
    values = convert(spec["values"])
    require(shape(values) == spec["shape"], "Tensor shape mismatch")
    flat = flatten(values)
    return {"dtype": spec["dtype"], "shape": spec["shape"], "values": values,
            "little_endian_bytes_sha256": digest(struct.pack("<" + code * len(flat), *flat))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--prior-source", type=Path, required=True)
    parser.add_argument("--recovery-source", type=Path, required=True)
    args = parser.parse_args()
    root = args.archive
    manifest_path = safe_file(root, MANIFEST_NAME)
    manifest = read_json(manifest_path)
    require(manifest["schema"] == "melo6-public-data-content-manifest-v1", "Manifest schema")
    require(manifest["repository"] == "jiying2007/kws-data", "Repository mismatch")
    require(manifest["archive_path"] == ARCHIVE_PATH, "Archive path mismatch")
    require(manifest["manifest_self_excluded"] is True, "Manifest self-exclusion contract")
    check_rows(root, manifest["files"], with_blob=True)
    require(sorted(row["path"] for row in manifest["files"]) ==
            [p for p in tree_files(root) if p != MANIFEST_NAME], "Archive file set mismatch")
    require(all(row["bytes"] <= 1024 * 1024 for row in manifest["files"]), "Oversize evidence file")
    require(manifest["research_status"] == {"asr": "pending", "ctc_label": None,
            "training_admission": False, "machine_evidence": "weak", "human_truth": "UNKNOWN"},
            "Initial research status mismatch")
    audit_path = safe_file(root, "independent-recovery-audit-result.json")
    require(digest(audit_path.read_bytes()) == AUDIT_SHA, "Pinned independent audit mismatch")
    audit = read_json(audit_path)
    require(audit["audit_status"] == "PASS_SIX_SAVED_GENERATION_AND_BLIND_HANDOFF", "Audit status")
    for directory, key, count in [("recovery", "recovery_evidence_allowlist", 32),
                                   ("prior-failure", "prior_failure_evidence_allowlist", 12)]:
        rows = audit["public_projection"][key]
        require(len(rows) == count, "Independent allowlist count")
        check_rows(root / directory, rows)
        require(sorted(row["path"] for row in rows) == tree_files(root / directory),
                "Evidence directory differs from independent allowlist")
    attempt_row = audit["public_projection"]["prelaunch_recovery_identity"]
    check_rows(root, [attempt_row])
    prior_attempt = read_json(root / "prior-failure/GENERATION-ATTEMPT.json")
    attempt = read_json(root / "GENERATION-ATTEMPT2.json")
    require(prior_attempt["plan_sha256"] == PRIOR_PLAN_SHA and
            attempt["plan_sha256"] == RECOVERY_PLAN_SHA == audit["plan_sha256"], "Attempt plan mismatch")
    old_plan, old_freeze = check_source(args.prior_source, PRIOR_PLAN_SHA, prior_attempt["source_sha256"])
    plan, freeze = check_source(args.recovery_source, RECOVERY_PLAN_SHA, attempt["source_sha256"])
    require(digest((args.recovery_source / "adapter-freeze.json").read_bytes()) ==
            audit["adapter_freeze_sha256"], "Recovery source freeze mismatch")
    require({k: v for k, v in old_plan.items() if k != "bindings"} ==
            {k: v for k, v in plan.items() if k != "bindings"}, "Scientific plan changed")
    for name in audit["unchanged_scientific_files_verified"]:
        require(safe_file(args.prior_source, name).read_bytes() ==
                safe_file(args.recovery_source, name).read_bytes(), "Scientific source changed: " + name)
    require(plan["model"]["sha256"] == audit["model_sha256"] == prior_attempt["model_sha256"],
            "Recorded model identity mismatch")
    old_ast = ast.parse((args.prior_source / "run_melo6.py").read_text())
    new_ast = ast.parse((args.recovery_source / "run_melo6.py").read_text())
    old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old_ast.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new_ast.body if isinstance(n, ast.FunctionDef)}
    for name in ["run_six", "input_arrays", "input_boundary", "bounded_write", "finalize", "main", "watchdog", "annotate_rss"]:
        require(old_functions[name] == new_functions[name], "Audited generation control flow changed")
    calls = [n for n in ast.walk(new_ast) if isinstance(n, ast.Call)]
    require(sum(isinstance(n.func, ast.Attribute) and n.func.attr == "InferenceSession" for n in calls) == 1,
            "Session constructor call site mismatch")
    require(sum(isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and
                n.func.value.id == "session" and n.func.attr == "run" for n in calls) == 1,
            "Inference call site mismatch")
    recovery = root / "recovery"
    generation = recovery / "generation"
    prior_generation = root / "prior-failure/tts/generation"
    check_generation_freeze(generation, RECOVERY_PLAN_SHA, 19)
    check_generation_freeze(prior_generation, PRIOR_PLAN_SHA, 1)
    prior_receipt = read_json(prior_generation / "generation-receipt.json")
    require(prior_receipt["plan_sha256"] == PRIOR_PLAN_SHA and prior_receipt["attempts_consumed"] == 0,
            "Prior attempt accounting mismatch")
    ids = ["melo6-%03d" % i for i in range(1, 7)]
    require(prior_receipt["generation_rows"] == [{"attempt_consumed": False, "source_id": sid,
            "status": "not_run"} for sid in ids], "Prior failed rows changed")
    stage = read_json(root / "prior-failure/tts/stage.json")
    require(stage["validation_code"] == "LOADED_OUTPUT_SIGNATURE" and stage["stage"] == "create_one_session",
            "Prior failure reason mismatch")
    cause = stage["private_cause"]
    require(set(cause) == {"bytes", "capture_locals", "public_artifact_allowed", "sha256", "status", "truncated"}
            and cause["capture_locals"] is False and cause["public_artifact_allowed"] is False
            and stage["raw_text_disclosed"] is False, "Unsafe cause retention fields")
    require(attempt["load_generation_attempt"] == 2 and attempt["new_session_max"] == 1
            and attempt["session_constructors_before"] == 1 and attempt["tts_calls_before"] == 0
            and attempt["tts_calls_this_attempt_max"] == attempt["cumulative_tts_max"] == 6,
            "Recovery prelaunch budget mismatch")
    receipt = read_json(generation / "generation-receipt.json")
    require(receipt["plan_sha256"] == RECOVERY_PLAN_SHA and receipt["attempts_consumed"] == 6
            and receipt["status"] == "six_candidates_generated" and len(receipt["generation_rows"]) == 6,
            "Recovery generation accounting mismatch")
    require(receipt["deterministic_waveform_claim"] is False, "Unexpected determinism claim")
    fixtures = read_json(args.recovery_source / "frozen-six-inputs.json")["fixtures"]
    require(len(fixtures) == 6 and [f["text"] for f in fixtures] == plan["phrase_order"], "Frozen phrase order")
    for sid, row, fixture, audio in zip(ids, receipt["generation_rows"], fixtures, audit["audio_checks"]):
        require(row["source_id"] == sid and row["status"] == "generated_candidate"
                and row["attempt_consumed"] is True and audio["source_id"] == sid, "Candidate row mismatch")
        claim = read_json(generation / (sid + ".started.json"))
        require(claim["source_id"] == sid and claim["attempt"] == 1 and claim["attempt_consumed"] is True
                and claim["plan_sha256"] == RECOVERY_PLAN_SHA and claim["requested_outputs"] == ["y"],
                "Started claim mismatch")
        require(claim["inputs"] == {key: tensor_boundary(value) for key, value in fixture["inputs"].items()},
                "Frozen input and actual claim mismatch")
        native_raw = (generation / (sid + ".native-f32.wav")).read_bytes()
        pcm_raw = (generation / (sid + ".pcm16.wav")).read_bytes()
        require(digest(native_raw) == row["native_wav_sha256"] == audio["native_wav_sha256"], "Native digest")
        require(digest(pcm_raw) == row["pcm_wav_sha256"] == audio["derived_wav_sha256"], "PCM digest")
        native = wav_chunks(native_raw)
        require([n for n, unused in native] == [b"fmt ", b"fact", b"data"], "Native chunks")
        require(native[0][1] == struct.pack("<HHIIHHH", 3, 1, 44100, 176400, 4, 32, 0), "Native WAV header")
        require(len(native[2][1]) % 4 == 0, "Native sample alignment")
        frames = len(native[2][1]) // 4
        require(struct.unpack("<I", native[1][1])[0] == frames == row["source_frames"] == audio["native_frames"],
                "Native frame count")
        require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", native[2][1])), "Nonfinite samples")
        require(digest(native[2][1]) == row["native_float_sha256"], "Native sample digest")
        pcm = wav_chunks(pcm_raw)
        require([n for n, unused in pcm] == [b"fmt ", b"data"], "PCM chunks")
        require(pcm[0][1] == struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16), "PCM WAV header")
        derivative_frames = (frames * 160 + 440) // 441
        require(len(pcm[1][1]) == derivative_frames * 2 and
                derivative_frames == row["conversion"]["derivative_frames"] == audio["derivative_frames"], "PCM frame count")
        require(digest(pcm[1][1]) == row["pcm_sha256"], "PCM sample digest")
        require(row["conversion"]["gain_normalization"] is False and row["conversion"]["trimmed"] is False
                and row["signal_flags"] == [] and 0 < row["call_and_conversion_seconds"] < 60, "Conversion receipt")
    runtime = read_json(recovery / "runtime-receipt.json")
    require(runtime["plan_sha256"] == RECOVERY_PLAN_SHA and runtime["session_count"] == runtime["set_seed_count"] == 1
            and runtime["versions"] == plan["runtime_versions"] and runtime["seed"] == 0
            and runtime["providers"] == ["CPUExecutionProvider"] and runtime["waveform_determinism_claim"] is False,
            "Runtime receipt mismatch")
    signature = read_json(recovery / "loaded-session-signature.json")
    require(signature["plan_sha256"] == RECOVERY_PLAN_SHA and signature["model_sha256"] == audit["model_sha256"]
            and signature["signature"] == audit["actual_loaded_signature"], "Loaded signature mismatch")
    require(read_json(recovery / "stage.json")["status"] == "complete" and
            read_json(recovery / "active-call.json") == {"inactive": True}, "Terminal recovery state")
    blind = recovery / "blind"
    blind_freeze = read_json(blind / "blind-input-freeze.json")
    export = read_json(blind / "export-receipt.json")
    require(set(blind_freeze) == {"files", "job_sha256", "schema"} and
            blind_freeze["schema"] == "qwen6-blind-input-freeze-v1", "Blind freeze schema")
    require(export["plan_sha256"] == RECOVERY_PLAN_SHA and export["clip_count"] == 6 and
            export["asr_run_performed"] is False, "Initial blind export status")
    require(digest((blind / "blind-inputs.zip").read_bytes()) == export["blind_archive_sha256"], "Blind archive digest")
    require(digest((blind / "blind-input-freeze.json").read_bytes()) == export["blind_freeze_sha256"], "Blind freeze digest")
    with zipfile.ZipFile(blind / "blind-inputs.zip") as archive:
        infos = archive.infolist()
        require(len(infos) == 7 and len({i.filename for i in infos}) == 7, "Blind member count")
        require(all(i.compress_type == zipfile.ZIP_STORED and not i.flag_bits & 1 for i in infos), "Blind ZIP mode")
        require(archive.testzip() is None, "Blind ZIP CRC")
        rows = [{"path": i.filename, "bytes": i.file_size, "sha256": digest(archive.read(i))} for i in infos]
        require(rows == blind_freeze["files"], "Blind member freeze mismatch")
        job_raw = archive.read("job.json")
        require(digest(job_raw) == blind_freeze["job_sha256"] == export["blind_job_sha256"], "Blind job digest")
        job = json.loads(job_raw)
        require(set(job) == {"schema", "clips"} and job["schema"] == "blind-asr-job-v1"
                and len(job["clips"]) == 6, "Blind job contract")
        for i, (clip, row) in enumerate(zip(job["clips"], receipt["generation_rows"]), 1):
            expected = {"audio_id": "clip-%06d" % i, "audio_path": "audio/" + row["pcm_wav_sha256"] + ".wav",
                        "wav_sha256": row["pcm_wav_sha256"]}
            require(clip == expected, "Blind clip contract")
            require(archive.read(clip["audio_path"]) == (generation / ("melo6-%03d.pcm16.wav" % i)).read_bytes(),
                    "Blind clip differs from generated derivative")
        require({i.filename for i in infos} == {"job.json"} | {c["audio_path"] for c in job["clips"]}, "Extra ZIP member")
    expected_handoff = [ARCHIVE_PATH + "/recovery/blind/blind-inputs.zip",
                        ARCHIVE_PATH + "/recovery/blind/blind-input-freeze.json"]
    require(manifest["asr_input_handoff_allowlist"] == expected_handoff, "ASR handoff paths")
    supervision = read_json(recovery / "supervision-receipt.json")
    require(supervision == audit["resources"]["supervision_receipt_preserved"], "Supervision receipt changed")
    require(supervision["returncode"] == 0 and supervision["termination_reason"] == "normal_exit", "Supervision failure")
    require(supervision["rss_observation_counts"]["unknown_samples"] == 0 and
            supervision["max_sampled"]["rss_bytes"] <= supervision["rss_threshold_bytes"], "RSS observations")
    result = {"status": "PASS", "archive_manifest_sha256": digest(manifest_path.read_bytes()),
              "archive_files_including_manifest": len(manifest["files"]) + 1,
              "recovery_evidence_files": 32, "prior_evidence_files": 12,
              "prior_source_bindings": len(old_freeze["source_sha256"]),
              "recovery_source_bindings": len(freeze["source_sha256"]),
              "both_generation_freezes_verified": True, "started_claims_verified": 6,
              "tensor_boundaries_verified": 42, "native_wav_headers_and_payloads_verified": 6,
              "derived_wav_headers_and_payloads_verified": 6, "blind_members_verified": 7,
              "blind_audio_bytes_match_derivatives": True, "plan_and_source_matching_verified": True,
              "generation_calls_recorded": 6, "session_constructions_recorded": 2,
              "calls_made_by_verifier": 0, "model_body_read_by_verifier": False,
              "resampling_recomputed_by_verifier": False,
              "resampling_evidence": "Pinned independent audit records full reconstruction as byte exact",
              "count_scope": "Two supplied attempt identities, control flow, saved claims and receipts; no historical census",
              "research_status": manifest["research_status"]}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, struct.error, zipfile.BadZipFile) as exc:
        detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(json.dumps({"status": "FAIL", "reason": detail}), file=sys.stderr)
        sys.exit(1)
