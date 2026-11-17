#!/usr/bin/env python3
"""Audit one fixed saved acquisition failure; no codec/model/subprocess/network.

The original artifact is intentionally not repaired. Its one absent, pinned
source/.github workflow is reported as a historical upload omission, so a
zero-compute conclusion must never be mistaken for a closed-manifest pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


ARTIFACT = {
    "artifact_id": 14317102541,
    "run_id": "44057968882",
    "head_sha": "d4edc97e4efb5612d16018f5cdeb42002d9eedfa",
    "bytes": 4257652,
    "sha256": "514b70f69470620612d6ea3dfa84a53369468d0f86a659863a4301c108266e12",
}
MANIFEST_SHA = "818e160774e8885ca4efa36bdaff13971e819079b66190781aae7c2fe3cd7a63"
OLD_WORKER_SHA = "5bab9b90862671e7532a20e0d6a0f70dd3bd19fd4e954268476285b6366885c8"
OMITTED = "source/.github/workflows/english-hardneg-natural32-once.yml"
WORKFLOW_PIN = {"bytes": 4423, "sha256": "2709d3cb7197624ca35e270974cd43bee2132d931d28dc109dda00d98b084d88"}
FALSE_FLAGS = ("human_gold", "trusted_negative", "training_admitted", "formal_training_admitted",
               "pronunciation_verified", "model_promotion_allowed", "qualification_allowed")
RUN_BINDING = {
    "repository": "jiying2007/kws-data", "run_id": ARTIFACT["run_id"],
    "head_sha": ARTIFACT["head_sha"], "run_attempt": 1, "event": "push",
    "main_before": "61e524e89c86950ef8e284ecaf18fddf82d11ca1", "public_repository": True,
}
SOURCE_IDS = {
    "training": (43867764366, 14225893574),
    "quality": (44042856615, 14310138100),
    "sdk": (43984579132, 14285590605),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def audit(path, old_workflow):
    raw = path.read_bytes()
    need(identity(raw) == {k: ARTIFACT[k] for k in ("bytes", "sha256")}, "fixed failure ZIP identity")
    files = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        need(len(infos) == 85 and sum(x.file_size for x in infos) < 32 * 1024**2, "fixed ZIP member/size budget")
        for item in infos:
            name = item.filename
            need(not item.is_dir() and name not in files and not PurePosixPath(name).is_absolute()
                 and "\\" not in name and "\0" not in name and
                 all(part not in ("", ".", "..") for part in name.split("/")), "unique regular member path")
            need(not item.flag_bits & 1 and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG),
                 "unencrypted nonsymlink regular member")
            files[name] = archive.read(item)
    manifest_raw = files["MANIFEST.json"]
    need(identity(manifest_raw)["sha256"] == MANIFEST_SHA, "fixed historical manifest identity")
    manifest = load(manifest_raw)
    declared = manifest["files"]
    actual = set(files) - {"MANIFEST.json"}
    missing, extra = set(declared) - actual, actual - set(declared)
    need(len(declared) == 85 and len(actual) == 84 and missing == {OMITTED} and not extra,
         "only exact fixed historical hidden-workflow omission is understood")
    need(declared[OMITTED] == WORKFLOW_PIN, "omitted workflow declared pin")
    for name in sorted(actual):
        need(identity(files[name]) == declared[name], "retained member bytes/hash: " + name)
    frozen_workflow = old_workflow.read_bytes()
    need(identity(frozen_workflow) == WORKFLOW_PIN, "separate frozen old workflow corroboration")
    need(manifest["supervised_success"] is False and manifest["downloads_excluded"] is True and
         manifest["authoritative_completion_receipt"] == "evidence/SUPERVISOR_EXIT.json", "historical failure authority")
    need(all(manifest[k] is False for k in FALSE_FLAGS), "manifest admission boundary")
    worker = files["source/automation/english_hardneg_natural32_once.py"]
    need(identity(worker)["sha256"] == OLD_WORKER_SHA, "historical worker source")
    plan = load(files["source/automation/english_hardneg_natural32_plan.json"])
    need(plan["worker_sha256"] == OLD_WORKER_SHA and plan["maximum_execution_seconds"] == 1200 and
         plan["maximum_model_processes"] == plan["maximum_decoder_processes"] == 32, "original fixed scope")
    for name, pin in plan["source_pins"].items():
        need(identity(files["source/" + name]) == pin, "original frozen helper pin: " + name)

    started = load(files["STARTED.json"])
    result = load(files["evidence/RESULT.json"])
    supervisor = load(files["evidence/SUPERVISOR_EXIT.json"])
    for obj in (started["binding"], result, supervisor):
        need(all(type(obj[k]) is type(v) and obj[k] == v for k, v in RUN_BINDING.items()), "actual run/head binding")
    need(started["worker_sha256"] == OLD_WORKER_SHA and started["maximum_seconds"] == 1200 and
         started["monotonic_deadline"] - started["monotonic_started"] == 1200, "supervisor deadline")
    need(result["passed"] is False and result["status"] == "FAILED_NO_RETRY" and result["reports"] == [], "failed result without trace reports")
    for key in ("consumer_build_attempts", "decoder_process_attempts", "model_process_attempts", "completed_validated_traces",
                "new_onnx_calls", "python_head_forwards", "new_tts_asr_phone_calls", "optimizer_updates", "verified_negative_count",
                "verified_negative_seconds", "new_independent_speaker_count", "retry_count"):
        need(type(result[key]) is int and result[key] == 0, "zero counter: " + key)
    need(all(result[k] is False for k in FALSE_FLAGS) and result["quality_improvement_claim"] is False and
         result["independent_far"] == result["independent_frr"] == "NOT_MEASURED", "no inferred quality/labels")
    need(supervisor["passed"] is False and supervisor["returncode"] == 1 and supervisor["timed_out"] is False
         and supervisor["error"] is None and supervisor["partial_receipt_errors"] == [] and supervisor["retry_count"] == 0,
         "supervisor observed normal worker failure")
    need(supervisor["cleanup"] == {"descendants_found": False, "terminated": False, "killed": False} and
         supervisor["wall_seconds"] == 307.016853929, "no descendants/exact supervisor wall time")

    commands = {"current-run": ["gh", "api", "repos/jiying2007/kws-data/actions/runs/44057968882"]}
    for name, (run_id, artifact_id) in SOURCE_IDS.items():
        commands[name + "-run"] = ["gh", "api", "repos/jiying2007/kws-data/actions/runs/" + str(run_id)]
        commands[name + "-artifact"] = ["gh", "api", "repos/jiying2007/kws-data/actions/artifacts/" + str(artifact_id)]
        commands[name + "-zip"] = ["gh", "api", "repos/jiying2007/kws-data/actions/artifacts/" + str(artifact_id) + "/zip"]
    commands["corpus-download"] = ["curl", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
                                   "--connect-timeout", "20", "--max-time", "300", "--retry", "0", "--silent", "--show-error",
                                   "https://www.openslr.org/resources/12/dev-clean.tar.gz"]
    stages = {"source-binding", "authenticate-artifacts", "select-original-corpus"}
    tags = set(commands) | {"stage-" + x for x in stages}
    expected_starts = {"evidence/" + tag + ".started.json" for tag in tags}
    expected_exits = {"evidence/" + tag + ".exit.json" for tag in tags}
    need({n for n in files if n.endswith(".started.json")} == expected_starts and
         {n for n in files if n.endswith(".exit.json")} == expected_exits, "exact 11 command + 3 stage start/exit pairs")
    pairs = []
    for tag in sorted(tags):
        prefix = "evidence/" + tag
        begin, end = load(files[prefix + ".started.json"]), load(files[prefix + ".exit.json"])
        need(end["retry_count"] == 0 and math.isfinite(end["wall_seconds"]) and end["wall_seconds"] >= 0, "ended no-retry step")
        need(started["monotonic_started"] <= begin["monotonic_started"] < started["monotonic_deadline"] and
             begin["monotonic_started"] + end["wall_seconds"] < started["monotonic_deadline"], "step within original deadline")
        if tag in commands:
            need(begin["argv"] == end["argv"] == commands[tag] and begin["kind"] is end["kind"] is None,
                 "all actual commands are exact noncompute acquisitions")
            need(begin["attempt"] == end["attempt"] == 1 and begin["retry_count"] == 0 and begin["cpu_threads"] == 1 and
                 begin["new_session"] is False and begin["global_deadline"] == started["monotonic_deadline"] and
                 end["timed_out"] is False and end["descendants_supervised_by_parent_process_group"] is True, "command supervision/attempt")
            need(end["returncode"] == (28 if tag == "corpus-download" else 0), "only corpus curl fails")
            need(end["error"] == ("ValueError:COMMAND_NONZERO_EXIT" if tag == "corpus-download" else None), "command error context")
        else:
            need(begin["stage"] == end["stage"] == tag[len("stage-"):], "stage pairing")
            need(end["error"] == ("ValueError:ValueError:COMMAND_NONZERO_EXIT" if tag == "stage-select-original-corpus" else None), "only corpus acquisition stage fails")
        pairs.append({"tag": tag, "kind": begin.get("kind"), "argv": begin.get("argv"),
                      "returncode": end.get("returncode"), "error": end["error"], "wall_seconds": end["wall_seconds"],
                      "start_receipt": identity(files[prefix + ".started.json"]), "exit_receipt": identity(files[prefix + ".exit.json"])})

    absent_prefixes = ("selectedinputs/", "traces/", "scores/", "candidates/", "bin/", "downloads/",
                       "evidence/decode-", "evidence/scan-", "evidence/build-native", "evidence/native-", "evidence/ffmpeg-")
    need(not any(n.startswith(absent_prefixes) for n in files), "no selected audio/partial trace/compute output")
    preflight = load(files["evidence/INPUT_PREFLIGHT.json"])
    need(preflight["status"] == "STATIC_ARCHIVE_AND_ROSTER_VERIFIED" and preflight["model_calls"] == 0,
         "completed static preflight only")
    corpus_stderr = files["evidence/corpus-download.stderr"].decode("utf-8")
    match = re.fullmatch(r"curl: \(28\) Operation timed out after (\d+) milliseconds with (\d+) out of (\d+) bytes received\n", corpus_stderr)
    need(match is not None and tuple(map(int, match.groups())) == (300000, 259543943, 337926286), "exact curl timeout/received-byte evidence")
    corpus_exit = load(files["evidence/corpus-download.exit.json"])
    need(corpus_exit["wall_seconds"] == 300.04594356099994, "exact corpus command duration")
    return {
        "schema": "independent-natural32-failed-acquisition-audit-v1",
        "status": "ZERO_COMPUTE_VERIFIED_WITH_ONE_EXPLICIT_HISTORICAL_SOURCE_UPLOAD_OMISSION",
        "audit_completed": True, "zero_compute_verified": True, "full_artifact_closure_passed": False,
        "artifact": ARTIFACT, "run_binding": RUN_BINDING,
        "api_verification": "Root separately verified API identity; this auditor performs no network call.",
        "manifest": {"identity": identity(manifest_raw), "zip_file_count_including_manifest": 85,
                     "declared_payload_members": 85, "retained_payload_members": 84,
                     "retained_payload_members_all_bytes_and_sha256_verified": True,
                     "closed_set": False, "undeclared_retained_members": [],
                     "missing_members": {OMITTED: WORKFLOW_PIN},
                     "separate_frozen_workflow_identity": identity(frozen_workflow),
                     "omission_corroboration": "Exact separately frozen local workflow matched; omitted bytes were not present in or inserted into the ZIP.",
                     "retained_members": {name: identity(files[name]) for name in sorted(actual)}},
        "execution": {"build_attempts": 0, "decode_attempts": 0, "model_attempts": 0, "completed_validated_traces": 0,
                      "observed_originals": 0, "observed_audio_seconds": 0, "observed_embedding_rows": 0, "observed_ready_rows": 0,
                      "command_pairs": 11, "stage_pairs": 3, "all_command_kinds_are_null": True,
                      "all_commands_are_exact_gh_or_curl_acquisition_calls": True, "start_exit_pairs": pairs,
                      "no_selected_audio_or_partial_trace_or_score_or_candidate_files": True,
                      "preexisting_sdk_binary_was_retained_but_not_executed": True},
        "failure": {"stage": "select-original-corpus", "command": "corpus-download", "curl_returncode": 28,
                    "curl_timeout_milliseconds": 300000, "received_bytes_reported": 259543943,
                    "expected_corpus_bytes": 337926286, "command_wall_seconds": corpus_exit["wall_seconds"],
                    "partial_download_bytes_verified": False,
                    "partial_download_measurement_source": "Exact authenticated curl stderr; downloads were excluded from the artifact.",
                    "stderr": corpus_stderr, "supervisor_wall_seconds": supervisor["wall_seconds"],
                    "global_deadline_reached": False, "descendants_found": False, "retry_count": 0},
        "quality_boundary": {"scan_completed": False, "candidates_produced": False, "quality_improvement_claim": False,
                             "verified_negative_seconds": 0, "independent_far": "NOT_MEASURED", "independent_frr": "NOT_MEASURED",
                             **dict.fromkeys(FALSE_FLAGS, False)},
        "audit_execution": {"models_executed": 0, "audio_decodes_executed": 0, "compiler_calls": 0,
                            "network_calls": 0, "subprocess_calls": 0, "artifact_modified": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--old-workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), "create-only audit output")
    report = audit(args.artifact, args.old_workflow)
    report["auditor"] = identity(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as out:
        out.write((json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode())
    print(json.dumps({"output": str(args.output), "zero_compute_verified": True, "full_artifact_closure_passed": False,
                      "retained_members_verified": 84, "known_missing_source_workflows": 1}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
