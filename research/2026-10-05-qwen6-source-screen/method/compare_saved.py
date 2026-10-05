"""Verify saved source-screen artifacts, then make a private comparison report.

Standard library only. No inference runtime imports, decoding, retries or source
artifact writes. Both model raw freezes are checked before any labeled input.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from private_compare import MODELS, _audio, summarize

HERE = Path(__file__).resolve().parent
SOURCE_IDS = [f"qwen6-{i:03d}" for i in range(1, 7)]
MODEL_DIRS = ("sensevoice", "qwen06")
require = _audio.require
digest = _audio.digest


def _pure_helpers():
    """Reuse the source functions verbatim without importing the runner/runtime."""
    namespace = {"Path": Path, "json": json, "hashlib": hashlib, "re": re}
    selections = [
        (HERE / "public/core/asr6_contract.py",
         {"SHA", "BATCH_IDS", "MAX_FRAMES", "MAX_JOB_BYTES", "MAX_WAV_BYTES", "MAX_ARCHIVE_BYTES"},
         {"digest", "unique", "decode", "validate_job", "validate_input_freeze"}),
        (HERE / "public/run_asr6.py", set(), {"adapt", "recover_rows"}),
    ]
    for path, constants, functions in selections:
        tree = ast.parse(path.read_bytes(), filename=str(path))
        selected = [node for node in tree.body if
                    (isinstance(node, ast.FunctionDef) and node.name in functions) or
                    (isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants)]
        require(len(selected) == len(constants) + len(functions), "Missing reviewed helper definitions")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


_helpers = _pure_helpers()


def _safe(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink input/output path")
    return path


def _read(path, limit=8 * 1024**2):
    path = _safe(path)
    require(path.is_file() and path.stat().st_size <= limit, "Missing/oversized regular input")
    raw = path.read_bytes()
    require(len(raw) <= limit, "Input grew beyond byte cap")
    return raw


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_audio.no_duplicate_keys,
                       parse_float=_audio.finite_json_float,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    _audio.validate_json_values(value)
    return value


def _tree(root, total_cap):
    root = _safe(root)
    require(root.is_dir(), "Missing artifact directory")
    files, directories, size = {}, set(), 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Artifact symlink")
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.add(name)
        else:
            raw = _read(path, min(total_cap, 8 * 1024**2))
            size += len(raw)
            require(size <= total_cap, "Artifact aggregate byte cap")
            files[name] = raw
    expected_dirs = {str(parent) for name in files for parent in PurePosixPath(name).parents
                     if str(parent) != "."}
    require(directories == expected_dirs, "Unexpected empty artifact directory")
    return files


def _verify_freeze(raw, payloads, schema, false_field):
    freeze = _json(raw)
    keys = {"files", false_field} | ({"schema"} if schema else set())
    require(type(freeze) is dict and set(freeze) == keys, "Freeze fields")
    require(freeze[false_field] is False and (not schema or freeze["schema"] == schema), "Freeze scope")
    require(type(freeze["files"]) is dict and set(freeze["files"]) == set(payloads), "Frozen membership differs")
    for name, expected in freeze["files"].items():
        _audio.valid_hash(expected)
        require(digest(payloads[name]) == expected, "Frozen file hash mismatch: " + name)
    return freeze


def _verify_asr(root, tts_sha, asr_sha):
    files = _tree(root, 20 * 1024**2)
    mandatory = {"artifact-freeze.json", "raw-freeze.json", "blind-job.json", "blind-input-freeze.json",
                 "final-status.json", "resources.json"}
    top = {name for name in files if "/" not in name}
    require(mandatory <= top <= mandatory | {"setup-summary.json"}, "ASR artifact top-level membership")
    raw = {name: data for name, data in files.items() if name.startswith("raw/")}
    require(set(files) == top | set(raw), "Unexpected artifact directory")
    _verify_freeze(files["artifact-freeze.json"], {k: v for k, v in files.items() if k != "artifact-freeze.json"},
                   "asr6-artifact-freeze-v1", "private_labels_joined")
    _verify_freeze(files["raw-freeze.json"], raw, "asr6-raw-freeze-v1", "labels_joined")
    common = {"model-raw-freeze.json", "not-run.json", "plan.json", "contract.json", "decoder-inputs.json",
              "environment.json", "outcomes.initial.json", "model-load-started.json", "model-load.json",
              "load-failure.json", "outcomes.json", "summary.json"}
    allowed = common | {f"outcomes.{i:04d}.json" for i in range(1, 7)} | {
        f"{audio_id}.{suffix}.json" for audio_id in _helpers["BATCH_IDS"]["all6"]
        for suffix in ("input-failure", "started", "decoder", "receipt")}
    for name in MODEL_DIRS:
        prefix = "raw/" + name + "/"
        members = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        require("model-raw-freeze.json" in members and set(members) <= allowed, "Model raw membership")
        _verify_freeze(members["model-raw-freeze.json"],
                       {k: v for k, v in members.items() if k != "model-raw-freeze.json"}, None, "labels_joined")
        if "not-run.json" in members:
            not_run = _json(members["not-run.json"])
            require(set(members) == {"not-run.json", "model-raw-freeze.json"}
                    and not_run == {"schema": "asr6-supervisor-not-run-v1", "status": "not_run",
                                    "model_decode_attempts": 0, "record_origin": "supervisor_no_model_load_receipt",
                                    "requested_ids": _helpers["BATCH_IDS"]["all6"]}, "Invalid zero-attempt not-run receipt")
    require(all(k.split("/")[1] in MODEL_DIRS for k in raw), "Unexpected raw model directory")
    resources = _json(files["resources.json"])
    require(resources["candidate_sha256"] == asr_sha and resources["tts_candidate_sha256"] == tts_sha,
            "ASR resources do not match reviewed paired candidates")
    _audio.valid_hash(resources["preregistered_plan_sha256"])
    job_sha = resources["blind_job_sha256"]
    job = _helpers["validate_job"](files["blind-job.json"], job_sha)
    blind = _helpers["validate_input_freeze"](files["blind-input-freeze.json"], resources["blind_input_freeze_sha256"])
    require(blind["job_sha256"] == job_sha, "Blind job/freeze mismatch")
    bound = {row["path"]: row for row in blind["files"]}
    require(set(bound) == {"job.json"} | {row["audio_path"] for row in job["clips"]}
            and bound["job.json"]["bytes"] == len(files["blind-job.json"]), "Blind freeze/job membership mismatch")
    results = {}
    for name, model_id in zip(MODEL_DIRS, MODELS):
        recovered = _helpers["recover_rows"](Path(root) / "raw" / name, job, resources.get(name, {}))
        results[model_id] = _helpers["adapt"](recovered, {"model_id": model_id}, job_sha)["clips"]
    # Recovery reads saved files. Detect source changes before exposing the plan.
    require(_tree(root, 20 * 1024**2) == files, "ASR artifact changed during recovery")
    return files, resources, job, bound, results


def compare_saved(plan, generation_root, asr_root, tts_candidate_sha256, asr_candidate_sha256):
    """Read verified saved evidence and return a JSON report; never write sources."""
    for expected in (tts_candidate_sha256, asr_candidate_sha256):
        _audio.valid_hash(expected)
    # This call MUST finish both model freeze checks before any private read.
    asr_files, resources, job, blind, results = _verify_asr(asr_root, tts_candidate_sha256, asr_candidate_sha256)
    plan_raw = _read(plan)
    plan_sha = digest(plan_raw)
    require(plan_sha == resources["preregistered_plan_sha256"], "Preregistered plan hash mismatch")
    planned = _json(plan_raw)
    require(planned["schema"] == "qwen6-fixed-source-screen-v1" and planned["heldout_voice"] == "Sohee"
            and planned["heldout_generation_allowed"] is False, "Wrong private plan scope")
    requests = planned["generation_requests"]
    require(type(requests) is list and [r["source_id"] for r in requests] == SOURCE_IDS, "Fixed generation order/IDs changed")
    tts_resources_raw = _read(Path(generation_root).parent / "resources.json")
    tts_resources = _json(tts_resources_raw)
    require(tts_resources["candidate_sha256"] == tts_candidate_sha256
            and tts_resources["asr_candidate_sha256"] == asr_candidate_sha256
            and tts_resources["preregistered_plan_sha256"] == plan_sha, "TTS resources paired identity mismatch")
    files = _tree(generation_root, 16 * 1024**2)
    expected = {"generation-receipt.json"} | {source_id + suffix for source_id in SOURCE_IDS
                                             for suffix in (".raw-float.wav", ".wav", ".started.json")}
    require(set(files) == expected | {"generation-freeze.json"}, "Generation requires exact 19-file payload plus freeze")
    freeze = _json(files["generation-freeze.json"])
    require(set(freeze) == {"schema", "files"} and freeze["schema"] == "qwen6-generation-freeze-v1"
            and type(freeze["files"]) is list and len(freeze["files"]) == 19, "Generation freeze schema/count")
    frozen = {}
    for row in freeze["files"]:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
                and row["path"] in expected and row["path"] not in frozen, "Generation freeze membership")
        _audio.valid_hash(row["sha256"])
        raw = files[row["path"]]
        require(type(row["bytes"]) is int and len(raw) == row["bytes"] and digest(raw) == row["sha256"],
                "Generation frozen bytes mismatch")
        frozen[row["path"]] = row
    receipt = _json(files["generation-receipt.json"])
    require(receipt["schema"] == "qwen6-generation-receipt-v1" and receipt["status"] == "six_candidates_generated"
            and receipt["plan_sha256"] == plan_sha and len(receipt["generation_rows"]) == 6,
            "Incomplete or differently planned generation")
    private_plan, metrics = [], {}
    for request, generated, clip in zip(requests, receipt["generation_rows"], job["clips"]):
        source_id = request["source_id"]
        require(all(generated.get(k) == v for k, v in request.items()) and generated["status"] == "generated_candidate",
                "Generation row differs from frozen plan/order")
        require(request["attempts_max"] == 1 and request["human_gold"] is False
                and request["training_admitted"] is False, "Wrong generation authority/attempts")
        require(_json(files[source_id + ".started.json"]) == {"source_id": source_id, "seed": request["seed"], "attempt": 1},
                "Generation attempt receipt mismatch")
        raw = files[source_id + ".wav"]
        require(digest(raw) == generated["audio_sha256"] == clip["wav_sha256"]
                and len(raw) == blind[clip["audio_path"]]["bytes"], "Derived WAV is not the frozen ASR input")
        require(digest(files[source_id + ".raw-float.wav"]) == generated["raw_audio_sha256"], "Raw float WAV identity mismatch")
        measured = _audio.wav_metrics(raw, _audio.flag_config())
        require(measured["header"]["sample_rate"] == 16000 and 0 < measured["header"]["frames"] <= 192000
                and len(raw) == 44 + 2 * measured["header"]["frames"]
                and measured["pcm_sha256"] == generated["pcm_sha256"], "Derived PCM format/identity mismatch")
        flags = generated["signal_flags"]
        require(type(flags) is list and all(type(flag) is str for flag in flags), "Generation signal flags")
        termination = generated["termination"]
        require(termination.get("ended_with_eos") is True or "generation_end_unverified" in flags,
                "Unverified generation termination lacks quarantine flag")
        require(not termination.get("hit_token_cap_without_eos") or "token_cap_without_eos" in flags,
                "Token-cap termination lacks quarantine flag")
        measured["generation_flags"] = list(flags)
        metrics[clip["audio_id"]] = measured
        require(type(request["intended_keyword_id"]) is int and request["intended_keyword_id"] in (1, 2), "Keyword ID")
        private_plan.append({"audio_id": clip["audio_id"], "role": {"train": "train", "development": "dev"}[request["prospective_role"]],
                             "preset": request["voice"], "keyword": "K" + str(request["intended_keyword_id"]),
                             "intended_text": request["intended_text"]})
    report = summarize(private_plan, results, metrics)
    report["saved_artifact_verification"] = {
        "both_model_raw_freezes_verified_before_private_read": True,
        "tts_candidate_sha256": tts_candidate_sha256, "asr_candidate_sha256": asr_candidate_sha256,
        "plan_sha256": plan_sha, "asr_artifact_freeze_sha256": digest(asr_files["artifact-freeze.json"]),
        "raw_freeze_sha256": digest(asr_files["raw-freeze.json"]),
        "generation_freeze_sha256": digest(files["generation-freeze.json"]),
        "tts_resources_sha256": digest(tts_resources_raw),
        "note": "Saved byte identities verified; this does not independently prove historical decoder behavior",
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "generation-root", "asr-root", "out", "tts-candidate-sha256", "asr-candidate-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    output = _safe(args.out)
    sources = (_safe(args.asr_root), _safe(args.generation_root).parent)
    require(not output.exists() and output != _safe(args.plan)
            and not any(output == root or root in output.parents for root in sources), "Output must be new and outside source artifacts")
    report = compare_saved(args.plan, args.generation_root, args.asr_root,
                           args.tts_candidate_sha256, args.asr_candidate_sha256)
    encoded = _audio.encode_json(report)
    with output.open("xb") as stream:
        stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
