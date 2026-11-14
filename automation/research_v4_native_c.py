#!/usr/bin/env python3
"""Consume the already trained English63 checkpoints for one native C comparison."""

from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import zipfile

# Reuse the existing bounded downloads, ZIP restoration, manifests and token-free
# child environment. Importing this stdlib module does not run its training CLI.
SPEC = importlib.util.spec_from_file_location(
    "research_v4_executor_core", Path(__file__).with_name("research_v4_executor.py"))
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

EXPERIMENT = "english63-saved-native-c-parity"
ENTRYPOINT = "research/english63_online_c/c_replay.py"
SOURCE_REQUEST = Path(__file__).parents[1] / core.REQUEST_DIR / "en63-online-20261114-v1.json"
CHECKPOINT = {
    "run_id": 43595338774,
    "artifact_id": 14114695344,
    "zip_sha256": "bc76805934df8eb158fa4259a75c26a4ea4276c15b83542acad5948a1f954b54",
    "data_commit": "14547faa42299ae9d819ba4bff9824c2471c412e",
    "training_pipeline_commit": "fc59bef7fe108ce6accbf51d8de0bfcaae19e0bc",
    "request_sha256": "ffa74269a0537115462da17b5050462e867cdb9980690105d2a92a4bdc0b7e53",
    "terminal_sha256": "7c840f2877c0d80a147b5d1287261faeb01495e61f8a2d8ab8cf9053c6cc91a4",
    "weights_sha256": {
        "gru": "71f731155a19538b533e4253d2e4b19d5c03328bd26da5263542159df066e623",
        "dsc_tcn": "167282c1037a28152bed4afaf49e53a4307bd8d5576a486b41c0a2291fdca935",
    },
}
FIXED = {
    "schema": "kws-research-v4-native-c-request-v1", "experiment": EXPERIMENT,
    "pipeline_repository": core.PIPELINE_REPOSITORY, "entrypoint": ENTRYPOINT,
    "source_artifacts": core.INPUTS, "trained_checkpoint": CHECKPOINT,
    "optimizer_updates": 0, "cpu_threads": 1, "chunk_plans": ["160", "mixed"],
    "numeric_abs_tolerance": 0.0002,
    "max_execution_seconds": 1200, "max_verification_seconds": 300,
    "new_audio_generated": False, "new_tts_asr_ipa_calls": 0,
    "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
}
COMPARISON_FILES = {
    "export/en63_generated.h", "export/EXPORT.json", "en63_replay", "BUILD.json",
    "gru-reference.npz", "dsc_tcn-reference.npz", "RESOURCES.json", "SUMMARY.json", "REPLAY.json",
} | {f"{kind}-{mode}-{plan}.trace" for kind in ("gru", "dsc_tcn")
     for mode in ("isolated", "continuous") for plan in ("160", "mixed")}


def load_request(path):
    path = Path(path)
    raw = path.read_bytes()
    core.require(len(raw) < 16384, "request too large")
    request = json.loads(raw, object_pairs_hook=core.unique_object)
    core.require(isinstance(request, dict) and set(request) == set(FIXED) | {"request_id", "pipeline_commit", "reason"}, "native C request keys differ from fixed task")
    for key, expected in FIXED.items():
        core.require(request[key] == expected and type(request[key]) is type(expected), "unsupported native C field: " + key)
    core.require(isinstance(request["request_id"], str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{2,79}", request["request_id"]), "invalid request_id")
    core.require(path.name == request["request_id"] + ".json", "request filename must match request_id")
    core.require(core.valid_sha(request["pipeline_commit"]), "native C source needs a reviewed full commit SHA")
    core.require(isinstance(request["reason"], str) and 1 <= len(request["reason"].strip()) <= 500, "request needs a bounded execution reason")
    return request, raw, core.sha256(raw)


def validate_checkpoint(result):
    request, _, digest = core.load_request(SOURCE_REQUEST)
    core.require(digest == CHECKPOINT["request_sha256"] and request["pipeline_commit"] == CHECKPOINT["training_pipeline_commit"], "consumed training request changed")
    terminal = core.validate_terminal(result, request, digest)
    core.require(terminal["run_id"] == str(CHECKPOINT["run_id"]) and terminal["run_attempt"] == "1" and terminal["data_commit"] == CHECKPOINT["data_commit"], "checkpoint was not produced by the pinned successful run")
    core.require(core.sha256((result / "TERMINAL.json").read_bytes()) == CHECKPOINT["terminal_sha256"], "original training terminal changed")
    for kind, expected in CHECKPOINT["weights_sha256"].items():
        core.require(core.sha256((result / "model" / (kind + "-weights.npz")).read_bytes()) == expected, "existing checkpoint changed: " + kind)


def validate_comparison(result):
    comparison = result / "comparison"
    core.require(not (comparison / "TERMINAL.json").exists(), "unexpected nested terminal")
    manifest = core.file_manifest(comparison)
    core.require(set(manifest) == COMPARISON_FILES, "native C comparison evidence is incomplete or contains unexpected files")
    core.require(all((comparison / name).stat().st_size > 0 for name in COMPARISON_FILES), "empty native C evidence")
    with (comparison / "en63_replay").open("rb") as stream:
        core.require(stream.read(4) == b"\x7fELF", "native Linux executable missing")
    for name in COMPARISON_FILES:
        if name.endswith(".trace"):
            with (comparison / name).open("rb") as stream:
                core.require(stream.read(8) == b"EN63TRC1", "native C trace header missing")
    summary = core.read_json(comparison / "SUMMARY.json")
    core.require(isinstance(summary, dict), "native C summary must be an object")
    expected = {
        "schema": "english63-saved-fp32-native-c-stream-parity-v1",
        "status": "MEASURED_NATIVE_C_STREAM_PARITY_NOT_KWS_QUALIFICATION",
        "source_audio_clips": 63, "source_pcm_samples": 2101760,
        "source_training_run_id": CHECKPOINT["run_id"], "optimizer_updates": 0,
        "new_tts_asr_ipa_calls": 0, "formal_training_admitted": False,
        "human_gold": False, "model_promotion_allowed": False,
        "model_selection_made": False, "threshold_selection": False,
        "numeric_tolerances": {"features": 0.0002, "logits": 0.0002, "probabilities": 0.0002},
    }
    expected.update({key: "NOT_RUN" for key in ("independent_english_frr", "continuous_negative_far_per_hour", "word_end_latency", "ssc305_board_profile")})
    for key, value in expected.items():
        core.require(summary.get(key) == value and type(summary.get(key)) is type(value), "native C summary boundary changed: " + key)
    rows = summary.get("models", [])
    core.require(isinstance(rows, list) and len(rows) == 2 and all(isinstance(row, dict) for row in rows) and {row.get("kind") for row in rows} == set(CHECKPOINT["weights_sha256"]), "native C candidate roster changed")
    for row in rows:
        fixed = {
            "source_weights_sha256": CHECKPOINT["weights_sha256"][row["kind"]],
            "isolated_clips": 63, "isolated_frames": 13010, "continuous_frames": 13134,
            "chunk_plans": ["160", "mixed"], "reference_forward_sequences": 64,
            "chunk_probabilities_bit_identical": True, "chunk_events_identical": True,
            "python_c_events_identical": True, "eof_padding_samples": 0, "eof_events": 0,
            "continuous_state_resets": 0, "resource_profile_scope": "native-linux-host-not-ssc305",
        }
        for key, value in fixed.items():
            core.require(row.get(key) == value and type(row.get(key)) is type(value), "native C model invariant failed: " + key)
        for key in ("max_feature_abs_error", "max_logit_abs_error", "max_probability_abs_error"):
            value = row.get(key)
            core.require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 0.0002, "native C numeric tolerance exceeded: " + key)
    replay = core.read_json(comparison / "REPLAY.json")
    core.require(isinstance(replay, dict), "native C verification must be an object")
    core.require(replay.get("schema") == "english63-native-c-saved-evidence-verification-v1" and replay.get("status") == "VERIFIED_SAVED_C_COMPARISON_ARTIFACTS", "native C saved-evidence verification missing")
    for key in ("optimizer_updates", "model_forward_calls", "new_audio_asr_calls"):
        core.require(type(replay.get(key)) is int and replay[key] == 0, "saved-evidence verification must not execute models")
    core.require(replay.get("formal_training_admitted") is False and replay.get("model_promotion_allowed") is False, "saved evidence cannot promote a model")
    core.require(replay.get("summary_sha256") == manifest["SUMMARY.json"], "verified summary identity differs")
    manifest.pop("REPLAY.json")
    core.require(replay.get("files_sha256") == manifest, "saved native C evidence manifest differs")


def validate_terminal(result, request, digest):
    terminal = core.read_json(result / "TERMINAL.json")
    core.require(isinstance(terminal, dict), "native C terminal must be an object")
    core.require(terminal.get("schema") == "kws-research-v4-native-c-terminal-v1" and terminal.get("request_sha256") == digest and terminal.get("request_id") == request["request_id"] and terminal.get("pipeline_commit") == request["pipeline_commit"], "native C terminal belongs to another request/code")
    core.require(terminal.get("optimizer_updates") == 0 and all(terminal.get(key) is False for key in ("formal_training_admitted", "human_gold", "model_promotion_allowed")), "native C terminal cannot train/promote")
    core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest and terminal.get("files_sha256") == core.file_manifest(result), "native C terminal file manifest changed")
    core.require(terminal.get("status") == "SUCCEEDED", "prior native C attempt failed; consumed request must not auto-retry")
    execution = core.read_json(result / "EXECUTION.json")
    core.require(isinstance(execution, dict), "native C execution receipt must be an object")
    core.require(execution.get("request_sha256") == digest and execution.get("compare_exit_code") == execution.get("verify_exit_code") == 0 and execution.get("source_checkpoint_before_after_verified") is True and execution.get("optimizer_updates") == 0, "native C terminal lacks actual comparison and readonly verification")
    validate_comparison(result)
    return terminal


def prepare(request_path, state):
    request, raw, digest = load_request(request_path)
    event = core.read_json(os.environ["GITHUB_EVENT_PATH"])
    core.require(os.environ.get("GITHUB_REPOSITORY") == core.REPOSITORY and os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF") == "refs/heads/main" and event["repository"]["private"] is False, "native C execution is public main only")
    state.mkdir()
    result_name = "research-v4-native-c-result-" + digest
    claim_name = "research-v4-native-c-claim-" + digest
    terminals, claims = core.named_artifacts(result_name), core.named_artifacts(claim_name)
    core.require(len(terminals) <= 1 and len(claims) <= 1, "ambiguous native C receipts; stop without repeating")
    if terminals:
        archive = state / "prior-result.zip"
        archive.write_bytes(core.artifact_bytes(terminals[0]))
        core.restore_archive(archive, state / "reused")
        validate_terminal(state / "reused", request, digest)
        core.output(execute=False, request_sha256=digest, result_artifact=result_name, prior_artifact_id=terminals[0]["id"])
        return
    core.require(not claims, "prior native C claim exists without a valid terminal; do not repeat")
    core.require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "native C rerun without retained receipts is blocked")
    for directory in ("inputs", "result", "claim"):
        (state / directory).mkdir()
    (state / "result/REQUEST.json").write_bytes(raw)
    run = core.github_get("/actions/runs/" + str(CHECKPOINT["run_id"]))
    core.require(run.get("status") == "completed" and run.get("conclusion") == "success" and run.get("run_attempt") == 1 and run.get("event") == "push" and run.get("head_sha") == CHECKPOINT["data_commit"], "source training run is not the pinned success")
    checkpoint = core.artifact_metadata(CHECKPOINT["artifact_id"], CHECKPOINT["zip_sha256"])
    core.require(checkpoint.get("workflow_run", {}).get("id") == CHECKPOINT["run_id"] and checkpoint.get("workflow_run", {}).get("head_sha") == CHECKPOINT["data_commit"], "trained artifact run identity differs")
    archive = state / "inputs/trained.zip"
    archive.write_bytes(core.artifact_bytes(checkpoint))
    core.restore_archive(archive, state / "inputs/trained-result")
    validate_checkpoint(state / "inputs/trained-result")
    for key, specification in core.INPUTS.items():
        metadata = core.artifact_metadata(specification["artifact_id"], specification["zip_sha256"])
        if key == "tts":
            core.require(metadata.get("workflow_run", {}).get("id") == 43242278962 and metadata.get("workflow_run", {}).get("head_sha") == "bec1a1421d1db2de17afb9c6e3155343becdb980", "original audio run identity changed")
        (state / "inputs" / (key + ".zip")).write_bytes(core.artifact_bytes(metadata))
    with zipfile.ZipFile(state / "inputs/review.zip") as handle:
        member = core.INPUTS["review"]["member"]
        core.require(handle.namelist().count(member) == 1 and handle.getinfo(member).file_size <= 8 * 1024 * 1024, "review member missing/duplicated/oversized")
        review = handle.read(member)
    core.require(core.sha256(review) == core.INPUTS["review"]["member_sha256"], "original review changed")
    (state / "inputs/review.json").write_bytes(review)
    core.write_json(state / "claim/CLAIM.json", {
        "schema": "kws-research-v4-native-c-start-claim-v1", "request_sha256": digest,
        "pipeline_commit": request["pipeline_commit"], "source_checkpoint_artifact_id": CHECKPOINT["artifact_id"],
        "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "optimizer_updates": 0, "status": "CONSUMED_BEFORE_NATIVE_C_COMPARISON",
    })
    core.output(execute=True, request_sha256=digest, result_artifact=result_name, claim_artifact=claim_name, prior_artifact_id="")


def run_bounded(command, *, pipeline, environment, log, seconds):
    # A timeout must terminate the compiler/native descendants, not just Python.
    with subprocess.Popen(command, cwd=pipeline, env=environment, stdout=log,
                          stderr=subprocess.STDOUT, start_new_session=True) as child:
        try:
            return child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
            raise


def execute(request_path, state, pipeline, python):
    request, _, digest = load_request(request_path)
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"], text=True).strip() == request["pipeline_commit"], "native C checkout SHA differs")
    core.require(not subprocess.check_output(["git", "-C", str(pipeline), "status", "--porcelain", "--untracked-files=no"], text=True).strip(), "native C tracked source is modified")
    script = pipeline / ENTRYPOINT
    core.require(script.is_file() and script.resolve().is_relative_to(pipeline.resolve()), "fixed native C entrypoint missing")
    core.require(core.read_json(state / "claim/CLAIM.json")["request_sha256"] == digest, "native C start claim differs")
    result = state / "result"
    environment = core.child_environment(python, state)
    base = [str(python), "-I", "-B", str(script), "--tts-zip", str(state / "inputs/tts.zip"),
            "--review-json", str(state / "inputs/review.json"), "--trained-result-dir", str(state / "inputs/trained-result")]
    execution = {"schema": "kws-research-v4-native-c-execution-v1", "request_sha256": digest,
                 "compare_exit_code": None, "verify_exit_code": None, "optimizer_updates": 0,
                 "source_checkpoint_before_after_verified": False, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        validate_checkpoint(state / "inputs/trained-result")
        for phase, argument, seconds in (("compare", "--output-dir", request["max_execution_seconds"]),
                                         ("verify", "--verify-result-dir", request["max_verification_seconds"])):
            with (result / (phase + ".log")).open("xb") as log:
                code = run_bounded(base + [argument, str(result / "comparison")], pipeline=pipeline, environment=environment, log=log, seconds=seconds)
            execution[phase + "_exit_code"] = code
            core.require(code == 0, phase + " process failed; retain partial native C evidence")
        validate_checkpoint(state / "inputs/trained-result")
        execution["source_checkpoint_before_after_verified"] = True
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        execution["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        execution["elapsed_seconds"] = round(time.monotonic() - started, 3)
        core.write_json(result / "EXECUTION.json", execution)
    core.require(execution["compare_exit_code"] == execution["verify_exit_code"] == 0 and execution["source_checkpoint_before_after_verified"], "native C comparison or no-inference verification failed")


def seal(request_path, state):
    request, _, digest = load_request(request_path)
    result = state / "result"
    terminal = {
        "schema": "kws-research-v4-native-c-terminal-v1", "status": "FAILED",
        "request_sha256": digest, "request_id": request["request_id"], "pipeline_commit": request["pipeline_commit"],
        "data_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "optimizer_updates": 0,
        "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        execution = core.read_json(result / "EXECUTION.json")
        core.require(isinstance(execution, dict), "native C execution receipt must be an object")
        core.require(execution.get("request_sha256") == digest and execution.get("compare_exit_code") == execution.get("verify_exit_code") == 0 and execution.get("source_checkpoint_before_after_verified") is True and execution.get("optimizer_updates") == 0, "native C execution/readonly source verification incomplete")
        validate_comparison(result)
        terminal["status"] = "SUCCEEDED"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        terminal["error"] = type(exc).__name__ + ": " + str(exc)
    terminal["files_sha256"] = core.file_manifest(result)
    core.write_json(result / "TERMINAL.json", terminal)
    core.output(terminal_status=terminal["status"])


def dispatch(args):
    if args.command == "prepare":
        prepare(args.request, args.state)
    elif args.command == "execute":
        core.require(args.pipeline is not None and args.pipeline.is_absolute() and args.python is not None and args.python.is_absolute(), "absolute native C checkout and Python executable required")
        execute(args.request, args.state, args.pipeline, args.python)
    elif args.command == "seal":
        seal(args.request, args.state)
    else:
        core.require(args.command == "check", "unsupported native C operation")
        request, _, digest = load_request(args.request)
        result = args.state / ("reused" if (args.state / "reused").exists() else "result")
        terminal = validate_terminal(result, request, digest)
        print(json.dumps({"status": terminal["status"], "request_sha256": digest,
                          "optimizer_updates": 0, "model_promotion_allowed": False}, sort_keys=True))
