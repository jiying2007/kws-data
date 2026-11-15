#!/usr/bin/env python3
"""One fixed C discontinuity experiment using already consumed public inputs."""

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "research_v4_native_c_original", Path(__file__).with_name("research_v4_native_c.py"))
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)
core = native.core

EXPERIMENT = "english63-native-c-state-contracts"
ENTRYPOINT = "research/english63_online_c/state_replay.py"
SOURCE_REQUEST = Path(__file__).parents[1] / core.REQUEST_DIR / "en63-native-c-20261114-v1.json"
BASELINE = {
    "run_id": 43605922752,
    "artifact_id": 14117638750,
    "zip_sha256": "e85d14c87727f93b6e33edc196af00a6c9ce023d9e7d41d844109617fa5fb397",
    "data_commit": "0d5db0eaa0101ee8009882aa86dc85dc513cd174",
    "pipeline_commit": "6eab7c5e72d44483a14c5095a53805ea94126070",
    "request_sha256": "2221d418d8890b16ffb1a65973b364b89bacb440946b634e76f2e4b675d64a27",
    "terminal_sha256": "e78d5d623675a08d39dbf20b39814ab7de15fffda669768ffa23ecc2d05387cb",
    "header_sha256": "912f443a85fddbc77c49f6cc52c2dbde6c395436d7d8d0818a4e6225cd90fc16",
}
FIXED = {
    "schema": "kws-research-v4-native-c-state-request-v1", "experiment": EXPERIMENT,
    "pipeline_repository": core.PIPELINE_REPOSITORY, "entrypoint": ENTRYPOINT,
    "source_artifacts": core.INPUTS, "native_baseline": BASELINE,
    "optimizer_updates": 0, "cpu_threads": 1,
    "max_execution_seconds": 600, "max_verification_seconds": 180,
    "new_audio_generated": False, "new_tts_asr_ipa_calls": 0,
    "new_independent_audio_seconds": 0,
    "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
}
CASES = (
    "baseline_saved_prefix", "reset_midstream", "reset_after_eof",
    "dual_instance_interleaved", "discontinuity_399", "discontinuity_400",
    "discontinuity_32401", "eof_no_padding", "invalid_call_atomicity", "invalid_model_atomicity",
)
RESULT_FILES = {
    "en63_generated.h", "en63_state_cases", "BUILD.json", "INPUTS.json",
    "PROTOCOL.json", "SOURCE.zip", "SUMMARY.json", "REPLAY.json",
} | {kind + suffix for kind in ("gru", "dsc_tcn") for suffix in (
    ".trace", "-native.json", ".stderr", "-baseline-prefix.trace")}


def load_request(path):
    path = Path(path)
    raw = path.read_bytes()
    core.require(len(raw) < 16384, "state request too large")
    request = json.loads(raw, object_pairs_hook=core.unique_object)
    core.require(isinstance(request, dict) and set(request) == set(FIXED) | {
        "request_id", "pipeline_commit", "reason"}, "state request keys differ from fixed task")
    for key, expected in FIXED.items():
        core.require(type(request[key]) is type(expected) and
                     json.dumps(request[key], sort_keys=True, allow_nan=False) ==
                     json.dumps(expected, sort_keys=True, allow_nan=False),
                     "unsupported state request field: " + key)
    core.require(isinstance(request["request_id"], str) and re.fullmatch(
        r"[a-z0-9][a-z0-9-]{2,79}", request["request_id"]), "invalid state request_id")
    core.require(path.name == request["request_id"] + ".json", "state request filename differs")
    core.require(core.valid_sha(request["pipeline_commit"]), "state source needs a reviewed full SHA")
    core.require(isinstance(request["reason"], str) and 1 <= len(request["reason"].strip()) <= 500,
                 "state request needs a bounded reason")
    return request, raw, core.sha256(raw)


def validate_baseline(result):
    request, _, digest = native.load_request(SOURCE_REQUEST)
    core.require(digest == BASELINE["request_sha256"] and
                 request["pipeline_commit"] == BASELINE["pipeline_commit"],
                 "consumed native C request changed")
    terminal = native.validate_terminal(result, request, digest)
    core.require(terminal.get("run_id") == str(BASELINE["run_id"]) and
                 terminal.get("run_attempt") == "1" and
                 terminal.get("data_commit") == BASELINE["data_commit"],
                 "native baseline was not the pinned successful run")
    core.require(core.sha256((result / "TERMINAL.json").read_bytes()) == BASELINE["terminal_sha256"],
                 "native baseline terminal changed")
    core.require(core.sha256((result / "comparison/export/en63_generated.h").read_bytes()) ==
                 BASELINE["header_sha256"], "already exported C weights changed")


def validate_inputs(state):
    validate_baseline(state / "inputs/baseline-result")
    for key, expected in core.INPUTS.items():
        core.require(core.sha256((state / "inputs" / (key + ".zip")).read_bytes()) ==
                     expected["zip_sha256"], "original input changed: " + key)
    core.require(core.sha256((state / "inputs/review.json").read_bytes()) ==
                 core.INPUTS["review"]["member_sha256"], "original review member changed")


def validate_result(result):
    """Check retained numeric evidence only; never execute the C binaries."""
    root = result / "state-contracts"
    core.require(not (root / "TERMINAL.json").exists(), "unexpected nested state terminal")
    manifest = core.file_manifest(root)
    core.require(set(manifest) == RESULT_FILES, "state evidence file closure is incomplete or unexpected")
    core.require(all((root / name).stat().st_size > 0 for name in RESULT_FILES if not name.endswith(".stderr")),
                 "empty state evidence")
    core.require(manifest["en63_generated.h"] == BASELINE["header_sha256"], "state exported header changed")
    with (root / "en63_state_cases").open("rb") as stream:
        core.require(stream.read(4) == b"\x7fELF", "state native executable missing")
    for kind in ("gru", "dsc_tcn"):
        for suffix, magic, size in ((".trace", b"EN63BND1", 859156),
                                    ("-baseline-prefix.trace", b"EN63TRC1", 67048)):
            path = root / (kind + suffix)
            with path.open("rb") as stream:
                core.require(path.stat().st_size == size and stream.read(8) == magic,
                             "state native trace missing or truncated")
    summary = core.read_json(root / "SUMMARY.json")
    core.require(isinstance(summary, dict), "state summary must be an object")
    expected = {
        "schema": "english63-native-c-state-contracts-v1",
        "status": "MEASURED_NATIVE_C_STATE_BOUNDARIES_NOT_KWS_QUALIFICATION",
        "source_audio_clips": 63, "source_pcm_samples": 2101760,
        "test_unique_pcm_samples": 128246,
        "source_native_run_id": BASELINE["run_id"],
        "source_header_sha256": BASELINE["header_sha256"],
        "optimizer_updates": 0, "new_tts_asr_ipa_calls": 0,
        "new_independent_audio_seconds": 0,
        "formal_training_admitted": False, "human_gold": False,
        "model_promotion_allowed": False, "model_selection_made": False,
        "threshold_selection": False,
    }
    expected.update({key: "NOT_RUN" for key in (
        "independent_english_frr", "continuous_negative_far_per_hour",
        "word_end_latency", "ssc305_board_profile")})
    for key, value in expected.items():
        core.require(summary.get(key) == value and type(summary.get(key)) is type(value),
                     "state summary boundary differs: " + key)
    rows = summary.get("models")
    core.require(isinstance(rows, list) and len(rows) == 2 and
                 all(isinstance(row, dict) for row in rows) and
                 {row.get("kind") for row in rows} == {"gru", "dsc_tcn"}, "state model roster differs")
    for row in rows:
        core.require(row.get("all_contracts_passed") is True and
                     row.get("baseline_trace_bit_identical") is True,
                     "state contracts or original trace compatibility failed")
        core.require(row.get("source_weights_sha256") == native.CHECKPOINT["weights_sha256"][row["kind"]],
                     "state source checkpoint identity differs")
        cases = row.get("cases")
        core.require(isinstance(cases, list) and len(cases) == len(CASES) and
                     all(isinstance(case, dict) and case.get("passed") is True for case in cases) and
                     {case.get("name") for case in cases} == set(CASES),
                     "state cases are failed, missing, duplicated or unexpected")
    replay = core.read_json(root / "REPLAY.json")
    core.require(isinstance(replay, dict) and
                 replay.get("schema") == "english63-native-c-state-saved-verification-v1" and
                 replay.get("status") == "VERIFIED_SAVED_NATIVE_C_STATE_CONTRACTS",
                 "state saved verification missing")
    for key in ("model_forward_calls", "optimizer_updates", "new_audio_asr_calls"):
        core.require(type(replay.get(key)) is int and replay[key] == 0,
                     "state verification must not execute models: " + key)
    core.require(replay.get("formal_training_admitted") is False and
                 replay.get("model_promotion_allowed") is False,
                 "state verification cannot promote models")
    core.require(replay.get("summary_sha256") == manifest["SUMMARY.json"],
                 "state verified summary identity differs")
    manifest.pop("REPLAY.json")
    core.require(replay.get("files_sha256") == manifest, "state saved file manifest differs")


def state_environment(python, state):
    """Pass only five public run identities needed by the fixed native runner."""
    metadata = {key: os.environ.get(key, "") for key in (
        "GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_REPOSITORY", "GITHUB_SHA")}
    core.require(metadata["GITHUB_ACTIONS"] == "true" and metadata["GITHUB_RUN_ID"].isdigit() and
                 int(metadata["GITHUB_RUN_ID"]) > 0 and metadata["GITHUB_RUN_ATTEMPT"] == "1" and
                 metadata["GITHUB_REPOSITORY"] == core.REPOSITORY and
                 core.valid_sha(metadata["GITHUB_SHA"]), "state child requires exact public Actions run identity")
    return core.child_environment(python, state) | metadata


def execution_ok(execution, digest):
    core.require(isinstance(execution, dict) and execution.get("request_sha256") == digest and
                 all(type(execution.get(key)) is int and execution[key] == 0
                     for key in ("execute_exit_code", "verify_exit_code")) and
                 execution.get("source_inputs_before_after_verified") is True and
                 type(execution.get("optimizer_updates")) is int and execution["optimizer_updates"] == 0,
                 "state execution or source input verification incomplete")


def validate_terminal(result, request, digest):
    terminal = core.read_json(result / "TERMINAL.json")
    core.require(isinstance(terminal, dict) and
                 terminal.get("schema") == "kws-research-v4-native-c-state-terminal-v1" and
                 terminal.get("request_sha256") == digest and
                 terminal.get("request_id") == request["request_id"] and
                 terminal.get("pipeline_commit") == request["pipeline_commit"],
                 "state terminal belongs to another request/code")
    core.require(type(terminal.get("optimizer_updates")) is int and terminal["optimizer_updates"] == 0 and
                 all(terminal.get(key) is False for key in (
                     "formal_training_admitted", "human_gold", "model_promotion_allowed")),
                 "state terminal cannot train/promote")
    core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest and
                 terminal.get("files_sha256") == core.file_manifest(result), "state terminal manifest changed")
    core.require(terminal.get("status") == "SUCCEEDED",
                 "prior state attempt failed; consumed request must not auto-retry")
    execution_ok(core.read_json(result / "EXECUTION.json"), digest)
    validate_result(result)
    return terminal


def prepare(request_path, state):
    request, raw, digest = load_request(request_path)
    event = core.read_json(os.environ["GITHUB_EVENT_PATH"])
    core.require(os.environ.get("GITHUB_REPOSITORY") == core.REPOSITORY and
                 os.environ.get("GITHUB_EVENT_NAME") == "push" and
                 os.environ.get("GITHUB_REF") == "refs/heads/main" and
                 event["repository"]["private"] is False, "state execution is public main only")
    state.mkdir()
    result_name = "research-v4-c-state-result-" + digest
    claim_name = "research-v4-c-state-claim-" + digest
    terminals, claims = core.named_artifacts(result_name), core.named_artifacts(claim_name)
    core.require(len(terminals) <= 1 and len(claims) <= 1, "ambiguous state receipts; stop without repeating")
    if terminals:
        archive = state / "prior-result.zip"
        archive.write_bytes(core.artifact_bytes(terminals[0]))
        core.restore_archive(archive, state / "reused")
        validate_terminal(state / "reused", request, digest)
        core.output(execute=False, request_sha256=digest, result_artifact=result_name,
                    prior_artifact_id=terminals[0]["id"])
        return
    core.require(not claims, "prior state claim exists without a valid terminal; do not repeat")
    core.require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "state rerun without retained receipts is blocked")
    for directory in ("inputs", "result", "claim"):
        (state / directory).mkdir()
    (state / "result/REQUEST.json").write_bytes(raw)
    run = core.github_get("/actions/runs/" + str(BASELINE["run_id"]))
    core.require(run.get("status") == "completed" and run.get("conclusion") == "success" and
                 run.get("run_attempt") == 1 and run.get("event") == "push" and
                 run.get("head_sha") == BASELINE["data_commit"], "native baseline run identity differs")
    metadata = core.artifact_metadata(BASELINE["artifact_id"], BASELINE["zip_sha256"])
    core.require(metadata.get("workflow_run", {}).get("id") == BASELINE["run_id"] and
                 metadata.get("workflow_run", {}).get("head_sha") == BASELINE["data_commit"],
                 "native baseline artifact belongs to another run")
    archive = state / "inputs/baseline.zip"
    archive.write_bytes(core.artifact_bytes(metadata))
    core.restore_archive(archive, state / "inputs/baseline-result")
    validate_baseline(state / "inputs/baseline-result")
    for key, specification in core.INPUTS.items():
        metadata = core.artifact_metadata(specification["artifact_id"], specification["zip_sha256"])
        if key == "tts":
            core.require(metadata.get("workflow_run", {}).get("id") == 43242278962 and
                         metadata.get("workflow_run", {}).get("head_sha") ==
                         "bec1a1421d1db2de17afb9c6e3155343becdb980", "original audio run identity changed")
        (state / "inputs" / (key + ".zip")).write_bytes(core.artifact_bytes(metadata))
    with zipfile.ZipFile(state / "inputs/review.zip") as handle:
        member = core.INPUTS["review"]["member"]
        core.require(handle.namelist().count(member) == 1 and
                     handle.getinfo(member).file_size <= 8 * 1024 * 1024,
                     "review member missing/duplicated/oversized")
        review = handle.read(member)
    core.require(core.sha256(review) == core.INPUTS["review"]["member_sha256"], "original review changed")
    (state / "inputs/review.json").write_bytes(review)
    core.write_json(state / "claim/CLAIM.json", {
        "schema": "kws-research-v4-native-c-state-start-claim-v1", "request_sha256": digest,
        "pipeline_commit": request["pipeline_commit"], "source_native_artifact_id": BASELINE["artifact_id"],
        "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "optimizer_updates": 0, "status": "CONSUMED_BEFORE_NATIVE_C_STATE_EXPERIMENT",
    })
    core.output(execute=True, request_sha256=digest, result_artifact=result_name,
                claim_artifact=claim_name, prior_artifact_id="")


def execute(request_path, state, pipeline, python):
    request, _, digest = load_request(request_path)
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"],
                                        text=True).strip() == request["pipeline_commit"],
                 "state experiment checkout SHA differs")
    core.require(not subprocess.check_output(["git", "-C", str(pipeline), "status", "--porcelain",
                                              "--untracked-files=no"], text=True).strip(),
                 "state experiment tracked source is modified")
    script = pipeline / ENTRYPOINT
    core.require(script.is_file() and script.resolve().is_relative_to(pipeline.resolve()),
                 "fixed state entrypoint missing")
    core.require(core.read_json(state / "claim/CLAIM.json")["request_sha256"] == digest,
                 "state start claim differs")
    result = state / "result"
    environment = state_environment(python, state)
    base = [str(python), "-I", "-B", str(script),
            "--baseline-result-dir", str(state / "inputs/baseline-result"),
            "--tts-zip", str(state / "inputs/tts.zip"), "--review-json", str(state / "inputs/review.json")]
    execution = {"schema": "kws-research-v4-native-c-state-execution-v1", "request_sha256": digest,
                 "execute_exit_code": None, "verify_exit_code": None, "optimizer_updates": 0,
                 "source_inputs_before_after_verified": False,
                 "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        validate_inputs(state)
        for phase, argument, seconds in (("execute", "--output-dir", request["max_execution_seconds"]),
                                         ("verify", "--verify-result-dir", request["max_verification_seconds"])):
            with (result / (phase + ".log")).open("xb") as log:
                code = native.run_bounded(base + [argument, str(result / "state-contracts")],
                                          pipeline=pipeline, environment=environment, log=log, seconds=seconds)
            execution[phase + "_exit_code"] = code
            core.require(code == 0, phase + " process failed; retain partial state evidence")
        validate_inputs(state)
        execution["source_inputs_before_after_verified"] = True
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        execution["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        execution["elapsed_seconds"] = round(time.monotonic() - started, 3)
        core.write_json(result / "EXECUTION.json", execution)
    execution_ok(execution, digest)


def seal(request_path, state):
    request, _, digest = load_request(request_path)
    result = state / "result"
    terminal = {
        "schema": "kws-research-v4-native-c-state-terminal-v1", "status": "FAILED",
        "request_sha256": digest, "request_id": request["request_id"], "pipeline_commit": request["pipeline_commit"],
        "data_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "optimizer_updates": 0,
        "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        execution_ok(core.read_json(result / "EXECUTION.json"), digest)
        validate_result(result)
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
        core.require(args.pipeline is not None and args.pipeline.is_absolute() and
                     args.python is not None and args.python.is_absolute(),
                     "absolute state experiment checkout and Python required")
        execute(args.request, args.state, args.pipeline, args.python)
    elif args.command == "seal":
        seal(args.request, args.state)
    else:
        core.require(args.command == "check", "unsupported state operation")
        request, _, digest = load_request(args.request)
        result = args.state / ("reused" if (args.state / "reused").exists() else "result")
        terminal = validate_terminal(result, request, digest)
        print(json.dumps({"status": terminal["status"], "request_sha256": digest,
                          "optimizer_updates": 0, "model_promotion_allowed": False}, sort_keys=True))
