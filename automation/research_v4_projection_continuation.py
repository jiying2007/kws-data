#!/usr/bin/env python3
"""One consumed gradient-projection study with immutable frozen comparators."""

from datetime import datetime, timezone
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "projection_fixed_background_helpers", Path(__file__).with_name("research_v4_background_continuation.py"))
background = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(background)
core, native, state_consumer = background.core, background.native, background.state_consumer
exact, fields = background.exact, background.fields
file_sha256, validate_file = background.file_sha256, background.validate_file
download_corpus = background.download_corpus
closed_archive = background.closed_archive

EXPERIMENT = "english63-one-sided-background-gradient-projection"
REQUEST_ID = "en63-projection48-20261115-v1"
ENTRYPOINT = "research/english63_projection_ab/run_projection.py"
RESULT_DIR = "projection-continuation"
PIPELINE_COMMIT = "e284a4e8cdcc41973a5fa3ab44f71d833b019d1d"
SOURCE_SHA256 = {**background.SOURCE_SHA256,
    ENTRYPOINT: "020b651bb08d6eebb1e228fbdf947b30331e6357c5319eb3ce229bf43d9770ff",
    "research/english63_projection_ab/projection_contract.py": "596473598c13f95a15c4fb8e5510e46be3fbac033158316c48f6af3e38e8afdf",
    "research/english63_projection_ab/protocol.json": "68db9ba92de313160b772b1950d3cf3e80ea41c607564e1a01c1a9a40a744b45"}
PRIOR = {
    "run_id": 43681928374, "artifact_id": 14149230666,
    "request_id": "en63-background48-20261115-v1",
    "zip_bytes": 7850625,
    "zip_sha256": "d1ac243201a7c29db5d1bef188812d77e2e3906c3d2f084da77557f505a8b9e7",
    "data_commit": "48e1234a506334d05e962f6a30367f2d50947598",
    "pipeline_commit": "849040b48bb813d61722de7e1cdd0d0df20ca828",
    "request_sha256": "e48b5f6d7a22198c254fb1971d00702cee1b1af25061ee08dbee8337e4a8dd0a",
    "terminal_sha256": "7d38996cf949bbfb9edd7256787c4ea545592ad3f8e307f85c3ef7df6443104c",
    "summary_sha256": "fc7eac1e3f10e6fdcea8ce915d12e74878a41c25c1aced64177ec444d2df2d4b",
    "inputs_sha256": "f1723743acbe564285bac570b0e0574b3da34e3a45e6c9420e7e40b7a8e32c80",
    "training_sha256": "7c4b00f40ee1eec01d5c14a6ccd1fb7ceead6956cc9d32ad9082c7676927fadd",
    "replay_sha256": "0a5d11320e98bec053a3f716df809b8bc5dfcd852598459c8ad51b4dcf8d9535",
    "weights_zip_sha256": "eac1dc6caf389909a413dbd86b648b55bcdd1768cc598fa32536b071458e2f83",
    "weights_sha256": {
        "control-gru.npz": "9e034303f5101f3a74929c89a385af5f3a11519707bced99acd100fe619178c0",
        "treatment-gru.npz": "94bfb50f87ec70bc1d9f25bad808ae5e0ccf38d3a60a1394983405ccd3e2aa2f",
        "control-dsc_tcn.npz": "47c0e3bf66c9211194130edba7abb41c3df9f1d3a8151957b30ce60d818bf627",
        "treatment-dsc_tcn.npz": "31f47fb380c1bf4bef3b0806f3966ae6dd19d8c97cf3664e42f442103e467e76"}}
FIXED = copy.deepcopy(background.FIXED)
FIXED.update(schema="kws-research-v4-projection-continuation-request-v1",
             experiment=EXPERIMENT, request_id=REQUEST_ID, entrypoint=ENTRYPOINT)
FIXED["source_artifacts"]["background"] = {key: PRIOR[key] for key in ('run_id',
 'artifact_id',
 'zip_bytes',
 'zip_sha256',
 'data_commit',
 'pipeline_commit',
 'request_id',
 'request_sha256',
 'inputs_sha256',
 'summary_sha256',
 'replay_sha256',
 'weights_sha256')}
FIXED["profile"].update({'variants': ['baseline', 'control', 'treatment', 'projected'],
 'training_arms': 2,
 'optimizer_updates': 96,
 'objective': 'same old30-MIL-CE and new128-background-MIL-CE losses; one-sided global weighted-background '
              'gradient projection',
 'training_group_seconds': 360,
 'native_model_passes': 8,
 'native_eval_frames': 575720,
 'benefit_gate': 'B>0,C>0,2*P<=B,2*P<=C;each-natural-class-P<=min(B,C);each-keyword-correct-clips>=max(B,C,2);each-keyword-wrong-other-clips<=min(B,C);ordinary-background-proposals<=min(B,C)',
 'training_forward_calls': 480,
 'training_padded_forward_frames': 4669824,
 'reused_frozen_training_arms': 4,
 'gradient_projection': {'old_gradient': 'g=grad(old30-MIL-CE)',
                         'weighted_background_gradient': 'h=sum(four microbatch gradients weighted 0.125 '
                                                         'each)=0.5*grad(mean new128-MIL-CE)',
                         'order': 'model.named_parameters(), verified against frozen exporter parameter '
                                  'roster',
                         'dot_product': 'math.fsum of elementwise products of exact FP32 gradient values '
                                        'converted to Python double',
                         'projection': 'd=dot(g,h);n=dot(g,g);if d<0 and '
                                       'n>0:h_prime=h-(d/n)*g;else:h_prime=h',
                         'zero_reference': 'leave h unchanged when n==0',
                         'epsilon': None,
                         'cast': 'compute projection and g+h_prime in FP64; one cast of combined gradient to '
                                 'FP32 before original clipping and AdamW',
                         'gradient_tolerance': 1e-06,
                         'tape_arrays': ['old_gradient',
                                         'weighted_background_gradient',
                                         'combined_preclip',
                                         'actual_clipped_gradient',
                                         'actual_parameter_delta',
                                         'parameter_after'],
                         'tape_dtype': '<f4',
                         'tape_shape': '(48,parameter_count)',
                         'delta': 'FP32(theta_after-theta_before); verify against successive parameter_after '
                                  'rows and original baseline',
                         'per_class_old_loss': 'classes 0,1,2 from same old30 forward; diagnostic only',
                         'optimizer': 'fresh original AdamW, learning_rate=0.001, weight_decay=0.01, '
                                      'clip_grad_norm_=1.0'}})
KINDS, VARIANTS = ("gru", "dsc_tcn"), ("baseline", "control", "treatment", "projected")
TRACE_FILES = {variant + "-" + kind + ".trace" for variant in VARIANTS for kind in KINDS}
RESULT_FILES = {"PROTOCOL.json", "INPUTS.json", "SOURCE.zip", "TRAINING.json", "BUILD.json",
                "MEASUREMENTS.json", "SUMMARY.json", "REPLAY.json", "native.stderr", "WEIGHTS.zip",
                "RUNTIME.zip", "TRACES.zip", "GRADIENTS.zip"}
UPDATE_BUDGET, FORWARD_FRAMES = 96, 575720
WORK_COUNTERS = {"optimizer_updates": UPDATE_BUDGET,
                 "training_forward_calls": 480, "training_padded_forward_frames": 4669824,
                 "native_forward_frames": FORWARD_FRAMES,
                 "native_initializations": 104, "native_finishes": 104,
                 "native_feed_calls": 575928, "native_supplied_samples": 92147872}
WEIGHT_FILES = set(PRIOR["weights_sha256"]) | {"projected-" + kind + ".npz" for kind in KINDS}
RUNTIME_FILES = {variant + extension for variant in VARIANTS for extension in (".h", ".bin")}
GRADIENT_FILES = {kind + "-gradients.npz" for kind in KINDS}
FALSE_FIELDS = background.FALSE_FIELDS
DOWNLOADS = {**background.DOWNLOADS, "background": {**PRIOR, "event": "push"}}


def validate_background(result):
    """Verify the prior terminal through its unchanged consumer before reuse."""
    root = result / background.RESULT_DIR
    for path, expected in ((result / "REQUEST.json", PRIOR["request_sha256"]),
                           (result / "TERMINAL.json", PRIOR["terminal_sha256"]),
                           (root / "SUMMARY.json", PRIOR["summary_sha256"]),
                           (root / "INPUTS.json", PRIOR["inputs_sha256"]),
                           (root / "TRAINING.json", PRIOR["training_sha256"]),
                           (root / "REPLAY.json", PRIOR["replay_sha256"]),
                           (root / "WEIGHTS.zip", PRIOR["weights_zip_sha256"])):
        core.require(path.is_file() and not path.is_symlink() and file_sha256(path) == expected,
                     "frozen background artifact member changed: " + path.name)
    request = background.request_value((result / "REQUEST.json").read_bytes())
    terminal = background.validate_terminal(result, request, PRIOR["request_sha256"])
    fields(terminal, {"run_id": str(PRIOR["run_id"]), "run_attempt": "1",
                      "data_commit": PRIOR["data_commit"], "pipeline_commit": PRIOR["pipeline_commit"]},
           "frozen background public run differs")
    weights = closed_archive(root / "WEIGHTS.zip", set(PRIOR["weights_sha256"]), 500000)
    core.require(all(core.sha256(raw) == PRIOR["weights_sha256"][name] for name, raw in weights.items()),
                 "frozen comparator checkpoints changed")


def validate_traces(path):
    """Use the unchanged geometry validator on exactly eight retained traces."""
    traces = closed_archive(path, TRACE_FILES, background.TRACE_BYTES)
    with tempfile.TemporaryDirectory(prefix="en63-projection-saved-traces-") as directory:
        work = []
        for name, raw in traces.items():
            target = Path(directory) / name
            target.write_bytes(raw)
            work.append(background.validate_trace_geometry(target))
    core.require(all(sum(row[key] for row in work) == value for key, value in WORK_COUNTERS.items()
                     if key.startswith("native_")), "projection total saved trace workload differs")


def request_value(raw):
    core.require(len(raw) < 32768, "projection request too large")
    request = json.loads(raw, object_pairs_hook=core.unique_object)
    core.require(type(request) is dict and set(request) == set(FIXED) | {"pipeline_commit", "reason"},
                 "projection request keys differ")
    for key, expected in FIXED.items():
        core.require(exact(request[key], expected), "unsupported projection request field: " + key)
    core.require(request["pipeline_commit"] == PIPELINE_COMMIT and core.valid_sha(PIPELINE_COMMIT),
                 "projection source differs from the reviewed commit")
    core.require(type(request["reason"]) is str and 1 <= len(request["reason"].strip()) <= 500,
                 "projection request needs a bounded reason")
    return request


def load_request(path):
    path = Path(path)
    raw = path.read_bytes()
    request = request_value(raw)
    core.require(path.name == REQUEST_ID + ".json", "projection request filename differs")
    return request, raw, core.sha256(raw)


def validate_inputs(state):
    background.validate_inputs(state)
    validate_file(state / "inputs/background.zip", PRIOR["zip_bytes"], PRIOR["zip_sha256"],
                  "frozen background result archive")
    validate_background(state / "inputs/background-result")


def claim_record(request, digest, run=None):
    run = run or {"run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"]}
    return {"schema": "kws-research-v4-projection-continuation-start-claim-v1", "request_sha256": digest,
            "pipeline_commit": request["pipeline_commit"], "source_checkpoint_artifact_id": 14114695344,
            "source_negative_artifact_id": 14090383839, "source_background_artifact_id": PRIOR["artifact_id"],
            "source_background_request_sha256": PRIOR["request_sha256"],
            "run_id": run["run_id"], "run_attempt": run["run_attempt"],
            "optimizer_updates": 0, "optimizer_update_budget": UPDATE_BUDGET,
            "status": "CONSUMED_BEFORE_GRADIENT_PROJECTION"}


def prepare(request_path, state):
    request, raw, digest = load_request(request_path)
    event = core.read_json(os.environ["GITHUB_EVENT_PATH"])
    core.require(os.environ.get("GITHUB_REPOSITORY") == core.REPOSITORY and
                 os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF") == "refs/heads/main" and
                 event["repository"]["private"] is False, "projection execution is public main only")
    state.mkdir()
    result_name, claim_name = "research-v4-projection-result-" + digest, "research-v4-projection-claim-" + digest
    terminals, claims = core.named_artifacts(result_name), core.named_artifacts(claim_name)
    core.require(len(terminals) <= 1 and len(claims) <= 1, "ambiguous projection receipts; stop without repeating")
    if terminals:
        archive = state / "prior-result.zip"
        archive.write_bytes(core.artifact_bytes(terminals[0]))
        core.restore_archive(archive, state / "reused")
        terminal = validate_terminal(state / "reused", request, digest)
        fields(terminals[0].get("workflow_run"), {"id": int(terminal["run_id"]), "head_sha": terminal["data_commit"]},
               "projection reused archive public run")
        core.output(execute=False, request_sha256=digest, result_artifact=result_name,
                    prior_artifact_id=terminals[0]["id"])
        return
    core.require(not claims, "prior projection claim without valid terminal; do not repeat")
    core.require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "projection rerun without receipts is blocked")
    for name in ("inputs", "result", "claim"):
        (state / name).mkdir()
    (state / "result/REQUEST.json").write_bytes(raw)
    for name, specification in DOWNLOADS.items():
        run = core.github_get("/actions/runs/" + str(specification["run_id"]))
        fields(run, {"status": "completed", "conclusion": "success", "run_attempt": 1,
                     "event": specification["event"], "head_sha": specification["data_commit"]},
               name + " original successful run")
        metadata = core.artifact_metadata(specification["artifact_id"], specification["zip_sha256"])
        fields(metadata, {"size_in_bytes": specification["zip_bytes"]}, name + " archive size")
        fields(metadata.get("workflow_run"), {"id": specification["run_id"], "head_sha": specification["data_commit"]},
               name + " archive public run")
        archive = state / "inputs" / (name + ".zip")
        archive.write_bytes(core.artifact_bytes(metadata))
        validate_file(archive, specification["zip_bytes"], specification["zip_sha256"], name + " archive")
        if name in ("trained", "negative", "background"):
            core.restore_archive(archive, state / "inputs" / (name + "-result"))
    with zipfile.ZipFile(state / "inputs/review.zip") as archive:
        member = core.INPUTS["review"]["member"]
        core.require(archive.namelist().count(member) == 1, "review member absent or duplicated")
        info = archive.getinfo(member)
        core.require(not info.is_dir() and info.file_size < 8 * 1024 * 1024 and
                     (info.external_attr >> 16) & 0o170000 != 0o120000, "invalid review member")
        review = archive.read(member)
    core.require(core.sha256(review) == core.INPUTS["review"]["member_sha256"], "review JSON digest changed")
    (state / "inputs/review.json").write_bytes(review)
    download_corpus(state / "inputs/dev-clean.tar.gz")
    validate_inputs(state)
    core.write_json(state / "claim/CLAIM.json", claim_record(request, digest))
    core.output(execute=True, request_sha256=digest, result_artifact=result_name,
                claim_artifact=claim_name, prior_artifact_id="")


def verified_source(archive_path):
    """Only the complete reviewed source closure may become an executable reader."""
    core.require(ENTRYPOINT in SOURCE_SHA256 and all(core.valid_sha(value, 64) for value in SOURCE_SHA256.values()),
                 "reviewed projection source closure incomplete")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        core.require(len(infos) == len(SOURCE_SHA256) and set(archive.namelist()) == set(SOURCE_SHA256) and
                     all(not info.is_dir() and 0 < info.file_size < 4 * 1024 * 1024 and
                         (info.external_attr >> 16) & 0o170000 != 0o120000 for info in infos),
                     "projection source path closure differs")
        source = {name: archive.read(name) for name in SOURCE_SHA256}
    core.require(all(core.sha256(raw) == SOURCE_SHA256[name] for name, raw in source.items()),
                 "projection source bytes differ from the reviewed allowlist")
    return source


def run_saved_verifier(root, source):
    with tempfile.TemporaryDirectory(prefix="en63-projection-saved-reader-") as directory:
        temporary = Path(directory)
        tree = temporary / "source"
        tree.mkdir()
        for name, raw in source.items():
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        python = Path(sys.executable).resolve()
        command = [str(python), "-I", "-S", "-B", str(tree / ENTRYPOINT), "--verify-result-dir", str(root.resolve())]
        process = subprocess.run(command, env=core.child_environment(python, temporary), cwd=tree,
                                 capture_output=True, text=True, timeout=180, check=False)
        core.require(type(process.returncode) is int and process.returncode == 0 and
                     len(process.stdout) < 65536 and process.stderr == "", "frozen projection saved-only verifier failed")
        return json.loads(process.stdout, object_pairs_hook=core.unique_object)


def validate_checkout_sources(pipeline):
    core.require(ENTRYPOINT in SOURCE_SHA256, "reviewed projection checkout source closure incomplete")
    for name, expected in SOURCE_SHA256.items():
        path = pipeline / name
        core.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(pipeline.resolve()) and
                     file_sha256(path) == expected, "projection checkout differs from reviewed source: " + name)


def execution_ok(execution, digest):
    fields(execution, {"schema": "kws-research-v4-projection-continuation-execution-v1", "request_sha256": digest,
           "execute_exit_code": 0, "verify_exit_code": 0, **WORK_COUNTERS,
           "optimizer_update_budget": UPDATE_BUDGET,
           "source_inputs_before_after_verified": True}, "projection execution incomplete")


def execute(request_path, state, pipeline, python):
    request, _, digest = load_request(request_path)
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"], text=True).strip() ==
                 request["pipeline_commit"], "projection checkout SHA differs")
    core.require(not subprocess.check_output(["git", "-C", str(pipeline), "status", "--porcelain",
                                              "--untracked-files=no"], text=True).strip(), "projection tracked source modified")
    validate_checkout_sources(pipeline)
    script = pipeline / ENTRYPOINT
    core.require(script.is_file() and script.resolve().is_relative_to(pipeline.resolve()), "fixed projection entrypoint missing")
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "show", request["pipeline_commit"] + ":" + ENTRYPOINT]) ==
                 script.read_bytes(), "projection entrypoint differs from reviewed commit")
    core.require(exact(core.read_json(state / "claim/CLAIM.json"), claim_record(request, digest)), "projection claim differs")
    result = state / "result"
    environment = state_consumer.state_environment(python, state)
    command = [str(python), "-I", "-B", str(script), "--trained-result-dir", str(state / "inputs/trained-result"),
               "--tts-zip", str(state / "inputs/tts.zip"), "--review-json", str(state / "inputs/review.json"),
               "--negative-evidence-dir", str(state / "inputs/negative-result"),
               "--source-tar", str(state / "inputs/dev-clean.tar.gz"),
               "--background-result-dir", str(state / "inputs/background-result" / background.RESULT_DIR),
               "--output-dir", str(result / RESULT_DIR)]
    execution = {"schema": "kws-research-v4-projection-continuation-execution-v1", "request_sha256": digest,
                 "execute_exit_code": None, "verify_exit_code": None, **dict.fromkeys(WORK_COUNTERS),
                 "optimizer_update_budget": UPDATE_BUDGET,
                 "source_inputs_before_after_verified": False, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        validate_inputs(state)
        with (result / "execute.log").open("xb") as log:
            execution["execute_exit_code"] = native.run_bounded(command, pipeline=pipeline, environment=environment,
                                                               log=log, seconds=1200)
        core.require(execution["execute_exit_code"] == 0, "projection execution failed; retain partial evidence")
        source = verified_source(result / RESULT_DIR / "SOURCE.zip")
        verified = run_saved_verifier(result / RESULT_DIR, source)
        core.write_json(result / "verify.log", verified)
        execution["verify_exit_code"] = 0
        summary = core.read_json(result / RESULT_DIR / "SUMMARY.json")
        core.require(type(summary) is dict, "projection summary must be an object")
        execution.update({key: summary.get(key) for key in WORK_COUNTERS})
        validate_inputs(state)
        validate_checkout_sources(pipeline)
        execution["source_inputs_before_after_verified"] = True
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        execution["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        execution["elapsed_seconds"] = round(time.monotonic() - started, 3)
        core.write_json(result / "EXECUTION.json", execution)
    execution_ok(execution, digest)


def validate_result(result):
    root = result / RESULT_DIR
    core.require(root.is_dir() and not root.is_symlink() and
                 all(path.is_file() and not path.is_symlink() for path in root.iterdir()),
                 "projection result must contain flat regular files")
    manifest = core.file_manifest(root)
    core.require(set(manifest) == RESULT_FILES and all((root / name).stat().st_size > 0
                 for name in RESULT_FILES if name != "native.stderr"), "projection result file closure differs")
    outer = {"REQUEST.json", "EXECUTION.json", "execute.log", "verify.log", "runtime.txt"} | {
        RESULT_DIR + "/" + name for name in RESULT_FILES}
    core.require(set(core.file_manifest(result)) == outer, "projection outer result closure differs")
    validate_traces(root / "TRACES.zip")
    gradients = closed_archive(root / "GRADIENTS.zip", GRADIENT_FILES, 8 * 1024 * 1024)
    core.require(all(raw.startswith(b"PK\x03\x04") for raw in gradients.values()), "projection gradient tapes missing")
    weights = closed_archive(root / "WEIGHTS.zip", WEIGHT_FILES, 500000)
    core.require(all(raw.startswith(b"PK\x03\x04") for raw in weights.values()), "projection saved NPZ missing")
    core.require(all(core.sha256(weights[name]) == digest for name, digest in PRIOR["weights_sha256"].items()),
                 "projection changed a frozen comparator checkpoint")
    runtime = closed_archive(root / "RUNTIME.zip", RUNTIME_FILES, 1024 * 1024)
    core.require(core.sha256(runtime["baseline.h"]) == FIXED["profile"]["baseline_header_sha256"] and
                 all(runtime[variant + ".bin"].startswith(b"\x7fELF") for variant in VARIANTS),
                 "projection frozen baseline header or native binaries differ")
    source = verified_source(root / "SOURCE.zip")
    summary, inputs, build, training, replay = [core.read_json(root / name) for name in (
        "SUMMARY.json", "INPUTS.json", "BUILD.json", "TRAINING.json", "REPLAY.json")]
    core.require(all(type(report) is dict for report in (summary, inputs, build, training, replay)),
                 "projection receipts must be objects")
    request_raw = (result / "REQUEST.json").read_bytes()
    request, digest = request_value(request_raw), core.sha256(request_raw)
    run = build.get("public_run")
    core.require(type(run) is dict and set(run) == {"repository", "head", "run_id", "run_attempt"} and
                 run["repository"] == core.REPOSITORY and core.valid_sha(run["head"]) and
                 type(run["run_id"]) is str and run["run_id"].isdigit() and int(run["run_id"]) > 0 and
                 run["run_attempt"] == "1", "projection run identity differs")
    for report in (summary, build):
        fields(report, {"public_run": run, "pipeline_head": request["pipeline_commit"]}, "projection run/source binding")
    fields(inputs, {"request_raw_utf8": request_raw.decode("utf-8"), "request_sha256": digest},
           "projection retained request identity")
    claim_raw = inputs.get("claim_raw_utf8")
    core.require(type(claim_raw) is str and core.sha256(claim_raw.encode()) == inputs.get("claim_sha256") and
                 exact(json.loads(claim_raw, object_pairs_hook=core.unique_object), claim_record(request, digest, run)),
                 "projection consumed claim identity differs")
    fields(build, {"source_sha256": SOURCE_SHA256}, "projection reviewed source closure")
    fields(training, {"optimizer_updates_started": UPDATE_BUDGET, "optimizer_updates_completed": UPDATE_BUDGET,
                      "active_training": None}, "projection training group incomplete")
    fields(summary, {"schema": "english63-projection-ab-result-v1", "experiment": EXPERIMENT,
           "request_sha256": digest, **WORK_COUNTERS,
           "new_independent_audio_seconds": 0, "new_tts_asr_ipa_calls": 0,
           **dict.fromkeys(FALSE_FIELDS, False)}, "projection summary budget or scientific boundary")
    expected_hashes = {name: digest for name, digest in manifest.items() if name != "REPLAY.json"}
    expected_replay = {"schema": "english63-projection-ab-replay-v1", "status": "VERIFIED_SAVED_PROJECTION_AB",
        "files": {name: {"bytes": (root / name).stat().st_size, "sha256": value} for name, value in expected_hashes.items()},
        "files_sha256": expected_hashes, "summary_sha256": manifest["SUMMARY.json"],
        "gradient_tape_sha256": manifest["GRADIENTS.zip"], "model_forward_calls": 0,
        "optimizer_updates": 0, "new_audio_asr_calls": 0, "formal_training_admitted": False,
        "model_promotion_allowed": False}
    core.require(exact(replay, expected_replay), "projection saved receipt manifest/scope differs")
    verified = run_saved_verifier(root, source)
    core.require(exact(verified, expected_replay), "frozen saved decoder did not reproduce the projection receipt")


def validate_terminal_identity(result, terminal):
    run = core.read_json(result / RESULT_DIR / "SUMMARY.json")["public_run"]
    fields(terminal, {"data_commit": run["head"], "run_id": run["run_id"], "run_attempt": run["run_attempt"]},
           "projection terminal public run binding")


def validate_terminal(result, request, digest):
    terminal = core.read_json(result / "TERMINAL.json")
    fields(terminal, {"schema": "kws-research-v4-projection-continuation-terminal-v1", "request_sha256": digest,
           "request_id": REQUEST_ID, "pipeline_commit": request["pipeline_commit"], **WORK_COUNTERS,
           "optimizer_update_budget": UPDATE_BUDGET,
           "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False},
           "projection terminal identity/scope")
    core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest and
                 terminal.get("files_sha256") == core.file_manifest(result), "projection terminal manifest changed")
    core.require(terminal.get("status") == "SUCCEEDED", "prior projection attempt failed; request stays consumed")
    execution_ok(core.read_json(result / "EXECUTION.json"), digest)
    validate_result(result)
    validate_terminal_identity(result, terminal)
    return terminal


def seal(request_path, state):
    request, _, digest = load_request(request_path)
    result = state / "result"
    terminal = {"schema": "kws-research-v4-projection-continuation-terminal-v1", "status": "FAILED",
        "request_sha256": digest, "request_id": REQUEST_ID, "pipeline_commit": request["pipeline_commit"],
        "data_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), **dict.fromkeys(WORK_COUNTERS),
        "optimizer_update_budget": UPDATE_BUDGET,
        "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
        "completed_at": datetime.now(timezone.utc).isoformat()}
    try:
        core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest, "retained projection request changed")
        execution = core.read_json(result / "EXECUTION.json")
        execution_ok(execution, digest)
        validate_result(result)
        validate_terminal_identity(result, terminal)
        terminal.update(status="SUCCEEDED", **WORK_COUNTERS)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        terminal["error"] = type(exc).__name__ + ": " + str(exc)
    terminal["files_sha256"] = core.file_manifest(result)
    core.write_json(result / "TERMINAL.json", terminal)
    core.output(terminal_status=terminal["status"])


def dispatch(args):
    if args.command == "prepare":
        prepare(args.request, args.state)
    elif args.command == "execute":
        core.require(args.pipeline is not None and args.pipeline.is_absolute() and
                     args.python is not None and args.python.is_absolute(), "absolute projection source/Python required")
        execute(args.request, args.state, args.pipeline, args.python)
    elif args.command == "seal":
        seal(args.request, args.state)
    else:
        core.require(args.command == "check", "unsupported projection operation")
        request, _, digest = load_request(args.request)
        result = args.state / ("reused" if (args.state / "reused").exists() else "result")
        terminal = validate_terminal(result, request, digest)
        print(json.dumps({"status": terminal["status"], "request_sha256": digest,
                          "optimizer_updates": UPDATE_BUDGET, "native_forward_frames": FORWARD_FRAMES,
                          "model_promotion_allowed": False}, sort_keys=True))
