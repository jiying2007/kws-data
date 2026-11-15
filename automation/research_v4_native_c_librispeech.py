#!/usr/bin/env python3
"""One frozen English63 native-C screen on retained non-gold read speech."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "research_v4_librispeech_pool_parent", Path(__file__).with_name("research_v4_native_c_pool.py"))
pool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pool)
core, native, state_consumer = pool.core, pool.native, pool.state_consumer
exact, fields, integer = pool.exact, pool.fields, pool.integer

FIXED = {'schema': 'kws-research-v4-native-c-librispeech-request-v1',
 'experiment': 'english63-native-c-librispeech-screen',
 'request_id': 'en63-native-c-librispeech-20261115-v1',
 'pipeline_repository': 'jiying2007/kws-pipeline',
 'entrypoint': 'research/english63_online_c/librispeech_screen.py',
 'native_state_ancestor': {'run_id': 43663731129,
                           'artifact_id': 14141171320,
                           'zip_bytes': 1597111,
                           'zip_sha256': '2fdd0c951d6cf964d6b73afff5726e7d82b365dae539876822e513d7af0ccaa7',
                           'data_commit': '16df89d871f423f8105a22b8bff5556e68685f18',
                           'pipeline_commit': '5ed17337558ef1fb48b06f737a2078fcb716a8d3',
                           'request_sha256': '5996025f47227d18154fbfa29e7a2bb2fce1322385f53b3d83bebe2e5d6d2e18',
                           'terminal_sha256': '00aa4b75d1ca3934992503ac5cd4a698c2ecbc7a48fab51fec660dab70c39e30',
                           'header_sha256': '912f443a85fddbc77c49f6cc52c2dbde6c395436d7d8d0818a4e6225cd90fc16'},
 'source_artifacts': {'negative_reference': {'run_id': 43538998200,
                                             'artifact_id': 14090383839,
                                             'zip_bytes': 47199,
                                             'zip_sha256': '7f28b4eddbee87435b9dc5b4c44e47a6d6ed200cbf9f4b8688d9b93cabeb4db4',
                                             'member': 'english-natural-read-speech-result.json',
                                             'member_sha256': '788a1b04b9ecb22fb96d4036d19456f460ce3ccecd4da69d53d96fea53b29510'},
                      'corpus': {'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
                                 'bytes': 337926286,
                                 'sha256': '76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3',
                                 'license': 'CC-BY-4.0'}},
 'profile': {'models': ['gru', 'dsc_tcn'],
             'source_clips': 517,
             'source_speakers': 40,
             'source_pcm_samples': 58646164,
             'source_seconds': 3665.38525,
             'order': 'source_member',
             'feed_chunk_samples': 160,
             'frames_per_pass': 366537,
             'feed_calls_per_pass': 366539,
             'final_feed_samples': 84,
             'full_passes': 2,
             'forward_frames': 733074,
             'supplied_samples_including_replays': 117292328,
             'pool_time_major': 0,
             'native_child_seconds': 180,
             'native_group_seconds': 420,
             'state_bytes': 38192,
             'cpu_affinity': 'lowest-allowed-single-cpu',
             'trace_record_bytes': 32},
 'optimizer_updates': 0,
 'cpu_threads': 1,
 'max_execution_seconds': 900,
 'max_verification_seconds': 180,
 'new_audio_generated': False,
 'new_tts_asr_ipa_calls': 0,
 'new_independent_audio_seconds': 0,
 'formal_training_admitted': False,
 'human_gold': False,
 'model_promotion_allowed': False}

EXPERIMENT, ENTRYPOINT, REQUEST_ID = (FIXED[key] for key in ("experiment", "entrypoint", "request_id"))
ANCESTOR, SOURCES, PROFILE = (FIXED[key] for key in ("native_state_ancestor", "source_artifacts", "profile"))
NEGATIVE = SOURCES["negative_reference"]
CORPUS = SOURCES["corpus"]
NEGATIVE_HEAD = "5da87fc5785cfa0b5ad26c451fc366b80e3d491b"
RESULT_DIR = "librispeech-screen"
RESULT_FILES = {
    "en63_generated.h", "en63_librispeech", "SOURCE.zip", "PROTOCOL.json", "INPUTS.json",
    "BUILD.json", "MEASUREMENTS.json", "SUMMARY.json", "REPLAY.json", "gru.trace", "dsc_tcn.trace", "native.stderr",
}
FLAGS = ["-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra", "-Werror", "-pedantic"]
SOURCE_SHA256 = {'research/english63_online_c/contract.py': '5a9c1d496daac3fc07dd80e2423c9a995a30f8da012abfac626ab8e5a9a265b5',
 'research/english63_online_c/en63_stream.c': 'b8cb0926c185d1200d27dae194dd83d27d7f08d83acd49d9526bc363d3cee500',
 'research/english63_online_c/en63_stream.h': '627f18dacede547d6aed9988a745071f4e4f988252d16a4646ebe4a09eeebeb7',
 'research/english63_online_c/librispeech_screen.c': '14a82fbe3383edde9903c0addf6a1a0590c55c1c16ca0d383060ece9835cc2e5',
 'research/english63_online_c/librispeech_screen.py': 'e416b11e501a3f788c3d13a221a9b9747a4e32819e03aaaf3f65cf2dc4da99d2',
 'research/english63_online_c/pool_profile.py': 'cc9485e64e31e3f80d6a2f9b951346e7ebcf3c94b06406f61eebc0dc6d93a577',
 'research/english63_weak_pilot/librispeech_negative_events.py': '2ed629ddd2b8aa3ec3b2dc03e8e6e825e6608fa7719c27e9b253380f7267c32b',
 'research/english63_weak_pilot/pcm_frontend_parity.py': '693a282d490c2d2a7f8468a4ed9ae569f2c8b0dde3bc4714001ed537520a12ae',
 'research/english63_weak_pilot/pretrained_quant_events.py': '97714a6d9cd7fecf1c98d0b0f24a091d4644c4924dacdd564b527ad8b2962348',
 'training/english_research_label_bridge.py': '5e1942dd989f0d9c8737cf1cb05090cba403780a2ca8b58816e05eb25249be2f'}
PIPELINE_COMMIT = 'd55a593449974860a64da004267cbd9ada8f1977'
FALSE_FIELDS = ("formal_training_admitted", "human_gold", "trusted_negative", "model_promotion_allowed",
                "model_selection_made", "threshold_selection", "runtime_default_changed", "new_audio_generated")
NOT_RUN = ("independent_english_frr", "continuous_negative_far_per_hour", "word_end_latency", "ssc305_board_profile")


def load_request(path):
    path = Path(path)
    raw = path.read_bytes()
    core.require(len(raw) < 16384, "read-speech request too large")
    request = json.loads(raw, object_pairs_hook=core.unique_object)
    core.require(type(request) is dict and set(request) == set(FIXED) | {"pipeline_commit", "reason"},
                 "read-speech request keys differ")
    for key, expected in FIXED.items():
        core.require(exact(request[key], expected), "unsupported read-speech request field: " + key)
    core.require(path.name == REQUEST_ID + ".json", "read-speech request filename differs")
    core.require(request["pipeline_commit"] == PIPELINE_COMMIT, "read-speech source differs from the fixed reviewed SHA")
    core.require(type(request["reason"]) is str and 1 <= len(request["reason"].strip()) <= 500,
                 "read-speech request needs a bounded reason")
    return request, raw, core.sha256(raw)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path, size, digest, label):
    core.require(path.is_file() and not path.is_symlink() and path.stat().st_size == size and
                 file_sha256(path) == digest, label + " size/hash changed")


def download_corpus(destination):
    """One unauthenticated fixed-URL bounded transfer; no redirect or retry."""
    temporary = destination.with_suffix(".part")
    core.require(not destination.exists() and not temporary.exists(), "corpus download is create-only")
    request = urllib.request.Request(CORPUS["url"], headers={"User-Agent": "kws-data-research-v4"})
    opener = urllib.request.build_opener(core.NoRedirect)
    started = time.monotonic()
    digest, size = hashlib.sha256(), 0
    with temporary.open("xb") as output:
        with opener.open(request, timeout=30) as response:
            core.require(response.status == 200 and response.geturl() == CORPUS["url"],
                         "corpus response must be the fixed HTTPS object")
            length = response.headers.get("Content-Length")
            core.require(length is None or length == str(CORPUS["bytes"]), "corpus declared length differs")
            while True:
                core.require(time.monotonic() - started <= 300, "corpus download exceeded wall budget")
                chunk = response.read(min(1024 * 1024, CORPUS["bytes"] - size + 1))
                core.require(time.monotonic() - started <= 300, "corpus download exceeded wall budget")
                if not chunk:
                    break
                size += len(chunk)
                core.require(size <= CORPUS["bytes"], "corpus exceeds its fixed byte limit")
                digest.update(chunk)
                output.write(chunk)
    core.require(size == CORPUS["bytes"] and digest.hexdigest() == CORPUS["sha256"],
                 "corpus transfer size/hash differs")
    temporary.rename(destination)


def validate_negative(directory):
    member = directory / NEGATIVE["member"]
    core.require(member.is_file() and not member.is_symlink() and
                 file_sha256(member) == NEGATIVE["member_sha256"], "original negative source evidence changed")


def validate_inputs(state):
    validate_file(state / "inputs/state.zip", ANCESTOR["zip_bytes"], ANCESTOR["zip_sha256"], "state ancestor")
    pool.validate_ancestor(state / "inputs/state-result")
    validate_file(state / "inputs/negative.zip", NEGATIVE["zip_bytes"], NEGATIVE["zip_sha256"], "negative ancestor")
    validate_negative(state / "inputs/negative-result")
    validate_file(state / "inputs/dev-clean.tar.gz", CORPUS["bytes"], CORPUS["sha256"], "original corpus")


def claim_record(request, digest, run=None):
    run = run or {"run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"]}
    return {"schema": "kws-research-v4-native-c-librispeech-start-claim-v1", "request_sha256": digest,
            "pipeline_commit": request["pipeline_commit"], "source_state_artifact_id": ANCESTOR["artifact_id"],
            "source_negative_artifact_id": NEGATIVE["artifact_id"], "run_id": run["run_id"],
            "run_attempt": run["run_attempt"], "optimizer_updates": 0,
            "status": "CONSUMED_BEFORE_NATIVE_C_LIBRISPEECH_SCREEN"}


def execution_ok(execution, digest):
    core.require(type(execution) is dict and execution.get("schema") ==
                 "kws-research-v4-native-c-librispeech-execution-v1" and execution.get("request_sha256") == digest and
                 all(type(execution.get(key)) is int and execution[key] == 0 for key in
                     ("execute_exit_code", "verify_exit_code", "optimizer_updates")) and
                 execution.get("source_inputs_before_after_verified") is True,
                 "read-speech execution or source input verification incomplete")


def verified_source(archive_path):
    """No bytes from the result archive execute before this exact allowlist passes."""
    core.require(len(SOURCE_SHA256) == 10 and ENTRYPOINT in SOURCE_SHA256 and
                 all(core.valid_sha(value, 64) for value in SOURCE_SHA256.values()), "reviewed source closure incomplete")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        core.require(len(infos) == len(SOURCE_SHA256) and set(archive.namelist()) == set(SOURCE_SHA256) and
                     all(not info.is_dir() and 0 < info.file_size < 1024 * 1024 and
                         (info.external_attr >> 16) & 0o170000 != 0o120000 for info in infos),
                     "read-speech source path closure differs")
        source = {name: archive.read(name) for name in SOURCE_SHA256}
    core.require(all(core.sha256(value) == SOURCE_SHA256[name] for name, value in source.items()),
                 "read-speech source bytes differ from the reviewed allowlist")
    return source


def run_saved_verifier(root, source):
    """Only the hash-bound stdlib saved reader runs, with no Actions identity or credentials."""
    with tempfile.TemporaryDirectory(prefix="en63-verified-saved-reader-") as directory:
        temporary = Path(directory)
        tree = temporary / "source"
        tree.mkdir()
        for name, raw in source.items():
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        python = Path(sys.executable).resolve()
        command = [str(python), "-I", "-S", "-B", str(tree / ENTRYPOINT), "--verify-result-dir", str(root.resolve())]
        result = subprocess.run(command, env=core.child_environment(python, temporary), cwd=tree,
                                capture_output=True, text=True, timeout=180, check=False)
        core.require(type(result.returncode) is int and result.returncode == 0 and
                     len(result.stdout) < 65536 and result.stderr == "", "frozen saved-only verifier failed")
        return json.loads(result.stdout, object_pairs_hook=core.unique_object)


def validate_result(result):
    root = result / RESULT_DIR
    core.require(root.is_dir() and not root.is_symlink() and
                 all(path.is_file() and not path.is_symlink() for path in root.iterdir()),
                 "read-speech result must contain flat regular files")
    manifest = core.file_manifest(root)
    core.require(set(manifest) == RESULT_FILES and all((root / name).stat().st_size > 0
                 for name in RESULT_FILES if name != "native.stderr"), "read-speech result file closure differs")
    outer = {"REQUEST.json", "EXECUTION.json", "execute.log", "verify.log", "runtime.txt"} | {
        RESULT_DIR + "/" + name for name in RESULT_FILES}
    core.require(set(core.file_manifest(result)) == outer, "read-speech outer result closure differs")
    core.require(manifest["en63_generated.h"] == ANCESTOR["header_sha256"], "read-speech model header changed")
    with (root / "en63_librispeech").open("rb") as stream:
        core.require(stream.read(4) == b"\x7fELF", "read-speech native binary missing")
    for kind in PROFILE["models"]:
        path = root / (kind + ".trace")
        with path.open("rb") as stream:
            core.require(path.stat().st_size == 16 + 32 * PROFILE["frames_per_pass"] and
                         stream.read(16) == b"EN63LSC1" + struct.pack("<II", 1, 32), "read-speech trace layout differs")
    source = verified_source(root / "SOURCE.zip")
    summary, build, inputs, measurements, replay = [core.read_json(root / name) for name in (
        "SUMMARY.json", "BUILD.json", "INPUTS.json", "MEASUREMENTS.json", "REPLAY.json")]
    request_raw = (result / "REQUEST.json").read_bytes()
    request = json.loads(request_raw, object_pairs_hook=core.unique_object)
    digest = core.sha256(request_raw)
    core.require(type(request) is dict and set(request) == set(FIXED) | {"pipeline_commit", "reason"} and
                 all(exact(request[key], value) for key, value in FIXED.items()) and
                 request["pipeline_commit"] == PIPELINE_COMMIT, "read-speech saved request differs")
    run = build.get("public_run")
    core.require(type(run) is dict and set(run) == {"repository", "head", "run_id", "run_attempt"} and
                 run["repository"] == core.REPOSITORY and core.valid_sha(run["head"]) and
                 type(run["run_id"]) is str and run["run_id"].isdigit() and int(run["run_id"]) > 0 and
                 run["run_attempt"] == "1", "read-speech run identity differs")
    for report in (summary, build, measurements):
        fields(report, {"public_run": run, "pipeline_head": request["pipeline_commit"]}, "read-speech run/source binding")
    fields(inputs, {"schema": "english63-librispeech-inputs-v1", "native_state_ancestor": ANCESTOR,
                   "negative_reference": NEGATIVE, "corpus": CORPUS, "header_sha256": ANCESTOR["header_sha256"],
                   "request_raw_utf8": request_raw.decode("utf-8"), "request_sha256": digest}, "read-speech original inputs")
    claim_raw = inputs.get("claim_raw_utf8")
    core.require(type(claim_raw) is str and core.sha256(claim_raw.encode()) == inputs.get("claim_sha256") and
                 exact(json.loads(claim_raw, object_pairs_hook=core.unique_object), claim_record(request, digest, run)),
                 "read-speech consumed claim identity differs")
    fields(build, {"source_sha256": SOURCE_SHA256, "header_sha256": ANCESTOR["header_sha256"],
                   "binary_sha256": manifest["en63_librispeech"], "exit_code": 0, "physical_ssc305": False},
           "read-speech build identities")
    fields(summary, {"schema": "english63-librispeech-screen-result-v1", "experiment": EXPERIMENT,
           "status": "MEASURED_CONTIGUOUS_READ_SPEECH_PROPOSALS_NOT_GOLD_FAR", "request_sha256": digest,
           "native_state_ancestor": ANCESTOR, "negative_reference": NEGATIVE, "corpus": CORPUS, "profile": PROFILE,
           "source_header_sha256": ANCESTOR["header_sha256"], "source_weights_sha256": native.CHECKPOINT["weights_sha256"],
           "source_sha256": SOURCE_SHA256, "source_manifest_sha256": manifest["INPUTS.json"],
           "measurements_sha256": manifest["MEASUREMENTS.json"], "source_clips": 517, "source_speakers": 40,
           "unique_source_samples": 58646164, "unique_source_seconds": 3665.38525, "passes": 2,
           "total_frames": 733074, "total_supplied_samples": 117292328, "new_independent_audio_seconds": 0,
           "source_reset_calls": 0, "inserted_samples": 0, "eof_padding_samples": 0, "new_tts_asr_ipa_calls": 0,
           "optimizer_updates": 0, "screening_scope": "previously-exposed-read-speech-untrusted-negative-not-household-FAR",
           **dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(NOT_RUN, "NOT_RUN")}, "read-speech summary scope")
    models = summary.get("models")
    core.require(type(models) is list and len(models) == 2 and
                 [row.get("kind") for row in models if type(row) is dict] == PROFILE["models"], "read-speech model roster differs")
    for row in models:
        kind = row["kind"]
        fields(row, {"trace_sha256": manifest[kind + ".trace"], "frames": 366537, "pcm_samples": 58646164,
                     "feed_calls": 366539, "final_feed_samples": 84, "state_bytes": 38192,
                     "source_weights_sha256": native.CHECKPOINT["weights_sha256"][kind],
                     "first_available_sample": 400, "last_available_sample": 58646160,
                     "eof_events": 0, "eof_padding_samples": 0}, "read-speech model counters")
        core.require(integer(row.get("proposal_count")) and type(row.get("events")) is list and
                     row["proposal_count"] == len(row["events"]), "read-speech proposal count differs")
    expected_hashes = {name: digest for name, digest in manifest.items() if name != "REPLAY.json"}
    expected_replay = {"schema": "english63-librispeech-screen-replay-v1", "status": "VERIFIED_SAVED_LIBRISPEECH_SCREEN",
        "files": {name: {"bytes": (root / name).stat().st_size, "sha256": value} for name, value in expected_hashes.items()},
        "files_sha256": expected_hashes, "summary_sha256": manifest["SUMMARY.json"], "model_forward_calls": 0,
        "optimizer_updates": 0, "new_audio_asr_calls": 0, "formal_training_admitted": False,
        "model_promotion_allowed": False, "runtime_default_changed": False}
    core.require(exact(replay, expected_replay), "read-speech saved receipt manifest/scope differs")
    verified = run_saved_verifier(root, source)
    core.require(exact(verified, expected_replay), "frozen saved decoder did not reproduce the retained receipt")


def validate_terminal_identity(result, terminal):
    run = core.read_json(result / RESULT_DIR / "SUMMARY.json")["public_run"]
    fields(terminal, {"data_commit": run["head"], "run_id": run["run_id"], "run_attempt": run["run_attempt"]},
           "read-speech terminal public run binding")


def validate_terminal(result, request, digest):
    terminal = core.read_json(result / "TERMINAL.json")
    fields(terminal, {"schema": "kws-research-v4-native-c-librispeech-terminal-v1", "request_sha256": digest,
           "request_id": REQUEST_ID, "pipeline_commit": request["pipeline_commit"], "optimizer_updates": 0,
           "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False},
           "read-speech terminal identity/scope")
    core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest and
                 terminal.get("files_sha256") == core.file_manifest(result), "read-speech terminal manifest changed")
    core.require(terminal.get("status") == "SUCCEEDED", "prior read-speech attempt failed; request stays consumed")
    execution_ok(core.read_json(result / "EXECUTION.json"), digest)
    validate_result(result)
    validate_terminal_identity(result, terminal)
    return terminal


def prepare(request_path, state):
    request, raw, digest = load_request(request_path)
    event = core.read_json(os.environ["GITHUB_EVENT_PATH"])
    core.require(os.environ.get("GITHUB_REPOSITORY") == core.REPOSITORY and
                 os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF") == "refs/heads/main" and
                 event["repository"]["private"] is False, "read-speech execution is public main only")
    state.mkdir()
    result_name, claim_name = "research-v4-c-librispeech-result-" + digest, "research-v4-c-librispeech-claim-" + digest
    terminals, claims = core.named_artifacts(result_name), core.named_artifacts(claim_name)
    core.require(len(terminals) <= 1 and len(claims) <= 1, "ambiguous read-speech receipts; stop without repeating")
    if terminals:
        archive = state / "prior-result.zip"
        archive.write_bytes(core.artifact_bytes(terminals[0]))
        core.restore_archive(archive, state / "reused")
        validate_terminal(state / "reused", request, digest)
        core.output(execute=False, request_sha256=digest, result_artifact=result_name,
                    prior_artifact_id=terminals[0]["id"])
        return
    core.require(not claims, "prior read-speech claim without valid terminal; do not repeat")
    core.require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "read-speech rerun without receipts is blocked")
    for directory in ("inputs", "result", "claim"):
        (state / directory).mkdir()
    (state / "result/REQUEST.json").write_bytes(raw)
    for name, specification, head, event_name in (
        ("state", ANCESTOR, ANCESTOR["data_commit"], "push"),
        ("negative", NEGATIVE, NEGATIVE_HEAD, "pull_request"),
    ):
        run = core.github_get("/actions/runs/" + str(specification["run_id"]))
        fields(run, {"status": "completed", "conclusion": "success", "run_attempt": 1,
                     "event": event_name, "head_sha": head}, name + " ancestor successful run")
        metadata = core.artifact_metadata(specification["artifact_id"], specification["zip_sha256"])
        fields(metadata, {"size_in_bytes": specification["zip_bytes"]}, name + " archive size")
        fields(metadata.get("workflow_run"), {"id": specification["run_id"], "head_sha": head},
               name + " archive public run")
        archive = state / "inputs" / (name + ".zip")
        archive.write_bytes(core.artifact_bytes(metadata))
        validate_file(archive, specification["zip_bytes"], specification["zip_sha256"], name + " archive")
        core.restore_archive(archive, state / "inputs" / (name + "-result"))
    pool.validate_ancestor(state / "inputs/state-result")
    validate_negative(state / "inputs/negative-result")
    download_corpus(state / "inputs/dev-clean.tar.gz")
    validate_inputs(state)
    core.write_json(state / "claim/CLAIM.json", claim_record(request, digest))
    core.output(execute=True, request_sha256=digest, result_artifact=result_name,
                claim_artifact=claim_name, prior_artifact_id="")


def execute(request_path, state, pipeline, python):
    request, _, digest = load_request(request_path)
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"], text=True).strip() ==
                 request["pipeline_commit"], "read-speech checkout SHA differs")
    core.require(not subprocess.check_output(["git", "-C", str(pipeline), "status", "--porcelain",
                                              "--untracked-files=no"], text=True).strip(), "read-speech tracked source modified")
    validate_checkout_sources(pipeline)
    script = pipeline / ENTRYPOINT
    core.require(script.is_file() and script.resolve().is_relative_to(pipeline.resolve()), "fixed read-speech entrypoint missing")
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "show", request["pipeline_commit"] + ":" + ENTRYPOINT]) ==
                 script.read_bytes(), "read-speech entrypoint differs from reviewed commit")
    core.require(exact(core.read_json(state / "claim/CLAIM.json"), claim_record(request, digest)), "read-speech claim differs")
    result = state / "result"
    environment = state_consumer.state_environment(python, state)
    base = [str(python), "-I", "-B", str(script)]
    inputs = ["--state-result-dir", str(state / "inputs/state-result"),
              "--negative-evidence-dir", str(state / "inputs/negative-result"),
              "--source-tar", str(state / "inputs/dev-clean.tar.gz")]
    execution = {"schema": "kws-research-v4-native-c-librispeech-execution-v1", "request_sha256": digest,
                 "execute_exit_code": None, "verify_exit_code": None, "optimizer_updates": 0,
                 "source_inputs_before_after_verified": False, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        validate_inputs(state)
        for phase, argument, seconds in (("execute", "--output-dir", request["max_execution_seconds"]),
                                         ("verify", "--verify-result-dir", request["max_verification_seconds"])):
            with (result / (phase + ".log")).open("xb") as log:
                command = base + (inputs if phase == "execute" else [])
                code = native.run_bounded(command + [argument, str(result / RESULT_DIR)],
                                          pipeline=pipeline, environment=environment, log=log, seconds=seconds)
            execution[phase + "_exit_code"] = code
            core.require(code == 0, phase + " process failed; retain partial read-speech evidence")
        validate_inputs(state)
        execution["source_inputs_before_after_verified"] = True
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        execution["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        execution["elapsed_seconds"] = round(time.monotonic() - started, 3)
        core.write_json(result / "EXECUTION.json", execution)
    execution_ok(execution, digest)


def validate_checkout_sources(pipeline):
    core.require(len(SOURCE_SHA256) == 10 and ENTRYPOINT in SOURCE_SHA256, "reviewed checkout source closure incomplete")
    for name, expected in SOURCE_SHA256.items():
        path = pipeline / name
        core.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(pipeline.resolve()) and
                     file_sha256(path) == expected, "read-speech checkout differs from reviewed source: " + name)


def seal(request_path, state):
    request, _, digest = load_request(request_path)
    result = state / "result"
    terminal = {"schema": "kws-research-v4-native-c-librispeech-terminal-v1", "status": "FAILED",
        "request_sha256": digest, "request_id": REQUEST_ID, "pipeline_commit": request["pipeline_commit"],
        "data_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "optimizer_updates": 0,
        "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
        "completed_at": datetime.now(timezone.utc).isoformat()}
    try:
        core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest, "retained read-speech request changed")
        execution_ok(core.read_json(result / "EXECUTION.json"), digest)
        validate_result(result)
        validate_terminal_identity(result, terminal)
        terminal["status"] = "SUCCEEDED"
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
                     args.python is not None and args.python.is_absolute(), "absolute read-speech source/Python required")
        execute(args.request, args.state, args.pipeline, args.python)
    elif args.command == "seal":
        seal(args.request, args.state)
    else:
        core.require(args.command == "check", "unsupported read-speech operation")
        request, _, digest = load_request(args.request)
        result = args.state / ("reused" if (args.state / "reused").exists() else "result")
        terminal = validate_terminal(result, request, digest)
        print(json.dumps({"status": terminal["status"], "request_sha256": digest,
                          "optimizer_updates": 0, "model_promotion_allowed": False}, sort_keys=True))
