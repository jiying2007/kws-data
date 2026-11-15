#!/usr/bin/env python3
"""One fixed paired C pool-access experiment; no training or generic runner."""

from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "research_v4_pool_state_ancestor", Path(__file__).with_name("research_v4_native_c_state.py"))
state_consumer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(state_consumer)
native = state_consumer.native
core = state_consumer.core

EXPERIMENT = "english63-native-c-pool-profile"
ENTRYPOINT = "research/english63_online_c/pool_profile.py"
REQUEST_ID = "en63-native-c-pool-20261115-v1"
SOURCE_REQUEST = Path(__file__).parents[1] / core.REQUEST_DIR / "en63-native-c-state-20261115-v1.json"
ANCESTOR = {
    "run_id": 43663731129,
    "artifact_id": 14141171320,
    "zip_bytes": 1597111,
    "zip_sha256": "2fdd0c951d6cf964d6b73afff5726e7d82b365dae539876822e513d7af0ccaa7",
    "data_commit": "16df89d871f423f8105a22b8bff5556e68685f18",
    "pipeline_commit": "5ed17337558ef1fb48b06f737a2078fcb716a8d3",
    "request_sha256": "5996025f47227d18154fbfa29e7a2bb2fce1322385f53b3d83bebe2e5d6d2e18",
    "terminal_sha256": "00aa4b75d1ca3934992503ac5cd4a698c2ecbc7a48fab51fec660dab70c39e30",
    "header_sha256": "912f443a85fddbc77c49f6cc52c2dbde6c395436d7d8d0818a4e6225cd90fc16",
}
BASELINE_TRACE_SHA = {
    "gru": "aaace50d91252d4a895961c2d9ec4af5d3971a3ce70024ac897bec4c3953a88b",
    "dsc_tcn": "fd449807d7508adb1fc1dddcaa4022b293988a4406ab3ceb3f3fb5fa220650d8",
}
PROFILE = {
    "macro": "EN63_POOL_TIME_MAJOR",
    "variants": {"baseline": 0, "candidate": 1},
    "models": ["gru", "dsc_tcn"],
    "source_pcm_samples": 2101760,
    "frames_per_pass": 13134,
    "feed_chunk_samples": 160,
    "feed_calls_per_pass": 13136,
    "pairs_per_model": {"trace": 1, "warmup": 1, "timed": 6},
    "trace_pair_order": ["baseline", "candidate"],
    "warmup_pair_order": ["baseline", "candidate"],
    "timed_pair_orders": [["baseline", "candidate"], ["candidate", "baseline"]] * 3,
    "full_passes": 32,
    "forward_frames": 420288,
    "supplied_samples_including_replays": 67256320,
    "cpu_affinity": "lowest-allowed-single-cpu",
    "native_group_seconds": 180,
    "native_child_seconds": 30,
    "state_bytes_max": 38192,
    "acceptance": {
        "median_paired_cpu_reduction_min": 0.05,
        "positive_cpu_pairs_min": 5,
        "paired_p99_median_ratio_max": 1.10,
        "exact_data_equality_required": True,
    },
}
FIXED = {
    "schema": "kws-research-v4-native-c-pool-request-v1",
    "experiment": EXPERIMENT,
    "request_id": REQUEST_ID,
    "pipeline_repository": core.PIPELINE_REPOSITORY,
    "entrypoint": ENTRYPOINT,
    "source_artifacts": core.INPUTS,
    "native_state_ancestor": ANCESTOR,
    "profile": PROFILE,
    "optimizer_updates": 0,
    "cpu_threads": 1,
    "max_execution_seconds": 600,
    "max_verification_seconds": 180,
    "new_audio_generated": False,
    "new_tts_asr_ipa_calls": 0,
    "new_independent_audio_seconds": 0,
    "formal_training_admitted": False,
    "human_gold": False,
    "model_promotion_allowed": False,
}
RESULT_FILES = {
    "en63_generated.h", "en63_pool_baseline", "en63_pool_candidate",
    "BUILD.json", "INPUTS.json", "PROTOCOL.json", "SOURCE.zip",
    "SUMMARY.json", "REPLAY.json", "MEASUREMENTS.json", "native.stderr",
} | {kind + "-" + variant + ".trace" for kind in ("gru", "dsc_tcn")
     for variant in ("baseline", "candidate")}
SOURCE_FILES = {
    "research/english63_online_c/pool_profile.py", "research/english63_online_c/pool_profile.c",
    "research/english63_online_c/en63_stream.c", "research/english63_online_c/en63_stream.h",
    "research/english63_online_c/contract.py", "training/english_research_label_bridge.py",
}
FLAGS = ["-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math",
         "-Wall", "-Wextra", "-Werror", "-pedantic"]
PARENT_EVIDENCE = {
    "TERMINAL.json": ANCESTOR["terminal_sha256"],
    "REQUEST.json": ANCESTOR["request_sha256"],
    "state-contracts/REPLAY.json": "0f3fe3bc2057ee37aeeaac16a67a8077658e204295d833dd660576273ed65ae7",
    "state-contracts/INPUTS.json": "f66635d4f6f4d881c778eec3d1d4b5f6670672123bef7bad01704167c82b7425",
}
FALSE_FIELDS = (
    "formal_training_admitted", "human_gold", "model_promotion_allowed", "model_selection_made",
    "threshold_selection", "runtime_default_changed", "acoustic_word_end_labels_created",
    "physical_ssc305_measured", "new_audio_generated", "actual_capture_discontinuity_measured",
)
UNMEASURED = ("independent_english_frr", "continuous_negative_far_per_hour", "word_end_latency", "ssc305_board_profile")


def exact(value, expected):
    """Preserve nested JSON scalar types, including bool/int and int/float."""
    return type(value) is type(expected) and json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        expected, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_request(path):
    path = Path(path)
    raw = path.read_bytes()
    core.require(len(raw) < 16384, "pool request too large")
    request = json.loads(raw, object_pairs_hook=core.unique_object)
    core.require(isinstance(request, dict) and set(request) == set(FIXED) | {
        "pipeline_commit", "reason"}, "pool request keys differ from fixed task")
    for key, expected in FIXED.items():
        core.require(exact(request[key], expected), "unsupported pool request field: " + key)
    core.require(path.name == REQUEST_ID + ".json", "pool request filename differs from fixed campaign")
    core.require(core.valid_sha(request["pipeline_commit"]), "pool source needs a reviewed full SHA")
    core.require(isinstance(request["reason"], str) and 1 <= len(request["reason"].strip()) <= 500,
                 "pool request needs a bounded reason")
    return request, raw, core.sha256(raw)


def validate_ancestor(result):
    request, _, digest = state_consumer.load_request(SOURCE_REQUEST)
    core.require(digest == ANCESTOR["request_sha256"] and
                 request["pipeline_commit"] == ANCESTOR["pipeline_commit"],
                 "consumed state request changed")
    terminal = state_consumer.validate_terminal(result, request, digest)
    core.require(terminal.get("run_id") == str(ANCESTOR["run_id"]) and
                 terminal.get("run_attempt") == "1" and
                 terminal.get("data_commit") == ANCESTOR["data_commit"],
                 "state ancestor was not the pinned successful run")
    core.require(core.sha256((result / "TERMINAL.json").read_bytes()) == ANCESTOR["terminal_sha256"],
                 "state ancestor terminal changed")
    core.require(core.sha256((result / "state-contracts/en63_generated.h").read_bytes()) ==
                 ANCESTOR["header_sha256"], "already exported state C weights changed")


def validate_inputs(state):
    archive = state / "inputs/state.zip"
    core.require(archive.stat().st_size == ANCESTOR["zip_bytes"] and
                 core.sha256(archive.read_bytes()) == ANCESTOR["zip_sha256"],
                 "state ancestor archive changed")
    validate_ancestor(state / "inputs/state-result")
    for key, expected in core.INPUTS.items():
        core.require(core.sha256((state / "inputs" / (key + ".zip")).read_bytes()) ==
                     expected["zip_sha256"], "original pool input changed: " + key)
    core.require(core.sha256((state / "inputs/review.json").read_bytes()) ==
                 core.INPUTS["review"]["member_sha256"], "original review member changed")


def validate_result(result):
    """Read saved pool evidence only; no binary or model is executed here."""
    root = result / "pool-profile"
    core.require(not (root / "TERMINAL.json").exists(), "unexpected nested pool terminal")
    core.require(root.is_dir() and all(path.is_file() and not path.is_symlink() for path in root.iterdir()),
                 "pool result must contain only flat regular files")
    manifest = core.file_manifest(root)
    core.require(set(manifest) == RESULT_FILES, "pool evidence file closure is incomplete or unexpected")
    core.require(all((root / name).stat().st_size > 0 for name in RESULT_FILES if name != "native.stderr"),
                 "empty pool evidence")
    core.require(manifest["en63_generated.h"] == ANCESTOR["header_sha256"], "pool exported header changed")
    expected_outer = {"REQUEST.json", "EXECUTION.json", "execute.log", "verify.log", "runtime.txt"} | {
        "pool-profile/" + name for name in RESULT_FILES}
    core.require(set(core.file_manifest(result)) == expected_outer, "pool outer result closure differs")
    for variant in ("baseline", "candidate"):
        with (root / ("en63_pool_" + variant)).open("rb") as stream:
            core.require(stream.read(4) == b"\x7fELF", "pool native executable missing")
    for kind in ("gru", "dsc_tcn"):
        for variant in ("baseline", "candidate"):
            path = root / (kind + "-" + variant + ".trace")
            with path.open("rb") as stream:
                core.require(path.stat().st_size == 2206528 and
                             stream.read(16) == b"EN63TRC1" + struct.pack("<II", 1, 168),
                             "pool trace missing or truncated")
            core.require(manifest[path.name] == BASELINE_TRACE_SHA[kind],
                         "pool trace differs from retained continuous baseline")
    validate_reports(root, manifest)


def validate_reports(root, manifest):
    summary, replay, protocol, build, inputs, measurements = [
        core.read_json(root / name) for name in (
            "SUMMARY.json", "REPLAY.json", "PROTOCOL.json", "BUILD.json", "INPUTS.json", "MEASUREMENTS.json")]
    core.require(all(type(value) is dict for value in (summary, replay, protocol, build, inputs, measurements)),
                 "pool reports must be objects")
    scope = {**dict.fromkeys(FALSE_FIELDS, False), **dict.fromkeys(UNMEASURED, "NOT_RUN"),
             "new_independent_audio_samples": 0, "new_independent_audio_seconds": 0,
             "optimizer_updates": 0, "new_tts_asr_ipa_calls": 0}
    common = {"native_state_ancestor": ANCESTOR, "source_header_sha256": ANCESTOR["header_sha256"],
              "source_tts_zip_sha256": core.INPUTS["tts"]["zip_sha256"],
              "source_review_json_sha256": core.INPUTS["review"]["member_sha256"],
              "source_weights_sha256": native.CHECKPOINT["weights_sha256"]}
    fields(protocol, common | scope | {"schema": "english63-pool-profile-protocol-v1",
           "experiment": EXPERIMENT, "profile": PROFILE, "pass_order": schedule(),
           "baseline_trace_sha256": BASELINE_TRACE_SHA, "compile_flags": FLAGS,
           "no_benefit_is_valid_completed_experiment": True,
           "no_result_driven_extra_passes_or_retries": True,
           "timed_region": "preloaded-pcm-en63_feed-and-constant-signature-callback",
           "per_feed_quantile": "nearest-rank-ceil(p*N)-1-on-sorted-positive-wall-nanoseconds",
           "resource_scope": "Linux-host-full-feed-chain-with-constant-signature-not-SSC305-or-acoustic-quality"}, "pool protocol")
    fields(summary, common | scope | {"schema": "english63-pool-profile-result-v1",
           "status": "SUCCEEDED", "experiment": EXPERIMENT, "source_training_run_id": 43595338774,
           "source_audio_clips": 63, "source_pcm_samples": 2101760, "unique_input_samples": 2101760,
           "total_supplied_samples": 67256320, "total_frames": 420288, "passes": 32,
           "source_before_after_verified": True, "input_bindings_sha256": manifest["INPUTS.json"],
           "measurements_sha256": manifest["MEASUREMENTS.json"],
           "repeated_diagnostic_audio_is_not_independent_exposure": True,
           "old_wholeclip_models_or_INT8_chunk8_verdicts_changed": False,
           "timed_region": "preloaded-pcm-en63_feed-and-constant-signature-callback",
           "clock_scope": "supplied-PCM-input-dependency-not-acoustic-word-end-or-wall-clock-latency",
           "resource_scope": "Linux-host-full-feed-chain-with-constant-signature-not-SSC305-or-acoustic-quality"}, "pool summary")
    with zipfile.ZipFile(root / "SOURCE.zip") as archive:
        info = archive.infolist()
        core.require(len(info) == len(SOURCE_FILES) and set(archive.namelist()) == SOURCE_FILES and
                     all(not item.is_dir() and item.file_size < 1024 * 1024 and
                         (item.external_attr >> 16) & 0o170000 != 0o120000 for item in info),
                     "pool source archive closure differs")
        source = {name: core.sha256(archive.read(name)) for name in SOURCE_FILES}
    fields(build, {"schema": "english63-pool-native-build-v1", "source_sha256": source,
                  "source_header_sha256": ANCESTOR["header_sha256"], "compile_flags": FLAGS,
                  "physical_ssc305": False}, "pool build")
    fields(summary, {"source_sha256": source}, "pool summary source")
    for key in ("compiler", "compiler_realpath", "source_root", "output_root"):
        core.require(type(build.get(key)) is str and Path(build[key]).is_absolute(), "pool build absolute " + key)
    core.require(core.valid_sha(build.get("compiler_sha256"), 64) and
                 type(build.get("compiler_version")) is str and build["compiler_version"], "pool compiler identity missing")
    run = build.get("public_run")
    core.require(type(run) is dict and set(run) == {"repository", "head", "run_id", "run_attempt"} and
                 run["repository"] == core.REPOSITORY and core.valid_sha(run["head"]) and
                 type(run["run_id"]) is str and run["run_id"].isdigit() and int(run["run_id"]) > 0 and
                 run["run_attempt"] == "1", "pool public run identity differs")
    request_raw = (root.parent / "REQUEST.json").read_bytes()
    request = json.loads(request_raw, object_pairs_hook=core.unique_object)
    digest = core.sha256(request_raw)
    core.require(type(request) is dict and set(request) == set(FIXED) | {"pipeline_commit", "reason"} and
                 all(exact(request[key], value) for key, value in FIXED.items()) and
                 core.valid_sha(request["pipeline_commit"]) and build.get("pipeline_head") == request["pipeline_commit"],
                 "pool saved request/source identity differs")
    for report in (summary, measurements):
        fields(report, {"public_run": run, "pipeline_head": request["pipeline_commit"],
                        "request_sha256": digest}, "pool result execution identity")
    compiles = build.get("compiles")
    core.require(type(compiles) is list and len(compiles) == 2, "pool requires exactly two native builds")
    binaries = {}
    for row, (variant, macro) in zip(compiles, PROFILE["variants"].items()):
        here = Path(build["source_root"]) / "research/english63_online_c"
        command = [build["compiler"], *FLAGS, "-DEN63_POOL_TIME_MAJOR=" + str(macro),
                   "-I", str(here), "-I", build["output_root"], str(here / "en63_stream.c"),
                   str(here / "pool_profile.c"), "-lm", "-o", str(Path(build["output_root"]) / ("en63_pool_" + variant))]
        fields(row, {"variant_name": variant, "variant": macro, "command": command,
                     "exit_code": 0, "binary_sha256": manifest["en63_pool_" + variant]}, "pool fixed compiler command")
        core.require(type(row.get("stdout")) is str and type(row.get("stderr")) is str, "pool compiler output missing")
        binaries[variant] = row["binary_sha256"]
    fields(inputs, common | {"schema": "english63-pool-profile-inputs-v1", "source_audio_clips": 63,
           "source_pcm_samples": 2101760, "unique_input_samples": 2101760, "test_pcm_bytes": 4203520,
           "pcm_operation": "byte-preserving-full63-concatenation", "new_audio_generation_or_resampling": False,
           "request_raw_utf8": request_raw.decode("utf-8"), "request_sha256": digest}, "pool input bindings")
    core.require(core.valid_sha(inputs.get("test_pcm_sha256"), 64), "pool PCM identity missing")
    parent = inputs.get("parent_evidence")
    core.require(type(parent) is dict and set(parent) == set(PARENT_EVIDENCE), "pool parent evidence closure differs")
    parsed = {}
    for name, expected_sha in PARENT_EVIDENCE.items():
        row = parent[name]
        core.require(type(row) is dict and set(row) == {"raw_utf8", "bytes", "sha256"} and
                     type(row["raw_utf8"]) is str, "pool parent evidence record differs")
        raw = row["raw_utf8"].encode("utf-8")
        fields(row, {"bytes": len(raw), "sha256": core.sha256(raw)}, "pool parent raw identity")
        core.require(row["sha256"] == expected_sha, "pool parent original bytes changed")
        parsed[name] = json.loads(raw, object_pairs_hook=core.unique_object)
    expected_rows = [{**row, "test_samples_used": row["source_samples"]}
                     for row in parsed["state-contracts/INPUTS.json"]["rows"]]
    core.require(len(expected_rows) == 63 and sum(row["source_samples"] for row in expected_rows) == 2101760 and
                 exact(inputs.get("rows"), expected_rows), "pool full63 role/source projection changed")
    claim_raw = inputs.get("claim_raw_utf8")
    core.require(type(claim_raw) is str, "pool consumed claim bytes missing")
    claim_raw = claim_raw.encode("utf-8")
    core.require(core.sha256(claim_raw) == inputs.get("claim_sha256") and
                 exact(json.loads(claim_raw, object_pairs_hook=core.unique_object), claim_record(request, digest, run)),
                 "pool consumed claim does not bind this run/request")
    integrity = {"source_before": source, "source_after": source,
                 "header_before": ANCESTOR["header_sha256"], "header_after": ANCESTOR["header_sha256"],
                 "tts_before": core.INPUTS["tts"]["zip_sha256"], "tts_after": core.INPUTS["tts"]["zip_sha256"],
                 "review_before": core.INPUTS["review"]["member_sha256"], "review_after": core.INPUTS["review"]["member_sha256"],
                 "pcm_before": inputs["test_pcm_sha256"], "pcm_after": inputs["test_pcm_sha256"],
                 "parent_terminal_before": ANCESTOR["terminal_sha256"], "parent_terminal_after": ANCESTOR["terminal_sha256"],
                 "parent_files_before": parsed["TERMINAL.json"]["files_sha256"],
                 "parent_files_after": parsed["TERMINAL.json"]["files_sha256"],
                 "binaries_before": binaries, "binaries_after": binaries,
                 "request_before": digest, "request_after": digest,
                 "claim_before": inputs["claim_sha256"], "claim_after": inputs["claim_sha256"]}
    fields(measurements, {"schema": "english63-pool-profile-measurements-v1", "source_integrity": integrity},
           "pool measurements")
    fields(summary, {"source_integrity": integrity}, "pool source before/after")
    validate_measurements(root, measurements, summary, build, binaries)
    manifest = {name: value for name, value in manifest.items() if name != "REPLAY.json"}
    expected_replay = {"schema": "english63-pool-profile-replay-v1", "status": "VERIFIED_SAVED_POOL_PROFILE",
                       "files_sha256": manifest,
                       "files": {name: {"bytes": (root / name).stat().st_size, "sha256": value}
                                 for name, value in manifest.items()},
                       "summary_sha256": manifest["SUMMARY.json"], "model_forward_calls": 0,
                       "optimizer_updates": 0, "new_audio_asr_calls": 0, "formal_training_admitted": False,
                       "model_promotion_allowed": False, "runtime_default_changed": False}
    core.require(exact(replay, expected_replay), "pool saved-only replay manifest or scope differs")


def fields(value, expected, label):
    core.require(type(value) is dict and all(key in value and exact(value[key], wanted)
                                          for key, wanted in expected.items()), label + " differs")


def integer(value, minimum=0, maximum=(1 << 63) - 1):
    return type(value) is int and minimum <= value <= maximum


def schedule():
    result = []
    for kind in PROFILE["models"]:
        pairs = [("trace", 0, PROFILE["trace_pair_order"]), ("warmup", 0, PROFILE["warmup_pair_order"])]
        pairs += [("timed", index, order) for index, order in enumerate(PROFILE["timed_pair_orders"])]
        for phase, pair, order in pairs:
            for variant in order:
                result.append({"run_index": len(result), "kind": kind, "phase": phase, "pair": pair,
                               "variant": PROFILE["variants"][variant], "variant_name": variant})
    return result


def validate_measurements(root, measurements, summary, build, binaries):
    allowed = measurements.get("initial_allowed_cpus")
    core.require(type(allowed) is list and allowed and all(integer(cpu, 0, 1023) for cpu in allowed) and
                 allowed == sorted(set(allowed)) and exact(measurements.get("selected_cpu"), allowed[0]) and
                 exact(measurements.get("final_allowed_cpus"), allowed), "pool single CPU affinity differs")
    elapsed = measurements.get("native_group_elapsed_seconds")
    core.require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 < elapsed <= 180,
                 "pool native group time budget differs")
    fields(summary, {"selected_cpu": allowed[0], "initial_allowed_cpus": allowed,
                     "native_group_elapsed_seconds": elapsed}, "pool summary CPU binding")
    records = measurements.get("passes")
    core.require(type(records) is list and len(records) == 32, "pool exact32 pass roster differs")
    kinds = summary.get("per_kind")
    core.require(type(kinds) is list and len(kinds) == 2 and
                 [row.get("kind") for row in kinds if type(row) is dict] == PROFILE["models"],
                 "pool per-kind roster differs")
    observations = {}
    for row in kinds:
        kind = row["kind"]
        raw = (root / (kind + "-baseline.trace")).read_bytes()
        signature = 14695981039346656037
        for byte in raw[16:]:
            signature = ((signature ^ byte) * 1099511628211) & ((1 << 64) - 1)
        proposals = sum(struct.unpack_from("<I", raw, 16 + index * 168 + 164)[0] != 0 for index in range(13134))
        fields(row, {"source_weights_sha256": native.CHECKPOINT["weights_sha256"][kind], "bit_identity": True,
                     "baseline_trace_sha256": BASELINE_TRACE_SHA[kind], "candidate_trace_sha256": BASELINE_TRACE_SHA[kind],
                     "signature64": format(signature, "016x"), "proposal_count": proposals,
                     "state_bytes": 38192, "state_bytes_increase": 0}, "pool model identity")
        observations[kind] = row
    timed = {kind: [] for kind in PROFILE["models"]}
    pcm_path = None
    for row, plan in zip(records, schedule()):
        fields(row, plan | {"status": "COMPLETED", "exit_code": 0, "timed_out": False,
                           "binary_sha256": binaries[plan["variant_name"]]}, "pool native pass roster/status")
        core.require(type(row.get("raw_stdout")) is str and type(row.get("raw_stderr")) is str,
                     "pool native original output missing")
        lines = row["raw_stdout"].splitlines()
        core.require(len(lines) == 2, "pool native output must contain two JSON records")
        body, lifetime = [json.loads(line, object_pairs_hook=core.unique_object) for line in lines]
        core.require(exact(body, row.get("worker")) and exact(lifetime, row.get("lifetime")),
                     "pool native raw_stdout typed binding differs")
        command = row.get("command")
        core.require(type(command) is list and len(command) == 7 and all(type(value) is str for value in command) and
                     Path(command[4]).is_absolute(), "pool native command shape differs")
        pcm_path = command[4] if pcm_path is None else pcm_path
        expected_command = [str(Path(build["output_root"]) / ("en63_pool_" + plan["variant_name"])), "--supervise",
                            plan["kind"], plan["phase"], pcm_path,
                            str(Path(build["output_root"]) / (plan["kind"] + "-" + plan["variant_name"] + ".trace"))
                            if plan["phase"] == "trace" else "-", str(allowed[0])]
        core.require(exact(command, expected_command), "pool native fixed invocation differs")
        model = observations[plan["kind"]]
        fixed = {"schema": "english63-pool-native-pass-v1", "kind": plan["kind"], "variant": plan["variant"],
                 "phase": plan["phase"], "affinity_cpu": allowed[0], "affinity_single_cpu": True,
                 "pcm_samples": 2101760, "frames": 13134, "feed_calls": 13136,
                 "proposal_count": model["proposal_count"], "signature64": model["signature64"],
                 "state_bytes": 38192, "weight_parameter_bytes": (6531 if plan["kind"] == "gru" else 4803) * 4,
                 "frontend_table_bytes": 3894, "input_pcm_bytes": 4203520,
                 "trace_io_enabled": plan["phase"] == "trace", "eof_padding_samples": 0, "eof_events": 0,
                 "timed_region": "preloaded-pcm-en63_feed-and-constant-signature-callback"}
        fields(body, fixed, "pool native counters/signature")
        core.require(set(body) == set(fixed) | {"cpu_ns", "wall_ns", "feed_ns", "inference_end_high_water_rss_kib"},
                     "pool native worker keys differ")
        fields(lifetime, {"schema": "english63-pool-child-lifetime-v1", "child_exit_code": 0, "child_signaled": False},
               "pool native lifetime exit")
        core.require(set(lifetime) == {"schema", "child_exit_code", "child_signaled", "process_lifetime_peak_rss_kib",
                     "child_user_us", "child_system_us"} and integer(body["inference_end_high_water_rss_kib"], 1) and
                     integer(lifetime["process_lifetime_peak_rss_kib"], body["inference_end_high_water_rss_kib"]) and
                     integer(lifetime["child_user_us"]) and integer(lifetime["child_system_us"]),
                     "pool native lifetime scope differs")
        if plan["phase"] == "timed":
            feed = body["feed_ns"]
            core.require(integer(body["cpu_ns"], 1) and integer(body["wall_ns"], 1, 30 * 1000000000 - 1) and
                         type(feed) is list and len(feed) == 13136 and all(integer(value, 1) for value in feed) and
                         sum(feed) <= body["wall_ns"], "pool timed feed observations differ")
            feed = sorted(feed)
            timed[plan["kind"]].append({"run_index": plan["run_index"], "pair": plan["pair"], "variant": plan["variant"],
                "variant_name": plan["variant_name"], "cpu_ns": body["cpu_ns"], "wall_ns": body["wall_ns"],
                "feed_p50_ns": feed[(50 * len(feed) + 99) // 100 - 1],
                "feed_p95_ns": feed[(95 * len(feed) + 99) // 100 - 1],
                "feed_p99_ns": feed[(99 * len(feed) + 99) // 100 - 1], "feed_max_ns": feed[-1],
                "state_bytes": body["state_bytes"], "inference_end_high_water_rss_kib": body["inference_end_high_water_rss_kib"],
                "process_lifetime_peak_rss_kib": lifetime["process_lifetime_peak_rss_kib"]})
        else:
            fields(body, {"cpu_ns": 0, "wall_ns": 0, "feed_ns": []}, "pool untimed pass has timing evidence")
    core.require((root / "native.stderr").read_text() == "".join(row["raw_stderr"] for row in records),
                 "pool original stderr aggregation differs")
    decisions = []
    for kind in PROFILE["models"]:
        rows = timed[kind]
        pairs = []
        for pair in range(6):
            baseline, candidate = [next(row for row in rows if row["pair"] == pair and row["variant"] == variant)
                                   for variant in (0, 1)]
            pairs.append({"pair": pair, "order": PROFILE["timed_pair_orders"][pair],
                          "baseline_cpu_ns": baseline["cpu_ns"], "candidate_cpu_ns": candidate["cpu_ns"],
                          "baseline_p99_ns": baseline["feed_p99_ns"], "candidate_p99_ns": candidate["feed_p99_ns"],
                          "cpu_reduction": 1 - candidate["cpu_ns"] / baseline["cpu_ns"],
                          "p99_ratio": candidate["feed_p99_ns"] / baseline["feed_p99_ns"]})
        reductions, ratios = [row["cpu_reduction"] for row in pairs], [row["p99_ratio"] for row in pairs]
        median, wins, p99 = statistics.median(reductions), sum(value > 0 for value in reductions), statistics.median(ratios)
        gates = {"median_cpu_reduction": median >= 0.05, "positive_cpu_pairs": wins >= 5,
                 "paired_p99": p99 <= 1.10, "state_not_increased": True, "bit_identity": True}
        decision = "BENEFIT" if all(gates.values()) else "NO_BENEFIT"
        fields(observations[kind], {"timed_passes": rows, "pairs": pairs, "paired_cpu_reductions": reductions,
               "paired_cpu_reduction_median": median, "paired_cpu_reduction_range": [min(reductions), max(reductions)],
               "positive_cpu_pairs": wins, "paired_p99_ratios": ratios, "paired_p99_ratio_median": p99,
               "decision": decision, "gate_observations": gates}, "pool paired decision recomputed from saved native measurements")
        decisions.append(decision)
    core.require(summary.get("decision") == ("BENEFIT" if all(value == "BENEFIT" for value in decisions) else "NO_BENEFIT"),
                 "pool aggregate decision differs")


def execution_ok(execution, digest):
    core.require(isinstance(execution, dict) and
                 execution.get("schema") == "kws-research-v4-native-c-pool-execution-v1" and
                 execution.get("request_sha256") == digest and
                 all(type(execution.get(key)) is int and execution[key] == 0
                     for key in ("execute_exit_code", "verify_exit_code")) and
                 execution.get("source_inputs_before_after_verified") is True and
                 type(execution.get("optimizer_updates")) is int and execution["optimizer_updates"] == 0,
                 "pool execution or source input verification incomplete")


def claim_record(request, digest, run=None):
    run = run or {"run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"]}
    return {
        "schema": "kws-research-v4-native-c-pool-start-claim-v1", "request_sha256": digest,
        "pipeline_commit": request["pipeline_commit"], "source_state_artifact_id": ANCESTOR["artifact_id"],
        "run_id": run["run_id"], "run_attempt": run["run_attempt"],
        "optimizer_updates": 0, "status": "CONSUMED_BEFORE_PAIRED_NATIVE_C_POOL_EXPERIMENT",
    }


def validate_terminal(result, request, digest):
    terminal = core.read_json(result / "TERMINAL.json")
    core.require(isinstance(terminal, dict) and
                 terminal.get("schema") == "kws-research-v4-native-c-pool-terminal-v1" and
                 terminal.get("request_sha256") == digest and
                 terminal.get("request_id") == REQUEST_ID and
                 terminal.get("pipeline_commit") == request["pipeline_commit"],
                 "pool terminal belongs to another request/code")
    core.require(type(terminal.get("optimizer_updates")) is int and terminal["optimizer_updates"] == 0 and
                 all(terminal.get(key) is False for key in (
                     "formal_training_admitted", "human_gold", "model_promotion_allowed")),
                 "pool terminal cannot train/promote")
    core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest and
                 terminal.get("files_sha256") == core.file_manifest(result), "pool terminal manifest changed")
    core.require(terminal.get("status") == "SUCCEEDED",
                 "prior pool attempt failed; consumed request must not auto-retry")
    execution_ok(core.read_json(result / "EXECUTION.json"), digest)
    validate_result(result)
    validate_terminal_identity(result, terminal)
    return terminal


def validate_terminal_identity(result, terminal):
    run = core.read_json(result / "pool-profile/SUMMARY.json")["public_run"]
    fields(terminal, {"data_commit": run["head"], "run_id": run["run_id"],
                      "run_attempt": run["run_attempt"]}, "pool terminal public run binding")


def prepare(request_path, state):
    request, raw, digest = load_request(request_path)
    event = core.read_json(os.environ["GITHUB_EVENT_PATH"])
    core.require(os.environ.get("GITHUB_REPOSITORY") == core.REPOSITORY and
                 os.environ.get("GITHUB_EVENT_NAME") == "push" and
                 os.environ.get("GITHUB_REF") == "refs/heads/main" and
                 event["repository"]["private"] is False, "pool execution is public main only")
    state.mkdir()
    result_name = "research-v4-c-pool-result-" + digest
    claim_name = "research-v4-c-pool-claim-" + digest
    terminals, claims = core.named_artifacts(result_name), core.named_artifacts(claim_name)
    core.require(len(terminals) <= 1 and len(claims) <= 1, "ambiguous pool receipts; stop without repeating")
    if terminals:
        archive = state / "prior-result.zip"
        archive.write_bytes(core.artifact_bytes(terminals[0]))
        core.restore_archive(archive, state / "reused")
        validate_terminal(state / "reused", request, digest)
        core.output(execute=False, request_sha256=digest, result_artifact=result_name,
                    prior_artifact_id=terminals[0]["id"])
        return
    core.require(not claims, "prior pool claim exists without a valid terminal; do not repeat")
    core.require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "pool rerun without retained receipts is blocked")
    for directory in ("inputs", "result", "claim"):
        (state / directory).mkdir()
    (state / "result/REQUEST.json").write_bytes(raw)
    run = core.github_get("/actions/runs/" + str(ANCESTOR["run_id"]))
    core.require(run.get("status") == "completed" and run.get("conclusion") == "success" and
                 exact(run.get("run_attempt"), 1) and run.get("event") == "push" and
                 run.get("head_sha") == ANCESTOR["data_commit"], "state ancestor run identity differs")
    metadata = core.artifact_metadata(ANCESTOR["artifact_id"], ANCESTOR["zip_sha256"])
    core.require(exact(metadata.get("size_in_bytes"), ANCESTOR["zip_bytes"]) and
                 exact(metadata.get("workflow_run", {}).get("id"), ANCESTOR["run_id"]) and
                 metadata.get("workflow_run", {}).get("head_sha") == ANCESTOR["data_commit"],
                 "state ancestor artifact belongs to another run or size")
    archive = state / "inputs/state.zip"
    archive.write_bytes(core.artifact_bytes(metadata))
    core.require(archive.stat().st_size == ANCESTOR["zip_bytes"] and
                 core.sha256(archive.read_bytes()) == ANCESTOR["zip_sha256"], "state ancestor ZIP identity differs")
    core.restore_archive(archive, state / "inputs/state-result")
    validate_ancestor(state / "inputs/state-result")
    for key, specification in core.INPUTS.items():
        metadata = core.artifact_metadata(specification["artifact_id"], specification["zip_sha256"])
        if key == "tts":
            core.require(exact(metadata.get("workflow_run", {}).get("id"), 43242278962) and
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
    validate_inputs(state)
    core.write_json(state / "claim/CLAIM.json", claim_record(request, digest))
    core.output(execute=True, request_sha256=digest, result_artifact=result_name,
                claim_artifact=claim_name, prior_artifact_id="")


def execute(request_path, state, pipeline, python):
    request, _, digest = load_request(request_path)
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "rev-parse", "HEAD"],
                                        text=True).strip() == request["pipeline_commit"],
                 "pool experiment checkout SHA differs")
    core.require(not subprocess.check_output(["git", "-C", str(pipeline), "status", "--porcelain",
                                              "--untracked-files=no"], text=True).strip(),
                 "pool experiment tracked source is modified")
    script = pipeline / ENTRYPOINT
    core.require(script.is_file() and script.resolve().is_relative_to(pipeline.resolve()),
                 "fixed pool entrypoint missing")
    core.require(subprocess.check_output(["git", "-C", str(pipeline), "show",
                                         request["pipeline_commit"] + ":" + ENTRYPOINT]) == script.read_bytes(),
                 "pool entrypoint differs from the reviewed commit")
    core.require(exact(core.read_json(state / "claim/CLAIM.json"), claim_record(request, digest)),
                 "pool start claim differs")
    result = state / "result"
    environment = state_consumer.state_environment(python, state)
    base = [str(python), "-I", "-B", str(script)]
    input_arguments = ["--state-result-dir", str(state / "inputs/state-result"),
                       "--tts-zip", str(state / "inputs/tts.zip"),
                       "--review-json", str(state / "inputs/review.json")]
    execution = {"schema": "kws-research-v4-native-c-pool-execution-v1", "request_sha256": digest,
                 "execute_exit_code": None, "verify_exit_code": None, "optimizer_updates": 0,
                 "source_inputs_before_after_verified": False,
                 "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        validate_inputs(state)
        for phase, argument, seconds in (("execute", "--output-dir", request["max_execution_seconds"]),
                                         ("verify", "--verify-result-dir", request["max_verification_seconds"])):
            with (result / (phase + ".log")).open("xb") as log:
                command = base + (input_arguments if phase == "execute" else [])
                code = native.run_bounded(command + [argument, str(result / "pool-profile")],
                                          pipeline=pipeline, environment=environment, log=log, seconds=seconds)
            execution[phase + "_exit_code"] = code
            core.require(code == 0, phase + " process failed; retain partial pool evidence")
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
        "schema": "kws-research-v4-native-c-pool-terminal-v1", "status": "FAILED",
        "request_sha256": digest, "request_id": REQUEST_ID, "pipeline_commit": request["pipeline_commit"],
        "data_commit": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "optimizer_updates": 0,
        "formal_training_admitted": False, "human_gold": False, "model_promotion_allowed": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        core.require(core.sha256((result / "REQUEST.json").read_bytes()) == digest,
                     "pool retained request changed")
        execution_ok(core.read_json(result / "EXECUTION.json"), digest)
        validate_result(result)
        validate_terminal_identity(result, terminal)
        terminal["status"] = "SUCCEEDED"
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
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
                     "absolute pool experiment checkout and Python required")
        execute(args.request, args.state, args.pipeline, args.python)
    elif args.command == "seal":
        seal(args.request, args.state)
    else:
        core.require(args.command == "check", "unsupported pool operation")
        request, _, digest = load_request(args.request)
        result = args.state / ("reused" if (args.state / "reused").exists() else "result")
        terminal = validate_terminal(result, request, digest)
        print(json.dumps({"status": terminal["status"], "request_sha256": digest,
                          "optimizer_updates": 0, "model_promotion_allowed": False}, sort_keys=True))
