#!/usr/bin/env python3
"""One fixed new-acoustic-coverage study; default CLI validates source only.

The pinned temporal module supplies only pure evidence/guard utilities. Its
worker and execute entry points are never called. No numerical library is
imported until this module's guarded, single-use public worker begins.
"""
from __future__ import annotations
import argparse
import collections
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import struct
import sys
import time
import zipfile

import english_temporal_quality_once as common
from english_temporal_quality_once import (
    need, sha, identity, read_json, write_json, validate_zip, summarize_case,
    aggregate, supervised_command, validate_head_ends, verify_remote_source,
    research_retention,
)

REPO = common.REPO
SCHEMA = "english-coverage-quality-once-v1"
MAIN_BEFORE = "5521b11f495c5b87d5d2ec0eb2597aa5730ea687"
TRAINING, INCUMBENT = common.TRAINING, common.INCUMBENT
TEMPORAL = {"run_id": 44031666420, "artifact_id": 14305587199,
    "name": "english-temporal-quality-44031666420", "head_sha": "b4982d7a2f39c4b2647a86cbf96d3701a1123b6a",
    "bytes": 17980402,
    "sha256": "6a1e38febf22b76a7cf2227362e51f783d5ad524b2d81b5677c22edc6dccc631"}
COMMON_WORKER_SHA = "15389e1101871352e6e7477c893f1dadf019fd57b572fa1388ec53d1dd8b0988"
TEMPORAL_PINS = {
    "cache/snr5-embeddings.npz": "46b1368d9ecaa7d1dfd349194cb45493e67d1e92e771006ee3b516a54d0aa04c",
    "cache/snr5-roster.json": "d73fa97176bb5044fb3aafe9eb275cc9da1f9ff6f32308b4ba3b430e5919fa9a",
    "evidence/SNR5_SUMMARY.json": "013435a1b4d5090dd28142d3e0cb485866d895a6cd4fb0bde228351ae38b7497",
    "evidence/SNR5_RECEIPTS.json": "9a869f63d331a5680a54086daa5ee7f152f1a170a147b91792086ee71f6307b1",
    "evidence/RESULT.json": "400cdb1c916d54543a2c6dd7c770c17ba5d2f07fa708d20a29b21d2145adc3ff",
}
FALSE_FLAGS, BUNDLE = common.FALSE_FLAGS, common.BUNDLE
RUNTIME = ("weak_embedding_contract.py", "weak_audio_cases.py", "causal_embedding.py", "causal_head.py",
           "portable_head.py", "english_causal_head.c", "english_causal_head.h", "head_replay.c")
FROZEN_DATA_SHA = "c2dfae52eda3a3adda952dfd23a250054540f517d6e6fc516aaf9dd2bc915d88"
C_SOURCE_PINS = {
    "english_causal_head.c": "e1e4a4cffd1758677ef820bd2183ec2ab3ab65d450af4d7eea35f565ebefa551",
    "english_causal_head.h": "557c528c7f2511810816a29768f6003b6a4a723b2fe7d60a2c1fdd353ba51de9",
    "head_replay.c": "36fb7cb1732247bc118a9f5707bae2d642fb6d23192201d535f311faa47a8798",
}
ASSETS = {
    "melspectrogram.onnx": {"bytes": 1087958, "sha256": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"},
    "embedding_model.onnx": {"bytes": 1326578, "sha256": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"},
}
SOURCE_HASH_KEYS = ("worker_sha256", "workflow_sha256", "common_worker_sha256", "data_helper_sha256")


def expected_plan():
    """A single data change; no restart of the closed objective search."""
    plan = common.expected_plan()
    plan.pop("positive_min_operator")
    plan.update({
        "schema": SCHEMA, "main_before": MAIN_BEFORE, "temporal_source": TEMPORAL,
        "snr_db": None,
        "planned_new_conditions": 127, "planned_new_training_conditions": 100,
        "planned_new_development_conditions": 27,
        "planned_maximum_embedding_rows": 4667, "planned_maximum_embedding_calls": 2635,
        "maximum_embedding_calls": 2635, "maximum_onnx_forwards": 5270,
        "maximum_training_embedding_calls": 2066, "maximum_new8_embedding_calls": 569,
        "maximum_head_case_passes": 318, "maximum_head_ready_outputs": 13039,
        "old_condition_head_case_passes": 264, "old_condition_head_ready_outputs": 11847,
        "positive_loss": "NEGATIVE_LOG_MEAN_TARGET_PROBABILITY",
        "planned_reused_snr5_conditions": 77, "eligible_reused_snr5_conditions": 71,
        "reused_snr5_embedding_rows": 2649,
        "evaluation_frontend": "REUSE_ALL_OLD_CACHES_PLUS_NEW8_SUFFIX_ONCE",
        "new_training_frontend_calls": "ONLY_NEW5_NEW15_SUFFIX_WINDOWS_MAX_2066",
        "training_conditions": ["clean", "snr10", "new5", "new15"],
        "evaluation_conditions": ["clean", "snr10", "snr5", "new8"],
        "augmented_condition_cycle": ["snr10", "new5", "new15"],
        "draws": "CLEAN_OLD_STEP_MOD_GROUP_AUG_CONDITION_STEP_MOD_3_INDEX_FLOOR_STEP_OVER_3_MOD_GROUP",
        "noise_binding": "NEW_DOMAIN_SAME_ROLE_DISTINCT_PARENT_TRACKS_OLD10_NEW5_NEW15",
        "new_condition_snr_db": {"new5": 5.0, "new15": 15.0, "new8": 8.0},
        "new_training_minimum_positive_parents_per_condition_class": 2,
        "new_training_minimum_positive_families_per_condition_class": 2,
        "new_training_all_negative_four_families_eight_templates_required": True,
        "prefix_reuse": "FIRST_16_BYTE_IDENTICAL_16000_SAMPLE_WINDOWS_COPIED_FROM_ORIGINAL_CLEAN_CACHE",
        "new8_preparation_after_candidate_frozen": True,
        "snr5_previous_evaluation_exposed": True,
        "failed_temporal_head_used_as_warmstart": False,
        "same_pool_objective_search_remains_closed": True,
        "new_data_coverage_arms": 1,
        "subsequent_same_pool_loss_seed_threshold_search_allowed": False,
        "same_pool_epoch_search_allowed": False,
        "maximum_host_c_head_processes": 1, "maximum_arm_c_head_processes": 1,
        "maximum_c_head_cases_per_backend": 291,
        "maximum_c_head_embedding_rows_per_backend": 16808,
        "maximum_c_head_ready_outputs_per_backend": 12443,
        "c_head_logit_tolerance": 0.0001, "c_head_probability_tolerance": 0.000002,
        "c_head_reference": "SAVED_CANDIDATE_PYTHON_EVALUATION_NO_EXTRA_MODEL_CALLS",
        "new_c_frontend_calls": 0, "ssc305": "NOT_RUN",
    })
    plan["research_retention"].update({"new8_no_previously_correct_case_regression": True,
        "new8_minimum_development_positive_cases_per_keyword": 1,
        "new8_all_27_planned_development_cases_eligible": True})
    return plan


def validate_plan(plan):
    expected = expected_plan()
    need(set(plan) == set(expected) | set(SOURCE_HASH_KEYS), "closed one-shot acoustic coverage plan")
    need(all(type(plan[key]) is type(value) and plan[key] == value for key, value in expected.items()),
         "fixed one-shot acoustic coverage choices")
    for key in SOURCE_HASH_KEYS:
        need(re.fullmatch(r"[0-9a-f]{64}", plan[key]) is not None, "source SHA: " + key)
    need(plan["common_worker_sha256"] == COMMON_WORKER_SHA, "immutable common utilities")


def source_contract(plan):
    here = Path(__file__).resolve()
    paths = {"worker_sha256": here,
        "common_worker_sha256": here.with_name("english_temporal_quality_once.py"),
        "data_helper_sha256": here.with_name("english_coverage_quality_data.py"),
        "workflow_sha256": here.parent.parent / ".github/workflows/english-coverage-quality-once.yml"}
    need(Path(common.__file__).resolve() == paths["common_worker_sha256"], "local immutable common module path")
    for key, path in paths.items():
        need(identity(path)["sha256"] == plan[key], "executed source identity: " + key)
    need(identity(here.with_name("english_temporal_quality_data.py"))["sha256"] == FROZEN_DATA_SHA,
         "transitive immutable data helper")
    return {key: {"path": str(path), **identity(path)} for key, path in paths.items()}


def execution_guard(plan, env=None, event=None):
    env = os.environ if env is None else env
    event = read_json(env["GITHUB_EVENT_PATH"]) if event is None else event
    need(env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_REPOSITORY") == REPO
         and env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == "refs/heads/main"
         and env.get("GITHUB_RUN_ATTEMPT") == "1", "public first-attempt main push required")
    repo = event.get("repository", {})
    need(repo.get("full_name") == REPO and repo.get("private") is False
         and event.get("before") == MAIN_BEFORE and event.get("after") == env.get("GITHUB_SHA")
         and re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")) is not None
         and re.fullmatch(r"[0-9]+", env.get("GITHUB_RUN_ID", "")) is not None,
         "exact new public main successor required")
    source_contract(plan)


def paired_comparison(reference, candidate):
    """Use the same case-level rule; give the new model an unambiguous name."""
    rows = common.paired_comparison(reference, candidate)
    for row in rows:
        row["coverage_research_error"] = row.pop("temporal_research_error")
        row["coverage_events"] = row.pop("temporal_events")
    return rows


def training_draws(groups, step, old_draws):
    """Three unchanged clean slots and three rotating acoustic-condition slots."""
    need(type(step) is int and 0 <= step < 96, "fixed training step")
    condition = ("snr10", "new5", "new15")[step % 3]
    used = []
    for current in ("clean", condition):
        occurrence = step if current == "clean" else step // 3
        for label in range(3):
            group = groups[(current, label)]
            need(group, "all prespecified condition-class groups remain nonempty")
            key = group[occurrence % len(group)]
            draw = {"condition": current, "id": key[1], "weak_class": label, "role": "train"}
            if current == "clean":
                need(draw == old_draws[step][label], "exact original clean draw")
            else:
                draw["condition_occurrence"] = occurrence
            used.append(draw)
    return used


def retention_record(paired, snr5_rows, new8_rows):
    retained = research_retention(paired, snr5_rows)
    criteria = retained["criteria"]
    criteria["new8_no_previously_correct_case_regression"] = not any(
        row["previously_correct_case_regressed"] for row in paired["new8"])
    criteria["new8_both_development_keywords_present"] = all(
        any(r["role"] == "development" and r["weak_class"] == label for r in new8_rows)
        for label in (1, 2))
    criteria["new8_all_27_planned_development_cases_eligible"] = (
        len(new8_rows) == len({row["parent_id"] for row in new8_rows}) == 27
        and collections.Counter((row["role"], row["weak_class"]) for row in new8_rows)
            == {("development", 0): 16, ("development", 1): 3, ("development", 2): 8})
    retained["passed"] = all(criteria.values())
    retained["status"] = "PASSED_RESEARCH_RETENTION_GATE" if retained["passed"] else "FAILED_RESEARCH_RETENTION_GATE_DIAGNOSTIC_ONLY"
    return retained


def compare_c_rows(raw_stdout, reference):
    """Parse saved C output only; fixtures need no compiler or model library."""
    lines = raw_stdout.decode("ascii").splitlines()
    need(len(lines) == len(reference) and reference, "complete C replay row count")
    max_logits = max_probs = 0.0
    events = []
    for line, row in zip(lines, reference):
        parts = line.split()
        need(len(parts) == 9 and parts[1] in ("0", "1") and parts[2] in ("0", "1", "2"), "C replay row shape")
        need((int(parts[0]), bool(int(parts[1])), int(parts[2])) ==
             (row["sample"], row["ready"], row["keyword"]), "C exact clock ready event parity")
        values = [float(value) for value in parts[3:]]
        need(all(math.isfinite(value) for value in values), "finite C replay values")
        max_logits = max(max_logits, max(abs(a - b) for a, b in zip(values[:3], row["logits"])))
        max_probs = max(max_probs, max(abs(a - b) for a, b in zip(values[3:], row["probabilities"])))
        if row["keyword"]:
            events.append({"sample": row["sample"], "keyword": row["keyword"]})
    need(max_logits <= 1e-4 and max_probs <= 2e-6, "original C head numerical tolerances")
    return {"passed": True, "exact_samples": True, "exact_ready": True, "exact_events": True,
        "max_abs_logit_error": max_logits, "max_abs_probability_error": max_probs,
        "input_embeddings": len(reference), "head_outputs": sum(bool(row["ready"]) for row in reference),
        "events": events}


def execute(args, plan):
    execution_guard(plan)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    evidence, downloads = root / "evidence", root / "downloads"
    evidence.mkdir(); downloads.mkdir()
    write_json(evidence / "PLAN.json", plan, create=True)
    write_json(evidence / "STARTED.json", {"schema": SCHEMA, "run_id": os.environ["GITHUB_RUN_ID"],
        "head_sha": os.environ["GITHUB_SHA"], "run_attempt": 1, "parent_pid": os.getpid(),
        "training_source": TRAINING, "incumbent_source": INCUMBENT, "temporal_source": TEMPORAL,
        "retry_count": 0, "numeric_attempt": 1, "same_pool_objective_search_remains_closed": True,
        "deadline_unix": time.time() + 900,
        **dict.fromkeys(FALSE_FLAGS, False)}, create=True)
    started = time.monotonic()

    def command(name, argv, limit=120, auth=False):
        remaining = 900 - (time.monotonic() - started)
        need(remaining > 7, "total execution deadline including cleanup reserve")
        env = dict(os.environ)
        env["KWS_COVERAGE_PARENT_PID"] = str(os.getpid())
        if not auth:
            for key in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"):
                env.pop(key, None)
        out = downloads / (name + ".stdout") if name.endswith("-zip") else evidence / (name + ".stdout")
        supervised_command(argv, out, evidence / (name + ".stderr"), min(limit, remaining - 7), env)
        return out

    try:
        for label, expected in (("training", TRAINING), ("incumbent", INCUMBENT), ("temporal", TEMPORAL)):
            run = read_json(command(label + "-run", ["gh", "api",
                f"repos/{REPO}/actions/runs/{expected['run_id']}"], auth=True))
            artifact = read_json(command(label + "-artifact", ["gh", "api",
                f"repos/{REPO}/actions/artifacts/{expected['artifact_id']}"], auth=True))
            verify_remote_source(run, artifact, expected)
            path = command(label + "-zip", ["gh", "api",
                f"repos/{REPO}/actions/artifacts/{expected['artifact_id']}/zip"], 240, True)
            destination = downloads / (label + ".zip")
            path.rename(destination)
            need(identity(destination) == {key: expected[key] for key in ("bytes", "sha256")},
                 "downloaded immutable source: " + label)
        write_json(evidence / "SOURCE_ACQUIRED.json", {label: identity(downloads / (label + ".zip"))
            for label in ("training", "incumbent", "temporal")}, create=True)
        command("model-worker", [sys.executable, "-B", Path(__file__).resolve(), "--plan", args.plan.resolve(),
                                  "--worker", "--output", root], 900)
        result = read_json(evidence / "RESULT.json")
        need(result.get("status") == "COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH", "worker completion")
        write_json(evidence / "PARENT_RESULT.json", {"status": "COMPLETED_ONE_SUPERVISED_WORKER",
            "worker_result": identity(evidence / "RESULT.json"), "wall_seconds": time.monotonic() - started,
            "retry_count": 0, "numeric_attempt": 1, **dict.fromkeys(FALSE_FLAGS, False)}, create=True)
    except BaseException as exc:
        write_json(evidence / "FAILED_NO_RETRY.json", {"status": "FAILED_NO_RETRY", "error_type": type(exc).__name__,
            "error": str(exc), "wall_seconds": time.monotonic() - started, "retry_count": 0,
            "partial_progress": read_json(evidence / "PROGRESS.json") if (evidence / "PROGRESS.json").exists() else None,
            **dict.fromkeys(FALSE_FLAGS, False)})
        raise
    finally:
        write_json(evidence / "SUPERVISOR_EXIT.json", {"wall_seconds": time.monotonic() - started,
            "maximum_seconds": 900, "retry_count": 0, "numeric_attempt": 1})


def worker(args, plan):
    """The only entry point that imports numerical libraries or executes models."""
    execution_guard(plan)
    root = args.output.resolve()
    evidence = root / "evidence"
    started = read_json(evidence / "STARTED.json")
    need(started["parent_pid"] == os.getppid()
         and os.environ.get("KWS_COVERAGE_PARENT_PID") == str(os.getppid())
         and started["run_id"] == os.environ["GITHUB_RUN_ID"]
         and started["head_sha"] == os.environ["GITHUB_SHA"], "single owning supervisor must launch worker")
    training, incumbent, temporal = (root / ("downloads/" + name + ".zip") for name in ("training", "incumbent", "temporal"))
    need(identity(training) == {k: TRAINING[k] for k in ("bytes", "sha256")}
         and identity(incumbent) == {k: INCUMBENT[k] for k in ("bytes", "sha256")}
         and identity(temporal) == {k: TEMPORAL[k] for k in ("bytes", "sha256")}, "authenticated worker inputs")
    acquired = read_json(evidence / "SOURCE_ACQUIRED.json")
    need(acquired["training"] == identity(training) and acquired["incumbent"] == identity(incumbent)
         and acquired["temporal"] == identity(temporal), "source acquisition precedes worker")
    write_json(evidence / "WORKER_STARTED.json", {"pid": os.getpid(), "parent_pid": os.getppid(),
        "run_id": os.environ["GITHUB_RUN_ID"], "numeric_attempt": 1}, create=True)
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    need(np.__version__ == "1.26.4" and torch.__version__ == "2.9.1+cpu"
         and onnx.__version__ == "1.17.0" and ort.__version__ == "1.20.1", "fixed numeric dependency versions")
    write_json(evidence / "RUNTIME.json", {"python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "numeric_threads": 1, "execution_provider": "CPUExecutionProvider",
        "soundfile_or_corpus_decoder_used": False}, create=True)
    progress = {"new_embedding_calls_started": 0, "new_embedding_calls_completed": 0,
        "onnx_forward_started": 0, "onnx_forward_completed": 0,
        "head_case_passes_started": 0, "head_case_passes_completed": 0, "head_ready_outputs": 0,
        "optimizer_updates_started": 0, "optimizer_updates_completed": 0,
        "torch_forward_calls_started": 0, "torch_forward_calls_completed": 0,
        "new_training_frontend_calls": 0, "new_tts_asr_phone_calls": 0, "retry_count": 0,
        "train_new_embedding_calls_started": 0, "train_new_embedding_calls_completed": 0,
        "new8_embedding_calls_started": 0, "new8_embedding_calls_completed": 0,
        "new8_head_case_passes_started": 0, "new8_head_case_passes_completed": 0,
        "prefix_embedding_rows_reused": 0, "new_c_frontend_calls": 0}
    for backend in ("host", "arm"):
        for kind in ("builds", "head_processes", "head_case_passes", "embedding_rows"):
            for endpoint in ("started", "completed"):
                progress[f"{backend}_c_{kind}_{endpoint}"] = 0
        progress[f"{backend}_c_head_ready_outputs"] = 0
    phases, frozen_sha = [], None

    def checkpoint():
        write_json(evidence / "PROGRESS.json", progress)

    def phase(name):
        phases.append({"phase_number": len(phases) + 1, "phase": name,
            "progress": dict(progress), "candidate_sha256": frozen_sha})
        write_json(evidence / "PHASES.json", phases)

    checkpoint(); phase("WORKER_STARTED")
    for folder in ("sources", "models", "cache", "predictions/reused", "audio/new", "background_tracks", "runtime"):
        (root / folder).mkdir(parents=True, exist_ok=False)
    copied = {}

    def copy_member(archive, member, destination, source_label, pins=None):
        raw = archive.read(member)
        if pins is not None and member in pins:
            need(sha(raw) == pins[member], "copied member pin: " + member)
        path = root / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as out:
            out.write(raw)
        copied[destination] = {"source": source_label, "member": member,
            "bytes": len(raw), "sha256": sha(raw)}
        return raw

    with zipfile.ZipFile(training) as original, zipfile.ZipFile(incumbent) as previous, zipfile.ZipFile(temporal) as frozen_condition:
        validate_zip(original); validate_zip(previous); validate_zip(frozen_condition)
        roster = json.loads(copy_member(original, "STREAM_ROSTER.json", "sources/STREAM_ROSTER.json", "training", common.PINS))
        feature_evidence = json.loads(copy_member(original, "FEATURE_EVIDENCE.json", "sources/FEATURE_EVIDENCE.json", "training", common.PINS))
        for name in ("FEATURE_PREFLIGHT.json", "PER_CASE_EVENTS.json", "LIBRISPEECH_SOURCES.json", "SELECTION.json"):
            copy_member(original, name, "sources/" + name, "training", common.PINS)
        cache_raw = copy_member(original, "actual-embedding-streams.npz", "cache/clean-original.npz", "training", common.PINS)
        baseline_raw = copy_member(original, "train/causal-head.npz", "models/baseline-head.npz", "training", common.PINS)
        need(previous.read("cache/clean-original.npz") == cache_raw
             and previous.read("models/baseline-head.npz") == baseline_raw,
             "same original baseline and clean cache in both authenticated artifacts")
        for name in RUNTIME:
            raw = copy_member(original, BUNDLE + "runtime/" + name, "sources/" + name, "training")
            if name in C_SOURCE_PINS:
                need(sha(raw) == C_SOURCE_PINS[name], "immutable original C runtime source")
        for name, expected in ASSETS.items():
            copy_member(original, BUNDLE + "models/" + name, "models/" + name, "training")
            need(identity(root / "models" / name) == expected, "unchanged frozen feature asset")
        for name in ("kws-pipeline-LICENSE", "openWakeWord-LICENSE"):
            copy_member(original, BUNDLE + "licenses/" + name, "sources/" + name, "training")
        background_manifest = json.loads(copy_member(previous, "background_tracks/MANIFEST.json",
                                                     "background_tracks/MANIFEST.json", "incumbent"))
        tracks = {}
        for name, expected in background_manifest["files"].items():
            need(PurePosixPath(name).name == name and name.endswith(".pcm"), "saved background leaf")
            raw = copy_member(previous, "background_tracks/" + name, "background_tracks/" + name, "incumbent")
            need(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "saved background identity")
            tracks[name[:-4]] = raw
        previous_training = json.loads(copy_member(previous, "evidence/TRAINING.json", "sources/PREVIOUS_TRAINING.json",
                                                   "incumbent", common.INCUMBENT_PINS))
        copy_member(previous, "evidence/RESULT.json", "sources/PREVIOUS_RESULT.json", "incumbent")
        copy_member(previous, "evidence/RUNTIME.json", "sources/PREVIOUS_RUNTIME.json", "incumbent")
        copy_member(previous, "evidence/MIX_ROSTER.json", "sources/PREVIOUS_MIX_RECEIPTS.json", "incumbent")
        copy_member(previous, "models/candidate-head.npz", "models/aug96-control-head.npz", "incumbent", common.INCUMBENT_PINS)
        mixed_raw = copy_member(previous, "cache/mixed-embeddings.npz", "cache/mixed-snr10.npz", "incumbent", common.INCUMBENT_PINS)
        snr10_roster = json.loads(copy_member(previous, "cache/mixed-roster.json", "cache/snr10-roster.json",
                                            "incumbent", common.INCUMBENT_PINS))
        for name in previous.namelist():
            if name.startswith("predictions/"):
                copy_member(previous, name, "predictions/reused/" + PurePosixPath(name).name, "incumbent")
        snr5_raw = copy_member(frozen_condition, "cache/snr5-embeddings.npz", "cache/snr5-embeddings.npz", "temporal", TEMPORAL_PINS)
        snr5_roster = json.loads(copy_member(frozen_condition, "cache/snr5-roster.json", "cache/snr5-roster.json", "temporal", TEMPORAL_PINS))
        snr5_summary = json.loads(copy_member(frozen_condition, "evidence/SNR5_SUMMARY.json", "evidence/SNR5_SUMMARY.json", "temporal", TEMPORAL_PINS))
        snr5_receipts = json.loads(copy_member(frozen_condition, "evidence/SNR5_RECEIPTS.json", "evidence/SNR5_RECEIPTS.json", "temporal", TEMPORAL_PINS))
        previous_temporal = json.loads(copy_member(frozen_condition, "evidence/RESULT.json", "sources/PREVIOUS_TEMPORAL_RESULT.json", "temporal", TEMPORAL_PINS))
        need(previous_temporal["status"] == "COMPLETED_WEAK_TEMPORAL_QUALITY_RESEARCH"
             and previous_temporal["research_retention"]["passed"] is False
             and previous_temporal["run_id"] == str(TEMPORAL["run_id"])
             and previous_temporal["head_sha"] == TEMPORAL["head_sha"], "completed rejected temporal arm is only a cache source")
        for stem in ("baseline-snr5", "aug96-snr5"):
            for suffix in ("-frames.jsonl", "-events.json"):
                name = "predictions/" + stem + suffix
                copy_member(frozen_condition, name, "predictions/reused/" + stem + suffix, "temporal")
        need(frozen_condition.read("models/baseline-head.npz") == baseline_raw
             and frozen_condition.read("models/aug96-control-head.npz") == previous.read("models/candidate-head.npz")
             and frozen_condition.read("cache/clean-original.npz") == cache_raw
             and frozen_condition.read("cache/mixed-snr10.npz") == mixed_raw,
             "third artifact shares exact original baseline/control/clean/noisy feature identities")
    here = Path(__file__).resolve()
    for source, destination in ((here, "sources/english_coverage_quality_once.py"),
            (here.with_name("english_temporal_quality_once.py"), "sources/english_temporal_quality_once.py"),
            (here.with_name("english_coverage_quality_data.py"), "sources/english_coverage_quality_data.py"),
            (here.with_name("english_temporal_quality_data.py"), "sources/english_temporal_quality_data.py"),
            (here.parent.parent / ".github/workflows/english-coverage-quality-once.yml", "sources/english-coverage-quality-once.yml")):
        (root / destination).write_bytes(source.read_bytes())
    write_json(evidence / "INPUTS.json", {"training_source": TRAINING, "incumbent_source": INCUMBENT,
        "temporal_source": TEMPORAL, "actual_training": identity(training), "actual_incumbent": identity(incumbent),
        "actual_temporal": identity(temporal), "source_contract": source_contract(plan), "copied_members": copied,
        "old_embedding_rows_reused": {"clean": 10354, "snr10": 2804, "snr5": 2649},
        "old_frontend_recomputation": False, "new_training_frontend_calls": "NEW_SUFFIX_ONLY",
        "new_evaluation_frontend_calls": "NEW8_SUFFIX_ONLY_AFTER_FREEZE", "failed_temporal_head_used_as_warmstart": False}, create=True)
    sys.path.insert(0, str(root / "sources"))
    import portable_head as portable
    import weak_embedding_contract as contract
    need(contract.CLASSES == {"background": 0, "Hey Vosen": 1, "Hey Mero": 2}, "original exact class map")
    weights = portable.load_weights(root / "models/baseline-head.npz")
    control = portable.load_weights(root / "models/aug96-control-head.npz")
    need(sha(portable.export_blob(weights)) == "acbf6bd314ab8b458ae2175b6268b306ed0102d7ffc68c1b6ffde87af1c41174",
         "original baseline export identity")
    by_feature, by_roster = ({r["id"]: r for r in feature_evidence}, {r["id"]: r for r in roster})
    need(len(roster) == len(by_feature) == len(by_roster) == 117 and set(by_feature) == set(by_roster),
         "fixed complete original clean roster")
    with np.load(io.BytesIO(cache_raw), allow_pickle=False) as saved:
        need(set(saved.files) == {"embeddings", "samples", "discontinuity_before"}, "original cache keys")
        embs, clocks, gaps = saved["embeddings"].copy(), saved["samples"].copy(), saved["discontinuity_before"].copy()
    need(embs.dtype == np.float32 and embs.shape == (10354, 96) and bool(np.isfinite(embs).all())
         and clocks.shape == gaps.shape == (10354,) and clocks.dtype == np.uint64 and gaps.dtype == np.bool_,
         "original clean cache shape and types")
    clean_streams, cursor = {}, 0
    for row in roster:
        begin, count = row["first_record"], row["records"]
        need(begin == cursor and row["source"] == by_feature[row["id"]]["source"], "original contiguous roster binding")
        end = begin + count
        local = [int(s) - row["offset"] for s in clocks[begin:end]]
        need(local == list(range(16000, row["observed_samples"] + 1, 1280))
             and gaps[begin:end].tolist() == [True] + [False] * (count - 1), "clean causal clocks and resets")
        validate_head_ends(row, by_feature[row["id"]]["head_instance_ends"], local)
        clean_streams[row["id"]] = (embs[begin:end].copy(), local)
        cursor = end
    need(cursor == 10354, "no unbound clean cache rows")
    need(len(snr10_roster) == len({r["id"] for r in snr10_roster}) == 76, "unchanged 76 eligible 10 dB conditions")
    snr10_streams, mixed_count = {}, 0
    with np.load(io.BytesIO(mixed_raw), allow_pickle=False) as saved:
        need(set(saved.files) == {f"{kind}{i:03d}" for i in range(76) for kind in ("e", "t")}, "exact old noisy cache keys")
        for i, row in enumerate(snr10_roster):
            need(row["id"] in by_roster and all(row[k] == v for k, v in by_roster[row["id"]].items()),
                 "unchanged 10 dB source identity and role")
            matrix, times = saved[f"e{i:03d}"].copy(), saved[f"t{i:03d}"].copy()
            local = [int(t) for t in times]
            need(times.dtype == np.uint64 and times.ndim == 1
                 and local == list(range(16000, row["observed_samples"] + 1, 1280))
                 and matrix.dtype == np.float32 and matrix.shape == (len(local), 96)
                 and bool(np.isfinite(matrix).all()), "old noisy cache causal geometry")
            validate_head_ends(row, by_feature[row["id"]]["head_instance_ends"], local)
            snr10_streams[row["id"]] = (matrix, local)
            mixed_count += len(local)
    need(mixed_count == 2804, "unchanged noisy embedding row count")
    need(len(snr5_roster) == len({row["id"] for row in snr5_roster}) == 71
         and len(snr5_receipts) == 77 and snr5_summary["eligible_embedding_calls"] == 2649
         and snr5_summary["condition_training_allowed"] is False, "unchanged complete frozen 5 dB evaluation subset")
    snr5_streams, snr5_count = {}, 0
    with np.load(io.BytesIO(snr5_raw), allow_pickle=False) as saved:
        need(set(saved.files) == {f"{kind}{i:03d}" for i in range(71) for kind in ("e", "t")}, "exact reused 5 dB cache keys")
        for index, row in enumerate(snr5_roster):
            need(row["id"] in by_roster and all(row[key] == value for key, value in by_roster[row["id"]].items()),
                 "unchanged 5 dB source identity, role and full-clip geometry")
            matrix, times = saved[f"e{index:03d}"].copy(), saved[f"t{index:03d}"].copy()
            local = [int(t) for t in times]
            need(times.dtype == np.uint64 and times.ndim == 1
                 and local == list(range(16000, row["observed_samples"] + 1, 1280))
                 and matrix.dtype == np.float32 and matrix.shape == (len(local), 96)
                 and bool(np.isfinite(matrix).all()), "reused 5 dB cache causal geometry")
            validate_head_ends(row, by_feature[row["id"]]["head_instance_ends"], local)
            snr5_streams[row["id"]] = (matrix, local)
            snr5_count += len(local)
    need(snr5_count == 2649, "exact previously computed 5 dB embedding row count")
    previous_draws = previous_training["draws"]
    need(previous_training["optimizer_updates"] == 96 and len(previous_draws) == 96
         and all(len(draw) == 6 for draw in previous_draws)
         and previous_training["warmstart"] == identity(root / "models/baseline-head.npz")
         and previous_training["candidate"] == identity(root / "models/aug96-control-head.npz")
         and previous_training["development_checkpoint_or_threshold_selection"] is False,
         "matched fixed-budget prior control")
    spec = importlib.util.spec_from_file_location("fixed_coverage_data", here.with_name("english_coverage_quality_data.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    import causal_embedding as embedding_runtime
    embedding = embedding_runtime.FrozenOnnxEmbedding(root / "models")
    need(embedding.geometry == read_json(root / "sources/FEATURE_PREFLIGHT.json")["geometry"],
         "identical frozen frontend geometry")
    write_json(evidence / "FEATURE_GEOMETRY.json", embedding.geometry, create=True)

    class CountedSession:
        def __init__(self, session):
            self.session = session

        def run(self, *values, **kwargs):
            need(progress["onnx_forward_started"] < 5270, "one fixed ONNX budget")
            progress["onnx_forward_started"] += 1; checkpoint()
            answer = self.session.run(*values, **kwargs)
            progress["onnx_forward_completed"] += 1; checkpoint()
            return answer

    embedding.mel, embedding.embedding = CountedSession(embedding.mel), CountedSession(embedding.embedding)
    feature_reuse = []

    def prepare_new(purpose):
        train = purpose == "train"
        tag, cache_tag = ("TRAIN_MIX", "train-new") if train else ("NEW8", "new8")
        group_counter = "train_new" if train else "new8"
        maximum = 2066 if train else 569
        if not train:
            need(frozen_sha is not None and identity(root / "models/coverage-head.npz")["sha256"] == frozen_sha,
                 "freeze precedes all new8 construction and model evaluation")
        phase(tag + "_DATA_PREPARATION_STARTED")
        summary, receipts, rows, pcm_by_key = helper.build_conditions(training, incumbent, purpose=purpose)
        need(len(receipts) == (100 if train else 27)
             and len(rows) == len({row["id"] for row in rows})
             and {row["id"] for row in rows} == {r["id"] for r in receipts if r["status"] == "MIXED"}
             and set(pcm_by_key) == {row["id"] for row in rows}, "complete fixed construction denominator")
        required = sum(row["records"] - 16 for row in rows)
        if train:
            need(summary["training_coverage"]["passed"] is True, "fixed train source and family coverage minimum")
        need(summary["new_embedding_calls_required"] == required <= maximum
             and summary["eligible_embedding_rows"] == sum(row["records"] for row in rows)
             and summary["prefix_cache_rows_reused"] == 16 * len(rows), "bounded fresh suffix count")
        write_json(evidence / (tag + "_SUMMARY.json"), summary, create=True)
        write_json(evidence / (tag + "_RECEIPTS.json"), receipts, create=True)
        write_json(root / "cache" / (cache_tag + "-roster.json"), rows, create=True)
        for row in rows:
            need(row["id"] == row["condition_id"] and row["parent_id"] in by_roster
                 and row["condition"] in (("new5", "new15") if train else ("new8",)), "new condition and parent identity")
            parent = by_roster[row["parent_id"]]
            need(all(row[key] == value for key, value in parent.items() if key != "id")
                 and row["role"] == ("train" if train else "development")
                 and row["mix"]["condition_training_allowed"] is train,
                 "no role label geometry or parent source mutation")
            raw = pcm_by_key[row["id"]]
            need(len(raw) == 2 * row["observed_samples"]
                 and sha(raw) == row["mix"]["combined_mixed_pcm_sha256"], "complete new observed PCM identity")
            (root / "audio/new" / (row["id"] + ".pcm")).write_bytes(raw)
        phase(tag + "_DATA_PREPARATION_COMPLETED")
        phase(tag + "_FEATURES_STARTED")
        streams, arrays = {}, {}
        for index, row in enumerate(rows):
            parent, raw = by_roster[row["parent_id"]], pcm_by_key[row["id"]]
            source = parent["source"]
            prefix_start = source["prefix_start_sample"]
            prefix = tracks[source["prefix_track"]][prefix_start * 2:(prefix_start + 35200) * 2]
            need(source["prefix_samples"] == source["clip_start_sample"] == 35200
                 and source["prefix_role"] == parent["role"]
                 and len(prefix) == 70400 and prefix == raw[:70400], "byte-identical complete original real prefix")
            windows = []
            for end in range(16000, 35201, 1280):
                original_window, new_window = prefix[2 * (end - 16000):2 * end], raw[2 * (end - 16000):2 * end]
                need(len(original_window) == 32000 and original_window == new_window,
                     "complete frozen stateless frontend input equality")
                windows.append({"available_sample": end, "input_pcm_bytes": 32000,
                                "input_pcm_sha256": sha(original_window)})
            need(windows == row["mix"]["prefix_input_windows"], "independent worker prefix-window check")
            original_vectors, original_clocks = clean_streams[row["parent_id"]]
            samples = list(range(16000, row["observed_samples"] + 1, 1280))
            need(len(samples) == row["records"] and samples[:16] == original_clocks[:16]
                 and samples[15] == 35200, "exact sixteen reusable frontend clocks")
            prefix_vectors = original_vectors[:16].copy()
            need(prefix_vectors.shape == (16, 96)
                 and prefix_vectors.tobytes(order="C") == original_vectors[:16].tobytes(order="C"),
                 "copied prefix embeddings preserve every float bit")
            values = np.frombuffer(raw, dtype="<i2")
            suffix = []
            for end in samples[16:]:
                need(progress["new_embedding_calls_started"] < 2635
                     and progress[group_counter + "_embedding_calls_started"] < required,
                     "prespecified purpose and total frontend budget")
                progress["new_embedding_calls_started"] += 1
                progress[group_counter + "_embedding_calls_started"] += 1
                if train:
                    progress["new_training_frontend_calls"] += 1
                checkpoint()
                suffix.append(embedding(values[end - 16000:end]))
                progress["new_embedding_calls_completed"] += 1
                progress[group_counter + "_embedding_calls_completed"] += 1; checkpoint()
            vectors = np.concatenate((prefix_vectors, np.asarray(suffix, dtype=np.float32)), axis=0)
            need(vectors.shape == (len(samples), 96) and bool(np.isfinite(vectors).all()), "new cache finite complete geometry")
            validate_head_ends(row, by_feature[row["parent_id"]]["head_instance_ends"], samples)
            streams[row["id"]] = (vectors, samples)
            arrays[f"e{index:03d}"] = vectors
            arrays[f"t{index:03d}"] = np.asarray(samples, dtype=np.uint64)
            feature_reuse.append({"condition_id": row["id"], "parent_id": row["parent_id"],
                "condition": row["condition"], "purpose": purpose, "copied_prefix_rows": 16,
                "clean_cache_row_span": [parent["first_record"], parent["first_record"] + 16],
                "prefix_input_windows": windows, "prefix_embedding_bytes": 6144,
                "prefix_embedding_sha256": sha(prefix_vectors.tobytes(order="C")),
                "new_suffix_rows": len(samples) - 16, "samples": samples, "ready_clocks": samples[15:]})
            progress["prefix_embedding_rows_reused"] += 16; checkpoint()
        np.savez(root / "cache" / (cache_tag + "-embeddings.npz"), **arrays)
        write_json(evidence / "FEATURE_REUSE.json", feature_reuse)
        need(progress[group_counter + "_embedding_calls_started"] ==
             progress[group_counter + "_embedding_calls_completed"] == required,
             "one suffix frontend pass per eligible window")
        phase(tag + "_FEATURES_COMPLETED")
        return summary, receipts, rows, streams

    train_summary, train_receipts, new_train_roster, new_train_streams = prepare_new("train")
    groups = {(condition, label): [] for condition in ("clean", "snr10", "new5", "new15") for label in range(3)}
    prepared, train_geometry = {}, []
    for fixed_condition, rows, streams in (("clean", roster, clean_streams), ("snr10", snr10_roster, snr10_streams),
                                           (None, new_train_roster, new_train_streams)):
        for row in rows:
            if row["role"] != "train":
                continue
            condition = fixed_condition or row["condition"]
            parent_id = row.get("parent_id", row["id"])
            matrix, samples = streams[row["id"]]
            ends = by_feature[parent_id]["head_instance_ends"]
            positions = validate_head_ends(row, ends, samples)
            windows = np.asarray([matrix[positions[t] - 15:positions[t] + 1] for t in ends], dtype=np.float32)
            normalized = ((windows - weights["mean96"]) / weights["scale96"]).reshape(len(windows), 1536)
            key = (condition, row["id"])
            prepared[key] = torch.from_numpy(normalized)
            groups[(condition, row["weak_class"])].append(key)
            train_geometry.append({"condition": condition, "id": row["id"], "parent_id": parent_id,
                "role": "train", "weak_class": row["weak_class"], "head_instance_ends": ends,
                "triplet_count": len(ends) - 2 if row["weak_class"] else None})
    need(all(groups.values()) and len(prepared) == sum(len(group) for group in groups.values()),
         "all twelve fixed train-only condition-class groups")
    write_json(evidence / "TRAINING_GEOMETRY.json", train_geometry, create=True)
    group_record = {condition + "/" + str(label): [key[1] for key in group]
                    for (condition, label), group in groups.items()}
    phase("TRAINING_STARTED")
    torch.set_num_threads(1); torch.manual_seed(20261117); torch.use_deterministic_algorithms(True)
    model = torch.nn.Sequential(torch.nn.Linear(1536, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
    need(sum(p.numel() for p in model.parameters()) == 49283, "unchanged head parameter count")
    with torch.no_grad():
        for parameter, key in ((model[0].weight, "w1"), (model[0].bias, "b1"), (model[2].weight, "w2"), (model[2].bias, "b2")):
            parameter.copy_(torch.from_numpy(weights[key]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    draws, losses = [], []
    for step in range(96):
        need(progress["optimizer_updates_started"] < 96, "optimizer budget")
        progress["optimizer_updates_started"] += 1; checkpoint()
        optimizer.zero_grad(set_to_none=True)
        used = training_draws(groups, step, previous_draws)
        terms = []
        for draw in used:
            condition, label = draw["condition"], draw["weak_class"]
            key = (condition, draw["id"])
            need(key in prepared and condition != "new8" and draw["role"] == "train", "training data boundary")
            need(progress["torch_forward_calls_started"] < 576, "training bag forward budget")
            progress["torch_forward_calls_started"] += 1; checkpoint()
            logp = torch.log_softmax(model(prepared[key]), dim=-1)
            progress["torch_forward_calls_completed"] += 1; checkpoint()
            if label == 0:
                terms.append(-logp[:, 0].mean())
            else:
                terms.append(math.log(len(logp)) - torch.logsumexp(logp[:, label], dim=0))
        need(len(terms) == 6 and used[:3] == previous_draws[step][:3], "six equal bags with unchanged clean anchors")
        loss = torch.stack(terms).mean()
        need(bool(torch.isfinite(loss)), "finite original-objective coverage loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        losses.append(float(loss.detach())); draws.append(used)
        progress["optimizer_updates_completed"] += 1; checkpoint()
    need(progress["torch_forward_calls_completed"] == 576 and len(draws) == 96, "fixed training completed exactly once")
    candidate = {"w1": model[0].weight.detach().numpy(), "b1": model[0].bias.detach().numpy(),
        "w2": model[2].weight.detach().numpy(), "b2": model[2].bias.detach().numpy(),
        "mean96": weights["mean96"].copy(), "scale96": weights["scale96"].copy()}
    np.savez(root / "models/coverage-head.npz", **candidate)
    candidate = portable.load_weights(root / "models/coverage-head.npz")
    normalization = {}
    for key in ("mean96", "scale96"):
        old, new = weights[key].tobytes(order="C"), candidate[key].tobytes(order="C")
        need(old == new == control[key].tobytes(order="C"), "byte-identical frozen train-only normalization")
        normalization[key] = {"bytes": len(old), "sha256": sha(old), "candidate_sha256": sha(new), "byte_identical": True}
    (root / "models/coverage-head.bin").write_bytes(portable.export_blob(candidate))
    frozen_sha = identity(root / "models/coverage-head.npz")["sha256"]
    phase("CANDIDATE_FROZEN")
    write_json(evidence / "CANDIDATE_FROZEN.json", {"phase_number": len(phases),
        "candidate": identity(root / "models/coverage-head.npz"), "candidate_bin": identity(root / "models/coverage-head.bin"),
        "normalization": normalization, "progress": dict(progress), "new8_model_calls_before_freeze": 0,
        "new8_preparation_before_freeze": False, "all_future_evaluation_uses_this_frozen_head": True}, create=True)
    write_json(evidence / "TRAINING.json", {"optimizer_updates": 96, "training_bag_forwards": 576,
        "warmstart": identity(root / "models/baseline-head.npz"), "control": identity(root / "models/aug96-control-head.npz"),
        "candidate": identity(root / "models/coverage-head.npz"), "losses": losses, "draws": draws,
        "clean_draws_exact_match": True, "eligible_groups": group_record,
        "augmented_condition_cycle": ["snr10", "new5", "new15"], "augmented_occurrence_index": "step//3",
        "previous_training": identity(root / "sources/PREVIOUS_TRAINING.json"),
        "normalization": normalization, "positive_loss": plan["positive_loss"],
        "background_loss": plan["background_loss"], "bag_aggregation": plan["bag_aggregation"],
        "comparison": "ONLY_TRAINING_ACOUSTIC_COVERAGE_AND_ITS_PREDECLARED_SAMPLER_CHANGED_EQUAL_UPDATE_BUDGET",
        "development_checkpoint_or_threshold_selection": False, "snr5_used_for_training": False,
        "old_snr5_used_for_training": False, "new8_used_for_training": False,
        "new_training_frontend_calls": progress["new_training_frontend_calls"],
        "same_pool_objective_search_remains_closed": True, **dict.fromkeys(FALSE_FLAGS, False)}, create=True)
    def evaluate(name, head, rows, streams, *, is_new8=False):
        need(frozen_sha is not None and identity(root / "models/coverage-head.npz")["sha256"] == frozen_sha,
             "candidate remains frozen before every evaluation pass")
        metrics = []
        with (root / "predictions" / (name + "-frames.jsonl")).open("x", encoding="utf-8") as out:
            for row in rows:
                need(progress["head_case_passes_started"] < 318, "fixed total Python head case budget")
                progress["head_case_passes_started"] += 1
                if is_new8:
                    progress["new8_head_case_passes_started"] += 1
                checkpoint()
                matrix, samples = streams[row["id"]]
                result = portable.reference(head, matrix, samples)
                events = [{"sample": r["sample"], "keyword": r["keyword"]} for r in result if r["keyword"]]
                for record in result:
                    out.write(json.dumps({"id": row["id"], **record}, separators=(",", ":"), allow_nan=False) + "\n")
                out.flush()
                progress["head_ready_outputs"] += sum(bool(r["ready"]) for r in result)
                progress["head_case_passes_completed"] += 1
                if is_new8:
                    progress["new8_head_case_passes_completed"] += 1
                checkpoint()
                metrics.append(summarize_case(row, events))
        write_json(root / "predictions" / (name + "-events.json"), metrics, create=True)
        return metrics

    phase("OLD_CONDITION_EVALUATION_STARTED")
    coverage_clean = evaluate("coverage-clean", candidate, roster, clean_streams)
    coverage_snr10 = evaluate("coverage-snr10", candidate, snr10_roster, snr10_streams)
    coverage_snr5 = evaluate("coverage-snr5", candidate, snr5_roster, snr5_streams)
    need(progress["head_case_passes_completed"] == 264 and progress["head_ready_outputs"] == 11847,
         "all original cached conditions evaluated exactly once")
    phase("OLD_CONDITION_EVALUATION_COMPLETED")
    new8_summary, new8_receipts, new8_roster, new8_streams = prepare_new("development_challenge")
    phase("NEW8_EVALUATION_STARTED")
    aug96_new8 = evaluate("aug96-new8", control, new8_roster, new8_streams, is_new8=True)
    coverage_new8 = evaluate("coverage-new8", candidate, new8_roster, new8_streams, is_new8=True)
    phase("NEW8_EVALUATION_COMPLETED")
    need(progress["head_case_passes_completed"] == 264 + 2 * len(new8_roster)
         and progress["head_ready_outputs"] == 11847 + 2 * sum(row["records"] - 15 for row in new8_roster)
         and progress["head_ready_outputs"] <= 13039,
         "complete bounded candidate and new challenge control evaluation")
    need(progress["new_embedding_calls_completed"] == train_summary["new_embedding_calls_required"]
            + new8_summary["new_embedding_calls_required"]
         and progress["onnx_forward_completed"] == 2 * progress["new_embedding_calls_completed"]
         and progress["prefix_embedding_rows_reused"] == 16 * (len(new_train_roster) + len(new8_roster)),
         "exact fresh suffix and reused prefix accounting")
    aug96_clean = read_json(root / "predictions/reused/candidate-clean-events.json")
    aug96_snr10 = read_json(root / "predictions/reused/candidate-snr10-events.json")
    aug96_snr5 = read_json(root / "predictions/reused/aug96-snr5-events.json")
    paired = {"clean": paired_comparison(aug96_clean, coverage_clean),
        "snr10": paired_comparison(aug96_snr10, coverage_snr10),
        "snr5": paired_comparison(aug96_snr5, coverage_snr5),
        "new8": paired_comparison(aug96_new8, coverage_new8)}
    retention = retention_record(paired, snr5_roster, new8_roster)
    write_json(evidence / "PAIRED_COMPARISONS.json", paired, create=True)
    write_json(evidence / "RESEARCH_RETENTION.json", retention, create=True)

    phase("C_HEAD_PARITY_PREPARATION_STARTED")
    candidate_sets = (("clean", roster, clean_streams), ("snr10", snr10_roster, snr10_streams),
                      ("snr5", snr5_roster, snr5_streams), ("new8", new8_roster, new8_streams))
    total_records = sum(row["records"] for _, rows, _ in candidate_sets for row in rows)
    need(total_records <= 16808, "bounded cached candidate C inputs")
    reference, case_map, record_cursor, origin = [], [], 0, 0
    stream_path = root / "runtime/head-stream.bin"
    reference_path = root / "runtime/reference-frames.jsonl"
    with stream_path.open("xb") as stream, reference_path.open("x", encoding="utf-8") as saved_reference:
        stream.write(b"EHSTR01\0" + struct.pack("<I", total_records))
        for condition, rows, streams in candidate_sets:
            with (root / "predictions" / ("coverage-" + condition + "-frames.jsonl")).open(encoding="utf-8") as frames:
                for row in rows:
                    matrix, samples = streams[row["id"]]
                    need(matrix.shape == (len(samples), 96) and len(samples) == row["records"], "C exact source cache span")
                    case_map.append({"condition": condition, "id": row["id"],
                        "parent_id": row.get("parent_id", row["id"]), "first_record": record_cursor,
                        "records": len(samples), "offset": origin, "observed_samples": row["observed_samples"],
                        "head_outputs": len(samples) - 15})
                    for index, local_sample in enumerate(samples):
                        text = frames.readline()
                        need(bool(text), "saved Python reference complete")
                        old = json.loads(text)
                        need(old["id"] == row["id"] and old["sample"] == local_sample
                             and old["ready"] is (index >= 15), "saved Python reference identity and causal clock")
                        global_sample = origin + local_sample
                        stream.write(struct.pack("<QI", global_sample, int(index == 0)))
                        raw_vector = matrix[index].astype("<f4", copy=False).tobytes(order="C")
                        need(len(raw_vector) == 384, "C embedding vector byte width")
                        stream.write(raw_vector)
                        translated = {**old, "condition": condition, "local_sample": local_sample,
                                      "sample": global_sample, "case_index": len(case_map) - 1}
                        reference.append(translated)
                        saved_reference.write(json.dumps(translated, separators=(",", ":"), allow_nan=False) + "\n")
                    record_cursor += len(samples)
                    origin += row["observed_samples"]
                need(frames.read() == "", "no unbound saved Python frames")
    need(record_cursor == total_records == len(reference) and len(case_map) == 264 + len(new8_roster)
         and len(case_map) <= 291 and sum(r["ready"] for r in reference) <= 12443,
         "closed monotonic C case mapping without new reference evaluation")
    write_json(root / "runtime/CASE_MAP.json", case_map, create=True)
    phase("C_HEAD_PARITY_PREPARATION_COMPLETED")

    def runtime_command(name, argv, limit):
        remaining = float(started["deadline_unix"]) - time.time()
        need(remaining > 10, "total worker deadline preserves supervisor cleanup reserve")
        env = dict(os.environ)
        for key in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"):
            env.pop(key, None)
        out = evidence / (name + ".stdout")
        supervised_command(argv, out, evidence / (name + ".stderr"), min(limit, remaining - 7), env)
        return out

    tool_versions = {}
    for name, argv in (("host-compiler-version", ["cc", "--version"]),
                       ("arm-compiler-version", ["arm-linux-gnueabihf-gcc", "--version"]),
                       ("qemu-version", ["qemu-arm", "--version"])):
        tool_versions[name] = identity(runtime_command(name, argv, 10))

    def c_parity(backend):
        tag = "native" if backend == "host" else "arm"
        phase(backend.upper() + "_C_PARITY_STARTED")
        need(identity(root / "models/coverage-head.npz")["sha256"] == frozen_sha
             and all(identity(root / "sources" / name)["sha256"] == digest for name, digest in C_SOURCE_PINS.items()),
             "frozen candidate and exact C source before compilation")
        compiler = "cc" if backend == "host" else "arm-linux-gnueabihf-gcc"
        executable = root / "runtime" / ("head-" + tag)
        compiler_argv = [compiler, "-std=c99", "-O2", "-ffp-contract=off", "-fno-fast-math",
            "-Wall", "-Wextra", "-Werror", *([] if backend == "host" else ["-static"]),
            str(root / "sources/english_causal_head.c"), str(root / "sources/head_replay.c"),
            "-lm", "-o", str(executable)]
        need(progress[backend + "_c_builds_started"] == 0, "one C build per backend")
        progress[backend + "_c_builds_started"] += 1; checkpoint()
        runtime_command("build-head-" + tag, compiler_argv, 120)
        progress[backend + "_c_builds_completed"] += 1; checkpoint()
        elf = executable.read_bytes()[:20]
        need(len(elf) == 20 and elf[:4] == b"\x7fELF" and elf[5] == 1
             and struct.unpack_from("<H", elf, 18)[0] == (62 if backend == "host" else 40),
             "actual requested native or ARM ELF machine")
        runner_argv = ([] if backend == "host" else ["qemu-arm"]) + [str(executable),
            str(root / "models/coverage-head.bin"), str(stream_path)]
        need(progress[backend + "_c_head_processes_started"] == 0, "one candidate C replay per backend")
        progress[backend + "_c_head_processes_started"] = 1
        progress[backend + "_c_head_case_passes_started"] = len(case_map)
        progress[backend + "_c_embedding_rows_started"] = total_records; checkpoint()
        raw_stdout = runtime_command("head-" + tag, runner_argv, 300)
        report = compare_c_rows(raw_stdout.read_bytes(), reference)
        progress[backend + "_c_head_processes_completed"] = 1
        progress[backend + "_c_head_case_passes_completed"] = len(case_map)
        progress[backend + "_c_embedding_rows_completed"] = total_records
        progress[backend + "_c_head_ready_outputs"] = report["head_outputs"]; checkpoint()
        report.update(schema="english-coverage-" + backend + "-c-head-parity-v1",
            scope="CANDIDATE_CACHED_EMBEDDING_HEAD_ONLY", cases=len(case_map),
            candidate=identity(root / "models/coverage-head.npz"),
            candidate_binary=identity(root / "models/coverage-head.bin"),
            stream_binary=identity(stream_path), raw_stdout=identity(raw_stdout),
            reference_frames=identity(reference_path), case_map=identity(root / "runtime/CASE_MAP.json"),
            source_hashes=C_SOURCE_PINS, executable=identity(executable),
            compiler_argv=compiler_argv, runner_argv=runner_argv, tool_versions=tool_versions,
            threshold=0.8, confirm_embeddings=3, cooldown_embeddings=13,
            independent_python_head_reexecution=False, c_frontend="NOT_RUN", ssc305="NOT_RUN",
            formal_qualification=False, ssc305_qualification=False, **dict.fromkeys(FALSE_FLAGS, False))
        write_json(evidence / ("HOST_PARITY.json" if backend == "host" else "ARM_PARITY.json"), report, create=True)
        phase(backend.upper() + "_C_PARITY_COMPLETED")
        return report

    host_parity = c_parity("host")
    arm_parity = c_parity("arm")
    need(all(host_parity[key] == arm_parity[key] for key in
        ("candidate", "candidate_binary", "stream_binary", "reference_frames", "case_map", "cases",
         "input_embeddings", "head_outputs", "events")), "both C backends bind the identical candidate stream and events")
    need(all(value == progress[key[:-8] + "_completed"] for key, value in progress.items()
             if key.endswith("_started")), "all counted attempts have one completed receipt")
    need(progress["optimizer_updates_completed"] == 96
         and progress["torch_forward_calls_completed"] == 576
         and progress["new_training_frontend_calls"] == progress["train_new_embedding_calls_completed"]
         and progress["new8_head_case_passes_completed"] == 2 * len(new8_roster)
         and all(progress[key] == 0 for key in ("new_tts_asr_phone_calls", "new_c_frontend_calls", "retry_count")),
         "one fixed coverage experiment with no hidden numerical work")
    need(identity(root / "models/coverage-head.npz")["sha256"] == frozen_sha,
         "candidate is unchanged after all evaluations and both C replays")
    phase("COMPLETED")
    result = {"schema": SCHEMA, "status": "COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH",
        "run_id": os.environ["GITHUB_RUN_ID"], "head_sha": os.environ["GITHUB_SHA"], "run_attempt": 1,
        "training_source": TRAINING, "incumbent_source": INCUMBENT, "temporal_source": TEMPORAL,
        "baseline": identity(root / "models/baseline-head.npz"), "aug96_control": identity(root / "models/aug96-control-head.npz"),
        "coverage_candidate": identity(root / "models/coverage-head.npz"),
        "coverage_candidate_binary": identity(root / "models/coverage-head.bin"),
        "planned_new_conditions": 127, "planned_new_training_conditions": 100,
        "planned_new_development_conditions": 27, "eligible_new_training_conditions": len(new_train_roster),
        "eligible_new8_conditions": len(new8_roster),
        "new_training_dispositions": dict(collections.Counter(r["status"] for r in train_receipts)),
        "new8_dispositions": dict(collections.Counter(r["status"] for r in new8_receipts)),
        "source_synthetic_cases": 77, "source_synthetic_train": 50, "source_synthetic_development": 27,
        "independent_source_count_increment": 0, "new8_new_source_holdout": False,
        "same_source_condition_is_not_independent_audio": True, "cross_batch_voice_family_exposure": True,
        "prior_development_patterns_informed_research_hypothesis": True,
        "baseline_and_aug96_old_condition_predictions_reused": True, "snr5_previous_evaluation_exposed": True,
        "old_snr5_used_for_training": False, "new8_used_for_training": False,
        "failed_temporal_head_used_as_warmstart": False,
        "frozen_embedding_rows_reused": {"clean": 10354, "snr10": 2804, "snr5": 2649},
        "new_counts": dict(progress), "research_retention": retention,
        "aug96_clean": aggregate(aug96_clean), "aug96_snr10": aggregate(aug96_snr10),
        "aug96_snr5": aggregate(aug96_snr5), "aug96_new8": aggregate(aug96_new8),
        "coverage_clean": aggregate(coverage_clean), "coverage_snr10": aggregate(coverage_snr10),
        "coverage_snr5": aggregate(coverage_snr5), "coverage_new8": aggregate(coverage_new8),
        "training_coverage": train_summary["training_coverage"],
        "same_pool_objective_search_remains_closed": True,
        "subsequent_same_pool_loss_seed_threshold_search_allowed": False,
        "same_pool_epoch_search_allowed": False, "additional_coverage_retries_allowed": False,
        "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN", "word_end_latency": "NOT_MEASURED",
        "new_head_c_numeric_parity": "PASSED_NATIVE_AND_ARM_QEMU_CACHED_HEAD_ONLY",
        "host_c_parity": identity(evidence / "HOST_PARITY.json"), "arm_c_parity": identity(evidence / "ARM_PARITY.json"),
        "full_pcm_c_frontend_chain": "NOT_RUN", "ssc305": "NOT_RUN", "model_delivered_as_product": False,
        **dict.fromkeys(FALSE_FLAGS, False)}
    write_json(evidence / "RESULT.json", result, create=True)
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "downloads" not in path.relative_to(root).parts:
            manifest[path.relative_to(root).as_posix()] = identity(path)
    write_json(evidence / "WORKER_MANIFEST.json", {"files": manifest,
        "excludes_downloads_and_later_supervisor_files": True,
        "later_files": ["evidence/model-worker.exit.json", "evidence/PARENT_RESULT.json", "evidence/SUPERVISOR_EXIT.json"]}, create=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = read_json(args.plan); validate_plan(plan); source_contract(plan)
    if args.execute or args.worker:
        need(args.output is not None, "create-only output required")

        def interrupted(_number, _frame):
            raise KeyboardInterrupt("termination requested")

        signal.signal(signal.SIGTERM, interrupted)
        (worker if args.worker else execute)(args, plan)
    else:
        print(json.dumps({"status": "SOURCE_PLAN_VALIDATED_NO_MODEL_EXECUTION", "schema": SCHEMA}))


if __name__ == "__main__":
    main()


