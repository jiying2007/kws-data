#!/usr/bin/env python3
"""One equal-budget temporal-loss experiment; default CLI is source-only.

Only a pinned first-attempt public main push may invoke the single supervised
worker. No model library is imported by source validation or unit fixtures.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import zipfile

REPO = "jiying2007/kws-data"
MAIN_BEFORE = "c4f20fcb9f8b92aa77125c85de393df678ac812f"
SCHEMA = "english-temporal-quality-once-v1"
BUNDLE = "english-snapshot160-embedding-research-bundle/"
TRAINING = {"run_id": 43867764366, "artifact_id": 14225893574,
    "name": "snapshot160-weak-embedding-research-43867764366",
    "head_sha": "d8be1c24b12cac0ba554e9a4a41e4b6ee0db18db", "bytes": 61290521,
    "sha256": "2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1"}
INCUMBENT = {"run_id": 44021084918, "artifact_id": 14300818291,
    "name": "english-quality-snr10-continued-44021084918",
    "head_sha": "453128e605498cae2482c76b19745b2204d9fd79", "bytes": 32878537,
    "sha256": "acc6def807b1dd5728111beba4801352d8de9cb453e530f2270005254f44fa15"}
PINS = {
    "actual-embedding-streams.npz": "32db279b822f5e5787b5eabe0afaf7766d3971c3200c2aa3ffcb3d165256e5d2",
    "PER_CASE_EVENTS.json": "b186b75f4f15d7f91eda79be1567f6c1b5f93e94684ca9b9b5f145c47d384bdf",
    "FEATURE_EVIDENCE.json": "944e2c143d9b2fd9784aaabb5f650f42fee69d92bbc90b2c3d4a01c380c57a8c",
    "STREAM_ROSTER.json": "53646ce22780e8647e5654334c633a3f4dc8c1dd9a78d0856a1877b99d75b12a",
    "train/causal-head.npz": "6d7677b674f4a5e324aa916c5b777792b641fdfd61af8edc2f4e490275d600c4",
}
INCUMBENT_PINS = {
    "models/candidate-head.npz": "cdceab93cd6d3d36bff62980e8c7d2a95896ad32ecc88cae04fa232393e3e9e7",
    "cache/mixed-embeddings.npz": "1c1242fbad78ab34d2f1a1394bfa295efeb95b2e0cedee28b0832608b9b05d61",
    "cache/mixed-roster.json": "b16f06c5d7b582fa99f8591960def4c0af43bb97547db08851a1edefe99a8465",
    "evidence/TRAINING.json": "5838c7b0ead9ad117d93635bd33b88189c391ba5f6d4d8aa41e799b4c34912c5",
}
FALSE_FLAGS = ("formal_training_admitted", "training_admitted", "human_gold", "trusted_negative",
    "pronunciation_verified", "model_promotion_allowed", "qualification_allowed")
RUNTIME = ("weak_embedding_contract.py", "causal_embedding.py", "portable_head.py")
SOURCE_HASH_KEYS = ("worker_sha256", "workflow_sha256", "data_helper_sha256")

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def identity(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "regular input file")
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            h.update(part)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, data, *, create=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if create:
        with path.open("x", encoding="utf-8") as dst:
            dst.write(raw)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(raw, encoding="utf-8")
        os.replace(temporary, path)

def validate_zip(archive, *, maximum=256 * 1024**2):
    infos = archive.infolist()
    need(len(infos) == len({x.filename for x in infos}) and len(infos) <= 1000
         and sum(i.file_size for i in infos) <= maximum, "bounded unique ZIP")
    for info in infos:
        name = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        need(not name.is_absolute() and ".." not in name.parts and "\\" not in info.filename
             and name.as_posix() == info.filename and not info.is_dir()
             and (not stat.S_IFMT(mode) or stat.S_ISREG(mode)), "regular safe ZIP member")

def archive_read(archive, name):
    raw = archive.read(name)
    if name in PINS:
        need(sha(raw) == PINS[name], "saved member SHA: " + name)
    return raw

def summarize_case(row, events):
    source = row.get("source", {})
    start = source.get("clip_start_sample")
    eof = source.get("original_clip_eof_sample")
    expected = row["weak_class"]
    need(row.get("role") in ("train", "development") and type(row.get("observed_samples")) is int
         and row["observed_samples"] > 0, "valid row role and observed length")
    need((start is None and eof is None) or
         (type(start) is int and type(eof) is int and 0 < start < eof <= row["observed_samples"]),
         "clip interval is inside observed PCM")
    need(type(expected) is int and expected in (0, 1, 2), "weak intent class")
    need(start is not None or expected == 0, "positive intent requires a source clip interval")
    need(all(type(e.get("sample")) is int and 0 < e["sample"] <= row["observed_samples"]
             and type(e.get("keyword")) is int and e["keyword"] in (1, 2) for e in events), "observed event clocks")
    need(all(a["sample"] < b["sample"] for a, b in zip(events, events[1:])), "strictly ordered unique events")
    prefix = [e for e in events if start is not None and e["sample"] <= start]
    after = [e for e in events if start is None or e["sample"] > start]
    correct = [e for e in after if expected and e["keyword"] == expected]
    wrong = [e for e in after if expected and e["keyword"] != expected]
    deficit = bool(expected and not correct)
    extra = max(0, len(correct) - 1)
    error = bool(prefix or wrong or deficit or extra or (expected == 0 and events))
    return {"id": row["id"], "role": row["role"], "weak_class": expected,
            "events": events, "matched_intent_after_clip_start": bool(correct),
            "prefix_events": prefix, "wrong_keyword_after_clip_start": wrong,
            "extra_same_keyword_events": extra, "no_matching_event_in_observed_context": deficit,
            "weak_negative_intent_events": len(events) if expected == 0 else 0,
            "research_error": error, "observed_samples": row["observed_samples"],
            "clip_start_sample": start, "original_clip_eof_sample": eof,
            "event_minus_clip_eof_samples": [e["sample"] - eof for e in correct] if eof is not None else [],
            "eof_prefix_event_count": sum(e["sample"] <= eof for e in events) if eof is not None else None,
            "no_event_status": "CENSORED_NOT_FRR" if not events and expected else None,
            "word_end_latency": "NOT_MEASURED", "independent_acoustic_evaluation": False}

def aggregate(metrics):
    groups = {}
    for row in metrics:
        key = row["role"] + "/" + str(row["weak_class"])
        g = groups.setdefault(key, {"cases": 0, "matched_intent_cases": 0, "research_error_cases": 0,
                                   "prefix_events": 0, "wrong_keyword_events": 0,
                                   "extra_same_keyword_events": 0, "weak_negative_intent_events": 0,
                                   "no_matching_event_in_observed_context": 0})
        g["cases"] += 1
        g["matched_intent_cases"] += int(row["matched_intent_after_clip_start"])
        g["research_error_cases"] += int(row["research_error"])
        g["prefix_events"] += len(row["prefix_events"])
        g["wrong_keyword_events"] += len(row["wrong_keyword_after_clip_start"])
        g["extra_same_keyword_events"] += row["extra_same_keyword_events"]
        g["weak_negative_intent_events"] += row["weak_negative_intent_events"]
        g["no_matching_event_in_observed_context"] += int(row["no_matching_event_in_observed_context"])
    return groups

def supervised_command(argv, stdout, stderr, timeout, env=None):
    begin, process, code, error = time.monotonic(), None, None, None
    try:
        with Path(stdout).open("xb") as out, Path(stderr).open("xb") as err:
            process = subprocess.Popen(list(map(str, argv)), stdout=out, stderr=err, env=env, start_new_session=True)
            code = process.wait(timeout=timeout)
        need(code == 0, "child command failed")
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        terminated = killed = False
        if process is not None:
            # A leader that exits first can leave a live descendant in its
            # session. Always clean the owned group, including after exit 0.
            def group_exists():
                try:
                    os.killpg(process.pid, 0)
                    return True
                except ProcessLookupError:
                    return False
            if group_exists():
                terminated = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                grace = time.monotonic() + 2
                while group_exists() and time.monotonic() < grace:
                    process.poll()
                    time.sleep(0.02)
                if group_exists():
                    killed = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if process.poll() is None:
                process.wait(timeout=5)
        if process is not None:
            code = process.returncode
        write_json(Path(stdout).with_suffix(".exit.json"), {"argv": list(map(str, argv)),
            "returncode": code, "error": error, "process_group_terminated": terminated,
            "process_group_killed": killed, "group_cleanup_checked_after_leader_exit": process is not None,
            "wall_seconds": time.monotonic() - begin, "timeout_seconds": timeout})


def expected_plan():
    """All experiment choices are fixed before execution; callers add three source hashes."""
    return {
        "schema": SCHEMA, "main_before": MAIN_BEFORE,
        "training_source": TRAINING, "incumbent_source": INCUMBENT,
        "snr_db": 5.0, "planned_new_conditions": 77,
        "planned_maximum_embedding_calls": 2834, "maximum_embedding_calls": 4000,
        "maximum_onnx_forwards": 8000, "maximum_head_case_passes": 424,
        "maximum_optimizer_updates": 96, "maximum_training_bag_forwards": 576,
        "maximum_execution_seconds": 900, "retry_count": 0,
        "new_tts_asr_phone_calls": 0, "new_training_frontend_calls": 0,
        "seed": 20261117, "cpu_threads": 1, "learning_rate": 0.001,
        "weight_decay": 0.0001, "gradient_clip_norm": 1.0,
        "threshold": 0.8, "confirm_embeddings": 3, "cooldown_embeddings": 13,
        "new_training_arms": 1, "optimizer": "FRESH_ADAMW",
        "warmstart": "ORIGINAL_BASELINE", "normalization": "FROZEN_ORIGINAL_TRAIN_ONLY",
        "positive_loss": "NEGATIVE_LOG_MEAN_THREE_CONSECUTIVE_MIN_TARGET_PROBABILITY",
        "positive_min_operator": "TORCH_AMIN_WITH_TIE_GRADIENT_SHARING",
        "background_loss": "MEAN_WINDOW_BACKGROUND_CROSS_ENTROPY",
        "bag_aggregation": "SIX_EQUAL_CONDITION_CLASS_TERMS",
        "training_conditions": ["clean", "snr10"],
        "evaluation_conditions": ["clean", "snr10", "snr5"],
        "draws": "EXACT_PREVIOUS_96_BY_6_DRAW_SEQUENCE",
        "positive_geometry": "UNCHANGED_ORIGINAL_MEL_SUPPORTED_HEAD_INSTANCE_ENDS",
        "development_tuning": False, "checkpoint_selection": False,
        "incumbent_retraining": False, "old_frontend_recomputation": False,
        "snr5_training_allowed": False, "snr5_evaluation_after_candidate_frozen": True,
        "mix_scope": "ORIGINAL_UNTRIMMED_CLIP_ONLY",
        "noise_binding": "EXACT_PREVIOUS_TRACK_OFFSET_AND_HASH_DOMAIN",
        "overflow_policy": "EXCLUDE_WITHOUT_REPLACEMENT_OR_CLIPPING",
        "research_retention": {
            "clean_no_previously_correct_case_regression": True,
            "snr10_no_previously_correct_case_regression": True,
            "snr10_minimum_rescued_development_misses": 1,
            "all_conditions_no_per_case_prefix_wrong_extra_or_negative_increase": True,
            "snr5_no_previously_correct_case_regression": True,
            "snr5_minimum_development_positive_cases_per_keyword": 1,
            "reference": "PREVIOUS_AUGMENTED_96_UPDATE_HEAD",
            "formal_promotion_allowed": False,
        },
        **dict.fromkeys(FALSE_FLAGS, False),
    }


def validate_plan(plan):
    expected = expected_plan()
    need(set(plan) == set(expected) | set(SOURCE_HASH_KEYS), "exact closed plan schema")
    need(all(type(plan[k]) is type(value) and plan[k] == value
             for k, value in expected.items()), "fixed registered experiment choices")
    for key in SOURCE_HASH_KEYS:
        need(re.fullmatch(r"[0-9a-f]{64}", plan[key]) is not None, "source SHA: " + key)


def source_contract(plan):
    here = Path(__file__).resolve()
    paths = {
        "worker_sha256": here,
        "data_helper_sha256": here.with_name("english_temporal_quality_data.py"),
        "workflow_sha256": here.parent.parent / ".github/workflows/english-temporal-quality-once.yml",
    }
    for key, path in paths.items():
        need(identity(path)["sha256"] == plan[key], "executed source identity: " + key)
    return {key: {"path": str(path), **identity(path)} for key, path in paths.items()}


def execution_guard(plan, env=None, event=None):
    env = os.environ if env is None else env
    event = read_json(env["GITHUB_EVENT_PATH"]) if event is None else event
    need(env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_REPOSITORY") == REPO
         and env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == "refs/heads/main"
         and env.get("GITHUB_RUN_ATTEMPT") == "1", "public first-attempt main push required")
    repo = event.get("repository", {})
    need(repo.get("full_name") == REPO and repo.get("private") is False
         and event.get("before") == plan["main_before"] and event.get("after") == env.get("GITHUB_SHA")
         and re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")) is not None
         and re.fullmatch(r"[0-9]+", env.get("GITHUB_RUN_ID", "")) is not None,
         "exact new public main successor required")
    source_contract(plan)


def verify_remote_source(run, artifact, expected):
    need(run.get("status") == "completed" and run.get("conclusion") == "success"
         and run.get("id") == expected["run_id"] and run.get("head_sha") == expected["head_sha"]
         and run.get("run_attempt") == 1 and run.get("event") == "push" and run.get("head_branch") == "main"
         and run.get("repository", {}).get("full_name") == REPO
         and run.get("repository", {}).get("private") is False,
         "fixed successful public source run")
    need(artifact.get("id") == expected["artifact_id"] and artifact.get("name") == expected["name"]
         and artifact.get("size_in_bytes") == expected["bytes"] and artifact.get("expired") is False
         and artifact.get("workflow_run", {}).get("id") == expected["run_id"]
         and artifact.get("workflow_run", {}).get("head_sha") == expected["head_sha"]
         and artifact.get("digest") == "sha256:" + expected["sha256"], "fixed public artifact identity")


def validate_head_ends(row, ends, samples):
    """Pure clock/weak-label geometry checks; no model is evaluated here."""
    need(ends and len(ends) == len(set(ends)) and all(type(t) is int for t in ends), "unique bag ends")
    positions = {t: i for i, t in enumerate(samples)}
    need(len(positions) == len(samples), "unique local stream samples")
    for end in ends:
        need(end in positions and positions[end] >= 15, "full original head history")
        i = positions[end]
        need(samples[i - 15:i + 1] == list(range(end - 15 * 1280, end + 1, 1280)),
             "no cross-gap history or invented samples")
    if row["weak_class"]:
        source = row["source"]
        start, eof = source["clip_start_sample"], source["original_clip_eof_sample"]
        need(len(ends) >= 3 and all(b - a == 1280 for a, b in zip(ends, ends[1:])),
             "three genuinely consecutive original outputs")
        need(all(end - 31840 <= start and end - 128 >= eof
                 and end <= row["observed_samples"] for end in ends), "unchanged full-clip actual mel support")
    return positions


def paired_comparison(reference, candidate):
    need(len(reference) == len(candidate) and len({r["id"] for r in reference}) == len(reference)
         and [r["id"] for r in reference] == [r["id"] for r in candidate], "paired case order and identity")
    comparisons = []
    for old, new in zip(reference, candidate):
        need(all(old[key] == new[key] for key in ("id", "role", "weak_class", "observed_samples",
                                                 "clip_start_sample", "original_clip_eof_sample")),
             "unchanged paired case semantics")
        increases = []
        for key in ("prefix_events", "wrong_keyword_after_clip_start"):
            if len(new[key]) > len(old[key]):
                increases.append(key)
        for key in ("extra_same_keyword_events", "weak_negative_intent_events"):
            if new[key] > old[key]:
                increases.append(key)
        comparisons.append({"id": old["id"], "role": old["role"], "weak_class": old["weak_class"],
            "previous_research_error": old["research_error"], "temporal_research_error": new["research_error"],
            "previously_correct_case_regressed": not old["research_error"] and new["research_error"],
            "event_error_categories_increased": increases,
            "development_miss_rescued": old["role"] == "development" and old["weak_class"] > 0
                and old["no_matching_event_in_observed_context"] and not new["research_error"],
            "previous_events": old["events"], "temporal_events": new["events"]})
    return comparisons


def research_retention(paired, snr5_rows):
    criteria = {}
    for condition in ("clean", "snr10", "snr5"):
        criteria[condition + "_no_previously_correct_case_regression"] = not any(
            row["previously_correct_case_regressed"] for row in paired[condition])
    criteria["snr10_at_least_one_rescued_development_miss"] = any(
        row["development_miss_rescued"] for row in paired["snr10"])
    criteria["all_conditions_no_per_case_event_error_increase"] = not any(
        row["event_error_categories_increased"] for cases in paired.values() for row in cases)
    criteria["snr5_both_development_keywords_present"] = all(
        any(r["role"] == "development" and r["weak_class"] == label for r in snr5_rows)
        for label in (1, 2))
    passed = all(criteria.values())
    return {"passed": passed, "status": "PASSED_RESEARCH_RETENTION_GATE" if passed else
            "FAILED_RESEARCH_RETENTION_GATE_DIAGNOSTIC_ONLY", "criteria": criteria,
            "reference": "PREVIOUS_AUGMENTED_96_UPDATE_HEAD", "formal_promotion_allowed": False,
            **dict.fromkeys(FALSE_FLAGS, False)}


def execute(args, plan):
    execution_guard(plan)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    evidence, downloads = root / "evidence", root / "downloads"
    evidence.mkdir(); downloads.mkdir()
    write_json(evidence / "PLAN.json", plan, create=True)
    write_json(evidence / "STARTED.json", {"schema": SCHEMA, "run_id": os.environ["GITHUB_RUN_ID"],
        "head_sha": os.environ["GITHUB_SHA"], "run_attempt": 1, "parent_pid": os.getpid(),
        "training_source": TRAINING, "incumbent_source": INCUMBENT, "retry_count": 0,
        "numeric_attempt": 1, **dict.fromkeys(FALSE_FLAGS, False)}, create=True)
    started = time.monotonic()

    def command(name, argv, limit=120, auth=False):
        remaining = 900 - (time.monotonic() - started)
        need(remaining > 7, "total execution deadline including cleanup reserve")
        env = dict(os.environ)
        env["KWS_TEMPORAL_PARENT_PID"] = str(os.getpid())
        if not auth:
            for key in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"):
                env.pop(key, None)
        out = downloads / (name + ".stdout") if name.endswith("-zip") else evidence / (name + ".stdout")
        supervised_command(argv, out, evidence / (name + ".stderr"), min(limit, remaining - 7), env)
        return out

    try:
        for label, expected in (("training", TRAINING), ("incumbent", INCUMBENT)):
            run = read_json(command(label + "-run", ["gh", "api",
                f"repos/{REPO}/actions/runs/{expected['run_id']}"], auth=True))
            artifact = read_json(command(label + "-artifact", ["gh", "api",
                f"repos/{REPO}/actions/artifacts/{expected['artifact_id']}"], auth=True))
            verify_remote_source(run, artifact, expected)
            path = command(label + "-zip", ["gh", "api",
                f"repos/{REPO}/actions/artifacts/{expected['artifact_id']}/zip"], 240, True)
            destination = downloads / (label + ".zip")
            path.rename(destination)
            need(identity(destination) == {k: expected[k] for k in ("bytes", "sha256")},
                 "downloaded immutable source: " + label)
        write_json(evidence / "SOURCE_ACQUIRED.json", {"training": identity(downloads / "training.zip"),
            "incumbent": identity(downloads / "incumbent.zip"), "new_corpus_downloads": 0,
            "automatic_retries": 0, "new_model_calls": 0}, create=True)
        command("model-worker", [sys.executable, "-B", Path(__file__).resolve(), "--plan", args.plan.resolve(),
                                  "--worker", "--output", root], 900)
        result = read_json(evidence / "RESULT.json")
        need(result.get("status") == "COMPLETED_WEAK_TEMPORAL_QUALITY_RESEARCH", "worker completion")
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
         and os.environ.get("KWS_TEMPORAL_PARENT_PID") == str(os.getppid())
         and started["run_id"] == os.environ["GITHUB_RUN_ID"]
         and started["head_sha"] == os.environ["GITHUB_SHA"], "single owning supervisor must launch worker")
    training, incumbent = root / "downloads/training.zip", root / "downloads/incumbent.zip"
    need(identity(training) == {k: TRAINING[k] for k in ("bytes", "sha256")}
         and identity(incumbent) == {k: INCUMBENT[k] for k in ("bytes", "sha256")}, "authenticated worker inputs")
    acquired = read_json(evidence / "SOURCE_ACQUIRED.json")
    need(acquired["training"] == identity(training) and acquired["incumbent"] == identity(incumbent)
         and acquired["automatic_retries"] == acquired["new_model_calls"] == 0, "source acquisition precedes worker")
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
        "execution_provider": "CPUExecutionProvider", "numeric_threads": 1,
        "soundfile_or_corpus_decoder_used": False}, create=True)
    progress = {"new_embedding_calls_started": 0, "new_embedding_calls_completed": 0,
        "onnx_forward_started": 0, "onnx_forward_completed": 0,
        "head_case_passes_started": 0, "head_case_passes_completed": 0, "head_ready_outputs": 0,
        "optimizer_updates_started": 0, "optimizer_updates_completed": 0,
        "torch_forward_calls_started": 0, "torch_forward_calls_completed": 0,
        "new_training_frontend_calls": 0, "new_tts_asr_phone_calls": 0, "retry_count": 0}
    phases, frozen_sha = [], None

    def checkpoint():
        write_json(evidence / "PROGRESS.json", progress)

    def phase(name):
        phases.append({"phase_number": len(phases) + 1, "phase": name,
            "progress": dict(progress), "candidate_sha256": frozen_sha})
        write_json(evidence / "PHASES.json", phases)

    checkpoint(); phase("WORKER_STARTED")
    for folder in ("sources", "models", "cache", "audio/snr5", "predictions/reused"):
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

    with zipfile.ZipFile(training) as original, zipfile.ZipFile(incumbent) as previous:
        validate_zip(original); validate_zip(previous)
        roster = json.loads(copy_member(original, "STREAM_ROSTER.json", "sources/STREAM_ROSTER.json", "training", PINS))
        feature_evidence = json.loads(copy_member(original, "FEATURE_EVIDENCE.json", "sources/FEATURE_EVIDENCE.json", "training", PINS))
        for name in ("FEATURE_PREFLIGHT.json", "PER_CASE_EVENTS.json", "LIBRISPEECH_SOURCES.json", "SELECTION.json"):
            copy_member(original, name, "sources/" + name, "training", PINS)
        cache_raw = copy_member(original, "actual-embedding-streams.npz", "cache/clean-original.npz", "training", PINS)
        baseline_raw = copy_member(original, "train/causal-head.npz", "models/baseline-head.npz", "training", PINS)
        need(previous.read("cache/clean-original.npz") == cache_raw
             and previous.read("models/baseline-head.npz") == baseline_raw,
             "same original baseline and clean cache in both authenticated artifacts")
        for name in RUNTIME:
            copy_member(original, BUNDLE + "runtime/" + name, "sources/" + name, "training")
        for name in ("melspectrogram.onnx", "embedding_model.onnx"):
            copy_member(original, BUNDLE + "models/" + name, "models/" + name, "training")
        for name in ("kws-pipeline-LICENSE", "openWakeWord-LICENSE"):
            copy_member(original, BUNDLE + "licenses/" + name, "sources/" + name, "training")
        copy_member(previous, "background_tracks/MANIFEST.json", "sources/PREVIOUS_BACKGROUND_MANIFEST.json", "incumbent")
        previous_training = json.loads(copy_member(previous, "evidence/TRAINING.json", "sources/PREVIOUS_TRAINING.json",
                                                   "incumbent", INCUMBENT_PINS))
        copy_member(previous, "evidence/RESULT.json", "sources/PREVIOUS_RESULT.json", "incumbent")
        copy_member(previous, "evidence/RUNTIME.json", "sources/PREVIOUS_RUNTIME.json", "incumbent")
        copy_member(previous, "evidence/MIX_ROSTER.json", "sources/PREVIOUS_MIX_RECEIPTS.json", "incumbent")
        copy_member(previous, "models/candidate-head.npz", "models/aug96-control-head.npz", "incumbent", INCUMBENT_PINS)
        mixed_raw = copy_member(previous, "cache/mixed-embeddings.npz", "cache/mixed-snr10.npz", "incumbent", INCUMBENT_PINS)
        snr10_roster = json.loads(copy_member(previous, "cache/mixed-roster.json", "cache/snr10-roster.json",
                                            "incumbent", INCUMBENT_PINS))
        for name in previous.namelist():
            if name.startswith("predictions/"):
                copy_member(previous, name, "predictions/reused/" + PurePosixPath(name).name, "incumbent")
    here = Path(__file__).resolve()
    for source, destination in ((here, "sources/english_temporal_quality_once.py"),
            (here.with_name("english_temporal_quality_data.py"), "sources/english_temporal_quality_data.py"),
            (here.parent.parent / ".github/workflows/english-temporal-quality-once.yml", "sources/english-temporal-quality-once.yml")):
        (root / destination).write_bytes(source.read_bytes())
    write_json(evidence / "INPUTS.json", {"training_source": TRAINING, "incumbent_source": INCUMBENT,
        "actual_training": identity(training), "actual_incumbent": identity(incumbent),
        "source_contract": source_contract(plan), "copied_members": copied,
        "old_embedding_rows_reused": {"clean": 10354, "snr10": 2804},
        "old_frontend_recomputation": False, "new_training_frontend_calls": 0}, create=True)
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
    previous_draws = previous_training["draws"]
    need(previous_training["optimizer_updates"] == 96 and len(previous_draws) == 96
         and all(len(draw) == 6 for draw in previous_draws)
         and previous_training["warmstart"] == identity(root / "models/baseline-head.npz")
         and previous_training["candidate"] == identity(root / "models/aug96-control-head.npz")
         and previous_training["development_checkpoint_or_threshold_selection"] is False,
         "matched fixed-budget prior control")
    groups = {(condition, label): [] for condition in ("clean", "snr10") for label in range(3)}
    prepared, train_geometry = {}, []
    for condition, rows, streams in (("clean", roster, clean_streams), ("snr10", snr10_roster, snr10_streams)):
        for row in rows:
            if row["role"] != "train":
                continue
            matrix, samples = streams[row["id"]]
            ends = by_feature[row["id"]]["head_instance_ends"]
            positions = validate_head_ends(row, ends, samples)
            windows = np.asarray([matrix[positions[t] - 15:positions[t] + 1] for t in ends], dtype=np.float32)
            normalized = ((windows - weights["mean96"]) / weights["scale96"]).reshape(len(windows), 1536)
            key = (condition, row["id"])
            prepared[key] = torch.from_numpy(normalized)
            groups[(condition, row["weak_class"])].append(key)
            train_geometry.append({"condition": condition, "id": row["id"], "role": "train",
                "weak_class": row["weak_class"], "head_instance_ends": ends,
                "triplet_count": len(ends) - 2 if row["weak_class"] else None})
    need(all(groups.values()), "same six fixed train-only groups")
    write_json(evidence / "TRAINING_GEOMETRY.json", train_geometry, create=True)
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
        terms, used = [], []
        for condition in ("clean", "snr10"):
            for label in range(3):
                group = groups[(condition, label)]
                key = group[step % len(group)]
                draw = {"condition": condition, "id": key[1], "weak_class": label, "role": "train"}
                need(draw == previous_draws[step][len(used)], "each prior draw exactly matches before its forward")
                need(progress["torch_forward_calls_started"] < 576, "training bag forward budget")
                progress["torch_forward_calls_started"] += 1; checkpoint()
                logp = torch.log_softmax(model(prepared[key]), dim=-1)
                progress["torch_forward_calls_completed"] += 1; checkpoint()
                if label == 0:
                    terms.append(-logp[:, 0].mean())
                else:
                    triple_logp = torch.amin(logp[:, label].unfold(0, 3, 1), dim=1)
                    terms.append(math.log(len(triple_logp)) - torch.logsumexp(triple_logp, dim=0))
                used.append(draw)
        need(used == previous_draws[step], "complete six-bag draw order matches")
        loss = torch.stack(terms).mean()
        need(bool(torch.isfinite(loss)), "finite temporal training loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        losses.append(float(loss.detach())); draws.append(used)
        progress["optimizer_updates_completed"] += 1; checkpoint()
    need(draws == previous_draws and progress["torch_forward_calls_completed"] == 576, "fixed training completed exactly once")
    candidate = {"w1": model[0].weight.detach().numpy(), "b1": model[0].bias.detach().numpy(),
        "w2": model[2].weight.detach().numpy(), "b2": model[2].bias.detach().numpy(),
        "mean96": weights["mean96"].copy(), "scale96": weights["scale96"].copy()}
    np.savez(root / "models/temporal-head.npz", **candidate)
    candidate = portable.load_weights(root / "models/temporal-head.npz")
    normalization = {}
    for key in ("mean96", "scale96"):
        old, new = weights[key].tobytes(order="C"), candidate[key].tobytes(order="C")
        need(old == new == control[key].tobytes(order="C"), "byte-identical frozen train-only normalization")
        normalization[key] = {"bytes": len(old), "sha256": sha(old), "candidate_sha256": sha(new), "byte_identical": True}
    (root / "models/temporal-head.bin").write_bytes(portable.export_blob(candidate))
    frozen_sha = identity(root / "models/temporal-head.npz")["sha256"]
    phase("CANDIDATE_FROZEN")
    write_json(evidence / "CANDIDATE_FROZEN.json", {"phase_number": len(phases), "candidate": identity(root / "models/temporal-head.npz"),
        "candidate_bin": identity(root / "models/temporal-head.bin"), "normalization": normalization,
        "progress": dict(progress), "snr5_model_calls_before_freeze": 0,
        "all_future_evaluation_uses_this_frozen_head": True}, create=True)
    write_json(evidence / "TRAINING.json", {"optimizer_updates": 96, "training_bag_forwards": 576,
        "warmstart": identity(root / "models/baseline-head.npz"), "control": identity(root / "models/aug96-control-head.npz"),
        "candidate": identity(root / "models/temporal-head.npz"), "losses": losses, "draws": draws,
        "draws_exact_match": True, "previous_training": identity(root / "sources/PREVIOUS_TRAINING.json"),
        "normalization": normalization, "positive_loss": plan["positive_loss"],
        "background_loss": plan["background_loss"], "bag_aggregation": plan["bag_aggregation"],
        "comparison": "ONLY_POSITIVE_TEMPORAL_POOLING_CHANGED_EQUAL_DATA_DRAWS_AND_UPDATE_BUDGET",
        "development_checkpoint_or_threshold_selection": False, "snr5_used_for_training": False,
        "new_training_frontend_calls": 0, **dict.fromkeys(FALSE_FLAGS, False)}, create=True)

    def evaluate(name, head, rows, streams):
        need(frozen_sha is not None and identity(root / "models/temporal-head.npz")["sha256"] == frozen_sha,
             "candidate remains frozen before evaluation")
        metrics = []
        with (root / "predictions" / (name + "-frames.jsonl")).open("x", encoding="utf-8") as out:
            for row in rows:
                need(progress["head_case_passes_started"] < 424, "fixed total head case pass budget")
                progress["head_case_passes_started"] += 1; checkpoint()
                matrix, samples = streams[row["id"]]
                result = portable.reference(head, matrix, samples)
                events = [{"sample": r["sample"], "keyword": r["keyword"]} for r in result if r["keyword"]]
                for record in result:
                    out.write(json.dumps({"id": row["id"], **record}, separators=(",", ":"), allow_nan=False) + "\n")
                out.flush()
                progress["head_ready_outputs"] += sum(bool(r["ready"]) for r in result)
                progress["head_case_passes_completed"] += 1; checkpoint()
                metrics.append(summarize_case(row, events))
        write_json(root / "predictions" / (name + "-events.json"), metrics, create=True)
        return metrics

    phase("OLD_CONDITION_EVALUATION_STARTED")
    temporal_clean = evaluate("temporal-clean", candidate, roster, clean_streams)
    temporal_snr10 = evaluate("temporal-snr10", candidate, snr10_roster, snr10_streams)
    phase("OLD_CONDITION_EVALUATION_COMPLETED")
    helper_path = here.with_name("english_temporal_quality_data.py")
    spec = importlib.util.spec_from_file_location("frozen_temporal_quality_data", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    phase("SNR5_DATA_PREPARATION_STARTED")
    summary, receipts, snr5_roster, pcm_by_id = helper.build_snr5_conditions(training, incumbent)
    need(len(receipts) == 77 and len(snr5_roster) <= 77
         and summary["eligible_embedding_calls"] <= 2834
         and set(pcm_by_id) == {row["id"] for row in snr5_roster}, "fixed bounded 5 dB conditions")
    write_json(evidence / "SNR5_SUMMARY.json", summary, create=True)
    write_json(evidence / "SNR5_RECEIPTS.json", receipts, create=True)
    write_json(root / "cache/snr5-roster.json", snr5_roster, create=True)
    for row in snr5_roster:
        raw = pcm_by_id[row["id"]]
        need(len(raw) == 2 * row["observed_samples"]
             and sha(raw) == row["mix"]["combined_mixed_pcm_sha256"], "saved 5 dB complete observed PCM")
        (root / "audio/snr5" / (row["id"] + ".pcm")).write_bytes(raw)
    phase("SNR5_DATA_PREPARATION_COMPLETED")
    phase("SNR5_FEATURES_STARTED")
    import causal_embedding as embedding_runtime
    embedding = embedding_runtime.FrozenOnnxEmbedding(root / "models")
    expected_geometry = read_json(root / "sources/FEATURE_PREFLIGHT.json")["geometry"]
    need(embedding.geometry == expected_geometry, "same exact original frontend geometry")

    class CountedSession:
        def __init__(self, session):
            self.session = session

        def run(self, *values, **kwargs):
            need(progress["onnx_forward_started"] < 8000, "ONNX forward budget")
            progress["onnx_forward_started"] += 1; checkpoint()
            answer = self.session.run(*values, **kwargs)
            progress["onnx_forward_completed"] += 1; checkpoint()
            return answer

    embedding.mel, embedding.embedding = CountedSession(embedding.mel), CountedSession(embedding.embedding)

    def embed(values):
        need(progress["new_embedding_calls_started"] < 4000
             and progress["new_embedding_calls_started"] < summary["eligible_embedding_calls"], "fixed embedding call budget")
        progress["new_embedding_calls_started"] += 1; checkpoint()
        vector = embedding(values)
        progress["new_embedding_calls_completed"] += 1; checkpoint()
        return vector

    snr5_streams, cache_arrays = {}, {}
    for index, row in enumerate(snr5_roster):
        values = np.frombuffer(pcm_by_id[row["id"]], dtype="<i2")
        samples = list(range(16000, len(values) + 1, 1280))
        vectors = np.asarray([embed(values[t - 16000:t]) for t in samples], dtype=np.float32)
        snr5_streams[row["id"]] = (vectors, samples)
        cache_arrays[f"e{index:03d}"] = vectors
        cache_arrays[f"t{index:03d}"] = np.asarray(samples, dtype=np.uint64)
    np.savez(root / "cache/snr5-embeddings.npz", **cache_arrays)
    need(progress["new_embedding_calls_completed"] == summary["eligible_embedding_calls"]
         and progress["onnx_forward_completed"] == 2 * summary["eligible_embedding_calls"], "exact one-pass fresh features")
    write_json(evidence / "FEATURE_GEOMETRY.json", embedding.geometry, create=True)
    phase("SNR5_FEATURES_COMPLETED")
    phase("SNR5_EVALUATION_STARTED")
    baseline_snr5 = evaluate("baseline-snr5", weights, snr5_roster, snr5_streams)
    aug96_snr5 = evaluate("aug96-snr5", control, snr5_roster, snr5_streams)
    temporal_snr5 = evaluate("temporal-snr5", candidate, snr5_roster, snr5_streams)
    phase("SNR5_EVALUATION_COMPLETED")
    need(progress["head_case_passes_completed"] == 193 + 3 * len(snr5_roster)
         and all(progress[start] == progress[end] for start, end in (
             ("new_embedding_calls_started", "new_embedding_calls_completed"),
             ("onnx_forward_started", "onnx_forward_completed"),
             ("head_case_passes_started", "head_case_passes_completed"),
             ("optimizer_updates_started", "optimizer_updates_completed"),
             ("torch_forward_calls_started", "torch_forward_calls_completed"))), "all bounded work completed without retry")
    aug96_clean = read_json(root / "predictions/reused/candidate-clean-events.json")
    aug96_snr10 = read_json(root / "predictions/reused/candidate-snr10-events.json")
    paired = {"clean": paired_comparison(aug96_clean, temporal_clean),
        "snr10": paired_comparison(aug96_snr10, temporal_snr10),
        "snr5": paired_comparison(aug96_snr5, temporal_snr5)}
    retention = research_retention(paired, snr5_roster)
    write_json(evidence / "PAIRED_COMPARISONS.json", paired, create=True)
    write_json(evidence / "RESEARCH_RETENTION.json", retention, create=True)
    phase("COMPLETED")
    result = {"schema": SCHEMA, "status": "COMPLETED_WEAK_TEMPORAL_QUALITY_RESEARCH",
        "run_id": os.environ["GITHUB_RUN_ID"], "head_sha": os.environ["GITHUB_SHA"], "run_attempt": 1,
        "training_source": TRAINING, "incumbent_source": INCUMBENT,
        "baseline": identity(root / "models/baseline-head.npz"), "aug96_control": identity(root / "models/aug96-control-head.npz"),
        "temporal_candidate": identity(root / "models/temporal-head.npz"),
        "planned_snr5_conditions": 77, "eligible_snr5_conditions": len(snr5_roster),
        "snr5_dispositions": dict(collections.Counter(r["status"] for r in receipts)),
        "source_synthetic_cases": 77, "source_synthetic_train": 50, "source_synthetic_development": 27,
        "same_source_condition_is_not_independent_audio": True, "cross_batch_voice_family_exposure": True,
        "prior_development_patterns_informed_research_hypothesis": True,
        "baseline_and_aug96_old_condition_predictions_reused": True,
        "frozen_embedding_rows_reused": {"clean": 10354, "snr10": 2804},
        "new_counts": dict(progress), "research_retention": retention,
        "aug96_clean": aggregate(aug96_clean), "aug96_snr10": aggregate(aug96_snr10),
        "temporal_clean": aggregate(temporal_clean), "temporal_snr10": aggregate(temporal_snr10),
        "baseline_snr5": aggregate(baseline_snr5), "aug96_snr5": aggregate(aug96_snr5),
        "temporal_snr5": aggregate(temporal_snr5), "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN",
        "word_end_latency": "NOT_MEASURED", "new_head_c_numeric_parity": "NOT_RUN", "ssc305": "NOT_RUN",
        "model_delivered_as_product": False, **dict.fromkeys(FALSE_FLAGS, False)}
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

