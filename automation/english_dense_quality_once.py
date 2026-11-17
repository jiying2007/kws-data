#!/usr/bin/env python3
"""Final same-pool dense-CE comparison; default CLI validates source only.

The pinned temporal module supplies only pure evidence/guard utilities. Its
worker and execute entry points are never called. No numerical library is
imported until this module's guarded, single-use public worker begins.
"""
from __future__ import annotations
import argparse
import collections
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
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
SCHEMA = "english-dense-quality-once-v1"
MAIN_BEFORE = "b4982d7a2f39c4b2647a86cbf96d3701a1123b6a"
TRAINING, INCUMBENT = common.TRAINING, common.INCUMBENT
TEMPORAL = {"run_id": 44031666420, "artifact_id": 14305587199,
    "name": "english-temporal-quality-44031666420", "head_sha": MAIN_BEFORE,
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
FALSE_FLAGS, BUNDLE, RUNTIME = common.FALSE_FLAGS, common.BUNDLE, common.RUNTIME
SOURCE_HASH_KEYS = ("worker_sha256", "workflow_sha256", "common_worker_sha256")


def expected_plan():
    """The final arm is fully fixed; source hashes are filled by the caller."""
    plan = common.expected_plan()
    plan.pop("positive_min_operator")
    plan.update({
        "schema": SCHEMA, "main_before": MAIN_BEFORE, "temporal_source": TEMPORAL,
        "planned_new_conditions": 0, "planned_maximum_embedding_calls": 0,
        "maximum_embedding_calls": 0, "maximum_onnx_forwards": 0,
        "maximum_head_case_passes": 264, "expected_head_ready_outputs": 11847,
        "positive_loss": "MEAN_WINDOW_TARGET_CROSS_ENTROPY",
        "planned_reused_snr5_conditions": 77, "eligible_reused_snr5_conditions": 71,
        "reused_snr5_embedding_rows": 2649,
        "evaluation_frontend": "REUSE_FROZEN_CACHES_FOR_ALL_THREE_CONDITIONS",
        "snr5_previous_evaluation_exposed": True,
        "failed_temporal_head_used_as_warmstart": False,
        "last_additional_objective_arm": True,
        "subsequent_same_pool_loss_seed_threshold_search_allowed": False,
    })
    return plan


def validate_plan(plan):
    expected = expected_plan()
    need(set(plan) == set(expected) | set(SOURCE_HASH_KEYS), "closed final dense comparison plan")
    need(all(type(plan[key]) is type(value) and plan[key] == value for key, value in expected.items()),
         "fixed final dense comparison choices")
    for key in SOURCE_HASH_KEYS:
        need(re.fullmatch(r"[0-9a-f]{64}", plan[key]) is not None, "source SHA: " + key)
    need(plan["common_worker_sha256"] == COMMON_WORKER_SHA, "immutable common utilities")


def source_contract(plan):
    here = Path(__file__).resolve()
    paths = {"worker_sha256": here,
        "common_worker_sha256": here.with_name("english_temporal_quality_once.py"),
        "workflow_sha256": here.parent.parent / ".github/workflows/english-dense-quality-once.yml"}
    need(Path(common.__file__).resolve() == paths["common_worker_sha256"], "local immutable common module path")
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
         and event.get("before") == MAIN_BEFORE and event.get("after") == env.get("GITHUB_SHA")
         and re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")) is not None
         and re.fullmatch(r"[0-9]+", env.get("GITHUB_RUN_ID", "")) is not None,
         "exact new public main successor required")
    source_contract(plan)


def paired_comparison(reference, candidate):
    """Use the same case-level rule; give the new model an unambiguous name."""
    rows = common.paired_comparison(reference, candidate)
    for row in rows:
        row["dense_research_error"] = row.pop("temporal_research_error")
        row["dense_events"] = row.pop("temporal_events")
    return rows


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
        "retry_count": 0, "numeric_attempt": 1, "last_additional_objective_arm": True,
        **dict.fromkeys(FALSE_FLAGS, False)}, create=True)
    started = time.monotonic()

    def command(name, argv, limit=120, auth=False):
        remaining = 900 - (time.monotonic() - started)
        need(remaining > 7, "total execution deadline including cleanup reserve")
        env = dict(os.environ)
        env["KWS_DENSE_PARENT_PID"] = str(os.getpid())
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
        need(result.get("status") == "COMPLETED_WEAK_DENSE_QUALITY_RESEARCH", "worker completion")
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
         and os.environ.get("KWS_DENSE_PARENT_PID") == str(os.getppid())
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
    need(np.__version__ == "1.26.4" and torch.__version__ == "2.9.1+cpu", "fixed numeric dependency versions")
    write_json(evidence / "RUNTIME.json", {"python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "numeric_threads": 1, "embedding_runtime_loaded": False,
        "onnx_forward_execution": "NOT_RUN", "soundfile_or_corpus_decoder_used": False}, create=True)
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
    for folder in ("sources", "models", "cache", "predictions/reused"):
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
            copy_member(original, BUNDLE + "runtime/" + name, "sources/" + name, "training")
        for name in ("kws-pipeline-LICENSE", "openWakeWord-LICENSE"):
            copy_member(original, BUNDLE + "licenses/" + name, "sources/" + name, "training")
        copy_member(previous, "background_tracks/MANIFEST.json", "sources/PREVIOUS_BACKGROUND_MANIFEST.json", "incumbent")
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
    for source, destination in ((here, "sources/english_dense_quality_once.py"),
            (here.with_name("english_temporal_quality_once.py"), "sources/english_temporal_quality_once.py"),
            (here.parent.parent / ".github/workflows/english-dense-quality-once.yml", "sources/english-dense-quality-once.yml")):
        (root / destination).write_bytes(source.read_bytes())
    write_json(evidence / "INPUTS.json", {"training_source": TRAINING, "incumbent_source": INCUMBENT,
        "temporal_source": TEMPORAL, "actual_training": identity(training), "actual_incumbent": identity(incumbent),
        "actual_temporal": identity(temporal), "source_contract": source_contract(plan), "copied_members": copied,
        "old_embedding_rows_reused": {"clean": 10354, "snr10": 2804, "snr5": 2649},
        "old_frontend_recomputation": False, "new_training_frontend_calls": 0,
        "new_evaluation_frontend_calls": 0, "failed_temporal_head_used_as_warmstart": False}, create=True)
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
                    terms.append(-logp[:, label].mean())
                used.append(draw)
        need(used == previous_draws[step], "complete six-bag draw order matches")
        loss = torch.stack(terms).mean()
        need(bool(torch.isfinite(loss)), "finite dense cross-entropy training loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        losses.append(float(loss.detach())); draws.append(used)
        progress["optimizer_updates_completed"] += 1; checkpoint()
    need(draws == previous_draws and progress["torch_forward_calls_completed"] == 576, "fixed training completed exactly once")
    candidate = {"w1": model[0].weight.detach().numpy(), "b1": model[0].bias.detach().numpy(),
        "w2": model[2].weight.detach().numpy(), "b2": model[2].bias.detach().numpy(),
        "mean96": weights["mean96"].copy(), "scale96": weights["scale96"].copy()}
    np.savez(root / "models/dense-head.npz", **candidate)
    candidate = portable.load_weights(root / "models/dense-head.npz")
    normalization = {}
    for key in ("mean96", "scale96"):
        old, new = weights[key].tobytes(order="C"), candidate[key].tobytes(order="C")
        need(old == new == control[key].tobytes(order="C"), "byte-identical frozen train-only normalization")
        normalization[key] = {"bytes": len(old), "sha256": sha(old), "candidate_sha256": sha(new), "byte_identical": True}
    (root / "models/dense-head.bin").write_bytes(portable.export_blob(candidate))
    frozen_sha = identity(root / "models/dense-head.npz")["sha256"]
    phase("CANDIDATE_FROZEN")
    write_json(evidence / "CANDIDATE_FROZEN.json", {"phase_number": len(phases), "candidate": identity(root / "models/dense-head.npz"),
        "candidate_bin": identity(root / "models/dense-head.bin"), "normalization": normalization,
        "progress": dict(progress), "snr5_model_calls_before_freeze": 0,
        "all_future_evaluation_uses_this_frozen_head": True}, create=True)
    write_json(evidence / "TRAINING.json", {"optimizer_updates": 96, "training_bag_forwards": 576,
        "warmstart": identity(root / "models/baseline-head.npz"), "control": identity(root / "models/aug96-control-head.npz"),
        "candidate": identity(root / "models/dense-head.npz"), "losses": losses, "draws": draws,
        "draws_exact_match": True, "previous_training": identity(root / "sources/PREVIOUS_TRAINING.json"),
        "normalization": normalization, "positive_loss": plan["positive_loss"],
        "background_loss": plan["background_loss"], "bag_aggregation": plan["bag_aggregation"],
        "comparison": "ONLY_POSITIVE_DENSE_CE_CHANGED_EQUAL_DATA_DRAWS_AND_UPDATE_BUDGET",
        "development_checkpoint_or_threshold_selection": False, "snr5_used_for_training": False,
        "new_training_frontend_calls": 0, **dict.fromkeys(FALSE_FLAGS, False)}, create=True)

    def evaluate(name, head, rows, streams):
        need(frozen_sha is not None and identity(root / "models/dense-head.npz")["sha256"] == frozen_sha,
             "candidate remains frozen before evaluation")
        metrics = []
        with (root / "predictions" / (name + "-frames.jsonl")).open("x", encoding="utf-8") as out:
            for row in rows:
                need(progress["head_case_passes_started"] < 264, "fixed total head case pass budget")
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

    phase("EVALUATION_STARTED")
    dense_clean = evaluate("dense-clean", candidate, roster, clean_streams)
    dense_snr10 = evaluate("dense-snr10", candidate, snr10_roster, snr10_streams)
    dense_snr5 = evaluate("dense-snr5", candidate, snr5_roster, snr5_streams)
    phase("EVALUATION_COMPLETED")
    need(progress["head_case_passes_completed"] == 264 and progress["head_ready_outputs"] == 11847
         and all(progress[key] == 0 for key in ("new_embedding_calls_started", "new_embedding_calls_completed",
             "onnx_forward_started", "onnx_forward_completed", "new_training_frontend_calls",
             "new_tts_asr_phone_calls", "retry_count"))
         and progress["optimizer_updates_started"] == progress["optimizer_updates_completed"] == 96
         and progress["torch_forward_calls_started"] == progress["torch_forward_calls_completed"] == 576
         and progress["head_case_passes_started"] == progress["head_case_passes_completed"],
         "one complete final arm, zero frontend calls and exact cached evaluation work")
    aug96_clean = read_json(root / "predictions/reused/candidate-clean-events.json")
    aug96_snr10 = read_json(root / "predictions/reused/candidate-snr10-events.json")
    aug96_snr5 = read_json(root / "predictions/reused/aug96-snr5-events.json")
    baseline_snr5 = read_json(root / "predictions/reused/baseline-snr5-events.json")
    paired = {"clean": paired_comparison(aug96_clean, dense_clean),
        "snr10": paired_comparison(aug96_snr10, dense_snr10),
        "snr5": paired_comparison(aug96_snr5, dense_snr5)}
    retention = research_retention(paired, snr5_roster)
    write_json(evidence / "PAIRED_COMPARISONS.json", paired, create=True)
    write_json(evidence / "RESEARCH_RETENTION.json", retention, create=True)
    phase("COMPLETED")
    result = {"schema": SCHEMA, "status": "COMPLETED_WEAK_DENSE_QUALITY_RESEARCH",
        "run_id": os.environ["GITHUB_RUN_ID"], "head_sha": os.environ["GITHUB_SHA"], "run_attempt": 1,
        "training_source": TRAINING, "incumbent_source": INCUMBENT, "temporal_source": TEMPORAL,
        "baseline": identity(root / "models/baseline-head.npz"), "aug96_control": identity(root / "models/aug96-control-head.npz"),
        "dense_candidate": identity(root / "models/dense-head.npz"),
        "planned_new_conditions": 0, "planned_reused_snr5_conditions": 77, "eligible_reused_snr5_conditions": 71,
        "snr5_dispositions": dict(collections.Counter(row["status"] for row in snr5_receipts)),
        "source_synthetic_cases": 77, "source_synthetic_train": 50, "source_synthetic_development": 27,
        "same_source_condition_is_not_independent_audio": True, "cross_batch_voice_family_exposure": True,
        "prior_development_patterns_informed_research_hypothesis": True,
        "baseline_and_aug96_predictions_reused": True, "snr5_previous_evaluation_exposed": True,
        "failed_temporal_head_used_as_warmstart": False,
        "frozen_embedding_rows_reused": {"clean": 10354, "snr10": 2804, "snr5": 2649},
        "new_counts": dict(progress), "research_retention": retention,
        "aug96_clean": aggregate(aug96_clean), "aug96_snr10": aggregate(aug96_snr10),
        "aug96_snr5": aggregate(aug96_snr5), "baseline_snr5": aggregate(baseline_snr5),
        "dense_clean": aggregate(dense_clean), "dense_snr10": aggregate(dense_snr10), "dense_snr5": aggregate(dense_snr5),
        "last_additional_objective_arm": True, "subsequent_same_pool_loss_seed_threshold_search_allowed": False,
        "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN", "word_end_latency": "NOT_MEASURED",
        "new_head_c_numeric_parity": "NOT_RUN", "ssc305": "NOT_RUN", "model_delivered_as_product": False,
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


