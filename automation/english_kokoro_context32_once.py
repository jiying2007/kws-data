"""Exactly one public context32 diagnostic; no synthesis, admission or training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import english_kokoro_context32_source as source
import english_kokoro_context32_observe as observer

SCHEMA = source.SCHEMA
ROSTER_SHA256 = "b002435faa022ff1d5c4d847c4630e86f612d402b5def0cbf5cc5d02d99f9ddb"
OLD48_SHA256 = "61d85c6161469682aabc49bb230b66454ead1f1d96ee135112e1780d1375b8a1"
CODE_PATHS = {
    "worker_sha256": "automation/english_kokoro_context32_once.py",
    "source_code_sha256": "automation/english_kokoro_context32_source.py",
    "observer_code_sha256": "automation/english_kokoro_context32_observe.py",
    "comparator_code_sha256": "automation/english_kokoro_context32_compare.py"}
need, sha, canonical, regular, loads, once, put, within = (
    source.need, source.sha, source.canonical, source.regular, source.loads, source.once, source.put, source.within)


def digest(value):
    return sha(canonical(value))


def read(path):
    return loads(regular(path))


def validate_plan(plan, *, check_sources=True):
    need(type(plan) is dict and plan.get("schema") == SCHEMA + "/plan" and type(plan.get("enabled")) is bool,
         "CONTEXT32_PLAN_SCHEMA")
    need(plan.get("campaign_id") == source.CAMPAIGN_ID and plan.get("previous") == source.PREVIOUS
         and plan.get("derivation") == source.DERIVATION, "EXACT_PRIOR_SOURCE_AND_DERIVATION")
    need(plan.get("main_before") == (source.MAIN_BEFORE if plan["enabled"] else None), "ONE_FIXED_SUCCESSOR_ONLY")
    budgets = {"declared_inputs": 16, "maximum_calls_per_observer": 16, "maximum_total_model_calls": 32,
        "retry_count": 0, "new_tts_calls": 0, "new_text_asr_calls": 0, "new_embedding_calls": 0,
        "new_training_calls": 0, "old_observed_crop_forwards": 0,
        "preparation_deadline_seconds": 600, "observation_deadline_seconds": 1800, "combine_deadline_seconds": 900}
    need(all(type(plan.get(k)) is int and plan[k] == v for k, v in budgets.items()), "EXACT_FINITE_BUDGETS")
    need(plan.get("diagnostic_only") is True and plan.get("source_material_reused") is True
         and plan.get("training_allowed") is False and plan.get("training_snapshot_allowed") is False
         and plan.get("automatic_expansion") is False and plan.get("old_labels_modified") is False,
         "DIAGNOSTIC_REUSED_MATERIAL_SCOPE")
    need(plan.get("observers") == observer.OBSERVERS and plan.get("observer_files") == observer.OBSERVER_FILES,
         "EXACT_ORIGINAL_TWO_PHONE_MODELS")
    rows = plan.get("input_roster")
    old = plan.get("old_observed_pcm_sha256")
    need(type(rows) is list and len(rows) == 16 and digest(rows) == ROSTER_SHA256, "FROZEN_ALL16_ACTUAL_ROSTER")
    need(type(old) is list and len(old) == len(set(old)) == 48 and old == sorted(old)
         and digest(old) == OLD48_SHA256, "FROZEN_OLD48_OBSERVED_PCM_EXCLUSION")
    need(len({r["id"] for r in rows}) == len({r["source_id"] for r in rows}) == 16
         and Counter(r["preset_id"] for r in rows) == {p: 4 for p in observer.PRESETS}
         and Counter(r["requested_class"] for r in rows) == {1: 8, 2: 8}
         and all(r["old_pcm_sha256"] in old and r["mode"] == "full-carrier" for r in rows), "ALL_FOUR_BY_FOUR_UNSELECTED")
    if check_sources:
        for key, path in CODE_PATHS.items():
            need(plan.get(key) == sha(regular(ROOT / path, 256 * 1024)), "NEW_SOURCE_PIN: " + key)
        helpers = plan.get("helper_sha256")
        need(type(helpers) is dict and len(helpers) == 24 and all(helpers.get(k) == v for k, v in source.LEGACY_PINS.items()),
             "IMMUTABLE_LEGACY_SOURCE_CLOSURE")
        for name, value in helpers.items():
            need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
                 and sha(regular(within(ROOT, name), 256 * 1024)) == value, "UNCHANGED_HELPER: " + name)
        need(helpers.get("automation/english_final_ipa_xlsr_once.py") == observer.CTC_HELPER_SHA256,
             "FROZEN_CTC_FUNCTION_SOURCE")
    return plan


def identity(env=None):
    env = os.environ if env is None else env
    return {"repository": env.get("GITHUB_REPOSITORY"), "commit": env.get("GITHUB_SHA"),
        "run_id": env.get("GITHUB_RUN_ID"), "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "event": env.get("GITHUB_EVENT_NAME")}


def execution_guard(plan, env=None, event=None):
    env = os.environ if env is None else env
    validate_plan(plan)
    event = read(Path(env["GITHUB_EVENT_PATH"])) if event is None else event
    run = identity(env)
    need(plan["enabled"] is True and run["repository"] == "jiying2007/kws-data"
         and run["event"] == "push" and run["run_attempt"] == "1" and env.get("GITHUB_REF") == "refs/heads/main"
         and type(run["run_id"]) is str and re.fullmatch(r"[1-9][0-9]*", run["run_id"])
         and type(run["commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", run["commit"]), "FIRST_PUBLIC_MAIN_PUSH_ONLY")
    need(event.get("before") == plan["main_before"] == source.MAIN_BEFORE and event.get("after") == run["commit"]
         and event["after"] != event["before"] and event.get("repository", {}).get("private") is False
         and event["repository"].get("full_name") == run["repository"], "EXACT_FIRST_SUCCESSOR_OF_ACTUAL_SOURCE_MAIN")
    return run


class Deadline(BaseException):
    pass


def set_deadline(seconds):
    def stop(signum, _frame):
        raise Deadline("FINITE_STAGE_DEADLINE_OR_TERM: " + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)


def file_manifest(root):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        need(not path.is_symlink(), "NO_OUTPUT_SYMLINK")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            need("private-model" not in name.split("/") and name != "FILES.json", "NO_PRIVATE_MODEL_IN_PACKET")
            raw = regular(path, allow_empty=True)
            rows.append({"path": name, "bytes": len(raw), "sha256": sha(raw)})
    need(len(rows) <= 10000 and sum(r["bytes"] for r in rows) <= 1024 * 1024**2, "CLOSED_PACKET_BOUNDS")
    return {"schema": SCHEMA + "/files", "files": rows}


def closed_readback(root):
    root = Path(root)
    manifest = read(root / "FILES.json")
    need(manifest == file_manifest_without_self(root), "CLOSED_BYTE_MANIFEST")
    return manifest


def file_manifest_without_self(root):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        need(not path.is_symlink(), "NO_OUTPUT_SYMLINK")
        if path.is_file() and path.relative_to(root).as_posix() != "FILES.json":
            name = path.relative_to(root).as_posix()
            need("private-model" not in name.split("/"), "NO_PRIVATE_MODEL_IN_PACKET")
            raw = regular(path, allow_empty=True)
            rows.append({"path": name, "bytes": len(raw), "sha256": sha(raw)})
    need(len(rows) <= 10000 and sum(r["bytes"] for r in rows) <= 1024 * 1024**2, "CLOSED_PACKET_BOUNDS")
    return {"schema": SCHEMA + "/files", "files": rows}


def retain_sources(output, plan):
    for name in sorted(set(CODE_PATHS.values()) | set(plan["helper_sha256"])):
        target = within(output, "code/" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        put(target, regular(ROOT / name, 256 * 1024))


def source_stage(plan, inputs, output):
    run = execution_guard(plan)
    set_deadline(plan["preparation_deadline_seconds"])
    try:
        result = source.prepare(plan, inputs, output, run)
    finally:
        signal.alarm(0)
    retain_sources(Path(output), plan)
    source.readback(output, plan, allow_partial=True)
    once(Path(output) / "FILES.json", file_manifest(Path(output)))
    return result


def source_packet_readback(path, plan, *, allow_partial=False):
    closed_readback(path)
    for name in set(CODE_PATHS.values()) | set(plan["helper_sha256"]):
        need(regular(within(path, "code/" + name)) == regular(ROOT / name), "RETAINED_EXACT_EXECUTED_SOURCE")
    return source.readback(path, plan, allow_partial=allow_partial)


def observation_stage(plan, provider, source_root, output, assets):
    run = execution_guard(plan)
    manifest = source_packet_readback(source_root, plan)
    need(manifest["run"] == run, "SAME_RUN_PREPARED_INPUTS")
    output, assets = Path(output).absolute(), Path(assets).absolute()
    need(output.parent.is_dir() and assets.parent.is_dir() and not output.exists() and not assets.exists()
         and output != assets and output not in assets.parents and assets not in output.parents,
         "DISTINCT_CREATE_ONLY_OBSERVER_AND_PRIVATE_MODEL")
    set_deadline(plan["observation_deadline_seconds"])
    try:
        result = observer.run(provider=provider, source_root=Path(source_root), output_root=output,
            asset_root=assets, plan=plan, run_identity=run, input_rows=manifest["rows"])
    finally:
        signal.alarm(0)
        if assets.exists():
            need(assets.is_dir() and not assets.is_symlink(), "PRIVATE_MODEL_CLEANUP_ROOT")
            shutil.rmtree(assets)
    once(output / "FILES.json", file_manifest(output))
    return result


def copy_tree(old, new):
    old, new = Path(old), Path(new)
    new.mkdir(parents=True, exist_ok=False)
    for path in sorted(old.rglob("*")):
        need(not path.is_symlink(), "NO_HANDOFF_SYMLINK")
        if path.is_file():
            rel = path.relative_to(old).as_posix()
            target = within(new, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            put(target, regular(path))


def observer_counts(path, provider, manifest, run, manifest_sha256):
    """Only actual retained per-input markers count; absence stays unknown."""
    path = Path(path)
    ledgers = []
    for row in manifest["rows"]:
        folder = path / row["id"]
        start, returned, capture = (folder / n for n in ("FORWARD_STARTED.json", "FORWARD_RETURNED.json", "CAPTURE.json"))
        n = int(start.is_file())
        r = int(returned.is_file())
        c = int(capture.is_file())
        need(0 <= c <= r <= n <= 1, "PER_INPUT_FORWARD_RECEIPT_ORDER")
        for file in (start, returned, capture):
            if file.is_file():
                doc = read(file)
                need(doc.get("id") == row["id"] and doc.get("source_id") == row["source_id"]
                     and doc.get("provider") == provider and doc.get("run") == run
                     and doc.get("input_pcm_sha256") == row.get("pcm_sha256")
                     and doc.get("input_wav_sha256") == row.get("wav_sha256")
                     and doc.get("input_samples") == row.get("pcm_samples")
                     and doc.get("retry_count") == 0 and doc.get("forward_attempts") == 1, "ACTUAL_NEW_INPUT_MARKER_BINDING")
        ledgers.append({"id": row["id"], "forward_attempts": n, "forward_returns": r, "captures": c})
    counts = {"declared_count": 16, "actual_forward_attempts": None, "native_forward_returns": None,
        "completed": None, "retry_count": 0, "rows": ledgers,
        "retained_forward_attempts_lower_bound": sum(r["forward_attempts"] for r in ledgers),
        "retained_native_forward_returns_lower_bound": sum(r["forward_returns"] for r in ledgers),
        "retained_capture_files_lower_bound": sum(r["captures"] for r in ledgers)}
    result_path = path / "RESULT.json"
    counts["terminal_receipt_present"] = result_path.is_file()
    counts["counter_scope"] = "complete_terminal_receipt" if result_path.is_file() else "retained_marker_lower_bound"
    counts["forward_attempts_upper_bound"] = 16
    if result_path.is_file():
        result = read(result_path)
        need(result.get("provider") == provider and result.get("run") == run
             and result.get("input_manifest_sha256") == manifest_sha256
             and result.get("declared_count") == 16 and result.get("retry_count") == 0
             and all(type(result.get(k)) is int for k in ("actual_forward_attempts", "native_forward_returns", "completed"))
             and 0 <= result["completed"] <= result["native_forward_returns"] <= result["actual_forward_attempts"] <= 16
             and result["actual_forward_attempts"] == counts["retained_forward_attempts_lower_bound"]
             and result["native_forward_returns"] >= counts["retained_native_forward_returns_lower_bound"]
             and result["completed"] == counts["retained_capture_files_lower_bound"],
             "SAVED_PROVIDER_COUNTERS")
        # The comparator verifies every ledger row, including an actual return
        # whose subsequent FORWARD_RETURNED file failed to be written.
        counts.update({k: result[k] for k in ("actual_forward_attempts", "native_forward_returns", "completed")})
        counts["forward_attempts_upper_bound"] = result["actual_forward_attempts"]
    return counts


def combine(plan, source_root, observers_root, output):
    run = execution_guard(plan)
    manifest = source_packet_readback(source_root, plan, allow_partial=True)
    need(manifest["run"] == run, "SAME_RUN_COMBINE_INPUTS")
    output = source.new_directory(output)
    # Source roots remain self-contained. Old source/probe ZIPs are not rebuilt.
    copy_tree(source_root, output / "prepared")
    observers = {}
    counts = {}
    for provider in observer.PROVIDERS:
        src = Path(observers_root) / ("kokoro-context32-observer-" + provider)
        dest = output / "observers" / provider
        if src.is_dir():
            if (src / "FILES.json").is_file():
                closed_readback(src)
            copy_tree(src, dest)
        else:
            dest.mkdir(parents=True, exist_ok=False)
        observers[provider] = dest
        counts[provider] = observer_counts(dest, provider, manifest, run, sha(regular(Path(source_root) / "INPUT_MANIFEST.json")))
    spec = importlib.util.spec_from_file_location("context32_pure_diagnostic_compare", HERE / "english_kokoro_context32_compare.py")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    comparison = comparator.compare(output / "prepared", observers, plan)
    comparison_pin = once(output / "COMPARISON.json", comparison)
    for provider in observer.PROVIDERS:
        need(all(comparison["providers"][provider].get(k) == counts[provider][k]
                 for k in ("actual_forward_attempts", "native_forward_returns", "completed")), "INDEPENDENT_COMPARISON_COUNTERS")
    known = all(c["terminal_receipt_present"] for c in counts.values())
    attempts = sum(c["actual_forward_attempts"] for c in counts.values()) if known else None
    returns = sum(c["native_forward_returns"] for c in counts.values()) if known else None
    completed = sum(c["completed"] for c in counts.values()) if known else None
    lower = sum(c["retained_forward_attempts_lower_bound"] for c in counts.values())
    upper = sum(c["forward_attempts_upper_bound"] for c in counts.values())
    need(0 <= lower <= upper <= 32 and (not known or 0 <= completed <= returns <= attempts <= 32), "TOTAL32_NO_RETRY_BOUND")
    result = {"schema": SCHEMA, "stage": "combine", "run": run, "status": "DIAGNOSTIC_PACKET_RETAINED",
        "declared_inputs": 16, "declared_observation_slots": 32, "observer_counts": counts,
        "actual_forward_attempts": attempts, "native_forward_returns": returns, "completed": completed,
        "counter_scope": "complete_terminal_receipts" if known else "actual_count_unknown_retained_bounds",
        "retained_forward_attempts_lower_bound": lower, "forward_attempts_upper_bound": upper,
        "retained_native_forward_returns_lower_bound": sum(c["retained_native_forward_returns_lower_bound"] for c in counts.values()),
        "retained_capture_files_lower_bound": sum(c["retained_capture_files_lower_bound"] for c in counts.values()),
        "maximum_total_model_calls": 32, "retry_count": 0,
        "source_material_reused": True, "new_derivative_inputs": sum(r.get("preparation_status") == "COMPLETE" for r in manifest["rows"]),
        "phone_forwards_on_reused_material": attempts, "phone_forwards_on_reused_material_lower_bound": lower,
        "phone_forwards_on_reused_material_upper_bound": upper, "old_observed_crop_forwards": 0,
        "new_tts_calls": 0, "new_text_asr_calls": 0, "new_embedding_calls": 0, "new_training_calls": 0,
        "new_model_calls_during_join": 0, "diagnostic_only": True, "old_labels_modified": False,
        "original_probe_decision": "STOP_DIAGNOSTIC_NO_EXPANSION", "training_allowed": False,
        "training_snapshot_allowed": False, "automatic_expansion": False, "high_quality_model_passed": False,
        "comparison_sha256": comparison_pin["sha256"]}
    once(output / "EXECUTION_PLAN.json", plan)
    once(output / "RESULT.json", result)
    once(output / "FILES.json", file_manifest(output))
    return result


def source_preflight(plan):
    validate_plan(plan)
    observer.source_preflight()
    return {"schema": SCHEMA, "source_check": "PASS", "enabled": plan["enabled"],
        "declared_new_full_inputs": 16, "maximum_phone_forwards": 32, "new_model_calls": 0,
        "new_resampling_calls": 0, "new_network_calls": 0, "training_allowed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-check", action="store_true")
    group.add_argument("--guard", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--source-readback", action="store_true")
    group.add_argument("--observe", choices=observer.PROVIDERS)
    group.add_argument("--combine", action="store_true")
    parser.add_argument("--inputs")
    parser.add_argument("--source")
    parser.add_argument("--observers")
    parser.add_argument("--output")
    parser.add_argument("--assets")
    args = parser.parse_args(argv)
    plan = validate_plan(read(args.plan))
    if args.source_check:
        result = source_preflight(plan)
    elif args.guard:
        result = {"schema": SCHEMA, "run": execution_guard(plan), "allowed": True, "maximum_phone_forwards": 32}
    elif args.prepare:
        need(args.inputs and args.output, "PREP_ARGUMENTS")
        result = source_stage(plan, args.inputs, args.output)
    elif args.source_readback:
        need(args.source, "READBACK_SOURCE")
        manifest = source_packet_readback(args.source, plan, allow_partial=True)
        count = sum(r.get("preparation_status") == "COMPLETE" for r in manifest["rows"])
        result = {"schema": SCHEMA, "source_ready": count == 16, "declared_count": 16, "complete_inputs": count, "new_model_calls": 0}
    elif args.observe:
        need(args.source and args.output and args.assets, "OBSERVER_ARGUMENTS")
        result = observation_stage(plan, args.observe, args.source, args.output, args.assets)
    else:
        need(args.source and args.observers and args.output, "COMBINE_ARGUMENTS")
        set_deadline(plan["combine_deadline_seconds"])
        try:
            result = combine(plan, args.source, args.observers, args.output)
        finally:
            signal.alarm(0)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.prepare:
        return 0 if result["status"] == "COMPLETE_NEW_INPUTS" else 1
    if args.observe:
        return 0 if result["status"] == "COMPLETE_DIAGNOSTIC" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
