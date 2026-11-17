#!/usr/bin/env python3
"""One source-recovery continuation of the unstarted 10 dB numerical study.

Import and default CLI are source-only. Actual model calls require a separately
guarded public Actions push, one attempt and one supervised worker process.
"""
from __future__ import annotations
import argparse
import ast
from array import array
import collections
import hashlib
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
import tarfile
import time
import wave
import zipfile

REPO = "jiying2007/kws-data"
MAIN_BEFORE = "b1b70a8a0ab08cdde87868788e8a18425c7e5cca"
SCHEMA = "english-quality-snr10-source-continuation-v1"
TRAINING = {"run_id": 43867764366, "artifact_id": 14225893574,
            "name": "snapshot160-weak-embedding-research-43867764366",
            "head_sha": "d8be1c24b12cac0ba554e9a4a41e4b6ee0db18db",
            "bytes": 61290521,
            "sha256": "2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1"}
CORPUS = {"url": "https://openslr.elda.org/resources/12/dev-clean.tar.gz", "bytes": 337926286,
          "sha256": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"}
PINS = {
    "actual-embedding-streams.npz": "32db279b822f5e5787b5eabe0afaf7766d3971c3200c2aa3ffcb3d165256e5d2",
    "PER_CASE_EVENTS.json": "b186b75f4f15d7f91eda79be1567f6c1b5f93e94684ca9b9b5f145c47d384bdf",
    "FEATURE_EVIDENCE.json": "944e2c143d9b2fd9784aaabb5f650f42fee69d92bbc90b2c3d4a01c380c57a8c",
    "STREAM_ROSTER.json": "53646ce22780e8647e5654334c633a3f4dc8c1dd9a78d0856a1877b99d75b12a",
    "SOURCE_SNAPSHOT.zip": "01512268a5400e11872b395df53ca17bd1f278229673f9a68a277fbcaa6e1700",
    "train/causal-head.npz": "6d7677b674f4a5e324aa916c5b777792b641fdfd61af8edc2f4e490275d600c4",
}
BUNDLE = "english-snapshot160-embedding-research-bundle/"
FALSE_FLAGS = ("formal_training_admitted", "training_admitted", "human_gold", "trusted_negative",
               "pronunciation_verified", "model_promotion_allowed", "qualification_allowed")
RUNTIME = ("weak_embedding_contract.py", "causal_embedding.py", "portable_head.py")
ORIGINAL_WORKER_SHA = "76f2ecd9a1269a587d4ff35971e338533e0d70d588780cc47e84910f8bb70cd9"
ORIGINAL_PLAN_SHA = "899ab737de1c4034ecabade7e88de81531516dc21e97ccfe61a2af1b2772c76d"
ORIGINAL_WORKFLOW_SHA = "5bad832e8f0297c95a744efd43ac613bcf557ead6c9acc1f748f26b32bfb7b26"
FIRST_FAILURE = {"run_id": 44019143948, "artifact_id": 14300895359,
    "name": "english-quality-snr10-44019143948", "head_sha": "b1b70a8a0ab08cdde87868788e8a18425c7e5cca",
    "bytes": 6346, "sha256": "43ad533ed64eb25421fd24478c69e1cb26f5412f864229f902c0641b498b47e3"}
FAILED_MEMBERS = {"evidence/FAILED_NO_RETRY.json", "evidence/STARTED.json", "evidence/PLAN.json",
    "evidence/SUPERVISOR_EXIT.json", "evidence/corpus-download.exit.json", "evidence/corpus-download.stderr",
    "evidence/corpus-download.stdout", "evidence/source-artifact.exit.json", "evidence/source-artifact.stderr",
    "evidence/source-artifact.stdout", "evidence/source-run.exit.json", "evidence/source-run.stderr",
    "evidence/source-run.stdout", "evidence/training-zip.stderr"}
UNCHANGED_FUNCTIONS = ("pcm_values", "pcm_bytes", "mix_pcm16", "select_noise_track", "summarize_case",
                       "candidate_decision", "validate_zip", "archive_read", "aggregate", "worker")


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


def validate_plan(plan):
    need(plan.get("schema") == SCHEMA and plan.get("main_before") == MAIN_BEFORE, "fixed plan identity")
    expected = {"snr_db": 10.0, "maximum_embedding_calls": 4000, "maximum_optimizer_updates": 96,
                "maximum_execution_seconds": 900, "planned_new_conditions": 77,
                "seed": 20261117, "retry_count": 0, "new_tts_asr_phone_calls": 0,
                "threshold": 0.8, "confirm_embeddings": 3, "cooldown_embeddings": 13}
    need(all(type(plan.get(k)) is type(v) and plan[k] == v for k, v in expected.items()), "fixed experiment budget")
    need(plan.get("training_source") == TRAINING and plan.get("corpus") == CORPUS, "immutable source pins")
    need(all(plan.get(k) is False for k in FALSE_FLAGS), "research-only authority")
    need(plan.get("candidate_trigger") == "TRAIN_ROLE_ERRORS_ONLY_AND_FIXED_MINIMUM_COVERAGE"
         and plan.get("development_tuning") is False and plan.get("normalization") == "FROZEN_ORIGINAL_TRAIN_ONLY"
         and plan.get("mix_scope") == "ORIGINAL_UNTRIMMED_CLIP_ONLY"
         and plan.get("overflow_policy") == "EXCLUDE_WITHOUT_REPLACEMENT_OR_CLIPPING", "fixed research semantics")
    need(re.fullmatch(r"[0-9a-f]{64}", plan.get("worker_sha256", "")) is not None, "worker pin")
    need(plan.get("first_failure") == FIRST_FAILURE and plan.get("first_failure_consumed") is True
         and type(plan.get("source_acquisition_attempt")) is int and plan["source_acquisition_attempt"] == 2
         and type(plan.get("numeric_attempt")) is int and plan["numeric_attempt"] == 1
         and plan.get("original_worker_sha256") == ORIGINAL_WORKER_SHA
         and plan.get("original_plan_sha256") == ORIGINAL_PLAN_SHA, "explicit source continuation, no numeric retry")


def validate_original_sources():
    """The original once files stay immutable; all numerical function text is identical."""
    here = Path(__file__).resolve().parent
    original = (here / "english_quality_snr10_once.py").read_bytes()
    original_plan = (here / "english_quality_snr10_plan.json").read_bytes()
    original_workflow = (here.parent / ".github/workflows/english-quality-snr10-once.yml").read_bytes()
    need(sha(original) == ORIGINAL_WORKER_SHA and sha(original_plan) == ORIGINAL_PLAN_SHA
         and sha(original_workflow) == ORIGINAL_WORKFLOW_SHA, "original failed once sources remain byte-identical")
    texts = [original.decode(), Path(__file__).read_text()]
    functions = [{n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)} for text in texts]
    need(all(functions[0][name] == functions[1][name] for name in UNCHANGED_FUNCTIONS), "unchanged model worker and numerical function source")
    return {"original_worker_sha256": ORIGINAL_WORKER_SHA, "original_plan_sha256": ORIGINAL_PLAN_SHA,
            "original_workflow_sha256": ORIGINAL_WORKFLOW_SHA,
            "identical_function_sha256": {name: sha(functions[0][name].encode()) for name in UNCHANGED_FUNCTIONS}}


def verify_failed_layout(members):
    """Pure layout/receipt semantics; the production gate first fixes whole ZIP bytes."""
    need(set(members) == FAILED_MEMBERS, "exact 14-member failure evidence, no worker invocation files")
    def record(name):
        return json.loads(members["evidence/" + name])
    failed, started, plan = record("FAILED_NO_RETRY.json"), record("STARTED.json"), record("PLAN.json")
    need(sha(members["evidence/PLAN.json"]) == ORIGINAL_PLAN_SHA, "exact unmodified original quality plan")
    need(started.get("head_sha") == FIRST_FAILURE["head_sha"] and started.get("run_id") == str(FIRST_FAILURE["run_id"])
         and started.get("run_attempt") == 1 and started.get("training") == TRAINING, "first failure execution binding")
    need(failed.get("status") == "FAILED_NO_RETRY" and failed.get("partial_progress", "MISSING") is None
         and failed.get("retry_count") == 0 and failed.get("error_type") == "ValueError"
         and failed.get("error") == "child command failed" and all(failed.get(k) is False for k in FALSE_FLAGS),
         "failed before any worker progress")
    download = record("corpus-download.exit.json")
    expected_argv = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "0",
        "--connect-timeout", "30", "--max-time", "300", "--output",
        "/home/runner/work/_temp/english-quality-snr10/downloads/dev-clean.tar.gz",
        "https://www.openslr.org/resources/12/dev-clean.tar.gz"]
    need(download.get("argv") == expected_argv and download.get("returncode") == 28
         and download.get("error") == "ValueError" and download.get("group_cleanup_checked_after_leader_exit") is True
         and members["evidence/corpus-download.stderr"] == b"curl: (28) SSL connection timeout\n",
         "exact first TLS source acquisition failure")
    need(members["evidence/corpus-download.stdout"] == b"" and members["evidence/training-zip.stderr"] == b"",
         "no hidden failed-command output")
    for name in ("source-artifact", "source-run"):
        need(record(name + ".exit.json").get("returncode") == 0 and members["evidence/" + name + ".stderr"] == b"",
             "original training metadata was acquired")
    need(plan.get("worker_sha256") == ORIGINAL_WORKER_SHA and plan.get("maximum_optimizer_updates") == 96
         and plan.get("maximum_embedding_calls") == 4000 and plan.get("snr_db") == 10.0,
         "original numerical budget remains unconsumed")
    return {"first_failure_consumed": True, "first_failure": FIRST_FAILURE, "closed_member_count": 14,
        "failure_stage": "CORPUS_TLS_CONNECTION", "numeric_worker_invocations_in_first_run": 0,
        "first_run_new_embedding_calls": 0, "first_run_optimizer_updates": 0,
        "proof_basis": "Exact original source invokes model-worker only after successful corpus command and identity check; its stdout/stderr/exit and all worker start/progress files are absent from the authenticated complete evidence roster.",
        "source_acquisition_attempt": 2, "numeric_attempt": 1}


def verify_failed_proof(raw):
    need(type(raw) is bytes and len(raw) == FIRST_FAILURE["bytes"] and sha(raw) == FIRST_FAILURE["sha256"],
         "fixed complete first-failure ZIP")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        validate_zip(archive, maximum=1024 * 1024)
        return verify_failed_layout({name: archive.read(name) for name in archive.namelist()})


def assert_same_research_parameters(plan):
    original = read_json(Path(__file__).with_name("english_quality_snr10_plan.json"))
    shared = set(original) - {"schema", "main_before", "worker_sha256", "corpus"}
    need(all(plan.get(k) == original[k] for k in shared)
         and plan["corpus"]["bytes"] == original["corpus"]["bytes"]
         and plan["corpus"]["sha256"] == original["corpus"]["sha256"], "all quality parameters and corpus bytes unchanged")


def execution_guard(plan, env=None, event=None):
    env = os.environ if env is None else env
    event = read_json(env["GITHUB_EVENT_PATH"]) if event is None else event
    need(env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_REPOSITORY") == REPO
         and env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == "refs/heads/main"
         and env.get("GITHUB_RUN_ATTEMPT") == "1", "public first-attempt main push required")
    repo = event.get("repository", {})
    need(repo.get("full_name") == REPO and repo.get("private") is False
         and event.get("before") == plan["main_before"] and event.get("after") == env.get("GITHUB_SHA")
         and re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")) is not None,
         "exact new main successor required")
    need(sha(Path(__file__).read_bytes()) == plan["worker_sha256"], "executed source SHA")


def pcm_values(raw):
    need(type(raw) is bytes and len(raw) > 0 and len(raw) % 2 == 0, "PCM16LE bytes")
    values = array("h")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def pcm_bytes(values):
    result = array("h", values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def mix_pcm16(source_pcm, noise_pcm, snr_db=10.0):
    """Full-clip RMS, no trimming/fade/gain/clipping/replacement; PCM rounding only."""
    need(type(snr_db) is float and snr_db == 10.0, "one fixed SNR")
    source, noise = pcm_values(source_pcm), pcm_values(noise_pcm)
    need(len(source) == len(noise), "matching mix length")
    ps = math.fsum(float(x) ** 2 for x in source) / len(source)
    pn = math.fsum(float(x) ** 2 for x in noise) / len(noise)
    receipt = {"snr_definition": "full-original-clip-rms-before-mixture", "target_snr_db": snr_db,
               "samples": len(source), "signal_power": ps, "noise_power": pn,
               "source_pcm_sha256": sha(source_pcm), "noise_pcm_sha256": sha(noise_pcm),
               "gain_on_signal": 1.0, "clipped_samples": 0, "fade_samples": 0}
    if ps == 0 or pn == 0:
        return None, {**receipt, "status": "MIX_ZERO_ENERGY_EXCLUDED"}
    gain = math.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    values = [round(float(s) + gain * float(n)) for s, n in zip(source, noise)]
    peak = max(abs(x) for x in values)
    receipt.update(noise_gain=gain, unbounded_peak=peak)
    if min(values) < -32768 or max(values) > 32767:
        return None, {**receipt, "status": "MIX_HEADROOM_EXCLUDED"}
    mixed = pcm_bytes(values)
    actual_noise_power = math.fsum(float(m - s) ** 2 for m, s in zip(values, source)) / len(source)
    if actual_noise_power == 0:
        return None, {**receipt, "status": "MIX_QUANTIZED_NOISE_ZERO_EXCLUDED"}
    actual_snr = 10 * math.log10(ps / actual_noise_power)
    # Reject an unusably tiny signal rather than assert nominal SNR after rounding.
    if abs(actual_snr - snr_db) > 0.05:
        return None, {**receipt, "status": "MIX_QUANTIZATION_SNR_EXCLUDED", "actual_snr_db": actual_snr}
    return mixed, {**receipt, "status": "MIXED", "actual_snr_db": actual_snr,
                   "mixed_pcm_sha256": sha(mixed)}


def select_noise_track(case_id, role, frames, tracks):
    need(role in ("train", "development") and type(frames) is int and frames > 0, "mix role/length")
    eligible = sorted([t for t in tracks if t["role"] == role and t["frames"] >= frames], key=lambda x: x["id"])
    need(eligible and len({t["id"] for t in eligible}) == len(eligible), "same-role noise pool")
    digest = sha(("english-quality-snr10-v1/" + case_id).encode())
    track = eligible[int(digest[:16], 16) % len(eligible)]
    start = int(digest[16:32], 16) % (track["frames"] - frames + 1)
    return track, start


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


def candidate_decision(baseline_noisy_metrics, mixed_roster):
    need(len({r["id"] for r in mixed_roster}) == len(mixed_roster)
         and len({r["id"] for r in baseline_noisy_metrics}) == len(baseline_noisy_metrics), "unique noisy case identities")
    need({r["id"] for r in mixed_roster} == {r["id"] for r in baseline_noisy_metrics}, "complete noisy diagnostics")
    by_id = {r["id"]: r for r in mixed_roster}
    need(all((r["role"], r["weak_class"]) == (by_id[r["id"]]["role"], by_id[r["id"]]["weak_class"])
             and type(r["research_error"]) is bool for r in baseline_noisy_metrics), "diagnostic roles and boolean errors")
    train = [r for r in mixed_roster if r["role"] == "train"]
    metrics = {r["id"]: r for r in baseline_noisy_metrics if r["role"] == "train"}
    need(len(metrics) == len(train) and set(metrics) == {r["id"] for r in train}, "complete train diagnostics")
    coverage = {}
    for label in range(3):
        rows = [r for r in train if r["weak_class"] == label]
        families = sorted({r["source_family_id"] for r in rows})
        coverage[str(label)] = {"cases": len(rows), "families": families,
                               "passed": len(rows) >= (8 if label == 0 else 2) and (label == 0 or len(families) >= 2)}
    errors = sum(bool(r["research_error"]) for r in metrics.values())
    valid = all(r["passed"] for r in coverage.values())
    reason = ("BASELINE_ONLY_NO_VALID_AUGMENTED_TRAINING" if not valid else
              "BASELINE_ONLY_NO_TRAIN_ERROR" if errors == 0 else "FIXED_TRAIN_ERROR_TRIGGERED_CANDIDATE")
    return {"execute": valid and errors > 0, "reason": reason, "train_errors": errors,
            "coverage": coverage, "development_used_for_decision": False}


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


def execute(args, plan):
    execution_guard(plan)
    source_equality = validate_original_sources()
    assert_same_research_parameters(plan)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    evidence, downloads = root / "evidence", root / "downloads"
    evidence.mkdir(); downloads.mkdir()
    write_json(evidence / "PLAN.json", plan, create=True)
    write_json(evidence / "STARTED.json", {"run_id": os.environ["GITHUB_RUN_ID"],
        "head_sha": os.environ["GITHUB_SHA"], "run_attempt": 1, "training": TRAINING,
        "first_failure_consumed": True, "source_acquisition_attempt": 2, "numeric_attempt": 1,
        "retry_count": 0, **dict.fromkeys(FALSE_FLAGS, False)}, create=True)
    started = time.monotonic()
    def command(name, argv, limit=120, auth=False):
        remaining = 900 - (time.monotonic() - started)
        need(remaining > 7, "total execution deadline including cleanup reserve")
        env = dict(os.environ)
        if not auth:
            for key in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"):
                env.pop(key, None)
        out = downloads / (name + ".stdout") if name in ("training-zip", "first-failure-zip") else evidence / (name + ".stdout")
        supervised_command(argv, out, evidence / (name + ".stderr"), min(limit, remaining - 7), env)
        return out
    try:
        prior_run = read_json(command("first-failure-run", ["gh", "api", f"repos/{REPO}/actions/runs/{FIRST_FAILURE['run_id']}"], auth=True))
        prior_artifact = read_json(command("first-failure-artifact", ["gh", "api", f"repos/{REPO}/actions/artifacts/{FIRST_FAILURE['artifact_id']}"], auth=True))
        need(prior_run["status"] == "completed" and prior_run["conclusion"] == "failure"
             and prior_run["id"] == FIRST_FAILURE["run_id"] and prior_run["head_sha"] == FIRST_FAILURE["head_sha"]
             and prior_run["run_attempt"] == 1 and prior_run["event"] == "push" and prior_run["head_branch"] == "main"
             and prior_run["repository"]["full_name"] == REPO and prior_run["repository"]["private"] is False,
             "first source failure is a fixed terminal public run")
        need(prior_artifact["id"] == FIRST_FAILURE["artifact_id"] and prior_artifact["name"] == FIRST_FAILURE["name"]
             and prior_artifact["size_in_bytes"] == FIRST_FAILURE["bytes"] and prior_artifact["expired"] is False
             and prior_artifact["workflow_run"]["id"] == FIRST_FAILURE["run_id"]
             and prior_artifact["workflow_run"]["head_sha"] == FIRST_FAILURE["head_sha"]
             and prior_artifact["digest"] == "sha256:" + FIRST_FAILURE["sha256"], "first failure artifact identity")
        failed_download = command("first-failure-zip", ["gh", "api", f"repos/{REPO}/actions/artifacts/{FIRST_FAILURE['artifact_id']}/zip"], 120, True)
        failed_path = downloads / "failed-first-run.zip"
        failed_download.rename(failed_path)
        proof = verify_failed_proof(failed_path.read_bytes())
        (evidence / "first-failure.zip").write_bytes(failed_path.read_bytes())
        write_json(evidence / "SOURCE_RECOVERY_GATE.json", {**proof, "source_equality": source_equality,
            "public_terminal_metadata_checked": True, "mirror": CORPUS["url"],
            "quality_parameters_unchanged": True}, create=True)
        run = read_json(command("source-run", ["gh", "api", f"repos/{REPO}/actions/runs/{TRAINING['run_id']}"], auth=True))
        artifact = read_json(command("source-artifact", ["gh", "api", f"repos/{REPO}/actions/artifacts/{TRAINING['artifact_id']}"], auth=True))
        need(run["status"] == "completed" and run["conclusion"] == "success" and run["head_sha"] == TRAINING["head_sha"]
             and run["run_attempt"] == 1 and run["event"] == "push" and run["head_branch"] == "main"
             and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False, "original public training execution")
        need(artifact["id"] == TRAINING["artifact_id"] and artifact["name"] == TRAINING["name"]
             and artifact["size_in_bytes"] == TRAINING["bytes"] and artifact["expired"] is False
             and artifact["workflow_run"]["id"] == TRAINING["run_id"]
             and artifact["workflow_run"]["head_sha"] == TRAINING["head_sha"]
             and artifact["digest"] == "sha256:" + TRAINING["sha256"], "original training artifact")
        path = command("training-zip", ["gh", "api", f"repos/{REPO}/actions/artifacts/{TRAINING['artifact_id']}/zip"], 240, True)
        training = downloads / "training.zip"
        path.rename(training)
        need(identity(training) == {k: TRAINING[k] for k in ("bytes", "sha256")}, "training ZIP identity")
        corpus = downloads / "dev-clean.tar.gz"
        command("corpus-download", ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "0",
                                     "--connect-timeout", "30", "--max-time", "300", "--output", corpus, CORPUS["url"]], 300)
        need(identity(corpus) == {k: CORPUS[k] for k in ("bytes", "sha256")}, "corpus archive identity")
        write_json(evidence / "SOURCE_ACQUIRED.json", {"source_acquisition_attempt": 2, "numeric_attempt": 1,
            "first_failure_consumed": True, "corpus": CORPUS, "actual": identity(corpus),
            "mirror_requests_in_this_continuation": 1, "automatic_fallbacks": 0}, create=True)
        command("background-prepare", [sys.executable, "-B", Path(__file__).resolve(), "--plan", args.plan.resolve(),
                                        "--prepare-noise", "--output", root], 180)
        command("model-worker", [sys.executable, "-B", Path(__file__).resolve(), "--plan", args.plan.resolve(),
                                  "--worker", "--output", root], 900)
        result = read_json(evidence / "RESULT.json")
        need(result.get("status") == "COMPLETED_WEAK_QUALITY_RESEARCH", "worker completion")
    except BaseException as exc:
        write_json(evidence / "FAILED_NO_RETRY.json", {"status": "FAILED_NO_RETRY", "error_type": type(exc).__name__,
            "error": str(exc), "wall_seconds": time.monotonic() - started, "retry_count": 0,
            "partial_progress": read_json(evidence / "PROGRESS.json") if (evidence / "PROGRESS.json").exists() else None,
            **dict.fromkeys(FALSE_FLAGS, False)})
        raise
    finally:
        write_json(evidence / "SUPERVISOR_EXIT.json", {"wall_seconds": time.monotonic() - started,
                                                     "maximum_seconds": 900, "retry_count": 0})


def prepare_noise(args, plan):
    """Materialize historical PCM only; no feature/model library is imported."""
    execution_guard(plan)
    root = args.output.resolve()
    verify_failed_proof((root / "downloads/failed-first-run.zip").read_bytes())
    import numpy as np
    import soundfile as sf
    need(np.__version__ == "1.26.4" and sf.__version__ == "0.13.1", "original PCM decoder versions")
    training, corpus = root / "downloads/training.zip", root / "downloads/dev-clean.tar.gz"
    need(identity(training) == {k: TRAINING[k] for k in ("bytes", "sha256")}
         and identity(corpus) == {k: CORPUS[k] for k in ("bytes", "sha256")}, "authenticated source-only preparation inputs")
    destination = root / "background_tracks"
    destination.mkdir(exist_ok=False)
    write_json(root / "evidence/NOISE_PREPARATION_STARTED.json", {"new_model_calls": 0, "source_acquisition_attempt": 2,
        "first_failure_consumed": True, "independent_source_count_increment": 0}, create=True)
    with zipfile.ZipFile(training) as archive:
        validate_zip(archive)
        source = json.loads(archive.read("LIBRISPEECH_SOURCES.json"))
    tracks = source["actual_research_tracks"]
    need(len(tracks) == 40 and collections.Counter(t["role"] for t in tracks) == {"train": 30, "development": 10},
         "historical forty-track roles")
    originals = {r["id"]: r for r in source["decoded_originals"]}
    required = {f["source_id"] for t in tracks for f in t["fragments"]}
    decoded = {}
    with tarfile.open(corpus, "r:gz") as archive:
        for name in sorted(required):
            info = archive.getmember(name)
            need(info.isfile() and info.size < 16 * 1024**2, "regular original noise FLAC")
            raw = archive.extractfile(info).read()
            original = originals[name]
            need(sha(raw) == original["flac_sha256"], "original noise FLAC hash")
            with sf.SoundFile(io.BytesIO(raw)) as sound:
                need(sound.samplerate == 16000 and sound.channels == 1 and sound.subtype == "PCM_16"
                     and sound.frames == original["frames"], "original noise FLAC geometry")
                pcm = sound.read(dtype="int16", always_2d=False).astype("<i2", copy=False).tobytes()
            need(sha(pcm) == original["pcm_sha256"], "original noise decoded PCM hash")
            decoded[name] = pcm
    files = {}
    for track in tracks:
        pieces, offset = [], 0
        for fragment in track["fragments"]:
            need(fragment["track_start_sample"] == offset
                 and sha(decoded[fragment["source_id"]]) == fragment["source_pcm_sha256"], "historical source fragment identity")
            start, count = fragment["source_start_sample"], fragment["samples"]
            part = decoded[fragment["source_id"]][2 * start:2 * (start + count)]
            need(len(part) == 2 * count, "historical source fragment length")
            pieces.append(part); offset += count
        raw = b"".join(pieces)
        need(offset == track["frames"] == 256000 and sha(raw) == track["pcm_sha256"], "exact historical sixteen-second PCM track")
        name = track["id"] + ".pcm"
        (destination / name).write_bytes(raw)
        files[name] = {"bytes": len(raw), "sha256": sha(raw), "role": track["role"], "speaker": track["speaker"],
                       "fragments": track["fragments"]}
    write_json(destination / "MANIFEST.json", {"status": "FORTY_HISTORICAL_PCM_TRACKS_PRESERVED",
        "files": files, "pcm_bytes": sum(f["bytes"] for f in files.values()), "source_training": TRAINING,
        "corpus": CORPUS, "license": "CC-BY-4.0", "attribution": "LibriSpeech by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur; source audiobooks from LibriVox.",
        "official_resource_page": "https://www.openslr.org/12/", "sample_rate": 16000, "channels": 1,
        "new_model_calls": 0, "new_audio_generated": False, "independent_source_count_increment": 0}, create=True)
    write_json(root / "evidence/NOISE_PREPARATION_COMPLETED.json", {"tracks": 40, "pcm_bytes": 20480000,
        "decoded_original_flacs": len(required), "new_model_calls": 0,
        "manifest": identity(destination / "MANIFEST.json"), "historical_pcm_identities_all_matched": True}, create=True)


def worker(args, plan):
    """All actual feature, head and optimizer execution is isolated here."""
    execution_guard(plan)
    import numpy as np
    import soundfile as sf
    import torch
    import onnx
    import onnxruntime as ort
    need(np.__version__ == "1.26.4" and sf.__version__ == "0.13.1" and torch.__version__ == "2.9.1+cpu", "fixed numeric dependencies")
    need(onnx.__version__ == "1.17.0" and ort.__version__ == "1.20.1", "fixed ONNX runtime dependencies")
    root = args.output.resolve()
    evidence = root / "evidence"
    write_json(evidence / "WORKER_STARTED.json", {"pid": os.getpid(), "run_id": os.environ["GITHUB_RUN_ID"]}, create=True)
    write_json(evidence / "RUNTIME.json", {"python": sys.version, "numpy": np.__version__,
        "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__, "torch": torch.__version__,
        "onnx": onnx.__version__, "onnxruntime": ort.__version__, "execution_provider": "CPUExecutionProvider",
        "numeric_threads": 1}, create=True)
    progress = {"new_embedding_calls_started": 0, "new_embedding_calls_completed": 0,
                "onnx_forward_started": 0, "onnx_forward_completed": 0,
                "head_case_passes_started": 0, "head_case_passes_completed": 0,
                "head_ready_outputs": 0, "optimizer_updates_started": 0, "optimizer_updates_completed": 0,
                "torch_forward_calls_started": 0, "torch_forward_calls_completed": 0,
                "new_tts_asr_phone_calls": 0, "retry_count": 0}
    def checkpoint():
        write_json(evidence / "PROGRESS.json", progress)
    checkpoint()
    for folder in ("sources", "models", "cache", "audio", "predictions"):
        (root / folder).mkdir(exist_ok=False)
    training = root / "downloads/training.zip"
    need(identity(training) == {k: TRAINING[k] for k in ("bytes", "sha256")}, "worker training archive")
    with zipfile.ZipFile(training) as saved:
        validate_zip(saved)
        roster = json.loads(archive_read(saved, "STREAM_ROSTER.json"))
        feature_evidence = json.loads(archive_read(saved, "FEATURE_EVIDENCE.json"))
        old_events = json.loads(archive_read(saved, "PER_CASE_EVENTS.json"))
        source_rows = json.loads(saved.read("LIBRISPEECH_SOURCES.json"))
        need(len(roster) == len(feature_evidence) == len(old_events) == 117, "original 117 cases")
        cache_raw = archive_read(saved, "actual-embedding-streams.npz")
        (root / "cache/clean-original.npz").write_bytes(cache_raw)
        for name in ("STREAM_ROSTER.json", "FEATURE_EVIDENCE.json", "PER_CASE_EVENTS.json", "LIBRISPEECH_SOURCES.json", "SELECTION.json"):
            (root / "sources" / name).write_bytes(archive_read(saved, name))
        for name in RUNTIME:
            (root / "sources" / name).write_bytes(saved.read(BUNDLE + "runtime/" + name))
        for name in ("melspectrogram.onnx", "embedding_model.onnx"):
            (root / "models" / name).write_bytes(saved.read(BUNDLE + "models/" + name))
        (root / "models/baseline-head.npz").write_bytes(archive_read(saved, "train/causal-head.npz"))
        for name in ("kws-pipeline-LICENSE", "openWakeWord-LICENSE"):
            (root / "sources" / name).write_bytes(saved.read(BUNDLE + "licenses/" + name))
        with zipfile.ZipFile(io.BytesIO(archive_read(saved, "SOURCE_SNAPSHOT.zip"))) as snapshot:
            validate_zip(snapshot)
            semantic = json.loads(snapshot.read("SNAPSHOT.json"))
            decisions = json.loads(saved.read("SELECTION.json"))
            original = {r["cell_id"]: r for r in semantic["source_rows"]}
            selected_ids = {r["id"] for r in roster if not r["id"].startswith("libri-")}
            need(len(selected_ids) == 77 and selected_ids == set(decisions["eligible_cell_ids"]), "original selected 77, no new admission")
            wavs = {}
            for name in [f"inputs/old-source/shard-{i:02d}.zip" for i in range(9)] + ["inputs/new/source.zip"]:
                with zipfile.ZipFile(io.BytesIO(snapshot.read(name))) as shard:
                    validate_zip(shard)
                    for member in shard.namelist():
                        if member.endswith("/audio-16k.wav"):
                            cell_id = PurePosixPath(member).parent.name
                            if cell_id in selected_ids:
                                need(cell_id not in wavs, "unique selected source WAV")
                                raw = shard.read(member)
                                need(sha(raw) == original[cell_id]["wav_sha256"], "original WAV SHA")
                                with wave.open(io.BytesIO(raw)) as source:
                                    need((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) ==
                                         (1, 2, 16000, "NONE"), "original PCM16 WAV")
                                    pcm = source.readframes(source.getnframes())
                                need(len(pcm) == 2 * original[cell_id]["frames"] and sha(pcm) == original[cell_id]["pcm_sha256"], "original source PCM")
                                wavs[cell_id] = pcm
            need(set(wavs) == selected_ids, "all selected audio retained")
    sys.path.insert(0, str(root / "sources"))
    import causal_embedding as embedding_runtime
    import portable_head as portable
    import weak_embedding_contract as contract
    weights = portable.load_weights(root / "models/baseline-head.npz")
    need(sha(portable.export_blob(weights)) == "acbf6bd314ab8b458ae2175b6268b306ed0102d7ffc68c1b6ffde87af1c41174", "baseline weight export identity")
    by_feature = {r["id"]: r for r in feature_evidence}
    by_roster = {r["id"]: r for r in roster}
    need(len(by_feature) == len(by_roster) == 117 and set(by_feature) == set(by_roster), "unique feature identities")
    with np.load(io.BytesIO(cache_raw), allow_pickle=False) as saved:
        need(set(saved.files) == {"embeddings", "samples", "discontinuity_before"}, "original cache keys")
        embs, clocks, gaps = saved["embeddings"].copy(), saved["samples"].copy(), saved["discontinuity_before"].copy()
    need(embs.dtype == np.float32 and embs.shape == (10354, 96) and bool(np.isfinite(embs).all())
         and clocks.shape == gaps.shape == (10354,) and clocks.dtype == np.uint64 and gaps.dtype == np.bool_, "original cache geometry")
    clean_streams = {}
    cursor = 0
    for row in roster:
        begin, count = row["first_record"], row["records"]
        need(begin == cursor and row["source"] == by_feature[row["id"]]["source"], "contiguous original roster")
        end = begin + count
        local = [int(s) - row["offset"] for s in clocks[begin:end]]
        need(local == list(range(16000, row["observed_samples"] + 1, 1280))
             and gaps[begin:end].tolist() == [True] + [False] * (count - 1), "saved causal stream clocks")
        clean_streams[row["id"]] = (embs[begin:end].copy(), local)
        cursor = end
    need(cursor == len(embs), "no trailing cache rows")
    old_by_id = {r["id"]: r for r in old_events}
    baseline_clean = [summarize_case(r, old_by_id[r["id"]]["events"]) for r in roster]
    write_json(root / "predictions/baseline-clean-events-reused.json", baseline_clean, create=True)
    # Reconstruct only the exact fragments needed by the original forty tracks.
    tracks = source_rows["actual_research_tracks"]
    need(len(tracks) == 40 and collections.Counter(t["role"] for t in tracks) == {"train": 30, "development": 10}, "30/10 background roles")
    roles = contract.speaker_roles([t["speaker"] for t in tracks])
    need(all(t["role"] == roles[t["speaker"]] for t in tracks), "unchanged background speaker split")
    originals = {r["id"]: r for r in source_rows["decoded_originals"]}
    required = {f["source_id"] for t in tracks for f in t["fragments"]}
    corpus = root / "downloads/dev-clean.tar.gz"
    need(identity(corpus) == {k: CORPUS[k] for k in ("bytes", "sha256")}, "worker corpus identity")
    decoded = {}
    with tarfile.open(corpus, "r:gz") as archive:
        for name in sorted(required):
            member = archive.getmember(name)
            need(member.isfile() and member.size < 16 * 1024**2, "regular original FLAC")
            raw = archive.extractfile(member).read()
            prior = originals[name]
            need(sha(raw) == prior["flac_sha256"], "original FLAC SHA")
            with sf.SoundFile(io.BytesIO(raw)) as sound:
                need(sound.samplerate == 16000 and sound.channels == 1 and sound.subtype == "PCM_16"
                     and sound.frames == prior["frames"], "original FLAC format")
                pcm = sound.read(dtype="int16", always_2d=False).astype("<i2", copy=False).tobytes()
            need(sha(pcm) == prior["pcm_sha256"], "decoded original PCM SHA")
            decoded[name] = pcm
    for track in tracks:
        pieces, offset = [], 0
        for fragment in track["fragments"]:
            need(fragment["track_start_sample"] == offset and sha(decoded[fragment["source_id"]]) == fragment["source_pcm_sha256"], "original fragment binding")
            start, count = fragment["source_start_sample"], fragment["samples"]
            part = decoded[fragment["source_id"]][2 * start:2 * (start + count)]
            need(len(part) == 2 * count, "original fragment length")
            pieces.append(part); offset += count
        track["pcm16le"] = b"".join(pieces)
        need(offset == track["frames"] == 256000 and sha(track["pcm16le"]) == track["pcm_sha256"], "original 16-second track")
    track_map = {t["id"]: t for t in tracks}
    mixed_roster, mixes, mixed_pcm = [], [], {}
    for row in roster:
        if row["id"] not in selected_ids:
            continue
        src = row["source"]
        original_pcm = wavs[row["id"]]
        for kind in ("prefix", "context"):
            need(track_map[src[kind + "_track"]]["role"] == row["role"] == src[kind + "_role"], "source context role")
        def span(kind):
            track = track_map[src[kind + "_track"]]
            start, count = src[kind + "_start_sample"], src[kind + "_samples"]
            return track["pcm16le"][2 * start:2 * (start + count)]
        prefix, context = span("prefix"), span("context")
        clean = prefix + original_pcm + context
        need(len(clean) == row["observed_samples"] * 2 and sha(clean) == src["combined_pcm_sha256"], "clean PCM reconstruction identity")
        noise, offset = select_noise_track(row["id"], row["role"], len(original_pcm) // 2, tracks)
        noise_pcm = noise["pcm16le"][2 * offset:2 * offset + len(original_pcm)]
        mixed, receipt = mix_pcm16(original_pcm, noise_pcm)
        receipt.update(id=row["id"], role=row["role"], weak_class=row["weak_class"],
                       source_family_id=original[row["id"]]["source_family_id"], noise_track=noise["id"],
                       noise_role=noise["role"], noise_start_sample=offset,
                       noise_track_pcm_sha256=noise["pcm_sha256"],
                       original_combined_pcm_sha256=sha(clean), independent_source_count_increment=0)
        if mixed is not None:
            combined = prefix + mixed + context
            receipt["combined_mixed_pcm_sha256"] = sha(combined)
            mixed_pcm[row["id"]] = combined
            mixed_roster.append({**row, "source_family_id": receipt["source_family_id"], "mix": receipt})
            (root / "audio" / (row["id"] + ".pcm")).write_bytes(combined)
        mixes.append(receipt)
    write_json(evidence / "MIX_ROSTER.json", mixes, create=True)
    write_json(root / "cache/mixed-roster.json", mixed_roster, create=True)
    need(len(mixes) == 77, "planned mixed denominator")
    embedding = embedding_runtime.FrozenOnnxEmbedding(root / "models")
    class CountedSession:
        def __init__(self, session):
            self.session = session
        def run(self, *values, **kwargs):
            need(progress["onnx_forward_started"] < 8000, "ONNX forward budget")
            progress["onnx_forward_started"] += 1
            checkpoint()
            answer = self.session.run(*values, **kwargs)
            progress["onnx_forward_completed"] += 1
            checkpoint()
            return answer
    embedding.mel, embedding.embedding = CountedSession(embedding.mel), CountedSession(embedding.embedding)
    def embed(values):
        need(progress["new_embedding_calls_started"] < 4000, "embedding call budget")
        progress["new_embedding_calls_started"] += 1
        checkpoint()
        vector = embedding(values)
        progress["new_embedding_calls_completed"] += 1
        checkpoint()
        return vector
    mixed_streams, cache_arrays = {}, {}
    for index, row in enumerate(mixed_roster):
        values = np.frombuffer(mixed_pcm[row["id"]], dtype="<i2")
        samples = list(range(16000, len(values) + 1, 1280))
        vectors = np.asarray([embed(values[t - 16000:t]) for t in samples], dtype=np.float32)
        mixed_streams[row["id"]] = (vectors, samples)
        cache_arrays[f"e{index:03d}"] = vectors
        cache_arrays[f"t{index:03d}"] = np.asarray(samples, dtype=np.uint64)
    np.savez(root / "cache/mixed-embeddings.npz", **cache_arrays)
    write_json(evidence / "FEATURE_GEOMETRY.json", embedding.geometry, create=True)
    def evaluate(name, model, rows, streams):
        metrics = []
        with (root / "predictions" / (name + "-frames.jsonl")).open("x", encoding="utf-8") as out:
            for row in rows:
                need(progress["head_case_passes_started"] < 271, "fixed head case pass budget")
                progress["head_case_passes_started"] += 1
                checkpoint()
                matrix, samples = streams[row["id"]]
                result = portable.reference(model, matrix, samples)
                events = [{"sample": r["sample"], "keyword": r["keyword"]} for r in result if r["keyword"]]
                for record in result:
                    out.write(json.dumps({"id": row["id"], **record}, separators=(",", ":"), allow_nan=False) + "\n")
                out.flush()
                progress["head_ready_outputs"] += sum(bool(r["ready"]) for r in result)
                progress["head_case_passes_completed"] += 1
                checkpoint()
                metrics.append(summarize_case(row, events))
        write_json(root / "predictions" / (name + "-events.json"), metrics, create=True)
        return metrics
    baseline_noisy = evaluate("baseline-snr10", weights, mixed_roster, mixed_streams)
    decision = candidate_decision(baseline_noisy, mixed_roster)
    write_json(evidence / "CANDIDATE_DECISION.json", decision, create=True)
    candidate_clean = candidate_noisy = None
    if decision["execute"]:
        torch.set_num_threads(1); torch.manual_seed(20261117); torch.use_deterministic_algorithms(True)
        model = torch.nn.Sequential(torch.nn.Linear(1536, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
        with torch.no_grad():
            for parameter, key in ((model[0].weight, "w1"), (model[0].bias, "b1"), (model[2].weight, "w2"), (model[2].bias, "b2")):
                parameter.copy_(torch.from_numpy(weights[key]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        groups = {(condition, label): [] for condition in ("clean", "snr10") for label in range(3)}
        prepared = {}
        for condition, rows, streams in (("clean", roster, clean_streams), ("snr10", mixed_roster, mixed_streams)):
            for row in rows:
                if row["role"] != "train":
                    continue
                matrix, samples = streams[row["id"]]
                ends = by_feature[row["id"]]["head_instance_ends"]
                positions = {t: i for i, t in enumerate(samples)}
                need(ends and all(t in positions and positions[t] >= 15 for t in ends), "fixed original bag geometry")
                windows = np.asarray([matrix[positions[t] - 15:positions[t] + 1] for t in ends], dtype=np.float32)
                normalized = ((windows - weights["mean96"]) / weights["scale96"]).reshape(len(windows), 1536)
                key = (condition, row["id"])
                prepared[key] = torch.from_numpy(normalized)
                groups[(condition, row["weak_class"])].append(key)
        need(all(groups.values()), "six fixed augmented training groups")
        draws, losses = [], []
        for step in range(96):
            progress["optimizer_updates_started"] += 1; checkpoint()
            optimizer.zero_grad(set_to_none=True)
            terms, used = [], []
            for condition in ("clean", "snr10"):
                for label in range(3):
                    group = groups[(condition, label)]
                    key = group[step % len(group)]
                    progress["torch_forward_calls_started"] += 1; checkpoint()
                    logp = torch.log_softmax(model(prepared[key]), dim=-1)
                    progress["torch_forward_calls_completed"] += 1; checkpoint()
                    terms.append(-logp[:, 0].mean() if label == 0 else math.log(len(logp)) - torch.logsumexp(logp[:, label], dim=0))
                    used.append({"condition": condition, "id": key[1], "weak_class": label, "role": "train"})
            loss = torch.stack(terms).mean()
            need(bool(torch.isfinite(loss)), "finite candidate loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach())); draws.append(used)
            progress["optimizer_updates_completed"] += 1; checkpoint()
        candidate = {"w1": model[0].weight.detach().numpy(), "b1": model[0].bias.detach().numpy(),
                     "w2": model[2].weight.detach().numpy(), "b2": model[2].bias.detach().numpy(),
                     "mean96": weights["mean96"].copy(), "scale96": weights["scale96"].copy()}
        np.savez(root / "models/candidate-head.npz", **candidate)
        candidate = portable.load_weights(root / "models/candidate-head.npz")
        need(all(np.array_equal(candidate[k], weights[k]) for k in ("mean96", "scale96")), "normalization frozen")
        (root / "models/candidate-head.bin").write_bytes(portable.export_blob(candidate))
        write_json(evidence / "TRAINING.json", {"optimizer_updates": 96, "warmstart": identity(root / "models/baseline-head.npz"),
            "candidate": identity(root / "models/candidate-head.npz"), "losses": losses, "draws": draws,
            "comparison": "augmentation_plus_96_additional_updates_not_pure_loss_ablation",
            "development_checkpoint_or_threshold_selection": False}, create=True)
        candidate_clean = evaluate("candidate-clean", candidate, roster, clean_streams)
        candidate_noisy = evaluate("candidate-snr10", candidate, mixed_roster, mixed_streams)
    result = {"schema": SCHEMA, "status": "COMPLETED_WEAK_QUALITY_RESEARCH", "run_id": os.environ["GITHUB_RUN_ID"],
        "head_sha": os.environ["GITHUB_SHA"], "run_attempt": 1, "source_training": TRAINING,
        "planned_mixed_cases": 77, "eligible_mixed_cases": len(mixed_roster),
        "mix_dispositions": dict(collections.Counter(r["status"] for r in mixes)),
        "source_synthetic_cases": 77, "source_synthetic_train": 50, "source_synthetic_development": 27,
        "same_source_condition_is_not_independent_audio": True, "cross_batch_voice_family_exposure": True,
        "baseline_clean_events_reused": True, "clean_frozen_embedding_calls_reused": 10354,
        "new_counts": progress, "candidate_decision": decision,
        "baseline_clean": aggregate(baseline_clean), "baseline_snr10": aggregate(baseline_noisy),
        "candidate_clean": aggregate(candidate_clean) if candidate_clean is not None else None,
        "candidate_snr10": aggregate(candidate_noisy) if candidate_noisy is not None else None,
        "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN", "word_end_latency": "NOT_MEASURED",
        "ssc305": "NOT_RUN", "model_delivered_as_product": False, **dict.fromkeys(FALSE_FLAGS, False)}
    write_json(evidence / "RESULT.json", result, create=True)
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "downloads" not in path.relative_to(root).parts:
            manifest[path.relative_to(root).as_posix()] = identity(path)
    write_json(evidence / "WORKER_MANIFEST.json", {"files": manifest, "excludes_downloads_and_later_supervisor_exit": True}, create=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--prepare-noise", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = read_json(args.plan); validate_plan(plan)
    need(sha(Path(__file__).read_bytes()) == plan["worker_sha256"], "source identity")
    validate_original_sources()
    assert_same_research_parameters(plan)
    if args.execute or args.worker or args.prepare_noise:
        need(args.output is not None, "create-only output required")
        def interrupted(_number, _frame):
            raise KeyboardInterrupt("termination requested")
        signal.signal(signal.SIGTERM, interrupted)
        if args.worker or args.prepare_noise:
            proof = verify_failed_proof((args.output.resolve() / "downloads/failed-first-run.zip").read_bytes())
            recorded = read_json(args.output.resolve() / "evidence/SOURCE_RECOVERY_GATE.json")
            need(all(recorded.get(k) == v for k, v in proof.items()) and recorded.get("quality_parameters_unchanged") is True,
                 "source recovery proof must precede any numerical or PCM worker")
        (prepare_noise if args.prepare_noise else worker if args.worker else execute)(args, plan)
    else:
        print(json.dumps({"status": "SOURCE_PLAN_VALIDATED_NO_MODEL_EXECUTION", "schema": SCHEMA}))


if __name__ == "__main__":
    main()
