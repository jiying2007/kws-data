"""Blind, single-pass observations of the complete Mero-v1 source cohort.

The verified Pocket12 helpers supply model loading and native XLSR readback.
This module owns the new cohort's ledger; it imports no numerical runtime.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import signal

import english_pocket12_observe as native

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = "english-mero-meh-observer-v1"
PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme")
need, sha, regular, read, put, write = (
    native.need, native.sha, native.regular, native.read, native.put, native.write)
file_records = native.file_records
identity = native.identity
_load = native._load
FLAGS = {"target_conditioned": False, "input_scope": "ENTIRE_UNCROPPED_SHORT_PCM", "retry_count": 0}


def source_check(plan=None):
    measured = {name: sha(regular(ROOT / name)) for name in native.HELPER_PINS}
    need(measured == native.HELPER_PINS, "IMMUTABLE_OBSERVER_HELPERS")
    if plan is not None:
        need(plan["providers"] == list(PROVIDERS)
             and plan["maximum_calls_per_observer"] == 144
             and plan["maximum_total_model_calls"] == 720
             and plan["retry_count"] == 0, "NEW_COHORT_FIXED_BUDGET")
    return {"source_check": "PASS", "providers": list(PROVIDERS),
            "new_model_calls": 0, "new_network_calls": 0}


def base_record(provider, row, run_identity, source_sha):
    return {"schema": SCHEMA, "provider": provider, "id": row["id"], "run": run_identity,
            "source_sha256": source_sha, "input_pcm_sha256": row["pcm_sha256"],
            "input_wav_sha256": row["wav_sha256"], "input_samples": row["pcm_samples"],
            "observer": identity(provider), "observer_code_sha256": sha(regular(Path(__file__))), **FLAGS}


def initial_rows(rows):
    return [{"id": row["id"], "status": "not_started", "input_started": False,
             "forward_attempts": 0, "forward_returned": False, "capture_saved": False} for row in rows]


def _execute_rows(provider, source_root, rows, output, run_identity, source_sha, backend, ledger):
    import english_mero_meh_generate as generate
    for row, progress in zip(rows, ledger):
        directory = output / "rows" / row["id"]
        directory.mkdir(parents=True)
        base = base_record(provider, row, run_identity, source_sha)
        try:
            write(directory / "STARTED.json", base)
            progress["input_started"] = True
            wav = generate.within(source_root, row["wav_path"])
            pcm = regular(generate.within(source_root, row["pcm_path"]))
            need(sha(pcm) == row["pcm_sha256"] and sha(regular(wav)) == row["wav_sha256"],
                 "UNCHANGED_ENTIRE_INPUT")
            if provider == "phoneme":
                prepare, forward, materialize = backend
                geometry_row = {"pcm_samples": row["pcm_samples"], "full_to_crop_offset_16k": 0}
                batch, facts, normalized, mask = prepare(pcm, geometry_row)
                processor = {"processor": facts | {"normalization_scope": "ENTIRE_UNCROPPED_SHORT_PCM"},
                             "normalized_input": put(directory / "normalized-input.f32", normalized),
                             "attention_mask": put(directory / "attention-mask.i32", mask)}
                write(directory / "INPUT.json", base | processor)
            write(directory / "FORWARD_STARTED.json", base)
            progress["forward_attempts"] = 1
            result = forward(batch) if provider == "phoneme" else backend(wav, directory)
            progress["forward_returned"] = True
            write(directory / "FORWARD_RETURNED.json", base)
            if provider == "phoneme":
                values, logits = materialize(result, geometry_row)
                captured = processor | {"logits": put(directory / "logits.f32", logits), "values": values,
                    "frame_geometry": native.native_module().frame_geometry(row["pcm_samples"], 0)}
            else:
                need(type(result) is str and len(result) <= 16384, "RAW_TRANSCRIPT_STRING")
                captured = {"raw_transcript": result}
            pin = write(directory / "CAPTURE.json", base | captured | {"status": "success",
                        "input_coverage_complete": True, "decoder_output_complete": True})
            progress.update(status="success", capture_saved=True, capture_sha256=pin["sha256"])
        except BaseException as exc:
            progress.update(status="failed", error=type(exc).__name__ + ": " + str(exc)[:2000])
            write(directory / "FAILED.json", base | progress)
            if not isinstance(exc, Exception):
                raise


def _finish(output, provider, ledger, run_identity, source_sha, failure):
    n = len(ledger)
    attempts, returned, completed = (sum(row[key] for row in ledger) for key in
                                    ("forward_attempts", "forward_returned", "capture_saved"))
    need(0 <= completed <= returned <= attempts <= n, "COUNTER_INVARIANTS")
    result = {"schema": SCHEMA, "provider": provider, "run": run_identity, "source_sha256": source_sha,
        "status": "COMPLETE" if completed == n and failure is None else "INCOMPLETE", "denominator": n,
        "actual_forward_attempts": attempts, "native_forward_returns": returned, "completed_count": completed,
        "failed_count": sum(row["status"] == "failed" for row in ledger),
        "not_started_count": sum(row["status"] == "not_started" for row in ledger),
        "initialization_error": failure, "rows": ledger, **FLAGS}
    write(output / "RESULT.json", result)
    write(output / "FILES.json", {"schema": SCHEMA + "/files", "files": file_records(output),
                                  "self_excluded": "FILES.json"})
    return result


def run(*, provider, source_root, output_root, asset_root, plan, run_identity):
    import english_mero_meh_generate as generate
    import english_mero_meh_once as once
    once.source_preflight(plan)
    need(once.execution_identity(plan) == run_identity and provider in PROVIDERS, "GUARDED_OBSERVER")
    corpus, rows = generate.validate_source(source_root, run_identity)
    need(len(rows) == len(corpus["rows"]) == plan["native_tts_request_budget"], "COMPLETE_SOURCE_REQUIRED")
    output, assets, source_root = map(lambda p: Path(p).absolute(), (output_root, asset_root, source_root))
    for path in (output, assets):
        need(not path.exists() and path.parent.is_dir()
             and not any(p.is_symlink() for p in (path, *path.parents)), "FRESH_PRIVATE_DIRECTORY")
    need(all(a != b and a not in b.parents and b not in a.parents
             for a, b in ((output, assets), (source_root, assets), (output, source_root))), "SEPARATE_DIRECTORIES")
    output.mkdir(mode=0o700)
    source_sha = sha(regular(source_root / "CORPUS_MANIFEST.json"))
    ledger = initial_rows(rows)
    write(output / "STARTED.json", {"schema": SCHEMA, "provider": provider, "run": run_identity,
        "source_sha256": source_sha, "denominator": len(rows), "maximum_calls": len(rows), **FLAGS})
    failure = None
    def expired(signum, frame):
        raise TimeoutError("MODEL_INITIALIZATION_DEADLINE")
    previous = signal.signal(signal.SIGALRM, expired)
    try:
        write(output / "MODEL_LOAD_STARTED.json", {"provider": provider, "actual_forward_attempts": 0})
        signal.alarm(plan["time_budget_seconds"]["initialization"])
        loaded = _load(provider, assets, output)
        signal.alarm(0)
        backend, lock = (loaded[:3], loaded[3]) if provider == "phoneme" else loaded
        write(output / "MODEL_LOCK.json", lock)
        _execute_rows(provider, source_root, rows, output, run_identity, source_sha, backend, ledger)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if assets.exists():
            shutil.rmtree(assets)
    return _finish(output, provider, ledger, run_identity, source_sha, failure)


def _verify_lock(provider, root):
    lock = read(root / "MODEL_LOCK.json")
    if provider != "phoneme":
        need(lock == {"observer": identity(provider), "runtime_versions": native.TEXT_RUNTIME[provider],
            "adapter_sha256": native.HELPER_PINS["automation/english_seven_voice_asr_once.py"],
            "initialization_model_calls": 0, "target_conditioned": False}, "FROZEN_BLIND_ASR_LOCK")
        return None
    helper = native.native_module()
    expected = {"model": helper.OBSERVERS[provider], "files": helper.OBSERVER_FILES[provider],
        "runtime_versions": helper.RUNTIME_VERSIONS, "native_forward_attempts_at_initialization": 0,
        "model_parameters_modified": False, "target_conditioned": False}
    need(all(lock.get(key) == value for key, value in expected.items()), "FROZEN_NATIVE_XLSR_LOCK")
    for name, pin in helper.OBSERVER_FILES[provider].items():
        if name != "pytorch_model.bin":
            raw = regular(root / "model-metadata" / name)
            need({"bytes": len(raw), "sha256": sha(raw)} == pin, "FROZEN_PHONE_METADATA")
    return read(root / "model-metadata" / "vocab.json")


def readback(*, provider, source_root, observer_root, plan, run_identity):
    """Reconstruct calls and outputs from saved bytes, including missing terminals."""
    import english_mero_meh_generate as generate
    source_check(plan)
    need(provider in PROVIDERS, "FIXED_PROVIDER")
    corpus, successful = generate.validate_source(source_root, run_identity)
    rows, root = corpus["rows"], None if observer_root is None else Path(observer_root)
    n = len(rows)
    need(n == len(successful) == plan["native_tts_request_budget"], "COMPLETE_SOURCE_FOR_JOIN")
    source_sha = sha(regular(Path(source_root) / "CORPUS_MANIFEST.json"))
    terminal = bool(root and (root / "RESULT.json").is_file() and (root / "FILES.json").is_file())
    result = read(root / "RESULT.json") if terminal else None
    vocab = _verify_lock(provider, root) if root and (root / "MODEL_LOCK.json").is_file() else None
    if root and root.exists():
        allowed = {"STARTED.json", "MODEL_LOAD_STARTED.json", "MODEL_LOCK.json", "RESULT.json", "FILES.json",
                   "rows", "UNOBSERVED.json"} | ({"model-metadata"} if provider == "phoneme" else set())
        need({p.name for p in root.iterdir()} <= allowed, "NO_EXTRA_OBSERVER_FILES")
        if (root / "rows").exists():
            need({p.name for p in (root / "rows").iterdir()} <= {r["id"] for r in rows}, "NO_EXTRA_INPUT_ROWS")
        if (root / "STARTED.json").exists():
            need(read(root / "STARTED.json") == {"schema": SCHEMA, "provider": provider, "run": run_identity,
                 "source_sha256": source_sha, "denominator": n, "maximum_calls": n, **FLAGS}, "START_BINDING")
    if terminal:
        need(read(root / "FILES.json") == {"schema": SCHEMA + "/files", "files": file_records(root),
             "self_excluded": "FILES.json"}, "EXACT_OBSERVER_FILE_CLOSURE")
        expected = {"schema": SCHEMA, "provider": provider, "run": run_identity,
                    "source_sha256": source_sha, "denominator": n, **FLAGS}
        need(all(result.get(k) == v for k, v in expected.items()), "OBSERVER_RESULT_BINDING")
        need([r["id"] for r in result["rows"]] == [r["id"] for r in rows], "ALL_PLANNED_ROWS_IN_ORDER")
        need((root / "STARTED.json").is_file() and (root / "MODEL_LOAD_STARTED.json").is_file(), "START_RECEIPTS")
    observations, ledger = [], []
    for index, row in enumerate(rows):
        directory = root / "rows" / row["id"] if root else None
        exists = lambda name: bool(directory and (directory / name).is_file())
        if directory and directory.exists():
            allowed = {"STARTED.json", "FORWARD_STARTED.json", "FORWARD_RETURNED.json", "FAILED.json", "CAPTURE.json"}
            if provider == "phoneme":
                allowed |= {"INPUT.json", "normalized-input.f32", "attention-mask.i32", "logits.f32"}
            need({p.name for p in directory.iterdir()} <= allowed, "NO_EXTRA_ROW_CALL_OR_CAPTURE")
        base = base_record(provider, row, run_identity, source_sha)
        progress = {"id": row["id"], "status": "unknown", "input_started": exists("STARTED.json"),
                    "forward_attempts": int(exists("FORWARD_STARTED.json")),
                    "forward_returned": exists("FORWARD_RETURNED.json"), "capture_saved": exists("CAPTURE.json")}
        for filename in ("STARTED.json", "FORWARD_STARTED.json", "FORWARD_RETURNED.json", "FAILED.json", "CAPTURE.json"):
            if exists(filename):
                saved = read(directory / filename)
                need(all(saved.get(k) == v for k, v in base.items()), "ROW_CAPTURE_BINDING")
        if exists("FAILED.json"):
            failed = read(directory / "FAILED.json")
            need(type(failed.get("forward_returned")) is bool and failed.get("capture_saved") is False
                 and failed.get("forward_attempts") == progress["forward_attempts"], "FAILURE_LEDGER")
            progress["forward_returned"] |= failed["forward_returned"]
        need(not progress["forward_attempts"] or progress["input_started"], "START_BEFORE_CALL")
        need(not progress["forward_returned"] or progress["forward_attempts"] == 1, "CALL_BEFORE_RETURN")
        observation = {"id": row["id"], "status": "missing", "input_pcm_sha256": row["pcm_sha256"],
                       "input_samples": row["pcm_samples"], "target_conditioned": False}
        if progress["capture_saved"]:
            need(progress["forward_returned"] and not exists("FAILED.json")
                 and (root / "MODEL_LOCK.json").is_file(), "CAPTURE_AFTER_NATIVE_RETURN")
            capture = read(directory / "CAPTURE.json")
            need(capture.get("status") == "success" and capture.get("input_coverage_complete") is True
                 and capture.get("decoder_output_complete") is True, "FULL_CAPTURE")
            if provider == "phoneme":
                pcm = regular(generate.within(source_root, row["pcm_path"]))
                observation["phone_tokens"] = native._verify_phone(directory, capture, row, pcm, vocab)
                saved_input = read(directory / "INPUT.json")
                expected_input = base | {k: capture[k] for k in ("processor", "normalized_input", "attention_mask")}
                need(saved_input == expected_input, "SAVED_PRE_FORWARD_INPUT")
            else:
                need(type(capture.get("raw_transcript")) is str and len(capture["raw_transcript"]) <= 16384,
                     "COMPLETE_RAW_TRANSCRIPT")
                observation["raw_transcript"] = capture["raw_transcript"]
            progress.update(status="success", capture_sha256=sha(regular(directory / "CAPTURE.json")))
            observation.update(status="success", capture_path=f"observers/{provider}/rows/{row['id']}/CAPTURE.json",
                capture_sha256=progress["capture_sha256"], input_coverage_complete=True, decoder_output_complete=True)
        elif exists("FAILED.json"):
            progress["status"] = observation["status"] = "failed"
        elif terminal:
            need(not progress["input_started"], "STARTED_ROW_REQUIRES_TERMINAL_RECORD")
            progress["status"] = "not_started"
        if terminal:
            claimed = result["rows"][index]
            need(all(type(claimed.get(k)) is type(v) and claimed[k] == v for k, v in progress.items()),
                 "TERMINAL_LEDGER_RECONSTRUCTED")
        observations.append(observation)
        ledger.append(progress)
    attempts, returned, completed = (sum(r[k] for r in ledger) for k in
                                    ("forward_attempts", "forward_returned", "capture_saved"))
    counts = {"actual_forward_attempts": attempts, "native_forward_returns": returned, "completed_count": completed,
              "failed_count": sum(r["status"] == "failed" for r in ledger),
              "not_started_count": sum(r["status"] == "not_started" for r in ledger)}
    if terminal:
        need(all(type(result.get(k)) is int and result[k] == v for k, v in counts.items()), "ACTUAL_COUNTS")
        need(result["status"] == ("COMPLETE" if completed == n and result["initialization_error"] is None else "INCOMPLETE"),
             "COMPLETENESS_RECONSTRUCTED")
    summary = {"provider": provider, "denominator": n, "terminal_receipt_present": terminal,
        "actual_forward_attempts": attempts if terminal else None, "attempts_lower_bound": attempts,
        "attempts_upper_bound": attempts if terminal else n, "native_forward_returns": returned if terminal else None,
        "completed_count": completed, "failed_count": counts["failed_count"],
        "not_started_count": counts["not_started_count"] if terminal else None,
        "unknown_count": sum(r["status"] == "unknown" for r in ledger),
        "status": result["status"] if terminal else "INCOMPLETE_UNKNOWN_COUNTERS"}
    return {"summary": summary, "rows": observations, "new_model_calls": 0}
