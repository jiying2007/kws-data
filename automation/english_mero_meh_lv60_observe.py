"""Bounded LV60 diagnostic of 24 authenticated, unchanged whole short PCMs.

Only install()/observe() reach a runtime or network. Saved-byte readback is
stdlib-only. The original corpus, four observations and admission are immutable.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import english_pocket12_observe as saved
import english_kokoro_context32_mask_observe as native

HERE = Path(__file__).resolve().parent
SCHEMA = "english-mero-meh-lv60-observer-v1"
CAMPAIGN = "english-mero-meh-lv60-diagnostic-v1"
PROVIDER = "phoneme_lv60"
PRESETS = ("af_bella", "af_sarah", "am_michael", "am_puck")
CELLS = ("mero-slow", "mero-fast", "nero", "memo", "hello", "open-door")
HELPERS = {
    "english_pocket12_observe.py": "07b93e626b5e822b09869c0f06d2179140d938c1c795f97c2d286e112b2c0426",
    "english_kokoro_context32_mask_observe.py": "3f184bdeb7536f260b2d4e3ee77476d95590d05007ae251bcb93c51ce2be01af",
    "english_final_ipa_xlsr_once.py": "716928f7707fc662f62e59a988c0c264123be301abac4c23fe835bc09c052b3a",
}
FLAGS = {"diagnostic_only": True, "training_allowed": False, "target_conditioned": False,
         "input_scope": "ENTIRE_UNCROPPED_SHORT_PCM", "retry_count": 0, "warmup_count": 0,
         "carrier_added": False, "cropping_performed": False}
need, sha, regular, read, put, write, file_records = (
    saved.need, saved.sha, saved.regular, saved.read, saved.put, saved.write, saved.file_records)
within = native.within
canonical = native.canonical


def source_check(plan):
    need({n: sha(regular(HERE / n)) for n in HELPERS} == HELPERS, "IMMUTABLE_LV60_HELPERS")
    fixed = {"campaign_id": CAMPAIGN, "provider": PROVIDER, "diagnostic_only": True,
        "training_allowed": False, "retry_count": 0, "warmup_count": 0,
        "maximum_calls_per_observer": 24, "maximum_total_model_calls": 24,
        "normalization_match_required": True, "model": native.OBSERVERS[PROVIDER],
        "model_assets": native.OBSERVER_FILES[PROVIDER], "runtime_versions": native.RUNTIME_VERSIONS}
    need(all(type(plan.get(k)) is type(v) and plan[k] == v for k, v in fixed.items()), "FIXED_LV60_DIAGNOSTIC")
    panel = plan.get("panel_rows")
    need(type(panel) is list and len(panel) == 24
         and [(r["preset_id"], r["cell_id"]) for r in panel] == [(p, c) for p in PRESETS for c in CELLS]
         and len({r["id"] for r in panel}) == 24, "FIXED_24_ROW_PANEL")
    return {"source_check": "PASS", "denominator": 24, "new_model_calls": 0}


@contextmanager
def _frozen_imports(root, records):
    directory = Path(root) / "frozen_source/automation"
    names = {Path(r["path"]).stem for r in records
             if Path(r["path"]).parent.as_posix() == "frozen_source/automation" and r["path"].endswith(".py")}
    previous = {n: sys.modules[n] for n in names if n in sys.modules}
    old_path = list(sys.path)
    for name in names:
        sys.modules.pop(name, None)
    sys.path[:0] = [str(directory), str(directory.parent)]
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def _load_source_component(name):
    return importlib.import_module("english_mero_meh_" + name)


def source_readback(source_root, plan):
    """Authenticate the original closed packet and reconstruct all four readers."""
    source_check(plan)
    root = Path(source_root)
    index_raw = regular(root / "FILES.json")
    need(sha(index_raw) == plan["source_files_manifest_sha256"], "ORIGINAL_FILES_MANIFEST_PIN")
    index = json.loads(index_raw)
    need(index == {"schema": "english-mero-meh-corpus-files-v1", "files": file_records(root)},
         "ORIGINAL_EXACT_FILE_CLOSURE")
    pins = {r["path"]: r["sha256"] for r in index["files"]}
    source_plan = read(root / "EXECUTION_PLAN.json")
    need(all(pins.get("frozen_source/" + name) == pin
             for name, pin in source_plan["source_files_sha256"].items()), "AUTHENTICATED_FROZEN_READERS")
    corpus, evidence = read(root / "CORPUS_MANIFEST.json"), read(root / "MACHINE_EVIDENCE.json")
    need(corpus["run"] == plan["source_run"] and corpus["campaign_id"] == "english-mero-meh-corpus-v1"
         and len(corpus["rows"]) == 144, "ORIGINAL_SOURCE_IDENTITY")
    with _frozen_imports(root, index["files"]):
        generate, observe = _load_source_component("generate"), _load_source_component("observe")
        status = generate.source_readback(root, source_plan, corpus["run"])
        need(status.get("source_ready") is True, "ORIGINAL_COMPLETE_SOURCE_BYTES")
        packets = {p: observe.readback(provider=p, source_root=root, observer_root=root / "observers" / p,
            plan=source_plan, run_identity=corpus["run"]) for p in ("qwen", "whisper", "parakeet", "phoneme")}
    observations = {p: {r["id"]: r for r in packet["rows"]} for p, packet in packets.items()}
    reconstructed = [{"id": r["id"], "observations": {p: {k: v for k, v in observations[p][r["id"]].items()
        if k != "id"} for p in packets}} for r in corpus["rows"]]
    need(evidence["rows"] == reconstructed and evidence["source_readback"] == status
         and evidence["observer_summaries"] == {p: x["summary"] for p, x in packets.items()}
         and evidence["run"] == corpus["run"] and evidence["denominator"] == 144
         and evidence["actual_total_model_calls"] == 720 and status["tts_attempts"] == 144
         and all(x["summary"]["actual_forward_attempts"] == x["summary"]["completed_count"] == 144
                 for x in packets.values()), "ORIGINAL_MACHINE_EVIDENCE_RECONSTRUCTED")
    by_id, selected = {r["id"]: r for r in corpus["rows"]}, []
    required = {"id", "preset_id", "cell_id", "speed", "pcm_path", "pcm_sha256", "pcm_samples",
                "wav_path", "wav_sha256", "source_row_sha256"}
    for pin in plan["panel_rows"]:
        row = by_id[pin["id"]]
        need(required <= pin.keys() and pin["source_row_sha256"] == sha(canonical(row))
             and all(row.get(k) == v for k, v in pin.items() if k != "source_row_sha256"), "EXACT_ORIGINAL_PANEL_ROW")
        selected.append(row)
    return {"source_ready": True, "source_run": corpus["run"], "source_denominator": 144,
        "source_manifest_sha256": sha(regular(root / "CORPUS_MANIFEST.json")),
        "source_files_manifest_sha256": sha(index_raw), "denominator": 24, "panel_rows": selected,
        "actual_source_model_calls": 720, "new_model_calls": 0, "admission_modified": False}


def _guard(plan, run):
    source_check(plan)
    actual = {k: os.environ.get(v) for k, v in {"repository": "GITHUB_REPOSITORY", "run_id": "GITHUB_RUN_ID",
        "commit": "GITHUB_SHA", "run_attempt": "GITHUB_RUN_ATTEMPT", "event": "GITHUB_EVENT_NAME"}.items()}
    need(plan.get("enabled") is True and actual == run and run["repository"] == "jiying2007/kws-data"
         and run["event"] == "push" and run["run_attempt"] == "1"
         and os.environ.get("GITHUB_REF") == "refs/heads/main"
         and run["run_id"] != plan["source_run"]["run_id"], "GUARDED_DISTINCT_DIAGNOSTIC_RUN")


def install(plan, output):
    """Install the pinned CPU observer runtime; no weights or forwards here."""
    source_check(plan)
    need(plan.get("enabled") is True and sys.version_info[:2] == (3, 12), "ENABLED_PYTHON_3_12_RUNTIME")
    root = Path(output).absolute()
    need(not root.exists() and root.parent.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
         "FRESH_RUNTIME_DIRECTORY")
    root.mkdir(mode=0o700)
    python = root / "venv/bin/python"
    result = {"schema": SCHEMA + "/installation", "provider": PROVIDER, "status": "STARTED",
              "python_path": str(python), "new_model_calls": 0, "retry_count": 0, "commands": []}
    deadline = time.monotonic() + plan["time_budget_seconds"]["initialization"]
    pip = [str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
           "--retries", "0", "--timeout", "60"]
    commands = [("venv", [sys.executable, "-m", "venv", str(root / "venv")]),
        ("torch", pip + ["--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.9.1+cpu"]),
        ("dependencies", pip + [k + "==" + v for k, v in native.RUNTIME_VERSIONS.items() if k != "torch"]),
        ("pip-check", [str(python), "-I", "-m", "pip", "check"])]
    try:
        for label, command in commands:
            record = {"step": label, "status": "STARTED"}
            result["commands"].append(record)
            with (root / (label + ".stdout.txt")).open("xb") as out, (root / (label + ".stderr.txt")).open("xb") as err:
                remaining = deadline - time.monotonic()
                need(remaining > 0, "INSTALLATION_DEADLINE")
                process = subprocess.run(command, stdout=out, stderr=err, timeout=remaining, check=False)
            record.update(status="RETURNED", returncode=process.returncode)
            need(process.returncode == 0, "INSTALL_STEP_FAILED:" + label)
        result["status"] = "COMPLETE"
    except Exception as exc:
        result.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        write(root / "INSTALLATION.json", result)
    return result


def _base(row, plan, run, source_sha, reference):
    return {"schema": SCHEMA, "campaign_id": CAMPAIGN, "provider": PROVIDER, "id": row["id"],
        "run": run, "source_run": plan["source_run"], "source_sha256": source_sha,
        "source_files_manifest_sha256": plan["source_files_manifest_sha256"],
        "input_pcm_sha256": row["pcm_sha256"], "input_wav_sha256": row["wav_sha256"],
        "input_samples": row["pcm_samples"], "observer": native.OBSERVERS[PROVIDER],
        "observer_code_sha256": sha(regular(Path(__file__))), "xlsr_reference": reference, **FLAGS}


def _reference(source_root, row):
    path = "observers/phoneme/rows/" + row["id"] + "/CAPTURE.json"
    raw = regular(within(source_root, path))
    capture = json.loads(raw)
    need(capture["provider"] == "phoneme" and capture["observer"] == native.OBSERVERS["phoneme"]
         and capture["input_pcm_sha256"] == row["pcm_sha256"] and capture["input_samples"] == row["pcm_samples"],
         "ORIGINAL_SAME_PCM_XLSR_CAPTURE")
    return {"capture_path": path, "capture_sha256": sha(raw),
            "normalized_input_sha256": capture["normalized_input"]["sha256"],
            "attention_mask_sha256": capture["attention_mask"]["sha256"], "input_samples": row["pcm_samples"]}


def _verify_lock(root):
    lock = read(root / "MODEL_LOCK.json")
    _, function_sha = native.ctc_function(regular(HERE / "english_final_ipa_xlsr_once.py"))
    expected = {"schema": native.SCHEMA, "model": native.OBSERVERS[PROVIDER], "files": native.OBSERVER_FILES[PROVIDER],
        "runtime_versions": native.RUNTIME_VERSIONS, "native_processor_source_files": native.PROCESSOR_SOURCE_FILES,
        "ctc_helper_sha256": native.CTC_HELPER_SHA256, "ctc_function_sha256": function_sha,
        "blank_id": 0, "skipped_special_ids": [0, 1, 2, 3], "collapse_before_blank_deletion": True,
        "weights_only": True, "strict_state_dict_loading": True, "eval": True, "model_parameters_modified": False,
        "trust_remote_code": False, "phonemizer_used": False, "target_conditioned": False,
        "new_model_constructions": 1, "native_forward_attempts_at_initialization": 0}
    need(all(type(lock.get(k)) is type(v) and lock[k] == v for k, v in expected.items()), "FROZEN_LV60_NATIVE_LOCK")
    for name, pin in native.OBSERVER_FILES[PROVIDER].items():
        if name != "pytorch_model.bin":
            raw = regular(root / "model-metadata" / name)
            need({"bytes": len(raw), "sha256": sha(raw)} == pin, "FROZEN_LV60_METADATA")
    need(lock["preprocessor_config"] == read(root / "model-metadata/preprocessor_config.json"), "PINNED_PREPROCESSOR")
    return read(root / "model-metadata/vocab.json")


def _verify_prepared(directory, record, row, reference):
    n = row["pcm_samples"]
    expected = {"sample_rate": 16000, "pcm_samples": n, "normalized_shape": [1, n],
        "normalized_dtype": "torch.float32", "attention_mask_shape": [1, n], "attention_mask_dtype": "torch.int32",
        "attention_mask_sum": n, "attention_mask_unique_values": [1], "valid_input_samples": n,
        "padded_input_samples": n, "right_padding_samples": 0, "normalization_scope": FLAGS["input_scope"]}
    need(all(record["processor"].get(k) == v for k, v in expected.items()), "WHOLE_NATIVE_PREPARED_GEOMETRY")
    for key, name in (("normalized_input", "normalized-input.f32"), ("attention_mask", "attention-mask.i32")):
        raw = regular(directory / name)
        need(len(raw) == n * 4 and record[key] == {"path": name, "bytes": len(raw), "sha256": sha(raw)},
             "SAVED_PREPARED_ARRAY_BYTES")
        need(sha(raw) == reference[key + "_sha256"], "IDENTICAL_XLSR_PREPARED_INPUT")


def observe(plan, source_root, output, observer_run, backend_factory=None):
    _guard(plan, observer_run)
    proof = source_readback(source_root, plan)
    root, source_root = Path(output).absolute(), Path(source_root).absolute()
    assets = root.parent / (root.name + "-private-assets")
    for path in (root, assets):
        need(not path.exists() and path.parent.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)),
             "FRESH_PRIVATE_DIRECTORY")
        need(path != source_root and path not in source_root.parents and source_root not in path.parents,
             "SOURCE_AND_OBSERVER_SEPARATE")
    root.mkdir(mode=0o700)
    rows, source_sha = proof["panel_rows"], proof["source_manifest_sha256"]
    common = {"schema": SCHEMA, "campaign_id": CAMPAIGN, "provider": PROVIDER, "run": observer_run,
        "source_run": plan["source_run"], "source_sha256": source_sha, "denominator": 24, **FLAGS}
    ledger, failure = saved.initial_rows(rows), None
    write(root / "STARTED.json", common | {"maximum_calls": 24})
    try:
        write(root / "MODEL_LOAD_STARTED.json", common | {"actual_forward_attempts": 0})
        prepare, forward, materialize, lock = (backend_factory or native._load_native)(PROVIDER, assets, root)
        write(root / "MODEL_LOCK.json", lock)
        _verify_lock(root)
        for row, progress in zip(rows, ledger):
            directory = root / "rows" / row["id"]
            directory.mkdir(parents=True)
            base = _base(row, plan, observer_run, source_sha, _reference(source_root, row))
            try:
                write(directory / "STARTED.json", base)
                progress["input_started"] = True
                pcm = regular(within(source_root, row["pcm_path"]))
                need(sha(pcm) == row["pcm_sha256"] and len(pcm) == 2 * row["pcm_samples"], "UNCHANGED_WHOLE_PCM")
                geometry = {"pcm_samples": row["pcm_samples"], "full_to_crop_offset_16k": 0}
                batch, facts, normalized, mask = prepare(pcm, geometry)
                processor = {"processor": facts | {"normalization_scope": FLAGS["input_scope"]},
                    "normalized_input": put(directory / "normalized-input.f32", normalized),
                    "attention_mask": put(directory / "attention-mask.i32", mask)}
                write(directory / "INPUT.json", base | processor)
                _verify_prepared(directory, processor, row, base["xlsr_reference"])
                write(directory / "FORWARD_STARTED.json", base)
                progress["forward_attempts"] = 1
                result = forward(batch)
                progress["forward_returned"] = True
                write(directory / "FORWARD_RETURNED.json", base)
                values, logits = materialize(result, geometry)
                capture = base | processor | {"logits": put(directory / "logits.f32", logits), "values": values,
                    "frame_geometry": native.frame_geometry(row["pcm_samples"], 0), "status": "success",
                    "input_coverage_complete": True, "decoder_output_complete": True}
                pin = write(directory / "CAPTURE.json", capture)
                progress.update(status="success", capture_saved=True, capture_sha256=pin["sha256"])
            except BaseException as exc:
                progress.update(status="failed", error=type(exc).__name__ + ": " + str(exc)[:2000])
                write(directory / "FAILED.json", base | progress)
                if not isinstance(exc, Exception):
                    raise
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
    finally:
        if assets.exists():
            shutil.rmtree(assets)
    counts = _counts(ledger)
    result = common | counts | {"status": "COMPLETE" if counts["completed_count"] == 24 and failure is None else "INCOMPLETE",
                              "initialization_error": failure, "rows": ledger}
    write(root / "RESULT.json", result)
    write(root / "FILES.json", {"schema": SCHEMA + "/files", "files": file_records(root)})
    return result


def _counts(ledger):
    result = {"actual_forward_attempts": sum(r["forward_attempts"] for r in ledger),
        "native_forward_returns": sum(r["forward_returned"] for r in ledger),
        "completed_count": sum(r["capture_saved"] for r in ledger),
        "failed_count": sum(r["status"] == "failed" for r in ledger),
        "not_started_count": sum(r["status"] == "not_started" for r in ledger)}
    need(len(ledger) == 24 and 0 <= result["completed_count"] <= result["native_forward_returns"]
         <= result["actual_forward_attempts"] <= 24, "FIXED_SINGLE_PASS_CALL_CAP")
    return result


def readback(source_root, observer_root, plan, expected_run=None):
    proof = source_readback(source_root, plan)
    root = None if observer_root is None else Path(observer_root)
    terminal = bool(root and (root / "RESULT.json").is_file() and (root / "FILES.json").is_file())
    result = read(root / "RESULT.json") if terminal else None
    started = read(root / "STARTED.json") if root and (root / "STARTED.json").is_file() else None
    run = expected_run or (started["run"] if started else None)
    need(run is not None and run != plan["source_run"], "DISTINCT_EXPECTED_OBSERVER_RUN")
    common = {"schema": SCHEMA, "campaign_id": CAMPAIGN, "provider": PROVIDER, "run": run,
        "source_run": plan["source_run"], "source_sha256": proof["source_manifest_sha256"], "denominator": 24, **FLAGS}
    vocab = _verify_lock(root) if root and (root / "MODEL_LOCK.json").is_file() else None
    if root and root.exists():
        need(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), "REGULAR_OBSERVER_ROOT")
        need({p.name for p in root.iterdir()} <= {"STARTED.json", "MODEL_LOAD_STARTED.json", "MODEL_LOCK.json",
             "RESULT.json", "FILES.json", "rows", "model-metadata"}, "NO_EXTRA_OBSERVER_FILES")
        if (root / "rows").exists():
            need({p.name for p in (root / "rows").iterdir()} <= {r["id"] for r in proof["panel_rows"]}, "NO_EXTRA_PANEL_ROWS")
        if started:
            need(started == common | {"maximum_calls": 24}, "START_BINDING")
    if terminal:
        need(read(root / "FILES.json") == {"schema": SCHEMA + "/files", "files": file_records(root)}, "EXACT_DIAGNOSTIC_CLOSURE")
        need(all(result.get(k) == v for k, v in common.items()) and started is not None
             and read(root / "MODEL_LOAD_STARTED.json") == common | {"actual_forward_attempts": 0}, "TERMINAL_RUN_BINDING")
        need([r["id"] for r in result["rows"]] == [r["id"] for r in proof["panel_rows"]], "ALL_24_PLANNED_ROWS")
    ledger, observations = [], []
    for index, row in enumerate(proof["panel_rows"]):
        directory = root / "rows" / row["id"] if root else None
        exists = lambda n: bool(directory and (directory / n).is_file())
        if directory and directory.exists():
            need({p.name for p in directory.iterdir()} <= {"STARTED.json", "INPUT.json", "FORWARD_STARTED.json",
                "FORWARD_RETURNED.json", "FAILED.json", "CAPTURE.json", "normalized-input.f32", "attention-mask.i32", "logits.f32"},
                 "NO_EXTRA_ROW_CALL_OR_CAPTURE")
        base = _base(row, plan, run, proof["source_manifest_sha256"], _reference(source_root, row))
        progress = {"id": row["id"], "status": "unknown", "input_started": exists("STARTED.json"),
                    "forward_attempts": int(exists("FORWARD_STARTED.json")), "forward_returned": exists("FORWARD_RETURNED.json"),
                    "capture_saved": exists("CAPTURE.json")}
        for filename in ("STARTED.json", "INPUT.json", "FORWARD_STARTED.json", "FORWARD_RETURNED.json", "FAILED.json", "CAPTURE.json"):
            if exists(filename):
                value = read(directory / filename)
                need(all(value.get(k) == v for k, v in base.items()), "ROW_PROVENANCE_BINDING")
        if exists("FAILED.json"):
            failed = read(directory / "FAILED.json")
            need(type(failed.get("forward_returned")) is bool and failed.get("capture_saved") is False
                 and type(failed.get("forward_attempts")) is int and failed["forward_attempts"] == progress["forward_attempts"], "FAILURE_LEDGER")
            progress["forward_returned"] |= failed["forward_returned"]
        need(not progress["forward_attempts"] or progress["input_started"], "START_BEFORE_CALL")
        need(not progress["forward_returned"] or progress["forward_attempts"] == 1, "CALL_BEFORE_RETURN")
        if progress["forward_attempts"]:
            need(exists("INPUT.json"), "CALL_REQUIRES_SAVED_INPUT")
            _verify_prepared(directory, read(directory / "INPUT.json"), row, base["xlsr_reference"])
        observation = {"id": row["id"], "status": "missing", "input_pcm_sha256": row["pcm_sha256"],
            "input_samples": row["pcm_samples"], "source_run": plan["source_run"], "run": run, **FLAGS}
        if progress["capture_saved"]:
            need(progress["forward_returned"] and not exists("FAILED.json") and vocab is not None, "CAPTURE_AFTER_NATIVE_RETURN")
            capture = read(directory / "CAPTURE.json")
            need(capture.get("status") == "success" and capture.get("input_coverage_complete") is True
                 and capture.get("decoder_output_complete") is True, "WHOLE_COMPLETE_CAPTURE")
            pcm = regular(within(source_root, row["pcm_path"]))
            tokens = saved._verify_phone(directory, capture, row, pcm, vocab)
            need(read(directory / "INPUT.json") == base | {k: capture[k] for k in ("processor", "normalized_input", "attention_mask")},
                 "SAVED_PRE_FORWARD_INPUT")
            need(capture["normalized_input"]["sha256"] == base["xlsr_reference"]["normalized_input_sha256"]
                 and capture["attention_mask"]["sha256"] == base["xlsr_reference"]["attention_mask_sha256"], "IDENTICAL_XLSR_PREPARED_INPUT")
            progress.update(status="success", capture_sha256=sha(regular(directory / "CAPTURE.json")))
            observation.update(status="success", phone_tokens=tokens, capture_path="rows/" + row["id"] + "/CAPTURE.json",
                capture_sha256=progress["capture_sha256"], normalized_input_sha256=capture["normalized_input"]["sha256"],
                xlsr_reference=base["xlsr_reference"], input_coverage_complete=True, decoder_output_complete=True)
        elif exists("FAILED.json"):
            progress["status"] = observation["status"] = "failed"
        elif terminal:
            need(not progress["input_started"], "STARTED_REQUIRES_TERMINAL_ROW")
            progress["status"] = "not_started"
        if terminal:
            need(all(type(result["rows"][index].get(k)) is type(v) and result["rows"][index][k] == v
                     for k, v in progress.items()), "TERMINAL_LEDGER_RECONSTRUCTED")
        ledger.append(progress)
        observations.append(observation)
    counts = _counts(ledger)
    if terminal:
        need(all(type(result.get(k)) is int and result[k] == v for k, v in counts.items()), "ACTUAL_CALL_COUNTS")
        need(result["status"] == ("COMPLETE" if counts["completed_count"] == 24 and result["initialization_error"] is None
                                  else "INCOMPLETE"), "COMPLETENESS_RECONSTRUCTED")
    summary = common | counts | {"status": result["status"] if terminal else "INCOMPLETE_UNKNOWN_COUNTERS",
        "terminal_receipt_present": terminal, "actual_forward_attempts": counts["actual_forward_attempts"] if terminal else None,
        "native_forward_returns": counts["native_forward_returns"] if terminal else None,
        "not_started_count": counts["not_started_count"] if terminal else None,
        "attempts_lower_bound": counts["actual_forward_attempts"],
        "attempts_upper_bound": counts["actual_forward_attempts"] if terminal else 24,
        "unknown_count": sum(r["status"] == "unknown" for r in ledger), "missing_terminal_is_zero": False}
    return {"summary": summary, "rows": observations, "new_model_calls": 0}
