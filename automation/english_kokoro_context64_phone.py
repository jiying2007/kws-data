"""One parent-PCM phone observation per fixed model; no child-phone claim.

The two native loaders, processor, model call and materialization come from
the exact publicly exercised context32 implementation.  This wrapper supplies
a new public execution guard, source binding and finite parent ledger.  The
saved-byte reader uses only standard-library functions and never loads a
processor or model.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import re
import struct
import sys
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import english_kokoro_context32_mask_observe as native
import english_kokoro_context32_mask_compare as math_reader

SCHEMA = "english-kokoro-context64-phone-v1"
CAMPAIGN_ID = "kokoro-exact-child-context-controls64-v1"
MAIN_BEFORE = "2ca611210126e2f78939e6e50f7cd0157c76d5dd"
WORKFLOW = "jiying2007/kws-data/.github/workflows/english-kokoro-context64-once.yml@refs/heads/main"
NATIVE_PATH = "automation/english_kokoro_context32_mask_observe.py"
NATIVE_SHA = "3f184bdeb7536f260b2d4e3ee77476d95590d05007ae251bcb93c51ce2be01af"
MATH_PATH = "automation/english_kokoro_context32_mask_compare.py"
MATH_SHA = "4f676ab681e56d5d56f58269865b99f29c8d0e13eeeb0e07b4f16934ecfe0c6d"
PROVIDERS = native.PROVIDERS
OBSERVERS = native.OBSERVERS
OBSERVER_FILES = native.OBSERVER_FILES
PROCESSOR_SOURCE_FILES = native.PROCESSOR_SOURCE_FILES
RUNTIME_VERSIONS = native.RUNTIME_VERSIONS
PARENT_COUNT = 48
INPUT_SCOPE = "parent_full_16k_pcm"
need, sha, canonical = native.need, native.sha, native.canonical
regular, within, loads = math_reader.regular, math_reader.within, math_reader.loads
once, put = native.once, native.put


def checked_equal(actual, expected, code):
    need(type(actual) is dict and all(k in actual and type(actual[k]) is type(v)
         and actual[k] == v for k, v in expected.items()), code)


def source_preflight(plan=None):
    for name, pin in ((NATIVE_PATH, NATIVE_SHA), (MATH_PATH, MATH_SHA)):
        need(sha(regular(ROOT / name)) == pin, "IMMUTABLE_NATIVE_OR_MATH_HELPER: " + name)
        if plan is not None:
            need(plan.get("helper_sha256", {}).get(name) == pin, "PLAN_HELPER_PIN: " + name)
    native.source_preflight()
    if plan is not None:
        need(plan.get("phone_code_sha256") == sha(regular(Path(__file__))), "NEW_PHONE_SOURCE_PIN")
        need(plan.get("observers") == OBSERVERS and plan.get("observer_files") == OBSERVER_FILES,
             "UNCHANGED_PHONE_MODELS_AND_ASSETS")
    return {"schema": SCHEMA, "source_check": "PASS", "new_model_calls": 0,
            "new_processor_calls": 0, "new_network_calls": 0,
            "native_helper_sha256": NATIVE_SHA, "math_helper_sha256": MATH_SHA}


def public_guard(plan, provider, run_identity, env=None):
    env = os.environ if env is None else env
    source_preflight(plan)
    need(provider in PROVIDERS and plan.get("enabled") is True
         and plan.get("campaign_id") == CAMPAIGN_ID and plan.get("main_before") == MAIN_BEFORE,
         "EXACT_ENABLED_NEW_CAMPAIGN")
    need(plan.get("parent_phone_calls_per_provider") == 48
         and plan.get("child_text_calls_per_provider") == 64
         and plan.get("native_tts_request_budget") == 48
         and plan.get("total_high_level_model_calls_max") == 336
         and plan.get("retry_count") == 0 and plan.get("training_allowed") is False
         and plan.get("automatic_expansion") is False, "FINITE_DIAGNOSTIC_BUDGET")
    actual = {"repository": env.get("GITHUB_REPOSITORY"), "commit": env.get("GITHUB_SHA"),
              "run_id": env.get("GITHUB_RUN_ID"), "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
              "event": env.get("GITHUB_EVENT_NAME")}
    need(env.get("GITHUB_ACTIONS") == "true" and env.get("GITHUB_WORKFLOW_REF") == WORKFLOW
         and actual == run_identity and actual["repository"] == "jiying2007/kws-data"
         and actual["event"] == "push" and actual["run_attempt"] == "1"
         and env.get("GITHUB_REF") == "refs/heads/main"
         and type(actual["commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", actual["commit"])
         and type(actual["run_id"]) is str and re.fullmatch(r"[1-9][0-9]*", actual["run_id"]),
         "FIRST_PUBLIC_MAIN_WORKFLOW_ONLY")
    event = loads(regular(Path(env["GITHUB_EVENT_PATH"])))
    need(event.get("before") == MAIN_BEFORE and event.get("after") == actual["commit"]
         and event["after"] != event["before"]
         and event.get("repository", {}).get("full_name") == actual["repository"]
         and event["repository"].get("private") is False, "EXACT_PUBLIC_SUCCESSOR")
    return actual


def validate_inputs(source_root, parent_rows, run_identity):
    """Bind only complete fresh parents; retain all failed-source denominators."""
    source_root = Path(source_root)
    raw_result = regular(source_root / "source/RESULT.json", 16 * 1024**2)
    result = loads(raw_result)
    need(result.get("run") == run_identity and result.get("parents") == parent_rows
         and type(parent_rows) is list and len(parent_rows) == PARENT_COUNT,
         "EXACT_ACTUAL_PARENT_ROSTER")
    ids, pcms, checked = set(), set(), []
    for row in parent_rows:
        need(type(row) is dict and type(row.get("id")) is str
             and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["id"])
             and row["id"] not in ids, "UNIQUE_PARENT_ID")
        ids.add(row["id"])
        status = row.get("generation_status")
        need(status in ("success", "failed", "not_started"), "EXPLICIT_PARENT_GENERATION_STATE")
        if status != "success":
            continue
        samples = row.get("pcm_samples")
        need(type(samples) is int and 400 <= samples <= 192000
             and row.get("sample_rate") == 16000 and row.get("channels") == 1
             and row.get("sample_width") == 2, "FULL_PARENT_PCM_GEOMETRY")
        pcm = regular(within(source_root, row["pcm_path"]))
        wav = regular(within(source_root, row["wav_path"]))
        need(len(pcm) == samples * 2 and sha(pcm) == row.get("pcm_sha256")
             and sha(wav) == row.get("wav_sha256") and row["pcm_sha256"] not in pcms,
             "UNIQUE_EXACT_PARENT_PCM_WAV")
        with wave.open(io.BytesIO(wav), "rb") as stream:
            need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                  stream.getnframes(), stream.getcomptype()) == (1, 2, 16000, samples, "NONE")
                 and stream.readframes(samples) == pcm, "PARENT_WAV_PCM_EQUAL")
        pcms.add(row["pcm_sha256"])
        checked.append((row, pcm))
    return checked, sha(raw_result)


def initial_rows(parent_rows):
    return [{"id": row["id"], "parent_id": row["id"], "status": "NOT_STARTED_NO_RETRY",
             "input_started": False, "forward_attempts": 0, "forward_returned": False,
             "capture_saved": False, "source_available": row["generation_status"] == "success",
             "retry_count": 0} for row in parent_rows]


def base_fields(row, provider, run_identity, source_sha):
    return {"schema": SCHEMA, "run": run_identity, "id": row["id"], "parent_id": row["id"],
            "provider": provider, "observer_identity": native.observer_identity(provider),
            "input_scope": INPUT_SCOPE, "parent_pcm_sha256": row["pcm_sha256"],
            "parent_wav_sha256": row["wav_sha256"], "parent_samples": row["pcm_samples"],
            "input_pcm_sha256": row["pcm_sha256"], "input_wav_sha256": row["wav_sha256"],
            "input_samples": row["pcm_samples"], "source_result_sha256": source_sha,
            "phone_code_sha256": sha(regular(Path(__file__))),
            "native_helper_sha256": NATIVE_SHA,
            "native_processor_source_files": PROCESSOR_SOURCE_FILES,
            "target_conditioned": False, "requested_text_passed": False,
            "training_allowed": False, "diagnostic_only": True, "retry_count": 0,
            "child_independent_phone_evidence": False}


def execute_rows(checked, output, provider, run_identity, source_sha, backend, ledger):
    """No extra calls for failed serialization or incomplete model returns."""
    prepare, forward, materialize = backend
    by_id = {row["id"]: row for row in ledger}
    for row, pcm in checked:
        progress = by_id[row["id"]]
        directory = Path(output) / row["id"]
        directory.mkdir(mode=0o700)
        base = base_fields(row, provider, run_identity, source_sha)
        # The native component receives only the PCM geometry. It receives no
        # text, class, cell, preset, crop position or reference hypothesis.
        geometry_input = {"pcm_samples": row["pcm_samples"], "full_to_crop_offset_16k": 0}
        try:
            need(sum(x["forward_attempts"] for x in ledger) < PARENT_COUNT, "PARENT_FORWARD_CEILING")
            once(directory / "STARTED.json", base | {"forward_attempts": 0, "forward_returned": False})
            progress.update(input_started=True, status="STARTED")
            batch, facts, normalized_raw, mask_raw = prepare(pcm, geometry_input)
            facts = dict(facts) | {"normalization_scope": "Entire exact parent PCM; no child normalization or child forward."}
            normalized = put(directory / "normalized-input.f32", normalized_raw) | {
                "dtype": "float32-little-endian", "order": "C", "shape": [1, row["pcm_samples"]]}
            mask = put(directory / "attention-mask.i32", mask_raw) | {
                "dtype": "int32-little-endian", "order": "C", "shape": [1, row["pcm_samples"]]}
            need(normalized["bytes"] == mask["bytes"] == 4 * row["pcm_samples"],
                 "EXACT_PROCESSOR_ARRAY_BYTES")
            once(directory / "INPUT.json", base | {"processor": facts,
                 "normalized_input": normalized, "attention_mask": mask})
            once(directory / "FORWARD_STARTED.json", base | {"forward_attempts": 1, "forward_returned": False})
            progress["forward_attempts"] = 1
            returned = forward(batch)
            progress["forward_returned"] = True
            once(directory / "FORWARD_RETURNED.json", base | {"forward_attempts": 1, "forward_returned": True})
            values, logits_raw = materialize(returned, geometry_input)
            geometry = native.frame_geometry(row["pcm_samples"], 0)
            frames = geometry["frames"]
            need(values["shape"] == [1, frames, 392] and len(values["frame_ids"]) == frames
                 and all(type(i) is int and 0 <= i < 392 for i in values["frame_ids"])
                 and len(logits_raw) == frames * 392 * 4, "FULL_PARENT_LOGIT_GEOMETRY")
            logits = put(directory / "logits.f32", logits_raw) | {
                k: values[k] for k in ("shape", "dtype", "order")}
            ids = once(directory / "frame-ids.json", {"schema": SCHEMA, "id": row["id"],
                       "parent_id": row["id"], "frame_ids": values["frame_ids"]})
            capture = base | {"status": "success", "forward_attempts": 1, "forward_returned": True,
                "input_coverage_complete": True, "decoder_output_complete": True,
                "processor": facts, "normalized_input": normalized, "attention_mask": mask,
                "logits": logits, "frame_ids": {"path": "frame-ids.json"} | ids,
                "phone_tokens": values["phone_tokens"],
                "argmax_special_frame_counts": values["argmax_special_frame_counts"],
                "frame_geometry": geometry, "ctc_helper_sha256": native.CTC_HELPER_SHA256,
                "blank_id": 0, "skipped_special_ids": [0, 1, 2, 3],
                "collapse_before_blank_deletion": True, "old_labels_modified": False,
                "whole_target_pronunciation_gold": False}
            pin = once(directory / "CAPTURE.json", capture)
            progress.update(status="success", capture_saved=True,
                capture_path=row["id"] + "/CAPTURE.json", capture_sha256=pin["sha256"])
        except BaseException as exc:
            progress.update(status="failed", failure=type(exc).__name__ + ": " + str(exc)[:2000])
            try:
                once(directory / "FAILED.json", base | progress)
            except Exception as write_error:
                progress["failure_receipt_write_error"] = type(write_error).__name__ + ": " + str(write_error)[:500]
            if not isinstance(exc, Exception):
                raise


def finish(output, provider, run_identity, ledger, failure, source_sha):
    attempts = sum(row["forward_attempts"] for row in ledger)
    returns = sum(row["forward_returned"] for row in ledger)
    complete = sum(row["capture_saved"] for row in ledger)
    need(len(ledger) == PARENT_COUNT and 0 <= complete <= returns <= attempts <= PARENT_COUNT
         and all(type(row["forward_attempts"]) is int and row["forward_attempts"] in (0, 1)
                 and type(row["forward_returned"]) is bool
                 and (not row["forward_returned"] or row["forward_attempts"] == 1)
                 for row in ledger), "PARENT_LEDGER_INVARIANTS")
    passed = failure is None and complete == returns == attempts == PARENT_COUNT
    result = {"schema": SCHEMA, "run": run_identity, "provider": provider,
        "status": "COMPLETE_DIAGNOSTIC" if passed else "INCOMPLETE_DIAGNOSTIC_NO_RETRY",
        "input_scope": INPUT_SCOPE, "declared_count": PARENT_COUNT,
        "valid_source_count": sum(row["source_available"] for row in ledger),
        "input_started": sum(row["input_started"] for row in ledger),
        "actual_forward_attempts": attempts, "native_forward_returns": returns, "completed": complete,
        "failed": sum(row["status"] == "failed" for row in ledger),
        "not_started": sum(row["status"] == "NOT_STARTED_NO_RETRY" for row in ledger),
        "retry_count": 0, "training_allowed": False, "old_labels_modified": False,
        "child_independent_phone_evidence": False, "source_result_sha256": source_sha,
        "failure": failure, "rows": ledger}
    once(Path(output) / "RESULT.json", result)
    return result


def run(*, provider, source_root, output_root, asset_root, plan, run_identity, parent_rows):
    public_guard(plan, provider, run_identity)
    checked, source_sha = validate_inputs(source_root, parent_rows, run_identity)
    output = Path(output_root)
    need(output.is_absolute() and not output.exists() and output.parent.is_dir()
         and not any(p.is_symlink() for p in output.parents), "CREATE_ONLY_NEW_PHONE_OUTPUT")
    output.mkdir(mode=0o700)
    ledger = initial_rows(parent_rows)
    once(output / "STARTED.json", {"schema": SCHEMA, "run": run_identity, "provider": provider,
         "input_scope": INPUT_SCOPE, "source_result_sha256": source_sha,
         "declared_count": PARENT_COUNT, "valid_source_count": len(checked),
         "maximum_calls": PARENT_COUNT, "retry_count": 0,
         "phone_code_sha256": sha(regular(Path(__file__))), "native_helper_sha256": NATIVE_SHA})
    failure = None
    try:
        if checked:
            once(output / "MODEL_LOAD_STARTED.json", {"schema": SCHEMA, "run": run_identity,
                 "provider": provider, "actual_forward_attempts": 0})
            prepare, forward, materialize, lock = native._load_native(provider, asset_root, output)
            once(output / "MODEL_LOCK.json", {"schema": SCHEMA + "/model-lock", "run": run_identity,
                 "provider": provider, "observer_identity": native.observer_identity(provider),
                 "source_result_sha256": source_sha, "phone_code_sha256": sha(regular(Path(__file__))),
                 "native_initializer_origin": {"path": NATIVE_PATH, "sha256": NATIVE_SHA},
                 "native_lock": lock, "new_model_constructions": 1,
                 "native_forward_attempts_at_initialization": 0})
            execute_rows(checked, output, provider, run_identity, source_sha,
                         (prepare, forward, materialize), ledger)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
    return finish(output, provider, run_identity, ledger, failure, source_sha)


def _verify_lock(root, provider, source_sha, plan, run_identity):
    checked_equal(loads(regular(root / "MODEL_LOAD_STARTED.json")), {
        "schema": SCHEMA, "run": run_identity, "provider": provider,
        "actual_forward_attempts": 0}, "MODEL_LOAD_STARTED_IDENTITY")
    lock = loads(regular(root / "MODEL_LOCK.json"))
    checked_equal(lock, {"schema": SCHEMA + "/model-lock", "run": run_identity, "provider": provider,
        "observer_identity": native.observer_identity(provider), "source_result_sha256": source_sha,
        "phone_code_sha256": plan["phone_code_sha256"],
        "native_initializer_origin": {"path": NATIVE_PATH, "sha256": NATIVE_SHA},
        "new_model_constructions": 1, "native_forward_attempts_at_initialization": 0}, "PHONE_LOCK_BINDING")
    value = lock.get("native_lock")
    checked_equal(value, {"schema": native.SCHEMA, "model": OBSERVERS[provider],
        "files": OBSERVER_FILES[provider], "runtime_versions": RUNTIME_VERSIONS,
        "native_processor_source_files": PROCESSOR_SOURCE_FILES,
        "ctc_helper_sha256": native.CTC_HELPER_SHA256,
        "blank_id": 0, "skipped_special_ids": [0, 1, 2, 3], "collapse_before_blank_deletion": True,
        "weights_only": True, "strict_state_dict_loading": True, "eval": True,
        "model_parameters_modified": False, "trust_remote_code": False,
        "phonemizer_used": False, "target_conditioned": False,
        "new_model_constructions": 1, "native_forward_attempts_at_initialization": 0}, "UNCHANGED_NATIVE_MODEL_CONTRACT")
    for name, pin in OBSERVER_FILES[provider].items():
        if name != "pytorch_model.bin":
            raw = regular(root / "model-metadata" / name)
            need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "ACTUAL_NATIVE_METADATA_PIN")
    config = loads(regular(root / "model-metadata/preprocessor_config.json"))
    need(value.get("preprocessor_config") == config, "ACTUAL_PREPROCESSOR_CONFIG")
    _, expected_function = native.ctc_function(regular(HERE / "english_final_ipa_xlsr_once.py"))
    need(value.get("ctc_function_sha256") == expected_function, "ACTUAL_CTC_FUNCTION_PIN")
    return math_reader.vocabulary(regular(root / "model-metadata/vocab.json"))


def _verify_markers(directory, base):
    present = {}
    states = {"STARTED.json": (0, False), "FORWARD_STARTED.json": (1, False),
              "FORWARD_RETURNED.json": (1, True), "CAPTURE.json": (1, True)}
    for name in (*states, "FAILED.json"):
        path = directory / name
        if not path.exists():
            continue
        value = loads(regular(path))
        checked_equal(value, base, "ROW_RECEIPT_PARENT_RUN_IDENTITY")
        if name in states:
            a, returned = states[name]
            checked_equal(value, {"forward_attempts": a, "forward_returned": returned}, "ROW_RECEIPT_COUNTER")
        else:
            need(value.get("status") == "failed" and value.get("capture_saved") is False
                 and type(value.get("forward_attempts")) is int and value["forward_attempts"] in (0, 1)
                 and type(value.get("forward_returned")) is bool
                 and (not value["forward_returned"] or value["forward_attempts"] == 1),
                 "FAILED_RECEIPT_COUNTER")
        present[name] = value
    need(not ("CAPTURE.json" in present and "FAILED.json" in present), "NO_BOTH_FAILED_AND_CAPTURE")
    need("FORWARD_STARTED.json" not in present or "STARTED.json" in present, "FORWARD_REQUIRES_INPUT_START")
    need("FORWARD_RETURNED.json" not in present or "FORWARD_STARTED.json" in present, "RETURN_REQUIRES_FORWARD")
    need("CAPTURE.json" not in present or "FORWARD_RETURNED.json" in present, "CAPTURE_REQUIRES_RETURN")
    if "FAILED.json" in present:
        failed = present["FAILED.json"]
        need("STARTED.json" in present
             and failed["forward_attempts"] == int("FORWARD_STARTED.json" in present)
             and ("FORWARD_RETURNED.json" not in present or failed["forward_returned"]),
             "FAILED_RECEIPT_MATCHES_ACTUAL_MARKERS")
    return present


def _read_capture(directory, row, capture, vocab, pcm, base):
    checked_equal(capture, base, "CAPTURE_PARENT_IDENTITY")
    checked_equal(capture, {"status": "success", "forward_attempts": 1, "forward_returned": True,
        "input_coverage_complete": True, "decoder_output_complete": True,
        "ctc_helper_sha256": native.CTC_HELPER_SHA256, "blank_id": 0,
        "skipped_special_ids": [0, 1, 2, 3], "collapse_before_blank_deletion": True,
        "old_labels_modified": False, "whole_target_pronunciation_gold": False}, "CAPTURE_COMPLETE_NO_GOLD")
    input_value = loads(regular(directory / "INPUT.json"))
    checked_equal(input_value, base, "PROCESSOR_PARENT_IDENTITY")
    need(all(input_value.get(k) == capture.get(k) for k in
             ("processor", "normalized_input", "attention_mask")), "PROCESSOR_CAPTURE_JOIN")
    checked_equal(capture.get("normalized_input"), {
        "dtype": "float32-little-endian", "order": "C", "shape": [1, row["pcm_samples"]]},
        "SAVED_NORMALIZED_INPUT_FORMAT")
    geometry = native.frame_geometry(row["pcm_samples"], 0)
    need(capture.get("frame_geometry") == geometry, "FULL_PARENT_FRAME_GRID")
    frames = geometry["frames"]
    descriptor = capture.get("logits")
    checked_equal(descriptor, {"shape": [1, frames, 392], "dtype": "float32-little-endian",
                              "order": "C"}, "SAVED_FULL_LOGIT_FORMAT")
    raw = math_reader.pinned(directory, descriptor, "logits.f32")
    ids = math_reader.argmax_ids(raw, frames)
    ids_doc = loads(math_reader.pinned(directory, capture.get("frame_ids"), "frame-ids.json"))
    checked_equal(ids_doc, {"schema": SCHEMA, "id": row["id"], "parent_id": row["id"],
                           "frame_ids": ids}, "EXACT_SAVED_PARENT_ARGMAX_IDS")
    need(all(type(value) is int for value in ids_doc["frame_ids"]), "INTEGER_IDS")
    tokens = [r["token"] for r in math_reader.ctc_runs(ids, vocab)]
    need(tokens == capture.get("phone_tokens") and 1 <= len(tokens) <= min(300, frames)
         and capture.get("argmax_special_frame_counts") == {str(k): ids.count(k) for k in (0, 1, 2, 3)},
         "EXACT_SAVED_PARENT_CTC")
    processor = math_reader.processor_readback(directory, row, capture, pcm)
    return {"status": "success", "capture": capture, "capture_sha256": sha(regular(directory / "CAPTURE.json")),
            "frame_ids": ids, "phone_tokens": tokens, "vocab": vocab,
            "by_id": {value: key for key, value in vocab.items()}, "frames": frames,
            "processor_readback": processor, "full_logit_values_verified": frames * 392,
            "input_scope": INPUT_SCOPE}


def _verify_packet_closure(root, known_terminal):
    path = root / "FILES.json"
    need(path.is_file() or not known_terminal, "PHONE_TERMINAL_REQUIRES_CLOSED_FILES")
    if not path.is_file():
        return
    manifest = loads(regular(path, 16 * 1024**2))
    files, total = [], 0
    for child in sorted(root.rglob("*")):
        if child.is_file() and child != path:
            need(not any(p.is_symlink() for p in (child, *child.parents))
                 and child.stat().st_size <= 256 * 1024**2, "BOUNDED_PHONE_PACKET_FILE")
            raw = child.read_bytes()
            total += len(raw)
            files.append({"path": child.relative_to(root).as_posix(),
                          "bytes": len(raw), "sha256": sha(raw)})
    need(len(files) <= 20000 and total <= 1024**3
         and manifest == {"schema": "english-kokoro-context64-once-v1/files", "files": files},
         "EXACT_CLOSED_PHONE_PACKET_FILES")


def readback(*, provider, observer_root, source_root, plan, run_identity, parent_rows):
    """Recompute actual saved evidence; no native backend is called here."""
    source_preflight(plan)
    need(provider in PROVIDERS, "KNOWN_PHONE_PROVIDER")
    checked, source_sha = validate_inputs(source_root, parent_rows, run_identity)
    root = Path(observer_root)
    need(not root.is_symlink(), "NO_OBSERVER_ROOT_SYMLINK")
    if root.exists():
        need(root.is_dir() and all(not p.is_symlink() for p in root.rglob("*")), "REGULAR_PROVIDER_TREE")
        allowed_dirs = {row["id"] for row in parent_rows} | {"model-metadata"}
        need(all(p.name in allowed_dirs for p in root.iterdir() if p.is_dir()),
             "NO_EXTRA_PARENT_OBSERVATIONS")
    available = {row["id"]: (row, pcm) for row, pcm in checked}
    known = (root / "RESULT.json").is_file()
    _verify_packet_closure(root, known)
    terminal = loads(regular(root / "RESULT.json")) if known else None
    if known:
        checked_equal(terminal, {"schema": SCHEMA, "run": run_identity, "provider": provider,
            "input_scope": INPUT_SCOPE, "declared_count": PARENT_COUNT,
            "valid_source_count": len(checked), "retry_count": 0, "training_allowed": False,
            "old_labels_modified": False, "child_independent_phone_evidence": False,
            "source_result_sha256": source_sha}, "PHONE_TERMINAL_IDENTITY")
        need(type(terminal.get("rows")) is list and len(terminal["rows"]) == PARENT_COUNT
             and [r.get("id") for r in terminal["rows"]] == [r["id"] for r in parent_rows],
             "COMPLETE48_TERMINAL_LEDGER")
        ledger = {r["id"]: r for r in terminal["rows"]}
    else:
        ledger = {}
    if (root / "STARTED.json").exists():
        checked_equal(loads(regular(root / "STARTED.json")), {
            "schema": SCHEMA, "run": run_identity, "provider": provider, "input_scope": INPUT_SCOPE,
            "source_result_sha256": source_sha, "declared_count": PARENT_COUNT,
            "valid_source_count": len(checked), "maximum_calls": PARENT_COUNT, "retry_count": 0,
            "phone_code_sha256": plan["phone_code_sha256"], "native_helper_sha256": NATIVE_SHA},
            "PROVIDER_STARTED_IDENTITY")
    elif known:
        need(False, "TERMINAL_REQUIRES_PROVIDER_STARTED")
    vocab = _verify_lock(root, provider, source_sha, plan, run_identity) if (root / "MODEL_LOCK.json").exists() else None
    result_rows = {}
    counters = {"input_started": 0, "actual_forward_attempts": 0, "native_forward_returns": 0,
                "completed": 0, "failed": 0, "not_started": 0}
    lower = 0
    for original in parent_rows:
        ident = original["id"]
        directory = root / ident
        value = {"status": "UNAVAILABLE", "capture": None, "frame_ids": None,
                 "phone_tokens": None, "vocab": None, "by_id": None}
        markers = {}
        if ident in available:
            row, pcm = available[ident]
            base = base_fields(row, provider, run_identity, source_sha)
            markers = _verify_markers(directory, base)
            if markers:
                need((root / "STARTED.json").is_file() and vocab is not None,
                     "ROW_REQUIRES_PROVIDER_AND_VERIFIED_MODEL")
            if "CAPTURE.json" in markers:
                value = _read_capture(directory, row, markers["CAPTURE.json"], vocab, pcm, base)
            elif "FAILED.json" in markers:
                value["status"] = "failed"
                value["failure"] = markers["FAILED.json"].get("failure")
        else:
            need(not directory.exists(), "NO_MODEL_RECEIPTS_FOR_FAILED_SOURCE")
        lower += int("FORWARD_STARTED.json" in markers)
        if known:
            progress = ledger[ident]
            # A native return is recorded in memory before its marker is
            # written. If that write fails, the bound FAILED receipt and the
            # terminal ledger still record the consumed return. Missing
            # receipts alone never establish a return, and a success capture
            # continues to require the actual return marker.
            returned = ("FORWARD_RETURNED.json" in markers or
                        markers.get("FAILED.json", {}).get("forward_returned") is True)
            checked_equal(progress, {"id": ident, "parent_id": ident,
                "source_available": ident in available, "retry_count": 0,
                "input_started": "STARTED.json" in markers,
                "forward_attempts": int("FORWARD_STARTED.json" in markers),
                "forward_returned": returned,
                "capture_saved": "CAPTURE.json" in markers}, "LEDGER_MATCHES_ACTUAL_MARKERS")
            status = ("success" if "CAPTURE.json" in markers else
                      "failed" if "FAILED.json" in markers else "NOT_STARTED_NO_RETRY")
            need(progress.get("status") == status, "EXPLICIT_TERMINAL_ROW_STATE")
            if status == "success":
                need(progress.get("capture_path") == ident + "/CAPTURE.json"
                     and progress.get("capture_sha256") == value["capture_sha256"],
                     "LEDGER_CAPTURE_SHA")
            value["status"] = status
            counters["input_started"] += progress["input_started"]
            counters["actual_forward_attempts"] += progress["forward_attempts"]
            counters["native_forward_returns"] += progress["forward_returned"]
            counters["completed"] += progress["capture_saved"]
            counters["failed"] += status == "failed"
            counters["not_started"] += status == "NOT_STARTED_NO_RETRY"
        result_rows[ident] = value
    if known:
        checked_equal(terminal, counters, "TERMINAL_COUNTERS_RECOUNTED")
        need(0 <= counters["completed"] <= counters["native_forward_returns"]
             <= counters["actual_forward_attempts"] <= len(checked)
             and counters["completed"] + counters["failed"] + counters["not_started"] == PARENT_COUNT,
             "TERMINAL_FINITE_COUNTER_RELATION")
        complete = terminal.get("failure") is None and counters["completed"] == PARENT_COUNT
        need(terminal.get("status") == ("COMPLETE_DIAGNOSTIC" if complete else "INCOMPLETE_DIAGNOSTIC_NO_RETRY"),
             "TERMINAL_COMPLETE_STATUS")
        lower = upper = counters["actual_forward_attempts"]
        status = terminal["status"]
    else:
        complete, upper, status = False, len(checked), "UNAVAILABLE_TERMINAL_COUNTERS_UNKNOWN"
        counters = {key: None for key in counters}
    summary = {"provider": provider, "declared_count": PARENT_COUNT, "complete": complete,
               **counters, "forward_attempts_lower_bound": lower,
               "forward_attempts_upper_bound": upper, "status": status,
               "input_scope": INPUT_SCOPE, "child_independent_phone_evidence": False}
    return {"schema": SCHEMA + "/saved-byte-readback", "summary": summary,
            "rows": result_rows, "source_result_sha256": source_sha,
            "new_model_calls": 0, "new_processor_calls": 0}
