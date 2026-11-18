"""Four frozen, blind observers of the same twelve complete short PCM inputs.

Only run() reaches a model. Import, source_check and readback use stdlib only.
The public worker guards the first main push; this module also checks that event.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = "english-pocket12-observer-v1"
TEXT_PROVIDERS = ("qwen", "whisper", "parakeet")
PROVIDERS = TEXT_PROVIDERS + ("phoneme",)
FLAGS = {"diagnostic_only": True, "training_allowed": False,
         "permanently_excluded_from_training": True, "target_conditioned": False}
HELPER_PINS = {'automation/english_blind_asr.py': 'e232cb219cabcc3877366bc446e5180d203d43b094695dba18fddee9b584669e',
 'automation/english_context64_policy_v2_reference.py': 'fde516d9f1f2457cdb0047e8d8aee6a667c68fa6ea21f83d22f135e4bc9069da',
 'automation/english_final_ipa_xlsr_once.py': '716928f7707fc662f62e59a988c0c264123be301abac4c23fe835bc09c052b3a',
 'automation/english_ipa_qwen17_final_ipa_probe_once.py': '15207db7182e4cb6d85349059b4aff91e721c70e8782fe901d4893f55224be76',
 'automation/english_kokoro_context32_mask_observe.py': '3f184bdeb7536f260b2d4e3ee77476d95590d05007ae251bcb93c51ce2be01af',
 'automation/english_parakeet_real_once.py': '366cd278f570a37ce68dd20e47e2bee1422daaa109c693ac42ab4bd4c2418881',
 'automation/english_qwen_asr_real_once.py': '14e8150ee76bf356fad0927e93d20981ccfaf32696beacf871810e26e8ff7990',
 'automation/english_qwen_asr_sourcefix_once.py': '9a000ed25440980333f11c21eec879b1945d87fa9f3d5e1794fc7090fdc96417',
 'automation/english_qwen_capture.py': 'be99d27dcff84107feca42d648afa863863cd5c9a3ffffaa505d287c29b2104f',
 'automation/english_qwen_real_once.py': '60e995b4cc39818b8795d8730f1eba81bc57a8bd7c778fc2df040b92f0bca95e',
 'automation/english_seven_voice_asr_once.py': '17771ca272204771982dbdc8e959f7fd75838b6598e84f09ac5819d8e3c79184',
 'automation/english_seven_voice_tts_once.py': 'e9dcf10dd94909edc13bf65ebcc31b2f3df6bdc81ab1876c94cbe28ec936ad0e',
 'automation/english_synthetic192_training_core.py': '0896d375f47318adbc42f63835cb85fc77441ae2472d9df8212dfca4dd7e62c0',
 'automation/english_whisper_real_once.py': '0f455c46061c6e3e4da2eec781234875fe2f59f257f11bacea8105e951593b6d',
 'automation/multilingual_campaign.py': '3cda7b79d1cf4e00f7b2a4aeff6c254f8b49d2adcf527a5ca51093ef60d1ef4f',
 'kws_data_contract/__init__.py': '3b3c0ef178e7115b2206e895c19128603272c0d3ad8a22eade280d372f884608',
 'kws_data_contract/identity.py': '7cb39db2c8fe07e6fada96735bae526c2f442775fc8e4786940add80abbba3e8',
 'research/auto-tts-voice-clone-20261102/campaign.py': '55f07e3604ac72a6ff912f25dc86f706678c6bb7661714b3dec3c1646e8d766f',
 'research/auto-tts-voice-clone-20261102/runner.py': 'df3e8858e25f041ecbc1ef7f10e7b7715722ad7ad8a2f5e53f120c172d3d40c1'}
TEXT_MODELS = {
    "qwen": {"recognizer_id": "qwen", "model_family": "qwen3-asr",
        "model_id": "Qwen/Qwen3-ASR-0.6B", "model_revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "model_sha256": "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea",
        "decode_config": {"language": "English", "context": "", "max_new_tokens": 128,
            "batch_size": 1, "dtype": "float32", "device": "cpu", "attention": "sdpa"}},
    "whisper": {"recognizer_id": "whisper", "model_family": "whisper",
        "model_id": "openai/whisper-small", "model_revision": "31243bad24cc746f07d4c8bfdd2d974872cb1803",
        "model_sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
        "decode_config": {"language": "en", "fp16": False, "initial_prompt": None,
            "condition_on_previous_text": False, "temperature": 0.0, "device": "cpu"}},
    "parakeet": {"recognizer_id": "parakeet", "model_family": "parakeet-tdt",
        "model_id": "nvidia/parakeet-tdt-0.6b-v3", "model_revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "model_sha256": "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e",
        "decode_config": {"device": "cpu", "prompt": None, "threads": 4, "timeout_seconds": 180}},
}
TEXT_RUNTIME = {
    "qwen": {"torch": "2.9.1+cpu", "torchaudio": "2.9.1+cpu", "qwen-asr": "0.0.6", "huggingface-hub": "0.36.2"},
    "whisper": {"torch": "2.9.1+cpu", "openai-whisper": "20250625"},
    "parakeet": {"huggingface-hub": "0.36.2", "hf-xet": "1.2.0"},
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path, limit=64 * 1024**2):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and 0 < path.stat().st_size <= limit, "REGULAR_BOUNDED_FILE: " + str(path))
    return path.read_bytes()


def read(path):
    return json.loads(regular(path))


def put(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return {"path": Path(path).name, "bytes": len(raw), "sha256": sha(raw)}


def write(path, value):
    return put(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n").encode())


def native_module():
    import english_kokoro_context32_mask_observe as native
    return native


def identity(provider):
    need(provider in PROVIDERS, "FIXED_FOUR_PROVIDERS")
    return native_module().OBSERVERS[provider] if provider == "phoneme" else TEXT_MODELS[provider]


def source_check(plan=None):
    measured = {p: sha(regular(ROOT / p)) for p in HELPER_PINS}
    need(len(measured) == 19 and measured == HELPER_PINS, "FROZEN_HELPER_BYTES")
    if plan is not None:
        need(plan.get("helper_sha256") == HELPER_PINS, "EXACT_HELPER_PINS")
        need(plan.get("observer_code_sha256") == sha(regular(Path(__file__))), "OBSERVER_SOURCE_PIN")
        need(plan.get("training_allowed") is False and plan.get("retry_count") == 0,
             "DIAGNOSTIC_NO_RETRY")
        need(plan.get("maximum_calls_per_observer") == 12
             and plan.get("maximum_total_model_calls") == 60, "FIXED_MODEL_CALL_BUDGET")
    return {"source_check": "PASS", "helper_sha256": measured, "providers": list(PROVIDERS),
            "planned_calls_per_provider": 12, "new_model_calls": 0, "new_network_calls": 0}


def public_guard(plan, provider, run_identity):
    source_check(plan)
    need(provider in PROVIDERS and plan.get("enabled") is True, "ENABLED_FIXED_PROVIDER")
    actual = {"repository": os.environ.get("GITHUB_REPOSITORY"), "commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "event": os.environ.get("GITHUB_EVENT_NAME")}
    need(actual == run_identity and actual["repository"] == "jiying2007/kws-data"
         and actual["event"] == "push" and actual["run_attempt"] == "1"
         and os.environ.get("GITHUB_REF") == "refs/heads/main"
         and re.fullmatch(r"[0-9a-f]{40}", actual["commit"] or "")
         and re.fullmatch(r"[1-9][0-9]*", actual["run_id"] or ""), "FIRST_PUBLIC_MAIN_PUSH")
    event = read(os.environ["GITHUB_EVENT_PATH"])
    need(re.fullmatch(r"[0-9a-f]{40}", plan.get("main_before") or "")
         and event.get("before") == plan["main_before"] and event.get("after") == actual["commit"]
         and event["before"] != event["after"] and event.get("repository", {}).get("private") is False
         and event.get("repository", {}).get("full_name") == actual["repository"], "EXACT_PUBLIC_SUCCESSOR")


def _load(provider, assets, output):
    if provider == "phoneme":
        return native_module()._load_native(provider, assets, output)
    versions = {name: importlib.metadata.version(name) for name in TEXT_RUNTIME[provider]}
    need(versions == TEXT_RUNTIME[provider], "FIXED_ASR_RUNTIME")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import english_seven_voice_asr_once as legacy
    assets.mkdir(mode=0o700)
    decode = legacy._load(provider, assets)
    return decode, {"observer": identity(provider), "runtime_versions": versions,
                    "adapter_sha256": HELPER_PINS["automation/english_seven_voice_asr_once.py"],
                    "initialization_model_calls": 0, "target_conditioned": False}


def initial_rows(rows):
    return [{"id": r["id"], "status": "not_started", "input_started": False,
             "forward_attempts": 0, "forward_returned": False, "capture_saved": False} for r in rows]


def base_record(provider, row, run_identity, source_sha):
    return {"schema": SCHEMA, "provider": provider, "id": row["id"], "run": run_identity,
        "source_sha256": source_sha, "input_pcm_sha256": row["pcm_sha256"],
        "input_wav_sha256": row["wav_sha256"], "input_samples": row["pcm_samples"],
        "observer": identity(provider), "observer_code_sha256": sha(regular(Path(__file__))),
        "input_scope": "ENTIRE_UNCROPPED_SHORT_PCM", "retry_count": 0, **FLAGS}


def _execute_rows(provider, source_root, rows, output, run_identity, source_sha, backend, ledger):
    """Private model-free test seam; public run always selects the frozen backend."""
    import english_pocket12_generate as generate
    for row, progress in zip(rows, ledger):
        directory = output / "rows" / row["id"]
        directory.mkdir(parents=True)
        base = base_record(provider, row, run_identity, source_sha)
        try:
            write(directory / "STARTED.json", base)
            progress["input_started"] = True
            wav = generate.within(source_root, row["wav_path"])
            pcm = regular(generate.within(source_root, row["pcm_path"]))
            need(sha(pcm) == row["pcm_sha256"] and sha(regular(wav)) == row["wav_sha256"], "UNCHANGED_INPUT_BYTES")
            if provider == "phoneme":
                geometry = native_module().frame_geometry(row["pcm_samples"], 0)
                prepare, forward, materialize = backend
                # No intent, text or class is passed to the processor or model.
                geometry_row = {"pcm_samples": row["pcm_samples"], "full_to_crop_offset_16k": 0}
                batch, facts, normalized, mask = prepare(pcm, geometry_row)
                facts = facts | {"normalization_scope": "ENTIRE_UNCROPPED_SHORT_PCM"}
                processor = {"processor": facts, "normalized_input": put(directory / "normalized-input.f32", normalized),
                             "attention_mask": put(directory / "attention-mask.i32", mask)}
                write(directory / "INPUT.json", base | processor)
            write(directory / "FORWARD_STARTED.json", base)
            progress["forward_attempts"] = 1
            result = forward(batch) if provider == "phoneme" else backend(wav, directory)
            progress["forward_returned"] = True
            write(directory / "FORWARD_RETURNED.json", base)
            if provider == "phoneme":
                values, logits = materialize(result, geometry_row)
                captured = processor | {"logits": put(directory / "logits.f32", logits),
                    "values": values, "frame_geometry": geometry}
            else:
                need(type(result) is str and len(result) <= 16384, "RAW_TRANSCRIPT_STRING")
                captured = {"raw_transcript": result}
            pin = write(directory / "CAPTURE.json", base | captured | {"status": "success"})
            progress.update(status="success", capture_saved=True, capture_sha256=pin["sha256"])
        except BaseException as exc:
            progress.update(status="failed", error=type(exc).__name__ + ": " + str(exc)[:2000])
            write(directory / "FAILED.json", base | progress)
            if not isinstance(exc, Exception):
                raise


def file_records(output):
    records = []
    for path in sorted(output.rglob("*")):
        need(not path.is_symlink(), "NO_ARTIFACT_SYMLINK")
        if path.is_file() and path != output / "FILES.json":
            raw = regular(path)
            records.append({"path": path.relative_to(output).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    return records


def _finish(output, provider, rows, run_identity, source_sha, failure):
    attempts, returned, completed = (sum(r[k] for r in rows) for k in
                                    ("forward_attempts", "forward_returned", "capture_saved"))
    result = {"schema": SCHEMA, "provider": provider, "run": run_identity, "source_sha256": source_sha,
        "status": "COMPLETE" if completed == 12 and failure is None else "INCOMPLETE",
        "denominator": 12, "actual_forward_attempts": attempts, "native_forward_returns": returned,
        "completed_count": completed, "failed_count": sum(r["status"] == "failed" for r in rows),
        "not_started_count": sum(r["status"] == "not_started" for r in rows),
        "retry_count": 0, "initialization_error": failure, "rows": rows, **FLAGS}
    need(len(rows) == 12 and 0 <= completed <= returned <= attempts <= 12, "COUNTER_INVARIANTS")
    write(output / "RESULT.json", result)
    write(output / "FILES.json", {"schema": SCHEMA + "/files", "files": file_records(output),
                                  "self_excluded": "FILES.json"})
    return result


def run(*, provider, source_root, output_root, asset_root, plan, run_identity):
    public_guard(plan, provider, run_identity)
    import english_pocket12_generate as generate
    source, rows = generate.validate_source(source_root, run_identity)
    need(len(rows) == len(source["rows"]) == 12, "ALL_TWELVE_SOURCE_CAPTURES_REQUIRED")
    output, assets, source_root = (Path(p).absolute() for p in (output_root, asset_root, source_root))
    for path in (output, assets):
        need(not path.exists() and path.parent.is_dir()
             and not any(p.is_symlink() for p in (path, *path.parents)), "FRESH_PRIVATE_DIRECTORY")
    need(all(assets != p and assets not in p.parents and p not in assets.parents
             for p in (output, source_root)), "PRIVATE_ASSETS_SEPARATE_FROM_ARTIFACTS")
    need(output != source_root and output not in source_root.parents and source_root not in output.parents,
         "SOURCE_AND_OUTPUT_SEPARATE")
    output.mkdir(mode=0o700)
    source_sha, ledger = sha(regular(source_root / "SOURCE.json")), initial_rows(rows)
    write(output / "STARTED.json", {"schema": SCHEMA, "provider": provider, "run": run_identity,
        "source_sha256": source_sha, "denominator": 12, "maximum_calls": 12, **FLAGS})
    failure = None
    def expired(signum, frame):
        raise TimeoutError("MODEL_INITIALIZATION_DEADLINE_840S")
    previous = signal.signal(signal.SIGALRM, expired)
    try:
        write(output / "MODEL_LOAD_STARTED.json", {"provider": provider, "actual_forward_attempts": 0})
        signal.alarm(840)
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


def _verify_phone(directory, capture, row, pcm, vocab):
    native = native_module()
    geometry = native.frame_geometry(row["pcm_samples"], 0)
    need(capture["frame_geometry"] == geometry, "COMPLETE_SHORT_FRAME_GEOMETRY")
    arrays = {}
    for key, filename in (("normalized_input", "normalized-input.f32"),
                          ("attention_mask", "attention-mask.i32"), ("logits", "logits.f32")):
        raw = regular(directory / filename)
        need(capture[key] == {"path": filename, "bytes": len(raw), "sha256": sha(raw)}, "SAVED_ARRAY_PIN")
        arrays[key] = raw
    n, facts = row["pcm_samples"], capture["processor"]
    need(len(arrays["normalized_input"]) == 4 * n and arrays["attention_mask"] == struct.pack("<i", 1) * n,
         "COMPLETE_FLOAT32_INPUT_AND_NATIVE_INT32_MASK")
    expected = {"sample_rate": 16000, "pcm_samples": n, "normalized_shape": [1, n],
        "normalized_dtype": "torch.float32", "attention_mask_shape": [1, n], "attention_mask_dtype": "torch.int32",
        "attention_mask_sum": n, "attention_mask_unique_values": [1], "valid_input_samples": n,
        "padded_input_samples": n, "right_padding_samples": 0, "normalization_scope": "ENTIRE_UNCROPPED_SHORT_PCM"}
    need(all(facts.get(k) == v for k, v in expected.items()), "NATIVE_PROCESSOR_FACTS")
    x = [v[0] / 32768.0 for v in struct.iter_unpack("<h", pcm)]
    mean = math.fsum(x) / n
    scale = math.sqrt(math.fsum((v - mean)**2 for v in x) / n + 1e-7)
    normalized = [v[0] for v in struct.iter_unpack("<f", arrays["normalized_input"])]
    need(all(math.isfinite(v) for v in normalized)
         and max(abs(a - (b - mean) / scale) for a, b in zip(normalized, x)) <= 4e-5,
         "NORMALIZED_INPUT_RECOMPUTED_FROM_SAME_PCM")
    need(len(arrays["logits"]) == geometry["frames"] * 392 * 4, "ALL_LOGIT_FRAMES_PRESENT")
    ids = []
    for frame in struct.iter_unpack("<392f", arrays["logits"]):
        need(all(math.isfinite(v) for v in frame), "FINITE_SAVED_LOGITS")
        ids.append(max(range(392), key=frame.__getitem__))
    values = capture["values"]
    ctc, _ = native.ctc_function(regular(HERE / "english_final_ipa_xlsr_once.py"))
    tokens = ctc(ids, vocab, {0, 1, 2, 3})
    need(values == {"shape": [1, geometry["frames"], 392], "dtype": "float32-little-endian", "order": "C",
        "frame_ids": ids, "phone_tokens": tokens,
        "argmax_special_frame_counts": {str(k): ids.count(k) for k in (0, 1, 2, 3)}}, "SAVED_LOGITS_ARGMAX_AND_CTC")
    return tokens


def readback(*, provider, source_root, observer_root, plan, run_identity):
    source_check(plan)
    need(provider in PROVIDERS, "FIXED_PROVIDER")
    import english_pocket12_generate as generate
    source, successful = generate.validate_source(source_root, run_identity)
    need(len(successful) == 12, "SOURCE_COMPLETE_FOR_OBSERVATION_JOIN")
    source_sha = sha(regular(Path(source_root) / "SOURCE.json"))
    root = None if observer_root is None else Path(observer_root)
    terminal = bool(root and (root / "RESULT.json").is_file() and (root / "FILES.json").is_file())
    result = None
    if root is not None and root.exists():
        need(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), "REGULAR_OBSERVER_ROOT")
        allowed = {"STARTED.json", "MODEL_LOAD_STARTED.json", "MODEL_LOCK.json", "RESULT.json", "FILES.json", "rows", "UNOBSERVED.json"}
        if provider == "phoneme":
            allowed.add("model-metadata")
        need({p.name for p in root.iterdir()} <= allowed, "NO_UNDECLARED_OBSERVER_FILES")
        if (root / "UNOBSERVED.json").exists():
            receipt = read(root / "UNOBSERVED.json")
            need({p.name for p in root.iterdir()} == {"UNOBSERVED.json"}
                 and receipt.get("command_step_outcome") in ("success", "failure", "skipped", "cancelled")
                 and receipt == {"stage": provider, "run": run_identity,
                    "command_step_outcome": receipt["command_step_outcome"], "has_terminal_result": False,
                    "diagnostic_only": True, "training_allowed": False, "retry_count": 0,
                    "missing_terminal_is_zero": False, "actual_model_calls": None}, "UNOBSERVED_IS_UNKNOWN_ONLY")
        if (root / "rows").exists():
            need({p.name for p in (root / "rows").iterdir()} <= {r["id"] for r in source["rows"]},
                 "NO_UNDECLARED_INPUT_ROWS")
        if (root / "STARTED.json").is_file():
            need(read(root / "STARTED.json") == {"schema": SCHEMA, "provider": provider, "run": run_identity,
                "source_sha256": source_sha, "denominator": 12, "maximum_calls": 12, **FLAGS}, "OBSERVER_START_BINDING")
    if terminal:
        manifest, result = read(root / "FILES.json"), read(root / "RESULT.json")
        need(manifest == {"schema": SCHEMA + "/files", "self_excluded": "FILES.json", "files": file_records(root)},
             "COMPLETE_OBSERVER_FILE_CLOSURE")
        need(all(result.get(k) == v for k, v in {"schema": SCHEMA, "provider": provider,
             "run": run_identity, "source_sha256": source_sha, "denominator": 12, "retry_count": 0, **FLAGS}.items()),
             "TERMINAL_OBSERVER_IDENTITY")
        need(type(result.get("rows")) is list and [r.get("id") for r in result["rows"]]
             == [r["id"] for r in source["rows"]], "TERMINAL_TWELVE_ROWS")
        need((root / "STARTED.json").is_file() and (root / "MODEL_LOAD_STARTED.json").is_file(),
             "TERMINAL_REQUIRES_START_RECEIPTS")
    vocab = None
    if root and (root / "MODEL_LOCK.json").is_file():
        lock = read(root / "MODEL_LOCK.json")
        if provider == "phoneme":
            native = native_module()
            need(lock.get("model") == native.OBSERVERS[provider] and lock.get("files") == native.OBSERVER_FILES[provider]
                 and lock.get("runtime_versions") == native.RUNTIME_VERSIONS
                 and lock.get("native_forward_attempts_at_initialization") == 0
                 and lock.get("model_parameters_modified") is False and lock.get("target_conditioned") is False,
                 "FROZEN_NATIVE_PHONE_LOCK")
            for name, pin in native.OBSERVER_FILES[provider].items():
                if name != "pytorch_model.bin":
                    raw = regular(root / "model-metadata" / name)
                    need({"bytes": len(raw), "sha256": sha(raw)} == pin, "FROZEN_PHONE_METADATA")
            vocab = read(root / "model-metadata" / "vocab.json")
        else:
            need(lock == {"observer": identity(provider), "runtime_versions": TEXT_RUNTIME[provider],
                "adapter_sha256": HELPER_PINS["automation/english_seven_voice_asr_once.py"],
                "initialization_model_calls": 0, "target_conditioned": False}, "FROZEN_BLIND_ASR_LOCK")
    observations, reconstructed = [], []
    for index, row in enumerate(source["rows"]):
        directory = None if root is None else root / "rows" / row["id"]
        exists = lambda name: bool(directory and (directory / name).is_file())
        if directory and directory.exists():
            allowed = {"STARTED.json", "FORWARD_STARTED.json", "FORWARD_RETURNED.json", "FAILED.json", "CAPTURE.json"}
            if provider == "phoneme":
                allowed |= {"INPUT.json", "normalized-input.f32", "attention-mask.i32", "logits.f32"}
            need({p.name for p in directory.iterdir()} <= allowed, "NO_EXTRA_ROW_CALL_OR_CAPTURE")
        progress = {"id": row["id"], "status": "unknown", "input_started": exists("STARTED.json"),
            "forward_attempts": int(exists("FORWARD_STARTED.json")), "forward_returned": exists("FORWARD_RETURNED.json"),
            "capture_saved": exists("CAPTURE.json")}
        base = base_record(provider, row, run_identity, source_sha)
        for filename in ("STARTED.json", "FORWARD_STARTED.json", "FORWARD_RETURNED.json", "FAILED.json", "CAPTURE.json"):
            if exists(filename):
                stored = read(directory / filename)
                need(all(stored.get(k) == v for k, v in base.items()), "ROW_CAPTURE_INPUT_BINDING")
        if exists("FAILED.json"):
            failure = read(directory / "FAILED.json")
            need(type(failure.get("forward_returned")) is bool
                 and type(failure.get("forward_attempts")) is int
                 and failure["forward_attempts"] == progress["forward_attempts"]
                 and failure.get("capture_saved") is False, "FAILED_ROW_CALL_LEDGER")
            # A native return remains a return if writing its separate receipt failed.
            progress["forward_returned"] = progress["forward_returned"] or failure["forward_returned"]
        need(not progress["forward_attempts"] or progress["input_started"], "START_BEFORE_CALL")
        need(not progress["forward_returned"] or progress["forward_attempts"] == 1, "CALL_BEFORE_RETURN")
        observation = {"id": row["id"], "status": "missing", "raw_transcript": None, "phone_tokens": None}
        if progress["capture_saved"]:
            need(progress["forward_returned"] and root is not None and (root / "MODEL_LOCK.json").is_file()
                 and not exists("FAILED.json"), "CAPTURE_AFTER_NATIVE_RETURN")
            capture = read(directory / "CAPTURE.json")
            need(capture.get("status") == "success", "CAPTURE_SUCCESS")
            if provider == "phoneme":
                need(vocab is not None, "PINNED_NATIVE_VOCAB")
                pcm = regular(generate.within(source_root, row["pcm_path"]))
                observation["phone_tokens"] = _verify_phone(directory, capture, row, pcm, vocab)
                saved_input = read(directory / "INPUT.json")
                need(all(saved_input.get(k) == v for k, v in (base | {k: capture[k] for k in
                     ("processor", "normalized_input", "attention_mask")}).items()), "SAVED_PRE_FORWARD_INPUT")
            else:
                need(type(capture.get("raw_transcript")) is str and len(capture["raw_transcript"]) <= 16384,
                     "COMPLETE_RAW_ASR_TRANSCRIPT")
                observation["raw_transcript"] = capture["raw_transcript"]
            progress.update(status="success", capture_sha256=sha(regular(directory / "CAPTURE.json")))
            observation.update(status="success", capture_sha256=progress["capture_sha256"])
        elif exists("FAILED.json"):
            progress["status"] = "failed"
        elif terminal:
            need(not progress["input_started"], "TERMINAL_STARTED_ROW_NEEDS_FAILURE_OR_CAPTURE")
            progress["status"] = "not_started"
        if terminal:
            claimed = result["rows"][index]
            need(all(type(claimed.get(k)) is type(v) and claimed.get(k) == v for k, v in progress.items()),
                 "TERMINAL_ROW_LEDGER_RECOMPUTED")
        observation.update(status=progress["status"], input_started=progress["input_started"],
            forward_attempts=progress["forward_attempts"] if terminal else None,
            forward_attempts_lower_bound=progress["forward_attempts"],
            native_return_observed=progress["forward_returned"],
            failure=read(directory / "FAILED.json").get("error") if exists("FAILED.json") else None)
        reconstructed.append(progress)
        observations.append(observation)
    attempts, returned, completed = (sum(r[k] for r in reconstructed) for k in
                                    ("forward_attempts", "forward_returned", "capture_saved"))
    if terminal:
        counts = {"actual_forward_attempts": attempts, "native_forward_returns": returned,
            "completed_count": completed, "failed_count": sum(r["status"] == "failed" for r in reconstructed),
            "not_started_count": sum(r["status"] == "not_started" for r in reconstructed)}
        need(all(type(result.get(k)) is int and result[k] == v for k, v in counts.items()), "ACTUAL_CALL_COUNTERS_RECOMPUTED")
        need(result["status"] == ("COMPLETE" if completed == 12 and result["initialization_error"] is None else "INCOMPLETE"),
             "OBSERVER_COMPLETENESS_RECOMPUTED")
    summary = {"provider": provider, "denominator": 12, "terminal_receipt_present": terminal,
        "actual_forward_attempts": attempts if terminal else None,
        "attempts_lower_bound": attempts, "attempts_upper_bound": attempts if terminal else 12,
        "native_forward_returns": returned if terminal else None, "completed_count": completed,
        "failed_count": sum(r["status"] == "failed" for r in reconstructed),
        "not_started_count": sum(r["status"] == "not_started" for r in reconstructed) if terminal else None,
        "unknown_count": sum(r["status"] == "unknown" for r in reconstructed),
        "status": result["status"] if terminal else "INCOMPLETE_UNKNOWN_COUNTERS", **FLAGS}
    return {"summary": summary, "rows": observations, "new_model_calls": 0}


if __name__ == "__main__":
    need(sys.argv[1:] == ["--source-check"], "NO_DIRECT_MODEL_CLI")
    print(json.dumps(source_check(), ensure_ascii=False, sort_keys=True))
