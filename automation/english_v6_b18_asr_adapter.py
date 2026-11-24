"""Future B18 raw-first ASR capture; importing this module executes no model.

Only explicit Unicode White_Space is normalized. Original returned text and
Parakeet stdout/stderr are persisted before payload validation, including failure.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unicodedata
import wave

HERE = Path(__file__).resolve().parent
TEXT_PROVIDERS = ("qwen", "whisper", "parakeet")
RAW_SCHEMA = "english-v6-b18-native-raw-return-v1"
WHITE_SPACE = frozenset((*range(0x0009, 0x000E), 0x0020, 0x0085, 0x00A0,
                        0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029,
                        0x202F, 0x205F, 0x3000))
PARSER_CONTRACT = {
    "schema": "english-v6-b18-prospective-asr-parser-v1",
    "raw_encoding": "UTF-8 strict",
    "maximum_raw_decoded_codepoints_before_normalization": 4096,
    "white_space_codepoints": [f"U+{n:04X}" for n in sorted(WHITE_SPACE)],
    "normalization": "Map only listed White_Space to U+0020; collapse U+0020 runs and trim U+0020 edges.",
    "reject_other_unicode_categories": ["Cc", "Cf", "Cs"],
    "require_nonempty_normalized_text": True,
    "lexical_edits": False, "punctuation_edits": False,
    "case_edits": False, "unicode_compatibility_normalization": False,
    "line_extraction": False, "log_removal": False, "target_hints": False,
    "raw_saved_before_validation": True,
    "invalid_empty_missing_return_policy": "FAILED_AND_WHOLE_PANEL_STOP_NO_RETRY",
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


PARSER_CONTRACT_SHA256 = sha(canonical(PARSER_CONTRACT))


def parser_contract():
    return json.loads(canonical(PARSER_CONTRACT))


def pin(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         "REGULAR_ORIGINAL_EVIDENCE_FILE")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"path": path.name, "bytes": size, "sha256": digest.hexdigest()}


def put(path, raw):
    path = Path(path)
    need(type(raw) is bytes and path.parent.is_dir()
         and not any(p.is_symlink() for p in (path, *path.parents)), "CREATE_ONLY_RAW_FILE")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(path)


def record(path, value):
    # ensure_ascii retains even an invalid isolated surrogate without replacing it.
    return put(path, json.dumps(value, ensure_ascii=True, sort_keys=True,
                               indent=2, allow_nan=False).encode("ascii") + b"\n")


def normalize_raw(raw):
    need(type(raw) is bytes, "RAW_UTF8_BYTES_REQUIRED")
    text = raw.decode("utf-8", errors="strict")
    need(len(text) <= 4096, "RAW_DECODED_TEXT_OVERSIZE_BEFORE_NORMALIZATION")
    need(all(ord(ch) in WHITE_SPACE or unicodedata.category(ch) not in ("Cc", "Cf", "Cs")
             for ch in text), "NON_WHITE_SPACE_CONTROL_CHARACTER")
    folded = "".join(" " if ord(ch) in WHITE_SPACE else ch for ch in text)
    normalized = " ".join(part for part in folded.split(" ") if part)
    need(bool(normalized), "EMPTY_NATIVE_TRANSCRIPT")
    return normalized


def _payload_valid(directory, binding, raw_pin, normalized):
    value = {**binding, "schema": RAW_SCHEMA, "payload_valid": True,
             "parser_contract_sha256": PARSER_CONTRACT_SHA256,
             "original_payload": raw_pin, "normalized_transcript": normalized,
             "normalized_utf8_sha256": sha(normalized.encode("utf-8")),
             "whitespace_only_normalization": True}
    validated = record(directory / "PAYLOAD_VALID.json", value)
    return {"raw_transcript": normalized,
            "transcript_semantics": "explicit-White_Space-normalized; original raw payload retained",
            "raw_evidence": {"payload": raw_pin, "validation": validated,
                             "parser_contract_sha256": PARSER_CONTRACT_SHA256}}


def _text_field(value):
    if type(value) in (str, int, bool, type(None)):
        return {"python_type": type(value).__name__, "value": value}
    if type(value) is bytes:
        return {"python_type": "bytes", "hex": value.hex()}
    # Non-text native structures are never admitted. This is diagnostic only.
    try:
        representation = repr(value)
    except Exception as exc:
        representation = "<repr failed: " + type(exc).__name__ + ">"
    return {"python_type": type(value).__name__, "repr": representation,
            "lossless_text_payload_available": False}


def _python_envelope(provider, native):
    envelope = {"provider": provider, "container_type": type(native).__name__}
    selected = None
    shape_valid = False
    if provider == "qwen" and type(native) is list:
        fields = []
        for item in native:
            try:
                value = item.text
                fields.append(_text_field(value))
            except Exception as exc:
                fields.append({"python_type": type(item).__name__, "text_read_error": type(exc).__name__})
        envelope["returned_text_fields"] = fields
        shape_valid = len(fields) == 1 and fields[0].get("python_type") == "str"
        if shape_valid:
            selected = fields[0]["value"]
    elif provider == "whisper" and type(native) is dict:
        envelope["text_field_present"] = "text" in native
        envelope["returned_text_field"] = _text_field(native.get("text"))
        shape_valid = "text" in native and type(native["text"]) is str
        if shape_valid:
            selected = native["text"]
    else:
        envelope["unexpected_native_value"] = _text_field(native)
    envelope["native_shape_valid"] = shape_valid
    return envelope, selected


def capture_python(provider, directory, binding, invoke):
    """Model-free injection seam; production invoke is the exact native transcribe call."""
    need(provider in ("qwen", "whisper"), "PYTHON_ASR_PROVIDER")
    directory = Path(directory)
    record(directory / "FORWARD_STARTED.json", {**binding, "attempt": 1})
    native = invoke()
    # Returning and saving a valid payload are deliberately separate facts.
    record(directory / "FORWARD_RETURNED.json", {**binding, "native_return_observed": True})
    envelope, text = _python_envelope(provider, native)
    raw_pin = record(directory / "RAW_RETURN.json", {**binding, "schema": RAW_SCHEMA, **envelope})
    need(envelope["native_shape_valid"], "NATIVE_TEXT_RESULT_SHAPE")
    raw = text.encode("utf-8", errors="strict")
    utf8_pin = put(directory / "RAW_TEXT.utf8", raw)
    normalized = normalize_raw(raw)
    result = _payload_valid(directory, binding, raw_pin, normalized)
    result["raw_evidence"]["utf8_text"] = utf8_pin
    return result


def capture_process(command, environment, directory, binding, *, popen=None):
    """Preserve native stdout/stderr byte-for-byte, including failed or killed calls."""
    directory = Path(directory)
    command_receipt = {**binding, "schema": RAW_SCHEMA, "command": list(command),
                       "command_sha256": sha(canonical(list(command))),
                       "cwd": str(directory), "environment": environment,
                       "timeout_seconds": 180, "stdout_transform": None, "stderr_transform": None}
    command_pin = record(directory / "NATIVE_COMMAND.json", command_receipt)
    process, code, failure, timed_out = None, None, None, False
    factory = subprocess.Popen if popen is None else popen
    with (directory / "RAW_STDOUT.bin").open("xb") as stdout, (directory / "RAW_STDERR.bin").open("xb") as stderr:
        try:
            record(directory / "FORWARD_STARTED.json", {**binding, "attempt": 1,
                                                       "native_command_sha256": command_pin["sha256"]})
            process = factory(command, cwd=str(directory), env=environment,
                              stdout=stdout, stderr=stderr)
            try:
                code = process.wait(timeout=180)
            except subprocess.TimeoutExpired as exc:
                timed_out, failure = True, exc
                process.kill()
                code = process.wait()
        except BaseException as exc:
            failure = exc
            if process is not None and code is None:
                if process.poll() is None:
                    process.kill()
                code = process.wait()
        finally:
            stdout.flush(); stderr.flush()
            os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    if failure is None:
        record(directory / "FORWARD_RETURNED.json", {**binding, "native_return_observed": True,
                                                     "native_exit_code": code})
    raw_pin = record(directory / "RAW_PROCESS.json", {**binding, "schema": RAW_SCHEMA,
        "command": command_pin, "stdout": pin(directory / "RAW_STDOUT.bin"),
        "stderr": pin(directory / "RAW_STDERR.bin"), "process_started": process is not None,
        "native_exit_code": code, "native_return_observed": failure is None,
        "timed_out": timed_out, "exception_type": None if failure is None else type(failure).__name__})
    if failure is not None:
        raise failure
    need(code == 0, "PARAKEET_NATIVE_PROCESS_FAILED")
    normalized = normalize_raw((directory / "RAW_STDOUT.bin").read_bytes())
    return _payload_valid(directory, binding, raw_pin, normalized)


def _load(provider, assets, output):
    """Frozen model assets/decode, with new raw-preserving capture wrappers only."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    if str(HERE.parent) not in sys.path:
        sys.path.insert(0, str(HERE.parent))
    import english_pocket12_observe as old
    old.source_check()
    need(provider in TEXT_PROVIDERS, "THREE_FROZEN_ASR_PROVIDERS")
    versions = {name: importlib.metadata.version(name) for name in old.TEXT_RUNTIME[provider]}
    need(versions == old.TEXT_RUNTIME[provider], "FIXED_ASR_RUNTIME")
    assets.mkdir(mode=0o700)
    lock = {"observer": old.identity(provider), "runtime_versions": versions,
            "adapter_sha256": sha(Path(__file__).read_bytes()),
            "original_adapter_sha256": old.HELPER_PINS["automation/english_seven_voice_asr_once.py"],
            "parser_contract_sha256": PARSER_CONTRACT_SHA256,
            "initialization_model_calls": 0, "target_conditioned": False,
            "raw_before_validation": True, "decode_config_unchanged": True}
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    if provider == "qwen":
        import english_qwen_asr_real_once as qwen
        folder = qwen.lock_model(assets)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from qwen_asr import Qwen3ASRModel
        torch.set_num_threads(min(os.cpu_count() or 1, 4))
        engine = Qwen3ASRModel.from_pretrained(str(folder), dtype=torch.float32, device_map="cpu",
            max_inference_batch_size=1, max_new_tokens=128, attn_implementation="sdpa")
        def decode(wav, directory, binding):
            return capture_python(provider, directory, binding,
                lambda: engine.transcribe(audio=str(wav), language="English", context=""))
        return decode, lock
    if provider == "whisper":
        import english_whisper_real_once as whisper_old
        weights = whisper_old.pinned_model(assets)
        import torch
        import numpy as np
        import whisper
        torch.set_num_threads(min(os.cpu_count() or 1, 4))
        engine = whisper.load_model(str(weights), device="cpu")
        def decode(wav, directory, binding):
            with wave.open(str(wav), "rb") as stream:
                need(stream.getnchannels() == 1 and stream.getsampwidth() == 2
                     and stream.getframerate() == 16000 and stream.getcomptype() == "NONE",
                     "WHISPER_NATIVE_WAV_GEOMETRY")
                raw = stream.readframes(stream.getnframes())
                need(not stream.readframes(1), "WHISPER_EXTRA_PCM")
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            return capture_python(provider, directory, binding,
                lambda: engine.transcribe(pcm, language="en", fp16=False, verbose=False,
                    initial_prompt=None, condition_on_previous_text=False, temperature=0.0))
        return decode, lock
    import english_parakeet_real_once as parakeet
    binary, runtime = parakeet.verify_runtime(assets)
    model, model_pin = parakeet.verify_gguf(assets)
    lock.update(native_runtime_release=runtime, native_binary=pin(binary), native_model=model_pin)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    def decode(wav, directory, binding):
        environment = {"PATH": "/usr/bin:/bin", "HOME": str(directory), "LANG": "C.UTF-8",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "NEMO_SPEECH_MODEL_DIR": str(assets), "OMP_NUM_THREADS": "4"}
        return capture_process([str(binary), "transcribe", str(wav), "--model", str(model),
                                "--device", "cpu"], environment, directory, binding)
    return decode, lock


def _read_pinned(directory, metadata):
    need(type(metadata) is dict and set(metadata) == {"path", "bytes", "sha256"}
         and Path(metadata["path"]).name == metadata["path"], "RAW_EVIDENCE_RELATIVE_FILE")
    path = directory / metadata["path"]
    need(pin(path) == metadata, "RAW_EVIDENCE_BYTES_AND_SHA")
    return path.read_bytes()


def validate_capture(provider, directory, binding, capture):
    """Read saved raw output; never infer text from a failed/missing payload."""
    directory = Path(directory)
    started = json.loads((directory / "FORWARD_STARTED.json").read_bytes())
    returned = json.loads((directory / "FORWARD_RETURNED.json").read_bytes())
    need(all(started.get(k) == returned.get(k) == v for k, v in binding.items())
         and started["attempt"] == 1 and returned["native_return_observed"] is True,
         "ACTUAL_STARTED_AND_RETURNED_BOUND_TO_ORIGINAL")
    evidence = capture["raw_evidence"]
    need(evidence["parser_contract_sha256"] == PARSER_CONTRACT_SHA256, "FROZEN_ASR_PARSER")
    payload = json.loads(_read_pinned(directory, evidence["payload"]))
    need(all(payload.get(k) == v for k, v in binding.items())
         and payload["schema"] == RAW_SCHEMA, "RAW_RETURN_SOURCE_BINDING")
    if provider == "parakeet":
        need(evidence["payload"]["path"] == "RAW_PROCESS.json"
             and payload["process_started"] is True and payload["native_exit_code"] == 0
             and payload["native_return_observed"] is True and payload["timed_out"] is False
             and payload["exception_type"] is None and returned["native_exit_code"] == 0,
             "COMPLETE_SUCCESSFUL_NATIVE_PROCESS_RETURN")
        command_raw = _read_pinned(directory, payload["command"])
        command = json.loads(command_raw)
        need(started["native_command_sha256"] == sha(command_raw)
             and all(command.get(k) == v for k, v in binding.items())
             and command["schema"] == RAW_SCHEMA, "NATIVE_COMMAND_SOURCE_BINDING")
        argv = command["command"]
        need(len(argv) == 7 and Path(argv[0]).name == "nemo-speech"
             and argv[1] == "transcribe" and Path(argv[2]).name == binding["id"] + ".wav"
             and argv[3] == "--model" and Path(argv[4]).name == "parakeet-tdt-0.6b-v3.q8_0.gguf"
             and argv[5:] == ["--device", "cpu"] and command["command_sha256"] == sha(canonical(argv))
             and command["timeout_seconds"] == 180 and command["stdout_transform"] is None
             and command["stderr_transform"] is None, "UNCHANGED_NATIVE_COMMAND_NO_TEXT_FILTER")
        environment = command["environment"]
        need(environment == {"PATH": "/usr/bin:/bin", "HOME": command["cwd"], "LANG": "C.UTF-8",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "NEMO_SPEECH_MODEL_DIR": str(Path(argv[4]).parent), "OMP_NUM_THREADS": "4"},
             "UNCHANGED_NATIVE_RUNTIME_ENVIRONMENT")
        _read_pinned(directory, payload["stderr"])
        raw = _read_pinned(directory, payload["stdout"])
    else:
        need(evidence["payload"]["path"] == "RAW_RETURN.json"
             and payload["native_shape_valid"] is True and payload["provider"] == provider,
             "COMPLETE_PYTHON_TEXT_RETURN")
        if provider == "qwen":
            need(payload["container_type"] == "list" and len(payload["returned_text_fields"]) == 1,
                 "QWEN_ORIGINAL_SINGLE_RESULT")
            field = payload["returned_text_fields"][0]
        else:
            need(provider == "whisper" and payload["container_type"] == "dict"
                 and payload["text_field_present"] is True, "WHISPER_ORIGINAL_TEXT_FIELD")
            field = payload["returned_text_field"]
        need(field["python_type"] == "str" and type(field["value"]) is str, "ORIGINAL_RETURNED_TEXT_TYPE")
        raw = _read_pinned(directory, evidence["utf8_text"])
        need(evidence["utf8_text"]["path"] == "RAW_TEXT.utf8"
             and raw == field["value"].encode("utf-8", errors="strict"), "LOSSLESS_ORIGINAL_RETURNED_TEXT")
    normalized = normalize_raw(raw)
    validated = json.loads(_read_pinned(directory, evidence["validation"]))
    need(evidence["validation"]["path"] == "PAYLOAD_VALID.json"
         and validated == {**binding, "schema": RAW_SCHEMA, "payload_valid": True,
             "parser_contract_sha256": PARSER_CONTRACT_SHA256, "original_payload": evidence["payload"],
             "normalized_transcript": normalized, "normalized_utf8_sha256": sha(normalized.encode("utf-8")),
             "whitespace_only_normalization": True}
         and capture["raw_transcript"] == normalized,
         "SAVED_RAW_PAYLOAD_REPARSES_EXACTLY")
    return normalized
