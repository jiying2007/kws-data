"""Read saved context32 bytes; never load, invoke, choose or relabel a model.

Native masks are saved as int32 without changing the model batch.
The prior run remains terminal with 32 preprocessing failures and zero forwards.
Full-input suffixes and a fixed convolution-coordinate region are descriptive
diagnostics. Neither is a whole-phrase admission result or acoustic ground truth.
The immutable old crop observations and old STOP decision remain unchanged.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import unicodedata
import zipfile

import english_kokoro_context32_source as source
import english_kokoro_context32_mask_observe as observe

SCHEMA = "english-kokoro-context32-maskfix-comparison-v1"
POLICY_SHA256 = "fde516d9f1f2457cdb0047e8d8aee6a667c68fa6ea21f83d22f135e4bc9069da"
PHONE_PATTERNS = {
    "mero-us-eh-v1": r"h?eɪmɛr(?:oʊ|ou|o)",
    "mero-us-i-v1": r"h?eɪmir(?:oʊ|ou|o)",
    "vosen-us-s-v1": r"h?eɪv(?:oʊ|ou|o)s[əɐʌ]n",
    "vosen-us-z-v1": r"h?eɪv(?:oʊ|ou|o)z[əɐʌ]n",
}
SKIP_IDS = frozenset((0, 1, 2, 3))
FOCUS_TOKENS = ("ɔ", "o", "oʊ", "aʊ", "ɛ", "iː", "ɹ", "v", "s", "z", "ə", "n")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def regular(path, limit=32 * 1024**2):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and 0 < path.stat().st_size <= limit, "REGULAR_BOUNDED_FILE")
    return path.read_bytes()


def within(root, name):
    need(type(name) is str and "\\" not in name and "\0" not in name, "SAFE_RELATIVE_PATH")
    relative = PurePosixPath(name)
    need(not relative.is_absolute() and name == relative.as_posix() and relative.parts
         and all(p not in ("", ".", "..") for p in name.split("/")), "SAFE_RELATIVE_PATH")
    path = Path(root).joinpath(*relative.parts)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "NO_SYMLINK")
    return path


def pinned(root, value, expected_name):
    need(type(value) is dict and value.get("path") == expected_name
         and type(value.get("bytes")) is int and type(value.get("sha256")) is str,
         "EXACT_CAPTURE_MEMBER_NAME")
    raw = regular(within(root, value["path"]))
    need(len(raw) == value["bytes"] and sha(raw) == value["sha256"], "CAPTURE_MEMBER_HASH")
    return raw


def vocabulary(raw):
    need(sha(raw) == observe.COMMON_FILES["vocab.json"]["sha256"], "FROZEN_VOCAB_BYTES")
    value = loads(raw)
    need(type(value) is dict and len(value) == 392
         and all(type(k) is str and k and type(v) is int for k, v in value.items())
         and set(value.values()) == set(range(392)), "EXACT392_VOCABULARY")
    return value


def argmax_ids(raw, frames, vocab_size=392):
    """Independent little-endian reconstruction, including first-index ties."""
    need(type(frames) is int and 1 <= frames <= 2000 and type(vocab_size) is int
         and vocab_size == 392 and type(raw) is bytes and len(raw) == frames * vocab_size * 4,
         "COMPLETE_LOGIT_SHAPE")
    output = []
    for frame in struct.iter_unpack("<" + "f" * vocab_size, raw):
        need(all(math.isfinite(x) for x in frame), "FINITE_ALL_LOGITS")
        output.append(max(range(vocab_size), key=frame.__getitem__))
    return output


def ctc_runs(ids, vocab, offset_frame=0):
    need(type(ids) is list and len(ids) <= 2000 and type(offset_frame) is int and offset_frame >= 0
         and all(type(i) is int and i in vocab.values() for i in ids), "VALID_FRAME_IDS")
    reverse = {number: token for token, number in vocab.items()}
    need(len(reverse) == len(vocab), "UNIQUE_VOCAB_IDS")
    result = []
    start = 0
    while start < len(ids):
        end = start + 1
        while end < len(ids) and ids[end] == ids[start]:
            end += 1
        token_id = ids[start]
        if token_id not in SKIP_IDS:
            first, stop = start + offset_frame, end + offset_frame
            result.append({"token_id": token_id, "token": reverse[token_id],
                "frame_start": first, "frame_stop_exclusive": stop,
                "nominal_conv_support_union_samples": [first * 320, (stop - 1) * 320 + 400]})
        start = end
    return result


def normalized_phones(tokens):
    need(type(tokens) is list and all(type(t) is str and t and len(t) <= 40
         and not any(ord(c) < 32 for c in t) for t in tokens), "PHONE_TOKENS_REQUIRED")
    value = unicodedata.normalize("NFKC", "".join(tokens))
    value = re.sub(r"[\sˈˌːˑ͜͡]", "", value)
    return value.replace("ɹ", "r").replace("ɾ", "r")


def phone_diagnostics(tokens):
    normalized = normalized_phones(tokens)
    whole, suffix = [], []
    for name, pattern in PHONE_PATTERNS.items():
        if re.fullmatch(pattern, normalized):
            whole.append(name)
        match = re.search(pattern + r"\Z", normalized)
        if match is not None:
            suffix.append({"hypothesis": name, "normalized_start": match.start(),
                "normalized_end": match.end(), "matched_suffix": match.group(0),
                "normalized_prefix": normalized[:match.start()],
                "hey_h_present": match.group(0).startswith("h")})
    # All terminal r/vowel tails are exposed, including rejected vowels. This is
    # not a corrective mapping: ɔ, o, oʊ and aʊ retain distinct spellings/IDs.
    tail = re.search(r"m([ɛi])r(.*)\Z", normalized)
    return {"phone_tokens": tokens, "normalized_phones": normalized,
        "whole_pattern_hypotheses": whole, "terminal_suffix_matches": suffix,
        "mero_terminal_core": None if tail is None else tail.group(1),
        "mero_terminal_after_r": None if tail is None else tail.group(2),
        "admission_decision": None, "acoustic_gold": False}


def close_number(actual, expected, label, tolerance=1e-9):
    need(type(actual) in (int, float) and math.isfinite(actual)
         and abs(actual - expected) <= tolerance * (1 + abs(expected)), label)


def processor_readback(directory, row, capture, pcm):
    """Check saved input statistics and its PCM relationship, without a model."""
    raw = pinned(directory, capture.get("normalized_input"), "normalized-input.f32")
    mask = pinned(directory, capture.get("attention_mask"), "attention-mask.i32")
    count = row["pcm_samples"]
    need(capture["attention_mask"].get("dtype") == "int32-little-endian"
         and capture["attention_mask"].get("order") == "C"
         and capture["attention_mask"].get("shape") == [1, count], "NATIVE_INT32_MASK_FORMAT")
    need(len(raw) == count * 4 and len(mask) == count * 4
         and mask == struct.pack("<i", 1) * count, "UNPADDED_PROCESSOR_INPUT_BYTES")
    normalized = [x[0] for x in struct.iter_unpack("<f", raw)]
    need(all(math.isfinite(x) for x in normalized), "FINITE_NORMALIZED_INPUT")
    values = [x[0] / 32768.0 for x in struct.iter_unpack("<h", pcm)]
    need(len(values) == count and count > 0, "PROCESSOR_PCM_LENGTH")
    mean = math.fsum(values) / count
    variance = math.fsum((x - mean) ** 2 for x in values) / count
    norm_mean = math.fsum(normalized) / count
    norm_std = math.sqrt(math.fsum((x - norm_mean) ** 2 for x in normalized) / count)
    facts = capture.get("processor")
    need(type(facts) is dict and facts.get("sample_rate") == 16000
         and facts.get("pcm_samples") == count and facts.get("pcm_divisor") == 32768.0
         and facts.get("do_normalize") is True
         and facts.get("normalized_shape") == facts.get("attention_mask_shape") == [1, count]
         and facts.get("normalized_dtype") == "torch.float32" and facts.get("attention_mask_dtype") == "torch.int32"
         and facts.get("attention_mask_sum") == count and facts.get("attention_mask_unique_values") == [1]
         and facts.get("valid_input_samples") == facts.get("padded_input_samples") == count
         and facts.get("right_padding_samples") == 0, "PROCESSOR_METADATA_GEOMETRY")
    close_number(facts.get("pcm_float_mean"), mean, "PCM_MEAN_READBACK")
    close_number(facts.get("pcm_float_std_population"), math.sqrt(variance), "PCM_STD_READBACK")
    close_number(facts.get("normalized_mean"), norm_mean, "NORMALIZED_MEAN_READBACK")
    close_number(facts.get("normalized_std_population"), norm_std, "NORMALIZED_STD_READBACK")
    scale = math.sqrt(variance + 1e-7)
    discrepancy = max(abs(got - (x - mean) / scale) for x, got in zip(values, normalized))
    # The real extractor uses NumPy float32 reductions. This cross-check uses
    # float64 stdlib arithmetic, hence an explicit representation tolerance.
    allowance = 5e-6 * (1 + max(abs(x) for x in normalized))
    need(discrepancy <= allowance, "NORMALIZED_INPUT_PCM_RELATIONSHIP")
    return {"saved_input_bytes_verified": True, "pcm_mean": mean,
        "pcm_std_population": math.sqrt(variance), "normalized_mean": norm_mean,
        "normalized_std_population": norm_std,
        "float64_formula_max_absolute_difference": discrepancy,
        "float32_crosscheck_absolute_allowance": allowance,
        "bitwise_normalization_recomputation_claimed": False}


def expected_base(row, provider, run):
    return {"schema": observe.SCHEMA, "run": run, "id": row["id"], "source_id": row["source_id"],
        "provider": provider, "observer_identity": observe.observer_identity(provider),
        "input_pcm_sha256": row["pcm_sha256"], "input_wav_sha256": row["wav_sha256"],
        "input_samples": row["pcm_samples"], "diagnostic_only": True,
        "target_conditioned": False, "training_allowed": False, "retry_count": 0,
        "observer_code_sha256": sha(regular(Path(observe.__file__))),
        "native_processor_source_files": observe.PROCESSOR_SOURCE_FILES}


def check_base(value, base, label):
    need(type(value) is dict and all(k in value and type(value[k]) is type(v) and value[k] == v
         for k, v in base.items()), label)


def check_marker(value, base, name):
    check_base(value, base, "ROW_MARKER_IDENTITY")
    states = {"STARTED.json": (0, False), "FORWARD_STARTED.json": (1, False),
        "FORWARD_RETURNED.json": (1, True), "CAPTURE.json": (1, True)}
    if name in states:
        attempts, returned = states[name]
        check_base(value, {"forward_attempts": attempts, "forward_returned": returned},
            "ROW_MARKER_FORWARD_STATE")
    if name == "CAPTURE.json":
        need(value.get("status") == "success", "CAPTURE_MARKER_SUCCESS")
    if name == "FAILED.json":
        need(type(value.get("forward_attempts")) is int and value["forward_attempts"] in (0, 1)
             and type(value.get("forward_returned")) is bool
             and (not value["forward_returned"] or value["forward_attempts"] == 1)
             and value.get("status") == "failed" and value.get("capture_saved") is False,
             "FAILED_MARKER_FORWARD_STATE")


def read_capture(directory, row, provider, run, vocab, pcm):
    directory = Path(directory)
    raw = regular(directory / "CAPTURE.json")
    capture = loads(raw)
    base = expected_base(row, provider, run)
    check_base(capture, base, "CAPTURE_IDENTITY_INPUT_RUN")
    need(capture.get("status") == "success" and capture.get("forward_attempts") == 1
         and capture.get("forward_returned") is True and capture.get("input_coverage_complete") is True
         and capture.get("decoder_output_complete") is True
         and capture.get("ctc_helper_sha256") == observe.CTC_HELPER_SHA256
         and capture.get("blank_id") == 0 and capture.get("skipped_special_ids") == [0, 1, 2, 3]
         and capture.get("collapse_before_blank_deletion") is True
         and capture.get("old_crop_labels_modified") is False and capture.get("whole_target_pronunciation_gold") is False,
         "CAPTURE_COMPLETE_DIAGNOSTIC_ONLY")
    input_value = loads(regular(directory / "INPUT.json"))
    check_base(input_value, base, "INPUT_IDENTITY_INPUT_RUN")
    need(all(input_value.get(k) == capture.get(k) for k in ("processor", "normalized_input", "attention_mask")),
         "INPUT_CAPTURE_SAME_PROCESSOR_BYTES")
    geometry = observe.frame_geometry(row["pcm_samples"], row["full_to_crop_offset_16k"])
    need(capture.get("frame_geometry") == geometry, "FULL_LOGIT_FRAME_GEOMETRY")
    frames = geometry["frames"]
    saved_logits = capture.get("logits")
    need(type(saved_logits) is dict and saved_logits.get("shape") == [1, frames, 392]
         and saved_logits.get("dtype") == "float32-little-endian" and saved_logits.get("order") == "C",
         "LOGIT_FORMAT_NO_TRUNCATION")
    logits_raw = pinned(directory, saved_logits, "logits.f32")
    ids = argmax_ids(logits_raw, frames)
    ids_value = loads(pinned(directory, capture.get("frame_ids"), "frame-ids.json"))
    need(ids_value.get("schema") == observe.SCHEMA and ids_value.get("id") == row["id"]
         and type(ids_value.get("frame_ids")) is list
         and all(type(i) is int for i in ids_value["frame_ids"]) and ids_value["frame_ids"] == ids,
         "INDEPENDENT_ARGMAX_FRAME_IDS")
    full_runs = ctc_runs(ids, vocab)
    tokens = [r["token"] for r in full_runs]
    # A successful native CAPTURE must satisfy the unchanged producer helper.
    # Empty/overlong native greedy output is retained by that producer as a
    # returned forward with FAILED.json; pure region diagnostics may be empty.
    need(tokens == capture.get("phone_tokens") and 1 <= len(tokens) <= min(300, frames)
         and capture.get("argmax_special_frame_counts") == {str(k): ids.count(k) for k in SKIP_IDS},
         "INDEPENDENT_COLLAPSE_THEN_SKIP_CTC")
    first = (row["full_to_crop_offset_16k"] + 319) // 320
    need(0 <= first < frames and (frames - 1) * 320 + 400 <= row["pcm_samples"], "FIXED_TARGET_REGION_COORDINATES")
    target_runs = ctc_runs(ids[first:], vocab, first)
    reverse = {value: key for key, value in vocab.items()}
    focuses = [(token, vocab[token]) for token in FOCUS_TOKENS if token in vocab]
    target_frames = []
    for t in range(first, frames):
        values = struct.unpack_from("<392f", logits_raw, t * 392 * 4)
        ranked = sorted(range(392), key=lambda i: (-values[i], i))[:3]
        target_frames.append({"frame": t, "nominal_conv_support_samples": [320*t, 320*t+400],
            "top3": [{"id": i, "token": reverse[i], "logit": values[i]} for i in ranked],
            "focus_token_logits": {token: values[i] for token, i in focuses}})
    return {"capture_sha256": sha(raw), "logits_sha256": sha(logits_raw),
        "frame_ids_sha256": capture["frame_ids"]["sha256"], "frames": frames,
        "independent_argmax_and_ctc_verified": True,
        "processor_readback": processor_readback(directory, row, capture, pcm),
        "full_greedy": phone_diagnostics(tokens), "full_token_runs": full_runs,
        "fixed_region": {"rule": "Keep full-input frames whose nominal convolution start is at or after the old crop offset.",
            "frame_start": first, "frame_stop_exclusive": frames, "crop_offset_samples": row["full_to_crop_offset_16k"],
            "first_frame_offset_from_crop_samples": first*320-row["full_to_crop_offset_16k"],
            "nominal_grid_phase_aligned": row["full_to_crop_offset_16k"] % 320 == 0,
            "old_crop_output_equivalence_claimed": False, "duration_boundary_is_acoustic_gold": False,
            "independent_model_forward": False, "whole_phrase_admission": False,
            "greedy": phone_diagnostics([r["token"] for r in target_runs]), "token_runs": target_runs},
        "fixed_region_frame_logits": target_frames,
        "logits_are_calibrated_probabilities": False,
        "nominal_support_is_transformer_receptive_field": False}


def verify_lock(root, provider, manifest_sha, plan):
    lock = loads(regular(root / "MODEL_LOCK.json"))
    expected = {"schema": observe.SCHEMA, "model": observe.OBSERVERS[provider],
        "files": observe.OBSERVER_FILES[provider], "runtime_versions": observe.RUNTIME_VERSIONS,
        "ctc_helper_sha256": observe.CTC_HELPER_SHA256, "blank_id": 0,
        "skipped_special_ids": [0, 1, 2, 3], "collapse_before_blank_deletion": True,
        "weights_only": True, "strict_state_dict_loading": True, "eval": True,
        "model_parameters_modified": False, "trust_remote_code": False, "phonemizer_used": False,
        "target_conditioned": False, "new_model_constructions": 1,
        "native_forward_attempts_at_initialization": 0,
        "observer_identity": observe.observer_identity(provider),
        "observer_code_sha256": plan["observer_code_sha256"], "input_manifest_sha256": manifest_sha}
    expected["native_processor_source_files"] = observe.PROCESSOR_SOURCE_FILES
    check_base(lock, expected, "FIXED_OBSERVER_MODEL_LOCK")
    _, function_sha = observe.ctc_function(regular(observe.HERE / "english_final_ipa_xlsr_once.py"))
    need(lock.get("ctc_function_sha256") == function_sha, "IMMUTABLE_CTC_FUNCTION_SHA")
    for name, pin in observe.OBSERVER_FILES[provider].items():
        if name == "pytorch_model.bin":
            continue
        raw = regular(root / "model-metadata" / name)
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "SAVED_PROCESSOR_MODEL_METADATA")
    need(lock.get("preprocessor_config") == loads(regular(root / "model-metadata" / "preprocessor_config.json")),
         "SAVED_PREPROCESSOR_CONFIG")
    return vocabulary(regular(root / "model-metadata" / "vocab.json"))


def read_provider(root, rows, manifest_sha, run, provider, plan, *, source_root=None):
    need(provider in observe.PROVIDERS and len(rows) == 16, "TWO_BY16_DENOMINATOR")
    root = Path(root)
    need(not any(p.is_symlink() for p in (root, *root.parents)), "NO_PROVIDER_SYMLINK")
    allowed = {r["id"] for r in rows}
    if root.is_dir():
        need(not {p.name for p in root.iterdir() if p.is_dir() and p.name != "model-metadata"} - allowed,
             "NO_EXTRA_OBSERVER_INPUT_DIRECTORIES")
    result_path = root / "RESULT.json"
    if not result_path.is_file():
        retained = []
        for row in rows:
            folder = root / row["id"]
            markers = [name for name in ("STARTED.json", "FORWARD_STARTED.json", "FORWARD_RETURNED.json", "CAPTURE.json", "FAILED.json")
                       if (folder / name).is_file()]
            for name in markers:
                check_marker(loads(regular(folder / name)), expected_base(row, provider, run), name)
            lower = int("FORWARD_STARTED.json" in markers or "FORWARD_RETURNED.json" in markers or "CAPTURE.json" in markers)
            retained.append({"id": row["id"], "source_id": row["source_id"], "status": "MISSING_FINAL_LEDGER",
                "retained_markers": markers, "forward_attempts": None, "forward_attempts_lower_bound": lower,
                "forward_attempts_upper_bound": 1, "capture": None})
        return {"provider": provider, "status": "UNAVAILABLE_FINAL_LEDGER", "declared_count": 16,
            "actual_forward_attempts": None, "native_forward_returns": None, "completed": None,
            "forward_attempts_lower_bound": sum(r["forward_attempts_lower_bound"] for r in retained),
            "forward_attempts_upper_bound": 16, "rows": retained, "complete": False}
    result_raw = regular(result_path)
    result = loads(result_raw)
    check_base(result, {"schema": observe.SCHEMA, "provider": provider, "run": run,
        "declared_count": 16, "retry_count": 0, "new_tts_calls": 0, "new_text_asr_calls": 0,
        "old_cropped_input_forwards": 0, "training_allowed": False, "old_labels_modified": False,
        "diagnostic_only": True, "input_manifest_sha256": manifest_sha}, "OBSERVER_RESULT_BINDING")
    ledger = result.get("rows")
    need(type(ledger) is list and len(ledger) == 16 and [r.get("id") for r in ledger] == [r["id"] for r in rows],
         "COMPLETE_ORDERED_PROVIDER_LEDGER")
    start = loads(regular(root / "STARTED.json"))
    check_base(start, {"schema": observe.SCHEMA, "provider": provider, "run": run,
        "input_manifest_sha256": manifest_sha, "maximum_calls": 16, "retry_count": 0,
        "observer_identity": observe.observer_identity(provider), "diagnostic_only": True}, "PROVIDER_START_BINDING")
    vocab = verify_lock(root, provider, manifest_sha, plan) if (root / "MODEL_LOCK.json").is_file() else None
    output = []
    for row, item in zip(rows, ledger):
        need(item.get("source_id") == row["source_id"] and item.get("retry_count") == 0
             and type(item.get("forward_attempts")) is int and item["forward_attempts"] in (0, 1)
             and all(type(item.get(k)) is bool for k in ("input_started", "forward_returned", "capture_saved"))
             and (not item["forward_returned"] or item["forward_attempts"] == 1)
             and (not item["capture_saved"] or item["forward_returned"]), "SINGLE_CALL_LEDGER_ROW")
        folder = root / row["id"]
        base = expected_base(row, provider, run)
        markers = {name: folder / name for name in ("STARTED.json", "FORWARD_STARTED.json", "FORWARD_RETURNED.json", "CAPTURE.json", "FAILED.json")}
        for name, path in markers.items():
            if path.is_file():
                check_marker(loads(regular(path)), base, name)
        need(markers["STARTED.json"].is_file() == item["input_started"]
             and markers["FORWARD_STARTED.json"].is_file() == bool(item["forward_attempts"])
             and (not markers["FORWARD_RETURNED.json"].is_file() or item["forward_returned"])
             and markers["CAPTURE.json"].is_file() == item["capture_saved"], "LEDGER_MARKER_COUNTS")
        if item["forward_attempts"]:
            need(vocab is not None and item["input_started"], "FORWARD_HAS_PINNED_MODEL_AND_INPUT")
        capture = None
        if item["status"] == "success":
            need(item["forward_attempts"] == 1 and item["forward_returned"] and item["capture_saved"]
                 and markers["FORWARD_RETURNED.json"].is_file() and not markers["FAILED.json"].exists()
                 and item.get("capture_path") == row["id"] + "/CAPTURE.json"
                 and item.get("capture_sha256") == sha(regular(markers["CAPTURE.json"])), "SUCCESS_LEDGER_RECEIPTS")
            need(source_root is not None, "SOURCE_PCM_REQUIRED_FOR_CAPTURE_READBACK")
            pcm = regular(within(source_root, row["pcm_path"]))
            need(sha(pcm) == row["pcm_sha256"], "SOURCE_PCM_HASH")
            capture = read_capture(folder, row, provider, run, vocab, pcm)
        elif item["status"] == "NOT_STARTED_NO_RETRY":
            need(not item["input_started"] and item["forward_attempts"] == 0
                 and not item["forward_returned"] and not item["capture_saved"]
                 and not any(path.exists() for path in markers.values()), "NOT_STARTED_HAS_NO_CONSUMED_RECEIPTS")
        else:
            need(item["status"] == "failed" and not item["capture_saved"]
                 and type(item.get("failure")) is str and item["failure"], "EXPLICIT_FAILED_ROW")
            if markers["FAILED.json"].is_file():
                failed = loads(regular(markers["FAILED.json"]))
                need(all(failed.get(k) == item.get(k) for k in ("status", "input_started", "forward_attempts", "forward_returned", "capture_saved", "failure")),
                     "FAILED_ROW_LEDGER_JOIN")
            else:
                need(type(item.get("failure_receipt_write_error")) is str, "MISSING_FAILURE_RECEIPT_EXPLICIT")
        output.append({"id": row["id"], "source_id": row["source_id"], "status": item["status"],
            "forward_attempts": item["forward_attempts"], "forward_returned": item["forward_returned"],
            "forward_return_receipt_present": markers["FORWARD_RETURNED.json"].is_file(),
            "failure": item.get("failure"), "capture": capture})
    counts = {"actual_forward_attempts": sum(r["forward_attempts"] for r in ledger),
        "native_forward_returns": sum(r["forward_returned"] for r in ledger),
        "completed": sum(r["capture_saved"] for r in ledger),
        "input_started": sum(r["input_started"] for r in ledger),
        "failed": sum(r["status"] == "failed" for r in ledger),
        "not_started": sum(r["status"] == "NOT_STARTED_NO_RETRY" for r in ledger)}
    need(all(type(result.get(k)) is int and result[k] == v for k, v in counts.items())
         and counts["completed"] + counts["failed"] + counts["not_started"] == 16, "EXACT_OBSERVER_COUNTERS")
    good = counts["completed"] == counts["actual_forward_attempts"] == counts["native_forward_returns"] == 16 and result.get("failure") is None
    need(result.get("status") == ("COMPLETE_DIAGNOSTIC" if good else "INCOMPLETE_DIAGNOSTIC_NO_RETRY"), "RESULT_COMPLETENESS_STATUS")
    return {"provider": provider, "status": result["status"], "result_sha256": sha(result_raw),
        "declared_count": 16, **counts, "forward_attempts_lower_bound": counts["actual_forward_attempts"],
        "forward_attempts_upper_bound": counts["actual_forward_attempts"], "rows": output, "complete": good}


def original_probe(source_root):
    raw = regular(Path(source_root) / "inputs" / "probe.zip", 128 * 1024**2)
    files = source.archive_bytes(raw, "probe")
    need(sha(files["policy/english_phonetic_silver.py"]) == POLICY_SHA256, "ORIGINAL_POLICY_BYTES_UNCHANGED")
    analysis = loads(files["PROBE_ANALYSIS.json"])
    need(analysis.get("continuation", {}).get("status") == "STOP_DIAGNOSTIC_NO_EXPANSION"
         and len(analysis.get("rows", [])) == 48, "ORIGINAL_STOP_AND48_DENOMINATOR")
    return files, {r["id"]: r for r in analysis["rows"]}


def old_capture(files, old_rows, row, provider):
    path = "evidence/" + provider + "/" + row["source_id"] + "/CAPTURE.json"
    raw = files[path]
    value = loads(raw)
    check_base(value, {"schema": "english-kokoro48-native-capture-v1", "id": row["source_id"],
        **observe.observer_identity(provider), "source_pcm_sha256": row["old_pcm_sha256"],
        "source_samples": row["old_pcm_samples"], "status": "success", "input_coverage_complete": True,
        "decoder_output_complete": True, "target_conditioned": False,
        "diagnostic_only": provider == "phoneme_lv60"}, "ORIGINAL_CROP_CAPTURE_IDENTITY")
    original = old_rows[row["source_id"]]
    view = phone_diagnostics(value["phone_tokens"])
    field = "xlsr_normalized_phones" if provider == "phoneme" else "lv60_normalized_phones"
    need(original.get(field) == view["normalized_phones"], "ORIGINAL_PHONE_NORMALIZATION_UNCHANGED")
    return {"capture_path": path, "capture_sha256": sha(raw), **view,
        "frame_ids_available": False, "logits_available": False,
        "original_reference_tier": original["reference_tier"],
        "original_reference_supported": original["reference_supported"],
        "original_reference_hypothesis": original["reference_observed_hypothesis"],
        "original_reference_reason": original["reason"], "old_decision_modified": False}


def compare(*, prepared_root: Path, observer_roots: dict[str, Path], source_plan: dict, plan: dict, run_identity: dict) -> dict:
    source_root = prepared_root
    need(set(observer_roots) == set(observe.PROVIDERS), "EXACT_TWO_OBSERVER_ROOTS")
    need(plan.get("comparator_code_sha256") == sha(regular(Path(__file__)))
         and plan.get("observer_code_sha256") == sha(regular(Path(observe.__file__)))
         and plan.get("prepared_source_code_sha256") == sha(regular(Path(source.__file__))), "FROZEN_COMPARISON_SOURCE")
    manifest = source.readback(source_root, source_plan, allow_partial=True)
    manifest_sha = sha(regular(Path(source_root) / "INPUT_MANIFEST.json"))
    need(manifest_sha == plan.get("input_manifest_sha256")
         and manifest["run"] == plan.get("prior_run_identity"), "PRIOR_PREPARED_INPUT_IDENTITY")
    need(type(run_identity) is dict and set(run_identity) == {"repository", "commit", "run_id", "run_attempt", "event"}
         and run_identity["repository"] == "jiying2007/kws-data" and run_identity["event"] == "push"
         and run_identity["run_attempt"] == "1"
         and type(run_identity["commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", run_identity["commit"])
         and type(run_identity["run_id"]) is str and re.fullmatch(r"[1-9][0-9]*", run_identity["run_id"])
         and run_identity["run_id"] != manifest["run"]["run_id"]
         and run_identity["commit"] != manifest["run"]["commit"], "NEW_FIRST_FORWARD_RUN_IDENTITY")
    rows, run = manifest["rows"], run_identity
    files, old_rows = original_probe(source_root)
    providers = {p: read_provider(observer_roots[p], rows, manifest_sha, run, p, plan, source_root=source_root)
                 for p in observe.PROVIDERS}
    comparisons = []
    for row in rows:
        prepared = row.get("preparation_status") == "COMPLETE"
        derivation = loads(regular(within(source_root, row["derivation_path"]))) if prepared else None
        observed = {}
        for provider in observe.PROVIDERS:
            new = next(r for r in providers[provider]["rows"] if r["id"] == row["id"])
            old = old_capture(files, old_rows, row, provider)
            capture = new["capture"]
            full = None if capture is None else capture["full_greedy"]
            region = None if capture is None else capture["fixed_region"]["greedy"]
            observed[provider] = {"old_crop": old, "new_full": new,
                "requested_hypothesis": row["pronunciation_hypothesis"],
                "old_crop_whole_pattern_matches_request": row["pronunciation_hypothesis"] in old["whole_pattern_hypotheses"],
                "new_full_terminal_suffix_matches_request": None if full is None else any(x["hypothesis"] == row["pronunciation_hypothesis"] for x in full["terminal_suffix_matches"]),
                "new_fixed_region_whole_pattern_matches_request": None if region is None else row["pronunciation_hypothesis"] in region["whole_pattern_hypotheses"],
                "whole_phrase_admission_change": None, "old_reference_decision_preserved": True}
        comparisons.append({"id": row["id"], "source_id": row["source_id"], "preset_id": row["preset_id"],
            "cell_id": row["cell_id"], "requested_class": row["requested_class"],
            "pronunciation_hypothesis": row["pronunciation_hypothesis"], "preparation_status": row["preparation_status"],
            "old_crop_pcm_sha256": row["old_pcm_sha256"], "new_full_pcm_sha256": row.get("pcm_sha256"),
            "native_full_f32_sha256": row["native_full_f32_sha256"],
            "full_to_crop_offset_16k": row["full_to_crop_offset_16k"],
            "conv_grid_offset_mod_320": row["conv_grid_offset_mod_320"],
            "sample_comparison": None if derivation is None else derivation["old_crop_comparison"],
            "observers": observed})
    summary = []
    for cell in observe.CELLS:
        subset = [r for r in comparisons if r["cell_id"] == cell]
        for provider in observe.PROVIDERS:
            values = [r["observers"][provider] for r in subset]
            summary.append({"cell_id": cell, "provider": provider, "declared_count": 4,
                "completed_new_captures": sum(v["new_full"]["capture"] is not None for v in values),
                "old_crop_whole_pattern_matches_request": sum(v["old_crop_whole_pattern_matches_request"] for v in values),
                "new_full_terminal_suffix_matches_request": sum(v["new_full_terminal_suffix_matches_request"] is True for v in values),
                "new_fixed_region_whole_pattern_matches_request": sum(v["new_fixed_region_whole_pattern_matches_request"] is True for v in values),
                "new_capture_missing": sum(v["new_full"]["capture"] is None for v in values),
                "counts_are_diagnostic_pattern_matches_not_admission": True})
    complete = all(p["complete"] for p in providers.values()) and all(r["preparation_status"] == "COMPLETE" for r in rows)
    known = all(p["actual_forward_attempts"] is not None for p in providers.values())
    lower = sum(p["forward_attempts_lower_bound"] for p in providers.values())
    upper = sum(p["forward_attempts_upper_bound"] for p in providers.values())
    return {"schema": SCHEMA, "campaign_id": observe.CAMPAIGN_ID, "run": run, "source_run": manifest["run"],
        "status": "COMPLETE_DIAGNOSTIC_NO_ADMISSION" if complete else "INCOMPLETE_DIAGNOSTIC_NO_ADMISSION",
        "source_input_count": 16, "declared_observer_pairs": 32,
        "actual_forward_attempts": lower if known else None,
        "forward_attempts_lower_bound": lower, "forward_attempts_upper_bound": upper,
        "native_forward_returns": sum(p["native_forward_returns"] for p in providers.values()) if known else None,
        "completed_new_captures": sum(r["capture"] is not None for p in providers.values() for r in p["rows"]),
        "input_manifest_sha256": manifest_sha,
        "source_material_reused": True, "old_cropped_input_forwards": 0,
        "prior_input_preprocessing_attempts": 32, "prior_model_constructions": 2,
        "prior_phone_forward_attempts": 0, "recovery_of_failed_preprocessing": True,
        "new_input_derivations": 0, "phone_forward_replays": 0,
        "new_tts_calls": 0, "new_text_asr_calls": 0, "new_embedding_calls": 0, "new_training_calls": 0,
        "retry_count": 0, "old_labels_modified": False, "old_probe_decision": "STOP_DIAGNOSTIC_NO_EXPANSION",
        "training_allowed": False, "automatic_expansion": False, "selected_observer": None,
        "high_quality_model_passed": False, "diagnostic_only": True,
        "comparison_rules": {"phone_patterns": PHONE_PATTERNS, "policy_sha256": POLICY_SHA256,
            "normalization": "join-tokens-strip-stress-length-and-ties-map-rhotics-v1",
            "all_patterns_evaluated_without_target_conditioning": True,
            "full_suffix_match_is_whole_phrase_admission": False,
            "fixed_region_is_independent_model_forward": False,
            "old_frame_ids_or_logits_invented": False,
            "causal_scope": "Combined input context, independent normalization, FIR left boundary and convolution-grid sensitivity; no single mechanism isolated.",
            "nominal_conv_support_is_transformer_receptive_field": False,
            "acoustic_gold": False, "observer_winner_selected": False},
        "providers": {p: {k:v for k,v in info.items() if k != "rows"} for p, info in providers.items()},
        "by_cell_and_provider": summary, "rows": comparisons}
