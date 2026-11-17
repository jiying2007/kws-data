"""One Kokoro48 source probe with five blind observations; never training admission.

Source checks and the final join use only the standard library. The four presets
are synthesis styles, not authenticated independent people. Existing reference
rules stay fixed; LV60 is a diagnostic column only. No old audio is consumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import struct
import stat
import zipfile
import sys
import time
import urllib.request
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import english_kokoro48_recipe as recipe

SCHEMA = "english-kokoro48-execution-v1"
MAX_FILE = 16 * 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme", "phoneme_lv60")
REFERENCE_PROVIDERS = PROVIDERS[:4]
PRESETS = ("af_bella", "af_sarah", "am_michael", "am_puck")
MODES = ("short", "carrier")
RECOVERY = {
    "recovery_id": "kokoro48-zero-call-runtime-recovery-44114592216-v1",
    "repository": "jiying2007/kws-data", "previous_run_id": 44114592216,
    "previous_head_sha": "64404e9ebf6a38c6ce674345c6904548f33b7079",
    "previous_workflow": ".github/workflows/english-kokoro48-once.yml",
    "previous_recipe_sha256": "c5f207b5ecc59fa845ac77a8f6083ebab3f7d5eb87a31b6cfd4460d99dba880f",
    "ordered_request_roster_sha256": "507b40976f8219189655f75fc466375f8a66f9cd2ae9a671a7fa1f7a51a39c3e",
    "required_previous_model_attempts": 0, "previous_audio_replayed": 0,
    "automatic_retry": False, "runtime_only_change": True,
    "artifacts": {
        "source": {"id": 14342765906, "name": "kokoro48-source", "bytes": 2451122,
            "sha256": "dd63cc088ae627a340e9e5528e896fe331abd783d6a6000bc9834616ab78f825"},
        "runtime": {"id": 14342038028, "name": "kokoro48-generation-runtime", "bytes": 6609,
            "sha256": "2966990e1c34e801c50a7f83c2e38444fd2906b9b832d47254ff1fabb45fc550"},
        "probe": {"id": 14341299389, "name": "english-kokoro48-probe-44114592216", "bytes": 2690447,
            "sha256": "d2d21d11ad56f9fb45cbe15272a7d0dbb74c1a2cc07e5e828cdc072a530fa083"},
    },
}
POLICY = {
    "repository": "jiying2007/kws-pipeline",
    "commit": "011341c8541705a80da93ae423cfdb3d1ac9a82a",
    "module_path": "training/english_phonetic_silver.py",
    "module_bytes": 22786,
    "module_sha256": "fde516d9f1f2457cdb0047e8d8aee6a667c68fa6ea21f83d22f135e4bc9069da",
    "module_git_blob_sha1": "882694ef1cb66dcedd9234cce7c629b29f553a11",
    "profile_path": "configs/research/english-pronunciation-profile-20261115.json",
    "profile_bytes": 8591,
    "profile_sha256": "1b4112a27566d370e513d239ab9649b6c3cab3c8ca495c49c3810619214e4632",
    "profile_git_blob_sha1": "275145f6b9e1d5a6a2bd9f168eb792ad4d61d0a7",
    "rule_id": "english-synthetic-machine-admission-v2",
    "rules_sha256": "0a02fbef84f3de275b67908038a9fcd5c91a579138ce65d47037dffa48c60383",
}
PROBE_CRITERIA = {
    "reference_providers": list(REFERENCE_PROVIDERS),
    "diagnostic_only_providers": ["phoneme_lv60"],
    "reference_tiers": ["silver", "weak"],
    "minimum_balanced_presets_per_mode": 3,
    "minimum_mero_eh_request_with_observed_eh_presets_per_mode": 3,
    "ordinary_shared_across_modes": True,
    "minimum_positive_support_windows": 3,
    "positive_geometry": {"prefix_samples": 38400, "suffix_samples": 16000,
        "first_end": 16000, "stride": 1280, "ready_end": 35200,
        "support_left_offset": 31840, "support_right_offset": 128},
    "mode_preference": ["short", "carrier"],
    "failure_status": "STOP_DIAGNOSTIC_NO_EXPANSION",
    "success_status": "PROPOSE_SEPARATELY_PREREGISTERED_NEW_BATCH",
    "training_snapshot_allowed": False,
}
LV60_FILES = {
    "pytorch_model.bin": {"bytes": 1263535127, "sha256": "3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e"},
    "config.json": {"bytes": 1856, "sha256": "4609fb49b7e1d28aecb2840da1926c40bd915bc6f1120a940afacf7159bbfb13"},
    "preprocessor_config.json": {"bytes": 212, "sha256": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335"},
    "tokenizer_config.json": {"bytes": 321, "sha256": "1273ec458332cc55226ec596904bae811420755ace7e5d69462b61c4f480b457"},
    "special_tokens_map.json": {"bytes": 85, "sha256": "bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675"},
    "vocab.json": {"bytes": 4637, "sha256": "d732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0"},
}

OBSERVERS = {'qwen': {'recognizer_id': 'qwen', 'model_family': 'qwen3-asr', 'model_id': 'Qwen/Qwen3-ASR-0.6B', 'model_revision': '5eb144179a02acc5e5ba31e748d22b0cf3e303b0', 'model_sha256': '79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea', 'decode_config': {'language': 'English', 'context': '', 'max_new_tokens': 128, 'batch_size': 1, 'dtype': 'float32', 'device': 'cpu', 'attention': 'sdpa'}}, 'whisper': {'recognizer_id': 'whisper', 'model_family': 'whisper', 'model_id': 'openai/whisper-small', 'model_revision': '31243bad24cc746f07d4c8bfdd2d974872cb1803', 'model_sha256': '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794', 'decode_config': {'language': 'en', 'fp16': False, 'initial_prompt': None, 'condition_on_previous_text': False, 'temperature': 0.0, 'device': 'cpu'}}, 'parakeet': {'recognizer_id': 'parakeet', 'model_family': 'parakeet-tdt', 'model_id': 'nvidia/parakeet-tdt-0.6b-v3', 'model_revision': '541d1f99c6b0c3cd0b11a95167540bb8edefd82b', 'model_sha256': 'e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e', 'decode_config': {'device': 'cpu', 'prompt': None, 'threads': 4, 'timeout_seconds': 180}}, 'phoneme': {'recognizer_id': 'phoneme', 'model_family': 'wav2vec2-xlsr', 'model_id': 'facebook/wav2vec2-xlsr-53-espeak-cv-ft', 'model_revision': '2c733782da5604684829819a5eb744c193fe9398', 'model_sha256': '04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551', 'decode_config': {'device': 'cpu', 'decoder': 'ctc-greedy', 'target_input': None, 'special_tokens_removed': True, 'phonemizer_used': False, 'weights_only': True}}, 'phoneme_lv60': {'recognizer_id': 'phoneme_lv60', 'model_family': 'wav2vec2-lv60', 'model_id': 'facebook/wav2vec2-lv-60-espeak-cv-ft', 'model_revision': 'ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4', 'model_sha256': '3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e', 'decode_config': {'device': 'cpu', 'decoder': 'ctc-greedy', 'target_input': None, 'special_tokens_removed': True, 'phonemizer_used': False, 'weights_only': True}}}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha(canonical(value))


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("JSON_NONFINITE")))


def regular(path, limit=MAX_FILE):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and not any(p.is_symlink() for p in path.parents), "REGULAR_FILE")
    need(0 < path.stat().st_size <= limit, "FILE_SIZE")
    return path.read_bytes()


def read(path):
    return loads(regular(path))


def once(path, value):
    path = Path(path)
    need(path.parent.is_dir() and not path.exists() and not path.is_symlink(), "CREATE_ONCE_FILE")
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def new_directory(path):
    path = Path(path).absolute()
    need(path.parent.is_dir() and not path.exists() and not any(p.is_symlink() for p in (path, *path.parents)), "NEW_OUTPUT_DIRECTORY")
    path.mkdir(mode=0o700)
    return path


def within(root, name):
    need(type(name) is str and "\\" not in name, "SAFE_RELATIVE_PATH")
    rel = PurePosixPath(name)
    need(name == rel.as_posix() and rel.parts and not rel.is_absolute()
         and all(p not in ("", ".", "..") for p in rel.parts), "SAFE_RELATIVE_PATH")
    path = Path(root).joinpath(*rel.parts)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "NO_SYMLINK_PATH")
    return path


def identity(env=None):
    env = os.environ if env is None else env
    return {"repository": env.get("GITHUB_REPOSITORY"), "commit": env.get("GITHUB_SHA"),
        "run_id": env.get("GITHUB_RUN_ID"), "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "event": env.get("GITHUB_EVENT_NAME")}


def observer_identity(provider):
    row = OBSERVERS[provider]
    return {k: row[k] for k in ("recognizer_id", "model_family", "model_sha256")} | {"decode_config_sha256": digest(row["decode_config"])}


def validate_plan(plan, *, check_sources=True):
    need(type(plan) is dict and plan.get("schema") == SCHEMA and type(plan.get("enabled")) is bool, "PLAN_SCHEMA_ENABLE")
    need(all(type(plan.get(key)) is int for key in ("maximum_tts_calls", "maximum_calls_per_observer",
        "maximum_total_model_calls", "retry_count", "old_audio_model_calls", "new_kws_calls",
        "generation_deadline_seconds", "observation_deadline_seconds")), "INTEGER_BUDGETS")
    frozen = recipe.recipe()
    need(plan.get("campaign_id") == frozen.get("campaign_id") == recipe.CAMPAIGN_ID, "CAMPAIGN_ID")
    need(plan.get("recipe_sha256") == digest(frozen), "RECIPE_PIN")
    need(plan.get("zero_call_runtime_recovery") == RECOVERY, "EXACT_ZERO_CALL_RUNTIME_RECOVERY")
    need(plan.get("maximum_tts_calls") == plan.get("maximum_calls_per_observer") == 48
         and plan.get("maximum_total_model_calls") == 288 and plan.get("retry_count") == 0
         and plan.get("old_audio_model_calls") == plan.get("new_kws_calls") == 0, "FINITE_NEW_CALL_BUDGET")
    need(plan.get("generation_deadline_seconds") == 900 and plan.get("observation_deadline_seconds") == 3600, "STAGE_DEADLINE")
    need(plan.get("diagnostic_only") is True and plan.get("training_allowed") is False
         and plan.get("training_snapshot_allowed") is False, "PROBE_NEVER_TRAINS")
    need(plan.get("reference_contract") == POLICY and plan.get("probe_criteria") == PROBE_CRITERIA, "FROZEN_REFERENCE_AND_CONTINUATION_RULES")
    need(plan.get("observers") == OBSERVERS and plan.get("lv60_files") == LV60_FILES, "EXACT_FIVE_OBSERVERS")
    rows = frozen.get("rows")
    need(type(rows) is list and len(rows) == 48 and len({r["id"] for r in rows}) == 48
         and len({r["request_sha256"] for r in rows}) == 48, "EXACT_48_SOURCE_ROSTER")
    need({r["preset_id"] for r in rows} == set(PRESETS) and all(r["role"] == "train" for r in rows), "PRESET_TRAIN_ONLY_SCOPE")
    for preset in PRESETS:
        selected = [r for r in rows if r["preset_id"] == preset]
        need(len(selected) == 12 and len({r["source_group_id"] for r in selected}) == 1, "PRESET_SOURCE_FAMILY")
        for mode in MODES:
            cells = [r for r in selected if r["mode"] == mode]
            need(len(cells) == 4 and sorted(r["requested_class"] for r in cells) == [1, 1, 2, 2]
                 and {r["pronunciation_hypothesis"] for r in cells} == {
                    "mero-us-eh-v1", "mero-us-i-v1", "vosen-us-s-v1", "vosen-us-z-v1"}, "FOUR_HYPOTHESES_IN_BOTH_MODES")
        ordinary = [r for r in selected if r["mode"] == "ordinary"]
        need(len(ordinary) == 4 and all(r["requested_class"] == 0 for r in ordinary)
             and {r["requested_text"] for r in ordinary} == {"What time is it?", "Please turn on the kitchen light.",
                "The package is by the front door.", "Could you open the window?"}, "FOUR_ORDINARY_CONTROLS")
    if check_sources:
        for key, path in (("worker_sha256", Path(__file__)),
                          ("recipe_code_sha256", Path(recipe.__file__)),
                          ("generator_code_sha256", HERE / "english_kokoro48_generate.py"),
                          ("runtime_installer_sha256", HERE / "english_kokoro48_runtime.py")):
            need(plan.get(key) == sha(regular(path, 128 * 1024)), "SOURCE_PIN: " + key)
        helpers = plan.get("helper_sha256")
        need(type(helpers) is dict and len(helpers) == 19, "PINNED_HELPER_CLOSURE")
        for name, value in helpers.items():
            need(type(value) is str and HEX64.fullmatch(value)
                 and sha(regular(within(ROOT, name), 256 * 1024)) == value, "HELPER_SOURCE_PIN: " + name)
    if plan["enabled"]:
        need(type(plan.get("next_main_before")) is str and HEX40.fullmatch(plan["next_main_before"]), "EXACT_MAIN_PREDECESSOR")
        need(plan["next_main_before"] == RECOVERY["previous_head_sha"], "ONE_RECOVERY_FIXED_PREDECESSOR")
    else:
        need(plan.get("next_main_before") is None, "DISABLED_HAS_NO_EXECUTION_PREDECESSOR")
    return frozen


def execution_guard(plan, env=None, event=None):
    env = os.environ if env is None else env
    if event is None:
        event = read(Path(env["GITHUB_EVENT_PATH"]))
    validate_plan(plan)
    need(plan["enabled"] and env.get("GITHUB_REPOSITORY") == "jiying2007/kws-data"
         and env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == "refs/heads/main"
         and env.get("GITHUB_RUN_ATTEMPT") == "1", "PROTECTED_FIRST_MAIN_ONLY")
    need(type(event.get("repository")) is dict and event["repository"].get("private") is False
         and event["repository"].get("full_name") == "jiying2007/kws-data", "PUBLIC_REPOSITORY_ONLY")
    need(event.get("before") == plan["next_main_before"] and event.get("after") == env.get("GITHUB_SHA")
         and type(env.get("GITHUB_SHA")) is str and HEX40.fullmatch(env["GITHUB_SHA"])
         and event["after"] != event["before"], "EXACT_PUSH_IDENTITY")
    return identity(env)


class Deadline(BaseException):
    pass


def set_deadline(seconds):
    def stop(_signum, _frame):
        raise Deadline("FINITE_STAGE_DEADLINE")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)


def source_stage(plan, output, recovery_source, recovery_runtime, recovery_probe):
    run = execution_guard(plan)
    proof = verify_zero_call_recovery(plan, recovery_source, recovery_runtime, recovery_probe)
    output = new_directory(output)
    (output / "recovery").mkdir(mode=0o700)
    for key, path in (("source", recovery_source), ("runtime", recovery_runtime), ("probe", recovery_probe)):
        with (output / "recovery" / (key + ".zip")).open("xb") as stream:
            stream.write(regular(path))
    once(output / "RECOVERY_RECEIPT.json", proof)
    once(output / "EXECUTION_PLAN.json", plan)
    once(output / "RECIPE.json", recipe.recipe())
    once(output / "STARTED.json", {"schema": SCHEMA, "stage": "generate", "run": run,
        "maximum_calls": 48, "retry_count": 0, "training_snapshot_allowed": False})
    started = time.monotonic()
    error = None
    asset_root = output / "private-model"
    asset_root.mkdir(mode=0o700)
    set_deadline(plan["generation_deadline_seconds"])
    try:
        import english_kokoro48_generate as generator
        generator.generate(plan, output, run=run, asset_root=asset_root)
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        signal.alarm(0)
        if asset_root.exists():
            shutil.rmtree(asset_root)
        corpus = source_readback(output, plan)
        counts = corpus["source_attempts"]
        result = {"schema": SCHEMA, "stage": "generate", "run": run,
            "status": "COMPLETE" if counts["completed"] == 48 and error is None else "INCOMPLETE_NO_RETRY",
            "declared_count": 48, "actual_forward_attempts": counts["started"], "completed": counts["completed"],
            "valid_count": corpus["valid_count"], "retry_count": 0, "old_audio_model_calls": 0,
            "elapsed_seconds": time.monotonic() - started, "error_type": error,
            "training_snapshot_allowed": False}
        once(output / "RESULT.json", result)
    return result


def verify_wav(path, row):
    raw = regular(path)
    need(sha(raw) == row["wav_sha256"], "WAV_HASH")
    with wave.open(io.BytesIO(raw), "rb") as stream:
        n = stream.getnframes()
        need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) ==
             (1, 2, 16000, "NONE") and n == row["pcm_samples"] and 0 < n <= 2400000, "WAV_GEOMETRY")
        pcm = stream.readframes(n)
        need(not stream.readframes(1), "WAV_TRAILING_FRAMES")
    need(len(pcm) == n * 2 and sha(pcm) == row["pcm_sha256"], "PCM_HASH")
    return pcm


def geometry(samples):
    """Integer support only; exact original core geometry, no feature computation."""
    need(type(samples) is int and samples > 0, "POSITIVE_SOURCE_SAMPLES")
    observed = 38400 + samples + 16000
    ends = [end for end in range(16000, observed + 1, 1280) if end >= 35200
        and end - 31840 <= 38400 and end - 128 >= 38400 + samples]
    return {"positive_support_ends": ends, "positive_support_count": len(ends),
        "minimum": 3, "eligible": len(ends) >= 3, "new_embedding_calls": 0}


def file_record(path):
    raw = regular(path)
    return {"bytes": len(raw), "sha256": sha(raw)}


def verify_generation_audio(root, expected, receipt, vocab):
    """Bind original float source, complete tail, duration crop and observed WAV."""
    prefix = "source/audio/" + expected["id"] + "/"
    mapping = {"native_full_f32": "native-full.f32", "native_full_wav": "native-full.wav",
        "native_crop_f32": "native-crop.f32", "duration": "pred_dur.json", "crop": "CROP.json",
        "wav": "audio-16k.wav", "pcm": "audio-16k.pcm"}
    raw_files = {}
    for key, basename in mapping.items():
        need(receipt.get(key + "_path") == prefix + basename, "SOURCE_AUDIO_PATH")
        raw = regular(within(root, prefix + basename))
        need(sha(raw) == receipt.get(key + "_sha256"), "SOURCE_AUDIO_HASH: " + key)
        raw_files[key] = raw
    native_samples = receipt.get("native_samples")
    need(type(native_samples) is int and 0 < native_samples <= 600 * 50 * (len(expected["raw_phonemes"]) + 2)
         and len(raw_files["native_full_f32"]) == native_samples * 4, "FULL_NATIVE_FLOAT_GEOMETRY")
    floats = struct.unpack("<" + "f" * native_samples, raw_files["native_full_f32"])
    need(all(math.isfinite(x) for x in floats), "NATIVE_NONFINITE")
    with wave.open(io.BytesIO(raw_files["native_full_wav"]), "rb") as stream:
        need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes(), stream.getcomptype()) ==
             (1, 2, 24000, native_samples, "NONE"), "FULL_NATIVE_WAV_GEOMETRY")
        native_pcm = stream.readframes(native_samples)
    need(len(native_pcm) == native_samples * 2, "FULL_NATIVE_WAV_BYTES")
    expected_pcm = struct.pack("<" + "h" * native_samples,
        *(round(max(-1.0, min(32767 / 32768, value)) * 32768) for value in floats))
    need(native_pcm == expected_pcm, "FULL_NATIVE_WAV_EXACT_FLOAT_QUANTIZATION")
    duration = loads(raw_files["duration"])
    need(duration.get("id") == expected["id"] and duration.get("phonemes") == expected["raw_phonemes"]
         and duration.get("includes_bos_eos") is True and duration.get("samples_per_frame") == 600
         and duration.get("sample_rate") == 24000 and duration.get("boundary_is_acoustic_gold") is False, "DURATION_BINDING")
    durs = duration.get("pred_dur")
    ids = duration.get("input_ids")
    need(type(durs) is list and type(ids) is list and len(durs) == len(ids) == len(expected["raw_phonemes"]) + 2
         and all(type(x) is int and 1 <= x <= 50 for x in durs) and ids[0] == ids[-1] == 0
         and ids == receipt.get("phoneme_ids") == [0, *[vocab[c] for c in expected["raw_phonemes"]], 0]
         and native_samples == 600 * sum(durs), "PREDICTED_DURATION_GEOMETRY")
    crop = loads(raw_files["crop"])
    start = receipt.get("crop_start_native_sample")
    end = receipt.get("crop_end_native_sample")
    count = receipt.get("crop_native_samples")
    need(type(start) is int and type(end) is int and 0 <= start < end == native_samples
         and count == end - start, "CROP_KEEPS_NATIVE_RIGHT_TAIL")
    need(crop.get("mode") == expected["mode"] and crop.get("target_start_char") == expected["target_start_char"]
         and crop.get("crop_start_native_sample") == start and crop.get("crop_end_native_sample") == end
         and crop.get("crop_native_samples") == count and crop.get("right_endpoint") == "native_eof"
         and crop.get("human_word_end") is False and crop.get("boundary_is_acoustic_gold") is False
         and crop.get("recrop_count") == 0 and crop.get("observer_feedback_used") is False
         and crop.get("prediction_rule") == recipe.CROP and crop.get("actual_added_silence_samples") == 0, "CROP_RECEIPT_BINDING")
    if expected["mode"] == "carrier":
        boundary = 600 * (max(durs[0] - 3, 0) + sum(durs[1:expected["target_start_char"] + 1]))
        need(crop.get("predicted_target_start_native_sample") == boundary
             and crop.get("padding_before_samples") == 3600
             and crop.get("boundary_source") == "kokoro_predicted_duration"
             and start == max(0, boundary - 3600), "FIXED_DURATION_LEFT_CROP")
    else:
        need(start == 0 and crop.get("padding_before_samples") == 0
             and crop.get("boundary_source") == "whole_native_utterance", "NO_SHORT_OR_ORDINARY_CROP")
    need(raw_files["native_crop_f32"] == raw_files["native_full_f32"][start * 4:end * 4], "EXACT_FLOAT_SUBSTRING")
    need(receipt.get("pcm_samples") == (count * 2 + 2) // 3, "RESAMPLE_LENGTH_2_OVER_3")
    pcm = verify_wav(within(root, receipt["wav_path"]), receipt)
    need(raw_files["pcm"] == pcm and receipt.get("sample_rate") == 16000
         and receipt.get("channels") == 1 and receipt.get("sample_width") == 2, "OBSERVED_PCM_COMPANION")
    return {k: receipt[k] for key in mapping for k in (key + "_path", key + "_sha256")} | {
        k: receipt[k] for k in ("pcm_samples", "native_samples", "crop_start_native_sample",
            "crop_end_native_sample", "crop_native_samples", "sample_rate", "channels", "sample_width")}


def verify_source_lock(root):
    lock = read(root / "source/MODEL_LOCK.json")
    need(lock.get("campaign_id") == recipe.CAMPAIGN_ID
         and all(lock.get(k) == value for k, value in recipe.MODEL.items())
         and lock.get("runtime_versions") == recipe.RUNTIME_VERSIONS
         and lock.get("package") == recipe.PACKAGE
         and lock.get("all_component_weights_verified") is True
         and lock.get("preset_mixing") is False and lock.get("runtime_g2p_calls") == 0
         and lock.get("forward_attempts_at_lock") == 0
         and lock.get("human_speaker_identity_claimed") is False, "SOURCE_MODEL_LOCK")
    assets = lock.get("assets")
    need(type(assets) is list and len(assets) == len(recipe.assets()), "SOURCE_ASSET_CLOSURE")
    by_path = {a["path"]: a for a in assets}
    need(len(by_path) == len(assets), "SOURCE_DUPLICATE_ASSET")
    for expected in recipe.assets():
        actual = by_path.get(expected["path"], {})
        need(all(actual.get(k) == value for k, value in expected.items())
             and actual.get("verified") is True and actual.get("download_attempts") == 1, "SOURCE_ASSET_PIN")
        if expected["path"] != recipe.MODEL["model_path"]:
            raw = regular(within(root, "source/runtime/" + expected["path"]))
            need(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "RETAINED_SOURCE_ASSET_BYTES")
    installed = lock.get("installed_package_files")
    need(type(installed) is list and len(installed) == len(recipe.PACKAGE_FILES), "INSTALLED_PACKAGE_CLOSURE")
    need({r["path"]: [r["bytes"], r["sha256"]] for r in installed} == recipe.PACKAGE_FILES, "INSTALLED_PACKAGE_PIN")
    for name, (size, value) in recipe.PACKAGE_FILES.items():
        raw = regular(within(root, "source/runtime/installed/" + name))
        need(len(raw) == size and sha(raw) == value, "RETAINED_INSTALLED_PACKAGE_BYTES")
    wheel = regular(root / "source/runtime" / recipe.PACKAGE["filename"])
    need(len(wheel) == recipe.PACKAGE["bytes"] and sha(wheel) == recipe.PACKAGE["sha256"], "RETAINED_WHEEL_BYTES")
    proof = lock.get("weight_loading")
    need(type(proof) is list and len(proof) == 5 and {r["component"] for r in proof} ==
         {"bert", "bert_encoder", "predictor", "text_encoder", "decoder"}
         and all(r.get("all_keys_shapes_dtypes_values_equal") is True
                 and type(r.get("key_count")) is int and r["key_count"] > 0 for r in proof), "CHECKPOINT_READBACK_PROOF")
    config = read(root / "source/runtime/config.json")
    vocab = config.get("vocab")
    need(type(vocab) is dict and all(all(c in vocab for c in r["raw_phonemes"]) for r in recipe.recipe()["rows"]), "RETAINED_NATIVE_VOCAB")
    return vocab


def source_readback(root, plan):
    root = Path(root)
    frozen = recipe.recipe()
    need(read(root / "RECIPE.json") == frozen and read(root / "EXECUTION_PLAN.json") == plan, "SOURCE_RECIPE_OR_PLAN")
    recovery_proof = verify_zero_call_recovery(plan, root / "recovery/source.zip",
        root / "recovery/runtime.zip", root / "recovery/probe.zip")
    need(read(root / "RECOVERY_RECEIPT.json") == recovery_proof, "SOURCE_RECOVERY_RECEIPT_BINDING")
    start = read(root / "STARTED.json")
    need(start.get("schema") == SCHEMA and start.get("stage") == "generate"
         and start.get("maximum_calls") == 48 and start.get("retry_count") == 0
         and start.get("training_snapshot_allowed") is False, "SOURCE_STAGE_STARTED")
    for name in ("RECIPE.json", "EXECUTION_PLAN.json"):
        copy = root / "source" / name
        if copy.is_file():
            need(regular(copy) == regular(root / name), "SOURCE_NESTED_PLAN_BYTES")
    generation_start = root / "source/STARTED.json"
    if generation_start.is_file():
        begun_stage = read(generation_start)
        need(begun_stage.get("run") == start["run"] and begun_stage.get("declared_count") == 48
             and begun_stage.get("maximum_calls") == 48 and begun_stage.get("retry_count") == 0
             and begun_stage.get("diagnostic_only") is True
             and begun_stage.get("training_allowed") is False, "SOURCE_NATIVE_STAGE_IDENTITY")
    rows, attempts, completions, failures = [], 0, 0, 0
    vocab = verify_source_lock(root) if (root / "source" / "MODEL_LOCK.json").is_file() else None
    allowed_ids = {r["id"] for r in frozen["rows"]}
    audio_root = root / "source" / "audio"
    if audio_root.exists():
        need(audio_root.is_dir() and not audio_root.is_symlink()
             and all(p.name in allowed_ids and p.is_dir() and not p.is_symlink() for p in audio_root.iterdir()), "SOURCE_AUDIO_ROSTER")
    for expected in frozen["rows"]:
        row = dict(expected)
        folder = audio_root / expected["id"]
        begun, done, failed = (folder / name for name in ("STARTED.json", "COMPLETED.json", "FAILED.json"))
        row.update(generation_status="NOT_STARTED", wav_path=None, wav_sha256=None,
            pcm_path=None, pcm_sha256=None, pcm_samples=None, sample_rate=16000, channels=1, sample_width=2)
        if begun.is_file():
            receipt_start = read(begun)
            need(receipt_start.get("id") == expected["id"] and receipt_start.get("request_sha256") == expected["request_sha256"]
                 and receipt_start.get("attempt") == 1 and receipt_start.get("preset_id") == expected["preset_id"]
                 and receipt_start.get("mode") == expected["mode"] and receipt_start.get("role") == "train"
                 and receipt_start.get("phonemes") == expected["raw_phonemes"]
                 and receipt_start.get("target_start_char") == expected["target_start_char"]
                 and receipt_start.get("voice_pack_index") == len(expected["raw_phonemes"]) - 1
                 and receipt_start.get("seed") == expected["seed"]
                 and receipt_start.get("synthesis_config") == recipe.SYNTHESIS
                 and receipt_start.get("direct_api") == "KModel.forward"
                 and receipt_start.get("runtime_g2p_calls") == receipt_start.get("retry_count") == 0
                 and receipt_start.get("training_allowed") is False and receipt_start.get("diagnostic_only") is True, "SOURCE_SINGLE_REQUEST_STARTED")
            attempts += 1
        if done.is_file():
            need(begun.is_file() and not failed.exists() and (root / "source" / "MODEL_LOCK.json").is_file(), "SOURCE_TERMINAL_CLOSURE")
            receipt = read(done)
            need(receipt.get("status") == "COMPLETE" and receipt.get("id") == expected["id"]
                 and receipt.get("request_sha256") == expected["request_sha256"]
                 and all(receipt.get(k) == receipt_start.get(k) for k in ("preset_id", "mode", "role", "phonemes",
                    "phoneme_ids", "target_start_char", "seed", "voice_pack_index"))
                 and receipt.get("generation_model_sha256") == recipe.MODEL["model_sha256"]
                 and receipt.get("voice_pack_sha256") == recipe.VOICE_ASSETS[expected["preset_id"]]["sha256"]
                 and receipt.get("forward_attempts") == 1 and receipt.get("forward_returned") is True
                 and receipt.get("retry_count") == 0 and receipt.get("training_allowed") is False
                 and receipt.get("diagnostic_only") is True and receipt.get("human_gold") is False, "SOURCE_COMPLETE_BINDING")
            measured = verify_generation_audio(root, expected, receipt, vocab)
            row.update(measured)
            row.update(generation_status="success", source_receipt_sha256=sha(regular(done)),
                source_start_sha256=sha(regular(begun)))
            completions += 1
        elif failed.is_file():
            failure = read(failed)
            need(failure.get("id") == expected["id"] and failure.get("status") in
                 ("FAILED_NO_RETRY", "NOT_STARTED_NO_RETRY", "INTERRUPTED_NO_RETRY")
                 and failure.get("request_sha256") == expected["request_sha256"]
                 and failure.get("forward_attempts") == int(begun.is_file())
                 and failure.get("retry_count") == 0, "FAILED_SOURCE_IDENTITY")
            row["generation_status"] = failure["status"]
            failures += int(begun.is_file())
        elif begun.is_file():
            row["generation_status"] = "UNKNOWN_AFTER_START_NO_RETRY"
        if row["generation_status"] == "success" and not 400 <= row["pcm_samples"] <= 192000:
            row["generation_status"] = "OBSERVER_DURATION_QUARANTINED"
        rows.append(row)
    result_file = root / "source" / "RESULT.json"
    if result_file.is_file():
        terminal = read(result_file)
        need(terminal.get("actual_forward_attempts") == attempts and terminal.get("completed") == completions
             and terminal.get("retry_count") == 0 and terminal.get("old_audio_model_calls") == 0
             and terminal.get("run") == start["run"] and terminal.get("declared_count") == 48
             and terminal.get("training_allowed") is False and terminal.get("diagnostic_only") is True
             and type(terminal.get("rows")) is list and len(terminal["rows"]) == 48
             and {r["id"] for r in terminal["rows"]} == allowed_ids, "SOURCE_STAGE_COUNT_CLOSURE")
    hashes = [r["pcm_sha256"] for r in rows if r["generation_status"] == "success"]
    duplicates = {value for value, count in Counter(hashes).items() if count > 1}
    for row in rows:
        if row["pcm_sha256"] in duplicates:
            row["generation_status"] = "DUPLICATE_PCM_QUARANTINED"
    return {"schema": "english-kokoro48-source-v1", "run": start["run"], "campaign_id": recipe.CAMPAIGN_ID,
        "recipe_sha256": digest(frozen), "declared_count": 48, "rows": rows,
        "zero_call_runtime_recovery": {"recovery_id": RECOVERY["recovery_id"], "receipt_sha256": digest(recovery_proof)},
        "valid_count": sum(r["generation_status"] == "success" for r in rows),
        "source_attempts": {"started": attempts, "completed": completions, "failed": failures,
            "interrupted_after_start": attempts - completions - failures,
            "not_started": 48 - attempts},
        "duplicate_pcm_hashes": sorted(duplicates), "retry_count": 0,
        "human_gold": False, "presets_are_authenticated_human_speakers": False,
        "training_snapshot_allowed": False, "evaluation_scope": "TRAIN_ONLY_TTS_MECHANISM_PROBE"}


def load_lv60(folder):
    """Independent exact-pinned diagnostic loader; never rewrites XLSR constants."""
    from huggingface_hub import snapshot_download
    row = OBSERVERS["phoneme_lv60"]
    snapshot_download(repo_id=row["model_id"], revision=row["model_revision"], allow_patterns=list(LV60_FILES),
        local_dir=str(folder), max_workers=2, token=False)
    manifest = {}
    for name, expected in LV60_FILES.items():
        path = folder / name
        need(path.is_file() and not path.is_symlink(), "LV60_ASSET_FILE")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        manifest[name] = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
        need(manifest[name] == expected, "LV60_EXACT_ASSET: " + name)
    vocab = read(folder / "vocab.json")
    config_json = read(folder / "config.json")
    tokenizer = read(folder / "tokenizer_config.json")
    special = read(folder / "special_tokens_map.json")
    need(len(vocab) == 392 and set(vocab.values()) == set(range(392)) and vocab.get("iː") == 30
         and vocab.get("z") == 21 and config_json.get("pad_token_id") == 0
         and config_json.get("vocab_size") == 392, "LV60_EXACT_SHARED_VOCAB")
    skip = {0}
    for value in list(special.values()) + [tokenizer.get("word_delimiter_token")]:
        token = value.get("content") if isinstance(value, dict) else value
        if type(token) is str and token in vocab:
            skip.add(vocab[token])
    need(skip == {0, 1, 2, 3}, "LV60_SPECIAL_IDS")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    import english_final_ipa_xlsr_once as xlsr
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
    need(extractor.sampling_rate == 16000, "LV60_SAMPLE_RATE")
    config = Wav2Vec2Config.from_json_file(str(folder / "config.json"))
    model = Wav2Vec2ForCTC(config)
    state = torch.load(str(folder / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    need(isinstance(state, dict) and all(type(k) is str and isinstance(v, torch.Tensor) for k, v in state.items()), "LV60_TENSOR_STATE_ONLY")
    model.load_state_dict(state, strict=True)
    del state
    model.to("cpu").eval()
    def predict(wav):
        with wave.open(str(wav), "rb") as stream:
            frames = stream.getnframes()
            need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) ==
                 (1, 2, 16000, "NONE") and 400 <= frames <= 192000, "LV60_INPUT_PCM_GEOMETRY")
            pcm = stream.readframes(frames)
        need(len(pcm) == frames * 2, "LV60_INPUT_PCM_BYTES")
        values = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        need(bool(np.isfinite(values).all()), "LV60_INPUT_NONFINITE")
        features = extractor(values, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**features).logits
            need(logits.ndim == 3 and logits.shape[0] == 1 and logits.shape[2] == 392
                 and 1 <= logits.shape[1] <= 2000 and bool(torch.isfinite(logits).all()), "LV60_OUTPUT_GEOMETRY")
            ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        return xlsr.ctc_tokens(ids, vocab, skip), frames
    return predict, {"model_id": row["model_id"], "revision": row["model_revision"], "files": manifest,
        "blank_id": 0, "skipped_special_ids": sorted(skip), "weights_only": True,
        "trust_remote_code": False, "phonemizer_used": False, "diagnostic_only": True}


def load_observer(provider, model_root, plan):
    if provider == "phoneme_lv60":
        return load_lv60(model_root)
    if provider == "phoneme":
        import english_final_ipa_xlsr_once as xlsr
        decode, lock = xlsr.load_runtime(model_root)
        need(lock["files"]["vocab.json"] == LV60_FILES["vocab.json"], "BOTH_PHONE_VOCABS_BYTE_IDENTICAL")
        return decode, lock
    import english_seven_voice_asr_once as seven
    decode = seven._load(provider, model_root)
    return decode, {"verified_pinned_loader": "english_seven_voice_asr_once._load", "provider": provider,
        "source_sha256": plan["helper_sha256"]["automation/english_seven_voice_asr_once.py"]}


def observe(provider, source, plan, output):
    run = execution_guard(plan)
    need(provider in PROVIDERS, "KNOWN_PROVIDER")
    corpus = source_readback(source, plan)
    need(corpus["run"] == run, "SAME_RUN_SOURCE")
    rows = [r for r in corpus["rows"] if r["generation_status"] == "success"]
    output = new_directory(output)
    once(output / "STARTED.json", {"schema": SCHEMA, "stage": "observe", "run": run,
        "provider": provider, "source_corpus_sha256": digest(corpus), "valid_source_count": len(rows),
        "maximum_calls": 48, "retry_count": 0, "observer_identity": observer_identity(provider),
        "diagnostic_only": provider == "phoneme_lv60"})
    model_root = output / "private-model"
    model_root.mkdir(mode=0o700)
    calls = completed = 0
    error = None
    started = time.monotonic()
    set_deadline(plan["observation_deadline_seconds"])
    try:
        if rows:
            decode, actual_lock = load_observer(provider, model_root, plan)
            once(output / "MODEL_LOCK.json", {"identity": observer_identity(provider),
                "model": OBSERVERS[provider], "actual_runtime_lock": actual_lock})
        for row in rows:
            folder = output / row["id"]
            folder.mkdir(mode=0o700)
            source_wav = within(source, row["wav_path"])
            verify_wav(source_wav, row)
            once(folder / "STARTED.json", {"schema": SCHEMA, "provider": provider, "id": row["id"],
                "source_pcm_sha256": row["pcm_sha256"], "source_wav_sha256": row["wav_sha256"],
                "source_samples": row["pcm_samples"], "attempt": 1, "target_conditioned": False,
                "requested_text_passed": False, "role_passed": False})
            need(calls < 48, "OBSERVER_CALL_CEILING")
            calls += 1
            try:
                capture = observer_identity(provider) | {"schema": "english-kokoro48-native-capture-v1",
                    "id": row["id"], "source_pcm_sha256": row["pcm_sha256"],
                    "source_wav_sha256": row["wav_sha256"], "source_samples": row["pcm_samples"],
                    "status": "success", "target_conditioned": False,
                    "input_coverage_complete": True, "decoder_output_complete": True,
                    "diagnostic_only": provider == "phoneme_lv60"}
                if provider in ("phoneme", "phoneme_lv60"):
                    tokens, frames = decode(source_wav)
                    need(frames == row["pcm_samples"] and type(tokens) is list and 1 <= len(tokens) <= 300
                         and all(type(t) is str and 0 < len(t) <= 40 and not any(ord(c) < 32 for c in t) for t in tokens), "PHONE_OUTPUT_BINDING")
                    capture["phone_tokens"] = tokens
                else:
                    text = decode(source_wav, folder)
                    need(type(text) is str and 0 < len(text.strip()) <= 512
                         and not any(ord(c) < 32 for c in text.strip()), "ASR_OUTPUT_TEXT")
                    capture["raw_text"] = text
                once(folder / "CAPTURE.json", capture)
                completed += 1
            except BaseException as exc:
                once(folder / "FAILED.json", {"schema": SCHEMA, "id": row["id"], "provider": provider,
                    "status": "FAILED_NO_RETRY", "error_type": type(exc).__name__})
                if not isinstance(exc, Exception):
                    raise
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        signal.alarm(0)
        if model_root.exists():
            shutil.rmtree(model_root)
        result = {"schema": SCHEMA, "stage": "observe", "run": run, "provider": provider,
            "status": "COMPLETE" if completed == len(rows) and error is None else "INCOMPLETE_NO_RETRY",
            "declared_count": 48, "valid_source_count": len(rows), "actual_forward_attempts": calls,
            "completed": completed, "error_type": error, "elapsed_seconds": time.monotonic() - started,
            "retry_count": 0, "old_audio_model_calls": 0, "training_snapshot_allowed": False}
        once(output / "RESULT.json", result)
    return result


def copy_closed_tree(source, destination, maximum=512 * 1024**2):
    source, destination = Path(source), Path(destination)
    need(source.is_dir() and not source.is_symlink() and not destination.exists(), "COPY_TREE_GEOMETRY")
    files, total = [], 0
    for path in source.rglob("*"):
        need(not path.is_symlink(), "COPY_TREE_SYMLINK")
        if path.is_file():
            total += path.stat().st_size
            need(total <= maximum and path.stat().st_size <= MAX_FILE, "COPY_TREE_BYTE_LIMIT")
            files.append(path)
    destination.mkdir(mode=0o700)
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)


def collect_evidence(root, corpus):
    rows = []
    counts = {}
    expected = {r["id"]: r for r in corpus["rows"]}
    for provider in PROVIDERS:
        parent = Path(root) / "evidence" / provider
        count = {"declared_count": 48, "started": 0, "completed": 0, "failed": 0,
            "interrupted_after_start": 0, "not_started": 48, "retry_count": 0,
            "diagnostic_only": provider == "phoneme_lv60"}
        if parent.exists():
            need(all(p.name in expected and not p.is_symlink() for p in parent.iterdir() if p.is_dir()), "OBSERVER_SOURCE_ROSTER")
        for row in corpus["rows"]:
            folder = parent / row["id"]
            start, capture, failed = (folder / name for name in ("STARTED.json", "CAPTURE.json", "FAILED.json"))
            if start.is_file():
                value = read(start)
                need(row["generation_status"] == "success" and value.get("id") == row["id"]
                     and value.get("provider") == provider and value.get("attempt") == 1
                     and value.get("source_pcm_sha256") == row["pcm_sha256"]
                     and value.get("source_wav_sha256") == row["wav_sha256"]
                     and value.get("source_samples") == row["pcm_samples"]
                     and value.get("target_conditioned") is False
                     and value.get("requested_text_passed") is False and value.get("role_passed") is False, "OBSERVER_SINGLE_BLIND_ATTEMPT")
                count["started"] += 1
            need(not (capture.exists() and failed.exists()), "ONE_OBSERVER_TERMINAL")
            if capture.is_file():
                need(start.is_file() and (parent / "MODEL_LOCK.json").is_file(), "CAPTURE_STARTED_AND_LOADED")
                count["completed"] += 1
            elif failed.is_file():
                value = read(failed)
                need(start.is_file() and value.get("id") == row["id"] and value.get("provider") == provider
                     and value.get("status") == "FAILED_NO_RETRY", "FAILED_OBSERVER_BINDING")
                count["failed"] += 1
        count["interrupted_after_start"] = count["started"] - count["completed"] - count["failed"]
        count["not_started"] = 48 - count["started"]
        if (parent / "RESULT.json").is_file():
            result = read(parent / "RESULT.json")
            need(result.get("actual_forward_attempts") == count["started"]
                 and result.get("completed") == count["completed"] and result.get("retry_count") == 0
                 and result.get("old_audio_model_calls") == 0, "OBSERVER_STAGE_COUNT_CLOSURE")
            count["stage_status"] = result.get("status")
        else:
            count["stage_status"] = "UNKNOWN_WITHOUT_TERMINAL_RESULT"
        counts[provider] = count
    for row in corpus["rows"]:
        value = {"id": row["id"], "observations": [], "phone_observation": None,
            "diagnostic_phone_observation": None}
        for provider in PROVIDERS:
            path = Path(root) / "evidence" / provider / row["id"] / "CAPTURE.json"
            if not path.is_file():
                continue
            capture = read(path)
            need(row["generation_status"] == "success" and capture.get("id") == row["id"], "CAPTURE_VALID_SOURCE")
            need(all(capture.get(k) == v for k, v in observer_identity(provider).items()), "CAPTURE_MODEL_BINDING")
            need(capture.get("source_pcm_sha256") == row["pcm_sha256"]
                 and capture.get("source_wav_sha256") == row["wav_sha256"]
                 and capture.get("source_samples") == row["pcm_samples"], "CAPTURE_SOURCE_BINDING")
            need(capture.get("diagnostic_only") is (provider == "phoneme_lv60"), "LV60_DIAGNOSTIC_BOUNDARY")
            observation = capture | {"capture_path": path.relative_to(root).as_posix(), "capture_sha256": sha(regular(path))}
            if provider == "phoneme":
                value["phone_observation"] = observation
            elif provider == "phoneme_lv60":
                value["diagnostic_phone_observation"] = observation
            else:
                value["observations"].append(observation)
        rows.append(value)
    need(len(rows) == 48, "COMPLETE_48_EVIDENCE_DENOMINATOR")
    total_calls = corpus["source_attempts"]["started"] + sum(c["started"] for c in counts.values())
    need(total_calls <= 288, "GLOBAL_MODEL_CALL_CEILING")
    return {"schema": "english-kokoro48-machine-evidence-v1", "rows": rows, "observer_counts": counts,
        "total_actual_model_attempts": total_calls, "maximum_total_model_attempts": 288,
        "new_model_calls_during_join": 0, "retry_count": 0, "training_snapshot_allowed": False}


def fetch_policy(target):
    target.mkdir(mode=0o700)
    for kind, filename in (("module", "english_phonetic_silver.py"), ("profile", "profile.json")):
        url = f"https://raw.githubusercontent.com/{POLICY['repository']}/{POLICY['commit']}/{POLICY[kind + '_path']}"
        request = urllib.request.Request(url, headers={"User-Agent": "english-kokoro48-pinned-reference/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(256 * 1024 + 1)
        need(len(raw) == POLICY[kind + "_bytes"] and sha(raw) == POLICY[kind + "_sha256"], "PINNED_REFERENCE_BYTES")
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        need(blob == POLICY[kind + "_git_blob_sha1"], "REFERENCE_GIT_BLOB")
        with (target / filename).open("xb") as stream:
            stream.write(raw)
    once(target / "IDENTITY.json", POLICY)


def load_reference(target):
    path = Path(target) / "english_phonetic_silver.py"
    need(sha(regular(path)) == POLICY["module_sha256"], "ACTUAL_REFERENCE_MODULE_PIN")
    spec = importlib.util.spec_from_file_location("kokoro48_fixed_reference_rules", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    need(module.RULE_ID == POLICY["rule_id"] and module.RULES_SHA256 == POLICY["rules_sha256"], "ACTUAL_REFERENCE_RULE_PIN")
    need(all(tuple(observer_identity(provider)[key] for key in ("model_family", "model_sha256", "decode_config_sha256"))
             == module.MODEL_PINS[provider] for provider in REFERENCE_PROVIDERS), "REFERENCE_OBSERVER_IDENTITIES")
    return module


def continuation(reference_rows):
    """A prospective expansion suggestion only; no row is admitted to training."""
    need(len(reference_rows) == 48 and len({r["id"] for r in reference_rows}) == 48, "PROBE_RESULT_DENOMINATOR")
    modes = {}
    for mode in MODES:
        presets = []
        for preset in PRESETS:
            own = [r for r in reference_rows if r["preset_id"] == preset]
            positive = [r for r in own if r["mode"] == mode and r["reference_supported"]]
            ordinary = [r["id"] for r in own if r["mode"] == "ordinary" and r["reference_supported"]]
            mero = [r["id"] for r in positive if r["requested_class"] == 2]
            vosen = [r["id"] for r in positive if r["requested_class"] == 1]
            eh = [r["id"] for r in positive if r["pronunciation_hypothesis"] == "mero-us-eh-v1"
                  and r["reference_observed_hypothesis"] == "mero-us-eh-v1"]
            presets.append({"preset_id": preset, "mero_supported_ids": mero, "vosen_supported_ids": vosen,
                "ordinary_supported_ids": ordinary, "mero_eh_request_observed_eh_ids": eh,
                "balanced": bool(mero and vosen and ordinary), "mero_eh_supported": bool(eh)})
        balanced = [p["preset_id"] for p in presets if p["balanced"]]
        eh = [p["preset_id"] for p in presets if p["mero_eh_supported"]]
        modes[mode] = {"presets": presets, "balanced_presets": balanced,
            "mero_eh_request_observed_eh_presets": eh, "balanced_and_eh_presets": sorted(set(balanced) & set(eh)),
            "balanced_count": len(balanced), "mero_eh_count": len(eh),
            "passes": len(balanced) >= 3 and len(eh) >= 3}
    chosen = next((mode for mode in MODES if modes[mode]["passes"]), None)
    return {"modes": modes, "selected_mode": chosen,
        "status": "PROPOSE_SEPARATELY_PREREGISTERED_NEW_BATCH" if chosen else "STOP_DIAGNOSTIC_NO_EXPANSION",
        "training_snapshot_allowed": False, "automatic_expansion": False,
        "high_quality_model_passed": False, "diagnostic_lv60_used_for_decision": False}


def analyze(root, corpus, evidence, module):
    by_id = {r["id"]: r for r in evidence["rows"]}
    rows = []
    for row in corpus["rows"]:
        item = by_id[row["id"]]
        result = {k: row[k] for k in ("id", "preset_id", "source_group_id", "role", "mode", "cell_id",
            "requested_class", "requested_text", "pronunciation_hypothesis")}
        result.update(reference_tier="unknown", reference_class=None, reference_observed_hypothesis=None,
            reference_supported=False, reason="SOURCE_NOT_VALID", normalized_texts=[None, None, None],
            xlsr_normalized_phones=None, lv60_normalized_phones=None, lv60_whole_phrase_hypotheses=[],
            integer_geometry=None, training_snapshot_allowed=False)
        if row["generation_status"] == "success":
            pcm_ok, pcm_reason = module.verify_pcm(row, root)
            observations = {o["recognizer_id"]: o for o in item["observations"]}
            need(set(observations) <= {"qwen", "whisper", "parakeet"}, "ONLY_THREE_REFERENCE_TRANSCRIPTS")
            texts = [module.verify_observation(observations.get(p), observer_identity(p), row, root)
                     for p in ("qwen", "whisper", "parakeet")]
            phones = module.verify_observation(item["phone_observation"], observer_identity("phoneme"), row, root, phones=True)
            diagnostic = module.verify_observation(item["diagnostic_phone_observation"], observer_identity("phoneme_lv60"), row, root, phones=True)
            tier, cls, hypothesis, reason = module.decide_label(row, texts, phones)
            g = geometry(row["pcm_samples"]) if row["requested_class"] else None
            if not pcm_ok:
                tier, cls, hypothesis, reason = "unknown", None, None, pcm_reason
            elif g is not None and not g["eligible"]:
                tier, cls, hypothesis, reason = "unknown", None, None, "FEWER_THAN_THREE_WHOLE_CLIP_SUPPORT_WINDOWS"
            diagnostic_matches = [name for name, (_, pattern) in module.RULES["phone_patterns"].items()
                if diagnostic and re.fullmatch(pattern, diagnostic)]
            result.update(reference_tier=tier, reference_class=cls, reference_observed_hypothesis=hypothesis,
                reference_supported=tier in ("silver", "weak"), reason=reason, normalized_texts=texts,
                xlsr_normalized_phones=phones, lv60_normalized_phones=diagnostic,
                lv60_whole_phrase_hypotheses=diagnostic_matches, integer_geometry=g)
        rows.append(result)
    decision = continuation(rows)
    summaries = []
    for mode in ("short", "carrier", "ordinary"):
        for hypothesis in sorted({r["pronunciation_hypothesis"] for r in rows if r["mode"] == mode}):
            selected = [r for r in rows if r["mode"] == mode and r["pronunciation_hypothesis"] == hypothesis]
            summaries.append({"mode": mode, "requested_hypothesis": hypothesis, "declared_count": len(selected),
                "tiers": dict(Counter(r["reference_tier"] for r in selected)),
                "reasons": dict(Counter(r["reason"] for r in selected)),
                "observed_reference_hypotheses": dict(Counter(r["reference_observed_hypothesis"] or "UNKNOWN" for r in selected)),
                "lv60_whole_hypotheses": dict(Counter(name for r in selected for name in r["lv60_whole_phrase_hypotheses"]))})
    return {"schema": "english-kokoro48-probe-analysis-v1", "campaign_id": recipe.CAMPAIGN_ID,
        "rule_id": module.RULE_ID, "rules_sha256": module.RULES_SHA256,
        "reference_contract": POLICY, "criteria": PROBE_CRITERIA, "declared_count": 48,
        "rows": rows, "by_mode_and_hypothesis": summaries, "continuation": decision,
        "training_snapshot_allowed": False, "training_admitted": False, "human_gold": False,
        "new_model_calls_during_analysis": 0, "new_embedding_calls": 0,
        "scope": "TRAIN_ONLY_MECHANISM_PROBE_NOT_INDEPENDENT_TEST_OR_MODEL_QUALITY_ACCEPTANCE"}


def combine(source, observers, plan, output):
    run = execution_guard(plan)
    corpus = source_readback(source, plan)
    need(corpus["run"] == run, "SAME_RUN_CORPUS")
    output = new_directory(output)
    if (Path(source) / "source").is_dir():
        copy_closed_tree(Path(source) / "source", output / "source")
    else:
        (output / "source").mkdir(mode=0o700)
    (output / "generation").mkdir(mode=0o700)
    for name in ("RECIPE.json", "EXECUTION_PLAN.json", "STARTED.json", "RESULT.json", "RECOVERY_RECEIPT.json"):
        original = Path(source) / name
        if original.is_file():
            with (output / "generation" / name).open("xb") as stream:
                stream.write(regular(original))
    copy_closed_tree(Path(source) / "recovery", output / "recovery", 16 * 1024**2)
    once(output / "RECOVERY_RECEIPT.json", read(Path(source) / "RECOVERY_RECEIPT.json"))
    (output / "evidence").mkdir(mode=0o700)
    for provider in PROVIDERS:
        path = Path(observers) / ("kokoro48-observer-" + provider)
        if path.is_dir():
            started = read(path / "STARTED.json")
            need(started.get("schema") == SCHEMA and started.get("run") == run
                 and started.get("provider") == provider and started.get("maximum_calls") == 48
                 and started.get("retry_count") == 0 and started.get("source_corpus_sha256") == digest(corpus)
                 and started.get("observer_identity") == observer_identity(provider), "OBSERVER_HANDOFF_IDENTITY")
            copy_closed_tree(path, output / "evidence" / provider, 32 * 1024**2)
    once(output / "CORPUS_MANIFEST.json", corpus)
    evidence = collect_evidence(output, corpus)
    once(output / "MACHINE_EVIDENCE.json", evidence)
    once(output / "EXECUTION_PLAN.json", plan)
    once(output / "RECIPE.json", recipe.recipe())
    fetch_policy(output / "policy")
    module = load_reference(output / "policy")
    analysis = analyze(output, corpus, evidence, module)
    once(output / "PROBE_ANALYSIS.json", analysis)
    result = {"schema": SCHEMA, "stage": "combine", "run": run, "status": "DIAGNOSTIC_PROBE_PACKET_RETAINED",
        "probe_decision": analysis["continuation"]["status"], "selected_mode": analysis["continuation"]["selected_mode"],
        "declared_count": 48, "valid_sources": corpus["valid_count"], "source_attempts": corpus["source_attempts"],
        "observer_counts": evidence["observer_counts"], "total_actual_model_attempts": evidence["total_actual_model_attempts"],
        "maximum_total_model_attempts": 288, "retry_count": 0, "new_model_calls_during_join": 0,
        "training_snapshot_allowed": False, "training_admitted": False, "automatic_expansion": False,
        "human_gold": False, "physical_ssc305_execution": False, "high_quality_model_passed": False,
        "analysis_sha256": sha(regular(output / "PROBE_ANALYSIS.json"))}
    once(output / "RESULT.json", result)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            record = file_record(path)
            files.append({"path": path.relative_to(output).as_posix(), **record})
    once(output / "FILES.json", {"schema": "english-kokoro48-files-v1", "files": files})
    return result


def recovery_archive(path, key):
    expected = RECOVERY["artifacts"][key]
    raw = regular(path)
    need(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "EXACT_PREVIOUS_API_ARCHIVE: " + key)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        need(len(names) == len(set(names)) <= 2000 and sum(info.file_size for info in infos) <= 32 * 1024**2, "RECOVERY_ZIP_BOUND")
        for info in infos:
            need(type(info.filename) is str and "\\" not in info.filename and "\0" not in info.filename
                 and not info.is_dir() and not info.flag_bits & 1 and info.file_size <= MAX_FILE
                 and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG)
                 and not PurePosixPath(info.filename).is_absolute()
                 and all(p not in ("", ".", "..") for p in info.filename.split("/")), "RECOVERY_ZIP_SAFE_MEMBER")
        files = {name: archive.read(name) for name in names}
    if key == "probe":
        manifest = loads(files["FILES.json"])
        need(manifest.get("schema") == "english-kokoro48-files-v1", "OLD_CLOSED_MANIFEST_SCHEMA")
        rows = manifest.get("files")
        need(type(rows) is list and len({r["path"] for r in rows}) == len(rows)
             and {r["path"] for r in rows} | {"FILES.json"} == set(files), "OLD_CLOSED_MEMBER_SET")
        for row in rows:
            member = files[row["path"]]
            need(len(member) == row["bytes"] and sha(member) == row["sha256"], "OLD_CLOSED_MEMBER_BYTES")
    return files


def verify_zero_call_documents(source, runtime, probe):
    """Inspect saved zero records, never reclassify old UNKNOWN observations."""
    old_run = {"repository": RECOVERY["repository"], "commit": RECOVERY["previous_head_sha"],
        "run_id": str(RECOVERY["previous_run_id"]), "run_attempt": "1", "event": "push"}
    frozen = recipe.recipe()
    old_recipe = loads(source["RECIPE.json"])
    need(digest(old_recipe) == RECOVERY["previous_recipe_sha256"], "EXACT_PREVIOUS_RECIPE")
    comparable = dict(old_recipe)
    comparable["runtime_versions"] = frozen["runtime_versions"]
    need(comparable == frozen and old_recipe["rows"] == frozen["rows"], "ONLY_RUNTIME_CHANGED_IDENTICAL_REQUESTS")
    roster = [{k: row[k] for k in ("id", "request_sha256")} for row in old_recipe["rows"]]
    need(len(roster) == 48 and len({r["id"] for r in roster}) == 48
         and digest(roster) == RECOVERY["ordered_request_roster_sha256"], "EXACT_48_UNSTARTED_REQUEST_IDENTITIES")
    old_plan = loads(source["EXECUTION_PLAN.json"])
    need(old_plan.get("reference_contract") == POLICY and old_plan.get("probe_criteria") == PROBE_CRITERIA
         and old_plan.get("observers") == OBSERVERS and old_plan.get("lv60_files") == LV60_FILES
         and old_plan.get("maximum_tts_calls") == old_plan.get("maximum_calls_per_observer") == 48
         and old_plan.get("maximum_total_model_calls") == 288 and old_plan.get("retry_count") == 0,
         "OLD_REFERENCE_RULES_AND_BUDGET_UNCHANGED")
    need(source["RECIPE.json"] == source["source/RECIPE.json"]
         and source["EXECUTION_PLAN.json"] == source["source/EXECUTION_PLAN.json"], "OLD_INNER_OUTER_PLAN_BYTES")
    for name in ("STARTED.json", "source/STARTED.json", "RESULT.json", "source/RESULT.json"):
        need(loads(source[name]).get("run") == old_run, "EXACT_PREVIOUS_SOURCE_RUN")
    outer, inner = loads(source["RESULT.json"]), loads(source["source/RESULT.json"])
    need(outer.get("status") == inner.get("status") == "INCOMPLETE_NO_RETRY"
         and all(doc.get("declared_count") == 48 and doc.get("actual_forward_attempts") == 0
                 and doc.get("completed") == 0 and doc.get("retry_count") == 0
                 and doc.get("old_audio_model_calls") == 0 for doc in (outer, inner))
         and outer.get("valid_count") == 0 and inner.get("actual_forward_returns") == 0
         and inner.get("kws_forward_calls") == inner.get("training_calls") == inner.get("runtime_g2p_calls") == 0
         and inner.get("training_allowed") is False and inner.get("diagnostic_only") is True
         and inner.get("error") == {"type": "ModuleNotFoundError", "code": "No module named 'click'"}, "ACTUAL_PREVIOUS_ZERO_CALL_INITIALIZATION_FAILURE")
    expected_failures = {"source/audio/" + row["id"] + "/FAILED.json" for row in roster}
    need({name for name in source if name.startswith("source/audio/")} == expected_failures,
         "NO_PREVIOUS_CELL_STARTED_OR_AUDIO")
    for row in roster:
        receipt = loads(source["source/audio/" + row["id"] + "/FAILED.json"])
        need(receipt.get("id") == row["id"] and receipt.get("request_sha256") == row["request_sha256"]
             and receipt.get("status") == "NOT_STARTED_NO_RETRY" and receipt.get("forward_attempts") == 0
             and receipt.get("forward_returned") is False and receipt.get("retry_count") == 0
             and receipt.get("training_allowed") is False, "ALL_48_PREVIOUS_REQUESTS_UNSTARTED")
    for files in (source, runtime, probe):
        need(not any(name.lower().endswith((".wav", ".pcm", ".f32")) or name.endswith(("/CAPTURE.json", "/COMPLETED.json"))
                     or name.endswith(("MODEL_LOAD_STARTED.json", "MODEL_LOCK.json")) for name in files),
             "NO_PREVIOUS_AUDIO_CAPTURE_OR_MODEL_CONSTRUCTION")
    need(loads(runtime["GUARD.json"]) == old_run and loads(runtime["COMMAND_EXIT.json"]) == {"returncode": 2, "retry_count": 0}, "OLD_RUNTIME_EXIT_AND_GUARD")
    final = loads(probe["RESULT.json"])
    corpus = loads(probe["CORPUS_MANIFEST.json"])
    evidence = loads(probe["MACHINE_EVIDENCE.json"])
    analysis = loads(probe["PROBE_ANALYSIS.json"])
    need(loads(runtime["SOURCE_READBACK.json"]) == corpus and corpus.get("run") == old_run,
         "OLD_RUNTIME_SOURCE_READBACK_CLOSURE")
    need(corpus.get("valid_count") == 0 and corpus.get("declared_count") == 48
         and corpus.get("source_attempts") == {"started": 0, "completed": 0, "failed": 0,
             "interrupted_after_start": 0, "not_started": 48}
         and len(corpus.get("rows", [])) == 48
         and all(row.get("generation_status") == "NOT_STARTED_NO_RETRY" and row.get("wav_path") is None
                 and row.get("pcm_path") is None and row.get("pcm_sha256") is None for row in corpus["rows"]), "OLD_ZERO_SOURCE_CORPUS")
    need(final.get("run") == old_run and final.get("total_actual_model_attempts") == 0
         and final.get("status") == "DIAGNOSTIC_PROBE_PACKET_RETAINED"
         and final.get("probe_decision") == "STOP_DIAGNOSTIC_NO_EXPANSION"
         and final.get("training_snapshot_allowed") is False and final.get("training_admitted") is False
         and final.get("automatic_expansion") is False and final.get("high_quality_model_passed") is False,
         "OLD_FINAL_PROBE_ZERO_CALLS")
    need(set(evidence.get("observer_counts", {})) == set(PROVIDERS)
         and all(c.get("declared_count") == 48 and c.get("started") == c.get("completed") == c.get("failed") == 0
                 and c.get("not_started") == 48 and c.get("retry_count") == 0
                 for c in evidence["observer_counts"].values())
         and evidence.get("total_actual_model_attempts") == 0 and len(evidence.get("rows", [])) == 48
         and all(row.get("observations") == [] and row.get("phone_observation") is None
                 and row.get("diagnostic_phone_observation") is None for row in evidence["rows"]), "OLD_FIVE_OBSERVERS_ALL_ZERO")
    need(len(analysis.get("rows", [])) == 48 and all(row.get("reference_tier") == "unknown"
         and row.get("reference_supported") is False for row in analysis["rows"])
         and analysis.get("training_snapshot_allowed") is False
         and analysis.get("continuation", {}).get("status") == "STOP_DIAGNOSTIC_NO_EXPANSION", "OLD_UNKNOWN_PRESERVED_WITHOUT_RECLASSIFICATION")
    need(loads(probe["policy/IDENTITY.json"]) == POLICY
         and sha(probe["policy/english_phonetic_silver.py"]) == POLICY["module_sha256"]
         and sha(probe["policy/profile.json"]) == POLICY["profile_sha256"], "EXACT_UNCHANGED_REFERENCE_POLICY")
    for name, raw in source.items():
        target = "generation/" + name if name in ("RECIPE.json", "EXECUTION_PLAN.json", "STARTED.json", "RESULT.json") else name
        need(probe.get(target) == raw, "OLD_SOURCE_PROBE_BYTE_BINDING")
    return {"schema": "english-kokoro48-zero-call-runtime-recovery-v1", "recovery": RECOVERY,
        "status": "VERIFIED_PREVIOUS_ZERO_CALLS_REQUESTS_NEVER_STARTED", "previous_run": old_run,
        "previous_recipe_sha256": digest(old_recipe), "current_recipe_sha256": digest(frozen),
        "ordered_request_roster_sha256": digest(roster), "same_48_requests_seeds_presets_phones": True,
        "previous_tts_attempts": 0, "previous_observer_attempts": 0, "previous_audio_files": 0,
        "old_unknown_labels_reclassified": False, "new_model_calls_during_recovery_check": 0,
        "automatic_retry": False, "training_snapshot_allowed": False}


def verify_zero_call_recovery(plan, source_zip, runtime_zip, probe_zip):
    need(plan.get("zero_call_runtime_recovery") == RECOVERY, "EXACT_ZERO_CALL_RUNTIME_RECOVERY")
    source = recovery_archive(source_zip, "source")
    runtime = recovery_archive(runtime_zip, "runtime")
    probe = recovery_archive(probe_zip, "probe")
    return verify_zero_call_documents(source, runtime, probe)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--source-check", action="store_true")
    actions.add_argument("--guard", action="store_true")
    actions.add_argument("--verify-zero-call-recovery", action="store_true")
    actions.add_argument("--generate", action="store_true")
    actions.add_argument("--read-source", action="store_true")
    actions.add_argument("--observe", choices=PROVIDERS)
    actions.add_argument("--combine", action="store_true")
    parser.add_argument("--recovery-source", type=Path)
    parser.add_argument("--recovery-runtime", type=Path)
    parser.add_argument("--recovery-probe", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--observers", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    validate_plan(plan)
    if args.source_check:
        result = {"status": "SOURCE_ONLY_NO_MODEL", "enabled": plan["enabled"], "declared_sources": 48,
            "maximum_total_model_attempts": 288, "training_snapshot_allowed": False}
    elif args.guard:
        result = execution_guard(plan)
    elif args.verify_zero_call_recovery:
        need(all(p is not None for p in (args.recovery_source, args.recovery_runtime, args.recovery_probe)), "THREE_OLD_ARCHIVES_REQUIRED")
        result = verify_zero_call_recovery(plan, args.recovery_source, args.recovery_runtime, args.recovery_probe)
        if args.output is not None:
            once(args.output, result)
    elif args.generate:
        need(args.output is not None and all(p is not None for p in
            (args.recovery_source, args.recovery_runtime, args.recovery_probe)), "OUTPUT_AND_OLD_PROOF_REQUIRED")
        result = source_stage(plan, args.output, args.recovery_source, args.recovery_runtime, args.recovery_probe)
    elif args.read_source:
        need(args.source is not None, "SOURCE_REQUIRED")
        corpus = source_readback(args.source, plan)
        if args.output is not None:
            once(args.output, corpus)
        result = {"status": "SOURCE_READBACK", "valid_count": corpus["valid_count"], "corpus_sha256": digest(corpus),
            "training_snapshot_allowed": False}
    elif args.observe:
        need(args.source is not None and args.output is not None, "OBSERVER_INPUTS_REQUIRED")
        result = observe(args.observe, args.source, plan, args.output)
    else:
        need(all(x is not None for x in (args.source, args.observers, args.output)), "COMBINE_INPUTS_REQUIRED")
        result = combine(args.source, args.observers, plan, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result.get("status") == "INCOMPLETE_NO_RETRY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
