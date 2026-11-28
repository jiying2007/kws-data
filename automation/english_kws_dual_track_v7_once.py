"""One finite fresh-source Piper campaign and first blind observers.

Source checks and packet joins use the standard library. Heavy imports occur only
inside guarded public-main execution stages. Every attempted inference has a
create-once receipt; interruptions never make a cell eligible for retry.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import sys
import time
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import english_kws_dual_track_v7_recipe as recipe

SCHEMA = "english-kws-dual-track-v7-kokoro-execution-plan-v1"
MAX_FILE = 4 * 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def need(condition, reason):
    if not condition:
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
    return json.loads(raw, object_pairs_hook=unique)


def regular(path, limit=MAX_FILE):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and
         not any(x.is_symlink() for x in path.parents), "REGULAR_FILE")
    need(0 < path.stat().st_size <= limit, "FILE_SIZE")
    return path.read_bytes()


def read(path):
    return loads(regular(path))


def once(path, value):
    path = Path(path)
    need(path.parent.is_dir() and not path.exists() and not path.is_symlink(), "CREATE_ONCE_FILE")
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def new_directory(path):
    path = Path(path).absolute()
    need(path.parent.is_dir() and not path.exists() and
         not any(p.is_symlink() for p in (path, *path.parents)), "NEW_OUTPUT_DIRECTORY")
    path.mkdir(mode=0o700)
    return path


def within(root, name):
    rel = PurePosixPath(name)
    need(type(name) is str and name == rel.as_posix() and rel.parts and
         not rel.is_absolute() and not any(p in ("", ".", "..") for p in rel.parts), "SAFE_RELATIVE_PATH")
    path = Path(root).joinpath(*rel.parts)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "NO_SYMLINK_PATH")
    return path


def identity(env=None):
    env = os.environ if env is None else env
    return {
        "repository": env.get("GITHUB_REPOSITORY"),
        "event": env.get("GITHUB_EVENT_NAME"),
        "ref": env.get("GITHUB_REF"),
        "sha": env.get("GITHUB_SHA"),
        "run_id": env.get("GITHUB_RUN_ID"),
        "attempt": env.get("GITHUB_RUN_ATTEMPT"),
    }


def validate_plan(plan, *, check_sources=True, allow_autofill=False):
    need(plan.get("schema") == SCHEMA and plan.get("enabled") is True,
         "ENABLED_KOKORO_V7_EXECUTION_PLAN")
    need(plan.get("source_repository") == "jiying2007/kws-data"
         and plan.get("next_main_before") == "5902f1f8d4fdc8096a864eb5d814c253d6f58ff2",
         "EXACT_PUBLIC_MAIN_SUCCESSOR")
    need(plan.get("source_plan_blob_sha1") == recipe.SOURCE_PLAN_BLOB_SHA1
         and plan.get("registration_blob_sha1") == recipe.REGISTRATION_BLOB_SHA1,
         "EXACT_KOKORO_SOURCE_PLAN_AND_REGISTRATION_BLOBS")
    source_plan, registration = recipe.source_documents()
    source_plan_sha = sha(regular(recipe.SOURCE_PLAN_PATH))
    registration_sha = sha(regular(recipe.REGISTRATION_PATH))
    need(plan.get("source_plan_sha256") == source_plan_sha
         and registration["source_map_evidence"]["source_plan_sha256"] == source_plan_sha,
         "KOKORO_SOURCE_PLAN_SHA256")
    frozen = recipe.recipe()
    need(plan.get("recipe_sha256") == recipe.digest(frozen)
         or (allow_autofill and plan.get("recipe_sha256") == "AUTO"),
         "FROZEN_KOKORO_RECIPE_SHA256")
    need(plan.get("source_count") == recipe.SOURCE_COUNT == 96
         and frozen["source_count"] == 96
         and plan.get("source_role_counts") == {"train": 6, "development": 2}
         and frozen["source_role_counts"] == {"train": 6, "development": 2}
         and len(frozen["source_groups"]) == 8
         and len(frozen["rows"]) == 96,
         "EXACT_96_ROW_6_2_KOKORO_TRAIN_DEV_PANEL")
    role_counts = {}
    for group in frozen["source_groups"]:
        role_counts[group["role"]] = role_counts.get(group["role"], 0) + 1
    need(role_counts == {"train": 6, "development": 2}
         and len({group["source_speaker_id"] for group in frozen["source_groups"]}) == 8,
         "EXACT_EIGHT_UNIQUE_VOICE_GROUPS")
    need(plan.get("observer_providers") == list(recipe.PROVIDERS)
         and plan.get("tts_requests") == 96
         and plan.get("observer_requests_per_provider") == 96
         and plan.get("total_high_level_model_requests") == 480
         and plan.get("new_kws_model_forwards") == 0
         and plan.get("optimizer_updates") == 0
         and plan.get("retries") == 0
         and plan.get("automatic_expansion") is False,
         "FIXED_480_CALL_NO_TRAINING_NO_RETRY_BUDGET")
    need(plan.get("model") == recipe.MODEL
         and plan.get("preflight_schema") == "english-kws-dual-track-v7-kokoro-source-preflight-v1",
         "EXACT_KOKORO_MODEL_AND_PREFLIGHT_PINS")
    need(plan.get("minimum_pcm_samples") == 6400
         and plan.get("maximum_pcm_samples") == 28672
         and plan.get("native_sample_rate") == 24000
         and plan.get("output_sample_rate") == 16000,
         "KOKORO_AUDIO_GEOMETRY")
    need(plan.get("calibration_audio_generated") is False
         and plan.get("locked_test_audio_generated") is False
         and plan.get("calibration_audio_opened") is False
         and plan.get("locked_test_audio_opened") is False
         and plan.get("training_allowed") is False
         and plan.get("promotion_allowed") is False,
         "NO_CALIBRATION_OR_LOCKED_TEST_AUDIO_AND_NO_TRAINING")
    need(plan.get("execution") == {
        "event": "push",
        "ref": "refs/heads/main",
        "run_attempt": 1,
        "retries": 0,
        "main_successor_only": True,
        "no_manual_dispatch": True,
    }, "EXACT_ONE_SHOT_MAIN_PUSH_EXECUTION")
    computed = {
        "worker_sha256": sha(regular(Path(__file__))),
        "recipe_code_sha256": sha(regular(HERE / "english_kws_dual_track_v7_recipe.py")),
        "preflight_script_sha256": sha(regular(HERE / "english_kws_dual_track_v7_preflight.py")),
    }
    if check_sources:
        for key, value in computed.items():
            expected = plan.get(key)
            need(expected == value or (allow_autofill and expected == "AUTO"),
                 "PINNED_SOURCE_CODE_HASH:" + key)
        helpers = plan.get("helper_sha256")
        need(isinstance(helpers, dict) and 5 <= len(helpers) <= 12,
             "PINNED_OBSERVER_HELPER_INVENTORY")
        for name, expected in helpers.items():
            path = HERE / name
            need(path.is_file() and not path.is_symlink()
                 and sha(regular(path)) == expected,
                 "PINNED_OBSERVER_HELPER_SHA256:" + name)
    return {
        "plan": plan,
        "source_plan": source_plan,
        "registration": registration,
        "recipe": frozen,
        "source_plan_sha256": source_plan_sha,
        "registration_sha256": registration_sha,
        "computed_source_hashes": computed,
        "recipe_sha256": recipe.digest(frozen),
    }


def execution_guard(plan, env=None, event=None):
    env = os.environ if env is None else env
    if event is None:
        event = read(Path(env["GITHUB_EVENT_PATH"]))
    validate_plan(plan)
    need(plan["enabled"] and env.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == "refs/heads/main" and
         env.get("GITHUB_RUN_ATTEMPT") == "1", "PROTECTED_FIRST_MAIN_ONLY")
    need(type(event.get("repository")) is dict and event["repository"].get("private") is False,
         "PUBLIC_REPOSITORY_ONLY")
    need(event.get("before") == plan["next_main_before"] and event.get("after") == env.get("GITHUB_SHA") and
         type(env.get("GITHUB_SHA")) is str and HEX40.fullmatch(env["GITHUB_SHA"]), "EXACT_PUSH_IDENTITY")
    return identity(env)


class Deadline(BaseException):
    pass


def set_deadline(seconds):
    def stop(_signum, _frame):
        raise Deadline("FINITE_STAGE_DEADLINE")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)


def verify_preflight(preflight_dir, plan, run):
    preflight_dir = Path(preflight_dir)
    report_path = preflight_dir / "source-license-preflight.json"
    report_raw = regular(report_path, 1024 * 1024)
    report = read_json(report_path)
    frozen = recipe.recipe()
    need(report.get("schema") == "english-kws-dual-track-v7-kokoro-source-preflight-v1"
         and report.get("status") == "PASS"
         and report.get("run") == run
         and report.get("source_plan_sha256") == plan["source_plan_sha256"]
         and report.get("source_plan_blob_sha1") == recipe.SOURCE_PLAN_BLOB_SHA1
         and report.get("registration_blob_sha1") == recipe.REGISTRATION_BLOB_SHA1,
         "PINNED_KOKORO_PREFLIGHT_AND_CURRENT_RUN")
    expected_no_execution = {
        "model_weights_downloaded": False,
        "voice_weights_downloaded": False,
        "audio_downloaded": False,
        "tts_calls": 0,
        "asr_calls": 0,
        "phoneme_observer_calls": 0,
        "kws_forward_calls": 0,
        "optimizer_updates": 0,
        "training_allowed": False,
    }
    need(report.get("no_execution") == expected_no_execution,
         "PREFLIGHT_MUST_NOT_DOWNLOAD_WEIGHTS_OR_GENERATE_AUDIO")
    expected_voices = {
        str(group["source_speaker_index"]): group["source_speaker_id"]
        for group in frozen["source_groups"]
    }
    source_map = report.get("source_map", {})
    need(source_map.get("selected_voice_ids") == expected_voices
         and source_map.get("selected_count") == 8
         and source_map.get("role_counts") == {"train": 6, "development": 2}
         and source_map.get("voice_id_unique") is True
         and source_map.get("human_speaker_identity_claimed") is False,
         "EXACT_KOKORO_VOICE_SOURCE_MAP")
    phoneme_map = report.get("phoneme_map", {})
    need(phoneme_map.get("missing_characters") == []
         and phoneme_map.get("covered_count") == len(phoneme_map.get("required_characters", [])),
         "KOKORO_RAW_IPA_VOCAB_COVERAGE")
    model = report.get("model", {})
    need(model.get("repository") == recipe.MODEL["repository"]
         and model.get("revision") == recipe.MODEL["revision"]
         and model.get("checkpoint_path") == recipe.MODEL["checkpoint_path"]
         and model.get("expected_checkpoint_sha256") == recipe.MODEL["checkpoint_sha256"]
         and model.get("native_sample_rate") == 24000
         and model.get("output_sample_rate") == 16000
         and model.get("license") == "Apache-2.0",
         "EXACT_KOKORO_MODEL_REVISION_AND_LICENSE")
    license_row = report.get("license", {})
    need(license_row.get("model_license") == "Apache-2.0"
         and license_row.get("inference_code_license") == "Apache-2.0"
         and license_row.get("inference_code_commit") == recipe.MODEL["source_commit"]
         and license_row.get("generated_audio_derivative_clearance") ==
             "NOT_ASSERTED_LEGAL_REVIEW_REQUIRED_BEFORE_COMMERCIAL_PRODUCT_CLAIM",
         "KOKORO_LICENSE_AND_NO_COMMERCIAL_CLEARANCE_CLAIM")
    files = {
        "config.json": model.get("config_sha256"),
        "VOICES.md": model.get("voices_index_sha256"),
        "README.md": model.get("model_card_sha256"),
        "KOKORO_LICENSE": license_row.get("inference_code_license_sha256"),
    }
    for name, expected in files.items():
        raw = regular(preflight_dir / name, 4 * 1024 * 1024)
        need(isinstance(expected, str) and sha(raw) == expected,
             "KOKORO_PREFLIGHT_FILE_SHA256:" + name)
    config = read_json(preflight_dir / "config.json")
    vocab = config.get("vocab")
    need(isinstance(vocab, dict) and vocab, "KOKORO_CONFIG_VOCAB")
    selected_plan, _ = recipe.source_documents()
    required_chars = {char for cell in selected_plan["cells"] for char in cell["raw_ipa"]}
    need(required_chars <= set(vocab), "KOKORO_RUNTIME_VOCAB_RECHECK")
    return report


def source_model(folder, preflight_dir):
    from importlib.metadata import version
    from huggingface_hub import hf_hub_download
    need(version("piper-tts") == recipe.MODEL["piper_version"], "PIPER_VERSION")
    preflight_dir = Path(preflight_dir)
    config_path = preflight_dir / "piper-config.json"
    cb = regular(config_path, 1024 * 1024)
    need(len(cb) == recipe.MODEL["config_bytes"] and hashlib.md5(cb).hexdigest() == recipe.MODEL["config_md5"],
         "PIPER_CONFIG_BYTES")
    config = loads(cb)
    need(config.get("num_speakers") == 904 and config.get("audio", {}).get("sample_rate") == 22050 and
         config.get("espeak", {}).get("voice") == "en-us", "PIPER_GEOMETRY")
    need(all(config["speaker_id_map"].get(f"p{speaker}") == index for index, speaker in recipe.NEW_VOICES),
         "EXACT_NEW_SPEAKER_MAP")
    required_chars = {char for row in recipe.recipe()["rows"] for char in row["raw_phonemes"]}
    need(required_chars.issubset(config.get("phoneme_id_map", {})), "DIRECT_PHONEME_MAP_COVERAGE")
    card = regular(preflight_dir / "MODEL_CARD", 4096)
    need(len(card) == recipe.MODEL_CARD["bytes"] and hashlib.md5(card).hexdigest() == recipe.MODEL_CARD["md5"],
         "MODEL_CARD_PIN")
    paths = Path(hf_hub_download(repo_id=recipe.MODEL["repository"],
        filename=recipe.MODEL["model_path"], revision=recipe.MODEL["revision"],
        local_dir=str(folder), token=False))
    mb = regular(paths, 150 * 1024**2)
    need(len(mb) == recipe.MODEL["model_bytes"] and sha(mb) == recipe.MODEL["model_sha256"],
         "PIPER_MODEL_BYTES")
    from piper import PiperVoice
    voice = PiperVoice.load(str(paths), config_path=str(config_path), use_cuda=False)
    need(callable(voice.phonemes_to_ids) and callable(voice.phoneme_ids_to_audio), "DIRECT_PHONEME_API")
    import piper.voice
    api_bytes = regular(Path(piper.voice.__file__), 128 * 1024)
    engine = read(preflight_dir / "source-license-preflight.json")["license"]["piper_engine"]
    lock = dict(recipe.MODEL) | {
        "config_sha256": sha(cb), "model_card_sha256": sha(card),
        "engine_setup_sha256": engine["setup_sha256"], "engine_copying_sha256": engine["copying_sha256"],
        "source_speaker_map": {f"p{s}": i for i, s in recipe.NEW_VOICES},
        "installed_voice_api_sha256": sha(api_bytes), "model_bytes_verified": True,
        "direct_api": "PiperVoice.phonemes_to_ids -> PiperVoice.phoneme_ids_to_audio",
    }
    return voice, lock, api_bytes

def request_phonemes(voice, row):
    if row["generation_mode"] == "direct-phoneme-ids":
        phones = list(row["raw_phonemes"])
    else:
        sentences = voice.phonemize(row["requested_text"])
        need(type(sentences) is list and len(sentences) == 1 and type(sentences[0]) is list, "ONE_G2P_SENTENCE")
        phones = sentences[0]
    need(0 < len(phones) <= 256 and all(type(x) is str and x in voice.config.phoneme_id_map for x in phones), "KNOWN_PHONEMES_ONLY")
    ids = voice.phonemes_to_ids(phones)
    need(type(ids) is list and 0 < len(ids) <= 1024 and all(type(i) is int and 0 <= i < 130 for i in ids), "DIRECT_PHONE_IDS")
    return phones, ids


def write_audio(folder, audio):
    import numpy as np
    from scipy.signal import resample_poly
    need(isinstance(audio, np.ndarray) and audio.ndim == 1 and
         552 <= len(audio) <= 264600 and bool(np.isfinite(audio).all()), "NATIVE_AUDIO_GEOMETRY")
    x = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    need(0.003 <= rms <= 0.6 and 0.01 <= peak <= 1.1, "NATIVE_AUDIO_LEVEL")
    native = np.rint(np.clip(x, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes()
    y = resample_poly(np.frombuffer(native, dtype="<i2").astype(np.float32) / 32768.0, 320, 441)
    need(400 <= len(y) <= 192000 and bool(np.isfinite(y).all()), "DERIVED_AUDIO_GEOMETRY")
    pcm = np.rint(np.clip(y, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes()
    for name, raw, rate in (("native-22050.wav", native, 22050), ("audio-16k.wav", pcm, 16000)):
        with (folder / name).open("xb") as output:
            with wave.open(output, "wb") as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate); stream.writeframes(raw)
    with (folder / "audio-16k.pcm").open("xb") as output:
        output.write(pcm)
    return {"pcm_sha256": sha(pcm), "pcm_samples": len(pcm) // 2,
        "wav_sha256": sha(regular(folder / "audio-16k.wav")),
        "native_wav_sha256": sha(regular(folder / "native-22050.wav")),
        "native_samples": len(native) // 2, "sample_rate": 16000, "channels": 1,
        "sample_width": 2, "native_rms": rms, "native_peak": peak,
        "resampler": "scipy.signal.resample_poly(320,441)", "synthetic_silence_added": False}


def generate(plan, output, preflight_dir):
    run = execution_guard(plan)
    verify_preflight(preflight_dir, plan, run)
    output = new_directory(output)
    frozen = recipe.recipe()
    once(output / "RECIPE.json", frozen)
    once(output / "EXECUTION_PLAN.json", plan)
    copy_closed_tree(preflight_dir, output / "preflight", 2 * 1024**2)
    once(output / "STARTED.json", {"schema": SCHEMA, "stage": "generate", "run": run,
        "maximum_calls": recipe.SOURCE_COUNT, "retry_count": 0, "model_calls_at_start": 0,
        "preflight_sha256": sha(regular(output / "preflight" / "source-license-preflight.json"))})
    model_root = output / "private-model"
    model_root.mkdir(mode=0o700)
    calls = 0
    completed = 0
    error = None
    start = time.monotonic()
    set_deadline(plan["generation_deadline_seconds"])
    try:
        voice, lock, api_bytes = source_model(model_root, output / "preflight")
        once(output / "MODEL_LOCK.json", lock)
        (output / "installed-piper-voice.py").write_bytes(api_bytes)
        from piper import SynthesisConfig
        import onnxruntime
        onnxruntime.set_seed(20261127)
        for row in frozen["rows"]:
            folder = output / "audio" / row["id"]
            folder.mkdir(parents=True, mode=0o700)
            try:
                phones, ids = request_phonemes(voice, row)
                once(folder / "STARTED.json", {"schema": SCHEMA, "stage": "generate", "id": row["id"],
                    "request_sha256": row["request_sha256"], "attempt": 1, "phonemes": phones,
                    "phoneme_ids": ids, "speaker_index": row["source_speaker_index"],
                    "source_group_id": row["source_group_id"], "role": row["role"], "cell_id": row["cell_id"],
                    "length_scale": row["length_scale"], "direct_api": True, "raw_bracket_parser_used": False})
                need(calls < recipe.SOURCE_COUNT, "SOURCE_CALL_CEILING")
                calls += 1
                audio = voice.phoneme_ids_to_audio(ids, SynthesisConfig(
                    speaker_id=row["source_speaker_index"], length_scale=row["length_scale"],
                    noise_scale=0.333, noise_w_scale=0.333, normalize_audio=False, volume=1.0))
                measured = write_audio(folder, audio)
                once(folder / "COMPLETED.json", {"schema": SCHEMA, "id": row["id"], "status": "COMPLETE",
                    "request_sha256": row["request_sha256"], "generation_model_sha256": recipe.MODEL["model_sha256"],
                    "model_card_sha256": lock["model_card_sha256"], "phonemes": phones, "phoneme_ids": ids,
                    "source_group_id": row["source_group_id"], "role": row["role"], "cell_id": row["cell_id"],
                    "length_scale": row["length_scale"], "observed_target_pronunciation": False,
                    "human_gold": False, **measured})
                completed += 1
            except BaseException as exc:
                once(folder / "FAILED.json", {"schema": SCHEMA, "id": row["id"], "status": "FAILED_NO_RETRY",
                    "error_type": type(exc).__name__, "started_receipt_exists": (folder / "STARTED.json").is_file()})
                if not isinstance(exc, Exception):
                    raise
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        signal.alarm(0)
        if model_root.exists():
            shutil.rmtree(model_root)
        result = {"schema": SCHEMA, "stage": "generate", "run": run,
            "status": "COMPLETE" if completed == recipe.SOURCE_COUNT and error is None else "INCOMPLETE_NO_RETRY",
            "declared_count": recipe.SOURCE_COUNT, "actual_forward_attempts": calls, "completed": completed,
            "error_type": error, "elapsed_seconds": time.monotonic() - start,
            "retry_count": 0, "old_audio_model_calls": 0, "kws_forward_calls": 0,
            "preflight_sha256": sha(regular(output / "preflight" / "source-license-preflight.json"))}
        once(output / "RESULT.json", result)
    return result

def verify_wav(path, row):
    raw = regular(path)
    need(sha(raw) == row["wav_sha256"], "WAV_HASH")
    with wave.open(io.BytesIO(raw), "rb") as stream:
        n = stream.getnframes()
        need((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) ==
             (1, 2, 16000, "NONE") and n == row["pcm_samples"] and 400 <= n <= 192000, "WAV_GEOMETRY")
        pcm = stream.readframes(n)
    need(len(pcm) == n * 2 and sha(pcm) == row["pcm_sha256"], "PCM_HASH")
    return pcm


def source_readback(source, plan):
    source = Path(source)
    frozen = recipe.recipe()
    need(read(source / "RECIPE.json") == frozen and read(source / "EXECUTION_PLAN.json") == plan,
         "SOURCE_RECIPE_OR_PLAN")
    start = read(source / "STARTED.json")
    need(start.get("schema") == SCHEMA and start.get("stage") == "generate" and
         start.get("maximum_calls") == recipe.SOURCE_COUNT and start.get("retry_count") == 0,
         "SOURCE_STARTED")
    preflight = verify_preflight(source / "preflight", plan, start["run"])
    need(start.get("preflight_sha256") == sha(regular(source / "preflight" / "source-license-preflight.json")),
         "SOURCE_PREFLIGHT_BINDING")
    lock_path = source / "MODEL_LOCK.json"
    if lock_path.is_file():
        lock = read(lock_path)
        need(all(lock.get(k) == v for k, v in recipe.MODEL.items()), "SOURCE_MODEL_LOCK")
        cb = regular(source / "preflight" / "piper-config.json", 1024 * 1024)
        need(sha(cb) == lock["config_sha256"] and hashlib.md5(cb).hexdigest() == recipe.MODEL["config_md5"],
             "SOURCE_CONFIG_PIN")
        need(sha(regular(source / "preflight" / "MODEL_CARD", 4096)) == lock["model_card_sha256"],
             "SOURCE_MODEL_CARD_PIN")
        need(sha(regular(source / "preflight" / "PIPER_SETUP.py", 128 * 1024)) == lock["engine_setup_sha256"] and
             sha(regular(source / "preflight" / "COPYING", 128 * 1024)) == lock["engine_copying_sha256"],
             "SOURCE_ENGINE_LICENSE_PIN")
        need(sha(regular(source / "installed-piper-voice.py", 128 * 1024)) == lock["installed_voice_api_sha256"],
             "PIPER_API_CAPTURE")
    rows = []
    for expected in frozen["rows"]:
        row = dict(expected)
        folder = source / "audio" / row["id"]
        done = folder / "COMPLETED.json"
        begun = folder / "STARTED.json"
        failed = folder / "FAILED.json"
        row.update(generation_status="NOT_STARTED", pcm_path=None, pcm_sha256=None, pcm_samples=None,
            wav_path=None, wav_sha256=None, sample_rate=16000, channels=1)
        if done.is_file():
            need(begun.is_file() and not failed.exists() and lock_path.is_file(), "SOURCE_TERMINAL_CLOSURE")
            receipt = read(done)
            receipt_start = read(begun)
            need(receipt.get("status") == "COMPLETE" and receipt.get("id") == row["id"] and
                 receipt.get("request_sha256") == row["request_sha256"] and
                 receipt.get("generation_model_sha256") == recipe.MODEL["model_sha256"] and
                 receipt.get("model_card_sha256") == lock["model_card_sha256"] and
                 receipt.get("human_gold") is False, "SOURCE_COMPLETE_BINDING")
            need(receipt_start.get("id") == row["id"] and receipt_start.get("request_sha256") == row["request_sha256"] and
                 receipt_start.get("attempt") == 1 and receipt_start.get("direct_api") is True and
                 receipt_start.get("raw_bracket_parser_used") is False and
                 receipt_start.get("phonemes") == receipt.get("phonemes") and
                 receipt_start.get("phoneme_ids") == receipt.get("phoneme_ids") and
                 receipt_start.get("speaker_index") == row["source_speaker_index"] and
                 receipt_start.get("source_group_id") == row["source_group_id"] and
                 receipt_start.get("role") == row["role"] and receipt_start.get("cell_id") == row["cell_id"] and
                 receipt_start.get("length_scale") == row["length_scale"], "SOURCE_DIRECT_API_RECEIPT")
            row.update({k: receipt[k] for k in ("pcm_sha256", "pcm_samples", "wav_sha256")})
            pcm = verify_wav(folder / "audio-16k.wav", row)
            need(regular(folder / "audio-16k.pcm") == pcm, "RAW_PCM_COPY")
            need(sha(regular(folder / "native-22050.wav")) == receipt["native_wav_sha256"], "NATIVE_WAV_HASH")
            source_plan, _registration = recipe.source_documents()
            audio_policy = source_plan["audio_policy"]
            minimum_model_samples = plan["positive_min_samples"]
            maximum_model_samples = audio_policy["positive_geometry_max_samples"]
            if not minimum_model_samples <= row["pcm_samples"] <= maximum_model_samples:
                row.update(generation_status="MODEL_INPUT_GEOMETRY_QUARANTINED",
                    geometry_reason="REQUIRES_AT_LEAST_6400_PCM16_SAMPLES_FOR_35_CAUSAL_CTC_FRAMES_AND_MAX_28672")
            else:
                row.update(generation_status="success")
            row.update(pcm_path=f"source/audio/{row['id']}/audio-16k.pcm",
                wav_path=f"source/audio/{row['id']}/audio-16k.wav", actual_phonemes=receipt["phonemes"],
                actual_phoneme_ids=receipt["phoneme_ids"], source_receipt_sha256=sha(regular(done)))
        elif failed.is_file():
            need(read(failed).get("id") == row["id"], "FAILED_SOURCE_ID")
            row["generation_status"] = "FAILED_NO_RETRY"
        elif begun.is_file():
            row["generation_status"] = "UNKNOWN_AFTER_START_NO_RETRY"
        rows.append(row)
    pcm_hashes = [row["pcm_sha256"] for row in rows if row["generation_status"] == "success"]
    duplicates = {digest for digest in pcm_hashes if pcm_hashes.count(digest) > 1}
    for row in rows:
        if row["pcm_sha256"] in duplicates:
            row["generation_status"] = "DUPLICATE_PCM_QUARANTINED"
    corpus = {"schema": "english-kws-dual-track-v7-corpus-v1", "run": start["run"],
        "recipe_sha256": recipe.digest(frozen), "declared_count": recipe.SOURCE_COUNT,
        "valid_count": sum(row["generation_status"] == "success" for row in rows),
        "source_groups": recipe.source_groups(), "rows": rows, "human_gold": False,
        "evaluation_scope": "FRESH_SYNTHETIC_SOURCE_GROUPS_WITH_SHARED_PIPER_PRETRAINING",
        "independent_human_speaker_claim": False, "duplicate_pcm_count": len(duplicates),
        "preflight_sha256": sha(regular(source / "preflight" / "source-license-preflight.json")),
        "license_evidence": preflight["license"], "kws_forward_calls": 0, "optimizer_updates": 0}
    return corpus

def observe(provider, source, plan, output):
    run = execution_guard(plan)
    need(provider in recipe.PROVIDERS, "KNOWN_PROVIDER")
    corpus = source_readback(source, plan)
    need(corpus["run"] == run, "SAME_RUN_SOURCE")
    rows = [row for row in corpus["rows"] if row["generation_status"] == "success"]
    output = new_directory(output)
    once(output / "STARTED.json", {"schema": SCHEMA, "stage": "observe", "run": run,
        "provider": provider, "source_corpus_sha256": recipe.digest(corpus),
        "valid_source_count": len(rows), "maximum_calls": recipe.SOURCE_COUNT, "retry_count": 0,
        "observer_identity": recipe.observer_identity(provider),
        "target_conditioned": False, "requested_text_passed": False, "role_passed": False})
    model_root = output / "private-model"
    model_root.mkdir(mode=0o700)
    calls = 0
    completed = 0
    error = None
    start = time.monotonic()
    set_deadline(plan["observation_deadline_seconds"])
    try:
        if rows:
            if provider == "phoneme":
                import english_final_ipa_xlsr_once as xlsr
                decode, actual_lock = xlsr.load_runtime(model_root)
            else:
                import english_seven_voice_asr_once as seven
                decode = seven._load(provider, model_root)
                actual_lock = {"verified_pinned_loader": "english_seven_voice_asr_once._load",
                    "provider": provider, "source_sha256": plan["helper_sha256"]["automation/english_seven_voice_asr_once.py"]}
            once(output / "MODEL_LOCK.json", {"identity": recipe.observer_identity(provider),
                "model": recipe.OBSERVERS[provider], "actual_runtime_lock": actual_lock})
        for row in rows:
            folder = output / row["id"]
            folder.mkdir(mode=0o700)
            source_wav = Path(source) / "audio" / row["id"] / "audio-16k.wav"
            verify_wav(source_wav, row)
            once(folder / "STARTED.json", {"schema": SCHEMA, "provider": provider, "id": row["id"],
                "source_pcm_sha256": row["pcm_sha256"], "source_wav_sha256": row["wav_sha256"],
                "source_samples": row["pcm_samples"], "attempt": 1, "target_conditioned": False,
                "requested_text_passed": False, "role_passed": False})
            need(calls < recipe.SOURCE_COUNT, "OBSERVER_CALL_CEILING")
            calls += 1
            try:
                capture = recipe.observer_identity(provider) | {"schema": "english-kws-dual-track-v7-native-capture-v1",
                    "id": row["id"], "source_pcm_sha256": row["pcm_sha256"],
                    "source_wav_sha256": row["wav_sha256"], "source_samples": row["pcm_samples"],
                    "status": "success", "target_conditioned": False,
                    "input_coverage_complete": True, "decoder_output_complete": True}
                if provider == "phoneme":
                    tokens, frames = decode(source_wav)
                    need(frames == row["pcm_samples"] and type(tokens) is list and 1 <= len(tokens) <= 300 and
                         all(type(token) is str and 0 < len(token) <= 40 for token in tokens), "PHONE_OUTPUT_BINDING")
                    capture["phone_tokens"] = tokens
                else:
                    text = decode(source_wav, folder)
                    need(type(text) is str and 0 < len(text.strip()) <= 512 and
                         not any(ord(char) < 32 for char in text.strip()), "ASR_OUTPUT_TEXT")
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
            "declared_count": recipe.SOURCE_COUNT, "valid_source_count": len(rows),
            "actual_forward_attempts": calls, "completed": completed, "error_type": error,
            "elapsed_seconds": time.monotonic() - start, "retry_count": 0, "old_audio_model_calls": 0,
            "target_conditioned": False}
        once(output / "RESULT.json", result)
    return result

def copy_closed_tree(source, destination, maximum=300 * 1024**2):
    source = Path(source); destination = Path(destination)
    need(source.is_dir() and not source.is_symlink() and not destination.exists(), "COPY_TREE_GEOMETRY")
    files = []
    total = 0
    for p in source.rglob("*"):
        need(not p.is_symlink(), "COPY_TREE_SYMLINK")
        if p.is_file():
            total += p.stat().st_size
            need(total <= maximum, "COPY_TREE_BYTE_LIMIT")
            files.append(p)
    destination.mkdir(mode=0o700)
    for p in files:
        target = destination / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with p.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)


def capture_evidence(root, corpus):
    rows = []
    for row in corpus["rows"]:
        result = {"id": row["id"], "observations": [], "phone_observation": None}
        for provider in recipe.PROVIDERS:
            path = Path(root) / "evidence" / provider / row["id"] / "CAPTURE.json"
            if not path.is_file():
                continue
            capture = read(path)
            need(row["generation_status"] == "success", "CAPTURE_REQUIRES_VALID_SOURCE")
            need(capture.get("id") == row["id"] and
                 all(capture.get(key) == value for key, value in recipe.observer_identity(provider).items()),
                 "CAPTURE_MODEL_BINDING")
            need(capture.get("source_pcm_sha256") == row["pcm_sha256"] and
                 capture.get("source_wav_sha256") == row["wav_sha256"] and
                 capture.get("source_samples") == row["pcm_samples"], "CAPTURE_SOURCE_BINDING")
            observation = capture | {"capture_path": path.relative_to(root).as_posix(),
                "capture_sha256": sha(regular(path))}
            if provider == "phoneme":
                result["phone_observation"] = observation
            else:
                result["observations"].append(observation)
        rows.append(result)
    need(len({row["id"] for row in corpus["rows"]}) == recipe.SOURCE_COUNT, "COMPLETE_EVIDENCE_DENOMINATOR")
    return {"schema": "english-kws-dual-track-v7-machine-evidence-v1", "rows": rows,
        "human_gold": False, "new_model_calls_during_join": 0, "optimizer_updates": 0}

def combine(source, observers, plan, output):
    run = execution_guard(plan)
    corpus = source_readback(source, plan)
    need(corpus["run"] == run, "SAME_RUN_CORPUS")
    output = new_directory(output)
    copy_closed_tree(source, output / "source")
    (output / "evidence").mkdir(mode=0o700)
    observer_statuses = {}
    for provider in recipe.PROVIDERS:
        path = Path(observers) / ("v7-observer-" + provider)
        if path.is_dir():
            started_path = path / "STARTED.json"
            if started_path.is_file():
                started = read(started_path)
                need(started.get("run") == run and started.get("provider") == provider and
                     started.get("source_corpus_sha256") == recipe.digest(corpus) and
                     started.get("observer_identity") == recipe.observer_identity(provider), "OBSERVER_HANDOFF_IDENTITY")
                copy_closed_tree(path, output / "evidence" / provider, 32 * 1024**2)
                result_path = output / "evidence" / provider / "RESULT.json"
                observer_statuses[provider] = read(result_path).get("status") if result_path.is_file() else "MISSING_TERMINAL_RESULT"
            else:
                copy_closed_tree(path, output / "evidence" / provider, 32 * 1024**2)
                observer_statuses[provider] = "FAILED_BEFORE_START"
        else:
            observer_statuses[provider] = "MISSING_ARTIFACT"
    once(output / "CORPUS_MANIFEST.json", corpus)
    evidence = capture_evidence(output, corpus)
    once(output / "MACHINE_EVIDENCE.json", evidence)
    once(output / "EXECUTION_PLAN.json", plan)
    all_observers_complete = all(status == "COMPLETE" for status in observer_statuses.values())
    source_result_path = Path(source) / "RESULT.json"
    source_status = read(source_result_path).get("status") if source_result_path.is_file() else "UNKNOWN_WITHOUT_TERMINAL_RESULT"
    complete = corpus["valid_count"] == recipe.SOURCE_COUNT and source_status == "COMPLETE" and all_observers_complete
    result = {"schema": SCHEMA, "stage": "combine", "run": run,
        "status": "MACHINE_EVIDENCE_PACKET_RETAINED" if complete else "INCOMPLETE_NO_RETRY",
        "declared_count": recipe.SOURCE_COUNT, "valid_sources": corpus["valid_count"],
        "observer_statuses": observer_statuses, "source_status": source_status,
        "retry_count": 0, "new_model_calls": 0, "kws_forward_calls": 0,
        "optimizer_updates": 0, "training_allowed": False, "promotion_allowed": False,
        "human_gold": False, "physical_ssc305_execution": False,
        "preflight_sha256": corpus["preflight_sha256"]}
    once(output / "RESULT.json", result)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            raw = regular(path, MAX_FILE)
            files.append({"path": path.relative_to(output).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    once(output / "FILES.json", {"schema": "english-kws-dual-track-v7-files-v1", "files": files})
    return result
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--source-check", action="store_true")
    actions.add_argument("--guard", action="store_true")
    actions.add_argument("--generate", action="store_true")
    actions.add_argument("--read-source", action="store_true")
    actions.add_argument("--observe", choices=recipe.PROVIDERS)
    actions.add_argument("--combine", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--observers", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    validate_plan(plan, allow_autofill=args.source_check)
    if args.source_check:
        result = {"status": "SOURCE_ONLY_NO_MODEL", "enabled": plan["enabled"],
            "declared_sources": recipe.SOURCE_COUNT, "source_groups": len(recipe.source_groups()),
            "role_counts": {role: sum(group["role"] == role for group in recipe.source_groups()) for role in recipe.ROLES},
            "worker_sha256": sha(regular(Path(__file__), 256 * 1024)),
            "recipe_code_sha256": sha(regular(Path(recipe.__file__), 128 * 1024)),
            "preflight_script_sha256": sha(regular(ROOT / "automation/english_kws_dual_track_v7_preflight.py", 128 * 1024)),
            "recipe_sha256": recipe.digest(recipe.recipe()),
            "source_plan_sha256": recipe.recipe()["source_plan_sha256"],
            "source_plan_blob_sha1": recipe.SOURCE_PLAN_BLOB_SHA1,
            "registration_blob_sha1": recipe.REGISTRATION_BLOB_SHA1,
            "maximum_tts_calls": 144, "maximum_calls_per_observer": 144,
            "maximum_total_model_calls": 720, "retry_count": 0,
            "training_allowed": False, "promotion_allowed": False}
    elif args.guard:
        result = execution_guard(plan)
    elif args.generate:
        need(args.output is not None and args.preflight_dir is not None, "GENERATE_INPUTS_REQUIRED")
        result = generate(plan, args.output, args.preflight_dir)
    elif args.read_source:
        need(args.source is not None, "SOURCE_REQUIRED")
        corpus = source_readback(args.source, plan)
        if args.output is not None:
            once(args.output, corpus)
        result = {"status": "SOURCE_READBACK", "valid_count": corpus["valid_count"],
            "corpus_sha256": recipe.digest(corpus)}
    elif args.observe:
        need(args.source is not None and args.output is not None, "OBSERVER_INPUTS_REQUIRED")
        result = observe(args.observe, args.source, plan, args.output)
    else:
        need(all(value is not None for value in (args.source, args.observers, args.output)),
             "COMBINE_INPUTS_REQUIRED")
        result = combine(args.source, args.observers, plan, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result.get("status") == "INCOMPLETE_NO_RETRY" else 0


if __name__ == "__main__":
    raise SystemExit(main())

