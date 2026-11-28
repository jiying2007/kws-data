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
        need(isinstance(helpers, dict) and 5 <= len(helpers) <= 20,
             "PINNED_OBSERVER_HELPER_INVENTORY")
        for name, expected in helpers.items():
            relative = Path(name)
            path = (HERE.parent / relative) if relative.parts and relative.parts[0] == "automation" else (HERE / relative)
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
    raw_vocab = config.get("vocab")
    need(isinstance(raw_vocab, dict) and raw_vocab, "KOKORO_CONFIG_VOCAB")
    vocab = {phoneme: int(token_id) for token_id, phoneme in raw_vocab.items()}
    selected_plan, _ = recipe.source_documents()
    required_chars = {char for cell in selected_plan["cells"] for char in cell["raw_ipa"]}
    need(required_chars <= set(vocab), "KOKORO_RUNTIME_VOCAB_RECHECK")
    return report


def source_model(folder, preflight_dir, run):
    import importlib.metadata
    import importlib.machinery
    import sys
    import types
    import torch
    from huggingface_hub import hf_hub_download

    folder = Path(folder)
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_dir = folder / "hf-cache"
    cache_dir.mkdir(mode=0o700, exist_ok=True)
    model_config = recipe.MODEL
    repo_id = model_config["repository"]
    revision = model_config["revision"]

    config_path = Path(hf_hub_download(
        repo_id=repo_id, filename=model_config["config_path"],
        revision=revision, cache_dir=cache_dir, token=False))
    model_path = Path(hf_hub_download(
        repo_id=repo_id, filename=model_config["checkpoint_path"],
        revision=revision, cache_dir=cache_dir, token=False))
    config_raw = regular(config_path, 4 * 1024 * 1024)
    model_raw = regular(model_path, 1024 * 1024 * 1024)
    config_sha = sha(config_raw)
    model_sha = sha(model_raw)
    need(model_sha == model_config["checkpoint_sha256"],
         "PINNED_KOKORO_CHECKPOINT_SHA256")
    preflight = read_json(Path(preflight_dir) / "source-license-preflight.json")
    need(config_sha == preflight["model"]["config_sha256"],
         "KOKORO_CONFIG_PREFLIGHT_RUNTIME_SHA256")

    # Import KModel without executing kokoro.__init__ / KPipeline / Misaki G2P.
    # The model receives the frozen IPA string directly; no eSpeak or text G2P
    # dependency is installed or invoked.
    package_dir = Path(importlib.metadata.distribution("kokoro").locate_file("kokoro"))
    need(package_dir.is_dir() and (package_dir / "model.py").is_file(),
         "KOKORO_DIRECT_MODEL_PACKAGE")
    sys.modules.pop("kokoro", None)
    package = types.ModuleType("kokoro")
    package.__path__ = [package_dir.as_posix()]
    package.__package__ = "kokoro"
    package_spec = importlib.machinery.ModuleSpec("kokoro", loader=None, is_package=True)
    package_spec.submodule_search_locations = [package_dir.as_posix()]
    package.__spec__ = package_spec
    sys.modules["kokoro"] = package
    from kokoro.model import KModel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = KModel(repo_id=repo_id, config=config_path.as_posix(),
                   model=model_path.as_posix(), device="cpu")
    model.eval()
    vocab = model.vocab
    selected_plan, _ = recipe.source_documents()
    required_chars = sorted({char for cell in selected_plan["cells"] for char in cell["raw_ipa"]})
    missing = [char for char in required_chars if char not in vocab]
    need(not missing, "KOKORO_RUNTIME_RAW_IPA_VOCAB:" + repr(missing))
    vocab_subset = {char: int(vocab[char]) for char in required_chars}

    voice_styles = {}
    voice_packs = {}
    for group in recipe.source_groups():
        voice_id = group["source_speaker_id"]
        voice_path = Path(hf_hub_download(
            repo_id=repo_id, filename="voices/" + voice_id + ".pt",
            revision=revision, cache_dir=cache_dir, token=False))
        voice_raw = regular(voice_path, 8 * 1024 * 1024)
        style = torch.load(voice_path, map_location="cpu", weights_only=True)
        need(torch.is_tensor(style) and tuple(style.shape) == (1, 256)
             and style.dtype == torch.float32 and bool(torch.isfinite(style).all()),
             "KOKORO_VOICE_STYLE_GEOMETRY:" + voice_id)
        voice_styles[voice_id] = style.contiguous()
        voice_packs[voice_id] = {
            "path": "voices/" + voice_id + ".pt",
            "bytes": len(voice_raw),
            "sha256": sha(voice_raw),
            "style_shape": [1, 256],
            "dtype": "float32",
        }

    try:
        package_version = importlib.metadata.version("kokoro")
        transformers_version = importlib.metadata.version("transformers")
        huggingface_hub_version = importlib.metadata.version("huggingface-hub")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("PINNED_KOKORO_RUNTIME_DEPENDENCIES") from exc
    model_lock = {
        "schema": "english-kws-dual-track-v7-kokoro-model-lock-v1",
        "run": run,
        "repository": repo_id,
        "revision": revision,
        "checkpoint_path": model_config["checkpoint_path"],
        "checkpoint_bytes": len(model_raw),
        "checkpoint_sha256": model_sha,
        "config_path": model_config["config_path"],
        "config_bytes": len(config_raw),
        "config_sha256": config_sha,
        "voice_packs": voice_packs,
        "inference_package": "kokoro==" + package_version,
        "inference_source_commit": model_config["source_commit"],
        "inference_code_license": "Apache-2.0",
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "huggingface_hub_version": huggingface_hub_version,
        "phoneme_vocab": vocab_subset,
        "phoneme_vocab_sha256": sha(canonical(vocab_subset)),
        "raw_ipa_mode": "direct KModel.forward; no G2P, no eSpeak, no orthographic fallback",
        "native_sample_rate": 24000,
        "output_sample_rate": 16000,
        "model_license": "Apache-2.0",
        "human_speaker_identity_claimed": False,
        "calibration_audio_generated": False,
        "locked_test_audio_generated": False,
        "training_allowed": False,
        "product_promotion_allowed": False,
    }
    return model, voice_styles, model_lock


def request_phonemes(row, model):
    phonemes = row["raw_phonemes"]
    need(isinstance(phonemes, str) and phonemes, "FROZEN_RAW_IPA_STRING")
    ids = []
    for char in phonemes:
        token_id = model.vocab.get(char)
        need(token_id is not None, "KOKORO_RAW_IPA_TOKEN:" + repr(char))
        ids.append(int(token_id))
    return phonemes, ids


def write_wav(path, pcm_bytes, sample_rate):
    import wave
    path = Path(path)
    with wave.open(path.as_posix(), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm_bytes)
    return path


def write_audio(output, row, audio):
    import numpy as np
    from scipy.signal import resample_poly

    audio = audio.detach().cpu().numpy().astype(np.float32, copy=False).squeeze()
    need(audio.ndim == 1 and 600 <= audio.size <= 288000
         and np.isfinite(audio).all(), "KOKORO_NATIVE_AUDIO_GEOMETRY")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    need(0.0005 <= rms <= 0.8 and 0.005 <= peak <= 1.0,
         "KOKORO_AUDIO_LEVEL_GATE")
    native_pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    native_bytes = native_pcm.tobytes()
    resampled = resample_poly(audio.astype(np.float64), 2, 3)
    need(resampled.size >= 6400 and resampled.size <= 28672
         and np.isfinite(resampled).all(), "KOKORO_16K_RESAMPLE_GEOMETRY")
    resampled_peak = float(np.max(np.abs(resampled)))
    need(resampled_peak <= 1.0, "KOKORO_RESAMPLED_PCM_NO_CLIPPING")
    pcm = np.rint(np.clip(resampled, -1.0, 1.0) * 32767.0).astype("<i2")
    pcm_bytes = pcm.tobytes()
    need(len(pcm_bytes) == pcm.size * 2, "KOKORO_PCM16_BYTE_GEOMETRY")

    audio_dir = Path(output) / "source" / "audio" / row["id"]
    audio_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    native_name = "native-24k.wav"
    pcm_name = "audio-16k.pcm"
    wav_name = "audio-16k.wav"
    native_path = write_wav(audio_dir / native_name, native_bytes, 24000)
    (audio_dir / pcm_name).write_bytes(pcm_bytes)
    wav_path = write_wav(audio_dir / wav_name, pcm_bytes, 16000)
    return {
        "native_wav_path": (Path("source/audio") / row["id"] / native_name).as_posix(),
        "native_wav_sha256": sha(regular(native_path)),
        "native_wav_bytes": native_path.stat().st_size,
        "native_samples": int(native_pcm.size),
        "native_sample_rate": 24000,
        "pcm_path": (Path("source/audio") / row["id"] / pcm_name).as_posix(),
        "pcm_sha256": sha(pcm_bytes),
        "pcm_bytes": len(pcm_bytes),
        "pcm_samples": int(pcm.size),
        "sample_rate": 16000,
        "channels": 1,
        "wav_path": (Path("source/audio") / row["id"] / wav_name).as_posix(),
        "wav_sha256": sha(regular(wav_path)),
        "wav_bytes": wav_path.stat().st_size,
        "native_peak": peak,
        "native_rms": rms,
        "resampled_peak": resampled_peak,
    }


def generate(plan, preflight_dir, output):
    import shutil
    import torch

    run = execution_guard(plan)
    preflight = verify_preflight(preflight_dir, plan, run)
    output = Path(output)
    new_directory(output)
    model_root = output / "private-model"
    model, voice_styles, model_lock = source_model(model_root, preflight_dir, run)
    model_lock["preflight_report_sha256"] = sha(regular(
        Path(preflight_dir) / "source-license-preflight.json", 1024 * 1024))
    model_lock["source_plan_sha256"] = plan["source_plan_sha256"]
    model_lock_sha = write_once(output / "MODEL_LOCK.json", model_lock)["sha256"]

    rows = recipe.recipe()["rows"]
    success = 0
    failed = 0
    receipts = output / "source" / "receipts"
    receipts.mkdir(mode=0o700, parents=True)
    for row in rows:
        receipt_dir = receipts / row["id"]
        receipt_dir.mkdir(mode=0o700)
        try:
            voice_id = row["source_speaker_id"]
            need(voice_id in voice_styles, "REGISTERED_KOKORO_VOICE_STYLE:" + voice_id)
            phonemes, phoneme_ids = request_phonemes(row, model)
            seed = int(row["request_sha256"][:16], 16)
            torch.manual_seed(seed)
            with torch.inference_mode():
                audio = model(phonemes, voice_styles[voice_id], float(row["speed"]))
            audio_info = write_audio(output, row, audio)
            receipt = {
                "schema": "english-kws-dual-track-v7-kokoro-source-receipt-v1",
                "run": run,
                "id": row["id"],
                "request_sha256": row["request_sha256"],
                "source_group_id": row["source_group_id"],
                "source_speaker_id": voice_id,
                "role": row["role"],
                "cell_id": row["cell_id"],
                "requested_text": row["requested_text"],
                "requested_phonemes": phonemes,
                "requested_phoneme_ids": phoneme_ids,
                "speed": row["speed"],
                "length_scale": row["length_scale"],
                "seed": seed,
                "model_revision": model_lock["revision"],
                "model_sha256": model_lock["checkpoint_sha256"],
                "config_sha256": model_lock["config_sha256"],
                "voice_pack_sha256": model_lock["voice_packs"][voice_id]["sha256"],
                "model_lock_sha256": model_lock_sha,
                "retry": False,
                "retry_allowed": False,
                "human_recording": False,
                "actual_phonemes_claimed": False,
                "calibration_audio_generated": False,
                "locked_test_audio_generated": False,
                "product_promotion_allowed": False,
                **audio_info,
            }
            write_once(receipt_dir / "COMPLETED.json", receipt)
            success += 1
        except BaseException as exc:
            audio_dir = output / "source" / "audio" / row["id"]
            if audio_dir.exists():
                shutil.rmtree(audio_dir)
            failure = {
                "schema": "english-kws-dual-track-v7-kokoro-source-failure-v1",
                "run": run,
                "id": row["id"],
                "request_sha256": row["request_sha256"],
                "source_group_id": row["source_group_id"],
                "source_speaker_id": row["source_speaker_id"],
                "role": row["role"],
                "status": "FAILED_NO_RETRY",
                "exception_type": type(exc).__name__,
                "reason": str(exc)[:520],
                "retry": False,
                "retry_allowed": False,
                "calibration_audio_generated": False,
                "locked_test_audio_generated": False,
            }
            write_once(receipt_dir / "FAILED.json", failure)
            failed += 1
    shutil.rmtree(model_root, ignore_errors=True)
    summary = {
        "schema": "english-kws-dual-track-v7-kokoro-generation-summary-v1",
        "run": run,
        "model_lock_sha256": model_lock_sha,
        "declared_count": recipe.SOURCE_COUNT,
        "success_count": success,
        "failed_count": failed,
        "retry_count": 0,
        "calibration_audio_generated": False,
        "locked_test_audio_generated": False,
        "training_allowed": False,
        "product_promotion_allowed": False,
    }
    write_once(output / "GENERATION_SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def source_readback(source, plan, run, output=None):
    import io
    import wave

    source = Path(source)
    model_lock_raw = regular(source / "MODEL_LOCK.json", 2 * 1024 * 1024)
    model_lock = json.loads(model_lock_raw, object_pairs_hook=recipe.unique_pairs)
    model_lock_sha = sha(model_lock_raw)
    frozen = recipe.recipe()
    voice_ids = {group["source_speaker_id"] for group in frozen["source_groups"]}
    need(model_lock.get("schema") == "english-kws-dual-track-v7-kokoro-model-lock-v1"
         and model_lock.get("run") == run
         and model_lock.get("repository") == recipe.MODEL["repository"]
         and model_lock.get("revision") == recipe.MODEL["revision"]
         and model_lock.get("checkpoint_sha256") == recipe.MODEL["checkpoint_sha256"]
         and model_lock.get("source_plan_sha256") == plan["source_plan_sha256"]
         and model_lock.get("product_promotion_allowed") is False,
         "PINNED_KOKORO_MODEL_LOCK")
    need(set(model_lock.get("voice_packs", {})) == voice_ids,
         "EXACT_KOKORO_VOICE_PACK_LOCK_SET")
    vocab = model_lock.get("phoneme_vocab")
    need(isinstance(vocab, dict)
         and sha(canonical(vocab)) == model_lock.get("phoneme_vocab_sha256"),
         "KOKORO_PHONEME_VOCAB_LOCK")

    def member(relative, expected_sha, limit):
        rel = Path(relative)
        need(not rel.is_absolute() and ".." not in rel.parts and ".git" not in rel.parts,
             "SAFE_KOKORO_SOURCE_MEMBER")
        path = source / rel
        raw = regular(path, limit)
        need(sha(raw) == expected_sha, "KOKORO_SOURCE_MEMBER_SHA256:" + str(relative))
        return raw

    def inspect_wav(raw, expected_rate, expected_samples):
        with wave.open(io.BytesIO(raw), "rb") as stream:
            params = (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                      stream.getcomptype(), stream.getnframes())
            decoded = stream.readframes(stream.getnframes())
        need(params == (1, 2, expected_rate, "NONE", expected_samples),
             "KOKORO_WAV_GEOMETRY")
        return decoded

    completed_rows = []
    seen_pcm = set()
    valid_count = 0
    for frozen_row in frozen["rows"]:
        receipt_dir = source / "source" / "receipts" / frozen_row["id"]
        completed_path = receipt_dir / "COMPLETED.json"
        failed_path = receipt_dir / "FAILED.json"
        need(not (completed_path.exists() and failed_path.exists()),
             "KOKORO_ROW_CANNOT_BE_BOTH_COMPLETE_AND_FAILED")
        common = {
            key: frozen_row[key] for key in (
                "source_group_id", "source_speaker_id", "source_speaker_index",
                "voice_family", "role", "id", "request_sha256", "cell_id",
                "requested_text", "requested_class", "pronunciation_hypothesis",
                "kind", "length_scale", "raw_phonemes", "candidate_labels")
        }
        if failed_path.is_file() and not completed_path.exists():
            failure_raw = regular(failed_path, 1024 * 1024)
            failure = json.loads(failure_raw, object_pairs_hook=recipe.unique_pairs)
            need(failure.get("id") == frozen_row["id"]
                 and failure.get("request_sha256") == frozen_row["request_sha256"]
                 and failure.get("run") == run
                 and failure.get("status") == "FAILED_NO_RETRY"
                 and failure.get("retry_allowed") is False,
                 "KOKORO_FAILED_ROW_NO_RETRY_RECEIPT")
            completed_rows.append({
                **common,
                "generation_status": "FAILED_NO_RETRY",
                "source_generation_status": "FAILED_NO_RETRY",
                "source_receipt_sha256": sha(failure_raw),
                "actual_phonemes": None,
                "actual_phoneme_ids": None,
                "pcm_path": None, "wav_path": None,
                "pcm_sha256": None, "wav_sha256": None,
                "pcm_samples": None, "sample_rate": None, "channels": None,
                "voice_id": frozen_row["source_speaker_id"],
                "tts_model_sha256": model_lock["checkpoint_sha256"],
                "voice_pack_sha256": model_lock["voice_packs"][frozen_row["source_speaker_id"]]["sha256"],
            })
            continue
        if not completed_path.is_file():
            completed_rows.append({
                **common,
                "generation_status": "FAILED_NO_RETRY",
                "source_generation_status": "MISSING_NO_RETRY_RECEIPT",
                "source_receipt_sha256": None,
                "actual_phonemes": None,
                "actual_phoneme_ids": None,
                "pcm_path": None, "wav_path": None,
                "pcm_sha256": None, "wav_sha256": None,
                "pcm_samples": None, "sample_rate": None, "channels": None,
                "voice_id": frozen_row["source_speaker_id"],
                "tts_model_sha256": model_lock["checkpoint_sha256"],
                "voice_pack_sha256": model_lock["voice_packs"][frozen_row["source_speaker_id"]]["sha256"],
            })
            continue

        receipt_raw = regular(completed_path, 1024 * 1024)
        receipt = json.loads(receipt_raw, object_pairs_hook=recipe.unique_pairs)
        voice_id = frozen_row["source_speaker_id"]
        voice_lock = model_lock["voice_packs"][voice_id]
        expected_ids = [vocab[char] for char in frozen_row["raw_phonemes"]]
        need(receipt.get("schema") == "english-kws-dual-track-v7-kokoro-source-receipt-v1"
             and receipt.get("run") == run
             and receipt.get("id") == frozen_row["id"]
             and receipt.get("request_sha256") == frozen_row["request_sha256"]
             and receipt.get("source_group_id") == frozen_row["source_group_id"]
             and receipt.get("source_speaker_id") == voice_id
             and receipt.get("role") == frozen_row["role"]
             and receipt.get("cell_id") == frozen_row["cell_id"]
             and receipt.get("requested_text") == frozen_row["requested_text"]
             and receipt.get("requested_phonemes") == frozen_row["raw_phonemes"]
             and receipt.get("requested_phoneme_ids") == expected_ids
             and receipt.get("speed") == frozen_row["speed"]
             and receipt.get("length_scale") == frozen_row["length_scale"]
             and receipt.get("model_revision") == model_lock["revision"]
             and receipt.get("model_sha256") == model_lock["checkpoint_sha256"]
             and receipt.get("config_sha256") == model_lock["config_sha256"]
             and receipt.get("voice_pack_sha256") == voice_lock["sha256"]
             and receipt.get("model_lock_sha256") == model_lock_sha
             and receipt.get("retry") is False
             and receipt.get("retry_allowed") is False
             and receipt.get("actual_phonemes_claimed") is False
             and receipt.get("calibration_audio_generated") is False
             and receipt.get("locked_test_audio_generated") is False,
             "KOKORO_COMPLETED_SOURCE_RECEIPT_BINDING")

        native_raw = member(receipt["native_wav_path"], receipt["native_wav_sha256"], 8 * 1024 * 1024)
        native_samples = receipt["native_samples"]
        inspect_wav(native_raw, 24000, native_samples)
        pcm_raw = member(receipt["pcm_path"], receipt["pcm_sha256"], 1024 * 1024)
        wav_raw = member(receipt["wav_path"], receipt["wav_sha256"], 4 * 1024 * 1024)
        pcm_samples = receipt["pcm_samples"]
        need(6400 <= pcm_samples <= 28672
             and len(pcm_raw) == pcm_samples * 2
             and receipt.get("sample_rate") == 16000
             and receipt.get("channels") == 1
             and inspect_wav(wav_raw, 16000, pcm_samples) == pcm_raw,
             "KOKORO_PCM_WAV_BYTE_IDENTITY_AND_GEOMETRY")
        if receipt["pcm_sha256"] in seen_pcm:
            completed_rows.append({
                **common,
                "generation_status": "FAILED_NO_RETRY",
                "source_generation_status": "DUPLICATE_PCM_QUARANTINED",
                "source_receipt_sha256": sha(receipt_raw),
                "actual_phonemes": None, "actual_phoneme_ids": None,
                "pcm_path": None, "wav_path": None,
                "pcm_sha256": None, "wav_sha256": None,
                "pcm_samples": None, "sample_rate": None, "channels": None,
                "voice_id": voice_id,
                "tts_model_sha256": model_lock["checkpoint_sha256"],
                "voice_pack_sha256": voice_lock["sha256"],
            })
            continue
        seen_pcm.add(receipt["pcm_sha256"])
        valid_count += 1
        completed_rows.append({
            **common,
            "generation_status": "success",
            "source_generation_status": "success",
            "source_receipt_sha256": sha(receipt_raw),
            "actual_phonemes": None,
            "actual_phoneme_ids": None,
            "pcm_path": receipt["pcm_path"],
            "wav_path": receipt["wav_path"],
            "pcm_sha256": receipt["pcm_sha256"],
            "wav_sha256": receipt["wav_sha256"],
            "pcm_samples": pcm_samples,
            "sample_rate": 16000,
            "channels": 1,
            "voice_id": voice_id,
            "tts_model_sha256": model_lock["checkpoint_sha256"],
            "voice_pack_sha256": voice_lock["sha256"],
            "native_wav_path": receipt["native_wav_path"],
            "native_wav_sha256": receipt["native_wav_sha256"],
            "native_samples": native_samples,
        })

    corpus = {
        "schema": recipe.CORPUS_SCHEMA,
        "run": run,
        "source_plan_sha256": plan["source_plan_sha256"],
        "recipe_sha256": recipe.digest(frozen),
        "model_lock_sha256": model_lock_sha,
        "preflight_sha256": model_lock["preflight_report_sha256"],
        "preflight_report_sha256": model_lock["preflight_report_sha256"],
        "source_groups": frozen["source_groups"],
        "declared_count": recipe.SOURCE_COUNT,
        "valid_count": valid_count,
        "rows": completed_rows,
        "human_gold": False,
        "calibration_audio_generated": False,
        "locked_test_audio_generated": False,
        "training_allowed": False,
        "product_promotion_allowed": False,
        "license_evidence": {
            "model_license": model_lock["model_license"],
            "inference_code_license": model_lock["inference_code_license"],
            "voice_asset_license_scope": "Pinned model repository metadata; no human speaker identity or commercial derivative-rights claim",
            "generated_audio_derivative_clearance": "NOT_ASSERTED_LEGAL_REVIEW_REQUIRED_BEFORE_COMMERCIAL_PRODUCT_CLAIM",
        },
        "status": "COMPLETE" if valid_count == recipe.SOURCE_COUNT else "INCOMPLETE_NO_RETRY",
    }
    if output is not None:
        write_once(output, corpus)
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
                capture = recipe.observer_identity(provider) | {"schema": "english-kws-dual-track-v7-kokoro-observer-capture-v1",
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
    return {"schema": "english-kws-dual-track-v7-kokoro-machine-evidence-v1", "rows": rows,
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

