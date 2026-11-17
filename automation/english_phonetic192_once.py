"""One finite direct-phone Piper campaign and first blind observers on new voices.

Source checks and packet joins use the standard library. Heavy imports occur only
inside guarded public-main execution stages. Every attempted inference has a
create-once receipt; interruptions never make a cell eligible for retry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import sys
import time
import urllib.request
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import english_phonetic192_recipe as recipe

SCHEMA = "english-synthetic192-execution-v1"
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
    return {"repository": env.get("GITHUB_REPOSITORY"), "commit": env.get("GITHUB_SHA"),
        "run_id": env.get("GITHUB_RUN_ID"), "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "event": env.get("GITHUB_EVENT_NAME")}


def validate_plan(plan, *, check_sources=True):
    need(plan.get("schema") == SCHEMA and type(plan.get("enabled")) is bool, "PLAN_SCHEMA_ENABLE")
    frozen = recipe.recipe()
    need(plan.get("recipe_sha256") == recipe.digest(frozen), "RECIPE_PIN")
    need(plan.get("campaign_id") == frozen["campaign_id"] == recipe.CAMPAIGN_ID
         and plan.get("silver_admission", {}).get("source_design_id") == recipe.CAMPAIGN_ID, "EXACT_NEW_SOURCE_DESIGN")
    need(plan.get("synthesis_config") == frozen["synthesis_config"] == recipe.SYNTHESIS, "EXACT_SYNTHESIS_PARAMETERS")
    need(plan.get("maximum_tts_calls") == 192 and plan.get("maximum_calls_per_observer") == 192
         and plan.get("retry_count") == 0 and plan.get("new_kws_calls") == 0, "FINITE_CALL_BUDGET")
    need(plan.get("generation_deadline_seconds") == 900 and
         plan.get("observation_deadline_seconds") == 3600, "STAGE_DEADLINE")
    need(len(frozen["rows"]) == 192 and len({r["id"] for r in frozen["rows"]}) == 192,
         "EXACT_SOURCE_ROSTER")
    speakers = {r["source_speaker_id"] for r in frozen["rows"]}
    need(len(speakers) == 24 and not speakers.intersection(frozen["exclusion_speakers"]), "OLD_SOURCE_EXCLUSION")
    groups = recipe.source_groups()
    need([sum(g["role"] == role for g in groups) for role in recipe.ROLES] == [12, 4, 4, 4], "FROZEN_SOURCE_SPLIT")
    if check_sources:
        need(plan.get("worker_sha256") == sha(regular(Path(__file__), 128 * 1024)), "WORKER_SOURCE_PIN")
        need(plan.get("recipe_code_sha256") == sha(regular(Path(recipe.__file__), 128 * 1024)), "RECIPE_SOURCE_PIN")
        helpers = plan.get("helper_sha256")
        need(type(helpers) is dict and 12 <= len(helpers) <= 24, "PINNED_HELPER_CLOSURE")
        for name, digest in helpers.items():
            need(HEX64.fullmatch(digest) is not None and sha(regular(within(ROOT, name), 256 * 1024)) == digest,
                 "HELPER_SOURCE_PIN: " + name)
    contract = plan.get("admission_contract", {})
    need(contract.get("repository") == "jiying2007/kws-pipeline" and
         contract.get("module_path") == "training/english_phonetic_silver.py" and
         contract.get("profile_path") == "configs/research/english-pronunciation-profile-20261115.json", "CONTRACT_LOCATION")
    if plan["enabled"]:
        need(type(plan.get("next_main_before")) is str and HEX40.fullmatch(plan["next_main_before"]), "EXACT_MAIN_PREDECESSOR")
        need(type(contract.get("commit")) is str and HEX40.fullmatch(contract["commit"]), "CONTRACT_COMMIT_PIN")
        need(all(type(contract.get(k)) is str and HEX64.fullmatch(contract[k])
                 for k in ("module_sha256", "profile_sha256")), "CONTRACT_BYTES_PINS")
        silver = plan.get("silver_admission")
        need(type(silver) is dict and silver.get("source_groups") == groups and
             silver.get("excluded_source_speaker_ids") == frozen["exclusion_speakers"], "SILVER_SOURCE_ROSTER")
        need(silver.get("cells") == [{k: c[k] for k in ("cell_id", "requested_text", "requested_class", "pronunciation_hypothesis")}
              for c in recipe.CELLS], "SILVER_CELL_ROSTER")
        need(silver.get("recognizers") == [recipe.observer_identity(p) for p in ("qwen", "whisper", "parakeet")]
             and silver.get("phone_recognizer") == recipe.observer_identity("phoneme"), "SILVER_OBSERVER_PINS")
    return frozen


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


def source_model(folder):
    from importlib.metadata import version
    from huggingface_hub import hf_hub_download
    need(version("piper-tts") == recipe.MODEL["piper_version"], "PIPER_VERSION")
    paths = {}
    for kind in ("config", "model"):
        paths[kind] = Path(hf_hub_download(repo_id=recipe.MODEL["repository"],
            filename=recipe.MODEL[kind + "_path"], revision=recipe.MODEL["revision"],
            local_dir=str(folder), token=False))
    cb = regular(paths["config"], 1024 * 1024)
    mb = regular(paths["model"], 150 * 1024**2)
    need(len(cb) == recipe.MODEL["config_bytes"] and hashlib.md5(cb).hexdigest() == recipe.MODEL["config_md5"] and sha(cb) == recipe.MODEL["config_sha256"], "PIPER_CONFIG_BYTES")
    need(len(mb) == recipe.MODEL["model_bytes"] and sha(mb) == recipe.MODEL["model_sha256"], "PIPER_MODEL_BYTES")
    config = loads(cb)
    need(config.get("num_speakers") == 904 and config.get("audio", {}).get("sample_rate") == 22050
         and config.get("espeak", {}).get("voice") == "en-us", "PIPER_GEOMETRY")
    need(all(config["speaker_id_map"].get(f"p{speaker}") == index for index, speaker in recipe.NEW_VOICES), "EXACT_NEW_SPEAKER_MAP")
    from piper import PiperVoice
    voice = PiperVoice.load(str(paths["model"]), config_path=str(paths["config"]), use_cuda=False)
    need(callable(voice.phonemes_to_ids) and callable(voice.phoneme_ids_to_audio), "DIRECT_PHONEME_API")
    import piper.voice
    api_bytes = regular(Path(piper.voice.__file__), 128 * 1024)
    lock = dict(recipe.MODEL) | {"config_sha256": sha(cb),
        "source_speaker_map": {f"p{s}": i for i, s in recipe.NEW_VOICES},
        "installed_voice_api_sha256": sha(api_bytes), "model_bytes_verified": True,
        "direct_api": "PiperVoice.phonemes_to_ids -> PiperVoice.phoneme_ids_to_audio"}
    return voice, lock, cb, api_bytes


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
    if row["generation_mode"] == "direct-phoneme-ids":
        need(ids == recipe.DIRECT_PHONE_IDS.get(row["cell_id"]), "EXACT_REVIEWED_DIRECT_PHONE_IDS")
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


def generate(plan, output):
    run = execution_guard(plan)
    output = new_directory(output)
    frozen = recipe.recipe()
    once(output / "RECIPE.json", frozen)
    once(output / "EXECUTION_PLAN.json", plan)
    once(output / "STARTED.json", {"schema": SCHEMA, "stage": "generate", "run": run,
        "maximum_calls": 192, "retry_count": 0, "model_calls_at_start": 0})
    model_root = output / "private-model"; model_root.mkdir(mode=0o700)
    calls = 0; completed = 0; error = None; start = time.monotonic()
    set_deadline(plan["generation_deadline_seconds"])
    try:
        voice, lock, config_bytes, api_bytes = source_model(model_root)
        once(output / "MODEL_LOCK.json", lock)
        (output / "piper-config.json").write_bytes(config_bytes)
        (output / "installed-piper-voice.py").write_bytes(api_bytes)
        from piper import SynthesisConfig
        import onnxruntime
        onnxruntime.set_seed(20261117)
        for row in frozen["rows"]:
            folder = output / "audio" / row["id"]
            folder.mkdir(parents=True, mode=0o700)
            try:
                phones, ids = request_phonemes(voice, row)
                once(folder / "STARTED.json", {"schema": SCHEMA, "stage": "generate", "id": row["id"],
                    "request_sha256": row["request_sha256"], "attempt": 1, "phonemes": phones,
                    "phoneme_ids": ids, "speaker_index": row["source_speaker_index"],
                    "direct_api": True, "raw_bracket_parser_used": False,
                    "synthesis_config": recipe.SYNTHESIS, "campaign_id": recipe.CAMPAIGN_ID})
                need(calls < 192, "SOURCE_CALL_CEILING")
                calls += 1
                audio = voice.phoneme_ids_to_audio(ids, SynthesisConfig(speaker_id=row["source_speaker_index"],
                    **recipe.SYNTHESIS))
                measured = write_audio(folder, audio)
                once(folder / "COMPLETED.json", {"schema": SCHEMA, "id": row["id"], "status": "COMPLETE",
                    "request_sha256": row["request_sha256"], "generation_model_sha256": recipe.MODEL["model_sha256"],
                    "phonemes": phones, "phoneme_ids": ids, "observed_target_pronunciation": False,
                    "human_gold": False, "synthesis_config": recipe.SYNTHESIS, "campaign_id": recipe.CAMPAIGN_ID, **measured})
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
            "status": "COMPLETE" if completed == 192 and error is None else "INCOMPLETE_NO_RETRY",
            "declared_count": 192, "actual_forward_attempts": calls, "completed": completed,
            "error_type": error, "elapsed_seconds": time.monotonic() - start,
            "retry_count": 0, "old_audio_model_calls": 0}
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
    need(read(source / "RECIPE.json") == frozen and read(source / "EXECUTION_PLAN.json") == plan, "SOURCE_RECIPE_OR_PLAN")
    start = read(source / "STARTED.json")
    need(start.get("schema") == SCHEMA and start.get("stage") == "generate" and
         start.get("maximum_calls") == 192 and start.get("retry_count") == 0, "SOURCE_STARTED")
    lock_path = source / "MODEL_LOCK.json"
    if lock_path.is_file():
        lock = read(lock_path)
        need(all(lock.get(k) == v for k, v in recipe.MODEL.items()), "SOURCE_MODEL_LOCK")
        cb = regular(source / "piper-config.json", 1024 * 1024)
        need(sha(cb) == lock["config_sha256"] and hashlib.md5(cb).hexdigest() == recipe.MODEL["config_md5"] and sha(cb) == recipe.MODEL["config_sha256"], "SOURCE_CONFIG_PIN")
        need(sha(regular(source / "installed-piper-voice.py", 128 * 1024)) == lock["installed_voice_api_sha256"], "PIPER_API_CAPTURE")
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
            receipt = read(done); receipt_start = read(begun)
            need(receipt.get("status") == "COMPLETE" and receipt.get("id") == row["id"] and
                 receipt.get("request_sha256") == row["request_sha256"] and
                 receipt.get("generation_model_sha256") == recipe.MODEL["model_sha256"] and
                 receipt.get("human_gold") is False, "SOURCE_COMPLETE_BINDING")
            need(receipt_start.get("id") == row["id"] and receipt_start.get("request_sha256") == row["request_sha256"]
                 and receipt_start.get("attempt") == 1 and receipt_start.get("direct_api") is True and
                 receipt_start.get("raw_bracket_parser_used") is False and
                 receipt_start.get("phonemes") == receipt.get("phonemes") and
                 receipt_start.get("phoneme_ids") == receipt.get("phoneme_ids") and
                 receipt_start.get("speaker_index") == row["source_speaker_index"]
                 and receipt_start.get("synthesis_config") == receipt.get("synthesis_config") == recipe.SYNTHESIS
                 and receipt_start.get("campaign_id") == receipt.get("campaign_id") == recipe.CAMPAIGN_ID, "SOURCE_DIRECT_API_RECEIPT")
            if row["generation_mode"] == "direct-phoneme-ids":
                need(receipt["phonemes"] == list(row["raw_phonemes"]) and
                     receipt["phoneme_ids"] == recipe.DIRECT_PHONE_IDS[row["cell_id"]], "SAVED_REVIEWED_DIRECT_PHONE_REQUEST")
            row.update({k: receipt[k] for k in ("pcm_sha256", "pcm_samples", "wav_sha256")})
            pcm = verify_wav(folder / "audio-16k.wav", row)
            need(regular(folder / "audio-16k.pcm") == pcm, "RAW_PCM_COPY")
            need(sha(regular(folder / "native-22050.wav")) == receipt["native_wav_sha256"], "NATIVE_WAV_HASH")
            row.update(generation_status="success", pcm_path=f"source/audio/{row['id']}/audio-16k.pcm",
                wav_path=f"source/audio/{row['id']}/audio-16k.wav", actual_phonemes=receipt["phonemes"],
                actual_phoneme_ids=receipt["phoneme_ids"], source_receipt_sha256=sha(regular(done)))
        elif failed.is_file():
            need(read(failed).get("id") == row["id"], "FAILED_SOURCE_ID")
            row["generation_status"] = "FAILED_NO_RETRY"
        elif begun.is_file():
            row["generation_status"] = "UNKNOWN_AFTER_START_NO_RETRY"
        rows.append(row)
    pcm_hashes = [r["pcm_sha256"] for r in rows if r["generation_status"] == "success"]
    duplicates = {d for d in pcm_hashes if pcm_hashes.count(d) > 1}
    for row in rows:
        if row["pcm_sha256"] in duplicates:
            row["generation_status"] = "DUPLICATE_PCM_QUARANTINED"
    corpus = {"schema": "english-synthetic192-corpus-v1", "run": start["run"],
        "campaign_id": recipe.CAMPAIGN_ID, "source_design_id": recipe.CAMPAIGN_ID,
        "recipe_sha256": recipe.digest(frozen), "declared_count": 192,
        "valid_count": sum(r["generation_status"] == "success" for r in rows),
        "source_groups": recipe.source_groups(), "rows": rows, "human_gold": False,
        "evaluation_scope": "FRESH_SOURCE_GROUPS_WITH_SHARED_PIPER_PRETRAINING",
        "independent_human_speaker_claim": False, "duplicate_pcm_count": len(duplicates)}
    return corpus


def observe(provider, source, plan, output):
    run = execution_guard(plan)
    need(provider in recipe.PROVIDERS, "KNOWN_PROVIDER")
    corpus = source_readback(source, plan)
    need(corpus["run"] == run, "SAME_RUN_SOURCE")
    rows = [r for r in corpus["rows"] if r["generation_status"] == "success"]
    output = new_directory(output)
    once(output / "STARTED.json", {"schema": SCHEMA, "stage": "observe", "run": run,
        "provider": provider, "source_corpus_sha256": recipe.digest(corpus),
        "valid_source_count": len(rows), "maximum_calls": 192, "retry_count": 0,
        "observer_identity": recipe.observer_identity(provider)})
    model_root = output / "private-model"; model_root.mkdir(mode=0o700)
    calls = completed = 0; error = None; start = time.monotonic()
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
            folder = output / row["id"]; folder.mkdir(mode=0o700)
            source_wav = Path(source) / "audio" / row["id"] / "audio-16k.wav"
            verify_wav(source_wav, row)
            once(folder / "STARTED.json", {"schema": SCHEMA, "provider": provider, "id": row["id"],
                "source_pcm_sha256": row["pcm_sha256"], "source_wav_sha256": row["wav_sha256"],
                "source_samples": row["pcm_samples"], "attempt": 1,
                "target_conditioned": False, "requested_text_passed": False, "role_passed": False})
            need(calls < 192, "OBSERVER_CALL_CEILING")
            calls += 1
            try:
                capture = recipe.observer_identity(provider) | {"schema": "english-synthetic192-native-capture-v1",
                    "id": row["id"], "source_pcm_sha256": row["pcm_sha256"],
                    "source_wav_sha256": row["wav_sha256"], "source_samples": row["pcm_samples"],
                    "status": "success", "target_conditioned": False,
                    "input_coverage_complete": True, "decoder_output_complete": True}
                if provider == "phoneme":
                    tokens, frames = decode(source_wav)
                    need(frames == row["pcm_samples"] and type(tokens) is list and 1 <= len(tokens) <= 300
                         and all(type(t) is str and 0 < len(t) <= 40 for t in tokens), "PHONE_OUTPUT_BINDING")
                    capture["phone_tokens"] = tokens
                else:
                    text = decode(source_wav, folder)
                    need(type(text) is str and 0 < len(text.strip()) <= 512 and
                         not any(ord(ch) < 32 for ch in text.strip()), "ASR_OUTPUT_TEXT")
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
            "declared_count": 192, "valid_source_count": len(rows), "actual_forward_attempts": calls,
            "completed": completed, "error_type": error, "elapsed_seconds": time.monotonic() - start,
            "retry_count": 0, "old_audio_model_calls": 0}
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
    by_id = {r["id"]: r for r in corpus["rows"]}
    for row in corpus["rows"]:
        result = {"id": row["id"], "observations": [], "phone_observation": None}
        for provider in recipe.PROVIDERS:
            path = Path(root) / "evidence" / provider / row["id"] / "CAPTURE.json"
            if not path.is_file():
                continue
            capture = read(path)
            need(row["generation_status"] == "success", "CAPTURE_REQUIRES_VALID_SOURCE")
            need(capture.get("id") == row["id"] and all(capture.get(k) == v for k, v in recipe.observer_identity(provider).items()),
                 "CAPTURE_MODEL_BINDING")
            need(capture.get("source_pcm_sha256") == row["pcm_sha256"] and
                 capture.get("source_wav_sha256") == row["wav_sha256"] and
                 capture.get("source_samples") == row["pcm_samples"], "CAPTURE_SOURCE_BINDING")
            observation = capture | {"capture_path": path.relative_to(root).as_posix(), "capture_sha256": sha(regular(path))}
            if provider == "phoneme":
                result["phone_observation"] = observation
            else:
                result["observations"].append(observation)
        rows.append(result)
    need(len(by_id) == 192, "COMPLETE_EVIDENCE_DENOMINATOR")
    return {"schema": "english-synthetic-machine-evidence-v1", "rows": rows,
        "human_gold": False, "new_model_calls_during_join": 0}


def fetch_policy(contract, target):
    target.mkdir(mode=0o700)
    for kind, name in (("module", "english_synthetic_silver.py"), ("profile", "profile.json")):
        path = contract[kind + "_path"]
        url = f"https://raw.githubusercontent.com/{contract['repository']}/{contract['commit']}/{path}"
        request = urllib.request.Request(url, headers={"User-Agent": "english-phonetic192-pinned-policy/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(256 * 1024 + 1)
        need(0 < len(raw) <= 256 * 1024 and sha(raw) == contract[kind + "_sha256"], "PINNED_POLICY_BYTES")
        (target / name).write_bytes(raw)
    once(target / "IDENTITY.json", contract)


def combine(source, observers, plan, output):
    run = execution_guard(plan)
    corpus = source_readback(source, plan)
    need(corpus["run"] == run, "SAME_RUN_CORPUS")
    output = new_directory(output)
    copy_closed_tree(source, output / "source")
    (output / "evidence").mkdir(mode=0o700)
    for provider in recipe.PROVIDERS:
        path = Path(observers) / ("phonetic192-observer-" + provider)
        if path.is_dir():
            started = read(path / "STARTED.json")
            need(started.get("run") == run and started.get("provider") == provider and
                 started.get("source_corpus_sha256") == recipe.digest(corpus) and
                 started.get("observer_identity") == recipe.observer_identity(provider), "OBSERVER_HANDOFF_IDENTITY")
            copy_closed_tree(path, output / "evidence" / provider, 32 * 1024**2)
    once(output / "CORPUS_MANIFEST.json", corpus)
    evidence = capture_evidence(output, corpus)
    once(output / "MACHINE_EVIDENCE.json", evidence)
    once(output / "EXECUTION_PLAN.json", plan)
    fetch_policy(plan["admission_contract"], output / "policy")
    spec = importlib.util.spec_from_file_location("phonetic192_pinned_admission", output / "policy" / "english_synthetic_silver.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    admission = module.build_admission(corpus, evidence, output,
        profile=read(output / "policy" / "profile.json"), plan=plan)
    once(output / "ADMISSION.json", admission)
    once(output / "RESULT.json", {"schema": SCHEMA, "stage": "combine", "run": run,
        "status": "MACHINE_EVIDENCE_PACKET_RETAINED", "declared_count": 192,
        "valid_sources": corpus["valid_count"], "retry_count": 0, "new_model_calls": 0,
        "admission_sha256": sha(regular(output / "ADMISSION.json")),
        "source_status": read(Path(source) / "RESULT.json").get("status") if (Path(source) / "RESULT.json").is_file() else "UNKNOWN_WITHOUT_TERMINAL_RESULT",
        "human_gold": False, "physical_ssc305_execution": False})
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            raw = regular(path, MAX_FILE)
            files.append({"path": path.relative_to(output).as_posix(), "bytes": len(raw), "sha256": sha(raw)})
    once(output / "FILES.json", {"schema": "english-synthetic192-files-v1", "files": files})
    return {"status": "MACHINE_EVIDENCE_PACKET_RETAINED", "source_count": 192, "valid_count": corpus["valid_count"], "files": len(files)}


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
    args = parser.parse_args(argv)
    plan = read(args.plan)
    validate_plan(plan)
    if args.source_check:
        result = {"status": "SOURCE_ONLY_NO_MODEL", "enabled": plan["enabled"], "declared_sources": 192}
    elif args.guard:
        result = execution_guard(plan)
    elif args.generate:
        need(args.output is not None, "OUTPUT_REQUIRED")
        result = generate(plan, args.output)
    elif args.read_source:
        need(args.source is not None, "SOURCE_REQUIRED")
        corpus = source_readback(args.source, plan)
        if args.output is not None:
            once(args.output, corpus)
        result = {"status": "SOURCE_READBACK", "valid_count": corpus["valid_count"], "corpus_sha256": recipe.digest(corpus)}
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
