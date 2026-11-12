"""Bounded real three-ASR follow-on for the 63 new English Qwen WAVs.

Each provider executes independently on the same hash-bound real WAVs.
Neither target text nor another provider's ASR observation enters inference.
This is weak, machine-only research and NEVER authenticates labels or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import english_blind_asr as blind
import english_qwen_asr_real_once as qwen
import english_whisper_real_once as whisper_old
import english_parakeet_real_once as parakeet_old
import english_seven_voice_tts_once as seven
import english_qwen_capture as eng
import multilingual_campaign as market

POLICY = "english-seven-voice-three-blind-asr-event-v1"
SOURCE_RUN_ID = 43242278962
SOURCE_HEAD_SHA = "bec1a1421d1db2de17afb9c6e3155343becdb980"
SOURCE_WORKFLOW = "English seven new Qwen voices 63 real WAVs once"
ARTIFACT_ROOT = "english-seven-voice-actual"
MAX_CALLS = 63
PROVIDERS = ("qwen", "whisper", "parakeet")
PROVENANCE = {
    "qwen": ("qwen3-asr-en", qwen.MODEL, qwen.REV),
    "whisper": (whisper_old.PROVIDER, whisper_old.MODEL,
                whisper_old.MODEL_SOURCE_COMMIT),
    "parakeet": (parakeet_old.PROVIDER, parakeet_old.MODEL,
                 parakeet_old.MODEL_COMMIT),
}


def need(ok, reason: str):
    if not ok:
        raise ValueError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_from_artifact(parent: Path) -> tuple[dict, list[tuple[dict, Path, dict]]]:
    blind.private_path(parent, "SEVEN_ASR_ARTIFACT_PARENT_NOT_PRIVATE")
    root = parent / ARTIFACT_ROOT
    need(root.is_dir() and not root.is_symlink() and
         not (parent / "source-plan.json").exists(),
         "SEVEN_ASR_NESTED_SOURCE_ROOT_MISSING")
    raw = market.read(root / "source-plan.json")
    need(type(raw.get("tts_pins")) is list and len(raw["tts_pins"]) == 1,
         "SEVEN_ASR_TTS_PIN_MISSING")
    pin = raw["tts_pins"][0]
    lock_sha = pin.get("model_assets_sha256")
    need(type(lock_sha) is str and
         market.HEX64.fullmatch(lock_sha) is not None,
         "SEVEN_ASR_MODEL_LOCK_SHA_INVALID")
    expected = seven.plan(lock_sha)
    need(market.canonical(raw) == market.canonical(expected),
         "SEVEN_ASR_PLAN_OR_SPEAKER_DRIFT")
    model_lock = root / "model-lock.json"
    _, checked_lock_sha = eng.media()._safe_file(model_lock, limit=1024*1024)
    need(checked_lock_sha == lock_sha, "SEVEN_ASR_MODEL_LOCK_DRIFT")
    summary = market.read(root / "summary.json")
    need(summary.get("schema") == seven.POLICY and
         summary.get("status") == "COMPLETE" and
         summary.get("planned_audio_count") == MAX_CALLS and
         summary.get("attempted_audio_count") == MAX_CALLS and
         summary.get("generated_audio_count") == MAX_CALLS and
         summary.get("previous_audio_regenerated") == 0 and
         summary.get("asr_calls") == 0 and
         summary.get("training_admitted") is False and
         summary.get("human_gold") is False,
         "SEVEN_ASR_SOURCE_NOT_COMPLETE")
    rows = summary.get("source_rows")
    need(type(rows) is list and len(rows) == MAX_CALLS and
         len({z["cell_id"] for z in rows}) == MAX_CALLS and
         {z["cell_id"] for z in rows} == {z["cell_id"] for z in raw["cells"]},
         "SEVEN_ASR_ROW_ROSTER_DRIFT")
    by_id = {z["cell_id"]: z for z in rows}
    actual = []
    for cell in raw["cells"]:
        observed = by_id[cell["cell_id"]]
        need(observed["status"] == "GENERATED_UNADMITTED" and
             observed["source_family_id"] == cell["source_family_id"] and
             observed["role"] == cell["role"] and
             observed["intent"] == cell["intent"] and
             observed["training_admitted"] is False,
             "SEVEN_ASR_ROW_PROVENANCE_DRIFT")
        original = blind.verify_capture(raw, root / "capture", cell)
        need(original is not None, "SEVEN_ASR_WAV_MISSING")
        wav, pcm = original
        need(observed["wav_sha256"] == pcm["wav_sha256"],
             "SEVEN_ASR_ORIGINAL_WAV_SUMMARY_DRIFT")
        actual.append((cell, wav, pcm))
    need(len(actual) == MAX_CALLS, "SEVEN_ASR_REAL_WAV_COUNT_DRIFT")
    return raw, actual


def validate_trigger_event(path: Path) -> dict:
    event = market.read(path)
    run = event.get("workflow_run")
    need(type(run) is dict and
         event.get("action") == "completed" and
         run.get("id") == SOURCE_RUN_ID and
         run.get("name") == SOURCE_WORKFLOW and
         run.get("head_sha") == SOURCE_HEAD_SHA and
         run.get("head_branch") == "main" and
         run.get("status") == "completed" and
         run.get("conclusion") == "success" and
         run.get("event") == "push" and
         run.get("run_attempt") == 1,
         "SEVEN_ASR_SOURCE_WORKFLOW_EVENT_INVALID")
    return run


def _load(provider: str, model_root: Path):
    """Only explicit execute mode reaches one real, exact-pinned provider."""
    need(provider in PROVIDERS, "SEVEN_ASR_UNKNOWN_PROVIDER")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    if provider == "qwen":
        folder = qwen.lock_model(model_root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from qwen_asr import Qwen3ASRModel
        torch.set_num_threads(min(os.cpu_count() or 1, 4))
        engine = Qwen3ASRModel.from_pretrained(
            str(folder), dtype=torch.float32, device_map="cpu",
            max_inference_batch_size=1, max_new_tokens=128,
            attn_implementation="sdpa")
        def decode(wav: Path, folder: Path) -> str:
            values = engine.transcribe(audio=str(wav), language="English", context="")
            need(type(values) is list and len(values) == 1 and
                 type(values[0].text) is str, "QWEN_NATIVE_RESULT_SHAPE")
            return values[0].text
        return decode
    if provider == "whisper":
        weights = whisper_old.pinned_model(model_root)
        import torch
        import numpy as np
        import whisper
        torch.set_num_threads(min(os.cpu_count() or 1, 4))
        engine = whisper.load_model(str(weights), device="cpu")
        def decode(wav: Path, folder: Path) -> str:
            with wave.open(str(wav), "rb") as stream:
                need(stream.getnchannels() == 1 and stream.getsampwidth() == 2 and
                     stream.getframerate() == 16000 and
                     stream.getcomptype() == "NONE", "WHISPER_NATIVE_WAV_GEOMETRY")
                raw = stream.readframes(stream.getnframes())
                need(not stream.readframes(1), "WHISPER_EXTRA_PCM")
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            out = engine.transcribe(pcm, language="en", fp16=False, verbose=False,
                                    initial_prompt=None,
                                    condition_on_previous_text=False,
                                    temperature=0.0)
            return out["text"]
        return decode
    binary, _ = parakeet_old.verify_runtime(model_root)
    model, _ = parakeet_old.verify_gguf(model_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    def decode(wav: Path, folder: Path) -> str:
        env = {"PATH":"/usr/bin:/bin", "HOME":str(folder), "LANG":"C.UTF-8",
               "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
               "NEMO_SPEECH_MODEL_DIR":str(model_root), "OMP_NUM_THREADS":"4"}
        result = subprocess.run(
            [str(binary), "transcribe", str(wav),
             "--model", str(model), "--device", "cpu"],
            cwd=str(folder), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, check=False, timeout=180)
        need(result.returncode == 0, "PARAKEET_NATIVE_PROCESS_FAILED")
        return result.stdout.strip()
    return decode


def capture(provider: str, parent: Path, output: Path,
            *, decoder=None, event_path: Path | None = None) -> dict:
    need(provider in PROVIDERS, "SEVEN_ASR_PROVIDER_INVALID")
    if decoder is None:
        need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
             os.environ.get("GITHUB_EVENT_NAME") == "workflow_run" and
             os.environ.get("GITHUB_RUN_ATTEMPT") == "1" and
             event_path is not None, "SEVEN_ASR_EVENT_ONLY")
        validate_trigger_event(event_path)
    plan, cells = source_from_artifact(parent)
    blind.private_path(output, "SEVEN_ASR_OUTPUT_NOT_PRIVATE")
    need(output.parent.is_dir() and not output.exists(),
         "SEVEN_ASR_OUTPUT_CREATE_ONLY")
    output.mkdir(mode=0o700)
    provider_id, model_id, revision = PROVENANCE[provider]
    eng.media()._once(output / "started.json", {
        "schema": POLICY, "provider": provider,
        "provider_id": provider_id, "model_id": model_id,
        "model_revision_sha": revision,
        "source_tts_run_id": SOURCE_RUN_ID,
        "source_tts_head_sha": SOURCE_HEAD_SHA,
        "source_plan_sha256": sha(market.canonical(plan)),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "model_calls_authorized": decoder is None,
        "training_admitted": False, "no_retry": True})
    model_root = output / "private-pretrained"
    model_root.mkdir(mode=0o700)
    rows = []
    failure = None
    try:
        if decoder is None:
            decoder = _load(provider, model_root)
        for cell, wav, pcm in cells:
            folder = output / "cells" / cell["cell_id"]
            need(not folder.exists(), "SEVEN_ASR_ALREADY_CONSUMED")
            folder.mkdir(parents=True, mode=0o700)
            eng.media()._once(folder / "started.json", {
                "schema": POLICY, "cell_id": cell["cell_id"],
                "provider_id": provider_id,
                "wav_sha256": pcm["wav_sha256"],
                "model_revision_sha": revision, "attempt": 1,
                "training_admitted": False})
            status = "FAILED_NO_RETRY"
            try:
                text = decoder(wav, folder)
                need(type(text) is str and 0 < len(text) <= 512 and
                     not any(ord(ch) < 32 for ch in text),
                     "SEVEN_ASR_TEXT_INVALID")
                native = {
                    "schema": blind.RAW_SCHEMA, "provider_id": provider_id,
                    "model_id": model_id, "model_revision_sha": revision,
                    "audio_wav_sha256": pcm["wav_sha256"],
                    "processed_start_frame": 0, "processed_end_frame": pcm["frames"],
                    "decoder_completed": True, "actual_text": text}
                eng.media()._once(folder / "native.json", native)
                need(blind.native_result(folder / "native.json", {
                    "provider_id": provider_id, "model_id": model_id,
                    "model_revision_sha": revision,
                }, cell, pcm) == native,
                     "SEVEN_ASR_NATIVE_PROVENANCE_DRIFT")
                eng.media()._once(folder / "completed.json", {
                    "schema": POLICY, "cell_id": cell["cell_id"],
                    "provider_id": provider_id, "wav_sha256": pcm["wav_sha256"],
                    "native_sha256": sha((folder / "native.json").read_bytes()),
                    "status": "BLIND_ASR_UNADMITTED",
                    "training_admitted": False, "human_gold": False})
                status = "BLIND_ASR_UNADMITTED"
            except Exception as error:
                eng.media()._once(folder / "failed.json", {
                    "schema": POLICY, "cell_id": cell["cell_id"],
                    "status": "FAILED_NO_RETRY",
                    "error_type": type(error).__name__,
                    "training_admitted": False})
            rows.append({
                "cell_id": cell["cell_id"], "wav_sha256": pcm["wav_sha256"],
                "source_family_id": cell["source_family_id"],
                "role": cell["role"], "status": status,
            })
    except BaseException as error:
        failure = type(error).__name__
    finally:
        ok = sum(x["status"] == "BLIND_ASR_UNADMITTED" for x in rows)
        summary = {
            "schema": POLICY, "status": "COMPLETE" if ok == MAX_CALLS and failure is None
            else "INCOMPLETE_NO_RETRY", "provider": provider,
            "provider_id": provider_id, "model_id": model_id,
            "model_revision_sha": revision, "source_tts_run_id": SOURCE_RUN_ID,
            "source_plan_sha256": sha(market.canonical(plan)),
            "source_audio_count": MAX_CALLS, "asr_attempts": len(rows),
            "completed_asr": ok, "error_type": failure,
            "rows": rows, "training_admitted": False,
            "human_gold": False, "three_asr_completed": False,
        }
        eng.media()._once(output / "summary.json", summary)
        if model_root.exists():
            shutil.rmtree(model_root)
    return summary


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--provider", choices=PROVIDERS, required=True)
    cli.add_argument("--artifact-parent", type=Path, required=True)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--event-file", type=Path)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    try:
        plan, rows = source_from_artifact(args.artifact_parent)
        if not args.execute:
            print(json.dumps({
                "schema": POLICY, "provider": args.provider,
                "source_audio_count": len(rows), "new_asr_calls": 0,
                "source_plan_sha256": sha(market.canonical(plan)),
                "training_admitted": False}, sort_keys=True))
            return 0
        need(args.output is not None, "SEVEN_ASR_OUTPUT_REQUIRED")
        result = capture(args.provider, args.artifact_parent, args.output,
                         event_path=args.event_file)
        print(json.dumps({
            "status": result["status"], "provider": result["provider"],
            "asr_attempts": result["asr_attempts"],
            "completed_asr": result["completed_asr"],
            "training_admitted": result["training_admitted"]
        }, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except (ValueError, OSError, wave.Error) as error:
        print("SEVEN_VOICE_ASR_STOP: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
