"""One-shot independent Whisper-small English transcription of ten retained TTS WAVs.

No intended wake text, source prompt, or Qwen3-ASR transcript is supplied to
Whisper. The MIT model weight and Python source revision are pinned separately.
Generated machine observations are never training gold or FAR/FRR evidence.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import wave

import english_blind_asr as blind
import english_qwen_asr_sourcefix_once as sourcefix
import english_qwen_capture as english
import english_qwen_real_once as pilot
import multilingual_campaign as market

POLICY = "english-whisper-small-real-once-20261112-v1"
PROVIDER = "whisper-small-en"
MODEL = "openai/whisper-small"
MODEL_SOURCE_COMMIT = "31243bad24cc746f07d4c8bfdd2d974872cb1803"
PACKAGE = "20250625"
WEIGHT_SHA256 = "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
MODEL_URL = ("https://openaipublic.azureedge.net/main/whisper/models/"
             + WEIGHT_SHA256 + "/small.pt")
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-whisper-real-20261112.json"
MAX_AUDIO = 10
MAX_DOWNLOAD_BYTES = 1536 * 1024**2


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def need(ok: bool, error: str):
    if not ok:
        raise ValueError(error)


def request() -> dict:
    original = market.read(TRIGGER)
    expected = {
        "schema": POLICY, "original_tts_run_id": 43214159818,
        "original_tts_head_sha": "f2b91001e4fd56a03bbce9bc6b0c1cae7fad1dc8",
        "original_audio_artifact_id": 13942568634,
        "original_audio_artifact_sha256": "0235b0064006f79bfe9c8fa551c2a1e41da61c86e0fd987a90ee2d47842a9f21",
        "model_id": MODEL, "model_source_commit": MODEL_SOURCE_COMMIT,
        "model_weight_sha256": WEIGHT_SHA256,
        "python_package_version": PACKAGE,
        "max_new_asr_calls": MAX_AUDIO,
        "training_admitted": False, "human_gold": False,
        "three_asr_completed": False,
    }
    need(original == expected, "WHISPER_TRIGGER_POLICY_CHANGED")
    return expected


def source_inputs(parent: Path):
    """Verify original source WAVs, never peek at any recognizer's transcript."""
    original, plan, count = sourcefix.original_audio_root(parent)
    need(count == MAX_AUDIO and plan["asr_calls"] == 0 and
         plan["training_admitted"] is False, "WHISPER_SOURCE_ROLE_DRIFT")
    source = sourcefix.core.verified_source(original)[1]
    need(len(source) == MAX_AUDIO, "WHISPER_SOURCE_COUNT_DRIFT")
    return plan, source


def pinned_model(destination: Path) -> Path:
    """Only official fixed-weight URL; streaming bytes and max length guarded."""
    need(destination.is_dir(), "WHISPER_DOWNLOAD_PARENT_MISSING")
    need(shutil.disk_usage(destination).free >= MAX_DOWNLOAD_BYTES + 1024**3,
         "WHISPER_DISK_BUDGET")
    path = destination / "small.pt"
    need(not path.exists(), "WHISPER_MODEL_MUST_BE_NEW")
    h = hashlib.sha256()
    total = 0
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "kws-research-pin/1"})
    with urllib.request.urlopen(request, timeout=120) as src, path.open("xb") as output:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            need(total <= MAX_DOWNLOAD_BYTES, "WHISPER_MODEL_DOWNLOAD_OVER_LIMIT")
            h.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    need(total > 1024**2 and h.hexdigest() == WEIGHT_SHA256,
         "WHISPER_MODEL_CHECKPOINT_SHA_DRIFT")
    return path


def run(parent: Path, output: Path) -> dict:
    claim = request()
    need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_REF") == "refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
         "WHISPER_REAL_ONCE_MAIN_ONLY")
    blind.private_path(output, "WHISPER_RESULT_NOT_PRIVATE")
    need(output.parent.is_dir() and not output.exists(), "WHISPER_RESULT_CREATE_ONLY")
    plan, sources = source_inputs(parent)
    output.mkdir(mode=0o700)
    english.media()._once(output / "started.json", {
        "schema": POLICY, "model_id": MODEL,
        "model_source_commit": MODEL_SOURCE_COMMIT,
        "model_weight_sha256": WEIGHT_SHA256,
        "source_tts_artifact_sha256": claim["original_audio_artifact_sha256"],
        "source_plan_sha256": sha(market.canonical(plan)),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "training_admitted": False, "no_retry": True})
    model_root = output / "private-model"
    model_root.mkdir(mode=0o700)
    observations = []
    attempts = 0
    failure = None
    try:
        weight_path = pinned_model(model_root)
        import numpy as np
        import torch
        import whisper
        torch.set_num_threads(min(os.cpu_count() or 1, 4))
        model = whisper.load_model(str(weight_path), device="cpu")
        for cell, wav, pcm in sources:
            folder = output / "cells" / cell["cell_id"]
            need(not folder.exists(), "WHISPER_ATTEMPT_ALREADY_EXISTS")
            folder.mkdir(parents=True, mode=0o700)
            english.media()._once(folder / "started.json", {
                "schema": POLICY, "cell_id": cell["cell_id"],
                "provider_id": PROVIDER, "wav_sha256": pcm["wav_sha256"],
                "model_weight_sha256": WEIGHT_SHA256,
                "retry_count": 0, "training_admitted": False})
            attempts += 1
            row_status = "FAILED_NO_RETRY"
            try:
                with wave.open(str(wav), "rb") as src:
                    need(src.getnchannels() == 1 and src.getsampwidth() == 2 and
                         src.getframerate() == 16000 and
                         src.getnframes() == pcm["frames"],
                         "WHISPER_SOURCE_PCM_FORMAT")
                    samples = src.readframes(src.getnframes())
                    need(not src.readframes(1), "WHISPER_SOURCE_PCM_TRAILING")
                audio = np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0
                need(len(audio) == pcm["frames"], "WHISPER_AUDIO_LENGTH_CHANGED")
                # No desired text, prompt injection, other ASR result or hotword bias.
                result = model.transcribe(
                    audio, language="en", fp16=False, verbose=False,
                    initial_prompt=None, condition_on_previous_text=False,
                    temperature=0.0)
                actual_text = result.get("text")
                need(type(actual_text) is str and 0 < len(actual_text) <= 512,
                     "WHISPER_TRANSCRIPT_EMPTY_OR_OVERSIZE")
                native = {"schema": blind.RAW_SCHEMA, "provider_id": PROVIDER,
                          "model_id": MODEL,
                          "model_revision_sha": MODEL_SOURCE_COMMIT,
                          "audio_wav_sha256": pcm["wav_sha256"],
                          "processed_start_frame": 0,
                          "processed_end_frame": pcm["frames"],
                          "decoder_completed": True, "actual_text": actual_text}
                english.media()._once(folder / "native.json", native)
                need(blind.native_result(folder / "native.json", {
                    "provider_id": PROVIDER, "model_id": MODEL,
                    "model_revision_sha": MODEL_SOURCE_COMMIT
                }, cell, pcm) == native, "WHISPER_NATIVE_RESULT_IDENTITY")
                english.media()._once(folder / "completed.json", {
                    "schema": POLICY, "cell_id": cell["cell_id"],
                    "provider_id": PROVIDER, "status": "BLIND_ASR_UNADMITTED",
                    "native_sha256": sha((folder / "native.json").read_bytes()),
                    "wav_sha256": pcm["wav_sha256"],
                    "training_admitted": False, "human_gold": False})
                row_status = "BLIND_ASR_UNADMITTED"
            except Exception as exc:
                english.media()._once(folder / "failed.json", {
                    "schema": POLICY, "cell_id": cell["cell_id"],
                    "status": "FAILED_NO_RETRY",
                    "error_type": type(exc).__name__,
                    "training_admitted": False})
            observations.append({"cell_id": cell["cell_id"],
                                 "wav_sha256": pcm["wav_sha256"],
                                 "status": row_status})
    except BaseException as exc:
        failure = type(exc).__name__
    finally:
        successful = sum(row["status"] == "BLIND_ASR_UNADMITTED" for row in observations)
        summary = {
            "schema": POLICY,
            "status": "COMPLETE" if successful == MAX_AUDIO and failure is None
            else "INCOMPLETE_NO_RETRY",
            "model_id": MODEL, "model_source_commit": MODEL_SOURCE_COMMIT,
            "model_weight_sha256": WEIGHT_SHA256,
            "python_package_version": version("openai-whisper") if attempts else None,
            "source_tts_run_id": claim["original_tts_run_id"],
            "source_plan_sha256": sha(market.canonical(plan)),
            "input_audio_count": MAX_AUDIO, "asr_attempts": attempts,
            "completed_asr": successful, "error_type": failure,
            "rows": observations, "three_asr_completed": False,
            "training_admitted": False, "human_gold": False}
        english.media()._once(output / "summary.json", summary)
        if model_root.exists():
            shutil.rmtree(model_root)
    return summary


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--artifact-parent", type=Path)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--verify-source", action="store_true")
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    try:
        request()
        need(args.verify_source != args.execute and args.artifact_parent is not None,
             "WHISPER_EXPLICIT_MODE_REQUIRED")
        if args.verify_source:
            plan, rows = source_inputs(args.artifact_parent)
            print(json.dumps({"schema": POLICY, "status": "SOURCE_VERIFIED_ONLY",
                              "input_audio_count": len(rows),
                              "source_plan_sha256": sha(market.canonical(plan)),
                              "asr_calls": 0, "training_admitted": False}, sort_keys=True))
            return 0
        need(args.output is not None, "WHISPER_NEW_OUTPUT_REQUIRED")
        result = run(args.artifact_parent, args.output)
        print(json.dumps({k: result[k] for k in (
            "status", "asr_attempts", "completed_asr",
            "training_admitted")}, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except (OSError, ValueError, wave.Error) as exc:
        print("WHISPER_CAPTURE_STOP: " + type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
