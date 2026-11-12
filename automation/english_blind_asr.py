"""Blinded three-ASR runner for private English Qwen KWS research capture.

Only reads hash-verified generated WAVs; adapters see audio, model directory
and output path, NEVER the intended wake text, label, or speaker identity.
No implicit network/model downloads, no retries, no training admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import wave

import english_qwen_capture as english
import multilingual_campaign as market

POLICY = "en-market-three-blind-asr-v1"
ADAPTER_SCHEMA = "en-market-three-asr-adapters-v1"
MODEL_SCHEMA = "kws-native-asr-model-assets-v1"
RAW_SCHEMA = "en-market-native-asr-result-v1"
MAX_CALLS = 12
MAX_MODEL_BYTES = 8 * 1024**3
NAMES = ("qwen3-asr-en", "whisper-small-en", "parakeet-tdt-v3-en")
REL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_/-]{0,120}\Z")
FILE = re.compile(r"[a-zA-Z0-9_.-][a-zA-Z0-9_./-]{0,180}\Z")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def private_path(path: Path, message: str):
    market.need(path.is_absolute() and path.resolve() != english.ROOT and
                english.ROOT not in path.resolve().parents and
                not any(p.is_symlink() for p in (path, *path.parents)), message)


def verify_assets(pin: dict, lock_path: Path, model_root: Path) -> Path:
    runner = english.media()
    private_path(lock_path, "ASR_MODEL_LOCK_NOT_PRIVATE")
    _, lock_hash = runner._safe_file(lock_path, limit=1024 * 1024)
    market.need(lock_hash == pin["model_assets_sha256"],
                "ASR_MODEL_LOCK_SHA_CHANGED")
    lock = market.read(lock_path)
    market.need(type(lock) is dict and
                set(lock) == {"schema", "provider_id", "revision", "relative_dir",
                              "files"} and
                lock["schema"] == MODEL_SCHEMA and
                lock["provider_id"] == pin["provider_id"] and
                lock["revision"] == pin["model_revision_sha"],
                "ASR_MODEL_LOCK_IDENTITY")
    rel = lock["relative_dir"]
    market.need(type(rel) is str and REL.fullmatch(rel) is not None and
                ".." not in Path(rel).parts, "ASR_MODEL_RELATIVE_DIR")
    folder = model_root / rel
    market.need(folder.is_dir() and
                not any(p.is_symlink() for p in (folder, *folder.parents)),
                "ASR_MODEL_DIR_UNSAFE")
    files = lock["files"]
    market.need(type(files) is list and 1 <= len(files) <= 250, "ASR_MODEL_FILE_BUDGET")
    names, total = set(), 0
    for item in files:
        market.need(type(item) is dict and
                    set(item) == {"path", "bytes", "sha256"}, "ASR_MODEL_FILE_FIELDS")
        name = item["path"]
        market.need(type(name) is str and FILE.fullmatch(name) is not None and
                    ".." not in Path(name).parts and name not in names,
                    "ASR_MODEL_FILE_PATH")
        names.add(name)
        market.need(type(item["bytes"]) is int and
                    0 < item["bytes"] <= MAX_MODEL_BYTES and
                    type(item["sha256"]) is str and
                    market.HEX64.fullmatch(item["sha256"]) is not None,
                    "ASR_MODEL_FILE_PIN")
        n, actual = runner._safe_file(folder / name, limit=MAX_MODEL_BYTES)
        market.need(n == item["bytes"] and actual == item["sha256"],
                    "ASR_MODEL_FILE_CHANGED")
        total += n
        market.need(total <= MAX_MODEL_BYTES, "ASR_MODEL_BYTES_LIMIT")
    actual = set()
    for path in folder.rglob("*"):
        market.need(not path.is_symlink(), "ASR_MODEL_FILE_ALIAS")
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    market.need(actual == names, "ASR_MODEL_UNLISTED_OR_MISSING")
    return folder


def verify_adapters(plan: dict, lock_path: Path, model_root: Path) -> dict:
    runner = english.media()
    private_path(lock_path, "ASR_ADAPTER_LOCK_NOT_PRIVATE")
    lock = market.read(lock_path)
    market.need(set(lock) == {"schema", "adapters"} and
                lock["schema"] == ADAPTER_SCHEMA and
                type(lock["adapters"]) is list and len(lock["adapters"]) == 3,
                "ASR_ADAPTER_LOCK_SCHEMA")
    pins = {r["provider_id"]: r for r in plan["asr_pins"]}
    market.need(set(pins) == set(NAMES), "ENGLISH_THREE_MODEL_ROSTER_REQUIRED")
    adapters = {}
    for row in lock["adapters"]:
        market.need(type(row) is dict and set(row) == {
            "provider_id", "model_revision_sha", "model_assets_sha256",
            "model_manifest_path", "executable_path",
            "executable_sha256", "timeout_seconds"}, "ASR_ADAPTER_FIELDS")
        provider = row["provider_id"]
        market.need(provider in pins and provider not in adapters and
                    row["model_revision_sha"] == pins[provider]["model_revision_sha"] and
                    row["model_assets_sha256"] == pins[provider]["model_assets_sha256"],
                    "ASR_ADAPTER_PIN_MISMATCH")
        path = Path(row["executable_path"])
        private_path(path, "ASR_EXECUTABLE_NOT_PRIVATE")
        market.need(type(row["executable_sha256"]) is str and
                    market.HEX64.fullmatch(row["executable_sha256"]) is not None and
                    type(row["timeout_seconds"]) is int and
                    1 <= row["timeout_seconds"] <= 300,
                    "ASR_ADAPTER_TIMEOUT_OR_DIGEST")
        _, digest = runner._safe_file(path, limit=8 * 1024 * 1024)
        market.need(digest == row["executable_sha256"] and
                    os.access(path, os.X_OK), "ASR_EXECUTABLE_CHANGED_OR_NOT_EXECUTABLE")
        model_dir = verify_assets(pins[provider],
                                  Path(row["model_manifest_path"]), model_root)
        adapters[provider] = {
            "executable": path, "model_dir": model_dir,
            "model_id": pins[provider]["model_id"],
            "revision": pins[provider]["model_revision_sha"],
            "timeout": row["timeout_seconds"]}
    return adapters


def verify_capture(plan: dict, capture_root: Path, cell: dict) -> tuple[Path, dict] | None:
    """Missing genuine capture is not converted into fake input or a negative."""
    runner = english.media()
    lock = market.read(capture_root / "plan-lock.json")
    expected = {"policy": english.POLICY,
                "plan_sha256": sha(market.canonical(plan)),
                "model_calls_admitted_to_training": False}
    market.need(lock == expected, "SOURCE_CAPTURE_PLAN_DRIFT")
    folder = capture_root / "cells" / cell["cell_id"]
    completed = folder / "completed.json"
    if not completed.exists():
        return None
    market.need((folder / "started.json").is_file() and
                not folder.is_symlink(), "MISSING_SOURCE_STARTED_RECEIPT")
    record = market.read(completed)
    market.need(type(record) is dict and
                record.get("schema") == english.POLICY and
                record.get("cell_id") == cell["cell_id"] and
                record.get("plan_sha256") == expected["plan_sha256"] and
                record.get("status") == "GENERATED_UNADMITTED" and
                record.get("actual_text_verified") is False and
                record.get("training_admitted") is False and
                record.get("qualification_allowed") is False,
                "CAPTURE_NOT_ADMISSIBLE_FOR_ASR")
    wav = folder / "audio-16k.wav"
    size, digest = runner._safe_file(wav, limit=2 * 1024 * 1024)
    audio = record.get("audio")
    market.need(type(audio) is dict and
                audio.get("wav_sha256") == digest and
                audio.get("wav_bytes") == size,
                "CAPTURE_WAV_SHA_DRIFT")
    with wave.open(str(wav), "rb") as stream:
        count = stream.getnframes()
        market.need(stream.getnchannels() == 1 and stream.getsampwidth() == 2 and
                    stream.getframerate() == 16000 and
                    stream.getcomptype() == "NONE" and
                    count == audio.get("audio_frames") and
                    audio.get("sample_rate_hz") == 16000 and
                    0 < count <= 16000 * 12, "CAPTURE_WAV_GEOMETRY")
        raw_pcm = stream.readframes(count)
        market.need(len(raw_pcm) == 2 * count and
                    not stream.readframes(1) and
                    sha(raw_pcm) == audio.get("pcm_sha256"),
                    "CAPTURE_PCM_IDENTITY_DRIFT")
    return wav, {"wav_sha256": digest, "frames": count}


def native_result(path: Path, pin: dict, cell: dict, pcm: dict) -> dict:
    english.media()._safe_file(path, limit=65536)
    row = market.read(path)
    market.need(type(row) is dict and set(row) == {
        "schema", "provider_id", "model_id", "model_revision_sha",
        "audio_wav_sha256", "processed_start_frame", "processed_end_frame",
        "decoder_completed", "actual_text"}, "ASR_NATIVE_RESULT_FIELDS")
    market.need(row["schema"] == RAW_SCHEMA and
                row["provider_id"] == pin["provider_id"] and
                row["model_id"] == pin["model_id"] and
                row["model_revision_sha"] == pin["model_revision_sha"] and
                row["audio_wav_sha256"] == pcm["wav_sha256"] and
                row["processed_start_frame"] == 0 and
                row["processed_end_frame"] == pcm["frames"] and
                row["decoder_completed"] is True and
                type(row["actual_text"]) is str and
                0 < len(row["actual_text"]) <= 512 and
                not any(ord(char) < 32 for char in row["actual_text"]),
                "ASR_NATIVE_RESULT_INVALID")
    return row


def machine_screen(plan: dict, out: Path, cell: dict) -> dict | None:
    runner = english.media()
    results = []
    for provider in NAMES:
        file = out / "cells" / cell["cell_id"] / provider / "completed.json"
        if not file.exists():
            return None
        row = market.read(file)
        market.need(row.get("status") == "BLIND_ASR_UNADMITTED" and
                    row.get("training_admitted") is False and
                    row.get("provider_id") == provider, "ASR_SCREEN_SOURCE_DRIFT")
        results.append(row)
    folded = []
    for row in results:
        try:
            folded.append(market.normalize(row["actual_text"], "en-US"))
        except ValueError:
            folded.append(None)
    intended = market.normalize(cell["text_requested"], "en-US")
    same = len(set(folded)) == 1 and folded[0] is not None
    verdict = ("MACHINE_WAKE_MATCH_REVIEW_REQUIRED"
               if same and folded[0] == intended and cell["intent"] == "wake"
               else "MACHINE_NEGATIVE_PROMPT_REVIEW_REQUIRED"
               if same and folded[0] == intended and cell["intent"] != "wake"
               else "MACHINE_OTHER_OR_DISAGREE_REVIEW_REQUIRED")
    return {"policy": POLICY, "cell_id": cell["cell_id"],
            "asr_families": list(NAMES), "machine_verdict": verdict,
            "three_models_responded": True, "all_normalized_agree": same,
            "actual_label_authenticated": False, "human_gold": False,
            "training_admitted": False, "qualification_allowed": False}


def dispatch(plan: dict, capture_root: Path, out: Path, adapters: dict,
             adapter_hash: str, *, max_calls: int, executor=subprocess.run) -> dict:
    runner = english.media()
    market.need(type(max_calls) is int and 0 <= max_calls <= MAX_CALLS,
                "ASR_CALL_BUDGET")
    private_path(out, "PRIVATE_ASR_OUTPUT_REQUIRED")
    locked = {"policy": POLICY, "plan_sha256": sha(market.canonical(plan)),
              "adapter_lock_sha256": adapter_hash, "training_admitted": False}
    if not out.exists():
        market.need(out.parent.is_dir(), "ASR_OUTPUT_PARENT_MISSING")
        out.mkdir(mode=0o700)
        runner._once(out / "plan-lock.json", locked)
    else:
        market.need(out.is_dir() and
                    stat.S_IMODE(out.stat().st_mode) == 0o700 and
                    market.read(out / "plan-lock.json") == locked,
                    "ASR_RESUME_SOURCE_OR_ADAPTER_DRIFT")
    pins = {r["provider_id"]: r for r in plan["asr_pins"]}
    calls = 0
    for cell in plan["cells"]:
        source = verify_capture(plan, capture_root, cell)
        if source is None:
            continue
        wav, pcm = source
        for provider in NAMES:
            folder = out / "cells" / cell["cell_id"] / provider
            if (folder / "started.json").exists():
                continue
            if calls >= max_calls:
                break
            market.need(not folder.exists(), "ASR_UNRECORDED_ATTEMPT")
            folder.mkdir(parents=True, mode=0o700)
            runner._once(folder / "started.json", {
                "policy": POLICY, "cell_id": cell["cell_id"],
                "provider_id": provider, "wav_sha256": pcm["wav_sha256"],
                "adapter_lock_sha256": adapter_hash,
                "retry_count": 0, "training_admitted": False})
            calls += 1
            adapter = adapters[provider]
            target = folder / "native.json"
            cmd = [str(adapter["executable"]), "--audio", str(wav),
                   "--model-dir", str(adapter["model_dir"]),
                   "--output", str(target)]
            env = {"PATH": "/usr/bin:/bin", "HOME": str(folder),
                   "LANG": "C.UTF-8", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"}
            try:
                result = executor(cmd, cwd=str(folder), env=env,
                                  timeout=adapter["timeout"], close_fds=True,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, check=False)
                market.need(result.returncode == 0, "ASR_NATIVE_EXECUTION_FAILED")
                row = native_result(target, pins[provider], cell, pcm)
                runner._once(folder / "completed.json", {
                    "policy": POLICY, "cell_id": cell["cell_id"],
                    "provider_id": provider, "status": "BLIND_ASR_UNADMITTED",
                    "actual_text": row["actual_text"],
                    "wav_sha256": pcm["wav_sha256"],
                    "training_admitted": False, "human_gold": False})
            except Exception as exc:
                runner._once(folder / "failed.json", {
                    "policy": POLICY, "cell_id": cell["cell_id"],
                    "provider_id": provider, "status": "FAILED_NO_RETRY",
                    "error_type": type(exc).__name__,
                    "training_admitted": False})
        screen = machine_screen(plan, out, cell)
        if screen is not None:
            screen_path = out / "cells" / cell["cell_id"] / "machine-screen.json"
            if not screen_path.exists():
                runner._once(screen_path, screen)
        if calls >= max_calls:
            break
    return {"policy": POLICY, "asr_attempts_this_call": calls,
            "training_admitted": False, "human_gold": False,
            "actual_labels_verified": False}


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--request", required=True, type=Path)
    cli.add_argument("--capture-root", type=Path)
    cli.add_argument("--adapter-lock", type=Path)
    cli.add_argument("--model-root", type=Path)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--max-calls", type=int, default=9)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    try:
        plan = english.prepare(market.read(args.request))
        market.need(type(args.max_calls) is int and 0 <= args.max_calls <= MAX_CALLS,
                    "ASR_CALL_BUDGET")
        if not args.execute:
            print(json.dumps({"policy": POLICY, "max_calls": args.max_calls,
                              "asr_calls": 0, "training_admitted": False}))
            return 0
        market.need(args.capture_root is not None and args.adapter_lock is not None and
                    args.model_root is not None and args.out is not None and
                    os.environ.get("HF_HUB_OFFLINE") == "1" and
                    os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                    "ASR_EXPLICIT_OFFLINE_EXECUTION_REQUIRED")
        adapters = verify_adapters(plan, args.adapter_lock, args.model_root)
        adapter_hash = sha(args.adapter_lock.read_bytes())
        report = dispatch(plan, args.capture_root, args.out, adapters,
                          adapter_hash, max_calls=args.max_calls)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, ValueError, wave.Error) as error:
        print("ENGLISH_ASR_STOP: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
