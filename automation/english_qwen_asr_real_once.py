"""One-shot genuine Qwen3-ASR-0.6B English observation on retained real TTS WAVs.

Uses only the original 10-WAV GitHub artifact. Pinned pretrained model
checkpoint bytes, offline transcribe, no wake text to ASR and no training
admission. Model loads once; unknown calls are not silently replayed.
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
import wave

import english_blind_asr as blind
import english_qwen_capture as english
import english_qwen_real_once as pilot
import multilingual_campaign as market

POLICY = "english-qwen-asr-real-once-20261112-v1"
MODEL = "Qwen/Qwen3-ASR-0.6B"
REV = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
PROVIDER = "qwen3-asr-en"
SOURCE_RUN_ID = 43214159818
SOURCE_HEAD = "f2b91001e4fd56a03bbce9bc6b0c1cae7fad1dc8"
SOURCE_ARTIFACT_ID = 13942568634
SOURCE_ARTIFACT_DIGEST = "0235b0064006f79bfe9c8fa551c2a1e41da61c86e0fd987a90ee2d47842a9f21"
MAX_CALLS = 10
MAX_DOWNLOAD_BYTES = 3 * 1024**3
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-qwen-asr-real-20261112.json"
ASSETS = [
  {
    "path": "chat_template.json",
    "bytes": 1161,
    "sha256": "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff"
  },
  {
    "path": "config.json",
    "bytes": 6193,
    "sha256": "76d3ae4601ce939830b2517f4a6cadb86cc51316c3900af6b020b051c21a478c"
  },
  {
    "path": "generation_config.json",
    "bytes": 142,
    "sha256": "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c"
  },
  {
    "path": "merges.txt",
    "bytes": 1671853,
    "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
  },
  {
    "path": "model.safetensors",
    "bytes": 1876091704,
    "sha256": "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
  },
  {
    "path": "preprocessor_config.json",
    "bytes": 330,
    "sha256": "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d"
  },
  {
    "path": "tokenizer_config.json",
    "bytes": 12487,
    "sha256": "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c"
  },
  {
    "path": "vocab.json",
    "bytes": 2776833,
    "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
  }
]

def need(condition, name):
    if not condition:
        raise ValueError(name)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def request() -> dict:
    claim = market.read(TRIGGER)
    expected = {
        "schema": POLICY, "source_run_id": SOURCE_RUN_ID,
        "source_head": SOURCE_HEAD,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_sha256": SOURCE_ARTIFACT_DIGEST,
        "model_id": MODEL, "model_revision": REV,
        "max_asr_calls": MAX_CALLS, "training_admitted": False,
        "three_asr_completed": False,
    }
    need(claim == expected, "ASR_SINGLE_CLAIM_CHANGED")
    return expected


def verified_source(root: Path) -> tuple[dict, list[tuple[dict, Path, dict]]]:
    """Match every original source cell and WAV; no hidden missing samples."""
    blind.private_path(root, "SOURCE_ROOT_NOT_PRIVATE")
    plan = blind.load_pilot_plan(root / "source-plan.json")
    summary = market.read(root / "summary.json")
    need(summary.get("schema") == pilot.POLICY and
         summary.get("status") == "COMPLETE" and
         summary.get("generated_audio_count") == MAX_CALLS and
         summary.get("asr_calls") == 0 and
         summary.get("training_admitted") is False,
         "TTS_SOURCE_NOT_FULLY_COMPLETE")
    reports = summary.get("source_rows")
    need(type(reports) is list and len(reports) == MAX_CALLS and
         {r["cell_id"] for r in reports} == {c["cell_id"] for c in plan["cells"]} and
         all(row.get("status") == "GENERATED_UNADMITTED" and
             row.get("training_admitted") is False for row in reports),
         "TTS_SOURCE_ROWS_INVALID")
    by_id = {r["cell_id"]: r for r in reports}
    ordered = []
    for cell in plan["cells"]:
        original = blind.verify_capture(plan, root / "capture", cell)
        need(original is not None, "SOURCE_WAV_OR_RECEIPT_MISSING")
        wav, pcm = original
        claimed = by_id[cell["cell_id"]]
        need(claimed.get("wav_sha256") == pcm["wav_sha256"],
             "SOURCE_SUMMARY_WAV_BINDING_DRIFT")
        ordered.append((cell, wav, pcm))
    return plan, ordered


def lock_model(root: Path) -> Path:
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL, revision=REV, files_metadata=True, token=False)
    need(info.sha == REV and info.private is False and
         not getattr(info, "gated", False), "PINNED_QWEN_ASR_UPSTREAM")
    total = sum(f["bytes"] for f in ASSETS)
    need(total <= MAX_DOWNLOAD_BYTES and
         shutil.disk_usage(root).free >= total + 2 * 1024**3,
         "ASR_DOWNLOAD_OR_DISK_BUDGET")
    model_dir = root / "model"
    need(not model_dir.exists(), "MODEL_PREEXISTS")
    model_dir.mkdir(mode=0o700)
    result = snapshot_download(repo_id=MODEL, revision=REV,
                               allow_patterns=[f["path"] for f in ASSETS],
                               local_dir=str(model_dir), token=False)
    need(Path(result).resolve() == model_dir.resolve(), "ASR_MODEL_PATH_CHANGED")
    cache = model_dir / ".cache"
    if cache.exists():
        need(cache.is_dir() and not cache.is_symlink(), "ASR_CACHE_ALIAS")
        shutil.rmtree(cache)
    found = set()
    for path in model_dir.rglob("*"):
        need(not path.is_symlink(), "ASR_MODEL_SYMLINK")
        if path.is_file():
            found.add(path.relative_to(model_dir).as_posix())
    need(found == {f["path"] for f in ASSETS}, "ASR_MODEL_FILE_SET_CHANGED")
    for item in ASSETS:
        n, digest = english.media()._safe_file(model_dir / item["path"],
                                              limit=MAX_DOWNLOAD_BYTES)
        need(n == item["bytes"] and digest == item["sha256"],
             "ASR_MODEL_PINNED_BYTES_MISMATCH")
    return model_dir


def run(source: Path, output: Path) -> dict:
    _ = request()
    need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_REF") == "refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
         "REAL_ASR_ONCE_MAIN_ONLY")
    blind.private_path(output, "REAL_ASR_OUTPUT_NOT_PRIVATE")
    need(not output.exists() and output.parent.is_dir(), "ASR_OUTPUT_CREATE_ONCE")
    plan, cells = verified_source(source)
    output.mkdir(mode=0o700)
    english.media()._once(output / "started.json", {
        "schema": POLICY, "source_run_id": SOURCE_RUN_ID,
        "source_head": SOURCE_HEAD, "source_plan_sha256": sha(market.canonical(plan)),
        "source_artifact_sha256": SOURCE_ARTIFACT_DIGEST,
        "model_revision": REV, "run_id": os.environ.get("GITHUB_RUN_ID"),
        "training_admitted": False, "no_retry": True})
    model_root = output / "model-assets"
    model_root.mkdir(mode=0o700)
    total_attempts = 0
    rows = []
    fatal = None
    try:
        directory = lock_model(model_root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from qwen_asr import Qwen3ASRModel
        torch.set_num_threads(min(os.cpu_count() or 1, 4))
        model = Qwen3ASRModel.from_pretrained(
            str(directory), dtype=torch.float32, device_map="cpu",
            max_inference_batch_size=1, max_new_tokens=128,
            attn_implementation="sdpa")
        for cell, wav, pcm in cells:
            receipt = output / "cells" / cell["cell_id"]
            receipt.mkdir(parents=True, mode=0o700)
            english.media()._once(receipt / "started.json", {
                "schema": POLICY, "cell_id": cell["cell_id"],
                "provider_id": PROVIDER, "wav_sha256": pcm["wav_sha256"],
                "model_revision_sha": REV, "retry_count": 0,
                "training_admitted": False})
            total_attempts += 1
            status = "FAILED_NO_RETRY"
            try:
                result = model.transcribe(audio=str(wav), language="English", context="")
                need(type(result) is list and len(result) == 1 and
                     type(result[0].text) is str and
                     0 < len(result[0].text) <= 512, "ASR_EMPTY_OR_INVALID_TEXT")
                transcript = result[0].text
                native = {
                    "schema": blind.RAW_SCHEMA, "provider_id": PROVIDER,
                    "model_id": MODEL, "model_revision_sha": REV,
                    "audio_wav_sha256": pcm["wav_sha256"],
                    "processed_start_frame": 0, "processed_end_frame": pcm["frames"],
                    "decoder_completed": True, "actual_text": transcript}
                english.media()._once(receipt / "native.json", native)
                need(blind.native_result(receipt / "native.json",
                    {"provider_id": PROVIDER, "model_id": MODEL,
                     "model_revision_sha": REV},cell,pcm) == native,
                     "NATIVE_RESULT_VERIFY_FAILED")
                english.media()._once(receipt / "completed.json", {
                    "schema": POLICY, "status": "BLIND_ASR_UNADMITTED",
                    "cell_id": cell["cell_id"], "provider_id": PROVIDER,
                    "wav_sha256": pcm["wav_sha256"],
                    "native_sha256": sha((receipt / "native.json").read_bytes()),
                    "training_admitted": False, "human_gold": False})
                status = "BLIND_ASR_UNADMITTED"
            except Exception as e:
                english.media()._once(receipt / "failed.json", {
                    "schema": POLICY, "status": "FAILED_NO_RETRY",
                    "cell_id": cell["cell_id"], "error_type": type(e).__name__,
                    "training_admitted": False})
            rows.append({"cell_id":cell["cell_id"],"status":status,
                         "wav_sha256":pcm["wav_sha256"]})
    except BaseException as e:
        fatal = type(e).__name__
    finally:
        status = "COMPLETE" if fatal is None and len(rows) == MAX_CALLS and all(
            x["status"] == "BLIND_ASR_UNADMITTED" for x in rows) else "FAILED_OR_PARTIAL_NO_RETRY"
        result = {
            "schema": POLICY, "status": status,
            "qwen_asr_package": (version("qwen-asr") if total_attempts else None),
            "source_audio_count": MAX_CALLS, "asr_attempts": total_attempts,
            "completed_asr": sum(x["status"] == "BLIND_ASR_UNADMITTED" for x in rows),
            "model_id": MODEL, "model_revision_sha": REV,
            "source_plan_sha256": sha(market.canonical(plan)),
            "source_run_id": SOURCE_RUN_ID, "rows": rows,
            "error_type": fatal, "three_asr_completed": False,
            "training_admitted": False, "human_gold": False}
        english.media()._once(output / "summary.json", result)
        if model_root.exists():
            shutil.rmtree(model_root)
    return result


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-root", type=Path)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    try:
        request()
        if not args.execute:
            print(json.dumps({"schema": POLICY, "expected_source_audio": MAX_CALLS,
                              "asr_calls": 0, "training_admitted": False}))
            return 0
        need(args.source_root is not None and args.output is not None,
             "EXPLICIT_REAL_ASR_PATHS")
        result = run(args.source_root, args.output)
        print(json.dumps({k: result[k] for k in ("status", "asr_attempts",
                         "completed_asr", "training_admitted")}, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except (ValueError, OSError, wave.Error) as exc:
        print("ASR_CAPTURE_STOP: " + type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
