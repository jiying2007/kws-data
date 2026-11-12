"""One-shot, public-GitHub-runner Qwen English audio pilot; no labels or training.

Exact upstream model commit, stable preset ancestry, bounded calls, immutable
receipts and no implicit retries. The full ASR and training chains are separate.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import english_qwen_capture as eng
import multilingual_campaign as market

POLICY = "english-qwen-real-capture-once-20261112-v1"
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-qwen-real-20261112.json"
MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
REV = "85e237c12c027371202489a0ec509ded67b5e4b5"
SPEAKERS = (("ryan", "train-candidate"), ("aiden", "development-only"))
PROMPTS = (("Hey Vosen", "wake", 1), ("Hey Mero", "wake", 2),
           ("Hey Wosen", "near_miss", None), ("Hey Miro", "near_miss", None),
           ("What time is it?", "ordinary_negative", None))
MAX_CALLS = len(SPEAKERS) * len(PROMPTS)
MAX_DOWNLOAD_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 2 * 1024**3


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(ok, message: str):
    if not ok:
        raise ValueError(message)


def read_trigger() -> dict:
    value = market.read(TRIGGER)
    require(set(value) == {"schema", "model_id", "model_revision", "wake_phrases",
                           "max_tts_calls", "training_admitted", "asr_calls"},
            "TRIGGER_FIELDS")
    require(value == {
        "schema": POLICY, "model_id": MODEL, "model_revision": REV,
        "wake_phrases": ["Hey Vosen", "Hey Mero"], "max_tts_calls": MAX_CALLS,
        "training_admitted": False, "asr_calls": 0}, "TRIGGER_POLICY_CHANGED")
    return value


def pilot_plan(lock_sha: str) -> dict:
    market.need(market.HEX64.fullmatch(lock_sha) is not None, "MODEL_LOCK_REQUIRED")
    voices = [{"id": preset, "source_family_id": "qwen3-preset-" + preset,
               "role": role, "origin_kind": "owned-synthetic",
               "reference_pcm_sha256": None, "rights_evidence_sha256": None}
              for preset, role in SPEAKERS]
    cells = []
    for voice in voices:
        for phrase, intent, index in PROMPTS:
            params = {"policy": POLICY, "voice_id": voice["id"],
                      "source_family_id": voice["source_family_id"],
                      "text": phrase, "model_revision": REV}
            cells.append({
                "cell_id": "en-" + digest(market.canonical(params))[:24],
                "voice_id": voice["id"], "source_family_id": voice["source_family_id"],
                "role": voice["role"], "text_requested": phrase,
                "intent": intent, "requested_keyword_index": index})
    require(len(cells) == MAX_CALLS and len({x["cell_id"] for x in cells}) == MAX_CALLS,
            "CELL_IDENTITY")
    return {
        "schema": POLICY, "locale": "en-US",
        "wake_phrases_requested": ["Hey Vosen", "Hey Mero"],
        "voice_sources": voices, "cells": cells,
        "tts_pins": [{"provider_id": "qwen3-custom-en", "model_id": MODEL,
                      "model_revision_sha": REV, "model_assets_sha256": lock_sha}],
        "asr_status": "NOT_RUN_NO_ASR_PINS_IN_TTS_ONLY_STAGE",
        "asr_calls": 0, "human_gold": False,
        "training_admitted": False, "qualification_allowed": False,
    }


def materialize_model(root: Path):
    """Download a single public immutable revision with a strict byte budget."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(repo_id=MODEL, revision=REV, files_metadata=True,
                              token=False)
    require(info.sha == REV and info.private is False and
            not getattr(info, "gated", False), "UPSTREAM_REVISION_OR_RIGHTS")
    sizes = [getattr(file, "size", None) for file in (info.siblings or ())]
    require(sizes and all(type(x) is int and 0 <= x <= MAX_DOWNLOAD_BYTES for x in sizes),
            "UPSTREAM_SIZE_METADATA")
    require(sum(sizes) <= MAX_DOWNLOAD_BYTES, "DOWNLOAD_BUDGET")
    require(shutil.disk_usage(root).free >= sum(sizes) + MIN_DISK_FREE_BYTES,
            "DISK_BUDGET")
    folder = root / "preset"
    require(not folder.exists(), "MODEL_DOWNLOAD_ONCE")
    folder.mkdir(mode=0o700)
    path = snapshot_download(repo_id=MODEL, revision=REV, local_dir=str(folder),
                             token=False)
    require(Path(path).resolve() == folder.resolve(), "MODEL_ROOT_DRIFT")
    cache = folder / ".cache"
    if cache.exists():
        require(cache.is_dir() and not cache.is_symlink(), "MODEL_CACHE_UNSAFE")
        shutil.rmtree(cache)
    rows = []
    for f in sorted(folder.rglob("*")):
        require(not f.is_symlink(), "MODEL_FILE_LINK")
        if f.is_dir():
            continue
        require(f.is_file() and ".cache" not in f.relative_to(folder).parts,
                "MODEL_FILE_UNSAFE")
        b, checksum = eng.media()._safe_file(f, limit=MAX_DOWNLOAD_BYTES)
        rows.append({"path": f.relative_to(folder).as_posix(), "bytes": b,
                     "sha256": checksum})
    require(1 <= len(rows) <= 250 and sum(x["bytes"] for x in rows) <= MAX_DOWNLOAD_BYTES,
            "MODEL_SNAPSHOT_INCOMPLETE")
    return {"schema": eng.MODEL_SCHEMA, "models": [{
        "model_id": MODEL, "revision": REV, "relative_dir": "preset",
        "files": rows}]}


def run(root: Path) -> dict:
    _ = read_trigger()
    require(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
            os.environ.get("GITHUB_REF") == "refs/heads/main" and
            os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
            "GITHUB_SINGLE_EXECUTION_ONLY")
    require(root.is_absolute() and root != eng.ROOT and
            eng.ROOT not in root.parents and not root.exists() and
            root.parent.is_dir() and not root.parent.is_symlink(),
            "NEW_PRIVATE_OUTPUT_ONLY")
    root.mkdir(mode=0o700)
    eng.media()._once(root / "started.json", {
        "schema": POLICY, "status": "started",
        "model_revision": REV, "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "trigger_sha256": digest(TRIGGER.read_bytes()),
        "no_retry": True, "training_admitted": False})
    model_dir = root / "models"
    model_dir.mkdir(mode=0o700)
    receipt = None
    try:
        lock = materialize_model(model_dir)
        lock_path = root / "model-lock.json"
        eng.media()._once(lock_path, lock)
        plan = pilot_plan(digest(lock_path.read_bytes()))
        eng.verify_model(plan, lock_path, model_dir)
        eng.media()._once(root / "source-plan.json", plan)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        captured = eng.capture(plan, root / "capture",
                               eng.QwenEnglish(model_dir / "preset"),
                               max_calls=MAX_CALLS)
        rows = []
        for cell in plan["cells"]:
            folder = root / "capture" / "cells" / cell["cell_id"]
            completed = folder / "completed.json"
            failed = folder / "failed.json"
            state = "GENERATED_UNADMITTED" if completed.is_file() else (
                "FAILED_NO_RETRY" if failed.is_file() else "NOT_RUN")
            item = {"cell_id": cell["cell_id"], "source_family_id": cell["source_family_id"],
                    "intent": cell["intent"], "status": state,
                    "training_admitted": False}
            if completed.is_file():
                row = market.read(completed)
                item["wav_sha256"] = row["audio"]["wav_sha256"]
                item["pcm_sha256"] = row["audio"]["pcm_sha256"]
            rows.append(item)
        ok = sum(x["status"] == "GENERATED_UNADMITTED" for x in rows)
        receipt = {
            "schema": POLICY, "status": "COMPLETE" if ok == MAX_CALLS
            else "INCOMPLETE_NO_RETRY", "generated_audio_count": ok,
            "planned_audio_count": MAX_CALLS, "asr_calls": 0,
            "source_rows": rows, "capture_receipt": captured,
            "model_lock_sha256": digest(lock_path.read_bytes()),
            "qwen_tts_package": version("qwen-tts"),
            "training_admitted": False, "human_gold": False}
    except BaseException as error:
        receipt = {"schema": POLICY, "status": "FAILED_NO_RETRY",
                   "error_type": type(error).__name__, "generated_audio_count": 0,
                   "asr_calls": 0, "training_admitted": False,
                   "human_gold": False}
        eng.media()._once(root / "failed.json", receipt)
        raise
    finally:
        if receipt is not None:
            eng.media()._once(root / "summary.json", receipt)
        # Model weights must not enter the public audio artifact.
        if model_dir.exists():
            shutil.rmtree(model_dir)
    return receipt


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    try:
        trigger = read_trigger()
        if not args.execute:
            print(json.dumps({"schema": POLICY, "planned_audio_count": MAX_CALLS,
                              "model_revision": trigger["model_revision"],
                              "actual_model_calls": 0, "training_admitted": False}))
            return 0
        require(args.out is not None, "EXPLICIT_OUTPUT")
        result = run(args.out)
        print(json.dumps({k: result[k] for k in
                         ("status", "generated_audio_count", "asr_calls",
                          "training_admitted")}, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except (ValueError, OSError) as error:
        print("ENGLISH_PILOT_STOP: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
