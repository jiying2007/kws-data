"""Bounded English-market Qwen CustomVoice capture; model execution is opt-in.

Consumes a private current-market request, checks exact local model bytes, writes
private audio/receipts and NEVER retries started cells. Output is not admitted.
Chinese campaigns and historical Qwen batches are not read or regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

import multilingual_campaign as market

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "research" / "auto-tts-voice-clone-20261102"
MODEL_SCHEMA = "qwen3-tts-offline-asset-lock-v1"
POLICY = "en-market-qwen-customvoice-capture-v1"
MAX_ATTEMPTS = 16
SPEAKERS = {
    "aiden": "Aiden", "dylan": "Dylan", "eric": "Eric",
    "ono_anna": "Ono_Anna", "ryan": "Ryan", "serena": "Serena",
    "sohee": "Sohee", "uncle_fu": "Uncle_Fu", "vivian": "Vivian",
}
REL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_/-]{0,120}\Z")
FILE = re.compile(r"[a-zA-Z0-9_.-][a-zA-Z0-9_./-]{0,180}\Z")


def media():
    if str(LEGACY) not in sys.path:
        sys.path.insert(0, str(LEGACY))
    import runner
    return runner


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def prepare(request: dict) -> dict:
    """Recompute the real planner's exact current English roster."""
    plan = market.plan(request)
    market.require_market_candidates(request)
    market.need(plan["locale"] == "en-US" and
                len(plan["tts_pins"]) == 1 and
                plan["tts_pins"][0]["provider_id"] == "qwen3-custom-en",
                "ENGLISH_QWEN_PRESET_ONLY")
    for voice in plan["voice_sources"]:
        preset = SPEAKERS.get(voice["id"])
        market.need(preset is not None and
                    voice["source_family_id"] == "qwen3-preset-" + preset.lower() and
                    voice["origin_kind"] == "owned-synthetic" and
                    voice["reference_pcm_sha256"] is None and
                    voice["rights_evidence_sha256"] is None,
                    "SYNTHETIC_PRESET_SOURCE_IDENTITY_REQUIRED")
    plan["market_candidate_policy"] = market.MARKET_CANDIDATE_POLICY
    plan["market_candidate_roster_verified"] = True
    return plan


def verify_model(plan: dict, lock_path: Path, asset_root: Path) -> Path:
    """Verify exact frozen local Qwen bytes; never download a model."""
    runner = media()
    _, lock_sha = runner._safe_file(lock_path, limit=1024 * 1024)
    model = plan["tts_pins"][0]
    market.need(lock_sha == model["model_assets_sha256"],
                "MODEL_LOCK_SHA_NOT_PINNED")
    lock = market.read(lock_path)
    market.need(set(lock) == {"schema", "models"} and
                lock["schema"] == MODEL_SCHEMA and
                type(lock["models"]) is list and len(lock["models"]) == 1,
                "MODEL_LOCK_SCHEMA")
    row = lock["models"][0]
    market.need(type(row) is dict and
                set(row) == {"model_id", "revision", "relative_dir", "files"} and
                row["model_id"] == model["model_id"] and
                row["revision"] == model["model_revision_sha"],
                "MODEL_IDENTITY_DRIFT")
    rel = row["relative_dir"]
    market.need(type(rel) is str and REL.fullmatch(rel) is not None and
                ".." not in Path(rel).parts, "MODEL_RELATIVE_DIR")
    folder = asset_root / rel
    market.need(folder.is_dir() and
                not any(p.is_symlink() for p in (folder, *folder.parents)),
                "MODEL_FOLDER_UNSAFE")
    files = row["files"]
    market.need(type(files) is list and 1 <= len(files) <= 250, "MODEL_FILES_LIMIT")
    names, total = set(), 0
    for item in files:
        market.need(type(item) is dict and
                    set(item) == {"path", "bytes", "sha256"}, "MODEL_FILE_FIELDS")
        name = item["path"]
        market.need(type(name) is str and FILE.fullmatch(name) is not None and
                    ".." not in Path(name).parts and name not in names,
                    "MODEL_FILE_PATH")
        names.add(name)
        market.need(type(item["bytes"]) is int and 0 < item["bytes"] <= 8 * 1024**3 and
                    type(item["sha256"]) is str and
                    market.HEX64.fullmatch(item["sha256"]) is not None,
                    "MODEL_FILE_PIN")
        size, digest = runner._safe_file(folder / name, limit=8 * 1024**3)
        market.need(size == item["bytes"] and digest == item["sha256"],
                    "MODEL_FILE_BYTES_CHANGED")
        total += size
        market.need(total <= 8 * 1024**3, "MODEL_BYTES_LIMIT")
    actual = set()
    for file in folder.rglob("*"):
        market.need(not file.is_symlink(), "MODEL_FILE_ALIAS")
        if file.is_file():
            actual.add(file.relative_to(folder).as_posix())
    market.need(actual == names, "MODEL_UNLISTED_OR_MISSING_FILES")
    return folder


class QwenEnglish:
    """Real local Qwen inference; imported only inside an explicit execute lane."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.model = None

    def synthesize(self, cell: dict, speaker: str):
        import torch
        from qwen_tts import Qwen3TTSModel
        if self.model is None:
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            self.model = Qwen3TTSModel.from_pretrained(
                str(self.folder), device_map="cpu", dtype=torch.float32,
                attn_implementation="sdpa")
        torch.manual_seed(int(sha(cell["cell_id"].encode("ascii"))[:12], 16))
        wavs, rate = self.model.generate_custom_voice(
            text=cell["text_requested"], language="English", speaker=speaker)
        market.need(len(wavs) == 1, "QWEN_ONE_AUDIO_PER_CELL")
        return wavs[0], int(rate)


def capture(plan: dict, root: Path, backend, *, max_calls: int, writer=None) -> dict:
    """Create-once private results. Failure or interruption consumes the cell."""
    runner = media()
    market.need(type(max_calls) is int and 0 <= max_calls <= MAX_ATTEMPTS,
                "CAPTURE_CALL_BUDGET")
    market.need(root.is_absolute() and root.resolve() != ROOT and
                ROOT not in root.resolve().parents and
                not any(p.is_symlink() for p in (root, *root.parents)),
                "PRIVATE_OUTPUT_REQUIRED")
    locked = {"policy": POLICY, "plan_sha256": sha(market.canonical(plan)),
              "model_calls_admitted_to_training": False}
    if not root.exists():
        market.need(root.parent.is_dir(), "OUTPUT_PARENT_NOT_FOUND")
        root.mkdir(mode=0o700)
        runner._once(root / "plan-lock.json", locked)
    else:
        market.need(root.is_dir() and
                    stat.S_IMODE(root.stat().st_mode) == 0o700 and
                    market.read(root / "plan-lock.json") == locked,
                    "CAPTURE_RESUME_IDENTITY_DRIFT")
    if writer is None:
        writer = runner._write_waves
    voice = {v["id"]: SPEAKERS[v["id"]] for v in plan["voice_sources"]}
    attempts = 0
    for cell in plan["cells"]:
        folder = root / "cells" / cell["cell_id"]
        if (folder / "started.json").exists():
            continue
        if attempts >= max_calls:
            break
        market.need(not folder.exists(), "UNRECORDED_ATTEMPT_DIRECTORY")
        folder.mkdir(parents=True, mode=0o700)
        runner._once(folder / "started.json", {
            "schema": POLICY, "cell_id": cell["cell_id"],
            "plan_sha256": locked["plan_sha256"], "model_calls": 1,
            "source_family_id": cell["source_family_id"],
            "retry_count": 0, "training_admitted": False})
        attempts += 1
        try:
            samples, rate = backend.synthesize(cell, voice[cell["voice_id"]])
            metrics = runner.quality(samples, rate)
            assets = writer(folder, samples, rate, metrics)
            market.need(type(assets) is dict and
                        all(type(assets.get(k)) is str and
                            market.HEX64.fullmatch(assets[k]) is not None
                            for k in ("native_sha256", "wav_sha256", "pcm_sha256")) and
                        type(assets.get("audio_frames")) is int and
                        assets["audio_frames"] > 0 and
                        type(assets.get("wav_bytes")) is int and
                        assets["wav_bytes"] > 44 and
                        assets.get("sample_rate_hz") == 16000,
                        "AUDIO_OUTPUT_RECEIPT")
            runner._once(folder / "completed.json", {
                "schema": POLICY, "cell_id": cell["cell_id"],
                "plan_sha256": locked["plan_sha256"], "status": "GENERATED_UNADMITTED",
                "technical": metrics, "audio": assets, "actual_text_verified": False,
                "human_gold": False, "training_admitted": False,
                "qualification_allowed": False})
        except Exception as exc:
            runner._once(folder / "failed.json", {
                "schema": POLICY, "cell_id": cell["cell_id"],
                "status": "FAILED_NO_RETRY", "error_type": type(exc).__name__,
                "training_admitted": False})
    return {"policy": POLICY, "attempts_this_call": attempts,
            "total_started": sum((root / "cells" / x["cell_id"] / "started.json").exists()
                                 for x in plan["cells"]),
            "total_completed": sum((root / "cells" / x["cell_id"] / "completed.json").exists()
                                   for x in plan["cells"]),
            "asr_calls": 0, "training_admitted": False}


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--request", required=True, type=Path)
    cli.add_argument("--model-lock", type=Path)
    cli.add_argument("--model-root", type=Path)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--max-calls", type=int, default=8)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    try:
        plan = prepare(market.read(args.request))
        market.need(type(args.max_calls) is int and 0 <= args.max_calls <= MAX_ATTEMPTS,
                    "CAPTURE_CALL_BUDGET")
        if not args.execute:
            print(json.dumps({"policy": POLICY, "planned_cells": len(plan["cells"]),
                              "model_calls": 0, "training_admitted": False}))
            return 0
        market.need(args.model_lock is not None and args.model_root is not None and
                    args.out is not None and
                    os.environ.get("HF_HUB_OFFLINE") == "1" and
                    os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                    "EXPLICIT_OFFLINE_EXECUTION_REQUIRED")
        folder = verify_model(plan, args.model_lock, args.model_root)
        print(json.dumps(capture(plan, args.out, QwenEnglish(folder),
                                 max_calls=args.max_calls), sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        print("ENGLISH_CAPTURE_STOP: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
