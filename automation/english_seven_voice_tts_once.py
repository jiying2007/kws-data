"""One-shot multi-voice Qwen English KWS source expansion, without training labels.

Seven new Qwen preset voice families, never the two consumed by original pilot.
The 63 new generation cells are split into create-only 16-call stages; failed or
interrupted cells stay consumed. No existing WAV is regenerated or relabeled.
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

import english_qwen_capture as capture
import english_qwen_real_once as pilot
import multilingual_campaign as market

POLICY = "english-seven-voice-real-tts-20261112-v1"
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-seven-voice-tts-20261112.json"
ORIGINAL_TTS_RUN = 43214159818
ORIGINAL_TTS_AUDIO_COUNT = 10
VOICES = (
    ("dylan", "train-candidate"),
    ("eric", "train-candidate"),
    ("ono_anna", "train-candidate"),
    ("sohee", "train-candidate"),
    ("vivian", "train-candidate"),
    ("serena", "development-only"),
    ("uncle_fu", "development-only"),
)
# Punctuation variants are extra acoustic requests, not additional independent voices.
# Near-miss intent stays unverified and MUST NOT be promoted to a trusted negative.
PROMPTS = (
    ("Hey Vosen", "wake", 1, "plain"),
    ("Hey, Vosen!", "wake", 1, "punctuated"),
    ("Hey Mero", "wake", 2, "plain"),
    ("Hey, Mero!", "wake", 2, "punctuated"),
    ("Hey Wosen", "near_miss", None, "other_confusable"),
    ("Hey Miro", "near_miss", None, "other_confusable"),
    ("Hey Rosen", "near_miss", None, "other_confusable"),
    ("What time is it?", "ordinary_negative", None, "conversation"),
    ("Hey, can you hear me?", "ordinary_negative", None, "conversation"),
)
MAX_CALLS = len(VOICES) * len(PROMPTS)
MAX_PER_STAGE = capture.MAX_ATTEMPTS
STAGES = (16, 16, 16, 15)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def need(ok, error: str):
    if not ok:
        raise ValueError(error)


def claim() -> dict:
    expected = {
        "schema": POLICY,
        "model_id": pilot.MODEL, "model_revision": pilot.REV,
        "wake_phrases": ["Hey Vosen", "Hey Mero"],
        "original_tts_run_id": ORIGINAL_TTS_RUN,
        "original_audio_reused_not_regenerated": ORIGINAL_TTS_AUDIO_COUNT,
        "new_preset_voices": [v[0] for v in VOICES],
        "max_new_tts_cells": MAX_CALLS,
        "stage_budgets": list(STAGES),
        "training_admitted": False, "asr_calls": 0,
    }
    need(market.read(TRIGGER) == expected, "SEVEN_VOICE_TRIGGER_DRIFT")
    need(MAX_CALLS == 63 and sum(STAGES) == MAX_CALLS and
         len(STAGES) == 4 and max(STAGES) <= MAX_PER_STAGE,
         "SEVEN_VOICE_BUDGET_DRIFT")
    return expected


def plan(model_lock_sha256: str) -> dict:
    need(type(model_lock_sha256) is str and
         market.HEX64.fullmatch(model_lock_sha256) is not None,
         "REAL_MODEL_LOCK_REQUIRED")
    old = {p for p, _ in pilot.SPEAKERS}
    new = {p for p, _ in VOICES}
    need(not old.intersection(new) and len(new) == 7 and
         all(p in capture.SPEAKERS for p in new),
         "CONSUMED_PILOT_SPEAKER_OR_UNKNOWN_PRESET")
    need(tuple(market.normalize(w, "en-US") for w in ("Hey Vosen", "Hey Mero"))
         == market.MARKET_ENGLISH_CANDIDATES,
         "ENGLISH_WAKE_ROSTER_DRIFT")
    voices = [{
        "id": p, "source_family_id": "qwen3-preset-" + p,
        "role": role, "origin_kind": "owned-synthetic",
        "reference_pcm_sha256": None, "rights_evidence_sha256": None
    } for p, role in VOICES]
    cells = []
    for v in voices:
        for phrase, intent, index, subtype in PROMPTS:
            identity = {
                "schema": POLICY, "voice_id": v["id"], "text": phrase,
                "model_revision": pilot.REV, "source_family_id": v["source_family_id"],
            }
            cells.append({
                "cell_id": "en-" + sha(market.canonical(identity))[:24],
                "voice_id": v["id"], "source_family_id": v["source_family_id"],
                "role": v["role"], "text_requested": phrase, "intent": intent,
                "requested_keyword_index": index, "challenge": subtype,
            })
    need(len(cells) == MAX_CALLS and
         len({r["cell_id"] for r in cells}) == MAX_CALLS,
         "DUPLICATE_SEVEN_VOICE_CELL")
    return {
        "schema": POLICY, "locale": "en-US",
        "wake_phrases_requested": ["Hey Vosen", "Hey Mero"],
        "voice_sources": voices, "cells": cells,
        "tts_pins": [{"provider_id": "qwen3-custom-en",
                      "model_id": pilot.MODEL,
                      "model_revision_sha": pilot.REV,
                      "model_assets_sha256": model_lock_sha256}],
        "asr_status": "NOT_RUN_NO_ASR_PINS_IN_TTS_ONLY_STAGE",
        "asr_calls": 0, "human_gold": False,
        "training_admitted": False, "qualification_allowed": False,
        "no_independent_human_evidence": True,
    }


def run(root: Path) -> dict:
    claim()
    need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_REF") == "refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
         "SEVEN_VOICE_MAIN_ONCE_REQUIRED")
    need(root.is_absolute() and root.parent.is_dir() and
         root != capture.ROOT and capture.ROOT not in root.parents and
         not root.exists() and not root.parent.is_symlink(),
         "NEW_PRIVATE_SEVEN_VOICE_RESULT_REQUIRED")
    root.mkdir(mode=0o700)
    capture.media()._once(root / "started.json", {
        "schema": POLICY, "source_code_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "trigger_sha256": sha(TRIGGER.read_bytes()),
        "tts_attempt_ceiling": MAX_CALLS, "old_tts_run_untouched": ORIGINAL_TTS_RUN,
        "training_admitted": False, "no_retry": True})
    model_root = root / "models"
    model_root.mkdir(mode=0o700)
    result = None
    stage_receipts = []
    try:
        # Downloads one exact public pretrained checkpoint. No TTS call during acquisition.
        lock = pilot.materialize_model(model_root)
        lock_path = root / "model-lock.json"
        capture.media()._once(lock_path, lock)
        source = plan(sha(lock_path.read_bytes()))
        capture.verify_model(source, lock_path, model_root)
        capture.media()._once(root / "source-plan.json", source)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        backend = capture.QwenEnglish(model_root / "preset")
        for stage_index, budget in enumerate(STAGES):
            receipt = capture.capture(source, root / "capture", backend,
                                      max_calls=budget)
            stage_receipts.append({
                "stage_index": stage_index, "budget": budget,
                "actual_new_attempts": receipt["attempts_this_call"],
                "total_started": receipt["total_started"],
                "total_completed": receipt["total_completed"],
                "training_admitted": False})
            capture.media()._once(root / ("stage-%02d.json" % stage_index),
                                  stage_receipts[-1])
        rows = []
        for cell in source["cells"]:
            path = root / "capture" / "cells" / cell["cell_id"]
            completed = path / "completed.json"
            failed = path / "failed.json"
            status = ("GENERATED_UNADMITTED" if completed.is_file()
                      else "FAILED_NO_RETRY" if failed.is_file() else "NOT_RUN")
            item = {"cell_id": cell["cell_id"], "source_family_id": cell["source_family_id"],
                    "role": cell["role"], "intent": cell["intent"],
                    "status": status, "training_admitted": False}
            if completed.is_file():
                receipt = market.read(completed)
                item["wav_sha256"] = receipt["audio"]["wav_sha256"]
                item["pcm_sha256"] = receipt["audio"]["pcm_sha256"]
            rows.append(item)
        done = sum(row["status"] == "GENERATED_UNADMITTED" for row in rows)
        attempted = sum(row["status"] != "NOT_RUN" for row in rows)
        result = {
            "schema": POLICY,
            "status": ("COMPLETE" if done == MAX_CALLS and attempted == MAX_CALLS
                       else "INCOMPLETE_NO_RETRY"),
            "planned_audio_count": MAX_CALLS, "attempted_audio_count": attempted,
            "generated_audio_count": done, "asr_calls": 0,
            "previous_audio_regenerated": 0, "source_rows": rows,
            "stage_receipts": stage_receipts,
            "model_lock_sha256": sha(lock_path.read_bytes()),
            "qwen_tts_package": version("qwen-tts"),
            "training_admitted": False, "human_gold": False,
        }
    except BaseException as error:
        result = {
            "schema": POLICY, "status": "FAILED_NO_RETRY",
            "error_type": type(error).__name__,
            "attempted_audio_count": sum(x["actual_new_attempts"] for x in stage_receipts),
            "planned_audio_count": MAX_CALLS,
            "previous_audio_regenerated": 0, "asr_calls": 0,
            "training_admitted": False, "human_gold": False,
            "stage_receipts": stage_receipts,
        }
        capture.media()._once(root / "failed.json", result)
        raise
    finally:
        if result is not None:
            capture.media()._once(root / "summary.json", result)
        if model_root.exists():
            shutil.rmtree(model_root)
    return result


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    try:
        claim()
        if not args.execute:
            print(json.dumps({
                "schema": POLICY, "new_voices": len(VOICES),
                "planned_calls": MAX_CALLS, "previous_audio_regenerated": 0,
                "model_calls": 0, "training_admitted": False}))
            return 0
        need(args.output is not None, "REAL_OUTPUT_REQUIRED")
        result = run(args.output)
        print(json.dumps({k: result[k] for k in (
            "status", "attempted_audio_count", "generated_audio_count",
            "asr_calls", "training_admitted")}, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except (ValueError, OSError) as exc:
        print("SEVEN_VOICE_TTS_STOP: " + type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
