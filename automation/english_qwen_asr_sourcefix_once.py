"""A second, explicitly separate one-time English Qwen-ASR execution claim.

The original 43218378495 job stopped before model setup/calls because its
downloaded TTS ZIP has one immutable parent directory. It MUST NOT be rerun.
This recovery selects the verified source directory, fixes the Qwen-ASR API
context argument, and preserves the failure and original real WAV identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import english_qwen_asr_real_once as core

POLICY = "english-qwen-asr-sourcefix-once-20261112-v1"
PREVIOUS_FAILED_RUN_ID = 43218378495
PREVIOUS_FAILED_HEAD = "386c81d83865011c180b5d1c5d72a4cd0402b189"
SOURCE_SUBDIR = "en-qwen-actual-20261112"
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-qwen-asr-sourcefix-20261112.json"


def claim() -> dict:
    expected = {
        "schema": POLICY, "previous_failed_run_id": PREVIOUS_FAILED_RUN_ID,
        "previous_failed_head_sha": PREVIOUS_FAILED_HEAD,
        "previous_real_asr_calls": 0,
        "original_tts_run_id": core.SOURCE_RUN_ID,
        "original_tts_head_sha": core.SOURCE_HEAD,
        "original_tts_artifact_id": core.SOURCE_ARTIFACT_ID,
        "original_tts_artifact_sha256": core.SOURCE_ARTIFACT_DIGEST,
        "extracted_audio_subdir": SOURCE_SUBDIR,
        "model_id": core.MODEL, "model_revision": core.REV,
        "max_new_asr_calls": core.MAX_CALLS,
        "training_admitted": False, "human_gold": False,
        "three_asr_completed": False,
    }
    value = core.market.read(TRIGGER)
    core.need(value == expected, "ASR_CORRECTION_CLAIM_DRIFT")
    core.request()  # Original, immutable first-run source identity remains present.
    return expected


def original_audio_root(parent: Path) -> tuple[Path, dict, int]:
    """Fail closed if the ZIP parent layout or any audio receipt is different."""
    core.blind.private_path(parent, "ASR_ARTIFACT_PARENT_NOT_PRIVATE")
    core.need(parent.is_dir() and (parent / SOURCE_SUBDIR).is_dir(),
              "ASR_EXPECTED_ZIP_PARENT_MISSING")
    core.need(not (parent / "source-plan.json").exists(),
              "ASR_UNEXPECTED_ROOT_LEVEL_PLAN")
    source = parent / SOURCE_SUBDIR
    plan, cells = core.verified_source(source)
    core.need(len(cells) == core.MAX_CALLS and
              len({cell["cell_id"] for cell, _, _ in cells}) == core.MAX_CALLS,
              "ASR_ORIGINAL_WAV_COUNT_DRIFT")
    return source, plan, len(cells)


def run(parent: Path, output: Path):
    _ = claim()
    source, _, _ = original_audio_root(parent)
    return core.run(source, output)


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--artifact-parent", type=Path)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--verify-source", action="store_true")
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    try:
        frozen = claim()
        core.need(args.verify_source != args.execute and
                  args.artifact_parent is not None,
                  "ONE_EXPLICIT_RECOVERY_MODE_REQUIRED")
        if args.verify_source:
            _, plan, count = original_audio_root(args.artifact_parent)
            print(json.dumps({"policy": POLICY, "status": "SOURCE_VERIFIED_NO_ASR",
                              "audio_count": count, "asr_calls": 0,
                              "source_plan_sha256": hashlib.sha256(
                                  core.market.canonical(plan)).hexdigest(),
                              "training_admitted": False}, sort_keys=True))
            return 0
        core.need(args.output is not None, "NEW_ASR_OUTPUT_REQUIRED")
        result = run(args.artifact_parent, args.output)
        print(json.dumps({key: result[key] for key in (
            "status", "source_audio_count", "asr_attempts",
            "completed_asr", "training_admitted")}, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except (OSError, ValueError) as error:
        print("ENGLISH_ASR_SOURCEFIX_STOP: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
