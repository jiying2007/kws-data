"""Separate immutable one-shot Parakeet recovery after zero-call native setup failure.

The first Parakeet run 43231581918 never downloaded GGUF nor entered ASR.
NVIDIA v0.3.0 CPU release was safely repaired and validated on protected main
by PR #95. The old run is consumed; a new trigger, output and run ID are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import english_parakeet_real_once as native

POLICY = "english-parakeet-native-symlink-recovery-20261112-v1"
PREVIOUS_RUN = 43231581918
PREVIOUS_HEAD = "b87652d3b1c4f68a182cdc0ffa521059f53a9123"
PREVIOUS_ARTIFACT = 13949214717
PREVIOUS_ARTIFACT_SHA256 = "e769fd64b44c84ee4d02fa60b5e220640a778108aa812b94445111188997ee2f"
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-parakeet-runtimefix-20261112.json"


def claim() -> dict:
    expected = {
        "schema": POLICY, "prior_failure_run_id": PREVIOUS_RUN,
        "prior_failure_head": PREVIOUS_HEAD,
        "prior_failure_artifact_id": PREVIOUS_ARTIFACT,
        "prior_failure_artifact_sha256": PREVIOUS_ARTIFACT_SHA256,
        "prior_asr_attempts": 0, "prior_gguf_downloaded": False,
        "source_tts_run_id": 43214159818,
        "source_tts_artifact_id": 13942568634,
        "source_tts_artifact_sha256": native.request()["original_audio_artifact_sha256"],
        "model_id": native.MODEL,
        "model_revision": native.MODEL_COMMIT,
        "model_gguf_sha256": native.MODEL_SHA256,
        "runtime_cpu_release_sha256": native.RUNTIME_SHA256,
        "max_new_asr_calls": 10,
        "human_gold": False, "training_admitted": False,
        "three_asr_completed": False,
    }
    value = native.market.read(TRIGGER)
    native.need(value == expected, "PARAKEET_RECOVERY_CLAIM_DRIFT")
    return expected


def verify_prior_receipt(path: Path) -> dict:
    """Prior result is a separate, immutable zero-call failure, never a redo."""
    source = native.market.read(path)
    native.need(source.get("schema") == native.POLICY and
                source.get("status") == "INCOMPLETE_NO_RETRY" and
                source.get("asr_attempts") == 0 and
                source.get("completed_asr") == 0 and
                source.get("runtime_receipt") is None and
                source.get("gguf_receipt") is None and
                source.get("error_type") == "ValueError" and
                source.get("model_weight_sha256") == native.MODEL_SHA256 and
                source.get("source_tts_run_id") == 43214159818,
                "ORIGINAL_PARAKEET_ZERO_CALL_FAILURE_UNPROVEN")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-parent", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prior-summary", type=Path)
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        _ = claim()
        native.need(args.verify_source != args.execute and
                    args.artifact_parent is not None and
                    args.prior_summary is not None,
                    "PARAKEET_RECOVERY_EXPLICIT_INPUTS")
        verify_prior_receipt(args.prior_summary)
        if args.verify_source:
            plan, audio = native.source_inputs(args.artifact_parent)
            print(json.dumps({
                "schema": POLICY, "status": "SOURCE_AND_PREVIOUS_FAILURE_VERIFIED",
                "original_audio_count": len(audio),
                "source_plan_sha256": hashlib.sha256(
                    native.market.canonical(plan)).hexdigest(),
                "asr_calls": 0, "training_admitted": False}, sort_keys=True))
            return 0
        native.need(args.output is not None, "PARAKEET_RECOVERY_OUTPUT_REQUIRED")
        result = native.run(args.artifact_parent, args.output)
        print(json.dumps({key: result[key] for key in (
            "status", "asr_attempts", "completed_asr", "training_admitted"
        )}, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except (OSError, ValueError) as exc:
        print("PARAKEET_RECOVERY_STOP:" + type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
