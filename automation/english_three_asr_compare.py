"""Compare three actual, independent English ASR result receipts on fixed WAVs.

Research-only consumer. No model downloads/inference, no label authentication and
no automatic training admission. Exact source plan and individual native result
SHA256 receipts are checked before any text comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import english_blind_asr as blind
import english_qwen_asr_real_once as qwen
import english_whisper_real_once as whisper
import english_parakeet_real_once as parakeet
import english_qwen_asr_sourcefix_once as sourcefix
import english_qwen_capture as english
import multilingual_campaign as market

POLICY = "english-real-three-asr-evidence-20261112-v1"
TRIGGER = Path(__file__).resolve().parents[1] / ".github/triggers/english-three-asr-20261112.json"
ROOT_NAMES = {
    "qwen3-asr-en": "en-asr-cli-recovery-results",
    "whisper-small-en": "en-whisper-real-results",
    "parakeet-tdt-v3-en": "en-parakeet-recovery-actual",
}
PROVIDERS = {
    "qwen3-asr-en": (qwen.MODEL, qwen.REV, qwen.POLICY),
    "whisper-small-en": (whisper.MODEL, whisper.MODEL_SOURCE_COMMIT, whisper.POLICY),
    "parakeet-tdt-v3-en": (parakeet.MODEL, parakeet.MODEL_COMMIT, parakeet.POLICY),
}
ARTIFACTS = {
    "tts": (43214159818, 13942568634,
            "0235b0064006f79bfe9c8fa551c2a1e41da61c86e0fd987a90ee2d47842a9f21"),
    "qwen3-asr-en": (43226774479, 13948268182,
                     "ffaaab0c8178c6a41d1d844d6e63a740c6eed97d108990210ce1e4edada2545c"),
    "whisper-small-en": (43229354622, 13950635761,
                         "6c2c0971f091c11e62a8a81eab8d214dc8e1d0a960fcb208e756f5e57e4607bf"),
    "parakeet-tdt-v3-en": (43235162509, 13952845421,
                            "de09e3eba132bcd6844cea61af2c5926763d51b81aa4de039b9e5b7aecca9dcf"),
}
KNOWN_SOURCE_PLAN_SHA = "c06748748bcac9e7eeea6589ff5ff85b363b8fd382a56740d43852c4e1a0a309"


def need(ok: bool, error: str):
    if not ok:
        raise ValueError(error)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def claim() -> dict:
    pinned = {
        "schema": POLICY,
        "artifacts": {k:{"run_id":v[0],"artifact_id":v[1],
                          "sha256":v[2]} for k,v in ARTIFACTS.items()},
        "source_plan_sha256": KNOWN_SOURCE_PLAN_SHA,
        "three_model_families": list(ROOT_NAMES),
        "new_model_calls": 0, "training_admitted": False,
        "human_gold": False, "qualification_allowed": False,
    }
    declared = market.read(TRIGGER)
    need(declared == pinned, "THREE_ASR_EXACT_SOURCE_TRIGGER_DRIFT")
    return pinned


def validate_model_root(base: Path, provider: str, plan_sha: str) -> Path:
    blind.private_path(base, "ASR_EVIDENCE_ROOT_NOT_PRIVATE")
    directory = base / ROOT_NAMES[provider]
    need(directory.is_dir() and not directory.is_symlink(),
         "ASR_EVIDENCE_ROOT_LAYOUT")
    summary = market.read(directory / "summary.json")
    model_id, revision, policy = PROVIDERS[provider]
    need(summary.get("schema") == policy and
         summary.get("status") == "COMPLETE" and
         summary.get("model_id") == model_id and
         summary.get("source_plan_sha256") == plan_sha and
         summary.get("completed_asr") == 10 and
         summary.get("asr_attempts") == 10 and
         summary.get("training_admitted") is False and
         summary.get("human_gold") is False and
         summary.get("three_asr_completed") is False,
         "ASR_SOURCE_COMPLETION_NOT_AUTHENTICATED:" + provider)
    claimed_revision = summary.get("model_revision_sha",
                                   summary.get("model_source_commit"))
    need(claimed_revision == revision, "ASR_MODEL_REVISION_DRIFT:" + provider)
    records = summary.get("rows")
    need(type(records) is list and len(records) == 10 and
         len({r["cell_id"] for r in records}) == 10 and
         all(r.get("status") == "BLIND_ASR_UNADMITTED" for r in records),
         "ASR_OUTCOME_SET_NOT_CLOSED:" + provider)
    return directory


def compare(original: Path, evidence: dict[str, Path]) -> dict:
    plan, inputs = sourcefix.original_audio_root(original)[1:3]
    need(sha(market.canonical(plan)) == KNOWN_SOURCE_PLAN_SHA,
         "THREE_ASR_SOURCE_PLAN_IDENTITY_DRIFT")
    need(set(evidence) == set(ROOT_NAMES), "THREE_ASR_MISSING_OR_DUPLICATE_FAMILY")
    checked = {p:validate_model_root(root,p,KNOWN_SOURCE_PLAN_SHA)
               for p,root in evidence.items()}
    observed = []
    for cell,wav,pcm in sourcefix.core.verified_source(
            sourcefix.original_audio_root(original)[0])[1]:
        texts = {}
        digests = {}
        for provider,root in checked.items():
            provider_dir = root / "cells" / cell["cell_id"]
            native_path = provider_dir / "native.json"
            receipt_path = provider_dir / "completed.json"
            model_id,revision,_ = PROVIDERS[provider]
            native = blind.native_result(native_path, {
                "provider_id":provider,"model_id":model_id,
                "model_revision_sha":revision},cell,pcm)
            size, native_sha = english.media()._safe_file(native_path, limit=65536)
            receipt = market.read(receipt_path)
            need(receipt.get("status") == "BLIND_ASR_UNADMITTED" and
                 receipt.get("training_admitted") is False and
                 receipt.get("cell_id") == cell["cell_id"] and
                 receipt.get("provider_id") == provider and
                 receipt.get("wav_sha256") == pcm["wav_sha256"] and
                 receipt.get("native_sha256") == native_sha,
                 "NATIVE_ASR_RECEIPT_NOT_BOUND:" + provider)
            texts[provider] = native["actual_text"]
            digests[provider] = native_sha
        norms = {name:market.normalize(value,"en-US")
                 for name,value in texts.items()}
        groups = Counter(norms.values())
        maximum = max(groups.values())
        expected = market.normalize(cell["text_requested"],"en-US")
        all_same = maximum == 3
        exact_wake = all_same and expected in groups and cell["intent"] == "wake"
        status = ("THREE_IDENTICAL" if all_same
                  else "TWO_IDENTICAL" if maximum == 2
                  else "ALL_DIFFERENT")
        review = ("POSITIVE_CANDIDATE_PENDING_HUMAN_REVIEW" if exact_wake
                  else "WAKE_PRONUNCIATION_REVIEW_REQUIRED" if cell["intent"] == "wake"
                  else "NEGATIVE_SPOKEN_CONTENT_REVIEW_REQUIRED")
        observed.append({
            "cell_id":cell["cell_id"],"voice_id":cell["voice_id"],
            "source_family_id":cell["source_family_id"],"role":cell["role"],
            "intent":cell["intent"],"requested_text":cell["text_requested"],
            "wav_sha256":pcm["wav_sha256"],"native_sha256":digests,
            "actual_texts":texts,"normalized_texts":norms,
            "agreement":status,"all_models_agree":all_same,
            "three_models_match_intended":all_same and expected in groups,
            "review_state":review,"human_gold":False,
            "training_admitted":False,"qualified_negative":False,
        })
    need(len(observed) == 10 and len({r["cell_id"] for r in observed}) == 10,
         "THREE_ASR_NOT_ALL_TEN")
    hist = Counter(r["agreement"] for r in observed)
    wake = [r for r in observed if r["intent"] == "wake"]
    return {
        "schema": POLICY, "status": "MACHINE_THREE_ASR_COMPARISON_COMPLETE",
        "source_tts_run_id": ARTIFACTS["tts"][0],
        "source_plan_sha256": KNOWN_SOURCE_PLAN_SHA,
        "source_audio_count": len(observed),
        "complete_native_observations": len(observed) * 3,
        "model_families": list(ROOT_NAMES),
        "agreement_distribution": {k:hist.get(k,0) for k in
                 ("THREE_IDENTICAL","TWO_IDENTICAL","ALL_DIFFERENT")},
        "wake_requests": len(wake),
        "strict_three_model_wake_match_count": sum(
           r["three_models_match_intended"] for r in wake),
        "three_model_identical_request_total": sum(
           r["three_models_match_intended"] for r in observed),
        "rows": observed, "new_asr_calls": 0, "new_tts_calls": 0,
        "new_training_calls": 0, "training_admitted": False,
        "human_gold": False, "qualified_negative_hours": 0,
        "production_model_selected": False,
        "conclusion": "MACHINE_ONLY_NO_TRAINING_ADMISSION",
    }


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--original",type=Path,required=True)
    cli.add_argument("--qwen",type=Path,required=True)
    cli.add_argument("--whisper",type=Path,required=True)
    cli.add_argument("--parakeet",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args(argv)
    try:
        claim()
        report=compare(args.original,{
            "qwen3-asr-en":args.qwen,
            "whisper-small-en":args.whisper,
            "parakeet-tdt-v3-en":args.parakeet})
        need(args.out.parent.is_dir() and not args.out.exists() and
             not args.out.is_symlink(), "COMPARISON_REPORT_CREATE_ONLY")
        english.media()._once(args.out,report)
        print(json.dumps({k:report[k] for k in (
            "status","source_audio_count","complete_native_observations",
            "agreement_distribution","strict_three_model_wake_match_count",
            "training_admitted")},sort_keys=True))
        return 0
    except (ValueError,OSError,KeyError) as error:
        print("THREE_ASR_COMPARISON_STOP:"+type(error).__name__,
              file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
