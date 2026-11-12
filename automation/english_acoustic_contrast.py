"""Read-only 63-WAV English nearphone *acoustic distance* experiment.

Reuses exact completed Qwen TTS and source-linked three-ASR ledger archives;
performs no TTS/ASR/KWS inference, tuning, or training. Log-mel+DTW distances
are same-synthetic-voice diagnostics, NOT phoneme truth, FAR/FRR or admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import wave

# Python -I -S removes script directory. Only checked-out sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_seven_voice_asr_once as source_asr
import english_seven_voice_review as review
import english_seven_voice_tts_once as source_tts
import english_blind_asr as blind
import multilingual_campaign as market

POLICY = "english-63-paired-logmel-dtw-acoustic-contrast-v1"
TRIGGER = (Path(__file__).resolve().parents[1] /
           ".github/triggers/english-acoustic-contrast-20261113.json")
REVIEW_ARTIFACT_ID = 13965259534
REVIEW_RUN_ID = 43267739514
REVIEW_ZIP_SHA256 = "3786baa9fd3575437a05e97956fafdf3298eac23de2538b485b8342ef4d25133"
REVIEW_JSON_SHA256 = "60054a3532491631a47b094ea7ff2bc86ef16530ff510f968bc9edcdd253a9ad"
PROMPTS = tuple(p[0] for p in source_tts.PROMPTS)
CONTRASTS = (
    ("vosen_vs_wosen", "Hey Vosen", "Hey, Vosen!", "Hey Wosen"),
    ("vosen_vs_rosen", "Hey Vosen", "Hey, Vosen!", "Hey Rosen"),
    ("mero_vs_miro", "Hey Mero", "Hey, Mero!", "Hey Miro"),
)
VOICES = tuple(name for name, _ in source_tts.VOICES)


def need(condition: bool, error: str) -> None:
    if not condition:
        raise ValueError(error)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def claim() -> dict:
    frozen = {
        "schema": POLICY,
        "source_tts_run_id": source_asr.SOURCE_RUN_ID,
        "source_tts_artifact_id": review.ARCHIVES["tts"][0],
        "source_tts_zip_sha256": review.ARCHIVES["tts"][1],
        "source_review_run_id": REVIEW_RUN_ID,
        "source_review_artifact_id": REVIEW_ARTIFACT_ID,
        "source_review_zip_sha256": REVIEW_ZIP_SHA256,
        "source_review_json_sha256": REVIEW_JSON_SHA256,
        "source_audio_count": 63,
        "source_asr_observations": 189,
        "contrast_names": [c[0] for c in CONTRASTS],
        "feature_policy": "16k-mono-logmel40-25ms-10ms-cmvn-dtw-fullpath-v1",
        "new_tts_calls": 0, "new_asr_calls": 0,
        "new_neural_model_calls": 0,
        "training_admitted": False, "human_gold": False,
        "no_scheduled_tasks": True,
    }
    need(market.read(TRIGGER) == frozen, "ACOUSTIC_CONTRAST_TRIGGER_DRIFT")
    return frozen


def gate() -> None:
    need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_EVENT_NAME") == "push" and
         os.environ.get("GITHUB_REF") == "refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
         "ACOUSTIC_CONTRAST_PROTECTED_MAIN_ONCE_ONLY")


def load_exact_sources(tts_zip: Path, review_zip: Path, root: Path):
    """Authenticates actual WAVs/PCM and 63 saved ASR observations, no model."""
    t = root / "tts"
    a = root / "asr_review"
    t.mkdir(mode=0o700)
    a.mkdir(mode=0o700)
    review.unpack_pinned(tts_zip, review.ARCHIVES["tts"][1], t)
    review.unpack_pinned(review_zip, REVIEW_ZIP_SHA256, a)
    plan, cells = source_asr.source_from_artifact(t)
    raw = (a / "english-63-machine-review.json").read_bytes()
    need(sha256(raw) == REVIEW_JSON_SHA256, "ACOUSTIC_REVIEW_JSON_DRIFT")
    ledger = json.loads(raw)
    need(ledger.get("schema") == review.POLICY and
         ledger.get("status") == "MACHINE_REVIEW_COMPLETE_NOT_ADMITTED" and
         ledger.get("source_tts_run_id") == source_asr.SOURCE_RUN_ID and
         ledger.get("source_tts_artifact_id") == review.ARCHIVES["tts"][0] and
         ledger.get("three_asr_run_id") == review.ASR_RUN_ID and
         ledger.get("three_asr_head_sha") == review.ASR_HEAD_SHA and
         ledger.get("artifact_pins") == {
             k: {"id":v[0], "zip_sha256":v[1]}
             for k,v in review.ARCHIVES.items()} and
         ledger.get("counts", {}).get("total") == 63 and
         ledger.get("counts", {}).get("three_way_completed_calls") == 189 and
         ledger.get("human_gold") is False and
         ledger.get("training_admitted") is False and
         ledger.get("acoustic_gold_admitted") == 0,
         "ACOUSTIC_REVIEW_LEDGER_SOURCE_DRIFT")
    rows = ledger.get("rows")
    need(type(rows) is list and len(rows) == 63,
         "ACOUSTIC_REVIEW_LEDGER_COUNT")
    by_id = {r.get("cell_id"):r for r in rows if type(r) is dict}
    need(len(by_id) == 63 and
         set(by_id) == {cell["cell_id"] for cell,_,_ in cells},
         "ACOUSTIC_REVIEW_LEDGER_CELL_ROSTER")
    pcm_hashes = set()
    for cell, _, pcm in cells:
        record = by_id[cell["cell_id"]]
        need(record.get("wav_sha256") == pcm["wav_sha256"] and
             record.get("voice_id") == cell["voice_id"] and
             record.get("source_family_id") == cell["source_family_id"] and
             record.get("role") == cell["role"] and
             record.get("requested_text") == cell["text_requested"] and
             record.get("requested_intent") == cell["intent"] and
             record.get("requested_keyword_index") == cell["requested_keyword_index"] and
             record.get("human_gold") is False and
             record.get("training_admitted") is False and
             record.get("trusted_negative") is False and
             record.get("acoustic_pronunciation_verified") is False and
             type(record.get("asr")) is dict and
             set(record["asr"]) == set(source_asr.PROVIDERS),
             "ACOUSTIC_REVIEW_CELL_IDENTITY_DRIFT")
    return plan, cells, ledger


def _mel_bank(np):
    freq = np.linspace(0.0, 8000.0, 257)
    mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    bounds = 700.0 * (
        10.0 ** (np.linspace(mel(70), mel(7600), 42) / 2595.0) - 1.0)
    bank = np.zeros((40, 257), dtype=np.float64)
    for i in range(40):
        left, center, right = bounds[i:i + 3]
        bank[i] = np.maximum(0.0, np.minimum(
            (freq-left)/(center-left), (right-freq)/(right-center)))
        bank[i] /= max(1e-12, float(bank[i].sum()))
    return bank


def features(wav_path: Path, bank=None):
    """No ASR; 25-ms Hann / 10-ms hop, energy trim, logmel40, clip CMVN."""
    import numpy as np
    with wave.open(str(wav_path), "rb") as wav:
        need(wav.getnchannels() == 1 and wav.getsampwidth() == 2 and
             wav.getframerate() == 16000 and wav.getcomptype() == "NONE" and
             wav.getnframes() >= 400, "ACOUSTIC_WAV_GEOMETRY")
        frames_count = wav.getnframes()
        raw = wav.readframes(frames_count)
        need(len(raw) == frames_count * 2 and not wav.readframes(1),
             "ACOUSTIC_WAV_SHORT_READ")
    pcm_hash = sha256(raw)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    need(np.isfinite(samples).all(), "ACOUSTIC_NONFINITE_PCM")
    padded = np.pad(samples, (200, 200))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160]
    energy = np.sqrt((frames * frames).mean(axis=1))
    active = np.where(energy >= max(0.002, float(energy.max()) * 0.06))[0]
    need(active.size > 0, "ACOUSTIC_NO_ACTIVE_SPEECH")
    frames = frames[max(0, int(active[0]) - 3):
                    min(len(frames), int(active[-1]) + 4)]
    b = bank if bank is not None else _mel_bank(np)
    power = np.abs(np.fft.rfft(frames * np.hanning(400), 512, axis=1)) ** 2
    logmel = np.log(np.maximum(power @ b.T, 1e-8))
    z = (logmel - logmel.mean(axis=0)) / np.maximum(logmel.std(axis=0), 0.25)
    need(5 <= len(z) <= 1200 and np.isfinite(z).all(),
         "ACOUSTIC_FEATURE_BUDGET")
    return z, {
        "pcm_sha256": pcm_hash, "audio_frames": frames_count,
        "active_feature_frames": len(z),
        "clip_rms": round(float(np.sqrt(np.mean(samples * samples))), 7),
    }


def dtw(a, b) -> float:
    """Full-path deterministic frame DTW. No training or threshold fitting."""
    import numpy as np
    need(a.ndim == 2 and b.ndim == 2 and
         a.shape[1] == b.shape[1] == 40 and
         5 <= len(a) <= 1200 and 5 <= len(b) <= 1200,
         "ACOUSTIC_DTW_SHAPE")
    d = np.sqrt(np.maximum(0., ((a*a).sum(1)[:,None] +
        (b*b).sum(1)[None,:] - 2 * a @ b.T)) / 40)
    n,m = d.shape
    prev = np.full(m+1, float("inf"))
    prev[0] = 0.0
    prevsteps = np.zeros(m+1, dtype=np.int32)
    for i in range(1, n+1):
        curr = np.full(m+1, float("inf"))
        steps = np.zeros(m+1, dtype=np.int32)
        for j in range(1,m+1):
            best = min(((prev[j-1],prevsteps[j-1]),
                        (prev[j],prevsteps[j]),
                        (curr[j-1],steps[j-1])), key=lambda x:(x[0],x[1]))
            curr[j] = float(d[i-1,j-1]) + best[0]
            steps[j] = best[1] + 1
        prev,prevsteps=curr,steps
    need(prevsteps[m] > 0 and math.isfinite(float(prev[m])),
         "ACOUSTIC_DTW_PATH")
    return round(float(prev[m] / prevsteps[m]), 6)


def contrast(features_by_voice: dict, cell_by_voice: dict) -> dict:
    need(set(features_by_voice) == set(VOICES) ==
         set(cell_by_voice), "ACOUSTIC_VOICE_ROSTER")
    voice_rows = []
    all_margins = {key:[] for key,_,_,_ in CONTRASTS}
    for name,role in source_tts.VOICES:
        by_prompt = features_by_voice[name]
        ids = cell_by_voice[name]
        need(set(by_prompt) == set(PROMPTS) and
             set(ids) == set(PROMPTS), "ACOUSTIC_PROMPT_ROSTER")
        per = {}
        for label,p0,p1,near in CONTRASTS:
            same = dtw(by_prompt[p0],by_prompt[p1])
            # Each nearphone is compared with both pre-existing wake variants.
            near0 = dtw(by_prompt[p0],by_prompt[near])
            near1 = dtw(by_prompt[p1],by_prompt[near])
            best = min(near0,near1)
            need(same > 0.0, "ACOUSTIC_SAME_VARIANT_ZERO_UNSTABLE")
            ratio = round(best/same,6)
            near_at_least_as_close = best <= same
            all_margins[label].append(ratio)
            per[label] = {
                "same_wake_variant_dtw": same,
                "nearest_nearphone_dtw": best,
                "nearphone_to_same_variant_ratio": ratio,
                "nearphone_not_farther_than_same": near_at_least_as_close,
                "same_wake_cells": [ids[p0],ids[p1]], "nearphone_cell": ids[near],
                "diagnostic_only": True, "training_admitted": False,
                "human_gold": False, "trusted_negative": False,
            }
        voice_rows.append({
            "voice_id": name, "source_family_id": "qwen3-preset-"+name,
            "role": role, "comparisons": per,
            "training_admitted": False, "human_gold": False,
        })
    by_contrast = {}
    for key,_,_,_ in CONTRASTS:
        ratios = all_margins[key]
        flags = [r["comparisons"][key]["nearphone_not_farther_than_same"]
                 for r in voice_rows]
        by_contrast[key] = {
            "source_families": len(flags),
            "nearphone_not_farther_than_same": sum(flags),
            "median_ratio": round(statistics.median(ratios),6),
            "min_ratio": min(ratios), "max_ratio": max(ratios),
            "independent_human_speaker_count": 0,
        }
    return {"by_voice": voice_rows, "by_contrast": by_contrast}


def execute(tts_zip: Path, review_zip: Path) -> dict:
    claim()
    gate()
    import numpy as np
    with tempfile.TemporaryDirectory(prefix="en-acoustic63-") as tmp:
        plan, cells, ledger = load_exact_sources(tts_zip, review_zip, Path(tmp))
        need(plan["wake_phrases_requested"] == ["Hey Vosen","Hey Mero"],
             "ACOUSTIC_WAKE_PHRASES_DRIFT")
        by_voice = {}
        cell_ids = {}
        source_pcm_hashes = set()
        cell_receipts = []
        bank = _mel_bank(np)
        for cell,wav,pcm in cells:
            features_one, acoustics = features(wav, bank)
            need(acoustics["pcm_sha256"] not in source_pcm_hashes,
                 "ACOUSTIC_DUPLICATE_PCM")
            source_pcm_hashes.add(acoustics["pcm_sha256"])
            name = cell["voice_id"]
            by_voice.setdefault(name,{})[cell["text_requested"]] = features_one
            cell_ids.setdefault(name,{})[cell["text_requested"]] = cell["cell_id"]
            cell_receipts.append({
                "cell_id":cell["cell_id"],
                "wav_sha256":pcm["wav_sha256"],
                "pcm_sha256":acoustics["pcm_sha256"],
                "voice_id":name,
                "source_family_id":cell["source_family_id"],
                "role":cell["role"],"requested_intent":cell["intent"],
                "text_requested":cell["text_requested"],
                "active_feature_frames":acoustics["active_feature_frames"],
                "audio_frames":acoustics["audio_frames"],
                "training_admitted":False, "human_gold":False,
            })
        need(len(cell_receipts) == 63 and len(source_pcm_hashes) == 63,
             "ACOUSTIC_SOURCE_COUNT")
        results = contrast(by_voice,cell_ids)
        return {
            "schema": POLICY, "status":"SOURCE_BOUND_ACOUSTIC_DIAGNOSTIC_COMPLETE",
            "tts_run_id":source_asr.SOURCE_RUN_ID,
            "tts_zip_sha256":review.ARCHIVES["tts"][1],
            "review_run_id":REVIEW_RUN_ID,
            "review_zip_sha256":REVIEW_ZIP_SHA256,
            "review_json_sha256":REVIEW_JSON_SHA256,
            "new_tts_calls":0,"new_asr_calls":0,"new_neural_model_calls":0,
            "sample_count":len(cell_receipts),
            "asr_observations_reused":ledger["counts"]["three_way_completed_calls"],
            "independent_human_speakers":0,
            "method":"16k-logmel40-25ms-10ms-clip-CMVN-fullpath-DTW-v1",
            "interpretation":"Relative same-synthetic-voice similarity only; NOT phonetic truth, FAR/FRR, or KWS.",
            **results, "sample_receipts":cell_receipts,
            "training_admitted":False,"human_gold":False,"trusted_negative":False,
            "independent_evaluation":"NOT_RUN","physical_ssc305":"NOT_RUN",
        }


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-zip",type=Path)
    parser.add_argument("--review-zip",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    try:
        if not args.execute:
            print(json.dumps({"schema":POLICY,"contrast_names":[x[0] for x in CONTRASTS],
                "new_tts_calls":0,"new_asr_calls":0,"new_neural_model_calls":0,
                "training_admitted":False}))
            return 0
        need(all(x is not None for x in (args.tts_zip,args.review_zip,args.output)),
             "ACOUSTIC_EXPLICIT_INPUT_OUTPUT_REQUIRED")
        dest=args.output
        blind.private_path(dest,"ACOUSTIC_PRIVATE_OUTPUT_REQUIRED")
        need(dest.is_absolute() and dest.parent.is_dir() and not dest.exists(),
             "ACOUSTIC_OUTPUT_CREATE_ONLY")
        result=execute(args.tts_zip,args.review_zip)
        with dest.open("x",encoding="utf-8") as stream:
            json.dump(result,stream,ensure_ascii=False,sort_keys=True,indent=2)
            stream.write("\n")
        dest.chmod(0o600)
        print(json.dumps({
            "status":result["status"],"sample_count":result["sample_count"],
            "by_contrast":result["by_contrast"],
            "training_admitted":False},sort_keys=True))
        return 0
    except (ValueError,OSError,wave.Error) as exc:
        print("ENGLISH_ACOUSTIC_STOP: "+type(exc).__name__+": "+str(exc),
              file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
