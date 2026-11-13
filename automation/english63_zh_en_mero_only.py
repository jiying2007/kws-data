"""Real English63 'Hey Mero'-ONLY bilingual Zipformer KWS event study.

The upstream Zh-En 3M phoneme lexicon contains HEY and MERO but not VOSEN.
Do not invent any VOSEN pronunciation. No new TTS, ASR, label admission,
training, threshold tuning, or FAR/FRR; all 63 unchanged WAVs are machine
weak-source observational clips. The Mero-only key may fire on other prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english63_zipformer_events_once as reference
import english_seven_voice_asr_once as audio_source
import english_seven_voice_review as source_review
import multilingual_campaign as market

POLICY = "english63-zh-en-mero-only-real-events-v1"
TRIGGER = (Path(__file__).resolve().parents[1] /
           ".github/triggers/english63-zh-en-mero-only-20261113.json")
PUBLISHED_ZH_EN_SHA = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
ORIGINAL_TTS_SHA = "f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc"
KEYWORD_ALIAS = "HEY_MERO"
BLOCKED_NAME = "VOSEN"
TARGET_TEXT = "HEY MERO"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def claim():
    frozen = {
        "schema": POLICY,
        "original_tts_run_id": audio_source.SOURCE_RUN_ID,
        "original_tts_artifact_id": source_review.ARCHIVES["tts"][0],
        "original_tts_zip_sha256": ORIGINAL_TTS_SHA,
        "source_clip_count": 63,
        "pretrained_model": "zh-en",
        "pretrained_model_sha256": PUBLISHED_ZH_EN_SHA,
        "keyword_original_text": "Hey Mero",
        "official_phone_lexicon_only": True,
        "vosen_supported": False,
        "fixed_threshold": 0.25,
        "new_tts_calls": 0, "new_asr_calls": 0,
        "optimizer_updates": 0,
        "model_promotion_allowed": False,
        "training_admitted": False,
        "true_far_frr": "NOT_RUN",
        "no_scheduled_patrol": True,
    }
    need(market.read(TRIGGER) == frozen, "EN_MERO_FROZEN_SOURCE_CLAIM_DRIFT")
    return frozen


def protected_main():
    import os
    need(os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_REPOSITORY_OWNER") == "jiying2007" and
         os.environ.get("GITHUB_EVENT_NAME") == "push" and
         os.environ.get("GITHUB_REF") == "refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1",
         "EN_MERO_FIRST_PROTECTED_MAIN_ONLY")


def verify_official_phone_words(lexicon: Path):
    need(lexicon.is_file() and not lexicon.is_symlink() and
         0 < lexicon.stat().st_size <= 6 * 1024 * 1024,
         "EN_MERO_ORIGINAL_PHONE_LEXICON_REQUIRED")
    with lexicon.open(encoding="utf-8") as handle:
        words = {line.split()[0] for line in handle if line.strip()}
    need("HEY" in words and "MERO" in words and BLOCKED_NAME not in words,
         "EN_MERO_WORDS_NOT_SUPPORTED_OR_VOSEN_UNBLOCKED")
    return True


def encode_one_keyword(parts: dict, directory: Path):
    """Use original official en.phone; no synthetic G2P, no fallback."""
    verify_official_phone_words(parts["lexicon"])
    raw = directory / "mero-original.txt"
    out = directory / "mero-encoded.txt"
    raw.write_text(TARGET_TEXT + " @" + KEYWORD_ALIAS + "\n",
                   encoding="utf-8")
    cmd = ["sherpa-onnx-cli", "text2token",
           "--tokens", str(parts["tokens"]),
           "--tokens-type", "phone+ppinyin",
           "--lexicon", str(parts["lexicon"]),
           str(raw), str(out)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           check=False, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("EN_MERO_OFFICIAL_TOKENIZER_UNAVAILABLE") from exc
    need(p.returncode == 0 and out.is_file() and not out.is_symlink(),
         "EN_MERO_OFFICIAL_TEXT2TOKEN_FAILED: " + p.stderr[-500:])
    lines = [x.strip() for x in out.read_text(encoding="utf-8").splitlines()
             if x.strip()]
    need(len(lines) == 1 and
         lines[0].endswith(" @" + KEYWORD_ALIAS) and
         len(lines[0].split()) >= 3,
         "EN_MERO_SINGLE_KW_TOKENIZATION_FAILED")
    return out


def validate_receipt(result):
    need(type(result) is dict and result.get("schema") == POLICY and
         result.get("status") ==
         "EXECUTED_REAL_ZH_EN_MERO_ONLY_WEAK_CLIP_EVENTS" and
         result.get("pretrained_archive_sha256") == PUBLISHED_ZH_EN_SHA and
         result.get("original_tts_zip_sha256") == ORIGINAL_TTS_SHA and
         result.get("source_clip_count") == 63 and
         type(result.get("clip_results")) is list and
         len(result["clip_results"]) == 63 and
         len({x.get("cell_id") for x in result["clip_results"]}) == 63 and
         result.get("keyword_original_text") == "Hey Mero" and
         result.get("vosen_not_evaluated") is True and
         result.get("new_tts_calls") == result.get("new_asr_calls") ==
         result.get("optimizer_updates") == 0 and
         result.get("model_promotion_allowed") is False and
         result.get("training_admitted") is False and
         result.get("human_gold") is False and
         result.get("real_far_per_hour") == "NOT_RUN" and
         result.get("real_frr") == "NOT_RUN" and
         result.get("ssc305_board_profile") == "NOT_RUN",
         "EN_MERO_RESULT_MAY_NOT_PROMOTE")
    for row in result["clip_results"]:
        need(type(row) is dict and row.get("training_admitted") is False and
             row.get("human_gold") is False and
             row.get("trusted_negative") is False and
             type(row.get("events")) is list and
             type(row.get("pcm_sha256")) is str and
             len(row["pcm_sha256"]) == 64,
             "EN_MERO_UNKNOWN_GOLD_OR_EVENT")
        for event in row["events"]:
            need(type(event) is dict and
                 event.get("mapped_keyword") == KEYWORD_ALIAS and
                 event.get("is_streaming_event") is True,
                 "EN_MERO_EVENT_UNKNOWN_KEYWORD")
    return True


def preflight_inputs(tts_zip: Path, archive: Path):
    """Genuine official phone/token and 63 PCM source readback, no KWS calls."""
    claim()
    with tempfile.TemporaryDirectory(prefix="mero-only-preflight-") as tmp:
        root = Path(tmp)
        parent = root / "source"
        parent.mkdir(mode=0o700)
        source_review.unpack_pinned(tts_zip, ORIGINAL_TTS_SHA, parent)
        plan, cells = audio_source.source_from_artifact(parent)
        need(len(cells) == 63 and
             plan.get("wake_phrases_requested") ==
             ["Hey Vosen", "Hey Mero"], "EN_MERO_PREFLIGHT_SOURCE_DRIFT")
        total_samples = 0
        for cell, wav, evidence in cells:
            samples, pcm_sha = reference.read_pcm(wav)
            reference.verify_source_readback(wav, evidence, samples, pcm_sha)
            total_samples += len(samples)
        model_root = root / "model"
        model_root.mkdir(mode=0o700)
        parts, digest = reference.extract_locked_model(
            archive, "zh-en", model_root)
        need(digest == PUBLISHED_ZH_EN_SHA,
             "EN_MERO_MODEL_SHA_DRIFT")
        keyword = encode_one_keyword(parts, root)
        return {
            "schema": "english63-zh-en-mero-source-only-preflight-v1",
            "status": "63_ORIGINAL_PCM_AND_OFFICIAL_MERO_PHONEMES_READY",
            "original_tts_zip_sha256": ORIGINAL_TTS_SHA,
            "model_archive_sha256": digest,
            "source_clip_count": len(cells),
            "total_audio_samples": total_samples,
            "keyword_sha256": reference.sha_file(keyword),
            "keyword_line": keyword.read_text(encoding="utf-8").strip(),
            "source_audio_regenerated": False,
            "new_tts_calls": 0, "new_asr_calls": 0,
            "kws_inference_calls": 0, "optimizer_updates": 0,
            "model_promotion_allowed": False,
            "training_admitted": False,
            "vosen_not_evaluated": True,
        }


def execute(tts_zip: Path, archive: Path):
    claim()
    protected_main()
    import sherpa_onnx
    with tempfile.TemporaryDirectory(prefix="en63-meroonly-") as tmp:
        root = Path(tmp)
        source_parent = root / "tts-source"
        source_parent.mkdir(mode=0o700)
        source_review.unpack_pinned(tts_zip, ORIGINAL_TTS_SHA,
                                    source_parent)
        plan, cells = audio_source.source_from_artifact(source_parent)
        need(len(cells) == 63 and plan.get("wake_phrases_requested") ==
             ["Hey Vosen", "Hey Mero"], "EN_MERO_ORIGINAL_63_MISMATCH")
        model_root = root / "model"
        model_root.mkdir(mode=0o700)
        parts, archive_sha = reference.extract_locked_model(
            archive, "zh-en", model_root)
        need(archive_sha == PUBLISHED_ZH_EN_SHA,
             "EN_MERO_MODEL_SOURCE_SHA")
        keywords = encode_one_keyword(parts, root)
        spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(parts["tokens"]), encoder=str(parts["encoder"]),
            decoder=str(parts["decoder"]), joiner=str(parts["joiner"]),
            keywords_file=str(keywords),
            num_threads=1, sample_rate=16000, feature_dim=80,
            max_active_paths=4, keywords_score=1.0,
            keywords_threshold=0.25, num_trailing_blanks=1, provider="cpu")
        start = time.perf_counter()
        rows = []
        by_requested_text = Counter()
        audio_samples = 0
        for cell, wav, evidence in sorted(cells,
                                           key=lambda x: x[0]["cell_id"]):
            samples, pcm_hash = reference.read_pcm(wav)
            reference.verify_source_readback(wav, evidence, samples, pcm_hash)
            events, steps = reference.stream_events(spotter, samples)
            need(all(event.get("mapped_keyword") == KEYWORD_ALIAS
                     for event in events), "EN_MERO_UNKNOWN_WAKE_EVENT")
            by_requested_text[cell["text_requested"]] += len(events)
            audio_samples += len(samples)
            rows.append({
                "cell_id": cell["cell_id"],
                "wav_sha256": evidence["wav_sha256"],
                "pcm_sha256": pcm_hash,
                "voice_source_family": cell["source_family_id"],
                "voice_role": cell["role"],
                "requested_text": cell["text_requested"],
                "requested_intent": cell["intent"],
                "audio_samples": len(samples),
                "decoder_steps": steps,
                "events": events,
                "training_admitted": False,
                "human_gold": False,
                "trusted_negative": False,
            })
        result = {
            "schema": POLICY,
            "status": "EXECUTED_REAL_ZH_EN_MERO_ONLY_WEAK_CLIP_EVENTS",
            "pretrained_archive_sha256": archive_sha,
            "pretrained_model": "zh-en",
            "pretrained_model_release_asset_id": 331699066,
            "original_tts_artifact_id": source_review.ARCHIVES["tts"][0],
            "original_tts_zip_sha256": ORIGINAL_TTS_SHA,
            "keyword_original_text": "Hey Mero",
            "keyword_alias": KEYWORD_ALIAS,
            "keyword_sha256": reference.sha_file(keywords),
            "keyword_line": keywords.read_text(encoding="utf-8").strip(),
            "source_clip_count": len(rows),
            "total_input_audio_samples": audio_samples,
            "total_clip_events": sum(len(row["events"]) for row in rows),
            "events_by_requested_text_not_gold": dict(by_requested_text),
            "clip_results": rows,
            "fixed_decoder": {
                "threads": 1, "feature_dim": 80, "max_active_paths": 4,
                "keywords_score": 1.0, "keywords_threshold": 0.25,
                "num_trailing_blanks": 1,
                "input_partition_samples": reference.FEED_SAMPLES,
                "end_padding_samples": reference.TAIL_SAMPLES,
            },
            "sherpa_onnx_version": version("sherpa-onnx"),
            "host_elapsed_sec": round(time.perf_counter() - start, 4),
            "vosen_not_evaluated": True,
            "new_tts_calls": 0, "new_asr_calls": 0,
            "optimizer_updates": 0, "model_promotion_allowed": False,
            "training_admitted": False, "human_gold": False,
            "real_far_per_hour": "NOT_RUN", "real_frr": "NOT_RUN",
            "ssc305_board_profile": "NOT_RUN",
        }
        validate_receipt(result)
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--tts-zip", type=Path)
    p.add_argument("--model-archive", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    try:
        claim()
        need(not (args.execute and args.preflight),
             "EN_MERO_NO_MIXED_EXECUTION_MODES")
        if args.preflight:
            need(args.tts_zip is not None and
                 args.model_archive is not None and args.output is None,
                 "EN_MERO_EXPLICIT_PREFLIGHT_INPUTS")
            print(json.dumps(preflight_inputs(args.tts_zip, args.model_archive),
                             sort_keys=True))
            return 0
        if not args.execute:
            need(all(x is None for x in
                     (args.tts_zip, args.model_archive, args.output)),
                 "EN_MERO_NO_INFERENCE_DEFAULT")
            print(json.dumps({"schema": POLICY, "candidate": "Hey Mero",
                              "real_inference_started": False,
                              "vosen_not_evaluated": True,
                              "model_promotion_allowed": False}))
            return 0
        need(args.tts_zip is not None and
             args.model_archive is not None and
             args.output is not None and args.output.is_absolute() and
             args.output.parent.is_dir() and
             not args.output.parent.is_symlink() and
             not args.output.exists(),
             "EN_MERO_CREATE_ONLY_OUTPUTS")
        result = execute(args.tts_zip, args.model_archive)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, sort_keys=True, indent=2, ensure_ascii=False)
            f.write("\n")
        args.output.chmod(0o600)
        print(json.dumps({"status": result["status"],
                          "source_clips": result["source_clip_count"],
                          "event_count": result["total_clip_events"],
                          "events_by_requested_text_not_gold":
                              result["events_by_requested_text_not_gold"],
                          "vosen_not_evaluated": True,
                          "model_promotion_allowed": False},
                         sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, wave.Error) as exc:
        print("EN_MERO_ONLY_STOP: " + type(exc).__name__ +
              ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
