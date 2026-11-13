"""Real, frozen GigaSpeech Zipformer vendor-positive controls.

Diagnose why the same official streaming KWS model emitted zero events on
63 unchanged English synthetic clips. Here only the official model archive's
own 0.wav/1.wav and test_keywords.txt are used. No new source audio, no
TTS/ASR, no training, no keyword-threshold sweep, no model promotion.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english63_zipformer_events_once as prior

POLICY = "english63-gigaspeech-vendor-positive-canary-v1"
BASELINE_RUN_ID = 43443799469
BASELINE_ARTIFACT_ID = 14046722243
BASELINE_EVENTS_PER_63 = 0
MODEL_ASSET_ID = 166405338
MODEL_ARCHIVE_SHA256 = "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a"
MODEL_ARCHIVE_BYTES = 17626723
VENDOR_FILES = (
    ("0.wav", "LIGHT UP"),
    ("1.wav", "LOVELY CHILD"),
)
SOURCE_REUSE = False


def need(ok, code):
    if not ok:
        raise ValueError(code)


def source_contract(archive):
    """A source-only guard; actual archive checks run before any decoder."""
    need(isinstance(archive, Path) and archive.is_absolute() and
         archive.is_file() and not archive.is_symlink() and
         archive.stat().st_size == MODEL_ARCHIVE_BYTES,
         "EN_VENDOR_ARCHIVE_GEOMETRY")
    need(prior.sha_file(archive) == MODEL_ARCHIVE_SHA256,
         "EN_VENDOR_ARCHIVE_DIGEST_DRIFT")
    return True


def verified_vendor_files(parts, model_hash):
    need(model_hash == MODEL_ARCHIVE_SHA256,
         "EN_VENDOR_LOCK_NOT_MATCHING_ARCHIVE")
    model_root = parts["tokens"].parent
    keywords = model_root / "test_wavs" / "test_keywords.txt"
    need(keywords.is_file() and not keywords.is_symlink() and
         0 < keywords.stat().st_size <= 16384,
         "EN_VENDOR_ORIGINAL_KEYWORD_LIST_MISSING")
    raw = keywords.read_text(encoding="utf-8")
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    need(1 <= len(lines) <= 80 and
         all(1 <= len(x) <= 256 for x in lines),
         "EN_VENDOR_KEYWORD_LIST_GEOMETRY")
    wavs = {}
    for filename, _ in VENDOR_FILES:
        path = model_root / "test_wavs" / filename
        need(path.is_file() and not path.is_symlink() and
             path.stat().st_size <= 5 * 1024 * 1024,
             "EN_VENDOR_ORIGINAL_WAV_MISSING")
        wavs[filename] = path
    return keywords, wavs


def validate_receipt(report):
    need(type(report) is dict and report.get("schema") == POLICY and
         report.get("status") ==
         "EXECUTED_REAL_OFFICIAL_VENDOR_CONTROLS_NOT_ENGLISH63_QUALIFICATION" and
         report.get("baseline_english63_event_run_id") == BASELINE_RUN_ID and
         report.get("baseline_english63_event_count") == 0 and
         report.get("model_archive_sha256") == MODEL_ARCHIVE_SHA256 and
         report.get("vendor_clip_count") == len(VENDOR_FILES) and
         report.get("new_tts_calls") == report.get("new_asr_calls") ==
         report.get("optimizer_updates") == 0 and
         report.get("source_english63_replayed") is False and
         report.get("formal_training_admitted") is False and
         report.get("human_gold") is False and
         report.get("model_promotion_allowed") is False and
         report.get("independent_human_far_frr") == "NOT_RUN" and
         report.get("ssc305_board_profile") == "NOT_RUN" and
         type(report.get("canary_rows")) is list and
         len(report["canary_rows"]) == len(VENDOR_FILES),
         "EN_VENDOR_CANARY_RECEIPT_UNSAFE")
    seen = set()
    for row in report["canary_rows"]:
        need(type(row) is dict and
             row.get("filename") in {x[0] for x in VENDOR_FILES} and
             row.get("filename") not in seen and
             row.get("documented_vendor_keyword") ==
             dict(VENDOR_FILES)[row["filename"]] and
             type(row.get("audio_samples")) is int and
             row["audio_samples"] > 0 and
             type(row.get("decoded_steps")) is int and
             row["decoded_steps"] >= 0 and
             type(row.get("events")) is list and
             type(row.get("documented_keyword_detected")) is bool and
             row.get("acoustic_gold") is False and
             type(row.get("wav_sha256")) is str and
             len(row["wav_sha256"]) == 64 and
             type(row.get("pcm_sha256")) is str and
             len(row["pcm_sha256"]) == 64 and
             len(row["events"]) <= prior.EVENT_LIMIT,
             "EN_VENDOR_ROW_SCHEMA")
        seen.add(row["filename"])
        for event in row["events"]:
            need(type(event) is dict and
                 type(event.get("keyword")) is str and
                 type(event.get("token_timestamps_sec")) is list and
                 event.get("is_streaming_event") is True,
                 "EN_VENDOR_EVENT_SHAPE")
        need(row["documented_keyword_detected"] is any(
             event["keyword"].strip().upper() ==
             row["documented_vendor_keyword"] for event in row["events"]),
             "EN_VENDOR_FORGED_POSITIVE_CONTROL")
    need(report.get("vendor_positive_control_passed") is
         all(r["documented_keyword_detected"] for r in report["canary_rows"]),
         "EN_VENDOR_FALSE_POSITIVE_CONTROL_PASS")
    return True


def execute(archive):
    source_contract(archive)
    import sherpa_onnx
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gigaspeech-vendor-canary-") as tmp:
        root = Path(tmp)
        model_root = root / "model"
        model_root.mkdir(mode=0o700)
        parts, model_hash = prior.extract_locked_model(archive, "gigaspeech",
                                                        model_root)
        keyword_path, wavs = verified_vendor_files(parts, model_hash)
        spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(parts["tokens"]),
            encoder=str(parts["encoder"]),
            decoder=str(parts["decoder"]),
            joiner=str(parts["joiner"]),
            keywords_file=str(keyword_path),
            num_threads=1, sample_rate=prior.SAMPLE_RATE, feature_dim=80,
            max_active_paths=4, keywords_score=1.0, keywords_threshold=0.25,
            num_trailing_blanks=1, provider="cpu")
        rows = []
        for filename, documented_keyword in VENDOR_FILES:
            path = wavs[filename]
            samples, pcm_sha = prior.read_pcm(path)
            events, decoded = prior.stream_events(spotter, samples)
            matched = any(str(row["keyword"]).upper().strip() ==
                          documented_keyword for row in events)
            rows.append({
                "filename": filename,
                "documented_vendor_keyword": documented_keyword,
                "documented_keyword_detected": matched,
                "wav_sha256": prior.sha_file(path),
                "pcm_sha256": pcm_sha,
                "audio_samples": len(samples),
                "decoded_steps": decoded,
                "events": events,
                "acoustic_gold": False,
            })
        result = {
            "schema": POLICY,
            "status":
                "EXECUTED_REAL_OFFICIAL_VENDOR_CONTROLS_NOT_ENGLISH63_QUALIFICATION",
            "model": "gigaspeech",
            "model_archive_sha256": model_hash,
            "model_release_asset_id": MODEL_ASSET_ID,
            "model_digest_provenance": "observed_official_release_bytes_not_upstream_attestation",
            "keyword_file_sha256": prior.sha_file(keyword_path),
            "sherpa_onnx_version": version("sherpa-onnx"),
            "vendor_clip_count": len(rows),
            "canary_rows": rows,
            "vendor_positive_control_passed": all(
                row["documented_keyword_detected"] for row in rows),
            "fixed_configuration": {
                "num_threads": 1, "feature_dim": 80,
                "max_active_paths": 4, "keywords_score": 1.0,
                "keywords_threshold": 0.25, "num_trailing_blanks": 1,
                "input_partition_samples": prior.FEED_SAMPLES,
                "end_padding_samples": prior.TAIL_SAMPLES,
            },
            "baseline_english63_event_run_id": BASELINE_RUN_ID,
            "baseline_english63_artifact_id": BASELINE_ARTIFACT_ID,
            "baseline_english63_event_count": BASELINE_EVENTS_PER_63,
            "seconds_host_observed": round(time.perf_counter() - started, 3),
            "source_english63_replayed": False,
            "new_tts_calls": 0,
            "new_asr_calls": 0,
            "optimizer_updates": 0,
            "formal_training_admitted": False,
            "human_gold": False,
            "model_promotion_allowed": False,
            "independent_human_far_frr": "NOT_RUN",
            "ssc305_board_profile": "NOT_RUN",
        }
        validate_receipt(result)
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    try:
        if not args.execute:
            need(args.archive is None and args.output is None,
                 "EN_VENDOR_NO_INFER_WITHOUT_EXECUTE")
            print(json.dumps({"schema": POLICY, "vendor_files":
                              [x[0] for x in VENDOR_FILES],
                              "real_inference_run": False,
                              "model_promotion_allowed": False}))
            return 0
        need(args.archive is not None and args.output is not None and
             args.output.is_absolute() and args.output.parent.is_dir() and
             not args.output.exists() and not args.output.is_symlink(),
             "EN_VENDOR_CREATE_ONLY_PATH")
        result = execute(args.archive)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False,
                      sort_keys=True)
            handle.write("\n")
        args.output.chmod(0o600)
        print(json.dumps({"status": result["status"],
                          "positive_control_passed":
                              result["vendor_positive_control_passed"],
                          "vendor_clips": result["vendor_clip_count"],
                          "events_per_vendor_clip": {
                              x["filename"]: len(x["events"])
                              for x in result["canary_rows"]},
                          "model_promotion_allowed": False},
                         sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
        print("EN_VENDOR_CANARY_STOP: " +
              type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
