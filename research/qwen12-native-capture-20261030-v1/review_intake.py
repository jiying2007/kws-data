"""Bind locally exported blind-audition observations to the immutable Qwen12 audio.

This accepts *self-reported observations*, not human-gold signatures, label
admission, rights clearance, test-set authority, or model-training permission.
It never reads ASR transcripts, calls models, or changes any source WAV.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

import fresh_contract as fresh

SOURCE_MAP_SHA256 = "14b1fa88d20acb88a4f05f214adebdc6a1762f0929cb6395e9f5098295793b01"
ORIGINAL_ARCHIVE_SHA256 = "78f8b8736db1ab1dafc15a3bdddbbab64799ee0c975b6f3dab647604d633a8d1"
REVIEW_SCHEMA = "qwen12-manual-auditory-observation-v1"
VERDICTS = ("not_reviewed", "complete", "ambiguous", "truncated", "noise", "no_audio")
TARGETS = ("你好小窝", "小窝小窝")
REVIEW_KEYS = {"schema", "reviewer_entered_identity", "reviewer_identity_verified",
               "reviewer_saw_transcripts_before_listening", "created_at_claimed_by_browser",
               "source", "independent_review_verified", "training_admitted",
               "qualification_allowed", "rows"}
ROW_KEYS = {"audio_id", "wav_sha256", "heard_text", "verdict", "notes"}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def _unique(pairs):
    obj = {}
    for k, v in pairs:
        require(k not in obj, "DUPLICATE_JSON_KEY")
        obj[k] = v
    return obj


def _finite_float(token):
    value = float(token)
    require(math.isfinite(value), "NONFINITE_JSON")
    return value


def read_json(path, limit):
    data = fresh.bounded_read(path, limit)
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")),
                          parse_float=_finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError("INVALID_JSON") from err


def safe_text(value, max_len, *, allow_empty=True):
    require(type(value) is str and len(value) <= max_len, "TEXT_TYPE_OR_LENGTH")
    require(allow_empty or len(value.strip()) > 0, "EMPTY_REQUIRED_TEXT")
    require(all(ord(ch) >= 32 and ord(ch) != 127 for ch in value), "CONTROL_CHAR")
    return value


def normalize_heard(text):
    # *Only* typographical normalization. Never convert 窝/沃/屋/吴/王.
    value = unicodedata.normalize("NFKC", text)
    return re.sub(r'[\s，,。.!！?？、:：;；"“”‘’]', "", value)


def verify_handoff(root):
    root = Path(root)
    job = fresh.validate_blind(root / "blind")  # validates all 12 original WAV/PCM extents and hashes
    source_path = root / "control" / "source-map.json"
    raw = fresh.bounded_read(source_path, 65536)
    require(hashlib.sha256(raw).hexdigest() == SOURCE_MAP_SHA256, "SOURCE_MAP_DRIFT")
    sources = read_json(source_path, 65536)
    require(type(sources) is dict and sources.get("schema") == "qwen16-label-source-map-v1" and
            sources.get("archive_sha256") == ORIGINAL_ARCHIVE_SHA256 and
            sources.get("rights_policy_review") == "PENDING", "UNREVIEWED_SOURCE_IDENTITY_REQUIRED")
    mappings = sources.get("rows")
    require(type(mappings) is list and len(mappings) == 12 and
            type(job.get("clips")) is list and len(job["clips"]) == 12, "SOURCE_COUNT")
    source_ids = set()
    for audio, mapping in zip(job["clips"], mappings):
        require(type(mapping) is dict and
                mapping.get("audio_id") == audio["audio_id"] and
                mapping.get("file_sha256") == audio["wav_sha256"] and
                mapping.get("proposed_split") == "train" and
                mapping.get("source_type") == "tts" and
                mapping.get("synthetic") is True and
                mapping.get("actual_text") is None and mapping.get("actual_tokens") is None and
                all(mapping.get(k) is False for k in ("ctc_training_allowed",
                     "evaluation_gold_allowed", "qualification_allowed")), "SOURCE_ROLE_OR_AUDIO_DRIFT")
        sid = mapping.get("source_id")
        require(type(sid) is str and sid not in source_ids, "DUPLICATE_SOURCE_ID")
        source_ids.add(sid)
    require(len(source_ids) == 12, "SOURCE_ID_COUNT")
    return mappings


def audit_observation(review, mappings):
    require(type(review) is dict and set(review) == REVIEW_KEYS and
            review["schema"] == REVIEW_SCHEMA and
            review["source"] == "qwen12-blind-main-41055537577", "REVIEW_SCHEMA_OR_KEYS")
    safe_text(review["reviewer_entered_identity"], 128, allow_empty=False)
    safe_text(review["created_at_claimed_by_browser"], 64, allow_empty=False)
    # Browser identity/time are only claims. No review signature is authenticated.
    require(review["reviewer_identity_verified"] is False and
            review["independent_review_verified"] is False and
            review["training_admitted"] is False and
            review["qualification_allowed"] is False and
            type(review["reviewer_saw_transcripts_before_listening"]) is bool,
            "FORBIDDEN_CLAIM")
    rows = review["rows"]
    require(type(rows) is list and len(rows) == 12, "REVIEW_ROW_COUNT")
    by_audio = {m["audio_id"]: m for m in mappings}
    require(len(by_audio) == 12, "SOURCE_DUPLICATE")
    seen = set()
    output = []
    for row in rows:
        require(type(row) is dict and set(row) == ROW_KEYS, "REVIEW_ROW_KEYS")
        audio_id = row["audio_id"]
        require(type(audio_id) is str and audio_id in by_audio and audio_id not in seen,
                "UNKNOWN_OR_DUPLICATE_AUDIO_ID")
        seen.add(audio_id)
        mapping = by_audio[audio_id]
        require(row["wav_sha256"] == mapping["file_sha256"] and
                type(row["wav_sha256"]) is str, "REVIEW_WAV_MISMATCH")
        heard = safe_text(row["heard_text"], 96)
        safe_text(row["notes"], 256)
        verdict = row["verdict"]
        require(type(verdict) is str and verdict in VERDICTS, "REVIEW_VERDICT")
        if verdict in ("not_reviewed", "no_audio"):
            require(not heard.strip(), "NO_AUDIO_MAY_NOT_HAVE_HEARD_TEXT")
        if verdict == "complete":
            require(bool(normalize_heard(heard)), "COMPLETE_REQUIRES_VERBATIM_TEXT")
        normalized = normalize_heard(heard)
        # This remains a user observation even if all records were filled.
        output.append({"audio_id": audio_id, "wav_sha256": mapping["file_sha256"],
                       "pcm_sha256": mapping["pcm_sha256"],
                       "audio_frames": mapping["audio_frames"],
                       "heard_text": heard, "normalized_text": normalized,
                       "quality_verdict": verdict,
                       "target_text_observed": verdict == "complete" and normalized in TARGETS,
                       "training_admitted": False})
    require(seen == set(by_audio), "REVIEW_INCOMPLETE_ROSTER")
    output.sort(key=lambda x: x["audio_id"])
    counts = {status: sum(r["quality_verdict"] == status for r in output) for status in VERDICTS}
    return {
        "schema": "qwen12-manual-observation-verified-binding-v1",
        "evidence_class": "self_reported_auditory_observation_only",
        "source_map_sha256": SOURCE_MAP_SHA256,
        "original_archive_sha256": ORIGINAL_ARCHIVE_SHA256,
        "observed_clip_count": 12,
        "quality_counts": counts,
        "target_text_observed_count": sum(r["target_text_observed"] for r in output),
        "reviewer_identity_authenticated": False,
        "blind_independence_authenticated": False,
        "reviewer_saw_transcripts_before_listening_claim": review["reviewer_saw_transcripts_before_listening"],
        "rights_policy_review": "PENDING",
        "training_admitted": False, "qualification_allowed": False,
        "human_gold": False, "release_authorized": False,
        "status": "OBSERVATIONS_BOUND_TO_SOURCE_NOT_ADMITTED",
        "rows": output,
    }


def run(handoff_root, review_file):
    mappings = verify_handoff(handoff_root)
    review = read_json(review_file, 98304)
    return audit_observation(review, mappings)


def publish_new(path, result):
    dest = Path(path)
    require(dest.parent.is_dir() and not dest.is_symlink() and not dest.exists(),
            "OUTPUT_MUST_BE_NEW")
    body = fresh.canonical(result) + b"\n"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".qwen-review-", suffix=".tmp",
                                         dir=dest.parent, delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, dest, follow_symlinks=False)  # create only; never clobber
        require(dest.read_bytes() == body, "OUTPUT_CHANGED")
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.handoff_root, args.review)
        publish_new(args.output, result)
    except (ValueError, OSError) as err:
        print("QWEN12_REVIEW_NOT_ADMITTED: " + type(err).__name__, file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "observed_clip_count": 12,
                      "training_admitted": False, "qualification_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
