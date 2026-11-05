"""Preserve publisher text without relabeling it as human-verified speech.

Only a separately supplied, audio-only content observation can enable the new
ICL path. These are machine evidence, not authenticated human annotations.
"""
from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
import re
import unicodedata

from .source_registry import canonical, read_bytes, read_json, require

FORMATS = ("aishell3-content-v1", "path-text-tsv-v1")
PROVIDED = "dataset-provided"
CHECKED = "dataset-provided-content-checked"
CHECK_SCHEMA = "kws-reference-content-checks-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
REV = re.compile(r"[0-9a-f]{40}\Z")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(text):
    """Fold orthography without deleting English word boundaries/contractions."""
    value = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    value = " ".join("".join(
        " " if unicodedata.category(c).startswith("P") and c != "'" else c
        for c in value).split())
    # Mandarin transcribers may insert spaces or punctuation between characters.
    # Keep every other word boundary: "the rapist" must not equal "therapist".
    return re.sub(r"(?<=[\u3400-\u4dbf\u4e00-\u9fff]) +"
                  r"(?=[\u3400-\u4dbf\u4e00-\u9fff])", "", value)


def _text(text):
    require(type(text) is str and 1 <= len(text) <= 180 and
            all(ord(c) >= 32 for c in text) and any(c.isalnum() for c in text)
            and normalize(text), "REFERENCE_TEXT")


def parse_record(format_name, line):
    require(format_name in FORMATS and type(line) is str and
            "\n" not in line and "\r" not in line and "\t" in line,
            "TRANSCRIPT_SOURCE_RECORD")
    key, value = line.split("\t", 1)
    path = PurePosixPath(key)
    require(key and not path.is_absolute() and path.as_posix() == key and
            all(p not in (".", "..", "") for p in path.parts) and
            "\\" not in key, "TRANSCRIPT_RECORD_PATH")
    if format_name == "aishell3-content-v1":
        require(path.name == key and re.fullmatch(r"SSB[0-9]+\.wav", key),
                "AISHELL_TRANSCRIPT_ID")
        tokens = value.split()
        require(tokens and len(tokens) % 2 == 0, "AISHELL_CHARACTER_PINYIN_PAIRS")
        text = "".join(tokens[::2])
    else:
        text = value.strip()
    _text(text)
    return key, text


def document(path=None, expected_sha=None, format_name=None):
    if path is None:
        require(expected_sha is None and format_name is None,
                "TRANSCRIPT_SOURCE_REQUIRED")
        return None
    require(type(expected_sha) is str and SHA.fullmatch(expected_sha) and
            format_name in FORMATS, "PINNED_TRANSCRIPT_SOURCE_REQUIRED")
    raw = read_bytes(path, 32 * 1024 * 1024)
    require(sha(raw) == expected_sha, "TRANSCRIPT_SOURCE_CHANGED")
    rows = {}
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        key, text = parse_record(format_name, line)
        require(key not in rows, "DUPLICATE_TRANSCRIPT_RECORD")
        rows[key] = {"source_line_number": number, "source_record": line, "text": text}
    return {"sha256": expected_sha, "format": format_name, "rows": rows}


def validate_observation(row):
    require(type(row) is dict and set(row) == {
        "engine", "model_id", "model_revision", "model_assets_sha256",
        "wav_sha256", "pcm_sha256", "sample_rate_hz", "audio_frames",
        "actual_text", "input_mode", "decoder_completed"}, "REFERENCE_CONTENT_FIELDS")
    require(all(type(row[k]) is str and 1 <= len(row[k]) <= 200
                for k in ("engine", "model_id")) and
            type(row["model_revision"]) is str and REV.fullmatch(row["model_revision"]),
            "REFERENCE_CONTENT_MODEL")
    for key in ("model_assets_sha256", "wav_sha256", "pcm_sha256"):
        require(type(row[key]) is str and SHA.fullmatch(row[key]), "REFERENCE_CONTENT_SHA")
    require(type(row["sample_rate_hz"]) is int and row["sample_rate_hz"] in (16000, 24000)
            and type(row["audio_frames"]) is int and row["audio_frames"] > 0 and
            row["input_mode"] == "audio-only" and row["decoder_completed"] is True and
            type(row["actual_text"]) is str and len(row["actual_text"]) <= 1024,
            "REFERENCE_CONTENT_NOT_AUDIO_ONLY_COMPLETE")


def content_checks(path=None):
    if path is None:
        return None, {}
    value = read_json(path)
    require(type(value) is dict and set(value) == {"schema", "observations"} and
            value["schema"] == CHECK_SCHEMA and type(value["observations"]) is list and
            len(value["observations"]) <= 768, "REFERENCE_CONTENT_CHECK_SCHEMA")
    by_wav, seen = {}, set()
    for row in value["observations"]:
        validate_observation(row)
        key = (row["wav_sha256"], row["engine"])
        require(key not in seen, "DUPLICATE_REFERENCE_CONTENT_CHECK")
        seen.add(key)
        by_wav.setdefault(row["wav_sha256"], []).append(row)
        require(len(by_wav[row["wav_sha256"]]) <= 3, "REFERENCE_CONTENT_COUNT")
    return sha(canonical(value)), by_wav


def _checked(text, observations, ref):
    for row in observations:
        validate_observation(row)
        require(all(row[a] == ref[b] for a, b in (
            ("wav_sha256", "wav_sha256"), ("pcm_sha256", "pcm_sha256"),
            ("sample_rate_hz", "sample_rate_hz"), ("audio_frames", "frames"))),
            "REFERENCE_CONTENT_CHECK_AUDIO_MISMATCH")
    return bool(observations) and all(normalize(row["actual_text"]) == normalize(text)
                                      for row in observations)


def attach(index, row, measured, source, checks):
    if source is None:
        return None, "unknown", None
    key = (PurePosixPath(row["original_path"]).name
           if source["format"] == "aishell3-content-v1" else row["original_path"])
    item = source["rows"].get(key)
    if item is None:
        return None, "unknown", None  # An absent publisher record is not invented.
    if source["format"] == "aishell3-content-v1":
        require(key[:7] == row["speaker_key"], "TRANSCRIPT_ORIGINAL_SPEAKER_MISMATCH")
    checks_sha, by_wav = checks
    observations = by_wav.get(measured["wav_sha256"], [])
    text = item["text"]
    evidence = {"dataset_id": index["dataset_id"], "dataset_release": index["dataset_release"],
                "original_path": row["original_path"],
                "original_sha256": row["original_sha256"],
                "normalized_wav_sha256": measured["wav_sha256"],
                "source_format": source["format"],
                "source_document_sha256": source["sha256"],
                "source_line_number": item["source_line_number"],
                "source_record": item["source_record"],
                "source_record_sha256": sha(item["source_record"].encode("utf-8")),
                "text_sha256": sha(text.encode("utf-8")),
                "content_checks_sha256": checks_sha, "content_observations": observations}
    status = CHECKED if _checked(text, observations, measured) else PROVIDED
    return text, status, evidence


def validate_reference(ref):
    status, text, proof = (ref.get("reference_text_status"), ref.get("reference_text"),
                           ref.get("reference_text_provenance"))
    if status == "unknown":
        require(text is None and proof is None, "UNVERIFIED_TRANSCRIPT_MUST_BE_NULL")
        return
    _text(text)
    if status == "verified":
        # Existing explicit review is not manufactured by dataset ingestion.
        require(proof is None, "VERIFIED_TRANSCRIPT_NOT_DATASET_PROVIDED")
        return
    require(status in (PROVIDED, CHECKED) and type(proof) is dict and set(proof) == {
        "dataset_id", "dataset_release", "original_path", "original_sha256",
        "normalized_wav_sha256", "source_format", "source_document_sha256",
        "source_line_number", "source_record", "source_record_sha256", "text_sha256",
        "content_checks_sha256", "content_observations"}, "TRANSCRIPT_PROVENANCE")
    for field in ("original_sha256", "normalized_wav_sha256", "source_document_sha256",
                  "source_record_sha256", "text_sha256"):
        require(type(proof[field]) is str and SHA.fullmatch(proof[field]),
                "TRANSCRIPT_PROVENANCE_SHA")
    require(proof["dataset_id"] == ref["dataset_id"] and
            type(proof["dataset_release"]) is str and 1 <= len(proof["dataset_release"]) <= 80 and
            type(proof["original_path"]) is str and 1 <= len(proof["original_path"]) <= 260 and
            type(proof["source_record"]) is str and 1 <= len(proof["source_record"]) <= 8192 and
            proof["original_sha256"] == ref["parent_source_sha256"] and
            proof["normalized_wav_sha256"] == ref["wav_sha256"] and
            type(proof["source_line_number"]) is int and proof["source_line_number"] > 0 and
            sha(proof["source_record"].encode("utf-8")) == proof["source_record_sha256"] and
            sha(text.encode("utf-8")) == proof["text_sha256"], "TRANSCRIPT_ORIGINAL_BINDING")
    key, actual = parse_record(proof["source_format"], proof["source_record"])
    expected = (PurePosixPath(proof["original_path"]).name
                if proof["source_format"] == "aishell3-content-v1" else proof["original_path"])
    require(key == expected and actual == text, "TRANSCRIPT_RECORD_TEXT_MISMATCH")
    observations = proof["content_observations"]
    require(type(observations) is list and len(observations) <= 3,
            "REFERENCE_CONTENT_COUNT")
    require(len({row.get("engine") for row in observations if type(row) is dict}) ==
            len(observations), "DUPLICATE_REFERENCE_CONTENT_CHECK")
    if proof["content_checks_sha256"] is not None:
        require(type(proof["content_checks_sha256"]) is str and
                SHA.fullmatch(proof["content_checks_sha256"]), "REFERENCE_CONTENT_MANIFEST_SHA")
    require(not observations or proof["content_checks_sha256"] is not None,
            "REFERENCE_CONTENT_MANIFEST_MISSING")
    computed = CHECKED if _checked(text, observations, ref) else PROVIDED
    require(computed == status, "REFERENCE_CONTENT_STATUS_NOT_EVIDENCED")
