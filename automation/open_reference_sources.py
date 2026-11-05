#!/usr/bin/env python3
"""Build a PRIVATE open-corpus reference catalog for the single research importer.

This is an offline, no-model, non-admitting candidate screen. Open recording
licences NEVER automatically grant permission to clone a recognizable speaker.
The research-only importer independently requires frozen dataset/release
research-use scope and source evidence before private publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "auto-tts-voice-clone-20261102"))
import bulk_pool as pool  # noqa: E402
from kws_data_contract import reference_text  # noqa: E402

REGISTRY = ROOT / "automation" / "open_reference_datasets.v1.json"
REGISTRY_BLOB = "0050babfa9b1b48e7013d6847b76d64e3b904324"
INDEX_SCHEMA = "kws-local-open-reference-index-v1"
CONVERSION_SCHEMA = "kws-local-reference-conversion-receipt-v1"
CATALOG_SCHEMA = "kws-local-open-audio-catalog-v2"
HEX = re.compile(r"[0-9a-f]{64}\Z")
IDENT = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
MAX_CANDIDATE_ROWS = 256
MAX_SPEAKERS = 32
MAX_ORIGINAL_BYTES = 16 * 1024 * 1024
MAX_SOURCE_TERMS_BYTES = 128 * 1024


def require(value, msg):
    if not value:
        raise ValueError(msg)


def sha(value):
    require(type(value) is str and HEX.fullmatch(value), "SHA256_REQUIRED")
    return value


def confined_file(root, relative, limit):
    root = Path(root).absolute()
    require(root.is_dir() and
            not any(x.is_symlink() for x in (root, *root.parents)),
            "INPUT_ROOT_ALIAS")
    require(type(relative) is str and 1 <= len(relative) <= 260 and
            "\\" not in relative and "\0" not in relative, "INPUT_PATH_SYNTAX")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and pure.as_posix() == relative and
            all(x not in (".", "..", ".git") and
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,70}", x)
                for x in pure.parts), "INPUT_PATH_TRAVERSAL")
    path = root
    for part in pure.parts:
        path = path / part
        require(not path.is_symlink(), "INPUT_PATH_ALIAS")
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and
            1 <= st.st_size <= limit, "INPUT_FILE_SIZE_OR_TYPE")
    return path


def sha_file(path, limit):
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and
            1 <= st.st_size <= limit, "INPUT_FILE_SIZE_OR_TYPE")
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            n += len(block)
            require(n <= limit, "INPUT_READ_LIMIT")
            h.update(block)
        end = os.fstat(stream.fileno())
    require(n == st.st_size and
            (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink) ==
            (end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns, end.st_ctime_ns, end.st_nlink),
            "INPUT_CHANGED_DURING_READ")
    return h.hexdigest()


def registry():
    raw = REGISTRY.read_bytes()
    expect = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    require(expect == REGISTRY_BLOB, "SOURCE_CANDIDATE_REGISTRY_DRIFT")
    data = pool.read_json(REGISTRY, 65536)
    require(type(data) is dict and data.get("schema") == "kws-public-reference-candidates-v1"
            and data.get("rights_authenticated") is False
            and data.get("voice_cloning_authorized") is False
            and data.get("training_admitted") is False
            and set(data) == {"schema", "policy", "verified_review_date",
                              "dataset_candidates", "rights_authenticated",
                              "voice_cloning_authorized", "training_admitted"},
            "INVALID_PUBLIC_SOURCE_REGISTRY")
    entries = data.get("dataset_candidates")
    require(type(entries) is list and len(entries) >= 2, "PUBLIC_SOURCE_CANDIDATE_COUNT")
    ids = set()
    for item in entries:
        require(type(item) is dict and
                type(item.get("dataset_id")) is str and
                IDENT.fullmatch(item["dataset_id"]) and
                item["dataset_id"] not in ids and
                item.get("reference_status") in
                ("CANDIDATE_REVIEW_REQUIRED", "BLOCKED_FOR_CLONING"), "PUBLIC_SOURCE_POLICY")
        ids.add(item["dataset_id"])
    return {item["dataset_id"]: item for item in entries}


def check_index(index, dataset):
    require(type(index) is dict and
            set(index) == {"schema", "dataset_id", "dataset_release",
                           "license_snapshot_sha256", "rows"} and
            index["schema"] == INDEX_SCHEMA and
            index["dataset_id"] == dataset["dataset_id"] and
            type(index["dataset_release"]) is str and
            1 <= len(index["dataset_release"]) <= 80 and
            all(32 <= ord(ch) < 127 for ch in index["dataset_release"]),
            "SOURCE_INDEX_IDENTITY")
    sha(index["license_snapshot_sha256"])
    rows = index["rows"]
    require(type(rows) is list and 1 <= len(rows) <= MAX_CANDIDATE_ROWS,
            "SOURCE_INDEX_ROW_BUDGET")
    for row in rows:
        require(type(row) is dict and
                set(row) == {"speaker_key", "original_path", "original_sha256",
                             "normalized_path", "normalized_sha256", "conversion_receipt_path"} and
                type(row["speaker_key"]) is str and
                1 <= len(row["speaker_key"]) <= 128 and
                not any(ord(x) < 32 for x in row["speaker_key"]),
                "SOURCE_INDEX_ROW")
        sha(row["original_sha256"])
        sha(row["normalized_sha256"])
    return rows


def check_conversion(row, evidence_root):
    path = confined_file(evidence_root, row["conversion_receipt_path"], 65536)
    data = pool.read_json(path, 65536)
    require(type(data) is dict and
            set(data) == {"schema", "original_sha256", "normalized_sha256",
                          "conversion", "conversion_recipe_sha256"} and
            data["schema"] == CONVERSION_SCHEMA and
            data["original_sha256"] == row["original_sha256"] and
            data["normalized_sha256"] == row["normalized_sha256"],
            "CONVERSION_SOURCE_RECEIPT_MISMATCH")
    if data["conversion"] == "identity":
        require(data["conversion_recipe_sha256"] is None and
                row["original_sha256"] == row["normalized_sha256"],
                "INVALID_IDENTITY_CONVERSION")
    else:
        require(data["conversion"] == "externally-documented" and
                sha(data["conversion_recipe_sha256"]), "UNTRUSTED_CONVERSION_RECIPE")
    # This only checks the declared conversion evidence's consistency, not
    # that a third-party converter was actually executed or reviewed.


def candidates(index, original_root, normalized_root, evidence_root, terms_file,
               max_speakers, datasets=None, *, transcript_source=None,
               transcript_sha256=None, transcript_format=None, reference_content_checks=None):
    require(type(max_speakers) is int and
            1 <= max_speakers <= MAX_SPEAKERS, "CANDIDATE_SPEAKER_BUDGET")
    catalog = registry() if datasets is None else datasets
    item = catalog.get(index.get("dataset_id")) if type(index) is dict else None
    require(type(item) is dict and
            item.get("reference_status") == "CANDIDATE_REVIEW_REQUIRED",
            "DATASET_NOT_APPROVED_FOR_REFERENCE_SCREEN")
    rows = check_index(index, item)
    text_source = reference_text.document(transcript_source, transcript_sha256,
                                           transcript_format)
    text_checks = reference_text.content_checks(reference_content_checks)
    terms_path = Path(terms_file)
    require(not any(p.is_symlink() for p in (terms_path, *terms_path.parents)),
            "TERMS_FILE_ALIAS")
    require(sha_file(terms_path, MAX_SOURCE_TERMS_BYTES) ==
            index["license_snapshot_sha256"], "TERMS_SNAPSHOT_DRIFT")
    chosen = {}
    original_owner = {}
    for row in rows:
        raw = confined_file(original_root, row["original_path"], MAX_ORIGINAL_BYTES)
        measured_raw = sha_file(raw, MAX_ORIGINAL_BYTES)
        require(measured_raw == row["original_sha256"],
                "ORIGINAL_DATASET_AUDIO_CHANGED")
        measured = pool.measure_reference(normalized_root, row["normalized_path"])
        require(measured["wav_sha256"] == row["normalized_sha256"],
                "NORMALIZED_REFERENCE_WAV_CHANGED")
        check_conversion(row, evidence_root)
        speaker = row["speaker_key"]
        old_owner = original_owner.setdefault(row["original_sha256"], speaker)
        require(old_owner == speaker, "CROSS_SPEAKER_SAME_SOURCE_AUDIO")
        rank = (abs(measured["frames"] / measured["sample_rate_hz"] - 5.0),
                measured["wav_sha256"])
        if speaker not in chosen or rank < chosen[speaker][0]:
            chosen[speaker] = rank, row, measured
    winners = sorted(chosen.items(), key=lambda pair: pair[0])[:max_speakers]
    seen_pcm, references = set(), []
    for speaker, (_rank, row, measured) in winners:
        require(measured["pcm_sha256"] not in seen_pcm,
                "CROSS_SPEAKER_DUPLICATE_NORMALIZED_PCM")
        seen_pcm.add(measured["pcm_sha256"])
        text, text_status, provenance = reference_text.attach(
            index, row, measured, text_source, text_checks)
        references.append({
            "speaker_key": speaker,
            "audio_path": row["normalized_path"],
            "wav_sha256": measured["wav_sha256"],
            "source_original_sha256": row["original_sha256"],
            "reference_text": text, "reference_text_status": text_status,
            "reference_text_provenance": provenance,
        })
    require(bool(references), "NO_REFERENCE_CANDIDATES")
    return {
        "schema": CATALOG_SCHEMA,
        "rows": references,
    }


def create_only(path, body):
    destination = Path(path).absolute()
    require(destination != ROOT and ROOT not in destination.parents and
            destination.parent.is_dir() and
            not destination.exists() and not destination.is_symlink() and
            not any(parent.is_symlink() for parent in destination.parents),
            "PRIVATE_NEW_OUTPUT_REQUIRED")
    fd = os.open(destination, os.O_CREAT | os.O_WRONLY | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(body)
        out.flush()
        os.fsync(out.fileno())


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--index", required=True, type=Path)
    cli.add_argument("--original-root", required=True, type=Path)
    cli.add_argument("--normalized-root", required=True, type=Path)
    cli.add_argument("--conversion-evidence-root", required=True, type=Path)
    cli.add_argument("--license-snapshot", required=True, type=Path)
    cli.add_argument("--max-speakers", type=int, default=8)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--transcript-source", type=Path)
    cli.add_argument("--transcript-sha256")
    cli.add_argument("--transcript-format", choices=reference_text.FORMATS)
    cli.add_argument("--reference-content-checks", type=Path)
    args = cli.parse_args(argv)
    index = pool.read_json(args.index)
    prepared = candidates(index, args.original_root, args.normalized_root,
                          args.conversion_evidence_root, args.license_snapshot,
                          args.max_speakers, transcript_source=args.transcript_source,
                          transcript_sha256=args.transcript_sha256,
                          transcript_format=args.transcript_format,
                          reference_content_checks=args.reference_content_checks)
    if args.execute:
        create_only(args.output, pool.canonical(prepared) + b"\n")
    print(json.dumps({
        "status": "PRIVATE_CANDIDATE_CATALOG_ONLY" if args.execute else "READ_ONLY_CANDIDATE_PREVIEW",
        "dataset_id": index["dataset_id"], "reference_candidates": len(prepared["rows"]),
        "source_and_wav_bytes_checked": True,
        "conversion_execution_authenticated": False,
        "rights_externally_authenticated": False,
        "voice_cloning_authorized": False,
        "training_admitted": False, "qualification_allowed": False, "model_calls": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("OPEN_SOURCE_REFERENCE_HOLD: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
