#!/usr/bin/env python3
"""Single KWS technical-research source importer (no parallel release lane).

Preserves strict original/normalized WAV and source-family evidence, opaque
speaker IDs, private create-only receipts, and reviewed research-use scope.
Does not create model training or deployment authority; never runs a model.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "auto-tts-voice-clone-20261102"))
import bulk_pool as pool  # noqa: E402
import open_reference_sources as open_source  # noqa: E402
from kws_data_contract import source_registry  # noqa: E402
from kws_data_contract import reference_text  # noqa: E402

SCHEMA = "kws-dataset-research-tts-scope-v2"
LIMIT = 32
ALLOWED_PURPOSE = "offline-kws-tts-research"
ALLOWED_SOURCE_LICENSE = frozenset(("Apache-2.0", "CC-BY-4.0", "CC0-1.0"))
CATALOG_SCHEMA = open_source.CATALOG_SCHEMA
MAX_CATALOG_ENTRIES = 10000


def require(ok, code):
    if not ok:
        raise ValueError(code)


def _catalog(catalog):
    pool.require(type(catalog) is dict and set(catalog) == {"schema", "rows"} and
              catalog["schema"] == CATALOG_SCHEMA, "CATALOG_SCHEMA")
    rows = catalog["rows"]
    pool.require(type(rows) is list and
              1 <= len(rows) <= MAX_CATALOG_ENTRIES, "CATALOG_COUNT")
    keys = {"speaker_key", "audio_path", "wav_sha256",
            "source_original_sha256", "reference_text",
            "reference_text_status", "reference_text_provenance"}
    seen = set()
    for row in rows:
        pool.require(type(row) is dict and set(row) == keys and
                  type(row["speaker_key"]) is str and
                  1 <= len(row["speaker_key"]) <= 256,
                  "CATALOG_ROW")
        pool._sha(row["wav_sha256"])
        pool._sha(row["source_original_sha256"])
        pool.require((row["reference_text_status"] == "unknown" and
                   row["reference_text"] is None) or
                  (row["reference_text_status"] in (reference_text.PROVIDED,
                                                     reference_text.CHECKED) and
                   type(row["reference_text"]) is str and
                   1 <= len(row["reference_text"]) <= 180),
                  "REFERENCE_TRANSCRIPT_STATUS")
        pool.require(row["wav_sha256"] not in seen, "DUPLICATE_REFERENCE_CLIP")
        seen.add(row["wav_sha256"])
    return rows



def publish_private(manifest, rights, audio_root, out):
    root = Path(audio_root).absolute()
    output = Path(out).absolute()
    pool.require(root.is_dir() and output.parent.is_dir() and
              not output.exists() and not output.is_symlink() and
              not any(p.is_symlink() for p in (root, *root.parents,
                                               *output.parents)),
              "PRIVATE_DESTINATION_MUST_BE_NEW")
    pool.require(rights, "MISSING_RIGHTS_RECORDS")
    relative_dir = Path(rights[0][0]).parent
    pool.require(len(relative_dir.parts) == 1 and not relative_dir.is_absolute(),
                 "RIGHTS_DIRECTORY")
    pool._id(relative_dir.name, "RIGHTS_DIRECTORY")
    rights_dir = root / relative_dir
    pool.require(not rights_dir.is_symlink(), "UNSAFE_FILE_ALIAS")
    rights_dir.mkdir(mode=0o700, exist_ok=True)
    lock = os.open(rights_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Serialize publishers while create-only links become single-link files.
        # An existing manifest only sees its already completed, unchanged receipts.
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix=".bulk-rights-", dir=root) as tmp:
            temp = Path(tmp)
            missing = []
            for rel, body in rights:
                path = temp / Path(rel).name
                pool.require(Path(rel).parent == relative_dir and not path.exists(),
                             "RIGHTS_NAME_COLLISION")
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
                pool._file(temp, path.name, 65536)
                dest = root / rel
                if dest.exists() or dest.is_symlink():
                    existing, _ = pool._file(root, rel, 65536)
                    pool.require(existing == body, "PREEXISTING_RIGHTS_CONFLICT")
                else:
                    missing.append((path, dest))
            # Stage complete bytes before publication; never replace a receipt.
            for path, dest in missing:
                os.link(path, dest)
                path.unlink()
            os.fsync(lock)
        # Re-open audio and all new/reused receipts before publishing a new manifest.
        pool.validate_sources(manifest, root)
        body = pool.canonical(manifest) + b"\n"
        fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(lock)



def validate_research_scope(scope, index, research_evidence, license_snapshot):
    """Validate explicit release-specific offline research scope."""
    require(type(scope) is dict and set(scope) == {
        "schema", "dataset_id", "dataset_release", "source_type",
        "source_license", "source_terms_sha256", "research_review_sha256",
        "intended_use", "research_tts_references_permitted",
        "public_voice_export_permitted",
        "forbidden_uses", "review_status"
    } and scope["schema"] == SCHEMA, "RESEARCH_SCOPE_SCHEMA")
    allowed = open_source.registry()
    dataset_id = scope["dataset_id"]
    require(type(dataset_id) is str and dataset_id in allowed and
            allowed[dataset_id]["reference_status"] == "CANDIDATE_REVIEW_REQUIRED",
            "DATASET_NOT_ELIGIBLE_FOR_RESEARCH_TTS_REFERENCE")
    require(type(index) is dict and scope["dataset_id"] == index.get("dataset_id")
            and scope["dataset_release"] == index.get("dataset_release"),
            "RESEARCH_DATASET_RELEASE_DRIFT")
    require(scope["source_type"] == "reviewed-open-audio" and
            scope["source_license"] in ALLOWED_SOURCE_LICENSE,
            "RESEARCH_SOURCE_LICENSE")
    pool._sha(scope["source_terms_sha256"])
    pool._sha(scope["research_review_sha256"])
    require(scope["source_terms_sha256"] == index.get("license_snapshot_sha256") and
            hashlib.sha256(license_snapshot).hexdigest() ==
            scope["source_terms_sha256"], "RESEARCH_SOURCE_TERMS_DRIFT")
    require(hashlib.sha256(research_evidence).hexdigest() ==
            scope["research_review_sha256"], "RESEARCH_REVIEW_EVIDENCE_DRIFT")
    require(scope["intended_use"] == ALLOWED_PURPOSE and
            scope["research_tts_references_permitted"] is True and
            scope["public_voice_export_permitted"] is False and
            scope["forbidden_uses"] == [] and
            scope["review_status"] == "independently-reviewed",
            "RESEARCH_TTS_SCOPE_NOT_REVIEWED")
    # This is a checked research declaration, not cryptographic proof of
    # a speaker's consent, a license grant, or model output authorization.
    return scope


def prepare(index, original_root, audio_root, conversion_root, terms_path,
            scope, research_evidence, speaker_registry, *, max_refs=8,
            transcript_source=None, transcript_sha256=None, transcript_format=None,
            reference_content_checks=None):
    require(type(max_refs) is int and 1 <= max_refs <= LIMIT,
            "RESEARCH_SPEAKER_BUDGET")
    registered_sources = source_registry.validate(speaker_registry)
    terms_path = Path(terms_path)
    require(not any(x.is_symlink() for x in (terms_path, *terms_path.parents)),
            "RESEARCH_TERMS_SYMLINK")
    terms_bytes, _ = pool._file(terms_path.parent, terms_path.name, 128 * 1024)
    validate_research_scope(scope, index, research_evidence, terms_bytes)
    # Reuse the *existing* source-original WAV/normalized PCM/conversion checks.
    candidates = open_source.candidates(index, original_root, audio_root,
                    conversion_root, terms_path, max_refs,
                    transcript_source=transcript_source, transcript_sha256=transcript_sha256,
                    transcript_format=transcript_format,
                    reference_content_checks=reference_content_checks)
    rows = _catalog(candidates)
    selected = {}
    for row in rows:
        measured = pool.measure_reference(audio_root, row["audio_path"])
        require(measured["wav_sha256"] == row["wav_sha256"],
                "RESEARCH_REFERENCE_SOURCE_DRIFT")
        identity = registered_sources.get((scope["dataset_id"], row["speaker_key"]))
        require(identity is not None, "SPEAKER_NOT_REGISTERED")
        family = identity["source_family_id"]
        rank = (abs(measured["frames"] / measured["sample_rate_hz"] - 5),
                row["wav_sha256"])
        if family not in selected or rank < selected[family][0]:
            selected[family] = (rank, row, measured, identity)
    require(1 <= len(selected) <= max_refs, "RESEARCH_NO_VALID_REFERENCES")
    result, rights = [], []
    for family, (_, row, measured, identity) in sorted(selected.items()):
        refid = "ref-" + hashlib.sha256(
            (family + measured["wav_sha256"]).encode()).hexdigest()[:24]
        review = {
            "schema": "kws-clone-source-rights-review-v1",
            "reference_id": refid,
            "wav_sha256": measured["wav_sha256"],
            "source_terms_sha256": scope["source_terms_sha256"],
            "usage": "synthetic-kws-voice-cloning",
            "basis": pool._ALLOWED_BASIS["reviewed-open-audio"],
            "prohibited_uses": [],
            "status": "source-terms-reviewed",
            "rights_record_origin": "independently-reviewed-declaration",
        }
        review_bytes = pool.canonical(review)
        review_sha = hashlib.sha256(review_bytes).hexdigest()
        path = "bulk-rights-" + scope["dataset_id"] + "/" + review_sha + ".json"
        # Do not count dataset sources as independent real qualification voices.
        rights.append((path, review_bytes))
        result.append({
            "reference_id": refid, "speaker_group_id": identity["speaker_group_id"],
            "source_family_id": family,
            "dataset_id": scope["dataset_id"],
            "source_type": "reviewed-open-audio",
            "role": identity["role"],
            "audio_path": row["audio_path"],
            "wav_sha256": measured["wav_sha256"],
            "pcm_sha256": measured["pcm_sha256"],
            "sample_rate_hz": measured["sample_rate_hz"],
            "frames": measured["frames"],
            "reference_text": row["reference_text"],
            "reference_text_status": row["reference_text_status"],
            "reference_text_provenance": row["reference_text_provenance"],
            "source_license": scope["source_license"],
            "source_terms_sha256": scope["source_terms_sha256"],
            "rights_record_path": path,
            "rights_record_sha256": review_sha,
            "parent_source_sha256": row["source_original_sha256"],
        })
    require(len({x["source_family_id"] for x in result}) == len(result),
            "RESEARCH_FAMILY_COLLISION")
    return {"schema": pool.SCHEMA, "references": result,
            "training_admitted": False, "qualification_allowed": False}, rights


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "original-root", "audio-root", "conversion-root",
                 "license-snapshot", "research-scope", "research-review-evidence",
                 "speaker-registry", "output"):
        cli.add_argument("--" + name, required=True, type=Path)
    cli.add_argument("--max-refs", type=int, default=8)
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--transcript-source", type=Path)
    cli.add_argument("--transcript-sha256")
    cli.add_argument("--transcript-format", choices=reference_text.FORMATS)
    cli.add_argument("--reference-content-checks", type=Path)
    args = cli.parse_args(argv)
    index = pool.read_json(args.index)
    scope = pool.read_json(args.research_scope)
    evidence, _ = pool._file(args.research_review_evidence.parent,
            args.research_review_evidence.name, 65536)
    registry = source_registry.read_json(args.speaker_registry)
    manifest, rights = prepare(index, args.original_root, args.audio_root,
            args.conversion_root, args.license_snapshot, scope, evidence,
            registry, max_refs=args.max_refs, transcript_source=args.transcript_source,
            transcript_sha256=args.transcript_sha256, transcript_format=args.transcript_format,
            reference_content_checks=args.reference_content_checks)
    if args.execute:
        dest = args.output.absolute()
        require(dest != ROOT and ROOT not in dest.parents and
                not any(x.is_symlink() for x in (dest, *dest.parents)),
                "RESEARCH_MANIFEST_PRIVATE_ONLY")
        publish_private(manifest, rights, args.audio_root, dest)
    print(json.dumps({
        "status": "RESEARCH_PRIVATE_SOURCE_ONLY" if args.execute
                  else "RESEARCH_SOURCE_PREVIEW_ONLY",
        "references": len(manifest["references"]),
        "scope": ALLOWED_PURPOSE,
        "training_admitted": False, "qualification_allowed": False, "actual_legal_authority_authenticated": False,
        "new_tts_calls": 0, "new_asr_calls": 0
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print("RESEARCH_SOURCE_INTAKE_HOLD: " + str(exc),
              file=sys.stderr)
        raise SystemExit(2)
