"""Offline batch importer for speaker-separated, explicitly approved voice references.

Source records, selected WAVs and rights evidence remain in PRIVATE storage.
No networking, dataset scraping, identity lookup, cloning or recording occurs.
An externally supplied rights declaration is NOT cryptographic proof of consent.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import bulk_pool as p

CATALOG_SCHEMA = "kws-local-open-audio-catalog-v1"
PERMIT_SCHEMA = "kws-external-voice-clone-permit-v1"
MAX_CATALOG_ENTRIES = 10000


def _id(value):
    return p._id(value)


def _validate_permit(permit, rights_evidence_sha):
    p.require(type(permit) is dict and set(permit) == {
        "schema", "dataset_id", "source_type", "source_license",
        "source_terms_sha256", "rights_review_evidence_sha256",
        "voice_cloning_explicitly_allowed",
        "commercial_synthetic_kws_allowed",
        "forbidden_uses", "review_status"
    } and permit["schema"] == PERMIT_SCHEMA, "PERMIT_SCHEMA")
    _id(permit["dataset_id"])
    source = permit["source_type"]
    p.require(source in p._SOURCE_TYPES, "SOURCE_TYPE")
    p.require(permit["source_license"] in (
        "CC0-1.0", "CC-BY-4.0", "Apache-2.0", "custom-explicit-cloning-license"),
        "SOURCE_LICENSE")
    p._sha(permit["source_terms_sha256"])
    p.require(permit["rights_review_evidence_sha256"] == rights_evidence_sha and
              permit["voice_cloning_explicitly_allowed"] is True and
              permit["commercial_synthetic_kws_allowed"] is True and
              permit["forbidden_uses"] == [] and
              permit["review_status"] == "independently-reviewed",
              "NO_EXPLICIT_DATASET_VOICE_CLONE_SCOPE")
    # A CC0 field alone never qualifies. E.g. individual datasets can state
    # that voice cloning is forbidden or that speaker identification is banned.
    return permit


def _catalog(catalog):
    p.require(type(catalog) is dict and set(catalog) == {"schema", "rows"} and
              catalog["schema"] == CATALOG_SCHEMA, "CATALOG_SCHEMA")
    rows = catalog["rows"]
    p.require(type(rows) is list and
              1 <= len(rows) <= MAX_CATALOG_ENTRIES, "CATALOG_COUNT")
    keys = {"speaker_key", "audio_path", "wav_sha256",
            "source_original_sha256", "reference_text",
            "reference_text_status"}
    seen = set()
    for row in rows:
        p.require(type(row) is dict and set(row) == keys and
                  type(row["speaker_key"]) is str and
                  1 <= len(row["speaker_key"]) <= 256,
                  "CATALOG_ROW")
        p._sha(row["wav_sha256"])
        p._sha(row["source_original_sha256"])
        p.require((row["reference_text_status"] == "unknown" and
                   row["reference_text"] is None) or
                  (row["reference_text_status"] == "verified" and
                   type(row["reference_text"]) is str and
                   1 <= len(row["reference_text"]) <= 180),
                  "REFERENCE_TRANSCRIPT_STATUS")
        p.require(row["wav_sha256"] not in seen, "DUPLICATE_REFERENCE_CLIP")
        seen.add(row["wav_sha256"])
    return rows


def prepare(catalog, permit, rights_raw, audio_root, pseudonym_key,
            *, max_refs=100):
    p.require(type(pseudonym_key) is bytes and
              32 <= len(pseudonym_key) <= 128, "SECRET_PSEUDONYM_KEY")
    p.require(type(max_refs) is int and 1 <= max_refs <= p.MAX_REFERENCES,
              "MAX_REFERENCES")
    license_sha = hashlib.sha256(rights_raw).hexdigest()
    _validate_permit(permit, license_sha)
    rows = _catalog(catalog)
    chosen = {}
    for row in rows:
        measured = p.measure_reference(audio_root, row["audio_path"])
        p.require(measured["wav_sha256"] == row["wav_sha256"],
                  "SOURCE_WAV_CHANGED")
        token = hmac.new(pseudonym_key,
            (permit["dataset_id"] + "\0" + row["speaker_key"]).encode("utf-8"),
            hashlib.sha256).hexdigest()
        # Select the best 5-second original recording from the same exact
        # opaque source speaker; never create a new speaker by changing seed.
        distance = abs(measured["frames"] / measured["sample_rate_hz"] - 5.0)
        candidate = (distance, row["wav_sha256"], row, measured)
        if token not in chosen or (distance, row["wav_sha256"]) < chosen[token][:2]:
            chosen[token] = candidate
    selected = sorted(chosen.items(), key=lambda pair:pair[0])[:max_refs]
    p.require(selected, "NO_VALID_REFERENCES")
    records = []
    rights = []
    for token, (_distance, _digest, row, measured) in selected:
        speaker = "spk-" + token[:24]
        refid = "ref-" + hashlib.sha256(
            (token + measured["wav_sha256"]).encode()).hexdigest()[:24]
        # Review claims stay local; the tool never checks who signed an external
        # rights document and thus cannot certify permissions.
        rights_rel = "bulk-rights-" + permit["dataset_id"] + "/" + refid + ".json"
        review = {
            "schema":"kws-clone-source-rights-review-v1",
            "reference_id":refid,
            "wav_sha256":measured["wav_sha256"],
            "source_terms_sha256":permit["source_terms_sha256"],
            "usage":"synthetic-kws-voice-cloning",
            "basis":p._ALLOWED_BASIS[permit["source_type"]],
            "prohibited_uses":[],
            "status":"source-terms-reviewed",
            "rights_record_origin":"independently-reviewed-declaration"
        }
        rights.append((rights_rel, p.canonical(review)))
        records.append({
            "reference_id":refid, "speaker_group_id":speaker,
            "source_family_id":"fam-" + token[:24],
            "dataset_id":permit["dataset_id"],
            "source_type":permit["source_type"],
            # Stable speaker-group partition; no synthetic evaluation qualifies
            # as an independent real-human/final-frontend sealed test.
            "role":("development-only" if int(token[:4],16)%10 < 2
                    else "train-candidate"),
            "audio_path":row["audio_path"],
            "wav_sha256":measured["wav_sha256"],
            "pcm_sha256":measured["pcm_sha256"],
            "sample_rate_hz":measured["sample_rate_hz"],
            "frames":measured["frames"],
            "reference_text":row["reference_text"],
            "reference_text_status":row["reference_text_status"],
            "source_license":permit["source_license"],
            "source_terms_sha256":permit["source_terms_sha256"],
            "rights_record_path":rights_rel,
            "rights_record_sha256":hashlib.sha256(p.canonical(review)).hexdigest(),
            "parent_source_sha256":row["source_original_sha256"]
        })
    return {"schema":p.SCHEMA,"references":records,
            "training_admitted":False,"qualification_allowed":False},rights


def publish_private(manifest, rights, audio_root, out):
    root = Path(audio_root).absolute()
    output = Path(out).absolute()
    p.require(root.is_dir() and output.parent.is_dir() and
              not output.exists() and not output.is_symlink(),
              "PRIVATE_DESTINATION_MUST_BE_NEW")
    rights_dir = root / Path(rights[0][0]).parent
    p.require(not rights_dir.exists(), "PREEXISTING_RIGHTS_DIRECTORY")
    with tempfile.TemporaryDirectory(prefix=".bulk-rights-", dir=root) as tmp:
        temp = Path(tmp)
        for rel, body in rights:
            path = temp/Path(rel).name
            p.require(path.parent == temp and not path.exists(),
                      "RIGHTS_NAME_COLLISION")
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        temp.rename(rights_dir)
    # Re-open the original audio and the newly created right records, including
    # their content-addressed permission scope, before publishing the manifest.
    p.validate_sources(manifest, root)
    body = p.canonical(manifest) + b"\n"
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",required=True,type=Path)
    parser.add_argument("--permit",required=True,type=Path)
    parser.add_argument("--rights-evidence",required=True,type=Path)
    parser.add_argument("--audio-root",required=True,type=Path)
    parser.add_argument("--pseudonym-key-file",required=True,type=Path)
    parser.add_argument("--max-refs",type=int,default=100)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    try:
        catalog=p.read_json(args.catalog)
        permit=p.read_json(args.permit)
        rights_raw,_=p._file(args.rights_evidence.parent,
                             args.rights_evidence.name,65536)
        key,_=p._file(args.pseudonym_key_file.parent,
                      args.pseudonym_key_file.name,128)
        data,right_records=prepare(catalog,permit,rights_raw,args.audio_root,
                                   key,max_refs=args.max_refs)
        if args.execute:
            publish_private(data,right_records,args.audio_root,args.output)
        status="PUBLISHED_PRIVATE_NONADMITTED" if args.execute else "DRY_RUN_NO_WRITE"
    except (OSError,ValueError) as err:
        print("BULK_INGEST_NOT_ADMITTED: "+type(err).__name__,file=sys.stderr)
        return 2
    print(json.dumps({"status":status,"selected_references":len(data["references"]),
                      "model_calls":0,"training_admitted":False,
                      "rights_authentication_performed":False},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
