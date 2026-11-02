"""Large, source-separated KWS voice-clone corpus planning; no audio/model execution.

Open licensing and recorded speaker consent are independent of each other.
CC0 (or a public URL) alone is NEVER sufficient to claim clone permission.
The private operator-supplied rights record is verified for byte binding but
cannot itself authenticate the legal authority of its author.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import unicodedata
import wave

SCHEMA = "kws-bulk-clone-sources-v1"
PLAN_SCHEMA = "kws-bulk-clone-corpus-plan-v1"
MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MAX_REFERENCES = 512
MAX_CELLS = 16384
MAX_SOURCES_BYTES = 1024 * 1024
MAX_REF_BYTES = 2 * 1024 * 1024
_ID = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_ROLES = ("train-candidate", "development-only")
_SOURCE_TYPES = ("synthetic-original", "human-with-consent", "reviewed-open-audio")
_ALLOWED_BASIS = {
    "synthetic-original": "synthetic-provenance-reviewed",
    "human-with-consent": "speaker-explicit-consent",
    "reviewed-open-audio": "dataset-terms-and-voice-rights-reviewed",
}
POSITIVE = (
    ("你好小窝", 1), ("你好，小窝", 1),
    ("小窝小窝", 2), ("小窝，小窝", 2),
)
HARD_NEGATIVE = (
    "你好小屋", "你好小吴", "你好小我", "你好小王",
    "小屋小屋", "小沃小沃", "你好你好", "小窝",
)
GENERAL_NEGATIVE = (
    "今天晚上吃什么", "请把灯打开", "我想听一首音乐", "现在外面下雨了",
)
# Prompt classes are intended only; model outputs must be re-audited.
_PROMPTS = ([(text, "positive-intent", kw) for text, kw in POSITIVE] +
            [(text, "hard-negative-intent", None) for text in HARD_NEGATIVE] +
            [(text, "speech-negative-intent", None) for text in GENERAL_NEGATIVE])


def require(condition, why):
    if not condition:
        raise ValueError(why)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pairs(items):
    out = {}
    for key, val in items:
        require(key not in out, "DUPLICATE_JSON_FIELD")
        out[key] = val
    return out


def _number(raw):
    value = float(raw)
    require(math.isfinite(value), "NONFINITE_JSON")
    return value


def read_json(path, limit=MAX_SOURCES_BYTES):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            "JSON_PATH_ALIAS")
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1
            and 1 <= st.st_size <= limit, "JSON_SIZE_OR_LINK")
    with path.open("rb") as stream:
        body = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    require(len(body) == st.st_size and len(body) <= limit and
            (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink) ==
            (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns,
             after.st_nlink), "JSON_CHANGED")
    try:
        return json.loads(body.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")),
                          parse_float=_number)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError("INVALID_JSON") from err


def _id(value, error="INVALID_IDENTIFIER"):
    require(type(value) is str and bool(_ID.fullmatch(value)), error)
    return value


def _sha(value):
    require(type(value) is str and bool(_SHA.fullmatch(value)), "INVALID_SHA256")
    return value


def _file(root, relative, limit):
    require(type(relative) is str and 1 <= len(relative) <= 260 and
            "\\" not in relative and not relative.startswith("/") and
            str(PurePosixPath(relative)) == relative, "RELATIVE_FILE_PATH")
    parts = relative.split("/")
    require(all(part not in (".", "..", "") and
                re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,70}", part)
                for part in parts), "UNSAFE_FILE_PATH")
    root = Path(root).absolute()
    path = root / relative
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)) and
            not any(p.is_symlink() for p in (path, *path.parents)),
            "UNSAFE_FILE_ALIAS")
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and
            1 <= st.st_size <= limit, "FILE_SIZE_OR_LINK")
    sha = hashlib.sha256()
    chunks = []
    with path.open("rb") as stream:
        remaining = st.st_size
        while remaining:
            b = stream.read(min(1024 * 1024, remaining))
            require(bool(b), "TRUNCATED_SOURCE")
            sha.update(b)
            chunks.append(b)
            remaining -= len(b)
        after = os.fstat(stream.fileno())
    require((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
             after.st_ctime_ns, after.st_nlink), "FILE_CHANGED_DURING_READ")
    return b"".join(chunks), sha.hexdigest()


def measure_reference(root, relative):
    raw, wav_sha = _file(root, relative, MAX_REF_BYTES)
    require(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and
            int.from_bytes(raw[4:8], "little") + 8 == len(raw), "REFERENCE_RIFF")
    with wave.open(io.BytesIO(raw), "rb") as w:
        rate = w.getframerate()
        frames = w.getnframes()
        require((w.getnchannels(), w.getsampwidth(), w.getcomptype()) ==
                (1, 2, "NONE") and rate in (16000, 24000) and
                rate <= frames <= rate * 15, "REFERENCE_AUDIO_FORMAT_OR_DURATION")
        pcm = w.readframes(frames)
        require(not w.readframes(1) and len(pcm) == frames * 2 and
                len(raw) == len(pcm) + 44, "REFERENCE_PCM_EXTENT")
    return {"wav_sha256": wav_sha, "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "sample_rate_hz": rate, "frames": frames, "source_bytes": len(raw)}


def validate_sources(manifest, audio_root):
    require(type(manifest) is dict and set(manifest) ==
            {"schema", "references", "training_admitted", "qualification_allowed"}
            and manifest["schema"] == SCHEMA and
            manifest["training_admitted"] is False and
            manifest["qualification_allowed"] is False, "SOURCE_MANIFEST_SCHEMA")
    refs = manifest["references"]
    require(type(refs) is list and 1 <= len(refs) <= MAX_REFERENCES, "REFERENCE_COUNT")
    rows = []
    ids = set()
    used_wavs = set()
    family_roles = {}
    allowed = {
        "reference_id", "speaker_group_id", "source_family_id", "dataset_id",
        "source_type", "role", "audio_path", "wav_sha256", "pcm_sha256",
        "sample_rate_hz", "frames", "reference_text", "reference_text_status",
        "source_license", "source_terms_sha256", "rights_record_path",
        "rights_record_sha256", "parent_source_sha256"
    }
    for ref in refs:
        require(type(ref) is dict and set(ref) == allowed, "REFERENCE_FIELDS")
        reference_id = _id(ref["reference_id"], "REFERENCE_ID")
        family = _id(ref["source_family_id"], "SOURCE_FAMILY")
        speaker = _id(ref["speaker_group_id"], "SPEAKER_GROUP")
        _id(ref["dataset_id"], "DATASET_ID")
        require(reference_id not in ids, "DUPLICATE_REFERENCE")
        ids.add(reference_id)
        role = ref["role"]
        require(type(role) is str and role in _ROLES, "SPLIT_ROLE")
        for group in ("family:" + family, "speaker:" + speaker):
            require(group not in family_roles or family_roles[group] == role,
                    "VOICE_FAMILY_SPLIT_LEAK")
            family_roles[group] = role
        kind = ref["source_type"]
        require(type(kind) is str and kind in _SOURCE_TYPES, "SOURCE_KIND")
        require(ref["source_license"] in ("Apache-2.0", "CC0-1.0", "CC-BY-4.0",
                                          "custom-explicit-cloning-license"),
                "UNREVIEWED_LICENSE")
        _sha(ref["source_terms_sha256"])
        if ref["parent_source_sha256"] is not None:
            _sha(ref["parent_source_sha256"])
        require(ref["reference_text_status"] in ("verified", "unknown"),
                "REFERENCE_TRANSCRIPT_STATUS")
        if ref["reference_text_status"] == "verified":
            require(type(ref["reference_text"]) is str and
                    1 <= len(ref["reference_text"]) <= 180 and
                    all(ord(ch) >= 32 for ch in ref["reference_text"]),
                    "REFERENCE_TRANSCRIPT")
        else:
            require(ref["reference_text"] is None, "UNVERIFIED_TRANSCRIPT_MUST_BE_NULL")
        values = measure_reference(audio_root, ref["audio_path"])
        for k in ("wav_sha256", "pcm_sha256", "sample_rate_hz", "frames"):
            require(type(ref[k]) is type(values[k]) and ref[k] == values[k],
                    "REFERENCE_AUDIO_IDENTITY_DRIFT")
        require(ref["wav_sha256"] not in used_wavs, "DUPLICATE_REFERENCE_AUDIO")
        used_wavs.add(ref["wav_sha256"])
        evidence, evidence_sha = _file(audio_root, ref["rights_record_path"], 65536)
        require(evidence_sha == _sha(ref["rights_record_sha256"]), "RIGHTS_FILE_DRIFT")
        try:
            rights = json.loads(evidence.decode("utf-8"), object_pairs_hook=_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise ValueError("RIGHTS_FILE_INVALID") from err
        require(type(rights) is dict and set(rights) == {
            "schema", "reference_id", "wav_sha256", "source_terms_sha256",
            "usage", "basis", "prohibited_uses", "status", "rights_record_origin"
        } and rights["schema"] == "kws-clone-source-rights-review-v1" and
                rights["reference_id"] == reference_id and
                rights["wav_sha256"] == values["wav_sha256"] and
                rights["source_terms_sha256"] == ref["source_terms_sha256"] and
                rights["usage"] == "synthetic-kws-voice-cloning" and
                rights["basis"] == _ALLOWED_BASIS[kind] and
                rights["status"] == "source-terms-reviewed" and
                rights["rights_record_origin"] == "independently-reviewed-declaration" and
                type(rights["prohibited_uses"]) is list and
                len(rights["prohibited_uses"]) == 0, "SOURCE_CLONE_RIGHTS_NOT_REVIEWED")
        # A JSON declaration can be falsified: still requires actual external
        # authorisation/record review. CC0 by itself NEVER bypasses this check.
        rows.append(ref)
    return rows



def _cells(refs, repetitions, batch_id):
    cells = []
    for ref in refs:
        for repetition in range(repetitions):
            for index, (text, kind, keyword_id) in enumerate(_PROMPTS):
                identity = hashlib.sha256(canonical({
                    "batch_id": batch_id, "ref": ref["reference_id"], "text": text,
                    "prompt_index": index, "repetition": repetition
                })).hexdigest()
                cells.append({
                    "cell_id": "bulk-" + identity[:24],
                    "reference_id": ref["reference_id"],
                    "speaker_group_id": ref["speaker_group_id"],
                    "source_family_id": ref["source_family_id"],
                    "role": ref["role"],
                    "prompt": text, "intended_kind": kind,
                    "intended_keyword_id": keyword_id,
                    "seed": 1000000 + int(identity[:8], 16) % 1000000000,
                    "repetition": repetition, "status": "NOT_RUN",
                    "label_authority": "intended_only",
                    "synthetic": True,
                    "training_admitted": False, "qualification_allowed": False
                })
    cells.sort(key=lambda x:(x["reference_id"],x["cell_id"]))
    return cells


def plan(manifest, audio_root, *, repetitions=2, batch_id="pilot01",
         revision=None, max_calls=None):
    refs = validate_sources(manifest, audio_root)
    _id(batch_id, "BATCH_ID")
    require(type(repetitions) is int and 1 <= repetitions <= 8, "REPETITIONS")
    require(type(revision) is str and bool(_HEX40.fullmatch(revision)), "PINNED_BASE_REVISION")
    cells = _cells(refs, repetitions, batch_id)
    require(len(cells) <= MAX_CELLS, "PLANNED_SAMPLES_EXCEED_LIMIT")
    cells.sort(key=lambda x:(x["reference_id"], x["cell_id"]))
    ceiling = len(cells) if max_calls is None else max_calls
    require(type(ceiling) is int and 1 <= ceiling <= len(cells) and ceiling <= MAX_CELLS,
            "MAX_CALLS")
    return {
        "schema": PLAN_SCHEMA, "batch_id": batch_id, "model_id": MODEL,
        "model_revision": revision,
        "source_manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
        "references": refs, "cells": cells,
        "budgets": {"max_total_calls": ceiling, "max_calls_per_invocation": 64,
                    "max_retries": 0, "max_paid_usd": 0},
        "repetitions": repetitions, "model_calls": 0,
        "source_rights_authenticated": False,
        "actual_spoken_labels_verified": False,
        "training_admitted": False, "qualification_allowed": False,
        "shipping_approved": False
    }


def verify_plan(value):
    require(type(value) is dict and set(value) == {
        "schema","batch_id","model_id","model_revision",
        "source_manifest_sha256","references","cells","budgets",
        "repetitions","model_calls","source_rights_authenticated",
        "actual_spoken_labels_verified","training_admitted",
        "qualification_allowed","shipping_approved"
    } and value.get("schema") == PLAN_SCHEMA, "PLAN_SCHEMA")
    # No source WAV at this stage; prove immutable plan identity by checking
    # the original private source during run with a separate rehash.
    require(value.get("training_admitted") is False and
            value.get("qualification_allowed") is False and
            value.get("source_rights_authenticated") is False and
            value.get("actual_spoken_labels_verified") is False and
            value.get("shipping_approved") is False, "UNTRUSTED_PLAN_AUTHORITY")
    refs = value.get("references")
    cells = value.get("cells")
    require(type(refs) is list and 1 <= len(refs) <= MAX_REFERENCES and
            type(cells) is list and 1 <= len(cells) <= MAX_CELLS, "PLAN_COUNTS")
    require(type(value.get("model_revision")) is str and
            bool(_HEX40.fullmatch(value["model_revision"])) and
            value.get("model_id") == MODEL, "MODEL_PIN")
    budgets = value.get("budgets")
    require(type(budgets) is dict and
            budgets.get("max_retries") == 0 and
            budgets.get("max_paid_usd") == 0 and
            type(budgets.get("max_calls_per_invocation")) is int and
            1 <= budgets["max_calls_per_invocation"] <= 64 and
            type(budgets.get("max_total_calls")) is int and
            1 <= budgets["max_total_calls"] <= len(cells), "PLAN_BUDGET")
    require(type(value.get("repetitions")) is int and
            1<=value["repetitions"]<=8 and
            type(value.get("batch_id")) is str and
            bool(_ID.fullmatch(value["batch_id"])) and
            type(value.get("source_manifest_sha256")) is str and
            bool(_SHA.fullmatch(value["source_manifest_sha256"])) and
            value.get("model_calls") == 0, "PLAN_METADATA")
    require(canonical(cells)==canonical(_cells(refs,value["repetitions"],
            value["batch_id"])),"CELL_PLAN_CHANGED")
    require(len({row["cell_id"] for row in cells}) == len(cells) and
            len({row["reference_id"] for row in refs}) == len(refs), "PLAN_DUPLICATE")
    require(all(type(x) is dict and x.get("training_admitted") is False
                and x.get("label_authority") == "intended_only"
                and x.get("qualification_allowed") is False for x in cells),
            "CELL_AUTHORITY")
    return value


def main(argv=None):
    import argparse
    import sys
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sources",type=Path,required=True)
    cli.add_argument("--audio-root",type=Path,required=True)
    cli.add_argument("--model-revision",required=True)
    cli.add_argument("--batch-id",default="pilot01")
    cli.add_argument("--repetitions",type=int,default=2)
    cli.add_argument("--max-total-calls",type=int)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args(argv)
    try:
        m=read_json(args.sources)
        result=plan(m,args.audio_root,repetitions=args.repetitions,
                    batch_id=args.batch_id,revision=args.model_revision,
                    max_calls=args.max_total_calls)
        require(args.output.parent.is_dir() and not args.output.exists()
                and not args.output.is_symlink(),"PLAN_OUTPUT_MUST_BE_NEW")
        body=canonical(result)+b"\n"
        fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(body)
            stream.flush();os.fsync(stream.fileno())
    except (ValueError,OSError) as err:
        print("BULK_SPEAKER_PLAN_NOT_ADMITTED: "+type(err).__name__,file=sys.stderr)
        return 2
    print(json.dumps({"references":len(result["references"]),
        "planned_clips":len(result["cells"]),
        "max_total_model_calls":result["budgets"]["max_total_calls"],
        "training_admitted":False},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
