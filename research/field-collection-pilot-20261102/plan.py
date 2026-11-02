"""Offline product-KWS field recording pilot: deterministic prompts and byte-only audit.

Does NOT capture audio, authenticate humans/rights, generate labels, run models,
publish recordings, admit training, or claim final AFE/physical SSC305 evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import tempfile

SCHEMA = "xiaowo-field-collection-plan-v1"
PARTICIPANTS_SCHEMA = "xiaowo-field-participants-v1"
RECORDINGS_SCHEMA = "xiaowo-field-recordings-v1"
IDENTITY = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
SPLITS = ("train-candidate", "development", "sealed-evaluation")
TARGETS = (("你好小窝", 1), ("小窝小窝", 2))
CONFUSABLES = ("你好小屋", "你好小吴", "你好小王", "小屋小屋", "小沃小沃", "你好你好")
MAX_WAV = 64 * 1024 * 1024
MAX_MANIFEST = 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _pairs(items):
    out = {}
    for key, value in items:
        require(key not in out, "DUPLICATE_JSON_KEY")
        out[key] = value
    return out


def _number(value):
    v = float(value)
    require(math.isfinite(v), "NONFINITE_JSON")
    return v


def _json_file(path):
    source=Path(path).absolute()
    require(hasattr(os,"O_NOFOLLOW") and
            not any(p.is_symlink() for p in (source,*source.parents)),
            "MANIFEST_ALIAS")
    handle=None
    try:
        handle=os.open(source,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        before=os.fstat(handle)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and
                0<before.st_size<=MAX_MANIFEST,"MANIFEST_FILE_BOUNDS")
        with os.fdopen(handle,"rb") as stream:
            handle=None
            raw=stream.read(MAX_MANIFEST+1)
            after=os.fstat(stream.fileno())
        require(len(raw)==before.st_size and len(raw)<=MAX_MANIFEST,
                "MANIFEST_CHANGED_OR_OVERSIZED")
        for key in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_nlink"):
            require(getattr(before,key)==getattr(after,key),"MANIFEST_MUTATED_DURING_READ")
    except OSError as exc:
        raise ValueError("MANIFEST_UNSAFE_FILE") from exc
    finally:
        if handle is not None:os.close(handle)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")),
                          parse_float=_number)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("INVALID_JSON") from exc


def _id(value):
    require(type(value) is str and IDENTITY.fullmatch(value), "INVALID_OPAQUE_ID")
    return value


def _sha(value):
    require(type(value) is str and HASH.fullmatch(value), "INVALID_SHA256")
    return value


def participants(value):
    require(type(value) is dict and set(value) == {"schema", "participants"} and
            value["schema"] == PARTICIPANTS_SCHEMA, "PARTICIPANT_CONTRACT")
    rows = value["participants"]
    require(type(rows) is list and 1 <= len(rows) <= 100, "PARTICIPANT_COUNT")
    seen_sessions, seen_speakers, family_split = set(), {}, {}
    result = []
    keys = {"speaker_id", "session_id", "source_family_id", "role", "room_id", "device_id", "afe_profile_sha256"}
    for row in rows:
        require(type(row) is dict and set(row) == keys, "PARTICIPANT_FIELDS")
        for key in ("speaker_id", "session_id", "source_family_id", "room_id", "device_id"):
            _id(row[key])
        if row["afe_profile_sha256"] is not None:
            _sha(row["afe_profile_sha256"])
        role = row["role"]
        require(type(role) is str and role in SPLITS, "UNKNOWN_DATA_ROLE")
        require(row["session_id"] not in seen_sessions, "DUPLICATE_SESSION")
        seen_sessions.add(row["session_id"])
        for k, label, seen in ((row["speaker_id"], "SPEAKER_SPLIT_LEAK", seen_speakers),
                                (row["source_family_id"], "FAMILY_SPLIT_LEAK", family_split)):
            require(k not in seen or seen[k] == role, label)
            seen[k] = role
        result.append(row)
    return sorted(result, key=lambda r:(r["speaker_id"],r["session_id"]))


def _slot(row, prompt, kind, keyword_id, distance, azimuth, repetition):
    data = {
        "speaker_id": row["speaker_id"], "session_id": row["session_id"],
        "source_family_id": row["source_family_id"],
        "role": row["role"], "room_id": row["room_id"], "device_id": row["device_id"],
        "afe_profile_sha256": row["afe_profile_sha256"],
        "distance_m": distance, "azimuth_deg": azimuth, "repetition": repetition,
        "prompt": prompt, "intent_kind": kind, "intended_keyword_id": keyword_id,
        "audio_recorded": False, "human_verbatim_reviewed": False,
        "rights_approved": False, "training_admitted": False,
    }
    seed = json.dumps({key:data[key] for key in ("speaker_id","session_id","distance_m",
        "azimuth_deg","repetition","prompt")}, ensure_ascii=False,sort_keys=True,separators=(",",":"))
    data["slot_id"] = "s-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return data


def build_plan(participant_data):
    rows = participants(participant_data)
    slots = []
    for row in rows:
        for distance in (1,3,5):
            for azimuth in (0,180):
                for phrase, kw_id in TARGETS:
                    for repetition in (1,2):
                        slots.append(_slot(row, phrase, "wake-intent", kw_id, distance, azimuth, repetition))
        for distance in (3,5):
            for phrase in CONFUSABLES:
                slots.append(_slot(row, phrase, "confusable-intent", None, distance, 0, 1))
        # Natural speech/background must be audited for hidden wake events before
        # being used as a negative. Ten-minute instructions are NOT audio evidence.
        slots.append(_slot(row, "录制真实家庭自然交谈或环境声，保留所有意外唤醒词",
                           "continuous-negative-candidate", None, 3, 0, 1))
    require(len({r["slot_id"] for r in slots}) == len(slots), "SLOT_COLLISION")
    slots.sort(key=lambda r: hashlib.sha256(r["slot_id"].encode()).hexdigest())
    return {
        "schema": SCHEMA,
        "source_type": "human-field-recording-candidate",
        "scope": "private-local-pilot-only",
        "roles_are_declared_not_authenticated": True,
        "participants": len(rows),
        "slots": slots,
        "human_rights_authenticated": False,
        "audio_labels_verified": False,
        "training_admitted": False,
        "qualification_allowed": False,
        "shipping_approved": False,
        "model_calls": 0,
    }


def _relative(path):
    require(type(path) is str and len(path) <= 512 and "\\" not in path, "AUDIO_PATH")
    parts = path.split("/")
    require(1 <= len(parts) <= 8 and all(0 < len(p) <= 64 and
            p not in (".","..") and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*",p) for p in parts)
            and path.endswith(".wav") and str(PurePosixPath(path)) == path, "AUDIO_PATH")
    return parts


def _read_wav(root, relative, expected_size=MAX_WAV):
    parts = _relative(relative)
    flags = os.O_RDONLY | os.O_DIRECTORY
    require(hasattr(os,"O_NOFOLLOW"), "LINUX_NOFOLLOW_REQUIRED")
    directory = os.open(str(root), flags | os.O_NOFOLLOW)
    handle = None
    try:
        for part in parts[:-1]:
            next_dir = os.open(part, flags | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory);directory = next_dir
        handle = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(handle)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                46 <= before.st_size <= expected_size, "UNSAFE_WAV_FILE")
        with os.fdopen(handle,"rb") as stream:
            handle = None
            header = stream.read(44)
            require(len(header) == 44 and header[:4] == b"RIFF" and header[8:16] == b"WAVEfmt ",
                    "CANONICAL_RIFF_REQUIRED")
            riff_size = struct.unpack_from("<I",header,4)[0]
            fmt_size,format_tag,channels,rate,byte_rate,alignment,bits = struct.unpack_from("<IHHIIHH",header,16)
            require(fmt_size == 16 and format_tag == 1 and channels == 1 and
                    rate == 16000 and byte_rate == 32000 and alignment == 2 and bits == 16 and
                    header[36:40] == b"data", "PCM16_MONO_16KHZ_REQUIRED")
            pcm_bytes = struct.unpack_from("<I",header,40)[0]
            require(pcm_bytes > 0 and pcm_bytes % 2 == 0 and
                    pcm_bytes + 44 == before.st_size and riff_size + 8 == before.st_size,
                    "WAV_EXTENT_MISMATCH")
            original = hashlib.sha256();original.update(header)
            pcm = hashlib.sha256()
            remaining = pcm_bytes
            while remaining:
                chunk = stream.read(min(1024*1024,remaining))
                require(bool(chunk), "WAV_TRUNCATED")
                original.update(chunk);pcm.update(chunk);remaining-=len(chunk)
            require(not stream.read(1),"WAV_TRAILING_BYTES")
            after = os.fstat(stream.fileno())
            for f in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_nlink"):
                require(getattr(before,f) == getattr(after,f), "WAV_MUTATED_DURING_READ")
        return {"wav_sha256":original.hexdigest(),"pcm_sha256":pcm.hexdigest(),
                "frames":pcm_bytes//2,"bytes":before.st_size}
    except OSError as exc:
        raise ValueError("UNSAFE_WAV_FILE") from exc
    finally:
        if handle is not None:os.close(handle)
        os.close(directory)


def canonical_plan_bytes(plan):
    return json.dumps(plan,ensure_ascii=False,sort_keys=True,
                      separators=(",",":"),allow_nan=False).encode("utf-8")


def plan_sha256(plan):
    return hashlib.sha256(canonical_plan_bytes(plan)).hexdigest()


def audit_recordings(plan, recordings, audio_root, *, participant_data,
                     expected_plan_sha256):
    # Freeze externally at planning time; never infer participants from edited slots.
    expected=build_plan(participant_data)
    _sha(expected_plan_sha256)
    require(canonical_plan_bytes(plan)==canonical_plan_bytes(expected),
            "PLAN_PARTICIPANT_BINDING_DRIFT")
    require(plan_sha256(plan)==expected_plan_sha256,"FROZEN_PLAN_IDENTITY_DRIFT")
    require(type(plan) is dict and plan.get("schema") == SCHEMA and
            plan.get("training_admitted") is False and plan.get("qualification_allowed") is False,
            "IMMUTABLE_NONADMITTING_PLAN_REQUIRED")
    slots = plan.get("slots")
    require(type(slots) is list and len(slots) <= 10000, "PLAN_SLOTS")
    mapping = {row["slot_id"]:row for row in slots}
    require(len(mapping) == len(slots), "DUPLICATE_PLAN_SLOT")
    require(type(recordings) is dict and set(recordings) == {"schema","recordings"} and
            recordings["schema"] == RECORDINGS_SCHEMA and
            type(recordings["recordings"]) is list and len(recordings["recordings"]) <= len(slots),
            "RECORDING_ROSTER")
    root = Path(audio_root).absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root,*root.parents)), "AUDIO_ROOT_ALIAS")
    seen_ids, paths, source_wavs, source_pcms = set(),set(),set(),set()
    counts = {role:{"wake-intent":0,"confusable-intent":0,"continuous-negative-candidate":0}
              for role in SPLITS}
    for row in recordings["recordings"]:
        keys={"slot_id","path","wav_sha256","pcm_sha256","frames","human_verbatim_reviewed",
              "rights_approved","training_admitted"}
        require(type(row) is dict and set(row) == keys, "RECORDING_FIELDS")
        slot_id = row["slot_id"]
        require(type(slot_id) is str and slot_id in mapping and slot_id not in seen_ids,
                "UNKNOWN_OR_DUPLICATE_SLOT")
        seen_ids.add(slot_id)
        require(row["human_verbatim_reviewed"] is False and row["rights_approved"] is False and
                row["training_admitted"] is False,"FORGED_REVIEW_OR_RIGHTS")
        path = row["path"]
        require(path not in paths, "DUPLICATE_SOURCE_PATH")
        paths.add(path)
        _sha(row["wav_sha256"]);_sha(row["pcm_sha256"])
        require(type(row["frames"]) is int and row["frames"] > 0, "FRAME_TYPE")
        actual = _read_wav(root,path)
        require(actual["wav_sha256"] == row["wav_sha256"] and
                actual["pcm_sha256"] == row["pcm_sha256"] and
                actual["frames"] == row["frames"], "WAV_IDENTITY_DRIFT")
        require(actual["wav_sha256"] not in source_wavs and
                actual["pcm_sha256"] not in source_pcms, "DUPLICATE_AUDIO_PAYLOAD")
        source_wavs.add(actual["wav_sha256"]);source_pcms.add(actual["pcm_sha256"])
        planrow = mapping[slot_id]
        counts[planrow["role"]][planrow["intent_kind"]] += 1
    return {"schema":"xiaowo-field-recording-structure-audit-v1",
        "source_type":"human-field-recording-candidate",
        "planned_slots":len(slots),
        "plan_sha256":expected_plan_sha256,
        "verified_recording_bytes":len(seen_ids),
        "all_slots_present":len(seen_ids)==len(slots),
        "count_by_role_and_declared_intent":counts,
        "human_identity_authenticated":False,"rights_approved":False,
        "actual_spoken_text_verified":False,"audio_labels_verified":False,
        "source_root_admitted":False,"training_admitted":False,
        "qualification_allowed":False,"shipping_approved":False,
        "physical_ssc305_qualification":False,"model_calls":0,
        "status":"STRUCTURAL_CANDIDATES_ONLY_NOT_TRAINING"}


def _publish(path, data):
    target = Path(path)
    require(target.parent.is_dir() and not target.exists() and not target.is_symlink(),
            "OUTPUT_MUST_BE_NEW")
    body=json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(mode="wb",dir=target.parent,
                                         prefix=".xiaowo-",delete=False) as stream:
            tmp=Path(stream.name)
            stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.link(tmp,target,follow_symlinks=False)
    finally:
        if tmp is not None:tmp.unlink(missing_ok=True)


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cmd=cli.add_subparsers(dest="command",required=True)
    plan=cmd.add_parser("plan");plan.add_argument("--participants",required=True)
    plan.add_argument("--output",required=True)
    verify=cmd.add_parser("verify");verify.add_argument("--plan",required=True)
    verify.add_argument("--recordings",required=True);verify.add_argument("--audio-root",required=True)
    verify.add_argument("--output",required=True)
    verify.add_argument("--participants",required=True)
    verify.add_argument("--expected-plan-sha256",required=True)
    args=cli.parse_args(argv)
    try:
        if args.command=="plan":
            result=build_plan(_json_file(args.participants))
        else:
            result=audit_recordings(_json_file(args.plan),_json_file(args.recordings),args.audio_root,
                participant_data=_json_file(args.participants),
                expected_plan_sha256=args.expected_plan_sha256)
        _publish(args.output,result)
    except (ValueError,OSError) as error:
        print("FIELD_CORPUS_NOT_ADMITTED: "+type(error).__name__,file=sys.stderr)
        return 2
    print(json.dumps({"status":result.get("status","PLANNED_NOT_RECORDED"),
        "training_admitted":False,"model_calls":0,
        "plan_sha256":plan_sha256(result) if args.command=="plan" else result["plan_sha256"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
