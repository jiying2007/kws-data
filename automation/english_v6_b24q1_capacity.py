"""Finite byte envelopes for the newly approved B24Q1 evidence, no scientific grant.

All values are decimal bytes. Measured R1 STOP archives total308,497,677;
five final API bodies plus two logs total1,697,604. Retaining that full
310,195,281-byte proof, fixed TRAIN/LIVE50,759,903, unchanged maximum TTS
109,588,224 and metadata32,000,000 already yields502,543,408 expanded bytes
before natural originals. Existing conservative ZIP/header accounting yields
593,072,007 bytes, so the new PRIMARY SOURCE limit is768,000,000.

Replica SOURCE retains the complete authenticated new PRIMARY proof once;
its limit is1,280,000,000. Replica quality producer retains that proof plus
both code roots; its limit is1,024,000,000. Actual full primary proof and
whole24 natural headers must fit before any replica budget is consumed.
Nested Q1 PRIMARY SOURCE remains768,000,000; historical R1 SOURCE remains
512,000,000; every historical ordinary archive keeps128,000,000. No bytes
are discarded or synthesized to fit; an oversized package stops before calls.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import zipfile

CORE_LIMIT = 128_000_000
COMPOSITE_LIMIT = 1_024_000_000
SOURCE_LIMIT = 1_280_000_000
PRIMARY_SOURCE_LIMIT = 768_000_000
R1_SOURCE_LIMIT = 512_000_000
REPLICA_SOURCE_LIMIT = SOURCE_LIMIT
REPLICA_PRODUCER_LIMIT = COMPOSITE_LIMIT
MAX_MEMBERS = 8192
MAX_PATH_BYTES = 1024
NEW_METADATA_LIMIT = 32_000_000
CONSUMPTION_RESERVE = 32_768
SCHEMA = "english-v6-b24q1-prospective-composite-capacity-v1"
PRIMARY_FILES = {"api/" + name + ".json" for name in ("run", "commit", "artifacts")} | {
    "originals/" + name + ".zip" for name in ("result", "producer", "consumed")} | {
    "inputs/" + name + ".zip" for name in ("B0", "B9", "pair", "source", "screen",
        "native_qwen", "native_whisper", "native_parakeet", "native_xlsr", "native_lv60",
        "train_producer", "source_producer")}
TRAIN_FILES = {"TRAIN_" + name + ".zip" for name in ("consumed", "models", "producer", "receipts")}
KNOWN_FILES = TRAIN_FILES | {"AUTHORIZATION.json", "HISTORY.json", "TRAIN_LIVE.json",
                            "OPENING.json", "CORPUS_TRANSPORT.json"}
PRIMARY_KNOWN_FILES = TRAIN_FILES | {"AUTHORIZATION.json", "HISTORY.json", "TRAIN_LIVE.json", "CORPUS_TRANSPORT.json"}
R1_STOP_FILES = {name + ".zip" for name in ("source", "source_producer", "source_consumed", "transport",
    *("native_" + p for p in ("qwen", "whisper", "parakeet", "xlsr", "lv60")),
    *("consumed_" + p for p in ("qwen", "whisper", "parakeet", "xlsr", "lv60")))} | {
    "metadata/source_" + name + ".json" for name in ("run", "artifacts", "commit", "tree", "jobs")} | {
    "PARAKEET_JOB.log", "READBACK_JOB.log"}
TTS_COUNT = 96
NATURAL_COUNT = 24
TTS_DERIVED_SAMPLES = 240_000
# scipy's complete 320/441 resample has ceil(native_samples*320/441)
# output samples. Thus every valid original has at most 330750 native samples.
TTS_NATIVE_SAMPLES = TTS_DERIVED_SAMPLES * 441 // 320
TTS_AUDIO_BOUND = TTS_COUNT * (2 * TTS_NATIVE_SAMPLES + 2 * TTS_DERIVED_SAMPLES + 44)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def archive_limit(role):
    need(role in ("core", "primary_source", "replication_source", "replication_producer"), "explicit capacity role only")
    if role == "primary_source":
        return PRIMARY_SOURCE_LIMIT
    return CORE_LIMIT if role == "core" else (SOURCE_LIMIT if role == "replication_source" else COMPOSITE_LIMIT)


def safe_name(name, directory=False):
    need(type(name) is str and name and "\\" not in name and "\0" not in name,
         "safe complete archive member name")
    value = name[:-1] if directory and name.endswith("/") else name
    path = PurePosixPath(value)
    need(value and not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/"))
         and len(name.encode("utf-8")) <= MAX_PATH_BYTES, "bounded relative archive path")
    return value


def zip_bound(total, members, path_bytes):
    """Conservative DEFLATE + local/central ZIP header envelope, in bytes.

    Each file reserves n+ceil(n/8)+ceil(n/64)+18 DEFLATE bytes. Aggregating
    the two ceilings adds another 2*N bytes for rounding. Each member also
    reserves 30+46 header bytes, a 24-byte data descriptor, two <=128-byte
    extra fields, and its filename twice. The
    1024-byte ending covers ZIP64 and a <=512-byte archive comment. Actual
    archives are still checked for their exact compressed and expanded size.
    """
    need(all(type(v) is int and v >= 0 for v in (total, members, path_bytes)), "integer byte budget")
    return total + (total + 7) // 8 + (total + 63) // 64 + 376 * members + 2 * path_bytes + 1024


def check_archive(raw, role):
    """Return an open ZipFile after size, every member and every CRC pass."""
    maximum = archive_limit(role)
    need(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded complete " + role + " ZIP")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    try:
        entries = archive.infolist()
        need(len(entries) <= MAX_MEMBERS and len(entries) == len({i.filename for i in entries})
             and sum(i.file_size for i in entries) <= maximum, "unique bounded complete archive members")
        need(len(archive.comment) <= 512, "bounded archive comment")
        for item in entries:
            safe_name(item.filename, item.is_dir())
            local = raw[item.header_offset:item.header_offset + 30]
            need(len(local) == 30 and local[:4] == b"PK\x03\x04"
                 and int.from_bytes(local[26:28], "little") <= MAX_PATH_BYTES
                 and int.from_bytes(local[28:30], "little") <= 128,
                 "bounded original local ZIP header and extra fields")
            need(not stat.S_ISLNK(item.external_attr >> 16) and not (item.flag_bits & 1)
                 and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                 and len(item.extra) <= 128 and not item.comment,
                 "safe unencrypted stored/deflated original member")
        need(archive.testzip() is None, "every original member CRC")
        return archive
    except BaseException:
        archive.close()
        raise


def inventory(directory):
    """Hash all regular files; also count possible directory ZIP entries."""
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink(), "regular capacity root")
    files, directories = {}, []
    pending = [directory]
    while pending:
        current = pending.pop()
        for path in sorted(current.iterdir()):
            name = path.relative_to(directory).as_posix()
            safe_name(name)
            mode = path.lstat().st_mode
            need(not stat.S_ISLNK(mode), "no capacity-root symlinks")
            if stat.S_ISDIR(mode):
                directories.append(name + "/")
                pending.append(path)
            else:
                need(stat.S_ISREG(mode), "regular complete proof member only")
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                files[name] = dict(bytes=path.stat().st_size, sha256=digest)
            need(len(files) + len(directories) <= MAX_MEMBERS, "bounded complete directory members")
    return dict(files=files, directories=sorted(directories))


def _facts(value, prefix=""):
    names = [prefix + name for name in value["files"]] + [prefix + name for name in value["directories"]]
    for name in names:
        safe_name(name, name.endswith("/"))
    return dict(bytes=sum(v["bytes"] for v in value["files"].values()),
                members=len(names), path_bytes=sum(len(name.encode("utf-8")) for name in names))


def _primary(directory):
    value = inventory(directory)
    need(set(value["files"]) == PRIMARY_FILES, "all original PRIMARY API, three quality ZIPs and twelve input ZIPs")
    for name, item in value["files"].items():
        maximum = PRIMARY_SOURCE_LIMIT if name == "inputs/source.zip" else CORE_LIMIT
        need(item["bytes"] <= maximum, "explicit original PRIMARY member byte boundary: " + name)
    # Capacity does not grant quality. The original R1 reader reopens the nested
    # source, its all-fourteen STOP proof and every preserved scientific return.
    return value


def _r1_stop(directory):
    """Capacity inventory only; the immutable original R1 reader authenticates it."""
    value = inventory(directory)
    need(set(value["files"]) == R1_STOP_FILES, "complete original fourteen-archive R1 STOP, five APIs and two logs")
    for name, item in value["files"].items():
        maximum = R1_SOURCE_LIMIT if name == "source.zip" else CORE_LIMIT
        need(item["bytes"] <= maximum, "unchanged nested original R1 member limit: " + name)
    return value


def natural_envelope(corpus, design):
    """Measure all 24 whole originals, including duration-rejected originals.

    The caller must authenticate the complete corpus before calling this helper.
    Parsing the 42-byte header is metadata inspection, not a codec invocation.
    """
    corpus = Path(corpus)
    need(corpus.is_file() and not corpus.is_symlink() and corpus.stat().st_size == design["upstream"]["bytes"],
         "same complete already-authenticated corpus length")
    roster = design["roster"]
    need(len(roster) == NATURAL_COUNT and len({r["source_id"] for r in roster}) == NATURAL_COUNT,
         "all twenty-four fixed whole natural originals")
    requests = {}
    for row in roster:
        need(all(re.fullmatch(r"[0-9]+", row[k]) for k in ("speaker", "chapter"))
             and re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+", row["source_id"]), "fixed natural identifiers")
        requests[f"LibriSpeech/train-clean-100/{row['speaker']}/{row['chapter']}/{row['source_id']}.flac"] = row
    found = {}
    with tarfile.open(corpus, "r|gz") as archive:
        for member in archive:
            if member.name not in requests:
                continue
            need(member.isfile() and member.name not in found and member.size >= 42, "one complete native member per fixed ID")
            handle = archive.extractfile(member)
            need(handle is not None, "native member header is readable")
            header = handle.read(42)
            need(len(header) == 42 and header[:4] == b"fLaC" and header[4] & 0x7f == 0
                 and int.from_bytes(header[5:8], "big") == 34, "complete native FLAC STREAMINFO")
            packed = int.from_bytes(header[18:26], "big")
            samples = packed & ((1 << 36) - 1)
            need(packed >> 44 == 16000 and ((packed >> 41) & 7) + 1 == 1
                 and ((packed >> 36) & 31) + 1 == 16 and samples > 0 and header[26:42] != b"\0" * 16,
                 "whole mono16k PCM16 sample count with native checksum")
            found[member.name] = dict(source_id=requests[member.name]["source_id"], flac_bytes=member.size,
                samples=samples, wav_bytes=2 * samples + 44, streaminfo_sha256=sha(header))
    need(set(found) == set(requests), "all twenty-four original FLAC headers retained in capacity budget")
    return dict(members=found, bytes=sum(r["flac_bytes"] + r["wav_bytes"] for r in found.values()),
                original_count=NATURAL_COUNT, duration_rejects_retained=True)


def primary_source_budget(directory, corpus, natural_plan):
    """Account for the whole immutable R1 proof before the new primary budget."""
    directory = Path(directory)
    need(not (directory / "CONSUMPTION.json").exists() and not (directory / "CAPACITY.json").exists(),
         "capacity precedes new primary consumption and is recorded only once")
    return _primary_budget(directory, corpus, natural_plan)


def _primary_budget(directory, corpus, natural_plan):
    directory = Path(directory)
    stop = _r1_stop(directory / "ORIGINAL_B24R1_STOP")
    all_files = inventory(directory)["files"]
    # These two later receipts are new metadata already covered by the reserve;
    # lineage independently binds their exact bytes and consumed-marker identity.
    known = {name: value for name, value in all_files.items() if not name.startswith("ORIGINAL_B24R1_STOP/")
             and name not in {"CAPACITY.json", "CONSUMPTION.json"}}
    need(PRIMARY_KNOWN_FILES <= set(known) and any(name.startswith("TRANSPORT/") for name in known),
         "all original TRAIN, actual authorization and complete transport receipts")
    need(all(known[name]["bytes"] <= CORE_LIMIT for name in TRAIN_FILES), "unchanged original TRAIN limits")
    need(not any(name.startswith(("wav/", "native/", "flac/")) for name in known), "no new source audio before capacity admission")
    natural = natural_envelope(corpus, natural_plan)
    stop_bytes = sum(v["bytes"] for v in stop["files"].values())
    known_bytes = sum(v["bytes"] for v in known.values())
    expanded = stop_bytes + known_bytes + TTS_AUDIO_BOUND + natural["bytes"] + NEW_METADATA_LIMIT
    packed = zip_bound(expanded, MAX_MEMBERS, MAX_MEMBERS * MAX_PATH_BYTES)
    need(expanded <= PRIMARY_SOURCE_LIMIT and packed <= PRIMARY_SOURCE_LIMIT,
         "STOP before new primary consumption: whole SOURCE envelope exceeds 768000000 bytes")
    return dict(schema=SCHEMA, role="primary_source", capacity_only=True,
        original_quality_authorized=False, new_model_calls=0, new_source_calls=0, new_observer_calls=0,
        limit_bytes=PRIMARY_SOURCE_LIMIT, original_artifact_limit_bytes=CORE_LIMIT,
        nested_original_R1_source_limit_bytes=R1_SOURCE_LIMIT,
        maximum_members=MAX_MEMBERS, maximum_path_bytes=MAX_PATH_BYTES,
        r1_stop_files=stop["files"], r1_stop_bytes=stop_bytes, known_files=known, known_bytes=known_bytes,
        original_train_artifacts=4, tts_original_count=TTS_COUNT,
        tts_native_samples_max=TTS_NATIVE_SAMPLES, tts_derived_samples_max=TTS_DERIVED_SAMPLES,
        tts_audio_bytes_max=TTS_AUDIO_BOUND, natural=natural, natural_plan_sha256=sha(canonical(natural_plan)),
        corpus_sha256_registered=natural_plan["upstream"]["sha256"], complete_corpus_verification_required=True,
        new_metadata_bytes_max=NEW_METADATA_LIMIT, maximum_expanded_bytes=expanded, maximum_zip_bytes=packed,
        no_member_truncation=True, no_scientific_gate_changed=True)


def primary_generation_capacity(directory, corpus, natural_plan):
    """Recompute the exact saved pre-generation envelope before any new call."""
    directory = Path(directory)
    path = directory / "CAPACITY.json"
    need(path.is_file() and not path.is_symlink(), "original saved capacity receipt before source entry")
    stored = json.loads(path.read_bytes())
    expected = _primary_budget(directory, corpus, natural_plan)
    need(stored == expected, "exact original primary capacity and every proof/header byte before generation")
    return expected


def source_budget(directory, corpus, natural_plan):
    """Compute a bounded package envelope before any source budget is consumed."""
    directory = Path(directory)
    need(not (directory / "CONSUMPTION.json").exists() and not (directory / "CAPACITY.json").exists(),
         "capacity admission precedes source budget consumption")
    return _replica_budget(directory, corpus, natural_plan)


def _replica_budget(directory, corpus, natural_plan):
    directory = Path(directory)
    primary = _primary(directory / "PRIMARY_QUALITY")
    all_files = inventory(directory)["files"]
    need(KNOWN_FILES <= set(all_files), "all four original TRAIN archives and exact known authorization metadata")
    known = {name: value for name, value in all_files.items()
             if name in KNOWN_FILES or name.startswith("TRANSPORT/")}
    need(any(name.startswith("TRANSPORT/") for name in known), "complete already-authenticated transport receipts")
    need(all(known[name]["bytes"] <= CORE_LIMIT for name in TRAIN_FILES), "original TRAIN artifacts remain bounded")
    natural = natural_envelope(corpus, natural_plan)
    primary_bytes = sum(v["bytes"] for v in primary["files"].values())
    known_bytes = sum(v["bytes"] for v in known.values())
    expanded = primary_bytes + known_bytes + TTS_AUDIO_BOUND + natural["bytes"] + NEW_METADATA_LIMIT
    # The original 8192-member ceiling remains in force. Reserve its full
    # filename/header capacity before any unknown TTS metadata exists.
    packed = zip_bound(expanded, MAX_MEMBERS, MAX_MEMBERS * MAX_PATH_BYTES)
    need(expanded <= SOURCE_LIMIT and packed <= SOURCE_LIMIT,
         "STOP before source consumption: complete replica SOURCE capacity envelope exceeds 1280 MB")
    return dict(schema=SCHEMA, role="replication_source", capacity_only=True,
        original_quality_authorized=False, new_model_calls=0, new_source_calls=0, new_observer_calls=0,
        limit_bytes=SOURCE_LIMIT, original_artifact_limit_bytes=CORE_LIMIT,
        maximum_members=MAX_MEMBERS, maximum_path_bytes=MAX_PATH_BYTES,
        primary_files=primary["files"], primary_bytes=primary_bytes,
        known_files=known, known_bytes=known_bytes, original_train_artifacts=4,
        tts_original_count=TTS_COUNT, tts_native_samples_max=TTS_NATIVE_SAMPLES,
        tts_derived_samples_max=TTS_DERIVED_SAMPLES, tts_audio_bytes_max=TTS_AUDIO_BOUND,
        natural=natural, natural_plan_sha256=sha(canonical(natural_plan)),
        corpus_sha256_registered=natural_plan["upstream"]["sha256"], complete_corpus_verification_required=True,
        new_metadata_bytes_max=NEW_METADATA_LIMIT, maximum_expanded_bytes=expanded, maximum_zip_bytes=packed,
        no_member_truncation=True, no_scientific_gate_changed=True)


def replica_generation_capacity(directory, corpus, natural_plan):
    """Authenticate the complete actual PRIMARY envelope again before replica entry."""
    directory = Path(directory)
    path = directory / "CAPACITY.json"
    need(path.is_file() and not path.is_symlink(), "original saved replica capacity receipt before source entry")
    stored = json.loads(path.read_bytes())
    expected = _replica_budget(directory, corpus, natural_plan)
    need(stored == expected, "exact original replica capacity and every proof/header byte before generation")
    return expected


def source_members(directory):
    """Check every actual retained SOURCE byte before any observer is consumed."""
    directory = Path(directory)
    budget = json.loads((directory / "CAPACITY.json").read_bytes())
    need(budget["schema"] == SCHEMA and budget["role"] == "replication_source"
         and budget["limit_bytes"] == SOURCE_LIMIT and budget["original_artifact_limit_bytes"] == CORE_LIMIT
         and budget["tts_audio_bytes_max"] == TTS_AUDIO_BOUND and budget["new_metadata_bytes_max"] == NEW_METADATA_LIMIT,
         "same fixed pre-consumption capacity limits")
    primary = _primary(directory / "PRIMARY_QUALITY")
    need(primary["files"] == budget["primary_files"], "every original PRIMARY byte survives in SOURCE")
    value = inventory(directory)
    files = value["files"]
    need(all(files.get(name) == facts for name, facts in budget["known_files"].items())
         and KNOWN_FILES <= set(budget["known_files"]), "all original TRAIN/authorization/transport bytes survive in SOURCE")
    natural_rows = {row["source_id"]: row for row in budget["natural"]["members"].values()}
    need(len(natural_rows) == NATURAL_COUNT
         and all(type(row["samples"]) is int and row["samples"] > 0
                 and type(row["flac_bytes"]) is int and row["flac_bytes"] >= 42
                 and row["wav_bytes"] == 2 * row["samples"] + 44 for row in natural_rows.values())
         and budget["natural"]["bytes"] == sum(row["flac_bytes"] + row["wav_bytes"] for row in natural_rows.values()),
         "all original natural slots and exact whole-PCM arithmetic in saved budget")
    wav_names = {name for name in files if name.startswith("wav/")}
    flac_names = {name for name in files if name.startswith("flac/")}
    natural_wavs = set()
    for ident, row in natural_rows.items():
        stem = "b24q1-replication-natural-" + ident
        wav, flac = "wav/" + stem + ".wav", "flac/" + stem + ".flac"
        need(files.get(wav, {}).get("bytes") == row["wav_bytes"]
             and files.get(flac, {}).get("bytes") == row["flac_bytes"], "whole natural FLAC and PCM lengths, including rejects")
        natural_wavs.add(wav)
    need(len(flac_names) == NATURAL_COUNT and len(wav_names) == TTS_COUNT + NATURAL_COUNT,
         "all and only 120 complete source WAVs and 24 original FLACs")
    synthetic_wavs = wav_names - natural_wavs
    need(len(synthetic_wavs) == TTS_COUNT
         and all(files[name]["bytes"] <= 2 * TTS_DERIVED_SAMPLES + 44 for name in synthetic_wavs),
         "96 bounded whole synthetic PCM originals")
    native = {name: facts for name, facts in files.items() if name.startswith("native/")}
    native_by_id = {}
    for name, facts in native.items():
        parts = PurePosixPath(name).parts
        need(len(parts) == 3 and re.fullmatch(r"[0-9]{3,}\.s16le", parts[2]), "all original native chunks")
        native_by_id[parts[1]] = native_by_id.get(parts[1], 0) + facts["bytes"]
    need(set(native_by_id) == {PurePosixPath(name).stem for name in synthetic_wavs}
         and all(size <= 2 * TTS_NATIVE_SAMPLES for size in native_by_id.values()),
         "all 96 original native PCM outputs fit the complete-resample bound")
    reserved = set(budget["known_files"]) | {"PRIMARY_QUALITY/" + name for name in primary["files"]}
    payload = wav_names | flac_names | set(native)
    metadata = set(files) - reserved - payload
    need(all(name.endswith(".json") for name in metadata)
         and sum(files[name]["bytes"] for name in metadata) <= NEW_METADATA_LIMIT,
         "complete new metadata fits its fixed envelope, without dropping members")
    facts = _facts(value)
    bound = zip_bound(facts["bytes"], facts["members"], facts["path_bytes"])
    expanded = budget["primary_bytes"] + budget["known_bytes"] + TTS_AUDIO_BOUND + budget["natural"]["bytes"] + NEW_METADATA_LIMIT
    need(budget["primary_bytes"] == sum(v["bytes"] for v in primary["files"].values())
         and budget["known_bytes"] == sum(v["bytes"] for v in budget["known_files"].values())
         and expanded == budget["maximum_expanded_bytes"]
         and zip_bound(expanded, MAX_MEMBERS, MAX_MEMBERS * MAX_PATH_BYTES) == budget["maximum_zip_bytes"]
         and facts["bytes"] <= expanded <= SOURCE_LIMIT
         and bound <= budget["maximum_zip_bytes"] <= SOURCE_LIMIT,
         "complete measured source and conservative upload envelope within 1280 MB")
    return dict(schema=SCHEMA, role="replication_source", capacity_only=True,
        original_quality_authorized=False, actual_bytes=facts["bytes"], maximum_zip_bytes=bound,
        members=facts["members"], primary_files=len(primary["files"]), original_wavs=120,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0)


def primary_source_members(directory):
    """Check every actual retained SOURCE byte before any observer is consumed."""
    directory = Path(directory)
    budget = json.loads((directory / "CAPACITY.json").read_bytes())
    need(budget["schema"] == SCHEMA and budget["role"] == "primary_source"
         and budget["limit_bytes"] == PRIMARY_SOURCE_LIMIT and budget["original_artifact_limit_bytes"] == CORE_LIMIT
         and budget["nested_original_R1_source_limit_bytes"] == R1_SOURCE_LIMIT
         and budget["tts_audio_bytes_max"] == TTS_AUDIO_BOUND and budget["new_metadata_bytes_max"] == NEW_METADATA_LIMIT,
         "same fixed pre-consumption capacity limits")
    primary = _r1_stop(directory / "ORIGINAL_B24R1_STOP")
    need(primary["files"] == budget["r1_stop_files"], "every original R1 STOP byte survives in SOURCE")
    value = inventory(directory)
    files = value["files"]
    need(all(files.get(name) == facts for name, facts in budget["known_files"].items())
         and PRIMARY_KNOWN_FILES <= set(budget["known_files"]), "all original TRAIN/authorization/transport bytes survive in SOURCE")
    natural_rows = {row["source_id"]: row for row in budget["natural"]["members"].values()}
    need(len(natural_rows) == NATURAL_COUNT
         and all(type(row["samples"]) is int and row["samples"] > 0
                 and type(row["flac_bytes"]) is int and row["flac_bytes"] >= 42
                 and row["wav_bytes"] == 2 * row["samples"] + 44 for row in natural_rows.values())
         and budget["natural"]["bytes"] == sum(row["flac_bytes"] + row["wav_bytes"] for row in natural_rows.values()),
         "all original natural slots and exact whole-PCM arithmetic in saved budget")
    wav_names = {name for name in files if name.startswith("wav/")}
    flac_names = {name for name in files if name.startswith("flac/")}
    natural_wavs = set()
    for ident, row in natural_rows.items():
        stem = "b24q1-primary-natural-" + ident
        wav, flac = "wav/" + stem + ".wav", "flac/" + stem + ".flac"
        need(files.get(wav, {}).get("bytes") == row["wav_bytes"]
             and files.get(flac, {}).get("bytes") == row["flac_bytes"], "whole natural FLAC and PCM lengths, including rejects")
        natural_wavs.add(wav)
    need(len(flac_names) == NATURAL_COUNT and len(wav_names) == TTS_COUNT + NATURAL_COUNT,
         "all and only 120 complete source WAVs and 24 original FLACs")
    synthetic_wavs = wav_names - natural_wavs
    need(len(synthetic_wavs) == TTS_COUNT
         and all(files[name]["bytes"] <= 2 * TTS_DERIVED_SAMPLES + 44 for name in synthetic_wavs),
         "96 bounded whole synthetic PCM originals")
    native = {name: facts for name, facts in files.items() if name.startswith("native/")}
    native_by_id = {}
    for name, facts in native.items():
        parts = PurePosixPath(name).parts
        need(len(parts) == 3 and re.fullmatch(r"[0-9]{3,}\.s16le", parts[2]), "all original native chunks")
        native_by_id[parts[1]] = native_by_id.get(parts[1], 0) + facts["bytes"]
    need(set(native_by_id) == {PurePosixPath(name).stem for name in synthetic_wavs}
         and all(size <= 2 * TTS_NATIVE_SAMPLES for size in native_by_id.values()),
         "all 96 original native PCM outputs fit the complete-resample bound")
    reserved = set(budget["known_files"]) | {"ORIGINAL_B24R1_STOP/" + name for name in primary["files"]}
    payload = wav_names | flac_names | set(native)
    metadata = set(files) - reserved - payload
    need(all(name.endswith(".json") for name in metadata)
         and sum(files[name]["bytes"] for name in metadata) <= NEW_METADATA_LIMIT,
         "complete new metadata fits its fixed envelope, without dropping members")
    facts = _facts(value)
    bound = zip_bound(facts["bytes"], facts["members"], facts["path_bytes"])
    expanded = budget["r1_stop_bytes"] + budget["known_bytes"] + TTS_AUDIO_BOUND + budget["natural"]["bytes"] + NEW_METADATA_LIMIT
    need(budget["r1_stop_bytes"] == sum(v["bytes"] for v in primary["files"].values())
         and budget["known_bytes"] == sum(v["bytes"] for v in budget["known_files"].values())
         and expanded == budget["maximum_expanded_bytes"]
         and zip_bound(expanded, MAX_MEMBERS, MAX_MEMBERS * MAX_PATH_BYTES) == budget["maximum_zip_bytes"]
         and facts["bytes"] <= expanded <= PRIMARY_SOURCE_LIMIT
         and bound <= budget["maximum_zip_bytes"] <= PRIMARY_SOURCE_LIMIT,
         "complete measured source and conservative upload envelope within 768000000 bytes")
    return dict(schema=SCHEMA, role="primary_source", capacity_only=True,
        original_quality_authorized=False, actual_bytes=facts["bytes"], maximum_zip_bytes=bound,
        members=facts["members"], r1_stop_files=len(primary["files"]), original_wavs=120,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0)


def producer_members(directory):
    """Measure the exact five producer upload roots before replica scoring."""
    directory = Path(directory)
    primary = _primary(directory / "primary_quality")
    pieces = []
    for name in ("replication_producer", "source_producer"):
        value = inventory(directory / name)
        need(value["files"], "complete nonempty original code producer " + name)
        pieces.append(_facts(value, name + "/"))
    pieces.append(_facts(primary, "primary_quality/"))
    for name in ("FETCH.json", "CONSUMPTION.json"):
        path = directory / name
        if name == "CONSUMPTION.json" and not path.exists():
            size = CONSUMPTION_RESERVE
        else:
            need(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), "actual producer metadata " + name)
            size = path.stat().st_size
            if name == "CONSUMPTION.json":
                need(size <= CONSUMPTION_RESERVE, "bounded exact future consumption receipt")
        pieces.append(dict(bytes=size, members=1, path_bytes=len(name.encode())))
    # Include the three root directory entries in case the uploader emits them.
    total = {key: sum(p[key] for p in pieces) for key in ("bytes", "members", "path_bytes")}
    total["members"] += 3
    total["path_bytes"] += sum(len(name) + 1 for name in ("replication_producer", "source_producer", "primary_quality"))
    bound = zip_bound(total["bytes"], total["members"], total["path_bytes"])
    need(total["members"] <= MAX_MEMBERS and total["bytes"] <= COMPOSITE_LIMIT and bound <= COMPOSITE_LIMIT,
         "STOP before scoring consumption: complete replica producer envelope exceeds 1024 MB")
    return dict(schema=SCHEMA, role="replication_producer", capacity_only=True,
        original_quality_authorized=False, actual_bytes=total["bytes"], maximum_zip_bytes=bound,
        members=total["members"], primary_files=len(primary["files"]),
        future_consumption_reserved=not (directory / "CONSUMPTION.json").exists(),
        new_model_calls=0, new_source_calls=0, new_observer_calls=0)


def primary_producer_members(directory):
    """Original 128 MB PRIMARY producer cap, checked before scoring consumption."""
    directory = Path(directory)
    pieces = []
    for name in ("primary_producer", "source_producer"):
        value = inventory(directory / name)
        need(value["files"], "complete nonempty code producer " + name)
        pieces.append(_facts(value, name + "/"))
    for name in ("FETCH.json", "CONSUMPTION.json"):
        path = directory / name
        if name == "CONSUMPTION.json" and not path.exists():
            size = CONSUMPTION_RESERVE
        else:
            need(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), "actual producer metadata " + name)
            size = path.stat().st_size
            if name == "CONSUMPTION.json":
                need(size <= CONSUMPTION_RESERVE, "bounded exact consumption receipt")
        pieces.append(dict(bytes=size, members=1, path_bytes=len(name.encode())))
    total = {key: sum(p[key] for p in pieces) for key in ("bytes", "members", "path_bytes")}
    total["members"] += 2
    total["path_bytes"] += len("primary_producer/") + len("source_producer/")
    bound = zip_bound(total["bytes"], total["members"], total["path_bytes"])
    need(total["members"] <= MAX_MEMBERS and total["bytes"] <= CORE_LIMIT and bound <= CORE_LIMIT,
         "STOP before scoring consumption: complete primary producer exceeds128000000 bytes")
    return dict(schema=SCHEMA, role="primary_producer", capacity_only=True, original_quality_authorized=False,
        actual_bytes=total["bytes"], maximum_zip_bytes=bound, members=total["members"], limit_bytes=CORE_LIMIT,
        future_consumption_reserved=not (directory / "CONSUMPTION.json").exists(),
        new_model_calls=0, new_source_calls=0, new_observer_calls=0)
