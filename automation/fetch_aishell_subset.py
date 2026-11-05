#!/usr/bin/env python3
"""Retrieve a bounded official AISHELL-3 prefix and retain complete WAV members.

This is source acquisition, not voice/content approval. A prefix digest never
claims to authenticate the full archive. Existing reference identities can be
excluded so an expansion does not silently replace their original recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import wave

URL = "https://openslr.trmal.net/resources/93/data_aishell3.tgz"
MAX_PREFIX = 256 * 1024 * 1024
CHUNK = 1024 * 1024
MAX_MEMBER = 16 * 1024 * 1024
MAX_EXPANDED = 1024 * 1024 * 1024
WAV_PATH = re.compile(r"(train|test)/wav/(SSB[0-9]{4})/(SSB[0-9]{8}\.wav)\Z")
METADATA = {"ReadMe.txt", "ChangeLog", "spk-info.txt", "phone_set.txt",
            "train/content.txt", "test/content.txt"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def fetch_prefix(output, *, etag, target_bytes, seed=None, seed_sha256=None,
                 opener=urllib.request.urlopen):
    """Fetch exact ranges; reject changed releases and range-ignoring servers."""
    output = Path(output)
    require(not output.exists(), "OUTPUT_ALREADY_EXISTS")
    require(type(target_bytes) is int and 1 <= target_bytes <= MAX_PREFIX,
            "PREFIX_BUDGET")
    require(type(etag) is str and etag.startswith('"') and etag.endswith('"')
            and len(etag) < 256 and "\n" not in etag, "STRONG_ETAG_REQUIRED")
    seed_bytes = 0
    if seed is not None:
        seed = Path(seed)
        require(seed.is_file() and not seed.is_symlink(), "SEED_NOT_REGULAR")
        require(digest(seed) == seed_sha256, "SEED_DIGEST_MISMATCH")
        seed_bytes = seed.stat().st_size
        require(0 < seed_bytes < target_bytes, "SEED_PREFIX_SIZE")
    require(output.parent.is_dir(), "OUTPUT_PARENT_MISSING")
    ranges, total_size = [], None
    with output.open("xb") as destination:
        if seed is not None:
            with seed.open("rb") as source:
                shutil.copyfileobj(source, destination, 1024 * 1024)
        for start in range(seed_bytes, target_bytes, CHUNK):
            end = min(start + CHUNK, target_bytes) - 1
            request = urllib.request.Request(URL, headers={
                "Range": f"bytes={start}-{end}", "If-Range": etag,
                "Accept-Encoding": "identity"})
            for attempt in (1, 2):
                try:
                    response = opener(request, timeout=45)
                    break
                except urllib.error.HTTPError as error:
                    print(json.dumps({"range_start": start, "range_end": end,
                                      "http_status": error.code, "attempt": attempt}), flush=True)
                    if error.code not in (502, 503, 504) or attempt == 2:
                        raise
                    time.sleep(1)
            with response:
                require(response.status == 206, "EXACT_RANGE_RESPONSE_REQUIRED")
                require(response.headers.get("ETag") == etag, "ARCHIVE_RELEASE_CHANGED")
                match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)",
                                     response.headers.get("Content-Range", ""))
                require(match is not None and tuple(map(int, match.groups()[:2])) ==
                        (start, end), "CONTENT_RANGE_MISMATCH")
                size = int(match.group(3))
                require(size > target_bytes and (total_size is None or total_size == size),
                        "ARCHIVE_SIZE_CHANGED")
                total_size = size
                count, checksum = 0, hashlib.sha256()
                while count < end - start + 1:
                    block = response.read(min(1024 * 1024, end - start + 1 - count))
                    require(block, "INCOMPLETE_HTTP_RANGE")
                    destination.write(block)
                    checksum.update(block)
                    count += len(block)
                require(not response.read(1), "HTTP_RANGE_EXCEEDS_BOUND")
                ranges.append({"start": start, "end": end, "bytes": count,
                               "sha256": checksum.hexdigest(), "etag": etag})
            destination.flush()
            os.fsync(destination.fileno())
            write_json(output.with_name(output.name + f".range-{start:012d}.json"),
                       {"url": URL, "archive_bytes": total_size, **ranges[-1]})
            print(json.dumps({"downloaded_prefix_bytes": end + 1,
                              "target_prefix_bytes": target_bytes}), flush=True)
    result = {"schema": "kws-official-prefix-retrieval-v1", "url": URL,
              "etag": etag, "archive_bytes": total_size, "complete_archive": False,
              "prefix_bytes": target_bytes, "prefix_sha256": digest(output),
              "seed_bytes": seed_bytes, "seed_sha256": seed_sha256,
              "seed_path": str(seed.absolute()) if seed is not None else None,
              "seed_release_authenticated_by_this_fetch": False if seed is not None else None,
              "retrieved_ranges": ranges, "model_calls": 0}
    write_json(output.with_suffix(output.suffix + ".retrieval.json"), result)
    return result


def wav_info(raw):
    with wave.open(io.BytesIO(raw), "rb") as stream:
        rate, frames = stream.getframerate(), stream.getnframes()
        require(stream.getnchannels() == 1 and stream.getsampwidth() == 2
                and stream.getcomptype() == "NONE", "WAV_FORMAT")
        require(rate in (16000, 24000, 44100, 48000) and rate <= frames <= 15 * rate,
                "WAV_DURATION_OR_RATE")
        pcm = stream.readframes(frames)
        require(len(pcm) == frames * 2 and not stream.readframes(1), "WAV_TRUNCATED")
    return {"sample_rate_hz": rate, "frames": frames, "duration_seconds": frames / rate,
            "pcm_sha256": hashlib.sha256(pcm).hexdigest()}


def select_members(prefix, output, *, exclude_speakers=(), max_speakers=24):
    """Extract only fully read regular members; never extract paths from a tar."""
    output = Path(output)
    require(not output.exists(), "SUBSET_ALREADY_EXISTS")
    require(type(max_speakers) is int and 1 <= max_speakers <= 32, "SPEAKER_BUDGET")
    output.mkdir(mode=0o700)
    selected, metadata, excluded = {}, {}, set(exclude_speakers)
    expanded, complete_members, truncated = 0, 0, False
    # Selected bodies are bounded to 32 x 16 MiB; practical clips are much smaller.
    with tarfile.open(prefix, mode="r|gz") as archive:
        try:
            for member in archive:
                expanded += max(0, member.size)
                require(expanded <= MAX_EXPANDED, "EXPANDED_PREFIX_BUDGET")
                match = WAV_PATH.fullmatch(member.name)
                if member.name not in METADATA and match is None:
                    continue
                require(member.isfile() and not member.issym() and not member.islnk(),
                        "SOURCE_MEMBER_NOT_REGULAR")
                require(0 < member.size <= MAX_MEMBER, "SOURCE_MEMBER_SIZE")
                source = archive.extractfile(member)
                require(source is not None, "SOURCE_MEMBER_MISSING")
                raw = source.read(member.size + 1)
                require(len(raw) == member.size, "INCOMPLETE_TAR_MEMBER")
                complete_members += 1
                record = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                          "tar_offset": member.offset, "tar_data_offset": member.offset_data}
                if member.name in METADATA:
                    require(member.name not in metadata, "DUPLICATE_METADATA_MEMBER")
                    target = output / "original" / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(raw)
                    metadata[member.name] = record
                    continue
                speaker = match.group(2)
                require(match.group(3).startswith(speaker), "WAV_SPEAKER_MISMATCH")
                if speaker in excluded:
                    continue
                if speaker not in selected and len(selected) >= max_speakers:
                    continue
                try:
                    info = wav_info(raw)
                except (ValueError, wave.Error, EOFError):
                    continue
                rank = (abs(info["duration_seconds"] - 5.0), record["sha256"])
                if speaker not in selected or rank < selected[speaker][0]:
                    selected[speaker] = (rank, member.name, raw,
                        {**record, **info, "speaker_key": speaker,
                         "original_partition": match.group(1)})
        except (EOFError, tarfile.ReadError):
            # Only complete member bodies above are retained. CRC/full-archive
            # verification is unavailable for an intentional compressed prefix.
            truncated = True
    rows = []
    for speaker, (_, name, raw, info) in sorted(selected.items()):
        target = output / "original" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        rows.append({"original_path": name, **info})
    result = {"schema": "kws-official-reference-subset-v1",
              "prefix_sha256": digest(prefix), "prefix_bytes": Path(prefix).stat().st_size,
              "full_archive_verified": False, "prefix_ended_before_archive": truncated,
              "requested_new_speakers": max_speakers, "selected_speakers": len(rows),
              "excluded_existing_speakers": sorted(excluded), "complete_members_read": complete_members,
              "metadata": metadata, "references": rows, "model_calls": 0,
              "reference_content_checked": False, "training_admitted": False}
    write_json(output / "source-subset.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--etag", required=True)
    fetch.add_argument("--bytes", type=int, required=True)
    fetch.add_argument("--seed", type=Path)
    fetch.add_argument("--seed-sha256")
    select = commands.add_parser("select")
    select.add_argument("--prefix", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--speaker-registry", type=Path, required=True)
    select.add_argument("--max-speakers", type=int, default=24)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        result = fetch_prefix(args.output, etag=args.etag, target_bytes=args.bytes,
                              seed=args.seed, seed_sha256=args.seed_sha256)
    else:
        registry = json.loads(args.speaker_registry.read_text())
        require(registry.get("schema") == "kws-private-speaker-registry-v1", "REGISTRY_SCHEMA")
        result = select_members(args.prefix, args.output, max_speakers=args.max_speakers,
            exclude_speakers=[row["speaker_key"] for row in registry["entries"]
                              if row["dataset_id"] == "aishell3-slr93"])
    print(json.dumps({key: value for key, value in result.items()
                     if key not in ("references", "metadata", "retrieved_ranges")}, sort_keys=True))


if __name__ == "__main__":
    main()
