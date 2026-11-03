#!/usr/bin/env python3
"""Bounded offline preparation of official-corpus WAVs for existing KWS sources.

Does not download speech, clone any voice, certify a license or run a model.
For each original speaker, selects one SHA-pinned clip near five seconds,
converts it to a canonical PCM16 WAV, and creates the exact private index and
conversion receipts required by open_reference_sources.py. Nothing is
published as a training or voice-clone approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "auto-tts-voice-clone-20261102"))
import bulk_pool as pool  # noqa: E402
import open_reference_sources as reference  # noqa: E402

SCHEMA = "kws-private-open-corpus-wav-selection-v1"
MAX_ROWS = 128
MAX_SPEAKERS = 16
MAX_CLIP_BYTES = 16 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 32 * 1024 * 1024
RATES = (16000, 24000, 44100, 48000)
SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def source_info(path):
    with wave.open(str(path), "rb") as stream:
        require(stream.getnchannels() == 1 and stream.getsampwidth() == 2
                and stream.getcomptype() == "NONE", "ORIGINAL_NOT_MONO_PCM16")
        rate, count = stream.getframerate(), stream.getnframes()
        require(rate in RATES and rate <= count <= 15 * rate,
                "ORIGINAL_RATE_OR_DURATION")
        data = stream.readframes(count)
        require(len(data) == count * 2 and not stream.readframes(1),
                "ORIGINAL_PCM_INCOMPLETE")
    return {"rate": rate, "frames": count, "duration": count / rate,
            "pcm_sha256": hashlib.sha256(data).hexdigest()}, data


def validate(selection, original_root, terms_file):
    require(type(selection) is dict and
            set(selection) == {"schema", "dataset_id", "dataset_release",
                               "license_snapshot_sha256", "clips"} and
            selection["schema"] == SCHEMA, "SELECTION_SCHEMA")
    known = reference.registry()
    ident = selection["dataset_id"]
    require(type(ident) is str and ident in known and
            known[ident]["reference_status"] == "CANDIDATE_REVIEW_REQUIRED",
            "DATASET_NOT_IN_REVIEWABLE_REFERENCE_LANE")
    require(type(selection["dataset_release"]) is str and
            1 <= len(selection["dataset_release"]) <= 80 and
            selection["dataset_release"].isascii() and
            selection["dataset_release"].isprintable(), "RELEASE_ID")
    require(type(selection["license_snapshot_sha256"]) is str and
            SHA.fullmatch(selection["license_snapshot_sha256"]), "LICENSE_DIGEST")
    require(reference.sha_file(Path(terms_file), reference.MAX_SOURCE_TERMS_BYTES) ==
            selection["license_snapshot_sha256"], "LICENSE_SNAPSHOT_DRIFT")
    clips = selection["clips"]
    require(type(clips) is list and 1 <= len(clips) <= MAX_ROWS,
            "SOURCE_CLIP_COUNT")
    chosen, original_owners, seen_paths = {}, {}, set()
    for row in clips:
        require(type(row) is dict and
                set(row) == {"speaker_key", "original_path", "original_sha256"},
                "SOURCE_CLIP_FIELDS")
        speaker = row["speaker_key"]
        require(type(speaker) is str and
                1 <= len(speaker) <= 128 and speaker == speaker.strip() and
                all(ord(c) >= 32 for c in speaker), "SOURCE_SPEAKER_KEY")
        original = row["original_path"]
        require(original not in seen_paths, "SOURCE_CLIP_DUPLICATE_PATH")
        seen_paths.add(original)
        expected_sha = row["original_sha256"]
        require(type(expected_sha) is str and SHA.fullmatch(expected_sha),
                "SOURCE_SHA256")
        path = reference.confined_file(original_root, original, MAX_CLIP_BYTES)
        actual_sha = reference.sha_file(path, MAX_CLIP_BYTES)
        require(actual_sha == expected_sha, "ORIGINAL_SOURCE_BYTES_CHANGED")
        owner = original_owners.setdefault(actual_sha, speaker)
        require(owner == speaker, "ORIGINAL_AUDIO_CROSSES_SPEAKERS")
        meta, _ = source_info(path)
        rank = (abs(meta["duration"] - 5.0), actual_sha)
        if speaker not in chosen or rank < chosen[speaker][0]:
            chosen[speaker] = (rank, row, meta)
    require(bool(chosen) and len(chosen) <= MAX_ROWS, "NO_VALID_CLIPS")
    return sorted(chosen.items(), key=lambda item: item[0])


def converted_pcm(path, rate, converter, converter_sha, source_pcm):
    if rate in (16000, 24000):
        return source_pcm, rate, None
    require(converter is not None and converter_sha is not None,
            "PINNED_FFMPEG_REQUIRED")
    program = Path(converter)
    require(program.is_absolute() and
            not any(x.is_symlink() for x in (program, *program.parents)),
            "FFMPEG_PATH_ALIAS")
    require(type(converter_sha) is str and SHA.fullmatch(converter_sha) and
            reference.sha_file(program, MAX_EXECUTABLE_BYTES) == converter_sha,
            "FFMPEG_EXECUTABLE_DRIFT")
    target_rate = 16000
    args = [str(program), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(target_rate), "-f", "s16le",
            "-c:a", "pcm_s16le", "pipe:1"]
    with tempfile.TemporaryFile() as temp:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=temp,
                                stderr=subprocess.DEVNULL, timeout=30, check=False,
                                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
                                     "LANG": "C.UTF-8"})
        require(result.returncode == 0, "FFMPEG_CONVERSION_FAILED")
        n = temp.tell()
        require(2 * target_rate <= n <= 2 * 15 * target_rate and n % 2 == 0,
                "CONVERTED_PCM_SIZE")
        temp.seek(0)
        pcm = temp.read(2 * 15 * target_rate + 1)
        require(len(pcm) == n, "CONVERTED_PCM_READ_DRIFT")
    recipe = {"converter_sha256": converter_sha,
              "audio_filter": "ffmpeg-pcm16-mono-swr-v1",
              "target_rate_hz": target_rate,
              "source_rate_hz": rate}
    return pcm, target_rate, hashlib.sha256(pool.canonical(recipe)).hexdigest()


def normalized_wav(destination, pcm, rate):
    with wave.open(str(destination), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return pool.measure_reference(destination.parent, destination.name)


def build(selection, original_root, license_file, output_root, *,
          max_speakers, converter=None, converter_sha=None, execute=False):
    require(type(max_speakers) is int and
            1 <= max_speakers <= MAX_SPEAKERS, "SPEAKER_BUDGET")
    originals = validate(selection, original_root, license_file)
    chosen = originals[:max_speakers]
    require(bool(chosen), "EMPTY_SELECTION")
    need_convert = any(info["rate"] not in (16000, 24000)
                       for _, (_, _, info) in chosen)
    if need_convert:
        require(converter is not None and converter_sha is not None,
                "PINNED_FFMPEG_REQUIRED")
    if not execute:
        return {"status": "SOURCE_PREVIEW_ONLY", "selected": len(chosen),
                "resample_required": sum(info["rate"] not in (16000, 24000)
                                         for _, (_, _, info) in chosen),
                "model_calls": 0, "rights_authenticated": False,
                "training_admitted": False, "shipping_approved": False}
    root = Path(output_root).absolute()
    require(root != ROOT and ROOT not in root.parents and
            root.parent.is_dir() and
            not any(x.is_symlink() for x in (root, *root.parents)) and
            not root.exists(), "PRIVATE_CREATE_ONLY_OUTPUT")
    root.mkdir(mode=0o700)
    (root / "normalized").mkdir(mode=0o700)
    (root / "conversions").mkdir(mode=0o700)
    index_rows = []
    for speaker, (_, row, info) in chosen:
        original = reference.confined_file(original_root, row["original_path"],
                                           MAX_CLIP_BYTES)
        require(reference.sha_file(original, MAX_CLIP_BYTES) ==
                row["original_sha256"], "SOURCE_CHANGED_BEFORE_CONVERSION")
        _, original_pcm = source_info(original)
        pcm, rate, recipe_sha = converted_pcm(original, info["rate"], converter,
                                               converter_sha, original_pcm)
        require(len(pcm) // 2 >= rate and len(pcm) // 2 <= 15 * rate,
                "NORMALIZED_DURATION")
        stem = "ref-" + hashlib.sha256((selection["dataset_id"] + "\0" +
                speaker + "\0" + row["original_sha256"]).encode()).hexdigest()[:24]
        file_rel = stem + ".wav"
        wav = root / "normalized" / file_rel
        metrics = normalized_wav(wav, pcm, rate)
        kind = ("identity" if metrics["wav_sha256"] == row["original_sha256"]
                else "externally-documented")
        convert = {"schema": reference.CONVERSION_SCHEMA,
                   "original_sha256": row["original_sha256"],
                   "normalized_sha256": metrics["wav_sha256"],
                   "conversion": kind,
                   "conversion_recipe_sha256": None if kind == "identity" else (
                        recipe_sha or hashlib.sha256(pool.canonical({
                            "action": "canonical-wave-pcm16-header-v1",
                            "original_rate": info["rate"]})).hexdigest())}
        receipt_rel = stem + ".json"
        receipt = root / "conversions" / receipt_rel
        reference.create_only(receipt, pool.canonical(convert) + b"\n")
        index_rows.append({"speaker_key": speaker,
                           "original_path": row["original_path"],
                           "original_sha256": row["original_sha256"],
                           "normalized_path": file_rel,
                           "normalized_sha256": metrics["wav_sha256"],
                           "conversion_receipt_path": receipt_rel})
    index = {"schema": reference.INDEX_SCHEMA,
             "dataset_id": selection["dataset_id"],
             "dataset_release": selection["dataset_release"],
             "license_snapshot_sha256": selection["license_snapshot_sha256"],
             "rows": index_rows}
    reference.create_only(root / "source-index.json", pool.canonical(index) + b"\n")
    # Re-read actual source, normalized WAV and receipts with the pre-existing
    # strict importer. This remains a candidate and never creates clone permits.
    reference.candidates(index, original_root, root / "normalized",
                         root / "conversions", license_file, max_speakers)
    return {"status": "PRIVATE_CANDIDATES_PREPARED_ONLY",
            "selected": len(index_rows), "model_calls": 0,
            "rights_authenticated": False, "voice_cloning_authorized": False,
            "training_admitted": False, "qualification_allowed": False,
            "shipping_approved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--license-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffmpeg-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    selected = pool.read_json(args.selection, 1024 * 1024)
    status = build(selected, args.original_root, args.license_snapshot,
                   args.output_root, max_speakers=args.max_speakers,
                   converter=args.ffmpeg, converter_sha=args.ffmpeg_sha256,
                   execute=args.execute)
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print("PUBLIC_CORPUS_PREPARATION_HOLD: " + type(exc).__name__, file=sys.stderr)
        raise SystemExit(2)
