#!/usr/bin/env python3
"""Audio-only active-ASR inputs from already normalized private references.

Retain the actual original PCM16 format and the existing conversion receipt.
The decoder receives the normalized 16 kHz samples divided by 32768, with no
new resampling, gain, padding, text prompt or speaker identity fields.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
from pathlib import Path
import struct
import sys
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from active_asr_weak_view import canonical, loads, read_bytes, relative, require, save, save_json, sha
import open_reference_sources as sources

SCHEMA = "kws-reference-active-asr-input-bindings-v1"


def pcm16(raw):
    require(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "REFERENCE_INPUT_RIFF")
    # Some official AISHELL test WAVs retain an oversized outer RIFF length
    # after editing. Their declared data chunks are complete. Keep those bytes
    # and disclose the stale outer extent; never accept a truncated PCM chunk.
    offset, chunks = 12, {}
    while offset + 8 <= len(raw):
        kind, size = struct.unpack_from("<4sI", raw, offset)
        require(kind not in chunks and offset + 8 + size <= len(raw), "REFERENCE_INPUT_CHUNK_EXTENT")
        chunks[kind] = raw[offset + 8:offset + 8 + size]
        offset += 8 + size + size % 2
    require(offset == len(raw) and b"fmt " in chunks and b"data" in chunks,
            "REFERENCE_INPUT_CHUNK_EXTENT")
    with wave.open(io.BytesIO(raw), "rb") as reader:
        rate, frames = reader.getframerate(), reader.getnframes()
        require((reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) == (1, 2, "NONE") and
                rate in (16000, 24000, 44100, 48000) and rate <= frames <= rate * 15,
                "REFERENCE_INPUT_PCM16_FORMAT")
        pcm = reader.readframes(frames)
        require(len(pcm) == frames * 2 and pcm == chunks[b"data"] and not reader.readframes(1),
                "REFERENCE_INPUT_PCM_EXTENT")
    return pcm, {"sha256": sha(pcm), "sample_rate": rate, "channels": 1,
                 "frames": frames, "dtype": "<i2"}


def float_from_pcm16(pcm):
    values = [value[0] / 32768 for value in struct.iter_unpack("<h", pcm)]
    return struct.pack("<" + "f" * len(values), *values)


def prepare(source_manifest, expected_source_sha, index_path, original_root, audio_root,
            conversion_root, terms_path, output):
    """Create one new private, text-free input package, without model calls."""
    source_raw = read_bytes(source_manifest)
    require(sha(source_raw) == expected_source_sha, "REFERENCE_SOURCE_MANIFEST_CHANGED")
    manifest = loads(source_raw)
    refs = sources.pool.validate_sources(manifest, audio_root)
    require(1 <= len(refs) <= sources.MAX_SPEAKERS, "REFERENCE_ASR_BATCH_SIZE")
    index_raw = read_bytes(index_path)
    index = loads(index_raw)
    sources.candidates(index, original_root, audio_root, conversion_root, terms_path, len(refs))
    by_wav = {row["normalized_sha256"]: row for row in index["rows"]}
    require(len(by_wav) == len(index["rows"]), "REFERENCE_INDEX_DUPLICATE_AUDIO")
    output = Path(output).absolute()
    require(not output.exists() and output.parent.is_dir() and sources.ROOT not in output.parents and
            not any(p.is_symlink() for p in (output, *output.parents)), "NEW_PRIVATE_REFERENCE_INPUTS_REQUIRED")
    bodies, clips, bindings = {}, [], []
    archive_stream = io.BytesIO()
    with zipfile.ZipFile(archive_stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for number, ref in enumerate(refs):
            row = by_wav[ref["wav_sha256"]]
            require(row["original_sha256"] == ref["parent_source_sha256"], "REFERENCE_SOURCE_PARENT_CHANGED")
            identifier = f"clip-{number:03d}"
            original = read_bytes(Path(original_root) / relative(row["original_path"]), 16 * 1024 * 1024)
            normalized = read_bytes(Path(audio_root) / relative(ref["audio_path"]), 2 * 1024 * 1024)
            original_pcm, original_info = pcm16(original)
            normalized_pcm, normalized_info = pcm16(normalized)
            require(sha(original) == row["original_sha256"] and sha(normalized) == ref["wav_sha256"] and
                    normalized_info["sample_rate"] == ref["sample_rate_hz"] == 16000 and
                    normalized_info["frames"] == ref["frames"] and normalized_info["sha256"] == ref["pcm_sha256"],
                    "REFERENCE_NORMALIZED_AUDIO_CHANGED")
            converted = float_from_pcm16(normalized_pcm)
            names = {"original": "original/" + identifier + ".wav",
                     "normalized": "normalized/" + identifier + ".wav",
                     "float": "pcm16k/" + identifier + ".f32",
                     "conversion": "conversions/" + identifier + ".json"}
            conversion = read_bytes(Path(conversion_root) / relative(row["conversion_receipt_path"]))
            bodies.update({names["original"]: original, names["normalized"]: normalized,
                           names["float"]: converted, names["conversion"]: conversion})
            member = "original/" + identifier + ".wav"
            archive.writestr(zipfile.ZipInfo(member), original)
            clips.append({"id": identifier, "original_member": member,
                          "original_wav": {"path": names["original"], "sha256": sha(original), "bytes": len(original)},
                          "original_pcm": original_info,
                          "derived_pcm": {"path": names["float"], "sha256": sha(converted),
                                          "sample_rate": 16000, "channels": 1, "frames": ref["frames"], "dtype": "<f4"}})
            bindings.append({"clip_id": identifier, "reference_id": ref["reference_id"],
                             "source_family_id": ref["source_family_id"], "speaker_group_id": ref["speaker_group_id"],
                             "role": ref["role"], "original_path": row["original_path"],
                             "original_wav_sha256": sha(original), "original_pcm_sha256": sha(original_pcm),
                             "original_riff_declared_bytes": int.from_bytes(original[4:8], "little") + 8,
                             "original_riff_extent_matches": int.from_bytes(original[4:8], "little") + 8 == len(original),
                             "normalized_wav": {"path": names["normalized"], "sha256": sha(normalized),
                                                "bytes": len(normalized)}, "normalized_pcm": normalized_info,
                             "conversion_receipt_path": names["conversion"], "conversion_receipt_sha256": sha(conversion)})
    archive_raw = archive_stream.getvalue()
    inputs = {"schema": "active-asr-inputs-v1", "source_archive_sha256": sha(archive_raw),
              "created_at": datetime.now(timezone.utc).isoformat(), "preparation_source_sha256": sha(Path(__file__).read_bytes()),
              "derivation": {"variant": "existing-normalized-reference-pcm16-v1",
                             "source": "actual original mono PCM16 WAV; original sample rate retained per clip",
                             "operation": "existing normalized mono16k PCM16 samples divided by 32768 and encoded <f4",
                             "prior_conversion_receipts": "reference-bindings.json; not decoder context",
                             "original_riff_policy": "retain original bytes; require complete chunks and disclose stale outer extents",
                             "resampling_during_asr_preparation": False, "normalization": False,
                             "clipping": False, "trimming": False, "padding": False}, "clips": clips}
    input_raw = canonical(inputs) + b"\n"
    binding = {"schema": SCHEMA, "inputs_manifest_sha256": sha(input_raw),
               "source_manifest_sha256": expected_source_sha, "source_index_sha256": sha(index_raw),
               "target_text_included": False, "not_decoder_input": True, "rows": bindings}
    output.mkdir(mode=0o700)
    for name, raw in bodies.items():
        save(output, name, raw)
    save(output, "source-original.zip", archive_raw)
    save(output, "inputs.json", input_raw)
    save_json(output, "reference-bindings.json", binding)
    save(output, "preparation-source.py", Path(__file__).read_bytes())
    verify(output)
    return binding


def verify(root):
    """Recheck exact original/normalized/float sample linkage before recognition."""
    root = Path(root)
    raw = read_bytes(root / "inputs.json")
    inputs = loads(raw)
    require(set(inputs) == {"schema", "source_archive_sha256", "created_at", "preparation_source_sha256", "derivation", "clips"} and
            inputs["schema"] == "active-asr-inputs-v1" and
            inputs["derivation"]["variant"] == "existing-normalized-reference-pcm16-v1",
            "REFERENCE_INPUT_FORMAT_CHANGED")
    bindings = loads(read_bytes(root / "reference-bindings.json"))
    require(bindings["schema"] == SCHEMA and bindings["inputs_manifest_sha256"] == sha(raw) and
            bindings["target_text_included"] is False and bindings["not_decoder_input"] is True and
            [row["clip_id"] for row in bindings["rows"]] == [clip["id"] for clip in inputs["clips"]] and
            len({clip["id"] for clip in inputs["clips"]}) == len(inputs["clips"]), "REFERENCE_INPUT_BINDING_CHANGED")
    archive_raw = read_bytes(root / "source-original.zip", 256 * 1024 * 1024)
    require(sha(archive_raw) == inputs["source_archive_sha256"], "REFERENCE_INPUT_ARCHIVE_CHANGED")
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        require(len(archive.namelist()) == len(inputs["clips"]) and
                set(archive.namelist()) == {clip["original_member"] for clip in inputs["clips"]},
                "REFERENCE_INPUT_ARCHIVE_ROSTER")
        for clip, binding in zip(inputs["clips"], bindings["rows"]):
            require(set(clip) == {"id", "original_member", "original_wav", "original_pcm", "derived_pcm"} and
                    clip["derived_pcm"]["sample_rate"] == 16000 and clip["derived_pcm"]["channels"] == 1 and
                    clip["derived_pcm"]["dtype"] == "<f4", "REFERENCE_DECODER_INPUT_FORMAT")
            original = read_bytes(root / relative(clip["original_wav"]["path"]), 16 * 1024 * 1024)
            normalized = read_bytes(root / relative(binding["normalized_wav"]["path"]), 2 * 1024 * 1024)
            derived = read_bytes(root / relative(clip["derived_pcm"]["path"]), 2 * 1024 * 1024)
            _, original_info = pcm16(original)
            normalized_pcm, normalized_info = pcm16(normalized)
            require(sha(original) == clip["original_wav"]["sha256"] == binding["original_wav_sha256"] and
                    len(original) == clip["original_wav"]["bytes"] and
                    original_info == clip["original_pcm"] and original_info["sha256"] == binding["original_pcm_sha256"] and
                    int.from_bytes(original[4:8], "little") + 8 == binding["original_riff_declared_bytes"] and
                    binding["original_riff_extent_matches"] == (int.from_bytes(original[4:8], "little") + 8 == len(original)) and
                    archive.read(clip["original_member"]) == original, "REFERENCE_ORIGINAL_CHANGED")
            require(sha(normalized) == binding["normalized_wav"]["sha256"] and
                    len(normalized) == binding["normalized_wav"]["bytes"] and
                    normalized_info == binding["normalized_pcm"] and normalized_info["sample_rate"] == 16000 and
                    derived == float_from_pcm16(normalized_pcm) and sha(derived) == clip["derived_pcm"]["sha256"] and
                    len(derived) == clip["derived_pcm"]["frames"] * 4, "REFERENCE_FLOAT_CONVERSION_CHANGED")
            conversion_raw = read_bytes(root / relative(binding["conversion_receipt_path"]))
            conversion = loads(conversion_raw)
            require(sha(conversion_raw) == binding["conversion_receipt_sha256"] and
                    conversion["original_sha256"] == sha(original) and conversion["normalized_sha256"] == sha(normalized),
                    "REFERENCE_CONVERSION_RECEIPT_CHANGED")
    return inputs, bindings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "index", "original-root", "audio-root", "conversion-root", "terms", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sources-sha256", required=True)
    args = parser.parse_args(argv)
    result = prepare(args.sources, args.sources_sha256, args.index, args.original_root, args.audio_root,
                     args.conversion_root, args.terms, args.output)
    print(canonical({"clips": len(result["rows"]), "inputs_sha256": result["inputs_manifest_sha256"],
                     "new_model_calls": 0, "resampling_during_asr_preparation": False}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
