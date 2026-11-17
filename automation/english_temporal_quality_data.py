#!/usr/bin/env python3
"""Build and audit one frozen 5 dB condition using saved PCM, stdlib only.

This module never invokes an acoustic model, a head, an optimizer, or a network.
It preserves the original 77 weak-intent IDs and their roles.  The 5 dB mixtures
are evaluation conditions derived from previously exposed recordings, not new
independent samples or verified acoustic labels.  No condition selection or
replacement is performed after a headroom failure.
"""
from __future__ import annotations

import argparse
from array import array
import collections
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
import wave
import zipfile


TRAINING_IDENTITY = {
    "bytes": 61290521,
    "sha256": "2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1",
}
PREVIOUS_IDENTITY = {
    "bytes": 32878537,
    "sha256": "acc6def807b1dd5728111beba4801352d8de9cb453e530f2270005254f44fa15",
}
SNR_DB = 5.0
NOISE_SELECTION_DOMAIN = "english-quality-snr10-v1/"
SCHEMA = "english-temporal-fixed-snr5-data-audit-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def unique_json(raw):
    def unique_pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def no_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=no_constant)


class Archive:
    def __init__(self, raw):
        self.zip = zipfile.ZipFile(io.BytesIO(raw))
        infos = self.zip.infolist()
        require(len(infos) == len({i.filename for i in infos}) <= 1200,
                "unique bounded archive members")
        require(sum(i.file_size for i in infos) <= 256 * 1024**2,
                "bounded archive expansion")
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = stat.S_IFMT(info.external_attr >> 16)
            require(not path.is_absolute() and ".." not in path.parts
                    and "\\" not in info.filename and path.as_posix() == info.filename
                    and not info.is_dir() and mode in (0, stat.S_IFREG),
                    "safe regular archive member")
        self.names = tuple(self.zip.namelist())

    def read(self, name):
        return self.zip.read(name)

    def json(self, name):
        return unique_json(self.read(name))


def pinned_archive(path, expected):
    raw = Path(path).read_bytes()
    require(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"],
            "pinned source archive identity: " + str(path))
    return Archive(raw)


def pcm_values(raw):
    require(len(raw) > 0 and len(raw) % 2 == 0, "nonempty PCM16 bytes")
    return [item[0] for item in struct.iter_unpack("<h", raw)]


def mix_pcm16_5db(source_pcm, noise_pcm):
    """The prior full-clip RMS arithmetic at the single frozen SNR of 5.0 dB."""
    signal, noise = pcm_values(source_pcm), pcm_values(noise_pcm)
    require(len(signal) == len(noise), "equal source/noise lengths")
    ps = math.fsum(float(x) ** 2 for x in signal) / len(signal)
    pn = math.fsum(float(x) ** 2 for x in noise) / len(noise)
    receipt = {
        "snr_definition": "full-original-clip-rms-before-mixture",
        "target_snr_db": SNR_DB, "samples": len(signal),
        "signal_power": ps, "noise_power": pn,
        "source_pcm_sha256": sha(source_pcm), "noise_pcm_sha256": sha(noise_pcm),
        "gain_on_signal": 1.0, "clipped_samples": 0, "fade_samples": 0,
    }
    if ps == 0 or pn == 0:
        return None, {**receipt, "status": "MIX_ZERO_ENERGY_EXCLUDED"}
    gain = math.sqrt(ps / (pn * 10 ** (SNR_DB / 10)))
    values = [round(float(s) + gain * float(n)) for s, n in zip(signal, noise)]
    receipt.update(noise_gain=gain, unbounded_peak=max(abs(x) for x in values))
    if min(values) < -32768 or max(values) > 32767:
        return None, {**receipt, "status": "MIX_HEADROOM_EXCLUDED"}
    mixed = array("h", values)
    if sys.byteorder != "little":
        mixed.byteswap()
    raw = mixed.tobytes()
    actual_noise_power = math.fsum(float(m - s) ** 2
                                   for m, s in zip(values, signal)) / len(signal)
    if actual_noise_power == 0:
        return None, {**receipt, "status": "MIX_QUANTIZED_NOISE_ZERO_EXCLUDED"}
    actual_snr = 10 * math.log10(ps / actual_noise_power)
    if abs(actual_snr - SNR_DB) > 0.05:
        return None, {**receipt, "status": "MIX_QUANTIZATION_SNR_EXCLUDED",
                      "actual_snr_db": actual_snr}
    return raw, {**receipt, "status": "MIXED", "actual_snr_db": actual_snr,
                 "mixed_pcm_sha256": sha(raw)}


def select_noise_track(case_id, role, frames, tracks):
    """Preserve the original 10 dB hash domain, track pool, and offset rule."""
    eligible = sorted((t for t in tracks if t["role"] == role and t["frames"] >= frames),
                      key=lambda t: t["id"])
    require(eligible and len({t["id"] for t in eligible}) == len(eligible),
            "same-role unique noise pool")
    digest = sha((NOISE_SELECTION_DOMAIN + case_id).encode("utf-8"))
    track = eligible[int(digest[:16], 16) % len(eligible)]
    offset = int(digest[16:32], 16) % (track["frames"] - frames + 1)
    return track, offset


def source_pcm(training, selected_ids):
    snapshot = Archive(training.read("SOURCE_SNAPSHOT.zip"))
    source = snapshot.json("SNAPSHOT.json")
    rows = {row["cell_id"]: row for row in source["source_rows"]}
    require(len(rows) == 160 and source["source_denominator"] == 160,
            "original 160-source denominator")
    waves = {}
    shards = ["inputs/old-source/shard-%02d.zip" % i for i in range(9)]
    shards.append("inputs/new/source.zip")
    for shard_name in shards:
        shard = Archive(snapshot.read(shard_name))
        for name in shard.names:
            case_id = PurePosixPath(name).parent.name
            if not name.endswith("/audio-16k.wav") or case_id not in selected_ids:
                continue
            require(case_id not in waves, "one original WAV per selected case")
            raw = shard.read(name)
            require(sha(raw) == rows[case_id]["wav_sha256"], "original WAV identity")
            with wave.open(io.BytesIO(raw), "rb") as audio:
                require(audio.getnchannels() == 1 and audio.getsampwidth() == 2
                        and audio.getframerate() == 16000 and audio.getcomptype() == "NONE"
                        and audio.getnframes() == rows[case_id]["frames"],
                        "original mono 16 kHz PCM16 WAV geometry")
                pcm = audio.readframes(audio.getnframes())
            require(len(pcm) == rows[case_id]["frames"] * 2
                    and sha(pcm) == rows[case_id]["pcm_sha256"], "original PCM identity")
            waves[case_id] = pcm
    require(set(waves) == selected_ids, "all 77 selected original clips recovered")
    return rows, waves


def grouped_counts(rows):
    return dict(sorted(collections.Counter(
        row["role"] + "/" + str(row["weak_class"]) for row in rows).items()))


def build_snr5_conditions(training_zip, previous_zip, output_dir=None):
    """Return (summary, all_77_receipts, eligible_rows, observed_pcm_by_id).

    Optional output contains JSON receipts only; PCM remains in memory.  Source
    reconstruction is lossless outside the mixed original clip.  The caller may
    persist returned PCM during the separately authorized remote experiment.
    """
    training = pinned_archive(training_zip, TRAINING_IDENTITY)
    previous = pinned_archive(previous_zip, PREVIOUS_IDENTITY)
    selection = training.json("SELECTION.json")
    selected_ids = set(selection["eligible_cell_ids"])
    roster = training.json("STREAM_ROSTER.json")
    rows = [row for row in roster if row["id"] in selected_ids]
    require(len(roster) == 117 and len(rows) == len(selected_ids) == 77
            and len({row["id"] for row in roster}) == 117,
            "unchanged 117 clean streams and 77 unique selected clips")
    require(grouped_counts(rows) == {"train/0": 32, "train/1": 8, "train/2": 10,
                                    "development/0": 16, "development/1": 3,
                                    "development/2": 8}, "fixed roles and class mapping")
    source_rows, originals = source_pcm(training, selected_ids)
    libri = training.json("LIBRISPEECH_SOURCES.json")
    tracks = libri["actual_research_tracks"]
    track_map = {track["id"]: track for track in tracks}
    require(len(tracks) == len(track_map) == 40
            and collections.Counter(t["role"] for t in tracks)
            == {"train": 30, "development": 10}, "original 40 background tracks")
    require(libri["source_pool_is_previously_exposed"] is True,
            "historical background exposure retained")
    track_manifest = previous.json("background_tracks/MANIFEST.json")["files"]
    background = {}
    for track in tracks:
        name = track["id"] + ".pcm"
        raw = previous.read("background_tracks/" + name)
        item = track_manifest[name]
        require(len(raw) == item["bytes"] == track["frames"] * 2 == 512000
                and sha(raw) == item["sha256"] == track["pcm_sha256"]
                and item["role"] == track["role"] and item["speaker"] == track["speaker"],
                "historical background PCM, role, and speaker identity")
        background[track["id"]] = raw
    old_receipts = previous.json("evidence/MIX_ROSTER.json")
    old_by_id = {row["id"]: row for row in old_receipts}
    require(len(old_receipts) == len(old_by_id) == 77 and set(old_by_id) == selected_ids,
            "all 77 prior planned conditions bound, including prior exclusion")
    receipts, eligible, pcm_by_id = [], [], {}
    for row in rows:
        case_id, role, src = row["id"], row["role"], row["source"]
        original, original_meta, old = originals[case_id], source_rows[case_id], old_by_id[case_id]
        require(sha(original) == src["source_pcm_sha256"] == old["source_pcm_sha256"]
                and original_meta["role"] == role == old["role"]
                and old["weak_class"] == row["weak_class"]
                and original_meta["source_family_id"] == old["source_family_id"],
                "source clip, class, role, and family identity")
        def span(kind):
            track = track_map[src[kind + "_track"]]
            start, count = src[kind + "_start_sample"], src[kind + "_samples"]
            require(track["role"] == role == src[kind + "_role"]
                    and 0 <= start < start + count <= track["frames"],
                    "unchanged same-role observed context")
            return background[track["id"]][2 * start:2 * (start + count)]
        prefix, context = span("prefix"), span("context")
        clean = prefix + original + context
        require(len(prefix) == 35200 * 2 and len(context) == 5120 * 2
                and len(clean) == row["observed_samples"] * 2
                and sha(clean) == src["combined_pcm_sha256"] == old["original_combined_pcm_sha256"],
                "exact original clean stream and context geometry")
        track, offset = select_noise_track(case_id, role, len(original) // 2, tracks)
        noise = background[track["id"]][2 * offset:2 * offset + len(original)]
        require(track["id"] == old["noise_track"] and offset == old["noise_start_sample"]
                and track["role"] == old["noise_role"] == role
                and track["pcm_sha256"] == old["noise_track_pcm_sha256"]
                and sha(noise) == old["noise_pcm_sha256"] and old["target_snr_db"] == 10.0,
                "preserved prior hash-domain track and offset")
        mixed, receipt = mix_pcm16_5db(original, noise)
        require(receipt["signal_power"] == old["signal_power"]
                and receipt["noise_power"] == old["noise_power"],
                "unchanged original signal and noise RMS basis")
        receipt.update(
            id=case_id, role=role, weak_class=row["weak_class"],
            source_family_id=original_meta["source_family_id"],
            noise_track=track["id"], noise_role=track["role"], noise_start_sample=offset,
            noise_track_pcm_sha256=track["pcm_sha256"], original_combined_pcm_sha256=sha(clean),
            previous_snr10_receipt_sha256=sha(canonical_json(old)),
            independent_source_count_increment=0, condition_training_allowed=False,
        )
        if mixed is not None:
            combined = prefix + mixed + context
            receipt["combined_mixed_pcm_sha256"] = sha(combined)
            eligible.append({**row, "source_family_id": original_meta["source_family_id"],
                             "mix": receipt})
            pcm_by_id[case_id] = combined
        receipts.append(receipt)
    require(len(receipts) == 77, "complete planned condition denominator")
    embeddings = lambda items: sum(len(range(16000, r["observed_samples"] + 1, 1280)) for r in items)
    summary = {
        "schema": SCHEMA, "status": "FIXED_CONDITION_PCM_AUDITED_NO_MODEL_CALLS",
        "training_source": TRAINING_IDENTITY, "previous_snr10_source": PREVIOUS_IDENTITY,
        "snr_db": SNR_DB, "mix_scope": "ORIGINAL_UNTRIMMED_CLIP_ONLY",
        "noise_selection_domain": NOISE_SELECTION_DOMAIN,
        "overflow_policy": "EXCLUDE_WITHOUT_REPLACEMENT_OR_CLIPPING",
        "planned_conditions": len(rows), "eligible_conditions": len(eligible),
        "planned_by_role_class": grouped_counts(rows), "eligible_by_role_class": grouped_counts(eligible),
        "dispositions": dict(sorted(collections.Counter(r["status"] for r in receipts).items())),
        "excluded": [{key: r[key] for key in ("id", "role", "weak_class", "source_family_id", "status", "unbounded_peak")
                      if key in r} for r in receipts if r["status"] != "MIXED"],
        "planned_embedding_calls_before_exclusions": embeddings(rows),
        "eligible_embedding_calls": embeddings(eligible),
        "eligible_onnx_forwards": 2 * embeddings(eligible),
        "eligible_embedding_calls_by_role": {role: embeddings([r for r in eligible if r["role"] == role])
                                             for role in ("train", "development")},
        "planned_original_clip_samples": sum(len(originals[r["id"]]) // 2 for r in rows),
        "planned_observed_samples": sum(r["observed_samples"] for r in rows),
        "planned_receipts_semantic_sha256": sha(canonical_json(receipts)),
        "eligible_roster_semantic_sha256": sha(canonical_json(eligible)),
        "independent_source_count_increment": 0, "condition_training_allowed": False,
        "new_model_calls": 0, "new_tts_asr_phone_calls": 0, "optimizer_updates": 0,
        "human_gold": False, "pronunciation_verified": False, "trusted_negative": False,
        "formal_training_admitted": False, "qualification_allowed": False,
        "model_promotion_allowed": False, "word_end_latency": "NOT_MEASURED",
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        for name, value in (("DATA_AUDIT.json", summary), ("MIX_ROSTER.json", receipts),
                            ("ELIGIBLE_ROSTER.json", eligible)):
            with (destination / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
    return summary, receipts, eligible, pcm_by_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-zip", type=Path, required=True)
    parser.add_argument("--previous-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    summary, _, _, _ = build_snr5_conditions(args.training_zip, args.previous_zip, args.output_dir)
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
