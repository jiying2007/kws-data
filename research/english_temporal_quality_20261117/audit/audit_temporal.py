#!/usr/bin/env python3
"""Independent read-only audit of the fixed temporal English quality artifact.

Standard library only: parse authenticated saved evidence, reconstruct PCM,
check cached feature identities, and replay event rules from saved logits.
No model, tensor library, matrix product, subprocess or network is invoked.
Binary and trace readers originate in the prior independent artifact auditor;
this module never imports the producer worker or its data-mixing helper.
"""
from __future__ import annotations
import argparse
import ast
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

TRAINING_PIN = {"bytes": 61290521, "sha256": "2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1"}
INCUMBENT_PIN = {"bytes": 32878537, "sha256": "acc6def807b1dd5728111beba4801352d8de9cb453e530f2270005254f44fa15"}
BASELINE_HEAD_SHA = "6d7677b674f4a5e324aa916c5b777792b641fdfd61af8edc2f4e490275d600c4"
INCUMBENT_HEAD_SHA = "cdceab93cd6d3d36bff62980e8c7d2a95896ad32ecc88cae04fa232393e3e9e7"
OLD_WORKER_SHA = "1164043eb44ea109e4c97ea86a20754c18c171d3804f5b0f370d9ab6778544d2"
FALSE_FLAGS = ("formal_training_admitted", "training_admitted", "human_gold", "trusted_negative", "pronunciation_verified", "model_promotion_allowed", "qualification_allowed")
SHAPES = {"mean96": (96,), "scale96": (96,), "w1": (32, 1536), "b1": (32,), "w2": (3, 32), "b2": (3,)}
FLOAT = struct.Struct("<f")

def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ident(raw):
    return {"bytes": len(raw), "sha256": digest(raw)}


def f32(value):
    return FLOAT.unpack(FLOAT.pack(value))[0]


def json_bytes(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


class Archive:
    def __init__(self, raw, limit=256 * 1024**2):
        self.raw = raw
        self.zip = zipfile.ZipFile(io.BytesIO(raw))
        infos = self.zip.infolist()
        require(len(infos) == len({i.filename for i in infos}) <= 1200, "unique bounded ZIP members")
        require(sum(i.file_size for i in infos) <= limit, "bounded uncompressed ZIP bytes")
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
                    and path.as_posix() == info.filename and not info.is_dir()
                    and (not stat.S_IFMT(mode) or stat.S_ISREG(mode)), "unsafe ZIP member")
        self.names = set(self.zip.namelist())

    def read(self, path):
        return self.zip.read(path)

    def json(self, path):
        return json_bytes(self.read(path))


def npy(raw):
    require(raw[:6] == b"\x93NUMPY" and raw[6:8] in (b"\x01\x00", b"\x02\x00"), "supported NPY header")
    version = raw[6]
    header_size = 2 if version == 1 else 4
    length = int.from_bytes(raw[8:8 + header_size], "little")
    offset = 8 + header_size + length
    require(length < 8192 and offset <= len(raw), "bounded NPY header")
    meta = ast.literal_eval(raw[8 + header_size:offset].decode("latin1"))
    require(set(meta) == {"descr", "fortran_order", "shape"} and meta["fortran_order"] is False,
            "plain contiguous numeric NPY")
    dtype, shape = meta["descr"], meta["shape"]
    require(dtype in ("<f4", "<u8", "|b1") and isinstance(shape, tuple)
            and all(type(n) is int and 0 <= n <= 1000000 for n in shape), "fixed numeric NPY shape/dtype")
    count = math.prod(shape)
    width = {"<f4": 4, "<u8": 8, "|b1": 1}[dtype]
    require(len(raw) - offset == count * width and count <= 5000000, "complete finite-sized NPY payload")
    payload = raw[offset:]
    if dtype == "<f4":
        require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", payload)), "finite saved float values")
    if dtype == "|b1":
        require(set(payload) <= {0, 1}, "boolean saved gap markers")
    return {"dtype": dtype, "shape": shape, "payload": payload, "sha256": digest(raw)}


def npz(raw):
    source = Archive(raw, 64 * 1024**2)
    require(all(n.endswith(".npy") and "/" not in n for n in source.names), "flat numeric NPZ")
    return {n[:-4]: npy(source.read(n)) for n in source.names}


def integers(record):
    require(record["dtype"] == "<u8", "unsigned clock values")
    return [x[0] for x in struct.iter_unpack("<Q", record["payload"])]


def pcm(raw):
    require(len(raw) > 0 and len(raw) % 2 == 0, "nonempty PCM16")
    return [x[0] for x in struct.iter_unpack("<h", raw)]


def event_observation(row, events):
    """Recompute weak-intent diagnostics, never infer acoustic ground truth."""
    source = row["source"]
    start, eof, label = source.get("clip_start_sample"), source.get("original_clip_eof_sample"), row["weak_class"]
    require(type(label) is int and label in range(3) and row["role"] in ("train", "development"), "row class/role")
    require((start is None and eof is None and label == 0) or
            (type(start) is int and type(eof) is int and 0 < start < eof <= row["observed_samples"]), "observed clip geometry")
    require(all(type(e["sample"]) is int and 0 < e["sample"] <= row["observed_samples"]
                and type(e["keyword"]) is int and e["keyword"] in (1, 2) for e in events), "event range/type")
    require(all(a["sample"] < b["sample"] for a, b in zip(events, events[1:])), "ordered events")
    before = [e for e in events if start is not None and e["sample"] <= start]
    after = [e for e in events if start is None or e["sample"] > start]
    matches = [e for e in after if label and e["keyword"] == label]
    wrong = [e for e in after if label and e["keyword"] != label]
    extra = max(0, len(matches) - 1)
    missing = bool(label and not matches)
    return {"id": row["id"], "role": row["role"], "weak_class": label,
            "source_kind": "read_speech" if row["id"].startswith("libri-") else "synthetic_intent",
            "matched": bool(matches), "prefix_count": len(before), "wrong_count": len(wrong),
            "extra_count": extra, "missing_in_observation": missing,
            "weak_negative_disagreement_count": len(events) if label == 0 else 0,
            "research_error": bool(before or wrong or extra or missing or (label == 0 and events)),
            "matched_event_minus_clip_eof_samples": [e["sample"] - eof for e in matches] if eof is not None else [],
            "event_count": len(events), "events": events,
            "eof_prefix_event_count": sum(e["sample"] <= eof for e in events) if eof is not None else None,
            "original_eof_no_matching_event_censored": bool(label and not any(e["sample"] <= eof for e in matches)),
            "observation_no_event_status": "CENSORED_NOT_FRR" if label and not events else None,
            "word_end_latency": "NOT_MEASURED"}


def compare_reported_event(row, calculated, reported):
    require((reported["id"], reported["role"], reported["weak_class"]) ==
            (row["id"], row["role"], row["weak_class"]), "reported event row identity")
    require(reported["matched_intent_after_clip_start"] == calculated["matched"]
            and len(reported["prefix_events"]) == calculated["prefix_count"]
            and len(reported["wrong_keyword_after_clip_start"]) == calculated["wrong_count"]
            and reported["extra_same_keyword_events"] == calculated["extra_count"]
            and reported["no_matching_event_in_observed_context"] == calculated["missing_in_observation"]
            and reported["weak_negative_intent_events"] == calculated["weak_negative_disagreement_count"]
            and reported["research_error"] == calculated["research_error"], "independent event metrics disagree")
    require(reported["event_minus_clip_eof_samples"] == calculated["matched_event_minus_clip_eof_samples"]
            and reported["word_end_latency"] == "NOT_MEASURED"
            and reported["independent_acoustic_evaluation"] is False, "clip clock meaning preserved")
    require(reported["eof_prefix_event_count"] == calculated["eof_prefix_event_count"]
            and reported["no_event_status"] == calculated["observation_no_event_status"],
            "EOF prefix and finite observation remain censored, never FRR or word-end labels")


def grouped(metrics, *, source_kind=False):
    result = {}
    for row in metrics:
        key = row["role"] + "/" + str(row["weak_class"])
        if source_kind:
            key += "/" + row["source_kind"]
        group = result.setdefault(key, {"cases": 0, "matched_intent_cases": 0, "research_error_cases": 0,
            "prefix_events": 0, "wrong_keyword_events": 0, "extra_same_keyword_events": 0,
            "weak_negative_intent_events": 0, "no_matching_event_in_observed_context": 0})
        group["cases"] += 1
        group["matched_intent_cases"] += int(row["matched"])
        group["research_error_cases"] += int(row["research_error"])
        group["prefix_events"] += row["prefix_count"]
        group["wrong_keyword_events"] += row["wrong_count"]
        group["extra_same_keyword_events"] += row["extra_count"]
        group["weak_negative_intent_events"] += row["weak_negative_disagreement_count"]
        group["no_matching_event_in_observed_context"] += int(row["missing_in_observation"])
    return result


def read_trace(artifact, name, rows, clocks_by_id):
    records = [json_bytes(line) for line in artifact.read("predictions/" + name + "-frames.jsonl").splitlines() if line]
    reported = artifact.json("predictions/" + name + "-events.json")
    require(len(reported) == len(rows) and len({r["id"] for r in reported}) == len(rows), "trace event roster")
    reports = {r["id"]: r for r in reported}
    offset, readiness, max_probability_error = 0, 0, 0.0
    metrics = []
    for row in rows:
        times = clocks_by_id[row["id"]]
        frames = records[offset:offset + len(times)]
        require(len(frames) == len(times), "complete frame count")
        offset += len(times)
        candidate = confirmations = blocked_until = 0
        events = []
        for index, (frame, sample) in enumerate(zip(frames, times)):
            require(frame["id"] == row["id"] and type(frame["sample"]) is int and frame["sample"] == sample
                    and type(frame["ready"]) is bool and frame["ready"] == (index >= 15)
                    and type(frame["keyword"]) is int and frame["keyword"] in (0, 1, 2), "frame identity/clock/readiness")
            logits, probabilities = frame["logits"], frame["probabilities"]
            require(len(logits) == len(probabilities) == 3 and all(type(x) in (int, float) and math.isfinite(x)
                    for x in logits + probabilities), "finite recorded logits and probabilities")
            expected_event = 0
            if not frame["ready"]:
                require(logits == [0.0] * 3 and probabilities == [0.0] * 3, "no scores before full history")
            else:
                readiness += 1
                m = max(logits)
                # The recorded head subtracts the maximum in float32 before
                # converting to float64 for exp; retain that exact boundary.
                exponentials = [math.exp(f32(float(v) - m)) for v in logits]
                denominator = math.fsum(exponentials)
                independently_normalized = [f32(v / denominator) for v in exponentials]
                error = max(abs(a - b) for a, b in zip(independently_normalized, probabilities))
                max_probability_error = max(max_probability_error, error)
                require(error <= 2e-7 and all(0 <= p <= 1 for p in probabilities), "recorded softmax does not match logits")
                selected = max(range(3), key=lambda i: independently_normalized[i])
                if selected == 0 or independently_normalized[selected] < f32(0.8) or sample < blocked_until:
                    candidate = confirmations = 0
                else:
                    confirmations = confirmations + 1 if selected == candidate else 1
                    candidate = selected
                    if confirmations == 3:
                        expected_event = selected
                        blocked_until = sample + 13 * 1280
                        candidate = confirmations = 0
            require(frame["keyword"] == expected_event, "fixed event rule differs from saved event")
            if expected_event:
                events.append({"sample": sample, "keyword": expected_event})
        require(reports[row["id"]]["events"] == events, "frame/event-list disagreement")
        metric = event_observation(row, events)
        compare_reported_event(row, metric, reports[row["id"]])
        metrics.append(metric)
    require(offset == len(records), "no unassigned prediction frames")
    return metrics, {"frames": len(records), "ready_outputs": readiness,
                     "case_passes": len(rows), "maximum_softmax_probability_error": max_probability_error}


def original_audio(training, ids):
    snapshot = Archive(training.read("SOURCE_SNAPSHOT.zip"))
    semantic = snapshot.json("SNAPSHOT.json")
    rows = {r["cell_id"]: r for r in semantic["source_rows"]}
    waveforms = {}
    for member in [f"inputs/old-source/shard-{i:02d}.zip" for i in range(9)] + ["inputs/new/source.zip"]:
        shard = Archive(snapshot.read(member))
        for name in shard.names:
            if name.endswith("/audio-16k.wav") and PurePosixPath(name).parent.name in ids:
                case_id = PurePosixPath(name).parent.name
                raw = shard.read(name)
                require(case_id not in waveforms and digest(raw) == rows[case_id]["wav_sha256"], "unique original WAV identity")
                with wave.open(io.BytesIO(raw)) as audio:
                    require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) ==
                            (1, 2, 16000, "NONE"), "original PCM16 WAV geometry")
                    signal = audio.readframes(audio.getnframes())
                require(len(signal) == rows[case_id]["frames"] * 2 and digest(signal) == rows[case_id]["pcm_sha256"], "original audio PCM hash")
                waveforms[case_id] = signal
    require(set(waveforms) == set(ids), "all original selected waveforms available")
    return rows, waveforms


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def unique_rows(rows, name):
    require(isinstance(rows, list), name + " is a list")
    keyed = {row["id"]: row for row in rows}
    require(len(keyed) == len(rows), name + " contains unique cases")
    return keyed


def reject_formal_claims(value, path="root"):
    """All explicit admission flags stay false, including nested receipts."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in (*FALSE_FLAGS, "formal_promotion_allowed", "formal_qualification", "ssc305_qualification",
                       "model_delivered_as_product"):
                require(item is False, "formal claim in weak research: " + path + "/" + key)
            reject_formal_claims(item, path + "/" + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_formal_claims(item, path + "/" + str(index))


def expected_clocks(row):
    require(type(row["observed_samples"]) is int and row["observed_samples"] >= 35200,
            "complete causal observation")
    return list(range(16000, row["observed_samples"] + 1, 1280))


def audit_clean_cache(training, raw):
    """Validate every record boundary and whole-clip supported positive bag."""
    require(raw == training.read("actual-embedding-streams.npz"), "clean cache is copied verbatim")
    cache = npz(raw)
    require(set(cache) == {"embeddings", "samples", "discontinuity_before"}
            and cache["embeddings"]["dtype"] == "<f4"
            and cache["embeddings"]["shape"] == (10354, 96)
            and cache["samples"]["shape"] == (10354,)
            and cache["discontinuity_before"]["dtype"] == "|b1"
            and cache["discontinuity_before"]["shape"] == (10354,), "original clean cache contract")
    roster = training.json("STREAM_ROSTER.json")
    features = unique_rows(training.json("FEATURE_EVIDENCE.json"), "original feature evidence")
    require(len(unique_rows(roster, "original roster")) == len(features) == 117,
            "original 117 feature/stream identities")
    preflight = training.json("FEATURE_PREFLIGHT.json")
    geometry = preflight["geometry"]
    require(geometry["status"] == "FIXED_ONNX_CONV_GEOMETRY_VERIFIED"
            and geometry["head_support_start_before_available"] == 31840
            and geometry["support_end_before_available"] == 128, "fixed real mel support")
    times, gaps = integers(cache["samples"]), cache["discontinuity_before"]["payload"]
    clocks, bags, cursor = {}, [], 0
    for row in roster:
        feature = features[row["id"]]
        start, count = row["first_record"], row["records"]
        require(type(start) is int and type(count) is int and start == cursor and count >= 16,
                "contiguous original feature allocation")
        require((row["role"], row["weak_class"], row["source"]) ==
                (feature["role"], feature["weak_class"], feature["source"]), "feature/source/label identity")
        local = [sample - row["offset"] for sample in times[start:start + count]]
        require(local == expected_clocks(row) and len(local) == count
                and gaps[start:start + count] == bytes([1] + [0] * (count - 1)),
                "each clean stream starts at a discontinuity and has exact 80 ms hops")
        clocks[row["id"]] = local
        ends = feature["head_instance_ends"]
        if row["weak_class"]:
            src = row["source"]
            clip_start, eof = src["clip_start_sample"], src["original_clip_eof_sample"]
            supported = [t for t in local[15:] if t >= eof and t - 35200 <= clip_start
                         and t - 31840 <= clip_start and t - 128 >= eof]
            require(ends == supported and len(ends) >= 3
                    and all(b - a == 1280 for a, b in zip(ends, ends[1:])),
                    "positive bag uses only contiguous whole-clip mel-supported windows")
            bags.append({"id": row["id"], "role": row["role"], "weak_class": row["weak_class"],
                         "supported_ends": ends, "triplets": len(ends) - 2})
        else:
            clip_start = row["source"].get("clip_start_sample")
            original_negative_ends = [t for t in local[15:] if clip_start is None or t > clip_start]
            require(ends == original_negative_ends, "negative bag retains original post-start ready outputs")
        cursor += count
    require(cursor == len(times) == 10354, "all original clean records assigned exactly once")
    return roster, clocks, bags


def audit_condition_cache(raw, rows, clean_clocks, label):
    cache = npz(raw)
    require(set(cache) == {f"{prefix}{i:03d}" for i in range(len(rows)) for prefix in ("e", "t")},
            "closed condition cache: " + label)
    clocks, total = {}, 0
    unique_rows(rows, label + " roster")
    for i, row in enumerate(rows):
        wanted = clean_clocks[row["id"]]
        emb, time = cache[f"e{i:03d}"], cache[f"t{i:03d}"]
        require(emb["dtype"] == "<f4" and emb["shape"] == (len(wanted), 96)
                and time["shape"] == (len(wanted),) and integers(time) == wanted,
                "condition embedding identity/geometry: " + label + "/" + row["id"])
        clocks[row["id"]] = wanted
        total += len(wanted)
    return clocks, total


def verify_draws(draws, previous_draws, clean_rows, noisy_rows):
    require(draws == previous_draws and len(draws) == 96, "exact prior 96 draw sequences")
    for step, used in enumerate(draws):
        expected = []
        for condition, rows in (("clean", clean_rows), ("snr10", noisy_rows)):
            for label in range(3):
                group = [row for row in rows if row["role"] == "train" and row["weak_class"] == label]
                require(group, "all six original training groups exist")
                expected.append({"condition": condition, "id": group[step % len(group)]["id"],
                                 "weak_class": label, "role": "train"})
        require(used == expected, "unchanged six-bag draw excludes development and 5 dB")
    return {"steps": 96, "bag_forwards": 576, "semantic_sha256": digest(canonical(draws)),
            "same_as_completed_control": True, "development_draws": 0, "snr5_draws": 0}


def reconstruct_snr5(training, previous):
    """Independent scalar PCM reconstruction; never imports producer helpers."""
    roster = training.json("STREAM_ROSTER.json")
    selected_ids = set(training.json("SELECTION.json")["eligible_cell_ids"])
    chosen_rows = [row for row in roster if row["id"] in selected_ids]
    require(len(chosen_rows) == len(selected_ids) == 77, "all 77 planned original clips")
    source_rows, originals = original_audio(training, selected_ids)
    tracks = training.json("LIBRISPEECH_SOURCES.json")["actual_research_tracks"]
    track_map = unique_rows(tracks, "historical tracks")
    require(len(tracks) == 40 and collections.Counter(t["role"] for t in tracks) ==
            {"train": 30, "development": 10}, "same previously exposed 40-track pool")
    manifest = previous.json("background_tracks/MANIFEST.json")["files"]
    background = {}
    for track in tracks:
        name = track["id"] + ".pcm"
        raw = previous.read("background_tracks/" + name)
        pin = manifest[name]
        require(ident(raw) == {k: pin[k] for k in ("bytes", "sha256")}
                and len(raw) == track["frames"] * 2 == 512000
                and digest(raw) == track["pcm_sha256"]
                and pin["role"] == track["role"] and pin["speaker"] == track["speaker"],
                "historical noise payload and role")
        background[track["id"]] = raw
    old_mixes = previous.json("evidence/MIX_ROSTER.json")
    old_by_id = unique_rows(old_mixes, "prior mix receipts")
    require(len(old_mixes) == 77 and set(old_by_id) == selected_ids, "prior 77 receipts preserved")
    receipts, eligible, pcm_by_id = [], [], {}
    for row in chosen_rows:
        case_id, src = row["id"], row["source"]
        signal_raw, old = originals[case_id], old_by_id[case_id]
        require(digest(signal_raw) == src["source_pcm_sha256"] == old["source_pcm_sha256"]
                and old["weak_class"] == row["weak_class"]
                and row["role"] == old["role"] == source_rows[case_id]["role"],
                "unchanged source clip, intent and role")
        spans = {}
        for kind in ("prefix", "context"):
            track = track_map[src[kind + "_track"]]
            offset, count = src[kind + "_start_sample"], src[kind + "_samples"]
            require(track["role"] == row["role"] == src[kind + "_role"]
                    and 0 <= offset < offset + count <= track["frames"], "same-role untouched context")
            spans[kind] = background[track["id"]][2 * offset:2 * (offset + count)]
        clean = spans["prefix"] + signal_raw + spans["context"]
        require(len(spans["prefix"]) == 70400 and len(spans["context"]) == 10240
                and len(clean) == row["observed_samples"] * 2
                and digest(clean) == src["combined_pcm_sha256"] == old["original_combined_pcm_sha256"],
                "entire original unmodified context and source reconstruction")
        count = len(signal_raw) // 2
        pool = sorted([t for t in tracks if t["role"] == row["role"] and t["frames"] >= count],
                      key=lambda t: t["id"])
        h = digest(("english-quality-snr10-v1/" + case_id).encode())
        track = pool[int(h[:16], 16) % len(pool)]
        offset = int(h[16:32], 16) % (track["frames"] - count + 1)
        noise_raw = background[track["id"]][2 * offset:2 * (offset + count)]
        require((track["id"], offset, row["role"], track["pcm_sha256"], digest(noise_raw)) ==
                (old["noise_track"], old["noise_start_sample"], old["noise_role"],
                 old["noise_track_pcm_sha256"], old["noise_pcm_sha256"]), "fixed prior noise choice")
        signal, noise = pcm(signal_raw), pcm(noise_raw)
        ps = math.fsum(float(x) ** 2 for x in signal) / count
        pn = math.fsum(float(x) ** 2 for x in noise) / count
        require(ps == old["signal_power"] and pn == old["noise_power"], "same original RMS basis")
        receipt = {"snr_definition": "full-original-clip-rms-before-mixture", "target_snr_db": 5.0,
                   "samples": count, "signal_power": ps, "noise_power": pn,
                   "source_pcm_sha256": digest(signal_raw), "noise_pcm_sha256": digest(noise_raw),
                   "gain_on_signal": 1.0, "clipped_samples": 0, "fade_samples": 0}
        mixed = None
        if ps == 0 or pn == 0:
            receipt["status"] = "MIX_ZERO_ENERGY_EXCLUDED"
        else:
            gain = math.sqrt(ps / (pn * 10 ** 0.5))
            values = [round(float(s) + gain * float(n)) for s, n in zip(signal, noise)]
            receipt.update(noise_gain=gain, unbounded_peak=max(abs(x) for x in values))
            if min(values) < -32768 or max(values) > 32767:
                receipt["status"] = "MIX_HEADROOM_EXCLUDED"
            else:
                residual = math.fsum(float(v - s) ** 2 for v, s in zip(values, signal)) / count
                if residual == 0:
                    receipt["status"] = "MIX_QUANTIZED_NOISE_ZERO_EXCLUDED"
                else:
                    snr = 10 * math.log10(ps / residual)
                    if abs(snr - 5.0) > 0.05:
                        receipt.update(status="MIX_QUANTIZATION_SNR_EXCLUDED", actual_snr_db=snr)
                    else:
                        mixed = struct.pack("<" + str(count) + "h", *values)
                        receipt.update(status="MIXED", actual_snr_db=snr, mixed_pcm_sha256=digest(mixed))
        receipt.update(id=case_id, role=row["role"], weak_class=row["weak_class"],
                       source_family_id=source_rows[case_id]["source_family_id"],
                       noise_track=track["id"], noise_role=track["role"], noise_start_sample=offset,
                       noise_track_pcm_sha256=track["pcm_sha256"], original_combined_pcm_sha256=digest(clean),
                       previous_snr10_receipt_sha256=digest(canonical(old)),
                       independent_source_count_increment=0, condition_training_allowed=False)
        if mixed is not None:
            full = spans["prefix"] + mixed + spans["context"]
            receipt["combined_mixed_pcm_sha256"] = digest(full)
            eligible.append({**row, "source_family_id": source_rows[case_id]["source_family_id"], "mix": receipt})
            pcm_by_id[case_id] = full
        receipts.append(receipt)
    total = sum(len(expected_clocks(row)) for row in eligible)
    summary = {"planned_conditions": 77, "eligible_conditions": len(eligible),
               "dispositions": dict(sorted(collections.Counter(r["status"] for r in receipts).items())),
               "eligible_by_role_class": dict(sorted(collections.Counter(
                   r["role"] + "/" + str(r["weak_class"]) for r in eligible).items())),
               "planned_embedding_calls_before_exclusions": sum(len(expected_clocks(r)) for r in chosen_rows),
               "eligible_embedding_calls": total, "eligible_onnx_forwards": 2 * total,
               "planned_receipts_semantic_sha256": digest(canonical(receipts)),
               "eligible_roster_semantic_sha256": digest(canonical(eligible))}
    require(summary["planned_embedding_calls_before_exclusions"] == 2834 and total <= 2834 <= 4000,
            "predeclared finite 77-condition frontend budget")
    return summary, receipts, eligible, pcm_by_id


def compare_cases(before, after, condition):
    old, new = unique_rows(before, condition + " control"), unique_rows(after, condition + " candidate")
    require(set(old) == set(new), "same paired case subset: " + condition)
    pairs = []
    for prior in before:
        current = new[prior["id"]]
        require((prior["role"], prior["weak_class"]) == (current["role"], current["weak_class"]),
                "paired roles/classes")
        safety_fields = ("prefix_count", "wrong_count", "extra_count", "weak_negative_disagreement_count")
        introduced = {key: current[key] - prior[key] for key in safety_fields if current[key] > prior[key]}
        pairs.append({"condition": condition, "id": prior["id"], "role": prior["role"],
                      "weak_class": prior["weak_class"], "control_error": prior["research_error"],
                      "candidate_error": current["research_error"],
                      "rescued": prior["research_error"] and not current["research_error"],
                      "regressed": not prior["research_error"] and current["research_error"],
                      "introduced_event_errors": introduced,
                      "control_matched": prior["matched"], "candidate_matched": current["matched"],
                      "control_missing_in_observation": prior["missing_in_observation"],
                      "control_clip_eof_offsets": prior["matched_event_minus_clip_eof_samples"],
                      "candidate_clip_eof_offsets": current["matched_event_minus_clip_eof_samples"]})
    return pairs


def strict_gate(paired, snr5_rows):
    """Predeclared incumbent-relative gate; no aggregate can hide a regression."""
    require({r["condition"] for r in paired} == {"clean", "snr10", "snr5"}, "all three gate conditions")
    reasons = []
    regressions = [r for r in paired if r["regressed"]]
    new_errors = [r for r in paired if r["introduced_event_errors"]]
    if any(r["condition"] == "clean" for r in regressions):
        reasons.append("CLEAN_CASE_REGRESSION")
    if any(r["condition"] == "snr10" for r in regressions):
        reasons.append("SNR10_CASE_REGRESSION")
    if any(r["condition"] == "snr5" for r in regressions):
        reasons.append("SNR5_CASE_REGRESSION")
    if new_errors:
        reasons.append("NEW_PREFIX_WRONG_EXTRA_OR_NEGATIVE_EVENT")
    dev_rescues = [r for r in paired if r["condition"] == "snr10" and r["role"] == "development"
                   and r["weak_class"] in (1, 2) and r["control_missing_in_observation"] and r["rescued"]]
    if not dev_rescues:
        reasons.append("NO_SNR10_DEVELOPMENT_MISS_RESCUED")
    coverage = {str(label): sum(r["role"] == "development" and r["weak_class"] == label
                               for r in snr5_rows) for label in (1, 2)}
    if min(coverage.values()) < 1:
        reasons.append("SNR5_DEVELOPMENT_KEYWORD_COVERAGE_INSUFFICIENT")
    return {"accepted": not reasons,
            "decision": "RESEARCH_CANDIDATE_RETAINED" if not reasons else "RESEARCH_CANDIDATE_REJECTED",
            "reasons": reasons, "paired_case_regressions": len(regressions),
            "new_event_error_cases": len(new_errors), "snr10_development_rescues": len(dev_rescues),
            "snr5_development_positive_coverage": coverage,
            "model_promotion_allowed": False, "qualification_allowed": False}


def authenticate_sources(training, previous):
    require(ident(training.raw) == TRAINING_PIN, "fixed original training ZIP SHA/bytes")
    require(ident(previous.raw) == INCUMBENT_PIN, "fixed consumed incumbent ZIP SHA/bytes")
    require(digest(training.read("train/causal-head.npz")) == BASELINE_HEAD_SHA
            and digest(previous.read("models/candidate-head.npz")) == INCUMBENT_HEAD_SHA,
            "exact original and completed-control head identities")
    require(previous.read("models/baseline-head.npz") == training.read("train/causal-head.npz"),
            "completed control used this exact baseline")


def input_preflight(training, previous):
    authenticate_sources(training, previous)
    rows, clocks, positive_bags = audit_clean_cache(training, training.read("actual-embedding-streams.npz"))
    previous_rows = previous.json("cache/mixed-roster.json")
    by_id = unique_rows(rows, "clean roster")
    require(len(previous_rows) == 76 and all(all(row[key] == by_id[row["id"]][key]
            for key in by_id[row["id"]]) for row in previous_rows), "unchanged prior 76 noisy streams")
    _, previous_embeddings = audit_condition_cache(previous.read("cache/mixed-embeddings.npz"),
                                                    previous_rows, clocks, "snr10")
    require(previous_embeddings == 2804, "2804 consumed noisy feature rows reused")
    draw_receipt = previous.json("evidence/TRAINING.json")
    draws = verify_draws(draw_receipt["draws"], draw_receipt["draws"], rows, previous_rows)
    summary, receipts, eligible, _ = reconstruct_snr5(training, previous)
    return {"schema": "independent-temporal-input-preflight-v1", "status": "INPUTS_INDEPENDENTLY_VERIFIED",
            "source_training_zip": ident(training.raw), "source_incumbent_zip": ident(previous.raw),
            "clean_streams": len(rows), "clean_embedding_rows": 10354,
            "snr10_streams": len(previous_rows), "snr10_embedding_rows": previous_embeddings,
            "positive_supported_bags": positive_bags, "draws": draws, "snr5": summary,
            "new_model_calls_by_this_auditor": 0, "model_forward_numerics_recomputed": False,
            "mixture_arithmetic_independently_reconstructed": True,
            "same_source_condition_is_not_independent_audio": True,
            "independent_source_count_increment": 0, "word_end_latency": "NOT_MEASURED",
            **dict.fromkeys(FALSE_FLAGS, False)}


def ast_form(source):
    return ast.dump(ast.parse(source), include_attributes=False)


def source_only_loss_contract(raw):
    """Inspect frozen source expressions; this does not execute producer code."""
    tree = ast.parse(raw.decode("utf-8"))
    worker = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"), None)
    require(worker is not None, "reviewed worker source is present")
    assignments = {}
    for node in ast.walk(worker):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments.setdefault(node.targets[0].id, []).append(node.value)
    expected_expressions = {
        "windows": 'np.asarray([matrix[positions[t] - 15:positions[t] + 1] for t in ends], dtype=np.float32)',
        "normalized": '((windows - weights["mean96"]) / weights["scale96"]).reshape(len(windows), 1536)',
        "model": 'torch.nn.Sequential(torch.nn.Linear(1536, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))',
        "optimizer": 'torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)',
        "logp": 'torch.log_softmax(model(prepared[key]), dim=-1)',
        "triple_logp": 'torch.amin(logp[:, label].unfold(0, 3, 1), dim=1)',
        "loss": 'torch.stack(terms).mean()',
    }
    for name, expression in expected_expressions.items():
        nodes = assignments.get(name, [])
        wanted = ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)
        require(len(nodes) == 1 and ast.dump(nodes[0], include_attributes=False) == wanted,
                "same-data/optimizer or sole positive-loss source expression: " + name)
    losses = [node for node in ast.walk(worker) if isinstance(node, ast.If)
              and ast.dump(node.test, include_attributes=False) ==
              ast.dump(ast.parse("label == 0", mode="eval").body, include_attributes=False)]
    require(len(losses) == 1, "one explicit background-versus-positive loss branch")
    branch = losses[0]
    wanted_background = ast.parse("terms.append(-logp[:, 0].mean())").body
    wanted_positive = ast.parse("triple_logp = torch.amin(logp[:, label].unfold(0, 3, 1), dim=1)\n"
                               "terms.append(math.log(len(triple_logp)) - torch.logsumexp(triple_logp, dim=0))").body
    require([ast.dump(n, include_attributes=False) for n in branch.body] ==
            [ast.dump(n, include_attributes=False) for n in wanted_background]
            and [ast.dump(n, include_attributes=False) for n in branch.orelse] ==
            [ast.dump(n, include_attributes=False) for n in wanted_positive],
            "unchanged background CE and exact mean of triplet minima surrogate")
    calls = [ast.unparse(node) for node in ast.walk(worker) if isinstance(node, ast.Call)]
    for required_call in ("torch.set_num_threads(1)", "torch.manual_seed(20261117)",
                          "torch.use_deterministic_algorithms(True)", "optimizer.zero_grad(set_to_none=True)",
                          "loss.backward()", "optimizer.step()",
                          "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)"):
        require(calls.count(required_call) == 1, "exact one original training operation: " + required_call)
    loops = [node for node in ast.walk(worker) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "step"]
    require(len(loops) == 1 and ast.unparse(loops[0].iter) == "range(96)", "one fixed 96-update loop")
    forbidden_top_imports = {"torch", "numpy", "onnx", "onnxruntime", "soundfile"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            require(not {name.name.split(".")[0] for name in node.names} & forbidden_top_imports,
                    "no top-level numeric model imports")
        elif isinstance(node, ast.ImportFrom):
            require((node.module or "").split(".")[0] not in forbidden_top_imports,
                    "no top-level numeric model imports")
    return {"source_sha256": digest(raw), "exact_temporal_amin_mean_pooling_checked": True,
            "original_background_loss_checked": True, "original_head_normalization_optimizer_checked": True,
            "static_inspection_only": True, "numerical_loss_or_gradient_tested_locally": False}


def verify_phases(phases, frozen, final_counts, candidate_sha):
    names = ["WORKER_STARTED", "TRAINING_STARTED", "CANDIDATE_FROZEN",
             "OLD_CONDITION_EVALUATION_STARTED", "OLD_CONDITION_EVALUATION_COMPLETED",
             "SNR5_DATA_PREPARATION_STARTED", "SNR5_DATA_PREPARATION_COMPLETED",
             "SNR5_FEATURES_STARTED", "SNR5_FEATURES_COMPLETED",
             "SNR5_EVALUATION_STARTED", "SNR5_EVALUATION_COMPLETED", "COMPLETED"]
    require([item["phase"] for item in phases] == names
            and [item["phase_number"] for item in phases] == list(range(1, len(names) + 1)),
            "single closed ordered phase ledger")
    prior = {key: 0 for key in final_counts}
    for index, item in enumerate(phases):
        progress = item["progress"]
        require(set(progress) == set(final_counts) and all(type(value) is int and
                prior[key] <= value <= final_counts[key] for key, value in progress.items()),
                "monotonic bounded phase counters")
        require(item["candidate_sha256"] == (None if index < 2 else candidate_sha),
                "one candidate hash after freezing")
        if index < 8:
            require(progress["new_embedding_calls_started"] == progress["new_embedding_calls_completed"] ==
                    progress["onnx_forward_started"] == progress["onnx_forward_completed"] == 0,
                    "no 5 dB model work before its post-freeze phase")
        if index < 2:
            require(all(value == 0 for value in progress.values()), "no numeric work before training phase")
        else:
            require(progress["optimizer_updates_started"] == progress["optimizer_updates_completed"] == 96
                    and progress["torch_forward_calls_started"] == progress["torch_forward_calls_completed"] == 576,
                    "training complete and immutable after candidate freeze")
        if index == 2:
            require(progress["head_case_passes_started"] == progress["head_case_passes_completed"] ==
                    progress["head_ready_outputs"] == 0, "freeze before every new candidate evaluation")
        if 4 <= index < 10:
            require(progress["head_case_passes_started"] == progress["head_case_passes_completed"] == 193,
                    "only 193 old-condition evaluations precede 5 dB evaluation")
        if index >= 8:
            for key in ("new_embedding_calls_started", "new_embedding_calls_completed",
                        "onnx_forward_started", "onnx_forward_completed"):
                require(progress[key] == final_counts[key], "all 5 dB embeddings computed exactly once")
        if index >= 10:
            require(progress == final_counts, "all numeric work complete at evaluation-completed phase")
        prior = progress
    require(phases[-1]["progress"] == final_counts and frozen["phase_number"] == 3
            and frozen["progress"] == phases[2]["progress"]
            and frozen["candidate"]["sha256"] == candidate_sha
            and frozen["snr5_model_calls_before_freeze"] == 0
            and frozen["all_future_evaluation_uses_this_frozen_head"] is True,
            "frozen candidate receipt is bound to phase ledger")
    return {"phases": len(phases), "candidate_frozen_phase": 3,
            "snr5_feature_start_phase": 8, "snr5_evaluation_start_phase": 10,
            "same_frozen_candidate_through_completion": True,
            "source_and_receipt_order_checked_not_external_runtime_attestation": True}


def verify_plan(plan):
    fixed = {"snr_db": 5.0, "planned_new_conditions": 77, "planned_maximum_embedding_calls": 2834,
             "maximum_embedding_calls": 4000, "maximum_onnx_forwards": 8000,
             "maximum_head_case_passes": 424, "maximum_optimizer_updates": 96,
             "maximum_training_bag_forwards": 576, "maximum_execution_seconds": 900,
             "retry_count": 0, "new_tts_asr_phone_calls": 0, "new_training_frontend_calls": 0,
             "seed": 20261117, "cpu_threads": 1, "learning_rate": 0.001, "weight_decay": 0.0001,
             "gradient_clip_norm": 1.0, "threshold": 0.8, "confirm_embeddings": 3, "cooldown_embeddings": 13,
             "new_training_arms": 1, "optimizer": "FRESH_ADAMW", "warmstart": "ORIGINAL_BASELINE",
             "normalization": "FROZEN_ORIGINAL_TRAIN_ONLY",
             "positive_loss": "NEGATIVE_LOG_MEAN_THREE_CONSECUTIVE_MIN_TARGET_PROBABILITY",
             "positive_min_operator": "TORCH_AMIN_WITH_TIE_GRADIENT_SHARING",
             "background_loss": "MEAN_WINDOW_BACKGROUND_CROSS_ENTROPY",
             "bag_aggregation": "SIX_EQUAL_CONDITION_CLASS_TERMS",
             "training_conditions": ["clean", "snr10"], "evaluation_conditions": ["clean", "snr10", "snr5"],
             "draws": "EXACT_PREVIOUS_96_BY_6_DRAW_SEQUENCE",
             "positive_geometry": "UNCHANGED_ORIGINAL_MEL_SUPPORTED_HEAD_INSTANCE_ENDS",
             "development_tuning": False, "checkpoint_selection": False, "incumbent_retraining": False,
             "old_frontend_recomputation": False, "snr5_training_allowed": False,
             "snr5_evaluation_after_candidate_frozen": True,
             "mix_scope": "ORIGINAL_UNTRIMMED_CLIP_ONLY",
             "noise_binding": "EXACT_PREVIOUS_TRACK_OFFSET_AND_HASH_DOMAIN",
             "overflow_policy": "EXCLUDE_WITHOUT_REPLACEMENT_OR_CLIPPING"}
    for key, value in fixed.items():
        require(type(plan.get(key)) is type(value) and plan[key] == value, "frozen plan value: " + key)
    retention = {"clean_no_previously_correct_case_regression": True,
                 "snr10_no_previously_correct_case_regression": True,
                 "snr10_minimum_rescued_development_misses": 1,
                 "all_conditions_no_per_case_prefix_wrong_extra_or_negative_increase": True,
                 "snr5_no_previously_correct_case_regression": True,
                 "snr5_minimum_development_positive_cases_per_keyword": 1,
                 "reference": "PREVIOUS_AUGMENTED_96_UPDATE_HEAD", "formal_promotion_allowed": False}
    require(plan.get("research_retention") == retention and all(plan.get(key) is False for key in FALSE_FLAGS),
            "predeclared strict research retention and no product admission")
    require({key: plan["training_source"][key] for key in TRAINING_PIN} == TRAINING_PIN
            and {key: plan["incumbent_source"][key] for key in INCUMBENT_PIN} == INCUMBENT_PIN,
            "plan pinned sources")
    reject_formal_claims(plan)


def expected_retention_record(gate, paired):
    criteria = {condition + "_no_previously_correct_case_regression": not any(
        row["regressed"] for row in paired if row["condition"] == condition)
        for condition in ("clean", "snr10", "snr5")}
    criteria["snr10_at_least_one_rescued_development_miss"] = gate["snr10_development_rescues"] >= 1
    criteria["all_conditions_no_per_case_event_error_increase"] = gate["new_event_error_cases"] == 0
    criteria["snr5_both_development_keywords_present"] = min(gate["snr5_development_positive_coverage"].values()) >= 1
    return {"passed": gate["accepted"],
            "status": "PASSED_RESEARCH_RETENTION_GATE" if gate["accepted"] else
                      "FAILED_RESEARCH_RETENTION_GATE_DIAGNOSTIC_ONLY",
            "criteria": criteria, "reference": "PREVIOUS_AUGMENTED_96_UPDATE_HEAD",
            "formal_promotion_allowed": False, **dict.fromkeys(FALSE_FLAGS, False)}


def audit(quality, training, previous, expected_run=None, expected_head=None):
    authenticate_sources(training, previous)
    plan, result = quality.json("evidence/PLAN.json"), quality.json("evidence/RESULT.json")
    verify_plan(plan)
    require(result["status"] == "COMPLETED_WEAK_TEMPORAL_QUALITY_RESEARCH"
            and result["schema"] == plan["schema"] and result["run_attempt"] == 1,
            "one completed weak temporal research execution")
    require(expected_run is None or str(result["run_id"]) == str(expected_run), "expected public run identity")
    require(expected_head is None or result["head_sha"] == expected_head, "expected public source commit")
    require(all(result.get(key) is False for key in FALSE_FLAGS), "result cannot grant formal admission")
    require(result["training_source"] == plan["training_source"]
            and result["incumbent_source"] == plan["incumbent_source"], "result/plan source binding")
    require("evidence/FAILED_NO_RETRY.json" not in quality.names, "completed artifact also records a failure")
    for name in quality.names:
        if name.endswith(".json"):
            reject_formal_claims(quality.json(name), name)
    manifest = quality.json("evidence/WORKER_MANIFEST.json")["files"]
    for name, pin in manifest.items():
        require(name in quality.names and ident(quality.read(name)) == pin, "worker manifest byte identity: " + name)
    later = {"evidence/WORKER_MANIFEST.json", "evidence/model-worker.exit.json",
             "evidence/PARENT_RESULT.json", "evidence/SUPERVISOR_EXIT.json"}
    require(quality.names - set(manifest) == later, "closed artifact with exactly known post-worker receipts")
    supervisor = quality.json("evidence/SUPERVISOR_EXIT.json")
    require(supervisor["retry_count"] == 0 and supervisor["numeric_attempt"] == 1
            and supervisor["maximum_seconds"] == 900 and 0 <= supervisor["wall_seconds"] <= 900,
            "bounded one-attempt supervisor")
    worker_exit = quality.json("evidence/model-worker.exit.json")
    require(worker_exit["returncode"] == 0 and worker_exit["error"] is None
            and worker_exit["group_cleanup_checked_after_leader_exit"] is True, "complete supervised worker exit")
    parent = quality.json("evidence/PARENT_RESULT.json")
    require(parent["status"] == "COMPLETED_ONE_SUPERVISED_WORKER" and parent["retry_count"] == 0
            and parent["numeric_attempt"] == 1
            and parent["worker_result"] == ident(quality.read("evidence/RESULT.json")), "parent binds complete worker result")
    inputs = quality.json("evidence/INPUTS.json")
    require(inputs["actual_training"] == TRAINING_PIN and inputs["actual_incumbent"] == INCUMBENT_PIN
            and inputs["training_source"] == plan["training_source"]
            and inputs["incumbent_source"] == plan["incumbent_source"]
            and inputs["old_embedding_rows_reused"] == {"clean": 10354, "snr10": 2804}
            and inputs["old_frontend_recomputation"] is False and inputs["new_training_frontend_calls"] == 0,
            "input receipts bind exact reused features and no recomputation")
    copied = inputs["copied_members"]
    for destination, info in copied.items():
        require(info["source"] in ("training", "incumbent"), "known copied-member source")
        source = training if info["source"] == "training" else previous
        raw = source.read(info["member"])
        require(quality.read(destination) == raw and ident(raw) == {k: info[k] for k in ("bytes", "sha256")},
                "verbatim authenticated copied member: " + destination)
    required_copies = {"sources/" + n: (training, n) for n in (
        "STREAM_ROSTER.json", "FEATURE_EVIDENCE.json", "FEATURE_PREFLIGHT.json", "PER_CASE_EVENTS.json",
        "LIBRISPEECH_SOURCES.json", "SELECTION.json")}
    required_copies.update({"cache/clean-original.npz": (training, "actual-embedding-streams.npz"),
        "models/baseline-head.npz": (training, "train/causal-head.npz"),
        "models/aug96-control-head.npz": (previous, "models/candidate-head.npz"),
        "cache/mixed-snr10.npz": (previous, "cache/mixed-embeddings.npz"),
        "cache/snr10-roster.json": (previous, "cache/mixed-roster.json"),
        "sources/PREVIOUS_TRAINING.json": (previous, "evidence/TRAINING.json")})
    bundle = "english-snapshot160-embedding-research-bundle/"
    for filename in ("causal_embedding.py", "portable_head.py", "weak_embedding_contract.py"):
        required_copies["sources/" + filename] = (training, bundle + "runtime/" + filename)
    for filename in ("melspectrogram.onnx", "embedding_model.onnx"):
        required_copies["models/" + filename] = (training, bundle + "models/" + filename)
    for name in previous.names:
        if name.startswith("predictions/"):
            required_copies["predictions/reused/" + PurePosixPath(name).name] = (previous, name)
    for destination, (source, member) in required_copies.items():
        require(destination in copied and quality.read(destination) == source.read(member),
                "mandatory original evidence reused verbatim: " + destination)
    contract_tree = ast.parse(quality.read("sources/weak_embedding_contract.py").decode())
    class_values = [ast.literal_eval(node.value) for node in contract_tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "CLASSES" for target in node.targets)]
    require(class_values == [{"background": 0, "Hey Vosen": 1, "Hey Mero": 2}], "exact original keyword class mapping")
    for key, filename in (("worker_sha256", "english_temporal_quality_once.py"),
                          ("data_helper_sha256", "english_temporal_quality_data.py"),
                          ("workflow_sha256", "english-temporal-quality-once.yml")):
        raw = quality.read("sources/" + filename)
        require(digest(raw) == plan[key] and inputs["source_contract"][key]["sha256"] == plan[key]
                and inputs["source_contract"][key]["bytes"] == len(raw), "executed source pin: " + key)
    source_check = source_only_loss_contract(quality.read("sources/english_temporal_quality_once.py"))
    runtime, old_runtime = quality.json("evidence/RUNTIME.json"), previous.json("evidence/RUNTIME.json")
    for key in ("numpy", "torch", "onnx", "onnxruntime", "execution_provider", "numeric_threads"):
        require(runtime[key] == old_runtime[key], "same numerical dependency/runtime: " + key)
    require(runtime["soundfile_or_corpus_decoder_used"] is False, "no repeated historical PCM decode")
    roster, clean_clocks, positive_bags = audit_clean_cache(training, quality.read("cache/clean-original.npz"))
    snr10_rows = quality.json("cache/snr10-roster.json")
    require(len(snr10_rows) == 76, "same complete incumbent noisy subset")
    snr10_clocks, snr10_count = audit_condition_cache(quality.read("cache/mixed-snr10.npz"), snr10_rows,
                                                    clean_clocks, "snr10")
    require(snr10_count == 2804, "all 2804 consumed 10 dB cache rows")
    summary, receipts, snr5_rows, expected_pcm = reconstruct_snr5(training, previous)
    require(quality.json("evidence/SNR5_RECEIPTS.json") == receipts
            and quality.json("cache/snr5-roster.json") == snr5_rows, "independent entire 77-receipt and eligible-roster match")
    reported_summary = quality.json("evidence/SNR5_SUMMARY.json")
    require(all(reported_summary[key] == value for key, value in summary.items()), "independent mix summary")
    require(reported_summary["condition_training_allowed"] is False
            and reported_summary["independent_source_count_increment"] == 0, "new condition is evaluation-only and same source")
    for case_id, raw in expected_pcm.items():
        require(quality.read("audio/snr5/" + case_id + ".pcm") == raw, "independently reconstructed complete 5 dB PCM")
    require({name for name in quality.names if name.startswith("audio/")} ==
            {"audio/snr5/" + case_id + ".pcm" for case_id in expected_pcm}, "excluded clips have no replacement or hidden audio")
    snr5_clocks, new_embeddings = audit_condition_cache(quality.read("cache/snr5-embeddings.npz"), snr5_rows,
                                                       clean_clocks, "snr5")
    require(new_embeddings == summary["eligible_embedding_calls"], "exact one-pass new 5 dB cache")
    require(quality.json("evidence/FEATURE_GEOMETRY.json") == training.json("FEATURE_PREFLIGHT.json")["geometry"],
            "unchanged actual frontend support geometry")
    training_receipt = quality.json("evidence/TRAINING.json")
    previous_training = previous.json("evidence/TRAINING.json")
    draw_check = verify_draws(training_receipt["draws"], previous_training["draws"], roster, snr10_rows)
    losses = training_receipt["losses"]
    require(training_receipt["optimizer_updates"] == len(losses) == 96
            and training_receipt["training_bag_forwards"] == 576
            and all(type(x) is float and math.isfinite(x) and x >= 0 for x in losses), "96 finite training loss receipts")
    require(training_receipt["draws_exact_match"] is True
            and training_receipt["previous_training"] == ident(previous.read("evidence/TRAINING.json"))
            and training_receipt["development_checkpoint_or_threshold_selection"] is False
            and training_receipt["snr5_used_for_training"] is False
            and training_receipt["new_training_frontend_calls"] == 0
            and training_receipt["comparison"] == "ONLY_POSITIVE_TEMPORAL_POOLING_CHANGED_EQUAL_DATA_DRAWS_AND_UPDATE_BUDGET",
            "same-budget control comparison and no new evaluation-driven training")
    for key in ("positive_loss", "background_loss", "bag_aggregation"):
        require(training_receipt[key] == plan[key], "recorded exact training loss contract")
    features = unique_rows(training.json("FEATURE_EVIDENCE.json"), "feature evidence")
    expected_training_geometry = [{"condition": condition, "id": row["id"], "role": "train",
        "weak_class": row["weak_class"], "head_instance_ends": features[row["id"]]["head_instance_ends"],
        "triplet_count": len(features[row["id"]]["head_instance_ends"]) - 2 if row["weak_class"] else None}
        for condition, rows in (("clean", roster), ("snr10", snr10_rows)) for row in rows if row["role"] == "train"]
    require(quality.json("evidence/TRAINING_GEOMETRY.json") == expected_training_geometry,
            "all training bags use exact original support windows, never evaluation-only rows")
    weights, head_pins = {}, {}
    for label, filename in (("baseline", "baseline-head.npz"), ("control", "aug96-control-head.npz"),
                            ("candidate", "temporal-head.npz")):
        raw = quality.read("models/" + filename)
        head_pins[label], weights[label] = ident(raw), npz(raw)
        require(set(weights[label]) == set(SHAPES) and all(weights[label][k]["shape"] == shape
                and weights[label][k]["dtype"] == "<f4" for k, shape in SHAPES.items()), "unchanged head array types/shapes")
        require(all(v[0] > 0 for v in struct.iter_unpack("<f", weights[label]["scale96"]["payload"])),
                "positive frozen normalization scale")
    normalization = {}
    for key in ("mean96", "scale96"):
        raw = weights["baseline"][key]["payload"]
        require(raw == weights["control"][key]["payload"] == weights["candidate"][key]["payload"],
                "byte-identical original normalization in all heads")
        normalization[key] = {"bytes": len(raw), "sha256": digest(raw), "candidate_sha256": digest(raw), "byte_identical": True}
    require(training_receipt["normalization"] == normalization
            and training_receipt["warmstart"] == head_pins["baseline"]
            and training_receipt["control"] == head_pins["control"]
            and training_receipt["candidate"] == head_pins["candidate"], "training binds saved head and normalization bytes")
    require(any(weights["candidate"][key]["payload"] != weights["baseline"][key]["payload"]
                for key in ("w1", "b1", "w2", "b2")), "new candidate has changed trainable weights")
    binary = b"EHKWS01\0" + b"".join(weights["candidate"][key]["payload"] for key in SHAPES)
    require(quality.read("models/temporal-head.bin") == binary, "portable head export contains exact saved candidate arrays")
    frozen = quality.json("evidence/CANDIDATE_FROZEN.json")
    require(frozen["candidate"] == head_pins["candidate"] and frozen["candidate_bin"] == ident(binary)
            and frozen["normalization"] == normalization, "frozen receipt head/normalization identity")
    metrics, new_traces, reused_traces = {}, [], []
    jobs = (("temporal_clean", "temporal-clean", roster, clean_clocks),
            ("temporal_snr10", "temporal-snr10", snr10_rows, snr10_clocks),
            ("baseline_snr5", "baseline-snr5", snr5_rows, snr5_clocks),
            ("aug96_snr5", "aug96-snr5", snr5_rows, snr5_clocks),
            ("temporal_snr5", "temporal-snr5", snr5_rows, snr5_clocks))
    for label, name, rows, clocks in jobs:
        metrics[label], trace = read_trace(quality, name, rows, clocks)
        new_traces.append({"name": name, **trace})
    for label, name, rows, clocks in (("aug96_clean", "candidate-clean", roster, clean_clocks),
                                     ("aug96_snr10", "candidate-snr10", snr10_rows, snr10_clocks),
                                     ("baseline_snr10", "baseline-snr10", snr10_rows, snr10_clocks)):
        metrics[label], trace = read_trace(quality, "reused/" + name, rows, clocks)
        reused_traces.append({"name": name, **trace, "new_model_calls": 0})
    historical = unique_rows(training.json("PER_CASE_EVENTS.json"), "original baseline clean events")
    reported = unique_rows(quality.json("predictions/reused/baseline-clean-events-reused.json"), "reused baseline clean events")
    require(set(historical) == set(reported) == {r["id"] for r in roster}, "complete reused baseline-clean event roster")
    metrics["baseline_clean"] = []
    for row in roster:
        events = historical[row["id"]]["events"]
        require(reported[row["id"]]["events"] == events, "original baseline-clean events never replayed")
        observation = event_observation(row, events)
        compare_reported_event(row, observation, reported[row["id"]])
        metrics["baseline_clean"].append(observation)
    counts = quality.json("evidence/PROGRESS.json")
    expected_counts = {"new_embedding_calls_started": new_embeddings, "new_embedding_calls_completed": new_embeddings,
        "onnx_forward_started": 2 * new_embeddings, "onnx_forward_completed": 2 * new_embeddings,
        "head_case_passes_started": 193 + 3 * len(snr5_rows), "head_case_passes_completed": 193 + 3 * len(snr5_rows),
        "head_ready_outputs": sum(t["ready_outputs"] for t in new_traces),
        "optimizer_updates_started": 96, "optimizer_updates_completed": 96,
        "torch_forward_calls_started": 576, "torch_forward_calls_completed": 576,
        "new_training_frontend_calls": 0, "new_tts_asr_phone_calls": 0, "retry_count": 0}
    require(counts == expected_counts == result["new_counts"]
            and sum(t["case_passes"] for t in new_traces) == expected_counts["head_case_passes_completed"]
            and counts["head_case_passes_completed"] <= 424 and new_embeddings <= 2834 <= 4000,
            "all execution counts agree with complete saved cache/trace/draw records and budget")
    phase_check = verify_phases(quality.json("evidence/PHASES.json"), frozen, counts, head_pins["candidate"]["sha256"])
    paired = []
    for condition in ("clean", "snr10", "snr5"):
        paired.extend(compare_cases(metrics["aug96_" + condition], metrics["temporal_" + condition], condition))
    gate = strict_gate(paired, snr5_rows)
    expected_retention = expected_retention_record(gate, paired)
    require(quality.json("evidence/RESEARCH_RETENTION.json") == result["research_retention"] == expected_retention,
            "independent strict case-level acceptance/rejection gate")
    reported_pairs = quality.json("evidence/PAIRED_COMPARISONS.json")
    require(set(reported_pairs) == {"clean", "snr10", "snr5"}, "closed paired comparison conditions")
    safety_map = {"prefix_count": "prefix_events", "wrong_count": "wrong_keyword_after_clip_start",
                  "extra_count": "extra_same_keyword_events", "weak_negative_disagreement_count": "weak_negative_intent_events"}
    for condition in reported_pairs:
        expected_pairs = [row for row in paired if row["condition"] == condition]
        require([r["id"] for r in reported_pairs[condition]] == [r["id"] for r in expected_pairs], "paired report order")
        old = unique_rows(metrics["aug96_" + condition], "paired old")
        new = unique_rows(metrics["temporal_" + condition], "paired new")
        for report, row in zip(reported_pairs[condition], expected_pairs):
            wanted = {"id": row["id"], "role": row["role"], "weak_class": row["weak_class"],
                "previous_research_error": row["control_error"], "temporal_research_error": row["candidate_error"],
                "previously_correct_case_regressed": row["regressed"],
                "event_error_categories_increased": [safety_map[key] for key in safety_map if key in row["introduced_event_errors"]],
                "development_miss_rescued": row["role"] == "development" and row["weak_class"] > 0
                    and row["control_missing_in_observation"] and not row["candidate_error"],
                "previous_events": old[row["id"]]["events"], "temporal_events": new[row["id"]]["events"]}
            require(report == wanted, "independent paired comparison record")
    for label in ("aug96_clean", "aug96_snr10", "temporal_clean", "temporal_snr10", "baseline_snr5", "aug96_snr5", "temporal_snr5"):
        require(result[label] == grouped(metrics[label]), "independent result aggregate: " + label)
    require(result["baseline"] == head_pins["baseline"] and result["aug96_control"] == head_pins["control"]
            and result["temporal_candidate"] == head_pins["candidate"]
            and result["planned_snr5_conditions"] == 77 and result["eligible_snr5_conditions"] == len(snr5_rows)
            and result["snr5_dispositions"] == summary["dispositions"]
            and result["frozen_embedding_rows_reused"] == {"clean": 10354, "snr10": 2804}, "result byte/count bindings")
    for flag in ("same_source_condition_is_not_independent_audio", "cross_batch_voice_family_exposure",
                 "prior_development_patterns_informed_research_hypothesis", "baseline_and_aug96_old_condition_predictions_reused"):
        require(result[flag] is True, "required research limitation: " + flag)
    require(result["independent_far"] == result["independent_frr"] == result["new_head_c_numeric_parity"] == result["ssc305"] == "NOT_RUN"
            and result["word_end_latency"] == "NOT_MEASURED" and result["model_delivered_as_product"] is False,
            "no unsupported generalization, device, parity or latency claim")
    paired_groups = {}
    for row in paired:
        key = row["condition"] + "/" + row["role"] + "/" + str(row["weak_class"])
        item = paired_groups.setdefault(key, {"cases": 0, "rescued": 0, "regressed": 0, "new_event_error_cases": 0})
        item["cases"] += 1; item["rescued"] += int(row["rescued"]); item["regressed"] += int(row["regressed"])
        item["new_event_error_cases"] += bool(row["introduced_event_errors"])
    return {"schema": "independent-english-temporal-quality-audit-v1", "status": "SAVED_TEMPORAL_EVIDENCE_VERIFIED",
        "quality_artifact": ident(quality.raw), "source_training_zip": ident(training.raw), "source_incumbent_zip": ident(previous.raw),
        "run_id": result["run_id"], "head_sha": result["head_sha"], "supervisor_wall_seconds": supervisor["wall_seconds"],
        "new_model_calls_by_this_auditor": 0, "model_forward_numerics_recomputed": False,
        "head_logit_weight_correspondence_independently_recomputed": False,
        "worker_manifest_files": len(manifest), "source_contract": source_check,
        "source_copy_bindings": len(copied), "clean_embedding_rows_reused": 10354, "snr10_embedding_rows_reused": 2804,
        "positive_supported_bags": positive_bags, "training_draws": draw_check,
        "weights": head_pins, "normalization": normalization, "portable_export_bytes_verified": True,
        "phases": phase_check, "counts": counts, "snr5": summary, "snr5_receipts": receipts,
        "new_trace_checks": new_traces, "reused_trace_checks": reused_traces,
        "metrics": {name: grouped(rows, source_kind=True) for name, rows in metrics.items()},
        "paired_summary": paired_groups, "paired_cases": paired, "research_gate": gate,
        "original_eof_views": [{"condition": label, "id": row["id"], "role": row["role"], "weak_class": row["weak_class"],
            "no_matching_event_by_original_eof": row["original_eof_no_matching_event_censored"],
            "interpretation": "CENSORED_NOT_FRR", "word_end_latency": "NOT_MEASURED"}
            for label, rows in metrics.items() for row in rows if row["weak_class"]],
        "same_source_condition_is_not_independent_audio": True, "independent_source_count_increment": 0,
        "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN", "new_head_c_numeric_parity": "NOT_RUN",
        "ssc305": "NOT_RUN", "word_end_latency": "NOT_MEASURED", **dict.fromkeys(FALSE_FLAGS, False)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, help="completed temporal-quality artifact ZIP")
    parser.add_argument("--training", type=Path, required=True, help="pinned original training ZIP")
    parser.add_argument("--incumbent", type=Path, required=True, help="pinned completed 10 dB ZIP")
    parser.add_argument("--expected-run")
    parser.add_argument("--expected-head")
    parser.add_argument("--preflight", action="store_true", help="only audit original inputs and new PCM, zero model execution")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require(args.preflight != bool(args.quality), "choose exactly one of --preflight or --quality")
    training, previous = Archive(args.training.read_bytes()), Archive(args.incumbent.read_bytes())
    report = (input_preflight(training, previous) if args.preflight else
              audit(Archive(args.quality.read_bytes()), training, previous, args.expected_run, args.expected_head))
    payload = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                          "bytes": len(payload.encode()), "sha256": digest(payload.encode()),
                          "new_model_calls_by_this_auditor": 0}, sort_keys=True))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
