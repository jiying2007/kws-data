"""Measure ordinary TRAIN errors for the frozen B9 control before any B11 fit."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import wave
import zipfile


GAINS = (1.0, 0.75, 0.5)
MIN_HARD_GROUPS = 2
# Immutable public GitHub Actions artifact IDs and original ZIP digests.
ARCHIVES = {
    "b9": (14689218171, "1a8b270200e0c72ab1025f139d16abde43c209f142d72e71a7201b847e06a612"),
    "mono70-audit": (14503604626, "12a1eb39c7a458a466fec2e93ba59e5036319bd9db4b4bdc47e7e6e1c5990f1b"),
    "fresh24-source": (14422271973, "26396073f5be2f91e35575bedb586737405162388e32bfe85633ed3d796a9583"),
    "fresh18-source": (14429578824, "34cf7879f974f62c9f6046a72a964002f30bf59d2186a423ac382a76ff69d0e9"),
    "piper24-source": (14434542792, "d1f75716fefd954d59c3289c37e7e6670965ee98917436676896312e4b781aa1"),
    "newtrain24-source": (14486048125, "3db5e99caf33e04ae920df1c93205f38f98cbc0f326705e2367dc8c67dc02341"),
    "fresh-source": (14507208152, "2fbc26386136cfd12dddb01074f3c9c4556c8fc7c49819663f2d11a2499eaa1c"),
    "fresh-screen": (14509252921, "e8bcdf32a0ac138de8bdd90c4f6cd6e419d4ccae42bbd393e75b5daac9502158"),
    "b10-source": (14708400220, "dbc57d762922e60ff23253af2463d5d64e1c8021c74fad6e2f0a9ecb9fd87465"),
    "b10-screen": (14740257228, "ff4d647a224f430e50a76c1f7b42854ad9b41324db0f6b2f2823aaa4c63adb12"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def select_rows(archives):
    old = json.loads(archives["mono70-audit"].read(
        "english-v5-mero-mono-audit-44470829024.json"))["source_role_rows"]
    parent = set(json.loads(archives["b9"].read("SOURCE.json"))["train_pcm_sha256"])
    require(len(parent) == 79, "expected 79 original B9 TRAIN hashes")
    rows = []
    for row in old:
        if row["role"] == "train" and row["mono_class"] == 0:
            require(row["source_pcm_sha256"] in parent and row["weak_class"] == 0
                    and row["machine_reason"] == "BLIND_ORDINARY_NO_WAKE",
                    "prior cohort missing independent ordinary admission")
            rows.append(dict(row, archive=row["corpus"] + "-source",
                             wav_sha256=row["source_wav_sha256"]))
    for prefix in ("fresh", "b10"):
        reviewed = json.loads(archives[prefix + "-screen"].read("SCREEN.json"))["rows"]
        for row in reviewed:
            if row["role"] != "train" or row["observed_class"] != 0:
                continue
            if prefix == "fresh" and row["pcm_sha256"] not in parent:
                continue
            require(row["reason"] == "BLIND_ORDINARY_NO_WAKE" and row["requested_class"] == 0,
                    "qualified ordinary TRAIN required")
            rows.append(dict(row, archive=prefix + "-source",
                             source_pcm_sha256=row["pcm_sha256"]))
    require(len(rows) == len({r["id"] for r in rows})
            == len({r["source_pcm_sha256"] for r in rows}) == 41,
            "41 distinct ordinary source clips required")
    require(len({r["source_group_id"] for r in rows}) == 35
            and sum(r["source_pcm_sha256"] in parent for r in rows) == 35,
            "35 old clips and six new B10 clips in the frozen TRAIN source groups")
    require(all(r["tier"] in ("silver", "weak") and r["human_gold"] is False
                and r["requested_class"] == 0 for r in rows), "machine-reviewed ordinary only")
    return sorted(rows, key=lambda r: r["id"])


def read_train_pcm(archive, row):
    require(row["role"] == "train" and re.fullmatch(r"[\w.-]+", row["id"]),
            "read approved TRAIN path only")
    raw = archive.read("wav/" + row["id"] + ".wav")
    require(digest(raw) == row["wav_sha256"], "WAV hash mismatch")
    with wave.open(io.BytesIO(raw), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(),
                 wav.getcomptype()) == (1, 2, 16000, row["samples"], "NONE"), "PCM format")
        pcm = wav.readframes(wav.getnframes())
    require(digest(pcm) == row["source_pcm_sha256"], "PCM hash mismatch")
    return pcm


def run(inputs, output, audit_only):
    with ExitStack() as stack:
        archives = {}
        for name, (_, expected) in ARCHIVES.items():
            path = inputs / (name + ".zip")
            require(digest(path.read_bytes()) == expected, "archive SHA mismatch: " + name)
            archives[name] = stack.enter_context(zipfile.ZipFile(path))
        rows = select_rows(archives)
        clips = [(r, read_train_pcm(archives[r["archive"]], r)) for r in rows]
    if audit_only:
        print(f"authenticated {len(clips)} original TRAIN WAVs; zero forwards")
        return

    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("frozen_b9",
        Path(__file__).with_name("english_v6_b9_dev528_four_frozen_once.py"))
    b9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b9)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    models, _ = b9.verify_b9_archive(inputs / "b9.zip", torch, np)
    model, bank, mean, scale = models["B9_control"]
    output.mkdir()
    records, arrays = [], {}
    with torch.inference_mode():
        for row, pcm in clips:
            scores = {}
            for gain in GAINS:
                scaled = b9.ref.gain_pcm(pcm, gain, np)
                features = b9.v5.features(scaled, row, torch, np, bank)
                tensor = torch.from_numpy(((features - mean) / scale).astype(np.float32))[None]
                logits = model(tensor)
                require(logits.shape == (1, len(features), 4) and torch.isfinite(logits).all(),
                        "invalid frozen frame logits")
                decoded = b9.v5.ctc_greedy(logits, torch)
                key = row["id"] + "__" + str(gain)
                arrays[key] = logits[0].cpu().numpy()
                scores[str(gain)] = dict(events=int(decoded["mero_event"]) + int(decoded["extra_events"]),
                                         tokens=decoded["collapsed_tokens"], logit_array=key)
            records.append(dict(id=row["id"], group=row["source_group_id"],
                                pcm_sha256=row["source_pcm_sha256"], tier=row["tier"], gains=scores))
    hard_groups = sorted({r["group"] for r in records if r["gains"]["1.0"]["events"] > 0})
    counts = {str(g): sum(r["gains"][str(g)]["events"] > 0 for r in records) for g in GAINS}
    logit_path = output / "TRAIN_FRAME_LOGITS.npz"
    np.savez_compressed(logit_path, **arrays)
    result = dict(schema="english-v6-b11-frozen-train-hardness-v1",
                  run_id=os.getenv("GITHUB_RUN_ID"), git_sha=os.getenv("GITHUB_SHA"),
                  parent_pt_sha256=b9.MODELS["B9_control"]["pt"],
                  original_clips=len(records), source_groups=35, actual_forwards=len(arrays),
                  false_wake_clips_by_gain=counts, original_gain_hard_groups=hard_groups,
                  minimum_hard_groups=MIN_HARD_GROUPS,
                  hard_group_gate_passed=len(hard_groups) >= MIN_HARD_GROUPS,
                  optimizer_updates=0, dev_cal_locked_forwards=0, new_audio_calls=0,
                  promotion_allowed=False, frame_logits_sha256=digest(logit_path.read_bytes()),
                  input_archives=ARCHIVES, records=records)
    (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "input_archives")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-inputs", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.list_inputs:
        for key, (artifact, _) in ARCHIVES.items():
            print(key, artifact)
    else:
        run(args.inputs, args.output, args.audit_only)
