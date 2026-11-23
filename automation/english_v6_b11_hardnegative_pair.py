"""B11 one-factor hard-negative continuation, entered after independent source review."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b10_hardneg_paired_trainonce as b10

need = b10.need
SCHEMA = "english-v6-b11-matched-diverse-hardnegative-v1"
BATCH = "b11-diverse-train-supplement-v1"
PRIMARY = "5a51f8c33c6250ed1b33166833970e43b64d3339e4081ea669b3d332aae7d1a0"


def hard_rows(source, final_archive, expected_final_sha):
    """Use new real parent events on independently screened original TRAIN PCM only."""
    manifest_raw = (source / "MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_raw)
    need(b10.digest(final_archive.read_bytes()) == expected_final_sha, "supplement result ZIP SHA")
    with zipfile.ZipFile(final_archive) as z:
        labels = z.read("SCREEN.json")
        result = json.loads(z.read("score/RESULT.json"))
        need(b10.digest(labels) == result["screen_sha256"] and
             b10.digest(z.read("score/FRAME_LOGITS.npz")) == result["logits_sha256"],
             "independent source review and complete parent logits")
    screen = json.loads(labels)
    need(screen["batch"] == result["batch"] == manifest["batch"] == BATCH and
         screen["source_manifest_sha256"] == b10.digest(manifest_raw) and
         result["parent_pt_sha256"] == b10.b9.MODELS["B9_control"]["pt"] and
         result["optimizer_updates"] == screen["kws_forwards_at_admission"] == 0 and
         result["original_hard_groups"] == [] and result["promotion_allowed"] is False,
         "frozen source/decision/checkpoint binding")
    source_rows = {r["id"]: r for r in manifest["rows"]}
    approved = {}
    for r in screen["rows"]:
        raw = source_rows[r["id"]]
        need(all(r[k] == v for k, v in raw.items() if k != "observed_class")
             and r["role"] == "train" and r["requested_class"] == 0,
             "TRAIN identity and original waveform metadata")
        if r["tier"] == "unknown":
            need(r["observed_class"] is None, "unknown excluded")
        else:
            need(r["tier"] in ("silver", "weak") and r["observed_class"] == 0, "reviewed ordinary only")
            approved[r["id"]] = r
    scores = result["scored"]
    need(len(source_rows) == len(manifest["rows"]) == len(screen["rows"]) == 24 and
         {r["id"] for r in screen["rows"]} == set(source_rows) and
         len(scores) == len(approved) == result["originals"] and
         {x["id"] for x in scores} == set(approved) and 3 * len(scores) == result["kws_forwards"],
         "all 24 qualified or excluded, each scored original once at three gains")
    hard = []
    for x in scores:
        r = approved[x["id"]]
        need(x["pcm_sha256"] == r["pcm_sha256"] and x["group"] == r["source_group_id"] and
             x["tier"] == r["tier"], "parent scoring attached to original")
        if x["gains"]["1.0"]["events"] > 0:
            hard.append(dict(r, mono_class=0, source_pcm_sha256=r["pcm_sha256"]))
    families = sorted({r["source_group_id"] for r in hard})
    need(len({r["source_pcm_sha256"] for r in hard}) == len(hard) and len(families) >= 2 and families == result["combined_hard_groups"] == result["new_hard_groups"] and
         result["train_hardness_gate"] == "PASS", "B11 STOP: under two measured hard TRAIN families")
    return sorted(hard, key=lambda r: r["id"])


def freeze_draw(rows, steps):
    families = defaultdict(list)
    for row in rows:
        families[row["source_group_id"]].append(row)
    order = sorted(families)
    need(len(order) >= 2 and len({r["id"] for r in rows}) == len(rows), "two distinct hard families")
    for values in families.values():
        values.sort(key=lambda r: r["id"])
    draw = []
    for step in range(steps):
        group = order[step % len(order)]
        bucket = families[group]
        draw.append(bucket[(step // len(order)) % len(bucket)])
    return draw


def reviewed_primary(screen_raw, expected_sha, excluded_train):
    need(b10.digest(screen_raw) == expected_sha, "pinned primary QA metadata")
    labels = json.loads(screen_raw)
    rows = labels["rows"]
    need(labels["plan_sha256"] == PRIMARY and labels["frozen_before_kws"] is True and
         labels["optimizer_updates"] == labels["kws_forwards"] == 0 and
         len(rows) == len({r["id"] for r in rows}) == len({r["pcm_sha256"] for r in rows}) == 48 and
         all(r["role"] == "development" and r["panel"] == "primary" for r in rows) and
         not (set(excluded_train) & {r["source_group_id"] for r in rows}), "unexposed disjoint primary panel")
    assigned = defaultdict(set)
    for r in rows:
        if r["tier"] in ("silver", "weak"):
            need(r["observed_class"] in (0, 2) and r["observed_class"] == r["requested_class"],
                 "reviewed DEV class and requested role")
            assigned[r["source_group_id"]].add(r["observed_class"])
    paired = sorted(g for g,v in assigned.items() if v == {0,2})
    need(len(paired) >= 4 and paired == labels["paired_source_groups"] and labels["meets_four_pair_minimum"],
         "four independent DEV families with qualified positives and ordinary negatives")
    return paired


def original_training(inputs):
    from contextlib import ExitStack
    import english_v6_b11_train_hardness as negative
    import english_v6_b12_positive_train as positive

    with ExitStack() as stack:
        archives = {}
        for name, (_, expected) in negative.ARCHIVES.items():
            path = inputs / (name + ".zip")
            need(b10.digest(path.read_bytes()) == expected, "frozen TRAIN archive " + name)
            archives[name] = stack.enter_context(zipfile.ZipFile(path))
        parent_set = set(json.loads(archives["b9"].read("SOURCE.json"))["train_pcm_sha256"])
        positives = [dict(r, mono_class=1, archive=r["shard"]) for r in positive.select(archives)]
        negatives = [dict(r, mono_class=0) for r in negative.select_rows(archives)
                     if r["source_pcm_sha256"] in parent_set]
        rows = positives + negatives
        need(len(rows) == len(parent_set) == 79 and {r["source_pcm_sha256"] for r in rows} == parent_set,
             "exact B9 TRAIN inventory")
        audio = {r["id"]: negative.read_train_pcm(archives[r["archive"]], r) for r in rows}
    return rows, audio, b10.b6.b4.input_groups(rows)


def make_features(rows, waves, frontend, torch, np):
    mel, mean, scale = frontend
    cache = {}
    for r in rows:
        raw = waves[r["id"]]
        need(r["role"] == "train" and b10.digest(raw) == r["source_pcm_sha256"], "TRAIN waveform SHA")
        slots = {}
        for gain in b10.GAINS:
            gained = b10.b6.b4.gain_pcm(raw, gain, np)
            slots[gain] = {}
            for prefix in b10.PREFIXES:
                pcm = b"\0\0" * prefix + gained
                features = b10.b6.b4.v5.features(pcm, dict(r, samples=len(pcm)//2), torch, np, mel)
                values = ((features - mean) / scale).astype(np.float32)
                need(values.ndim == 2 and values.shape[1] == 40 and np.isfinite(values).all(),
                     "finite TRAIN 40-Mel features")
                slots[gain][prefix] = torch.from_numpy(values)[None]
        cache[r["id"]] = slots
    return cache


def execute(inputs, source, measured, primary_screen, hashes, output):
    """Input paths for archived TRAIN data and one frozen DEV metadata file (no DEV audio)."""
    import wave
    import io

    extra = hard_rows(source, measured, hashes["measured"])
    rows, waves, groups = original_training(inputs)
    screen_raw = primary_screen.read_bytes()
    need(json.loads(screen_raw)["source_manifest_sha256"] == hashes["primary_source_manifest_sha256"],
         "authentic primary source manifest")
    paired = reviewed_primary(screen_raw, hashes["primary_screen"], groups[0])
    need({r["source_group_id"] for r in extra} <= set(groups[0]) and
         not ({r["source_pcm_sha256"] for r in rows} & {r["source_pcm_sha256"] for r in extra}),
         "extra TRAIN examples only in previously qualified TRAIN families")
    draws = freeze_draw(extra, b10.STEPS)
    supplemental_waves = {}
    for r in extra:
        raw = (source / "wav" / (r["id"] + ".wav")).read_bytes()
        need(b10.digest(raw) == r["wav_sha256"], "exact full supplemental TRAIN WAV")
        with wave.open(io.BytesIO(raw)) as w:
            need((w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) ==
                 (1, 2, 16000, r["samples"]), "original TRAIN PCM format")
            supplemental_waves[r["id"]] = w.readframes(w.getnframes())
    slots = [{"step": i + 1, "group": r["source_group_id"], "id": r["id"],
              "source_pcm_sha256": r["source_pcm_sha256"],
              "prefix": b10.PREFIXES[(i // b10.TRAIN_GROUPS + i % 3) % len(b10.PREFIXES)]}
             for i, r in enumerate(draws)]
    for i, slot in enumerate(slots):
        for klass, key in ((0, "ordinary"), (1, "positive")):
            family = sorted(groups[klass])[i % b10.TRAIN_GROUPS]
            choices = sorted(groups[klass][family], key=lambda r: r["id"])
            drawn = choices[(i // b10.TRAIN_GROUPS) % len(choices)]
            slot[key] = {"id": drawn["id"], "group": family, "pcm_sha256": drawn["source_pcm_sha256"]}
    output.mkdir()
    b10.write(output / "SOURCE.json", {"schema": SCHEMA, "hard_train": extra,
              "old_train": [dict(id=r["id"], group=r["source_group_id"], sha256=r["source_pcm_sha256"])
                            for r in rows], "qualified_primary_groups_metadata_only": paired,
              "source_sha256": b10.digest((source/"MANIFEST.json").read_bytes()), "input_shas": hashes})
    b10.write(output / "COMMON_DRAW.json", slots)
    draw_sha = b10.digest((output / "COMMON_DRAW.json").read_bytes())

    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    weights, _ = b10.b9.verify_b9_archive(inputs / "b9.zip", torch, np)
    net, *frontend = weights["B9_control"]
    prior = {k: v.detach().clone() for k,v in net.state_dict().items()}
    base_cache = make_features(rows, waves, frontend, torch, np)
    extra_cache = make_features(extra, supplemental_waves, frontend, torch, np)
    jobs = {}
    states = {}
    for name, legacy in (("control", "control"), ("challenger", "hard_negative")):
        path = output / name
        path.mkdir()
        rec = b10.fit(legacy, prior, rows, groups, base_cache, draws, extra_cache,
                      *frontend, path, torch, np, F)
        for step in rec["trace"]:
            planned = slots[step["step"] - 1]
            need(step["prefix"] == planned["prefix"] and step["hard_negative_source_id"] ==
                 (planned["id"] if name == "challenger" else None), "actual shared step schedule")
        stem = "mero43-b11-" + name + "192"
        files = {}
        for ext in ("pt", "npz", "onnx"):
            old = path / "models" / (b10.MODEL_STEMS[legacy] + "." + ext)
            target = old.with_name(stem + "." + ext)
            old.rename(target)
            info = rec["files"][old.name]
            need(b10.digest(target.read_bytes()) == info["sha256"], "exact serialized model member")
            files[target.name] = info
        states[name] = torch.load(path/"models"/(stem+".pt"), map_location="cpu", weights_only=True)
        jobs[name] = {"optimizer_updates": rec["optimizer_updates"], "files": files,
                      "hard_negative_loss_weight": rec["hard_negative_loss_weight"], "trace": rec["trace"],
                      "shared_hard_draw_sha256": draw_sha, "stem": name + "/models/" + stem}
    distinct = sum((x != states["challenger"][k]).sum().item() for k,x in states["control"].items())
    need(distinct > 0 and len(slots) == b10.STEPS == 192, "paired model geometry and parameter movement")
    b10.write(output / "RESULT.json", {"schema": SCHEMA, **jobs,
              "parent_pt_sha256": b10.b9.MODELS["B9_control"]["pt"], "actual_optimizer_updates": 384,
              "trained_model_parameters": b10.PARAMS, "different_trained_float_values": distinct,
              "dev_cal_locked_forwards": 0, "promotion_allowed": False,
              "common_seed": b10.SEED, "common_learning_rate": b10.LR, "common_hard_draw_sha256": draw_sha})

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("measured", type=Path)
    parser.add_argument("primary_screen", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    need(os.environ.get("GITHUB_ACTIONS") == "true" and
         os.environ.get("GITHUB_REPOSITORY") == "jiying2007/kws-data" and
         os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "one public first-attempt paired run")
    hashes = {"measured":"5b8f2dfab1285eb9f544e538dd3508c2c1c9405df12c06f99ba03b61ca29ee8c",
              "primary_screen":os.environ["KWS_PRIMARY_SCREEN_SHA256"],
              "primary_review_zip":os.environ["KWS_PRIMARY_REVIEW_ZIP_SHA256"],
              "primary_review_artifact_id":int(os.environ["KWS_PRIMARY_REVIEW_ARTIFACT"]),
              "primary_review_run":45095073669,
              "primary_source_manifest_sha256":"590d79cb529978b50674c9d6070b934ebaf8b192aafaa92394d21881ccf14a6e",
              "supplement_source_zip":"47d3076b74ea5c6392e00b2bf110afcb91791e91a4a4b9e1fe78b0d2e2cf9bec"}
    execute(args.inputs, args.source, args.measured, args.primary_screen, hashes, args.output)


if __name__ == "__main__":
    main()
