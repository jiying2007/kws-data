"""Sealed independent DEV evaluation of B0, B9 and a matched research pair.

Each evaluation is bound to one cohort, source/screen/checkpoint ZIP and parent
diagnostic via the tracked run specification. Scoring never changes thresholds.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b11_dev_panel as panel
import english_v6_b9_dev528_four_frozen_once as b9
import english_v6_mero_frozen_b0_b1_paired_once as b0

GAINS = ("1.0", "0.75", "0.5")
NAMES = ("B0", "B9", "control", "challenger")
need = b9.need


def sha(buf):
    return hashlib.sha256(buf).hexdigest()


def opened(path, lock):
    need(path.is_file() and path.stat().st_size == lock["bytes"]
         and sha(path.read_bytes()) == lock["sha256"], "archive SHA and length")
    z = zipfile.ZipFile(path)
    need(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())), "archive CRC/unique names")
    return z


def snapshots(directory, spec, torch, np):
    blobs = {}
    with opened(directory / "B0.zip", spec["inputs"]["B0"]) as ar, tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # B0 helper reads these four named members, using the original file hashes.
        for name in ["RESULT.json"] + ["models/mero-fsmn-ctc-expanded256." + e for e in ("pt", "npz", "onnx")]:
            dest = root / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(ar.read(name))
        net, mel, mean, scale, hashes = b0.model(torch, np, root, "B0")
        blobs["B0"] = (net, mel, mean, scale, hashes["pt"])
    with opened(directory / "B9.zip", spec["inputs"]["B9"]):
        old, _ = b9.verify_b9_archive(directory / "B9.zip", torch, np)
    blobs["B9"] = (*old["B9_control"], b9.MODELS["B9_control"]["pt"])
    with opened(directory / "pair.zip", spec["inputs"]["pair"]) as ar:
        receipt = json.loads(ar.read("RESULT.json"))
        need(receipt["schema"] == spec["pair_schema"] and receipt["parent_pt_sha256"] == blobs["B9"][-1]
             and receipt["actual_optimizer_updates"] == 384 and receipt["trained_model_parameters"] == 43876
             and receipt["dev_cal_locked_forwards"] == 0 and receipt["promotion_allowed"] is False,
             "matched completed research training")
        for name, stem in spec["pair_stems"].items():
            need(name in ("challenger", "control") and receipt[name]["optimizer_updates"] == 192,
                 "real same-job arm and update count")
            members = {}
            for ext in ("pt", "npz", "onnx"):
                filename = stem + "." + ext
                raw = ar.read(filename)
                f = receipt[name]["files"][Path(filename).name]
                need(sha(raw) == f["sha256"] and len(raw) == f["bytes"], "candidate model member")
                members[ext] = raw
            weights = torch.load(io.BytesIO(members["pt"]), map_location="cpu", weights_only=True)
            net = b9.v5.build_model(torch)
            net.load_state_dict(weights, strict=True)
            net.eval()
            with np.load(io.BytesIO(members["npz"]), allow_pickle=False) as npz:
                need(set(npz.files) == set(weights) | {"mel_filters", "train_mean", "train_scale"}
                     and all(np.array_equal(npz[k], t.cpu().numpy()) for k, t in weights.items()),
                     "consistent PT and NPZ tensors")
                frontend = tuple(npz[k].copy() for k in ("mel_filters", "train_mean", "train_scale"))
                need(all(np.array_equal(x, y) for x, y in zip(frontend, old["B9_control"][1:])),
                     "identical frozen parent frontend")
            blobs[name] = (net, *frontend, sha(members["pt"]))
    need(set(blobs) == set(NAMES), "four different roles")
    return blobs


def decision(rows, hypothesis):
    def count(r, g, arm):
        return r["gains"][g][arm]["events"]
    rescued, new_false, lost, extra = set(), [], [], []
    positives = [r for r in rows if r["positive"]]
    negatives = [r for r in rows if not r["positive"]]
    for r in negatives:
        for g in GAINS:
            if any(count(r, g, "challenger") > count(r, g, m) for m in ("B9", "control")):
                new_false.append([r["id"], g])
    for r in positives:
        for g in GAINS:
            if not count(r, g, "challenger") and any(count(r, g, m) for m in ("B0", "control")):
                lost.append([r["id"], g])
            if any(max(0, count(r, g, "challenger")-1) > max(0, count(r, g, m)-1)
                   for m in ("control", "B9")):
                extra.append([r["id"], g])
    # A positive family is detected at gain g if any independently admitted
    # canonical positive in that family has a complete event. Preserve the
    # row-level losses separately so changes within a family remain visible.
    family_hits = {
        g: {m: {r["group"] for r in positives if count(r, g, m) > 0} for m in NAMES}
        for g in GAINS
    }
    lost_family_conditions = [[gid, g, m] for g in GAINS for m in ("B0", "control")
                              for gid in sorted(family_hits[g][m] - family_hits[g]["challenger"])]
    if hypothesis == "B11":
        positives_pass = all(len(family_hits[g]["challenger"]) >= len(family_hits[g]["B0"])
                             for g in GAINS)
        groups = {r["group"] for r in negatives}
        for gid in groups:
            group = [r for r in negatives if r["group"] == gid]
            totals = {m: sum(count(r, "1.0", m) for r in group) for m in NAMES}
            if totals["challenger"] < min(totals["B9"], totals["control"]):
                rescued.add(gid)
    elif hypothesis == "B12":
        positives_pass = not lost_family_conditions
        for r in positives:
            if (r["verified_faster"] and all(count(r, "1.0", m) for m in ("B9","control","challenger"))
                    and not any(count(r, "0.5", m) for m in ("B9", "control")) and count(r, "0.5", "challenger")):
                rescued.add(r["group"])
    else:
        raise ValueError("unregistered hypothesis")
    summary = {}
    for m in NAMES:
        summary[m] = {g: {"positive_hits": sum(count(r, g, m) > 0 for r in positives),
                          "positive_originals": len(positives),
                          "ordinary_false_clips": sum(count(r, g, m) > 0 for r in negatives),
                          "ordinary_events": sum(count(r, g, m) for r in negatives),
                          "ordinary_originals": len(negatives)} for g in GAINS}
    return {"summary": summary, "improved_groups": sorted(rescued), "new_ordinary_false_or_extra": new_false,
            "positive_group_hits": {g:{m:sorted(groups) for m,groups in hits.items()} for g,hits in family_hits.items()},
            "positive_gate_passed": positives_pass, "lost_positive_group_conditions": lost_family_conditions,
            "diagnostic_lost_positive_original_conditions": lost, "diagnostic_extra_positive_events": extra,
            "panel_gate": "PASS" if len(rescued) >= 2 and not new_false and positives_pass else "NO_PROMOTION"}


def infer(wav, row, nets, arrays, torch, np):
    with wave.open(io.BytesIO(wav)) as audio:
        need(audio.getparams()[:3] == (1,2,16000) and audio.getnframes() == row["samples"], "whole PCM format")
        pcm = audio.readframes(audio.getnframes())
    need(sha(wav) == row["wav_sha256"] and sha(pcm) == row["pcm_sha256"], "source PCM hash")
    conditions = {}
    with torch.inference_mode():
        for gain in GAINS:
            signal = b9.ref.gain_pcm(pcm, float(gain), np)
            conditions[gain] = {}
            for name in NAMES:
                net, mel, mean, scale, _ = nets[name]
                feats = b9.v5.features(signal, row, torch, np, mel)
                x = torch.from_numpy(((feats-mean)/scale).astype(np.float32)).unsqueeze(0)
                y = net(x)
                need(y.shape == (1, len(feats),4) and bool(torch.isfinite(y).all()), "finite 4-class logits")
                dec = b9.v5.ctc_greedy(y, torch)
                key = f"{row['id']}__{gain}__{name}"
                arrays[key] = y[0].detach().cpu().numpy().astype("<f4")
                events = int(dec["mero_event"]) + int(dec["extra_events"])
                # First complete event is the end-token onset in collapsed CTC.
                path = y.argmax(-1)[0].tolist()
                onsets = [(i, t) for i, t in enumerate(path) if t and (i == 0 or path[i-1] != t)]
                fire = [onsets[j+2][0] for j in range(max(0, len(onsets)-2))
                        if [t for _, t in onsets[j:j+3]] == [1,2,3]]
                need(len(fire) == events, "frame-clock parity")
                conditions[gain][name] = {"events": events, "tokens": dec["collapsed_tokens"], "frames": len(feats),
                                          "event_frames": fire, "array": key, "sha256": sha(arrays[key].tobytes())}
    return conditions


def select(source, review, panel_name):
    manifest_bytes = source.read("MANIFEST.json")
    manifest = json.loads(manifest_bytes)
    plan = panel.source_plan(panel_name)
    plan_sha = sha(panel.b10.canon(plan))
    labels = json.loads(review.read("SCREEN.json"))
    template = plan["rows"]
    originals = manifest["rows"]
    decisions = labels["rows"]
    need(manifest["schema"] == panel.SCHEMA and manifest["panel"] == panel_name
         and manifest["plan_sha256"] == plan_sha and manifest["tts_calls"] == len(originals) == len(template) == 48
         and manifest["new_kws_forwards"] == 0 and manifest["human_gold"] is False
         and labels["source_manifest_sha256"] == sha(manifest_bytes) and labels["plan_sha256"] == plan_sha
         and labels["frozen_before_kws"] is True and labels["kws_forwards"] == labels["optimizer_updates"] == 0
         and labels["human_gold"] is labels["promotion_allowed"] is False, "independent source/QA snapshot")
    by_id = {r["id"]: r for r in decisions}
    need(len(by_id) == len(decisions) == 48 and len({r["pcm_sha256"] for r in originals}) == 48,
         "complete independent original PCM roster")
    durations, classes = defaultdict(dict), defaultdict(set)
    selected = []
    for row, frozen in zip(originals, template, strict=True):
        need(all(row.get(k) == v for k, v in frozen.items()) and row["role"] == "development"
             and row["training_allowed"] is False and row["whole_native_resample"] is True and row["crop_calls"] == 0,
             "unchanged full-original source and role")
        verdict = by_id[row["id"]]
        need(all(verdict[k] == v for k,v in row.items() if k != "observed_class"),
             "unchanged PCM and role in screen")
        if row["cell_id"] in ("mero-fast", "mero-slow"):
            durations[row["source_group_id"]][row["cell_id"]] = row["samples"]
        if verdict["observed_class"] is None:
            need(verdict["tier"] == "unknown", "unknown excluded")
            continue
        need(verdict["tier"] in ("silver","weak")
             and verdict["observed_class"] == row["requested_class"]
             and row["requested_class"] in (0,2), "independent class admission")
        classes[row["source_group_id"]].add(verdict["observed_class"])
        selected.append(verdict)
    complete = sorted(g for g, v in classes.items() if v == {0,2})
    need(complete == labels["paired_source_groups"] and len(complete) >= 4 and labels["meets_four_pair_minimum"],
         "four complete independently admitted DEV families")
    faster = {g for g, cells in durations.items() if cells["mero-fast"] < cells["mero-slow"]}
    return selected, complete, faster


def run(spec, directory, output):
    import numpy as np
    import torch
    need(spec["hypothesis"] in ("B11","B12") and spec["panel"] in panel.GROUPS, "registered trial")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    output.mkdir(parents=True, exist_ok=False)
    with opened(directory / "source.zip", spec["inputs"]["source"]) as source, opened(
            directory / "screen.zip", spec["inputs"]["screen"]) as screen:
        rows, paired, faster = select(source, screen, spec["panel"])
        models = snapshots(directory, spec, torch, np)
        records, arrays = [], {}
        for row in rows:
            positive = row["observed_class"] == 2
            gid = row["source_group_id"]
            gains = infer(source.read("wav/"+row["id"]+".wav"), row, models, arrays, torch, np)
            records.append({"id": row["id"], "group": gid, "cell": row["cell_id"],
                            "positive": positive, "tier": row["tier"], "pcm_sha256": row["pcm_sha256"],
                            "verified_faster": positive and row["cell_id"] == "mero-fast" and gid in faster,
                            "gains": gains})
    result = decision(records, spec["hypothesis"])
    logfile = output/"ALL_FRAME_LOGITS.npz"
    np.savez_compressed(logfile, **arrays)
    need(len(arrays) == len(records) * 12, "four models, three fixed gains per original")
    result.update(schema="english-v6-4-frozen-panel-v1", trial=spec, records=records, admitted=len(records),
                  unknown_excluded=48-len(records), complete_source_groups=paired,
                  distinct_groups_scored=len({r["group"] for r in records}),
                  trained_pt_sha256={m: models[m][-1] for m in NAMES},
                  frame_clock={"sample_rate":16000, "hop":160, "fft":512, "center":False},
                  logits_sha256=sha(logfile.read_bytes()), actual_forwards=len(arrays),
                  optimizer_updates=0, threshold_search=False, physical_ssc305="NOT_RUN", product_promotion=False)
    (output/"RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records","trial")}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(json.loads(args.spec.read_bytes()), args.input, args.output)
