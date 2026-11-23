"""Join five blind observations by original PCM, freeze labels, score admitted TRAIN only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import struct
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_hifi_phones_once as phones
import english_pocket12_observe as asr
import english_v6_b9_dev528_four_frozen_once as b9
from english_v6_b11_supplement import BATCH, PROVIDERS, read_source, decision, need, policy, record, pcm_tools


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen(source, evidence, output):
    rows = read_source(source)
    source_hash = sha(source / "MANIFEST.json")
    by_id = {r["id"]: r for r in rows}
    texts = {k: {} for k in by_id}
    posterior = {k: {} for k in by_id}
    coverage = {}
    for p in PROVIDERS:
        root = evidence / p
        report = json.loads((root / "RESULT.json").read_bytes())
        need(report["schema"] == BATCH and report["provider"] == p
             and report["source_manifest_sha256"] == source_hash
             and report["attempted"] == report["denominator"] == 24
             and len(report["rows"]) == 24 and report["target_conditioned"] is False
             and report["initialization_error"] is None, "provider / dataset binding")
        need({x["id"] for x in report["rows"]} == by_id.keys(), "complete distinct row IDs")
        model = json.loads((root / "MODEL.json").read_bytes())
        need(model["target_conditioned"] is False, "unprompted observer")
        if p in policy.TEXT_MODELS:
            need(model["observer"] == asr.identity(p), "locked ASR source")
        else:
            idx = policy.PHONE_MODELS.index(p)
            name, rev = pcm_tools.PHONE_MODELS[idx]
            need(model["model"] == name and model["revision"].startswith(rev)
                 and model["weights_sha256"] == phones.WEIGHT_SHA[idx], "locked phoneme source")
            voc = json.loads((root / "VOCAB.json").read_bytes())
            need(voc["model"] == name and voc["revision"] == model["revision"]
                 and voc["pad_token_id"] == 0, "native vocab lock")
        coverage[p] = 0
        for item in report["rows"]:
            sid = item["id"]
            d = root / "rows" / sid
            need(item["source_wav_sha256"] == by_id[sid]["wav_sha256"]
                 and item["source_pcm_sha256"] == by_id[sid]["pcm_sha256"]
                 and item["source_samples"] == by_id[sid]["samples"]
                 and item["provider"] == p and item["target_conditioned"] is False,
                 "complete original PCM join")
            if item["status"] == "failed":
                need(json.loads((d / "FAILED.json").read_bytes()) == item, "failure receipt")
                continue
            need(item["status"] == "success" and item["input_coverage_complete"] is True
                 and json.loads((d / "CAPTURE.json").read_bytes()) == item, "capture receipt")
            if p in policy.TEXT_MODELS:
                texts[sid][p] = item["raw_transcript"]
            else:
                raw = (root / "logits" / (sid + ".f32")).read_bytes()
                need(len(raw) == 4 * item["logit_frames"] * item["vocab_size"]
                     and item["vocab_size"] == 392, "all original logits")
                need(hashlib.sha256(raw).hexdigest() == item["logits_sha256"], "signed phone capture")
                policy.verified_phone_input(raw, voc["vocab"], pad_token_id=0)
                ids = [max(range(392), key=frame.__getitem__) for frame in struct.iter_unpack("<392f", raw)]
                need(phones.ctc(ids, voc["vocab"], 0) == item["raw_phone_tokens"], "saved CTC parity")
                posterior[sid][p] = {"logits": raw, "vocab": voc["vocab"],
                                     "sha256": item["logits_sha256"]}
            coverage[p] += 1
        need(coverage[p] == report["complete"], "observation count")
    qualified = []
    for row in rows:
        sid = row["id"]
        tier = decision(row, texts[sid], posterior[sid])
        qualified.append({**row, "tier": tier, "observed_class": 0 if tier in ("silver", "weak") else None})
    record(output, {"batch": BATCH, "source_manifest_sha256": source_hash, "coverage": coverage,
                    "rows": qualified, "kws_forwards_at_admission": 0, "human_gold": False})


def measure(source, admission, checkpoint, old, output):
    import numpy as np
    import torch
    original = {r["id"]: r for r in read_source(source)}
    labels = json.loads(admission.read_bytes())
    need(labels["source_manifest_sha256"] == sha(source / "MANIFEST.json")
         and len(labels["rows"]) == 24 and {r["id"] for r in labels["rows"]} == original.keys(),
         "freeze cohort and labels before KWS")
    base = json.loads(old.read_bytes())
    need(base["original_clips"] == 41 and base["source_groups"] == 35
         and base["actual_forwards"] == 123 and base["hard_group_gate_passed"] is False
         and base["parent_pt_sha256"] == b9.MODELS["B9_control"]["pt"],
         "supplement runs only after insufficient original TRAIN hard groups")
    output.mkdir()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    models, _ = b9.verify_b9_archive(checkpoint, torch, np)
    net, bank, mean, scale = models["B9_control"]
    scored, arrays = [], {}
    with torch.inference_mode():
        for r in labels["rows"]:
            need(all(r[k] == v for k, v in original[r["id"]].items() if k != "observed_class"),
                 "immutable TRAIN metadata")
            if r["tier"] == "unknown":
                need(r["observed_class"] is None, "unknown excluded")
                continue
            need(r["role"] == "train" and r["tier"] in ("silver", "weak")
                 and r["observed_class"] == 0, "TRAIN ordinary only")
            sid = r["id"]
            with wave.open(str(source / "wav" / (sid + ".wav")), "rb") as w:
                pcm = w.readframes(w.getnframes())
            gains = {}
            for gain in (1., .75, .5):
                audio = b9.ref.gain_pcm(pcm, gain, np)
                f = b9.v5.features(audio, r, torch, np, bank)
                x = torch.from_numpy(((f - mean) / scale).astype(np.float32)).unsqueeze(0)
                y = net(x)
                need(y.shape == (1, len(f), 4) and bool(torch.isfinite(y).all()), "finite aligned logits")
                dec = b9.v5.ctc_greedy(y, torch)
                key = f"{sid}__{gain}"
                arrays[key] = y[0].cpu().numpy()
                gains[str(gain)] = {"events": int(dec["mero_event"]) + int(dec["extra_events"]),
                                    "tokens": dec["collapsed_tokens"], "logits": key}
            scored.append({"id": sid, "pcm_sha256": r["pcm_sha256"], "group": r["source_group_id"],
                           "form": r["form"], "tier": r["tier"], "gains": gains})
    np.savez_compressed(output / "FRAME_LOGITS.npz", **arrays)
    new = sorted({x["group"] for x in scored if x["gains"]["1.0"]["events"] > 0})
    groups = sorted(set(new) | set(base["original_gain_hard_groups"]))
    result = {"batch": BATCH, "screen_sha256": sha(admission), "original_diagnostic_sha256": sha(old),
              "parent_pt_sha256": base["parent_pt_sha256"], "scored": scored, "originals": len(scored),
              "unknown": 24 - len(scored), "kws_forwards": len(arrays),
              "logits_sha256": sha(output / "FRAME_LOGITS.npz"), "new_hard_groups": new,
              "original_hard_groups": base["original_gain_hard_groups"], "combined_hard_groups": groups,
              "train_hardness_gate": "PASS" if len(groups) >= 2 else "STOP",
              "optimizer_updates": 0, "promotion_allowed": False}
    record(output / "RESULT.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "scored"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    for mode, names in (("screen", ("source", "evidence", "output")),
                        ("measure", ("source", "admission", "checkpoint", "old", "output"))):
        cmd = subs.add_parser(mode)
        for name in names:
            cmd.add_argument(name, type=Path)
    a = p.parse_args()
    kwargs = vars(a)
    func = screen if kwargs.pop("mode") == "screen" else measure
    func(**kwargs)
