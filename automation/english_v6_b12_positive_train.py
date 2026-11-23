"""Frozen B9 gain sensitivity on its 44 independently admitted positive TRAIN originals."""
import argparse
from collections import Counter
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import zipfile

GAINS = (1.0, 0.75, 0.5)
HELPER_HEAD = "9b4177a940fc75f8988832c5832cf49085873e28"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def require(ok, message):
    if not ok:
        raise ValueError(message)


def delivery(source, manifest):
    cell = source["cell_id"]
    record = {"requested_cell": cell, "samples": source["samples"], "verified_faster": False}
    if cell not in {"mero-fast", "mero-slow"}:
        return record
    other_cell = "mero-slow" if cell == "mero-fast" else "mero-fast"
    peers = [x for x in manifest if x["cell_id"] == other_cell and x["role"] == "train"
             and x["source_group_id"] == source["source_group_id"]
             and x["source_provider"] == source["source_provider"]]
    require(len(peers) == 1, "unique same-source TRAIN tempo counterpart required")
    peer = peers[0]
    text_field = "requested_text" if "requested_text" in source else "generation_render_text"
    require(source[text_field] == peer[text_field], "different prompt in tempo pair")
    if source["source_provider"] == "official-piper-libritts-904":
        field = "generation_length_scale"
        faster = source[field] < 1 < peer[field]
    else:
        require(source["source_provider"] == "official-kokoro-v1-direct-phonemes", "unknown rate semantics")
        field = "speed"
        faster = source[field] > 1 > peer[field]
    record.update(rate_field=field, value=source[field], paired_value=peer[field],
                  paired_id=peer["id"], paired_samples=peer["samples"],
                  verified_faster=faster and source["samples"] < peer["samples"])
    return record


def select(archives):
    base = json.loads(archives["mono70-audit"].read(
        "english-v5-mero-mono-audit-44470829024.json"))["source_role_rows"]
    parent = set(json.loads(archives["b9"].read("SOURCE.json"))["train_pcm_sha256"])
    require(len(parent) == 79, "frozen B9 original TRAIN hashes")
    admit = []
    for row in base:
        if row["role"] == "train" and row["mono_class"] == 1:
            require(row["weak_class"] == 2 and row["ctc_targets"] == [1, 2, 3], "prior positive admission")
            admit.append(dict(row, shard=row["corpus"] + "-source", wav_sha256=row["source_wav_sha256"],
                              reason=row["machine_reason"]))
    for row in json.loads(archives["fresh-screen"].read("SCREEN.json"))["rows"]:
        if row["role"] == "train" and row["observed_class"] == 2 and row["pcm_sha256"] in parent:
            admit.append(dict(row, shard="fresh-source", source_pcm_sha256=row["pcm_sha256"]))
    require(len(admit) == len({r["id"] for r in admit}) == len({r["source_pcm_sha256"] for r in admit}) == 44,
            "all 44 frozen positive originals required")
    require(len({r["source_group_id"] for r in admit}) == 35, "all 35 original TRAIN source groups required")
    manifests = {key: json.loads(archives[key].read("MANIFEST.json"))["rows"]
                 for key in {r["shard"] for r in admit}}
    for row in admit:
        require(row["source_pcm_sha256"] in parent and row["tier"] in ("silver", "weak")
                and row["requested_class"] == 2 and row["human_gold"] is False
                and row["reason"] == "TWO_PHONE_CTC_MARGINS_PLUS_BLIND_ASR", "source admission")
        manifest = manifests[row["shard"]]
        matches = [x for x in manifest if x["id"] == row["id"]]
        require(len(matches) == 1, "source ID")
        src = matches[0]
        require(src["pcm_sha256"] == row["source_pcm_sha256"] and
                all(src[k] == row[k] for k in ("role", "samples", "source_group_id", "wav_sha256", "requested_class")),
                "manifest identity mismatch")
        row["delivery"] = delivery(src, manifest)
    return sorted(admit, key=lambda r: r["id"])


def execute(args):
    helper = load(args.helpers / "automation/english_v6_b11_train_hardness.py")
    pins = {k: v for k, v in helper.ARCHIVES.items() if not k.startswith("b10")}
    if args.list_inputs:
        for key, (aid, _) in pins.items():
            print(key, aid)
        return
    with ExitStack() as stack:
        archives = {}
        for key, (_, sha) in pins.items():
            path = args.inputs / (key + ".zip")
            require(helper.digest(path.read_bytes()) == sha, "ZIP SHA256 mismatch: " + key)
            archives[key] = stack.enter_context(zipfile.ZipFile(path))
        rows = select(archives)
        clips = [(r, helper.read_train_pcm(archives[r["shard"]], r)) for r in rows]
    if args.audit_only:
        print("Bound 44 TRAIN positives in 35 groups",
              Counter((r["delivery"]["requested_cell"], r["delivery"]["verified_faster"]) for r in rows))
        return

    import numpy as np
    import torch

    args.output.mkdir()
    frozen = load(args.helpers / "automation/english_v6_b9_dev528_four_frozen_once.py")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    nets, _ = frozen.verify_b9_archive(args.inputs / "b9.zip", torch, np)
    model, bank, mean, scale = nets["B9_control"]
    arrays, records = {}, []
    with torch.inference_mode():
        for row, pcm in clips:
            scores = {}
            for gain in GAINS:
                audio = frozen.ref.gain_pcm(pcm, gain, np)
                features = frozen.v5.features(audio, row, torch, np, bank)
                inputs = torch.from_numpy(((features - mean) / scale).astype(np.float32))[None]
                logits = model(inputs)
                arr = logits.squeeze(0).cpu().numpy()
                require(arr.shape == (1 + (row["samples"] - 512) // 160, 4)
                        and np.isfinite(arr).all(), "logit clock or finiteness")
                decoded = frozen.v5.ctc_greedy(logits, torch)
                key = f"{row['id']}__gain{gain}"
                arrays[key] = arr
                scores[str(gain)] = {"events": int(decoded["mero_event"]) + int(decoded["extra_events"]),
                                     "tokens": decoded["collapsed_tokens"], "array": key,
                                     "frames": len(arr), "logits_sha256": helper.digest(arr.astype("<f4").tobytes())}
            records.append({"id": row["id"], "group": row["source_group_id"], "tier": row["tier"],
                            "pcm_sha256": row["source_pcm_sha256"], "wav_sha256": row["wav_sha256"],
                            "delivery": row["delivery"], "gains": scores})
    require(len(arrays) == 132, "44 x 3 actual forward outputs required")
    failed = [r for r in records if r["gains"]["1.0"]["events"] and not r["gains"]["0.5"]["events"]]
    fast_fail = [r for r in failed if r["delivery"]["verified_faster"]]
    fast_groups = sorted({r["group"] for r in fast_fail})
    hei_groups = sorted({r["group"] for r in fast_fail if 1 not in r["gains"]["0.5"]["tokens"]})
    output = args.output / "TRAIN_POSITIVE_FRAME_LOGITS.npz"
    np.savez_compressed(output, **arrays)
    report = {"schema": "english-v6-b12-frozen-train-positive-diagnostic-v1", "run": os.environ["GITHUB_RUN_ID"],
              "head": os.environ["SCORE_HEAD"], "helpers_head": HELPER_HEAD, "archives": pins,
              "model_pt_sha256": frozen.MODELS["B9_control"]["pt"], "originals": len(records),
              "source_groups": 35, "actual_forwards": len(arrays), "optimizer_updates": 0,
              "new_tts_asr_calls": 0, "dev_cal_locked_forwards": 0, "promotion_allowed": False,
              "positives_hit_by_gain": {str(g): sum(r["gains"][str(g)]["events"] > 0 for r in records) for g in GAINS},
              "verified_faster_clips": sum(r["delivery"]["verified_faster"] for r in records),
              "clean_hit_half_miss_groups": sorted({r["group"] for r in failed}),
              "verified_faster_clean_hit_half_miss_groups": fast_groups,
              "verified_faster_clean_hit_half_miss_hey_absent_groups": hei_groups,
              "two_group_trigger": len(fast_groups) >= 2,
              "frame_clock": {"sample_rate": 16000, "fft": 512, "hop": 160, "center": False},
              "logits_npz_sha256": helper.digest(output.read_bytes()), "records": records}
    (args.output / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({k: v for k, v in report.items() if k not in ("records", "archives")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helpers", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-inputs", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    execute(parser.parse_args())
