"""Frozen B11 P1 diagnostic on the six already exposed DEV528 recordings."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

ARCHIVES = {
    # ID: (original successful run, byte length, SHA256)
    14693632800: (44870282676, 1206914, "d606ee883a1abd38cc95a278f6b7b7064476c8b00add1163e27c247c41ab7032"),
    14694540474: (44871646677, 8508, "ae3062102541dbfc6f18a3d8db6a170fad529b0e97095b31bbc0ee6b8b6d026b"),
    14689218171: (44860855857, 1001753, "1a8b270200e0c72ab1025f139d16abde43c209f142d72e71a7201b847e06a612"),
    14740303244: (44987190871, 1000738, "3c7247e301aa57ef442b116ad28dc3356e80b88942cea687fcebb8af8270b33d"),
    14697439433: (44881040200, 94305, "87b378c67a4aaf161c948cf7a257304ea519ab9e8ea0151e6f0ca7075386f120"),
}
COHORT = {
    "530-mero-slow": ("piper-libritts-p1513", 1),
    "532-mero-fast": ("piper-libritts-p664", 1),
    "534-mero-slow": ("piper-libritts-p5154", 1),
    "536-mero-fast": ("piper-libritts-p6308", 1),
    "535-ordinary-time": ("piper-libritts-p6288", 0),
    "537-ordinary-time": ("piper-libritts-p4731", 0),
}
GAINS = (1.0, 0.75, 0.5)
DATA_COMMIT = "a180e7e6fc20a30618a76b1a1f4492e639cd0f9b"
PLAN_COMMIT = "3a439ac786822e58aa7a98b6cbb54d44b328f5e4"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, document):
    with path.open("x") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def audit_archives(folder):
    receipts = {}
    for artifact, (run, size, digest) in ARCHIVES.items():
        raw = (folder / f"{artifact}.zip").read_bytes()
        require(len(raw) == size and sha(raw) == digest, f"artifact mismatch: {artifact}")
        with ZipFile(io.BytesIO(raw)) as archive:
            require(archive.testzip() is None, f"CRC error: {artifact}")
            require(len(set(archive.namelist())) == len(archive.namelist()), "duplicate member")
            members = {name: sha(archive.read(name)) for name in archive.namelist()}
        receipts[str(artifact)] = {"run_id": run, "zip_bytes": size, "zip_sha256": digest,
                                   "members_sha256": members}
    return receipts


def extract_sources(folder):
    for aid, name in [(14693632800, "source"), (14694540474, "screen")]:
        target = folder / name
        target.mkdir()
        with ZipFile(folder / f"{aid}.zip") as archive:
            for member in archive.namelist():
                require(not Path(member).is_absolute() and ".." not in Path(member).parts, "unsafe ZIP")
            archive.extractall(target)


def run(args):
    import numpy as np
    import torch

    args.output.mkdir()
    receipts = audit_archives(args.inputs)
    write_json(args.output / "INPUT_ARCHIVES.json", receipts)
    extract_sources(args.inputs)
    spec = importlib.util.spec_from_file_location("frozen_b10",
        args.data_repo / "automation" / "english_v6_b10_frozen_dev540_once.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    b9 = frozen.b9
    eligible, _ = b9.select(args.inputs / "source", args.inputs / "screen")
    fixed = {"merov6b9dev12-" + key: value for key, value in COHORT.items()}
    selected = [r for r in eligible if r["id"] in fixed]
    require(len(selected) == len(fixed) == len({r["source_group_id"] for r in selected}), "cohort size")
    for row in selected:
        require((row["source_group_id"], row["mono_class"]) == fixed[row["id"]]
                and row["role"] == "development", "source identity")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    parents, _ = b9.verify_b9_archive(args.inputs / "14689218171.zip", torch, np)
    parent = parents["B9_control"]
    bank, mean, scale = parent[1:]
    models = {"B9_parent": parent, **frozen.verify_models(
        args.inputs / "14740303244.zip", torch, np, bank, mean, scale)}
    with ZipFile(args.inputs / "14697439433.zip") as archive:
        historical = {x["id"]: x for x in json.loads(archive.read("RESULT.json"))["per_original"]}
        with np.load(io.BytesIO(archive.read("ORIGINAL_FRAME_LOGITS.npz")), allow_pickle=False) as npz:
            reference = {row["id"]: npz[row["id"] + "__B9_control"] for row in selected}
    arrays = {}
    sources = []
    with torch.inference_mode():
        for row in selected:
            sid = row["id"]
            pcm = b9.ref.source_pcm(row, args.inputs / "source")
            record = {"id": sid, "group": row["source_group_id"], "tier": row["machine_tier"],
                      "positive": bool(row["mono_class"]), "wav_sha256": row["wav_sha256"],
                      "pcm_sha256": sha(pcm), "samples": len(pcm) // 2, "conditions": {}}
            for gain in GAINS:
                audio = b9.ref.gain_pcm(pcm, gain, np)
                features = b9.v5.features(audio, row, torch, np, bank)
                inputs = torch.from_numpy(((features - mean) / scale).astype(np.float32)).unsqueeze(0)
                condition = {}
                for label, (model, *_) in models.items():
                    logits = model(inputs)
                    array = logits.squeeze(0).cpu().numpy()
                    frame_count = 1 + (record["samples"] - 512) // 160
                    require(array.shape == (frame_count, 4) and np.isfinite(array).all(), "frame geometry")
                    decoded = b9.v5.ctc_greedy(logits, torch)
                    value = int(decoded["mero_event"]) + int(decoded["extra_events"])
                    key = f"{sid}__{label}__gain{gain}"
                    arrays[key] = array
                    path = array.argmax(axis=1).tolist()
                    onsets = [[i, token] for i, token in enumerate(path)
                              if token and (i == 0 or path[i-1] != token)]
                    condition[label] = {"events": value, "tokens": decoded["collapsed_tokens"],
                                        "token_onsets_frame0": onsets, "array_key": key,
                                        "frames": frame_count, "logits_sha256": sha(array.astype("<f4").tobytes())}
                    if label == "B9_parent":
                        old = historical[sid]["conditions"][str(gain)]["B9_control"]
                        require(decoded["collapsed_tokens"] == old["collapsed_tokens"]
                                and value == old["full_phrase_event_count"], "historical B9 CTC mismatch")
                        if gain == 1.0:
                            error = float(np.max(np.abs(reference[sid] - array)))
                            require(error < 5e-5, f"historical B9 logits mismatch: {error}")
                            condition[label]["reference_max_abs_logit_error"] = error
                record["conditions"][str(gain)] = condition
            sources.append(record)
    require(len(arrays) == 54, "6 originals x 3 gains x 3 frozen models")
    archive = args.output / "ALL_FRAME_LOGITS.npz"
    np.savez_compressed(archive, **arrays)
    report = {"schema": "english-v6-b11-exposed-transfer-v1", "status": "EXPOSED_DIAGNOSTIC_COMPLETE",
              "run": os.environ["GITHUB_RUN_ID"], "head": os.environ["SCORE_HEAD"],
              "data_helpers_commit": DATA_COMMIT, "preregistration_commit": PLAN_COMMIT,
              "model_pt_sha256": {"B9_parent": frozen.B9_PARENT_SHA, **frozen.B10_PTS},
              "frame_clock": {"sample_rate_hz": 16000, "hop_samples": 160, "fft_samples": 512,
                              "hann_window_samples": 400, "center": False,
                              "fft_start": "160*frame0", "fft_end_exclusive": "160*frame0+512"},
              "independent_originals": 6, "fresh_evaluation_originals": 0, "frozen_forward_count": len(arrays),
              "optimizer_updates": 0, "new_tts_asr_phone_calls": 0, "cal_locked_opened": False,
              "model_selection_allowed": False, "promotion_allowed": False,
              "logits_npz_sha256": sha(archive.read_bytes()), "torch": torch.__version__, "numpy": np.__version__,
              "per_original": sources}
    write_json(args.output / "RESULT.json", report)
    for s in sources:
        print(s["id"], {g: {m: v["events"] for m, v in c.items()} for g, c in s["conditions"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in ("inputs", "output", "data-repo"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
