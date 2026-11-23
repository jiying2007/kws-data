"""Frozen B12 paired TRAIN experiment, gated by completed B11 and source receipts."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import zipfile

PLAN_SHA = "dfbeca5982400a11d7aab4da2c095a9524fafbc7b3bc340b9ac6e6811385b381"
B10_PAIR = (14740303244, "3c7247e301aa57ef442b116ad28dc3356e80b88942cea687fcebb8af8270b33d")
ORDINARY_DIAG = (14782043774, "e0e029b80fd5ec57d38f9ff588450217432a1132e2bfccd431c87b06689753dd")
POSITIVE_DIAG = (14782234724, "59e4f39122debbcb3151977a5f6481ba59cb5766a826840a5abaa8653e215bc3")
SCHEMA = "english-v6-b12-paired-tempo-v1"
N = 192
TEMPO_ARM = "tempo_challenger"
STEMS = {"matched_control": "mero43-b12-matched-control192", TEMPO_ARM: "mero43-b12-tempo108-192"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_training_rows(archives, ordinary, positive):
    parent = json.loads(archives["b9"].read("SOURCE.json"))
    ids = set(parent["train_pcm_sha256"])
    combined = [(dict(r, archive=r["shard"]), 1) for r in positive.select(archives)]
    combined += [(r, 0) for r in ordinary.select_rows(archives) if r["source_pcm_sha256"] in ids]
    require(len(ids) == len(combined) == 79 and {r["source_pcm_sha256"] for r, _ in combined} == ids,
            "exact B9 training hashes required")
    rows, groups = {}, {0: defaultdict(list), 1: defaultdict(list)}
    for original, cls in combined:
        row = dict(original, mono_class=cls)
        require(row["id"] not in rows and row["role"] == "train", "duplicate or ineligible TRAIN row")
        rows[row["id"]] = (row, ordinary.read_train_pcm(archives[row["archive"]], row))
        groups[cls][row["source_group_id"]].append(row["id"])
    require(set(groups[0]) == set(groups[1]) and len(groups[0]) == 35, "source-group geometry")
    return rows, groups


def tempo_wave(binary, raw):
    fmt = ["-t", "raw", "-r", "16000", "-e", "signed-integer", "-b", "16", "-c", "1", "-L", "-"]
    command = [binary, "-R", "-D", *fmt, *fmt, "tempo", "-s", "1.08"]
    warped = subprocess.run(command, input=raw, capture_output=True, timeout=15, check=True).stdout
    require(len(warped) % 2 == 0 and 0.90 * len(raw) < len(warped) < 0.96 * len(raw), "SoX tempo geometry")
    return warped


def schedule(groups):
    trace = []
    order = {cls: sorted(by_group) for cls, by_group in groups.items()}
    for step in range(N):
        draws = []
        for cls in (0, 1):
            group = order[cls][step % 35]
            members = sorted(groups[cls][group])
            sid = members[(step // 35) % len(members)]
            draws.append({"class": cls, "group": group, "id": sid})
        trace.append({"step": step + 1, "draws": draws, "prefix": (6400, 12800, 19200)[(step // 35 + step % 3) % 3],
                      "tempo_in_challenger": bool(step // 35 % 2)})
    return trace


def prepare_cache(rows, b10, bank, mean, scale, sox, torch, np):
    cache, manifest = {}, []
    for sid, (r, original) in sorted(rows.items()):
        faster = tempo_wave(sox, original)
        manifest.append({"id": sid, "source_group_id": r["source_group_id"], "role": r["role"], "tier": r["tier"],
                         "class": r["mono_class"], "wav_sha256": r["wav_sha256"], "pcm_sha256": sha(original),
                         "samples": len(original) // 2, "tempo_pcm_sha256": sha(faster), "tempo_samples": len(faster) // 2})
        for transformed, pcm in ((False, original), (True, faster)):
            for gain in b10.GAINS:
                gained = b10.b6.b4.gain_pcm(pcm, gain, np)
                for prefix in b10.PREFIXES:
                    full = b"\x00\x00" * prefix + gained
                    feats = b10.b9.v5.features(full, dict(r, samples=len(full) // 2), torch, np, bank)
                    cache[sid, transformed, gain, prefix] = torch.from_numpy(((feats - mean) / scale).astype(np.float32))[None]
    return cache, manifest


def export(model, label, output, frontend, example, architecture, torch, np):
    import onnx
    from onnx.reference import ReferenceEvaluator

    directory = output / label / "models"
    directory.mkdir(parents=True)
    stem = STEMS[label]
    model.eval()
    pt, npz, onx = [directory / (stem + suffix) for suffix in (".pt", ".npz", ".onnx")]
    torch.save(model.state_dict(), pt)
    state = torch.load(pt, map_location="cpu", weights_only=True)
    restored = architecture(torch)
    restored.load_state_dict(state, strict=True)
    restored.eval()
    require(sum(x.numel() for x in restored.parameters()) == 43876, "model parameters")
    bank, mean, scale = frontend
    np.savez(npz, mel_filters=bank, train_mean=mean, train_scale=scale, **{k: v.numpy() for k, v in state.items()})
    with np.load(npz, allow_pickle=False) as saved:
        require(np.array_equal(saved["mel_filters"], bank) and np.array_equal(saved["train_mean"], mean)
                and np.array_equal(saved["train_scale"], scale), "frozen frontend parity")
        require(set(saved.files) == set(state) | {"mel_filters", "train_mean", "train_scale"}
                and all(np.array_equal(saved[k], v.numpy()) and np.isfinite(saved[k]).all()
                        for k, v in state.items()), "PT to NPZ tensor parity")
    torch.onnx.export(restored, example, str(onx), opset_version=17, dynamo=False,
                      input_names=["fbank40"], output_names=["frame_ctc_logits"],
                      dynamic_axes={"fbank40": {1: "frames"}, "frame_ctc_logits": {1: "frames"}})
    graph = onnx.load(onx)
    onnx.checker.check_model(graph, full_check=True)
    with torch.no_grad():
        reference = restored(example).numpy()
    result = ReferenceEvaluator(graph).run(None, {"fbank40": example.numpy()})[0]
    require(result.shape == reference.shape and np.isfinite(result).all(), "ONNX shape")
    max_error = float(np.max(np.abs(result - reference)))
    require(max_error < 0.001, "ONNX export numerical mismatch")
    return {"files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in (pt, npz, onx)},
            "onnx_train_probe_max_abs_logit_error": max_error}


def fit(label, weights, cache, rows, draws, frontend, b10, output, torch, np):
    from torch.nn import functional as F

    torch.manual_seed(b10.SEED)
    net = b10.b9.v5.build_model(torch)
    net.load_state_dict(weights, strict=True)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=b10.LR, weight_decay=0.001)
    trace = []
    receipt_file = output / (label + "-completed-updates.jsonl")
    for step in draws:
        opt.zero_grad(set_to_none=True)
        prefix = step["prefix"]
        transformed = label == TEMPO_ARM and step["tempo_in_challenger"]
        losses, consistency = [], []
        for draw in step["draws"]:
            sid = draw["id"]
            tier = rows[sid][0]["tier"]
            posteriors = []
            for gain in b10.GAINS:
                cost, logp = b10.b6.suffix_loss(net, cache[sid, transformed, gain, prefix],
                                                 draw["class"], prefix, tier, torch, F, True)
                losses.append(cost)
                posteriors.append(logp)
            clean, half = posteriors
            start = prefix // 160 - 3
            require(clean.shape == half.shape, "cross-gain frame alignment")
            kl = F.kl_div(half[:, start:, :], clean[:, start:, :].detach(),
                          reduction="none", log_target=True).sum(-1).mean()
            consistency.append(kl.clamp(min=0.0, max=b10.b6.KL_CLIP))
        loss = sum(losses) / 4 + b10.b6.WEIGHT * sum(consistency) / 2
        require(bool(torch.isfinite(loss)), "finite paired loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
        opt.step()
        item = {"step": step["step"], "tempo_used": transformed, "loss": float(loss.detach()), "gradient_norm": float(norm)}
        trace.append(item)
        with receipt_file.open("a") as receipt:
            receipt.write(json.dumps(item, allow_nan=False) + "\n")
    example = cache[sorted(rows)[0], False, 1.0, 6400]
    out = export(net, label, output, frontend, example, b10.b9.v5.build_model, torch, np)
    dump(output / label / "OPTIMIZER.json", trace)
    out.update(optimizer_updates=len(trace), optimizer_trace_sha256=sha((output / label / "OPTIMIZER.json").read_bytes()))
    return out


def run(a):
    helpers = a.helpers / "automation"
    ordinary = load(helpers / "english_v6_b11_train_hardness.py")
    positive = load(helpers / "english_v6_b12_positive_train.py")
    pins = dict(ordinary.ARCHIVES, positive_diag=POSITIVE_DIAG, ordinary_diag=ORDINARY_DIAG, b10_pair=B10_PAIR)
    if a.list_inputs:
        for key, (aid, _) in pins.items():
            print(key, aid)
        return
    require(sha((helpers / "english_v6_b12_conditional_tempo_plan.json").read_bytes()) == PLAN_SHA, "wrong frozen recipe")
    with ExitStack() as stack:
        archives = {}
        for key, (_, digest) in pins.items():
            path = a.inputs / (key + ".zip")
            require(sha(path.read_bytes()) == digest, "archive SHA: " + key)
            archives[key] = stack.enter_context(zipfile.ZipFile(path))
        initial = archives["ordinary_diag"].read("RESULT.json")
        initial_train = json.loads(initial)
        require(initial_train["actual_forwards"] == 123 and initial_train["source_groups"] == 35
                and initial_train["hard_group_gate_passed"] is False, "original B11 measurements")
        diag = json.loads(archives["positive_diag"].read("RESULT.json"))
        require(diag["two_group_trigger"] and len(diag["verified_faster_clean_hit_half_miss_hey_absent_groups"]) >= 2
                and diag["actual_forwards"] == 132, "verified TRAIN gate")
        rows, groups = original_training_rows(archives, ordinary, positive)
    draws = schedule(groups)
    require(sum(x["tempo_in_challenger"] for x in draws) == 87 and len(draws) == 192, "fixed schedule")
    if a.audit_only:
        print("Verified 79 source-bound TRAIN inputs / 35 groups, 192 predetermined draws, 87 tempo slots.")
        return
    gate_contract = json.loads((Path(__file__).with_name("english_v6_b12_gate_pins.json")).read_bytes())
    require(sha(a.decision.read_bytes()) == gate_contract["b11_terminal_zip_sha256"], "B11 decision identity")
    with zipfile.ZipFile(a.decision) as archive:
        require(archive.testzip() is None, "terminal decision CRC")
        result = json.loads(archive.read("RESULT.json"))
        identity = json.loads(archive.read("RUN.json"))
        require(sha(archive.read("ALL_FRAME_LOGITS.npz")) == result["logits_sha256"] and
                int(identity["run"]) == gate_contract["b11_terminal_run"] and
                identity["code"] == gate_contract["b11_terminal_head"], "B11 actual numerical run")
        require(result["schema"] == "english-v6-4-frozen-panel-v1" and
                result["trial"]["hypothesis"] == "B11" and result["trial"]["panel"] == gate_contract["terminal_panel"]
                and result["panel_gate"] == "NO_PROMOTION" and result["admitted"] > 0
                and result["actual_forwards"] == 12 * result["admitted"] and result["optimizer_updates"] == 0
                and result["trained_pt_sha256"]["challenger"] ==
                    "b4535be7e3ba59d647dab94d103872cc921ecb452405c8b32bb033cc76d4aaa3",
                "B11 must actually reach terminal nonpromotion on its own DEV panel")
    dev_group_sets = []
    for name, expected_digest in gate_contract["b12_panel_sha256"].items():
        raw = (a.panel_root / (name + ".json")).read_bytes()
        require(sha(raw) == expected_digest, "B12 source plan digest: " + name)
        plan = json.loads(raw)
        require(plan["schema"] == "english-v6-b12-frozen-dev-panel-v1" and plan["panel"] == name
                and len(plan["rows"]) == 48 and all(r["role"] == "development" and not r["training_allowed"]
                for r in plan["rows"]), "independent B12 panel only")
        dev_group_sets.append({r["source_group_id"] for r in plan["rows"]})
    require(len(dev_group_sets) == 2 and all(len(s) == 12 for s in dev_group_sets)
            and not (dev_group_sets[0] & dev_group_sets[1])
            and not (set(groups[0]) & set.union(*dev_group_sets)), "prospective B12 DEV source boundaries")

    import numpy as np
    import torch
    import onnx

    a.output.mkdir()
    dump(a.output / "STARTED.json", {"run": os.environ["GITHUB_RUN_ID"], "head": os.environ["SCORE_HEAD"],
         "schema": SCHEMA, "optimizer_updates_budget": 384, "b11_terminal_gate": result["panel_gate"]})
    b10 = load(helpers / "english_v6_b10_hardneg_paired_trainonce.py")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    parent_models, _ = b10.b9.verify_b9_archive(a.inputs / "b9.zip", torch, np)
    model, bank, mean, scale = parent_models["B9_control"]
    weights = {k: t.detach().clone() for k, t in model.state_dict().items()}
    sox = shutil.which("sox")
    require(sox is not None, "SoX required")
    version = subprocess.check_output([sox, "--version"], text=True).strip()
    require(version.endswith("SoX v14.4.2"), "frozen SoX DSP version")
    linked = subprocess.check_output(["ldd", sox], text=True)
    match = re.search(r"^\s*libsox\.so\.\d+ => (\S+)", linked, re.M)
    require(match is not None, "SoX shared library identity")
    cache, manifest = prepare_cache(rows, b10, bank, mean, scale, sox, torch, np)
    feature_file = a.output / "TRAIN_INPUTS.npz"
    feature_arrays = {f"{sid}__tempo{int(warped)}__gain{gain}__prefix{prefix}": tensor.numpy()
                      for (sid, warped, gain, prefix), tensor in cache.items()}
    require(len(feature_arrays) == 79 * 2 * 2 * 3, "all source-bound model inputs")
    np.savez_compressed(feature_file, **feature_arrays)
    dump(a.output / "SOURCE.json", {"schema": SCHEMA, "train_inputs_npz_sha256": sha(feature_file.read_bytes()), "old_train_originals": 79, "old_train_groups": 35,
         "parent_pt_sha256": b10.b9.MODELS["B9_control"]["pt"], "inputs": pins, "rows": manifest,
         "schedule": draws, "predeclared_recipe_sha256": PLAN_SHA, "gate_contract": gate_contract,
         "primary_and_replication_reserved_source_groups": [sorted(s) for s in dev_group_sets],
         "sox_version": version,
         "sox_binary_sha256": sha(Path(sox).read_bytes()), "libsox_sha256": sha(Path(match.group(1)).read_bytes()),
         "tempo_flags": ["-R", "-D", "tempo", "-s", "1.08"], "holdout_audio_in_train": False})
    env = {"torch": torch.__version__, "numpy": np.__version__, "onnx": onnx.__version__,
           "python": platform.python_version(), "image_version": os.getenv("ImageVersion")}
    dump(a.output / "ENVIRONMENT.json", env)
    args = (weights, cache, rows, draws, (bank, mean, scale), b10, a.output, torch, np)
    control = fit("matched_control", *args)
    challenger = fit(TEMPO_ARM, *args)
    def state(label):
        return torch.load(a.output / label / "models" / (STEMS[label] + ".pt"), weights_only=True, map_location="cpu")
    left, right = state("matched_control"), state(TEMPO_ARM)
    different = sum(int(torch.count_nonzero(left[k] != right[k])) for k in left)
    require(different > 0, "identical paired outputs")
    with zipfile.ZipFile(a.inputs / "b10_pair.zip") as archive:
        raw = archive.read("matched_control/models/mero43-b10-matched-control192.npz")
        with np.load(io.BytesIO(raw), allow_pickle=False) as old:
            control_drift = {k: float(np.max(np.abs(v.numpy() - old[k]))) for k, v in left.items()}
            control_float_differences = sum(int(np.count_nonzero(v.numpy() != old[k])) for k, v in left.items())
    dump(a.output / "RESULT.json", {"schema": SCHEMA, "status": "TRAINED_RESEARCH_PAIR_NO_PROMOTION",
         "run": os.environ["GITHUB_RUN_ID"], "head": os.environ["SCORE_HEAD"], "control": control, "challenger": challenger,
         "old_train_originals": 79, "old_train_groups": 35, "trained_model_parameters": 43876,
         "actual_optimizer_updates": 384, "different_trained_float_values": different,
         "historical_b10_control_different_float_values": control_float_differences,
         "historical_b10_control_max_abs_delta": max(control_drift.values()),
         "parent_pt_sha256": b10.b9.MODELS["B9_control"]["pt"], "dev_cal_locked_forwards": 0,
         "new_tts_asr_calls": 0, "promotion_allowed": False, "source_sha256": sha((a.output / "SOURCE.json").read_bytes())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("helpers", "inputs", "output", "decision", "panel-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--list-inputs", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    run(parser.parse_args())
