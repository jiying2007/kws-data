"""B17 paired B0-origin TRAIN experiment with a one-sided ordinary margin ceiling.

No DEV waveform, threshold search, fresh source generation, or promotion occurs
here. Both arms share positive retention; only the ordinary B0 ceiling differs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b15_nonblank_contrast_pair as b15
import english_v6_mero_frozen_b0_b1_paired_once as baseline

b13, b10, need = b15.b13, b15.b10, b15.need
HERE = Path(__file__).resolve().parent
DESIGN_FILE = HERE / "english_v6_b17_b0_margin_design.json"
ASSESSMENT = HERE / "english_v6_b17_panels" / "assessment.json"
DESIGN_SHA = "f9ba0ce59b0deed805b313063e648470e61af6c9122d08f6fe9aa5ea9e324831"
ASSESSMENT_SHA = "7fdfe9cd47b18f64429fece944468af8adfde6c6ba012a57236a8a33aae1a948"
B0_PT = "1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
SCHEMA = "english-v6-b17-b0-ordinary-margin-pair-v1"
REGISTRATION = "B17_B0_ORDINARY_CEILING_FIXED_20261124"
RUN_BRANCH = "refs/heads/research/en-v6-b17-b0-margin-train-once-20261124"
GAINS = (1.0, 0.75, 0.5)
STEPS, SEED, LR = 192, 20261120, 0.00006
NATURAL_WEIGHT = POSITIVE_WEIGHT = CEILING_WEIGHT = 0.08


def read_design():
    need(re.fullmatch(r"[0-9a-f]{64}", DESIGN_SHA) is not None
         and re.fullmatch(r"[0-9a-f]{64}", ASSESSMENT_SHA) is not None,
         "B17 protocol and independent assessment must be frozen before execution")
    raw, assessment = DESIGN_FILE.read_bytes(), ASSESSMENT.read_bytes()
    need(b10.digest(raw) == DESIGN_SHA and b10.digest(assessment) == ASSESSMENT_SHA,
         "immutable B17 experiment and prospective independent assessment")
    d, a = json.loads(raw), json.loads(assessment)
    p = d["pair"]
    need(d["schema"] == "english-v6-b17-b0-ordinary-ceiling-design-v1"
         and p["parent_pt_sha256"] == B0_PT and p["updates_per_arm"] == STEPS
         and p["seed"] == SEED and p["adamw_learning_rate"] == LR
         and p["natural_negative_ctc_weight"] == NATURAL_WEIGHT
         and p["shared_positive_retention_weight"] == POSITIVE_WEIGHT
         and p["control_ceiling_weight"] == 0
         and p["challenger_ceiling_weight"] == CEILING_WEIGHT
         and d["independent_assessment_sha256"] == ASSESSMENT_SHA
         and a["freeze_before_optimizer"] is True
         and d["promotion_allowed"] is False,
         "fixed B0 parent and exactly one changed ordinary TRAIN loss")
    need(b10.STEPS == STEPS and b10.SEED == SEED and b10.LR == LR
         and b10.GAINS == (1.0, 0.5) and b10.PREFIXES == (6400, 12800, 19200)
         and b10.TRAIN_GROUPS == 35 and b10.PARAMS == 43876
         and b10.b6.WEIGHT == 0.03 and b10.b6.KL_CLIP == 8.0,
         "unchanged shared source and optimizer recipe")
    b15.read_design()  # Authenticate inherited source and positive-teacher protocol.
    return d


def guard(prepared_parent, env=None, event=None, parent=None):
    e = os.environ if env is None else env
    need(re.fullmatch(r"[0-9a-f]{40}", prepared_parent or "") is not None,
         "explicit reviewed predecessor required; placeholder cannot run")
    need(all(e.get(k) == v for k, v in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "jiying2007/kws-data",
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": RUN_BRANCH,
        "GITHUB_RUN_ATTEMPT": "1"}.items()), "first authorized public B17 branch event only")
    need(re.fullmatch(r"[0-9a-f]{40}", e.get("GITHUB_SHA", "")) is not None
         and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")) is not None,
         "actual B17 run identity")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent
         and event.get("after") == e["GITHUB_SHA"] and event.get("ref") == RUN_BRANCH
         and event.get("repository", {}).get("full_name") == "jiying2007/kws-data"
         and event.get("repository", {}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "exact prepared parent, public repository, and no replay or forced push")
    return dict(run_id=e["GITHUB_RUN_ID"], head=e["GITHUB_SHA"], parent=parent)


def fetch(inputs, prepared_parent):
    guard(prepared_parent)
    read_design()
    b15.fetch(inputs)
    pin = baseline.MODELS["B0"]
    run = json.loads(b13.gh("actions/runs/" + str(pin["run"])))
    need(run["head_sha"] == pin["head"] and run["conclusion"] == "success"
         and run["run_attempt"] == 1, "authentic completed deployed B0 source run")
    raw = b13.gh("actions/artifacts/" + str(pin["id"]) + "/zip")
    need(len(raw) == pin["bytes"] and b10.digest(raw) == pin["sha"], "exact B0 archive")
    (inputs / "B0.zip").write_bytes(raw)
    (inputs / "B17_DESIGN.json").write_bytes(DESIGN_FILE.read_bytes())
    (inputs / "B17_ASSESSMENT.json").write_bytes(ASSESSMENT.read_bytes())


def ordinary_row(row):
    need(row.get("role") == "train" and type(row.get("mono_class")) is int
         and row["mono_class"] == 0 and row.get("source_group_id") and row.get("id")
         and re.fullmatch(r"[0-9a-f]{64}", row.get("source_pcm_sha256", "")) is not None,
         "ordinary ceiling accepts only authenticated ordinary TRAIN rows")
    return row


def scalar_ceiling(student, teacher):
    """Small independent arithmetic oracle for the tensor loss tests."""
    need(len(student) == len(teacher) == 4
         and all(math.isfinite(v) for v in (*student, *teacher)), "finite four-class row")
    return sum(max(0.0, (student[k] - student[0]) - (teacher[k] - teacher[0]))
               for k in (1, 2, 3)) / 3.0


def ordinary_margin_loss(student, teacher, row, torch):
    ordinary_row(row)
    need(student.ndim == teacher.ndim == 2 and student.shape == teacher.shape
         and student.shape[1] == 4 and len(student) > 0
         and bool(torch.isfinite(student).all()) and bool(torch.isfinite(teacher).all()),
         "identical finite native source/gain/prefix frame grids")
    # log_softmax has an identical additive normalization for all four classes;
    # its differences equal raw-logit differences, so reuse the CTC forward.
    ref = teacher.detach()
    sm = student[:, 1:] - student[:, :1]
    tm = ref[:, 1:] - ref[:, :1]
    return torch.relu(sm - tm).mean()


def shared_schedule(groups, extra):
    draws = b13.paired.freeze_draw(extra, STEPS)
    slots = []
    for step, natural in enumerate(draws):
        ordinary_row(natural)
        slot = dict(step=step + 1, natural=natural,
                    prefix=b10.PREFIXES[(step // 35 + step % 3) % len(b10.PREFIXES)])
        for klass, key in ((0, "ordinary"), (1, "positive")):
            gid = sorted(groups[klass])[step % 35]
            rows = sorted(groups[klass][gid], key=lambda r: r["id"])
            slot[key] = rows[(step // 35) % len(rows)]
        ordinary_row(slot["ordinary"])
        slots.append(slot)
    need(len(slots) == STEPS, "bounded complete shared schedule")
    return slots


def load_parents(inputs, torch, np):
    raw = (inputs / "B0.zip").read_bytes()
    pin = baseline.MODELS["B0"]
    need(len(raw) == pin["bytes"] and b10.digest(raw) == pin["sha"], "B0 original ZIP identity")
    with zipfile.ZipFile(io.BytesIO(raw)) as ar, tempfile.TemporaryDirectory() as tmp:
        need(ar.testzip() is None, "B0 archive CRC")
        root = Path(tmp)
        for name in ["RESULT.json"] + ["models/mero-fsmn-ctc-expanded256." + e
                                         for e in ("pt", "npz", "onnx")]:
            dest = root / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(ar.read(name))
        b0net, mel, mean, scale, hashes = baseline.model(torch, np, root, "B0")
    need(hashes["pt"] == B0_PT, "exact deployed B0 checkpoint, not B9")
    models, _ = b10.b9.verify_b9_archive(inputs / "b9.zip", torch, np)
    b9net, *b9front = models["B9_control"]
    need(all(np.array_equal(left, right) for left, right in zip((mel, mean, scale), b9front)),
         "B0/B9 exact same mel bank, mean and scale before teacher transfer")
    for model in (b0net, b9net):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return b0net, b9net, (mel, mean, scale)


def diagnostic_gate(records):
    negative_groups, recovered_groups = set(), set()
    for row in records:
        need(row["role"] == "train", "diagnostic can only use TRAIN")
        for state in row["gains"].values():
            if not row["positive"] and state["mean_positive_margin_increase"] > 0:
                negative_groups.add(row["group"])
            if row["positive"] and state["B9"]["events"] > 0 and state["B0"]["events"] == 0:
                recovered_groups.add(row["group"])
    passed = len(negative_groups) >= 2 and len(recovered_groups) >= 1
    return dict(status="TRAIN_SIGNAL" if passed else "STOP_TRAIN_HYPOTHESIS_UNSUPPORTED",
                ordinary_margin_elevation_groups=sorted(negative_groups),
                b9_recovered_b0_positive_groups=sorted(recovered_groups),
                optimizer_start_allowed=passed, promotion_allowed=False)


def train_diagnostic(inputs, rows, waves, b0net, b9net, frontend, output, torch, np):
    """Reuse 132 saved positive B9 arrays; 396 new necessary TRAIN forwards."""
    mel, mean, scale = frontend
    with zipfile.ZipFile(inputs / "diagnostic.zip") as ar:
        old = json.loads(ar.read("RESULT.json"))
        raw = ar.read("ALL_FRAME_LOGITS.npz")
    need(b10.digest(raw) == old["logits_sha256"] and old["actual_forwards"] == 396,
         "original saved positive B9 arrays")
    previous = {r["id"]: r for r in old["records"]}
    arrays, records, new_forwards, reused = {}, [], 0, 0
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved, torch.inference_mode():
        for index, row in enumerate(sorted(rows, key=lambda r: r["id"])):
            need(row["role"] == "train" and b10.digest(waves[row["id"]]) == row["source_pcm_sha256"],
                 "only approved TRAIN original PCM")
            positive = row["mono_class"] == 1
            record = dict(id=row["id"], group=row["source_group_id"], role="train", positive=positive,
                          pcm_sha256=row["source_pcm_sha256"], gains={})
            for gain in GAINS:
                pure = b15.diagnostic.frozen.ref.gain_pcm(waves[row["id"]], gain, np)
                f = b15.diagnostic.frozen.v5.features(pure, row, torch, np, mel)
                x = torch.from_numpy(((f - mean) / scale).astype(np.float32))[None]
                matrices = {"B0": b0net(x)[0].cpu().numpy().copy()}
                new_forwards += 1
                if positive:
                    oldrow = previous[row["id"]]
                    need(oldrow["pcm_sha256"] == row["source_pcm_sha256"]
                         and oldrow["group"] == row["source_group_id"], "saved B9 positive source identity")
                    slot = oldrow["gains"][str(gain)]["B9"]
                    matrices["B9"] = saved[slot["array"]].copy()
                    need(b10.digest(matrices["B9"].tobytes()) == slot["sha256"], "saved B9 array SHA")
                    reused += 1
                else:
                    ordinary_row(row)
                    matrices["B9"] = b9net(x)[0].cpu().numpy().copy()
                    new_forwards += 1
                state = {}
                for name, matrix in matrices.items():
                    need(matrix.shape == (len(f), 4) and np.isfinite(matrix).all(), "finite actual TRAIN logits")
                    key = f"r{index:03d}__g{str(gain).replace('.', '_')}__{name}"
                    arrays[key] = matrix
                    tensor = torch.from_numpy(matrix)[None]
                    tokens, events = b15.diagnostic.decode(tensor, torch)
                    state[name] = dict(array=key, sha256=b10.digest(matrix.tobytes()),
                                       events=len(events), event_frames=events, tokens=tokens,
                                       origin="saved_original_B13_TRAIN_diagnostic" if name == "B9" and positive
                                              else "new_B17_TRAIN_forward")
                diff = ((matrices["B9"][:, 1:] - matrices["B9"][:, :1])
                        - (matrices["B0"][:, 1:] - matrices["B0"][:, :1]))
                state["mean_positive_margin_increase"] = float(np.maximum(diff, 0).mean())
                record["gains"][str(gain)] = state
            records.append(record)
    need(len(rows) == 88 and new_forwards == 396 and reused == 132 and len(arrays) == 528,
         "exact bounded 88-source diagnostic; reuse original positive B9 scores")
    path = output / "TRAIN_DIAGNOSTIC_LOGITS.npz"
    np.savez_compressed(path, **arrays)
    report = dict(schema="english-v6-b17-train-mechanism-diagnostic-v1", records=records,
                  **diagnostic_gate(records), actual_new_model_forwards=new_forwards,
                  reused_saved_b9_positive_arrays=reused, saved_arrays=len(arrays),
                  logits_sha256=b10.digest(path.read_bytes()), optimizer_updates=0,
                  dev_cal_locked_forwards=0, new_audio_asr_calls=0)
    b10.write(output / "TRAIN_DIAGNOSTIC.json", report)
    return report


def make_ordinary_teachers(slots, base_cache, natural_cache, b0net, output, torch, np):
    cache, arrays, audit = {}, {}, []
    with torch.inference_mode():
        for slot in slots:
            for source, features in ((slot["ordinary"], base_cache), (slot["natural"], natural_cache)):
                ordinary_row(source)
                for gain in b10.GAINS:
                    key = (source["id"], gain, slot["prefix"])
                    if key in cache:
                        continue
                    x = features[source["id"]][gain][slot["prefix"]]
                    matrix = b0net(x)[0].detach().cpu().numpy().copy()
                    need(matrix.shape == (x.shape[1], 4) and np.isfinite(matrix).all(), "frozen ordinary teacher grid")
                    name = f"teacher_{len(cache):04d}"
                    arrays[name] = matrix
                    cache[key] = torch.from_numpy(matrix)
                    audit.append(dict(id=source["id"], source_group_id=source["source_group_id"],
                                      role="train", mono_class=0, pcm_sha256=source["source_pcm_sha256"],
                                      prefix_samples=slot["prefix"], gain=gain, array=name,
                                      sha256=b10.digest(matrix.tobytes()), frames=len(matrix)))
    path = output / "B0_ORDINARY_TEACHER_LOGITS.npz"
    np.savez_compressed(path, **arrays)
    b10.write(output / "B0_ORDINARY_TEACHER.json", dict(parent_pt_sha256=B0_PT, rows=audit,
               actual_model_forwards=len(cache), logits_sha256=b10.digest(path.read_bytes()),
               reused_for_both_arms=True, positive_rows=0, dev_cal_locked_forwards=0))
    return cache, len(cache)


def ceiling_loss(samples, teachers, torch):
    grouped = defaultdict(lambda: defaultdict(list))
    for row, gain, prefix, logp in samples:
        ordinary_row(row)
        teacher = teachers[(row["id"], gain, prefix)]
        loss = ordinary_margin_loss(logp[0], teacher, row, torch)
        grouped[row["source_group_id"]][row["id"]].append(loss)
    need(len(samples) == 4 and len(grouped) == 2, "same two ordinary groups and two gains as base CTC")
    values = [torch.stack([torch.stack(gains).mean() for gains in originals.values()]).mean()
              for originals in grouped.values()]
    return torch.stack(values).mean()


def fit(arm, initial, slots, base_cache, natural_cache, positive_paths, teachers,
        frontend, output, torch, np, F):
    need(arm in ("control", "challenger"), "registered pair only")
    coefficient = 0.0 if arm == "control" else CEILING_WEIGHT
    torch.manual_seed(SEED)
    net = b10.b6.b4.v5.build_model(torch)
    net.load_state_dict(initial, strict=True)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.001)
    positive_trace, trace = [], []
    retain = b15.retention_loss(positive_paths, POSITIVE_WEIGHT, positive_trace, torch)
    counts, phase = defaultdict(int), ["optimization"]
    def count_forward(_module, _args, _result):
        counts[phase[0]] += 1
    handle = net.register_forward_hook(count_forward)
    started = time.perf_counter()
    for step, slot in enumerate(slots):
        optimizer.zero_grad(set_to_none=True)
        prefix = slot["prefix"]
        losses, consistency, ordinary_outputs = [], [], []
        for klass, kind in ((0, "ordinary"), (1, "positive")):
            row = slot[kind]
            pair = {}
            for gain in b10.GAINS:
                base, logp = b10.b6.suffix_loss(net, base_cache[row["id"]][gain][prefix], klass,
                    prefix, row["tier"], torch, F, True)
                losses.append(base)
                pair[gain] = logp
                if klass == 0:
                    ordinary_outputs.append((row, gain, prefix, logp))
            clean, half = pair[1.0], pair[0.5]
            at = prefix // 160 - 3
            need(clean.shape == half.shape and 0 < at < half.shape[1] - 5, "same native gain/suffix grid")
            kl = F.kl_div(half[:, at:, :], clean[:, at:, :].detach(),
                          reduction="none", log_target=True).sum(-1).mean()
            need(bool(torch.isfinite(kl)), "finite shared gain consistency")
            consistency.append(kl.clamp(min=0.0, max=b10.b6.KL_CLIP))
        natural_losses = []
        row = slot["natural"]
        for gain in b10.GAINS:
            loss, logp = b10.b6.suffix_loss(net, natural_cache[row["id"]][gain][prefix], 0,
                prefix, row["tier"], torch, F, True)
            natural_losses.append(loss)
            ordinary_outputs.append((row, gain, prefix, logp))
        positive = retain(step, net)
        ceiling = ceiling_loss(ordinary_outputs, teachers, torch)
        loss = (sum(losses) / 4 + b10.b6.WEIGHT * sum(consistency) / 2
                + NATURAL_WEIGHT * sum(natural_losses) / 2 + positive + coefficient * ceiling)
        need(bool(torch.isfinite(loss)), "finite fixed B17 objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        trace.append(dict(step=step + 1, prefix=prefix, ordinary_id=slot["ordinary"]["id"],
                          positive_id=slot["positive"]["id"], natural_id=slot["natural"]["id"],
                          loss=float(loss.detach()), positive_retention=float(positive.detach()),
                          ordinary_ceiling=float(ceiling.detach()), ceiling_coefficient=coefficient))
    need(len(trace) == len(positive_trace) == STEPS, "192 actual optimizer updates per arm")
    net.eval()
    directory = output / "models"
    directory.mkdir()
    stem = "mero43-b17-" + arm + "192"
    pt = directory / (stem + ".pt")
    torch.save(net.state_dict(), pt)
    state = torch.load(pt, map_location="cpu", weights_only=True)
    rebuilt = b10.b6.b4.v5.build_model(torch)
    rebuilt.load_state_dict(state, strict=True)
    rebuilt.eval()
    need(sum(p.numel() for p in rebuilt.parameters()) == 43876
         and all(bool(torch.isfinite(v).all()) for v in state.values())
         and all(torch.equal(v, net.state_dict()[k]) for k, v in state.items()),
         "saved full 43876-parameter checkpoint exact reload")
    mel, mean, scale = frontend
    npz = directory / (stem + ".npz")
    np.savez(npz, mel_filters=mel, train_mean=mean, train_scale=scale,
             **{k: v.cpu().numpy() for k, v in state.items()})
    first = slots[0]["ordinary"]["id"]
    example = base_cache[first][1.0][slots[0]["prefix"]]
    phase[0] = "checkpoint_reload_parity"
    with torch.inference_mode():
        expected, observed = net(example), rebuilt(example)
    need(torch.equal(expected, observed), "new saved PT actual reload forward equality")
    counts["checkpoint_reload_parity"] += 1  # The independently rebuilt model.
    with np.load(npz, allow_pickle=False) as encoded:
        need(set(encoded.files) == set(state) | {"mel_filters", "train_mean", "train_scale"}
             and all(np.array_equal(encoded[k], v.cpu().numpy()) for k, v in state.items()),
             "saved PT and NPZ numerical tensor identity")
    phase[0] = "onnx_export_trace"
    onnx = directory / (stem + ".onnx")
    torch.onnx.export(net, example, onnx.as_posix(), opset_version=17, dynamo=False,
                      input_names=["fbank40"], output_names=["frame_ctc_logits"],
                      dynamic_axes={"fbank40": {1: "frames"}, "frame_ctc_logits": {1: "frames"}})
    handle.remove()
    import onnx as onnx_api
    onnx_api.checker.check_model(onnx_api.load(onnx))
    need(all(path.stat().st_size > 10000 for path in (pt, npz, onnx)), "complete actual model exports")
    b10.write(output / "OPTIMIZER_TRACE.json", trace)
    b10.write(output / "POSITIVE_RETENTION_TRACE.json", positive_trace)
    return dict(stem=arm + "/models/" + stem, optimizer_updates=STEPS,
                parent_pt_sha256=B0_PT, natural_loss_weight=NATURAL_WEIGHT,
                positive_retention_weight=POSITIVE_WEIGHT, ordinary_ceiling_weight=coefficient,
                forward_counts=dict(counts), optimizer_seconds=round(time.perf_counter() - started, 3),
                files={p.name: dict(bytes=p.stat().st_size, sha256=b10.digest(p.read_bytes()))
                       for p in (pt, npz, onnx)}), state


def execute(inputs, output, prepared_parent):
    identity = guard(prepared_parent)
    design = read_design()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), "one new output, no overwrite")
    need((inputs / "B17_DESIGN.json").read_bytes() == DESIGN_FILE.read_bytes()
         and (inputs / "B17_ASSESSMENT.json").read_bytes() == ASSESSMENT.read_bytes(),
         "fetched and current pre-optimizer registrations identical")
    output.mkdir()
    b10.write(output / "STARTED.json", dict(schema=SCHEMA, **identity, maximum_optimizer_updates=384,
               design_sha256=DESIGN_SHA, assessment_sha256=ASSESSMENT_SHA,
               new_tts_asr_calls=0, dev_cal_locked_forwards=0, promotion_allowed=False))
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    old_design = b15.read_design()
    need(b10.digest((inputs / "diagnostic.zip").read_bytes()) == old_design["diagnostic"]["sha256"]
         and b10.digest((inputs / "pair.zip").read_bytes()) == old_design["historical_b13"]["sha256"],
         "original positive teacher and reviewed source schedule archives")
    pins = json.loads(b13.PINS.read_bytes())
    extra, extra_waves = b13.reviewed_gate(inputs, pins, np)
    rows, waves, groups = b13.paired.original_training(inputs)
    with zipfile.ZipFile(inputs / "pair.zip") as ar:
        source = json.loads(ar.read("SOURCE.json"))
        historical_slots = json.loads(ar.read("COMMON_DRAW.json"))
    need(len(rows) == 79 and len(extra) == 9 and len({r["source_group_id"] for r in extra}) == 8
         and {r["source_pcm_sha256"] for r in rows} == {r["pcm_sha256"] for r in source["base_train"]}
         and extra == source["natural_train"] and not (set(waves) & set(extra_waves)),
         "same 79 synthetic plus nine owner-reviewed natural TRAIN originals")
    b0net, b9net, frontend = load_parents(inputs, torch, np)
    all_rows, all_waves = rows + extra, {**waves, **extra_waves}
    diagnostic = train_diagnostic(inputs, all_rows, all_waves, b0net, b9net, frontend, output, torch, np)
    if not diagnostic["optimizer_start_allowed"]:
        b10.write(output / "RESULT.json", dict(schema=SCHEMA, **identity,
                   status=diagnostic["status"], actual_optimizer_updates=0,
                   actual_new_model_forwards=diagnostic["actual_new_model_forwards"],
                   diagnostic_sha256=b10.digest((output / "TRAIN_DIAGNOSTIC.json").read_bytes()),
                   dev_cal_locked_forwards=0, new_tts_asr_calls=0, promotion_allowed=False))
        return
    slots = shared_schedule(groups, extra)
    need(len(historical_slots) == len(slots), "same historical TRAIN draw count")
    for slot, historical in zip(slots, historical_slots):
        need(slot["prefix"] == historical["prefix"]
             and slot["natural"]["id"] == historical["id"]
             and slot["natural"]["source_pcm_sha256"] == historical["source_pcm_sha256"],
             "unchanged reviewed natural original and silence-prefix draw")
        for kind in ("ordinary", "positive"):
            row = slot[kind]
            need(historical[kind] == dict(id=row["id"], group=row["source_group_id"],
                                          pcm_sha256=row["source_pcm_sha256"]),
                 "unchanged base TRAIN source-group and original draw")
    b10.write(output / "COMMON_DRAW.json", slots)
    cache = b13.paired.make_features(rows, waves, frontend, torch, np)
    extra_cache = b13.paired.make_features(extra, extra_waves, frontend, torch, np)
    positive_paths, audit = b15.prepare_paths(inputs, rows, waves, frontend, old_design, np, torch)
    teachers, teacher_count = make_ordinary_teachers(slots, cache, extra_cache, b0net, output, torch, np)
    b10.write(output / "SOURCE.json", dict(base_train=source["base_train"], natural_train=extra,
               positive_retention=audit, registered_optimizer=REGISTRATION,
               independent_assessment_sha256=ASSESSMENT_SHA, design_sha256=DESIGN_SHA,
               parent_pt_sha256=B0_PT, positive_teacher_pt_sha256=b10.b9.MODELS["B9_control"]["pt"],
               input_zip_sha256={p.name: b10.digest(p.read_bytes()) for p in sorted(inputs.glob("*.zip"))},
               forbidden_sources_opened=False))
    initial = {k: v.detach().clone() for k, v in b0net.state_dict().items()}
    arms, states = {}, {}
    for arm in ("control", "challenger"):
        dest = output / arm
        dest.mkdir()
        arms[arm], states[arm] = fit(arm, initial, slots, cache, extra_cache,
                                      positive_paths, teachers, frontend, dest, torch, np, F)
    differing = sum(int(torch.count_nonzero(states["control"][k] != v))
                    for k, v in states["challenger"].items())
    need(differing > 0 and all(v["optimizer_updates"] == 192 for v in arms.values())
         and arms["control"]["forward_counts"]["optimization"] == arms["challenger"]["forward_counts"]["optimization"],
         "same exact exposure and 384 actual paired updates yield different learned tensors")
    b10.write(output / "RESULT.json", dict(schema=SCHEMA, status="TRAINED_PAIR_REQUIRES_FRESH_INDEPENDENT_DEV",
               **identity, **arms, registration=REGISTRATION, parent_pt_sha256=B0_PT,
               trained_model_parameters=43876, actual_optimizer_updates=384,
               different_trained_float_values=differing, seed=SEED, common_learning_rate=LR,
               design_sha256=DESIGN_SHA, assessment_sha256=ASSESSMENT_SHA,
               independent_assessment_sha256=ASSESSMENT_SHA,
               common_draw_sha256=b10.digest((output / "COMMON_DRAW.json").read_bytes()),
               actual_new_diagnostic_forwards=diagnostic["actual_new_model_forwards"],
               reused_saved_diagnostic_arrays=diagnostic["reused_saved_b9_positive_arrays"],
               actual_ordinary_teacher_forwards=teacher_count,
               actual_student_forwards=sum(sum(a["forward_counts"].values()) for a in arms.values()),
               positive_path_combinations=len(audit), positive_originals=43, positive_families=34,
               baseline_originals=79, natural_originals=9, natural_families=8,
               dev_cal_locked_forwards=0, new_tts_asr_calls=0, promotion_allowed=False,
               physical_ssc305="NOT_RUN", field_far_per_hour="NOT_MEASURED"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("fetch", "execute"))
    ap.add_argument("inputs", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--prepared-parent", required=True)
    args = ap.parse_args()
    if args.mode == "fetch":
        fetch(args.inputs, args.prepared_parent)
    else:
        need(args.output is not None, "output directory required")
        execute(args.inputs, args.output, args.prepared_parent)
