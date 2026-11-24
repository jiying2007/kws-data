"""B21 TRAIN-only paired nonblank versus full-competitor positive retention.

Both arms cold-start B0, retain B17 ordinary ceiling, and use a shared repaired
source/prefix coverage schedule. No historical numerical replay is claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_b0_margin_pair as b17

b15, b13, b10, need = b17.b15, b17.b13, b17.b10, b17.need
HERE = Path(__file__).resolve().parent
DESIGN_FILE = HERE / "english_v6_b21_competing_margin_design.json"
ASSESSMENT = HERE / "english_v6_b21_panels" / "assessment.json"
DESIGN_SHA = "76a1fd1353de00f712023a2b20256a8e622557551823cacd8b808d3f0549b0a1"
ASSESSMENT_SHA = "65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc"
B0_PT = b17.B0_PT
SCHEMA = "english-v6-b21-positive-competing-margin-pair-v1"
REGISTRATION = "B21_FULL_COMPETITOR_TRAIN_FIXED_20261124"
RUN_BRANCH = "refs/heads/research/en-v6-b21-blank-retention-train-once-20261124"
GAINS = (1.0, 0.75, 0.5)
STEPS, SEED, LR = 192, 20261120, 0.00006
NATURAL_WEIGHT = POSITIVE_WEIGHT = CEILING_WEIGHT = 0.08
B17_ARCHIVE = dict(run_id=45319580193, head="9654cfe1c28ed84fe7d1d2a91dab450d8efb1abf",
    artifact_id=14887048023, bytes=2506252,
    sha256="1e31bd9c801657c7850fa6fb05c5468cbe8b4ab5bb459b0146c07c4c905ff86d")
B17_DIAGNOSTIC_SHA = "2035f85a8acbe40c981a6e9440b6e8195fdc1112dfb76c8ddaa4f36eb1a7878e"
B17_ARRAYS_SHA = "9f569f783ec9f21af173c821d49485be2ab31881ef21be9bdc69ac9792451f4f"
B17_SOURCE_SHA = "90779521f6fe2c0b97ff586c9c7fdd1c4942a0259b3fc0c2cdeaac4c34f55a63"


def read_design():
    need(all(re.fullmatch(r"[0-9a-f]{64}", h) is not None for h in (DESIGN_SHA, ASSESSMENT_SHA)),
         "B21 design and independent assessment must be frozen before execution")
    raw, assessment = DESIGN_FILE.read_bytes(), ASSESSMENT.read_bytes()
    need(b10.digest(raw) == DESIGN_SHA and b10.digest(assessment) == ASSESSMENT_SHA,
         "immutable B21 design and independent assessment")
    d, a = json.loads(raw), json.loads(assessment)
    p = d["pair"]
    need(d["schema"] == "english-v6-b21-positive-competing-margin-design-v1"
         and p["parent_pt_sha256"] == B0_PT and p["updates_per_arm"] == STEPS
         and p["seed"] == SEED and p["adamw_learning_rate"] == LR
         and p["shared_natural_ctc_weight"] == NATURAL_WEIGHT
         and p["shared_positive_retention_weight"] == POSITIVE_WEIGHT
         and p["shared_ordinary_ceiling_weight"] == CEILING_WEIGHT
         and p["control_competitors"] == "other_two_nonblank"
         and p["challenger_competitors"] == "other_three_including_blank"
         and d["independent_assessment_sha256"] == ASSESSMENT_SHA
         and a["freeze_before_optimizer_and_new_source"] is True and d["promotion_allowed"] is False,
         "fixed B0 parent, shared ordinary ceiling and one changed positive competitor set")
    need(b10.STEPS == STEPS and b10.SEED == SEED and b10.LR == LR
         and b10.GAINS == (1.0, 0.5) and b10.PREFIXES == (6400, 12800, 19200)
         and b10.TRAIN_GROUPS == 35 and b10.PARAMS == 43876
         and b10.b6.WEIGHT == 0.03 and b10.b6.KL_CLIP == 8.0,
         "fixed inherited source, features and optimizer recipe")
    at = a["training"]
    expected_assessment = {
        "shared_parent": "B0", "shared_parent_pt_sha256": B0_PT,
        "parameters": 43876, "updates_per_arm": STEPS, "seed": SEED,
        "adamw_learning_rate": LR, "adamw_weight_decay": 0.001, "gradient_norm_clip": 5,
        "synthetic_train_originals": 79, "synthetic_positive_originals": 44,
        "synthetic_ordinary_originals": 35, "owner_reviewed_natural_windows": 9,
        "natural_train_readers": 8, "shared_ctc_gains": [1.0, 0.5],
        "shared_silence_prefix_samples": [6400, 12800, 19200],
        "shared_gain_consistency_weight": 0.03,
        "shared_natural_negative_ctc_weight": NATURAL_WEIGHT,
        "shared_ordinary_B0_margin_ceiling_weight": CEILING_WEIGHT,
        "shared_positive_retention_weight": POSITIVE_WEIGHT,
        "eligible_positive_original_gain_combinations": 120,
        "eligible_positive_originals": 43, "eligible_positive_groups": 34,
        "retention_gains": [1.0, 0.75, 0.5],
        "only_arm_contrast": {
            "control": "target-minus-max-other-two-nonblank-token-logits",
            "challenger": "target-minus-max-other-three-class-logits-including-blank"},
        "historical_B17_replay_claim": False, "new_training_audio": False,
        "new_diagnostic_forwards": 0, "dev_cal_locked_forwards_in_training": 0,
        "threshold_search": False}
    need(all(at.get(k) == v for k, v in expected_assessment.items()),
         "independent B21 assessment and implementation share the exact training recipe")
    b17.read_design()
    return d


def guard(prepared_parent, env=None, event=None, parent=None):
    e = os.environ if env is None else env
    need(re.fullmatch(r"[0-9a-f]{40}", prepared_parent or "") is not None,
         "explicit reviewed B21 predecessor required")
    need(all(e.get(k) == v for k, v in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "jiying2007/kws-data",
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": RUN_BRANCH,
        "GITHUB_RUN_ATTEMPT": "1"}.items()), "first public B21 branch event only")
    need(re.fullmatch(r"[0-9a-f]{40}", e.get("GITHUB_SHA", "")) is not None
         and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")) is not None,
         "actual B21 run identity")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent
         and event.get("after") == e["GITHUB_SHA"] and event.get("ref") == RUN_BRANCH
         and event.get("repository", {}).get("full_name") == "jiying2007/kws-data"
         and event.get("repository", {}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "exact prepared parent, public repository and no replay/forced push")
    return dict(run_id=e["GITHUB_RUN_ID"], head=e["GITHUB_SHA"], parent=parent)


def fetch(inputs, prepared_parent):
    guard(prepared_parent)
    read_design()
    b15.fetch(inputs)
    pin = b17.baseline.MODELS["B0"]
    for name, p in (("B0", dict(run_id=pin["run"], head=pin["head"], artifact_id=pin["id"],
                               bytes=pin["bytes"], sha256=pin["sha"])), ("b17", B17_ARCHIVE)):
        run = json.loads(b13.gh("actions/runs/" + str(p["run_id"])))
        need(run["head_sha"] == p["head"] and run["conclusion"] == "success"
             and run["run_attempt"] == 1, "authentic completed " + name + " source run")
        raw = b13.gh("actions/artifacts/" + str(p["artifact_id"]) + "/zip")
        need(len(raw) == p["bytes"] and b10.digest(raw) == p["sha256"], "exact " + name + " archive")
        (inputs / (name + ".zip")).write_bytes(raw)
    (inputs / "B21_DESIGN.json").write_bytes(DESIGN_FILE.read_bytes())
    (inputs / "B21_ASSESSMENT.json").write_bytes(ASSESSMENT.read_bytes())


def competitors(target, arm):
    need(type(target) is int and target in (1, 2, 3) and arm in ("control", "challenger"),
         "registered positive token and paired arm required")
    return [k for k in ((1, 2, 3) if arm == "control" else (0, 1, 2, 3)) if k != target]


def scalar_retention(student, teacher, target, arm):
    need(len(student) == len(teacher) == 4
         and all(math.isfinite(v) for v in (*student, *teacher)), "finite four-class logits")
    other = competitors(target, arm)
    return max(0.0, (teacher[target] - max(teacher[k] for k in other))
                   - (student[target] - max(student[k] for k in other)))


def shared_schedule(groups, extra):
    """Cover every original/prefix using width alone, never any model score."""
    gids = sorted(groups[0])
    need(gids == sorted(groups[1]) and len(gids) == 35, "same 35 paired TRAIN source groups")
    widths = {g: max(len(groups[k][g]) for k in (0, 1)) for g in gids}
    need(Counter(widths.values()) == {1: 26, 2: 9}, "fixed 79-original structural source roster")
    # Five complete passes, then all nine two-original groups and eight singleton
    # groups. Sorting uses group width and lexical identity, never a score.
    order = gids * 5 + sorted(gids, key=lambda g: (-widths[g], g))[:17]
    draws = b13.paired.freeze_draw(extra, STEPS)
    visits, slots = Counter(), []
    for step, (gid, natural) in enumerate(zip(order, draws)):
        b17.ordinary_row(natural)
        visit = visits[gid]
        prefix = b10.PREFIXES[(gids.index(gid) + visit // widths[gid]) % len(b10.PREFIXES)]
        slot = dict(step=step + 1, natural=natural, prefix=prefix,
                    source_group_id=gid, group_visit=visit, group_width=widths[gid])
        for klass, key in ((0, "ordinary"), (1, "positive")):
            rows = sorted(groups[klass][gid], key=lambda r: r["id"])
            slot[key] = rows[visit % len(rows)]
        b17.ordinary_row(slot["ordinary"])
        visits[gid] += 1
        slots.append(slot)
    need(len(slots) == STEPS, "exactly 192 common structural-coverage draws")
    schedule_coverage(slots, groups, extra)
    return slots


def schedule_coverage(slots, groups, extra):
    need(len(slots) == STEPS, "192 actual source draws")
    expected = {r["id"]: r for k in (0, 1) for rows in groups[k].values() for r in rows}
    need(len(expected) == 79, "all 79 original synthetic TRAIN sources")
    exposed, natural = defaultdict(set), defaultdict(set)
    for slot in slots:
        need(slot["prefix"] in b10.PREFIXES, "registered silence prefix")
        for kind in ("ordinary", "positive"):
            row = slot[kind]
            need(row["role"] == "train" and row == expected.get(row["id"]),
                 "exact source identity and TRAIN role in common draw")
            exposed[row["id"]].add(slot["prefix"])
        row = slot["natural"]
        need(row in extra, "only nine reviewed natural rows")
        natural[row["id"]].add(slot["prefix"])
    need(set(exposed) == set(expected) and all(v == set(b10.PREFIXES) for v in exposed.values()),
         "every synthetic original receives all three registered prefixes")
    need(set(natural) == {r["id"] for r in extra}
         and all(v == set(b10.PREFIXES) for v in natural.values()),
         "every reviewed natural original receives all three registered prefixes")
    return dict(synthetic_originals=len(exposed), natural_originals=len(natural),
                synthetic_original_prefix_pairs=sum(map(len, exposed.values())),
                natural_original_prefix_pairs=sum(map(len, natural.values())),
                synthetic_prefixes={k: sorted(v) for k, v in sorted(exposed.items())},
                natural_prefixes={k: sorted(v) for k, v in sorted(natural.items())},
                all_originals_all_three_prefixes=True, score_driven_sampling=False)


def saved_train_mechanism(archive, np):
    """Authenticate and inspect saved TRAIN logits only: no model import/forward."""
    raw = Path(archive).read_bytes()
    need(len(raw) == B17_ARCHIVE["bytes"] and b10.digest(raw) == B17_ARCHIVE["sha256"],
         "original completed B17 training artifact")
    with zipfile.ZipFile(io.BytesIO(raw)) as ar:
        need(ar.testzip() is None, "B17 original archive CRC")
        report_raw, array_raw, source_raw = (ar.read(n) for n in
            ("TRAIN_DIAGNOSTIC.json", "TRAIN_DIAGNOSTIC_LOGITS.npz", "SOURCE.json"))
    need(b10.digest(report_raw) == B17_DIAGNOSTIC_SHA
         and b10.digest(array_raw) == B17_ARRAYS_SHA and b10.digest(source_raw) == B17_SOURCE_SHA,
         "byte-frozen original B17 TRAIN arrays and source receipt")
    report, source = json.loads(report_raw), json.loads(source_raw)
    need(report["dev_cal_locked_forwards"] == 0 and report["optimizer_updates"] == 0
         and source["forbidden_sources_opened"] is False and len(report["records"]) == 88,
         "preoptimizer TRAIN-only diagnostic origin")
    eligible = {(r["id"], str(float(r["gain"]))): r for r in source["positive_retention"]}
    need(len(eligible) == 120 and len({k[0] for k in eligible}) == 43
         and len({r["group"] for r in eligible.values()}) == 34,
         "same original 120 positive teacher paths, not an error-selected subset")
    cases, seen, expected_arrays = [], set(), set()
    with np.load(io.BytesIO(array_raw), allow_pickle=False) as arrays:
        for row in report["records"]:
            need(row["role"] == "train" and row["id"] not in seen
                 and type(row["positive"]) is bool, "unique TRAIN-only original identity")
            seen.add(row["id"])
            for gain, state in row["gains"].items():
                matrices = {}
                need(gain in {"1.0", "0.75", "0.5"}, "three registered original gains")
                for role in ("B0", "B9"):
                    pin = state[role]
                    m = arrays[pin["array"]]
                    expected_arrays.add(pin["array"])
                    need(m.ndim == 2 and m.shape[1] == 4 and len(m) > 0 and np.isfinite(m).all()
                         and b10.digest(m.tobytes()) == pin["sha256"], "finite complete authentic TRAIN array")
                    need(bool(b15.completed_word(m, np)) == bool(pin["events"]), "saved word-event readback")
                    matrices[role] = m
                need(matrices["B0"].shape == matrices["B9"].shape, "same original native frame grid")
                key = (row["id"], gain)
                if not row["positive"] or key not in eligible or state["B9"]["events"] == 0 or state["B0"]["events"] != 0:
                    continue
                accepted = eligible[key]
                need(accepted["group"] == row["group"] and accepted["pcm_sha256"] == row["pcm_sha256"]
                     and accepted["b9_array_sha256"] == state["B9"]["sha256"], "same admitted positive TRAIN path")
                spans = b15.completed_word(matrices["B9"], np)
                blank_only = []
                for token, start, end in spans:
                    a = matrices["B0"][start:end]
                    other = competitors(token, "control")
                    mask = (a[:, token] > a[:, other].max(axis=1)) & (a[:, token] < a[:, 0])
                    blank_only.append(dict(token=token, start=start, end=end,
                                           blank_only_frame_indices=(np.flatnonzero(mask) + start).tolist()))
                count = sum(len(s["blank_only_frame_indices"]) for s in blank_only)
                cases.append(dict(id=row["id"], group=row["group"], gain=gain,
                                  pcm_sha256=row["pcm_sha256"], B0_array_sha256=state["B0"]["sha256"],
                                  B9_array_sha256=state["B9"]["sha256"], spans=blank_only,
                                  blank_only_frames=count, qualifying=count > 0))
        need(set(arrays.files) == expected_arrays and len(expected_arrays) == 528,
             "all 528 original arrays checked, no diagnostic array substitution")
    groups = sorted({r["group"] for r in cases if r["qualifying"]})
    passed = len(groups) >= 2
    return dict(schema="english-v6-b21-saved-train-blank-mechanism-v1", source=B17_ARCHIVE,
        diagnostic_sha256=B17_DIAGNOSTIC_SHA, arrays_sha256=B17_ARRAYS_SHA,
        source_manifest_sha256=B17_SOURCE_SHA, authenticated_arrays=528, records=cases,
        eligible_positive_paths=120, B9_hit_B0_miss_conditions=len(cases),
        qualifying_conditions=sum(r["qualifying"] for r in cases), qualifying_groups=groups,
        blank_only_frame_count=sum(r["blank_only_frames"] for r in cases),
        minimum_qualifying_source_groups=2, optimizer_start_allowed=passed,
        status="TRAIN_SIGNAL" if passed else "STOP_TRAIN_HYPOTHESIS_UNSUPPORTED",
        new_model_forwards=0, optimizer_updates=0, new_tts_asr_calls=0,
        dev_cal_locked_forwards=0, promotion_allowed=False,
        limits="Mechanism evidence only; no claim of causation for any previous DEV result or future quality pass.")


def prepare_paths(inputs, rows, waves, frontend, old_design, np, torch):
    control, audit = b15.prepare_paths(inputs, rows, waves, frontend, old_design, np, torch)
    with zipfile.ZipFile(inputs / "diagnostic.zip") as ar:
        doc = json.loads(ar.read("RESULT.json"))
        raw = ar.read("ALL_FRAME_LOGITS.npz")
    need(b10.digest(raw) == doc["logits_sha256"], "same frozen B9 positive arrays")
    records = {r["id"]: r for r in doc["records"]}
    audit_index = defaultdict(list)
    for item in audit:
        audit_index[(item["group"], item["id"])].append(item)
    challenger = defaultdict(dict)
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        for group, originals in control.items():
            for uid, conditions in originals.items():
                paired = []
                need(len(conditions) == len(audit_index[(group, uid)]), "paired original/gain cardinality")
                for (x, old_spans), pin in zip(conditions, audit_index[(group, uid)]):
                    matrix = arrays[records[uid]["gains"][pin["gain"]]["B9"]["array"]]
                    need(b10.digest(matrix.tobytes()) == pin["b9_array_sha256"], "exact same B9 source/gain")
                    aligned = []
                    for token, start, end, old_other, _ in old_spans:
                        need(old_other == competitors(token, "control"), "control competitors unchanged")
                        other = competitors(token, "challenger")
                        ref = matrix[start:end, token] - matrix[start:end][:, other].max(axis=1)
                        need(np.all(ref >= 0) and len(ref) > 0, "same complete B9 positive nonblank span")
                        aligned.append((token, start, end, other, torch.from_numpy(ref.copy())))
                    paired.append((x, aligned))
                challenger[group][uid] = paired
    need(control.keys() == challenger.keys(), "same positive source groups in both arms")
    return {"control": control, "challenger": challenger}, audit


def retention_loss(by_group, weight, trace, torch):
    """Same group/original/gain/token averaging with explicit frozen references."""
    groups = sorted(by_group)
    def calculate(step, model):
        group, original_losses = groups[step % len(groups)], []
        for uid in sorted(by_group[group]):
            gain_losses = []
            for x, spans in by_group[group][uid]:
                matrix = model(x)[0]
                token_losses = []
                for token, start, end, other, reference in spans:
                    part = matrix[start:end]
                    margin = part[:, token] - part[:, other].max(dim=1).values
                    token_losses.append(torch.relu(reference.detach() - margin).mean())
                gain_losses.append(torch.stack(token_losses).mean())
            original_losses.append(torch.stack(gain_losses).mean())
        base = torch.stack(original_losses).mean()
        need(bool(torch.isfinite(base)), "finite B21 positive path margin loss")
        trace.append(dict(step=step + 1, group=group, eligible_originals=len(original_losses),
                          unweighted_loss=float(base.detach())))
        return base * weight
    return calculate


def fit(arm, initial, slots, base_cache, natural_cache, positive_paths, teachers,
        frontend, output, torch, np, F):
    need(arm in ("control", "challenger"), "registered pair only")
    torch.manual_seed(SEED)
    net = b10.b6.b4.v5.build_model(torch)
    net.load_state_dict(initial, strict=True)
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.001)
    positive_trace, trace = [], []
    retain = retention_loss(positive_paths[arm], POSITIVE_WEIGHT, positive_trace, torch)
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
            row, pair = slot[kind], {}
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
        natural_losses, row = [], slot["natural"]
        for gain in b10.GAINS:
            loss, logp = b10.b6.suffix_loss(net, natural_cache[row["id"]][gain][prefix], 0,
                prefix, row["tier"], torch, F, True)
            natural_losses.append(loss)
            ordinary_outputs.append((row, gain, prefix, logp))
        positive = retain(step, net)
        ceiling = b17.ceiling_loss(ordinary_outputs, teachers, torch)
        loss = (sum(losses) / 4 + b10.b6.WEIGHT * sum(consistency) / 2
                + NATURAL_WEIGHT * sum(natural_losses) / 2 + positive + CEILING_WEIGHT * ceiling)
        need(bool(torch.isfinite(loss)), "finite fixed B21 objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        trace.append(dict(step=step + 1, prefix=prefix, ordinary_id=slot["ordinary"]["id"],
                          positive_id=slot["positive"]["id"], natural_id=slot["natural"]["id"],
                          loss=float(loss.detach()), positive_retention=float(positive.detach()),
                          ordinary_ceiling=float(ceiling.detach()), ceiling_coefficient=CEILING_WEIGHT,
                          positive_competitors="other_two_nonblank" if arm == "control" else "other_three_including_blank"))
    need(len(trace) == len(positive_trace) == STEPS, "192 actual optimizer updates per arm")
    net.eval()
    directory = output / "models"
    directory.mkdir()
    stem = "mero43-b21-" + arm + "192"
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
    counts["checkpoint_reload_parity"] += 1
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
    need(all(path.stat().st_size > 10000 for path in (pt, npz, onnx)), "actual full model exports")
    b10.write(output / "OPTIMIZER_TRACE.json", trace)
    b10.write(output / "POSITIVE_RETENTION_TRACE.json", positive_trace)
    return dict(stem=arm + "/models/" + stem, optimizer_updates=STEPS,
                parent_pt_sha256=B0_PT, natural_loss_weight=NATURAL_WEIGHT,
                positive_retention_weight=POSITIVE_WEIGHT, ordinary_ceiling_weight=CEILING_WEIGHT,
                positive_competitors="other_two_nonblank" if arm == "control" else "other_three_including_blank",
                forward_counts=dict(counts), optimizer_seconds=round(time.perf_counter() - started, 3),
                files={p.name: dict(bytes=p.stat().st_size, sha256=b10.digest(p.read_bytes()))
                       for p in (pt, npz, onnx)}), state



def validate_pair_receipt(doc):
    """Validate trained-pair shape/identity; bytes need independent file hashes."""
    need(doc.get("schema") == SCHEMA
         and doc.get("status") == "TRAINED_PAIR_REQUIRES_FRESH_INDEPENDENT_DEV"
         and doc.get("actual_optimizer_updates") == 384
         and doc.get("trained_model_parameters") == 43876
         and doc.get("parent_pt_sha256") == B0_PT
         and doc.get("baseline_originals") == 79 and doc.get("natural_originals") == 9
         and doc.get("positive_path_combinations") == 120
         and doc.get("dev_cal_locked_forwards") == 0
         and doc.get("actual_new_diagnostic_forwards") == 0
         and doc.get("promotion_allowed") is False
         and doc.get("numerical_B17_replay_claimed") is False,
         "actual B21 TRAIN-only paired receipt")
    for name in ("design_sha256", "assessment_sha256", "common_draw_sha256",
                 "draw_coverage_sha256", "train_mechanism_sha256"):
        need(re.fullmatch(r"[0-9a-f]{64}", doc.get(name, "")) is not None, "frozen B21 receipt digest " + name)
    need(doc["assessment_sha256"] == doc.get("independent_assessment_sha256"), "one independent assessment")
    for arm in ("control", "challenger"):
        a = doc.get(arm, {})
        stem = "mero43-b21-" + arm + "192"
        need(a.get("stem") == arm + "/models/" + stem and a.get("optimizer_updates") == 192
             and a.get("parent_pt_sha256") == B0_PT
             and a.get("ordinary_ceiling_weight") == CEILING_WEIGHT
             and a.get("positive_retention_weight") == POSITIVE_WEIGHT
             and a.get("natural_loss_weight") == NATURAL_WEIGHT
             and a.get("positive_competitors") == ("other_two_nonblank" if arm == "control" else "other_three_including_blank"),
             "B21 paired arm and single changed positive competitor set")
        need(set(a.get("files", {})) == {stem + "." + ext for ext in ("pt", "npz", "onnx")}, "three actual B21 exports")
        for pin in a["files"].values():
            need(type(pin.get("bytes")) is int and pin["bytes"] > 10000
                 and re.fullmatch(r"[0-9a-f]{64}", pin.get("sha256", "")) is not None, "B21 actual artifact byte pin")
    return doc


def execute(inputs, output, prepared_parent):
    identity = guard(prepared_parent)
    read_design()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(), "one new output, no overwrite")
    need((inputs / "B21_DESIGN.json").read_bytes() == DESIGN_FILE.read_bytes()
         and (inputs / "B21_ASSESSMENT.json").read_bytes() == ASSESSMENT.read_bytes(),
         "fetched and current preoptimizer registrations identical")
    output.mkdir()
    b10.write(output / "STARTED.json", dict(schema=SCHEMA, **identity, maximum_optimizer_updates=384,
               design_sha256=DESIGN_SHA, assessment_sha256=ASSESSMENT_SHA,
               new_tts_asr_calls=0, dev_cal_locked_forwards=0, promotion_allowed=False))
    import numpy as np
    mechanism = saved_train_mechanism(inputs / "b17.zip", np)
    b10.write(output / "TRAIN_MECHANISM.json", mechanism)
    if not mechanism["optimizer_start_allowed"]:
        b10.write(output / "RESULT.json", dict(schema=SCHEMA, **identity, status=mechanism["status"],
                   actual_optimizer_updates=0, actual_new_model_forwards=0,
                   dev_cal_locked_forwards=0, new_tts_asr_calls=0, promotion_allowed=False))
        return
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    old_design = b15.read_design()
    need(b10.digest((inputs / "diagnostic.zip").read_bytes()) == old_design["diagnostic"]["sha256"]
         and b10.digest((inputs / "pair.zip").read_bytes()) == old_design["historical_b13"]["sha256"],
         "original positive teacher and historical source archives")
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
    slots = shared_schedule(groups, extra)
    need(len(historical_slots) == len(slots), "same bounded 192-step exposure")
    for slot, historical in zip(slots, historical_slots):
        need(slot["natural"]["id"] == historical["id"]
             and slot["natural"]["source_pcm_sha256"] == historical["source_pcm_sha256"],
             "unchanged reviewed natural original draw; shared prefixes intentionally repaired")
    b10.write(output / "COMMON_DRAW.json", slots)
    b10.write(output / "DRAW_COVERAGE.json", schedule_coverage(slots, groups, extra))
    b0net, _b9net, frontend = b17.load_parents(inputs, torch, np)
    cache = b13.paired.make_features(rows, waves, frontend, torch, np)
    extra_cache = b13.paired.make_features(extra, extra_waves, frontend, torch, np)
    positive_paths, audit = prepare_paths(inputs, rows, waves, frontend, old_design, np, torch)
    teachers, teacher_count = b17.make_ordinary_teachers(slots, cache, extra_cache, b0net, output, torch, np)
    need(teacher_count == 264, "all 44 ordinary originals at three prefixes and two gains, shared by both arms")
    b10.write(output / "SOURCE.json", dict(base_train=source["base_train"], natural_train=extra,
               positive_retention=audit, registered_optimizer=REGISTRATION,
               independent_assessment_sha256=ASSESSMENT_SHA, design_sha256=DESIGN_SHA,
               parent_pt_sha256=B0_PT, positive_teacher_pt_sha256=b10.b9.MODELS["B9_control"]["pt"],
               input_zip_sha256={p.name: b10.digest(p.read_bytes()) for p in sorted(inputs.glob("*.zip"))},
               forbidden_sources_opened=False, numerical_B17_replay_claimed=False,
               shared_schedule_repaired=True))
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
         "identical exposure and 384 paired updates produce different learned tensors")
    result = dict(schema=SCHEMA, status="TRAINED_PAIR_REQUIRES_FRESH_INDEPENDENT_DEV",
               **identity, **arms, registration=REGISTRATION, parent_pt_sha256=B0_PT,
               trained_model_parameters=43876, actual_optimizer_updates=384,
               different_trained_float_values=differing, seed=SEED, common_learning_rate=LR,
               design_sha256=DESIGN_SHA, assessment_sha256=ASSESSMENT_SHA,
               independent_assessment_sha256=ASSESSMENT_SHA,
               common_draw_sha256=b10.digest((output / "COMMON_DRAW.json").read_bytes()),
               draw_coverage_sha256=b10.digest((output / "DRAW_COVERAGE.json").read_bytes()),
               train_mechanism_sha256=b10.digest((output / "TRAIN_MECHANISM.json").read_bytes()),
               actual_new_diagnostic_forwards=0, reused_saved_diagnostic_arrays=528,
               actual_ordinary_teacher_forwards=teacher_count,
               actual_student_forwards=sum(sum(a["forward_counts"].values()) for a in arms.values()),
               positive_path_combinations=len(audit), positive_originals=43, positive_families=34,
               baseline_originals=79, natural_originals=9, natural_families=8,
               dev_cal_locked_forwards=0, new_tts_asr_calls=0, promotion_allowed=False,
               numerical_B17_replay_claimed=False, physical_ssc305="NOT_RUN", field_far_per_hour="NOT_MEASURED")
    validate_pair_receipt(result)
    b10.write(output / "RESULT.json", result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("fetch", "execute", "saved-mechanism"))
    ap.add_argument("inputs", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--prepared-parent")
    args = ap.parse_args()
    if args.mode == "saved-mechanism":
        import numpy as np
        report = saved_train_mechanism(args.inputs, np)
        if args.output:
            need(not args.output.exists(), "saved mechanism report cannot overwrite an existing report")
            b10.write(args.output, report)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
    elif args.mode == "fetch":
        fetch(args.inputs, args.prepared_parent)
    else:
        need(args.output is not None, "output directory required")
        execute(args.inputs, args.output, args.prepared_parent)
