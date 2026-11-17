#!/usr/bin/env python3
"""Independent standard-library audit of one acoustic-coverage research arm.

Authenticates immutable inputs; reconstructs new PCM conditions independently;
checks exact cache reuse, training draws, frozen-candidate order and saved
Python/C traces. No neural network, gradient, matrix product, C executable,
tensor framework, external process or network is executed by this auditor.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

FROZEN_DENSE_SHA = "5c772633ac89773650f7c9c64a224334d9903ff812d8fc82c8e09ecbdc7685e1"
FROZEN_DENSE_BYTES = 36900
PLAN_FIXED_SEMANTIC_SHA256 = "1a8a20cd678e1854ec009af8279ba241822d74f26d8807be891aa0da1c12d3ec"
SOURCE_HASH_KEYS = ("worker_sha256", "workflow_sha256", "common_worker_sha256", "data_helper_sha256")
DOMAIN = "english-coverage-new-acoustic-conditions-20261117-v1"
DATA_SCHEMA = "english-coverage-condition-data-v1"
PURPOSES = {
    "train": {"role": "train", "conditions": (("new5", 5.0), ("new15", 15.0)),
              "sources": 50, "planned_conditions": 100, "planned_rows": 3666, "maximum_new_embedding_calls": 2066},
    "development_challenge": {"role": "development", "conditions": (("new8", 8.0),),
              "sources": 27, "planned_conditions": 27, "planned_rows": 1001, "maximum_new_embedding_calls": 569},
}


def load_frozen_dense():
    paths = [parent / relative for parent in Path(__file__).resolve().parents for relative in
             ("english_dense/audit/audit_dense.py", "english_dense_quality_20261117/audit/audit_dense.py")]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise ValueError("frozen dense auditor absent from sibling research directory")
    raw = path.read_bytes()
    if len(raw) != FROZEN_DENSE_BYTES or hashlib.sha256(raw).hexdigest() != FROZEN_DENSE_SHA:
        raise ValueError("frozen dense auditor SHA/bytes changed")
    spec = importlib.util.spec_from_file_location("_coverage_frozen_dense_auditor_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dense = load_frozen_dense()
legacy = dense.legacy
require, ident, digest, canonical = legacy.require, legacy.ident, legacy.digest, legacy.canonical
FALSE_FLAGS, SHAPES = legacy.FALSE_FLAGS, legacy.SHAPES


def scalar_mix(signal_raw, noise_raw, target):
    """Independent scalar PCM arithmetic; preserve exclusions without repair."""
    require(target in (5.0, 8.0, 15.0), "only prespecified new SNR values")
    signal, noise = legacy.pcm(signal_raw), legacy.pcm(noise_raw)
    require(len(signal) == len(noise), "same complete clip/noise length")
    n = len(signal)
    ps = math.fsum(float(v) ** 2 for v in signal) / n
    pn = math.fsum(float(v) ** 2 for v in noise) / n
    receipt = {"snr_definition": "full-original-clip-rms-before-mixture", "target_snr_db": target,
        "samples": n, "signal_power": ps, "noise_power": pn,
        "source_pcm_sha256": digest(signal_raw), "noise_pcm_sha256": digest(noise_raw),
        "gain_on_signal": 1.0, "clipped_samples": 0, "fade_samples": 0}
    if ps == 0 or pn == 0:
        return None, {**receipt, "status": "MIX_ZERO_ENERGY_EXCLUDED"}
    gain = math.sqrt(ps / (pn * 10 ** (target / 10)))
    mixed = [round(float(s) + gain * float(n)) for s, n in zip(signal, noise)]
    receipt.update(noise_gain=gain, unbounded_peak=max(abs(v) for v in mixed))
    if min(mixed) < -32768 or max(mixed) > 32767:
        return None, {**receipt, "status": "MIX_HEADROOM_EXCLUDED"}
    residual = math.fsum(float(v - s) ** 2 for v, s in zip(mixed, signal)) / n
    if residual == 0:
        return None, {**receipt, "status": "MIX_QUANTIZED_NOISE_ZERO_EXCLUDED"}
    actual = 10 * math.log10(ps / residual)
    if abs(actual - target) > 0.05:
        return None, {**receipt, "status": "MIX_QUANTIZATION_SNR_EXCLUDED", "actual_snr_db": actual}
    raw = struct.pack("<" + str(n) + "h", *mixed)
    return raw, {**receipt, "status": "MIXED", "actual_snr_db": actual, "mixed_pcm_sha256": digest(raw)}


def select_new_noise(parent, purpose, condition, frames, tracks, excluded):
    require(purpose in PURPOSES and condition in dict(PURPOSES[purpose]["conditions"]),
            "condition/purpose fixed before evaluation")
    role = PURPOSES[purpose]["role"]
    pool = sorted((t for t in tracks if t["role"] == role and t["frames"] >= frames
                   and t["id"] not in excluded), key=lambda t: t["id"])
    require(pool and len({t["id"] for t in pool}) == len(pool), "nonempty same-role distinct track pool")
    key = digest((DOMAIN + "/" + purpose + "/" + condition + "/" + parent).encode())
    track = pool[int(key[:16], 16) % len(pool)]
    return track, int(key[16:32], 16) % (track["frames"] - frames + 1)


def new_condition_id(parent, purpose, condition, source_sha, track, offset):
    request = {"schema": DATA_SCHEMA, "domain": DOMAIN, "parent_id": parent, "purpose": purpose,
        "condition": condition, "source_pcm_sha256": source_sha, "noise_track": track["id"],
        "noise_track_pcm_sha256": track["pcm_sha256"], "noise_start_sample": offset,
        "target_snr_db": dict(PURPOSES[purpose]["conditions"])[condition]}
    return "encov-" + condition + "-" + digest(canonical(request))[:24]


def prefix_window_proof(clean, mixed):
    require(len(clean) == len(mixed) and len(clean) >= 70400 and clean[:70400] == mixed[:70400],
            "same original complete prefix before sample 35200")
    proofs = []
    for end in range(16000, 35201, 1280):
        original = clean[2 * (end - 16000):2 * end]
        condition = mixed[2 * (end - 16000):2 * end]
        require(len(original) == 32000 and original == condition, "exact one-second frontend PCM input")
        proofs.append({"available_sample": end, "input_pcm_bytes": 32000, "input_pcm_sha256": digest(original)})
    require(len(proofs) == 16, "exact sixteen eligible prefix cache rows")
    return proofs


def training_coverage(rows):
    groups = {}
    for condition in ("new5", "new15"):
        for label in range(3):
            group = [r for r in rows if r["condition"] == condition and r["weak_class"] == label]
            require(all(r["role"] == "train" for r in group), "coverage never admits a development parent")
            sources = {r["parent_id"] for r in group}
            families = sorted({r["mix"]["source_family_id"] for r in group})
            texts = sorted({r["mix"]["source_synthesis_text"] for r in group})
            accepted = len(sources) >= 2 and len(families) >= 2 if label else len(families) == 4 and len(texts) == 8
            groups[condition + "/" + str(label)] = {"sources": len(sources), "families": families,
                "source_texts": texts, "coverage_passed": accepted}
    return {"passed": all(g["coverage_passed"] for g in groups.values()), "groups": groups,
        "minimum_positive_source_count": 2, "minimum_positive_family_count": 2,
        "negative_all_four_families_and_eight_templates_required": True}


def reconstruct_conditions(training, incumbent, purpose):
    """Never import the worker/helper; derive PCM and receipts from pinned inputs.

    development_challenge is called only while auditing completed post-freeze
    evidence, never by this module's preflight path.
    """
    require(purpose in PURPOSES, "fixed condition purpose")
    spec = PURPOSES[purpose]
    roster = training.json("STREAM_ROSTER.json")
    selected = set(training.json("SELECTION.json")["eligible_cell_ids"])
    parents = [r for r in roster if r["id"] in selected]
    require(len(parents) == len(selected) == 77, "unchanged original source denominator")
    require(collections.Counter((r["role"], r["weak_class"]) for r in parents) ==
            {("train", 0): 32, ("train", 1): 8, ("train", 2): 10,
             ("development", 0): 16, ("development", 1): 3, ("development", 2): 8}, "unchanged roles/classes")
    chosen = [r for r in parents if r["role"] == spec["role"]]
    require(len(chosen) == spec["sources"], "purpose-specific original parents")
    metadata, original = legacy.original_audio(training, {r["id"] for r in chosen})
    libri = training.json("LIBRISPEECH_SOURCES.json")
    tracks = libri["actual_research_tracks"]
    track_map = legacy.unique_rows(tracks, "original background tracks")
    require(len(tracks) == 40 and collections.Counter(t["role"] for t in tracks) == {"train": 30, "development": 10}
            and libri["source_pool_is_previously_exposed"] is True, "fixed exposed background-role partition")
    manifest = incumbent.json("background_tracks/MANIFEST.json")["files"]
    backgrounds = {}
    for track in tracks:
        name = track["id"] + ".pcm"
        raw, pin = incumbent.read("background_tracks/" + name), manifest[name]
        require(ident(raw) == {k: pin[k] for k in ("bytes", "sha256")}
                and len(raw) == track["frames"] * 2 == 512000
                and digest(raw) == track["pcm_sha256"]
                and pin["role"] == track["role"] and pin["speaker"] == track["speaker"], "pinned full background track")
        backgrounds[track["id"]] = raw
    old_receipts = incumbent.json("evidence/MIX_ROSTER.json")
    old_map = legacy.unique_rows(old_receipts, "old 10 dB receipts")
    require(len(old_receipts) == 77 and set(old_map) == selected, "complete original mix dispositions")
    receipts, eligible, streams = [], [], {}
    for parent in chosen:
        pid, src = parent["id"], parent["source"]
        raw, meta, old = original[pid], metadata[pid], old_map[pid]
        require(digest(raw) == src["source_pcm_sha256"] == old["source_pcm_sha256"]
                and parent["role"] == meta["role"] == old["role"]
                and meta["source_family_id"] == old["source_family_id"]
                and parent["weak_class"] == old["weak_class"], "immutable parent/source/role/intent")
        spans = {}
        for kind in ("prefix", "context"):
            track = track_map[src[kind + "_track"]]
            start, count = src[kind + "_start_sample"], src[kind + "_samples"]
            require(track["role"] == src[kind + "_role"] == parent["role"]
                    and 0 <= start < start + count <= track["frames"], "same-role original context range")
            spans[kind] = backgrounds[track["id"]][2 * start:2 * (start + count)]
        clean = spans["prefix"] + raw + spans["context"]
        require(len(spans["prefix"]) == 70400 and len(spans["context"]) == 10240
                and len(clean) == parent["observed_samples"] * 2
                and digest(clean) == src["combined_pcm_sha256"] == old["original_combined_pcm_sha256"],
                "reconstructed original clean stream byte identity")
        excluded = {old["noise_track"]}
        require(track_map[old["noise_track"]]["role"] == parent["role"], "original noise role")
        for condition, snr in spec["conditions"]:
            track, offset = select_new_noise(pid, purpose, condition, len(raw) // 2, tracks, excluded)
            excluded.add(track["id"])
            noise = backgrounds[track["id"]][2 * offset:2 * offset + len(raw)]
            cid = new_condition_id(pid, purpose, condition, digest(raw), track, offset)
            mixed, receipt = scalar_mix(raw, noise, snr)
            receipt.update(id=cid, condition_id=cid, parent_id=pid, condition=condition, purpose=purpose,
                role=parent["role"], weak_class=parent["weak_class"], source_family_id=meta["source_family_id"],
                source_synthesis_text=meta["synthesis_text"], source_variant=meta["variant"],
                canonical_phrase=meta["canonical_phrase"], target_ipa=meta["target_ipa"],
                noise_track=track["id"], noise_role=track["role"], noise_start_sample=offset,
                noise_track_pcm_sha256=track["pcm_sha256"], old_snr10_noise_track=old["noise_track"],
                old_snr10_noise_start_sample=old["noise_start_sample"], old_snr10_receipt_sha256=digest(canonical(old)),
                original_combined_pcm_sha256=digest(clean), new_noise_domain=DOMAIN,
                prefix_pcm_sha256=digest(spans["prefix"]), context_pcm_sha256=digest(spans["context"]),
                prefix_samples=35200, context_samples=5120, independent_source_count_increment=0,
                condition_training_allowed=(purpose == "train"), old_snr5_cache_used_for_training=False,
                **dict.fromkeys(FALSE_FLAGS, False))
            if mixed is not None:
                full = spans["prefix"] + mixed + spans["context"]
                receipt.update(combined_mixed_pcm_sha256=digest(full), prefix_cache_rows_reused=16,
                               prefix_input_windows=prefix_window_proof(clean, full))
                eligible.append({**parent, "id": cid, "condition_id": cid, "parent_id": pid,
                                 "condition": condition, "mix": receipt})
                require(cid not in streams, "unique independently reconstructed condition")
                streams[cid] = full
            receipts.append(receipt)
    require(len(receipts) == len({r["id"] for r in receipts}) == spec["planned_conditions"]
            and not ({r["id"] for r in receipts} & selected), "complete disjoint new condition IDs")
    planned_rows = sum(r["records"] for r in chosen) * len(spec["conditions"])
    actual_rows = sum(r["records"] for r in eligible)
    calls = actual_rows - 16 * len(eligible)
    require(planned_rows == spec["planned_rows"] and 0 <= calls <= spec["maximum_new_embedding_calls"],
            "new suffix calls within frozen geometry ceiling")
    counts = lambda rows: dict(sorted(collections.Counter(r["condition"] + "/" + r["role"] + "/" +
                                                          str(r["weak_class"]) for r in rows).items()))
    summary = {"schema": DATA_SCHEMA, "status": "FIXED_NEW_CONDITIONS_CONSTRUCTED", "purpose": purpose,
        "role": spec["role"], "noise_domain": DOMAIN, "source_cases": spec["sources"],
        "planned_conditions": len(receipts), "eligible_conditions": len(eligible),
        "excluded_conditions": len(receipts) - len(eligible), "planned_embedding_rows": planned_rows,
        "eligible_embedding_calls": actual_rows, "eligible_embedding_rows": actual_rows,
        "prefix_cache_rows_reused": 16 * len(eligible), "new_embedding_calls_required": calls,
        "maximum_new_embedding_calls": spec["maximum_new_embedding_calls"], "onnx_forward_calls_required": 2 * calls,
        "planned_by_condition_role_class": counts(receipts), "eligible_by_condition_role_class": counts(eligible),
        "excluded_statuses": dict(sorted(collections.Counter(r["status"] for r in receipts if r["status"] != "MIXED").items())),
        "condition_ids": [r["id"] for r in receipts], "eligible_condition_ids": [r["id"] for r in eligible],
        "receipts_semantic_sha256": digest(canonical(receipts)), "roster_semantic_sha256": digest(canonical(eligible)),
        "original_training_source": ident(training.raw), "incumbent_source": ident(incumbent.raw),
        "same_source_condition_is_not_independent_audio": True, "independent_source_count_increment": 0,
        "new_tts_asr_phone_calls": 0, "old_snr5_cache_used_for_training": False,
        "development_challenge_training_allowed": False, "condition_training_allowed": purpose == "train",
        "overflow_policy": "EXCLUDE_WITHOUT_REPLACEMENT_OR_CLIPPING", **dict.fromkeys(FALSE_FLAGS, False)}
    if purpose == "train":
        summary["training_coverage"] = training_coverage(eligible)
    return summary, receipts, eligible, streams


def expected_draws(clean, snr10, train_new):
    groups = {("clean", label): [r for r in clean if r["role"] == "train" and r["weak_class"] == label]
              for label in range(3)}
    for condition in ("snr10", "new5", "new15"):
        pool = snr10 if condition == "snr10" else [r for r in train_new if r["condition"] == condition]
        require(condition == "snr10" or all(r["role"] == "train" for r in pool), "no development parent in new training cache")
        for label in range(3):
            group = [r for r in pool if r["role"] == "train" and r["weak_class"] == label]
            require(group, "nonempty original train-role noisy group")
            groups[condition, label] = group
    draws = []
    for step in range(96):
        condition, occurrence = ("snr10", "new5", "new15")[step % 3], step // 3
        used = []
        for name, index in (("clean", step), (condition, occurrence)):
            for label in range(3):
                group = groups[name, label]
                draw = {"condition": name, "id": group[index % len(group)]["id"],
                        "weak_class": label, "role": "train"}
                if name != "clean":
                    draw["condition_occurrence"] = occurrence
                used.append(draw)
        draws.append(used)
    return draws


def verify_coverage_draws(draws, previous, clean, snr10, new):
    require(len(draws) == len(previous) == 96 and draws == expected_draws(clean, snr10, new),
            "fixed clean/noisy occurrence-indexed 96 by 6 draw sequence")
    require(all(step[:3] == old[:3] for step, old in zip(draws, previous)), "all 288 original clean draws unchanged")
    distribution = collections.Counter((d["condition"], d["weak_class"]) for step in draws for d in step)
    require(distribution == {**{("clean", c): 96 for c in range(3)},
                             **{(k, c): 32 for k in ("snr10", "new5", "new15") for c in range(3)}},
            "exact equal noisy-condition occurrence budgets")
    require(all(d["role"] == "train" and d["condition"] not in ("snr5", "new8") for step in draws for d in step),
            "old 5 dB cache and new challenge excluded from every training draw")
    return {"steps": 96, "bag_forwards": 576, "clean_draws_byte_semantics_unchanged": True,
        "clean_draws": 288, "noisy_condition_class_draws": 32,
        "semantic_sha256": digest(canonical(draws)), "development_draws": 0,
        "legacy_snr5_cache_draws": 0, "new8_draws": 0,
        "all_six_draws_identical_to_old_control": False,
        "draw_id_counts": dict(sorted(collections.Counter(d["condition"] + "/" + d["id"]
             for step in draws for d in step).items()))}


def verify_new_cache(raw, rows, original_rows, original_cache):
    arrays = legacy.npz(raw)
    require(set(arrays) == {f"{p}{i:03d}" for i in range(len(rows)) for p in ("e", "t")}, "closed new cache roster")
    parents = legacy.unique_rows(original_rows, "clean parents")
    clean_arrays = legacy.npz(original_cache)
    payload = clean_arrays["embeddings"]["payload"]
    clocks, prefix_pins = {}, []
    for index, row in enumerate(rows):
        parent = parents[row["parent_id"]]
        times = legacy.expected_clocks(parent)
        values, recorded = arrays[f"e{index:03d}"], arrays[f"t{index:03d}"]
        require(row["records"] == len(times) and values["dtype"] == "<f4" and values["shape"] == (len(times), 96)
                and recorded["shape"] == (len(times),) and legacy.integers(recorded) == times,
                "new stream retains exact parent clock, finite embedding rows and discontinuity boundary")
        begin = parent["first_record"] * 96 * 4
        prior = payload[begin:begin + 16 * 96 * 4]
        require(len(prior) == 6144 and values["payload"][:6144] == prior, "first sixteen embedding rows copied byte-for-byte")
        require(row["mix"]["prefix_cache_rows_reused"] == 16, "PCM proof accompanies prefix reuse")
        clocks[row["id"]] = times
        prefix_pins.append({"id": row["id"], "parent_id": row["parent_id"], "prefix_rows": 16,
                            "embedding_bytes": 6144, "embedding_sha256": digest(prior),
                            "new_suffix_rows": len(times) - 16})
    return clocks, arrays, prefix_pins


def coverage_gate(paired, old_snr5, new8):
    require({p["condition"] for p in paired} == ({"clean", "snr10", "snr5"} | ({"new8"} if new8 else set())),
            "all available paired conditions, with empty challenge retained as a rejection")
    old = [p for p in paired if p["condition"] != "new8"]
    gate = legacy.strict_gate(old, old_snr5)
    extra = [p for p in paired if p["condition"] == "new8"]
    coverage = {str(c): sum(r["role"] == "development" and r["weak_class"] == c for r in new8) for c in (1, 2)}
    reasons = list(gate["reasons"])
    if any(p["regressed"] for p in extra):
        reasons.append("NEW8_CASE_REGRESSION")
    if any(p["introduced_event_errors"] for p in extra):
        reasons.append("NEW8_PREFIX_WRONG_EXTRA_OR_NEGATIVE_EVENT")
    if min(coverage.values()) < 1:
        reasons.append("NEW8_DEVELOPMENT_KEYWORD_COVERAGE_INSUFFICIENT")
    complete = (len(new8) == len({r["parent_id"] for r in new8}) == 27 and
                collections.Counter((r["role"], r["weak_class"]) for r in new8) ==
                {("development", 0): 16, ("development", 1): 3, ("development", 2): 8})
    if not complete:
        reasons.append("NEW8_PLANNED_DEVELOPMENT_COVERAGE_INCOMPLETE")
    gate.update(accepted=not reasons, decision="RESEARCH_CANDIDATE_RETAINED" if not reasons else "RESEARCH_CANDIDATE_REJECTED",
        reasons=reasons, paired_case_regressions=sum(p["regressed"] for p in paired),
        new_event_error_cases=sum(bool(p["introduced_event_errors"]) for p in paired),
        new8_development_positive_coverage=coverage,
        new8_all_27_planned_development_cases_eligible=complete)
    return gate


def coverage_retention(gate, paired):
    original_pairs = [p for p in paired if p["condition"] != "new8"]
    retention = legacy.expected_retention_record(gate, original_pairs)
    retention["criteria"].update(
        new8_no_previously_correct_case_regression=not any(p["regressed"] for p in paired if p["condition"] == "new8"),
        new8_both_development_keywords_present=min(gate["new8_development_positive_coverage"].values()) >= 1,
        new8_all_27_planned_development_cases_eligible=gate["new8_all_27_planned_development_cases_eligible"])
    return retention


def compare_c_stdout(raw, reference):
    """Compare saved C output with already validated Python frames; never execute C."""
    lines = raw.decode("ascii").splitlines()
    require(len(lines) == len(reference), "complete saved C output row count")
    errors = [0.0, 0.0]
    events = []
    for line, frame in zip(lines, reference):
        fields = line.split()
        require(len(fields) == 9, "saved C output has exactly nine columns")
        sample, ready, keyword = (int(v) for v in fields[:3])
        require(ready in (0, 1) and (sample, bool(ready), keyword) ==
                (frame["sample"], frame["ready"], frame["keyword"]), "exact C clock/readiness/event parity")
        numbers = [float(v) for v in fields[3:]]
        require(all(math.isfinite(v) for v in numbers), "finite C logits/probabilities")
        errors[0] = max(errors[0], *(abs(a - b) for a, b in zip(numbers[:3], frame["logits"])))
        errors[1] = max(errors[1], *(abs(a - b) for a, b in zip(numbers[3:], frame["probabilities"])))
        if keyword:
            events.append({"sample": sample, "keyword": keyword})
    require(errors[0] <= 1e-4 and errors[1] <= 2e-6, "fixed native/ARM saved-logit/probability tolerance")
    return {"input_embeddings": len(reference), "head_outputs": sum(f["ready"] for f in reference),
            "max_abs_logit_error": errors[0], "max_abs_probability_error": errors[1], "events": events}


def source_only_coverage_contract(raw):
    """Static comparison of the sole train loop; never import the producer."""
    tree = ast.parse(raw.decode("utf-8"))
    worker = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker"), None)
    require(worker is not None, "single explicit numerical worker")
    def lexical_nodes(node):
        yield node
        if node is not worker and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        for child in ast.iter_child_nodes(node):
            yield from lexical_nodes(child)
    scope = list(lexical_nodes(worker))
    assignments = {}
    for node in scope:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments.setdefault(node.targets[0].id, []).append(node.value)
    wanted = {
        "windows": 'np.asarray([matrix[positions[t] - 15:positions[t] + 1] for t in ends], dtype=np.float32)',
        "normalized": '((windows - weights["mean96"]) / weights["scale96"]).reshape(len(windows), 1536)',
        "model": 'torch.nn.Sequential(torch.nn.Linear(1536, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))',
        "optimizer": 'torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)',
        "logp": 'torch.log_softmax(model(prepared[key]), dim=-1)',
        "loss": 'torch.stack(terms).mean()',
    }
    for name, text in wanted.items():
        nodes = assignments.get(name, [])
        expected = ast.dump(ast.parse(text, mode="eval").body, include_attributes=False)
        require(len(nodes) == 1 and ast.dump(nodes[0], include_attributes=False) == expected,
                "fixed original training expression: " + name)
    branches = [n for n in scope if isinstance(n, ast.If) and ast.unparse(n.test) == "label == 0"]
    require(len(branches) == 1, "one unchanged original weak-intent loss branch")
    for nodes, expression in ((branches[0].body, "terms.append(-logp[:, 0].mean())"),
        (branches[0].orelse, "terms.append(math.log(len(logp)) - torch.logsumexp(logp[:, label], dim=0))")):
        require([ast.dump(n, include_attributes=False) for n in nodes] ==
                [ast.dump(n, include_attributes=False) for n in ast.parse(expression).body],
                "mean target probability original positive loss and unchanged background CE")
    calls = [ast.unparse(n) for n in scope if isinstance(n, ast.Call)]
    for expression in ("torch.set_num_threads(1)", "torch.manual_seed(20261117)",
        "torch.use_deterministic_algorithms(True)", "optimizer.zero_grad(set_to_none=True)",
        "loss.backward()", "optimizer.step()",
        "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)"):
        require(calls.count(expression) == 1, "unchanged single training operation: " + expression)
    loops = [n for n in scope if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step"]
    require(len(loops) == 1 and ast.unparse(loops[0].iter) == "range(96)", "one fixed 96-update training loop")
    preparation = [n for n in scope if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "prepare_new"]
    freeze = [n for n in scope if isinstance(n, ast.Call) and ast.unparse(n) == "phase('CANDIDATE_FROZEN')"]
    require(len(preparation) == 2 and len(freeze) == 1 and [ast.unparse(n) for n in preparation] ==
            ["prepare_new('train')", "prepare_new('development_challenge')"]
            and preparation[0].lineno < loops[0].lineno < loops[0].end_lineno < freeze[0].lineno < preparation[1].lineno,
            "new challenge materialization is lexically after the sole completed training and candidate freeze")
    evaluation = [n for n in scope if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "evaluate"]
    require([ast.unparse(n) for n in evaluation] == [
        "evaluate('coverage-clean', candidate, roster, clean_streams)",
        "evaluate('coverage-snr10', candidate, snr10_roster, snr10_streams)",
        "evaluate('coverage-snr5', candidate, snr5_roster, snr5_streams)",
        "evaluate('aug96-new8', control, new8_roster, new8_streams, is_new8=True)",
        "evaluate('coverage-new8', candidate, new8_roster, new8_streams, is_new8=True)"]
        and all(n.lineno > freeze[0].lineno for n in evaluation), "only fixed candidate old264 and two new8 head passes")
    all_calls = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    require(not any(c.startswith(("common.worker(", "common.execute(", "portable.check_parity(")) for c in all_calls),
            "neither consumed old worker nor reference-recomputing C checker may execute")
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    require(not any((isinstance(n, ast.Import) and any(a.name.split(".")[0] in
                    {"numpy", "torch", "onnx", "onnxruntime"} for a in n.names)) or
                    (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in
                    {"numpy", "torch", "onnx", "onnxruntime"}) for n in imports),
            "no numerical import before guarded public worker")
    return {"source_sha256": digest(raw), "original_positive_mean_probability_loss_checked": True,
        "unchanged_geometry_normalization_optimizer_checked": True, "one_fixed_96_update_loop": True,
        "consumed_worker_and_extra_python_c_reference_forbidden": True,
        "static_inspection_only": True, "local_numerical_loss_or_gradient_execution": False}


def input_preflight(training, incumbent, temporal):
    rows, clocks, bags = dense.authenticated_inputs(training, incumbent, temporal)
    summary, receipts, new_rows, _ = reconstruct_conditions(training, incumbent, "train")
    require(summary["training_coverage"]["passed"] is True, "all prespecified training source/family/template coverage minima")
    draws = expected_draws(rows["clean"], rows["snr10"], new_rows)
    checked = verify_coverage_draws(draws, incumbent.json("evidence/TRAINING.json")["draws"],
                                   rows["clean"], rows["snr10"], new_rows)
    return {"schema": "independent-coverage-input-preflight-v1", "status": "TRAIN_CONDITIONS_AND_FROZEN_INPUTS_VERIFIED",
        "source_training_zip": ident(training.raw), "source_incumbent_zip": ident(incumbent.raw),
        "source_temporal_cache_zip": ident(temporal.raw), "frozen_dense_auditor_sha256": FROZEN_DENSE_SHA,
        "frozen_temporal_auditor_sha256": dense.FROZEN_AUDITOR_SHA,
        "train_condition_summary": summary, "train_receipts": receipts,
        "training_draws": checked, "positive_supported_bags": bags,
        "new8_condition_materialization": "NOT_RUN_BEFORE_CANDIDATE_FREEZE",
        "new8_old_parent_clock_ceiling": {"cases": 27, "embedding_rows": 1001, "new_suffix_rows": 569},
        "maximum_new_embedding_calls": 2635, "maximum_onnx_forwards": 5270,
        "maximum_python_head_case_passes": 318, "maximum_c_head_cases_per_backend": 291,
        "new_model_calls_by_this_auditor": 0, "independent_source_count_increment": 0,
        "same_source_condition_is_not_independent_audio": True, **dict.fromkeys(FALSE_FLAGS, False)}


def expected_counts(train_summary, new8_summary):
    train = train_summary["new_embedding_calls_required"]
    new8 = new8_summary["new_embedding_calls_required"]
    cases, new_rows = new8_summary["eligible_conditions"], new8_summary["eligible_embedding_rows"]
    require(0 <= train <= 2066 and 0 <= new8 <= 569 and cases <= 27 and new_rows <= 1001,
            "all actual counts remain within original preregistered ceilings")
    total, ready = train + new8, new_rows - 15 * cases
    counts = {"new_embedding_calls_started": total, "new_embedding_calls_completed": total,
        "onnx_forward_started": 2 * total, "onnx_forward_completed": 2 * total,
        "head_case_passes_started": 264 + 2 * cases, "head_case_passes_completed": 264 + 2 * cases,
        "head_ready_outputs": 11847 + 2 * ready,
        "optimizer_updates_started": 96, "optimizer_updates_completed": 96,
        "torch_forward_calls_started": 576, "torch_forward_calls_completed": 576,
        "new_training_frontend_calls": train, "new_tts_asr_phone_calls": 0, "retry_count": 0,
        "train_new_embedding_calls_started": train, "train_new_embedding_calls_completed": train,
        "new8_embedding_calls_started": new8, "new8_embedding_calls_completed": new8,
        "new8_head_case_passes_started": 2 * cases, "new8_head_case_passes_completed": 2 * cases,
        "prefix_embedding_rows_reused": train_summary["prefix_cache_rows_reused"] + new8_summary["prefix_cache_rows_reused"],
        "new_c_frontend_calls": 0}
    for backend in ("host", "arm"):
        for kind, count in (("builds", 1), ("head_processes", 1), ("head_case_passes", 264 + cases),
                            ("embedding_rows", 15807 + new_rows)):
            for endpoint in ("started", "completed"):
                counts[f"{backend}_c_{kind}_{endpoint}"] = count
        counts[f"{backend}_c_head_ready_outputs"] = 11847 + ready
    return counts


def verify_coverage_phases(phases, frozen, final, candidate_sha, train_summary, new8_summary):
    calculated = expected_counts(train_summary, new8_summary)
    require(final == calculated and all(type(v) is int and v >= 0 for v in final.values()),
            "exact completed Python/frontend/training/native/ARM counters")
    names = ["WORKER_STARTED", "TRAIN_MIX_DATA_PREPARATION_STARTED", "TRAIN_MIX_DATA_PREPARATION_COMPLETED",
        "TRAIN_MIX_FEATURES_STARTED", "TRAIN_MIX_FEATURES_COMPLETED", "TRAINING_STARTED", "CANDIDATE_FROZEN",
        "OLD_CONDITION_EVALUATION_STARTED", "OLD_CONDITION_EVALUATION_COMPLETED",
        "NEW8_DATA_PREPARATION_STARTED", "NEW8_DATA_PREPARATION_COMPLETED", "NEW8_FEATURES_STARTED",
        "NEW8_FEATURES_COMPLETED", "NEW8_EVALUATION_STARTED", "NEW8_EVALUATION_COMPLETED",
        "C_HEAD_PARITY_PREPARATION_STARTED", "C_HEAD_PARITY_PREPARATION_COMPLETED",
        "HOST_C_PARITY_STARTED", "HOST_C_PARITY_COMPLETED", "ARM_C_PARITY_STARTED",
        "ARM_C_PARITY_COMPLETED", "COMPLETED"]
    require([p["phase"] for p in phases] == names and
            [p["phase_number"] for p in phases] == list(range(1, len(names) + 1)), "closed 22-phase once-only execution")
    state, checkpoints = dict.fromkeys(final, 0), []
    train_calls = train_summary["new_embedding_calls_required"]
    for name in names:
        if name == "TRAIN_MIX_FEATURES_COMPLETED":
            for key in ("new_embedding_calls_started", "new_embedding_calls_completed",
                        "train_new_embedding_calls_started", "train_new_embedding_calls_completed", "new_training_frontend_calls"):
                state[key] = train_calls
            state["onnx_forward_started"] = state["onnx_forward_completed"] = 2 * train_calls
            state["prefix_embedding_rows_reused"] = train_summary["prefix_cache_rows_reused"]
        elif name == "CANDIDATE_FROZEN":
            for key in ("optimizer_updates_started", "optimizer_updates_completed"):
                state[key] = 96
            for key in ("torch_forward_calls_started", "torch_forward_calls_completed"):
                state[key] = 576
        elif name == "OLD_CONDITION_EVALUATION_COMPLETED":
            state["head_case_passes_started"] = state["head_case_passes_completed"] = 264
            state["head_ready_outputs"] = 11847
        elif name == "NEW8_FEATURES_COMPLETED":
            for key in ("new_embedding_calls_started", "new_embedding_calls_completed", "onnx_forward_started",
                        "onnx_forward_completed", "new8_embedding_calls_started", "new8_embedding_calls_completed",
                        "prefix_embedding_rows_reused"):
                state[key] = final[key]
        elif name == "NEW8_EVALUATION_COMPLETED":
            for key in ("head_case_passes_started", "head_case_passes_completed", "head_ready_outputs",
                        "new8_head_case_passes_started", "new8_head_case_passes_completed"):
                state[key] = final[key]
        elif name in ("HOST_C_PARITY_COMPLETED", "ARM_C_PARITY_COMPLETED"):
            prefix = "host_c_" if name.startswith("HOST") else "arm_c_"
            for key in final:
                if key.startswith(prefix):
                    state[key] = final[key]
        checkpoints.append(dict(state))
    for index, phase in enumerate(phases):
        require(phase["progress"] == checkpoints[index] and all(type(v) is int for v in phase["progress"].values()),
                "phase counts prohibit early challenge, replay, cross-backend or extra calls: " + phase["phase"])
        require(phase["candidate_sha256"] == (None if index < 6 else candidate_sha), "same candidate after freeze")
    require(frozen["phase_number"] == 7 and frozen["progress"] == checkpoints[6]
            and frozen["candidate"]["sha256"] == candidate_sha and frozen["new8_model_calls_before_freeze"] == 0
            and frozen["new8_preparation_before_freeze"] is False
            and frozen["all_future_evaluation_uses_this_frozen_head"] is True,
            "new8 data creation and every evaluation follow fixed candidate freeze")
    return {"phases": len(names), "candidate_frozen_phase": 7, "new8_first_preparation_phase": 10,
        "new8_first_feature_phase": 12, "new8_first_head_evaluation_phase": 14,
        "native_first_head_phase": 18, "arm_first_head_phase": 20,
        "source_and_receipt_order_not_external_runtime_attestation": True}


def verify_saved_c_parity(quality, training, candidate_pin, candidate_binary_pin, rows, clocks, arrays):
    """Rebuild only stream bytes and global-clock mapping from cached features."""
    reference, mapping, chunks = [], [], []
    origin = cursor = 0
    for condition in ("clean", "snr10", "snr5", "new8"):
        frames = [legacy.json_bytes(line) for line in quality.read("predictions/coverage-" + condition + "-frames.jsonl").splitlines()]
        frame_cursor = 0
        for i, row in enumerate(rows[condition]):
            times = clocks[condition][row["id"]]
            mapping.append({"condition": condition, "id": row["id"], "parent_id": row.get("parent_id", row["id"]),
                "first_record": cursor, "records": len(times), "offset": origin,
                "observed_samples": row["observed_samples"], "head_outputs": len(times) - 15})
            if condition == "clean":
                begin = row["first_record"] * 384
                payload = arrays["clean"]["embeddings"]["payload"][begin:begin + len(times) * 384]
            else:
                payload = arrays[condition][f"e{i:03d}"]["payload"]
            for index, local in enumerate(times):
                frame = frames[frame_cursor]
                frame_cursor += 1
                require(frame["id"] == row["id"] and frame["sample"] == local, "saved Python reference local identity")
                global_sample = origin + local
                chunks.append(struct.pack("<QI", global_sample, int(index == 0)) + payload[index * 384:(index + 1) * 384])
                reference.append({**frame, "condition": condition, "local_sample": local,
                                  "sample": global_sample, "case_index": len(mapping) - 1})
            cursor += len(times)
            origin += row["observed_samples"]
        require(frame_cursor == len(frames), "complete saved reference-to-cache assignment")
    raw_stream = b"EHSTR01\0" + struct.pack("<I", cursor) + b"".join(chunks)
    require(cursor <= 16808 and len(mapping) <= 291 and len(raw_stream) == 12 + 396 * cursor,
            "bounded original C head ABI stream")
    require(quality.read("runtime/head-stream.bin") == raw_stream, "every C input float/clock/reset exactly matches saved cache")
    require(quality.json("runtime/CASE_MAP.json") == mapping, "closed same candidate ordered C case mapping")
    recorded_reference = [legacy.json_bytes(line) for line in quality.read("runtime/reference-frames.jsonl").splitlines()]
    require(recorded_reference == reference, "C reference is saved Python result with only explicit global-clock translation")
    c_names = ("english_causal_head.c", "english_causal_head.h", "head_replay.c")
    hashes = {name: digest(training.read("english-snapshot160-embedding-research-bundle/runtime/" + name)) for name in c_names}
    for name, expected in hashes.items():
        require(digest(quality.read("sources/" + name)) == expected, "unchanged original C head source: " + name)
    reports = {}
    for backend, tag, machine, elf_class in (("host", "native", 62, 2), ("arm", "arm", 40, 1)):
        report = quality.json("evidence/" + backend.upper() + "_PARITY.json")
        raw = quality.read("evidence/head-" + tag + ".stdout")
        actual = compare_c_stdout(raw, reference)
        require(all(report[k] == v for k, v in actual.items()), "independently calculated saved C parity errors and events")
        expected = {"schema": "english-coverage-" + backend + "-c-head-parity-v1", "passed": True,
            "scope": "CANDIDATE_CACHED_EMBEDDING_HEAD_ONLY", "cases": len(mapping),
            "candidate": candidate_pin, "candidate_binary": candidate_binary_pin,
            "stream_binary": ident(raw_stream), "raw_stdout": ident(raw),
            "reference_frames": ident(quality.read("runtime/reference-frames.jsonl")),
            "case_map": ident(quality.read("runtime/CASE_MAP.json")), "source_hashes": hashes,
            "executable": ident(quality.read("runtime/head-" + tag)), "exact_samples": True,
            "exact_ready": True, "exact_events": True, "threshold": 0.8, "confirm_embeddings": 3,
            "cooldown_embeddings": 13, "independent_python_head_reexecution": False,
            "c_frontend": "NOT_RUN", "ssc305": "NOT_RUN", "formal_qualification": False, "ssc305_qualification": False,
            **dict.fromkeys(FALSE_FLAGS, False)}
        require(all(report[k] == v for k, v in expected.items()), "C proof binds candidate, caches, source, ABI and fixed gate")
        executable = quality.read("runtime/head-" + tag)
        require(len(executable) >= 20 and executable[:4] == b"\x7fELF" and executable[4] == elf_class
                and executable[5] == 1 and struct.unpack_from("<H", executable, 18)[0] == machine,
                "actual saved native/ARM ELF machine")
        compile_args = report["compiler_argv"]
        prefix = ["cc" if backend == "host" else "arm-linux-gnueabihf-gcc", "-std=c99", "-O2", "-ffp-contract=off",
                  "-fno-fast-math", "-Wall", "-Wextra", "-Werror"] + ([] if backend == "host" else ["-static"])
        require(compile_args[:len(prefix)] == prefix and len(compile_args) == len(prefix) + 5
                and compile_args[-3:-1] == ["-lm", "-o"]
                and [Path(p).name for p in compile_args[len(prefix):len(prefix) + 2]] == ["english_causal_head.c", "head_replay.c"]
                and Path(compile_args[-1]).name == "head-" + tag, "fixed floating-point C compiler flags and two original sources")
        runner = report["runner_argv"]
        require(len(runner) == (3 if backend == "host" else 4) and (backend == "host" or runner[0] == "qemu-arm")
                and [Path(v).name for v in runner[-3:]] == ["head-" + tag, "coverage-head.bin", "head-stream.bin"],
                "one candidate-only cached stream invocation per backend")
        for command in ("build-head-" + tag, "head-" + tag):
            exit_record = quality.json("evidence/" + command + ".exit.json")
            require(exit_record["returncode"] == 0 and exit_record["error"] is None
                    and exit_record["group_cleanup_checked_after_leader_exit"] is True
                    and exit_record["argv"] == (compile_args if command.startswith("build-") else runner),
                    "successful supervised compile/replay with descendants accounted for")
        require(set(report["tool_versions"]) == {"host-compiler-version", "arm-compiler-version", "qemu-version"},
                "saved compiler/QEMU identities")
        for name, pin in report["tool_versions"].items():
            require(pin == ident(quality.read("evidence/" + name + ".stdout")) and pin["bytes"] > 0,
                    "nonempty preserved tool version receipt")
        reports[backend] = {**expected, **actual, "compared_locally_without_executing_c_or_head": True}
    require(reports["host"]["events"] == reports["arm"]["events"], "both C backends agree on every saved event")
    return reports


def verify_plan(plan):
    """Pin all 88 fixed fields, including nested inputs and exact strong gates."""
    require(len(plan) == 92 and all(k in plan for k in SOURCE_HASH_KEYS), "closed 92-field coverage protocol")
    fixed = {k: v for k, v in plan.items() if k not in SOURCE_HASH_KEYS}
    require(len(fixed) == 88 and digest(canonical(fixed)) == PLAN_FIXED_SEMANTIC_SHA256,
            "complete preregistered protocol is unchanged, including 27/27 challenge and old10 rescue gate")
    for key in SOURCE_HASH_KEYS:
        require(isinstance(plan[key], str) and len(plan[key]) == 64 and set(plan[key]) <= set("0123456789abcdef"),
                "source contract SHA-256: " + key)
    require(plan["common_worker_sha256"] == "15389e1101871352e6e7477c893f1dadf019fd57b572fa1388ec53d1dd8b0988",
            "immutable old common helpers are not a new numerical arm")
    legacy.reject_formal_claims(plan)


def artifact_boundary(quality, plan, result, expected_run, expected_head):
    verify_plan(plan)
    require(result["schema"] == plan["schema"] == "english-coverage-quality-once-v1"
            and result["status"] == "COMPLETED_WEAK_COVERAGE_QUALITY_RESEARCH"
            and result["run_attempt"] == 1, "completed first-attempt public coverage run")
    require(expected_run is not None and str(result["run_id"]) == str(expected_run)
            and expected_head is not None and result["head_sha"] == expected_head,
            "explicit external public run/commit binding required for actual audit")
    require("evidence/FAILED_NO_RETRY.json" not in quality.names, "no incomplete or retried numerical result")
    for name in quality.names:
        if name.endswith(".json"):
            legacy.reject_formal_claims(quality.json(name), name)
    manifest = quality.json("evidence/WORKER_MANIFEST.json")
    require(manifest["excludes_downloads_and_later_supervisor_files"] is True
            and manifest["later_files"] == ["evidence/model-worker.exit.json", "evidence/PARENT_RESULT.json", "evidence/SUPERVISOR_EXIT.json"],
            "exact worker manifest boundary")
    for name, pin in manifest["files"].items():
        require(name in quality.names and ident(quality.read(name)) == pin, "manifest byte identity: " + name)
    require(quality.names - set(manifest["files"]) == {"evidence/WORKER_MANIFEST.json", *manifest["later_files"]},
            "all artifact members assigned to worker or known supervisor receipts")
    supervisor = quality.json("evidence/SUPERVISOR_EXIT.json")
    require(supervisor["retry_count"] == 0 and supervisor["numeric_attempt"] == 1
            and supervisor["maximum_seconds"] == 900 and 0 <= supervisor["wall_seconds"] <= 900,
            "finite once-only supervised execution")
    worker_exit = quality.json("evidence/model-worker.exit.json")
    require(worker_exit["returncode"] == 0 and worker_exit["error"] is None
            and worker_exit["group_cleanup_checked_after_leader_exit"] is True, "worker and descendants completed")
    parent = quality.json("evidence/PARENT_RESULT.json")
    require(parent["status"] == "COMPLETED_ONE_SUPERVISED_WORKER" and parent["retry_count"] == 0
            and parent["numeric_attempt"] == 1 and parent["worker_result"] == ident(quality.read("evidence/RESULT.json")),
            "parent binds exactly one saved worker result")
    started = quality.json("evidence/STARTED.json")
    require(started["run_id"] == str(expected_run) and started["head_sha"] == expected_head
            and started["run_attempt"] == 1 and started["retry_count"] == 0 and started["numeric_attempt"] == 1,
            "start and completion are the same first-attempt public run")
    return len(manifest["files"]), supervisor["wall_seconds"]


def copied_source_contract(quality, training, incumbent, temporal, plan, result):
    inputs = quality.json("evidence/INPUTS.json")
    require(inputs["old_embedding_rows_reused"] == {"clean": 10354, "snr10": 2804, "snr5": 2649}
            and inputs["old_frontend_recomputation"] is False
            and inputs["new_training_frontend_calls"] == "NEW_SUFFIX_ONLY"
            and inputs["new_evaluation_frontend_calls"] == "NEW8_SUFFIX_ONLY_AFTER_FREEZE"
            and inputs["failed_temporal_head_used_as_warmstart"] is False, "old caches and fixed original warmstart")
    sources = {"training": training, "incumbent": incumbent, "temporal": temporal}
    for label, source in sources.items():
        require(inputs["actual_" + label] == ident(source.raw)
                and inputs[label + "_source"] == plan[label + "_source"] == result[label + "_source"],
                "immutable source ZIP identity: " + label)
    copied = inputs["copied_members"]
    dense.verify_copied_members(quality, copied, sources)
    required = {"sources/" + name: (training, name) for name in ("STREAM_ROSTER.json", "FEATURE_EVIDENCE.json",
        "FEATURE_PREFLIGHT.json", "PER_CASE_EVENTS.json", "LIBRISPEECH_SOURCES.json", "SELECTION.json")}
    required.update({"cache/clean-original.npz": (training, "actual-embedding-streams.npz"),
        "models/baseline-head.npz": (training, "train/causal-head.npz"),
        "models/aug96-control-head.npz": (incumbent, "models/candidate-head.npz"),
        "cache/mixed-snr10.npz": (incumbent, "cache/mixed-embeddings.npz"),
        "cache/snr10-roster.json": (incumbent, "cache/mixed-roster.json"),
        "cache/snr5-embeddings.npz": (temporal, "cache/snr5-embeddings.npz"),
        "cache/snr5-roster.json": (temporal, "cache/snr5-roster.json"),
        "evidence/SNR5_RECEIPTS.json": (temporal, "evidence/SNR5_RECEIPTS.json"),
        "evidence/SNR5_SUMMARY.json": (temporal, "evidence/SNR5_SUMMARY.json"),
        "sources/PREVIOUS_TEMPORAL_RESULT.json": (temporal, "evidence/RESULT.json"),
        "sources/PREVIOUS_TRAINING.json": (incumbent, "evidence/TRAINING.json"),
        "sources/PREVIOUS_RESULT.json": (incumbent, "evidence/RESULT.json"),
        "sources/PREVIOUS_RUNTIME.json": (incumbent, "evidence/RUNTIME.json"),
        "sources/PREVIOUS_MIX_RECEIPTS.json": (incumbent, "evidence/MIX_ROSTER.json")})
    bundle = "english-snapshot160-embedding-research-bundle/"
    for name in ("weak_embedding_contract.py", "weak_audio_cases.py", "causal_embedding.py", "causal_head.py",
                 "portable_head.py", "english_causal_head.c", "english_causal_head.h", "head_replay.c"):
        required["sources/" + name] = (training, bundle + "runtime/" + name)
    for name in ("melspectrogram.onnx", "embedding_model.onnx"):
        required["models/" + name] = (training, bundle + "models/" + name)
    for name in ("kws-pipeline-LICENSE", "openWakeWord-LICENSE"):
        required["sources/" + name] = (training, bundle + "licenses/" + name)
    for name in incumbent.names:
        if name.startswith("predictions/"):
            required["predictions/reused/" + Path(name).name] = (incumbent, name)
        elif name.startswith("background_tracks/"):
            required[name] = (incumbent, name)
    for stem in ("baseline-snr5", "aug96-snr5"):
        for suffix in ("-frames.jsonl", "-events.json"):
            name = "predictions/" + stem + suffix
            required["predictions/reused/" + stem + suffix] = (temporal, name)
    require(set(copied) == set(required), "closed exact original cache/model/source/prediction copy table")
    for destination, (source, member) in required.items():
        require(quality.read(destination) == source.read(member), "mandatory original byte-for-byte reuse: " + destination)
    names = {"worker_sha256": "english_coverage_quality_once.py", "workflow_sha256": "english-coverage-quality-once.yml",
             "common_worker_sha256": "english_temporal_quality_once.py", "data_helper_sha256": "english_coverage_quality_data.py"}
    for key, name in names.items():
        raw = quality.read("sources/" + name)
        require(digest(raw) == plan[key] and {k: inputs["source_contract"][key][k] for k in ("bytes", "sha256")} == ident(raw),
                "actual executed worker/helper/workflow source binding")
    require(quality.read("sources/english_temporal_quality_once.py") == temporal.read("sources/english_temporal_quality_once.py")
            and digest(quality.read("sources/english_temporal_quality_data.py")) ==
            "c2dfae52eda3a3adda952dfd23a250054540f517d6e6fc516aaf9dd2bc915d88", "immutable transitive helper sources")
    source_review = source_only_coverage_contract(quality.read("sources/english_coverage_quality_once.py"))
    return source_review, len(copied)


def verify_weights(quality, training_receipt, frozen):
    weights, pins = {}, {}
    for label, name in (("baseline", "baseline-head.npz"), ("control", "aug96-control-head.npz"), ("candidate", "coverage-head.npz")):
        raw = quality.read("models/" + name)
        weights[label], pins[label] = legacy.npz(raw), ident(raw)
        require(set(weights[label]) == set(SHAPES) and all(weights[label][k]["dtype"] == "<f4"
                and weights[label][k]["shape"] == shape for k, shape in SHAPES.items()), "fixed finite float32 head geometry")
        require(all(value[0] > 0 for value in struct.iter_unpack("<f", weights[label]["scale96"]["payload"])), "positive normalization scales")
    normalization = {}
    for key in ("mean96", "scale96"):
        raw = weights["baseline"][key]["payload"]
        require(raw == weights["control"][key]["payload"] == weights["candidate"][key]["payload"], "normalization bytes frozen")
        normalization[key] = {"bytes": len(raw), "sha256": digest(raw), "candidate_sha256": digest(raw), "byte_identical": True}
    require(training_receipt["warmstart"] == pins["baseline"] and training_receipt["control"] == pins["control"]
            and training_receipt["candidate"] == frozen["candidate"] == pins["candidate"]
            and training_receipt["normalization"] == frozen["normalization"] == normalization,
            "original baseline warmstart, reference and frozen candidate array identities")
    require(any(weights["candidate"][k]["payload"] != weights["baseline"][k]["payload"] for k in ("w1", "b1", "w2", "b2")),
            "new trained head arrays differ from original starting point")
    binary = b"EHKWS01\0" + b"".join(weights["candidate"][k]["payload"] for k in SHAPES)
    require(len(binary) == 197908 and quality.read("models/coverage-head.bin") == binary and frozen["candidate_bin"] == ident(binary),
            "portable head blob exact ABI and saved array byte order")
    return weights, pins, normalization, binary


def verify_new_conditions(quality, training, incumbent):
    reconstructed, arrays, clocks, prefix, all_reuse = {}, {}, {}, {}, []
    original_rows = training.json("STREAM_ROSTER.json")
    parent_map = legacy.unique_rows(original_rows, "original parents")
    for purpose, tag, cache in (("train", "TRAIN_MIX", "train-new"), ("development_challenge", "NEW8", "new8")):
        summary, receipts, rows, pcm = reconstruct_conditions(training, incumbent, purpose)
        require(quality.json("evidence/" + tag + "_RECEIPTS.json") == receipts,
                "all planned independently reconstructed PCM receipts, including exclusions")
        reported_summary = quality.json("evidence/" + tag + "_SUMMARY.json")
        require(all(reported_summary[k] == v for k, v in summary.items()), "independent complete data summary")
        require(quality.json("cache/" + cache + "-roster.json") == rows, "exact eligible inherited-source roster and order")
        for key, raw in pcm.items():
            require(quality.read("audio/new/" + key + ".pcm") == raw, "independent new condition PCM bytes")
        local_clocks, local_arrays, proof = verify_new_cache(quality.read("cache/" + cache + "-embeddings.npz"),
            rows, original_rows, training.read("actual-embedding-streams.npz"))
        for row, checked in zip(rows, proof):
            parent = parent_map[row["parent_id"]]
            times = local_clocks[row["id"]]
            all_reuse.append({"condition_id": row["id"], "parent_id": row["parent_id"], "condition": row["condition"],
                "purpose": purpose, "copied_prefix_rows": 16,
                "clean_cache_row_span": [parent["first_record"], parent["first_record"] + 16],
                "prefix_input_windows": row["mix"]["prefix_input_windows"], "prefix_embedding_bytes": 6144,
                "prefix_embedding_sha256": checked["embedding_sha256"], "new_suffix_rows": len(times) - 16,
                "samples": times, "ready_clocks": times[15:]})
        reconstructed[cache] = (summary, receipts, rows)
        arrays[cache], clocks[cache], prefix[cache] = local_arrays, local_clocks, proof
    require(quality.json("evidence/FEATURE_REUSE.json") == all_reuse, "complete PCM-window and float-byte prefix provenance")
    wanted_audio = {"audio/new/" + r["id"] + ".pcm" for _, _, rows in reconstructed.values() for r in rows}
    require({n for n in quality.names if n.startswith("audio/")} == wanted_audio, "no unaccounted newly materialized audio")
    require(reconstructed["train-new"][0]["training_coverage"]["passed"] is True,
            "new training conditions satisfy source/family/template coverage before any optimizer update")
    return reconstructed, arrays, clocks, prefix


def audit(quality, training, incumbent, temporal, expected_run, expected_head):
    rows, clocks, supported_bags = dense.authenticated_inputs(training, incumbent, temporal)
    plan, result = quality.json("evidence/PLAN.json"), quality.json("evidence/RESULT.json")
    manifest_count, wall_seconds = artifact_boundary(quality, plan, result, expected_run, expected_head)
    source_review, copied_count = copied_source_contract(quality, training, incumbent, temporal, plan, result)
    runtime, old_runtime = quality.json("evidence/RUNTIME.json"), incumbent.json("evidence/RUNTIME.json")
    require(all(runtime[k] == old_runtime[k] for k in ("numpy", "torch", "onnx", "onnxruntime", "numeric_threads"))
            and runtime["execution_provider"] == "CPUExecutionProvider" and runtime["soundfile_or_corpus_decoder_used"] is False,
            "same original single-thread numeric runtime and no new corpus decoder")
    require(quality.json("evidence/FEATURE_GEOMETRY.json") == training.json("FEATURE_PREFLIGHT.json")["geometry"],
            "same original frozen stateless frontend geometry")
    constructed, new_arrays, new_clocks, prefix_pins = verify_new_conditions(quality, training, incumbent)
    train_summary, train_receipts, train_rows = constructed["train-new"]
    challenge_summary, challenge_receipts, challenge_rows = constructed["new8"]
    rows["new8"], clocks["new8"] = challenge_rows, new_clocks["new8"]
    training_receipt, frozen = quality.json("evidence/TRAINING.json"), quality.json("evidence/CANDIDATE_FROZEN.json")
    previous_draws = incumbent.json("evidence/TRAINING.json")["draws"]
    draw_check = verify_coverage_draws(training_receipt["draws"], previous_draws, rows["clean"], rows["snr10"], train_rows)
    losses = training_receipt["losses"]
    require(training_receipt["optimizer_updates"] == len(losses) == 96 and training_receipt["training_bag_forwards"] == 576
            and all(type(x) is float and math.isfinite(x) and x >= 0 for x in losses), "one complete finite fixed-budget training arm")
    require(training_receipt["clean_draws_exact_match"] is True
            and training_receipt["augmented_condition_cycle"] == ["snr10", "new5", "new15"]
            and training_receipt["augmented_occurrence_index"] == "step//3"
            and training_receipt["previous_training"] == ident(incumbent.read("evidence/TRAINING.json"))
            and training_receipt["comparison"] == "ONLY_TRAINING_ACOUSTIC_COVERAGE_AND_ITS_PREDECLARED_SAMPLER_CHANGED_EQUAL_UPDATE_BUDGET"
            and training_receipt["development_checkpoint_or_threshold_selection"] is False
            and training_receipt["snr5_used_for_training"] is False and training_receipt["old_snr5_used_for_training"] is False
            and training_receipt["new8_used_for_training"] is False
            and training_receipt["new_training_frontend_calls"] == train_summary["new_embedding_calls_required"]
            and training_receipt["same_pool_objective_search_remains_closed"] is True,
            "only new train acoustic coverage and its fixed sampler changed; old5/new8 remain evaluation-only")
    for key in ("positive_loss", "background_loss", "bag_aggregation"):
        require(training_receipt[key] == plan[key], "fixed original weak-intent objective: " + key)
    eligible_groups = {}
    for condition in ("clean", "snr10", "new5", "new15"):
        pool = rows[condition] if condition in rows else [r for r in train_rows if r["condition"] == condition]
        for label in range(3):
            eligible_groups[condition + "/" + str(label)] = [r["id"] for r in pool if r["role"] == "train" and r["weak_class"] == label]
    require(training_receipt["eligible_groups"] == eligible_groups, "complete eligible training groups in immutable original order")
    feature_map = legacy.unique_rows(training.json("FEATURE_EVIDENCE.json"), "original head support")
    geometry = []
    for condition, pool in (("clean", rows["clean"]), ("snr10", rows["snr10"]), (None, train_rows)):
        for row in pool:
            if row["role"] != "train":
                continue
            parent = row.get("parent_id", row["id"])
            ends = feature_map[parent]["head_instance_ends"]
            geometry.append({"condition": condition or row["condition"], "id": row["id"], "parent_id": parent,
                "role": "train", "weak_class": row["weak_class"], "head_instance_ends": ends,
                "triplet_count": len(ends) - 2 if row["weak_class"] else None})
    require(quality.json("evidence/TRAINING_GEOMETRY.json") == geometry,
            "whole original clip-supported positive bags; unchanged real-context negative bag geometry")
    weights, pins, normalization, binary = verify_weights(quality, training_receipt, frozen)
    metrics, new_traces, reused_traces = {}, [], []
    for condition in ("clean", "snr10", "snr5", "new8"):
        name = "coverage-" + condition
        metrics["coverage_" + condition], check = legacy.read_trace(quality, name, rows[condition], clocks[condition])
        new_traces.append({"name": name, **check})
        if condition == "new8":
            control_name = "aug96-new8"
        else:
            control_name = "reused/" + ("aug96-snr5" if condition == "snr5" else "candidate-" + condition)
        metrics["aug96_" + condition], check = legacy.read_trace(quality, control_name, rows[condition], clocks[condition])
        (new_traces if condition == "new8" else reused_traces).append({"name": control_name, **check})
    expected_new_predictions = {"predictions/" + stem + suffix for stem in
        ("coverage-clean", "coverage-snr10", "coverage-snr5", "coverage-new8", "aug96-new8")
        for suffix in ("-frames.jsonl", "-events.json")}
    require({n for n in quality.names if n.startswith("predictions/") and not n.startswith("predictions/reused/")} ==
            expected_new_predictions, "no extra head/condition prediction passes in evidence")
    counts = quality.json("evidence/PROGRESS.json")
    require(counts == result["new_counts"] == expected_counts(train_summary, challenge_summary)
            and sum(t["case_passes"] for t in new_traces) == counts["head_case_passes_completed"]
            and sum(t["ready_outputs"] for t in new_traces) == counts["head_ready_outputs"],
            "saved traces and independent suffix/update/replay budget arithmetic agree")
    phases = verify_coverage_phases(quality.json("evidence/PHASES.json"), frozen, counts,
                                    pins["candidate"]["sha256"], train_summary, challenge_summary)
    paired = [p for condition in ("clean", "snr10", "snr5", "new8") for p in legacy.compare_cases(
        metrics["aug96_" + condition], metrics["coverage_" + condition], condition)]
    gate = coverage_gate(paired, rows["snr5"], rows["new8"])
    retention = coverage_retention(gate, paired)
    require(quality.json("evidence/RESEARCH_RETENTION.json") == result["research_retention"] == retention,
            "identical stricter old264 and complete27 challenge incumbent-relative gate")
    reported = quality.json("evidence/PAIRED_COMPARISONS.json")
    require(set(reported) == {"clean", "snr10", "snr5", "new8"}, "closed four-condition paired report")
    safety = {"prefix_count": "prefix_events", "wrong_count": "wrong_keyword_after_clip_start",
              "extra_count": "extra_same_keyword_events", "weak_negative_disagreement_count": "weak_negative_intent_events"}
    for condition in reported:
        expected = [p for p in paired if p["condition"] == condition]
        before = legacy.unique_rows(metrics["aug96_" + condition], "paired old control")
        after = legacy.unique_rows(metrics["coverage_" + condition], "paired coverage candidate")
        require(len(reported[condition]) == len(expected), "every paired source condition retained")
        for actual, row in zip(reported[condition], expected):
            wanted = {"id": row["id"], "role": row["role"], "weak_class": row["weak_class"],
                "previous_research_error": row["control_error"], "coverage_research_error": row["candidate_error"],
                "previously_correct_case_regressed": row["regressed"],
                "event_error_categories_increased": [safety[k] for k in safety if k in row["introduced_event_errors"]],
                "development_miss_rescued": row["role"] == "development" and row["weak_class"] > 0
                    and row["control_missing_in_observation"] and not row["candidate_error"],
                "previous_events": before[row["id"]]["events"], "coverage_events": after[row["id"]]["events"]}
            require(actual == wanted, "independent case-level regression, rescue and event-error decisions")
    for key, values in metrics.items():
        require(result[key] == legacy.grouped(values), "independently recomputed research aggregate: " + key)
    arrays = {"clean": legacy.npz(quality.read("cache/clean-original.npz")),
              "snr10": legacy.npz(quality.read("cache/mixed-snr10.npz")),
              "snr5": legacy.npz(quality.read("cache/snr5-embeddings.npz")), "new8": new_arrays["new8"]}
    parity = verify_saved_c_parity(quality, training, pins["candidate"], ident(binary), rows, clocks, arrays)
    require(result["baseline"] == pins["baseline"] and result["aug96_control"] == pins["control"]
            and result["coverage_candidate"] == pins["candidate"] and result["coverage_candidate_binary"] == ident(binary),
            "result and actual saved weights bind the same baseline/control/candidate")
    result_expectations = {
        "planned_new_conditions": 127, "planned_new_training_conditions": 100, "planned_new_development_conditions": 27,
        "eligible_new_training_conditions": len(train_rows), "eligible_new8_conditions": len(challenge_rows),
        "new_training_dispositions": dict(collections.Counter(r["status"] for r in train_receipts)),
        "new8_dispositions": dict(collections.Counter(r["status"] for r in challenge_receipts)),
        "source_synthetic_cases": 77, "source_synthetic_train": 50, "source_synthetic_development": 27,
        "independent_source_count_increment": 0, "new8_new_source_holdout": False,
        "same_source_condition_is_not_independent_audio": True, "cross_batch_voice_family_exposure": True,
        "prior_development_patterns_informed_research_hypothesis": True,
        "baseline_and_aug96_old_condition_predictions_reused": True, "snr5_previous_evaluation_exposed": True,
        "old_snr5_used_for_training": False, "new8_used_for_training": False, "failed_temporal_head_used_as_warmstart": False,
        "frozen_embedding_rows_reused": {"clean": 10354, "snr10": 2804, "snr5": 2649},
        "training_coverage": train_summary["training_coverage"], "same_pool_objective_search_remains_closed": True,
        "subsequent_same_pool_loss_seed_threshold_search_allowed": False, "same_pool_epoch_search_allowed": False,
        "additional_coverage_retries_allowed": False, "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN",
        "word_end_latency": "NOT_MEASURED", "new_head_c_numeric_parity": "PASSED_NATIVE_AND_ARM_QEMU_CACHED_HEAD_ONLY",
        "host_c_parity": ident(quality.read("evidence/HOST_PARITY.json")), "arm_c_parity": ident(quality.read("evidence/ARM_PARITY.json")),
        "full_pcm_c_frontend_chain": "NOT_RUN", "ssc305": "NOT_RUN", "model_delivered_as_product": False,
        **dict.fromkeys(FALSE_FLAGS, False)}
    require(all(result[k] == v and type(result[k]) is type(v) for k, v in result_expectations.items()),
            "all result source counts, stop rules, preserved exclusions and research-only limits verified")
    paired_summary = {}
    for row in paired:
        key = row["condition"] + "/" + row["role"] + "/" + str(row["weak_class"])
        item = paired_summary.setdefault(key, {"cases": 0, "rescued": 0, "regressed": 0, "new_event_error_cases": 0})
        item["cases"] += 1
        item["rescued"] += int(row["rescued"])
        item["regressed"] += int(row["regressed"])
        item["new_event_error_cases"] += bool(row["introduced_event_errors"])
    return {"schema": "english-coverage-independent-audit-v1", "status": "VERIFIED",
        "quality_artifact": ident(quality.raw), "expected_run": str(expected_run), "expected_head": expected_head,
        "expected_external_binding": True, "run_id": result["run_id"], "head_sha": result["head_sha"],
        "auditor_sha256": digest(Path(__file__).read_bytes()), "frozen_dense_auditor_sha256": FROZEN_DENSE_SHA,
        "frozen_temporal_auditor_sha256": dense.FROZEN_AUDITOR_SHA,
        "source_training_zip": ident(training.raw), "source_incumbent_zip": ident(incumbent.raw),
        "source_temporal_cache_zip": ident(temporal.raw), "candidate_head": pins["candidate"], "candidate_binary": ident(binary),
        "class_mapping": {"background": 0, "Hey Vosen": 1, "Hey Mero": 2},
        "supervisor_wall_seconds": wall_seconds, "worker_manifest_files": manifest_count, "source_copy_bindings": copied_count,
        "source_contract": {**source_review, "plan_fixed_semantic_sha256": PLAN_FIXED_SEMANTIC_SHA256,
                            **{k: plan[k] for k in SOURCE_HASH_KEYS}},
        "train_condition_summary": train_summary, "new8_condition_summary": challenge_summary,
        "prefix_cache_bindings": prefix_pins, "positive_supported_bags": supported_bags,
        "training_draws": draw_check, "weights": pins, "normalization": normalization,
        "portable_export_bytes_verified": True, "phases": phases, "counts": counts,
        "new_trace_checks": new_traces, "reused_trace_checks": reused_traces,
        "metrics": {k: legacy.grouped(v, source_kind=True) for k, v in metrics.items()},
        "paired_summary": paired_summary, "paired_cases": paired, "research_gate": gate,
        "research_retention": retention, "parity": parity,
        "new_model_calls_by_this_auditor": 0, "new_c_processes_by_this_auditor": 0,
        "new_audio_mixes_by_this_auditor": 127, "model_forward_numerics_recomputed": False,
        "head_logit_weight_correspondence_independently_recomputed": False,
        "new_suffix_embedding_values_independently_recomputed": False,
        "independent_source_count_increment": 0, "same_source_condition_is_not_independent_audio": True,
        "development_conditions_already_exposed": True, "new8_is_new_source_holdout": False,
        "high_training_window_scores_prove_development_cause": False,
        "training_coverage_is_hypothesis_not_guaranteed_gain": True,
        "same_pool_objective_search_remains_closed": True, "additional_coverage_retries_allowed": False,
        "fallback_if_gate_fails": "KEEP_PREVIOUS_AUG96_HEAD_AND_STOP_THIS_FIXED_DATA_ARM",
        "original_eof_views": [{"condition": name, "id": r["id"], "role": r["role"], "weak_class": r["weak_class"],
            "no_matching_event_by_original_eof": r["original_eof_no_matching_event_censored"],
            "interpretation": "CENSORED_NOT_FRR", "word_end_latency": "NOT_MEASURED"}
            for name, values in metrics.items() for r in values if r["weak_class"]],
        "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN", "word_end_latency": "NOT_MEASURED",
        "new_head_c_numeric_parity": "SAVED_NATIVE_AND_ARM_QEMU_CACHED_HEAD_PROOFS_VERIFIED",
        "full_pcm_c_frontend_chain": "NOT_RUN", "ssc305": "NOT_RUN", **dict.fromkeys(FALSE_FLAGS, False)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--expected-run")
    parser.add_argument("--expected-head")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    require(args.preflight != bool(args.quality), "choose exactly --preflight or --quality")
    sources = [legacy.Archive(path.read_bytes()) for path in (args.training, args.incumbent, args.temporal)]
    report = input_preflight(*sources) if args.preflight else audit(legacy.Archive(args.quality.read_bytes()),
        *sources, args.expected_run, args.expected_head)
    payload = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as out:
            out.write(payload)
        print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
            "bytes": len(payload.encode()), "sha256": digest(payload.encode()), "new_model_calls_by_this_auditor": 0}, sort_keys=True))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()

