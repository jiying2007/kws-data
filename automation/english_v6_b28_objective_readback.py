"""Independent stdlib readback of B28's saved natural no-event objective.

This component accepts decoded original metadata and *raw* little-endian FP32
payloads. Its caller must first authenticate the original archives, file byte
counts/SHA256, strict NPY <f4/C-order headers, and the enclosing NPZ SHA256. It
then checks the entire fixed forward schedule and every payload itself before
recomputing all 384 natural losses and all 192 scalar sums. No model, gradient,
optimizer, audio, NumPy or Torch operation is performed here.

The suffix DP uses the independently frozen float64 reference. The unchanged
training prefix uses FP32 log_softmax/logsumexp/softplus, whereas this reader
uses the mathematically equivalent raw-logit expression in Python float64.
FP32 prefix rounding, Torch softplus's large-value shortcut, and mixed-precision
scalar addition explain non-bitwise equality. The prospective, unadjustable
comparison is rel_tol=abs_tol=2e-6 for per-gain, per-step and total scalars.
Underflow of an extremely rare event to zero is not evidence of impossibility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import english_v6_b28_no_event_reference as reference

OBJECTIVE_SCHEMA = "english-v6-b28-natural-suffix-no-event-v1"
TRACE_SCHEMA = "english-v6-b28-natural-no-event-trace-v1"
OPTIMIZER_SCHEMA = "english-v6-b28-optimizer-trace-v1"
REPORT_SCHEMA = "english-v6-b28-natural-objective-readback-v1"
OBJECTIVE_SHA = "1321eef19cf02d8ce8009e307d4fc384eeac7f59a57e2e07c05dbf79d273350a"
REFERENCE_SHA = "034a4827fb9ff3091a5e23ea25e2eab0bc23ee7263bce644fd52741e6b80f969"
COMMON_DRAW_SHA = "90b41a650ad5439524bef4bc6d13394fd65034e98c2c5a7862e0a0c1de78a83d"
REL_TOL = ABS_TOL = 2e-6
GAINS = (1.0, 0.5)
PREFIXES = (6400, 12800, 19200)
INNER_NEGATIVE = 0.6
PREFIX_WEIGHT = 0.5
NATURAL_WEIGHT = 0.16
CEILING_WEIGHT = 0.08
TRACE_FIELDS = frozenset(("step", "id", "group", "gain", "prefix", "tier",
    "array", "raw_logits_sha256", "frames", "suffix_start", "prefix_blank_end", "weighted_loss"))
OPTIMIZER_FIELDS = frozenset(("step", "prefix", "ordinary_id", "positive_id", "natural_id",
    "loss", "synthetic_ctc", "gain_consistency", "natural_no_event", "positive_retention",
    "ordinary_ceiling", "ceiling_coefficient", "preclip_gradient_norm", "gradient_clipped"))


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def expected_forward_schedule(draws, source):
    """Rebuild the original schedule, including all unchanged retention rows."""
    need(type(draws) is list and len(draws) == 192
         and type(source) is dict and type(source.get("positive_retention")) is list
         and len(source["positive_retention"]) == 120
         and type(source.get("retention_additions")) is list
         and len(source["retention_additions"]) == 2,
         "all 192 original draws and 122 original positive retention conditions")
    groups = defaultdict(lambda: defaultdict(list))
    for row in source["positive_retention"]:
        groups[row["group"]][row["id"]].append(dict(row, teacher="B9", newly_protected=False))
    for row in source["retention_additions"]:
        groups[row["group"]][row["id"]].append(dict(row, newly_protected=True))
    need(len(groups) == 34 and sum(len(items) for items in groups.values()) == 43,
         "all 34 original positive groups and 43 original positive identities")
    samples = {}
    for slot in draws:
        need(type(slot) is dict and type(slot.get("prefix")) is int
             and slot["prefix"] in PREFIXES, "one original fixed prefix per draw")
        for kind in ("ordinary", "positive", "natural"):
            row = slot[kind]
            need(type(row) is dict and type(row.get("id")) is str and row["id"]
                 and type(row.get("samples")) is int and row["samples"] >= 512
                 and samples.get(row["id"], row["samples"]) == row["samples"],
                 "same complete original sample length throughout the fixed draw")
            samples[row["id"]] = row["samples"]
    schedule = []
    for step, slot in enumerate(draws, 1):
        for kind in ("ordinary", "positive", "natural"):
            for gain in GAINS:
                schedule.append(dict(step=step,
                    kind="natural_ordinary" if kind == "natural" else "synthetic_" + kind,
                    id=slot[kind]["id"], gain=gain, prefix=slot["prefix"],
                    frames=(slot[kind]["samples"] + slot["prefix"] - 512) // 160 + 1))
        group = sorted(groups)[(step - 1) % 34]
        for uid in sorted(groups[group]):
            need(uid in samples, "original retention identity occurs in the fixed draw")
            for row in groups[group][uid]:
                need(type(row["gain"]) in (str, int, float), "original retention gain type")
                gain = float(row["gain"])
                need(number(gain) and gain > 0
                     and type(row.get("teacher")) is str,
                     "original finite retention gain and teacher identity")
                schedule.append(dict(step=step, kind="positive_retention", id=uid, group=group,
                    gain=gain, prefix=0, teacher=row["teacher"],
                    newly_protected=row["newly_protected"], frames=(samples[uid] - 512) // 160 + 1))
    need(len(schedule) == 1844 and Counter(row["kind"] for row in schedule) ==
         {"synthetic_ordinary": 384, "synthetic_positive": 384,
          "natural_ordinary": 384, "positive_retention": 692},
         "the exact original 1844 optimization returns by role")
    return schedule


def raw_frames(payload, frames):
    need(type(payload) is bytes and type(frames) is int and frames > 0
         and len(payload) == frames * 4 * 4, "complete raw FP32 (T,4) payload")
    values = tuple(struct.iter_unpack("<4f", payload))
    need(all(math.isfinite(value) for row in values for value in row),
         "every raw FP32 logit is finite")
    return values


def _logsumexp(values):
    largest = max(values)
    return largest + math.log(math.fsum(math.exp(value - largest) for value in values))


def _softplus(value):
    return value + math.log1p(math.exp(-value)) if value >= 0 else math.log1p(math.exp(value))


def natural_loss_terms(payload, frames, prefix, tier):
    """Pure original-array calculation; the caller supplies no tunable knobs."""
    need(type(prefix) is int and prefix in PREFIXES and tier in ("silver", "weak"),
         "one original natural prefix and admitted source tier")
    raw = raw_frames(payload, frames)
    suffix_start, prefix_blank_end = prefix // 160 - 3, prefix // 160 - 8
    need(0 < prefix_blank_end < suffix_start < frames - 5, "IMMUTABLE_B3_SUFFIX_CTC_START")
    suffix = reference.no_event_nll(raw[suffix_start:])
    # Remove the common logit offset before logsumexp to preserve small margins.
    prefix_peak = max(_softplus(_logsumexp(tuple(value - row[0] for value in row[1:])))
                      for row in raw[:prefix_blank_end])
    weighted = (1.0 if tier == "silver" else 0.75) * (INNER_NEGATIVE * suffix + PREFIX_WEIGHT * prefix_peak)
    need(all(number(value) and value >= 0 for value in (suffix, prefix_peak, weighted)),
         "finite independently reconstructed natural objective")
    return dict(suffix_no_event_nll=suffix, prefix_peak=prefix_peak, weighted_loss=weighted)


def _compare(actual, expected, name, errors):
    need(number(actual) and number(expected)
         and math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), name)
    absolute = abs(actual - expected)
    denominator = max(abs(actual), abs(expected))
    errors.append((absolute, absolute / denominator if denominator else 0.0))


def _errors(values):
    return dict(comparisons=len(values), max_absolute_error=max(v[0] for v in values),
                max_relative_error=max(v[1] for v in values))


def verify_natural_objective(forward_index, raw_payloads, natural_trace,
                             optimizer_trace, source, draws):
    """Verify decoded originals after the outer reader authenticates containers.

    raw_payloads maps every ``forward_XXXXX`` name to the original NPY payload
    bytes, after strict <f4/C-order NPY parsing. All metadata are parsed JSON
    objects, with draws the unchanged COMMON_DRAW.json array. The complete
    outer TRAIN reader remains responsible for source/teacher/provenance and
    all seven original observation file bytes/SHA256. This function returns
    mathematical consistency evidence only, never execution authorization.
    """
    need(sha(Path(reference.__file__).read_bytes()) == REFERENCE_SHA,
         "the independently frozen no-event reference bytes")
    need(sha(encoded(draws)) == COMMON_DRAW_SHA, "the exact original canonical COMMON_DRAW")
    schedule = expected_forward_schedule(draws, source)
    need(type(forward_index) is dict and set(forward_index) ==
         {"records", "count", "arrays_sha256", "pending_unreturned_forward", "logits_are_raw"}
         and type(forward_index["count"]) is int and forward_index["count"] == 1844
         and type(forward_index["records"]) is list and len(forward_index["records"]) == 1844
         and forward_index["pending_unreturned_forward"] is None
         and forward_index["logits_are_raw"] is True and valid_hash(forward_index["arrays_sha256"]),
         "complete original raw-forward ledger with no unreturned entry")
    keys = {"forward_" + str(i).zfill(5) for i in range(1844)}
    need(type(raw_payloads) is dict and set(raw_payloads) == keys,
         "all and only 1844 complete original raw payloads")
    natural = []
    all_frames = 0
    for i, (row, expected) in enumerate(zip(forward_index["records"], schedule, strict=True)):
        key = "forward_" + str(i).zfill(5)
        need(type(row) is dict and set(row) == set(expected) | {"array", "sha256"}
             and row["array"] == key and valid_hash(row["sha256"])
             and all(type(row[name]) is type(value) and row[name] == value for name, value in expected.items()),
             "exact original schedule, gain, prefix, type and order at every return")
        payload = raw_payloads[key]
        raw_frames(payload, row["frames"])
        need(sha(payload) == row["sha256"], "each complete raw array SHA256")
        all_frames += row["frames"]
        if row["kind"] == "natural_ordinary":
            natural.append(row)
    header = dict(schema=TRACE_SCHEMA, objective_schema=OBJECTIVE_SCHEMA,
                  objective_sha256=OBJECTIVE_SHA, reference_sha256=REFERENCE_SHA,
                  common_draw_sha256=COMMON_DRAW_SHA, count=384)
    need(type(natural_trace) is dict and set(natural_trace) == set(header) | {"records"}
         and all(type(natural_trace[key]) is type(value) and natural_trace[key] == value
                 for key, value in header.items())
         and type(natural_trace["records"]) is list and len(natural_trace["records"]) == 384,
         "exact B28 natural observation schema, fixed mathematics and all 384 rows")
    per_gain_errors, per_step_errors, total_errors = [], [], []
    recomputed, recorded = defaultdict(list), defaultdict(list)
    suffix_values, prefix_values = [], []
    for trace, row in zip(natural_trace["records"], natural, strict=True):
        slot = draws[row["step"] - 1]
        original = slot["natural"]
        expected = {name: row[name] for name in ("step", "id", "gain", "prefix", "array", "frames")}
        expected.update(group=original["source_group_id"], tier=original["tier"],
                        raw_logits_sha256=row["sha256"], suffix_start=row["prefix"] // 160 - 3,
                        prefix_blank_end=row["prefix"] // 160 - 8)
        need(type(trace) is dict and set(trace) == TRACE_FIELDS
             and all(type(trace[key]) is type(value) and trace[key] == value for key, value in expected.items()),
             "exact natural trace row identity, source tier, raw payload and boundaries")
        terms = natural_loss_terms(raw_payloads[row["array"]], row["frames"], row["prefix"], original["tier"])
        _compare(trace["weighted_loss"], terms["weighted_loss"],
                 "saved per-gain objective matches its complete original FP32 array", per_gain_errors)
        recomputed[row["step"]].append(terms["weighted_loss"])
        recorded[row["step"]].append(trace["weighted_loss"])
        suffix_values.append(terms["suffix_no_event_nll"])
        prefix_values.append(terms["prefix_peak"])
    need(type(optimizer_trace) is dict and set(optimizer_trace) == {"schema", "count", "records"}
         and optimizer_trace["schema"] == OPTIMIZER_SCHEMA and type(optimizer_trace["count"]) is int
         and optimizer_trace["count"] == 192 and type(optimizer_trace["records"]) is list
         and len(optimizer_trace["records"]) == 192, "exact B28 optimizer schema and 192 original rows")
    for step, (row, slot) in enumerate(zip(optimizer_trace["records"], draws, strict=True), 1):
        expected = dict(step=step, prefix=slot["prefix"], ordinary_id=slot["ordinary"]["id"],
                        positive_id=slot["positive"]["id"], natural_id=slot["natural"]["id"])
        need(type(row) is dict and set(row) == OPTIMIZER_FIELDS
             and all(type(row[key]) is type(value) and row[key] == value for key, value in expected.items())
             and all(number(row[key]) for key in
                     ("loss", "synthetic_ctc", "gain_consistency", "natural_no_event", "positive_retention",
                      "ordinary_ceiling", "preclip_gradient_norm"))
             and type(row["ceiling_coefficient"]) is float and row["ceiling_coefficient"] == CEILING_WEIGHT
             and row["natural_no_event"] >= 0 and row["preclip_gradient_norm"] >= 0
             and type(row["gradient_clipped"]) is bool
             and row["gradient_clipped"] == (row["preclip_gradient_norm"] > 5)
             and len(recomputed[step]) == len(recorded[step]) == 2,
             "same original optimizer identity and finite components, with exact gain pairing")
        from_arrays = NATURAL_WEIGHT * sum(recomputed[step]) / 2
        from_trace = NATURAL_WEIGHT * sum(recorded[step]) / 2
        _compare(row["natural_no_event"], from_arrays,
                 "per-step natural contribution recomputed from both original gain arrays", per_step_errors)
        _compare(row["natural_no_event"], from_trace,
                 "per-step natural contribution agrees with its two original gain observations", per_step_errors)
        total = sum(row[key] for key in ("synthetic_ctc", "gain_consistency", "natural_no_event", "positive_retention"))
        total += row["ceiling_coefficient"] * row["ordinary_ceiling"]
        _compare(row["loss"], total, "all original scalar components sum to each saved total", total_errors)
    return dict(schema=REPORT_SCHEMA, objective_schema=OBJECTIVE_SCHEMA,
        objective_sha256=OBJECTIVE_SHA, reference_sha256=REFERENCE_SHA, common_draw_sha256=COMMON_DRAW_SHA,
        natural_objective_returns_verified=384, natural_no_event_terms_recomputed=True,
        optimization_arrays_verified=1844, verified_optimization_frames=all_frames,
        optimizer_trace_entries_verified=192, natural_gain_order=list(GAINS),
        suffix_scope="suffix_only_DFA_reset_to_A_no_cross_boundary_event_claim",
        suffix_no_event_nll_maximum=max(suffix_values), prefix_peak_maximum=max(prefix_values),
        scalar_comparison=dict(rel_tol=REL_TOL, abs_tol=ABS_TOL,
            fixed_before_training=True, bitwise_equality_claimed=False,
            rounding_sources=["FP32 prefix log_softmax and logsumexp", "FP32 softplus shortcut and temporal maximum",
                              "FP32 legacy components and mixed FP32/float64 scalar addition",
                              "independent float64 log-mass recurrence and reduction order"],
            per_gain=_errors(per_gain_errors), per_step=_errors(per_step_errors), total=_errors(total_errors)),
        observation_canonical_sha256={"TRAIN_FORWARD_INDEX.json": sha(encoded(forward_index)),
            "NATURAL_OBJECTIVE_TRACE.json": sha(encoded(natural_trace)),
            "OPTIMIZER_TRACE.json": sha(encoded(optimizer_trace))},
        original_container_authentication_required=True, original_npy_headers_required="<f4/C-order/(T,4)",
        gradients_recomputed=False, optimizer_updates_replayed=False,
        new_model_calls=0, new_optimizer_updates=0, promotion_allowed=False)
