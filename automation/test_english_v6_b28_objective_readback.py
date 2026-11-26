"""Invented FP32-array rejection contracts; no original data or model calls."""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import itertools
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_objective_readback as reader


def uniform_no_event(frames):
    # Independent exact integer path counting with the actual collapse order.
    # State holds the previous raw token and the last two collapsed nonblanks;
    # paths completing 1,2,3 are removed. No frozen DFA table is consulted.
    states = {(-1, ()): 1}
    for _ in range(frames):
        after = defaultdict(int)
        for (previous, tail), count in states.items():
            for token in range(4):
                emitted = tail + (token,) if token != previous and token != 0 else tail
                if emitted[-3:] != (1, 2, 3):
                    after[token, emitted[-2:]] += count
        states = after
    return -math.log(sum(states.values()) / (4 ** frames))


def fixtures():
    # 34 groups / 43 identities / 122 conditions; first22 group widths sum82,
    # so five complete rotations plus22 groups produce exactly692 old returns.
    conditions, ids = [], []
    widths = [4] * 16 + [3] * 6 + [4] * 4 + [3] * 8
    for group_number, width in enumerate(widths):
        group = "invented-group-%02d" % group_number
        group_ids = ["invented-positive-%02d-a" % group_number]
        if group_number < 9:
            group_ids.append("invented-positive-%02d-b" % group_number)
        ids.extend(group_ids)
        for i in range(width):
            conditions.append(dict(group=group, id=group_ids[i % len(group_ids)],
                                   gain=str((1.0, 0.5, 0.25, 0.125)[i])))
    source = dict(positive_retention=conditions[:-2],
                  retention_additions=[dict(row, teacher="B0") for row in conditions[-2:]])

    def row(uid, tier="silver"):
        return dict(id=uid, samples=1120, source_group_id="invented-source-" + uid, tier=tier)

    draws = [dict(prefix=6400, ordinary=row("ordinary"), positive=row(ids[i % 43]),
                  natural=row("natural", "silver" if i % 2 == 0 else "weak")) for i in range(192)]
    schedule = reader.expected_forward_schedule(draws, source)
    payloads, records = {}, []
    for i, entry in enumerate(schedule):
        key = "forward_" + str(i).zfill(5)
        payloads[key] = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0) * entry["frames"]
        records.append(dict(entry, array=key, sha256=reader.sha(payloads[key])))
    index = dict(records=records, count=1844, arrays_sha256="a" * 64,
                 pending_unreturned_forward=None, logits_are_raw=True)
    draw_hash = reader.sha(reader.encoded(draws))
    observations = []
    by_step = defaultdict(list)
    for entry in records:
        if entry["kind"] != "natural_ordinary":
            continue
        original = draws[entry["step"] - 1]["natural"]
        terms = uniform_no_event(entry["frames"] - 37)
        value = (1.0 if original["tier"] == "silver" else 0.75) * (0.6 * terms + 0.5 * math.log(4.0))
        observation = {key: entry[key] for key in ("step", "id", "gain", "prefix", "array", "frames")}
        observation.update(group=original["source_group_id"], tier=original["tier"],
                           raw_logits_sha256=entry["sha256"], suffix_start=37,
                           prefix_blank_end=32, weighted_loss=value)
        observations.append(observation)
        by_step[entry["step"]].append(value)
    trace = dict(schema=reader.TRACE_SCHEMA, objective_schema=reader.OBJECTIVE_SCHEMA,
        objective_sha256=reader.OBJECTIVE_SHA, reference_sha256=reader.REFERENCE_SHA,
        common_draw_sha256=draw_hash, count=384, records=observations)
    steps = []
    for step, draw in enumerate(draws, 1):
        natural = 0.16 * sum(by_step[step]) / 2
        steps.append(dict(step=step, prefix=draw["prefix"], ordinary_id=draw["ordinary"]["id"],
            positive_id=draw["positive"]["id"], natural_id=draw["natural"]["id"],
            loss=0.4 + 0.03 + natural + 0.08 + 0.08 * 0.05,
            synthetic_ctc=0.4, gain_consistency=0.03, natural_no_event=natural,
            positive_retention=0.08, ordinary_ceiling=0.05, ceiling_coefficient=0.08,
            preclip_gradient_norm=6.0 if step % 2 else 3.0, gradient_clipped=bool(step % 2)))
    optimizer = dict(schema=reader.OPTIMIZER_SCHEMA, count=192, records=steps)
    return [index, payloads, trace, optimizer, source, draws], draw_hash


class OriginalObjectiveReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.draw_hash = fixtures()

    def verify(self, inputs=None):
        # A test-only fictional source hash; production has no count/hash/tol
        # override parameter and never accepts these invented identities.
        with mock.patch.object(reader, "COMMON_DRAW_SHA", self.draw_hash):
            return reader.verify_natural_objective(*(self.original if inputs is None else inputs))

    def rejected(self, mutate, pattern=None):
        data = copy.deepcopy(self.original)
        mutate(data)
        context = self.assertRaisesRegex(ValueError, pattern) if pattern else self.assertRaises(ValueError)
        with context:
            self.verify(data)

    def test_complete_invented_arrays_and_exact_integer_reference(self):
        report = self.verify()
        self.assertEqual(report["natural_objective_returns_verified"], 384)
        self.assertEqual(report["optimization_arrays_verified"], 1844)
        self.assertEqual(report["optimizer_trace_entries_verified"], 192)
        self.assertEqual(report["scalar_comparison"]["per_gain"]["comparisons"], 384)
        self.assertEqual(report["scalar_comparison"]["per_step"]["comparisons"], 384)
        self.assertEqual(report["scalar_comparison"]["total"]["comparisons"], 192)
        self.assertFalse(report["gradients_recomputed"])
        self.assertFalse(report["optimizer_updates_replayed"])
        self.assertEqual((report["new_model_calls"], report["new_optimizer_updates"]), (0, 0))
        self.assertEqual((reader.REL_TOL, reader.ABS_TOL), (2e-6, 2e-6))

    def test_small_array_matches_bruteforce_and_tier_weight(self):
        no_event = 0
        for path in itertools.product(range(4), repeat=7):
            collapsed = [token for i, token in enumerate(path) if i == 0 or token != path[i - 1]]
            collapsed = [token for token in collapsed if token != 0]
            no_event += not any(collapsed[i:i + 3] == [1, 2, 3] for i in range(len(collapsed) - 2))
        exact = -math.log(no_event / (4 ** 7))
        payload = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0) * 44
        terms = reader.natural_loss_terms(payload, 44, 6400, "silver")
        weak = reader.natural_loss_terms(payload, 44, 6400, "weak")
        self.assertAlmostEqual(terms["suffix_no_event_nll"], exact, places=13)
        self.assertAlmostEqual(terms["prefix_peak"], math.log(4.0), places=13)
        self.assertAlmostEqual(weak["weighted_loss"], terms["weighted_loss"] * 0.75, places=13)

    def test_prefix_reference_preserves_large_common_fp32_offsets(self):
        baseline = reader.natural_loss_terms(bytes(44 * 4 * 4), 44, 6400, "silver")
        for offset in (float(2**60), -float(2**60)):
            payload = struct.pack("<4f", offset, offset, offset, offset) * 44
            with self.subTest(offset=offset):
                terms = reader.natural_loss_terms(payload, 44, 6400, "silver")
                self.assertAlmostEqual(terms["prefix_peak"], math.log(4.0), places=13)
                self.assertEqual(terms, baseline)

    def test_prefix_peak_uses_temporal_maximum_and_excludes_gap(self):
        uniform = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
        elevated = struct.pack("<4f", -3.0, 0.0, 0.0, 0.0)
        base = uniform * 44
        one_prefix = elevated + uniform * 43
        one_gap = uniform * 32 + elevated + uniform * 11
        first = reader.natural_loss_terms(one_prefix, 44, 6400, "silver")
        ordinary = reader.natural_loss_terms(base, 44, 6400, "silver")
        gap = reader.natural_loss_terms(one_gap, 44, 6400, "silver")
        self.assertAlmostEqual(first["prefix_peak"], math.log1p(3 * math.exp(3)), places=13)
        self.assertEqual(first["suffix_no_event_nll"], ordinary["suffix_no_event_nll"])
        self.assertEqual(gap, ordinary)

    def test_complete_rowset_missing_duplicate_or_extra_arrays_rejected(self):
        self.rejected(lambda x: x[1].pop("forward_00000"))
        self.rejected(lambda x: x[1].update(extra=b""))
        self.rejected(lambda x: x[0]["records"].__setitem__(1, copy.deepcopy(x[0]["records"][0])))
        self.rejected(lambda x: x[0].update(pending_unreturned_forward={"kind": "natural_ordinary"}))

    def test_payload_hash_shape_corruption_and_nonfinite_rejected(self):
        self.rejected(lambda x: x[1].__setitem__("forward_00000", x[1]["forward_00000"][:-1]))
        self.rejected(lambda x: x[1].__setitem__("forward_00000", struct.pack("<f", 1.0) + x[1]["forward_00000"][4:]))
        for value in (math.nan, math.inf, -math.inf):
            def alter(x, value=value):
                body = struct.pack("<f", value) + x[1]["forward_00004"][4:]
                x[1]["forward_00004"] = body
                x[0]["records"][4]["sha256"] = reader.sha(body)
            self.rejected(alter, "finite")

    def test_schedule_gain_prefix_and_frames_cannot_be_relabeled(self):
        for key, value in (("gain", 0.25), ("prefix", 12800), ("frames", 45), ("step", True)):
            self.rejected(lambda x, key=key, value=value: x[0]["records"][4].update({key: value}))
        self.rejected(lambda x: x[5][0].update(prefix=12800), "COMMON_DRAW")

    def test_math_identity_and_old_schema_rejected(self):
        for key, value in (("schema", "english-v6-b27-natural-no-event-trace-v1"),
                           ("objective_schema", "blank-occupancy"), ("objective_sha256", "f" * 64),
                           ("reference_sha256", "0" * 64), ("common_draw_sha256", "1" * 64), ("count", 383)):
            self.rejected(lambda x, key=key, value=value: x[2].update({key: value}))
        with mock.patch.object(reader, "REFERENCE_SHA", "0" * 64):
            with self.assertRaisesRegex(ValueError, "reference bytes"):
                self.verify()

    def test_natural_trace_requires_complete_ordered_identity(self):
        self.rejected(lambda x: x[2]["records"].pop())
        self.rejected(lambda x: x[2]["records"].__setitem__(1, copy.deepcopy(x[2]["records"][0])))
        for key, value in (("gain", 0.5), ("prefix_blank_end", 31), ("suffix_start", 38),
                           ("tier", "weak"), ("group", "different"), ("array", "forward_00000"),
                           ("raw_logits_sha256", "f" * 64)):
            self.rejected(lambda x, key=key, value=value: x[2]["records"][0].update({key: value}))

    def test_per_gain_scalar_injection_and_nonfinite_rejected(self):
        for value in (42.0, math.nan, math.inf, -1.0, True):
            self.rejected(lambda x, value=value: x[2]["records"][0].update(weighted_loss=value))

    def test_per_step_injection_rejected_even_when_total_is_adjusted(self):
        def alter(x):
            x[3]["records"][0]["natural_no_event"] += 0.01
            x[3]["records"][0]["loss"] += 0.01
        self.rejected(alter, "per-step natural")

    def test_optimizer_total_coefficients_and_legacy_name_rejected(self):
        self.rejected(lambda x: x[3]["records"][0].update(loss=30.0), "scalar components")
        self.rejected(lambda x: x[3]["records"][0].update(ceiling_coefficient=0.16))
        self.rejected(lambda x: x[3]["records"][0].update(gradient_clipped=False))
        def legacy(x):
            row = x[3]["records"][0]
            row["natural_ctc"] = row.pop("natural_no_event")
        self.rejected(legacy)
        self.rejected(lambda x: x[3].update(schema="english-v6-b27-optimizer-trace-v1"))
        self.rejected(lambda x: x[3]["records"].pop())

    def test_uniform_small_array_rejects_invalid_boundary_and_tier(self):
        payload = bytes(44 * 4 * 4)
        for prefix, tier in ((6399, "silver"), (6400, "unknown"), (19200, "silver")):
            with self.assertRaises(ValueError):
                reader.natural_loss_terms(payload, 44, prefix, tier)

    def test_imports_are_stdlib_and_frozen_reference_only(self):
        self.assertFalse({"torch", "numpy", "onnxruntime", "transformers"} & set(sys.modules))
        self.assertEqual(hashlib.sha256(Path(reader.reference.__file__).read_bytes()).hexdigest(),
                         "034a4827fb9ff3091a5e23ea25e2eab0bc23ee7263bce644fd52741e6b80f969")


if __name__ == "__main__":
    unittest.main()
