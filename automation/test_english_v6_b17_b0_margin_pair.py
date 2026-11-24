"""B17 source contract and optional mandatory-in-training numerical gradient tests."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_b0_margin_pair as b17

NUMERIC = "--numeric" in sys.argv
if NUMERIC:
    sys.argv.remove("--numeric")
    import torch


def row(positive=False, role="train"):
    return dict(id="synthetic-approved", source_group_id="piper-train-fixture",
                source_pcm_sha256="a" * 64, mono_class=int(positive), role=role)


class SourceContract(unittest.TestCase):
    def test_protocol_and_assessment_are_immutable(self):
        d = b17.read_design()
        a = json.loads(b17.ASSESSMENT.read_bytes())
        self.assertTrue(a["freeze_before_optimizer"])
        self.assertEqual(d["pair"]["parent_pt_sha256"], b17.B0_PT)
        self.assertEqual(d["pair"]["updates_per_arm"], 192)
        self.assertEqual(d["pair"]["shared_positive_retention_weight"], 0.08)
        self.assertEqual(d["pair"]["natural_negative_ctc_weight"], 0.08)
        self.assertEqual(d["pair"]["control_ceiling_weight"], 0)
        self.assertEqual(d["pair"]["challenger_ceiling_weight"], 0.08)
        self.assertEqual(d["retention"]["eligible_combinations_by_gain"],
                         {"1.0": 42, "0.75": 42, "0.5": 36})
        self.assertFalse(d["promotion_allowed"])

    def test_forbid_dev_and_positive_ceiling(self):
        self.assertEqual(b17.ordinary_row(row()), row())
        for bad in (row(True), row(role="development"), row(role="calibration"), row(role="locked")):
            with self.assertRaises(ValueError):
                b17.ordinary_row(bad)
        bad = row()
        bad["mono_class"] = False
        with self.assertRaises(ValueError):
            b17.ordinary_row(bad)

    def test_bad_identity_rejected(self):
        for key in ("id", "source_group_id", "source_pcm_sha256"):
            bad = row()
            bad[key] = ""
            with self.assertRaises(ValueError):
                b17.ordinary_row(bad)

    def test_single_sided_loss_and_shift_invariance(self):
        teacher = [0.0, 1.0, 1.0, -1.0]
        student = [0.0, 3.0, 1.0, -1.0]
        self.assertAlmostEqual(b17.scalar_ceiling(student, teacher), 2 / 3)
        self.assertEqual(b17.scalar_ceiling(teacher, teacher), 0)
        self.assertEqual(b17.scalar_ceiling([0, 0, 0, -2], teacher), 0)
        self.assertEqual(b17.scalar_ceiling([s + 100 for s in student], teacher), 2 / 3)
        self.assertEqual(b17.scalar_ceiling(student, [s - 100 for s in teacher]), 2 / 3)
        self.assertGreater(b17.scalar_ceiling([-1, 1, 1, -1], teacher), 0)

    def test_scalar_nonfinite_inputs_rejected(self):
        for v in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                b17.scalar_ceiling([0, v, 1, 2], [0, 1, 2, 3])

    def test_train_hypothesis_requires_both_mechanisms(self):
        positive = {"role": "train", "positive": True, "group": "p",
                    "gains": {"0.5": {"B9": {"events": 1}, "B0": {"events": 0},
                                        "mean_positive_margin_increase": 0}}}
        negative = {"role": "train", "positive": False, "group": "n1",
                    "gains": {"1.0": {"B9": {"events": 1}, "B0": {"events": 0},
                                        "mean_positive_margin_increase": 1}}}
        other = dict(negative, group="n2")
        self.assertFalse(b17.diagnostic_gate([positive, negative])["optimizer_start_allowed"])
        self.assertFalse(b17.diagnostic_gate([negative, other])["optimizer_start_allowed"])
        passed = b17.diagnostic_gate([positive, negative, other])
        self.assertTrue(passed["optimizer_start_allowed"])
        self.assertFalse(passed["promotion_allowed"])
        self.assertEqual(passed["status"], "TRAIN_SIGNAL")
        with self.assertRaises(ValueError):
            b17.diagnostic_gate([dict(positive, role="development"), negative, other])

    def test_duplicate_group_is_not_two_groups(self):
        fixture = {"role": "train", "positive": False, "group": "same",
                   "gains": {"1.0": {"mean_positive_margin_increase": 1}}}
        r = b17.diagnostic_gate([fixture, copy.deepcopy(fixture)])
        self.assertFalse(r["optimizer_start_allowed"])
        self.assertEqual(r["ordinary_margin_elevation_groups"], ["same"])

    def test_strict_single_run_guard(self):
        old, new = "a" * 40, "b" * 40
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "jiying2007/kws-data",
               "GITHUB_EVENT_NAME": "push", "GITHUB_REF": b17.RUN_BRANCH,
               "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": new, "GITHUB_RUN_ID": "1234"}
        event = {"before": old, "after": new, "ref": b17.RUN_BRANCH,
                 "repository": {"full_name": "jiying2007/kws-data", "private": False},
                 "forced": False, "deleted": False}
        self.assertEqual(b17.guard(old, env, event, old)["head"], new)
        for altered in (dict(env, GITHUB_RUN_ATTEMPT="2"), dict(env, GITHUB_REF="refs/heads/main")):
            with self.assertRaises(ValueError):
                b17.guard(old, altered, event, old)
        for altered in (dict(event, before="c" * 40), dict(event, forced=True),
                        dict(event, repository={"full_name": "jiying2007/kws-data", "private": True})):
            with self.assertRaises(ValueError):
                b17.guard(old, env, altered, old)
        with self.assertRaises(ValueError):
            b17.guard("__PREPARED_MAIN_SHA__", env, event, old)


@unittest.skipUnless(NUMERIC, "numerical tests required after pinned Torch installation in training job")
class NumericalLoss(unittest.TestCase):
    def test_positive_margin_gradient_and_frozen_teacher(self):
        teacher = torch.tensor([[0., 1., 1., -1.]], requires_grad=True)
        student = torch.tensor([[0., 3., 1., -1.]], requires_grad=True)
        loss = b17.ordinary_margin_loss(student, teacher, row(), torch)
        self.assertAlmostEqual(float(loss.detach()), 2 / 3, places=6)
        loss.backward()
        self.assertTrue(torch.allclose(student.grad, torch.tensor([[-1/3, 1/3, 0, 0]])))
        self.assertIsNone(teacher.grad)

    def test_reduced_nonblank_never_penalized(self):
        student = torch.tensor([[1., 0., 0., -2.]], requires_grad=True)
        teacher = torch.tensor([[0., 1., 1., -1.]])
        loss = b17.ordinary_margin_loss(student, teacher, row(), torch)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_reuse_ctc_log_softmax_exact_margin(self):
        torch.manual_seed(17)
        teacher, student = torch.randn(30, 4), torch.randn(30, 4)
        raw = b17.ordinary_margin_loss(student, teacher, row(), torch)
        logp = b17.ordinary_margin_loss(student.log_softmax(-1), teacher, row(), torch)
        shifted = b17.ordinary_margin_loss(student + 100, teacher - 50, row(), torch)
        self.assertTrue(torch.allclose(raw, logp, atol=1e-6))
        self.assertTrue(torch.allclose(raw, shifted, atol=1e-5))

    def test_positive_and_mismatched_shapes_rejected(self):
        values = torch.zeros(4, 4)
        with self.assertRaises(ValueError):
            b17.ordinary_margin_loss(values, values, row(True), torch)
        with self.assertRaises(ValueError):
            b17.ordinary_margin_loss(values, torch.zeros(5, 4), row(), torch)
        with self.assertRaises(ValueError):
            b17.ordinary_margin_loss(values, torch.full((4, 4), float("nan")), row(), torch)


if __name__ == "__main__":
    unittest.main()
