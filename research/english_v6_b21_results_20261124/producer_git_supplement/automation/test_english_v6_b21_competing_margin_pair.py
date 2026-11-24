"""B21 finite source exposure, blank competitor and authorization checks.

Numerical tests are mandatory in the pinned Actions training job. They operate on
small hand-constructed tensors, never KWS models or audio.
"""
import copy
from collections import defaultdict
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_competing_margin_pair as b21

NUMERIC = "--numeric" in sys.argv
if NUMERIC:
    sys.argv.remove("--numeric")
    import torch

ROSTER = Path(__file__).with_name("english_v6_b21_train_roster.json")
ROSTER_SHA = "b10ea5a4892fb7415b400fadd29eccbd66684a4dc7262d498db264463b583f4f"


def actual_roster():
    raw = ROSTER.read_bytes()
    if b21.b10.digest(raw) != ROSTER_SHA:
        raise ValueError("original B17 source metadata fixture changed")
    doc = json.loads(raw)
    groups = {0: defaultdict(list), 1: defaultdict(list)}
    for row in doc["synthetic"]:
        groups[row["mono_class"]][row["source_group_id"]].append(row)
    return groups, doc["natural"]


class Protocol(unittest.TestCase):
    def test_design_recipe_is_fixed_and_future_gated(self):
        d = json.loads(b21.DESIGN_FILE.read_bytes())
        self.assertEqual(d["pair"]["parent_pt_sha256"], b21.B0_PT)
        self.assertEqual(d["pair"]["updates_per_arm"], 192)
        self.assertEqual(d["pair"]["shared_positive_retention_weight"], .08)
        self.assertEqual(d["pair"]["shared_ordinary_ceiling_weight"], .08)
        self.assertEqual(d["positive_retention"]["eligible_combinations_by_gain"], {"1.0":42,"0.75":42,"0.5":36})
        self.assertEqual(d["saved_train_mechanism"]["new_diagnostic_forwards"], 0)
        self.assertFalse(d["promotion_allowed"])
        with patch.object(b21, "DESIGN_SHA", "unbound"):
            with self.assertRaises(ValueError):
                b21.read_design()
        if not b21.DESIGN_SHA.startswith("__"):
            b21.read_design()  # Mandatory authenticated contract once finalized.

    def test_only_competitor_set_changes(self):
        for target in (1,2,3):
            control = b21.competitors(target, "control")
            challenger = b21.competitors(target, "challenger")
            self.assertEqual(control, [k for k in (1,2,3) if k != target])
            self.assertEqual(challenger, [0] + control)
        for target, arm in ((0,"control"),(True,"control"),(1,"unknown")):
            with self.assertRaises(ValueError):
                b21.competitors(target, arm)

    def test_blank_suppression_is_invisible_only_to_control(self):
        teacher, student = [0.,3.,0.,0.], [4.,3.,0.,0.]
        self.assertEqual(b21.scalar_retention(student, teacher, 1, "control"), 0.)
        self.assertEqual(b21.scalar_retention(student, teacher, 1, "challenger"), 4.)
        for arm in ("control","challenger"):
            self.assertEqual(b21.scalar_retention(teacher, teacher, 1, arm), 0.)
            self.assertEqual(b21.scalar_retention([v+10 for v in student], [v-7 for v in teacher], 1, arm),
                             b21.scalar_retention(student, teacher, 1, arm))
            with self.assertRaises(ValueError):
                b21.scalar_retention([float("nan"),3,0,0], teacher, 1, arm)

    def test_actual_all_originals_receive_all_three_prefixes(self):
        groups, natural = actual_roster()
        slots = b21.shared_schedule(groups, natural)
        coverage = b21.schedule_coverage(slots, groups, natural)
        self.assertEqual(len(slots), 192)
        self.assertEqual(coverage["synthetic_originals"], 79)
        self.assertEqual(coverage["synthetic_original_prefix_pairs"], 237)
        self.assertEqual(coverage["natural_original_prefix_pairs"], 27)
        self.assertTrue(coverage["all_originals_all_three_prefixes"])
        for prefixes in list(coverage["synthetic_prefixes"].values()) + list(coverage["natural_prefixes"].values()):
            self.assertEqual(prefixes, [6400,12800,19200])
        legacy = defaultdict(set)
        gids = sorted(groups[0])
        for step in range(192):
            legacy[gids[step%35]].add(b21.b10.PREFIXES[(step//35+step%3)%3])
        self.assertEqual({len(v) for v in legacy.values()}, {1})

    def test_tail_is_structural_and_natural_original_draw_unchanged(self):
        groups, natural = actual_roster()
        slots = b21.shared_schedule(groups, natural)
        gids = sorted(groups[0])
        widths = {g:max(len(groups[k][g]) for k in (0,1)) for g in gids}
        expected = sorted(gids, key=lambda g:(-widths[g],g))[:17]
        self.assertEqual([s["source_group_id"] for s in slots[175:]], expected)
        self.assertEqual([s["natural"] for s in slots], b21.b13.paired.freeze_draw(natural, 192))
        # Dict/input iteration ordering cannot alter the shared schedule.
        reversed_groups = {k:dict(reversed(list(v.items()))) for k,v in groups.items()}
        self.assertEqual(slots, b21.shared_schedule(reversed_groups, natural))

    def test_missing_original_prefix_is_rejected(self):
        groups, natural = actual_roster()
        slots = b21.shared_schedule(groups,natural)
        damaged = copy.deepcopy(slots)
        uid = damaged[0]["positive"]["id"]
        for slot in damaged:
            if slot["positive"]["id"] == uid:
                slot["prefix"] = 6400
        with self.assertRaises(ValueError):
            b21.schedule_coverage(damaged,groups,natural)
        damaged = copy.deepcopy(slots)
        damaged[0]["positive"]["role"] = "development"
        with self.assertRaises(ValueError):
            b21.schedule_coverage(damaged,groups,natural)

    def test_strict_once_only_public_run_guard(self):
        old,new = "a"*40,"b"*40
        env = dict(GITHUB_ACTIONS="true",GITHUB_REPOSITORY="jiying2007/kws-data",GITHUB_EVENT_NAME="push",
                   GITHUB_REF=b21.RUN_BRANCH,GITHUB_RUN_ATTEMPT="1",GITHUB_SHA=new,GITHUB_RUN_ID="1234")
        event = dict(before=old,after=new,ref=b21.RUN_BRANCH,
                     repository=dict(full_name="jiying2007/kws-data",private=False),forced=False,deleted=False)
        self.assertEqual(b21.guard(old,env,event,old)["head"],new)
        for altered in (dict(env,GITHUB_RUN_ATTEMPT="2"),dict(env,GITHUB_REF="refs/heads/main")):
            with self.assertRaises(ValueError):
                b21.guard(old,altered,event,old)
        for altered in (dict(event,before="c"*40),dict(event,forced=True),
                        dict(event,repository=dict(full_name="jiying2007/kws-data",private=True))):
            with self.assertRaises(ValueError):
                b21.guard(old,env,altered,old)
        with self.assertRaises(ValueError):
            b21.guard("__B21_REVIEWED_TRAIN_PARENT__",env,event,old)

    def test_invalid_pair_receipts_never_allow_quality(self):
        for doc in ({},{"schema":b21.SCHEMA},{"schema":b21.SCHEMA,"status":"success","promotion_allowed":True}):
            with self.assertRaises(ValueError):
                b21.validate_pair_receipt(doc)


@unittest.skipUnless(NUMERIC, "requires pinned Torch in actual training job")
class NumericalMargins(unittest.TestCase):
    def fixture(self, arm, blank=4.):
        student = torch.tensor([[[blank,3.,0.,0.]]],requires_grad=True)
        teacher = torch.tensor([0.,3.,0.,0.],requires_grad=True)
        other = b21.competitors(1,arm)
        reference = (teacher[1]-teacher[other].max()).reshape(1)
        paths = {"group":{"original":[(None,[(1,0,1,other,reference)])]}}
        trace = []
        loss = b21.retention_loss(paths,1.,trace,torch)(0,lambda _:student)
        return student,teacher,loss,trace

    def test_full_competitor_corrects_blank_with_no_teacher_gradient(self):
        student,teacher,loss,trace = self.fixture("challenger")
        self.assertEqual(float(loss.detach()),4.)
        loss.backward()
        self.assertTrue(torch.equal(student.grad,torch.tensor([[[1.,-1.,0.,0.]]])))
        self.assertIsNone(teacher.grad)
        self.assertEqual(trace[0]["unweighted_loss"],4.)

    def test_control_does_not_penalize_already_winning_nonblank(self):
        student,teacher,loss,_ = self.fixture("control")
        self.assertEqual(float(loss.detach()),0.)
        loss.backward()
        self.assertTrue(torch.equal(student.grad,torch.zeros_like(student)))
        self.assertIsNone(teacher.grad)

    def test_one_sided_positive_margin_no_penalty_after_recovery(self):
        student,teacher,loss,_ = self.fixture("challenger",blank=-2.)
        self.assertEqual(float(loss.detach()),0.)
        loss.backward()
        self.assertTrue(torch.equal(student.grad,torch.zeros_like(student)))
        self.assertIsNone(teacher.grad)

    def test_ordinary_ceiling_unchanged_and_frozen(self):
        row=dict(id="fixture",source_group_id="train",source_pcm_sha256="a"*64,mono_class=0,role="train")
        student=torch.tensor([[0.,3.,1.,-1.]],requires_grad=True)
        teacher=torch.tensor([[0.,1.,1.,-1.]],requires_grad=True)
        loss=b21.b17.ordinary_margin_loss(student,teacher,row,torch)
        self.assertAlmostEqual(float(loss.detach()),2/3,places=6)
        loss.backward()
        self.assertTrue(torch.allclose(student.grad,torch.tensor([[-1/3,1/3,0.,0.]])))
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
