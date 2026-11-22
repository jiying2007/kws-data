"""B10 native saved-only QA: strict source-role, policy and no-training regressions."""
from __future__ import annotations
from collections import Counter
import importlib.util
import os
from pathlib import Path
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("b10_native_saved",HERE/
                   "english_v6_b10_hardneg_saved_screen_once.py")
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)

class B10SavedSourceScreenTests(TestCase):
    def test_pinned_original_and_90_of_90_native_contract(self):
        x=m.source_check()
        self.assertEqual(x["status"],"SAVED_SOURCE_ONLY_NO_NEW_MODEL_FORWARD")
        self.assertEqual(x["source_run"],44906200481)
        self.assertEqual(x["observer_run"],44908125673)
        self.assertEqual((x["originals"],x["new_train_candidates"],
                          x["new_dev_originals"]),(18,6,12))
        self.assertEqual(x["native_observations_expected"],90)
        self.assertEqual(x["new_model_forwards"],0)
        self.assertEqual(x["new_optimizer_updates"],0)
        self.assertFalse(x["cal_locked_opened"])
        self.assertFalse(x["product_promotion_allowed"])
        self.assertEqual(set(m.ARCHIVES),set(m.observer.PROVIDERS)|{"source"})
        self.assertEqual(len({x[0] for x in m.ARCHIVES.values()}),6)

    def test_contradictory_asr_never_relabelled_to_train(self):
        for src,classified in ((2,0),(0,2)):
            row={"requested_class":src,"samples":16000}
            with mock.patch.object(m.policy,"decide",return_value={
                    "tier":"silver","class":classified,"reason":"fake",
                    "margins":{},"training_allowed":False}):
                actual=m.decide_one(row,{}, {})
                self.assertEqual(actual["tier"],"unknown")
                self.assertIsNone(actual["class"])
                self.assertFalse(actual["training_allowed"])

    def test_missing_consensus_cannot_become_ordinary(self):
        row={"requested_class":0,"samples":16000}
        with mock.patch.object(m.policy,"decide",return_value={
                "tier":"unknown","class":None,"reason":"MISSING_FIVE_MODEL_OBSERVATIONS",
                "margins":{},"training_allowed":False}):
            observed=m.decide_one(row,{}, {})
            self.assertEqual(observed["tier"],"unknown")
            self.assertIsNone(observed["class"])

    @staticmethod
    def source_rows():
        rows=[]
        for speaker in range(448,454):
            rows.append({"id":f"b10-train-{speaker}-ordinary-time-hard",
                    "role":"train","cell_id":"ordinary-time-hard",
                    "source_group_id":f"train-{speaker}"})
        for speaker in range(540,546):
            for cell in ("mero-fast","ordinary-time"):
                rows.append({"id":f"b10-dev-{speaker}-{cell}",
                    "role":"development","cell_id":cell,
                    "source_group_id":f"dev-{speaker}"})
        return rows

    def test_insufficient_dev_qualified_groups_fails_closed(self):
        source=self.source_rows()
        decisions=[]
        for row in source:
            cls=0 if row["role"]=="train" or row["cell_id"]=="ordinary-time" else None
            decisions.append({**row,"tier":"silver" if cls is not None else "unknown",
                              "observed_class":cls})
        qa=m.admission(source,decisions)
        self.assertEqual(qa["train_hard_negative_qualified"],6)
        self.assertEqual(qa["development_qualified_groups"],0)
        self.assertFalse(qa["B10_research_optimizer_allowed"])
        self.assertFalse(qa["minimum_6_train_2_dev_gate_passed"])

    def test_two_truly_qualified_dev_groups_allow_research_only(self):
        source=self.source_rows()
        decisions=[]
        for row in source:
            cls=0 if row["cell_id"]!="mero-fast" else (
                  2 if row["source_group_id"] in ("dev-540","dev-541") else None)
            decisions.append({**row,"tier":"silver" if cls is not None else "unknown",
                              "observed_class":cls})
        qa=m.admission(source,decisions)
        self.assertEqual(qa["train_hard_negative_qualified"],6)
        self.assertEqual(qa["development_qualified_groups"],2)
        self.assertTrue(qa["B10_research_optimizer_allowed"])
        self.assertFalse(qa["development_audio_in_optimizer"])
        decisions[0]["observed_class"]=None
        decisions[0]["tier"]="unknown"
        qa=m.admission(source,decisions)
        self.assertFalse(qa["minimum_6_train_2_dev_gate_passed"])

    def test_source_group_overlap_denied(self):
        rows=self.source_rows()
        d=[{**r,"observed_class":0,"tier":"silver"} for r in rows]
        d[-1]["source_group_id"]=d[0]["source_group_id"]
        with self.assertRaisesRegex(ValueError,"TRAIN_DEV_SOURCE_GROUP_SEPARATION"):
            m.admission(rows,d)

    def test_reject_pr_and_schedule_and_retries(self):
        with mock.patch.dict(os.environ,{"GITHUB_ACTIONS":"false"},clear=True):
            with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_SAVED_ONLY_NO_REPLAY"):
                m.guard()

if __name__=="__main__":
    main()
