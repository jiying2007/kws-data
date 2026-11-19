"""Model-free regression for exact newlocked8 source qualification evidence."""
from __future__ import annotations
from pathlib import Path
import importlib.util
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "newlocked8_final_test",HERE/"english_v5_mero_locked8_saved_machine_gate_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def sample_groups(approved=1):
    groups=[]
    for i in range(8):
        items=[("mero-fast",2,"silver" if i<approved else "unknown"),
               ("mero-slow",2,"unknown"),
               ("ordinary-time",0,"silver"),
               ("nearphone-meero-challenge",None,"unknown")]
        for cell,requested,tier in items:
            groups.append({
                "id":f"mero8locked-{i+472}-{cell}",
                "speaker_index":472+i,
                "source_group_id":f"piper-libritts-p{30000+i}",
                "role":"locked-test","cell_id":cell,
                "requested_class":requested,
                "observed_class":requested if tier!="unknown" else None,
                "tier":tier,"human_gold":False,"training_allowed":False})
    return groups

class V5NewLockedSourceMachineGateTests(TestCase):
    def test_exact_precommitted_six_artifacts_and_no_new_model(self):
        d=v.source_check()
        self.assertEqual(d["status"],"FROZEN_OLD_GATE_AND_FIVE_REAL_NEW_LOCKED_NATIVE_ZIPS")
        self.assertEqual(d["source_run"],44491090037)
        self.assertEqual(d["source_rows"],32)
        self.assertEqual(d["new_locked_speakers"],8)
        self.assertEqual(d["original_native_observer_run"],44492052970)
        self.assertEqual(d["original_model_forwards_already_retained"],160)
        self.assertEqual(d["new_native_or_kws_forward_calls"],0)
        self.assertEqual(d["new_optimizer_steps"],0)
        self.assertFalse(d["training_allowed"])
        self.assertFalse(d["product_promotion"])
        self.assertEqual(v.PRIOR_COUNTS,{
          "train":11,"development":2,"calibration":2,"locked-test":1})
        self.assertEqual(v.MIN_COUNTS,{
          "train":6,"development":2,"calibration":2,"locked-test":2})
        self.assertEqual(len({x[0] for x in v.ARCHIVES.values()}),5)

    def test_new_one_source_unlocks_synthetic_machine_gate_only(self):
        qual=v.count_new_locked(sample_groups(1))
        self.assertEqual(len(qual),1)
        result={**v.PRIOR_COUNTS}
        result["locked-test"]+=len(qual)
        self.assertEqual(result,{"train":11,"development":2,
                                  "calibration":2,"locked-test":2})
        self.assertTrue(all(result[k]>=n for k,n in v.MIN_COUNTS.items()))
        self.assertFalse(v.source_check()["product_promotion"])

    def test_no_new_source_does_not_pass(self):
        self.assertEqual(v.count_new_locked(sample_groups(0)),[])
        p={**v.PRIOR_COUNTS}
        p["locked-test"]+=0
        self.assertFalse(all(p[k]>=n for k,n in v.MIN_COUNTS.items()))

    def test_no_legacy_meero_as_ordinary_and_no_unknown_relabel(self):
        rows=sample_groups(1)
        rows[3]["tier"]="silver";rows[3]["observed_class"]=0
        with self.assertRaisesRegex(ValueError,"NEARPHONE_CAN_NEVER_BE_ORDINARY"):
            v.count_new_locked(rows)
        rows=sample_groups(1)
        rows[0]["tier"]="unknown";rows[0]["observed_class"]=2
        self.assertEqual(len(v.count_new_locked(rows)),0)

    def test_source_roles_are_immutable(self):
        rows=sample_groups(1)
        rows[4]["role"]="train"
        with self.assertRaisesRegex(ValueError,"ONLY_FROZEN_NEW_LOCKED8"):
            v.count_new_locked(rows)
        rows=sample_groups(1)
        rows.pop()
        with self.assertRaises(ValueError):
            v.count_new_locked(rows)

    def test_valid_first_main_and_fail_closed_pr_or_retry(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"987"}
        receipt=v.guard(env=env,event=event,parent=v.BEFORE)
        self.assertEqual(receipt["new_model_forwards"],0)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.BEFORE)

if __name__=="__main__":
    main()
