"""Pure stdlib contracts for untouched prospective DEV8 five-native evidence."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest import TestCase, main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v6_dev8_saved_screen_test",HERE/"english_v6_mero_prospective_dev8_saved_screen_once.py")
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

def source_cell(group, cell, label, tier):
    return {"source_group_id":f"piper-new-{group}",
            "role":"development","cell_id":cell,
            "requested_class":None if cell=="nearphone-meero-challenge"
                              else (0 if cell=="ordinary-time" else 2),
            "observed_class":label,"tier":tier,
            "human_gold":False,"training_allowed":False}

def cohort(group=0, ordinary=True, positive=True):
    return [
        source_cell(group,"mero-fast",2 if positive else None,
                    "silver" if positive else "unknown"),
        source_cell(group,"mero-slow",None,"unknown"),
        source_cell(group,"ordinary-time",0 if ordinary else None,
                    "weak" if ordinary else "unknown"),
        source_cell(group,"nearphone-meero-challenge",None,"unknown")]

class ProspectiveEightSavedEvidenceTests(TestCase):
    def test_exact_saved_six_original_archives_zero_new_forwards(self):
        receipt=s.source_check()
        self.assertEqual(receipt["saved_source_rows"],32)
        self.assertEqual(receipt["review_rows"],32)
        self.assertEqual(receipt["observer_archive_count"],5)
        self.assertEqual(receipt["new_audio_or_model_forwards"],0)
        self.assertEqual(receipt["new_kws_model_forwards"],0)
        self.assertEqual(receipt["new_optimizer_updates"],0)
        self.assertEqual(receipt["new_locked_test_forwards"],0)
        self.assertFalse(receipt["training_allowed"])
        self.assertEqual(s.ORIGINAL_RUN,44556480241)
        self.assertEqual(s.ORIGINAL_STATUS,"success")
        self.assertEqual(s.SOURCE_RUN,44554345276)
        self.assertEqual(s.ORIGINAL_SHA,s.BEFORE)

    def test_nearphone_prompt_never_classifies_ordinary(self):
        d=s.policy.decide({"requested_class":None,"pcm_samples":12500},{},{})
        self.assertEqual(d["tier"],"unknown")
        self.assertIsNone(d["class"])
        self.assertFalse(d["training_allowed"])

    def test_requires_two_prospective_groups_each_with_positive_and_ordinary(self):
        good=cohort(0)+cohort(1)
        counts,groups=s.count_eligible(good)
        self.assertEqual(counts,{"development":2})
        self.assertEqual(len(groups),2)
        count2,_=s.count_eligible(cohort(0)+cohort(1,ordinary=False))
        self.assertEqual(count2,{"development":1})
        self.assertFalse(count2["development"]>=s.MIN_DEVELOPMENT_GROUPS)

    def test_no_fabricated_positive_or_trusted_nearphone(self):
        count,_=s.count_eligible(cohort(0,positive=False))
        self.assertEqual(count,{"development":0})
        data=cohort(0)
        data[-1]["tier"]="silver"
        data[-1]["observed_class"]=0
        with self.assertRaisesRegex(ValueError,"NEARPHONE_CHALLENGE"):
            s.count_eligible(data)

    def test_role_leak_and_incomplete_source_stop(self):
        data=cohort(0)
        data[0]["role"]="train"
        with self.assertRaisesRegex(ValueError,"ONLY_COMPLETE_FOUR_CELL"):
            s.count_eligible(data)
        with self.assertRaisesRegex(ValueError,"ONLY_COMPLETE_FOUR_CELL"):
            s.count_eligible(cohort(0)[:3])

    def test_first_main_only_no_pr_or_rerun(self):
        event={"before":s.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":s.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":s.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"99"}
        self.assertEqual(s.guard(env=env,event=event,parent=s.BEFORE)
                         ["new_model_forwards"],0)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            s.guard(env=env,event=event,parent=s.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):
            s.guard(env=env,event=event,parent=s.BEFORE)

if __name__=="__main__":
    main()
