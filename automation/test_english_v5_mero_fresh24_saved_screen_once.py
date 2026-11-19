"""Original numerical evidence source checks; never call native models in tests."""
from __future__ import annotations
from collections import Counter
import importlib.util
from pathlib import Path
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v5_saved_screen_test",HERE/"english_v5_mero_fresh24_saved_screen_once.py")
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

class SavedFiveSourceTests(TestCase):
    def test_exact_six_archives_and_known_one_empty_asr(self):
        c=s.source_check()
        self.assertEqual(c["saved_source_rows"],96)
        self.assertEqual(c["review_rows"],64)
        self.assertEqual(c["observer_archive_count"],5)
        self.assertEqual(c["new_audio_or_model_forwards"],0)
        self.assertEqual(c["new_optimizer_updates"],0)
        self.assertEqual(c["new_locked_test_forwards"],0)
        self.assertFalse(c["training_allowed"])
        self.assertEqual(s.ORIGINAL_SHA,s.BEFORE)
        self.assertEqual(s.ORIGINAL_RUN,44481695979)
        self.assertEqual(s.ORIGINAL_STATUS,"failure")
        self.assertEqual(s.MISSING_ID,"mero24-452-mero-slow")

    def test_unlabeled_nearphone_always_unknown_without_models(self):
        row={"requested_class":None,"samples":12000}
        d=s.policy.decide({**row,"pcm_samples":12000},{},{})
        self.assertEqual(d["tier"],"unknown")
        self.assertIsNone(d["class"])
        self.assertFalse(d["training_allowed"])

    def test_failing_parakeet_absence_is_unknown_and_not_negative(self):
        d=s.policy.decide({"requested_class":2,"pcm_samples":12446,
                           "positive_duration_pass":True},
                          {"qwen":"Hey Mero","whisper":"Hey Mero",
                           "parakeet":None},{})
        self.assertEqual(d["tier"],"unknown")
        self.assertIsNone(d["class"])

    def test_both_classes_from_original_group_are_mandatory(self):
        def r(g,role,cls,tier):
            return {"source_group_id":g,"role":role,
                    "observed_class":cls,"tier":tier}
        counts,members=s.count_eligible([
            r("a","train",2,"silver"),r("a","train",0,"weak"),
            r("b","development",2,"silver"),r("b","development",0,"unknown"),
            r("c","train",0,"silver"),r("c","train",None,"unknown"),
            r("d","development",2,"weak"),r("d","development",0,"silver")])
        self.assertEqual(counts,{"train":1,"development":1})
        self.assertEqual(set(members),{"a","d"})
        self.assertFalse(all(counts[k]>=v for k,v in s.MIN_TRAIN_DEV.items()))

    def test_first_main_guard_failclosed(self):
        ev={"before":s.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
            "repository":{"full_name":s.REPO,"private":False},
            "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":s.REPO,
             "GITHUB_EVENT_NAME":"pull_request","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"123"}
        with self.assertRaises(ValueError):
            s.guard(env=env,event=ev,parent=s.BEFORE)
        env["GITHUB_EVENT_NAME"]="push"
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            s.guard(env=env,event=ev,parent=s.BEFORE)

if __name__=="__main__":
    main()
