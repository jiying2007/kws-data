"""No numerical forward on PR: fixed source and dev-only candidate selection."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("dev_only_v2",HERE/"english_canonical_dropout_challenger_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DropoutRegularizedChallenger(unittest.TestCase):
    def test_new_model_never_reuses_locked_pcm_for_selection(self):
        actual=m.source_check()
        self.assertEqual(actual["max_real_updates"],384)
        self.assertEqual(actual["train_only_regularization"],
                         {"dropout":0.25,"adamw_learning_rate":0.0003})
        self.assertEqual(actual["dev_selection_steps"],[96,192,384])
        self.assertEqual(actual["source_groups"],
            {"train":8,"development":4,"calibration":3,"locked-test":3})
        self.assertEqual(actual["dev_working_threshold"],0.8)
        self.assertFalse(actual["old_locked_kws_test_reuse"])
        self.assertFalse(actual["calibration_audio_opened"])
        self.assertFalse(actual["locked_audio_opened"])
        self.assertFalse(actual["training_allowed"])
    def test_candidate_order_and_dev_metrics_are_fixed(self):
        def candidate(update,miss,events):
            return {"update":update,"threshold":0.8,
                    "selection_key":[miss,events,miss/2,update],
                    "development":{"source_groups":4,
                        "classes":{"0":{"cases":8},
                                   "1":{"cases":8},"2":{"cases":10}}}}
        c=[candidate(96,.7,1),candidate(192,.5,2),candidate(384,.5,3)]
        self.assertEqual(m.choose(c)["update"],192)
        with self.assertRaisesRegex(ValueError,"EXACT_THREE_NEW_DEV_ONLY_CANDIDATES"):
            m.choose(c[:-1])
        c[0]["threshold"]=0.9
        with self.assertRaisesRegex(ValueError,"PREDECLARED_DEV_WORKING_POINT"):
            m.choose(c)
    def test_public_main_first_parent_and_attempt_one(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"44412345",
             "GITHUB_SHA":"a"*40}
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(m.guard(env,event,m.BEFORE)["before"],m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"EXACT_PUBLIC_MAIN_FIRST_ATTEMPT"):
            m.guard(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1";event["repository"]["private"]=True
        with self.assertRaisesRegex(ValueError,"EXACT_FIRST_MAIN_SUCCESSOR_WITHOUT_REPLAY"):
            m.guard(env,event,m.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaisesRegex(ValueError,"ORIGINAL_FROZEN_MAIN_FIRST_PARENT"):
            m.guard(env,event,"b"*40)
    def test_only_new_dev_head_and_not_main_release(self):
        self.assertEqual(m.MODEL_PARAMETERS,49283)
        self.assertEqual(m.SEED,20261121)
        self.assertEqual(m.LR,0.0003)
        self.assertEqual(m.DROPOUT,0.25)
        self.assertEqual(m.UPDATES,(96,192,384))
        self.assertEqual(m.base.MAGIC,b"EHKWS01\x00")
        self.assertEqual(m.base.SHAPES["w1"],(32,1536))
if __name__=="__main__":
    unittest.main()
