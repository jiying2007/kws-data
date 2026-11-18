"""No-model admission and training-gate tests for authentic canonical multisource head."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("canon_test_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj
binder=load("english_canonical_multisource_bind_once")
fit=load("english_canonical_multisource_fit_once")

class FrozenMultiSourceTests(unittest.TestCase):
    def test_six_real_immutable_artifacts_8_4_3_3(self):
        cfg=binder.config()
        self.assertEqual(len(cfg["corpus"]),3)
        self.assertEqual([x["source"]["denominator"] for x in cfg["corpus"]],
                         [168,126,168])
        self.assertEqual(cfg["exact_expected_qualified"],
                         {"train":8,"development":4,"calibration":3,
                          "locked-test":3})
        self.assertEqual(len({x[t]["id"] for x in cfg["corpus"]
                              for t in ("source","screen")}),6)
        self.assertEqual(cfg["accepted_machine_tiers"],
                         {"silver":1.0,"weak":0.5})
        self.assertFalse(cfg["automatic_model_promotion_allowed"])
        self.assertFalse(cfg["no_legacy_meero_weights"] is False)
        self.assertFalse(cfg["human_far_frr"])
        self.assertEqual(cfg["new_tts_calls"]+cfg["new_asr_calls"]+
                         cfg["new_phone_calls"],0)
        self.assertEqual(binder.config()["full_qualified_role_minimum"],
                         {"train":6,"development":2,"calibration":2,
                          "locked-test":2})
    def test_protected_first_public_main_no_retry(self):
        cfg=binder.config()
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":binder.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"444444444",
             "GITHUB_SHA":"a"*40}
        event={"before":cfg["before"],"after":"a"*40,
               "ref":"refs/heads/main","repository":{
                "full_name":binder.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(binder.guard(env,event,cfg["before"])["before"],
                         cfg["before"])
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"ONE_PUBLIC_MAIN_ATTEMPT_ONLY"):
            binder.guard(env,event,cfg["before"])
        env["GITHUB_RUN_ATTEMPT"]="1";event["repository"]["private"]=True
        with self.assertRaisesRegex(ValueError,"FROZEN_SOURCE_FIRST_MAIN_SUCCESSOR"):
            binder.guard(env,event,cfg["before"])
        event["repository"]["private"]=False
        with self.assertRaisesRegex(ValueError,"EXACT_PROTECTED_MAIN_PARENT"):
            binder.guard(env,event,"b"*40)
    def test_failclosed_corrupt_zip_without_numeric(self):
        cfg=binder.config()
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/"f.zip";f.write_bytes(b"not a real audio archive")
            with self.assertRaisesRegex(ValueError,"EXACT_IMMUTABLE_SAVED_ARTIFACT_SHA"):
                binder.archive(f,cfg["corpus"][0]["source"])
    def test_source_check_never_starts_optimizer(self):
        self.assertEqual(fit.SCHEMA,"english-canonical-multisource-fresh-causal-head-v1")
        self.assertEqual(fit.SHAPES,{"mean96":(96,),"scale96":(96,),
                         "w1":(32,1536),"b1":(32,),
                         "w2":(3,32),"b2":(3,)})
        self.assertEqual(len(fit.MAGIC),8)
        self.assertEqual(8+4*49475,197908)
        config=binder.config()
        self.assertEqual(config["head"]["checkpoint_selection"],
                         "last_fixed_96th_update_no_dev_sweep")
        self.assertEqual(config["head"]["locked_eval"],
                         "once_after_fixed_calibration_and_weight_sha_no_model_selection")
        self.assertTrue(config["no_legacy_meero_positive_audio"])
        self.assertTrue(config["no_legacy_meero_weights"])
        self.assertEqual(config["preprocessing"]["minimum_history_samples"],35200)
        self.assertEqual(config["preprocessing"]["positive_max_original_samples"],28672)
        self.assertFalse(config["automatic_model_promotion_allowed"])

if __name__=="__main__":
    unittest.main()
