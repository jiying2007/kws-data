"""Source-only B5 training guards. Model training is prohibited on PR."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b5_source_contract",HERE/"english_v6_b5_hey_margin_trainonly_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class B5Test(unittest.TestCase):
    def test_exact_source_policy_and_single_loss_change(self):
        c=m.source_check()
        self.assertEqual(c["status"],"SOURCE_ONLY_NO_TRAIN_OR_MODEL_FORWARD")
        self.assertEqual(c["target"],"Hey Mero")
        self.assertEqual(c["train_source_groups"],35)
        self.assertEqual(c["per_arm_optimizer_updates"],192)
        self.assertEqual(c["maximum_new_optimizer_updates"],384)
        self.assertEqual(c["only_changed_factor"],
                         "EARLY_SUFFIX_HEY_VS_BLANK_AUXILIARY")
        self.assertEqual(c["same_gains_both_arms"],[1.,.5])
        self.assertEqual(c["early_hey_frame_max"],48)
        self.assertEqual(c["hey_loss_weight"],.08)
        self.assertTrue(c["baseline_replay_required"])
        self.assertTrue(c["same_43876_param_b3_parent"])
        self.assertTrue(c["same_seed_and_optimizer"])
        self.assertFalse(c["product_promotion_allowed"])
        self.assertFalse(c["model_selection_allowed"])
        self.assertEqual(c["dev_model_forwards"] if "dev_model_forwards" in c
                         else c["development_model_forwards"],0)
        self.assertFalse(c["cal_locked_opened"])
        self.assertEqual(c["new_tts_asr"],0)
        self.assertFalse(c["work_codex"])
        self.assertFalse(c["new_schedule"])
        self.assertFalse(c["private_repo"])

    def test_immutable_real_provenance_and_no_dev_reuse(self):
        self.assertEqual(m.BEFORE,
            "1cf55598f76aa739da93d5b39578f118f73f96ea")
        self.assertEqual(m.B4_TRAIN["run"],44665669003)
        self.assertEqual(m.B4_TRAIN["id"],14597862669)
        self.assertEqual(m.B4_TRAIN["member"],
                         "models/mero43-b4-paired05-192.pt")
        self.assertEqual(m.B4_TRAIN["pt_sha256"],
            "c795f8b5f6bf8e9c37d195126fc25006dbde92fb71591473c05854e3d3698f15")
        self.assertEqual(m.EXPOSED_DEV["run"],44685455242)
        self.assertEqual(m.EXPOSED_DEV["id"],14606730739)
        self.assertEqual(set(m.ARMS),{"control","early_hey"})
        self.assertEqual(m.STEPS,m.b4.STEPS)
        self.assertEqual(m.SEED,m.b4.SEED)
        self.assertEqual(m.LR,m.b4.LR)
        self.assertEqual(m.PREFIXES,m.b4.PREFIX_SAMPLES)
        self.assertEqual(m.GAINS,m.b4.GAINS["challenger_clean_half"])

    def test_no_optimizer_on_pr_or_replayed_main(self):
        with patch.dict(os.environ,{
            "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
            "GITHUB_EVENT_NAME":"pull_request",
            "GITHUB_REF":"refs/pull/286/merge",
            "GITHUB_RUN_ATTEMPT":"1"},clear=True):
            with self.assertRaisesRegex(ValueError,
                "FIRST_PROTECTED_PUBLIC_MAIN_ONLY"):
                m.guard()
        with TemporaryDirectory() as td:
            event=Path(td)/"event.json"
            event.write_text(json.dumps({
                "before":"f"*40,"after":"a"*40,"ref":"refs/heads/main",
                "repository":{"full_name":m.REPO,"private":False},
                "forced":False,"deleted":False}))
            env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
                 "GITHUB_EVENT_NAME":"push",
                 "GITHUB_REF":"refs/heads/main",
                 "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
                 "GITHUB_RUN_ID":"12345","GITHUB_EVENT_PATH":str(event)}
            with patch.dict(os.environ,env,clear=True),patch.object(
                m.subprocess,"check_output",return_value=m.BEFORE+"\n"):
                with self.assertRaisesRegex(ValueError,
                    "NO_OUT_OF_ORDER_OR_REPLAY"):
                    m.guard()

    def test_new_output_and_artifact_fail_closed(self):
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,
                                       "BYTE_AUTHENTIC_B4_HALF_ORIGINAL"):
                m.verify_replay(Path(td)/"missing.zip",
                                Path(td)/"candidate.pt",None)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,
                                       "GITHUB_READ_ONLY_TOKEN_REQUIRED"):
                m.provenance()

    def test_train_metadata_rejects_dev_gradient_and_unbalanced_groups(self):
        train=[]
        for i in range(35):
            for klass in (0,1):
                train.append({"id":f"{i}-{klass}","role":"train",
                              "mono_class":klass,
                              "source_group_id":f"family{i}",
                              "source_pcm_sha256":"a"*64,"tier":"silver"})
        grouped=m.b4.input_groups(train)
        self.assertEqual(len(grouped[0]),35)
        self.assertEqual(len(grouped[1]),35)
        train[0]["role"]="development"
        with self.assertRaisesRegex(ValueError,
                                    "ONLY_AUTHENTICATED_TRAIN_PCM_ROWS"):
            m.b4.input_groups(train)
        train[0]["role"]="train"
        train.pop()
        with self.assertRaisesRegex(ValueError,"COMPLETE_BALANCED_35"):
            m.b4.input_groups(train)

if __name__=="__main__":
    unittest.main()
