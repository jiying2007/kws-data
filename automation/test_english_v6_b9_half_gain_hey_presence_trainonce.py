#!/usr/bin/env python3
"""B9 one-factor source-only tests; PR runs MUST NOT fit or score audio."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "B9_half_gain_hey_presence_contract",
    HERE/"english_v6_b9_half_gain_hey_presence_trainonce.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class B9TrainOnlyContract(unittest.TestCase):
    def test_real_frozen_b6_parent_and_no_product_quality_claim(self):
        s=m.source_check()
        self.assertEqual(s["status"],"SOURCE_ONLY_NO_TRAIN_OR_MODEL_FORWARD")
        self.assertEqual(s["keyword"],"Hey Mero")
        self.assertEqual(s["architecture"],"FSMN-CTC-43876")
        self.assertEqual(s["same_35_source_disjoint_train_groups"],35)
        self.assertEqual(s["per_arm_optimizer_updates"],192)
        self.assertEqual(s["max_actual_optimizer_updates"],384)
        self.assertEqual(s["control_weight"],0.0)
        self.assertEqual(s["challenger_weight"],0.03)
        self.assertEqual(s["new_factor"],
            "CLIP_LEVEL_HEY_PRESENCE_POSITIVE_HALF_GAIN_TRAIN_ONLY")
        self.assertTrue(s["no_guessed_hey_frame_labels"])
        self.assertTrue(s["unchanged_suffix_ctc"])
        self.assertTrue(s["unchanged_blank_prefix"])
        self.assertTrue(s["unchanged_gain_kl"])
        self.assertEqual(s["unchanged_ordinary_ctc_weight"],0.60)
        self.assertTrue(s["same_seed_optimizer_source_prefixes_gain_policy"])
        self.assertEqual(s["prior_B7_exact_replay"],"FAILED_UNCHANGED")
        self.assertEqual(s["dev_gradients"],0)
        self.assertEqual(s["dev_model_forwards"],0)
        self.assertEqual(s["tts_asr_phone_calls"],0)
        self.assertTrue(s["B8_dev520_is_exposed_and_not_train"])
        self.assertFalse(s["automated_model_selection"])
        self.assertFalse(s["product_promotion"])
        self.assertFalse(s["cal_locked_access"])
        self.assertFalse(s["private_repo"])
        self.assertFalse(s["work_codex"])
        self.assertFalse(s["new_scheduled_patrol"])

    def test_exact_b8_real_training_and_dev520_receipts(self):
        self.assertEqual(m.BEFORE,
            "0ac4c653a7ae6b3cf90812ce6bd0402b4ba0d04f")
        self.assertEqual(m.B8_TRAIN["run"],44746571932)
        self.assertEqual(m.B8_TRAIN["artifact"],14634066012)
        self.assertEqual(m.B8_TRAIN["bytes"],1001683)
        self.assertEqual(m.B8_TRAIN["digest"],
            "6015c6269f87654193d7350afd0c5b44f799f5f71b5303e41f2734c17006d15b")
        self.assertEqual(m.B8_DEV["run"],44766393361)
        self.assertEqual(m.B8_DEV["artifact"],14645830993)
        self.assertEqual(m.B8_DEV["bytes"],49216)
        self.assertEqual(m.B8_DEV["digest"],
            "3f39922e0ec0aad5baa4ffc161c46ebe95d3507defe158442bb4a761e0f0eee1")
        self.assertEqual(m.STEPS,192)
        self.assertEqual(m.WEIGHT,0.03)
        self.assertEqual(m.PARAMS,43876)
        self.assertEqual(m.b8.B6["pt_sha256"],
            "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04")

    def test_one_shot_exact_protected_main_never_pr_or_rerun(self):
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "deleted":False,"forced":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"12345"}
        self.assertEqual(m.validate_main_event(env,event,m.BEFORE)["parent"],
                         m.BEFORE)
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaisesRegex(ValueError,"FIRST_MAIN_ONLY_NO_PR"):
            m.validate_main_event(env,event,m.BEFORE)
        env["GITHUB_EVENT_NAME"]="push"
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"FIRST_MAIN_ONLY_NO_PR"):
            m.validate_main_event(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["before"]="f"*40
        with self.assertRaisesRegex(ValueError,"EXACT_B9_PROTECTED_MAIN"):
            m.validate_main_event(env,event,m.BEFORE)
        event["before"]=m.BEFORE
        with self.assertRaisesRegex(ValueError,"EXACT_B9_PROTECTED_MAIN"):
            m.validate_main_event(env,event,"f"*40)
        event["repository"]["private"]=True
        with self.assertRaisesRegex(ValueError,"EXACT_B9_PROTECTED_MAIN"):
            m.validate_main_event(env,event,m.BEFORE)

    def test_only_positive_half_gain_receives_new_loss(self):
        src=(HERE/"english_v6_b9_half_gain_hey_presence_trainonce.py").read_text()
        self.assertIn("if klass==1 and gain==0.5:",src)
        self.assertIn("onsets.append(presence_loss(",src)
        self.assertIn("torch.log1p(-p).sum(dim=1)",src)
        self.assertIn("prob_ever_hey=-torch.expm1(log_no_hey)",src)
        self.assertIn("loss=loss+WEIGHT*onsets[0]",src)
        self.assertIn('b6.fit("consistency",parent,rows,groups,cache,',src)
        self.assertIn("source_audit(a)",src)
        self.assertIn('r["role"]=="train" for r in rows',src)
        self.assertIn('r["role"]=="development" for r in dev',src)
        self.assertIn("EXACT_B6_B9_TRAIN_PCM_SHA_LIST",src)
        self.assertIn('and all(parent[k].shape==b3[k].shape for k in parent)',src)
        self.assertIn('torch.use_deterministic_algorithms(True)',src)
        self.assertIn('same_job_parent":True',src)
        self.assertIn('"fresh_independent_B9_DEV":"NOT_RUN"',src)
        self.assertIn('"model_selection_allowed":False',src)
        self.assertIn('"product_promotion_allowed":False',src)
        self.assertIn('"B7_historical_exact_replay":"FAILED_UNCHANGED"',src)
        self.assertNotIn("train_on_dev",src)
        self.assertNotIn("override_B7_replay",src)

    def test_no_model_zip_means_no_new_fit(self):
        with self.assertRaisesRegex(ValueError,
                                    "BYTE_AUTHENTIC_SUCCESSFUL_B6_ZIP_REQUIRED"):
            m.b8.load_original_B6("/missing/actual-B6.zip",None)

if __name__=="__main__":
    unittest.main()
