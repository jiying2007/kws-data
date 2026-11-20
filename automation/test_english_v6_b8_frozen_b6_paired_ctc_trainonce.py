#!/usr/bin/env python3
"""B8 strict source-only tests. NEVER fit a model in pull_request checks."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "B8_frozen_B6_experiment",
    HERE/"english_v6_b8_frozen_b6_paired_ctc_trainonce.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class B8FrozenParentContract(unittest.TestCase):
    def test_real_historical_parent_is_byte_pinned_and_B7_stays_failed(self):
        row=m.source_check()
        self.assertEqual(row["status"],"SOURCE_ONLY_NO_OPTIMIZER_OR_MODEL_FORWARD")
        self.assertEqual(row["experiment"],
                         "B8_NEW_FROZEN_B6_PARENT_PAIRED_CONTINUATION")
        self.assertTrue(row["not_a_B7_replay_or_recovery"])
        self.assertTrue(row["b7_strict_historical_replay_stays_failed"])
        self.assertEqual(row["exact_immutable_initial_model_zip"],14615025178)
        self.assertEqual(row["exact_initial_pt_sha256"],
            "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04")
        self.assertEqual(m.B7_STOP["run"],44721025043)
        self.assertEqual(m.B7_STOP["conclusion"],"failure")
        self.assertEqual(m.B7_DRIFT["run"],44738689144)
        self.assertEqual(m.B7_DRIFT["id"],14629527557)
        self.assertEqual(row["source_groups"],35)
        self.assertEqual(row["per_arm_real_optimizer_updates"],192)
        self.assertEqual(row["maximum_new_optimizer_updates"],384)
        self.assertTrue(row["same_job_parent_tensor_bitwise_equal"])
        self.assertEqual(row["control_ordinary_ctc_weight"],0.60)
        self.assertEqual(row["challenger_ordinary_ctc_weight"],0.85)
        self.assertEqual(row["one_changed_factor"],
                         "TRAIN_ONLY_ORDINARY_NEGATIVE_CTC_060_TO_085")
        self.assertTrue(row["unchanged_seed_prefixes_gains_loss_architecture"])
        self.assertTrue(row["unmodified_B6_B7_training_implementations"])
        self.assertFalse(row["model_promotion_allowed"])

    def test_first_main_only_cannot_train_from_pr_or_replay(self):
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"12345"}
        self.assertEqual(m.validate_main_event(env,event,m.BEFORE)["parent"],m.BEFORE)
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_ONLY_NO_PR_OR_RERUN"):
            m.validate_main_event(env,event,m.BEFORE)
        env["GITHUB_EVENT_NAME"]="push"; env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_ONLY_NO_PR_OR_RERUN"):
            m.validate_main_event(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"; event["before"]="f"*40
        with self.assertRaisesRegex(ValueError,"EXACT_NEW_B8_FIRST_PROTECTED_MAIN_SUCCESSOR"):
            m.validate_main_event(env,event,m.BEFORE)
        event["before"]=m.BEFORE
        with self.assertRaisesRegex(ValueError,"EXACT_NEW_B8_FIRST_PROTECTED_MAIN_SUCCESSOR"):
            m.validate_main_event(env,event,"f"*40)

    def test_frozen_parent_missing_zip_fails_before_any_tensor(self):
        with self.assertRaisesRegex(ValueError,
                                    "BYTE_AUTHENTIC_SUCCESSFUL_B6_ZIP_REQUIRED"):
            m.load_original_B6("/not-a-real-B6-original.zip", None)

    def test_only_existing_train_paths_and_prior_model_code(self):
        src=(HERE/"english_v6_b8_frozen_b6_paired_ctc_trainonce.py").read_text()
        self.assertIn('b6.fit("consistency",parent',src)
        self.assertIn("b7.fit_challenger(",src)
        self.assertIn("prior_hashes == train_hashes",src)
        self.assertIn('and all(r["role"] == "train" for r in rows)',src)
        self.assertIn('and all(r["role"] == "development" for r in dev)',src)
        self.assertIn("EXACT_B6_B8_TRAIN_PCM_SHA_LIST_NO_LEAKAGE",src)
        self.assertIn('"model_selection_allowed": False',src)
        self.assertIn('"product_promotion_allowed": False',src)
        self.assertIn('"B7_historical_exact_replay": "FAILED_UNCHANGED"',src)
        self.assertNotIn("train_on_dev",src)
        self.assertNotIn("override_replay",src)
        self.assertEqual(m.B6_BLOB,"7731f0292fc0b30dd1d6dbb7e4b92275e255f711")
        self.assertEqual(m.B7_BLOB,"f0997e1b3d16665a41b7dd09310a2e54bd13404f")

    def test_no_exposed_dev_cal_locked_or_new_audio_in_experiment(self):
        row=m.source_check()
        self.assertEqual(row["dev_gradients"],0)
        self.assertEqual(row["dev_forwards"],0)
        self.assertEqual(row["new_tts_asr_phone"],0)
        self.assertFalse(row["cal_locked_opened"])
        self.assertFalse(row["work_codex"])
        self.assertFalse(row["private_repo"])
        self.assertFalse(row["schedule"])

if __name__=="__main__":
    unittest.main()
