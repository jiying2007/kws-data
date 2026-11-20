"""B4 true matched low-gain FSMN training source-only regression; no optimizer in PR."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b4_exact_matched_train",HERE/"english_v6_b4_paired05_trainonly_once.py")
b4=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b4)

class TestB4Source(unittest.TestCase):
    def test_exact_b3_source_pins_training_only_two_arms(self):
        q=b4.source_check()
        self.assertEqual(q["status"],"SOURCE_ONLY_NO_OPTIMIZER_OR_AUDIO_FORWARD")
        self.assertEqual(q["target"],"Hey Mero")
        self.assertEqual(q["model_parameters"],43876)
        self.assertEqual(q["train_source_groups"],35)
        self.assertEqual(q["frozen_parent_artifact"],14590093798)
        self.assertEqual(q["prior_ctc_origin_run"],44664029084)
        self.assertTrue(q["same_b3_initialization"])
        self.assertTrue(q["same_optimizer_and_CTC"])
        self.assertEqual(q["per_trajectory_updates"],192)
        self.assertEqual(q["max_total_new_optimizer_updates"],384)
        self.assertEqual(q["only_changed_factor"],
                         "TRAIN_PCM16_GAIN_ON_SECOND_PAIRED_PASS")
        self.assertEqual(q["two_trajectories"],
                         ["control_clean_clean","challenger_clean_half"])
        self.assertEqual(b4.GAINS["control_clean_clean"],(1.,1.))
        self.assertEqual(b4.GAINS["challenger_clean_half"],(1.,.5))
        self.assertEqual(q["seed"],20261120)
        self.assertEqual(q["prefix_samples"],[6400,12800,19200])
        self.assertEqual(q["dev_model_forwards"],0)
        self.assertEqual(q["new_tts_asr_calls"],0)
        self.assertFalse(q["cal_locked_access"])
        self.assertFalse(q["product_promotion"])
        self.assertFalse(q["model_selection_allowed"])
        self.assertFalse(q["work_or_codex"])
        self.assertFalse(q["new_private_repo"])
        self.assertFalse(q["new_schedule"])

    def test_b3_sha_and_licensing_boundary_is_not_claimed_as_product(self):
        self.assertEqual(b4.B3["run"],44651053897)
        self.assertEqual(b4.B3["id"],14590093798)
        self.assertEqual(b4.B3["stem"],"mero43-onset256")
        self.assertEqual(b4.B3["pt"],
                         "682e2689e82aa91d850516c06b2be61efc64731cb1dee3b70a7d2a8e01925a46")
        self.assertEqual(b4.BEFORE,"5443f4e6b3526bdd0aaef6b00a28594b044dbf50")
        self.assertEqual(b4.PREVIOUS_CTC_ARTIFACT,14596776938)
        self.assertEqual(b4.PREVIOUS_CTC_SHA,
                         "d725bddbf73b2b6f3d1a89bb46f1ad1f6aa63634af84b762587460fa6bd00cee")

    def test_exact_35_balanced_source_groups_and_exclusions(self):
        train=[]
        for index in range(35):
            for klass in (0,1):
                train.append({"id":f"t{index}_{klass}",
                    "role":"train","mono_class":klass,
                    "source_group_id":f"family{index:02d}",
                    "source_pcm_sha256":"a"*64,"tier":"silver"})
        groups=b4.input_groups(train)
        self.assertEqual(len(groups[0]),35)
        self.assertEqual(len(groups[1]),35)
        self.assertEqual(set(groups[0]),set(groups[1]))
        train[0]["role"]="development"
        with self.assertRaisesRegex(ValueError,"ONLY_AUTHENTICATED_TRAIN"):
            b4.input_groups(train)
        train[0]["role"]="train"
        train[0]["mono_class"]=None
        with self.assertRaisesRegex(ValueError,"ONLY_AUTHENTICATED_TRAIN"):
            b4.input_groups(train)
        train[0]["mono_class"]=0
        del train[0]
        with self.assertRaisesRegex(ValueError,"COMPLETE_BALANCED"):
            b4.input_groups(train)

    def test_invalid_branch_context_never_starts_optimizer(self):
        with patch.dict("os.environ",{
           "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":"jiying2007/kws-data",
           "GITHUB_EVENT_NAME":"pull_request",
           "GITHUB_REF":"refs/pull/281/merge",
           "GITHUB_RUN_ATTEMPT":"1"},clear=True):
            with self.assertRaisesRegex(ValueError,"ONE_PROTECTED_PUBLIC_MAIN"):
                b4.guard()
        with patch.dict("os.environ",{},clear=True):
            with self.assertRaisesRegex(ValueError,"READONLY_GITHUB_TOKEN_REQUIRED"):
                b4.verify_provenance()

    def test_gain_factor_is_standalone_and_no_new_keyword(self):
        self.assertEqual(set(b4.GAINS),set(b4.ARTIFACT_STEMS))
        self.assertEqual(b4.PREFIX_SAMPLES,(6400,12800,19200))
        self.assertEqual(b4.TRAIN_GROUPS,35)
        self.assertEqual(b4.PARAMS,43876)
        self.assertEqual(b4.LR,0.00006)
        with self.assertRaisesRegex(ValueError,"ONLY_CLEAN_AND_HALF"):
            b4.gain_pcm(b"\0\0",0.75,None)
        self.assertEqual(b4.gain_pcm(b"\x01\x00",1.0,None),b"\x01\x00")

if __name__=="__main__":
    unittest.main()
