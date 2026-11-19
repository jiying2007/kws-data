"""Source-only immutable parent / gain hypothesis checks, no model inference."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "gain_refit_test", HERE/"english_v5_mero_gain_invariant_refit_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SingleMeroGainRefit(unittest.TestCase):
    def test_parent_real_weight_and_explicit_gain_grid(self):
        v=m.source_check()
        self.assertEqual(v["only_keyword"],"Hey Mero")
        self.assertEqual(v["frozen_parent_run"],44471889167)
        self.assertEqual(v["frozen_parent_artifact"],14504422674)
        self.assertEqual(v["gain_levels"],m.GAIN_LEVELS)
        self.assertEqual(v["new_optimizer_updates"],0)
        self.assertEqual(m.UPDATES,144)
        self.assertEqual(m.LR,0.0001)
        self.assertEqual(m.PARENT_ZIP_SHA,
            "1c6c96129fe0223a92e09b6141d70c90ff72f7a3bd8144e29827f0cddaa476d0")
        self.assertEqual(m.PARENT_PT_SHA,
            "483c7ede52f2308a461202da7823af38d1353e98287aba3f926705b1e0ea9fd3")
        self.assertFalse(v["formal_promotion_allowed"])
        self.assertFalse(v["dev_audio_in_training"])
        self.assertFalse(v["calibration_or_locked_opened"])

    def test_do_not_accept_unverified_fake_parent_weights(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"not-real-model.zip"
            p.write_bytes(b"fake")
            with self.assertRaises(ValueError):
                m.parent_packet(p)

    def test_first_same_repo_PR_only_and_no_retries(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"pull_request","GITHUB_HEAD_REF":m.BRANCH,
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"44480000000"}
        event={"action":"opened",
            "repository":{"full_name":m.REPO,"private":False},
            "pull_request":{"head":{"ref":m.BRANCH,
                "sha":"a"*40,"repo":{"full_name":m.REPO}},
                            "base":{"ref":"main"}}}
        self.assertEqual(m.run_guard(env,event)["head"],"a"*40)
        for test_env,test_event in (
            ({**env,"GITHUB_RUN_ATTEMPT":"2"},event),
            ({**env,"GITHUB_EVENT_NAME":"push"},event),
            (env,{**event,"action":"closed"}),
            (env,{**event,"repository":{"full_name":m.REPO,"private":True}}),
            (env,{**event,"pull_request":{**event["pull_request"],
                   "head":{"ref":"other","sha":"a"*40,
                           "repo":{"full_name":m.REPO}}}),
        ):
            with self.assertRaises(ValueError):
                m.run_guard(test_env,test_event)

    def test_train_gain_is_not_dev_selection_and_no_forged_phone_timing(self):
        code=(HERE/"english_v5_mero_gain_invariant_refit_once.py").read_text()
        self.assertIn("def derived_level",code)
        self.assertIn("train_only_consistency",code)
        self.assertIn("optimizer.step()",code)
        self.assertIn("teacher[row[",code)
        self.assertIn("parent.authorized_source",code)
        self.assertIn("old_other_keyword_gradients",code)
        self.assertIn("original_pcm_only_no_gain_or_threshold_selection",code)
        self.assertIn('"real_far_frr":"NOT_MEASURED"',code)
        self.assertIn('"product_promotion_allowed":False',code)
        self.assertNotIn("Hey Vosen",code)
        self.assertNotIn("ctc_phone_timestamp_pseudo_truth",code)
        self.assertNotIn("for step in range(512)",code)


if __name__=="__main__":
    unittest.main()
