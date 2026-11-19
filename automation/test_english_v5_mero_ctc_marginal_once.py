"""No numeric work: exact frozen V5 full-sequence CTC marginal study contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "english_v5_ctc_ratio_source_test",
    HERE/"english_v5_mero_ctc_marginal_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExactMonoCTCMarginal(unittest.TestCase):
    def test_only_authenticated_previous_weights_and_17_inputs(self):
        r=m.source_check()
        self.assertEqual(r["original_model_run"],44453447586)
        self.assertEqual(r["model_artifact"],14496732088)
        self.assertEqual(r["max_new_model_forwards"],17)
        self.assertEqual(r["new_optimizer_updates"],0)
        self.assertEqual(r["new_generated_audio"],0)
        self.assertFalse(r["calibration_or_locked_used"])

    def test_exact_protected_main_guard_and_no_retry(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"445000000"}
        event={"before":m.BEFORE,"after":"a"*40,
               "ref":"refs/heads/main","forced":False,"deleted":False,
               "repository":{"full_name":m.REPO,"private":False}}
        self.assertEqual(m.guard(env,event,m.BEFORE)["original_weight_run"],
                         44453447586)
        with self.assertRaises(ValueError):
            m.guard({**env,"GITHUB_RUN_ATTEMPT":"2"},event,m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env,{**event,"before":"b"*40},m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env,{**event,"repository":{"full_name":m.REPO,
                "private":True}},m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env,event,"b"*40)

    def test_only_original_dev_comparison_not_threshold_selection(self):
        data=[
            {"mono_class":1,"margin":{"target_minus_all_blank":4.0}},
            {"mono_class":1,"margin":{"target_minus_all_blank":3.0}},
            {"mono_class":0,"margin":{"target_minus_all_blank":1.0}},
            {"mono_class":0,"margin":{"target_minus_all_blank":2.0}},
        ]
        r=m.separate(data)
        self.assertEqual(r["positive_count"],2)
        self.assertEqual(r["ordinary_count"],2)
        self.assertTrue(r["strict_margin_separation_observed"])
        self.assertEqual(r["exploratory_pairwise_rank_fraction"],1.0)
        self.assertFalse(r["calibration_performed"])
        self.assertFalse(r["release_threshold_selected"])
        data[-1]["margin"]["target_minus_all_blank"]=5.0
        r=m.separate(data)
        self.assertFalse(r["strict_margin_separation_observed"])
        self.assertEqual(r["exploratory_pairwise_rank_fraction"],0.5)

    def test_missing_signed_model_zip_never_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                m.saved.audit_model(Path(tmp)/"missing-model.zip")

    def test_no_optimization_no_new_audio_or_fake_model_promotion(self):
        text=(HERE/"english_v5_mero_ctc_marginal_once.py").read_text()
        self.assertNotIn("optimizer.step(",text)
        self.assertNotIn("torch.onnx.export(",text)
        self.assertNotIn("threshold = float(",text)
        self.assertIn("ctc_loss(",text)
        self.assertIn("old_calibration_or_locked_reopened",text)
        self.assertIn("is_streaming_event_detector",text)


if __name__=="__main__":
    unittest.main()
