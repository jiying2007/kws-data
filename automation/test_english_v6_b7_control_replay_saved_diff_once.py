"""Only original frozen B6 versus failed B7 control: zero-optimizer audit tests."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b7_control_drift_test",
    HERE/"english_v6_b7_control_replay_saved_diff_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class B7SavedControlReplayTests(TestCase):
    def test_actual_original_and_failed_ZIP_sha_is_frozen(self):
        status=m.source_check()
        self.assertEqual(status["status"],
            "SAVED_ORIGINAL_REPLAY_DRIFT_DIAGNOSTIC_ONLY")
        self.assertEqual(status["original_B6_run"],44704535156)
        self.assertEqual(status["failed_B7_run"],44721025043)
        self.assertEqual(status["original_B6_artifact"],14615025178)
        self.assertEqual(status["failed_B7_artifact"],14622635105)
        self.assertEqual(m.B6["pt"],
            "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04")
        self.assertEqual(m.B7_FAILED["digest"],
            "64eda306ba86887ceca3902c6bc2d3309574f558a992e432c5bd0c5b009295e4")
        self.assertEqual(m.B7_FAILED["job_id"],136236479236)
        self.assertEqual(status["new_model_forwards"],0)
        self.assertEqual(status["new_optimizer_updates"],0)
        self.assertFalse(status["b7_candidate_fitted"])
        self.assertFalse(status["source_and_tensor_guard_relaxed"])
        self.assertFalse(status["product_promotion_allowed"])
        self.assertEqual(m.PARENT,
            "f9f2f56eece785da0d0c79d24dca12cdbcc29b29")

    def test_missing_untrusted_zip_is_a_stop_before_any_torch(self):
        with self.assertRaisesRegex(ValueError,
                                    "EXACT_UNMUTATED_B6_OR_B7_ZIP_BYTES"):
            m.read_zip(Path("/missing/never-successful-original.zip"),m.B6)
        with self.assertRaisesRegex(ValueError,
                                    "READ_ONLY_GITHUB_ACTIONS_TOKEN"):
            m.remote_check("")

    def test_first_main_only_no_pr_rerun_or_wrong_parent(self):
        event={"before":m.PARENT,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"12345"}
        self.assertEqual(m.guard(env=env,event=event,parent=m.PARENT)["before"],m.PARENT)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_SAVED_ONLY"):
            m.guard(env=env,event=event,parent=m.PARENT)
        env["GITHUB_RUN_ATTEMPT"]="1";env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_SAVED_ONLY"):
            m.guard(env=env,event=event,parent=m.PARENT)
        env["GITHUB_EVENT_NAME"]="push";event["before"]="f"*40
        with self.assertRaisesRegex(ValueError,"EXACT_FAILED_B7_MAIN_SUCCESSOR_ONCE"):
            m.guard(env=env,event=event,parent=m.PARENT)

    def test_diff_is_observation_not_permission_to_train(self):
        src=(HERE/"english_v6_b7_control_replay_saved_diff_once.py").read_text()
        self.assertIn('training_recovery_permitted":False',src)
        self.assertIn('"previous_exact_tensor_gate_relaxed":False',src)
        self.assertIn("B6_GAIN_KL_CONTROL_TENSOR_REPLAY_DRIFT_STOP",src)
        self.assertIn("GENUINE_NONZERO_B6_CONTROL_REPLAY_MISMATCH",src)
        self.assertIn('"new_optimizer_updates":0',src)
        self.assertNotIn("optimizer.step()",src)
        self.assertNotIn("torch.onnx.export",src)

if __name__=="__main__":
    main()
