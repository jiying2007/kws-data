"""Source-only tests for frozen 43K PyTorch/ONNX comparison, no model calls."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest import TestCase, main, mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "mero43_onnx_test",HERE/"english_v5_mero43_onnx_parity_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

class HeyMero43kParityContracts(TestCase):
    def test_exact_one_model_and_new_development_source(self):
        p=v.source_check()
        self.assertEqual(p["status"],"SOURCE_ONLY_PT_ONNX_PARITY_NOT_RUN")
        self.assertEqual(p["original_model_artifact"],14514228406)
        self.assertEqual(p["original_dev_audio_count"],10)
        self.assertEqual(p["model_onnx_sha256"],v.ONNX_SHA256)
        self.assertEqual(p["max_abs_logit_tolerance"],0.001)
        self.assertEqual(p["max_mean_abs_logit_tolerance"],0.0001)
        self.assertEqual(p["new_optimizer_calls"],0)
        self.assertEqual(p["new_asr_tts_calls"],0)
        self.assertEqual(p["new_locked_or_calibration_forwards"],0)
        self.assertFalse(p["product_promotion"])
    def test_only_original_machine_verified_ten_no_locked(self):
        rows=[{"id":f"test-{i}","role":"development",
               "source_group_id":f"source-{i%8}",
               "mono_class":int(i<2)} for i in range(10)]
        with mock.patch.object(v.prior,"select",return_value=(rows,{
              "no_locked_or_calibration":True})):
            out,receipt=v.audit(Path("/tmp/source"),Path("/tmp/screen"))
        self.assertEqual(len(out),10)
        self.assertTrue(receipt["no_locked_or_calibration"])
        rows[0]["role"]="locked-test"
        # Source evidence is always verified by the upstream verifier,
        # and no role mutation is allowed here.
        with mock.patch.object(v.prior,"select",return_value=(rows,{
                "no_locked_or_calibration":False})):
            with self.assertRaises(ValueError):
                v.audit(Path("/tmp/source"),Path("/tmp/screen"))
    def test_repeated_main_or_pr_is_not_inference(self):
        ev={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
            "repository":{"full_name":v.REPO,"private":False},
            "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
           "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
           "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
           "GITHUB_RUN_ID":"300"}
        self.assertEqual(v.guard(env=env,event=ev,parent=v.BEFORE)["new_optimizer_calls"],0)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):v.guard(env=env,event=ev,parent=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1";env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):v.guard(env=env,event=ev,parent=v.BEFORE)
if __name__=="__main__":main()
