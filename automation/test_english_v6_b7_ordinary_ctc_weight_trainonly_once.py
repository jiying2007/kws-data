"""B7 source-only exact B6 replay and ordinary CTC 0.60->0.85 contracts.

PR execution contains zero PyTorch installs, no optimizer and no model forward.
The actual B7 model must be produced exclusively on the one protected main push.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b7_ordinary_ctc_source_contract",
    HERE/"english_v6_b7_ordinary_ctc_weight_trainonly_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class B7MatchedTrainOnlyTests(TestCase):
    def test_frozen_B6_parent_35_groups_and_single_changed_factor(self):
        p=m.source_check()
        self.assertEqual(p["schema"],m.SCHEMA)
        self.assertEqual(p["status"],"SOURCE_ONLY_NO_TRAIN_OR_MODEL_FORWARD")
        self.assertEqual(p["train_voice_groups"],35)
        self.assertEqual(p["matched_control_optimizer_updates"],192)
        self.assertEqual(p["challenger_optimizer_updates"],192)
        self.assertEqual(p["maximum_new_optimizer_updates"],384)
        self.assertEqual(p["one_changed_factor"],
                         "ORDINARY_CTC_WEIGHT_060_TO_085_TRAIN_ONLY")
        self.assertEqual(p["ordinary_baseline_weight"],0.6)
        self.assertEqual(p["ordinary_challenger_weight"],0.85)
        self.assertEqual(p["gain_kl_weight_unchanged"],0.03)
        self.assertTrue(p["B6_tensor_replay_mandatory"])
        self.assertTrue(p["same_prefixes_and_source_groups"])
        self.assertEqual(p["new_tts_asr_phone_calls"],0)
        self.assertEqual(p["development_optimizer_rows"],0)
        self.assertEqual(p["new_model_inference_on_dev"],0)
        self.assertFalse(p["calibration_or_locked_access"])
        self.assertFalse(p["product_promotion_allowed"])
        self.assertFalse(p["work_codex"])
        self.assertFalse(p["new_schedule"])

    def test_exact_genuine_model_source_bytes_and_frozen_loss_identity(self):
        self.assertEqual(m.BEFORE,
             "04001cd9c293ddcea0ffe10652cfa464885b3b17")
        self.assertEqual(m.B6_MODEL["run"],44704535156)
        self.assertEqual(m.B6_MODEL["id"],14615025178)
        self.assertEqual(m.B6_MODEL["member"],
             "models/mero43-b6-gain-consistency192.pt")
        self.assertEqual(m.B6_MODEL["pt_sha256"],
             "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04")
        self.assertEqual(m.B6_MODEL["zip_sha256"],
             "1c310b0d4945f7aa40f2576f231c94eb48b33316a6c4f79c20215b1ac80bab8d")
        self.assertEqual(m.B6_DEV512["run"],44710802470)
        self.assertEqual(m.B6_DEV512["head"],
             "d9bbb5d0ea29ea2774067734ba24d55eccef97fc")
        self.assertEqual(m.B6_SAVED_CTC["run"],44718953294)
        self.assertEqual(m.B6_SAVED_CTC["id"],14619669982)
        self.assertEqual(m.B6_SAVED_CTC["head"],m.BEFORE)
        self.assertEqual(m.STEPS,m.b6.STEPS)
        self.assertEqual(m.SEED,m.b6.SEED)
        self.assertEqual(m.PREFIXES,m.b6.PREFIXES)
        self.assertEqual(m.GAINS,m.b6.GAINS)
        self.assertEqual(m.KL_WEIGHT,m.b6.WEIGHT)
        self.assertEqual(m.KL_CLIP,m.b6.KL_CLIP)
        self.assertAlmostEqual(m.NEW_ORDINARY_WEIGHT-m.OLD_ORDINARY_WEIGHT,0.25)
        self.assertEqual(m.PARAMETERS,43876)
        self.assertEqual(m.TRAIN_GROUPS,35)

    def test_original_B6_algorithm_and_one_delta_visible_in_source(self):
        script=(HERE/"english_v6_b7_ordinary_ctc_weight_trainonly_once.py").read_text()
        self.assertIn('control=b6.fit("consistency",parent,train,groups,cache,',script)
        self.assertIn("b6.suffix_loss(",script)
        self.assertIn("if klass==0:",script)
        self.assertIn("base_loss=base_loss+empty_target_ctc_weight_delta(",script)
        self.assertIn("loss=sum(losses)/len(losses)+KL_WEIGHT*sum(consistency)/len(consistency)",script)
        self.assertIn("clean[:,start:,:].detach()",script)
        self.assertIn("verify_B6_replay(",script)
        self.assertNotIn("DEV512.wav",script)

    def test_unavailable_or_drifted_model_archive_stops_before_any_optimizer(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,
                                        "BYTE_AUTHENTIC_B6_ORIGINAL_MODEL_ZIP"):
                m.verify_B6_replay(Path(tmp)/"missing.zip",
                                   Path(tmp)/"candidate.pt",None)
        with mock.patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,
                                        "READONLY_PUBLIC_GITHUB_ACTIONS_TOKEN"):
                m.provenance()

    def test_only_real_first_main_and_no_pr_rerun(self):
        with mock.patch.dict(os.environ,{
            "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
            "GITHUB_EVENT_NAME":"pull_request",
            "GITHUB_REF":"refs/pull/299/merge",
            "GITHUB_RUN_ATTEMPT":"1"},clear=True):
            with self.assertRaisesRegex(ValueError,
                                        "FIRST_PROTECTED_PUBLIC_MAIN_ONLY"):
                m.guard()
        with TemporaryDirectory() as tmp:
            event=Path(tmp)/"event.json"
            event.write_text(json.dumps({
                "before":"f"*40,"after":"a"*40,"ref":"refs/heads/main",
                "repository":{"full_name":m.REPO,"private":False},
                "forced":False,"deleted":False}))
            env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
                 "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
                 "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
                 "GITHUB_RUN_ID":"9876","GITHUB_EVENT_PATH":str(event)}
            with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(
                m.subprocess,"check_output",return_value=m.BEFORE+"\n"):
                with self.assertRaisesRegex(ValueError,
                                            "EXACT_FIRST_MAIN_SUCCESSOR"):
                    m.guard()

if __name__=="__main__":
    main()
