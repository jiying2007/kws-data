"""Stdlib-only contracts for Hey Mero 43K train-only noise/echo candidate."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest import TestCase,main
import tempfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("mero43_robust_test",
    HERE/"english_v5_mero43_trainonly_robust_challenger_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

class RobustCandidateContracts(TestCase):
    def test_original_43k_checkpoint_and_no_retraining_on_pr(self):
        data=v.source_check()
        self.assertEqual(data["status"],"SOURCE_ONLY_NO_OPTIMIZER_OR_AUDIO_FORWARD")
        self.assertEqual(data["keyword"],"Hey Mero")
        self.assertEqual(data["source_train_groups"],35)
        self.assertEqual(data["new_prospective_dev_machine_positive"],2)
        self.assertEqual(data["new_prospective_dev_ordinary"],8)
        self.assertEqual(data["predeclared_updates"],288)
        self.assertEqual(data["predeclared_learning_rate"],0.000055)
        self.assertEqual(data["augmentation_snr_db"],(None,10,5))
        self.assertFalse(data["old_exposed_dev_audio_opened"])
        self.assertFalse(data["calibration_locked_audio_opened"])
        self.assertFalse(data["product_promotion_allowed"])
        self.assertEqual(v.MODEL["id"],14514228406)
        self.assertEqual(v.MODEL["params"],43876)

    def test_first_main_no_replay_guard(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"44599999999"}
        self.assertEqual(v.guard(env=env,event=event,parent_sha=v.BEFORE)
                         ["parent_sha"],v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"ONLY_FIRST"):
            v.guard(env=env,event=event,parent_sha=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["before"]="b"*40
        with self.assertRaisesRegex(ValueError,"ONE_FROZEN_MAIN_SUCCESSOR"):
            v.guard(env=env,event=event,parent_sha=v.BEFORE)
        event["before"]=v.BEFORE
        with self.assertRaisesRegex(ValueError,"EXACT_UNREPLAYABLE"):
            v.guard(env=env,event=event,parent_sha="c"*40)
        event["repository"]["private"]=True
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent_sha=v.BEFORE)

    def test_audit_cannot_accept_missing_original_model_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            fake=Path(folder)/"fake.zip"
            fake.write_bytes(b"no real model")
            # Reject admission before any acoustic work: fixed model ZIP.
            self.assertNotEqual(v.sha(fake.read_bytes()),v.MODEL["sha"])
            self.assertNotEqual(fake.stat().st_size,501276)

    def test_exact_experimental_boundaries(self):
        self.assertEqual(v.DEV_CONDITIONS,
                         ("original","attenuated_0p5","snr10"))
        self.assertEqual(v.STEPS,288)
        self.assertEqual(v.GAIN,tuple(v.gain.GAIN_LEVELS))
        self.assertEqual(v.SNR,(None,10,5))
        self.assertEqual(v.BEFORE,"f2f60076b5524c2c68ac0d279916303559a5f7bf")
        self.assertEqual(v.MAX_DEV,10)
        self.assertEqual(v.SCHEMA,
            "en-v5-frozen43k-trainonly-gain-noise-reflections-288-v1")

if __name__=="__main__":main()
