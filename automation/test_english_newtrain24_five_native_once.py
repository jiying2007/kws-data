"""Fail-closed, no-model checks of new source-bound independent five-observer review."""
from __future__ import annotations
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("newtrain24_five_observers",
    HERE/"english_newtrain24_five_native_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ExactNewtrain24Observer(unittest.TestCase):
    def test_exact_new_authentic_source_and_readonly_five_models(self):
        result=m.source_check()
        self.assertEqual(m.SOURCE_RUN,44431769603)
        self.assertEqual(m.SOURCE["id"],14486048125)
        self.assertEqual(m.SOURCE["size"],3353181)
        self.assertEqual(m.SOURCE["digest"],
            "sha256:3db5e99caf33e04ae920df1c93205f38f98cbc0f326705e2367dc8c67dc02341")
        self.assertEqual(m.original_identity(),{
            "run_id":"44431769603","commit":"9113713ad4fcb87e111f373db35922f61775de1e",
            "before":"7971b74e3c59fe23a5609c2df6e47939427ea984"})
        self.assertEqual(result["source_rows"],168)
        self.assertEqual(result["observer_roster"],["qwen","whisper","parakeet","xlsr","lv60"])
        self.assertEqual(result["new_tts_calls"],0)
        self.assertFalse(result["training_allowed"])
        p=m.source_worker.protocol()
        self.assertEqual([g["role"] for g in p["groups"]],["train"]*18+["development"]*6)
        self.assertEqual([g["speaker_index"] for g in p["groups"]],list(range(424,448)))
    def test_first_public_main_and_no_consumed_run_retry(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
          "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
          "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"444400011",
          "GITHUB_SHA":"a"*40}
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
          "repository":{"full_name":m.REPO,"private":False},
          "forced":False,"deleted":False}
        self.assertEqual(m.guard(env,event,m.BEFORE)["source_run"],44431769603)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):m.guard(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["repository"]["private"]=True
        with self.assertRaises(ValueError):m.guard(env,event,m.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaises(ValueError):m.guard(env,event,"b"*40)
    def test_missing_source_fails_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"BOUNDED_SOURCE_FILE"):
                m.source_rows(tmp)
        self.assertTrue(m.source_check()["status"]=="SOURCE_PINNED")
        self.assertFalse(m.source_worker.protocol()["training_allowed"])
        self.assertEqual(m.FROZEN_POLICY_GIT_SHA1,
            "0a688325d0d36e83ef6623ac678511df24a92374")
    def test_workflow_preserves_native_gates_and_limits(self):
        w=(HERE.parent/".github/workflows/english-newtrain24-five-native-once.yml").read_text()
        self.assertIn("github.event.before == '"+m.BEFORE+"'",w)
        self.assertIn("github.run_attempt == 1",w)
        self.assertIn("matrix:",w)
        self.assertIn("provider: [qwen, whisper, parakeet, xlsr, lv60]",w)
        self.assertIn("--audit-source",w)
        self.assertIn("--verify-remote",w)
        self.assertIn("--verify-source",w)
        self.assertIn("--observe --provider",w)
        self.assertIn("if-no-files-found: warn",w)
        self.assertNotIn("continue-on-error: true",w)
        self.assertNotIn("schedule:",w)
if __name__=="__main__":unittest.main()
