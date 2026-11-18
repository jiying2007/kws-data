"""Offline guard and source-isolation tests for 1.7B stronger clone pilot."""
from collections import Counter
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(n):
    spec=importlib.util.spec_from_file_location("test_q17_"+n,HERE/(n+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
g=load("english_qwen17_clone_probe_once")
p=load("english_qwen17_two_phone_once")

class SourceBound(unittest.TestCase):
    def test_exact_model_commit_and_train_reference_only(self):
        r=g.source_check()
        self.assertEqual(r["maximum_requests"],8)
        self.assertEqual(r["heldout_donor_calls"],0)
        self.assertEqual(r["new_tts_calls"],0)
        self.assertFalse(r["training_allowed"])
        self.assertEqual(g.TRAIN_DONORS,("af_bella","am_puck"))
        self.assertEqual(g.MODEL[1],"fd4b254389122332181a7c3db7f27e918eec64e3")
    def test_no_requested_text_can_be_used_as_label(self):
        simulated={
            name:{"source_group_id":"kokoro-v1_0-preset:"+name,
                  "joined_pcm_sha256":"a"*64,
                  "wav_sha256":"b"*64,
                  "reference_text":"Hello there. Good morning."}
            for name in g.TRAIN_DONORS}
        rows=g.planned(simulated)
        self.assertEqual(len(rows),8)
        self.assertEqual({r["role"] for r in rows},{"train"})
        self.assertEqual({r["requested_brand_class"] for r in rows},{1,2})
        self.assertTrue(all(r["observed_class"] is None and
                            r["training_allowed"] is False for r in rows))
        self.assertEqual(len({r["id"] for r in rows}),8)
    def test_no_public_run_replay_or_hidden_attempt(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":g.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"12345",
             "GITHUB_SHA":"f"*40}
        event={"before":g.BEFORE,"after":"f"*40,"ref":"refs/heads/main",
               "repository":{"full_name":g.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(g.guard(env,event,g.BEFORE)["parent"],g.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            g.guard(env,event,g.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["before"]="0"*40
        with self.assertRaises(ValueError):
            g.guard(env,event,g.BEFORE)
    def test_exact_two_native_unprompted_phone_model_ids(self):
        self.assertEqual(p.e.MODEL,g.MODEL)
        self.assertEqual(len(p.c.PHONE_MODELS),2)
        self.assertEqual(p.admit.M_XLSR.pattern.startswith("h?eɪmɛ"),True)
        self.assertEqual(p.admit.V_LV60.pattern.find("s"),p.admit.V_LV60.pattern.find("s"))
        self.assertFalse(g.source_check()["training_allowed"])

if __name__=="__main__":unittest.main()
