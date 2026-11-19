"""No numerical forward on PR: fixed source and dev-only candidate selection."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import io,json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("dev_only_v2",HERE/"english_canonical_dev_only_v2_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DevOnlyV2(unittest.TestCase):
    def test_new_model_never_reuses_locked_pcm_for_selection(self):
        actual=m.source_check()
        self.assertEqual(actual["max_real_updates"],384)
        self.assertEqual(actual["dev_selection_steps"],[96,192,384])
        self.assertEqual(actual["source_groups"],
            {"train":8,"development":4,"calibration":3,"locked-test":3})
        self.assertEqual(actual["dev_working_threshold"],0.8)
        self.assertFalse(actual["old_locked_kws_test_reuse"])
        self.assertFalse(actual["calibration_audio_opened"])
        self.assertFalse(actual["locked_audio_opened"])
        self.assertFalse(actual["training_allowed"])
    def test_candidate_order_and_dev_metrics_are_fixed(self):
        def candidate(update,miss,events):
            return {"update":update,"threshold":0.8,
                    "selection_key":[miss,events,miss/2,update],
                    "development":{"source_groups":4,
                        "classes":{"0":{"cases":8},
                                   "1":{"cases":8},"2":{"cases":10}}}}
        c=[candidate(96,.7,1),candidate(192,.5,2),candidate(384,.5,3)]
        self.assertEqual(m.choose(c)["update"],192)
        with self.assertRaisesRegex(ValueError,"EXACT_THREE_NEW_DEV_ONLY_CANDIDATES"):
            m.choose(c[:-1])
        c[0]["threshold"]=0.9
        with self.assertRaisesRegex(ValueError,"PREDECLARED_DEV_WORKING_POINT"):
            m.choose(c)
    def test_public_main_first_parent_and_attempt_one(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"44412345",
             "GITHUB_SHA":"a"*40}
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(m.guard(env,event,m.BEFORE)["before"],m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"EXACT_PUBLIC_MAIN_FIRST_ATTEMPT"):
            m.guard(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1";event["repository"]["private"]=True
        with self.assertRaisesRegex(ValueError,"EXACT_FIRST_MAIN_SUCCESSOR_WITHOUT_REPLAY"):
            m.guard(env,event,m.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaisesRegex(ValueError,"ORIGINAL_FROZEN_MAIN_FIRST_PARENT"):
            m.guard(env,event,"b"*40)
    def test_zero_forward_attestor_rejects_prior_any_model_step(self):
        self.assertEqual(m.PRIOR_RUN,44425553398)
        self.assertEqual(m.PRIOR_NUMERIC_JOB,135229803469)
        self.assertEqual(m.BEFORE,m.PRIOR_SHA)
        step_names=[
            m.PRIOR_FAILED_STEP,
            "Read six original archives under a fresh numeric job identity",
            "Verify original 63 Silver/Weak research rows and disjoint 8/4/3/3 roles without numerical inference",
            "Install exactly one original CPU frozen-ONNX plus fresh PyTorch training stack",
            "Fetch only the two original 0.5.1 frozen feature models, validate bytes before training",
            "Run real fresh-seed 384-update train/dev-only candidate selection, no calibration or locked PCM"]
        packet={"run":{"id":m.PRIOR_RUN,"event":"push",
                    "status":"completed","conclusion":"failure",
                    "run_attempt":1,"head_sha":m.PRIOR_SHA,
                    "head_branch":"main",
                    "repository":{"full_name":m.REPO,"private":False}},
                "jobs":{"jobs":[
                    {"name":"source-contract","conclusion":"success"},
                    {"name":"exact-six-artifact-audit","conclusion":"success"},
                    {"id":m.PRIOR_NUMERIC_JOB,"name":"dev-only-numeric-fit",
                     "conclusion":"failure","steps":[
                        {"name":name,"conclusion":"failure" if idx==0 else "skipped"}
                        for idx,name in enumerate(step_names)]}]},
                "artifacts":{"total_count":0,"artifacts":[]}}
        class Response:
            def __init__(self,p):self.payload=p
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def read(self,n):return json.dumps(self.payload).encode()
        def fake_urlopen(req,timeout=0):
            url=req.full_url
            if url.endswith("/jobs?per_page=100"):return Response(packet["jobs"])
            if url.endswith("/artifacts?per_page=100"):return Response(packet["artifacts"])
            if url.endswith("/runs/"+str(m.PRIOR_RUN)):return Response(packet["run"])
            raise AssertionError("unexpected attestor url")
        with patch.object(m,"guard",return_value={}),patch.object(m.urllib.request,"urlopen",side_effect=fake_urlopen):
            self.assertEqual(m.verify_prior_zero("offline-test-token")["original_numeric_updates"],0)
            packet["jobs"]["jobs"][-1]["steps"][-1]["conclusion"]="success"
            with self.assertRaisesRegex(ValueError,"NO_PRIOR_MODEL_FORWARDS"):
                m.verify_prior_zero("offline-test-token")
            packet["jobs"]["jobs"][-1]["steps"][-1]["conclusion"]="skipped"
            packet["artifacts"]={"total_count":1,"artifacts":[{"name":"fake-numeric"}]}
            with self.assertRaisesRegex(ValueError,"NO_PARTIAL_NEW_V2_MODEL"):
                m.verify_prior_zero("offline-test-token")

    def test_only_new_dev_head_and_not_main_release(self):
        self.assertEqual(m.MODEL_PARAMETERS,49283)
        self.assertEqual(m.SEED,20261119)
        self.assertEqual(m.UPDATES,(96,192,384))
        self.assertEqual(m.base.MAGIC,b"EHKWS01\x00")
        self.assertEqual(m.base.SHAPES["w1"],(32,1536))
if __name__=="__main__":
    unittest.main()
