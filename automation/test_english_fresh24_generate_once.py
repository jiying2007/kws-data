"""Fresh24 source tests — no TTS, no downloaded models, no new numeric calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("fresh_generate",HERE/"english_fresh24_generate_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ProspectiveSource(unittest.TestCase):
    def test_24_official_pinned_unused_sources_and_budgets(self):
        report=m.source_check()
        self.assertEqual(report["source_groups"],24)
        self.assertEqual(report["planned_neutral_calls"],48)
        self.assertEqual(report["planned_clone_calls"],168)
        self.assertFalse(report["training_allowed"])
        plans=m.source_plan()
        self.assertEqual(len({x["asset_sha256"] for x in plans["voice_groups"]}),24)
        self.assertTrue(set(x["name"] for x in plans["voice_groups"]).isdisjoint(
            m.policy.OLD_PRESETS))
        self.assertEqual([x["requested_class"] for x in plans["cells"]],
            [2,2,1,1,0,None,None])
    def test_no_preexposed_source_and_no_generated_label(self):
        p=m.source_plan()
        requests=m.neutral_requests(p)
        self.assertEqual(len(requests),24)
        self.assertTrue(all(r["training_allowed"] is False and
            r["human_gold"] is False for r in requests))
        fake={r["id"]:{**r,"pcm_sha256":str(i+1).zfill(64),
              "wav_sha256":str(i+100).zfill(64)}
              for i,r in enumerate(requests)}
        rows=m.clone_requests(fake)
        self.assertEqual(len(rows),168)
        self.assertEqual(len({r["id"] for r in rows}),168)
        self.assertTrue(all(x["observed_class"] is None and
                    x["training_allowed"] is False for x in rows))
        self.assertEqual(len([x for x in rows if x["requested_class"] is None]),48)
        self.assertEqual([len([r for r in rows if r["role"]==role])
            for role in ("train","development","calibration","locked-test")],
            [84,28,28,28])
        self.assertEqual({r["requested_target"] for r in rows},
                         {"hey_mero","hey_vosen",None})
        self.assertTrue(all("Meero" not in r["generation_render_text"]
            for r in rows if r["requested_class"] in (1,2)))
    def test_missing_source_forbid_selective_recensus(self):
        p=m.source_plan()
        requests=m.neutral_requests(p)
        fake={r["id"]:{**r,"pcm_sha256":"a"*64,"wav_sha256":"b"*64}
              for r in requests[:-1]}
        with self.assertRaisesRegex(ValueError,"EXACT_NEW_REFERENCE_IDS"):
            m.clone_requests(fake)
    def test_source_guard_is_only_first_public_main_successor(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPOSITORY,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"442999000",
             "GITHUB_SHA":"a"*40}
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPOSITORY,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(m.guard(env,event,m.BEFORE)["before"],m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"PUBLIC_FIRST_MAIN_ONLY"):
            m.guard(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["repository"]["private"]=True
        with self.assertRaisesRegex(ValueError,"ONE_EXACT_SOURCE_SUCCESSOR"):
            m.guard(env,event,m.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaisesRegex(ValueError,"PINNED_FIRST_PARENT"):
            m.guard(env,event,"b"*40)
    def test_zero_forward_prior_recovery_is_provably_no_numeric_call(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            original={"run_id":str(m.PRIOR_ZERO_RUN),
                      "commit":m.PRIOR_SOURCE_COMMIT,
                      "before":m.PRIOR_SOURCE_BEFORE}
            started={"schema":m.REFERENCE_SCHEMA,"run":original,
                     "denominator":24,"max_calls":48,
                     "training_allowed":False,
                     "source_plan_sha256":m.source_check()["source_plan_sha256"]}
            failed={"schema":m.REFERENCE_SCHEMA,"run":original,
                    "completed_references":0,"native_forward_returns":0,
                    "remaining_possible_calls":48,
                    "failure":"ModuleNotFoundError:No module named 'num2words'",
                    "retry":False,"training_allowed":False}
            m.write(root/"STARTED.json",started)
            m.write(root/"FAILED.json",failed)
            self.assertEqual(m.verify_zero_forward_packet(root)["prior_new_audio_calls"],0)
            failed["native_forward_returns"]=1
            (root/"FAILED.json").write_text(json.dumps(failed),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"PRIOR_NUMERICAL_FORWARD_COUNT_NOT_ZERO"):
                m.verify_zero_forward_packet(root)
            failed["native_forward_returns"]=0
            failed["failure"]="UnrelatedRuntimeFailure"
            (root/"FAILED.json").write_text(json.dumps(failed),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"PRIOR_NUMERICAL_FORWARD_COUNT_NOT_ZERO"):
                m.verify_zero_forward_packet(root)
    def test_recovery_source_commit_remains_frozen(self):
        self.assertEqual(m.PRIOR_SOURCE_COMMIT,
                         "dfbb95f003b3845439ccbe5e5eb4cb80e58e83f8")
        self.assertEqual(m.PRIOR_SOURCE_BEFORE,
                         "9899bf3cd505abcc6c363efd4490cf84fcdfc89d")
        self.assertEqual(m.PRIOR_ARTIFACT_ID,14419837155)
        self.assertEqual(m.BEFORE,m.PRIOR_SOURCE_COMMIT)

    def test_new_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"candidate"
            root.mkdir()
            with self.assertRaises(ValueError):
                m.safe_root(root)


if __name__=="__main__":
    unittest.main()
