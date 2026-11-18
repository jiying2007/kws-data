"""No-model, no-network prospective 24-Piper source reviewer regression."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("piper24_test_"+name,HERE/(name+".py"))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod
worker=load("english_piper_fresh24_five_native_once")
screen=load("english_piper_fresh24_combine_once")

def fake_rows():
    voices={i:f"p{9000+i}" for i in range(400,424)}
    rows=worker.source_worker.planned(voices)
    for i,row in enumerate(rows):
        row.update(samples=16000,positive_duration_pass=True,
                   wav_sha256=f"{i+1:064x}",pcm_sha256=f"{i+500:064x}")
    return rows

class NewPiperSource(unittest.TestCase):
    def test_source_and_prior_artifacts_are_exactly_pinned(self):
        result=screen.source_check()
        self.assertEqual(result["source_rows"],168)
        self.assertEqual(result["source_artifact_id"],14434542792)
        self.assertEqual(result["piper_new_plan_git_blob_sha1"],
                         "d477ecec2a07910fc1a91063752aff1acc9006ef")
        self.assertEqual(result["frozen_policy_git_blob_sha1"],
                         "0a688325d0d36e83ef6623ac678511df24a92374")
        self.assertEqual(screen.PRIOR["artifact_id"],14429884666)
        self.assertEqual(worker.SOURCE["digest"],
            "sha256:d1f75716fefd954d59c3289c37e7e6670965ee98917436676896312e4b781aa1")
        self.assertFalse(result["training_allowed"])
    def test_only_main_first_attempt_after_actual_piper_generation(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":worker.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"444001001",
             "GITHUB_SHA":"a"*40}
        event={"before":worker.BEFORE,"after":"a"*40,
               "ref":"refs/heads/main",
               "repository":{"full_name":worker.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(worker.guard(env,event,worker.BEFORE)["source_run"],
                         44321329022)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):worker.guard(env,event,worker.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["repository"]["private"]=True
        with self.assertRaises(ValueError):worker.guard(env,event,worker.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaises(ValueError):worker.guard(env,event,"b"*40)
    def test_original_cumulative_stop_retained_without_reclassification(self):
        data={"schema":"english-fresh18-joined-prospective-blind-source-screen-v1",
              "status":"STOP_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE",
              "original_source_run":44310115097,"source_artifact_id":14429578824,
              "all_five_native_observers_complete":True,
              "cumulative_machine_source_qualification_passed":False,
              "training_snapshot_allowed":False,"new_optimizer_updates":0,
              "machine_review":{"cumulative_qualified_groups_by_role":
                     screen.FROZEN_PRIOR_GROUPS,"denominator":126,
                     "new_model_weights":False}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            screen.c.record(root/"SCREEN.json",data)
            self.assertEqual(screen.readback_prior(root),screen.FROZEN_PRIOR_GROUPS)
            data["training_snapshot_allowed"]=True
            (root/"SCREEN.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):screen.readback_prior(root)
    def test_missing_every_blind_model_never_invents_new_silver(self):
        rows=fake_rows()
        texts={r["id"]:{} for r in rows}
        phones={r["id"]:{} for r in rows}
        result=screen.evaluate(rows,texts,phones,screen.FROZEN_PRIOR_GROUPS)
        self.assertEqual(result["tier_counts"],{"unknown":168})
        self.assertEqual(result["new_qualified_groups_by_role"],
             {"train":0,"development":0,"locked-test":0})
        self.assertEqual(result["cumulative_qualified_groups_by_role"],
             {"train":4,"development":1,"calibration":3,"locked-test":1})
        self.assertFalse(result["training_snapshot_allowed"])
    def test_mock_coverage_is_never_KWS_training_or_human_gold(self):
        rows=fake_rows()
        t={r["id"]:{} for r in rows};p={r["id"]:{} for r in rows}
        def diagnostic_only(row,*args):
            if row["requested_class"] is None:
                return {"tier":"unknown","class":None,
                        "reason":"TEST_CONTRAST_ONLY","margins":{},
                        "training_allowed":False}
            return {"tier":"silver","class":row["requested_class"],
                    "reason":"SYNTHETIC_TEST_NOT_AN_OBSERVATION",
                    "margins":{},"training_allowed":False}
        with patch.object(screen.policy,"decide",diagnostic_only):
            result=screen.evaluate(rows,t,p,screen.FROZEN_PRIOR_GROUPS)
        self.assertEqual(result["cumulative_qualified_groups_by_role"],
                         {"train":16,"development":7,
                          "calibration":3,"locked-test":7})
        self.assertTrue(result["cumulative_machine_source_qualification_passed"])
        self.assertFalse(result["training_snapshot_allowed"])
        self.assertEqual(result["new_optimizer_updates"],0)
        self.assertFalse(result["new_model_weights"])
        self.assertTrue(all(not x["training_allowed"] for x in result["rows"]))

if __name__=="__main__":
    unittest.main()
