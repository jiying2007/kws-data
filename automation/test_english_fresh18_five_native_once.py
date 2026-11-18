"""Offline fail-closed new18 five-native / immutable prior evidence tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("native18_test_"+name,HERE/(name+".py"))
    x=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x)
    return x
worker=load("english_fresh18_five_native_once")
screen=load("english_fresh18_combine_once")


def fake_rows():
    rows=worker.source_worker.rows()
    for n,row in enumerate(rows):
        row.update(samples=16000,positive_duration_pass=True,
            wav_sha256=f"{n+1:064x}",pcm_sha256=f"{n+130:064x}")
    return rows


class Fresh18CumulativeAdmission(unittest.TestCase):
    def test_actual_source_and_unmodified_frozen_phone_rule_pins(self):
        data=screen.source_check()
        self.assertEqual(data["source_rows"],126)
        self.assertEqual(data["source_artifact_id"],14429578824)
        self.assertEqual(data["original_policy_git_blob_sha1"],
                         "0a688325d0d36e83ef6623ac678511df24a92374")
        self.assertEqual(data["new18_plan_git_blob_sha1"],
                         "c2ee2f7bf2fbdeb8821920d6cb5e6890d0df1f76")
        self.assertEqual(worker.SOURCE["size"],3359173)
        self.assertEqual(worker.SOURCE["digest"],"sha256:"+
                         "34cf7879f974f62c9f6046a72a964002f30bf59d2186a423ac382a76ff69d0e9")
        self.assertFalse(data["training_allowed"])
        self.assertEqual(screen.FROZEN_PRIOR_GROUPS,
                         {"train":4,"development":1,"calibration":3,"locked-test":0})
        self.assertEqual(worker.source_worker.plan()["groups_by_role"],
                         {"train":10,"development":4,"locked-test":4})

    def test_exact_first_public_main_only_no_replay(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":worker.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"44399999",
             "GITHUB_SHA":"a"*40}
        event={"before":worker.BEFORE,"after":"a"*40,
               "ref":"refs/heads/main",
               "repository":{"full_name":worker.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(worker.guard(env,event,worker.BEFORE)["source_run"],44310115097)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_ONLY"):
            worker.guard(env,event,worker.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["repository"]["private"]=True
        with self.assertRaisesRegex(ValueError,"ONE_EXACT_MAIN_SUCCESSOR_REVIEW"):
            worker.guard(env,event,worker.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaisesRegex(ValueError,"EXACT_FIRST_MAIN_PARENT"):
            worker.guard(env,event,"b"*40)

    def test_exact_original_168_stop_is_not_trainable(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            original={
                "schema":"english-fresh24-prospective-blind-source-screen-v1",
                "status":"STOP_CANONICAL_6_2_2_2_SOURCE_COVERAGE",
                "source_run":44289263966,
                "source_artifact_id":14422271973,
                "all_five_native_observers_complete":True,
                "training_snapshot_allowed":False,"new_optimizer_updates":0,
                "machine_review":{
                    "qualified_groups_by_role":screen.FROZEN_PRIOR_GROUPS,
                    "total_qualified_synthetic_source_groups":8,
                    "denominator":168,"training_snapshot_allowed":False}}
            screen.c.record(root/"SCREEN.json",original)
            self.assertEqual(screen.readback_prior(root),screen.FROZEN_PRIOR_GROUPS)
            original["training_snapshot_allowed"]=True
            (root/"SCREEN.json").write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,"PREVIOUS_ORIGINAL_IMMUTABLE"):
                screen.readback_prior(root)

    def test_no_new_observations_never_invent_source_coverage(self):
        rows=fake_rows()
        text={r["id"]:{} for r in rows}
        phones={r["id"]:{} for r in rows}
        report=screen.evaluate(rows,text,phones,screen.FROZEN_PRIOR_GROUPS)
        self.assertEqual(report["status"],"STOP_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE")
        self.assertEqual(report["new_qualified_groups_by_role"],
                         {"train":0,"development":0,"locked-test":0})
        self.assertEqual(report["cumulative_qualified_groups_by_role"],
                         {"train":4,"development":1,"calibration":3,"locked-test":0})
        self.assertFalse(report["training_snapshot_allowed"])
        self.assertEqual(report["tier_counts"],{"unknown":126})

    def test_synthetic_mock_coverage_proves_only_source_logic_not_weights(self):
        rows=fake_rows()
        text={r["id"]:{} for r in rows}
        phones={r["id"]:{} for r in rows}
        def mocked(record,*args):
            if record["requested_class"] is None:
                return {"tier":"unknown","class":None,
                        "reason":"FIXTURE_DIAGNOSTIC_NOT_ACOUSTIC_TRUTH",
                        "margins":{},"training_allowed":False}
            return {"tier":"silver","class":record["requested_class"],
                    "reason":"TEST_ONLY_NO_NEW_MODEL_FORWARD",
                    "margins":{},"training_allowed":False}
        with patch.object(screen.policy,"decide",mocked):
            result=screen.evaluate(rows,text,phones,screen.FROZEN_PRIOR_GROUPS)
        self.assertEqual(result["status"],"PASS_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE")
        self.assertEqual(result["new_qualified_groups_by_role"],
                         {"train":10,"development":4,"locked-test":4})
        self.assertEqual(result["cumulative_qualified_groups_by_role"],
                         {"train":14,"development":5,"calibration":3,"locked-test":4})
        self.assertFalse(result["training_snapshot_allowed"])
        self.assertFalse(result["new_model_weights"])
        self.assertEqual(result["new_optimizer_updates"],0)
        self.assertTrue(all(row["training_allowed"] is False for row in result["rows"]))

if __name__=="__main__":
    unittest.main()
