"""Non-network, non-model, triple-interpreter V5 original-source observer guards."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import importlib.util
import json
from unittest import TestCase,main, mock
from tempfile import TemporaryDirectory

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v5_fresh24_five_test",HERE/"english_v5_mero_fresh24_five_native_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def fixture_rows():
    ids={i:f"p{30000+i}" for i in range(448,472)}
    rows=v.source_worker.planned(ids)
    for pos,row in enumerate(rows):
        row.update(pcm_sha256=f"{pos:064x}",nearphone_is_background=False)
    return rows

def manifest(rows):
    return {
        "schema":v.source_worker.SCHEMA,"run":v.original_identity(),
        "source_plan_sha256":v.source_worker.source_check()["plan_sha256"],
        "denominator":96,"completed":96,"new_tts_calls":96,
        "new_asr_calls":0,"new_phone_calls":0,
        "training_allowed":False,"human_gold":False,"rows":rows}

class V5Fresh24FiveNativeTests(TestCase):
    def test_source_pins_no_new_tts_and_five_reviews(self):
        c=v.source_check()
        self.assertEqual(c["source_rows"],96)
        self.assertEqual(c["review_rows"],64)
        self.assertEqual(c["observer_roster"],
                         ["qwen","whisper","parakeet","xlsr","lv60"])
        self.assertEqual(c["max_new_forwards_per_observer"],64)
        self.assertEqual(c["calibration_model_forwards"],0)
        self.assertEqual(c["locked_test_model_forwards"],0)
        self.assertEqual(c["new_tts_calls"],0)
        self.assertFalse(c["training_allowed"])
        self.assertEqual(v.original_identity()["attempt"],1)
        self.assertEqual(v.SOURCE["id"],14507208152)

    def test_only_train_development_rows_are_eligible_for_review(self):
        with TemporaryDirectory() as tmp:
            rows=fixture_rows()
            (Path(tmp)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"authenticated"}):
                reviewed=v.source_rows(tmp)
        self.assertEqual(len(reviewed),64)
        self.assertEqual(Counter(r["role"] for r in reviewed),
                         {"train":48,"development":16})
        self.assertEqual(Counter(r["requested_class"] for r in reviewed),
                         {2:32,0:16,None:16})
        self.assertNotIn("calibration",{r["role"] for r in reviewed})
        self.assertNotIn("locked-test",{r["role"] for r in reviewed})
        self.assertTrue(all(r["training_allowed"] is False for r in reviewed))

    def test_nearphone_may_not_be_relabelled_ordinary(self):
        with TemporaryDirectory() as tmp:
            rows=fixture_rows()
            rows[3]["requested_class"]=0
            (Path(tmp)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"fixture"}):
                with self.assertRaisesRegex(ValueError,"UNLABELED_NEARPHONE"):
                    v.source_rows(tmp)

    def test_duplicate_identity_or_calibration_role_leak_is_rejected(self):
        with TemporaryDirectory() as tmp:
            rows=fixture_rows()
            rows[0]["pcm_sha256"]=rows[1]["pcm_sha256"]
            (Path(tmp)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"fixture"}):
                with self.assertRaises(ValueError):
                    v.source_rows(tmp)
        with TemporaryDirectory() as tmp:
            rows=fixture_rows()
            rows[-1]["role"]="train"
            (Path(tmp)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"fixture"}):
                with self.assertRaises(ValueError):
                    v.source_rows(tmp)

    def test_numeric_never_runs_on_pr_or_retry(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"pull_request","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"1"}
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.BEFORE)
        env["GITHUB_EVENT_NAME"]="push"
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.BEFORE)

if __name__=="__main__":
    main()
