"""Frozen sealed-role V5 source QA tests; no model inference on PR."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import importlib.util
import json
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v5_sealed_observer_test",
    HERE/"english_v5_mero_fresh24_sealed_source_five_native_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def raw_rows():
    voices={i:f"p{30000+i}" for i in range(448,472)}
    rows=v.source_worker.planned(voices)
    for i,row in enumerate(rows):
        row.update(pcm_sha256=f"{i:064x}",nearphone_is_background=False)
    return rows

def manifest(rows):
    return {"schema":v.source_worker.SCHEMA,
            "run":v.original_identity(),
            "source_plan_sha256":v.source_worker.source_check()["plan_sha256"],
            "denominator":96,"completed":96,"new_tts_calls":96,
            "new_asr_calls":0,"new_phone_calls":0,
            "training_allowed":False,"human_gold":False,
            "rows":rows}

class SealedSourceLabelOnlyTests(TestCase):
    def test_exact_source_and_no_kws_or_train_dev_forward(self):
        p=v.source_check()
        self.assertEqual(p["source_rows"],96)
        self.assertEqual(p["review_rows"],32)
        self.assertEqual(p["max_new_native_forwards_per_observer"],32)
        self.assertEqual(p["train_or_dev_review_clips"],0)
        self.assertEqual(p["kws_model_forwards"],0)
        self.assertEqual(p["kws_locked_test_events"],0)
        self.assertEqual(p["calibration_label_review_clips"],16)
        self.assertEqual(p["locked_test_label_review_clips"],16)
        self.assertFalse(p["training_allowed"])
        self.assertFalse(p["product_promotion_allowed"])
        self.assertEqual(p["observer_roster"],[
            "qwen","whisper","parakeet","xlsr","lv60"])

    def test_only_frozen_cal_and_locked_sources_enter_blind_label_qc(self):
        with TemporaryDirectory() as d:
            rows=raw_rows()
            (Path(d)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"READBACK_OK"}):
                subset=v.source_rows(d)
        self.assertEqual(len(subset),32)
        self.assertEqual(Counter(x["role"] for x in subset),
                         {"calibration":16,"locked-test":16})
        self.assertEqual(Counter(x["requested_class"] for x in subset),
                         {2:16,0:8,None:8})
        self.assertEqual({x["speaker_index"] for x in subset},set(range(464,472)))
        self.assertEqual(len({x["source_group_id"] for x in subset}),8)
        self.assertTrue(all(x["training_allowed"] is False for x in subset))

    def test_nearphone_never_ordinary_and_no_role_reassignment(self):
        with TemporaryDirectory() as d:
            rows=raw_rows()
            rows[-1]["requested_class"]=0
            (Path(d)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={}):
                with self.assertRaisesRegex(ValueError,"ONLY_FROZEN_CAL_AND_LOCKED"):
                    v.source_rows(d)
        with TemporaryDirectory() as d:
            rows=raw_rows()
            rows[-1]["role"]="train"
            (Path(d)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={}):
                with self.assertRaises(ValueError):
                    v.source_rows(d)

    def test_duplicate_pcm_source_identity_rejected(self):
        with TemporaryDirectory() as d:
            rows=raw_rows()
            rows[0]["pcm_sha256"]=rows[1]["pcm_sha256"]
            (Path(d)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={}):
                with self.assertRaisesRegex(ValueError,"DUPLICATE_PCM"):
                    v.source_rows(d)

    def test_first_valid_main_parent_passes_guard_without_model_calls(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"456"}
        receipt=v.guard(env=env,event=event,parent=v.BEFORE)
        self.assertEqual(receipt["run_id"],"456")
        self.assertEqual(receipt["source_commit"],v.SOURCE_COMMIT)
        self.assertEqual(v.source_check()["kws_model_forwards"],0)

    def test_first_public_main_only_no_pr_or_retry(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"pull_request","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"7"}
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.BEFORE)
        env["GITHUB_EVENT_NAME"]="push"
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.BEFORE)

if __name__=="__main__":
    main()
