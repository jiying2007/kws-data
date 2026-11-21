#!/usr/bin/env python3
"""B9 DEV528 60 original WAVs: source-only five-native screening contracts."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import importlib.util
import json
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location(
    "b9_dev528_five_native",HERE/"english_v6_b9_dev528_five_native_once.py")
v=importlib.util.module_from_spec(sp)
sp.loader.exec_module(v)

def generated():
    indices={i:f"p{30000+i}" for i in range(528,540)}
    rows=v.source_worker.planned(indices)
    for i,row in enumerate(rows):
        row.update(pcm_sha256=f"{i:064x}",nearphone_is_background=False)
    return rows

def manifest(rows):
    return {"schema":v.source_worker.SCHEMA,"run":v.original_identity(),
            "source_plan_sha256":v.source_worker.source_check()["plan_sha256"],
            "denominator":60,"completed":60,"new_tts_calls":60,
            "new_asr_calls":0,"new_phone_calls":0,
            "training_allowed":False,"human_gold":False,"rows":rows}

class B9BlindFiveNativeTests(TestCase):
    def test_actual_byte_bound_source_and_no_quality_inference(self):
        c=v.source_check()
        self.assertEqual(c["status"],"ORIGINAL_SOURCE_PINNED")
        self.assertEqual(c["source_run"],44870282676)
        self.assertEqual(c["source_artifact_id"],14693632800)
        self.assertEqual(c["source_rows"],60)
        self.assertEqual(c["review_rows"],60)
        self.assertEqual(c["max_new_native_forwards_per_observer"],60)
        self.assertEqual(c["observer_roster"],[
            "qwen","whisper","parakeet","xlsr","lv60"])
        self.assertEqual(c["locked_test_label_review_clips"],0)
        self.assertEqual(c["train_dev_calibration_review_clips"],60)
        self.assertEqual(c["kws_model_forwards"],0)
        self.assertEqual(c["kws_locked_test_events"],0)
        self.assertEqual(c["new_optimizer_updates"],0)
        self.assertFalse(c["training_allowed"])
        self.assertEqual(v.BEFORE,"2bd78ea27b2439e92c82ee5672712eba2e86405a")
        self.assertEqual(v.SOURCE_COMMIT,"2bd78ea27b2439e92c82ee5672712eba2e86405a")
        self.assertEqual(v.SOURCE_PARENT,"846c745ebcee58a1beb27300ddcd7ec8e2178916")
        self.assertEqual(v.SOURCE["digest"],
           "sha256:d606ee883a1abd38cc95a278f6b7b7064476c8b00add1163e27c247c41ab7032")
        self.assertEqual(v.SOURCE["size"],1206914)
        self.assertEqual(v.source_worker.source_check()["new_speaker_indices"],[528,539])
        self.assertTrue(v.SOURCE["name"].startswith("english-v6-b9-fresh-dev528-original-"))

    def test_all_60_source_roles_unobserved_no_prompt_gold(self):
        with TemporaryDirectory() as td:
            rows=generated()
            (Path(td)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"AUTH"}):
                out=v.source_rows(td)
        self.assertEqual(len(out),60)
        self.assertEqual({x["speaker_index"] for x in out},set(range(528,540)))
        self.assertEqual(len({x["source_group_id"] for x in out}),12)
        self.assertEqual(Counter(r["role"] for r in out),{"development":60})
        self.assertEqual(Counter(r["requested_class"] for r in out),
                         {2:24,0:24,None:12})
        self.assertEqual(Counter(r["cell_id"] for r in out),
                         {"mero-fast":12,"mero-slow":12,"ordinary-time":12,
                          "hard-hey-mary":12,"nearphone-meero-challenge":12})
        self.assertTrue(all(r["observed_class"] is None for r in out))
        self.assertTrue(all(r["training_allowed"] is False for r in out))
        self.assertTrue(all(r["human_gold"] is False for r in out))

    def test_unknown_nearphone_cannot_silently_become_negative(self):
        with TemporaryDirectory() as td:
            rows=generated()
            rows[4]["requested_class"]=0
            (Path(td)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={}):
                with self.assertRaisesRegex(ValueError,"ONLY_TWELVE_NEW_PROSPECTIVE"):
                    v.source_rows(td)

    def test_hard_ordinary_candidate_not_auto_relabelled_positive(self):
        with TemporaryDirectory() as td:
            rows=generated()
            rows[3]["requested_class"]=2
            (Path(td)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={}):
                with self.assertRaisesRegex(ValueError,"ONLY_TWELVE_NEW_PROSPECTIVE"):
                    v.source_rows(td)

    def test_cross_role_and_duplicate_fail_closed(self):
        with TemporaryDirectory() as td:
            rows=generated();rows[-1]["role"]="train"
            (Path(td)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={}):
                with self.assertRaises(ValueError):v.source_rows(td)
        with TemporaryDirectory() as td:
            rows=generated();rows[0]["pcm_sha256"]=rows[1]["pcm_sha256"]
            (Path(td)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={}):
                with self.assertRaisesRegex(ValueError,"NO_DUPLICATED_ORIGINAL_PCM"):
                    v.source_rows(td)

    def test_protected_first_main_not_pr_or_retry(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"123"}
        self.assertEqual(v.guard(env=env,event=event,parent=v.BEFORE)["source_run"],
                         v.SOURCE_RUN)
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):v.guard(env=env,event=event,parent=v.BEFORE)
        env["GITHUB_EVENT_NAME"]="push";env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):v.guard(env=env,event=event,parent=v.BEFORE)

if __name__=="__main__":
    main()
