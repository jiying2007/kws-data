"""Source-only tests for exact real prospective-dev8 five-native QA; zero inference."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import importlib.util
import json
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v6_b8_dev520_five_test",HERE/"english_v6_b8_dev520_five_native_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def generated():
    ids={i:f"p{30000+i}" for i in range(520,528)}
    rows=v.source_worker.planned(ids)
    for i,row in enumerate(rows):
        row.update(pcm_sha256=f"{i:064x}",nearphone_is_background=False)
    return rows

def manifest(rows):
    return {"schema":v.source_worker.SCHEMA,"run":v.original_identity(),
            "source_plan_sha256":v.source_worker.source_check()["plan_sha256"],
            "denominator":32,"completed":32,"new_tts_calls":32,
            "new_asr_calls":0,"new_phone_calls":0,
            "training_allowed":False,"human_gold":False,"rows":rows}

class ProspectiveDev8NativeReviewTests(TestCase):
    def test_exact_original_source_identity_and_no_kws_calls(self):
        c=v.source_check()
        self.assertEqual(c["status"],"ORIGINAL_SOURCE_PINNED")
        self.assertEqual(c["source_run"],44748801347)
        self.assertEqual(c["source_artifact_id"],14634696242)
        self.assertEqual(c["source_rows"],32)
        self.assertEqual(c["review_rows"],32)
        self.assertEqual(c["max_new_native_forwards_per_observer"],32)
        self.assertEqual(c["observer_roster"],[
            "qwen","whisper","parakeet","xlsr","lv60"])
        self.assertEqual(c["locked_test_label_review_clips"],0)
        self.assertEqual(c["train_dev_calibration_review_clips"],32)
        self.assertEqual(c["kws_model_forwards"],0)
        self.assertEqual(c["kws_locked_test_events"],0)
        self.assertEqual(c["new_optimizer_updates"],0)
        self.assertFalse(c["training_allowed"])
        self.assertEqual(v.BEFORE,"f606f35a4b3232cbfcec3bd2a14e8939551169fd")
        self.assertEqual(v.SOURCE_COMMIT,"f606f35a4b3232cbfcec3bd2a14e8939551169fd")
        self.assertEqual(v.SOURCE_PARENT,"453b26bdd4ab2bdc96b27fe0dfb9ac10bdd38711")
        self.assertEqual(v.SOURCE["digest"],
                         "sha256:1045f91d272e8f199d73dee6b39067bdbb9238dc3d4f05f74e95af54aa0c4700")
        self.assertEqual(v.SOURCE["size"],628555)
        self.assertEqual(v.source_worker.source_check()["new_speaker_indices"],[520,527])


    def test_only_eight_new_locked_groups_with_original_unlabeled_contrast(self):
        with TemporaryDirectory() as tmp:
            rows=generated()
            (Path(tmp)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"AUTH"}):
                out=v.source_rows(tmp)
        self.assertEqual(len(out),32)
        self.assertEqual({x["speaker_index"] for x in out},set(range(520,528)))
        self.assertEqual(len({x["source_group_id"] for x in out}),8)
        self.assertEqual(Counter(r["role"] for r in out),{"development":32})
        self.assertEqual(Counter(r["requested_class"] for r in out),{2:16,0:8,None:8})
        self.assertTrue(all(r["observed_class"] is None for r in out))
        self.assertTrue(all(r["training_allowed"] is False for r in out))

    def test_no_unlabeled_challenge_background_or_role_reassignment(self):
        with TemporaryDirectory() as tmp:
            rows=generated()
            rows[3]["requested_class"]=0
            (Path(tmp)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"AUTH"}):
                with self.assertRaisesRegex(ValueError,"ONLY_EIGHT_NEW_PROSPECTIVE_DEV"):
                    v.source_rows(tmp)
        with TemporaryDirectory() as tmp:
            rows=generated()
            rows[-1]["role"]="train"
            (Path(tmp)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={"status":"AUTH"}):
                with self.assertRaises(ValueError):
                    v.source_rows(tmp)

    def test_duplicate_original_pcm_fails(self):
        with TemporaryDirectory() as tmp:
            rows=generated()
            rows[0]["pcm_sha256"]=rows[1]["pcm_sha256"]
            (Path(tmp)/"MANIFEST.json").write_text(json.dumps(manifest(rows)))
            with mock.patch.object(v.source_worker,"readback",return_value={}):
                with self.assertRaisesRegex(ValueError,"NO_DUPLICATED_ORIGINAL_PCM"):
                    v.source_rows(tmp)

    def test_valid_first_public_main_allowed_but_no_pr_or_retry(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"999"}
        receipt=v.guard(env=env,event=event,parent=v.BEFORE)
        self.assertEqual(receipt["source_commit"],v.SOURCE_COMMIT)
        self.assertEqual(receipt["source_run"],v.SOURCE_RUN)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):v.guard(env=env,event=event,parent=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):v.guard(env=env,event=event,parent=v.BEFORE)

if __name__=="__main__":
    main()
