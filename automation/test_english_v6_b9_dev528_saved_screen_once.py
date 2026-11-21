"""B9 DEV528 pure-stdlib source-only tests for 299 saved native observations."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import json
from tempfile import TemporaryDirectory
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b9_saved_source_audit",HERE/"english_v6_b9_dev528_saved_screen_once.py")
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

CELLS=("mero-fast","mero-slow","ordinary-time",
       "hard-hey-mary","nearphone-meero-challenge")

def source_cell(group,cell,label,tier):
    return {"source_group_id":f"piper-new-{group}","role":"development",
            "cell_id":cell,
            "requested_class":(None if cell=="nearphone-meero-challenge"
                              else 0 if cell in ("ordinary-time","hard-hey-mary") else 2),
            "observed_class":label,"tier":tier,
            "human_gold":False,"training_allowed":False}

def cohort(group=0,ordinary=True,positive=True,hard=False):
    return [
       source_cell(group,"mero-fast",2 if positive else None,
                   "silver" if positive else "unknown"),
       source_cell(group,"mero-slow",None,"unknown"),
       source_cell(group,"ordinary-time",0 if ordinary else None,
                   "weak" if ordinary else "unknown"),
       source_cell(group,"hard-hey-mary",0 if hard else None,
                   "weak" if hard else "unknown"),
       source_cell(group,"nearphone-meero-challenge",None,"unknown")]

def fakes():
    return [{
      "id":f"merov6b9dev12-{528+i//5}-{CELLS[i%5]}",
      "wav_sha256":f"{i+101:064x}","pcm_sha256":f"{i+201:064x}",
      "samples":11000+i,"source_group_id":f"piper-new-{i//5}",
      "cell_id":CELLS[i%5],"role":"development",
      "requested_class":None if i%5==4 else (0 if i%5 in (2,3) else 2),
      "observed_class":None,"human_gold":False,"training_allowed":False
    } for i in range(60)]

class B9SavedReadbackTests(TestCase):
    def test_original_six_immutable_archives_and_one_explicit_native_failure(self):
        p=s.source_check()
        self.assertEqual(p["status"],
           "SAVED_PROSPECTIVE_B9_DEV528_299_OF_300_REAL_OBSERVATIONS_PINNED")
        self.assertEqual(p["original_observer_run"],44870958716)
        self.assertEqual(p["saved_source_rows"],60)
        self.assertEqual(p["review_rows"],60)
        self.assertEqual(p["observer_archive_count"],5)
        self.assertEqual(p["new_audio_or_model_forwards"],0)
        self.assertEqual(p["new_kws_model_forwards"],0)
        self.assertEqual(p["new_optimizer_updates"],0)
        self.assertEqual(p["new_locked_test_forwards"],0)
        self.assertFalse(p["training_allowed"])
        self.assertEqual(s.BEFORE,"fedc502a5b38445be1c6b143a94553a38850bcc1")
        self.assertEqual(s.SOURCE_RUN,44870282676)
        self.assertEqual(s.SOURCE["id"],14693632800)
        self.assertEqual(s.SOURCE["bytes"],1206914)
        self.assertEqual(s.SOURCE["digest"],
          "sha256:d606ee883a1abd38cc95a278f6b7b7064476c8b00add1163e27c247c41ab7032")
        self.assertEqual({k:v[0] for k,v in s.ARCHIVES.items()},
          {"qwen":14694033180,"whisper":14693594464,
           "parakeet":14693289906,"xlsr":14694245820,"lv60":14693522883})
        self.assertEqual(s.PARAKEET_FAILED_ID,"merov6b9dev12-531-mero-slow")
        self.assertEqual(s.PARAKEET_FAILED_ERROR,
          "ValueError:RAW_UNPROMPTED_COMPLETE_ASR_TEXT")
        self.assertEqual(s.observer.SOURCE_COMMIT,
          "2bd78ea27b2439e92c82ee5672712eba2e86405a")
        self.assertEqual(s.observer.source_worker.source_check()["new_speaker_indices"],
                         [528,539])

    def test_count_eligible_12_groups_five_cells_only(self):
        good=cohort(0)+cohort(1)
        counts,groups=s.count_eligible(good)
        self.assertEqual(counts,{"development":2})
        self.assertEqual(len(groups),2)
        bad,_=s.count_eligible(cohort(0)+cohort(1,ordinary=False))
        self.assertEqual(bad["development"],1)
        self.assertEqual(s.count_eligible(cohort(0,hard=True))[0],
                         {"development":1})
        self.assertEqual(s.count_eligible(cohort(0,positive=False))[0],
                         {"development":0})

    def test_no_pseudo_gold_nearphone_or_hard_mary(self):
        a=cohort()
        a[-1]["tier"]="silver";a[-1]["observed_class"]=0
        with self.assertRaisesRegex(ValueError,"NEARPHONE_CHALLENGE"):
            s.count_eligible(a)
        a=cohort();a[3]["observed_class"]=2;a[3]["tier"]="silver"
        with self.assertRaisesRegex(ValueError,"HARD_MARY"):
            s.count_eligible(a)
        a=cohort();a[0]["role"]="train"
        with self.assertRaisesRegex(ValueError,"ONLY_COMPLETE_FIVE_CELL"):
            s.count_eligible(a)
        with self.assertRaisesRegex(ValueError,"ONLY_COMPLETE_FIVE_CELL"):
            s.count_eligible(cohort()[:4])

    def test_missing_parakeet_saved_failure_does_not_impute_a_transcript(self):
        rows=fakes()
        with TemporaryDirectory() as td:
            root=Path(td)
            (root/"MODEL.json").write_text(json.dumps({
                "observer":s.observer.identity("parakeet"),
                "target_conditioned":False,"initialization_model_calls":0}))
            captures=[]
            for row in rows:
                d=root/"rows"/row["id"]
                d.mkdir(parents=True)
                base={
                    "id":row["id"],"provider":"parakeet",
                    "source_wav_sha256":row["wav_sha256"],
                    "source_pcm_sha256":row["pcm_sha256"],
                    "source_samples":row["samples"],
                    "target_conditioned":False,"training_allowed":False}
                (d/"STARTED.json").write_text(json.dumps(base))
                if row["id"]==s.PARAKEET_FAILED_ID:
                    receipt={**base,"status":"failed","error":s.PARAKEET_FAILED_ERROR}
                    (d/"FAILED.json").write_text(json.dumps(receipt))
                else:
                    receipt={**base,"status":"success","input_coverage_complete":True,
                             "raw_transcript":"hey mero"}
                    (d/"CAPTURE.json").write_text(json.dumps(receipt))
                captures.append(receipt)
            result={"schema":s.observer.SCHEMA,"provider":"parakeet",
                    "run":s.original_identity(),"source_artifact_id":s.SOURCE["id"],
                    "denominator":60,"attempted":60,"complete":59,
                    "missing_or_failed":1,"new_tts_calls":0,"target_conditioned":False,
                    "human_gold":False,"training_allowed":False,
                    "initialization_error":None,"status":"PARTIAL_WITH_FAILURE_RECEIPTS",
                    "rows":captures}
            (root/"RESULT.json").write_text(json.dumps(result))
            packet,proof=s.review_packet(root,"parakeet",rows)
            self.assertEqual(proof["missing_ids"],[s.PARAKEET_FAILED_ID])
            self.assertEqual(proof["complete"],59)
            self.assertEqual(set(packet),{r["id"] for r in rows})
            self.assertIsNone(packet[s.PARAKEET_FAILED_ID])
            self.assertEqual(sum(x is not None for x in packet.values()),59)
            failed=(root/"rows"/s.PARAKEET_FAILED_ID/"FAILED.json")
            invalid=json.loads(failed.read_text())
            invalid["error"]="fabricated_valid_transcript"
            failed.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError,"PARAKEET_ONLY_EXACT_ONE"):
                s.review_packet(root,"parakeet",rows)

    def test_explicit_failed_source_is_unknown_not_auto_rescored(self):
        src=(HERE/"english_v6_b9_dev528_saved_screen_once.py").read_text()
        self.assertIn('if values["parakeet"][sid] is None:',src)
        self.assertIn('"class":None',src)
        self.assertIn('"training_allowed":False',src)
        self.assertIn("ONE_AUTHENTIC_PAR-AKEET_CAPTURE_FAILED",src)
        self.assertIn('"native_successful_captures":299',src)
        self.assertNotIn("re_run_inference",src)

    def test_first_exact_main_only_no_pull_request_or_repeat(self):
        event={"before":s.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":s.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":s.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"101"}
        self.assertEqual(s.guard(env=env,event=event,parent=s.BEFORE)["new_model_forwards"],0)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):s.guard(env=env,event=event,parent=s.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1";env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):s.guard(env=env,event=event,parent=s.BEFORE)

if __name__=="__main__":
    main()
