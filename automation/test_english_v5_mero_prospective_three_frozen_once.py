"""Pure stdlib contracts: zero audio generation, inference or training on PR."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "mero_prospective_three_test",HERE/"english_v5_mero_prospective_three_frozen_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def sources():
    original=[];reviewed=[]
    for index in range(480,488):
        for cell in ("mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge"):
            id=f"newdev-{index}-{cell}"
            klass=(2 if cell in ("mero-fast","mero-slow")
                   else 0 if cell=="ordinary-time" else None)
            orig={"id":id,"speaker_index":index,"role":"development",
               "cell_id":cell,"requested_class":klass,
               "source_group_id":f"piper-libritts-p{index+30000}",
               "pcm_sha256":hashlib.sha256(id.encode()).hexdigest(),
               "wav_sha256":hashlib.sha256((id+"wav").encode()).hexdigest(),
               "samples":9000,"training_allowed":False,"human_gold":False}
            original.append(orig)
            valid=(cell=="ordinary-time" or
                   (index in (480,481) and cell=="mero-fast"))
            reviewed.append({**orig,"tier":"silver" if valid else "unknown",
                "observed_class":klass if valid else None})
    return original,reviewed

def packet(rows):
    return {"schema":m.saved.SCHEMA,
      "status":"PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY",
      "source_run":m.SOURCE["run"],"original_observer_run":44512239544,
      "source_group_counts":{"development":2},
      "tier_counts":{"silver":10,"unknown":22},
      "observed_label_counts":{"0":8,"2":2},
      "nearphone_as_ordinary_negative":False,
      "machine_only_no_human_gold":True,
      "new_audio_or_model_forwards":0,"new_optimizer_updates":0,
      "new_kws_model_forwards":0,"training_snapshot_allowed":False,
      "locked_kws_test_opened":False,"product_promotion":False,
      "source_group_identity_members":["piper-libritts-p30480","piper-libritts-p30481"],
      "rows":rows}

class FrozenProspectiveContracts(TestCase):
    def check(self,change=None):
        old,review=sources()
        if change:change(old,review)
        with TemporaryDirectory() as tmp:
            Path(tmp,"SCREEN.json").write_text(json.dumps(packet(review)))
            with mock.patch.object(m.saved,"source_rows",return_value=old):
                return m.select(Path(tmp),Path(tmp))
    def test_exact_frozen_candidates_without_model_calls(self):
        p=m.source_check()
        self.assertEqual(p["new_dev_positive"],2)
        self.assertEqual(p["new_dev_ordinary"],8)
        self.assertEqual(p["fixed_phone_threshold"],0.25)
        self.assertEqual(p["optimizer_steps"],0)
        self.assertEqual(p["new_asr_tts"],0)
        self.assertFalse(p["product_promotion_allowed"])
        self.assertFalse(p["locked_audio_access"])
        self.assertEqual(set(p["models"]),{"fsmn43k256","fsmn93k448"})
    def test_qualified_source_disjoint_2_positive_8_ordinary(self):
        rows,proof=self.check()
        self.assertEqual(len(rows),10)
        self.assertEqual(sum(r["mono_class"]==1 for r in rows),2)
        self.assertEqual(sum(r["mono_class"]==0 for r in rows),8)
        self.assertEqual(len({r["source_group_id"] for r in rows}),8)
        self.assertTrue(proof["no_locked_or_calibration"])
    def test_no_nearphone_negative_promoted(self):
        def contaminate(old,review):
            review[3]["tier"]="silver";review[3]["observed_class"]=0
        with self.assertRaises(ValueError):self.check(contaminate)
    def test_no_locked_role_reassignment(self):
        def contaminate(old,review):review[1]["role"]="locked-test"
        with self.assertRaises(ValueError):self.check(contaminate)
    def test_original_report_printed_onnx_sha_not_pt_sha(self):
        # The historical 256-update main log prints RESULT.new_model_weights
        # ["mero-fsmn-ctc-expanded256.onnx"]["sha256"], not PT.
        pin=m.MODELS["fsmn43k256"]
        self.assertEqual(pin["onnx"],
            "10bf30da57291f78f56db196db1137309f2c3b91dfe67bf0aa45ebb8cec05b1c")
        self.assertNotIn("pt",pin)
        self.assertEqual(pin["id"],14514228406)
        self.assertEqual(pin["sha"],
            "91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702")

    def test_no_duplicate_or_missing_source(self):
        def contaminate(old,review):review[-1]["id"]=review[0]["id"]
        with self.assertRaises(ValueError):self.check(contaminate)
    def test_no_first_main_retry_or_pr_inference(self):
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
          "repository":{"full_name":m.REPO,"private":False},
          "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
            "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
            "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
            "GITHUB_RUN_ID":"100"}
        self.assertTrue(m.guard(env=env,event=event,parent=m.BEFORE)["no_replay"])
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):m.guard(env=env,event=event,parent=m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):m.guard(env=env,event=event,parent=m.BEFORE)

if __name__=="__main__":main()
