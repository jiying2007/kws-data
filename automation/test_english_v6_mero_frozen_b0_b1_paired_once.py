"""V6 source-only B0/B1 frozen model and new-development label tests."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
 "v6_b0_b1_test",HERE/"english_v6_mero_frozen_b0_b1_paired_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fake_rows():
    original=[];review=[];qualified=[]
    for n in range(488,496):
        group="piper-libritts-p"+str(n+30000)
        if n<492:qualified.append(group)
        for cell in ("mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge"):
            sid="merov6dev8-"+str(n)+"-"+cell
            intended=(2 if cell.startswith("mero-") else
                      0 if cell=="ordinary-time" else None)
            row={"id":sid,"speaker_index":n,"role":"development","cell_id":cell,
                 "source_group_id":group,"requested_class":intended,
                 "pcm_sha256":hashlib.sha256(sid.encode()).hexdigest(),
                 "wav_sha256":hashlib.sha256((sid+"wav").encode()).hexdigest(),
                 "samples":9000,"training_allowed":False,"human_gold":False}
            original.append(row)
            positive=n<492 and cell=="mero-fast" or n in (488,489) and cell=="mero-slow"
            tier="weak" if positive and n==491 else (
                 "silver" if positive or cell=="ordinary-time" else "unknown")
            review.append({**row,"tier":tier,
                 "observed_class":2 if positive else
                      0 if cell=="ordinary-time" else None})
    return original,review,qualified

def run_select(change=None):
    orig,review,qualified=fake_rows()
    if change:change(orig,review)
    receipt={"schema":m.saved.SCHEMA,
      "status":"PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY",
      "source_run":m.SOURCE["run"],"original_observer_run":44556480241,
      "source_group_counts":{"development":4},
      "tier_counts":{"silver":13,"weak":1,"unknown":18},
      "observed_label_counts":{"0":8,"2":6},
      "nearphone_as_ordinary_negative":False,
      "machine_only_no_human_gold":True,
      "new_audio_or_model_forwards":0,"new_optimizer_updates":0,
      "new_kws_model_forwards":0,"training_snapshot_allowed":False,
      "locked_kws_test_opened":False,"product_promotion":False,
      "source_group_identity_members":qualified,"rows":review}
    with TemporaryDirectory() as tmp:
        (Path(tmp)/"SCREEN.json").write_text(json.dumps(receipt))
        with mock.patch.object(m.saved,"source_rows",return_value=orig):
            return m.select(tmp,tmp)

class V6FrozenContracts(TestCase):
    def test_exact_weights_and_source_are_pinned(self):
        p=m.source_check()
        self.assertEqual(p["models"],{"B0":14514228406,"B1":14539983171})
        self.assertEqual(p["new_reviewed_positive"],6)
        self.assertEqual(p["new_reviewed_ordinary"],8)
        self.assertEqual(p["new_optimizer_steps"],0)
        self.assertFalse(p["product_promotion_allowed"])
    def test_new_qualified_6_pos_8_ordinary(self):
        rows,proof=run_select()
        self.assertEqual(len(rows),14)
        self.assertEqual(sum(x["mono_class"] for x in rows),6)
        self.assertEqual(proof["qualified_positive_sources"],
                         ["piper-libritts-p30488","piper-libritts-p30489",
                          "piper-libritts-p30490","piper-libritts-p30491"])
    def test_unknown_nearphone_not_negative(self):
        def bad(orig,review):review[3]["tier"]="silver";review[3]["observed_class"]=0
        with self.assertRaises(ValueError):run_select(bad)
    def test_never_train_or_rename_old_audio(self):
        def bad(orig,review):review[0]["role"]="train"
        with self.assertRaises(ValueError):run_select(bad)
    def test_first_main_only(self):
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"100"}
        self.assertTrue(m.guard(env=env,event=event,parent=m.BEFORE)["no_replay"])
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):m.guard(env=env,event=event,parent=m.BEFORE)
if __name__=="__main__":main()
