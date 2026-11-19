"""All source-only V5 expanded-FSMN contract tests; no Torch or model calls."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import importlib.util
import json
import tempfile
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v5_fresh_train_test",HERE/"english_v5_mero_fresh24_plus70_fit_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def rows_and_screen():
    voices={i:f"p{30000+i}" for i in range(448,472)}
    originals=[r for r in v.review.observer.source_worker.planned(voices)
               if r["role"] in ("train","development")]
    for i,r in enumerate(originals):
        r.update(wav_sha256=f"{i+1000:064x}",
                 pcm_sha256=f"{i+2000:064x}",
                 samples=13000,nearphone_is_background=False)
    selected=[];naccepted=0
    for r in originals:
        idx=r["speaker_index"]
        # 11/12 train + 2/4 dev: index 448..458 and 460..461.
        qualified=idx in set(range(448,459))|{460,461}
        eligible=(r["cell_id"]=="ordinary-time" or
                  (qualified and r["cell_id"]=="mero-fast") or
                  (idx in (448,449,450) and r["cell_id"]=="mero-slow"))
        if eligible:
            tier="weak" if naccepted<7 else "silver"
            naccepted+=1
        else:
            tier="unknown"
        selected.append({
             "id":r["id"],"source_group_id":r["source_group_id"],
             "role":r["role"],"speaker_index":r["speaker_index"],
             "cell_id":r["cell_id"],"requested_class":r["requested_class"],
             "observed_class":r["requested_class"] if eligible else None,
             "tier":tier,"pcm_sha256":r["pcm_sha256"],
             "wav_sha256":r["wav_sha256"],"training_allowed":False,
             "human_gold":False})
    assert naccepted==32
    screen={"schema":v.review.SCHEMA,
            "status":"PASS_NEW_TRAIN_DEV_MACHINE_SCREEN_ONLY",
            "source_run":v.FRESH_SOURCE["run"],
            "original_observer_run":v.review.ORIGINAL_RUN,
            "source_group_counts":v.FRESH_QUALIFIED,
            "tier_counts":{"silver":25,"weak":7,"unknown":32},
            "observed_label_counts":{"0":16,"2":16},
            "full_6_2_2_2_gate_passed":False,
            "training_snapshot_allowed":False,
            "locked_kws_test_opened":False,
            "rows":selected}
    return originals,screen

class ExpandedTrainTests(TestCase):
    def test_contract_against_saved_real_model_and_machine_source_gate(self):
        r=v.source_check()
        self.assertEqual(r["status"],"NEW_TRAIN_ONLY_MODEL_NOT_EXECUTED")
        self.assertEqual(r["model_parameters"],43876)
        self.assertEqual(r["max_real_optimizer_updates"],256)
        self.assertEqual(r["old_audited_train_pcm"],54)
        self.assertEqual(r["new_source_qualified_train_groups"],11)
        self.assertEqual(r["new_source_qualified_dev_groups"],2)
        self.assertEqual(r["new_observer_model_calls"],0)
        self.assertEqual(r["new_tts_calls"],0)
        self.assertFalse(r["calibration_or_locked_kws_audio_opened"])
        self.assertFalse(r["product_promotion_allowed"])
        self.assertEqual(v.PARENT["id"],14505594074)

    def test_new_research_binding_same_source_ordinary_and_canonical(self):
        originals,screen=rows_and_screen()
        with tempfile.TemporaryDirectory() as d:
            Path(d,"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(v.review.observer,"source_rows",
                                   return_value=originals):
                selected,report=v.screen_new(d,d)
        self.assertEqual(report["qualified_groups"],{
            "train":11,"development":2})
        self.assertEqual(report["selected_by_role"]["train"],
                         sum(r["role"]=="train" for r in selected))
        self.assertEqual(report["selected_by_role"]["development"],
                         sum(r["role"]=="development" for r in selected))
        self.assertEqual(Counter(r["mono_class"] for r in selected),
                         {0:16,1:16})
        self.assertFalse(report["nearphone_ever_negative"])
        self.assertTrue(all(r["tier"] in ("silver","weak") for r in selected))
        self.assertTrue(all(r["research_train_binding_created"] if
            "research_train_binding_created" in r else
            r["research_training_bind"]== (r["role"]=="train")
            for r in selected))

    def test_old_meero_never_training_negative(self):
        originals,screen=rows_and_screen()
        screen["rows"][3]["tier"]="silver"
        screen["rows"][3]["observed_class"]=0
        with tempfile.TemporaryDirectory() as d:
            Path(d,"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(v.review.observer,"source_rows",
                                   return_value=originals):
                with self.assertRaises(ValueError):
                    v.screen_new(d,d)

    def test_source_role_and_pcm_hash_drift_stop(self):
        originals,screen=rows_and_screen()
        screen["rows"][0]["pcm_sha256"]="f"*64
        with tempfile.TemporaryDirectory() as d:
            Path(d,"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(v.review.observer,"source_rows",
                                   return_value=originals):
                with self.assertRaisesRegex(ValueError,"WHOLE_PCM_IMMUTABLE"):
                    v.screen_new(d,d)

    def test_full_6_2_2_2_gate_only_machine_research(self):
        d={"schema":v.full.SCHEMA,
           "status":"PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE",
           "full_6_2_2_2_synthetic_machine_gate":True,
           "cumulative_machine_group_counts":{
              "train":11,"development":2,"calibration":2,"locked-test":5},
           "new_locked_source_groups_machine_qualified":4,
           "new_tts_asr_phone_model_calls":0,
           "kws_locked_evaluation":"NOT_RUN",
           "new_optimizer_updates":0,"new_model_weights":False,
           "training_snapshot_allowed":False,
           "product_promotion_allowed":False}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,"RESULT.json").write_text(json.dumps(d))
            self.assertTrue(v.frozen_gate(tmp)["research_only"])
            d["cumulative_machine_group_counts"]["locked-test"]=1
            Path(tmp,"RESULT.json").write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                v.frozen_gate(tmp)

    def test_exact_first_protected_main_and_no_replay_or_pr(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"987"}
        r=v.guard(env=env,event=event,parent_commit=v.BEFORE)
        self.assertEqual(r["new_tts_asr_calls"],0)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent_commit=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent_commit=v.BEFORE)

if __name__=="__main__":
    main()
