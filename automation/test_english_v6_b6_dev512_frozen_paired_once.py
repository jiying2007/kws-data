"""Saved-only original cohort and immutable three-model source contracts.

Actual original PT inference is prohibited on PR and runs only on the
first protected public-main successor with frozen post-B6 DEV512 artifacts.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v6_b6_dev512_frozen_source_test",
    HERE/"english_v6_b6_dev512_frozen_paired_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

CELLS=("mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge")
def dataset():
    originals=[];decisions=[]
    for i in range(8):
        for n,cell in enumerate(CELLS):
            requested=0 if n==2 else (None if n==3 else 2)
            positive=(i<5 and n==0) or (i==0 and n==1)
            observed=2 if positive else (0 if n==2 else None)
            tier=("weak" if i==0 and n==1 else
                  "silver" if positive or n==2 else "unknown")
            row={"id":f"new512-g{i}-{cell}","speaker_index":512+i,
                "source_group_id":f"g{i}","role":"development",
                "cell_id":cell,"requested_class":requested,
                "pcm_sha256":f"{i*4+n:064x}",
                "wav_sha256":f"{i*4+n+128:064x}",
                "samples":12000,"human_gold":False,
                "training_allowed":False}
            originals.append(row)
            decisions.append({**row,"tier":tier,"observed_class":observed})
    screen={"schema":m.saved.SCHEMA,
       "status":"PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY",
       "source_run":m.SOURCE["run"],
       "original_observer_run":44708054481,
       "source_group_counts":{"development":5},
       "tier_counts":{"silver":13,"weak":1,"unknown":18},
       "observed_label_counts":{"0":8,"2":6},
       "source_group_identity_members":[f"g{i}" for i in range(5)],
       "full_6_2_2_2_gate_passed":False,
       "nearphone_as_ordinary_negative":False,
       "machine_only_no_human_gold":True,
       "new_audio_or_model_forwards":0,"new_optimizer_updates":0,
       "new_kws_model_forwards":0,"training_snapshot_allowed":False,
       "locked_kws_test_opened":False,"product_promotion":False,
       "rows":decisions}
    return originals,screen

class B6FreshFrozenSourceOnly(TestCase):
    def test_actual_models_and_original_dev_sha_pins(self):
        r=m.source_check()
        self.assertEqual(r["status"],"SOURCE_ONLY_NO_NEURAL_FORWARD")
        self.assertEqual(r["new_source_run"],44706870732)
        self.assertEqual(r["saved_machine_screen_run"],44709245389)
        self.assertEqual(r["machine_qualified_positive"],6)
        self.assertEqual(r["machine_qualified_ordinary"],8)
        self.assertEqual(r["excluded_unknown"],18)
        self.assertEqual(r["qualified_development_groups"],5)
        self.assertEqual(r["frozen_gains"],[1.0,0.75,0.5])
        self.assertEqual(set(r["models"]),
                         {"B4_half","B6_control","B6_consistency"})
        self.assertEqual(m.ARCHIVES["B6"]["id"],14615025178)
        self.assertEqual(m.ARCHIVES["B6"]["sha"],
           "1c310b0d4945f7aa40f2576f231c94eb48b33316a6c4f79c20215b1ac80bab8d")
        self.assertEqual(m.MODELS["B6_control"]["pt"],
           "c9cac4c16a9d574b6bc9baef14ac239517a2ed9bfbe2e3ff2fb69972587c9430")
        self.assertEqual(m.MODELS["B6_consistency"]["pt"],
           "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04")
        self.assertEqual(m.BEFORE,"7b902b5ce7c86e5a4b250282e62a4c649ec47dc9")
        self.assertFalse(r["model_selection_allowed"])
        self.assertFalse(r["product_promotion_allowed"])
        self.assertEqual(r["new_model_forwards_in_PR"],0)

    def test_only_14_qualified_dev_originals_and_five_groups(self):
        originals,screen=dataset()
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                rows,receipt=m.select(tmp,tmp)
        self.assertEqual(len(rows),14)
        self.assertEqual(sum(r["mono_class"]==1 for r in rows),6)
        self.assertEqual(sum(r["mono_class"]==0 for r in rows),8)
        self.assertEqual(len(receipt["qualified_positive_source_groups"]),5)
        self.assertFalse(any(r["cell_id"]=="nearphone-meero-challenge" for r in rows))
        self.assertTrue(all(r["role"]=="development" for r in rows))

    def test_nearphone_and_dev_role_fail_closed(self):
        originals,screen=dataset()
        screen["rows"][-1]["tier"]="silver"
        screen["rows"][-1]["observed_class"]=0
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"NEARPHONE_MUST_REMAIN_UNKNOWN"):
                    m.select(tmp,tmp)
        originals,screen=dataset()
        originals[0]["role"]="locked-test"
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"BYTE_IDENTITY_ASR_SOURCE_BINDING"):
                    m.select(tmp,tmp)

    def test_contradictory_or_missing_screen_gate_stops(self):
        originals,screen=dataset()
        screen["tier_counts"]["unknown"]=19
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"FROZEN_INDEPENDENT_MACHINE_QA_ONLY"):
                    m.select(tmp,tmp)

    def test_one_first_public_main_no_pr_or_retry(self):
        event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"999"}
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"event.json";path.write_text(json.dumps(event))
            env["GITHUB_EVENT_PATH"]=str(path)
            with mock.patch.dict("os.environ",env,clear=True),mock.patch.object(
                    m.subprocess,"check_output",return_value=m.BEFORE):
                self.assertEqual(m.guard()["parent"],m.BEFORE)
            env["GITHUB_EVENT_NAME"]="pull_request"
            with mock.patch.dict("os.environ",env,clear=True):
                with self.assertRaisesRegex(ValueError,"FIRST_PROTECTED_PUBLIC_MAIN_ONLY"):
                    m.guard()
            env["GITHUB_EVENT_NAME"]="push";env["GITHUB_RUN_ATTEMPT"]="2"
            with mock.patch.dict("os.environ",env,clear=True):
                with self.assertRaisesRegex(ValueError,"FIRST_PROTECTED_PUBLIC_MAIN_ONLY"):
                    m.guard()
        with self.assertRaisesRegex(ValueError,"GITHUB_READ_ONLY_TOKEN_REQUIRED"):
            m.provenance("")

if __name__=="__main__":
    main()
