"""No optimizer/forward on PR; exact B8 DEV520 source and frozen model contracts."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock
import importlib.util
import json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b8_dev520_frozen_source_contract",
    HERE/"english_v6_b8_dev520_frozen_receipt_v2_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

CELLS=("mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge")

def synthetic_cohort():
    original=[];screen=[]
    for i in range(8):
        for n,cell in enumerate(CELLS):
            requested=0 if n==2 else (None if n==3 else 2)
            positive=(i<3 and n==0)
            ordinary=n==2
            obs=2 if positive else (0 if ordinary else None)
            tier=("silver" if positive or (ordinary and i<6) else
                  "weak" if ordinary else "unknown")
            row={"id":f"b8-520-{i}-{cell}","speaker_index":520+i,
                 "source_group_id":f"piper-new-{i}","role":"development",
                 "cell_id":cell,"requested_class":requested,
                 "pcm_sha256":f"{i*4+n:064x}",
                 "wav_sha256":f"{i*4+n+128:064x}",
                 "samples":12000,"human_gold":False,"training_allowed":False}
            original.append(row)
            screen.append({**row,"tier":tier,"observed_class":obs})
    report={"schema":m.saved.SCHEMA,
        "status":"PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY",
        "source_run":m.SOURCE["run"],
        "original_observer_run":44750685991,
        "source_group_counts":{"development":3},
        "tier_counts":{"silver":9,"weak":2,"unknown":21},
        "observed_label_counts":{"0":8,"2":3},
        "source_group_identity_members":[f"piper-new-{i}" for i in range(3)],
        "full_6_2_2_2_gate_passed":False,
        "nearphone_as_ordinary_negative":False,
        "machine_only_no_human_gold":True,
        "new_audio_or_model_forwards":0,"new_optimizer_updates":0,
        "new_kws_model_forwards":0,"training_snapshot_allowed":False,
        "locked_kws_test_opened":False,"product_promotion":False,
        "rows":screen}
    return original,report

class FrozenB8Dev520Tests(TestCase):
    def test_all_original_sha_and_model_pins(self):
        s=m.source_check()
        self.assertEqual(s["status"],"SOURCE_ONLY_NO_NEURAL_FORWARD")
        self.assertEqual(s["new_source_run"],44748801347)
        self.assertEqual(s["saved_machine_screen_run"],44761801165)
        self.assertEqual(s["machine_qualified_positive"],3)
        self.assertEqual(s["machine_qualified_ordinary"],8)
        self.assertEqual(s["excluded_unknown"],21)
        self.assertEqual(s["qualified_development_groups"],3)
        self.assertEqual(s["max_posttrain_forward_conditions"],99)
        self.assertEqual(s["frozen_gains"],[1.0,0.75,0.5])
        self.assertEqual(set(s["models"]),
                         {"B6_consistency","B8_control","B8_challenger"})
        self.assertEqual(m.ARCHIVES["B8"]["id"],14634066012)
        self.assertEqual(m.ARCHIVES["B8"]["sha"],
            "6015c6269f87654193d7350afd0c5b44f799f5f71b5303e41f2734c17006d15b")
        self.assertEqual(m.MODELS["B8_control"]["pt"],
            "a5cc0935c51afa1f82faed48cd50286c5bb5a9d136b0b4f752c176beb76df1e1")
        self.assertEqual(m.MODELS["B8_challenger"]["pt"],
            "cc16b81f84e4f21c5347ca153f405ac41ed116e3eedfaeed5677d1a4bc86e22f")
        self.assertEqual(m.SOURCE["run"],44748801347)
        self.assertEqual(m.SCREEN["run"],44761801165)
        self.assertEqual(m.BEFORE,"6749c028de03f6554058acc43d60fda7120dff9f")
        self.assertEqual(s["new_model_forwards_in_PR"],0)
        self.assertFalse(s["model_selection_allowed"])
        self.assertFalse(s["product_promotion_allowed"])

    def test_three_positive_eight_ordinary_exact_source_group(self):
        originals,screen=synthetic_cohort()
        with TemporaryDirectory() as folder:
            (Path(folder)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                rows,audit=m.select(folder,folder)
        self.assertEqual(len(rows),11)
        self.assertEqual(Counter(r["mono_class"] for r in rows),{0:8,1:3})
        self.assertEqual(len(audit["qualified_positive_source_groups"]),3)
        self.assertTrue(all(r["role"]=="development" for r in rows))
        self.assertFalse(any(r["cell_id"]=="nearphone-meero-challenge"
                             for r in rows))

    def test_nearphone_and_cross_role_fail_closed(self):
        originals,screen=synthetic_cohort()
        screen["rows"][-1]["tier"]="silver"
        screen["rows"][-1]["observed_class"]=0
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"NEARPHONE_MUST_REMAIN_UNKNOWN"):
                    m.select(tmp,tmp)
        originals,screen=synthetic_cohort()
        originals[0]["role"]="train"
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"BYTE_IDENTITY_ASR_SOURCE_BINDING"):
                    m.select(tmp,tmp)

    def test_reject_fabricated_screen_summary(self):
        originals,screen=synthetic_cohort()
        screen["tier_counts"]["unknown"]=20
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"FROZEN_INDEPENDENT_B8_MACHINE_SOURCE_QA_ONLY"):
                    m.select(tmp,tmp)

    def test_strict_first_main_guard_and_no_pr_model_inference(self):
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
        with self.assertRaisesRegex(ValueError,"GITHUB_READ_ONLY_TOKEN_REQUIRED"):
            m.provenance("")

    def test_frozen_previous_first_main_failed_before_any_valid_score(self):
        self.assertEqual(m.PRIOR_FAILED_RUN,44764450043)
        self.assertEqual(m.PRIOR_FAILED_HEAD,
                         "6749c028de03f6554058acc43d60fda7120dff9f")
        self.assertEqual(m.PRIOR_FAILED_ID,14644170620)
        self.assertEqual(m.PRIOR_FAILED_JOB,136385640635)
        self.assertEqual(m.PRIOR_FAILED_BYTES,708)
        self.assertEqual(m.PRIOR_FAILED_ZIP_SHA,
            "ec66066712b07d8be41cdf1f53764cfbf7bd8a4b776431120f04f13fa712e053")
        self.assertEqual(m.SCREEN["head"],
            "858e7550362bd92b51642934569071670fb6c5df")
        with self.assertRaisesRegex(ValueError,
                                    "EXACT_FAILED_B8_FIRST_MAIN_ORIGINAL_ZIP_SHA"):
            m.verify_previous_failed_archive("/missing/b8-prior-score-STOP.zip")
        src=(HERE/"english_v6_b8_dev520_frozen_receipt_v2_once.py").read_text()
        self.assertIn('and result.get("model_parameters") is None',src)
        self.assertIn('sum(p.numel() for p in net.parameters())==PARAMETERS',src)
        self.assertIn('verify_previous_failed_archive(prior_failed_zip)',src)
        self.assertIn('PRIOR_FAILED_ZIP_SHA',src)
        self.assertIn('wf.count(',src)
        self.assertNotIn('optimizer.step()',src)

if __name__=="__main__":
    main()
