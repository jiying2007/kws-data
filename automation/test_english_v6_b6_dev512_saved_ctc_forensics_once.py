"""B6 saved 126-forward immutable source, CTC path and no-promotion tests."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
from pathlib import Path
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b6_saved_ctc_forensics_test",
    HERE/"english_v6_b6_dev512_saved_ctc_forensics_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PARENT={
    "1.0":{0,1,2},
    "0.75":{0,1,2,3},
    "0.5":{0},
}
CHALLENGER={
    "1.0":{0,1,2,3,4},
    "0.75":{0,1,2},
    "0.5":{0},
}

def fixture():
    originals=[]
    for i in range(14):
        positive=i<6
        conditions={}
        for gain in m.GAINS:
            conditions[gain]={}
            for name in m.MODEL_PT:
                if positive:
                    hit=i in (CHALLENGER[gain] if name=="B6_consistency"
                              else PARENT[gain])
                else:
                    hit=name=="B6_consistency" and gain=="1.0" and i==6
                if hit:
                    path=[1,2,3]
                elif positive and gain=="0.5" and i==5:
                    path=[1,3]
                elif positive:
                    path=[1,2]
                else:
                    path=[3,2]
                events=m.ctc_events(path)
                assert events==int(hit)
                conditions[gain][name]={
                    "full_phrase_event_count":events,
                    "any_complete_event":bool(events),
                    "first_frame_argmax":0,
                    "any_meh_argmax":2 in path,
                    "collapsed_tokens":path,
                    "frames":100,
                    "ctc_extra_events":max(0,events-1),
                }
        originals.append({
           "id":f"merov6b6dev8-{512+i%8}-{'mero-fast' if positive else 'ordinary-time'}-{i}",
           "source_group_id":f"g{i%5}" if positive else f"g{i%8}",
           "role":"development","machine_tier":"silver" if i%2==0 else "weak",
           "pcm_sha256":f"{i+1:064x}","positive":positive,
           "conditions":conditions,
        })
    summary={}
    for gain in m.GAINS:
        summary[gain]={}
        for name in m.MODEL_PT:
            pos=originals[:6];neg=originals[6:]
            summary[gain][name]={
                "positive_originals":6,
                "positive_hits":sum(r["conditions"][gain][name]["any_complete_event"]
                                    for r in pos),
                "ordinary_originals":8,
                "ordinary_false_or_extra_events":sum(
                    r["conditions"][gain][name]["full_phrase_event_count"] for r in neg),
                "ordinary_complete_event_clips":sum(
                    r["conditions"][gain][name]["any_complete_event"] for r in neg),
                "positive_missing_meh_argmax":sum(
                    not r["conditions"][gain][name]["any_meh_argmax"] for r in pos),
            }
    return {
        "schema":m.RESULT_SCHEMA,
        "status":"B6_NEW_SOURCE_FROZEN_THREE_WAY_DEV_DIAGNOSTIC_ONLY",
        "model_pt_sha256":m.MODEL_PT.copy(),
        "qualified_positive_source_groups":5,
        "B4_half_B6_control_identical_tensors":True,
        "derived_conditions_do_not_increase_original_denominator":True,
        "same_tts_engine_not_human_gold":True,
        "cal_locked_opened":False,"new_optimizer_steps":0,
        "new_tts_asr_calls":0,"model_selection_allowed":False,
        "winner":None,"product_promotion_allowed":False,
        "conditions":[1.0,0.75,0.5],
        "artifacts":{"source":m.SOURCE_ARTIFACT,
                     "screen":m.SCREEN_ARTIFACT,
                     "models":{"B4":14597862669,"B6":14615025178}},
        "source_receipt":{
             "status":"AUDITED_SIX_NEW_POSITIVE_EIGHT_ORDINARY",
             "new_originals":14,
             "qualified_positive_source_groups":[f"g{i}" for i in range(5)],
             "human_gold":False,"no_cal_locked":True,
        },
        "per_original":originals,"summary":summary,
    }

class Original126CaseSourceOnly(TestCase):
    def test_original_evidence_is_immutable_and_nonpromotable(self):
        p=m.source_check()
        self.assertEqual(p["status"],
             "SOURCE_ONLY_126_PREVIOUSLY_EXECUTED_FROZEN_CASES")
        self.assertEqual(p["result_run"],44710802470)
        self.assertEqual(p["result_artifact"],14616493798)
        self.assertEqual(p["source_artifact"],14616065335)
        self.assertEqual(p["machine_screen_artifact"],14616173070)
        self.assertEqual(p["original_model_forward_cases"],126)
        self.assertEqual(p["new_tts_asr_phone_model_forwards"],0)
        self.assertEqual(p["new_optimizer_updates"],0)
        self.assertFalse(p["product_promotion_allowed"])
        self.assertEqual(len(m.ORIGINAL["digest"]),64)
        self.assertEqual(m.ORIGINAL["digest"],
             "b900473e57a5b53793a42020838371c1c12f4ee70d73480f2f68c03aab37801f")

    def test_valid_real_aggregate_all_gain_and_single_ordinary_false_wake(self):
        proof=m.analyze(fixture())
        self.assertEqual(proof["status"],
             "PASS_IMMUTABLE_B6_126_CASE_CTC_FORENSICS_RESEARCH_ONLY")
        self.assertEqual(proof["by_gain"]["1.0"]["B4_half"]["positive_hits"],3)
        self.assertEqual(proof["by_gain"]["1.0"]["B6_consistency"]["positive_hits"],5)
        self.assertEqual(proof["by_gain"]["0.75"]["B6_consistency"]["positive_hits"],3)
        self.assertEqual(proof["by_gain"]["0.5"]["B6_consistency"]["positive_hits"],1)
        self.assertEqual(proof["by_gain"]["1.0"]["B6_consistency"]["ordinary_events"],1)
        self.assertEqual(len(proof["ordinary_false_wake_origins"]),1)
        self.assertEqual(proof["ordinary_false_wake_origins"][0]["gain"],"1.0")
        self.assertEqual(proof["ordinary_false_wake_origins"][0]["B6_ctc"],[1,2,3])
        self.assertEqual(len(proof["positive_discordances"]["1.0"]),2)
        self.assertEqual(len(proof["positive_discordances"]["0.75"]),1)
        self.assertEqual(len(proof["positive_discordances"]["0.5"]),0)
        self.assertFalse(proof["product_promotion_allowed"])
        self.assertEqual(proof["new_audio_asr_phone_model_forwards"],0)

    def test_reject_corrupted_any_ctc_token_event_or_summary(self):
        original=fixture()
        original["per_original"][6]["conditions"]["1.0"]["B6_consistency"][
            "collapsed_tokens"]=[1,3]
        with self.assertRaisesRegex(ValueError,"RECOMPUTED_EXACT_CTC"):
            m.analyze(original)
        original=fixture()
        original["summary"]["1.0"]["B6_consistency"]["positive_hits"]=6
        with self.assertRaisesRegex(ValueError,"PER_ORIGINAL_SUMS"):
            m.analyze(original)

    def test_reject_control_drift_even_when_aggregates_match(self):
        original=fixture()
        original["per_original"][1]["conditions"]["0.5"]["B6_control"][
            "first_frame_argmax"]=2
        with self.assertRaisesRegex(ValueError,"EXACT_PARENT_AND_CONTROL"):
            m.analyze(original)

    def test_unknown_prompt_and_gold_never_become_ordinary_or_product(self):
        original=fixture()
        original["per_original"][13]["machine_tier"]="unknown"
        with self.assertRaisesRegex(ValueError,"REAL_14_DISTINCT"):
            m.analyze(original)
        original=fixture()
        original["product_promotion_allowed"]=True
        with self.assertRaisesRegex(ValueError,"NO_NEW_MODEL_OR_HUMAN_GOLD"):
            m.analyze(original)

    def test_guard_only_one_exact_protected_main_successor(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"999"}
        event={"before":m.PARENT,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(m.guard(env=env,event=event,parent=m.PARENT)["parent"],m.PARENT)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"EXACT_FIRST_PUBLIC_MAIN"):
            m.guard(env=env,event=event,parent=m.PARENT)
        env["GITHUB_RUN_ATTEMPT"]="1";env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaisesRegex(ValueError,"EXACT_FIRST_PUBLIC_MAIN"):
            m.guard(env=env,event=event,parent=m.PARENT)

if __name__=="__main__":
    main()
