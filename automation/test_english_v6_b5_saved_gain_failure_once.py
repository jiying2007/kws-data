"""Unit-tested saved-only B5 paired DEV504 pathology, no neural forward."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
from pathlib import Path
from unittest import TestCase, main

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "b5_saved_forensic_test", HERE/"english_v6_b5_saved_gain_failure_once.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rows = []
    for i in range(13):
        positive = i < 5
        conditions = {}
        for gain in m.GAINS:
            entries = {}
            for model in m.MODELS:
                if not positive:
                    hit = False
                elif gain == "0.5":
                    hit = i < (2 if model == "B5_early_hey" else 3)
                else:
                    hit = True
                entries[model] = {
                    "full_phrase_event_count": int(hit),
                    "any_complete_event": hit,
                    "first_frame_argmax": 0,
                    "any_meh_argmax": not (
                        positive and gain == "0.5"
                        and model == "B5_early_hey" and i == 4),
                    "collapsed_tokens": [1,2,3] if hit else [1,2],
                    "frames": 90,
                    "ctc_extra_events": 0,
                }
            conditions[gain] = entries
        rows.append({
            "id": f"dev504-{i}",
            "source_group_id": f"g{i % 3}" if positive else f"n{i}",
            "role": "development",
            "machine_tier": "silver" if i % 2 == 0 else "weak",
            "pcm_sha256": f"{i+1:064x}",
            "positive": positive,
            "conditions": conditions,
        })
    summary = {}
    for gain in m.GAINS:
        summary[gain] = {}
        for model in m.MODELS:
            pos = rows[:5]
            neg = rows[5:]
            summary[gain][model] = {
                "positive_hits": sum(x["conditions"][gain][model]["any_complete_event"] for x in pos),
                "positive_originals": 5,
                "ordinary_originals": 8,
                "ordinary_complete_event_clips": 0,
                "ordinary_false_or_extra_events": 0,
                "positive_missing_meh_argmax": sum(
                    not x["conditions"][gain][model]["any_meh_argmax"] for x in pos),
            }
    return {
        "schema": m.RESULT_SCHEMA,
        "status": "B5_NEW_SOURCE_FROZEN_THREE_WAY_DEV_DIAGNOSTIC_ONLY",
        "model_pt_sha256": m.MODELS.copy(),
        "B4_half_B5_control_identical_tensors": True,
        "qualified_positive_source_groups": 3,
        "same_tts_engine_not_human_gold": True,
        "cal_locked_opened": False,
        "model_selection_allowed": False,
        "product_promotion_allowed": False,
        "new_optimizer_steps": 0,
        "new_tts_asr_calls": 0,
        "derived_conditions_do_not_increase_original_denominator": True,
        "summary": summary,
        "per_original": rows,
    }


class B5SavedOnlyTests(TestCase):
    def test_original_117_inference_identity_and_zero_new_neural(self):
        x = m.source_check()
        self.assertEqual(x["status"],"ONLY_PREVIOUSLY_SAVED_B5_RESULT")
        self.assertEqual(x["original_result_id"],14611963808)
        self.assertEqual(x["original_forward_cases"],117)
        self.assertEqual(x["original_positive"],5)
        self.assertEqual(x["original_ordinary"],8)
        self.assertEqual(x["independent_qualified_dev_groups"],3)
        self.assertEqual(x["new_model_forwards"],0)
        self.assertEqual(x["new_optimizer_updates"],0)
        self.assertFalse(x["cal_locked_access"])
        self.assertFalse(x["product_promotion_allowed"])
        self.assertEqual(m.ORIGINAL["digest"],
            "67267fcdf78535a75854645de60e7ad662aaa5016788b62e9901c1f073e9b016")

    def test_original_logit_summaries_and_half_gain_regression(self):
        result = m.analyze(fixture())
        self.assertEqual(result["status"],
                         "PASS_IMMUTABLE_B5_LOWGAIN_CTC_PATHOLOGY_DIAGNOSTIC_ONLY")
        self.assertEqual(result["by_gain"]["0.5"]["B4_half"]["positive_hits"],3)
        self.assertEqual(result["by_gain"]["0.5"]["B5_control"]["positive_hits"],3)
        self.assertEqual(result["by_gain"]["0.5"]["B5_early_hey"]["positive_hits"],2)
        self.assertEqual(result["half_gain_B5_net_new_positive_misses"],1)
        self.assertEqual(len(result["half_gain_discordant_positive_origins"]),1)
        self.assertFalse(result["half_gain_frame_logits_saved"])
        self.assertFalse(result["startup_or_phone_timing_cause_proven"])
        self.assertFalse(result["product_promotion_allowed"])
        self.assertEqual(result["new_model_forwards"],0)

    def test_reject_tampered_source_class_and_counts(self):
        raw = fixture()
        raw["per_original"][-1]["positive"] = True
        with self.assertRaisesRegex(ValueError,"EXACT_13"):
            m.analyze(raw)
        raw = fixture()
        raw["summary"]["0.5"]["B5_early_hey"]["positive_hits"] = 3
        with self.assertRaisesRegex(ValueError,"ORIGINAL_NUMERIC_SUMMARY"):
            m.analyze(raw)

    def test_reject_model_replay_drift(self):
        raw = fixture()
        raw["per_original"][2]["conditions"]["0.5"]["B5_control"]["first_frame_argmax"] = 2
        with self.assertRaisesRegex(ValueError,"EXACT_B4_HALF_B5_CONTROL"):
            m.analyze(raw)

    def test_reject_unknown_as_ordinary_and_model_promotion(self):
        raw = fixture()
        raw["per_original"][12]["machine_tier"] = "unknown"
        with self.assertRaisesRegex(ValueError,"EXACT_13"):
            m.analyze(raw)
        raw = fixture()
        raw["model_selection_allowed"] = True
        with self.assertRaisesRegex(ValueError,"ORIGINAL_NONPROMOTABLE"):
            m.analyze(raw)

    def test_first_main_only_no_pr_rerun(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"1000"}
        event={"before":m.PARENT,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(m.guard(env=env,event=event,parent=m.PARENT)["before"],m.PARENT)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN"):
            m.guard(env=env,event=event,parent=m.PARENT)
        env["GITHUB_RUN_ATTEMPT"]="1";env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN"):
            m.guard(env=env,event=event,parent=m.PARENT)


if __name__ == "__main__":
    main()
