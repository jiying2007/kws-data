"""Fail-closed model-free readback of a measured Mero-only research artifact.

The retained JSON is a compact aggregation of the original Actions artifact.
This does NOT recompute source audio, authenticate human truth, or run any model.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs" / "ENGLISH63_ZH_EN_MERO_EVENT_READBACK_20261113.json"
EXPECTED_ARTIFACT_SHA = "aef82fd070d7057a5f025cecc1a92a1700c8f97440015844d09783bc048e8bf4"
EXPECTED_MEMBER_SHA = "0e8614392c8fb6d263ae7e5cc1e074610773e8ec046dd7028baa76899733c383"
EXPECTED_TTS_SHA = "f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc"
EXPECTED_ZHEN_SHA = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"

SOURCE_PROMPTS = {
    "Hey Mero", "Hey, Mero!", "Hey Miro", "Hey Rosen", "Hey Vosen",
    "Hey Wosen", "Hey, Vosen!", "Hey, can you hear me?", "What time is it?"
}
TARGETS = {"Hey Mero":4,"Hey, Mero!":5}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def verify_summary(v):
    need(type(v) is dict and
         v.get("schema") == "english63-zh-en-mero-only-observed-event-readback-v1" and
         v.get("status") ==
         "SOURCE_BOUND_SYNTHETIC_EVENT_DIAGNOSTIC_NOT_MODEL_QUALIFICATION",
         "MERO_READBACK_SCHEMA")
    source = v["source"]
    need(source.get("repository") == "jiying2007/kws-data" and
         source.get("implementation_merge_sha") ==
         "8400da16e4f093641716b649e7c5d5f688977a40" and
         source.get("actual_workflow_run_id") == 43448541871 and
         source.get("artifact_id") == 14049636426 and
         source.get("artifact_zip_sha256") == EXPECTED_ARTIFACT_SHA and
         source.get("receipt_member_sha256") == EXPECTED_MEMBER_SHA and
         source.get("receipt_member_bytes") == 40048 and
         source.get("original_tts_run_id") == 43242278962 and
         source.get("original_tts_artifact_id") == 13954744587 and
         source.get("original_tts_zip_sha256") == EXPECTED_TTS_SHA and
         source.get("official_model_archive_sha256") == EXPECTED_ZHEN_SHA,
         "MERO_READBACK_PROVENANCE")
    experiment = v["experiment"]
    need(experiment.get("keyword_original_text") == "Hey Mero" and
         experiment.get("official_phone_tokens") ==
         "HH EY1 M EH1 R OW0 @HEY_MERO" and
         experiment.get("candidate_hey_vosen_not_evaluated") is True and
         experiment.get("keyword_count") == 1 and
         experiment.get("synthetic_source_families") == 7 and
         experiment.get("original_audio_clips") == 63 and
         experiment.get("original_audio_samples") == 2101760 and
         experiment.get("model_adaptation_or_finetune") is False and
         experiment.get("decoder_policy") == {
             "threads":1,"feature_dim":80,"max_active_paths":4,
             "keywords_score":1.0,"keywords_threshold":0.25,
             "num_trailing_blanks":1,"feed_samples":1600,
             "tail_samples":10560},
         "MERO_READBACK_PROTOCOL")
    obs = v["observed"]
    need(type(obs["requested_text"]) is dict and
         set(obs["requested_text"]) == SOURCE_PROMPTS and
         all(type(x) is dict and x.get("clips")==7 and
             type(x.get("events")) is int and x["events"] >= 0 and
             type(x.get("clips_with_event")) is int and
             0 <= x["clips_with_event"] <= 7 and
             x["events"] >= x["clips_with_event"]
             for x in obs["requested_text"].values()),
         "MERO_READBACK_REQUEST_ROSTER")
    for name, count in TARGETS.items():
        need(obs["requested_text"][name]["events"] == count and
             obs["requested_text"][name]["clips_with_event"] == count,
             "MERO_READBACK_TARGET_COUNT")
    for name in SOURCE_PROMPTS - set(TARGETS):
        need(obs["requested_text"][name]["events"] == 0 and
             obs["requested_text"][name]["clips_with_event"] == 0,
             "MERO_READBACK_OTHER_PROMPT_COUNT")
    need(obs.get("total_event_count") == 9 and
         obs.get("total_clips_with_event") == 9 and
         obs.get("target_requested_clips") == 14 and
         obs.get("target_requested_clips_with_event") == 9 and
         obs.get("target_requested_event_count") == 9 and
         obs.get("other_requested_clips") == 49 and
         obs.get("other_requested_clips_with_event") == 0 and
         obs.get("other_requested_event_count") == 0 and
         sum(x["clips"] for x in obs["requested_text"].values()) == 63 and
         sum(x["events"] for x in obs["requested_text"].values()) == 9,
         "MERO_READBACK_SOURCE_EVENT_COUNT")
    families = obs["synthetic_voice_families"]
    need(type(families) is dict and len(families) == 7 and
         all(x.get("clips") == 9 and x.get("target_requested") == 2 and
             type(x.get("target_clips_with_event")) is int and
             type(x.get("total_events")) is int and
             x["target_clips_with_event"] == x["total_events"] and
             0 <= x["total_events"] <= 2 for x in families.values()) and
         sum(x["total_events"] for x in families.values()) == 9 and
         obs.get("synthetic_voices_with_at_least_one_event") ==
         sum(x["total_events"] > 0 for x in families.values()) == 6 and
         obs.get("event_tokens_missing") == 9 and
         obs.get("event_token_timestamps_missing") == 9 and
         obs.get("input_availability_is_not_end_of_word_latency") is True and
         obs.get("host_timing_excludes_model_load_and_ssc305") is True,
         "MERO_READBACK_VOICE_OR_LATENCY")
    compare = v["comparison"]
    need(compare.get("english_gigaspeech_pretained_run_id") == 43443799469 and
         compare.get("english_gigaspeech_event_count_on_source63") == 0 and
         compare.get("gigaspeech_vendor_control_run_id") == 43445799002 and
         compare.get("gigaspeech_vendor_positive_control_passed") is True and
         compare.get("model_keyword_lists_differ") is True and
         compare.get("no_winner_from_untrusted_weak_labels") is True,
         "MERO_READBACK_NO_FAKE_MODEL_WIN")
    limits = v["limits"]
    must_false = (
        "source_requested_text_is_acoustic_gold", "human_gold",
        "formal_training_admitted", "independent_test_qualified",
        "model_on_ssc305", "model_promotion_allowed")
    must_true = (
        "nearphones_untrusted_as_negatives", "no_new_tts_or_asr_calls",
        "no_new_audio_generated", "no_new_training_or_threshold_sweep")
    need(all(limits.get(key) is False for key in must_false) and
         all(limits.get(key) is True for key in must_true) and
         limits.get("human_speaker_count") == 0 and
         all(limits.get(key) == "NOT_RUN" for key in (
             "continuous_negative_far_per_hour","independent_human_frr",
             "accent_far_field_3to5m","latency_p99")),
         "MERO_READBACK_FALSE_QUALIFICATION")
    return True


class MeroMeasuredReadbackCases(unittest.TestCase):
    def test_original_measured_source_only(self):
        v=json.loads(RECEIPT.read_text(encoding="utf-8"))
        self.assertTrue(verify_summary(v))
        self.assertEqual(v["observed"]["total_clips_with_event"],9)

    def test_untrustworthy_positive_or_fake_far_fails_closed(self):
        source=json.loads(RECEIPT.read_text(encoding="utf-8"))
        mutations=(
            lambda x:x["source"].__setitem__("artifact_zip_sha256","f"*64),
            lambda x:x["source"].__setitem__("receipt_member_sha256","f"*64),
            lambda x:x["experiment"].__setitem__("candidate_hey_vosen_not_evaluated",False),
            lambda x:x["observed"].__setitem__("total_clips_with_event",10),
            lambda x:x["observed"]["requested_text"]["Hey Miro"].__setitem__("events",1),
            lambda x:x["observed"]["synthetic_voice_families"]["qwen3-preset-sohee"].__setitem__("total_events",2),
            lambda x:x["observed"].__setitem__("event_token_timestamps_missing",0),
            lambda x:x["comparison"].__setitem__("model_keyword_lists_differ",False),
            lambda x:x["limits"].__setitem__("model_promotion_allowed",True),
            lambda x:x["limits"].__setitem__("continuous_negative_far_per_hour",0.0),
            lambda x:x["limits"].__setitem__("human_gold",True),
        )
        for mutate in mutations:
            trial=copy.deepcopy(source)
            mutate(trial)
            with self.assertRaises(ValueError):
                verify_summary(trial)


if __name__=="__main__":
    unittest.main()
