#!/usr/bin/env python3
"""Verify new post-B8 source ancestry and freeze DEV-only purpose (no TTS on PR)."""
from __future__ import annotations
import importlib.util
from collections import Counter
from pathlib import Path
from unittest import TestCase, main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b8_posttrain_new_dev520",
    HERE/"english_v6_b8_fresh_dev520_source_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def fake_mapping():
    used=(set(v.previous.old.OLD_LIBRI_IDS)|
          set(v.previous.old.OLD_PIPER_IDS)|
          {speaker for _,speaker in v.previous.old.NEW_VOICES})
    first=max(used)+1000
    return {"num_speakers":904,
            "speaker_id_map":{"p"+str(first+i):i for i in range(904)}}

class B8NewDev520SourceTests(TestCase):
    def test_frozen_after_real_B8_and_unchanged_32_source_budget(self):
        p=v.plan()
        c=v.source_check()
        self.assertEqual(c["status"],"SOURCE_ONLY_NO_AUDIO_GENERATED")
        self.assertEqual(c["new_speaker_indices"],[520,527])
        self.assertEqual(c["new_speech_wav_budget"],32)
        self.assertEqual(c["new_optimizer_updates"],0)
        self.assertEqual(c["new_asr_phone_calls"],0)
        self.assertFalse(c["nearphone_is_eligible_ordinary"])
        self.assertFalse(c["independent_human_gold"])
        self.assertFalse(c["product_promotion"])
        self.assertEqual(c["source_roles"],{"development":8})
        self.assertEqual(p["status"],"FROZEN_BEFORE_ANY_NEW_WAV")
        self.assertEqual(p["source_count"],32)
        self.assertEqual(p["target"]["keyword"],"Hey Mero")
        self.assertEqual(p["target"]["active_english_keyword_count"],1)
        self.assertEqual([x["speaker_index"] for x in p["source_groups"]],
                         list(range(520,528)))
        self.assertEqual(p["minimum_groups_eligible"],
                         {"train":6,"development":2,"calibration":2,"locked-test":2})
        self.assertEqual(p["source_guards"]["known_prior_piper_indices"][:18],
                         "0..57 and 400..519")
        self.assertTrue(p["source_guards"]["nearphone_is_unlabeled_challenge"])
        self.assertTrue(p["source_guards"]["no_trusted_speech_gold_from_prompt"])
        self.assertEqual(p["budget"]["max_new_piper_tts_forwards"],32)
        self.assertEqual(p["budget"]["max_new_asr_forwards_at_source_stage"],0)
        self.assertEqual(p["budget"]["retry_count"],0)
        self.assertIn("488..519",p["excluded_prior_development_sources"]["piper_speaker_indices"])
        model=p["b8_frozen_models"]
        self.assertEqual(model["data_main_sha"],
            "453b26bdd4ab2bdc96b27fe0dfb9ac10bdd38711")
        self.assertEqual(model["real_train_actions_run"],44746571932)
        self.assertEqual(model["real_train_artifact_id"],14634066012)
        self.assertEqual(model["real_train_artifact_zip_sha256"],
            "6015c6269f87654193d7350afd0c5b44f799f5f71b5303e41f2734c17006d15b")
        self.assertEqual(model["B8_control_pt_sha256"],
            "a5cc0935c51afa1f82faed48cd50286c5bb5a9d136b0b4f752c176beb76df1e1")
        self.assertEqual(model["B8_challenger_pt_sha256"],
            "cc16b81f84e4f21c5347ca153f405ac41ed116e3eedfaeed5677d1a4bc86e22f")
        self.assertEqual(model["B8_optimizer_updates"],384)
        self.assertEqual(model["B8_trained_tensor_values_different"],43755)
        self.assertEqual(model["B7_original_strict_replay"],"FAILED_UNCHANGED")
        self.assertFalse(model["human_gold"])
        self.assertFalse(model["release_allowed"])

    def test_exact_new_speaker_map_and_prior_identity_rejection(self):
        choices=v.native_speakers(fake_mapping())
        self.assertEqual(list(choices),list(range(520,528)))
        self.assertEqual(len(set(choices.values())),8)
        rows=v.planned(choices)
        self.assertEqual(len(rows),32)
        self.assertEqual({r["role"] for r in rows},{"development"})
        self.assertEqual(Counter(x["requested_class"] for x in rows),
                         {2:16,0:8,None:8})
        self.assertTrue(all(x["observed_class"] is None for x in rows))
        self.assertTrue(all(not x["training_allowed"] for x in rows))
        self.assertTrue(all(x["label_basis"]==
                            "SOURCE_REQUEST_NOT_ACOUSTIC_TRUTH" for x in rows))
        self.assertTrue(all(x["id"].startswith("merov6b8dev8-") for x in rows))
        config=fake_mapping()
        pairs=config["speaker_id_map"]
        first=next(k for k,v in pairs.items() if v==520)
        del pairs[first]
        prior=next(iter(v.previous.old.OLD_LIBRI_IDS))
        pairs["p"+str(prior)]=520
        with self.assertRaisesRegex(ValueError,"DUPLICATE_PRIOR_HUMAN"):
            v.native_speakers(config)

    def test_audio_stage_never_runs_on_pr_or_older_main(self):
        ev={"before":v.MAIN_BEFORE,"after":"a"*40,"ref":"refs/heads/main",
            "repository":{"full_name":v.REPO,"private":False},
            "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"1"}
        self.assertEqual(v.guard(env=env,event=ev,parent=v.MAIN_BEFORE)["attempt"],1)
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=ev,parent=v.MAIN_BEFORE)
        env["GITHUB_EVENT_NAME"]="push";env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=ev,parent=v.MAIN_BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=ev,parent="f"*40)

if __name__=="__main__":
    main()
