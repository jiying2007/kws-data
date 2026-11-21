#!/usr/bin/env python3
"""B9 DEV528-539: source-only provenance checks; PR must not generate WAVs."""
from __future__ import annotations
from collections import Counter
import importlib.util
from pathlib import Path
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location(
    "b9_fresh_dev528_source",HERE/"english_v6_b9_fresh_dev528_source_once.py")
v=importlib.util.module_from_spec(sp)
sp.loader.exec_module(v)

def fake_mapping():
    used=set(v.previous.old.OLD_LIBRI_IDS)|set(v.previous.old.OLD_PIPER_IDS)
    used|={sid for _,sid in v.previous.old.NEW_VOICES}
    first=max(used)+1000
    return {"num_speakers":904,"speaker_id_map":{
        "p"+str(first+i):i for i in range(904)}}

class B9Dev528Source(TestCase):
    def test_all_prior_and_new_model_pins(self):
        p=v.plan();c=v.source_check()
        self.assertEqual(c["status"],"SOURCE_ONLY_NO_AUDIO_GENERATED")
        self.assertEqual(c["new_speaker_indices"],[528,539])
        self.assertEqual(c["new_speech_wav_budget"],60)
        self.assertEqual(c["source_roles"],{"development":12})
        self.assertEqual(c["new_optimizer_updates"],0)
        self.assertEqual(c["new_asr_phone_calls"],0)
        self.assertFalse(c["nearphone_is_eligible_ordinary"])
        self.assertFalse(c["independent_human_gold"])
        self.assertFalse(c["product_promotion"])
        self.assertEqual(p["source_count"],60)
        self.assertEqual(p["status"],"FROZEN_BEFORE_ANY_NEW_WAV")
        self.assertEqual(p["target"]["keyword"],"Hey Mero")
        self.assertEqual(p["target"]["ipa"],"/heɪ ˈmɛroʊ/")
        self.assertEqual(p["model"]["num_speakers"],904)
        self.assertEqual(p["model"]["file_sha256"],
             "9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f")
        self.assertEqual([x["speaker_index"] for x in p["source_groups"]],
                         list(range(528,540)))
        self.assertTrue(p["source_guards"]["hard_ordinary_requires_five_native_source_review"])
        self.assertTrue(p["source_guards"]["all_TTS_intents_are_unverified"])
        self.assertTrue(p["source_guards"]["nearphone_is_unlabeled_challenge"])
        self.assertFalse(p["source_guards"]["hard_nearphone_is_automatically_ordinary"])
        self.assertEqual(p["budget"],{
            "max_new_piper_tts_forwards":60,
            "max_new_asr_forwards_at_source_stage":0,
            "max_new_kws_train_updates":0,"retry_count":0})
        m=p["b9_frozen_models"]
        self.assertEqual(m["data_main_sha"],
            "846c745ebcee58a1beb27300ddcd7ec8e2178916")
        self.assertEqual(m["real_train_actions_run"],44860855857)
        self.assertEqual(m["real_train_artifact_id"],14689218171)
        self.assertEqual(m["real_train_artifact_zip_sha256"],
            "1a8b270200e0c72ab1025f139d16abde43c209f142d72e71a7201b847e06a612")
        self.assertEqual(m["B9_control_pt_sha256"],
            "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655")
        self.assertEqual(m["B9_challenger_pt_sha256"],
            "d672dcc87ca9e5d2c6dcf5952251a3c40fe8e7956d3b7c8753667c3093761cea")
        self.assertEqual(m["B9_optimizer_updates"],384)
        self.assertEqual(m["B9_trained_tensor_values_different"],43754)
        self.assertFalse(m["human_gold"])
        self.assertFalse(m["release_allowed"])
    def test_exact_new_piper_identity_and_frozen_roles(self):
        choices=v.native_speakers(fake_mapping())
        self.assertEqual(list(choices),list(range(528,540)))
        self.assertEqual(len(set(choices.values())),12)
        rows=v.planned(choices)
        self.assertEqual(len(rows),60)
        self.assertEqual(len({r["source_group_id"] for r in rows}),12)
        self.assertEqual(Counter(r["requested_class"] for r in rows),
                         {2:24,0:24,None:12})
        self.assertEqual(Counter(r["cell_id"] for r in rows),{
            "mero-fast":12,"mero-slow":12,"ordinary-time":12,
            "hard-hey-mary":12,"nearphone-meero-challenge":12})
        self.assertTrue(all(x["role"]=="development" for x in rows))
        self.assertTrue(all(x["observed_class"] is None for x in rows))
        self.assertTrue(all(x["human_gold"] is False for x in rows))
        self.assertTrue(all(x["training_allowed"] is False for x in rows))
        self.assertTrue(all(x["label_basis"]==
             "SOURCE_REQUEST_NOT_ACOUSTIC_TRUTH" for x in rows))
        self.assertTrue(all(x["id"].startswith("merov6b9dev12-") for x in rows))
        data=fake_mapping();pairs=data["speaker_id_map"]
        key=next(k for k,x in pairs.items() if x==528)
        del pairs[key]
        prior=next(iter(v.previous.old.OLD_LIBRI_IDS))
        pairs["p"+str(prior)]=528
        with self.assertRaisesRegex(ValueError,"DUPLICATE_PRIOR_HUMAN"):
            v.native_speakers(data)
    def test_only_first_exact_main_push_can_generate(self):
        event={"before":v.MAIN_BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"1234"}
        self.assertEqual(v.guard(env=env,event=event,parent=v.MAIN_BEFORE)["attempt"],1)
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.MAIN_BEFORE)
        env["GITHUB_EVENT_NAME"]="push";env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.MAIN_BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent="f"*40)
        event["repository"]["private"]=True
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.MAIN_BEFORE)
if __name__=="__main__":
    main()
