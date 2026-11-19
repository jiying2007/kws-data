"""New original Piper prospective_dev8 source-only contract, no models on PR."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import importlib.util
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
   "v5_newprospective_dev8_test",HERE/"english_v5_mero_prospective_dev8_source_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def fake_mapping():
    return {"num_speakers":904,"speaker_id_map":{
            "p"+str(20000+i):i for i in range(904)}}

class NewProspectiveDev8PiperSourceTests(TestCase):
    def test_exact_frozen_speaker_and_new_budget_contract(self):
        plan=v.plan()
        check=v.source_check()
        self.assertEqual(check["new_speaker_indices"],[480,487])
        self.assertEqual(check["new_speech_wav_budget"],32)
        self.assertEqual(check["source_roles"],{"development":8})
        self.assertEqual(check["new_optimizer_updates"],0)
        self.assertEqual(check["new_asr_phone_calls"],0)
        self.assertEqual(check["only_english_keyword"],"Hey Mero")
        self.assertFalse(check["nearphone_is_eligible_ordinary"])
        self.assertFalse(check["product_promotion"])
        self.assertEqual(plan["source_count"],32)
        self.assertEqual(plan["frozen_reference"]["strong_model_pt_sha256"],"5025b9e8ee6379074abc615a2e32859ea3e90ed436a855402e28bfe54a479ead")
        self.assertEqual(plan["frozen_reference"]["phone_exposed_dev_positive_hits"],2)
        self.assertEqual([x["speaker_index"] for x in plan["source_groups"]],
                         list(range(480,488)))
        self.assertEqual(Counter(x["role"] for x in plan["source_groups"]),
                         {"development":8})
        self.assertEqual([x["requested_class"] for x in plan["cells"]],
                         [2,2,0,None])

    def test_new_model_source_ancestry_truly_disjoint(self):
        original=fake_mapping()
        chosen=v.native_speakers(original)
        self.assertEqual(set(chosen),set(range(480,488)))
        self.assertEqual(len(set(chosen.values())),8)
        ids={int(label[1:]) for label in chosen.values()}
        old_ids={int(label[1:]) for label,index in original["speaker_id_map"].items()
                 if index in (set(range(58))|set(range(400,480)))}
        self.assertFalse(ids & old_ids)
        self.assertEqual(len(v.planned(chosen)),32)

    def test_old_human_identity_or_duplicate_is_a_hard_stop(self):
        config=fake_mapping()
        pairs=config["speaker_id_map"]
        del pairs["p"+str(20000+472)]
        forbidden=next(iter(v.previous.old.OLD_LIBRI_IDS))
        pairs["p"+str(forbidden)]=472
        with self.assertRaisesRegex(ValueError,"DUPLICATE_PRIOR_HUMAN"):
            v.native_speakers(config)

    def test_untouched_normal_and_nearphone_are_not_negative(self):
        rows=v.planned(v.native_speakers(fake_mapping()))
        self.assertEqual(Counter(r["requested_class"] for r in rows),
                         {2:16,0:8,None:8})
        self.assertEqual({r["role"] for r in rows},{"development"})
        self.assertTrue(all(r["observed_class"] is None for r in rows))
        self.assertTrue(all(r["training_allowed"] is False for r in rows))
        self.assertTrue(all(r["label_basis"]=="SOURCE_REQUEST_NOT_ACOUSTIC_TRUTH"
                            for r in rows))
        self.assertTrue(all(r["id"].startswith("merodev8-")
                            for r in rows))

    def test_no_pr_or_replay_first_main_only(self):
        event={"before":v.MAIN_BEFORE,"after":"a"*40,
               "ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"991"}
        self.assertEqual(v.guard(env=env,event=event,parent=v.MAIN_BEFORE)["attempt"],1)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.MAIN_BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.MAIN_BEFORE)

if __name__=="__main__":
    main()
