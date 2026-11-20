"""New original Piper prospective_dev8 source-only contract, no models on PR."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import importlib.util
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
   "v6_b6_prospective_dev512_test",HERE/"english_v6_b6_fresh_dev512_source_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def fake_mapping():
    previous_ids=(set(v.previous.old.OLD_LIBRI_IDS)
                  |set(v.previous.old.OLD_PIPER_IDS)
                  |{speaker for _,speaker in v.previous.old.NEW_VOICES})
    start=max(previous_ids)+1000
    return {"num_speakers":904,"speaker_id_map":{
            "p"+str(start+i):i for i in range(904)}}

class NewProspectiveDev8PiperSourceTests(TestCase):
    def test_exact_frozen_speaker_and_new_budget_contract(self):
        plan=v.plan()
        check=v.source_check()
        self.assertEqual(check["new_speaker_indices"],[512,519])
        self.assertEqual(check["new_speech_wav_budget"],32)
        self.assertEqual(check["source_roles"],{"development":8})
        self.assertEqual(check["new_optimizer_updates"],0)
        self.assertEqual(check["new_asr_phone_calls"],0)
        self.assertEqual(check["only_english_keyword"],"Hey Mero")
        self.assertFalse(check["nearphone_is_eligible_ordinary"])
        self.assertFalse(check["product_promotion"])
        self.assertEqual(plan["source_count"],32)
        self.assertEqual(plan["b6_frozen_models"]["real_train_actions_run"],44704535156)
        self.assertEqual(plan["b6_frozen_models"]["real_train_artifact_id"],14615025178)
        self.assertEqual(plan["b6_frozen_models"]["real_train_artifact_zip_sha256"],
                         "1c310b0d4945f7aa40f2576f231c94eb48b33316a6c4f79c20215b1ac80bab8d")
        self.assertEqual(plan["b6_frozen_models"]["B6_control_pt_sha256"],
                         "c9cac4c16a9d574b6bc9baef14ac239517a2ed9bfbe2e3ff2fb69972587c9430")
        self.assertEqual(plan["b6_frozen_models"]["B6_consistency_pt_sha256"],
                         "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04")
        self.assertTrue(plan["b6_frozen_models"]["B4_half_exact_tensor_replay"])
        self.assertFalse(plan["b6_frozen_models"]["release_allowed"])
        self.assertTrue(plan["b6_frozen_models"]["prohibit_new_DEV_training_or_selection_before_source_QA"])
        self.assertIn("400..511",plan["source_guards"]["known_prior_piper_indices"])

        self.assertEqual(plan["frozen_reference"]["robust_pt_sha256"],"20d0c4632224cf332460493695c0dd21927a80ed13034a9531f5d5230d2f7e05")
        self.assertEqual(plan["frozen_reference"]["prior_both_models_positive_hits"],2)
        self.assertEqual([x["speaker_index"] for x in plan["source_groups"]],
                         list(range(512,520)))
        self.assertEqual(Counter(x["role"] for x in plan["source_groups"]),
                         {"development":8})
        self.assertEqual([x["requested_class"] for x in plan["cells"]],
                         [2,2,0,None])

    def test_new_model_source_ancestry_truly_disjoint(self):
        original=fake_mapping()
        chosen=v.native_speakers(original)
        self.assertEqual(set(chosen),set(range(512,520)))
        self.assertEqual(len(set(chosen.values())),8)
        ids={int(label[1:]) for label in chosen.values()}
        old_ids={int(label[1:]) for label,index in original["speaker_id_map"].items()
                 if index in (set(range(58))|set(range(400,512)))}
        self.assertFalse(ids & old_ids)
        self.assertEqual(len(v.planned(chosen)),32)

    def test_old_human_identity_or_duplicate_is_a_hard_stop(self):
        config=fake_mapping()
        pairs=config["speaker_id_map"]
        chosen=next(label for label,index in pairs.items() if index==512)
        del pairs[chosen]
        forbidden=next(iter(v.previous.old.OLD_LIBRI_IDS))
        pairs["p"+str(forbidden)]=512
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
        self.assertTrue(all(r["id"].startswith("merov6b6dev8-")
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
