"""Stdlib-only source, provenance, split and non-promotion regressions."""
from __future__ import annotations
from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "en_v5_student_source",HERE/"english_v5_mero_sequence_margin_student_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def observations():
    out=[]
    for role,n in (("train",10),("development",5)):
        for k in range(n):
            for label in (0,1):
                out.append({"id":f"{role}_{k}_{label}",
                    "role":role,"source_group_id":f"{role}_{k}",
                    "mono_class":label,"source_pcm_sha256":f"{len(out)+1:064x}"})
    return out


class MeroStudentContract(unittest.TestCase):
    def test_only_one_selected_keyword_and_immutable_teacher(self):
        c=m.source_check()
        self.assertEqual(c["positive_ctc"],[1,2,3])
        self.assertEqual(c["only_keyword"],"Hey Mero")
        self.assertEqual(c["teacher_run"],44461057432)
        self.assertEqual(c["teacher_artifact"],14500132512)
        self.assertEqual(c["teacher_time_alignment"],
                         "UNAVAILABLE_NO_PHONEME_TIMESTAMPS")
        self.assertEqual(c["new_teacher_inference"],0)
        self.assertEqual(c["new_tts_asr_calls"],0)
        self.assertEqual(c["new_optimizer_updates"],0)
        self.assertFalse(c["promotion_allowed"])
        self.assertFalse(c["calibration_locked_opened"])
        self.assertEqual(m.SEED,20261120)
        self.assertEqual(m.UPDATES,512)
        self.assertEqual(m.PHRASE,[1,2,3])
        self.assertEqual(m.TEACHER_MODEL_SHA,
                         "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6")

    def test_public_same_repo_pr_guard_and_retries_denied(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"pull_request","GITHUB_HEAD_REF":m.BRANCH,
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"44470000100"}
        event={"repository":{"full_name":m.REPO,"private":False},
            "action":"opened","pull_request":{
                "head":{"ref":m.BRANCH,"sha":"a"*40,
                        "repo":{"full_name":m.REPO}},
                "base":{"ref":"main"}}}
        self.assertEqual(m.guard(env,event)["head_sha"],"a"*40)
        for mutation in (
            ({**env,"GITHUB_RUN_ATTEMPT":"2"},event),
            ({**env,"GITHUB_REPOSITORY":"bad"},event),
            (env,{**event,"action":"closed"}),
            (env,{**event,"repository":{"full_name":m.REPO,"private":True}}),
            (env,{**event,"pull_request":{**event["pull_request"],
                  "base":{"ref":"not-main"}}}),
        ):
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):m.guard(*mutation)

    def test_groups_are_independent_for_each_target(self):
        grouped=m.source_round_robin([r for r in observations()
                                       if r["role"]=="train"])
        self.assertEqual(len(grouped[0]),10)
        self.assertEqual(len(grouped[1]),10)
        damaged=[r for r in observations() if r["role"]=="train"
                 and r["source_group_id"] not in ("train_8","train_9",
                                                  "train_7","train_6","train_5")]
        with self.assertRaises(ValueError):m.source_round_robin(damaged)

    def test_teacher_receipt_rejects_unverified_or_modified_zip(self):
        rows=[r for r in observations() if r["role"]=="train"]
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder)/"fake-archive.zip"
            src.write_bytes(b"fake")
            with self.assertRaises(ValueError):
                m.verified_teacher_train(src,rows)

    def test_result_is_research_only_and_old_weights_not_reused(self):
        body=(HERE/"english_v5_mero_sequence_margin_student_once.py").read_text()
        self.assertIn("F.ctc_loss(",body)
        self.assertIn("torch.optim.AdamW",body)
        self.assertIn("v5.ctc_greedy(logits,torch)",body)
        self.assertIn("teacher_parameters_transferred",body)
        self.assertIn('"promotion_to_locked_or_arm_or_product":False',body)
        self.assertIn('"previously_exposed":True',body)
        self.assertIn('"old_other_keyword_gradients":0',body)
        self.assertIn('"dev_teacher_events_used":False',body)
        self.assertNotIn("Hey Vosen",body)
        self.assertNotIn("ctc_phone_alignment",body)
        self.assertNotIn("saved_mono_model",body)
        self.assertNotIn("for step in range(240)",body)


if __name__=="__main__":
    unittest.main()
