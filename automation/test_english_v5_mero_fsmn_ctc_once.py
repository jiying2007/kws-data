"""Pure standard-library V5 tests, no audio generation, no model calls."""
from __future__ import annotations
from collections import Counter
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "english_v5_mono_test",HERE/"english_v5_mero_fsmn_ctc_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def toy_rows():
    result=[]
    for role,n in (("train",6),("development",2)):
        for index in range(n):
            family="newtrain24" if index==0 else "piper24"
            group=f"{role}_{index}"
            for item,label in enumerate((0,2,1)):
                k=len(result)+1
                result.append({
                    "id":f"r{k}","corpus":family,"weak_class":label,
                    "requested_class":label,
                    "role":role,"source_group_id":group,
                    "source_pcm_sha256":f"{k:064x}",
                    "tier":"silver" if item%2==0 else "weak",
                    "human_gold":False,
                    "training_allowed_in_original":False,
                    "samples":16000,
                })
    return result


class SingleMeroOnlyContract(unittest.TestCase):
    def test_exact_target_and_single_keyword_model_is_frozen(self):
        c=v.config()
        x=v.source_check()
        self.assertEqual(c["active_keyword_count"],1)
        self.assertEqual(c["target"]["text"],"Hey Mero")
        self.assertEqual(c["target"]["ipa"],"/heɪ ˈmɛroʊ/")
        self.assertEqual(c["vocabulary"],{"0":"<blank>","1":"HEY",
                                        "2":"MEH","3":"ROH"})
        self.assertEqual(x["active_target_ids"],["hey_mero"])
        self.assertEqual(x["positive_ctc"],[1,2,3])
        self.assertEqual(x["negative_ctc"],[])
        self.assertEqual(c["model"]["hidden"],96)
        self.assertEqual(c["model"]["left_memory"],4)
        self.assertEqual(c["model"]["max_updates"],240)
        self.assertEqual(c["model"]["center"],False)
        self.assertTrue(c["completion"]["fresh_locked_test_required"])
        self.assertFalse(x["ssc305_measured"])

    def test_failed_real_model_cannot_be_promoted(self):
        c=v.config()
        receipt=c["last_real_training_result"]
        self.assertEqual(receipt["status"],
            "REJECTED_RESEARCH_CANDIDATE_DEV_QUALITY_FAILED")
        self.assertEqual(receipt["actual_training_run_id"],44453447586)
        self.assertEqual(receipt["trained_parameters"],43876)
        self.assertEqual(receipt["optimizer_updates"],240)
        self.assertEqual(receipt["model_artifact_id"],14496732088)
        self.assertEqual(receipt["dev"]["positive_misses"],6)
        self.assertEqual(receipt["dev"]["positive_original_wavs"],6)
        self.assertFalse(receipt["full_sequence_ctc_diagnostic"][
            "dev_margins_strictly_separate"])
        self.assertEqual(receipt["full_sequence_ctc_diagnostic"]["run_id"],
                         44456534389)
        self.assertFalse(receipt["promotion_allowed"])
        self.assertEqual(receipt["physical_ssc305"],"NOT_RUN")
        self.assertEqual(c["active_keyword_count"],1)

    def test_old_other_keyword_never_becomes_negative_or_positive(self):
        original=toy_rows()
        selected,summary=v.reviewed_roles(original)
        self.assertEqual(len(original),24)
        self.assertEqual(len(selected),16)
        self.assertEqual(summary["excluded_historical_other_target"],8)
        self.assertEqual(summary["qualified_groups"],
                         {"train":6,"development":2})
        self.assertEqual(summary["qualified_new_groups"],
                         {"train":1,"development":1})
        self.assertEqual(Counter(r["mono_class"] for r in selected),
                         Counter({0:8,1:8}))
        self.assertEqual(Counter(tuple(r["ctc_targets"]) for r in selected),
                         Counter({():8,(1,2,3):8}))
        self.assertTrue(all(r["weak_class"] in (0,2) for r in selected))
        self.assertFalse(summary["old_calibration_or_locked_audio_used"])

    def test_role_mixing_is_forbidden(self):
        original=toy_rows()
        original[1]["role"]="development"
        with self.assertRaises(ValueError):
            v.reviewed_roles(original)

    def test_no_unreviewed_or_old_locked_class_enters(self):
        for mod,value in (("tier","unknown"),("role","locked-test"),
                          ("human_gold",True),("weak_class",None)):
            original=toy_rows()
            original[2][mod]=value
            with self.subTest(field=mod):
                with self.assertRaises(ValueError):
                    v.reviewed_roles(original)

    def test_new_source_truly_required(self):
        original=toy_rows()
        for row in original:
            row["corpus"]="piper24"
        with self.assertRaises(ValueError):
            v.reviewed_roles(original)

    def test_first_protected_main_only_and_no_retry(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"444500001",
             "GITHUB_SHA":"a"*40}
        event={"before":v.BEFORE,"after":"a"*40,
               "ref":"refs/heads/main","forced":False,"deleted":False,
               "repository":{"full_name":v.REPO,"private":False}}
        self.assertEqual(v.guard(env,event,v.BEFORE)["parent_sha"],v.BEFORE)
        with self.assertRaises(ValueError):
            v.guard({**env,"GITHUB_RUN_ATTEMPT":"2"},event,v.BEFORE)
        with self.assertRaises(ValueError):
            v.guard(env,{**event,"before":"b"*40},v.BEFORE)
        with self.assertRaises(ValueError):
            v.guard(env,{**event,"repository":
                  {"full_name":v.REPO,"private":True}},v.BEFORE)
        with self.assertRaises(ValueError):
            v.guard(env,event,"b"*40)

    def test_source_writes_none_and_no_old_kw_event_api(self):
        raw=(HERE/"english_v5_mero_fsmn_ctc_once.py").read_text()
        self.assertNotIn("Hey Vosen",raw)
        self.assertNotIn("Vozen",raw)
        self.assertNotIn("VOSEN",v.config()["vocabulary"].values())
        self.assertIn("torch.nn.functional.ctc_loss",raw)
        self.assertIn("torch.stft",raw)
        self.assertIn("center=False",raw)
        self.assertIn("model_frozen_before_dev",raw)


if __name__=="__main__":
    unittest.main()
