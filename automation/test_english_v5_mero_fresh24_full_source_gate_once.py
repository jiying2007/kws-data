"""Pure-source regression for the exact source96 machine 6/2/2/2 gate."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import importlib.util
from unittest import TestCase,main

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v5_full96_gate_tests",HERE/"english_v5_mero_fresh24_full_source_gate_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def row(g,role,cell,kind,tier):
    return {"source_group_id":g,"role":role,"cell_id":cell,
            "requested_class":kind,
            "observed_class":kind if tier in ("silver","weak") else None,
            "tier":tier}

def source_group(g,role,accepted=True):
    return [row(g,role,"mero-fast",2,"silver" if accepted else "unknown"),
            row(g,role,"mero-slow",2,"unknown"),
            row(g,role,"ordinary-time",0,"silver"),
            row(g,role,"nearphone-meero-challenge",None,"unknown")]

class V5FullSourceQualityGateTests(TestCase):
    def test_exact_original_saved_six_artifacts_and_zero_models(self):
        z=v.source_check()
        self.assertEqual(z["status"],"PINNED_SIX_SOURCE_SCREEN_ARCHIVES")
        self.assertEqual(z["old_64_machine_screen_artifact"],14509252921)
        self.assertEqual(z["new_sealed_source_run"],44485062398)
        self.assertEqual(z["sealed_source_count"],32)
        self.assertEqual(z["new_native_forward_calls"],0)
        self.assertEqual(z["kws_forward_calls"],0)
        self.assertEqual(z["new_optimizer_steps"],0)
        self.assertFalse(z["locked_test_kws_opened"])
        self.assertEqual(v.MIN,{"train":6,"development":2,
                                "calibration":2,"locked-test":2})
        self.assertEqual(set(v.ARCHIVES),set(v.sealed.PROVIDERS))
        self.assertEqual(len({x[0] for x in v.ARCHIVES.values()}),5)

    def test_exact_four_four_frozen_machine_eligibility(self):
        dataset=[]
        for i in range(4):
            dataset+=source_group(f"c{i}","calibration",accepted=(i<2))
            dataset+=source_group(f"l{i}","locked-test",accepted=(i<2))
        counts,qual=v.qualified_by_role(dataset)
        self.assertEqual(counts,{"calibration":2,"locked-test":2})
        self.assertEqual(len(qual),4)
        self.assertNotIn("train",counts)
        self.assertNotIn("development",counts)
        self.assertTrue(all(n>=v.MIN[k] for k,n in counts.items()))

    def test_failing_two_source_roles_never_pass_or_relabel(self):
        a=[]
        for i in range(4):
            a+=source_group(f"c{i}","calibration",accepted=(i<1))
            a+=source_group(f"l{i}","locked-test",accepted=(i<2))
        count,_=v.qualified_by_role(a)
        full={"train":11,"development":2,**count}
        self.assertFalse(all(full[k]>=n for k,n in v.MIN.items()))
        self.assertEqual(count["calibration"],1)
        self.assertEqual(count["locked-test"],2)

    def test_no_nearphone_or_unknown_negative_label(self):
        proof=v.policy.decide({"requested_class":None,
                "pcm_samples":12000},{},{})
        self.assertEqual(proof["tier"],"unknown")
        self.assertIsNone(proof["class"])
        self.assertFalse(proof["training_allowed"])
        x=source_group("g","locked-test")
        x[3]["tier"]="silver";x[3]["observed_class"]=0
        count,_=v.qualified_by_role(x)
        self.assertEqual(count,{"calibration":0,"locked-test":1})
        # A claimed 0 on nearphone cannot be consumed as extra source class
        # because machine_decisions uses frozen policy.decide(None) -> UNKNOWN.

    def test_source_group_cannot_move_to_train_or_mix_roles(self):
        x=source_group("g","calibration")
        x[1]["role"]="train"
        with self.assertRaisesRegex(ValueError,"EXACT_4_4_SEALED_GROUP_ROLE"):
            v.qualified_by_role(x)

    def test_first_main_guard_passes_and_retries_stop(self):
        ev={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
            "repository":{"full_name":v.REPO,"private":False},
            "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"918"}
        good=v.guard(env=env,event=ev,parent=v.BEFORE)
        self.assertEqual(good["new_model_calls"],0)
        self.assertTrue(good["no_locked_kws"])
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=ev,parent=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=ev,parent=v.BEFORE)

if __name__=="__main__":
    main()
