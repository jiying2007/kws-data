"""V6 B3 C11/ARM numerical bridge: strict source-only/no-inference tests."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b3_num_parity_module",HERE/"english_v6_b3_c11_arm_parity_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class SourceOnly(unittest.TestCase):
    def test_immutable_real_frozen_b3_and_no_quality_claim(self):
        result=m.source_check()
        self.assertEqual(result["status"],"SOURCE_ONLY_NO_ACOUSTIC_FORWARDS")
        self.assertEqual(result["model_artifact"],14590093798)
        self.assertEqual(result["model_parameters"],43876)
        self.assertEqual(result["original_reviewed_dev_positive"],6)
        self.assertEqual(result["original_reviewed_dev_ordinary"],8)
        self.assertEqual(result["unknown_excluded"],18)
        self.assertEqual(result["expected_native_and_arm_conditions"],
                         ["original","one_second_zero_prefix"])
        self.assertEqual(result["max_logit_tolerance"],0.005)
        self.assertFalse(result["product_promotion"])
        self.assertFalse(result["model_selection_allowed"])
        self.assertFalse(result["new_training"])
        self.assertFalse(result["locked_calibration_opened"])
        self.assertEqual(m.BEFORE,"0ff1255055277b82c523addd3411b6152ace9da6")
        self.assertEqual(m.B3["stem"],"mero43-onset256")
        self.assertEqual(m.B3["name"],"english-v6-mero43-suffixctc256-r4-44651053897")
        self.assertEqual(m.B3["bytes"],503345)
        self.assertEqual(m.B3["sha"],
            "59373f0e7e5f9d4ed35d6b116cdd6f8ac8eb5abcfa1acb5311ef658c59a20e04")
        self.assertEqual(m.B3["pt"],
            "682e2689e82aa91d850516c06b2be61efc64731cb1dee3b70a7d2a8e01925a46")
        self.assertEqual(m.B3["npz"],
            "cf3022ea7dd29e7e830b73ce613485a18d1d300467605cb330447cdce6e3ee89")
        self.assertEqual(m.B3["onnx"],
            "5ca791d495f6c57a3608bc412effca15924e9e32e088d45d6eb96e8f97092031")

    def test_fail_closed_without_exact_public_main(self):
        with patch.dict(os.environ,{"GITHUB_EVENT_NAME":"pull_request"}):
            with self.assertRaises(ValueError):m.guard()
        with self.assertRaises(ValueError):m.provenance("")
        with self.assertRaises(ValueError):m.provenance(None)

    def test_exact_new_main_synthetic_guard_and_moved_parent(self):
        after="a"*40
        event={"before":m.BEFORE,"after":after,"ref":"refs/heads/main",
               "repository":{"full_name":m.REPO,"private":False},
               "forced":False,"deleted":False}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"event.json"
            path.write_text(json.dumps(event))
            env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
                 "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
                 "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":after,
                 "GITHUB_RUN_ID":"12345","GITHUB_EVENT_PATH":str(path)}
            with patch.dict(os.environ,env),patch.object(
                    m.subprocess,"check_output",return_value=m.BEFORE+"\n"):
                x=m.guard()
                self.assertEqual(x["main_sha"],after)
            event["before"]="b"*40
            path.write_text(json.dumps(event))
            with patch.dict(os.environ,env),patch.object(
                    m.subprocess,"check_output",return_value=m.BEFORE+"\n"):
                with self.assertRaises(ValueError):m.guard()

    def test_invented_event_clock_is_strict(self):
        with tempfile.TemporaryDirectory() as td:
            file=Path(td)/"clock.csv"
            file.write_text("0,512,0\n1,672,1\n2,832,0\n",encoding="ascii")
            self.assertEqual(m.c_events(file,3),[672])
            file.write_text("0,512,0\n1,671,1\n2,832,0\n",encoding="ascii")
            with self.assertRaises(ValueError):m.c_events(file,3)

    def test_missing_byte_bound_archive_cannot_be_assumed_qualified(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises((FileNotFoundError,ValueError)):
                m.audit(root/"missing",root/"missing-screen",root/"missing-b3.zip")
        self.assertEqual(m.COMPILERS,("gcc","clang","cortex-a32-qemu"))
        self.assertEqual(len(m.CONDITIONS)*m.ORIGINAL_COUNT,28)

if __name__=="__main__":
    unittest.main()
