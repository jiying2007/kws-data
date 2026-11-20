"""B3 continuous-state experiment source-only checks; no model forwards on PR."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b3_continuous",HERE/"english_v6_b3_continuous_c11_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def fixture():
    raw=b"\x01\x00"*400
    rows=[]
    for i in range(8):
        rows.append({"id":f"ordinary{i}","mono_class":0,"role":"development",
                     "source_group_id":f"piper-test-{i}",
                     "source_pcm_sha256":hashlib.sha256(raw).hexdigest()})
    for i in range(6):
        rows.append({"id":f"mero{i}","mono_class":1,"role":"development",
                     "source_group_id":f"piper-test-{i}",
                     "source_pcm_sha256":hashlib.sha256(raw).hexdigest()})
    return rows,raw

class TestSourceOnly(unittest.TestCase):
    def test_parent_b3_identity_and_nonpromotion(self):
        result=m.source_check()
        self.assertEqual(result["status"],"SOURCE_ONLY_NO_AUDIO_FORWARD")
        self.assertEqual(result["B3_artifact"],14590093798)
        self.assertEqual(result["prior_software_parity_run"],44653612830)
        self.assertEqual(result["source_positive"],6)
        self.assertEqual(result["source_ordinary"],8)
        self.assertEqual(result["unknown_excluded"],18)
        self.assertEqual(result["far_per_hour"],"NOT_MEASURED")
        self.assertEqual(result["ssc305_board"],"NOT_RUN")
        self.assertTrue(result["one_stream_without_reset"])
        self.assertTrue(result["no_training"])
        self.assertTrue(result["no_tts_asr"])
        self.assertFalse(result["independent_development_quality"])
        self.assertFalse(result["model_selection_allowed"])
        self.assertEqual(m.BEFORE,"acbd6b2bab42374d2125e93537e040bc32b9c1f7")

    def test_four_predeclared_case_orders_and_original_byte_hash(self):
        rows,raw=fixture()
        scenarios=m.select_rows(rows)
        self.assertEqual(set(scenarios),set(m.CASES))
        self.assertEqual(len(scenarios["mixed"]),14)
        self.assertEqual(len(scenarios["reverse"]),14)
        self.assertEqual(len(scenarios["ordinary_only"]),8)
        self.assertFalse(scenarios["silence_5s"])
        self.assertEqual(scenarios["reverse"],list(reversed(scenarios["mixed"])))
        for case in m.CASES:
            combined,segments=m.assemble(rows,lambda r: raw,case)
            self.assertEqual(len(combined)%2,0)
            self.assertTrue(combined.startswith(b"\0"*32000))
            self.assertEqual(len(segments),len(scenarios[case]))
            self.assertLessEqual(len(combined),2*16000*120)
        badraw=b"\x03\x00"*400
        with self.assertRaisesRegex(ValueError,"ORIGINAL_PCM_SHA"):
            m.assemble(rows,lambda r: badraw,"mixed")

    def test_unknown_duplicate_or_source_role_rejected(self):
        rows,_=fixture()
        with self.assertRaisesRegex(ValueError,"ONLY_14"):
            m.select_rows(rows+[rows[0]])
        rows[0]["mono_class"]=None
        with self.assertRaisesRegex(ValueError,"ONLY_14"):
            m.select_rows(rows)
        rows,_=fixture()
        rows[0]["role"]="locked-test"
        with self.assertRaisesRegex(ValueError,"ONLY_14"):
            m.select_rows(rows)

    def test_event_clock_classification_not_a_word_end_claim(self):
        rows,raw=fixture()
        _,segments=m.assemble(rows,lambda r: raw,"ordinary_only")
        self.assertEqual(len(segments),8)
        self.assertEqual(segments[0]["start"],16000)
        self.assertEqual(segments[0]["end"],16400)
        self.assertEqual(segments[0]["event_window_end"],20400)
        aligned=m.classify([1000,16000,16410,21000],[segments[0]])
        self.assertEqual([x["segment_class"] for x in aligned],[None,0,0,None])
        self.assertEqual(aligned[1]["offset_from_segment_start_samples"],0)
        with self.assertRaisesRegex(ValueError,"NON_OVERLAPPING"):
            m.classify([16050],[segments[0],segments[0]])

    def test_guard_prohibits_pr_replay_without_network(self):
        with patch.dict("os.environ",{"GITHUB_ACTIONS":"true",
                                     "GITHUB_REPOSITORY":"jiying2007/kws-data",
                                     "GITHUB_EVENT_NAME":"pull_request",
                                     "GITHUB_REF":"refs/pull/278/merge",
                                     "GITHUB_RUN_ATTEMPT":"1"},clear=True):
            with self.assertRaisesRegex(ValueError,"FIRST_PROTECTED"):
                m.guard()

if __name__=="__main__":
    unittest.main()
