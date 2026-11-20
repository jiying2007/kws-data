"""V6 B3 CTC gap/token-source: standard-lib source-only regression."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v6_b3_ctc_source",HERE/"english_v6_b3_ctc_gap_origin_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestV6B3GapSource(unittest.TestCase):
    def test_frozen_source_and_do_not_promotion(self):
        s=m.source_check()
        self.assertEqual(s["status"],"SOURCE_ONLY_NO_REAL_ACOUSTIC_FORWARDS")
        self.assertEqual(s["model_artifact"],14590093798)
        self.assertEqual(s["source_previous_gain_run"],44662162126)
        self.assertEqual(s["original_source_count"],14)
        self.assertEqual(s["positive_originals"],6)
        self.assertEqual(s["ordinary_originals"],8)
        self.assertEqual(s["unknown_excluded"],18)
        self.assertEqual(s["replay_gaps_samples"],[8000,16000,32000,1000000])
        self.assertEqual(s["cases"],["mixed","reverse"])
        self.assertFalse(s["source_bytes_changed"])
        self.assertFalse(s["new_training"])
        self.assertFalse(s["new_tts_asr"])
        self.assertFalse(s["cal_locked_read"])
        self.assertFalse(s["model_selection_allowed"])
        self.assertEqual(s["physical_ssc305"],"NOT_RUN")
        self.assertFalse(s["product_promotion"])
        self.assertEqual(m.BEFORE,"db0c4b699b16fcd765b430702a8f786715529eb5")

    def test_token_anchors_are_from_same_original_or_cross(self):
        segments=[
           {"id":"ordinary-A","start":100,"end":400},
           {"id":"mero-B","start":16100,"end":16500}
        ]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"events.csv"
            path.write_text("16300,200,300\n16400,16110,16150\n")
            events=m.parse_events(path,segments)
            self.assertEqual(len(events),2)
            self.assertEqual(events[0]["hey_source"],"ordinary-A")
            self.assertEqual(events[0]["meh_source"],"ordinary-A")
            self.assertEqual(events[0]["event_source"],"mero-B")
            self.assertFalse(events[0]["tokens_from_same_original"])
            self.assertTrue(events[0]["cross_or_unattributed_token_source"])
            self.assertTrue(events[1]["tokens_from_same_original"])
            self.assertFalse(events[1]["cross_or_unattributed_token_source"])
            self.assertEqual(events[1]["hey_to_meh_samples"],40)
            self.assertEqual(events[1]["meh_to_roh_samples"],250)

    def test_ambiguous_bounded_source_rejected(self):
        a={"id":"A","start":0,"end":100}
        b={"id":"B","start":50,"end":150}
        with self.assertRaisesRegex(ValueError,"AMBIGUOUS_SEGMENT"):
            m.locate_sample([a,b],60)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"events.csv"
            path.write_text("500,600,700\n")
            with self.assertRaisesRegex(ValueError,"CHRONOLOGICAL"):
                m.parse_events(path,[])
            path.write_text("foo,bar,baz\n")
            with self.assertRaisesRegex(ValueError,"EXACT_THREE"):
                m.parse_events(path,[])

    def test_expected_previous_exact_events_not_synthetic_fresh_eval(self):
        self.assertEqual(m.BASELINE_CLOCKS["mixed"],
                         [112352,168832,223552,280832,339712])
        self.assertEqual(m.BASELINE_CLOCKS["reverse"],
                         [83712,141792,199392,254272,311552,368512])
        self.assertEqual(m.PREVIOUS_ARTIFACT,14595464491)
        self.assertEqual(m.PREVIOUS_SHA,
                         "0ba7cd287afcedbbd7625b6987cc89dbeb2c8717c1506b905e8a01671e9ea3cb")
        self.assertTrue(m.REPLAY_C.is_file())

    def test_no_pr_context_replay(self):
        with patch.dict("os.environ",{
            "GITHUB_ACTIONS":"true",
            "GITHUB_REPOSITORY":"jiying2007/kws-data",
            "GITHUB_EVENT_NAME":"pull_request",
            "GITHUB_REF":"refs/pull/280/merge",
            "GITHUB_RUN_ATTEMPT":"1"},clear=True):
            with self.assertRaisesRegex(ValueError,"FIRST_MAIN_NO_REPLAY_ONLY"):
                m.guard()
        with self.assertRaisesRegex(ValueError,"GITHUB_READONLY_TOKEN_REQUIRED"):
            m.remote_proof("")

if __name__=="__main__":
    unittest.main()
