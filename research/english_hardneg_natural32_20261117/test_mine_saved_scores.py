"""Invented PCM/probability fixtures only; no learned model or real audio."""
from __future__ import annotations

import copy
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mine_saved_scores as mining


MODEL = mining.digest(b"invented model identity; no model exists")
CONFIG = mining.digest(b"invented fixed configuration")


def fixture(sources=1, frames=160000):
    roster = {"coverage_head_binary_sha256": MODEL, "rows": []}
    pcm, scores, traces = {}, {}, {}
    for i in range(sources):
        rid = "test-source-%02d" % i
        raw = struct.pack("<h", i + 1) * frames
        row = {
            "recording_id": rid, "source_member": "invented/" + rid + ".flac",
            "speaker_id": str(i), "source_family_id": "invented-speaker-" + str(i),
            "source_role": "train", "old_bg48_role": "train", "current_coverage_role": "train",
            "sample_rate": 16000, "channels": 1, "sample_width": 2, "frames": frames,
            "pcm_sha256": mining.digest(raw), "label_status": "UNKNOWN_REVIEW_REQUIRED",
            "expected_events": None, **mining.FALSE_FLAGS,
        }
        ready = [{"sample": sample, "probabilities": [.8, .1, .1], "keyword": 0}
                 for sample in range(mining.FIRST_READY, frames + 1, mining.HOP)]
        row["expected_ready_rows"] = len(ready)
        roster["rows"].append(row)
        pcm[rid], scores[rid] = raw, ready
        traces[rid] = mining.digest(("invented-raw-C-trace-" + rid).encode())
    evidence = {"trace_sha256_by_id": traces, "synthetic_test_only": True}
    return roster, pcm, scores, evidence


def run(inputs, **kwargs):
    roster, pcm, scores, evidence = inputs
    return mining.build_ledger(roster, pcm, scores, model_sha256=kwargs.get("model_sha256", MODEL),
                               config_sha256=kwargs.get("config_sha256", CONFIG), source_evidence=evidence)


def set_score(inputs, index, probability, keyword=1, event=0, source=0):
    row = inputs[2]["test-source-%02d" % source][index]
    other = 2 if keyword == 1 else 1
    row["probabilities"] = [1 - probability - .01, .01, .01]
    row["probabilities"][keyword] = probability
    row["probabilities"][other] = .01
    row["keyword"] = event


class MiningTests(unittest.TestCase):
    def test_empty_queue_keeps_every_source_and_both_peaks(self):
        result = run(fixture(sources=3))
        self.assertEqual(result["status"], "NO_CANDIDATE_IN_SAVED_OBSERVATION")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["counts"]["selected_candidates"], 0)
        self.assertEqual(len(result["source_summaries"]), 3)
        for source in result["source_summaries"]:
            self.assertEqual(source["source_utterance_continuity"], "natural_individual")
            for k in ("1", "2"):
                self.assertEqual(source["by_keyword"][k]["peak_probability"], .1)
                self.assertEqual(source["by_keyword"][k]["peak_sample"], 35200)
            self.assertEqual(source["review"], "UNREVIEWED")
        self.assertFalse(result["training_tsv_emitted"])
        self.assertEqual(result["verified_negative_seconds"], 0)

    def test_exact_half_threshold_and_event_or_condition(self):
        inputs = fixture()
        set_score(inputs, 5, .499999)
        self.assertFalse(run(inputs)["candidates"])
        set_score(inputs, 5, .5)
        self.assertEqual(len(run(inputs)["candidates"]), 1)
        # The caller authenticates emitted events from its raw trace. Mining
        # records an event even if its score is below the proposal threshold.
        set_score(inputs, 5, .4, event=1)
        row = run(inputs)["candidates"][0]
        self.assertEqual(row["triggering_keywords"], [1])
        self.assertEqual(row["event_count"], 1)
        self.assertEqual(row["by_keyword"]["1"]["proposal_threshold_samples"], [])

    def test_cross_keyword_overlap_is_one_pcm_window_with_separate_evidence(self):
        inputs = fixture()
        set_score(inputs, 0, .8, keyword=1)
        set_score(inputs, 2, .9, keyword=2, event=2)
        result = run(inputs)
        self.assertEqual(len(result["candidates"]), 1)
        row = result["candidates"][0]
        self.assertEqual(row["triggering_keywords"], [1, 2])
        self.assertEqual(row["start_sample"], 0)
        self.assertEqual(row["end_sample_exclusive"], 35200 + 2 * 1280 + 16000)
        self.assertEqual(row["by_keyword"]["1"]["peak_probability"], .8)
        self.assertEqual(row["by_keyword"]["2"]["peak_probability"], .9)
        self.assertEqual(row["by_keyword"]["2"]["event_samples"], [37760])
        self.assertTrue(row["pre_context_clipped_at_source_start"])

    def test_overlap_merging_is_transitive(self):
        inputs = fixture()
        for index in (0, 25, 50):
            set_score(inputs, index, .6)
        result = run(inputs)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(len(result["candidates"][0]["seeds"]), 3)
        self.assertEqual(result["candidates"][0]["end_sample_exclusive"], 115200)

    def test_source_end_clips_real_context_without_padding(self):
        inputs = fixture()
        set_score(inputs, -1, .75, keyword=2)
        row = run(inputs)["candidates"][0]
        last = inputs[2]["test-source-00"][-1]["sample"]
        self.assertEqual(row["start_sample"], last - 38400)
        self.assertEqual(row["end_sample_exclusive"], 160000)
        self.assertTrue(row["post_context_clipped_at_source_end"])
        self.assertEqual(row["window_pcm_sha256"], mining.digest(inputs[1]["test-source-00"][2 * row["start_sample"]:320000]))
        self.assertEqual(row["window_samples"], 160000 - row["start_sample"])

    def test_per_source_cap_events_first_then_probability(self):
        inputs = fixture(frames=240000)
        for index, p in ((0, .8), (45, .6), (90, .7), (135, .95)):
            set_score(inputs, index, p, event=1 if index == 0 else 0)
        result = run(inputs)
        self.assertEqual(result["counts"]["merged_candidates"], 4)
        self.assertEqual(result["counts"]["selected_candidates"], 2)
        self.assertEqual(result["counts"]["excluded_by_per_source_cap"], 2)
        self.assertEqual(result["candidates"][0]["event_count"], 1)
        self.assertEqual(result["candidates"][1]["maximum_keyword_probability"], .95)
        self.assertEqual(result["source_summaries"][0]["selected_candidate_windows"], 2)

    def test_global_cap_and_source_order_are_deterministic(self):
        inputs = fixture(sources=17, frames=240000)
        for source in range(17):
            set_score(inputs, 0, .6 + source * .001, source=source)
            set_score(inputs, 90, .8 + source * .001, source=source)
        result = run(inputs)
        self.assertEqual(result["counts"]["merged_candidates"], 34)
        self.assertEqual(result["counts"]["selected_candidates"], 32)
        self.assertEqual(result["counts"]["excluded_by_global_cap"], 2)
        self.assertEqual({c["recording_id"] for c in result["excluded_candidates"]}, {"test-source-00", "test-source-01"})
        reordered = copy.deepcopy(inputs)
        reordered[0]["rows"].reverse()
        result2 = run(reordered)
        self.assertEqual([c["candidate_id"] for c in result["candidates"]],
                         [c["candidate_id"] for c in result2["candidates"]])

    def test_empty_ready_short_source_is_censored_not_invented(self):
        result = run(fixture(frames=16000))
        self.assertEqual(result["counts"]["ready_rows"], 0)
        self.assertEqual(result["candidates"], [])
        self.assertIsNone(result["source_summaries"][0]["by_keyword"]["1"]["peak_probability"])

    def test_complete_clock_is_required(self):
        for mutation in ("missing", "duplicate", "reorder", "not_ready", "outside"):
            inputs = fixture()
            rows = inputs[2]["test-source-00"]
            if mutation == "missing": rows.pop(4)
            elif mutation == "duplicate": rows[4] = copy.deepcopy(rows[3])
            elif mutation == "reorder": rows[3], rows[4] = rows[4], rows[3]
            elif mutation == "not_ready": rows[4]["ready"] = False
            else: rows[-1]["sample"] = 160001
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): run(inputs)

    def test_pcm_and_model_identity_are_required(self):
        inputs = fixture()
        inputs[1]["test-source-00"] = bytes([9]) + inputs[1]["test-source-00"][1:]
        with self.assertRaises(ValueError): run(inputs)
        with self.assertRaises(ValueError): run(fixture(), model_sha256="f" * 64)
        with self.assertRaises(ValueError): run(fixture(), config_sha256="invalid")

    def test_development_or_upgraded_labels_are_rejected(self):
        for field, value in (("source_role", "development"), ("old_bg48_role", "study-development"),
                             ("current_coverage_role", "development"), ("trusted_negative", True),
                             ("human_gold", True), ("training_admitted", True),
                             ("expected_events", []), ("label_status", "negative")):
            inputs = fixture()
            inputs[0]["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): run(inputs)

    def test_nonfinite_bad_probability_and_bad_event_rejected(self):
        bad_values = ([float("nan"), 0, 1], [float("inf"), 0, 0], [-.1, .1, 1],
                      [True, 0, 0], [.9, .2, .1], [.9, .1], [1.1, 0, 0])
        for probabilities in bad_values:
            inputs = fixture()
            inputs[2]["test-source-00"][0]["probabilities"] = probabilities
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError): run(inputs)
        for keyword in (True, 3, -1, 1.0):
            inputs = fixture()
            inputs[2]["test-source-00"][0]["keyword"] = keyword
            with self.subTest(keyword=keyword), self.assertRaises(ValueError): run(inputs)

    def test_missing_trace_or_extra_source_rejected(self):
        inputs = fixture()
        inputs[3]["trace_sha256_by_id"].clear()
        with self.assertRaises(ValueError): run(inputs)
        inputs = fixture()
        inputs[2]["unexpected"] = []
        with self.assertRaises(ValueError): run(inputs)

    def test_hashes_bind_model_config_source_and_window(self):
        inputs = fixture()
        set_score(inputs, 5, .7)
        first = run(inputs)
        second = run(inputs, config_sha256=mining.digest(b"different fixed configuration"))
        a, b = first["candidates"][0], second["candidates"][0]
        self.assertNotEqual(a["candidate_id"], b["candidate_id"])
        self.assertEqual(a["window_pcm_sha256"], b["window_pcm_sha256"])
        self.assertEqual(a["source_pcm_sha256"], inputs[0]["rows"][0]["pcm_sha256"])
        self.assertEqual(a["model_sha256"], MODEL)
        self.assertEqual(a["config_sha256"], CONFIG)
        self.assertEqual(a["original_c_trace_sha256"], inputs[3]["trace_sha256_by_id"]["test-source-00"])
        self.assertEqual(a["input_ready_rows_sha256"], mining.digest(mining.canonical(inputs[2]["test-source-00"])))
        for key in mining.FALSE_FLAGS:
            self.assertIs(a[key], False)
            self.assertIs(first[key], False)


if __name__ == "__main__":
    unittest.main()
