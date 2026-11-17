"""Independent semantic fixtures; no model, audio artifact, or network calls.

These checks use small invented PCM arrays and event records. They test the
physical mixing and decision contracts without repeating a real experiment.
"""
from __future__ import annotations

import array
import copy
import math
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_quality_snr10_once as quality


def pcm(values):
    data = array.array("h", values)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def samples(raw):
    data = array.array("h")
    data.frombytes(raw)
    if sys.byteorder != "little":
        data.byteswap()
    return list(data)


class MixingContract(unittest.TestCase):
    def test_ten_db_measured_on_actual_quantized_difference(self):
        signal = [1000, 1000, -1000, -1000] * 40
        noise = [500, -500, 500, -500] * 40
        original, background = pcm(signal), pcm(noise)
        mixed, receipt = quality.mix_pcm16(original, background)
        self.assertEqual(receipt["status"], "MIXED")
        self.assertIsInstance(mixed, bytes)
        observed = samples(mixed)
        self.assertEqual(len(observed), len(signal))
        # This measurement is independent of any gain/SNR value in the receipt.
        error_power = sum((a - b) ** 2 for a, b in zip(observed, signal))
        signal_power = sum(a * a for a in signal)
        measured = 10 * math.log10(signal_power / error_power)
        self.assertAlmostEqual(measured, 10.0, delta=0.04)
        # Orthogonal signs make clipping or signal renormalization observable.
        self.assertEqual(sum(a * b for a, b in zip(observed, signal)), signal_power)
        self.assertEqual(original, pcm(signal))
        self.assertEqual(background, pcm(noise))
        self.assertEqual(quality.mix_pcm16(original, background), (mixed, receipt))

    def test_full_clip_rms_includes_existing_silence(self):
        signal = [2000, -2000] * 20 + [0] * 120
        noise = [1000, -1000] * 80
        mixed, receipt = quality.mix_pcm16(pcm(signal), pcm(noise))
        self.assertEqual(receipt["status"], "MIXED")
        observed = samples(mixed)
        measured = 10 * math.log10(
            sum(x * x for x in signal) /
            sum((x - y) ** 2 for x, y in zip(observed, signal))
        )
        self.assertAlmostEqual(measured, 10.0, delta=0.04)
        self.assertTrue(any(observed[40:]))

    def test_overflow_is_accounted_exclusion_never_saturation(self):
        mixed, receipt = quality.mix_pcm16(pcm([32760, -32760] * 80), pcm([1000, -1000] * 80))
        self.assertIsNone(mixed)
        self.assertEqual(receipt["status"], "MIX_HEADROOM_EXCLUDED")

    def test_zero_signal_and_zero_noise_have_no_invented_snr(self):
        for signal, noise in (([0] * 160, [1000, -1000] * 80),
                              ([1000, -1000] * 80, [0] * 160)):
            with self.subTest(signal_zero=not any(signal)):
                mixed, receipt = quality.mix_pcm16(pcm(signal), pcm(noise))
                self.assertIsNone(mixed)
                self.assertEqual(receipt["status"], "MIX_ZERO_ENERGY_EXCLUDED")

    def test_partial_or_unequal_pcm_is_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            quality.mix_pcm16(b"\x00", pcm([1000]))
        with self.assertRaises((ValueError, TypeError)):
            quality.mix_pcm16(pcm([1000, -1000]), pcm([1000]))

    def test_nonfinite_snr_cannot_create_nan_pcm(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                quality.mix_pcm16(pcm([1000, -1000]), pcm([1000, -1000]), value)


class SourceSplitContract(unittest.TestCase):
    def setUp(self):
        self.tracks = [
            {"id": "train-a", "role": "train", "frames": 1000},
            {"id": "train-b", "role": "train", "frames": 800},
            {"id": "dev-a", "role": "development", "frames": 1200},
        ]

    def test_selection_never_crosses_role_and_has_bounded_span(self):
        for role in ("train", "development"):
            for case_id in ("case-a", "case-b", "case-c", "case-d"):
                with self.subTest(role=role, case_id=case_id):
                    selected, start = quality.select_noise_track(case_id, role, 160, self.tracks)
                    self.assertEqual(selected["role"], role)
                    self.assertIs(type(start), int)
                    self.assertGreaterEqual(start, 0)
                    self.assertLessEqual(start + 160, selected["frames"])

    def test_selection_does_not_depend_on_archive_enumeration_order(self):
        before = copy.deepcopy(self.tracks)
        first = quality.select_noise_track("immutable-case", "train", 160, self.tracks)
        reordered = quality.select_noise_track("immutable-case", "train", 160, list(reversed(self.tracks)))
        self.assertEqual(first, reordered)
        self.assertEqual(self.tracks, before)

    def test_missing_train_capacity_does_not_fall_back_to_development(self):
        with self.assertRaises(ValueError):
            quality.select_noise_track("case", "train", 1100, self.tracks)


class EventMeaningContract(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "invented-clip", "role": "train", "weak_class": 1,
                    "observed_samples": 33920,
                    "source": {"clip_start_sample": 16000,
                               "original_clip_eof_sample": 28800}}

    def test_prefix_event_is_not_credited_to_clip(self):
        result = quality.summarize_case(self.row, [{"sample": 16000, "keyword": 1}])
        self.assertFalse(result["matched_intent_after_clip_start"])
        self.assertEqual(len(result["prefix_events"]), 1)
        self.assertTrue(result["no_matching_event_in_observed_context"])
        self.assertTrue(result["research_error"])

    def test_correct_event_does_not_hide_wrong_keyword(self):
        result = quality.summarize_case(self.row, [{"sample": 17280, "keyword": 2},
                                                  {"sample": 18560, "keyword": 1}])
        self.assertTrue(result["matched_intent_after_clip_start"])
        self.assertEqual(len(result["wrong_keyword_after_clip_start"]), 1)
        self.assertTrue(result["research_error"])

    def test_additional_same_keyword_event_is_separately_counted(self):
        result = quality.summarize_case(self.row, [{"sample": 17280, "keyword": 1},
                                                  {"sample": 33920, "keyword": 1}])
        self.assertTrue(result["matched_intent_after_clip_start"])
        self.assertEqual(result["extra_same_keyword_events"], 1)
        self.assertEqual(result["wrong_keyword_after_clip_start"], [])

    def test_absent_positive_event_is_censored_not_frr(self):
        result = quality.summarize_case(self.row, [])
        self.assertTrue(result["no_matching_event_in_observed_context"])
        self.assertTrue(result["research_error"])
        self.assertEqual(result["no_event_status"], "CENSORED_NOT_FRR")
        self.assertFalse(result["independent_acoustic_evaluation"])

    def test_file_eof_difference_does_not_become_word_end_latency(self):
        result = quality.summarize_case(self.row, [{"sample": 28800, "keyword": 1}])
        self.assertEqual(result["event_minus_clip_eof_samples"], [0])
        self.assertEqual(result["eof_prefix_event_count"], 1)
        self.assertEqual(result["word_end_latency"], "NOT_MEASURED")

    def test_negative_intent_reports_events_without_claiming_gold(self):
        row = {**self.row, "weak_class": 0}
        quiet = quality.summarize_case(row, [])
        active = quality.summarize_case(row, [{"sample": 17280, "keyword": 2}])
        self.assertFalse(quiet["research_error"])
        self.assertTrue(active["research_error"])
        self.assertEqual(active["weak_negative_intent_events"], 1)
        self.assertFalse(active["independent_acoustic_evaluation"])

    def test_invalid_event_clocks_and_types_are_rejected(self):
        cases = [
            [{"sample": 17280, "keyword": True}],
            [{"sample": True, "keyword": 1}],
            [{"sample": 17280.0, "keyword": 1}],
            [{"sample": 33921, "keyword": 1}],
            [{"sample": 0, "keyword": 1}],
            [{"sample": 18560, "keyword": 1}, {"sample": 17280, "keyword": 2}],
            [{"sample": 17280, "keyword": 1}, {"sample": 17280, "keyword": 2}],
        ]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(ValueError):
                quality.summarize_case(self.row, events)

    def test_invalid_clip_region_is_rejected_even_with_no_events(self):
        for change in ({"clip_start_sample": 28801},
                       {"original_clip_eof_sample": 33921},
                       {"clip_start_sample": True},
                       {"clip_start_sample": None, "original_clip_eof_sample": None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                quality.summarize_case({**self.row, "source": {**self.row["source"], **change}}, [])


class CandidateTriggerContract(unittest.TestCase):
    def setUp(self):
        self.roster = []
        for label, count in ((0, 8), (1, 2), (2, 2)):
            for index in range(count):
                self.roster.append({"id": f"train-{label}-{index}", "role": "train",
                                    "weak_class": label, "source_family_id": f"family-{index}",
                                    "mix": {"status": "MIXED"}})
        self.roster.append({"id": "dev-1-0", "role": "development", "weak_class": 1,
                            "source_family_id": "held-dev-family", "mix": {"status": "MIXED"}})
        self.metrics = [{"id": row["id"], "role": row["role"], "weak_class": row["weak_class"],
                         "research_error": False} for row in self.roster]

    def test_development_error_never_starts_training(self):
        self.metrics[-1]["research_error"] = True
        decision = quality.candidate_decision(self.metrics, self.roster)
        self.assertFalse(decision["execute"])
        self.assertEqual(decision["train_errors"], 0)
        self.assertFalse(decision["development_used_for_decision"])

    def test_one_train_error_and_complete_coverage_allow_only_fixed_candidate(self):
        self.metrics[0]["research_error"] = True
        decision = quality.candidate_decision(self.metrics, self.roster)
        self.assertTrue(decision["execute"])
        self.assertEqual(decision["train_errors"], 1)
        self.assertTrue(all(group["passed"] for group in decision["coverage"].values()))

    def test_insufficient_positive_voice_coverage_does_not_train_on_clean_only(self):
        self.metrics[0]["research_error"] = True
        for row in self.roster:
            if row["role"] == "train" and row["weak_class"] == 1:
                row["source_family_id"] = "only-one-family"
        decision = quality.candidate_decision(self.metrics, self.roster)
        self.assertFalse(decision["execute"])
        self.assertEqual(decision["reason"], "BASELINE_ONLY_NO_VALID_AUGMENTED_TRAINING")

    def test_missing_training_metric_cannot_silently_count_as_success(self):
        with self.assertRaises(ValueError):
            quality.candidate_decision(self.metrics[1:], self.roster)

    def test_duplicate_metric_cannot_overwrite_original_result(self):
        duplicate = {**self.metrics[0], "research_error": True}
        with self.assertRaises(ValueError):
            quality.candidate_decision([*self.metrics, duplicate], self.roster)

    def test_metric_class_must_match_the_training_source(self):
        self.metrics[0]["weak_class"] = 1
        with self.assertRaises(ValueError):
            quality.candidate_decision(self.metrics, self.roster)

    def test_metric_error_must_be_boolean(self):
        self.metrics[0]["research_error"] = "false"
        with self.assertRaises(ValueError):
            quality.candidate_decision(self.metrics, self.roster)


@unittest.skipUnless(sys.platform == "linux", "process-group fixture uses Linux process state")
class SupervisorContract(unittest.TestCase):
    @staticmethod
    def alive(pid):
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return False
        # A zombie has stopped executing and cannot outlive the compute budget.
        return text.rsplit(")", 1)[1].strip().split()[0] != "Z"

    def exercise_group(self, *, leader_exits):
        child_code = ("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                      "print('ready',flush=True); time.sleep(30)")
        leader_code = (
            "import os,subprocess,sys,time; "
            f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}],stdout=subprocess.PIPE); "
            "assert child.stdout.readline()==b'ready\\n'; "
            "print(os.getpid(),child.pid,flush=True); " +
            ("sys.exit(0)" if leader_exits else "time.sleep(30)")
        )
        with tempfile.TemporaryDirectory(prefix="kws-supervisor-fixture-") as temporary:
            root = Path(temporary)
            output, errors = root / "dummy.stdout", root / "dummy.stderr"
            begin = time.monotonic()
            try:
                if leader_exits:
                    quality.supervised_command([sys.executable, "-c", leader_code], output, errors, 5)
                else:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        quality.supervised_command([sys.executable, "-c", leader_code], output, errors, 0.25)
                parent_pid, child_pid = map(int, output.read_text().strip().split())
                # Do not confuse process exit with completion of all its children.
                limit = time.monotonic() + 1
                while self.alive(child_pid) and time.monotonic() < limit:
                    time.sleep(0.01)
                self.assertFalse(self.alive(child_pid), "supervised descendant survived its owning command")
                self.assertLess(time.monotonic() - begin, 10)
                receipt = json.loads(output.with_suffix(".exit.json").read_text())
                self.assertEqual(receipt["returncode"], 0 if leader_exits else -signal.SIGTERM)
            finally:
                # Also clean up if this fixture exposes the bug on an old worker.
                if output.exists() and output.read_text().strip():
                    parent_pid, child_pid = map(int, output.read_text().strip().split())
                    try:
                        os.killpg(parent_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_completed_leader_cannot_leave_a_descendant_running(self):
        self.exercise_group(leader_exits=True)

    def test_timeout_terminates_the_whole_process_group(self):
        self.exercise_group(leader_exits=False)


if __name__ == "__main__":
    unittest.main()
