"""Small independent fixtures for rejection paths; no model/compiler imports."""
import copy
import io
import json
from pathlib import Path
import struct
import unittest
import zipfile

import audit_coverage as audit


def packed(values):
    return struct.pack("<" + str(len(values)) + "h", *values)


def array_record(dtype, shape, payload):
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode("ascii")
    header += b" " * ((16 - (10 + len(header) + 1) % 16) % 16) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


def array_archive(records):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as out:
        for name, record in records.items():
            out.writestr(name + ".npy", array_record(*record))
    return stream.getvalue()


def data_fixture():
    clean = [{"id": f"parent-{label}-{n}", "role": "train", "weak_class": label}
             for label, size in ((0, 32), (1, 4), (2, 5)) for n in range(size)]
    noisy = [dict(row) for row in clean] + [{"id": "exposed-dev", "role": "development", "weak_class": 1}]
    new = [{**row, "parent_id": row["id"], "condition": condition, "id": condition + "/" + row["id"]}
           for condition in ("new5", "new15") for row in clean]
    previous = []
    for step in range(96):
        draws = []
        for condition, rows in (("clean", clean), ("snr10", noisy)):
            for label in range(3):
                group = [r for r in rows if r["role"] == "train" and r["weak_class"] == label]
                draws.append({"condition": condition, "id": group[step % len(group)]["id"], "weak_class": label, "role": "train"})
        previous.append(draws)
    return clean, noisy, new, previous


def paired_fixture():
    pairs = []
    for condition in ("clean", "snr10", "snr5", "new8"):
        for label in (1, 2):
            rescue = condition == "snr10" and label == 1
            pairs.append({"condition": condition, "id": condition + str(label), "role": "development",
                "weak_class": label, "regressed": False, "rescued": rescue,
                "control_missing_in_observation": rescue, "introduced_event_errors": {}})
    rows = [{"role": "development", "weak_class": label, "parent_id": f"parent-{label}-{n}"}
            for label, count in ((0, 16), (1, 3), (2, 8)) for n in range(count)]
    return pairs, rows


class AcousticConditionContracts(unittest.TestCase):
    def test_no_headroom_repair(self):
        raw, receipt = audit.scalar_mix(packed([32767, 32767]), packed([32767, 32767]), 5.0)
        self.assertIsNone(raw)
        self.assertEqual(receipt["status"], "MIX_HEADROOM_EXCLUDED")
        self.assertGreater(receipt["unbounded_peak"], 32767)
        self.assertEqual(receipt["clipped_samples"], 0)

    def test_silent_clip_is_excluded(self):
        raw, receipt = audit.scalar_mix(packed([0] * 8), packed([100, -100] * 4), 15.0)
        self.assertIsNone(raw)
        self.assertEqual(receipt["status"], "MIX_ZERO_ENERGY_EXCLUDED")

    def test_clip_only_rms_is_fixed(self):
        signal, noise = packed([1000, -1000] * 10), packed([100, -100] * 10)
        raw, receipt = audit.scalar_mix(signal, noise, 5.0)
        self.assertEqual(receipt["signal_power"], 1_000_000)
        self.assertEqual(receipt["noise_power"], 10_000)
        self.assertEqual(receipt["source_pcm_sha256"], audit.digest(signal))
        self.assertLess(abs(receipt["actual_snr_db"] - 5.0), 0.05)
        self.assertEqual(receipt["mixed_pcm_sha256"], audit.digest(raw))

    def test_prespecified_snrs_only(self):
        with self.assertRaises(ValueError):
            audit.scalar_mix(packed([1000]), packed([100]), 7.0)

    def test_role_partition_and_three_distinct_tracks(self):
        tracks = [{"id": f"train-{i}", "role": "train", "frames": 100, "pcm_sha256": str(i)} for i in range(5)]
        tracks += [{"id": "dev", "role": "development", "frames": 100, "pcm_sha256": "dev"}]
        excluded = {"train-0"}
        first, first_offset = audit.select_new_noise("p", "train", "new5", 20, tracks, excluded)
        excluded.add(first["id"])
        second, _ = audit.select_new_noise("p", "train", "new15", 20, tracks, excluded)
        self.assertEqual(first["role"], second["role"])
        self.assertEqual(first["role"], "train")
        self.assertEqual(len({"train-0", first["id"], second["id"]}), 3)
        self.assertTrue(0 <= first_offset <= 80)
        self.assertEqual((first, first_offset), audit.select_new_noise("p", "train", "new5", 20, list(reversed(tracks)), {"train-0"}))

    def test_training_cannot_select_new8(self):
        with self.assertRaises(ValueError):
            audit.select_new_noise("p", "train", "new8", 20, [], set())

    def test_new_condition_ids_bind_purpose_track_and_offset(self):
        track = {"id": "train-1", "pcm_sha256": "a" * 64}
        original = audit.new_condition_id("p", "train", "new5", "b" * 64, track, 2)
        changed = audit.new_condition_id("p", "train", "new5", "b" * 64, track, 3)
        self.assertNotEqual(original, changed)
        self.assertTrue(original.startswith("encov-new5-"))
        self.assertNotEqual(original, "p")

    def test_sixteen_windows_stop_at_clip_start(self):
        clean = packed([20] * 36000)
        mixed = clean[:70400] + packed([30] * 800)
        proof = audit.prefix_window_proof(clean, mixed)
        self.assertEqual(len(proof), 16)
        self.assertEqual([p["available_sample"] for p in proof], list(range(16000, 35201, 1280)))
        changed = bytearray(mixed); changed[70399] ^= 1
        with self.assertRaises(ValueError):
            audit.prefix_window_proof(clean, bytes(changed))


class TrainingDrawContracts(unittest.TestCase):
    def test_balanced_occurrences_preserve_clean(self):
        clean, noisy, new, old = data_fixture()
        draws = audit.expected_draws(clean, noisy, new)
        result = audit.verify_coverage_draws(draws, old, clean, noisy, new)
        self.assertEqual(result["bag_forwards"], 576)
        self.assertTrue(all(a[:3] == b[:3] for a, b in zip(draws, old)))
        for condition in ("new5", "new15"):
            negatives = [d[3]["id"] for d in draws if d[3]["condition"] == condition]
            self.assertEqual(len(negatives), 32)
            self.assertEqual(len(set(negatives)), 32)

    def test_step_instead_of_occurrence_is_rejected(self):
        clean, noisy, new, old = data_fixture()
        draws = audit.expected_draws(clean, noisy, new)
        draws[1][4]["id"] = "new5/parent-1-1"
        with self.assertRaises(ValueError):
            audit.verify_coverage_draws(draws, old, clean, noisy, new)

    def test_headroom_exclusion_does_not_trigger_replacement(self):
        clean, noisy, new, old = data_fixture()
        new = [r for r in new if r["id"] != "new5/parent-0-0"]
        draws = audit.expected_draws(clean, noisy, new)
        audit.verify_coverage_draws(draws, old, clean, noisy, new)
        negatives = [d[3]["id"] for d in draws if d[3]["condition"] == "new5"]
        self.assertEqual(len(negatives), 32)
        self.assertEqual(len(set(negatives)), 31)
        self.assertEqual(negatives[0], negatives[-1])
        self.assertNotIn("new5/parent-0-0", negatives)

    def test_dev_parent_cannot_be_silently_filtered_into_new_train_pool(self):
        clean, noisy, new, _ = data_fixture()
        new[0]["role"] = "development"
        with self.assertRaises(ValueError):
            audit.expected_draws(clean, noisy, new)

    def test_old_eval5_draw_cannot_be_admitted(self):
        clean, noisy, new, old = data_fixture()
        draws = audit.expected_draws(clean, noisy, new)
        draws[1][3]["condition"] = "snr5"
        with self.assertRaises(ValueError):
            audit.verify_coverage_draws(draws, old, clean, noisy, new)


class CacheAndGateContracts(unittest.TestCase):
    def test_prefix_float_bytes_and_clocks_are_bound(self):
        parent = {"id": "parent", "first_record": 0, "records": 21, "observed_samples": 41600}
        times = list(range(16000, 41601, 1280))
        raw = b"\0" * (21 * 96 * 4)
        old = array_archive({"embeddings": ("<f4", (21, 96), raw)})
        records = {"e000": ("<f4", (21, 96), raw),
                   "t000": ("<u8", (21,), struct.pack("<21Q", *times))}
        row = {**parent, "id": "new", "parent_id": "parent", "mix": {"prefix_cache_rows_reused": 16}}
        audit.verify_new_cache(array_archive(records), [row], [parent], old)
        changed = bytearray(raw); changed[:4] = struct.pack("<f", 0.5)
        records["e000"] = ("<f4", (21, 96), bytes(changed))
        with self.assertRaises(ValueError):
            audit.verify_new_cache(array_archive(records), [row], [parent], old)

    def test_success_still_requires_old10_rescue(self):
        pairs, rows = paired_fixture()
        self.assertTrue(audit.coverage_gate(pairs, rows, rows)["accepted"])
        for pair in pairs:
            pair["rescued"] = pair["control_missing_in_observation"] = False
        gate = audit.coverage_gate(pairs, rows, rows)
        self.assertFalse(gate["accepted"])
        self.assertIn("NO_SNR10_DEVELOPMENT_MISS_RESCUED", gate["reasons"])

    def test_new8_regression_cannot_be_hidden_by_old_gain(self):
        pairs, rows = paired_fixture()
        pairs[-1]["regressed"] = True
        self.assertIn("NEW8_CASE_REGRESSION", audit.coverage_gate(pairs, rows, rows)["reasons"])

    def test_new8_extra_event_rejects_already_wrong_case(self):
        pairs, rows = paired_fixture()
        pairs[-1]["introduced_event_errors"] = {"extra_count": 1}
        gate = audit.coverage_gate(pairs, rows, rows)
        self.assertFalse(gate["accepted"])
        self.assertFalse(audit.coverage_retention(gate, pairs)["criteria"]["all_conditions_no_per_case_event_error_increase"])

    def test_new8_both_keywords_required(self):
        pairs, rows = paired_fixture()
        self.assertIn("NEW8_DEVELOPMENT_KEYWORD_COVERAGE_INSUFFICIENT",
                      audit.coverage_gate(pairs, rows, rows[:1])["reasons"])

    def test_one_excluded_new8_case_rejects_even_with_both_keywords(self):
        pairs, rows = paired_fixture()
        result = audit.coverage_gate(pairs, rows, rows[1:])
        self.assertGreater(min(result["new8_development_positive_coverage"].values()), 0)
        self.assertIn("NEW8_PLANNED_DEVELOPMENT_COVERAGE_INCOMPLETE", result["reasons"])

    def test_duplicate_new8_parent_cannot_fill_denominator(self):
        pairs, rows = paired_fixture()
        rows[-1]["parent_id"] = rows[-2]["parent_id"]
        self.assertFalse(audit.coverage_gate(pairs, rows, rows)["new8_all_27_planned_development_cases_eligible"])

    def test_empty_challenge_is_verified_gate_failure_not_missing_case_exception(self):
        pairs, rows = paired_fixture()
        pairs = [p for p in pairs if p["condition"] != "new8"]
        result = audit.coverage_gate(pairs, rows, [])
        self.assertFalse(result["accepted"])
        self.assertIn("NEW8_PLANNED_DEVELOPMENT_COVERAGE_INCOMPLETE", result["reasons"])

    def test_original_eof_miss_is_censored(self):
        row = {"id": "p", "role": "development", "weak_class": 1, "observed_samples": 41600,
               "source": {"clip_start_sample": 35200, "original_clip_eof_sample": 36480}}
        result = audit.legacy.event_observation(row, [{"sample": 39040, "keyword": 1}])
        self.assertTrue(result["original_eof_no_matching_event_censored"])
        self.assertEqual(result["word_end_latency"], "NOT_MEASURED")
        self.assertFalse(result["research_error"])


class SavedCProofContracts(unittest.TestCase):
    def setUp(self):
        self.reference = [{"sample": 16000, "ready": False, "keyword": 0,
                           "logits": [0.0, 0.0, 0.0], "probabilities": [0.0, 0.0, 0.0]}]

    def test_equal_saved_rows_pass_without_c_execution(self):
        result = audit.compare_c_stdout(b"16000 0 0 0 0 0 0 0 0\n", self.reference)
        self.assertEqual(result["input_embeddings"], 1)
        self.assertEqual(result["max_abs_logit_error"], 0.0)

    def test_clock_event_and_ready_are_exact(self):
        for first in ("16001 0 0", "16000 1 0", "16000 0 1", "16000 2 0"):
            with self.subTest(first=first), self.assertRaises(ValueError):
                audit.compare_c_stdout((first + " 0 0 0 0 0 0\n").encode(), self.reference)

    def test_nan_missing_extra_and_over_tolerance_reject(self):
        for row in (b"", b"16000 0 0 nan 0 0 0 0 0\n", b"16000 0 0 0.000101 0 0 0 0 0\n",
                    b"16000 0 0 0 0 0 0.00000201 0 0\n", b"16000 0 0 0 0 0 0 0 0 99\n"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                audit.compare_c_stdout(row, self.reference)


class PreregisteredSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = next((parent / "automation/english_coverage_quality_once.py"
            for parent in Path(__file__).resolve().parents
            if (parent / "automation/english_coverage_quality_once.py").is_file()), None)
        if cls.worker is None:
            raise AssertionError("published coverage worker must accompany source fixtures")
        cls.raw = cls.worker.read_bytes()
        cls.plan = json.loads(cls.worker.with_name("english_coverage_quality_plan.json").read_text())

    def test_actual_source_and_closed_plan_are_standard_library_inspected(self):
        self.assertTrue(audit.source_only_coverage_contract(self.raw)["static_inspection_only"])
        audit.verify_plan(self.plan)

    def test_dense_loss_restart_is_rejected(self):
        changed = self.raw.replace(b"math.log(len(logp)) - torch.logsumexp(logp[:, label], dim=0)",
                                   b"-logp[:, label].mean()")
        self.assertNotEqual(changed, self.raw)
        with self.assertRaises(ValueError):
            audit.source_only_coverage_contract(changed)

    def test_nested_prefix_windows_do_not_shadow_training_geometry(self):
        self.assertIn(b"windows = []", self.raw)
        self.assertIn(b"windows = np.asarray", self.raw)
        self.assertTrue(audit.source_only_coverage_contract(self.raw)["unchanged_geometry_normalization_optimizer_checked"])

    def test_weakening_challenge_or_changing_epochs_rejects(self):
        for field, value in (("maximum_optimizer_updates", 192), ("maximum_embedding_calls", 3000),
                             ("development_tuning", True)):
            changed = copy.deepcopy(self.plan); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.verify_plan(changed)
        changed = copy.deepcopy(self.plan)
        changed["research_retention"]["new8_all_27_planned_development_cases_eligible"] = False
        with self.assertRaises(ValueError):
            audit.verify_plan(changed)

    def test_actual_suffix_and_two_c_backend_budgets(self):
        train = {"new_embedding_calls_required": 2008, "prefix_cache_rows_reused": 1536}
        challenge = {"new_embedding_calls_required": 569, "eligible_conditions": 27,
                     "eligible_embedding_rows": 1001, "prefix_cache_rows_reused": 432}
        counts = audit.expected_counts(train, challenge)
        self.assertEqual(counts["new_embedding_calls_completed"], 2577)
        self.assertEqual(counts["onnx_forward_completed"], 5154)
        self.assertEqual(counts["head_case_passes_completed"], 318)
        self.assertEqual(counts["head_ready_outputs"], 13039)
        self.assertEqual(counts["host_c_head_case_passes_completed"], 291)
        self.assertEqual(counts["arm_c_embedding_rows_completed"], 16808)
        self.assertEqual(counts["arm_c_head_ready_outputs"], 12443)


if __name__ == "__main__":
    unittest.main()
