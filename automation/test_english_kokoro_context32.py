"""Pure source/receipt fixtures; no real audio derivation or model calls."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_kokoro_context32_once as worker
source = worker.source


def plan():
    return json.loads((HERE / "english_kokoro_context32_plan.json").read_bytes())


def pcm(values):
    return struct.pack("<" + "h" * len(values), *values)


class SourceContractTests(unittest.TestCase):
    def test_real_source_pins_and_disabled_or_bound_state(self):
        frozen = plan()
        worker.validate_plan(frozen)
        frozen.update(enabled=False, main_before=None)
        worker.validate_plan(frozen)
        frozen.update(enabled=True, main_before=source.MAIN_BEFORE)
        worker.validate_plan(frozen)

    def test_any_other_successor_binding_is_rejected(self):
        frozen = plan()
        frozen.update(enabled=True, main_before="a" * 40)
        with self.assertRaisesRegex(ValueError, "ONE_FIXED_SUCCESSOR"):
            worker.validate_plan(frozen)

    def test_roster_removal_or_reordering_is_rejected(self):
        for change in (lambda r: r.pop(), lambda r: r.reverse()):
            frozen = plan(); change(frozen["input_roster"])
            with self.assertRaisesRegex(ValueError, "FROZEN_ALL16"):
                worker.validate_plan(frozen)

    def test_crop_native_or_preset_identity_cannot_drift(self):
        for key, value in (("crop_start_native_sample", 0), ("preset_id", "different"), ("native_full_f32_sha256", "0" * 64)):
            frozen = plan(); frozen["input_roster"][0][key] = value
            with self.assertRaisesRegex(ValueError, "FROZEN_ALL16"):
                worker.validate_plan(frozen)

    def test_old48_exclusion_cannot_be_shortened_or_fabricated(self):
        for change in (lambda v: v.pop(), lambda v: v.__setitem__(0, "0" * 64)):
            frozen = plan(); change(frozen["old_observed_pcm_sha256"])
            with self.assertRaisesRegex(ValueError, "FROZEN_OLD48"):
                worker.validate_plan(frozen)

    def test_budgets_and_scope_cannot_grow(self):
        for key, value in (("maximum_total_model_calls", 33), ("new_tts_calls", 1), ("retry_count", 1),
                           ("training_allowed", True), ("old_labels_modified", True), ("automatic_expansion", True)):
            frozen = plan(); frozen[key] = value
            with self.assertRaises(ValueError):
                worker.validate_plan(frozen)

    def test_only_actual_two_phone_pins(self):
        frozen = plan(); frozen["observers"]["phoneme"]["model_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "EXACT_ORIGINAL_TWO"):
            worker.validate_plan(frozen)

    def test_pinned_legacy_closure_is_not_active_old_plan_copy(self):
        frozen = plan()
        self.assertEqual(len(frozen["helper_sha256"]), 24)
        self.assertNotIn("automation/english_kokoro48_plan.json", frozen["helper_sha256"])
        self.assertNotIn(".github/workflows/english-kokoro48-once.yml", frozen["helper_sha256"])

    def test_first_public_main_push_only(self):
        frozen = plan(); frozen.update(enabled=True, main_before=source.MAIN_BEFORE)
        env = {"GITHUB_REPOSITORY": "jiying2007/kws-data", "GITHUB_SHA": "1" * 40,
            "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main"}
        event = {"before": source.MAIN_BEFORE, "after": env["GITHUB_SHA"],
            "repository": {"full_name": env["GITHUB_REPOSITORY"], "private": False}}
        self.assertEqual(worker.execution_guard(frozen, env, event)["run_id"], "123")
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                           ("GITHUB_REF", "refs/heads/research"), ("GITHUB_REPOSITORY", "someone/fork")):
            with self.assertRaises(ValueError):
                worker.execution_guard(frozen, env | {key: value}, event)
        with self.assertRaises(ValueError):
            worker.execution_guard(frozen, env, event | {"before": "2" * 40})

    def test_disabled_plan_has_no_guard_permission(self):
        frozen = plan(); frozen.update(enabled=False, main_before=None)
        with self.assertRaises(ValueError):
            worker.execution_guard(frozen, {}, {})

    def test_sample_differences_keep_nonidentical_left_boundary(self):
        full = pcm([90, 91, 92, 1, 3, 7, 8, 9])
        old = pcm([2, 4, 7, 8, 9])
        result = source.difference(full, old, 3)
        self.assertEqual(result["differing_samples"], 2)
        self.assertEqual(result["matching_suffix_crop_range"], [2, 5])
        self.assertEqual(result["matching_suffix_full_range"], [5, 8])
        self.assertEqual(result["maximum_absolute_sample_difference"], 1)
        self.assertFalse(result["entire_pcm_suffix_equal"])
        self.assertFalse(result["frame_equivalence_claimed"])

    def test_equal_sample_suffix_still_not_equal_transformer_frames(self):
        values = pcm([1, 2, 3])
        result = source.difference(pcm([8, 9]) + values, values, 2)
        self.assertTrue(result["entire_pcm_suffix_equal"])
        self.assertEqual(result["matching_suffix_crop_range"], [0, 3])
        self.assertFalse(result["model_normalization_equivalence_claimed"])

    def test_suffix_length_cannot_be_assumed_or_truncated(self):
        with self.assertRaisesRegex(ValueError, "ALIGNED_SUFFIX_LENGTH"):
            source.difference(pcm([1, 2, 3, 4]), pcm([3, 4]), 1)

    def test_wav_pcm_identity(self):
        raw = pcm([-32768, -1, 0, 1, 32767])
        with source.wave.open(source.io.BytesIO(source.wav_bytes(raw)), "rb") as stream:
            self.assertEqual((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()), (1, 2, 16000, 5))
            self.assertEqual(stream.readframes(5), raw)

    def test_safe_paths_and_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../escape", "/absolute", "a//b", "a/./b", "a\\b"):
                with self.assertRaises(ValueError):
                    source.within(directory, name)
        with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
            source.loads('{"x":1,"x":2}')

    def test_raw_zip_identity_is_required_before_inspection(self):
        with self.assertRaisesRegex(ValueError, "EXACT_OLD_ARCHIVE"):
            source.archive_bytes(b"not the original API ZIP", "source")

    def test_closed_manifest_detects_unknown_file_and_byte_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source.put(root / "one.json", b"{}")
            source.once(root / "FILES.json", worker.file_manifest(root))
            worker.closed_readback(root)
            (root / "one.json").write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "CLOSED_BYTE"):
                worker.closed_readback(root)

    def test_missing_observer_is_unknown_not_zero_call_proof(self):
        frozen = plan()
        manifest = {"rows": [{**r, "pcm_sha256": None, "wav_sha256": None, "pcm_samples": None} for r in frozen["input_roster"]]}
        with tempfile.TemporaryDirectory() as directory:
            result = worker.observer_counts(Path(directory), "phoneme", manifest, {"mock": True}, "0" * 64)
        self.assertEqual(result["declared_count"], 16)
        self.assertIsNone(result["actual_forward_attempts"])
        self.assertEqual(result["retained_forward_attempts_lower_bound"], 0)
        self.assertEqual(result["counter_scope"], "retained_marker_lower_bound")
        self.assertEqual(result["forward_attempts_upper_bound"], 16)

    def test_reported_return_and_retained_return_receipt_remain_separate(self):
        frozen = plan()
        rows = [{**r, "pcm_sha256": "a" * 64, "wav_sha256": "b" * 64, "pcm_samples": 24} for r in frozen["input_roster"]]
        run = {"mock_fixture": True}
        first = rows[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / first["id"]).mkdir()
            source.once(root / first["id"] / "FORWARD_STARTED.json", {"id": first["id"], "source_id": first["source_id"],
                "provider": "phoneme", "run": run, "input_pcm_sha256": first["pcm_sha256"],
                "input_wav_sha256": first["wav_sha256"], "input_samples": 24, "retry_count": 0, "forward_attempts": 1})
            source.once(root / "RESULT.json", {"provider": "phoneme", "run": run, "input_manifest_sha256": "c" * 64,
                "declared_count": 16, "retry_count": 0, "actual_forward_attempts": 1,
                "native_forward_returns": 1, "completed": 0})
            result = worker.observer_counts(root, "phoneme", {"rows": rows}, run, "c" * 64)
        # Full ledger validation is independently performed by compare(); this
        # wrapper must not contradict a returned-then-receipt-write-failed row.
        self.assertEqual(result["native_forward_returns"], 1)
        self.assertEqual(result["retained_native_forward_returns_lower_bound"], 0)
        self.assertEqual(result["completed"], 0)

    def test_no_forbidden_import_or_execution_in_source_checks(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "kokoro", "transformers", "numpy", "scipy")))


class PreparationReceiptTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        inputs = root / "inputs"; inputs.mkdir()
        for name in source.INPUT_FILES:
            source.put(inputs / name, b"fixture retained input; old ZIP verification is separately tested")
        carriers, frozen, source_files = [], [], {}
        generated = {}
        for i in range(16):
            old = {"id": "enk48-" + format(i, "024x"), "request_sha256": format(i, "064x"),
                "preset_id": worker.observer.PRESETS[i//4], "cell_id": worker.observer.CELLS[i%4],
                "requested_class": 2 if i%4 < 2 else 1,
                "pronunciation_hypothesis": ["mero-us-eh-v1", "mero-us-i-v1", "vosen-us-s-v1", "vosen-us-z-v1"][i%4],
                "native_samples": 36, "crop_start_native_sample": 18, "crop_end_native_sample": 36,
                "native_full_f32_path": f"native/{i}.f32", "pcm_path": f"old/{i}.pcm", "pcm_samples": 12,
                "duration_sha256": "d" * 64, "crop_sha256": "c" * 64}
            native = struct.pack("<36f", *([0.125 + i/100] * 36))
            old_pcm = pcm([i + 100] * 12)
            old.update(native_full_f32_sha256=source.sha(native), pcm_sha256=source.sha(old_pcm))
            source_files[old["native_full_f32_path"]] = native
            source_files[old["pcm_path"]] = old_pcm
            generated[native] = pcm([i + 1] * 12) + old_pcm
            carriers.append(old); frozen.append(source.static_row(old))
        old_hashes = sorted([r["pcm_sha256"] for r in carriers] + [format(i + 1000, "064x") for i in range(32)])
        original = {"input_roster": frozen, "old_observed_pcm_sha256": old_hashes, "carriers": carriers,
            "source": source_files, "api_files": {}, "old_plan_sha256": "1" * 64, "old_stop_analysis_sha256": "2" * 64}
        frozen_plan = {"enabled": True, "campaign_id": source.CAMPAIGN_ID,
            "input_roster": frozen, "old_observed_pcm_sha256": old_hashes,
            "source_code_sha256": source.sha(Path(source.__file__).read_bytes())}
        numpy = types.ModuleType("numpy")
        scipy = types.ModuleType("scipy"); signal = types.ModuleType("scipy.signal")
        signal.resample_poly = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("NO_REAL_RESAMPLING_IN_FIXTURES"))
        scipy.signal = signal
        return inputs, original, frozen_plan, generated, {"numpy": numpy, "scipy": scipy, "scipy.signal": signal}

    def perform(self, root, *, fail_proof=False, fail_failure_receipt=False, collision=False):
        inputs, original, frozen_plan, generated, modules = self.fixture(root)
        if collision:
            original["old_observed_pcm_sha256"][0] = source.sha(next(iter(generated.values())))
        original_once = source.once
        def writer(path, value):
            if fail_proof and Path(path).name == "DERIVATION.json":
                raise OSError("mock proof disk failure")
            if fail_failure_receipt and Path(path).name == "FAILED.json":
                raise OSError("mock failure receipt disk failure")
            return original_once(path, value)
        output = Path(root) / "prepared"
        with patch.object(source, "verified_original", return_value=original), \
             patch.object(source.importlib.metadata, "version", side_effect=lambda name: source.DERIVATION["runtime_versions"][name]), \
             patch.dict(sys.modules, modules), \
             patch.object(source, "numerical_derivative", side_effect=lambda raw, *_args: generated[raw]) as derivative, \
             patch.object(source, "once", side_effect=writer):
            result = source.prepare(frozen_plan, inputs, output, {"mock_fixture": True})
            manifest = source.readback(output, frozen_plan, allow_partial=True)
        return result, manifest, derivative.call_count

    def test_all_sixteen_mock_outputs_preserve_full_roster_and_sample_proof(self):
        with tempfile.TemporaryDirectory() as root:
            result, manifest, calls = self.perform(root)
        self.assertEqual((result["actual_derivation_attempts"], result["new_derivative_inputs"], calls), (16, 16, 16))
        self.assertEqual(len(manifest["rows"]), 16)
        self.assertEqual(result["status"], "COMPLETE_NEW_INPUTS")
        self.assertFalse(manifest["training_allowed"])

    def test_proof_write_failure_cannot_mark_complete(self):
        with tempfile.TemporaryDirectory() as root:
            result, manifest, calls = self.perform(root, fail_proof=True)
        self.assertEqual((result["actual_derivation_attempts"], result["new_derivative_inputs"], calls), (1, 0, 1))
        self.assertEqual(manifest["rows"][0]["preparation_status"], "FAILED_NO_RETRY")
        self.assertNotIn("derivation_sha256", manifest["rows"][0])
        self.assertEqual(sum(r["preparation_status"] == "NOT_STARTED_NO_RETRY" for r in manifest["rows"]), 15)

    def test_failure_receipt_error_does_not_erase_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            result, manifest, calls = self.perform(root, fail_proof=True, fail_failure_receipt=True)
        self.assertEqual(result["new_derivative_inputs"], 0)
        self.assertEqual(len(manifest["rows"]), 16)
        self.assertIn("mock failure receipt", manifest["rows"][0]["failure_receipt_write_error"])

    def test_any_collision_with_old_pcm_stops_before_observer_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            result, manifest, calls = self.perform(root, collision=True)
        self.assertEqual((calls, result["new_derivative_inputs"]), (1, 0))
        self.assertIn("FULL_INPUT_NOT_PREVIOUSLY_OBSERVED", result["failure"])
        self.assertEqual(len(manifest["rows"]), 16)


if __name__ == "__main__":
    unittest.main()
