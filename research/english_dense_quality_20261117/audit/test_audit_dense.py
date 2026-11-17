"""Focused fixtures for cache-only boundaries; no numerical model execution."""
from __future__ import annotations
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_dense as audit


class Store:
    def __init__(self, files):
        self.files = files

    def read(self, name):
        return self.files[name]


def worker_source():
    paths = [parent / "automation/english_dense_quality_once.py" for parent in Path(__file__).resolve().parents]
    source = next((path for path in paths if path.is_file()), None)
    if source is None:
        raise AssertionError("dense worker must be present in repository automation directory")
    return source.read_bytes()


def phases_fixture():
    final = dict(audit.EXPECTED_COUNTS)
    zero = dict.fromkeys(final, 0)
    trained = {**zero, "optimizer_updates_started": 96, "optimizer_updates_completed": 96,
               "torch_forward_calls_started": 576, "torch_forward_calls_completed": 576}
    names = ["WORKER_STARTED", "TRAINING_STARTED", "CANDIDATE_FROZEN",
             "EVALUATION_STARTED", "EVALUATION_COMPLETED", "COMPLETED"]
    phases = [{"phase_number": i + 1, "phase": name,
               "progress": dict(zero if i < 2 else trained if i < 4 else final),
               "candidate_sha256": None if i < 2 else "a" * 64} for i, name in enumerate(names)]
    frozen = {"phase_number": 3, "progress": trained, "candidate": {"sha256": "a" * 64},
              "snr5_model_calls_before_freeze": 0, "all_future_evaluation_uses_this_frozen_head": True}
    return phases, frozen, final


def gate_fixture():
    pairs = [{"condition": condition, "id": condition, "role": "development", "weak_class": 1,
              "regressed": False, "rescued": condition == "snr10", "candidate_error": False,
              "control_missing_in_observation": condition == "snr10", "introduced_event_errors": {}}
             for condition in ("clean", "snr10", "snr5")]
    coverage = [{"id": "v", "role": "development", "weak_class": 1},
                {"id": "m", "role": "development", "weak_class": 2}]
    return pairs, coverage


class DenseAuditFixtures(unittest.TestCase):
    def test_frozen_dependency_is_loaded_only_after_exact_identity(self):
        source = Path(audit.legacy.__file__).read_bytes()
        self.assertEqual(audit.ident(source), {"bytes": audit.FROZEN_AUDITOR_BYTES,
                                              "sha256": audit.FROZEN_AUDITOR_SHA})
        with mock.patch.object(audit.hashlib, "sha256", return_value=SimpleNamespace(hexdigest=lambda: "0" * 64)):
            with self.assertRaisesRegex(ValueError, "source SHA/bytes changed"):
                audit.load_frozen_auditor()

    def test_cache_payload_must_be_copied_not_regenerated_or_relabelled(self):
        source = Store({"cache/snr5.npz": b"frozen-five-db-cache"})
        output = Store({"cache/snr5.npz": source.read("cache/snr5.npz")})
        copied = {"cache/snr5.npz": {"source": "temporal", "member": "cache/snr5.npz",
                                      **audit.ident(source.read("cache/snr5.npz"))}}
        self.assertEqual(audit.verify_copied_members(output, copied, {"temporal": source}), 1)
        output.files["cache/snr5.npz"] = b"newly-recreated-cache"
        with self.assertRaises(ValueError):
            audit.verify_copied_members(output, copied, {"temporal": source})

    def test_undeclared_cache_source_cannot_enter_the_comparison(self):
        copied = {"cache/x": {"source": "fresh", "member": "x", "bytes": 1, "sha256": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "known copied input source"):
            audit.verify_copied_members(Store({}), copied, {"temporal": Store({})})

    def test_fixed_dense_source_contract_is_static_only(self):
        result = audit.source_only_dense_contract(worker_source())
        self.assertTrue(result["dense_positive_mean_ce_checked"])
        self.assertFalse(result["local_numerical_loss_or_gradient_execution"])

    def test_positive_sum_would_change_weighting_and_is_rejected(self):
        raw = worker_source()
        changed = raw.replace(b"terms.append(-logp[:, label].mean())", b"terms.append(-logp[:, label].sum())")
        self.assertNotEqual(changed, raw)
        with self.assertRaises(ValueError):
            audit.source_only_dense_contract(changed)

    def test_consumed_temporal_worker_cannot_be_called_again(self):
        raw = worker_source()
        changed = raw.replace(b"    import numpy as np", b"    common.worker(args, plan)\n    import numpy as np", 1)
        self.assertNotEqual(changed, raw)
        with self.assertRaisesRegex(ValueError, "cannot be re-executed"):
            audit.source_only_dense_contract(changed)

    def test_frontend_import_is_rejected_even_if_counters_claim_zero(self):
        raw = worker_source()
        changed = raw.replace(b"    import numpy as np", b"    import onnxruntime\n    import numpy as np", 1)
        with self.assertRaisesRegex(ValueError, "cannot import a frontend"):
            audit.source_only_dense_contract(changed)

    def test_six_phase_zero_frontend_execution(self):
        phases, frozen, final = phases_fixture()
        self.assertTrue(audit.verify_dense_phases(phases, frozen, final, "a" * 64)["all_frontend_counters_zero"])
        phases[1]["progress"]["new_embedding_calls_started"] = 1
        with self.assertRaises(ValueError):
            audit.verify_dense_phases(phases, frozen, final, "a" * 64)

    def test_extra_candidate_pass_cannot_hide_in_complete_counters(self):
        phases, frozen, final = phases_fixture()
        final["head_case_passes_started"] = final["head_case_passes_completed"] = 265
        for phase in phases[4:]:
            phase["progress"] = dict(final)
        with self.assertRaisesRegex(ValueError, "exact final"):
            audit.verify_dense_phases(phases, frozen, final, "a" * 64)

    def test_early_evaluation_or_changed_frozen_head_is_rejected(self):
        phases, frozen, final = phases_fixture()
        phases[2]["progress"]["head_case_passes_started"] = 1
        with self.assertRaises(ValueError):
            audit.verify_dense_phases(phases, frozen, final, "a" * 64)
        phases, frozen, final = phases_fixture()
        phases[3]["candidate_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            audit.verify_dense_phases(phases, frozen, final, "a" * 64)

    def test_training_success_cannot_replace_a_development_miss_rescue(self):
        pairs, coverage = gate_fixture()
        pairs[1]["role"] = "train"
        gate = audit.legacy.strict_gate(pairs, coverage)
        self.assertFalse(gate["accepted"])
        self.assertIn("NO_SNR10_DEVELOPMENT_MISS_RESCUED", gate["reasons"])

    def test_noisy_rescue_cannot_excuse_a_clean_background_event(self):
        pairs, coverage = gate_fixture()
        pairs[0]["regressed"] = True
        pairs[0]["introduced_event_errors"] = {"weak_negative_disagreement_count": 1}
        gate = audit.legacy.strict_gate(pairs, coverage)
        self.assertFalse(gate["accepted"])
        self.assertIn("CLEAN_CASE_REGRESSION", gate["reasons"])
        self.assertIn("NEW_PREFIX_WRONG_EXTRA_OR_NEGATIVE_EVENT", gate["reasons"])

    def test_auditor_imports_no_numeric_or_network_runtime(self):
        modules = set()
        for node in ast.walk(ast.parse(Path(audit.__file__).read_text())):
            if isinstance(node, ast.Import):
                modules.update(name.name.split(".")[0] for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add((node.module or "").split(".")[0])
        self.assertFalse(modules & {"numpy", "torch", "onnx", "onnxruntime", "subprocess", "requests", "urllib", "soundfile"})


if __name__ == "__main__":
    unittest.main()
