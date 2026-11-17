"""Small stdlib fixtures for corruption, event gates and evidence boundaries.

These tests contain synthetic saved records only. They never import or execute
any neural model, optimizer, tensor framework, feature model or C head.
"""
from __future__ import annotations
import ast
import copy
import io
import json
import math
from pathlib import Path
import struct
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_temporal as audit


class MemoryArtifact:
    def __init__(self, records, reports):
        self.files = {
            "predictions/fixture-frames.jsonl": b"\n".join(audit.canonical(r) for r in records) + b"\n",
            "predictions/fixture-events.json": audit.canonical(reports),
        }

    def read(self, name):
        return self.files[name]

    def json(self, name):
        return audit.json_bytes(self.read(name))


def event_report(row, events):
    calc = audit.event_observation(row, events)
    return {"id": row["id"], "role": row["role"], "weak_class": row["weak_class"],
            "events": events, "matched_intent_after_clip_start": calc["matched"],
            "prefix_events": [e for e in events if e["sample"] <= row["source"]["clip_start_sample"]],
            "wrong_keyword_after_clip_start": [e for e in events if e["sample"] > row["source"]["clip_start_sample"]
                and row["weak_class"] and e["keyword"] != row["weak_class"]],
            "extra_same_keyword_events": calc["extra_count"],
            "no_matching_event_in_observed_context": calc["missing_in_observation"],
            "weak_negative_intent_events": calc["weak_negative_disagreement_count"],
            "research_error": calc["research_error"],
            "event_minus_clip_eof_samples": calc["matched_event_minus_clip_eof_samples"],
            "word_end_latency": "NOT_MEASURED", "independent_acoustic_evaluation": False,
            "eof_prefix_event_count": calc["eof_prefix_event_count"],
            "no_event_status": calc["observation_no_event_status"]}


def trace_fixture(pattern, event_indices):
    clocks = list(range(16000, 16000 + (15 + len(pattern)) * 1280, 1280))
    row = {"id": "fixture-positive", "role": "development", "weak_class": 1,
           "observed_samples": clocks[-1], "source": {"clip_start_sample": 35200,
           "original_clip_eof_sample": min(40000, clocks[-1])}}
    frames = []
    for i, sample in enumerate(clocks):
        logits = [0.0, 0.0, 0.0]
        probabilities = [0.0, 0.0, 0.0]
        if i >= 15:
            logits[pattern[i - 15]] = 4.0
            exp = [math.exp(v - 4.0) for v in logits]
            probabilities = [audit.f32(v / math.fsum(exp)) for v in exp]
        frames.append({"id": row["id"], "sample": sample, "ready": i >= 15,
                       "keyword": 1 if i in event_indices else 0,
                       "logits": logits, "probabilities": probabilities})
    events = [{"sample": clocks[i], "keyword": 1} for i in event_indices]
    return row, clocks, frames, event_report(row, events)


def read_fixture(row, clocks, frames, report):
    return audit.read_trace(MemoryArtifact(frames, [report]), "fixture", [row], {row["id"]: clocks})


def minimal_npy(dtype, shape, payload):
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode("ascii") + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload


def zipped(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        for name, raw in files.items():
            out.writestr(name, raw)
    return buffer.getvalue()


def passing_pairs():
    rows = []
    for condition in ("clean", "snr10", "snr5"):
        rows.append({"condition": condition, "id": condition, "role": "development", "weak_class": 1,
                     "regressed": False, "rescued": condition == "snr10", "candidate_error": False,
                     "control_missing_in_observation": condition == "snr10", "introduced_event_errors": {}})
    coverage = [{"id": "v", "role": "development", "weak_class": 1},
                {"id": "m", "role": "development", "weak_class": 2}]
    return rows, coverage


def phase_fixture():
    final = {"new_embedding_calls_started": 2649, "new_embedding_calls_completed": 2649,
             "onnx_forward_started": 5298, "onnx_forward_completed": 5298,
             "head_case_passes_started": 406, "head_case_passes_completed": 406,
             "head_ready_outputs": 15015, "optimizer_updates_started": 96, "optimizer_updates_completed": 96,
             "torch_forward_calls_started": 576, "torch_forward_calls_completed": 576,
             "new_training_frontend_calls": 0, "new_tts_asr_phone_calls": 0, "retry_count": 0}
    names = ["WORKER_STARTED", "TRAINING_STARTED", "CANDIDATE_FROZEN", "OLD_CONDITION_EVALUATION_STARTED",
             "OLD_CONDITION_EVALUATION_COMPLETED", "SNR5_DATA_PREPARATION_STARTED", "SNR5_DATA_PREPARATION_COMPLETED",
             "SNR5_FEATURES_STARTED", "SNR5_FEATURES_COMPLETED", "SNR5_EVALUATION_STARTED",
             "SNR5_EVALUATION_COMPLETED", "COMPLETED"]
    phases = []
    for i, name in enumerate(names):
        counts = dict.fromkeys(final, 0)
        if i >= 2:
            for key in ("optimizer_updates_started", "optimizer_updates_completed", "torch_forward_calls_started", "torch_forward_calls_completed"):
                counts[key] = final[key]
        if i >= 4:
            counts.update(head_case_passes_started=193, head_case_passes_completed=193, head_ready_outputs=10263)
        if i >= 8:
            counts.update(new_embedding_calls_started=2649, new_embedding_calls_completed=2649,
                          onnx_forward_started=5298, onnx_forward_completed=5298)
        if i >= 10:
            counts = dict(final)
        phases.append({"phase_number": i + 1, "phase": name, "progress": counts,
                       "candidate_sha256": None if i < 2 else "a" * 64})
    frozen = {"phase_number": 3, "progress": phases[2]["progress"], "candidate": {"sha256": "a" * 64},
              "snr5_model_calls_before_freeze": 0, "all_future_evaluation_uses_this_frozen_head": True}
    return phases, frozen, final


class SavedEvidenceFixtures(unittest.TestCase):
    def test_parser_rejects_duplicate_and_nonfinite_json(self):
        for raw in (b'{"x":1,"x":2}', b'{"loss":NaN}'):
            with self.assertRaises(ValueError):
                audit.json_bytes(raw)

    def test_archive_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            audit.Archive(zipped({"../outside.txt": b"x"}))

    def test_npy_rejects_object_and_nonfinite_saved_arrays(self):
        for raw in (minimal_npy("|O", (1,), b"12345678"),
                    minimal_npy("<f4", (1,), struct.pack("<f", float("nan")))):
            with self.assertRaises(ValueError):
                audit.npy(raw)

    def test_condition_cache_rejects_a_hidden_clock_gap(self):
        rows = [{"id": "a"}]
        raw = zipped({"e000.npy": minimal_npy("<f4", (2, 96), b"\0" * (2 * 96 * 4)),
                      "t000.npy": minimal_npy("<u8", (2,), struct.pack("<QQ", 16000, 18560))})
        with self.assertRaises(ValueError):
            audit.audit_condition_cache(raw, rows, {"a": [16000, 17280]}, "fixture")

    def test_three_consecutive_frames_required(self):
        row, times, frames, report = trace_fixture([1, 0, 1, 1, 0], [])
        metrics, trace = read_fixture(row, times, frames, report)
        self.assertTrue(metrics[0]["missing_in_observation"])
        self.assertEqual(trace["maximum_softmax_probability_error"], 0.0)

    def test_cooldown_boundary_is_causal_and_exact(self):
        row, times, frames, report = trace_fixture([1] * 20, [17, 32])
        metrics, _ = read_fixture(row, times, frames, report)
        self.assertEqual(metrics[0]["event_count"], 2)
        self.assertEqual(metrics[0]["extra_count"], 1)
        corrupted = copy.deepcopy(frames)
        corrupted[30]["keyword"] = 1
        with self.assertRaises(ValueError):
            read_fixture(row, times, corrupted, report)

    def test_recorded_softmax_and_early_readiness_are_checked(self):
        row, times, frames, report = trace_fixture([1, 1, 1], [17])
        read_fixture(row, times, frames, report)
        bad_prob = copy.deepcopy(frames)
        bad_prob[16]["probabilities"] = [0.0, 1.0, 0.0]
        with self.assertRaises(ValueError):
            read_fixture(row, times, bad_prob, report)
        bad_ready = copy.deepcopy(frames)
        bad_ready[14]["ready"] = True
        with self.assertRaises(ValueError):
            read_fixture(row, times, bad_ready, report)

    def test_missing_or_extra_saved_frames_rejected(self):
        row, times, frames, report = trace_fixture([1, 1, 1], [17])
        for changed in (frames[:-1], frames + [frames[-1]]):
            with self.assertRaises(ValueError):
                read_fixture(row, times, changed, report)

    def test_missed_observation_stays_censored(self):
        row, times, frames, report = trace_fixture([0, 0, 0, 0], [])
        metrics, _ = read_fixture(row, times, frames, report)
        self.assertEqual(metrics[0]["observation_no_event_status"], "CENSORED_NOT_FRR")
        self.assertTrue(metrics[0]["original_eof_no_matching_event_censored"])
        report["no_event_status"] = "FALSE_REJECT"
        with self.assertRaises(ValueError):
            read_fixture(row, times, frames, report)

    def test_draws_cannot_switch_to_development_or_snr5(self):
        clean = [{"id": "c" + str(i), "role": "train", "weak_class": i} for i in range(3)]
        noisy = [{"id": "n" + str(i), "role": "train", "weak_class": i} for i in range(3)]
        one = [{"condition": condition, "id": row["id"], "weak_class": row["weak_class"], "role": "train"}
               for condition, rows in (("clean", clean), ("snr10", noisy)) for row in rows]
        draws = [copy.deepcopy(one) for _ in range(96)]
        self.assertEqual(audit.verify_draws(draws, draws, clean, noisy)["bag_forwards"], 576)
        for key, value in (("condition", "snr5"), ("role", "development")):
            changed = copy.deepcopy(draws)
            changed[55][2][key] = value
            with self.assertRaises(ValueError):
                audit.verify_draws(changed, changed, clean, noisy)

    def test_gate_rejects_regression_despite_a_development_rescue(self):
        pairs, coverage = passing_pairs()
        self.assertTrue(audit.strict_gate(pairs, coverage)["accepted"])
        pairs[-1]["regressed"] = True
        gate = audit.strict_gate(pairs, coverage)
        self.assertFalse(gate["accepted"])
        self.assertIn("SNR5_CASE_REGRESSION", gate["reasons"])

    def test_gate_rejects_new_safety_event_on_already_bad_case(self):
        pairs, coverage = passing_pairs()
        pairs[0]["introduced_event_errors"] = {"wrong_count": 1}
        self.assertIn("NEW_PREFIX_WRONG_EXTRA_OR_NEGATIVE_EVENT", audit.strict_gate(pairs, coverage)["reasons"])

    def test_gate_requires_actual_prior_miss_and_both_development_keywords(self):
        pairs, coverage = passing_pairs()
        pairs[1]["control_missing_in_observation"] = False
        gate = audit.strict_gate(pairs, coverage[:1])
        self.assertIn("NO_SNR10_DEVELOPMENT_MISS_RESCUED", gate["reasons"])
        self.assertIn("SNR5_DEVELOPMENT_KEYWORD_COVERAGE_INSUFFICIENT", gate["reasons"])

    def test_nested_admission_cannot_be_silently_enabled(self):
        with self.assertRaises(ValueError):
            audit.reject_formal_claims({"receipt": [{"trusted_negative": True}]})

    def test_candidate_freeze_must_precede_new_condition_models(self):
        phases, frozen, final = phase_fixture()
        audit.verify_phases(phases, frozen, final, "a" * 64)
        phases[1]["progress"]["new_embedding_calls_started"] = 1
        with self.assertRaises(ValueError):
            audit.verify_phases(phases, frozen, final, "a" * 64)

    def test_candidate_cannot_change_between_evaluation_conditions(self):
        phases, frozen, final = phase_fixture()
        phases[9]["candidate_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            audit.verify_phases(phases, frozen, final, "a" * 64)

    def test_auditor_has_only_stdlib_imports(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        forbidden = {"numpy", "torch", "onnx", "onnxruntime", "subprocess", "requests", "urllib", "soundfile"}
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(name.name.split(".")[0] for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
        self.assertFalse(imports & forbidden)

    def test_frozen_worker_source_contract_without_execution(self):
        here = Path(__file__).resolve()
        candidates = [parent / "automation/english_temporal_quality_once.py" for parent in here.parents]
        worker = next((path for path in candidates if path.is_file()), None)
        self.assertIsNotNone(worker, "worker source must be in repository automation directory")
        raw = worker.read_bytes()
        result = audit.source_only_loss_contract(raw)
        self.assertTrue(result["exact_temporal_amin_mean_pooling_checked"])
        changed = raw.replace(b"torch.amin(logp[:, label].unfold(0, 3, 1), dim=1)",
                              b"torch.amax(logp[:, label].unfold(0, 3, 1), dim=1)")
        self.assertNotEqual(changed, raw)
        with self.assertRaises(ValueError):
            audit.source_only_loss_contract(changed)


if __name__ == "__main__":
    unittest.main()
