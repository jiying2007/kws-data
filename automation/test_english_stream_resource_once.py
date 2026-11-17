"""Offline fixed-input and diagnostic evidence checks; no model/network calls."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("resource_worker", HERE / "english_stream_resource_once.py")
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class FixedInputTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((HERE / "english_stream_resource_inputs.json").read_text(encoding="utf-8"))
        # Source commit values are intentionally invented; this test calls no
        # GitHub guard or execution path, and never claims source validation.
        self.plan["pipeline_commit"] = "a" * 40
        self.plan["main_before"] = "b" * 40

    def test_saved_track_geometry_and_counts_validate(self):
        WORKER.validate_plan(self.plan)
        tracks = self.plan["actual_research_tracks"]
        self.assertEqual(sum(t["frames"] for t in tracks), 512000)
        self.assertEqual([len(t["fragments"]) for t in tracks], [3, 5])

    def test_source_pool_cannot_be_relabelled_unexposed(self):
        self.plan["source_pool_is_previously_exposed"] = False
        with self.assertRaisesRegex(ValueError, "exposure"):
            WORKER.validate_plan(self.plan)

    def test_trusted_negative_promotion_rejected(self):
        self.plan["actual_research_tracks"][0]["trusted_negative"] = True
        with self.assertRaisesRegex(ValueError, "resource-only"):
            WORKER.validate_plan(self.plan)

    def test_fragment_gap_rejected_before_any_download(self):
        self.plan["actual_research_tracks"][0]["fragments"][1]["track_start_sample"] += 1
        with self.assertRaisesRegex(ValueError, "geometry"):
            WORKER.validate_plan(self.plan)

    def test_fragment_path_traversal_rejected(self):
        self.plan["actual_research_tracks"][0]["fragments"][0]["source_id"] = "../outside.flac"
        with self.assertRaisesRegex(ValueError, "fragment path"):
            WORKER.validate_plan(self.plan)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        # Invented 16+63 node metadata preserves the pinned graph dimensions for
        # validator tests only. These are not measured nodes or model outputs.
        self.exported = {"graphs": {
            "melspectrogram.onnx": {"nodes": [{"op": "Conv"} for _ in range(16)]},
            "embedding_model.onnx": {"nodes": [{"op": "LeakyRelu"} for _ in range(63)]}}}
        self.measured = {"measurement_completed": True, "score_observer_enabled": False,
                         "input_samples": 256000, "embeddings": 188, "ready_outputs": 173,
                         "events": 0, "event_digest_fnv1a64": "cbf29ce484222325",
                         "resets": 0, "eof_padding_samples": 0}
        self.baseline = copy.deepcopy(self.measured)
        rows = []
        for graph, op, count in (("mel", "Conv", 16), ("embedding", "LeakyRelu", 63)):
            rows.append({"graph": graph, "node": -1, "op": -1, "calls": 188,
                         "wall_seconds": .01, "process_cpu_seconds": .01})
            rows.extend({"graph": graph, "node": n, "op": WORKER.OP[op], "calls": 188,
                         "wall_seconds": .02, "process_cpu_seconds": .02} for n in range(count))
        self.profile = {"schema": "english-graph-diagnostic-v1", "valid": True,
                        "instrumented": True, "headline_performance": False, "rows": rows}

    def verify(self):
        WORKER.verify_diagnostic(self.profile, self.measured, self.baseline, self.exported)

    def test_complete_invented_metadata_is_accepted(self):
        self.verify()

    def test_missing_node_rejected(self):
        self.profile["rows"].pop()
        with self.assertRaisesRegex(ValueError, "complete diagnostic"):
            self.verify()

    def test_wrong_operator_rejected(self):
        self.profile["rows"][1]["op"] = 999
        with self.assertRaisesRegex(ValueError, "node identity"):
            self.verify()

    def test_wrong_call_count_rejected(self):
        self.profile["rows"][1]["calls"] = 187
        with self.assertRaisesRegex(ValueError, "call count"):
            self.verify()

    def test_event_count_change_rejected(self):
        self.measured["events"] = 1
        with self.assertRaisesRegex(ValueError, "mismatch: events"):
            self.verify()

    def test_event_digest_change_rejected_even_with_same_count(self):
        self.measured["event_digest_fnv1a64"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "mismatch: event_digest"):
            self.verify()

    def test_duplicate_node_cannot_replace_missing_node(self):
        self.profile["rows"][-1] = copy.deepcopy(self.profile["rows"][-2])
        with self.assertRaisesRegex(ValueError, "node identity"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
