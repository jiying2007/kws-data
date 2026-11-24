"""Metadata-only independence and immutability tests for conditional B23."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).with_name("english_v6_b23_source_metadata.py")
SPEC = importlib.util.spec_from_file_location("b23_metadata_test_target", PATH)
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.hashes = metadata.build_documents()
        cls.previous, cls.old, cls.history, cls.manifest, _, _ = metadata.inputs()

    def test_entire_original_and_reserve_exclusion(self):
        ancestry = self.documents["ancestry.json"]
        for key in ("all_pcm_sha256", "all_source_groups", "canonical_people",
                    "synthetic_excluded_people", "natural_excluded_people"):
            self.assertLessEqual(set(self.history[key]), set(ancestry[key]), key)
        self.assertEqual(len(ancestry["all_pcm_sha256"]), len(self.history["all_pcm_sha256"]) + 120)
        all_b22 = set(ancestry["B22_exposed_primary_people"]) | set(ancestry["B22_sealed_replication_people"])
        self.assertEqual(len(all_b22), 72)
        for key in ("synthetic_excluded_people", "natural_excluded_people"):
            self.assertLessEqual(all_b22, set(ancestry[key]))
        self.assertEqual(ancestry["prior_never_reclaimed_reserves"], self.history["additional_never_reclaimed_reserves"])

    def test_72_people_disjoint_across_every_cohort(self):
        all_people = []
        a, ancestry = self.documents["assessment.json"], self.documents["ancestry.json"]
        for panel in ("primary", "replication"):
            synthetic = [self.previous.canonical(p) for p in a[panel]["synthetic"]["speaker_id_map"].values()]
            natural = a[panel]["natural"]["speakers"]
            self.assertEqual((len(synthetic), len(natural)), (24, 12))
            self.assertFalse(set(synthetic) & set(ancestry["synthetic_excluded_people"]))
            self.assertFalse(set(natural) & set(ancestry["natural_excluded_people"]))
            all_people.extend(synthetic + natural)
        self.assertEqual(len(all_people), 72)
        self.assertEqual(len(set(all_people)), 72)

    def test_unchanged_numerical_and_source_rules(self):
        a = self.documents["assessment.json"]
        for key in ("source_gate", "gains", "quality_criteria", "quality_assessment_sha256",
                    "observer_scientific_contract", "maximum_native_observer_calls",
                    "maximum_primary_KWS_forwards", "maximum_public_Piper_synthesize_calls",
                    "maximum_natural_original_decodes"):
            self.assertEqual(a[key], self.old[key], key)
        self.assertEqual(a["training"]["control_pt_sha256"], metadata.CONTROL_PT)
        self.assertEqual(a["training"]["design_sha256"], metadata.DESIGN_SHA)
        self.assertTrue(a["original_B22_NO_PROMOTION_required_before_any_new_budget"])
        for key in ("execution_authorized", "promotion_allowed", "replication_opened",
                    "threshold_search", "training_allowed_for_sources", "B22_quality_results_read"):
            self.assertIs(a[key], False, key)

    def test_whole_originals_and_fixed_cells(self):
        for panel in ("primary", "replication"):
            rows = self.documents[panel + ".json"]["rows"]
            self.assertEqual(len(rows), 96)
            self.assertEqual(sum(r["requested_class"] == 2 for r in rows), 48)
            self.assertEqual(sum(r["requested_class"] == 0 for r in rows), 48)
            self.assertEqual({(r["cell_id"], r["length_scale"]) for r in rows},
                             {("mero-fast", 1.4), ("mero-slow", 1.8), ("question", 1.0), ("request", 1.08)})
            natural = self.documents["natural-" + panel + ".json"]
            self.assertEqual(len(natural["roster"]), 24)
            self.assertEqual(len(set(natural["selected_utterance_ids"])), 24)
            self.assertEqual(natural["utterance_selection"], self.old[panel]["natural"]["utterance_selection"])
            self.assertTrue(all(r["training_allowed"] is False and r["observed_class"] is None for r in rows))

    def test_frozen_files_reproduce_without_hash_alias_mutation(self):
        before = metadata.encoded(self.documents["assessment.json"])
        report = metadata.validate()
        self.assertEqual(report["status"], "PASS_METADATA_ONLY")
        self.assertEqual(report["hashes"][metadata.DESIGN_PATH], metadata.DESIGN_SHA)
        self.assertEqual(before, metadata.encoded(self.documents["assessment.json"]))
        fresh, _ = metadata.build_documents()
        self.assertEqual(before, metadata.encoded(fresh["assessment.json"]))

    def test_frozen_roster_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=metadata.ROOT) as temp:
            path = Path(temp)
            for name, value in self.documents.items():
                (path / name).write_bytes(metadata.encoded(value))
            altered = copy.deepcopy(self.documents["assessment.json"])
            altered["primary"]["natural"]["speakers"][0] = "1"
            (path / "assessment.json").write_bytes(metadata.encoded(altered))
            # validate checks bytes before adding generated files to its inventory.
            with mock.patch.object(metadata, "PANELS", path):
                with self.assertRaisesRegex(ValueError, "complete deterministic frozen B23 metadata: assessment.json"):
                    metadata.validate()

    def test_omitted_original_source_is_rejected(self):
        args = list(metadata.inputs())
        args[3] = copy.deepcopy(args[3])
        args[3]["rows"].pop()
        with mock.patch.object(metadata, "inputs", return_value=tuple(args)):
            with self.assertRaisesRegex(ValueError, "complete original120"):
                metadata.build_documents()

    def test_metadata_only_io_and_imports(self):
        original_import, original_read = builtins.__import__, Path.read_bytes
        denied = {"torch", "numpy", "torchaudio", "transformers", "piper", "soundfile",
                  "onnxruntime", "requests", "httpx"}
        reads = []
        def guarded_import(name, *args, **kwargs):
            self.assertNotIn(name.split(".")[0], denied)
            return original_import(name, *args, **kwargs)
        def guarded_read(path):
            self.assertIn(path.suffix, {".py", ".json"})
            reads.append(path)
            return original_read(path)
        with mock.patch("builtins.__import__", side_effect=guarded_import), \
                mock.patch.object(Path, "read_bytes", guarded_read):
            report = metadata.validate()
        self.assertTrue(reads)
        for key in ("source_calls", "observer_calls", "kws_forwards", "optimizer_updates"):
            self.assertEqual(report[key], 0)
        self.assertFalse(report["source_execution_authorized"])
        self.assertFalse(denied & set(sys.modules))


if __name__ == "__main__":
    unittest.main()
