from __future__ import annotations
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_kws_dual_track_v7 as v7


class V7RegistrationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = v7.load()
        cls.plan = v7.load_source_plan()

    def test_exact_brand_pronunciations_and_cross_negatives(self):
        self.assertEqual(self.value["candidates"]["VOSEN"]["ipa"], "/heɪ ˈvoʊ.sən/")
        self.assertEqual(self.value["candidates"]["MERO"]["ipa"], "/heɪ ˈmɛroʊ/")
        self.assertEqual(self.value["candidates"]["VOSEN"]["cross_trigger_negative"], "Hey Mero")
        self.assertEqual(self.value["candidates"]["MERO"]["cross_trigger_negative"], "Hey Vosen")

    def test_vosen_is_voiceless_s_not_voiced_z(self):
        contract = self.value["candidates"]["VOSEN"]["pronunciation_contract"]
        self.assertEqual(contract["medial_consonant"], "voiceless /s/")
        self.assertEqual(contract["forbidden_variant"], "voiced /z/")
        self.assertEqual(contract["source_ipa"], "heɪ vˈoʊsən")
        self.assertFalse(self.value["historical_campaigns"]["vosen_schwa16"]["v7_positive_eligible"])

    def test_historical_stop_is_not_training_admitted(self):
        history = self.value["historical_campaigns"]["mero_meh_44218734553"]
        self.assertEqual(history["terminal_status"], "STOP_FRESH_CORPUS_COVERAGE")
        self.assertEqual(history["final_training_rows"], 0)
        self.assertFalse(history["v7_training_eligible"])

    def test_paired_source_panel_has_exact_6_2_2_2_groups(self):
        groups = self.plan["source_selection"]["roles"]
        self.assertEqual([g["speaker_index"] for g in groups], list(range(496, 508)))
        from collections import Counter
        self.assertEqual(Counter(g["role"] for g in groups),
                         {"train": 6, "development": 2, "calibration": 2, "locked-test": 2})
        self.assertEqual(len(self.plan["cells"]), 12)
        self.assertEqual(self.plan["budget"]["tts_requests"], 144)
        self.assertEqual(self.plan["budget"]["total_high_level_model_requests"], 720)

    def test_paired_candidate_relative_cross_trigger_labels(self):
        cells = {c["id"]: c for c in self.plan["cells"]}
        self.assertEqual(cells["vosen-slow"]["class_by_candidate"], {"VOSEN": 1, "MERO": 0})
        self.assertEqual(cells["mero-slow"]["class_by_candidate"], {"VOSEN": 0, "MERO": 1})
        self.assertTrue(self.plan["paired_candidate_policy"]["same_original_pcm_sha256_referenced_in_both_candidate_manifests"])

    def test_no_voiced_z_vosen_positive_in_source_plan(self):
        for cell in self.plan["cells"]:
            if cell["id"].startswith("vosen"):
                self.assertEqual(cell["raw_ipa"], "heɪ vˈoʊsən")
                self.assertNotIn("z", cell["raw_ipa"])

    def test_models_and_training_state_are_independent(self):
        for item in self.value["candidates"].values():
            self.assertEqual(item["role"], "single-keyword detector")
            self.assertFalse(item["promotion_allowed"])
            self.assertIsNone(item["model_sha256"])
            self.assertIsNone(item["training_design_sha256"])

    def test_budgets_cannot_be_inferred_from_b28(self):
        for item in self.value["candidates"].values():
            self.assertIsNone(item["budget"]["optimizer_updates"])
            self.assertIsNone(item["budget"]["model_forwards"])
        self.assertFalse(self.value["migration"]["B28_one_shot_retry_allowed"])

    def test_mutated_pronunciation_rejected(self):
        bad = copy.deepcopy(self.value)
        bad["candidates"]["VOSEN"]["ipa"] = "/heɪ ˈvoʊzən/"
        with self.assertRaises(ValueError):
            v7.validate(bad)

    def test_mutated_voicing_contract_rejected(self):
        bad = copy.deepcopy(self.value)
        bad["candidates"]["VOSEN"]["pronunciation_contract"]["medial_consonant"] = "voiced /z/"
        with self.assertRaises(ValueError):
            v7.validate(bad)

    def test_shared_optimizer_or_weights_rejected(self):
        bad = copy.deepcopy(self.value)
        bad["shared_runtime"]["model_parameters_shared"] = True
        with self.assertRaises(ValueError):
            v7.validate(bad)

    def test_locked_test_mining_rejected(self):
        bad = copy.deepcopy(self.value)
        bad["shared_negative_pool_policy"]["locked_test_never_used_for_training_or_mining"] = False
        with self.assertRaises(ValueError):
            v7.validate(bad)

    def test_activation_or_promotion_rejected(self):
        bad = copy.deepcopy(self.value)
        bad["execution_enabled"] = True
        with self.assertRaises(ValueError):
            v7.validate(bad)
        bad = copy.deepcopy(self.value)
        bad["promotion"]["allowed"] = True
        with self.assertRaises(ValueError):
            v7.validate(bad)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            v7._unique_pairs([("x", 1), ("x", 2)])


if __name__ == "__main__":
    unittest.main()
