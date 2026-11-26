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

    def test_exact_brand_pronunciations_and_cross_negatives(self):
        self.assertEqual(self.value["candidates"]["VOSEN"]["ipa"], "/heɪ ˈvoʊ.sən/")
        self.assertEqual(self.value["candidates"]["MERO"]["ipa"], "/heɪ ˈmɛroʊ/")
        self.assertEqual(self.value["candidates"]["VOSEN"]["cross_trigger_negative"], "Hey Mero")
        self.assertEqual(self.value["candidates"]["MERO"]["cross_trigger_negative"], "Hey Vosen")

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
        for key, value in (("execution_enabled", True),):
            bad = copy.deepcopy(self.value)
            bad[key] = value
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
