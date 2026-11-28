"""Source-only contracts for the pinned Kokoro V7 train/dev campaign."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "automation"))
import english_kws_dual_track_v7_recipe as recipe
import english_kws_dual_track_v7_once as once


class KokoroV7SourceContracts(unittest.TestCase):
    def test_frozen_source_panel_has_exactly_six_train_and_two_development_voices(self):
        source_plan, registration = recipe.source_documents()
        groups = recipe.source_groups()
        self.assertEqual(len(groups), 8)
        self.assertEqual(len({group["source_speaker_id"] for group in groups}), 8)
        self.assertEqual({group["role"] for group in groups}, {"train", "development"})
        self.assertEqual(sum(group["role"] == "train" for group in groups), 6)
        self.assertEqual(sum(group["role"] == "development" for group in groups), 2)
        self.assertEqual(registration["source_map_evidence"]["role_counts"],
                         {"train": 6, "development": 2})
        self.assertEqual(source_plan["split"], {"train": 6, "development": 2})

    def test_exact_96_rows_use_pinned_raw_ipa_and_candidate_relative_labels(self):
        frozen = recipe.recipe()
        self.assertEqual(frozen["source_count"], 96)
        self.assertEqual(len(frozen["rows"]), 96)
        self.assertEqual(len({row["id"] for row in frozen["rows"]}), 96)
        self.assertEqual(frozen["candidate_ids"], ["hey-vosen-v7", "hey-mero-v7"])
        for row in frozen["rows"]:
            self.assertTrue(row["raw_phonemes"])
            self.assertIn(row["requested_class"], (0, 1, 2))
            self.assertIn(row["voice_id"], {group["source_speaker_id"]
                                            for group in frozen["source_groups"]})
            self.assertFalse(row.get("actual_phonemes"))

    def test_model_and_voice_assets_are_pinned_and_raw_ipa_bypasses_g2p(self):
        self.assertEqual(recipe.MODEL["repository"], "hexgrad/Kokoro-82M")
        self.assertEqual(recipe.MODEL["revision"], "f3ff3571791e39611d31c381e3a41a3af07b4987")
        self.assertEqual(recipe.MODEL["checkpoint_sha256"],
                         "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4")
        self.assertIn("no G2P", recipe.MODEL["phoneme_path"])
        self.assertIn("no eSpeak", recipe.MODEL["phoneme_path"])
        self.assertEqual(recipe.MODEL["native_sample_rate"], 24000)
        self.assertEqual(recipe.MODEL["output_sample_rate"], 16000)

    def test_calibration_and_locked_test_are_reserved_but_not_generated(self):
        source_plan, _ = recipe.source_documents()
        reservations = source_plan["future_evaluation_voice_reservations"]
        self.assertEqual(len(reservations["calibration"]), 2)
        self.assertEqual(len(reservations["locked_test"]), 2)
        self.assertFalse(reservations["generated"])
        self.assertFalse(reservations["observed"])
        self.assertFalse(reservations["audio_opened"])
        self.assertTrue(all(group["role"] in ("train", "development")
                            for group in recipe.source_groups()))

    def test_source_check_is_inert_and_declares_no_training_or_retries(self):
        report = once.source_check()
        self.assertEqual(report["status"], "KOKORO_V7_SOURCE_PLAN_VALID_NO_AUDIO_OR_MODEL_EXECUTION")
        self.assertEqual(report["source_count"], 96)
        self.assertEqual(report["source_role_counts"], {"train": 6, "development": 2})
        self.assertEqual(report["tts_requests"], 96)
        self.assertEqual(report["observer_requests_per_provider"], 96)
        self.assertEqual(report["total_high_level_model_requests"], 480)
        self.assertEqual(report["new_kws_model_forwards"], 0)
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertEqual(report["retries"], 0)
        self.assertFalse(report["calibration_audio_generated"])
        self.assertFalse(report["locked_test_audio_generated"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
