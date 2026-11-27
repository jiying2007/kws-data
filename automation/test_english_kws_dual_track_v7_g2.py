from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUTOMATION = ROOT / "automation"
sys.path.insert(0, str(AUTOMATION))

import english_kws_dual_track_v7_recipe as recipe
import english_kws_dual_track_v7_once as worker

PLAN = json.loads((AUTOMATION / "english_kws_dual_track_v7_execution_plan.json").read_text())


class V7G2ContractTests(unittest.TestCase):
    def test_frozen_roster_and_role_split(self):
        frozen = recipe.recipe()
        groups = recipe.source_groups()
        self.assertEqual(recipe.SOURCE_COUNT, 144)
        self.assertEqual(len(groups), 12)
        self.assertEqual(len(frozen["rows"]), 144)
        self.assertEqual(
            {role: sum(group["role"] == role for group in groups) for role in recipe.ROLES},
            {"train": 6, "development": 2, "calibration": 2, "locked-test": 2},
        )
        self.assertTrue(all(
            sum(row["source_group_id"] == group["source_group_id"] for row in frozen["rows"]) == 12
            for group in groups
        ))

    def test_candidate_relative_targets_and_cross_negatives(self):
        frozen = recipe.recipe()
        for group in frozen["source_groups"]:
            rows = [row for row in frozen["rows"] if row["source_group_id"] == group["source_group_id"]]
            self.assertEqual(sum(row["candidate_labels"]["VOSEN"] == 1 for row in rows), 3)
            self.assertEqual(sum(row["candidate_labels"]["MERO"] == 1 for row in rows), 3)
            self.assertEqual(sum(row["kind"] == "contrast-negative" for row in rows), 4)
            self.assertEqual(sum(row["kind"] == "ordinary-negative" for row in rows), 2)
            self.assertEqual(sum(row["requested_class"] == 1 for row in rows), 3)
            self.assertEqual(sum(row["requested_class"] == 2 for row in rows), 3)
            self.assertEqual(sum(row["requested_class"] == 0 for row in rows), 6)

    def test_all_requests_are_unique_and_synthetic_only(self):
        frozen = recipe.recipe()
        ids = [row["id"] for row in frozen["rows"]]
        request_hashes = [row["request_sha256"] for row in frozen["rows"]]
        self.assertEqual(len(set(ids)), 144)
        self.assertEqual(len(set(request_hashes)), 144)
        self.assertTrue(all(row["human_gold"] is False for row in frozen["rows"]))
        self.assertTrue(all(row["generation_mode"] == "direct-phoneme-ids" for row in frozen["rows"]))
        self.assertEqual(frozen["new_tts_calls_max"], 144)
        self.assertEqual(frozen["new_calls_per_observer_max"], 144)
        self.assertEqual(frozen["retry_count"], 0)
        self.assertEqual(frozen["kws_forward_calls"], 0)
        self.assertEqual(frozen["optimizer_updates"], 0)

    def test_pinned_models_and_license_boundary(self):
        self.assertEqual(recipe.MODEL["revision"], "de3dcfdf1912bb49726dc4aa11c26017ce2ac62a")
        self.assertEqual(recipe.MODEL["model_sha256"],
            "9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f")
        self.assertEqual(recipe.MODEL_CARD["md5"], "cdeac934f1154489924a071470b22365")
        self.assertEqual(recipe.MODEL_CARD["dataset_license"], "CC BY 4.0")
        self.assertEqual(recipe.ENGINE["commit"], "639388b6317fc4731e91d53da42aea68fd4166ff")
        self.assertEqual(recipe.ENGINE["license"], "GPL-3.0-or-later")
        self.assertEqual(set(recipe.PROVIDERS), set(recipe.OBSERVERS))
        self.assertEqual(recipe.MODEL["num_speakers"], 904)

    def test_execution_plan_is_g2_only_and_one_shot(self):
        self.assertTrue(PLAN["enabled"])
        self.assertEqual(PLAN["next_main_before"], "6691d49a42066f6b832f2c4e7c899eb4752f3550")
        self.assertEqual(PLAN["maximum_tts_calls"], 144)
        self.assertEqual(PLAN["maximum_calls_per_observer"], 144)
        self.assertEqual(PLAN["maximum_total_model_calls"], 720)
        self.assertEqual(PLAN["retry_count"], 0)
        self.assertFalse(PLAN["training_allowed"])
        self.assertFalse(PLAN["promotion_allowed"])
        self.assertEqual(PLAN["new_kws_calls"], 0)
        self.assertEqual(PLAN["optimizer_updates"], 0)
        self.assertEqual(PLAN["positive_min_samples"], 6400)
        self.assertEqual(PLAN["positive_max_samples"], 28672)
        self.assertEqual(PLAN["source_plan_sha256"], recipe.recipe()["source_plan_sha256"])
        worker.validate_plan(PLAN, allow_autofill=True)

    def test_preflight_is_metadata_only(self):
        source = (AUTOMATION / "english_kws_dual_track_v7_preflight.py").read_text()
        self.assertIn("model_weights_downloaded", source)
        self.assertIn("audio_downloaded", source)
        self.assertIn("tts_calls", source)
        self.assertNotIn("hf_hub_download", source)
        self.assertNotIn("PiperVoice.load", source)
        self.assertNotIn("phoneme_ids_to_audio", source)


if __name__ == "__main__":
    unittest.main()
