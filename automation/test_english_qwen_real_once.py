"""No-model checks for one-time English Qwen hosted capture."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_qwen_real_once as pilot
import multilingual_campaign as market


class OneTimeEnglishPilotTests(unittest.TestCase):
    def test_frozen_launch_is_source_only_by_default(self):
        claim = pilot.read_trigger()
        self.assertEqual(claim["max_tts_calls"], 10)
        self.assertFalse(claim["training_admitted"])
        with mock.patch.object(pilot, "materialize_model",
                               side_effect=AssertionError("model download forbidden")):
            self.assertEqual(pilot.main([]), 0)
        self.assertEqual(claim["wake_phrases"], ["Hey Vosen", "Hey Mero"])

    def test_pilot_plan_has_real_preset_ancestry_and_no_asr_pseudopins(self):
        first = pilot.pilot_plan("a" * 64)
        self.assertEqual(first, pilot.pilot_plan("a" * 64))
        self.assertEqual(len(first["cells"]), 10)
        self.assertEqual({x["source_family_id"] for x in first["cells"]},
                         {"qwen3-preset-ryan", "qwen3-preset-aiden"})
        self.assertEqual(set(x["role"] for x in first["cells"]),
                         {"train-candidate", "development-only"})
        self.assertEqual([x["text_requested"] for x in first["cells"][:5]],
                         [x[0] for x in pilot.PROMPTS])
        self.assertEqual({x["requested_keyword_index"]
                          for x in first["cells"] if x["intent"] == "wake"}, {1, 2})
        self.assertEqual(sum(x["intent"] == "near_miss" for x in first["cells"]), 4)
        self.assertEqual(first["asr_calls"], 0)
        self.assertNotIn("asr_pins", first)
        self.assertFalse(first["training_admitted"])
        self.assertFalse(first["human_gold"])
        with self.assertRaises(ValueError):
            pilot.pilot_plan("unknown-model-lock")

    def test_trigger_is_exact_and_refuses_modified_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            trigger = Path(tmp) / "trigger.json"
            original = pilot.TRIGGER
            with mock.patch.object(pilot, "TRIGGER", trigger):
                valid = market.read(original)
                for field, changed in [
                    ("max_tts_calls", 12),
                    ("wake_phrases", ["Hey Mero", "Hey Vosen"]),
                    ("training_admitted", True),
                    ("model_revision", "0" * 40),
                ]:
                    claim = copy.deepcopy(valid)
                    claim[field] = changed
                    trigger.write_bytes(market.canonical(claim) + b"\n")
                    with self.assertRaisesRegex(ValueError, "TRIGGER_POLICY_CHANGED"):
                        pilot.read_trigger()
                trigger.write_text('{"schema":"x","schema":"y"}')
                with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
                    pilot.read_trigger()

    def test_execute_rejects_wrong_runner_without_touching_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "should-not-exist"
            with mock.patch.dict("os.environ", {
                    "GITHUB_REPOSITORY": "other/repo",
                    "GITHUB_REF": "refs/heads/main",
                    "GITHUB_RUN_ATTEMPT": "1"}, clear=False):
                with self.assertRaisesRegex(ValueError, "GITHUB_SINGLE_EXECUTION_ONLY"):
                    pilot.run(output)
            self.assertFalse(output.exists())

    def test_model_identity_is_locked_to_upstream_public_commit(self):
        value = pilot.pilot_plan("b" * 64)
        pin = value["tts_pins"][0]
        self.assertEqual(pin["provider_id"], "qwen3-custom-en")
        self.assertEqual(pin["model_revision_sha"], pilot.REV)
        self.assertEqual(pin["model_assets_sha256"], "b" * 64)
        self.assertEqual(len(pilot.REV), 40)
        self.assertLessEqual(pilot.MAX_CALLS, 16)


if __name__ == "__main__":
    unittest.main()
