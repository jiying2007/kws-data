"""No pretrained models or downloads; verify seven-new-voice bounded campaign."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_qwen_real_once as old
import english_qwen_capture as capture
import english_seven_voice_tts_once as new
import multilingual_campaign as market
from test_multilingual_campaign import FakeEnglishBackend, fixture_wave_writer


class SevenVoiceSourceTests(unittest.TestCase):
    def test_claim_is_frozen_and_planning_only(self):
        claim = new.claim()
        self.assertEqual(claim["max_new_tts_cells"], 63)
        self.assertEqual(claim["stage_budgets"], [16, 16, 16, 15])
        self.assertEqual(claim["original_tts_run_id"], 43214159818)
        self.assertEqual(claim["original_audio_reused_not_regenerated"], 10)
        self.assertEqual(new.main([]), 0)
        with self.assertRaises(ValueError):
            new.plan("not-a-model-checkpoint")

    def test_seven_new_speakers_do_not_repeat_original_pilot(self):
        plan = new.plan("a" * 64)
        self.assertEqual(plan, new.plan("a" * 64))
        previous = old.pilot_plan("a" * 64)
        self.assertEqual(len(plan["cells"]), 63)
        self.assertEqual(len(plan["voice_sources"]), 7)
        self.assertEqual(set(c["role"] for c in plan["cells"]),
                         {"train-candidate", "development-only"})
        self.assertFalse(set(c["cell_id"] for c in plan["cells"]).intersection(
            c["cell_id"] for c in previous["cells"]))
        self.assertFalse(set(c["source_family_id"] for c in plan["cells"]).intersection(
            c["source_family_id"] for c in previous["cells"]))
        self.assertEqual(sum(c["intent"] == "wake" for c in plan["cells"]), 28)
        self.assertEqual(sum(c["intent"] == "near_miss" for c in plan["cells"]), 21)
        self.assertEqual(sum(c["intent"] == "ordinary_negative" for c in plan["cells"]), 14)
        self.assertEqual({c["requested_keyword_index"] for c in plan["cells"] if c["intent"] == "wake"}, {1,2})
        self.assertEqual({c["source_family_id"] for c in plan["cells"] if c["role"] == "development-only"},
                         {"qwen3-preset-serena", "qwen3-preset-uncle_fu"})
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["human_gold"])
        self.assertEqual(plan["asr_calls"], 0)

    def test_63_cells_are_generated_in_bounded_no_repeat_stages(self):
        plan = new.plan("b" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "real-looking-fixture"
            fake = FakeEnglishBackend()
            old_calls = 0
            for i,budget in enumerate(new.STAGES):
                row = capture.capture(plan, out, fake, max_calls=budget,
                                      writer=fixture_wave_writer)
                self.assertEqual(row["attempts_this_call"], budget)
                old_calls += budget
                self.assertEqual(row["total_started"], old_calls)
                self.assertEqual(row["total_completed"], old_calls)
            self.assertEqual(len(fake.calls), 63)
            self.assertEqual(len({c for c,_ in fake.calls}), 63)
            result = capture.capture(plan, out, fake, max_calls=16,
                                     writer=fixture_wave_writer)
            self.assertEqual(result["attempts_this_call"], 0)
            self.assertEqual(result["total_started"], 63)
            self.assertEqual(len(fake.calls), 63)
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            first = out / "cells" / plan["cells"][0]["cell_id"] / "completed.json"
            self.assertFalse(market.read(first)["training_admitted"])

    def test_terminal_failed_cell_never_replayed(self):
        plan = new.plan("b" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "intentional-one-failure"
            fake = FakeEnglishBackend(fail_first=True)
            first = capture.capture(plan, out, fake, max_calls=16,
                                    writer=fixture_wave_writer)
            self.assertEqual((first["total_started"], first["total_completed"]),
                             (16,15))
            second = capture.capture(plan, out, fake, max_calls=16,
                                     writer=fixture_wave_writer)
            self.assertEqual((second["total_started"], second["total_completed"]),
                             (32,31))
            self.assertEqual(len(fake.calls), 32)
            # A pre-consumed failed cell must remain failed, not replace its receipt.
            cell = out / "cells" / plan["cells"][0]["cell_id"]
            self.assertTrue((cell / "started.json").is_file())
            self.assertTrue((cell / "failed.json").is_file())
            self.assertFalse((cell / "completed.json").exists())
            with self.assertRaisesRegex(ValueError, "CAPTURE_CALL_BUDGET"):
                capture.capture(plan, out, fake, max_calls=17,
                                writer=fixture_wave_writer)

    def test_wrong_runner_never_downloads_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"never-created"
            with mock.patch.dict("os.environ",{
                    "GITHUB_REPOSITORY":"other/project",
                    "GITHUB_REF":"refs/heads/main",
                    "GITHUB_RUN_ATTEMPT":"1"},clear=False):
                with mock.patch.object(old, "materialize_model",
                                       side_effect=AssertionError("must not download")):
                    with self.assertRaisesRegex(ValueError,"SEVEN_VOICE_MAIN_ONCE_REQUIRED"):
                        new.run(out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
