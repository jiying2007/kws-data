"""No-model proof that the consumed English TTS artifact has a nested root."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_qwen_asr_real_once as asr
import english_qwen_asr_sourcefix_once as fix
import english_qwen_capture as english
import english_qwen_real_once as pilot
import multilingual_campaign as market
from test_multilingual_campaign import FakeEnglishBackend, fixture_wave_writer


class EnglishAsrSourcefixTests(unittest.TestCase):
    def populate(self, parent: Path):
        root = parent / fix.SOURCE_SUBDIR
        root.mkdir()
        plan = pilot.pilot_plan("a" * 64)
        (root / "source-plan.json").write_bytes(market.canonical(plan) + b"\n")
        generated = english.capture(plan, root / "capture",
                                    FakeEnglishBackend(), max_calls=10,
                                    writer=fixture_wave_writer)
        self.assertEqual(generated["total_completed"], 10)
        rows = []
        for cell in plan["cells"]:
            record = market.read(root / "capture" / "cells" / cell["cell_id"] /
                                 "completed.json")
            rows.append({
                "cell_id": cell["cell_id"], "status": "GENERATED_UNADMITTED",
                "training_admitted": False,
                "wav_sha256": record["audio"]["wav_sha256"]})
        (root / "summary.json").write_bytes(market.canonical({
            "schema": pilot.POLICY, "status": "COMPLETE",
            "generated_audio_count": 10, "asr_calls": 0,
            "training_admitted": False, "source_rows": rows}) + b"\n")
        # The actual downloaded Actions ZIP has a sibling audit directory.
        (parent / "en-qwen-audit").mkdir()
        (parent / "en-qwen-audit" / "installed-packages.txt").write_text(
            "invented source-only test")
        return root, plan

    def test_independent_claim_carries_previous_failure_without_replaying_it(self):
        value = fix.claim()
        self.assertEqual(value["previous_failed_run_id"], 43218378495)
        self.assertEqual(value["previous_real_asr_calls"], 0)
        self.assertEqual(value["max_new_asr_calls"], 10)
        self.assertEqual(value["original_tts_artifact_id"], asr.SOURCE_ARTIFACT_ID)
        self.assertFalse(value["training_admitted"])
        with mock.patch.object(fix, "run", side_effect=AssertionError("no models")):
            self.assertEqual(fix.main([]), 2)

    def test_nested_source_layout_validates_before_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root, original = self.populate(parent)
            actual_root, plan, count = fix.original_audio_root(parent)
            self.assertEqual(actual_root, root)
            self.assertEqual(plan, original)
            self.assertEqual(count, 10)
            self.assertEqual(fix.main(["--artifact-parent", str(parent),
                                       "--verify-source"]), 0)
            self.assertEqual(fix.main(["--artifact-parent", str(parent)]), 2)
            # This is the exact failure from the first consumed run.
            with self.assertRaisesRegex(ValueError, "INPUT_FILE_UNSAFE_OR_OVERSIZE"):
                asr.verified_source(parent)
            with mock.patch.object(asr, "run",
                                   side_effect=AssertionError("real model prohibited")):
                with self.assertRaisesRegex(AssertionError, "real model prohibited"):
                    fix.run(parent, parent / "fresh-output")

    def test_actual_command_line_imports_sibling_modules_and_verifies_all_wavs(self):
        """Regression for the previously consumed -I/-S startup failure."""
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            self.populate(parent)
            result = subprocess.run(
                [sys.executable, "-B", str(Path(fix.__file__)),
                 "--artifact-parent", str(parent), "--verify-source"],
                check=False, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout.strip())
            self.assertEqual(report["status"], "SOURCE_VERIFIED_NO_ASR")
            self.assertEqual(report["audio_count"], 10)
            self.assertEqual(report["asr_calls"], 0)
            self.assertFalse(report["training_admitted"])

    def test_wrong_directory_and_modified_original_audio_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root, _ = self.populate(parent)
            first = asr.verified_source(root)[1][0][1]
            first.write_bytes(first.read_bytes() + b"unexpected")
            with self.assertRaisesRegex(ValueError, "CAPTURE_WAV_SHA_DRIFT"):
                fix.original_audio_root(parent)
            with self.assertRaisesRegex(ValueError, "ASR_EXPECTED_ZIP_PARENT_MISSING"):
                fix.original_audio_root(root)

    def test_qwen_transcribe_api_uses_context_not_obsolete_prompt(self):
        tree = ast.parse(Path(asr.__file__).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Attribute) and n.func.attr == "transcribe"]
        self.assertEqual(len(calls), 1)
        opts = {row.arg: row.value for row in calls[0].keywords}
        self.assertNotIn("prompt", opts)
        self.assertIn("context", opts)
        self.assertIsInstance(opts["context"], ast.Constant)
        self.assertEqual(opts["context"].value, "")


if __name__ == "__main__":
    unittest.main()
