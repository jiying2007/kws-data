"""Zero-model regression around one-time real Whisper-small English ASR."""
from __future__ import annotations

import ast
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_whisper_real_once as target
from test_english_qwen_asr_sourcefix_once import EnglishAsrSourcefixTests


class EnglishWhisperOnceTests(unittest.TestCase):
    def test_pins_and_default_never_execute_model(self):
        value = target.request()
        self.assertEqual(value["model_id"], "openai/whisper-small")
        self.assertEqual(value["model_source_commit"], target.MODEL_SOURCE_COMMIT)
        self.assertEqual(value["model_weight_sha256"], target.WEIGHT_SHA256)
        self.assertEqual(value["python_package_version"], "20250625")
        self.assertEqual(value["max_new_asr_calls"], 10)
        self.assertFalse(value["training_admitted"])
        self.assertFalse(value["three_asr_completed"])
        self.assertEqual(len(target.MODEL_SOURCE_COMMIT), 40)
        self.assertEqual(len(target.WEIGHT_SHA256), 64)

    def test_real_cli_replays_original_tts_wavs_not_qwen_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            EnglishAsrSourcefixTests().populate(parent)
            plan, entries = target.source_inputs(parent)
            self.assertEqual(len(entries), 10)
            self.assertEqual(plan["locale"], "en-US")
            self.assertEqual({cell["role"] for cell, _, _ in entries},
                             {"train-candidate", "development-only"})
            p = subprocess.run(
                [sys.executable, "-B", str(Path(target.__file__)),
                 "--artifact-parent", str(parent), "--verify-source"],
                capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)["asr_calls"], 0)
            self.assertEqual(json.loads(p.stdout)["input_audio_count"], 10)
            self.assertFalse(json.loads(p.stdout)["training_admitted"])

    def test_modified_original_audio_is_rejected_before_whisper_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            EnglishAsrSourcefixTests().populate(parent)
            first = target.source_inputs(parent)[1][0][1]
            first.write_bytes(first.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "CAPTURE_WAV_SHA_DRIFT"):
                target.source_inputs(parent)

    def test_external_weights_must_match_exact_sha_and_cannot_be_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with mock.patch.object(target.urllib.request, "urlopen",
                                   return_value=io.BytesIO(b"no real weights")), \
                 mock.patch.object(target.shutil, "disk_usage",
                                   return_value=type("Disk", (), {"free": 7 * 1024**3})()):
                with self.assertRaisesRegex(ValueError, "WHISPER_MODEL_CHECKPOINT_SHA_DRIFT"):
                    target.pinned_model(folder)
                self.assertEqual((folder / "small.pt").read_bytes(), b"no real weights")

    def test_wrong_execution_host_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not-created"
            with mock.patch.dict("os.environ", {
                    "GITHUB_REPOSITORY": "other/repository",
                    "GITHUB_REF": "refs/heads/main",
                    "GITHUB_RUN_ATTEMPT": "1"}, clear=False):
                with self.assertRaisesRegex(ValueError, "WHISPER_REAL_ONCE_MAIN_ONLY"):
                    target.run(Path(tmp), output)
            self.assertFalse(output.exists())

    def test_decode_is_blind_to_wake_words_and_prior_asr_results(self):
        tree = ast.parse(Path(target.__file__).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Attribute) and n.func.attr == "transcribe"]
        self.assertEqual(len(calls), 1)
        kwargs = {row.arg: row.value for row in calls[0].keywords}
        self.assertEqual(kwargs["language"].value, "en")
        self.assertIsNone(kwargs["initial_prompt"].value)
        self.assertIs(kwargs["condition_on_previous_text"].value, False)
        self.assertEqual(kwargs["temperature"].value, 0.0)
        self.assertNotIn("prompt", kwargs)
        self.assertNotIn("hotwords", kwargs)


if __name__ == "__main__":
    unittest.main()
