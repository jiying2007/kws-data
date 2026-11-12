"""Offline source-bound tests for real Qwen3-ASR pilot (no model calls)."""
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
import english_qwen_asr_real_once as asr
import english_qwen_capture as english
import english_qwen_real_once as pilot
import multilingual_campaign as market
from test_multilingual_campaign import FakeEnglishBackend, fixture_wave_writer


class RealQwenAsrSourceTests(unittest.TestCase):
    def test_exact_launch_and_model_preflight(self):
        claim = asr.request()
        self.assertEqual(claim["source_run_id"], 43214159818)
        self.assertEqual(claim["model_revision"], asr.REV)
        self.assertEqual(asr.main([]), 0)
        self.assertFalse(claim["training_admitted"])
        self.assertEqual(sum(r["bytes"] for r in asr.ASSETS), 1880558833)
        self.assertEqual({r["path"] for r in asr.ASSETS}, {
            "model.safetensors", "chat_template.json", "config.json",
            "generation_config.json", "merges.txt", "preprocessor_config.json",
            "tokenizer_config.json", "vocab.json"})
        self.assertEqual(len(asr.REV), 40)

    def test_original_waveforms_replayed_from_saved_manifest_no_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = pilot.pilot_plan("a" * 64)
            (root / "source-plan.json").write_bytes(market.canonical(plan) + b"\n")
            backend = FakeEnglishBackend()
            cap = english.capture(plan, root / "capture", backend,
                                  max_calls=10, writer=fixture_wave_writer)
            self.assertEqual(cap["total_completed"], 10)
            rows = []
            for cell in plan["cells"]:
                record = market.read(root / "capture" / "cells" / cell["cell_id"] /
                                     "completed.json")
                rows.append({"cell_id": cell["cell_id"], "status": "GENERATED_UNADMITTED",
                             "training_admitted": False,
                             "wav_sha256": record["audio"]["wav_sha256"]})
            summary = {"schema": pilot.POLICY, "status": "COMPLETE",
                       "generated_audio_count": 10, "asr_calls": 0,
                       "training_admitted": False, "source_rows": rows}
            (root / "summary.json").write_bytes(market.canonical(summary) + b"\n")
            original, wavs = asr.verified_source(root)
            self.assertEqual(original, plan)
            self.assertEqual(len(wavs), 10)
            self.assertEqual([row["cell_id"] for row, _, _ in wavs],
                             [row["cell_id"] for row in plan["cells"]])
            self.assertEqual(len(backend.calls), 10)
            self.assertEqual({row["source_family_id"] for row in original["cells"]},
                             {"qwen3-preset-ryan", "qwen3-preset-aiden"})

            # The producer's exact receipt, not requested text, is authoritative.
            changed = copy.deepcopy(summary)
            changed["source_rows"][0]["wav_sha256"] = "f" * 64
            (root / "summary.json").write_bytes(market.canonical(changed) + b"\n")
            with self.assertRaisesRegex(ValueError, "SOURCE_SUMMARY_WAV_BINDING_DRIFT"):
                asr.verified_source(root)
            (root / "summary.json").write_bytes(market.canonical(summary) + b"\n")
            source = copy.deepcopy(plan)
            source["cells"][0]["text_requested"] = "faked english"
            (root / "source-plan.json").write_bytes(market.canonical(source) + b"\n")
            with self.assertRaisesRegex(ValueError, "PILOT_SOURCE_PLAN_DRIFT"):
                asr.verified_source(root)

    def test_wrong_runner_ref_is_never_new_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not-created"
            with mock.patch.dict("os.environ", {
                    "GITHUB_REPOSITORY": "untrusted/repo",
                    "GITHUB_REF": "refs/heads/main",
                    "GITHUB_RUN_ATTEMPT": "1"}, clear=False):
                with self.assertRaisesRegex(ValueError, "REAL_ASR_ONCE_MAIN_ONLY"):
                    asr.run(Path(tmp), output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
