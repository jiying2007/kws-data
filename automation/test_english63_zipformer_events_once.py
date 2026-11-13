"""Model-free hostile/source-only checks for the real English Zipformer event probe."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english63_zipformer_events_once as probe
import multilingual_campaign as market


class English63ZipformerEventTests(unittest.TestCase):
    def test_original_63_and_pretrained_source_identity(self):
        claim = probe.frozen_claim()
        self.assertEqual(claim["source_clips"], 63)
        self.assertEqual(claim["source_run_id"], 43242278962)
        self.assertEqual(claim["source_artifact_id"], 13954744587)
        self.assertEqual(claim["new_tts_calls"], 0)
        self.assertEqual(claim["new_asr_calls"], 0)
        self.assertFalse(claim["model_promotion_allowed"])
        self.assertIsNone(claim["models"]["gigaspeech"]["sha256"])
        self.assertEqual(claim["models"]["zh-en"]["sha256"],
                         "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6")

    def test_protected_once_gate_rejects_pr_or_retry(self):
        env = {"GITHUB_REPOSITORY": "jiying2007/kws-data",
               "GITHUB_REPOSITORY_OWNER": "jiying2007",
               "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
               "GITHUB_RUN_ATTEMPT": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            probe.gate()
        for name, other in (
            ("GITHUB_REPOSITORY", "attacker/kws-data"),
            ("GITHUB_REPOSITORY_OWNER", "attacker"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_REF", "refs/heads/research/other"),
            ("GITHUB_RUN_ATTEMPT", "2"),
        ):
            with mock.patch.dict(os.environ, {**env, name: other}, clear=True):
                with self.assertRaisesRegex(ValueError,
                                            "EN_ZIPFORMER_PROTECTED_MAIN_ONCE_ONLY"):
                    probe.gate()

    def test_mutated_frozen_claim_fails(self):
        base = probe.frozen_claim()
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp) / "frozen.json"
            with mock.patch.object(probe, "TRIGGER", temp):
                temp.write_bytes(market.canonical(base) + b"\n")
                self.assertEqual(probe.frozen_claim(), base)
                for dotted, payload in (
                    (("source_clips",), 62),
                    (("new_tts_calls",), 1),
                    (("formal_training_admitted",), True),
                    (("models", "zh-en", "sha256"), "0" * 64),
                    (("models", "gigaspeech", "sha256"), "0" * 64),
                ):
                    altered = copy.deepcopy(base)
                    dest = altered
                    for key in dotted[:-1]:
                        dest = dest[key]
                    dest[dotted[-1]] = payload
                    temp.write_bytes(market.canonical(altered) + b"\n")
                    with self.assertRaisesRegex(ValueError,
                                                "EN_ZIPFORMER_FROZEN_TRIGGER_DRIFT"):
                        probe.frozen_claim()

    def test_corrupt_archive_stops_before_model_or_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "wrong.tar.bz2"
            fake.write_bytes(b"not an official archive")
            for model in probe.MODEL_SPECS:
                with self.assertRaisesRegex(ValueError,
                                            "EN_ZIPFORMER_ARCHIVE_BYTES_DRIFT"):
                    probe.extract_locked_model(fake, model, root)
            with self.assertRaisesRegex(ValueError,
                                        "EN_ZIPFORMER_UNKNOWN_MODEL"):
                probe.extract_locked_model(fake, "unknown", root)

    def test_event_aliases_never_create_false_labels(self):
        self.assertEqual(probe.normalize_event_keyword("HEY VOSEN"), "HEY_VOSEN")
        self.assertEqual(probe.normalize_event_keyword("HEY_MERO"), "HEY_MERO")
        self.assertEqual(probe.normalize_event_keyword("HEY WOSEN"), "UNEXPECTED_KEYWORD")
        with self.assertRaisesRegex(ValueError,
                                    "EN_ZIPFORMER_KEYWORD_RESULT_INVALID"):
            probe.normalize_event_keyword("")

    def test_no_far_frr_or_model_promotion_in_receipt(self):
        rows = [{"cell_id": "id-" + str(i), "human_gold": False,
                 "training_admitted": False, "trusted_negative": False,
                 "events": []} for i in range(63)]
        base = {"schema": probe.POLICY,
                "status": "EXECUTED_REAL_PRETRAINED_SYNTHETIC_CLIP_EVENTS",
                "model_name": "gigaspeech", "source_wav_count": 63,
                "clip_results": rows,
                "new_tts_calls": 0, "new_asr_calls": 0,
                "optimizer_updates": 0,
                "model_promotion_allowed": False,
                "formal_training_admitted": False,
                "independent_human_gold": False,
                "continuous_negative_far_per_hour": "NOT_RUN",
                "true_frr": "NOT_RUN", "ssc305_profile": "NOT_RUN"}
        self.assertTrue(probe.validate_result(base))
        for key, bad in (("model_promotion_allowed", True),
                         ("formal_training_admitted", True),
                         ("continuous_negative_far_per_hour", 0.0),
                         ("true_frr", 0.0),
                         ("ssc305_profile", {"rss_kib": 1}),
                         ("source_wav_count", 62)):
            tampered = copy.deepcopy(base)
            tampered[key] = bad
            with self.assertRaisesRegex(ValueError,
                                        "EN_ZIPFORMER_FALSE_PROMOTION_OR_RECEIPT"):
                probe.validate_result(tampered)
        tampered = copy.deepcopy(base)
        tampered["clip_results"][2]["trusted_negative"] = True
        with self.assertRaisesRegex(ValueError,
                                    "EN_ZIPFORMER_CELL_FALSE_LABEL"):
            probe.validate_result(tampered)

    def test_offline_cli_does_not_download_or_infer(self):
        p = subprocess.run([sys.executable, "-I", "-S", "-B",
                            str(Path(probe.__file__))],
                           capture_output=True, text=True, check=False,
                           timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        receipt = json.loads(p.stdout)
        self.assertEqual(receipt["original_audio_reused"], 63)
        self.assertFalse(receipt["model_promotion_allowed"])
        p = subprocess.run([sys.executable, "-I", "-S", "-B",
                            str(Path(probe.__file__)), "--execute"],
                           capture_output=True, text=True, check=False,
                           timeout=20)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("EN_ZIPFORMER_EXPLICIT_INPUTS_REQUIRED", p.stderr)


if __name__ == "__main__":
    unittest.main()
