"""Model-free hostile source checks for exact English63 bilingual Mero-only KWS."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english63_zh_en_mero_only as mero
import multilingual_campaign as market


class MeroOnlySourceTests(unittest.TestCase):
    def test_frozen_source_reuses_63_no_english_regeneration(self):
        claim = mero.claim()
        self.assertEqual(claim["source_clip_count"], 63)
        self.assertEqual(claim["pretrained_model"], "zh-en")
        self.assertEqual(claim["original_tts_run_id"], 43242278962)
        self.assertEqual(claim["original_tts_artifact_id"], 13954744587)
        self.assertEqual(claim["keyword_original_text"], "Hey Mero")
        self.assertFalse(claim["vosen_supported"])
        self.assertFalse(claim["model_promotion_allowed"])
        self.assertEqual(claim["true_far_frr"], "NOT_RUN")

    def test_any_changed_trigger_is_rejected_before_model(self):
        original = mero.claim()
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "source.json"
            with mock.patch.object(mero, "TRIGGER", f):
                f.write_bytes(market.canonical(original) + b"\n")
                self.assertEqual(mero.claim(), original)
                for key, bad in (
                    ("source_clip_count", 64),
                    ("vosen_supported", True),
                    ("model_promotion_allowed", True),
                    ("pretrained_model_sha256", "0" * 64),
                    ("new_tts_calls", 1),
                ):
                    altered = copy.deepcopy(original)
                    altered[key] = bad
                    f.write_bytes(market.canonical(altered) + b"\n")
                    with self.assertRaisesRegex(ValueError,
                                                "EN_MERO_FROZEN_SOURCE_CLAIM_DRIFT"):
                        mero.claim()

    def test_only_protected_first_main_run_can_actually_infer(self):
        env = {
            "GITHUB_REPOSITORY": "jiying2007/kws-data",
            "GITHUB_REPOSITORY_OWNER": "jiying2007",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ATTEMPT": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            mero.protected_main()
        for name, bad in (
            ("GITHUB_REPOSITORY", "attacker/kws-data"),
            ("GITHUB_REPOSITORY_OWNER", "attacker"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_REF", "refs/heads/experiment"),
            ("GITHUB_RUN_ATTEMPT", "2"),
        ):
            with mock.patch.dict(os.environ, {**env, name: bad}, clear=True):
                with self.assertRaisesRegex(ValueError,
                                            "EN_MERO_FIRST_PROTECTED_MAIN_ONLY"):
                    mero.protected_main()

    def test_official_phone_lexicon_requires_mero_but_blocks_vosen(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "en.phone"
            f.write_text("HEY HH EY1\nMERO M EH1 R OW0\n")
            self.assertTrue(mero.verify_official_phone_words(f))
            f.write_text("HEY HH EY1\nVOSEN V OW1 S AH0 N\n")
            with self.assertRaisesRegex(ValueError, "EN_MERO_WORDS_NOT_SUPPORTED"):
                mero.verify_official_phone_words(f)
            f.write_text("HEY HH EY1\nMERO M EH1 R OW0\nVOSEN V OW1 S AH0 N\n")
            with self.assertRaisesRegex(ValueError, "EN_MERO_WORDS_NOT_SUPPORTED"):
                mero.verify_official_phone_words(f)

    def test_one_keyword_requires_official_cli_and_one_phoneme_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            lex = d / "en.phone"
            lex.write_text("HEY HH EY1\nMERO M EH1 R OW0\n")
            tokens = d / "tokens.txt"
            tokens.write_text("HH 0\n")
            def valid_run(cmd, **kwargs):
                self.assertIn("--tokens-type", cmd)
                self.assertIn("phone+ppinyin", cmd)
                self.assertIn("--lexicon", cmd)
                self.assertIn(str(lex), cmd)
                self.assertEqual((d / "mero-original.txt").read_text(),
                                 "HEY MERO @HEY_MERO\n")
                (d / "mero-encoded.txt").write_text(
                    "HH EY1 M EH1 R OW0 @HEY_MERO\n")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            with mock.patch.object(mero.subprocess, "run", side_effect=valid_run):
                path = mero.encode_one_keyword({"lexicon": lex, "tokens": tokens}, d)
            self.assertEqual(path.read_text().strip(),
                             "HH EY1 M EH1 R OW0 @HEY_MERO")

    def test_false_far_or_gold_promotion_never_passes_public_receipt(self):
        rows = [{"cell_id": f"id-{i}", "training_admitted": False,
                 "human_gold": False, "trusted_negative": False,
                 "pcm_sha256": "f" * 64, "events": []} for i in range(63)]
        valid = {
            "schema": mero.POLICY,
            "status": "EXECUTED_REAL_ZH_EN_MERO_ONLY_WEAK_CLIP_EVENTS",
            "pretrained_archive_sha256": mero.PUBLISHED_ZH_EN_SHA,
            "original_tts_zip_sha256": mero.ORIGINAL_TTS_SHA,
            "source_clip_count": 63, "clip_results": rows,
            "keyword_original_text": "Hey Mero",
            "vosen_not_evaluated": True,
            "new_tts_calls": 0, "new_asr_calls": 0, "optimizer_updates": 0,
            "model_promotion_allowed": False, "training_admitted": False,
            "human_gold": False, "real_far_per_hour": "NOT_RUN",
            "real_frr": "NOT_RUN", "ssc305_board_profile": "NOT_RUN",
        }
        self.assertTrue(mero.validate_receipt(valid))
        for key, bad in (
            ("model_promotion_allowed", True),
            ("training_admitted", True),
            ("real_far_per_hour", 0.0),
            ("real_frr", 0.0),
            ("vosen_not_evaluated", False),
            ("source_clip_count", 62),
        ):
            changed = copy.deepcopy(valid)
            changed[key] = bad
            with self.assertRaises(ValueError):
                mero.validate_receipt(changed)
        changed = copy.deepcopy(valid)
        changed["clip_results"][0]["trusted_negative"] = True
        with self.assertRaises(ValueError):
            mero.validate_receipt(changed)

    def test_default_cli_is_model_free_and_preflight_requires_both_archives(self):
        script = Path(mero.__file__)
        p = subprocess.run([sys.executable, "-I", "-S", "-B", str(script)],
                           capture_output=True, text=True, check=False, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        obj = json.loads(p.stdout)
        self.assertEqual(obj["candidate"], "Hey Mero")
        self.assertTrue(obj["vosen_not_evaluated"])
        self.assertFalse(obj["real_inference_started"])
        for arguments in (["--preflight"], ["--execute"]):
            p = subprocess.run([sys.executable, "-I", "-S", "-B", str(script),
                                *arguments], capture_output=True, text=True,
                               check=False, timeout=20)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("EN_MERO_", p.stderr)


if __name__ == "__main__":
    unittest.main()
