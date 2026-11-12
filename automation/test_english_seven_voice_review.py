"""Invented audio/ASR receipts only: source-bound English review counterexamples."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_seven_voice_asr_once as asr
import english_seven_voice_review as review
import multilingual_campaign as market
from test_english_seven_voice_asr_once import invented_source


def invented_three_asr(root: Path):
    parent = root / "source"
    parent.mkdir()
    plan, _ = invented_source(parent)
    wanted = {cell["cell_id"]: cell["text_requested"] for cell in plan["cells"]}
    paths = {}
    for provider in asr.PROVIDERS:
        out = root / provider
        result = asr.capture(provider, parent, out,
                             decoder=lambda _wav, folder: wanted[folder.name])
        assert result["completed_asr"] == 63
        paths[provider] = out
    return parent, paths, plan


class EnglishReviewTests(unittest.TestCase):
    def test_synthetic_source_consensus_is_not_gold_or_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            parent, paths, plan = invented_three_asr(Path(temp))
            report = review.build_review(parent, paths)
            self.assertEqual(report["counts"]["total"], 63)
            self.assertEqual(report["counts"]["three_way_completed_calls"], 189)
            self.assertEqual(report["counts"]["three_way_request_exact"], 63)
            self.assertEqual(report["counts"]["three_way_unanimous"], 63)
            self.assertEqual(report["by_intent"]["wake_vosen"]["total"], 14)
            self.assertEqual(report["by_intent"]["wake_mero"]["total"], 14)
            self.assertEqual(report["by_intent"]["near_miss"]["total"], 21)
            self.assertEqual(report["by_intent"]["ordinary_negative"]["total"], 14)
            self.assertEqual(len(report["rows"]), 63)
            self.assertEqual(report["status"], "MACHINE_REVIEW_COMPLETE_NOT_ADMITTED")
            self.assertFalse(report["training_admitted"])
            self.assertEqual(report["acoustic_gold_admitted"], 0)
            self.assertEqual(report["model_calls_performed_by_this_audit"], 0)
            self.assertEqual({row["role"] for row in report["rows"]},
                             {"train-candidate", "development-only"})
            self.assertEqual({row["review_lane"] for row in report["rows"]
                              if row["requested_intent"] == "near_miss"},
                             {"NEAR_MISS_AMBIGUOUS_QUARANTINE"})
            for row in report["rows"]:
                self.assertFalse(row["training_admitted"])
                self.assertFalse(row["human_gold"])
                self.assertFalse(row["trusted_negative"])
                self.assertFalse(row["acoustic_pronunciation_verified"])
                self.assertEqual(set(row["asr"]), set(asr.PROVIDERS))

    def test_tampered_asr_native_or_summary_is_rejected_before_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            parent, paths, plan = invented_three_asr(Path(temp))
            cell = plan["cells"][0]
            native = paths["whisper"] / "en-seven-asr-result" / "cells" / cell["cell_id"] / "native.json"
            value = market.read(native)
            value["actual_text"] = "Hey Mero"  # old receipt digest must reject
            native.write_bytes(market.canonical(value) + b"\n")
            with self.assertRaisesRegex(ValueError, "ASR_REVIEW_CELL_COMPLETION_DRIFT"):
                review.build_review(parent, paths)

    def test_mismatched_original_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent, paths, plan = invented_three_asr(Path(temp))
            wav = parent / asr.ARTIFACT_ROOT / "capture" / "cells" / plan["cells"][0]["cell_id"] / "audio-16k.wav"
            wav.write_bytes(wav.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "CAPTURE_WAV_SHA_DRIFT"):
                review.build_review(parent, paths)

    def test_artifact_digest_and_zip_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "extract"
            target.mkdir()
            src = root / "source.zip"
            with zipfile.ZipFile(src, "w") as z:
                z.writestr("../escape", "bad")
            with self.assertRaisesRegex(ValueError, "ASR_REVIEW_ARTIFACT_DIGEST_DRIFT"):
                review.unpack_pinned(src, "0" * 64, target)
            with self.assertRaisesRegex(ValueError, "ASR_REVIEW_ZIP_PATH_OR_ALIAS"):
                review.unpack_pinned(src, review.digest_file(src), target)
            self.assertFalse((root / "escape").exists())
            with zipfile.ZipFile(src, "w") as z:
                z.writestr("example/info.json", "{}")
            review.unpack_pinned(src, review.digest_file(src), target)
            self.assertEqual((target / "example/info.json").read_text(), "{}")

    def test_cyrillic_transcript_is_not_equated_to_english_keyword(self):
        self.assertNotEqual(review.normalize_observation("Хей, воза."),
                            review.normalize_observation("Hey Vosen"))
        self.assertEqual(review.normalize_observation("Hey, Vosen!"),
                         review.normalize_observation("hey vosen"))
        self.assertNotEqual(review.normalize_observation("Hey Miro"),
                            review.normalize_observation("Hey Mero"))
        self.assertEqual(review.ARCHIVES["tts"][0], 13954744587)
        self.assertEqual(review.ARCHIVES["parakeet"][0], 13958425005)


if __name__ == "__main__":
    unittest.main()
