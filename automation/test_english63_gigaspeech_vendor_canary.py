"""No-model regression guards for original GigaSpeech vendor-positive diagnosis."""
from __future__ import annotations

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
import english63_gigaspeech_vendor_canary as vendor


def invented_event(name):
    return {
        "keyword": name,
        "mapped_keyword": "UNEXPECTED_KEYWORD",
        "token_timestamps_sec": [0.5],
        "tokens": ["TEST"],
        "available_after_input_sec": 0.6,
        "is_streaming_event": True,
    }


def valid_report():
    rows = []
    for filename, target in vendor.VENDOR_FILES:
        rows.append({
            "filename": filename, "documented_vendor_keyword": target,
            "documented_keyword_detected": True,
            "wav_sha256": "a" * 64, "pcm_sha256": "b" * 64,
            "audio_samples": 32000, "decoded_steps": 2,
            "events": [invented_event(target)],
            "acoustic_gold": False,
        })
    return {
        "schema": vendor.POLICY,
        "status": "EXECUTED_REAL_OFFICIAL_VENDOR_CONTROLS_NOT_ENGLISH63_QUALIFICATION",
        "baseline_english63_event_run_id": vendor.BASELINE_RUN_ID,
        "baseline_english63_event_count": vendor.BASELINE_EVENTS_PER_63,
        "model_archive_sha256": vendor.MODEL_ARCHIVE_SHA256,
        "vendor_clip_count": 2,
        "canary_rows": rows,
        "vendor_positive_control_passed": True,
        "new_tts_calls": 0, "new_asr_calls": 0, "optimizer_updates": 0,
        "source_english63_replayed": False,
        "formal_training_admitted": False, "human_gold": False,
        "model_promotion_allowed": False,
        "independent_human_far_frr": "NOT_RUN",
        "ssc305_board_profile": "NOT_RUN",
    }


class English63VendorCanarySourceTests(unittest.TestCase):
    def test_canary_is_vendor_only_and_binds_zero_event_source_run(self):
        self.assertEqual(vendor.BASELINE_RUN_ID, 43443799469)
        self.assertEqual(vendor.BASELINE_ARTIFACT_ID, 14046722243)
        self.assertEqual(vendor.BASELINE_EVENTS_PER_63, 0)
        self.assertEqual([x[0] for x in vendor.VENDOR_FILES],
                         ["0.wav", "1.wav"])
        self.assertEqual([x[1] for x in vendor.VENDOR_FILES],
                         ["LIGHT UP", "LOVELY CHILD"])
        self.assertEqual(vendor.MODEL_ARCHIVE_BYTES, 17626723)
        self.assertEqual(vendor.MODEL_ARCHIVE_SHA256,
                         "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a")
        self.assertTrue(vendor.validate_receipt(valid_report()))

    def test_empty_actual_events_never_claim_success(self):
        value = valid_report()
        for row in value["canary_rows"]:
            row["events"] = []
            row["documented_keyword_detected"] = False
        value["vendor_positive_control_passed"] = False
        self.assertTrue(vendor.validate_receipt(value))
        value["vendor_positive_control_passed"] = True
        with self.assertRaisesRegex(ValueError,
                                    "EN_VENDOR_FALSE_POSITIVE_CONTROL_PASS"):
            vendor.validate_receipt(value)

    def test_forged_positive_or_model_promotion_rejected(self):
        original = valid_report()
        changes = (
            lambda r: r.__setitem__("model_promotion_allowed", True),
            lambda r: r.__setitem__("source_english63_replayed", True),
            lambda r: r.__setitem__("human_gold", True),
            lambda r: r.__setitem__("new_tts_calls", 1),
            lambda r: r.__setitem__("independent_human_far_frr", 0.0),
            lambda r: r.__setitem__("baseline_english63_event_count", 1),
            lambda r: r["canary_rows"][0].__setitem__("events", []),
            lambda r: r["canary_rows"][1].__setitem__(
                "documented_keyword_detected", False),
            lambda r: r["canary_rows"][0].__setitem__(
                "filename", "../0.wav"),
        )
        for change in changes:
            receipt = copy.deepcopy(original)
            change(receipt)
            with self.assertRaises(ValueError):
                vendor.validate_receipt(receipt)

    def test_model_archive_wrong_identity_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp).resolve() / "invented.tar.bz2"
            model.write_bytes(b"not a genuine model")
            with self.assertRaisesRegex(ValueError,
                                        "EN_VENDOR_ARCHIVE_GEOMETRY"):
                vendor.source_contract(model)
            with mock.patch.object(vendor, "MODEL_ARCHIVE_BYTES", model.stat().st_size):
                with self.assertRaisesRegex(ValueError,
                                            "EN_VENDOR_ARCHIVE_DIGEST_DRIFT"):
                    vendor.source_contract(model)

    def test_vendor_files_require_genuine_archive_member_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "official-simulated"
            model.mkdir()
            (model / "tokens.txt").write_text("X 1\n")
            members = model / "test_wavs"
            members.mkdir()
            (members / "test_keywords.txt").write_text("X Y\n")
            for name, _ in vendor.VENDOR_FILES:
                (members / name).write_bytes(b"invented.wav")
            parts = {"tokens": model / "tokens.txt"}
            keyword, wavs = vendor.verified_vendor_files(
                parts, vendor.MODEL_ARCHIVE_SHA256)
            self.assertEqual(keyword, members / "test_keywords.txt")
            self.assertEqual(set(wavs), {"0.wav", "1.wav"})
            (members / "1.wav").unlink()
            with self.assertRaisesRegex(ValueError,
                                        "EN_VENDOR_ORIGINAL_WAV_MISSING"):
                vendor.verified_vendor_files(parts, vendor.MODEL_ARCHIVE_SHA256)

    def test_source_only_cli_refuses_inference_without_real_archive(self):
        path = Path(vendor.__file__)
        p = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(path)],
            capture_output=True, text=True, check=False, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        value = json.loads(p.stdout)
        self.assertEqual(value["vendor_files"], ["0.wav", "1.wav"])
        self.assertFalse(value["real_inference_run"])
        p = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(path), "--execute"],
            capture_output=True, text=True, check=False, timeout=20)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("EN_VENDOR_CREATE_ONLY_PATH", p.stderr)


if __name__ == "__main__":
    unittest.main()
