#!/usr/bin/env python3
"""Small PCM/data-boundary tests. No model libraries or saved-corpus replay."""
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import stat
import struct
import unittest
import warnings
import zipfile

MODULE = Path(__file__).with_name("english_temporal_quality_data.py")
SPEC = importlib.util.spec_from_file_location("english_temporal_quality_data", MODULE)
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def pcm(values):
    return b"".join(struct.pack("<h", value) for value in values)


def archive_bytes(entries):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w") as archive:
            for name, value in entries:
                archive.writestr(name, value)
    return stream.getvalue()


class FixedConditionTests(unittest.TestCase):
    def test_fixed_5db_quantized_pcm_and_unchanged_inputs(self):
        # Treat this as nonzero ordinary speech intent: mixing is label-blind.
        source = pcm([1000, -1000, 1000, -1000] * 16)
        noise = pcm([100, 100, -100, -100] * 16)
        identities = (hashlib.sha256(source).hexdigest(), hashlib.sha256(noise).hexdigest())
        mixed, receipt = DATA.mix_pcm16_5db(source, noise)
        self.assertEqual(receipt["status"], "MIXED")
        self.assertEqual(receipt["target_snr_db"], 5.0)
        self.assertEqual(len(mixed), len(source))
        values = DATA.pcm_values(mixed)
        # Analytical gain = 10 / sqrt(10**0.5); resulting noise is +/-562.
        self.assertEqual(values[:4], [1562, -438, 438, -1562])
        achieved = 10 * math.log10(1000**2 / 562**2)
        self.assertAlmostEqual(receipt["actual_snr_db"], achieved, places=12)
        self.assertLessEqual(abs(achieved - 5.0), 0.05)
        self.assertEqual(receipt["gain_on_signal"], 1.0)
        self.assertEqual(receipt["clipped_samples"], 0)
        self.assertEqual(receipt["fade_samples"], 0)
        self.assertEqual(identities, (hashlib.sha256(source).hexdigest(), hashlib.sha256(noise).hexdigest()))
        with self.assertRaises(TypeError):
            DATA.mix_pcm16_5db(source, noise, snr_db=10.0)

    def test_headroom_failure_discards_entire_positive_and_negative_peak_clips(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                source, noise = pcm([sign * 30000] * 32), pcm([sign * 1000] * 32)
                before = (source, noise)
                mixed, receipt = DATA.mix_pcm16_5db(source, noise)
                self.assertIsNone(mixed)
                self.assertEqual(receipt["status"], "MIX_HEADROOM_EXCLUDED")
                self.assertGreater(receipt["unbounded_peak"], 32768)
                self.assertNotIn("mixed_pcm_sha256", receipt)
                self.assertEqual(receipt["clipped_samples"], 0)
                self.assertEqual((source, noise), before)

    def test_zero_energy_and_quantization_failures_are_explicit_exclusions(self):
        for source, noise in (([0] * 32, [100] * 32), ([100] * 32, [0] * 32)):
            mixed, receipt = DATA.mix_pcm16_5db(pcm(source), pcm(noise))
            self.assertIsNone(mixed)
            self.assertEqual(receipt["status"], "MIX_ZERO_ENERGY_EXCLUDED")
        mixed, receipt = DATA.mix_pcm16_5db(pcm([1, -1] * 32), pcm([1] * 64))
        self.assertIsNone(mixed)
        self.assertEqual(receipt["status"], "MIX_QUANTIZATION_SNR_EXCLUDED")
        self.assertGreater(abs(receipt["actual_snr_db"] - 5.0), 0.05)

    def test_original_noise_domain_same_role_filter_and_offset_are_fixed(self):
        tracks = [
            {"id": "speaker-z", "role": "train", "frames": 1000},
            {"id": "speaker-a", "role": "train", "frames": 1200},
            {"id": "dev-only", "role": "development", "frames": 1500},
            {"id": "too-short", "role": "train", "frames": 9},
        ]
        case_id, frames = "frozen-case", 64
        digest = hashlib.sha256(("english-quality-snr10-v1/" + case_id).encode()).hexdigest()
        expected_track = sorted(tracks[:2], key=lambda row: row["id"])[int(digest[:16], 16) % 2]
        expected_offset = int(digest[16:32], 16) % (expected_track["frames"] - frames + 1)
        for input_tracks in (tracks, list(reversed(tracks))):
            track, offset = DATA.select_noise_track(case_id, "train", frames, input_tracks)
            self.assertEqual((track, offset), (expected_track, expected_offset))
            self.assertEqual(track["role"], "train")
            self.assertLessEqual(offset + frames, track["frames"])
        track, _ = DATA.select_noise_track(case_id, "development", frames, tracks)
        self.assertEqual(track["id"], "dev-only")
        with self.assertRaises(ValueError):
            DATA.select_noise_track(case_id, "unknown", frames, tracks)

    def test_unsafe_and_duplicate_archive_members_are_rejected(self):
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        for entries in (
            [("../escape", b"x")], [("/absolute", b"x")], [("bad\\name", b"x")],
            [("file", b"x"), ("file", b"y")], [(symlink, b"target")],
        ):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                DATA.Archive(archive_bytes(entries))
        good = DATA.Archive(archive_bytes([("safe/file.json", b'{"ok":true}')]))
        self.assertEqual(good.json("safe/file.json"), {"ok": True})

    def test_duplicate_or_nonfinite_json_is_rejected(self):
        for raw in (b'{"id":1,"id":2}', b'{"x":{"id":1,"id":2}}',
                    b'{"value":NaN}', b'{"value":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                DATA.unique_json(raw)
        self.assertEqual(DATA.unique_json(b'{"a":1,"b":[2]}'), {"a": 1, "b": [2]})


if __name__ == "__main__":
    unittest.main()
