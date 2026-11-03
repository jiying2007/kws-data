#!/usr/bin/env python3
"""Synthetic PCM fixtures only; no public audio, TTS, ASR or network."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "automation"))
sys.path.insert(0, str(ROOT / "research" / "auto-tts-voice-clone-20261102"))
import prepare_open_reference_pilot as prep
import open_reference_sources as importer
import bulk_pool as pool


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fake_wav(path, rate, seconds, sample):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", sample) * (rate * seconds))


class LocalOfficialCorpusPrepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original"
        self.original.mkdir()
        self.license = self.root / "official-terms.txt"
        self.license.write_text("Fake official terms snapshot: NOT real authorization")
        self.selection = {
            "schema": prep.SCHEMA,
            "dataset_id": "aishell3-slr93",
            "dataset_release": "invented-fixture-only",
            "license_snapshot_sha256": sha(self.license.read_bytes()),
            "clips": [],
        }
        self.add_clip("speaker-one", "first-3.wav", 16000, 3, 2200)
        self.add_clip("speaker-one", "first-5.wav", 24000, 5, 2300)
        self.add_clip("speaker-two", "second-5.wav", 16000, 5, 3400)
        self.out = self.root / "private-output"

    def add_clip(self, speaker, name, rate, seconds, sample):
        file = self.original / name
        fake_wav(file, rate, seconds, sample)
        self.selection["clips"].append({
            "speaker_key": speaker, "original_path": file.name,
            "original_sha256": sha(file.read_bytes())
        })

    def build(self, **kw):
        options = dict(max_speakers=8, execute=False)
        options.update(kw)
        return prep.build(self.selection, self.original, self.license,
                          self.out, **options)

    def test_no_write_preview_and_choose_nearest_five_seconds(self):
        status = self.build()
        self.assertEqual(status["status"], "SOURCE_PREVIEW_ONLY")
        self.assertEqual(status["selected"], 2)
        self.assertFalse(self.out.exists())
        chosen = prep.validate(self.selection, self.original, self.license)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(chosen[0][1][1]["original_path"], "first-5.wav")

    def test_canonical_private_16k_24k_and_downstream_catalog(self):
        status = self.build(execute=True)
        self.assertEqual(status["status"], "PRIVATE_CANDIDATES_PREPARED_ONLY")
        self.assertEqual(status["selected"], 2)
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode), 0o700)
        self.assertFalse(status["voice_cloning_authorized"])
        self.assertFalse(status["training_admitted"])
        index_file = self.out / "source-index.json"
        self.assertEqual(stat.S_IMODE(index_file.stat().st_mode), 0o600)
        index = pool.read_json(index_file)
        self.assertEqual(len(index["rows"]), 2)
        for row in index["rows"]:
            self.assertEqual(row["normalized_sha256"],
                             sha((self.out / "normalized" /
                                  row["normalized_path"]).read_bytes()))
            receipt = pool.read_json(self.out / "conversions" /
                                     row["conversion_receipt_path"])
            self.assertEqual(receipt["original_sha256"], row["original_sha256"])
        catalog = importer.candidates(index, self.original,
                    self.out / "normalized", self.out / "conversions",
                    self.license, 8)
        self.assertEqual(len(catalog["rows"]), 2)
        self.assertTrue(all(x["reference_text_status"] == "unknown"
                            for x in catalog["rows"]))
        with self.assertRaisesRegex(ValueError, "PRIVATE_CREATE_ONLY_OUTPUT"):
            self.build(execute=True)

    def test_changed_source_or_terms_cannot_build(self):
        target = self.original / "first-5.wav"
        target.write_bytes(target.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "ORIGINAL_SOURCE_BYTES_CHANGED"):
            self.build()
        target.write_bytes(target.read_bytes()[:-6])
        self.license.write_text("Changed fake terms")
        with self.assertRaisesRegex(ValueError, "LICENSE_SNAPSHOT_DRIFT"):
            self.build()

    def test_same_original_audio_cannot_be_second_speaker(self):
        row = dict(self.selection["clips"][2])
        row["speaker_key"] = "third-speaker"
        row["original_path"] = "second-alias.wav"
        (self.original / row["original_path"]).write_bytes(
            (self.original / "second-5.wav").read_bytes())
        self.selection["clips"].append(row)
        with self.assertRaisesRegex(ValueError, "ORIGINAL_AUDIO_CROSSES_SPEAKERS"):
            self.build()

    def test_source_licenses_never_promote_to_clone_permission(self):
        self.selection["dataset_id"] = "wenetspeech-slr121"
        with self.assertRaisesRegex(ValueError, "DATASET_NOT_IN_REVIEWABLE_REFERENCE_LANE"):
            self.build()
        self.selection["dataset_id"] = "commonvoice-mdc"
        with self.assertRaisesRegex(ValueError, "DATASET_NOT_IN_REVIEWABLE_REFERENCE_LANE"):
            self.build()

    def test_bad_input_limits_and_aliases(self):
        with self.assertRaisesRegex(ValueError, "SPEAKER_BUDGET"):
            self.build(max_speakers=True)
        with self.assertRaisesRegex(ValueError, "SPEAKER_BUDGET"):
            self.build(max_speakers=17)
        self.selection["clips"][0]["original_path"] = "../escape.wav"
        with self.assertRaises(ValueError):
            self.build()

    def test_44k_conversion_uses_bounded_pinned_local_ffmpeg(self):
        program = shutil.which("ffmpeg")
        self.assertIsNotNone(program, "CI runner must have offline ffmpeg")
        program = Path(program).resolve()
        self.add_clip("speaker-three", "third-5.wav", 44100, 5, 4100)
        original = self.original / "third-5.wav"
        original_sha = sha(original.read_bytes())
        with self.assertRaisesRegex(ValueError, "PINNED_FFMPEG_REQUIRED"):
            self.build()
        digest = prep.reference.sha_file(program, prep.MAX_EXECUTABLE_BYTES)
        with self.assertRaisesRegex(ValueError, "FFMPEG_EXECUTABLE_DRIFT"):
            self.build(converter=program, converter_sha="0" * 64,
                       execute=True)
        self.assertFalse(self.out.exists())
        result = self.build(converter=program, converter_sha=digest,
                            execute=True)
        self.assertEqual(result["selected"], 3)
        index = pool.read_json(self.out / "source-index.json")
        target = next(x for x in index["rows"]
                      if x["original_path"] == "third-5.wav")
        wav = self.out / "normalized" / target["normalized_path"]
        with wave.open(str(wav), "rb") as inp:
            self.assertEqual(inp.getframerate(), 16000)
            self.assertEqual(inp.getnchannels(), 1)
            self.assertEqual(inp.getnframes(), 80000)
        self.assertEqual(original_sha, sha(original.read_bytes()))
        receipt = pool.read_json(self.out / "conversions" /
                                 target["conversion_receipt_path"])
        self.assertEqual(receipt["conversion"], "externally-documented")
        self.assertEqual(len(receipt["conversion_recipe_sha256"]), 64)

    def test_output_catalog_never_uses_public_voice_as_gold(self):
        result = self.build(execute=True)
        self.assertFalse(result["qualification_allowed"])
        self.assertNotIn("shipping_approved", result)
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse((self.out / "permit.json").exists())


if __name__ == "__main__":
    unittest.main()
