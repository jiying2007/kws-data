#!/usr/bin/env python3
"""Fictional PCM16 references; no network, model or decoder execution."""
from pathlib import Path
import struct
import sys
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_asr_inputs as adapter
import test_research_only_bulk_ingest as fixture_module


class ReferenceInputTests(unittest.TestCase):
    def setUp(self):
        f = fixture_module.ResearchOnlySourceTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        # Model an existing recorded conversion: original 44.1k, normalized16k.
        for number, row in enumerate(f.index["rows"]):
            with wave.open(str(f.original / row["original_path"]), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(44100)
                writer.writeframes(struct.pack("<h", 2000 + number * 1000) * (5 * 44100))
            if number == 0:
                path = f.original / row["original_path"]
                raw = path.read_bytes()
                path.write_bytes(raw[:4] + struct.pack("<I", len(raw) - 8 + 8734) + raw[8:])
            row["original_sha256"] = adapter.sha((f.original / row["original_path"]).read_bytes())
            conversion = {"schema": fixture_module.reference.CONVERSION_SCHEMA,
                          "original_sha256": row["original_sha256"], "normalized_sha256": row["normalized_sha256"],
                          "conversion": "externally-documented", "conversion_recipe_sha256": "a" * 64}
            (f.conv / row["conversion_receipt_path"]).write_bytes(adapter.canonical(conversion) + b"\n")
        text = f.root / "publisher.tsv"
        text.write_text("clip0.wav\tFORBIDDEN TARGET CONTEXT\nclip1.wav\tANOTHER PUBLISHER SENTENCE\n")
        manifest, rights = f.prepare(transcript_source=text, transcript_sha256=adapter.sha(text.read_bytes()),
                                     transcript_format="path-text-tsv-v1")
        self.sources = f.root / "sources.json"
        fixture_module.research.publish_private(manifest, rights, f.audio, self.sources)
        self.index = f.root / "index.json"
        self.index.write_bytes(adapter.canonical(f.index) + b"\n")
        self.output = f.root / "asr-inputs"

    def prepare(self, expected=None):
        return adapter.prepare(self.sources, expected or adapter.sha(self.sources.read_bytes()),
                               self.index, self.f.original, self.f.audio, self.f.conv,
                               self.f.terms, self.output)

    def test_original_and_normalized_formats_are_honest_and_no_text_reaches_inputs(self):
        binding = self.prepare()
        inputs, rebound = adapter.verify(self.output)
        self.assertEqual(binding, rebound)
        self.assertEqual(len(inputs["clips"]), 2)
        self.assertFalse(inputs["derivation"]["resampling_during_asr_preparation"])
        self.assertEqual(sum(not row["original_riff_extent_matches"] for row in binding["rows"]), 1)
        self.assertNotIn(b"FORBIDDEN TARGET", (self.output / "inputs.json").read_bytes())
        self.assertNotIn(b"ANOTHER PUBLISHER", (self.output / "reference-bindings.json").read_bytes())
        for clip, row in zip(inputs["clips"], binding["rows"]):
            self.assertEqual(clip["original_pcm"]["sample_rate"], 44100)
            self.assertEqual(clip["original_pcm"]["dtype"], "<i2")
            self.assertEqual(clip["derived_pcm"]["frames"], row["normalized_pcm"]["frames"])
            normalized, _ = adapter.pcm16((self.output / row["normalized_wav"]["path"]).read_bytes())
            raw = (self.output / clip["derived_pcm"]["path"]).read_bytes()
            self.assertEqual(raw, adapter.float_from_pcm16(normalized))
        self.assertEqual(struct.unpack("<2f", adapter.float_from_pcm16(struct.pack("<2h", -32768, 32767))),
                         (-1.0, 32767 / 32768))
        original = (self.output / inputs["clips"][0]["original_wav"]["path"]).read_bytes()
        with self.assertRaisesRegex(ValueError, "REFERENCE_INPUT_CHUNK_EXTENT"):
            adapter.pcm16(original[:-2])

    def test_rehashed_float_tamper_cannot_change_the_normalized_samples(self):
        self.prepare()
        inputs, binding = adapter.verify(self.output)
        clip = inputs["clips"][0]
        pcm = self.output / clip["derived_pcm"]["path"]
        raw = pcm.read_bytes()
        pcm.write_bytes(struct.pack("<f", .75) + raw[4:])
        clip["derived_pcm"]["sha256"] = adapter.sha(pcm.read_bytes())
        raw_input = adapter.canonical(inputs) + b"\n"
        (self.output / "inputs.json").write_bytes(raw_input)
        binding["inputs_manifest_sha256"] = adapter.sha(raw_input)
        (self.output / "reference-bindings.json").write_bytes(adapter.canonical(binding) + b"\n")
        with self.assertRaisesRegex(ValueError, "REFERENCE_FLOAT_CONVERSION_CHANGED"):
            adapter.verify(self.output)

    def test_wrong_source_pin_and_existing_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "REFERENCE_SOURCE_MANIFEST_CHANGED"):
            self.prepare("a" * 64)
        self.prepare()
        before = {path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "NEW_PRIVATE_REFERENCE_INPUTS_REQUIRED"):
            self.prepare()
        self.assertEqual(before, {path: path.read_bytes() for path in before})


if __name__ == "__main__":
    unittest.main()
