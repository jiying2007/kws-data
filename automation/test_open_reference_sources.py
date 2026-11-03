#!/usr/bin/env python3
"""Artificial local audio only; public-source matching does not clone a speaker."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "automation"))
sys.path.insert(0, str(ROOT / "research" / "auto-tts-voice-clone-20261102"))
import open_reference_sources as sources  # noqa: E402
import bulk_pool as pool  # noqa: E402
import research_only_bulk_ingest as importer  # noqa: E402


def hexdigest(data):
    return hashlib.sha256(data).hexdigest()


def artificial_wav(path, seconds, value):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", value) * (16000 * seconds))


class OpenSourceCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.original = self.work / "original"
        self.normalized = self.work / "normalized"
        self.evidence = self.work / "evidence"
        for root in (self.original, self.normalized, self.evidence):
            root.mkdir()
        self.terms = self.work / "publisher-terms.txt"
        self.terms.write_text("Artificial publisher license snapshot fixture", encoding="utf-8")
        self.out = self.work / "candidate-catalog.json"
        self.index = {"schema": sources.INDEX_SCHEMA, "dataset_id": "aishell3-slr93",
                      "dataset_release": "SLR93-fake-fixture-not-real",
                      "license_snapshot_sha256": hexdigest(self.terms.read_bytes()),
                      "rows": []}
        self.add_clip("speaker-one", "one-3", 3, 1000)
        self.add_clip("speaker-one", "one-5", 5, 1200)
        self.add_clip("speaker-two", "two-5", 5, 2000)

    def add_clip(self, speaker, key, seconds, value, *, converted=False):
        raw = self.original / (key + ".wav")
        wav = self.normalized / (key + ".wav")
        artificial_wav(raw, seconds, value)
        if converted:
            artificial_wav(wav, seconds, value + 100)
        else:
            wav.write_bytes(raw.read_bytes())
        receipt_path = self.evidence / (key + ".json")
        declared = {
            "schema": sources.CONVERSION_SCHEMA,
            "original_sha256": hexdigest(raw.read_bytes()),
            "normalized_sha256": hexdigest(wav.read_bytes()),
            "conversion": "externally-documented" if converted else "identity",
            "conversion_recipe_sha256": "a" * 64 if converted else None,
        }
        receipt_path.write_bytes(pool.canonical(declared) + b"\n")
        self.index["rows"].append({
            "speaker_key": speaker, "original_path": raw.name,
            "original_sha256": declared["original_sha256"],
            "normalized_path": wav.name,
            "normalized_sha256": declared["normalized_sha256"],
            "conversion_receipt_path": receipt_path.name,
        })

    def run_catalog(self, **kwargs):
        return sources.candidates(self.index, self.original, self.normalized,
                                  self.evidence, self.terms, 8, **kwargs)

    def test_registry_pinned_and_policy_closed(self):
        datasets = sources.registry()
        self.assertEqual(len(datasets), 7)
        self.assertEqual(datasets["aishell3-slr93"]["reference_status"],
                         "CANDIDATE_REVIEW_REQUIRED")
        self.assertEqual(datasets["wenetspeech-slr121"]["reference_status"],
                         "BLOCKED_FOR_CLONING")
        self.assertEqual(datasets["commonvoice-mdc"]["reference_status"],
                         "BLOCKED_FOR_CLONING")

    def test_selects_one_original_per_speaker_and_no_false_transcript(self):
        catalog = self.run_catalog()
        rows = catalog["rows"]
        self.assertEqual(catalog["schema"], importer.CATALOG_SCHEMA)
        importer._catalog(catalog)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["speaker_key"] for r in rows},
                         {"speaker-one", "speaker-two"})
        self.assertIn("one-5.wav", [r["audio_path"] for r in rows])
        self.assertNotIn("one-3.wav", [r["audio_path"] for r in rows])
        for row in rows:
            self.assertEqual(row["reference_text_status"], "unknown")
            self.assertIsNone(row["reference_text"])
            self.assertEqual(len(row["source_original_sha256"]), 64)

    def test_public_license_does_not_grant_research_scope(self):
        catalog = self.run_catalog()
        self.assertEqual(catalog["schema"], sources.CATALOG_SCHEMA)
        scope = {"schema": importer.SCHEMA, "dataset_id": "aishell3-slr93",
                 "dataset_release": self.index["dataset_release"],
                 "source_type": "reviewed-open-audio",
                 "source_license": "Apache-2.0",
                 "source_terms_sha256": self.index["license_snapshot_sha256"],
                 "research_review_sha256": hexdigest(b"not-reviewed"),
                 "intended_use": importer.ALLOWED_PURPOSE,
                 "research_tts_references_permitted": False,
                 "public_voice_export_permitted": False,
                 "forbidden_uses": [], "review_status": "unreviewed"}
        with self.assertRaisesRegex(ValueError, "RESEARCH_TTS_SCOPE_NOT_REVIEWED"):
            importer.validate_research_scope(
                scope, self.index, b"not-reviewed", self.terms.read_bytes())

    def test_noncommercial_third_party_dataset_blocked_even_if_present(self):
        for dataset_id in ("wenetspeech-slr121", "commonvoice-mdc"):
            with self.subTest(dataset_id=dataset_id):
                self.index["dataset_id"] = dataset_id
                with self.assertRaisesRegex(ValueError, "DATASET_NOT_APPROVED_FOR_REFERENCE_SCREEN"):
                    self.run_catalog()

    def test_source_and_normalized_mutations_fail_closed(self):
        path = self.original / self.index["rows"][0]["original_path"]
        path.write_bytes(path.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "ORIGINAL_DATASET_AUDIO_CHANGED"):
            self.run_catalog()
        path.write_bytes(path.read_bytes()[:-1])
        normalized = self.normalized / self.index["rows"][0]["normalized_path"]
        normalized.write_bytes(normalized.read_bytes() + b"x")
        with self.assertRaises(ValueError):
            self.run_catalog()

    def test_conversion_proof_and_terms_snapshot_cannot_drift(self):
        receipt = self.evidence / self.index["rows"][0]["conversion_receipt_path"]
        value = json.loads(receipt.read_text())
        value["original_sha256"] = "0" * 64
        receipt.write_bytes(pool.canonical(value))
        with self.assertRaisesRegex(ValueError, "CONVERSION_SOURCE_RECEIPT_MISMATCH"):
            self.run_catalog()
        value["original_sha256"] = self.index["rows"][0]["original_sha256"]
        receipt.write_bytes(pool.canonical(value))
        self.terms.write_text("changed terms")
        with self.assertRaisesRegex(ValueError, "TERMS_SNAPSHOT_DRIFT"):
            self.run_catalog()

    def test_converted_audio_is_candidate_only_not_execution_evidence(self):
        self.add_clip("speaker-three", "third-5", 5, 3000, converted=True)
        cat = self.run_catalog()
        self.assertEqual(len(cat["rows"]), 3)
        self.assertEqual(cat["rows"][-1]["reference_text_status"], "unknown")

    def test_duplicate_pcm_and_same_original_cross_speaker_rejected(self):
        self.add_clip("speaker-three", "three-5", 5, 4000, converted=True)
        third = self.index["rows"][-1]
        wav = self.normalized / third["normalized_path"]
        wav.write_bytes((self.normalized / "two-5.wav").read_bytes())
        third["normalized_sha256"] = hexdigest(wav.read_bytes())
        receipt = self.evidence / third["conversion_receipt_path"]
        value = json.loads(receipt.read_text())
        value["normalized_sha256"] = third["normalized_sha256"]
        receipt.write_bytes(pool.canonical(value))
        with self.assertRaisesRegex(ValueError, "CROSS_SPEAKER_DUPLICATE_NORMALIZED_PCM"):
            self.run_catalog()
        # A raw original clip cannot be relabelled as a new source speaker.
        self.index["rows"][-1]["original_path"] = "two-5.wav"
        self.index["rows"][-1]["original_sha256"] = self.index["rows"][2]["original_sha256"]
        value["original_sha256"] = self.index["rows"][-1]["original_sha256"]
        receipt.write_bytes(pool.canonical(value))
        with self.assertRaisesRegex(ValueError, "CROSS_SPEAKER_SAME_SOURCE_AUDIO"):
            self.run_catalog()

    def test_output_create_once_0600_and_stdout_has_no_speaker_identity(self):
        index_file = self.work / "source-index.json"
        index_file.write_bytes(pool.canonical(self.index) + b"\n")
        command = ["--index", str(index_file),
                   "--original-root", str(self.original),
                   "--normalized-root", str(self.normalized),
                   "--conversion-evidence-root", str(self.evidence),
                   "--license-snapshot", str(self.terms),
                   "--max-speakers", "2", "--output", str(self.out)]
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(sources.main(command), 0)
        self.assertFalse(self.out.exists())
        self.assertNotIn("speaker-one", out.getvalue())
        with redirect_stdout(io.StringIO()) as produced:
            self.assertEqual(sources.main(command + ["--execute"]), 0)
        self.assertTrue(self.out.is_file())
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode), 0o600)
        self.assertEqual(len(pool.read_json(self.out)["rows"]), 2)
        self.assertNotIn("speaker-one", produced.getvalue())
        with self.assertRaisesRegex(ValueError, "PRIVATE_NEW_OUTPUT_REQUIRED"):
            sources.create_only(self.out, b"overwrite")

    def test_symlink_and_unlisted_license_do_not_authorize(self):
        target = self.original / self.index["rows"][0]["original_path"]
        alias = self.original / "alias.wav"
        alias.symlink_to(target)
        self.index["rows"][0]["original_path"] = alias.name
        with self.assertRaisesRegex(ValueError, "INPUT_PATH_ALIAS"):
            self.run_catalog()

    def test_invalid_index_entries_and_budget(self):
        with self.assertRaisesRegex(ValueError, "CANDIDATE_SPEAKER_BUDGET"):
            sources.candidates(self.index, self.original, self.normalized,
                               self.evidence, self.terms, True)
        self.index["rows"][0]["unexpected"] = "silently accepted?"
        with self.assertRaisesRegex(ValueError, "SOURCE_INDEX_ROW"):
            self.run_catalog()


if __name__ == "__main__":
    unittest.main()
