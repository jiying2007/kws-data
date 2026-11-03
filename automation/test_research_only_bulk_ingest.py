#!/usr/bin/env python3
"""Fake PCM and fictional review bytes only: no private speaker or model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "automation"))
sys.path.insert(0, str(ROOT / "research" / "auto-tts-voice-clone-20261102"))
import research_only_bulk_ingest as research
import open_reference_sources as reference
import bulk_pool as pool


def sha(data):
    return hashlib.sha256(data).hexdigest()


def wav_file(path, number):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(struct.pack("<h", number) * (5 * 16000))


class ResearchOnlySourceTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.original = self.root / "official-original"
        self.audio = self.root / "normalized"
        self.conv = self.root / "conversion"
        for folder in (self.original, self.audio, self.conv):
            folder.mkdir()
        self.terms = self.root / "terms.txt"
        self.terms.write_bytes(b"fictional AISHELL-3 research terms mock - NOT legal rights")
        self.review = b"fictional independently reviewed publisher terms - mock only"
        self.key = b"r" * 40
        rows = []
        for i in range(2):
            name = "clip" + str(i) + ".wav"
            wav_file(self.original / name, 2000 + i * 1000)
            (self.audio / name).write_bytes((self.original / name).read_bytes())
            digest = sha((self.original / name).read_bytes())
            receipt = {"schema": reference.CONVERSION_SCHEMA,
                       "original_sha256": digest, "normalized_sha256": digest,
                       "conversion": "identity", "conversion_recipe_sha256": None}
            (self.conv / ("clip" + str(i) + ".json")).write_bytes(
                pool.canonical(receipt) + b"\n")
            rows.append({"speaker_key": "speaker-" + str(i),
                         "original_path": name, "original_sha256": digest,
                         "normalized_path": name, "normalized_sha256": digest,
                         "conversion_receipt_path": "clip" + str(i) + ".json"})
        self.index = {"schema": reference.INDEX_SCHEMA,
                      "dataset_id": "aishell3-slr93",
                      "dataset_release": "fictional-test-v1",
                      "license_snapshot_sha256": sha(self.terms.read_bytes()),
                      "rows": rows}
        self.scope = {"schema": research.SCHEMA, "dataset_id": "aishell3-slr93",
                      "dataset_release": self.index["dataset_release"],
                      "source_type": "reviewed-open-audio",
                      "source_license": "Apache-2.0",
                      "source_terms_sha256": self.index["license_snapshot_sha256"],
                      "research_review_sha256": sha(self.review),
                      "intended_use": research.ALLOWED_PURPOSE,
                      "research_tts_references_permitted": True,
                      "public_voice_export_permitted": False,
                      "forbidden_uses": [],
                      "review_status": "independently-reviewed"}

    def prepare(self, **kw):
        values = dict(max_refs=8)
        values.update(kw)
        return research.prepare(self.index, self.original, self.audio, self.conv,
                                self.terms, self.scope, self.review, self.key,
                                **values)

    def test_research_scope_has_no_release_permission_field(self):
        manifest, rights = self.prepare()
        self.assertEqual(len(manifest["references"]), 2)
        self.assertEqual(len(rights), 2)
        self.assertFalse(manifest["training_admitted"])
        self.assertFalse(manifest["qualification_allowed"])
        self.assertTrue(all(row["reference_text_status"] == "unknown"
                            for row in manifest["references"]))
        self.assertEqual(len({x["source_family_id"] for x in manifest["references"]}), 2)
        altered = dict(self.scope, obsolete_release_flag=True)
        with self.assertRaisesRegex(ValueError, "RESEARCH_SCOPE_SCHEMA"):
            research.validate_research_scope(
                altered, self.index, self.review, self.terms.read_bytes())
        self.assertFalse(
            (ROOT / "research/auto-tts-voice-clone-20261102/bulk_ingest.py").exists())

    def test_publish_compatible_with_existing_bulk_planner(self):
        manifest, rights = self.prepare()
        out = self.root / "research-source-manifest.json"
        research.publish_private(manifest, rights, self.audio, out)
        self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o600)
        self.assertEqual(len(pool.validate_sources(pool.read_json(out), self.audio)), 2)
        plan = pool.plan(manifest, self.audio, repetitions=1, batch_id="pilot03",
                         revision="b" * 40, max_calls=16)
        self.assertEqual(plan["budgets"]["max_paid_usd"], 0)
        self.assertFalse(plan["source_rights_authenticated"])
        self.assertFalse(plan["training_admitted"])
        self.assertFalse(plan["shipping_approved"])
        with self.assertRaises(ValueError):
            research.publish_private(manifest, rights, self.audio, out)

    def test_inadequate_research_review_cannot_generate_source(self):
        for patch in (
            {"research_tts_references_permitted": False},
            {"review_status": "not-reviewed"},
            {"obsolete_release_flag": True},
            {"public_voice_export_permitted": True},
            {"forbidden_uses": ["cloning"]},
            {"intended_use": "unrelated-execution-mode"},
            {"research_review_sha256": "0" * 64},
        ):
            with self.subTest(patch=patch):
                changed = self.scope.copy()
                changed.update(patch)
                with self.assertRaises(ValueError):
                    research.prepare(self.index, self.original, self.audio,
                                     self.conv, self.terms, changed, self.review,
                                     self.key, max_refs=2)

    def test_source_drift_and_mismatch_do_not_pass(self):
        self.scope["dataset_release"] = "not-original"
        with self.assertRaisesRegex(ValueError, "RESEARCH_DATASET_RELEASE_DRIFT"):
            self.prepare()
        self.scope["dataset_release"] = self.index["dataset_release"]
        file = self.original / "clip0.wav"
        file.write_bytes(file.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "ORIGINAL_DATASET_AUDIO_CHANGED"):
            self.prepare()

    def test_source_registry_rejects_unreviewed_media_clone_lane(self):
        self.index["dataset_id"] = "wenetspeech-slr121"
        self.scope["dataset_id"] = "wenetspeech-slr121"
        with self.assertRaisesRegex(ValueError, "DATASET_NOT_ELIGIBLE_FOR_RESEARCH_TTS_REFERENCE"):
            self.prepare()

    def test_cli_dry_run_never_writes_or_runs_model(self):
        def save(name, val):
            path = self.root / name
            path.write_bytes(pool.canonical(val) + b"\n")
            return str(path)
        arguments = [
            "--index", save("index.json", self.index),
            "--original-root", str(self.original),
            "--audio-root", str(self.audio),
            "--conversion-root", str(self.conv),
            "--license-snapshot", str(self.terms),
            "--research-scope", save("scope.json", self.scope),
            "--research-review-evidence", str(self.root / "review.txt"),
            "--pseudonym-key-file", str(self.root / "key.bin"),
            "--output", str(self.root / "out.json"),
        ]
        (self.root / "review.txt").write_bytes(self.review)
        (self.root / "key.bin").write_bytes(self.key)
        self.assertEqual(research.main(arguments), 0)
        self.assertFalse((self.root / "out.json").exists())
        self.assertEqual(research.main(arguments + ["--execute"]), 0)
        self.assertTrue((self.root / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
