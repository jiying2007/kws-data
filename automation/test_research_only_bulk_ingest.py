#!/usr/bin/env python3
"""Fake PCM and fictional review bytes only: no private speaker or model calls."""
from __future__ import annotations

import hashlib
import copy
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
import bulk_runner
from kws_data_contract import source_registry


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
        self.registry = source_registry.extend(source_registry.new_registry(),
                          self.index["dataset_id"], [r["speaker_key"] for r in rows])
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
                                self.terms, self.scope, self.review, self.registry,
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
        self.assertNotIn("shipping_approved", plan)
        with self.assertRaises(ValueError):
            research.publish_private(manifest, rights, self.audio, out)

    def test_32_reference_intake_keeps_existing_registry_entries(self):
        original_entries = copy.deepcopy(self.registry["entries"])
        for index in range(2, 32):
            name = "clip" + str(index) + ".wav"
            wav_file(self.original / name, 5000 + index * 100)
            raw = (self.original / name).read_bytes()
            (self.audio / name).write_bytes(raw)
            digest = sha(raw)
            receipt_name = "clip" + str(index) + ".json"
            (self.conv / receipt_name).write_bytes(pool.canonical({
                "schema": reference.CONVERSION_SCHEMA, "original_sha256": digest,
                "normalized_sha256": digest, "conversion": "identity", "conversion_recipe_sha256": None}))
            self.index["rows"].append({"speaker_key": "speaker-" + str(index),
                "original_path": name, "original_sha256": digest,
                "normalized_path": name, "normalized_sha256": digest,
                "conversion_receipt_path": receipt_name})
        self.registry = source_registry.extend(self.registry, self.index["dataset_id"],
                                                [row["speaker_key"] for row in self.index["rows"]])
        manifest, rights = self.prepare(max_refs=32)
        self.assertEqual(len(manifest["references"]), 32)
        self.assertEqual(self.registry["entries"][:2], original_entries)
        research.publish_private(manifest, rights, self.audio, self.root / "d2-sources.json")
        self.assertEqual(len(pool.validate_sources(manifest, self.audio)), 32)
        with self.assertRaisesRegex(ValueError, "RESEARCH_SPEAKER_BUDGET"):
            self.prepare(max_refs=33)

    def test_publish_appends_new_references_and_terms_without_changing_old_receipts(self):
        initial, rights = self.prepare(max_refs=1)
        out = self.root / "initial.json"
        research.publish_private(initial, rights, self.audio, out)
        original = {out: out.read_bytes(), **{self.audio / rel: body for rel, body in rights}}

        expanded, rights = self.prepare()
        research.publish_private(expanded, rights, self.audio, self.root / "expanded.json")
        self.assertEqual(len(pool.validate_sources(expanded, self.audio)), 2)
        for path, body in original.items():
            self.assertEqual(path.read_bytes(), body)

        old_paths = {r["rights_record_path"] for r in expanded["references"]}
        old_rights = {self.audio / rel: body for rel, body in rights}
        self.terms.write_bytes(b"updated fictional research terms - mock only")
        self.index["license_snapshot_sha256"] = sha(self.terms.read_bytes())
        self.scope["source_terms_sha256"] = self.index["license_snapshot_sha256"]
        revised, rights = self.prepare()
        self.assertTrue(old_paths.isdisjoint(r["rights_record_path"]
                                            for r in revised["references"]))
        research.publish_private(revised, rights, self.audio, self.root / "revised.json")
        self.assertEqual(len(pool.validate_sources(revised, self.audio)), 2)
        for path, body in {**original, **old_rights}.items():
            self.assertEqual(path.read_bytes(), body)

    def test_publish_rejects_conflicting_existing_receipt_without_overwriting(self):
        manifest, rights = self.prepare()
        rel, body = rights[0]
        dest = self.audio / rel
        dest.parent.mkdir()
        dest.write_bytes(body + b"\n")
        original = dest.read_bytes()
        output = self.root / "conflicting.json"
        with self.assertRaisesRegex(ValueError, "PREEXISTING_RIGHTS_CONFLICT"):
            research.publish_private(manifest, rights, self.audio, output)
        self.assertEqual(dest.read_bytes(), original)
        self.assertFalse(output.exists())
        self.assertEqual(list(dest.parent.iterdir()), [dest])

    def test_publish_preserves_legacy_receipt_names(self):
        manifest, rights = self.prepare()
        legacy = copy.deepcopy(manifest)
        legacy_rights = []
        by_path = dict(rights)
        for ref in legacy["references"]:
            body = by_path[ref["rights_record_path"]]
            ref["rights_record_path"] = str(Path(ref["rights_record_path"]).with_name(
                ref["reference_id"] + ".json"))
            legacy_rights.append((ref["rights_record_path"], body))
        out = self.root / "legacy.json"
        research.publish_private(legacy, legacy_rights, self.audio, out)
        old_bytes = out.read_bytes()
        research.publish_private(manifest, rights, self.audio, self.root / "new.json")
        self.assertEqual(out.read_bytes(), old_bytes)
        for rel, body in legacy_rights:
            self.assertEqual((self.audio / rel).read_bytes(), body)
        self.assertEqual(len(pool.validate_sources(legacy, self.audio)), 2)
        self.assertEqual(len(pool.validate_sources(manifest, self.audio)), 2)

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
                                     self.registry, max_refs=2)

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
            "--speaker-registry", save("speakers.json", self.registry),
            "--output", str(self.root / "out.json"),
        ]
        (self.root / "review.txt").write_bytes(self.review)
        self.assertEqual(research.main(arguments), 0)
        self.assertFalse((self.root / "out.json").exists())
        self.assertEqual(research.main(arguments + ["--execute"]), 0)
        self.assertTrue((self.root / "out.json").exists())

    def test_source_assignments_survive_reloading_registry(self):
        first, _ = self.prepare()
        path = self.root / "speaker-registry.json"
        source_registry.save(path, self.registry, None)
        self.registry = source_registry.read_json(path)
        second, _ = self.prepare()
        self.assertEqual(first, second)
        self.registry = source_registry.extend(self.registry, self.index["dataset_id"],
                                               ["new-speaker"])
        self.assertEqual(first, self.prepare()[0])

    def test_unregistered_speaker_does_not_get_an_ad_hoc_identity(self):
        self.registry["entries"] = self.registry["entries"][:1]
        with self.assertRaisesRegex(ValueError, "SPEAKER_NOT_REGISTERED"):
            self.prepare()

    def test_dataset_transcript_is_preserved_and_only_content_checked_text_enables_icl(self):
        unknown, rights = self.prepare()
        initial_path = self.root / "unknown.json"
        research.publish_private(unknown, rights, self.audio, initial_path)
        old_files = [initial_path, *(self.audio / rel for rel, _ in rights)]
        before = {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                  for path in old_files}
        text = "今天的天气不错"
        document = self.root / "publisher-text.tsv"
        document.write_text("clip0.wav\t" + text + "\n", encoding="utf-8")
        options = dict(transcript_source=document, transcript_sha256=sha(document.read_bytes()),
                       transcript_format="path-text-tsv-v1")
        provided, provided_rights = self.prepare(**options)
        self.assertEqual(provided_rights, rights)
        provided_path = self.root / "with-text.json"
        research.publish_private(provided, provided_rights, self.audio, provided_path)
        before[provided_path] = (provided_path.read_bytes(), provided_path.stat().st_ino,
                                 provided_path.stat().st_mtime_ns)
        ref = next(r for r in provided["references"] if r["audio_path"] == "clip0.wav")
        self.assertEqual(ref["reference_text"], text)
        self.assertEqual(ref["reference_text_status"], "dataset-provided")
        self.assertEqual(ref["reference_text_provenance"]["original_sha256"],
                         self.index["rows"][0]["original_sha256"])
        calls = []
        class FakeModel:
            def create_voice_clone_prompt(self, **kwargs):
                calls.append(kwargs)
                return ["no-real-model"]
        backend = bulk_runner.QwenBulkClone(self.root)
        backend._model = lambda: FakeModel()
        backend._reference = lambda ref, root: ([0.1] * 100, 16000)
        backend._prompt(ref, self.audio)
        self.assertTrue(calls[-1]["x_vector_only_mode"])
        observed = {"engine": "sensevoice", "model_id": "iic/SenseVoiceSmall",
                    "model_revision": "b" * 40, "model_assets_sha256": "c" * 64,
                    "wav_sha256": ref["wav_sha256"], "pcm_sha256": ref["pcm_sha256"],
                    "sample_rate_hz": ref["sample_rate_hz"], "audio_frames": ref["frames"],
                    "actual_text": text, "input_mode": "audio-only", "decoder_completed": True}
        checks = self.root / "checks.json"
        checks.write_bytes(pool.canonical({"schema": pool.reference_text.CHECK_SCHEMA,
                                          "observations": [observed]}))
        checked, rights = self.prepare(**options, reference_content_checks=checks)
        self.assertEqual(rights, provided_rights)
        checked_ref = next(r for r in checked["references"] if r["audio_path"] == "clip0.wav")
        self.assertEqual(checked_ref["reference_text_status"],
                         "dataset-provided-content-checked")
        self.assertNotEqual(checked_ref["reference_text_status"], "verified")
        research.publish_private(checked, rights, self.audio, self.root / "content-checked.json")
        for path, unchanged in before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns),
                             unchanged)
        for stage in (unknown, provided, checked):
            self.assertEqual(len(pool.validate_sources(stage, self.audio)), 2)
            plan = pool.plan(stage, self.audio, repetitions=1, batch_id="upgrade",
                             revision="b" * 40, max_calls=16)
            self.assertFalse(plan["training_admitted"])
        backend._prompt(checked_ref, self.audio)
        self.assertFalse(calls[-1]["x_vector_only_mode"])
        self.assertEqual(calls[-1]["ref_text"], text)
        backend._prompt(checked_ref, self.audio)
        self.assertEqual(len(calls), 2)
        altered = copy.deepcopy(checked_ref)
        altered["reference_text"] = "你好小窝"
        with self.assertRaisesRegex(ValueError, "TRANSCRIPT_ORIGINAL_BINDING"):
            backend._prompt(altered, self.audio)
        # A conflicting machine transcript leaves the publisher text intact,
        # but cannot turn it into an ICL-ready/human-verified reference.
        observed["actual_text"] = "完全不同的句子"
        checks.write_bytes(pool.canonical({"schema": pool.reference_text.CHECK_SCHEMA,
                                          "observations": [observed]}))
        mismatch, _ = self.prepare(**options, reference_content_checks=checks)
        mismatch_ref = next(r for r in mismatch["references"] if r["audio_path"] == "clip0.wav")
        self.assertEqual(mismatch_ref["reference_text"], text)
        self.assertEqual(mismatch_ref["reference_text_status"], "dataset-provided")


if __name__ == "__main__":
    unittest.main()
