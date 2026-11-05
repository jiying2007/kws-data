#!/usr/bin/env python3
"""Fictional model/audit identities only; no model or network calls."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_asr_checks as checks
import active_asr_weak_view as weak


class ReferenceCheckIdentityTests(unittest.TestCase):
    def test_native_checkpoint_and_implementation_commit_are_separate(self):
        lock = {"source_repository": "https://github.com/openai/whisper", "source_revision": "a" * 40}
        raw = weak.canonical(lock)
        assets = {"model": "openai/whisper-small", "sha256": checks.WHISPER_SMALL_SHA,
                  "source_sha256": weak.sha(raw)}
        row = checks.model_fields("whisper", assets, {}, lock, raw)
        self.assertEqual(row["model_revision_kind"], "native-checkpoint-sha256")
        self.assertEqual(row["model_revision"], checks.WHISPER_SMALL_SHA)
        self.assertEqual(row["implementation_source_commit"], "a" * 40)
        self.assertEqual(row["model_assets_sha256"], weak.sha(weak.canonical(assets)))
        with self.assertRaisesRegex(ValueError, "REFERENCE_WHISPER_MODEL_IDENTITY"):
            checks.model_fields("whisper", dict(assets, sha256="a" * 40), {}, lock, raw)

    def test_repository_commit_is_bound_to_complete_assets_with_no_invented_code_revision(self):
        assets = {"model_id": "fictional/model", "revision": "a" * 40, "files": []}
        locks = {"qwen06": {"asset_lock": assets, "asset_lock_sha256": weak.sha(weak.canonical(assets))}}
        row = checks.model_fields("qwen06", assets, locks, {}, b"")
        self.assertEqual(row["model_revision_kind"], "model-repository-commit")
        self.assertEqual(row["model_revision"], "a" * 40)
        self.assertIsNone(row["implementation_source_commit"])
        with self.assertRaisesRegex(ValueError, "REFERENCE_REPOSITORY_MODEL_IDENTITY"):
            checks.model_fields("qwen06", dict(assets, revision="b" * 40), locks, {}, b"")

    def test_changed_native_body_cannot_reuse_an_older_successful_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, asr = root / "inputs", root / "asr"
            audio.mkdir()
            asr.mkdir()
            weak.save(asr, "native/fixture.json", b"original native evidence")
            files = {"native/fixture.json": {"bytes": 24, "sha256": weak.sha(b"original native evidence")}}
            inventory = {"files": files, "files_sha256": weak.sha(weak.canonical(files))}
            weak.save_json(asr, "audit/file-hashes.json", inventory)
            audit = {"status": "complete", "errors": [], "input_manifest_sha256": "a" * 64,
                     "verified_matrix_cells": 3, "original_recognition_calls": 3,
                     "duplicate_model_clip_pairs": 0, "inventory_sha256": weak.sha((asr / "audit/file-hashes.json").read_bytes()),
                     "target_context_audit": {"passed": True, "intended_text_and_peer_results_passed_to_native_recognizers": False}}
            weak.save_json(asr, "audit/final-summary.json", audit)
            checks.verify_audit(asr, audio, "a" * 64, 1)
            (asr / "native/fixture.json").write_bytes(b"altered  native evidence")
            with self.assertRaisesRegex(ValueError, "REFERENCE_ASR_EVIDENCE_(CHANGED|SIZE)"):
                checks.verify_audit(asr, audio, "a" * 64, 1)


if __name__ == "__main__":
    unittest.main()
