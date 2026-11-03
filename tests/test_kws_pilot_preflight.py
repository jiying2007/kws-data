#!/usr/bin/env python3
"""No-model tests for source-bound small-pilot planning and real byte locks."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "research/auto-tts-voice-clone-20261102"))
import bulk_pool as pool  # noqa: E402
import bulk_runner as tts  # noqa: E402
from test_bulk import Fixture  # noqa: E402
import kws_pilot_preflight as pilot  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


class PilotReadinessTests(Fixture):
    def setUp(self):
        super().setUp()
        self.plan = pool.plan(self.sources, self.root, repetitions=1,
                              batch_id="pilot-01", revision="b" * 40,
                              max_calls=32)
        self.model_root = self.root / "tts-model"
        self.model_root.mkdir()
        (self.model_root / "config.json").write_bytes(b"offline-tiny-model-fixture")
        body = (self.model_root / "config.json").read_bytes()
        self.model_lock = {
            "schema": tts.LOCK_SCHEMA, "model_id": pool.MODEL,
            "revision": self.plan["model_revision"],
            "files": [{"path": "config.json", "sha256": sha(body),
                       "bytes": len(body)}],
        }
        self.inventory = {"schema": pilot.ASR_ASSETS_SCHEMA, "models": []}
        for adapter in self.adapters["adapters"]:
            engine = adapter["engine"]
            root = self.root / ("native-" + engine)
            root.mkdir()
            (root / "weights.bin").write_bytes(("fake-" + engine).encode())
            contents = (root / "weights.bin").read_bytes()
            files = [{"path": "weights.bin", "sha256": sha(contents),
                      "bytes": len(contents)}]
            declaration = {
                "schema": pilot.ASR_DIGEST_SCHEMA,
                "engine": engine, "model_id": adapter["model_id"],
                "model_revision": adapter["model_revision"], "files": files,
            }
            adapter["model_assets_sha256"] = sha(pool.canonical(declaration))
            self.inventory["models"].append({
                "engine": engine, "model_id": adapter["model_id"],
                "model_revision": adapter["model_revision"],
                "root": str(root), "files": files,
            })

    def audit(self, calls=32, families=2):
        return pilot.audit(self.plan, self.sources, self.root,
                           calls, 3 * calls, families)

    def test_32_cell_pilot_covers_two_phrases_two_families_and_both_negatives(self):
        result = self.audit()
        self.assertEqual(result["status"], "SOURCE_CONSISTENT_PILOT_PLANNED")
        self.assertEqual(result["coverage_gaps"], [])
        self.assertEqual(result["selected_coverage"]["source_families"], 2)
        for key in ("kw1", "kw2", "hard_negatives", "ordinary_negatives"):
            self.assertGreater(result["selected_coverage"][key], 0)
        self.assertTrue(result["selected_coverage"]["train_source_present"])
        self.assertFalse(result["training_admitted"])
        self.assertFalse(result["rights_externally_authenticated"])
        self.assertEqual(result["new_model_calls"], 0)
        self.assertGreater(result["min_tts_calls_for_coverage"], 16)

    def test_short_prefix_discloses_imbalanced_family_coverage(self):
        result = self.audit(16)
        self.assertEqual(result["status"], "PILOT_COVERAGE_HOLD")
        self.assertIn("INSUFFICIENT_SOURCE_FAMILIES", result["coverage_gaps"])
        self.assertEqual(result["selected_coverage"]["source_families"], 1)
        self.assertEqual(result["min_tts_calls_for_coverage"], self.audit()["min_tts_calls_for_coverage"])

    def test_no_silent_partial_asr_or_boolean_budget(self):
        with self.assertRaisesRegex(ValueError, "PILOT_THREE_ASR_PER_WAV_REQUIRED"):
            pilot.audit(self.plan, self.sources, self.root, 32, 95, 2)
        with self.assertRaisesRegex(ValueError, "PILOT_TTS_BUDGET"):
            pilot.audit(self.plan, self.sources, self.root, True, 3, 2)
        with self.assertRaisesRegex(ValueError, "PILOT_FAMILY_REQUIREMENT"):
            pilot.audit(self.plan, self.sources, self.root, 32, 96, 0)

    def test_actual_asr_weight_and_executable_bytes_verified(self):
        result = pilot.verify_native_assets(self.adapters, self.inventory)
        self.assertEqual(result["models_verified"], 3)
        self.assertGreater(result["total_model_bytes_verified"], 0)
        with patch.object(tts.QwenBulkClone, "_model",
                          side_effect=AssertionError("must never load a model")):
            self.assertEqual(self.audit()["coverage_gaps"], [])
            self.assertEqual(pilot.tts.asset_lock(self.model_root, self.model_lock,
                                                  self.plan["model_revision"]),
                             self.model_root)

    def test_weight_mutation_and_unlisted_file_are_blocked(self):
        root = Path(self.inventory["models"][0]["root"])
        target = root / "weights.bin"
        target.write_bytes(b"changed-even-if-adapter-sha-looks-valid")
        with self.assertRaisesRegex(ValueError, "ASR_MODEL_FILE_CHANGED"):
            pilot.verify_native_assets(self.adapters, self.inventory)
        target.write_bytes(b"fake-" + self.inventory["models"][0]["engine"].encode())
        (root / "unlisted.bin").write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "ASR_UNLISTED_OR_MISSING_MODEL_FILE"):
            pilot.verify_native_assets(self.adapters, self.inventory)

    def test_model_lock_claim_drift_rejected(self):
        adapters = copy.deepcopy(self.adapters)
        adapters["adapters"][0]["model_assets_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "ASR_ASSET_DECLARATION_SHA_MISMATCH"):
            pilot.verify_native_assets(adapters, self.inventory)
        inventory = copy.deepcopy(self.inventory)
        inventory["models"][1]["model_revision"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "ASR_ASSET_MODEL_IDENTITY"):
            pilot.verify_native_assets(self.adapters, inventory)

    def test_symlink_weights_and_executable_mutation_are_blocked(self):
        root = Path(self.inventory["models"][0]["root"])
        original = root / "weights.bin"
        original.unlink()
        original.symlink_to(self.exe)
        with self.assertRaisesRegex(ValueError, "ASSET_SYMLINK"):
            pilot.verify_native_assets(self.adapters, self.inventory)
        original.unlink()
        original.write_bytes(b"fake-" + self.inventory["models"][0]["engine"].encode())
        self.exe.write_bytes(self.exe.read_bytes() + b"\n# mutated")
        with self.assertRaisesRegex(ValueError, "ASR_EXECUTABLE_CHANGED"):
            pilot.verify_native_assets(self.adapters, self.inventory)

    def test_stale_source_and_rights_do_not_pass_plan_check(self):
        bad = copy.deepcopy(self.sources)
        bad["references"][0]["source_family_id"] = "different"
        with self.assertRaises(ValueError):
            pilot.audit(self.plan, bad, self.root, 32, 96, 2)
        rights = self.root / self.references[0]["rights_record_path"]
        rights.write_bytes(rights.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.audit()

    def test_rejects_external_families_that_plan_cannot_cover(self):
        result = self.audit(32, 3)
        self.assertIn("INSUFFICIENT_SOURCE_FAMILIES", result["coverage_gaps"])
        self.assertIsNone(result["min_tts_calls_for_coverage"])

    def test_cli_reports_real_bytes_only_and_never_creates_audio(self):
        def save(name, val):
            path = self.root / name
            path.write_bytes(pool.canonical(val) + b"\n")
            return str(path)
        args = [
            "--sources", save("sources.json", self.sources),
            "--plan", save("plan.json", self.plan),
            "--audio-root", str(self.root),
            "--model-root", str(self.model_root),
            "--model-lock", save("model-lock.json", self.model_lock),
            "--adapter-lock", save("adapter-lock.json", self.adapters),
            "--asr-assets", save("asr-assets.json", self.inventory),
            "--max-tts", "32", "--max-asr", "96",
        ]
        before = {str(x.relative_to(self.root)) for x in self.root.rglob("*")}
        with patch.object(tts.QwenBulkClone, "_model",
                          side_effect=AssertionError("model loaded")):
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(pilot.main(args), 0)
        after = {str(x.relative_to(self.root)) for x in self.root.rglob("*")}
        self.assertEqual(before, after)
        report = json.loads(out.getvalue())
        self.assertEqual(report["models_verified"], 3)
        self.assertGreater(report["tts_model_bytes_verified"], 0)
        self.assertFalse(report["spoken_labels_authenticated"])
        self.assertNotIn("shipping_approved", report)
        self.assertNotIn(str(self.root), out.getvalue())


if __name__ == "__main__":
    unittest.main()
