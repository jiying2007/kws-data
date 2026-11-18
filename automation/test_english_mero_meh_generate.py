"""Artificial native buffers test protocol boundaries, not acoustic quality."""
from __future__ import annotations

from array import array
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_mero_meh_generate as g

MOCK_RUN = {"repository": "jiying2007/kws-data", "run_id": "999999",
            "commit": "a" * 40, "run_attempt": "1", "event": "push"}
MOCK_VOCAB = {c: n + 1 for n, c in enumerate(sorted({c for r in g.recipe.rows() for c in r["raw_phonemes"]}))}


def fixture_plan():
    names = list(g.LEGACY_PINS) + ["automation/english_mero_meh_generate.py",
        "automation/english_mero_meh_recipe.py", g.recipe.PROFILE["path"], "EXCLUSION_EVIDENCE.json"]
    return {"campaign_id": g.recipe.CAMPAIGN_ID, "native_tts_request_budget": 144,
        "maximum_total_model_calls": 720, "retry_count": 0, "warmup_count": 0,
        "time_budget_seconds": {"initialization": 30, "generation": 90},
        "source_files_sha256": {p: g.sha((g.ROOT / p).read_bytes()) for p in names},
        "recipe_sha256": g.recipe.digest(g.recipe.recipe()), "enabled": False, "main_before": None}


class MockBackend:
    """No native or numerical imports. Each output is visibly artificial."""
    def __init__(self, output, assets):
        self.vocab = MOCK_VOCAB
        self.indices = {r["id"]: i for i, r in enumerate(g.recipe.rows())}
        self.calls = []

    def forward(self, row, ids):
        self.calls.append(row["id"])
        return row

    def materialize(self, row):
        duration = [2] * (len(row["raw_phonemes"]) + 2)
        index = self.indices[row["id"]]
        values = array("f", (((i % 31) - 15) / 1024 + index / 32768 for i in range(600 * sum(duration))))
        if sys.byteorder != "little":
            values.byteswap()
        return values.tobytes(), duration

    def resample(self, raw):
        # A deliberately artificial format-only substitute. This is not scipy
        # output and cannot be accepted by the production native-lock verifier.
        count = len(raw) // 4 * 2 // 3
        return raw[:count * 4]


def mock_source(root, plan=None, backend=MockBackend):
    source = Path(root) / "source"
    result = g.generate(plan or fixture_plan(), source, Path(root) / "assets", MOCK_RUN, backend_factory=backend)
    return source, result


def refresh_manifest(root):
    (Path(root) / "FILES.json").unlink()
    g.once(Path(root) / "FILES.json", {"schema": g.FILES_SCHEMA, "files": g.file_records(root)})


class SourceBoundaryTests(unittest.TestCase):
    def test_source_contract_without_numerical_imports(self):
        self.assertEqual(g.source_preflight()["source_check"], "PASS")
        self.assertFalse(set(("numpy", "torch", "scipy", "kokoro")) & set(sys.modules))
        self.assertEqual(g.whole_support_geometry(28672)["whole_support_windows"], 3)
        self.assertEqual(g.whole_support_geometry(28673)["whole_support_windows"], 2)
        self.assertFalse(g.pcm_statistics(b"\x01\x00" * 6399)["pcm_quality_passed"])

    def test_real_protocol_writes_and_reads_every_planned_row(self):
        with tempfile.TemporaryDirectory() as folder:
            source, result = mock_source(folder)
            self.assertEqual((result["tts_attempts"], result["tts_returned"], result["successful_count"]), (144, 144, 144))
            # Only native initialization is substituted. Bytes, counters,
            # durations, request tokens and closure follow production code.
            with mock.patch.object(g, "verify_native_lock", return_value=MOCK_VOCAB):
                readback = g.source_readback(source, fixture_plan(), MOCK_RUN)
            self.assertTrue(readback["source_ready"])
            self.assertEqual(len(readback["rows"]), 144)
            self.assertTrue(g.read(source / "CORPUS_MANIFEST.json")["training_allowed"])
            with self.assertRaises((ValueError, FileNotFoundError)):
                g.validate_source(source, MOCK_RUN)  # mocks never pass actual native proof

    def test_returned_but_unmaterializable_keeps_return_counter(self):
        class Broken(MockBackend):
            def materialize(self, row):
                if self.indices[row["id"]] == 0:
                    raise ValueError("artificial_materialization_failure")
                return super().materialize(row)
        with tempfile.TemporaryDirectory() as folder:
            source, result = mock_source(folder, backend=Broken)
            self.assertEqual((result["tts_attempts"], result["tts_returned"], result["successful_count"], result["failed_count"]),
                             (144, 144, 143, 1))
            with mock.patch.object(g, "verify_native_lock", return_value=MOCK_VOCAB):
                self.assertFalse(g.source_readback(source)["source_ready"])

    def test_interruption_preserves_not_started_denominator(self):
        class Stopped(MockBackend):
            def forward(self, row, ids):
                if self.indices[row["id"]] == 1:
                    raise g.GenerationInterrupted("artificial_stop")
                return super().forward(row, ids)
        with tempfile.TemporaryDirectory() as folder:
            source, result = mock_source(folder, backend=Stopped)
            self.assertEqual((result["tts_attempts"], result["tts_returned"], result["successful_count"],
                              result["failed_count"], result["not_started_count"]), (2, 1, 1, 1, 142))
            with mock.patch.object(g, "verify_native_lock", return_value=MOCK_VOCAB):
                corpus, successful = g.validate_source(source)
            self.assertEqual((len(corpus["rows"]), len(successful)), (144, 1))

    def test_tampering_rejected_even_if_archive_manifest_refreshed(self):
        with tempfile.TemporaryDirectory() as folder:
            source, result = mock_source(folder)
            row = g.read(source / "CORPUS_MANIFEST.json")["rows"][0]
            path = source / row["pcm_path"]
            raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
            refresh_manifest(source)
            with mock.patch.object(g, "verify_native_lock", return_value=MOCK_VOCAB):
                with self.assertRaisesRegex(ValueError, "EXACT_FINAL_PCM"):
                    g.validate_source(source)

    def test_initialization_failure_is_zero_calls_with_all_rows(self):
        def failed(output, assets):
            raise RuntimeError("artificial_initialization_failure")
        with tempfile.TemporaryDirectory() as folder:
            source, result = mock_source(folder, backend=failed)
            corpus, successful = g.validate_source(source)
            self.assertEqual((result["tts_attempts"], result["not_started_count"], len(successful)), (0, 144, 0))
            self.assertEqual(len(corpus["rows"]), 144)

    def test_old_audio_exclusions_apply_across_both_hash_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = MockBackend(None, None)
            raw, _ = backend.materialize(g.recipe.rows()[0])
            exclusions = g.exclusion_sets()
            exclusions["pcm_sha256"].add(g.sha(raw))
            with mock.patch.object(g, "exclusion_sets", return_value=exclusions):
                source, result = mock_source(folder)
                row = g.read(source / "CORPUS_MANIFEST.json")["rows"][0]
                self.assertEqual(result["successful_count"], 143)
                self.assertIn("NO_REUSED_NATIVE_AUDIO", row["error"])
                with mock.patch.object(g, "verify_native_lock", return_value=MOCK_VOCAB):
                    self.assertFalse(g.source_readback(source)["source_ready"])


if __name__ == "__main__":
    unittest.main()
