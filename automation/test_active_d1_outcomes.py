#!/usr/bin/env python3
"""Fictional numeric/audio/ASR fixtures in the actual extension receipt formats."""
from __future__ import annotations

from pathlib import Path
import shutil
import struct
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_weak_view as weak
import test_active_d1_weak_view as d1_fixture
from test_active_asr_weak_view import float_wav


def encoded(value):
    return weak.canonical(value) + b"\n"


def npy(pcm):
    header = repr({"descr": "<f4", "fortran_order": False, "shape": (len(pcm) // 4,)}).encode()
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + pcm


class D1OutcomesTests(unittest.TestCase):
    asr_fixture = d1_fixture.D1WeakViewTests.asr_fixture
    rehash_package = d1_fixture.D1WeakViewTests.rehash_package

    def setUp(self):
        d1_fixture.D1WeakViewTests.setUp(self)
        self.output = self.root / "extension-outputs"
        self.output.mkdir()
        self.generation.rename(self.output / "generation")
        self.generation = self.output / "generation"
        self.prepared = self.root / "stage"
        self.prepared.mkdir()
        self.failed_cell = self.plan["cells"][8]["cell_id"]
        self.binding = {"plan_canonical_sha256": self.plan_sha,
                        "plan_file_sha256": weak.sha(self.plan_path.read_bytes()),
                        **{name: "a" * 64 for name in ("model_assets_sha256", "source_manifest_sha256",
                            "producer_source_sha256", "runtime_setup_sha256")},
                        "generation": {"fixture_only": True, "max_new_tokens": 256}}
        self.stage = {"schema": "active-tts-extension-stage-v1", "prepared": str(self.prepared),
                      "binding": self.binding, "target_count": 24,
                      "target_cells": [c["cell_id"] for c in self.plan["cells"][:24]],
                      "initial_reused_cells": [c["cell_id"] for c in self.plan["cells"][:8]],
                      "chunks": [[c["cell_id"] for c in self.plan["cells"][8:24]]],
                      "maximum_new_calls": 16, "maximum_job_seconds": 600, "stage_budget_minutes": 20,
                      "max_retries": 0, "training_admitted": False, "paired_cases_included": False}
        weak.save_json(self.prepared, "stage.json", self.stage)
        self.job = {"schema": "active-tts-extension-job-v1", "stage": str(self.prepared / "stage.json"),
                    "stage_sha256": weak.sha(encoded(self.stage)), "job_id": "chunk-001", "chunk_index": 0,
                    "cells": self.stage["chunks"][0], "maximum_new_calls": 16, "timeout_seconds": 600}
        weak.save_json(self.prepared, "job.json", self.job)
        self.manifest["clips"] = self.manifest["clips"][8:]
        self.outcomes = {"schema": "active-tts-stage-outcomes-v1", "stage": self.job["stage"],
                         "stage_sha256": self.job["stage_sha256"], "job_id": self.job["job_id"],
                         "job_recipe_sha256": weak.sha(encoded(self.job)), "binding": self.binding,
                         "plan_canonical_sha256": self.plan_sha, "selected_cells": self.job["cells"],
                         "new_call_ids": self.job["cells"], "reused_cell_ids": [], "rows": []}
        refs = {r["reference_id"]: r for r in self.plan["references"]}
        with zipfile.ZipFile(self.archive, "w") as archive:
            for clip, cell in zip(self.manifest["clips"], self.plan["cells"][8:24]):
                name = cell["cell_id"]
                ref = refs[cell["reference_id"]]
                generated, native = self.generation / "cells" / name, self.output / "native" / name
                raw = (generated / "native-f32.wav").read_bytes()
                if name == self.failed_cell and self._testMethodName == "test_short_failed_native_retains_pcm_with_zero_loss":
                    pcm = struct.pack("<3f", 0.001, -0.001, 0.002)
                    raw = float_wav(pcm)
                    derived = struct.pack("<2f", 0.001, 0.002)
                    clip["original_wav"].update(sha256=weak.sha(raw), bytes=len(raw))
                    clip["original_pcm"].update(sha256=weak.sha(pcm), frames=3)
                    clip["derived_pcm"].update(sha256=weak.sha(derived), frames=2)
                    (self.inputs / clip["original_wav"]["path"]).write_bytes(raw)
                    (self.inputs / clip["derived_pcm"]["path"]).write_bytes(derived)
                pcm = weak.float_wav_pcm(raw)
                weak.save(native, "returned-native.wav", raw)
                weak.save(native, "returned-array.npy", npy(pcm))
                checked = ref["reference_text_status"] in ("verified", "dataset-provided-content-checked")
                weak.save_json(native, "call.json", {"schema": "active-tts-generation-call-v1", "cell_id": name,
                    "reference_id": ref["reference_id"], "source_family_id": ref["source_family_id"],
                    "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
                    "reference_text_status": ref["reference_text_status"],
                    "reference_mode": "ICL" if checked else "x-vector-only", "reference_text_passed": checked,
                    "seed": cell["seed"], "intended_text": cell["prompt"], "generation": self.binding["generation"],
                    "reference_prompt_seconds": 0.0, "attempt": 1})
                weak.save_json(native, "native.json", {"schema": "active-tts-native-return-v1",
                    "array": self.record(native / "returned-array.npy"), "wav": self.record(native / "returned-native.wav"),
                    "dtype": "<f4", "shape": [len(pcm) // 4], "sample_rate_hz": 24000,
                    "samples_sha256": weak.sha(pcm), "all_finite": True, "technical_gate_applied": False})
                weak.save_json(native, "returned.json", {"generation_seconds": 0.1, "total_seconds": 0.2,
                    "native_stop_reason": "not_exposed_by_voice_clone_API"})
                failed = name == self.failed_cell
                if failed:
                    for filename in ("completed.json", "audio-16k.wav", "native-f32.wav"):
                        (generated / filename).unlink()
                    weak.save_json(generated, "failed.json", {"schema": "kws-bulk-clone-failure-v1", "cell_id": name,
                        "status": "FAILED_NO_RETRY", "error_type": "ValueError", "training_admitted": False})
                identity = weak.sha(weak.canonical({"binding": self.binding, "cell": cell, "reference": ref}))
                claim = {"schema": "active-tts-plan-call-claim-v1", "cell_id": name, "identity_sha256": identity,
                         "attempt": 1, "owner": {"stage_sha256": self.job["stage_sha256"], "job_id": self.job["job_id"],
                         "job_recipe_sha256": weak.sha(encoded(self.job))}, "started_at": "2026-11-05T00:00:00+00:00"}
                record = {"schema": "active-tts-plan-cell-v1", "origin": "extension-call", "cell_id": name,
                          "identity_sha256": identity, "claim_sha256": weak.sha(encoded(claim)),
                          "new_generation_calls": 1, "training_admitted": False,
                          "status": "FAILED_NO_RETRY" if failed else "GENERATED_PENDING_ASR",
                          "technical_failure_reason": "FICTIONAL_GATE_FAILURE" if failed else None,
                          "output_root": str(self.output), "generation_root": str(self.generation),
                          "native_wav": ("native/" + name + "/returned-native.wav" if failed else
                                         "generation/cells/" + name + "/native-f32.wav"),
                          "files": {p.relative_to(self.output).as_posix(): self.record(p)
                                    for folder in (generated, native) for p in folder.rglob("*") if p.is_file()}}
                self.outcomes["rows"].append({"cell_id": name, "ledger_record_sha256": weak.sha(encoded(record)),
                                              "record": record, "claim": claim})
                archive.writestr(clip["original_member"], raw)
        self.refresh_inputs()
        self.refresh_outcomes()

    @staticmethod
    def record(path):
        raw = path.read_bytes()
        return {"bytes": len(raw), "sha256": weak.sha(raw)}

    def refresh_inputs(self):
        self.manifest["source_archive_sha256"] = weak.sha(self.archive.read_bytes())
        (self.inputs / "inputs.json").write_bytes(encoded(self.manifest))
        self.input_sha = weak.sha(encoded(self.manifest))
        shutil.rmtree(self.asr)
        self.asr.mkdir()
        cells = {c["cell_id"]: c for c in self.plan["cells"]}
        for clip in self.manifest["clips"]:
            cell = cells[clip["original_member"].split("/")[1]]
            for model in weak.MODELS:
                self.asr_fixture(clip, model, cell["prompt"])

    def refresh_outcomes(self):
        for row in self.outcomes["rows"]:
            row["record"]["claim_sha256"] = weak.sha(encoded(row["claim"]))
            row["ledger_record_sha256"] = weak.sha(encoded(row["record"]))
        (self.prepared / "outcomes.json").write_bytes(encoded(self.outcomes))

    def options(self):
        return {"d1_plan": self.plan_path, "d1_plan_sha256": self.plan_sha,
                "d1_outcomes": self.prepared / "outcomes.json",
                "d1_outcomes_sha256": weak.sha((self.prepared / "outcomes.json").read_bytes()),
                "d1_stage": self.prepared / "stage.json", "d1_job_recipe": self.prepared / "job.json"}

    def create(self):
        path = self.root / "view-v2"
        report = weak.create_view(self.inputs / "inputs.json", self.input_sha, self.archive, self.asr, path,
                                  fsmn_tokens=self.tokens, **self.options())
        return path, report

    def mutate_file(self, filename, edit):
        row = self.outcomes["rows"][0]
        relative = "native/" + self.failed_cell + "/" + filename
        path = self.output / relative
        value = weak.loads(path.read_bytes()); edit(value)
        path.write_bytes(encoded(value))
        row["record"]["files"][relative] = self.record(path)
        self.refresh_outcomes()

    def test_mixed_outcomes_remain_replayable_without_original_roots(self):
        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        mapping = weak.build_d1_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **self.options())
        self.assertEqual(len(mapping["rows"]), 16)
        self.assertEqual(mapping["rows"][0]["quality_reasons"], ["original_generator_signal_invalid"])
        self.assertNotIn("intended_text", weak.canonical(mapping).decode())
        path, report = self.create()
        self.assertEqual(report["source_adapter"]["schema"], weak.D1_ADAPTER_V2)
        self.assertEqual((report["statistics"]["machine_candidates"], report["statistics"]["uncertain"]), (15, 1))
        failed = report["items"][0]
        self.assertEqual(failed["source"]["original_generation_row"]["status"], "FAILED_NO_RETRY")
        self.assertEqual(failed["source"]["generation_outcome"], "failed-native")
        self.assertEqual([o["status"] for o in failed["asr"]], ["complete"] * 3)
        self.assertEqual(failed["decision"]["keyword_targets"], [None, None])
        self.assertIsNone(failed["decision"]["full_text"])
        self.assertEqual(failed["decision"]["keyword_mask"], [False, False])
        self.assertEqual(failed["decision"]["keyword_weight"], [0.0, 0.0])
        self.assertFalse(failed["decision"]["ctc_mask"])
        self.assertEqual(failed["decision"]["ctc_weight"], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        shutil.rmtree(self.output); shutil.rmtree(self.prepared)
        rows = [*weak.load_training_rows(path), *weak.load_development_rows(path)]
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["generation_outcome"] == "completed" for row in rows))

    def test_short_failed_native_retains_pcm_with_zero_loss(self):
        path, report = self.create()
        self.assertEqual(report["items"][0]["audio_derivation"]["frames"], 2)
        self.assertIsNone(report["items"][0]["decision"]["full_text"])
        self.assertEqual(len([*weak.load_training_rows(path), *weak.load_development_rows(path)]), 15)

    def test_external_outcomes_pin_is_required(self):
        options = self.options(); options["d1_outcomes_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "D1_OUTCOMES_PIN_CHANGED"):
            weak.build_d1_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **options)

    def test_rehashed_wrong_reference_mode_and_call_owner_rejected(self):
        self.mutate_file("call.json", lambda row: row.update(reference_mode="wrong"))
        with self.assertRaisesRegex(ValueError, "D1_NATIVE_CALL_REFERENCE_MODE_JOIN"):
            self.create()
        self.outcomes["rows"][0]["claim"]["owner"]["job_id"] = "another-job"
        self.refresh_outcomes()
        with self.assertRaisesRegex(ValueError, "D1_NEW_CALL_OWNER"):
            self.create()

    def test_rehashed_array_and_wav_cannot_disagree(self):
        native = self.output / "native" / self.failed_cell
        array = native / "returned-array.npy"
        raw = array.read_bytes(); array.write_bytes(raw[:-4] + struct.pack("<f", 0.123))
        self.outcomes["rows"][0]["record"]["files"][array.relative_to(self.output).as_posix()] = self.record(array)
        self.mutate_file("native.json", lambda row: row.update(array=self.record(array),
            samples_sha256=weak.sha(weak.float_wav_pcm((native / "returned-native.wav").read_bytes())[:-4] + struct.pack("<f", .123))))
        with self.assertRaisesRegex(ValueError, "D1_NATIVE_ARRAY_PCM_JOIN"):
            self.create()

    def test_failed_receipt_cannot_be_promoted_to_completed(self):
        self.outcomes["rows"][0]["record"]["status"] = "GENERATED_PENDING_ASR"
        self.refresh_outcomes()
        with self.assertRaisesRegex(ValueError, "D1_NATIVE_FILE_ROSTER"):
            self.create()

    def test_rehashed_view_cannot_assign_failed_native_negative_loss(self):
        path, report = self.create()
        report["items"][0]["decision"].update(keyword_targets=[0, 0], keyword_mask=[True, True],
                                             keyword_weight=[1.0, 1.0])
        (path / "report.json").write_bytes(encoded(report)); self.rehash_package(path)
        with self.assertRaisesRegex(ValueError, "WEAK_VIEW_DIFFERS_FROM_RETAINED_EVIDENCE"):
            list(weak.load_training_rows(path))

    def test_reused_outcome_must_not_be_new_asr_input(self):
        name = self.outcomes["new_call_ids"][0]
        self.outcomes["new_call_ids"] = self.outcomes["new_call_ids"][1:]
        self.outcomes["reused_cell_ids"] = [name]
        self.refresh_outcomes()
        with self.assertRaisesRegex(ValueError, "D1_ARCHIVE_MUST_BE_NEW_CALL_SUBSET"):
            self.create()

    def test_explicit_artifact_relocation_keeps_original_receipts(self):
        previous = str(self.output)
        moved = self.output.with_name("relocated-artifacts")
        self.output.rename(moved)
        options = self.options(); options["d1_artifact_roots"] = {previous: str(moved)}
        mapping = weak.build_d1_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **options)
        self.assertEqual(len(mapping["rows"]), 16)
        self.assertEqual(self.outcomes["rows"][0]["record"]["output_root"], previous)

    def test_unrecognizable_row_stays_in_outcomes_without_becoming_input(self):
        row = self.outcomes["rows"][0]
        row["record"]["native_wav"] = None
        row["record"]["technical_failure_reason"] = "NONFINITE_ARRAY"
        self.refresh_outcomes()
        clips = self.manifest["clips"][1:]
        self.manifest["clips"] = clips
        with zipfile.ZipFile(self.archive, "w") as archive:
            for clip in clips:
                archive.writestr(clip["original_member"], (self.inputs / clip["original_wav"]["path"]).read_bytes())
        self.refresh_inputs()
        path, report = self.create()
        self.assertEqual(len(report["items"]), 15)
        self.assertEqual(weak.loads((path / "evidence/source/d1/outcomes.json").read_bytes()), self.outcomes)
        self.assertEqual(len([*weak.load_training_rows(path), *weak.load_development_rows(path)]), 15)

    def test_failed_reference_ancestor_still_participates_in_split_guard(self):
        path, report = self.create()
        source = report["items"][0]["source"]
        opposite = "development-only" if source["role"] == "train-candidate" else "train-candidate"
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK"):
            list(weak.load_training_rows(path, other_split_sources=[
                {"role": opposite, "wav_sha256": source["parent_source_sha256"]}]))


if __name__ == "__main__":
    unittest.main()
