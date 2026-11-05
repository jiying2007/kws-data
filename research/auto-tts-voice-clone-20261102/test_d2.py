"""D2 executable regressions with artificial PCM, no neural models or network."""
from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bulk_asr
import bulk_pool as pool
import bulk_runner
import d2_pool as d2
from kws_data_contract import source_registry
from test_bulk import Fixture, FakeClone, hash_bytes, write_ref, writer


class D2Tests(Fixture):
    def setUp(self):
        super().setUp()
        self.old_sources = self.d1_sources()
        self.parent = copy.deepcopy(pool.plan(self.old_sources, self.root, batch_id="d1-fixture",
                                               revision="b" * 40, layout=pool.D1_LAYOUT))
        self.parent_raw = pool.canonical(self.parent)
        self.registry = source_registry.new_registry()
        self.registry["entries"] = [{"speaker_key": "original-" + str(index),
                                    **{key: ref[key] for key in d2.ASSIGNMENT_KEYS}}
                                   for index, ref in enumerate(self.old_sources["references"])]
        self.plan = self.make_plan()

    def make_plan(self, **changes):
        arguments = dict(parent=self.parent, parent_sha=d2.digest(self.parent),
                         registry=self.registry, stage=d2.EXISTING, batch_id="d2-fixture")
        arguments.update(changes)
        return d2.plan(self.old_sources, self.root, **arguments)

    def new_sources(self):
        self.registry = source_registry.extend(self.registry, "test-open-audio",
                                                ["new-" + str(i) for i in range(24)])
        result = {"schema": pool.SCHEMA, "references": [],
                  "training_admitted": False, "qualification_allowed": False}
        for index, identity in enumerate(self.registry["entries"][8:]):
            ref = copy.deepcopy(self.old_sources["references"][0])
            ref.update({key: identity[key] for key in d2.ASSIGNMENT_KEYS})
            ref.update(reference_id="new-ref-" + str(index), audio_path="new-ref-" + str(index) + ".wav",
                       rights_record_path="new-rights-" + str(index) + ".json")
            measured = write_ref(self.root / ref["audio_path"], 15000 + index * 100)
            ref.update({key: measured[key] for key in ("wav_sha256", "pcm_sha256", "frames", "sample_rate_hz")})
            ref["parent_source_sha256"] = measured["wav_sha256"]
            proof = pool.read_json(self.root / self.old_sources["references"][0]["rights_record_path"])
            proof.update(reference_id=ref["reference_id"], wav_sha256=ref["wav_sha256"])
            body = pool.canonical(proof)
            (self.root / ref["rights_record_path"]).write_bytes(body)
            ref["rights_record_sha256"] = hash_bytes(body)
            result["references"].append(ref)
        return result

    def new_plan(self, sources):
        return d2.plan(sources, self.root, parent=self.parent, parent_sha=d2.digest(self.parent),
                       registry=self.registry, stage=d2.NEW, batch_id="d2-fixture")

    def test_two_independent_increments_make_exact_32_by_64_without_reissuing_d1(self):
        first_ids = {cell["cell_id"] for cell in self.plan["cells"]}
        second = self.new_plan(self.new_sources())
        d2.verify_plan(second)
        all_cells = self.parent["cells"] + self.plan["cells"] + second["cells"]
        self.assertEqual(len(all_cells), 2048)
        self.assertEqual(len({row["cell_id"] for row in all_cells}), 2048)
        self.assertEqual(len(second["cells"]), 1536)
        by_source = {}
        for cell in all_cells:
            by_source.setdefault(cell["source_family_id"], []).append(cell)
        self.assertEqual(len(by_source), 32)
        for own in by_source.values():
            self.assertEqual(len(own), 64)
            self.assertEqual(d2.requested_counts(own), d2.TARGET_COUNTS)
        self.assertEqual(Counter(row["role"] for row in all_cells),
                         {"train-candidate": 1536, "development-only": 512})
        self.assertEqual(pool.canonical(self.parent), self.parent_raw)
        self.assertEqual(first_ids, {cell["cell_id"] for cell in self.make_plan()["cells"]})
        self.assertIsNone(d2.summary(second)["observed_usable_cells"])

    def test_existing_stage_works_before_any_new_source_exists_and_is_not_new_authority(self):
        self.assertEqual(len(self.registry["entries"]), 8)
        self.assertEqual(len(self.plan["cells"]), 256)
        d2.verify_plan(self.plan)
        self.assertEqual(bulk_runner.rebuild_plan(self.plan, self.old_sources, self.root), self.plan)
        self.assertTrue(all(cell["label_authority"] == "intended_only" for cell in self.plan["cells"]))
        self.assertEqual(self.plan["budgets"]["max_retries"], 0)
        self.assertEqual(self.plan["budgets"]["max_calls_per_invocation"], 64)
        self.assertFalse(self.plan["training_admitted"])

    def test_old_split_and_reference_ancestry_are_fixed(self):
        old = copy.deepcopy(self.old_sources)
        self.old_sources["references"][0]["role"] = "development-only"
        with self.assertRaisesRegex(ValueError, "D2_ORIGINAL_SOURCE_CHANGED"):
            self.make_plan()
        self.old_sources = old
        self.old_sources["references"][0]["parent_source_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "D2_ORIGINAL_SOURCE_CHANGED"):
            self.make_plan()

    def test_registry_cannot_silently_reassign_old_source(self):
        self.registry["entries"][0]["role"] = "development-only"
        with self.assertRaisesRegex(ValueError, "D2_SOURCE_NOT_IN_PERSISTENT_REGISTRY"):
            self.make_plan()

    def test_parent_digest_and_self_assigned_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "D2_PARENT_PLAN_CHANGED"):
            self.make_plan(parent_sha="f" * 64)
        for key, value in (("training_admitted", True), ("actual_spoken_labels_verified", True)):
            forged = copy.deepcopy(self.plan)
            forged[key] = value
            with self.assertRaisesRegex(ValueError, "D2_PLAN_CHANGED"):
                d2.verify_plan(forged)
        forged = copy.deepcopy(self.plan)
        forged["cells"][0]["label_authority"] = "asr-observed"
        with self.assertRaisesRegex(ValueError, "D2_PLAN_CHANGED"):
            d2.verify_plan(forged)

    def test_missing_new_references_do_not_masquerade_as_24(self):
        with self.assertRaisesRegex(ValueError, "D2_STAGE_REFERENCE_COUNT"):
            self.make_plan(stage=d2.NEW)

    def test_same_d1_audio_cannot_be_a_new_speaker_even_in_same_split(self):
        sources = self.new_sources()
        ref = sources["references"][0]
        anchor = self.old_sources["references"][0]
        ref.update({key: anchor[key] for key in
                    ("audio_path", "wav_sha256", "pcm_sha256", "frames", "sample_rate_hz", "parent_source_sha256")})
        proof = pool.read_json(self.root / ref["rights_record_path"])
        proof["wav_sha256"] = ref["wav_sha256"]
        body = pool.canonical(proof)
        (self.root / ref["rights_record_path"]).write_bytes(body)
        ref["rights_record_sha256"] = hash_bytes(body)
        with self.assertRaisesRegex(ValueError, "D2_REFERENCE_ANCESTRY_ALIAS"):
            self.new_plan(sources)

    def test_existing_bulk_worker_resumes_only_new_cells_without_retries(self):
        backend = FakeClone()
        self.gen(8)
        next_cell = self.plan["cells"][8]
        folder = self.tts / "cells" / next_cell["cell_id"]
        folder.mkdir()
        (folder / "started.json").write_bytes(b'{"fixture":"consumed unknown"}')
        result = bulk_runner.run(self.plan, self.old_sources, self.root, self.tts,
                                 backend, max_new_calls=8, writer=writer)
        self.assertEqual(result["total_started"], 17)
        self.assertEqual(result["total_completed"], 16)
        self.assertNotIn(next_cell["cell_id"], backend.calls)
        self.assertFalse(set(backend.calls) & {cell["cell_id"] for cell in self.parent["cells"]})
        self.assertEqual(pool.canonical(self.parent), self.parent_raw)

    def gen(self, amount):
        return bulk_runner.run(self.plan, self.old_sources, self.root, self.tts,
                               self.backend, max_new_calls=amount, writer=writer)

    def test_existing_native_asr_dispatch_accepts_increment_schema(self):
        self.gen(2)
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            result = bulk_asr.dispatch(self.plan, self.tts, self.asr, self.adapters,
                                       max_new_calls=6, executor=self.simulate_asr, execute=True)
        self.assertEqual(result["actual_calls"], 6)
        report = bulk_asr.screen_results(self.plan, self.tts, self.asr, self.adapters)
        self.assertFalse(report["training_admitted"])


if __name__ == "__main__":
    unittest.main()
