"""Context plans and bounded native protocol using fictional PCM only."""
from collections import Counter
import copy
from pathlib import Path
import struct
import sys
import unittest

import context_supplement as context
import test_d2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "automation"))
import active_asr_weak_view as weak
import test_active_d1_weak_view as audio_fixture
from test_active_d1_outcomes import npy


class FictionalBoundedBackend:
    """The real protocol retains native return BEFORE the downstream gate."""
    def __init__(self, plan, output, selected, *, fail=False):
        self.recipe = context.runner_recipe(plan, selected)
        self.output, self.fail, self.calls = Path(output), fail, []

    def synthesize(self, cell, ref, audio_root):
        if cell["cell_id"] not in self.recipe["selected_cells"] or cell["cell_id"] in self.calls:
            raise ValueError("FICTIONAL_CALL_SCOPE")
        self.calls.append(cell["cell_id"])
        values, rate = audio_fixture.FictionalClone().synthesize(cell, ref, audio_root)
        if self.fail:
            values = [0.0] * len(values)
        pcm = struct.pack("<" + "f" * len(values), *values)
        raw = audio_fixture.weak_fixture.float_wav(pcm)
        folder = self.output / "native" / cell["cell_id"]
        weak.save(folder, "returned-native.wav", raw)
        weak.save(folder, "returned-array.npy", npy(pcm))
        checked = ref["reference_text_status"] in ("verified", "dataset-provided-content-checked")
        weak.save_json(folder, "call.json", {"schema": "active-tts-generation-call-v1", "cell_id": cell["cell_id"],
            "reference_id": ref["reference_id"], "source_family_id": ref["source_family_id"],
            "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
            "reference_text_status": ref["reference_text_status"], "reference_mode": "ICL" if checked else "x-vector-only",
            "reference_text_passed": checked, "seed": cell["seed"], "intended_text": cell["prompt"],
            "generation": self.recipe["generation"], "reference_prompt_seconds": 0.0, "attempt": 1})
        weak.save_json(folder, "native.json", {"schema": "active-tts-native-return-v1",
            "array": {"bytes": len(npy(pcm)), "sha256": weak.sha(npy(pcm))},
            "wav": {"bytes": len(raw), "sha256": weak.sha(raw)}, "dtype": "<f4", "shape": [len(values)],
            "sample_rate_hz": rate, "samples_sha256": weak.sha(pcm), "all_finite": True, "technical_gate_applied": False})
        weak.save_json(folder, "returned.json", {"generation_seconds": 0.0, "total_seconds": 0.0,
            "native_stop_reason": "not_exposed_by_voice_clone_API"})
        return values, rate


class ContextTests(unittest.TestCase):
    def setUp(self):
        fixture = test_d2.D2Tests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture, self.root, self.sources = fixture, fixture.root, fixture.old_sources
        self.parent = fixture.plan
        self.parent_bytes = context.pool.canonical(self.parent)
        self.plan = context.plan(self.parent, context.digest(self.parent), self.sources,
                                 self.root, batch_id="context-fixture")

    def test_sixteen_exact_coverage_original_roles_and_no_parent_mutation(self):
        context.verify_plan(self.plan)
        self.assertEqual(context.pool.canonical(self.parent), self.parent_bytes)
        self.assertEqual(Counter(c["context_class"] for c in self.plan["cells"]), dict.fromkeys(context.CLASSES, 4))
        self.assertEqual(Counter(c["role"] for c in self.plan["cells"]), {"train-candidate": 12, "development-only": 4})
        for ref in self.plan["references"]:
            cells = [c for c in self.plan["cells"] if c["reference_id"] == ref["reference_id"]]
            self.assertEqual({c["intended_keyword_id"] for c in cells}, {1, 2})
            self.assertEqual(len({c["context_class"] for c in cells}), 2)
        old = {c["cell_id"] for c in self.parent["cells"] + self.parent["d1_parent"]["plan"]["cells"]}
        self.assertTrue(old.isdisjoint(c["cell_id"] for c in self.plan["cells"]))
        self.assertTrue(all(c["label_authority"] == "intended_only" and c["actual_text"] is None for c in self.plan["cells"]))
        self.assertEqual(context.summary(self.plan)["new_model_calls"], 0)
        self.assertIsNone(context.summary(self.plan)["observed_usable_cells"])

    def test_eight_alternative_is_explicitly_bounded_and_uses_all_identities(self):
        value = context.plan(self.parent, context.digest(self.parent), self.sources, self.root,
                             count=8, batch_id="context-small")
        self.assertEqual(len({c["reference_id"] for c in value["cells"]}), 8)
        self.assertEqual(Counter(c["context_class"] for c in value["cells"]), dict.fromkeys(context.CLASSES, 2))
        self.assertEqual(Counter(c["intended_keyword_id"] for c in value["cells"]), {1: 4, 2: 4})
        for invalid in (0, 7, 9, 17, True):
            with self.assertRaisesRegex(ValueError, "CONTEXT_BOUNDED_REQUEST_COUNT"):
                context.plan(self.parent, context.digest(self.parent), self.sources, self.root,
                             count=invalid, batch_id="invalid")

    def test_requested_positions_and_nonoverlapping_repetition_are_explicit(self):
        for cell in self.plan["cells"]:
            text = weak.normalize(cell["prompt"])
            keyword = context.KEYWORDS[cell["intended_keyword_id"] - 1]
            self.assertEqual(text.count(keyword), cell["requested_keyword_occurrences"])
            if cell["context_class"] == "prefix":
                self.assertTrue(text.startswith(keyword) and len(text) > len(keyword))
            elif cell["context_class"] == "suffix":
                self.assertTrue(text.endswith(keyword) and len(text) > len(keyword))
            elif cell["context_class"] == "middle":
                self.assertTrue(0 < text.index(keyword) < len(text) - len(keyword))
            else:
                self.assertEqual(text, keyword * 2)

    def test_changed_prompt_reference_role_seed_or_authority_is_rejected(self):
        for field, value in (("prompt", "你好小窝"), ("role", "independent-test"),
                             ("seed", 0), ("actual_text", "你好小窝"), ("label_authority", "gold")):
            forged = copy.deepcopy(self.plan); forged["cells"][0][field] = value
            with self.assertRaisesRegex(ValueError, "CONTEXT_PLAN_CHANGED"):
                context.verify_plan(forged)
        forged = copy.deepcopy(self.plan); forged["training_admitted"] = True
        with self.assertRaisesRegex(ValueError, "CONTEXT_PLAN_CHANGED"):
            context.verify_plan(forged)
        with self.assertRaisesRegex(ValueError, "CONTEXT_REQUIRES_FROZEN_EXISTING_EIGHT_D2"):
            context.plan(self.parent, "0" * 64, self.sources, self.root, batch_id="bad-parent")

    def test_frozen_bulk_and_d2_validators_reject_context_schema(self):
        with self.assertRaises(ValueError):
            context.bulk_runner.verify_plan(self.plan)
        with self.assertRaises(ValueError):
            context.d2_pool.verify_plan(self.plan)

    def test_bounded_backend_runs_two_chunks_with_native_and_no_retry(self):
        generation = self.root / "context-generation"
        first = [c["cell_id"] for c in self.plan["cells"][:8]]
        backend = FictionalBoundedBackend(self.plan, self.root / "job-1", first)
        result = context.run(self.plan, self.sources, self.root, generation, backend,
                             max_new_calls=8, writer=audio_fixture.fixture_writer)
        self.assertEqual((result["attempts_this_call"], result["total_started"], result["total_completed"]), (8, 8, 8))
        second = [c["cell_id"] for c in self.plan["cells"][8:]]
        backend2 = FictionalBoundedBackend(self.plan, self.root / "job-2", second)
        result = context.run(self.plan, self.sources, self.root, generation, backend2,
                             max_new_calls=8, writer=audio_fixture.fixture_writer)
        self.assertEqual((result["attempts_this_call"], result["total_started"], result["total_completed"]), (8, 16, 16))
        result = context.run(self.plan, self.sources, self.root, generation, backend2,
                             max_new_calls=8, writer=audio_fixture.fixture_writer)
        self.assertEqual(result["attempts_this_call"], 0)
        self.assertEqual(len(backend.calls) + len(backend2.calls), 16)
        for ident in first:
            self.assertEqual((generation / "cells" / ident / "native-f32.wav").read_bytes(),
                             (self.root / "job-1/native" / ident / "returned-native.wav").read_bytes())

    def test_failed_native_and_consumed_unknown_stay_consumed(self):
        generation = self.root / "context-generation"
        first, second = [c["cell_id"] for c in self.plan["cells"][:2]]
        backend = FictionalBoundedBackend(self.plan, self.root / "job-1", [first])
        context.run(self.plan, self.sources, self.root, generation, backend,
                    max_new_calls=1, writer=audio_fixture.fixture_writer)
        (generation / "cells" / first / "completed.json").unlink()
        backend2 = FictionalBoundedBackend(self.plan, self.root / "job-2", [second], fail=True)
        result = context.run(self.plan, self.sources, self.root, generation, backend2,
                             max_new_calls=1, writer=audio_fixture.fixture_writer)
        self.assertEqual((result["attempts_this_call"], result["total_started"], result["total_completed"]), (1, 2, 0))
        self.assertEqual(backend2.calls, [second])
        self.assertTrue((generation / "cells" / second / "failed.json").exists())
        self.assertTrue((self.root / "job-2/native" / second / "returned-native.wav").exists())
        self.assertFalse((generation / "cells" / first / "failed.json").exists())

    def test_unbounded_or_changed_native_recipe_cannot_call_backend(self):
        first = [c["cell_id"] for c in self.plan["cells"][:8]]
        backend = FictionalBoundedBackend(self.plan, self.root / "job", first)
        backend.recipe["generation"]["max_new_tokens"] = 999999
        with self.assertRaisesRegex(ValueError, "CONTEXT_BOUNDED_NATIVE_BACKEND_REQUIRED"):
            context.run(self.plan, self.sources, self.root, self.root / "generation", backend,
                        max_new_calls=8, writer=audio_fixture.fixture_writer)
        self.assertEqual(backend.calls, [])
        self.assertFalse((self.root / "generation").exists())


if __name__ == "__main__":
    unittest.main()
