"""Additive v2 source contracts; real planners, fixed vocabulary, fictional PCM."""
from collections import Counter
import copy
import hashlib
from pathlib import Path
import unittest

import context_supplement_v2 as context
import test_context_supplement as fixture


class FictionalBoundedBackend(fixture.FictionalBoundedBackend):
    def __init__(self, plan, output, selected, *, fail=False):
        self.recipe = context.runner_recipe(plan, selected)
        self.output, self.fail, self.calls = Path(output), fail, []


class ContextV2Tests(unittest.TestCase):
    def setUp(self):
        previous = fixture.ContextTests(); previous.setUp(); self.addCleanup(previous.doCleanups)
        self.previous, self.root, self.sources = previous, previous.root, previous.sources
        self.parent = previous.plan
        self.parent_bytes = context.pool.canonical(self.parent)
        self.vocab = context.FIXTURE.read_bytes()
        self.plan = context.plan(self.parent, context.digest(self.parent), self.sources,
                                 self.root, self.vocab, count=16, batch_id="context-v2-fixture")

    def test_fixed_vocabulary_fixture_and_preflight_do_not_create_observed_truth(self):
        self.assertEqual(hashlib.sha256(self.vocab).hexdigest(), context.VOCABULARY_FILE_SHA256)
        binding = context.vocabulary_binding(self.vocab)
        self.assertEqual(context.digest(binding["value"]), context.VOCABULARY_SHA256)
        self.assertEqual(binding, self.plan["vocabulary"])
        for cell in self.plan["cells"]:
            self.assertEqual(context.requested_text_preflight(cell["prompt"], binding),
                             cell["requested_text_preflight"])
            self.assertEqual(cell["requested_text_preflight"]["scope"], "requested_text_only")
            self.assertIsNone(cell["actual_text"])
            self.assertEqual(cell["label_authority"], "intended_only")
            self.assertNotIn("target_ids", cell)
            self.assertNotIn("token_ids", cell["requested_text_preflight"])
        self.assertFalse(self.plan["requested_text_preflight_is_observed_label"])
        self.assertEqual(context.summary(self.plan)["new_model_calls"], 0)
        self.assertIsNone(context.summary(self.plan)["observed_usable_cells"])

    def test_old_oov_templates_fail_without_altering_old_plan_or_vocabulary(self):
        binding = self.plan["vocabulary"]
        old = [row for row in self.parent["cells"] if row["context_class"] in ("prefix", "middle")]
        self.assertEqual(len(old), 8)
        for cell in old:
            with self.assertRaisesRegex(ValueError, "CONTEXT_V2_REQUESTED_FULLTEXT_OOV"):
                context.requested_text_preflight(cell["prompt"], binding)
        self.assertEqual(context.pool.canonical(self.parent), self.parent_bytes)
        self.assertEqual(context.FIXTURE.read_bytes(), self.vocab)

    def test_sixteen_exact_coverage_original_roles_and_explicit_new_revision(self):
        context.verify_plan(self.plan)
        self.assertEqual(Counter(c["context_class"] for c in self.plan["cells"]), dict.fromkeys(context.CLASSES, 4))
        self.assertEqual(Counter(c["intended_keyword_id"] for c in self.plan["cells"]), {1: 8, 2: 8})
        self.assertEqual(Counter(c["role"] for c in self.plan["cells"]), {"train-candidate": 12, "development-only": 4})
        for ref in self.plan["references"]:
            cells = [c for c in self.plan["cells"] if c["reference_id"] == ref["reference_id"]]
            self.assertEqual(len(cells), 2)
            self.assertEqual({c["intended_keyword_id"] for c in cells}, {1, 2})
            self.assertEqual(len({c["context_class"] for c in cells}), 2)
            self.assertEqual({c["role"] for c in cells}, {ref["role"]})
        repeated_text = [c for c in self.plan["cells"] if c["same_requested_text_as_parent"]]
        self.assertEqual(len(repeated_text), 8)
        self.assertEqual({c["context_class"] for c in repeated_text}, {"suffix", "repeated-call"})
        self.assertEqual({c["explicit_additive_revision"] for c in self.plan["cells"]}, {"fixed-vocabulary-v2"})
        for cell in self.plan["cells"]:
            text = context.pool.reference_text.normalize(cell["prompt"])
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

    def test_default_eight_corrects_prefix_middle_for_all_identities(self):
        value = context.plan(self.parent, context.digest(self.parent), self.sources,
                             self.root, self.vocab, batch_id="context-v2-small")
        self.assertEqual(len({c["reference_id"] for c in value["cells"]}), 8)
        self.assertEqual(Counter(c["context_class"] for c in value["cells"]), {"prefix": 4, "middle": 4})
        self.assertEqual(Counter(c["intended_keyword_id"] for c in value["cells"]), {1: 4, 2: 4})
        self.assertEqual(Counter(c["role"] for c in value["cells"]), {"train-candidate": 6, "development-only": 2})
        for invalid in (0, 7, 9, 17, True):
            with self.assertRaisesRegex(ValueError, "CONTEXT_V2_REQUEST_COUNT"):
                context.plan(self.parent, context.digest(self.parent), self.sources, self.root,
                             self.vocab, count=invalid, batch_id="invalid")

    def test_all_historical_ids_and_versions_stay_disjoint(self):
        d2 = self.parent["d2_parent"]["plan"]
        old = {c["cell_id"] for c in self.parent["cells"] + d2["cells"] + d2["d1_parent"]["plan"]["cells"]}
        self.assertTrue(old.isdisjoint(c["cell_id"] for c in self.plan["cells"]))
        self.assertEqual(self.plan["context_parent"]["plan"], self.parent)
        self.assertEqual(self.plan["d2_parent"], self.parent["d2_parent"])
        for name in ("d1_cells_generated_again", "d2_cells_generated_again", "context_v1_cells_generated_again"):
            self.assertEqual(self.plan[name], 0)
        for validator in (context.v1.verify_plan, context.v1.d2_pool.verify_plan,
                          context.v1.bulk_runner.verify_plan):
            with self.assertRaises(ValueError):
                validator(self.plan)

    def test_fixed_vocabulary_bytes_mapping_and_pins_cannot_be_changed(self):
        with self.assertRaisesRegex(ValueError, "CONTEXT_V2_FIXED_VOCABULARY_FILE_CHANGED"):
            context.vocabulary_binding(self.vocab + b" ")
        changed = copy.deepcopy(self.plan["vocabulary"]["value"])
        changed["token_to_id"]["厅"] = 0
        with self.assertRaisesRegex(ValueError, "CONTEXT_V2_FIXED_VOCABULARY_CHANGED"):
            context.vocabulary_binding(changed)
        for field in ("file_sha256", "canonical_sha256", "base_tokens_sha256"):
            changed = copy.deepcopy(self.plan); changed["vocabulary"][field] = "0" * 64
            with self.assertRaisesRegex(ValueError, "CONTEXT_V2_VOCABULARY_BINDING_CHANGED"):
                context.verify_plan(changed)

    def test_parent_pin_historical_binding_or_source_mutation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CONTEXT_V2_REQUIRES_FROZEN_CONTEXT16_PARENT"):
            context.plan(self.parent, "0" * 64, self.sources, self.root, self.vocab, batch_id="bad-parent")
        for mutate in (
            lambda p: p["context_parent"].update(plan_sha256="0" * 64),
            lambda p: p["d2_parent"].update(plan_sha256="0" * 64),
            lambda p: p["references"][0].update(role="development-only"),
        ):
            changed = copy.deepcopy(self.plan); mutate(changed)
            with self.assertRaises(ValueError):
                context.verify_plan(changed)
        ref = self.sources["references"][0]
        audio = self.root / ref["audio_path"]
        audio.write_bytes(audio.read_bytes() + b"changed")
        with self.assertRaises(ValueError):
            context.rebuild(self.plan, self.sources, self.root)

    def test_requested_text_coverage_truth_or_generation_mutation_is_rejected(self):
        for field, value in (("prompt", "你好小窝客厅"), ("role", "independent-test"),
                             ("seed", 0), ("actual_text", "你好小窝"), ("label_authority", "gold"),
                             ("explicit_additive_revision", "old-retry"),
                             ("requested_text_preflight", {"all_characters_representable": True})):
            changed = copy.deepcopy(self.plan); changed["cells"][0][field] = value
            with self.assertRaises(ValueError):
                context.verify_plan(changed)
        for field, value in (("training_admitted", True), ("requested_text_preflight_is_observed_label", True),
                             ("context_v1_cells_generated_again", 1)):
            changed = copy.deepcopy(self.plan); changed[field] = value
            with self.assertRaisesRegex(ValueError, "CONTEXT_V2_PLAN_CHANGED"):
                context.verify_plan(changed)
        changed = copy.deepcopy(self.plan); changed["generation"]["max_new_tokens"] = 999999
        with self.assertRaisesRegex(ValueError, "CONTEXT_V2_PLAN_CHANGED"):
            context.verify_plan(changed)

    def test_two_native_chunks_and_consumed_requests_are_never_replayed(self):
        generation = self.root / "context-v2-generation"
        all_calls = []
        for index in range(2):
            selected = [c["cell_id"] for c in self.plan["cells"][8*index:8*index+8]]
            backend = FictionalBoundedBackend(self.plan, self.root / ("job-" + str(index)), selected)
            result = context.run(self.plan, self.sources, self.root, generation, backend,
                                 max_new_calls=8, writer=fixture.audio_fixture.fixture_writer)
            self.assertEqual((result["attempts_this_call"], result["total_started"], result["total_completed"]),
                             (8, 8*(index+1), 8*(index+1)))
            all_calls.extend(backend.calls)
            for ident in selected:
                self.assertEqual((generation / "cells" / ident / "native-f32.wav").read_bytes(),
                                 (backend.output / "native" / ident / "returned-native.wav").read_bytes())
        self.assertEqual(len(set(all_calls)), 16)
        result = context.run(self.plan, self.sources, self.root, generation, backend,
                             max_new_calls=8, writer=fixture.audio_fixture.fixture_writer)
        self.assertEqual(result["attempts_this_call"], 0)

    def test_failed_native_and_consumed_unknown_are_retained_without_retry(self):
        generation = self.root / "context-v2-generation"
        first, second = [c["cell_id"] for c in self.plan["cells"][:2]]
        backend = FictionalBoundedBackend(self.plan, self.root / "job-1", [first])
        context.run(self.plan, self.sources, self.root, generation, backend,
                    max_new_calls=1, writer=fixture.audio_fixture.fixture_writer)
        (generation / "cells" / first / "completed.json").unlink()
        backend2 = FictionalBoundedBackend(self.plan, self.root / "job-2", [second], fail=True)
        result = context.run(self.plan, self.sources, self.root, generation, backend2,
                             max_new_calls=1, writer=fixture.audio_fixture.fixture_writer)
        self.assertEqual((result["attempts_this_call"], result["total_started"], result["total_completed"]), (1, 2, 0))
        self.assertEqual(backend2.calls, [second])
        self.assertTrue((generation / "cells" / second / "failed.json").exists())
        self.assertTrue((backend2.output / "native" / second / "returned-native.wav").exists())
        self.assertFalse((generation / "cells" / first / "failed.json").exists())

    def test_mutated_native_budget_refuses_before_any_call(self):
        selected = [c["cell_id"] for c in self.plan["cells"][:8]]
        backend = FictionalBoundedBackend(self.plan, self.root / "job", selected)
        backend.recipe["generation"]["max_new_tokens"] = 999999
        with self.assertRaisesRegex(ValueError, "CONTEXT_V2_BOUNDED_NATIVE_BACKEND_REQUIRED"):
            context.run(self.plan, self.sources, self.root, self.root / "generation", backend,
                        max_new_calls=8, writer=fixture.audio_fixture.fixture_writer)
        self.assertEqual(backend.calls, [])
        self.assertFalse((self.root / "generation").exists())


if __name__ == "__main__":
    unittest.main()
