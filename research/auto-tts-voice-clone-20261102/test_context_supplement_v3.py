"""New24 full-text contracts with fictional PCM; no model or network."""
from collections import Counter
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import context_supplement_v3 as context
import test_context_supplement as fixture


class FictionalBoundedBackend(fixture.FictionalBoundedBackend):
    def __init__(self, plan, output, selected, *, fail=False):
        self.recipe = context.runner_recipe(plan, selected)
        self.output, self.fail, self.calls = Path(output), fail, []


class ContextV3Tests(unittest.TestCase):
    def setUp(self):
        previous = fixture.ContextTests(); previous.setUp(); self.addCleanup(previous.doCleanups)
        self.previous, self.root = previous, previous.root
        self.parent_context = context.v2.plan(previous.plan, context.digest(previous.plan),
            previous.sources, self.root, context.FIXTURE.read_bytes(), count=16, batch_id="context-v2-fixture")
        self.sources = previous.fixture.new_sources()
        self.parent_d2 = previous.fixture.new_plan(self.sources)
        self.parents_before = context.pool.canonical([self.parent_d2, self.parent_context])
        self.plan = context.plan(self.parent_d2, context.digest(self.parent_d2),
            self.parent_context, context.digest(self.parent_context), self.sources, self.root,
            context.FIXTURE.read_bytes(), batch_id="context-v3-fixture")

    def test_exact72_mixed_first_chunk_and_unchanged_family_splits(self):
        context.verify_plan(self.plan)
        self.assertEqual(len(self.plan["references"]), 24)
        self.assertEqual(Counter(c["intended_keyword_id"] for c in self.plan["cells"]), {1:24, 2:24, None:24})
        self.assertEqual(Counter(c["role"] for c in self.plan["cells"]), {"train-candidate":54,"development-only":18})
        self.assertEqual(Counter(c["context_class"] for c in self.plan["cells"]),
                         {**dict.fromkeys(context.CLASSES,12), "ordinary-speech":24})
        self.assertEqual(Counter(c["intended_keyword_id"] for c in self.plan["cells"][:8]), {1:3,2:3,None:2})
        for ref in self.plan["references"]:
            rows = [c for c in self.plan["cells"] if c["reference_id"] == ref["reference_id"]]
            self.assertEqual({c["intended_keyword_id"] for c in rows}, {1,2,None})
            self.assertEqual(len(rows),3)
            for c in rows:
                for field in ("role","speaker_group_id","source_family_id"):
                    self.assertEqual(c[field],ref[field])
        self.assertEqual(context.pool.canonical([self.parent_d2,self.parent_context]), self.parents_before)

    def test_all_historical_prompt_hashes_and_ids_are_excluded(self):
        history = context.historical_plans(self.parent_d2,self.parent_context)
        self.assertEqual(set(history), {"d1","d2-existing-eight","d2-new-twenty-four","context-v1","context-v2"})
        old = [c for p in history.values() for c in p["cells"]]
        ids = {c["cell_id"] for c in old}; prompts = {context.prompt_sha(c["prompt"]) for c in old}
        self.assertEqual(len(ids),2080)
        self.assertTrue(ids.isdisjoint(c["cell_id"] for c in self.plan["cells"]))
        self.assertTrue(prompts.isdisjoint(c["normalized_prompt_sha256"] for c in self.plan["cells"]))
        self.assertEqual(self.plan["history_exclusion"]["normalized_prompt_sha256"],sorted(prompts))
        self.assertEqual(len({c["request_payload_sha256"] for c in self.plan["cells"]}),72)
        original = context.templates()
        for parent in history.values():
            changed = copy.deepcopy(original)
            changed[0]["prompt"] = "，" + parent["cells"][0]["prompt"] + "。"
            with self.subTest(parent=parent["schema"]), patch.object(context,"templates",return_value=changed):
                with self.assertRaises(ValueError):
                    context.verify_plan(self.plan)

    def test_fixed_vocabulary_preflight_remains_requested_intent_only(self):
        for c in self.plan["cells"]:
            self.assertEqual(c["requested_text_preflight"],
                             context.requested_text_preflight(c["prompt"],self.plan["vocabulary"]))
            self.assertEqual(c["normalized_prompt_sha256"],context.prompt_sha(c["prompt"]))
            self.assertIsNone(c["actual_text"])
            self.assertEqual(c["label_authority"],"intended_only")
            self.assertFalse(c["historical_prompt_reuse"])
            self.assertNotIn("target_ids",c)
            text = context.pool.reference_text.normalize(c["prompt"])
            if c["intended_keyword_id"] is None:
                self.assertNotIn("小窝",text)
                self.assertEqual(c["requested_keyword_occurrences"],0)
                self.assertEqual(c["generation_category"],"ordinary-negative")
            else:
                keyword = context.KEYWORDS[c["intended_keyword_id"]-1]
                self.assertEqual(text.count(keyword),c["requested_keyword_occurrences"])
        self.assertFalse(self.plan["requested_text_preflight_is_observed_label"])
        self.assertIsNone(context.summary(self.plan)["observed_usable_cells"])
        self.assertEqual(context.summary(self.plan)["new_model_calls"],0)

    def test_request_parent_generation_or_split_mutation_fails_before_claim(self):
        mutations = [lambda p:p.update(training_admitted=True),
                     lambda p:p["generation"].update(max_new_tokens=999999),
                     lambda p:p["cells"][0].update(role="independent-test"),
                     lambda p:p["cells"][0].update(request_payload_sha256="0"*64),
                     lambda p:p["d2_parent"].update(plan_sha256="0"*64),
                     lambda p:p["context_parent"].update(plan_sha256="0"*64),
                     lambda p:p["history_exclusion"].update(normalized_prompt_sha256=[]),
                     lambda p:p["budgets"].update(max_total_calls=73)]
        for change in mutations:
            forged=copy.deepcopy(self.plan);change(forged)
            with self.assertRaises(ValueError):context.verify_plan(forged)
        with self.assertRaises(ValueError):
            context.plan(self.previous.parent,context.digest(self.previous.parent),
                self.parent_context,context.digest(self.parent_context),self.previous.sources,self.root,
                context.FIXTURE.read_bytes(),batch_id="not-new24")

    def test_native_fictional_chunk_preserves_complete_and_failed_return(self):
        generation=self.root/"context-v3-generation"
        selected=[c["cell_id"] for c in self.plan["cells"][:8]]
        backend=FictionalBoundedBackend(self.plan,self.root/"job-1",selected)
        result=context.run(self.plan,self.sources,self.root,generation,backend,max_new_calls=8,
                           writer=fixture.audio_fixture.fixture_writer)
        self.assertEqual((result["attempts_this_call"],result["total_started"],result["total_completed"]),(8,8,8))
        for ident in selected:
            self.assertEqual((generation/"cells"/ident/"native-f32.wav").read_bytes(),
                             (backend.output/"native"/ident/"returned-native.wav").read_bytes())
        # An unresolved old claim remains consumed. The independent owner must
        # stop on it; this low-level protocol never synthesizes it again.
        (generation/"cells"/selected[0]/"completed.json").unlink()
        ident=self.plan["cells"][8]["cell_id"]
        failed=FictionalBoundedBackend(self.plan,self.root/"job-2",[ident],fail=True)
        context.run(self.plan,self.sources,self.root,generation,failed,max_new_calls=1,
                    writer=fixture.audio_fixture.fixture_writer)
        self.assertEqual(failed.calls,[ident])
        self.assertTrue((generation/"cells"/ident/"failed.json").is_file())
        self.assertTrue((failed.output/"native"/ident/"returned-native.wav").is_file())

    def test_nine_call_or_unbound_generation_recipe_refuses_without_output(self):
        selected=[c["cell_id"] for c in self.plan["cells"][:8]]
        for count in (9,True):
            backend=FictionalBoundedBackend(self.plan,self.root/"job",selected)
            with self.assertRaisesRegex(ValueError,"CONTEXT_V3_RUN_INVOCATION_BUDGET"):
                context.run(self.plan,self.sources,self.root,self.root/"gen",backend,max_new_calls=count)
        backend=FictionalBoundedBackend(self.plan,self.root/"job",selected)
        backend.recipe["generation"]["temperature"]=1.0
        with self.assertRaisesRegex(ValueError,"CONTEXT_V3_BOUNDED_NATIVE_BACKEND_REQUIRED"):
            context.run(self.plan,self.sources,self.root,self.root/"gen",backend,max_new_calls=8)
        self.assertEqual(backend.calls,[])
        self.assertFalse((self.root/"gen").exists())


if __name__ == "__main__":
    unittest.main()
