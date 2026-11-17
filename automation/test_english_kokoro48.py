"""Meaningful source/probe boundaries; stdlib fixtures never call a model."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_kokoro48_recipe as r
import english_kokoro48_once as w


class Kokoro48Tests(unittest.TestCase):
    def plan(self):
        return json.loads(Path(__file__).with_name("english_kokoro48_plan.json").read_bytes())

    def result_rows(self):
        return [dict(row, reference_supported=False, reference_observed_hypothesis=None,
            reference_tier="unknown") for row in r.recipe()["rows"]]

    def mark(self, rows, mode, presets=3, *, eh=True, ordinary=True, vosen=True):
        for row in rows:
            if row["preset_id"] not in w.PRESETS[:presets]:
                continue
            selected = (row["mode"] == mode and ((row["requested_class"] == 2 and
                (eh or row["pronunciation_hypothesis"] == "mero-us-i-v1")) or
                (vosen and row["requested_class"] == 1))) or (ordinary and row["mode"] == "ordinary")
            if selected:
                row.update(reference_supported=True, reference_tier="silver",
                    reference_observed_hypothesis=row["pronunciation_hypothesis"] if row["requested_class"] else None)

    def test_frozen_48_cells_and_real_source_pins(self):
        frozen = w.validate_plan(self.plan())
        self.assertEqual(len(frozen["rows"]), 48)
        self.assertEqual({row["role"] for row in frozen["rows"]}, {"train"})
        self.assertEqual(len({row["id"] for row in frozen["rows"]}), 48)
        self.assertEqual({row["preset_id"] for row in frozen["rows"]}, set(w.PRESETS))
        self.assertNotIn("am_adam", w.PRESETS)
        self.assertEqual(sum(row["mode"] == "ordinary" for row in frozen["rows"]), 16)

    def test_budget_policy_and_training_flag_tampering_rejected(self):
        for key, value in (("maximum_tts_calls", 49), ("maximum_calls_per_observer", 49),
            ("maximum_total_model_calls", 289), ("retry_count", 1), ("new_kws_calls", 1),
            ("training_allowed", True), ("training_snapshot_allowed", True), ("diagnostic_only", False),
            ("generation_deadline_seconds", 1800), ("recipe_sha256", "0" * 64)):
            plan = self.plan(); plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.validate_plan(plan)
        plan = self.plan(); plan["reference_contract"]["commit"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "FROZEN_REFERENCE"):
            w.validate_plan(plan)
        plan = self.plan(); plan["probe_criteria"]["minimum_balanced_presets_per_mode"] = 2
        with self.assertRaisesRegex(ValueError, "FROZEN_REFERENCE"):
            w.validate_plan(plan)

    def test_all_three_source_modules_and_helpers_are_pinned(self):
        for key in ("worker_sha256", "recipe_code_sha256", "generator_code_sha256"):
            plan = self.plan(); plan[key] = "0" * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "SOURCE_PIN"):
                w.validate_plan(plan)
        plan = self.plan(); first = next(iter(plan["helper_sha256"])); plan["helper_sha256"][first] = "0" * 64
        with self.assertRaisesRegex(ValueError, "HELPER_SOURCE_PIN"):
            w.validate_plan(plan)

    def test_first_public_main_push_guard_and_no_retry(self):
        plan = self.plan(); plan.update(enabled=True, next_main_before="a" * 40)
        env = {"GITHUB_REPOSITORY": "jiying2007/kws-data", "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "b" * 40}
        event = {"repository": {"private": False, "full_name": "jiying2007/kws-data"}, "before": "a" * 40, "after": "b" * 40}
        w.execution_guard(plan, env, event)
        for change in ({"GITHUB_RUN_ATTEMPT": "2"}, {"GITHUB_EVENT_NAME": "workflow_dispatch"},
            {"GITHUB_REF": "refs/heads/branch"}, {"GITHUB_REPOSITORY": "other/fork"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                w.execution_guard(plan, env | change, event)
        with self.assertRaises(ValueError):
            w.execution_guard(plan, env, event | {"before": "c" * 40})
        with self.assertRaises(ValueError):
            w.execution_guard(plan, env, event | {"repository": {"private": True, "full_name": "jiying2007/kws-data"}})
        plan.update(enabled=False, next_main_before=None)
        with self.assertRaisesRegex(ValueError, "PROTECTED_FIRST_MAIN_ONLY"):
            w.execution_guard(plan, env, event)

    def test_partial_source_preserves_all_48_without_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan = self.plan()
            w.once(root / "RECIPE.json", r.recipe())
            w.once(root / "EXECUTION_PLAN.json", plan)
            w.once(root / "STARTED.json", {"schema": w.SCHEMA, "stage": "generate", "run": {"fixture": True},
                "maximum_calls": 48, "retry_count": 0, "training_snapshot_allowed": False})
            first = r.recipe()["rows"][0]
            folder = root / "source/audio" / first["id"]; folder.mkdir(parents=True)
            w.once(folder / "STARTED.json", {"id": first["id"], "request_sha256": first["request_sha256"], "attempt": 1,
                "preset_id": first["preset_id"], "mode": first["mode"], "role": "train",
                "phonemes": first["raw_phonemes"], "target_start_char": first["target_start_char"],
                "voice_pack_index": len(first["raw_phonemes"]) - 1, "seed": first["seed"],
                "synthesis_config": r.SYNTHESIS, "direct_api": "KModel.forward", "runtime_g2p_calls": 0,
                "retry_count": 0, "training_allowed": False, "diagnostic_only": True})
            corpus = w.source_readback(root, plan)
            self.assertEqual(len(corpus["rows"]), 48)
            self.assertEqual(corpus["valid_count"], 0)
            self.assertEqual(corpus["source_attempts"], {"started": 1, "completed": 0,
                "failed": 0, "interrupted_after_start": 1, "not_started": 47})
            self.assertEqual(corpus["rows"][0]["generation_status"], "UNKNOWN_AFTER_START_NO_RETRY")
            self.assertFalse(corpus["training_snapshot_allowed"])

    def test_source_metadata_and_scope_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); frozen = r.recipe(); frozen["rows"][0]["role"] = "locked-test"
            w.once(root / "RECIPE.json", frozen); w.once(root / "EXECUTION_PLAN.json", self.plan())
            with self.assertRaisesRegex(ValueError, "SOURCE_RECIPE_OR_PLAN"):
                w.source_readback(root, self.plan())

    def test_integer_geometry_matches_frozen_core_at_boundaries(self):
        path = Path(__file__).with_name("english_synthetic192_training_core.py")
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("need", "frame_ends", "positive_ends")]
        self.assertEqual(len(functions), 3)
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        for samples in (400, 6399, 6400, 12800, 16000, 24000, 28800, 31840, 32000, 40000, 160000):
            expected = namespace["positive_ends"](38400, samples, 38400 + samples + 16000)
            actual = w.geometry(samples)
            self.assertEqual(actual["positive_support_ends"], expected)
            self.assertEqual(actual["eligible"], len(expected) >= 3)
            self.assertEqual(actual["new_embedding_calls"], 0)

    def test_unknowns_and_lv60_alone_cannot_expand(self):
        rows = self.result_rows()
        for row in rows:
            row["lv60_whole_phrase_hypotheses"] = [row["pronunciation_hypothesis"]]
        result = w.continuation(rows)
        self.assertEqual(result["status"], "STOP_DIAGNOSTIC_NO_EXPANSION")
        self.assertFalse(result["training_snapshot_allowed"])
        self.assertFalse(result["diagnostic_lv60_used_for_decision"])

    def test_i_only_mero_does_not_satisfy_eh_gate(self):
        rows = self.result_rows(); self.mark(rows, "short", presets=4, eh=False)
        result = w.continuation(rows)
        self.assertEqual(result["modes"]["short"]["balanced_count"], 4)
        self.assertEqual(result["modes"]["short"]["mero_eh_count"], 0)
        self.assertIsNone(result["selected_mode"])

    def test_three_presets_and_ordinary_are_both_required(self):
        for count, ordinary, vosen in ((2, True, True), (4, False, True), (4, True, False)):
            rows = self.result_rows(); self.mark(rows, "short", count, ordinary=ordinary, vosen=vosen)
            with self.subTest(count=count, ordinary=ordinary, vosen=vosen):
                self.assertIsNone(w.continuation(rows)["selected_mode"])

    def test_eh_intent_with_i_observation_does_not_count_as_eh(self):
        rows = self.result_rows(); self.mark(rows, "short", presets=4)
        for row in rows:
            if row["pronunciation_hypothesis"] == "mero-us-eh-v1":
                row["reference_observed_hypothesis"] = "mero-us-i-v1"
        result = w.continuation(rows)
        self.assertEqual(result["modes"]["short"]["balanced_count"], 4)
        self.assertEqual(result["modes"]["short"]["mero_eh_count"], 0)
        self.assertIsNone(result["selected_mode"])

    def test_mode_selection_is_preregistered_and_never_training(self):
        rows = self.result_rows(); self.mark(rows, "carrier", presets=3)
        result = w.continuation(rows)
        self.assertEqual(result["selected_mode"], "carrier")
        self.mark(rows, "short", presets=3)
        result = w.continuation(rows)
        self.assertEqual(result["selected_mode"], "short")
        self.assertEqual(result["status"], "PROPOSE_SEPARATELY_PREREGISTERED_NEW_BATCH")
        self.assertFalse(result["training_snapshot_allowed"])
        self.assertFalse(result["automatic_expansion"])
        self.assertFalse(result["high_quality_model_passed"])

    def test_partial_observers_preserve_48_and_empty_evidence(self):
        corpus = {"rows": [dict(row, generation_status="NOT_STARTED", pcm_sha256=None,
            wav_sha256=None, pcm_samples=None) for row in r.recipe()["rows"]], "source_attempts": {"started": 0}}
        with tempfile.TemporaryDirectory() as temp:
            result = w.collect_evidence(Path(temp), corpus)
        self.assertEqual(len(result["rows"]), 48)
        self.assertEqual(result["total_actual_model_attempts"], 0)
        for count in result["observer_counts"].values():
            self.assertEqual(count["not_started"], 48)
        self.assertFalse(result["training_snapshot_allowed"])

    def test_capture_cannot_be_invented_without_started_and_lock(self):
        rows = [dict(row, generation_status="success", pcm_sha256="a" * 64,
            wav_sha256="b" * 64, pcm_samples=16000) for row in r.recipe()["rows"]]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "evidence/qwen" / rows[0]["id"]; folder.mkdir(parents=True)
            w.once(folder / "CAPTURE.json", {"raw_text": "Hey Mero."})
            with self.assertRaisesRegex(ValueError, "CAPTURE_STARTED_AND_LOADED"):
                w.collect_evidence(Path(temp), {"rows": rows, "source_attempts": {"started": 48}})

    def test_lv60_is_not_a_reference_vote_and_source_only_has_no_heavy_import(self):
        self.assertEqual(w.REFERENCE_PROVIDERS, ("qwen", "whisper", "parakeet", "phoneme"))
        self.assertNotIn("phoneme_lv60", w.PROBE_CRITERIA["reference_providers"])
        tree = ast.parse(Path(w.__file__).read_text())
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertNotIn("build_admission", calls)
        self.assertNotIn("validate_plan", calls)
        self.assertFalse({"torch", "numpy", "transformers", "kokoro"} & set(sys.modules))

    def test_create_once_and_safe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); w.once(root / "STARTED.json", {"attempt": 1})
            with self.assertRaisesRegex(ValueError, "CREATE_ONCE_FILE"):
                w.once(root / "STARTED.json", {"attempt": 2})
            for name in ("../outside", "/absolute", "a/../b", "a//b", "a\\b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    w.within(root, name)


if __name__ == "__main__":
    unittest.main()
