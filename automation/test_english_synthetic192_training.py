"""Source-only boundary fixtures; no numeric runtime, model, process, or network."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


c = module('english_synthetic192_training_core.py', 'fixture_synthetic192_core')
w = module('english_synthetic192_train_once.py', 'fixture_synthetic192_worker')


class Contracts(unittest.TestCase):
    def test_actual_source_closure_and_inert_import(self):
        plan = w.loads(w.regular(w.ROOT/w.PLAN_PATH))
        w.source_preflight(plan)
        self.assertFalse(any(name in sys.modules for name in ('torch', 'numpy', 'onnx', 'onnxruntime')))

    def test_common_libri_identity_blocks_collision(self):
        for old in (3615, '003615', 'librispeech:3615', 'libri-speaker-3615', 'librispeech-dev-clean-speaker-3615'):
            for fresh in ('p3615', 'libritts-speaker-3615', 'piper-libritts-3615'):
                with self.assertRaisesRegex(ValueError, 'CROSS_LIBRITTS'):
                    c.require_libri_disjoint([old], [fresh])
        c.require_libri_disjoint(['2078'], ['p3615'])
        with self.assertRaisesRegex(ValueError, 'KNOWN_LIBRI'):
            c.libri_identity('unverified-3615')

    def test_context_is_retained_pcm_and_role_deterministic(self):
        raw = struct.pack('<6400h', *([1500, -1500]*3200))
        row = {'id': 'fixture', 'role': 'train', 'source_group_id': 'p3615', 'pcm_samples': 6400}
        pcm, context = c.contextualize(raw, row, 'clean')
        self.assertEqual((pcm, context), c.contextualize(raw, row, 'clean'))
        self.assertEqual(len(pcm), 2*(38400+6400+16000))
        self.assertEqual(pcm[2*38400:2*(38400+6400)], raw)
        self.assertEqual(context['eof_padding_samples'], 0)
        self.assertEqual(context['word_end_latency'], 'NOT_MEASURED')
        self.assertNotEqual(pcm, c.contextualize(raw, dict(row, role='locked-test'), 'clean')[0])

    def test_whole_clip_geometry_masks_transition(self):
        ends = c.positive_ends(38400, 16000, 70400)
        self.assertGreaterEqual(len(ends), 3)
        self.assertTrue(all(end-128 >= 54400 and end-31840 <= 38400 for end in ends))
        self.assertEqual(c.positive_ends(38400, 40000, 94400), [])

    def test_class_loss_masks_entire_source_without_resplit(self):
        rows = [{'role': role, 'source_group_id': role+str(i), 'weak_class': label,
                 'tier': 'silver', 'geometry_eligible': True}
                for role, count in c.RECIPE['minimum_balanced_groups'].items()
                for i in range(count) for label in (0, 1, 2)]
        self.assertTrue(c.postgeometry_admission(rows)['passed'])
        rows[2]['geometry_eligible'] = False
        result = c.postgeometry_admission(rows)
        self.assertFalse(result['passed'])
        self.assertEqual(result['balanced_source_groups']['train'], 5)
        self.assertTrue(all(not row['usable'] for row in rows[:3]))
        self.assertEqual({row['role'] for row in rows}, set(c.ROLES))

    def test_augmentation_count_does_not_buy_source_weight(self):
        bags = [{'weak_class': label, 'source_group_id': str(group), 'tier_weight': 1.0}
                for label in (0, 1, 2) for group in (0, 1)]
        bags += [dict(bags[0]) for _ in range(5)]
        weights = c.bag_weights(bags)
        first = sum(weight for bag, weight in zip(bags, weights) if bag['weak_class'] == 0 and bag['source_group_id'] == '0')
        self.assertAlmostEqual(first, weights[1])
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_early_wake_is_error_not_full_phrase_gain(self):
        row = {'id': 'fresh', 'parent_id': 'source', 'role': 'development', 'source_group_id': 'p3615',
            'weak_class': 1, 'tier': 'weak', 'condition': 'clean', 'pronunciation_hypothesis': 'mero-e-v1',
            'matched_pronunciation_hypothesis': 'mero-e-v1', 'clip_start_sample': 38400,
            'original_clip_eof_sample': 54400, 'samples': 70400}
        early = c.fresh_summary(row, [{'sample': 54527, 'keyword': 1}])
        exact = c.fresh_summary(row, [{'sample': 54528, 'keyword': 1}])
        late = c.fresh_summary(row, [{'sample': 70401, 'keyword': 1}])
        self.assertFalse(early['matched_complete_synthetic_phrase'])
        self.assertTrue(early['research_error'])
        self.assertTrue(exact['matched_complete_synthetic_phrase'])
        self.assertFalse(late['matched_complete_synthetic_phrase'])
        self.assertEqual(exact['matched_pronunciation_hypothesis'], 'mero-e-v1')
        self.assertEqual(exact['word_end_latency'], 'NOT_MEASURED')

    def test_plateau_and_budget_are_distinct(self):
        self.assertIsNone(c.should_stop(15, 1))
        self.assertEqual(c.should_stop(16, 1), 'DEVELOPMENT_EVENT_PLATEAU')
        self.assertEqual(c.should_stop(64, 63), 'BUDGET_EXHAUSTED_NOT_CONVERGENCE')

    def test_workflow_hidden_sources_and_exact_enabled_guard(self):
        raw = w.regular(w.ROOT/w.WORKFLOW_PATH)
        plan = w.loads(w.regular(w.ROOT/w.PLAN_PATH))
        w.workflow_contract(raw, plan)
        for missing in (b'include-hidden-files: true', b'NUMEXPR_NUM_THREADS: "1"', b'python-version: "3.12"'):
            with self.assertRaises(ValueError): w.workflow_contract(raw.replace(missing, b''), plan)
        enabled = dict(plan, enabled=True, main_before='a'*40)
        with self.assertRaisesRegex(ValueError, 'WORKFLOW_EXACT_ENABLED_BINDING'):
            w.workflow_contract(raw, enabled)

    def test_first_public_main_successor_guard(self):
        plan = {'enabled': True, 'main_before': 'a'*40}
        env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': w.REPO, 'GITHUB_EVENT_NAME': 'push',
            'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_ID': '123', 'GITHUB_SHA': 'b'*40}
        event = {'before': 'a'*40, 'after': 'b'*40, 'repository': {'private': False, 'full_name': w.REPO}}
        w.execution_identity(plan, env, event, 'b'*40, 'a'*40)
        with self.assertRaises(ValueError): w.execution_identity(plan, dict(env, GITHUB_RUN_ATTEMPT='2'), event, 'b'*40, 'a'*40)
        with self.assertRaises(ValueError): w.execution_identity(plan, env, event, 'b'*40, 'c'*40)
        with self.assertRaises(ValueError): w.execution_identity(plan, env, dict(event, repository={'private': True, 'full_name': w.REPO}), 'b'*40, 'a'*40)

    def test_partial_source_requires_actual_successful_combine(self):
        plan = {'corpus_source': {'run_id': 123, 'head_sha': 'a'*40}, 'corpus_combine_job': {'job_id': 456, 'name': 'combine'}}
        job = {'id': 456, 'name': 'combine', 'run_id': 123, 'head_sha': 'a'*40, 'status': 'completed', 'conclusion': 'success'}
        self.assertEqual(w.verify_combine_job(job, plan)['job_id'], 456)
        for altered in ({'conclusion': 'failure'}, {'status': 'in_progress'}, {'run_id': 124}, {'head_sha': 'b'*40}):
            with self.assertRaises(ValueError): w.verify_combine_job(dict(job, **altered), plan)

    def test_partial_numeric_receipt_has_actual_not_planned_counts(self):
        def failed_core(output, helper, progress, save_progress):
            progress['new_embedding_calls_started'] = 2
            progress['new_embedding_calls_completed'] = 1
            save_progress()
            raise ValueError('FIXTURE_INTERRUPTED_BEFORE_TRAINING')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output/'evidence').mkdir()
            binding = {'fixture': True}
            w.once(output/'evidence/WORKER_STARTED.json', {'pid': os.getppid(), 'binding': binding})
            with patch.object(w, 'execution_identity', return_value=binding), patch.object(w.os, 'getpgrp', return_value=os.getppid()), patch.dict(os.environ, {'KWS_SYN192_NUMERIC_PARENT_PID': str(os.getppid())}), patch.object(w, 'load_module', return_value=types.SimpleNamespace(train_and_evaluate=failed_core)):
                self.assertEqual(w.numeric(types.SimpleNamespace(output=output), {}), 1)
            result = json.loads((output/'evidence/NUMERIC_RESULT.json').read_bytes())
            self.assertFalse(result['passed'])
            self.assertEqual(result['actual_progress']['new_embedding_calls_started'], 2)
            self.assertEqual(result['actual_progress']['new_embedding_calls_completed'], 1)
            self.assertEqual(result['actual_progress']['optimizer_updates_completed'], 0)

    def test_reviewed_fit_is_part_of_pretest_gate(self):
        tree = ast.parse((HERE/'english_synthetic192_training_core.py').read_text())
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'gates' for target in node.targets)]
        mapping = {key.value: ast.unparse(value) for key, value in zip(assignments[0].value.keys, assignments[0].value.values)}
        self.assertEqual(mapping['source_balanced_reviewed_ce_decreased'], "fit['passed']")
        self.assertEqual(mapping['legacy_all_291_case_regression'], "paired['passed']")

    def test_native_clock_and_numeric_tolerance_fail_closed(self):
        ref = {'sample': 35200, 'ready': True, 'keyword': 1, 'logits': [0.0, 2.0, 0.0],
            'probabilities': [0.1, 0.8, 0.1], 'case_index': 0, 'condition': 'fresh', 'id': 'fixture', 'local_sample': 35200}
        self.assertTrue(c.verify_native(b'35200 1 1 0 2 0 0.1 0.8 0.1\n', [ref])['passed'])
        with self.assertRaises(ValueError): c.verify_native(b'35201 1 1 0 2 0 0.1 0.8 0.1\n', [ref])
        with self.assertRaises(ValueError): c.verify_native(b'35200 1 1 0 2.001 0 0.1 0.8 0.1\n', [ref])


if __name__ == '__main__':
    unittest.main()
