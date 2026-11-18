"""Offline protocol tests; models, numerical replay, and network never run."""
from pathlib import Path
import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
import zipfile
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load('english_mero_meh_fullpcm_once')
binder = load('bind_english_mero_meh_fullpcm')
m = adapter.driver()
h, inc = m.helpers()


def fixture():
    source = dict(m.PREVIOUS, run_id=123, artifact_id=456, head_sha='a' * 40,
        name='english-mero-meh-training-123')
    head = {'bytes': 197908, 'sha256': '1' * 64}
    npz = {'bytes': 199356, 'sha256': '2' * 64}
    plan = m.fixed_plan(h.sha((HERE / 'english_mero_meh_fullpcm_once.py').read_bytes()),
        source, 'b' * 40, head, npz)
    result = {'schema': adapter.TRAIN_SCHEMA, 'status': adapter.TRAIN_STATUS,
        'source_design_id': adapter.CAMPAIGN, 'locked_test_opened': True,
        'passed': True, 'research_retention_allowed': True, 'numerical_execution_completed': True,
        'run_id': 123, 'head_sha': 'a' * 40, 'run_attempt': 1, 'repository': m.REPO,
        'public_repository': True, 'strong_english_fsmn_ctc': False, 'ssc305_qualification': False,
        'candidate': {'bin': head, 'npz': npz}, 'gates': dict.fromkeys(m.GATES, True)}
    run = {'id': 123, 'head_sha': 'a' * 40, 'run_attempt': 1,
        'repository': {'full_name': m.REPO, 'private': False},
        'path': adapter.TRAIN_WORKFLOW, 'event': 'push', 'head_branch': 'main',
        'status': 'completed', 'conclusion': 'success'}
    return plan, result, run


class MeroFullPcmContracts(unittest.TestCase):
    def test_source_only_cli_accepts_disabled_or_bound_plan_without_subprocess(self):
        disabled = m.fixed_plan(h.sha((HERE / 'english_mero_meh_fullpcm_once.py').read_bytes()))
        enabled, _, _ = fixture()
        for plan in (disabled, enabled):
            with self.subTest(enabled=plan['enabled']), tempfile.TemporaryDirectory() as directory:
                plan_path = Path(directory) / 'fixture-plan.json'
                plan_path.write_bytes(h.canonical(plan))
                with mock.patch.object(m, 'PLAN', str(plan_path)), \
                     mock.patch.object(m.subprocess, 'Popen', side_effect=AssertionError('NO_PROCESS')), \
                     mock.patch.object(m.subprocess, 'check_output', side_effect=AssertionError('NO_PROCESS')), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(m.main([]), 0)
                report = h.loads(output.getvalue())
                self.assertIs(report['enabled'], plan['enabled'])
                self.assertEqual(report['model_calls'], 0)
                self.assertEqual(report['fixed_cases'], 14)
                self.assertEqual(report['maximum_fullpcm_processes'], 28)

    def test_original_numerical_and_supervision_code_is_used_directly(self):
        original = load('english_synthetic192_fullpcm_once')
        for name in ('selected_fresh', 'fresh_reference', 'select_references', 'stream_identity',
                     'execution_identity', 'worker', 'supervise', 'prepare_qemu'):
            self.assertEqual(getattr(m, name).__code__, getattr(original, name).__code__, name)
        self.assertIs(m.worker.__globals__, m.__dict__)
        self.assertEqual(m.CONSUMERS, original.CONSUMERS)
        self.assertEqual(m.GATES, original.GATES)
        self.assertEqual(h.LIMITS, {'embedding': {'atol': 0.001, 'rtol': 0.0002},
            'logits': {'atol': 0.003, 'rtol': 0.0002}, 'probabilities': {'atol': 0.0005, 'rtol': 0.0}})

    def test_changed_driver_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'automation').mkdir()
            (root / adapter.DRIVER).write_text('raise AssertionError("MUST_NOT_LOAD")\n')
            with mock.patch.object(adapter, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'UNCHANGED_FULLPCM_NUMERICAL_DRIVER'):
                    adapter.driver()

    def test_new_identity_acceptance_does_not_mutate_saved_training_document(self):
        plan, result, _ = fixture()
        before = copy.deepcopy(result)
        m.source_preflight(plan, h)
        m.candidate_gate(result, plan)
        self.assertEqual(result, before)
        for change in ({'schema': 'english-synthetic192-training-once-v1'},
                       {'status': 'RETAINED_RESEARCH_SYNTHETIC192_HEAD'},
                       {'source_design_id': 'other'}, {'locked_test_opened': False},
                       {'passed': False}, {'strong_english_fsmn_ctc': True},
                       {'ssc305_qualification': True}):
            with self.assertRaises(ValueError):
                m.candidate_gate(dict(result, **change), plan)
        for key in result['gates']:
            bad = copy.deepcopy(result)
            bad['gates'][key] = False
            with self.assertRaises(ValueError):
                m.candidate_gate(bad, plan)

    def test_old_artifact_extra_compute_and_tuning_are_not_admitted(self):
        plan, _, _ = fixture()
        for change in ({'threshold': 0.79}, {'confirm_embeddings': 2}, {'cooldown_embeddings': 12},
                       {'maximum_fullpcm_processes': 29}, {'new_compilations': 1},
                       {'new_checkpoint_selections': 1}, {'threshold_tuning_runs': 1},
                       {'human_native_pronunciation_verified': True},
                       {'locked_test_replay_use': 'CHOOSE_ANOTHER_CHECKPOINT'}):
            with self.assertRaises(ValueError):
                m.source_preflight(dict(plan, **change), h)
        bad = copy.deepcopy(plan)
        bad['candidate_source']['name'] = 'english-synthetic192-training-123'
        with self.assertRaisesRegex(ValueError, 'REAL_MERO_MEH_CANDIDATE_ARTIFACT_PIN'):
            m.source_preflight(bad, h)

    def test_run_must_be_the_actual_first_public_new_training_workflow(self):
        _, _, run = fixture()
        adapter.validate_training_run(run, m.need)
        for change in ({'path': '.github/workflows/english-synthetic192-train-once.yml'},
                       {'event': 'workflow_dispatch'}, {'head_branch': 'research'},
                       {'run_attempt': 2}, {'conclusion': 'failure'},
                       {'repository': {'full_name': m.REPO, 'private': True}}):
            with self.assertRaises(ValueError):
                adapter.validate_training_run(dict(run, **change), m.need)

    def test_locked_choice_uses_id_class_condition_not_scores_or_input_order(self):
        rows = []
        for label in (0, 1, 2):
            for condition in m.CONDITIONS:
                common = dict(role='locked-test', weak_class=label, condition=condition,
                    usable=True, candidate_reference_available=True)
                rows.extend([dict(common, id=f'z-{label}-{condition}', events=[[label, 90000]]),
                    dict(common, id=f'a-{label}-{condition}', events=[])])
        document = {'schema': 'english-synthetic192-training-once-v1/fresh-cases',
            'rows': rows, 'references_bound_to_selected_weights': True, 'locked_test_opened': True}
        selected = m.selected_fresh(document)
        self.assertEqual(len(selected), 9)
        self.assertTrue(all(row['id'].startswith('a-') for row in selected))
        for row in rows:
            row['events'] = [[2, 35200]]
        reversed_document = dict(document, rows=list(reversed(rows)))
        self.assertEqual([r['id'] for r in selected], [r['id'] for r in m.selected_fresh(reversed_document)])
        with self.assertRaises(ValueError):
            m.selected_fresh(dict(document, locked_test_opened=False))

    def test_receipt_adaptation_is_one_initial_write_without_changed_pass_or_numbers(self):
        original = mock.Mock()
        writer = adapter.ReceiptProtocol(original)
        record = {'schema': adapter.SCHEMA + '/result',
            'status': 'VERIFIED_SYNTHETIC192_HEAD_FULLPCM_NATIVE_ARM', 'passed': True,
            'completed_comparisons': 28, 'comparisons': [{'numeric': [0.001, 0.002]}]}
        before = copy.deepcopy(record)
        writer.once(Path('/unused/evidence/RESULT.json'), record)
        self.assertEqual(record, before)
        self.assertEqual(original.once.call_count, 1)
        emitted = original.once.call_args.args[1]
        self.assertEqual(emitted['status'], 'VERIFIED_MERO_MEH_HEAD_FULLPCM_NATIVE_ARM')
        self.assertEqual(emitted['comparisons'], record['comparisons'])
        self.assertIs(emitted['passed'], True)
        self.assertEqual(emitted['locked_test_replay_use'], adapter.LOCKED_USE)
        self.assertEqual(emitted['new_checkpoint_selections'], 0)
        self.assertEqual(emitted['threshold_tuning_runs'], 0)
        self.assertIs(emitted['recommended_pronunciation_contract_frozen'], True)
        self.assertIs(emitted['human_native_pronunciation_verified'], False)

    def test_binder_checks_real_zip_manifest_then_calls_saved_byte_preflight(self):
        plan, result, run = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / 'candidate.zip'
            result_raw = h.canonical(result)
            manifest = {'files': {'evidence/RESULT.json': h.identity(result_raw)}}
            with zipfile.ZipFile(candidate, 'w') as archive:
                archive.writestr('evidence/RESULT.json', result_raw)
                archive.writestr('MANIFEST.json', h.canonical(manifest))
            pin = m.stream_identity(candidate)
            artifact = {'id': 456, 'name': 'english-mero-meh-training-123', 'expired': False,
                'size_in_bytes': pin['bytes'], 'digest': 'sha256:' + pin['sha256'],
                'workflow_run': {'id': 123, 'head_sha': 'a' * 40}}
            (root / 'run.json').write_bytes(h.canonical(run))
            (root / 'artifact.json').write_bytes(h.canonical(artifact))
            prepared = {'previous': mock.Mock(), 'candidate': mock.Mock(),
                'summary': {'model_calls': 0, 'fixture': True}}
            argv = ['--previous', str(root / 'previous.zip'), '--candidate', str(candidate),
                '--run-metadata', str(root / 'run.json'), '--artifact-metadata', str(root / 'artifact.json'),
                '--main-before', 'b' * 40, '--output-plan', str(root / 'bound-plan.json')]
            with mock.patch.object(binder, 'load_adapter', return_value=adapter), \
                 mock.patch.object(adapter, 'driver', return_value=m), \
                 mock.patch.object(m, 'prepare_inputs', return_value=prepared) as preflight, \
                 mock.patch.object(m.subprocess, 'Popen', side_effect=AssertionError('NO_PROCESS')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(binder.main(argv), 0)
            preflight.assert_called_once()
            bound = h.loads((root / 'bound-plan.json').read_bytes())
            m.source_preflight(bound, h)
            self.assertTrue(bound['enabled'])
            self.assertEqual(bound['candidate_source']['sha256'], pin['sha256'])
            self.assertEqual(bound['candidate_head'], result['candidate']['bin'])
            self.assertEqual(bound['new_checkpoint_selections'], 0)
            prepared['previous'].z.close.assert_called_once()
            prepared['candidate'].z.close.assert_called_once()

    def test_execution_still_fails_outside_public_actions_before_any_command(self):
        plan, _, _ = fixture()
        with mock.patch.dict(m.os.environ, {}, clear=True), \
             mock.patch.object(m.subprocess, 'check_output', side_effect=AssertionError('NO_PROCESS')):
            with self.assertRaisesRegex(ValueError, 'NEW_PUBLIC_MAIN_FIRST_PUSH_ONLY'):
                m.execution_identity(plan, h)


if __name__ == '__main__':
    unittest.main()
