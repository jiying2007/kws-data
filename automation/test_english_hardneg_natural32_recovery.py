"""Recovery-only source/mock checks. No subprocess, decoder, compiler or model."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKER = HERE / 'english_hardneg_natural32_recovery_once.py'
if not WORKER.exists():
    WORKER = HERE / 'natural32_recovery_once.py'
spec = importlib.util.spec_from_file_location('natural32_recovery_under_test', WORKER)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
REPO_ROOT = HERE.parent if (HERE.parent / w.ORIGINAL_WORKER).exists() else HERE.parent / 'kws-data'


def failure_fixture():
    binding = {'repository': w.REPO, 'run_id': str(w.FAILURE['run_id']), 'head_sha': w.FAILURE['head_sha'],
               'main_before': w.ROSTER_SOURCE_BASE, 'run_attempt': 1, 'event': 'push', 'public_repository': True}
    result = {'passed': False, 'status': 'FAILED_NO_RETRY', 'reports': [],
              'candidate_head': w.HEAD, 'native_library': w.LIBRARY,
              'configuration_sha256': w.sha(w.canonical(w.CONFIG)), **binding,
              **dict.fromkeys(w.FALSE_FLAGS, False)}
    for key in ('consumer_build_attempts', 'decoder_process_attempts', 'model_process_attempts',
                'completed_validated_traces', 'new_onnx_calls', 'python_head_forwards', 'optimizer_updates',
                'new_tts_asr_phone_calls', 'new_mixes', 'retry_count'):
        result[key] = 0
    supervisor = {'passed': False, 'returncode': 1, 'timed_out': False, 'error': None,
                  'partial_receipt_errors': [], 'retry_count': 0, 'wall_seconds': 307.016853929,
                  'cleanup': {'descendants_found': False, 'terminated': False, 'killed': False}, **binding}
    start = {'binding': binding, 'worker_sha256': w.ORIGINAL_FILES[w.ORIGINAL_WORKER]['sha256'],
             'network_calls_at_start': 0, 'maximum_seconds': 1200, 'monotonic_started': 0,
             'monotonic_deadline': 1200, 'retry_count': 0}
    commands = {}
    names = {'current-run', 'corpus-download'} | {
        source + '-' + suffix for source in ('training', 'quality', 'sdk') for suffix in ('run', 'artifact', 'zip')}
    for name in names:
        argv = ['gh', 'api', 'synthetic-source']
        if name == 'corpus-download':
            argv = ['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https',
                    '--connect-timeout', '20', '--max-time', '300', '--retry', '0', '--silent', '--show-error', w.CORPUS['url']]
        common = {'argv': argv, 'kind': None, 'attempt': 1, 'retry_count': 0}
        begun = {**common, 'new_session': False, 'cpu_threads': 1, 'global_deadline': 1200, 'monotonic_started': 10}
        ended = {**common, 'timed_out': False, 'returncode': 28 if name == 'corpus-download' else 0,
                 'error': 'ValueError:COMMAND_NONZERO_EXIT' if name == 'corpus-download' else None}
        commands[name] = (begun, ended)
    return result, supervisor, start, commands


class RecoveryTests(unittest.TestCase):
    def test_original_four_files_remain_exact(self):
        for name, pin in w.ORIGINAL_FILES.items():
            self.assertEqual(w.identity((REPO_ROOT / name).read_bytes()), pin)

    def test_roster_base_configuration_and_compute_budget_unchanged(self):
        old = w.loads((REPO_ROOT / w.ORIGINAL_PLAN).read_bytes())
        plan = w.expected_plan(w.sha(WORKER.read_bytes()))
        self.assertEqual(w.CONFIG, old['configuration'])
        self.assertEqual(w.sha(w.canonical(w.CONFIG)), '13ff76c46b35d8d1c427bc237cfc3153771f281379ad86df8203f685aa2e7af4')
        self.assertEqual(w.ROSTER_SOURCE_BASE, old['main_before'])
        self.assertNotEqual(w.BEFORE, w.ROSTER_SOURCE_BASE)
        self.assertEqual(plan['acquisition_execution_attempt'], 2)
        self.assertEqual(plan['prior_compute_attempts'], 0)
        self.assertEqual(plan['maximum_model_processes'], plan['aggregate_maximum_model_processes'])
        self.assertEqual(plan['maximum_model_processes'], 32)
        self.assertEqual(plan['maximum_execution_seconds'], 1200)
        self.assertEqual(plan['retry_count'], 0)

    def test_decoder_trace_event_and_cleanup_functions_are_unchanged(self):
        old = ast.parse((REPO_ROOT / w.ORIGINAL_WORKER).read_text())
        new = ast.parse(WORKER.read_text())
        a = {x.name: ast.dump(x, include_attributes=False) for x in old.body if isinstance(x, ast.FunctionDef)}
        b = {x.name: ast.dump(x, include_attributes=False) for x in new.body if isinstance(x, ast.FunctionDef)}
        for name in ('flac_streaminfo', 'select_corpus', 'decode_argv', 'replay_events',
                     'validate_trace_metadata', 'cleanup_group', 'supervise', 'close_interrupted_receipts'):
            self.assertEqual(a[name], b[name], name)

    def test_prior_failure_has_exact_zero_compute(self):
        w.verify_failure_semantics(*failure_fixture())
        for field in ('consumer_build_attempts', 'decoder_process_attempts', 'model_process_attempts',
                      'completed_validated_traces', 'optimizer_updates'):
            for value in (1, None, False):
                args = failure_fixture()
                args[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'ZERO_COMPUTE'):
                    w.verify_failure_semantics(*args)

    def test_prior_supervisor_timeout_residue_or_uncertainty_rejected(self):
        for field, value in [('timed_out', True), ('returncode', -9), ('error', 'unknown'),
                             ('partial_receipt_errors', ['bad']), ('passed', True), ('wall_seconds', 1200)]:
            args = failure_fixture()
            args[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                w.verify_failure_semantics(*args)
        args = failure_fixture()
        args[1]['cleanup']['descendants_found'] = True
        with self.assertRaises(ValueError):
            w.verify_failure_semantics(*args)

    def test_any_prior_model_command_missing_exit_or_extra_command_rejected(self):
        args = failure_fixture()
        args[3]['corpus-download'][0]['kind'] = 'model'
        with self.assertRaisesRegex(ValueError, 'NO_COMPUTE_COMMAND'):
            w.verify_failure_semantics(*args)
        args = failure_fixture()
        args[3].pop('sdk-zip')
        with self.assertRaisesRegex(ValueError, 'EXACT_ACQUISITION_COMMAND_SET'):
            w.verify_failure_semantics(*args)
        args = failure_fixture()
        args[3]['scan-0'] = copy.deepcopy(args[3]['current-run'])
        with self.assertRaisesRegex(ValueError, 'EXACT_ACQUISITION_COMMAND_SET'):
            w.verify_failure_semantics(*args)

    def test_only_the_recorded_rc28_acquisition_failure_is_eligible(self):
        for field, value in [('returncode', 0), ('returncode', 1), ('timed_out', True), ('attempt', 2), ('retry_count', 1)]:
            args = failure_fixture()
            args[3]['corpus-download'][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                w.verify_failure_semantics(*args)
        args = failure_fixture()
        args[0]['head_sha'] = 'f' * 40
        with self.assertRaisesRegex(ValueError, 'RUN_BINDING'):
            w.verify_failure_semantics(*args)

    def test_inventory_rejects_every_other_missing_or_extra_file(self):
        omission = 'source/' + w.ORIGINAL_WORKFLOW
        files = {f'file-{i}': {'bytes': 0, 'sha256': '0' * 64} for i in range(84)}
        files[omission] = w.ORIGINAL_FILES[w.ORIGINAL_WORKFLOW]
        manifest = {'files': files, 'supervised_success': False, 'downloads_excluded': True,
                    'authoritative_completion_receipt': 'evidence/SUPERVISOR_EXIT.json'}
        base = set(files) - {omission} | {'MANIFEST.json'}
        for names in (base - {'file-0'}, base | {'unexpected'}, base | {omission}, base - {'file-0'} | {'other'}):
            fake = mock.Mock()
            fake.z.namelist.return_value = list(names)
            fake.read.return_value = w.canonical(manifest)
            with mock.patch.object(w, 'Archive', return_value=fake), \
                 mock.patch.object(w, 'sha', return_value=w.FAILURE_MANIFEST_SHA), \
                 self.assertRaisesRegex(ValueError, 'EXACT_SINGLE_HISTORICAL_UPLOAD_OMISSION'):
                w.verify_failure_artifact('no-file', REPO_ROOT)
            fake.close.assert_called_once()

    def test_recovery_acquisition_is_one_600_second_request(self):
        runner = w.Runner(Path('/synthetic-output'), 1000)
        captured = []
        def command(tag, argv, **kwargs):
            captured.append((tag, argv, kwargs))
            return Path('/synthetic-output/downloads/dev-clean.tar.gz')
        with mock.patch.object(runner, 'command', side_effect=command), \
             mock.patch.object(w, 'file_identity', return_value={k: w.CORPUS[k] for k in ('bytes', 'sha256')}), \
             mock.patch.object(w, 'once'):
            runner.download_corpus()
        self.assertEqual(len(captured), 1)
        _, argv, kwargs = captured[0]
        self.assertEqual(argv[argv.index('--max-time') + 1], '600')
        self.assertEqual(argv[argv.index('--retry') + 1], '0')
        self.assertEqual(kwargs['timeout'], 605)

    def test_failure_proof_precedes_every_new_acquisition_and_compute_stage(self):
        source = WORKER.read_text()
        child = source[source.index('def child('):source.index('def group_exists(')]
        proof = child.index("with runner.stage('verify-prior-acquisition-failure')")
        for stage in ('authenticate-artifacts', 'select-original-corpus', 'decode-originals', 'build-native-consumer', 'scan-natural32'):
            self.assertLess(proof, child.index("with runner.stage('" + stage + "')"))
        self.assertIn("runner.download('prior-failure', FAILURE, 'failure')", child)


if __name__ == '__main__':
    unittest.main()
