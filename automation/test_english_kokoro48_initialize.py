"""Pure-stdlib/mock boundary tests; no runtime install, model, network or audio."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('initialization_fixture', HERE/'english_kokoro48_initialize.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def environment(event_path):
    # Synthetic fixture identifiers, never passed to a remote operation.
    return {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': 'jiying2007/kws-data',
        'GITHUB_EVENT_NAME': 'pull_request', 'GITHUB_RUN_ATTEMPT': '1',
        'GITHUB_SHA': '1'*40, 'GITHUB_RUN_ID': '123', 'GITHUB_EVENT_PATH': str(event_path)}


def event():
    return {'fixture_only': True,
        'repository': {'full_name': 'jiying2007/kws-data', 'private': False},
        'pull_request': {'head': {'sha': '2'*40, 'repo': {'full_name': 'jiying2007/kws-data'}}}}


def construction_markers(output, *, returned=True):
    h.g.once(output/'source/MODEL_LOAD_STARTED.json', {'fixture_only': True, 'forward_attempts': 0})
    if returned:
        h.g.once(output/'source/MODEL_CONSTRUCTED.json', {'fixture_only': True,
            'constructor_and_runtime_transforms_returned': True, 'model_instances': 1,
            'model_forward_calls': 0})


def packet(output, *, passed=True):
    records = []
    failed = [] if passed else ['decoder']
    for component in h.COMPONENTS:
        relative = 'source/weight-diagnostics/' + component + '.json'
        row = {'fixture_only': True, 'schema': h.g.SCHEMA+'/checkpoint-component',
            'component': component, 'passed': component not in failed}
        digest = h.g.once(output/relative, row)
        records.append({'component': component, 'path': relative,
            'bytes': (output/relative).stat().st_size, 'sha256': digest, 'passed': row['passed']})
    summary = {'fixture_only': True, 'schema': h.g.SCHEMA+'/checkpoint-readback',
        'passed': passed, 'component_count': 5, 'passed_components': 5-len(failed),
        'failed_components': failed, 'component_files': [r['path'] for r in records],
        'component_records': records, 'allowed_key_exceptions': []}
    digest = h.g.once(output/'source/weight-diagnostics/SUMMARY.json', summary)
    if passed:
        h.g.once(output/'source/MODEL_LOCK.json', {'fixture_only': True,
            'all_component_weights_verified': True,
            'weight_diagnostics': 'source/weight-diagnostics/SUMMARY.json',
            'weight_diagnostics_sha256': digest})


def unused_forward(*args, **kwargs):
    raise AssertionError('FIXTURE_FORWARD_MUST_NEVER_BE_CALLED')


class PublicGuard(unittest.TestCase):
    def test_only_public_same_repository_first_pr_is_allowed(self):
        env = environment('/fixture/event.json')
        h.allowed_event(env, event())
        for key, value in (('GITHUB_ACTIONS', 'false'), ('GITHUB_EVENT_NAME', 'push'),
            ('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_REPOSITORY', 'fixture/fork'),
            ('GITHUB_SHA', 'not-a-commit'), ('GITHUB_RUN_ID', 'not-a-run')):
            changed = {**env, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.allowed_event(changed, event())
        for kind in ('private', 'fork'):
            changed = event()
            if kind == 'private':
                changed['repository']['private'] = True
            else:
                changed['pull_request']['head']['repo']['full_name'] = 'fixture/fork'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                h.allowed_event(env, changed)

    def test_source_has_one_loader_call_and_no_forward_or_rows_call(self):
        tree = ast.parse((HERE/'english_kokoro48_initialize.py').read_text())
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(x.name.split('.')[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module.split('.')[0])
        self.assertFalse(set(imports).intersection({'torch', 'kokoro', 'numpy', 'scipy', 'transformers'}))
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))]
        self.assertEqual(calls.count('_load_native'), 1)
        self.assertFalse(set(calls).intersection({'generate', '_run_rows', 'forward', 'unused_forward', 'KModel'}))


class InitializationBoundary(unittest.TestCase):
    def execute(self, loader):
        with tempfile.TemporaryDirectory(prefix='initialization-fixture-') as temporary:
            root = Path(temporary)
            event_path = root/'EVENT.json'
            event_path.write_text(json.dumps(event()))
            output = root/'result'
            with patch.dict(os.environ, environment(event_path)), patch.object(h.g, '_load_native', side_effect=loader) as mocked:
                result = h.run(output)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(result, json.loads((output/'RESULT.json').read_text()))
            self.assertEqual(list(root.glob('kokoro48-initialization-assets-*')), [])
            self.assertEqual(result['model_forward_calls'], 0)
            self.assertEqual(result['tts_request_attempts'], 0)
            self.assertFalse(result['training_allowed'])
            return copy.deepcopy(result)

    def test_success_drops_returned_forward_without_invoking_it(self):
        def loader(output, private):
            construction_markers(output)
            packet(output)
            return unused_forward, {'fixture': 1}
        result = self.execute(loader)
        self.assertEqual(result['status'], 'INITIALIZATION_ONLY_PASSED')
        self.assertEqual(result['model_construction_attempts'], 1)
        self.assertEqual(result['model_construction_and_runtime_transforms_returned'], 1)
        self.assertTrue(result['diagnostic_packet_complete'])
        self.assertTrue(result['model_verified'])
        self.assertEqual(result['audio_files'], [])

    def test_complete_failure_packet_is_not_a_verified_model(self):
        def loader(output, private):
            construction_markers(output)
            packet(output, passed=False)
            raise ValueError('CHECKPOINT_EXACT_KEYS')
        result = self.execute(loader)
        self.assertEqual(result['status'], 'INITIALIZATION_ONLY_FAILED')
        self.assertEqual(result['failure']['code'], 'CHECKPOINT_EXACT_KEYS')
        self.assertTrue(result['diagnostic_packet_complete'])
        self.assertFalse(result['model_verified'])
        self.assertFalse(result['strict_model_lock_created'])

    def test_constructor_failure_and_readback_failure_are_distinct(self):
        def loader(output, private):
            construction_markers(output, returned=False)
            raise RuntimeError('FIXTURE_CONSTRUCTOR_DID_NOT_RETURN')
        result = self.execute(loader)
        self.assertEqual(result['model_construction_attempts'], 1)
        self.assertEqual(result['model_construction_and_runtime_transforms_returned'], 0)
        self.assertFalse(result['diagnostic_packet_complete'])
        self.assertFalse(result['model_verified'])

    def test_failure_before_constructor_is_not_counted_as_construction(self):
        def loader(output, private):
            raise RuntimeError('FIXTURE_ASSET_FAILURE')
        result = self.execute(loader)
        self.assertEqual(result['model_construction_attempts'], 0)
        self.assertEqual(result['model_construction_and_runtime_transforms_returned'], 0)
        self.assertIsNone(result['checkpoint_readback_summary'])

    def test_broken_summary_preserves_original_failure_and_result(self):
        def loader(output, private):
            construction_markers(output)
            h.g.put(output/'source/weight-diagnostics/SUMMARY.json', b'{fixture incomplete')
            raise ValueError('FIXTURE_PRIMARY_READBACK_FAILURE')
        result = self.execute(loader)
        self.assertEqual(result['failure']['code'], 'FIXTURE_PRIMARY_READBACK_FAILURE')
        self.assertEqual(result['summary_read_failure']['type'], 'JSONDecodeError')
        self.assertFalse(result['diagnostic_packet_complete'])
        self.assertFalse(result['model_verified'])

    def test_sigterm_and_deadline_use_finally_without_retry(self):
        for number, code in ((signal.SIGTERM, 'INITIALIZATION_ONLY_SIGTERM'),
            (signal.SIGALRM, 'INITIALIZATION_ONLY_DEADLINE')):
            original = signal.getsignal(number)
            def loader(output, private):
                construction_markers(output, returned=False)
                signal.getsignal(number)(number, None)
            result = self.execute(loader)
            self.assertEqual(result['failure']['code'], code)
            self.assertIs(signal.getsignal(number), original)
            self.assertFalse(result['model_verified'])

    def test_audio_file_even_from_fake_loader_forbids_success(self):
        def loader(output, private):
            construction_markers(output)
            packet(output)
            h.g.put(output/'source/fixture.wav', b'NOT_REAL_AUDIO_FIXTURE')
            return unused_forward, {}
        result = self.execute(loader)
        self.assertEqual(result['failure']['code'], 'INITIALIZATION_MUST_NOT_CREATE_AUDIO')
        self.assertFalse(result['model_verified'])

    def test_packet_byte_mismatch_cannot_claim_complete_or_verified(self):
        def loader(output, private):
            construction_markers(output)
            packet(output)
            with (output/'source/weight-diagnostics/decoder.json').open('ab') as stream:
                stream.write(b' ')
            return unused_forward, {}
        result = self.execute(loader)
        self.assertEqual(result['failure']['code'], 'EXACT_COMPONENT_DIAGNOSTIC_BYTES')
        self.assertIsNotNone(result['summary_read_failure'])
        self.assertFalse(result['diagnostic_packet_complete'])
        self.assertFalse(result['model_verified'])

    def test_existing_output_blocks_loader_before_any_initialization(self):
        with tempfile.TemporaryDirectory(prefix='initialization-fixture-') as temporary:
            root = Path(temporary)
            event_path = root/'EVENT.json'
            event_path.write_text(json.dumps(event()))
            output = root/'result'
            output.mkdir()
            with patch.dict(os.environ, environment(event_path)), patch.object(h.g, '_load_native') as mocked:
                with self.assertRaisesRegex(ValueError, 'NEW_INITIALIZATION_RECEIPT_DIRECTORY'):
                    h.run(output)
                mocked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
