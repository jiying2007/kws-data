"""Invented byte/report fixtures only; no models, fitting, or network calls.

The new worker owns source/observer ZIP and selection tests. These tests stub
that pure validator explicitly, then independently exercise downstream saved
weights, three identities, pin/cache bindings and the delivered infer contract.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package = load('_terminal_package_test', HERE / 'terminal144_package.py')
fixtures = load('_previous_package_test_fixtures', package.AUTOMATION / 'test_final_embedding_package.py')
put, sha, canonical = fixtures.put, package.sha, package.canonical
SOURCE_DOCS = ('STARTED.json', 'SOURCE_BINDING.json', 'OBSERVER_BINDING.json', 'SOURCE_LEDGER.json',
               'COMBINED_OBSERVATIONS.json', 'SELECTION.json', 'TRAINING_PINS.json', 'OBSERVATION_PINS.json')


def refresh(root, docs):
    """Keep invented input identities explicit rather than loading a real model."""
    started, result = docs['STARTED.json'], docs['RESULT.json']
    binding, pins, selected = (docs[name] for name in
                              ('SOURCE_BINDING.json', 'TRAINING_PINS.json', 'SELECTION.json'))
    started['training_pins_sha256'] = sha(canonical(pins))
    started['worker_code_sha256'] = pins['code_sha256']
    result.update(started)
    bound = {
        'source_ledger_canonical_sha256': binding['source_ledger_canonical_sha256'],
        'source_binding_sha256': sha(canonical(binding)),
        'observation_pins_sha256': sha(canonical(docs['OBSERVATION_PINS.json'])),
        'combined_observations_sha256': sha(canonical(docs['COMBINED_OBSERVATIONS.json']) + b'\n'),
        'observer_artifact_manifest_sha256': sha(canonical(docs['OBSERVER_BINDING.json'])),
        'observation_manifest_sha256': selected['observer_artifact_manifest_sha256'],
        'recipe_sha256': binding['recipe_sha256'], 'selection_sha256': sha(canonical(selected)),
        'selection_policy_sha256': selected['policy_sha256'],
        'legacy_worker_sha256': pins['legacy_worker_sha256'],
        'terminal_observer_code_sha256': pins['terminal_observer_code_sha256'],
        'pipeline_source_sha256': sha(canonical(docs['FEATURE_PREFLIGHT.json']['pipeline_source'])),
        'feature_evidence_sha256': sha(canonical(docs['FEATURE_EVIDENCE.json']) + b'\n'),
        'feature_stream_sha256': package.identity(root / 'actual-embedding-streams.npz')['sha256'],
        'worker_code_sha256': started['worker_code_sha256'],
        'corpus_sha256': docs['LIBRISPEECH_SOURCES.json']['corpus']['sha256'],
        'reference_sha256': docs['LIBRISPEECH_SOURCES.json']['reference']['member_sha256']}
    trained = package.read_json(root / 'train/RESULT.json')
    trained['input_binding'] = result['input_binding'] = bound
    docs['TRAINING_STARTED.json'] = {'input_binding': bound, 'maximum_optimizer_updates': 96,
                                    'development_tuning': False, 'no_retry': True}
    for name, value in docs.items():
        put(root / name, value)
    put(root / 'train/RESULT.json', trained)


def fake_terminal_training(root):
    docs = fixtures.fake_training(root)
    source = {'run_id': '43816869386', 'github_sha': '8' * 40, 'run_attempt': '1', 'conclusion': 'failure'}
    observer = {'run_id': '10001', 'github_sha': '9' * 40, 'run_attempt': '1', 'test_backend': False}
    scope = {'source_identity': source, 'observer_identity': observer, 'source_denominator': 144,
             'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'valid_source_count': 143,
             'source_dispositions': {'VALID_UNOBSERVED_SOURCE': 143, 'FAILED_NO_RETRY': 1}}
    docs['STARTED.json'].update(scope, run_id='10002', run_attempt='1', github_sha='a' * 40)
    docs['RESULT.json'].update(scope, status=package.SUCCESS, new_tts_asr_phone_calls=0,
        exposed_read_speech_tracks={'seconds': 400.0, 'tracks': 40})
    counts = {group: {'count': count, 'families': ['ARTIFICIAL_FAMILY']}
              for group, count in (('train:0', 32), ('train:1', 32), ('train:2', 31),
                                   ('development:0', 16), ('development:1', 16), ('development:2', 16))}
    minimum = {'passed': True, 'counts': counts}
    docs['RESULT.json']['minimum'] = minimum
    docs['MINIMUM_AFTER_DEDUPLICATION.json'] = minimum
    docs['MINIMUM_AFTER_FEATURE_GEOMETRY.json'] = minimum
    docs['SOURCE_BINDING.json'] = {**scope, 'test_backend': False, 'recipe_sha256': 'e' * 64,
                                  'source_ledger_canonical_sha256': '2' * 64}
    docs['OBSERVER_BINDING.json'] = {'observer_identity': observer, 'ARTIFICIAL_FIXTURE': True,
                                   'five_original_artifact_records': ['fixture'] * 5}
    docs['SOURCE_LEDGER.json'] = {**scope, 'ARTIFICIAL_FIXTURE': True, 'rows': [
        {'cell_id': 'fixture-%03d' % n, 'source_disposition':
         'VALID_UNOBSERVED_SOURCE' if n < 143 else 'FAILED_NO_RETRY'} for n in range(144)]}
    docs['COMBINED_OBSERVATIONS.json'] = {'status': 'FOUR_OBSERVERS_COMPLETE', 'ARTIFICIAL_FIXTURE': True}
    docs['SELECTION.json']['policy_sha256'] = '3' * 64
    docs['TRAINING_PINS.json'] = {'code_sha256': 'b' * 64, 'legacy_worker_sha256': 'c' * 64,
        'terminal_observer_code_sha256': 'd' * 64, 'pipeline_commit': '1' * 40, 'ARTIFICIAL_FIXTURE': True}
    docs['OBSERVATION_PINS.json'] = {'ARTIFICIAL_FIXTURE': True}
    docs['LIBRISPEECH_SOURCES.json'] = {'corpus': {'sha256': '4' * 64},
                                       'reference': {'member_sha256': '5' * 64}}
    refresh(root, docs)
    return docs


def source_validation_stub(root):
    return {name: package.read_json(root / name) for name in SOURCE_DOCS}


def fake_build_inputs(base):
    training = base / 'training-result'
    docs = fake_terminal_training(training)
    data = base / 'data' / 'automation'
    data.mkdir(parents=True)
    for name in (*package.AUTOMATION_FILES, 'english_final_embedding_infer.py', 'package_final_embedding.py'):
        (data / name).write_bytes(('ARTIFICIAL FIXTURE SOURCE: ' + name).encode())
    pipeline = base / 'pipeline'
    files = {}
    for name in package.contract.RUNTIME_FILES:
        target = pipeline / 'research/english_final_embedding' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(('ARTIFICIAL RUNTIME: ' + name).encode())
        files[name] = package.identity(target)['sha256']
    helper = pipeline / 'research/english63_online_c/librispeech_screen.py'
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b'ARTIFICIAL SOURCE HELPER')
    audit = pipeline / 'configs/research/english-embedding-backbone-assets-20261116.json'
    audit.parent.mkdir(parents=True)
    audit.write_bytes(b'{"ARTIFICIAL_FIXTURE":true}')
    (pipeline / 'LICENSE').write_bytes(b'ARTIFICIAL DOWNSTREAM LICENSE')
    upstream = base / 'upstream-license'
    upstream.write_bytes(b'ARTIFICIAL UPSTREAM LICENSE')
    assets = base / 'assets'
    assets.mkdir()
    asset_constants = {}
    for name in package.contract.ASSETS:
        (assets / name).write_bytes(('ARTIFICIAL ONNX PLACEHOLDER: ' + name).encode())
        entry = package.identity(assets / name)
        asset_constants[name] = (entry['bytes'], entry['sha256'])
    docs['FEATURE_PREFLIGHT.json']['asset_identities'] = {k: list(v) for k, v in asset_constants.items()}
    docs['FEATURE_PREFLIGHT.json']['pipeline_source'].update(feature_head_files_sha256=files,
        librispeech_source_helper_sha256=package.identity(helper)['sha256'])
    for field, name in (('code_sha256', 'terminal144_train_once.py'),
                        ('legacy_worker_sha256', 'english_final144_embedding_once.py'),
                        ('terminal_observer_code_sha256', 'terminal144_observe_once.py')):
        docs['TRAINING_PINS.json'][field] = package.identity(data / name)['sha256']
    put(data / 'terminal144_training_pins.json', docs['TRAINING_PINS.json'])
    put(data / 'terminal144_observation_pins.json', docs['OBSERVATION_PINS.json'])
    (base / 'training-pip-freeze.txt').write_bytes(b'ARTIFICIAL DEPENDENCY FREEZE\n')
    refresh(training, docs)
    args = argparse.Namespace(training_result=training, pipeline_source=pipeline, assets=assets,
        upstream_license=upstream, output=base / 'bundle')

    def git_bytes(command, text=False):
        if command[3:5] == ['rev-parse', 'HEAD']:
            answer = ('1' if Path(command[2]) == pipeline else 'a') * 40 + '\n'
            return answer if text else answer.encode()
        if command[3:5] == ['status', '--porcelain']:
            return '' if text else b''
        if command[3] == 'show':
            name = command[4].split(':automation/')[1]
            return ((HERE / name) if name == 'terminal144_package.py' else (data / name)).read_bytes()
        raise AssertionError('unexpected git command: ' + repr(command))

    stack = ExitStack()
    stack.enter_context(patch.object(package, 'AUTOMATION', data))
    stack.enter_context(patch.object(package, 'source_evidence', side_effect=source_validation_stub))
    stack.enter_context(patch.object(package.contract, 'ASSETS', asset_constants))
    lid = package.identity(upstream)
    stack.enter_context(patch.object(package.contract, 'UPSTREAM_LICENSE', (lid['bytes'], lid['sha256'])))
    stack.enter_context(patch.object(package.subprocess, 'check_output', side_effect=git_bytes))
    return args, docs, stack


class TerminalPackageTests(unittest.TestCase):
    def test_default_is_zero_model_and_zero_write(self):
        code = ('import importlib.util,sys;'
                f's=importlib.util.spec_from_file_location("fixture",{str(HERE / "terminal144_package.py")!r});'
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main([]);'
                'sys.exit(any(k in sys.modules for k in ("numpy","onnx","onnxruntime","torch")))')
        run = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['model_calls'], 0)

    def test_new_source_validator_is_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'training'
            fake_terminal_training(root)
            with patch.object(package, 'source_evidence', side_effect=ValueError('SOURCE_REJECTED')) as validator:
                with self.assertRaisesRegex(ValueError, 'SOURCE_REJECTED'):
                    package.training_evidence(root)
                validator.assert_called_once_with(root)

    def test_complete_separate_three_run_fixture_and_return_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'training'
            fake_terminal_training(root)
            with patch.object(package, 'source_evidence', side_effect=source_validation_stub):
                evidence, trained, weights, binary = package.training_evidence(root)
            self.assertEqual(evidence['RESULT.json']['valid_source_count'], 143)
            self.assertEqual(trained['optimizer_updates'], 96)
            self.assertEqual(weights['sha256'], trained['weights_sha256'])
            self.assertEqual(binary['bytes'], 197908)

    def test_original_or_observer_identity_cannot_be_the_training_identity(self):
        for kind in ('source_identity', 'observer_identity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'training'
                docs = fake_terminal_training(root)
                for key in ('run_id', 'github_sha'):
                    docs['STARTED.json'][kind][key] = docs['STARTED.json'][key]
                refresh(root, docs)
                with patch.object(package, 'source_evidence', side_effect=source_validation_stub):
                    with self.assertRaisesRegex(ValueError, 'DISTINCT_SOURCE_OBSERVER'):
                        package.training_evidence(root)

    def test_old_complete_status_or_changed_denominator_does_not_pass(self):
        for key, value in (('status', 'EXECUTED_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY'),
                           ('source_denominator', 143), ('optimizer_updates', 95),
                           ('new_tts_asr_phone_calls', 1), ('test_backend', True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'training'
                docs = fake_terminal_training(root)
                docs['RESULT.json'][key] = value
                put(root / 'RESULT.json', docs['RESULT.json'])
                with patch.object(package, 'source_evidence', side_effect=source_validation_stub):
                    with self.assertRaises(ValueError):
                        package.training_evidence(root)

    def test_saved_npz_cache_and_arm_stream_corruption_are_rejected(self):
        for name in ('train/causal-head.npz', 'actual-embedding-streams.npz', 'arm-parity/stream.bin'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'training'
                fake_terminal_training(root)
                (root / name).write_bytes(b'CHANGED ARTIFICIAL BYTES')
                with patch.object(package, 'source_evidence', side_effect=source_validation_stub):
                    with self.assertRaises(ValueError):
                        package.training_evidence(root)

    def test_two_observer_manifest_meanings_cannot_be_interchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'training'
            docs = fake_terminal_training(root)
            wrong = docs['SELECTION.json']['observer_artifact_manifest_sha256']
            for name in ('RESULT.json', 'TRAINING_STARTED.json', 'train/RESULT.json'):
                doc = package.read_json(root / name)
                doc['input_binding']['observer_artifact_manifest_sha256'] = wrong
                put(root / name, doc)
            with patch.object(package, 'source_evidence', side_effect=source_validation_stub):
                with self.assertRaisesRegex(ValueError, 'ACTUAL_TERMINAL_TRAINING_INPUT_BINDINGS'):
                    package.training_evidence(root)

    def test_partial_failure_and_changed_saved_pin_are_rejected(self):
        for mode in ('failure', 'pin', 'arm_event'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'training'
                docs = fake_terminal_training(root)
                if mode == 'failure':
                    put(root / 'FAILURE.json', {'status': 'FAILED_NO_RETRY'})
                elif mode == 'pin':
                    docs['TRAINING_PINS.json']['code_sha256'] = 'f' * 64
                    put(root / 'TRAINING_PINS.json', docs['TRAINING_PINS.json'])
                else:
                    docs['PARITY.json']['arm-parity']['events'] = [{'sample': 35200, 'keyword': 1}]
                    put(root / 'PARITY.json', docs['PARITY.json'])
                with patch.object(package, 'source_evidence', side_effect=source_validation_stub):
                    with self.assertRaises(ValueError):
                        package.training_evidence(root)

    def test_source_card_preserves_failed_denominator_and_three_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'training'
            docs = fake_terminal_training(root)
            card = package.model_card(docs['RESULT.json'], package.read_json(root / 'train/RESULT.json'),
                                      docs['FEATURE_PREFLIGHT.json'], 1234)
            self.assertIn('all 144 declared cells', card)
            self.assertIn('143 verified original audio sources', card)
            self.assertIn('FAILED_NO_RETRY', card)
            self.assertNotIn('all 144 generated cells', card)
            for run in ('43816869386', '10001', '10002'):
                self.assertIn(run, card)
            self.assertIn('CENSORED_NOT_FRR', card)

    def test_full_artificial_package_matches_fixed_infer_contract_and_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, docs, stack = fake_build_inputs(Path(tmp))
            with stack:
                result = package.build(args)
                manifest = package.contract.verify_bundle(args.output)
                self.assertEqual(result['model_calls'], 0)
                self.assertEqual(manifest['provenance']['source_denominator'], 144)
                self.assertEqual(manifest['provenance']['valid_source_count'], 143)
                self.assertEqual(manifest['provenance']['training_identity']['run_id'], '10002')
                self.assertEqual(manifest['provenance']['source_identity']['run_id'], '43816869386')
                self.assertTrue(set('evidence/' + n for n in package.EVIDENCE) <= set(manifest['files']))
                with self.assertRaisesRegex(ValueError, 'CREATE_ONLY_BUNDLE_DIRECTORY'):
                    package.build(args)

    def test_exact_assets_license_and_committed_worker_pins_required(self):
        for mode in ('asset', 'license', 'saved_pins'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                args, _, stack = fake_build_inputs(Path(tmp))
                with stack:
                    if mode == 'asset':
                        (args.assets / 'embedding_model.onnx').write_bytes(b'WRONG ASSET')
                    elif mode == 'license':
                        args.upstream_license.write_bytes(b'REWRITTEN LICENSE')
                    else:
                        put(package.AUTOMATION / 'terminal144_training_pins.json', {'WRONG': True})
                    with self.assertRaises(ValueError):
                        package.build(args)
                    self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
