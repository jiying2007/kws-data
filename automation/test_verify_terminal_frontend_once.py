"""Source-only verification-worker boundaries; all values and pins are invented."""
from __future__ import annotations
import copy
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('verification_source_fixture', HERE / 'verify_terminal_frontend_once.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def fake_pins():
    pins = worker.pins_template()
    pins.update(enabled=True, main_before='a' * 40, worker_code_sha256='c' * 64,
                frontend_sources_sha256={name: 'd' * 64 for name in worker.FRONTEND_FILES})
    pins['head_training'].update(run_id='123', github_sha='e' * 40, artifact_id=1001,
        zip_bytes=100, zip_sha256='1' * 64, result_member='RESULT.json', result_sha256='f' * 64,
        weights_member='train/causal-head.npz', weights_sha256='1' * 64,
        head_binary_member='causal-head.bin', head_binary_sha256='2' * 64,
        source_identity=copy.deepcopy(worker.SOURCE_IDENTITY),
        observer_identity={'run_id': '122', 'github_sha': 'd' * 40, 'run_attempt': '1', 'test_backend': False})
    pins['static_export'].update(run_id='124', github_sha=worker.PIPELINE_COMMIT,
        artifact_id=1002, zip_bytes=100, zip_sha256='2' * 64, result_member='RESULT.json', result_sha256='3' * 64,
        generated={name: {'member': 'generated/' + name, 'bytes': 1, 'sha256': '4' * 64}
                   for name in worker.GENERATED_FILES})
    return pins


def fake_environment():
    env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': 'jiying2007/kws-data',
           'GITHUB_EVENT_NAME': 'push', 'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1',
           'GITHUB_RUN_ID': '456', 'GITHUB_SHA': '3' * 40}
    event = {'before': 'a' * 40, 'after': '3' * 40,
             'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
    return env, event


def fake_row(sample=16000):
    return {'sample': sample, **{stage: [0.0] * count for stage, count in worker.STAGE_COUNTS.items()}}


def put_zip(path, files, pin):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    raw = path.read_bytes()
    pin.update(zip_bytes=len(raw), zip_sha256=worker.sha(raw))


def artificial_static_export(root, pins):
    """Tiny invented C/JSON bytes and metadata; never compile or run these."""
    pipeline = root / 'pipeline'
    frontend = pipeline / 'research/frontend'
    frontend.mkdir(parents=True)
    code, hashes = {}, {}
    for name in worker.FRONTEND_FILES:
        content = ('ARTIFICIAL SOURCE FIXTURE ' + name).encode()
        (frontend / name).write_bytes(content)
        hashes[name] = worker.sha(content)
        code['research/frontend/' + name] = {'bytes': len(content), 'sha256': hashes[name]}
    pins['frontend_sources_sha256'] = hashes
    graphs = {name: {'asset_bytes': size, 'asset_sha256': digest, 'constant_bytes': 4,
                    'workspace_bytes': 4, 'input': {'name': 'ARTIFICIAL_INPUT'},
                    'output': {'name': 'ARTIFICIAL_OUTPUT'}} for name, (size, digest) in worker.ASSETS.items()}
    generated = {'fixed_models.c': b'/* ARTIFICIAL C BYTES, NO MODEL */\n',
                 'fixed_models.h': b'/* ARTIFICIAL HEADER */\n'}
    manifest = {'schema': 'fixed-feature-pure-export-v1', 'real_inference_performed': False,
                'numerical_parity_established': False, 'ssc305_qualification': False,
                'graphs': graphs, 'generated_c_sha256': worker.sha(generated['fixed_models.c'])}
    generated['export_manifest.json'] = worker.canonical(manifest) + b'\n'
    entries = {name: {'member': 'generated/' + name, 'bytes': len(raw), 'sha256': worker.sha(raw)}
               for name, raw in generated.items()}
    pins['static_export']['generated'] = entries
    assets = {name: {'bytes': size, 'sha256': digest} for name, (size, digest) in worker.ASSETS.items()}
    report = {'schema': worker.STATIC_SCHEMA, 'status': 'EXPORTED_STATIC_GRAPHS_ONLY',
        'repository': 'jiying2007/kws-pipeline', 'run_id': pins['static_export']['run_id'],
        'run_attempt': 1, 'source_head': worker.PIPELINE_COMMIT, 'export_attempted': True,
        'parser_versions': {'numpy': '1.26.4', 'onnx': '1.17.0'}, 'model_calls': 0,
        'onnx_inference_calls': 0, 'c_inference_calls': 0, 'optimizer_updates': 0, 'tts_asr_phone_calls': 0,
        'numerical_parity_established': False, 'ssc305_qualification': False,
        'assets': assets, 'expected_assets': assets, 'code_files': code, 'graphs': graphs,
        'generated': {'generated/' + name: {k: entry[k] for k in ('bytes', 'sha256')}
                      for name, entry in entries.items()}}
    files = {'RESULT.json': worker.canonical(report) + b'\n',
             **{'generated/' + name: raw for name, raw in generated.items()}}
    pins['static_export']['result_sha256'] = worker.sha(files['RESULT.json'])
    archive = root / 'static.zip'
    put_zip(archive, files, pins['static_export'])
    args = argparse.Namespace(pipeline_source=pipeline, frontend_source=frontend, static_export_zip=archive)
    return args, report, generated, files, hashes


def artificial_training_result(pins):
    head = pins['head_training']
    return {'schema': worker.TRAINING_SCHEMA, 'status': worker.TRAINING_SUCCESS,
        **{key: copy.deepcopy(head[key]) for key in ('run_id', 'github_sha', 'run_attempt',
             'source_identity', 'observer_identity', 'worker_code_sha256')},
        'source_denominator': 144, 'valid_source_count': 143,
        'source_dispositions': {'VALID_UNOBSERVED_SOURCE': 143, 'FAILED_NO_RETRY': 1},
        'original_campaign_status': 'INCOMPLETE_NO_RETRY', 'test_backend': False,
        **dict.fromkeys(worker.FALSE_FLAGS, False)}


class VerificationSourceTests(unittest.TestCase):
    def test_disabled_template_has_no_guessed_artifact_or_training_identity(self):
        pins = worker.pins_template()
        self.assertFalse(pins['enabled'])
        self.assertIsNone(pins['main_before'])
        self.assertIsNone(pins['worker_code_sha256'])
        for group in ('head_training', 'static_export'):
            for key in ('run_id', 'github_sha', 'artifact_id', 'zip_bytes', 'zip_sha256', 'result_sha256'):
                self.assertIsNone(pins[group][key])
        self.assertIsNone(pins['head_training']['weights_sha256'])
        self.assertIsNone(pins['head_training']['head_binary_sha256'])
        with self.assertRaisesRegex(ValueError, 'DISABLED'):
            worker.validate_pins(pins)
        with self.assertRaises(ValueError):
            worker.validate_pins({**pins, 'enabled': True})

    def test_frozen_numeric_plan_and_core_are_identical_to_original_draft(self):
        self.assertEqual(worker.sha(worker.canonical(worker.PLAN)),
                         'cdb8cce44b3ba88c92afeaf2b1d64124190c46198091f375d036a099964a16fb')
        names = ('synthetic_pcm', 'chunks', 'trace_bytes', 'parse_trace', 'errors', 'head_comparison',
                 'comparisons', 'replay_summary', 'build_replays', 'attach_head', 'reference_trace')
        tree = ast.parse((HERE / 'verify_terminal_frontend_once.py').read_text())
        body = [ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual(worker.sha(json.dumps(body, separators=(',', ':')).encode()),
                         '6d60b5b9e26b948be6ef823a6eb4b8213c627b997d32257a9c1dd6ac73d03901')

    def test_missing_actual_artifact_pins_and_identity_reuse_are_rejected(self):
        for group, key in (('head_training', 'artifact_id'), ('head_training', 'zip_sha256'),
                           ('static_export', 'result_sha256'), ('static_export', 'run_id')):
            pins = fake_pins()
            pins[group][key] = None
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                worker.validate_pins(pins)
        pins = fake_pins()
        env, event = fake_environment()
        for prior in (pins['head_training'], pins['head_training']['source_identity'],
                      pins['head_training']['observer_identity']):
            with self.subTest(run=prior['run_id']), self.assertRaisesRegex(ValueError, 'NEW_VERIFICATION'):
                worker.execution_guard(pins, {**env, 'GITHUB_RUN_ID': prior['run_id']}, event, 'c' * 64)

    def test_new_training_schema_keeps_144_dispositions_and_separate_identities(self):
        pins = fake_pins()
        result = artificial_training_result(pins)
        worker.verify_training_profile(result, pins['head_training'])
        for key, value in (('schema', 'OLD_ALL_COMPLETE_SCHEMA'),
                           ('status', 'EXECUTED_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY'),
                           ('source_denominator', 143), ('valid_source_count', 144),
                           ('original_campaign_status', 'COMPLETE'), ('test_backend', True)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'ACTUAL_TERMINAL_TRAINING_PROFILE'):
                worker.verify_training_profile({**result, key: value}, pins['head_training'])
        wrong = copy.deepcopy(result)
        wrong['observer_identity']['run_id'] = wrong['run_id']
        with self.assertRaises(ValueError):
            worker.verify_training_profile(wrong, pins['head_training'])

    def test_training_zip_binds_original_result_npz_and_binary_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, pins = Path(tmp), fake_pins()
            training = root / 'training-result'
            (training / 'train').mkdir(parents=True)
            files = {'RESULT.json': worker.canonical(artificial_training_result(pins)) + b'\n',
                     'train/causal-head.npz': b'ARTIFICIAL NPZ, NOT A MODEL',
                     'causal-head.bin': b'ARTIFICIAL HEAD BINARY'}
            for name, raw in files.items():
                (training / name).write_bytes(raw)
            for key, name in (('result_sha256', 'RESULT.json'), ('weights_sha256', 'train/causal-head.npz'),
                              ('head_binary_sha256', 'causal-head.bin')):
                pins['head_training'][key] = worker.sha(files[name])
            archive = root / 'training.zip'
            put_zip(archive, files, pins['head_training'])
            args = argparse.Namespace(training_result=training, training_zip=archive)
            worker.training_archive(args, pins)
            (training / 'train/causal-head.npz').write_bytes(b'CHANGED LOCAL NPZ')
            with self.assertRaisesRegex(ValueError, 'ACTUAL_TRAINING_ARTIFACT_MEMBER'):
                worker.training_archive(args, pins)
            (training / 'train/causal-head.npz').write_bytes(files['train/causal-head.npz'])
            put_zip(archive, {**files, '../outside': b'NO EXTRACTION'}, pins['head_training'])
            with self.assertRaisesRegex(ValueError, 'SAFE_ARTIFACT_MEMBER'):
                worker.training_archive(args, pins)

    def test_static_export_consumes_only_exact_original_generated_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pins = fake_pins()
            args, report, generated, _, hashes = artificial_static_export(Path(tmp), pins)
            worker.validate_pins(pins)
            actual, original_bytes = worker.static_export_archive(args, pins, hashes)
            self.assertEqual(actual, report)
            self.assertEqual(original_bytes, generated)
            (args.frontend_source / 'fixed_frontend.c').write_bytes(b'CHANGED SOURCE')
            with self.assertRaisesRegex(ValueError, 'STATIC_EXPORT_SOURCE_BYTES'):
                worker.static_export_archive(args, pins, hashes)

    def test_partial_static_export_or_nonzero_calls_are_not_evidence(self):
        for key, value in (('status', 'FAILED_NO_RETRY'), ('onnx_inference_calls', 1),
                           ('source_head', 'f' * 40), ('numerical_parity_established', True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                pins = fake_pins()
                args, report, _, files, hashes = artificial_static_export(Path(tmp), pins)
                report[key] = value
                files['RESULT.json'] = worker.canonical(report) + b'\n'
                pins['static_export']['result_sha256'] = worker.sha(files['RESULT.json'])
                put_zip(args.static_export_zip, files, pins['static_export'])
                with self.assertRaisesRegex(ValueError, 'COMPLETED_ZERO_INFERENCE'):
                    worker.static_export_archive(args, pins, hashes)

    def test_static_c_manifest_and_zip_tampering_fail_before_models(self):
        for mode in ('zip', 'generated', 'manifest', 'failed'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                pins = fake_pins()
                args, report, _, files, hashes = artificial_static_export(Path(tmp), pins)
                if mode == 'zip':
                    args.static_export_zip.write_bytes(args.static_export_zip.read_bytes() + b'CHANGED ZIP')
                else:
                    if mode == 'generated':
                        files['generated/fixed_models.c'] = b'CHANGED C'
                    elif mode == 'failed':
                        files['FAILURE.json'] = b'{}'
                    else:
                        manifest = json.loads(files['generated/export_manifest.json'])
                        manifest['generated_c_sha256'] = 'f' * 64
                        raw = worker.canonical(manifest) + b'\n'
                        files['generated/export_manifest.json'] = raw
                        changed = {'bytes': len(raw), 'sha256': worker.sha(raw)}
                        pins['static_export']['generated']['export_manifest.json'].update(changed)
                        report['generated']['generated/export_manifest.json'] = changed
                        files['RESULT.json'] = worker.canonical(report) + b'\n'
                        pins['static_export']['result_sha256'] = worker.sha(files['RESULT.json'])
                    put_zip(args.static_export_zip, files, pins['static_export'])
                with self.assertRaises(ValueError):
                    worker.static_export_archive(args, pins, hashes)

    def test_default_contract_imports_no_ml_or_creates_output(self):
        path = HERE / 'verify_terminal_frontend_once.py'
        code = ('import importlib.util,sys;'
                f's=importlib.util.spec_from_file_location("fixture",{str(path)!r});'
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main([]);'
                'sys.exit(any(k in sys.modules for k in ("numpy","onnx","onnxruntime","torch")))')
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], cwd=tmp,
                                 check=True, capture_output=True, text=True)
            result = json.loads(run.stdout)
            self.assertEqual(result['model_calls'], 0)
            self.assertFalse(result['execution_enabled'])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_explicit_execute_still_rejects_disabled_before_event_or_ml(self):
        with tempfile.TemporaryDirectory() as tmp:
            pins = Path(tmp) / 'pins.json'
            pins.write_text(json.dumps({'schema': worker.PINS_SCHEMA, 'enabled': False}))
            cli = ['--execute', '--pins', str(pins)]
            for name in ('pipeline-source', 'frontend-source', 'training-result', 'training-zip', 'static-export-zip', 'assets', 'output'):
                cli += ['--' + name, str(Path(tmp) / name)]
            with self.assertRaisesRegex(ValueError, 'DISABLED'):
                worker.main(cli)
            self.assertFalse((Path(tmp) / 'output').exists())

    def test_once_guard_and_all_environment_boundaries(self):
        pins, (env, event) = fake_pins(), fake_environment()
        worker.execution_guard(pins, env, event, 'c' * 64)
        for key, value in (('GITHUB_ACTIONS', 'false'), ('GITHUB_RUN_ATTEMPT', '2'),
                           ('GITHUB_REF', 'refs/heads/branch'), ('GITHUB_EVENT_NAME', 'workflow_dispatch'),
                           ('GITHUB_REPOSITORY', 'another/repository')):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'FIRST_PINNED_PUBLIC_MAIN_PUSH_ONLY'):
                worker.execution_guard(pins, {**env, key: value}, event, 'c' * 64)
        for altered in ({**event, 'before': '4' * 40}, {**event, 'after': '5' * 40},
                        {**event, 'repository': {**event['repository'], 'private': True}}):
            with self.assertRaises(ValueError):
                worker.execution_guard(pins, env, altered, 'c' * 64)
        with self.assertRaises(ValueError):
            worker.execution_guard(pins, env, event, '6' * 64)

    def test_missing_head_source_or_changed_plan_cannot_enable(self):
        for change in ('head', 'source', 'plan', 'admission', 'retry'):
            pins = fake_pins()
            if change == 'head':
                del pins['head_training']['weights_sha256']
            elif change == 'source':
                del pins['frontend_sources_sha256']['fixed_frontend.c']
            elif change == 'plan':
                pins['plan_sha256'] = '0' * 64
            elif change == 'admission':
                pins['qualification_allowed'] = True
            else:
                pins['no_retry'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.validate_pins(pins)

    def test_artificial_pcm_is_fixed_and_chunking_consumes_each_sample_once(self):
        values = worker.synthetic_pcm()
        self.assertEqual(list(values), ['zero', 'sine400', 'lcg14'])
        for name, raw in values.items():
            self.assertEqual(len(raw), 96000)
            self.assertEqual(worker.sha(raw), worker.PCM_SHA[name])
            chunks = list(worker.chunks(raw))
            self.assertEqual([len(chunk) for chunk in chunks[:4]], list(worker.CHUNKS))
            self.assertEqual(sum(map(len, chunks)), 48000)
            self.assertEqual(b''.join(struct.pack('<%dh' % len(chunk), *chunk) for chunk in chunks), raw)
        self.assertEqual(values['sine400'][:80], values['sine400'][80:160])
        self.assertEqual(worker.SAMPLES, tuple(16000 + i * 1280 for i in range(26)))
        self.assertEqual(sum(sample >= 35200 for sample in worker.SAMPLES), 11)

    def test_trace_contract_is_exact_and_rejects_extra_eof_output(self):
        rows = [fake_row(16000), fake_row(17280)]
        raw = worker.trace_bytes(rows)
        parsed = worker.parse_trace(raw, (16000, 17280))
        self.assertEqual([row['sample'] for row in parsed], [16000, 17280])
        self.assertEqual(len(raw), 8 + 2 * 22624)
        for changed in (raw[:-1], raw + b'\0', b'WRONGMAG' + raw[8:]):
            with self.assertRaisesRegex(ValueError, 'TRACE_LENGTH_OR_MAGIC'):
                worker.parse_trace(changed, (16000, 17280))

    def test_wrong_clock_stage_count_and_nonfinite_are_rejected(self):
        raw = worker.trace_bytes([fake_row()])
        for stage, count, sample in ((2, 3104, 16000), (1, 3103, 16000), (1, 3104, 16001)):
            changed = raw[:8] + struct.pack('<IIQ', stage, count, sample) + raw[24:]
            with self.assertRaisesRegex(ValueError, 'TRACE_STAGE_COUNT_CLOCK'):
                worker.parse_trace(changed, (16000,))
        with self.assertRaisesRegex(ValueError, 'NONFINITE_C_TRACE'):
            worker.parse_trace(raw[:24] + struct.pack('<f', float('nan')) + raw[28:], (16000,))
        row = fake_row()
        row[4][0] = 0.5
        with self.assertRaisesRegex(ValueError, 'EXACT_HEAD_DISCRETE_FIELDS'):
            worker.parse_trace(worker.trace_bytes([row]), (16000,))

    def test_predeclared_error_table_is_frozen_in_manifest_plan(self):
        expected = {'raw_mel': (0.01, 2e-5), 'normalized_mel': (0.001, 2e-5),
                    'embedding': (0.001, 2e-4), 'head_logits': (0.003, 2e-4),
                    'head_probabilities': (0.0005, 0.0)}
        self.assertEqual(worker.LIMITS, expected)
        self.assertEqual(worker.PLAN['limits'], {name: {'atol': a, 'rtol': r}
                                               for name, (a, r) in expected.items()})
        self.assertEqual(worker.PLAN['maximum_reference_onnx_session_runs'], 156)
        self.assertEqual(worker.PLAN['real_audio_cases'], 0)

    def test_c_summary_binds_observed_samples_history_events_and_memory(self):
        rows = [fake_row(sample) for sample in worker.SAMPLES]
        for row in rows:
            row[4][0] = float(row['sample'] >= 35200)
        summary = {'samples': 48000, 'embeddings': 26, 'head_ready': 11, 'events': 0,
                   'head_enabled': True, 'workspace_bytes': 64000, 'frontend_state_bytes': 32128,
                   'ssc305_qualification': False}
        self.assertEqual(worker.replay_summary(json.dumps(summary), rows), summary)
        for key, value in (('samples', 49280), ('head_ready', 12), ('events', 1),
                           ('head_enabled', False), ('ssc305_qualification', True), ('workspace_bytes', 0)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'C_PROCESS_SAMPLE'):
                worker.replay_summary(json.dumps({**summary, key: value}), rows)

    def test_error_count_worst_positions_and_zero_relative_semantics(self):
        result = worker.errors([[0.0, 2.0, 100.0]], [[0.0006, 2.002, 100.002]], [16000], 0.001, 0.00001)
        self.assertEqual(result['elements'], 3)
        self.assertEqual(result['violations'], 1)
        self.assertEqual(result['max_relative_at']['component'], 1)
        self.assertEqual(result['max_relative_at']['sample'], 16000)
        self.assertEqual(result['zero_reference_mismatch_count'], 1)
        self.assertEqual(result['nonzero_reference_elements'], 2)
        self.assertGreater(result['max_tolerance_ratio'], 1)
        all_zero = worker.errors([[0.0]], [[0.0]], [16000], 0.001, 0.0)
        self.assertEqual(all_zero['max_abs_at']['component'], 0)
        self.assertIsNone(all_zero['max_relative_error_nonzero_reference'])
        self.assertIsNone(all_zero['max_relative_at'])

    def test_no_events_is_not_acoustic_trigger_coverage(self):
        reference = [fake_row()]
        result = worker.comparisons(reference, copy.deepcopy(reference))
        self.assertTrue(result['passed'])
        self.assertEqual(result['reference_event_count'], 0)
        self.assertEqual(result['candidate_event_count'], 0)
        self.assertFalse(result['acoustic_trigger_coverage_established'])

    def test_event_mismatch_fails_even_when_probability_error_is_allowed(self):
        row = fake_row(37760)
        row[4] = [1.0, 1.0, 0.0, 2.0, 0.0, 0.1, 0.8002, 0.0998]
        actual = copy.deepcopy(row)
        actual[4][1], actual[4][6], actual[4][7] = 0.0, 0.7999, 0.1001
        result = worker.head_comparison([row], [actual])
        self.assertEqual(result['metrics']['head_probabilities']['violations'], 0)
        self.assertEqual(result['event_mismatches'], 1)
        self.assertFalse(result['passed'])
        self.assertEqual((result['reference_event_count'], result['candidate_event_count']), (1, 0))

    def test_same_embedding_head_keeps_original_stricter_limits(self):
        row, actual = fake_row(35200), fake_row(35200)
        row[4][0] = actual[4][0] = 1.0
        actual[4][2], actual[4][5] = 0.00015, 0.0000025
        self.assertTrue(worker.head_comparison([row], [actual])['passed'])
        strict = worker.head_comparison([row], [actual], strict=True)
        self.assertEqual(strict['metrics']['head_logits']['violations'], 1)
        self.assertEqual(strict['metrics']['head_probabilities']['violations'], 1)
        self.assertFalse(strict['passed'])

    def test_source_manifest_labels_scope_and_never_claims_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'fixture.txt').write_bytes(b'ARTIFICIAL SOURCE FIXTURE ONLY')
            worker.manifest(root)
            saved = worker.read_json(root / 'MANIFEST.json')
            self.assertEqual(saved['scope'], worker.PLAN['scope'])
            self.assertEqual(saved['files']['fixture.txt']['sha256'], worker.sha(b'ARTIFICIAL SOURCE FIXTURE ONLY'))
            self.assertTrue(all(saved[flag] is False for flag in worker.FALSE_FLAGS))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    unittest.main()
