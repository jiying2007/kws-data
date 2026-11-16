"""Invented bytes and isolated guards only: no C process, NumPy, or model inference."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('snapshot160_frontend_fixture', HERE / 'verify_snapshot160_frontend_once.py')
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)


def enabled_fixture():
    pins = worker.pins_template()
    pins.update(enabled=True, main_before='1' * 40,
                worker_code_sha256=worker.sha(Path(worker.__file__).read_bytes()))
    pins['head_training'].update(run_id='101', github_sha='3' * 40, artifact_id=201,
        artifact_name='invented-training', zip_bytes=100, zip_sha256='4' * 64,
        result_member='training-result/RESULT.json', result_sha256='5' * 64,
        weights_member='training-result/train/causal-head.npz', weights_sha256='6' * 64,
        head_binary_member='training-result/causal-head.bin', head_binary_sha256='7' * 64,
        snapshot_sha256='8' * 64)
    return pins


def environment_fixture(pins):
    env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': worker.REPOSITORY, 'GITHUB_EVENT_NAME': 'push',
           'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_ID': '102', 'GITHUB_SHA': '9' * 40}
    event = {'ref': env['GITHUB_REF'], 'before': pins['main_before'], 'after': env['GITHUB_SHA'],
             'repository': {'full_name': worker.REPOSITORY, 'private': False}}
    return env, event


def api_fixture(pin, repository):
    run = {'id': int(pin['run_id']), 'head_sha': pin['github_sha'], 'run_attempt': 1,
           'head_branch': 'main', 'event': 'push', 'status': 'completed', 'conclusion': 'success',
           'repository': {'full_name': repository, 'private': False}}
    artifact = {'id': pin['artifact_id'], 'name': pin['artifact_name'], 'size_in_bytes': pin['zip_bytes'],
                'digest': 'sha256:' + pin['zip_sha256'], 'expired': False,
                'workflow_run': {'id': int(pin['run_id']), 'head_sha': pin['github_sha'], 'head_branch': 'main'}}
    return run, artifact


def invented_traces():
    full = bytearray(b'EFFTR01\0')
    backbone = bytearray(full)
    for index, clock in enumerate(range(16000, 48001, 1280)):
        for stage, count in ((1, 3104), (2, 2432), (3, 96), (4, 8)):
            values = [-0.0 if i == 0 else ((i + index + stage) % 17 - 8) / 16 for i in range(count)]
            if stage == 4:
                values = [float(index >= 15), 0.0, .1, -.2, .3, .3, .2, .5] if index >= 15 else [0.0] * 8
            record = struct.pack('<IIQ', stage, count, clock) + struct.pack('<%df' % count, *values)
            full.extend(record)
            if stage != 4:
                backbone.extend(record)
    return bytes(full), bytes(backbone)


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.backbone = invented_traces()

    def test_default_disabled_zero_runtime_imports(self):
        self.assertFalse(any(name in sys.modules for name in ('numpy', 'torch', 'onnx', 'onnxruntime')))
        pins = worker.pins_template()
        self.assertFalse(pins['enabled'])
        self.assertIsNone(pins['main_before'])
        self.assertEqual(pins['training_support_sha256'], worker.TRAINING_SUPPORT)
        self.assertTrue(all(value is None for key, value in pins['head_training'].items() if key != 'run_attempt'))
        worker.validate_pins(pins)
        with self.assertRaisesRegex(ValueError, 'DISABLED_OR_MISSING'):
            worker.validate_pins(pins, execution=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pins.json'
            path.write_text(json.dumps(pins), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(worker.main(['--pins', str(path)]), 0)
            self.assertEqual(json.loads(output.getvalue())['new_model_calls'], 0)

    def test_pending_schema_cannot_hide_missing_fields(self):
        pins = worker.pins_template()
        del pins['head_training']['weights_sha256']
        with self.assertRaisesRegex(ValueError, 'PENDING_TRAINING_SCHEMA'):
            worker.validate_pins(pins)
        pins = enabled_fixture(); pins['head_training']['run_attempt'] = True
        with self.assertRaises(ValueError):
            worker.validate_pins(pins)

    def test_new_plan_retains_all_numerical_limits_and_geometry(self):
        self.assertEqual(worker.sha(worker.canonical(worker.PLAN)), worker.FIXED_PLAN_SHA)
        changed = {'scope', 'maximum_reference_onnx_session_runs'}
        for key, value in worker.full.PLAN.items():
            if key not in changed:
                self.assertEqual(worker.PLAN[key], value, key)
        self.assertEqual(worker.PLAN['maximum_reference_onnx_session_runs'], 0)
        self.assertEqual(worker.PLAN['maximum_reference_head_streams'], 9)
        self.assertEqual(worker.PLAN['reference_head_embedding_steps'], 234)
        self.assertEqual(worker.PLAN['reference_head_ready_outputs'], 99)
        self.assertEqual(worker.PLAN['maximum_c_processes_per_backend'], 3)
        self.assertEqual(worker.PLAN['maximum_build_processes_per_backend'], 1)
        self.assertEqual(worker.PLAN['same_embedding_head_limits'],
                         {'logits_atol': 1e-4, 'probabilities_atol': 2e-6, 'rtol': 0.0})

    def test_public_first_successor_guard_rejects_every_wrong_axis(self):
        pins = enabled_fixture(); env, event = environment_fixture(pins)
        worker.execution_guard(pins, env, event)
        for key, bad in (('GITHUB_ACTIONS', 'false'), ('GITHUB_REPOSITORY', 'fork/repo'),
                         ('GITHUB_EVENT_NAME', 'workflow_dispatch'), ('GITHUB_REF', 'refs/heads/topic'),
                         ('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_RUN_ID', '101'),
                         ('GITHUB_SHA', pins['head_training']['github_sha'])):
            changed = {**env, key: bad}
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.execution_guard(pins, changed, event)
        for changed in ({**event, 'before': 'a' * 40}, {**event, 'ref': 'refs/heads/topic'},
                        {**event, 'repository': {'full_name': worker.REPOSITORY, 'private': True}}):
            with self.assertRaises(ValueError):
                worker.execution_guard(pins, env, changed)
        with self.assertRaisesRegex(ValueError, 'EXACT_FULLCHAIN_WORKER'):
            worker.execution_guard(pins, env, event, worker_sha='0' * 64)

    def test_frozen_limits_false_flags_and_mel_identity_cannot_change(self):
        for mutate in (lambda p: p.update(human_gold=True), lambda p: p['mel'].update(zip_sha256='0' * 64),
                       lambda p: p.update(plan_sha256='0' * 64),
                       lambda p: p.update(candidate_source_sha256='0' * 64)):
            pins = enabled_fixture(); mutate(pins)
            with self.assertRaises(ValueError):
                worker.validate_pins(pins)
        for key in ('artifact_id', 'zip_bytes', 'result_sha256', 'head_binary_sha256'):
            pins = enabled_fixture(); pins['head_training'][key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_pins(pins)

    def test_successful_run_and_artifact_are_independently_bound(self):
        pin = enabled_fixture()['head_training']; run, artifact = api_fixture(pin, worker.REPOSITORY)
        worker.api_identity(run, artifact, pin, worker.REPOSITORY)
        for key, value in (('status', 'in_progress'), ('conclusion', 'failure'), ('run_attempt', 2),
                           ('head_sha', '0' * 40), ('head_branch', 'branch')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.api_identity({**run, key: value}, artifact, pin, worker.REPOSITORY)
        for key, value in (('id', 999), ('name', 'another'), ('expired', True),
                           ('digest', 'sha256:' + '0' * 64), ('size_in_bytes', 101)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.api_identity(run, {**artifact, key: value}, pin, worker.REPOSITORY)

    def test_projection_keeps_full_frontend_bits_including_signed_zero(self):
        self.assertEqual(len(self.raw), 588232)
        self.assertEqual(len(self.backbone), 586984)
        self.assertEqual(worker.frontend_projection(self.raw), self.backbone)
        changed = bytearray(self.raw); changed[27] ^= 0x80  # -0.0 -> +0.0, numerical equality is insufficient.
        self.assertEqual(worker.full.parse_trace(self.raw)[0][1][0], worker.full.parse_trace(changed)[0][1][0])
        self.assertNotEqual(worker.frontend_projection(changed), self.backbone)

    def test_projection_rejects_wrong_clock_stage_length_or_nonfinite(self):
        cases = [self.raw[:-1], b'badmagic' + self.raw[8:]]
        for offset, value in ((8, 2), (16, 16001), (24, 0x7f800000)):
            changed = bytearray(self.raw); struct.pack_into('<I', changed, offset, value); cases.append(changed)
        for raw in cases:
            with self.assertRaises(ValueError):
                worker.frontend_projection(raw)

    def test_strict_head_gate_uses_saved_thresholds_and_exact_events(self):
        reference = worker.full.parse_trace(self.raw)
        actual = copy.deepcopy(reference)
        values = list(actual[-1][4]); values[2] += .0002; actual[-1][4] = values
        self.assertTrue(worker.full.head_comparison(reference, actual)['passed'])
        self.assertFalse(worker.full.head_comparison(reference, actual, strict=True)['passed'])
        actual = copy.deepcopy(reference); values = list(actual[-1][4]); values[1] = 1.; actual[-1][4] = values
        self.assertFalse(worker.full.head_comparison(reference, actual)['passed'])

    def test_head_reference_attempt_is_reserved_before_failure_and_not_retried(self):
        rows = worker.full.parse_trace(self.raw); accounting = worker.fresh_accounting()
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / 'started.json'
            def fail(*unused):
                self.assertTrue(receipt.is_file())
                raise ValueError('invented head failure')
            with mock.patch.object(worker.full, 'attach_head', side_effect=fail) as call:
                with self.assertRaisesRegex(ValueError, 'invented head failure'):
                    worker.attach_once(rows, None, None, accounting, receipt)
                self.assertEqual(call.call_count, 1)
            self.assertEqual(accounting['reference_head_stream_attempts'], 1)
            self.assertEqual(accounting['reference_head_stream_completions'], 0)
            self.assertEqual(accounting['reference_head_embedding_steps'], 0)
            self.assertEqual(json.loads(receipt.read_bytes())['retry_count'], 0)

    def test_nine_reference_stream_budget_has_no_tenth_call(self):
        rows = worker.full.parse_trace(self.raw); accounting = worker.fresh_accounting()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(worker.full, 'attach_head', return_value=rows) as call:
            for i in range(9):
                worker.attach_once(rows, None, None, accounting, Path(temporary) / ('started-%s.json' % i))
            with self.assertRaisesRegex(ValueError, 'NINE_HEAD_REFERENCE_STREAMS'):
                worker.attach_once(rows, None, None, accounting, Path(temporary) / 'tenth.json')
            self.assertEqual(call.call_count, 9)
            self.assertFalse((Path(temporary) / 'tenth.json').exists())
        self.assertEqual(accounting['reference_head_embedding_steps'], 234)
        self.assertEqual(accounting['reference_head_ready_outputs'], 99)
        self.assertEqual(accounting['new_onnx_session_attempts'], 0)

    def test_training_extraction_binds_raw_result_npz_and_binary_and_is_create_only(self):
        pin = enabled_fixture()['head_training']
        values = {pin['result_member']: b'{"invented":true}', pin['weights_member']: b'invented-NPZ',
                  pin['head_binary_member']: b'invented-head'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root / 'training.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name, value in values.items():
                    z.writestr(name, value)
            pin.update(zip_bytes=archive.stat().st_size, zip_sha256=worker.sha(archive.read_bytes()))
            for member, digest in (('result_member', 'result_sha256'), ('weights_member', 'weights_sha256'),
                                   ('head_binary_member', 'head_binary_sha256')):
                pin[digest] = worker.sha(values[pin[member]])
            training = worker.extract_training(archive, pin, root / 'readback')
            self.assertEqual((training / 'train/causal-head.npz').read_bytes(), b'invented-NPZ')
            with self.assertRaises(FileExistsError):
                worker.extract_training(archive, pin, root / 'readback')
            bad = {**pin, 'weights_sha256': '0' * 64}
            with self.assertRaisesRegex(ValueError, 'PINNED_TRAINING_MEMBER'):
                worker.extract_training(archive, bad, root / 'other')

    def test_training_archive_rejects_unsafe_paths_and_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'unsafe.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('../escape', b'invented')
            pin = {'zip_bytes': path.stat().st_size, 'zip_sha256': worker.sha(path.read_bytes())}
            with self.assertRaisesRegex(ValueError, 'SAFE_ARTIFACT_MEMBER'):
                worker.full.checked_archive(path, pin)
            with self.assertRaisesRegex(ValueError, 'EXACT_ARTIFACT_ZIP_BYTES'):
                worker.full.checked_archive(path, {**pin, 'zip_sha256': '0' * 64})

    def test_full_chain_build_uses_exact_candidate_and_head_without_reassociation(self):
        for backend in worker.BACKENDS:
            command = list(map(str, worker.build_command('invented-cc', backend, '/pipeline', '/generated', '/program')))
            self.assertIn('-DEF_WITH_HEAD', command)
            self.assertIn('-ffp-contract=off', command)
            self.assertIn('-fno-fast-math', command)
            self.assertIn('/pipeline/research/english_mel_specialization/fixed_frontend_mel.c', command)
            self.assertIn('/pipeline/research/english_final_embedding/english_causal_head.c', command)
            self.assertEqual('-static' in command, backend == 'arm')
            self.assertNotIn('-ffast-math', command)

    def test_failure_before_inputs_preserves_partial_receipt_and_zero_reserved_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); pins_path = root / 'pins.json'; pins_path.write_text(json.dumps(enabled_fixture()))
            args = type('Arguments', (), {'pins': pins_path, 'output': root / 'result'})()
            fake_env, _ = environment_fixture(enabled_fixture())
            with mock.patch.dict(worker.os.environ, fake_env), mock.patch.object(worker, 'execution_guard'), \
                 mock.patch.object(worker, 'read_json', return_value=enabled_fixture()), \
                 mock.patch.object(worker, 'source_preflight', side_effect=ValueError('invented source failure')):
                self.assertEqual(worker.execute(args), 2)
            failure = json.loads((args.output / 'FAILURE.json').read_bytes())
            self.assertEqual(failure['status'], 'FAILED_NO_RETRY')
            self.assertEqual(failure['stage'], 'saved-input-preflight')
            self.assertEqual(failure['accounting'], worker.fresh_accounting())
            self.assertFalse((args.output / 'RESULT.json').exists())
            manifest = json.loads((args.output / 'MANIFEST.json').read_bytes())
            self.assertEqual(set(manifest['files']), {'STARTED.json', 'EXECUTION_PINS.json', 'FAILURE.json'})

    def test_worker_cannot_call_ort_or_old_execution_path(self):
        tree = ast.parse(Path(worker.__file__).read_text())
        forbidden_calls = {'reference_trace', 'InferenceSession', 'FrozenOnnxEmbedding', 'train_cached'}
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & forbidden_calls)
        imported = {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)}
        self.assertFalse(imported & {'torch', 'onnx', 'onnxruntime'})
        self.assertEqual(worker.PLAN['separate_upstream_onnx_downloads'], 0)
        self.assertEqual(worker.PLAN['new_ort_inference_calls'], 0)
        self.assertIsNone(worker.fresh_accounting()['saved_training_archive_contains_onnx'])


if __name__ == '__main__':
    unittest.main()
