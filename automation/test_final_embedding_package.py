"""Artificial source fixtures only: no real ONNX, training, or audio inference."""
from __future__ import annotations
from contextlib import ExitStack
import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer = load('fixture_infer', HERE / 'english_final_embedding_infer.py')
package = load('fixture_package', HERE / 'package_final_embedding.py')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(infer.canonical(value) + b'\n')


def fake_bundle(root):
    """Tests replace only immutable asset constants with tiny invented bytes."""
    root.mkdir()
    names = {'english_final_embedding_infer.py', 'requirements.txt', 'MODEL_CARD.md',
             'models/causal-head.npz', 'models/causal-head.bin',
             'licenses/openWakeWord-LICENSE', 'licenses/kws-pipeline-LICENSE',
             'evidence/RESULT.json', 'evidence/TRAIN_RESULT.json', 'evidence/PARITY.json'}
    names.update('runtime/' + name for name in infer.RUNTIME_FILES)
    names.update('models/' + name for name in infer.ASSETS)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('ARTIFICIAL SOURCE FIXTURE ' + name).encode())
    files = {name: infer.file_identity(root / name) for name in names}
    manifest = {'schema': infer.SCHEMA, 'status': 'COMPLETE_RESEARCH_BUNDLE', 'test_backend': False,
                'policy': infer.POLICY, 'files': files, **dict.fromkeys(infer.FALSE_FLAGS, False),
                'provenance': {'upstream_commit': infer.UPSTREAM_COMMIT,
                    'weights_sha256': files['models/causal-head.npz']['sha256'],
                    'head_binary_sha256': files['models/causal-head.bin']['sha256']}}
    put(root / 'manifest.json', manifest)
    assets = {name: (files['models/' + name]['bytes'], files['models/' + name]['sha256']) for name in infer.ASSETS}
    license_id = files['licenses/openWakeWord-LICENSE']
    stack = ExitStack()
    stack.enter_context(patch.object(infer, 'ASSETS', assets))
    stack.enter_context(patch.object(infer, 'UPSTREAM_LICENSE', (license_id['bytes'], license_id['sha256'])))
    return manifest, stack


def fake_training(root):
    """Invented report/byte identities exercise guards, never fit or load a model."""
    root.mkdir()
    (root / 'train').mkdir()
    (root / 'train/causal-head.npz').write_bytes(b'ARTIFICIAL NPZ PLACEHOLDER NOT A MODEL')
    binary = b'EHKWS01\0' + bytes(197900)
    (root / 'causal-head.bin').write_bytes(binary)
    stream = b'EHSTR01\0' + struct.pack('<I', 16) + bytes(396 * 16)
    report = {'schema': 'english-causal-head-c-parity-v1', 'head_binary_sha256': infer.sha(binary),
              'stream_binary_sha256': infer.sha(stream), 'input_embeddings': 16, 'head_outputs': 1,
              'max_abs_logit_error': 0.0, 'max_abs_probability_error': 0.0, 'events': [],
              'threshold': 0.8, 'confirm_embeddings': 3, 'cooldown_embeddings': 13,
              'formal_qualification': False, 'ssc305_qualification': False}
    for name in ('host-parity', 'arm-parity'):
        (root / name).mkdir()
        (root / name / 'head.bin').write_bytes(binary)
        (root / name / 'stream.bin').write_bytes(stream)
    (root / 'actual-embedding-streams.npz').write_bytes(b'ARTIFICIAL EMBEDDINGS PLACEHOLDER')
    docs = {name: {} for name in package.EVIDENCE}
    started = {'run_id': '123', 'github_sha': 'a' * 40, 'test_backend': False,
               'worker_code_sha256': 'b' * 64, 'training_pins_sha256': 'c' * 64,
               **dict.fromkeys(infer.FALSE_FLAGS, False)}
    binding = {'run_id': '123', 'github_sha': 'a' * 40, 'test_backend': False,
               'source_manifest_sha256': 'd' * 64, 'recipe_sha256': 'e' * 64}
    selected = {'observer_artifact_manifest_sha256': 'f' * 64, 'ARTIFICIAL_FIXTURE': True}
    preflight = {'geometry': {'status': 'FIXED_ONNX_CONV_GEOMETRY_VERIFIED'},
                 'pipeline_source': {'commit': '1' * 40},
                 'asset_identities': {name: list(value) for name, value in infer.ASSETS.items()}}
    docs.update({'STARTED.json': started, 'SOURCE_BINDING.json': binding, 'SELECTION.json': selected,
                 'FEATURE_PREFLIGHT.json': preflight, 'FEATURE_EVIDENCE.json': [],
                 'MINIMUM_AFTER_DEDUPLICATION.json': {'passed': True},
                 'MINIMUM_AFTER_FEATURE_GEOMETRY.json': {'passed': True},
                 'PARITY.json': {'host-parity': report, 'arm-parity': copy.deepcopy(report)}})
    bound = {'source_manifest_sha256': binding['source_manifest_sha256'],
             'observation_manifest_sha256': selected['observer_artifact_manifest_sha256'],
             'recipe_sha256': binding['recipe_sha256'], 'selection_sha256': infer.sha(infer.canonical(selected)),
             'pipeline_source_sha256': infer.sha(infer.canonical(preflight['pipeline_source'])),
             'feature_evidence_sha256': infer.sha(infer.canonical([]) + b'\n'),
             'feature_stream_sha256': infer.file_identity(root / 'actual-embedding-streams.npz')['sha256'],
             'worker_code_sha256': started['worker_code_sha256']}
    weights_sha = infer.file_identity(root / 'train/causal-head.npz')['sha256']
    result = {**started, 'status': 'EXECUTED_WEAK_RESEARCH_WITH_HOST_AND_ARM_HEAD_PARITY',
              'optimizer_updates': 96, 'weights_sha256': weights_sha, 'head_binary_sha256': infer.sha(binary),
              'source_denominator': 144, 'minimum': {'passed': True}, 'input_binding': bound,
              'arm_scope': 'SAVED_CAUSAL_HEAD_ONLY_UNDER_QEMU'}
    result.update(dict.fromkeys(('independent_frr', 'independent_far', 'word_end_latency',
                                'ssc305', 'deployment_frontend_arm_parity'), 'NOT_RUN'))
    trained = {'status': 'EXECUTED_WEAK_RESEARCH_ONLY', 'optimizer_updates': 96, 'losses': [1.0] * 96,
               'optimizer_bag_draws': [['fixture']] * 96, 'weights_sha256': weights_sha,
               'input_binding': bound, 'saved_replay': [{'max_abs_logit_error': 0.0}],
               'spec': {'history': 16, 'embedding_width': 96, 'hidden': 32, 'classes': 3,
                        'seed': 20261116, 'optimizer_updates': 96, 'checkpoint_selection': False},
               **dict.fromkeys(infer.FALSE_FLAGS, False)}
    docs['RESULT.json'], docs['TRAINING_STARTED.json'] = result, {'input_binding': bound}
    for name, document in docs.items():
        put(root / name, document)
    put(root / 'train/RESULT.json', trained)
    return docs


class InventedFeatures:
    """Only a callback clock fixture, not an ONNX or acoustic implementation."""
    def __init__(self):
        self.observed, self.next_end, self.finished = 0, 16000, False

    def accept(self, samples):
        self.observed += len(samples)
        rows = []
        while self.next_end <= self.observed:
            rows.append({'available_sample': self.next_end, 'embedding': (0.0,) * 96})
            self.next_end += 1280
        return rows

    def finish(self):
        self.finished = True
        return []


class PackageTests(unittest.TestCase):
    def test_defaults_are_standard_library_only(self):
        for name in ('english_final_embedding_infer.py', 'package_final_embedding.py'):
            code = ('import importlib.util,sys;'
                    f's=importlib.util.spec_from_file_location("fixture",{str(HERE / name)!r});'
                    'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main([]);'
                    'sys.exit(any(k in sys.modules for k in ("numpy","onnx","onnxruntime","torch")))')
            run = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)['model_calls'], 0)

    def test_complete_file_identities_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bundle'
            manifest, stack = fake_bundle(root)
            with stack:
                self.assertEqual(infer.verify_bundle(root), manifest)
                (root / 'models/causal-head.npz').write_bytes(b'CHANGED')
                with self.assertRaisesRegex(ValueError, 'BUNDLE_FILE_IDENTITY'):
                    infer.verify_bundle(root)

    def test_manifest_cannot_change_policy_or_admit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bundle'
            manifest, stack = fake_bundle(root)
            with stack:
                for key, value in (('test_backend', True), ('qualification_allowed', True),
                                   ('policy', {**infer.POLICY, 'threshold': 0.7})):
                    put(root / 'manifest.json', {**manifest, key: value})
                    with self.assertRaisesRegex(ValueError, 'FIXED_RESEARCH_BUNDLE_CONTRACT'):
                        infer.verify_bundle(root)

    def test_unlisted_file_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bundle'
            manifest, stack = fake_bundle(root)
            with stack:
                (root / 'extra').write_bytes(b'not listed')
                with self.assertRaisesRegex(ValueError, 'UNLISTED_BUNDLE_FILE'):
                    infer.verify_bundle(root)
                (root / 'extra').unlink()
                put(root / 'manifest.json', {**manifest, 'files': {**manifest['files'],
                    '../escape': {'bytes': 0, 'sha256': '0' * 64}}})
                with self.assertRaisesRegex(ValueError, 'MANIFEST_PATH'):
                    infer.verify_bundle(root)

    def test_symlink_and_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bundle'
            _, stack = fake_bundle(root)
            with stack:
                (root / 'unexpected-link').symlink_to(root / 'requirements.txt')
                with self.assertRaisesRegex(ValueError, 'BUNDLE_SYMLINK'):
                    infer.verify_bundle(root)
            path = Path(tmp) / 'duplicate.json'
            path.write_text('{"status":1,"status":2}')
            with self.assertRaisesRegex(ValueError, 'DUPLICATE_JSON_KEY'):
                infer.read_json(path)

    def test_training_requires_all_actual_weight_and_stream_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'training'
            fake_training(root)
            package.training_evidence(root)
            (root / 'arm-parity/stream.bin').write_bytes(b'CHANGED')
            with self.assertRaisesRegex(ValueError, 'ACTUAL_REPLAY_STREAM_IDENTITY'):
                package.training_evidence(root)

    def test_partial_training_simulation_and_failure_cannot_be_packaged(self):
        for change in ('partial', 'simulated', 'failed', 'weight'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'training'
                docs = fake_training(root)
                if change == 'partial':
                    docs['RESULT.json']['optimizer_updates'] = 95
                elif change == 'simulated':
                    docs['RESULT.json']['test_backend'] = True
                elif change == 'failed':
                    put(root / 'FAILURE.json', {'status': 'FAILED'})
                else:
                    (root / 'train/causal-head.npz').write_bytes(b'CHANGED')
                put(root / 'RESULT.json', docs['RESULT.json'])
                with self.assertRaises(ValueError):
                    package.training_evidence(root)

    def test_arm_events_and_fixed_threshold_must_match(self):
        for key, value in (('events', [{'sample': 35200, 'keyword': 1}]), ('threshold', 0.7)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'training'
                docs = fake_training(root)
                docs['PARITY.json']['arm-parity'][key] = value
                put(root / 'PARITY.json', docs['PARITY.json'])
                with self.assertRaises(ValueError):
                    package.training_evidence(root)

    def test_startup_observed_wav_eof_and_refractory(self):
        for frames in (15999, 16000, 35199, 35200, 37759, 37760, 56961):
            raw = struct.pack('<%dh' % frames, *(index % 100 for index in range(frames)))
            memory = io.BytesIO()
            with wave.open(memory, 'wb') as target:
                target.setparams((1, 2, 16000, frames, 'NONE', 'not compressed'))
                target.writeframes(raw)
            memory.seek(0)
            features, trace = InventedFeatures(), []
            head = infer.HistoryHead(lambda _: ([0.0, 1.0, 0.0], [0.05, 0.9, 0.05]))
            with wave.open(memory, 'rb') as source:
                result = infer.consume_wav(source, features, head, trace.append)
            self.assertEqual(result['observed_samples'], frames)
            self.assertEqual(result['pcm_sha256'], infer.sha(raw))
            self.assertTrue(features.finished)
            expected = list(range(16000, frames + 1, 1280))
            self.assertEqual([row['sample'] for row in trace], expected)
            self.assertEqual([row['sample'] for row in trace if row['ready']], [x for x in expected if x >= 35200])
            self.assertEqual([event['sample'] for event in result['events']], [x for x in (37760, 56960) if x <= frames])
            self.assertLessEqual(len(head.history), 16)
            self.assertEqual(result['eof']['nontrigger_status'], 'CENSORED_NOT_FRR')
            self.assertEqual(result['eof']['padding_samples'], 0)

    def test_threshold_rounding_class_change_and_background_clear_confirmations(self):
        policy = infer.FixedEvents()
        threshold = infer.THRESHOLD_F32
        self.assertEqual(policy.step([0.1, threshold, 0.05], 35200), 0)
        self.assertEqual(policy.step([0.1, threshold - 1e-8, 0.05], 36480), 0)
        self.assertEqual(policy.step([0.1, threshold, 0.05], 37760), 0)
        self.assertEqual(policy.step([0.1, 0.05, threshold], 39040), 0)
        self.assertEqual(policy.step([0.1, 0.05, threshold], 40320), 0)
        self.assertEqual(policy.step([0.9, 0.05, 0.05], 41600), 0)
        self.assertEqual(policy.step([0.1, 0.05, threshold], 42880), 0)
        self.assertEqual(policy.step([0.1, 0.05, threshold], 44160), 0)
        self.assertEqual(policy.step([0.1, 0.05, threshold], 45440), 2)

    def test_wrong_wav_geometry_rejected_before_callbacks(self):
        memory = io.BytesIO()
        with wave.open(memory, 'wb') as target:
            target.setparams((2, 2, 16000, 1, 'NONE', 'not compressed'))
            target.writeframes(bytes(4))
        memory.seek(0)
        features = InventedFeatures()
        with wave.open(memory, 'rb') as source, self.assertRaisesRegex(ValueError, 'WAV_MUST_BE_MONO'):
            infer.consume_wav(source, features, None, None)
        self.assertEqual(features.observed, 0)

    def test_parent_relative_output_is_outside_bundle_and_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / 'bundle'
            bundle.mkdir()
            output = infer.create_output_directory(bundle, bundle / '../result')
            self.assertEqual(output, Path(tmp) / 'result')
            for requested in (output, bundle / 'result'):
                with self.assertRaisesRegex(ValueError, 'NEW_OUTPUT_DIRECTORY_OUTSIDE_BUNDLE'):
                    infer.create_output_directory(bundle, requested)

    def test_numpy_math_matches_frozen_reference_on_invented_arrays(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('optional invented-array check requires NumPy; stdlib fixtures remain active')
        paths = (HERE / 'portable_head.py',
                 HERE.parent / 'pipeline-source/research/english_final_embedding/portable_head.py',
                 HERE.parent / 'kws-pipeline/research/english_final_embedding/portable_head.py')
        path = next((path for path in paths if path.is_file()), None)
        if path is None:
            self.skipTest('fixed pipeline source is not present in this source-only checkout')
        portable = load('fixture_portable', path)
        random = np.random.default_rng(119)
        weights = {name: random.normal(0, 0.02, shape).astype(np.float32) for name, shape in portable.SHAPES.items()}
        weights['scale96'] = np.ones(96, dtype=np.float32)
        weights['b2'] = np.asarray([0, 7, 0], dtype=np.float32)
        embeddings = random.normal(0, 1, (80, 96)).astype(np.float32)
        samples = [16000 + 1280 * i for i in range(len(embeddings))]
        expected = portable.reference(weights, embeddings, samples)
        state = infer.HistoryHead(infer.numpy_scorer(weights))
        actual = [state.step(row.tolist(), sample) for row, sample in zip(embeddings, samples)]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    unittest.main()
