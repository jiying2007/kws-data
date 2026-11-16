"""Artificial WAV/receipt tests using the existing capture and verifier, no models."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final144_runner as stage


class ArtificialMedia:
    def _once(self, path, value):
        stage.once(path, value)

    def quality(self, samples, rate):
        return {'artificial_test_fixture': True}

    def _safe_file(self, path, limit):
        raw = stage.regular(path, limit)
        return len(raw), stage.sha(raw)

    def _write_waves(self, folder, samples, rate, technical):
        pcm = int(samples).to_bytes(2, 'little') * 1600
        wav = folder / 'audio-16k.wav'
        with wave.open(str(wav), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        native = b'ARTIFICIAL-NATIVE-' + str(samples).encode()
        (folder / 'native-f32.wav').write_bytes(native)
        return {'native_sha256': stage.sha(native), 'wav_sha256': stage.sha(wav.read_bytes()),
                'pcm_sha256': stage.sha(pcm), 'audio_frames': 1600,
                'wav_bytes': wav.stat().st_size, 'sample_rate_hz': 16000}


class ArtificialTTS:
    def __init__(self, failure=None):
        self.calls, self.failure = [], failure

    def synthesize(self, cell):
        self.calls.append(cell)
        if self.failure and len(self.calls) == 1:
            raise self.failure('ARTIFICIAL FAILURE')
        return len(self.calls), 24000


class RunnerTests(unittest.TestCase):
    def setUp(self):
        # Only an unused planner type is missing from this sparse checkout.
        stub = types.ModuleType('kws_data_contract.identity')
        def unused(*args, **kwargs):
            raise AssertionError('The generic planner must not be invoked by this test')
        stub.SourceRoles = unused
        self.modules = patch.dict(sys.modules, {'kws_data_contract.identity': stub})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.capture = importlib.import_module('english_qwen_capture')
        self.media = ArtificialMedia()
        self.media_patch = patch.object(self.capture, 'media', return_value=self.media)
        self.media_patch.start()
        self.addCleanup(self.media_patch.stop)
        self.env = patch.dict(os.environ, {'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '111',
                                          'GITHUB_RUN_ATTEMPT': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.pins = stage.loads(stage.regular(stage.recipe.PIN_FILE, 65536))
        self.lock = {'model_id': stage.recipe.MODEL, 'revision': stage.recipe.REVISION,
                     'explicitly_artificial': True}

    @contextlib.contextmanager
    def sources(self, *, failed_shard=None, lock_change=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            root.mkdir()
            backend = ArtificialTTS()
            for index in range(9):
                chosen = ArtificialTTS(RuntimeError) if index == failed_shard else backend
                lock = {**self.lock, 'different': True} if index == lock_change else self.lock
                stage.generate(root / ('shard-%02d' % index), self.pins, index,
                               runtime=(chosen, self.media, lock))
            yield root, backend

    def test_pending_lock_and_default_source_only_cannot_execute(self):
        stage.check_modules()
        with tempfile.TemporaryDirectory() as directory:
            pending = Path(directory) / 'pending-pins.json'
            stage.once(pending, {**self.pins, 'next_main_before': None})
            with patch.object(stage, 'generate') as generate, contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(stage.main([]), 0)
                self.assertEqual(json.loads(stdout.getvalue())['status'], 'SOURCE_CONTRACT_ONLY')
                with self.assertRaisesRegex(ValueError, 'MISSING_OR_INVALID_PIN:next_main_before'):
                    stage.main(['--generate', '--shard', '0', '--pins', str(pending), '--output', '/tmp/not-created'])
                generate.assert_not_called()
        self.assertEqual(self.pins['useful_scope'], 'ONE_BOUNDED_REQUESTED_INTENT_DATA_EXPERIMENT_ONLY')
        self.assertFalse(self.pins['useful_evidence']['pronunciation_pass'])

    def test_shards_are_exact_ordered_slices_with_run_bound_plans(self):
        whole = stage.recipe.recipe()
        plans = [stage.shard_plan(i, 'f' * 64, self.pins, True) for i in range(9)]
        self.assertEqual([c for p in plans for c in p['cells']], whole['cells'])
        self.assertTrue(all(len(p['cells']) == 16 and p['max_tts_calls'] == 16 for p in plans))
        self.assertTrue(all(p['global_recipe_sha256'] == stage.sha(stage.canonical(whole)) for p in plans))
        self.assertEqual(len({stage.sha(stage.canonical(p)) for p in plans}), 9)
        with patch.dict(os.environ, {'GITHUB_RUN_ID': '112'}):
            self.assertNotEqual(stage.shard_plan(0, 'f' * 64, self.pins, True), plans[0])
        for index in (-1, 9, True):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'FIXED_SHARD_INDEX'):
                stage.shard_plan(index, None, self.pins, True)

    def test_nine_shards_use_one_sixteen_call_capture_each_and_preserve_roles(self):
        with patch.object(self.capture, 'capture', wraps=self.capture.capture) as capture, self.sources() as (root, backend):
            binding, cells = stage.source_cells(root, self.pins, allow_simulated=True)
            self.assertEqual(capture.call_count, 9)
            self.assertTrue(all(c.kwargs['max_calls'] == 16 for c in capture.call_args_list))
            self.assertEqual(len(backend.calls), 144)
            self.assertEqual(len({c['cell_id'] for c in backend.calls}), 144)
            self.assertEqual(len(binding['source_shards']), 9)
            self.assertEqual(sum(r['role'] == 'train' for r, _ in cells), 96)
            self.assertEqual(sum(r['role'] == 'development' for r, _ in cells), 48)
            self.assertEqual([c['phrase'] for c in backend.calls],
                             [c['synthesis_text'] for c in stage.recipe.recipe()['cells']])
            with self.assertRaisesRegex(ValueError, 'CREATE_ONLY_PRIVATE_OUTPUT'):
                stage.generate(root / 'shard-00', self.pins, 0, runtime=(backend, self.media, self.lock))
            self.assertEqual(len(backend.calls), 144)
            self.assertTrue(all(all(r[k] is False for k in stage.recipe.FLAGS) for r, _ in cells))

    def test_missing_partial_or_simulated_shards_block_before_observer(self):
        with self.sources(failed_shard=4) as (root, backend):
            calls = []
            with self.assertRaisesRegex(ValueError, 'SOURCE_NOT_COMPLETE'):
                stage.observe('qwen', root, root.parent / 'observer', self.pins,
                              backend=lambda *a: calls.append(a))
            self.assertEqual(calls, [])
        with self.sources() as (root, _):
            with self.assertRaisesRegex(ValueError, 'SIMULATED_SOURCE_REJECTED'):
                stage.source_cells(root, self.pins)
            (root / 'shard-08').rename(root.parent / 'omitted-shard')
            with self.assertRaisesRegex(ValueError, 'EXACT_NINE_SOURCE_SHARDS'):
                stage.source_cells(root, self.pins, allow_simulated=True)

    def test_same_run_and_model_lock_identity_are_required_across_shards(self):
        with self.sources() as (root, _), patch.dict(os.environ, {'GITHUB_RUN_ID': '222'}):
            with self.assertRaisesRegex(ValueError, 'SOURCE_RUN_CODE_RECIPE_PIN_BINDING'):
                stage.source_cells(root, self.pins, allow_simulated=True)
        with self.sources(lock_change=3) as (root, _):
            with self.assertRaisesRegex(ValueError, 'SHARD_MODEL_LOCK_BYTES_DIFFER'):
                stage.source_cells(root, self.pins, allow_simulated=True)

    def test_started_only_is_unknown_and_conflicting_terminal_is_invalid(self):
        with self.sources() as (root, _):
            shard = root / 'shard-00'
            plan = stage.loads((shard / 'source-plan.json').read_bytes())
            folder = shard / 'capture/cells' / plan['cells'][0]['cell_id']
            (folder / 'completed.json').unlink()
            rows = stage.capture_rows(plan, shard / 'capture')
            self.assertEqual(rows[0]['status'], 'STARTED_UNKNOWN_NO_RETRY')
            self.assertTrue(rows[0]['attempted'])
            with self.assertRaisesRegex(ValueError, 'SOURCE_RECEIPTS_OR_AUDIO_NOT_COMPLETE_UNIQUE'):
                stage.source_cells(root, self.pins, allow_simulated=True)
        with self.sources() as (root, _):
            shard = root / 'shard-00'
            plan = stage.loads((shard / 'source-plan.json').read_bytes())
            folder = shard / 'capture/cells' / plan['cells'][0]['cell_id']
            stage.once(folder / 'failed.json', {'status': 'FAILED_NO_RETRY'})
            self.assertEqual(stage.capture_rows(plan, shard / 'capture')[0]['status'], 'INVALID_SOURCE_RECEIPT')

    def test_altered_started_and_pcm_are_not_complete(self):
        for alteration in ('started', 'pcm'):
            with self.subTest(alteration=alteration), self.sources() as (root, _):
                shard = root / 'shard-00'
                plan = stage.loads((shard / 'source-plan.json').read_bytes())
                folder = shard / 'capture/cells' / plan['cells'][0]['cell_id']
                if alteration == 'started':
                    value = stage.loads((folder / 'started.json').read_bytes())
                    value['model_calls'] = True
                    (folder / 'started.json').write_bytes(stage.canonical(value))
                else:
                    wav = folder / 'audio-16k.wav'
                    wav.write_bytes(wav.read_bytes()[:-2] + b'\x7f\x7f')
                with self.assertRaisesRegex(ValueError, 'SOURCE_RECEIPTS_OR_AUDIO_NOT_COMPLETE_UNIQUE'):
                    stage.source_cells(root, self.pins, allow_simulated=True)

    def test_observer_144_audio_only_calls_partial_failure_and_no_replay(self):
        calls = []
        def backend(wav, folder):
            self.assertEqual(wav.name, 'audio.wav')
            self.assertEqual(wav.parent, folder)
            calls.append(stage.sha(wav.read_bytes()))
            if len(calls) == 1:
                raise RuntimeError('ARTIFICIAL DECODER FAILURE')
            return 'Artificial transcript'
        with self.sources() as (root, _):
            output = root.parent / 'qwen-observations'
            result = stage.observe('qwen', root, output, self.pins, backend=backend)
            self.assertEqual((result['attempts'], result['completed']), (144, 143))
            self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
            self.assertTrue(result['test_backend'])
            self.assertEqual(len(set(calls)), 144)
            self.assertEqual(len(list(output.glob('observation-*/started.json'))), 144)
            self.assertEqual(len(list(output.glob('observation-*/failed.json'))), 1)
            self.assertEqual(list(output.rglob('*.wav')), [])
            with self.assertRaisesRegex(ValueError, 'CREATE_ONLY_PRIVATE_OUTPUT'):
                stage.observe('qwen', root, output, self.pins, backend=backend)
            self.assertEqual(len(calls), 144)

    def test_phone_output_is_xlsr_pair_and_never_admits_sources(self):
        with self.sources() as (root, _):
            result = stage.observe('phones', root, root.parent / 'phones', self.pins,
                                   backend=lambda wav, folder: (['m', 'i', 'ɹ', 'oʊ'], 1600))
            self.assertEqual((result['attempts'], result['completed']), (144, 144))
            self.assertEqual(result['status'], 'SIMULATED_COMPLETE')
            self.assertTrue(all(all(r[k] is False for k in stage.recipe.FLAGS) for r in result['rows']))
            self.assertTrue(all(r['raw_ipa_tokens'] == ['m', 'i', 'ɹ', 'oʊ'] for r in result['rows']))

    def test_four_observer_join_binds_all_rows_and_raw_summary_hashes(self):
        with self.sources() as (root, _):
            observations = root.parent / 'observations'
            observations.mkdir()
            for provider in stage.PROVIDERS:
                backend = ((lambda wav, folder: (['v', 'oʊ', 'z', 'ə', 'n'], 1600))
                           if provider == 'phones' else (lambda wav, folder: 'Artificial transcript'))
                stage.observe(provider, root, observations / ('observer-' + provider), self.pins, backend=backend)
            binding, joined = stage.review_rows(root, observations, self.pins, allow_simulated=True)
            self.assertEqual(len(joined), 144)
            self.assertTrue(binding['test_backend'])
            self.assertEqual(set(binding['observer_summaries_sha256']), set(stage.PROVIDERS))
            self.assertEqual(set(joined[0][0]['transcripts']), {'qwen', 'whisper', 'parakeet'})
            self.assertFalse(joined[0][0]['pronunciation_verified'])
            phone_path = observations / 'observer-phones/summary.json'
            self.assertEqual(binding['observer_summaries_sha256']['phones'], stage.sha(phone_path.read_bytes()))
            value = stage.loads(phone_path.read_bytes())
            value['rows'][0]['wav_sha256'] = 'f' * 64
            phone_path.write_bytes(stage.canonical(value))
            with self.assertRaisesRegex(ValueError, 'OBSERVER_SOURCE_ROW_DRIFT'):
                stage.review_rows(root, observations, self.pins, allow_simulated=True)

    def test_exact_evidence_readback_does_not_enable_pending_execution(self):
        xlsr = importlib.import_module('english_final_ipa_xlsr_once')
        old = {'status': 'FOUR_OBSERVERS_COMPLETE_REVIEW_REQUIRED', 'test_backend': False,
               'run_id': str(self.pins['pilot_run_id']), 'github_sha': self.pins['pilot_head_sha'],
               'source_plan_sha256': self.pins['pilot_source_plan_sha256'],
               'model_lock_sha256': self.pins['pilot_model_lock_sha256'], 'code_sha256': self.pins['pilot_code_sha256']}
        evidence = {'schema': xlsr.POLICY, 'run_id': str(self.pins['xlsr_run_id']), 'status': 'COMPLETE',
                    'test_backend': False, 'attempts': 8, 'completed': 8, 'model_revision': xlsr.REVISION,
                    'model_weights_sha256': xlsr.WEIGHTS_SHA, **xlsr.FLAGS,
                    'rows': [{'cell_id': r[0], 'wav_sha256': r[4], 'status': 'COMPLETE', 'test_backend': False}
                             for r in xlsr.original.ROSTER + xlsr.RENDER_ROSTER]}
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'pilot.json', Path(directory) / 'xlsr.json'
            stage.once(first, old)
            stage.once(second, evidence)
            pins = {**self.pins, 'pilot_review_json_sha256': stage.sha(first.read_bytes()),
                    'xlsr_summary_sha256': stage.sha(second.read_bytes()), 'next_main_before': None}
            checked = stage.validate_evidence(pins, first, second)
            self.assertEqual(checked['model_calls'], 0)
            self.assertFalse(checked['pronunciation_pass'])
            with self.assertRaisesRegex(ValueError, 'MISSING_OR_INVALID_PIN:next_main_before'):
                stage.recipe.validate_execution_pins(pins)
            first.write_bytes(first.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'PILOT_REVIEW_EVIDENCE'):
                stage.validate_evidence(pins, first, second)

    def test_interrupt_consumes_cell_and_preserves_partial_generator_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'shard-00'
            backend = ArtificialTTS(KeyboardInterrupt)
            with self.assertRaises(KeyboardInterrupt):
                stage.generate(output, self.pins, 0, runtime=(backend, self.media, self.lock))
            result = stage.loads((output / 'result.json').read_bytes())
            self.assertEqual((result['tts_attempts'], result['generated_audio']), (1, 0))
            self.assertEqual(result['rows'][0]['status'], 'STARTED_UNKNOWN_NO_RETRY')
            self.assertEqual(result['rows'][1]['status'], 'NOT_STARTED')
            self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')


if __name__ == '__main__':
    unittest.main()
