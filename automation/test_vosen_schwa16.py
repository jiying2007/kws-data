"""Invented PCM and receipts only; no real model, network, or execution environment."""
from __future__ import annotations
import contextlib
import copy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vosen_schwa16_once as stage


class ArtificialMedia:
    def _once(self, path, value):
        stage.helpers()['utility'].once(path, value)

    def quality(self, samples, rate):
        return {'test_backend': True, 'invented_fixture': True}

    def _safe_file(self, path, limit):
        raw = stage.helpers()['utility'].regular(path, limit)
        return len(raw), stage.sha(raw)

    def _write_waves(self, folder, samples, rate, technical):
        pcm = struct.pack('<h', samples) * 1600
        wav_path = folder / 'audio-16k.wav'
        with wave.open(str(wav_path), 'wb') as wav:
            wav.setparams((1, 2, 16000, 1600, 'NONE', 'not compressed'))
            wav.writeframes(pcm)
        native_pcm = struct.pack('<f', samples / 32768) * 1600
        native = b'RIFF' + struct.pack('<I', 36 + len(native_pcm)) + b'WAVEfmt ' + \
                 struct.pack('<IHHIIHH', 16, 3, 1, 16000, 64000, 4, 32) + b'data' + \
                 struct.pack('<I', len(native_pcm)) + native_pcm
        (folder / 'native-f32.wav').write_bytes(native)
        return {'native_sha256': stage.sha(native), 'wav_sha256': stage.sha(wav_path.read_bytes()),
            'pcm_sha256': stage.sha(pcm), 'audio_frames': 1600,
            'wav_bytes': wav_path.stat().st_size, 'sample_rate_hz': 16000}


class ArtificialTTS:
    def __init__(self, *, fail_at=None, interrupt_at=None, duplicate=False):
        self.calls = []
        self.fail_at, self.interrupt_at, self.duplicate = fail_at, interrupt_at, duplicate

    def synthesize(self, cell):
        self.calls.append(copy.deepcopy(cell))
        number = len(self.calls)
        if number == self.interrupt_at:
            raise stage.DeadlineExceeded('invented interruption')
        if number == self.fail_at:
            raise ValueError('invented source failure')
        return (1 if self.duplicate and number == 2 else number), 24000


class Schwa16Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='schwa16-invented-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.h = stage.helpers()
        self.media = ArtificialMedia()
        patcher = patch.object(self.h['capture'], 'media', return_value=self.media)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.pins = stage.pins_template()
        self.lock = {'model_id': stage.recipe.MODEL, 'revision': stage.recipe.REVISION,
                     'test_backend': True, 'not_actual_assets': True}

    def source(self, **kwargs):
        backend = ArtificialTTS(**kwargs)
        root = self.root / 'source'
        result = stage.generate(root, self.pins, runtime=(backend, self.media, self.lock))
        return root, result, backend

    def observe(self, source, providers=stage.PROVIDERS, *, failing=None):
        parent = self.root / 'observers'
        parent.mkdir(exist_ok=True)
        calls = {}
        for provider in providers:
            calls[provider] = []
            def decode(wav, folder, provider=provider):
                self.assertEqual(wav.name, 'audio.wav')
                self.assertIsInstance(wav, Path)
                self.assertIsInstance(folder, Path)
                with wave.open(str(wav), 'rb') as stream:
                    frames = stream.getnframes()
                    sample = struct.unpack('<h', stream.readframes(1))[0]
                calls[provider].append(sample)
                if provider == failing and len(calls[provider]) == 2:
                    raise ValueError('invented decoder failure')
                return (['h', 'eɪ', 'v', 'oʊ', 'z', 'ə', 'n'], frames) if provider == 'phones' else 'Hey Vosen.'
            stage.observe(provider, source, self.pins, parent / ('observer-' + provider), backend=decode)
        return parent, calls

    def test_exact_new_roster_changes_only_phonetic_instruction_and_identity(self):
        plan = stage.recipe.recipe()
        old = {c['cell_id']: c for c in stage.recipe.legacy_recipe()['cells']}
        self.assertEqual(len(plan['cells']), 16)
        self.assertEqual([c['voice_id'] for c in plan['cells']], ['aiden'] * 8 + ['eric'] * 8)
        for cell in plan['cells']:
            before = old[cell['historical_comparator_cell_id']]
            for key in before:
                if key not in ('cell_id', 'instruction'):
                    self.assertEqual(cell[key], before[key])
            self.assertIn(stage.recipe.PHONETIC_INSTRUCTION, cell['instruction'])
            self.assertEqual(cell['role'], 'train')
            self.assertEqual(cell['request_fingerprint_sha256'], stage.recipe.request_fingerprint(cell))
        self.assertTrue(all(v is False for v in stage.recipe.FLAGS.values()))

    def test_fingerprints_reject_renamed_old_request_and_cover_actual_decoding(self):
        cells = copy.deepcopy(stage.recipe.recipe()['cells'])
        old = stage.recipe.legacy_recipe()['cells'][8]
        cells[0] = {**old, 'cell_id': 'a-new-id-and-seed-do-not-make-a-new-request'}
        with self.assertRaisesRegex(ValueError, 'REPEATED_OLD_144_REQUEST'):
            stage.recipe.assert_new_requests(cells)
        self.assertEqual(stage.recipe.DECODING['language'], 'English')
        self.assertIs(stage.recipe.DECODING['do_sample'], False)
        self.assertEqual([stage.recipe.DECODING[k] for k in ('dtype', 'device_map', 'attn_implementation')],
                         ['bfloat16', 'cpu', 'sdpa'])

    def test_default_source_only_and_disabled_pins_cannot_generate(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, patch.object(stage, 'generate') as generate:
            self.assertEqual(stage.main(['--source-check', '--pins', str(self.root / 'missing.json')]), 0)
            self.assertEqual(json.loads(output.getvalue())['model_calls'], 0)
            generate.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'DISABLED_OR_UNPINNED_WORKER'):
            stage.execution_guard(self.pins, {}, {})
        self.assertFalse(any(name == 'torch' or name == 'numpy' or name.startswith('onnxruntime') for name in sys.modules))

    def test_only_first_public_main_attempt_one_with_exact_worker_is_authorized(self):
        pins = {**self.pins, 'enabled': True, 'guard_before': '1' * 40,
                'worker_sha256': stage.sha(Path(stage.__file__).read_bytes())}
        env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1',
               'GITHUB_RUN_ID': '12345', 'GITHUB_SHA': '2' * 40}
        event = {'ref': 'refs/heads/main', 'before': '1' * 40, 'after': '2' * 40,
                 'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        self.assertEqual(stage.execution_guard(pins, env, event)['run_id'], '12345')
        for changed in [{'GITHUB_ACTIONS': 'false'}, {'GITHUB_RUN_ATTEMPT': '2'}, {'GITHUB_EVENT_NAME': 'pull_request'},
                        {'GITHUB_REF': 'refs/heads/feature'}, {'GITHUB_REPOSITORY': 'other/repo'}]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'FIRST_PUBLIC_MAIN'):
                stage.execution_guard(pins, {**env, **changed}, event)
        for changed in [{'ref': 'refs/heads/feature'}, {'before': '3' * 40}, {'after': '4' * 40},
                        {'repository': {'full_name': 'jiying2007/kws-data', 'private': True}}]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'FIRST_PUBLIC_MAIN'):
                stage.execution_guard(pins, env, {**event, **changed})
        with self.assertRaisesRegex(ValueError, 'UNPINNED_WORKER'):
            stage.execution_guard({**pins, 'worker_sha256': 'a' * 64}, env, event)
        with self.assertRaisesRegex(ValueError, 'ENABLED_PINS_INCOMPLETE'):
            stage.validate_pins({**pins, 'guard_before': None})

    def test_pins_reject_budget_recipe_helper_or_admission_drift(self):
        for key, value in [('max_tts_calls', 17), ('max_calls_per_provider', 17),
                           ('recipe_sha256', 'a' * 64), ('formal_training_admitted', True),
                           ('generation_deadline_seconds', 9999), ('no_retry', False)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'FROZEN_PROFILE_PINS'):
                stage.validate_pins({**self.pins, key: value})
        pins = copy.deepcopy(self.pins)
        pins['helper_sha256']['english_qwen_capture'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'FROZEN_PROFILE_PINS'):
            stage.validate_pins(pins)

    def test_sixteen_capture_calls_and_create_only_with_intent_flags(self):
        with patch.object(self.h['capture'], 'capture', wraps=self.h['capture'].capture) as capture:
            root, result, backend = self.source()
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(capture.call_args.kwargs['max_calls'], 16)
        self.assertEqual(len(backend.calls), 16)
        self.assertEqual(result['status'], 'SIMULATED_COMPLETE')
        self.assertTrue(result['terminal'])
        source = stage.source_evidence(root, self.pins, allow_simulated=True)
        self.assertEqual(len(source['valid']), 16)
        self.assertEqual(len(source['verified_generated']), 16)
        self.assertTrue(all(row[k] is False for row in source['ledger']['rows'] for k in stage.recipe.FLAGS))
        with self.assertRaisesRegex(ValueError, 'CREATE_ONLY'):
            stage.generate(root, self.pins, runtime=(backend, self.media, self.lock))
        self.assertEqual(len(backend.calls), 16)

    def test_one_source_failure_keeps_sixteen_dispositions_and_observes_only_fifteen(self):
        root, result, backend = self.source(fail_at=3)
        self.assertEqual(len(backend.calls), 16)
        self.assertEqual((result['status'], result['tts_attempts'], result['generated_audio']),
                         ('INCOMPLETE_NO_RETRY', 16, 15))
        source = stage.source_evidence(root, self.pins, allow_simulated=True)
        self.assertEqual(source['ledger']['source_dispositions'], {stage.VALID: 15, 'FAILED_NO_RETRY': 1})
        parent, calls = self.observe(root)
        self.assertEqual([len(calls[p]) for p in stage.PROVIDERS], [15] * 4)
        joined = stage.review_rows(root, parent, self.pins, allow_simulated=True)
        self.assertEqual(len(joined['rows']), 16)
        self.assertTrue(all(v['observation_status'] == 'NOT_ELIGIBLE_SOURCE'
                            for v in joined['rows'][2]['observations'].values()))

    def test_interruption_is_started_unknown_and_not_started_not_fake_audio(self):
        root, result, backend = self.source(interrupt_at=3)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual((result['tts_attempts'], result['generated_audio'], result['capture_stages_completed']), (3, 2, 0))
        source = stage.source_evidence(root, self.pins, allow_simulated=True)
        self.assertEqual(source['ledger']['source_dispositions'],
                         {stage.VALID: 2, 'STARTED_UNKNOWN_NO_RETRY': 1, 'NOT_STARTED_NO_RETRY': 13})
        self.assertTrue(all(row['pcm_sha256'] is None for row in source['ledger']['rows'][2:]))
        parent, calls = self.observe(root)
        self.assertEqual(sum(map(len, calls.values())), 8)
        self.assertEqual(stage.combine(root, parent, self.pins, allow_simulated=True)['valid_count'], 2)

    def test_verified_duplicates_retain_pcm_identity_but_are_not_observed(self):
        root, result, _ = self.source(duplicate=True)
        source = stage.source_evidence(root, self.pins, allow_simulated=True)
        self.assertEqual((result['generated_audio'], len(source['verified_generated']), len(source['valid'])), (16, 16, 14))
        rows = source['ledger']['rows']
        self.assertEqual(rows[0]['pcm_sha256'], rows[1]['pcm_sha256'])
        self.assertTrue(all(r['source_disposition'] == 'DUPLICATE_PCM_QUARANTINED' and
                            r['source_status'] == 'GENERATED_UNADMITTED' and r['frames'] == 1600 for r in rows[:2]))
        parent, calls = self.observe(root)
        self.assertEqual(sum(map(len, calls.values())), 56)

    def test_simulated_source_cannot_enter_real_reader_or_observer(self):
        root, _, _ = self.source()
        with self.assertRaisesRegex(ValueError, 'SIMULATED_SOURCE_REJECTED'):
            stage.source_evidence(root, self.pins)
        with self.assertRaisesRegex(ValueError, 'SIMULATED_SOURCE_REJECTED'):
            stage.combine(root, self.root / 'absent', self.pins)

    def test_cross_run_observer_rejected_before_decoding(self):
        root, _, _ = self.source()
        identity = {'run_id': 'ANOTHER_TEST_RUN', 'github_sha': '0' * 40, 'run_attempt': '1', 'test_backend': True}
        calls = []
        with patch.object(stage, 'run_identity', return_value=identity), self.assertRaisesRegex(ValueError, 'SAME_RUN'):
            stage.observe('qwen', root, self.pins, self.root / 'unused', backend=lambda *args: calls.append(args))
        self.assertEqual(calls, [])
        self.assertFalse((self.root / 'unused').exists())

    def test_missing_source_terminal_or_modified_wave_cannot_be_consumed(self):
        root, _, _ = self.source()
        terminal = (root / 'result.json').read_bytes()
        (root / 'result.json').unlink()
        with self.assertRaises(ValueError):
            stage.source_evidence(root, self.pins, allow_simulated=True)
        (root / 'result.json').write_bytes(terminal)
        path = root / 'capture/cells' / stage.recipe.recipe()['cells'][0]['cell_id'] / 'audio-16k.wav'
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, 'SOURCE_RECEIPTS_CHANGED'):
            stage.source_evidence(root, self.pins, allow_simulated=True)

    def test_all_observers_audio_only_once_and_raw_join_preserved(self):
        root, _, _ = self.source()
        parent, calls = self.observe(root)
        joined = stage.review_rows(root, parent, self.pins, allow_simulated=True)
        self.assertEqual(sum(map(len, calls.values())), 64)
        self.assertTrue(all(values == list(range(1, 17)) for values in calls.values()))
        self.assertEqual(joined['rows'][0]['observations']['phones']['raw_ipa_tokens'], ['h', 'eɪ', 'v', 'oʊ', 'z', 'ə', 'n'])
        self.assertEqual(joined['rows'][0]['observations']['qwen']['text'], 'Hey Vosen.')
        self.assertEqual(set(joined['binding']['observer_summaries_sha256']), set(stage.PROVIDERS))
        with self.assertRaisesRegex(ValueError, 'CREATE_ONLY'):
            stage.observe('qwen', root, self.pins, parent / 'observer-qwen', backend=lambda *_: 'Hey Vosen.')

    def test_partial_observer_failure_retains_raw_results_without_retry(self):
        root, _, _ = self.source()
        parent, calls = self.observe(root, failing='qwen')
        result = stage.combine(root, parent, self.pins, allow_simulated=True)
        self.assertEqual(len(calls['qwen']), 16)
        self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
        self.assertEqual((result['providers']['qwen']['attempts'], result['providers']['qwen']['completed']), (16, 15))
        failed = result['rows'][1]['observations']['qwen']
        self.assertEqual((failed['observation_status'], failed['error_message']),
                         ('FAILED_NO_RETRY', 'invented decoder failure'))
        reviewed = stage.review_rows(root, parent, self.pins, allow_simulated=True)
        self.assertEqual(reviewed['status'], 'INCOMPLETE_NO_RETRY')
        self.assertEqual(sum(all(v['observation_status'] == 'COMPLETE' for v in row['observations'].values())
                             for row in reviewed['rows']), 15)

    def test_partial_provider_requires_original_model_lock(self):
        root, _, _ = self.source()
        parent, _ = self.observe(root, providers=('qwen',), failing='qwen')
        (parent / 'observer-qwen/model-lock.json').unlink()
        with self.assertRaises(ValueError):
            stage.combine(root, parent, self.pins, allow_simulated=True)

    def test_missing_observer_terminal_or_entire_provider_remains_unknown(self):
        root, _, _ = self.source()
        parent, _ = self.observe(root, providers=('qwen',))
        (parent / 'observer-qwen/summary.json').unlink()
        result = stage.combine(root, parent, self.pins, allow_simulated=True)
        self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
        self.assertEqual(result['providers']['qwen']['status'], 'UNKNOWN_MISSING_PROVIDER_TERMINAL')
        self.assertEqual(result['providers']['qwen']['attempts'], 16)
        self.assertIsNone(result['providers']['phones']['attempts'])
        self.assertEqual(result['rows'][0]['observations']['phones']['observation_status'], 'UNKNOWN_MISSING_PROVIDER_OUTPUT')

    def test_observer_receipt_pcm_binding_and_cross_run_summary_cannot_drift(self):
        root, _, _ = self.source()
        parent, _ = self.observe(root, providers=('qwen',))
        path = parent / 'observer-qwen/started.json'
        original = path.read_bytes()
        value = json.loads(original)
        value['observer_identity']['run_id'] = 'OTHER_TEST_RUN'
        path.write_bytes(json.dumps(value).encode())
        with self.assertRaisesRegex(ValueError, 'CROSS_RUN_OR_OBSERVER_BINDING'):
            stage.combine(root, parent, self.pins, allow_simulated=True)
        path.write_bytes(original)
        path = parent / 'observer-qwen/cells' / stage.recipe.recipe()['cells'][0]['cell_id'] / 'completed.json'
        value = json.loads(path.read_bytes())
        value['pcm_sha256'] = '0' * 64
        path.write_bytes(json.dumps(value).encode())
        with self.assertRaisesRegex(ValueError, 'OBSERVER_TERMINAL_BINDING'):
            stage.combine(root, parent, self.pins, allow_simulated=True)

    def test_deadline_handler_restores_process_signal_state(self):
        before = stage.signal.getsignal(stage.signal.SIGALRM)
        with self.assertRaises(stage.DeadlineExceeded):
            with stage.deadline(30):
                stage.signal.getsignal(stage.signal.SIGALRM)(stage.signal.SIGALRM, None)
        self.assertEqual(stage.signal.getsignal(stage.signal.SIGALRM), before)
        self.assertEqual(stage.signal.getitimer(stage.signal.ITIMER_REAL), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
