"""Offline artificial-byte tests; never execute ASR, TTS, phoneme models or network."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_ipa_qwen17_final_ipa_probe_once as probe


def raw_json(value):
    return json.dumps(value, ensure_ascii=False).encode()


def fixture_members():
    prefix = 'en-ipa-probe/'
    lock = raw_json({'model_id': probe.MODEL, 'revision': probe.REV})
    lock_sha = probe.sha(lock)
    members = {prefix + 'model-lock.json': lock,
               prefix + 'started.json': raw_json({'run_id': str(probe.SOURCE_RUN),
                                                  'source_sha': probe.SOURCE_HEAD}),
               'en-ipa-pip-freeze.txt': b'INVENTED FIXTURE, NO MODEL EXECUTION\n'}
    roster, cells, rows = [], [], []
    for i, original in enumerate(probe.ROSTER):
        cid, phrase, speaker = original[:3]
        pcm = bytes((i + 1, 0)) * 80
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        wav, native = buffer.getvalue(), b'artificial-native-' + str(i).encode()
        wav_sha, pcm_sha = probe.sha(wav), probe.sha(pcm)
        roster.append((cid, phrase, speaker, 80, wav_sha, pcm_sha))
        cells.append({'cell_id': cid, 'phrase': phrase, 'speaker': speaker, 'ipa': probe.IPAS[phrase]})
        rows.append({'cell_id': cid, 'status': 'GENERATED_UNADMITTED',
                     'wav_sha256': wav_sha, 'pcm_sha256': pcm_sha})
        folder = prefix + 'cells/' + cid + '/'
        members[folder + 'audio-16k.wav'] = wav
        members[folder + 'native-f32.wav'] = native
        members[folder + 'started.json'] = raw_json({'cell_id': cid})
        members[folder + 'completed.json'] = raw_json({
            'cell_id': cid, 'phrase': phrase, 'speaker': speaker, 'ipa_target': probe.IPAS[phrase],
            **probe.FALSE_FLAGS, 'audio': {'wav_sha256': wav_sha, 'pcm_sha256': pcm_sha,
              'native_sha256': probe.sha(native), 'wav_bytes': len(wav), 'audio_frames': 80}})
    members[prefix + 'source-plan.json'] = raw_json({
        'model': probe.MODEL, 'revision': probe.REV, 'model_lock_sha256': lock_sha,
        'cells': cells, **probe.FALSE_FLAGS})
    members[prefix + 'result.json'] = raw_json({
        'schema': probe.POLICY, 'status': 'COMPLETE', 'attempts': 4, 'completed': 4,
        'asr_calls': 0, 'model_lock_sha256': lock_sha, 'rows': rows, **probe.FALSE_FLAGS})
    return members, roster, lock_sha


@contextlib.contextmanager
def fixture(change=None):
    members, roster, lock_sha = fixture_members()
    if change:
        change(members)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'source.zip'
        with zipfile.ZipFile(path, 'w') as z:
            for name, raw in members.items():
                z.writestr(name, raw)
        with patch.object(probe, 'SOURCE_SHA', probe.sha(path.read_bytes())), \
             patch.object(probe, 'LOCK_SHA', lock_sha), patch.object(probe, 'ROSTER', tuple(roster)):
            yield path


def mutate_json(members, name, modify):
    value = json.loads(members[name])
    modify(value)
    members[name] = raw_json(value)


def summaries(root):
    for provider in probe.PROVIDERS:
        folder = root / provider
        folder.mkdir()
        value = {'schema': probe.ASR_POLICY, 'status': 'COMPLETE', 'provider': provider,
                 'source_run_id': probe.SOURCE_RUN, 'source_artifact_id': probe.SOURCE_ARTIFACT,
                 'source_zip_sha256': probe.SOURCE_SHA, 'asr_attempts': 4, 'completed_asr': 4,
                 'test_backend': False, **probe.FALSE_FLAGS, 'rows': [
            {'cell_id': r[0], 'wav_sha256': r[4], 'frames': r[3], 'status': 'COMPLETE',
             'text': 'invented test transcript', 'decoder_completed': True} for r in probe.ROSTER]}
        (folder / 'summary.json').write_bytes(raw_json(value))


class ProbeTests(unittest.TestCase):
    def test_fixed_readings_and_four_cells(self):
        value = probe.read_trigger()
        self.assertEqual(len(value['cells']), 4)
        self.assertEqual({c['ipa'] for c in value['cells']}, set(probe.IPAS.values()))
        self.assertFalse(value['training_admitted'])
        self.assertEqual(value['asr_calls'], 0)

    def test_exact_artifact_and_new_source_identity(self):
        self.assertEqual(probe.SOURCE_RUN, 43793684582)
        self.assertEqual(probe.SOURCE_ARTIFACT, 14193452864)
        self.assertEqual(probe.SOURCE_SHA, 'e737354ad8ed69d511f061781f5b6cc3c86ff1ea70ba9c72084c5ac4cd1c738f')
        self.assertEqual(len({r[4] for r in probe.ROSTER}), 4)

    def test_complete_fixture(self):
        with fixture() as path:
            cells = probe.source_cells(path)
            self.assertEqual(len(cells), 4)
            self.assertTrue(all(c[2] == 80 for c in cells))

    def test_outer_sha_rejects_corruption(self):
        with fixture() as path:
            path.write_bytes(path.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'SOURCE_ZIP_SHA'):
                probe.source_cells(path)

    def test_extra_member_rejected_even_with_updated_outer_sha(self):
        with fixture(lambda m: m.update({'../escape': b'x'})) as path:
            with self.assertRaisesRegex(ValueError, 'SOURCE_MEMBER_SET'):
                probe.source_cells(path)

    def test_missing_member_rejected(self):
        with fixture(lambda m: m.pop('en-ipa-pip-freeze.txt')) as path:
            with self.assertRaisesRegex(ValueError, 'SOURCE_MEMBER_SET'):
                probe.source_cells(path)

    def test_pcm_changed_under_same_receipt_rejected(self):
        def change(m):
            name = next(n for n in m if n.endswith('audio-16k.wav'))
            m[name] = m[name][:-2] + b'\x7f\x7f'
        with fixture(change) as path:
            with self.assertRaisesRegex(ValueError, 'SOURCE_WAV_PCM_BINDING'):
                probe.source_cells(path)

    def test_promoted_source_rejected(self):
        def change(m):
            mutate_json(m, 'en-ipa-probe/result.json', lambda x: x.update(training_admitted=True))
        with fixture(change) as path:
            with self.assertRaisesRegex(ValueError, 'SOURCE_TERMINAL_STATUS'):
                probe.source_cells(path)

    def test_old_ipa_rejected(self):
        def change(m):
            mutate_json(m, 'en-ipa-probe/source-plan.json',
                        lambda x: x['cells'][0].update(ipa='/heɪ ˈmɪəroʊ/'))
        with fixture(change) as path:
            with self.assertRaisesRegex(ValueError, 'SOURCE_CELL_RECEIPT'):
                probe.source_cells(path)

    def test_duplicate_cell_rejected(self):
        def change(m):
            mutate_json(m, 'en-ipa-probe/source-plan.json',
                        lambda x: x['cells'].__setitem__(1, x['cells'][0]))
        with fixture(change) as path:
            with self.assertRaisesRegex(ValueError, 'SOURCE_DUPLICATE_CELL'):
                probe.source_cells(path)

    def test_symlink_input_rejected(self):
        with fixture() as path:
            link = path.parent / 'link.zip'
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'SYMLINK_PATH'):
                probe.source_cells(link)

    def test_duplicate_json_rejected(self):
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_JSON_KEY'):
            probe.loads('{"x": 1, "x": 2}')

    def test_tts_cannot_be_reexecuted(self):
        with self.assertRaisesRegex(ValueError, 'TTS_ALREADY_CONSUMED'):
            probe.main(['--execute', '--output', '/tmp/must-not-be-created'])

    def test_four_blind_calls_no_target_arguments(self):
        seen = []
        def backend(wav, folder):
            self.assertEqual(wav.name, 'audio.wav')
            self.assertEqual(wav.parent, folder)
            seen.append(probe.sha(wav.read_bytes()))
            return 'machine observation'
        with fixture() as path:
            output = path.parent / 'result'
            result = probe.capture('qwen', path, output, backend=backend)
            self.assertEqual(result['asr_attempts'], 4)
            self.assertEqual(result['completed_asr'], 4)
            self.assertTrue(result['test_backend'])
            self.assertFalse(result['training_admitted'])
            self.assertEqual(seen, [r[4] for r in probe.ROSTER])
            self.assertFalse(list(output.rglob('*.wav')))
            with self.assertRaisesRegex(ValueError, 'CREATE_ONLY_OUTPUT'):
                probe.capture('qwen', path, output, backend=backend)

    def test_failure_is_counted_without_retry(self):
        calls = []
        def backend(wav, folder):
            calls.append(str(wav))
            if len(calls) == 1:
                raise RuntimeError('fixture error')
            return 'ok'
        with fixture() as path:
            result = probe.capture('whisper', path, path.parent / 'out', backend=backend)
            self.assertEqual(len(calls), 4)
            self.assertEqual(result['completed_asr'], 3)
            self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')

    def test_partial_cli_has_nonzero_exit(self):
        with patch.object(probe, 'capture', return_value={'status': 'INCOMPLETE_NO_RETRY'}):
            self.assertEqual(probe.main(['--provider', 'qwen', '--source-zip', '/tmp/source',
                                         '--output', '/tmp/result', '--execute-asr']), 2)

    def test_once_guard_requires_exact_main_successor(self):
        env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'a' * 40}
        event = {'before': probe.SOURCE_HEAD, 'after': 'a' * 40,
                 'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        probe.execution_guard(event, env)
        for key, value in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_REF', 'refs/heads/other'),
                           ('GITHUB_EVENT_NAME', 'pull_request'), ('GITHUB_REPOSITORY', 'fork/repo')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.execution_guard(event, {**env, key: value})
        for changed in [{**event, 'before': '0' * 40},
                        {**event, 'repository': {'full_name': 'jiying2007/kws-data', 'private': True}}]:
            with self.assertRaises(ValueError):
                probe.execution_guard(changed, env)

    def test_combination_preserves_unadmitted_status(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            summaries(root)
            result = probe.combine(root, root / 'combined.json')
            self.assertEqual(result['asr_calls'], 12)
            self.assertFalse(result['training_admitted'])
            self.assertTrue(all(c['acoustic_review'] == 'PENDING' for c in result['cells']))

    def test_combination_rejects_fake_or_wrong_source(self):
        for change in [{'test_backend': True}, {'source_zip_sha256': '0' * 64},
                       {'provider': 'whisper'}, {'completed_asr': 3}, {'training_admitted': True}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                summaries(root)
                path = root / 'qwen/summary.json'
                value = json.loads(path.read_bytes())
                value.update(change)
                path.write_bytes(raw_json(value))
                with self.assertRaisesRegex(ValueError, 'ASR_SUMMARY_IDENTITY'):
                    probe.combine(root, root / 'combined.json')

    def test_workflow_has_no_tts_or_schedules(self):
        text = (probe.HERE.parent / '.github/workflows/english-ipa-qwen17-final-ipa-probe-once.yml').read_text()
        self.assertNotIn('generate-pronunciation-probes:', text)
        self.assertNotIn('schedule:', text)
        self.assertNotIn('workflow_dispatch:', text)
        self.assertNotIn('qwen-tts==', text)
        self.assertIn('max-parallel: 2', text)
        self.assertIn("github.event.before == '" + probe.SOURCE_HEAD + "'", text)
        self.assertIn('needs: blind-asr', text)
        self.assertIn('github.run_attempt == 1', text)

    def test_phone_guard_is_a_distinct_exact_public_main_successor(self):
        env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'a' * 40}
        event = {'before': probe.PHONE_PREDECESSOR, 'after': 'a' * 40,
                 'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        self.assertEqual(probe.PHONE_PREDECESSOR, 'e844bfbff4d579397eb93de0f2e27bf38fe1af8a')
        probe.phone_execution_guard(event, env)
        for key, value in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_REF', 'refs/heads/other'),
                           ('GITHUB_EVENT_NAME', 'pull_request'), ('GITHUB_REPOSITORY', 'fork/repo'),
                           ('GITHUB_SHA', 'b' * 40)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'PHONE_ONE_SUCCESSOR'):
                probe.phone_execution_guard(event, {**env, key: value})
        for change in [{'before': probe.SOURCE_HEAD}, {'before': '0' * 40}, {'after': None},
                       {'repository': {'full_name': 'jiying2007/kws-data', 'private': True}}]:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'PHONE_ONE_SUCCESSOR'):
                probe.phone_execution_guard({**event, **change}, env)
        with self.assertRaisesRegex(ValueError, 'ONE_SUCCESSOR_PUSH_ONLY'):
            probe.execution_guard(event, env)

    def test_four_phone_calls_use_only_original_anonymous_wavs(self):
        seen = []
        def backend(wav):
            self.assertEqual(wav.name, 'audio.wav')
            self.assertRegex(wav.parent.name, r'^observation-0[0-3]$')
            seen.append(probe.sha(wav.read_bytes()))
            with wave.open(str(wav), 'rb') as src:
                frames = src.getnframes()
            return ['h', 'ˈeɪ'], ['h', 'eɪ'], frames
        with fixture() as path:
            output = path.parent / 'phones'
            result = probe.capture_phones(path, output, backend=backend)
            self.assertEqual(result['status'], 'SIMULATED_COMPLETE')
            self.assertEqual((result['phoneme_attempts'], result['completed_phonemes']), (4, 4))
            self.assertEqual(seen, [r[4] for r in probe.ROSTER])
            self.assertTrue(result['test_backend'])
            self.assertTrue(all(result[k] is False for k in probe.FALSE_FLAGS))
            self.assertEqual(result['new_tts_calls'], 0)
            self.assertEqual(result['new_asr_calls'], 0)
            self.assertEqual(result['new_optimizer_steps'], 0)
            self.assertFalse(list(output.rglob('*.wav')))
            self.assertEqual(len(list(output.glob('observation-*/completed.json'))), 4)
            self.assertEqual(len(list(output.glob('observation-*/started.json'))), 4)
            self.assertEqual(json.loads((output / 'summary.json').read_bytes()), result)
            with self.assertRaisesRegex(ValueError, 'CREATE_ONLY_OUTPUT'):
                probe.capture_phones(path, output, backend=backend)
            self.assertEqual(len(seen), 4)

    def test_phone_partial_failure_retains_receipts_without_retry(self):
        calls = []
        def backend(wav):
            calls.append(wav)
            if len(calls) == 2:
                raise RuntimeError('invented fixture failure')
            return ['h', 'eɪ'], ['h', 'eɪ'], 80
        with fixture() as path:
            output = path.parent / 'phones'
            result = probe.capture_phones(path, output, backend=backend)
            self.assertEqual(len(calls), 4)
            self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
            self.assertEqual((result['phoneme_attempts'], result['completed_phonemes']), (4, 3))
            self.assertEqual(len(list(output.glob('observation-*/failed.json'))), 1)
            self.assertEqual(result['rows'][1]['error_type'], 'RuntimeError')
            self.assertFalse(list(output.rglob('*.wav')))

    def test_phone_interruption_records_unknown_completion_and_stops(self):
        calls = []
        def backend(wav):
            calls.append(wav)
            raise KeyboardInterrupt()
        with fixture() as path:
            output = path.parent / 'phones'
            with self.assertRaises(KeyboardInterrupt):
                probe.capture_phones(path, output, backend=backend)
            result = json.loads((output / 'summary.json').read_bytes())
            self.assertEqual((result['phoneme_attempts'], result['completed_phonemes']), (1, 0))
            self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
            self.assertEqual(result['error_type'], 'KeyboardInterrupt')
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(list(output.glob('observation-*/failed.json'))), 1)

    def test_phone_invalid_frame_or_tokens_do_not_count_as_completion(self):
        for result in [(['h', 'eɪ'], ['h', 'eɪ'], 79),
                       (['h', '\n'], ['h', 'eɪ'], 80),
                       ('not a token list', ['h', 'eɪ'], 80)]:
            with self.subTest(result=result), fixture() as path:
                observed = probe.capture_phones(path, path.parent / 'phones', backend=lambda _: result)
                self.assertEqual(observed['phoneme_attempts'], 4)
                self.assertEqual(observed['completed_phonemes'], 0)
                self.assertEqual(observed['status'], 'INCOMPLETE_NO_RETRY')

    def test_phone_bad_source_and_budget_rejected_before_backend(self):
        calls = []
        with fixture() as path:
            path.write_bytes(path.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'SOURCE_ZIP_SHA'):
                probe.capture_phones(path, path.parent / 'phones', backend=lambda p: calls.append(p))
            self.assertEqual(calls, [])
            self.assertFalse((path.parent / 'phones').exists())
        with fixture() as path:
            cells = probe.source_cells(path)
            with patch.object(probe, 'source_cells', return_value=cells + cells[:1]):
                with self.assertRaisesRegex(ValueError, 'PHONE_SOURCE_COUNT'):
                    probe.capture_phones(path, path.parent / 'phones', backend=lambda p: calls.append(p))
            self.assertEqual(calls, [])

    def test_phone_real_mode_enforces_gate_before_creating_output_or_loading_models(self):
        with fixture() as path:
            event_file = path.parent / 'event.json'
            event_file.write_text(json.dumps({'before': probe.SOURCE_HEAD}))
            with patch.dict('os.environ', {'GITHUB_EVENT_PATH': str(event_file)}, clear=True):
                with self.assertRaisesRegex(ValueError, 'PHONE_ONE_SUCCESSOR_PUSH_ONLY'):
                    probe.capture_phones(path, path.parent / 'phones')
            self.assertFalse((path.parent / 'phones').exists())

    def test_phone_partial_cli_and_exclusive_modes(self):
        with patch.object(probe, 'capture_phones', return_value={'status': 'INCOMPLETE_NO_RETRY'}):
            self.assertEqual(probe.main(['--source-zip', '/tmp/source', '--output', '/tmp/result',
                                         '--execute-phones']), 2)
        for argv in [['--execute-asr', '--execute-phones'],
                     ['--execute-phones', '--provider', 'qwen', '--source-zip', '/tmp/source',
                      '--output', '/tmp/result'],
                     ['--combine', '/tmp/sources', '--execute-phones', '--output', '/tmp/result']]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                probe.main(argv)


if __name__ == '__main__':
    unittest.main()
