"""Artificial PCM and injected backends only; no model downloads or real calls."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_final_ipa_rendering_once as render


class FakeBackend:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def synthesize(self, cell):
        self.calls.append(dict(cell))
        if len(self.calls) == self.fail_at:
            raise RuntimeError('Artificial TTS failure')
        return len(self.calls), 24000


class FakeWriter:
    def quality(self, samples, rate):
        return {'native_frames': 2400, 'fixture': True}

    def _write_waves(self, folder, samples, rate, technical):
        pcm = int(samples).to_bytes(2, 'little') * 1600
        wav = folder / 'audio-16k.wav'
        with wave.open(str(wav), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(pcm)
        native = b'Explicitly artificial native audio ' + str(samples).encode()
        (folder / 'native-f32.wav').write_bytes(native)
        raw = wav.read_bytes()
        return {'native_sha256': render.sha(native), 'native_bytes': len(native),
                'wav_sha256': render.sha(raw), 'pcm_sha256': render.sha(pcm),
                'audio_frames': 1600, 'wav_bytes': len(raw), 'sample_rate_hz': 16000}


@contextlib.contextmanager
def source(fail_at=None):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        backend = FakeBackend(fail_at)
        with patch.dict('os.environ', {'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '12345',
                                      'GITHUB_RUN_ATTEMPT': '1'}, clear=True):
            result = render.generate(root / 'source', runtime=(backend, FakeWriter(), {
                'model_id': render.MODEL, 'revision': render.REV, 'fixture': True}))
            yield root, backend, result


def edit(path, mutate):
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def fake_observer(provider, calls, fail_at=None):
    def execute(wav, folder):
        if wav.name != 'audio.wav' or wav.parent != folder:
            raise ValueError('Nonanonymous path')
        calls.append(render.sha(wav.read_bytes()))
        if len(calls) == fail_at:
            raise RuntimeError('Artificial observer failure')
        if provider == 'phones':
            return ['h', 'eɪ', 'm', 'i'], ['h', 'eɪ', 'm', 'i'], 1600
        return 'Invented machine transcript'
    return execute


def observer_fixtures(root):
    """Handwritten documents for parser tests; never evidence of an actual call."""
    binding, cells = render.source_cells(root / 'source', allow_simulated=True)
    folder = root / 'documents'
    folder.mkdir()
    for provider in render.PROVIDERS:
        out = folder / provider
        out.mkdir()
        rows = []
        for cell, _, info in cells:
            row = {'cell_id': cell['cell_id'], **info, **render.FALSE_FLAGS,
                   'status': 'COMPLETE', 'decoder_completed': True}
            if provider == 'phones':
                row.update(raw_ipa_tokens=['h', 'i'], stressless_phone_tokens=['h', 'i'])
            else:
                row['text'] = 'artificial parser fixture'
            rows.append(row)
        render.once(out / 'summary.json', {**render.identity(False), **binding,
                    'provider': provider, 'status': 'COMPLETE', 'attempts': 4, 'completed': 4,
                    'rows': rows, 'new_tts_calls': 0, 'new_optimizer_steps': 0})
    return folder


class RenderingTests(unittest.TestCase):
    def test_fixed_recipe_and_distinct_new_cells(self):
        render.check_pins()
        cells = render.recipe()['cells']
        self.assertEqual(len(cells), 4)
        self.assertEqual(len({c['cell_id'] for c in cells}), 4)
        self.assertEqual({c['canonical_phrase'] for c in cells}, {'Hey Mero', 'Hey Vosen'})
        self.assertEqual({c['synthesis_text'] for c in cells}, {'Hey Meero.', 'Hey Vozen.'})
        self.assertEqual({c['target_ipa'] for c in cells}, {'/heɪ ˈmiːroʊ/', '/heɪ ˈvoʊzən/'})
        self.assertTrue(all(c['cell_id'].startswith('enrender-') for c in cells))
        self.assertFalse(render.recipe()['training_admitted'])

    def test_guard_requires_exact_first_public_main_push(self):
        env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1',
               'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '12345'}
        event = {'before': render.PREDECESSOR, 'after': 'a' * 40,
                 'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        render.execution_guard(event, env)
        for key, value in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_EVENT_NAME', 'pull_request'),
                           ('GITHUB_REF', 'refs/heads/fork'), ('GITHUB_RUN_ID', None),
                           ('GITHUB_SHA', 'b' * 40), ('GITHUB_REPOSITORY', 'fork/repo')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'ONE_PUBLIC_MAIN'):
                render.execution_guard(event, {**env, key: value})
        for change in [{'before': 'b' * 40}, {'after': None},
                       {'repository': {'full_name': 'jiying2007/kws-data', 'private': True}}]:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'ONE_PUBLIC_MAIN'):
                render.execution_guard({**event, **change}, env)

    def test_generation_uses_only_new_synthesis_text_and_four_calls(self):
        with source() as (root, backend, result):
            self.assertEqual(result['status'], 'SIMULATED_COMPLETE')
            self.assertEqual((result['tts_attempts'], result['generated_audio']), (4, 4))
            self.assertEqual([c['phrase'] for c in backend.calls],
                             ['Hey Meero.', 'Hey Meero.', 'Hey Vozen.', 'Hey Vozen.'])
            self.assertTrue(all(set(c) == {'cell_id', 'phrase', 'speaker', 'instruction'}
                                for c in backend.calls))
            self.assertTrue(result['test_backend'])
            self.assertTrue(all(result[k] is False for k in render.FALSE_FLAGS))
            _, cells = render.source_cells(root / 'source', allow_simulated=True)
            self.assertEqual(len(cells), 4)
            with self.assertRaisesRegex(ValueError, 'CREATE_ONLY_OUTPUT'):
                render.generate(root / 'source', runtime=(backend, FakeWriter(), {}))
            self.assertEqual(len(backend.calls), 4)

    def test_generation_partial_failure_has_no_retry_or_consumable_success(self):
        with source(fail_at=2) as (root, backend, result):
            self.assertEqual(len(backend.calls), 4)
            self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
            self.assertEqual(result['generated_audio'], 3)
            self.assertEqual(len(list((root / 'source').glob('cells/*/failed.json'))), 1)
            with self.assertRaisesRegex(ValueError, 'SOURCE_NOT_COMPLETE'):
                render.source_cells(root / 'source', allow_simulated=True)

    def test_real_source_consumer_rejects_simulation(self):
        with source() as (root, _, _):
            with self.assertRaisesRegex(ValueError, 'SIMULATED_SOURCE_REJECTED'):
                render.source_cells(root / 'source')

    def test_source_run_recipe_and_cell_receipts_must_match(self):
        changes = [
            ('started.json', lambda x: x.update(run_id='54321')),
            ('result.json', lambda x: x.update(github_sha='b' * 40)),
            ('started.json', lambda x: x.update(code_sha256='0' * 64)),
            ('started.json', lambda x: x.update(call_budget=3)),
            ('source-plan.json', lambda x: x['cells'][0].update(synthesis_text='Hey Mero.')),
            ('result.json', lambda x: x['rows'].__setitem__(1, x['rows'][0])),
            ('result.json', lambda x: x['rows'][0].update(status='FAILED_NO_RETRY')),
            ('model-lock.json', lambda x: x.update(revision='0' * 40)),
        ]
        for name, mutate in changes:
            with self.subTest(name=name), source() as (root, _, _):
                edit(root / 'source' / name, mutate)
                with self.assertRaises(ValueError):
                    render.source_cells(root / 'source', allow_simulated=True)
        with source() as (root, _, _):
            path = next((root / 'source').glob('cells/*/completed.json'))
            edit(path, lambda x: x.update(canonical_phrase='Hey Meero'))
            with self.assertRaisesRegex(ValueError, 'SOURCE_CELL_RENDERING'):
                render.source_cells(root / 'source', allow_simulated=True)

    def test_pcm_corruption_symlink_and_extra_cell_rejected(self):
        with source() as (root, _, _):
            wav = next((root / 'source').glob('cells/*/audio-16k.wav'))
            wav.write_bytes(wav.read_bytes()[:-2] + b'xx')
            with self.assertRaisesRegex(ValueError, 'SOURCE_WAV_PCM_RECEIPT'):
                render.source_cells(root / 'source', allow_simulated=True)
        with source() as (root, _, _):
            original = root / 'source/source-plan.json'
            link = root / 'plan.json'
            original.rename(link)
            original.symlink_to(link)
            with self.assertRaisesRegex(ValueError, 'SYMLINK_PATH'):
                render.source_cells(root / 'source', allow_simulated=True)
        with source() as (root, _, _):
            (root / 'source/cells/extra').mkdir()
            with self.assertRaisesRegex(ValueError, 'SOURCE_CELL_DIRECTORIES'):
                render.source_cells(root / 'source', allow_simulated=True)

    def test_each_observer_uses_four_anonymous_original_wavs(self):
        for provider in render.PROVIDERS:
            with self.subTest(provider=provider), source() as (root, _, _):
                calls = []
                out = root / provider
                result = render.observe(provider, root / 'source', out,
                                        backend=fake_observer(provider, calls))
                _, cells = render.source_cells(root / 'source', allow_simulated=True)
                self.assertEqual(calls, [info['wav_sha256'] for _, _, info in cells])
                self.assertEqual((result['attempts'], result['completed']), (4, 4))
                self.assertEqual(result['status'], 'SIMULATED_COMPLETE')
                self.assertTrue(result['test_backend'])
                self.assertTrue(all(result[k] is False for k in render.FALSE_FLAGS))
                self.assertFalse(list(out.rglob('*.wav')))
                with self.assertRaisesRegex(ValueError, 'CREATE_ONLY_OUTPUT'):
                    render.observe(provider, root / 'source', out,
                                   backend=fake_observer(provider, calls))
                self.assertEqual(len(calls), 4)

    def test_observer_failure_and_interruption_do_not_retry(self):
        with source() as (root, _, _):
            calls = []
            result = render.observe('qwen', root / 'source', root / 'qwen',
                                    backend=fake_observer('qwen', calls, fail_at=1))
            self.assertEqual((result['attempts'], result['completed']), (4, 3))
            self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
            self.assertEqual(len(calls), 4)
        with source() as (root, _, _):
            def interrupt(wav, folder):
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                render.observe('phones', root / 'source', root / 'phones', backend=interrupt)
            value = json.loads((root / 'phones/summary.json').read_bytes())
            self.assertEqual((value['attempts'], value['completed']), (1, 0))
            self.assertEqual(value['error_type'], 'KeyboardInterrupt')

    def test_phone_wrong_frames_and_asr_empty_text_do_not_complete(self):
        for provider, backend in [('phones', lambda *args: (['h', 'i'], ['h', 'i'], 1599)),
                                  ('qwen', lambda *args: '')]:
            with self.subTest(provider=provider), source() as (root, _, _):
                result = render.observe(provider, root / 'source', root / provider, backend=backend)
                self.assertEqual((result['attempts'], result['completed']), (4, 0))
                self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')

    def test_real_generation_and_observation_require_event_before_side_effects(self):
        with source() as (root, _, _):
            event = root / 'event.json'
            event.write_text('{}')
            with patch.dict('os.environ', {'GITHUB_EVENT_PATH': str(event)}, clear=True):
                for operation in [lambda: render.generate(root / 'real'),
                                  lambda: render.observe('qwen', root / 'source', root / 'real')]:
                    with self.assertRaisesRegex(ValueError, 'ONE_PUBLIC_MAIN'):
                        operation()
                self.assertFalse((root / 'real').exists())

    def test_combine_preserves_canonical_names_and_unadmitted_boundary(self):
        with source() as (root, _, _):
            folder = observer_fixtures(root)
            result = render.combine(folder, root / 'combined.json')
            self.assertEqual(result['status'], 'FOUR_OBSERVERS_COMPLETE_REVIEW_REQUIRED')
            self.assertEqual((result['asr_calls'], result['phoneme_calls']), (12, 4))
            self.assertEqual(len(result['cells']), 4)
            self.assertEqual({c['canonical_phrase'] for c in result['cells']}, {'Hey Mero', 'Hey Vosen'})
            self.assertTrue(all(result[k] is False for k in render.FALSE_FLAGS))
            self.assertTrue(all(c['acoustic_review'] == 'PENDING' for c in result['cells']))

    def test_combine_rejects_cross_run_simulated_partial_and_wrong_source(self):
        changes = [lambda x: x.update(run_id='99999'), lambda x: x.update(test_backend=True),
                   lambda x: x.update(completed=3), lambda x: x.update(training_admitted=True),
                   lambda x: x.update(code_sha256='0' * 64),
                   lambda x: x['source_cells'][0].update(wav_sha256='0' * 64),
                   lambda x: x['rows'][0].update(wav_sha256='0' * 64),
                   lambda x: x['rows'].__setitem__(1, x['rows'][0]),
                   lambda x: x.update(source_plan_sha256='0' * 64)]
        for mutate in changes:
            with self.subTest(mutate=mutate), source() as (root, _, _):
                folder = observer_fixtures(root)
                edit(folder / 'qwen/summary.json', mutate)
                with self.assertRaises(ValueError):
                    render.combine(folder, root / 'combined.json')
                self.assertFalse((root / 'combined.json').exists())

    def test_injected_observations_are_never_combinable_as_real_results(self):
        with source() as (root, _, _):
            observations = root / 'observations'
            observations.mkdir()
            for provider in render.PROVIDERS:
                render.observe(provider, root / 'source', observations / provider,
                               backend=fake_observer(provider, []))
            with self.assertRaisesRegex(ValueError, 'OBSERVER_IDENTITY_OR_COMPLETION'):
                render.combine(observations, root / 'combined.json')

    def test_cli_partial_exit_and_mode_exclusion(self):
        with patch.object(render, 'generate', return_value={'status': 'INCOMPLETE_NO_RETRY'}):
            self.assertEqual(render.main(['--generate', '--output', '/tmp/not-created']), 2)
        with patch.object(render, 'observe', return_value={'status': 'INCOMPLETE_NO_RETRY'}):
            self.assertEqual(render.main(['--observe', 'phones', '--source-parent', '/tmp/source',
                                          '--output', '/tmp/not-created']), 2)
        with contextlib.redirect_stderr(sys.stdout), self.assertRaises(SystemExit):
            render.main(['--generate', '--observe', 'qwen'])


if __name__ == '__main__':
    unittest.main()
