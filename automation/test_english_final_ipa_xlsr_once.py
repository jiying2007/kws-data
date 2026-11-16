"""Offline source/receipt/CTC boundary tests; no model, downloads or inference."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_final_ipa_xlsr_once as stage
from test_english_ipa_qwen17_final_ipa_probe_once import fixture as original_fixture


def raw_json(value):
    return json.dumps(value, ensure_ascii=False).encode()


def mutate_json(members, name, modify):
    value = json.loads(members[name])
    modify(value)
    members[name] = raw_json(value)


def rendered_members():
    identity = {'schema': 'english-final-ipa-orthographic-rendering-v1',
                'github_sha': stage.PREDECESSOR, 'run_id': str(stage.RENDER_RUN),
                'run_attempt': '1', 'code_sha256': stage.RENDER_CODE_SHA,
                'recipe_sha256': stage.RENDER_RECIPE_SHA, 'test_backend': False, **stage.FLAGS}
    members = {'started.json': raw_json(identity), 'model-lock.json': b'{"artificial_fixture":true}'}
    roster, cells, rows = [], [], []
    for index, original in enumerate(stage.RENDER_ROSTER):
        cid, phrase, speaker = original[:3]
        frames = 1600
        pcm = bytes((index + 11, 0)) * frames
        out = io.BytesIO()
        with wave.open(out, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        wav, native = out.getvalue(), b'artificial-native-' + str(index).encode()
        wav_sha, pcm_sha = stage.sha(wav), stage.sha(pcm)
        roster.append((cid, phrase, speaker, frames, wav_sha, pcm_sha))
        cell = {'cell_id': cid, 'canonical_phrase': phrase, 'speaker': speaker,
                'target_ipa': stage.original.IPAS[phrase]}
        cells.append(cell)
        rows.append({'cell_id': cid, 'status': 'GENERATED_UNADMITTED',
                     'wav_sha256': wav_sha, 'pcm_sha256': pcm_sha})
        folder = 'cells/' + cid + '/'
        members[folder + 'audio-16k.wav'] = wav
        members[folder + 'native-f32.wav'] = native
        members[folder + 'started.json'] = raw_json({**identity, 'cell_id': cid, 'attempt': 1})
        members[folder + 'completed.json'] = raw_json({
            **identity, **cell, 'status': 'GENERATED_UNADMITTED',
            'audio': {'wav_sha256': wav_sha, 'pcm_sha256': pcm_sha,
                      'native_sha256': stage.sha(native), 'wav_bytes': len(wav), 'audio_frames': frames}})
    members['source-plan.json'] = raw_json({'cells': cells})
    members['result.json'] = raw_json({**identity, 'status': 'COMPLETE', 'tts_attempts': 4,
                                     'generated_audio': 4, 'asr_calls': 0, 'phoneme_calls': 0,
                                     'optimizer_steps': 0, 'rows': rows})
    return members, tuple(roster)


@contextlib.contextmanager
def rendered_fixture(change=None):
    members, roster = rendered_members()
    plan_sha = stage.sha(members['source-plan.json'])
    lock_sha = stage.sha(members['model-lock.json'])
    if change:
        change(members)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'rendered.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
        with patch.object(stage, 'RENDER_ZIP_SHA', stage.sha(path.read_bytes())), \
             patch.object(stage, 'RENDER_PLAN_SHA', plan_sha), \
             patch.object(stage, 'RENDER_LOCK_SHA', lock_sha), patch.object(stage, 'RENDER_ROSTER', roster):
            yield path


@contextlib.contextmanager
def both_sources():
    with original_fixture() as first, rendered_fixture() as second:
        yield first, second


def fake_phone(wav):
    with wave.open(str(wav), 'rb') as stream:
        return ['h', 'eɪ', 'm', 'iː', 'ɹ', 'oʊ'], stream.getnframes()


def valid_guard():
    env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
           'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1',
           'GITHUB_SHA': 'b' * 40, 'GITHUB_RUN_ID': '999'}
    event = {'before': stage.PREDECESSOR, 'after': env['GITHUB_SHA'],
             'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
    return event, env


class XLSRTests(unittest.TestCase):
    def test_fixed_model_and_source_pins(self):
        stage.check_helper()
        self.assertEqual(stage.MODEL, 'facebook/wav2vec2-xlsr-53-espeak-cv-ft')
        self.assertEqual(stage.REVISION, '2c733782da5604684829819a5eb744c193fe9398')
        self.assertEqual(stage.WEIGHTS_BYTES, 1263535127)
        self.assertEqual(stage.WEIGHTS_SHA, '04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551')
        self.assertEqual(stage.RENDER_RUN, 43809825038)
        self.assertEqual(stage.RENDER_ARTIFACT, 14199609338)
        self.assertEqual(len({r[4] for r in stage.RENDER_ROSTER + stage.original.ROSTER}), 8)

    def test_helper_change_is_rejected(self):
        with patch.object(stage, 'SOURCE_HELPER_BLOB', '0' * 40):
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_SOURCE_HELPER_CHANGED'):
                stage.check_helper()

    def test_eight_sources_keep_rendering_and_ipa_identity(self):
        with both_sources() as (first, second):
            cells = stage.source_cells(first, second)
            self.assertEqual(len(cells), 8)
            self.assertEqual([x[0]['source_group'] for x in cells], ['original'] * 4 + ['rendered'] * 4)
            self.assertEqual(cells[4][0]['synthesis_text'], 'Hey Meero.')
            self.assertEqual(cells[6][0]['synthesis_text'], 'Hey Vozen.')
            self.assertEqual(cells[4][0]['target_ipa'], '/heɪ ˈmiːroʊ/')
            self.assertEqual(cells[6][0]['canonical_phrase'], 'Hey Vosen')

    def test_render_outer_hash_rejects_changed_bytes(self):
        with rendered_fixture() as path:
            path.write_bytes(path.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'RENDER_ZIP_SHA'):
                stage.rendered_cells(path)

    def test_extra_missing_and_symlink_source_rejected(self):
        for change in (lambda m: m.update({'../escape': b'x'}), lambda m: m.pop('started.json')):
            with self.subTest(change=change), rendered_fixture(change) as path:
                with self.assertRaisesRegex(ValueError, 'RENDER_MEMBER_SET'):
                    stage.rendered_cells(path)
        with rendered_fixture() as path:
            link = path.parent / 'linked.zip'
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'SYMLINK_PATH'):
                stage.rendered_cells(link)

    def test_plan_or_lock_change_rejected(self):
        for name in ('source-plan.json', 'model-lock.json'):
            with self.subTest(name=name), rendered_fixture(lambda m: m.update({name: b'{}'})) as path:
                with self.assertRaisesRegex(ValueError, 'RENDER_PLAN_LOCK_SHA'):
                    stage.rendered_cells(path)

    def test_cross_run_simulated_and_promoted_sources_rejected(self):
        for name in ('started.json', 'result.json'):
            for key, value in (('run_id', '777'), ('github_sha', 'c' * 40), ('test_backend', True),
                               ('human_gold', True), ('training_admitted', True)):
                change = lambda m: mutate_json(m, name, lambda r: r.update({key: value}))
                with self.subTest(name=name, key=key), rendered_fixture(change) as path:
                    with self.assertRaisesRegex(ValueError, 'RENDER_SOURCE_IDENTITY'):
                        stage.rendered_cells(path)

    def test_partial_and_unknown_source_never_complete(self):
        for change in ({'generated_audio': 3}, {'status': 'UNKNOWN'}, {'asr_calls': 1}):
            with self.subTest(change=change), rendered_fixture(
                    lambda m: mutate_json(m, 'result.json', lambda r: r.update(change))) as path:
                with self.assertRaisesRegex(ValueError, 'RENDER_NOT_COMPLETE'):
                    stage.rendered_cells(path)

    def test_started_and_completed_cell_identity_rejected(self):
        for leaf, key, value in (('started.json', 'test_backend', True),
                                 ('completed.json', 'cell_id', 'other-cell'),
                                 ('completed.json', 'run_id', 'other-run')):
            name = 'cells/' + stage.RENDER_ROSTER[0][0] + '/' + leaf
            change = lambda m: mutate_json(m, name, lambda r: r.update({key: value}))
            with self.subTest(leaf=leaf, key=key), rendered_fixture(change) as path:
                with self.assertRaisesRegex(ValueError, 'RENDER_CELL_RUN'):
                    stage.rendered_cells(path)

    def test_pcm_change_rejected_even_with_new_outer_hash(self):
        name = 'cells/' + stage.RENDER_ROSTER[0][0] + '/audio-16k.wav'
        with rendered_fixture(lambda m: m.update({name: m[name][:-2] + b'\x7f\x7f'})) as path:
            with self.assertRaisesRegex(ValueError, 'RENDER_WAV_PCM_BINDING'):
                stage.rendered_cells(path)

    def test_ctc_collapse_precedes_blank_removal(self):
        vocab = {'<pad>': 0, 'iː': 30, 'z': 21, '<s>': 1}
        self.assertEqual(stage.ctc_tokens([0, 30, 30, 0, 30, 21, 21, 1], vocab, {0, 1}),
                         ['iː', 'iː', 'z'])
        for ids in ([0, 0], [99], [True], [], [30] * 2001):
            with self.subTest(ids=ids[:4]), self.assertRaises(ValueError):
                stage.ctc_tokens(ids, vocab, {0, 1})

    def test_public_first_successor_guard_rejects_rerun_fork_wrong_branch(self):
        event, env = valid_guard()
        stage.execution_guard(event, env)
        for key, value in (('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_EVENT_NAME', 'workflow_dispatch'),
                           ('GITHUB_REPOSITORY', 'fork/kws-data'), ('GITHUB_REF', 'refs/heads/topic')):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'FIRST_PUBLIC_MAIN_SUCCESSOR'):
                stage.execution_guard(event, {**env, key: value})
        for change in ({'before': 'a' * 40}, {'after': 'c' * 40},
                       {'repository': {'full_name': 'jiying2007/kws-data', 'private': True}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'FIRST_PUBLIC_MAIN_SUCCESSOR'):
                stage.execution_guard({**event, **change}, env)

    def test_eight_audio_only_calls_simulated_not_gold_and_create_only(self):
        seen = []
        def backend(wav):
            self.assertEqual(wav.name, 'audio.wav')
            self.assertTrue(wav.parent.name.startswith('observation-'))
            seen.append(stage.sha(wav.read_bytes()))
            return fake_phone(wav)
        with both_sources() as (first, second):
            output = first.parent / 'result'
            value = stage.execute(first, second, output, backend=backend)
            self.assertEqual(value['attempts'], 8)
            self.assertEqual(value['completed'], 8)
            self.assertEqual(len(set(seen)), 8)
            self.assertEqual(value['status'], 'SIMULATED_COMPLETE')
            self.assertTrue(value['test_backend'])
            self.assertEqual(value['new_tts_calls'] + value['new_asr_calls'] + value['new_optimizer_steps'], 0)
            self.assertTrue(all(value[k] is False for k in stage.FLAGS))
            self.assertEqual(len(list(output.glob('observation-*/started.json'))), 8)
            self.assertEqual(len(list(output.glob('observation-*/completed.json'))), 8)
            self.assertEqual(list(output.rglob('*.wav')), [])
            with self.assertRaisesRegex(ValueError, 'CREATE_ONLY_OUTPUT'):
                stage.execute(first, second, output, backend=backend)
            self.assertEqual(len(seen), 8)

    def test_failure_preserves_all_attempts_without_retry(self):
        seen = []
        def backend(wav):
            seen.append(wav)
            if len(seen) == 1:
                raise RuntimeError('invented decoder failure')
            return fake_phone(wav)
        with both_sources() as (first, second):
            output = first.parent / 'partial'
            value = stage.execute(first, second, output, backend=backend)
            self.assertEqual((value['attempts'], value['completed']), (8, 7))
            self.assertEqual(value['status'], 'INCOMPLETE_NO_RETRY')
            self.assertEqual(value['rows'][0]['error_type'], 'RuntimeError')
            self.assertEqual(len(list(output.glob('observation-*/failed.json'))), 1)
            self.assertEqual(len(seen), 8)

    def test_invalid_decoder_outputs_never_complete(self):
        for output_value in (([], 80), (['iː'], 81), (['bad\ntoken'], 80), (['iː'], float('nan'))):
            with self.subTest(value=output_value), both_sources() as (first, second):
                value = stage.execute(first, second, first.parent / 'invalid', backend=lambda _: output_value)
                self.assertEqual((value['attempts'], value['completed']), (8, 0))
                self.assertTrue(all(r['status'] == 'FAILED_NO_RETRY' for r in value['rows']))

    def test_interrupt_retains_started_and_terminal_summary(self):
        def interrupt(_):
            raise KeyboardInterrupt()
        with both_sources() as (first, second):
            output = first.parent / 'interrupted'
            with self.assertRaises(KeyboardInterrupt):
                stage.execute(first, second, output, backend=interrupt)
            value = stage.loads((output / 'summary.json').read_bytes())
            self.assertEqual((value['attempts'], value['completed']), (1, 0))
            self.assertEqual(value['status'], 'INCOMPLETE_NO_RETRY')
            self.assertEqual(len(list(output.glob('observation-*/started.json'))), 1)

    def test_real_gate_rejects_before_loading_or_creating_output(self):
        with both_sources() as (first, second):
            event, env = valid_guard()
            env['GITHUB_RUN_ATTEMPT'] = '2'
            event_file = first.parent / 'event.json'
            event_file.write_bytes(raw_json(event))
            output = first.parent / 'forbidden'
            with patch.dict(os.environ, {**env, 'GITHUB_EVENT_PATH': str(event_file)}, clear=True), \
                 patch.object(stage, 'load_runtime') as loader:
                with self.assertRaisesRegex(ValueError, 'FIRST_PUBLIC_MAIN_SUCCESSOR'):
                    stage.execute(first, second, output)
                loader.assert_not_called()
            self.assertFalse(output.exists())

    def test_model_cache_is_outside_artifact_and_cleaned(self):
        with both_sources() as (first, second):
            event, env = valid_guard()
            event_file = first.parent / 'event.json'
            event_file.write_bytes(raw_json(event))
            output, seen = first.parent / 'real-path-mocked-loader', []
            def loader(folder):
                seen.append(folder)
                self.assertFalse(folder.is_relative_to(output))
                (folder / 'pytorch_model.bin').write_bytes(b'ARTIFICIAL SMALL TEST FILE')
                return fake_phone, {'test_model_loader': True}
            with patch.dict(os.environ, {**env, 'GITHUB_EVENT_PATH': str(event_file)}, clear=True), \
                 patch.object(stage, 'load_runtime', side_effect=loader):
                value = stage.execute(first, second, output)
            self.assertEqual(value['completed'], 8)
            self.assertEqual(len(seen), 1)
            self.assertFalse(seen[0].exists())
            self.assertTrue((output / 'model-lock.json').is_file())
            self.assertEqual(list(output.rglob('*.bin')), [])

    def test_cli_source_only_never_loads_model(self):
        with both_sources() as (first, second), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(stage, 'load_runtime') as loader:
            self.assertEqual(stage.main([]), 0)
            self.assertEqual(stage.main(['--original-zip', str(first), '--rendered-zip', str(second)]), 0)
            loader.assert_not_called()

    def test_cli_model_load_failure_returns_two_with_zero_attempts(self):
        with both_sources() as (first, second):
            event, env = valid_guard()
            event_file = first.parent / 'event.json'
            event_file.write_bytes(raw_json(event))
            output = first.parent / 'model-load-failure'
            with patch.dict(os.environ, {**env, 'GITHUB_EVENT_PATH': str(event_file)}, clear=True), \
                 patch.object(stage, 'load_runtime', side_effect=RuntimeError('invented load error')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(stage.main(['--execute', '--original-zip', str(first),
                                            '--rendered-zip', str(second), '--output', str(output)]), 2)
            value = stage.loads((output / 'summary.json').read_bytes())
            self.assertEqual((value['attempts'], value['completed']), (0, 0))
            self.assertEqual(value['error_type'], 'RuntimeError')


if __name__ == '__main__':
    unittest.main()
