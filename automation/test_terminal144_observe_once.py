"""Invented receipt/WAV fixtures only; no real model, network, or Actions env."""
from __future__ import annotations
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
import wave
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('terminal144_under_test', HERE / 'terminal144_observe_once.py')
stage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stage
spec.loader.exec_module(stage)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage.once(path, value)


def wav_bytes(value):
    pcm = struct.pack('<160h', *([value] * 160))
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setparams((1, 2, 16000, 160, 'NONE', 'not compressed'))
        wav.writeframes(pcm)
    floats = struct.pack('<160f', *([value / 32768] * 160))
    native = b'RIFF' + struct.pack('<I', 36 + len(floats)) + b'WAVEfmt ' + \
             struct.pack('<IHHIIHH', 16, 3, 1, 16000, 64000, 4, 32) + \
             b'data' + struct.pack('<I', len(floats)) + floats
    return output.getvalue(), pcm, native


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='terminal144-invented-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = {'model_id': stage.recipe.MODEL, 'revision': stage.recipe.REVISION,
                     'test_backend': True, 'not_actual_assets': True}
        lock_raw = (json.dumps(self.lock, ensure_ascii=False, sort_keys=True) + '\n').encode()
        patcher = mock.patch.dict(stage.SOURCE, {'source_model_lock_sha256': stage.sha(lock_raw)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bundle, self.pins = self.build_bundle(self.root / 'fixture')
        self.ledger, _ = stage.read_sources(self.bundle, self.pins, self.root / 'initial-audit', allow_simulated=True)
        self.pins.update(terminal_evidence_canonical_sha256=self.ledger['terminal_evidence_canonical_sha256'],
                         source_ledger_canonical_sha256=stage.sha(stage.canonical(self.ledger)),
                         valid_cells=self.ledger['valid_cells'], max_calls_per_provider=self.ledger['valid_count'])

    def build_bundle(self, parent, duplicate=False):
        bundle = parent / 'bundle'
        bundle.mkdir(parents=True)
        pins = stage.pins_template()
        whole = stage.recipe.recipe()
        identity = stage.source_identity(True)
        artifacts = []
        for index in range(9):
            root = parent / ('generated-%02d' % index)
            root.mkdir()
            cells = whole['cells'][index * 16:(index + 1) * 16]
            counts = {'cells': 16, 'train': sum(c['role'] == 'train' for c in cells),
                'development': sum(c['role'] == 'development' for c in cells),
                'wake_candidates': sum(c['intent'] == 'wake-candidate' for c in cells),
                'ordinary_negative_candidates': sum(c['intent'] == 'ordinary-negative-candidate' for c in cells),
                'independent_humans': 0, 'independent_test_cells': 0}
            plan = {**whole, 'cells': cells, 'counts': counts, 'global_counts': whole['counts'],
                'global_recipe_sha256': stage.SOURCE['source_recipe_sha256'], 'shard_index': index,
                'shard_count': 9, 'max_tts_calls': 16, 'capture_stage_budgets': [16],
                'capture_identity': identity, 'model_lock_sha256': stage.SOURCE['source_model_lock_sha256']}
            base = {**identity, 'shard_index': index, 'shard_count': 9}
            write(root / 'started.json', {**base, 'operation': 'generate', 'call_budget': 16})
            write(root / 'source-plan.json', plan)
            write(root / 'model-lock.json', self.lock)
            plan_sha = stage.sha(stage.canonical(plan))
            write(root / 'capture/plan-lock.json', {'policy': 'en-market-qwen-customvoice-capture-v1',
                'plan_sha256': plan_sha, 'model_calls_admitted_to_training': False})
            if index == 0:
                for ordinal, cell in enumerate(cells[:6]):
                    folder = root / 'capture/cells' / cell['cell_id']
                    folder.mkdir(parents=True)
                    start = {'schema': 'en-market-qwen-customvoice-capture-v1',
                        'cell_id': cell['cell_id'], 'plan_sha256': plan_sha, 'model_calls': 1,
                        'retry_count': 0, 'source_family_id': cell['source_family_id'], 'training_admitted': False}
                    if ordinal == 5:
                        start['plan_sha256'] = '0' * 64
                    write(folder / 'started.json', start)
                    if ordinal < 3:
                        wav, pcm, native = wav_bytes(2 if duplicate and ordinal == 2 else ordinal + 1)
                        (folder / 'audio-16k.wav').write_bytes(wav)
                        (folder / 'native-f32.wav').write_bytes(native)
                        write(folder / 'completed.json', {'schema': start['schema'], 'cell_id': cell['cell_id'],
                            'plan_sha256': plan_sha, 'status': 'GENERATED_UNADMITTED', 'actual_text_verified': False,
                            'training_admitted': False, 'qualification_allowed': False, 'human_gold': False,
                            'audio': {'wav_sha256': stage.sha(wav), 'wav_bytes': len(wav),
                                'pcm_sha256': stage.sha(pcm), 'native_sha256': stage.sha(native),
                                'audio_frames': 160, 'sample_rate_hz': 16000}})
                    elif ordinal == 3:
                        write(folder / 'failed.json', {'schema': start['schema'], 'cell_id': cell['cell_id'],
                            'training_admitted': False, 'status': 'FAILED_NO_RETRY', 'error_type': 'ValueError'})
            rows = stage.original.capture_rows(plan, root / 'capture')
            write(root / 'result.json', {**base, 'status': 'INCOMPLETE_NO_RETRY',
                'planned_audio': 16, 'global_planned_audio': 144,
                'tts_attempts': sum(r['attempted'] for r in rows),
                'generated_audio': sum(r['status'] == 'GENERATED_UNADMITTED' for r in rows),
                'source_plan_sha256': stage.sha((root / 'source-plan.json').read_bytes()),
                'model_lock_sha256': stage.SOURCE['source_model_lock_sha256'],
                'capture_stages_completed': 0, 'error_type': 'InventedInterruptedSource',
                'asr_calls': 0, 'phoneme_calls': 0, 'optimizer_steps': 0, 'rows': rows})
            archive = bundle / ('shard-%02d.zip' % index)
            with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as zipped:
                for path in sorted(root.rglob('*')):
                    if path.is_file():
                        zipped.write(path, path.relative_to(root).as_posix())
            pin = {'shard_index': index, 'artifact_id': 1000 + index,
                'zip_bytes': archive.stat().st_size, 'zip_sha256': stage.sha(archive.read_bytes()),
                'source_plan_sha256': stage.sha((root / 'source-plan.json').read_bytes()),
                'source_result_sha256': stage.sha((root / 'result.json').read_bytes()),
                'model_lock_sha256': stage.SOURCE['source_model_lock_sha256']}
            pins['source_artifacts'].append(pin)
            artifacts.append({'id': pin['artifact_id'], 'name': 'shard-%02d' % index,
                'size_in_bytes': pin['zip_bytes'], 'digest': 'sha256:' + pin['zip_sha256'], 'expired': False,
                'workflow_run': {'id': stage.SOURCE['source_run_id'],
                    'head_sha': stage.SOURCE['source_head_sha'], 'head_branch': 'main'}})
        run = {'id': stage.SOURCE['source_run_id'], 'head_sha': stage.SOURCE['source_head_sha'],
            'head_branch': 'main', 'run_attempt': 1, 'event': 'push', 'status': 'completed', 'conclusion': 'failure',
            'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        jobs = []
        names = ['source-contract', 'evidence'] + ['generate (%02d)' % i for i in range(9)] + \
                ['observe', 'train-weak-intent']
        for number, name in enumerate(names):
            jobs.append({'id': 5000 + number, 'name': name, 'run_id': run['id'], 'head_sha': run['head_sha'],
                'run_attempt': 1, 'status': 'completed', 'steps': [],
                'conclusion': 'skipped' if name in ('observe', 'train-weak-intent') else
                              ('failure' if name.startswith('generate') else 'success')})
        write(bundle / 'run.json', run)
        write(bundle / 'jobs.json', {'total_count': len(jobs), 'jobs': jobs})
        write(bundle / 'artifacts.json', {'total_count': len(artifacts), 'artifacts': artifacts})
        return bundle, pins

    def rewrite_zip(self, changes):
        archive = self.bundle / 'shard-00.zip'
        with zipfile.ZipFile(archive) as zipped:
            contents = {name: zipped.read(name) for name in zipped.namelist()}
        contents.update(changes)
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as zipped:
            for name, raw in contents.items():
                zipped.writestr(name, raw)
        pin = self.pins['source_artifacts'][0]
        pin.update(zip_bytes=archive.stat().st_size, zip_sha256=stage.sha(archive.read_bytes()))
        if 'result.json' in changes:
            pin['source_result_sha256'] = stage.sha(changes['result.json'])
        metadata = json.loads((self.bundle / 'artifacts.json').read_bytes())
        metadata['artifacts'][0].update(size_in_bytes=pin['zip_bytes'], digest='sha256:' + pin['zip_sha256'])
        (self.bundle / 'artifacts.json').write_text(json.dumps(metadata))
        self.pins.update(source_ledger_canonical_sha256=None, valid_cells=None, max_calls_per_provider=None)

    def test_full_denominator_partial_source_and_no_environment_mutation(self):
        before = dict(stage.os.environ)
        ledger, eligible = stage.read_sources(self.bundle, self.pins, self.root / 'read-again', allow_simulated=True)
        self.assertEqual(dict(stage.os.environ), before)
        self.assertEqual(len(ledger['rows']), 144)
        self.assertEqual(len(eligible), 3)
        self.assertEqual(ledger['source_dispositions'], {stage.VALID: 3, 'FAILED_NO_RETRY': 1,
            'STARTED_UNKNOWN_NO_RETRY': 1, 'INVALID_SOURCE_RECEIPT': 1, 'NOT_STARTED': 138})
        self.assertEqual(ledger['rows'][3]['source_error_type'], 'ValueError')
        self.assertTrue(all(r[k] is False for r in ledger['rows'] for k in stage.recipe.FLAGS))
        self.assertEqual(ledger['original_campaign_status'], 'INCOMPLETE_NO_RETRY')

    def test_simulated_source_is_rejected_by_normal_reader(self):
        with self.assertRaisesRegex(ValueError, 'SIMULATED_SOURCE_REJECTED'):
            stage.read_sources(self.bundle, self.pins, self.root / 'real-reader')

    def test_all_nine_terminal_and_actual_skipped_jobs_are_required(self):
        original_jobs = json.loads((self.bundle / 'jobs.json').read_bytes())
        for name, changes in [('generate (08)', {'status': 'in_progress', 'conclusion': None}),
                              ('observe', {'conclusion': 'success'}),
                              ('train-weak-intent', {'steps': [{'conclusion': 'success'}]})]:
            with self.subTest(job=name, changes=changes):
                jobs = copy.deepcopy(original_jobs)
                next(j for j in jobs['jobs'] if j['name'] == name).update(changes)
                (self.bundle / 'jobs.json').write_text(json.dumps(jobs))
                with self.assertRaises(ValueError):
                    stage.terminal_evidence(self.bundle)
        (self.bundle / 'jobs.json').write_text(json.dumps(original_jobs))
        run = json.loads((self.bundle / 'run.json').read_bytes())
        run['conclusion'] = 'success'
        (self.bundle / 'run.json').write_text(json.dumps(run))
        with self.assertRaisesRegex(ValueError, 'TERMINAL_FAILURE'):
            stage.terminal_evidence(self.bundle)

    def test_every_duplicate_pcm_is_excluded_without_changing_source_status(self):
        bundle, pins = self.build_bundle(self.root / 'duplicate', duplicate=True)
        ledger, eligible = stage.read_sources(bundle, pins, self.root / 'dedup', allow_simulated=True)
        self.assertEqual(len(eligible), 1)
        self.assertEqual(ledger['source_dispositions']['EXCLUDED_DUPLICATE_PCM'], 2)
        self.assertEqual([r['source_status'] for r in ledger['rows'][1:3]], ['GENERATED_UNADMITTED'] * 2)

    def test_native_mismatch_is_invalid_not_a_negative_or_eligible(self):
        cid = self.ledger['rows'][0]['cell_id']
        self.rewrite_zip({'capture/cells/' + cid + '/native-f32.wav': b'invented-corruption'})
        ledger, eligible = stage.read_sources(self.bundle, self.pins, self.root / 'native-mismatch', allow_simulated=True)
        self.assertEqual(len(eligible), 2)
        self.assertEqual(ledger['rows'][0]['source_status'], 'GENERATED_UNADMITTED')
        self.assertEqual(ledger['rows'][0]['source_disposition'], 'INVALID_SOURCE_RECEIPT')
        self.assertIs(ledger['rows'][0]['trusted_negative'], False)

    def test_exact_zip_pins_and_safe_member_roster(self):
        with (self.bundle / 'shard-00.zip').open('ab') as stream:
            stream.write(b'invented-tail')
        with self.assertRaisesRegex(ValueError, 'SOURCE_ZIP_BYTES'):
            stage.read_sources(self.bundle, self.pins, self.root / 'wrong-zip', allow_simulated=True)
        self.rewrite_zip({'../outside': b'not permitted'})
        with self.assertRaisesRegex(ValueError, 'SOURCE_ZIP_ROSTER'):
            stage.read_sources(self.bundle, self.pins, self.root / 'traversal', allow_simulated=True)
        self.assertFalse((self.root / 'outside').exists())

    def test_fixed_valid_cohort_cannot_silently_change(self):
        pins = copy.deepcopy(self.pins)
        pins['valid_cells'] = pins['valid_cells'][:-1]
        pins['max_calls_per_provider'] -= 1
        with self.assertRaisesRegex(ValueError, 'FROZEN_COHORT_CHANGED'):
            stage.read_sources(self.bundle, pins, self.root / 'reduced-n', allow_simulated=True)

    def test_enabled_first_successor_guard_only_accepts_new_identity(self):
        pins = copy.deepcopy(self.pins)
        pins.update(code_sha256=stage.sha(Path(stage.__file__).read_bytes()),
                    next_main_before=stage.SOURCE['source_head_sha'])
        env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1',
               'GITHUB_RUN_ID': '99999999999', 'GITHUB_SHA': 'a' * 40}
        event = {'before': pins['next_main_before'], 'after': env['GITHUB_SHA'],
                 'repository': {'full_name': env['GITHUB_REPOSITORY'], 'private': False}}
        with self.assertRaisesRegex(ValueError, 'EXECUTION_DISABLED'):
            stage.execution_guard(pins, event, env)
        pins['enabled'] = True
        stage.execution_guard(pins, event, env)
        for changed in ({**env, 'GITHUB_RUN_ATTEMPT': '2'}, {},
                        {**env, 'GITHUB_RUN_ID': str(stage.SOURCE['source_run_id'])}):
            with self.assertRaisesRegex(ValueError, 'FIRST_NEW_PUBLIC_MAIN'):
                stage.execution_guard(pins, event, changed)
        with self.assertRaises(ValueError):
            stage.execution_guard(pins, {**event, 'repository': {'full_name': env['GITHUB_REPOSITORY'], 'private': True}}, env)

    def test_one_call_per_valid_cell_partial_failure_and_no_retry(self):
        calls = []
        def backend(wav, folder):
            self.assertEqual(wav.name, 'audio.wav')
            self.assertTrue(wav.is_file())
            self.assertNotIn(self.root / 'observation', wav.parents)
            calls.append(stage.sha(wav.read_bytes()))
            if len(calls) == 2:
                raise ValueError('invented single failure token=invented-sensitive-value ' + 'x' * 2500)
            return 'invented transcript'
        output = self.root / 'observation'
        result = stage.observe('qwen', self.bundle, self.pins, output, backend=backend)
        self.assertEqual(len(calls), self.pins['max_calls_per_provider'])
        self.assertEqual(result['attempts'], 3)
        self.assertEqual(result['completed'], 2)
        self.assertEqual(len(result['rows']), 144)
        self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
        self.assertEqual(result['rows'][1]['observation_status'], 'FAILED_NO_RETRY')
        self.assertEqual(len(result['rows'][1]['error_message']), 2000)
        self.assertIn('invented single failure', result['rows'][1]['error_message'])
        self.assertNotIn('invented-sensitive-value', result['rows'][1]['error_message'])
        self.assertEqual(result['rows'][3]['observation_status'], 'NOT_ELIGIBLE_SOURCE')
        self.assertTrue(result['test_backend'])
        self.assertFalse(list(output.rglob('*.wav')))
        self.assertFalse((output / 'model').exists())
        with self.assertRaisesRegex(ValueError, 'CREATE_ONLY'):
            stage.observe('qwen', self.bundle, self.pins, output, backend=backend)
        self.assertEqual(len(calls), 3)

    def test_interrupted_call_remains_unknown_and_is_not_retried(self):
        calls = []
        def backend(wav, folder):
            calls.append(wav.name)
            raise KeyboardInterrupt('invented interruption')
        output = self.root / 'interrupted'
        with self.assertRaises(KeyboardInterrupt):
            stage.observe('whisper', self.bundle, self.pins, output, backend=backend)
        result = json.loads((output / 'summary.json').read_bytes())
        self.assertEqual(calls, ['audio.wav'])
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(result['completed'], 0)
        self.assertEqual(result['error_message'], 'invented interruption')
        self.assertEqual(result['rows'][0]['observation_status'], 'STARTED_UNKNOWN_NO_RETRY')
        self.assertEqual(result['rows'][1]['observation_status'], 'NOT_STARTED')

    def test_phone_frame_binding_and_empty_output_retained(self):
        values = iter([([], 160), (['m', 'i'], 159), (['v', 'z'], 160)])
        result = stage.observe('phones', self.bundle, self.pins, self.root / 'phones',
                               backend=lambda wav, folder: next(values))
        self.assertEqual(result['attempts'], 3)
        self.assertEqual(result['completed'], 2)
        self.assertEqual(result['rows'][0]['raw_ipa_tokens'], [])
        self.assertEqual(result['rows'][1]['observation_status'], 'FAILED_NO_RETRY')
        self.assertTrue(all(r[k] is False for r in result['rows'] for k in stage.recipe.FLAGS))

    def make_observers(self, partial=False):
        parent = self.root / ('partial-observers' if partial else 'observers')
        parent.mkdir()
        for provider in stage.PROVIDERS:
            calls = []
            def backend(wav, folder, provider=provider):
                calls.append(wav.name)
                if partial and provider == 'qwen' and len(calls) == 2:
                    raise ValueError('invented failure')
                return (['invented'], 160) if provider == 'phones' else 'invented'
            stage.observe(provider, self.bundle, self.pins, parent / ('observer-' + provider), backend=backend)
        return parent

    def test_join_all_missing_still_preserves_144_and_unknown_counts(self):
        result = stage.combine(self.bundle, self.root / 'missing', self.pins, allow_simulated=True)
        self.assertEqual(len(result['rows']), 144)
        self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
        self.assertTrue(all(v['attempts'] is None for v in result['providers'].values()))
        self.assertEqual(result['rows'][0]['observations']['qwen']['observation_status'], 'UNKNOWN_MISSING_PROVIDER_OUTPUT')
        self.assertEqual(result['rows'][3]['observations']['qwen']['observation_status'], 'NOT_ELIGIBLE_SOURCE')

    def test_join_rejects_cross_run_and_binds_partial_model_lock(self):
        parent = self.make_observers(partial=True)
        result = stage.combine(self.bundle, parent, self.pins, allow_simulated=True)
        self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
        self.assertEqual(result['providers']['qwen']['completed'], 2)
        model = parent / 'observer-qwen/model-lock.json'
        raw = model.read_bytes()
        model.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'MODEL_LOCK'):
            stage.combine(self.bundle, parent, self.pins, allow_simulated=True)
        model.write_bytes(raw)
        for path in (parent / 'observer-whisper').rglob('*.json'):
            if path.name in ('SOURCE_LEDGER.json', 'model-lock.json'):
                continue
            value = json.loads(path.read_bytes())
            if 'observer_run_id' in value:
                value['observer_run_id'] = 'invented-different-run'
                path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'CROSS_RUN'):
            stage.combine(self.bundle, parent, self.pins, allow_simulated=True)

    def test_missing_summary_does_not_claim_completion_from_good_rows(self):
        parent = self.make_observers()
        (parent / 'observer-qwen/summary.json').unlink()
        result = stage.combine(self.bundle, parent, self.pins, allow_simulated=True)
        self.assertEqual(result['status'], 'INCOMPLETE_NO_RETRY')
        self.assertEqual(result['providers']['qwen']['completed'], 3)
        self.assertEqual(result['providers']['qwen']['status'], 'UNKNOWN_MISSING_PROVIDER_TERMINAL')
        self.assertIs(result['observer_summaries_sha256']['qwen'], None)

    def test_default_and_template_are_disabled_zero_ml(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(stage.main([]), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value['status'], 'SOURCE_CONTRACT_ONLY')
        self.assertEqual(value['model_calls'], 0)
        template = stage.pins_template()
        self.assertFalse(template['enabled'])
        self.assertIsNone(template['valid_cells'])
        self.assertIsNone(template['max_calls_per_provider'])
        self.assertFalse({'numpy', 'torch', 'onnxruntime', 'transformers'} & set(sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
