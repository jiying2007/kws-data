"""Standard-library/mock contracts only: zero model, processor or resampler calls."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_kokoro_context64_generate as g
r = g.recipe
VOCAB = {' ': 16, '.': 4, '?': 6, 'A': 24, 'I': 25, 'O': 31, 'b': 44, 'd': 46,
    'f': 48, 'h': 50, 'i': 51, 'j': 52, 'k': 53, 'l': 54, 'm': 55, 'n': 56,
    'p': 58, 's': 61, 't': 62, 'u': 63, 'v': 64, 'w': 65, 'z': 68, 'æ': 72,
    'ð': 81, 'ŋ': 112, 'ɑ': 69, 'ɔ': 76, 'ə': 83, 'ɛ': 86, 'ɜ': 87, 'ɪ': 102,
    'ɹ': 123, 'ʊ': 135, 'ʌ': 138, 'ʤ': 82, 'ʧ': 133, 'ˈ': 156}


def fixture_plan():
    # Only historical exclusions are read from the shipped plan. These are real
    # frozen past hashes; synthetic future run identities below are mock-only.
    disk = json.loads((HERE / 'english_kokoro_context64_plan.json').read_text())
    return {'campaign_id': r.CAMPAIGN_ID, 'recipe_sha256': r.digest(r.recipe()),
        'maximum_tts_calls': 48, 'maximum_parent_count': 48, 'maximum_child_count': 64,
        'maximum_resample_calls': 48, 'generation_deadline_seconds': 1800,
        'retry_count': 0, 'diagnostic_only': True, 'training_allowed': False,
        'automatic_expansion': False, 'total_high_level_model_calls_max': 336,
        'enabled': False, 'main_before': g.MAIN_BEFORE,
        'source_exposure_exclusion': copy.deepcopy(disk['source_exposure_exclusion']),
        'source_exposure_exclusion_sha256': g.EXPOSURE_SHA256,
        'generator_code_sha256': g.sha(Path(g.__file__).read_bytes()),
        'recipe_code_sha256': g.sha(Path(r.__file__).read_bytes()),
        'helper_sha256': {'automation/english_kokoro48_generate.py': r.LEGACY_GENERATOR_SHA256,
            'automation/english_kokoro48_recipe.py': r.LEGACY_RECIPE_SHA256,
            'automation/english_synthetic192_training_core.py': r.GEOMETRY_HELPER_SHA256}}


def fake_rows(parent_count=1):
    frozen = r.recipe()
    parents = [g._unmeasured(x, 'parent') for x in frozen['parents'][:parent_count]]
    ids = {p['id'] for p in parents}
    children = [g._unmeasured(x, 'child') for x in frozen['children'] if x['parent_id'] in ids]
    return parents, children


def fixture_duration(parent):
    return [2] * (len(parent['raw_phonemes']) + 2)


class FakeTensor:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value


class FakeNativeResult:
    def __init__(self, parent):
        self.audio = FakeTensor(object())
        self.pred_dur = FakeTensor(fixture_duration(parent))


def fake_parent_bytes(output, parent, duration):
    """Explicit synthetic integer bytes, not synthesized audio or a resample."""
    n = 600 * sum(duration)
    parent.update(g.duration_geometry(parent, duration, n))
    samples = n * 2 // 3
    pcm = b''.join(struct.pack('<h', ((i * 17 + parent['seed']) % 65536) - 32768) for i in range(samples))
    folder = 'source/parents/' + parent['id'] + '/'
    parent.update(pcm_path=folder + 'parent-16k.pcm', wav_path=folder + 'parent-16k.wav',
        pcm_samples=samples, pcm_sha256=g.sha(pcm), sample_rate=16000, channels=1, sample_width=2,
        duration_path=folder + 'pred_dur.json')
    g.put(g.within(output, parent['pcm_path']), pcm)
    parent['wav_sha256'] = g.put(g.within(output, parent['wav_path']), g.native._wav(pcm, 16000))
    parent['duration_sha256'] = g.once(g.within(output, parent['duration_path']), {'fixture_only': True,
        'pred_dur': duration, 'phonemes': parent['raw_phonemes']})
    return pcm


def fake_parent_writer(output, parent, audio, duration, ids, counters, exposure, seen):
    # The production numerical writer is replaced in these ledger fixtures.
    parent['resample_attempts'] = 1
    counters['actual_resample_attempts'] += 1
    parent['resample_returned'] = True
    counters['actual_resample_returns'] += 1
    pcm = fake_parent_bytes(output, parent, duration)
    g._check_new_hash(parent['pcm_sha256'], exposure['old_pcm_sha256'], seen['pcm'], 'FIXTURE_PARENT_DUPLICATE')
    return pcm


class RecipeContracts(unittest.TestCase):
    def test_fixed_full_denominators_and_identity(self):
        frozen = r.recipe()
        self.assertEqual((len(frozen['parents']), len(frozen['children'])), (48, 64))
        for kind in ('positive', 'contrast', 'ordinary', 'outside'):
            self.assertEqual(sum(row['kind'] == kind for row in frozen['children']), 16)
        self.assertEqual(len({row['id'] for row in frozen['parents'] + frozen['children']}), 112)
        for row in frozen['parents'] + frozen['children']:
            self.assertEqual(row['role'], 'train')
            self.assertTrue(row['diagnostic_only'])
            self.assertFalse(row['training_allowed'])
            self.assertIsNone(row['human_speaker_identity'])
            self.assertEqual(row['source_group_id'], 'kokoro-v1_0-preset:' + row['preset_id'])
        self.assertTrue(frozen['permanently_excluded_from_training'])
        self.assertEqual(frozen['all_model_calls_max'], 336)

    def test_carrier_span_and_known_symbols(self):
        self.assertEqual(len(r.CARRIER_PHONEMES), 125)
        self.assertEqual(r.CARRIER_PHONEMES[78:101], r.CARRIER['boundary_control']['phonemes'])
        for p in r.recipe()['parents']:
            ids = g.phoneme_ids(p['raw_phonemes'], VOCAB)
            self.assertEqual(len(ids), len(p['raw_phonemes']) + 2)
            self.assertEqual((ids[0], ids[-1]), (0, 0))
            self.assertLessEqual(len(ids), 512)
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_PHONEME_TOKEN'):
            g.phoneme_ids('x', VOCAB)

    def test_native_assets_unchanged(self):
        self.assertEqual(r.assets(), g.native.recipe.assets())
        self.assertEqual(r.RUNTIME_VERSIONS, g.native.recipe.RUNTIME_VERSIONS)
        self.assertEqual(r.MODEL, g.native.recipe.MODEL)
        self.assertEqual(g.source_preflight(fixture_plan())['source_check'], 'PASS')

    def test_no_old_requests_and_fixed_exposure(self):
        plan = fixture_plan()
        g.validate_plan(plan)
        old = g.exclusion_sets(plan)
        self.assertEqual(len(old['old_pcm_sha256']), 64)
        self.assertEqual(len(old['old_native_full_f32_sha256']), 48)
        self.assertFalse({p['request_sha256'] for p in r.recipe()['parents']} & old['old_request_sha256'])
        plan['source_exposure_exclusion']['old_pcm_sha256'].pop()
        with self.assertRaisesRegex(ValueError, 'EXACT_RECORDED_OLD_EXPOSURE'):
            g.validate_plan(plan)

    def test_fixed_plan_cannot_enable_by_import(self):
        plan = fixture_plan()
        with self.assertRaisesRegex(ValueError, 'PLAN_ENABLED'):
            g.validate_plan(plan, executing=True)
        for key, value in [('maximum_tts_calls', 49), ('maximum_child_count', 63),
                           ('training_allowed', True), ('retry_count', 1), ('generation_deadline_seconds', 900)]:
            changed = copy.deepcopy(plan); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                g.validate_plan(changed)

    def test_public_guard_requires_exact_main_first_attempt(self):
        plan = fixture_plan(); plan['enabled'] = True
        with tempfile.TemporaryDirectory() as temp:
            event = Path(temp) / 'event.json'
            event.write_text(json.dumps({'before': g.MAIN_BEFORE, 'after': 'a' * 40,
                'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}))
            env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_SHA': 'a' * 40,
                'GITHUB_RUN_ID': '1', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_EVENT_NAME': 'push',
                'GITHUB_ACTIONS': 'true', 'GITHUB_WORKFLOW_REF': g.WORKFLOW,
                'GITHUB_REF': 'refs/heads/main', 'GITHUB_EVENT_PATH': str(event)}
            run = {'repository': env['GITHUB_REPOSITORY'], 'commit': env['GITHUB_SHA'],
                'run_id': '1', 'run_attempt': '1', 'event': 'push'}
            self.assertEqual(g.public_guard(plan, run, env), run)
            for key, value in [('GITHUB_ACTIONS', 'false'), ('GITHUB_RUN_ATTEMPT', '2'),
                               ('GITHUB_REF', 'refs/heads/local'), ('GITHUB_EVENT_NAME', 'workflow_dispatch')]:
                changed = dict(env); changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    g.public_guard(plan, run, changed)


class IntegerAndByteGeometry(unittest.TestCase):
    def setUp(self):
        self.p, self.children = fake_rows()
        self.p = self.p[0]
        self.d = fixture_duration(self.p)
        self.n = 600 * sum(self.d)

    def test_exact_duration_boundary_formula_nonuniform(self):
        self.d[0] = 6
        self.d[78], self.d[101], self.d[125] = 7, 9, 5
        geo = g.duration_geometry(self.p, self.d, 600 * sum(self.d))
        self.assertEqual(geo['target_start_prediction_native'], 600 * (3 + sum(self.d[1:126])))
        self.assertEqual(geo['target_start_parent16k'], 400 * (3 + sum(self.d[1:126])))
        self.assertEqual(geo['boundary_end_prediction_native'], 600 * (3 + sum(self.d[1:102])))
        self.assertFalse(geo['duration_boundary_is_acoustic_gold'])

    def test_invalid_duration_or_total_rejected(self):
        for change in ('length', 'zero', 'bool', 'float', 'large'):
            d = self.d.copy()
            if change == 'length': d.pop()
            if change == 'zero': d[10] = 0
            if change == 'bool': d[10] = True
            if change == 'float': d[10] = 2.0
            if change == 'large': d[10] = 51
            with self.subTest(change=change), self.assertRaises(ValueError):
                g.duration_geometry(self.p, d, self.n)
        with self.assertRaisesRegex(ValueError, 'TOTAL_SAMPLES'):
            g.duration_geometry(self.p, self.d, self.n + 1)

    def test_main_and_outside_fixed_coordinates(self):
        main, outside = self.children
        a = g.child_bounds(self.p, main, self.d, self.n)
        b = g.child_bounds(self.p, outside, self.d, self.n)
        self.assertEqual(a['child_start_native_sample'], 600 * sum(self.d[1:126]) - 3600)
        self.assertEqual(a['child_end_native_sample'], self.n)
        self.assertEqual(a['child_end_parent16k'], self.n * 2 // 3)
        self.assertEqual(b['child_start_native_sample'], 600 * sum(self.d[1:79]) - 3600)
        self.assertEqual(b['child_end_native_sample'], 600 * sum(self.d[1:102]) + 3600)
        self.assertLessEqual(b['child_end_parent16k'], b['target_start_parent16k'])
        self.assertEqual(b['child_resample_calls'], 0)

    def test_child_parent_and_character_tamper_rejected(self):
        for key, value in [('parent_id', 'fixture-not-the-parent'), ('child_character_range', [124, 135]),
                           ('end_rule', 'new-adaptive-rule')]:
            c = copy.deepcopy(self.children[0]); c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                g.child_bounds(self.p, c, self.d, self.n)

    def test_integer_support_matches_frozen_core_without_models(self):
        spec = importlib.util.spec_from_file_location('context64_fixture_old_geometry', HERE / 'english_synthetic192_training_core.py')
        core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
        for samples in (6400, 16000, 22000, 28000, 30400, 32000, 100000):
            value = g.whole_support_geometry(samples)
            expected = core.positive_ends(38400, samples, 38400 + samples + 16000)
            self.assertEqual(value['whole_support_embedding_end_samples'], expected)
            self.assertEqual(value['geometry_usable_positive'], len(expected) >= 3)
            self.assertFalse(value['geometry_audio_generated'])

    def test_saved_child_is_exact_parent_bytes_and_valid_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pcm = fake_parent_bytes(root, self.p, self.d)
            c = self.children[0]
            g.write_child_audio(root, self.p, c, self.d, pcm, g.exclusion_sets(fixture_plan()), {'pcm': {self.p['pcm_sha256']}})
            observed = (root / c['pcm_path']).read_bytes()
            self.assertEqual(observed, pcm[2 * c['child_start_parent16k']:2 * c['child_end_parent16k']])
            with wave.open(io.BytesIO((root / c['wav_path']).read_bytes()), 'rb') as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 16000))
                self.assertEqual(wav.readframes(wav.getnframes()), observed)
            proof = json.loads((root / c['derivation_path']).read_text())
            self.assertEqual(proof['parent_pcm_sha256'], g.sha(pcm))
            self.assertEqual(proof['child_resample_calls'], 0)
            self.assertFalse(proof['boundary_is_acoustic_gold'])

    def test_child_reuse_keeps_actual_bytes_but_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pcm = fake_parent_bytes(root, self.p, self.d)
            c = self.children[0]; b = g.child_bounds(self.p, c, self.d, self.n)
            old = g.exclusion_sets(fixture_plan())
            digest = g.sha(pcm[2 * b['child_start_parent16k']:2 * b['child_end_parent16k']])
            old['old_pcm_sha256'].add(digest)
            with self.assertRaisesRegex(g.SourceIntegrityError, 'CHILD_PCM_REUSE'):
                g.write_child_audio(root, self.p, c, self.d, pcm, old, {'pcm': set()})
            self.assertTrue((root / c['pcm_path']).is_file())
            self.assertEqual(c['pcm_sha256'], digest)

    def test_new_parent_and_child_share_one_uniqueness_namespace(self):
        seen = set(); old = set()
        g._check_new_hash('fixture-sha', old, seen, 'collision')
        with self.assertRaisesRegex(g.SourceIntegrityError, 'collision'):
            g._check_new_hash('fixture-sha', old, seen, 'collision')


class NativeCallLedger(unittest.TestCase):
    def _run(self, root, parents, children, forward, writer=fake_parent_writer):
        counters = g._counters()
        with mock.patch.object(g, 'write_parent_audio', side_effect=writer):
            g._run_parents(root, parents, children, forward, VOCAB, time.monotonic() + 60,
                           counters, g.exclusion_sets(fixture_plan()))
        return counters

    def test_two_children_share_one_parent_forward(self):
        parents, children = fake_rows()
        calls = []
        def forward(parent, ids):
            self.assertTrue((root / 'source/REQUESTS.json').exists())
            calls.append(parent['id']); return FakeNativeResult(parent)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); g.once(root / 'source/REQUESTS.json', {'fixture_only': True})
            counters = self._run(root, parents, children, forward)
            self.assertEqual((len(calls), counters['actual_forward_attempts'], counters['actual_forward_returns']), (1, 1, 1))
            self.assertEqual((counters['completed_parents'], counters['completed_children']), (1, 2))
            self.assertEqual((counters['actual_resample_attempts'], counters['actual_resample_returns']), (1, 1))
            for p in parents:
                self.assertEqual(json.loads((root / 'source/parents' / p['id'] / 'COMPLETED.json').read_text())['row'], p)
            for c in children:
                self.assertEqual(json.loads((root / 'source/children' / c['id'] / 'COMPLETED.json').read_text())['row'], c)
                self.assertEqual(c['child_model_calls'], 0)

    def test_native_exception_consumes_attempt_not_return(self):
        parents, children = fake_rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def broken(*args): raise RuntimeError('fixture-forward-failure')
            counters = self._run(root, parents, children, broken)
            self.assertEqual((counters['actual_forward_attempts'], counters['actual_forward_returns']), (1, 0))
            self.assertEqual(counters['completed_children'], 0)
            self.assertEqual(parents[0]['generation_status'], 'failed')
            self.assertTrue(all(c['generation_status'] == 'not_started' for c in children))
            failed = json.loads((root / 'source/parents' / parents[0]['id'] / 'FAILED.json').read_text())
            self.assertEqual((failed['forward_attempts'], failed['forward_returned']), (1, False))

    def test_materialization_failure_still_records_native_return(self):
        parents, children = fake_rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            counters = self._run(root, parents, children, lambda *args: object())
            self.assertEqual((counters['actual_forward_attempts'], counters['actual_forward_returns']), (1, 1))
            self.assertTrue(parents[0]['forward_returned'])
            self.assertEqual(counters['actual_resample_attempts'], 0)
            self.assertTrue((root / 'source/parents' / parents[0]['id'] / 'FORWARD_RETURNED.json').exists())

    def test_resample_return_then_save_failure_not_unconsumed(self):
        parents, children = fake_rows()
        def broken(output, row, audio, duration, ids, counters, exposure, seen):
            row.update(resample_attempts=1, resample_returned=True)
            counters['actual_resample_attempts'] += 1; counters['actual_resample_returns'] += 1
            raise OSError('fixture-save-after-return')
        with tempfile.TemporaryDirectory() as temp:
            counters = self._run(Path(temp), parents, children, lambda p, ids: FakeNativeResult(p), broken)
            self.assertEqual((counters['actual_forward_attempts'], counters['actual_forward_returns']), (1, 1))
            self.assertEqual((counters['actual_resample_attempts'], counters['actual_resample_returns']), (1, 1))
            self.assertEqual((counters['completed_parents'], counters['completed_children']), (0, 0))

    def test_child_save_failure_does_not_repeat_parent(self):
        parents, children = fake_rows(); calls = []
        def forward(p, ids): calls.append(p['id']); return FakeNativeResult(p)
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(g, 'write_child_audio', side_effect=OSError('fixture-child-write')):
            counters = self._run(Path(temp), parents, children, forward)
            self.assertEqual(len(calls), 1)
            self.assertEqual(counters['completed_parents'], 1)
            self.assertEqual(counters['completed_children'], 0)
            self.assertTrue(all(c['derivation_attempts'] == 1 for c in children))
            self.assertTrue(all((Path(temp) / 'source/children' / c['id'] / 'FAILED.json').is_file() for c in children))

    def test_unknown_token_before_call_not_counted(self):
        parents, children = fake_rows(); parents[0]['raw_phonemes'] += 'x'
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            counters = self._run(Path(temp), parents, children, lambda *a: calls.append(a))
            self.assertEqual(counters['actual_forward_attempts'], 0)
            self.assertFalse(calls)
            self.assertEqual(parents[0]['generation_status'], 'not_started')

    def test_signal_like_interrupt_aborts_remaining_calls(self):
        parents, children = fake_rows(2); counters = g._counters(); calls = []
        def stopped(p, ids): calls.append(p['id']); raise g.GenerationInterrupted('fixture-stop')
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(g.GenerationInterrupted):
            g._run_parents(Path(temp), parents, children, stopped, VOCAB, time.monotonic() + 60,
                          counters, g.exclusion_sets(fixture_plan()))
        self.assertEqual(len(calls), 1)
        self.assertEqual((counters['actual_forward_attempts'], counters['actual_forward_returns']), (1, 0))
        self.assertEqual(parents[1]['forward_attempts'], 0)

    def test_real_entry_load_failure_preserves_all_rows_with_mock_native(self):
        plan = fixture_plan(); plan['enabled'] = True
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def fail_load(output, private):
                requests = json.loads((output / 'source/REQUESTS.json').read_text())
                self.assertEqual((len(requests['parents']), len(requests['children'])), (48, 64))
                raise RuntimeError('fixture-initialization-failure')
            with mock.patch.object(g, 'public_guard'), mock.patch.object(g, '_load_native', side_effect=fail_load):
                result = g.generate(plan, root, run={'fixture_only': True})
            self.assertEqual((len(result['parents']), len(result['children'])), (48, 64))
            self.assertEqual((result['actual_forward_attempts'], result['actual_forward_returns']), (0, 0))
            self.assertFalse(result['complete']); self.assertFalse(result['training_allowed'])
            self.assertEqual(len(list((root / 'source/parents').glob('*/FAILED.json'))), 48)
            self.assertEqual(len(list((root / 'source/children').glob('*/FAILED.json'))), 64)
            self.assertFalse((root / 'source/_private_model').exists())

    def test_consumed_one_then_interrupt_preserves_full_denominators(self):
        plan = fixture_plan(); plan['enabled'] = True; called = []
        def forward(p, ids): called.append(p['id']); raise g.GenerationInterrupted('fixture-terminal-stop')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(g, 'public_guard'), mock.patch.object(g, '_load_native', return_value=(forward, VOCAB)):
                result = g.generate(plan, root, run={'fixture_only': True})
            self.assertEqual(len(called), 1)
            self.assertEqual((result['actual_forward_attempts'], result['actual_forward_returns']), (1, 0))
            self.assertEqual(sum(p['forward_attempts'] for p in result['parents']), 1)
            self.assertEqual(len(result['parents']), 48); self.assertEqual(len(result['children']), 64)
            self.assertTrue(all((root / 'source/children' / c['id'] / 'FAILED.json').exists() for c in result['children']))

    def test_production_resample_and_model_calls_are_single_explicit_sites(self):
        tree = ast.parse(Path(g.__file__).read_text())
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        resamples = [n for n in ast.walk(funcs['write_parent_audio']) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == 'resample_poly']
        self.assertEqual(len(resamples), 1)
        models = [n for n in ast.walk(funcs['_load_native']) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == 'model']
        self.assertEqual(len(models), 1)
        for name in ('write_child_audio', 'duration_geometry', 'child_bounds', 'whole_support_geometry'):
            self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(funcs[name])))
        self.assertFalse(any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and
            isinstance(t.value, ast.Name) and t.value.id == 'native' for t in n.targets) for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
