"""Source-only fixtures. No model package, network or real audio is loaded."""
from __future__ import annotations

import ast
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kokoro48_generator_fixture", HERE / "english_kokoro48_generate.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
r = g.recipe

# Actual subset of the pinned 2351-byte config used by these 48 requests.
VOCAB = {' ': 16, '.': 4, '?': 6, 'A': 24, 'I': 25, 'O': 31, 'b': 44,
    'd': 46, 'f': 48, 'h': 50, 'i': 51, 'j': 52, 'k': 53, 'l': 54, 'm': 55,
    'n': 56, 'p': 58, 's': 61, 't': 62, 'u': 63, 'v': 64, 'w': 65, 'z': 68,
    'æ': 72, 'ð': 81, 'ŋ': 112, 'ɑ': 69, 'ɔ': 76, 'ə': 83, 'ɛ': 86, 'ɜ': 87,
    'ɪ': 102, 'ɹ': 123, 'ʊ': 135, 'ʌ': 138, 'ʤ': 82, 'ʧ': 133, 'ˈ': 156}


def plan(enabled=False):
    return {"enabled": enabled, "campaign_id": r.CAMPAIGN_ID,
        "recipe_sha256": r.digest(r.recipe()), "maximum_tts_calls": 48,
        "retry_count": 0, "diagnostic_only": True, "training_allowed": False,
        "generation_deadline_seconds": 900}


def rows():
    return [g._unmeasured(row) for row in r.recipe()["rows"]]


def counters():
    return {"actual_forward_attempts": 0, "actual_forward_returns": 0, "completed": 0}


def fake_measurement(output, row, audio, duration, ids):
    # Deliberately fake fixture data; never a real artifact or model result.
    return {"fixture_only": True, "pcm_samples": 16000, "diagnostic_only": True,
            "training_allowed": False, "human_gold": False}


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.shape = (1,)
        self.dtype = "float32"
        self.device = SimpleNamespace(type="cpu")


FAKE_TORCH = SimpleNamespace(Tensor=FakeTensor, equal=lambda a, b: a.value == b.value)


class FakeModule:
    def __init__(self, values):
        self.values = dict(values)
        self.buffers = {}
        self._non_persistent_buffers_set = set()
    def state_dict(self):
        return dict(self.values)
    def named_parameters(self, **kwargs):
        return list(self.values.items())
    def named_modules(self, **kwargs):
        return [('', self)]
    def named_buffers(self, **kwargs):
        return list(self.buffers.items())


def checkpoint_fixture():
    checkpoint = {name: {"weight": FakeTensor(index)}
        for index, name in enumerate(("bert", "bert_encoder", "predictor", "text_encoder", "decoder"))}
    model = SimpleNamespace(**{name: FakeModule(values)
                             for name, values in checkpoint.items()})
    return model, checkpoint


class RecipeAndGeometry(unittest.TestCase):
    def test_import_and_preflight_have_no_heavy_top_level_import(self):
        tree = ast.parse((HERE / "english_kokoro48_generate.py").read_text())
        imported = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported += [x.name.split('.')[0] for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module.split('.')[0])
        self.assertFalse(set(imported).intersection({"torch", "kokoro", "numpy", "scipy", "transformers"}))
        g.validate_plan(plan())

    def test_exact_48_preset_mode_hypothesis_denominators(self):
        source = r.recipe()["rows"]
        self.assertEqual(len(source), 48)
        self.assertEqual(len({x['id'] for x in source}), 48)
        self.assertEqual(Counter(x['mode'] for x in source), {"short": 16, "carrier": 16, "ordinary": 16})
        self.assertEqual(Counter(x['requested_class'] for x in source), {0: 16, 1: 16, 2: 16})
        self.assertEqual(Counter(x['preset_id'] for x in source), dict.fromkeys(r.PRESETS, 12))
        self.assertEqual(r.PRESETS, ('af_bella', 'af_sarah', 'am_michael', 'am_puck'))
        self.assertTrue(all(x['role'] == 'train' and x['diagnostic_only'] and not x['training_allowed']
                            and x['human_speaker_identity'] is None for x in source))
        self.assertFalse(r.recipe()['auto_admission_allowed'])
        self.assertEqual(r.recipe()['all_model_calls_max'], 288)

    def test_carrier_is_101_and_all_frozen_tokens_exist(self):
        self.assertEqual(len(r.CARRIER_PHONEMES), 101)
        for row in r.recipe()['rows']:
            ids = g.phoneme_ids(row['raw_phonemes'], VOCAB)
            self.assertEqual(len(ids), len(row['raw_phonemes']) + 2)
            if row['mode'] == 'carrier':
                self.assertIn(len(row['raw_phonemes']), (111, 112))
                self.assertEqual(row['target_start_char'], 101)

    def test_exact_native_A_O_ids_include_bos_eos(self):
        self.assertEqual(g.phoneme_ids('hˈA mˈɛɹO.', VOCAB),
                         [0, 50, 156, 24, 16, 55, 156, 86, 123, 31, 4, 0])

    def test_unknown_is_not_silently_filtered(self):
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_PHONEME_TOKEN'):
            g.phoneme_ids('hˈA ❓mˈɛɹO.', VOCAB)
        with self.assertRaisesRegex(ValueError, 'PHONEME_TOKEN_ID_RANGE'):
            g.phoneme_ids('h', {'h': 0})

    def test_context_length_accounts_for_bos_eos(self):
        self.assertEqual(len(g.phoneme_ids('h' * 510, VOCAB)), 512)
        for value in ('', 'h' * 511):
            with self.assertRaisesRegex(ValueError, 'PHONEME_CONTEXT_LENGTH'):
                g.phoneme_ids(value, VOCAB)

    def test_pair_seed_same_and_no_extra_mode_draw(self):
        source = r.recipe()['rows']
        paired = {}
        for row in source:
            if row['mode'] == 'ordinary':
                continue
            key = (row['preset_id'], row['cell_id'])
            paired.setdefault(key, []).append(row['seed'])
        self.assertEqual(len(paired), 16)
        self.assertTrue(all(len(x) == 2 and x[0] == x[1] for x in paired.values()))

    def test_carrier_nonuniform_duration_exact_character_boundary(self):
        row = next(x for x in r.recipe()['rows'] if x['mode'] == 'carrier')
        # Prefix tokens have varying durations; this catches off-by-one slices.
        d = [5] + [1 + (i % 3) for i in range(len(row['raw_phonemes']))] + [7]
        size = sum(d) * 600
        crop = g.crop_bounds(row, d, size)
        prefix_sum = sum(d[i] for i in range(1, 102))
        self.assertEqual(crop['predicted_target_start_native_sample'], (2 + prefix_sum) * 600)
        self.assertEqual(crop['crop_start_native_sample'], (2 + prefix_sum) * 600 - 3600)
        self.assertEqual(crop['crop_end_native_sample'], size)
        self.assertEqual(crop['target_character_range'], [101, len(row['raw_phonemes'])])
        self.assertFalse(crop['boundary_is_acoustic_gold'])
        self.assertFalse(crop['human_word_end'])
        self.assertTrue(crop['prediction_rule']['preceding_space_duration_counted_in_full'])

    def test_short_and_ordinary_keep_all_native_samples(self):
        for row in (x for x in r.recipe()['rows'] if x['mode'] != 'carrier'):
            d = [12] + [2] * len(row['raw_phonemes']) + [9]
            crop = g.crop_bounds(row, d, 600 * sum(d))
            self.assertEqual(crop['crop_start_native_sample'], 0)
            self.assertEqual(crop['crop_end_native_sample'], 600 * sum(d))
            self.assertEqual(crop['actual_added_silence_samples'], 0)

    def test_bos_less_than_offset_clamps_without_erasing_prefix(self):
        row = next(x for x in r.recipe()['rows'] if x['mode'] == 'carrier')
        d = [1] + [2] * len(row['raw_phonemes']) + [1]
        crop = g.crop_bounds(row, d, 600 * sum(d))
        self.assertEqual(crop['predicted_target_start_native_sample'], 600 * 202)

    def test_invalid_duration_and_native_length_fail(self):
        row = r.recipe()['rows'][0]
        d = [2] * (len(row['raw_phonemes']) + 2)
        for bad in (d[:-1], [0] + d[1:], [True] + d[1:], [51] + d[1:]):
            with self.assertRaisesRegex(ValueError, 'NATIVE_DURATION_VECTOR'):
                g.crop_bounds(row, bad, 600 * sum(d))
        with self.assertRaisesRegex(ValueError, 'NATIVE_DURATION_SAMPLE_GEOMETRY'):
            g.crop_bounds(row, d, 600 * sum(d) - 1)

    def test_altered_carrier_is_not_recropped(self):
        row = copy.deepcopy(next(x for x in r.recipe()['rows'] if x['mode'] == 'carrier'))
        row['raw_phonemes'] = 'h' + row['raw_phonemes'][1:]
        d = [2] * (len(row['raw_phonemes']) + 2)
        with self.assertRaisesRegex(ValueError, 'EXACT_PREREGISTERED_CARRIER'):
            g.crop_bounds(row, d, 600 * sum(d))

    def test_disabled_or_budget_changed_plan_cannot_execute(self):
        with self.assertRaisesRegex(ValueError, 'PLAN_ENABLED'):
            g.validate_plan(plan(), executing=True)
        for key, value in (('maximum_tts_calls', 49), ('retry_count', 1),
                           ('training_allowed', True), ('diagnostic_only', False),
                           ('recipe_sha256', '0' * 64), ('maximum_tts_calls', True)):
            changed = plan(True)
            changed[key] = value
            with self.assertRaises(ValueError):
                g.validate_plan(changed, executing=True)


class PartialAccounting(unittest.TestCase):
    def setUp(self):
        # _run_rows' fake callback returns a plain pair, explicitly replacing
        # only native-tensor materialization in these source-only fixtures.
        self.output_patch = patch.object(g, 'materialize_native_result', side_effect=lambda x: x)
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)

    def test_all_48_single_fake_calls_are_counted(self):
        source, count, calls = rows(), counters(), []
        def fake_forward(row, ids):
            calls.append(row['id'])
            return object(), []
        with tempfile.TemporaryDirectory(prefix='kokoro-fixture-') as tmp, patch.object(g, 'write_audio', fake_measurement):
            g._run_rows(Path(tmp), source, fake_forward, VOCAB, time.monotonic() + 10, count)
            self.assertEqual(count, {'actual_forward_attempts': 48, 'actual_forward_returns': 48, 'completed': 48})
            self.assertEqual(len(set(calls)), 48)
            self.assertEqual(len(list(Path(tmp).glob('source/audio/*/STARTED.json'))), 48)
            self.assertTrue(all(x['generation_status'] == 'success' and not x['training_allowed'] for x in source))

    def test_native_failure_is_not_retried(self):
        source, count, calls = rows(), counters(), []
        first = source[0]['id']
        def fake_forward(row, ids):
            calls.append(row['id'])
            if row['id'] == first:
                raise RuntimeError('fixture_native_failure')
            return object(), []
        with tempfile.TemporaryDirectory(prefix='kokoro-fixture-') as tmp, patch.object(g, 'write_audio', fake_measurement):
            g._run_rows(Path(tmp), source, fake_forward, VOCAB, time.monotonic() + 10, count)
            self.assertEqual(count, {'actual_forward_attempts': 48, 'actual_forward_returns': 47, 'completed': 47})
            self.assertEqual(calls.count(first), 1)
            fail = json.loads((Path(tmp) / 'source/audio' / first / 'FAILED.json').read_text())
            self.assertEqual(fail['forward_attempts'], 1)
            self.assertFalse(fail['forward_returned'])

    def test_write_failure_keeps_returned_forward_in_count(self):
        source, count = rows(), counters()
        first = source[0]['id']
        def measure(output, row, audio, duration, ids):
            if row['id'] == first:
                raise OSError('fixture_disk_failure')
            return fake_measurement(output, row, audio, duration, ids)
        with tempfile.TemporaryDirectory(prefix='kokoro-fixture-') as tmp, patch.object(g, 'write_audio', measure):
            g._run_rows(Path(tmp), source, lambda row, ids: (object(), []), VOCAB, time.monotonic() + 10, count)
            self.assertEqual(count, {'actual_forward_attempts': 48, 'actual_forward_returns': 48, 'completed': 47})
            fail = json.loads((Path(tmp) / 'source/audio' / first / 'FAILED.json').read_text())
            self.assertTrue(fail['forward_returned'])
            self.assertEqual(fail['retry_count'], 0)

    def test_unknown_before_forward_consumes_zero_calls(self):
        source, count, calls = rows()[:1], counters(), []
        source[0]['raw_phonemes'] += '❓'
        with tempfile.TemporaryDirectory(prefix='kokoro-fixture-') as tmp:
            g._run_rows(Path(tmp), source, lambda *args: calls.append(1), VOCAB, time.monotonic() + 10, count)
            self.assertEqual(calls, [])
            self.assertEqual(count, counters())
            self.assertEqual(source[0]['generation_status'], 'not_started')

    def test_malformed_return_still_counts_a_returned_forward(self):
        source, count = rows()[:1], counters()
        with tempfile.TemporaryDirectory(prefix='kokoro-fixture-') as tmp:
            g._run_rows(Path(tmp), source, lambda *args: None, VOCAB, time.monotonic() + 10, count)
            self.assertEqual(count, {'actual_forward_attempts': 1, 'actual_forward_returns': 1, 'completed': 0})
            failure = json.loads((Path(tmp) / 'source/audio' / source[0]['id'] / 'FAILED.json').read_text())
            self.assertTrue(failure['forward_returned'])

    def test_load_failure_retains_all_48_not_started(self):
        with tempfile.TemporaryDirectory(prefix='kokoro-fixture-') as tmp, \
             patch.object(g, '_load_native', side_effect=RuntimeError('fixture_load_failure')):
            result = g.generate(plan(True), Path(tmp), run={'run_attempt': '1', 'fixture_only': True})
            self.assertEqual(result['actual_forward_attempts'], 0)
            self.assertEqual(result['completed'], 0)
            self.assertEqual(len(result['rows']), 48)
            self.assertTrue(all(x['generation_status'] == 'not_started' for x in result['rows']))
            self.assertEqual(len(list(Path(tmp).glob('source/audio/*/FAILED.json'))), 48)
            self.assertFalse(result['auto_admission_allowed'])

    def test_interruption_keeps_one_attempt_and_47_unstarted(self):
        def interrupted(row, ids):
            raise KeyboardInterrupt('fixture_interrupted')
        with tempfile.TemporaryDirectory(prefix='kokoro-fixture-') as tmp, \
             patch.object(g, '_load_native', return_value=(interrupted, VOCAB)):
            result = g.generate(plan(True), Path(tmp), run={'run_attempt': '1', 'fixture_only': True})
            self.assertEqual(result['actual_forward_attempts'], 1)
            self.assertEqual(result['actual_forward_returns'], 0)
            self.assertEqual(Counter(x['generation_status'] for x in result['rows']), {'failed': 1, 'not_started': 47})
            self.assertEqual(len(list(Path(tmp).glob('source/audio/*/FAILED.json'))), 48)

    def test_existing_source_blocks_repeat_before_loader(self):
        with tempfile.TemporaryDirectory(prefix='kokoro-fixture-') as tmp, patch.object(g, '_load_native') as loader:
            (Path(tmp) / 'source').mkdir()
            (Path(tmp) / 'source' / 'old-receipt').write_text('fixture')
            with self.assertRaisesRegex(ValueError, 'EMPTY_CREATE_ONCE_DIRECTORY'):
                g.generate(plan(True), Path(tmp), run={'run_attempt': 1})
            loader.assert_not_called()


class CheckpointCompleteness(unittest.TestCase):
    def test_complete_checkpoint_and_explicit_module_prefix(self):
        model, checkpoint = checkpoint_fixture()
        self.assertEqual(len(g._strict_checkpoint_readback(model, checkpoint, FAKE_TORCH)), 5)
        prefixed = {name: {'module.' + k: v for k, v in values.items()} for name, values in checkpoint.items()}
        proof = g._strict_checkpoint_readback(model, prefixed, FAKE_TORCH)
        self.assertTrue(all(x['mapping'] == 'strip_exact_module_dot_prefix' for x in proof))

    def test_missing_or_changed_weights_do_not_use_strict_false(self):
        for mutation in ('component', 'key', 'value', 'mixed_prefix'):
            model, checkpoint = checkpoint_fixture()
            checkpoint = {name: dict(values) for name, values in checkpoint.items()}
            if mutation == 'component':
                del checkpoint['bert']
            elif mutation == 'key':
                checkpoint['bert']['extra'] = FakeTensor(1)
            elif mutation == 'value':
                checkpoint['bert']['weight'] = FakeTensor(-1)
            else:
                checkpoint['bert']['module.other'] = FakeTensor(1)
            with self.assertRaises(ValueError):
                g._strict_checkpoint_readback(model, checkpoint, FAKE_TORCH)

    def test_key_failure_preserves_all_five_component_diagnostics(self):
        model, checkpoint = checkpoint_fixture()
        checkpoint['bert']['fixture_unexpected'] = FakeTensor(7)
        with tempfile.TemporaryDirectory(prefix='checkpoint-fixture-') as tmp:
            with self.assertRaisesRegex(ValueError, 'CHECKPOINT_EXACT_KEYS'):
                g._strict_checkpoint_readback(model, checkpoint, FAKE_TORCH, output=Path(tmp))
            summary = json.loads((Path(tmp)/'source/weight-diagnostics/SUMMARY.json').read_text())
            self.assertEqual(summary['component_count'], 5)
            self.assertEqual(summary['passed_components'], 4)
            self.assertEqual(summary['failed_components'], ['bert'])
            self.assertFalse(summary['passed'])
            self.assertEqual(len(summary['component_records']), 5)
            for record in summary['component_records']:
                raw = (Path(tmp)/record['path']).read_bytes()
                self.assertEqual(record['sha256'], g.sha(raw))
                self.assertEqual(record['bytes'], len(raw))
            detail = json.loads((Path(tmp)/'source/weight-diagnostics/bert.json').read_text())
            self.assertEqual(detail['unexpected_in_checkpoint'], ['fixture_unexpected'])
            self.assertEqual(detail['missing_in_checkpoint'], [])
            self.assertTrue(detail['common_key_comparisons'][0]['exact_value_equal'])
            self.assertEqual(detail['allowed_key_exceptions'], [])

    def test_value_mismatch_is_retained_and_never_reloaded(self):
        model, checkpoint = checkpoint_fixture()
        checkpoint['decoder']['weight'] = FakeTensor(-99)
        with tempfile.TemporaryDirectory(prefix='checkpoint-fixture-') as tmp:
            with self.assertRaisesRegex(ValueError, 'CHECKPOINT_EXACT_TENSOR_READBACK'):
                g._strict_checkpoint_readback(model, checkpoint, FAKE_TORCH, output=Path(tmp))
            detail = json.loads((Path(tmp)/'source/weight-diagnostics/decoder.json').read_text())
            self.assertEqual(detail['mismatched_common_keys'], ['weight'])
            self.assertFalse(detail['tensor_values_reloaded'])
            self.assertEqual(model.decoder.values['weight'].value, 4)

    def test_nonpersistent_buffer_inventory_is_not_a_key_exception(self):
        model, checkpoint = checkpoint_fixture()
        model.bert.buffers['fixture_buffer'] = FakeTensor(3)
        model.bert._non_persistent_buffers_set.add('fixture_buffer')
        checkpoint['bert']['fixture_buffer'] = FakeTensor(3)
        with tempfile.TemporaryDirectory(prefix='checkpoint-fixture-') as tmp:
            with self.assertRaisesRegex(ValueError, 'CHECKPOINT_EXACT_KEYS'):
                g._strict_checkpoint_readback(model, checkpoint, FAKE_TORCH, output=Path(tmp))
            detail = json.loads((Path(tmp)/'source/weight-diagnostics/bert.json').read_text())
            self.assertFalse(detail['named_buffers'][0]['persistent'])
            self.assertEqual(detail['unexpected_in_checkpoint'], ['fixture_buffer'])
            self.assertEqual(detail['allowed_key_exceptions'], [])


class RuntimeDependencyRepair(unittest.TestCase):
    def test_only_runtime_versions_changed_from_actual_unstarted_requests(self):
        # Canonical digests read from run 44114592216 source/RECIPE.json.
        # Its 48 requests all stopped before any forward or PCM existed.
        source = r.recipe()
        self.assertEqual(r.digest(source['rows']),
            'c40c65690e5d12196e0c95a28047a6f5181d4e18278c9fde641827b13e9cc3fe')
        self.assertEqual(r.digest({k: v for k, v in source.items() if k != 'runtime_versions'}),
            '89467c9db4d5132c6d867f2f9e6aea8822a4ca2ec4bdb24b3fba5c601779349f')

    def test_exact_click_and_observed_typer_pins(self):
        self.assertEqual(r.RUNTIME_VERSIONS, {
            'kokoro': '0.9.4', 'torch': '2.9.1+cpu', 'numpy': '1.26.4',
            'scipy': '1.15.3', 'transformers': '4.46.3', 'huggingface-hub': '0.36.2',
            'misaki': '0.9.4', 'loguru': '0.7.3', 'spacy': '3.7.5',
            'spacy-curated-transformers': '0.2.2', 'attrs': '25.3.0',
            'click': '8.3.1', 'typer': '0.29.1', 'typer-slim': '0.24.0'})

    def test_missing_or_changed_dependency_stops_before_assets_and_all_forwards(self):
        # Mock metadata only. This verifies the guard and zero-attempt ledger;
        # it is not installation, an import smoke test, or model execution.
        for bad_name, bad_value in (('click', None), ('click', '8.5.1'),
                                    ('typer', '0.24.0'), ('typer-slim', '0.23.2')):
            with self.subTest(package=bad_name, version=bad_value):
                def version(name):
                    if name == bad_name:
                        if bad_value is None:
                            raise g.importlib.metadata.PackageNotFoundError(name)
                        return bad_value
                    return r.RUNTIME_VERSIONS[name]
                with tempfile.TemporaryDirectory(prefix='kokoro-runtime-fixture-') as tmp, \
                     patch.object(g.importlib.metadata, 'version', side_effect=version), \
                     patch.object(g.importlib.metadata, 'distribution') as distribution, \
                     patch.object(g, '_download') as download:
                    result = g.generate(plan(True), Path(tmp), run={'run_attempt': '1', 'fixture_only': True})
                    self.assertEqual(result['actual_forward_attempts'], 0)
                    self.assertEqual(result['actual_forward_returns'], 0)
                    self.assertEqual(result['completed'], 0)
                    self.assertEqual(len(result['rows']), 48)
                    self.assertTrue(all(x['generation_status'] == 'not_started' for x in result['rows']))
                    self.assertEqual(len(list(Path(tmp).glob('source/audio/*/FAILED.json'))), 48)
                    distribution.assert_not_called()
                    download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
