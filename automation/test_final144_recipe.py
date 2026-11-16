"""Pure planning and artificial capture-interface tests; zero real model calls."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final144_recipe as batch
sys.path.insert(0, str(HERE.parent / 'kws-data/automation'))
try:
    import english_qwen_capture as existing_capture
except ModuleNotFoundError as exc:
    # The sparse read-only checkout lacks an unrelated planner dependency.
    # Stub only that unused type; the actual capture and JSON validation code run.
    if exc.name not in ('kws_data_contract', 'kws_data_contract.identity'):
        raise
    absent = types.ModuleType('kws_data_contract.identity')
    def unused_planner_type(*args, **kwargs):
        raise AssertionError('The generic planner must not be invoked by this interface test')
    absent.SourceRoles = unused_planner_type
    with patch.dict(sys.modules, {'kws_data_contract.identity': absent}):
        import english_qwen_capture as existing_capture


class FakeMedia:
    def _once(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as dst:
            dst.write(batch.canonical(value))

    def quality(self, samples, rate):
        return {'explicitly_artificial': True}


class FakeBackend:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def synthesize(self, cell):
        self.calls.append(cell)
        if len(self.calls) == self.fail_at:
            raise RuntimeError('Deliberate artificial failure')
        return b'No actual audio', 24000


def fake_writer(folder, samples, rate, technical):
    return {'native_sha256': '1' * 64, 'wav_sha256': '2' * 64, 'pcm_sha256': '3' * 64,
            'audio_frames': 1600, 'wav_bytes': 3244, 'sample_rate_hz': 16000}


def completed_test_pins():
    pins = json.loads(batch.PIN_FILE.read_text())
    for name, value in list(pins.items()):
        if value is None:
            pins[name] = (123 if name.endswith('_id') else 'a' * 40
                          if name == 'next_main_before' else 'b' * 64)
    pins['pilot_review_status'] = 'USEFUL_FOR_WEAK_RESEARCH'
    pins['pilot_machine_review_useful'] = True
    return pins


class BatchTests(unittest.TestCase):
    def test_exact_144_roles_intents_and_unique_cells(self):
        recipe = batch.recipe()
        cells = recipe['cells']
        self.assertEqual(len(cells), 144)
        self.assertEqual(len({c['cell_id'] for c in cells}), 144)
        self.assertEqual(Counter(c['role'] for c in cells), {'train': 96, 'development': 48})
        self.assertEqual(Counter(c['intent'] for c in cells),
                         {'wake-candidate': 96, 'ordinary-negative-candidate': 48})
        self.assertEqual(Counter(c['voice_id'] for c in cells), {v: 24 for v, _ in batch.VOICES})
        self.assertEqual(sum(recipe['capture_stage_budgets']), 144)
        self.assertTrue(all(x == 16 for x in recipe['capture_stage_budgets']))
        self.assertTrue(all(all(c[k] is False for k in batch.FLAGS) for c in cells))
        self.assertTrue(all(c['requested_keyword_index'] == {'Hey Vosen': 1, 'Hey Mero': 2}[c['canonical_phrase']]
                            for c in cells if c['intent'] == 'wake-candidate'))

    def test_family_split_has_no_overlap_and_is_not_human_test(self):
        cells = batch.recipe()['cells']
        train = {c['source_family_id'] for c in cells if c['role'] == 'train'}
        development = {c['source_family_id'] for c in cells if c['role'] == 'development'}
        self.assertFalse(train & development)
        self.assertEqual(development, {'qwen3-preset-ryan', 'qwen3-preset-vivian'})
        self.assertTrue(all(c['independent_human'] is False for c in cells))
        self.assertTrue(all(c['family_exposed_in_rendering_pilot']
                            for c in cells if c['voice_id'] in ('aiden', 'ryan')))

    def test_negatives_are_ordinary_intents_not_nearphone_or_gold(self):
        cells = batch.recipe()['cells']
        negative = [c for c in cells if c['intent'] == 'ordinary-negative-candidate']
        self.assertEqual({c['synthesis_text'] for c in negative}, set(batch.ORDINARY_NEGATIVES))
        self.assertTrue(all(c['canonical_phrase'] is None and c['requested_keyword_index'] is None
                            and c['trusted_negative'] is False for c in negative))
        for cell in negative:
            self.assertFalse(any(x in cell['synthesis_text'].lower()
                                 for x in ('mero', 'vosen', 'meero', 'vozen', 'miro', 'rosen')))

    def test_rendering_selection_records_meero_and_original_vosen(self):
        wake = [c for c in batch.recipe()['cells'] if c['intent'] == 'wake-candidate']
        self.assertTrue(all('Meero' in c['synthesis_text'] for c in wake if c['canonical_phrase'] == 'Hey Mero'))
        self.assertTrue(all('Vosen' in c['synthesis_text'] and 'Vozen' not in c['synthesis_text']
                            for c in wake if c['canonical_phrase'] == 'Hey Vosen'))
        self.assertTrue(all(c['pronunciation_verified'] is False for c in wake))

    def test_pending_pins_reject_and_fixed_source_pins_are_preserved(self):
        pins = json.loads(batch.PIN_FILE.read_text())
        self.assertEqual(pins['pilot_run_id'], 43809825038)
        self.assertEqual(pins['pilot_source_artifact_id'], 14199609338)
        self.assertEqual(pins['pilot_head_sha'], '0641690f2af5468992485319d04ba7c2b4fe6a61')
        with self.assertRaisesRegex(ValueError, 'MISSING_OR_INVALID_PIN'):
            batch.validate_execution_pins({**pins, 'next_main_before': None})
        filled = dict(pins)
        if filled['next_main_before'] is None:
            filled['next_main_before'] = filled['next_main_before_candidate']
        batch.validate_execution_pins(filled)
        with self.assertRaisesRegex(ValueError, 'FIRST_PUBLIC_MAIN_SUCCESSOR_ONLY'):
            batch.execution_guard(filled, {}, {})

    def test_completed_pin_shape_still_requires_useful_review_and_unadmitted_flags(self):
        pins = completed_test_pins()
        batch.validate_execution_pins(pins)
        for key, value in [('pilot_review_json_sha256', None), ('recipe_sha256', '0' * 64),
                           ('pilot_review_status', 'PENDING'), ('pilot_machine_review_useful', False),
                           ('formal_training_admitted', True), ('max_tts_calls', 145),
                           ('pilot_source_artifact_id', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                batch.validate_execution_pins({**pins, key: value})

    def test_first_successor_guard_rejects_private_fork_and_rerun(self):
        pins = completed_test_pins()
        env = {'GITHUB_REPOSITORY': 'jiying2007/kws-data', 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'c' * 40}
        event = {'before': pins['next_main_before'], 'after': 'c' * 40,
                 'repository': {'full_name': 'jiying2007/kws-data', 'private': False}}
        batch.execution_guard(pins, event, env)
        for key, value in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_REPOSITORY', 'fork/repo')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                batch.execution_guard(pins, event, {**env, key: value})
        with self.assertRaises(ValueError):
            batch.execution_guard(pins, {**event, 'repository': {
                'full_name': 'jiying2007/kws-data', 'private': True}}, env)

    def test_existing_capture_accepts_recipe_through_small_backend_adapter(self):
        plan = batch.recipe()
        backend = FakeBackend()
        adapter = batch.RenderingCaptureAdapter(backend)
        with tempfile.TemporaryDirectory() as d, \
             patch.object(existing_capture, 'media', return_value=FakeMedia()):
            root = Path(d) / 'capture'
            receipts = [existing_capture.capture(plan, root, adapter, max_calls=16,
                                                 writer=fake_writer) for _ in range(9)]
            self.assertEqual([r['attempts_this_call'] for r in receipts], [16] * 9)
            self.assertEqual((receipts[-1]['total_started'], receipts[-1]['total_completed']), (144, 144))
            self.assertEqual(len(backend.calls), 144)
            self.assertEqual([c['phrase'] for c in backend.calls],
                             [c['synthesis_text'] for c in plan['cells']])
            self.assertTrue(all(set(c) == {'cell_id', 'phrase', 'speaker', 'instruction'}
                                for c in backend.calls))

    def test_existing_capture_failure_is_consumed_without_retry(self):
        backend = FakeBackend(fail_at=1)
        adapter = batch.RenderingCaptureAdapter(backend)
        with tempfile.TemporaryDirectory() as d, \
             patch.object(existing_capture, 'media', return_value=FakeMedia()):
            root = Path(d) / 'capture'
            receipts = [existing_capture.capture(batch.recipe(), root, adapter, max_calls=16,
                                                 writer=fake_writer) for _ in range(9)]
            extra = existing_capture.capture(batch.recipe(), root, adapter, max_calls=16,
                                             writer=fake_writer)
            self.assertEqual(receipts[-1]['total_completed'], 143)
            self.assertEqual(extra['attempts_this_call'], 0)
            self.assertEqual(len(backend.calls), 144)
            self.assertEqual(len(list(root.glob('cells/*/failed.json'))), 1)


if __name__ == '__main__':
    unittest.main()
