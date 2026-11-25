"""B27 one-variable and fail-closed execution contracts; stdlib fixtures only."""
from __future__ import annotations
from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_natural_weight as train
import english_v6_b27_producer as producer


def closed_opening():
    return dict(schema='english-v6-b27-after-original-b24q1-no-promotion-v1', condition=train.CONDITION,
        enabled=False, quality_lock=None, failure_adapter=None, baseline_challenger_pt_sha256=train.CONTROL_PT,
        actual_primary_evaluator_sha256=None, actual_primary_producer_manifest_sha256=None,
        original_failure_report_sha256=None)


def closed_binding():
    return dict(schema='english-v6-b27-prospective-assessment-binding-v1',
        recipe_path=train.DESIGN.relative_to(train.ROOT).as_posix(), recipe_sha256=train.DESIGN_SHA,
        frozen=False, optimizer_allowed=False, assessment=None, files={}, original_B24Q1_source_exclusion=None)


class ScientificContracts(unittest.TestCase):
    def test_one_exact_preexisting_scalar_and_complete_frozen_budget(self):
        recipe = train.design()
        before, after = train.b24.design(), train.scientific_recipe()
        self.assertEqual(set(before), set(after))
        self.assertEqual({key for key in before if before[key] != after[key]}, {'natural_weight'})
        self.assertEqual((before['natural_weight'], after['natural_weight']), (0.08, 0.16))
        self.assertEqual(recipe['proposal']['sha256'], train.PROPOSAL_SHA)
        self.assertEqual(recipe['budget'], before['budget'])
        self.assertEqual((recipe['optimizer_updates'], recipe['maximum_new_model_forwards']), (192, 1847))
        self.assertEqual((after['union_retention_conditions'], len(after['input_archives'])), (122, 18))
        self.assertEqual(recipe['initial_pt_sha256'], train.b24.INITIAL_PT)
        self.assertNotEqual(train.INITIAL_PT, train.CONTROL_PT)
        for name in ('old_B25_condition_changed', 'old_B26_condition_changed', 'Q1_metrics_used_to_choose_parameters',
                     'Q1_assessment_or_budget_changed', 'parameter_specific_user_approval_claimed'):
            self.assertIs(recipe[name], False)

    def test_exact_numerical_code_object_and_one_private_scalar(self):
        engine, original = train.fit_engine(), train.b24.b23.fit
        self.assertIs(engine.__code__, original.__code__)
        self.assertIs(engine.__closure__, original.__closure__)
        self.assertEqual(set(engine.__globals__), set(original.__globals__))
        self.assertEqual({key for key in original.__globals__ if engine.__globals__[key] is not original.__globals__[key]}, {'old'})
        profile = engine.__globals__['old']
        self.assertEqual((profile.NATURAL_WEIGHT, train.b24.old.NATURAL_WEIGHT), (0.16, 0.08))
        for key, value in vars(train.b24.old).items():
            if key != 'NATURAL_WEIGHT': self.assertIs(getattr(profile, key), value, key)
        self.assertIs(engine.__globals__['positive_loss'], train.b24.b23.positive_loss)
        self.assertIs(engine.__globals__['prefix_peak_suffix_loss'], train.b24.b23.prefix_peak_suffix_loss)

    def test_original_export_wrapper_calls_one_fit_and_preserves_exact_bytes(self):
        initial, rest, runtime, calls = object(), [object() for _ in range(6)], [object() for _ in range(3)], []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            def fake(*args):
                calls.append(args)
                self.assertEqual(args, (initial, *rest, output, *runtime))
                (output / 'models').mkdir()
                files = {}
                for suffix in ('.pt', '.npz', '.onnx'):
                    name = 'mero43-b23-challenger192' + suffix
                    raw = ('fixture-' + suffix).encode()
                    (output / 'models' / name).write_bytes(raw)
                    files[name] = dict(bytes=len(raw), sha256=train.sha(raw))
                return dict(parent_pt_sha256=train.b24.old.B0_PT, optimizer_updates=192,
                            actual_new_model_forwards=1847, files=files)
            with patch.object(train, 'fit_engine', return_value=fake):
                result = train.fit(initial, *rest, output, *runtime)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['natural_weight'], 0.16)
            self.assertEqual(result['parent_pt_sha256'], train.INITIAL_PT)
            self.assertEqual(result['historical_control_pt_sha256'], train.CONTROL_PT)
            self.assertFalse(result['restored_optimizer_state'])
            self.assertEqual(set(result['files']), {train.STEM + ext for ext in ('.pt', '.npz', '.onnx')})
            for name, pin in result['files'].items():
                self.assertEqual(train.sha((output / 'models' / name).read_bytes()), pin['sha256'])

    def test_control_copy_uses_original_b24_and_never_initializer(self):
        payloads = {'challenger/models/mero43-b24-challenger192' + ext: ('B24-' + ext).encode()
                    for ext in ('.pt', '.npz', '.onnx')}
        receipt = dict(files={Path(name).name: dict(bytes=len(raw), sha256=train.sha(raw)) for name, raw in payloads.items()})
        body = io.BytesIO()
        with zipfile.ZipFile(body, 'w') as archive:
            for name, raw in payloads.items(): archive.writestr(name, raw)
        @contextmanager
        def original(_directory):
            with zipfile.ZipFile(io.BytesIO(body.getvalue())) as archive:
                yield archive, dict(challenger=receipt)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'models'
            with patch.object(train, 'original_models', side_effect=original):
                result = train.historical_control('/fixture', output)
            self.assertEqual((result['new_forwards'], result['new_optimizer_updates']), (0, 0))
            for name, raw in payloads.items(): self.assertEqual((output / Path(name).name).read_bytes(), raw)

    def test_wrong_old_model_bytes_reject_before_open_or_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'TRAIN_models.zip').write_bytes(b'not-original')
            with patch.object(train.failure, 'opened', side_effect=AssertionError('before interpretation')):
                with self.assertRaises(ValueError):
                    with train.original_models(temporary): pass


class ActivationContracts(unittest.TestCase):
    def test_three_transport_guards_reuse_original_code_in_private_globals(self):
        for name in ('guard', 'consume', 'verify_consumed'):
            new, old = train._reuse(name), getattr(train.b24, name)
            self.assertIs(new.__code__, old.__code__)
            self.assertIsNot(new.__globals__, old.__globals__)
            self.assertIs(new.__globals__['source_preflight'], train.source_preflight)
            self.assertEqual(new.__globals__['MARKER'], train.MARKER)
            self.assertEqual(old.__globals__['MARKER'], train.b24.MARKER)
        with self.assertRaises(ValueError): train._reuse('execute')

    def test_all_scientific_entries_close_before_io_when_opening_is_false(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opening = root / 'opening.json'; opening.write_bytes(train.encoded(closed_opening()))
            with (patch.object(train, 'OPENING', opening), patch.object(train, 'gh', side_effect=AssertionError('no network')),
                  patch.object(train.failure, 'verify_failure_original', side_effect=AssertionError('no proof execution'))):
                for operation in (lambda: train.fetch(root / 'new', 'a' * 40),
                                  lambda: train.consume(root / 'new', 'a' * 40),
                                  lambda: train.execute(root / 'new', root / 'output', 'a' * 40),
                                  lambda: train.source_preflight(root / 'new')):
                    with self.assertRaisesRegex(ValueError, 'sealed'): operation()
            self.assertFalse((root / 'new').exists())
            self.assertFalse((root / 'output').exists())

    def test_fresh_assessment_cannot_be_opened_by_boolean_flip_or_optimizer_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'binding.json'
            with patch.object(train, 'BINDING', path):
                value = closed_binding(); path.write_bytes(train.encoded(value))
                self.assertFalse(train.assessment_binding(False)['frozen'])
                with self.assertRaisesRegex(ValueError, 'sealed'): train.assessment_binding()
                for key in ('frozen', 'optimizer_allowed'):
                    value = closed_binding(); value[key] = True; path.write_bytes(train.encoded(value))
                    with self.assertRaises(ValueError): train.assessment_binding(False)

    def test_guard_still_requires_exact_first_public_workflow_only_child(self):
        parent, head = 'a' * 40, 'b' * 40
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=train.REPO, GITHUB_EVENT_NAME='push',
            GITHUB_REF=train.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=head, GITHUB_RUN_ID='123')
        event = dict(before=parent, after=head, ref=train.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=train.REPO, private=False))
        with patch.object(train, 'design'), patch.object(train, 'opening'), patch.object(train, 'assessment_binding'):
            identity = train.guard(parent, env, event, parent)
            self.assertEqual(identity['run_branch'], train.BRANCH)
            for key, value in (('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_REF', 'refs/heads/main'), ('GITHUB_EVENT_NAME', 'workflow_dispatch')):
                altered = dict(env, **{key: value})
                with self.assertRaises(ValueError): train.guard(parent, altered, event, parent)
            for key, value in (('before', 'c' * 40), ('forced', True), ('deleted', True)):
                with self.assertRaises(ValueError): train.guard(parent, env, dict(event, **{key: value}), parent)

    def test_complete_producer_and_exact_inert_or_two_pin_activated_workflow(self):
        names = producer.paths(False)
        self.assertIn('automation/english_v6_b27_training_readback.py', names)
        self.assertIn('automation/english_v6_b24_source_training_producer.json', names)
        original = json.loads((train.HERE / 'english_v6_b24_source_training_producer.json').read_bytes())
        self.assertTrue(set(original['files']) <= names)
        actual = (train.ROOT / train.WORKFLOW).read_bytes()
        template = (train.ROOT / producer.TEMPLATE).read_bytes()
        self.assertEqual(template.count(producer.TOKEN), 2)
        parent = re.search(rb"PREPARED_PARENT: '([^']+)'", actual).group(1)
        self.assertEqual(actual, template if parent == producer.TOKEN else producer.workflow(parent.decode()))
        self.assertNotIn(b'workflow_dispatch', template)
        self.assertIn(b'no-promotion/originals/', template)
        self.assertIn(b'EXTERNAL_INPUTS.json', template)
        self.assertNotIn(b'no-promotion/inputs/', template)
        actual_pins = {name: dict(bytes=(train.ROOT / name).stat().st_size) for name in names}
        self.assertLess(producer.capacity_receipt(actual_pins)['zip_bound'], train.capacity.CORE_LIMIT)
        with self.assertRaises(ValueError): producer.capacity_receipt({'over': dict(bytes=128_000_000)})

    def test_large_q1_source_remains_external_and_historical_caps_stay_fixed(self):
        q1 = {name: dict(bytes=1, sha256='a' * 64) for name in train.capacity.PRIMARY_FILES}
        q1['inputs/source.zip']['bytes'] = 367_799_776
        pins = train.design()['input_archives']
        old = {name: dict(bytes=1, sha256=pin['sha256']) for name, pin in pins.items()}
        def inventory(path): return dict(files=q1 if Path(path).name == 'no-promotion' else old, directories=[])
        with patch.object(train.capacity, 'inventory', side_effect=inventory):
            result = train.external_inputs('/fixture')
            self.assertEqual(len(result['files']), 30)
            self.assertEqual(result['files']['no-promotion/inputs/source.zip']['bytes'], 367_799_776)
            q1['inputs/pair.zip']['bytes'] = 128_000_001
            with self.assertRaises(ValueError): train.external_inputs('/fixture')


if __name__ == '__main__':
    unittest.main()
