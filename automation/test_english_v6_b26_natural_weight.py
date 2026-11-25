"""Inert B26 scientific contrast and complete original failure contracts.

All raw-logit fixtures are generated stdlib bytes, never model calls or permits.
No test imports a tensor runtime, loads a checkpoint or registers a new source.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b26_natural_weight as train
import english_v6_b26_b24r1_failure as failure
import english_v6_b24r1_quality_readback as original_reader
import test_english_v6_b24r1_quality_readback as fixtures


def function(module, name):
    return next(node for node in ast.parse(Path(module.__file__).read_bytes()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def syntax(node):
    return ast.dump(node, include_attributes=False)


def failed_fixture():
    members, spec, admission, rows = fixtures.fixture()
    result = fixtures.reader.document(members['RESULT.json'])
    with zipfile.ZipFile(io.BytesIO(members['ALL_FRAME_LOGITS.npz'])) as archive:
        arrays = {name: archive.read(name) for name in archive.namelist()}
    row = result['records'][0]
    key = row['id'] + '__1.0__challenger'
    labels = [0] * 17
    raw, body = fixtures.npy(labels)
    arrays[key + '.npy'] = raw
    members['raw_outputs/' + key + '.npy'] = fixtures.npy(labels, True)[0]
    row['gains']['1.0']['challenger'] = dict(array=key, frames=17, sha256=train.sha(body), **fixtures.primary.ctc_events(labels))
    members['ALL_FRAME_LOGITS.npz'] = fixtures.zip_bytes(arrays)
    result.update(fixtures.decision.decide(result['records']))
    result.update(logits_sha256=train.sha(members['ALL_FRAME_LOGITS.npz']),
        replication_open_allowed=False, candidate_engineering_validation_allowed=False)
    members['RESULT.json'] = fixtures.reader.canonical(result)
    return members, spec, admission, rows


def lock(panel='primary'):
    def pin(number):
        return dict(id=number, run_id=123, run_head='a' * 40, bytes=100, sha256=str(number) * 64)
    return dict(schema='english-v6-b24r1-original-quality-lock-v1', panel=panel,
        result=pin(1), producer=pin(2), consumed=pin(3), evaluator_sha256=failure.PRIMARY_CODE_SHA,
        inputs_sha256='b' * 64, producer_manifest_sha256='c' * 64, original_result_sha256='d' * 64)


class ScientificContracts(unittest.TestCase):
    def test_exact_frozen_proposal_and_only_one_recipe_field_changes(self):
        design = train.design()
        before, after = train.b24.design(), train.scientific_recipe()
        self.assertEqual(set(before), set(after))
        self.assertEqual({key for key in before if before[key] != after[key]}, {'natural_weight'})
        self.assertEqual((before['natural_weight'], after['natural_weight']), (0.08, 0.16))
        self.assertEqual(design['proposal']['sha256'], train.PROPOSAL_SHA)
        self.assertEqual(design['initial_pt_sha256'], train.b24.INITIAL_PT)
        self.assertNotEqual(train.INITIAL_PT, train.CONTROL_PT)
        self.assertEqual(after['updates'], 192)
        self.assertEqual(after['budget']['total_new_model_forwards'], 1847)
        self.assertEqual(after['union_retention_conditions'], 122)
        self.assertFalse(design['old_B25_condition_changed'])
        self.assertFalse(design['scientific_execution_authorized'])

    def test_exact_original_fit_code_uses_private_one_scalar_profile(self):
        engine = train.fit_engine()
        original = train.b24.b23.fit
        self.assertIs(engine.__code__, original.__code__)
        self.assertIs(engine.__closure__, original.__closure__)
        self.assertEqual(set(engine.__globals__), set(original.__globals__))
        self.assertEqual({key for key in original.__globals__
            if engine.__globals__[key] is not original.__globals__[key]}, {'old'})
        profile = engine.__globals__['old']
        self.assertIsNot(profile, train.b24.old)
        self.assertEqual(profile.NATURAL_WEIGHT, 0.16)
        self.assertEqual(train.b24.old.NATURAL_WEIGHT, 0.08)
        for key, value in vars(train.b24.old).items():
            if key != 'NATURAL_WEIGHT':
                self.assertIs(getattr(profile, key), value, key)
        self.assertIs(engine.__globals__['positive_loss'], train.b24.b23.positive_loss)
        self.assertIs(engine.__globals__['prefix_peak_suffix_loss'], train.b24.b23.prefix_peak_suffix_loss)

    def test_exact_b24_export_wrapper_calls_one_candidate_fit_and_preserves_bytes(self):
        initial = {'saved-b23-state': object()}
        other = [object() for _ in range(6)]
        runtime = [object() for _ in range(3)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            calls = []
            def fake_engine(*args):
                calls.append(args)
                self.assertIs(args[0], initial)
                self.assertEqual(args[1:7], tuple(other))
                self.assertIs(args[7], output)
                self.assertEqual(args[8:], tuple(runtime))
                directory = output / 'models'; directory.mkdir()
                files = {}
                for suffix in ('.pt', '.npz', '.onnx'):
                    name = 'mero43-b23-challenger192' + suffix
                    body = ('fixture-export' + suffix).encode()
                    (directory / name).write_bytes(body)
                    files[name] = dict(bytes=len(body), sha256=train.sha(body))
                return dict(parent_pt_sha256=train.b24.old.B0_PT, optimizer_updates=192,
                    actual_new_model_forwards=1847, files=files)
            with patch.object(train, 'fit_engine', return_value=fake_engine):
                result = train.fit(initial, *other, output, *runtime)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['natural_weight'], 0.16)
            self.assertEqual(result['parent_pt_sha256'], train.INITIAL_PT)
            self.assertEqual(result['historical_control_pt_sha256'], train.CONTROL_PT)
            self.assertEqual((result['new_teacher_forwards'], result['new_control_forwards']), (0, 0))
            self.assertFalse(result['restored_optimizer_state'])
            self.assertEqual(set(result['files']), {train.STEM + suffix for suffix in ('.pt', '.npz', '.onnx')})
            for name, pin in result['files'].items():
                self.assertEqual(train.sha((output / 'models' / name).read_bytes()), pin['sha256'])
            self.assertFalse(list((output / 'models').glob('*b23*')))
        self.assertEqual(train.b24.old.NATURAL_WEIGHT, 0.08)

    def test_historical_control_copies_b24_challenger_without_inference(self):
        members = {'challenger/models/mero43-b24-challenger192' + suffix: ('old-b24' + suffix).encode()
                   for suffix in ('.pt', '.npz', '.onnx')}
        receipt = dict(stem='challenger/models/mero43-b24-challenger192',
            files={Path(name).name: dict(bytes=len(raw), sha256=train.sha(raw)) for name, raw in members.items()})
        @contextmanager
        def original(_directory):
            with zipfile.ZipFile(io.BytesIO(fixtures.zip_bytes(members))) as archive:
                yield archive, dict(challenger=receipt)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'control'
            with patch.object(train, 'original_models', side_effect=original):
                result = train.historical_control('/fixture', output)
            self.assertEqual(result['new_forwards'], 0)
            self.assertEqual(result['new_optimizer_updates'], 0)
            for name, raw in members.items():self.assertEqual((output / Path(name).name).read_bytes(), raw)
        source = ast.get_source_segment(Path(train.__file__).read_text(), function(train, 'load_initial_state'))
        self.assertIn("'control/models/mero43-b23-challenger192'", source)
        self.assertIn('weights_only=True', source)
        self.assertNotIn('load_state_dict', source)
        self.assertNotIn('forward', [n.attr for n in ast.walk(function(train, 'load_initial_state')) if isinstance(n, ast.Attribute)])

    def test_changed_original_archive_rejects_before_interpretation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'TRAIN_models.zip'
            path.write_bytes(b'not-an-original')
            with patch.object(failure, 'opened', side_effect=AssertionError('must reject before ZIP open')):
                with self.assertRaises(ValueError):
                    with train.original_models(temporary):pass

    def test_no_workflow_no_roster_and_closed_binding(self):
        self.assertFalse(train.opening(False)['enabled'])
        self.assertFalse(train.assessment_binding(False)['frozen'])
        self.assertFalse((train.HERE / 'english_v6_b26_panels').exists())
        self.assertFalse(list((train.ROOT / '.github/workflows').glob('*b26*')))
        with self.assertRaisesRegex(ValueError, 'not implemented or registered'):train.assessment_binding()
        with tempfile.TemporaryDirectory() as temporary:
            value = json.loads(train.BINDING.read_bytes()); value['frozen'] = True
            path = Path(temporary) / 'binding.json'; path.write_text(json.dumps(value))
            with patch.object(train, 'BINDING', path):
                with self.assertRaises(ValueError):train.assessment_binding(False)

    def test_opening_blocks_before_any_quality_reader_or_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'uncreated'
            with patch.object(failure, 'verify_failure_original', side_effect=AssertionError('closed')):
                with self.assertRaisesRegex(ValueError, 'sealed'):train.source_preflight(destination)
            self.assertFalse(destination.exists())


class OriginalFailureContracts(unittest.TestCase):
    def test_saved_reader_diff_is_only_primary_verdict_and_permissions(self):
        self.assertEqual(train.sha(failure.BASE_READER_PATH.read_bytes()), failure.BASE_READER_SHA)
        before = ast.get_source_segment(Path(original_reader.__file__).read_text(), function(original_reader, 'saved_arrays'))
        after = ast.get_source_segment(Path(failure.__file__).read_text(), function(failure, 'saved_failure_arrays'))
        changes = (
            ('def saved_failure_arrays(', 'def saved_arrays('),
            ('Require exact NO_PROMOTION from every original raw return and unchanged B14.', 'Require an exact PASS from every original raw return and unchanged B14.'),
            ('    need(panel == "primary", "only original B24R1 PRIMARY failure is eligible")\n', ''),
            ('result["panel_gate"] == "NO_PROMOTION"', 'result["panel_gate"] == "PASS"'),
            ('only complete actual NO_PROMOTION; STOP, partial return, unknown consumption and PASS stay closed', 'only a complete actual PASS; STOP, partial return, unknown consumption or NO_PROMOTION stays closed'),
            ('result["replication_open_allowed"] is False', 'result["replication_open_allowed"] is (panel == "primary")'),
            ('result["candidate_engineering_validation_allowed"] is False', 'result["candidate_engineering_validation_allowed"] is (panel == "replication")'),
            ('failed candidate opens neither replication nor engineering', 'next-stage flags exactly match this original PASS'),
            ('replay["panel_gate"] == "NO_PROMOTION"', 'replay["panel_gate"] == "PASS"'),
            ('unchanged original B14 independently recomputes NO_PROMOTION', 'every unchanged original B14 condition independently recomputes to PASS'),
        )
        for left, right in changes:
            self.assertEqual(after.count(left), 1, left)
            after = after.replace(left, right)
        self.assertEqual(syntax(ast.parse(before)), syntax(ast.parse(after)))

    def test_provenance_source_and_all_raw_tensor_verification_are_preserved(self):
        before = copy.deepcopy(function(original_reader, 'verify_worker'))
        after = copy.deepcopy(function(failure, 'verify_failure_worker'))
        after.name = before.name
        for node in ast.walk(after):
            if isinstance(node, ast.Name) and node.id == 'saved_failure_arrays':node.id = 'saved_arrays'
        before.body[-1].body[-1] = after.body[-1].body[-1] = ast.Return(value=ast.Constant(value='verified-originals'))
        self.assertEqual(syntax(before), syntax(after))

    def test_complete_raw_failure_fixture_recomputes(self):
        members, spec, admission, rows = failed_fixture()
        value = failure.saved_failure_arrays(fixtures.zip_bytes(members), spec, 'primary', admission, rows,
            fixtures.primary, fixtures.decision)
        self.assertEqual(value['result']['panel_gate'], 'NO_PROMOTION')
        self.assertEqual(value['result']['actual_forwards'], 576)
        self.assertFalse(value['result']['replication_open_allowed'])
        self.assertFalse(value['result']['candidate_engineering_validation_allowed'])

    def test_pass_forged_summary_source_stop_partial_and_unknown_reject(self):
        for forged in (False, True):
            members, spec, admission, rows = fixtures.fixture()
            if forged:
                result = fixtures.reader.document(members['RESULT.json'])
                result.update(panel_gate='NO_PROMOTION', replication_open_allowed=False, candidate_engineering_validation_allowed=False)
                members['RESULT.json'] = fixtures.reader.canonical(result)
            with self.assertRaises(ValueError):
                failure.saved_failure_arrays(fixtures.zip_bytes(members), spec, 'primary', admission, rows, fixtures.primary, fixtures.decision)
        original, spec, admission, rows = failed_fixture()
        for kind in ('journal', 'raw', 'clock', 'unknown', 'stop', 'shortfall', 'partial', 'wrong_training_assessment'):
            members = dict(original)
            result = fixtures.reader.document(members['RESULT.json'])
            if kind == 'journal':del members[next(name for name in members if name.startswith('attempts/'))]
            if kind == 'raw':members[next(name for name in members if name.startswith('raw_outputs/'))] = fixtures.npy([0] * 17, True)[0]
            if kind == 'clock':result['records'][1]['gains']['1.0']['B0']['event_samples'] = [999]
            if kind == 'unknown':result['consumed_unknown'] = ['one']
            if kind == 'stop':result['panel_gate'] = 'STOP'
            if kind == 'shortfall':result['source_gate'] = 'SOURCE_GATE_STOP'
            if kind == 'partial':result['actual_forwards'] -= 1
            if kind == 'wrong_training_assessment':result['training_assessment_sha256'] = result['assessment_sha256']
            members['RESULT.json'] = fixtures.reader.canonical(result)
            with self.assertRaises(ValueError, msg=kind):
                failure.saved_failure_arrays(fixtures.zip_bytes(members), spec, 'primary', admission, rows, fixtures.primary, fixtures.decision)

    def test_only_primary_lock_and_exact_pre_result_reader_are_accepted(self):
        failure.validate_lock(lock())
        with self.assertRaises(ValueError):failure.validate_lock(lock('replication'))
        with patch.object(failure, 'BASE_READER_SHA', '0' * 64):
            with self.assertRaises(ValueError):failure.validate_lock(lock())
        members, spec, admission, rows = failed_fixture()
        with self.assertRaises(ValueError):
            failure.saved_failure_arrays(fixtures.zip_bytes(members), spec, 'replication', admission, rows, fixtures.primary, fixtures.decision)

    def test_reader_forbids_an_already_loaded_acoustic_runtime_before_worker(self):
        with patch.dict(sys.modules, {'torch': object()}), patch.object(failure.subprocess, 'check_output') as launch:
            with self.assertRaisesRegex(ValueError, 'runtime imports'):failure.verify_failure_original(Path('/fixture'), lock())
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
