"""Stdlib-only B24 closure, actual-parent and full failed-array contracts.

These generated array fixtures cannot authorize an actual experiment. No test
loads an acoustic checkpoint, imports a tensor runtime or calls a model.
"""
from __future__ import annotations
import ast
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24_b23_warm_start as train
import english_v6_b24_b23_failure as failure
import english_v6_b24_producer as producer
import english_v6_b23_quality_readback as original_reader
import test_english_v6_b23_quality_readback as fixtures


def function(module, name):
    return next(n for n in ast.parse(Path(module.__file__).read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


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


class Contracts(unittest.TestCase):
    def test_single_phase_initialization_and_unchanged_scientific_recipe(self):
        value = train.design()
        self.assertEqual(value['parent_pt_sha256'], train.INITIAL_PT)
        self.assertNotEqual(train.INITIAL_PT, train.old.B0_PT)
        self.assertEqual(value['budget'], train.b23.design()['budget'])
        self.assertEqual(value['historical_control']['pt_sha256'], train.INITIAL_PT)
        self.assertEqual(value['updates'], 192)
        self.assertEqual(value['budget']['total_new_model_forwards'], 1847)
        self.assertEqual(value['common_draw_sha256'], train.b23.design()['common_draw_sha256'])
        self.assertFalse(value['initialization']['exact_optimizer_continuation'])
        self.assertFalse(value['initialization']['old_B23_steps_reexecuted'])
        self.assertFalse(value['initialization']['restore_optimizer_state'])
        self.assertIs(train.prefix_peak_suffix_loss, train.b23.prefix_peak_suffix_loss)
        self.assertIs(train.positive_loss, train.b23.positive_loss)
        self.assertIs(train.Recorder, train.b23.Recorder)

    def test_wrapper_forwards_exact_warm_state_to_original_fit_once(self):
        initial = {'actual-b23-state': object()}
        others = [object() for _ in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            def original_fit(*args):
                self.assertIs(args[0], initial)
                self.assertEqual(args[1:7], tuple(others))
                self.assertIs(args[7], output)
                directory = output / 'models'
                directory.mkdir()
                files = {}
                for suffix in ('.pt', '.npz', '.onnx'):
                    name = 'mero43-b23-challenger192' + suffix
                    raw = ('unchanged-return' + suffix).encode()
                    (directory / name).write_bytes(raw)
                    files[name] = dict(bytes=len(raw), sha256=train.sha(raw))
                return dict(stem='challenger/models/mero43-b23-challenger192',
                    parent_pt_sha256=train.old.B0_PT, optimizer_updates=192, actual_new_model_forwards=1847, files=files)
            with patch.object(train.b23, 'fit', side_effect=original_fit) as fit:
                result = train.fit(initial, *others, output, None, None, None)
                fit.assert_called_once()
            self.assertEqual(result['parent_pt_sha256'], train.INITIAL_PT)
            self.assertFalse(result['restored_optimizer_state'])
            self.assertEqual(result['old_B23_optimizer_updates_reexecuted'], 0)
            self.assertEqual(set(result['files']), {train.STEM + e for e in ('.pt', '.npz', '.onnx')})
            for name, pin in result['files'].items():
                self.assertEqual(train.sha((output / 'models' / name).read_bytes()), pin['sha256'])
            self.assertFalse(list((output / 'models').glob('*b23*')))
        node = function(train, 'fit')
        calls = [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count('b23.fit'), 1)
        self.assertFalse(any('backward' in name or 'Adam' in name or 'load_state_dict' in name for name in calls))
        load = ast.get_source_segment(Path(train.__file__).read_text(), function(train, 'load_initial_state'))
        self.assertIn("'control/B23.zip'", load)
        self.assertIn('weights_only=True', load)
        self.assertNotIn('optimizer.load_state_dict', load)

    def test_exact_pre_result_reader_and_only_failed_verdict_changes(self):
        self.assertEqual(train.sha(failure.BASE_READER_PATH.read_bytes()), failure.BASE_READER_SHA)
        before = ast.get_source_segment(Path(original_reader.__file__).read_text(), function(original_reader, 'saved_arrays'))
        after = ast.get_source_segment(Path(failure.__file__).read_text(), function(failure, 'saved_failure_arrays'))
        replacements = (
            ('def saved_failure_arrays(', 'def saved_arrays('),
            ('Require exact NO_PROMOTION from every original raw return and unchanged B14.', 'Require an exact PASS from every original raw return and unchanged B14.'),
            ('result["panel_gate"] == "NO_PROMOTION"', 'result["panel_gate"] == "PASS"'),
            ('only complete actual NO_PROMOTION; STOP, partial return, unknown consumption and PASS stay closed', 'only a complete actual PASS; STOP, partial return, unknown consumption or NO_PROMOTION stays closed'),
            ('result["replication_open_allowed"] is False', 'result["replication_open_allowed"] is (panel == "primary")'),
            ('result["candidate_engineering_validation_allowed"] is False', 'result["candidate_engineering_validation_allowed"] is (panel == "replication")'),
            ('failed candidate cannot open any next-quality or engineering stage', 'next-stage flags exactly match this original PASS'),
            ('replay["panel_gate"] == "NO_PROMOTION"', 'replay["panel_gate"] == "PASS"'),
            ('unchanged B14 independently recomputes the original NO_PROMOTION', 'every unchanged original B14 condition independently recomputes to PASS'),
        )
        for left, right in replacements:
            self.assertEqual(after.count(left), 1)
            after = after.replace(left, right)
        self.assertEqual(syntax(ast.parse(before)), syntax(ast.parse(after)))

    def test_original_provenance_worker_is_unchanged_until_report(self):
        before = copy.deepcopy(function(original_reader, 'verify_worker'))
        after = copy.deepcopy(function(failure, 'verify_failure_worker'))
        after.name = before.name
        for node in ast.walk(after):
            if isinstance(node, ast.Name) and node.id == 'saved_failure_arrays': node.id = 'saved_arrays'
        # Only the returned summary changes after every original gate has passed.
        before.body[-1].body[-1] = after.body[-1].body[-1] = ast.Return(value=ast.Constant(value='verified-originals'))
        self.assertEqual(syntax(before), syntax(after))

    def test_complete_failure_recomputes_all_raw_arrays(self):
        members, spec, admission, rows = failed_fixture()
        value = failure.saved_failure_arrays(fixtures.zip_bytes(members), spec, 'primary', admission, rows,
                                             fixtures.primary, fixtures.decision)
        self.assertEqual(value['result']['panel_gate'], 'NO_PROMOTION')
        self.assertEqual(value['result']['actual_forwards'], 576)
        self.assertFalse(value['result']['replication_open_allowed'])
        self.assertFalse(value['result']['candidate_engineering_validation_allowed'])

    def test_pass_and_summary_only_forged_failure_reject(self):
        members, spec, admission, rows = fixtures.fixture()
        for forged in (False, True):
            copied = dict(members)
            if forged:
                result = fixtures.reader.document(copied['RESULT.json'])
                result.update(panel_gate='NO_PROMOTION', replication_open_allowed=False, candidate_engineering_validation_allowed=False)
                copied['RESULT.json'] = fixtures.reader.canonical(result)
            with self.assertRaises(ValueError):
                failure.saved_failure_arrays(fixtures.zip_bytes(copied), spec, 'primary', admission, rows,
                                             fixtures.primary, fixtures.decision)

    def test_partial_unknown_output_and_event_tampering_reject(self):
        original, spec, admission, rows = failed_fixture()
        for kind in ('journal', 'raw', 'clock', 'unknown', 'stop'):
            members = dict(original)
            result = fixtures.reader.document(members['RESULT.json'])
            if kind == 'journal': del members[next(n for n in members if n.startswith('attempts/'))]
            if kind == 'raw': members[next(n for n in members if n.startswith('raw_outputs/'))] = fixtures.npy([0] * 17, True)[0]
            if kind == 'clock': result['records'][1]['gains']['1.0']['B0']['event_samples'] = [999]
            if kind == 'unknown': result['consumed_unknown'] = ['one']
            if kind == 'stop': result['panel_gate'] = 'STOP'
            members['RESULT.json'] = fixtures.reader.canonical(result)
            with self.assertRaises(ValueError, msg=kind):
                failure.saved_failure_arrays(fixtures.zip_bytes(members), spec, 'primary', admission, rows,
                                             fixtures.primary, fixtures.decision)

    def test_sealed_opening_blocks_before_network_or_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closed = json.loads(train.OPENING.read_bytes())
            closed.update(enabled=False, quality_lock=None, failure_adapter=None,
                          actual_primary_evaluator_sha256=None, actual_primary_producer_manifest_sha256=None)
            path = root / 'opening.json'; path.write_text(json.dumps(closed))
            with patch.object(train, 'OPENING', path), patch.object(train, 'gh', side_effect=AssertionError('no network')):
                with self.assertRaisesRegex(ValueError, 'sealed'):
                    train.fetch(root / 'inputs', '1' * 40)
                with self.assertRaisesRegex(ValueError, 'sealed'):
                    train.execute(root / 'inputs', root / 'outputs', '1' * 40)
            self.assertFalse((root / 'inputs').exists())
            self.assertFalse((root / 'outputs').exists())

    def test_generic_enabled_switch_cannot_open_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = json.loads(train.OPENING.read_bytes())
            value.update(enabled=True, quality_lock=None, failure_adapter=None,
                         actual_primary_evaluator_sha256=None, actual_primary_producer_manifest_sha256=None)
            path = Path(tmp) / 'opening.json'; path.write_text(json.dumps(value))
            with patch.object(train, 'OPENING', path):
                with self.assertRaises(ValueError): train.opening()

    def test_previous_source_receipts_preserve_exact_member_bytes_and_reject_drift(self):
        originals = {'MANIFEST.json': b'{"rows": []}\n', 'HISTORY.json': b'{ "past": true }\n'}
        report = dict(original_manifest_sha256=train.sha(originals['MANIFEST.json']),
                      original_history_sha256=train.sha(originals['HISTORY.json']))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / 'no-promotion/inputs'; directory.mkdir(parents=True)
            (directory / 'source.zip').write_bytes(fixtures.zip_bytes(originals))
            pins = train.previous_source_metadata(root, report, create=True)
            self.assertEqual(pins, train.previous_source_metadata(root, report))
            for member, raw in originals.items():
                name = 'ORIGINAL_B23_SOURCE_' + member
                self.assertEqual((root / name).read_bytes(), raw)
                self.assertEqual(pins[name], dict(bytes=len(raw), sha256=train.sha(raw), original_member=member))
            with self.assertRaises(FileExistsError): train.previous_source_metadata(root, report, create=True)
            (root / 'ORIGINAL_B23_SOURCE_MANIFEST.json').write_bytes(b'{}\n')
            with self.assertRaises(ValueError): train.previous_source_metadata(root, report)
            with self.assertRaises(ValueError):
                train.previous_source_metadata(root, dict(report, original_history_sha256='0' * 64))

    def test_execution_identity_has_one_public_first_attempt(self):
        parent, head = '1' * 40, '2' * 40
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=train.REPO, GITHUB_EVENT_NAME='push',
            GITHUB_REF=train.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=head, GITHUB_RUN_ID='123')
        event = dict(before=parent, after=head, ref=train.BRANCH, forced=False, deleted=False,
            repository=dict(full_name=train.REPO, private=False))
        with patch.object(train, 'opening', return_value={}), patch.object(train, 'assessment_binding', return_value={}):
            self.assertEqual(train.guard(parent, env, event, parent)['run_id'], '123')
            for bad_env, bad_event in ((dict(env, GITHUB_RUN_ATTEMPT='2'), event), (env, dict(event, forced=True)),
                (env, dict(event, repository=dict(full_name=train.REPO, private=True)))):
                with self.assertRaises(ValueError): train.guard(parent, bad_env, bad_event, parent)

    def test_complete_original_producer_plus_new_resources(self):
        names = producer.paths()
        original = json.loads((train.ROOT / train.design()['original_b23_producer_manifest_path']).read_bytes())
        self.assertEqual(len(original['files']), 306)
        self.assertTrue(set(original['files']) <= names)
        self.assertTrue(set(train.assessment_binding()['files']) <= names)
        self.assertIn('automation/english_v6_b24_b23_failure.py', names)
        self.assertIn('automation/english_v6_b23_quality_readback.py', names)
        self.assertIn('automation/english_v6_b23_primary_eval.py', names)
        self.assertIn(train.design()['train_only_mechanism_memo']['path'], names)

    def test_entity_capture_imports_all_dependencies_without_fallback(self):
        names = producer.paths()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'captured'; root.mkdir()
            for name in names:
                target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(train.ROOT / name, target)
            source = '''import builtins, pathlib, runpy, socket, sys
original = builtins.__import__
def safe_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','numpy','onnxruntime','requests','httpx','transformers','piper'}:
        raise AssertionError('no model/network runtime')
    return original(name, *args, **kwargs)
builtins.__import__ = safe_import
class NoNetworkSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        raise AssertionError('no network')
socket.socket = NoNetworkSocket
root = pathlib.Path(sys.argv[1]).resolve()
sys.argv = [str(root / 'automation/english_v6_b24_b23_warm_start.py'), 'check']
runpy.run_path(sys.argv[0], run_name='__main__')
for value in list(sys.modules.values()):
    path = getattr(value, '__file__', None)
    if path and '/workspace/' in str(path):
        assert pathlib.Path(path).resolve().is_relative_to(root), path
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
'''
            output = subprocess.check_output([sys.executable, '-I', '-S', '-B', '-c', source, str(root)], text=True)
            self.assertIn('PASS: B24', output)

    def test_workflow_durable_budget_precedes_every_new_runtime_and_is_inert(self):
        template = (train.ROOT / producer.TEMPLATE).read_text()
        self.assertEqual(template.count(producer.TOKEN.decode()), 2)
        self.assertIn(train.BRANCH.removeprefix('refs/heads/'), template)
        self.assertNotIn('workflow_dispatch', template)
        self.assertLess(template.index('b23_warm_start.py consume '), template.index('torch==2.9.1+cpu'))
        self.assertLess(template.index('name: ' + train.MARKER), template.index('torch==2.9.1+cpu'))
        raw = (train.ROOT / train.WORKFLOW).read_text()
        if raw != template:
            parent = os.environ.get('PREPARED_PARENT', '')
            self.assertRegex(parent, r'^[0-9a-f]{40}$')
            self.assertEqual(raw, template.replace(producer.TOKEN.decode(), parent))
        self.assertNotIn('--numeric', template)
        self.assertIn('ORIGINAL_B23_SOURCE_MANIFEST.json', template)
        self.assertIn('ORIGINAL_B23_SOURCE_HISTORY.json', template)


if __name__ == '__main__':
    unittest.main()
