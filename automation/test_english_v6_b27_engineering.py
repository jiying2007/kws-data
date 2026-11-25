"""B27 provenance and invented-byte contracts; no models, compiler or network."""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b27_engineering as eng
import english_v6_b27_engineering_readback as reader
import english_v6_b27_quality_readback as quality
import bind_b27_engineering as binder
import test_english_v6_b24r1_engineering as old_tests
import test_english_v6_b24r1_engineering_readback as fixtures


def quality_lock(panel):
    value = dict(schema=quality.LOCK_SCHEMA, panel=panel,
        evaluator_sha256=eng.sha((HERE / ('english_v6_b27_' + panel + '_eval.py')).read_bytes()),
        external_proof_binding_sha256=eng.sha((HERE / ('english_v6_b27_' + panel + '_proof_binding.json')).read_bytes()),
        inputs_sha256='b' * 64, producer_manifest_sha256='c' * 64, original_result_sha256='d' * 64)
    for number, role in enumerate(('result', 'producer', 'consumed'), 1):
        value[role] = dict(id=number, run_id=100 if panel == 'primary' else 200,
                          run_head='e' * 40, bytes=100, sha256='f' * 64)
    return value


def dual():
    with patch.object(old_tests, 'eng', eng), patch.object(eng, 'CANDIDATE_PT', 'a' * 64):
        doc, proofs = old_tests.dual_fixture()
    doc['quality_locks'] = {panel: quality_lock(panel) for panel in ('primary', 'replication')}
    doc['original_training_report_sha256'] = '9' * 64
    for panel, value in proofs.items():
        value.update(schema=quality.REPORT_SCHEMA,
            original_B24_training_assessment_sha256=eng.ORIGINAL_B24_TRAINING_ASSESSMENT_SHA,
            external_proof_binding_sha256=doc['quality_locks'][panel]['external_proof_binding_sha256'],
            original_result_sha256=doc['quality_locks'][panel]['original_result_sha256'],
            original_training_report_sha256=doc['original_training_report_sha256'],
            original_B24Q1_failure_report_sha256=eng.Q1_FAILURE_REPORT_SHA,
            all_original_training_proof_verified=True, complete_original_Q1_failure_verified=True,
            replication_open_allowed=panel == 'primary', candidate_engineering_validation_allowed=panel == 'replication',
            automatic_next_assessment_allowed=False, promotion_allowed=False, product_released=False)
    return doc, proofs


def bound():
    doc = json.loads(eng.SPEC.read_bytes())
    other, _ = dual()
    doc.update(enabled=True, quality_reader_sha256=reader.QUALITY_READER_SHA,
               quality_locks=other['quality_locks'], model_pt_sha256=other['model_pt_sha256'],
               original_training_report_sha256=other['original_training_report_sha256'])
    doc['inputs']['pair'] = other['inputs']['pair']
    doc['candidate_files'] = {eng.STEM + '.' + ext: dict(bytes=20000, sha256='a' * 64 if ext == 'pt' else 'c' * 64)
                              for ext in ('pt', 'npz', 'onnx')}
    return doc


class Contracts(unittest.TestCase):
    def test_fetch_reuses_frozen_primary_api_bytes_and_each_actual_complete_readback_once(self):
        doc, proofs = dual(); doc['quality_reader_sha256'] = reader.QUALITY_READER_SHA
        calls = []
        acquire = SimpleNamespace(fetch=lambda *args, **kwargs: calls.append(('proof', args, kwargs)),
            fetch_files=lambda *args: calls.append(('original primary18', args)))
        verifier = SimpleNamespace(__file__=quality.__file__,
            verify_quality_original=lambda *args: calls.append(('verify primary', args)) or proofs['primary'],
            fetch_quality_original=lambda *args: calls.append(('fetch verify replica', args)) or proofs['replication'])
        frozen = dict(primary_quality_files={'saved API': 'invented exact file pin'},
                      primary_quality_transport={'saved API': 'invented exact original bytes'})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(eng, 'proof_bindings', return_value={'replication': frozen}), \
                 patch.object(eng, 'module', side_effect=lambda name: verifier if name == eng.QUALITY_READER else acquire):
                self.assertEqual(eng.quality(root, doc, fetch=True), proofs)
            self.assertEqual([value[0] for value in calls],
                ['proof', 'original primary18', 'verify primary', 'proof', 'fetch verify replica'])
            self.assertEqual(calls[1][1], (root / 'quality/primary', frozen['primary_quality_files'], frozen['primary_quality_transport']))
            self.assertEqual(calls[3][2]['primary_reuse'], root / 'quality/primary')
            self.assertEqual(calls[3][2]['training_reuse'], root / 'proof_roots/primary/training-origin')

    def test_binding_a_failed_or_partial_panel_never_creates_review_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'lock.json'; path.write_bytes(reader.canonical(quality_lock('primary')))
            with patch.object(quality, 'verify_quality_original', return_value=dict(panel_gate='NO_PROMOTION')), \
                 patch.object(eng, 'original_pair', side_effect=AssertionError('no candidate before dual PASS')):
                with self.assertRaisesRegex(ValueError, 'actual complete original primary PASS'):
                    binder.prepare_inputs(root, path, root, root, path, root, root / 'review')
            self.assertFalse((root / 'review').exists())
            with self.assertRaisesRegex(ValueError, 'fresh absolute review directory'):
                binder.prepare_inputs(root, path, root, root, path, root, root)

    def test_engineering_pin_generator_uses_actual_four_zips_and_all_eight_api_bodies(self):
        with patch.object(fixtures, 'reader', reader):
            previous, api = fixtures.api_fixture()
        for item in api['artifacts']['artifacts']:
            item['name'] = item['name'].replace('b24r1', 'b27')
        bodies = {
            'producer': fixtures.zipped({'PRODUCER.json': b'{}',
                **{'repository/automation/' + name: (HERE / name).read_bytes() for name in (
                    'english_v6_b27_engineering.py', 'english_v6_b27_quality_readback.py', 'english_v6_b27_engineering_inputs.json')}}),
            'result': fixtures.zipped({'RESULT.json': b'{"invented":true}', 'MANIFEST.json': b'{}'}),
            'consumed': fixtures.zipped({'CONSUMPTION.json': b'{}'}),
            'input_proof': fixtures.zipped({'FETCH.json': b'{}'})}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'api').mkdir(); (root / 'originals').mkdir()
            for role, raw in bodies.items():
                (root / 'originals' / (role + '.zip')).write_bytes(raw)
                item = next(x for x in api['artifacts']['artifacts'] if x['id'] == previous['artifacts'][role]['id'])
                item.update(size_in_bytes=len(raw), digest='sha256:' + eng.sha(raw))
            for role, value in api.items(): (root / 'api' / (role + '.json')).write_bytes(reader.canonical(value))
            result = binder.original_engineering_lock(root)
            self.assertEqual(set(result['artifacts']), reader.ROLES)
            self.assertEqual(set(result['api']), reader.API_ROLES)
            self.assertEqual(result['evaluator_sha256'], reader.ENGINEERING_SHA)
            self.assertEqual(result['artifacts']['result']['sha256'], eng.sha(bodies['result']))
            (root / 'originals/result.zip').write_bytes(b'changed')
            with self.assertRaises(ValueError): binder.original_engineering_lock(root)

    def test_exact_original_authority_and_shared_numeric_callables(self):
        self.assertEqual(eng.sha(eng.ORIGINAL.read_bytes()), eng.ORIGINAL_SHA)
        self.assertEqual(reader.sha(Path(eng.__file__).read_bytes()), reader.ENGINEERING_SHA)
        self.assertEqual(reader.sha(Path(quality.__file__).read_bytes()), reader.QUALITY_READER_SHA)
        for name in ('ctc', 'compare_logits', 'validate_events', 'load_candidate', 'build_sdk', 'Ledger',
                     'original_zip', 'verify_pipeline', 'fetch', 'preflight'):
            self.assertIs(getattr(eng, name), getattr(eng.original, name), name)
        for name in ('protocol', 'quality', 'original_pair', 'producer_paths'):
            self.assertIs(eng.original.preflight.__globals__[name], getattr(eng, name), name)

    def test_original_140_call_loop_changes_only_identity_strings(self):
        def functions(path):
            return {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}
        before = functions(eng.ORIGINAL)['execute']
        after = functions(Path(eng.__file__))['_execute']; after.name = 'execute'
        for node in ast.walk(after):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = node.value.replace('b27', 'b24r1').replace('B27', 'B24R1')
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))

    def test_original_numeric_and_archive_readback_helpers_remain_exact(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_bytes()).body
                    if isinstance(n, ast.FunctionDef)}
        before = functions(HERE / 'english_v6_b24r1_engineering_readback.py')
        after = functions(Path(reader.__file__))
        for name in ('read', 'opened', 'original', 'tree_members', 'validate_artifact', 'npy_body',
                     'pack_bytes', 'saved_arrays', 'installed_sdk', 'retained_api_metadata', 'validate_lock'):
            self.assertEqual(before[name], after[name], name)

    def test_inert_all_modes_stop_before_evidence_or_runtime(self):
        self.assertEqual(eng.source_check()['status'], 'INERT_AWAIT_ORIGINAL_DUAL_QUALITY_PASS')
        self.assertIsNone(eng.CANDIDATE_PT)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); closed = root / 'closed.json'; closed.write_text('{"enabled":false}')
            with patch.object(eng, 'SPEC', closed), patch.object(eng.original, 'SPEC', closed), \
                 patch.object(eng, 'module', side_effect=AssertionError('no proof or runtime import')), \
                 patch.object(eng.original, 'gh', side_effect=AssertionError('no network')):
                for action in (lambda: eng.fetch(root / 'inputs', root / 'pipeline', 'a' * 40),
                               lambda: eng.preflight(root / 'inputs', root / 'pipeline', 'a' * 40),
                               lambda: eng.execute(root / 'inputs', root / 'pipeline', root / 'output', 'a' * 40)):
                    with self.assertRaisesRegex(ValueError, 'INERT'): action()
            self.assertEqual(list(root.iterdir()), [closed])

    def test_both_actual_b27_passes_and_complete_training_qualification_are_required(self):
        doc, proofs = dual(); self.assertEqual(eng.validate_dual_quality(proofs, doc), proofs)
        mutations = (
            lambda x: x['replication'].update(panel_gate='NO_PROMOTION'),
            lambda x: x['primary'].update(panel_gate='STOP'),
            lambda x: x['replication'].update(schema='english-v6-b24q1-verified-original-quality-v1'),
            lambda x: x['replication'].update(all_original_training_proof_verified=False),
            lambda x: x['primary'].update(complete_original_Q1_failure_verified=False),
            lambda x: x['replication'].update(original_training_report_sha256='8' * 64),
            lambda x: x['primary'].update(original_B24Q1_failure_report_sha256='8' * 64),
            lambda x: x['replication'].update(external_proof_binding_sha256='8' * 64),
            lambda x: x['replication']['model_pt_sha256'].update(control=eng.INITIAL_PT),
            lambda x: x['replication']['source_inventory']['pcm_sha256'].pop(),
            lambda x: x['replication']['source_inventory']['canonical_people'].__setitem__(
                0, x['primary']['source_inventory']['canonical_people'][0]),
            lambda x: x['replication'].update(candidate_engineering_validation_allowed=False),
        )
        for change in mutations:
            broken = copy.deepcopy(proofs); change(broken)
            with self.assertRaises(ValueError): eng.validate_dual_quality(broken, doc)
        with self.assertRaises(ValueError): eng.validate_dual_quality({'primary': proofs['primary']}, doc)

    def test_bound_registration_preserves_new_model_roles_old_stops_and_numeric_caps(self):
        doc = bound()
        with tempfile.TemporaryDirectory() as temp, patch.object(eng, 'proof_bindings', return_value={}):
            path = Path(temp) / 'binding.json'
            with patch.object(eng, 'SPEC', path):
                path.write_text(json.dumps(doc)); self.assertEqual(eng.protocol(), doc)
                changes = (
                    lambda x: x.update(maximum_backend_calls=141),
                    lambda x: x.update(onnx_max_abs=.002),
                    lambda x: x.update(original_B24_training_assessment_sha256=eng.ASSESSMENT_SHA),
                    lambda x: x.update(B24Q1_terminal_NO_PROMOTION_preserved=False),
                    lambda x: x.update(old_B24Q1_replication_opened=True),
                    lambda x: x.update(assessment_sha256='7' * 64),
                    lambda x: x['quality_locks']['replication'].update(schema='english-v6-b24q1-original-quality-lock-v1'),
                    lambda x: x['quality_locks']['replication']['producer'].update(bytes=128_000_001),
                    lambda x: x['model_pt_sha256'].update(control=eng.INITIAL_PT),
                    lambda x: x['model_pt_sha256'].update(challenger=eng.CONTROL_PT),
                    lambda x: x['candidate_files'][eng.STEM + '.pt'].update(sha256='8' * 64),
                )
                for mutate in changes:
                    broken = copy.deepcopy(doc); mutate(broken); path.write_text(json.dumps(broken))
                    with self.assertRaises(ValueError): eng.protocol()

    def test_full_bound_train_is_the_only_source_of_pair_and_candidate_identity(self):
        doc = bound(); pin = doc['inputs']['pair']
        training = dict(model_pt_sha256=doc['model_pt_sha256'], run_id=pin['run_id'], run_head=pin['run_head'],
                        artifacts=dict(models={k: pin[k] for k in ('id', 'bytes', 'sha256')}))
        first = dict(files={'invented': {}}, training_lock=dict(training=training),
            training_reader_sha256='6' * 64, training_report_sha256=doc['original_training_report_sha256'])
        second = dict(copy.deepcopy(first), primary_quality_lock=doc['quality_locks']['primary'],
                      primary_quality_reader_sha256=doc['quality_reader_sha256'])
        bindings = dict(primary=first, replication=second)
        fake = SimpleNamespace(profile=lambda: None, binding=lambda panel: bindings[panel])
        # The exact existing metadata file digests remain the explicit lock pins.
        with patch.object(eng, 'module', return_value=fake):
            self.assertEqual(eng.proof_bindings(doc), bindings)
            wrong = copy.deepcopy(doc); wrong['model_pt_sha256']['challenger'] = '5' * 64
            with self.assertRaises(ValueError): eng.proof_bindings(wrong)
            second['training_report_sha256'] = '4' * 64
            with self.assertRaises(ValueError): eng.proof_bindings(doc)

    def test_full_original_train_api_is_called_before_any_pair_export(self):
        doc = bound(); receipt = {'fixture': 'incomplete actual TRAIN'}
        raw = fixtures.zipped({'RESULT.json': reader.canonical(receipt)})
        doc['inputs']['pair'].update(bytes=len(raw), sha256=eng.sha(raw))
        calls = []
        def reject(*args):
            calls.append(args)
            raise ValueError('full original TRAIN required')
        primary = SimpleNamespace(STEMS=eng.STEMS, MODEL_COMPARISON=eng.MODEL_COMPARISON,
            CONTROL_PT=eng.CONTROL_PT, validate_training_archive=reject)
        profile = SimpleNamespace(proof_scope=lambda *args: nullcontext())
        with tempfile.TemporaryDirectory() as temp, patch.object(eng, 'proof_bindings'), \
             patch.object(eng, 'module', side_effect=lambda name: profile if name.endswith('_profile') else primary):
            root = Path(temp); (root / 'pair.zip').write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'full original TRAIN required'): eng.original_pair(root, doc)
        self.assertEqual(len(calls), 1)

    def test_execute_retains_one_proof_scope_across_preflight_and_numeric_load(self):
        events = []
        @contextmanager
        def scoped(root, panel):
            events.append(('enter', root, panel))
            try: yield
            finally: events.append('destroy')
        profile = SimpleNamespace(proof_scope=scoped)
        with patch.object(eng, 'protocol', side_effect=lambda: events.append('protocol')), \
             patch.object(eng, 'module', return_value=profile), \
             patch.object(eng, '_execute', side_effect=lambda *args: events.append('whole original execution')):
            eng.execute(Path('/originals'), Path('/sdk'), Path('/output'), 'f' * 40)
        self.assertEqual(events, ['protocol', ('enter', Path('/originals/proof_roots/primary'), 'primary'),
                                  'whole original execution', 'destroy'])

    def test_capacity_preserves_all_evidence_and_all_core_archive_caps(self):
        import english_v6_b24q1_capacity as capacity
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'full.py').write_bytes(b'complete original producer')
            checked = eng.evidence_capacity(root)
            self.assertEqual(checked['maximum_bytes'], 128_000_000)
            self.assertEqual(checked['files_verified'], 1)
            oversized = dict(files={'entire-original': dict(bytes=113_000_000, sha256='a' * 64)}, directories=[])
            with patch.object(capacity, 'inventory', return_value=oversized), self.assertRaises(ValueError):
                eng.evidence_capacity(root)
        for panel in ('primary', 'replication'):
            lock = quality_lock(panel); quality.validate_lock(lock)
            for role in ('result', 'producer', 'consumed'):
                wrong = copy.deepcopy(lock); wrong[role]['bytes'] = 128_000_001
                with self.assertRaises(ValueError): quality.validate_lock(wrong)
        with self.assertRaises(ValueError): reader.artifact_pin({**quality_lock('primary')['result'], 'bytes': 128_000_001})

    def test_exact_140_invented_arrays_and_84_csvs_recompute_and_mutation_fails(self):
        files, report, rows, waves = fixtures.array_fixture()
        with reader.opened(fixtures.zipped(files)) as archive:
            reader.saved_arrays(archive, report, rows, waves, eng)
        key = next(name for name in files if name.endswith('.csv'))
        files[key] = files[key].replace(b'0,512,0', b'0,513,0', 1)
        with reader.opened(fixtures.zipped(files)) as archive, self.assertRaises(ValueError):
            reader.saved_arrays(archive, report, rows, waves, eng)

    def guard_fixture(self, root):
        locks = {panel: quality_lock(panel) for panel in ('primary', 'replication')}
        training = dict(training_reader_sha256=eng.sha((HERE / 'english_v6_b27_training_readback.py').read_bytes()),
                        training_lock=dict(schema='invented-original-training-lock-for-process-boundary'))
        commands = {}
        for panel in locks:
            evidence = root / 'quality' / panel
            for role in ('api', 'originals', 'inputs'): (evidence / role).mkdir(parents=True)
            proof = root / 'proof_roots' / panel; proof.mkdir(parents=True)
            lock = root / (panel + '.json'); lock.write_bytes(reader.canonical(locks[panel]))
            commands[panel] = [sys.executable, '-I', '-S', '-B', str(Path(quality.__file__).resolve()),
                              '--verify-worker', str(evidence), str(lock), str(proof)]
        origin = root / 'proof_roots/primary/training-origin'
        for name in ('trained', 'qualification'): (origin / name).mkdir(parents=True)
        lock = root / 'training.json'; lock.write_bytes(reader.canonical(training['training_lock']))
        commands['training'] = [sys.executable, '-I', '-S', '-B', str(HERE / 'english_v6_b27_training_readback.py'),
                               '--verify-worker', str(origin / 'trained'), str(lock), str(origin / 'qualification')]
        return locks, training, commands

    def test_readback_allows_only_two_quality_and_one_training_worker_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); locks, training, commands = self.guard_fixture(root)
            with patch.object(reader.subprocess, 'Popen', return_value='recorded') as launched:
                with reader.offline_engineering(root, locks, quality, training):
                    for key in ('primary', 'replication', 'training'):
                        self.assertEqual(subprocess.Popen(commands[key], stdout=subprocess.PIPE), 'recorded')
                        with self.assertRaises(ValueError): subprocess.Popen(commands[key])
                self.assertEqual(launched.call_count, 3)

    def test_readback_rejects_wrong_root_lock_reader_options_and_all_runtime_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); locks, training, commands = self.guard_fixture(root)
            with patch.object(reader.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
                with reader.offline_engineering(root, locks, quality, training):
                    for key in ('primary', 'training'):
                        command = commands[key]
                        with self.assertRaises(ValueError): subprocess.Popen(command, env={})
                        with self.assertRaises(ValueError): subprocess.Popen(command, cwd=str(root))
                        wrong = list(command); wrong[8] = str(root)
                        with self.assertRaises(ValueError): subprocess.Popen(wrong)
                        wrong = list(command); wrong[4] = str(HERE / 'english_v6_b24q1_quality_readback.py')
                        with self.assertRaises(ValueError): subprocess.Popen(wrong)
                        Path(command[7]).write_bytes(b'{}')
                        with self.assertRaises(ValueError): subprocess.Popen(command)
                    with self.assertRaises(ValueError): socket.socket()
                    with self.assertRaises(ValueError): __import__('numpy')

    def test_first_push_and_frozen_template_remain_exact_with_long_readback_budget(self):
        parent, head = 'a' * 40, 'b' * 40
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=eng.REPO, GITHUB_EVENT_NAME='push',
            GITHUB_REF='refs/heads/' + eng.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=head, GITHUB_RUN_ID='123')
        event = dict(before=parent, after=head, ref=env['GITHUB_REF'], forced=False, deleted=False,
                     repository=dict(full_name=eng.REPO, private=False))
        eng.validate_event(parent, head, [parent], [eng.WORKFLOW], env, event)
        with self.assertRaises(ValueError): eng.validate_event(parent, head, [parent], [eng.WORKFLOW], {**env, 'GITHUB_RUN_ATTEMPT': '2'}, event)
        template = eng.TEMPLATE.read_bytes(); self.assertEqual((HERE.parent / eng.WORKFLOW).read_bytes(), template)
        self.assertEqual(template.count(eng.PARENT_TOKEN.encode()), 2)
        text = template.decode(); self.assertNotIn('workflow_dispatch:', text); self.assertNotIn('contents: write', text)
        for limit in (360, 120, 90, 100): self.assertIn('timeout-minutes: ' + str(limit), text)
        self.assertLess(text.index('engineering.py preflight'), text.index('torch==2.9.1+cpu'))
        for name in ('FETCH.json', 'CONSUMPTION.json', 'api/runs.json', 'api/markers.json'):
            self.assertIn('${{ runner.temp }}/b27-engineering-inputs/' + name, text)


if __name__ == '__main__':
    unittest.main()
