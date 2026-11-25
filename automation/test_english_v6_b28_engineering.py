"""Inert B28 and invented-byte provenance contracts; no model or audio work."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b28_engineering as eng
import english_v6_b28_engineering_readback as reader


def frozen(name):
    path = HERE / name
    return path if path.is_file() else HERE.parent / 'frozen' / name


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def artifact(number, run=100):
    return dict(id=number, run_id=run, run_head='e' * 40, bytes=100, sha256='f' * 64)


def failure():
    """Shape fixture only; not a certificate or original scientific evidence."""
    return dict(schema='english-v6-b28-original-b27-failure-certification-binding-v1',
        original_reader_path='automation/english_v6_b27_quality_readback.py',
        original_reader_sha256='87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa',
        original_report_sha256='1' * 64,
        adapter_path='automation/english_v6_b27_primary_no_promotion_readback.py',
        adapter_sha256='2' * 64, adapter_report_schema='english-v6-b27-primary-rooted-no-promotion-readback-v1',
        adapter_report_sha256='3' * 64, original_standalone_continuation_succeeded=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0)


def inert_fixture():
    """A private disabled fixture remains disabled after future real binding."""
    value = json.loads(eng.SPEC.read_bytes())
    value['enabled'] = False
    for name in ('assessment_sha256', 'training_assessment_sha256', 'training_design_sha256',
                 'quality_reader_sha256', 'original_training_report_sha256', 'candidate_files',
                 'original_B27_failure_certification', 'original_B27_failure_report_sha256'):
        value[name] = None
    value['quality_locks'] = dict(primary=None, replication=None)
    value['inputs']['pair'] = None
    value['model_pt_sha256']['challenger'] = None
    return value


def dual_fixture():
    """Invented field values exercise rejection; never written as PASS output."""
    doc = inert_fixture()
    doc.update(enabled=True, assessment_sha256='4' * 64, training_assessment_sha256='4' * 64,
        training_design_sha256='5' * 64, original_training_report_sha256='6' * 64,
        original_B27_failure_certification=failure(), original_B27_failure_report_sha256='1' * 64,
        quality_reader_sha256='7' * 64)
    doc['model_pt_sha256']['challenger'] = '8' * 64
    doc['inputs']['pair'] = artifact(999, 300)
    doc['candidate_files'] = {eng.STEM + '.' + suffix: dict(bytes=20000,
        sha256='8' * 64 if suffix == 'pt' else '9' * 64) for suffix in ('pt', 'npz', 'onnx')}
    proofs = {}
    for panel, run in (('primary', 100), ('replication', 200)):
        lock = dict(schema='english-v6-b28-original-quality-lock-v1', panel=panel,
            **{role: artifact(run + n, run) for n, role in enumerate(('result', 'producer', 'consumed'))},
            **{name: 'a' * 64 for name in ('evaluator_sha256', 'inputs_sha256', 'producer_manifest_sha256',
                'original_result_sha256', 'external_proof_binding_sha256')})
        doc['quality_locks'][panel] = lock
        proof = dict(schema='english-v6-b28-verified-original-quality-v1', panel=panel, panel_gate='PASS',
            model_pt_sha256=copy.deepcopy(doc['model_pt_sha256']), assessment_sha256=doc['assessment_sha256'],
            training_design_sha256=doc['training_design_sha256'],
            training_assessment_sha256=doc['training_assessment_sha256'],
            original_B24_training_assessment_sha256=eng.ORIGINAL_B24_TRAINING_ASSESSMENT_SHA,
            pair_pin=doc['inputs']['pair'], original_result_sha256=lock['original_result_sha256'],
            external_proof_binding_sha256=lock['external_proof_binding_sha256'],
            original_training_report_sha256=doc['original_training_report_sha256'],
            original_lock_sha256=eng.sha(canonical(lock)),
            original_B27_failure_certification=copy.deepcopy(doc['original_B27_failure_certification']),
            original_B27_failure_report_sha256=doc['original_B27_failure_report_sha256'],
            natural_objective_returns_verified=384, external_training_originals=71,
            external_total_originals=71 if panel == 'primary' else 89,
            derived_B24_originals=5, derived_B24_is_new_qualification_source=False,
            derived_B24_origin='same-original-B27-TRAIN_receipts-members',
            primary_independently_verified=panel == 'replication',
            replication_open_allowed=panel == 'primary', candidate_engineering_validation_allowed=panel == 'replication',
            automatic_next_assessment_allowed=False, promotion_allowed=False, product_released=False,
            new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0,
            admission=dict(same_job_matched_control=False, historical_control_retrained=False,
                           model_comparison=eng.MODEL_COMPARISON),
            source_inventory={key: [panel + '-' + key + '-' + str(i) for i in range(120 if key == 'pcm_sha256' else 12)]
                for key in ('pcm_sha256', 'source_group_id', 'canonical_people')})
        proof.update(dict.fromkeys(('all_original_arrays_verified', 'original_source_admission_recomputed',
            'original_producer_git_verified', 'all_original_training_proof_verified', 'complete_original_Q1_failure_verified',
            'complete_original_B27_failure_verified', 'natural_no_event_terms_recomputed',
            'captured_code_matches_pinned_local_reader_root', 'old_stops_and_reserves_preserved'), True))
        proofs[panel] = proof
    return doc, proofs


class InertContracts(unittest.TestCase):
    def test_exact_original_numerical_code_objects_and_source(self):
        self.assertEqual(eng.sha(eng.ORIGINAL.read_bytes()), eng.ORIGINAL_SHA)
        self.assertEqual(eng.sha(Path(eng.__file__).read_bytes()), reader.ENGINEERING_SHA)
        for name in ('ctc', 'compare_logits', 'validate_events', 'load_candidate', 'build_sdk', 'Ledger',
                     'original_zip', 'verify_pipeline', 'fetch', 'preflight'):
            self.assertIs(getattr(eng, name), getattr(eng.original, name), name)
        for name in ('protocol', 'quality', 'original_pair', 'producer_paths'):
            self.assertIs(eng.original.preflight.__globals__[name], getattr(eng, name), name)

    def test_original_140_call_loop_changes_only_identity_text(self):
        def function(path, name):
            return next(node for node in ast.parse(path.read_bytes()).body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        before = function(eng.ORIGINAL, 'execute')
        after = function(Path(eng.__file__), '_execute'); after.name = 'execute'
        for node in ast.walk(after):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = node.value.replace('b28', 'b24r1').replace('B28', 'B24R1')
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))

    def test_saved_array_archive_pack_and_installed_sdk_readers_are_unchanged(self):
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_bytes()).body
                    if isinstance(node, ast.FunctionDef)}
        source = frozen('english_v6_b27_engineering_readback.py')
        self.assertEqual(eng.sha(source.read_bytes()), '98848744d966f35855c3408adec7e67601917009b565c6c8a978fd914ec2de79')
        before, after = functions(source), functions(Path(reader.__file__))
        for name in ('read', 'opened', 'original', 'tree_members', 'validate_artifact', 'npy_body',
                     'pack_bytes', 'saved_arrays', 'installed_sdk', 'retained_api_metadata'):
            self.assertEqual(before[name], after[name], name)

    def test_current_registration_is_inert_or_requires_genuine_bound_proof(self):
        doc = json.loads(eng.SPEC.read_bytes())
        value = eng.source_check()
        if doc['enabled'] is False:
            self.assertEqual(value['status'], 'INERT_AWAIT_ORIGINAL_DUAL_QUALITY_PASS')
            self.assertIsNone(reader.QUALITY_READER_SHA)
        else:
            # source_check already ran the complete metadata/proof-binding
            # protocol. A future real binding must pin the same B28 reader.
            self.assertIs(doc['enabled'], True)
            self.assertEqual(value['status'], 'BOUND_REQUIRES_ORIGINAL_DUAL_PASS_READBACK')
            self.assertTrue(eng.valid_hash(reader.QUALITY_READER_SHA))
            self.assertEqual(reader.QUALITY_READER_SHA, doc['quality_reader_sha256'])
        self.assertEqual(value['maximum_backend_calls'], 140)
        self.assertEqual(value['original_count'], 14)
        self.assertEqual(eng.BACKENDS, ('pytorch', 'onnxruntime', 'gcc', 'clang', 'cortex-a32-qemu'))
        self.assertEqual(eng.CONDITIONS, ('original', 'one_second_zero_prefix'))
        with self.assertRaisesRegex(ValueError, 'INERT'):
            reader.validate_lock({})

    def test_once_workflow_is_exact_template_or_exact_reviewed_parent_substitution(self):
        actual = HERE.parent / eng.WORKFLOW
        template = eng.TEMPLATE.read_bytes()
        self.assertEqual(template.count(eng.PARENT_TOKEN.encode()), 2)
        self.assertNotIn(b'workflow_dispatch', template)
        self.assertIn(b'github.run_attempt == 1', template)
        self.assertIn(b'github.event.before ==', template)

        def unchanged_or_bound(raw):
            if raw == template:
                return True
            values = re.findall(rb"^      PREPARED_PARENT: '([0-9a-f]{40})'$", raw, re.M)
            return len(values) == 1 and raw == template.replace(eng.PARENT_TOKEN.encode(), values[0])

        # The unchanged runtime guard separately verifies the real parent,
        # workflow-only change, branch, first public push and consumed marker.
        self.assertTrue(unchanged_or_bound(actual.read_bytes()))
        bound = template.replace(eng.PARENT_TOKEN.encode(), b'a' * 40)
        self.assertTrue(unchanged_or_bound(bound))
        self.assertFalse(unchanged_or_bound(template.replace(eng.PARENT_TOKEN.encode(), b'a' * 40, 1)))
        self.assertFalse(unchanged_or_bound(bound + b'\n'))

    def test_inert_entrypoints_stop_before_proof_network_audio_or_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            closed = root / 'disabled-fixture.json'; closed.write_bytes(canonical(inert_fixture()))
            with patch.object(eng, 'SPEC', closed), patch.object(eng.original, 'SPEC', closed), \
                 patch.object(eng, 'module', side_effect=AssertionError('proof/runtime import')), \
                 patch.object(eng.original, 'module', side_effect=AssertionError('proof/runtime import')), \
                 patch.object(eng.original, 'gh', side_effect=AssertionError('network call')), \
                 patch.object(eng.original, 'engineering_sources', side_effect=AssertionError('PCM read')):
                for action in (lambda: eng.fetch(root / 'inputs', root / 'pipeline', 'a' * 40),
                               lambda: eng.preflight(root / 'inputs', root / 'pipeline', 'a' * 40),
                               lambda: eng.execute(root / 'inputs', root / 'pipeline', root / 'output', 'a' * 40)):
                    with self.assertRaisesRegex(ValueError, 'INERT'):
                        action()
            self.assertEqual(list(root.iterdir()), [closed])

    def test_flipping_enabled_does_not_invent_proof_or_import_runtime(self):
        value = inert_fixture(); value['enabled'] = True
        with patch.object(eng, 'module', side_effect=AssertionError('unbound module import')):
            with self.assertRaisesRegex(ValueError, 'explicit before opening'):
                eng.protocol(value)

    def test_budget_criteria_rollback_and_every_old_gate_cannot_be_relaxed(self):
        original = inert_fixture()
        for field, value in (('original_count', 15), ('maximum_backend_calls', 141),
                ('onnx_max_abs', .002), ('c_mean_abs', .001), ('old_B27_replication_opened', True),
                ('old_B27_candidate_engineering_opened', True), ('old_B27_sources_used_for_scoring', True),
                ('B27_terminal_NO_PROMOTION_preserved', False), ('training_allowed', True),
                ('product_promotion_allowed', True), ('physical_ssc305', True), ('cal_locked_access', True),
                ('external_training_originals', 88), ('external_replication_originals', 88),
                ('natural_objective_returns_verified', 383), ('derived_B24_is_new_qualification_source', True),
                ('old_B27_optimizer_updates_reexecuted', 1)):
            changed = copy.deepcopy(original); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                eng.registration(changed)
        changed = copy.deepcopy(original); changed['inputs']['B0']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): eng.registration(changed)

    def test_only_new_b28_first_push_identity_can_consume_future_engineering(self):
        parent, head = 'a' * 40, 'b' * 40
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=eng.REPO, GITHUB_EVENT_NAME='push',
            GITHUB_REF='refs/heads/' + eng.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=head, GITHUB_RUN_ID='100')
        event = dict(before=parent, after=head, ref=env['GITHUB_REF'], forced=False, deleted=False,
                     repository=dict(full_name=eng.REPO, private=False))
        self.assertEqual(eng.validate_event(parent, head, [parent], [eng.WORKFLOW], env, event)['run_id'], 100)
        for changed in (dict(env, GITHUB_RUN_ATTEMPT='2'),
                        dict(env, GITHUB_REF='refs/heads/research/en-v6-b27-engineering-once-20261125')):
            with self.assertRaises(ValueError):
                eng.validate_event(parent, head, [parent], [eng.WORKFLOW], changed, event)
        with self.assertRaises(ValueError):
            eng.validate_event(parent, head, [parent], [eng.WORKFLOW], env, dict(event, forced=True))


class ProvenanceContracts(unittest.TestCase):
    def test_two_new_passes_require_same_candidate_failure_training_and_complete_proofs(self):
        doc, proofs = dual_fixture()
        self.assertEqual(eng.validate_dual_quality(proofs, doc), proofs)
        changes = (('panel_gate', 'NO_PROMOTION'), ('schema', 'english-v6-b27-verified-original-quality-v1'),
            ('complete_original_B27_failure_verified', False), ('complete_original_Q1_failure_verified', False),
            ('natural_no_event_terms_recomputed', False), ('natural_objective_returns_verified', 383),
            ('external_training_originals', 88), ('external_total_originals', 71),
            ('derived_B24_is_new_qualification_source', True), ('primary_independently_verified', False),
            ('candidate_engineering_validation_allowed', False), ('original_training_report_sha256', '0' * 64),
            ('original_B27_failure_report_sha256', '0' * 64), ('original_lock_sha256', '0' * 64),
            ('new_optimizer_updates', 1), ('new_model_calls', 0.0), ('old_stops_and_reserves_preserved', False),
            ('promotion_allowed', True), ('captured_code_matches_pinned_local_reader_root', False))
        for name, value in changes:
            changed = copy.deepcopy(proofs); changed['replication'][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                eng.validate_dual_quality(changed, doc)
        for old in (eng.CONTROL_PT, eng.INITIAL_PT, eng.B27_PT):
            changed = copy.deepcopy(doc); changed['model_pt_sha256']['challenger'] = old
            with self.assertRaises(ValueError): eng.validate_dual_quality(proofs, changed)

    def test_source_identity_overlap_or_missing_original_rejects_the_dual_pass(self):
        doc, proofs = dual_fixture()
        for name in ('pcm_sha256', 'source_group_id', 'canonical_people'):
            changed = copy.deepcopy(proofs)
            changed['replication']['source_inventory'][name][0] = changed['primary']['source_inventory'][name][0]
            with self.subTest(name=name), self.assertRaises(ValueError):
                eng.validate_dual_quality(changed, doc)
        changed = copy.deepcopy(proofs); changed['replication']['source_inventory']['pcm_sha256'].pop()
        with self.assertRaises(ValueError): eng.validate_dual_quality(changed, doc)

    def test_original_failure_adapter_is_distinct_and_cannot_claim_old_reader_success(self):
        doc, proofs = dual_fixture()
        for key, value in (('original_standalone_continuation_succeeded', True),
                ('adapter_report_sha256', None), ('original_reader_sha256', '0' * 64), ('new_model_calls', 1)):
            changed = copy.deepcopy(doc); changed['original_B27_failure_certification'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): eng.failure_certification(changed)
        proofs['replication']['original_B27_failure_certification']['adapter_sha256'] = '0' * 64
        with self.assertRaises(ValueError): eng.validate_dual_quality(proofs, doc)

    def test_quality_fetch_reuses_exact_primary_and_checks_each_complete_panel(self):
        doc, proofs = dual_fixture(); calls = []
        doc['quality_reader_sha256'] = eng.sha(Path(reader.__file__).read_bytes())
        verifier = SimpleNamespace(__file__=reader.__file__,
            verify_quality_original=lambda *args: calls.append(('verify primary', args)) or proofs['primary'],
            fetch_quality_original=lambda *args: calls.append(('fetch verify replica', args)) or proofs['replication'])
        acquire = SimpleNamespace(fetch=lambda *args, **kwargs: calls.append(('proof', args, kwargs)),
            fetch_files=lambda *args: calls.append(('original primary18', args)))
        bound = dict(primary_quality_files={'fixture': 'original pinned shape'}, primary_quality_transport={'fixture': 'transport shape'})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(eng, 'proof_bindings', return_value={'replication': bound}), \
                 patch.object(eng, 'module', side_effect=lambda name: verifier if name == eng.QUALITY_READER else acquire):
                self.assertEqual(eng.quality(root, doc, fetch=True), proofs)
            self.assertEqual([row[0] for row in calls], ['proof', 'original primary18', 'verify primary', 'proof', 'fetch verify replica'])
            self.assertEqual(calls[3][2], dict(training_reuse=root / 'proof_roots/primary/training-origin',
                                              primary_reuse=root / 'quality/primary'))

    def test_failure_stops_before_replica_acquisition_or_candidate_access(self):
        doc, proofs = dual_fixture(); calls = []
        doc['quality_reader_sha256'] = eng.sha(Path(reader.__file__).read_bytes())
        verifier = SimpleNamespace(__file__=reader.__file__,
            verify_quality_original=lambda *args: dict(panel_gate='NO_PROMOTION'))
        acquire = SimpleNamespace(fetch=lambda *args, **kwargs: calls.append(args[1]), fetch_files=lambda *args: None)
        bound = dict(primary_quality_files={}, primary_quality_transport={})
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(eng, 'proof_bindings', return_value={'replication': bound}), \
                 patch.object(eng, 'module', side_effect=lambda name: verifier if name == eng.QUALITY_READER else acquire):
                with self.assertRaisesRegex(ValueError, 'before the next panel'):
                    eng.quality(Path(temporary), doc, fetch=True)
        self.assertEqual(calls, ['primary'])

    def test_readback_accepts_only_exact_scoped_temporary_training_child_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quality = root / 'english_v6_b28_quality_readback.py'; quality.write_bytes(b'# invented quality reader\n')
            train = root / 'english_v6_b28_training_readback.py'; train.write_bytes(b'# invented TRAIN reader\n')
            original_root = root / 'proof_roots/primary/training-origin'
            evidence = original_root / 'trained'; evidence.mkdir(parents=True)
            staged = root / 'temporary-derived-inputs'; staged.mkdir()
            lock_path = root / 'lock.json'; lock = {'invented': 'TRAIN identity shape'}
            lock_path.write_bytes(canonical(lock))
            original_files, staged_files = {'original71': 'fixture'}, {'input66plusderived5': 'fixture'}
            profile = SimpleNamespace(training_child=lambda: dict(directory=evidence, inputs=staged, files=staged_files),
                inventory=lambda path: staged_files if path == staged else original_files)
            verifier = SimpleNamespace(__file__=str(quality), profile=profile, validate_lock=lambda value: value)
            training = dict(training_reader_sha256=eng.sha(train.read_bytes()), training_lock=lock, files=original_files)
            command = [sys.executable, '-I', '-S', '-B', str(train), '--verify-worker', str(evidence), str(lock_path), str(staged)]
            with patch.object(reader, 'QUALITY_READER_SHA', eng.sha(quality.read_bytes())), \
                 patch.object(subprocess, 'Popen', return_value='fake-child-no-execution') as invoked:
                with reader.offline_engineering(root, {'primary': {'panel': 'primary'}, 'replication': {'panel': 'replication'}}, verifier, training):
                    wrong = list(command); wrong[-1] = str(original_root / 'qualification')
                    with self.assertRaises(ValueError): subprocess.Popen(wrong)
                    self.assertEqual(subprocess.Popen(command), 'fake-child-no-execution')
                    with self.assertRaises(ValueError): subprocess.Popen(command)
                    with self.assertRaises(ValueError): subprocess.Popen([sys.executable, '-c', 'pass'])
                    with self.assertRaises(ValueError): __import__('numpy')
            self.assertEqual(invoked.call_count, 1)


if __name__ == '__main__':
    unittest.main()
