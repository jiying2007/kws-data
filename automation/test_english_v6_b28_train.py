"""B28 once-only provenance contracts; source AST and invented bytes only."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import inspect
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_train as train
import english_v6_b28_producer as producer


def closed_opening():
    return dict(schema='english-v6-b28-after-original-b27-no-promotion-v1', condition=train.CONDITION,
        enabled=False, quality_lock=None, quality_reader_sha256=None, original_failure_report_sha256=None,
        original_primary_files=None, original_primary_transport=None, original_training_binding_sha256=None,
        original_B27_challenger_pt_sha256=None, quality_import_adapter_path=None,
        quality_import_adapter_sha256=None, quality_import_adapter_report_sha256=None)


def closed_binding():
    return dict(schema='english-v6-b28-prospective-assessment-binding-v1',
        recipe_path=train.DESIGN.relative_to(train.ROOT).as_posix(), recipe_sha256=None,
        frozen=False, optimizer_allowed=False, assessment=None, files={}, original_B27_source_exclusion=None)


def original_event():
    parent, head = 'a' * 40, 'b' * 40
    env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=train.REPO, GITHUB_EVENT_NAME='push',
        GITHUB_REF=train.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=head, GITHUB_RUN_ID='123')
    event = dict(before=parent, after=head, ref=train.BRANCH, forced=False, deleted=False,
        repository=dict(full_name=train.REPO, private=False))
    return parent, head, env, event


class ExecutionContracts(unittest.TestCase):
    def test_actual_prepared_design_has_frozen_math_and_original_budget(self):
        value = train.design(False)
        self.assertEqual(value['budget'], train.b24.design()['budget'])
        self.assertEqual((value['optimizer_updates'], value['maximum_new_model_forwards']), (192, 1847))
        self.assertEqual(value['input_archives'], train.b24.design()['input_archives'])
        self.assertEqual(len(train.OBSERVATIONS), 7)
        self.assertIn('NATURAL_OBJECTIVE_TRACE.json', train.OBSERVATIONS)
        self.assertEqual(value['fit_adapter_sha256'], train.objective_fit.contract()['fit_adapter_sha256'])
        self.assertTrue({'numpy', 'torch', 'onnx', 'onnxruntime'}.isdisjoint(sys.modules))

    def test_unbound_design_fails_before_any_guard_or_network(self):
        with patch.object(train, 'DESIGN_SHA', None), patch.object(train, 'gh', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError, 'unbound'):
                train.guard('a' * 40)

    def test_closed_opening_and_missing_metadata_are_separate_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'opening.json'
            path.write_bytes(train.encoded(closed_opening()))
            with patch.object(train, 'OPENING', path):
                self.assertIs(train.opening(False)['enabled'], False)
                with self.assertRaisesRegex(ValueError, 'unbound'):
                    train.opening()
                value = closed_opening(); value['original_primary_files'] = {'fake': {}}
                path.write_bytes(train.encoded(value))
                with self.assertRaisesRegex(ValueError, 'fabricated'):
                    train.opening(False)
            path = Path(temporary) / 'binding.json'
            path.write_bytes(train.encoded(closed_binding()))
            with patch.object(train, 'BINDING', path), patch.object(train, 'DESIGN_SHA', None):
                self.assertIs(train.assessment_binding(False)['frozen'], False)
                with self.assertRaisesRegex(ValueError, 'fresh B28'):
                    train.assessment_binding()
                value = closed_binding(); value['files'] = {'invented.json': 'a' * 64}
                path.write_bytes(train.encoded(value))
                with self.assertRaisesRegex(ValueError, 'invented'):
                    train.assessment_binding(False)

    def test_original_guard_code_objects_are_reused_without_changing_old_globals(self):
        for name in ('guard', 'consume', 'verify_consumed'):
            function, original = train._reuse(name), getattr(train.b24, name)
            self.assertIs(function.__code__, original.__code__)
            self.assertIs(function.__closure__, original.__closure__)
            self.assertEqual(function.__globals__['WORKFLOW'], train.WORKFLOW)
            self.assertEqual(function.__globals__['MARKER'], train.MARKER)
            self.assertNotEqual(original.__globals__['WORKFLOW'], train.WORKFLOW)
        with self.assertRaises(ValueError):
            train._reuse('fit')

    def test_once_guard_rejects_attempt_two_force_private_wrong_branch_and_parent(self):
        parent, head, env, event = original_event()
        with patch.object(train, 'design'), patch.object(train, 'opening'), patch.object(train, 'assessment_binding'):
            value = train.guard(parent, env, event, parent)
            self.assertEqual((value['run_head'], value['run_attempt']), (head, 1))
            for key, replacement in [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_EVENT_NAME', 'workflow_dispatch'),
                                     ('GITHUB_REF', train.b27.BRANCH)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    train.guard(parent, dict(env, **{key: replacement}), event, parent)
            for key, replacement in [('forced', True), ('deleted', True), ('before', 'c' * 40)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    train.guard(parent, env, dict(event, **{key: replacement}), parent)
            private = dict(event, repository=dict(full_name=train.REPO, private=True))
            with self.assertRaises(ValueError):
                train.guard(parent, env, private, parent)

    def test_source_and_durable_once_marker_precede_scientific_imports(self):
        node = ast.parse(inspect.getsource(train.execute)).body[0]
        positions = {}
        for index, statement in enumerate(node.body):
            text = ast.unparse(statement)
            for name in ('verify_consumed(', 'source_preflight(', 'import numpy', 'import torch', 'objective_fit.fit('):
                if name in text and name not in positions:
                    positions[name] = index
        self.assertLess(positions['verify_consumed('], positions['source_preflight('])
        self.assertLess(positions['source_preflight('], positions['import numpy'])
        self.assertLess(positions['import torch'], positions['objective_fit.fit('])
        with patch.object(train, 'verify_consumed', side_effect=ValueError('inert guard')):
            with self.assertRaisesRegex(ValueError, 'inert guard'):
                train.execute(Path('/invented-input'), Path('/invented-output'), 'a' * 40)

    def test_exact_five_original_b24_members_are_retained_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / 'proof/training-origin/trained'; parent.mkdir(parents=True)
            payload = io.BytesIO()
            members = {'original_b24/' + name: ('INVENTED ' + name).encode() for name in train.OLD_B24_FILES}
            with zipfile.ZipFile(payload, 'w') as archive:
                for name, raw in members.items(): archive.writestr(name, raw)
                archive.writestr('FETCH.json', b'{}')
            raw = payload.getvalue()
            (parent / 'TRAIN_receipts.zip').write_bytes(raw)
            bound = dict(files={'trained/TRAIN_receipts.zip': dict(bytes=len(raw), sha256=train.sha(raw))})
            with patch.object(train.profile, 'binding', return_value=bound):
                first = train.retain_original_b24(root, create=True)
                self.assertEqual(set(first), train.OLD_B24_FILES)
                self.assertEqual(train.retain_original_b24(root), first)
                (root / 'original_b24/TRAIN_models.zip').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'unaltered'):
                    train.retain_original_b24(root)

    def test_original_receipt_member_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / 'proof/training-origin/trained'; parent.mkdir(parents=True)
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, 'w') as archive:
                for name in train.OLD_B24_FILES - {'TRAIN_models.zip'}:
                    archive.writestr('original_b24/' + name, b'invented')
            raw = payload.getvalue(); (parent / 'TRAIN_receipts.zip').write_bytes(raw)
            bound = dict(files={'trained/TRAIN_receipts.zip': dict(bytes=len(raw), sha256=train.sha(raw))})
            with patch.object(train.profile, 'binding', return_value=bound), self.assertRaisesRegex(ValueError, 'five original'):
                train.retain_original_b24(root, create=True)

    def test_full_readback_is_required_and_a_summary_or_pass_cannot_open_failure(self):
        wrong = dict(schema='english-v6-b27-verified-original-quality-v1', panel='primary', panel_gate='PASS')
        certificate = dict(schema=train.B27_IMPORT_REPORT_SCHEMA,
            original_reader_sha256=train.B27_READER_SHA, import_scope_adapter_sha256='b' * 64,
            original_report=wrong, original_report_sha256=train.sha(train.profile.canonical(wrong)),
            original_standalone_continuation_succeeded=False)
        bound = dict(quality_lock=dict(panel='primary'), original_primary_files={'placeholder': {}},
            quality_reader_sha256=train.B27_READER_SHA,
            quality_import_adapter_path=train.B27_IMPORT_ADAPTER_PATH, quality_import_adapter_sha256='b' * 64,
            quality_import_adapter_report_sha256=train.sha(train.profile.canonical(certificate)),
            original_failure_report_sha256=certificate['original_report_sha256'])
        callbacks = []
        def verify_once(*args):
            callbacks.append(args)
            return args[-1]()
        called = []
        def original(directory, lock, proof):
            called.append((directory, lock, proof))
            return certificate
        adapter = SimpleNamespace(__file__='/invented/adapter.py', verify_original=original,
            validate_report=lambda *_: None)
        with patch.object(train, 'opening', return_value=bound), patch.object(train, 'external_inputs'), \
             patch.object(train.evidence_scope, 'verify_once', side_effect=verify_once), \
             patch.object(train, 'quality_import_adapter', return_value=adapter), \
             patch.object(train.quality, 'verify_quality_original', side_effect=AssertionError('old invocation must remain recorded as failed')):
            with self.assertRaisesRegex(ValueError, 'terminal NO_PROMOTION'):
                train.previous_failure(Path('/invented'))
            self.assertEqual(called, [(Path('/invented/proof/primary-quality'), bound['quality_lock'], Path('/invented/proof'))])
            self.assertEqual(callbacks[0][0], 'b27_primary_quality')
            self.assertEqual(callbacks[0][4:7], (Path(adapter.__file__), 'b' * 64, bound['quality_import_adapter_report_sha256']))
            certificate['original_standalone_continuation_succeeded'] = True
            with self.assertRaisesRegex(ValueError, 'separate exact identities'):
                train.previous_failure(Path('/invented'))

    def test_failure_certification_binding_keeps_original_and_adapter_identities_separate(self):
        bound = dict(quality_reader_sha256=train.B27_READER_SHA, original_failure_report_sha256='a' * 64,
            quality_import_adapter_path=train.B27_IMPORT_ADAPTER_PATH, quality_import_adapter_sha256='b' * 64,
            quality_import_adapter_report_sha256='c' * 64)
        pin = train.failure_certification(bound)
        self.assertEqual(pin['original_reader_sha256'], train.B27_READER_SHA)
        self.assertEqual(pin['original_report_sha256'], 'a' * 64)
        self.assertEqual(pin['adapter_path'], train.B27_IMPORT_ADAPTER_PATH)
        self.assertEqual(pin['adapter_sha256'], 'b' * 64)
        self.assertEqual(pin['adapter_report_schema'], train.B27_IMPORT_REPORT_SCHEMA)
        self.assertEqual(pin['adapter_report_sha256'], 'c' * 64)
        self.assertIs(pin['original_standalone_continuation_succeeded'], False)
        for key in ('new_model_calls', 'new_source_calls', 'new_observer_calls', 'new_optimizer_updates'):
            self.assertEqual(pin[key], 0)
        actual = train.quality_import_adapter()
        self.assertEqual(train.sha(Path(actual.__file__).read_bytes()), train.B27_IMPORT_ADAPTER_SHA)
        self.assertEqual(train.sha(Path(train.quality.__file__).read_bytes()), train.B27_READER_SHA)
        self.assertEqual(actual.READER_SHA, train.B27_READER_SHA)
        with patch.object(train, 'B27_IMPORT_ADAPTER_SHA', None):
            with self.assertRaisesRegex(ValueError, 'unbound'):
                train.quality_import_adapter()

    def test_source_history_reopens_complete_catalog_scope_before_source_equality(self):
        source_spec = train.encoded(dict(inputs=dict(source=dict(id=123))))
        fetched = train.encoded(dict(source_metadata=dict(source=dict(original='invented'))))
        expected = dict(all_original72_people=['invented'], complete_original120_pcm=['invented'])
        active = [False]
        @contextmanager
        def scope(path):
            self.assertEqual(path, Path('/invented/metadata/train360'))
            active[0] = True
            try: yield
            finally: active[0] = False
        def checked_binding(**kwargs):
            self.assertTrue(active[0]); self.assertEqual(kwargs, dict(require_catalog=True))
            return dict(original_B27_source_exclusion=expected)
        def original_source(path, metadata, pin):
            self.assertTrue(active[0])
            self.assertEqual(path, Path('/invented/proof/primary-quality/inputs/source.zip'))
            self.assertEqual((metadata, pin), (dict(original='invented'), dict(id=123)))
            return expected
        class Archive:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, name):
                return source_spec if name.endswith('_inputs.json') else fetched
        metadata = SimpleNamespace(catalog_scope=scope, verify_original_source=original_source)
        opening = dict(quality_lock=dict(producer={}, inputs_sha256=train.sha(source_spec)))
        with patch.object(train, 'opening', return_value=opening), \
             patch.object(train.importlib, 'import_module', return_value=metadata), \
             patch.object(train, 'assessment_binding', side_effect=checked_binding), \
             patch.object(train.quality, 'original_archive', return_value=Archive()):
            self.assertEqual(train.source_history(Path('/invented')), expected)
        self.assertFalse(active[0])

    def test_role_caps_remain_fixed_and_complete_external_proof_is_not_packaged(self):
        small = {'invented.py': dict(bytes=100)}
        envelope = producer.capacity_receipt(small)
        self.assertEqual(envelope['limit'], 128000000)
        self.assertEqual(envelope['original_input_archives_embedded'], 0)
        with self.assertRaises(ValueError):
            producer.capacity_receipt({'oversized.py': dict(bytes=128000000)})
        with patch.object(train, 'retain_original_b24', return_value={'TRAIN_models.zip': dict(bytes=121000000)}), \
             patch.object(train, 'design', return_value=dict(archive_caps=dict(new_receipt_metadata_reserve=8000000))):
            with self.assertRaisesRegex(ValueError, '128MB'):
                train.receipt_capacity(Path('/invented'))

    def test_real_inherited_producer_closure_retains_exact_seven_bounded_archive_resources(self):
        names = producer.paths(False)
        manifest = producer.original_primary_manifest()
        pins = producer.original_archive_resources(names, manifest)
        expected = {
            'automation/english_v6_b21_continuation_failure/opened-original.zip',
            'automation/english_v6_b21_continuation_failure/producer-original.zip',
            'automation/english_v6_b21_continuation_probe/original.zip',
            'automation/english_v6_b21_fixed_mirror_failure/producer-original.zip',
            'automation/english_v6_b21_fixed_mirror_failure/transport-original.zip',
            'automation/english_v6_b27_source_preconsumption_failure/producer.zip',
            'automation/english_v6_b27_source_preconsumption_failure/transport.zip',
        }
        self.assertEqual(set(pins), expected)
        self.assertEqual(sum(pin['bytes'] for pin in pins.values()), 10515449)
        self.assertTrue(set(producer.previous.paths()) <= names)
        for role in ('primary', 'source'):
            self.assertTrue(set(manifest[role]['files']) <= names)
            self.assertTrue(set(manifest[role].get('resources', {})) <= names)
        for upstream in manifest['upstream_manifests'].values():
            self.assertTrue(set(upstream['files']) <= names)
        collector = producer.metadata._collector()
        self.assertEqual(len(collector.paths()), 13)
        self.assertTrue(set(collector.paths()) <= names)
        old_binding = train.failure.document((train.HERE / 'english_v6_b27_assessment_binding.json').read_bytes())
        self.assertTrue(set(old_binding['files']) <= names)
        self.assertIn(producer.metadata.PREVIOUS, names)
        self.assertTrue(set(collector.FROZEN) <= names)
        self.assertIn(train.B27_IMPORT_ADAPTER_PATH, names)
        self.assertIn('automation/test_english_v6_b27_primary_no_promotion_readback.py', names)
        envelope = producer.capacity_receipt({name: dict(bytes=(train.ROOT / name).stat().st_size) for name in names})
        self.assertEqual(envelope['limit'], 128000000)
        self.assertLess(envelope['zip_bound'], envelope['limit'])
        self.assertTrue({'numpy', 'torch', 'onnx', 'onnxruntime'}.isdisjoint(sys.modules))

    def test_inherited_archive_allowlist_rejects_omission_dynamic_zip_and_tampering(self):
        manifest = producer.original_primary_manifest()
        names = {name for name in manifest['resources'] if Path(name).suffix.lower() == '.zip'}
        for altered in (names - {next(iter(names))}, names | {'inputs/proof/training-origin/trained/TRAIN_models.zip'},
                        names | {'inputs/metadata/train360/originals/METADATA.ZIP'}):
            with self.subTest(names=altered), self.assertRaisesRegex(ValueError, 'seven exact'):
                producer.original_archive_resources(altered, manifest)
        target = train.ROOT / 'automation/english_v6_b21_continuation_failure/opened-original.zip'
        original = Path.read_bytes
        def changed(path):
            raw = original(path)
            return raw[:-1] + bytes([raw[-1] ^ 1]) if path == target else raw
        with patch.object(Path, 'read_bytes', changed), self.assertRaisesRegex(ValueError, 'all original inherited ZIP'):
            producer.original_archive_resources(names, manifest)
        target = train.ROOT / producer.ORIGINAL_PRIMARY_MANIFEST
        with patch.object(Path, 'read_bytes', changed), self.assertRaisesRegex(ValueError, 'exact complete original'):
            producer.original_primary_manifest()

    def test_inert_workflow_template_only_two_replacements_and_marker_before_torch(self):
        template = (train.ROOT / producer.TEMPLATE).read_bytes()
        self.assertEqual(template.count(producer.TOKEN), 2)
        checked_in = (train.ROOT / train.WORKFLOW).read_bytes()
        if checked_in != template:
            # On the single real first push, the runner is the reviewed child
            # whose only edit is substitution of its prepared parent token.
            self.assertEqual(checked_in, producer.workflow(os.environ.get('PREPARED_PARENT', '')))
        self.assertEqual(producer.workflow('a' * 40), template.replace(producer.TOKEN, b'a' * 40))
        with self.assertRaises(ValueError): producer.workflow('main')
        text = template.decode()
        self.assertLess(text.index('name: ' + train.MARKER), text.index('torch==2.9.1+cpu'))
        self.assertLess(text.index('torch==2.9.1+cpu'), text.index(' execute --inputs'))
        self.assertNotIn('b28-inputs/proof/', text)
        self.assertEqual(text.count('english-v6-b28-train-producer-'), 1)
        self.assertIn('include-hidden-files: true', text)
        self.assertIn('fetch-depth: 2', text)


if __name__ == '__main__':
    unittest.main()
