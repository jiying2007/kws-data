"""No-inference contracts for the B22 original-audio/text-only continuation."""
from __future__ import annotations
import ast
import base64
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import english_v6_b22_source as original
import english_v6_b22_source_continuation as source
import english_v6_b22_source_continuation_ancestry as lineage
import english_v6_b22_source_continuation_observe as observe
import english_v6_b22_source_continuation_producer as producer
import english_v6_b22_source_dynamic_preflight as dynamic


def function(name, filename):
    return next(node for node in ast.parse((HERE / filename).read_bytes()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def identity():
    return dict(run_id='123', run_head='b' * 40, prepared_parent='a' * 40,
                run_branch=lineage.BRANCH, run_attempt=1)


def registration():
    return dict(identity(), source_rows=120, max_calls=120, source_manifest_sha256='c' * 64,
        original_source_execution=lineage.origin.ORIGINAL_SOURCE_EXECUTION,
        origin_binding_sha256=lineage.origin_binding_sha())


def marker(expected, member):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr(member, json.dumps(expected))
    raw = buffer.getvalue()
    ident = expected['source_execution']
    return dict(artifact=dict(id=1, name=expected['marker_name'], expired=False,
        workflow_run=dict(id=int(ident['run_id']), head_sha=ident['run_head']),
        size_in_bytes=len(raw), digest='sha256:' + lineage.digest(raw)),
        original_zip_base64=base64.b64encode(raw).decode(), receipt=expected)


class ContinuationContracts(unittest.TestCase):
    def test_frozen_science_and_original_identity_unchanged(self):
        self.assertEqual(source.SCHEMA, original.SCHEMA)
        self.assertEqual(source.BATCH, original.BATCH)
        self.assertEqual(source.source_plan(), original.source_plan())
        contract = observe.source_check()
        scientific = {k: v for k, v in contract.items() if k not in ('schema', 'observer_sha256', 'source_indices')}
        self.assertEqual(scientific, lineage.read_plan()['observer_scientific_contract'])
        self.assertNotEqual(lineage.BRANCH, original.lineage.BRANCH)
        self.assertNotEqual(lineage.MARKER, original.lineage.MARKER)
        self.assertEqual(lineage.origin.local()['remaining_text_forward_calls'], 360)

    def test_admission_computation_unchanged(self):
        old = function('screen_document', 'english_v6_b22_source.py')
        new = function('screen_document', 'english_v6_b22_source_continuation.py')
        self.assertEqual([ast.dump(x, include_attributes=False) for x in old.body[:-1]],
                         [ast.dump(x, include_attributes=False) for x in new.body[:-1]])

    def test_native_capture_and_stop_body_unchanged(self):
        old = function('observe', 'english_v6_b22_source_observe.py')
        new = function('observe', 'english_v6_b22_source_continuation_observe.py')
        def capture(node):
            at = next(i for i, n in enumerate(node.body) if isinstance(n, ast.If)
                and ast.dump(n.test) == ast.dump(ast.parse('initialization_error is None', mode='eval').body))
            return [ast.dump(x, include_attributes=False) for x in node.body[at:]]
        self.assertEqual(capture(old), capture(new))
        self.assertNotIn('_load_phone(', ast.unparse(new))

    def test_no_generation_decode_or_training_entrypoints(self):
        text = (HERE / 'english_v6_b22_source_continuation.py').read_text()
        self.assertNotIn('def generate(', text)
        self.assertNotIn('natural.extract(', text)
        self.assertNotIn('render_panel(', text)
        self.assertEqual(lineage.PROVIDERS, ('qwen', 'whisper', 'parakeet'))
        self.assertEqual(lineage.PHONES, ('xlsr', 'lv60'))

    def test_first_public_push_guard_rejects_original_branch_and_rerun(self):
        ident = identity()
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=lineage.REPO,
            GITHUB_EVENT_NAME='push', GITHUB_REF=lineage.BRANCH, GITHUB_RUN_ATTEMPT='1',
            GITHUB_SHA=ident['run_head'], GITHUB_RUN_ID=ident['run_id'])
        event = dict(before=ident['prepared_parent'], after=ident['run_head'], ref=lineage.BRANCH,
            forced=False, deleted=False, repository=dict(full_name=lineage.REPO, private=False))
        self.assertEqual(lineage.execution_guard(ident['prepared_parent'], env, event, ident['prepared_parent']), ident)
        for bad in (dict(env, GITHUB_RUN_ATTEMPT='2'), dict(env, GITHUB_REF=original.lineage.BRANCH)):
            with self.assertRaises(ValueError):
                lineage.execution_guard(ident['prepared_parent'], bad, event, ident['prepared_parent'])
        for bad in (dict(event, forced=True), dict(event, repository=dict(full_name=lineage.REPO, private=True))):
            with self.assertRaises(ValueError):
                lineage.execution_guard(ident['prepared_parent'], env, bad, ident['prepared_parent'])

    def test_only_zero_call_text_can_get_new_durable_budget(self):
        reg = registration()
        for provider in lineage.PROVIDERS:
            expected = lineage.observer_document(provider, reg)
            self.assertEqual(expected['maximum_native_calls'], 120)
            self.assertEqual(expected['new_phone_calls'], 0)
            receipt = marker(expected, 'OBSERVER_CONSUMPTION.json')
            self.assertEqual(lineage.validate_observer_budget(receipt, provider, reg), expected)
            with self.assertRaises(ValueError):
                lineage.validate_observer_budget(receipt, 'whisper' if provider != 'whisper' else 'qwen', reg)
        for provider in lineage.PHONES:
            with self.assertRaises(ValueError):
                lineage.observer_document(provider, reg)
            with self.assertRaises(ValueError):
                lineage.marker_name(provider)
        with self.assertRaises(ValueError):
            lineage.observer_document('qwen', dict(reg, source_rows=119, max_calls=119))

    def test_reused_phone_bytes_cannot_be_relabelled_or_modified(self):
        expected = {'RESULT.json': b'{"run":"original"}\n', 'rows/a/RAW.txt': b'original fixture'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lineage.write_archive(expected, root / 'phone')
            with patch.object(lineage, 'archive_files', return_value=expected):
                lineage.require_phone_copy(root, 'xlsr', root / 'phone')
                (root / 'phone/RESULT.json').write_bytes(b'{"run":"continuation"}\n')
                with self.assertRaises(ValueError):
                    lineage.require_phone_copy(root, 'xlsr', root / 'phone')

    def test_complete_original_source_and_all19_runtime_helpers_captured(self):
        tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines()
        paths = producer.select_paths(tracked)
        self.assertTrue(set(lineage.origin.producer_manifest()['files']) <= paths)
        self.assertTrue(set(dynamic.required_paths()) <= paths)
        self.assertTrue(set(lineage.training_binding()['training']['artifacts']) == {'models', 'producer', 'receipts', 'consumed'})
        for missing in (dynamic.CAMPAIGN, dynamic.RUNNER):
            with self.assertRaises(ValueError):
                producer.select_paths([name for name in tracked if name != missing])
        self.assertFalse(any(name.startswith(producer.FUTURE_PREFIXES) for name in paths))

    def test_once_template_inert_and_marker_precedes_native_install(self):
        template = (ROOT / producer.TEMPLATE).read_bytes()
        runtime = (ROOT / lineage.WORKFLOW).read_bytes()
        self.assertEqual(template.count(producer.TOKEN), 2)
        self.assertTrue(runtime == template or bool(__import__('re').fullmatch(
            __import__('re').escape(template).replace(__import__('re').escape(producer.TOKEN), b'[0-9a-f]{40}'), runtime)))
        text = template.decode()
        self.assertIn('provider: [qwen, parakeet, whisper]', text)
        self.assertNotIn('piper-tts', text)
        self.assertNotIn('transformers==', text)
        self.assertIn('phones/xlsr/', text)
        self.assertIn('phones/lv60/', text)
        self.assertLess(text.index('name: english-v6-b22-primary-text-continuation-${{ matrix.provider }}-consumed'),
                        text.index('name: Install one pinned native observer runtime'))
        self.assertIn('/b22-source-producer/repository/automation/english_v6_b22_source_continuation.py', text)
        self.assertEqual(text.count('include-hidden-files: true'), text.count('retention-days: 90'))

    def test_real_dynamic_loader_imports_without_native_runtime(self):
        raw = subprocess.check_output([sys.executable, '-SIB', str(HERE / 'english_v6_b22_source_dynamic_preflight.py')])
        value = json.loads(raw)
        self.assertEqual(value['status'], 'PASS_IMPORT_LOCK_RUNNER_ONLY')
        self.assertTrue(value['code_reads_only'] and value['no_model'] and value['no_network'])
        self.assertFalse(value['model_construction_verified'] or value['acoustic_evidence'])

    @unittest.skipUnless(os.environ.get('B22_ORIGIN_DIRECTORY'), 'original seven ZIPs supplied only in the explicit readback contract job')
    def test_actual_origin_and_exact_reuse_without_native_calls(self):
        origin = Path(os.environ['B22_ORIGIN_DIRECTORY']).resolve()
        report = lineage.verify_origin(origin)
        self.assertEqual(report['phone_forward_counts'], dict.fromkeys(lineage.PHONES, 120))
        self.assertEqual(report['text_forward_counts'], dict.fromkeys(lineage.PROVIDERS, 0))
        with tempfile.TemporaryDirectory(prefix='b22-continuation-copy-readback-') as temporary:
            root = Path(temporary)
            (root / 'consumed').mkdir()
            expected = lineage.consumption_document(identity(), report)
            (root / 'consumed/CONSUMPTION.json').write_text(json.dumps(expected))
            with patch.object(lineage, 'execution_guard', return_value=identity()), \
                 patch.object(lineage, 'one_execution', return_value={}), \
                 patch.object(lineage, 'uploaded_marker', side_effect=lambda a, e, m: marker(e, m)):
                lineage.reuse(origin, root / 'source', root / 'phones', root / 'consumed', identity()['prepared_parent'])
            actual = source.evaluation_preflight(root / 'source')
            self.assertEqual(actual['source_execution'], identity())
            self.assertEqual(actual['original_source_execution'], report['original_source_execution'])
            self.assertEqual(len(actual['rows']), 120)
            self.assertEqual(source.registration(root / 'source')['run_id'], '123')
            for provider in lineage.PHONES:
                lineage.require_phone_copy(root / 'source', provider, root / 'phones' / provider)
            altered = root / 'source/MANIFEST.json'
            altered.write_bytes(altered.read_bytes() + b'\n')
            with self.assertRaises(ValueError):
                source.evaluation_preflight(root / 'source')


if __name__ == '__main__':
    unittest.main()
