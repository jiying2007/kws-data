"""Offline original-artifact binding fixtures; no source or model operations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent

def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

binder = load('bind_b23_primary.py', 'b23_binder_test')
primary = load('english_v6_b23_primary_eval.py', 'b23_primary_binding_test')


def fixture():
    groups = {}
    for name in binder.GROUPS:
        base = primary.BASELINES.get(name)
        run_id = base['run_id'] if base else (123 if name == 'train' else 456)
        head = base['run_head'] if base else ('a' if name == 'train' else 'b') * 40
        branch = primary.TRAIN_BRANCH if name == 'train' else primary.SOURCE_BRANCH
        run = dict(id=run_id, head_sha=head, head_branch=branch, event='push', run_attempt=1,
                   status='completed', conclusion='success', repository=dict(full_name=primary.REPO, private=False))
        items = []
        for index, role in enumerate(binder.ROLES):
            if binder.GROUP_FOR[role] != name:
                continue
            pin = base if base else dict(id=1000 + index, run_id=run_id, run_head=head, bytes=10000, sha256=f'{index + 1:064x}')
            item = dict(id=pin['id'], name='baseline' if base else binder.artifact_name(role, run_id),
                        digest='sha256:' + pin['sha256'], expired=False, size_in_bytes=pin['bytes'],
                        workflow_run=dict(id=run_id, head_sha=head))
            items.append(item)
        groups[name] = dict(run=run, commit=dict(sha=head), artifacts=dict(artifacts=items))
    return dict(schema='b23-original-historical-control-primary-binding-inputs-v1', groups=groups,
                archives={name:name + '.zip' for name in binder.ROLES})


class BindingTests(unittest.TestCase):
    def test_exact_code_pin_and_single_entrypoint(self):
        self.assertEqual(binder.DRIVER_SHA, binder.sha((HERE / 'english_v6_b23_primary_eval.py').read_bytes()))
        self.assertEqual((HERE / 'bind_b23_primary.py').read_text().count('if __name__ == "__main__":'), 1)

    def test_twelve_actual_original_roles_keep_both_producers(self):
        _, pins, _ = binder.metadata(fixture(), HERE, primary)
        self.assertEqual(len(pins), 12)
        self.assertEqual(pins['B0'], primary.BASELINES['B0'])
        self.assertEqual(pins['B9'], primary.BASELINES['B9'])
        self.assertEqual(pins['pair']['run_id'], pins['train_producer']['run_id'])
        self.assertEqual(pins['source']['run_id'], pins['source_producer']['run_id'])

    def test_observer_namespace_belongs_to_new_b23_source(self):
        for provider in binder.PROVIDERS:
            self.assertEqual(binder.artifact_name('native_' + provider, 456),
                             'b23-primary-dev-observer-' + provider)
        value = fixture()
        for item in value['groups']['source']['artifacts']['artifacts']:
            item['name'] = item['name'].replace('b23-primary-dev-observer-', 'b22-primary-dev-observer-')
        with self.assertRaises(ValueError): binder.metadata(value, HERE, primary)

    def test_observer_from_another_run_is_rejected(self):
        value = fixture()
        wrong = next(item for item in value['groups']['source']['artifacts']['artifacts']
                     if item['name'].endswith('whisper'))
        wrong['workflow_run']['head_sha'] = 'f' * 40
        with self.assertRaises(ValueError): binder.metadata(value, HERE, primary)

    def test_missing_or_duplicate_producer_is_rejected(self):
        for mutation in ('missing', 'duplicate'):
            value = fixture()
            items = value['groups']['source']['artifacts']['artifacts']
            item = next(v for v in items if 'source-producer-' in v['name'])
            if mutation == 'missing': items.remove(item)
            else: items.append(copy.deepcopy(item))
            with self.assertRaises(ValueError): binder.metadata(value, HERE, primary)

    def test_saved_binding_denies_model_and_network(self):
        with binder.source_protection():
            with self.assertRaises(ValueError): __import__('torch')
            with self.assertRaises(ValueError): __import__('numpy')
            import subprocess
            with self.assertRaises(AssertionError): subprocess.run(['gh', 'api', 'anything'])


if __name__ == '__main__':
    unittest.main()
