"""Offline entry metadata only; no pipeline checkout, artifacts or runtime."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('delivery_entry_contract', HERE / 'english_v6_b24r1_delivery_entry.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class EntryTests(unittest.TestCase):
    def test_committed_registration_is_complete_or_explicitly_inert(self):
        value = m.contract(json.loads(m.SPEC.read_bytes()))
        if not value['enabled']:
            self.assertIsNone(value['pipeline_commit']); self.assertIsNone(value['manifest_sha256'])
            self.assertIsNone(value['pipeline_files'])

    def test_null_binding_cannot_smuggle_partial_selection(self):
        value = dict(schema=m.SCHEMA, enabled=False, pipeline_repository='jiying2007/kws-pipeline',
                     pipeline_commit=None, manifest_sha256=None, pipeline_files=None)
        m.contract(value)
        for key in ('pipeline_commit', 'manifest_sha256', 'pipeline_files'):
            changed = dict(value); changed[key] = 'invented'
            with self.assertRaises(ValueError): m.contract(changed)
        changed = dict(value); changed['enabled'] = True
        with self.assertRaises(ValueError): m.contract(changed)

    def test_reviewed_checkout_cannot_be_branch_expression_or_external_repo(self):
        value = dict(schema=m.SCHEMA, enabled=True, pipeline_repository='jiying2007/kws-pipeline',
            pipeline_commit='a' * 40, manifest_sha256='b' * 64,
            pipeline_files={'research/english_v6_delivery/once.py': dict(bytes=12, sha256='c' * 64,
                git_blob_sha1='d' * 40, mode='100644')})
        m.contract(value)
        for key, bad in (('pipeline_commit', 'main'), ('pipeline_commit', 'a' * 40 + '\nx=y'),
                         ('pipeline_repository', 'other/repo'), ('pipeline_files', {})):
            changed = copy.deepcopy(value); changed[key] = bad
            with self.assertRaises(ValueError): m.contract(changed)

    def test_template_has_only_two_parent_substitutions_and_no_other_activation_route(self):
        template = (HERE / 'english_v6_b24r1_delivery_workflow_template.yml').read_bytes()
        workflow = (HERE.parent / '.github/workflows/english-v6-b24r1-delivery-once.yml').read_bytes()
        token = b'__B24R1_REVIEWED_DELIVERY_PARENT__'
        self.assertEqual(template.count(token), 2)
        if workflow != template:
            parent = re.search(rb"PREPARED_PARENT: '([a-f0-9]{40})'", workflow)
            self.assertIsNotNone(parent)
            self.assertEqual(workflow, template.replace(token, parent.group(1)))
        text = template.decode()
        for guard in ("github.event_name == 'push'", "github.repository == 'jiying2007/kws-data'",
                      "github.event.repository.private == false", "github.run_attempt == 1", 'needs: contracts'):
            self.assertIn(guard, text)
        self.assertNotIn('workflow_dispatch:', text); self.assertNotIn('workflow_run:', text)
        self.assertNotIn('schedule:', text); self.assertNotIn('secrets.', text)
        self.assertNotIn('contents: write', text); self.assertNotIn('actions: write', text)
        self.assertIn("ref: '${{ steps.selected.outputs.pipeline_commit }}'", text)
        self.assertIn("GH_TOKEN: '${{ github.token }}'", text)


if __name__ == '__main__': unittest.main()
