"""Bounded source-only B22 contracts; no audio/model load, network or scientific calls."""
from __future__ import annotations
import ast
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b22_source as source
import english_v6_b22_source_ancestry as lineage
import english_v6_b22_natural_sources as natural
import english_v6_b22_source_observe as observe
import english_v6_b22_source_producer as producer


def function(path, name):
    return next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def identity():
    return dict(run_id='123', run_head='b' * 40, prepared_parent='a' * 40,
                run_branch=lineage.BRANCH, run_attempt=1)


def registration():
    return dict(identity(), source_rows=120, max_calls=120, source_manifest_sha256='c' * 64)


class SourceContracts(unittest.TestCase):
    def test_exact_frozen_panels_and_unchanged_observers(self):
        plan = lineage.read_plan()
        self.assertFalse(plan['source_only_generation_before_model_training_completion_allowed'])
        self.assertEqual(plan['source_gate'], json.loads((lineage.PLANS / 'b21-assessment.json').read_bytes())['source_gate'])
        contract = source.code_contract()
        actual = {k: v for k, v in contract.items() if k not in ('schema', 'observer_sha256', 'source_indices')}
        self.assertEqual(actual, plan['observer_scientific_contract'])
        self.assertEqual(len(source.source_plan()['rows']), 96)
        source.validate_config(lineage.PLANS / 'piper-config.json')

    def test_screen_numerical_admission_body_unchanged(self):
        before = function(HERE / 'english_v6_b21_fixed_mirror_source.py', 'screen_document')
        after = function(HERE / 'english_v6_b22_source.py', 'screen_document')
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))

    def test_natural_decoder_and_rejection_body_unchanged(self):
        old = (HERE / 'english_v6_b21_natural_amendment_natural_sources.py').read_text().replace('b21-primary-natural-', 'b22-primary-natural-')
        tree = ast.parse(old)
        for name in ('extract', 'read_source', 'verify_corpus'):
            before = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            after = function(HERE / 'english_v6_b22_natural_sources.py', name)
            self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))

    def test_metadata_pick_matches_all24_precommitted_ids(self):
        full = json.loads((lineage.PLANS / 'FULL_METADATA_CATALOG.json').read_bytes())
        plan = natural.plan()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w') as archive:
            records = [('LibriSpeech/SPEAKERS.TXT', full['speakers_text'])]
            records.extend((r['path'], r['text']) for r in full['transcripts'])
            for name, content in records:
                raw = content.encode()
                item = tarfile.TarInfo(name)
                item.size = len(raw)
                archive.addfile(item, io.BytesIO(raw))
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            rows, catalog = natural.pick(archive, plan)
        self.assertEqual(rows, plan['roster'])
        self.assertEqual([r['source_id'] for r in rows], plan['selected_utterance_ids'])
        changed = copy.deepcopy(catalog)
        changed['transcripts'][0]['text'] += '\n'
        with self.assertRaises(ValueError):
            natural.rank_metadata(changed, plan)

    def test_noncontiguous_fixed_voice_roster_and_reject_duplicate(self):
        rows = [dict(r, pcm_sha256=hashlib.sha256(r['id'].encode()).hexdigest()) for r in source.source_plan()['rows']]
        reg = dict(registration(), source_rows=96, max_calls=96)
        observe.validate_rows(rows, reg['source_manifest_sha256'], reg)
        altered = copy.deepcopy(rows)
        altered[0] = altered[1]
        with self.assertRaises(ValueError):
            observe.validate_rows(altered, reg['source_manifest_sha256'], reg)
        altered = copy.deepcopy(rows)
        altered[0]['speaker_index'] = 846
        with self.assertRaises(ValueError):
            observe.validate_rows(altered, reg['source_manifest_sha256'], reg)

    def test_first_public_execution_and_no_retry(self):
        ident = identity()
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=lineage.REPO, GITHUB_EVENT_NAME='push',
                   GITHUB_REF=lineage.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=ident['run_head'], GITHUB_RUN_ID='123')
        event = dict(before=ident['prepared_parent'], after=ident['run_head'], ref=lineage.BRANCH,
                     forced=False, deleted=False, repository=dict(full_name=lineage.REPO, private=False))
        with patch.object(lineage, 'read_plan'):
            self.assertEqual(lineage.execution_guard(ident['prepared_parent'], env, event, ident['prepared_parent']), ident)
            for changed in (dict(env, GITHUB_RUN_ATTEMPT='2'), dict(env, GITHUB_REF='refs/heads/main')):
                with self.assertRaises(ValueError):
                    lineage.execution_guard(ident['prepared_parent'], changed, event, ident['prepared_parent'])
            with self.assertRaises(ValueError):
                lineage.execution_guard(ident['prepared_parent'], env, dict(event, forced=True), ident['prepared_parent'])

    def test_durable_marker_identity_and_cross_provider_rejection(self):
        reg = registration()
        expected = lineage.observer_document('qwen', reg)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('OBSERVER_CONSUMPTION.json', json.dumps(expected))
        raw = stream.getvalue()
        item = dict(id=1, name=expected['marker_name'], expired=False, size_in_bytes=len(raw),
                    digest='sha256:' + lineage.digest(raw), workflow_run=dict(id=123, head_sha=identity()['run_head']))
        proof = dict(artifact=item, original_zip_base64=base64.b64encode(raw).decode(), receipt=expected)
        self.assertEqual(lineage.validate_observer_budget(proof, 'qwen', reg), expected)
        with self.assertRaises(ValueError):
            lineage.validate_observer_budget(proof, 'whisper', reg)
        item['workflow_run']['id'] = 124
        with self.assertRaises(ValueError):
            lineage.validate_observer_budget(proof, 'qwen', reg)

    def test_complete_source_inventory_excludes_only_future_b22_bindings(self):
        tracked = subprocess.check_output(['git', '-C', str(HERE.parent), 'ls-files', '-z']).decode().split('\0')
        new = [p.relative_to(HERE.parent).as_posix() for p in HERE.glob('*b22_source*') if p.is_file()]
        new += ['automation/english_v6_b22_natural_sources.py', 'automation/test_english_v6_b22_sources.py', lineage.WORKFLOW]
        future = ['automation/english_v6_b22_primary_inputs.json', 'automation/english_v6_b22_primary_producer.json',
                  'automation/bind_b22_primary.py', 'automation/english_v6_b22_quality_readback.py']
        paths = producer.select_paths(set(tracked + new + future))
        self.assertFalse(set(future) & paths)
        old = json.loads((HERE / 'english_v6_b22_b21_producer_manifest.json').read_bytes())
        self.assertTrue(set(old['files']) <= paths)
        self.assertTrue(all(p.startswith('automation/') or p.startswith('.github/workflows/')
                            or p.startswith('research/english_v6_b22_training_20261124/initial_failure/')
                            or p in producer.EXTRA or p in old['files'] for p in paths))

    def test_inert_workflow_and_captured_execution_root(self):
        template = (HERE.parent / producer.TEMPLATE).read_bytes()
        actual = (HERE.parent / lineage.WORKFLOW).read_bytes()
        if os.environ.get('GITHUB_REF') == lineage.BRANCH and os.environ.get('PREPARED_PARENT'):
            self.assertEqual(template.replace(producer.TOKEN, os.environ['PREPARED_PARENT'].encode()), actual)
        else:
            self.assertEqual(template, actual)
        self.assertEqual(template.count(producer.TOKEN), 2)
        text = template.decode()
        self.assertIn('include-hidden-files: true', text)
        self.assertEqual(text.count('name: english-v6-b22-primary-source-producer-${{ github.run_id }}'), 3)
        for line in text.splitlines():
            if 'source.py' in line and ('python' in line or 'bin/python' in line):
                self.assertIn('$RUNNER_TEMP/b22-source-producer/repository/automation/english_v6_b22_source.py', line)
        self.assertLess(text.index('name: english-v6-b22-primary-source-consumed'), text.index('source.py" generate'))
        self.assertLess(text.index('name: english-v6-b22-primary-observer-${{ matrix.provider }}-consumed'),
                        text.index('Install one pinned native observer runtime'))
        self.assertNotIn('workflow_dispatch', text)

    def test_source_binding_remains_closed_without_actual_success(self):
        with patch.object(lineage, 'INPUTS') as path:
            path.read_bytes.return_value = json.dumps(dict(schema='english-v6-b22-completed-train-source-binding-v1',
                assessment_sha256=lineage.ASSESSMENT_SHA, bound=False, training=None)).encode()
            self.assertFalse(lineage.training_binding(False)['bound'])
            with self.assertRaises(ValueError):
                lineage.training_binding()
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('onnxruntime', sys.modules)
        self.assertNotIn('soundfile', sys.modules)


if __name__ == '__main__':
    unittest.main()
