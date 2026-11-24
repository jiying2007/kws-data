"""Bounded sealed-replica fixtures: no source/model load or external calls."""
from pathlib import Path
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_v6_b24_replication_source_ancestry as lineage
import english_v6_b24_replication_source as source
import english_v6_b24_replication_source_observe as observe
import english_v6_b24_replication_natural_sources as natural
import english_v6_b24_replication_source_producer as producer
import english_v6_b24_source as original
import english_v6_b24_source_observe as original_observe


class SealedSources(unittest.TestCase):
    def sealed_fixture(self):
        value=json.loads(lineage.OPENING.read_bytes())
        value['status']='SEALED_UNTIL_ORIGINAL_PRIMARY_PASS'
        for key in ('primary_quality_lock','quality_readback_sha256','pair_pin','model_pt_sha256'):
            value[key]=None
        return value

    def test_opening_null_and_blocks_before_any_external_operation(self):
        # The actual config may already be bound when this pre-execution fixture
        # runs. Test a sealed config without mutating or weakening that binding.
        fixture=self.sealed_fixture()
        with patch.object(lineage,'OPENING') as opening, \
             patch.object(lineage,'gh',side_effect=AssertionError('no network')), \
             patch.object(lineage.reader,'fetch_quality_original',side_effect=AssertionError('no fetch')):
            opening.read_bytes.return_value=json.dumps(fixture).encode()
            self.assertEqual(lineage.read_opening(False)['status'],'SEALED_UNTIL_ORIGINAL_PRIMARY_PASS')
            with tempfile.TemporaryDirectory() as tmp:
                dest=Path(tmp)/'unopened'
                with self.assertRaisesRegex(ValueError,'sealed'):
                    lineage.authorize(dest,'a'*40)
                self.assertFalse(dest.exists())

    def test_partial_lock_cannot_open_the_panel(self):
        value=self.sealed_fixture()
        value['primary_quality_lock']={}
        with patch.object(lineage,'OPENING') as path:
            path.read_bytes.return_value=json.dumps(value).encode()
            with self.assertRaises(ValueError):lineage.read_opening(False)

    def test_bound_source_opening_cannot_substitute_a_different_train_candidate(self):
        fixture=self.sealed_fixture()
        models=dict(B0='a'*64,B9='b'*64,control=original.lineage.metadata.CONTROL_PT,challenger=lineage.CANDIDATE_PT)
        pair=dict(id=10,run_id=20,run_head='c'*40,bytes=100,sha256='d'*64)
        train=dict(run_id=20,run_head='c'*40,model_pt_sha256=models,
                   artifacts={'models':dict(id=10,bytes=100,sha256='d'*64)})
        fixture.update(status='BOUND_ORIGINAL_PRIMARY_PASS',model_pt_sha256=models,pair_pin=pair,
                       primary_quality_lock={'panel':'primary'},quality_readback_sha256=lineage.digest(lineage.READER.read_bytes()))
        with patch.object(lineage,'OPENING') as path, \
             patch.object(lineage,'training_binding',return_value={'training':train}), \
             patch.object(lineage.reader,'validate_lock'), \
             patch.object(lineage,'gh',side_effect=AssertionError('no network')):
            path.read_bytes.return_value=json.dumps(fixture).encode()
            self.assertEqual(lineage.read_opening()['model_pt_sha256']['challenger'],lineage.CANDIDATE_PT)
            # Even if both the proposed opening and proposed TRAIN identity
            # agree on another model, the reviewed actual candidate is fixed.
            models['challenger']='e'*64
            path.read_bytes.return_value=json.dumps(fixture).encode()
            with self.assertRaisesRegex(ValueError,'same successful original B24'):
                lineage.read_opening()

    def test_replica_authorize_collects_complete_new_train_git_identity(self):
        # Temporary provenance fixture only; all execution and quality gates are mocked.
        head='b'*40
        tree=dict(sha='c'*40,truncated=False,tree=[])
        commit=dict(sha=head,commit=dict(tree=dict(sha=tree['sha'])))
        train=dict(run_head=head,run_id=123,artifacts={})
        responses={'commits/'+head:commit,'actions/runs/123':{'id':123},
                   'actions/runs/123/artifacts?per_page=100':{'artifacts':[]},
                   'git/trees/'+tree['sha']+'?recursive=1':tree}
        seen=[]
        def validate(value):
            self.assertEqual(value['commit'],commit)
            self.assertEqual(value['tree'],tree)
            seen.append(value)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            for name,value in {'execution_guard':{},'training_binding':{'training':train},
                'read_opening':{'primary_quality_lock':{}},'one_execution':None,'validate_primary':{},
                'verify_training':None,'make_history':{},'authorization_document':{},'require_authorization':{}}.items():
                stack.enter_context(patch.object(lineage,name,return_value=value))
            stack.enter_context(patch.object(lineage,'quality_reader'))
            stack.enter_context(patch.object(lineage,'validate_training_live',side_effect=validate))
            request=stack.enter_context(patch.object(lineage,'gh',side_effect=lambda endpoint:json.dumps(responses[endpoint]).encode()))
            lineage.authorize(Path(tmp)/'fixture','a'*40)
            self.assertEqual(len(seen),1)
            self.assertEqual(request.call_count,4)
            self.assertEqual(json.loads((Path(tmp)/'fixture/TRAIN_LIVE.json').read_bytes()),seen[0])

    def test_same_frozen72_disjoint_people_and_only_replica_requests(self):
        plan=lineage.read_plan()
        people=[]
        for panel in ('primary','replication'):
            group={v.removeprefix('p') for v in plan[panel]['synthetic']['speaker_id_map'].values()}
            group.update(plan[panel]['natural']['speakers'])
            self.assertEqual(len(group),36);people.append(group)
        self.assertFalse(people[0]&people[1])
        rows=source.source_plan()['rows']
        self.assertEqual(len(rows),96)
        self.assertTrue(all(r['id'].startswith('b24-replication-') for r in rows))
        with self.assertRaises(ValueError):source.source_plan('primary')
        self.assertEqual(natural.plan()['roster'],plan['replication']['natural']['roster'])

    def test_numerical_admission_and_native_observer_bodies_are_reused(self):
        self.assertIs(source.screen_document.__code__,original.screen_document.__code__)
        self.assertIs(observe.observe.__code__,original_observe.observe.__code__)
        self.assertIs(observe.load_observations.__code__,original_observe.load_observations.__code__)
        self.assertIsNot(source.screen_document.__globals__,original.screen_document.__globals__)
        self.assertEqual(source.screen_document.__globals__['SCHEMA'],source.SCHEMA)
        self.assertEqual(original.SCHEMA,'english-v6-b24-fresh-combined-source-v1')

    def test_replica_observer_uses_fixed_noncontiguous_voice_roster(self):
        contract=observe.source_check()
        self.assertEqual(contract['source_indices'],lineage.read_plan()['replication']['synthetic']['speaker_indices'])
        self.assertEqual(contract['schema'],'english-v6-b24-replication-five-observer-contract-v1')

    def test_real_check_reuses_the_registered_complete_piper_config(self):
        with patch.object(source,'validate_config',wraps=source.validate_config) as validate, redirect_stdout(io.StringIO()):
            source.check()
        validate.assert_called_once_with(original.PIPER_CONFIG)
        self.assertEqual(source.PIPER_CONFIG,original.PIPER_CONFIG)
        self.assertTrue(source.PIPER_CONFIG.is_file())

    def test_template_remains_inert_except_two_reviewed_parent_substitutions(self):
        template=(HERE.parent/producer.TEMPLATE).read_bytes()
        actual=(HERE.parent/lineage.WORKFLOW).read_bytes()
        self.assertEqual(template.count(producer.TOKEN),2)
        if actual!=template:
            import re
            parents=re.findall(rb"(?:PREPARED_PARENT: ')([0-9a-f]{40})'",actual)
            self.assertEqual(len(parents),1)
            self.assertEqual(template.replace(producer.TOKEN,parents[0]),actual)
        text=template.decode()
        self.assertIn('include-hidden-files: true',text)
        self.assertIn('b24-replica-source-producer/repository/automation/english_v6_b24_replication_source.py',text)
        self.assertNotIn('workflow_dispatch',text)
        capacity=text.index('    - name: Preserve capacity evidence before generator entry')
        consume=text.index('    - name: Consume TTS96')
        generate=text.index('    - name: Generate only the new frozen96')
        self.assertLess(consume,capacity);self.assertLess(capacity,generate)
        capacity_step=text[capacity:text.index('    - uses:',capacity)]
        self.assertIn('      if: always()',capacity_step)
        self.assertIn('path: ${{ runner.temp }}/prior-source/CAPACITY.json\n',capacity_step)
        self.assertNotIn('/CONSUMPTION.json',capacity_step)

    def test_source_producer_keeps_primary_and_excludes_future_replica_bindings(self):
        self.assertFalse('automation/english_v6_b24_primary_inputs.json'.startswith(producer.FUTURE_PREFIXES))
        for name in ('english_v6_b24_replication_inputs.json','english_v6_b24_replication_producer.json','english_v6_b24_replication_eval.py'):
            self.assertTrue(('automation/'+name).startswith(producer.FUTURE_PREFIXES))
        self.assertEqual(producer.TOKEN,b'__B24_REVIEWED_REPLICATION_SOURCE_PARENT__')

    def test_source_producer_inherits_complete_text_runtime_helper_authority(self):
        pins=producer.dynamic.required_paths()
        self.assertEqual(len(pins),19)
        self.assertTrue({producer.dynamic.CAMPAIGN,producer.dynamic.RUNNER} <= set(pins))
        root=HERE.parent
        original=json.loads((HERE/'english_v6_b23_source_training_producer.json').read_bytes())
        older=json.loads((HERE/'english_v6_b22_source_training_producer.json').read_bytes())
        self.assertEqual(len(original['files']),306)
        self.assertEqual(len(older['files']),264)
        self.assertTrue(set(older['files']) <= set(original['files']))
        # The B22 workflow had its one reviewed activation before the actual
        # B23 producer. Both original manifests retain their own exact bytes.
        changed={name for name,pin in older['files'].items()
                 if pin['sha256'] != original['files'][name]['sha256']}
        self.assertEqual(changed,{'.github/workflows/english-v6-b22-union-retention-train-continuation-once.yml'})
        tracked=set(original['files']) | set(pins) | producer.EXTRA
        for prefix in ('automation','.github/workflows','research/english_v6_b22_training_20261124/initial_failure'):
            tracked.update(path.relative_to(root).as_posix() for path in (root/prefix).rglob('*') if path.is_file())
        binding=json.loads((HERE/'english_v6_b24_source_inputs.json').read_bytes())
        if binding['bound']:
            tracked.update(json.loads((HERE/'english_v6_b24_source_training_producer.json').read_bytes())['files'])
        self.assertTrue(set(pins) <= producer.select_paths(tracked))
        with self.assertRaisesRegex(ValueError,'remain tracked'):
            producer.select_paths(tracked-{producer.dynamic.CAMPAIGN})


if __name__=='__main__':unittest.main()
