"""Bounded sealed-replica fixtures: no source/model load or external calls."""
from pathlib import Path
from contextlib import ExitStack, redirect_stdout
import ast
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_v6_b24q1_replication_source_ancestry as lineage
import english_v6_b24q1_replication_source as source
import english_v6_b24q1_replication_source_observe as observe
import english_v6_b24q1_replication_natural_sources as natural
import english_v6_b24q1_replication_source_producer as producer
import english_v6_b24q1_source as original
import english_v6_b24q1_source_observe as original_observe


class SealedSources(unittest.TestCase):
    def test_original_bound_train_authority_keeps_original_globals(self):
        import english_v6_b24_source_ancestry as train
        for name in ('training_binding','training_binding_sha','validate_training_live','verify_training'):
            self.assertIs(getattr(lineage,name),getattr(train,name))
            self.assertIs(getattr(lineage,name).__globals__,train.__dict__)
        bound=lineage.training_binding()
        self.assertIs(bound['bound'],True)
        self.assertEqual(bound['assessment_sha256'],original.lineage.metadata.TRAINING_ASSESSMENT_SHA)
        self.assertNotEqual(bound['assessment_sha256'],lineage.ASSESSMENT_SHA)
        self.assertEqual(bound['training']['model_pt_sha256']['challenger'],lineage.CANDIDATE_PT)

    def sealed_fixture(self):
        value=json.loads(lineage.OPENING.read_bytes())
        value['status']='SEALED_UNTIL_ORIGINAL_PRIMARY_PASS'
        for key in ('primary_quality_lock','quality_readback_sha256','primary_quality_report_sha256',
                    'primary_quality_files','pair_pin','model_pt_sha256'):
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
                       primary_quality_lock={'panel':'primary'},quality_readback_sha256=lineage.digest(lineage.READER.read_bytes()),
                       primary_quality_report_sha256='f'*64,
                       primary_quality_files={name:dict(bytes=1,sha256='0'*64) for name in lineage.capacity.PRIMARY_FILES})
        with patch.object(lineage,'OPENING') as path, \
             patch.object(lineage,'training_binding',return_value={'training':train}), \
             patch.object(lineage.reader,'validate_lock'), \
             patch.object(lineage,'validate_primary_files',return_value=None,create=True), \
             patch.object(lineage,'gh',side_effect=AssertionError('no network')):
            path.read_bytes.return_value=json.dumps(fixture).encode()
            self.assertEqual(lineage.read_opening()['model_pt_sha256']['challenger'],lineage.CANDIDATE_PT)
            # Even if both the proposed opening and proposed TRAIN identity
            # agree on another model, the reviewed actual candidate is fixed.
            models['challenger']='e'*64
            path.read_bytes.return_value=json.dumps(fixture).encode()
            with self.assertRaisesRegex(ValueError,'same successful original B24Q1'):
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
                'read_opening':{'primary_quality_lock':{}},'one_execution':None,'validate_primary':{},'verify_primary':{},
                'verify_training':None,'make_history':{},'authorization_document':{},'require_authorization':{}}.items():
                stack.enter_context(patch.object(lineage,name,return_value=value,create=True))
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
        self.assertTrue(all(r['id'].startswith('b24q1-replication-') for r in rows))
        with self.assertRaises(ValueError):source.source_plan('primary')
        self.assertEqual(natural.plan()['roster'],plan['replication']['natural']['roster'])

    def test_numerical_admission_and_native_observer_bodies_are_reused(self):
        self.assertIs(source.screen_document.__code__,original.screen_document.__code__)
        self.assertIs(observe.observe.__code__,original_observe.observe.__code__)
        self.assertIs(observe.load_observations.__code__,original_observe.load_observations.__code__)
        self.assertIsNot(source.screen_document.__globals__,original.screen_document.__globals__)
        self.assertEqual(source.screen_document.__globals__['SCHEMA'],source.SCHEMA)
        self.assertEqual(original.SCHEMA,'english-v6-b24q1-fresh-combined-source-v1')

    def test_replica_observer_uses_fixed_noncontiguous_voice_roster(self):
        contract=observe.source_check()
        self.assertEqual(contract['source_indices'],lineage.read_plan()['replication']['synthetic']['speaker_indices'])
        self.assertEqual(contract['schema'],'english-v6-b24q1-five-observer-contract-v1')
        self.assertEqual(contract['panel'],'replication')
        self.assertIs(contract['orchestration_changed'],True)

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
        self.assertIn('b24q1-replica-source-producer/repository/automation/english_v6_b24q1_replication_source.py',text)
        self.assertNotIn('workflow_dispatch',text)
        self.assertIn('  pull_request:',text)
        self.assertIn("if: github.event_name == 'push' &&",text)
        self.assertIn('    needs: contracts',text)
        self.assertIn('research/en-v6-b24q1-replication-sources-once-20261125',text)
        self.assertIn("for mode in '' -O -OO",text)
        self.assertEqual(text.count('include-hidden-files: true'),text.count('retention-days: 90'))
        capacity=text.index('    - name: Preserve capacity evidence before generator entry')
        consume=text.index('    - name: Consume fixed TTS96')
        generate=text.index('    - name: Generate only the new frozen96')
        self.assertLess(consume,capacity);self.assertLess(capacity,generate)
        capacity_step=text[capacity:text.index('    - uses:',capacity)]
        self.assertIn('      if: always()',capacity_step)
        self.assertIn('path: ${{ runner.temp }}/prior-source/CAPACITY.json\n',capacity_step)
        self.assertNotIn('/CONSUMPTION.json',capacity_step)

    def tracked_fixture(self):
        root=HERE.parent
        tracked=set(producer.EXTRA)|set(lineage.metadata.validate()['hashes'])
        for prefix in ('automation','.github/workflows','research/english_v6_b22_training_20261124/initial_failure',
                       'research/english_v6_b22_source_20261124/initial_failure'):
            tracked.update(p.relative_to(root).as_posix() for p in (root/prefix).rglob('*') if p.is_file())
        tracked.update(json.loads((HERE/'english_v6_b23_source_training_producer.json').read_bytes())['files'])
        tracked.update(producer.dynamic.required_paths())
        binding=json.loads((HERE/'english_v6_b24_source_inputs.json').read_bytes())
        if binding['bound']:
            tracked.update(json.loads((HERE/'english_v6_b24_source_training_producer.json').read_bytes())['files'])
        return tracked

    def test_complete_approved_resources_and_future_replica_exclusion(self):
        tracked=self.tracked_fixture();selected=producer.select_paths(tracked)
        self.assertTrue(set(lineage.metadata.validate()['hashes'])<=selected)
        self.assertTrue(producer.EXTRA<=selected)
        for name in ('english_v6_b24q1_primary_inputs.json','english_v6_b24q1_primary_producer.json',
                     'english_v6_b24q1_r1_git_transport.py','english_v6_b24q1_r1_git_transport.json',
                     'english_v6_b24q1_quarantine.py','english_v6_b24q1_capacity.py',
                     'english_v6_b24q1_evidence_scope.py','test_english_v6_b24q1_evidence_scope.py'):
            self.assertIn('automation/'+name,selected)
        future=['automation/english_v6_b24q1_replication_eval_future.py',
            'automation/english_v6_b24q1_replication_inputs_future.json',
            'automation/english_v6_b24q1_engineering_future.py']
        self.assertEqual(producer.select_paths(tracked|set(future)),selected)
        for name in ('research/english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json',
            'automation/english_v6_b24q1_quarantine.py','automation/english_v6_b24q1_orchestration_contract.json'):
            with self.subTest(name=name),self.assertRaises(ValueError):producer.select_paths(tracked-{name})

    def test_actual_captured_root_import_and_metadata_preflight_is_stdlib_only(self):
        selected=producer.select_paths(self.tracked_fixture())
        with tempfile.TemporaryDirectory() as temporary:
            repository=Path(temporary)/'repository';repository.mkdir()
            for name in selected:
                target=repository/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(HERE.parent/name,target)
            receipt=producer.captured_preflight(repository)
            self.assertIs(receipt['code_reads_only'],True)
            self.assertIs(receipt['no_model'],True);self.assertIs(receipt['no_network'],True)
            self.assertEqual(len(receipt['original_dynamic_helpers']['helper_pins']),19)
            self.assertEqual(receipt['expected_original_B24R1_STOP_location'],'PRIMARY_QUALITY/inputs/source.zip')

    def test_workflow_keeps_full_primary_proof_once_in_source_not_transport_upload(self):
        text=(HERE.parent/producer.TEMPLATE).read_text()
        start=text.index('        name: english-v6-b24q1-replication-transport-')
        end=text.index('    - name: Consume fixed TTS96',start)
        transport=text[start:end]
        self.assertIn('prior-source/PRIMARY_QUALITY/api/',transport)
        self.assertIn('prior-source/HISTORY.json',transport);self.assertIn('prior-source/OPENING.json',transport)
        self.assertNotIn('prior-source/PRIMARY_QUALITY/inputs/',transport)
        self.assertNotIn('prior-source/PRIMARY_QUALITY/originals/',transport)
        self.assertNotIn('prior-source/PRIMARY_QUALITY/\n',transport)
        tree=ast.parse(Path(lineage.__file__).read_text())
        retain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='retain_proof')
        code=ast.unparse(retain)
        self.assertEqual(code.count("shutil.copytree(directory / 'PRIMARY_QUALITY', output / 'PRIMARY_QUALITY')"),1)
        self.assertNotIn('ORIGINAL_B24R1_STOP',code)
        self.assertIn('CAPACITY.json',code)
        self.assertIn('    timeout-minutes: 180',text);self.assertIn('    timeout-minutes: 120',text)

    def test_full_slot_quarantine_policy_is_identical_across_primary_and_replica(self):
        policy=source.source_policy();primary=original.source_policy()
        for key in ('proposal_sha256','approval_sha256','orchestration_contract_sha256','quarantine_helper_sha256',
                    'orchestration_implementation_sha256','asr_adapter_sha256','parser_contract_sha256'):
            self.assertEqual(policy[key],primary[key])
        self.assertNotEqual(policy['observer_sha256'],primary['observer_sha256'])
        self.assertEqual(source.BATCH,'english-v6-b24q1-120-original-quarantine-replication-screen-v1')
        self.assertIs(observe.load_observations.__code__,original_observe.load_observations.__code__)
        body=ast.unparse(ast.parse(Path(original_observe.__file__).read_text()))
        self.assertIn('COMPLETE_NATIVE_SLOTS',body)
        self.assertIn('5 * count <= 600',body)

    def test_source_producer_keeps_primary_and_excludes_future_replica_bindings(self):
        self.assertFalse('automation/english_v6_b24q1_primary_inputs.json'.startswith(producer.FUTURE_PREFIXES))
        for name in ('english_v6_b24q1_replication_inputs.json','english_v6_b24q1_replication_producer.json','english_v6_b24q1_replication_eval.py'):
            self.assertTrue(('automation/'+name).startswith(producer.FUTURE_PREFIXES))
        self.assertEqual(producer.TOKEN,b'__B24Q1_REVIEWED_REPLICATION_SOURCE_PARENT__')

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
