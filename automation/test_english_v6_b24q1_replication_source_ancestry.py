"""Offline sealed Q1 opening, exact original TRAIN and independent capacity contracts."""
from pathlib import Path
from contextlib import ExitStack
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b24q1_replication_source_ancestry as lineage
import english_v6_b24q1_source as original


def proof_fixture(root):
    """Invented bytes exercise binding and whole-byte reuse, never quality."""
    proof, code = root/'PRIMARY_QUALITY', root/'captured'
    code.mkdir()
    for name in lineage.capacity.PRIMARY_FILES:
        path=proof/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(('fixture-'+name).encode())
    def pin(name,index):
        raw=(proof/name).read_bytes()
        return dict(id=index,run_id=123,run_head='b'*40,bytes=len(raw),sha256=lineage.digest(raw))
    roles=('B0','B9','pair','source','screen','native_qwen','native_whisper','native_parakeet','native_xlsr','native_lv60','train_producer','source_producer')
    pins={role:pin('inputs/'+role+'.zip',index+10) for index,role in enumerate(roles)}
    spec=dict(schema='english-v6-b24q1-fresh-primary-four-frozen-models-v1',scoring_allowed=True,
        inputs={key:pins[key] for key in ('B0','B9','pair','source','screen')},
        native_inputs={key.removeprefix('native_'):value for key,value in pins.items() if key.startswith('native_')},
        upstream_producers={key.removesuffix('_producer'):pins[key] for key in ('train_producer','source_producer')})
    spec_raw=lineage.reader.canonical(spec)
    (code/'english_v6_b24q1_primary_inputs.json').write_bytes(spec_raw)
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w') as archive:
        archive.writestr('primary_producer/automation/english_v6_b24q1_primary_inputs.json',spec_raw)
    (proof/'originals/producer.zip').write_bytes(output.getvalue())
    lock=dict(schema='english-v6-b24q1-original-quality-lock-v1',panel='primary',
        **{role:pin('originals/'+role+'.zip',index+1) for index,role in enumerate(('result','producer','consumed'))},
        evaluator_sha256=lineage.reader.PRIMARY_CODE_SHA,inputs_sha256=lineage.digest(spec_raw),
        producer_manifest_sha256='c'*64,original_result_sha256='d'*64)
    report=dict(panel='primary',panel_gate='PASS',pair_pin=pins['pair'],model_pt_sha256={'fixture':'unchanged'})
    opening=dict(primary_quality_lock=lock,quality_readback_sha256=lineage.digest(lineage.READER.read_bytes()),
        primary_quality_files=lineage.capacity._primary(proof)['files'],
        primary_quality_report_sha256=lineage.digest(lineage.reader.canonical(report)))
    return proof,code,lock,report,opening


class OriginalPrimaryScope(unittest.TestCase):
    def test_eighteen_pins_equal_the_original_lock_and_captured_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof,code,lock,report,opening=proof_fixture(Path(tmp))
            with patch.object(lineage,'HERE',code):
                self.assertEqual(len(lineage.validate_primary_files(opening)),18)
                changes=(lambda value:value['primary_quality_files'].pop('api/run.json'),
                    lambda value:value.update(primary_quality_report_sha256=None),
                    lambda value:value['primary_quality_files']['originals/result.zip'].update(sha256='0'*64),
                    lambda value:value['primary_quality_files']['inputs/native_qwen.zip'].update(sha256='0'*64),
                    lambda value:value['primary_quality_files']['inputs/source.zip'].update(bytes=768_000_001),
                    lambda value:value['primary_quality_files']['inputs/pair.zip'].update(bytes=128_000_001),
                    lambda value:value['primary_quality_lock'].update(inputs_sha256='0'*64))
                for mutate in changes:
                    changed=copy.deepcopy(opening);mutate(changed)
                    with self.assertRaises(ValueError):lineage.validate_primary_files(changed)
                raw=(code/'english_v6_b24q1_primary_inputs.json').read_bytes()
                (code/'english_v6_b24q1_primary_inputs.json').write_bytes(raw+b' ')
                with self.assertRaisesRegex(ValueError,'original PRIMARY producer SPEC'):
                    lineage.validate_primary_files(opening)

    def test_full_original_reader_once_per_scope_but_semantics_every_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof,code,lock,report,opening=proof_fixture(Path(tmp))
            with patch.object(lineage,'read_opening',return_value=opening), \
                 patch.object(lineage,'quality_reader',return_value=lineage.reader), \
                 patch.object(lineage.reader,'verify_quality_original',return_value=report) as full, \
                 patch.object(lineage,'validate_primary',side_effect=lambda value,permit:value) as semantics:
                with lineage.evidence_scope.scope():
                    self.assertEqual(lineage.verify_primary(proof),report)
                    self.assertEqual(lineage.verify_primary(proof),report)
                self.assertEqual(full.call_count,1)
                self.assertEqual(semantics.call_count,2)
                with lineage.evidence_scope.scope():lineage.verify_primary(proof)
                self.assertEqual(full.call_count,2)

    def test_same_size_original_mutation_is_fatal_before_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof,code,lock,report,opening=proof_fixture(Path(tmp))
            with patch.object(lineage,'read_opening',return_value=opening), \
                 patch.object(lineage,'quality_reader',return_value=lineage.reader), \
                 patch.object(lineage.reader,'verify_quality_original',return_value=report) as full, \
                 patch.object(lineage,'validate_primary',side_effect=lambda value,permit:value):
                with lineage.evidence_scope.scope():
                    lineage.verify_primary(proof)
                    path=proof/'inputs/native_qwen.zip';before=path.stat();raw=path.read_bytes()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:])
                    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'whole evidence file SHA-256'):
                        lineage.verify_primary(proof)
                self.assertEqual(full.call_count,1)

    def test_original_full_report_must_equal_the_frozen_canonical_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof,code,lock,report,opening=proof_fixture(Path(tmp))
            with patch.object(lineage,'read_opening',return_value=opening), \
                 patch.object(lineage,'quality_reader',return_value=lineage.reader), \
                 patch.object(lineage.reader,'verify_quality_original',return_value=dict(report,panel_gate='NO_PROMOTION')), \
                 patch.object(lineage,'validate_primary') as semantics:
                with self.assertRaisesRegex(ValueError,'canonical report SHA-256'):
                    lineage.verify_primary(proof)
                semantics.assert_not_called()

    def test_binder_freezes_only_full_original_pass_with_exact_eighteen_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proof,code,lock,report,opening=proof_fixture(root)
            target=root/'proposed-opening.json'
            with patch.object(lineage,'HERE',code), \
                 patch.object(lineage,'read_opening',side_effect=lambda *args:{'status':'SEALED_UNTIL_ORIGINAL_PRIMARY_PASS'}), \
                 patch.object(lineage.reader,'verify_quality_original',return_value=report) as full, \
                 patch.object(lineage,'validate_primary',side_effect=lambda value,permit:value):
                bound=lineage.bind_opening(proof,lock,target)
                self.assertEqual(bound['primary_quality_files'],opening['primary_quality_files'])
                self.assertEqual(bound['primary_quality_report_sha256'],opening['primary_quality_report_sha256'])
                self.assertEqual(target.read_bytes(),lineage.reader.canonical(bound))
                full.assert_called_once_with(proof,lock)
                with self.assertRaises(ValueError):lineage.bind_opening(proof,lock,target)

    def test_binder_failed_full_read_or_changed_original_produces_no_opening(self):
        for changed in (False,True):
            with self.subTest(changed_during_read=changed),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);proof,code,lock,report,opening=proof_fixture(root)
                target=root/'proposed-opening.json'
                def verify(*args):
                    if not changed:raise ValueError('original quality is NO_PROMOTION')
                    path=proof/'api/run.json';path.write_bytes(path.read_bytes()+b' ')
                    return report
                with patch.object(lineage,'HERE',code), \
                     patch.object(lineage,'read_opening',return_value={'status':'SEALED_UNTIL_ORIGINAL_PRIMARY_PASS'}), \
                     patch.object(lineage.reader,'verify_quality_original',side_effect=verify):
                    with self.assertRaises(ValueError):lineage.bind_opening(proof,lock,target)
                    self.assertFalse(target.exists())

    def test_outer_preflights_and_quality_worker_share_only_the_current_audit(self):
        import english_v6_b24q1_primary_eval as primary
        import english_v6_b24q1_replication_eval as replica
        import bind_b24q1_primary as binder
        import english_v6_b24q1_replication_bind as replica_binder
        for module,public,core in ((primary,'preflight','_preflight'),(replica,'preflight','_preflight'),
                                  (lineage.reader,'verify_worker','_verify_worker'),(binder,'bind','_bind'),
                                  (replica_binder,'bind','_bind')):
            with self.subTest(module=module.__name__):
                scopes=[]
                def inspect(*args):
                    active=lineage.evidence_scope._ACTIVE.get();self.assertIsInstance(active,dict)
                    with lineage.evidence_scope.scope():self.assertIs(lineage.evidence_scope._ACTIVE.get(),active)
                    scopes.append(active)
                    return 'saved-byte-only'
                with patch.object(module,core,side_effect=inspect):
                    arguments=(None,None,None) if public=='bind' else (None,None)
                    self.assertEqual(getattr(module,public)(*arguments),'saved-byte-only')
                    self.assertIsNone(lineage.evidence_scope._ACTIVE.get())
                    getattr(module,public)(*arguments)
                    self.assertIsNot(scopes[0],scopes[1])


class SealedSourceAncestry(unittest.TestCase):
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
        for key in ('primary_quality_lock','quality_readback_sha256','pair_pin','model_pt_sha256','primary_quality_report_sha256','primary_quality_files'):
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
             patch.object(lineage,'validate_primary_files'), \
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
                stack.enter_context(patch.object(lineage,name,return_value=value))
            stack.enter_context(patch.object(lineage,'quality_reader'))
            stack.enter_context(patch.object(lineage,'validate_training_live',side_effect=validate))
            request=stack.enter_context(patch.object(lineage,'gh',side_effect=lambda endpoint:json.dumps(responses[endpoint]).encode()))
            lineage.authorize(Path(tmp)/'fixture','a'*40)
            self.assertEqual(len(seen),1)
            self.assertEqual(request.call_count,4)
            self.assertEqual(json.loads((Path(tmp)/'fixture/TRAIN_LIVE.json').read_bytes()),seen[0])


    def test_replica_generation_has_own_transport_and_capacity_before_entry(self):
        import english_v6_b24q1_replication_transport as transport
        import english_v6_b24q1_replication_natural_sources as natural
        events=[]
        identity={"fixture": "original execution"}
        consumed={"source_execution": identity}
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            directory=Path(temp)
            (directory/"CONSUMPTION.json").write_text(json.dumps(consumed))
            stack.enter_context(patch.object(lineage,"execution_guard",return_value=identity))
            stack.enter_context(patch.object(lineage,"consumption_document",return_value=consumed))
            stack.enter_context(patch.object(transport,"verify_transport",side_effect=lambda *a:events.append("transport")))
            stack.enter_context(patch.object(natural,"plan",return_value={"fixture":"roster"}))
            check=stack.enter_context(patch.object(lineage.capacity,"replica_generation_capacity",side_effect=lambda *a:events.append("capacity")))
            stack.enter_context(patch.object(lineage.capacity,"primary_generation_capacity",side_effect=AssertionError("wrong panel")))
            stack.enter_context(patch.object(lineage,"one_execution",side_effect=lambda *a:events.append("marker") or {}))
            stack.enter_context(patch.object(lineage,"uploaded_marker",return_value={"fixture":"upload"}))
            self.assertEqual(lineage.require_generation_permit(directory,Path("corpus"),"a"*40),consumed)
            self.assertEqual(events,["transport","capacity","marker"])
            check.assert_called_once_with(directory,Path("corpus"),{"fixture":"roster"})
            with self.assertRaises(ValueError):lineage.require_generation_permit(directory,Path("corpus"),"a"*40)

    def test_original_primary_failure_never_becomes_replica_permission(self):
        with patch.object(lineage,"read_opening",return_value={"model_pt_sha256":{},"pair_pin":{},"primary_quality_lock":{}}):
            for panel_gate in ("NO_PROMOTION","STOP","STARTED"):
                with self.assertRaises(ValueError):lineage.validate_primary({"panel":"primary","panel_gate":panel_gate})


if __name__ == '__main__':
    unittest.main()
