"""Sealed replica scoring contracts; fixtures never create a quality permit."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_v6_b24_replication_eval as replica
import english_v6_b24_primary_eval as primary
import english_v6_b24_replication_bind as binder
import english_v6_b24_quality_readback as reader


class ReplicaContracts(unittest.TestCase):
    def test_bound_protocol_uses_b23_initialization_and_only_the_actual_b24_candidate(self):
        # These invented metadata pins test protocol structure only. The full
        # original-array reader is never replaced by this fixture at execution.
        doc=json.loads(replica.SPEC.read_bytes())
        doc.update(scoring_allowed=True,replication_opened=True,producer_manifest_sha256='a'*64,
                   quality_readback_sha256='b'*64,primary_quality_lock={'fixture_only':True})
        for index,(name,pin) in enumerate(replica.archive_pins(doc).items(),1):
            if name in primary.BASELINES:continue
            run=100 if name in ('pair','train_producer') else 200
            value=dict(id=1000+index,run_id=run,run_head=('%040x'%run),bytes=100,sha256=('%064x'%index))
            if name in replica.INPUT_KEYS:doc['inputs'][name]=value
            elif name.startswith('native_'):doc['native_inputs'][name.removeprefix('native_')]=value
            else:doc['upstream_producers'][name.removesuffix('_producer')]=value
        doc['model_pt_sha256']['challenger']=replica.CANDIDATE_PT
        permit=dict(primary_quality_lock=doc['primary_quality_lock'],quality_readback_sha256=doc['quality_readback_sha256'],
                    pair_pin=doc['inputs']['pair'],model_pt_sha256=doc['model_pt_sha256'])
        with patch.object(replica,'SPEC') as spec, patch.object(replica.opening,'read_opening',return_value=permit), \
             patch.object(replica,'gh',side_effect=AssertionError('no network')):
            spec.read_bytes.return_value=json.dumps(doc).encode()
            self.assertEqual(replica.protocol()['parent_pt_sha256'],primary.CONTROL_PT)
            self.assertNotEqual(primary.INITIAL_PT,primary.B0_PT)
            self.assertEqual(doc['model_pt_sha256']['B0'],primary.B0_PT)
            for field,bad in (('parent_pt_sha256',primary.B0_PT),('challenger','c'*64),('control',primary.B0_PT)):
                mutated=copy.deepcopy(doc)
                if field=='parent_pt_sha256':mutated[field]=bad
                else:mutated['model_pt_sha256'][field]=bad
                spec.read_bytes.return_value=json.dumps(mutated).encode()
                with self.assertRaises(ValueError):replica.protocol()

    def test_inert_rejects_before_network_paths_or_model_import(self):
        fixture=json.loads(replica.SPEC.read_bytes())
        fixture['scoring_allowed']=False
        with tempfile.TemporaryDirectory() as tmp:
            inputs,out=Path(tmp)/'inputs',Path(tmp)/'out'
            with patch.object(replica,'SPEC') as spec, \
                 patch.object(replica,'gh',side_effect=AssertionError('no network')), \
                 patch.object(replica,'primary_quality',side_effect=AssertionError('no evidence read before binding')):
                spec.read_bytes.return_value=json.dumps(fixture).encode()
                for call,args in ((replica.fetch,(inputs,'a'*40)),(replica.preflight,(inputs,'a'*40)),(replica.score,(inputs,out,'a'*40))):
                    with self.assertRaisesRegex(ValueError,'INERT'):call(*args)
            self.assertFalse(inputs.exists());self.assertFalse(out.exists())
        self.assertNotIn('torch',sys.modules)

    def test_immutable_models_and_scientific_helpers_are_identical(self):
        self.assertEqual(replica.STEMS,primary.STEMS)
        self.assertEqual(replica.GAINS,primary.GAINS)
        self.assertEqual(replica.GATE,primary.GATE)
        self.assertEqual(replica.MAX_FORWARDS,1440)
        for name in ('snapshots','validate_pair_receipt','ctc_events','exact_admitted_join','source_counts','validate_scored_join'):
            self.assertIs(getattr(replica,name).__code__,getattr(primary,name).__code__)
        self.assertIs(replica.Ledger,primary.Ledger)
        self.assertEqual(primary.SPEC.name,'english_v6_b24_primary_inputs.json')
        spec=json.loads(replica.SPEC.read_bytes())
        for role,pin in primary.BASELINES.items():
            self.assertEqual(spec['inputs'][role],pin,'immutable original '+role+' artifact including Git head')

    def test_replica_bind_and_quality_reader_routes_match(self):
        self.assertEqual(reader.EVALUATOR_STEMS['replication'],'english_v6_b24_replication')
        self.assertEqual(replica.SCHEMA,reader.SCHEMAS['replication'])
        self.assertEqual(binder.DRIVER_SHA,hashlib.sha256(Path(replica.__file__).read_bytes()).hexdigest())
        self.assertEqual(binder.artifact_name('source',123),'english-v6-b24-replication-source-123')
        self.assertEqual(binder.artifact_name('pair',123),'english-v6-b24-b23-warm-start-models-123')

    def test_large_container_permission_is_explicitly_source_only(self):
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w') as archive:
            archive.writestr('MANIFEST.json',json.dumps(dict(schema=replica.SOURCE_SCHEMA,panel='replication')))
        data=raw.getvalue()
        pin=dict(id=1,run_id=1,run_head='a'*40,bytes=len(data),sha256=replica.sha(data))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.zip';path.write_bytes(data)
            with patch.object(replica.original,'original_zip',wraps=replica.original.original_zip) as core, \
                 patch.object(replica.capacity,'check_archive',wraps=replica.capacity.check_archive) as composite:
                with replica.original_zip(path,pin,role='source'):pass
                composite.assert_called_once_with(data,'replication_source')
                core.assert_not_called()
            for role in ('B0','B9','pair','screen','native_qwen','train_producer','source_producer',None):
                with patch.object(replica.original,'original_zip',wraps=replica.original.original_zip) as core, \
                     patch.object(replica.capacity,'check_archive',side_effect=AssertionError('core role cannot use composite bound')):
                    with replica.original_zip(path,pin,role=role):pass
                    core.assert_called_once_with(path,pin)
            for role in ('producer','replication_producer','arbitrary'):
                with self.assertRaises(ValueError):replica.original_zip(path,pin,role=role)

    def test_bound_lock_cannot_bypass_full_original_pass_reader(self):
        permit=dict(primary_quality_lock={'fixture':'only'})
        with patch.object(replica.opening,'read_opening',return_value=permit), \
             patch.object(replica.opening,'quality_reader') as factory:
            factory.return_value.verify_quality_original.side_effect=ValueError('original NO_PROMOTION remains closed')
            with self.assertRaisesRegex(ValueError,'NO_PROMOTION'):
                replica.primary_quality(Path('/not-opened'),dict(primary_quality_lock=permit['primary_quality_lock']))
            factory.return_value.verify_quality_original.assert_called_once()

    def test_workflow_keeps_primary_proof_and_has_no_unconditional_execution(self):
        template=replica.WORKFLOW_TEMPLATE.read_text()
        self.assertEqual(template.count(replica.PARENT_TOKEN),2)
        self.assertIn('b24-inputs/primary_quality/',template)
        self.assertIn('b24-inputs/replication_producer/',template)
        self.assertIn('english-v6-b24-historical-control-replication-logits-',template)
        self.assertIn('include-hidden-files: true',template)
        self.assertNotIn('workflow_dispatch',template)

    def test_entire_original_b22_replica_scoring_retains_identical_numerics(self):
        def function(path):
            return next(node for node in ast.parse(path.read_bytes()).body
                        if isinstance(node,ast.FunctionDef) and node.name=='score')
        old=function(HERE/'english_v6_b22_replication_eval.py')
        new=function(Path(replica.__file__))
        removed=[]
        source_envelope=[]
        for node in ast.walk(new):
            if isinstance(node,ast.Call):
                for item in list(node.keywords):
                    if item.arg in ('B22_failed_primary_preserved','B23_failed_primary_preserved'):
                        self.assertIs(item.value.value,True)
                        node.keywords.remove(item);removed.append(item)
                    if isinstance(node.func,ast.Name) and node.func.id=='original_zip' and item.arg=='role':
                        self.assertEqual(item.value.value,'source')
                        node.keywords.remove(item);source_envelope.append(item)
        self.assertEqual(len(removed),2)
        self.assertEqual(len(source_envelope),1)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))


if __name__=='__main__':unittest.main()
