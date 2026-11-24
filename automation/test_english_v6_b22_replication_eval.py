"""Sealed replica scoring contracts; fixtures never create a quality permit."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_v6_b22_replication_eval as replica
import english_v6_b22_primary_eval as primary
import english_v6_b22_replication_bind as binder
import english_v6_b22_quality_readback as reader


class ReplicaContracts(unittest.TestCase):
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
        self.assertEqual(primary.SPEC.name,'english_v6_b22_primary_inputs.json')

    def test_replica_bind_and_quality_reader_routes_match(self):
        self.assertEqual(reader.EVALUATOR_STEMS['replication'],'english_v6_b22_replication')
        self.assertEqual(replica.SCHEMA,reader.SCHEMAS['replication'])
        self.assertEqual(binder.DRIVER_SHA,hashlib.sha256(Path(replica.__file__).read_bytes()).hexdigest())
        self.assertEqual(binder.artifact_name('source',123),'english-v6-b22-replication-source-123')
        self.assertEqual(binder.artifact_name('pair',123),'english-v6-b22-union-retention-models-123')

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
        self.assertIn('b22-inputs/primary_quality/',template)
        self.assertIn('b22-inputs/replication_producer/',template)
        self.assertIn('english-v6-b22-historical-control-replication-logits-',template)
        self.assertIn('include-hidden-files: true',template)
        self.assertNotIn('workflow_dispatch',template)


if __name__=='__main__':unittest.main()
