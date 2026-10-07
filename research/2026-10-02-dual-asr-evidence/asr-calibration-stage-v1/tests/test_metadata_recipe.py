"""Synthetic primitive/tensor fixtures, never a model forward."""
from collections import OrderedDict,namedtuple
import hashlib,json
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from asr_stage.architecture import sensevoice_schema,canonical_sha
from vendor.checkpoint_guard.checkpoint_contract import (SENSEVOICE_METADATA_RECIPE,
    SENSEVOICE_CHECKPOINT_METADATA_SHA256,SENSEVOICE_ARCH_METADATA_SHA256,
    sensevoice_architecture_metadata,prepare_metadata,CheckpointContractError)
from vendor.checkpoint_guard.exact_apply import exact_apply,require_verified_receipt

Result=namedtuple('Result','missing_keys unexpected_keys')
SCHEMA=sensevoice_schema()
def state(metadata,content):
    result=OrderedDict((k,(v,content)) for k,v in SCHEMA.items())
    result._metadata=OrderedDict((k,dict(v)) for k,v in metadata.items())
    return result
def describe(value,content):
    row=dict(value[0])
    if content:row['sha256']=hashlib.sha256(value[1]).hexdigest()
    return row
class Model:
    def __init__(self):
        self.metadata=sensevoice_architecture_metadata();self.values=state(self.metadata,b'old');self.mode=None;self.loads=0
    def state_dict(self):
        value=OrderedDict(self.values);value._metadata=OrderedDict((k,dict(v)) for k,v in self.metadata.items());return value
    def load_state_dict(self,cp,strict=True):
        self.loads+=1;assert strict is True
        assert set(cp._metadata)-set(self.metadata)=={'frontend','encoder.rwkv_encoders'}
        self.received_metadata=cp._metadata
        self.values=OrderedDict(cp)
        if self.mode=='mutate_checkpoint_metadata':cp._metadata['']['version']=2
        if self.mode=='mutate_arch_metadata':self.metadata['']['version']=2
        return Result([],[])
class Tests(unittest.TestCase):
    def setUp(self):
        self.model=Model();self.meta=dict(sensevoice_architecture_metadata(),frontend={'version':1},**{'encoder.rwkv_encoders':{'version':1}})
        self.cp=state(self.meta,b'new')
    def apply(self):return exact_apply(self.model,self.cp,describe,expected_schema=SCHEMA,expected_schema_sha256=canonical_sha(SCHEMA),metadata_recipe=SENSEVOICE_METADATA_RECIPE)
    def test_independent_topology_matches_retained_actual_observations(self):
        observations=ROOT.parent/'checkpoint-metadata-review-20261002'
        self.assertEqual(sensevoice_architecture_metadata(),json.loads((observations/'architecture-metadata.json').read_bytes())['metadata'])
        self.assertEqual(self.meta,json.loads((observations/'official-metadata.json').read_bytes())['metadata'])
        self.assertEqual(canonical_sha(self.meta),SENSEVOICE_CHECKPOINT_METADATA_SHA256)
        self.assertEqual(canonical_sha(sensevoice_architecture_metadata()),SENSEVOICE_ARCH_METADATA_SHA256)
    def test_preserved_exact_success_and_readback(self):
        receipt=self.apply();self.assertEqual(self.model.loads,1)
        self.assertEqual(self.model.received_metadata,self.meta)
        self.assertEqual(receipt['contract']['metadata']['checkpoint'],self.meta)
        require_verified_receipt(self.model,describe)
    def test_default_recipe_remains_rejected(self):
        with self.assertRaises(CheckpointContractError):exact_apply(self.model,self.cp,describe,expected_schema=SCHEMA,expected_schema_sha256=canonical_sha(SCHEMA))
    def test_checkpoint_mutation_cannot_mutate_expected_copy(self):
        self.model.mode='mutate_checkpoint_metadata'
        with self.assertRaises(CheckpointContractError):self.apply()
    def test_architecture_mutation_rejected(self):
        self.model.mode='mutate_arch_metadata'
        with self.assertRaises(CheckpointContractError):self.apply()
    def test_changes_rejected_before_load(self):
        modifications=[lambda cp:setattr(cp,'extra',1),lambda cp:cp._metadata.update({'other':{'version':1}}),lambda cp:cp._metadata.pop('frontend'),lambda cp:cp._metadata[''].update(version=True),lambda cp:cp._metadata[''].update(version=2),lambda cp:cp._metadata[''].update(assign_to_params_buffers=True),lambda cp:setattr(cp,'_metadata',{}),lambda cp:cp.pop(next(iter(cp)))]
        for change in modifications:
            with self.subTest(change=change):
                self.setUp();change(self.cp)
                with self.assertRaises(CheckpointContractError):self.apply()
                self.assertEqual(self.model.loads,0)
    def test_foreign_schema_rejected(self):
        with self.assertRaises(CheckpointContractError):prepare_metadata(self.cp,self.model.state_dict(),'0'*64,SENSEVOICE_METADATA_RECIPE)
    def test_detached_nested_metadata(self):
        frozen,contract=prepare_metadata(self.cp,self.model.state_dict(),canonical_sha(SCHEMA),SENSEVOICE_METADATA_RECIPE)
        frozen['']['version']=2
        self.assertEqual(contract['checkpoint']['']['version'],1)
        self.assertEqual(self.cp._metadata['']['version'],1)
    def test_post_load_model_metadata_mutation_rejected(self):
        self.apply();self.model.metadata['']['version']=2
        with self.assertRaises(CheckpointContractError):require_verified_receipt(self.model,describe)
if __name__=='__main__':unittest.main(verbosity=2)
