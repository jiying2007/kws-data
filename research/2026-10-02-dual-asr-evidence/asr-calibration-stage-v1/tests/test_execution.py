"""No runtime imports, weights, or real model inference in these tests."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asr_stage.execution import execute_primary, json_bytes, require_offline_flags
from asr_stage.architecture import canonical_sha
from asr_stage.adapters import QWEN


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.decoder = (ROOT/'pcm/decoder-inputs.json').read_bytes()
        self.inputs = {r['opaque_id']:r for r in json.loads(self.decoder)['clips']}
        sha = hashlib.sha256(self.decoder).hexdigest()
        contract = {'model_id':QWEN,'decoder_manifest_sha256':sha,'run_id':'fixture-run'}
        self.plan = {'schema':'asr-primary-run-plan-v1','model_id':QWEN,
            'model_root':'/fixture-model','pcm_root':'/fixture-pcm',
            'asset_lock':{},'asset_lock_sha256':'1'*64,'source_lock':[],
            'source_lock_sha256':'2'*64,'contract':contract,
            'contract_sha256':canonical_sha(contract),'decoder_manifest_sha256':sha,
            'clip_seconds':1}
        self.seen = []
    def binder(self, root, expected):
        self.assertEqual(root,'/fixture-pcm')
        self.assertEqual(expected.basename,expected.opaque_id+'.wav')
        row = self.inputs[expected.opaque_id]
        return types.SimpleNamespace(opaque_id=expected.opaque_id, descriptor=row['descriptor'],
            descriptor_sha256=row['binding_sha256'])
    def loader(self,*args):
        return types.SimpleNamespace(receipt={'fixture_only':True})
    def infer(self,runtime,bound,sha):
        self.seen.append(bound.opaque_id)
        return {'raw_text':'fixture raw','completeness':'complete','quality_flags':[]}, {'fixture_only':True}
    def run_fixture(self, **kwargs):
        return execute_primary(self.plan,self.root/'out',self.decoder,
            loader=kwargs.get('loader',self.loader),infer=kwargs.get('infer',self.infer),
            binder=self.binder,check_environment=False)
    def test_full_primary_runs_each_clip_once(self):
        result = self.run_fixture()
        self.assertEqual(result['status'],'complete')
        self.assertEqual(self.seen,list(self.inputs))
        self.assertEqual(result['attempted'],16)
        self.assertEqual(result['success'],16)
        self.assertTrue(result['first_clip_is_primary_canary'])
    def test_retained_hashes_bind_exact_bytes(self):
        self.run_fixture()
        output = self.root/'out'
        outcomes = json.loads((output/'outcomes.json').read_bytes())
        for row in outcomes:
            raw = (output/(row['opaque_id']+'.receipt.json')).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['execution_receipt_sha256'])
            receipt = json.loads(raw)
            evidence_raw = (output/(row['opaque_id']+'.decoder.json')).read_bytes()
            self.assertEqual(hashlib.sha256(evidence_raw).hexdigest(),receipt['decoder_evidence_sha256'])
            evidence = json.loads(evidence_raw)
            self.assertEqual(evidence['model'],receipt['model'])
            self.assertEqual(evidence['outcome']['raw_text'],row['raw_text'])
            self.assertEqual(receipt['execution_contract_sha256'],hashlib.sha256((output/'contract.json').read_bytes()).hexdigest())
    def test_no_retry_after_first_failure(self):
        def failed(*args):
            self.seen.append(args[1].opaque_id)
            raise RuntimeError('fixture-only failure')
        result = self.run_fixture(infer=failed)
        self.assertEqual(self.seen,['clip-0001'])
        self.assertEqual(result['not_run'],15)
        rows = json.loads((self.root/'out/outcomes.json').read_bytes())
        self.assertEqual(rows[0]['status'],'error')
        self.assertTrue(all(r['status']=='not_run' for r in rows[1:]))
    def test_load_failure_has_no_inference(self):
        def failed(*args): raise RuntimeError('fixture load failure')
        result = self.run_fixture(loader=failed)
        self.assertEqual(result['status'],'load_failed')
        self.assertEqual(self.seen,[])
        rows = json.loads((self.root/'out/outcomes.json').read_bytes())
        self.assertTrue(all(r['status']=='not_run' for r in rows))
    def test_existing_run_output_not_overwritten(self):
        (self.root/'out').mkdir()
        with self.assertRaises(FileExistsError): self.run_fixture()
    def test_plan_hash_mismatch(self):
        self.plan['contract_sha256']='0'*64
        with self.assertRaises(ValueError): self.run_fixture()
        self.assertFalse((self.root/'out').exists())
    def test_decoder_mismatch(self):
        self.plan['decoder_manifest_sha256']='0'*64
        with self.assertRaises(ValueError): self.run_fixture()
    def test_offline_flags_required(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(RuntimeError): require_offline_flags()
    def test_truthful_offline_disclosure(self):
        env = {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','CUDA_VISIBLE_DEVICES':''}
        with patch.dict('os.environ',env,clear=True):
            self.assertIs(require_offline_flags()['kernel_network_isolated'],False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
