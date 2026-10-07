"""Stdlib-only preparation tests. Real tensor/model imports are forbidden."""
import ast
import builtins
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
ORIGINAL_IMPORT = builtins.__import__
def guarded_import(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','numpy','transformers','qwen_asr','funasr','yaml','kaldi_native_fbank'}:
        raise AssertionError('Real model/runtime import forbidden in preparation fixtures')
    return ORIGINAL_IMPORT(name,*args,**kwargs)
builtins.__import__=guarded_import

from asr_stage.architecture import (qwen06_schema,sensevoice_schema,schema_summary,
    canonical_sha,validate_qwen_geometry,validate_sense_geometry)
from asr_stage.assets import MODELS,validate_asset_lock,verify_assets,read_small_locked
from asr_stage.decoding import (qwen_transcript,sense_transcript,CapturingTokenizer,
    qwen_generation_options,token_ids,bounded_text)
from asr_stage.runtime_support import assert_effective_qwen_attention
from asr_stage.adapters import (require_execution_declaration,validate_decoder_manifest,
    PreparedRuntime,_begin_clip,QWEN,SENSE)
from asr_stage.results import scientific_manifest,output_envelope
from vendor import qwen_loading_gate_v2 as qgate


class StageTests(unittest.TestCase):
    def test_qwen_schema_geometry_accounting(self):
        state=qwen06_schema();summary=schema_summary(state)
        self.assertEqual(summary['keys'],612)
        self.assertEqual(summary['elements'],938008576)
        self.assertEqual(summary['fp32_bytes'],3752034304)
        self.assertEqual(state['thinker.audio_tower.conv_out.weight']['shape'],[896,7680])
        self.assertEqual(state['thinker.model.layers.27.self_attn.q_proj.weight']['shape'],[2048,1024])
        self.assertEqual(state['thinker.model.layers.27.self_attn.k_proj.weight']['shape'],[1024,1024])
        self.assertNotIn('thinker.audio_tower.positional_embedding.positional_embedding',state)
        self.assertNotIn('thinker.model.rotary_emb.inv_freq',state)

    def test_sense_schema_geometry_accounting(self):
        state=sensevoice_schema();summary=schema_summary(state)
        self.assertEqual(summary['keys'],917)
        self.assertEqual(summary['elements'],233999167)
        self.assertEqual(summary['fp32_bytes'],935996668)
        self.assertEqual(state['encoder.encoders0.0.norm1.weight']['shape'],[560])
        self.assertEqual(state['encoder.encoders.48.norm1.weight']['shape'],[512])
        self.assertEqual(state['encoder.tp_encoders.19.self_attn.fsmn_block.weight']['shape'],[512,1,11])
        self.assertEqual(state['ctc.ctc_lo.weight']['shape'],[25055,512])
        self.assertEqual(state['embed.weight']['shape'],[16,560])

    def test_schema_candidate_files_are_exact_derivations(self):
        for name,fn in [('qwen06',qwen06_schema),('sensevoice',sensevoice_schema)]:
            saved=json.loads((ROOT/'locks'/f'{name}-expected-state.candidate.json').read_text())
            self.assertEqual(saved,fn())

    def test_no_dictionary_state_shared_between_calls(self):
        state=qwen06_schema();state.clear();self.assertEqual(len(qwen06_schema()),612)

    def test_generation_options_avoid_duplicate_keyword(self):
        options=qwen_generation_options()
        self.assertNotIn('return_dict_in_generate',options)
        self.assertEqual(options,{'max_new_tokens':256,'do_sample':False,'eos_token_id':[151645,151643]})

    def test_qwen_keeps_repetition_punctuation_and_spelling(self):
        row,meta=qwen_transcript('language Chinese<asr_text>小窝小窝，小屋！',[12,151645])
        self.assertEqual(row['raw_text'],'小窝小窝，小屋！')
        self.assertEqual(row['quality_flags'],[])
        self.assertTrue(meta['eos_terminated'])

    def test_qwen_limit_without_eos_is_incomplete(self):
        row,_=qwen_transcript('language Chinese<asr_text>小窝',[4]*256)
        self.assertEqual(row['completeness'],'incomplete');self.assertIn('incomplete',row['quality_flags'])

    def test_qwen_unrecognized_prefix_is_not_repaired(self):
        row,_=qwen_transcript('Chinese: 小窝',[151643])
        self.assertEqual(row['raw_text'],'Chinese: 小窝');self.assertIn('decoding_warning',row['quality_flags'])

    def test_qwen_body_metadata_causes_warning(self):
        row,_=qwen_transcript('language Chinese<asr_text>你好<|tag|>',[151645])
        self.assertIn('decoding_warning',row['quality_flags'])

    def test_qwen_nonspeech(self):
        row,_=qwen_transcript('language None<asr_text>',[151643])
        self.assertIn('non_speech',row['quality_flags'])

    def test_sense_strips_only_four_known_leading_tags(self):
        raw='<|zh|><|NEUTRAL|><|Speech|><|woitn|>小窝小窝，小屋'
        row,tags=sense_transcript(raw)
        self.assertEqual(row['raw_text'],'小窝小窝，小屋')
        self.assertEqual(tags,['zh','NEUTRAL','Speech','woitn'])
        self.assertEqual(row['quality_flags'],[])

    def test_sense_unknown_tags_keep_raw_text(self):
        raw='<|zh|><|newtag|><|Speech|><|woitn|>小窝'
        row,_=sense_transcript(raw);self.assertEqual(row['raw_text'],raw)
        self.assertEqual(row['completeness'],'unknown')

    def test_sense_itn_is_flagged(self):
        row,_=sense_transcript('<|zh|><|NEUTRAL|><|Speech|><|withitn|>一')
        self.assertIn('decoding_warning',row['quality_flags'])

    def test_sense_body_tag_not_globally_removed(self):
        row,_=sense_transcript('<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好<|zh|>')
        self.assertEqual(row['raw_text'],'你好<|zh|>');self.assertIn('decoding_warning',row['quality_flags'])

    def test_sense_unknown_emotion_or_event_abstains(self):
        row,_=sense_transcript('<|zh|><|EMO_UNKNOWN|><|Event_UNK|><|woitn|>一')
        self.assertIn('ambiguous',row['quality_flags'])

    def test_capture_records_exact_ids(self):
        class Tokenizer:
            def decode(self,ids):self.received=list(ids);return '<|zh|>重复重复'
        base=Tokenizer();cap=CapturingTokenizer(base);raw=cap.decode([3,4,4])
        self.assertEqual(cap.evidence(raw)['token_ids'],[3,4,4]);self.assertEqual(base.received,[3,4,4])

    def test_capture_requires_one_decode(self):
        cap=CapturingTokenizer(types.SimpleNamespace(decode=lambda ids:'raw'))
        with self.assertRaises(ValueError):cap.evidence('raw')
        cap.decode([1])
        with self.assertRaises(ValueError):cap.decode([1])

    def test_capture_output_mismatch(self):
        cap=CapturingTokenizer(types.SimpleNamespace(decode=lambda ids:'raw'));cap.decode([1])
        with self.assertRaises(ValueError):cap.evidence('repaired')

    def test_capture_rejects_mutating_tokenizer(self):
        def mutate(ids):ids.append(9);return 'raw'
        with self.assertRaises(ValueError):CapturingTokenizer(types.SimpleNamespace(decode=mutate)).decode([1])

    def test_capture_rejects_decode_options(self):
        with self.assertRaises(ValueError):CapturingTokenizer(None).decode([1],repair=True)

    def test_sense_empty_ctc_output_retained_as_unknown(self):
        capture=CapturingTokenizer(types.SimpleNamespace(decode=lambda ids:''))
        raw=capture.decode([]);self.assertEqual(capture.evidence(raw)['token_ids'],[])
        row,_=sense_transcript(raw)
        self.assertEqual(row['raw_text'],'');self.assertEqual(row['completeness'],'unknown')
        self.assertIn('decoding_warning',row['quality_flags'])

    def test_models_require_fresh_process(self):
        from asr_stage import adapters
        with patch.object(adapters,'_MODEL_PROCESS_CLAIMED',False):
            adapters.claim_fresh_model_process()
            with self.assertRaises(RuntimeError):adapters.claim_fresh_model_process()

    def test_attention_effective_nested_recipe(self):
        def model(implementation='eager',short=False):
            entries=[]
            for kind,count in [('Qwen3ASRAudioAttention',18),('Qwen3ASRTextAttention',28-int(short))]:
                cls=type(kind,(),{})
                for i in range(count):
                    instance=cls();instance.config=types.SimpleNamespace(_attn_implementation=implementation)
                    entries.append((f'{kind}.{i}',instance))
            return types.SimpleNamespace(named_modules=lambda:entries)
        self.assertEqual(len(assert_effective_qwen_attention(model())),46)
        with self.assertRaises(RuntimeError):assert_effective_qwen_attention(model('sdpa'))
        with self.assertRaises(RuntimeError):assert_effective_qwen_attention(model(short=True))

    def declaration(self):
        return {'schema':'asr-calibration-model-execution-v1','armed':True,'model_id':QWEN,
            'run_id':'fixture-one','asset_lock_sha256':'1'*64,'source_lock_sha256':'2'*64,
            'expected_schema_sha256':'3'*64,'decoder_manifest_sha256':'4'*64,'model_dtype':'float32',
            'batch_size':1,'network':'offline_configured_not_kernel_isolated','scope':'single_primary_decode_each_of_16_synthetic_clips'}

    def check_declaration(self,dec):
        return require_execution_declaration(dec,canonical_sha(dec),QWEN,'1'*64,'2'*64,'3'*64)

    def test_execution_declaration_valid_fixture(self):self.check_declaration(self.declaration())

    def test_real_decoder_manifest_has_only_label_free_bindings(self):
        raw=(ROOT/'pcm/decoder-inputs.json').read_bytes()
        index=validate_decoder_manifest(raw,hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(index),16)
        self.assertNotIn('human',raw.decode());self.assertNotIn('Serena',raw.decode());self.assertNotIn('你好',raw.decode())

    def test_manifest_byte_drift_rejected(self):
        raw=(ROOT/'pcm/decoder-inputs.json').read_bytes()
        with self.assertRaises(RuntimeError):validate_decoder_manifest(raw+b' ',hashlib.sha256(raw).hexdigest())

    def test_manifest_descriptor_drift_rejected_even_if_outer_rehashed(self):
        obj=json.loads((ROOT/'pcm/decoder-inputs.json').read_bytes());obj['clips'][0]['descriptor']['wav']['sha256']='0'*64
        raw=json.dumps(obj).encode()
        with self.assertRaises(RuntimeError):validate_decoder_manifest(raw,hashlib.sha256(raw).hexdigest())

    def test_manifest_duplicate_json_key(self):
        raw=b'{"schema":"x","schema":"x"}'
        with self.assertRaises(ValueError):validate_decoder_manifest(raw,hashlib.sha256(raw).hexdigest())

    def test_manifest_duplicate_clip(self):
        obj=json.loads((ROOT/'pcm/decoder-inputs.json').read_bytes());obj['clips'][1]=obj['clips'][0]
        raw=json.dumps(obj).encode()
        with self.assertRaises(RuntimeError):validate_decoder_manifest(raw,hashlib.sha256(raw).hexdigest())

    def test_manifest_missing_pcm_fields_rejected_even_when_rehashed(self):
        obj=json.loads((ROOT/'pcm/decoder-inputs.json').read_bytes())
        row=obj['clips'][0]
        row['descriptor']={'opaque_id':row['opaque_id'],'wav':{'sha256':row['descriptor']['wav']['sha256']}}
        row['binding_sha256']=canonical_sha(row['descriptor'])
        raw=json.dumps(obj).encode()
        with self.assertRaises(ValueError):validate_decoder_manifest(raw,hashlib.sha256(raw).hexdigest())

    def test_manifest_pcm_relationship_rejected_even_when_all_hashes_match(self):
        obj=json.loads((ROOT/'pcm/decoder-inputs.json').read_bytes())
        row=obj['clips'][0];pcm=row['descriptor']['float32_pcm']
        pcm['frame_count']+=1;pcm['bytes']+=4;pcm['shape'][0]+=1
        row['descriptor']['float32_pcm_descriptor_sha256']=canonical_sha(pcm)
        row['binding_sha256']=canonical_sha(row['descriptor'])
        raw=json.dumps(obj).encode()
        with self.assertRaises(ValueError):validate_decoder_manifest(raw,hashlib.sha256(raw).hexdigest())

    def test_begin_clip_binds_descriptor_and_blocks_second_attempt(self):
        raw=(ROOT/'pcm/decoder-inputs.json').read_bytes();sha=hashlib.sha256(raw).hexdigest();idx=validate_decoder_manifest(raw,sha)
        row=idx['clip-0001'];descriptor=row['descriptor'];encoded=json.dumps(descriptor,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
        bound=types.SimpleNamespace(opaque_id='clip-0001',descriptor=descriptor,descriptor_json=encoded,descriptor_sha256=row['binding_sha256'])
        rt=PreparedRuntime(QWEN,None,{},dict(decoder_manifest_sha256=sha),set(),idx)
        with patch('asr_stage.adapters.make_pcm_array',return_value='array'):
            self.assertEqual(_begin_clip(rt,bound,sha),'array')
            with self.assertRaises(RuntimeError):_begin_clip(rt,bound,sha)

    def test_failed_array_construction_consumes_primary_attempt(self):
        raw=(ROOT/'pcm/decoder-inputs.json').read_bytes();sha=hashlib.sha256(raw).hexdigest();idx=validate_decoder_manifest(raw,sha);row=idx['clip-0001']
        encoded=json.dumps(row['descriptor'],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
        bound=types.SimpleNamespace(opaque_id='clip-0001',descriptor=row['descriptor'],descriptor_json=encoded,descriptor_sha256=row['binding_sha256'])
        rt=PreparedRuntime(QWEN,None,{},dict(decoder_manifest_sha256=sha),set(),idx)
        with patch('asr_stage.adapters.make_pcm_array',side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):_begin_clip(rt,bound,sha)
        self.assertEqual(rt.consumed_ids,{'clip-0001'})

    def test_actual_human_labels_keep_unknown(self):
        obj=json.loads((ROOT/'human-labels.json').read_text())
        self.assertEqual(obj['human_label_counts'],{'positive':9,'negative':17,'unknown':6})
        for target,expected in [(0,(5,9,2)),(1,(4,8,4))]:
            self.assertEqual(tuple(sum(r['human_target_presence'][target]==v for r in obj['records']) for v in ['positive','negative','unknown']),expected)

    def test_all_source_files_parse_without_import(self):
        for path in list((ROOT/'asr_stage').glob('*.py'))+list((ROOT/'vendor').rglob('*.py')):
            ast.parse(path.read_text(),filename=str(path))
        self.assertFalse({'torch','numpy','funasr','qwen_asr','transformers'} & set(sys.modules))


for key,value in [('armed',False),('armed',1),('model_dtype','bfloat16'),('batch_size',True),('network','host'),('scope','all_audio'),('model_id',SENSE),('asset_lock_sha256','0'*64)]:
    def test(self,key=key,value=value):
        dec=self.declaration();dec[key]=value
        with self.assertRaises(RuntimeError):self.check_declaration(dec)
    setattr(StageTests,'test_reject_execution_'+key+'_'+str(value),test)

for name,value in [('empty',[]),('bool',[True]),('negative',[-1]),('tuple',(1,)),('limit',[1]*4097)]:
    def test(self,value=value):
        with self.assertRaises(ValueError):token_ids(value)
    setattr(StageTests,'test_reject_token_ids_'+name,test)


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        rev,names=MODELS[QWEN];files=[]
        for name in sorted(names):
            data=(name+' fixture').encode();(self.root/name).write_bytes(data)
            files.append({'filename':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                'source_url':f'https://huggingface.co/{QWEN}/resolve/{rev}/{name}'})
        self.lock={'schema':'asr-exact-model-assets-v1','model_id':QWEN,'revision':rev,'files':files}
    def tearDown(self):self.temp.cleanup()
    def verify(self):return verify_assets(self.root,self.lock,canonical_sha(self.lock),QWEN)
    def test_valid_all_exact_assets(self):self.assertEqual(len(self.verify()['verified_files']),8)
    def test_body_corruption(self):
        (self.root/'config.json').write_bytes(b'x')
        with self.assertRaises(ValueError):self.verify()
    def test_extra_file(self):
        (self.root/'extra').write_text('x')
        with self.assertRaises(ValueError):self.verify()
    def test_symlink_file(self):
        path=self.root/'config.json';path.unlink();path.symlink_to(self.root/'vocab.json')
        with self.assertRaises((ValueError,OSError)):self.verify()
    def test_hardlink_file(self):
        outside=self.root.parent/(self.root.name+'-alias');os.link(self.root/'config.json',outside)
        try:
            with self.assertRaises(ValueError):self.verify()
        finally:outside.unlink()
    def test_small_locked_body_hash(self):
        files=validate_asset_lock(self.lock,canonical_sha(self.lock),QWEN)
        self.assertEqual(read_small_locked(self.root,'config.json',files),'config.json fixture')
        (self.root/'config.json').write_bytes(b'z'*files['config.json']['bytes'])
        with self.assertRaises(ValueError):read_small_locked(self.root,'config.json',files)

for name,change in [
    ('missing',lambda lock:lock['files'].pop()),
    ('duplicate',lambda lock:lock['files'].__setitem__(0,dict(lock['files'][1]))),
    ('revision',lambda lock:lock.update(revision='main')),
    ('url',lambda lock:lock['files'][0].update(source_url='https://example.com/weight')),
    ('hash',lambda lock:lock['files'][0].update(sha256='bad')),
    ('bool_size',lambda lock:lock['files'][0].update(bytes=True)),
]:
    def test(self,change=change):
        change(self.lock)
        with self.assertRaises(ValueError):self.verify()
    setattr(AssetTests,'test_invalid_lock_'+name,test)


class QwenIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'tiny.safetensors'
        payload=struct.pack('<f',2.0)
        header=json.dumps({'weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}},separators=(',',':')).encode()
        header+=b' '*((-len(header))%8);raw=struct.pack('<Q',len(header))+header+payload;self.path.write_bytes(raw)
        schema={'weight':{'shape':[1],'dtype':'torch.float32'}}
        self.evidence=qgate.inspect_safetensors(self.path,schema,qgate.digest(schema),hashlib.sha256(raw).hexdigest())
        self.info={key:[] for key in qgate.REQUIRED_LOADING_FIELDS}
        self.tensor={'shape':[1],'dtype':'torch.float32','sha256':hashlib.sha256(payload).hexdigest()}
    def tearDown(self):self.temp.cleanup()
    def model(self,hashable=True):
        class Fake:
            def named_modules(self):return [('',self)]
            def state_dict(self):return {'weight':'tensor'}
            def __eq__(self,other):return True
            def __hash__(self):return 7
        if not hashable:Fake.__hash__=None
        return Fake()
    def test_equal_model_cannot_borrow_qwen_receipt(self):
        a,b=self.model(),self.model()
        receipt=qgate.verify_loaded_model(a,self.info,self.evidence,lambda _:dict(self.tensor))
        b._kws_qwen_weights_verified=True;b._kws_qwen_load_receipt=receipt
        with self.assertRaises(qgate.QwenLoadGateError):qgate.require_verified_receipt(b)
        qgate.require_verified_receipt(a)
    def test_unhashable_qwen_model_works(self):
        a=self.model(False);qgate.verify_loaded_model(a,self.info,self.evidence,lambda _:dict(self.tensor))
        qgate.require_verified_receipt(a);qgate.clear_verification(a)
        with self.assertRaises(qgate.QwenLoadGateError):qgate.require_verified_receipt(a)


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.human=json.loads((ROOT/'human-labels.json').read_text())
        models=[{'model_id':mid,'revision':MODELS[mid][0],'run_id':'fixture-'+str(i)} for i,mid in enumerate((QWEN,SENSE))]
        self.manifest=scientific_manifest(self.human,models,'fixture-probe')
        self.raw=json.dumps(self.manifest,ensure_ascii=False,indent=2).encode()+b'\n'
        self.sha=hashlib.sha256(self.raw).hexdigest()
        self.outcomes=[{'opaque_id':r['opaque_id'],'wav_sha256':r['wav_sha256'],'status':'success',
                       'raw_text':'合成测试','completeness':'complete','quality_flags':[],
                       'execution_receipt_sha256':'1'*64} for r in self.human['records']]
        self.decoder_raw=(ROOT/'pcm/decoder-inputs.json').read_bytes()
        self.decoder_sha=hashlib.sha256(self.decoder_raw).hexdigest()
        self.decoder_index=validate_decoder_manifest(self.decoder_raw,self.decoder_sha)
        self.receipts={}
        self.refresh_receipts()
    def refresh_receipts(self):
        self.receipts={}
        for row in self.outcomes:
            if row['status']=='not_run':continue
            receipt={k:row[k] for k in ('opaque_id','wav_sha256','status','raw_text','completeness','quality_flags')}
            receipt.update(schema='asr-clip-execution-receipt-v1',model=dict(self.manifest['models'][0]),
                input_binding_sha256=self.decoder_index[row['opaque_id']]['binding_sha256'],
                decoder_manifest_sha256=self.decoder_sha,attempt_started=True,model_forward_completed=row['status']=='success',
                decoder_evidence_sha256='8'*64,execution_contract_sha256='9'*64)
            raw=json.dumps(receipt,ensure_ascii=False).encode();sha=hashlib.sha256(raw).hexdigest()
            row['execution_receipt_sha256']=sha;self.receipts[sha]=raw
    def bundle(self,slot=0):
        return output_envelope(self.outcomes,self.human,self.manifest,self.sha,'2'*64,slot,manifest_bytes=self.raw,
            execution_receipts=self.receipts,decoder_manifest_bytes=self.decoder_raw,decoder_manifest_sha256=self.decoder_sha)
    def envelope(self):
        return self.bundle()[0]
    def replace_manifest(self):
        # Deliberately freeze malformed fixture bytes to test semantics beyond hashing.
        self.raw=json.dumps(self.manifest,ensure_ascii=True).encode()
        self.sha=hashlib.sha256(self.raw).hexdigest()
    def replace_receipt_bytes(self,raw,index=0):
        old=self.outcomes[index]['execution_receipt_sha256'];self.receipts.pop(old)
        sha=hashlib.sha256(raw).hexdigest()
        self.outcomes[index]['execution_receipt_sha256']=sha;self.receipts[sha]=raw
    def alter_receipt(self,change,index=0):
        receipt=json.loads(self.receipts[self.outcomes[index]['execution_receipt_sha256']])
        change(receipt)
        self.replace_receipt_bytes(json.dumps(receipt,ensure_ascii=True).encode(),index)
    def test_complete_envelope(self):
        out=self.envelope();self.assertEqual(len(out['records']),16)
        self.assertEqual(out['manifest_sha256'],self.sha)
        self.assertEqual(out['model'],self.manifest['models'][0])
    def test_failures_and_not_run_remain_explicit(self):
        self.outcomes[0].update(status='timeout',raw_text=None,completeness='unknown')
        self.outcomes[1].update(status='not_run',raw_text=None,completeness='unknown',execution_receipt_sha256=None)
        self.refresh_receipts()
        out=self.envelope();self.assertEqual(out['records'][0]['status'],'timeout')
        self.assertIsNone(out['records'][1]['raw_text'])
    def test_missing_outcome_rejected(self):
        self.outcomes.pop()
        with self.assertRaises(ValueError):self.envelope()
    def test_duplicate_outcome_rejected(self):
        self.outcomes[1]=dict(self.outcomes[0])
        with self.assertRaises(ValueError):self.envelope()
    def test_wrong_wav_rejected(self):
        self.outcomes[0]['wav_sha256']='0'*64
        with self.assertRaises(ValueError):self.envelope()
    def test_missing_execution_receipt_rejected(self):
        self.outcomes[0]['execution_receipt_sha256']=None
        with self.assertRaises(ValueError):self.envelope()
    def test_not_run_cannot_contain_prediction(self):
        self.outcomes[0].update(status='not_run',execution_receipt_sha256=None)
        with self.assertRaises(ValueError):self.envelope()
    def test_all_not_run_cannot_claim_execution(self):
        for row in self.outcomes:row.update(status='not_run',raw_text=None,completeness='unknown',execution_receipt_sha256=None)
        self.refresh_receipts()
        with self.assertRaisesRegex(ValueError,'All-not-run'):self.envelope()
    def test_unknown_label_not_promoted(self):
        before=json.loads(json.dumps(self.human));self.envelope()
        self.assertEqual(self.human,before)
        self.assertEqual(sum(r['human_target_presence'].count('unknown') for r in self.manifest['records']),6)
    def test_human_known_unknown_count_drift_rejected(self):
        for row in self.human['records']:
            if 'unknown' in row['human_target_presence']:
                row['human_target_presence'][row['human_target_presence'].index('unknown')]='negative';break
        with self.assertRaises(ValueError):scientific_manifest(self.human,self.manifest['models'],'fixture-probe')
    def test_manifest_bytes_must_match(self):
        self.manifest['probe_set_id']='changed'
        with self.assertRaises(ValueError):self.envelope()
    def test_same_outcomes_cannot_be_assigned_to_other_model(self):
        with self.assertRaises(ValueError):self.bundle(slot=1)
    def test_raw_receipt_digest_must_match(self):
        sha=self.outcomes[0]['execution_receipt_sha256'];self.receipts[sha]+=b' '
        with self.assertRaises(ValueError):self.envelope()
    def test_receipt_text_cannot_differ_from_outcome(self):
        self.outcomes[0]['raw_text']='repaired text'
        with self.assertRaises(ValueError):self.envelope()
    def test_receipt_input_binding_must_match(self):
        old=self.outcomes[0]['execution_receipt_sha256'];obj=json.loads(self.receipts.pop(old));obj['input_binding_sha256']='0'*64
        raw=json.dumps(obj).encode();new=hashlib.sha256(raw).hexdigest();self.receipts[new]=raw;self.outcomes[0]['execution_receipt_sha256']=new
        with self.assertRaises(ValueError):self.envelope()
    def test_receipt_sidecar_preserved(self):
        _,sidecar=self.bundle();self.assertEqual(len(sidecar['records']),16)
        self.assertEqual(sidecar['records'][0]['execution_receipt_sha256'],self.outcomes[0]['execution_receipt_sha256'])
        self.assertEqual(sidecar['records'][0]['model'],self.manifest['models'][0])
        self.assertEqual(sidecar['records'][0]['input_binding_sha256'],self.decoder_index['clip-0001']['binding_sha256'])
        self.assertIn('not inference authenticity',sidecar['scope'])
    def test_second_model_receipts_match_second_slot(self):
        for index in range(16):
            self.alter_receipt(lambda r:r.update(model=dict(self.manifest['models'][1])),index)
        envelope,sidecar=self.bundle(slot=1)
        self.assertEqual(envelope['model'],self.manifest['models'][1])
        self.assertEqual(len(sidecar['records']),16)
    def test_manifest_duplicate_json_keys_rejected(self):
        self.raw=b'{"schema_version":"discarded",'+self.raw.lstrip()[1:]
        self.sha=hashlib.sha256(self.raw).hexdigest()
        with self.assertRaises(ValueError):self.envelope()
    def test_not_run_still_requires_exact_decoder_membership(self):
        self.human['records'][15]['opaque_id']='clip-9999'
        self.outcomes[15].update(opaque_id='clip-9999',status='not_run',raw_text=None,
            completeness='unknown',execution_receipt_sha256=None)
        self.refresh_receipts()
        with self.assertRaisesRegex(ValueError,'membership'):self.envelope()
    def test_not_run_still_requires_decoder_wav_identity(self):
        self.human['records'][15]['wav_sha256']='0'*64
        self.manifest['records'][15]['wav_sha256']='0'*64;self.replace_manifest()
        self.outcomes[15].update(wav_sha256='0'*64,status='not_run',raw_text=None,
            completeness='unknown',execution_receipt_sha256=None)
        self.refresh_receipts()
        with self.assertRaisesRegex(ValueError,'WAV association'):self.envelope()
    def test_missing_raw_receipt_sidecar_rejected(self):
        self.receipts.pop(self.outcomes[0]['execution_receipt_sha256'])
        with self.assertRaises(ValueError):self.envelope()
    def test_unused_raw_receipt_sidecar_rejected(self):
        self.outcomes[0].update(status='not_run',raw_text=None,completeness='unknown',execution_receipt_sha256=None)
        with self.assertRaisesRegex(ValueError,'unused'):self.envelope()
    def test_swapped_clip_receipt_rejected(self):
        first,second=self.outcomes[:2]
        first['execution_receipt_sha256'],second['execution_receipt_sha256']=second['execution_receipt_sha256'],first['execution_receipt_sha256']
        with self.assertRaises(ValueError):self.envelope()
    def test_surrogate_transcript_rejected_even_when_receipt_matches(self):
        self.outcomes[0]['raw_text']='\ud800'
        self.alter_receipt(lambda r:r.update(raw_text='\ud800'))
        with self.assertRaises(ValueError):self.envelope()
    def test_unicode_transcript_preserved_without_normalization(self):
        text='合成，重复重复！é e\u0301 𐀀'
        self.outcomes[0]['raw_text']=text;self.refresh_receipts()
        self.assertEqual(self.envelope()['records'][0]['raw_text'],text)
    def test_1024_character_limit_remains(self):
        self.outcomes[0]['raw_text']='好'*1024;self.refresh_receipts();self.envelope()
        self.outcomes[0]['raw_text']+='好';self.refresh_receipts()
        with self.assertRaises(ValueError):self.envelope()
    def test_referenced_contract_and_decoder_sidecars_are_not_authenticated_here(self):
        self.alter_receipt(lambda r:r.update(decoder_evidence_sha256='0'*64,execution_contract_sha256='a'*64))
        _,sidecar=self.bundle()
        self.assertEqual(sidecar['records'][0]['decoder_evidence_sha256'],'0'*64)
        self.assertIn('independently verified',sidecar['scope'])


for name,change in [
    ('schema',lambda m:m.update(schema_version='incorrect')),
    ('execution_kind',lambda m:m.update(execution_kind='fixture')),
    ('target_order',lambda m:m['target_order'].reverse()),
    ('duplicate_row',lambda m:m['records'].insert(0,dict(m['records'][0],human_target_presence=['positive','positive']))),
    ('missing_row',lambda m:m['records'].pop()),
    ('extra_field',lambda m:m.update(extra=True)),
    ('model_revision',lambda m:m['models'][0].update(revision='0'*40)),
    ('model_fields',lambda m:m['models'][0].update(extra=True)),
    ('run_type',lambda m:m['models'][0].update(run_id=True)),
    ('run_size',lambda m:m['models'][0].update(run_id='x'*129)),
    ('probe_id_type',lambda m:m.update(probe_set_id=1)),
    ('probe_id_surrogate',lambda m:m.update(probe_set_id='\ud800')),
    ('nonfinite',lambda m:m.update(probe_set_id=float('inf'))),
]:
    def test(self,change=change):
        change(self.manifest);self.replace_manifest()
        with self.assertRaises(ValueError):self.envelope()
    setattr(ResultTests,'test_reject_scientific_manifest_'+name,test)

for name,change in [
    ('missing_field',lambda r:r.pop('status')),
    ('extra_field',lambda r:r.update(extra=True)),
    ('schema',lambda r:r.update(schema='incorrect')),
    ('model_id',lambda r:r['model'].update(model_id=SENSE)),
    ('model_revision',lambda r:r['model'].update(revision='0'*40)),
    ('model_run',lambda r:r['model'].update(run_id='another-run')),
    ('model_extra_field',lambda r:r['model'].update(extra=True)),
    ('clip',lambda r:r.update(opaque_id='clip-0002')),
    ('wav',lambda r:r.update(wav_sha256='0'*64)),
    ('status',lambda r:r.update(status='error')),
    ('completeness',lambda r:r.update(completeness='incomplete')),
    ('flags',lambda r:r.update(quality_flags=['ambiguous'])),
    ('decoder_manifest',lambda r:r.update(decoder_manifest_sha256='0'*64)),
    ('attempt_false',lambda r:r.update(attempt_started=False)),
    ('attempt_integer',lambda r:r.update(attempt_started=1)),
    ('forward_false',lambda r:r.update(model_forward_completed=False)),
    ('forward_integer',lambda r:r.update(model_forward_completed=1)),
    ('evidence_hash',lambda r:r.update(decoder_evidence_sha256=None)),
    ('contract_hash',lambda r:r.update(execution_contract_sha256='bad')),
]:
    def test(self,change=change):
        self.alter_receipt(change)
        with self.assertRaises(ValueError):self.envelope()
    setattr(ResultTests,'test_reject_receipt_'+name,test)

for name,raw in [
    ('invalid_json',b'{'),('not_object',b'[]'),('invalid_utf8',b'\xff'),
    ('duplicate_keys',b'{"schema":"one","schema":"two"}'),
    ('nested_duplicate_keys',b'{"model":{"run_id":"one","run_id":"two"}}'),
    ('nan',b'{"extra":NaN}'),('infinity',b'{"extra":1e999}'),
    ('oversized',b' '*(1024*1024+1)),
]:
    def test(self,raw=raw):
        self.replace_receipt_bytes(raw)
        with self.assertRaises(ValueError):self.envelope()
    setattr(ResultTests,'test_reject_receipt_json_'+name,test)


if __name__=='__main__':unittest.main(verbosity=2)
