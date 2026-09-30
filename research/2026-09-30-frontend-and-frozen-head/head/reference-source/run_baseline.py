"""Observed-development cFSMN baseline, not a calibrated KWS qualification.
Reuse audited upstream model, feature math and decoder; do not install SDKs.
"""
import ast, collections, hashlib, importlib.util, json, logging, math, pathlib, struct, subprocess, sys, types, wave
import numpy as np
import torch
import torch.nn.functional as F
ROOT=pathlib.Path(__file__).resolve().parent
DATA=pathlib.Path('/workspace/shared/kws-data-pinned')
RECEIPT=pathlib.Path('/workspace/shared/kws-data-consumer/build/observed-readback/native-export-receipt.json')
BOUNDARY=pathlib.Path('/workspace/shared/kws-data-consumer/build/boundary-control/run-qwen42')
COMMIT='2f9658ffa9568076ef547615c76861abed84f56e'
PHRASES={'你好小窝':1,'小窝小窝':2}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def audited_sources():
 expected={'upstream/wekws/model/fsmn.py':'62265f349ae1dd95580e614557ee58c0ee361fea9e41ac62c6634d5c6ff6c2d3',
 'upstream/wekws/model/cmvn.py':'6b25a77adb3b33361798f0d72110915827015fada0ddb0cc22e30e9d70afe05c',
 'upstream/wekws/bin/stream_kws_ctc.py':'2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53',
 'upstream/torchaudio/kaldi.py':'5cbea1a584ddea748f6f68a621d794e13334e88d9faa1d40986f7af32f196d29',
 'resources/tokens.txt':'41b2c566f0d16ed6a0913d7770520c16556a4bcca206e8012676f847be4b0980',
 'resources/base.pt':'d02b09c34f4a8bbb06f0dd1bf5eb58db3395eb7f1fd15c3625fe09d3a2492233',
 'resources/tokens_2599.txt':'33db42b824dbef866953e6c123293da316872d1fc06e04bc8fdced47d26d8a08',
 'resources/feature_transform.txt.80dim-l2r2':'278d109a9e4e70189e2d657ed52ea846446d5abdd58bcc9cb6b8800a52dac91c',
 'resources/inference_fsmn_4e_l10r2_250_128_fdim80_t2599.yaml':'d0e59ccf122b4679496679b2954b227a57625c859e850397ee878eb9f2c55eae'}
 for p,h in expected.items():assert sha(ROOT/p)==h,(p,'source drift')
 # The only torchaudio native/module dependency is MFCC DCT. Remove unused MFCC
 # and its import; all fbank functions and helpers are unmodified official AST.
 path=ROOT/'upstream/torchaudio/kaldi.py';tree=ast.parse(path.read_text())
 tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name=='torchaudio' for a in n.names)) and not (isinstance(n,ast.FunctionDef) and n.name in ('mfcc','_get_dct_matrix'))]
 assert not any(isinstance(n,ast.Name) and n.id=='torchaudio' for n in ast.walk(tree))
 ns={};exec(compile(tree,str(path),'exec'),ns)
 frontend=types.SimpleNamespace(fbank=ns['fbank'])
 # Preserve the exact official decoder + streaming feature/cache methods.
 path=ROOT/'upstream/wekws/bin/stream_kws_ctc.py';tree=ast.parse(path.read_text())
 names={'is_sublist','ctc_prefix_beam_search','KeyWordSpotter'}
 selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in selected}==names
 namespace=dict(torch=torch,np=np,F=F,struct=struct,math=math,logging=logging,defaultdict=collections.defaultdict)
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
 return frontend,namespace,expected

class Donor(torch.nn.Module):
 def __init__(self):
  super().__init__();state=torch.load(ROOT/'resources/base.pt',weights_only=True,map_location='cpu')
  fsmn=module('audited_fsmn',ROOT/'upstream/wekws/model/fsmn.py');cmvn=module('audited_cmvn',ROOT/'upstream/wekws/model/cmvn.py')
  self.global_cmvn=cmvn.GlobalCMVN(state['global_cmvn.mean'].clone(),state['global_cmvn.istd'].clone(),True)
  self.backbone=fsmn.FSMN(400,140,4,250,128,10,2,1,1,140,2599);self.load_state_dict(state,strict=True)
  self.logs=[]
 def forward(self,x,cache):
  y,c=self.backbone(self.global_cmvn(x),cache);self.logs.append(y.detach().cpu());return y,c

def make_spotter(window,model,frontend,ns,tokens):
 # Avoid unused SDK/YAML/lexicon initialization. Every operational attribute is
 # copied from official constructor/defaults and pinned donor configuration.
 def fbank(waveform,**kw):
  assert waveform.dtype==torch.float32 and waveform.ndim==2 and waveform.shape[0]==1
  return frontend.fbank(waveform,window_type=window,**kw)
 ns['kaldi']=types.SimpleNamespace(fbank=fbank)
 cls=ns['KeyWordSpotter'];k=cls.__new__(cls);torch.nn.Module.__init__(k)
 k.sample_rate=16000;k.wave_remained=np.array([]);k.num_mel_bins=80;k.frame_length=25;k.frame_shift=10
 k.downsampling=3;k.resolution=.01;k.context_expansion=True;k.left_context=2;k.right_context=2
 k.feature_remained=None;k.feats_ctx_offset=0;k.device=torch.device('cpu');k.model=model
 k.in_cache=torch.zeros(0,0,0);k.score_beam=3;k.path_beam=20;k.threshold=0.0
 k.min_frames=5;k.max_frames=250;k.interval_frames=50
 k.cur_hyps=[(tuple(),(1.0,0.0,[]))];k.hit_score=1.0;k.hit_keyword=None;k.activated=False
 k.total_frames=0;k.last_active_pos=-1;k.result={}
 class DirectKnownChars:
  def tokenize(self,text):
   chars=text.split();assert all(c in tokens for c in chars);return chars,[tokens.index(c) for c in chars]
 k.tokenizer=DirectKnownChars();k.set_keywords(','.join(PHRASES));return k

def inputs():
 assert sha(RECEIPT)=='8e4d5c13cc5694e813ef6dc58bedbe34d2b893721283f1945d6d944ca38570b5'
 assert subprocess.check_output(['git','-C',str(DATA),'rev-parse','HEAD'],text=True).strip()==COMMIT
 assert not subprocess.check_output(['git','-C',str(DATA),'status','--porcelain'],text=True).strip()
 receipt=json.loads(RECEIPT.read_text());rows=[dict(r,dataset_id=d['dataset_id']) for d in receipt['datasets'] for r in d['recordings']]
 assert len(rows)==42 and sum(r['kind']=='positive' for r in rows)==20
 source={r['recording']:r for r in rows};conditions={'raw':(DATA,rows)}
 for name in ['tail500ms','head500ms_tail500ms']:
  root=BOUNDARY/name;derived=json.loads((root/'derived-receipt.json').read_text());assert derived['canonical_source_receipt_sha256']==sha(RECEIPT)
  rs=[json.loads(l) for l in (root/'execution-inputs.jsonl').read_text().splitlines()];assert {r['recording'] for r in rs}==set(source)
  for r in rs:
   original=source[r['recording']];assert r['source_file_sha256']==original['file_sha256']
   assert all(r[k]==original[k] for k in ['keyword_id','kind','split','review_method','speaker_id','intended_text'])
   assert r['prefix_frames']==(8000 if name=='head500ms_tail500ms' else 0) and r['tail_frames']==8000
   with wave.open(str(DATA/original['path'])) as w:pcm=w.readframes(w.getnframes())
   with wave.open(str(root/r['path'])) as w:padded=w.readframes(w.getnframes())
   assert padded==bytes(2*r['prefix_frames'])+pcm+bytes(16000)
  conditions[name]=(root,rs)
 for root,rs in conditions.values():
  for r in rs:
   p=root/r['path'];assert sha(p)==r['file_sha256']
   with wave.open(str(p)) as w:
    assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,r['frames'])
    assert hashlib.sha256(w.readframes(w.getnframes())).hexdigest()==r['pcm_sha256']
 return conditions

def setup():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 frontend,ns,hashes=audited_sources();tokens=(ROOT/'resources/tokens_2599.txt').read_text().splitlines();assert len(tokens)==2599 and tokens[0]=='<blank>'
 kaldi=dict(l.split() for l in (ROOT/'resources/tokens.txt').read_text().splitlines())
 assert all(int(kaldi[t])==i+1 for i,t in enumerate(tokens) if i>1)
 conditions=inputs()
 spec={'driver_sha256':sha(pathlib.Path(__file__)),'scope':'observed-development-only-uncalibrated-diagnostic','data_commit':COMMIT,'receipt_sha256':sha(RECEIPT),
 'weights_sha256':hashes['resources/base.pt'],'sources':hashes,'keywords':{w:[tokens.index(c) for c in w] for w in PHRASES},
 'routes':[{'window':'hamming','role':'provider-published-window-with-WeKws-stream-decoder'}, {'window':'povey','role':'predeclared-WeKws-default-compatibility-control'}],
 'conditions':list(conditions),'clips_per_condition':42,'total_inferences':252,'threshold':0.0,'score_beam':3,'path_beam':20,'frame_probability_prune':0.05,
 'min_frames':5,'max_frames':250,'interval_frames':50,'feed_samples':4800,'frontend':'official-source-only-torchaudio-2.11.0-kaldi-fbank',
 'frontend_config':{'amplitude':'unscaled PCM16 as float32','sample_rate':16000,'mel_bins':80,'frame_length_ms':25,'hop_ms':10,'dither':0,'energy_floor':0.0,'snip_edges':True,'splice_left':2,'splice_right':2,'frame_skip':3},
 'eof_policy':'official-WeKws-demo-no-additional-flush-or-padding','lookahead_note':'two frontend right frames; four rorder2 FSMN layers implemented with delayed finite cache; timestamps are upstream decoder labels, not latency qualification',
 'training':False,'threshold_selection':False,'window_selection':False,'qualification_allowed':False,
 'uncertainty':'Provider inference YAML declares Hamming; WeKws defaults to Povey. These are two predefined routes, not proof of exact original base training frontend or an equal-FAR comparison.',
 'runtime':{'python':sys.version,'torch':torch.__version__,'torch_path':torch.__file__,'numpy':np.__version__,'numpy_path':np.__file__}}
 return frontend,ns,tokens,conditions,spec

def main():
 frontend,ns,tokens,conditions,spec=setup()
 if sys.argv[1]=='prepare':
  
  if (ROOT/'baseline-spec.json').exists():assert json.loads((ROOT/'baseline-spec.json').read_text())==spec
  else:save(ROOT/'baseline-spec.json',spec)
  # Silence floor/frame-count check of unchanged official fbank source.
  for window in ['hamming','povey']:
   x=torch.zeros(1,1600);y=frontend.fbank(x,num_mel_bins=80,dither=0,energy_floor=0,window_type=window)
   assert y.shape==(8,80) and torch.isfinite(y).all();assert torch.allclose(y,torch.full_like(y,math.log(torch.finfo(torch.float32).eps)))
  model=Donor().eval();smoke=[]
  with torch.inference_mode():
   for window in ['hamming','povey']:
    model.logs=[];k=make_spotter(window,model,frontend,ns,tokens)
    result=k.forward(bytes(9600));assert len(model.logs)==1
    logits=model.logs[0];assert logits.shape==(1,9,2599) and torch.isfinite(logits).all()
    smoke.append({'window':window,'synthetic_zero_pcm_samples':4800,'logit_shape':list(logits.shape),'cache_shape':list(k.in_cache.shape),'events':result})
  save(ROOT/'synthetic-smoke.json',smoke)
  print('preflight, source-only frontend and complete synthetic smoke PASS; no corpus inference');return
 assert sys.argv[1]=='run';assert json.loads((ROOT/'baseline-spec.json').read_text())==spec
 outroot=ROOT/'results';outroot.mkdir(exist_ok=False);model=Donor().eval();reports=[]
 with torch.inference_mode():
  for route in spec['routes']:
   window=route['window']
   for condition,(root,rows) in conditions.items():
    results=[]
    for row in rows:
     model.logs=[];k=make_spotter(window,model,frontend,ns,tokens);p=root/row['path'];assert sha(p)==row['file_sha256']
     with wave.open(str(p)) as w:pcm=w.readframes(w.getnframes())
     events=[]
     for start in range(0,len(pcm),9600):
      end=min(len(pcm),start+9600);result=k.forward(pcm[start:end])
      if result.get('state')==1:events.append(dict(result,keyword_id=PHRASES[result['keyword']],available_audio_samples=end//2,eof_flush=False))
     logits=torch.cat(model.logs,1)[0];ids=logits.argmax(1).tolist();collapsed=[];previous=None
     for idx in ids:
      if idx!=previous and idx!=0:collapsed.append(idx)
      previous=idx
     positive=row['kind']=='positive';hits=sum(e['keyword_id']==row['keyword_id'] for e in events) if positive else 0
     results.append(dict(row,events=events,target_hit=bool(hits) if positive else None,greedy_ctc_text=''.join(tokens[i] for i in collapsed),greedy_ctc_ids=collapsed,acoustic_frames=len(ids),logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest()))
    summary={'window':window,'condition':condition,'positive_target_hits':sum(r['target_hit'] is True for r in results),'positive_clips':20,'confusable_clips_with_events':sum(bool(r['events']) for r in results if r['kind']!='positive'),'confusable_events':sum(len(r['events']) for r in results if r['kind']!='positive'),'wrong_keyword_events':sum(e['keyword_id']!=r['keyword_id'] for r in results if r['kind']=='positive' for e in r['events']),'all_events':sum(len(r['events']) for r in results)}
    report=dict(summary=summary,recordings=results);save(outroot/f'{window}-{condition}.json',report);reports.append(summary);print(json.dumps(summary),flush=True)
 save(outroot/'summary.json',reports)
 save(outroot/'provenance.json',{'spec_sha256':sha(ROOT/'baseline-spec.json'),'script_sha256':sha(pathlib.Path(__file__)),'readbacks':{p.name:sha(p) for p in outroot.glob('*.json')},'sources':spec['sources'],'runtime':spec['runtime']})
 inputs();print('post-run all input identities verified')
if __name__=='__main__':main()
