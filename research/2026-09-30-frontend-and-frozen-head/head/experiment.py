"""Locked, local observed-development P/R frozen-encoder clip BCE experiment.
No imports of the failed native C implementation; no network or external writes.
"""
import argparse, copy, hashlib, importlib.metadata, importlib.util, json, math, os, pathlib, struct, sys, time, traceback, wave
import numpy as np
import torch
import torch.nn.functional as F
ROOT = pathlib.Path(__file__).resolve().parent
BASE = pathlib.Path('/workspace/shared/kws-cfsmn-baseline')
DATA = pathlib.Path('/workspace/shared/kws-data-pinned')
FLEURS = pathlib.Path('/workspace/shared/kws-fleurs-train-preflight')
QRECEIPT = pathlib.Path('/workspace/shared/kws-data-consumer/build/observed-readback/native-export-receipt.json')
PYTHON = '/workspace/shared/kws-lightweight-prototypes/venv/bin/python'
TARGETS = ['你好小窝', '小窝小窝']
THREAD_ENV = {k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']}
THREAD_ENV.update({'PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
EXPECTED = {
 str(QRECEIPT):'8e4d5c13cc5694e813ef6dc58bedbe34d2b893721283f1945d6d944ca38570b5',
 str(FLEURS/'audio-extraction-receipt.json'):'ea3332a8149aa678660c3d2ad3bf93ced433fca62923e104cc5d8e5e50557f61',
 str(BASE/'run_baseline.py'):'498af9c026d6172fc1fe92616d2edc2c0cc848dae2f410bcef3822912eb8cbd0'}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20), b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with open(p,'x') as f: json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
def read(p):return json.loads(pathlib.Path(p).read_text())
def load_module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def finite(x,name):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all().item(), 'nonfinite: '+name
 elif isinstance(x,np.ndarray):assert np.isfinite(x).all(), 'nonfinite: '+name
 elif isinstance(x,(tuple,list)):
  for i,v in enumerate(x):finite(v,f'{name}[{i}]')
 elif isinstance(x,dict):
  for k,v in x.items():finite(v,f'{name}.{k}')
 elif isinstance(x,float):assert math.isfinite(x),'nonfinite: '+name

def tensor_records(state):
 out={}
 for k,t in sorted(state.items()):
  finite(t,k); a=t.detach().cpu().contiguous().numpy()
  out[k]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
 return out

def state_hash(state):return hashlib.sha256(json.dumps(tensor_records(state),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def setup():
 assert sys.executable==PYTHON,(sys.executable,PYTHON)
 for k,v in THREAD_ENV.items():assert os.environ.get(k)==v,('thread environment mismatch',k)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 assert torch.get_num_threads()==torch.get_num_interop_threads()==1
 for p,h in EXPECTED.items():assert sha(p)==h,('fixed source drift',p)
 b=load_module('locked_donor_adapter',BASE/'run_baseline.py')
 frontend,ns,sources=b.audited_sources()
 tokens=(BASE/'resources/tokens_2599.txt').read_text().splitlines()
 return b,frontend,ns,tokens,{str(BASE/p):h for p,h in sources.items()}

def models(b):
 # Use published strict-loading donor, preserve exact FSMN.forward and CMVN.
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(1337)
  p=b.Donor().eval()
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(1337)
  r=b.Donor().eval()
  # Construct random FSMN from the upstream class at an explicitly reset RNG.
  torch.manual_seed(1337)
  r.backbone=type(p.backbone)(400,140,4,250,128,10,2,1,1,140,2599)
 for m in [p,r]:
  # Original forward unpacks the identity tuple at x6, yielding out_linear1.
  m.backbone.out_linear2=torch.nn.Identity()
  m.eval();m.requires_grad_(False)
  for name,module in m.named_modules():
   module.register_forward_hook(lambda _m,_x,y,n=name:finite(y,'encoder.'+n))
 assert all(torch.equal(p.global_cmvn.state_dict()[k],r.global_cmvn.state_dict()[k]) for k in p.global_cmvn.state_dict())
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(7331)
  head=torch.nn.Linear(140,2)
 assert sum(x.numel() for x in head.parameters())==282
 init={'P':{'state_sha256':state_hash(p.state_dict()),'tensors':tensor_records(p.state_dict())},
       'R':{'state_sha256':state_hash(r.state_dict()),'tensors':tensor_records(r.state_dict())},
       'head':{'state_sha256':state_hash(head.state_dict()),'tensors':tensor_records(head.state_dict())},
       'cmvn':{'state_sha256':state_hash(p.global_cmvn.state_dict()),'tensors':tensor_records(p.global_cmvn.state_dict())}}
 assert init['P']['state_sha256']!=init['R']['state_sha256']
 return {'P':p,'R':r},head,init

class Tagged:
 """No audio inference: deterministic ordinal tags replace synthetic fbank."""
 def __init__(self):self.count=0
 def fbank(self,w,**kw):
  n=1+(w.shape[1]-400)//160
  x=torch.arange(self.count,self.count+n,dtype=torch.float32)[:,None].repeat(1,80)
  self.count+=n;return x

def frame_contract(b,ns,tokens,samples):
 tag=Tagged();k=b.make_spotter('hamming',None,tag,ns,tokens);centers=[];counts=[]
 for start in range(0,samples,4800):
  x=k.accept_wave(bytes(2*min(4800,samples-start)))
  counts.append(0 if x is None else len(x))
  if x is not None:
   assert x.ndim==2 and x.shape[1]==400
   centers+=x[:,160].to(torch.int64).tolist()
 assert centers==list(range(0,3*len(centers),3)),('noncanonical feature centers',samples,centers)
 return {'encoded_frames':len(centers),'eligible_frames':max(0,len(centers)-8),
         'per_call_encoded_frames':counts,'first_eligible_output_index':8 if len(centers)>8 else None,
         'last_eligible_output_index':len(centers)-1 if len(centers)>8 else None,
         'nominal_center_fbank_first':0 if len(centers)>8 else None,
         'nominal_center_fbank_last':centers[-9] if len(centers)>8 else None,
         'no_flush_pending_pcm_samples':len(k.wave_remained)}

def verify_wav(row):
 p=pathlib.Path(row['absolute_path']);assert sha(p)==row['file_sha256'],('wav hash',row['recording'])
 with wave.open(str(p),'rb') as w:
  assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes(),w.getcomptype())==(1,2,16000,row['frames'],'NONE')
  pcm=w.readframes(w.getnframes())
 assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
 assert len(pcm)==2*row['frames']
 finite(np.frombuffer(pcm,dtype='<i2'),'pcm')
 return pcm

def dataset(b,ns,tokens,stage):
 git=read(ROOT/f'git-precheck-{stage}.json')
 assert git['repository']==str(DATA) and git['commit']=='2f9658ffa9568076ef547615c76861abed84f56e'
 assert git['status_porcelain']=='' and git['clean'] is True
 assert sha(git['git_executable'])==git['git_executable_sha256']
 q=read(QRECEIPT);rows=[]
 for d in q['datasets']:
  for original in d['recordings']:
   x=copy.deepcopy(original);x['dataset_id']=d['dataset_id'];x['absolute_path']=str(DATA/x['path']);x['source']='Qwen'
   assert x['review_verdict']=='accepted'
   x['labels']=[int(x['kind']=='positive' and x['keyword_id']==i+1) for i in range(2)]
   assert x['labels']==[int(t in x['intended_text']) for t in TARGETS]
   x['recipe_role']='train' if x['split']=='train' and x['review_method']=='human' else ('primary_observed_development' if x['split'] in ['development_a','development_b'] and x['review_method']=='human' else 'supplementary_readout_only')
   x['label_evidence']=x['review_evidence_class'];rows.append(x)
 assert len(rows)==42
 f=read(FLEURS/'audio-extraction-receipt.json');package=read(FLEURS/'local-source-package.json')
 assert package['admitted_for_training'] is False
 for original in f['records']:
  assert original['admitted_for_training'] is False and original['role']=='train' and original['human_reviewed_by_us'] is False
  assert all(t not in original['normalized_text'] for t in TARGETS)
  rows.append({'recording':original['source_id'],'source':'FLEURS','absolute_path':original['derived_path'],
   'file_sha256':original['derived_wav_sha256'],'pcm_sha256':original['derived_pcm_sha256'],'frames':original['frames'],
   'duration_s':original['duration_s'],'source_id':original['source_id'],'split':'native_train','recipe_role':'train',
   'labels':[0,0],'label_evidence':'source-provided-transcription','human_reviewed_by_us':False,
   'source_receipt_admitted_for_training':False,'recipe_specific_authorization':'parent-approved-fixed-two-output-keyword-absence-only-2026-09-30',
   'intended_text':original['normalized_text'],'source_record':original})
 assert len(rows)==74
 train=[x for x in rows if x['recipe_role']=='train'];dev=[x for x in rows if x['recipe_role']=='primary_observed_development']
 assert len(train)==44 and len(dev)==8
 assert sum(x['source']=='Qwen' for x in train)==12 and sum(x['source']=='FLEURS' for x in train)==32
 assert np.array([x['labels'] for x in train]).sum(axis=0).tolist()==[3,3]
 assert sum(x['kind']=='positive' for x in dev)==4
 assert {x['speaker_id'] for x in dev}=={'Serena','Eric'}
 assert not ({x['speaker_id'] for x in train if x['source']=='Qwen'} & {x['speaker_id'] for x in dev})
 assert len({x['recording'] for x in rows})==len(rows) and len({x['pcm_sha256'] for x in rows})==len(rows)
 # Frame counts depend only on sample counts and the pinned upstream index oracle.
 for x in rows:
  verify_wav(x);x['frame_contract']=frame_contract(b,ns,tokens,x['frames'])
  assert x['frame_contract']['eligible_frames']>0,('no eligible frames',x['recording'])
 return rows

def extract_pcm(pcm,models,b,frontend,ns,tokens,expected=None):
 k=b.make_spotter('hamming',None,frontend,ns,tokens)
 streams={name:{'cache':torch.zeros(0,0,0),'parts':[]} for name in models};calls=[]
 with torch.inference_mode():
  for start in range(0,len(pcm),9600):
   feats=k.accept_wave(pcm[start:start+9600]);calls.append(0 if feats is None else len(feats))
   if feats is None or len(feats)==0:continue
   finite(feats,'spliced_fbank')
   for name,m in models.items():
    # Avoid Donor.logs growth; identical original CMVN and FSMN operations.
    out,cache=m.backbone(m.global_cmvn(feats[None]),streams[name]['cache'])
    assert out.shape==(1,len(feats),140);finite((out,cache),'encoded_and_cache')
    streams[name]['parts'].append(out[0].clone());streams[name]['cache']=cache
 pools={};info={}
 for name,s in streams.items():
  assert s['parts'], 'no encoded frames'
  frames=torch.cat(s['parts'],0);assert len(frames)>8,'no eligible frames'
  eligible=frames[8:];pool=eligible.mean(0);finite(pool,'clip_pool');pools[name]=pool.clone()
  info[name]={'encoded_frames':len(frames),'eligible_frames':len(eligible),'encoded_tensor_sha256':state_hash({'frames':frames}),
              'pooled_tensor_sha256':state_hash({'pool':pool}),'per_call_encoded_frames':calls}
  if expected:
   for key in ['encoded_frames','eligible_frames','per_call_encoded_frames']:assert info[name][key]==expected[key],(name,key)
 return pools,info

def synthetic_checks(b,frontend,ns,tokens,encoders,head):
 # Actual four memory blocks, zero conv taps: identity path delays 2 per layer.
 fm=b.module('marker_fsmn',BASE/'upstream/wekws/model/fsmn.py')
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(999)
  blocks=[fm.FSMNBlock(1,1,10,2,1,1).eval() for _ in range(4)]
 for m in blocks:
  with torch.no_grad():m.conv_left.weight.zero_();m.conv_right.weight.zero_()
 x=torch.arange(1,38,dtype=torch.float32).reshape(1,37,1)
 def process(parts):
  cache=[None]*4;out=[];start=0
  with torch.inference_mode():
   for n in parts:
    y=x[:,start:start+n];start+=n
    for i,m in enumerate(blocks):y,cache[i]=m((y,cache[i]))
    out.append(y)
  return torch.cat(out,1)
 expected=torch.cat([torch.zeros(1,8,1),x[:,:-8]],1)
 assert torch.equal(process([37]),expected) and torch.equal(process([1,3,9,8,16]),expected)
 assert torch.equal(expected[:,8:],x[:,:29])
 edge={str(n):frame_contract(b,ns,tokens,n) for n in [799,800,959,960,4799,4800,4801,9600,10399,10400,16321]}
 assert edge['4800']['encoded_frames']==9 and edge['4800']['eligible_frames']==1
 assert edge['9600']['encoded_frames']==19 and edge['9600']['eligible_frames']==11
 before={k:state_hash(m.state_dict()) for k,m in encoders.items()}
 audio=[]
 for label,a in [('silence',np.zeros(16321,dtype='<i2')),('bounded_ramp',((np.arange(16321)%4096)-2048).astype('<i2'))]:
  pool,info=extract_pcm(a.tobytes(),encoders,b,frontend,ns,tokens,edge['16321'])
  again,again_info=extract_pcm(a.tobytes(),encoders,b,frontend,ns,tokens,edge['16321'])
  assert info==again_info and all(torch.equal(pool[k],again[k]) for k in pool)
  audio.append({'case':label,'frames':info})
 # One artificial 44-vector optimizer smoke, no data or real encoder features.
 smoke=copy.deepcopy(head);opt=make_optimizer(smoke)
 xx=torch.arange(44*140,dtype=torch.float32).reshape(44,140)/6160
 yy=torch.zeros(44,2);yy[:3,0]=1;yy[3:6,1]=1
 logits=smoke(xx);loss=F.binary_cross_entropy_with_logits(logits,yy,pos_weight=torch.tensor([41/3,41/3]));finite(loss,'synthetic_loss')
 loss.backward();finite([p.grad for p in smoke.parameters()],'synthetic_gradient');torch.nn.utils.clip_grad_norm_(smoke.parameters(),1,error_if_nonfinite=True);opt.step();finite(smoke.state_dict(),'synthetic_head');finite(opt.state_dict(),'synthetic_optimizer')
 assert before=={k:state_hash(m.state_dict()) for k,m in encoders.items()}
 return {'four_actual_FSMN_blocks_identity_marker_delay':8,'whole_vs_chunks_exact':True,'reset_replay_exact':True,
         'edge_frame_contracts':edge,'synthetic_audio_checks':audio,'artificial_optimizer_smoke_updates':1,'corpus_model_inference':False,'corpus_training':False}

def make_optimizer(head):
 return torch.optim.AdamW(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,maximize=False,capturable=False,differentiable=False,fused=False)

def runtime():
 import torch.nn.modules.linear,torch.nn.modules.conv,torch.optim.adamw,torch.nn.functional
 selected=[sys.executable,torch.__file__,np.__file__,torch.nn.modules.linear.__file__,torch.nn.modules.conv.__file__,sys.modules['torch.optim.adamw'].__file__,torch.nn.functional.__file__]
 return {'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'numpy':np.__version__,
 'platform':dict(zip(['system','node','release','version','machine'],os.uname())),'machine':os.uname().machine,'byteorder':sys.byteorder,
 'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
 'selected_runtime_file_hashes':{str(p):sha(p) for p in selected},'thread_environment':THREAD_ENV,
 'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
 'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'torch_build_config':torch.__config__.show()}

def preflight():
 b,frontend,ns,tokens,sources=setup();rows=dataset(b,ns,tokens,'preflight');encoders,head,init=models(b)
 tests=synthetic_checks(b,frontend,ns,tokens,encoders,head)
 # Explicitly preserve all prior receipts and failed outcomes in place.
 source_files=[FLEURS/'local-source-package.json',FLEURS/'selection-spec.json',FLEURS/'selected-metadata.json',
  BASE/'baseline-spec.json',BASE/'upstream-source-manifest.json',pathlib.Path('/workspace/shared/kws-two-output-bce-option.md'),
  pathlib.Path('/workspace/shared/kws-fsmn-native-spec/SPEC.md')]
 sources.update(EXPECTED);sources.update({str(p):sha(p) for p in source_files})
 train=[r for r in rows if r['recipe_role']=='train']
 spec={'experiment_id':'frozen-keyword-head-pr-v1','status':'awaiting-independent-preflight-review',
 'scope':'one-fixed-P-R-local-observed-development-clip-classifier-experiment',
 'authorization':'Parent task-specific approval for exactly12 human-reviewed Qwen TRAIN plus fixed32 FLEURS native TRAIN source-labelled target-absence; original FLEURS admitted_for_training=false unchanged',
 'source_paths_sha256':sources,'data_commit':'2f9658ffa9568076ef547615c76861abed84f56e',
 'targets':TARGETS,'train_count':44,'qwen_train_human_count':12,'fleurs_train_source_count':32,
 'positive_counts':[3,3],'negative_counts':[41,41],'pos_weight':[41/3,41/3],
 'arms':['P','R'],'random_encoder_seed':1337,'independent_shared_head_seed':7331,
 'encoder':'pinned Python donor FSMN.forward with out_linear2 replaced by tuple-preserving Identity; published CMVN retained in both; P published weights, R freshly seeded upstream default initialization; frozen eval',
 'feature_dim':140,'trainable_parameters_per_arm':282,'feature_dtype':'float32',
 'frontend':{'sample_rate':16000,'pcm':'unscaled PCM16 float32','window':'hamming','mel_bins':80,'frame_length_samples':400,'hop_samples':160,'dither':0,'energy_floor':0,'snip_edges':True,'splice':[-2,2],'skip':3,'cmvn_dim':400,'chunk_samples':4800,'eof':'one final actual short call then stop; no additional padding or flush'},
 'frame_mask':'zero-based emitted encoded index >= 8; nominal center index encoded_index-8; arithmetic mean of eligible140D frames; reject zero eligible; delayed final8 nominal steps unobserved',
 'normalization':{'fit':'44 TRAIN pooled vectors only, per-arm, float32','mean':'torch.mean(dim=0)','std':'torch.std(dim=0,correction=0)','std_floor':1e-5,'dev_updates':False},
 'optimizer':{'name':'AdamW','steps':100,'lr':.001,'weight_decay':0,'betas':[.9,.999],'eps':1e-8,'amsgrad':False,'foreach':False,'fused':False,'full_batch':44,'grad_norm_cap':1,'loss':'BCEWithLogits pos_weight=[41/3,41/3] reduction=mean'},
 'evaluation':{'threshold':.5,'rule':'sigmoid(logit)>=0.5 independently','primary':'8 human-reviewed Qwen observed A/B dev; unweighted BCE per-output, label group and voice; all16 output bits','representation_gate':'P mean unweighted dev BCE < R separately for both outputs','keyword_exact_gate':'all8 dev clips have both bits correct including no extra output activation','supplement':'all42 Qwen retaining original split/review roles','qualification_allowed':False},
 'resource_limits':{'cpu_seconds_process_group':600,'rlimit_cpu_soft':600,'rlimit_cpu_hard':605,'wall_seconds':900,'rss_bytes':2147483648,'minimum_launch_available_bytes':3221225472,'minimum_available_bytes':1073741824,'monitor_interval_seconds':.1,'termination_grace_seconds':3},
 'prohibitions':['no development training or normalization','no hyperparameter or threshold sweep','no alternate seed','no rerun after failure','no failed C kernels','no40 HI-MIA or later6 failed Qwen','no source receipt edits','no fresh-holdout/FAR/streaming/device-latency/product pass claim'],
 'limitations':['12/32 mix confounds keyword absence with corpus/recording style','FLEURS transcript/audio errors remain possible','four nominal delayed steps per two-block pair; masked initial8, unobserved tail','weighted loss scores are not calibrated probabilities','frozen random encoder comparison is representation utility, not full from-scratch training','prior CTC/distillation/native numerical failures unchanged'],
 'runtime':runtime(),'train_recording_order':[r['recording'] for r in train]}
 for name,value in [('dataset.json',rows),('initialization.json',init),('synthetic-checks.json',tests),('spec.json',spec)]:save(ROOT/name,value)
 locked=['dataset.json','initialization.json','synthetic-checks.json','spec.json','experiment.py','launch.py','test_contract.py','unit-tests.json','tests-resource.json','git-precheck-preflight.json']
 lock={p:sha(ROOT/p) for p in locked};save(ROOT/'preflight-lock.json',lock)
 print(json.dumps({'preflight':'PASS','lock_sha256':sha(ROOT/'preflight-lock.json'),'train':44,'dev':8,'all_Qwen':42,'corpus_training':False}),flush=True)

def check_lock(require_review=False):
 lock=read(ROOT/'preflight-lock.json')
 for p,h in lock.items():assert sha(ROOT/p)==h,('preflight drift',p)
 spec=read(ROOT/'spec.json')
 for p,h in spec['source_paths_sha256'].items():assert sha(p)==h,('source drift',p)
 if require_review:
  a=read(ROOT/'independent-approval.json');assert a['approved'] is True and a['preflight_lock_sha256']==sha(ROOT/'preflight-lock.json')
 return spec

def normalize_train(x):
 assert x.shape==(44,140);finite(x,'train_pool')
 mean=x.mean(0);raw=x.std(0,correction=0);std=raw.clamp_min(1e-5);out=(x-mean)/std
 finite((mean,raw,std,out),'train_normalization')
 return out,mean,std,raw

def train_head(x,y,head,out,name):
 assert x.shape==(44,140) and y.shape==(44,2)
 positives=y.sum(0);weight=(len(y)-positives)/positives;assert positives.tolist()==[3,3]
 assert torch.equal(weight,torch.tensor([41/3,41/3]))
 model=copy.deepcopy(head);init=state_hash(model.state_dict());opt=make_optimizer(model);history=[]
 with open(out/f'{name}-steps.jsonl','x') as f:
  for step in range(1,101):
   opt.zero_grad(set_to_none=True);logits=model(x);finite(logits,'training_logits')
   loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=weight,reduction='mean');finite(loss,'training_loss');loss.backward()
   assert all(p.grad is not None for p in model.parameters());finite([p.grad for p in model.parameters()],'training_gradient')
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);finite(norm,'grad_norm');opt.step()
   finite(model.state_dict(),'trained_head');finite(opt.state_dict(),'optimizer_state')
   record={'step':step,'weighted_train_BCE_before_update':float(loss.detach()),'gradient_norm_before_clip':float(norm)}
   history.append(record);f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
 assert len(history)==100 and all(int(s['step'].item())==100 for s in opt.state.values())
 return model,{'initial_head_state_sha256':init,'final_head_state_sha256':state_hash(model.state_dict()),'optimizer_updates':100,'training_history':history}

def evaluate_rows(rows,pools,model,mean,std):
 x=torch.stack(pools);finite(x,'evaluation_pool');z=(x-mean)/std;finite(z,'evaluation_normalization')
 y=torch.tensor([r['labels'] for r in rows],dtype=torch.float32)
 with torch.no_grad():
  logits=model(z);probs=logits.sigmoid();loss=F.binary_cross_entropy_with_logits(logits,y,reduction='none');finite((logits,probs,loss),'evaluation')
  decisions=(probs>=.5).to(torch.int64)
 return [{**{k:r.get(k) for k in ['recording','source','split','speaker_id','review_method','recipe_role','label_evidence','intended_text','labels']},
          'logits':l.tolist(),'sigmoid_scores_uncalibrated':p.tolist(),'decisions':d.tolist(),'unweighted_BCE_by_output':v.tolist(),
          'both_bits_correct':bool(torch.equal(d,yy.to(torch.int64)))} for r,l,p,d,v,yy in zip(rows,logits,probs,decisions,loss,y)]

def aggregate(evaluated):
 primary=[x for x in evaluated if x['recipe_role']=='primary_observed_development'];assert len(primary)==8
 loss=np.array([x['unweighted_BCE_by_output'] for x in primary]);labels=np.array([x['labels'] for x in primary])
 return {'primary_unweighted_BCE_by_output':loss.mean(0).tolist(),'primary_unweighted_BCE_mean':float(loss.mean()),
 'primary_positive_label_BCE_by_output':[float(loss[labels[:,i]==1,i].mean()) for i in range(2)],
 'primary_negative_label_BCE_by_output':[float(loss[labels[:,i]==0,i].mean()) for i in range(2)],
 'primary_exact_clips':sum(x['both_bits_correct'] for x in primary),'primary_exact_bits':sum(sum(a==b for a,b in zip(x['labels'],x['decisions'])) for x in primary),
 'primary_voice':{v:{'clips':[x for x in primary if x['speaker_id']==v],
   'unweighted_BCE_by_output':np.array([x['unweighted_BCE_by_output'] for x in primary if x['speaker_id']==v]).mean(0).tolist()} for v in ['Serena','Eric']}}

def run():
 spec=check_lock(require_review=True);b,frontend,ns,tokens,_=setup();assert runtime()==spec['runtime'],'runtime drift'
 current=dataset(b,ns,tokens,'run');rows=read(ROOT/'dataset.json');assert current==rows,'dataset drift'
 encoders,head,init=models(b);assert init==read(ROOT/'initialization.json'),'initialization drift'
 out=ROOT/'results';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
 train=[r for r in rows if r['recipe_role']=='train'];qwen=[r for r in rows if r['source']=='Qwen']
 pools={name:{} for name in encoders};frames=[]
 def extract(row):
  values,receipt=extract_pcm(verify_wav(row),encoders,b,frontend,ns,tokens,row['frame_contract'])
  for name,v in values.items():pools[name][row['recording']]=v.clone()
  item={'recording':row['recording'],'recipe_role':row['recipe_role'],'arms':receipt};frames.append(item)
  with open(out/'extraction.jsonl','a') as f:f.write(json.dumps(item,allow_nan=False)+'\n')
 for row in train:extract(row)
 print('All44 TRAIN features extracted; no development model inference or training yet',flush=True)
 heads={};stats={};training={};artifacts={}
 y=torch.tensor([r['labels'] for r in train],dtype=torch.float32)
 for name in ['P','R']:
  features=torch.stack([pools[name][r['recording']] for r in train]);x,mean,std,raw=normalize_train(features)
  heads[name],training[name]=train_head(x.clone(),y.clone(),head,out,name);stats[name]=(mean,std)
  training[name]['normalization_fit_recordings']=[r['recording'] for r in train]
  training[name]['normalization_hashes']=tensor_records({'mean':mean,'std':std,'raw_std':raw})
  training[name]['std_floored_dimensions']=int((raw<1e-5).sum())
  artifacts.update({f'{name}_train_pool':features.numpy(),f'{name}_norm_mean':mean.numpy(),f'{name}_norm_std':std.numpy(),f'{name}_norm_raw_std':raw.numpy(),f'{name}_head_weight':heads[name].weight.detach().numpy(),f'{name}_head_bias':heads[name].bias.detach().numpy()})
  save(out/f'{name}-training.json',training[name]);print(f'{name}: completed exactly100 updates',flush=True)
 assert training['P']['initial_head_state_sha256']==training['R']['initial_head_state_sha256']==init['head']['state_sha256']
 # All real optimizations complete before reading any development feature vectors.
 for row in qwen:
  if row['recording'] not in pools['P']:extract(row)
 reports={};summary={}
 for name in ['P','R']:
  reports[name]=evaluate_rows(qwen,[pools[name][r['recording']] for r in qwen],heads[name],*stats[name]);summary[name]=aggregate(reports[name])
  save(out/f'{name}-qwen42-readout.json',reports[name]);artifacts[f'{name}_qwen42_pool']=torch.stack([pools[name][r['recording']] for r in qwen]).numpy()
  assert state_hash(encoders[name].state_dict())==init[name]['state_sha256'],'frozen encoder changed'
  assert all(not p.requires_grad and p.grad is None for p in encoders[name].parameters())
 representation=[a<b for a,b in zip(summary['P']['primary_unweighted_BCE_by_output'],summary['R']['primary_unweighted_BCE_by_output'])]
 summary['gates']={'P_lower_BCE_than_R_by_output':representation,'representation_advantage_both_outputs':all(representation),
  'P_primary_all8_exact':summary['P']['primary_exact_clips']==8,'R_primary_all8_exact':summary['R']['primary_exact_clips']==8,
  'streaming_or_product_qualification':False}
 np.savez(out/'pooled-normalization-head-artifacts.npz',**artifacts)
 summary['resource']={'inner_wall_seconds':time.monotonic()-start,'inner_cpu_seconds':time.process_time()-cpu}
 summary['scope']='tiny repeatedly observed development only; no FAR, fresh holdout, streaming event or SSC305 performance claim'
 check_lock(require_review=True)
 for row in rows:verify_wav(row)
 assert read(FLEURS/'local-source-package.json')['admitted_for_training'] is False
 save(out/'summary.json',summary)
 save(out/'provenance.json',{'preflight_lock_sha256':sha(ROOT/'preflight-lock.json'),'approval_sha256':sha(ROOT/'independent-approval.json'),
  'runtime':runtime(),'frozen_state_unchanged':True,'source_files_unchanged':True,'all74_WAV_PCM_identities_unchanged':True,
  'results_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
 print(json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['preflight','run']);args=a.parse_args()
 try:globals()[args.action]()
 except BaseException:
  if args.action=='run' and (ROOT/'results').exists():
   p=ROOT/'results'/'failure.json'
   if not p.exists():save(p,{'status':'FAILED_STOP_NO_RETRY','exception':traceback.format_exc()})
  raise
