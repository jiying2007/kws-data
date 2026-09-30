"""Stage-one fail-fast teacher-reference gate; never trains a student."""
import hashlib, importlib.util, json, pathlib, resource, sys, time, wave
ROOT=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/workspace/shared/kws-cfsmn-baseline')
REAL=pathlib.Path('/workspace/shared/kws-real-negative-preflight')
s=importlib.util.spec_from_file_location('b',BASE/'run_baseline.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
torch=b.torch
TOKENS=['<blank>','你','好','小','窝','OTHER']

def tensor_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().contiguous().numpy().tobytes())
 return h.hexdigest()

def aggregate(probabilities, ids):
 assert probabilities.shape[-1]==2599 and len(ids)==5 and len(set(ids))==5
 assert torch.isfinite(probabilities).all() and (probabilities>=0).all()
 assert torch.allclose(probabilities.sum(-1),torch.ones_like(probabilities[...,0]),atol=1e-6,rtol=0)
 mask=torch.ones(2599,dtype=torch.bool);mask[ids]=False
 q=torch.cat([probabilities[...,ids],probabilities[...,mask].sum(-1,keepdim=True)],dim=-1)
 assert torch.allclose(q.sum(-1),torch.ones_like(q[...,0]),atol=1e-6,rtol=0)
 return q

class AggregatedTeacher(torch.nn.Module):
 def __init__(self,teacher,ids):
  super().__init__();self.teacher=teacher;self.ids=ids;self.q=[]
 def forward(self,x,cache):
  with torch.no_grad():
   logits,cache=self.teacher(x,cache);q=aggregate(logits.double().softmax(-1),self.ids).float();self.q.append(q.detach().clone())
   # log(0) may be -inf; no mass is clipped or discarded. Upstream softmax
   # reconstructs the complete six-way distribution to floating-point tolerance.
   logq=q.log();assert torch.allclose(logq.softmax(-1),q,atol=1e-6,rtol=0)
   return logq,cache

def make(model,front,ns):
 k=b.make_spotter('hamming',model,front,ns,TOKENS);k.keywords_idxset.add(5);assert k.keywords_idxset==set(range(6));return k

def setup():
 front,ns,dict2599,conditions,old=b.setup();teacher=b.Donor().eval()
 for p in teacher.parameters():p.requires_grad_(False)
 ids=[0]+[dict2599.index(c) for c in TOKENS[1:5]]
 wrapper=AggregatedTeacher(teacher,ids).eval()
 with torch.random.fork_rng():
  torch.manual_seed(7331);head=torch.nn.Linear(140,6)
 rawroot,raw=conditions['raw'];train=[r for r in raw if r['split']=='train'];dev=[r for r in raw if r['split'] in ('development_a','development_b')]
 assert len(train)==30 and len(dev)==12
 real=json.loads((REAL/'audio-restoration-receipt.json').read_text());asr=json.loads((REAL/'asr-screen-results.json').read_text())
 assert real['selected_count']==40 and len(real['recordings'])==40 and asr['admitted_clips']==0 and asr['gate_passed'] is False
 usage=[]
 for row in train:
  usage.append({'recording':row['recording'],'source':'qwen','role':'train-unlabeled-distillation','path':str(rawroot/row['path']),'file_sha256':row['file_sha256'],'pcm_sha256':row['pcm_sha256'],'frames':row['frames'],'transcript_consumed':False})
 for row in real['recordings']:
  assert row['historical_role']=='train' and row['admitted_for_training'] is False
  usage.append({'recording':row['recording'],'source':'hi-mia-cw','role':'historical-project-train-unlabeled-distillation','path':row['audio_path'],'file_sha256':row['file_sha256'],'pcm_sha256':row['pcm_sha256'],'frames':row['frames'],'supervised_admission':False,'transcript_consumed':False})
 assert len(usage)==70 and len({r['recording'] for r in usage})==70
 for row in usage:
  path=pathlib.Path(row['path']);assert b.sha(path)==row['file_sha256']
  with wave.open(str(path)) as w:
   assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,row['frames'])
   assert hashlib.sha256(w.readframes(w.getnframes())).hexdigest()==row['pcm_sha256']
 receipt={'stage':'teacher-reference-only-no-student-training','script_sha256':b.sha(pathlib.Path(__file__)),'proposal_sha256':b.sha(ROOT/'PROPOSAL.md'),'baseline_driver_sha256':b.sha(BASE/'run_baseline.py'),'baseline_spec_sha256':b.sha(BASE/'baseline-spec.json'),'donor_sha256':b.sha(BASE/'resources/base.pt'),'teacher_state_sha256':tensor_hash(teacher.state_dict()),'compact_head_initial_sha256':tensor_hash(head.state_dict()),'compact_head_seed':7331,'real_receipt_sha256':b.sha(REAL/'audio-restoration-receipt.json'),'failed_supervised_screen_sha256':b.sha(REAL/'asr-screen-results.json'),'supervised_admitted_real_clips':0,'unlabeled_usage':usage,'teacher_class_ids':ids,'aggregation':'full2599softmax then preserve blank+four target classes and sum all2594 remaining classes asOTHER','temperature':1,'probability_precision':'FP32 donor logits; float64 fullsoftmax+aggregation; sixq castFP32 for decoder; no target renormalization','decoder_search_ids':list(range(6)),'keyword_ids':{'你好小窝':[1,2,3,4],'小窝小窝':[3,4,3,4]},'decoder':{'threshold':0,'score_beam':3,'path_beam':20,'pruning':.05,'min_frames':5,'max_frames':250,'interval_frames':50,'inherited_suffix_bug':True,'first_hit_per_chunk':True},'reference_gate':{'raw_development_clips':12,'requires_at_least_one_hit_each_word':True,'failure_action':'stop-no-student-training','no_threshold_search':True},'future_student':{'not_executed_by_this_script':True,'steps':100,'optimizer':'AdamW','lr':.001,'weight_decay':0,'gradient_norm_cap':1,'objective':'KL(q_teacher||p_student), meanframesperclip then mean70clips','development_clip_mean_kl_max':.05,'target_class_mae_max':.05,'eligible_teacher_probability_min':.05,'retention':'lose no referencehit developmentclip; no new confusableclip or wrongwordevent'},'runtime':old['runtime']}
 return front,ns,conditions,wrapper,dev,receipt

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(600,600))
 front,ns,conditions,model,dev,receipt=setup()
 if sys.argv[1]=='prepare':
  with torch.no_grad():
   k=make(model,front,ns);k.forward(bytes(9600));assert model.q[0].shape==(1,9,6)
  if (ROOT/'preflight.json').exists():assert receipt==json.loads((ROOT/'preflight.json').read_text())
  else:b.save(ROOT/'preflight.json',receipt)
  print('70 unlabeled identities, initialhead and teacher hashes, synthetic-only smoke ready; no corpus model inference');return
 assert sys.argv[1]=='reference' and receipt==json.loads((ROOT/'preflight.json').read_text())
 dest=ROOT/'teacher-reference.json';assert not dest.exists()
 rows=[];root,_=conditions['raw'];start_cpu=time.process_time();start_wall=time.perf_counter()
 with torch.no_grad():
  for row in dev:
   model.q=[];model.teacher.logs=[];k=make(model,front,ns)
   with wave.open(str(root/row['path'])) as w:pcm=w.readframes(w.getnframes())
   events=[]
   for start in range(0,len(pcm),9600):
    end=min(len(pcm),start+9600);e=k.forward(pcm[start:end])
    if e.get('state')==1:events.append(dict(e,keyword_id=b.PHRASES[e['keyword']],available_audio_samples=end//2,eof_flush=False))
   q=torch.cat(model.q,1)[0];positive=row['kind']=='positive'
   rows.append(dict(row,events=events,target_hit=any(e['keyword_id']==row['keyword_id'] for e in events) if positive else None,probability_shape=list(q.shape),aggregate_q_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest(),aggregate_q=q.tolist()))
 hits={str(word):sum(r['target_hit'] is True for r in rows if r['keyword_id']==word) for word in (1,2)}
 passed=all(n>0 for n in hits.values())
 result={'preflight_sha256':b.sha(ROOT/'preflight.json'),'cpu_s':time.process_time()-start_cpu,'wall_s':time.perf_counter()-start_wall,'reference_hits_by_keyword':hits,'nonvacuity_gate_passed':passed,'student_training_executed':False,'decision':'reference-valid-student-code-review-required' if passed else 'stop-no-student-training','confusable_trigger_clips':sum(bool(r['events']) for r in rows if r['kind']=='confusable'),'recordings':rows,'supervised_admission_remains':0,'limits':'observeddevelopment uncalibrated complete-mass six-way teacher reference; not original2599keyword-only readout or product qualification'}
 b.inputs();b.save(dest,result);print(json.dumps({k:v for k,v in result.items() if k!='recordings'}),flush=True)

if __name__=='__main__':main()
