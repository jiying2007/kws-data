"""Single fixed head-only distillation; requires approved stage-two preflight."""
import copy, hashlib, importlib.util, json, pathlib, resource, sys, time, wave
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ref',ROOT/'reference_gate.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
b=r.b;torch=r.torch

def new_head():
 with torch.random.fork_rng():
  torch.manual_seed(7331);return torch.nn.Linear(140,6)

def kl_clip(logits,q):
 assert logits.shape==q.shape and logits.shape[-1]==6 and torch.isfinite(logits).all() and torch.isfinite(q).all()
 assert torch.allclose(q.sum(-1),torch.ones_like(q[:,0]),atol=1e-6,rtol=0)
 loss=torch.nn.functional.kl_div(logits.log_softmax(-1),q,reduction='none').sum(-1).mean()
 assert torch.isfinite(loss);return loss

def prepare():
 front,ns,conditions,teacher,dev,old=r.setup()
 assert old==json.loads((ROOT/'preflight.json').read_text())
 reference=json.loads((ROOT/'teacher-reference.json').read_text());assert reference['nonvacuity_gate_passed'] is True and reference['student_training_executed'] is False
 assert reference['preflight_sha256']==b.sha(ROOT/'preflight.json') and all(reference['reference_hits_by_keyword'][str(k)]>=1 for k in (1,2))
 head=new_head();assert r.tensor_hash(head.state_dict())==old['compact_head_initial_sha256']
 new={'script_sha256':b.sha(pathlib.Path(__file__)),'reference_driver_sha256':b.sha(ROOT/'reference_gate.py'),'reference_preflight_sha256':b.sha(ROOT/'preflight.json'),'teacher_reference_sha256':b.sha(ROOT/'teacher-reference.json'),'initial_head_sha256':r.tensor_hash(head.state_dict()),'teacher_state_sha256':r.tensor_hash(teacher.teacher.state_dict()),'unlabeled_usage_sha256':hashlib.sha256(json.dumps(old['unlabeled_usage'],sort_keys=True).encode()).hexdigest(),'supervised_admitted_real_clips':0,'objective':'KL(q_teacher||p_student); mean over each clips frames, then equal mean70clips','head_seed':7331,'optimizer':'AdamW','lr':.001,'weight_decay':0,'steps':100,'grad_norm_cap':1,'temperature':1,'cpu_threads':1,'hard_process_cpu_seconds':600,'criteria':{'requires':'BOTH fidelity AND retention; failure of EITHER stops','dev_clip_mean_kl_max':.05,'target_mae_max':.05,'target_eligible_teacher_q_min':.05,'each_target_requires_eligible_frames':True,'no_lost_reference_hit_clips':True,'no_new_confusable_trigger_clips':True,'no_new_wrong_keyword_events':True},'training_labels_consumed':False,'evaluation_scope':'12nativeQwen observeddevelopment primary; all42x3 reported; not qualification'}
 return front,ns,conditions,teacher,head,old,reference,new

def read_pcm(row):
 p=pathlib.Path(row['path']);assert b.sha(p)==row['file_sha256']
 with wave.open(str(p)) as w:
  pcm=w.readframes(w.getnframes());assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256'];return pcm

def extract(row,teacher,front,ns):
 pcm=read_pcm(row);teacher.q=[];teacher.teacher.logs=[];features=[];k=r.make(teacher,front,ns)
 def capture(module,args):
  x=args[0][0] if isinstance(args[0],tuple) else args[0];features.append(x.detach().clone())
 hook=teacher.teacher.backbone.out_linear2.register_forward_pre_hook(capture)
 with torch.no_grad():
  for start in range(0,len(pcm),9600):
   x=k.accept_wave(pcm[start:start+9600])
   if x is not None:_,k.in_cache=teacher(x.unsqueeze(0),k.in_cache)
 hook.remove();x=torch.cat(features,1)[0];q=torch.cat(teacher.q,1)[0];assert x.shape[0]==q.shape[0] and x.shape[1]==140
 return x,q

def run():
 resource.setrlimit(resource.RLIMIT_CPU,(600,600))
 front,ns,conditions,teacher,head,old,reference,new=prepare();assert new==json.loads((ROOT/'student-preflight.json').read_text())
 out=ROOT/'student-results';out.mkdir(exist_ok=False);startc=time.process_time();startw=time.perf_counter();frozen=r.tensor_hash(teacher.teacher.state_dict())
 def watchdog():
  if time.process_time()>599:raise RuntimeError('CPU cap; incomplete experiment, no early-checkpoint selection')
 pairs=[];cache=[]
 for row in old['unlabeled_usage']:
  watchdog();x,q=extract(row,teacher,front,ns);pairs.append((x,q));cache.append({'recording':row['recording'],'frames':len(x),'feature_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),'teacher_q_sha256':hashlib.sha256(q.numpy().tobytes()).hexdigest()})
 extractc=time.process_time()-startc;opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0);curve=[];gradnorms=[];trainc=time.process_time();trainw=time.perf_counter()
 for step in range(101):
  watchdog();opt.zero_grad(set_to_none=True);loss=torch.stack([kl_clip(head(x),q) for x,q in pairs]).mean();curve.append(float(loss.detach()))
  if step==100:break
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1);assert torch.isfinite(norm);gradnorms.append(float(norm));opt.step()
  assert all(torch.isfinite(p).all() for p in head.parameters())
 traincpu=time.process_time()-trainc;trainwall=time.perf_counter()-trainw
 assert r.tensor_hash(teacher.teacher.state_dict())==frozen
 # Export a self-contained compact model state without altering the teacher.
 teacher.teacher.logs=[];student=copy.deepcopy(teacher.teacher);student.backbone.out_linear2.linear=copy.deepcopy(head);student.backbone.output_dim=6;student.backbone.out_linear2.output_dim=6;student.logs=[];student.eval()
 checkpoint=out/'compact-final.pt';torch.save(student.state_dict(),checkpoint)
 results={};primary_rows=[]
 with torch.no_grad():
  for name,(root,rows) in conditions.items():
   records=[]
   for row in rows:
    watchdog();student.logs=[];k=r.make(student,front,ns)
    with wave.open(str(root/row['path'])) as w:pcm=w.readframes(w.getnframes())
    events=[]
    for start in range(0,len(pcm),9600):
     end=min(len(pcm),start+9600);event=k.forward(pcm[start:end])
     if event.get('state')==1:events.append(dict(event,keyword_id=b.PHRASES[event['keyword']],available_audio_samples=end//2,eof_flush=False))
    logits=torch.cat(student.logs,1)[0];prob=logits.softmax(-1);positive=row['kind']=='positive'
    record=dict(row,events=events,target_hit=any(e['keyword_id']==row['keyword_id'] for e in events) if positive else None,probabilities=prob.tolist(),logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest())
    records.append(record)
    if name=='raw' and row['split'] in ('development_a','development_b'):primary_rows.append(record)
   summary={'positive_hits':sum(x['target_hit'] is True for x in records),'positive_clips':20,'confusable_with_events':sum(bool(x['events']) for x in records if x['kind']=='confusable'),'confusable_clips':22,'wrong_keyword_events':sum(e['keyword_id']!=x['keyword_id'] for x in records if x['kind']=='positive' for e in x['events'])}
   b.save(out/f'{name}.json',{'summary':summary,'recordings':records});results[name]=summary
   print(json.dumps({'condition':name,**summary}),flush=True)
 assert len(primary_rows)==12;ref={x['recording']:x for x in reference['recordings']};perclip=[];target_errors={i:[] for i in range(1,5)};all_errors={i:[] for i in range(6)}
 lost=[];newconf=[];newwrong=[]
 for row in primary_rows:
  original=ref[row['recording']];p=torch.tensor(row['probabilities']);q=torch.tensor(original['aggregate_q']);assert p.shape==q.shape
  kl=torch.nn.functional.kl_div(p.log(),q,reduction='none').sum(-1);assert torch.isfinite(kl).all()
  error=(p-q).abs();perclip.append({'recording':row['recording'],'mean_kl':float(kl.mean()),'p95_abs_error':float(torch.quantile(error.flatten(),.95))})
  for i in range(6):all_errors[i].extend(error[:,i].tolist())
  for i in range(1,5):target_errors[i].extend(error[q[:,i]>=.05,i].tolist())
  if original['target_hit'] is True and row['target_hit'] is not True:lost.append(row['recording'])
  if row['kind']=='confusable' and row['events'] and not original['events']:newconf.append(row['recording'])
  if row['kind']=='positive':
   oldwrong=sum(e['keyword_id']!=row['keyword_id'] for e in original['events']);wrong=sum(e['keyword_id']!=row['keyword_id'] for e in row['events'])
   if wrong>oldwrong:newwrong.append(row['recording'])
 mean_kl=sum(x['mean_kl'] for x in perclip)/12
 target_stats={str(i):{'eligible_frames':len(es),'mae':sum(es)/len(es) if es else None} for i,es in target_errors.items()}
 fidelity=mean_kl<=.05 and all(x['eligible_frames']>0 and x['mae']<=.05 for x in target_stats.values());retention=not (lost or newconf or newwrong)
 report={'preflight_sha256':b.sha(ROOT/'student-preflight.json'),'training_updates':100,'training_clip_count':70,'supervised_admission_remains':0,'training_clip_equal_kl_curve':curve,'gradient_norms_before_clipping':gradnorms,'extraction_cpu_s':extractc,'training_cpu_s':traincpu,'training_wall_s':trainwall,'total_cpu_s':time.process_time()-startc,'total_wall_s':time.perf_counter()-startw,'checkpoint_sha256':b.sha(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'parameters':sum(p.numel() for p in student.parameters()),'fp32_parameter_bytes':sum(p.numel()*p.element_size() for p in student.parameters()),'initial_head_sha256':new['initial_head_sha256'],'final_head_sha256':r.tensor_hash(head.state_dict()),'unchanged_teacher_sha256':frozen,'cache':cache,'conditions':results,'development_fidelity':{'mean_clip_kl':mean_kl,'targets':target_stats,'all_class_mae':{str(i):sum(es)/len(es) for i,es in all_errors.items()},'perclip':perclip,'passed':fidelity},'development_retention':{'lost_teacher_hit_clips':lost,'new_confusable_trigger_clips':newconf,'new_wrong_word_clips':newwrong,'passed':retention},'compression_diagnostic_pass':fidelity and retention,'failure_action':'stop; no extra steps/tuning/replacement','qualification_allowed':False}
 b.inputs()
 for row in old['unlabeled_usage']:assert b.sha(pathlib.Path(row['path']))==row['file_sha256']
 watchdog();b.save(out/'summary.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('cache','training_clip_equal_kl_curve','gradient_norms_before_clipping')}),flush=True)

if __name__=='__main__':
 if sys.argv[1]=='prepare':
  *_,receipt=prepare()
  if (ROOT/'student-preflight.json').exists():assert receipt==json.loads((ROOT/'student-preflight.json').read_text())
  else:b.save(ROOT/'student-preflight.json',receipt)
  print('student fixed protocol/head/input identities prepared; no training or corpus model inference')
 elif sys.argv[1]=='run':run()
 else:raise ValueError('prepare|run')
