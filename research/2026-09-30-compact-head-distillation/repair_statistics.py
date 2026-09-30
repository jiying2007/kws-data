"""Read-only replay of existing final checkpoint; no optimizer or model updates."""
import hashlib, importlib.util, json, pathlib, resource, time, wave
ROOT=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('ref',ROOT/'reference_gate.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
b=r.b;torch=r.torch

def stable_kl(q,logits):
 assert q.shape==logits.shape and torch.isfinite(q).all() and (q>=0).all() and torch.isfinite(logits).all()
 assert torch.allclose(q.sum(-1),torch.ones_like(q[:,0]),atol=1e-6,rtol=0)
 q=q.double();logp=logits.double().log_softmax(-1);terms=torch.zeros_like(q);mask=q>0
 terms[mask]=q[mask]*(q[mask].log()-logp[mask])
 value=terms.sum(-1);assert torch.isfinite(value).all();return value

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(600,600))
 out=ROOT/'student-results';checkpoint=out/'compact-final.pt';expected='b61fed9ef590a9c922ee2a6ddc6584b3f29330c45bd4c07d386db459a5514bba';assert b.sha(checkpoint)==expected
 front,ns,_,conditions,_=b.setup();model=b.Donor().eval();model.backbone.out_linear2.linear=torch.nn.Linear(140,6);model.backbone.output_dim=6;model.backbone.out_linear2.output_dim=6
 model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'),strict=True)
 for p in model.parameters():p.requires_grad_(False)
 state_before=r.tensor_hash(model.state_dict());original=json.loads((out/'raw.json').read_text());student={x['recording']:x for x in original['recordings']};teacher={x['recording']:x for x in json.loads((ROOT/'teacher-reference.json').read_text())['recordings']}
 perclip=[];err={i:[] for i in range(6)};eligible={i:[] for i in range(1,5)};zeros={'p0_q0':0,'p0_qpositive':0};lost=[];newconf=[];newwrong=[];root,rows=conditions['raw'];start=time.process_time()
 with torch.no_grad():
  for row in rows:
   if row['recording'] not in teacher:continue
   model.logs=[];k=r.make(model,front,ns)
   with wave.open(str(root/row['path'])) as w:pcm=w.readframes(w.getnframes())
   events=[]
   for startbyte in range(0,len(pcm),9600):
    end=min(len(pcm),startbyte+9600);e=k.forward(pcm[startbyte:end])
    if e.get('state')==1:events.append(dict(e,keyword_id=b.PHRASES[e['keyword']],available_audio_samples=end//2,eof_flush=False))
   logits=torch.cat(model.logs,1)[0];before=student[row['recording']];q=torch.tensor(teacher[row['recording']]['aggregate_q']);p=logits.softmax(-1)
   assert hashlib.sha256(logits.numpy().tobytes()).hexdigest()==before['logits_sha256']
   assert events==before['events'] and torch.equal(p,torch.tensor(before['probabilities']))
   kl=stable_kl(q,logits);error=(p-q).abs()
   z0=int(((p==0)&(q==0)).sum());zp=int(((p==0)&(q>0)).sum());zeros['p0_q0']+=z0;zeros['p0_qpositive']+=zp
   perclip.append({'recording':row['recording'],'ideal_softmax_mean_kl':float(kl.mean()),'p95_abs_probability_error':float(torch.quantile(error.flatten(),.95)),'frame_count':len(q),'unchanged_logits_events_probabilities':True,'p0_q0_terms':z0,'p0_qpositive_terms':zp})
   for i in range(6):err[i].extend(error[:,i].tolist())
   for i in range(1,5):eligible[i].extend(error[q[:,i]>=.05,i].tolist())
   ref=teacher[row['recording']]
   if ref['target_hit'] is True and before['target_hit'] is not True:lost.append(row['recording'])
   if row['kind']=='confusable' and events and not ref['events']:newconf.append(row['recording'])
   if row['kind']=='positive' and sum(e['keyword_id']!=row['keyword_id'] for e in events)>sum(e['keyword_id']!=row['keyword_id'] for e in ref['events']):newwrong.append(row['recording'])
 assert len(perclip)==12 and r.tensor_hash(model.state_dict())==state_before and b.sha(checkpoint)==expected
 mean_kl=sum(x['ideal_softmax_mean_kl'] for x in perclip)/12;stats={str(i):{'eligible_frames':len(x),'mae':sum(x)/len(x) if x else None} for i,x in eligible.items()}
 fidelity=mean_kl<=.05 and all(x['eligible_frames']>0 and x['mae']<=.05 for x in stats.values());retention=not(lost or newconf or newwrong)
 report={'repair_script_sha256':b.sha(pathlib.Path(__file__)),'original_script_sha256':b.sha(ROOT/'student.py'),'original_preflight_sha256':b.sha(ROOT/'student-preflight.json'),'original_log_sha256':b.sha(ROOT/'student-run.log'),'checkpoint_sha256_before_after':expected,'original_raw_result_sha256':b.sha(out/'raw.json'),'teacher_reference_sha256':b.sha(ROOT/'teacher-reference.json'),'model_state_sha256_before_after':state_before,'model_updates':0,'optimizer_present':False,'replayed_clips':12,'cpu_s_statistics_only':time.process_time()-start,'original_failure_preserved':'FP32 probabilities rounded to0 then p.log caused nonfinite kl_div; original script/results/log unchanged','kl_definition':'KL(q_teacher||softmax(student_FP32_logits)) evaluated stably via float64 log_softmax; q is frozen reference FP32; q0 terms explicitly0','rounded_distribution_note':'Actual saved FP32 p has zeros; whereq>0,p0 its literal discrete KL is infinite. Stable ideal-softmax KL is different from KL of rounded probabilities, and both facts are reported.','rounded_probability_zero_terms':zeros,'ideal_softmax_dev_mean_clip_kl':mean_kl,'target_mae_unchanged_fp32':stats,'all_class_mae_unchanged_fp32':{str(i):sum(v)/len(v) for i,v in err.items()},'perclip':perclip,'fidelity_passed':fidelity,'retention_passed':retention,'lost_teacher_hit_clips':lost,'new_confusable_trigger_clips':newconf,'new_wrong_word_clips':newwrong,'compression_diagnostic_passed':fidelity and retention,'training_curve_and_exact_training_cpu':'not persisted before original reporting assertion; cannot reconstruct execution history from checkpoint; no rerun','supervised_admission_remains':0,'decision':'stop-no-additional-updates-or-tuning' if not(fidelity and retention) else 'diagnostic-only-no-product-qualification'}
 b.inputs();dest=out/'statistics-repair.json';assert not dest.exists();b.save(dest,report);print(json.dumps({k:v for k,v in report.items() if k!='perclip'}),flush=True)

if __name__=='__main__':main()
