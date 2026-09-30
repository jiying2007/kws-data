"""Read existing JSON only: no torch, inference, training, or numeric KL invention."""
import hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'student-results'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 teacher=load(ROOT/'teacher-reference.json');reference={r['recording']:r for r in teacher['recordings']}
 assert len(reference)==12
 conditions={};raw=None
 for name in ('raw','tail500ms','head500ms_tail500ms'):
  data=load(OUT/f'{name}.json');rows=data['recordings'];assert len(rows)==42 and len({r['recording'] for r in rows})==42
  measured={'positive_hits':sum(r['target_hit'] is True for r in rows),'positive_clips':sum(r['kind']=='positive' for r in rows),'confusable_with_events':sum(bool(r['events']) for r in rows if r['kind']=='confusable'),'confusable_clips':sum(r['kind']=='confusable' for r in rows),'wrong_keyword_events':sum(e['keyword_id']!=r['keyword_id'] for r in rows if r['kind']=='positive' for e in r['events'])}
  assert measured==data['summary'];conditions[name]=measured
  if name=='raw':raw=rows
 dev=[r for r in raw if r['recording'] in reference];assert len(dev)==12
 errors={i:[] for i in range(1,5)};zero0=zeropositive=0;lost=[];newconf=[];newwrong=[];perclip=[]
 for row in dev:
  before=reference[row['recording']];q=before['aggregate_q'];p=row['probabilities'];assert len(q)==len(p)
  for qq,pp in zip(q,p):
   assert len(qq)==len(pp)==6 and all(math.isfinite(x) and x>=0 for x in qq+pp)
   assert abs(sum(qq)-1)<1e-6 and abs(sum(pp)-1)<1e-6
   for x,y in zip(qq,pp):
    if y==0:
     if x==0:zero0+=1
     else:zeropositive+=1
   for i in range(1,5):
    if qq[i]>=.05:errors[i].append(abs(qq[i]-pp[i]))
  if before['target_hit'] is True and row['target_hit'] is not True:lost.append(row['recording'])
  if row['kind']=='confusable' and row['events'] and not before['events']:newconf.append(row['recording'])
  if row['kind']=='positive' and sum(e['keyword_id']!=row['keyword_id'] for e in row['events'])>sum(e['keyword_id']!=row['keyword_id'] for e in before['events']):newwrong.append(row['recording'])
  perclip.append({'recording':row['recording'],'speaker_id':row['speaker_id'],'kind':row['kind'],'keyword_id':row['keyword_id'],'teacher_hit':before['target_hit'],'student_hit':row['target_hit'],'events':row['events']})
 stats={str(i):{'eligible_frames':len(values),'mae':sum(values)/len(values),'limit':.05,'passed':sum(values)/len(values)<=.05} for i,values in errors.items()}
 assert all(not s['passed'] for s in stats.values())
 report={'scope':'failed fixed100update compacthead distillation; read-only saved-output audit','training_updates_completed':100,'update_count_evidence':'checkpoint write and all evaluation outputs occur only after step0..100 loop with100 optimizer.step calls in immutable approved driver; no second run','checkpoint_sha256':sha(OUT/'compact-final.pt'),'checkpoint_bytes':(OUT/'compact-final.pt').stat().st_size,'student_script_sha256':sha(ROOT/'student.py'),'student_preflight_sha256':sha(ROOT/'student-preflight.json'),'teacher_reference_sha256':sha(ROOT/'teacher-reference.json'),'failure_log_sha256':sha(ROOT/'student-run.log'),'readback_script_sha256':sha(pathlib.Path(__file__)),'conditions':conditions,'development':{'positive_clips':6,'confusable_clips':6,'teacher_positive_hits':sum(r['target_hit'] is True for r in reference.values()),'student_positive_hits':sum(r['target_hit'] is True for r in dev),'all_hit_speakers':sorted({r['speaker_id'] for r in dev if r['target_hit'] is True}),'lost_teacher_hit_clips':lost,'new_confusable_trigger_clips':newconf,'new_wrong_word_clips':newwrong,'retention_passed':not(lost or newconf or newwrong),'target_probability_mae':stats,'ideal_softmax_kl':None,'ideal_softmax_kl_status':'not reliably recoverable from saved rounded probabilities; original logits not retained, no inference replay performed','literal_rounded_probability_kl':'infinite because oneq>0,p=0 term','student_zero_probability_terms':zero0+zeropositive,'q0_p0_terms':zero0,'qpositive_p0_terms':zeropositive,'fidelity_passed':False,'perclip':perclip},'compression_diagnostic_passed':False,'decision':'stop; no extra updates, threshold changes, C model export or candidate promotion','training_curve':None,'exact_training_cpu_and_wall':None,'missing_telemetry_reason':'original script wrote these only after final reporting; assertion aborted before persistence; no reconstruction or invented values','stable_kl_fix':'repair_statistics.py helper masks q=0 and uses float64log_softmax;4synthetic tests pass; replay main was NOT executed','supervised_admission_real_clips':0,'qualification_allowed':False,'future_causal_explanations':'limitedheadcapacity, optimization budget or teacher/decoder effects remain hypotheses, not established causes'}
 (OUT/'failure-readback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('development',)},ensure_ascii=False))
if __name__=='__main__':main()
