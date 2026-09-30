"""Offline integrity/semantic checks; never imports torch or runs inference."""
import gzip,hashlib,importlib.util,io,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
EXPECTED_CHECKPOINT='b61fed9ef590a9c922ee2a6ddc6584b3f29330c45bd4c07d386db459a5514bba'
EXPECTED_DONOR='d02b09c34f4a8bbb06f0dd1bf5eb58db3395eb7f1fd15c3625fe09d3a2492233'
REQUIRED_FILES=frozenset(('compression-manifest.json', 'PROPOSAL.md', 'README.md', 'RESULTS.md', 'copied-file-origins.json', 'donor-reference.json', 'export_head.py', 'head-float32.json', 'historical-source/baseline-spec.json', 'historical-source/download-receipt.json', 'historical-source/run_baseline.py', 'historical-source/upstream-source-manifest.json', 'preflight.json', 'rebuild-verification.json', 'rebuild.py', 'reconstruction-proof.json', 'reference-run.log', 'reference_gate.py', 'repair_statistics.py', 'student-preflight.json', 'student-results/failure-readback.json', 'student-results/head500ms_tail500ms.json.gz', 'student-results/raw.json.gz', 'student-results/tail500ms.json.gz', 'student-run.log', 'student.py', 'summarize_failure.py', 'teacher-reference.json.gz', 'test_aggregate.py', 'test_repair_statistics.py', 'test_student.py', 'test_verify.py', 'tests.log', 'verify.py'))

COPIED_FILES=frozenset(('PROPOSAL.md', 'RESULTS.md', 'preflight.json', 'reference-run.log', 'reference_gate.py', 'repair_statistics.py', 'student-preflight.json', 'student-results/failure-readback.json', 'student-results/head500ms_tail500ms.json', 'student-results/raw.json', 'student-results/tail500ms.json', 'student-run.log', 'student.py', 'summarize_failure.py', 'teacher-reference.json', 'test_aggregate.py', 'test_repair_statistics.py', 'test_student.py', 'tests.log', 'historical-source/run_baseline.py', 'historical-source/baseline-spec.json', 'historical-source/upstream-source-manifest.json', 'historical-source/download-receipt.json'))

COMPRESSED_LOGICAL=frozenset(('student-results/raw.json','student-results/tail500ms.json','student-results/head500ms_tail500ms.json','teacher-reference.json'))
MAX_EXPANDED=1024*1024
MAX_COMPRESSED=512*1024
MAX_TOTAL_EXPANDED=3*1024*1024

def logical_bytes(p):
 if p.exists():return p.read_bytes()
 packed=p.with_name(p.name+'.gz')
 if packed.stat().st_size>MAX_COMPRESSED:raise ValueError('Compressed size limit')
 with gzip.GzipFile(fileobj=io.BytesIO(packed.read_bytes())) as stream:
  data=stream.read(MAX_EXPANDED+1)
 if len(data)>MAX_EXPANDED:raise ValueError('Expanded size limit')
 return data

def sha(p):return hashlib.sha256(logical_bytes(p)).hexdigest()
def load(p):return json.loads(logical_bytes(p).decode('utf-8'))

def near(a,b):
 if type(a) not in (int,float) or not math.isfinite(a) or not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9):raise ValueError('Outcome arithmetic mismatch')
def verify(root=ROOT):
 root=pathlib.Path(root)
 if root.is_symlink():raise ValueError('Root symlink')
 root=root.resolve();m=load(root/'archive-manifest.json')
 if type(m.get('schema_version')) is not int or m['schema_version']!=1 or m.get('purpose')!='failed-compact-head-distillation-head-only':raise ValueError('Manifest schema/purpose')
 seen=set()
 for item in m['files']:
  rel=pathlib.Path(item['path'])
  if rel.is_absolute() or '..' in rel.parts or item['path'] in seen:raise ValueError('Unsafe/duplicate path')
  p=root/rel
  if not p.resolve().is_relative_to(root) or any((root/pathlib.Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):raise ValueError('Symlink/containment')
  if p.is_file() and p.stat().st_size>(MAX_COMPRESSED if p.suffix=='.gz' else MAX_EXPANDED):raise ValueError('Stored size limit')
  if type(item['bytes']) is not int or item['bytes']<0 or not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('File identity mismatch')
  if p.suffix!='.gz':p.read_text(encoding='utf-8')
  seen.add(item['path'])
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}-{'archive-manifest.json'}
 if seen!=REQUIRED_FILES or actual!=REQUIRED_FILES:raise ValueError('Mandatory inventory mismatch')
 compression=load(root/'compression-manifest.json')
 if type(compression.get('schema_version')) is not int or compression['schema_version']!=1 or compression.get('format')!='gzip' or compression.get('purpose')!='lossless-original-json-storage':raise ValueError('Compression schema')
 rows=compression['files']
 if len(rows)!=4 or {x['logical_path'] for x in rows}!=COMPRESSED_LOGICAL:raise ValueError('Compression inventory')
 total=0
 for item in rows:
  name=item['logical_path']
  if item['stored_path']!=name+'.gz' or type(item['expanded_bytes']) is not int or not 0<item['expanded_bytes']<=MAX_EXPANDED or type(item['compressed_bytes']) is not int or not 0<item['compressed_bytes']<=MAX_COMPRESSED:raise ValueError('Compression bounds/path')
  packed=root/item['stored_path'];data=logical_bytes(root/name)
  if packed.stat().st_size!=item['compressed_bytes'] or sha(packed)!=item['compressed_sha256'] or len(data)!=item['expanded_bytes'] or hashlib.sha256(data).hexdigest()!=item['expanded_sha256']:raise ValueError('Expanded/compressed identity')
  json.loads(data.decode('utf-8'));total+=len(data)
 if total>MAX_TOTAL_EXPANDED:raise ValueError('Total expansion limit')
 s=importlib.util.spec_from_file_location('head_rebuild',root/'rebuild.py');module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
 head=module.read_head(root/'head-float32.json');proof=load(root/'reconstruction-proof.json');donor=load(root/'donor-reference.json');failure=load(root/'student-results/failure-readback.json')
 if head['original_full_checkpoint_sha256']!=EXPECTED_CHECKPOINT or proof['original_full_checkpoint_sha256']!=EXPECTED_CHECKPOINT or failure['checkpoint_sha256']!=EXPECTED_CHECKPOINT:raise ValueError('Original checkpoint identity')
 if head['donor_checkpoint_sha256']!=EXPECTED_DONOR or proof['donor_checkpoint_sha256']!=EXPECTED_DONOR or donor['weight']['sha256']!=EXPECTED_DONOR or donor['weight']['included_in_archive'] is not False:raise ValueError('Donor identity')
 if proof['all_non_head_tensors_equal_donor'] is not True or proof['head_float32_roundtrip_exact'] is not True:raise ValueError('Reconstruction proof failed')
 for name,tensor in head['tensors'].items():
  if any(proof['tensors'][name][k]!=tensor[k] for k in ('shape','dtype','data_sha256')):raise ValueError('Head/proof mismatch')
 original_records=load(root/'copied-file-origins.json')
 if len(original_records)!=len(COPIED_FILES) or {x['path'] for x in original_records}!=COPIED_FILES:raise ValueError('Copied source coverage')
 for original in original_records:
  p=root/original['path']
  if sha(p)!=original['source_sha256'] or len(logical_bytes(p))!=original['source_bytes']:raise ValueError('Historical source drift')
 pre=load(root/'preflight.json');studentpre=load(root/'student-preflight.json')
 bindings=[('reference_gate.py',pre['script_sha256']),('PROPOSAL.md',pre['proposal_sha256']),('historical-source/run_baseline.py',pre['baseline_driver_sha256']),('historical-source/baseline-spec.json',pre['baseline_spec_sha256']),('student.py',studentpre['script_sha256']),('reference_gate.py',studentpre['reference_driver_sha256']),('preflight.json',studentpre['reference_preflight_sha256']),('teacher-reference.json',studentpre['teacher_reference_sha256']),('student.py',failure['student_script_sha256']),('student-preflight.json',failure['student_preflight_sha256']),('teacher-reference.json',failure['teacher_reference_sha256']),('student-run.log',failure['failure_log_sha256']),('summarize_failure.py',failure['readback_script_sha256'])]
 if any(sha(root/name)!=h for name,h in bindings):raise ValueError('Experiment provenance binding')
 if pre['supervised_admitted_real_clips']!=0 or studentpre['supervised_admitted_real_clips']!=0 or len(pre['unlabeled_usage'])!=70 or len({x['recording'] for x in pre['unlabeled_usage']})!=70:raise ValueError('Unlabeled scope')
 if sum(x['source']=='qwen' for x in pre['unlabeled_usage'])!=30 or sum(x['source']=='hi-mia-cw' for x in pre['unlabeled_usage'])!=40 or any(x['transcript_consumed'] is not False for x in pre['unlabeled_usage']):raise ValueError('Unlabeled role drift')
 if sha(root/'historical-source/run_baseline.py')!=load(root/'historical-source/baseline-spec.json')['driver_sha256']:raise ValueError('Baseline driver provenance')
 raw=load(root/'student-results/raw.json')['recordings'];ref=load(root/'teacher-reference.json')['recordings'];lookup={x['recording']:x for x in ref};dev=[x for x in raw if x['recording'] in lookup]
 if len(raw)!=42 or len(dev)!=12 or len(lookup)!=12:raise ValueError('Coverage')
 errors={i:[] for i in range(1,5)};zero0=zerop=0;lost=[];newconf=[];newwrong=[]
 for row in dev:
  old=lookup[row['recording']]
  if old['target_hit'] is True and row['target_hit'] is not True:lost.append(row['recording'])
  if row['kind']=='confusable' and row['events'] and not old['events']:newconf.append(row['recording'])
  if row['kind']=='positive' and sum(e['keyword_id']!=row['keyword_id'] for e in row['events'])>sum(e['keyword_id']!=row['keyword_id'] for e in old['events']):newwrong.append(row['recording'])
  if len(row['probabilities'])!=len(old['aggregate_q']):raise ValueError('Frame alignment')
  for p,q in zip(row['probabilities'],old['aggregate_q'],strict=True):
   if len(p)!=6 or len(q)!=6 or not all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in p+q) or abs(sum(p)-1)>1e-6 or abs(sum(q)-1)>1e-6:raise ValueError('Probability data')
   for pv,qv in zip(p,q):
    if pv==0:
     if qv==0:zero0+=1
     else:zerop+=1
   for i in range(1,5):
    if q[i]>=.05:errors[i].append(abs(p[i]-q[i]))
 for i,values in errors.items():
  recorded=failure['development']['target_probability_mae'][str(i)];near(recorded['eligible_frames'],len(values));near(recorded['mae'],sum(values)/len(values))
  if recorded['passed'] is not False or recorded['mae']<=.05:raise ValueError('Changed failed fidelity')
 if zero0!=269 or zerop!=1 or lost or failure['compression_diagnostic_passed'] is not False or failure['development']['ideal_softmax_kl'] is not None or failure['training_curve'] is not None or failure['exact_training_cpu_and_wall'] is not None:raise ValueError('Changed failure/missing-data boundary')
 teacherdata=load(root/'teacher-reference.json')
 def check_hit(row):
  expected=any(e['keyword_id']==row['keyword_id'] for e in row['events']) if row['kind']=='positive' else None
  if row['target_hit'] is not expected:raise ValueError('Target-hit contradicts events')
  for e in row['events']:
   if type(e['keyword_id']) is not int or e['keyword_id'] not in (1,2) or e['keyword']!={1:'你好小窝',2:'小窝小窝'}[e['keyword_id']]:raise ValueError('Event keyword binding')
 for row in ref:check_hit(row)
 refhits={str(i):sum(x['target_hit'] is True and x['keyword_id']==i for x in ref) for i in (1,2)}
 if teacherdata['reference_hits_by_keyword']!=refhits or refhits!={'1':2,'2':1} or teacherdata['nonvacuity_gate_passed'] is not True or teacherdata['student_training_executed'] is not False or teacherdata['confusable_trigger_clips']!=sum(bool(x['events']) for x in ref if x['kind']=='confusable'):raise ValueError('Teacher gate contradiction')
 for condition,hits in [('raw',9),('tail500ms',10),('head500ms_tail500ms',9)]:
  d=load(root/f'student-results/{condition}.json');rows=d['recordings']
  if len(rows)!=42 or len({x['recording'] for x in rows})!=42:raise ValueError('Observed coverage mismatch')
  for row in rows:check_hit(row)
  calculated={'positive_hits':sum(x['target_hit'] is True for x in rows),'positive_clips':sum(x['kind']=='positive' for x in rows),'confusable_with_events':sum(bool(x['events']) for x in rows if x['kind']=='confusable'),'confusable_clips':sum(x['kind']=='confusable' for x in rows),'wrong_keyword_events':sum(e['keyword_id']!=x['keyword_id'] for x in rows if x['kind']=='positive' for e in x['events'])}
  if d['summary']!=calculated or failure['conditions'][condition]!=calculated or calculated!={'positive_hits':hits,'positive_clips':20,'confusable_with_events':0,'confusable_clips':22,'wrong_keyword_events':0}:raise ValueError('Observed outcome mismatch')
 development=failure['development']
 expected_dev={'positive_clips':6,'confusable_clips':6,'teacher_positive_hits':sum(x['target_hit'] is True for x in ref),'student_positive_hits':sum(x['target_hit'] is True for x in dev),'all_hit_speakers':sorted({x['speaker_id'] for x in dev if x['target_hit'] is True}),'lost_teacher_hit_clips':lost,'new_confusable_trigger_clips':newconf,'new_wrong_word_clips':newwrong,'retention_passed':not(lost or newconf or newwrong),'student_zero_probability_terms':zero0+zerop,'q0_p0_terms':zero0,'qpositive_p0_terms':zerop,'fidelity_passed':False}
 if any(development[k]!=value or (isinstance(value,bool) and development[k] is not value) for k,value in expected_dev.items()) or failure['qualification_allowed'] is not False or failure['supervised_admission_real_clips']!=0 or failure['training_updates_completed']!=100:raise ValueError('Failure flags/counters contradiction')
 return {'verified':True,'files':len(seen),'head_parameters':846,'full_weights_included':False,'candidate_failed':True,'inference_executed':False}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
