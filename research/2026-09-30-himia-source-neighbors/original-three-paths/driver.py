"""Fixed source-labelled clip readback. Worker imports models only after binding checks."""
import hashlib,importlib.util,json,os,pathlib,resource,signal,subprocess,sys,time,wave,math
ROOT=pathlib.Path(__file__).resolve().parent
C=pathlib.Path('/workspace/shared/kws-data-consumer');B=C/'build/observed-readback';S=pathlib.Path('/workspace/shared/kws-sherpa-int8');F=pathlib.Path('/workspace/shared/kws-cfsmn-baseline');DATA=pathlib.Path('/workspace/shared/kws-data-pinned')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def heartbeat(p,v):
 tmp=p.with_suffix('.tmp');dump(tmp,v);tmp.replace(p)
def pcm(row):
 p=pathlib.Path(row['path']);assert sha(p)==row['file_sha256']
 with wave.open(str(p)) as w:
  assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,row['frames'])
  b=w.readframes(w.getnframes()+1)
 assert len(b)==2*row['frames'] and hashlib.sha256(b).hexdigest()==row['pcm_sha256'];return b

def checked_events(events):
 json.dumps(events,allow_nan=False)
 for e in events:
  if type(e['keyword_id']) is not int or e['keyword_id'] not in (1,2):raise ValueError('unknown keyword')
 return events

def make_engine(route):
 if route=='c':
  def run(row,b):
   m=C/'models/registry/model-749187ec1d66'
   p=subprocess.run([str(B/'kws_wav'),str(m/'xiaowo-model.kwm'),str(m/'xiaowo-keywords.kwk'),row['path'],row['recording']],capture_output=True,text=True,check=True,timeout=9)
   events=[json.loads(x) for x in p.stdout.splitlines() if x.strip()]
   assert all(e.get('recording')==row['recording'] for e in events)
   return checked_events(events)
  return run
 if route=='sherpa':
  import numpy as np,sherpa_onnx
  assert sherpa_onnx.__version__=='1.13.8' and np.__version__=='2.2.6'
  cfg=json.loads((S/'config.json').read_text());kw={k:cfg[k] for k in ['num_threads','sample_rate','feature_dim','max_active_paths','keywords_score','keywords_threshold','num_trailing_blanks','provider']}
  for x in ['encoder','decoder','joiner']:kw[x]=str(S/'models'/f'{x}-epoch-12-avg-2-chunk-16-left-64.onnx')
  kw['encoder']=str(S/'models'/cfg['encoder_file']);kw.update(tokens=str(S/'models/tokens.txt'),keywords_file=str(S/'keywords.txt'));model=sherpa_onnx.KeywordSpotter(**kw)
  def run(row,b):
   samples=np.frombuffer(b,dtype='<i2').astype(np.float32)/32768.;stream=model.create_stream();ev=[]
   def drain(fed,eof):
    while model.is_ready(stream):
     model.decode_stream(stream);word=model.get_result(stream)
     if word:
      ev.append(dict(keyword_id={'你好小窝':1,'小窝小窝':2}[word],phrase=word,available_audio_samples=fed,eof_flush=eof,tokens=model.tokens(stream),token_timestamps=model.timestamps(stream)));model.reset_stream(stream)
   for start in range(0,len(samples),320):
    end=min(start+320,len(samples));stream.accept_waveform(16000,samples[start:end]);drain(end,False)
   stream.input_finished();drain(len(samples),True);return checked_events(ev)
  return run
 if route=='cfsmn':
  import torch,numpy as np
  assert torch.__version__=='2.13.0+cpu' and np.__version__=='2.5.2'
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  spec=importlib.util.spec_from_file_location('frozen_cfsmn',F/'run_baseline.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  front,ns,_=mod.audited_sources();tokens=(F/'resources/tokens_2599.txt').read_text().splitlines();model=mod.Donor().eval()
  def run(row,b):
   model.logs=[];k=mod.make_spotter('hamming',model,front,ns,tokens);events=[]
   with torch.inference_mode():
    for start in range(0,len(b),9600):
     end=min(start+9600,len(b));v=k.forward(b[start:end])
     if v.get('state')==1:events.append(dict(v,keyword_id=mod.PHRASES[v['keyword']],available_audio_samples=end//2,eof_flush=False))
    assert all(torch.isfinite(x).all() for x in model.logs)
   model.logs=[];return checked_events(events)
  return run
 raise ValueError(route)

def event_identity(events):
 # Compare complete deterministic observations, including scores/timestamps.
 return json.dumps(events,sort_keys=True,ensure_ascii=False,allow_nan=False)
def timeout(signum,frame):raise TimeoutError('per-clip 10s wall cap')
def io_snapshot():
 return {k:int(v) for k,v in (line.split(':') for line in pathlib.Path('/proc/self/io').read_text().splitlines())}
def usage():
 a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
 return dict(cpu_s=a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime,self_maxrss_kib=a.ru_maxrss,children_maxrss_kib=b.ru_maxrss,inblock=a.ru_inblock+b.ru_inblock,outblock=a.ru_oublock+b.ru_oublock,major_faults=a.ru_majflt+b.ru_majflt,minor_faults=a.ru_minflt+b.ru_minflt)
def worker(route):
 assert os.environ.get('ORT_DISABLE_TELEMETRY')=='1'
 initial_io=io_snapshot()
 spec=json.loads((ROOT/'run-spec.json').read_text())
 for p,h in spec['bindings'].items():assert sha(p)==h,('binding drift',p)
 assert sys.version==spec['runtimes'][route]['python']
 assert subprocess.check_output(['git','-C',str(DATA),'rev-parse','HEAD'],text=True).strip()=='2f9658ffa9568076ef547615c76861abed84f56e'
 receipt=json.loads((B/'native-export-receipt.json').read_text());qw=[dict(r,path=str(DATA/r['path']),dataset_id=d['dataset_id']) for d in receipt['datasets'] for r in d['recordings']];assert len(qw)==42
 negative=json.loads((ROOT/'inputs.json').read_text());rows=negative['rows'];assert len(rows)==7006
 reference=json.loads(pathlib.Path(spec['reference_files'][route]).read_text())
 if route=='c':expected={r['recording']:[] for r in qw}
 else:expected={r['recording']:r['events'] for r in reference['recordings']}
 assert set(expected)=={r['recording'] for r in qw}
 out=ROOT/'runs'/route;run=make_engine(route);signal.signal(signal.SIGALRM,timeout);counts={};before={}
 with (out/'records.jsonl').open('x') as f:
  for stage,records in [('qwen_before',qw),('himia',rows),('qwen_after',qw)]:
   g={'clips':0,'events':0,'triggered_clips':0,'frames':0,'positive_hits':0};counts[stage]=g
   for row in records:
    t=time.monotonic();heartbeat(out/'clip-heartbeat.json',dict(active=True,started=t,stage=stage,recording=row['recording']));signal.setitimer(signal.ITIMER_REAL,10)
    try:events=run(row,pcm(row))
    finally:
     signal.setitimer(signal.ITIMER_REAL,0);heartbeat(out/'clip-heartbeat.json',dict(active=False,started=t))
    identity=event_identity(events)
    if stage!='himia':
     assert identity==event_identity(expected[row['recording']]),('Qwen baseline mismatch',row['recording'])
     if stage=='qwen_before':before[row['recording']]=identity
     else:assert identity==before[row['recording']]
    rec=dict(stage=stage,**row,events=events,wall_s=time.monotonic()-t)
    f.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    g['clips']+=1;g['frames']+=row['frames'];g['events']+=len(events);g['triggered_clips']+=bool(events)
    if row.get('kind')=='positive':g['positive_hits']+=any(e['keyword_id']==row['keyword_id'] for e in events)
   dump(out/'progress.json',counts)
 final_io=io_snapshot();u=usage()
 assert u['cpu_s']<=900,'final cumulative CPU budget'
 dump(out/'complete.json',dict(status='completed',counts=counts,source_label_only=True,qualification_allowed=False,spec_sha256=sha(ROOT/'run-spec.json'),resource_usage=u,self_io_delta={k:final_io[k]-initial_io[k] for k in initial_io},io_scope='worker /proc counters, includes hashing/report bookkeeping; child I/O not exhaustively traced'))

def process_group(pgid):
 rss=0;cpu=0.;found=0;ticks=os.sysconf('SC_CLK_TCK');pages=os.sysconf('SC_PAGE_SIZE')
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   a=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(a[2])!=pgid:continue
   found+=1;cpu+=sum(int(a[i]) for i in [11,12,13,14])/ticks;rss+=int(a[21])*pages
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return dict(cpu_s=cpu,rss_bytes=rss,processes=found)
def monitor(p,start,clip_path,clip_limit=10,wall_limit=1200,cpu_limit=900,rss_limit=1024**3):
 peak=0;cpu=0.;reason=None;code=None
 try:
  while p.poll() is None:
   m=process_group(p.pid);peak=max(peak,m['rss_bytes']);cpu=max(cpu,m['cpu_s']);now=time.monotonic()
   if now-start>wall_limit:reason='wall budget'
   elif cpu>cpu_limit:reason='CPU budget'
   elif peak>rss_limit:reason='process group RSS budget'
   if clip_path.exists():
    h=json.loads(clip_path.read_text())
    if h['active'] and now-h['started']>clip_limit:reason='hard per-clip wall budget'
   if reason:break
   time.sleep(.05)
 except Exception as exc:
  reason='monitor error: '+type(exc).__name__+': '+str(exc)
 finally:
  # Own this session from Popen; always reap/terminate the whole group, even on monitor errors.
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  code=p.wait()
 return dict(returncode=code,stop_reason=reason,wall_s=time.monotonic()-start,observed_group_cpu_s=cpu,observed_group_peak_rss_bytes=peak,poll_interval_s=.05,status='completed' if code==0 and not reason else 'invalid')
def supervise(route):
 out=ROOT/'runs'/route;out.mkdir(parents=True,exist_ok=False)
 runtime={'c':sys.executable,'sherpa':'/workspace/shared/kws-external-baseline/venv/bin/python','cfsmn':'/workspace/shared/kws-lightweight-prototypes/venv/bin/python'}[route]
 start=time.monotonic();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',ORT_DISABLE_TELEMETRY='1')
 with (out/'console.log').open('x') as log:
  p=subprocess.Popen([runtime,str(ROOT/'driver.py'),'worker',route],start_new_session=True,stdout=log,stderr=subprocess.STDOUT,env=env)
  result=monitor(p,start,out/'clip-heartbeat.json')
 dump(out/'supervisor.json',result)
 if result['status']!='completed':raise SystemExit(1)
if __name__=='__main__':
 if sys.argv[1]=='worker':worker(sys.argv[2])
 elif sys.argv[1]=='run':supervise(sys.argv[2])
 else:raise SystemExit('use run or worker')
