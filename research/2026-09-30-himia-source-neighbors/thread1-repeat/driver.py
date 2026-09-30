"""Single-variable sherpa API num_threads=1 repeat; original audited driver remains immutable."""
import importlib.util,json,pathlib,sys,os,time,signal,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
ORIGINAL=pathlib.Path('/workspace/shared/kws-himia-readback')
spec=importlib.util.spec_from_file_location('audited_base',ORIGINAL/'driver.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.ROOT=ROOT

def make_engine(route):
 assert route=='sherpa'
 import numpy as np,sherpa_onnx
 assert sherpa_onnx.__version__=='1.13.8' and np.__version__=='2.2.6'
 cfg=json.loads((base.S/'config.json').read_text());assert cfg['num_threads']==2
 kw={k:cfg[k] for k in ['num_threads','sample_rate','feature_dim','max_active_paths','keywords_score','keywords_threshold','num_trailing_blanks','provider']}
 kw['num_threads']=1 # The single experimental configuration change.
 for x in ['encoder','decoder','joiner']:kw[x]=str(base.S/'models'/f'{x}-epoch-12-avg-2-chunk-16-left-64.onnx')
 kw['encoder']=str(base.S/'models'/cfg['encoder_file']);kw.update(tokens=str(base.S/'models/tokens.txt'),keywords_file=str(base.S/'keywords.txt'));model=sherpa_onnx.KeywordSpotter(**kw)
 previous=(ORIGINAL/'runs/sherpa/records.jsonl').open();calls=0
 def run(row,b):
  nonlocal calls
  old=json.loads(next(previous));calls+=1
  assert old['recording']==row['recording'] and old['file_sha256']==row['file_sha256'] and old['pcm_sha256']==row['pcm_sha256']
  samples=np.frombuffer(b,dtype='<i2').astype(np.float32)/32768.;stream=model.create_stream();ev=[]
  def drain(fed,eof):
   while model.is_ready(stream):
    model.decode_stream(stream);word=model.get_result(stream)
    if word:
     ev.append(dict(keyword_id={'你好小窝':1,'小窝小窝':2}[word],phrase=word,available_audio_samples=fed,eof_flush=eof,tokens=model.tokens(stream),token_timestamps=model.timestamps(stream)));model.reset_stream(stream)
  for start in range(0,len(samples),320):
   end=min(start+320,len(samples));stream.accept_waveform(16000,samples[start:end]);drain(end,False)
  stream.input_finished();drain(len(samples),True);base.checked_events(ev)
  assert base.event_identity(ev)==base.event_identity(old['events']),('two-config event mismatch',row['recording'])
  if calls==7090:assert previous.read()=='';previous.close()
  return ev
 return run
base.make_engine=make_engine

def observe_worker(pid):
 """Worker PID only, excluding supervisor; group CPU/RSS remain separately monitored."""
 p=pathlib.Path('/proc')/str(pid);fields=dict(l.split(':',1) for l in (p/'status').read_text().splitlines() if ':' in l)
 return dict(worker_threads=int(fields['Threads']),worker_pid=pid)

def monitor(p,start,clip_path):
 peak=0;cpu=0.;reason=None;thread_samples=[]
 try:
  while p.poll() is None:
   m=base.process_group(p.pid);peak=max(peak,m['rss_bytes']);cpu=max(cpu,m['cpu_s']);now=time.monotonic()
   try:thread_samples.append(dict(elapsed_s=now-start,**observe_worker(p.pid)))
   except FileNotFoundError:
    if p.poll() is None:raise
   if now-start>1200:reason='wall budget'
   elif cpu>900:reason='CPU budget'
   elif peak>1024**3:reason='process group RSS budget'
   if clip_path.exists():
    h=json.loads(clip_path.read_text())
    if h['active'] and now-h['started']>10:reason='hard per-clip wall budget'
   if reason:break
   time.sleep(.05)
 except Exception as exc:reason='monitor error: '+type(exc).__name__+': '+str(exc)
 finally:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  code=p.wait()
 base.dump(clip_path.parent/'worker-thread-samples.json',thread_samples)
 return dict(returncode=code,stop_reason=reason,wall_s=time.monotonic()-start,observed_group_cpu_s=cpu,observed_group_peak_rss_bytes=peak,poll_interval_s=.05,status='completed' if code==0 and not reason else 'invalid',worker_threads_min=min(x['worker_threads'] for x in thread_samples) if thread_samples else None,worker_threads_max=max(x['worker_threads'] for x in thread_samples) if thread_samples else None,thread_scope='worker process only; supervisor excluded; sampled, not exhaustive')
base.monitor=monitor
if __name__=='__main__':
 if sys.argv[1]=='worker':base.worker('sherpa')
 elif sys.argv[1]=='run':base.supervise('sherpa')
 else:raise SystemExit('use run or worker')
