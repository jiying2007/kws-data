#!/usr/bin/env python3
"""One-thread x86 resource profile; independent of frozen two-thread quality run."""
import os,json,pathlib,hashlib,time,resource,platform,sys
ROOT=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/workspace/shared/kws-external-baseline')
DATA=pathlib.Path('/workspace/shared/kws-data-pinned')
RECEIPT=pathlib.Path('/workspace/shared/kws-data-consumer/build/observed-readback/native-export-receipt.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snap():
 io={k:int(v) for k,v in (line.split(':') for line in pathlib.Path('/proc/self/io').read_text().splitlines())}
 status={}
 for line in pathlib.Path('/proc/self/status').read_text().splitlines():
  if line.startswith(('VmRSS:','VmHWM:','VmSize:','Threads:')):status[line.split(':')[0]]=line.split(':')[1].strip()
 r=resource.getrusage(resource.RUSAGE_SELF)
 return dict(io=io,status=status,user_cpu_s=r.ru_utime,system_cpu_s=r.ru_stime,max_rss_kib=r.ru_maxrss,minor_faults=r.ru_minflt,major_faults=r.ru_majflt)
def delta(a,b):
 return {'io':{k:b['io'][k]-a['io'][k] for k in a['io']},'minor_faults':b['minor_faults']-a['minor_faults'],'major_faults':b['major_faults']-a['major_faults'],'user_cpu_s':b['user_cpu_s']-a['user_cpu_s'],'system_cpu_s':b['system_cpu_s']-a['system_cpu_s']}
def percentiles(values):
 a=sorted(values)
 def q(p):
  x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
 return {'count':len(a),'p50_ms':q(.5)/1e6,'p95_ms':q(.95)/1e6,'p99_ms':q(.99)/1e6,'max_ms':a[-1]/1e6}
start=snap();t=time.perf_counter();c=time.process_time()
import numpy as np
import sherpa_onnx
import wave
imports={'wall_s':time.perf_counter()-t,'cpu_s':time.process_time()-c,'before':start,'after':snap()}
assert sha(RECEIPT)=='8e4d5c13cc5694e813ef6dc58bedbe34d2b893721283f1945d6d944ca38570b5'
receipt=json.loads(RECEIPT.read_text());rows=[x for d in receipt['datasets'] for x in d['recordings']];assert len(rows)==42
preload_before=snap();t=time.perf_counter();audio=[]
for row in rows:
 p=DATA/row['path'];assert sha(p)==row['file_sha256']
 with wave.open(str(p)) as w:
  assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2
  pcm=w.readframes(w.getnframes())
 assert hashlib.sha256(pcm).hexdigest()==row['pcm_sha256']
 audio.append((row,np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.0))
preload={'wall_s':time.perf_counter()-t,'before':preload_before,'after':snap(),'total_pcm_frames':sum(len(a) for _,a in audio),'audio_seconds':sum(len(a)/16000 for _,a in audio),'numpy_array_bytes':sum(a.nbytes for _,a in audio)}
# Asset hashing intentionally precedes timed constructor load; cache is consequently warm/unknown.
expected={x['path']:x for x in json.loads((BASE/'evidence/downloads.json').read_text())}
assets=[]
for name in ['encoder-epoch-12-avg-2-chunk-16-left-64.onnx','decoder-epoch-12-avg-2-chunk-16-left-64.onnx','joiner-epoch-12-avg-2-chunk-16-left-64.onnx','tokens.txt']:
 p=BASE/'models'/name;h=sha(p);assert h==expected['models/'+name]['expected_sha256'];assets.append({'file':name,'bytes':p.stat().st_size,'sha256':h})
config=json.loads((BASE/'config.json').read_text());config['num_threads']=1
kw={x:config[x] for x in ['num_threads','sample_rate','feature_dim','max_active_paths','keywords_score','keywords_threshold','num_trailing_blanks','provider']}
for comp in ['encoder','decoder','joiner']:kw[comp]=str(BASE/'models'/f'{comp}-epoch-12-avg-2-chunk-16-left-64.onnx')
kw.update(tokens=str(BASE/'models/tokens.txt'),keywords_file=str(BASE/'keywords.txt'))
load_before=snap();t=time.perf_counter();c=time.process_time();kws=sherpa_onnx.KeywordSpotter(**kw)
load={'wall_s':time.perf_counter()-t,'cpu_s':time.process_time()-c,'before':load_before,'after':snap()}
assert load['after']['status']['Threads']=='1', 'one-thread resource protocol not achieved'
all_feed_ns=[];all_ready_ns=[];all_idle_ns=[]
def pass_audio(measure):
 feed_times=[];ready_times=[];idle_times=[];decode_calls=0;events=[];eof_times=[];stream_create_ns=[]
 for row,samples in audio:
  tick=time.perf_counter_ns();stream=kws.create_stream();stream_create_ns.append(time.perf_counter_ns()-tick)
  for first in range(0,len(samples),320):
   last=min(first+320,len(samples));tick=time.perf_counter_ns();stream.accept_waveform(16000,samples[first:last]);calls=0
   while kws.is_ready(stream):
    kws.decode_stream(stream);calls+=1
    phrase=kws.get_result(stream)
    if phrase:events.append({'recording':row['recording'],'phrase':phrase,'after_samples':last,'eof':False});kws.reset_stream(stream)
   duration=time.perf_counter_ns()-tick
   if measure:
    feed_times.append(duration);(ready_times if calls else idle_times).append(duration)
   decode_calls+=calls
  tick=time.perf_counter_ns();stream.input_finished()
  while kws.is_ready(stream):
   kws.decode_stream(stream);decode_calls+=1;phrase=kws.get_result(stream)
   if phrase:events.append({'recording':row['recording'],'phrase':phrase,'after_samples':len(samples),'eof':True});kws.reset_stream(stream)
  eof_times.append(time.perf_counter_ns()-tick)
 return feed_times,ready_times,idle_times,decode_calls,events,eof_times,stream_create_ns
warm_before=snap();t=time.perf_counter();c=time.process_time();warm=pass_audio(False)
warmup={'clips':42,'wall_s':time.perf_counter()-t,'cpu_s':time.process_time()-c,'before':warm_before,'after':snap(),'events':len(warm[4])}
runs=[];steady_before=snap();repeat_raw=[]
for repeat in range(3):
 before=snap();t=time.perf_counter();c=time.process_time();feed,ready,idle,n,events,eof,create=pass_audio(True);cpu=time.process_time()-c;wall=time.perf_counter()-t;after=snap();seconds=preload['audio_seconds']
 item={'repeat':repeat+1,'clips':42,'audio_s':seconds,'wall_s':wall,'cpu_s':cpu,'wall_rtf':wall/seconds,'cpu_rtf':cpu/seconds,'decode_calls':n,'events':len(events),'feed_plus_ready_decode':percentiles(feed),'feeds_with_decode':percentiles(ready),'feeds_without_decode':percentiles(idle),'eof_drain':percentiles(eof),'stream_creation':percentiles(create),'before':before,'after':after,'delta':delta(before,after)}
 runs.append(item);all_feed_ns+=feed;all_ready_ns+=ready;all_idle_ns+=idle;repeat_raw.append({'repeat':repeat+1,'feed_ns':feed,'eof_ns':eof,'stream_creation_ns':create,'events':events})
steady_after=snap()
identity={'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'sherpa_onnx':sherpa_onnx.__version__,'logical_cpus':os.cpu_count(),'affinity_count':len(os.sched_getaffinity(0)),'cpu_model':next((x.split(':',1)[1].strip() for x in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown')}
for name in ['cpu.max','memory.max','cpu.stat']:
 p=pathlib.Path('/sys/fs/cgroup')/name
 if p.exists():identity['cgroup_'+name]=p.read_text().strip()
result={'schema_version':1,'scope':'x86 Python/ORT persistent-model microbenchmark, not SSC305 qualification or planned C working RAM','assets':assets,'asset_total_bytes':sum(x['bytes'] for x in assets),'identity':identity,'config':config,'protocol':{'model_loads':1,'audio_preloaded_float32':True,'warmup_passes':1,'measured_passes':3,'clips_per_pass':42,'feed_samples':320,'model_thread_setting':1,'fresh_stream_per_clip':True,'tail_padding_samples':0,'quality_result_mutated':False,'cache':'not cold; assets hashed before constructor; OS cache otherwise unknown; no cache drops','io_note':'/proc/self/io rchar/syscr include proc/status bookkeeping; read_bytes is storage-layer accounting, not mapped model size; report major/minor faults separately','timing_note':'wall feed timing includes accept_waveform, is_ready, all ready decode calls, result check and event reset; no real-time pacing; partial final block may be shorter than20ms; percentile includes all original clips and repeated passes; EOF/stream construction separately reported','memory_note':'RSS includes Python, NumPy, ORT, loaded weights, preloaded audio and instrumentation; peak is process-lifetime ru_maxrss, not incremental model workspace'},'native_export_sha256':sha(RECEIPT),'config_source_sha256':sha(BASE/'config.json'),'keywords_sha256':sha(BASE/'keywords.txt'),'script_sha256':sha(pathlib.Path(__file__)),'imports':imports,'audio_preload':preload,'model_load':dict(load,delta=delta(load['before'],load['after'])),'warmup':warmup,'runs':runs,'steady':{'before':steady_before,'after':steady_after,'delta':delta(steady_before,steady_after),'audio_s':3*preload['audio_seconds'],'cpu_s':sum(x['cpu_s'] for x in runs),'wall_s':sum(x['wall_s'] for x in runs),'cpu_rtf':sum(x['cpu_s'] for x in runs)/(3*preload['audio_seconds']),'wall_rtf':sum(x['wall_s'] for x in runs)/(3*preload['audio_seconds']),'feed_plus_ready_decode':percentiles(all_feed_ns),'feeds_with_decode':percentiles(all_ready_ns),'feeds_without_decode':percentiles(all_idle_ns)}}
(ROOT/'measurement.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(ROOT/'raw-timings.json').write_text(json.dumps(repeat_raw,ensure_ascii=False)+'\n');print(json.dumps(result['steady'],indent=2));print('model_load',json.dumps(result['model_load']));print('assets',result['asset_total_bytes'])
