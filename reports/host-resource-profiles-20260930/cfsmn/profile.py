"""Single-thread x86 observational profile of unchanged Hamming/300ms route."""
import time, pathlib, json, os, resource, hashlib, wave, importlib.util, platform
OUT=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/workspace/shared/kws-cfsmn-baseline')
def io(): return {k:int(v) for k,v in (line.split(':') for line in pathlib.Path('/proc/self/io').read_text().splitlines())}
def delta(a,b): return {k:b[k]-a[k] for k in a}
def rss(): return {'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'status':{l.split(':')[0]:l.split(':')[1].strip() for l in pathlib.Path('/proc/self/status').read_text().splitlines() if l.startswith(('VmRSS:','VmHWM:','Threads:'))}}
start_io=io();start_cpu=time.process_time();start_wall=time.perf_counter()
spec=importlib.util.spec_from_file_location('baseline',BASE/'run_baseline.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
frontend,ns,tokens,conditions,baseline_spec=b.setup()
assert b.sha(BASE/'baseline-spec.json')=='26016a32d42ff25b77dbf011064e07cc52abe5dfec0c9a2a994c766734300831'
model=b.Donor().eval();root,rows=conditions['raw']
expected={r['recording']:r for r in json.loads((BASE/'results/hamming-raw.json').read_text())['recordings']}
startup={'cpu_s':time.process_time()-start_cpu,'wall_s':time.perf_counter()-start_wall,'io_delta':delta(start_io,io()),'memory':rss(),'scope':'Python after stdlib imports; includes torch/numpy imports, source/input verification for all three conditions and donor load; warm page cache uncontrolled'}
stats={};state_max={};chunks=[]
def measured(name,fn):
 def wrapped(*args,**kw):
  c=time.process_time();w=time.perf_counter();value=fn(*args,**kw);entry=stats.setdefault(name,{'cpu_s':0.,'wall_s':0.,'calls':0});entry['cpu_s']+=time.process_time()-c;entry['wall_s']+=time.perf_counter()-w;entry['calls']+=1;return value
 return wrapped
model.forward=measured('model_cmvn_and_full_2599_head',model.forward)
def inspect_state(k):
 for name in ['in_cache','feature_remained','wave_remained']:
  v=getattr(k,name)
  if v is None:continue
  if isinstance(v,b.torch.Tensor):logical=v.numel()*v.element_size();storage=v.untyped_storage().nbytes()
  else:logical=v.nbytes;storage=logical
  entry=state_max.setdefault(name,{'max_logical_bytes':0,'max_backing_storage_bytes':0})
  entry['max_logical_bytes']=max(entry['max_logical_bytes'],logical);entry['max_backing_storage_bytes']=max(entry['max_backing_storage_bytes'],storage)
def clip(row):
 model.logs=[]
 begin_c=time.process_time();begin_w=time.perf_counter()
 k=b.make_spotter('hamming',model,frontend,ns,tokens)
 for name in ['accept_wave','decode_keywords','execute_detection']:setattr(k,name,measured(name,getattr(k,name)))
 with wave.open(str(root/row['path'])) as w:pcm=w.readframes(w.getnframes())
 events=[]
 for start in range(0,len(pcm),9600):
  end=min(len(pcm),start+9600);c=time.process_time();w=time.perf_counter();r=k.forward(pcm[start:end]);chunks.append({'samples':(end-start)//2,'cpu_s':time.process_time()-c,'wall_s':time.perf_counter()-w})
  if r.get('state')==1:events.append(dict(r,keyword_id=b.PHRASES[r['keyword']],available_audio_samples=end//2,eof_flush=False))
 end_c=time.process_time();end_w=time.perf_counter();inspect_state(k)
 # Identity checks are deliberately outside clip timing.
 logits=b.torch.cat(model.logs,1)[0];e=expected[row['recording']]
 assert hashlib.sha256(logits.numpy().tobytes()).hexdigest()==e['logits_sha256']
 assert events==e['events']
 return {'recording':row['recording'],'audio_s':len(pcm)/32000,'cpu_s':end_c-begin_c,'wall_s':end_w-begin_w,'event_and_logits_identical':True}
with b.torch.inference_mode():
 for row in rows[:3]:clip(row)
 warmup={'clips':3,'audio_s':sum(r['frames']/16000 for r in rows[:3]),'memory':rss()}
 passes=[]
 for repeat in range(3):
  stats.clear();chunks.clear();before=io();c=time.process_time();w=time.perf_counter();recordings=[clip(r) for r in rows];wc=time.perf_counter()-w;cc=time.process_time()-c;after=io()
  duration=sum(r['audio_s'] for r in recordings);pipeline_c=sum(r['cpu_s'] for r in recordings);pipeline_w=sum(r['wall_s'] for r in recordings)
  components={k:dict(v,cpu_rtf=v['cpu_s']/duration,wall_rtf=v['wall_s']/duration) for k,v in stats.items()}
  def distribution(key):
   a=sorted(r[key] for r in chunks);return {'min':a[0],'median':a[len(a)//2],'p95':a[min(len(a)-1,int(.95*len(a)))],'max':a[-1]}
  passes.append({'repeat':repeat+1,'audio_s':duration,'pipeline_cpu_s':pipeline_c,'pipeline_wall_s':pipeline_w,'pipeline_cpu_rtf':pipeline_c/duration,'pipeline_wall_rtf':pipeline_w/duration,'components':components,'residual_cpu_s':pipeline_c-sum(v['cpu_s'] for v in stats.values()),'measurement_loop_including_identity_checks':{'cpu_s':cc,'wall_s':wc},'io_delta':delta(before,after),'memory':rss(),'chunk_count':len(chunks),'chunk_cpu_s_distribution':distribution('cpu_s'),'chunk_wall_s_distribution':distribution('wall_s'),'recordings':recordings})
  print(json.dumps({k:v for k,v in passes[-1].items() if k not in ['recordings','components']}),flush=True)
params=sum(p.numel() for p in model.parameters());buffers=sum(p.numel()*p.element_size() for p in model.buffers())
result={'profile_sha256':b.sha(pathlib.Path(__file__)),'baseline_spec_sha256':b.sha(BASE/'baseline-spec.json'),'baseline_driver_sha256':b.sha(BASE/'run_baseline.py'),'environment':{'machine':platform.machine(),'platform':platform.platform(),'python':platform.python_version(),'torch':b.torch.__version__,'torch_threads':b.torch.get_num_threads(),'torch_interop_threads':b.torch.get_num_interop_threads(),'cpu_model':next(l.split(':',1)[1].strip() for l in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'affinity_cpu_count':len(os.sched_getaffinity(0))},'scope':'Hamming, original 42 clips, warmup 3 clips, 3 passes, unchanged 300ms-chunk official path fed offline; not hard-real-time or SSC305 measurement','static':{'checkpoint_bytes':(BASE/'resources/base.pt').stat().st_size,'checkpoint_sha256':b.sha(BASE/'resources/base.pt'),'learned_parameters':params,'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'cmvn_buffer_bytes':buffers,'tensor_bytes':sum(p.numel()*p.element_size() for p in model.state_dict().values()),'pcm_input_bytes_per_audio_second':32000},'state_observed_max':state_max,'startup':startup,'warmup':warmup,'passes':passes,'final_memory':rss(),'caveats':['Peak RSS is whole PyTorch/Python research process, not model or C working set','Donor forward retains logits for parity; full pipeline includes 2599 softmax, official Python decoder, WAV read, fresh spotter creation and timing instrumentation','Measured component CPU times exclude instrument wrapper bookkeeping; residual includes softmax/orchestration and remaining work','No page-cache dropping; /proc read_bytes reflects storage I/O attribution, rchar includes cached reads and proc snapshots','No 20ms deadline measurement; 300ms buffering and finite lookahead retained','State tensor bytes are not full C working-memory estimate; allocator/workspaces/decoder/frontend transient memory not included']}
b.save(OUT/'profile.json',result)
