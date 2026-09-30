"""Bounded host-only model resource profile. No training or audio/feature extraction."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import pathlib
import platform
import resource
import sys
import time
import numpy as np
import torch

ROOT=pathlib.Path(__file__).resolve().parent
STUDY=pathlib.Path('/workspace/shared/kws-lightweight-prototypes')
sys.path.insert(0,str(STUDY))
from models import CausalClassifier


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def read_io():
    try:
        fd=os.open('/proc/self/io',os.O_RDONLY)
        try:raw=os.read(fd,4096)
        finally:os.close(fd)
        return {k:int(v) for k,v in (line.split(':') for line in raw.decode().splitlines())}
    except OSError:return None


def status():
    try:
        fields={k:v.strip() for k,v in (line.split(':',1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)}
        return {k:fields.get(k) for k in ('VmRSS','VmHWM','VmSize','Threads')}
    except OSError:return {}


def summary(values):
    return {'mean_us':float(np.mean(values))/1000,'p50_us':float(np.quantile(values,.5))/1000,
            'p95_us':float(np.quantile(values,.95))/1000,'p99_us':float(np.quantile(values,.99))/1000,
            'max_us':float(np.max(values))/1000}


def storage_bytes(tensors):
    storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
    return sum(storages.values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--architecture',choices=['A','B'],required=True)
    args=parser.parse_args();arch=args.architecture
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    spec=json.loads((STUDY/'spec.json').read_text())
    assert sha(STUDY/'models.py')==spec['source_hashes']['models.py']
    assert sha(STUDY/'features.npz')==spec['feature_cache_sha256']
    checkpoint_path=STUDY/'results'/(arch+'-seed1337')/'last.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model=CausalClassifier(arch).eval();model.load_state_dict(checkpoint['state_dict'])
    cache=np.load(STUDY/'features.npz')
    all_features=np.concatenate([cache[k] for k in sorted(cache.files)])
    tensor=torch.from_numpy(all_features)[None]
    frames=[tensor[:,i:i+1] for i in range(tensor.shape[1])]
    timings_wall=np.empty(3000,dtype=np.int64);timings_cpu=np.empty(3000,dtype=np.int64)
    timer_wall=np.empty(1000,dtype=np.int64);timer_cpu=np.empty(1000,dtype=np.int64)
    for i in range(1000):
        start_wall=time.perf_counter_ns();start_cpu=time.process_time_ns()
        end_cpu=time.process_time_ns();end_wall=time.perf_counter_ns()
        timer_wall[i]=end_wall-start_wall;timer_cpu[i]=end_cpu-start_cpu
    state=None
    with torch.inference_mode():
        for i in range(200):
            _,state=model.chunk(frames[i%len(frames)],state)
        state_logical=sum(x.numel()*x.element_size() for x in state)
        state_backing=storage_bytes(state)
        rss_before=status()
        usage_before=resource.getrusage(resource.RUSAGE_SELF)
        io_before=read_io()
        block_start=time.perf_counter_ns();block_cpu=time.process_time_ns()
        for i in range(3000):
            frame=frames[(200+i)%len(frames)]
            start_wall=time.perf_counter_ns();start_cpu=time.process_time_ns()
            _,state=model.chunk(frame,state)
            end_cpu=time.process_time_ns();end_wall=time.perf_counter_ns()
            timings_cpu[i]=end_cpu-start_cpu;timings_wall[i]=end_wall-start_wall
        block_cpu_ns=time.process_time_ns()-block_cpu
        block_wall_ns=time.perf_counter_ns()-block_start
        io_after=read_io()
        usage_after=resource.getrusage(resource.RUSAGE_SELF)
        rss_after=status()
    params=list(model.parameters())
    fp32_bytes=sum(p.numel()*p.element_size() for p in params)
    weight_elements=sum(p.numel() for name,p in model.named_parameters() if name.endswith('weight'))
    bias_elements=sum(p.numel() for name,p in model.named_parameters() if name.endswith('bias'))
    wall=summary(timings_wall);cpu=summary(timings_cpu)
    report={'schema_version':1,'architecture':arch,'seed':1337,'evidence_scope':'host-PyTorch-model-only-not-SSC305',
      'source_spec_sha256':sha(STUDY/'spec.json'),'model_code_sha256':sha(STUDY/'models.py'),
      'checkpoint_sha256':sha(checkpoint_path),'checkpoint_serialized_bytes':checkpoint_path.stat().st_size,
      'feature_cache_sha256':sha(STUDY/'features.npz'),'profile_script_sha256':sha(pathlib.Path(__file__)),
      'host':{'machine':platform.machine(),'kernel':platform.release(),'python':sys.version,'torch':torch.__version__,
              'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
              'affinity_cpu_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
              'process_cpu_affinity_pinned':False,'mkldnn':False},
      'measured_memory':{'parameters':sum(p.numel() for p in params),'fp32_parameter_payload_bytes':fp32_bytes,
                         'fp32_parameter_backing_storage_bytes':storage_bytes(params),
                         'causal_state_logical_tensor_bytes':state_logical,
                         'causal_state_backing_storage_bytes':state_backing,
                         'state_note':'returned cache views retain joined-history-plus-current-frame storage; not a C fixed arena',
                         'ru_maxrss_kib':usage_after.ru_maxrss,'status_before_window':rss_before,'status_after_window':rss_after,
                         'rss_scope':'whole Python/PyTorch process, model, loaded feature cache, timing arrays, allocator; not fixed KWS RAM'},
      'estimate_only':{'int8_weight_plus_fp32_bias_bytes':weight_elements+4*bias_elements,
                       'excluded':['quantization scales','alignment','frontend','working memory','code'],
                       'quantization_executed':False},
      'protocol':{'input':'existing32-dimensional Clogmel features; no PCM/frontend in timed window',
                  'frame_hop_ms':20,'warmup_frames':200,'measured_frames':3000,'repeats':3,'frames_per_repeat':1000,
                  'batch':1,'incremental_state':True,'timed_api':'CausalClassifier.chunk(B=1,T=1,F=32)',
                  'includes':'Python/PyTorch dispatch, dynamic tensor allocation and kernel execution; timer overhead retained',
                  'io':'inputs preloaded; no file/log/network operations in measured frame loop; reports written after window'},
      'wall_per_frame':wall,'process_cpu_per_frame':cpu,
      'whole_window_wall_seconds':block_wall_ns/1e9,'whole_window_process_cpu_seconds':block_cpu_ns/1e9,
      'host_one_core_equivalent_at50fps_percent_from_cpu_mean':cpu['mean_us']*50/10000,
      'empty_timer_overhead_wall':summary(timer_wall),'empty_timer_overhead_cpu':summary(timer_cpu),
      'io_raw_delta':None if io_before is None or io_after is None else {k:io_after[k]-io_before[k] for k in io_before},
      'io_observer_caveat':'proc/self/io snapshots themselves cause read syscall/character counts; do not infer frame-loop reads from raw syscr/rchar alone',
      'rusage_window_delta':{'minor_faults':usage_after.ru_minflt-usage_before.ru_minflt,
                            'major_faults':usage_after.ru_majflt-usage_before.ru_majflt,
                            'input_blocks':usage_after.ru_inblock-usage_before.ru_inblock,
                            'output_blocks':usage_after.ru_oublock-usage_before.ru_oublock},
      'per_repeat':[{'repeat':i,'wall':summary(timings_wall[i*1000:(i+1)*1000]),
                      'cpu':summary(timings_cpu[i*1000:(i+1)*1000])} for i in range(3)],
      'target_cpu_claim':None}
    raw=ROOT/(arch+'-timings.npz');np.savez(raw,wall_ns=timings_wall,cpu_ns=timings_cpu)
    report['timings_sha256']=sha(raw)
    (ROOT/(arch+'-profile.json')).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'architecture':arch,'cpu':cpu,'wall':wall,'rss_kib':usage_after.ru_maxrss,
                      'parameters_bytes':fp32_bytes,'state_logical_bytes':state_logical,'state_backing_bytes':state_backing,
                      'checkpoint_bytes':checkpoint_path.stat().st_size,'io':report['io_raw_delta']},indent=2))

if __name__=='__main__':main()
