"""Fresh isolated native worker. Reuse capture functions, never old main/run.

The host stages exact upstream bytes and this adapter in read-only /code. The
fresh contract is explicitly injected into the receipt API before importing any
model. The old campaign's audio/remaining35/dispatch functions are never called.
"""
from pathlib import Path
import hashlib
import importlib.util
import os
import re
import signal
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import fresh_contract as fresh


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def verify_completed(directory,job,family):
    """Exact completed roster and raw receipts, never inferred from exit code."""
    import json
    directory=Path(directory);rows=fresh.remaining_plan(job,family)
    fresh.require({p.name for p in directory.glob('clip-*')}=={r['audio_id'] for r in rows},'complete fresh family roster')
    ids=[]
    for row in rows:
        root=directory/row['audio_id'];bodies={name:fresh.bounded_read(root/name,8*1024**2) for name in
            ('started.json','completed.json','raw-transcript.json','raw-native-output.json')}
        start=json.loads(bodies['started.json']);end=json.loads(bodies['completed.json'])
        fresh.validate_start_slot(job,family,audio_id=row['audio_id'],
            measured={k:start[k] for k in ('file_sha256','pcm_sha256','audio_frames')},
            source_id=start['source_id'],request_id=start['request_id'],run_id=start['run_id'])
        fresh.require(start['schema']=='retained-asr-started-v1' and end['schema']=='retained-asr-completed-v1' and
            end['status']=='success' and end['decoder_completed'] is True and type(end['processed_start_frame']) is int and
            end['processed_start_frame']==0 and type(end['processed_end_frame']) is int and end['processed_end_frame']==row['audio_frames'],
            'real complete native return required')
        for key,value in start.items():
            if key!='schema':fresh.require(fresh.canonical(end.get(key))==fresh.canonical(value),'start/completion binding drift')
        fresh.require(end['started_sha256']==hashlib.sha256(bodies['started.json']).hexdigest() and
            end['raw_output_sha256']==hashlib.sha256(bodies['raw-transcript.json']).hexdigest() and
            end['raw_native_output_sha256']==hashlib.sha256(bodies['raw-native-output.json']).hexdigest(),'raw receipt hash drift')
        fresh.require(start['original_job_sha256']==hashlib.sha256(fresh.canonical(job)).hexdigest(),'fresh job binding')
        ids.append(row['audio_id'])
    return {'calls':12,'completed':12,'audio_ids':ids}


def bind_fresh_api():
    old=ROOT/'research/qwen12-native-admission-v1'
    retained=load('fresh_retained_primitives',old/'campaign_contract.py')
    # Generic file primitives/resource constants only, not historical admission.
    for name in ('digest','identity','read_json','write_once','atomic','LIMITS'):
        setattr(fresh,name,getattr(retained,name))
    fresh.BOUNDARY={'scope':fresh.CAMPAIGN,'training_admitted':False,'human_gold':False,
        'qualification_allowed':False,'evaluation_gold_allowed':False,'historical_outputs_reused':False}
    fresh.current_family_counts=lambda job,output,family,isolated=False:verify_completed(output,job,family)
    sys.modules['campaign_contract']=fresh
    sys.path.insert(0,str(old))
    native=load('fresh_reused_native_functions',old/'native_worker.py')
    fresh.require(native.planned_clips is fresh.planned_clips and native.validate_original_job is fresh.validate_original_job,
        'fresh API injection failed')
    return native


def main(argv=None):
    args=sys.argv[1:] if argv is None else argv
    fresh.require(len(args)==2 and args[0] in fresh.FAMILIES and args[1] in ('prepare','decode'),'fixed fresh family/mode')
    family,mode=args;runtime=Path('/runtime');inputs=Path('/input');output=Path('/output')
    fresh.require(re.fullmatch('[0-9a-f]{40}',os.environ.get('KWS_EXECUTED_HEAD_SHA','')) and
        re.fullmatch('[1-9][0-9]{0,19}',os.environ.get('KWS_RUN_ID','')),'real run provenance')
    native=bind_fresh_api()
    native._FAILURE_FAMILY=family;native._FAILURE_MODE=mode
    try:
        scope=native.module('qwen12_kernel_scope',ROOT/'research/qwen12-native-admission-v1/runtime_scope.py').verify_runtime_scope()
        fresh.require(output.is_dir() and not list(output.iterdir()),'fresh isolated output')
        guard=native.install_inference_guard(family)
        job=fresh.validate_blind(inputs)
        for name,value in [('run-identity.json',{'schema':'qwen12-real-actions-identity-v1',
                'run_id':int(os.environ['KWS_RUN_ID']),'executed_head_sha':os.environ['KWS_EXECUTED_HEAD_SHA']}),
                ('kernel-scope.json',scope),('cpu-process-guard.json',guard),('blind-job.json',job)]:
            fresh.write_once(output/name,value)
        native.progress(output,'load',0);signal.signal(signal.SIGALRM,native.alarm)
        if family=='whisper':native.whisper(runtime,inputs,output,job,mode)
        else:native.qwen_sense(family,runtime,inputs,output,job,mode)
        if mode=='prepare':
            fresh.require(not list(output.glob('clip-*')),'prepare performed a recognition attempt')
            result={'calls':0,'completed':0,'audio_ids':[],'status':'prepared_no_recognition'}
        else:result={**verify_completed(output,job,family),'status':'all12_fresh_native_complete'}
        fresh.write_once(output/'worker-complete.json',{**result,**fresh.BOUNDARY})
    except BaseException as error:
        native.write_failure(error,output)
        return 1
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0


if __name__=='__main__':raise SystemExit(main())
