"""One prospective public Qwen12 capture; no TTS, training or old execute/main.

Only the fixed reviewed branch with a newly created permanent tag claim may
enter, including its first activation by fast-forward after source-only preparation. Source-only PR checks never call execute. All three native configurations
are observed and frozen before any audio forward. Existing CPU/kernel/acquisition
limits are reused, not weakened. Failed slots are consumed; no retry/resume path.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import fresh_contract as fresh
PIPELINE_SHA='702583717af435d99a2aa5810aa504355a4ab8c7'
EXECUTION_BRANCH='research/qwen12-native-capture-once-20261031-v1'
CLAIM_REF='refs/tags/qwen12-native-capture-once-20261031-v1-consumed'
REPO='jiying2007/kws-data'
MIB=1024**2
FROZEN_SOURCE_SHA256='b2e6101faf3653960c40047e5c5e20f7bc851eac61cc56ef2e9c27ea19590342'
# Explicit original bytes retained at PIPELINE_SHA. Never rewrite the historical
# freeze, replace a current checkout file, or revive a historical execution.
SOURCE_ARCHIVES={
    'research/qwen6_asr/core/asr_stage/adapters.py':
        'research/consolidation/archive/source-revisions/sense-runtime-binding-20261017/research/qwen6_asr/core/asr_stage/adapters.py',
    'research/qwen6_asr/core/asr_stage/execution.py':
        'research/consolidation/archive/source-revisions/sense-runtime-binding-20261017/research/qwen6_asr/core/asr_stage/execution.py',
    'research/qwen6_asr/core/sense_binding.py':
        'research/consolidation/archive/source-revisions/sense-runtime-binding-20261017/research/qwen6_asr/core/sense_binding.py',
    'research/qwen6_asr/core/vendor/qwen_loading_gate_v2.py':
        'research/consolidation/archive/source-revisions/prepare-r1-hotpath-20261012/qwen_loading_gate_v2.py',
    'research/source-screen32-v1/asr_worker.py':
        'research/consolidation/archive/source-revisions/prepare-r1-hotpath-20261012/asr_worker.py',
}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],timeout=30).decode().strip()


def validate_execution(env,event,claim):
    fresh.require(env.get('GITHUB_REPOSITORY')==REPO and env.get('GITHUB_EVENT_NAME')=='push' and
        env.get('GITHUB_REF')=='refs/heads/'+EXECUTION_BRANCH and env.get('GITHUB_RUN_ATTEMPT')=='1', 'fixed single-claim push scope')
    head=env.get('GITHUB_SHA','');run=env.get('GITHUB_RUN_ID','')
    fresh.require(re.fullmatch('[0-9a-f]{40}',head) and re.fullmatch('[1-9][0-9]{0,19}',run),'actual source/run identity')
    fresh.require(type(event.get('created')) is bool and event.get('deleted') is False and event.get('after')==head and
        event.get('repository',{}).get('full_name')==REPO and event.get('repository',{}).get('private') is False,'public branch push event')
    fresh.require(claim.get('ref')==CLAIM_REF and claim.get('object',{}).get('sha')==head and
        claim['object'].get('type')=='commit','separate permanent tag creation required')
    return head,run


def source_members(pipeline,freeze,profile):
    """Read exact dependency bytes; no import, execution or checkout mutation."""
    fresh.require(profile in ('setup_code','inference_code'),'fixed source profile')
    raw=fresh.bounded_read(pipeline/'research/qwen12-native-admission-v1/execution-freeze.json',MIB)
    fresh.require(hashlib.sha256(raw).hexdigest()==FROZEN_SOURCE_SHA256 and
        fresh.canonical(freeze)==fresh.canonical(json.loads(raw)),'original source freeze required')
    members={}
    for name,identity in freeze[profile].items():
        path=Path(name)
        fresh.require(not path.is_absolute() and '..' not in path.parts,'fixed source path')
        body=fresh.bounded_read(pipeline/SOURCE_ARCHIVES.get(name,name),8*MIB)
        fresh.require(len(body)==identity['bytes'] and hashlib.sha256(body).hexdigest()==identity['sha256'],'upstream source drift')
        members[name]=body
    # The outer freeze and the worker's internal dependency map must agree.
    # Inspect literals with AST, never import historical recognition code here.
    tree=ast.parse(members['research/source-screen32-v1/asr_worker.py'])
    assignments=[n for n in tree.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='RETAINED_PINS' for t in n.targets)]
    fresh.require(len(assignments)==1,'one retained dependency map required')
    pins=ast.literal_eval(assignments[0].value)
    fresh.require(type(pins) is dict and bool(pins),'nonempty retained dependency map')
    for name,digest in pins.items():
        full='research/qwen6_asr/'+name
        fresh.require(full in members and hashlib.sha256(members[full]).hexdigest()==digest,'embedded dependency pin mismatch')
    return members


def stage_sources(pipeline,freeze,profile,destination):
    members=source_members(pipeline,freeze,profile)
    destination.mkdir()
    for name,body in members.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
    new=destination/'research'/fresh.CAMPAIGN;new.mkdir(parents=True)
    # Copies are explicit staged aliases, not edits to retained repository bytes.
    aliases={'native_worker.py':HERE/'worker.py','fresh_contract.py':HERE/'fresh_contract.py'}
    for name in ('cpu_gate.py','setup_adapter.py','runtime_scope.py','campaign_contract.py'):
        aliases[name]=pipeline/'research/qwen12-native-admission-v1'/name
    bindings={}
    for name,source in aliases.items():
        body=fresh.bounded_read(source,1024**2);(new/name).write_bytes(body)
        bindings['research/'+fresh.CAMPAIGN+'/'+name]={'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
    gate=destination/'research/source-screen32-whisper-v1/attempt-3/cpu_gate.py'
    if profile=='setup_code':
        gate.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(new/'cpu_gate.py',gate)
    return bindings


def retain(root,public,receipt):
    """Publish a verified snapshot, preserving native output/<family> references."""
    fresh.require(not public.exists() and not public.is_symlink(),'new artifact destination required')
    names=('host-receipt.json','terminal-outcomes.json','image-transfer.json','cpu-receipt.json',
        'cleanup-receipt.json','supervision-receipt.json','asr-setup-public.json','staged-source.json')
    try:
        with tempfile.TemporaryDirectory(prefix='.capture-public-',dir=public.parent) as temp:
            staged=Path(temp)/'artifact';staged.mkdir()
            paths=[]
            for name in names:
                source=root/name
                if source.exists() or source.is_symlink():
                    fresh.require(source.is_file() and not source.is_symlink(),'regular host receipt required')
                    paths.append(source)
            for name in ('frozen','output','policy','whisper-setup'):
                directory=root/name
                if directory.exists() or directory.is_symlink():
                    fresh.require(directory.is_dir() and not directory.is_symlink(),'regular output directory required')
                    for path in sorted(directory.rglob('*')):
                        fresh.require(not path.is_symlink() and (path.is_dir() or path.is_file()),'regular evidence member required')
                        if path.is_file():paths.append(path)
            total=0;members={}
            for source in paths:
                body=fresh.bounded_read(source,32*MIB);total+=len(body)
                fresh.require(total<=256*MIB,'public raw evidence cap')
                name=source.relative_to(root).as_posix()
                fresh.require(name not in members,'duplicate native member')
                target=staged/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
                fresh.require(target.read_bytes()==body,'copied member changed')
                members[name]={'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
            manifest={'schema':'fresh-qwen12-native-artifact-v1','files':members,'training_admitted':False,
                'qualification_allowed':False,'raw_bytes':total,'status':receipt['status']}
            (staged/'artifact-freeze.json').write_bytes(fresh.canonical(manifest))
            staged.rename(public)
    except BaseException as error:
        # No partial archive is promoted. Do not remove or overwrite a destination
        # created by another caller; only a new directory receives this failure.
        if not public.exists() and not public.is_symlink():
            public.mkdir()
            (public/'publication-failure.json').write_bytes(fresh.canonical({'status':'PUBLICATION_FAILED',
                'error_type':type(error).__name__,'training_admitted':False,'model_outcome':receipt.get('status')}))
        raise


def supervise_shared_setup(host,original,docker,code,runtime,inputs,output,deadline):
    """Use the intact shared setup package, never its historical capture entry."""
    previous=host.EXPERIMENT
    host.EXPERIMENT='qwen12-native-admission-v1'
    try:
        return host.supervise(original,docker,code,runtime,inputs,output,'setup',deadline)
    finally:
        host.EXPERIMENT=previous



def capture_layout(root):
    """Use the retained monitor's exact writable and frozen directory contract."""
    return root/'output', root/'frozen', root/'policy'


def preflight_layout(docker, root, limit):
    """Exercise the real filesystem sampler before any image/runtime acquisition."""
    output, frozen, policy = capture_layout(root)
    for directory, scope in ((output, 'public_output'), (frozen, 'public_frozen'),
                             (policy, 'public_policy')):
        fresh.require(directory.is_dir() and not directory.is_symlink() and
                      not any(directory.iterdir()), 'empty fresh phase directory required')
        docker.sample_bytes(directory, scope, limit)
    for family in fresh.FAMILIES:
        path = output/family
        path.mkdir()
        try:
            docker.sample_bytes(path, 'phase_output', limit)
        finally:
            # Remove only the empty directory this preflight just created.
            path.rmdir()


def freeze_prepared(output, frozen):
    """Move all three completed preparation snapshots; decode gets new outputs."""
    fresh.require(output.is_dir() and not output.is_symlink() and
                  frozen.is_dir() and not frozen.is_symlink() and
                  not any(frozen.iterdir()), 'new frozen destination required')
    fresh.require({p.name for p in output.iterdir()} == set(fresh.FAMILIES),
                  'complete three-family preparation roster required')
    for family in fresh.FAMILIES:
        fresh.require((output/family).is_dir() and not (output/family).is_symlink(),
                      'regular prepared family required')
    for family in fresh.FAMILIES:
        (output/family).rename(frozen/family)


def execute(pipeline,claim_path):
    env=os.environ;event=json.loads(Path(env['GITHUB_EVENT_PATH']).read_bytes())
    head,run=validate_execution(env,event,json.loads(claim_path.read_bytes()))
    data=HERE.parents[1]
    fresh.require(git(data,'rev-parse','HEAD')==head and not git(data,'status','--porcelain'),'clean reviewed data head')
    fresh.require(git(pipeline,'rev-parse','HEAD')==PIPELINE_SHA and not git(pipeline,'status','--porcelain'),'clean pinned pipeline')
    root=Path(env['RUNNER_TEMP'])/(fresh.CAMPAIGN+'-'+run);root.mkdir()
    with open(env['GITHUB_OUTPUT'],'a') as output:output.write('evidence_root='+str(root/'artifact')+'\n')
    inputs=root/'input';decoded,prepared,policy=capture_layout(root)
    for p in (prepared,decoded,policy):p.mkdir()
    old=pipeline/'research/qwen12-native-admission-v1';sys.path.insert(0,str(old))
    host=load('fresh_capture_host_primitives',old/'hosted_run.py')
    # This changes only the library's worker path/name selection. Its execute/main
    # and historical input/admission functions are NEVER invoked.
    host.EXPERIMENT=fresh.CAMPAIGN
    previous=pipeline/'research/source-screen32-v1';sys.path.insert(0,str(previous))
    original=load('fresh_scope_constructors',previous/'hosted_run.py')
    started=time.monotonic();docker=host.MeteredDocker(root,started+3600)
    receipt={'schema':'fresh-qwen12-native-host-v1','status':'started','head_sha':head,'run_id':run,
        'pipeline_sha':PIPELINE_SHA,'campaign':fresh.CAMPAIGN,'maximum_calls':36,'maximum_retries':0,
        'training_admitted':False,'qualification_allowed':False,'new_tts_calls':0,'stages':{}}
    failed=False;job=fresh.original_job();stage='input'
    def save():host.atomic(root/'host-receipt.json',receipt)
    def interrupted(*_):raise TimeoutError('host 3600-second wall ceiling')
    signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    signal.setitimer(signal.ITIMER_REAL,3600);save()
    try:
        stage='layout_preflight'
        preflight_layout(docker,root,host.LIMITS['output_bytes'])
        stage='input'
        helper=load('fresh_archived_input',data/'research/qwen16-label-inputs-20261030-v1/prepare.py')
        helper.materialize(data/helper.ARCHIVE_PATH,root/'handoff')
        shutil.copytree(root/'handoff/blind',inputs);fresh.validate_blind(inputs)
        freeze=json.loads((old/'execution-freeze.json').read_bytes())
        code=root/'code';setup_code=root/'setup-code'
        binding={'inference':stage_sources(pipeline,freeze,'inference_code',code),
                 'setup':stage_sources(pipeline,freeze,'setup_code',setup_code)}
        host.write_once(root/'staged-source.json',{'pipeline_sha':PIPELINE_SHA,'data_head':head,'aliases':binding,
            'source_selection':{'freeze_sha256':FROZEN_SOURCE_SHA256,'original_archive_paths':SOURCE_ARCHIVES}})
        receipt['capacity']=original.host_capacity()
        fresh.require(shutil.disk_usage(root).free>=12*1024**3,'12GiB initial free disk')
        stage='image';original.pull_image(json.loads((previous/'container-lock.json').read_bytes()),root/'image-transfer.json')
        asr=root/'asr-runtime';asr.mkdir();out=root/'asr-setup';out.mkdir()
        stage='asr_setup';before=time.monotonic();receipt['stages'][stage]='started';save()
        supervise_shared_setup(host,original,docker,setup_code,asr,inputs,out,min(docker.deadline,before+1800))
        setup_seconds=time.monotonic()-before;receipt['stages'][stage]='complete'
        host.asr_setup_projection(asr,root/'asr-setup-public.json')
        third=pipeline/'research/source-screen32-whisper-v1/attempt-3';sys.path.insert(0,str(third))
        bounded=load('fresh_bounded_whisper_setup',third/'bounded_host.py')
        whisper=root/'whisper-runtime';whisper.mkdir();out=root/'whisper-setup';out.mkdir()
        stage='whisper_setup';before=time.monotonic();receipt['stages'][stage]='started';save()
        fresh.require(setup_seconds<1800,'remaining combined setup budget')
        host.whisper_setup(bounded,docker,setup_code,whisper,out,root,run,min(docker.deadline,before+1800-setup_seconds))
        setup_seconds+=time.monotonic()-before;receipt['stages'][stage]='complete'
        runtimes={'qwen06':asr,'sensevoice':asr,'whisper':whisper}
        def prepare(family,seconds):
            nonlocal stage
            stage=family+'_prepare';out=decoded/family;out.mkdir();docker.stage='setup';receipt['stages'][stage]='started';save()
            host.supervise(original,docker,code,runtimes[family],inputs,out,family,min(docker.deadline,time.monotonic()+seconds),mode='prepare')
            result=host.read_json(out/'worker-complete.json')
            fresh.require(result['calls']==0 and type(result['calls']) is int and not list(out.glob('clip-*')),'prepare must not recognize audio')
            receipt['stages'][stage]='complete'
        def freeze_policy():
            nonlocal stage
            stage='policy_freeze'
            # build_policy consumes only the already-observed three native configs.
            sys.path.insert(0,str(old))
            receipts=load('fresh_host_native_policy',old/'worker_receipts.py')
            value=receipts.build_policy({f:decoded/f for f in fresh.FAMILIES})
            host.write_once(policy/'policy.json',value)
            receipt['policy_sha256']=hashlib.sha256((policy/'policy.json').read_bytes()).hexdigest()
            freeze_prepared(decoded,prepared)
            receipt['all_three_prepared_before_decode']=True;save()
        def decode(family,seconds):
            nonlocal stage
            stage=family+'_decode';out=decoded/family;out.mkdir();docker.stage=family;receipt['stages'][stage]='started';save()
            host.supervise(original,docker,code,runtimes[family],inputs,out,family,min(docker.deadline,time.monotonic()+seconds),
                mode='decode',frozen=prepared/family,policy=policy)
            worker=load('fresh_host_result_verifier',HERE/'worker.py')
            worker.verify_completed(out,job,family,expected_run=run,expected_head=head);receipt['stages'][stage]='complete'
        host.recognition_stages(prepare,freeze_policy,decode,setup_seconds=setup_seconds,record=receipt)
        receipt['status']='36_NATIVE_CAPTURES_COMPLETE_LABEL_ADMISSION_PENDING'
    except BaseException as error:
        failed=True;receipt.update(status='STOPPED_NO_RETRY',failure_stage=stage,error_type=type(error).__name__)
        if docker.supervision_failure is not None:receipt['supervision_failure']=docker.supervision_failure[1]
    finally:
        try:docker.finish()
        except BaseException as error:failed=True;receipt.update(status='CLEANUP_OR_CPU_ACCOUNTING_FAILED',cleanup_error_type=type(error).__name__)
        outcomes=[]
        for row in fresh.remaining_plan(job):
            slot=decoded/row['family']/row['audio_id']
            state='NOT_RUN'
            if (slot/'started.json').exists():state='STARTED_FAILED_OR_INCOMPLETE'
            if (slot/'completed.json').exists():state='COMPLETION_RECORDED_PENDING_INDEPENDENT_RECHECK'
            outcomes.append({**row,'status':state})
        host.write_once(root/'terminal-outcomes.json',outcomes)
        receipt.update(wall_seconds=time.monotonic()-started,supervision=docker.supervisions,cleanup=docker.cleanup)
        save();signal.setitimer(signal.ITIMER_REAL,0)
        retain(root,root/'artifact',receipt)
    return 1 if failed else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',type=Path,required=True);parser.add_argument('--claim',type=Path,required=True)
    args=parser.parse_args()
    try:raise SystemExit(execute(args.pipeline.resolve(),args.claim.resolve()))
    except Exception as error:
        print(json.dumps({'status':'FAILED_NO_RETRY','error_type':type(error).__name__}),file=sys.stderr)
        raise SystemExit(1)
