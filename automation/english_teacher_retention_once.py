"""One predeclared train-only teacher-retention experiment, not a PR186 retry.

The consumed incremental worker and its rejection remain unchanged. This file
imports its exact input authentication and replay helpers, but owns a separate
plan, execution identity, supervisor, training function and output directory.
Default is source-only. No local numeric execution is permitted.
"""
from __future__ import annotations
import argparse
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPO='jiying2007/kws-data'
SCHEMA='english-train-teacher-retention-once-v1'
WORKER_PATH='automation/english_teacher_retention_once.py'
PLAN_PATH='automation/english_teacher_retention_plan.json'
WORKFLOW_PATH='.github/workflows/english-teacher-retention-once.yml'
HELPER_PATH='automation/english_hardneg_increment_once.py'
HELPER_PIN={'bytes':69681,'sha256':'ac89f4564bb3f93661beb02f01f7766616322a39096c94a4bdac84560f5a9ec1'}
ORIGINAL_PLAN_PATH='automation/english_hardneg_increment_plan.json'
DISTILLATION={'temperature':2.0,'weight':10.0,'teacher':'FROZEN_COVERAGE_HEAD',
    'teacher_unique_train_bags':225,'teacher_head_windows':8370,'teacher_calls_per_unique_bag':1,
    'teacher_role':'train','teacher_conditions':['clean','snr10','new5','new15'],
    'teacher_on_development_or_snr5_new8':False,'teacher_on_reviewed_negatives':False,
    'teacher_parameter_updates':0,'student_extra_forwards':0,
    'direction':'KL_TEACHER_TO_STUDENT','aggregation':'MEAN_WINDOWS_THEN_MEAN_SIX_ORIGINAL_BAGS',
    'objective':'MEAN_SEVEN_ORIGINAL_SUPERVISED_TERMS_PLUS_10_TIMES_4_TIMES_MEAN_SIX_TEMPERATURE2_KL',
    'cached_before_first_student_update':True,'parameter_sweep':False}

def get_helper():
    path=ROOT/HELPER_PATH
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('REGULAR_IMMUTABLE_HELPER')
    raw=path.read_bytes()
    if {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}!=HELPER_PIN:
        raise ValueError('EXACT_IMMUTABLE_HELPER')
    spec=importlib.util.spec_from_file_location('teacher_retention_frozen_helper',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

frozen=get_helper()
need=frozen.need
sha=frozen.sha
identity=frozen.identity
file_identity=frozen.file_identity
regular=frozen.regular
canonical=frozen.canonical
loads=frozen.loads
put=frozen.put
once=frozen.once
load_module=frozen.load_module
checkpoint=frozen.checkpoint
restore_streams=frozen.restore_streams
fixed_draws=frozen.fixed_draws
verify_saved_npz_bin=frozen.verify_saved_npz_bin
summarize_case=frozen.summarize_case
paired_case_gate=frozen.paired_case_gate
reviewed_fit=frozen.reviewed_fit
native_parity=frozen.native_parity
prepare_saved_inputs=frozen.prepare_saved_inputs
prepare_dependencies=frozen.prepare_dependencies
Runner=frozen.Runner
cleanup_group=frozen.cleanup_group
close_interrupted_receipts=frozen.close_interrupted_receipts
SOURCES=frozen.SOURCES
BASE_NPZ=frozen.BASE_NPZ
BASE_BIN=frozen.BASE_BIN
ROSTER_PATH=frozen.ROSTER_PATH
FROZEN_PARSER=frozen.FROZEN_PARSER
FALSE_FLAGS=frozen.FALSE_FLAGS
CONFIG=frozen.CONFIG
RECIPE={**frozen.RECIPE,'distillation':DISTILLATION}


def expected_plan(worker_sha,source_pins,main_before=None,enabled=False):
    return {'schema':SCHEMA,'repository':REPO,'enabled':enabled,'main_before':main_before,
        'worker_sha256':worker_sha,'source_pins':source_pins,'sources':SOURCES,
        'base_npz':BASE_NPZ,'base_bin':BASE_BIN,'configuration':CONFIG,'recipe':RECIPE,
        'rejected_predecessor':{'run_id':44076645690,'head_sha':'df4f870baf178dcf0121fca661b5692ebbd5cca9',
            'status':'REJECTED_RESEARCH_REGRESSION_GATE','rerun':False},
        'maximum_execution_seconds':1200,'numeric_processes':1,'native_head_processes':1,
        'new_optimizer_updates':96,'student_training_bag_forwards':672,'teacher_train_bag_forwards':225,
        'teacher_head_windows':8370,'teacher_development_forwards':0,'teacher_reviewed_negative_forwards':0,
        'candidate_head_cases':323,'candidate_embedding_rows':18982,'candidate_ready_rows':14137,
        'new_frontend_calls':0,'new_audio_decodes':0,'new_compilations':0,'new_tts_asr_calls':0,
        'new_audio_mixes':0,'checkpoint_selection':False,'automatic_retrain_after_gate_failure':False,
        'threshold_changed':False,'research_gates_changed':False,'retry_count':0,
        'arm':'NOT_RUN','new_full_pcm_chain':'NOT_RUN','ssc305':'NOT_RUN',
        'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED',
        'scope':'REVIEWED_NEGATIVE_HEAD_FIT_WITH_TRAIN_ONLY_FUNCTIONAL_RETENTION',
        **dict.fromkeys(FALSE_FLAGS,False)}


def source_preflight(root,plan):
    original=loads(regular(root/ORIGINAL_PLAN_PATH))
    admission=frozen.source_preflight(root,original)
    source_pins={**original['source_pins'],
        HELPER_PATH:HELPER_PIN,ORIGINAL_PLAN_PATH:identity(regular(root/ORIGINAL_PLAN_PATH)),
        WORKFLOW_PATH:identity(regular(root/WORKFLOW_PATH))}
    expected=expected_plan(sha(regular(root/WORKER_PATH)),source_pins,plan.get('main_before'),plan.get('enabled'))
    need(canonical(plan)==canonical(expected),'EXACT_NEW_TEACHER_PLAN')
    need(type(plan['enabled']) is bool and (re.fullmatch('[0-9a-f]{40}',plan['main_before'] or '')
        if plan['enabled'] else plan['main_before'] is None),'NEW_PLAN_ENABLE_BINDING')
    workflow_contract(regular(root/WORKFLOW_PATH).decode('utf-8'),plan)
    return admission


def workflow_contract(source,plan):
    blocks=re.findall(r'(?ms)^  retention-training:\n(.*?)(?=^  [a-z][a-z-]*:|\Z)',source)
    need(len(blocks)==1,'ONE_NEW_WORKFLOW_EXECUTION_JOB')
    job=blocks[0]
    conditions=re.findall(r'(?m)^    if: >-\n((?:      [^\n]*\n)+)',job)
    need(len(conditions)==1,'ONE_NEW_EXECUTION_CONDITION')
    actual=' '.join(conditions[0].split())
    before=plan['main_before'] if plan['enabled'] else 'DISABLED_PENDING_NEW_SOURCE_REVIEW'
    expected=('' if plan['enabled'] else 'false && ')+(
        "github.repository == 'jiying2007/kws-data' && github.event.repository.private == false && "
        "github.event_name == 'push' && github.ref == 'refs/heads/main' && github.run_attempt == 1 && "
        "github.event.before == '"+before+"'")
    need(actual==expected,'WORKFLOW_EXACT_PLAN_MAIN_BEFORE')
    need(re.findall(r'(?m)^          include-hidden-files: ([^\n]+)$',job)==['true'],
         'MANIFEST_HIDDEN_SOURCE_FILES_RETAINED')
    need("      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1\n"
         "        with:\n          python-version: '3.12'\n" in job,'PINNED_PYTHON312_SETUP')
    need("          NUMEXPR_NUM_THREADS: '1'\n" in job and
         "          PYTHONHASHSEED: '20261117'\n" in job,'FROZEN_NUMERIC_ENVIRONMENT')


def partial_training_receipt(progress,losses,supervised_losses,distillation_losses,draws):
    return {'progress':dict(progress),'losses':losses,'supervised_losses':supervised_losses,
        'distillation_losses':distillation_losses,'draws':draws,'retry_count':0,
        'teacher_train_bag_forwards_started':progress['teacher_train_forwards_started'],
        'teacher_train_bag_forwards_completed':progress['teacher_train_forwards_completed'],
        'teacher_train_head_windows_completed':progress['teacher_train_head_windows'],
        'planned_teacher_train_bag_forwards':225,'planned_teacher_train_head_windows':8370}


def execution_identity(plan,env=None,event=None,head=None,parent=None):
    env=os.environ if env is None else env
    need(plan['enabled'] is True and env.get('GITHUB_ACTIONS')=='true' and env.get('GITHUB_REPOSITORY')==REPO
         and env.get('GITHUB_EVENT_NAME')=='push' and env.get('GITHUB_REF')=='refs/heads/main'
         and env.get('GITHUB_RUN_ATTEMPT')=='1','NEW_PUBLIC_MAIN_ONCE_ONLY')
    commit,run_id=env.get('GITHUB_SHA',''),env.get('GITHUB_RUN_ID','')
    need(re.fullmatch('[0-9a-f]{40}',commit) is not None and run_id.isdigit(),'ACTUAL_NEW_RUN_HEAD')
    if event is None:event=loads(regular(env['GITHUB_EVENT_PATH']))
    need(event.get('before')==plan['main_before'] and event.get('after')==commit
         and event.get('ref')=='refs/heads/main' and event.get('repository',{}).get('private') is False
         and event.get('repository',{}).get('full_name')==REPO and event.get('deleted') is False
         and event.get('forced') is False,'EXACT_NEW_PUBLIC_SUCCESSOR')
    if head is None:head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,timeout=10).strip()
    if parent is None:parent=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD^1'],text=True,timeout=10).strip()
    need(head==commit and parent==plan['main_before'],'NEW_WORKTREE_EXACT_SUCCESSOR')
    return {'repository':REPO,'run_id':run_id,'head_sha':commit,'main_before':plan['main_before'],
        'run_attempt':1,'event':'push','public_repository':True}


def teacher_keys(geometry,all_draws):
    keys=sorted({(r['condition'],r['id']) for draw in all_draws for r in draw['original']})
    bykey={(r['condition'],r['id']):r for r in geometry}
    need(len(keys)==len(bykey)==225 and set(keys)==set(bykey),'ALL_225_ORIGINAL_TRAIN_BAGS')
    need(all(bykey[k]['role']=='train' and k[0] in DISTILLATION['teacher_conditions'] for k in keys),
         'TEACHER_TRAIN_ONLY_NO_REVIEWED_NO_DEVELOPMENT')
    need(sum(len(bykey[k]['head_instance_ends']) for k in keys)==8370,'EXACT_8370_TEACHER_WINDOWS')
    return keys


def result_boundary(binding):
    return {'schema':SCHEMA,**binding,'status':'FAILED_NO_RETRY','passed':False,'retry_count':0,
        'new_audio_decodes':0,'new_frontend_calls':0,'new_tts_asr_calls':0,'new_compilations':0,
        'human_review_scope':'NINE_WINDOW_PCM_IDENTITIES_ONLY','whole_utterance_labels_changed':0,
        'new_independent_speaker_count':0,'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED',
        'full_pcm_new_head':'NOT_RUN','arm_new_head':'NOT_RUN','ssc305':'NOT_RUN',
        'strong_english_fsmn_ctc':False,'teacher_retention_mechanism':DISTILLATION,'weak_original_positive_labels_preserved':True,
        'formal_quality_improvement_claim':False,**dict.fromkeys(FALSE_FLAGS,False)}

def numeric(args,plan):
    output=args.output.absolute();binding=execution_identity(plan)
    need(os.environ.get('KWS_RETENTION_NUMERIC_PARENT_PID')==str(os.getppid()) and
         os.getpgrp()==os.getppid(), 'NUMERIC_ONLY_IN_SUPERVISED_WORKER_GROUP')
    worker=loads(regular(output/'evidence/WORKER_STARTED.json'))
    need(worker['pid']==os.getppid() and worker['binding']==binding,'NUMERIC_PARENT_BINDING')
    need(not (output/'evidence/NUMERIC_STARTED.json').exists(),'NO_NUMERIC_REPLAY')
    once(output/'evidence/NUMERIC_STARTED.json',{'pid':os.getpid(),'parent_pid':os.getppid(),
        'attempt':1,'retry_count':0,'binding':binding,'recipe':RECIPE})
    progress={'optimizer_updates_started':0,'optimizer_updates_completed':0,'training_bag_forwards_started':0,
        'training_bag_forwards_completed':0,'candidate_head_cases_started':0,'candidate_head_cases_completed':0,
        'candidate_embedding_rows':0,'candidate_ready_rows':0,'teacher_train_forwards_started':0,'teacher_train_forwards_completed':0,
        'teacher_train_head_windows':0,'teacher_development_forwards':0,'teacher_reviewed_forwards':0,
        'new_frontend_calls':0,'new_audio_decodes':0}
    def ck():checkpoint(output/'evidence/NUMERIC_PROGRESS.json',progress)
    ck();losses=[];distillation_losses=[];supervised_losses=[];draws=[];outcome={'status':'NUMERIC_FAILED_NO_RETRY','passed':False}
    try:
        import numpy as np
        import torch
        need(np.__version__=='1.26.4' and torch.__version__=='2.9.1+cpu','EXACT_NUMERIC_RUNTIME')
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261117);torch.use_deterministic_algorithms(True)
        once(output/'evidence/RUNTIME.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
            'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'frontend_runtime_imported':False})
        portable=load_module(output/'inputs/coverage/sources/portable_head.py','retention_portable_head')
        baseline=portable.load_weights(output/'inputs/coverage/models/coverage-head.npz')
        need(identity(portable.export_blob(baseline))==BASE_BIN,'LOADED_BASE_NPZ_BIN')
        allrows,streams,evaluation,geometry,previous_training=restore_streams(output,np)
        roster=loads(regular(output/'source'/ROSTER_PATH));selection=loads(regular(output/'evidence/ADMITTED_HEAD_INPUTS.json'))
        parser=load_module(output/'source'/FROZEN_PARSER,'retention_numeric_trace_parser')
        old_natural={};natural_order=[]
        for row in roster['rows']:
            rid=row['recording_id'];parsed=parser.parse_trace(regular(output/'inputs/natural/traces'/(rid+'.trace')),
                {'samples':row['frames'],'records':row['expected_embedding_rows'],'ready':row['expected_ready_rows']})
            key=('natural',rid);streams[key]=(np.asarray(parsed['embeddings'],dtype=np.float32),[r['sample'] for r in parsed['scores']])
            allrows[key]={'id':rid,'role':'train','weak_class':None,'observed_samples':row['frames'],'source':row}
            old_natural[rid]=parsed['scores'];natural_order.append(key)
        prepared={}
        def make_windows(key,ends):
            matrix,times=streams[key];positions={t:i for i,t in enumerate(times)}
            need(all(t in positions and positions[t]>=15 for t in ends),'COMPLETE_ORIGINAL_HISTORY')
            windows=np.asarray([matrix[positions[t]-15:positions[t]+1] for t in ends],dtype=np.float32)
            return torch.from_numpy(((windows-baseline['mean96'])/baseline['scale96']).reshape(len(windows),1536))
        for g in geometry:prepared[(g['condition'],g['id'])]=make_windows((g['condition'],g['id']),g['head_instance_ends'])
        for s in selection['sources']:prepared[('reviewed',s['recording_id'])]=make_windows(('natural',s['recording_id']),s['head_ends'])
        all_draws=fixed_draws(previous_training['draws'],[s['recording_id'] for s in selection['sources']])
        once(output/'evidence/TRAINING_DRAWS.json',{'draws':all_draws,'recipe':RECIPE,'source_order':[s['recording_id'] for s in selection['sources']]})
        once(output/'evidence/TRAINING_GEOMETRY.json',{'original_bags':geometry,'reviewed':selection,
            'original_positive_label_kind':'WEAK_CLIP_INTENT','all_original_roles_unchanged':True})
        model=torch.nn.Sequential(torch.nn.Linear(1536,32),torch.nn.ReLU(),torch.nn.Linear(32,3))
        need(sum(p.numel() for p in model.parameters())==49283,'UNCHANGED_HEAD_PARAMETER_COUNT')
        with torch.no_grad():
            for parameter,key in ((model[0].weight,'w1'),(model[0].bias,'b1'),(model[2].weight,'w2'),(model[2].bias,'b2')):
                parameter.copy_(torch.from_numpy(baseline[key]))
        # The frozen teacher is called exactly once for each scheduled original
        # training bag before any student optimizer update. No reviewed natural
        # bag or exposed development condition is a teacher input.
        teacher_cache={};teacher_order=teacher_keys(geometry,all_draws);teacher_manifest=[]
        teacher=torch.nn.Sequential(torch.nn.Linear(1536,32),torch.nn.ReLU(),torch.nn.Linear(32,3))
        teacher.load_state_dict(model.state_dict());teacher.eval();teacher.requires_grad_(False)
        teacher_parameter_pins={name:identity(value.detach().numpy().tobytes()) for name,value in teacher.state_dict().items()}
        with torch.no_grad():
            for index,key in enumerate(teacher_order):
                need(progress['optimizer_updates_started']==0 and progress['teacher_train_forwards_started']<225,
                     'TEACHER_ONLY_BEFORE_STUDENT_UPDATES')
                progress['teacher_train_forwards_started']+=1;ck()
                logits=teacher(prepared[key]);logp=torch.log_softmax(logits/2.0,dim=-1).detach()
                need(bool(torch.isfinite(logp).all()) and not logp.requires_grad,'FINITE_DETACHED_TEACHER')
                teacher_cache[key]=logp
                raw=logp.numpy().astype('<f4',copy=False).tobytes()
                member=f'cache/teacher-logp-{index:03d}.f32le';put(output/member,raw)
                teacher_manifest.append({'condition':key[0],'id':key[1],'role':'train','member':member,
                    'shape':list(logp.shape),'identity':identity(raw),'temperature':2.0})
                progress['teacher_train_forwards_completed']+=1
                progress['teacher_train_head_windows']+=len(logp);ck()
        need(progress['teacher_train_forwards_completed']==225 and progress['teacher_train_head_windows']==8370,
             'EXACT_TRAIN_TEACHER_CACHE_COMPLETE')
        once(output/'evidence/TEACHER_CACHE.json',{'base_npz':BASE_NPZ,'base_bin':BASE_BIN,'recipe':DISTILLATION,
            'bags':teacher_manifest,'parameter_pins':teacher_parameter_pins,'model_calls':225,
            'head_windows':8370,'student_updates_before_freeze':0,'development_calls':0,'reviewed_calls':0})
        need({name:identity(value.detach().numpy().tobytes()) for name,value in teacher.state_dict().items()}
             ==teacher_parameter_pins,'FROZEN_TEACHER_PARAMETERS_UNCHANGED')
        del teacher
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=0.0001)
        need(len(optimizer.state)==0,'ADAMW_STATE_RESET')
        for draw in all_draws:
            need(progress['optimizer_updates_started']<96,'ONE_FIXED_TRAINING_BUDGET')
            progress['optimizer_updates_started']+=1;ck();optimizer.zero_grad(set_to_none=True);terms=[];distill_terms=[]
            uses=[((r['condition'],r['id']),r['weak_class']) for r in draw['original']]
            uses.append((('reviewed',draw['reviewed_source']),0))
            for key,label in uses:
                need(progress['training_bag_forwards_started']<672,'EXACT_SEVEN_BAG_BUDGET')
                progress['training_bag_forwards_started']+=1;ck()
                logits=model(prepared[key])
                logp=torch.log_softmax(logits,dim=-1)
                if key[0]!='reviewed':
                    need(key in teacher_cache,'EXACT_ORIGINAL_TRAIN_TEACHER_KEY')
                    student_logp=torch.log_softmax(logits/2.0,dim=-1)
                    teacher_logp=teacher_cache[key]
                    # Per-window KL, then mean within this original bag.
                    distill_terms.append((teacher_logp.exp()*(teacher_logp-student_logp)).sum(dim=-1).mean())
                progress['training_bag_forwards_completed']+=1;ck()
                terms.append(-logp[:,0].mean() if label==0 else math.log(len(logp))-torch.logsumexp(logp[:,label],dim=0))
            need(len(terms)==7 and len(distill_terms)==6,'UNCHANGED_SEVEN_TERMS_AND_SIX_RETENTION_TERMS')
            supervised_loss=torch.stack(terms).mean()
            distillation_loss=4.0*torch.stack(distill_terms).mean()
            loss=supervised_loss+10.0*distillation_loss
            need(bool(torch.isfinite(loss)),'FINITE_TEACHER_RETENTION_LOSS')
            supervised_losses.append(float(supervised_loss.detach()))
            distillation_losses.append(float(distillation_loss.detach()))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()));draws.append(draw);progress['optimizer_updates_completed']+=1;ck()
        weights={'w1':model[0].weight.detach().numpy(),'b1':model[0].bias.detach().numpy(),
            'w2':model[2].weight.detach().numpy(),'b2':model[2].bias.detach().numpy(),
            'mean96':baseline['mean96'].copy(),'scale96':baseline['scale96'].copy()}
        (output/'models').mkdir(exist_ok=True)
        np.savez(output/'models/retention-head.npz',**weights)
        candidate=portable.load_weights(output/'models/retention-head.npz')
        binary=portable.export_blob(candidate);put(output/'models/retention-head.bin',binary)
        array_pins=verify_saved_npz_bin(regular(output/'models/retention-head.npz'),binary)
        need(all(candidate[k].tobytes()==baseline[k].tobytes() for k in ('mean96','scale96')),'NORMALIZATION_BYTE_IDENTICAL')
        need(any(candidate[k].tobytes()!=baseline[k].tobytes() for k in ('w1','b1','w2','b2')),'ACTUAL_UPDATED_WEIGHTS')
        frozen={'npz':file_identity(output/'models/retention-head.npz'),'bin':identity(binary),'arrays':array_pins}
        once(output/'evidence/CANDIDATE_FROZEN.json',{**frozen,'progress':dict(progress),
            'evaluation_model_calls_before_freeze':0,'checkpoint_selection':False,'all_later_evaluation_uses_frozen_weights':True})
        once(output/'evidence/TRAINING.json',{'recipe':RECIPE,'base_npz':BASE_NPZ,'base_bin':BASE_BIN,
            'candidate':frozen,'optimizer_state_reset':True,'optimizer_updates':96,'training_bag_forwards':672,
            'losses':losses,'supervised_losses':supervised_losses,'distillation_losses':distillation_losses,
            'teacher_train_bag_forwards':225,'teacher_train_head_windows':8370,'draws':draws,'normalization_byte_identical':True,'original_roles_unchanged':True,
            'original_positive_labels':'WEAK','snr5_new8_training_used':False})
        old_metrics=[];new_metrics=[];new_natural={};reference=[];case_map=[];offset=0
        for condition in ('clean','snr10','snr5','new8'):
            old_metrics.extend([dict(r,condition=condition) for r in loads(regular(output/'inputs/coverage/predictions'/('coverage-'+condition+'-events.json')))])
        with (output/'predictions/candidate-frames.jsonl').open('x') as prediction, (output/'runtime/candidate-head-stream.bin').open('xb') as stream:
            stream.write(b'EHSTR01\0'+struct.pack('<I',18982))
            for case_index,key in enumerate(evaluation+natural_order):
                condition,rid=key;row=allrows[key];matrix,times=streams[key]
                need(file_identity(output/'models/retention-head.npz')==frozen['npz'],'FROZEN_BEFORE_EACH_EVAL')
                need(progress['candidate_head_cases_started']<323,'FIXED_CANDIDATE_CASE_BUDGET')
                progress['candidate_head_cases_started']+=1;ck()
                scores=portable.reference(candidate,matrix,times)
                events=[{'sample':r['sample'],'keyword':r['keyword']} for r in scores if r['keyword']]
                if condition=='natural':new_natural[rid]=scores
                else:new_metrics.append(dict(summarize_case(row,events),condition=condition))
                for ordinal,r in enumerate(scores):
                    prediction.write(json.dumps({'condition':condition,'id':rid,**r},separators=(',',':'),allow_nan=False)+'\n')
                    stream.write(struct.pack('<QI',offset+r['sample'],int(ordinal==0)))
                    stream.write(matrix[ordinal].astype('<f4',copy=False).tobytes())
                    reference.append(dict(r,case_index=case_index,condition=condition,id=rid,
                                          local_sample=r['sample'],sample=offset+r['sample']))
                case_map.append({'case_index':case_index,'condition':condition,'id':rid,'offset':offset,
                    'observed_samples':row['observed_samples'],'embedding_rows':len(scores),'ready_rows':sum(r['ready'] for r in scores)})
                offset+=row['observed_samples'];progress['candidate_head_cases_completed']+=1
                progress['candidate_embedding_rows']+=len(scores);progress['candidate_ready_rows']+=sum(r['ready'] for r in scores);ck()
        need(progress['candidate_head_cases_completed']==323 and progress['candidate_embedding_rows']==18982 and
             progress['candidate_ready_rows']==14137,'ALL_FIXED_CANDIDATE_REPLAYS_COMPLETE')
        once(output/'runtime/CASE_MAP.json',case_map)
        with (output/'runtime/reference-frames.jsonl').open('x') as f:
            for row in reference:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        paired=paired_case_gate(old_metrics,new_metrics);fit=reviewed_fit(selection,old_natural,new_natural)
        natural_events=[{'id':rid,'sample':r['sample'],'keyword':r['keyword']} for rid,rows in new_natural.items() for r in rows if r['keyword']]
        once(output/'predictions/candidate-events.json',new_metrics)
        once(output/'evidence/PAIRED_REGRESSION.json',paired)
        once(output/'evidence/REVIEWED_FIT.json',fit)
        once(output/'evidence/NATURAL32_PAIRED.json',{'old_event_count':0,'new_event_count':len(natural_events),'new_events':natural_events,
            'whole_originals_labeled_negative':False,'reviewed_sources':8,'other_sources':24,
            'other_source_labels':'UNKNOWN','independent_far':'NOT_MEASURED'})
        outcome={'status':'FIXED_TRAINING_AND_SAVED_CANDIDATE_REPLAYS_COMPLETE','passed':True,
            'candidate':frozen,'progress':progress,'research_gate_before_native':paired['passed'] and fit['passed'] and not natural_events,
            'gates':{'all_291_case_regression':paired['passed'],'natural32_zero_events':not natural_events,
                     'source_balanced_reviewed_ce_decreased':fit['passed'],'actual_weights_changed':True,
                     'normalization_byte_identical':True},'independent_far':'NOT_MEASURED','independent_frr':'NOT_MEASURED'}
    except BaseException as exc:
        outcome['error']=type(exc).__name__+':'+str(exc)
        put(output/'evidence/NUMERIC_TRACEBACK.txt',traceback.format_exc().encode())
    finally:
        once(output/'evidence/TRAINING_PARTIAL.json',partial_training_receipt(
            progress,losses,supervised_losses,distillation_losses,draws))
        once(output/'evidence/NUMERIC_RESULT.json',outcome)
    return 0 if outcome['passed'] else 1

def child(args,plan,admission):
    output=args.output.absolute();binding=execution_identity(plan)
    need(os.environ.get('KWS_RETENTION_PARENT_PID')==str(os.getppid()) and os.getpgrp()==os.getpid(),
         'ONLY_SUPERVISED_PUBLIC_WORKER')
    started=loads(regular(output/'STARTED.json'))
    need(started['binding']==binding and started['worker_sha256']==plan['worker_sha256'],'CHILD_SOURCE_BINDING')
    runner=Runner(output,started['monotonic_deadline']);result=result_boundary(binding);prepared=None
    once(output/'evidence/WORKER_STARTED.json',{'pid':os.getpid(),'parent_pid':os.getppid(),'binding':binding,'retry_count':0})
    try:
        for name in ('source','inputs','cache','predictions','runtime','models','downloads'):(output/name).mkdir()
        with runner.stage('source-binding'):
            run=loads(regular(runner.command('current-run',['gh','api','repos/'+REPO+'/actions/runs/'+binding['run_id']],auth=True)))
            need(run.get('id')==int(binding['run_id']) and run.get('head_sha')==binding['head_sha'] and
                 run.get('run_attempt')==1 and run.get('event')=='push' and run.get('head_branch')=='main' and
                 run.get('repository',{}).get('private') is False and run.get('repository',{}).get('full_name')==REPO,
                 'CURRENT_PUBLIC_RUN_BINDING')
            for name in plan['source_pins']:put(output/'source'/name,regular(ROOT/name))
            for name in (WORKER_PATH,PLAN_PATH):put(output/'source'/name,regular(ROOT/name))
            once(output/'source/CONFIG.json',CONFIG);once(output/'source/RECIPE.json',RECIPE)
        with runner.stage('authenticate-saved-inputs'):
            quality=runner.download('quality');natural=runner.download('natural')
            prepared=prepare_saved_inputs(ROOT,quality,natural,admission,output)
        with runner.stage('numeric-runtime-preparation'):
            prepare_dependencies(runner)
        with runner.stage('train-only-teacher-retention-and-regression'):
            os.environ['KWS_RETENTION_NUMERIC_PARENT_PID']=str(os.getpid())
            runner.command('retention-numeric',[sys.executable,'-I','-B',str(Path(__file__).resolve()),
                '--_numeric','--output',str(output)],timeout=600,kind='numeric')
            numeric_result=loads(regular(output/'evidence/NUMERIC_RESULT.json'))
            need(numeric_result['passed'] is True,'NUMERIC_COMPLETED_RESULT')
        with runner.stage('native-saved-binary-head-parity'):
            raw=regular(output/'inputs/coverage/runtime/head-native')
            need(raw[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',raw,18)[0]==62,'PINNED_NATIVE_X86_64_ELF')
            consumer=output/'runtime/head-native';put(consumer,raw);consumer.chmod(0o500)
            npz=regular(output/'models/retention-head.npz');binary=regular(output/'models/retention-head.bin')
            verify_saved_npz_bin(npz,binary)
            stdout=runner.command('head-native',[str(consumer),str(output/'models/retention-head.bin'),
                str(output/'runtime/candidate-head-stream.bin')],timeout=180,kind='native')
            reference=[loads(line) for line in regular(output/'runtime/reference-frames.jsonl').splitlines()]
            parity=native_parity(regular(stdout),reference)
            need(regular(consumer)==raw and regular(output/'models/retention-head.npz')==npz and
                 regular(output/'models/retention-head.bin')==binary,'NATIVE_INPUTS_UNCHANGED')
            once(output/'evidence/HEAD_NATIVE_PARITY.json',{**parity,'consumer':identity(raw),
                 'candidate_bin':identity(binary),'candidate_npz':identity(npz),'command_receipt':'head-native.exit.json'})
        need(runner.counts=={'numeric':1,'native':1},'EXACT_ONE_TRAINING_ONE_NATIVE_EXECUTION')
        gates={**numeric_result['gates'],'native_saved_npz_bin_parity':True}
        passed=all(gates.values())
        result.update(status='RETAINED_RESEARCH_TEACHER_RETENTION_HEAD' if passed else 'REJECTED_RESEARCH_REGRESSION_GATE',
            passed=passed,numerical_execution_completed=True,weights_retained=True,candidate=numeric_result['candidate'],
            progress=numeric_result['progress'],gates=gates,optimizer_updates=96,training_bag_forwards=672,
            research_retention_allowed=passed,current_default_model_replaced=False,
            owner_reviewed_negative_windows=9,owner_reviewed_negative_seconds=30.555,
            admitted_full_context_head_inputs=134,owner_reviewed_original_sources=8,
            reviewed_source_total_equal_weight=True,teacher_train_bag_forwards=225,teacher_train_head_windows=8370,teacher_development_forwards=0,teacher_reviewed_forwards=0)
    except BaseException as exc:
        result['error']=type(exc).__name__+':'+str(exc)
        put(output/'evidence/FAILURE_TRACEBACK.txt',traceback.format_exc().encode())
        if (output/'models/retention-head.npz').exists():
            result['retained_npz']=file_identity(output/'models/retention-head.npz')
        if (output/'models/retention-head.bin').exists():
            result['retained_bin']=file_identity(output/'models/retention-head.bin')
    finally:
        if prepared is not None:prepared['quality'].close();prepared['natural'].close()
        result['process_attempts']=runner.counts
        once(output/'evidence/RESULT.json',result)
    return 0 if result['passed'] else 1

def supervise(args, plan):
    binding = execution_identity(plan)
    output = args.output.absolute()
    need(output.parent.is_dir() and not output.exists() and
         not any(p.is_symlink() for p in (output, *output.parents)), 'NEW_SUPERVISED_OUTPUT_ONLY')
    output.mkdir(mode=0o700)
    (output / 'evidence').mkdir()
    started, process, timed_out, error, cleanup = time.monotonic(), None, False, None, {}
    deadline = started + 1200
    once(output / 'STARTED.json', {'schema': SCHEMA + '/started', 'binding': binding,
        'parent_pid': os.getpid(), 'worker_sha256': plan['worker_sha256'], 'monotonic_started': started,
        'monotonic_deadline': deadline, 'maximum_seconds': 1200, 'retry_count': 0, 'network_calls_at_start': 0})
    env = dict(os.environ, KWS_RETENTION_PARENT_PID=str(os.getpid()))
    argv = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--_worker', '--output', str(output)]
    def cancelled(signum, _frame):
        raise InterruptedError('SUPERVISOR_SIGNAL_' + str(signum))
    handlers = {sig: signal.signal(sig, cancelled) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (output / 'evidence/SUPERVISOR.stdout').open('xb') as out, (output / 'evidence/SUPERVISOR.stderr').open('xb') as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       env=env, cwd=str(ROOT), start_new_session=True)
            once(output / 'evidence/WORKER_LAUNCH.json', {'pid': process.pid, 'process_group': process.pid,
                 'argv': argv, 'attempt': 1, 'retry_count': 0})
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    raise TimeoutError('SUPERVISOR_1200_SECONDS')
                time.sleep(0.1)
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if process is not None:
            try:
                cleanup = cleanup_group(process.pid)
            except BaseException as exc:
                cleanup = {'error': type(exc).__name__ + ':' + str(exc), 'descendants_found': True}
                error = error or 'PROCESS_GROUP_CLEANUP_FAILED'
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                error = error or 'WORKER_NOT_REAPED'
        receipt = {'schema': SCHEMA + '/supervisor', 'returncode': None if process is None else process.returncode,
            'timed_out': timed_out, 'error': error, 'wall_seconds': time.monotonic() - started,
            'retry_count': 0, 'cleanup': cleanup, **binding}
        worker_result = None
        if (output / 'evidence/RESULT.json').exists():
            try:
                worker_result = loads(regular(output / 'evidence/RESULT.json'))
            except (ValueError, OSError) as exc:
                receipt['error'] = 'WORKER_RESULT_UNREADABLE:' + str(exc)
        receipt['passed'] = (process is not None and process.returncode == 0 and not timed_out and
            receipt['error'] is None and not cleanup.get('descendants_found') and
            type(worker_result) is dict and worker_result.get('passed') is True and
            all(worker_result.get(k) == v for k, v in binding.items()))
        receipt['partial_receipt_errors'] = close_interrupted_receipts(output, receipt)
        if receipt['partial_receipt_errors']:
            receipt['passed'] = False
        once(output / 'evidence/SUPERVISOR_EXIT.json', receipt)
        if type(worker_result) is not dict:
            fallback = result_boundary(binding)
            fallback.update(error='SUPERVISOR_TERMINATED_BEFORE_WORKER_RESULT', supervisor=receipt,
                            numeric_or_native_process_attempts=None if receipt['partial_receipt_errors'] else
                            sum(loads(regular(p)).get('kind') in ('numeric', 'native') for p in (output / 'evidence').glob('*.started.json')),
                            optimizer_updates_completed='UNKNOWN_IF_INTERRUPTED')
            result_path = output / 'evidence/RESULT.json'
            if result_path.exists():
                fallback['preserved_damaged_worker_result'] = file_identity(result_path)
                result_path = output / 'evidence/RESULT_SUPERVISOR_FAILURE.json'
            once(result_path, fallback)
        files = {p.relative_to(output).as_posix(): file_identity(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and 'downloads' not in p.relative_to(output).parts}
        once(output / 'MANIFEST.json', {'schema': SCHEMA + '/artifact', 'files': files,
             'downloads_excluded': True, 'supervised_success': receipt['passed'],
             'authoritative_completion_receipt': 'evidence/SUPERVISOR_EXIT.json', **dict.fromkeys(FALSE_FLAGS, False)})
    print(canonical(receipt).decode(), end='')
    return 0 if receipt['passed'] else 1

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    mode.add_argument('--_numeric',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--output',type=Path);args=parser.parse_args(argv)
    plan=loads(regular(ROOT/PLAN_PATH));admission=source_preflight(ROOT,plan)
    if args.execute or args._worker or args._numeric:
        need(args.output is not None,'EXPLICIT_EXECUTION_OUTPUT')
        if args._numeric:return numeric(args,plan)
        return child(args,plan,admission) if args._worker else supervise(args,plan)
    need(args.output is None,'SOURCE_ONLY_HAS_NO_EXECUTION_OUTPUT')
    print(canonical({'status':'SOURCE_CONTRACT_ONLY','new_model_calls':0,'new_audio_decodes':0,
        'new_network_calls':0,'enabled':plan['enabled'],'planned_updates':96,'planned_teacher_train_bags':225,'planned_reviewed_head_inputs':134,
        'candidate_cases':323,'native_processes':1,'maximum_seconds':1200,
        'worker_sha256':plan['worker_sha256']}).decode(),end='')
    return 0

if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
