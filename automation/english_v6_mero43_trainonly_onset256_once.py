#!/usr/bin/env python3
"""V6 single Hey Mero 43K onset-debias experiment: authenticated TRAIN only.

Reuse 35 already machine-accepted TRAIN source groups and the byte-frozen B1
43,876-parameter checkpoint. Add deterministic in-memory silent PCM prefixes
(0.4/0.8/1.2 s), CTC full-sequence loss and a known-silent-prefix blank
penalty. No new TTS/ASR, no development/Cal/Locked model inference, no
threshold tuning or product/model promotion. One new 256-step optimization.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json, os, re, subprocess, time, zipfile
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(name):
    spec=importlib.util.spec_from_file_location('v6_onset_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

robust=load('english_v5_mero43_trainonly_robust_challenger_once')
v5=robust.v5
SCHEMA='english-v6-mero43-trainonly-causal-onset-prefix256-v1'
REPO='jiying2007/kws-data'
BEFORE='0e3957370215ddf3d502441b6b859a684d92bf21'
STEPS=256
SEED=20261120
LR=0.00012
PREFIX_SAMPLES=(6400,12800,19200)
B1_ZIP='bab018a63aaf4cc760ee835b8554ebacf73e6ee77c37adc7c9d5008b1a03d7ce'
B1_PT='20d0c4632224cf332460493695c0dd21927a80ed13034a9531f5d5230d2f7e05'
B1_NPZ='76f413772e2820b3aab27124e1f6b627d9de19e9520068ac205bdea497f0b648'

def need(x,msg):
    if not x: raise ValueError(msg)

def sha(v): return hashlib.sha256(v).hexdigest()

def record(path,obj):
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)
        f.write('\n')

def source_check():
    base=robust.source_check()
    need(base['status']=='SOURCE_ONLY_NO_OPTIMIZER_OR_AUDIO_FORWARD'
         and base['keyword']=='Hey Mero'
         and base['source_train_groups']==35
         and base['new_prospective_dev_machine_positive']==2
         and base['new_prospective_dev_ordinary']==8
         and base['frozen_model_zip_sha256']==robust.MODEL['sha']
         and base['calibration_locked_audio_opened'] is False
         and STEPS==256 and PREFIX_SAMPLES==(6400,12800,19200)
         and SEED==20261120 and LR==0.00012
         and all(len(x)==64 and re.fullmatch('[a-f0-9]{64}',x) for x in (B1_ZIP,B1_PT,B1_NPZ)),
         'PINNED_B1_AND_35_REAL_TRAIN_GROUPS')
    return {'schema':SCHEMA,'status':'SOURCE_ONLY_NO_NEW_OPTIMIZER',
        'target':'Hey Mero','parameters':43876,'train_groups':35,
        'frozen_parent':B1_ZIP,'new_steps':STEPS,
        'prefix_samples':list(PREFIX_SAMPLES),
        'blank_only_prefix_min_gap_frames':8,
        'dev_model_inference':False,'new_tts_or_asr':False,
        'cal_locked_access':False,'product_promotion':False}

def guard():
    source_check()
    require={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':REPO,
             'GITHUB_EVENT_NAME':'push','GITHUB_REF':'refs/heads/main',
             'GITHUB_RUN_ATTEMPT':'1'}
    need(all(os.environ.get(k)==v for k,v in require.items()),'ONLY_FIRST_PUBLIC_MAIN_ONSET_TRAIN')
    head=os.environ.get('GITHUB_SHA','')
    event=json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_bytes())
    need(re.fullmatch('[a-f0-9]{40}',head) is not None
         and re.fullmatch('[1-9][0-9]*',os.environ.get('GITHUB_RUN_ID','')) is not None
         and event.get('before')==BEFORE and event.get('after')==head
         and event.get('ref')=='refs/heads/main'
         and event.get('repository',{}).get('full_name')==REPO
         and event.get('repository',{}).get('private') is False
         and event.get('forced') is False and event.get('deleted') is False
         and subprocess.check_output(['git','rev-parse','HEAD^1'],cwd=ROOT,text=True).strip()==BEFORE,
         'NO_REPLAY_OR_MOVED_MAIN')
    return {'run':os.environ['GITHUB_RUN_ID'],'source_commit':head,'parent':BEFORE}

def frozen_b1(model_zip,torch,np):
    need(model_zip.is_file() and not model_zip.is_symlink()
         and sha(model_zip.read_bytes())==B1_ZIP,'FROZEN_B1_FULL_ZIP_SHA')
    with zipfile.ZipFile(model_zip) as z:
        result=json.loads(z.read('RESULT.json'))
        need(result['model_parameters']==43876 and result['actual_optimizer_updates']==288
             and result['train_source_groups']==35
             and result['status']=='RESEARCH_EXPOSED_DEV_DIAGNOSTIC_ONLY_NO_PROMOTION'
             and result['cal_locked_opened'] is False
             and result['product_promotion_allowed'] is False,'FROZEN_B1_TRAIN_EVIDENCE')
        pt='models/mero-fsmn-ctc-trainonly-robust288.pt'
        npz='models/mero-fsmn-ctc-trainonly-robust288.npz'
        need(sha(z.read(pt))==B1_PT and sha(z.read(npz))==B1_NPZ,'FROZEN_B1_PT_NPZ_SHA')
        weights=torch.load(io.BytesIO(z.read(pt)),map_location='cpu',weights_only=True)
        model=v5.build_model(torch);model.load_state_dict(weights,strict=True)
        model.eval()
        with np.load(io.BytesIO(z.read(npz)),allow_pickle=False) as pack:
            need(all(np.array_equal(pack[k],v.detach().cpu().numpy()) for k,v in weights.items()),
                 'FROZEN_B1_PT_NPZ_SAME_TENSORS')
            bank=pack['mel_filters'].copy()
            mean=pack['train_mean'].copy()
            scale=pack['train_scale'].copy()
    need(bank.shape==(257,40) and mean.shape==scale.shape==(40,)
         and (scale>0).all() and np.isfinite(bank).all()
         and np.isfinite(mean).all() and np.isfinite(scale).all(),
         'FROZEN_B1_40MEL_NORMALIZATION')
    return model,bank,mean,scale

def train(rows,old_archives,fresh_source,b1_model_zip,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    model,bank,mean,scale=frozen_b1(b1_model_zip,torch,np)
    grouped={0:defaultdict(list),1:defaultdict(list)}
    for r in rows: grouped[r['mono_class']][r['source_group_id']].append(r)
    need(len(grouped[0])==len(grouped[1])==35
         and all(r['role']=='train' for r in rows)
         and all({0,1}=={k for k in (0,1) if group in grouped[k]}
                 for group in grouped[0]),'TRAIN_ONLY_SOURCE35_BALANCED')
    sources={};cache={}
    try:
        for spec in v5.binding.config()['corpus']:
            sources[spec['key']]=v5.binding.archive(old_archives/(spec['key']+'-source.zip'),spec['source'])
        sources['newtrain24']=v5.binding.archive(old_archives/'newtrain24-source.zip',v5.prior.SOURCE)
        for row in rows:
            raw=robust.sourcefit.readpcm(row,sources,fresh_source)
            need(sha(raw)==row['source_pcm_sha256'],'TRAIN_PCM_IMMUTABLE_SHA')
            cache[row['id']]={}
            for n in PREFIX_SAMPLES:
                candidate=b'\x00\x00'*n+raw
                derived={**row,'samples':row['samples']+n}
                feat=v5.features(candidate,derived,torch,np,bank)
                x=torch.from_numpy(((feat-mean)/scale).astype(np.float32)).unsqueeze(0)
                guard_frames=n//160-8
                need(guard_frames>0 and guard_frames<len(feat)-8,'EXACT_PREFIX_GUARD')
                cache[row['id']][n]=(x,guard_frames,sha(candidate))
    finally:
        for obj in sources.values():obj.close()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.001)
    trace=[];started=time.perf_counter()
    for step in range(STEPS):
        model.train();opt.zero_grad(set_to_none=True);losses=[]
        prefix=PREFIX_SAMPLES[(step//35+step%3)%len(PREFIX_SAMPLES)]
        for klass in (0,1):
            family=sorted(grouped[klass])[step%35]
            selections=sorted(grouped[klass][family],key=lambda x:x['id'])
            row=selections[(step//35)%len(selections)]
            x,n_blank,_=cache[row['id']][prefix]
            logits=model(x);lprobs=logits.log_softmax(-1)
            target=torch.tensor([1,2,3] if klass else [],dtype=torch.long)
            ctc=F.ctc_loss(lprobs.transpose(0,1),target,
                 torch.tensor([lprobs.shape[1]],dtype=torch.long),
                 torch.tensor([len(target)],dtype=torch.long),blank=0,
                 reduction='none',zero_infinity=False)[0]
            # Exact in-memory TRAIN-only silent prefix. Exclude eight boundary
            # frames; no guessed onset timestamps or forced phone alignments.
            prefix_logits=logits[:, :n_blank, :]
            blank_margin=torch.logsumexp(prefix_logits[...,1:],dim=-1)-prefix_logits[...,0]
            onset=F.softplus(blank_margin).mean()
            losses.append((ctc*(1.0 if klass else 0.6)+0.5*onset)*
                         (1.0 if row['tier']=='silver' else 0.75))
        total=sum(losses)/2
        need(bool(torch.isfinite(total)),'FINITE_PREFIX_CTC_AND_BLANK_LOSS')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
        opt.step()
        if step%16==0 or step==STEPS-1:
            trace.append({'step':step+1,'loss':float(total.detach()),
                          'neg':float(losses[0].detach()),
                          'pos':float(losses[1].detach()),
                          'prefix_samples':prefix})
    model.eval()
    dst=out/'models';dst.mkdir()
    pt=dst/'mero43-onset256.pt';torch.save(model.state_dict(),pt)
    loaded=v5.build_model(torch);loaded.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=True)
    loaded.eval()
    npz=dst/'mero43-onset256.npz'
    np.savez(npz,mel_filters=bank,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy() for k,v in loaded.state_dict().items()})
    example=cache[sorted(rows,key=lambda r:r['id'])[0]['id']][PREFIX_SAMPLES[0]][0]
    onnx=dst/'mero43-onset256.onnx'
    torch.onnx.export(loaded,example,onnx.as_posix(),opset_version=17,dynamo=False,
       input_names=['fbank40'],output_names=['frame_ctc_logits'],
       dynamic_axes={'fbank40':{1:'frames'},'frame_ctc_logits':{1:'frames'}})
    need(all(p.is_file() and p.stat().st_size>10000 for p in (pt,npz,onnx)),
         'ACTUAL_NEW_PYTORCH_NPZ_ONNX_WEIGHTS')
    # This is an IN-SAMPLE train-only mechanistic probe, never new DEV.
    probe=[]
    with torch.no_grad():
        for row in sorted(rows,key=lambda r:r['id'])[:min(len(rows),12)]:
            x,n_blank,_=cache[row['id']][PREFIX_SAMPLES[-1]]
            outlog=loaded(x).argmax(-1).reshape(-1).tolist()
            probe.append({'train_id':row['id'],'first_argmax':int(outlog[0]),
                          'blank_prefix_argmax_fraction':round(sum(t==0 for t in outlog[:n_blank])/n_blank,6)})
    return {'schema':SCHEMA,'status':'RESEARCH_TRAIN_ONLY_PREFIX_ONSET_MODEL_NOT_PROMOTABLE',
       'actual_optimizer_steps':STEPS,'source_train_groups':35,
       'parent_b1_zip_sha256':B1_ZIP,
       'losses':trace,'wall_optimizer_s':round(time.perf_counter()-started,3),
       'model_parameter_count':sum(t.numel() for t in loaded.parameters()),
       'new_model_hashes':{p.name:{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in (pt,npz,onnx)},
       'mechanism_probe':'TRAIN_ONLY_NOT_INDEPENDENT','train_only_prefix_probe':probe,
       'new_audio_tts_asr_calls':0,'old_dev_model_forwards':0,'new_dev_model_forwards':0,
       'cal_locked_access':False,'physical_ssc305':'NOT_RUN','human_frr_far':'NOT_MEASURED',
       'model_selection_allowed':False,'product_promotion_allowed':False}

def execute(a):
    g=guard()
    out=a.output
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(),'ONE_NEW_OUTPUT_DIRECTORY')
    out.mkdir();record(out/'STARTED.json',{'schema':SCHEMA,'run':g,'new_optimizer_steps':0})
    try:
        training,dev,proof=robust.audit(a.old_archives,a.old_audit,a.fresh_source,
             a.fresh_screen,a.source_gate,a.prospective_source,a.prospective_screen,a.original_model_zip)
        need(len({r['source_group_id'] for r in training})==35
             and all(r['role']=='train' for r in training)
             and all(r['role']=='development' for r in dev),
             'NO_OLD_DEV_CAL_LOCKED_GRADIENTS')
        record(out/'SOURCE.json',{'audit':proof,'train_pcm_shas':sorted(r['source_pcm_sha256'] for r in training),
            'no_dev_model_forwards':True})
        result=train(training,a.old_archives,a.fresh_source,a.b1_model_zip,out)
        record(out/'RESULT.json',result)
        print(json.dumps({'status':result['status'],'steps':result['actual_optimizer_steps'],
            'sha':result['new_model_hashes']['mero43-onset256.pt']['sha256'],
            'train_only_blank_probe':result['train_only_prefix_probe'],
            'product_promotion_allowed':False},sort_keys=True))
    except BaseException as e:
        record(out/'STOP.json',{'status':'STOP_NO_REPLAY','type':type(e).__name__,
             'reason':str(e)[:420],'no_cal_locked_access':True})
        raise

def main():
    p=argparse.ArgumentParser()
    for flag in ('--source-check','--guard','--audit','--execute'):
        p.add_argument(flag,action='store_true')
    for s in ('old-archives','old-audit','fresh-source','fresh-screen','source-gate',
              'prospective-source','prospective-screen','original-model-zip',
              'b1-model-zip','output'):
        p.add_argument('--'+s,type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.audit,a.execute))==1,'EXACT_ONE_STAGE')
    if a.source_check:print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:print(json.dumps(guard(),sort_keys=True));return
    for name in ('old_archives','old_audit','fresh_source','fresh_screen','source_gate',
                 'prospective_source','prospective_screen','original_model_zip'):
        need(getattr(a,name) is not None,'REQUIRED_8_SOURCE_ASSETS_'+name)
    if a.audit:
        t,d,receipt=robust.audit(a.old_archives,a.old_audit,a.fresh_source,a.fresh_screen,
                  a.source_gate,a.prospective_source,a.prospective_screen,a.original_model_zip)
        print(json.dumps({'status':'AUTHENTICATED_TRAIN_ONLY_NO_FORWARDS',
            'train_rows':len(t),'dev_rows_not_used':len(d),'train_groups':35},sort_keys=True));return
    need(a.b1_model_zip and a.output,'EXACT_NEW_B1_AND_OUTPUT_REQUIRED')
    execute(a)

if __name__=='__main__':main()
