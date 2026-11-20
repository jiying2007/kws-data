#!/usr/bin/env python3
"""B3 frozen C11 low-gain / context diagnostics on already-exposed synthetic DEV.

Exactly 14 independently machine-screened original DEV PCM (6 Hey Mero, 8 ordinary),
derived in memory under gain 1.0, 0.75, 0.50 and four predeclared 490-slow contexts.
One-shot first-main protected Actions, source-only PR, no model training, no
threshold tuning, CAL/LOCKED, new audio, ASR, human FAR/FRR or promotion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location(
    "b3_continuous_evidence_source",HERE/"english_v6_b3_continuous_c11_once.py")
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
b3=prior.b3

SCHEMA="english-v6-b3-c11-lowgain-and-490-predecessor-diagnostic-v1"
REPO="jiying2007/kws-data"
BEFORE="4995bfe0813f4bd6b65fa75a9c6764b14d64bd4e"
PREVIOUS_RUN=44658362723
PREVIOUS_ARTIFACT=14594762757
PREVIOUS_ARTIFACT_SHA="045c022a0d57a8750d289da10d307ceb50ad5e512120f7f31209bd0df62151c1"
GAINS=(1.0,0.75,0.5)
CONTEXTS=("exact_prev488","aligned_silence","aligned_prev488","aligned_prev489")
TARGET="merov6dev8-490-mero-slow"
PREFIX=16000
TRAILING=16000
SLOT=32000
GRACE=4000
COMPILERS=("gcc","clang")
MAX_ABS=0.005
MAX_MEAN=0.0005

def need(ok,message):
    if not ok:
        raise ValueError(message)

def write(path,record):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(record,f,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")

def source_check():
    evidence=prior.source_check()
    need(evidence["status"]=="SOURCE_ONLY_NO_AUDIO_FORWARD"
         and evidence["B3_artifact"]==14590093798
         and evidence["source_positive"]==6
         and evidence["source_ordinary"]==8
         and evidence["unknown_excluded"]==18
         and prior.BEFORE=="acbd6b2bab42374d2125e93537e040bc32b9c1f7"
         and b3.B3["pt"]=="682e2689e82aa91d850516c06b2be61efc64731cb1dee3b70a7d2a8e01925a46"
         and len(GAINS)==3 and GAINS==(1.0,0.75,0.5)
         and len(CONTEXTS)==4 and len(COMPILERS)==2
         and PREFIX==TRAILING==16000 and SLOT==32000 and GRACE==4000,
         "FROZEN_B3_PREDECLARED_EXPOSED_DEV_LOWGAIN_HYPOTHESIS")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_KWS_AUDIO_FORWARD",
            "keyword":"Hey Mero","model_artifact":b3.B3["id"],
            "model_parameters":43876,"prior_completed_stream_run":PREVIOUS_RUN,
            "input_qualified_positive":6,"input_qualified_ordinary":8,
            "excluded_unknown":18,"gain_values":list(GAINS),
            "context_cases":list(CONTEXTS),"one_changed_factor_per_case":True,
            "max_acoustic_cases":14*3+len(CONTEXTS),
            "model_selection_allowed":False,"new_training":False,
            "new_audio_tts_asr":False,"cal_locked_read":False,
            "physical_ssc305":False,"real_far_per_hour":"NOT_MEASURED",
            "human_3_5m_frr":"NOT_MEASURED",
            "product_promotion":False}

def verify_prior(token):
    need(isinstance(token,str) and len(token)>10,
         "GITHUB_READ_ONLY_PROVENANCE_REQUIRED")
    old=prior.verify_prior(token)
    def get(path):
        req=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+"/actions/"+path,
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"kws-v6-b3-gain-diagnostic"})
        with urllib.request.urlopen(req,timeout=35) as f:
            data=f.read(1048577)
        need(len(data)<=1048576,"BOUNDED_REMOTE_PROVENANCE")
        return json.loads(data)
    r=get("runs/"+str(PREVIOUS_RUN))
    need(r.get("id")==PREVIOUS_RUN and r.get("head_sha")==BEFORE
         and r.get("run_attempt")==1 and r.get("event")=="push"
         and r.get("status")=="completed" and r.get("conclusion")=="success",
         "REAL_PREVIOUS_C11_CONTINUOUS_SUCCESS")
    a=get("artifacts/"+str(PREVIOUS_ARTIFACT))
    need(a.get("id")==PREVIOUS_ARTIFACT
         and a.get("name")=="english-v6-b3-continuous-c11-44658362723"
         and a.get("digest")=="sha256:"+PREVIOUS_ARTIFACT_SHA
         and a.get("size_in_bytes")==3789
         and a.get("expired") is False
         and a.get("workflow_run",{}).get("id")==PREVIOUS_RUN,
         "EXACT_PREVIOUS_CONTINUOUS_RESULT_ARCHIVE")
    return {"verified_previous_run":PREVIOUS_RUN,
            "verified_previous_artifact":PREVIOUS_ARTIFACT,
            "verified_previous_zip_sha256":PREVIOUS_ARTIFACT_SHA,
            "prior_verified_b3":old,"no_optimizer":True}

def guard():
    source_check()
    env=os.environ
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"FIRST_PUBLIC_MAIN_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head)
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "EXACT_MAIN_IDENTITY")
    event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False
         and subprocess.check_output(["git","rev-parse","HEAD^1"],
                                     cwd=ROOT,text=True).strip()==BEFORE,
         "NO_REPLAY_NO_MOVED_MAIN")
    return {"run_id":int(env["GITHUB_RUN_ID"]),
            "commit":head,"prior":verify_prior(env.get("GH_TOKEN"))}

def gain_pcm(pcm,gain):
    need(isinstance(pcm,bytes) and 512*2<=len(pcm)<=16000*2*6
         and len(pcm)%2==0 and gain in GAINS,
         "BOUNDED_PCM16_GAIN_INPUT")
    if gain==1.0:
        return pcm
    n=len(pcm)//2
    samples=struct.unpack("<"+str(n)+"h",pcm)
    scaled=[max(-32768,min(32767,round(x*gain))) for x in samples]
    return struct.pack("<"+str(n)+"h",*scaled)

def source_rows(rows):
    order=prior.select_rows(rows)
    by_id={r["id"]:r for r in rows}
    need(len(by_id)==14
         and TARGET in by_id and by_id[TARGET]["mono_class"]==1
         and all(by_id["merov6dev8-"+str(i)+"-ordinary-time"]["mono_class"]==0
                 for i in (488,489)), "EXACT_490_AND_TWO_ORDINARY_PREDECESSORS")
    return order,by_id

def solo(raw):
    # The same amount of leading silence in every gain condition.
    need(len(raw)>1024 and len(raw)%2==0,"FULL_PCM16")
    target_start=PREFIX
    return b"\0"*(2*PREFIX)+raw+b"\0"*(2*TRAILING),target_start,len(raw)//2

def context_pcm(by_id,reader,case):
    need(case in CONTEXTS,"DECLARED_490_CONTEXT_CASE")
    target=reader(by_id[TARGET])
    need(b3.sha(target)==by_id[TARGET]["source_pcm_sha256"],
         "EXACT_490_ORIGINAL_SOURCE_SHA")
    if case=="exact_prev488":
        pred=reader(by_id["merov6dev8-488-ordinary-time"])
        need(b3.sha(pred)==by_id["merov6dev8-488-ordinary-time"]["source_pcm_sha256"],
             "EXACT_488_SOURCE_SHA")
        lead=b"\0"*(2*PREFIX)+pred+b"\0"*(2*TRAILING)
    else:
        if case=="aligned_silence":
            content=b""
        else:
            i=488 if case=="aligned_prev488" else 489
            pred=by_id["merov6dev8-"+str(i)+"-ordinary-time"]
            content=reader(pred)
            need(b3.sha(content)==pred["source_pcm_sha256"],
                 "IMMUTABLE_PREDECESSOR_PCM_SHA")
        need(len(content)//2<=SLOT,"FIXED_PREFIX_SLOT_CANNOT_CROP")
        lead=b"\0"*(2*PREFIX)+content+b"\0"*(2*(SLOT-len(content)//2))
    start=len(lead)//2
    return lead+target+b"\0"*(2*TRAILING),start,len(target)//2

def observed_argmax(logit,first_sample,samples):
    import numpy as np
    need(logit.ndim==2 and logit.shape[1]==4
         and np.isfinite(logit).all(),"FINITE_FOUR_TOKEN_C11_LOGITS")
    pred=logit.argmax(-1)
    clocks=512+160*np.arange(logit.shape[0])
    mask=(clocks>=first_sample)&(clocks<first_sample+samples+GRACE)
    selected=logit[mask]
    need(selected.shape[0]>0,"SOURCE_FRAME_WINDOW_PRESENT")
    winner=selected.argmax(-1)
    histogram={name:int(np.sum(winner==index)) for index,name in
               enumerate(("BLANK","HEY","MEH","ROH"))}
    meh=selected[:,2]-np.max(selected[:,[0,1,3]],axis=1)
    need(np.isfinite(meh).all(),"FINITE_MEH_MARGIN")
    collapse=[]
    prev=0
    for token in winner:
        value=int(token)
        if value!=prev and value!=0:
            collapse.append(value)
        prev=value
    return {"per_target_frame_argmax_histogram":histogram,
            "target_meh_argmax_any":histogram["MEH"]>0,
            "best_meh_winner_margin":float(meh.max()),
            "collapsed_nonblank_tokens":collapse[:200],
            "collapsed_token_count":len(collapse)}

def run_one(pcm,start,native_model,executables,scratch):
    import numpy as np
    need(len(pcm)%2==0 and len(pcm)<=16000*2*20,
         "BOUNDED_EPHEMERAL_ACOUSTIC_SEQUENCE")
    scratch.mkdir()
    raw=scratch/"audio.pcm"
    raw.write_bytes(pcm)
    results={}
    arrays={}
    for kind,binary in executables.items():
        logits=scratch/(kind+".f32")
        events=scratch/(kind+".events.csv")
        call=subprocess.run([str(binary),str(native_model),str(raw),
                             str(logits),str(events)],
                             capture_output=True,text=True,check=True,timeout=75)
        need("MERO43_C_PCM_STREAM_OK" in call.stderr,
             "REAL_C11_B3_GAIN_AUDIO_FORWARD_"+kind)
        rawlog=logits.read_bytes()
        need(len(rawlog)>0 and len(rawlog)%16==0,"CTC_LOGITS_BYTE_COUNT")
        array=np.frombuffer(rawlog,dtype="<f4").reshape(-1,4)
        clock=b3.c_events(events,len(array))
        results[kind]={"sha256_logits":b3.sha(rawlog),
             "num_frames":len(array),"event_clocks":clock}
        arrays[kind]=array
    a,b=arrays["gcc"],arrays["clang"]
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
         "GCC_CLANG_FINITE_AND_SAME_FRAMES")
    errors=np.abs(a.astype(np.float64)-b.astype(np.float64))
    need(float(errors.max())<=MAX_ABS
         and float(errors.mean())<=MAX_MEAN
         and np.array_equal(a.argmax(-1),b.argmax(-1))
         and results["gcc"]["event_clocks"]==results["clang"]["event_clocks"],
         "GCC_CLANG_RESEARCH_GAIN_EVENT_OR_LOGIT_MISMATCH")
    observations=observed_argmax(a,start,len(pcm)//2-start-TRAILING)
    clocks=results["gcc"]["event_clocks"]
    matching=[v for v in clocks if start<=v<start+observations.get("_unused",0)]
    # All event clocks are absolute availability, not actual word-end.
    # Use full original duration from the caller for official attribution.
    return observations,results,float(errors.max()),clocks

def execute(source,screen,b3zip,pipeline,output):
    import numpy as np
    import torch
    receipt=guard()
    need(output.is_absolute() and output.parent.is_dir()
         and not output.exists(),"NEW_EVIDENCE_OUTPUT_ONLY")
    output.mkdir()
    write(output/"STARTED.json",{"schema":SCHEMA,"receipt":receipt,
       "new_training":False,"new_tts_asr":False,"dev_exposed":True,
       "cal_locked_opened":False})
    try:
        rows,source_receipt=b3.audit(source,screen,b3zip)
        _,by_id=source_rows(rows)
        native=b3.pipeline_files(pipeline)
        packer=b3.imported(native/"pack_mero43.py","b3_gain_original_pack_geometry")
        _,_,_,_,packed=b3.load_b3(torch,np,b3zip,packer)
        need(len(packed)==216960,"UNCHANGED_B3_C11_ABI")
        model=output/"ephemeral-b3.pack"
        model.write_bytes(packed)
        binaries={name:b3.compile_c(native,name,output/("b3-gain-"+name))
                  for name in COMPILERS}
        records=[]
        for row in sorted(rows,key=lambda r:r["id"]):
            original=b3.v6.pcm(row,source)
            need(b3.sha(original)==row["source_pcm_sha256"],
                 "AUTHENTIC_ORIGINAL_DEV_PCM")
            for gain in GAINS:
                scaled=gain_pcm(original,gain)
                pcm,start,n=solo(scaled)
                label=row["id"]+"-"+str(gain).replace(".","p")
                scratch=output/("scratch-"+label)
                observed,platforms,delta,events=run_one(
                    pcm,start,model,binaries,scratch)
                target_events=[int(x) for x in events
                               if start<=x<start+n+GRACE]
                records.append({"kind":"gain","id":row["id"],
                    "source_group_id":row["source_group_id"],
                    "class":row["mono_class"],"gain":gain,
                    "original_pcm_sha256":row["source_pcm_sha256"],
                    "derived_pcm_sha256":b3.sha(pcm),
                    "original_samples":n,"event_count":len(events),
                    "target_event_count":len(target_events),
                    "target_event_offsets":[v-start for v in target_events],
                    "unattributed_event_count":len(events)-len(target_events),
                    "gcc_clang_logit_max_abs":delta,
                    "platforms":platforms,**observed})
                shutil.rmtree(scratch)
        for case in CONTEXTS:
            pcm,start,n=context_pcm(by_id,lambda r: b3.v6.pcm(r,source),case)
            scratch=output/("scratch-context-"+case)
            observed,platforms,delta,events=run_one(
                pcm,start,model,binaries,scratch)
            target_events=[int(x) for x in events if start<=x<start+n+GRACE]
            records.append({"kind":"context","case":case,"id":TARGET,
                 "class":1,"gain":1.0,"start_samples":start,
                 "original_samples":n,"derived_pcm_sha256":b3.sha(pcm),
                 "event_count":len(events),"target_event_count":len(target_events),
                 "target_event_offsets":[v-start for v in target_events],
                 "unattributed_event_count":len(events)-len(target_events),
                 "gcc_clang_logit_max_abs":delta,
                 "platforms":platforms,**observed})
            shutil.rmtree(scratch)
        need(len(records)==14*3+len(CONTEXTS),"EXACT_BOUNDED_GAIN_CONTEXT_MATRIX")
        aggregates=[]
        for gain in GAINS:
            subset=[x for x in records if x["kind"]=="gain" and x["gain"]==gain]
            need(len(subset)==14 and
                 sum(x["class"]==1 for x in subset)==6,
                 "SAME_SOURCE_SET_EACH_GAIN")
            aggregates.append({"gain":gain,"positive_hits_of_six":sum(
                x["class"]==1 and x["target_event_count"]>0 for x in subset),
                "ordinary_any_events_of_eight":sum(
                x["class"]==0 and x["event_count"]>0 for x in subset),
                "positive_missing_meh_argmax":sum(
                x["class"]==1 and not x["target_meh_argmax_any"] for x in subset)})
        result={"schema":SCHEMA,"status":
          "PASS_EXPOSED_DEV_B3_C11_GAIN_AND_CONTEXT_MECHANISM_NO_PROMOTION",
          "receipt":receipt,"source_receipt":source_receipt,
          "B3_model_artifact":b3.B3["id"],"B3_model_sha256":b3.B3["pt"],
          "source_count":14,"positive_count":6,"ordinary_count":8,
          "unknown_excluded":18,"runs":len(records),"records":records,
          "aggregate":aggregates,
          "unseen_source_dev":"NOT_RUN",
          "new_optimizer_updates":0,"new_tts_asr_audio":0,
          "developer_diagnostics_only":True,
          "calibration_locked_opened":False,
          "real_human_frr":"NOT_MEASURED","natural_far_per_hour":"NOT_MEASURED",
          "physical_ssc305":"NOT_RUN","model_selection_allowed":False,
          "product_promotion_allowed":False}
        write(output/"RESULT.json",result)
        print(json.dumps({"status":result["status"],"gain_summaries":aggregates,
            "context_results":{x["case"]:x["target_event_count"] for x in records
                               if x["kind"]=="context"},
            "product_promotion_allowed":False},sort_keys=True))
    except BaseException as e:
        write(output/"STOP.json",{"schema":SCHEMA,
              "status":"STOP_EXPOSED_GAIN_NO_RETRY",
              "reason_type":type(e).__name__,"reason":str(e)[:400],
              "calibration_locked_opened":False,
              "product_promotion_allowed":False})
        raise
    finally:
        for x in output.glob("scratch-*"):
            if x.is_dir():shutil.rmtree(x)
        for x in output.glob("b3-gain-*"):
            if x.is_file():x.unlink()
        if (output/"ephemeral-b3.pack").exists():
            (output/"ephemeral-b3.pack").unlink()

def main():
    ap=argparse.ArgumentParser()
    cmds=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--guard","--execute"):
        cmds.add_argument(flag,action="store_true")
    for key in ("source","screen","b3-zip","pipeline","output"):
        ap.add_argument("--"+key,type=Path)
    args=ap.parse_args()
    if args.source_check:
        print(json.dumps(source_check(),sort_keys=True))
    elif args.guard:
        print(json.dumps(guard(),sort_keys=True))
    else:
        need(all(x is not None for x in (args.source,args.screen,
             args.b3_zip,args.pipeline,args.output)),
             "REQUIRE_FULL_IMMUTABLE_INPUTS")
        execute(args.source,args.screen,args.b3_zip,args.pipeline,args.output)

if __name__=="__main__":
    main()
