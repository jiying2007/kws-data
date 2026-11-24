#!/usr/bin/env python3
"""B10 TRAIN-only paired FSMN-CTC hard-negative experiment.

Both 192-update arms share immutable B9 same-job control checkpoint,
same 79 old TRAIN originals, same 35 groups, optimizer, CTC/KL and seed.
Exactly one changed factor: challenger adds a bounded 0.08-weight loss from
six NEW independently machine-qualified TRAIN ordinary WAVs. Never use DEV
PCM in gradients, threshold selection, CAL/LOCKED, or automatic promotion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("b10_paired_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

b9=load("english_v6_b9_dev528_four_frozen_once")
b9training=load("english_v6_b9_half_gain_hey_presence_trainonce")
b6=b9training.b6
screen=load("english_v6_b10_hardneg_saved_screen_once")
observer=screen.observer
c=screen.c
phones=screen.phones
REPO="jiying2007/kws-data"
SCHEMA="english-v6-b10-b9control-fsmn-hardnegative-paired-train-v1"
BEFORE="cf3d995d830e69f875c7921ff0ac95498666ecf9"
STEPS=192
PARAMS=43876
TRAIN_GROUPS=35
SEED=20261120
LR=0.00006
HARD_WEIGHT=0.08
GAINS=(1.0,0.5)
PREFIXES=(6400,12800,19200)
ARMS=("control","hard_negative")
MODEL_STEMS={"control":"mero43-b10-matched-control192",
             "hard_negative":"mero43-b10-train-hardneg192"}
ARTIFACTS={
 "b9":(14689218171,44860855857,
       "846c745ebcee58a1beb27300ddcd7ec8e2178916",
       1001753,"1a8b270200e0c72ab1025f139d16abde43c209f142d72e71a7201b847e06a612"),
 "source":(14708400220,44906200481,
           "115738743e6c2cfa0cc55df9df772bb00e4c673c",
           371843,"dbc57d762922e60ff23253af2463d5d64e1c8021c74fad6e2f0a9ecb9fd87465"),
 "screen":(14740257228,44986730086,
           BEFORE,3474,
           "ff4d647a224f430e50a76c1f7b42854ad9b41324db0f6b2f2823aaa4c63adb12")}

def need(ok,reason):
    if not ok: raise ValueError(reason)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path):
    return json.loads(c.read(Path(path)))

def write(path,obj):
    with Path(path).open("x",encoding="utf-8") as handle:
        json.dump(obj,handle,ensure_ascii=False,sort_keys=True,
                  indent=2,allow_nan=False)
        handle.write("\n")

def source_check():
    previous=screen.source_check()
    need(previous["status"]=="SAVED_SOURCE_ONLY_NO_NEW_MODEL_FORWARD"
         and b6.SEED==SEED and b6.STEPS==STEPS
         and b6.LR==LR and b6.PREFIXES==PREFIXES
         and b6.GAINS==GAINS and b6.SOURCE_GROUPS==TRAIN_GROUPS
         and b6.PARAMS==PARAMS
         and b6.WEIGHT==0.03 and b6.KL_CLIP==8.0
         and ARMS==("control","hard_negative")
         and MODEL_STEMS["control"]!=MODEL_STEMS["hard_negative"]
         and HARD_WEIGHT==0.08
         and b9.MODELS["B9_control"]["pt"]==
              "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"
         and ARTIFACTS["screen"][0]==14740257228
         and ARTIFACTS["source"][0]==14708400220
         and ARTIFACTS["b9"][0]==14689218171,
         "EXACT_FROZEN_B9_CONTROL_AND_B10_SILVER_TRAIN_SOURCE")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_OPTIMIZER_OR_MODEL_FORWARD",
            "keyword":"Hey Mero","frozen_b9_control_pt_sha256":
                  b9.MODELS["B9_control"]["pt"],
            "same_old_train_groups":TRAIN_GROUPS,
            "original_old_train_rows":79,
            "new_train_ordinary_hard_rows":6,
            "new_train_groups":6,
            "qualified_development_groups_required":2,
            "common_parent":True,"common_seed":SEED,
            "common_optimization_steps":STEPS,
            "new_changed_factor":"SIX_INDEPENDENTLY_REVIEWED_TRAIN_HARD_NEGATIVES_ONLY",
            "control_hard_negative_loss_weight":0.0,
            "challenger_hard_negative_loss_weight":HARD_WEIGHT,
            "max_optimizer_updates":2*STEPS,
            "new_audio_asr_phone_calls":0,
            "new_model_forwards_in_pr":0,
            "no_old_DEV520_DEV528_training":True,
            "no_DEV_gradients_or_forwards":True,
            "no_CAL_LOCKED":True,
            "model_selection_allowed":False,
            "product_promotion_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    e=os.environ if env is None else env
    need(all(e.get(k)==v for k,v in {
         "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
         "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
         "GITHUB_RUN_ATTEMPT":"1"}.items()),
         "FIRST_PUBLIC_MAIN_ONLY_NO_REPLAY")
    head=e.get("GITHUB_SHA","")
    need(re.fullmatch(r"[0-9a-f]{40}",head) is not None
         and re.fullmatch(r"[1-9][0-9]*",e.get("GITHUB_RUN_ID","")) is not None,
         "REAL_B10_MAIN_RUN_ID")
    if event is None: event=read(e["GITHUB_EVENT_PATH"])
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                         cwd=ROOT,text=True).strip()
    need(event.get("before")==BEFORE and event.get("after")==head
         and parent==BEFORE and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "EXACT_FIRST_B10_TRAIN_SUCCESSOR_ONLY")
    return {"run_id":e["GITHUB_RUN_ID"],"head":head,"parent":BEFORE}

def remote(token):
    need(type(token) is str and len(token)>10,
         "READONLY_GITHUB_SOURCE_AND_MODEL_PROVENANCE_REQUIRED")
    def get(path):
        req=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+"/actions/"+path,
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"kws-v6-b10-real-paired-train"})
        with urllib.request.urlopen(req,timeout=40) as res:
            raw=res.read(1000001)
        need(len(raw)<=1000000,"BOUNDED_GITHUB_B10_SOURCE_METADATA")
        return json.loads(raw)
    for label,(aid,rid,head,size,sha) in ARTIFACTS.items():
        run=get("runs/"+str(rid))
        need(run.get("id")==rid and run.get("head_sha")==head
             and run.get("head_branch")=="main"
             and run.get("event")=="push"
             and run.get("run_attempt")==1
             and run.get("status")=="completed"
             and run.get("conclusion")=="success",
             "EXACT_B10_ORIGINAL_PROTECTED_MAIN_RUN_"+label)
        artifact=get("artifacts/"+str(aid))
        need(artifact.get("id")==aid
             and artifact.get("size_in_bytes")==size
             and artifact.get("digest")=="sha256:"+sha
             and artifact.get("expired") is False
             and artifact.get("workflow_run",{}).get("id")==rid
             and artifact.get("workflow_run",{}).get("head_sha")==head,
             "EXACT_B10_ORIGINAL_MODEL_SOURCE_SCREEN_ARCHIVE_"+label)
    return {key:value[0] for key,value in ARTIFACTS.items()}

def training_rows(source_dir,screen_dir,old_rows,groups):
    """Metadata only for DEV; open WAV solely for six accepted TRAIN."""
    packet=read(Path(source_dir)/"MANIFEST.json")
    receipt=read(Path(screen_dir)/"SCREEN.json")
    need(packet.get("schema")==observer.source_worker.SCHEMA
         and packet.get("completed")==18 and packet.get("denominator")==18
         and receipt.get("schema")==screen.SCHEMA
         and receipt.get("status")=="PASS_B10_TRAIN_DEV_MACHINE_RESEARCH_GATE_ONLY"
         and receipt.get("source_run")==44906200481
         and receipt.get("observer_run")==44908125673
         and receipt.get("complete_native_attempts")==90
         and receipt.get("admission",{}).get("train_hard_negative_qualified")==6
         and receipt.get("admission",{}).get("development_qualified_groups")==2
         and receipt.get("admission",{}).get("minimum_6_train_2_dev_gate_passed") is True
         and receipt.get("admission",{}).get("B10_research_optimizer_allowed") is True
         and receipt.get("admission",{}).get("development_audio_in_optimizer") is False
         and receipt.get("new_optimizer_updates")==0
         and receipt.get("product_promotion_allowed") is False
         and len(receipt.get("rows",[]))==18
         and len(packet.get("rows",[]))==18,
         "ACTUAL_PASS_B10_FROZEN_90_NATIVE_SOURCE_SCREEN_ONLY")
    need(receipt.get("admission")==screen.admission(packet["rows"],receipt["rows"]),
         "RECOMPUTED_B10_QUALIFIED_GROUPS_AND_MACHINE_LABEL_COUNTS")
    source={r["id"]:r for r in packet["rows"]}
    decisions={r["id"]:r for r in receipt["rows"]}
    need(len(source)==len(decisions)==18
         and set(source)==set(decisions),
         "EXACT_B10_SOURCE_MANIFEST_SCREEN_JOIN")
    old_shas={r["source_pcm_sha256"] for r in old_rows}
    old_ids={r["id"] for r in old_rows}
    need(len(old_shas)==len(old_rows)==79
         and len(old_ids)==79
         and len(groups[0])==len(groups[1])==TRAIN_GROUPS,
         "ORIGINAL_79_TRAIN_ROWS_35_GROUPS")
    accepted=[]
    for sid,entry in source.items():
        r=decisions[sid]
        need(all(r.get(k)==entry.get(k) for k in
                 ("role","cell_id","source_group_id","speaker_index",
                  "pcm_sha256","wav_sha256","samples"))
             and entry.get("training_allowed") is False
             and entry.get("observed_class") is None
             and r.get("human_gold") is False,
             "ORIGINAL_UNLABELED_TTS_REQUEST_NOT_TRAIN_TRUTH")
        if r["role"]=="train":
            need(r["cell_id"]=="ordinary-time-hard"
                 and r["observed_class"]==0
                 and r["tier"]=="silver"
                 and r["reason"]=="BLIND_ORDINARY_NO_WAKE"
                 and r["source_group_id"] in groups[0]
                 and r["source_group_id"] in groups[1]
                 and r["pcm_sha256"] not in old_shas
                 and sid not in old_ids,
                 "ONLY_SIX_NEW_INDEPENDENT_SILVER_TRAIN_NEGATIVE")
            accepted.append({**entry,
                 "source_pcm_sha256":r["pcm_sha256"],
                 "mono_class":0,"tier":"silver"})
        else:
            need(r["role"]=="development"
                 and r["cell_id"] in ("mero-fast","ordinary-time"),
                 "DEV_ROWS_METADATA_ONLY_NO_GRADIENTS")
    need(len(accepted)==6
         and len({r["source_group_id"] for r in accepted})==6
         and len({r["source_pcm_sha256"] for r in accepted})==6
         and all(r["role"]=="train" for r in accepted),
         "ALL_SIX_QUALIFIED_TRAIN_SOURCES_DISJOINT")
    return sorted(accepted,key=lambda x:x["id"]),{
        "six_new_train_pcm_sha256":sorted(r["source_pcm_sha256"]
                                         for r in accepted),
        "six_qualified_groups":sorted(r["source_group_id"]
                                     for r in accepted),
        "dev_source_metadata_count":12,
        "qualified_dev_groups":receipt["admission"]["development_qualified_source_groups"],
        "dev_pcm_in_gradients":False,
        "new_hard_negative_silver":6}

def cache_new_train(rows,source,mel,mean,scale,torch,np):
    cache={}
    for row in rows:
        path=Path(source)/"wav"/(row["id"]+".wav")
        measured=c.inspect_wav(path)
        need(all(measured[k]==row[k] for k in
                 ("wav_sha256","pcm_sha256","samples")),
             "B10_ACTUAL_NEW_TRAIN_WAV_SHA_AND_SAMPLES")
        pcm,count=phones.pcm_of(path)
        need(count==row["samples"]
             and digest(pcm)==row["source_pcm_sha256"],
             "BYTE_AUTHENTIC_NEW_TRAIN_PCM_ONLY")
        gains={}
        for gain in GAINS:
            raw=b6.b4.gain_pcm(pcm,gain,np)
            gains[gain]={}
            for prefix in PREFIXES:
                silence=b"\x00\x00"*prefix+raw
                args={**row,"samples":count+prefix}
                feat=b6.b4.v5.features(silence,args,torch,np,mel)
                data=((feat-mean)/scale).astype(np.float32)
                need(data.ndim==2 and data.shape[1]==40
                     and np.isfinite(data).all(),
                     "FINITE_HARD_NEGATIVE_40MEL_SUFFIX_INPUT")
                gains[gain][prefix]=torch.from_numpy(data).unsqueeze(0)
        cache[row["id"]]=gains
    need(len(cache)==6,"SIX_NEW_TRAIN_NEGATIVE_SOURCE_CACHE_ONLY")
    return cache

def fit(arm,parent,rows,groups,cache,extra,extra_cache,mel,mean,scale,
        output,torch,np,F,*,auxiliary_loss=None):
    need(arm in ARMS,"B10_TWO_FROZEN_SINGLE_FACTOR_ARMS")
    weight=HARD_WEIGHT if arm=="hard_negative" else 0.0
    torch.manual_seed(SEED)
    net=b6.b4.v5.build_model(torch)
    net.load_state_dict(parent,strict=True)
    net.train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.001)
    sorted_groups={k:sorted(groups[k]) for k in (0,1)}
    trace=[];start=time.perf_counter()
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        prefix=PREFIXES[(step//TRAIN_GROUPS+step%3)%len(PREFIXES)]
        loss_items=[];kl_items=[]
        for klass in (0,1):
            gid=sorted_groups[klass][step%TRAIN_GROUPS]
            candidates=sorted(groups[klass][gid],key=lambda r:r["id"])
            row=candidates[(step//TRAIN_GROUPS)%len(candidates)]
            pair={}
            for gain in GAINS:
                base,logp=b6.suffix_loss(
                    net,cache[row["id"]][gain][prefix],klass,
                    prefix,row["tier"],torch,F,True)
                loss_items.append(base);pair[gain]=logp
            clean,half=pair[1.0],pair[0.5]
            at=prefix//160-3
            need(clean.shape==half.shape and 0<at<half.shape[1]-5,
                 "EXACT_B9_PARENT_MATCHED_HALF_GAIN_CTC_FRAME_GRID")
            kl=F.kl_div(half[:,at:,:],
                clean[:,at:,:].detach(),reduction="none",
                log_target=True).sum(-1).mean()
            need(bool(torch.isfinite(kl)),
                 "FINITE_UNCHANGED_POSTERIOR_KL")
            kl_items.append(kl.clamp(min=0.0,max=b6.KL_CLIP))
        need(len(loss_items)==4 and len(kl_items)==2,
             "ORIGINAL_FOUR_CTC_AND_TWO_GAIN_KL_TERMS")
        loss=sum(loss_items)/4+b6.WEIGHT*sum(kl_items)/2
        new_id=None
        if weight:
            newrow=extra[step%len(extra)];new_id=newrow["id"]
            negative=[b6.suffix_loss(
                net,extra_cache[new_id][gain][prefix],0,prefix,
                newrow["tier"],torch,F,False) for gain in GAINS]
            loss=loss+weight*sum(negative)/len(negative)
        if auxiliary_loss is not None:
            loss=loss+auxiliary_loss(step,net)
        need(bool(torch.isfinite(loss)),
             "FINITE_B10_PAIRED_HARD_NEGATIVE_CTC_OBJECTIVE")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(),5.0,error_if_nonfinite=True)
        optimizer.step()
        if step==0 or (step+1)%16==0:
            trace.append({"step":step+1,"prefix":prefix,
                          "hard_negative_source_id":new_id,
                          "hard_negative_loss_weight":weight,
                          "loss":float(loss.detach())})
    net.eval()
    models=output/"models";models.mkdir(parents=True,exist_ok=False)
    stem=MODEL_STEMS[arm]
    pt=models/(stem+".pt")
    torch.save(net.state_dict(),pt)
    learned=torch.load(pt,map_location="cpu",weights_only=True)
    verified=b6.b4.v5.build_model(torch)
    verified.load_state_dict(learned,strict=True)
    verified.eval()
    need(sum(p.numel() for p in verified.parameters())==PARAMS
         and all(bool(torch.isfinite(p).all()) for p in verified.parameters()),
         "ACTUAL_43876_FINITE_TRAINED_B10_WEIGHTS")
    npz=models/(stem+".npz")
    np.savez(npz,mel_filters=mel,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy()
                  for k,v in verified.state_dict().items()})
    example=cache[sorted(rows,key=lambda r:r["id"])[0]["id"]][1.0][6400]
    onnx=models/(stem+".onnx")
    torch.onnx.export(verified,example,onnx.as_posix(),
        opset_version=17,dynamo=False,
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(all(x.is_file() and x.stat().st_size>10000 for x in (pt,npz,onnx)),
         "ACTUAL_B10_PT_NPZ_ONNX_SAVED")
    return {"arm":arm,"optimizer_updates":STEPS,
            "common_parent":b9.MODELS["B9_control"]["pt"],
            "hard_negative_loss_weight":weight,
            "new_silver_train_sources":len(extra) if weight else 0,
            "trace":trace,
            "optimizer_seconds":round(time.perf_counter()-start,3),
            "files":{file.name:{"sha256":digest(file.read_bytes()),
                               "bytes":file.stat().st_size}
                     for file in (pt,npz,onnx)}}

def execute(args):
    identity=guard()
    out=Path(args.output)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "B10_ONE_NEW_TRAIN_OUTPUT_NO_RETRY")
    out.mkdir()
    write(out/"STARTED.json",{"schema":SCHEMA,"run":identity,
         "maximum_optimizer_updates":2*STEPS,
         "new_tts_asr_phone_calls":0,"dev_gradients":0,
         "dev_model_forwards":0,"cal_locked_access":False,
         "product_promotion_allowed":False})
    try:
        receipts=remote(os.environ.get("GH_TOKEN"))
        old_rows,dev,source,groups=b6.source_audit(args)
        need(len(old_rows)==79 and len(groups[0])==len(groups[1])==TRAIN_GROUPS
             and all(r["role"]=="train" for r in old_rows)
             and all(r["role"]=="development" for r in dev),
             "EXACT_OLD_SEVENTY_NINE_TRAIN_AND_DEV_METADATA_ONLY")
        extra,approved=training_rows(
            args.b10_source,args.b10_screen,old_rows,groups)
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        weights,b9source=b9.verify_b9_archive(
            args.b9_model_zip,torch,np)
        net,mel,mean,scale=weights["B9_control"]
        prior={k:v.detach().clone() for k,v in net.state_dict().items()}
        need(sorted(r["source_pcm_sha256"] for r in old_rows)==
             sorted(b9source["train_pcm_sha256"]),
             "EXACT_B9_PARENT_79_TRAIN_ORIGINAL_SHA_LIST")
        old_cache=b6.b4.cache_train(old_rows,args.old_archives,
                  args.fresh_source,mel,mean,scale,torch,np)
        extra_cache=cache_new_train(
            extra,args.b10_source,mel,mean,scale,torch,np)
        write(out/"SOURCE.json",{"schema":SCHEMA,"old_train_originals":79,
            "old_train_groups":TRAIN_GROUPS,
            "old_train_shas":sorted(r["source_pcm_sha256"] for r in old_rows),
            "new_train_admission":approved,"model_artifacts":receipts,
            "parent_pt_sha256":b9.MODELS["B9_control"]["pt"],
            "dev_pcm_in_gradients":False,
            "dev_model_forwards":0,"cal_locked_opened":False})
        write(out/"ENVIRONMENT.json",{
            "python":sys.version,"torch":torch.__version__,
            "numpy":np.__version__,"platform":platform.platform(),
            "mkldnn_enabled":torch.backends.mkldnn.enabled,
            "torch_threads":torch.get_num_threads(),
            "runner_image":os.environ.get("ImageVersion")})
        control_dir=out/"matched_control";control_dir.mkdir()
        challenge_dir=out/"new_hardneg";challenge_dir.mkdir()
        c0=fit("control",prior,old_rows,groups,old_cache,extra,extra_cache,
               mel,mean,scale,control_dir,torch,np,F)
        c1=fit("hard_negative",prior,old_rows,groups,old_cache,extra,extra_cache,
               mel,mean,scale,challenge_dir,torch,np,F)
        def load_pt(folder,stem):
            return torch.load(folder/"models"/(stem+".pt"),
                              map_location="cpu",weights_only=True)
        a=load_pt(control_dir,MODEL_STEMS["control"])
        b=load_pt(challenge_dir,MODEL_STEMS["hard_negative"])
        need(set(a)==set(b)==set(prior)
             and all(a[k].shape==b[k].shape==prior[k].shape for k in a),
             "EXACT_SAME_FSMN_CTC_GEOMETRY_BOTH_ARMS")
        distinct=sum(int(torch.count_nonzero(a[k]!=b[k]).item()) for k in a)
        need(distinct>0 and len(c0["trace"])==len(c1["trace"])>8
             and [(x["step"],x["prefix"]) for x in c0["trace"]]==
                 [(x["step"],x["prefix"]) for x in c1["trace"]],
             "REAL_B10_SINGLE_FACTOR_WEIGHT_DIVERGENCE_AND_STEP_SCHEDULE")
        receipt={"schema":SCHEMA,
            "status":"RESEARCH_B10_REAL_SILVER_TRAIN_HARDNEG_PAIRED_WEIGHTS_NO_PROMOTION",
            "frozen_b9_parent_pt_sha256":b9.MODELS["B9_control"]["pt"],
            "trained_model_parameters":PARAMS,"source_train_groups":TRAIN_GROUPS,
            "old_train_originals":79,"new_silver_hard_negatives":6,
            "new_train_group_count":6,
            "source_screen_run":44986730086,
            "source_screen_artifact":14740257228,
            "single_factor":"ADDITIONAL_SIX_SOURCE_QUALIFIED_TRAIN_HARD_NEGATIVE_CTC_LOSS",
            "control_new_loss_weight":0.0,
            "challenger_new_loss_weight":HARD_WEIGHT,
            "common_seed":SEED,"common_optimizer":"AdamW",
            "common_gains":list(GAINS),"common_prefixes":list(PREFIXES),
            "control":c0,"challenger":c1,
            "actual_optimizer_updates":STEPS*2,
            "different_trained_float_values":distinct,
            "new_audio_asr_phone_calls":0,"dev_gradients":0,
            "dev_model_forwards":0,"cal_locked_opened":False,
            "new_unseen_DEV_frozen_eval":"NOT_RUN",
            "b10_c11_arm32_parity":"NOT_RUN",
            "human_3_5m_frr":"NOT_MEASURED",
            "natural_far_per_hour":"NOT_MEASURED",
            "ssc305_physical":"NOT_RUN",
            "model_selection_allowed":False,
            "product_promotion_allowed":False,
            "B0_only_installed_software_baseline":True}
        write(out/"RESULT.json",receipt)
        print(json.dumps({"status":receipt["status"],
             "actual_optimizer_updates":receipt["actual_optimizer_updates"],
             "control_files":c0["files"],"challenger_files":c1["files"],
             "different_trained_float_values":distinct,
             "product_promotion_allowed":False},sort_keys=True))
    except BaseException as error:
        write(out/"STOP.json",{"schema":SCHEMA,"status":"STOP_B10_NO_QUALIFIED_WEIGHTS",
             "reason":str(error)[:500],"error_type":type(error).__name__,
             "cal_locked_opened":False,"model_selection_allowed":False,
             "product_promotion_allowed":False})
        raise

def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--guard","--audit","--execute"):
        mode.add_argument(flag,action="store_true")
    for key in ("old-archives","old-audit","fresh-source","fresh-screen",
                "source-gate","prospective-source","prospective-screen",
                "original-model-zip","b9-model-zip","b10-source",
                "b10-screen","output"):
        p.add_argument("--"+key,type=Path)
    a=p.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    required=("old_archives","old_audit","fresh_source",
              "fresh_screen","source_gate","prospective_source",
              "prospective_screen","original_model_zip",
              "b10_source","b10_screen")
    need(all(getattr(a,k) is not None for k in required),
         "EXACT_OLD_TRAIN_B9_PARENT_AND_B10_SILVER_SCREEN_ASSETS")
    if a.audit:
        rows,dev,receipt,groups=b6.source_audit(a)
        approved,pins=training_rows(a.b10_source,a.b10_screen,rows,groups)
        print(json.dumps({"status":"B10_SOURCE_ONLY_REAL_SILVER_NEGATIVE_QUALIFIED",
            "original_train_rows":len(rows),
            "old_train_groups":len(groups[0]),
            "new_silver_train_originals":len(approved),
            "qualified_dev_groups":len(pins["qualified_dev_groups"]),
            "new_optimizer_updates":0,"dev_model_forwards":0},sort_keys=True))
        return
    need(a.b9_model_zip is not None and a.output is not None,
         "ORIGINAL_B9_MODEL_AND_B10_NEW_OUTPUT_REQUIRED")
    execute(a)

if __name__=="__main__":
    main()
