#!/usr/bin/env python3
"""B9 real TRAIN-only paired test: low-gain HEY presence without frame pseudolabels.

Use the same frozen successful B6 FSMN-CTC parent, same 35 authenticated TRAIN
source groups, identical CTC suffix+blank-prefix, AdamW, seed, 192 updates,
clean/half-gain train examples and posterior-KL as the B8 control. The single
new factor is a small clip-level HEY-presence probability loss on half-gain
positive TRAIN suffixes. No inferred HEY boundary, DEV520 gradients/forwards,
new generation, threshold tuning, CAL/LOCKED, or automatic model promotion.
The original B7 failed exact replay and the B8 quality nonpromotion remain.
"""
from __future__ import annotations
import argparse
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
spec=importlib.util.spec_from_file_location(
    "b9_frozen_b8_training_reference",
    HERE/"english_v6_b8_frozen_b6_paired_ctc_trainonce.py")
b8=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b8)
b6=b8.b6
REPO="jiying2007/kws-data"
BEFORE="0ac4c653a7ae6b3cf90812ce6bd0402b4ba0d04f"
SCHEMA="english-v6-b9-half-gain-positive-hey-presence-v1"
STEPS=192
TRAIN_GROUPS=35
PARAMS=43876
WEIGHT=0.03
STEM="mero43-b9-halfgain-hey-presence192"
B8_TRAIN={"run":44746571932,
          "sha":"453b26bdd4ab2bdc96b27fe0dfb9ac10bdd38711",
          "artifact":14634066012,"bytes":1001683,
          "digest":"6015c6269f87654193d7350afd0c5b44f799f5f71b5303e41f2734c17006d15b"}
B8_DEV={"run":44766393361,"sha":BEFORE,
        "artifact":14645830993,"bytes":49216,
        "digest":"3f39922e0ec0aad5baa4ffc161c46ebe95d3507defe158442bb4a761e0f0eee1"}

def need(ok,reason):
    if not ok:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def save(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def source_check():
    prev=b8.source_check()
    need(prev["status"]=="SOURCE_ONLY_NO_OPTIMIZER_OR_MODEL_FORWARD"
         and prev["exact_immutable_initial_model_zip"]==14615025178
         and prev["exact_initial_pt_sha256"]==
              "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04"
         and prev["source_groups"]==TRAIN_GROUPS
         and prev["per_arm_real_optimizer_updates"]==STEPS
         and b6.STEPS==STEPS
         and b6.SOURCE_GROUPS==TRAIN_GROUPS
         and b6.PARAMS==PARAMS
         and b6.SEED==20261120
         and b6.PREFIXES==(6400,12800,19200)
         and b6.GAINS==(1.0,0.5)
         and b6.WEIGHT==WEIGHT
         and b6.KL_CLIP==8.0
         and b6.LR==0.00006
         and B8_TRAIN["run"]==44746571932
         and B8_DEV["artifact"]==14645830993
         and B8_DEV["sha"]==BEFORE,
         "B9_EXACT_B6_B8_IMMUTABLE_SOURCE_AND_ONE_FACTOR_TRAIN")
    return {
      "schema":SCHEMA,"status":"SOURCE_ONLY_NO_TRAIN_OR_MODEL_FORWARD",
      "keyword":"Hey Mero","architecture":"FSMN-CTC-43876",
      "experiment":"B9_INDEPENDENT_SINGLE_FACTOR_HALF_GAIN_POSITIVE_HEY_PRESENCE",
      "same_frozen_B6_parent_pt_sha256":b8.B6["pt_sha256"],
      "same_35_source_disjoint_train_groups":TRAIN_GROUPS,
      "control_weight":0.0,"challenger_weight":WEIGHT,
      "unchanged_suffix_ctc":True,"unchanged_ordinary_ctc_weight":0.60,
      "unchanged_blank_prefix":True,"unchanged_gain_kl":0.03,
      "new_factor":"CLIP_LEVEL_HEY_PRESENCE_POSITIVE_HALF_GAIN_TRAIN_ONLY",
      "no_guessed_hey_frame_labels":True,
      "same_seed_optimizer_source_prefixes_gain_policy":True,
      "per_arm_optimizer_updates":STEPS,
      "max_actual_optimizer_updates":2*STEPS,
      "prior_B7_exact_replay":"FAILED_UNCHANGED",
      "B8_dev520_is_exposed_and_not_train":True,
      "B8_DEV_frozen_run":B8_DEV["run"],
      "B8_DEV_zip_sha256":B8_DEV["digest"],
      "dev_gradients":0,"dev_model_forwards":0,
      "tts_asr_phone_calls":0,"cal_locked_access":False,
      "automated_model_selection":False,"product_promotion":False,
      "new_scheduled_patrol":False,"private_repo":False,"work_codex":False}

def provenance():
    previous=b8.provenance()
    token=os.environ.get("GH_TOKEN","")
    need(len(token)>10,"GITHUB_READ_ONLY_PROVENANCE_TOKEN_REQUIRED")
    def get(path):
        req=urllib.request.Request(
          "https://api.github.com/repos/"+REPO+"/actions/"+path,
          headers={"Accept":"application/vnd.github+json",
                   "User-Agent":"kws-b9-half-gain-train-only",
                   "Authorization":"Bearer "+token})
        with urllib.request.urlopen(req,timeout=40) as res:
            raw=res.read(1000001)
        need(len(raw)<=1000000,"BOUNDED_GITHUB_PROVENANCE")
        return json.loads(raw)
    for record in (B8_TRAIN,B8_DEV):
        run=get("runs/"+str(record["run"]))
        need(run.get("id")==record["run"]
             and run.get("head_sha")==record["sha"]
             and run.get("head_branch")=="main"
             and run.get("event")=="push"
             and run.get("run_attempt")==1
             and run.get("status")=="completed"
             and run.get("conclusion")=="success",
             "B9_ACTUAL_B8_TRAIN_AND_DEV520_SUCCESS_REQUIRED")
        art=get("artifacts/"+str(record["artifact"]))
        need(art.get("id")==record["artifact"]
             and art.get("size_in_bytes")==record["bytes"]
             and art.get("digest")=="sha256:"+record["digest"]
             and art.get("expired") is False
             and art.get("workflow_run",{}).get("id")==record["run"]
             and art.get("workflow_run",{}).get("head_sha")==record["sha"],
             "B9_B8_IMMUTABLE_ORIGINAL_ARTIFACT_REQUIRED")
    return {"old_B7_provenance":previous,
            "B8_real_trained_two_arms":B8_TRAIN["artifact"],
            "B8_real_DEV520_99_forward_result":B8_DEV["artifact"],
            "B8_not_a_model_promotion":True}

def validate_main_event(env,event,parent):
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),
        "FIRST_MAIN_ONLY_NO_PR_NO_RETRY_NO_SCHEDULE")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch(r"[0-9a-f]{40}",head) is not None
         and re.fullmatch(r"[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None
         and event.get("before")==BEFORE
         and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("deleted") is False
         and event.get("forced") is False
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and parent==BEFORE,
         "EXACT_B9_PROTECTED_MAIN_SUCCESSOR_NO_REPLAY")
    return {"run":int(env["GITHUB_RUN_ID"]),"head":head,"parent":BEFORE}

def guard():
    source_check()
    event=json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
    parent=subprocess.check_output(["git","rev-parse","HEAD^1"],cwd=ROOT,
                                   text=True).strip()
    proof=validate_main_event(os.environ,event,parent)
    proof["historical_provenance"]=provenance()
    return proof

def presence_loss(logp,prefix,tier,torch):
    """P(any HEY token after prefix) over entire suffix, no guessed HEY time."""
    start=prefix//160-3
    need(logp.ndim==3 and logp.shape[0]==1 and logp.shape[2]==4
         and 0<start<logp.shape[1]-5,
         "B9_ORIGINAL_SUFFIX_GRID_AND_FOUR_CTC_TOKENS")
    p=logp[:,start:,1].exp().clamp(min=1.0e-7,max=1.0-1.0e-7)
    log_no_hey=torch.log1p(-p).sum(dim=1)
    prob_ever_hey=-torch.expm1(log_no_hey)
    loss=-torch.log(prob_ever_hey.clamp(min=1.0e-7)).mean()
    need(bool(torch.isfinite(loss)),"B9_FINITE_HEY_PRESENCE_LOSS")
    return loss*(1.0 if tier=="silver" else 0.75)

def synthetic_gradient_check(torch):
    # 64 invented frames; not actual audio inference or training.
    values=torch.full((1,64,4),-2.0)
    values[:,:,0]=3.0
    values[:,:,1]=-8.0
    values.requires_grad_()
    low=presence_loss(values.log_softmax(-1),6400,"silver",torch)
    low.backward()
    need(values.grad is not None
         and bool(torch.isfinite(values.grad).all())
         and float(values.grad[:,:,1].abs().sum())>0,
         "B9_SYNTHETIC_HEY_LOSS_FINITE_NONZERO_GRADIENT")
    high=values.detach().clone()
    high[:,:,1]=0.0
    better=presence_loss(high.log_softmax(-1),6400,"silver",torch)
    need(float(better)<float(low),
         "B9_HEY_EXISTENCE_LOSS_MONOTONIC_IN_INVENTED_LOGITS")
    return {"status":"PASS_INVENTED_LOGIT_LOSS_GRADIENT_ONLY",
            "actual_acoustic_model_forwards":0}

def challenger_fit(parent,rows,groups,cache,mel,mean,scale,out,torch,np,F):
    torch.manual_seed(b6.SEED)
    net=b6.b4.v5.build_model(torch)
    net.load_state_dict(parent,strict=True)
    net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=b6.LR,weight_decay=0.001)
    trace=[]
    sorted_groups={k:sorted(groups[k]) for k in (0,1)}
    start_time=time.perf_counter()
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        prefix=b6.PREFIXES[(step//35+step%3)%len(b6.PREFIXES)]
        losses=[];consistency=[];onsets=[]
        for klass in (0,1):
            gid=sorted_groups[klass][step%TRAIN_GROUPS]
            candidates=sorted(groups[klass][gid],key=lambda r:r["id"])
            row=candidates[(step//TRAIN_GROUPS)%len(candidates)]
            pair={}
            for gain in b6.GAINS:
                x=cache[row["id"]][gain][prefix]
                base,logp=b6.suffix_loss(
                    net,x,klass,prefix,row["tier"],torch,F,True)
                losses.append(base)
                pair[gain]=logp
                if klass==1 and gain==0.5:
                    onsets.append(presence_loss(logp,prefix,row["tier"],torch))
            clean,half=pair[1.0],pair[0.5]
            start=prefix//160-3
            need(clean.shape==half.shape and 0<start<half.shape[1]-5,
                 "B9_B6_MATCHED_GAIN_FRAME_GRID")
            kl=F.kl_div(half[:,start:,:],clean[:,start:,:].detach(),
                        reduction="none",log_target=True).sum(-1).mean()
            need(bool(torch.isfinite(kl)),"B9_ORIGINAL_B6_FINITE_KL")
            consistency.append(kl.clamp(min=0.0,max=b6.KL_CLIP))
        need(len(losses)==4 and len(consistency)==2 and len(onsets)==1,
             "B9_ONLY_ONE_HALF_GAIN_POSITIVE_TRAIN_ONSET")
        loss=sum(losses)/len(losses)+b6.WEIGHT*sum(consistency)/2
        loss=loss+WEIGHT*onsets[0]
        need(bool(torch.isfinite(loss)),"B9_FINITE_TRAIN_OBJECTIVE")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.0,error_if_nonfinite=True)
        opt.step()
        if step==0 or (step+1)%16==0:
            trace.append({"step":step+1,"prefix_samples":prefix,
                          "loss":float(loss.detach()),"gains":list(b6.GAINS),
                          "hey_presence_weight":WEIGHT})
    net.eval()
    dest=out/"models"
    dest.mkdir(exist_ok=True)
    pt=dest/(STEM+".pt")
    torch.save(net.state_dict(),pt)
    saved=torch.load(pt,weights_only=True,map_location="cpu")
    checker=b6.b4.v5.build_model(torch)
    checker.load_state_dict(saved,strict=True)
    checker.eval()
    need(sum(p.numel() for p in checker.parameters())==PARAMS
         and all(bool(torch.isfinite(p).all()) for p in checker.parameters()),
         "B9_FINITE_43876_SAVED_WEIGHTS")
    npz=dest/(STEM+".npz")
    np.savez(npz,mel_filters=mel,train_mean=mean,train_scale=scale,
            **{k:v.detach().cpu().numpy()
               for k,v in checker.state_dict().items()})
    example=cache[sorted(rows,key=lambda r:r["id"])[0]["id"]][1.0][6400]
    onnx=dest/(STEM+".onnx")
    torch.onnx.export(checker,example,onnx.as_posix(),opset_version=17,
        dynamo=False,input_names=["fbank40"],
        output_names=["frame_ctc_logits"],
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(all(p.is_file() and p.stat().st_size>10000 for p in (pt,npz,onnx)),
         "B9_REAL_PT_NPZ_ONNX_REQUIRED")
    return {"arm":"half_gain_hey_presence",
            "optimizer_updates":STEPS,"train_source_groups":TRAIN_GROUPS,
            "changed_factor":"POSITIVE_HALF_GAIN_HEY_EXISTENCE_LOSS_ONLY",
            "hey_presence_weight":WEIGHT,
            "no_phone_frame_pseudolabels":True,
            "trace":trace,"elapsed_optimizer_seconds":
                round(time.perf_counter()-start_time,3),
            "files":{p.name:{"sha256":sha(p.read_bytes()),
                              "bytes":p.stat().st_size}
                     for p in (pt,npz,onnx)}}

def execute(a):
    identity=guard()
    out=a.output
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "B9_SINGLE_BOUND_TRAIN_EXECUTION")
    out.mkdir()
    save(out/"STARTED.json",{"schema":SCHEMA,"identity":identity,
         "max_optimizer_updates":STEPS*2,"new_audio":0,"dev_gradients":0,
         "dev_model_forwards":0,"cal_locked_access":False,
         "model_selection_allowed":False})
    try:
        rows,dev,source,groups=b6.source_audit(a)
        need(len(groups[0])==len(groups[1])==TRAIN_GROUPS
             and all(r["role"]=="train" for r in rows)
             and all(r["role"]=="development" for r in dev),
             "EXACT_35_TRAIN_GROUPS_AND_DEV_METADATA_ONLY")
        train_hashes=sorted(r["source_pcm_sha256"] for r in rows)
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        synthetic_gradient_check(torch)
        parent,prior_hashes,features=b8.load_original_B6(a.b6_model_zip,torch)
        need(train_hashes==prior_hashes,
             "EXACT_B6_B9_TRAIN_PCM_SHA_LIST")
        b3,mel,mean,scale=b6.b4.authenticate_b3(a.b3_model_zip,torch,np)
        need(set(parent)==set(b3)
             and all(parent[k].shape==b3[k].shape for k in parent),
             "EXACT_B6_B9_FROZEN_GEOMETRY")
        with np.load(io.BytesIO(features),allow_pickle=False) as saved:
            need(np.array_equal(saved["mel_filters"],mel)
                 and np.array_equal(saved["train_mean"],mean)
                 and np.array_equal(saved["train_scale"],scale),
                 "EXACT_B6_FROZEN_LOGMEL_NORMALIZATION")
        del b3
        save(out/"SOURCE.json",{
            "schema":SCHEMA,"train_pcm_sha256":train_hashes,
            "source_evidence":source,
            "original_B6_pt_sha256":b8.B6["pt_sha256"],
            "dev_metadata_count":len(dev),
            "dev_pcm_in_gradients":False})
        save(out/"ENVIRONMENT.json",{
            "python":sys.version,"torch":torch.__version__,
            "numpy":np.__version__,"platform":platform.platform(),
            "mkldnn_enabled":torch.backends.mkldnn.enabled,
            "torch_threads":torch.get_num_threads(),
            "runner_image":os.environ.get("ImageVersion")})
        cache=b6.b4.cache_train(rows,a.old_archives,a.fresh_source,
                               mel,mean,scale,torch,np)
        control_dir=out/"control_b6_weight0"
        challenger_dir=out/"challenger_half_hey003"
        control_dir.mkdir();challenger_dir.mkdir()
        control=b6.fit("consistency",parent,rows,groups,cache,
                       mel,mean,scale,control_dir,torch,np,F)
        need(control["optimizer_updates"]==STEPS
             and control["gain_consistency_enabled"] is True,
             "B9_TRUE_B6_SAME_JOB_CONTROL")
        challenger=challenger_fit(
            parent,rows,groups,cache,mel,mean,scale,challenger_dir,torch,np,F)
        need(challenger["optimizer_updates"]==STEPS,
             "B9_TRUE_ONE_FACTOR_CHALLENGER")
        control_pt=control_dir/"models"/(b6.ARMS["consistency"]+".pt")
        challenger_pt=challenger_dir/"models"/(STEM+".pt")
        cw=torch.load(control_pt,map_location="cpu",weights_only=True)
        qw=torch.load(challenger_pt,map_location="cpu",weights_only=True)
        need(set(cw)==set(qw)==set(parent)
             and all(cw[k].shape==qw[k].shape==parent[k].shape for k in cw),
             "B9_MATCHED_FSMN_GEOMETRY")
        difference=sum(int(torch.count_nonzero(cw[k]!=qw[k]).item())
                       for k in cw)
        need(difference>0,"B9_REAL_TRAINED_WEIGHT_DIVERGENCE_REQUIRED")
        control_trace=[(r["step"],r["prefix_samples"]) for r in control["trace"]]
        trial_trace=[(r["step"],r["prefix_samples"]) for r in challenger["trace"]]
        need(control_trace==trial_trace and len(control_trace)>8,
             "B9_SAME_SEED_AND_PREFIX_SCHEDULE")
        result={
          "schema":SCHEMA,
          "status":"RESEARCH_B9_REAL_HALF_GAIN_HEY_TRAIN_NO_PROMOTION",
          "frozen_B6_parent_pt_sha256":b8.B6["pt_sha256"],
          "same_job_parent":True,"same_35_train_source_groups":TRAIN_GROUPS,
          "single_factor":"CLIP_LEVEL_HEY_PRESENCE_POSITIVE_HALF_GAIN",
          "control_new_loss_weight":0.0,"challenger_new_loss_weight":WEIGHT,
          "unchanged_ordinary_ctc_weight":0.60,
          "unchanged_gain_KL_weight":b6.WEIGHT,
          "same_seed":b6.SEED,
          "same_prefixes":list(b6.PREFIXES),
          "same_gains":list(b6.GAINS),
          "control":control,"challenger":challenger,
          "actual_optimizer_updates":2*STEPS,
          "model_parameters":PARAMS,
          "different_trained_float_values":difference,
          "B7_historical_exact_replay":"FAILED_UNCHANGED",
          "B8_DEV520_quality":"NO_B8_GAIN_OVER_FROZEN_B6_PROVEN",
          "new_tts_asr_phone":0,"dev_gradients":0,
          "dev_model_forwards":0,"cal_locked_opened":False,
          "fresh_independent_B9_DEV":"NOT_RUN",
          "c11_arm32_parity":"NOT_RUN","ssc305_physical":"NOT_RUN",
          "human_3_5m_frr":"NOT_MEASURED",
          "natural_far_per_hour":"NOT_MEASURED",
          "model_selection_allowed":False,
          "product_promotion_allowed":False}
        save(out/"RESULT.json",result)
        print(json.dumps({"status":result["status"],
           "actual_optimizer_updates":2*STEPS,
           "control_pt_sha256":control["files"][b6.ARMS["consistency"]+".pt"]["sha256"],
           "challenger_pt_sha256":challenger["files"][STEM+".pt"]["sha256"],
           "different_tensor_values":difference,
           "product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        save(out/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_B9_NO_QUALIFIED_MODEL",
             "error_type":type(exc).__name__,"reason":str(exc)[:450],
             "B7_historical_exact_replay":"FAILED_UNCHANGED",
             "model_selection_allowed":False,
             "product_promotion_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    mode=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--provenance","--guard",
                 "--audit","--toy","--execute"):
        mode.add_argument(flag,action="store_true")
    for key in ("old-archives","old-audit","fresh-source","fresh-screen",
                "source-gate","prospective-source","prospective-screen",
                "original-model-zip","b3-model-zip","b6-model-zip","output"):
        ap.add_argument("--"+key,type=Path)
    a=ap.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.provenance:
        print(json.dumps(provenance(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    if a.toy:
        import torch
        print(json.dumps(synthetic_gradient_check(torch),sort_keys=True));return
    keys=("old_archives","old_audit","fresh_source","fresh_screen",
          "source_gate","prospective_source","prospective_screen",
          "original_model_zip")
    need(all(getattr(a,k) is not None for k in keys),
         "EIGHT_EXISTING_TRAIN_SOURCE_ASSETS_REQUIRED")
    if a.audit:
        rows,dev,_source,groups=b6.source_audit(a)
        print(json.dumps({"status":"B9_ONLY_SOURCE_AUDIT",
            "train_rows":len(rows),"train_groups":len(groups[0]),
            "dev_metadata_only":len(dev),
            "new_model_forwards":0,"new_optimizer_updates":0},sort_keys=True))
        return
    need(a.b3_model_zip and a.b6_model_zip and a.output,
         "FROZEN_B3_GEOMETRY_B6_PARENT_AND_OUTPUT_REQUIRED")
    execute(a)

if __name__=="__main__":
    main()
