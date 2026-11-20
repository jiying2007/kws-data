#!/usr/bin/env python3
"""B7 one-factor TRAIN-only ordinary CTC weight vs exact B6 gain-KL replay.

Actual 35 authenticated TRAIN voice groups only. Frozen B3 parent and identical
B6 clean/half posterior consistency, seed, prefixes, optimizer, augmentation.
The challenger changes ONLY ordinary CTC loss coefficient 0.60 -> 0.85.
No observed DEV512 original enters gradients, no extra audio, ASR, CAL or LOCKED.
Model and threshold promotion prohibited until new independent DEV source QA.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
_spec=importlib.util.spec_from_file_location(
    "b7_frozen_b6_gain_kl_reference",
    HERE/"english_v6_b6_gain_posterior_consistency_trainonly_once.py")
b6=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b6)
REPO="jiying2007/kws-data"
BEFORE="04001cd9c293ddcea0ffe10652cfa464885b3b17"
SCHEMA="english-v6-b7-trainonly-ordinary-ctc-weight-085-v1"
OLD_ORDINARY_WEIGHT=0.60
NEW_ORDINARY_WEIGHT=0.85
STEPS=192
TRAIN_GROUPS=35
PARAMETERS=43876
SEED=20261120
GAINS=(1.0,0.5)
PREFIXES=(6400,12800,19200)
KL_WEIGHT=0.03
KL_CLIP=8.0
STEM="mero43-b7-ordinary-ctc085-192"
B6_MODEL={
  "run":44704535156,
  "head":"e46dfa72e7f824a888ae6510a9951f1fe518b222",
  "id":14615025178,
  "name":"english-v6-b6-replay-vs-gain-consistency-44704535156",
  "bytes":1000729,
  "zip_sha256":"1c310b0d4945f7aa40f2576f231c94eb48b33316a6c4f79c20215b1ac80bab8d",
  "member":"models/mero43-b6-gain-consistency192.pt",
  "pt_sha256":"0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04",
}
B6_DEV512={
  "run":44710802470,
  "head":"d9bbb5d0ea29ea2774067734ba24d55eccef97fc",
  "id":14616493798,
  "name":"english-v6-b6-dev512-three-frozen-paired-44710802470",
  "bytes":41502,
  "zip_sha256":"b900473e57a5b53793a42020838371c1c12f4ee70d73480f2f68c03aab37801f",
}
B6_SAVED_CTC={
  "run":44718953294,
  "head":BEFORE,
  "id":14619669982,
  "name":"english-v6-b6-dev512-saved-ctc-forensics-44718953294",
  "bytes":1712,
  "zip_sha256":"25351b75c0bfe4248ce4d62774dfa1ab207afeeceb5422e935116e013bfc791e",
}

def need(ok,reason):
    if not ok:raise ValueError(reason)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def save(path,obj):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(obj,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        stream.write("\n")

def source_check():
    original=b6.source_check()
    need(original["status"]=="SOURCE_ONLY_NO_TRAIN_OR_MODEL_FORWARD"
         and original["only_changed_factor"]==
              "CLEAN_TEACHER_HALF_GAIN_SUFFIX_POSTERIOR_KL"
         and original["train_source_groups"]==TRAIN_GROUPS
         and original["per_arm_optimizer_updates"]==STEPS
         and original["baseline_replay_required"] is True
         and original["same_43876_param_b3_parent"] is True
         and original["new_tts_asr"]==0
         and original["development_model_forwards"]==0
         and original["cal_locked_opened"] is False
         and b6.STEPS==STEPS and b6.SOURCE_GROUPS==TRAIN_GROUPS
         and b6.PARAMS==PARAMETERS and b6.SEED==SEED
         and b6.PREFIXES==PREFIXES and b6.GAINS==GAINS
         and b6.WEIGHT==KL_WEIGHT and b6.KL_CLIP==KL_CLIP
         and b6.LR==0.00006
         and NEW_ORDINARY_WEIGHT==0.85
         and OLD_ORDINARY_WEIGHT==0.60
         and 0<NEW_ORDINARY_WEIGHT-OLD_ORDINARY_WEIGHT<0.5
         and len(B6_MODEL["pt_sha256"])==64
         and all(len(record["zip_sha256"])==64
                 for record in (B6_MODEL,B6_DEV512,B6_SAVED_CTC))
         and B6_SAVED_CTC["head"]==BEFORE,
         "BYTE_AUTHENTIC_B6_BASELINE_AND_ONLY_ORDINARY_CTC_WEIGHT_DELTA")
    return {
       "schema":SCHEMA,"status":"SOURCE_ONLY_NO_TRAIN_OR_MODEL_FORWARD",
       "keyword":"Hey Mero",
       "parent_B3_artifact":b6.b4.B3["id"],
       "immutable_B6_artifact":B6_MODEL["id"],
       "immutable_B6_pt_sha256":B6_MODEL["pt_sha256"],
       "B6_DEV512_readonly":B6_DEV512["id"],
       "B6_saved_false_wake_readonly":B6_SAVED_CTC["id"],
       "train_voice_groups":TRAIN_GROUPS,
       "matched_control_optimizer_updates":STEPS,
       "challenger_optimizer_updates":STEPS,
       "maximum_new_optimizer_updates":STEPS*2,
       "one_changed_factor":"ORDINARY_CTC_WEIGHT_060_TO_085_TRAIN_ONLY",
       "ordinary_baseline_weight":OLD_ORDINARY_WEIGHT,
       "ordinary_challenger_weight":NEW_ORDINARY_WEIGHT,
       "gain_kl_weight_unchanged":KL_WEIGHT,
       "same_seed":True,"same_prefixes_and_source_groups":True,
       "B6_tensor_replay_mandatory":True,
       "new_tts_asr_phone_calls":0,"development_optimizer_rows":0,
       "new_model_inference_on_dev":0,
       "calibration_or_locked_access":False,
       "model_selection_allowed":False,"product_promotion_allowed":False,
       "work_codex":False,"private_repo":False,"new_schedule":False,
    }

def provenance():
    token=os.environ.get("GH_TOKEN","")
    need(len(token)>10,"READONLY_PUBLIC_GITHUB_ACTIONS_TOKEN")
    older=b6.provenance()
    root="https://api.github.com/repos/"+REPO+"/actions/"
    def get(suffix):
        req=urllib.request.Request(root+suffix,headers={
             "Authorization":"Bearer "+token,
             "Accept":"application/vnd.github+json",
             "User-Agent":"kws-v6-b7-ordinary-ctc-trainonly"})
        with urllib.request.urlopen(req,timeout=35) as resp:
            raw=resp.read(1000001)
        need(len(raw)<=1000000,"BOUNDED_ORIGINAL_METADATA")
        return json.loads(raw)
    for record in (B6_MODEL,B6_DEV512,B6_SAVED_CTC):
        run=get("runs/"+str(record["run"]))
        need(run.get("id")==record["run"]
             and run.get("head_sha")==record["head"]
             and run.get("head_branch")=="main"
             and run.get("event")=="push"
             and run.get("run_attempt")==1
             and run.get("status")=="completed"
             and run.get("conclusion")=="success",
             "ACTUAL_FROZEN_B6_OR_EXPOSED_DEV_RUN")
        art=get("artifacts/"+str(record["id"]))
        need(art.get("id")==record["id"]
             and art.get("name")==record["name"]
             and art.get("size_in_bytes")==record["bytes"]
             and art.get("digest")=="sha256:"+record["zip_sha256"]
             and art.get("expired") is False
             and art.get("workflow_run",{}).get("id")==record["run"]
             and art.get("workflow_run",{}).get("head_sha")==record["head"],
             "ORIGINAL_FROZEN_B6_RESULT_SAVED_HASH")
    return {"B6":B6_MODEL["id"],"DEV512_research_only":B6_DEV512["id"],
            "saved_DEV512_forensics":B6_SAVED_CTC["id"],
            "train_ancestry":older}

def guard():
    source_check()
    env=os.environ
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),
        "FIRST_PROTECTED_PUBLIC_MAIN_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "REAL_FIRST_MAIN_JOB_ID")
    event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("deleted") is False and event.get("forced") is False
         and subprocess.check_output(
             ["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()==BEFORE,
         "EXACT_FIRST_MAIN_SUCCESSOR_NO_REPLAY_OR_DEV_ACCESS")
    return {"run_id":env["GITHUB_RUN_ID"],"head":head,"before":BEFORE,
            "originals":provenance()}

def verify_B6_replay(archive_path,control_pt,torch):
    archive=Path(archive_path)
    need(archive.is_file() and not archive.is_symlink()
         and archive.stat().st_size==B6_MODEL["bytes"]
         and digest(archive.read_bytes())==B6_MODEL["zip_sha256"],
         "BYTE_AUTHENTIC_B6_ORIGINAL_MODEL_ZIP")
    with zipfile.ZipFile(archive) as z:
        result=json.loads(z.read("RESULT.json"))
        need(result.get("schema")==b6.SCHEMA
             and result.get("status")==
                "RESEARCH_B6_TRAIN_ONLY_GAIN_POSTERIOR_KL_NO_PROMOTION"
             and result.get("actual_optimizer_updates")==384
             and result.get("train_source_groups")==TRAIN_GROUPS
             and result.get("model_parameters")==PARAMETERS
             and result.get("only_changed_train_factor")==
                "STOPGRAD_CLEAN_TO_HALF_SUFFIX_POSTERIOR_KL"
             and result.get("B4_half_baseline_replay",{}).get("status")==
                "PASS_EXACT_B4_HALF_TENSOR_REPLAY"
             and result.get("dev_model_forwards")==0
             and result.get("cal_locked_opened") is False
             and result.get("product_promotion_allowed") is False,
             "REAL_B6_MATCHED_GAIN_KL_384_TRAIN_UPDATES")
        raw=z.read(B6_MODEL["member"])
        need(digest(raw)==B6_MODEL["pt_sha256"],
             "EXACT_B6_CONSISTENCY_SAVED_PT_SHA")
    authentic=torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
    replayed=torch.load(control_pt,weights_only=True,map_location="cpu")
    need(set(authentic)==set(replayed)
         and all(torch.equal(authentic[k],replayed[k]) for k in authentic),
         "B6_GAIN_KL_CONTROL_TENSOR_REPLAY_DRIFT_STOP")
    return {"status":"PASS_EXACT_B6_GAIN_KL_TENSOR_REPLAY",
            "original_B6_pt_sha256":B6_MODEL["pt_sha256"],
            "control_pt_sha256":digest(Path(control_pt).read_bytes()),
            "all_tensors_exactly_equal":True}

def empty_target_ctc_weight_delta(logp,prefix,tier,torch,F):
    """Only additive change: 0.25 of the *existing* negative CTC blank loss."""
    start=prefix//160-3
    need(0<start<logp.shape[1]-5,"EXACT_EXISTING_POST_PREFIX_CTC_FRAMES")
    tail=logp[:,start:,:]
    need(tail.shape[0]==1 and tail.shape[2]==4,
         "ONLY_FOUR_TOKEN_B3_ORIGINAL_CTC")
    target=torch.tensor([],dtype=torch.long)
    old_ctc=F.ctc_loss(
        tail.transpose(0,1),target,
        torch.tensor([tail.shape[1]],dtype=torch.long),
        torch.tensor([0],dtype=torch.long),
        blank=0,reduction="none",zero_infinity=False)[0]
    need(bool(torch.isfinite(old_ctc)),"FINITE_TRAIN_ONLY_ORDINARY_CTC")
    return (NEW_ORDINARY_WEIGHT-OLD_ORDINARY_WEIGHT)*old_ctc * (
             1.0 if tier=="silver" else 0.75)

def fit_challenger(original,rows,groups,cache,mel,mean,scale,out,torch,np,F):
    torch.manual_seed(SEED)
    net=b6.b4.v5.build_model(torch)
    net.load_state_dict(original,strict=True)
    net.train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=b6.LR,weight_decay=0.001)
    trace=[]
    starts=time.perf_counter()
    sorted_groups={k:sorted(groups[k]) for k in (0,1)}
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        prefix=PREFIXES[(step//35+step%3)%len(PREFIXES)]
        losses=[]
        consistency=[]
        for klass in (0,1):
            group_id=sorted_groups[klass][step%TRAIN_GROUPS]
            candidates=sorted(groups[klass][group_id],key=lambda r:r["id"])
            row=candidates[(step//TRAIN_GROUPS)%len(candidates)]
            pair={}
            for gain in GAINS:
                x=cache[row["id"]][gain][prefix]
                base_loss,logp=b6.suffix_loss(
                    net,x,klass,prefix,row["tier"],torch,F,True)
                if klass==0:
                    base_loss=base_loss+empty_target_ctc_weight_delta(
                        logp,prefix,row["tier"],torch,F)
                losses.append(base_loss)
                pair[gain]=logp
            clean,half=pair[1.0],pair[0.5]
            start=prefix//160-3
            need(clean.shape==half.shape and 0<start<half.shape[1]-5,
                 "EXACT_B6_GAIN_PAIR_FRAME_GRID")
            kl=F.kl_div(
                 half[:,start:,:],clean[:,start:,:].detach(),
                 reduction="none",log_target=True).sum(-1).mean()
            need(bool(torch.isfinite(kl)),"UNCHANGED_FINITE_B6_GAIN_KL")
            consistency.append(kl.clamp(min=0.0,max=KL_CLIP))
        need(len(losses)==4 and len(consistency)==2,
             "SAME_TWO_TRAIN_CLASSES_TWO_GAINS")
        loss=sum(losses)/len(losses)+KL_WEIGHT*sum(consistency)/len(consistency)
        need(bool(torch.isfinite(loss)),"FINITE_B7_ONE_FACTOR_TRAIN_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(),5.0,error_if_nonfinite=True)
        optimizer.step()
        if step==0 or (step+1)%16==0:
            trace.append({"step":step+1,"loss":float(loss.detach()),
                          "prefix_samples":prefix,"gains":list(GAINS),
                          "ordinary_ctc_weight":NEW_ORDINARY_WEIGHT,
                          "positive_ctc_weight":1.0,
                          "gain_kl_weight":KL_WEIGHT})
    net.eval()
    dest=out/"models"
    dest.mkdir(exist_ok=True)
    pt=dest/(STEM+".pt")
    torch.save(net.state_dict(),pt)
    saved=torch.load(pt,weights_only=True,map_location="cpu")
    checker=b6.b4.v5.build_model(torch)
    checker.load_state_dict(saved,strict=True)
    checker.eval()
    need(sum(p.numel() for p in checker.parameters())==PARAMETERS,
         "SAME_FROZEN_43876_PARAMETER_FSMN_CTC")
    npz=dest/(STEM+".npz")
    np.savez(npz,mel_filters=mel,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy()
                for k,v in checker.state_dict().items()})
    example=cache[sorted(rows,key=lambda r:r["id"])[0]["id"]][1.0][6400]
    onnx=dest/(STEM+".onnx")
    torch.onnx.export(
        checker,example,onnx.as_posix(),opset_version=17,dynamo=False,
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(all(p.is_file() and p.stat().st_size>10000 for p in (pt,npz,onnx)),
         "REAL_PT_NPZ_ONNX_B7_CHECKPOINT")
    return {"arm":"ordinary_ctc085",
            "optimizer_updates":STEPS,
            "train_source_groups":TRAIN_GROUPS,
            "changed_loss":"ORDINARY_CTC_WEIGHT_ONLY",
            "ordinary_ctc_weight":NEW_ORDINARY_WEIGHT,
            "gain_kl_weight":KL_WEIGHT,
            "trace":trace,
            "elapsed_optimizer_seconds":round(time.perf_counter()-starts,3),
            "files":{p.name:{"sha256":digest(p.read_bytes()),"bytes":p.stat().st_size}
                     for p in (pt,npz,onnx)}}

def execute(a):
    provenance_receipt=guard()
    out=a.output
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "ONE_NEW_B7_TRAIN_OUTPUT_ONLY")
    out.mkdir()
    save(out/"STARTED.json",{
        "schema":SCHEMA,"run":provenance_receipt,
        "maximum_new_optimizer_updates":STEPS*2,
        "new_tts_asr_phone_calls":0,"dev_model_forwards":0,
        "cal_locked_opened":False,"model_selection_allowed":False})
    try:
        train,dev,receipt,groups=b6.source_audit(a)
        need(len(groups[0])==len(groups[1])==TRAIN_GROUPS
             and all(x["role"]=="train" for x in train)
             and all(x["role"]=="development" for x in dev),
             "ONLY_35_ALREADY_AUDITED_TRAIN_GROUPS_FOR_GRADIENTS")
        save(out/"SOURCE.json",{
            "train_pcm_sha256":sorted(r["source_pcm_sha256"] for r in train),
            "source_evidence":receipt,
            "dev_rows_metadata_only":len(dev),"dev_pcm_in_gradients":False})
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        parent,mel,mean,scale=b6.b4.authenticate_b3(a.b3_model_zip,torch,np)
        cache=b6.b4.cache_train(
            train,a.old_archives,a.fresh_source,mel,mean,scale,torch,np)
        control=b6.fit("consistency",parent,train,groups,cache,
                       mel,mean,scale,out,torch,np,F)
        replay=verify_B6_replay(
            a.b6_model_zip,
            out/"models"/(b6.ARMS["consistency"]+".pt"),torch)
        challenger=fit_challenger(
            parent,train,groups,cache,mel,mean,scale,out,torch,np,F)
        need(control["optimizer_updates"]==challenger["optimizer_updates"]==STEPS
             and control["gain_consistency_enabled"] is True
             and challenger["ordinary_ctc_weight"]==NEW_ORDINARY_WEIGHT
             and replay["status"]=="PASS_EXACT_B6_GAIN_KL_TENSOR_REPLAY",
             "MATCHED_B6_AND_B7_ACTUAL_192_STEP_TRAJECTORIES")
        result={
           "schema":SCHEMA,
           "status":"RESEARCH_B7_REAL_ORDINARY_CTC_WEIGHT_ONLY_TRAIN_NO_PROMOTION",
           "original_B3_parent":b6.b4.B3["id"],
           "actual_original_B6_parent":B6_MODEL["id"],
           "exact_B6_tensor_replay":replay,
           "only_changed_train_factor":"ORDINARY_CTC_060_TO_085",
           "ordinary_old_weight":OLD_ORDINARY_WEIGHT,
           "ordinary_new_weight":NEW_ORDINARY_WEIGHT,
           "positive_ctc_weight_unchanged":1.0,
           "blank_prefix_loss_weight_unchanged":0.5,
           "gain_KL_weight_unchanged":KL_WEIGHT,
           "train_source_groups":TRAIN_GROUPS,
           "new_optimizer_updates":STEPS*2,
           "per_arm_updates":STEPS,
           "model_parameters":PARAMETERS,
           "control":control,"challenger":challenger,
           "new_tts_asr_phone_calls":0,"dev_model_forwards":0,
           "dev_pcm_in_gradients":False,"cal_locked_opened":False,
           "model_selection_allowed":False,"winner":None,
           "product_promotion_allowed":False,
           "physical_ssc305":"NOT_RUN",
           "human_3_5m_frr":"NOT_MEASURED",
           "natural_far_per_hour":"NOT_MEASURED",
           "fresh_independent_qualified_development":"NOT_RUN"}
        save(out/"RESULT.json",result)
        print(json.dumps({
            "status":result["status"],
            "new_optimizer_updates":result["new_optimizer_updates"],
            "baseline_replay":replay["status"],
            "B7_candidate_pt_sha256":challenger["files"][STEM+".pt"]["sha256"],
            "product_promotion_allowed":False},sort_keys=True))
    except BaseException as e:
        save(out/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_B7_NO_VALID_NEW_WEIGHTS",
             "error_type":type(e).__name__,"reason":str(e)[:450],
             "product_promotion_allowed":False,
             "dev_pcm_in_gradients":False,"cal_locked_opened":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    flags=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--provenance","--guard","--audit","--execute"):
        flags.add_argument(flag,action="store_true")
    for name in ("old-archives","old-audit","fresh-source","fresh-screen",
                 "source-gate","prospective-source","prospective-screen",
                 "original-model-zip","b3-model-zip","b6-model-zip","output"):
        ap.add_argument("--"+name,type=Path)
    a=ap.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.provenance:
        print(json.dumps(provenance(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    required=("old_archives","old_audit","fresh_source","fresh_screen",
              "source_gate","prospective_source","prospective_screen",
              "original_model_zip")
    need(all(getattr(a,k) is not None for k in required),
         "ONLY_BYTE_AUTHENTICATED_HISTORICAL_TRAIN_SOURCES")
    if a.audit:
        train,dev,source,groups=b6.source_audit(a)
        print(json.dumps({"status":"AUDITED_35_TRAIN_GROUPS_NO_OPTIMIZER",
             "train_rows":len(train),"train_groups":len(groups[0]),
             "dev_metadata_rows":len(dev),"new_optimizer_updates":0},sort_keys=True))
        return
    need(a.b3_model_zip is not None and a.b6_model_zip is not None
         and a.output is not None,"EXACT_FROZEN_B3_AND_B6_MODEL_INPUT")
    execute(a)

if __name__=="__main__":
    main()
