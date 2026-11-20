#!/usr/bin/env python3
"""B6 Hey Mero TRAIN-only clean/half CTC posterior consistency.

Both matched 192-step trajectories start from identical frozen B3 and reuse
only the 35 byte-authenticated TRAIN groups. Original B4-half tensor replay
is mandatory. The challenger only adds stop-gradient clean-to-half-gain CTC
posterior KL over matched speech suffix frames: no guessed HEY timestamps,
new TTS, CAL/LOCKED, DEV gradients or product promotion.
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
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location(
    "b6_exact_frozen_b4_training",HERE/"english_v6_b4_paired05_trainonly_once.py")
b4=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b4)
REPO="jiying2007/kws-data"
BEFORE="a682594431fd6d181edb7e5b6922755aa3917bc4"
SCHEMA="english-v6-b6-matched-trainonly-gain-posterior-consistency-v1"
STEPS=192
SOURCE_GROUPS=35
PARAMS=43876
SEED=20261120
LR=0.00006
PREFIXES=(6400,12800,19200)
GAINS=(1.0,0.5)
WEIGHT=0.03
KL_CLIP=8.0
ARMS={"control":"mero43-b6-replay-control192",
      "consistency":"mero43-b6-gain-consistency192"}
B4_TRAIN={"run":44665669003,"head":"bcdb6c94d262d40148580c1886ac3856c53c18d5",
    "id":14597862669,"name":"english-v6-b4-matched-clean-vs-half-44665669003",
    "bytes":1000659,
    "zip_sha256":"6eba8013da5ef06aa5046eab3f8d48b912af5f80941d98f94d0affae23764cb2",
    "member":"models/mero43-b4-paired05-192.pt",
    "pt_sha256":"c795f8b5f6bf8e9c37d195126fc25006dbde92fb71591473c05854e3d3698f15"}
EXPOSED_DEV={"run":44685455242,"head":"1cf55598f76aa739da93d5b39578f118f73f96ea",
    "id":14606730739,"name":"english-v6-b4-posttrain-three-frozen-paired-44685455242",
    "bytes":54887,
    "zip_sha256":"d57464ba599504b777fa3b58f53823985ee4241cb4f0b3b76384109314360e6e"}

B5_QUALITY={"run":44700786677,"head":"1668301a2116cabb0e78ca247ae1102b1ab25d98",
    "id":14611963808,"name":"english-v6-b5-dev504-three-frozen-paired-44700786677",
    "bytes":39379,
    "zip_sha256":"67267fcdf78535a75854645de60e7ad662aaa5016788b62e9901c1f073e9b016"}
B5_FORENSIC={"run":44702527315,"head":"a682594431fd6d181edb7e5b6922755aa3917bc4",
    "id":14613556816,"name":"english-v6-b5-saved-gain-pathology-44702527315",
    "bytes":1444,
    "zip_sha256":"628ea7cae4f46911a809d7cfacb38e38a0e9b764e5f1a1925932f7451281a578"}

def need(ok,why):
    if not ok: raise ValueError(why)

def digest(data): return hashlib.sha256(data).hexdigest()

def save(path,obj):
    with Path(path).open("x",encoding="utf8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def source_check():
    x=b4.source_check()
    need(x["status"]=="SOURCE_ONLY_NO_OPTIMIZER_OR_AUDIO_FORWARD"
         and x["train_source_groups"]==SOURCE_GROUPS
         and x["frozen_parent_artifact"]==14590093798
         and b4.STEPS==STEPS and b4.SEED==SEED and b4.LR==LR
         and b4.PREFIX_SAMPLES==PREFIXES
         and b4.GAINS["challenger_clean_half"]==GAINS
         and len(ARMS)==2 and list(ARMS)==["control","consistency"]
         and PARAMS==43876 and WEIGHT==0.03 and KL_CLIP==8.0
         and B4_TRAIN["pt_sha256"].startswith("c795f8b5")
         and EXPOSED_DEV["head"]=="1cf55598f76aa739da93d5b39578f118f73f96ea"
         and B5_QUALITY["id"]==14611963808
         and B5_FORENSIC["id"]==14613556816
         and B5_FORENSIC["head"]==BEFORE,
         "EXACT_B4_CONTROL_AND_TRAIN_GAIN_KL_ONLY")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_TRAIN_OR_MODEL_FORWARD",
       "target":"Hey Mero","same_43876_param_b3_parent":True,
       "frozen_b3_artifact":b4.B3["id"],
       "b4_half_replay_original_artifact":B4_TRAIN["id"],
       "b4_half_replay_original_pt_sha256":B4_TRAIN["pt_sha256"],
       "exposed_dev_receipt_read_only":EXPOSED_DEV["id"],
       "B5_117_forward_readonly":B5_QUALITY["id"],
       "B5_saved_pathology_readonly":B5_FORENSIC["id"],
       "train_source_groups":SOURCE_GROUPS,
       "same_gains_both_arms":list(GAINS),"same_seed_and_optimizer":True,
       "only_changed_factor":"CLEAN_TEACHER_HALF_GAIN_SUFFIX_POSTERIOR_KL",
       "gain_consistency_weight":WEIGHT,"kl_clip":KL_CLIP,
       "teacher_clean_stop_gradient":True,
       "per_arm_optimizer_updates":STEPS,
       "maximum_new_optimizer_updates":2*STEPS,
       "baseline_replay_required":True,"new_tts_asr":0,
       "development_model_forwards":0,"cal_locked_opened":False,
       "new_schedule":False,"work_codex":False,"private_repo":False,
       "winner":None,"model_selection_allowed":False,
       "product_promotion_allowed":False}

def provenance():
    import urllib.request
    token=os.environ.get("GH_TOKEN","")
    need(len(token)>10,"GITHUB_READ_ONLY_TOKEN_REQUIRED")
    x=b4.verify_provenance()
    base="https://api.github.com/repos/"+REPO+"/actions/"
    def get(part):
        req=urllib.request.Request(base+part,headers={
            "Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "User-Agent":"kws-v6-b6-matched-gain-kl-train"})
        with urllib.request.urlopen(req,timeout=35) as response:
            raw=response.read(1048577)
        need(len(raw)<=1048576,"BOUND_REMOTE_PROVENANCE")
        return json.loads(raw)
    for item in (B4_TRAIN,EXPOSED_DEV,B5_QUALITY,B5_FORENSIC):
        run=get("runs/"+str(item["run"]))
        need(run.get("id")==item["run"]
             and run.get("head_sha")==item["head"]
             and run.get("head_branch")=="main"
             and run.get("run_attempt")==1
             and run.get("status")=="completed"
             and run.get("conclusion")=="success"
             and run.get("event")=="push",
             "EXACT_REAL_B4_TRAIN_AND_PRIOR_DEV_RUN")
        artifact=get("artifacts/"+str(item["id"]))
        need(artifact.get("id")==item["id"]
             and artifact.get("name")==item["name"]
             and artifact.get("size_in_bytes")==item["bytes"]
             and artifact.get("digest")=="sha256:"+item["zip_sha256"]
             and artifact.get("expired") is False
             and artifact.get("workflow_run",{}).get("id")==item["run"]
             and artifact.get("workflow_run",{}).get("head_sha")==item["head"],
             "EXACT_BYTE_BOUND_PRIOR_MODEL_OR_DEV")
    return {"prior_b3":x,"b4_train_id":B4_TRAIN["id"],
            "prior_dev_id":EXPOSED_DEV["id"],
            "prior_B5_quality":B5_QUALITY["id"],
            "prior_B5_saved_pathology":B5_FORENSIC["id"]}

def guard():
    source_check()
    e=os.environ
    need(all(e.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),
        "FIRST_PROTECTED_PUBLIC_MAIN_ONLY")
    sha=e.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha) is not None
         and re.fullmatch("[1-9][0-9]*",e.get("GITHUB_RUN_ID","")) is not None,
         "REAL_HEAD_AND_RUN_ID")
    event=json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==sha
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False
         and subprocess.check_output(["git","rev-parse","HEAD^1"],
                             cwd=ROOT,text=True).strip()==BEFORE,
         "NO_OUT_OF_ORDER_OR_REPLAY")
    return {"run_id":int(e["GITHUB_RUN_ID"]),"head_sha":sha,
            "evidence":provenance()}

def source_audit(a):
    rows,dev,receipt=b4.robust.audit(
        a.old_archives,a.old_audit,a.fresh_source,a.fresh_screen,
        a.source_gate,a.prospective_source,a.prospective_screen,
        a.original_model_zip)
    group=b4.input_groups(rows)
    need(len(group[0])==len(group[1])==SOURCE_GROUPS
         and all(r["role"]=="train" for r in rows)
         and all(r["role"]=="development" for r in dev),
         "EXACT_35_AUTHENTIC_TRAIN_GROUPS_NO_DEV_GRADIENT")
    return rows,dev,receipt,group

def hey_loss(model,x,klass,prefix,tier,torch,F,enabled):
    # B4's CTC suffix and silent-prefix loss remain byte-for-byte formulas.
    logit=model(x).log_softmax(-1)
    start=prefix//160-3
    blank=prefix//160-8
    need(0<blank<start<logit.shape[1]-5,
         "IMMUTABLE_B3_SUFFIX_CTC_START")
    tail=logit[:,start:,:]
    labels=torch.tensor([1,2,3] if klass else [],dtype=torch.long)
    ctc=F.ctc_loss(tail.transpose(0,1),labels,
                   torch.tensor([tail.shape[1]],dtype=torch.long),
                   torch.tensor([len(labels)],dtype=torch.long),
                   blank=0,reduction="none",zero_infinity=False)[0]
    silent=logit[:,:blank,:]
    blank_margin=torch.logsumexp(silent[...,1:],dim=-1)-silent[...,0]
    loss=ctc*(1.0 if klass else 0.6)+0.5*F.softplus(blank_margin).mean()
    if enabled:
        # No fabricated "Hey" timestamps; only a weak window-level
        # class label from immutable TRAIN (the exposed DEV has no gradients).
        n=min(EARLY_MAX,(tail.shape[1]+1)//2)
        need(n>=3,"TRAIN_ONLY_EARLY_SUFFIX_EXISTS")
        early=tail[:,:n,:]
        score=(torch.logsumexp(early[...,1],dim=1)
               -torch.logsumexp(early[...,0],dim=1))
        aux=(F.softplus(MARGIN-score) if klass
             else F.softplus(MARGIN+score)).mean()
        loss=loss+WEIGHT*aux
    return loss*(1.0 if tier=="silver" else 0.75)

def fit(label,original,rows,groups,cache,mel,mean,scale,directory,torch,np,F):
    need(label in ARMS,"ONLY_MATCHED_TWO_ARMS")
    enabled=label=="early_hey"
    torch.manual_seed(SEED)
    net=b4.v5.build_model(torch)
    net.load_state_dict(original,strict=True)
    net.train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.001)
    starts=time.perf_counter()
    trace=[]
    sorted_groups={k:sorted(groups[k]) for k in (0,1)}
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        prefix=PREFIXES[(step//35+step%3)%len(PREFIXES)]
        losses=[]
        for klass in (0,1):
            group_id=sorted_groups[klass][step%SOURCE_GROUPS]
            candidates=sorted(groups[klass][group_id],key=lambda r:r["id"])
            row=candidates[(step//SOURCE_GROUPS)%len(candidates)]
            for gain in GAINS:
                losses.append(hey_loss(
                    net,cache[row["id"]][gain][prefix],klass,prefix,
                    row["tier"],torch,F,enabled))
        loss=sum(losses)/len(losses)
        need(bool(torch.isfinite(loss)),"FINITE_MATCHED_CTC_AUX_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(),5.0,error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%16==0 or step==0:
            trace.append({"step":step+1,"loss":float(loss.detach()),
                          "prefix_samples":prefix,"gain_policy":list(GAINS),
                          "hey_aux":enabled})
    net.eval()
    dest=directory/"models"
    dest.mkdir(exist_ok=True)
    stem=ARMS[label]
    pt=dest/(stem+".pt")
    torch.save(net.state_dict(),pt)
    fresh=torch.load(pt,weights_only=True,map_location="cpu")
    check=b4.v5.build_model(torch)
    check.load_state_dict(fresh,strict=True)
    check.eval()
    need(sum(p.numel() for p in check.parameters())==PARAMS,
         "UNCHANGED_43876_PARAM_WEIGHTS")
    npz=dest/(stem+".npz")
    np.savez(npz,mel_filters=mel,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy()
                for k,v in check.state_dict().items()})
    example=cache[sorted(rows,key=lambda r:r["id"])[0]["id"]][1.0][6400]
    onnx=dest/(stem+".onnx")
    torch.onnx.export(check,example,onnx.as_posix(),
        opset_version=17,dynamo=False,
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(all(p.is_file() and p.stat().st_size>10000 for p in (pt,npz,onnx)),
         "REAL_PT_NPZ_ONNX_SAVED")
    return {"arm":label,"optimizer_updates":STEPS,"train_source_groups":SOURCE_GROUPS,
            "same_half_gain_policy":list(GAINS),"hey_aux_enabled":enabled,
            "trace":trace,"elapsed_optimizer_seconds":round(time.perf_counter()-starts,3),
            "files":{p.name:{"sha256":digest(p.read_bytes()),"bytes":p.stat().st_size}
                     for p in (pt,npz,onnx)}}

def verify_replay(archive_path,control_pt,torch):
    need(archive_path.is_file() and not archive_path.is_symlink()
         and archive_path.stat().st_size==B4_TRAIN["bytes"]
         and digest(archive_path.read_bytes())==B4_TRAIN["zip_sha256"],
         "BYTE_AUTHENTIC_B4_HALF_ORIGINAL")
    with zipfile.ZipFile(archive_path) as zipf:
        receipt=json.loads(zipf.read("RESULT.json"))
        need(receipt.get("status")=="RESEARCH_B4_MATCHED_TRAINONLY_REAL_WEIGHTS_NO_PROMOTION"
             and receipt.get("actual_total_optimizer_steps")==384
             and receipt.get("source_train_groups")==SOURCE_GROUPS
             and receipt.get("model_parameters")==PARAMS
             and receipt.get("cal_locked_opened") is False
             and receipt.get("product_promotion_allowed") is False,
             "B4_REAL_HALF_TRAIN_NONPROMOTION_RECEIPT")
        raw=zipf.read(B4_TRAIN["member"])
        need(digest(raw)==B4_TRAIN["pt_sha256"],"ORIGINAL_B4_HALF_PT_SHA")
    original=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    candidate=torch.load(control_pt,map_location="cpu",weights_only=True)
    need(set(original)==set(candidate)
         and all(torch.equal(original[k],candidate[k]) for k in original),
         "B4_HALF_REPLAY_TENSOR_DRIFT_STOP")
    return {"status":"PASS_EXACT_B4_HALF_TENSOR_REPLAY",
            "prior_b4_half_pt_sha256":B4_TRAIN["pt_sha256"],
            "new_control_pt_sha256":digest(control_pt.read_bytes()),
            "all_checkpoint_tensors_equal":True}

def execute(a):
    identity=guard()
    out=a.output
    need(out.is_absolute() and out.parent.is_dir()
         and not out.exists(),"ONE_NEW_B5_TRAIN_OUTPUT_ONLY")
    out.mkdir()
    save(out/"STARTED.json",{"schema":SCHEMA,"identity":identity,
         "max_optimizer_updates":STEPS*2,"new_tts_asr_calls":0,
         "cal_locked_access":False,"dev_forwards":0})
    try:
        rows,dev,receipt,groups=source_audit(a)
        save(out/"SOURCE.json",{"train_shas":sorted(r["source_pcm_sha256"] for r in rows),
             "dev_metadata_only":len(dev),"dev_pcm_in_optimizer":False,
             "source_evidence":receipt})
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        original,mel,mean,scale=b4.authenticate_b3(a.b3_model_zip,torch,np)
        cache=b4.cache_train(rows,a.old_archives,a.fresh_source,
                             mel,mean,scale,torch,np)
        control=fit("control",original,rows,groups,cache,mel,mean,scale,
                    out,torch,np,F)
        replay=verify_replay(a.b4_model_zip,
             out/"models"/(ARMS["control"]+".pt"),torch)
        challenge=fit("early_hey",original,rows,groups,cache,mel,mean,scale,
                      out,torch,np,F)
        need(control["optimizer_updates"]==challenge["optimizer_updates"]==STEPS
             and not control["hey_aux_enabled"] and challenge["hey_aux_enabled"],
             "SINGLE_AUX_CHANGE_MATCHED_TWO_TRAJECTORIES")
        result={"schema":SCHEMA,
          "status":"RESEARCH_B5_MATCHED_TRAIN_ONLY_HEY_AUX_MODEL_NO_PROMOTION",
          "B3_parent_artifact":b4.B3["id"],
          "B3_parent_pt_sha256":b4.B3["pt"],
          "B4_half_baseline_replay":replay,
          "only_changed_train_factor":"EARLY_SUFFIX_HEY_BLANK_AUXILIARY",
          "auxiliary_weight":WEIGHT,"auxiliary_margin":MARGIN,
          "max_early_suffix_frames":EARLY_MAX,
          "train_source_groups":SOURCE_GROUPS,
          "per_arm_steps":STEPS,"actual_optimizer_updates":STEPS*2,
          "model_parameters":PARAMS,
          "arms":[control,challenge],
          "new_tts_asr_calls":0,"dev_model_forwards":0,
          "cal_locked_opened":False,
          "independent_dev_quality":"NOT_RUN",
          "physical_ssc305":"NOT_RUN",
          "natural_far_hour":"NOT_MEASURED",
          "new_c11_arm_parity":"NOT_RUN",
          "model_selection_allowed":False,"product_promotion_allowed":False}
        save(out/"RESULT.json",result)
        print(json.dumps({"status":result["status"],
             "actual_optimizer_updates":result["actual_optimizer_updates"],
             "baseline_replay":replay["status"],
             "pt":{a["arm"]:a["files"][ARMS[a["arm"]]+".pt"]["sha256"]
                   for a in result["arms"]},
             "product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        save(out/"STOP.json",{"schema":SCHEMA,"status":"STOP_B5_NO_REPLAY",
             "error_type":type(exc).__name__,"reason":str(exc)[:450],
             "model_selection_allowed":False,
             "product_promotion_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    modes=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--provenance","--guard","--audit","--execute"):
        modes.add_argument(flag,action="store_true")
    for key in ("old-archives","old-audit","fresh-source","fresh-screen",
                "source-gate","prospective-source","prospective-screen",
                "original-model-zip","b3-model-zip","b4-model-zip","output"):
        ap.add_argument("--"+key,type=Path)
    a=ap.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.provenance:
        print(json.dumps(provenance(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    req=("old_archives","old_audit","fresh_source","fresh_screen",
         "source_gate","prospective_source","prospective_screen",
         "original_model_zip")
    need(all(getattr(a,k) is not None for k in req),
         "EIGHT_BYTE_AUTHENTIC_TRAIN_SOURCE_ASSETS_REQUIRED")
    if a.audit:
        t,d,receipt,group=source_audit(a)
        print(json.dumps({"status":"SOURCE_AUDITED_NO_TRAIN",
             "train_rows":len(t),"train_groups":len(group[1]),
             "dev_rows_metadata_only":len(d),
             "dev_model_forwards":0},sort_keys=True));return
    need(a.b3_model_zip is not None and a.b4_model_zip is not None
         and a.output is not None,"ORIGINAL_B3_AND_B4_MODELS_REQUIRED")
    execute(a)

if __name__=="__main__":main()
