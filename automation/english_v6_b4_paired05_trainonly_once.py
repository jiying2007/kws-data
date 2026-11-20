#!/usr/bin/env python3
"""B4: source-audited matched TRAIN-only gain ablation from immutable B3 FSMN-CTC.

Two same-seed, same-source, same-optimizer, same 192-step 43,876-param CTC
trajectories start at exact B3 PT/NPZ. Control receives clean/clean;
challenger receives clean/half-amplitude PCM16. No other training loss,
architecture, tokenizer, normalization or prefix policy changes. No DEV
acoustic forwards, CAL, LOCKED, new TTS/ASR, thresholds or product promotion.
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
import struct
import subprocess
import time
import zipfile
from collections import defaultdict

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def imported(filename,tag):
    spec=importlib.util.spec_from_file_location(tag,HERE/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

prior=imported("english_v6_mero43_trainonly_suffix_ctc256_once.py",
               "b4_prior_exact_suffix")
b3audit=imported("english_v6_b3_c11_arm_parity_once.py",
                 "b4_exact_B3_model_identity")
v5=prior.v5
robust=prior.robust
REPO="jiying2007/kws-data"
BEFORE="5443f4e6b3526bdd0aaef6b00a28594b044dbf50"
SCHEMA="english-v6-b4-matched-b3-trainonly-05gain-ablation-v1"
B3=b3audit.B3
PREVIOUS_CTC_RUN=44664029084
PREVIOUS_CTC_ARTIFACT=14596776938
PREVIOUS_CTC_SHA="d725bddbf73b2b6f3d1a89bb46f1ad1f6aa63634af84b762587460fa6bd00cee"
TRAIN_GROUPS=35
PARAMS=43876
STEPS=192
SEED=20261120
LR=0.00006
PREFIX_SAMPLES=(6400,12800,19200)
GAINS={"control_clean_clean":(1.0,1.0),
       "challenger_clean_half":(1.0,0.5)}
ARTIFACT_STEMS={
 "control_clean_clean":"mero43-b4-clean192",
 "challenger_clean_half":"mero43-b4-paired05-192"
}

def need(valid,reason):
    if not valid:raise ValueError(reason)

def sha(data):return hashlib.sha256(data).hexdigest()

def write(path,obj):
    with Path(path).open("x",encoding="utf-8") as out:
        json.dump(obj,out,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        out.write("\n")

def source_check():
    r=prior.source_check()
    b=b3audit.source_check()
    need(r["status"]=="SOURCE_ONLY_NO_NEW_OPTIMIZER"
         and r["train_groups"]==TRAIN_GROUPS
         and b["model_artifact"]==B3["id"]
         and b["status"]=="SOURCE_ONLY_NO_ACOUSTIC_FORWARDS"
         and b["model_parameters"]==PARAMS
         and B3["id"]==14590093798
         and B3["run"]==44651053897
         and B3["sha"]==
           "59373f0e7e5f9d4ed35d6b116cdd6f8ac8eb5abcfa1acb5311ef658c59a20e04"
         and B3["pt"]==
           "682e2689e82aa91d850516c06b2be61efc64731cb1dee3b70a7d2a8e01925a46"
         and B3["stem"]=="mero43-onset256"
         and PREVIOUS_CTC_RUN==44664029084
         and PREVIOUS_CTC_ARTIFACT==14596776938
         and STEPS==192 and SEED==20261120 and LR==0.00006
         and PREFIX_SAMPLES==(6400,12800,19200)
         and GAINS=={"control_clean_clean":(1.0,1.0),
                     "challenger_clean_half":(1.0,0.5)}
         and len(GAINS)==2 and len(ARTIFACT_STEMS)==2,
         "IMMUTABLE_B3_AND_ONLY_GAIN_FACTOR_ON_TRAIN")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_OPTIMIZER_OR_AUDIO_FORWARD",
            "target":"Hey Mero","model_parameters":PARAMS,
            "train_source_groups":TRAIN_GROUPS,"frozen_parent_artifact":B3["id"],
            "frozen_parent_pt_sha256":B3["pt"],
            "prior_ctc_origin_run":PREVIOUS_CTC_RUN,
            "experiment":"MATCHED_CONTROL_VS_PAIRED_HALF_GAIN",
            "only_changed_factor":"TRAIN_PCM16_GAIN_ON_SECOND_PAIRED_PASS",
            "same_b3_initialization":True,"same_optimizer_and_CTC":True,
            "two_trajectories":list(GAINS),
            "per_trajectory_updates":STEPS,
            "max_total_new_optimizer_updates":STEPS*2,
            "seed":SEED,"learning_rate":LR,
            "prefix_samples":list(PREFIX_SAMPLES),
            "dev_model_forwards":0,"new_tts_asr_calls":0,
            "cal_locked_access":False,"new_private_repo":False,
            "work_or_codex":False,"new_schedule":False,
            "model_selection_allowed":False,
            "product_promotion":False}

def verify_provenance():
    import urllib.request
    token=os.environ.get("GH_TOKEN")
    need(isinstance(token,str) and len(token)>10,
         "READONLY_GITHUB_TOKEN_REQUIRED")
    previous=b3audit.provenance(token)
    base="https://api.github.com/repos/"+REPO+"/actions/"
    def get(suffix):
        request=urllib.request.Request(base+suffix,headers={
            "Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "User-Agent":"en-v6-b4-gain-matched-training"})
        with urllib.request.urlopen(request,timeout=35) as f:
            body=f.read(1048577)
        need(len(body)<=1048576,"BOUNDED_REMOTE_METADATA")
        return json.loads(body)
    run=get("runs/"+str(PREVIOUS_CTC_RUN))
    need(run.get("id")==PREVIOUS_CTC_RUN
         and run.get("head_sha")==BEFORE
         and run.get("run_attempt")==1 and run.get("event")=="push"
         and run.get("status")=="completed" and run.get("conclusion")=="success",
         "PRIOR_B3_CTC_SOURCE_TRACE_ACTUAL_SUCCESS")
    a=get("artifacts/"+str(PREVIOUS_CTC_ARTIFACT))
    need(a.get("id")==PREVIOUS_CTC_ARTIFACT
         and a.get("name")==
            "english-v6-b3-ctc-gap-origin-44664029084"
         and a.get("digest")=="sha256:"+PREVIOUS_CTC_SHA
         and a.get("size_in_bytes")==2938 and a.get("expired") is False
         and a.get("workflow_run",{}).get("id")==PREVIOUS_CTC_RUN,
         "PRIOR_B3_TOKEN_ORIGIN_REPORT_BYTE_PIN")
    return {"parent_b3":previous,"ctc_source_run":PREVIOUS_CTC_RUN,
            "ctc_source_artifact":PREVIOUS_CTC_ARTIFACT,
            "ctc_source_zip_sha256":PREVIOUS_CTC_SHA}

def guard():
    source_check()
    env=os.environ
    required={"GITHUB_ACTIONS":"true",
              "GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push",
              "GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),
         "ONE_PROTECTED_PUBLIC_MAIN_PUSH_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[a-f0-9]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "VALID_RUN_AND_SHA")
    ev=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(ev.get("before")==BEFORE and ev.get("after")==head
         and ev.get("ref")=="refs/heads/main"
         and ev.get("repository",{}).get("full_name")==REPO
         and ev.get("repository",{}).get("private") is False
         and ev.get("forced") is False and ev.get("deleted") is False
         and subprocess.check_output(["git","rev-parse","HEAD^1"],
                                     cwd=ROOT,text=True).strip()==BEFORE,
         "NO_MOVED_PARENT_OR_REPLAY")
    return {"run":int(env["GITHUB_RUN_ID"]),
            "main_sha":head,
            "previous_completed":verify_provenance()}

def authenticate_b3(archive_path,torch,np):
    need(archive_path.is_file() and not archive_path.is_symlink()
         and archive_path.stat().st_size==B3["bytes"]
         and sha(archive_path.read_bytes())==B3["sha"],
         "EXACT_ACTUALLY_TRAINED_B3_ZIP_SHA")
    with zipfile.ZipFile(archive_path) as arc:
        expected={"STARTED.json","SOURCE.json","RESULT.json"}
        expected|={"models/"+B3["stem"]+"."+ext
                   for ext in ("pt","npz","onnx")}
        need(set(arc.namelist())==expected,
             "NO_EXTRA_OR_MISSING_B3_MEMBERS")
        receipt=json.loads(arc.read("RESULT.json"))
        need(receipt["actual_optimizer_steps"]==256
             and receipt["source_train_groups"]==TRAIN_GROUPS
             and receipt["model_parameter_count"]==PARAMS
             and receipt["status"]==
              "RESEARCH_TRAIN_ONLY_SUFFIX_CTC_MODEL_NOT_PROMOTABLE"
             and receipt["cal_locked_access"] is False
             and receipt["product_promotion_allowed"] is False,
             "B3_AUTHENTIC_TRAINING_AND_NONPROMOTION")
        pt=arc.read("models/"+B3["stem"]+".pt")
        packed=arc.read("models/"+B3["stem"]+".npz")
        need(sha(pt)==B3["pt"] and sha(packed)==B3["npz"]
             and sha(arc.read("models/"+B3["stem"]+".onnx"))==B3["onnx"],
             "B3_FROZEN_PT_NPZ_ONNX_SHA")
    weights=torch.load(io.BytesIO(pt),map_location="cpu",weights_only=True)
    loaded=v5.build_model(torch)
    loaded.load_state_dict(weights,strict=True)
    loaded.eval()
    with np.load(io.BytesIO(packed),allow_pickle=False) as fields:
        need(all(np.array_equal(fields[k],v.detach().cpu().numpy())
                 for k,v in weights.items()),"FROZEN_B3_PT_NPZ_SAME_TENSORS")
        mel=fields["mel_filters"].copy()
        mean=fields["train_mean"].copy()
        scale=fields["train_scale"].copy()
    need(mel.shape==(257,40) and mean.shape==(40,)
         and scale.shape==(40,) and np.isfinite(mel).all()
         and np.isfinite(mean).all() and np.isfinite(scale).all()
         and (scale>0).all()
         and sum(v.numel() for v in weights.values())==PARAMS,
         "UNCHANGED_43876_PARAM_40MEL_MODEL")
    return weights,mel,mean,scale

def gain_pcm(pcm,gain,np):
    need(gain in (1.0,0.5)
         and isinstance(pcm,bytes) and len(pcm)%2==0,
         "ONLY_CLEAN_AND_HALF_FULL_PCM16")
    if gain==1.0:
        return pcm
    frame=np.frombuffer(pcm,dtype="<i2").astype(np.float32)
    derived=np.rint(frame*gain).clip(-32768,32767).astype("<i2").tobytes()
    need(len(derived)==len(pcm),"NO_AUDIO_DURATION_CHANGE")
    return derived

def input_groups(rows):
    group={0:defaultdict(list),1:defaultdict(list)}
    need(rows and all(r["role"]=="train" and
        r["mono_class"] in (0,1) and
        isinstance(r["source_pcm_sha256"],str) and
        len(r["source_pcm_sha256"])==64 for r in rows),
        "ONLY_AUTHENTICATED_TRAIN_PCM_ROWS")
    for row in rows:
        group[row["mono_class"]][row["source_group_id"]].append(row)
    need(len(group[0])==len(group[1])==TRAIN_GROUPS
         and set(group[0])==set(group[1]),
         "COMPLETE_BALANCED_35_SOURCE_GROUPS")
    return group

def loss_for(model,x,klass,prefix_samples,tier,torch,F):
    logit=model(x).log_softmax(-1)
    s=prefix_samples//160-3
    blank=prefix_samples//160-8
    need(0<blank<s<logit.shape[1]-5,
         "SAME_KNOWN_SUFFIX_CTC_BOUNDARY_AS_B3")
    cut=logit[:,s:,:]
    labels=torch.tensor([1,2,3] if klass else [],dtype=torch.long)
    ctc=F.ctc_loss(cut.transpose(0,1),labels,
                   torch.tensor([cut.shape[1]],dtype=torch.long),
                   torch.tensor([len(labels)],dtype=torch.long),
                   blank=0,reduction="none",zero_infinity=False)[0]
    x0=logit[:, :blank, :]
    # Same B3 0.5 * softplus(logsumexp(nonblank)-blank).
    blank_margin=torch.logsumexp(x0[...,1:],dim=-1)-x0[...,0]
    onset=F.softplus(blank_margin).mean()
    return (ctc*(1.0 if klass else 0.6)+0.5*onset)*(
        1.0 if tier=="silver" else 0.75)

def cache_train(rows,old_archives,fresh_source,mel,mean,scale,torch,np):
    archived={}
    cache={}
    try:
        for entry in v5.binding.config()["corpus"]:
            archived[entry["key"]]=v5.binding.archive(
               old_archives/(entry["key"]+"-source.zip"),entry["source"])
        archived["newtrain24"]=v5.binding.archive(
            old_archives/"newtrain24-source.zip",v5.prior.SOURCE)
        for row in rows:
            raw=robust.sourcefit.readpcm(row,archived,fresh_source)
            need(sha(raw)==row["source_pcm_sha256"],
                 "TRAIN_SOURCE_BYTE_SHA_IMMUTABLE")
            d={}
            for gain in (1.0,0.5):
                adjusted=gain_pcm(raw,gain,np)
                d[gain]={}
                for prefix in PREFIX_SAMPLES:
                    pcm=b"\0\0"*prefix+adjusted
                    derived={**row,"samples":row["samples"]+prefix}
                    feat=v5.features(pcm,derived,torch,np,mel)
                    data=((feat-mean)/scale).astype(np.float32)
                    need(data.ndim==2 and data.shape[1]==40
                         and np.isfinite(data).all(),"FINITE_40MEL_TRAIN_INPUT")
                    d[gain][prefix]=torch.from_numpy(data).unsqueeze(0)
            cache[row["id"]]=d
    finally:
        for arc in archived.values():arc.close()
    need(len(cache)==len(rows),"ONE_CACHE_PER_TRAIN_SOURCE_ONLY")
    return cache

def train_trajectory(label,weights,rows,groups,cache,mel,mean,scale,
                     output,torch,np,F):
    gains=GAINS[label]
    stem=ARTIFACT_STEMS[label]
    torch.manual_seed(SEED)
    model=v5.build_model(torch)
    model.load_state_dict(weights,strict=True)
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.001)
    record=[]
    start=time.perf_counter()
    sorted_groups={k:sorted(groups[k]) for k in (0,1)}
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        prefix=PREFIX_SAMPLES[(step//35+step%3)%len(PREFIX_SAMPLES)]
        weighted=[]
        for klass in (0,1):
            source_group=sorted_groups[klass][step%TRAIN_GROUPS]
            candidates=sorted(groups[klass][source_group],key=lambda r:r["id"])
            row=candidates[(step//TRAIN_GROUPS)%len(candidates)]
            for gain in gains:
                weighted.append(loss_for(
                    model,cache[row["id"]][gain][prefix],klass,prefix,
                    row["tier"],torch,F))
        loss=sum(weighted)/len(weighted)
        need(bool(torch.isfinite(loss)),
             "FINITE_EXACT_PAIRED_CTC_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),5.0,error_if_nonfinite=True)
        opt.step()
        if (step+1)%16==0 or step==0:
            record.append({"step":step+1,
                           "loss":float(loss.detach()),
                           "prefix_samples":prefix,
                           "gain_policy":list(gains)})
    model.eval()
    dest=output/"models"
    dest.mkdir(exist_ok=True)
    pt=dest/(stem+".pt")
    torch.save(model.state_dict(),pt)
    newweights=torch.load(pt,weights_only=True,map_location="cpu")
    loaded=v5.build_model(torch)
    loaded.load_state_dict(newweights,strict=True)
    loaded.eval()
    need(sum(p.numel() for p in loaded.parameters())==PARAMS,
         "UNCHANGED_43876_FSMN_PARAMETERS")
    npz=dest/(stem+".npz")
    np.savez(npz,mel_filters=mel,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy()
                for k,v in loaded.state_dict().items()})
    example=cache[sorted(rows,key=lambda r:r["id"])[0]["id"]][1.0][6400]
    onnx=dest/(stem+".onnx")
    torch.onnx.export(
        loaded,example,onnx.as_posix(),opset_version=17,dynamo=False,
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(all(p.is_file() and p.stat().st_size>10000
             for p in (pt,npz,onnx)),
         "BOTH_ACTUAL_NEW_PT_NPZ_ONNX_WEIGHTS")
    # In-sample only mechanistic diagnostics, never model selection.
    probes=[]
    positives=sorted((r for r in rows if r["mono_class"]==1),
                     key=lambda r:r["id"])[:12]
    with torch.no_grad():
        for row in positives:
            for gain in (1.0,0.5):
                x=cache[row["id"]][gain][12800]
                out=loaded(x).squeeze(0)
                prediction=v5.ctc_greedy(out.unsqueeze(0),torch)
                tokens=out.argmax(-1).tolist()
                probes.append({
                    "train_id":row["id"],"gain":gain,
                    "first_frame_argmax":int(tokens[0]),
                    "full_word_events":
                      int(prediction["mero_event"])+int(prediction["extra_events"]),
                    "any_meh_token":2 in tokens[12800//160-3:],
                    "known_silent_prefix_blank_fraction":round(
                       sum(v==0 for v in tokens[:12800//160-8])/
                       (12800//160-8),6)})
    return {"label":label,"optimizer_updates":STEPS,"source_groups":TRAIN_GROUPS,
            "gain_policy":list(gains),
            "optimizer_trace":record,
            "wall_training_seconds":round(time.perf_counter()-start,3),
            "sha256":{p.name:{"sha256":sha(p.read_bytes()),"bytes":p.stat().st_size}
                      for p in (pt,npz,onnx)},
            "train_only_probe":probes,
            "new_eval_source_groups":0,
            "model_selection_allowed":False}

def execute(a):
    source_id=guard()
    out=a.output
    need(out.is_absolute() and out.parent.is_dir()
         and not out.exists(),"ONE_NEW_B4_EXPERIMENT_OUTPUT")
    out.mkdir()
    write(out/"STARTED.json",{"schema":SCHEMA,"id":source_id,
         "max_train_updates":STEPS*2,"dev_model_forwards":0,
         "cal_locked_access":False})
    try:
        rows,dev,proof=robust.audit(
            a.old_archives,a.old_audit,a.fresh_source,a.fresh_screen,
            a.source_gate,a.prospective_source,a.prospective_screen,
            a.original_model_zip)
        input_groups(rows)
        need(all(r["role"]=="development" for r in dev),
             "DEV_METADATA_AUDIT_NO_DEV_FORWARDS")
        write(out/"SOURCE.json",{
            "train_input_pcm_sha256":sorted(r["source_pcm_sha256"] for r in rows),
            "source_roles_only_train":True,
            "dev_labels_not_used_for_optimizer":True,
            "dev_audio_model_forwards":0,
            "source_provenance":proof})
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        groups=input_groups(rows)
        weights,mel,mean,scale=authenticate_b3(a.b3_model_zip,torch,np)
        cache=cache_train(rows,a.old_archives,a.fresh_source,mel,mean,scale,torch,np)
        trajectories=[]
        for label in GAINS:
            trajectories.append(train_trajectory(
                label,weights,rows,groups,cache,mel,mean,scale,
                out,torch,np,F))
        need(len(trajectories)==2 and
             all(t["optimizer_updates"]==STEPS for t in trajectories),
             "TWO_ACTUAL_MATCHED_CONTROL_AND_CHALLENGER_FITS")
        final={
            "schema":SCHEMA,
            "status":"RESEARCH_B4_MATCHED_TRAINONLY_REAL_WEIGHTS_NO_PROMOTION",
            "B3_parent_artifact_id":B3["id"],
            "B3_parent_pt_sha256":B3["pt"],
            "per_trajectory_steps":STEPS,
            "actual_total_optimizer_steps":STEPS*2,
            "source_train_groups":TRAIN_GROUPS,
            "only_train_changed_factor":"SECOND_PAIRED_PCM16_GAIN_1_VS_HALF",
            "same_ctc_loss_and_suffix_as_B3":True,
            "same_optimizer_seed_lr_schedule":True,
            "same_B3_init_both_arms":True,
            "model_parameters":PARAMS,"trajectories":trajectories,
            "new_tts_asr_calls":0,
            "cal_locked_opened":False,
            "dev_model_forwards":0,
            "independent_heldout_quality":"NOT_RUN",
            "physical_ssc305":"NOT_RUN",
            "human_frr":"NOT_MEASURED",
            "natural_far_per_hour":"NOT_MEASURED",
            "new_model_c11_parity":"NOT_RUN",
            "product_promotion_allowed":False,
            "model_selection_allowed":False}
        write(out/"RESULT.json",final)
        print(json.dumps({
            "status":final["status"],
            "total_real_optimizer_updates":final["actual_total_optimizer_steps"],
            "trajectories":[{"name":x["label"],"weights":x["sha256"][ARTIFACT_STEMS[x["label"]]+".pt"]["sha256"]}
                            for x in trajectories],
            "product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        write(out/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_B4_TRAINONLY_WITH_NO_REPLAY",
             "error_type":type(exc).__name__,
             "reason":str(exc)[:440],
             "product_promotion_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    modes=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--provenance","--guard","--audit","--execute"):
        modes.add_argument(flag,action="store_true")
    for key in ("old-archives","old-audit","fresh-source","fresh-screen",
                "source-gate","prospective-source","prospective-screen",
                "original-model-zip","b3-model-zip","output"):
        ap.add_argument("--"+key,type=Path)
    a=ap.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.provenance:
        print(json.dumps(verify_provenance(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    required=("old_archives","old_audit","fresh_source","fresh_screen",
              "source_gate","prospective_source","prospective_screen",
              "original_model_zip")
    need(all(getattr(a,k) is not None for k in required),
         "REQUIRE_ALL_8_PREVIOUS_SOURCE_ASSETS")
    if a.audit:
        rows,dev,receipt=robust.audit(
            a.old_archives,a.old_audit,a.fresh_source,a.fresh_screen,
            a.source_gate,a.prospective_source,a.prospective_screen,
            a.original_model_zip)
        input_groups(rows)
        print(json.dumps({"status":"SOURCE_AUDITED_NO_OPTIMIZER",
             "train_groups":TRAIN_GROUPS,"train_rows":len(rows),
             "excluded_dev_rows":len(dev),
             "dev_acoustic_forwards":0},sort_keys=True));return
    need(a.b3_model_zip is not None and a.output is not None,
         "B3_AND_NEW_OUTPUT_REQUIRED")
    execute(a)

if __name__=="__main__":
    main()
