#!/usr/bin/env python3
"""Read-only forensic comparison of authenticated original B6 and failed B7 control.

Both training runs used public immutable TRAIN-only source groups. B7 STOPPED
before its challenger trained because exact B6 consistency replay disagreed.
Do NOT relax tensor equality or use this saved analysis for product promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

REPO="jiying2007/kws-data"
SCHEMA="english-v6-b7-b6-control-replay-drift-saved-forensics-v1"
PARENT="f9f2f56eece785da0d0c79d24dca12cdbcc29b29"
B6={
 "run":44704535156,
 "head":"e46dfa72e7f824a888ae6510a9951f1fe518b222",
 "id":14615025178,
 "name":"english-v6-b6-replay-vs-gain-consistency-44704535156",
 "size":1000729,
 "digest":"1c310b0d4945f7aa40f2576f231c94eb48b33316a6c4f79c20215b1ac80bab8d",
 "pt":"0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04",
}
B7_FAILED={
 "run":44721025043,
 "head":PARENT,
 "id":14622635105,
 "name":"english-v6-b7-replay-vs-ordinary-ctc085-44721025043",
 "size":502250,
 "digest":"64eda306ba86887ceca3902c6bc2d3309574f558a992e432c5bd0c5b009295e4",
 "job_id":136236479236,
}
MEMBER="models/mero43-b6-gain-consistency192.pt"
NEW_CANDIDATE_STEM="mero43-b7-ordinary-ctc085-192"

def need(ok,reason):
    if not ok:raise ValueError(reason)
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
def save(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write("\n")
def source_check():
    need(all(len(x["digest"])==64 for x in (B6,B7_FAILED))
         and len(B6["pt"])==64
         and B7_FAILED["head"]==PARENT
         and B6["id"]==14615025178
         and B7_FAILED["id"]==14622635105
         and B7_FAILED["job_id"]==136236479236,
         "EXACT_B6_SUCCESS_AND_B7_PRE_CHALLENGER_FAILED_ARTIFACT")
    return {"schema":SCHEMA,
        "status":"SAVED_ORIGINAL_REPLAY_DRIFT_DIAGNOSTIC_ONLY",
        "original_B6_run":B6["run"],
        "failed_B7_run":B7_FAILED["run"],
        "original_B6_artifact":B6["id"],
        "failed_B7_artifact":B7_FAILED["id"],
        "new_model_forwards":0,"new_optimizer_updates":0,
        "new_tts_asr_phone_calls":0,
        "b7_candidate_fitted":False,
        "source_and_tensor_guard_relaxed":False,
        "cal_locked_access":False,
        "model_selection_allowed":False,"product_promotion_allowed":False}
def guard(env=None,event=None,parent=None):
    source_check()
    e=os.environ if env is None else env
    need(all(e.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"FIRST_PUBLIC_MAIN_SAVED_ONLY")
    h=e.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",h)
         and re.fullmatch("[1-9][0-9]*",e.get("GITHUB_RUN_ID","")),
         "VALID_ONE_SHOT_DIAGNOSTIC_RUN")
    if event is None:
        event=json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],text=True).strip()
    need(event.get("before")==PARENT
         and event.get("after")==h
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False
         and event.get("deleted") is False
         and parent==PARENT,"EXACT_FAILED_B7_MAIN_SUCCESSOR_ONCE")
    return {"run_id":e["GITHUB_RUN_ID"],"head":h,"before":PARENT}
def remote_check(token):
    need(type(token) is str and len(token)>=10,"READ_ONLY_GITHUB_ACTIONS_TOKEN")
    def get(path):
        req=urllib.request.Request("https://api.github.com/repos/"+REPO+path,
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"kws-b7-replay-diff-saved-forensics"})
        with urllib.request.urlopen(req,timeout=40) as f:
            raw=f.read(1000001)
        need(len(raw)<=1000000,"BOUNDED_REMOTE_METADATA")
        return json.loads(raw)
    for record,conclusion in ((B6,"success"),(B7_FAILED,"failure")):
        run=get("/actions/runs/"+str(record["run"]))
        need(run.get("id")==record["run"]
             and run.get("head_sha")==record["head"]
             and run.get("head_branch")=="main"
             and run.get("event")=="push"
             and run.get("run_attempt")==1
             and run.get("status")=="completed"
             and run.get("conclusion")==conclusion,
             "AUTHENTIC_REPLAY_RUN_SUCCESS_OR_FAILURE_RECEIPT")
        a=get("/actions/artifacts/"+str(record["id"]))
        need(a.get("id")==record["id"]
             and a.get("name")==record["name"]
             and a.get("size_in_bytes")==record["size"]
             and a.get("digest")=="sha256:"+record["digest"]
             and a.get("expired") is False
             and a.get("workflow_run",{}).get("id")==record["run"]
             and a.get("workflow_run",{}).get("head_sha")==record["head"],
             "EXACT_ORIGINAL_SAVED_WEIGHTS_ZIP_SHA")
    failed_jobs=get("/actions/runs/"+str(B7_FAILED["run"])+"/jobs?per_page=100")
    matches=[j for j in failed_jobs.get("jobs",[])
             if j.get("id")==B7_FAILED["job_id"]]
    need(len(matches)==1
         and matches[0].get("name")=="exact-b3-trainonly"
         and matches[0].get("conclusion")=="failure",
         "ACTUAL_ONE_FAILED_B7_TRAIN_JOB")
    return {"original_B6_zip":B6["id"],"failed_B7_zip":B7_FAILED["id"],
            "prior_replay_failure":"B6_GAIN_KL_CONTROL_TENSOR_REPLAY_DRIFT_STOP"}

def read_zip(path,record):
    path=Path(path)
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size==record["size"]
         and digest(path.read_bytes())==record["digest"],
         "EXACT_UNMUTATED_B6_OR_B7_ZIP_BYTES")
    with zipfile.ZipFile(path) as z:
        need(z.testzip() is None,"WHOLE_ARTIFACT_ZIP_CRC")
        names=set(z.namelist())
        pt=z.read(MEMBER)
        result={name:json.loads(z.read(name))
                for name in ("SOURCE.json","RESULT.json","STOP.json","STARTED.json")
                if name in names}
        need(not any(NEW_CANDIDATE_STEM in name for name in names),
             "FAILED_B7_MUST_NOT_HAVE_NEW_CHALLENGER_WEIGHT")
    return pt,result

def analyze(original_zip,failed_zip,torch):
    original,original_meta=read_zip(original_zip,B6)
    failed,failed_meta=read_zip(failed_zip,B7_FAILED)
    orig=original_meta.get("RESULT.json",{})
    stopped=failed_meta.get("STOP.json",{})
    need(digest(original)==B6["pt"]
         and orig.get("status")=="RESEARCH_B6_TRAIN_ONLY_GAIN_POSTERIOR_KL_NO_PROMOTION"
         and orig.get("actual_optimizer_updates")==384
         and orig.get("train_source_groups")==35
         and orig.get("B4_half_baseline_replay",{}).get("status")==
             "PASS_EXACT_B4_HALF_TENSOR_REPLAY"
         and orig.get("dev_model_forwards")==0
         and orig.get("cal_locked_opened") is False,
         "ACTUAL_B6_TRAINING_AND_EXACT_FROZEN_PARENT")
    need(stopped.get("schema")=="english-v6-b7-trainonly-ordinary-ctc-weight-085-v1"
         and stopped.get("status")=="STOP_B7_NO_VALID_NEW_WEIGHTS"
         and stopped.get("reason")=="B6_GAIN_KL_CONTROL_TENSOR_REPLAY_DRIFT_STOP"
         and stopped.get("dev_pcm_in_gradients") is False
         and stopped.get("cal_locked_opened") is False
         and stopped.get("product_promotion_allowed") is False,
         "NO_B7_CANDIDATE_AND_NO_BYPASSED_GATE")
    ref_source=original_meta.get("SOURCE.json",{})
    failed_source=failed_meta.get("SOURCE.json",{})
    need(ref_source.get("train_shas")==failed_source.get("train_pcm_sha256")
         and ref_source.get("dev_pcm_in_optimizer") is False
         and failed_source.get("dev_pcm_in_gradients") is False
         and ref_source.get("dev_metadata_only")==failed_source.get("dev_rows_metadata_only"),
         "BYTE_EQUAL_TRAIN_SOURCE_IDENTITY_BETWEEN_RUNS")
    a=torch.load(io.BytesIO(original),weights_only=True,map_location="cpu")
    b=torch.load(io.BytesIO(failed),weights_only=True,map_location="cpu")
    need(type(a) is dict or hasattr(a,"keys"),"REAL_TENSOR_STATE_MAPPING")
    need(set(a)==set(b) and len(a)>0,"SAME_MODEL_PARAMETER_KEYS")
    per_tensor=[]
    total_params=0
    different=0
    sum_abs=0.0
    sum_sq=0.0
    overall_max=0.0
    for key in sorted(a):
        orig_tensor=a[key]
        new_tensor=b[key]
        need(orig_tensor.shape==new_tensor.shape and
             orig_tensor.dtype==new_tensor.dtype and
             bool(torch.isfinite(orig_tensor).all()) and
             bool(torch.isfinite(new_tensor).all()),
             "MATCHING_FINITE_TRAINED_43K_FSMN_WEIGHT_SHAPES")
        count=orig_tensor.numel()
        old=orig_tensor.detach().to(torch.float64).reshape(-1)
        new=new_tensor.detach().to(torch.float64).reshape(-1)
        delta=(new-old).abs()
        changed=int((orig_tensor!=new_tensor).sum().item())
        max_abs=float(delta.max().item()) if count else 0.0
        mean_abs=float(delta.mean().item()) if count else 0.0
        rms=(float(torch.mean(delta*delta).item())**0.5) if count else 0.0
        total_params+=count
        different+=changed
        sum_abs+=mean_abs*count
        sum_sq+=rms*rms*count
        overall_max=max(overall_max,max_abs)
        per_tensor.append({
            "tensor":key,"numel":count,"dtype":str(orig_tensor.dtype),
            "shape":list(orig_tensor.shape),
            "different_values":changed,
            "max_absolute_difference":max_abs,
            "mean_absolute_difference":mean_abs,
            "rms_difference":rms})
    need(total_params>=43876 and different>0,
         "GENUINE_NONZERO_B6_CONTROL_REPLAY_MISMATCH")
    per_tensor.sort(key=lambda x:(x["max_absolute_difference"],x["different_values"]),reverse=True)
    return {
       "schema":SCHEMA,
       "status":"PASS_REPLAY_DRIFT_QUANTIFIED_NO_B7_TRAIN_OR_PROMOTION",
       "original_B6_model_pt_sha256":B6["pt"],
       "failed_control_pt_sha256":digest(failed),
       "source_identity_byte_equal":True,
       "model_parameter_names_equal":True,
       "total_tensor_values":total_params,
       "differing_tensor_values":different,
       "fraction_values_different":different/total_params,
       "global_max_absolute_difference":overall_max,
       "mean_absolute_difference":sum_abs/total_params,
       "rms_difference":(sum_sq/total_params)**0.5,
       "all_tensor_details":per_tensor,
       "no_challenger_weights_produced":True,
       "previous_exact_tensor_gate_relaxed":False,
       "actual_new_model_forwards":0,
       "new_optimizer_updates":0,
       "new_tts_asr_phone":0,
       "training_recovery_permitted":False,
       "cal_locked_access":False,
       "model_selection_allowed":False,
       "product_promotion_allowed":False,
       "numeric_drift_is_not_by_itself_a_proven_causal_explanation":True}

def evaluate(original_path,failed_path,out):
    source_check()
    identity=guard()
    output=Path(out)
    need(output.is_absolute() and not output.exists()
         and output.parent.is_dir(),
         "ONE_NEW_BYTE_BOUND_DIAGNOSTIC_OUTPUT")
    output.mkdir()
    save(output/"STARTED.json",{"schema":SCHEMA,
        "main":identity,"old_artifact":B6["id"],
        "failed_artifact":B7_FAILED["id"],
        "new_model_forwards":0,"new_optimizer_updates":0})
    try:
        remote=remote_check(os.environ.get("GH_TOKEN"))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        proof=analyze(original_path,failed_path,torch)
        proof["remote_provenance"]=remote
        proof["new_run"]=identity
        save(output/"DIFF.json",proof)
        print(json.dumps({
            "status":proof["status"],
            "total_tensor_values":proof["total_tensor_values"],
            "differing_tensor_values":proof["differing_tensor_values"],
            "global_max_absolute_difference":proof["global_max_absolute_difference"],
            "rms_difference":proof["rms_difference"],
            "top_changed_tensors":proof["all_tensor_details"][:5],
            "new_optimizer_updates":0,
            "training_recovery_permitted":False,
            "product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        save(output/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_B7_REPLAY_DIAGNOSTIC_UNTRUSTED",
             "error_type":type(exc).__name__,"reason":str(exc)[:450],
             "new_optimizer_updates":0,"product_promotion_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    args=ap.add_mutually_exclusive_group(required=True)
    args.add_argument("--source-check",action="store_true")
    args.add_argument("--evaluate",action="store_true")
    ap.add_argument("--original-zip",type=Path)
    ap.add_argument("--failed-zip",type=Path)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True))
        return
    need(a.original_zip is not None and a.failed_zip is not None
         and a.output is not None,"TWO_ORIGINAL_IMMUTABLE_ARTIFACTS_REQUIRED")
    evaluate(a.original_zip,a.failed_zip,a.output)

if __name__=="__main__":
    main()
