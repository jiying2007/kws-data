"""One frozen-weight V5 Hey Mero diagnosis on original train/dev PCM only.

The first 240-step optimizer / model ZIP is ALREADY FINISHED. Never train,
generate speech, call ASR/phones, open old calibration or locked PCM, tune a
threshold, or promote based on the saved development readback.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SCHEMA="english-v5-mero-saved-ctc-blank-and-generalization-diagnostic-v1"
REPO="jiying2007/kws-data"
BEFORE="0ea22c50024df7ab8797a4fb9517a01a261832db"
ORIGINAL_RUN=44453447586
MODEL_ARTIFACT={"id":14496732088,
    "name":"english-v5-mero-fsmn-ctc-44453447586",
    "size":505111,
    "sha256":"35974c692198f4a016e5a8a9320252679d0431766a2612ea03bb805ace0987cc"}
MAX_TRAIN_DIAGNOSTIC=6
MAX_DEV_DIAGNOSTIC=11
FORWARDS_LIMIT=17


def load(name):
    spec=importlib.util.spec_from_file_location(
        "en_v5_diag_"+name,HERE/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v5=load("english_v5_mero_fsmn_ctc_once")
binding=load("english_canonical_multisource_bind_once")
prior=load("english_newtrain24_real_fit_once")


def need(cond,reason):
    if not cond:raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_check():
    old=v5.source_check()
    need(old["active_target_ids"]==["hey_mero"]
         and old["positive_ctc"]==[1,2,3]
         and old["negative_ctc"]==[]
         and len(v5.ARCHIVES)==8
         and ORIGINAL_RUN==44453447586
         and MODEL_ARTIFACT["id"]==14496732088
         and FORWARDS_LIMIT==MAX_TRAIN_DIAGNOSTIC+MAX_DEV_DIAGNOSTIC
         and MODEL_ARTIFACT["sha256"]==
             "35974c692198f4a016e5a8a9320252679d0431766a2612ea03bb805ace0987cc",
         "REPLAY_ONLY_ONE_REAL_FROZEN_V5_MERO_WEIGHT_ZIP")
    return {"status":"READ_ONLY_MODEL_DERIVED_DEV_DIAG",
            "original_model_artifact":MODEL_ARTIFACT["id"],
            "max_new_KWS_forwards":FORWARDS_LIMIT,
            "new_optimizer_updates":0,"new_TTS_ASR_CTC_phone_forwards":0,
            "old_calibration_or_locked_replay":False}


def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"ONE_FIRST_MAIN_SAVED_REPLAY_ONLY")
    sha=env.get("GITHUB_SHA","")
    need(bool(re.fullmatch(r"[0-9a-f]{40}",sha))
         and bool(re.fullmatch(r"[1-9][0-9]*",env.get("GITHUB_RUN_ID",""))),
         "ACTUAL_NEW_REVIEW_RUN")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==sha
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False
         and event.get("deleted") is False,"EXACT_PROTECTED_MAIN_SUCCESSOR")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                       cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"FROZEN_MERO_FIRST_MAIN_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"head":sha,
            "original_model_run":ORIGINAL_RUN}


def check_original_git(token):
    need(type(token) is str and len(token)>=10,
         "READ_ONLY_GITHUB_ACTIONS_TOKEN")
    def get(path):
        r=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+"/actions/"+path,
            headers={"Authorization":"Bearer "+token,
              "Accept":"application/vnd.github+json",
              "User-Agent":"english-v5-frozen-weight-readback"})
        with urllib.request.urlopen(r,timeout=40) as stream:
            raw=stream.read(1000001)
        need(len(raw)<=1000000,"SMALL_GITHUB_RUN_METADATA")
        return json.loads(raw)
    original=get("runs/"+str(ORIGINAL_RUN))
    meta=get("artifacts/"+str(MODEL_ARTIFACT["id"]))
    need(original.get("id")==ORIGINAL_RUN
         and original.get("head_sha")==BEFORE
         and original.get("head_branch")=="main"
         and original.get("event")=="push"
         and original.get("run_attempt")==1
         and original.get("status")=="completed"
         and original.get("conclusion")=="success"
         and meta.get("id")==MODEL_ARTIFACT["id"]
         and meta.get("name")==MODEL_ARTIFACT["name"]
         and meta.get("size_in_bytes")==MODEL_ARTIFACT["size"]
         and meta.get("digest")=="sha256:"+MODEL_ARTIFACT["sha256"]
         and meta.get("expired") is False
         and meta.get("workflow_run",{}).get("id")==ORIGINAL_RUN
         and meta.get("workflow_run",{}).get("head_sha")==BEFORE,
         "ONLY_IMMUTABLE_ORIGINAL_240_STEP_ACTUAL_SAVED_MODEL")
    return True


def verified_model_packet(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size==MODEL_ARTIFACT["size"]
         and digest(path.read_bytes())==MODEL_ARTIFACT["sha256"],
         "PINNED_ORIGINAL_MODEL_ARTIFACT_ZIP_HASH")
    outer=zipfile.ZipFile(path)
    required={"RESULT.json","TRAINING.json","ADMISSION.json",
              "models/mero-fsmn-ctc-v5.pt",
              "models/mero-fsmn-ctc-v5.npz",
              "models/mero-fsmn-ctc-v5.onnx"}
    need(required<=set(outer.namelist())
         and len(set(outer.namelist()))==len(outer.namelist())
         and len(outer.namelist())<=12,
         "ALL_ORIGINAL_TRAINING_AND_MODEL_FILES_REQUIRED")
    state=json.loads(outer.read("RESULT.json"))
    trace=json.loads(outer.read("TRAINING.json"))
    intake=json.loads(outer.read("ADMISSION.json"))
    need(state["status"]=="ACTUAL_MONO_MERO_FSMN_CTC_RESEARCH_MODEL_TRAINED"
         and state["keyword_id"]=="hey_mero"
         and state["active_keyword_count"]==1
         and state["optimizer_steps"]==240
         and state["parameters"]==43876
         and state["quality_promotion"] is False
         and state["new_locked_or_calibration"] is False
         and state["development"]["positive_misses"]==6
         and state["development"]["positive_original_wavs"]==6
         and state["development"]["ordinary_false_or_extra"]==0
         and state["development"]["ordinary_original_wavs"]==5
         and trace["updates"]==240
         and len(trace["losses"])==len(trace["draws"])==240
         and all(type(v) in (int,float) and math.isfinite(v)
                 for v in trace["losses"])
         and trace["historical_locked_audio_read"] is False
         and trace["historical_other_keyword_grads"]==0
         and intake["status"]=="V5_REVIEWED_MONO_SOURCE_BOUND"
         and len(intake["rows"])==35
         and intake["training_calibration_or_locked"] is False,
         "EXACT_ACTUAL_FIRST_V5_RESULT_TRAIN_STEPS_AND_SAFETY_GATES")
    for name,record in state["artifacts"].items():
        member="models/"+name
        raw=outer.read(member)
        need(len(raw)==record["bytes"] and digest(raw)==record["sha256"],
             "EVERY_SAVED_MODEL_FILE_CROSS_HASH")
    return outer,state,trace,intake


def audit_model(path):
    z,model,trace,intake=verified_model_packet(path)
    try:
        return {"status":"AUTHENTICATED_ORIGINAL_240_STEP_MONO_MODEL",
                "model_run":ORIGINAL_RUN,
                "model_artifact":MODEL_ARTIFACT["id"],
                "model_parameter_count":model["parameters"],
                "actual_optimizer_updates":len(trace["losses"]),
                "input_wavs":len(intake["rows"]),
                "dev_positive_misses":model["development"]["positive_misses"],
                "pr_model_forwards":0,
                "new_audio_calls":0}
    finally:z.close()


def choose_train_diagnostics(rows):
    chosen=[]
    for clazz in (0,1):
        seen=set()
        for row in sorted((x for x in rows if
                     x["role"]=="train" and x["mono_class"]==clazz),
                     key=lambda x:(x["source_group_id"],x["id"])):
            if row["source_group_id"] not in seen:
                chosen.append(row);seen.add(row["source_group_id"])
            if len(seen)==3:break
        need(len(seen)==3,"THREE_DISJOINT_TRAIN_SOURCES_PER_CLASS_DIAGNOSTIC")
    return chosen


def replay(saved_model,source_folder,out):
    identity=guard()
    output=Path(out)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "UNREPEATED_DIAGNOSTIC_OUTPUT")
    output.mkdir()
    binding.once(output/"STARTED.json",{"schema":SCHEMA,
        "run":identity,"input_model":MODEL_ARTIFACT,
        "real_forward_calls":0,"new_optimizer_updates":0})
    models={}
    try:
        check_original_git(os.environ.get("GH_TOKEN"))
        z,original,training,intake=verified_model_packet(saved_model)
        sources=Path(source_folder)
        rows=list(intake["rows"])
        live,_=v5.admission(sources)
        need(rows==live and
             len(rows)==35 and
             Counter(r["role"] for r in rows)==
             {"train":24,"development":11},
             "SAVED_AND_RECOMPUTED_ORIGINAL_35_MONO_PCMS")
        for spec in binding.config()["corpus"]:
            models[spec["key"]]=binding.archive(
                sources/(spec["key"]+"-source.zip"),spec["source"])
        models["newtrain24"]=binding.archive(
            sources/"newtrain24-source.zip",prior.SOURCE)
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled=False
        model=v5.build_model(torch)
        params=torch.load(io.BytesIO(
            z.read("models/mero-fsmn-ctc-v5.pt")),
            map_location="cpu",weights_only=True)
        model.load_state_dict(params,strict=True);model.eval()
        array_bytes=io.BytesIO(z.read("models/mero-fsmn-ctc-v5.npz"))
        with np.load(array_bytes,allow_pickle=False) as f:
            mel=f["mel_filters"].copy()
            mean=f["train_mean"].copy();scale=f["train_scale"].copy()
            for name,value in params.items():
                need(name in f.files and
                     np.array_equal(f[name],
                     value.detach().cpu().numpy()),
                     "PREVIOUS_SAVED_NPZ_TORCH_WEIGHTS_EXACT")
        need(np.array_equal(mel,v5.make_mel_bank(np))
             and mean.shape==(40,) and scale.shape==(40,)
             and np.isfinite(mean).all() and (scale>0).all(),
             "EXACT_FROZEN_TRAIN_ONLY_MEL_FRONTEND")
        selected=[r for r in rows if r["role"]=="development"]
        selected+=choose_train_diagnostics(rows)
        need(len(selected)==FORWARDS_LIMIT
             and len({r["id"] for r in selected})==FORWARDS_LIMIT,
             "ONE_PASS_EACH_11_DEV_AND_6_NEW_TRAIN_DIAGNOSTIC")
        observations=[]
        with torch.no_grad():
            for row in selected:
                original_pcm=v5.base.load_source_pcm(row,models)
                frames=v5.features(original_pcm,row,torch,np,mel)
                x=torch.from_numpy(((frames-mean)/scale).astype(
                    np.float32)).unsqueeze(0)
                scores=model(x)[0]
                probs=torch.softmax(scores,dim=-1)
                tokens=v5.ctc_greedy(scores,torch)
                argmax=scores.argmax(-1).tolist()
                hist=Counter(argmax)
                observations.append({"id":row["id"],"source_group":
                    row["source_group_id"],"role":row["role"],
                    "class":row["mono_class"],"frames":len(argmax),
                    "ctc_collapsed":tokens["collapsed_tokens"],
                    "event":tokens["mero_event"],
                    "blank_fraction":hist[0]/len(argmax),
                    "argmax_token_count":[hist[k] for k in range(4)],
                    "mean_token_prob":probs.mean(0).tolist(),
                    "pcm_sha256":row["source_pcm_sha256"]})
        p=[r for r in observations if r["class"]==1]
        t=[r for r in p if r["role"]=="train"]
        d=[r for r in p if r["role"]=="development"]
        blank=lambda seq:sum(x["blank_fraction"] for x in seq)/len(seq)
        if blank(d)>0.98:
            hypothesis="STRONG_BLANK_COLLAPSE_ON_DEVELOPMENT"
        elif any(x["ctc_collapsed"] for x in d):
            hypothesis="PARTIAL_CTC_PHONEME_SEQUENCE_OR_GENERALIZATION"
        elif any(x["event"] for x in t):
            hypothesis="OLD_TRAIN_TO_NEW_DEV_SOURCE_GENERALIZATION_GAP"
        else:
            hypothesis="INDETERMINATE_CTC_TRAIN_OR_DECODE_FAILURE"
        loss=training["losses"]
        result={"schema":SCHEMA,
            "status":"SAVED_V5_WEIGHT_DIAGNOSTIC_COMPLETE",
            "hypothesis_not_proven_root_cause":hypothesis,
            "model_original_sha256":
              original["artifacts"]["mero-fsmn-ctc-v5.onnx"]["sha256"],
            "actual_optimizer_updates_in_this_run":0,
            "actual_model_forward_calls":len(observations),
            "old_locked_or_calibration_reopened":False,
            "training_first20_mean_loss":sum(loss[:20])/20,
            "training_last20_mean_loss":sum(loss[-20:])/20,
            "train_positive_diagnostic_event_count":
                 sum(x["event"] for x in t),
            "train_positive_diagnostic_cases":len(t),
            "dev_positive_event_count":sum(x["event"] for x in d),
            "dev_positive_cases":len(d),
            "dev_blank_ratio":blank(d),
            "train_blank_ratio":blank(t),
            "dev_negative_event_count":sum(x["event"] for x in observations
                 if x["role"]=="development" and x["class"]==0),
            "rows":observations,
            "new_tts_asr_phone_calls":0,
            "not_independent_far_frr":True,
            "ssc305":"NOT_MEASURED"}
        binding.once(output/"DIAGNOSTIC.json",result)
        print(json.dumps({k:v for k,v in result.items() if k!="rows"},
                         sort_keys=True))
    except BaseException as e:
        binding.once(output/"STOP.json",{
            "schema":SCHEMA,"status":"STOP_BOUNDED_DIAG_NO_RETRY",
            "error_type":type(e).__name__,"reason":str(e)[:500],
            "new_optimizer_updates":0,"old_locked_reuse":False})
        raise
    finally:
        for item in models.values():item.close()


def main():
    p=argparse.ArgumentParser()
    for opt in ("--source-check","--audit-model","--guard","--replay-dev"):
        p.add_argument(opt,action="store_true")
    p.add_argument("--model-zip");p.add_argument("--sources");p.add_argument("--output")
    a=p.parse_args()
    need(sum((a.source_check,a.audit_model,a.guard,a.replay_dev))==1,
         "ONE_SAVED_WEIGHT_DIAGNOSTIC_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.model_zip,"ORIGINAL_SIGNED_MODEL_ARTIFACT_REQUIRED")
    if a.audit_model:
        print(json.dumps(audit_model(a.model_zip),sort_keys=True));return
    need(a.sources and a.output,"ORIGINAL_DEV_PCM_AND_NEW_OUTPUT_REQUIRED")
    replay(a.model_zip,a.sources,a.output)


if __name__=="__main__":main()
