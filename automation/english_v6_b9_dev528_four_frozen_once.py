#!/usr/bin/env python3
"""B9 DEV528: actual four-way frozen model scoring on NEW QA-qualified original PCM.

B6 original, B8 ordinary-CTC 0.85 challenger and both same-job B9 arms.
Four positive and twelve ordinary original DEV WAVs are the only admissible
denominators (12 real synthetic voice groups, 44 UNKNOWN excluded). Three
fixed 1/0.75/0.5 gain perturbations are NOT new independent samples.
Read/write no training, CAL, LOCKED, threshold, model promotion or TTS/ASR.
Pin immutable original model+source+QA artifacts, real frame logits/events.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(name):
    spec=importlib.util.spec_from_file_location(
        "b9_bounded_source_"+name,HERE/(name+".py"))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

saved=load("english_v6_b9_dev528_saved_screen_once")
ref=load("english_v6_b8_dev520_frozen_receipt_v2_once")
v5=ref.v5
REPO="jiying2007/kws-data"
SCHEMA="english-v6-b9-dev528-four-frozen-new-source-v1"
BEFORE="1b47e4a5d0c30ae59bc48e754772852c90b28a55"
PINNED_B8_EVALUATOR_BLOB="d4c08b504945b1928b01e73ac9b11ef1dd13e29b"
SOURCE={"id":14693632800,"run":44870282676,
        "head":"2bd78ea27b2439e92c82ee5672712eba2e86405a",
        "name":"english-v6-b9-fresh-dev528-original-44870282676",
        "bytes":1206914,
        "sha":"d606ee883a1abd38cc95a278f6b7b7064476c8b00add1163e27c247c41ab7032"}
SCREEN={"id":14694540474,"run":44871646677,
        "head":BEFORE,
        "name":"english-v6-b9-dev528-saved-machine-screen-44871646677",
        "bytes":8508,
        "sha":"ae3062102541dbfc6f18a3d8db6a170fad529b0e97095b31bbc0ee6b8b6d026b"}
B9={"id":14689218171,"run":44860855857,
    "head":"846c745ebcee58a1beb27300ddcd7ec8e2178916",
    "name":"english-v6-b9-half-gain-hey-presence-44860855857",
    "bytes":1001753,
    "sha":"1a8b270200e0c72ab1025f139d16abde43c209f142d72e71a7201b847e06a612"}
B6=ref.ARCHIVES["B6"]
B8=ref.ARCHIVES["B8"]
MODELS={
 "B6_consistency":{"archive":"B6",
   "pt":"0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04"},
 "B8_challenger":{"archive":"B8",
   "pt":"cc16b81f84e4f21c5347ca153f405ac41ed116e3eedfaeed5677d1a4bc86e22f"},
 "B9_control":{"archive":"B9","arm":"control",
   "path":"control_b6_weight0/models/mero43-b6-gain-consistency192",
   "pt":"7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"},
 "B9_challenger":{"archive":"B9","arm":"challenger",
   "path":"challenger_half_hey003/models/mero43-b9-halfgain-hey-presence192",
   "pt":"d672dcc87ca9e5d2c6dcf5952251a3c40fe8e7956d3b7c8753667c3093761cea"}
}
GAIN=(1.0,0.75,0.5)
SOURCE_ORIGINALS=60
POSITIVE=4
ORDINARY=12
UNKNOWN=44
SOURCE_GROUPS=12
QUALIFIED_GROUPS=4
PARAMS=43876
TRAINED_B9_DIFFERENCES=43754
PARAKEET_FAILED_ID="merov6b9dev12-531-mero-slow"
QUALIFIED_FAMILY_IDS={
    "piper-libritts-p1513","piper-libritts-p5154",
    "piper-libritts-p6308","piper-libritts-p664"}

def need(ok,reason):
    if not ok:raise ValueError(reason)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def blob(path):
    raw=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(raw)).encode("ascii")+b"\0"+raw).hexdigest()

def save(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)
        f.write("\n")

def source_check():
    prev=saved.source_check()
    need(prev["status"]==
           "SAVED_PROSPECTIVE_B9_DEV528_299_OF_300_REAL_OBSERVATIONS_PINNED"
         and prev["original_observer_run"]==44870958716
         and prev["saved_source_rows"]==SOURCE_ORIGINALS
         and prev["new_kws_model_forwards"]==0
         and saved.BEFORE=="fedc502a5b38445be1c6b143a94553a38850bcc1"
         and saved.SOURCE_RUN==SOURCE["run"]
         and saved.SOURCE["id"]==SOURCE["id"]
         and saved.SOURCE["digest"]=="sha256:"+SOURCE["sha"]
         and saved.PARAKEET_FAILED_ID==PARAKEET_FAILED_ID
         and saved.REVIEW_COUNT==SOURCE_ORIGINALS
         and saved.MIN_DEVELOPMENT_GROUPS==2
         and SCREEN["head"]==BEFORE
         and len(MODELS)==4
         and set(MODELS)=={"B6_consistency","B8_challenger",
                          "B9_control","B9_challenger"}
         and ref.MODELS["B6_consistency"]["pt"]==MODELS["B6_consistency"]["pt"]
         and ref.MODELS["B8_challenger"]["pt"]==MODELS["B8_challenger"]["pt"]
         and ref.ARCHIVES["B6"]==B6 and ref.ARCHIVES["B8"]==B8
         and blob(HERE/"english_v6_b8_dev520_frozen_receipt_v2_once.py")==
                 PINNED_B8_EVALUATOR_BLOB
         and GAIN==(1.0,0.75,0.5)
         and (POSITIVE,ORDINARY,UNKNOWN,QUALIFIED_GROUPS)==(4,12,44,4)
         and B9["id"]==14689218171
         and TRAINED_B9_DIFFERENCES==43754,
         "EXACT_FROZEN_B6_B8_B9_AND_NEW_299_OBSERVER_SOURCE")
    wf=(ROOT/".github/workflows/english-v6-b9-dev528-four-frozen-once.yml").read_text()
    need(all(wf.count("'"+x["sha"]+"'")==1 for x in (B6,B8,B9))
         and "github.event.before == '"+BEFORE+"'" in wf,
         "EXACT_FROZEN_MODEL_ZIPS_AND_ONE_MAIN_SUCCESSOR")
    return {
      "schema":SCHEMA,"status":"SOURCE_ONLY_NO_NEURAL_FORWARD",
      "keyword":"Hey Mero","model_parameters":PARAMS,
      "models":{n:{"artifact":(B9 if x["archive"]=="B9"
                                 else B6 if x["archive"]=="B6" else B8)["id"],
                   "pt_sha256":x["pt"]} for n,x in MODELS.items()},
      "new_source_run":SOURCE["run"],"saved_screen_run":SCREEN["run"],
      "source_originals":SOURCE_ORIGINALS,
      "positive_originals":POSITIVE,"ordinary_originals":ORDINARY,
      "unknown_excluded":UNKNOWN,"qualified_groups":QUALIFIED_GROUPS,
      "fixed_gains":list(GAIN),"max_actual_frozen_forwards":192,
      "new_training_updates":0,"new_audio_asr_phone_calls":0,
      "new_model_forwards_in_PR":0,
      "cal_locked_opened":False,"no_old_DEV520_reuse":True,
      "model_selection_allowed":False,"product_promotion_allowed":False
    }

def guard():
    source_check()
    e=os.environ
    need(all(e.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),
        "FIRST_PUBLIC_PROTECTED_MAIN_ONLY")
    head=e.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",e.get("GITHUB_RUN_ID","")) is not None,
         "VALID_GITHUB_RUN_ID_AND_COMMIT")
    event=json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE
         and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False
         and event.get("deleted") is False
         and subprocess.check_output(["git","rev-parse","HEAD^1"],cwd=ROOT,
                                    text=True).strip()==BEFORE,
         "EXACT_FIRST_NEW_B9_DEV_SOURCE_SCORE_NO_REPLAY")
    return {"run_id":e["GITHUB_RUN_ID"],"head":head,"parent":BEFORE}

def provenance(token):
    need(isinstance(token,str) and len(token)>10,
         "READONLY_GITHUB_PROVENANCE_TOKEN_REQUIRED")
    def get(uri):
        req=urllib.request.Request(
          "https://api.github.com/repos/"+REPO+"/actions/"+uri,
          headers={"Authorization":"Bearer "+token,
                   "Accept":"application/vnd.github+json",
                   "User-Agent":"kws-b9-original-source-frozen-research"})
        with urllib.request.urlopen(req,timeout=40) as res:
            raw=res.read(1000001)
        need(len(raw)<=1000000,"BOUND_RESEARCH_METADATA")
        return json.loads(raw)
    artifacts=(SOURCE,SCREEN,B6,B8,B9)
    for obj in artifacts:
        run=get("runs/"+str(obj["run"]))
        need(run.get("id")==obj["run"]
             and run.get("head_sha")==obj["head"]
             and run.get("head_branch")=="main"
             and run.get("run_attempt")==1
             and run.get("event")=="push"
             and run.get("status")=="completed"
             and run.get("conclusion")=="success",
             "EXACT_PROTECTED_MAIN_ARCHIVE_ORIGIN")
        art=get("artifacts/"+str(obj["id"]))
        need(art.get("id")==obj["id"]
             and art.get("name")==obj["name"]
             and art.get("size_in_bytes")==obj["bytes"]
             and art.get("digest")=="sha256:"+obj["sha"]
             and art.get("expired") is False
             and art.get("workflow_run",{}).get("id")==obj["run"]
             and art.get("workflow_run",{}).get("head_sha")==obj["head"],
             "ACTUAL_BYTE_AUTHENTIC_MODEL_AND_SOURCE_ARCHIVES")
    return {"source":SOURCE["id"],"screen":SCREEN["id"],
            "models":{k:o["id"] for k,o in (("B6",B6),("B8",B8),("B9",B9))}}

def select(source,screen):
    original=saved.source_rows(source)
    doc=json.loads((Path(screen)/"SCREEN.json").read_bytes())
    need(doc.get("schema")==saved.SCHEMA
         and doc.get("status")=="PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY"
         and doc.get("source_run")==SOURCE["run"]
         and doc.get("original_observer_run")==44870958716
         and doc.get("source_group_counts")=={"development":QUALIFIED_GROUPS}
         and doc.get("tier_counts")=={"silver":15,"weak":1,"unknown":UNKNOWN}
         and doc.get("observed_label_counts")=={"0":ORDINARY,"2":POSITIVE}
         and doc.get("native_attempts")==300
         and doc.get("native_successful_captures")==299
         and doc.get("parakeet_failed_source_id")==PARAKEET_FAILED_ID
         and doc.get("originals_with_five_complete_native")==59
         and doc.get("hard_hey_mary_requested_is_not_gold") is True
         and doc.get("full_6_2_2_2_gate_passed") is False
         and doc.get("nearphone_as_ordinary_negative") is False
         and doc.get("machine_only_no_human_gold") is True
         and doc.get("new_audio_or_model_forwards")==0
         and doc.get("new_optimizer_updates")==0
         and doc.get("new_kws_model_forwards")==0
         and doc.get("training_snapshot_allowed") is False
         and doc.get("locked_kws_test_opened") is False
         and doc.get("product_promotion") is False,
         "FROZEN_INDEPENDENT_B9_NEW_SOURCE_QA_ONLY")
    rows={r["id"]:r for r in original}
    grouped=defaultdict(list);seen=set();eligible=[]
    for row in doc.get("rows",[]):
        sid=row.get("id")
        need(sid in rows and sid not in seen
             and row.get("role")=="development"
             and row.get("human_gold") is False
             and row.get("training_allowed") is False,
             "ONLY_DISTINCT_DEV_ROWS_ALLOWED")
        seen.add(sid)
        src=rows[sid]
        need(all(row.get(k)==src.get(k) for k in (
                 "source_group_id","speaker_index","role","cell_id",
                 "requested_class","pcm_sha256","wav_sha256","samples")),
             "EXACT_SOURCE_AUDIO_SHA_AND_ROLE_QUALITY_JOIN")
        grouped[row["source_group_id"]].append(row)
        if src["cell_id"] in ("nearphone-meero-challenge","hard-hey-mary"):
            need(row["tier"]=="unknown" and row["observed_class"] is None,
                 "NO_UNQUALIFIED_NEARPHONE_OR_HARD_NEGATIVE_ADMISSION")
        if sid==PARAKEET_FAILED_ID:
            need(row["tier"]=="unknown" and row["observed_class"] is None
                 and row["reason"]=="ONE_AUTHENTIC_PAR-AKEET_CAPTURE_FAILED",
                 "NO_FAILED_PAR_AKEET_AS_POSITIVE_OR_ORDINARY")
        if row["tier"] in ("silver","weak"):
            val=row["observed_class"]
            need(val in (0,2) and val==src["requested_class"]
                 and src["cell_id"] not in ("nearphone-meero-challenge","hard-hey-mary"),
                 "ONLY_SOURCE_QUALIFIED_MODEL_FORWARDS")
            eligible.append({**src,
              "source_pcm_sha256":src["pcm_sha256"],
              "mono_class":int(val==2),"machine_tier":row["tier"]})
        else:
            need(row["tier"]=="unknown" and row["observed_class"] is None,
                 "NO_UNKNOWN_TEST_LEAK")
    need(len(seen)==len(rows)==SOURCE_ORIGINALS
         and len(grouped)==SOURCE_GROUPS
         and all(len(c)==5 for c in grouped.values()),
         "EXACT_NEW_60_PCM_IN_12_FIVE_CELL_SOURCE_GROUPS")
    qualified={g for g,c in grouped.items()
               if {0,2}<={r["observed_class"] for r in c
                          if r["tier"] in ("silver","weak")}}
    positives={r["source_group_id"] for r in eligible if r["mono_class"]==1}
    need(qualified==positives==QUALIFIED_FAMILY_IDS
         and set(doc["source_group_identity_members"])==qualified
         and Counter(r["mono_class"] for r in eligible)=={0:ORDINARY,1:POSITIVE},
         "ONLY_FOUR_SOURCE_POSITIVE_AND_TWELVE_ORDINARY")
    return sorted(eligible,key=lambda x:x["id"]),{
       "status":"AUDITED_B9_NEW_FOUR_POSITIVE_TWELVE_ORDINARY",
       "originals":SOURCE_ORIGINALS,"source_groups":SOURCE_GROUPS,
       "machine_positive":POSITIVE,"machine_ordinary":ORDINARY,
       "unknown_excluded":UNKNOWN,
       "qualified_development_groups":sorted(qualified),
       "unqualified_hardneg_count":12,
       "human_gold":False,"cal_locked_opened":False}

def verify_b9_archive(path,torch,np):
    path=Path(path)
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size==B9["bytes"]
         and sha(path.read_bytes())==B9["sha"],
         "EXACT_ORIGINAL_B9_TRAINING_ZIP_HASH")
    with zipfile.ZipFile(path) as z:
        need(z.testzip() is None,"ALL_B9_ORIGINAL_TRAINING_ZIP_CRC")
        result=json.loads(z.read("RESULT.json"))
        source=json.loads(z.read("SOURCE.json"))
        need(result.get("schema")=="english-v6-b9-half-gain-positive-hey-presence-v1"
             and result.get("status")=="RESEARCH_B9_REAL_HALF_GAIN_HEY_TRAIN_NO_PROMOTION"
             and result.get("actual_optimizer_updates")==384
             and result.get("same_job_parent") is True
             and result.get("same_35_train_source_groups")==35
             and result.get("model_parameters")==PARAMS
             and result.get("different_trained_float_values")==TRAINED_B9_DIFFERENCES
             and result.get("frozen_B6_parent_pt_sha256")==MODELS["B6_consistency"]["pt"]
             and result.get("single_factor")=="CLIP_LEVEL_HEY_PRESENCE_POSITIVE_HALF_GAIN"
             and result.get("control_new_loss_weight")==0.0
             and result.get("challenger_new_loss_weight")==0.03
             and result.get("unchanged_ordinary_ctc_weight")==0.60
             and result.get("unchanged_gain_KL_weight")==0.03
             and result.get("dev_gradients")==0
             and result.get("dev_model_forwards")==0
             and result.get("cal_locked_opened") is False
             and result.get("model_selection_allowed") is False
             and result.get("product_promotion_allowed") is False
             and result.get("B7_historical_exact_replay")=="FAILED_UNCHANGED"
             and source.get("dev_pcm_in_gradients") is False
             and len(source.get("train_pcm_sha256",[]))>=35,
             "REAL_384_TRAIN_UPDATES_B9_SOURCE_AND_NOT_QUALITY_CLAIM")
        control=result["control"];challenger=result["challenger"]
        need(control["optimizer_updates"]==challenger["optimizer_updates"]==192
             and challenger["hey_presence_weight"]==0.03
             and control["gain_consistency_enabled"] is True,
             "B9_TRUE_MATCHED_NEW_LOSS_SINGLE_FACTOR")
        controls=[(r["step"],r["prefix_samples"]) for r in control["trace"]]
        challenges=[(r["step"],r["prefix_samples"]) for r in challenger["trace"]]
        need(controls==challenges and len(controls)>8,
             "EXACT_B9_SAME_JOB_CONTROL_AND_CHALLENGER_SCHEDULE")
        tensors={}
        for name,model in MODELS.items():
            if model["archive"]!="B9":continue
            arm=result[model["arm"]]
            path0=model["path"];basename=path0.rsplit("/",1)[-1]
            buf={}
            for ext in ("pt","npz","onnx"):
                key=basename+"."+ext
                raw=z.read(path0+"."+ext)
                need(len(raw)>10000 and sha(raw)==arm["files"][key]["sha256"]
                     and len(raw)==arm["files"][key]["bytes"],
                     "ORIGINAL_B9_TRAINED_"+name+"_"+ext)
                buf[ext]=raw
            need(sha(buf["pt"])==model["pt"],"EXACT_B9_SAVED_PT_IDENTITY_"+name)
            weights=torch.load(io.BytesIO(buf["pt"]),map_location="cpu",
                               weights_only=True)
            net=v5.build_model(torch)
            net.load_state_dict(weights,strict=True);net.eval()
            with np.load(io.BytesIO(buf["npz"]),allow_pickle=False) as ar:
                need(set(ar.files)==set(weights)|{"mel_filters","train_mean","train_scale"}
                     and all(np.array_equal(ar[k],v.detach().cpu().numpy())
                             for k,v in weights.items()),
                     "B9_PYTORCH_NPZ_SAVED_TENSOR_PARITY")
                bank=ar["mel_filters"].copy()
                mean=ar["train_mean"].copy()
                scale=ar["train_scale"].copy()
            need(sum(p.numel() for p in net.parameters())==PARAMS
                 and bank.shape==(257,40) and mean.shape==(40,)
                 and scale.shape==(40,) and np.isfinite(bank).all()
                 and np.isfinite(mean).all() and np.isfinite(scale).all()
                 and bool((scale>0).all()),
                 "FINITELY_NORMALIZED_43876_PURE_FSMN_B9")
            tensors[name]=(net,bank,mean,scale)
    a=tensors["B9_control"][0].state_dict()
    b=tensors["B9_challenger"][0].state_dict()
    need(set(a)==set(b)
         and sum(int(torch.count_nonzero(a[k]!=b[k]).item())
                 for k in a)==TRAINED_B9_DIFFERENCES,
         "ACTUAL_43754_B9_DISTINCT_LEARNED_FLOATS")
    return tensors,source

def execute(source,screen,b6_zip,b8_zip,b9_zip,output):
    import numpy as np
    import torch
    run=guard()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "ONE_NEW_FROZEN_DEV528_OUTPUT_ONLY")
    output.mkdir()
    save(output/"STARTED.json",{"schema":SCHEMA,"run":run,
         "models":list(MODELS),"max_real_model_forward_conditions":192,
         "cal_locked_opened":False,"new_optimizer_updates":0,
         "new_tts_asr_calls":0,"model_promotion_allowed":False})
    try:
        remote=provenance(os.environ.get("GH_TOKEN"))
        rows,screen_receipt=select(source,screen)
        need(len(rows)==POSITIVE+ORDINARY,"ONLY_16_NEW_APPROVED_ORIGINALS")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        old6=ref.verify_model_archive(b6_zip,"B6",torch,np)
        old8=ref.verify_model_archive(b8_zip,"B8",torch,np)
        b9,b9source=verify_b9_archive(b9_zip,torch,np)
        with zipfile.ZipFile(b6_zip) as z6,zipfile.ZipFile(b8_zip) as z8:
            s6=json.loads(z6.read("SOURCE.json"))
            s8=json.loads(z8.read("SOURCE.json"))
        need(s6.get("train_shas")==s8.get("train_pcm_sha256")
             and s6.get("train_shas")==b9source.get("train_pcm_sha256")
             and s6.get("dev_pcm_in_optimizer") is False
             and s8.get("dev_pcm_in_gradients") is False,
             "EXACT_IDENTICAL_TRAIN_SOURCE_SHA_NO_DEVELOPMENT_LEAKAGE")
        models={"B6_consistency":old6["B6_consistency"],
                "B8_challenger":old8["B8_challenger"],
                **b9}
        need(set(models)==set(MODELS),"ONLY_FOUR_BYTE_AUTHENTIC_FROZEN_MODELS")
        bank,mean,scale=models["B6_consistency"][1:]
        for name,(_,mb,mm,ms) in models.items():
            need(np.array_equal(bank,mb) and np.array_equal(mean,mm)
                 and np.array_equal(scale,ms),
                 "EXACT_SAME_FROZEN_MEL_FRONTEND_"+name)
        results=[];original_logits={}
        with torch.no_grad():
            for row in rows:
                pcm=ref.source_pcm(row,source)
                conditions={}
                for gain in GAIN:
                    p=ref.gain_pcm(pcm,gain,np)
                    features=v5.features(p,row,torch,np,bank)
                    need(features.ndim==2 and features.shape[1]==40
                         and np.isfinite(features).all(),
                         "FINITE_FROZEN_MEL40_FRAMES")
                    model_input=torch.from_numpy(
                        ((features-mean)/scale).astype(np.float32)).unsqueeze(0)
                    per={}
                    for name,(net,*_) in models.items():
                        logits=net(model_input)
                        need(logits.shape[-1]==4
                             and bool(torch.isfinite(logits).all()),
                             "FOUR_TOKEN_FSMN_CTC_FINITE_LOGITS")
                        argmax=logits.argmax(-1).reshape(-1).tolist()
                        decoded=v5.ctc_greedy(logits,torch)
                        n=int(decoded["mero_event"])+int(decoded["extra_events"])
                        per[name]={
                          "full_phrase_event_count":n,
                          "any_complete_event":bool(n),
                          "first_frame_argmax":int(argmax[0]),
                          "any_meh_argmax":2 in argmax,
                          "any_hey_argmax":1 in argmax,
                          "collapsed_tokens":decoded["collapsed_tokens"],
                          "frames":len(argmax),
                          "ctc_extra_events":int(decoded["extra_events"])}
                        if gain==1.0:
                            original_logits[row["id"]+"__"+name]=(
                                logits.squeeze(0).cpu().numpy().astype(np.float32))
                    conditions[str(gain)]=per
                results.append({
                    "id":row["id"],"source_group_id":row["source_group_id"],
                    "role":"development","tier":row["machine_tier"],
                    "pcm_sha256":row["source_pcm_sha256"],
                    "positive":bool(row["mono_class"]),
                    "conditions":conditions})
        need(len(results)==POSITIVE+ORDINARY
             and Counter(r["positive"] for r in results)=={
                 True:POSITIVE,False:ORDINARY},
             "ALL_SIXTEEN_QUALIFIED_ORIGINALS_WITHOUT_UNKNOWN")
        npz=output/"ORIGINAL_FRAME_LOGITS.npz"
        np.savez_compressed(npz,**original_logits)
        need(npz.is_file() and npz.stat().st_size>1000,
             "SAVE_ACTUAL_FRAME_LOGIT_ROWS_FOR_READBACK")
        summary={}
        for gain in GAIN:
            key=str(gain);summary[key]={}
            pos=[r for r in results if r["positive"]]
            neg=[r for r in results if not r["positive"]]
            for name in MODELS:
                summary[key][name]={
                    "positive_originals":len(pos),
                    "positive_hits":sum(
                        r["conditions"][key][name]["any_complete_event"] for r in pos),
                    "ordinary_originals":len(neg),
                    "ordinary_complete_event_clips":sum(
                        r["conditions"][key][name]["any_complete_event"] for r in neg),
                    "ordinary_false_or_extra_events":sum(
                        r["conditions"][key][name]["full_phrase_event_count"] for r in neg),
                    "positive_missing_hey_argmax":sum(
                        not r["conditions"][key][name]["any_hey_argmax"] for r in pos),
                    "positive_missing_meh_argmax":sum(
                        not r["conditions"][key][name]["any_meh_argmax"] for r in pos)}
        result={
          "schema":SCHEMA,
          "status":"B9_NEW_SOURCE_FOUR_WAY_FROZEN_DEV_RESEARCH_DIAGNOSTIC_ONLY",
          "run":run,"source_receipt":screen_receipt,
          "original_artifacts":remote,
          "model_pt_sha256":{k:v["pt"] for k,v in MODELS.items()},
          "original_frame_logits_zip_member_sha256":sha(npz.read_bytes()),
          "fixed_gains":list(GAIN),
          "original_positive":POSITIVE,"original_ordinary":ORDINARY,
          "unknown_excluded":UNKNOWN,"qualified_groups":QUALIFIED_GROUPS,
          "real_frozen_model_forward_conditions":len(results)*len(GAIN)*len(MODELS),
          "gains_are_derived_not_new_independent_source_samples":True,
          "no_qualifying_hard_negatives":True,
          "summary":summary,"per_original":results,
          "b7_original_exact_replay":"FAILED_UNCHANGED",
          "independent_synthetic_piper_only":True,
          "human_gold":False,"cal_locked_opened":False,
          "new_optimizer_updates":0,"new_tts_asr_phone_calls":0,
          "threshold_tuning_allowed":False,
          "model_selection_allowed":False,"product_promotion_allowed":False,
          "ssc305_physical":"NOT_RUN",
          "human_3_5m_frr":"NOT_MEASURED",
          "natural_far_per_hour":"NOT_MEASURED"}
        need(result["real_frozen_model_forward_conditions"]==192,
             "EXACT_192_REAL_FROZEN_NEW_DEV_FORWARD_CONDITIONS")
        save(output/"RESULT.json",result)
        print(json.dumps({"status":result["status"],"summary":summary,
            "real_model_forward_conditions":
               result["real_frozen_model_forward_conditions"],
            "product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        save(output/"STOP.json",{
             "schema":SCHEMA,"status":"STOP_B9_FROZEN_DEV_DIAGNOSTIC",
             "error_type":type(exc).__name__,
             "reason":str(exc)[:460],
             "cal_locked_opened":False,
             "model_selection_allowed":False,
             "product_promotion_allowed":False})
        raise

def main():
    p=argparse.ArgumentParser()
    m=p.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--guard","--audit","--execute"):
        m.add_argument(flag,action="store_true")
    for name in ("source","screen","b6-zip","b8-zip","b9-zip","output"):
        p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.source is not None and a.screen is not None,
         "EXACT_NEW_DEV_AUDIO_AND_FROZEN_SCREEN_REQUIRED")
    if a.audit:
        rows,receipt=select(a.source,a.screen)
        print(json.dumps({"status":"SOURCE_AUDIT_NO_MODEL_FORWARD",
              "eligible_originals":len(rows),"receipt":receipt,
              "new_optimizer_updates":0,"new_model_forwards":0},
              sort_keys=True));return
    need(a.b6_zip and a.b8_zip and a.b9_zip and a.output,
         "THREE_ORIGINAL_TRAINING_ARCHIVES_AND_OUTPUT_REQUIRED")
    execute(a.source,a.screen,a.b6_zip,a.b8_zip,a.b9_zip,a.output)

if __name__=="__main__":
    main()
