#!/usr/bin/env python3
"""B10 post-train: three real frozen FSMN-CTC weights on untouched DEV540–545.

The two same-job B10 arms are compared on identical eight machine-qualified
new synthetic DEV original PCM, with B9 frozen parent for historical context.
No gradients, threshold search, inferred human gold, CAL/LOCKED or promotion.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    s=importlib.util.spec_from_file_location("b10_frozen_"+name,HERE/(name+".py"))
    x=importlib.util.module_from_spec(s)
    s.loader.exec_module(x)
    return x

train=load("english_v6_b10_hardneg_paired_trainonce")
b9=train.b9
screen=train.screen
observer=screen.observer
c=screen.c
SCHEMA="english-v6-b10-fresh-dev540-three-frozen-paired-quality-v1"
REPO="jiying2007/kws-data"
BEFORE="cf908773f996ac3395b3841c7f219ef728abcc64"
SOURCE=train.ARTIFACTS["source"]
SCREEN=train.ARTIFACTS["screen"]
B9=train.ARTIFACTS["b9"]
TRAIN={"id":14740303244,"run":44987190871,"head":BEFORE,
       "name":"english-v6-b10-hardneg-real-paired-44987190871",
       "bytes":1000738,
       "sha":"3c7247e301aa57ef442b116ad28dc3356e80b88942cea687fcebb8af8270b33d"}
B10_PTS={
    "B10_control":"43e8dc8e1e4150421e55d44b4ad6c5813201947f1560408c0a47d83439608352",
    "B10_hard_negative":"9dc4d8696c16adde295f32b287653165e17fd3e0b650865679474d67616c4ae5"}
B9_PARENT_SHA=b9.MODELS["B9_control"]["pt"]
G=(1.0,0.75,0.5)
REVIEWED_ORIGINALS=18
ACCEPTED=8
POSITIVE=2
ORDINARY=6
UNKNOWN=4
PARAMS=43876
QUALIFIED_DEV_GROUPS={"piper-libritts-p7030","piper-libritts-p4257"}

def need(ok,reason):
    if not ok:raise ValueError(reason)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def read(path):
    return json.loads(c.read(Path(path)))
def write(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")

def source_check():
    prior=train.source_check()
    need(prior["status"]=="SOURCE_ONLY_NO_OPTIMIZER_OR_MODEL_FORWARD"
         and prior["max_optimizer_updates"]==384
         and prior["original_old_train_rows"]==79
         and train.BEFORE==SCREEN[2]
         and train.BEFORE=="cf3d995d830e69f875c7921ff0ac95498666ecf9",
         "B10_PRETRAINED_CONTRACT")
    need(SOURCE==(14708400220,44906200481,
                 "115738743e6c2cfa0cc55df9df772bb00e4c673c",
                 371843,"dbc57d762922e60ff23253af2463d5d64e1c8021c74fad6e2f0a9ecb9fd87465")
         and SCREEN==(14740257228,44986730086,
                 "cf3d995d830e69f875c7921ff0ac95498666ecf9",
                 3474,"ff4d647a224f430e50a76c1f7b42854ad9b41324db0f6b2f2823aaa4c63adb12")
         and B9[0]==14689218171 and B9_PARENT_SHA==
             "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"
         and len(B10_PTS)==2 and TRAIN["run"]==44987190871
         and (ACCEPTED,POSITIVE,ORDINARY,UNKNOWN)==(8,2,6,4)
         and len(QUALIFIED_DEV_GROUPS)==2
         and G==(1.0,0.75,0.5),
         "IMMUTABLE_B10_WEIGHTS_AND_EIGHT_NEW_DEV_ORIGINALS")
    wf=(ROOT/".github/workflows/english-v6-b10-frozen-dev540-once.yml").read_text()
    need("github.event.before == '"+BEFORE+"'" in wf
         and "'"+TRAIN["sha"]+"'" in wf,
         "EXACT_ONESHOT_FROZEN_DEV540_SOURCE_WORKFLOW")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_ZERO_FROZEN_MODEL_FORWARDS",
            "target":"Hey Mero","parent_pt_sha256":B9_PARENT_SHA,
            "b10_model_pt_sha256":B10_PTS,
            "run":TRAIN["run"],"model_archive_id":TRAIN["id"],
            "development_source_groups":6,"machine_positive":POSITIVE,
            "machine_ordinary":ORDINARY,"unknown_excluded":UNKNOWN,
            "fixed_gains":list(G),"planned_actual_model_forwards":ACCEPTED*len(G)*3,
            "new_optimizer_updates":0,"new_audio_asr_phone_calls":0,
            "pr_model_forwards":0,"threshold_sweep":False,
            "cal_locked_opened":False,"model_selection_allowed":False,
            "product_promotion_allowed":False}

def guard():
    source_check()
    e=os.environ
    need(all(e.get(k)==v for k,v in {
       "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
       "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
       "GITHUB_RUN_ATTEMPT":"1"}.items()),
       "FIRST_PROTECTED_MAIN_ONLY_NO_REPLAY")
    head=e.get("GITHUB_SHA","")
    need(re.fullmatch(r"[0-9a-f]{40}",head) is not None
         and re.fullmatch(r"[1-9][0-9]*",e.get("GITHUB_RUN_ID","")) is not None,
         "EXACT_GITHUB_MAIN_HEAD_AND_RUN_ID")
    event=read(e["GITHUB_EVENT_PATH"])
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("deleted") is False and event.get("forced") is False
         and subprocess.check_output(["git","rev-parse","HEAD^1"],cwd=ROOT,
               text=True).strip()==BEFORE,
         "EXACT_FIRST_SUCCESSOR_AFTER_B10_TRAIN_NO_REPLAY")
    return {"run_id":e["GITHUB_RUN_ID"],"head":head,"parent":BEFORE}

def provenance(token):
    need(type(token) is str and len(token)>10,"READ_ONLY_GITHUB_TOKEN_REQUIRED")
    def get(path):
        req=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+"/actions/"+path,
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"kws-v6-b10-real-frozen-dev-compare"})
        with urllib.request.urlopen(req,timeout=40) as r:
            data=r.read(1000001)
        need(len(data)<=1000000,"BOUNDED_B10_PROVENANCE")
        return json.loads(data)
    records=[
      (SOURCE[0],SOURCE[1],SOURCE[2],SOURCE[3],SOURCE[4]),
      (SCREEN[0],SCREEN[1],SCREEN[2],SCREEN[3],SCREEN[4]),
      (B9[0],B9[1],B9[2],B9[3],B9[4]),
      (TRAIN["id"],TRAIN["run"],TRAIN["head"],TRAIN["bytes"],TRAIN["sha"])]
    for aid,run_id,head,size,digest in records:
        run=get("runs/"+str(run_id))
        art=get("artifacts/"+str(aid))
        need(run.get("id")==run_id
             and run.get("head_sha")==head and run.get("head_branch")=="main"
             and run.get("event")=="push" and run.get("run_attempt")==1
             and run.get("status")=="completed" and run.get("conclusion")=="success"
             and art.get("id")==aid
             and art.get("size_in_bytes")==size
             and art.get("digest")=="sha256:"+digest
             and art.get("expired") is False
             and art.get("workflow_run",{}).get("id")==run_id
             and art.get("workflow_run",{}).get("head_sha")==head,
             "PROVEN_B10_SOURCE_AND_TRAIN_ARTIFACT_SHA_"+str(aid))
    return {"source":SOURCE[0],"screen":SCREEN[0],
            "b9":B9[0],"b10":TRAIN["id"]}

def select(source,old_train,old_screen,saved):
    originals=observer.source_rows(source,old_train,old_screen)
    need(len(originals)==REVIEWED_ORIGINALS,
         "ONLY_B10_EIGHTEEN_NEW_ORIGINALS")
    rec=read(Path(saved)/"SCREEN.json")
    need(rec.get("schema")==screen.SCHEMA
         and rec.get("status")=="PASS_B10_TRAIN_DEV_MACHINE_RESEARCH_GATE_ONLY"
         and rec.get("source_run")==SOURCE[1]
         and rec.get("observer_run")==44908125673
         and rec.get("complete_native_attempts")==90
         and rec.get("admission")==screen.admission(originals,rec["rows"])
         and rec["admission"]["train_hard_negative_qualified"]==6
         and rec["admission"]["development_qualified_groups"]==2
         and set(rec["admission"]["development_qualified_source_groups"])==
             QUALIFIED_DEV_GROUPS
         and rec["admission"]["B10_research_optimizer_allowed"] is True
         and rec["no_old_DEV520_528_training"] is True
         and rec["cal_locked_opened"] is False
         and rec["new_kws_model_forwards"]==0
         and rec["product_promotion_allowed"] is False,
         "B10_UNALTERED_FIVE_NATIVE_SOURCE_QUALITY_AND_NO_DEV_TRAINING")
    src={r["id"]:r for r in originals}
    decisions={r["id"]:r for r in rec["rows"]}
    need(len(src)==len(decisions)==REVIEWED_ORIGINALS
         and set(src)==set(decisions),
         "B10_SOURCE_ID_AND_FROZEN_SCREEN_EXACT_JOIN")
    qualified=[];unknown=[];train_ids=[]
    for sid,row in src.items():
        audit=decisions[sid]
        need(all(row[k]==audit[k] for k in
                 ("role","cell_id","source_group_id","speaker_index",
                  "pcm_sha256","wav_sha256","samples"))
             and row["human_gold"] is False
             and row["training_allowed"] is False
             and row["observed_class"] is None
             and audit["human_gold"] is False,
             "B10_NO_ORIGINAL_ROLE_LABEL_PCM_HASH_MUTATION")
        if row["role"]=="train":
            need(audit["cell_id"]=="ordinary-time-hard"
                 and audit["observed_class"]==0
                 and audit["tier"]=="silver",
                 "EXCLUDE_ALL_SIX_TRAIN_HARDNEG_FROM_DEV_SCORE")
            train_ids.append(sid)
        else:
            need(row["role"]=="development"
                 and audit["cell_id"] in ("mero-fast","ordinary-time"),
                 "EXACT_DEVELOPMENT_540_545_POSITIVE_ORDINARY_CELLS")
            if audit["tier"] in ("silver","weak"):
                need(audit["observed_class"]==audit["requested_class"]
                     and audit["observed_class"] in (0,2),
                     "ONLY_POSITIVE_AND_ORDINARY_INDEPENDENT_MACHINE_QA")
                qualified.append({**row,"source_pcm_sha256":row["pcm_sha256"],
                                  "positive":audit["observed_class"]==2,
                                  "tier":audit["tier"]})
            else:
                need(audit["tier"]=="unknown" and audit["observed_class"] is None,
                     "NO_UNQUALIFIED_SPEECH_USED_AS_DEV_DENOMINATOR")
                unknown.append(sid)
    need(len(qualified)==ACCEPTED and len(unknown)==UNKNOWN and len(train_ids)==6
         and Counter(r["positive"] for r in qualified)=={True:POSITIVE,False:ORDINARY}
         and {r["source_group_id"] for r in qualified if r["positive"]}==
             QUALIFIED_DEV_GROUPS
         and {r["speaker_index"] for r in qualified if r["positive"]}=={542,544}
         and {r["speaker_index"] for r in qualified if not r["positive"]}==
             set(range(540,546))
         and not set(train_ids) & {r["id"] for r in qualified},
         "B10_FRESH_EIGHT_APPROVED_DEV_ONLY_TWO_POSITIVE_SIX_ORDINARY")
    return sorted(qualified,key=lambda x:x["id"]),{
        "source_originals":18,"train_ids_excluded":sorted(train_ids),
        "dev_unknown_excluded":sorted(unknown),
        "qualified_source_groups":sorted(QUALIFIED_DEV_GROUPS),
        "machine_positive":POSITIVE,"ordinary":ORDINARY,
        "human_gold":False,"dev_training":False}

def verify_models(zip_file,torch,np,bank,mean,scale):
    path=Path(zip_file)
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size==TRAIN["bytes"]
         and sha(path.read_bytes())==TRAIN["sha"],
         "ACTUAL_B10_TRAIN_ZIP_BYTE_SHA_AND_SIZE")
    output={}
    with zipfile.ZipFile(path) as arc:
        need(arc.testzip() is None,"FULL_B10_PT_NPZ_ONNX_ZIP_CRC")
        result=json.loads(arc.read("RESULT.json"))
        source=json.loads(arc.read("SOURCE.json"))
        need(result.get("schema")==train.SCHEMA
             and result.get("status")==
               "RESEARCH_B10_REAL_SILVER_TRAIN_HARDNEG_PAIRED_WEIGHTS_NO_PROMOTION"
             and result.get("actual_optimizer_updates")==384
             and result.get("trained_model_parameters")==PARAMS
             and result.get("old_train_originals")==79
             and result.get("new_silver_hard_negatives")==6
             and result.get("new_train_group_count")==6
             and result.get("control_new_loss_weight")==0.0
             and result.get("challenger_new_loss_weight")==0.08
             and result.get("frozen_b9_parent_pt_sha256")==B9_PARENT_SHA
             and result.get("different_trained_float_values")==43751
             and result.get("dev_gradients")==0
             and result.get("dev_model_forwards")==0
             and result.get("cal_locked_opened") is False
             and result.get("model_selection_allowed") is False
             and result.get("product_promotion_allowed") is False
             and result["control"]["optimizer_updates"]==
                 result["challenger"]["optimizer_updates"]==192
             and source.get("parent_pt_sha256")==B9_PARENT_SHA
             and source.get("old_train_originals")==79
             and source.get("dev_pcm_in_gradients") is False
             and source.get("dev_model_forwards")==0,
             "REAL_B10_SINGLE_FACTOR_TWO_TRAINED_ARMS_NO_DEV_LEAK")
        for label,arm_key,folder,stem in (
            ("B10_control","control","matched_control","mero43-b10-matched-control192"),
            ("B10_hard_negative","challenger","new_hardneg","mero43-b10-train-hardneg192")):
            files=result[arm_key]["files"]
            blobs={}
            for ext in ("pt","npz","onnx"):
                name=stem+"."+ext
                raw=arc.read(folder+"/models/"+name)
                need(len(raw)>10000
                     and sha(raw)==files[name]["sha256"]
                     and len(raw)==files[name]["bytes"],
                     "EXACT_SAVED_B10_"+label+"_"+ext.upper())
                blobs[ext]=raw
            need(sha(blobs["pt"])==B10_PTS[label],
                 "B10_ORIGINAL_LEARNED_PT_SHA_"+label)
            weights=torch.load(io.BytesIO(blobs["pt"]),map_location="cpu",
                               weights_only=True)
            net=b9.v5.build_model(torch)
            net.load_state_dict(weights,strict=True)
            net.eval()
            need(sum(x.numel() for x in net.parameters())==PARAMS,
                 "EXACT_43K_B10_FSMN_GEOMETRY")
            with np.load(io.BytesIO(blobs["npz"]),allow_pickle=False) as ar:
                need(set(ar.files)==set(weights)|{
                         "mel_filters","train_mean","train_scale"}
                     and all(np.array_equal(ar[k],v.detach().cpu().numpy())
                             for k,v in weights.items())
                     and np.array_equal(ar["mel_filters"],bank)
                     and np.array_equal(ar["train_mean"],mean)
                     and np.array_equal(ar["train_scale"],scale),
                     "SAVED_B10_MODEL_NPZ_PT_AND_B9_FRONTEND_PARITY")
            output[label]=(net,bank,mean,scale)
    need(set(output)==set(B10_PTS),"EXACT_TWO_SAME_JOB_TRAINED_CANDIDATES")
    return output

def execute(source,old_train,old_screen,saved,b9zip,b10zip,output):
    import numpy as np
    import torch
    identity=guard()
    output=Path(output)
    need(output.is_absolute() and output.parent.is_dir()
         and not output.exists(),"ONE_NEW_FROZEN_DEV_SCORE_OUTPUT_ONLY")
    output.mkdir()
    write(output/"STARTED.json",{
        "schema":SCHEMA,"run":identity,"new_kws_forward_conditions_max":72,
        "new_optimizer_updates":0,"new_audio_asr_phone_calls":0,
        "cal_locked_opened":False,"model_selection_allowed":False,
        "product_promotion_allowed":False})
    try:
        upstream=provenance(os.environ.get("GH_TOKEN"))
        rows,audit=select(source,old_train,old_screen,saved)
        need(len(rows)==ACCEPTED,"ONLY_EIGHT_NEW_SOURCE_QUALIFIED_DEV_WAV")
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        parent,source_receipt=b9.verify_b9_archive(b9zip,torch,np)
        baseline=parent["B9_control"]
        bank,mean,scale=baseline[1:]
        newmodels=verify_models(b10zip,torch,np,bank,mean,scale)
        models={"B9_parent":baseline,**newmodels}
        need(len(models)==3 and set(models)=={
              "B9_parent","B10_control","B10_hard_negative"},
             "THREE_BYTE_PINNED_FROZEN_PARENT_AND_B10_ARMS")
        actual=[];saved_logits={}
        with torch.no_grad():
            for row in rows:
                pcm=b9.ref.source_pcm(row,source)
                cases={}
                for gain in G:
                    audio=b9.ref.gain_pcm(pcm,gain,np)
                    features=b9.v5.features(audio,row,torch,np,bank)
                    need(features.ndim==2 and features.shape[1]==40
                         and np.isfinite(features).all(),
                         "FROZEN_B10_VALID_LOGMEL40")
                    inp=torch.from_numpy(((features-mean)/scale)
                         .astype(np.float32)).unsqueeze(0)
                    values={}
                    for label,(net,*_) in models.items():
                        logits=net(inp)
                        need(logits.shape[0]==1 and logits.shape[-1]==4
                             and bool(torch.isfinite(logits).all()),
                             "FINITE_43876_FSMN_CTC_LOGITS")
                        decoded=b9.v5.ctc_greedy(logits,torch)
                        count=int(decoded["mero_event"])+int(decoded["extra_events"])
                        values[label]={
                          "complete_events":count,"hit":bool(count),
                          "collapsed_tokens":decoded["collapsed_tokens"],
                          "frames":int(logits.shape[1]),
                          "first_frame_argmax":int(logits.argmax(-1)[0,0]),
                          "extra_events":int(decoded["extra_events"])}
                        if gain==1.0:
                            saved_logits[row["id"]+"__"+label]=(
                                logits.squeeze(0).cpu().numpy().astype(np.float32))
                    cases[str(gain)]=values
                actual.append({
                    "id":row["id"],"source_group_id":row["source_group_id"],
                    "source_pcm_sha256":row["source_pcm_sha256"],
                    "role":"development","machine_tier":row["tier"],
                    "positive":row["positive"],"gain_models":cases})
        need(len(actual)==ACCEPTED
             and Counter(x["positive"] for x in actual)=={
                 True:POSITIVE,False:ORDINARY},
             "ALL_NEW_8_DEV_ORIGINALS_SCORED_WITHOUT_UNKNOWN")
        logit_file=output/"ORIGINAL_FRAME_LOGITS.npz"
        np.savez_compressed(logit_file,**saved_logits)
        need(len(saved_logits)==ACCEPTED*3 and logit_file.stat().st_size>1000,
             "SAVE_EXACT_24_ORIGINAL_GAIN_LOGIT_ARRAYS")
        summary={}
        for gain in G:
            key=str(gain)
            pos=[x for x in actual if x["positive"]]
            neg=[x for x in actual if not x["positive"]]
            summary[key]={}
            for model in models:
                summary[key][model]={
                   "positive_originals":POSITIVE,
                   "positive_hit_clips":sum(x["gain_models"][key][model]["hit"]
                                              for x in pos),
                   "ordinary_originals":ORDINARY,
                   "ordinary_false_event_clips":sum(
                         x["gain_models"][key][model]["hit"] for x in neg),
                   "ordinary_false_or_extra_events":sum(
                         x["gain_models"][key][model]["complete_events"] for x in neg)}
        doc={"schema":SCHEMA,
             "status":"REAL_B10_POSTTRAIN_FROZEN_NEW_DEV_RESEARCH_DIAGNOSTIC_ONLY",
             "run":identity,"immutable_evidence":upstream,
             "source_receipt":audit,
             "model_pt_sha256":{"B9_parent":B9_PARENT_SHA,**B10_PTS},
             "original_frame_logits_npz_sha256":sha(logit_file.read_bytes()),
             "originals":ACCEPTED,"positive_originals":POSITIVE,
             "ordinary_originals":ORDINARY,
             "unknown_excluded":UNKNOWN,
             "machine_qualified_dev_groups":len(QUALIFIED_DEV_GROUPS),
             "fixed_gains":list(G),
             "gains_are_not_new_independent_source_samples":True,
             "real_frozen_model_forward_conditions":len(actual)*len(G)*len(models),
             "summary":summary,"per_original":actual,
             "new_optimizer_updates":0,"new_tts_asr_phone_calls":0,
             "no_old_DEV520_528_training_or_evaluation":True,
             "DEV540_545_no_gradients":True,
             "human_gold":False,"cal_locked_opened":False,
             "threshold_tuning_allowed":False,
             "model_selection_allowed":False,"product_promotion_allowed":False,
             "physical_ssc305":"NOT_RUN","human_3_5m_frr":"NOT_MEASURED",
             "natural_far_per_hour":"NOT_MEASURED",
             "B0_only_installed_C11_SDK_baseline":True}
        need(doc["real_frozen_model_forward_conditions"]==72,
             "EXACT_B10_8_ORIGINAL_3_GAIN_3_MODEL_72_NUMERIC_FORWARDS")
        write(output/"RESULT.json",doc)
        print(json.dumps({"status":doc["status"],"real_forwards":72,
               "summary":summary,"product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        write(output/"STOP.json",{
            "schema":SCHEMA,"status":"STOP_B10_FROZEN_NEW_DEV_SCORE",
            "reason":str(exc)[:500],"error_type":type(exc).__name__,
            "new_optimizer_updates":0,"cal_locked_opened":False,
            "product_promotion_allowed":False})
        raise

def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--guard","--audit","--score"):
        mode.add_argument(flag,action="store_true")
    for name in ("source","train-source","train-screen","screen",
                 "b9-zip","b10-zip","output"):
        parser.add_argument("--"+name,type=Path)
    a=parser.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.source is not None and a.train_source is not None
         and a.train_screen is not None and a.screen is not None,
         "EXACT_B10_SOURCE_TRAIN_ROLE_AND_SAVED_SCREEN_REQUIRED")
    if a.audit:
        rows,proof=select(a.source,a.train_source,a.train_screen,a.screen)
        print(json.dumps({"status":"B10_NEW_DEV_SOURCE_ONLY_AUDIT",
            "qualified_originals":len(rows),"machine_positive":POSITIVE,
            "ordinary":ORDINARY,"unknown_excluded":UNKNOWN,
            "receipt":proof,"new_model_forwards":0},sort_keys=True));return
    need(a.b9_zip is not None and a.b10_zip is not None
         and a.output is not None,"B9_B10_ORIGINAL_FROZEN_ZIPS_REQUIRED")
    execute(a.source,a.train_source,a.train_screen,
            a.screen,a.b9_zip,a.b10_zip,a.output)

if __name__=="__main__":
    main()
