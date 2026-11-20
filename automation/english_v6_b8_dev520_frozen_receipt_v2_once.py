#!/usr/bin/env python3
"""B8 genuine frozen-model paired acoustic comparison on new independently reviewed DEV520.

3 machine-qualified positives and 8 ordinary originals across 3 new
source-disjoint synthetic DEV groups; 21 UNKNOWN excluded. Fixed gains
1.0/0.75/0.5 share each original source. No new training, TTS, ASR,
CAL/LOCKED or threshold/model selection. Record genuine frame logits.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
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
import wave
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("b4_real_qual_"+name,HERE/(name+".py"))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

saved=load("english_v6_b8_dev520_saved_screen_once")
upstream=load("english_v5_mero_prospective_three_frozen_once")
v5=upstream.v5
SCHEMA="english-v6-b8-dev520-three-frozen-receipt-aligned-v2"
REPO="jiying2007/kws-data"
BEFORE="6749c028de03f6554058acc43d60fda7120dff9f"
SOURCE={"id":14634696242,"run":44748801347,
        "head":"f606f35a4b3232cbfcec3bd2a14e8939551169fd",
        "name":"english-v6-b8-fresh-dev520-original-44748801347",
        "bytes":628555,
        "sha":"1045f91d272e8f199d73dee6b39067bdbb9238dc3d4f05f74e95af54aa0c4700"}
SCREEN={"id":14641941031,"run":44761801165,
        "head":"858e7550362bd92b51642934569071670fb6c5df",
        "name":"english-v6-b8-dev520-saved-machine-screen-44761801165",
        "bytes":5545,
        "sha":"3ca1e59c11f3a74c42eda7d15985243922097debcd8013abf14afb4fd75c3b7a"}
ARCHIVES={
  "B6":{"id":14615025178,"run":44704535156,
        "head":"e46dfa72e7f824a888ae6510a9951f1fe518b222",
        "name":"english-v6-b6-replay-vs-gain-consistency-44704535156",
        "bytes":1000729,
        "sha":"1c310b0d4945f7aa40f2576f231c94eb48b33316a6c4f79c20215b1ac80bab8d"},
  "B8":{"id":14634066012,"run":44746571932,
        "head":"453b26bdd4ab2bdc96b27fe0dfb9ac10bdd38711",
        "name":"english-v6-b8-frozen-b6-paired-ctc085-44746571932",
        "bytes":1001683,
        "sha":"6015c6269f87654193d7350afd0c5b44f799f5f71b5303e41f2734c17006d15b"}
}
MODELS={
  "B6_consistency":{"archive":"B6",
      "path":"models/mero43-b6-gain-consistency192",
      "pt":"0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04"},
  "B8_control":{"archive":"B8","arm":"control",
      "path":"control_ordinary060/models/mero43-b6-gain-consistency192",
      "pt":"a5cc0935c51afa1f82faed48cd50286c5bb5a9d136b0b4f752c176beb76df1e1"},
  "B8_challenger":{"archive":"B8","arm":"challenger",
      "path":"challenger_ordinary085/models/mero43-b7-ordinary-ctc085-192",
      "pt":"cc16b81f84e4f21c5347ca153f405ac41ed116e3eedfaeed5677d1a4bc86e22f"}
}
GAINS=(1.0,0.75,0.5)
SOURCE_ORIGINALS=32
POSITIVE=3
ORDINARY=8
UNKNOWN=21
SOURCE_GROUPS=8
QUALIFIED_GROUPS=3
PARAMETERS=43876
TRAINED_ARM_DIFFERENCES=43755

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")

def source_check():
    prev=saved.source_check()
    need(prev["status"]=="SAVED_PROSPECTIVE_DEV8_160_REAL_REVIEWS_PINNED"
         and prev["saved_source_rows"]==SOURCE_ORIGINALS
         and prev["original_observer_run"]==44750685991
         and prev["new_kws_model_forwards"]==0
         and saved.ORIGINAL_SHA=="e6e99c83858e9032e42dff43ff1b1cdf03466a19"
         and saved.BEFORE=="e6e99c83858e9032e42dff43ff1b1cdf03466a19"
         and saved.SOURCE_RUN==SOURCE["run"]
         and saved.SOURCE["id"]==SOURCE["id"]
         and saved.SOURCE["digest"]=="sha256:"+SOURCE["sha"]
         and SCREEN["head"]=="858e7550362bd92b51642934569071670fb6c5df"
         and len(MODELS)==3 and set(ARCHIVES)=={"B6","B8"}
         and GAINS==(1.0,0.75,0.5)
         and (POSITIVE,ORDINARY,UNKNOWN,QUALIFIED_GROUPS)==(3,8,21,3)
         and TRAINED_ARM_DIFFERENCES==43755
         and MODELS["B8_control"]["pt"].startswith("a5cc0935")
         and MODELS["B8_challenger"]["pt"].startswith("cc16b81f"),
         "EXACT_B8_NEW_SOURCE_FIVE_NATIVE_AND_THREE_FROZEN_MODELS")
    wf=(ROOT/".github/workflows/english-v6-b8-dev520-frozen-receipt-v2-once.yml").read_text()
    need(all(wf.count("'"+ARCHIVES[k]["sha"]+"'")==1
             for k in ("B6","B8"))
         and "github.event.before == '"+BEFORE+"'" in wf
         and wf.count("'"+PRIOR_FAILED_ZIP_SHA+"'")==1,
         "EXACT_WORKFLOW_MODEL_ZIPS_AND_FIRST_MAIN_SUCCESSOR")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_NEURAL_FORWARD",
            "keyword":"Hey Mero",
            "models":{k:{"artifact":ARCHIVES[v["archive"]]["id"],
                          "pt_sha256":v["pt"]}
                      for k,v in MODELS.items()},
            "new_source_run":SOURCE["run"],"saved_machine_screen_run":SCREEN["run"],
            "reviewed_originals":SOURCE_ORIGINALS,
            "machine_qualified_positive":POSITIVE,
            "machine_qualified_ordinary":ORDINARY,
            "excluded_unknown":UNKNOWN,
            "qualified_development_groups":QUALIFIED_GROUPS,
            "frozen_gains":list(GAINS),
            "max_posttrain_forward_conditions":(POSITIVE+ORDINARY)*len(GAINS)*len(MODELS),
            "new_model_forwards_in_PR":0,"new_optimizer_updates":0,
            "new_tts_asr_calls":0,"cal_locked_audio_access":False,
            "no_old_exposed_dev_reuse":True,"independent_human_gold":False,
            "model_selection_allowed":False,"product_promotion_allowed":False}

def guard():
    source_check()
    e=os.environ
    need(all(e.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"FIRST_PROTECTED_PUBLIC_MAIN_ONLY")
    head=e.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",e.get("GITHUB_RUN_ID","")) is not None,
         "REAL_RUN_AND_SHA")
    event=json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False
         and subprocess.check_output(["git","rev-parse","HEAD^1"],
                         cwd=ROOT,text=True).strip()==BEFORE,
         "NO_OUT_OF_ORDER_OR_REPLAY")
    return {"run_id":e["GITHUB_RUN_ID"],"source_commit":head,"parent":BEFORE}

def provenance(token):
    need(isinstance(token,str) and len(token)>10,
         "GITHUB_READ_ONLY_TOKEN_REQUIRED")
    root="https://api.github.com/repos/"+REPO+"/actions/"
    def get(suffix):
        req=urllib.request.Request(root+suffix,headers={
             "Authorization":"Bearer "+token,
             "Accept":"application/vnd.github+json",
             "User-Agent":"kws-v6-b8-frozen-dev520-paired-models"})
        with urllib.request.urlopen(req,timeout=35) as res:
            raw=res.read(1048577)
        need(len(raw)<=1048576,"BOUNDED_GITHUB_PROVENANCE")
        return json.loads(raw)
    for record in (SOURCE,SCREEN,*ARCHIVES.values()):
        run=get("runs/"+str(record["run"]))
        need(run.get("id")==record["run"]
             and run.get("head_sha")==record["head"]
             and run.get("run_attempt")==1
             and run.get("event")=="push"
             and run.get("status")=="completed"
             and run.get("conclusion")=="success",
             "ACTUAL_PROTECTED_MAIN_SOURCE_MODEL_TRAIN_OR_SCREEN_RUN")
        art=get("artifacts/"+str(record["id"]))
        need(art.get("id")==record["id"]
             and art.get("name")==record["name"]
             and art.get("size_in_bytes")==record["bytes"]
             and art.get("digest")=="sha256:"+record["sha"]
             and art.get("expired") is False
             and art.get("workflow_run",{}).get("id")==record["run"]
             and art.get("workflow_run",{}).get("head_sha")==record["head"],
             "MODEL_OR_QUALIFIED_AUDIO_ARTIFACT_SHA")
    return {"source":SOURCE["id"],"screen":SCREEN["id"],
            "models":{k:v["id"] for k,v in ARCHIVES.items()}}

def select(source,screen):
    originals=saved.source_rows(source)
    doc=json.loads((Path(screen)/"SCREEN.json").read_bytes())
    need(doc.get("schema")==saved.SCHEMA
         and doc.get("status")=="PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY"
         and doc.get("source_run")==SOURCE["run"]
         and doc.get("original_observer_run")==44750685991
         and doc.get("source_group_counts")=={"development":QUALIFIED_GROUPS}
         and doc.get("tier_counts")=={"silver":9,"weak":2,"unknown":UNKNOWN}
         and doc.get("observed_label_counts")=={"0":ORDINARY,"2":POSITIVE}
         and doc.get("full_6_2_2_2_gate_passed") is False
         and doc.get("nearphone_as_ordinary_negative") is False
         and doc.get("machine_only_no_human_gold") is True
         and doc.get("new_audio_or_model_forwards")==0
         and doc.get("new_optimizer_updates")==0
         and doc.get("new_kws_model_forwards")==0
         and doc.get("training_snapshot_allowed") is False
         and doc.get("locked_kws_test_opened") is False
         and doc.get("product_promotion") is False,
         "FROZEN_INDEPENDENT_B8_MACHINE_SOURCE_QA_ONLY")
    original={r["id"]:r for r in originals}
    seen=set();grouped=defaultdict(list);accepted=[]
    for row in doc.get("rows",[]):
        sid=row.get("id")
        need(sid in original and sid not in seen
             and row.get("role")=="development"
             and row.get("human_gold") is False
             and row.get("training_allowed") is False,
             "NEVER_DUPLICATE_OR_REUSE_NONDEV")
        seen.add(sid)
        source_row=original[sid]
        need(all(row.get(k)==source_row.get(k)
                 for k in ("source_group_id","speaker_index","role","cell_id",
                           "requested_class","pcm_sha256","wav_sha256","samples")),
             "BYTE_IDENTITY_ASR_SOURCE_BINDING")
        grouped[row["source_group_id"]].append(row)
        if source_row["cell_id"]=="nearphone-meero-challenge":
            need(row["tier"]=="unknown" and row["observed_class"] is None,
                 "NEARPHONE_MUST_REMAIN_UNKNOWN")
        if row.get("tier") in ("silver","weak"):
            observed=row["observed_class"]
            need(observed in (0,2) and observed==row["requested_class"]
                 and source_row["cell_id"]!="nearphone-meero-challenge",
                 "ONLY_MACHINE_QUALIFIED_CANONICAL_OR_ORDINARY")
            accepted.append({**source_row,
                 "source_pcm_sha256":source_row["pcm_sha256"],
                 "mono_class":int(observed==2),
                 "machine_tier":row["tier"]})
        else:
            need(row.get("tier")=="unknown" and row.get("observed_class") is None,
                 "NO_UNQUALIFIED_MODEL_FORWARD")
    need(len(seen)==len(original)==SOURCE_ORIGINALS
         and len(grouped)==SOURCE_GROUPS
         and all(len(v)==4 for v in grouped.values()),
         "FULL_32_ORIGINAL_8_GROUP_DEVELOPMENT_COHORT")
    candidate_groups={r["source_group_id"] for r in accepted
                      if r["mono_class"]==1}
    fully_qualified={k for k,group in grouped.items()
                     if {0,2}<={r["observed_class"] for r in group
                                  if r["tier"] in ("silver","weak")}}
    need(candidate_groups==fully_qualified
         and len(fully_qualified)==QUALIFIED_GROUPS
         and set(doc.get("source_group_identity_members",[]))==fully_qualified
         and Counter(r["mono_class"] for r in accepted)=={0:ORDINARY,1:POSITIVE},
         "EXACT_THREE_POSITIVE_EIGHT_ORDINARY_NO_UNKNOWN")
    return sorted(accepted,key=lambda x:x["id"]),{
       "status":"AUDITED_THREE_NEW_POSITIVE_EIGHT_ORDINARY",
       "source_groups":SOURCE_GROUPS,
       "qualified_positive_source_groups":sorted(fully_qualified),
       "new_originals":len(accepted),
       "human_gold":False,"no_cal_locked":True}

def source_pcm(row,root):
    path=Path(root)/"wav"/(row["id"]+".wav")
    need(path.is_file() and not path.is_symlink()
         and sha(path.read_bytes())==row["wav_sha256"],
         "WHOLE_ORIGINAL_WAV_SHA")
    with wave.open(str(path),"rb") as wav:
        need(wav.getnchannels()==1 and wav.getsampwidth()==2
             and wav.getframerate()==16000 and wav.getcomptype()=="NONE"
             and wav.getnframes()==row["samples"],"ORIGINAL_COMPLETE_16K_PCM")
        data=wav.readframes(wav.getnframes())
    need(sha(data)==row["source_pcm_sha256"]
         and len(data)==row["samples"]*2,
         "ORIGINAL_PCM_SHA_NO_CROP")
    return data

def verify_model_archive(archive_path,kind,torch,np):
    archive=ARCHIVES[kind]
    need(Path(archive_path).is_file() and not Path(archive_path).is_symlink()
         and Path(archive_path).stat().st_size==archive["bytes"]
         and sha(Path(archive_path).read_bytes())==archive["sha"],
         "IMMUTABLE_REAL_TRAINED_"+kind+"_ZIP")
    loaded={}
    with zipfile.ZipFile(archive_path) as z:
        need(z.testzip() is None,"ALL_ORIGINAL_MODEL_MEMBERS_CRC")
        result=json.loads(z.read("RESULT.json"))
        if kind=="B6":
            need(result.get("status")=="RESEARCH_B6_TRAIN_ONLY_GAIN_POSTERIOR_KL_NO_PROMOTION"
                 and result.get("schema")=="english-v6-b6-matched-trainonly-gain-posterior-consistency-v1"
                 and result.get("actual_optimizer_updates")==384
                 and result.get("train_source_groups")==35
                 and result.get("model_parameters")==PARAMETERS
                 and result.get("dev_model_forwards")==0
                 and result.get("cal_locked_opened") is False
                 and result.get("product_promotion_allowed") is False,
                 "ACTUAL_FROZEN_B6_PARENT_TRAINED_RESULT")
            arm={r["arm"]:r for r in result["arms"]}["consistency"]
            sources=json.loads(z.read("SOURCE.json"))
            need(sources.get("dev_pcm_in_optimizer") is False,
                 "B6_PARENT_ONLY_TRAIN_PCM")
        else:
            need(result.get("status")=="RESEARCH_B8_TRAINED_PAIRED_FROZEN_B6_PARENT_NO_PROMOTION"
                 and result.get("schema")=="english-v6-b8-frozen-b6-parent-paired-ordinary-ctc-v1"
                 and result.get("actual_optimizer_updates")==384
                 and result.get("same_job_two_arms") is True
                 and result.get("same_source_groups")==35
                 and result.get("model_parameters") is None
                 and result.get("same_parent_pt_sha256")==MODELS["B6_consistency"]["pt"]
                 and result.get("B7_historical_exact_replay")=="FAILED_UNCHANGED"
                 and result.get("only_different_optimization_factor")=="ORDINARY_CTC_060_VS_085"
                 and result.get("trained_tensor_values_different")==TRAINED_ARM_DIFFERENCES
                 and result.get("dev_model_forwards")==0
                 and result.get("cal_locked_opened") is False
                 and result.get("model_selection_allowed") is False
                 and result.get("product_promotion_allowed") is False
                 and result["control"]["optimizer_updates"]==192
                 and result["challenger"]["optimizer_updates"]==192,
                 "ACTUAL_B8_TWO_ARMS_384_TRAIN_UPDATES")
            sources=json.loads(z.read("SOURCE.json"))
            need(sources.get("dev_pcm_in_gradients") is False,
                 "B8_ONLY_AUTHENTICATED_TRAIN_PCM")
        for label,spec in MODELS.items():
            if spec["archive"]!=kind:continue
            if kind=="B6":
                arm={r["arm"]:r for r in result["arms"]}["consistency"]
            else:
                arm=result[spec["arm"]]
            basename=spec["path"].rsplit("/",1)[-1]
            chunks={}
            for ext in ("pt","npz","onnx"):
                key=basename+"."+ext
                raw=z.read(spec["path"]+"."+ext)
                need(len(raw)>10000,"NONEMPTY_FROZEN_"+label+"_"+ext)
                need(sha(raw)==arm["files"][key]["sha256"]
                     and len(raw)==arm["files"][key]["bytes"],
                     "ORIGINAL_TRAIN_SAVED_"+label+"_"+ext)
                chunks[ext]=raw
            need(sha(chunks["pt"])==spec["pt"],"EXACT_FROZEN_PT_SHA_"+label)
            weights=torch.load(io.BytesIO(chunks["pt"]),map_location="cpu",weights_only=True)
            net=v5.build_model(torch)
            net.load_state_dict(weights,strict=True);net.eval()
            with np.load(io.BytesIO(chunks["npz"]),allow_pickle=False) as arrays:
                need(set(arrays.files)==set(weights)|{"mel_filters","train_mean","train_scale"},
                     "EXACT_MODEL_ARRAY_KEYS")
                need(all(np.array_equal(arrays[k],v.detach().cpu().numpy())
                         for k,v in weights.items()),"PT_NPZ_TENSOR_PARITY_"+label)
                bank=arrays["mel_filters"].copy()
                mean=arrays["train_mean"].copy()
                scale=arrays["train_scale"].copy()
            need(sum(p.numel() for p in net.parameters())==PARAMETERS
                 and bank.shape==(257,40) and mean.shape==(40,)
                 and scale.shape==(40,) and np.isfinite(bank).all()
                 and np.isfinite(mean).all() and np.isfinite(scale).all()
                 and bool((scale>0).all()),
                 "FROZEN_43K_FOUR_TOKEN_MODEL_"+label)
            loaded[label]=(net,bank,mean,scale)
    return loaded

PRIOR_FAILED_RUN=44764450043
PRIOR_FAILED_HEAD="6749c028de03f6554058acc43d60fda7120dff9f"
PRIOR_FAILED_ID=14644170620
PRIOR_FAILED_BYTES=708
PRIOR_FAILED_ZIP_SHA="ec66066712b07d8be41cdf1f53764cfbf7bd8a4b776431120f04f13fa712e053"
PRIOR_FAILED_JOB=136385640635

def verify_previous_failed_archive(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size==PRIOR_FAILED_BYTES
         and sha(path.read_bytes())==PRIOR_FAILED_ZIP_SHA,
         "EXACT_FAILED_B8_FIRST_MAIN_ORIGINAL_ZIP_SHA")
    with zipfile.ZipFile(path) as arc:
        need(arc.testzip() is None and "STOP.json" in arc.namelist()
             and "RESULT.json" not in arc.namelist()
             and "ORIGINAL_FRAME_LOGITS.npz" not in arc.namelist(),
             "NO_PREVIOUS_VALID_B8_MODEL_SCORES")
        stop=json.loads(arc.read("STOP.json"))
    need(stop.get("status")=="STOP_B8_FROZEN_DEV_SCORE"
         and stop.get("type")=="ValueError"
         and stop.get("reason")=="ACTUAL_B8_TWO_ARMS_384_TRAIN_UPDATES"
         and stop.get("cal_locked_opened") is False
         and stop.get("product_promotion_allowed") is False,
         "UNALTERED_PRIOR_METADATA_ONLY_B8_STOP")
    return {"prior_failed_run":PRIOR_FAILED_RUN,
            "prior_failed_artifact":PRIOR_FAILED_ID,
            "no_previous_model_scores":True}

def gain_pcm(pcm,gain,np):
    need(gain in GAINS and len(pcm)%2==0
         and isinstance(pcm,bytes),"ONLY_PREDECLARED_GAIN_AND_PCM16")
    if gain==1.0:return pcm
    x=np.frombuffer(pcm,dtype="<i2").astype(np.float32)*gain
    return np.rint(x).clip(-32768,32767).astype("<i2").tobytes()

def execute(source,screen,b6_zip,b8_zip,prior_failed_zip,output):
    import numpy as np
    import torch
    identity=guard()
    prior_stop=verify_previous_failed_archive(prior_failed_zip)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "ONE_NEW_B8_FROZEN_DEV_SCORE_OUTPUT")
    output.mkdir()
    save(output/"STARTED.json",{"schema":SCHEMA,"run":identity,
         "models":list(MODELS),"maximum_model_forward_conditions":99,
         "prior_failed_first_main_attempt":prior_stop,
         "new_tts_asr_calls":0,"new_optimizer_steps":0,
         "cal_locked_opened":False,"auto_model_selection":False})
    try:
        artifacts=provenance(os.environ.get("GH_TOKEN"))
        rows,audit=select(source,screen)
        need(len(rows)==POSITIVE+ORDINARY,
             "ONLY_11_MACHINE_ACCEPTED_NEW_DEV_ORIGINAL_PCM")
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        models={}
        models.update(verify_model_archive(b6_zip,"B6",torch,np))
        models.update(verify_model_archive(b8_zip,"B8",torch,np))
        need(set(models)==set(MODELS),"ALL_THREE_ACTUALLY_TRAINED_MODELS")
        # Original TRAIN provenance must match exactly across the two archives.
        with zipfile.ZipFile(b6_zip) as z6,zipfile.ZipFile(b8_zip) as z8:
            original=json.loads(z6.read("SOURCE.json"))
            later=json.loads(z8.read("SOURCE.json"))
        need(original.get("train_shas")==later.get("train_pcm_sha256")
             and original.get("dev_pcm_in_optimizer") is False
             and later.get("dev_pcm_in_gradients") is False,
             "EXACT_B6_B8_TRAIN_SOURCE_SHA_IDENTITY")
        bank,mean,scale=models["B6_consistency"][1:]
        for name,(_,b,m,s) in models.items():
            need(np.array_equal(bank,b) and np.array_equal(mean,m)
                 and np.array_equal(scale,s),
                 "SAME_FROZEN_40MEL_FRONTEND_"+name)
        parent=models["B6_consistency"][0].state_dict()
        control=models["B8_control"][0].state_dict()
        challenge=models["B8_challenger"][0].state_dict()
        need(set(parent)==set(control)==set(challenge)
             and sum(int(torch.count_nonzero(control[k]!=challenge[k]))
                     for k in parent)==TRAINED_ARM_DIFFERENCES,
             "SAME_GEOMETRY_ACTUAL_B8_43755_DIFFERENT_FLOATS")
        report=[];logits_saved={}
        with torch.no_grad():
            for row in rows:
                raw=source_pcm(row,source)
                conditions={}
                for gain in GAINS:
                    xpcm=gain_pcm(raw,gain,np)
                    features=v5.features(xpcm,row,torch,np,bank)
                    need(features.ndim==2 and features.shape[1]==40
                         and np.isfinite(features).all(),
                         "SAME_ORIGINAL_40MEL_FRAMES")
                    normalized=torch.from_numpy(
                        ((features-mean)/scale).astype(np.float32)).unsqueeze(0)
                    frame_results={}
                    for name,(net,*_) in models.items():
                        logits=net(normalized)
                        need(logits.shape[-1]==4
                             and bool(torch.isfinite(logits).all()),
                             "FOUR_TOKEN_FROZEN_CTC_FRAMES")
                        tokens=logits.argmax(-1).reshape(-1).tolist()
                        event=v5.ctc_greedy(logits,torch)
                        count=int(event["mero_event"])+int(event["extra_events"])
                        frame_results[name]={
                            "full_phrase_event_count":count,
                            "any_complete_event":bool(count),
                            "first_frame_argmax":int(tokens[0]),
                            "any_meh_argmax":2 in tokens,
                            "collapsed_tokens":event["collapsed_tokens"],
                            "frames":len(tokens),
                            "ctc_extra_events":int(event["extra_events"])}
                        if gain==1.0:
                            logits_saved[row["id"]+"__"+name]=(
                                logits.squeeze(0).cpu().numpy().astype(np.float32))
                    conditions[str(gain)]=frame_results
                report.append({"id":row["id"],"source_group_id":row["source_group_id"],
                               "role":"development","machine_tier":row["machine_tier"],
                               "pcm_sha256":row["source_pcm_sha256"],
                               "positive":bool(row["mono_class"]),
                               "conditions":conditions})
        need(len(report)==POSITIVE+ORDINARY
             and Counter(x["positive"] for x in report)=={True:POSITIVE,False:ORDINARY},
             "NO_UNKNOWN_OR_DUPLICATES_IN_B8_DEV_SCORE")
        logits_path=output/"ORIGINAL_FRAME_LOGITS.npz"
        np.savez_compressed(logits_path,**logits_saved)
        need(logits_path.is_file() and logits_path.stat().st_size>1000,
             "ACTUAL_ORIGINAL_FRAME_LOGITS_ARCHIVED")
        summary={}
        for gain in GAINS:
            g=str(gain);summary[g]={}
            for name in MODELS:
                pos=[r for r in report if r["positive"]]
                neg=[r for r in report if not r["positive"]]
                summary[g][name]={
                    "positive_originals":len(pos),
                    "positive_hits":sum(r["conditions"][g][name]["any_complete_event"]
                                        for r in pos),
                    "ordinary_originals":len(neg),
                    "ordinary_false_or_extra_events":sum(
                        r["conditions"][g][name]["full_phrase_event_count"]
                        for r in neg),
                    "positive_missing_meh_argmax":sum(
                        not r["conditions"][g][name]["any_meh_argmax"]
                        for r in pos),
                    "ordinary_complete_event_clips":sum(
                        r["conditions"][g][name]["any_complete_event"]
                        for r in neg)}
        result={"schema":SCHEMA,
          "status":"B8_NEW_SOURCE_FROZEN_THREE_WAY_DEV_DIAGNOSTIC_ONLY",
          "run":identity,"source_receipt":audit,"qualified_source_groups":QUALIFIED_GROUPS,
          "artifacts":artifacts,
          "model_pt_sha256":{name:info["pt"] for name,info in MODELS.items()},
          "original_frame_logits_sha256":sha(logits_path.read_bytes()),
          "conditions":list(GAINS),"real_model_forward_conditions":len(report)*len(GAINS)*len(MODELS),
          "derived_conditions_do_not_increase_original_denominator":True,
          "summary":summary,"per_original":report,
          "independent_synthetic_development_only":True,
          "same_tts_engine_not_human_gold":True,
          "B7_historical_strict_replay":"FAILED_UNCHANGED",
          "B8_arms_have_43755_trained_different_floats":True,
          "cal_locked_opened":False,"new_optimizer_steps":0,"new_tts_asr_calls":0,
          "model_selection_allowed":False,"winner":None,
          "physical_ssc305":"NOT_RUN","real_human_3_5m_frr":"NOT_MEASURED",
          "natural_far_per_hour":"NOT_MEASURED","product_promotion_allowed":False}
        need(result["real_model_forward_conditions"]==99,
             "ONLY_99_FROZEN_DEV_FORWARD_CONDITIONS")
        save(output/"RESULT.json",result)
        print(json.dumps({"status":result["status"],"summary":summary,
             "real_model_forward_conditions":result["real_model_forward_conditions"],
             "product_promotion_allowed":False},sort_keys=True))
    except BaseException as e:
        save(output/"STOP.json",{"status":"STOP_B8_FROZEN_DEV_SCORE",
             "type":type(e).__name__,"reason":str(e)[:450],
             "cal_locked_opened":False,"product_promotion_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    flags=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--guard","--audit","--execute"):
        flags.add_argument(flag,action="store_true")
    for key in ("source","screen","b6-zip","b8-zip","prior-failed-zip","output"):
        ap.add_argument("--"+key,type=Path)
    a=ap.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.source is not None and a.screen is not None,
         "NEW_VERIFIED_B8_SOURCE_AND_SCREEN_REQUIRED")
    if a.audit:
        rows,proof=select(a.source,a.screen)
        print(json.dumps({**proof,"originals_for_scoring":len(rows),
                          "model_forwards":0},sort_keys=True));return
    need(a.b6_zip and a.b8_zip and a.prior_failed_zip and a.output,
         "TWO_ACTUAL_TRAINED_ZIPS_AND_PRIOR_STOP_REQUIRED")
    execute(a.source,a.screen,a.b6_zip,a.b8_zip,a.prior_failed_zip,a.output)

if __name__=="__main__":main()
