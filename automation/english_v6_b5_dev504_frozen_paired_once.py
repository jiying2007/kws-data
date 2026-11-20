#!/usr/bin/env python3
"""Pre-registered frozen B3 versus matched B4 control/half on unseen DEV496 source.

Exactly 5 machine-accepted positive and 8 ordinary original 16 kHz PCM from
5/8 genuinely new Piper source groups, qualified AFTER all 3 model weights
were frozen. Three original/derived gain conditions, NO training, no
threshold search, no CAL/LOCKED, no product/human/FAR claims.
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

saved=load("english_v6_b5_dev504_saved_screen_once")
upstream=load("english_v5_mero_prospective_three_frozen_once")
v5=upstream.v5
SCHEMA="english-v6-b5-dev504-three-frozen-new-source-original-paired-v1"
RECOVERY={"failed_run":44698504736,"head":"72a8309188ab95ac3c35e32ea347d0240bcd2881","failed_job":136161797047}
REPO="jiying2007/kws-data"
BEFORE="72a8309188ab95ac3c35e32ea347d0240bcd2881"
SOURCE={"id":14607979756,"run":44691501603,
        "head":"ed22816a60d7046a705be40341865dedba50c811",
        "name":"english-v6-b5-fresh-dev504-original-44691501603",
        "bytes":650465,
        "sha":"5ab5d84cd4c0ad661d4a89542d55d68f9f014648a554432f9e200b25e4ae4e06"}
SCREEN={"id":14611385826,"run":44696685618,
        "head":"027621f5f908961b3133ae15d26485821cb07b43","name":"english-v6-b5-dev504-saved-machine-screen-44696685618",
        "bytes":5601,
        "sha":"d5309d04ddc0c5eaac11befcefe85ce465925139340b0d9ec9cce35d78b71e75"}
ARCHIVES={
  "B4":{"id":14597862669,"run":44665669003,
        "head":"bcdb6c94d262d40148580c1886ac3856c53c18d5",
        "name":"english-v6-b4-matched-clean-vs-half-44665669003",
        "bytes":1000659,
        "sha":"6eba8013da5ef06aa5046eab3f8d48b912af5f80941d98f94d0affae23764cb2"},
  "B5":{"id":14608546071,"run":44689985361,
        "head":"b82b98eae9497c7aab672a01b338c1aa0293db50",
        "name":"english-v6-b5-replay-vs-hey-margin-44689985361",
        "bytes":1000484,
        "sha":"e945b156d52199569aa36963b786424df7940f0b9285a26af9509284024e081a"},
}
MODELS={
  "B4_half":{"archive":"B4","stem":"mero43-b4-paired05-192",
      "pt":"c795f8b5f6bf8e9c37d195126fc25006dbde92fb71591473c05854e3d3698f15",
      "npz":"2bc3e7a56f457849a2f2c8697509ea2a47558d90d52ecff3ae856186730d3f5c",
      "onnx":"c2cf9d80938f94b4d7c7f5a5bfe7400c5f3e50027b7afc03849ec624f399c128"},
  "B5_control":{"archive":"B5","stem":"mero43-b5-replay-control192",
      "pt":"0b0f09f4db86e9b10a05c194bba56071be46d5031e573a22183b785a3b65659a"},
  "B5_early_hey":{"archive":"B5","stem":"mero43-b5-early-hey192",
      "pt":"784f82b552c411f04631d5adc0664b4f07f305ca45d551a1de8352e844ee440d"},
}
GAINS=(1.0,0.75,0.5)
SOURCE_ORIGINALS=32
POSITIVE=5
ORDINARY=8
UNKNOWN=19
SOURCE_GROUPS=8
QUALIFIED_GROUPS=3
PARAMETERS=43876

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
         and prev["original_observer_run"]==44694776784
         and prev["new_kws_model_forwards"]==0
         and saved.ORIGINAL_SHA=="e043d58a1ad13dfdc27136c73237bedafe223eac"
         and saved.BEFORE=="e043d58a1ad13dfdc27136c73237bedafe223eac"
         and saved.SOURCE_RUN==SOURCE["run"]
         and saved.SOURCE["id"]==SOURCE["id"]
         and saved.SOURCE["digest"]=="sha256:"+SOURCE["sha"]
         and SCREEN["head"]=="027621f5f908961b3133ae15d26485821cb07b43"
         and len(MODELS)==3 and set(ARCHIVES)=={"B4","B5"}
         and list(GAINS)==[1.0,0.75,0.5]
         and (POSITIVE,ORDINARY,UNKNOWN,QUALIFIED_GROUPS)==(5,8,19,3),
         "EXACT_NEW_B5_SOURCE_SCREEN_B4_REPLAY_B5_MODEL_IDENTITY")
    workflow=(ROOT/".github/workflows/english-v6-b5-dev504-frozen-paired-once.yml").read_text(encoding="utf-8")
    need(workflow.count("'"+ARCHIVES["B4"]["sha"]+"'")==1
         and workflow.count("'"+ARCHIVES["B5"]["sha"]+"'")==1
         and len(ARCHIVES["B4"]["sha"])==len(ARCHIVES["B5"]["sha"])==64
         and "github.event.before == '"+BEFORE+"'" in workflow,
         "WORKFLOW_B4_B5_SHA_64_AND_FIRST_MAIN_GUARD_EQUAL_SOURCE")
    need(RECOVERY["head"]==BEFORE and RECOVERY["failed_run"]==44698504736,
         "EXACT_PRIOR_PREINFERENCE_FAILURE_IS_RECOVERABLE")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_NEURAL_FORWARD",
            "keyword":"Hey Mero","models":{
               k:{"artifact":ARCHIVES[v["archive"]]["id"],"pt_sha256":v["pt"]}
               for k,v in MODELS.items()},
            "new_source_run":SOURCE["run"],
            "saved_machine_screen_run":SCREEN["run"],
            "reviewed_originals":SOURCE_ORIGINALS,
            "machine_qualified_positive":POSITIVE,
            "machine_qualified_ordinary":ORDINARY,
            "excluded_unknown":UNKNOWN,
            "qualified_development_groups":QUALIFIED_GROUPS,
            "frozen_gains":list(GAINS),
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
             "User-Agent":"kws-v6-b4-fresh-dev496-paired-models"})
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
    prior=get("runs/"+str(RECOVERY["failed_run"]))
    need(prior.get("id")==RECOVERY["failed_run"]
         and prior.get("head_sha")==RECOVERY["head"]
         and prior.get("event")=="push"
         and prior.get("run_attempt")==1
         and prior.get("status")=="completed"
         and prior.get("conclusion")=="failure",
         "ACTUAL_PREINFERENCE_FAILED_RUN_NOT_SUCCESSFUL_TRAIN")
    jobs=get("runs/"+str(RECOVERY["failed_run"])+"/jobs?per_page=100").get("jobs",[])
    failed=[j for j in jobs if j.get("id")==RECOVERY["failed_job"]]
    need(len(failed)==1 and failed[0].get("name")=="real-three-frozen-models"
         and failed[0].get("conclusion")=="failure",
         "EXACT_PREVIOUS_FAILED_JOB")
    steps=failed[0].get("steps",[])
    need(sum(s.get("conclusion")=="failure" and
             s.get("name")=="Download trained B4-half and B5 matched originals and verify entire immutable archive byte hashes"
             for s in steps)==1
         and all(s.get("conclusion")=="skipped"
                 for s in steps if s.get("name","").startswith("Real 13 independent synthetic DEV original PCM")),
         "PREVIOUS_FAILURE_BEFORE_ANY_REAL_FROZEN_INFERENCE")
    old_art=get("runs/"+str(RECOVERY["failed_run"])+"/artifacts?per_page=100")
    need(old_art.get("total_count")==0 and old_art.get("artifacts")==[],
         "PREVIOUS_FAILED_RUN_HAS_NO_RESULTS_OR_ARTIFACTS")
    return {"source":SOURCE["id"],"screen":SCREEN["id"],
            "models":{k:v["id"] for k,v in ARCHIVES.items()}}

def select(source,screen):
    originals=saved.source_rows(source)
    doc=json.loads((Path(screen)/"SCREEN.json").read_bytes())
    need(doc.get("schema")==saved.SCHEMA
         and doc.get("status")=="PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY"
         and doc.get("source_run")==SOURCE["run"]
         and doc.get("original_observer_run")==44694776784
         and doc.get("source_group_counts")=={"development":QUALIFIED_GROUPS}
         and doc.get("tier_counts")=={"silver":11,"weak":2,"unknown":UNKNOWN}
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
         "FROZEN_INDEPENDENT_MACHINE_QA_ONLY")
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
         "EXACT_FIVE_POSITIVE_EIGHT_ORDINARY_NO_UNKNOWN")
    return sorted(accepted,key=lambda x:x["id"]),{
       "status":"AUDITED_FIVE_NEW_POSITIVE_EIGHT_ORDINARY",
       "source_groups":SOURCE_GROUPS,
       "qualified_positive_source_groups":sorted(fully_qualified),
       "new_originals":len(accepted),
       "human_gold":False,
       "no_cal_locked":True}

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
        result=json.loads(z.read("RESULT.json"))
        if kind=="B4":
            need(result.get("status")=="RESEARCH_B4_MATCHED_TRAINONLY_REAL_WEIGHTS_NO_PROMOTION"
                 and result.get("schema")=="english-v6-b4-matched-b3-trainonly-05gain-ablation-v1"
                 and result.get("actual_total_optimizer_steps")==384
                 and result.get("source_train_groups")==35
                 and result.get("model_parameters")==PARAMETERS
                 and result.get("cal_locked_opened") is False
                 and result.get("product_promotion_allowed") is False,
                 "EXACT_TRAINED_B4_HALF_PARENT")
            arms=None
        else:
            need(result.get("status")=="RESEARCH_B5_MATCHED_TRAIN_ONLY_HEY_AUX_MODEL_NO_PROMOTION"
                 and result.get("schema")=="english-v6-b5-matched-trainonly-early-hey-margin-v1"
                 and result.get("actual_optimizer_updates")==384
                 and result.get("per_arm_steps")==192
                 and result.get("train_source_groups")==35
                 and result.get("model_parameters")==PARAMETERS
                 and result.get("only_changed_train_factor")=="EARLY_SUFFIX_HEY_BLANK_AUXILIARY"
                 and result.get("dev_model_forwards")==0
                 and result.get("cal_locked_opened") is False
                 and result.get("model_selection_allowed") is False
                 and result.get("product_promotion_allowed") is False
                 and result.get("B4_half_baseline_replay",{}).get("status")
                    =="PASS_EXACT_B4_HALF_TENSOR_REPLAY",
                 "ACTUAL_B5_TRAIN_AND_EXACT_REPLAY")
            arms={arm["arm"]:arm for arm in result["arms"]}
            need(set(arms)=={"control","early_hey"},"TWO_B5_MATCHED_ARMS")
        for label,spec in MODELS.items():
            if spec["archive"]!=kind:continue
            stem=spec["stem"]
            chunks={}
            for ext in ("pt","npz","onnx"):
                chunks[ext]=z.read("models/"+stem+"."+ext)
                need(len(chunks[ext])>10000,"NONEMPTY_FROZEN_"+label+"_"+ext)
                if kind=="B4":
                    digest=spec[ext]
                else:
                    arm="control" if label=="B5_control" else "early_hey"
                    digest=arms[arm]["files"][stem+"."+ext]["sha256"]
                need(sha(chunks[ext])==digest,"AUTHENTIC_TRAINED_"+label+"_"+ext)
            need(sha(chunks["pt"])==spec["pt"],"FROZEN_PT_SHA_"+label)
            weights=torch.load(io.BytesIO(chunks["pt"]),map_location="cpu",weights_only=True)
            net=v5.build_model(torch)
            net.load_state_dict(weights,strict=True)
            net.eval()
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

def gain_pcm(pcm,gain,np):
    need(gain in GAINS and len(pcm)%2==0
         and isinstance(pcm,bytes),"ONLY_PREDECLARED_GAIN_AND_PCM16")
    if gain==1.0:return pcm
    x=np.frombuffer(pcm,dtype="<i2").astype(np.float32)*gain
    return np.rint(x).clip(-32768,32767).astype("<i2").tobytes()

def execute(source,screen,b4_zip,b5_zip,output):
    import numpy as np
    import torch
    identity=guard()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "ONE_NEW_FROZEN_DEV_SCORE_OUTPUT")
    output.mkdir()
    save(output/"STARTED.json",{"schema":SCHEMA,"run":identity,
         "models":["B4_half","B5_control","B5_early_hey"],
         "new_tts_asr_calls":0,"new_optimizer_steps":0,
         "cal_locked_opened":False,"auto_model_selection":False})
    try:
        artifacts=provenance(os.environ.get("GH_TOKEN"))
        rows,audit=select(source,screen)
        need(len(rows)==POSITIVE+ORDINARY,"ONLY_ACCEPTED_NEW_DEV_ORIGINAL_PCM")
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        models={}
        models.update(verify_model_archive(b5_zip,"B5",torch,np))
        models.update(verify_model_archive(b4_zip,"B4",torch,np))
        need(set(models)==set(MODELS),"ALL_THREE_FROZEN_ACOUSTIC_MODELS")
        bank,mean,scale=models["B4_half"][1:]
        parent=models["B4_half"][0].state_dict()
        replay=models["B5_control"][0].state_dict()
        need(set(parent)==set(replay)
             and all(torch.equal(parent[k],replay[k]) for k in parent),
             "EXACT_B4_HALF_TO_B5_CONTROL_TENSOR_REPLAY")
        for name,(_,other_bank,other_mean,other_scale) in models.items():
            need(np.array_equal(bank,other_bank) and np.array_equal(mean,other_mean)
                 and np.array_equal(scale,other_scale),
                 "SAME_FRONTEND_TRAIN_NORMALIZATION_FOR_"+name)
        report=[]
        original_logit_arrays={}
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
                    normalized=torch.from_numpy(((features-mean)/scale).astype(np.float32)).unsqueeze(0)
                    results={}
                    for name,(net,*_) in models.items():
                        logits=net(normalized)
                        need(logits.shape[-1]==4
                             and bool(torch.isfinite(logits).all()),
                             "FOUR_TOKEN_FROZEN_CTC_FRAMES")
                        tokens=logits.argmax(-1).reshape(-1).tolist()
                        events=v5.ctc_greedy(logits,torch)
                        positive_hits=int(events["mero_event"])+int(events["extra_events"])
                        results[name]={
                           "full_phrase_event_count":positive_hits,
                           "any_complete_event":bool(positive_hits),
                           "first_frame_argmax":int(tokens[0]),
                           "any_meh_argmax":2 in tokens,
                           "collapsed_tokens":events["collapsed_tokens"],
                           "frames":len(tokens),
                           "ctc_extra_events":int(events["extra_events"])}
                        if gain==1.0:
                            original_logit_arrays[row["id"]+"__"+name]=(
                                logits.squeeze(0).cpu().numpy().astype(np.float32))
                    conditions[str(gain)]=results
                report.append({"id":row["id"],"source_group_id":row["source_group_id"],
                               "role":"development","machine_tier":row["machine_tier"],
                               "pcm_sha256":row["source_pcm_sha256"],
                               "positive":bool(row["mono_class"]),
                               "conditions":conditions})
        need(len(report)==POSITIVE+ORDINARY
             and Counter(x["positive"] for x in report)=={True:POSITIVE,False:ORDINARY},
             "NO_UNKNOWN_IN_NEW_THREE_MODEL_SCORE")
        logit_path=output/"ORIGINAL_FRAME_LOGITS.npz"
        np.savez_compressed(logit_path,**original_logit_arrays)
        need(logit_path.is_file() and logit_path.stat().st_size>1000,
             "SAVE_AUTHENTIC_NEW_DEV_FRAME_LOGITS")
        summary={}
        for gain in GAINS:
            g=str(gain)
            summary[g]={}
            for name in MODELS:
                pos=[row for row in report if row["positive"]]
                neg=[row for row in report if not row["positive"]]
                summary[g][name]={
                    "positive_originals":len(pos),
                    "positive_hits":sum(r["conditions"][g][name]["any_complete_event"] for r in pos),
                    "ordinary_originals":len(neg),
                    "ordinary_false_or_extra_events":sum(
                        r["conditions"][g][name]["full_phrase_event_count"] for r in neg),
                    "positive_missing_meh_argmax":sum(
                        not r["conditions"][g][name]["any_meh_argmax"] for r in pos),
                    "ordinary_complete_event_clips":sum(
                        r["conditions"][g][name]["any_complete_event"] for r in neg)}
        results={"schema":SCHEMA,
          "status":"B5_NEW_SOURCE_FROZEN_THREE_WAY_DEV_DIAGNOSTIC_ONLY",
          "run":identity,"source_receipt":audit,
          "qualified_positive_source_groups":QUALIFIED_GROUPS,
          "artifacts":artifacts,
          "model_pt_sha256":{name:info["pt"] for name,info in MODELS.items()},
          "original_frame_logits_sha256":sha(logit_path.read_bytes()),
          "conditions":list(GAINS),
          "derived_conditions_do_not_increase_original_denominator":True,
          "summary":summary,"per_original":report,
          "independent_synthetic_development_only":True,
          "same_tts_engine_not_human_gold":True,
          "B4_half_B5_control_identical_tensors":True,
          "cal_locked_opened":False,
          "new_optimizer_steps":0,"new_tts_asr_calls":0,
          "model_selection_allowed":False,"winner":None,
          "physical_ssc305":"NOT_RUN",
          "real_human_3_5m_frr":"NOT_MEASURED",
          "natural_far_per_hour":"NOT_MEASURED",
          "product_promotion_allowed":False}
        save(output/"RESULT.json",results)
        print(json.dumps({"status":results["status"],"summary":summary,
            "product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        save(output/"STOP.json",{"status":"STOP_NO_FROZEN_SCORE_REPLAY",
             "type":type(exc).__name__,"reason":str(exc)[:450],
             "cal_locked_opened":False,"product_promotion_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    flags=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--guard","--audit","--execute"):
        flags.add_argument(flag,action="store_true")
    for key in ("source","screen","b4-zip","b5-zip","output"):
        ap.add_argument("--"+key,type=Path)
    a=ap.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.source is not None and a.screen is not None,
         "NEW_VERIFIED_SOURCE_AND_SCREEN_REQUIRED")
    if a.audit:
        rows,proof=select(a.source,a.screen)
        print(json.dumps({**proof,"originals_for_scoring":len(rows),
                          "model_forwards":0},sort_keys=True));return
    need(a.b4_zip and a.b5_zip and a.output,"EXACT_TWO_TRAINED_MODEL_ARCHIVES")
    execute(a.source,a.screen,a.b4_zip,a.b5_zip,a.output)

if __name__=="__main__":main()
