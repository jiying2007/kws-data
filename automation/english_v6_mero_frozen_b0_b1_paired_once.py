"""Hey Mero V6 independent prospective DEV: two prior frozen 43K FSMN checkpoints.

Only NEW 488..495 Piper development voices: six independently machine-qualified
canonical Mero originals, eight qualified ordinary originals. The other 18,
including all nearphone, remain UNKNOWN. No new TTS, ASR, training, model/
threshold tuning, old exposed development, CAL/LOCKED KWS or release claim.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import wave

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("v6cmp_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj
saved=load("english_v6_mero_prospective_dev8_saved_screen_once")
legacy=load("english_v5_mero_prospective_three_frozen_once")
v5=legacy.v5
SCHEMA="english-v6-mero-frozen-b0-b1-new-source-paired-v1"
REPO="jiying2007/kws-data"
BEFORE="e7f3e56819b4156cec4fd965e5f0529adcc99a73"
SOURCE={"id":14543254675,"run":44554345276,
 "head":"3af2027d2ca9d195e18b1cf8da8e07663d841bc6",
 "name":"english-v6-mero-prospective-dev8-original-44554345276",
 "bytes":646524,"sha":"530fccdd25abca9e76331dd8dad1f2440181158e140f2279ee952833a8d220f6"}
SCREEN={"id":14545584039,"run":44558582911,"head":BEFORE,
 "name":"english-v6-mero-prospective-dev8-saved-machine-screen-44558582911",
 "bytes":5586,"sha":"9fd68b177823a24446c50c163a4838967de7b6e9d3e2be23861d136d93af0c9e"}
MODELS={
 "B0":{"id":14514228406,"run":44495594627,"head":"d0b567fe37147aae6ab5d7bdec3721b879a0b22d",
   "name":"english-v5-mero-expanded-fsmn256-44495594627","bytes":501276,
   "sha":"91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702",
   "nameprefix":"mero-fsmn-ctc-expanded256","params":43876,"steps":256},
 "B1":{"id":14539983171,"run":44546464853,"head":"bb73e1742bc6ffb175c0eb3a3b2983b1b54b6917",
   "name":"english-v5-mero43-trainonly-robust288-44546464853","bytes":500910,
   "sha":"bab018a63aaf4cc760ee835b8554ebacf73e6ee77c37adc7c9d5008b1a03d7ce",
   "nameprefix":"mero-fsmn-ctc-trainonly-robust288","params":43876,"steps":288,
   "pt":"20d0c4632224cf332460493695c0dd21927a80ed13034a9531f5d5230d2f7e05"}}

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(blob):return hashlib.sha256(blob).hexdigest()
def save(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)
        f.write("\n")

def source_check():
    x=saved.source_check()
    need(x["status"]=="SAVED_PROSPECTIVE_DEV8_160_REAL_REVIEWS_PINNED"
         and x["saved_source_rows"]==32
         and x["original_observer_run"]==44556480241
         and x["new_kws_model_forwards"]==0
         and saved.SOURCE["id"]==SOURCE["id"]
         and saved.SOURCE_RUN==SOURCE["run"]
         and saved.BEFORE==BEFORE and saved.ORIGINAL_SHA==BEFORE
         and legacy.MODELS["fsmn43k256"]["id"]==MODELS["B0"]["id"]
         and legacy.MODELS["fsmn43k256"]["sha"]==MODELS["B0"]["sha"]
         and MODELS["B1"]["pt"]=="20d0c4632224cf332460493695c0dd21927a80ed13034a9531f5d5230d2f7e05"
         and {k:v["params"] for k,v in MODELS.items()}=={"B0":43876,"B1":43876},
         "EXACT_V6_B0_B1_ALREADY_FROZEN_SOURCE_AND_WEIGHTS")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_B0_B1_AUDIO_FORWARDS",
       "keyword":"Hey Mero","models":{k:v["id"] for k,v in MODELS.items()},
       "new_reviewed_positive":6,"new_reviewed_ordinary":8,
       "qualified_development_groups":4,"unknown_excluded":18,
       "new_optimizer_steps":0,"new_tts_asr_forwards":0,
       "new_cal_locked_reads":0,"model_selection_allowed":False,
       "product_promotion_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
      "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
      "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
      "GITHUB_RUN_ATTEMPT":"1"}.items()),"PROTECTED_MAIN_ONCE_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "EXACT_RUN_AND_COMMIT")
    if event is None:event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "NOT_FIRST_MAIN_PROSPECTIVE_RESEARCH_COMMIT")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                        cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"ORIGINAL_PARENT_NEVER_REPLAYED")
    return {"run_id":env["GITHUB_RUN_ID"],"no_replay":True}

def provenance(token):
    import urllib.request
    need(type(token) is str and len(token)>=10,"ACTION_TOKEN_REQUIRED")
    result={}
    for key,x in [("source",SOURCE),("screen",SCREEN),*MODELS.items()]:
        req=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+"/actions/artifacts/"+str(x["id"]),
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"mero-frozen-research-source"})
        with urllib.request.urlopen(req,timeout=40) as f:
            raw=f.read(1000001)
        need(len(raw)<1000001,"BOUND_GITHUB_METADATA")
        actual=json.loads(raw)
        need(actual.get("id")==x["id"]
             and actual.get("name")==x["name"]
             and actual.get("size_in_bytes")==x["bytes"]
             and actual.get("digest")=="sha256:"+x["sha"]
             and actual.get("expired") is False
             and actual.get("workflow_run",{}).get("id")==x["run"]
             and actual.get("workflow_run",{}).get("head_sha")==x["head"],
             "BYTE_AND_RUN_PIN_"+key)
        result[key]={"id":x["id"],"sha256":x["sha"]}
    return result

def select(source,screen):
    originals=saved.source_rows(source)
    s=json.loads((Path(screen)/"SCREEN.json").read_bytes())
    need(s.get("schema")==saved.SCHEMA
       and s.get("status")=="PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY"
       and s.get("source_run")==SOURCE["run"]
       and s.get("original_observer_run")==44556480241
       and s.get("source_group_counts")=={"development":4}
       and s.get("tier_counts")=={"silver":13,"weak":1,"unknown":18}
       and s.get("observed_label_counts")=={"0":8,"2":6}
       and s.get("nearphone_as_ordinary_negative") is False
       and s.get("machine_only_no_human_gold") is True
       and s.get("new_audio_or_model_forwards")==0
       and s.get("new_optimizer_updates")==0
       and s.get("new_kws_model_forwards")==0
       and s.get("training_snapshot_allowed") is False
       and s.get("locked_kws_test_opened") is False
       and s.get("product_promotion") is False,
       "REQUIRE_FROZEN_PROSPECTIVE_SOURCE_QUALITY_PASS")
    original={r["id"]:r for r in originals}
    observations=s.get("rows",[])
    need(len(originals)==len(original)==len(observations)==32,
         "EXACT_ALL_32_NEW_DEV_ORIGINALS")
    used=set();groups={};selected=[]
    for row in observations:
        sid=row.get("id")
        need(sid in original and sid not in used
             and row.get("role")=="development"
             and row.get("human_gold") is False
             and row.get("training_allowed") is False,
             "NO_DUPLICATE_OR_NONDEV_ROW")
        used.add(sid)
        r=original[sid]
        need(all(row.get(k)==r.get(k) for k in (
             "source_group_id","speaker_index","role","cell_id",
             "requested_class","pcm_sha256","wav_sha256","samples")),
             "BYTE_BOUND_ORIGINAL_SOURCE_IDENTITY")
        groups.setdefault(r["source_group_id"],[]).append(row)
        if r["cell_id"]=="nearphone-meero-challenge":
            need(row.get("tier")=="unknown" and row.get("observed_class") is None,
                 "NEARPHONE_UNKNOWN_NOT_NEGATIVE")
        if row.get("tier") in ("silver","weak"):
            label=row.get("observed_class")
            need(label in (0,2) and label==r["requested_class"]
                 and r["cell_id"]!="nearphone-meero-challenge",
                 "INDEPENDENT_ACOUSTIC_LABEL_ONLY")
            selected.append({**r,"source_pcm_sha256":r["pcm_sha256"],
                "mono_class":int(label==2),"machine_tier":row["tier"]})
        else:
            need(row.get("tier")=="unknown" and row.get("observed_class") is None,
                 "UNKNOWN_NEVER_ADMITTED")
    need(used==set(original) and len(groups)==8
         and all(len(g)==4 and {r["cell_id"] for r in g}=={
              "mero-fast","mero-slow","ordinary-time",
              "nearphone-meero-challenge"} for g in groups.values()),
         "ONE_COMPLETE_SOURCE_COHORT_488_495")
    qualified={g for g,rows in groups.items() if {0,2}<={
       r["observed_class"] for r in rows if r["tier"] in ("silver","weak")}}
    need(len(qualified)==4
         and set(s.get("source_group_identity_members",[]))==qualified
         and Counter(r["mono_class"] for r in selected)=={0:8,1:6}
         and {r["source_group_id"] for r in selected if r["mono_class"]==1}==qualified,
         "EXACT_FOUR_SOURCE_GROUPS_SIX_POSITIVES_EIGHT_ORDINARY")
    return sorted(selected,key=lambda r:r["id"]),{
        "status":"AUDITED_6_NEW_DEV_POSITIVE_8_NEW_DEV_ORDINARY",
        "source_groups":8,"qualified_positive_sources":sorted(qualified),
        "no_locked_or_calibration":True,"no_new_models":True}

def pcm(row,source):
    wav=Path(source)/"wav"/(row["id"]+".wav")
    need(wav.is_file() and not wav.is_symlink()
         and sha(wav.read_bytes())==row["wav_sha256"],"ORIGINAL_WAV_SHA")
    with wave.open(str(wav),"rb") as f:
        need(f.getnchannels()==1 and f.getsampwidth()==2
             and f.getframerate()==16000 and f.getcomptype()=="NONE"
             and f.getnframes()==row["samples"],"FULL_PCM16_NO_CROP")
        raw=f.readframes(f.getnframes())
    need(len(raw)==2*row["samples"] and sha(raw)==row["source_pcm_sha256"],
         "EXACT_ORIGINAL_AUDIO_SHA")
    return raw

def model(torch,np,folder,kind):
    spec=MODELS[kind];folder=Path(folder)
    need(folder.is_absolute() and folder.is_dir() and not folder.is_symlink(),
         "PREVIOUSLY_FROZEN_WEIGHT_FOLDER")
    report=json.loads((folder/"RESULT.json").read_bytes())
    if kind=="B0":
        need(report.get("model_parameters")==43876 and
             report.get("new_actual_optimizer_updates")==256 and
             report.get("status") in ("RESEARCH_NEW_SOURCE_DEV_NOT_QUALIFIED",
                                      "RESEARCH_EXPOSED_DEV_NOT_QUALIFIED") and
             report.get("product_promotion_allowed") is False,
             "REAL_256_STEP_FROZEN_BASELINE")
        detail=report["new_model_weights"]
    else:
        need(report.get("schema")=="en-v5-frozen43k-trainonly-gain-noise-reflections-288-v1"
             and report.get("status")=="RESEARCH_EXPOSED_DEV_DIAGNOSTIC_ONLY_NO_PROMOTION"
             and report.get("model_parameters")==43876
             and report.get("actual_optimizer_updates")==288
             and report.get("train_source_groups")==35
             and report.get("cal_locked_opened") is False
             and report.get("product_promotion_allowed") is False,
             "REAL_288_STEP_FROZEN_ROBUST_MODEL")
        detail=report["new_frozen_models"]
    paths={}
    for ext in ("pt","npz","onnx"):
        name=spec["nameprefix"]+"."+ext
        path=folder/"models"/name
        receipt=detail.get(name,{})
        need(path.is_file() and not path.is_symlink()
             and path.stat().st_size==receipt.get("bytes")
             and sha(path.read_bytes())==receipt.get("sha256"),
             "SAVED_FULL_WEIGHT_BYTES_"+kind+"_"+ext)
        paths[ext]=path
    if kind=="B1":
        need(sha(paths["pt"].read_bytes())==spec["pt"],
             "ORIGINAL_ROBUST_PT_PREDECLARED_SHA")
    net=v5.build_model(torch)
    net.load_state_dict(torch.load(paths["pt"],map_location="cpu",
                                    weights_only=True),strict=True)
    net.eval()
    with np.load(paths["npz"],allow_pickle=False) as weights:
        need(set(weights.files)==set(net.state_dict())|{
              "mel_filters","train_mean","train_scale"},
             "EXACT_FROZEN_PT_NPZ_TENSORS")
        bank=weights["mel_filters"].copy()
        mean=weights["train_mean"].copy()
        scale=weights["train_scale"].copy()
        need(bank.shape==(257,40) and mean.shape==scale.shape==(40,)
             and np.isfinite(bank).all() and np.isfinite(mean).all()
             and np.isfinite(scale).all() and (scale>0).all(),
             "FROZEN_FINITE_TRAIN_NORMALIZER")
        for key,tensor in net.state_dict().items():
            need(np.array_equal(weights[key],tensor.cpu().numpy()),
                 "REAL_MODEL_PT_NPZ_IDENTICAL_"+kind)
    return net,bank,mean,scale,{k:sha(v.read_bytes()) for k,v in paths.items()}

def execute(source,screen,baseline,challenger,output):
    import numpy as np
    import torch
    identity=guard()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "ONE_FROZEN_PROSPECTIVE_RESULT_FOLDER")
    output.mkdir()
    save(output/"STARTED.json",{"schema":SCHEMA,"event":identity,
         "new_optimizer_steps":0,"threshold_tuning":False,
         "calibration_or_locked_audio":False})
    try:
        pins=provenance(os.environ.get("GH_TOKEN"))
        examples,audit=select(source,screen)
        save(output/"ADMISSION.json",{"source":audit,"pins":pins})
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
        nets={k:model(torch,np,d,k) for k,d in
              (("B0",baseline),("B1",challenger))}
        need(all(np.array_equal(a,b) for a,b in zip(
              nets["B0"][1:4],nets["B1"][1:4])),
             "SAME_40MEL_AND_TRAIN_ONLY_NORMALIZATION")
        records=[];logits_saved={}
        with torch.no_grad():
            for row in examples:
                raw=pcm(row,source)
                half=np.rint(np.frombuffer(raw,dtype="<i2").astype(np.float32)*0.5
                        ).astype("<i2").tobytes()
                feats={"original":v5.features(raw,row,torch,np,nets["B0"][1]),
                       "half_gain":v5.features(half,row,torch,np,nets["B0"][1]),
                       "snr10":v5.features(raw,row,torch,np,nets["B0"][1],noise_db=10)}
                conditions={}
                for name,features in feats.items():
                    hits={}
                    for key,(net,bank,mean,scale,weights) in nets.items():
                        x=torch.from_numpy(((features-mean)/scale).astype(np.float32)).unsqueeze(0)
                        logits=net(x)
                        answer=v5.ctc_greedy(logits,torch)
                        hits[key]={"event":bool(answer["mero_event"]),
                             "extra":int(answer["extra_events"]),
                             "ctc_tokens":answer["collapsed_tokens"]}
                        if name=="original":
                            logits_saved[row["id"]+"__"+key]=(
                              logits.detach().cpu().numpy().astype(np.float32))
                    conditions[name]=hits
                records.append({"id":row["id"],"role":"development",
                    "positive":bool(row["mono_class"]),
                    "source_group_id":row["source_group_id"],
                    "pcm_sha256":row["source_pcm_sha256"],
                    "machine_tier":row["machine_tier"],"conditions":conditions})
        need(len(records)==14 and sum(r["positive"] for r in records)==6,
             "EXACT_SIX_POSITIVE_EIGHT_ORDINARY_SAME_INPUT")
        trace=output/"ORIGINAL_FRAME_LOGITS.npz"
        np.savez_compressed(trace,**logits_saved)
        need(trace.stat().st_size>100,"ACTUAL_PER_FRAME_MODEL_LOGITS")
        summary={}
        for condition in ("original","half_gain","snr10"):
            summary[condition]={}
            for key in ("B0","B1"):
                pos=[r for r in records if r["positive"]]
                neg=[r for r in records if not r["positive"]]
                summary[condition][key]={
                  "positive_originals":len(pos),
                  "positive_hits":sum(x["conditions"][condition][key]["event"] for x in pos),
                  "ordinary_originals":len(neg),
                  "ordinary_false_or_extra":sum(
                      int(x["conditions"][condition][key]["event"])+
                      x["conditions"][condition][key]["extra"] for x in neg)}
        result={"schema":SCHEMA,
          "status":"FRESH_SYNTHETIC_PAIRED_DEV_DIAGNOSTIC_NO_PROMOTION",
          "source_qualified":audit,"comparison":summary,"per_original":records,
          "actual_original_frame_logits_sha256":sha(trace.read_bytes()),
          "artifact_identity":pins,
          "model_weight_sha256":{k:v[-1] for k,v in nets.items()},
          "model_selection_allowed":False,"winner":None,
          "new_optimizer_updates":0,"new_tts_asr_calls":0,
          "old_exposed_dev_opened":False,"cal_locked_opened":False,
          "real_human_frr":"NOT_MEASURED","natural_far_hour":"NOT_MEASURED",
          "ssc305":"NOT_RUN","product_promotion_allowed":False}
        save(output/"RESULT.json",result)
        print(json.dumps({"status":result["status"],
              "comparison":summary,"product_promotion_allowed":False},
              sort_keys=True))
    except BaseException as exc:
        save(output/"STOP.json",{"status":"STOP_RESEARCH_NO_RETRY",
              "type":type(exc).__name__,"reason":str(exc)[:420],
              "no_CAL_LOCKED_access":True})
        raise

def main():
    parser=argparse.ArgumentParser()
    for flag in ("--source-check","--guard","--audit","--execute"):
        parser.add_argument(flag,action="store_true")
    for name in ("--source","--screen","--b0","--b1","--output"):
        parser.add_argument(name,type=Path)
    a=parser.parse_args()
    need(sum((a.source_check,a.guard,a.audit,a.execute))==1,
         "EXACT_ONE_PAIR_EXPERIMENT_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.source and a.screen,"EXACT_FRESH_SOURCE_AND_MACHINE_QA")
    if a.audit:
        rows,proof=select(a.source,a.screen)
        print(json.dumps({**proof,"selected_originals":len(rows)},
                         sort_keys=True));return
    need(a.b0 and a.b1 and a.output,"TWO_ORIGINAL_FROZEN_MODEL_DIRS")
    execute(a.source,a.screen,a.b0,a.b1,a.output)

if __name__=="__main__":main()
