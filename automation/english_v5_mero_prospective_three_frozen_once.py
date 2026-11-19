"""Hey Mero V5 frozen prospective development: 43k vs 93k vs phone 3M.

Only NEW 480..487 Piper development voices: two separately machine-qualified
canonical Mero originals, eight qualified ordinary originals. The other 22,
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
    spec=importlib.util.spec_from_file_location("v5cmp_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj
saved=load("english_v5_mero_prospective_dev8_saved_screen_once")
strong=load("english_v5_mero_strong_fsmn128_once")
small=load("english_v5_mero_fresh24_plus70_fit_once")
phone=load("english_v5_mero_newdev_phone_reference_once")
v5=strong.v5
SCHEMA="english-v5-prospective-three-frozen-models-v1"
REPO="jiying2007/kws-data"
BEFORE="980c2c44cd55a70a68266364b76ade947c62096f"
SOURCE={"id":14524071469,"run":44511329722,
 "head":"e0ac2a1be348d1d86e53e7a4fb28e2655d9b4e23",
 "name":"english-v5-mero-prospective-dev8-original-44511329722",
 "bytes":654281,"sha":"82ea1145bc380140c83a3c98276f11b7f0fce3c8ad0e8fd1cc82d34ac290d1b0"}
SCREEN={"id":14525005967,"run":44514289065,
 "head":"6e84d33e0978b2c731eb232e152080ed99ba1247",
 "name":"english-v5-mero-prospective-dev8-saved-machine-screen-44514289065",
 "bytes":5466,"sha":"d5c68b6ea5c69ab77a1a54828f77e9ef01fd53ff0c1d16ce072f080add0249dd"}
MODELS={
 "fsmn43k256":{"id":14514228406,"run":44495594627,
  "head":"d0b567fe37147aae6ab5d7bdec3721b879a0b22d",
  "name":"english-v5-mero-expanded-fsmn256-44495594627",
  "bytes":501276,"sha":"91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702",
  "nameprefix":"mero-fsmn-ctc-expanded256","params":43876,"steps":256,
  "onnx":"10bf30da57291f78f56db196db1137309f2c3b91dfe67bf0aa45ebb8cec05b1c"},
 "fsmn93k448":{"id":14516373314,"run":44500437774,
  "head":"826c3fb31ae2c0eb03fe7fe6e9a98ce5c9d10eba",
  "name":"english-v5-mero-strong-fsmn128-448-44500437774",
  "bytes":1047944,"sha":"ea4b70df6a24bdd867f6b5b7d1ccf178f646f57359b1a62cc937b7f7ef42c449",
  "nameprefix":"mero-strong-fsmn128x5-448","params":93444,"steps":448,
  "pt":"5025b9e8ee6379074abc615a2e32859ea3e90ed436a855402e28bfe54a479ead",
  "npz":"f933bf4565c986863a86e64749732b5e2ee75ac1af1a4d8dda479a90036e94e4",
  "onnx":"18f45048de341c36f7cd8f51843288fda5d4da5f02691a5213008e284515b1e0"}}
PHONEMODEL_SHA="68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
PHONEMODEL_BYTES=32885699

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(blob):return hashlib.sha256(blob).hexdigest()
def save(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)
        f.write("\n")

def source_check():
    a=saved.source_check()
    need(a["status"]=="SAVED_PROSPECTIVE_DEV8_160_REAL_REVIEWS_PINNED"
      and a["saved_source_rows"]==32 and a["original_observer_run"]==44512239544
      and a["new_audio_or_model_forwards"]==0
      and a["new_locked_test_forwards"]==0
      and strong.PARAMS==93444 and strong.UPDATES==448
      and small.PARAMETERS==43876 and small.UPDATES==256
      and phone.ref.MODEL_SHA256==PHONEMODEL_SHA
      and phone.ref.MODEL_BYTES==PHONEMODEL_BYTES
      and phone.DECODER["threshold"]==0.25
      and phone.DECODER["score"]==1.0
      and phone.DECODER["num_trailing_blanks"]==1
      and set(MODELS)=={"fsmn43k256","fsmn93k448"},
      "THREE_EXACT_FROZEN_RESEARCH_MODELS")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NOT_EXECUTED",
      "keyword":"Hey Mero","new_dev_positive":2,"new_dev_ordinary":8,
      "models":{k:v["id"] for k,v in MODELS.items()},
      "phone_model_sha256":PHONEMODEL_SHA,
      "fixed_phone_threshold":0.25,"optimizer_steps":0,
      "new_asr_tts":0,"locked_audio_access":False,
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
       and s.get("original_observer_run")==44512239544
       and s.get("source_group_counts")=={"development":2}
       and s.get("tier_counts")=={"silver":10,"unknown":22}
       and s.get("observed_label_counts")=={"0":8,"2":2}
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
         "ONE_COMPLETE_SOURCE_COHORT_480_487")
    qualified={g for g,rows in groups.items() if {0,2}<={
       r["observed_class"] for r in rows if r["tier"] in ("silver","weak")}}
    need(len(qualified)==2
         and set(s.get("source_group_identity_members",[]))==qualified
         and Counter(r["mono_class"] for r in selected)=={0:8,1:2}
         and {r["source_group_id"] for r in selected if r["mono_class"]==1}==qualified,
         "EXACT_TWO_SOURCE_INDEPENDENT_POSITIVES_EIGHT_ORDINARY")
    return sorted(selected,key=lambda r:r["id"]),{
        "status":"AUDITED_2_NEW_DEV_POSITIVE_8_NEW_DEV_ORDINARY",
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
         "EXACT_ORIGINAL_WEIGHT_FOLDER")
    report=json.loads((folder/"RESULT.json").read_bytes())
    need(report.get("model_parameters")==spec["params"]
         and (report.get("new_actual_optimizer_updates") if kind=="fsmn43k256"
              else report.get("optimizer_updates"))==spec["steps"]
         and report.get("product_promotion_allowed") is False
         and report.get("status") in (
              "RESEARCH_NEW_SOURCE_DEV_NOT_QUALIFIED",
              "RESEARCH_EXPOSED_DEV_NOT_QUALIFIED"),
         "REAL_REJECTED_FROZEN_WEIGHTS")
    detail=report["new_model_weights"] if kind=="fsmn43k256" else report["model_files"]
    found={}
    for ext in ("pt","npz","onnx"):
        name=spec["nameprefix"]+"."+ext
        path=folder/"models"/name
        need(path.is_file() and not path.is_symlink()
             and path.stat().st_size==detail.get(name,{}).get("bytes")
             and sha(path.read_bytes())==detail.get(name,{}).get("sha256"),
             "SAVED_PT_NPZ_ONNX_SHA_"+kind+"_"+ext)
        if ext in spec:
            need(sha(path.read_bytes())==spec[ext],
                 "FROZEN_PREDECLARED_WEIGHT_SHA")
        found[ext]=path
    net=(v5.build_model(torch) if kind=="fsmn43k256"
         else strong.build_strong_model(torch))
    net.load_state_dict(torch.load(found["pt"],map_location="cpu",
                                   weights_only=True),strict=True)
    net.eval()
    with np.load(found["npz"],allow_pickle=False) as weights:
        need(set(weights.files)==set(net.state_dict())|{
             "mel_filters","train_mean","train_scale"},
             "EXACT_PRETRAINED_STATE_AND_FRONTEND_KEYS")
        bank=weights["mel_filters"].copy()
        mean=weights["train_mean"].copy();scale=weights["train_scale"].copy()
        need(bank.shape==(257,40) and mean.shape==scale.shape==(40,)
             and np.isfinite(bank).all() and np.isfinite(mean).all()
             and np.isfinite(scale).all() and (scale>0).all(),
             "FROZEN_FINITE_TRAIN_ONLY_40_MEL")
        for key,tensor in net.state_dict().items():
            need(np.array_equal(weights[key],tensor.cpu().numpy()),
                 "PT_NPZ_EXACT_NUMERIC_PARITY")
    return net,bank,mean,scale,{x:sha(p.read_bytes()) for x,p in found.items()}

def execute(source,screen,model43,model93,phonefile,output):
    import numpy as np
    import torch
    import sherpa_onnx
    identity=guard()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "ONE_NEW_FROZEN_COMPARISON_OUTPUT")
    output.mkdir()
    save(output/"STARTED.json",{"schema":SCHEMA,"event":identity,
        "no_new_training":True,"no_threshold_tuning":True})
    try:
        pins=provenance(os.environ.get("GH_TOKEN"))
        examples,audit=select(source,screen)
        save(output/"ADMISSION.json",{"sources":audit,"pins":pins})
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
        nets={k:model(torch,np,d,k) for k,d in (
            ("fsmn43k256",model43),("fsmn93k448",model93))}
        need(phonefile.is_absolute() and phonefile.is_file()
             and phonefile.stat().st_size==PHONEMODEL_BYTES
             and phone.ref.hash_file(phonefile)==PHONEMODEL_SHA,
             "EXACT_FROZEN_OFFICIAL_PHONE_BYTES")
        with tempfile.TemporaryDirectory(prefix="v5-only-") as t:
            root=Path(t)/"model";root.mkdir()
            parts,pinned=phone.ref.pretrained.extract_locked_model(
                phonefile,"zh-en",root)
            need(pinned==PHONEMODEL_SHA,"OFFICIAL_PHONE_UNCHANGED")
            keyword=phone.ref.make_mero_keyword(parts,Path(t))
            d=phone.DECODER
            kws=sherpa_onnx.KeywordSpotter(
                tokens=str(parts["tokens"]),encoder=str(parts["encoder"]),
                decoder=str(parts["decoder"]),joiner=str(parts["joiner"]),
                keywords_file=str(keyword),num_threads=d["num_threads"],
                sample_rate=d["sample_rate"],feature_dim=d["feature_dim"],
                max_active_paths=d["max_active_paths"],keywords_score=d["score"],
                keywords_threshold=d["threshold"],
                num_trailing_blanks=d["num_trailing_blanks"],provider=d["provider"])
            records=[]
            with torch.no_grad():
                for row in examples:
                    raw=pcm(row,source)
                    hits={}
                    for kind,(net,bank,mean,scale,weights) in nets.items():
                        mel=v5.features(raw,row,torch,np,bank)
                        logits=net(torch.from_numpy(
                            ((mel-mean)/scale).astype(np.float32)).unsqueeze(0))
                        answer=v5.ctc_greedy(logits,torch)
                        hits[kind]={"event":bool(answer["mero_event"]),
                                    "extra":int(answer["extra_events"])}
                    arr=np.frombuffer(raw,dtype="<i2").astype(np.float32)/32768
                    events,steps=phone.ref.pretrained.stream_events(kws,arr)
                    need(all(x["mapped_keyword"]==phone.ref.KEYWORD for x in events),
                         "SINGLE_FROZEN_HEY_MERO_PHONE_KEYWORD")
                    hits["official_phone3m"]={"event":bool(events),
                      "extra":max(0,len(events)-1),"stream_steps":steps}
                    records.append({"id":row["id"],"role":"development",
                       "positive":bool(row["mono_class"]),
                       "source_group_id":row["source_group_id"],
                       "pcm_sha256":row["source_pcm_sha256"],
                       "machine_tier":row["machine_tier"],"candidates":hits})
        need(len(records)==10 and sum(r["positive"] for r in records)==2
             and len({r["source_group_id"] for r in records})==8,
             "EXACT_UNEXPOSED_NEW_DEV_DENOMINATORS")
        result={}
        for kind in ("fsmn43k256","fsmn93k448","official_phone3m"):
            positives=[r for r in records if r["positive"]]
            ordinary=[r for r in records if not r["positive"]]
            result[kind]={"positive":2,"positive_hits":sum(
                 r["candidates"][kind]["event"] for r in positives),
                 "ordinary":8,"ordinary_false_or_extra":sum(
                 int(r["candidates"][kind]["event"])
                 +r["candidates"][kind]["extra"] for r in ordinary)}
        report={"schema":SCHEMA,
          "status":"RESEARCH_ONLY_NEW_PROSPECTIVE_DEV_THREE_FROZEN_MODELS",
          "model_comparison":result,"full_original_events":records,
          "new_development_source_audit":audit,"artifact_pin_proof":pins,
          "model_weight_hashes":{k:v[-1] for k,v in nets.items()},
          "official_phone_model_sha":PHONEMODEL_SHA,
          "frozen_threshold":d["threshold"],"model_selection":False,
          "new_optimizer_updates":0,"new_tts_asr_phone_observer_calls":0,
          "old_exposed_dev_opened":False,"calibration_or_locked_kws_opened":False,
          "human_far_frr":"NOT_MEASURED","ssc305":"NOT_RUN",
          "product_promotion_allowed":False}
        save(output/"RESULT.json",report)
        print(json.dumps({"status":report["status"],"models":result,
                          "product_promotion_allowed":False},sort_keys=True))
    except BaseException as e:
        save(output/"STOP.json",{"status":"STOP_RESEARCH_NO_RETRY",
             "type":type(e).__name__,"reason":str(e)[:420],
             "no_calibration_or_locked_access":True})
        raise

def main():
    p=argparse.ArgumentParser()
    for flag in ("--source-check","--guard","--audit","--execute"):
        p.add_argument(flag,action="store_true")
    for arg in ("--source","--screen","--model43","--model93","--phone-model","--output"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.audit,a.execute))==1,
         "ONE_EXACT_FROZEN_COMPARISON_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.source and a.screen,"ORIGINAL_DEV_SOURCE_AND_MACHINE_REVIEW_REQUIRED")
    if a.audit:
        rows,receipt=select(a.source,a.screen)
        print(json.dumps(receipt,sort_keys=True));return
    need(a.model43 and a.model93 and a.phone_model and a.output,
         "THREE_ORIGINAL_MODEL_PACKAGES_REQUIRED")
    execute(a.source,a.screen,a.model43,a.model93,a.phone_model,a.output)
if __name__=="__main__":main()
