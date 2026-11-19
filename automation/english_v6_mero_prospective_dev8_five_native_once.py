"""Blind five native acoustic observers of 32 unexposed prospective DEV WAVs.

Reuse the byte-authenticated immutable generation artifact from run 44554345276.
Only source-label QA on original 8 unexposed development voice groups.
No KWS, model training, calibration/locked acoustic exposure, target-conditioned
ASR, guessed truth from the requested prompt or repeated prior TTS audio.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
# -I intentionally hides script cwd; only committed sibling modules are allowed.
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
def load(name):
    spec=importlib.util.spec_from_file_location("fresh24_observer_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

source_worker=load("english_v6_mero_prospective_dev8_source_once")
policy=load("english_fresh24_ctc_admission")
c=load("english_hifi_micro")
phone_worker=load("english_hifi_phones_once")
SCHEMA="english-v5-mero-prospective-dev8-five-native-source-labels-v1"
REPO="jiying2007/kws-data"
BEFORE="3af2027d2ca9d195e18b1cf8da8e07663d841bc6"
SOURCE_RUN=44554345276
SOURCE_COMMIT="3af2027d2ca9d195e18b1cf8da8e07663d841bc6"
SOURCE_PARENT="bb73e1742bc6ffb175c0eb3a3b2983b1b54b6917"
SOURCE_COUNT=32
REVIEW_COUNT=32
PROVIDERS=("qwen","whisper","parakeet","xlsr","lv60")
SOURCE={"id":14543254675,"name":"english-v6-mero-prospective-dev8-original-44554345276",
        "size":646524,
        "digest":"sha256:530fccdd25abca9e76331dd8dad1f2440181158e140f2279ee952833a8d220f6"}

def need(ok,reason):
    if not ok:raise ValueError(reason)

def original_identity():
    return {"run_id":str(SOURCE_RUN),"commit":SOURCE_COMMIT,
            "before":SOURCE_PARENT,"attempt":1}

def source_check():
    p=source_worker.source_check()
    need(p["only_english_keyword"]=="Hey Mero"
         and p["new_speaker_indices"]==[488,495]
         and p["new_speech_wav_budget"]==SOURCE_COUNT
         and p["source_roles"]=={"development":8}
         and p["new_optimizer_updates"]==0
         and p["new_asr_phone_calls"]==0
         and p["nearphone_is_eligible_ordinary"] is False
         and SOURCE_COUNT==32 and REVIEW_COUNT==32 and len(PROVIDERS)==5
         and SOURCE_COMMIT=="3af2027d2ca9d195e18b1cf8da8e07663d841bc6"
         and SOURCE_PARENT=="bb73e1742bc6ffb175c0eb3a3b2983b1b54b6917"
         and SOURCE["id"]==14543254675 and SOURCE["size"]==646524
         and SOURCE["digest"]=="sha256:530fccdd25abca9e76331dd8dad1f2440181158e140f2279ee952833a8d220f6",
         "FROZEN_NEW_LOCKED8_SOURCE32_FIVE_NATIVE_MODEL_LABEL_REVIEW")
    return {"status":"ORIGINAL_SOURCE_PINNED",
            "source_run":SOURCE_RUN,"source_artifact_id":SOURCE["id"],
            "source_rows":32,"review_rows":32,
            "observer_roster":list(PROVIDERS),
            "max_new_native_forwards_per_observer":32,
            "locked_test_label_review_clips":0,
            "kws_model_forwards":0,"kws_locked_test_events":0,
            "train_dev_calibration_review_clips":32,
            "new_tts_calls":0,"new_optimizer_updates":0,
            "training_allowed":False,"product_promotion_allowed":False}

def guard(env=None,event=None,parent=None):
    need(source_check()["status"]=="ORIGINAL_SOURCE_PINNED",
         "NO_NUMERICAL_EXECUTION_WITHOUT_VERIFIED_CLONE_ARCHIVE")
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"FIRST_PUBLIC_MAIN_ONLY")
    commit=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",commit) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "EXACT_NUMERIC_REVIEW_RUN")
    if event is None:event=json.loads(c.read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before")==BEFORE and event.get("after")==commit and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "ONE_EXACT_MAIN_SUCCESSOR_REVIEW")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                       cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"EXACT_FIRST_MAIN_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":commit,
            "source_run":SOURCE_RUN,"source_commit":SOURCE_COMMIT}

def verify_remote(token):
    guard()
    need(type(token) is str and len(token)>=10,"READONLY_GITHUB_ACTIONS_TOKEN")
    record=SOURCE
    req=urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/artifacts/{record['id']}",
        headers={"Authorization":"Bearer "+token,
                 "Accept":"application/vnd.github+json",
                 "User-Agent":"kws-pipernew-native-source-verified"})
    with urllib.request.urlopen(req,timeout=40) as stream:
        raw=stream.read(1000001)
    need(len(raw)<=1000000,"SMALL_SOURCE_METADATA")
    data=json.loads(raw)
    need(data.get("id")==record["id"] and
         data.get("name")==record["name"] and
         data.get("size_in_bytes")==record["size"] and
         data.get("digest")==record["digest"] and
         data.get("expired") is False and
         data.get("workflow_run",{}).get("id")==SOURCE_RUN and
         data.get("workflow_run",{}).get("head_sha")==SOURCE_COMMIT,
         "EXACT_AUTHENTIC_NEW_32_LOCKED_SOURCE_RUN")
    return {"artifact_id":record["id"],"digest":record["digest"]}

def source_rows(source):
    """SHA-audit all 32 original WAVs; only 8 NEW locked groups, no KWS."""
    source=Path(source)
    packet=json.loads(c.read(source/"MANIFEST.json"))
    need(packet.get("schema")==source_worker.SCHEMA
         and packet.get("run")==original_identity()
         and packet.get("source_plan_sha256")==source_worker.source_check()["plan_sha256"]
         and packet.get("denominator")==SOURCE_COUNT
         and packet.get("completed")==SOURCE_COUNT
         and packet.get("new_tts_calls")==SOURCE_COUNT
         and packet.get("new_asr_calls")==0 and packet.get("new_phone_calls")==0
         and packet.get("training_allowed") is False
         and packet.get("human_gold") is False
         and len(packet.get("rows",[]))==SOURCE_COUNT,
         "EXACT_NEW_PROSPECTIVE_DEV8_32_ORIGINAL_PCM_MANIFEST")
    source_worker.readback(source,original_identity())
    seen=set(); review=[]
    for row in packet["rows"]:
        need(row["source_group_id"] not in ("",None)
             and row["observed_class"] is None
             and row["training_allowed"] is False
             and row["human_gold"] is False
             and row["nearphone_is_background"] is False
             and row["pcm_sha256"] not in seen,
             "NO_DUPLICATED_ORIGINAL_PCM_OR_UNVERIFIED_LABEL")
        seen.add(row["pcm_sha256"]);review.append(row)
    from collections import Counter
    grouped=Counter(r["role"] for r in review)
    by_cell=Counter(r["cell_id"] for r in review)
    need(len(seen)==32 and len(review)==32
         and len({r["source_group_id"] for r in review})==8
         and grouped=={"development":32}
         and by_cell=={"mero-fast":8,"mero-slow":8,
                       "ordinary-time":8,"nearphone-meero-challenge":8}
         and {r["speaker_index"] for r in review}==set(range(488,496))
         and all(r["id"].startswith("merodev8-") for r in review)
         and all(r["requested_class"] is None
                 for r in review if r["cell_id"]=="nearphone-meero-challenge")
         and all(r["requested_class"]==0
                 for r in review if r["cell_id"]=="ordinary-time")
         and all(r["requested_class"]==2
                 for r in review if r["cell_id"].startswith("mero-")),
         "ONLY_EIGHT_NEW_PROSPECTIVE_DEV_SPEAKER_SOURCE_LABEL_QUALITY")
    return review

def identity(provider):
    need(provider in PROVIDERS,"ONLY_FIVE_FROZEN_NATIVE_OBSERVERS")
    if provider in ("xlsr","lv60"):
        return c.PHONE_MODELS[0 if provider=="xlsr" else 1]
    observer=load("english_pocket12_observe")
    return observer.identity(provider)

def observe(provider,source,out):
    need(provider in PROVIDERS,"ONLY_FIVE_FROZEN_NATIVE_OBSERVERS")
    event=guard()
    rows=source_rows(source)
    root=Path(out)
    need(root.is_absolute() and not root.exists(),"NEW_AUDIO_OBSERVER_PACKET_ONLY")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":SCHEMA,"provider":provider,
        "run":event,"source_run":SOURCE_RUN,"source_artifact_id":SOURCE["id"],
        "denominator":REVIEW_COUNT,"target_conditioned":False,"training_allowed":False})
    captures=[];error=None
    try:
        if provider in ("xlsr","lv60"):
            import numpy as np
            import torch
            from huggingface_hub import HfApi,snapshot_download
            from transformers import Wav2Vec2Config,Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            index=0 if provider=="xlsr" else 1
            name,rev=c.PHONE_MODELS[index]
            info=HfApi(token=False).model_info(name,revision=rev)
            need(info.sha.startswith(rev),"FROZEN_NATIVE_PHONE_COMMIT")
            cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("pipernew-phone-"+provider)
            folder=Path(snapshot_download(repo_id=name,revision=info.sha,
                         cache_dir=str(cache),max_workers=2,token=False))
            weights=folder/"pytorch_model.bin"
            need(weights.is_file() and weights.stat().st_size==1263535127,
                 "ORIGINAL_PHONE_WEIGHT_SIZE")
            h=hashlib.sha256()
            with weights.open("rb") as f:
                for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
            need(h.hexdigest()==phone_worker.WEIGHT_SHA[index],
                 "EXACT_NATIVE_PHONE_WEIGHT_SHA")
            vocab=phone_worker.read_verified_hf_metadata(folder,cache,"vocab.json")
            feature=Wav2Vec2FeatureExtractor.from_pretrained(str(folder),
                                              local_files_only=True)
            config=Wav2Vec2Config.from_json_file(str(folder/"config.json"))
            model=Wav2Vec2ForCTC(config)
            model.load_state_dict(torch.load(weights,map_location="cpu",
                                         weights_only=True),strict=True)
            model.eval()
            need(config.pad_token_id==0 and feature.sampling_rate==16000,
                 "NATIVE_COMPLETE_PCM16_RATE_AND_CTC_BLANK")
            c.record(root/"MODEL.json",{"model":name,"revision":info.sha,
                "weights_sha256":phone_worker.WEIGHT_SHA[index],
                "target_conditioned":False})
            c.record(root/"VOCAB.json",{"model":name,"revision":info.sha,
                "vocab":vocab,"pad_token_id":config.pad_token_id})
            (root/"logits").mkdir()
        else:
            observer=load("english_pocket12_observe")
            observer.source_check()
            assets=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("pipernew-asr-"+provider)
            backend,lock=observer._load(provider,assets,root)
            c.record(root/"MODEL.json",lock)
        for row in rows:
            sid=row["id"]
            wav=Path(source)/"wav"/(sid+".wav")
            measured=c.inspect_wav(wav)
            need(all(measured[k]==row[k] for k in ("wav_sha256","pcm_sha256","samples")),
                 "EVERY_NEW_PCM_UNALTERED_BEFORE_MODEL_FORWARD")
            directory=root/"rows"/sid
            directory.mkdir(parents=True,exist_ok=False)
            start={"id":sid,"provider":provider,
                "source_wav_sha256":row["wav_sha256"],
                "source_pcm_sha256":row["pcm_sha256"],
                "source_samples":row["samples"],
                "target_conditioned":False,"training_allowed":False}
            c.record(directory/"STARTED.json",start)
            try:
                if provider in ("xlsr","lv60"):
                    pcm,count=phone_worker.pcm_of(wav)
                    wave=np.frombuffer(pcm,dtype="<i2").astype("float32")/32768.0
                    inp=feature(wave,sampling_rate=16000,return_tensors="pt",
                                padding=False,return_attention_mask=True)
                    need(inp.input_values.shape[-1]==count and
                         inp.attention_mask.shape[-1]==count,
                         "COMPLETE_UNPADDED_NATIVE_INPUT")
                    with torch.inference_mode():
                        logits=model(input_values=inp.input_values,
                                  attention_mask=inp.attention_mask.int()).logits
                    raw=logits[0].detach().cpu().numpy().astype("<f4").tobytes()
                    tokens=phone_worker.ctc(logits.argmax(dim=-1)[0].tolist(),
                                            vocab,config.pad_token_id)
                    (root/"logits"/(sid+".f32")).write_bytes(raw)
                    result={**start,"status":"success",
                        "input_coverage_complete":True,
                        "raw_phone_tokens":tokens,
                        "logit_frames":logits.shape[1],
                        "vocab_size":logits.shape[2],
                        "logits_sha256":c.digest(raw)}
                else:
                    raw_text=backend(wav,directory)
                    need(type(raw_text) is str and 0<len(raw_text)<=4096 and
                         not any(ord(ch)<32 for ch in raw_text),
                         "RAW_UNPROMPTED_COMPLETE_ASR_TEXT")
                    result={**start,"status":"success",
                        "input_coverage_complete":True,"raw_transcript":raw_text}
                captures.append(result)
                c.record(directory/"CAPTURE.json",result)
            except Exception as exc:
                failed={**start,"status":"failed",
                        "error":type(exc).__name__+":"+str(exc)[:300]}
                captures.append(failed)
                c.record(directory/"FAILED.json",failed)
    except BaseException as exc:
        error=type(exc).__name__+":"+str(exc)[:300]
    finally:
        complete=sum(x["status"]=="success" for x in captures)
        c.record(root/"RESULT.json",{
            "schema":SCHEMA,"provider":provider,"run":event,
            "source_artifact_id":SOURCE["id"],
            "denominator":REVIEW_COUNT,"attempted":len(captures),
            "complete":complete,"missing_or_failed":REVIEW_COUNT-complete,
            "status":"COMPLETE" if complete==REVIEW_COUNT and error is None
                       else "PARTIAL_WITH_FAILURE_RECEIPTS" if error is None and len(captures)==REVIEW_COUNT
                       else "INCOMPLETE_NO_RETRY",
            "initialization_error":error,"rows":captures,
            "new_tts_calls":0,"new_asr_calls":len(captures)
                if provider in ("qwen","whisper","parakeet") else 0,
            "new_phone_calls":len(captures)
                if provider in ("xlsr","lv60") else 0,
            "target_conditioned":False,"human_gold":False,
            "training_allowed":False})
    need(error is None and len(captures)==REVIEW_COUNT and complete>=1,
         "REQUIRE_ALL_32_ATTEMPT_AND_EXPLICIT_PARTIAL_RECEIPTS")
    print(json.dumps({"status":"COMPLETE" if complete==REVIEW_COUNT else
       "PARTIAL_WITH_FAILURE_RECEIPTS","provider":provider,
       "successful_model_forwards":complete,"failed_receipts":REVIEW_COUNT-complete,
       "training_allowed":False,"kws_forward_calls":0},sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    flags=("--source-check","--guard","--verify-remote","--audit-source","--verify-source","--observe")
    for f in flags:p.add_argument(f,action="store_true")
    p.add_argument("--provider",choices=PROVIDERS)
    p.add_argument("--source")
    p.add_argument("--output")
    args=p.parse_args()
    need(sum(bool(getattr(args,n[2:].replace("-","_"))) for n in flags)==1,
         "ONE_EXACT_REVIEW_STAGE")
    if args.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if args.guard:
        print(json.dumps(guard(),sort_keys=True));return
    if args.verify_remote:
        print(json.dumps(verify_remote(os.environ.get("GH_TOKEN")),sort_keys=True));return
    need(args.source,"EXACT_SAVED_NEW_LOCKED8_32_ORIGINAL_WAV_REQUIRED")
    if args.audit_source:
        print(json.dumps({"status":"AUTHENTICATED_NEW_LOCKED8_32_NATIVE_LABEL_ONLY",
          "rows":len(source_rows(args.source)),
          "original_audio_artifact_id":SOURCE["id"],
          "new_numeric_model_forwards":0,"training_allowed":False},sort_keys=True))
        return
    if args.verify_source:
        guard()
        print(json.dumps({"status":"SOURCE_AUTHENTICATED",
              "rows":len(source_rows(args.source))},sort_keys=True))
        return
    need(args.provider and args.output,"ONE_OBSERVER_AND_NEW_OUTPUT")
    observe(args.provider,args.source,args.output)

if __name__=="__main__":
    main()
