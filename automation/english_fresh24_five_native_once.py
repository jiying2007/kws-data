"""Bounded five native blind observers for NEW fresh24 Qwen clone WAVs.

Research only. The 24 source presets, 12/4/4/4 roles and posterior rule are
frozen in the source campaign before source audio exists. No TTS or training.
Do not enable this worker until the exact 168-WAV public artifact is pinned.
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

source_worker=load("english_fresh24_generate_once")
policy=load("english_fresh24_ctc_admission")
c=load("english_hifi_micro")
phone_worker=load("english_hifi_phones_once")
SCHEMA="english-fresh24-native-five-observer-v1"
REPO="jiying2007/kws-data"
BEFORE="aca5d052daa78aa3c4f3f21dc462ab2a350d6efb"
SOURCE_RUN=44289263966
SOURCE_COMMIT=BEFORE
SOURCE_PARENT="dfbb95f003b3845439ccbe5e5eb4cb80e58e83f8"
SOURCE_COUNT=168
PROVIDERS=("qwen","whisper","parakeet","xlsr","lv60")
REFERENCE={
    "id":14419694053,
    "name":"english-fresh24-neutral-44289263966",
    "size":1368730,
    "digest":"sha256:87dd2bdb38205e8bc8d2a734c6ee398fafb119d584059fe028b2665bbeaa3e16"}
# This is intentionally a fail-closed, unpinned placeholder until new Qwen
# audio has actually completed. No model may be executed against a guess.
SOURCE={"id":0,"name":"english-fresh24-clone168-44289263966",
        "size":0,"digest":""}

def need(ok,reason):
    if not ok:raise ValueError(reason)

def original_identity():
    return {"run_id":str(SOURCE_RUN),"commit":SOURCE_COMMIT,"before":SOURCE_PARENT}

def source_check():
    design=source_worker.source_check()
    need(design["source_groups"]==24 and
         design["planned_neutral_calls"]==48 and
         design["planned_clone_calls"]==168 and
         design["training_allowed"] is False and
         policy.validate_protocol(source_worker.source_plan())["groups"]==24 and
         REFERENCE["id"]==14419694053 and
         len(PROVIDERS)==5 and len(set(PROVIDERS))==5,
         "FROZEN_24_NEW_SOURCE_ROSTER_AND_FIVE_NATIVE_MODELS")
    ready=(type(SOURCE["id"]) is int and SOURCE["id"]>0 and
           type(SOURCE["size"]) is int and SOURCE["size"]>0 and
           type(SOURCE["digest"]) is str and
           re.fullmatch("sha256:[0-9a-f]{64}",SOURCE["digest"]) is not None)
    # Retain an explicit fail-closed state until actual source pins exist.
    return {"status":"SOURCE_PINNED" if ready else "BLOCKED_UNPINNED_CLONE_ARTIFACT",
        "source_run":SOURCE_RUN,"new_tts_calls":0,
        "observer_roster":list(PROVIDERS),"source_rows":168,
        "max_new_forwards_per_observer":168,"training_allowed":False}

def guard(env=None,event=None,parent=None):
    need(source_check()["status"]=="SOURCE_PINNED",
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
    verified={}
    for name,record in (("reference",REFERENCE),("source",SOURCE)):
        req=urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/actions/artifacts/{record['id']}",
            headers={"Authorization":"Bearer "+token,
                "Accept":"application/vnd.github+json",
                "User-Agent":"kws-fresh24-five-native-pinned-source"})
        with urllib.request.urlopen(req,timeout=40) as stream:
            raw=stream.read(1000001)
        need(len(raw)<=1000000,"SMALL_AUTHENTICATED_GITHUB_META")
        got=json.loads(raw)
        need(got.get("id")==record["id"] and
             got.get("name")==record["name"] and
             got.get("size_in_bytes")==record["size"] and
             got.get("digest")==record["digest"] and
             got.get("expired") is False and
             got.get("workflow_run",{}).get("id")==SOURCE_RUN and
             got.get("workflow_run",{}).get("head_sha")==SOURCE_COMMIT,
             "EXACT_UNREPLAYED_168_CLONE_SOURCE:"+name)
        verified[name]={"artifact_id":record["id"],"digest":record["digest"]}
    return verified

def source_rows(source,neutral):
    """Independent full-PCM readback without calling original event guard."""
    design=source_worker.source_plan()
    plan_sha=source_worker.source_check()["source_plan_sha256"]
    neutral=Path(neutral);source=Path(source)
    refs=json.loads(c.read(neutral/"MANIFEST.json"))
    reqs=source_worker.neutral_requests(design)
    need(refs.get("schema")==source_worker.REFERENCE_SCHEMA and
         refs.get("run")==original_identity() and
         refs.get("frozen_plan_sha256")==plan_sha and
         refs.get("denominator")==24 and refs.get("completed")==24 and
         refs.get("new_neutral_model_calls")==48 and
         refs.get("human_gold") is False and
         refs.get("training_allowed") is False and
         len(refs.get("rows",[]))==24,
         "FROZEN_NEUTRAL_REFERENCE_EVENT_AND_COMPLETE_SOURCE")
    ref_rows={}
    seen=set()
    for actual,requested in zip(refs["rows"],reqs):
        need(all(actual.get(k)==v for k,v in requested.items()),
             "TWENTY_FOUR_EXACT_PRESPLIT_REFERENCE_ROLES")
        pcm=c.inspect_wav(neutral/"wav"/(requested["id"]+".wav"))
        need(pcm["basic_audio_ok"] and
             all(actual.get(k)==pcm[k] for k in ("wav_sha256","pcm_sha256","samples"))
             and actual["pcm_sha256"] not in seen,
             "ORIGINAL_UNCROPPED_REFERENCE_WAV_HASH")
        ref_rows[actual["id"]]=actual
        seen.add(actual["pcm_sha256"])
    expected=source_worker.clone_requests(ref_rows)
    packet=json.loads(c.read(source/"MANIFEST.json"))
    need(packet.get("schema")==source_worker.CLONE_SCHEMA and
         packet.get("run")==original_identity() and
         packet.get("frozen_plan_sha256")==plan_sha and
         packet.get("denominator")==168 and
         packet.get("completed")==168 and
         packet.get("new_clone_tts_calls")==168 and
         packet.get("new_asr_calls")==0 and
         packet.get("new_phone_calls")==0 and
         packet.get("training_allowed") is False and
         packet.get("human_gold") is False and
         len(packet.get("rows",[]))==168,
         "EXACT_NEW_168_GENERATED_BEFORE_ANY_REVIEW")
    result=[]
    for actual,requested in zip(packet["rows"],expected):
        need(all(actual.get(k)==v for k,v in requested.items())
             and actual.get("source_provider")=="qwen3-tts" and
             actual.get("human_gold") is False and
             actual.get("training_allowed") is False and
             actual.get("observed_class") is None,
             "EXACT_NEW_REQUEST_HAS_NO_AUTOMATIC_LABEL")
        pcm=c.inspect_wav(source/"wav"/(requested["id"]+".wav"))
        need(pcm["basic_audio_ok"] and
             all(actual.get(k)==pcm[k] for k in ("wav_sha256","pcm_sha256","samples"))
             and actual["pcm_sha256"] not in seen and
             actual.get("positive_duration_pass")==
                 (actual["requested_class"] not in (1,2)
                  or actual["samples"]<=policy.MAX_POSITIVE_SAMPLES),
             "NEW_FULL_UNCROPPED_PCM_AND_DURATION")
        seen.add(actual["pcm_sha256"])
        result.append(actual)
    need(len(seen)==192 and
         {r["source_group_id"] for r in result}==
         {r["source_group_id"] for r in reqs},
         "NO_CROSS_GROUP_DUPLICATE_OR_SOURCE_DRIFT")
    return result

def identity(provider):
    need(provider in PROVIDERS,"ONLY_FIVE_FROZEN_NATIVE_OBSERVERS")
    if provider in ("xlsr","lv60"):
        return c.PHONE_MODELS[0 if provider=="xlsr" else 1]
    observer=load("english_pocket12_observe")
    return observer.identity(provider)

def observe(provider,source,neutral,out):
    need(provider in PROVIDERS,"ONLY_FIVE_FROZEN_NATIVE_OBSERVERS")
    event=guard()
    rows=source_rows(source,neutral)
    root=Path(out)
    need(root.is_absolute() and not root.exists(),"NEW_AUDIO_OBSERVER_PACKET_ONLY")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":SCHEMA,"provider":provider,
        "run":event,"source_run":SOURCE_RUN,"source_artifact_id":SOURCE["id"],
        "denominator":168,"target_conditioned":False,"training_allowed":False})
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
            cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("fresh24-phone-"+provider)
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
            assets=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("fresh24-asr-"+provider)
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
            "denominator":168,"attempted":len(captures),
            "complete":complete,"missing_or_failed":168-complete,
            "status":"COMPLETE" if complete==168 and error is None
                       else "INCOMPLETE_NO_RETRY",
            "initialization_error":error,"rows":captures,
            "new_tts_calls":0,"new_asr_calls":len(captures)
                if provider in ("qwen","whisper","parakeet") else 0,
            "new_phone_calls":len(captures)
                if provider in ("xlsr","lv60") else 0,
            "target_conditioned":False,"human_gold":False,
            "training_allowed":False})
    need(error is None and complete==168 and len(captures)==168,
         "NO_FAKE_OBSERVER_COMPLETION_OR_MODEL_RETRY")
    print(json.dumps({"status":"COMPLETE","provider":provider,
       "actual_forwards":complete,"training_allowed":False},sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    flags=("--source-check","--guard","--verify-remote","--verify-source","--observe")
    for f in flags:p.add_argument(f,action="store_true")
    p.add_argument("--provider",choices=PROVIDERS)
    p.add_argument("--source")
    p.add_argument("--reference")
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
    need(args.source and args.reference,"BOTH_EXACT_SAVED_AUDIO_ARTIFACTS_REQUIRED")
    if args.verify_source:
        guard()
        print(json.dumps({"status":"SOURCE_AUTHENTICATED",
              "rows":len(source_rows(args.source,args.reference))},sort_keys=True))
        return
    need(args.provider and args.output,"ONE_OBSERVER_AND_NEW_OUTPUT")
    observe(args.provider,args.source,args.reference,args.output)

if __name__=="__main__":
    main()
