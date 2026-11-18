"""Once-only independent five-observer review of the 25 saved native Melo WAVs.

Source authenticated before model loading. Five independent observers see only
whole 16 kHz PCM, never the requested transcript or pronunciation. No TTS,
training, human-gold, speaker-group split, or automatic promotion.
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
# The verified existing ASR adapter imports sibling modules by their ordinary
# names. -I removes the script directory from sys.path; restore only this
# exact resolved, committed automation directory, never cwd/user paths.
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("melo_review_"+name,HERE/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
c=load("english_hifi_micro")
m=load("english_melo_cmu_once")
p=load("english_hifi_phones_once")
SCHEMA="english-melo-cmu-five-blind-observer-v1"
REPO="jiying2007/kws-data"
BEFORE="8d37825e8fd579bb9b83d1a118f080622aacf07d"
SOURCE_RUN=44271169127
SOURCE_COMMIT="3078a463e6a75de2c78e36b4302c8697bcf12477"
PRIOR_REVIEW_RUN=44274204197
PRIOR_REVIEW_COMMIT=BEFORE
RECOVER_PROVIDERS=("qwen","whisper","parakeet")
PRIOR_ARTIFACTS={
    "qwen":(14413149243,922,
        "sha256:2abae4cb0f62bc44bea30a01c8f21bccefe74c2be15faee6a342527f6d5b95d5"),
    "whisper":(14413807688,926,
        "sha256:44b43351b2c758a6750c157ff9b59fd119b4cd072ce9fc0d900a91ea568c3885"),
    "parakeet":(14412849452,927,
        "sha256:5fd518a94a6d402c1286a832a0776495cf535f2e7033d783c2f2a16264f33c59"),
    "xlsr":(14414251906,2384274,
        "sha256:9fc2d149fcb00c5094fe34a5b43e0092cda48ec691f08ee20477ec0140c67bd3"),
    "lv60":(14412894522,2380173,
        "sha256:3cf77f2c0b9711f65ae1e53932d01a98d387b95cd0ffa9664618d1bc63e521fa")
}
ARTIFACT_ID=14411564380
ARTIFACT_NAME="english-melo-cmu-source-44271169127"
ARTIFACT_BYTES=651511
ARTIFACT_DIGEST="sha256:add3754cde2b17c7cd95976ab317b6ecd2e4a4c45b89c90705944fa28cbc7d43"
PROVIDERS=("qwen","whisper","parakeet","xlsr","lv60")
SOURCE_COUNT=25

def need(ok,why):
    if not ok:raise ValueError(why)

def source_check():
    need(m.SCHEMA=="english-melo-cmu-native-phone-25-v1" and
         m.source_check()["planned_new_tts_calls"]==25 and
         len(m.planned())==25 and m.MAX_SYNTH==25,
         "ORIGINAL_SOURCE_CELLS_AND_NO_GENERATION")
    need(len(PROVIDERS)==5 and len(set(PROVIDERS))==5 and
         SOURCE_COMMIT=="3078a463e6a75de2c78e36b4302c8697bcf12477" and
         BEFORE=="8d37825e8fd579bb9b83d1a118f080622aacf07d" and
         PRIOR_REVIEW_RUN==44274204197 and
         ARTIFACT_ID>0 and ARTIFACT_DIGEST.startswith("sha256:") and
         ARTIFACT_BYTES>0 and
         set(PRIOR_ARTIFACTS)==set(PROVIDERS) and
         len({pin[0] for pin in PRIOR_ARTIFACTS.values()})==5,
         "FROZEN_FIRST_REVIEW_AND_SOURCE")
    return {"status":"SOURCE_ONLY","source_run":SOURCE_RUN,
            "source_artifact_id":ARTIFACT_ID,"source_rows":25,
            "model_calls":0,"new_tts":0,"training_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    expected={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in expected.items()),
         "FIRST_PUBLIC_MAIN_ONLY")
    sha=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "EXACT_RUN_ID")
    if event is None:event=json.loads(c.read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before")==BEFORE and event.get("after")==sha and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "EXACT_PROTECTED_SUCCESSOR")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"EXACT_FIRST_PARENT")
    return {"commit":sha,"run_id":env["GITHUB_RUN_ID"],
            "source_run":SOURCE_RUN,"source_commit":SOURCE_COMMIT}

def original_review_identity():
    return {"commit":PRIOR_REVIEW_COMMIT,"run_id":str(PRIOR_REVIEW_RUN),
            "source_run":SOURCE_RUN,"source_commit":SOURCE_COMMIT}

def verify_prior_failed(root,provider):
    """Only an authenticated zero-forward original failure can be resumed."""
    need(provider in RECOVER_PROVIDERS,"ONLY_THREE_FAILED_ASR")
    root=Path(root)
    need(root.is_dir() and not root.is_symlink() and
         {p.name for p in root.iterdir()}=={"STARTED.json","RESULT.json"} and
         all(not p.is_symlink() for p in root.iterdir()),
         "EXACT_ZERO_FORWARD_RECEIPT_FILES")
    start=json.loads(c.read(root/"STARTED.json"))
    result=json.loads(c.read(root/"RESULT.json"))
    need(start.get("schema")==SCHEMA and
         start.get("run")==original_review_identity() and
         start.get("provider")==provider and
         start.get("source_run")==SOURCE_RUN and
         start.get("source_artifact_id")==ARTIFACT_ID and
         start.get("denominator")==25 and
         start.get("target_conditioned") is False and
         start.get("training_allowed") is False,
         "ORIGINAL_FIRST_REVIEW_STARTED_AND_NO_TRAINING")
    need(result.get("schema")==SCHEMA and result.get("run")==original_review_identity() and
         result.get("provider")==provider and
         result.get("source_artifact_id")==ARTIFACT_ID and
         result.get("status")=="INCOMPLETE_NO_RETRY" and
         result.get("denominator")==25 and
         result.get("attempted")==0 and result.get("complete")==0 and
         result.get("missing_or_failed")==25 and result.get("rows")==[] and
         result.get("new_tts_calls")==0 and result.get("new_asr_calls")==0 and
         result.get("new_phone_calls")==0 and
         result.get("initialization_error")==
             "ModuleNotFoundError:No module named 'english_seven_voice_asr_once'" and
         result.get("human_gold") is False and
         result.get("training_allowed") is False,
         "ORIGINAL_NATIVE_ASR_ZERO_FORWARD_OR_UNEXPECTED_FAILURE")
    return {"status":"EXACT_ZERO_FORWARD_RECOVERY_AUTHENTICATED",
            "provider":provider,"original_calls":0,
            "original_run":PRIOR_REVIEW_RUN,
            "prior_artifact_id":PRIOR_ARTIFACTS[provider][0]}

def verify_remote(token):
    need(isinstance(token,str) and len(token)>=10,
         "READONLY_GITHUB_ACTIONS_TOKEN")
    url=f"https://api.github.com/repos/{REPO}/actions/artifacts/{ARTIFACT_ID}"
    req=urllib.request.Request(url,headers={
        "Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json",
        "User-Agent":"melo-cmu-pinned-five-blind-review"})
    with urllib.request.urlopen(req,timeout=30) as response:
        raw=response.read(1000001)
    need(0<len(raw)<=1000000,"SMALL_SIGNED_ARTIFACT_METADATA")
    value=json.loads(raw)
    need(value.get("id")==ARTIFACT_ID and
         value.get("name")==ARTIFACT_NAME and
         value.get("size_in_bytes")==ARTIFACT_BYTES and
         value.get("digest")==ARTIFACT_DIGEST and
         value.get("expired") is False and
         value.get("workflow_run",{}).get("id")==SOURCE_RUN and
         value.get("workflow_run",{}).get("head_sha")==SOURCE_COMMIT,
         "EXACT_VERIFIED_SOURCE_ARTIFACT")
    observed={"source":{"artifact_id":ARTIFACT_ID,"digest":ARTIFACT_DIGEST}}
    for provider,(artifact_id,size,digest) in PRIOR_ARTIFACTS.items():
        request=urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/actions/artifacts/{artifact_id}",
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"melo-cmu-original-evidence-preservation"})
        with urllib.request.urlopen(request,timeout=30) as response:
            prior=json.loads(response.read(1000001))
        need(prior.get("id")==artifact_id and
             prior.get("name")=="english-melo-cmu-observer-"+provider and
             prior.get("size_in_bytes")==size and
             prior.get("digest")==digest and
             prior.get("expired") is False and
             prior.get("workflow_run",{}).get("id")==PRIOR_REVIEW_RUN and
             prior.get("workflow_run",{}).get("head_sha")==PRIOR_REVIEW_COMMIT,
             "EXACT_ORIGINAL_OBSERVER_ARTIFACT:"+provider)
        observed[provider]={"artifact_id":artifact_id,"digest":digest}
    return observed

def source_rows(root):
    root=Path(root)
    authenticated=m.readback(root)
    need(authenticated=={"status":"SOURCE_AUTHENTICATED","rows":25,
                        "training_allowed":False,"new_labels":0},
         "ORIGINAL_FULL_PCM_SOURCE_VERIFICATION")
    source=json.loads(c.read(root/"MANIFEST.json"))
    started=json.loads(c.read(root/"STARTED.json"))
    lock=json.loads(c.read(root/"PHONEME_INPUT_LOCK.json"))
    need(started.get("run",{}).get("run_id")==str(SOURCE_RUN) and
         started.get("run",{}).get("commit")==SOURCE_COMMIT and
         started.get("planned_calls")==25 and
         started.get("training_allowed") is False,
         "SOURCE_RUN_IDENTITY")
    need(source.get("completed")==25 and source.get("new_tts_calls")==25
         and source.get("source_revision")==m.UPSTREAM_SHA and
         source.get("bert_revision")==m.BERT[1] and
         source.get("independent_asr")=="NOT_RUN" and
         source.get("independent_phone")=="NOT_RUN",
         "ORIGINAL_NEW_NUMERICAL_TTS_ONLY")
    need(lock.get("source_code_sha")==m.UPSTREAM_SHA and
         lock.get("official_model_sha")==m.MODEL[1] and
         lock.get("official_weight_sha256")==m.CHECKPOINT_SHA and
         lock.get("bert_sha")==m.BERT[1] and
         lock.get("requested_text_is_ground_truth") is False,
         "ACTUAL_NATIVE_PHONEME_INPUT_LOCK")
    for word,phones in (("Mero",["m","eh","r","ow"]),
                        ("Vosen",["v","ow","s","ah","n"])):
        entry=lock.get("input_phones",{}).get(word,{})
        need(entry.get("requested_critical")==phones and
             type(entry.get("matched_start_index")) is int and
             entry["matched_start_index"]>=0,
             "FROZEN_ACTUAL_UPSTREAM_PHONE_INPUT:"+word)
    rows=source["rows"]
    need(len({row["pcm_sha256"] for row in rows})==25 and
         all(row["requested_text_is_ground_truth"] is False and
             row["training_allowed"] is False and
             row["human_gold"] is False and
             row["role"]=="research-train-only" for row in rows),
         "NOT_TWELVE_INDEPENDENT_SPEAKER_GROUPS")
    return rows

def model_id(provider):
    need(provider in PROVIDERS,"FROZEN_PROVIDER")
    if provider in ("xlsr","lv60"):
        return c.PHONE_MODELS[0 if provider=="xlsr" else 1]
    observer=load("english_pocket12_observe")
    return observer.identity(provider)

def run(provider,source,out):
    need(provider in RECOVER_PROVIDERS,"NO_REPLAY_OF_COMPLETED_PHONEME_OBSERVERS")
    identity=guard()
    rows=source_rows(source)
    root=Path(out)
    need(root.is_absolute() and not root.exists(),"CREATE_NEW_OBSERVER_ROOT_ONLY")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":SCHEMA,"run":identity,
       "provider":provider,"source_run":SOURCE_RUN,"source_artifact_id":ARTIFACT_ID,
       "denominator":SOURCE_COUNT,"target_conditioned":False,
       "human_gold":False,"training_allowed":False})
    captures=[];failure=None
    try:
        if provider in ("xlsr","lv60"):
            import numpy as np
            import torch
            from huggingface_hub import HfApi,snapshot_download
            from transformers import Wav2Vec2Config,Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            idx=0 if provider=="xlsr" else 1
            name,revision=c.PHONE_MODELS[idx]
            info=HfApi(token=False).model_info(name,revision=revision)
            need(info.sha.startswith(revision),"PINNED_PHONE_REVISION")
            cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("melo-private-"+provider+"-cache")
            folder=Path(snapshot_download(repo_id=name,revision=info.sha,
                        cache_dir=str(cache),max_workers=2,token=False))
            weights=folder/"pytorch_model.bin"
            need(weights.is_file() and weights.stat().st_size==1263535127,
                 "NATIVE_PHONE_MODEL_WEIGHT_SIZE")
            h=hashlib.sha256()
            with weights.open("rb") as stream:
                for piece in iter(lambda:stream.read(1048576),b""):h.update(piece)
            need(h.hexdigest()==p.WEIGHT_SHA[idx],"NATIVE_PHONE_WEIGHT_SHA")
            vocab=p.read_verified_hf_metadata(folder,cache,"vocab.json")
            feature=Wav2Vec2FeatureExtractor.from_pretrained(str(folder),local_files_only=True)
            config=Wav2Vec2Config.from_json_file(str(folder/"config.json"))
            model=Wav2Vec2ForCTC(config)
            model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True),
                                  strict=True)
            model.eval()
            need(config.pad_token_id==0 and feature.sampling_rate==16000,
                 "NATIVE_PHONE_INPUT_CONFIG")
            c.record(root/"MODEL.json",{"model":name,"revision":info.sha,
                 "weights_sha256":p.WEIGHT_SHA[idx],"target_conditioned":False})
            c.record(root/"VOCAB.json",{"model":name,"revision":info.sha,
                 "vocab":vocab,"pad_token_id":config.pad_token_id})
            (root/"logits").mkdir()
        else:
            observer=load("english_pocket12_observe")
            observer.source_check()
            asset_dir=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("melo-private-"+provider+"-assets")
            backend,model_lock=observer._load(provider,asset_dir,root)
            c.record(root/"MODEL.json",model_lock)
        for row in rows:
            sid=row["id"]
            wav=Path(source)/"wav"/(sid+".wav")
            original=c.inspect_wav(wav)
            need(all(original[key]==row[key] for key in
                     ("wav_sha256","pcm_sha256","samples")),
                 "UNCHANGED_WAV_BEFORE_BLIND_MODEL_CALL")
            dirname=root/"rows"/sid
            dirname.mkdir(parents=True,exist_ok=False)
            start={"id":sid,"provider":provider,
                   "source_pcm_sha256":row["pcm_sha256"],
                   "source_wav_sha256":row["wav_sha256"],
                   "source_samples":row["samples"],
                   "target_conditioned":False,"training_allowed":False}
            c.record(dirname/"STARTED.json",start)
            try:
                if provider in ("xlsr","lv60"):
                    pcm,count=p.pcm_of(wav)
                    values=np.frombuffer(pcm,dtype="<i2").astype("float32")/32768.0
                    inputs=feature(values,sampling_rate=16000,return_tensors="pt",
                                   padding=False,return_attention_mask=True)
                    need(inputs.input_values.shape[-1]==count and
                         inputs.attention_mask.shape[-1]==count,
                         "FULL_PCM_UNPADDED_INPUT")
                    with torch.inference_mode():
                        tensor=model(input_values=inputs.input_values,
                                     attention_mask=inputs.attention_mask.int()).logits
                    raw=tensor[0].detach().cpu().numpy().astype("<f4").tobytes()
                    tokens=p.ctc(tensor.argmax(dim=-1)[0].tolist(),vocab,config.pad_token_id)
                    (root/"logits"/(sid+".f32")).write_bytes(raw)
                    captured={**start,"status":"success","input_coverage_complete":True,
                        "raw_phone_tokens":tokens,"logit_frames":tensor.shape[1],
                        "vocab_size":tensor.shape[2],"logits_sha256":c.digest(raw)}
                else:
                    phrase=backend(wav,dirname)
                    need(isinstance(phrase,str) and 0<len(phrase)<=4096 and
                         not any(ord(ch)<32 for ch in phrase),
                         "COMPLETE_BLIND_ASR_TEXT")
                    captured={**start,"status":"success","input_coverage_complete":True,
                        "raw_transcript":phrase}
                captures.append(captured)
                c.record(dirname/"CAPTURE.json",captured)
            except Exception as exc:
                failed={**start,"status":"failed",
                        "error":type(exc).__name__+":"+str(exc)[:300]}
                captures.append(failed)
                c.record(dirname/"FAILED.json",failed)
    except BaseException as exc:
        failure=type(exc).__name__+":"+str(exc)[:300]
    finally:
        complete=sum(x["status"]=="success" for x in captures)
        result={"schema":SCHEMA,"provider":provider,"run":identity,
            "source_artifact_id":ARTIFACT_ID,"denominator":SOURCE_COUNT,
            "attempted":len(captures),"complete":complete,
            "missing_or_failed":SOURCE_COUNT-complete,
            "status":"COMPLETE" if complete==SOURCE_COUNT and failure is None
                       else "INCOMPLETE_NO_RETRY",
            "initialization_error":failure,"rows":captures,
            "new_tts_calls":0,
            "new_phone_calls":len(captures) if provider in ("xlsr","lv60") else 0,
            "new_asr_calls":len(captures) if provider in ("qwen","whisper","parakeet") else 0,
            "human_gold":False,"training_allowed":False}
        c.record(root/"RESULT.json",result)
    if failure is not None or len(captures)!=complete or complete!=SOURCE_COUNT:
        raise RuntimeError("INCOMPLETE_FIVE_OBSERVER_NO_RETRY:"+str(failure))
    print(json.dumps({"status":"COMPLETE","provider":provider,
        "actual_forwards":complete,"new_tts_calls":0,"training_allowed":False},
        sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--guard",action="store_true")
    ap.add_argument("--verify-remote",action="store_true")
    ap.add_argument("--verify-source",action="store_true")
    ap.add_argument("--verify-prior-failed",action="store_true")
    ap.add_argument("--prior")
    ap.add_argument("--run",action="store_true")
    ap.add_argument("--provider",choices=PROVIDERS)
    ap.add_argument("--source")
    ap.add_argument("--output")
    a=ap.parse_args()
    need(sum(bool(x) for x in (a.source_check,a.guard,a.verify_remote,
                               a.verify_source,a.verify_prior_failed,a.run))==1,
         "ONE_MODE_ONLY")
    if a.source_check:print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:print(json.dumps(guard(),sort_keys=True));return
    if a.verify_remote:
        guard()
        print(json.dumps(verify_remote(os.environ.get("GH_TOKEN"))))
        return
    if a.verify_prior_failed:
        guard()
        need(a.prior and a.provider,"FAILED_PRIOR_AND_PROVIDER_REQUIRED")
        print(json.dumps(verify_prior_failed(a.prior,a.provider),sort_keys=True))
        return
    need(a.source,"SAVED_SOURCE_REQUIRED")
    if a.verify_source:
        guard()
        print(json.dumps({"status":"SOURCE_AUTHENTICATED","rows":len(source_rows(a.source))}))
        return
    need(a.provider and a.output,"EXACT_SOURCE_PROVIDER_AND_OUTPUT_REQUIRED")
    run(a.provider,a.source,a.output)

if __name__=="__main__":main()
