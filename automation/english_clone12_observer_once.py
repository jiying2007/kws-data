"""Independent real ASR/phoneme forward reader for saved 60-clip clone corpus.

Five original frozen providers, one provider per public-main job. Not a TTS
runner; the same original WAV bytes are authenticated before ANY inference.
No prompt/label/target is supplied to the ASR or phoneme model. Partial
failures are UNKNOWN and never retried or silently removed from denominator.
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
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
def load(name):
    spec=importlib.util.spec_from_file_location("clone12_observer_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
c=load("english_hifi_micro")
p=load("english_hifi_phones_once")
cohort=load("english_clone_cohort12_once")
policy=load("english_clone12_admission_policy")
PLAN=HERE/"english_clone12_evidence_plan.json"
PROVIDERS=("qwen","whisper","parakeet","xlsr","lv60")
SCHEMA="english-clone12-blind-five-observer-v1"
SOURCE_COUNT=60

def need(x,reason):
    if not x:raise ValueError(reason)
def read_plan():
    x=json.loads(c.read(PLAN))
    need(x.get("schema")=="english-clone12-independent-evidence-plan-v1" and
         x.get("repository")=="jiying2007/kws-data" and
         type(x.get("enabled")) is bool and
         x.get("main_before")=="337ae4f4f318483696a28f0ebc635877b623606a" and
         x.get("retry_count")==0 and
         x.get("maximum_asr_per_provider")==60 and
         x.get("maximum_phone_per_provider")==60 and
         x.get("independent_providers")==list(PROVIDERS) and
         x.get("source_selection")=="FROZEN_ALL_60_ROWS" and
         x.get("requested_text_is_truth") is False and
         x.get("training_allowed") is False and
         x.get("human_gold") is False,
         "IMMUTABLE_SIXTY_SOURCE_EVIDENCE_PLAN")
    s=x["source"]
    need(s["run_id"]==44257296925 and
         s["commit"]=="337ae4f4f318483696a28f0ebc635877b623606a" and
         s["artifact_name"]=="english-clone12-corpus-44257296925",
         "FROZEN_SOURCE_RUN_AND_HEAD")
    if x["enabled"]:
        need(type(s.get("artifact_id")) is int and s["artifact_id"]>0 and
             type(s.get("artifact_bytes")) is int and s["artifact_bytes"]>0 and
             isinstance(s.get("artifact_digest"),str) and
             re.fullmatch("sha256:[0-9a-f]{64}",s["artifact_digest"]),
             "EXACT_AUTHENTICATED_GENERATED_ARTIFACT_REQUIRED")
    rr=x["original_references"]
    need(rr=={"run_id":44256271861,
         "commit":"7c352e15466dd62b21a4733e96f08d51d938b647",
         "artifact_id":14406282094,
         "artifact_name":"english-neutral-clone-reference12-44256271861",
         "artifact_bytes":652901,
         "artifact_digest":"sha256:8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5"},
         "PINNED_SYNTHETIC_REFERENCE_ARTIFACT")
    return x

def source_check():
    x=read_plan()
    need(cohort.source_contract()["planned_new_tts_calls"]==60,
         "SAME_FROZEN_GENERATION_SOURCE")
    return {"status":"SOURCE_ONLY","enabled":x["enabled"],
            "candidate_source_rows":60,"independent_observers":5,
            "new_numeric_calls":0,"training_allowed":False}

def guard(env=None,event=None,parent=None):
    x=read_plan()
    need(x["enabled"] is True,"UNPINNED_SOURCE_DO_NOT_EXECUTE")
    env=os.environ if env is None else env
    expected={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":x["repository"],
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in expected.items()),"PUBLIC_FIRST_MAIN_ONLY")
    sha=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "EXACT_EVENT_IDENTITY")
    if event is None:event=json.loads(c.read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before")==x["main_before"] and event.get("after")==sha and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==x["repository"] and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "EXACT_MAIN_SUCCESSOR_ONCE")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                       cwd=c.ROOT,text=True).strip()
    need(parent==x["main_before"],"FIRST_PARENT_SHA")
    return {"commit":sha,"run_id":env["GITHUB_RUN_ID"],
            "source_run":x["source"]["run_id"]}

def remote_metadata(token):
    x=read_plan()
    need(x["enabled"] is True and isinstance(token,str) and len(token)>=10,
         "READONLY_ACTIONS_TOKEN")
    observed={}
    for name,record in (("source",x["source"]),("refs",x["original_references"])):
        artifact=record["artifact_id"]
        url=f"https://api.github.com/repos/{x['repository']}/actions/artifacts/{artifact}"
        req=urllib.request.Request(url,headers={
            "Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "User-Agent":"english-clone12-five-observer-pinned-source"})
        with urllib.request.urlopen(req,timeout=30) as response:
            raw=response.read(1_000_001)
        need(0<len(raw)<=1_000_000,"SMALL_SIGNED_SOURCE_METADATA")
        meta=json.loads(raw)
        need(meta.get("id")==artifact and
             meta.get("name")==record["artifact_name"] and
             meta.get("expired") is False and
             meta.get("size_in_bytes")==record["artifact_bytes"] and
             meta.get("digest")==record["artifact_digest"] and
             meta.get("workflow_run",{}).get("id")==record["run_id"] and
             meta.get("workflow_run",{}).get("head_sha")==record["commit"],
             "FROZEN_GENERATION_AND_DONOR_IDENTITY:"+name)
        observed[name]={"artifact_id":artifact,"digest":record["artifact_digest"]}
    return observed

def trusted_source(source,refs):
    x=read_plan()
    need(x["enabled"] is True,"SOURCE_NOT_PINNED")
    references=cohort.reference_rows(refs)
    result=cohort.readback(source,references)
    need(result["new_source_rows"]==60 and result["training_allowed"] is False,
         "NO_GENERATION_LABEL_ADMISSION")
    manifest=json.loads(c.read(Path(source)/"MANIFEST.json"))
    need(manifest.get("run",{}).get("run_id")==str(x["source"]["run_id"]) and
         manifest.get("run",{}).get("commit")==x["source"]["commit"],
         "SOURCE_RUN_COMMIT_PINNED_IN_MANIFEST")
    return manifest["rows"]

def model_id(provider):
    need(provider in PROVIDERS,"FIVE_FROZEN_OBSERVERS_ONLY")
    if provider in ("xlsr","lv60"):
        return c.PHONE_MODELS[0 if provider=="xlsr" else 1]
    import english_pocket12_observe as observe
    return observe.identity(provider)

def run(provider,source,refs,output):
    need(provider in PROVIDERS,"FIVE_FROZEN_PROVIDER_SET")
    identity=guard()
    rows=trusted_source(source,refs)
    root=Path(output)
    need(root.is_absolute() and not root.exists(),"NEW_OBSERVER_PACKET_ONLY")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{
        "schema":SCHEMA,"run":identity,"provider":provider,
        "source_artifact_id":read_plan()["source"]["artifact_id"],
        "source_reference_artifact_id":14406282094,
        "denominator":60,"max_forwards":60,"training_allowed":False,
        "target_conditioned":False,"label_basis":"NONE"})
    captures=[]
    error=None
    try:
        if provider in ("xlsr","lv60"):
            import numpy as np
            import torch
            from huggingface_hub import HfApi,snapshot_download
            from transformers import Wav2Vec2Config,Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            idx=0 if provider=="xlsr" else 1
            name,rev=c.PHONE_MODELS[idx]
            remote=HfApi(token=False).model_info(name,revision=rev)
            need(remote.sha.startswith(rev),"FROZEN_PHONE_COMMIT")
            cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("private-"+provider+"-cache")
            folder=Path(snapshot_download(repo_id=name,revision=remote.sha,
                  cache_dir=str(cache),max_workers=2,token=False))
            weights=folder/"pytorch_model.bin"
            need(weights.is_file() and weights.stat().st_size==1263535127,
                 "NATIVE_PHONE_WEIGHT_SIZE")
            h=hashlib.sha256()
            with weights.open("rb") as stream:
                for b in iter(lambda:stream.read(1048576),b""):
                    h.update(b)
            need(h.hexdigest()==p.WEIGHT_SHA[idx],"ORIGINAL_NATIVE_PHONE_WEIGHT_SHA")
            vocab=p.read_verified_hf_metadata(folder,cache,"vocab.json")
            feat=Wav2Vec2FeatureExtractor.from_pretrained(str(folder),
                                                       local_files_only=True)
            config=Wav2Vec2Config.from_json_file(str(folder/"config.json"))
            model=Wav2Vec2ForCTC(config)
            model.load_state_dict(torch.load(weights,map_location="cpu",
                                               weights_only=True),strict=True)
            model.eval()
            need(config.pad_token_id==0 and feat.sampling_rate==16000,
                 "ORIGINAL_NATIVE_PHONE_MODEL_CONFIG")
            lock={"model":name,"revision":remote.sha,
                  "weights_sha256":p.WEIGHT_SHA[idx],
                  "input":"16k complete float32 from verified PCM16",
                  "target_conditioned":False}
            c.record(root/"MODEL.json",lock)
            c.record(root/"VOCAB.json",{"model":name,"revision":remote.sha,
                                      "vocab":vocab,"pad_token_id":config.pad_token_id})
            (root/"logits").mkdir()
        else:
            import english_pocket12_observe as observer
            observer.source_check()
            assets=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("private-"+provider+"-assets")
            backend,lock=observer._load(provider,assets,root)
            c.record(root/"MODEL.json",lock)
        for row in rows:
            sid=row["id"]
            wav=Path(source)/"wav"/(sid+".wav")
            obs=c.inspect_wav(wav)
            need(all(obs[k]==row[k] for k in ("wav_sha256","pcm_sha256","samples")),
                 "EXACT_GENERATED_AUDIO_HASH")
            (root/"rows"/sid).mkdir(parents=True,exist_ok=False)
            started={"id":sid,"provider":provider,"input_wav_sha256":obs["wav_sha256"],
                     "input_pcm_sha256":obs["pcm_sha256"],
                     "source_samples":obs["samples"],"target_conditioned":False,
                     "training_allowed":False}
            c.record(root/"rows"/sid/"STARTED.json",started)
            try:
                if provider in ("xlsr","lv60"):
                    pcm,count=p.pcm_of(wav)
                    values=np.frombuffer(pcm,dtype="<i2").astype("float32")/32768
                    inp=feat(values,sampling_rate=16000,return_tensors="pt",
                              padding=False,return_attention_mask=True)
                    need(inp.input_values.shape[-1]==count and
                         inp.attention_mask.shape[-1]==count,
                         "WHOLE_PCM_UNPADDED_OBSERVER_INPUT")
                    with torch.inference_mode():
                        logits=model(input_values=inp.input_values,
                                     attention_mask=inp.attention_mask.int()).logits
                    raw=logits[0].detach().cpu().numpy().astype("<f4").tobytes()
                    ids=logits.argmax(dim=-1)[0].tolist()
                    tokens=p.ctc(ids,vocab,config.pad_token_id)
                    (root/"logits"/(sid+".f32")).write_bytes(raw)
                    result={**started,"status":"success",
                       "input_coverage_complete":True,
                       "raw_phone_tokens":tokens,
                       "logit_frames":logits.shape[1],
                       "vocab_size":logits.shape[2],
                       "logits_sha256":c.digest(raw)}
                else:
                    out=backend(wav,root/"rows"/sid)
                    need(isinstance(out,str) and 0<len(out)<=4096 and
                         not any(ord(ch)<32 for ch in out),
                         "RAW_COMPLETE_BLIND_ASR_TEXT")
                    result={**started,"status":"success",
                        "input_coverage_complete":True,
                        "raw_transcript":out}
                captures.append(result)
                c.record(root/"rows"/sid/"CAPTURE.json",result)
            except Exception as exc:
                failed={**started,"status":"failed",
                    "error":type(exc).__name__+":"+str(exc)[:300]}
                captures.append(failed)
                c.record(root/"rows"/sid/"FAILED.json",failed)
    except BaseException as exc:
        error=type(exc).__name__+":"+str(exc)[:300]
        raise
    finally:
        complete=sum(x["status"]=="success" for x in captures)
        c.record(root/"RESULT.json",{
            "schema":SCHEMA,"provider":provider,"run":identity,
            "source_artifact_id":read_plan()["source"]["artifact_id"],
            "denominator":60,"attempted":len(captures),
            "complete":complete,"missing_or_failed":60-complete,
            "status":"COMPLETE" if complete==60 and error is None
                                  else "INCOMPLETE_NO_RETRY",
            "initialization_error":error,"rows":captures,
            "new_tts_calls":0,"new_phone_calls":len(captures)
                if provider in ("xlsr","lv60") else 0,
            "new_asr_calls":len(captures)
                if provider in ("qwen","whisper","parakeet") else 0,
            "human_gold":False,"training_allowed":False})
    need(len(captures)==complete==60,"FIVE_OBSERVER_ROW_COMPLETE")
    print(json.dumps({"status":"COMPLETE","provider":provider,
                      "actual_forwards":complete,"training_allowed":False}))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--guard",action="store_true")
    ap.add_argument("--verify-remote",action="store_true")
    ap.add_argument("--run",action="store_true")
    ap.add_argument("--provider",choices=PROVIDERS)
    ap.add_argument("--source")
    ap.add_argument("--references")
    ap.add_argument("--output")
    a=ap.parse_args()
    need(sum(bool(x) for x in (a.source_check,a.guard,a.verify_remote,a.run))==1,
         "ONE_OBSERVER_MODE")
    if a.source_check:print(json.dumps(source_check()));return
    if a.guard:print(json.dumps(guard()));return
    if a.verify_remote:
        guard()
        print(json.dumps(remote_metadata(os.environ.get("GH_TOKEN"))))
        return
    need(a.provider and a.source and a.references and a.output,
         "RUN_REQUIRES_SOURCE_REFERENCES_PROVIDER_OUTPUT")
    run(a.provider,a.source,a.references,a.output)

if __name__=="__main__":main()
