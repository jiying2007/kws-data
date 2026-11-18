"""Two pinned independent complete-PCM phone models on eight new 1.7B clips.

Never calls TTS. Preserves raw CTC, full original float32 logits, normalized
unprompted PCM identities; all 8 remain research unadmitted.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent
def load(n):
    spec=importlib.util.spec_from_file_location("q17_phone_"+n,HERE/(n+".py"))
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    return v
c=load("english_hifi_micro")
p=load("english_hifi_phones_once")
e=load("english_qwen17_clone_probe_once")
admit=load("english_clone12_admission_policy")
SCHEMA="english-qwen17-two-train-style-phone-readback-v1"

def need(x,why):
    if not x:raise ValueError(why)

def readback(root,source,refs):
    rows=e.planned(e.source_refs(refs))
    info=e.readback(source,refs)
    root=Path(root);obj=json.loads(c.read(root/"OBSERVATIONS.json"))
    need(obj.get("schema")==SCHEMA and obj.get("completed")==16 and
         obj.get("training_allowed") is False and
         len(obj.get("rows",[]))==16,"EXACT_SIXTEEN_COMPLETE_PHONE_VOTES")
    lookup={}
    for r in obj["rows"]:
        key=(r["id"],r["model"])
        need(key not in lookup,"DUPLICATE_OBSERVER_ROW")
        lookup[key]=r
    need(len(lookup)==16 and {x["id"] for x in obj["rows"]}==
         {x["id"] for x in rows},"EXACT_EIGHT_WAV_AND_DUAL_MODEL_ROSTER")
    pairs={};categories=Counter()
    for row in rows:
        sid=row["id"]
        vals={}
        for idx,(model,rev) in enumerate(c.PHONE_MODELS):
            record=lookup[(sid,model)]
            orig=c.inspect_wav(Path(source)/"wav"/(sid+".wav"))
            need(record["wav_sha256"]==orig["wav_sha256"]
                 and record["pcm_sha256"]==orig["pcm_sha256"]
                 and record["sample_count"]==orig["samples"] and
                 record["target_conditioned"] is False and
                 record["training_allowed"] is False,
                 "SAVED_UNCONDITIONED_SOURCE_IDENTITIES")
            raw=c.read(root/"logits"/(str(idx)+"-"+sid+".f32"),40*1024*1024)
            need(c.digest(raw)==record["logits_sha256"] and
                 len(raw)==4*record["frames"]*record["vocab_size"] and
                 record["vocab_size"]==392,
                 "AUTHENTICATED_ALL_FLOAT32_NATIVE_LOGITS")
            vals["xlsr" if idx==0 else "lv60"]=record["raw_phone_tokens"]
        a=admit.flattened(vals["xlsr"])
        b=admit.flattened(vals["lv60"])
        if row["requested_brand_class"]==2:
            support=bool(admit.M_XLSR.fullmatch(a) and admit.M_LV60.fullmatch(b))
        else:
            support=bool(admit.V_XLSR.fullmatch(a) and admit.V_LV60.fullmatch(b))
        result={"id":sid,"donor":row["source_preset"],
                "requested_brand_class":row["requested_brand_class"],
                "dual_phone_support_only":support,
                "requires_three_asr":True,
                "eligible_training_label":False,
                "positive_duration_pass":next(x for x in
                    json.loads(c.read(Path(source)/"MANIFEST.json"))["rows"] if x["id"]==sid)["positive_duration_pass"]}
        pairs[sid]=result
        categories[(row["cell_id"],support)]+=1
    return {"schema":SCHEMA,"status":"PHONE_ONLY_RESEARCH_SCREEN",
            "original_8_wav_verified":True,
            "completed_phone_forwards":16,
            "support_by_cell":{cell:categories[(cell,True)] for cell,_,_,_ in e.RENDERS},
            "training_allowed":False,
            "asr_support":"NOT_RUN",
            "result_rows":list(pairs.values())}

def run(source,reference,output):
    e.guard()
    e.readback(source,reference)
    root=Path(output)
    need(root.is_absolute() and not root.exists(),"CREATE_ONCE")
    root.mkdir(parents=True)
    (root/"logits").mkdir()
    c.record(root/"STARTED.json",{"schema":SCHEMA,
         "model_synthesis_calls_this_stage":0,"planned_phone_calls":16,
         "training_allowed":False,"asr_not_called":True})
    receipts=[]
    try:
        import numpy as np
        import torch
        from transformers import Wav2Vec2Config,Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
        from huggingface_hub import HfApi,snapshot_download
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        rows=e.planned(e.source_refs(reference))
        for idx,(name,rev) in enumerate(c.PHONE_MODELS):
            actual=HfApi(token=False).model_info(name,revision=rev)
            need(actual.sha.startswith(rev),"PINNED_PHONE_REVISION")
            cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/("q17-phone-"+str(idx))
            folder=Path(snapshot_download(repo_id=name,revision=actual.sha,
                cache_dir=str(cache),max_workers=2,token=False))
            weights=folder/"pytorch_model.bin"
            need(weights.is_file() and weights.stat().st_size==1263535127,
                 "COMPLETE_ORIGINAL_PHONE_MODEL")
            h=hashlib.sha256()
            with weights.open("rb") as f:
                for b in iter(lambda:f.read(1048576),b""):h.update(b)
            need(h.hexdigest()==p.WEIGHT_SHA[idx],
                 "UNMODIFIED_ORIGINAL_PHONEME_WEIGHTS")
            vocab=p.read_verified_hf_metadata(folder,cache,"vocab.json")
            feat=Wav2Vec2FeatureExtractor.from_pretrained(str(folder),
                                                         local_files_only=True)
            config=Wav2Vec2Config.from_json_file(str(folder/"config.json"))
            model=Wav2Vec2ForCTC(config)
            model.load_state_dict(torch.load(weights,map_location="cpu",
                                         weights_only=True),strict=True)
            model.eval()
            need(config.pad_token_id==0 and feat.sampling_rate==16000,
                 "ORIGINAL_NATIVE_SAMPLE_GEOMETRY")
            c.record(root/(str(idx)+"-LOCK.json"),{"model":name,
                     "revision":actual.sha,"sha256":p.WEIGHT_SHA[idx],
                     "vocab":vocab,"pad":config.pad_token_id})
            for row in rows:
                sid=row["id"]
                wav=Path(source)/"wav"/(sid+".wav")
                meta=c.inspect_wav(wav)
                pcm,samples=p.pcm_of(wav)
                need(meta["pcm_sha256"]==c.digest(pcm),"ORIGINAL_COMPLETE_PCM")
                x=np.frombuffer(pcm,dtype="<i2").astype("float32")/32768
                tensor=feat(x,sampling_rate=16000,return_tensors="pt",
                            padding=False,return_attention_mask=True)
                with torch.inference_mode():
                    logits=model(input_values=tensor.input_values,
                                 attention_mask=tensor.attention_mask.int()).logits
                arr=logits[0].detach().cpu().numpy().astype("<f4").tobytes()
                ids=logits.argmax(dim=-1)[0].tolist()
                token=p.ctc(ids,vocab,config.pad_token_id)
                (root/"logits"/(str(idx)+"-"+sid+".f32")).write_bytes(arr)
                row_result={"id":sid,"model":name,"revision":actual.sha,
                     "wav_sha256":meta["wav_sha256"],
                     "pcm_sha256":meta["pcm_sha256"],
                     "sample_count":samples,
                     "raw_phone_tokens":token,
                     "logits_sha256":c.digest(arr),
                     "frames":logits.shape[1],"vocab_size":logits.shape[2],
                     "target_conditioned":False,"training_allowed":False}
                receipts.append(row_result)
                c.record(root/(str(idx)+"-"+sid+".json"),row_result)
            del model,feat
            import gc;gc.collect()
        need(len(receipts)==16,"ALL_PHONE_FORWARDS_RETURNED")
        c.record(root/"OBSERVATIONS.json",{"schema":SCHEMA,
             "completed":len(receipts),"rows":receipts,
             "training_allowed":False,"new_tts_calls":0,
             "new_asr_calls":0,"independent_human_gold":False})
        screen=readback(root,source,reference)
        c.record(root/"PHONE_SCREEN.json",screen)
        print(json.dumps({k:screen[k] for k in
             ("status","completed_phone_forwards","support_by_cell","training_allowed")}))
    except BaseException as exc:
        c.record(root/"FAILED.json",{"schema":SCHEMA,
             "error_type":type(exc).__name__,"error":str(exc)[:500],
             "completed_phone_forwards":len(receipts),
             "remaining":16-len(receipts),
             "retry_count":0,"training_allowed":False})
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-check",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--readback",action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--reference")
    parser.add_argument("--output")
    a=parser.parse_args()
    need(sum(bool(x) for x in (a.source_check,a.run,a.readback))==1,"ONE_MODE")
    if a.source_check:
        need(len(c.PHONE_MODELS)==2 and len(e.RENDERS)==4,
             "SOURCE_ONLY_TWO_MODEL_CONTRACT")
        print(json.dumps({"status":"SOURCE_ONLY","phone_calls":0,
                          "training_allowed":False}));return
    need(a.source and a.reference and a.output,"ACTUAL_INPUT_PATHS_REQUIRED")
    if a.run:run(a.source,a.reference,a.output)
    else:print(json.dumps(readback(a.output,a.source,a.reference)))

if __name__=="__main__":main()
