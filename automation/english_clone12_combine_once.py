"""Authenticate all 60 saved clone WAVs plus five blind observers and screen.

No TTS, model imports, inference, training, threshold tuning, resplitting or
relabels from requested text. Recompute dual-phone CTC tokens from every
retained complete logit tensor before prospective admission.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
def load(name):
    spec=importlib.util.spec_from_file_location("combine_"+name,HERE/(name+".py"))
    x=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x)
    return x
c=load("english_hifi_micro")
p=load("english_hifi_phones_once")
worker=load("english_clone12_observer_once")
policy=load("english_clone12_admission_policy")
SCHEMA="english-clone12-five-observer-combined-v1"
PROVIDERS=worker.PROVIDERS

def need(value,why):
    if not value:raise ValueError(why)

def read_provider(root,provider,source_rows):
    root=Path(root)/("english-clone12-observer-"+provider)
    ids={r["id"] for r in source_rows}
    if not (root/"RESULT.json").is_file():
        return {sid:None for sid in ids},{"provider":provider,
                "status":"MISSING_OBSERVER_ARTIFACT","success":0,"denominator":60}
    result=json.loads(c.read(root/"RESULT.json"))
    need(result.get("schema")==worker.SCHEMA and
         result.get("provider")==provider and result.get("denominator")==60 and
         result.get("source_artifact_id")==worker.read_plan()["source"]["artifact_id"] and
         result.get("training_allowed") is False and
         result.get("human_gold") is False and
         len(result.get("rows",[]))<=60,
         "BOUND_ORIGINAL_BLIND_OBSERVER_RESULT")
    byid={r["id"]:r for r in result["rows"]}
    need(len(byid)==len(result["rows"]) and set(byid)<=ids,
         "UNIQUE_SUBSET_OF_SIXTY_ORIGINAL_AUDIO")
    audio={r["id"]:r for r in source_rows}
    vocab=None
    if provider in ("xlsr","lv60") and byid:
        model=json.loads(c.read(root/"MODEL.json"))
        need(model.get("model")==worker.model_id(provider)[0] and
             model.get("weights_sha256")==p.WEIGHT_SHA[0 if provider=="xlsr" else 1],
             "FROZEN_NATIVE_PHONE_MODEL_PROVENANCE")
        rawv=json.loads(c.read(root/"VOCAB.json"))
        need(rawv.get("model")==model["model"] and rawv.get("revision")==model["revision"] and
             rawv.get("pad_token_id")==0,"NATIVE_PHONEME_VOCAB_LOCK")
        vocab=rawv["vocab"]
        need(type(vocab) is dict and len(vocab)==392 and len(set(vocab.values()))==392,
             "ORIGINAL_NATIVE_PHONEME_VOCAB")
    values={}
    success=0
    for sid,row in audio.items():
        capture=byid.get(sid)
        if capture is None or capture.get("status")!="success":
            values[sid]=None
            continue
        need(capture.get("target_conditioned") is False and
             capture.get("training_allowed") is False and
             capture.get("input_coverage_complete") is True and
             capture.get("input_wav_sha256")==row["wav_sha256"] and
             capture.get("input_pcm_sha256")==row["pcm_sha256"] and
             capture.get("source_samples")==row["samples"],
             "ACTUAL_BLIND_OBSERVER_INPUT_BOUND_TO_SOURCE")
        if provider in ("xlsr","lv60"):
            logits=c.read(root/"logits"/(sid+".f32"),40*1024*1024)
            n=capture.get("logit_frames")
            need(type(n) is int and 0<n<=2000 and
                 capture.get("vocab_size")==392 and len(logits)==n*392*4 and
                 c.digest(logits)==capture.get("logits_sha256"),
                 "EVERY_FULL_SAVED_PHONE_LOGIT")
            predictions=[]
            for frame in struct.iter_unpack("<392f",logits):
                need(all(math.isfinite(v) for v in frame),"FINITE_ORIGINAL_LOGIT_FRAME")
                predictions.append(max(range(392),key=frame.__getitem__))
            tokens=p.ctc(predictions,vocab,0)
            need(capture.get("raw_phone_tokens")==tokens,
                 "RECOMPUTED_NATIVE_ARGMAX_CTC_EXACT")
            values[sid]=tokens
        else:
            transcript=capture.get("raw_transcript")
            need(type(transcript) is str and
                 0<len(transcript)<=4096 and
                 not any(ord(ch)<32 for ch in transcript),
                 "EXACT_RAW_COMPLETED_ASR_TEXT")
            values[sid]=transcript
        success+=1
    need(success<=60,"CALL_COUNT_BOUND")
    return values,{"provider":provider,
        "status":result["status"],"success":success,"denominator":60,
        "original_reported_complete":result.get("complete"),
        "original_reported_attempted":result.get("attempted")}

def combine(source,refs,observers,out):
    worker.guard()
    rows=worker.trusted_source(source,refs)
    root=Path(out)
    need(root.is_absolute() and not root.exists(),"CREATE_FIRST_COMBINED_RECEIPT")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{
        "schema":SCHEMA,"denominator":60,"source_run":44257296925,
        "source_artifact_id":worker.read_plan()["source"]["artifact_id"],
        "observer_roster":PROVIDERS,"new_tts_calls":0,
        "new_inference_calls":0,"training_allowed":False})
    transcripts={row["id"]:{} for row in rows}
    phone={row["id"]:{} for row in rows}
    facts=[]
    for provider in PROVIDERS:
        values,fact=read_provider(observers,provider,rows)
        facts.append(fact)
        for sid,result in values.items():
            if provider in ("xlsr","lv60"):
                phone[sid][provider]=result
            else:transcripts[sid][provider]=result
    decisions=policy.evaluate(rows,transcripts,phone)
    all_complete=all(x["success"]==60 and x["status"]=="COMPLETE"
                     for x in facts)
    allowed=bool(all_complete and decisions["training_snapshot_allowed"])
    status=decisions["status"] if all_complete else "STOP_INCOMPLETE_FIVE_OBSERVERS"
    c.record(root/"SCREEN.json",{"schema":SCHEMA,"status":status,
       "source_run":44257296925,
       "source_artifact_id":worker.read_plan()["source"]["artifact_id"],
       "source_commit":worker.read_plan()["source"]["commit"],
       "source_roles_unchanged":True,"denominator":60,
       "observers":facts,
       "independent_five_observers_complete":all_complete,
       "phonetic_policy":policy.SCHEMA,
       "status_under_original_v1_policy":"NOT_REINTERPRETED",
       "screen":decisions,"training_snapshot_allowed":allowed,
       "human_gold":False,"independent_real_far_frr":False,
       "new_tts_calls":0,"new_asr_calls":0,
       "new_phone_calls":0,"optimizer_updates":0,
       "new_model_weights":False})
    print(json.dumps({"status":status,"training_snapshot_allowed":allowed,
       "denominator":60,"tier_counts":decisions["tier_counts"],
       "complete_groups_by_role":decisions["complete_groups_by_role"],
       "observers":facts},sort_keys=True))
    return allowed

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--combine",action="store_true")
    ap.add_argument("--source")
    ap.add_argument("--references")
    ap.add_argument("--observers")
    ap.add_argument("--output")
    args=ap.parse_args()
    need(args.source_check != args.combine,"ONE_MODE")
    if args.source_check:
        worker.source_check()
        need(PROVIDERS==("qwen","whisper","parakeet","xlsr","lv60"),
             "ORIGINAL_FIVE_OBSERVERS")
        print(json.dumps({"status":"SOURCE_ONLY","calls":0,
                          "training_allowed":False}))
        return
    need(all((args.source,args.references,args.observers,args.output)),
         "EVERY_INPUT_ARTIFACT_REQUIRED")
    combine(args.source,args.references,args.observers,args.output)

if __name__=="__main__":main()
