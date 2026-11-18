"""Model-free review of retained Melo25 with three blind ASR and two phones.

Never re-infer, regenerate, resplit or silently convert UNKNOWN to negative.
The pre-existing clone12 blind() phonetic/ASR rule is used unchanged for this
new prospective pilot; results are five synthetic styles, NOT 6/2/2/2 groups.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("melo_review_combine_"+name,HERE/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
c=load("english_hifi_micro")
p=load("english_hifi_phones_once")
worker=load("english_melo_cmu_observe_once")
policy=load("english_clone12_admission_policy")
SCHEMA="english-melo-cmu-five-observer-quality-screen-v1"

def need(ok,why):
    if not ok:raise ValueError(why)

def source_check():
    x=worker.source_check()
    need(tuple(worker.PROVIDERS)==tuple(policy.TEXT_MODELS)+tuple(policy.PHONE_MODELS)
         and policy.POSITIVE_MAX_SAMPLES==28672,
         "ORIGINAL_FROZEN_FIVE_OBSERVER_RULE")
    return {**x,"screen_rule":policy.SCHEMA,"training_allowed":False}

def observation_packet(root,provider,source_rows,review_run):
    root=Path(root)/("english-melo-cmu-observer-"+provider)
    # The original phone forward results are immutable; only the three
    # zero-forward ASR workers get new run identities. No 50-call replay.
    expected_run=(worker.original_review_identity()
                  if provider in ("xlsr","lv60") else review_run)
    ids={row["id"] for row in source_rows}
    if not (root/"RESULT.json").is_file():
        return {sid:None for sid in ids},{"provider":provider,
            "status":"MISSING_OBSERVER","complete":0,"denominator":25}
    result=json.loads(c.read(root/"RESULT.json"))
    need(result.get("schema")==worker.SCHEMA and
         result.get("provider")==provider and
         result.get("denominator")==25 and
         result.get("source_artifact_id")==worker.ARTIFACT_ID and
         result.get("run")==expected_run and
         result.get("training_allowed") is False and
         result.get("human_gold") is False and
         result.get("new_tts_calls")==0,
         "EXACT_OBSERVER_RESULT_AND_SOURCE")
    observed=result.get("rows",[])
    need(type(observed) is list and len(observed)<=25 and
         len({x.get("id") for x in observed})==len(observed) and
         {x.get("id") for x in observed}<=ids,
         "ONLY_ORIGINAL_25_OBSERVER_INPUT_IDS")
    file_model=json.loads(c.read(root/"MODEL.json")) if (root/"MODEL.json").is_file() else None
    vocab=None
    if provider in ("xlsr","lv60") and file_model:
        expected,rev=worker.model_id(provider)
        need(file_model.get("model")==expected and
             file_model.get("revision","").startswith(rev) and
             file_model.get("weights_sha256")==p.WEIGHT_SHA[0 if provider=="xlsr" else 1]
             and file_model.get("target_conditioned") is False,
             "EXACT_FROZEN_NATIVE_PHONE_MODEL")
        vocab_file=json.loads(c.read(root/"VOCAB.json"))
        need(vocab_file.get("model")==expected and
             vocab_file.get("revision")==file_model["revision"] and
             vocab_file.get("pad_token_id")==0,
             "FROZEN_NATIVE_VOCAB_PROVENANCE")
        vocab=vocab_file.get("vocab")
        need(type(vocab) is dict and len(vocab)==392 and
             len(set(vocab.values()))==392,
             "ORIGINAL_NATIVE_PHONE_VOCAB")
    if provider in ("qwen","whisper","parakeet") and file_model:
        expected=worker.model_id(provider)
        need(file_model.get("observer")==expected and
             file_model.get("target_conditioned") is False,
             "ORIGINAL_NATIVE_BLIND_TEXT_MODEL")
    mapped={x["id"]:x for x in observed}
    row_map={row["id"]:row for row in source_rows}
    values={}
    success=0
    for sid in ids:
        item=mapped.get(sid)
        if not item or item.get("status")!="success":
            values[sid]=None
            continue
        original=row_map[sid]
        need(item.get("provider")==provider and
             item.get("source_pcm_sha256")==original["pcm_sha256"] and
             item.get("source_wav_sha256")==original["wav_sha256"] and
             item.get("source_samples")==original["samples"] and
             item.get("target_conditioned") is False and
             item.get("training_allowed") is False and
             item.get("input_coverage_complete") is True,
             "OBSERVED_ORIGINAL_UNPROMPTED_FULL_PCM")
        saved=json.loads(c.read(root/"rows"/sid/"CAPTURE.json"))
        need(saved==item,"CAPTURE_AND_RESULT_SAME_NUMERIC_OBSERVATION")
        if provider in ("xlsr","lv60"):
            need(vocab is not None,"FROZEN_PHONE_MODEL_REQUIRED")
            logits=c.read(root/"logits"/(sid+".f32"),40*1024*1024)
            frames=item.get("logit_frames")
            need(type(frames) is int and 0<frames<=2000 and
                 item.get("vocab_size")==392 and
                 len(logits)==frames*392*4 and
                 c.digest(logits)==item.get("logits_sha256"),
                 "ALL_NATIVE_CTC_LOGITS_AUTHENTIC")
            predictions=[]
            for frame in struct.iter_unpack("<392f",logits):
                need(all(math.isfinite(v) for v in frame),"FINITE_SAVED_NATIVE_LOGITS")
                predictions.append(max(range(392),key=frame.__getitem__))
            tokens=p.ctc(predictions,vocab,0)
            need(tokens==item.get("raw_phone_tokens"),
                 "RECOMPUTED_COMPLETE_CTC_MATCHES_CAPTURE")
            values[sid]=tokens
        else:
            value=item.get("raw_transcript")
            need(type(value) is str and 0<len(value)<=4096 and
                 not any(ord(ch)<32 for ch in value),
                 "ORIGINAL_COMPLETE_ASR_TEXT")
            values[sid]=value
        success+=1
    need(success==result.get("complete") and
         success<=result.get("attempted",0)<=25 and
         result.get("status")==
         ("COMPLETE" if success==25 else "INCOMPLETE_NO_RETRY"),
         "ORIGINAL_OBSERVER_COMPLETENESS")
    return values,{"provider":provider,"status":result["status"],
        "complete":success,"denominator":25,
        "actual_run_id":expected_run["run_id"],
        "evidence_origin":"retained_50_phone_forwards" if provider in ("xlsr","lv60")
                         else "new_only_zero_forward_asr",
        "prior_artifact_id":worker.PRIOR_ARTIFACTS[provider][0]
                             if provider in ("xlsr","lv60") else None,
        "original_reported_attempted":result["attempted"]}

def one_decision(row,texts,phones):
    """Only observed consensus can become machine candidate, never human gold."""
    tier,label,reason=policy.blind(texts,phones)
    # Old Meero and voiced Vozen are deliberately phoneme contrasts; their
    # result must never become new positive OR treated as background.
    if row["requested_class"] is None:
        return "unknown",None,"NONCANONICAL_CONTRAST_NOT_TRAINING_NEGATIVE"
    if tier!="unknown" and label!=row["requested_class"]:
        return "unknown",None,"BLIND_RESULT_CONTRADICTS_SOURCE_INTENT"
    if tier!="unknown" and label in (1,2) and (
        row.get("positive_duration_pass") is not True or
        row.get("samples",100000000)>policy.POSITIVE_MAX_SAMPLES):
        return "unknown",None,"WHOLE_POSITIVE_TOO_LONG_NO_CROP"
    return tier,label,reason

def evaluate(rows,asrs,phones):
    need(type(rows) is list and len(rows)==25 and
         len({r["id"] for r in rows})==25 and
         set(asrs)==set(phones)=={r["id"] for r in rows},
         "EXACT_ORIGINAL_TWENTY_FIVE_CELLS")
    results=[]
    roles=defaultdict(list)
    for row in rows:
        sid=row["id"]
        need(row["role"]=="research-train-only" and
             row["speaker"] in worker.m.SPEAKERS and
             row["requested_class"] in (0,1,2,None),
             "ONLY_NEW_FIVE_STYLES_NOT_SPLIT_GROUPS")
        tier,label,reason=one_decision(row,asrs[sid],phones[sid])
        result={"id":sid,"style_id":row["style_id"],
            "requested_class":row["requested_class"],
            "cell_id":row["cell_id"],"tier":tier,
            "observed_class":label,"reason":reason,
            "wav_sha256":row["wav_sha256"],
            "pcm_sha256":row["pcm_sha256"],
            "samples":row["samples"],"positive_duration_pass":
            row["positive_duration_pass"],
            "human_gold":False,"training_allowed":False,
            "independent_real_far_frr":False}
        results.append(result)
        roles[row["speaker"]].append(result)
    need(len(roles)==5 and all(len(v)==5 for v in roles.values()),
         "FIVE_MELONET_STYLES_NOT_TWELVE_SOURCES")
    complete={}
    for key,group in roles.items():
        labels={r["observed_class"] for r in group if r["tier"] in ("silver","weak")}
        complete[key]=(labels=={0,1,2})
    return {"schema":SCHEMA,
        "status":"PILOT_ALL_FIVE_STYLES_SUPPORTED" if all(complete.values()) else
                 "PILOT_SOURCE_QUALITY_INCOMPLETE",
        "denominator":25,
        "tier_counts":dict(Counter(x["tier"] for x in results)),
        "reason_counts":dict(Counter(x["reason"] for x in results)),
        "class_counts":dict(Counter(str(x["observed_class"])
                             for x in results if x["tier"]!="unknown")),
        "complete_synthetic_styles":sum(complete.values()),
        "complete_by_style":complete,
        "role_coverage":{"train":0,"development":0,"calibration":0,"locked-test":0},
        "minimum_groups_by_role":policy.ROLES,
        "twelve_independent_source_groups_satisfied":False,
        "training_snapshot_allowed":False,
        "new_optimizer_updates":0,
        "new_model_weights":False,
        "human_gold":False,
        "independent_real_far_frr":False,
        "rows":results}

def combine(source,observers,out):
    review=worker.guard()
    rows=worker.source_rows(source)
    root=Path(out)
    need(root.is_absolute() and not root.exists(),"NEW_SCREEN_ONLY")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":SCHEMA,"run":review,
       "source_run":worker.SOURCE_RUN,"source_artifact_id":worker.ARTIFACT_ID,
       "new_tts_calls":0,"new_inference_calls":0,
       "human_gold":False,"training_allowed":False})
    texts={r["id"]:{} for r in rows}
    phones={r["id"]:{} for r in rows}
    info=[]
    for provider in worker.PROVIDERS:
        values,facts=observation_packet(observers,provider,rows,review)
        info.append(facts)
        for sid,value in values.items():
            (phones if provider in ("xlsr","lv60") else texts)[sid][provider]=value
    complete=all(x["complete"]==25 and x["status"]=="COMPLETE" for x in info)
    result=evaluate(rows,texts,phones)
    final_status=result["status"] if complete else "STOP_MISSING_FIVE_BLIND_OBSERVERS"
    c.record(root/"SCREEN.json",{
        "schema":SCHEMA,"status":final_status,
        "source_run":worker.SOURCE_RUN,"source_commit":worker.SOURCE_COMMIT,
        "source_artifact_id":worker.ARTIFACT_ID,
        "source_artifact_digest":worker.ARTIFACT_DIGEST,
        "review_run":review,
        "machine_review":result,
        "observer_reports":info,
        "original_native_phone_review_run":worker.PRIOR_REVIEW_RUN,
        "original_50_phone_forwards_reused":True,
        "new_native_phone_forwards":0,
        "original_three_asr_zero_forward_failures_authenticated":True,
        "all_five_native_observers_complete":complete,
        "prospective_rule":policy.SCHEMA,
        "old_prior_stops_unchanged":True,
        "new_audio_generated":0,"new_model_forwards":0,
        "training_snapshot_allowed":False,
        "human_gold":False,"locked_test_consumed":False,
        "independent_real_far_frr":False,
        "new_optimizer_updates":0,
        "new_model_weights":False})
    print(json.dumps({"status":final_status,"source_rows":25,
       "tier_counts":result["tier_counts"],
       "complete_synthetic_styles":result["complete_synthetic_styles"],
       "source_coverage_6_2_2_2":False,
       "observer_reports":info,
       "training_snapshot_allowed":False},sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--combine",action="store_true")
    ap.add_argument("--source")
    ap.add_argument("--observers")
    ap.add_argument("--output")
    a=ap.parse_args()
    need(a.source_check!=a.combine,"ONE_MODE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    need(all((a.source,a.observers,a.output)),"SAVED_SOURCE_AND_ALL_OBSERVERS_REQUIRED")
    combine(a.source,a.observers,a.output)

if __name__=="__main__":main()
