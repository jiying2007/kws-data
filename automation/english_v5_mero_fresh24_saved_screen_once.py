"""Reconstruct V5 Hey Mero machine tiers from five original immutable archives.

No TTS, ASR, phoneme/KWS forward, training, tuning, or test-set evaluation.
One original Parakeet empty transcript stays UNKNOWN with its exact failed
receipt; all other original model answers and full phone logits are retained.
Never infer an acoustic label from the requested TTS prompt or reassign roles.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location(
        "v5_saved_screen_"+name,HERE/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

observer=load("english_v5_mero_fresh24_five_native_once")
policy=load("english_fresh24_ctc_admission")
phones=load("english_hifi_phones_once")
c=load("english_hifi_micro")
SCHEMA="english-v5-mero-fresh24-saved-5-native-screen-no-model-v1"
REPO="jiying2007/kws-data"
BEFORE="782ef6418c6915e21e816c0f3d7d61f8315eb46d"
ORIGINAL_RUN=44481695979
ORIGINAL_SHA=BEFORE
ORIGINAL_STATUS="failure"
SOURCE_RUN=44478554596
MISSING_ID="mero24-452-mero-slow"
SOURCE={"id":14507208152,"name":"english-v5-mero-fresh24-original-44478554596",
        "bytes":1961038,
        "digest":"sha256:2fbc26386136cfd12dddb01074f3c9c4556c8fc7c49819663f2d11a2499eaa1c"}
ARCHIVES={
    "qwen":(14508139460,61556,"9ade049f03871f7fc46503d5b8347be2c3b3c4de6b2d1486b80f7f7392fb6513"),
    "whisper":(14509500505,61818,"96cde1d82450144fa13eac0e8693fd8354be5199edf6a1d4e9a33418abafb4f9"),
    "parakeet":(14508911712,61820,"bdb7ef7788befd925f9b9c5b518c19db4545ce0912d747546bb2579cb8a83793"),
    "xlsr":(14508483184,3219642,"c80be0d6abcf1a19f9435f8d84410f442215a837a1fce28b98975f6e2758fd26"),
    "lv60":(14508558196,3221033,"f4f16018aefc0632503339e3d2a4169db560276791ec239c6009ab8cdc461263"),
}
REVIEW_COUNT=64
MIN_TRAIN_DEV={"train":6,"development":2}

def need(v,reason):
    if not v:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read_json(path):
    return json.loads(c.read(Path(path)))

def source_check():
    x=observer.source_check()
    need(x["source_run"]==SOURCE_RUN and
         x["source_artifact_id"]==SOURCE["id"] and
         x["source_rows"]==96 and x["review_rows"]==REVIEW_COUNT and
         observer.BEFORE=="e3f79406f469061dc9ac7438ed6658ad66375e55" and
         tuple(observer.PROVIDERS)==("qwen","whisper","parakeet","xlsr","lv60") and
         observer.SOURCE["digest"]==SOURCE["digest"] and
         observer.SOURCE["size"]==SOURCE["bytes"] and
         len(ARCHIVES)==5 and
         len({v[0] for v in ARCHIVES.values()})==5 and
         all(re.fullmatch("[0-9a-f]{64}",v[2]) for v in ARCHIVES.values()) and
         policy.MIN_QUALIFIED=={
            "train":6,"development":2,"calibration":2,"locked-test":2} and
         MISSING_ID=="mero24-452-mero-slow",
         "EXACT_FROZEN_SOURCE_AND_FIVE_OBSERVER_ARCHIVE_LOCK")
    return {"status":"READBACK_SOURCE_PINNED",
            "original_observer_run":ORIGINAL_RUN,
            "original_observer_head":ORIGINAL_SHA,
            "saved_source_rows":96,"review_rows":REVIEW_COUNT,
            "known_parakeet_empty_transcript":MISSING_ID,
            "observer_archive_count":len(ARCHIVES),
            "new_audio_or_model_forwards":0,
            "new_optimizer_updates":0,
            "new_locked_test_forwards":0,
            "training_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    require={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in require.items()),
         "FIRST_PUBLIC_MAIN_PUSH_ONLY")
    head=env.get("GITHUB_SHA","")
    need(bool(re.fullmatch("[0-9a-f]{40}",head)) and
         bool(re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID",""))),
         "REAL_PUBLIC_ACTIONS_EVENT_ID")
    if event is None:
        event=read_json(env["GITHUB_EVENT_PATH"])
    need(event.get("before")==BEFORE and event.get("after")==head and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("deleted") is False and event.get("forced") is False,
         "EXACT_SAVED_RESEARCH_READBACK_SUCCESSOR")
    if parent is None:
        parent=subprocess.check_output(
            ["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"NO_REPLAY_OR_OUT_OF_ORDER_MERGE")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":head,
            "original_observer_run":ORIGINAL_RUN,
            "new_model_forwards":0}

def original_identity():
    return {"run_id":str(ORIGINAL_RUN),"commit":ORIGINAL_SHA,
            "source_run":SOURCE_RUN,"source_commit":observer.SOURCE_COMMIT}

def verify_remote(token):
    need(type(token) is str and len(token)>10,
         "ACTIONS_READONLY_GITHUB_TOKEN_REQUIRED")
    def get(path):
        req=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+path,
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"v5-mero-saved-evidence-readback"})
        with urllib.request.urlopen(req,timeout=40) as f:
            raw=f.read(1000001)
        need(len(raw)<1000001,"BOUNDED_REMOTE_METADATA")
        return json.loads(raw)
    run=get("/actions/runs/"+str(ORIGINAL_RUN))
    need(run.get("id")==ORIGINAL_RUN and
         run.get("head_sha")==ORIGINAL_SHA and
         run.get("head_branch")=="main" and
         run.get("event")=="push" and
         run.get("run_attempt")==1 and
         run.get("status")=="completed" and
         run.get("conclusion")==ORIGINAL_STATUS,
         "ORIGINAL_NATIVE_REVIEW_RUN_STATUS_IDENTITY")
    artifacts={"source":SOURCE}
    for name,(id,size,digest) in ARCHIVES.items():
        artifacts[name]={"id":id,
          "name":f"v5-mero24-observer-{name}-{ORIGINAL_RUN}",
          "bytes":size,"digest":"sha256:"+digest}
    for key,item in artifacts.items():
        a=get("/actions/artifacts/"+str(item["id"]))
        expected_run=SOURCE_RUN if key=="source" else ORIGINAL_RUN
        expected_sha=observer.SOURCE_COMMIT if key=="source" else ORIGINAL_SHA
        need(a.get("id")==item["id"] and
             a.get("name")==item["name"] and
             a.get("size_in_bytes")==item["bytes"] and
             a.get("digest")==item["digest"] and
             a.get("expired") is False and
             a.get("workflow_run",{}).get("id")==expected_run and
             a.get("workflow_run",{}).get("head_sha")==expected_sha,
             "EXACT_SAVED_ARTIFACT_SHA_FOR_"+key)
    return artifacts

def source_rows(source):
    return observer.source_rows(source)

def review_packet(root,provider,rows):
    """Verify exact frozen native model, all per-clip capture/failed receipts."""
    need(provider in observer.PROVIDERS,"KNOWN_FIVE_BLIND_PROVIDER")
    root=Path(root)
    src={r["id"]:r for r in rows}
    need(len(src)==REVIEW_COUNT,"EXACT_SOURCE_AUDIT_BEFORE_REVIEWS")
    result=read_json(root/"RESULT.json")
    expected=REVIEW_COUNT-1 if provider=="parakeet" else REVIEW_COUNT
    need(result.get("schema")==observer.SCHEMA and
         result.get("provider")==provider and
         result.get("run")==original_identity() and
         result.get("source_artifact_id")==SOURCE["id"] and
         result.get("denominator")==REVIEW_COUNT and
         result.get("attempted")==REVIEW_COUNT and
         result.get("complete")==expected and
         result.get("missing_or_failed")==REVIEW_COUNT-expected and
         result.get("new_tts_calls")==0 and
         result.get("target_conditioned") is False and
         result.get("human_gold") is False and
         result.get("training_allowed") is False and
         result.get("initialization_error") is None and
         result.get("status")==("INCOMPLETE_NO_RETRY" if provider=="parakeet"
                                 else "COMPLETE") and
         type(result.get("rows")) is list and
         len(result["rows"])==REVIEW_COUNT,
         "EXACT_ORIGINAL_OBSERVER_COMPLETION_OR_FAILURE")
    model=read_json(root/"MODEL.json")
    if provider in ("xlsr","lv60"):
        source_id,revision=observer.identity(provider)
        idx=0 if provider=="xlsr" else 1
        need(model.get("model")==source_id and
             model.get("revision","").startswith(revision) and
             model.get("weights_sha256")==phones.WEIGHT_SHA[idx] and
             model.get("target_conditioned") is False,
             "UNCHANGED_FROZEN_NATIVE_PHONE_WEIGHT_IDENTITY")
        vocab_doc=read_json(root/"VOCAB.json")
        vocab=vocab_doc["vocab"]
        need(vocab_doc.get("model")==source_id and
             vocab_doc.get("revision")==model["revision"] and
             vocab_doc.get("pad_token_id")==0 and
             type(vocab) is dict and len(vocab)==392 and
             len(set(vocab.values()))==392,
             "EXACT_UNCHANGED_392_PHONE_VOCAB")
    else:
        need(model.get("observer")==observer.identity(provider) and
             model.get("target_conditioned") is False and
             model.get("initialization_model_calls")==0,
             "UNCHANGED_BLIND_ASR_MODEL_WEIGHTS")
        vocab=None
    saved={}
    for original in result["rows"]:
        sid=original.get("id")
        need(sid in src and sid not in saved and
             original.get("provider")==provider and
             original.get("source_wav_sha256")==src[sid]["wav_sha256"] and
             original.get("source_pcm_sha256")==src[sid]["pcm_sha256"] and
             original.get("source_samples")==src[sid]["samples"] and
             original.get("target_conditioned") is False and
             original.get("training_allowed") is False,
             "ORIGINAL_COMPLETE_PCM_PROVENANCE_PER_OBSERVER")
        if original.get("status")=="failed":
            need(provider=="parakeet" and sid==MISSING_ID and
                 original.get("error")=="ValueError:RAW_UNPROMPTED_COMPLETE_ASR_TEXT" and
                 read_json(root/"rows"/sid/"FAILED.json")==original and
                 src[sid]["role"]=="train" and
                 src[sid]["requested_class"]==2,
                 "ONLY_ONE_ORIGINAL_UNSCORABLE_ASR_NEVER_RELABEL")
            saved[sid]=None
            continue
        need(original.get("status")=="success" and
             original.get("input_coverage_complete") is True and
             read_json(root/"rows"/sid/"CAPTURE.json")==original,
             "EXACT_UNPROMPTED_SUCCESS_RECEIPT")
        if provider in ("xlsr","lv60"):
            raw=c.read(root/"logits"/(sid+".f32"),40*1024*1024)
            n=original.get("logit_frames")
            need(type(n) is int and 0<n<=2000 and
                 original.get("vocab_size")==392 and
                 len(raw)==n*392*4 and
                 sha(raw)==original.get("logits_sha256"),
                 "FULL_ORIGINAL_NATIVE_PHONE_LOGIT_BYTES")
            ids=[]
            for frame in struct.iter_unpack("<392f",raw):
                need(all(math.isfinite(x) for x in frame),
                     "NO_NONFINITE_NATIVE_PHONE_LOGIT")
                ids.append(max(range(392),key=frame.__getitem__))
            need(phones.ctc(ids,vocab,0)==original.get("raw_phone_tokens"),
                 "VERIFY_ORIGINAL_CTC_ARGMAX_TOKENS")
            saved[sid]={"logits":raw,"vocab":vocab,
                        "sha256":original["logits_sha256"],"pad_token_id":0}
        else:
            transcript=original.get("raw_transcript")
            need(type(transcript) is str and 0<len(transcript)<=4096 and
                 not any(ord(ch)<32 for ch in transcript),
                 "EXACT_ORIGINAL_UNPROMPTED_ASR_TEXT")
            saved[sid]=transcript
    need(set(saved)==set(src) and
         sum(v is not None for v in saved.values())==expected and
         (set(k for k,v in saved.items() if v is None)=={MISSING_ID}
          if provider=="parakeet" else
          all(v is not None for v in saved.values())),
         "SAVE_BUT_NEVER_REPLAY_OR_IMPUTE_ORIGINAL_MISSING_ASR")
    return saved,{"provider":provider,"complete":expected,
                  "attempted":REVIEW_COUNT,
                  "status":result["status"],
                  "missing_ids":[k for k,v in saved.items() if v is None]}

def count_eligible(decisions):
    groups=defaultdict(list)
    for row in decisions:
        groups[row["source_group_id"]].append(row)
    counts=Counter()
    members=[]
    for gid,rows in groups.items():
        accepted={r["observed_class"] for r in rows
                  if r["tier"] in ("silver","weak")}
        roles={r["role"] for r in rows}
        need(len(roles)==1 and roles<=set(("train","development")),
             "SPEAKER_IDENTITY_NOT_DISJOINT")
        if {0,2}<=accepted:
            counts[next(iter(roles))]+=1
            members.append(gid)
    return {"train":counts["train"],"development":counts["development"]},members

def evaluate(src,observers,out):
    source_check()
    current=guard()
    output=Path(out)
    need(output.is_absolute() and not output.exists(),
         "ONE_NEW_NONMUTATING_SAVED_READBACK_OUTPUT")
    output.mkdir()
    c.record(output/"STARTED.json",{"schema":SCHEMA,"run":current,
        "source_artifact":SOURCE,"observer_archives":ARCHIVES,
        "new_model_forwards":0,"new_optimizer_updates":0,
        "training_allowed":False,"locked_kws_test_opened":False})
    try:
        attest=verify_remote(os.environ.get("GH_TOKEN"))
        rows=source_rows(src)
        need(len(rows)==REVIEW_COUNT,"EXACT_64_TRAIN_DEV_SOURCE_CLIPS")
        values={};reports=[]
        for provider in observer.PROVIDERS:
            values[provider],report=review_packet(
                Path(observers)/provider,provider,rows)
            reports.append(report)
        decision=[]
        for row in rows:
            sid=row["id"]
            trans={k:values[k][sid] for k in policy.TEXT_MODELS}
            logit={k:values[k][sid] for k in policy.PHONE_MODELS}
            observed=policy.decide(
                {**row,"pcm_samples":row["samples"]},trans,logit)
            if observed["tier"] in ("silver","weak") and (
                    observed["class"]!=row["requested_class"]):
                observed={"tier":"unknown","class":None,
                    "reason":"REQUEST_VERSUS_ACOUSTIC_CONTRADICTION",
                    "margins":{},"training_allowed":False}
            if sid==MISSING_ID:
                need(observed["tier"]=="unknown" and
                     observed["class"] is None,
                     "PARAKEET_EMPTY_TRANSCRIPT_MUST_REMAIN_UNKNOWN")
            decision.append({
                "id":sid,"speaker_index":row["speaker_index"],
                "source_group_id":row["source_group_id"],
                "role":row["role"],"cell_id":row["cell_id"],
                "requested_class":row["requested_class"],
                "tier":observed["tier"],"observed_class":observed["class"],
                "reason":observed["reason"],"phone_margins":observed["margins"],
                "wav_sha256":row["wav_sha256"],
                "pcm_sha256":row["pcm_sha256"],
                "samples":row["samples"],"human_gold":False,
                "training_allowed":False})
        need(len(decision)==64 and
             Counter(r["role"] for r in decision)=={
                 "train":48,"development":16} and
             Counter(r["requested_class"] for r in decision)=={
                 2:32,0:16,None:16},
             "ONLY_16_ORIGINAL_UNEXPOSED_TRAIN_DEV_SOURCE_GROUPS")
        qualified,members=count_eligible(decision)
        passes=all(qualified[k]>=v for k,v in MIN_TRAIN_DEV.items())
        data={"schema":SCHEMA,
              "status":("PASS_NEW_TRAIN_DEV_MACHINE_SCREEN_ONLY"
                        if passes else "STOP_NEW_TRAIN_DEV_MACHINE_SCREEN"),
              "source_run":SOURCE_RUN,"original_observer_run":ORIGINAL_RUN,
              "observed_original_archives":attest,"provider_reports":reports,
              "source_group_counts":qualified,
              "source_group_identity_members":members,
              "min_train_development_groups":MIN_TRAIN_DEV,
              "calibration_source_groups_machine_qualified":None,
              "locked_test_source_groups_machine_qualified":None,
              "full_6_2_2_2_gate_passed":False,
              "tier_counts":dict(Counter(x["tier"] for x in decision)),
              "reason_counts":dict(Counter(x["reason"] for x in decision)),
              "role_tier_counts":{role:dict(Counter(
                  x["tier"] for x in decision if x["role"]==role))
                  for role in ("train","development")},
              "observed_label_counts":dict(Counter(str(x["observed_class"])
                   for x in decision if x["tier"] in ("silver","weak"))),
              "parakeet_original_empty_transcript":MISSING_ID,
              "nearphone_as_ordinary_negative":False,
              "machine_only_no_human_gold":True,
              "new_audio_or_model_forwards":0,
              "new_optimizer_updates":0,"new_weights":False,
              "training_snapshot_allowed":False,
              "locked_kws_test_opened":False,
              "ssc305_measured":False,"product_promotion":False,
              "rows":decision}
        c.record(output/"SCREEN.json",data)
        print(json.dumps({k:data[k] for k in (
              "status","source_group_counts","tier_counts",
              "reason_counts","observed_label_counts",
              "full_6_2_2_2_gate_passed","training_snapshot_allowed")},
              sort_keys=True))
    except BaseException as error:
        c.record(output/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_SAVED_EVIDENCE_READBACK",
             "error_type":type(error).__name__,"error":str(error)[:500],
             "new_audio_or_model_forwards":0,"new_optimizer_updates":0,
             "no_calibration_or_locked_inference":True})
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--guard",action="store_true")
    p.add_argument("--evaluate",action="store_true")
    p.add_argument("--source",type=Path)
    p.add_argument("--observers",type=Path)
    p.add_argument("--output",type=Path)
    x=p.parse_args()
    need(sum((x.source_check,x.guard,x.evaluate))==1,
         "EXACT_SINGLE_SAVED_REVIEW_STAGE")
    if x.source_check:
        print(json.dumps(source_check(),sort_keys=True))
    elif x.guard:
        print(json.dumps(guard(),sort_keys=True))
    else:
        need(x.source and x.observers and x.output,
             "ONLY_PREVIOUSLY_SAVED_ORIGINAL_DATA_ALLOWED")
        evaluate(x.source,x.observers,x.output)

if __name__=="__main__":
    main()
