#!/usr/bin/env python3
"""B10 real saved five-native source screening: read only, no new acoustic forward.

Verify original full PCM and 90/90 native captures with immutable GitHub
artifact identities. Reuse frozen multi-ASR + two-phone source policy;
an intended TTS text is NEVER ground truth. Keep TRAIN and DEV separate.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
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
    spec=importlib.util.spec_from_file_location("b10_saved_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

observer=load("english_v6_b10_hardneg_five_native_once")
policy=load("english_fresh24_ctc_admission")
phones=load("english_hifi_phones_once")
c=load("english_hifi_micro")
SCHEMA="english-v6-b10-saved-native-screen-v1"
REPO="jiying2007/kws-data"
BEFORE="0c82174349bef08c9a819651211703ac25914bcb"
REVIEW_RUN=44908125673
SOURCE_RUN=44906200481
SOURCE_COMMIT="115738743e6c2cfa0cc55df9df772bb00e4c673c"
REVIEW_COMMIT=BEFORE
ARCHIVES={
 "source":(14708400220,"english-v6-b10-hardneg-train-dev-source-44906200481",
            371843,"dbc57d762922e60ff23253af2463d5d64e1c8021c74fad6e2f0a9ecb9fd87465"),
 "qwen":(14708582056,"b10-hardneg-native-qwen-44908125673",
         18609,"bbcd18c63615385734877f1f594f4f9adfaa3ff11dc0dfbeee5827d562328fe1"),
 "whisper":(14709245830,"b10-hardneg-native-whisper-44908125673",
            18644,"3be126bb9b489e7b6922eb0c2c3fea17762f0ed6398d873f33f0b66e64bde3c5"),
 "parakeet":(14708154254,"b10-hardneg-native-parakeet-44908125673",
             18617,"96ef7a21d2e7397847fc7639cb413b1a73ebb4774f51fdac8b44d82f255482e5"),
 "xlsr":(14708867456,"b10-hardneg-native-xlsr-44908125673",
         958206,"605938c2e243d071f35b81e075d8b4c30cfe2983b188cb07fbf02fd447d941aa"),
 "lv60":(14708457230,"b10-hardneg-native-lv60-44908125673",
         958750,"85f4fca54fb2de30d3fcb581d2692849c5778f199b081bc357ee0778f2b9e804")}

def need(ok,msg):
    if not ok: raise ValueError(msg)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path):
    return json.loads(c.read(Path(path)))

def source_check():
    p=observer.source_check()
    need(p["status"]=="B10_SOURCE_PINNED"
         and p["source_run"]==SOURCE_RUN
         and p["source_artifact_id"]==ARCHIVES["source"][0]
         and p["source_rows"]==18 and p["review_rows"]==18
         and p["source_roles"]=={"train":6,"development":12}
         and tuple(p["observer_roster"])==tuple(policy.TEXT_MODELS+policy.PHONE_MODELS)
         and observer.BEFORE==SOURCE_COMMIT
         and len(ARCHIVES)==6
         and len({x[0] for x in ARCHIVES.values()})==6
         and all(re.fullmatch(r"[0-9a-f]{64}",x[3]) for x in ARCHIVES.values())
         and policy.MIN_QUALIFIED=={"train":6,"development":2,
                                    "calibration":2,"locked-test":2},
         "PINNED_B10_18_PCM_90_NATIVE_AND_FROZEN_FIVE_MODEL_POLICY")
    return {"schema":SCHEMA,"status":"SAVED_SOURCE_ONLY_NO_NEW_MODEL_FORWARD",
            "source_run":SOURCE_RUN,"observer_run":REVIEW_RUN,
            "originals":18,"new_train_candidates":6,"new_dev_originals":12,
            "native_observations_expected":90,"new_model_forwards":0,
            "new_tts_asr_phone_calls":0,"new_optimizer_updates":0,
            "training_allowed_before_source_review":False,
            "no_old_DEV520_528_training":True,"cal_locked_opened":False,
            "product_promotion_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    e=os.environ if env is None else env
    need(all(e.get(k)==v for k,v in {
         "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
         "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
         "GITHUB_RUN_ATTEMPT":"1"}.items()),
         "FIRST_PUBLIC_MAIN_SAVED_ONLY_NO_REPLAY")
    head=e.get("GITHUB_SHA","")
    need(re.fullmatch(r"[0-9a-f]{40}",head) is not None
         and re.fullmatch(r"[1-9][0-9]*",e.get("GITHUB_RUN_ID","")) is not None,
         "EXACT_MAIN_HEAD_AND_RUN_ID")
    if event is None: event=read(e["GITHUB_EVENT_PATH"])
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                         cwd=ROOT,text=True).strip()
    need(event.get("before")==BEFORE
         and event.get("after")==head and parent==BEFORE
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("deleted") is False and event.get("forced") is False,
         "EXACT_FIRST_B10_SAVED_QA_MAIN_SUCCESSOR")
    return {"run_id":e["GITHUB_RUN_ID"],"head":head,"parent":BEFORE}

def remote(token):
    need(type(token) is str and len(token)>10,
         "READONLY_GITHUB_ARTIFACT_PROVENANCE_TOKEN")
    def get(path):
        req=urllib.request.Request("https://api.github.com/repos/"+REPO+
               "/actions/"+path,headers={
                "Authorization":"Bearer "+token,
                "Accept":"application/vnd.github+json",
                "User-Agent":"kws-v6-b10-saved-qa-once"})
        with urllib.request.urlopen(req,timeout=35) as resp:
            raw=resp.read(1000001)
        need(len(raw)<=1000000,"BOUNDED_GITHUB_METADATA")
        return json.loads(raw)
    for run_id,head in ((SOURCE_RUN,SOURCE_COMMIT),(REVIEW_RUN,REVIEW_COMMIT)):
        x=get("runs/"+str(run_id))
        need(x.get("id")==run_id and x.get("head_sha")==head
             and x.get("head_branch")=="main" and x.get("event")=="push"
             and x.get("status")=="completed" and x.get("conclusion")=="success"
             and x.get("run_attempt")==1,
             "EXACT_SUCCESSFUL_PUBLIC_SOURCE_AND_FIVE_NATIVE_RUN")
    for key,item in ARCHIVES.items():
        a=get("artifacts/"+str(item[0]))
        run_id=SOURCE_RUN if key=="source" else REVIEW_RUN
        head=SOURCE_COMMIT if key=="source" else REVIEW_COMMIT
        need(a.get("id")==item[0] and a.get("name")==item[1]
             and a.get("size_in_bytes")==item[2]
             and a.get("digest")=="sha256:"+item[3]
             and a.get("expired") is False
             and a.get("workflow_run",{}).get("id")==run_id
             and a.get("workflow_run",{}).get("head_sha")==head,
             "BYTE_AUTHENTIC_B10_ORIGINAL_NATIVE_ARCHIVES_"+key)
    return {k:v[0] for k,v in ARCHIVES.items()}

def extract_observer(root,provider,source):
    need(provider in observer.PROVIDERS,"ONLY_FIVE_SOURCE_REVIEWERS")
    root=Path(root)
    source_by_id={row["id"]:row for row in source}
    result=read(root/"RESULT.json")
    expected_run={"run_id":str(REVIEW_RUN),"commit":REVIEW_COMMIT,
                  "source_run":SOURCE_RUN,"source_commit":SOURCE_COMMIT}
    need(result.get("schema")==observer.SCHEMA
         and result.get("provider")==provider
         and result.get("run")==expected_run
         and result.get("source_artifact_id")==ARCHIVES["source"][0]
         and result.get("attempted")==18 and result.get("complete")==18
         and result.get("denominator")==18
         and result.get("missing_or_failed")==0
         and result.get("status")=="COMPLETE"
         and result.get("initialization_error") is None
         and result.get("target_conditioned") is False
         and result.get("training_allowed") is False
         and result.get("human_gold") is False
         and result.get("new_tts_calls")==0
         and len(result.get("rows",[]))==18,
         "EXACT_COMPLETE_18_SAVED_UNPROMPTED_NATIVE_CAPTURES")
    model=read(root/"MODEL.json")
    if provider in policy.TEXT_MODELS:
        need(model.get("observer")==observer.identity(provider)
             and model.get("target_conditioned") is False
             and model.get("initialization_model_calls")==0,
             "EXACT_FROZEN_ASR_SOURCE_MODEL")
        vocab=None
    else:
        name,revision=observer.identity(provider)
        index=0 if provider=="xlsr" else 1
        need(model.get("model")==name
             and model.get("revision")==revision
             and model.get("weights_sha256")==phones.WEIGHT_SHA[index]
             and model.get("target_conditioned") is False,
             "EXACT_FROZEN_NATIVE_PHONE_SOURCE_MODEL")
        vocab_doc=read(root/"VOCAB.json")
        vocab=vocab_doc["vocab"]
        need(vocab_doc.get("model")==name
             and vocab_doc.get("revision")==revision
             and vocab_doc.get("pad_token_id")==0
             and type(vocab)==dict and len(vocab)==392
             and len(set(vocab.values()))==392,
             "EXACT_NATIVE_CTC_PHONE_VOCAB")
    values={}
    for entry in result["rows"]:
        sid=entry.get("id")
        need(sid in source_by_id and sid not in values
             and entry.get("provider")==provider
             and entry.get("source_pcm_sha256")==source_by_id[sid]["pcm_sha256"]
             and entry.get("source_wav_sha256")==source_by_id[sid]["wav_sha256"]
             and entry.get("source_samples")==source_by_id[sid]["samples"]
             and entry.get("status")=="success"
             and entry.get("target_conditioned") is False
             and entry.get("training_allowed") is False
             and entry.get("input_coverage_complete") is True
             and read(root/"rows"/sid/"CAPTURE.json")==entry
             and (root/"rows"/sid/"STARTED.json").is_file()
             and not (root/"rows"/sid/"FAILED.json").exists(),
             "ALL_ORIGINAL_PCM_SHA_AND_FULL_OBSERVER_CAPTURES")
        if vocab is None:
            txt=entry.get("raw_transcript")
            need(type(txt) is str and 0<len(txt)<=4096
                 and not any(ord(ch)<32 for ch in txt),
                 "REAL_RAW_COMPLETE_ASR_TEXT")
            values[sid]=txt
        else:
            raw=c.read(root/"logits"/(sid+".f32"),40*1024*1024)
            n=entry.get("logit_frames")
            need(type(n) is int and 0<n<=2000
                 and entry.get("vocab_size")==392
                 and len(raw)==n*392*4
                 and sha(raw)==entry.get("logits_sha256"),
                 "EXACT_ORIGINAL_FULL_PHONE_LOGIT_HASH")
            indices=[]
            for frame in struct.iter_unpack("<392f",raw):
                need(all(math.isfinite(x) for x in frame),
                     "FINITE_COMPLETE_NATIVE_PHONE_LOGITS")
                indices.append(max(range(392),key=frame.__getitem__))
            need(phones.ctc(indices,vocab,0)==entry.get("raw_phone_tokens"),
                 "SAVED_ORIGINAL_PHONE_CTC_REPLAY_PARITY")
            values[sid]={"logits":raw,"vocab":vocab,
                         "sha256":entry["logits_sha256"],"pad_token_id":0}
    need(set(values)==set(source_by_id) and len(values)==18,
         "EXACT_18_SOURCE_ID_NATIVE_JOIN")
    return values,{"provider":provider,"complete":18,"attempted":18}

def decide_one(row,transcripts,phonesets):
    observed=policy.decide({**row,"pcm_samples":row["samples"]},
                           transcripts,phonesets)
    if observed["tier"] in ("silver","weak") and (
            observed["class"]!=row["requested_class"]):
        observed={"tier":"unknown","class":None,
                  "reason":"REQUEST_CONTRADICTS_INDEPENDENT_ACOUSTIC_LABEL",
                  "margins":{},"training_allowed":False}
    need(observed["tier"] in ("silver","weak","unknown")
         and observed["class"] in (0,2,None)
         and (observed["tier"]=="unknown")==(observed["class"] is None)
         and observed["training_allowed"] is False,
         "ONLY_VALID_FROZEN_SOURCE_POLICY_DECISIONS")
    return observed

def admission(rows,decisions):
    need(len(rows)==len(decisions)==18,
         "COMPLETE_B10_SOURCE_DECISIONS")
    by_id={row["id"]:row for row in rows}
    need(len(by_id)==18 and {x["id"] for x in decisions}==set(by_id),
         "NO_MISSING_DUPLICATE_ORIGINAL_SOURCE")
    train=[x for x in decisions if x["role"]=="train"]
    dev=[x for x in decisions if x["role"]=="development"]
    need(len(train)==6 and len(dev)==12
         and len({x["source_group_id"] for x in train})==6
         and len({x["source_group_id"] for x in dev})==6
         and not ({x["source_group_id"] for x in train} &
                  {x["source_group_id"] for x in dev}),
         "TRAIN_DEV_SOURCE_GROUP_SEPARATION")
    train_accepted=[x for x in train if x["observed_class"]==0
                    and x["tier"] in ("silver","weak")]
    groups=defaultdict(list)
    for x in dev: groups[x["source_group_id"]].append(x)
    qualified=[]
    for group,content in groups.items():
        need(len(content)==2 and {r["cell_id"] for r in content}==
             {"mero-fast","ordinary-time"},
             "COMPLETE_NEW_DEV_POSITIVE_ORDINARY_PAIR")
        if {x["observed_class"] for x in content}=={0,2}:
            qualified.append(group)
    # TRAIN-only machine qualification is not a general human-gold licence.
    # If prospective DEV is insufficient, never start a new optimizer.
    full_gate=len(train_accepted)==6 and len(qualified)>=2
    return {"train_hard_negative_qualified":len(train_accepted),
            "train_qualified_source_groups":
                sorted(x["source_group_id"] for x in train_accepted),
            "train_unknown":6-len(train_accepted),
            "development_qualified_groups":len(qualified),
            "development_qualified_source_groups":sorted(qualified),
            "development_machine_positives":sum(
                x["observed_class"]==2 for x in dev),
            "development_machine_ordinary":sum(
                x["observed_class"]==0 for x in dev),
            "development_unknown":sum(
                x["observed_class"] is None for x in dev),
            "minimum_6_train_2_dev_gate_passed":full_gate,
            "B10_research_optimizer_allowed":full_gate,
            "training_candidate_role":"TRAIN_ONLY",
            "development_audio_in_optimizer":False}

def audit(source,train_source,train_screen):
    x=observer.source_rows(source,train_source,train_screen)
    need(len(x)==18,"EXACT_SOURCE_READBACK")
    return {"status":"EXACT_SOURCE_AND_ROLE_RECHECK_NO_NEURAL_FORWARD",
            "rows":18,"train":6,"development":12,
            "old_training_group_identity_authenticated":True}

def screen(source,train_source,train_screen,archives,output):
    evidence=guard()
    output=Path(output)
    need(output.is_absolute() and not output.exists()
         and output.parent.is_dir(),
         "ONE_NEW_B10_SAVED_QA_OUTPUT_ONLY")
    output.mkdir()
    c.record(output/"STARTED.json",{
        "schema":SCHEMA,"run":evidence,"expected_originals":18,
        "expected_native_attempts":90,
        "new_model_forwards":0,"new_optimizer_updates":0,
        "cal_locked_opened":False})
    try:
        origin=remote(os.environ.get("GH_TOKEN"))
        source_rows=observer.source_rows(source,train_source,train_screen)
        need(len(source_rows)==18,
             "ORIGINAL_18_SOURCE_PCM_AND_LINEAGE")
        results={};reports=[]
        for provider in observer.PROVIDERS:
            results[provider],r=extract_observer(
                Path(archives)/provider,provider,source_rows)
            reports.append(r)
        decisions=[]
        for row in source_rows:
            sid=row["id"]
            observed=decide_one(row,
                {k:results[k][sid] for k in policy.TEXT_MODELS},
                {k:results[k][sid] for k in policy.PHONE_MODELS})
            need(row["training_allowed"] is False
                 and row["human_gold"] is False
                 and row["observed_class"] is None,
                 "UNTRUSTED_SOURCE_REQUEST_NOT_GOLD")
            decisions.append({
                "id":sid,"role":row["role"],"cell_id":row["cell_id"],
                "speaker_index":row["speaker_index"],
                "source_group_id":row["source_group_id"],
                "requested_class":row["requested_class"],
                "requested_text":row["requested_text"],
                "pcm_sha256":row["pcm_sha256"],
                "wav_sha256":row["wav_sha256"],
                "samples":row["samples"],
                "tier":observed["tier"],
                "observed_class":observed["class"],
                "reason":observed["reason"],
                "phone_margins":observed["margins"],
                "human_gold":False,
                "training_allowed_by_source_request":False})
        result=admission(source_rows,decisions)
        doc={"schema":SCHEMA,
             "status":("PASS_B10_TRAIN_DEV_MACHINE_RESEARCH_GATE_ONLY"
                       if result["minimum_6_train_2_dev_gate_passed"]
                       else "STOP_B10_INDEPENDENT_SOURCE_QUALITY_INSUFFICIENT"),
             "source_run":SOURCE_RUN,"observer_run":REVIEW_RUN,
             "original_artifacts":origin,
             "complete_native_attempts":90,
             "machine_only_not_human_gold":True,
             "source_counts":{"train":6,"development":12},
             "tier_counts":dict(Counter(x["tier"] for x in decisions)),
             "reason_counts":dict(Counter(x["reason"] for x in decisions)),
             "admission":result,"rows":decisions,
             "new_tts_asr_phone_calls":0,"new_kws_model_forwards":0,
             "new_optimizer_updates":0,"no_old_DEV520_528_training":True,
             "cal_locked_opened":False,"locked_test_opened":False,
             "physical_ssc305":"NOT_RUN",
             "natural_far_per_hour":"NOT_MEASURED",
             "product_promotion_allowed":False}
        c.record(output/"SCREEN.json",doc)
        print(json.dumps({k:doc[k] for k in
              ("status","complete_native_attempts","tier_counts",
               "reason_counts","admission","new_optimizer_updates",
               "product_promotion_allowed")},sort_keys=True))
    except BaseException as exc:
        c.record(output/"STOP.json",{
            "schema":SCHEMA,"status":"STOP_B10_AUTHENTIC_SAVED_SOURCE_REVIEW",
            "error_type":type(exc).__name__,
            "error":str(exc)[:500],"new_optimizer_updates":0,
            "cal_locked_opened":False,
            "product_promotion_allowed":False})
        raise

def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    for f in ("--source-check","--guard","--audit","--screen"):
        mode.add_argument(f,action="store_true")
    for v in ("source","train-source","train-screen","archives","output"):
        p.add_argument("--"+v,type=Path)
    a=p.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.source is not None and a.train_source is not None
         and a.train_screen is not None,
         "ONLY_ORIGINAL_NEW_B10_AND_OLD_TRAIN_SOURCE_ZIPS")
    if a.audit:
        print(json.dumps(audit(a.source,a.train_source,a.train_screen),
                         sort_keys=True));return
    need(a.archives is not None and a.output is not None,
         "FIVE_ORIGINAL_BLIND_NATIVE_OUTPUTS_REQUIRED")
    screen(a.source,a.train_source,a.train_screen,a.archives,a.output)

if __name__=="__main__":
    main()
