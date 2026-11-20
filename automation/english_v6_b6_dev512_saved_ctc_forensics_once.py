#!/usr/bin/env python3
"""Read-only authentic 126-case B4/B6 DEV512 acoustic failure forensics.

Reuse the immutable frozen-model result ZIP. No model inference, retraining,
new audio, ASR/phone forwards, decoder policy selection or CAL/LOCKED access.
Fixed gain variants are correlated views of six original positive sources.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

REPO="jiying2007/kws-data"
SCHEMA="english-v6-b6-dev512-saved-126-ctc-forensics-v1"
PARENT="d9bbb5d0ea29ea2774067734ba24d55eccef97fc"
ORIGINAL={
    "run":44710802470,
    "head":PARENT,
    "id":14616493798,
    "name":"english-v6-b6-dev512-three-frozen-paired-44710802470",
    "size":41502,
    "digest":"b900473e57a5b53793a42020838371c1c12f4ee70d73480f2f68c03aab37801f",
}
SOURCE_ARTIFACT=14616065335
SCREEN_ARTIFACT=14616173070
MODEL_PT={
    "B4_half":"c795f8b5f6bf8e9c37d195126fc25006dbde92fb71591473c05854e3d3698f15",
    "B6_control":"c9cac4c16a9d574b6bc9baef14ac239517a2ed9bfbe2e3ff2fb69972587c9430",
    "B6_consistency":"0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04",
}
RESULT_SCHEMA="english-v6-b6-dev512-three-frozen-new-source-original-paired-v1"
GAINS=("1.0","0.75","0.5")
HITS={
    "1.0":{"B4_half":3,"B6_control":3,"B6_consistency":5},
    "0.75":{"B4_half":4,"B6_control":4,"B6_consistency":3},
    "0.5":{"B4_half":1,"B6_control":1,"B6_consistency":1},
}
ORDINARY_EVENTS={
    "1.0":{"B4_half":0,"B6_control":0,"B6_consistency":1},
    "0.75":{"B4_half":0,"B6_control":0,"B6_consistency":0},
    "0.5":{"B4_half":0,"B6_control":0,"B6_consistency":0},
}
MISSING_MEH={
    "1.0":{"B4_half":0,"B6_control":0,"B6_consistency":0},
    "0.75":{"B4_half":0,"B6_control":0,"B6_consistency":0},
    "0.5":{"B4_half":1,"B6_control":1,"B6_consistency":1},
}

def need(ok,reason):
    if not ok:raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def record(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write("\n")

def source_check():
    need(len(ORIGINAL["digest"])==64
         and ORIGINAL["head"]==PARENT
         and ORIGINAL["run"]==44710802470
         and ORIGINAL["id"]==14616493798
         and all(re.fullmatch("[0-9a-f]{64}",d) for d in MODEL_PT.values())
         and all(set(HITS[g])==set(MODEL_PT)
                 and set(ORDINARY_EVENTS[g])==set(MODEL_PT)
                 and set(MISSING_MEH[g])==set(MODEL_PT) for g in GAINS)
         and HITS["0.5"]==dict.fromkeys(MODEL_PT,1)
         and ORDINARY_EVENTS["1.0"]["B6_consistency"]==1,
         "EXACT_REAL_B6_FROZEN_126_CASE_IMMUTABLE_SAVED_INPUT")
    return {"schema":SCHEMA,
            "status":"SOURCE_ONLY_126_PREVIOUSLY_EXECUTED_FROZEN_CASES",
            "result_run":ORIGINAL["run"],"result_artifact":ORIGINAL["id"],
            "source_artifact":SOURCE_ARTIFACT,
            "machine_screen_artifact":SCREEN_ARTIFACT,
            "positive_originals":6,"ordinary_originals":8,"unknown_excluded":18,
            "qualified_development_groups":5,"original_model_forward_cases":126,
            "new_tts_asr_phone_model_forwards":0,"new_optimizer_updates":0,
            "no_new_calibration_or_locked_access":True,
            "gain_derivatives_add_independent_source_count":False,
            "model_or_threshold_selection_allowed":False,
            "product_promotion_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    e=os.environ if env is None else env
    need(all(e.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"EXACT_FIRST_PUBLIC_MAIN_SAVED_ONLY")
    head=e.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",e.get("GITHUB_RUN_ID","")) is not None,
         "EXACT_REAL_SAVED_MAIN_EXECUTION_IDENTITY")
    if event is None:
        event=json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],text=True).strip()
    need(event.get("before")==PARENT and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False
         and event.get("deleted") is False
         and parent==PARENT,"REAL_B6_126_CASE_IMMEDIATE_FIRST_SUCCESSOR_ONLY")
    return {"run_id":e["GITHUB_RUN_ID"],"head":head,"parent":PARENT}

def verify_remote(token):
    need(type(token) is str and len(token)>=10,"READ_ONLY_GITHUB_ACTIONS_TOKEN")
    def get(endpoint):
        request=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+endpoint,
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"kws-v6-b6-saved-126-only"})
        with urllib.request.urlopen(request,timeout=40) as stream:
            raw=stream.read(1000001)
        need(len(raw)<=1000000,"ONLY_BOUNDED_GITHUB_METADATA")
        return json.loads(raw)
    run=get("/actions/runs/"+str(ORIGINAL["run"]))
    need(run.get("id")==ORIGINAL["run"]
         and run.get("head_sha")==ORIGINAL["head"]
         and run.get("head_branch")=="main"
         and run.get("event")=="push"
         and run.get("run_attempt")==1
         and run.get("status")=="completed"
         and run.get("conclusion")=="success",
         "AUTHENTIC_COMPLETE_126_CPU_MODEL_FORWARDS")
    art=get("/actions/artifacts/"+str(ORIGINAL["id"]))
    need(art.get("id")==ORIGINAL["id"]
         and art.get("name")==ORIGINAL["name"]
         and art.get("size_in_bytes")==ORIGINAL["size"]
         and art.get("digest")=="sha256:"+ORIGINAL["digest"]
         and art.get("expired") is False
         and art.get("workflow_run",{}).get("id")==ORIGINAL["run"]
         and art.get("workflow_run",{}).get("head_sha")==ORIGINAL["head"],
         "EXACT_126_FROZEN_RESULT_ORIGINAL_ARTIFACT")
    return {"run_id":ORIGINAL["run"],"artifact_id":ORIGINAL["id"],
            "archive_zip_sha256":ORIGINAL["digest"]}

def ctc_events(tokens):
    need(type(tokens) is list
         and all(type(t) is int and t in (1,2,3) for t in tokens),
         "REAL_COLLAPSED_CTC_NONBLANK_TOKEN_PATH")
    return sum(tokens[i:i+3]==[1,2,3] for i in range(max(0,len(tokens)-2)))

def analyze(result):
    need(result.get("schema")==RESULT_SCHEMA
         and result.get("status")=="B6_NEW_SOURCE_FROZEN_THREE_WAY_DEV_DIAGNOSTIC_ONLY"
         and result.get("model_pt_sha256")==MODEL_PT
         and result.get("qualified_positive_source_groups")==5
         and result.get("B4_half_B6_control_identical_tensors") is True
         and result.get("derived_conditions_do_not_increase_original_denominator") is True
         and result.get("same_tts_engine_not_human_gold") is True
         and result.get("cal_locked_opened") is False
         and result.get("new_optimizer_steps")==0
         and result.get("new_tts_asr_calls")==0
         and result.get("model_selection_allowed") is False
         and result.get("winner") is None
         and result.get("product_promotion_allowed") is False
         and result.get("conditions")==[1.0,0.75,0.5]
         and result.get("artifacts")=={
            "source":SOURCE_ARTIFACT,"screen":SCREEN_ARTIFACT,
            "models":{"B4":14597862669,"B6":14615025178}},
         "NO_NEW_MODEL_OR_HUMAN_GOLD_OR_UNTRUSTED_SOURCE_PROMOTION")
    rows=result.get("per_original")
    need(type(rows) is list and len(rows)==14
         and Counter(x.get("positive") for x in rows)=={True:6,False:8}
         and len({r.get("id") for r in rows})==14
         and len({r.get("pcm_sha256") for r in rows})==14
         and all(r.get("role")=="development"
                 and r.get("machine_tier") in ("silver","weak")
                 and re.fullmatch("[0-9a-f]{64}",r.get("pcm_sha256",""))
                 for r in rows),
         "REAL_14_DISTINCT_MACHINE_QUALIFIED_DEV512_ORIGINALS_ONLY")
    source_receipt=result.get("source_receipt")
    need(type(source_receipt) is dict
         and source_receipt.get("status")=="AUDITED_SIX_NEW_POSITIVE_EIGHT_ORDINARY"
         and source_receipt.get("new_originals")==14
         and len(source_receipt.get("qualified_positive_source_groups",[]))==5
         and source_receipt.get("no_cal_locked") is True
         and source_receipt.get("human_gold") is False,
         "INDEPENDENT_DEV512_SOURCE_QA_PROVENANCE")
    positive=[r for r in rows if r["positive"]]
    ordinary=[r for r in rows if not r["positive"]]
    summary=result.get("summary")
    need(type(summary) is dict and set(summary)==set(GAINS),
         "ONLY_THREE_REGISTERED_GAINS")
    by_gain={}
    discordant={}
    for gain in GAINS:
        need(set(summary[gain])==set(MODEL_PT),
             "EXACT_THREE_FROZEN_43K_COMPARATORS")
        for r in rows:
            cond=r.get("conditions",{}).get(gain)
            need(type(cond) is dict and set(cond)==set(MODEL_PT),
                 "ALL_14_ORIGINALS_THREE_MODELS_ALL_GAINS")
            for name,c in cond.items():
                n=c.get("full_phrase_event_count")
                path=c.get("collapsed_tokens")
                need(type(n) is int and n>=0
                     and type(c.get("frames")) is int and c["frames"]>0
                     and type(path) is list and len(path)<=c["frames"]
                     and c.get("first_frame_argmax") in (0,1,2,3)
                     and c.get("any_complete_event") is (n>0)
                     and c.get("ctc_extra_events")==max(0,n-1)
                     and ctc_events(path)==n
                     and c.get("any_meh_argmax") is (2 in path),
                     "RECOMPUTED_EXACT_CTC_GREEDY_FULL_WORD_PATH")
            need(cond["B4_half"]==cond["B6_control"],
                 "EXACT_PARENT_AND_CONTROL_PER_ORIGINAL_CTC_PARITY")
        by_gain[gain]={}
        changed=[]
        for name in MODEL_PT:
            actual_hits=sum(r["conditions"][gain][name]["any_complete_event"]
                            for r in positive)
            actual_false=sum(r["conditions"][gain][name]["full_phrase_event_count"]
                             for r in ordinary)
            false_clips=sum(r["conditions"][gain][name]["any_complete_event"]
                            for r in ordinary)
            meh_missing=sum(not r["conditions"][gain][name]["any_meh_argmax"]
                            for r in positive)
            entry=summary[gain][name]
            need(actual_hits==HITS[gain][name]==entry.get("positive_hits")
                 and actual_false==ORDINARY_EVENTS[gain][name]
                    ==entry.get("ordinary_false_or_extra_events")
                 and false_clips==entry.get("ordinary_complete_event_clips")
                 and entry.get("positive_originals")==6
                 and entry.get("ordinary_originals")==8
                 and meh_missing==MISSING_MEH[gain][name]
                    ==entry.get("positive_missing_meh_argmax"),
                 "PER_ORIGINAL_SUMS_MATCH_REAL_126_CASE_RESULT")
            by_gain[gain][name]={
                "positive_originals":6,"positive_hits":actual_hits,
                "ordinary_originals":8,"ordinary_events":actual_false,
                "ordinary_complete_event_clips":false_clips,
                "positive_missing_meh":meh_missing}
        for r in positive:
            parent=r["conditions"][gain]["B4_half"]
            challenger=r["conditions"][gain]["B6_consistency"]
            if parent["any_complete_event"]!=challenger["any_complete_event"]:
                changed.append({
                    "source_id":r["id"],"source_group_id":r["source_group_id"],
                    "pcm_sha256":r["pcm_sha256"],
                    "B4_hit":parent["any_complete_event"],
                    "B6_hit":challenger["any_complete_event"],
                    "B4_ctc":parent["collapsed_tokens"],
                    "B6_ctc":challenger["collapsed_tokens"],
                    "B6_has_meh":challenger["any_meh_argmax"]})
        need(sum(int(x["B6_hit"])-int(x["B4_hit"]) for x in changed)
             ==HITS[gain]["B6_consistency"]-HITS[gain]["B4_half"],
             "ONLY_REAL_PER_ORIGINAL_POSITIVE_DISCORDANCE")
        discordant[gain]=changed
    false_sources=[]
    for row in ordinary:
        for gain in GAINS:
            entry=row["conditions"][gain]["B6_consistency"]
            if entry["any_complete_event"]:
                false_sources.append({
                    "source_id":row["id"],"source_group_id":row["source_group_id"],
                    "pcm_sha256":row["pcm_sha256"],
                    "gain":gain,
                    "B4_ctc":row["conditions"][gain]["B4_half"]["collapsed_tokens"],
                    "B6_ctc":entry["collapsed_tokens"],
                    "B6_complete_events":entry["full_phrase_event_count"]})
    need(len(false_sources)==1
         and false_sources[0]["gain"]=="1.0"
         and false_sources[0]["B6_complete_events"]==1,
         "ONE_REAL_B6_SYNTHETIC_ORDINARY_FALSE_WAKE_ORIGIN")
    return {
        "schema":SCHEMA,"status":"PASS_IMMUTABLE_B6_126_CASE_CTC_FORENSICS_RESEARCH_ONLY",
        "input_artifact":ORIGINAL["id"],"saved_model_forwards":126,
        "qualified_positive_origins":6,"qualified_ordinary_origins":8,
        "qualified_source_groups":5,"unknown_excluded":18,
        "by_gain":by_gain,"positive_discordances":discordant,
        "ordinary_false_wake_origins":false_sources,
        "normal_gain_B6_gain_minus_B4_positive_hits":2,
        "medium_gain_B6_gain_minus_B4_positive_hits":-1,
        "half_gain_B6_gain_minus_B4_positive_hits":0,
        "low_gain_positive_hit_count":1,
        "all_three_gains_derived_from_same_originals":True,
        "only_normal_gain_full_frame_logits_saved":True,
        "natural_far_per_hour":"NOT_MEASURED",
        "real_human_3_5m_frr":"NOT_MEASURED",
        "physical_ssc305":"NOT_RUN",
        "new_audio_asr_phone_model_forwards":0,
        "new_optimizer_updates":0,
        "no_calibration_or_locked_access":True,
        "model_selection_allowed":False,
        "threshold_selection_allowed":False,
        "product_promotion_allowed":False,
    }

def evaluate(archive,out):
    source_check()
    run=guard()
    output=Path(out)
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "ONE_NEW_SAVED_ONLY_RECEIPT_DIRECTORY")
    output.mkdir()
    record(output/"STARTED.json",{"schema":SCHEMA,"run":run,
           "source_artifact":ORIGINAL["id"],"new_model_forwards":0})
    try:
        remote=verify_remote(os.environ.get("GH_TOKEN"))
        archive=Path(archive)
        need(archive.is_file() and not archive.is_symlink()
             and archive.stat().st_size==ORIGINAL["size"]
             and sha(archive.read_bytes())==ORIGINAL["digest"],
             "AUTHENTIC_126_CASE_FROZEN_RESULT_ZIP_DIGEST")
        with zipfile.ZipFile(archive) as z:
            need(z.testzip() is None,"COMPLETE_SAVED_ZIP_INTEGRITY")
            result=json.loads(z.read("RESULT.json"))
            logits=z.read("ORIGINAL_FRAME_LOGITS.npz")
            need(sha(logits)==result.get("original_frame_logits_sha256"),
                 "EXACT_SAVED_NORMAL_GAIN_PER_FRAME_LOGITS")
        evidence=analyze(result)
        evidence["authenticated_input"]=remote
        evidence["readback_run"]=run
        record(output/"FINDINGS.json",evidence)
        print(json.dumps({
            "status":evidence["status"],
            "normal_gain_synthetic_false_events":
                len(evidence["ordinary_false_wake_origins"]),
            "normal_gain_positive_discordances":
                len(evidence["positive_discordances"]["1.0"]),
            "medium_gain_positive_discordances":
                len(evidence["positive_discordances"]["0.75"]),
            "half_gain_positive_discordances":
                len(evidence["positive_discordances"]["0.5"]),
            "false_source_ids":
                [r["source_id"] for r in evidence["ordinary_false_wake_origins"]],
            "new_model_forwards":0},sort_keys=True))
    except BaseException as exc:
        record(output/"STOP.json",{"schema":SCHEMA,
               "status":"STOP_SAVED_B6_FORensics",
               "error_type":type(exc).__name__,"error":str(exc)[:500],
               "new_model_forwards":0,
               "product_promotion_allowed":False})
        raise

def main():
    parser=argparse.ArgumentParser()
    flags=parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--source-check",action="store_true")
    flags.add_argument("--evaluate",action="store_true")
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.source_check:
        print(json.dumps(source_check(),sort_keys=True))
        return
    need(args.archive is not None and args.output is not None,
         "ORIGINAL_126_RESULT_ARCHIVE_AND_NEW_OUTPUT_REQUIRED")
    evaluate(args.archive,args.output)

if __name__=="__main__":
    main()
