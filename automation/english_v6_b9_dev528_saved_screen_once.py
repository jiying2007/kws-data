"""V6 Hey Mero: screen 299/300 SAVED real unprompted native observations on B9 DEV60.

Twelve post-B9 unexposed source groups: Piper 528..539 DEVELOPMENT only.
All five ran 60 original attempts; exactly one Parakeet invalid-transcript failure is retained.
No model forwards, training, threshold tuning, calibration/locked audio access.
A nearphone or unsupported positive/ordinary label stays UNKNOWN.
Machine tier != real-human gold or SSC305 product release.
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

observer=load("english_v6_b9_dev528_five_native_once")
policy=load("english_fresh24_ctc_admission")
phones=load("english_hifi_phones_once")
c=load("english_hifi_micro")
SCHEMA="english-v6-b9-dev528-saved-five-native-source-screen-v1"
REPO="jiying2007/kws-data"
BEFORE="fedc502a5b38445be1c6b143a94553a38850bcc1"
ORIGINAL_RUN=44870958716
ORIGINAL_SHA="fedc502a5b38445be1c6b143a94553a38850bcc1"
ORIGINAL_STATUS="success"
SOURCE_RUN=44870282676
SOURCE={"id":14693632800,"name":"english-v6-b9-fresh-dev528-original-44870282676",
        "bytes":1206914,
        "digest":"sha256:d606ee883a1abd38cc95a278f6b7b7064476c8b00add1163e27c247c41ab7032"}
ARCHIVES={
    "qwen":(14694033180,59969,"e45858908677004940ac34caa4a1fa81356d0c19e37b34418b8f1962c1096547"),
    "whisper":(14693594464,60204,"da109a0b8ab0767d3f7fd5240a34db333ddfcf5104a5c27793c1aa357bb4dce6"),
    "parakeet":(14693289906,60242,"8dfa56b143eeca570ec9db7505676275624b6fb78c3a2f56fa538183b1d7598d"),
    "xlsr":(14694245820,3024889,"7f69eb00b09e8148841420c2d6fdcfe0ffeed052eb096674d56bda11fb303136"),
    "lv60":(14693522883,3021745,"5f69bf525af9890fe046715e02f5a4791fbd78a227cd72a45c261a98dff09884"),
}
REVIEW_COUNT=60
MIN_DEVELOPMENT_GROUPS=2

def need(v,reason):
    if not v:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read_json(path):
    return json.loads(c.read(Path(path)))

def source_check():
    x=observer.source_check()
    need(x["source_run"]==SOURCE_RUN
         and x["source_artifact_id"]==SOURCE["id"]
         and x["source_rows"]==REVIEW_COUNT
         and x["review_rows"]==REVIEW_COUNT
         and x["locked_test_label_review_clips"]==0
         and x["train_dev_calibration_review_clips"]==REVIEW_COUNT
         and observer.BEFORE=="2bd78ea27b2439e92c82ee5672712eba2e86405a"
         and tuple(observer.PROVIDERS)==("qwen","whisper","parakeet","xlsr","lv60")
         and observer.SOURCE["digest"]==SOURCE["digest"]
         and observer.SOURCE["size"]==SOURCE["bytes"]
         and ORIGINAL_RUN==44870958716
         and ORIGINAL_SHA==BEFORE
         and ORIGINAL_STATUS=="success"
         and SOURCE_RUN==44870282676
         and MIN_DEVELOPMENT_GROUPS==2
         and len(ARCHIVES)==5
         and len({v[0] for v in ARCHIVES.values()})==5
         and all(re.fullmatch("[0-9a-f]{64}",v[2]) for v in ARCHIVES.values())
         and policy.MIN_QUALIFIED=={
             "train":6,"development":2,"calibration":2,"locked-test":2},
         "EXACT_PROSPECTIVE_B9_DEV528_FIVE_OBSERVER_PINS")
    return {"status":"SAVED_PROSPECTIVE_B9_DEV528_299_OF_300_REAL_OBSERVATIONS_PINNED",
            "original_observer_run":ORIGINAL_RUN,
            "original_observer_head":ORIGINAL_SHA,
            "saved_source_rows":60,"review_rows":REVIEW_COUNT,
            "observer_archive_count":len(ARCHIVES),
            "new_audio_or_model_forwards":0,
            "new_optimizer_updates":0,
            "new_kws_model_forwards":0,
            "new_locked_test_forwards":0,
            "training_allowed":False,"model_promotion_allowed":False}

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
                     "User-Agent":"v6-mero-saved-evidence-readback"})
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
          "name":f"b9-dev528-native-{name}-{ORIGINAL_RUN}",
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

PARAKEET_FAILED_ID="merov6b9dev12-531-mero-slow"
PARAKEET_FAILED_ERROR="ValueError:RAW_UNPROMPTED_COMPLETE_ASR_TEXT"

def review_packet(root,provider,rows):
    """Authenticate every native attempt, including one pinned Parakeet failure."""
    need(provider in observer.PROVIDERS,"KNOWN_FIVE_BLIND_PROVIDER")
    root=Path(root)
    src={r["id"]:r for r in rows}
    need(len(src)==REVIEW_COUNT,"EXACT_SOURCE_AUDIT_BEFORE_REVIEWS")
    result=read_json(root/"RESULT.json")
    permitted_missing=1 if provider=="parakeet" else 0
    expected=REVIEW_COUNT-permitted_missing
    need(result.get("schema")==observer.SCHEMA
         and result.get("provider")==provider
         and result.get("run")==original_identity()
         and result.get("source_artifact_id")==SOURCE["id"]
         and result.get("denominator")==REVIEW_COUNT
         and result.get("attempted")==REVIEW_COUNT
         and result.get("complete")==expected
         and result.get("missing_or_failed")==permitted_missing
         and result.get("new_tts_calls")==0
         and result.get("target_conditioned") is False
         and result.get("human_gold") is False
         and result.get("training_allowed") is False
         and result.get("initialization_error") is None
         and result.get("status")==(
             "PARTIAL_WITH_FAILURE_RECEIPTS" if permitted_missing else "COMPLETE")
         and type(result.get("rows")) is list
         and len(result["rows"])==REVIEW_COUNT,
         "EXACT_NATIVE_ATTEMPT_COUNTS_NO_MASKING_FAILURES")
    model=read_json(root/"MODEL.json")
    if provider in ("xlsr","lv60"):
        source_id,revision=observer.identity(provider)
        idx=0 if provider=="xlsr" else 1
        need(model.get("model")==source_id
             and model.get("revision","").startswith(revision)
             and model.get("weights_sha256")==phones.WEIGHT_SHA[idx]
             and model.get("target_conditioned") is False,
             "UNCHANGED_FROZEN_NATIVE_PHONE_WEIGHT_IDENTITY")
        vocab_doc=read_json(root/"VOCAB.json")
        vocab=vocab_doc["vocab"]
        need(vocab_doc.get("model")==source_id
             and vocab_doc.get("revision")==model["revision"]
             and vocab_doc.get("pad_token_id")==0
             and type(vocab) is dict and len(vocab)==392
             and len(set(vocab.values()))==392,
             "EXACT_UNCHANGED_392_PHONE_VOCAB")
    else:
        need(model.get("observer")==observer.identity(provider)
             and model.get("target_conditioned") is False
             and model.get("initialization_model_calls")==0,
             "UNCHANGED_BLIND_ASR_MODEL_WEIGHTS")
        vocab=None
    saved={};failed=[]
    for original in result["rows"]:
        sid=original.get("id")
        need(sid in src and sid not in saved
             and original.get("provider")==provider
             and original.get("source_wav_sha256")==src[sid]["wav_sha256"]
             and original.get("source_pcm_sha256")==src[sid]["pcm_sha256"]
             and original.get("source_samples")==src[sid]["samples"]
             and original.get("target_conditioned") is False
             and original.get("training_allowed") is False,
             "ORIGINAL_COMPLETE_PCM_PROVENANCE_PER_OBSERVER")
        if original.get("status")=="failed":
            need(provider=="parakeet" and sid==PARAKEET_FAILED_ID
                 and original.get("error")==PARAKEET_FAILED_ERROR
                 and read_json(root/"rows"/sid/"FAILED.json")==original
                 and (root/"rows"/sid/"STARTED.json").is_file()
                 and not (root/"rows"/sid/"CAPTURE.json").exists(),
                 "PARAKEET_ONLY_EXACT_ONE_UNRECOVERED_FAIL_CLOSED")
            saved[sid]=None
            failed.append(sid)
            continue
        need(original.get("status")=="success"
             and original.get("input_coverage_complete") is True
             and read_json(root/"rows"/sid/"CAPTURE.json")==original
             and (root/"rows"/sid/"STARTED.json").is_file()
             and not (root/"rows"/sid/"FAILED.json").exists(),
             "ONLY_COMPLETE_UNPROMPTED_OBSERVERS_NO_IMPUTED_FAILURES")
        if provider in ("xlsr","lv60"):
            raw=c.read(root/"logits"/(sid+".f32"),40*1024*1024)
            n=original.get("logit_frames")
            need(type(n) is int and 0<n<=2000
                 and original.get("vocab_size")==392
                 and len(raw)==n*392*4
                 and sha(raw)==original.get("logits_sha256"),
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
            need(type(transcript) is str and 0<len(transcript)<=4096
                 and not any(ord(ch)<32 for ch in transcript),
                 "EXACT_ORIGINAL_UNPROMPTED_ASR_TEXT")
            saved[sid]=transcript
    need(set(saved)==set(src) and len(saved)==REVIEW_COUNT
         and failed==([PARAKEET_FAILED_ID] if provider=="parakeet" else [])
         and sum(v is None for v in saved.values())==permitted_missing,
         "ALL_60_ORIGINAL_NATIVE_IDS_WITH_EXPLICIT_MISSING_ONLY")
    return saved,{"provider":provider,"complete":expected,
                  "attempted":REVIEW_COUNT,
                  "status":result["status"],"missing_ids":failed}

def count_eligible(decisions):
    groups=defaultdict(list)
    for row in decisions:
        groups[row["source_group_id"]].append(row)
    qualified=[]
    for gid,rows in groups.items():
        need(len(rows)==5 and {r["cell_id"] for r in rows}=={
              "mero-fast","mero-slow","ordinary-time",
              "hard-hey-mary","nearphone-meero-challenge"}
             and {r["role"] for r in rows}=={"development"},
             "ONLY_COMPLETE_FIVE_CELL_PROSPECTIVE_B9_DEV_GROUP")
        near=next(r for r in rows
                  if r["cell_id"]=="nearphone-meero-challenge")
        need(near["tier"]=="unknown" and near["observed_class"] is None,
             "NEARPHONE_CHALLENGE_NEVER_TRUSTED_ORDINARY")
        hard=next(r for r in rows if r["cell_id"]=="hard-hey-mary")
        need(hard["observed_class"] in (None,0)
             and (hard["observed_class"] is None
                  or hard["tier"] in ("silver","weak")),
             "HARD_MARY_NOT_POSITIVE_OR_PROMPT_GOLD")
        accepted={r["observed_class"] for r in rows if r["tier"] in ("silver","weak")}
        if {0,2}<=accepted:
            qualified.append(gid)
    return {"development":len(qualified)},sorted(qualified)

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
        need(len(rows)==REVIEW_COUNT,"EXACT_60_NEW_PROSPECTIVE_B9_SOURCE_CLIPS")
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
            if values["parakeet"][sid] is None:
                need(sid==PARAKEET_FAILED_ID,
                     "NO_UNPINNED_NATIVE_SOURCE_MISSING_ALLOWED")
                observed={"tier":"unknown","class":None,
                          "reason":"ONE_AUTHENTIC_PAR-AKEET_CAPTURE_FAILED",
                          "margins":{},"training_allowed":False}
            else:
                observed=policy.decide(
                    {**row,"pcm_samples":row["samples"]},trans,logit)
            if observed["tier"] in ("silver","weak") and (
                    observed["class"]!=row["requested_class"]):
                observed={"tier":"unknown","class":None,
                    "reason":"REQUEST_VERSUS_ACOUSTIC_CONTRADICTION",
                    "margins":{},"training_allowed":False}
            need(row["role"]=="development"
                 and row["training_allowed"] is False
                 and row["human_gold"] is False,
                 "NO_NONDEV_OR_REQUEST_TRUSTED_SOURCE")
            if row["requested_class"] is None:
                need(observed["tier"]=="unknown"
                     and observed["class"] is None,
                     "UNLABELED_NEARPHONE_STAYS_UNKNOWN")
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
        need(len(decision)==60
             and len({r["source_group_id"] for r in decision})==12
             and Counter(r["role"] for r in decision)=={"development":60}
             and Counter(r["requested_class"] for r in decision)=={
                 2:24,0:24,None:12}
             and all(r["human_gold"] is False
                     and r["training_allowed"] is False for r in decision),
             "ONLY_12_NEW_UNEXPOSED_B9_DEV_SOURCE_GROUPS_NO_ROLE_SHIFT")
        qualified,members=count_eligible(decision)
        passes=qualified["development"]>=MIN_DEVELOPMENT_GROUPS
        data={"schema":SCHEMA,
              "status":("PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY"
                        if passes else
                        "STOP_PROSPECTIVE_DEV_SOURCE_QUALITY_INSUFFICIENT"),
              "source_run":SOURCE_RUN,"original_observer_run":ORIGINAL_RUN,
              "observed_original_archives":attest,"provider_reports":reports,
              "source_group_counts":qualified,
              "source_group_identity_members":members,
              "minimum_qualified_development_groups":MIN_DEVELOPMENT_GROUPS,
              "calibration_source_groups_machine_qualified":None,
              "locked_test_source_groups_machine_qualified":None,
              "full_6_2_2_2_gate_passed":False,
              "tier_counts":dict(Counter(x["tier"] for x in decision)),
              "reason_counts":dict(Counter(x["reason"] for x in decision)),
              "observed_label_counts":dict(Counter(str(x["observed_class"])
                   for x in decision if x["tier"] in ("silver","weak"))),
              "nearphone_as_ordinary_negative":False,
              "machine_only_no_human_gold":True,
              "new_audio_or_model_forwards":0,
              "native_attempts":300,"native_successful_captures":299,
              "parakeet_failed_source_id":PARAKEET_FAILED_ID,
              "originals_with_five_complete_native":59,
              "hard_hey_mary_requested_is_not_gold":True,
              "new_optimizer_updates":0,"new_weights":False,
              "new_kws_model_forwards":0,
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
