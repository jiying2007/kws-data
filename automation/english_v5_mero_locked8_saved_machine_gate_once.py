"""Hey Mero V5: exact newlocked8 evidence-only source gate, never a KWS test.

Nine? No: original source-96 quality gate had 11/2/2/1 groups, and exact
new 32 LOCKED-only WAV were generated from 8 wholly unused Piper speakers.
Five blind native model source-label ZIPs have been retained. Rejoin only
original 32 full PCM and original source-bound raw ASR / phone logits;
do not run ASR/phone/KWS again. NEVER treat prompt or nearphone as a label.

Even a successful 6/2/2/2 machine gate does not prove real human speech,
continuous negatives, 3-5m, SSC305 or KWS locked-test performance. This
readback has zero optimizer steps, zero audio generation and zero KWS calls.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(name):
    spec=importlib.util.spec_from_file_location(
        "v5_newlocked_final_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

review=load("english_v5_mero_locked8_five_native_once")
original=load("english_v5_mero_fresh24_full_source_gate_once")
policy=load("english_fresh24_ctc_admission")
c=load("english_hifi_micro")
SCHEMA="english-v5-mero-newlocked8-saved-five-native-6-2-2-2-machine-gate-v1"
REPO="jiying2007/kws-data"
BEFORE="57ff9cff2253510312bc73f27c605aadfb16b555"
SOURCE_RUN=44491090037
SOURCE_HEAD="57fbebdb57d925a6dd984e564611c1bfda4c94bf"
OBSERVER_RUN=44492052970
OBSERVER_HEAD=BEFORE
PRIOR_RUN=44487146445
PRIOR_HEAD="63f1e5be812c27010482b4faaef3df4524710a52"
PRIOR_READER_BLOB="1f63c7a69eccf243a3c257301d1e113dea579c86"
PRIOR_COUNTS={"train":11,"development":2,"calibration":2,"locked-test":1}
MIN_COUNTS={"train":6,"development":2,"calibration":2,"locked-test":2}
SOURCE={"id":14512598412,
        "name":"english-v5-mero-newlocked8-original-44491090037",
        "bytes":648581,
        "digest":"sha256:d95377ce67d4934a491316d19aec16f7c5176f3f161583ab87b508ef840fd9e3"}
PRIOR={"id":14510254835,
       "name":"english-v5-mero-source96-full-machine-gate-44487146445",
       "bytes":5145,
       "digest":"sha256:880767864d1594f3fdfbd3681256c868a77ca0365d9986fc7ab8ace1d7894577"}
ARCHIVES={
  "qwen":(14513072399,32442,"c50d0560a8862c4c54a7e8be2c0cead9fd73bb42cddf50d9c149f44f8e58b8bd"),
  "whisper":(14513571417,32535,"c4f6b1831b017fbf422d4ad9ac5406972998e8b2db3327573b9ed7443ab45012"),
  "parakeet":(14513272487,32476,"9532a081342dbb693af17625c0ae8fa0d688035b13c564e28e4336d78276ee0e"),
  "xlsr":(14512738146,1579703,"ef9a407d856eb2c95f592ae552e6af7758bb65d0ee50b37acbcc7a686107d8e6"),
  "lv60":(14513516650,1578387,"079776a0cc5c9218cdcbd0123cc9ae8ef81071beee4bfd7e4cd52605a81dbfed"),
}

def need(ok,reason):
    if not ok:
        raise ValueError(reason)

def readj(path):
    return json.loads(c.read(Path(path)))

def gitblob(path):
    raw=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()

# Reuse the original source-proof and CTC-logit parser without modifying its
# committed source. Only its module-local dataset identity is rebound to the
# exact new-source, pinned first-main five-native receipt. Product-facing and
# prior research pipeline modules are never changed or reconfigured.
original.sealed=review
original.SOURCE_RUN=SOURCE_RUN
original.SEALED_RUN=OBSERVER_RUN
original.SEALED_COMMIT=OBSERVER_HEAD
original.SOURCE={"id":SOURCE["id"],"name":SOURCE["name"],
                 "bytes":SOURCE["bytes"],"digest":SOURCE["digest"]}

def source_check():
    z=review.source_check()
    need(gitblob(HERE/"english_v5_mero_fresh24_full_source_gate_once.py")
         ==PRIOR_READER_BLOB
         and z["status"]=="ORIGINAL_SOURCE_PINNED"
         and z["source_run"]==SOURCE_RUN
         and z["source_artifact_id"]==SOURCE["id"]
         and z["source_rows"]==32 and z["review_rows"]==32
         and z["kws_model_forwards"]==0
         and z["train_dev_calibration_review_clips"]==0
         and z["locked_test_label_review_clips"]==32
         and review.SOURCE_COMMIT==SOURCE_HEAD
         and review.SOURCE_PARENT==PRIOR_HEAD
         and tuple(review.PROVIDERS)==tuple(policy.TEXT_MODELS)+tuple(policy.PHONE_MODELS)
         and len(ARCHIVES)==5
         and len({x[0] for x in ARCHIVES.values()})==5
         and MIN_COUNTS==policy.MIN_QUALIFIED
         and PRIOR_COUNTS=={"train":11,"development":2,"calibration":2,"locked-test":1}
         and PRIOR["id"]==14510254835,
         "EXACT_NEW32_SOURCE_AND_FROZEN_OLD_11_2_2_1_MACHINE_GATE")
    return {"status":"FROZEN_OLD_GATE_AND_FIVE_REAL_NEW_LOCKED_NATIVE_ZIPS",
            "source_run":SOURCE_RUN,"source_rows":32,
            "new_locked_speakers":8,"prior_groups":PRIOR_COUNTS,
            "original_native_observer_run":OBSERVER_RUN,
            "original_model_forwards_already_retained":160,
            "new_native_or_kws_forward_calls":0,
            "new_optimizer_steps":0,"old_locked_relabel":False,
            "training_allowed":False,"product_promotion":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    required={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),
         "ONLY_FIRST_PROTECTED_PUBLIC_MAIN_SAVED_GATE")
    head=env.get("GITHUB_SHA","")
    need(bool(re.fullmatch("[0-9a-f]{40}",head)) and
         bool(re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID",""))),
         "RUN_ID_AND_SHA_REQUIRED")
    if event is None:
        event=readj(env["GITHUB_EVENT_PATH"])
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "FROZEN_32_NEW_LOCKED_FIRST_MAIN_SUCCESSOR")
    if parent is None:
        parent=subprocess.check_output(
            ["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"REJECT_OLD_LOCKED_TEST_REPLAY")
    return {"run_id":env["GITHUB_RUN_ID"],"head":head,
            "original_observer_run":OBSERVER_RUN,"new_model_forwards":0}

def attest(token):
    need(type(token) is str and len(token)>=10,"READONLY_GITHUB_ACTIONS_TOKEN")
    def get(path):
        req=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+path,
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"v5-newlocked8-original-saved-machine-gate"})
        with urllib.request.urlopen(req,timeout=40) as f:
            payload=f.read(1000001)
        need(len(payload)<=1000000,"SMALL_SOURCE_METADATA_ONLY")
        return json.loads(payload)
    for name,runid,head in [
      ("prior",PRIOR_RUN,PRIOR_HEAD),
      ("observer",OBSERVER_RUN,OBSERVER_HEAD),
      ("source",SOURCE_RUN,SOURCE_HEAD)]:
        run=get("/actions/runs/"+str(runid))
        need(run.get("id")==runid and
             run.get("head_sha")==head and
             run.get("head_branch")=="main" and
             run.get("event")=="push" and
             run.get("run_attempt")==1 and
             run.get("status")=="completed" and
             run.get("conclusion")=="success",
             "EXACT_PRIOR_SOURCE_AND_NATIVE_SUCCESS_RUN_"+name)
    items={"source":SOURCE,"prior":PRIOR}
    for provider,(ident,size,digest) in ARCHIVES.items():
        items[provider]={"id":ident,
          "name":f"v5-mero8locked-native-{provider}-{OBSERVER_RUN}",
          "bytes":size,"digest":"sha256:"+digest}
    for name,known in items.items():
        z=get("/actions/artifacts/"+str(known["id"]))
        expected_run=(SOURCE_RUN if name=="source" else
                      PRIOR_RUN if name=="prior" else OBSERVER_RUN)
        expected_head=(SOURCE_HEAD if name=="source" else
                       PRIOR_HEAD if name=="prior" else OBSERVER_HEAD)
        need(z.get("id")==known["id"]
             and z.get("name")==known["name"]
             and z.get("size_in_bytes")==known["bytes"]
             and z.get("digest")==known["digest"]
             and z.get("expired") is False
             and z.get("workflow_run",{}).get("id")==expected_run
             and z.get("workflow_run",{}).get("head_sha")==expected_head,
             "NO_WRONG_OR_EXPIRED_ORIGINAL_ARCHIVE_"+name)
    return items

def verify_prior(path):
    doc=readj(Path(path)/"RESULT.json")
    need(doc.get("schema")=="english-v5-mero-96-frozen-machine-6-2-2-2-source-audit-v1"
         and doc.get("source_artifact",{}).get("id")==14507208152
         and doc.get("status")=="STOP_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_COVERAGE"
         and doc.get("full_machine_group_counts")==PRIOR_COUNTS
         and doc.get("minimum_groups")==MIN_COUNTS
         and doc.get("passing_6_2_2_2_machine_gate") is False
         and doc.get("production_promotion_allowed") is False
         and doc.get("training_snapshot_allowed") is False
         and doc.get("new_model_weights") is False
         and doc.get("new_optimizer_steps")==0
         and doc.get("new_tts_asr_phone_model_calls")==0
         and doc.get("calibration_locked_kws_audio_evaluation")=="NOT_RUN"
         and len(doc.get("sealed_rows",[]))==32,
         "EXACT_PRIOR_11_2_2_1_REJECTED_MACHINE_LOCKED_GATE")
    old_speaker={x["speaker_index"] for x in doc["sealed_rows"]}
    old_groups={x["source_group_id"] for x in doc["sealed_rows"]}
    old_pcm={x["pcm_sha256"] for x in doc["sealed_rows"]}
    need(old_speaker==set(range(464,472)) and len(old_groups)==8
         and len(old_pcm)==32 and
         all(x["role"] in ("calibration","locked-test")
             for x in doc["sealed_rows"]),
         "NO_ORIGINAL_CAL_LOCK_ROLE_LEAK")
    return doc,old_groups,old_pcm

def count_new_locked(rows):
    clustered=defaultdict(list)
    for row in rows:
        need(row["role"]=="locked-test" and
             row["speaker_index"] in range(472,480) and
             row["training_allowed"] is False and
             row["human_gold"] is False and
             row["requested_class"] in (0,2,None),
             "ONLY_FROZEN_NEW_LOCKED8_32_SOURCE_QUALITY")
        clustered[row["source_group_id"]].append(row)
    eligible=[]
    for source,items in clustered.items():
        need(len(items)==4 and len({r["speaker_index"] for r in items})==1
             and {r["cell_id"] for r in items}=={
               "mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge"},
             "EXACT_FOUR_ORIGINAL_CELLS_PER_NEW_LOCKED_SOURCE")
        approved={r["observed_class"] for r in items
                  if r["tier"] in ("silver","weak")}
        for r in items:
            if r["cell_id"]=="nearphone-meero-challenge":
                need(r["tier"]=="unknown" and r["observed_class"] is None,
                     "NEARPHONE_CAN_NEVER_BE_ORDINARY")
            if r["tier"] in ("silver","weak"):
                need(r["observed_class"]==r["requested_class"]
                     and r["observed_class"] in (0,2),
                     "NO_PROMPT_LABEL_OR_UNKNOWN_ADMISSION")
        if {0,2}<=approved:
            eligible.append(source)
    need(len(clustered)==8 and len(rows)==32,
         "EXACT_EIGHT_NEW_LOCKED_SOURCE_GROUPS")
    return eligible

def evaluate(source,prior,observers,output):
    source_check()
    current=guard()
    output=Path(output)
    need(output.is_absolute() and not output.exists(),
         "NEW_ORIGINAL_SOURCE_RESEARCH_RECEIPT_ONLY")
    output.mkdir()
    c.record(output/"STARTED.json",{"schema":SCHEMA,"run":current,
       "locked_test_kws_opened":False,"new_tts_calls":0,
       "new_asr_phone_forwards":0,"new_optimizer_steps":0})
    try:
        packages=attest(os.environ.get("GH_TOKEN"))
        old_result,old_groups,old_pcm=verify_prior(prior)
        rows=review.source_rows(source)
        need(len(rows)==32 and
             {x["speaker_index"] for x in rows}==set(range(472,480))
             and not ({x["source_group_id"] for x in rows}&old_groups)
             and not ({x["pcm_sha256"] for x in rows}&old_pcm),
             "GENUINELY_NEW_LOCKED_FAMILY_AND_PCM_NOT_OLD_TEST")
        values={p:original.packet(Path(observers)/p,p,rows)
                for p in review.PROVIDERS}
        verdicts=original.machine_decisions(rows,values)
        new_pass=count_new_locked(verdicts)
        total={**PRIOR_COUNTS}
        total["locked-test"]+=len(new_pass)
        full=all(total[key]>=minimum for key,minimum in MIN_COUNTS.items())
        result={"schema":SCHEMA,
           "status":("PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"
                    if full else "STOP_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"),
           "prior_source_artifact":packages["prior"],
           "new_source_artifact":packages["source"],
           "original_five_native_artifacts":{
               p:packages[p] for p in review.PROVIDERS},
           "prior_machine_group_counts":PRIOR_COUNTS,
           "new_locked_source_groups_machine_qualified":len(new_pass),
           "new_locked_qualified_group_ids":sorted(new_pass),
           "cumulative_machine_group_counts":total,
           "minimum_group_counts":MIN_COUNTS,
           "full_6_2_2_2_synthetic_machine_gate":full,
           "new_original_speaker_groups":8,"new_original_pcm_count":32,
           "new_source_label_tier_counts":dict(Counter(r["tier"] for r in verdicts)),
           "new_source_label_reason_counts":dict(Counter(r["reason"] for r in verdicts)),
           "new_source_observed_class_counts":dict(Counter(
               str(r["observed_class"]) for r in verdicts
               if r["tier"] in ("silver","weak"))),
           "new_tts_asr_phone_model_calls":0,
           "kws_locked_evaluation":"NOT_RUN",
           "no_train_dev_calibration_relabel":True,
           "no_old_locked_relabel":True,
           "new_optimizer_updates":0,"new_model_weights":False,
           "training_snapshot_allowed":False,
           "independent_human_gold":False,"ssc305":"NOT_RUN",
           "real_far_frr":"NOT_MEASURED",
           "product_promotion_allowed":False,
           "machine_source_rows":verdicts}
        c.record(output/"RESULT.json",result)
        print(json.dumps({k:result[k] for k in (
           "status","prior_machine_group_counts",
           "new_locked_source_groups_machine_qualified",
           "cumulative_machine_group_counts",
           "new_source_label_tier_counts","new_source_observed_class_counts",
           "full_6_2_2_2_synthetic_machine_gate",
           "product_promotion_allowed")},sort_keys=True))
    except BaseException as e:
        c.record(output/"STOP.json",{
            "schema":SCHEMA,"status":"STOP_ORIGINAL_5_NATIVE_SAVED_PROVENANCE",
            "error_type":type(e).__name__,"reason":str(e)[:500],
            "new_model_calls":0,"new_optimizer_updates":0,
            "locked_kws_test_not_opened":True})
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--guard",action="store_true")
    p.add_argument("--evaluate",action="store_true")
    p.add_argument("--source",type=Path)
    p.add_argument("--prior",type=Path)
    p.add_argument("--observers",type=Path)
    p.add_argument("--output",type=Path)
    x=p.parse_args()
    need(sum((x.source_check,x.guard,x.evaluate))==1,
         "EXACT_ONE_ORIGINAL_MODEL_FREE_MACHINE_SCREEN_STAGE")
    if x.source_check:
        print(json.dumps(source_check(),sort_keys=True))
    elif x.guard:
        print(json.dumps(guard(),sort_keys=True))
    else:
        need(x.source and x.prior and x.observers and x.output,
             "EXACT_ARCHIVED_SOURCE_PRIOR_AND_FIVE_NATIVE_REQUIRED")
        evaluate(x.source,x.prior,x.observers,x.output)

if __name__=="__main__":
    main()
