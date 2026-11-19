"""Final V5 Hey Mero original-PCM source group audit: 6/2/2/2 or STOP.

Only source-quality inference is reused; this code calls no ASR/phone/KWS
model and never opens any locked PCM for KWS. The immutable train/dev 64-row
five-observer screen plus five separate sealed-role five-observer ZIPs are
authenticated. Nearphone is always UNKNOWN; Silver/Weak only from the
independent frozen native phonetic/ASR policy, never the requested IPA.
All groups retain pre-TTS roles. A PASS means only synthetic MACHINE source
coverage; it is NOT a product release, human FAR/FRR or test result.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
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
        "en_v5_full96_sealed_"+name,HERE/(name+".py"))
    v=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    return v

sealed=load("english_v5_mero_fresh24_sealed_source_five_native_once")
old=load("english_v5_mero_fresh24_saved_screen_once")
policy=load("english_fresh24_ctc_admission")
phones=load("english_hifi_phones_once")
c=load("english_hifi_micro")

SCHEMA="english-v5-mero-96-frozen-machine-6-2-2-2-source-audit-v1"
REPO="jiying2007/kws-data"
BEFORE="c0a9b44ee4aaa7fb522c1e04179d787850b9e0d3"
SOURCE_RUN=44478554596
SEALED_RUN=44485062398
SEALED_COMMIT=BEFORE
SEALED_STATUS="success"
PRIOR_SCREEN_RUN=44483128198
SOURCE={"id":14507208152,"name":"english-v5-mero-fresh24-original-44478554596",
        "bytes":1961038,
        "digest":"sha256:2fbc26386136cfd12dddb01074f3c9c4556c8fc7c49819663f2d11a2499eaa1c"}
PRIOR={"id":14509252921,
       "name":"english-v5-mero-fresh24-saved-machine-screen-44483128198",
       "bytes":9282,
       "digest":"sha256:e8bcdf32a0ac138de8bdd90c4f6cd6e419d4ccae42bbd393e75b5daac9502158"}
ARCHIVES={
 "qwen":(14509818422,31608,"647116c73ce1c2e054a9696d5b263c6ad2393a6245fe423aff5b56ac065756ae"),
 "whisper":(14509594019,31702,"0b13fd3a501684e097e26334d4bd83267e738534a660bc05667d7799659946b8"),
 "parakeet":(14509653724,31673,"a26428bcca138953d4217164a09a4dac547cb186deee25e2b0f8fff96b65f93e"),
 "xlsr":(14509658765,1717896,"44a3eab0034d48cf74477cffea496bc60e8024d11d8272a5004bad70bacc8331"),
 "lv60":(14510871195,1718115,"8e4ac97fecc4fb9d13799d027683a9872ebec7789fada9ac97bcc3f194388cf0"),
}
MIN={"train":6,"development":2,"calibration":2,"locked-test":2}
REVIEW_N=32

def need(ok,why):
    if not ok:
        raise ValueError(why)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def readj(path):
    return json.loads(c.read(Path(path)))

def source_check():
    check=sealed.source_check()
    prior=old.source_check()
    need(check["status"]=="ORIGINAL_SOURCE_PINNED"
         and check["source_rows"]==96 and check["review_rows"]==32
         and check["train_or_dev_review_clips"]==0
         and check["kws_model_forwards"]==0
         and check["source_artifact_id"]==SOURCE["id"]
         and check["new_tts_calls"]==0
         and old.ORIGINAL_RUN==44481695979
         and prior["source_run"]==SOURCE_RUN
         and prior["review_rows"]==64
         and prior["training_allowed"] is False
         and tuple(sealed.PROVIDERS)==tuple(old.observer.PROVIDERS)
         and tuple(sealed.PROVIDERS)==tuple(policy.TEXT_MODELS)+tuple(policy.PHONE_MODELS)
         and len(ARCHIVES)==5 and len({x[0] for x in ARCHIVES.values()})==5
         and policy.MIN_QUALIFIED==MIN and SEALED_RUN==44485062398
         and SEALED_COMMIT=="c0a9b44ee4aaa7fb522c1e04179d787850b9e0d3"
         and PRIOR_SCREEN_RUN==44483128198,
         "EXACT_ORIGINAL_V5_SIX_ARCHIVES_AND_SEALED_8_ROLE_SPLIT")
    return {"status":"PINNED_SIX_SOURCE_SCREEN_ARCHIVES",
            "original_source_run":SOURCE_RUN,
            "old_64_machine_screen_artifact":PRIOR["id"],
            "new_sealed_source_run":SEALED_RUN,
            "sealed_source_count":32,"new_native_forward_calls":0,
            "kws_forward_calls":0,"new_optimizer_steps":0,
            "locked_test_kws_opened":False,
            "source_minimum_groups":MIN,
            "quality_release_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    required={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),
         "FIRST_PUBLIC_MAIN_ONLY_MACHINE_SCREEN")
    head=env.get("GITHUB_SHA","")
    need(bool(re.fullmatch("[0-9a-f]{40}",head)) and
         bool(re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID",""))),
         "VALID_PUBLIC_MAIN_ID")
    if event is None:
        event=readj(env["GITHUB_EVENT_PATH"])
    need(event.get("before")==BEFORE and
         event.get("after")==head and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "EXACT_FROZEN_POST_SEALED_REVIEW_SOURCE_SCREEN")
    if parent is None:
        parent=subprocess.check_output(
            ["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"NO_REPLAY_OF_MACHINE_SCREEN")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":head,
            "new_model_calls":0,"no_locked_kws":True}

def remote_attestation(token):
    need(type(token) is str and len(token)>=10,"READONLY_ACTIONS_TOKEN")
    def get(path):
        req=urllib.request.Request(
          "https://api.github.com/repos/"+REPO+path,
          headers={"Authorization":"Bearer "+token,
                   "Accept":"application/vnd.github+json",
                   "User-Agent":"v5-mono-machine-source-96-gate"})
        with urllib.request.urlopen(req,timeout=40) as stream:
            raw=stream.read(1000001)
        need(len(raw)<=1000000,"SMALL_GITHUB_METADATA_ONLY")
        return json.loads(raw)
    run=get("/actions/runs/"+str(SEALED_RUN))
    need(run.get("id")==SEALED_RUN and run.get("head_sha")==SEALED_COMMIT
         and run.get("head_branch")=="main" and run.get("event")=="push"
         and run.get("run_attempt")==1 and run.get("status")=="completed"
         and run.get("conclusion")==SEALED_STATUS,
         "FIVE_ORIGINAL_SEALED_MODEL_RUN_AUTHENTIC")
    artifacts={"source":SOURCE,"prior_screen":PRIOR}
    for provider,(ident,size,digest) in ARCHIVES.items():
        artifacts[provider]={"id":ident,
            "name":f"v5-mero24-sealed-label-{provider}-{SEALED_RUN}",
            "bytes":size,"digest":"sha256:"+digest}
    for key,expected in artifacts.items():
        record=get("/actions/artifacts/"+str(expected["id"]))
        runid=SOURCE_RUN if key=="source" else (
              PRIOR_SCREEN_RUN if key=="prior_screen" else SEALED_RUN)
        head=sealed.SOURCE_COMMIT if key=="source" else (
             old.BEFORE if key=="prior_screen" else SEALED_COMMIT)
        # old.BEFORE equals original 64-reviewer run HEAD, not prior
        # screen head; the saved screen was completed at a different main.
        if key=="prior_screen":
            head="742dea5aab2224bdfe2ad1076237820f60ce85d4"
        need(record.get("id")==expected["id"]
             and record.get("name")==expected["name"]
             and record.get("size_in_bytes")==expected["bytes"]
             and record.get("digest")==expected["digest"]
             and record.get("expired") is False
             and record.get("workflow_run",{}).get("id")==runid
             and record.get("workflow_run",{}).get("head_sha")==head,
             "HASHED_PUB_GITHUB_SIX_SOURCE_ARTIFACTS_"+key)
    return artifacts

def prior_snapshot(prior_path,full_rows):
    data=readj(Path(prior_path)/"SCREEN.json")
    need(data.get("schema")==old.SCHEMA
         and data.get("status")=="PASS_NEW_TRAIN_DEV_MACHINE_SCREEN_ONLY"
         and data.get("source_run")==SOURCE_RUN
         and data.get("original_observer_run")==old.ORIGINAL_RUN
         and data.get("source_group_counts")=={"train":11,"development":2}
         and data.get("observed_label_counts")=={"0":16,"2":16}
         and data.get("tier_counts")=={"silver":25,"unknown":32,"weak":7}
         and data.get("full_6_2_2_2_gate_passed") is False
         and data.get("training_snapshot_allowed") is False
         and data.get("locked_kws_test_opened") is False
         and len(data.get("rows",[]))==64,
         "EXACT_FROZEN_ORIGINAL_TRAIN_DEV_MACHINE_SCREEN")
    prior_roles={r["id"]:r for r in full_rows
                 if r["role"] in ("train","development")}
    need(len(prior_roles)==64,"SOURCE96_FROZEN_TRAIN_DEV_SPLIT")
    seen=set();groups=defaultdict(set);d_roles={}
    for row in data["rows"]:
        sid=row.get("id")
        need(sid in prior_roles and sid not in seen,"UNEXPECTED_EXPOSED_PRIOR_SOURCE")
        seen.add(sid)
        expected=prior_roles[sid]
        need(row.get("role")==expected["role"]
             and row.get("pcm_sha256")==expected["pcm_sha256"]
             and row.get("wav_sha256")==expected["wav_sha256"]
             and row.get("source_group_id")==expected["source_group_id"]
             and row.get("speaker_index")==expected["speaker_index"]
             and row.get("training_allowed") is False
             and row.get("human_gold") is False
             and row.get("requested_class")==expected["requested_class"],
             "PRIOR_SILVER_WEAK_UNKNOWN_UNCHANGED_PCM_ROLE")
        group=row["source_group_id"]
        need(group not in d_roles or d_roles[group]==row["role"],
             "PRIOR_GROUP_CROSS_ROLE")
        d_roles[group]=row["role"]
        if row["tier"] in ("silver","weak"):
            need(row["observed_class"]==row["requested_class"]
                 and row["observed_class"] in (0,2),
                 "PRIOR_ADMITTED_ONLY_CANONICAL_MERO_ORDINARY")
            groups[group].add(row["observed_class"])
        else:
            need(row["tier"]=="unknown" and row["observed_class"] is None,
                 "NO_UNKNOWN_RELABEL")
    need(seen==set(prior_roles) and len(d_roles)==16,
         "ALL_64_ORIGINAL_TRAIN_DEV_ROWS_PROVEN")
    qualified=Counter(d_roles[g] for g,classes in groups.items()
                      if {0,2}<=classes)
    need(qualified=={"train":11,"development":2},
         "EXACT_PREVIOUS_11_TRAIN_2_DEV_QUALIFIED_GROUPS")
    return {"train":11,"development":2},data

def machine_decisions(rows,values):
    need(len(rows)==32 and set(values)==set(sealed.PROVIDERS),
         "ONLY_32_SEALED_ACOUSTIC_REVIEW_SAMPLES")
    selected=[]
    for row in rows:
        sid=row["id"]
        transcript={k:values[k][sid] for k in policy.TEXT_MODELS}
        posterior={k:values[k][sid] for k in policy.PHONE_MODELS}
        v=policy.decide({**row,"pcm_samples":row["samples"]},
                        transcript,posterior)
        if v["tier"] in ("silver","weak") and (
              v["class"]!=row["requested_class"]):
            v={"tier":"unknown","class":None,
               "reason":"REQUEST_ACOUSTIC_CONTRADICTION",
               "margins":{},"training_allowed":False}
        if row["requested_class"] is None:
            need(v["tier"]=="unknown" and v["class"] is None,
                 "NEARPHONE_CHALLENGE_NEVER_AN_ORDINARY_NEGATIVE")
        selected.append({"id":sid,"source_group_id":row["source_group_id"],
             "role":row["role"],"speaker_index":row["speaker_index"],
             "cell_id":row["cell_id"],
             "requested_class":row["requested_class"],
             "observed_class":v["class"],"tier":v["tier"],
             "reason":v["reason"],"phone_margins":v["margins"],
             "pcm_sha256":row["pcm_sha256"],
             "wav_sha256":row["wav_sha256"],
             "human_gold":False,"training_allowed":False})
    return selected

def qualified_by_role(decisions):
    clusters=defaultdict(list)
    for row in decisions:
        clusters[row["source_group_id"]].append(row)
    roles={};groups=Counter();qualified=[]
    for gid,rows in clusters.items():
        role_set={r["role"] for r in rows}
        need(len(rows)==4 and len(role_set)==1
             and role_set<=set(("calibration","locked-test")),
             "EXACT_4_4_SEALED_GROUP_ROLE")
        role=next(iter(role_set))
        roles[gid]=role
        accepted={r["observed_class"] for r in rows
                  if r["tier"] in ("silver","weak")}
        if {0,2}<=accepted:
            groups[role]+=1
            qualified.append(gid)
    need(len(clusters)==8 and
         Counter(roles.values())=={"calibration":4,"locked-test":4},
         "NO_HISTORICAL_SOURCE_ROLE_MIGRATION")
    return {"calibration":groups["calibration"],
            "locked-test":groups["locked-test"]},qualified

def packet(root,provider,rows):
    root=Path(root)
    expected={r["id"]:r for r in rows}
    need(len(expected)==32,"ORIGINAL_UNIQUE_32_SOURCE_SAMPLES")
    doc=readj(root/"RESULT.json")
    need(doc.get("schema")==sealed.SCHEMA
         and doc.get("provider")==provider
         and doc.get("run")=={
             "run_id":str(SEALED_RUN),"commit":SEALED_COMMIT,
             "source_run":SOURCE_RUN,"source_commit":sealed.SOURCE_COMMIT}
         and doc.get("source_artifact_id")==SOURCE["id"]
         and doc.get("denominator")==32
         and doc.get("complete")==32
         and doc.get("attempted")==32
         and doc.get("missing_or_failed")==0
         and doc.get("status")=="COMPLETE"
         and doc.get("initialization_error") is None
         and doc.get("new_tts_calls")==0
         and doc.get("target_conditioned") is False
         and doc.get("human_gold") is False
         and doc.get("training_allowed") is False
         and len(doc.get("rows",[]))==32,
         "AUTHENTIC_UNREPEATED_COMPLETE_32_NATIVE_QUERIES")
    lock=readj(root/"MODEL.json")
    if provider in ("xlsr","lv60"):
        model,revision=sealed.identity(provider)
        i=0 if provider=="xlsr" else 1
        need(lock.get("model")==model and
             lock.get("revision","").startswith(revision)
             and lock.get("weights_sha256")==phones.WEIGHT_SHA[i]
             and lock.get("target_conditioned") is False,
             "ORIGINAL_PHONE_MODEL_WEIGHT_SHA")
        vocab=readj(root/"VOCAB.json")
        v=vocab.get("vocab")
        need(vocab.get("model")==model
             and vocab.get("revision")==lock["revision"]
             and vocab.get("pad_token_id")==0
             and type(v) is dict and len(v)==392
             and len(set(v.values()))==392,
             "392_PHONE_CTC_RECOGNIZER_PIN")
    else:
        need(lock.get("observer")==sealed.identity(provider)
             and lock.get("target_conditioned") is False
             and lock.get("initialization_model_calls")==0,
             "BLIND_NATIVE_TEXT_NO_KEYWORD_PROMPT")
    observed={}
    for row in doc["rows"]:
        sid=row.get("id")
        need(sid in expected and sid not in observed
             and row.get("status")=="success"
             and row.get("provider")==provider
             and row.get("source_wav_sha256")==expected[sid]["wav_sha256"]
             and row.get("source_pcm_sha256")==expected[sid]["pcm_sha256"]
             and row.get("source_samples")==expected[sid]["samples"]
             and row.get("target_conditioned") is False
             and row.get("input_coverage_complete") is True
             and row.get("training_allowed") is False
             and readj(root/"rows"/sid/"CAPTURE.json")==row,
             "ORIGINAL_32_WAV_FULL_PCM_RECEIPTS")
        if provider in ("xlsr","lv60"):
            raw=c.read(root/"logits"/(sid+".f32"),40*1024*1024)
            frames=row.get("logit_frames")
            need(type(frames) is int and 1<=frames<=2000
                 and row.get("vocab_size")==392
                 and len(raw)==frames*392*4
                 and sha(raw)==row.get("logits_sha256"),
                 "WHOLE_ORIGINAL_PHONE_CTC_LOGITS")
            argmax=[]
            for frame in struct.iter_unpack("<392f",raw):
                need(all(math.isfinite(x) for x in frame),
                     "NATIVE_PHONE_LOGIT_FINITE")
                argmax.append(max(range(392),key=frame.__getitem__))
            need(phones.ctc(argmax,v,0)==row.get("raw_phone_tokens"),
                 "ORIGINAL_PHONE_ARGMAX_NUMERIC_READBACK")
            observed[sid]={"logits":raw,"vocab":v,
                           "sha256":row["logits_sha256"],
                           "pad_token_id":0}
        else:
            txt=row.get("raw_transcript")
            need(type(txt) is str and 0<len(txt)<=4096
                 and not any(ord(ch)<32 for ch in txt),
                 "COMPLETE_BLIND_NATIVE_RAW_TEXT")
            observed[sid]=txt
    need(set(observed)==set(expected),"EVERY_SEALED_ORIGINAL_PCM_OBSERVED")
    return observed

def combine(source_dir,prior_screen,observers,out):
    source_check()
    receipt=guard()
    out=Path(out)
    need(out.is_absolute() and not out.exists(),
         "NEW_IMMUTABLE_SOURCE_QUALITY_PACKET_ONLY")
    out.mkdir()
    c.record(out/"STARTED.json",{
        "schema":SCHEMA,"run":receipt,
        "source_run":SOURCE_RUN,"sealed_observer_run":SEALED_RUN,
        "new_kws_model_forwards":0,"locked_kws_test_opened":False,
        "new_asr_phone_tts_calls":0})
    try:
        attested=remote_attestation(os.environ.get("GH_TOKEN"))
        all_rows=readj(Path(source_dir)/"MANIFEST.json")["rows"]
        sealed_rows=sealed.source_rows(source_dir)
        need(len(all_rows)==96 and len(sealed_rows)==32,
             "ORIGINAL_96_SOURCE_INTEGRITY_REQUIRED")
        prior_counts,prior_doc=prior_snapshot(prior_screen,all_rows)
        extra={}
        for provider in sealed.PROVIDERS:
            extra[provider]=packet(Path(observers)/provider,provider,sealed_rows)
        decisions=machine_decisions(sealed_rows,extra)
        new_counts,accepted=qualified_by_role(decisions)
        totals={**prior_counts,**new_counts}
        passes=all(totals[k]>=n for k,n in MIN.items())
        result={
           "schema":SCHEMA,
           "status":("PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_COVERAGE"
                     if passes else
                     "STOP_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_COVERAGE"),
           "source_artifact":attested["source"],
           "prior_screen_artifact":attested["prior_screen"],
           "sealed_native_artifacts":{
              p:attested[p] for p in sealed.PROVIDERS},
           "total_source_groups":24,"all_original_source_pcm":96,
           "old_eligible_train_dev_groups":prior_counts,
           "new_machine_eligible_sealed_groups":new_counts,
           "full_machine_group_counts":totals,
           "minimum_groups":MIN,
           "passing_6_2_2_2_machine_gate":passes,
           "passed_sealed_source_group_ids":accepted,
           "sealed_label_tier_counts":dict(Counter(r["tier"] for r in decisions)),
           "sealed_label_counts":dict(Counter(str(r["observed_class"])
                             for r in decisions if r["tier"] in ("silver","weak"))),
           "sealed_reason_counts":dict(Counter(r["reason"] for r in decisions)),
           "calibration_locked_kws_audio_evaluation":"NOT_RUN",
           "test_metrics":"NOT_RUN","real_3_5m":"NOT_MEASURED",
           "ssc305":"NOT_RUN","human_gold":False,
           "source_provenance":"original distinct synthetic Piper LibriTTS IDs, not independent human capture",
           "nearphone_as_negative":False,"new_model_weights":False,
           "new_optimizer_steps":0,"new_tts_asr_phone_model_calls":0,
           "training_snapshot_allowed":False,
           "production_promotion_allowed":False,
           "sealed_rows":decisions}
        c.record(out/"RESULT.json",result)
        print(json.dumps({k:result[k] for k in (
             "status","full_machine_group_counts",
             "sealed_label_tier_counts","sealed_label_counts",
             "sealed_reason_counts","passing_6_2_2_2_machine_gate",
             "production_promotion_allowed")},sort_keys=True))
    except BaseException as e:
        c.record(out/"STOP.json",{
             "schema":SCHEMA,"status":"STOP_IMMUTABLE_SAVED_SCREEN_PROOF",
             "exception":type(e).__name__,"reason":str(e)[:500],
             "new_model_calls":0,"new_optimizer_steps":0,
             "locked_kws_test_not_opened":True})
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--guard",action="store_true")
    p.add_argument("--combine",action="store_true")
    p.add_argument("--source",type=Path)
    p.add_argument("--prior",type=Path)
    p.add_argument("--observers",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.combine))==1,
         "EXACT_ONE_ORIGINAL_SAVED_READBACK_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True))
    elif a.guard:
        print(json.dumps(guard(),sort_keys=True))
    else:
        need(a.source and a.prior and a.observers and a.output,
             "ALL_ORIGINAL_SAVED_SOURCE_AND_SCREEN_INPUTS_REQUIRED")
        combine(a.source,a.prior,a.observers,a.output)

if __name__=="__main__":
    main()
