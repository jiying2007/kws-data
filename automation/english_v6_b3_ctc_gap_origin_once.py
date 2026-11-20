#!/usr/bin/env python3
"""Original B3 continuous C11 logit path -> unmodified CTC decoder gap replay.

One read-only diagnosis of cross-utterance HEY/MEH/ROH provenance on the *same*
already exposed six synthetic positives + eight ordinary. Strictly no
retraining, threshold calibration, new WAV/TTS/ASR, CAL/LOCKED, product claim.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location(
    "v6_b3_previous_gain",
    HERE/"english_v6_b3_lowgain_context_c11_once.py")
low=importlib.util.module_from_spec(spec)
spec.loader.exec_module(low)
prior=low.prior
b3=low.b3

SCHEMA="english-v6-b3-ctc-gap-and-token-source-original-c11-v1"
REPO="jiying2007/kws-data"
BEFORE="db0c4b699b16fcd765b430702a8f786715529eb5"
PREVIOUS_RUN=44662162126
PREVIOUS_ARTIFACT=14595464491
PREVIOUS_SHA="0ba7cd287afcedbbd7625b6987cc89dbeb2c8717c1506b905e8a01671e9ea3cb"
EXPOSED_CONTINUOUS_RUN=44658362723
GAPS=(8000,16000,32000,1000000)
CASES=("mixed","reverse")
BASELINE_CLOCKS={
    "mixed":[112352,168832,223552,280832,339712],
    "reverse":[83712,141792,199392,254272,311552,368512],
}
REPLAY_C=HERE/"en_v6_b3_ctc_gap_replay.c"

def need(test,why):
    if not test:raise ValueError(why)

def store(path,data):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def source_check():
    c=low.source_check()
    need(c["status"]=="SOURCE_ONLY_NO_KWS_AUDIO_FORWARD"
         and c["model_artifact"]==14590093798
         and c["input_qualified_positive"]==6
         and c["input_qualified_ordinary"]==8
         and c["excluded_unknown"]==18
         and low.BEFORE=="4995bfe0813f4bd6b65fa75a9c6764b14d64bd4e"
         and len(PREVIOUS_SHA)==64 and GAPS==(8000,16000,32000,1000000)
         and CASES==("mixed","reverse")
         and len(BASELINE_CLOCKS["mixed"])==5
         and len(BASELINE_CLOCKS["reverse"])==6
         and REPLAY_C.is_file(),
         "PREDECLARED_GAP_AND_B3_STREAM_SOURCE_ONLY")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_REAL_ACOUSTIC_FORWARDS",
            "model":"B3 43876","model_artifact":14590093798,
            "source_previous_gain_run":PREVIOUS_RUN,
            "original_source_count":14,"positive_originals":6,
            "ordinary_originals":8,"unknown_excluded":18,
            "replay_gaps_samples":list(GAPS),
            "cases":list(CASES),"truly_independent_dev":False,
            "source_bytes_changed":False,"new_training":False,
            "new_tts_asr":False,"cal_locked_read":False,
            "model_selection_allowed":False,"physical_ssc305":"NOT_RUN",
            "product_promotion":False}

def remote_proof(token):
    need(isinstance(token,str) and len(token)>10,
         "GITHUB_READONLY_TOKEN_REQUIRED")
    proven=low.verify_prior(token)
    def api(suffix):
        r=urllib.request.Request(
            "https://api.github.com/repos/"+REPO+"/actions/"+suffix,
            headers={"Authorization":"Bearer "+token,
                     "Accept":"application/vnd.github+json",
                     "User-Agent":"en-v6-b3-ctc-gap-provenance"})
        with urllib.request.urlopen(r,timeout=35) as resp:
            b=resp.read(1048577)
        need(len(b)<=1048576,"BOUNDED_GITHUB_METADATA")
        return json.loads(b)
    run=api("runs/"+str(PREVIOUS_RUN))
    need(run.get("id")==PREVIOUS_RUN and run.get("head_sha")==BEFORE
         and run.get("run_attempt")==1 and run.get("event")=="push"
         and run.get("status")=="completed" and run.get("conclusion")=="success",
         "EXACT_REAL_B3_GAIN_DIAGNOSTIC_COMPLETED")
    artifact=api("artifacts/"+str(PREVIOUS_ARTIFACT))
    need(artifact.get("id")==PREVIOUS_ARTIFACT
         and artifact.get("name")=="english-v6-b3-gain-context-c11-44662162126"
         and artifact.get("digest")=="sha256:"+PREVIOUS_SHA
         and artifact.get("size_in_bytes")==8238
         and artifact.get("expired") is False
         and artifact.get("workflow_run",{}).get("id")==PREVIOUS_RUN,
         "EXACT_GAIN_EVIDENCE_SHA_REQUIRED")
    return {"previous_gain":PREVIOUS_RUN,"artifact":PREVIOUS_ARTIFACT,
            "sha256":PREVIOUS_SHA,"previous_sources":proven}

def guard():
    source_check()
    env=os.environ
    need(all(env.get(k)==v for k,v in {
      "GITHUB_ACTIONS":"true",
      "GITHUB_REPOSITORY":REPO,
      "GITHUB_EVENT_NAME":"push",
      "GITHUB_REF":"refs/heads/main",
      "GITHUB_RUN_ATTEMPT":"1"}.items()),"FIRST_MAIN_NO_REPLAY_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[a-f0-9]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "RUN_ID_AND_HEAD_VALID")
    e=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(e.get("before")==BEFORE and e.get("after")==head
         and e.get("ref")=="refs/heads/main"
         and e.get("repository",{}).get("full_name")==REPO
         and e.get("repository",{}).get("private") is False
         and e.get("forced") is False and e.get("deleted") is False
         and subprocess.check_output(["git","rev-parse","HEAD^1"],
                                     cwd=ROOT,text=True).strip()==BEFORE,
         "EXACT_PROTECTED_MAIN_PARENT_ONLY")
    return {"commit":head,"run":int(env["GITHUB_RUN_ID"]),
            "previous":remote_proof(env.get("GH_TOKEN"))}

def locate_sample(segments,clock):
    matches=[r for r in segments if r["start"]<=clock<r["end"]+prior.GRACE_SAMPLES]
    need(len(matches)<=1,"AMBIGUOUS_SEGMENT_SAMPLE_ORIGIN")
    return matches[0]["id"] if matches else None

def parse_events(csv,segments):
    events=[]
    lines=csv.read_text("ascii").splitlines()
    for line in lines:
        fields=line.split(",")
        need(len(fields)==3 and all(re.fullmatch("[0-9]+",v) for v in fields),
             "EXACT_THREE_CTC_ANCHOR_TIMESTAMPS")
        event,hey,meh=map(int,fields)
        need(0<hey<meh<event,"CHRONOLOGICAL_CTC_TOKEN_ANCHORS")
        hs,ms,es=(locate_sample(segments,x) for x in (hey,meh,event))
        same=hs is not None and hs==ms==es
        events.append({"event_sample":event,"hey_sample":hey,
           "meh_sample":meh,"hey_source":hs,
           "meh_source":ms,"event_source":es,
           "tokens_from_same_original":same,
           "cross_or_unattributed_token_source":not same,
           "hey_to_meh_samples":meh-hey,"meh_to_roh_samples":event-meh})
    return events

def compile_replay(native,out):
    src=native/"mero_ctc.c"
    need(REPLAY_C.is_file() and src.is_file(),"C11_REPLAY_SOURCE_REQUIRED")
    subprocess.run(["gcc","-std=c11","-Wall","-Wextra","-Werror",
                    "-pedantic","-O2","-ffp-contract=off",
                    "-I",str(native),str(REPLAY_C),str(src),"-lm",
                    "-o",str(out)],
                   capture_output=True,text=True,check=True,timeout=100)
    need(out.is_file() and out.stat().st_size>10000,
         "ACTUAL_GCC_C11_DECODER_POLICY_BINARY")
    return out

def execute(source,screen,b3zip,pipeline,output):
    import numpy as np
    import torch
    receipt=guard()
    need(output.is_absolute() and output.parent.is_dir()
         and not output.exists(),"NEW_SOURCE_BOUNDED_OUTPUT")
    output.mkdir()
    store(output/"STARTED.json",{"schema":SCHEMA,"receipt":receipt,
        "training":False,"model_selection":False,
        "old_dev_reused":True,"cal_locked_opened":False})
    try:
        rows,source_receipt=b3.audit(source,screen,b3zip)
        original=prior.select_rows(rows)
        native=b3.pipeline_files(pipeline)
        packer=b3.imported(native/"pack_mero43.py","v6_gap_frozen_geometry")
        _,_,_,_,packed=b3.load_b3(torch,np,b3zip,packer)
        need(len(packed)==216960,"ORIGINAL_B3_PACK_ABI")
        pack=output/"model.tmp.pack"
        pack.write_bytes(packed)
        host=b3.compile_c(native,"gcc",output/"b3-host-gcc")
        replay=compile_replay(native,output/"b3-gap-c11")
        cases=[]
        for case in CASES:
            raw,segments=prior.assemble(rows,lambda r:b3.v6.pcm(r,source),case)
            need(len(segments)==14
                 and sum(x["class"]==1 for x in segments)==6,
                 "REAL_IDENTICAL_14_CONTINUOUS_ORIGINAL_SEGMENTS")
            scratch=output/("scratch-"+case)
            scratch.mkdir()
            pcm=scratch/"input.pcm"
            pcm.write_bytes(raw)
            logits=scratch/"b3.f32"
            csv=scratch/"baseline.events"
            done=subprocess.run([str(host),str(pack),str(pcm),
                  str(logits),str(csv)],capture_output=True,text=True,
                  timeout=130,check=True)
            need("MERO43_C_PCM_STREAM_OK" in done.stderr,
                 "REAL_B3_ORIGINAL_STREAM_AUDIO_MODEL_FORWARD")
            baseline=b3.c_events(csv,logits.stat().st_size//16)
            need(baseline==BASELINE_CLOCKS[case],
                 "IDENTICAL_REPLAY_OF_PREVIOUS_EXPOSED_CONTINUOUS_C11")
            policy=[]
            for gap in GAPS:
                out=scratch/("events-"+str(gap)+".csv")
                p=subprocess.run([str(replay),str(logits),str(gap),str(out)],
                                 capture_output=True,text=True,timeout=55,check=True)
                need("B3_GAP_CTC_REPLAY_OK" in p.stderr,
                     "ACTUAL_UNMODIFIED_C11_CTC_DECODER_POLICY")
                events=parse_events(out,segments)
                if gap==1000000:
                    need([x["event_sample"] for x in events]==baseline,
                         "EXACT_MATCH_TO_UNMODIFIED_LONG_GAP_HOST")
                policy.append({"gap_samples":gap,
                    "total_events":len(events),
                    "same_original_event_count":sum(
                       x["tokens_from_same_original"] for x in events),
                    "cross_or_unattributed_source_count":sum(
                       x["cross_or_unattributed_token_source"] for x in events),
                    "events":events})
            cases.append({"case":case,"reused_originals":14,
               "source_count":8,"positive_originals":6,
               "ordinary_originals":8,
               "input_pcm_sha256":b3.sha(raw),
               "frame_logits_sha256":b3.sha(logits.read_bytes()),
               "baseline_long_gap_event_clocks":baseline,
               "decoder_policy":policy})
            shutil.rmtree(scratch)
        result={"schema":SCHEMA,
          "status":"PASS_ORIGINAL_B3_GAP_TOKEN_SOURCE_MECHANISM_NO_PROMOTION",
          "receipt":receipt,"source":source_receipt,
          "B3_artifact_id":b3.B3["id"],
          "B3_pt_sha256":b3.B3["pt"],
          "unchanged_acoustic_model":True,
          "unchanged_decoder_source":True,
          "cases":cases,"new_optimizer_steps":0,
          "new_tts_asr_calls":0,
          "reused_exposed_synthetic_dev":True,
          "fresh_independent_quality":"NOT_RUN",
          "no_true_word_end_alignment":True,
          "natural_far_per_hour":"NOT_MEASURED",
          "real_human_3_5m_frr":"NOT_MEASURED",
          "physical_ssc305":"NOT_RUN",
          "cal_locked_opened":False,
          "model_selection_allowed":False,
          "decoder_policy_selection_allowed":False,
          "product_promotion_allowed":False}
        store(output/"RESULT.json",result)
        print(json.dumps({"status":result["status"],
             "cases":[{"case":row["case"],"gap":p["gap_samples"],
                 "events":p["total_events"],
                 "cross_source":p["cross_or_unattributed_source_count"]}
                 for row in cases for p in row["decoder_policy"]],
             "product_promotion_allowed":False},sort_keys=True))
    except BaseException as ex:
        store(output/"STOP.json",{"schema":SCHEMA,
           "status":"STOP_CTC_GAP_DIAGNOSTIC_NO_RETRY",
           "error":str(ex)[:400],"error_type":type(ex).__name__,
           "product_promotion_allowed":False})
        raise
    finally:
        for x in output.glob("scratch-*"):
            if x.is_dir():shutil.rmtree(x)
        for name in ("model.tmp.pack","b3-host-gcc","b3-gap-c11"):
            if (output/name).exists():(output/name).unlink()

def main():
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--guard","--execute"):
        group.add_argument(flag,action="store_true")
    for key in ("source","screen","b3-zip","pipeline","output"):
        ap.add_argument("--"+key,type=Path)
    args=ap.parse_args()
    if args.source_check:
        print(json.dumps(source_check(),sort_keys=True))
    elif args.guard:
        print(json.dumps(guard(),sort_keys=True))
    else:
        need(all(x is not None for x in (args.source,args.screen,
             args.b3_zip,args.pipeline,args.output)),
             "REQUIRE_IMMUTABLE_AUDIO_WEIGHTS_AND_C11_INPUTS")
        execute(args.source,args.screen,args.b3_zip,args.pipeline,args.output)

if __name__=="__main__":
    main()
