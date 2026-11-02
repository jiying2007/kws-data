"""Batch ASR over cloned KWS clips. Original source prompts stay on the host.

Subprocesses receive only opaque WAV and output paths; any already started
cell/engine attempt remains consumed. Machine consensus is never human gold.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import asr_dispatch as old_asr
import bulk_pool as p
import bulk_runner as tts
import campaign as c
import runner
import screen


def _audio_for(plan, generated, cell):
    folder=Path(generated)/"cells"/cell["cell_id"]
    output=folder/"completed.json"
    if not output.exists():return None
    result=p.read_json(output,65536)
    p.require(type(result) is dict and result.get("schema") ==
              "kws-bulk-clone-complete-v1" and
              result.get("cell_id")==cell["cell_id"] and
              result.get("status")=="GENERATED_PENDING_ASR" and
              result.get("training_admitted") is False and
              result.get("source_family_id")==cell["source_family_id"],
              "TTS_RECEIPT_DRIFT")
    measured=screen._audio_metadata(folder/"audio-16k.wav")
    p.require(all(result["audio"].get(field)==measured.get(field)
                  for field in ("wav_sha256","pcm_sha256","audio_frames","wav_bytes")),
              "TTS_WAV_HASH_MISMATCH")
    return folder/"audio-16k.wav",measured


def dispatch(plan, generated_root, output, adapters, *, max_new_calls,
             executor=subprocess.run, execute=False):
    p.verify_plan(plan)
    sources=old_asr.check_adapters(adapters)
    p.require(type(max_new_calls) is int and
              0<=max_new_calls<=64,"ASR_CALL_BUDGET")
    if not execute:
        return {"status":"DRY_RUN", "max_new_calls":max_new_calls,
                "actual_calls":0, "training_admitted":False}
    p.require(os.environ.get("HF_HUB_OFFLINE")=="1" and
              os.environ.get("TRANSFORMERS_OFFLINE")=="1",
              "OFFLINE_NATIVE_ASR_REQUIRED")
    for item in sources.values():
        _,sha=runner._safe_file(item["executable"],limit=8*1024*1024)
        p.require(sha==item["executable_sha256"],"ASR_BINARY_CHANGED")
    root=Path(output)
    digest=hashlib.sha256(p.canonical(plan)).hexdigest()
    if not root.exists():
        root.mkdir(parents=True)
        runner._once(root/"plan-lock.json",{
            "schema":"kws-bulk-asr-lock-v1","plan_sha256":digest,
            "training_admitted":False})
    else:
        old=p.read_json(root/"plan-lock.json")
        p.require(old.get("plan_sha256")==digest and
                  old.get("training_admitted") is False,"ASR_PLAN_CHANGED")
    started=sum((root/"clips"/cell["cell_id"]/engine/"started.json").exists()
                for cell in plan["cells"] for engine in c.ENGINES)
    p.require(started<=3*plan["budgets"]["max_total_calls"],"ASR_TOTAL_BUDGET")
    allowed=min(max_new_calls,3*plan["budgets"]["max_total_calls"]-started)
    count=0
    # Exactly the same fixed source/cell identities are reused across stages.
    for cell in plan["cells"][:plan["budgets"]["max_total_calls"]]:
        src=_audio_for(plan,generated_root,cell)
        if src is None:continue
        wav,meta=src
        for engine in c.ENGINES:
            folder=root/"clips"/cell["cell_id"]/engine
            if (folder/"started.json").exists():continue
            if count>=allowed:
                return {"actual_calls":count,"status":"ASR_PARTIAL",
                        "training_admitted":False}
            p.require(not folder.exists(),"FOREIGN_ASR_OUTPUT")
            folder.mkdir(parents=True)
            model=sources[engine]
            runner._once(folder/"started.json",{
                "schema":"kws-bulk-asr-start-v1","cell_id":cell["cell_id"],
                "wav_sha256":meta["wav_sha256"],
                "pcm_sha256":meta["pcm_sha256"],
                "audio_frames":meta["audio_frames"],
                "engine":engine,"model_revision":model["model_revision"],
                "attempts":1,"training_admitted":False})
            count+=1
            raw=folder/"native.json"
            argv=[str(wav) if x=="{audio}" else
                  str(raw) if x=="{output}" else x
                  for x in model["argv"]]
            env={"PATH":os.environ.get("PATH","/usr/bin:/bin"),
                 "HOME":str(folder),"LANG":"C.UTF-8",
                 "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                 "OMP_NUM_THREADS":"4"}
            try:
                completed=executor([model["executable"],*argv],
                                   cwd=str(folder),env=env,
                                   timeout=model["timeout_seconds"],
                                   close_fds=True,stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,check=False)
                p.require(completed.returncode==0,"ASR_EXECUTE_FAILED")
                _,raw_sha=runner._safe_file(raw,limit=8*1024*1024)
                observed=p.read_json(raw,8*1024*1024)
                p.require(type(observed) is dict and set(observed)=={
                    "audio_id","model_id","model_revision","engine",
                    "decoder_completed","processed_start_frame",
                    "processed_end_frame","actual_text"}, "ASR_RAW_SCHEMA")
                p.require(observed["audio_id"]==cell["cell_id"] and
                          observed["engine"]==engine and
                          observed["model_id"]==model["model_id"] and
                          observed["model_revision"]==model["model_revision"] and
                          observed["decoder_completed"] is True and
                          type(observed["processed_start_frame"]) is int and
                          observed["processed_start_frame"]==0 and
                          type(observed["processed_end_frame"]) is int and
                          observed["processed_end_frame"]==meta["audio_frames"],
                          "INCOMPLETE_OR_WRONG_ASR")
                c.normalize(observed["actual_text"])
                receipt={
                    "schema":screen.SCHEMA,"engine":engine,
                    "model_id":model["model_id"],
                    "model_revision":model["model_revision"],
                    "audio_id":cell["cell_id"],
                    "wav_sha256":meta["wav_sha256"],
                    "pcm_sha256":meta["pcm_sha256"],
                    "audio_frames":meta["audio_frames"],
                    "status":"success","decoder_completed":True,
                    "processed_start_frame":0,
                    "processed_end_frame":meta["audio_frames"],
                    "actual_text":observed["actual_text"],
                    "raw_native_sha256":raw_sha,"attempts":1}
                # Source/geometry and native output identities only.
                runner._once(folder/"completed.json",{
                    "schema":"kws-bulk-asr-complete-v1",
                    "observation":receipt,
                    "training_admitted":False,"qualification_allowed":False})
            except Exception as err:
                runner._once(folder/"failed.json",{
                    "schema":"kws-bulk-asr-failure-v1",
                    "cell_id":cell["cell_id"],"engine":engine,
                    "status":"FAILED_NO_RETRY",
                    "error_type":type(err).__name__,
                    "training_admitted":False})
    return {"actual_calls":count,"status":"ASR_PARTIAL_OR_COMPLETE",
            "training_admitted":False}


def screen_results(plan,gen_root,asr_root):
    p.verify_plan(plan)
    states=[]
    for cell in plan["cells"]:
        src=_audio_for(plan,gen_root,cell)
        if src is None:
            states.append({"cell_id":cell["cell_id"],"status":"NO_GENERATION",
                           "training_admitted":False})
            continue
        _,meta=src
        observed=[]
        for engine in c.ENGINES:
            receipt=Path(asr_root)/"clips"/cell["cell_id"]/engine/"completed.json"
            if not receipt.exists():continue
            result=p.read_json(receipt,65536)
            o=result["observation"]
            p.require(result.get("schema")=="kws-bulk-asr-complete-v1" and
                      result.get("training_admitted") is False and
                      result.get("qualification_allowed") is False and
                      o["audio_id"]==cell["cell_id"] and
                      o["wav_sha256"]==meta["wav_sha256"] and
                      o["pcm_sha256"]==meta["pcm_sha256"] and
                      o["audio_frames"]==meta["audio_frames"] and
                      o["status"]=="success" and
                      o["decoder_completed"] is True,
                      "ASR_RESULT_IDENTITY_DRIFT")
            native=Path(asr_root)/"clips"/cell["cell_id"]/engine/"native.json"
            _,sha=runner._safe_file(native,limit=8*1024*1024)
            p.require(sha==o["raw_native_sha256"],"ASR_NATIVE_CHANGED")
            observed.append(c.normalize(o["actual_text"]))
        expected=c.normalize(cell["prompt"])
        if len(observed)<3:
            status="ASR_INCOMPLETE_HOLD"
        elif all(x==expected for x in observed):
            status=("MACHINE_POSITIVE_CANDIDATE" if
                    cell["intended_kind"]=="positive-intent" else
                    "MACHINE_NEGATIVE_CANDIDATE")
        elif cell["intended_kind"]!="positive-intent" and
                any(x in ("你好小窝","小窝小窝") for x in observed):
            status="NEGATIVE_SPOKE_WAKE_OR_ASR_CONFLICT"
        else:
            status="ASR_DISAGREEMENT_OR_TTS_MISPRONUNCIATION"
        states.append({
            "cell_id":cell["cell_id"],
            "reference_id":cell["reference_id"],
            "source_family_id":cell["source_family_id"],
            "role":cell["role"],"intended_kind":cell["intended_kind"],
            "asr_completed":len(observed),
            "status":status,"training_admitted":False})
    counts=dict(Counter(x["status"] for x in states))
    return {"schema":"kws-bulk-clone-machine-screen-v1",
            "rows":states,"counts":counts,
            "human_gold":False,"rights_authenticated":False,
            "training_admitted":False,"qualification_allowed":False,
            "shipping_approved":False}


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--plan",required=True)
    cli.add_argument("--generated",required=True)
    cli.add_argument("--asr-output",required=True)
    cli.add_argument("--adapter-lock",required=True)
    cli.add_argument("--max-new-calls",type=int,default=48)
    cli.add_argument("--execute",action="store_true")
    cli.add_argument("--report")
    args=cli.parse_args(argv)
    try:
        plan=p.read_json(args.plan,8*1024*1024)
        p.verify_plan(plan)
        if args.execute:
            adapters=p.read_json(args.adapter_lock)
            result=dispatch(plan,args.generated,args.asr_output,adapters,
                            max_new_calls=args.max_new_calls,execute=True)
        else:
            result={"status":"NO_NEW_MODEL_CALL","actual_calls":0}
        if args.report:
            report=screen_results(plan,args.generated,args.asr_output)
            runner._once(Path(args.report),report)
    except (OSError,ValueError) as err:
        print("BULK_ASR_NOT_ADMITTED: "+type(err).__name__,file=sys.stderr)
        return 2
    print(json.dumps({**result,"training_admitted":False},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
