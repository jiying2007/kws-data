"""Run three pinned local native-ASR command adapters without target-text leakage.

The CLI calls real external adapters ONLY with --execute, and does not supply
generation prompts, target transcripts, keywords or reference voice metadata.
The adapters must produce native receipts; their identities remain provisional
until independent model-asset/source review. No automatic retry or human gold.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import campaign as c
import runner
import screen

LOCK_SCHEMA="offline-kws-native-asr-adapters-v1"
ALLOWED_TOKENS={"{audio}","{output}"}
MAX_ASR_CALLS=3*c.MAX_CELLS


def check_adapters(value):
    c.require(type(value) is dict and set(value)=={"schema","adapters"} and
              value["schema"]==LOCK_SCHEMA,"ASR_ADAPTER_LOCK_SCHEMA")
    entries=value["adapters"]
    c.require(type(entries) is list and len(entries)==3,"ALL_THREE_ASR_REQUIRED")
    found={}
    for row in entries:
        c.require(type(row) is dict and
                  set(row)=={"engine","model_id","model_revision",
                    "model_assets_sha256","executable","executable_sha256",
                    "argv","timeout_seconds"},"ASR_ADAPTER_FIELDS")
        engine=row["engine"]
        c.require(engine in c.ENGINES and engine not in found and
                  row["model_id"]==screen.MODEL_IDS[engine] and
                  type(row["model_revision"]) is str and
                  c._REV.fullmatch(row["model_revision"]),"ASR_ENGINE_ID")
        for field in ("model_assets_sha256","executable_sha256"):
            c.require(type(row[field]) is str and c._SHA.fullmatch(row[field]),
                      "ASR_PIN_REQUIRED")
        path=Path(row["executable"])
        c.require(path.is_absolute(),"ASR_EXECUTABLE_ABSOLUTE")
        argc=row["argv"]
        c.require(type(argc) is list and 2<=len(argc)<=20 and
                  all(type(a) is str and 1<=len(a)<=128 for a in argc)
                  and set(ALLOWED_TOKENS)<=set(argc),"OPAQUE_AUDIO_AND_OUTPUT_REQUIRED")
        c.require(all("{" not in a and "}" not in a for a in argc if a not in ALLOWED_TOKENS),
                  "UNSUPPORTED_ASR_TEMPLATE")
        # No extra flags that can accidentally insert an expected keyword.
        for arg in argc:
            if arg not in ALLOWED_TOKENS:
                c.require(all(w not in arg for w in c.TARGETS),"TARGET_LEAKAGE_IN_ASR_CMD")
        c.require(type(row["timeout_seconds"]) is int and
                  5<=row["timeout_seconds"]<=300,"BOUNDED_ASR_TIMEOUT")
        found[engine]=row
    c.require(set(found)==set(c.ENGINES),"MISSING_ASR_ENGINE")
    return found


def dispatch(plan,generation_root,asr_root,adapters,*,max_calls,execute=False,
             executor=subprocess.run):
    c.validate_plan(plan)
    by_engine=check_adapters(adapters)
    c.require(type(max_calls) is int and
              0<=max_calls<=min(MAX_ASR_CALLS,3*len(plan["cells"])),
              "ASR_CALL_BUDGET")
    root=Path(asr_root)
    if not execute:
        return {"schema":"asr-auto-dispatch-dry-run-v1",
            "potential_calls":min(max_calls,3*len(plan["cells"])),
            "actual_calls":0,"training_admitted":False}
    c.require(os.environ.get("HF_HUB_OFFLINE")=="1" and
              os.environ.get("TRANSFORMERS_OFFLINE")=="1",
              "OFFLINE_ASR_RUNTIME_REQUIRED")
    for row in by_engine.values():
        _,sha=runner._safe_file(row["executable"],limit=1024*1024*8)
        c.require(sha==row["executable_sha256"],"ASR_ADAPTER_BYTES_DRIFT")
    expected_plan_sha=runner._hash(c.canonical(plan))
    if not root.exists():
        root.mkdir(parents=True)
        runner._once(root/"plan-lock.json",{"schema":"asr-autonomous-lock-v1",
            "plan_sha256":expected_plan_sha,"model_calls_authorized":False,
            "training_admitted":False})
    else:
        lock=c.read_json(root/"plan-lock.json")
        c.require(lock.get("plan_sha256")==expected_plan_sha and
                  lock.get("training_admitted") is False,"ASR_RESUME_PLAN_DRIFT")
    attempts=0
    for cell in plan["cells"]:
        done=Path(generation_root)/"cells"/cell["cell_id"]/"completed.json"
        if not done.exists():continue
        receipt=c.read_json(done,65536)
        c.require(receipt.get("schema")=="tts-generated-cell-completed-v1" and
                  receipt.get("cell_id")==cell["cell_id"] and
                  receipt.get("training_admitted") is False and
                  receipt.get("plan_sha256")==expected_plan_sha,
                  "GENERATED_SOURCE_DRIFT")
        wav=Path(generation_root)/"cells"/cell["cell_id"]/"audio-16k.wav"
        measured=screen._audio_metadata(wav)
        c.require(all(measured.get(k)==receipt["audio"].get(k)
                      for k in ("wav_sha256","pcm_sha256","audio_frames")),
                  "SOURCE_WAV_DRIFT")
        for engine in c.ENGINES:
            row=by_engine[engine]
            directory=root/cell["cell_id"]/engine
            started=directory/"started.json"
            if started.exists():continue  # even if child died: no retry
            if attempts>=max_calls:
                return {"actual_calls":attempts,"training_admitted":False,
                        "status":"BOUNDED_ASR_PARTIAL_OR_COMPLETE"}
            c.require(not directory.exists(),"FOREIGN_ASR_ATTEMPT_DIRECTORY")
            directory.mkdir(parents=True)
            runner._once(started,{"schema":"asr-native-started-v1",
                "audio_id":cell["cell_id"],"engine":engine,
                "wav_sha256":measured["wav_sha256"],
                "plan_sha256":expected_plan_sha,
                "model_id":row["model_id"],
                "model_revision":row["model_revision"],
                "model_assets_sha256":row["model_assets_sha256"],
                "attempts":1,"training_admitted":False})
            attempts+=1
            output=directory/"native.json"
            rendered=[str(wav) if t=="{audio}" else
                      str(output) if t=="{output}" else t
                      for t in row["argv"]]
            # No prompt/target/split metadata in argv, environment or stdout.
            minimal_env={
                "PATH":os.environ.get("PATH","/usr/bin:/bin"),
                "HOME":str(directory),
                "LANG":"C.UTF-8",
                "HF_HUB_OFFLINE":"1",
                "TRANSFORMERS_OFFLINE":"1",
                "PYTHONUNBUFFERED":"1",
                "OMP_NUM_THREADS":"4",
            }
            try:
                completed=executor([row["executable"],*rendered],cwd=str(directory),
                    env=minimal_env,timeout=row["timeout_seconds"],close_fds=True,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
                c.require(completed.returncode==0,"NATIVE_ASR_FAILED")
                size,digest=runner._safe_file(output,limit=8*1024*1024)
                obs=c.read_json(output,8*1024*1024)
                # Adapter writes its own full raw JSON with signed byte identity.
                # The canonical record is produced by the supervisor after hash checks.
                c.require(type(obs) is dict and set(obs)=={"actual_text","decoder_completed",
                        "processed_start_frame","processed_end_frame","audio_id",
                        "model_id","model_revision","engine"},"NATIVE_ASR_RAW_CONTRACT")
                c.require(obs["audio_id"]==cell["cell_id"] and
                          obs["engine"]==engine and
                          obs["model_id"]==row["model_id"] and
                          obs["model_revision"]==row["model_revision"] and
                          obs["decoder_completed"] is True and
                          type(obs["processed_start_frame"]) is int and
                          obs["processed_start_frame"]==0 and
                          type(obs["processed_end_frame"]) is int and
                          obs["processed_end_frame"]==measured["audio_frames"],
                          "NATIVE_ASR_INCOMPLETE_OR_BAD_IDENTITY")
                c.normalize(obs["actual_text"])
                consolidated={"schema":screen.SCHEMA,"engine":engine,
                    "model_id":row["model_id"],
                    "model_revision":row["model_revision"],
                    "audio_id":cell["cell_id"],
                    "wav_sha256":measured["wav_sha256"],
                    "pcm_sha256":measured["pcm_sha256"],
                    "audio_frames":measured["audio_frames"],
                    "status":"success","decoder_completed":True,
                    "processed_start_frame":0,
                    "processed_end_frame":measured["audio_frames"],
                    "actual_text":obs["actual_text"],
                    "raw_native_sha256":digest,"attempts":1}
                screen.validate_native_observation(consolidated,cell,measured)
                runner._once(root/cell["cell_id"]/(engine+".json"),consolidated)
                runner._once(directory/"completed.json",{"status":"NATIVE_ASR_COMPLETE_UNADMITTED",
                    "native_bytes":size,"native_sha256":digest,
                    "observation_sha256":runner._hash(c.canonical(consolidated)),
                    "training_admitted":False})
            except Exception as error:
                runner._once(directory/"failed.json",{
                    "status":"FAILED_NO_RETRY","error_type":type(error).__name__,
                    "engine":engine,"audio_id":cell["cell_id"],
                    "training_admitted":False})
    return {"actual_calls":attempts,"training_admitted":False,
        "status":"BOUNDED_ASR_PARTIAL_OR_COMPLETE"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True);p.add_argument("--generation-root",required=True)
    p.add_argument("--asr-root",required=True);p.add_argument("--adapter-lock",required=True)
    p.add_argument("--max-calls",type=int,default=72)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args(argv)
    try:
        plan=c.validate_plan(c.read_json(args.plan))
        adapters=c.read_json(args.adapter_lock)
        result=dispatch(plan,args.generation_root,args.asr_root,adapters,
                        max_calls=args.max_calls,execute=args.execute)
    except (ValueError,OSError) as err:
        print("AUTO_ASR_UNADMITTED: "+type(err).__name__,file=sys.stderr)
        return 2
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
