"""Bounded, real Qwen3-TTS 0.6B CustomVoice -> synthetic-reference clone.

No ASR labels here. Audio is saved whole and never admitted for training.
One model per job, eight total TTS forwards, no retry or sample replacement.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("hifi_micro_contract",HERE/"english_hifi_micro.py")
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

def need(x,reason):c.need(x,reason)

def resources():
    disk=shutil.disk_usage(os.environ.get("RUNNER_TEMP","/tmp")).free
    avail=0
    with open("/proc/meminfo",encoding="ascii") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail=int(line.split()[1])*1024
    need(disk>=8*1024**3 and avail>=5*1024**3,
         "STOP_CPU_RESOURCE_FLOOR_NOT_A_MODEL_FAILURE")
    return {"available_disk_bytes":disk,"available_ram_bytes":avail}

def model_lock(model_id, revision, cache):
    from huggingface_hub import HfApi, snapshot_download
    details=HfApi(token=False).model_info(model_id,revision=revision)
    need(details.sha.startswith(revision),"SOURCE_REVISION_MISMATCH")
    path=snapshot_download(repo_id=model_id,revision=details.sha,
                           cache_dir=str(cache),max_workers=2,token=False)
    need(Path(path).is_dir(),"PINNED_MODEL_NOT_DOWNLOADED")
    return path,{"model":model_id,"revision":details.sha,"license":"apache-2.0",
                 "qwen_tts_runtime":"0.1.1","device":"cpu",
                 "attention":"eager","dtype":"float32"}

def encode_whole(wav,sr,root,row):
    import numpy as np
    from scipy.signal import resample_poly
    import wave
    need(sr==24000,"FROZEN_24K_GENERATION")
    data=np.asarray(wav,dtype=np.float32)
    need(data.ndim==1 and len(data)>0 and len(data)<=24000*15
         and np.all(np.isfinite(data)),"FINITE_WHOLE_AUDIO")
    values=resample_poly(data,2,3).astype(np.float32)
    need(3200<=len(values)<=240000 and np.all(np.isfinite(values)),
         "COMPLETE_OUTPUT_DURATION")
    pcm=np.rint(np.clip(values,-1,32767/32768)*32768).astype("<i2").tobytes()
    directory=root/"wav"
    directory.mkdir(parents=True,exist_ok=True)
    p=directory/(row["id"]+".wav")
    need(not p.exists(),"NEVER_REPLACE_GENERATED_AUDIO")
    with wave.open(str(p),"wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    observed=c.inspect_wav(p)
    need(observed["basic_audio_ok"],"BASIC_AUDIO_QUALITY_STOP_NO_REPLACEMENT")
    return {**row,**observed,"source_provider":"qwen3-tts",
            "human_gold":False,"training_allowed":False,
            "derived_16k_from_complete_24k":True}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("custom","clone"),required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--source")
    opts=parser.parse_args()
    c.contract()
    run=c.guard()
    root=Path(opts.output)
    need(root.is_absolute() and not root.exists(),"CREATE_NEW_OUTPUT_ONLY")
    root.mkdir(parents=True)
    source=None
    if opts.stage=="clone":
        need(opts.source,"CLONE_REQUIRES_PROVEN_SYNTHETIC_REFERENCE")
        source=c.verify_stage(opts.source,"custom")
        need(source["rows"][0]["id"]=="aiden-reference","REFERENCE_ROW_IDENTITY")
    else:
        need(opts.source is None,"CUSTOM_HAS_NO_SOURCE")
    plan=c.CUSTOM if opts.stage=="custom" else c.CLONE
    model_id,revision=c.CUSTOM_MODEL if opts.stage=="custom" else c.CLONE_MODEL
    c.record(root/"STARTED.json",{"schema":c.SCHEMA,"stage":opts.stage,
        "run":run,"planned":len(plan),"max_tts_forwards":len(plan),
        "prior_real_tts_forwards_this_run":0,"training_allowed":False})
    completed=[]
    try:
        facts=resources()
        import numpy as np
        import torch
        from qwen_tts import Qwen3TTSModel
        from importlib.metadata import version
        need(version("qwen-tts")=="0.1.1","PIN_QWEN_PACKAGE")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        path,lock=model_lock(model_id,revision,root/"models-cache")
        lock.update(facts)
        c.record(root/"MODEL.json",lock)
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        model=Qwen3TTSModel.from_pretrained(
            str(path),device_map="cpu",dtype=torch.float32,attn_implementation="eager")
        prompt=None
        if opts.stage=="clone":
            prompt=model.create_voice_clone_prompt(
                ref_audio=str(Path(opts.source)/"wav"/"aiden-reference.wav"),
                ref_text=c.REFERENCE,x_vector_only_mode=False)
        for i,item in enumerate(plan):
            if opts.stage=="custom":
                sid,speaker,utterance,kind=item
            else:
                sid,utterance,kind=item
                speaker="clone-of-aiden-reference"
            seed=int.from_bytes(hashlib.sha256(
                (c.SCHEMA+"/"+sid).encode()).digest()[:4],"big")
            torch.manual_seed(seed)
            np.random.seed(seed)
            with torch.inference_mode():
                if opts.stage=="custom":
                    wavs,sr=model.generate_custom_voice(
                        text=utterance,language="English",speaker=speaker)
                else:
                    wavs,sr=model.generate_voice_clone(
                        text=utterance,language="English",voice_clone_prompt=prompt)
            need(len(wavs)==1,"ONE_WHOLE_PCM_RETURN_PER_REQUEST")
            row={"id":sid,"speaker":speaker,"text":utterance,
                 "requested_kind":kind,"seed":seed,
                 "generated_index":i,"source_group":"synthetic-aiden" if
                 ("aiden" in sid) else "synthetic-ryan"}
            completed.append(encode_whole(wavs[0],sr,root,row))
            c.record(root/("row-"+sid+".json"),completed[-1])
        c.record(root/"MANIFEST.json",{"schema":c.SCHEMA,"stage":opts.stage,
            "model":lock,"planned":len(plan),"completed":len(completed),
            "rows":completed,"real_tts_forwards":len(completed),
            "new_human_recordings":0,"training_allowed":False,
            "independent_phone_result":"NOT_RUN_IN_THIS_STAGE"})
        c.verify_stage(root,opts.stage)
        print(json.dumps({"status":"GENERATED_RESEARCH_ONLY","stage":opts.stage,
                          "rows":len(completed),"training_allowed":False}))
    except BaseException as exc:
        c.record(root/"FAILED.json",{"schema":c.SCHEMA,"stage":opts.stage,
                 "error_type":type(exc).__name__,"error_message":str(exc)[:500],
                 "completed_forward_returns":len(completed),
                 "unknown_remaining":len(plan)-len(completed),
                 "automatic_retry":False,"training_allowed":False})
        raise

if __name__=="__main__":main()
