"""Real bounded Qwen3-1.7B Base clone control for selected TRAIN donor styles.

Research only. No old audio regenerated, no label from text, no held-out
speaker used for selecting the renderer. Two original synthetic train-role
references, exactly eight native forward requests, no retry or sweep.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("q17_"+name,HERE/(name+".py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m
c=load("english_hifi_micro")
q=load("english_hifi_qwen_once")
donor=load("english_clone_cohort12_once")
SCHEMA="english-qwen17-two-train-donor-8clip-v1"
REPO="jiying2007/kws-data"
BEFORE="9a201c689b1cf1e963969fca2ae88dc48f5124d8"
MODEL=("Qwen/Qwen3-TTS-12Hz-1.7B-Base","fd4b254389122332181a7c3db7f27e918eec64e3")
REFERENCE_RUN=44256271861
REFERENCE_ARTIFACT=14406282094
REF_ZIP_SHA="8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5"
TRAIN_DONORS=("af_bella","am_puck")
RENDERS=(
    ("mero-marrow","Hey marrow.",2,"/heɪ ˈmɛroʊ/"),
    ("mero-merrow","Hey merrow.",2,"/heɪ ˈmɛroʊ/"),
    ("vosen-vohsen","Hey Vohsen.",1,"/heɪ ˈvoʊsən/"),
    ("vosen-name","Hey Vosen.",1,"/heɪ ˈvoʊsən/"))
MAX_CALLS=8
MAX_POSITIVE_SAMPLES=28672

def need(ok,why):
    if not ok:raise ValueError(why)
def source_check():
    need(len(TRAIN_DONORS)==2 and TRAIN_DONORS==("af_bella","am_puck") and
         all(donor.ROLES[n]=="train" for n in TRAIN_DONORS) and
         len(RENDERS)==4 and len(TRAIN_DONORS)*len(RENDERS)==MAX_CALLS,
         "FROZEN_TWO_TRAIN_DONORS_EIGHT_REQUESTS")
    need(len({x[0] for x in RENDERS})==4 and "Meero" not in str(RENDERS),
         "NO_OLD_MEERO_RENDER")
    need(MODEL[1]=="fd4b254389122332181a7c3db7f27e918eec64e3",
         "EXACT_1_7B_UPSTREAM_WEIGHTS")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY",
            "model":MODEL,"new_tts_calls":0,"training_allowed":False,
            "maximum_requests":MAX_CALLS,"heldout_donor_calls":0}

def guard(env=None,event=None,parent=None):
    env=os.environ if env is None else env
    required={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),
         "ONE_PUBLIC_MAIN_ONLY")
    sha=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "PINNED_GITHUB_RUN")
    if event is None:event=json.loads(c.read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before")==BEFORE and event.get("after")==sha and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "PINNED_PUBLIC_FIRST_MAIN")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                      cwd=c.ROOT,text=True).strip()
    need(parent==BEFORE,"EXACT_FIRST_PARENT")
    return {"sha":sha,"run_id":env["GITHUB_RUN_ID"],"parent":BEFORE}

def source_refs(root):
    # Original donor tool verifies complete 12 reference WAV PCM, source
    # provenance, independent source-group identity and unchanged roles.
    refs=donor.reference_rows(root)
    need(len(refs)==12,"ORIGINAL_TWELVE_SOURCE_REFERENCE_SET")
    selected={name:refs[name] for name in TRAIN_DONORS}
    need(all(r["role"]=="train" for r in selected.values()),
         "ONLY_TRAIN_ROLE_REFERENCE_IN_QUALITY_PILOT")
    return selected

def planned(refs):
    need(set(refs)==set(TRAIN_DONORS),"TWO_FIXED_DONOR_REFERENCE_IDS")
    rows=[]
    for name in TRAIN_DONORS:
        for cell,text,klass,ipa in RENDERS:
            rows.append({"id":"q17-"+name+"-"+cell,
                         "source_preset":name,
                         "source_group_id":refs[name]["source_group_id"],
                         "role":"train",
                         "reference_pcm_sha256":refs[name]["joined_pcm_sha256"],
                         "reference_wav_sha256":refs[name]["wav_sha256"],
                         "reference_text":refs[name]["reference_text"],
                         "generation_render_text":text,
                         "cell_id":cell,"requested_brand_class":klass,
                         "requested_target_ipa":ipa,
                         "observed_class":None,"human_gold":False,
                         "training_allowed":False,
                         "label_basis":"REQUEST_IS_NOT_GOLD"})
    need(len(rows)==MAX_CALLS and len({x["id"] for x in rows})==MAX_CALLS,
         "EIGHT_UNIQUE_NEW_CLONE_REQUESTS")
    return rows

def readback(out,ref_root):
    refs=source_refs(ref_root)
    path=Path(out)
    packet=json.loads(c.read(path/"MANIFEST.json"))
    need(packet.get("schema")==SCHEMA and packet.get("model")==MODEL[0]
         and packet.get("model_revision")==MODEL[1]
         and packet.get("reference_artifact_id")==REFERENCE_ARTIFACT
         and packet.get("reference_run_id")==REFERENCE_RUN
         and packet.get("denominator")==MAX_CALLS
         and len(packet.get("rows",[]))==MAX_CALLS
         and packet.get("training_allowed") is False,
         "EXACT_QWEN_1_7B_MODEL_AND_ALL_EIGHT_SOURCE_ROWS")
    seen=set(x["joined_pcm_sha256"] for x in refs.values())
    for p,r in zip(planned(refs),packet["rows"]):
        need(all(r.get(k)==v for k,v in p.items()),"REQUEST_ROLE_OR_TARGET_DRIFT")
        observed=c.inspect_wav(path/"wav"/(p["id"]+".wav"))
        need(observed["basic_audio_ok"] and
             all(observed[k]==r[k] for k in ("wav_sha256","pcm_sha256","samples"))
             and r["pcm_sha256"] not in seen,
             "AUTHENTIC_FULL_PCM_NEW_WEIGHT_AUDIO")
        seen.add(r["pcm_sha256"])
        need(r["positive_duration_pass"]==(r["samples"]<=MAX_POSITIVE_SAMPLES),
             "OBSERVED_DURATION_DOES_NOT_CHANGE_AUDIO")
    return {"status":"AUTHENTIC_EIGHT_SYNTHETIC_RESEARCH_CLIPS",
            "source_roles":{"train":2},"max_training_positives":0,
            "training_allowed":False}

def generate(root,reference):
    source_check()
    identity=guard()
    refs=source_refs(reference)
    rows=planned(refs)
    root=Path(root)
    need(root.is_absolute() and not root.exists(),"CREATE_UNIQUE_OUTPUT")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":SCHEMA,"run":identity,
             "model":MODEL,"planned":MAX_CALLS,"training_allowed":False})
    out=[]
    try:
        q.resources()
        import numpy as np
        import torch
        from importlib.metadata import version
        from qwen_tts import Qwen3TTSModel
        need(version("qwen-tts")=="0.1.1","EXACT_QWEN_RUNTIME")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        private=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"qwen17-private-model"
        folder,lock=q.model_lock(MODEL[0],MODEL[1],private)
        c.record(root/"MODEL.json",lock)
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        model=Qwen3TTSModel.from_pretrained(str(folder),
                  device_map="cpu",dtype=torch.float32,attn_implementation="eager")
        prompts={}
        for name in TRAIN_DONORS:
            row=refs[name]
            prompts[name]=model.create_voice_clone_prompt(
                ref_audio=str(Path(reference)/"wav"/(row["id"]+".wav")),
                ref_text=row["reference_text"],x_vector_only_mode=False)
        for i,row in enumerate(rows):
            seed=int.from_bytes(hashlib.sha256(
                (SCHEMA+"/"+row["id"]).encode()).digest()[:4],"big")
            torch.manual_seed(seed);np.random.seed(seed)
            with torch.inference_mode():
                waves,sr=model.generate_voice_clone(
                    text=row["generation_render_text"],language="English",
                    voice_clone_prompt=prompts[row["source_preset"]])
            need(len(waves)==1,"EXACT_ONE_NATIVE_WAVE")
            result=q.encode_whole(waves[0],sr,root,{
                **row,"seed":seed,"generated_index":i,
                "speaker":"synthetic-"+row["source_preset"],
                "text":row["generation_render_text"],
                "requested_kind":row["cell_id"]})
            result["positive_duration_pass"]=(result["samples"]<=MAX_POSITIVE_SAMPLES)
            out.append(result)
            c.record(root/(row["id"]+".json"),result)
        c.record(root/"MANIFEST.json",{"schema":SCHEMA,
             "model":MODEL[0],"model_revision":MODEL[1],
             "run":identity,"reference_artifact_id":REFERENCE_ARTIFACT,
             "reference_run_id":REFERENCE_RUN,"reference_zip_sha256":REF_ZIP_SHA,
             "denominator":MAX_CALLS,"rows":out,
             "completed_native_forwards":len(out),
             "training_allowed":False,"human_gold":False,
             "independent_phone":"NOT_RUN","three_asr":"NOT_RUN"})
        print(json.dumps(readback(root,reference)))
    except BaseException as exc:
        c.record(root/"FAILED.json",{"schema":SCHEMA,
          "error":type(exc).__name__+":"+str(exc)[:400],
          "actual_returns":len(out),"unknown_remaining":MAX_CALLS-len(out),
          "no_retry":True,"training_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--guard",action="store_true")
    ap.add_argument("--run",action="store_true")
    ap.add_argument("--readback",action="store_true")
    ap.add_argument("--reference")
    ap.add_argument("--output")
    a=ap.parse_args()
    need(sum(bool(x) for x in (a.source_check,a.guard,a.run,a.readback))==1,
         "ONE_MODE")
    if a.source_check:print(json.dumps(source_check()));return
    if a.guard:print(json.dumps(guard()));return
    need(a.reference and a.output,"SAVED_DONOR_INPUT_AND_OUTPUT_REQUIRED")
    if a.run:generate(a.output,a.reference)
    else:print(json.dumps(readback(a.output,a.reference)))

if __name__=="__main__":main()
