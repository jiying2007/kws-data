"""Real 12-style Qwen3 synthetic-reference cloning, with immutable source cohorts.

Exactly 12 Kokoro synthetic donor styles × five predeclared phrases (60 WAVs).
Requested text, phonetic render and intended class are *NOT* training labels.
No historical Meero training replay, TTS clone retries or source-group moves.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("cohort_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
c=load("english_hifi_micro")
q=load("english_hifi_qwen_once")

SCHEMA="english-qwen3-synthetic-clone12-corpus-v1"
REPO="jiying2007/kws-data"
PREDECESSOR="7c352e15466dd62b21a4733e96f08d51d938b647"
SOURCE_RUN=44256271861
SOURCE_COMMIT=PREDECESSOR
ARTIFACT={"id":14406282094,"name":"english-neutral-clone-reference12-44256271861",
          "size_in_bytes":652901,
          "digest":"sha256:8b86b46759f6f2bd084cd37e1c804c72058fe52ec57f6c221c912eeca96092e5"}
ROLES={"af_bella":"train","af_sarah":"train","am_michael":"train",
       "am_puck":"train","af_nicole":"train","am_fenrir":"train",
       "af_aoede":"development","am_echo":"development",
       "af_kore":"calibration","af_alloy":"calibration",
       "af_nova":"locked-test","am_onyx":"locked-test"}
ROLE_FLOOR={"train":6,"development":2,"calibration":2,"locked-test":2}
# The two renderings per brand are prospective generation variants,
# never a post-observation waiver or phonetic-label substitution.
CELLS=(
    ("mero-marrow","Hey marrow.",2,"/heɪ ˈmɛroʊ/"),
    ("mero-merrow","Hey merrow.",2,"/heɪ ˈmɛroʊ/"),
    ("vosen-vohsen","Hey Vohsen.",1,"/heɪ ˈvoʊsən/"),
    ("vosen-name","Hey Vosen.",1,"/heɪ ˈvoʊsən/"),
    ("ordinary-time","What time is it?",0,None))
SOURCE_REF_SCHEMA="english-synthetic-clone-reference-12-v1"
MAX_CLONE_TTS=60
MAX_POSITIVE_SAMPLES=28672

def need(x,why):
    if not x:raise ValueError(why)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def source_contract():
    need(len(ROLES)==12 and Counter(ROLES.values())==Counter(ROLE_FLOOR),
         "ORIGINAL_PRESET_ROLE_COUNTS")
    need(len(CELLS)==5 and len({x[0] for x in CELLS})==5 and
         len(ROLES)*len(CELLS)==MAX_CLONE_TTS,
         "SIXTY_BOUNDED_NEW_CLONE_REQUESTS")
    need(Counter(cel[2] for cel in CELLS)==Counter({0:1,1:2,2:2}),
         "BOTH_BRANDS_AND_ONE_ORDINARY_PER_GROUP")
    need(CELLS[0][1]!=CELLS[1][1] and CELLS[2][1]!=CELLS[3][1] and
         "Meero" not in str(CELLS) and
         all(x[3]==(None if x[2]==0 else
                    ("/heɪ ˈmɛroʊ/" if x[2]==2 else "/heɪ ˈvoʊsən/"))
             for x in CELLS),"NO_OLD_MEERO_OR_LABEL_OVERRIDE")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY",
            "source_artifact_id":ARTIFACT["id"],"donor_groups":12,
            "planned_new_tts_calls":60,"trainable_rows":0,
            "new_asr_calls":0,"new_phone_calls":0,"training_allowed":False}

def source_metadata(meta):
    need(all(meta.get(k)==v for k,v in ARTIFACT.items()) and
         meta.get("expired") is False and
         meta.get("workflow_run",{}).get("id")==SOURCE_RUN and
         meta.get("workflow_run",{}).get("head_sha")==SOURCE_COMMIT,
         "ORIGINAL_FROZEN_REFERENCE_ARTIFACT_IDENTITY")
    return {"source_run":SOURCE_RUN,"source_artifact":ARTIFACT}

def remote_source_metadata(token):
    need(isinstance(token,str) and len(token)>=10,"READONLY_ACTIONS_TOKEN")
    url=f"https://api.github.com/repos/{REPO}/actions/artifacts/{ARTIFACT['id']}"
    req=urllib.request.Request(url,headers={
        "Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json",
        "User-Agent":"english-synthetic-clone12-pinned-source"})
    with urllib.request.urlopen(req,timeout=30) as response:
        raw=response.read(1_000_001)
    need(0<len(raw)<=1_000_000,"PINNED_SMALL_REMOTE_META")
    return source_metadata(json.loads(raw))

def guard(env=None,event=None,parent=None):
    env=os.environ if env is None else env
    expected={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in expected.items()),
         "PUBLIC_FIRST_MAIN_ONLY")
    commit=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",commit) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "ACTUAL_EXACT_COMMIT_RUN")
    if event is None:event=json.loads(c.read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before")==PREDECESSOR and event.get("after")==commit and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "EXACT_PROTECTED_MAIN_SUCCESSOR")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                       cwd=c.ROOT,text=True).strip()
    need(parent==PREDECESSOR,"FROZEN_MAIN_FIRST_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":commit,
            "source_previous":PREDECESSOR}

def reference_rows(root):
    root=Path(root)
    manifest=json.loads(c.read(root/"MANIFEST.json"))
    need(manifest.get("schema")==SOURCE_REF_SCHEMA and
         manifest.get("record_kind")=="SYNTHETIC_VOICE_CLONING_REFERENCES_ONLY" and
         manifest.get("reference_audio_count")==12 and
         manifest.get("roles")==ROLE_FLOOR and
         manifest.get("training_allowed") is False and
         manifest.get("human_gold") is False and
         manifest.get("source")=={"run":44218734553,"artifact_id":14389757284,
            "archive_sha256":"712e083b1bc7e1451c8eb43c2f99c94549d34f8d71db81d8cf2ef965e1df72da",
            "head_sha":"7e9797789cba7d1476d02f14e21af91dc3cd0712"},
         "EXACT_REUSED_NEUTRAL_PROVENANCE")
    refs={r["preset_id"]:r for r in manifest["rows"]}
    need(len(manifest["rows"])==len(refs)==12 and set(refs)==set(ROLES),
         "ALL_TWELVE_UNIQUE_REFERENCES")
    ids=[]
    for name,row in refs.items():
        need(row["id"]=="synthetic-"+name and
             row["role"]==ROLES[name] and
             row["source_group_id"]=="kokoro-v1_0-preset:"+name and
             row["reference_text"]=="Hello there. Good morning." and
             row["reference_kind"]=="SYNTHETIC_NEUTRAL_CLONE_SEED_ONLY" and
             row.get("training_allowed") is False and
             row.get("no_human_identity_gold") is True and
             row.get("independent_human_source_claim") is False,
             "REFERENCE_LINEAGE_AND_ROLE")
        wav=root/"wav"/(row["id"]+".wav")
        observed=c.inspect_wav(wav)
        need(observed["basic_audio_ok"] and
             all(observed[k]==row[v] for k,v in
                 (("wav_sha256","wav_sha256"),
                  ("pcm_sha256","joined_pcm_sha256"),
                  ("samples","pcm_samples"))) and
             40000<=observed["samples"]<=64000,
             "COMPLETE_ORIGINAL_SYNTHETIC_REFERENCE")
        ids.append(observed["pcm_sha256"])
    need(len(ids)==len(set(ids)),"DISTINCT_SYNTHETIC_REFERENCE_AUDIO")
    return refs

def planned(refs):
    need(set(refs)==set(ROLES),"ALL_SOURCE_GROUPS_PREPARED")
    result=[]
    for name in sorted(ROLES):
        ref=refs[name]
        for cell,text,clazz,ipa in CELLS:
            ident="qcl-"+name+"-"+cell
            result.append({"id":ident,"source_preset":name,
                "source_group_id":ref["source_group_id"],"role":ROLES[name],
                "clone_reference_wav_sha256":ref["wav_sha256"],
                "clone_reference_pcm_sha256":ref["joined_pcm_sha256"],
                "clone_reference_text":ref["reference_text"],
                "generation_render_text":text,"cell_id":cell,
                "requested_brand_class":clazz,"requested_target_ipa":ipa,
                "label_basis":"GENERATION_REQUEST_NOT_ACOUSTIC_TRUTH",
                "observed_class":None,"human_gold":False,"training_allowed":False})
    need(len(result)==60 and len({x["id"] for x in result})==60 and
         Counter(x["role"] for x in result)==
         Counter({role:n*5 for role,n in ROLE_FLOOR.items()}),
         "PRESPLIT_SIXTY_CLIPS")
    return result

def readback(output,refs):
    output=Path(output)
    packet=json.loads(c.read(output/"MANIFEST.json"))
    expected=planned(refs)
    need(packet.get("schema")==SCHEMA and
         packet.get("source_artifact")==ARTIFACT and
         packet.get("source_run_id")==SOURCE_RUN and
         packet.get("denominator")==MAX_CLONE_TTS and
         packet.get("training_allowed") is False and
         packet.get("human_gold") is False and
         len(packet.get("rows",[]))==60,"EXACT_FULL_SOURCE_PACKET")
    ref_pcm={x["joined_pcm_sha256"] for x in refs.values()}
    seen=set(ref_pcm)
    for actual,request in zip(packet["rows"],expected):
        need(all(actual.get(k)==v for k,v in request.items()) and
             actual.get("source_provider")=="qwen3-tts" and
             actual.get("derived_16k_from_complete_24k") is True,
             "SOURCE_REQUEST_AND_LINEAGE_MATCH")
        wav=c.inspect_wav(output/"wav"/(request["id"]+".wav"))
        need(wav["basic_audio_ok"] and
             all(actual.get(k)==wav[k] for k in
                 ("pcm_sha256","wav_sha256","samples")) and
             wav["pcm_sha256"] not in seen,
             "UNIQUE_NEW_UNCROPPED_PCM")
        seen.add(wav["pcm_sha256"])
        need(actual.get("positive_duration_pass")==
             (request["requested_brand_class"]==0 or
              actual["samples"]<=MAX_POSITIVE_SAMPLES),
             "DURATION_IS_OBSERVATION_NOT_LABEL")
    need(len(seen)==72,"DISTINCT_NEW_PCM_AND_REFERENCE")
    return {"schema":SCHEMA,"status":"SYNTHETIC_RESEARCH_CORPUS_SAVED",
            "new_source_rows":60,"source_groups":12,
            "new_training_labels":0,"training_allowed":False,
            "roles":ROLE_FLOOR}

def generate(output,reference):
    source_contract()
    identity=guard()
    refs=reference_rows(reference)
    requests=planned(refs)
    root=Path(output)
    need(root.is_absolute() and not root.exists(),"FRESH_OUTPUT_NO_REPLACEMENT")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":SCHEMA,"run":identity,
         "source_artifact":ARTIFACT,"denominator":60,
         "new_tts_calls_so_far":0,"training_allowed":False})
    completed=[]
    try:
        q.resources()
        import numpy as np
        import torch
        from importlib.metadata import version
        from qwen_tts import Qwen3TTSModel
        need(version("qwen-tts")=="0.1.1","EXACT_QWEN_TTS_0_1_1")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        model_id,revision=c.CLONE_MODEL
        folder,lock=q.model_lock(model_id,revision,
            Path(os.environ.get("RUNNER_TEMP","/tmp"))/"clone12-private-model")
        c.record(root/"MODEL.json",lock)
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        model=Qwen3TTSModel.from_pretrained(str(folder),device_map="cpu",
                                            dtype=torch.float32,
                                            attn_implementation="eager")
        prompts={}
        for name in sorted(ROLES):
            row=refs[name]
            prompts[name]=model.create_voice_clone_prompt(
                ref_audio=str(Path(reference)/"wav"/(row["id"]+".wav")),
                ref_text=row["reference_text"],x_vector_only_mode=False)
        for idx,row in enumerate(requests):
            seed=int.from_bytes(hashlib.sha256((SCHEMA+"/"+row["id"]).encode()).digest()[:4],"big")
            torch.manual_seed(seed)
            np.random.seed(seed)
            with torch.inference_mode():
                samples,sr=model.generate_voice_clone(
                    text=row["generation_render_text"],language="English",
                    voice_clone_prompt=prompts[row["source_preset"]])
            need(len(samples)==1,"EXACT_ONE_WAVE_RETURN")
            r=q.encode_whole(samples[0],sr,root,{
                **row,"seed":seed,"generated_index":idx,
                "speaker":"synthetic-"+row["source_preset"],
                "text":row["generation_render_text"],
                "requested_kind":row["cell_id"]})
            r["positive_duration_pass"]=(row["requested_brand_class"]==0 or
                                         r["samples"]<=MAX_POSITIVE_SAMPLES)
            completed.append(r)
            c.record(root/(row["id"]+".json"),r)
        c.record(root/"MANIFEST.json",{"schema":SCHEMA,"source_run_id":SOURCE_RUN,
            "source_artifact":ARTIFACT,"source_commit":SOURCE_COMMIT,
            "run":identity,"denominator":60,"completed":len(completed),
            "rows":completed,"new_tts_calls":len(completed),
            "training_allowed":False,"human_gold":False,
            "three_asr": "NOT_RUN","phoneme_observation":"NOT_RUN",
            "original_heldout_synthetic_source_roles":ROLE_FLOOR,
            "independent_human_speaker_claim":False})
        value=readback(root,refs)
        print(json.dumps(value,sort_keys=True))
    except BaseException as error:
        c.record(root/"FAILED.json",{"schema":SCHEMA,
            "error_type":type(error).__name__,
            "error_message":str(error)[:500],
            "completed_returned_audio":len(completed),
            "not_started_or_failed":60-len(completed),
            "retries":0,"training_allowed":False})
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-check",action="store_true")
    parser.add_argument("--guard",action="store_true")
    parser.add_argument("--verify-remote",action="store_true")
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--readback",action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--output")
    a=parser.parse_args()
    need(sum(bool(x) for x in (a.source_check,a.guard,a.verify_remote,
                               a.generate,a.readback))==1,"ONE_MODE_ONLY")
    if a.source_check:print(json.dumps(source_contract()));return
    if a.guard:print(json.dumps(guard()));return
    if a.verify_remote:
        guard()
        print(json.dumps(remote_source_metadata(os.environ.get("GH_TOKEN"))))
        return
    need(a.reference,"PINNED_REFERENCE_DIRECTORY_REQUIRED")
    if a.generate:
        need(a.output,"CREATE_OUTPUT_REQUIRED")
        generate(a.output,a.reference)
    else:
        need(a.output,"EXACT_READBACK_OUTPUT")
        print(json.dumps(readback(a.output,reference_rows(a.reference))))

if __name__=="__main__":main()
