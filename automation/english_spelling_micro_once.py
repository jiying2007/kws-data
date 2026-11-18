"""One preregistered pronunciation-render experiment; no new training labels.

Display brand names never change. Exact English TTS rendering text differs
from requested canonical KWS class only for independent acoustic diagnosis.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
def load(name):
    p=HERE/(name+".py")
    spec=importlib.util.spec_from_file_location("orthographic_"+name,p)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
c=load("english_hifi_micro")
q=load("english_hifi_qwen_once")
SCHEMA="english-brand-pronunciation-render-v1"
REPO="jiying2007/kws-data"
BEFORE="18e520d4295a2f421268f93d7197155090e70c83"
ORIGINAL_RUN=44251082652
ORIGINAL_REFERENCE_WAV_SHA="d3e92a72db2f9b2aa85e06ff55ecb778e1f321981db4581cad9cacd0f89ab2a6"
CANONICAL={"MERO":"/heɪ ˈmɛroʊ/","VOSEN":"/heɪ ˈvoʊsən/"}
CUSTOM=[
    ("aiden-marrow","Aiden","Hey marrow.","MERO","synthetic-aiden"),
    ("aiden-merrow","Aiden","Hey merrow.","MERO","synthetic-aiden"),
    ("aiden-vohsen","Aiden","Hey Vohsen.","VOSEN","synthetic-aiden"),
    ("aiden-vozen-control","Aiden","Hey Vozen.","Z_CONTROL","synthetic-aiden"),
    ("ryan-marrow","Ryan","Hey marrow.","MERO","synthetic-ryan"),
    ("ryan-merrow","Ryan","Hey merrow.","MERO","synthetic-ryan"),
    ("ryan-vohsen","Ryan","Hey Vohsen.","VOSEN","synthetic-ryan"),
    ("ryan-vozen-control","Ryan","Hey Vozen.","Z_CONTROL","synthetic-ryan")]
CLONE=[
    ("clone-aiden-marrow","Hey marrow.","MERO","synthetic-aiden"),
    ("clone-aiden-vohsen","Hey Vohsen.","VOSEN","synthetic-aiden")]
SOURCE_PCMS={
"54ce0c43a58ad3202a10e0333742faddf447f52dbff7e65e65d077e740ce4917",
"2745da7c611b7f1e46d91cdc562d3e884b3243ad9aab512337a5c25ab8bb636d",
"c21f041e3ad6ed820a8fb73ae217d4ba0aa1f3ad5b4053348dd8bc31055649a4",
"95f4081b5a30d346e638e7d2851bb1fa81fae4b833ca9f41c3e8120436e545f3",
"1c6603a2462b4a3efbc17fbad1de7cb3ed6bd7f3a7923e820ff68c2d61b42067",
"9fe5025369d35b3ac22ad5abe0b30071bd762ba07672b2680b01833e3424cdb5",
"091aa0cdd7dd55e1515696399d351e3bcc36fc660926eb93ae106bc1fa09bb2f",
"0227758b6445b4fe9398eacf32377496387f03dedb4e114c513ab6ab60975efd"}
def need(test,why):
    if not test:raise ValueError(why)
def source_check():
    need(len(CUSTOM)==8 and len(CLONE)==2 and
         len({x[0] for x in CUSTOM+CLONE})==10 and
         len(SOURCE_PCMS)==8,"EXACT_TEN_PREREGISTERED_UNIQUE_CALLS")
    need(set(x[3] for x in CUSTOM)=={"MERO","VOSEN","Z_CONTROL"} and
         set(x[2] for x in CLONE)=={"MERO","VOSEN"},"CANONICAL_WITH_Z_CONTRAST")
    need(all(x[2]!="Hey Mero." and x[2]!="Hey Vosen." for x in CUSTOM) and
         all(x[1]!="Hey Mero." and x[1]!="Hey Vosen." for x in CLONE),
         "NO_ALREADY_GENERATED_TEXT_REPLAY")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY","planned_tts":10,
            "planned_phone":20,"zero_repeated_old_tts":True,"training_allowed":False}
def guard(env=None,event=None,parent=None):
    env=os.environ if env is None else env
    required={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
      "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
      "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),"PUBLIC_FIRST_MAIN_ONLY")
    sha=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),"GITHUB_RUN_IDENTITY")
    if event is None:event=json.loads(c.read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before")==BEFORE and event.get("after")==sha and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("private") is False and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("forced") is False and event.get("deleted") is False,
         "PROTECTED_FIRST_PARENT_ONLY")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],cwd=c.ROOT,text=True).strip()
    need(parent==BEFORE,"FIRST_PARENT_SHA")
    return {"commit":sha,"run_id":env["GITHUB_RUN_ID"],"source_predecessor":BEFORE}
def item(row,stage):
    if stage=="custom":sid,speaker,render,target,group=row
    else:sid,render,target,group=row;speaker="synthetic-reference-aiden"
    return {"id":sid,"speaker":speaker,"generation_render_text":render,
            "brand_target":CANONICAL.get(target),"target_kind":target,
            "source_group":group,"reference_origin":"synthetic-aiden" if stage=="clone" else None,
            "training_allowed":False,"human_gold":False}
def verify(root,stage):
    rows=CUSTOM if stage=="custom" else CLONE
    root=Path(root);m=json.loads(c.read(root/"MANIFEST.json"))
    need(m.get("schema")==SCHEMA and m.get("stage")==stage and
         m.get("planned")==len(rows) and len(m.get("rows",[]))==len(rows),
         "EXACT_FIXED_GENERATION_MANIFEST")
    pcms=set(SOURCE_PCMS)
    for planned,observed in zip(rows,m["rows"]):
        expected=item(planned,stage)
        need(all(observed.get(k)==v for k,v in expected.items()) and
             observed.get("source_provider")=="qwen3-tts" and
             observed.get("training_allowed") is False,"UNALTERED_RENDER_TARGET_SEPARATION")
        identity=c.inspect_wav(root/"wav"/(expected["id"]+".wav"))
        need(identity["basic_audio_ok"] and all(observed.get(k)==identity[k]
             for k in ("wav_sha256","pcm_sha256","samples")),
             "COMPLETE_AUTHENTICATED_WAV")
        need(identity["pcm_sha256"] not in pcms,"NO_DUPLICATE_OLD_OR_NEW_PCM")
        pcms.add(identity["pcm_sha256"])
    return {"status":"AUTHENTICATED_RESEARCH_ONLY","stage":stage,
            "rows":len(m["rows"]),"new_training_labels":0}
def generate(stage,root,source=None):
    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel
    from importlib.metadata import version
    source_check()
    identity=guard()
    root=Path(root)
    need(root.is_absolute() and not root.exists(),"CREATE_OUTPUT_ONCE")
    if stage=="clone":
        need(source is not None,"CLONE_REQUIRES_IMMUTABLE_REFERENCE")
        original=c.verify_stage(source,"custom")
        ref=Path(source)/"wav"/"aiden-reference.wav"
        need(c.inspect_wav(ref)["wav_sha256"]==ORIGINAL_REFERENCE_WAV_SHA,
             "NO_REFERENCE_VOICE_SUBSTITUTION")
    else:need(source is None,"CUSTOM_CANNOT_IMPORT_OTHER_RECORDING")
    root.mkdir(parents=True)
    rows=CUSTOM if stage=="custom" else CLONE
    c.record(root/"STARTED.json",{"schema":SCHEMA,"run":identity,"stage":stage,
      "planned":len(rows),"training_allowed":False,"actual_calls":0})
    done=[]
    try:
        q.resources()
        need(version("qwen-tts")=="0.1.1","FROZEN_QWEN_TTS_RUNTIME")
        model_id,rev=c.CUSTOM_MODEL if stage=="custom" else c.CLONE_MODEL
        model_root,model_info=q.model_lock(model_id,rev,Path(os.environ.get("RUNNER_TEMP","/tmp"))/"orthographic-model-cache")
        c.record(root/"MODEL.json",model_info)
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        model=Qwen3TTSModel.from_pretrained(str(model_root),device_map="cpu",
                                            dtype=torch.float32,attn_implementation="eager")
        prompt=None
        if stage=="clone":
            prompt=model.create_voice_clone_prompt(ref_audio=str(ref),ref_text=c.REFERENCE,
                                                    x_vector_only_mode=False)
        for idx,planned in enumerate(rows):
            record=item(planned,stage)
            seed=int.from_bytes(hashlib.sha256((SCHEMA+"/"+record["id"]).encode()).digest()[:4],"big")
            torch.manual_seed(seed);np.random.seed(seed)
            with torch.inference_mode():
                if stage=="custom":
                    wav,sr=model.generate_custom_voice(
                        text=record["generation_render_text"],language="English",
                        speaker=record["speaker"])
                else:
                    wav,sr=model.generate_voice_clone(
                        text=record["generation_render_text"],language="English",
                        voice_clone_prompt=prompt)
            need(len(wav)==1,"ONE_NATIVE_PCM_PER_FIXED_REQUEST")
            generated=q.encode_whole(wav[0],sr,root,record|{"seed":seed,
                            "generated_index":idx})
            done.append(generated)
            c.record(root/(record["id"]+".json"),generated)
        c.record(root/"MANIFEST.json",{"schema":SCHEMA,"stage":stage,
                 "planned":len(rows),"rows":done,"new_tts_calls":len(done),
                 "training_allowed":False,"speaker_independence_claim":False})
        result=verify(root,stage)
        print(json.dumps(result))
    except BaseException as exc:
        c.record(root/"FAILED.json",{"schema":SCHEMA,"stage":stage,
                  "completed":len(done),"unobserved":len(rows)-len(done),
                  "error_type":type(exc).__name__,"error":str(exc)[:500],
                  "retry":False,"training_allowed":False})
        raise
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--guard",action="store_true")
    ap.add_argument("--generate",choices=("custom","clone"))
    ap.add_argument("--verify",choices=("custom","clone"))
    ap.add_argument("--output")
    ap.add_argument("--source")
    a=ap.parse_args()
    need(sum(bool(x) for x in (a.source_check,a.guard,a.generate,a.verify))==1,
         "ONE_MODE")
    if a.source_check:print(json.dumps(source_check()));return
    if a.guard:print(json.dumps(guard()));return
    if a.generate:generate(a.generate,a.output,a.source);return
    need(a.source,"READBACK_SOURCE_REQUIRED")
    print(json.dumps(verify(a.source,a.verify)))
if __name__=="__main__":main()
