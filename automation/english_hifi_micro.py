"""Small, frozen English pronunciation/voice-clone research contract.

No old sound is relabeled, no training, no human-gold claims. Only actual
GitHub main first-parent execution can reach a model in companion workers.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import wave

ROOT=Path(__file__).resolve().parents[1]
REPO="jiying2007/kws-data"
PREDECESSOR="163816b9e343d66eab67efd7cd6c7dcb34a68a6d"
SCHEMA="english-hifi-micro-20261118-v1"
CUSTOM_MODEL=("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice","85e237c")
CLONE_MODEL=("Qwen/Qwen3-TTS-12Hz-0.6B-Base","5d83992")
PHONE_MODELS=(
    ("facebook/wav2vec2-xlsr-53-espeak-cv-ft","2c733782da5604684829819a5eb744c193fe9398"),
    ("facebook/wav2vec2-lv-60-espeak-cv-ft","ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"))
REFERENCE="The quiet garden gate creaked open as the morning sunlight arrived."
CUSTOM=(
    ("aiden-reference","Aiden",REFERENCE,"reference"),
    ("aiden-mero","Aiden","Hey Mero.","canonical-mero-request"),
    ("aiden-meero-control","Aiden","Hey Meero.","legacy-pronunciation-control"),
    ("aiden-vosen","Aiden","Hey Vosen.","canonical-vosen-request"),
    ("ryan-mero","Ryan","Hey Mero.","canonical-mero-request"),
    ("ryan-vosen","Ryan","Hey Vosen.","canonical-vosen-request"))
CLONE=(
    ("clone-aiden-mero","Hey Mero.","canonical-mero-request"),
    ("clone-aiden-vosen","Hey Vosen.","canonical-vosen-request"))
BASELINE={"source_run":44107239806,"screen_run":44238057637,
          "screen_artifact":14398960405,"canonical_rows":144,
          "mero": {"silver":0,"weak":1,"unknown":23},
          "vosen":{"silver":5,"weak":3,"unknown":16},
          "complete_source_groups":{"train":0,"development":0,"calibration":0,"locked-test":0}}
EXPECTED_COUNTS={"custom":6,"clone":2}
MAX_TTS_CALLS=8
MAX_PHONE_CALLS=14

def need(condition,why):
    if not condition: raise ValueError(why)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def record(path,obj):
    path=Path(path)
    need(not path.exists(),"IMMUTABLE_RECEIPT")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,
                               allow_nan=False)+"\n",encoding="utf-8")

def read(path,max_bytes=1000000):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and
         0<path.stat().st_size<=max_bytes,"BOUNDED_SOURCE_FILE")
    return path.read_bytes()

def contract():
    need(len(CUSTOM)==6 and len(CLONE)==2 and MAX_TTS_CALLS==8 and
         MAX_PHONE_CALLS==(len(CUSTOM)+len(CLONE)-1)*2,
         "EXACT_EIGHT_SYNTHETIC_AND_FOURTEEN_PHONE")
    need(len({x[0] for x in CUSTOM+CLONE})==8 and
         all("Mero" not in x[2] or x[3]=="canonical-mero-request" for x in CUSTOM),
         "NO_HIDDEN_WAKE_IN_REFERENCE")
    need(CUSTOM[0][2]==REFERENCE and "Mero" not in REFERENCE and
         "Vosen" not in REFERENCE and all(x[2]!="Hey Meero." or
         x[3]=="legacy-pronunciation-control" for x in CUSTOM),
         "CONTROL_NOT_POSITIVE")
    need(all(re.fullmatch("[0-9a-f]{7,40}",rev) for name,rev in
             (CUSTOM_MODEL,CLONE_MODEL,*PHONE_MODELS)),"FROZEN_MODEL_REVISIONS")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY",
            "baseline":BASELINE,"custom":CUSTOM,"clone":CLONE,
            "model_calls":0,"training_allowed":False,
            "human_gold":False,"speaker_group_count_for_training":0}

def guard(env=None,event=None,parent=None):
    env=os.environ if env is None else env
    expected={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in expected.items()),"ONLY_FIRST_PUBLIC_MAIN_PUSH")
    sha=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha) and env.get("GITHUB_RUN_ID","").isdigit(),
         "EXACT_EXECUTION_IDENTITY")
    if event is None:
        event=json.loads(read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before")==PREDECESSOR and event.get("after")==sha
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "FROZEN_PREDECESSOR_PUBLIC_ONCE")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==PREDECESSOR,"EXACT_MAIN_FIRST_PARENT")
    return {"run":env["GITHUB_RUN_ID"],"sha":sha,"before":PREDECESSOR}

def inspect_wav(path):
    raw=read(path,24*1024*1024)
    with wave.open(io.BytesIO(raw),"rb") as w:
        need((w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype())==
             (1,2,16000,"NONE"),"PCM16_MONO_16K")
        count=w.getnframes()
        need(3200<=count<=240000,"BOUNDED_COMPLETE_WAVE")
        pcm=w.readframes(count)
    need(len(pcm)==count*2,"EXACT_PCM")
    from array import array
    import math
    samples=array("h")
    samples.frombytes(pcm)
    rms=math.sqrt(sum(x*x for x in samples)/len(samples))/32768
    rail=sum(x in (-32768,32767) for x in samples)/len(samples)
    return {"wav_sha256":digest(raw),"pcm_sha256":digest(pcm),
            "samples":count,"rms":rms,"rail_fraction":rail,
            "basic_audio_ok":rms>=0.0001 and rail<=0.01}

def verify_stage(root,stage):
    root=Path(root)
    info=json.loads(read(root/"MANIFEST.json"))
    expected=CUSTOM if stage=="custom" else CLONE
    need(info.get("schema")==SCHEMA and info.get("stage")==stage and
         info.get("planned")==len(expected) and
         len(info.get("rows",[]))==len(expected),
         "EXACT_FIXED_STAGE_ROWS")
    seen=set()
    for row,plan in zip(info["rows"],expected):
        need(row["id"]==plan[0] and row["text"]==plan[2 if stage=="custom" else 1]
             and row.get("requested_kind")==plan[-1] and
             row["id"] not in seen,"NO_ROW_SUBSTITUTION")
        seen.add(row["id"])
        need(row.get("source_provider")=="qwen3-tts" and
             row.get("human_gold") is False and
             row.get("training_allowed") is False,
             "RESEARCH_ONLY_ROWS")
        v=inspect_wav(root/"wav"/(row["id"]+".wav"))
        need(all(v[k]==row[k] for k in ("wav_sha256","pcm_sha256","samples"))
             and v["basic_audio_ok"],"AUTHENTICATED_AUDIO")
    return info

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true")
    p.add_argument("--guard",action="store_true")
    p.add_argument("--verify",choices=["custom","clone"])
    p.add_argument("--source")
    a=p.parse_args()
    need(sum(bool(x) for x in (a.check,a.guard,a.verify))==1,"ONE_MODE_ONLY")
    if a.check: print(json.dumps(contract()));return
    if a.guard: print(json.dumps(guard()));return
    need(a.source,"SOURCE_DIRECTORY_REQUIRED")
    out=verify_stage(a.source,a.verify)
    print(json.dumps({"status":"SOURCE_AUTHENTICATED","stage":a.verify,
                      "rows":len(out["rows"]),"training_allowed":False}))

if __name__=="__main__":main()
