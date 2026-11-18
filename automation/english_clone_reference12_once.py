"""Reuse original immutable Kokoro neutral PCM as twelve clone-reference voices.

No new TTS, phoneme, ASR or training inference. The joined neutral audio is
never eligible for KWS positives or negatives. It is *only* a synthetic
voice-cloning reference. Preserve all original source-group roles.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import urllib.request
import wave
from collections import Counter,defaultdict

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("read_zip_frozen",HERE/"english_canonical_piper_reuse_audit.py")
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

REPO="jiying2007/kws-data"
BEFORE="de43b57620b36c952d8e57887ba9016a090fc1de"
RUN=44218734553
HEAD="7e9797789cba7d1476d02f14e21af91dc3cd0712"
ART=14389757284
NAME="english-mero-meh-corpus-44218734553"
ZIP_BYTES=74476997
ZIP_SHA="712e083b1bc7e1451c8eb43c2f99c94549d34f8d71db81d8cf2ef965e1df72da"
SCHEMA="english-synthetic-clone-reference-12-v1"
EXPECTED_ROLES={
"af_bella":"train","af_sarah":"train","am_michael":"train",
"am_puck":"train","af_nicole":"train","am_fenrir":"train",
"af_aoede":"development","am_echo":"development",
"af_kore":"calibration","af_alloy":"calibration",
"af_nova":"locked-test","am_onyx":"locked-test"}
EXPECTED_COUNTS={"train":6,"development":2,"calibration":2,"locked-test":2}
NEUTRAL={"hello":"Hello there.","morning":"Good morning."}
GAP_SAMPLES=1600 # exactly 100 ms silent gap in CLONE REFERENCE ONLY
REF_TEXT="Hello there. Good morning."

def need(value,reason):
    if not value:raise ValueError(reason)
def sha(data):return hashlib.sha256(data).hexdigest()
def check_contract():
    need(len(EXPECTED_ROLES)==12 and Counter(EXPECTED_ROLES.values())==
         Counter(EXPECTED_COUNTS),"EXACT_INDEPENDENT_STYLE_ROLES")
    need(len(NEUTRAL)==2 and GAP_SAMPLES==1600 and
         all("Mero" not in v and "Vosen" not in v for v in NEUTRAL.values()),
         "NO_WAKE_IN_REFERENCE")
    return {"status":"SOURCE_ONLY","voices":12,"new_tts_calls":0,
            "new_asr_calls":0,"new_phone_calls":0,
            "training_positives":0,"training_negatives":0}

def original_audio(pcm,wav,source):
    need(sha(pcm)==source.get("pcm_sha256") and
         sha(wav)==source.get("wav_sha256"),"AUTHENTICATED_SOURCE_PCM_SHA")
    with wave.open(io.BytesIO(wav),"rb") as w:
        need((w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype())==
             (1,2,16000,"NONE") and
             w.getnframes()*2==len(pcm) and
             w.readframes(w.getnframes())==pcm,
             "EXACT_COMPLETE_MONO_PCM")
    # Independently recompute levels from the original full PCM, not from
    # a requested source label or a pre-existing quality flag.
    from array import array
    import math
    values=array("h")
    values.frombytes(pcm)
    rms=math.sqrt(sum(v*v for v in values)/len(values))/32768
    rail=sum(v in (-32768,32767) for v in values)/len(values)
    need(rms>=0.0001 and rail<=0.01,
         "INDEPENDENT_REFERENCE_PCM_LEVELS")
    return len(pcm)//2

def select(data):
    source=json.loads(data["CORPUS_MANIFEST.json"])
    admission=json.loads(data["ADMISSION.json"])
    final=json.loads(data["RESULT.json"])
    # The source manifest described prospective eligibility (True), but the
    # *actual* frozen post-observation decision was a training STOP. Never
    # rewrite the historic source's intent to invent a passing corpus.
    need(source.get("campaign_id")=="english-mero-meh-corpus-v1" and
         len(source.get("rows",[]))==144 and
         source.get("training_allowed") is True and
         source.get("human_gold") is False,
         "IMMUTABLE_ORIGINAL_PROSPECTIVE_SOURCE")
    need(admission.get("campaign_id")==source["campaign_id"] and
         admission.get("status")=="STOP_FRESH_CORPUS_COVERAGE" and
         admission.get("training_allowed") is False and
         admission.get("training_snapshot_allowed") is False and
         admission.get("balanced_groups_by_role")=={
             "train":0,"development":0,"calibration":0,"locked-test":0} and
         admission.get("unknown_is_background") is False and
         final.get("admission_status")==admission["status"] and
         final.get("training_allowed") is False,
         "IMMUTABLE_HISTORICAL_REJECTED_ADMISSION")
    index=defaultdict(dict)
    for row in source["rows"]:
        pid=row.get("preset_id");cell=row.get("cell_id")
        need(pid in EXPECTED_ROLES and row.get("role")==EXPECTED_ROLES[pid] and
             cell not in index[pid],"SOURCE_GROUP_ROLE_OR_DUPLICATE")
        index[pid][cell]=row
    need(set(index)==set(EXPECTED_ROLES) and all(len(x)==12 for x in index.values()),
         "FULL_ORIGINAL_12_PRESET_ROSTER")
    result=[]
    for name in sorted(EXPECTED_ROLES):
        audio=[]; original=[]
        for cell,text in NEUTRAL.items():
            row=index[name][cell]
            need(row.get("requested_text")==text and row.get("requested_class")==0
                 and row.get("generation_status")=="success"
                 and row.get("human_gold") is False
                 and row.get("source_group_id")=="kokoro-v1_0-preset:"+name
                 and row.get("pcm_quality_passed") is True,
                 "CLOSED_NEUTRAL_SYNTHETIC_REFERENCE")
            pcm=data[row["pcm_path"]];wav=data[row["wav_path"]]
            length=original_audio(pcm,wav,row)
            original.append({"id":row["id"],"cell_id":cell,"samples":length,
                "pcm_sha256":row["pcm_sha256"],"wav_sha256":row["wav_sha256"]})
            audio.append(pcm)
        joined=audio[0]+b"\x00\x00"*GAP_SAMPLES+audio[1]
        result.append({"id":"synthetic-"+name,"preset_id":name,
            "source_group_id":"kokoro-v1_0-preset:"+name,
            "role":EXPECTED_ROLES[name],
            "reference_text":REF_TEXT,"reference_kind":"SYNTHETIC_NEUTRAL_CLONE_SEED_ONLY",
            "components":original,"joined_pcm_sha256":sha(joined),
            "pcm_samples":len(joined)//2,"sample_rate":16000,
            "no_human_identity_gold":True,"training_allowed":False,
            "independent_human_source_claim":False,"wav_pcm":joined})
    need(len(result)==12 and Counter(x["role"] for x in result)==Counter(EXPECTED_COUNTS),
         "EXACT_REFERENCE_GROUP_FLOOR")
    return result

def fetch(token):
    need(type(token) is str and len(token)>=10,"READ_ONLY_GITHUB_TOKEN")
    opener=urllib.request.build_opener(audit.SafeArtifactRedirect())
    api=f"https://api.github.com/repos/{REPO}/actions/artifacts/{ART}"
    def get(url,limit):
        req=urllib.request.Request(url,headers={
            "Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "User-Agent":"english-kokoro-reference-12-once"})
        with opener.open(req,timeout=90) as response:
            b=response.read(limit+1)
        need(0<len(b)<=limit,"BOUNDED_ARTIFACT_BYTES")
        return b
    meta=json.loads(get(api,1_000_000))
    need(meta.get("id")==ART and meta.get("name")==NAME
         and meta.get("expired") is False
         and meta.get("size_in_bytes")==ZIP_BYTES
         and meta.get("digest")=="sha256:"+ZIP_SHA
         and meta.get("workflow_run",{}).get("id")==RUN
         and meta.get("workflow_run",{}).get("head_sha")==HEAD,
         "PINNED_HISTORICAL_SOURCE_ARTIFACT")
    archive=get(api+"/zip",ZIP_BYTES)
    need(len(archive)==ZIP_BYTES and sha(archive)==ZIP_SHA,
         "FROZEN_ARCHIVE_BYTE_IDENTITY")
    return audit.read_zip(archive)

def guard():
    env=os.environ
    expected={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
       "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
       "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in expected.items()),"FIRST_PUBLIC_MAIN_ONLY")
    sha_commit=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha_commit),"ACTUAL_MAIN_COMMIT_SHA")
    event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==sha_commit and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "SINGLE_APPROVED_REFERENCE_MATERIALIZATION")
    import subprocess
    parent=subprocess.check_output(["git","rev-parse","HEAD^1"],text=True).strip()
    need(parent==BEFORE,"FIRST_PARENT_SOURCE_BOUND")
    return {"commit":sha_commit,"before":BEFORE,
            "run_id":env["GITHUB_RUN_ID"]}

def write(root,selected):
    root=Path(root)
    need(root.is_absolute() and not root.exists(),"CREATE_NEVER_REPLACE")
    root.mkdir(parents=True)
    (root/"wav").mkdir()
    rows=[]
    for item in selected:
        row={k:v for k,v in item.items() if k!="wav_pcm"}
        path=root/"wav"/(row["id"]+".wav")
        with wave.open(str(path),"wb") as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes(item["wav_pcm"])
        wb=path.read_bytes()
        with wave.open(io.BytesIO(wb),"rb") as w:
            need(w.readframes(w.getnframes())==item["wav_pcm"],
                 "REFERENCE_WAV_READBACK")
        row["wav_sha256"]=sha(wb)
        rows.append(row)
    manifest={"schema":SCHEMA,"source":{"run":RUN,"artifact_id":ART,
         "archive_sha256":ZIP_SHA,"head_sha":HEAD},
         "record_kind":"SYNTHETIC_VOICE_CLONING_REFERENCES_ONLY",
         "reference_audio_count":12,"roles":EXPECTED_COUNTS,
         "gap_samples":GAP_SAMPLES,"rows":rows,
         "new_model_calls":0,"human_gold":False,"training_allowed":False}
    (root/"MANIFEST.json").write_text(json.dumps(
        manifest,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    return manifest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-check",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--output")
    a=parser.parse_args()
    need(a.source_check!=a.run,"ONE_EXECUTION_MODE")
    if a.source_check:print(json.dumps(check_contract()));return
    need(a.output,"EXACT_OUTPUT_PATH")
    guard()
    selected=select(fetch(os.environ.get("GH_TOKEN")))
    result=write(a.output,selected)
    print(json.dumps({"status":"SYNTHETIC_CLONE_REFERENCES_MATERIALIZED",
         "groups":result["reference_audio_count"],
         "roles":result["roles"],"new_model_calls":0,
         "training_positives":0}))

if __name__=="__main__":main()
