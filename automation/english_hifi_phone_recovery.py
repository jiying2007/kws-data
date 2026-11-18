"""Authenticate saved, immutable high-fidelity TTS audio before phone-only recovery.

This job MUST NOT synthesize or clone audio again. Independent phone inference
is the only allowed numerical work, on original #210 WAV bytes only.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import urllib.request
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("hifi_micro_recovery_contract",HERE/"english_hifi_micro.py")
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
RUN=44251082652
HEAD="ec58aa844b33f736c5a002dc81afac1d62509a28"
FILES={
    "custom": {"id":14404670058,"name":"english-hifi-micro-custom",
               "bytes":259592,"digest":"sha256:9618cf65f1bc3cac0492792dd633f7bdba014782837a4b0f440b1038b80b9216"},
    "clone": {"id":14404571043,"name":"english-hifi-micro-clone",
              "bytes":48306,"digest":"sha256:abd62865f52dcb82dc83c34391420485cc0e8e4391c2890cbaa3c6fa8ab514f9"}}

def check_metadata(item,meta):
    c.need(all(meta.get(k)==v for k,v in (("id",item["id"]),
             ("name",item["name"]),("size_in_bytes",item["bytes"]),
             ("digest",item["digest"]),("expired",False))),
           "IMMUTABLE_ARTIFACT_DIGEST_IDENTITY")
    r=meta.get("workflow_run",{})
    c.need(r.get("id")==RUN and r.get("head_sha")==HEAD,
           "AUTHENTICATED_ORIGINAL_SUCCESSFUL_TTS_RUN")
    return {"artifact_id":meta["id"],"original_run_id":RUN,
            "source_commit":HEAD,"sha256":meta["digest"],
            "independent_training_group_count":0}

def validate_all(metas):
    c.need(set(metas)==set(FILES),"EXACT_ORIGINAL_SOURCE_SET")
    return {key:check_metadata(FILES[key],metas[key]) for key in FILES}

def fetch_metadata(token):
    c.need(type(token) is str and len(token)>=10,"READONLY_ACTIONS_TOKEN_REQUIRED")
    out={}
    for name,info in FILES.items():
        url=f"https://api.github.com/repos/{c.REPO}/actions/artifacts/{info['id']}"
        req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token,
                       "Accept":"application/vnd.github+json",
                       "X-GitHub-Api-Version":"2022-11-28",
                       "User-Agent":"english-hifi-phone-recovery-once"})
        with urllib.request.urlopen(req,timeout=30) as resp:
            body=resp.read(1_000_001)
        c.need(0<len(body)<=1_000_000,"SMALL_AUTHENTICATED_META")
        out[name]=json.loads(body)
    return validate_all(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--verify-remote",action="store_true")
    a=ap.parse_args()
    c.need(a.source_check != a.verify_remote,"ONE_RECOVERY_MODE")
    if a.source_check:
        c.need(len(FILES)==2 and len({v["id"] for v in FILES.values()})==2,
                "TWO_DISTINCT_FROZEN_ARTIFACTS")
        print(json.dumps({"source_contract":"PASS","new_tts_calls":0,
              "new_voice_clone_calls":0,"new_phone_calls":0,"training_allowed":False}))
        return
    c.guard(recovery=True)
    print(json.dumps({"status":"FROZEN_ORIGINAL_AUDIO","artifacts":fetch_metadata(
        os.environ.get("GH_TOKEN")),"new_tts_calls":0,"new_clone_calls":0,
        "training_allowed":False},sort_keys=True))

if __name__=="__main__":main()
