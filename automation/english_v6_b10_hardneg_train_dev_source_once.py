#!/usr/bin/env python3
"""B10 source stage: six TRAIN hard negatives in existing train groups and six new DEV voices.

The six TRAIN voices are pinned to already-qualified source groups from the
original 6/2/2/2 machine-screened corpus. Six new Piper identities are DEV-only.
This stage performs exactly 18 TTS forwards, no ASR/phone/KWS inference,
no optimizer, and no CAL/LOCKED audio access.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import wave

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SPEC=importlib.util.spec_from_file_location(
    "b10_previous_piper_source",HERE/"english_piper_newtrain24_once.py")
previous=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(previous)
PLAN=HERE/"english_v6_b10_hardneg_train_dev_source_plan.json"
REPO="jiying2007/kws-data"
BEFORE="d61c6f23bdc8c99fd6306cb2e3311e8910196442"
SCHEMA="english-v6-b10-hardneg-train-dev-source-v1"
OUTPUT_COUNT=18
TRAIN_GROUPS=6
DEV_GROUPS=6
MAX_POSITIVE_SAMPLES=28672

def need(value,reason):
    if not value:
        raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canon(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False).encode()

def read_json(path):
    return json.loads(Path(path).read_bytes())

def write_once(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,ensure_ascii=False,
                  indent=2,allow_nan=False)
        f.write("\n")

def plan():
    obj=read_json(PLAN)
    model=obj["model"]
    old=previous.old.MODEL
    train=obj["train_groups"]
    dev=obj["development_speakers"]
    need(obj["schema"]==SCHEMA
         and obj["status"]=="FROZEN_BEFORE_ANY_NEW_WAV"
         and obj["target"]=={"keyword":"Hey Mero","ipa":"/heɪ ˈmɛroʊ/",
                             "language":"en-US","active_keyword_count":1}
         and obj["counts"]["new_original_wavs"]==OUTPUT_COUNT
         and obj["counts"]["train_hard_negative_candidates"]==TRAIN_GROUPS
         and obj["counts"]["development_groups"]==DEV_GROUPS
         and obj["counts"]["roles"]=={"train":6,"development":12}
         and [x["speaker_index"] for x in train]==[448,449,450,451,452,453]
         and [x["source_group_id"] for x in train]==[
             "piper-libritts-p7498","piper-libritts-p3905",
             "piper-libritts-p7384","piper-libritts-p2992",
             "piper-libritts-p30","piper-libritts-p497"]
         and dev==list(range(540,546))
         and obj["minimum_groups_eligible"]=={
             "train":6,"development":2,"calibration":2,"locked-test":2}
         and obj["budget"]=={
             "max_new_piper_tts_forwards":18,
             "max_new_asr_forwards_at_source_stage":0,
             "max_new_phone_forwards_at_source_stage":0,
             "max_new_kws_forwards":0,"max_new_optimizer_updates":0,
             "retry_count":0}
         and obj["guards"]["train_source_groups_must_already_be_machine_qualified"] is True
         and obj["guards"]["no_calibration_or_locked_audio_open"] is True
         and model["repo"]==old["repository"]
         and model["commit"]==old["revision"]
         and model["file"]==old["model_path"]
         and model["file_bytes"]==old["model_bytes"]
         and model["file_sha256"]==old["model_sha256"]
         and model["config"]==old["config_path"]
         and model["config_bytes"]==old["config_bytes"]
         and model["config_md5"]==old["config_md5"]
         and model["piper_tts_version"]==old["piper_version"]
         and model["native_rate"]==22050 and model["derived_rate"]==16000
         and model["num_speakers"]==904,
         "FROZEN_B10_HARDNEG_TRAIN_DEV_SOURCE_PLAN")
    return obj

def source_check():
    p=plan()
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_AUDIO_GENERATED",
            "keyword":"Hey Mero","new_original_wavs":OUTPUT_COUNT,
            "train_hard_negative_candidates":TRAIN_GROUPS,
            "development_groups":DEV_GROUPS,
            "development_originals":DEV_GROUPS*2,
            "source_roles":p["counts"]["roles"],
            "train_source_artifact":p["original_train_source"]["artifact_id"],
            "train_screen_artifact":p["original_train_screen"]["artifact_id"],
            "new_speaker_indices":[540,545],
            "new_optimizer_updates":0,"new_asr_phone_calls":0,
            "new_kws_forwards":0,"cal_locked_opened":False,
            "training_allowed":False,"product_promotion":False,
            "plan_sha256":sha(canon(p))}

def original_train_groups(train_source,train_screen):
    p=plan()
    manifest=read_json(Path(train_source)/"MANIFEST.json")
    screen=read_json(Path(train_screen)/"SCREEN.json")
    src_run=p["original_train_source"];screen_run=p["original_train_screen"]
    need(manifest.get("schema")=="english-v5-mero-fresh24-independent-source-once-v1"
         and manifest.get("run")=={
             "run_id":str(src_run["run"]),"commit":src_run["head"],
             "before":"90fa488738acde10eca3a3cd92e5cb5b31eff5c4",
             "attempt":1}
         and manifest.get("denominator")==96 and manifest.get("completed")==96
         and manifest.get("new_tts_calls")==96
         and manifest.get("new_asr_calls")==0
         and manifest.get("new_phone_calls")==0
         and manifest.get("training_allowed") is False
         and manifest.get("human_gold") is False
         and len(manifest.get("rows",[]))==96,
         "EXACT_ORIGINAL_TRAIN_SOURCE_MANIFEST")
    need(screen.get("schema")=="english-v5-mero-fresh24-saved-5-native-screen-no-model-v1"
         and screen.get("status")=="PASS_NEW_TRAIN_DEV_MACHINE_SCREEN_ONLY"
         and screen.get("source_run")==src_run["run"]
         and screen.get("original_observer_run")==44481695979
         and screen.get("source_group_counts")=={"development":2,"train":11}
         and screen.get("tier_counts")=={"silver":25,"unknown":32,"weak":7}
         and screen.get("observed_label_counts")=={"0":16,"2":16}
         and screen.get("training_snapshot_allowed") is False
         and screen.get("locked_kws_test_opened") is False
         and len(screen.get("rows",[]))==64,
         "EXACT_ORIGINAL_TRAIN_SOURCE_SCREEN")
    selected={}
    for group in p["train_groups"]:
        gid=group["source_group_id"]
        src=[r for r in manifest["rows"] if r.get("source_group_id")==gid]
        qa=[r for r in screen["rows"] if r.get("source_group_id")==gid]
        need(len(src)==4 and len(qa)==4
             and {r["cell_id"] for r in src}=={
                 "mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge"}
             and {r["cell_id"] for r in qa}=={
                 "mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge"}
             and all(r["role"]=="train" for r in src+qa)
             and all(r["speaker_index"]==group["speaker_index"]
                     and r["source_speaker_id"]==group["source_speaker_id"]
                     for r in src)
             and all(r["speaker_index"]==group["speaker_index"]
                     for r in qa)
             and {r["observed_class"] for r in qa
                  if r["tier"] in ("silver","weak")}=={0,2},
             "ONLY_PREQUALIFIED_EXISTING_TRAIN_GROUP_IDENTITY")
        selected[gid]={
            "speaker_index":group["speaker_index"],
            "source_speaker_id":group["source_speaker_id"],
            "old_train_pcm_sha256":sorted(r["pcm_sha256"] for r in src)}
    return selected

def native_speakers(config,train_source,train_screen):
    mapping=config.get("speaker_id_map",{})
    need(config.get("num_speakers")==904 and isinstance(mapping,dict)
         and len(mapping)==904 and set(mapping.values())==set(range(904)),
         "EXACT_904_SPEAKER_SOURCE_MAP")
    inv={i:label for label,i in mapping.items()}
    original=original_train_groups(train_source,train_screen)
    p=plan()
    for group in p["train_groups"]:
        need(inv[group["speaker_index"]]==group["source_speaker_id"]
             and group["source_group_id"] in original,
             "EXACT_EXISTING_TRAIN_SPEAKER_IDENTITY")
    used_index=set(range(58))|set(range(400,540))
    used_index|={i for i,_ in previous.old.NEW_VOICES}
    need(not set(p["development_speakers"]).intersection(used_index),
         "NEW_DEV_SPEAKER_INDEX_ALREADY_EXPOSED")
    used_numeric=set(previous.old.OLD_LIBRI_IDS)|set(previous.old.OLD_PIPER_IDS)
    used_numeric|={speaker for _,speaker in previous.old.NEW_VOICES}
    used_numeric|={int(inv[i][1:]) for i in used_index
                   if re.fullmatch(r"p[0-9]+",inv[i])}
    dev={i:inv[i] for i in p["development_speakers"]}
    need(len(set(dev.values()))==DEV_GROUPS
         and all(re.fullmatch(r"p[0-9]+",v)
                 and int(v[1:]) not in used_numeric for v in dev.values()),
         "NEW_DEV_HUMAN_SPEAKER_ANCESTRY_UNIQUE")
    combined={g["speaker_index"]:g["source_speaker_id"] for g in p["train_groups"]}
    combined.update(dev)
    return combined

def planned(choices):
    p=plan()
    need(set(choices)=={448,449,450,451,452,453,540,541,542,543,544,545},
         "EXACT_SIX_TRAIN_SPEAKERS_AND_SIX_NEW_DEV_SPEAKERS")
    rows=[]
    train_cell=p["cells"]["train_hard_negative"]
    for group in p["train_groups"]:
        idx=group["speaker_index"];speaker=choices[idx]
        rows.append({"id":f"b10-train-{idx}-ordinary-time-hard",
          "speaker_index":idx,"source_speaker_id":speaker,
          "source_group_id":group["source_group_id"],"role":"train",
          "cell_id":train_cell["id"],"requested_text":train_cell["text"],
          "requested_class":train_cell["requested_class"],
          "original_raw_ipa":train_cell["raw_ipa"],
          "generation_length_scale":train_cell["length_scale"],
          "observed_class":None,"human_gold":False,"training_allowed":False,
          "label_basis":"NEW_SOURCE_REQUEST_NOT_ACOUSTIC_TRUTH"})
    for idx in p["development_speakers"]:
        speaker=choices[idx]
        gid="piper-libritts-"+speaker
        for key in ("dev_positive","dev_ordinary"):
            cell=p["cells"][key]
            rows.append({"id":f"b10-dev-{idx}-{cell['id']}",
              "speaker_index":idx,"source_speaker_id":speaker,
              "source_group_id":gid,"role":"development",
              "cell_id":cell["id"],"requested_text":cell["text"],
              "requested_class":cell["requested_class"],
              "original_raw_ipa":cell["raw_ipa"],
              "generation_length_scale":cell["length_scale"],
              "observed_class":None,"human_gold":False,
              "training_allowed":False,
              "label_basis":"NEW_SOURCE_REQUEST_NOT_ACOUSTIC_TRUTH"})
    need(len(rows)==OUTPUT_COUNT
         and len({r["id"] for r in rows})==OUTPUT_COUNT
         and len({r["source_group_id"] for r in rows})==TRAIN_GROUPS+DEV_GROUPS
         and {r["role"] for r in rows}=={"train","development"},
         "EXACT_18_TRAIN_DEV_ORIGINAL_SOURCE_CELLS")
    return rows

def preflight_config(config_path,train_source,train_screen):
    p=plan();model=p["model"]
    config_path=Path(config_path)
    need(config_path.is_absolute() and config_path.is_file()
         and not config_path.is_symlink()
         and config_path.stat().st_size==model["config_bytes"]
         and hashlib.md5(config_path.read_bytes()).hexdigest()==model["config_md5"],
         "PINNED_OFFICIAL_904_SPEAKER_CONFIG_BYTES")
    config=read_json(config_path)
    choices=native_speakers(config,train_source,train_screen)
    phones=config.get("phoneme_id_map",{})
    need(isinstance(phones,dict) and len(phones)>0
         and all(ch==" " or ch in phones
                 for cell in p["cells"].values()
                 for ch in cell["raw_ipa"]),
         "ALL_PREDECLARED_IPA_PHONEMES_PRESENT")
    return {"status":"SOURCE_MAP_PREFLIGHT_PASS_NO_AUDIO",
            "speakers_by_index":{str(k):v for k,v in choices.items()},
            "train_groups":TRAIN_GROUPS,"new_dev_groups":DEV_GROUPS,
            "new_tts_calls":0,"new_asr_calls":0,"new_kws_forwards":0}

def verify_remote(token):
    need(isinstance(token,str) and len(token)>10,
         "READONLY_GITHUB_ACTIONS_TOKEN_REQUIRED")
    p=plan()
    def get(path):
        req=urllib.request.Request("https://api.github.com/repos/"+REPO+path,
          headers={"Authorization":"Bearer "+token,
                   "Accept":"application/vnd.github+json",
                   "User-Agent":"kws-v6-b10-hardneg-source-provenance"})
        with urllib.request.urlopen(req,timeout=40) as f:
            raw=f.read(1000001)
        need(len(raw)<=1000000,"BOUNDED_REMOTE_SOURCE_METADATA")
        return json.loads(raw)
    for item in (p["original_train_source"],p["original_train_screen"]):
        run=get("/actions/runs/"+str(item["run"]))
        need(run.get("id")==item["run"] and run.get("head_sha")==item["head"]
             and run.get("head_branch")=="main" and run.get("event")=="push"
             and run.get("run_attempt")==1 and run.get("status")=="completed"
             and run.get("conclusion")=="success",
             "ORIGINAL_MACHINE_QUALIFIED_TRAIN_SOURCE_RUN")
        art=get("/actions/artifacts/"+str(item["artifact_id"]))
        need(art.get("id")==item["artifact_id"] and art.get("name")==item["name"]
             and art.get("size_in_bytes")==item["bytes"]
             and art.get("digest")=="sha256:"+item["sha256"]
             and art.get("expired") is False
             and art.get("workflow_run",{}).get("id")==item["run"]
             and art.get("workflow_run",{}).get("head_sha")==item["head"],
             "ORIGINAL_TRAIN_SCREEN_ARTIFACT_BYTE_PROVENANCE")
    return {"train_source":p["original_train_source"]["artifact_id"],
            "train_screen":p["original_train_screen"]["artifact_id"]}

def guard(env=None,event=None,parent=None):
    source_check();env=os.environ if env is None else env
    required={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),
         "ONLY_FIRST_PUBLIC_MAIN_SOURCE_PUSH")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch(r"[0-9a-f]{40}",head) is not None
         and re.fullmatch(r"[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "REAL_PUBLIC_MAIN_EVENT")
    if event is None:event=read_json(env["GITHUB_EVENT_PATH"])
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "EXACT_FIRST_B10_SOURCE_MAIN_SUCCESSOR")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                        cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"NO_REPLAY_OR_OUT_OF_ORDER_SOURCE")
    return {"run_id":env["GITHUB_RUN_ID"],"head":head,"parent":BEFORE}

def readback(root,train_source,train_screen,identity=None):
    root=Path(root);p=plan()
    data=read_json(root/"MANIFEST.json")
    lock=read_json(root/"MODEL_LOCK.json")
    if identity is None:identity=guard()
    need(data.get("schema")==SCHEMA and data.get("run")==identity
         and data.get("source_plan_sha256")==source_check()["plan_sha256"]
         and data.get("denominator")==OUTPUT_COUNT
         and data.get("completed")==OUTPUT_COUNT
         and data.get("new_tts_calls")==OUTPUT_COUNT
         and data.get("new_asr_calls")==0 and data.get("new_phone_calls")==0
         and data.get("training_allowed") is False
         and data.get("human_gold") is False
         and lock.get("model")==p["model"]
         and lock.get("source_plan_sha256")==source_check()["plan_sha256"]
         and lock.get("speakers_by_index")==data.get("speakers_by_index"),
         "EXACT_B10_18_WAV_SOURCE_RECEIPT")
    choices={int(k):v for k,v in data["speakers_by_index"].items()}
    old=original_train_groups(train_source,train_screen)
    expected=planned(choices)
    seen=set()
    for row,orig in zip(data["rows"],expected):
        need(all(row.get(k)==v for k,v in orig.items())
             and row.get("source_provider")=="official-piper-libritts-904"
             and row.get("whole_native_resample") is True
             and row.get("crop_calls")==0,
             "IMMUTABLE_B10_SOURCE_ROLE_AND_REQUEST")
        wav=root/"wav"/(row["id"]+".wav")
        info=previous.c.inspect_wav(wav)
        need(info["basic_audio_ok"]
             and all(row.get(k)==info[k] for k in
                     ("wav_sha256","pcm_sha256","samples"))
             and row["pcm_sha256"] not in seen,
             "WHOLE_NEW_WAV_AND_PCM_SHA")
        if row["role"]=="train":
            need(row["source_group_id"] in old
                 and row["pcm_sha256"] not in old[row["source_group_id"]]["old_train_pcm_sha256"],
                 "NEW_TRAIN_HARDNEG_NOT_DUPLICATED_IN_EXISTING_GROUP")
        else:
            need(row["speaker_index"] in p["development_speakers"]
                 and row["source_group_id"] not in old,
                 "DEV_SPEAKERS_SOURCE_DISJOINT_FROM_TRAIN")
        seen.add(row["pcm_sha256"])
    return {"status":"18_NEW_B10_TRAIN_DEV_ORIGINAL_WAVS_VALIDATED",
            "train_hard_negative_candidates":TRAIN_GROUPS,
            "new_development_groups":DEV_GROUPS,
            "original_wavs":OUTPUT_COUNT,"new_asr_calls":0,
            "new_phone_calls":0,"new_kws_forwards":0,
            "training_allowed":False,"cal_locked_opened":False,
            "human_gold":False,"product_promotion":False}

def generate(output,train_source,train_screen):
    identity=guard()
    p=plan();out=Path(output)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "ONE_NEW_B10_SOURCE_OUTPUT_DIRECTORY")
    out.mkdir()
    write_once(out/"STARTED.json",{"schema":SCHEMA,"run":identity,
        "max_original_wavs":OUTPUT_COUNT,
        "source_plan_sha256":source_check()["plan_sha256"],
        "new_kws_forwards":0,"new_optimizer_updates":0,"retry_count":0})
    started=0;finished=[]
    try:
        verify_remote(os.environ.get("GH_TOKEN"))
        from importlib.metadata import version
        import numpy as np
        from scipy.signal import resample_poly
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice,SynthesisConfig
        need(version("piper-tts")=="1.8.0" and version("numpy")=="1.26.4",
             "EXACT_PINNED_CPU_PIPER_RUNTIME")
        model=p["model"]
        cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"b10-hardneg-piper"
        weights=Path(hf_hub_download(repo_id=model["repo"],filename=model["file"],
              revision=model["commit"],local_dir=str(cache),token=False))
        cfg=Path(hf_hub_download(repo_id=model["repo"],filename=model["config"],
              revision=model["commit"],local_dir=str(cache),token=False))
        need(weights.is_file() and weights.stat().st_size==model["file_bytes"]
             and previous.digest(weights)==model["file_sha256"]
             and cfg.is_file() and cfg.stat().st_size==model["config_bytes"]
             and hashlib.md5(cfg.read_bytes()).hexdigest()==model["config_md5"],
             "EXACT_OFFICIAL_PIPER_MODEL_AND_CONFIG")
        config=read_json(cfg)
        choices=native_speakers(config,train_source,train_screen)
        voice=PiperVoice.load(str(weights))
        need(voice.config.sample_rate==22050 and voice.config.num_speakers==904,
             "EXACT_NATIVE_LIBRITTS_PIPER_VOICE")
        write_once(out/"MODEL_LOCK.json",{"model":model,
            "source_plan_sha256":source_check()["plan_sha256"],
            "speakers_by_index":{str(k):v for k,v in choices.items()},
            "training_allowed":False})
        rows=planned(choices)
        for row in rows:
            phrase="[["+row["original_raw_ipa"]+"]]"
            phones=[s for seq in voice.phonemize(phrase) for s in seq]
            need(phones==list(row["original_raw_ipa"])
                 and all(s in voice.config.phoneme_id_map for s in phones if s!=" "),
                 "DIRECT_NATIVE_IPA_EXACT_BEFORE_SYNTHESIS")
            started+=1
            write_once(out/"rows"/(row["id"]+"-STARTED.json"),
                {"id":row["id"],"attempt":1,"role":row["role"],
                 "source_group_id":row["source_group_id"],"position":started})
            setting=SynthesisConfig(speaker_id=row["speaker_index"],
                length_scale=row["generation_length_scale"],
                noise_scale=0.333,noise_w_scale=0.333,
                normalize_audio=False,volume=1.0)
            chunks=list(voice.synthesize(phrase,setting))
            need(len(chunks)==1 and chunks[0].sample_rate==22050
                 and chunks[0].sample_channels==1 and chunks[0].sample_width==2,
                 "ONE_COMPLETE_NATIVE_PIPER_CHUNK")
            native=chunks[0].audio_int16_bytes
            audio=np.frombuffer(native,dtype="<i2").astype(np.float32)/32768.0
            resampled=resample_poly(audio,320,441).astype(np.float32)
            need(3200<=len(resampled)<=240000 and np.isfinite(resampled).all(),
                 "COMPLETE_16K_AUDIO_BOUNDS")
            pcm=np.rint(np.clip(resampled,-1.0,32767/32768)*32768).astype("<i2").tobytes()
            wav=out/"wav"/(row["id"]+".wav")
            previous.wav_write(wav,pcm,16000)
            measured=previous.c.inspect_wav(wav)
            need(measured["basic_audio_ok"],"NEW_AUDIO_BASIC_QUALITY")
            doc={**row,**measured,"source_provider":"official-piper-libritts-904",
                 "whole_native_resample":True,"native_pcm_sha256":sha(native),
                 "native_samples":len(native)//2,"actual_native_phones":"".join(phones),
                 "crop_calls":0,"human_gold":False,"training_allowed":False,
                 "positive_duration_pass":row["requested_class"]!=2 or
                    measured["samples"]<=MAX_POSITIVE_SAMPLES}
            finished.append(doc)
            write_once(out/"rows"/(row["id"]+"-COMPLETED.json"),doc)
        write_once(out/"MANIFEST.json",{"schema":SCHEMA,"run":identity,
            "source_plan_sha256":source_check()["plan_sha256"],
            "speakers_by_index":{str(k):v for k,v in choices.items()},
            "denominator":OUTPUT_COUNT,"completed":len(finished),
            "new_tts_calls":started,"new_asr_calls":0,"new_phone_calls":0,
            "training_allowed":False,"human_gold":False,"rows":finished})
        print(json.dumps(readback(out,train_source,train_screen,identity),sort_keys=True))
    except BaseException as exc:
        write_once(out/"STOP.json",{"schema":SCHEMA,"run":identity,
            "started_forwards":started,"complete_original_wavs":len(finished),
            "remaining":OUTPUT_COUNT-len(finished),
            "reason_type":type(exc).__name__,"reason":str(exc)[:420],
            "automatic_retry":False,"no_kws_or_optimizer":True})
        raise

def main():
    ap=argparse.ArgumentParser()
    modes=ap.add_mutually_exclusive_group(required=True)
    for flag in ("--source-check","--guard","--preflight-config","--generate","--readback"):
        modes.add_argument(flag,action="store_true")
    ap.add_argument("--model-config",type=Path)
    ap.add_argument("--train-source",type=Path)
    ap.add_argument("--train-screen",type=Path)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.train_source and a.train_screen,
         "ORIGINAL_TRAIN_SOURCE_AND_SCREEN_REQUIRED")
    if a.preflight_config:
        need(a.model_config is not None,"PINNED_OFFICIAL_CONFIG_REQUIRED")
        print(json.dumps(preflight_config(a.model_config,a.train_source,a.train_screen),sort_keys=True))
        return
    need(a.output is not None,"EXACT_B10_SOURCE_OUTPUT_REQUIRED")
    if a.generate:generate(a.output,a.train_source,a.train_screen)
    else:print(json.dumps(readback(a.output,a.train_source,a.train_screen),sort_keys=True))

if __name__=="__main__":
    main()
