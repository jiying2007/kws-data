"""One exact new locked-source backfill after V5 Hey Mero 11/2/2/1 STOP.

Freeze original Piper indices 472..479 BEFORE ANY new TTS/ASR/KWS output.
Generate exactly 32 genuinely new original full-WAV clips from 8 distinct
previously unseen LibriTTS human speaker identities. Do not regenerate 96,
64, 63 or 168 historical WAV; no old locked audio relabel, no KWS/optimizer,
no nearphone automatically treated as ordinary, no repeat attempt.
Independent native source QA happens only after this source stage.
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
import subprocess
import wave

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
_spec=importlib.util.spec_from_file_location(
    "en_v5_piper_proven_source",HERE/"english_piper_newtrain24_once.py")
previous=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
PLAN=HERE/"english_v5_mero_locked8_source_plan.json"
REPO="jiying2007/kws-data"
MAIN_BEFORE="63f1e5be812c27010482b4faaef3df4524710a52"
SCHEMA="english-v5-mero-locked8-independent-source-once-v1"
OUTPUT_COUNT=32
MAX_POSITIVE_SAMPLES=28672
GROUP_COUNTS={"locked-test":8}


def need(v,code):
    if not v:
        raise ValueError(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canon(v):
    return json.dumps(v,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False).encode()


def write_once(p,doc):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:
        json.dump(doc,f,sort_keys=True,ensure_ascii=False,
                  indent=2,allow_nan=False)
        f.write("\n")


def plan():
    obj=json.loads(PLAN.read_bytes())
    model=obj["model"]
    old=previous.old.MODEL
    groups=obj["source_groups"]
    cells=obj["cells"]
    need(obj["schema"]==
         "english-v5-mero-new-locked8-independent-piper-source-v1"
         and obj["status"]=="FROZEN_BEFORE_ANY_NEW_WAV"
         and obj["target"]=={
            "keyword":"Hey Mero","ipa":"/heɪ ˈmɛroʊ/",
            "active_english_keyword_count":1}
         and obj["source_count"]==OUTPUT_COUNT
         and [r["speaker_index"] for r in groups]==list(range(472,480))
         and Counter(r["role"] for r in groups)==GROUP_COUNTS
         and len({r["source_group_id"] for r in groups})==8
         and all(r["source_group_id"]==
                 "piper-libritts-index-"+str(r["speaker_index"])
                 for r in groups)
         and [r["id"] for r in cells]==
         ["mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge"]
         and [r["requested_class"] for r in cells]==[2,2,0,None]
         and [r["length_scale"] for r in cells]==[0.92,1.08,1,1]
         and [r["raw_ipa"] for r in cells]==[
             "heɪ mˈɛɹoʊ","heɪ mˈɛɹoʊ","wˈʌt tˈaɪm ɪz ɪt",
             "heɪ mˈiːɹoʊ"]
         and obj["minimum_groups_eligible"]==
             {"train":6,"development":2,"calibration":2,"locked-test":2}
         and obj["budget"]=={"max_new_piper_tts_forwards":32,
             "max_new_asr_forwards_at_source_stage":0,
             "max_new_kws_train_updates":0,"retry_count":0}
         and obj["source_guards"]["nearphone_is_unlabeled_challenge"] is True
         and obj["source_guards"]["source_ancestry_role_frozen"] is True
         and obj["prior_frozen_gate"]=={
            "run_id":44487146445,
            "status":"STOP_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_COVERAGE",
            "qualified_groups":{"train":11,"development":2,
                                "calibration":2,"locked-test":1},
            "additional_distinct_locked_groups_needed":1,
            "old_locked_pcm_relabel_allowed":False},
         "FROZEN_V5_MERO_8_NEW_LOCKED_SPEAKERS_32_SOURCE_DESIGN")
    need(model["repo"]==old["repository"]
         and model["commit"]==old["revision"]
         and model["file"]==old["model_path"]
         and model["file_bytes"]==old["model_bytes"]
         and model["file_sha256"]==old["model_sha256"]
         and model["config"]==old["config_path"]
         and model["config_bytes"]==old["config_bytes"]
         and model["config_md5"]==old["config_md5"]
         and model["piper_tts_version"]==old["piper_version"]
         and model["native_rate"]==22050
         and model["derived_rate"]==16000
         and model["num_speakers"]==904,
         "EXACT_EXISTING_PROVEN_OFFICIAL_PIPER_MODEL_IDENTITY")
    return obj


def source_check():
    p=plan()
    return {"schema":SCHEMA,
            "status":"SOURCE_ONLY_NO_AUDIO_GENERATED",
            "only_english_keyword":p["target"]["keyword"],
            "plan_sha256":sha(canon(p)),
            "new_speaker_indices":[472,479],
            "source_roles":GROUP_COUNTS,
            "new_speech_wav_budget":OUTPUT_COUNT,
            "nearphone_is_eligible_ordinary":False,
            "independent_human_gold":False,
            "new_optimizer_updates":0,
            "new_asr_phone_calls":0,
            "new_private_repo":False,
            "ssc305":"NOT_RUN","product_promotion":False}


def guard(env=None,event=None,parent=None):
    env=os.environ if env is None else env
    require={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in require.items()),
         "ONLY_FIRST_PUBLIC_MAIN_PUSH_NUMERIC")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head)
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "EXACT_PROTECTED_MAIN_EVENT_ID")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==MAIN_BEFORE
         and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False
         and event.get("deleted") is False,
         "FRESH_SOURCE_MAIN_PARENT_ONCE_ONLY")
    if parent is None:
        parent=subprocess.check_output(
            ["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==MAIN_BEFORE,"NO_OUT_OF_ORDER_SOURCE_GENERATION")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":head,
            "before":MAIN_BEFORE,"attempt":1}


def native_speakers(config):
    mapping=config.get("speaker_id_map",{})
    need(config.get("num_speakers")==904 and
         isinstance(mapping,dict) and len(mapping)==904 and
         set(mapping.values())==set(range(904)),
         "EXACT_904_SPEAKER_SOURCE_MAP")
    inv={i:label for label,i in mapping.items()}
    used_index=set(range(58))|set(range(400,472))
    used_index|={i for i,_ in previous.old.NEW_VOICES}
    frozen_indices=set(range(472,480))
    need(not frozen_indices.intersection(used_index),
         "REPEATED_HISTORICAL_PIPER_SPEAKER_INDEX")
    used_numeric=set(previous.old.OLD_LIBRI_IDS)|set(previous.old.OLD_PIPER_IDS)
    used_numeric|={speaker for _,speaker in previous.old.NEW_VOICES}
    used_numeric|={int(inv[i][1:]) for i in used_index
                   if re.fullmatch("p[0-9]+",inv[i])}
    chosen={i:inv[i] for i in sorted(frozen_indices)}
    need(len(set(chosen.values()))==8
         and all(re.fullmatch("p[0-9]+",name)
                 and int(name[1:]) not in used_numeric
                 for name in chosen.values()),
         "DUPLICATE_PRIOR_HUMAN_ORIGINAL_SPEAKER_ID")
    return chosen


def preflight_config(config_path):
    """Byte-authenticate the 904-ID map BEFORE any source model forwards."""
    p=plan()
    config_path=Path(config_path)
    expected=p["model"]
    need(config_path.is_absolute() and config_path.is_file()
         and not config_path.is_symlink()
         and config_path.stat().st_size==expected["config_bytes"]
         and hashlib.md5(config_path.read_bytes()).hexdigest()==
             expected["config_md5"],
         "PINNED_OFFICIAL_SPEAKER_CONFIG_BYTES")
    config=json.loads(config_path.read_bytes())
    chosen=native_speakers(config)
    items=planned(chosen)
    need(len(items)==OUTPUT_COUNT,"EXACT_PROSPECTIVE_SOURCE_CELLS")
    return {"schema":SCHEMA,"status":"8_NEW_LOCKED_SPEAKER_IDS_AUTHENTICATED",
            "index_range":[472,479],
            "source_speaker_ids":{str(k):v for k,v in chosen.items()},
            "distinct_original_speaker_ancestry":True,
            "predetermined_roles":GROUP_COUNTS,
            "new_audio_generated":0,"model_forwards":0,
            "training_allowed":False,"new_locked_audio_opened":False}


def planned(choices):
    p=plan()
    need(set(choices)==set(range(472,480)),"FROZEN_FULL_8_NEW_LOCKED_SPEAKER_MAP")
    rows=[]
    for g in p["source_groups"]:
        idx=g["speaker_index"];speaker=choices[idx]
        for cell in p["cells"]:
            rows.append({"id":f"mero8locked-{idx}-{cell['id']}",
              "speaker_index":idx,"source_speaker_id":speaker,
              "source_group_id":"piper-libritts-"+speaker,
              "role":g["role"],"cell_id":cell["id"],
              "requested_text":cell["text"],
              "requested_class":cell["requested_class"],
              "original_raw_ipa":cell["raw_ipa"],
              "generation_length_scale":cell["length_scale"],
              "observed_class":None,"human_gold":False,
              "training_allowed":False,
              "label_basis":"SOURCE_REQUEST_NOT_ACOUSTIC_TRUTH"})
    need(len(rows)==OUTPUT_COUNT
         and len({r["id"] for r in rows})==OUTPUT_COUNT
         and len({r["source_group_id"] for r in rows})==8,
         "DISTINCT_32_NEW_LOCKED_ORIGINAL_SOURCE_CELLS")
    return rows


def readback(root,identity=None):
    root=Path(root)
    p=plan()
    data=json.loads((root/"MANIFEST.json").read_bytes())
    lock=json.loads((root/"MODEL_LOCK.json").read_bytes())
    if identity is None:
        identity=guard()
    need(data.get("schema")==SCHEMA and data.get("run")==identity
         and data.get("source_plan_sha256")==source_check()["plan_sha256"]
         and data.get("denominator")==OUTPUT_COUNT
         and data.get("completed")==OUTPUT_COUNT
         and len(data.get("rows",[]))==OUTPUT_COUNT
         and data.get("new_tts_calls")==OUTPUT_COUNT
         and data.get("new_asr_calls")==0
         and data.get("new_phone_calls")==0
         and data.get("human_gold") is False
         and data.get("training_allowed") is False
         and lock.get("model")==p["model"]
         and lock.get("source_plan_sha256")==source_check()["plan_sha256"]
         and lock.get("source_speaker_ids")==data.get("speakers_by_index"),
         "GROUNDED_REAL_32_NEW_LOCKED_PIPER_SOURCE_RECEIPT")
    choices={int(k):v for k,v in data["speakers_by_index"].items()}
    expected=planned(choices)
    duplicate=set()
    for row,original in zip(data["rows"],expected):
        need(all(row.get(k)==v for k,v in original.items()) and
             row.get("source_provider")=="official-piper-libritts-904" and
             row.get("whole_native_resample") is True and
             row.get("crop_calls")==0 and
             row.get("nearphone_is_background") is False,
             "NEW_SOURCE_UNCHANGED_REQUEST_AND_ROLES")
        wav=root/"wav"/(row["id"]+".wav")
        info=previous.c.inspect_wav(wav)
        need(info["basic_audio_ok"] and
             all(row.get(k)==info[k] for k in
                 ("wav_sha256","pcm_sha256","samples")) and
             row["pcm_sha256"] not in duplicate and
             row["positive_duration_pass"]==
                 (row["requested_class"]!=2 or
                  row["samples"]<=MAX_POSITIVE_SAMPLES),
             "ACTUAL_WAV_AND_COMPLETE_UNTRIMMED_PCM_SHA")
        duplicate.add(row["pcm_sha256"])
    return {"status":"32_REAL_NEW_LOCKED_PIPER_AUDIO_VALIDATED",
            "source_groups":8,"source_roles":GROUP_COUNTS,
            "original_clips":OUTPUT_COUNT,
            "positive_qa_not_yet_admitted":True,
            "training_allowed":False,"independent_far":"NOT_MEASURED",
            "human_gold":False,"ssc305":"NOT_RUN"}


def generate(output):
    identity=guard()
    p=plan()
    out=Path(output)
    need(out.is_absolute() and out.parent.is_dir()
         and not out.exists(),"NEW_UNCHANGED_SOURCE_DIRECTORY")
    out.mkdir()
    write_once(out/"STARTED.json",{"schema":SCHEMA,"run":identity,
           "original_clip_limit":OUTPUT_COUNT,
           "source_plan_sha256":source_check()["plan_sha256"],
           "max_model_forwards":OUTPUT_COUNT,"retry_count":0})
    finished=[]
    started=0
    try:
        from importlib.metadata import version
        import numpy as np
        from scipy.signal import resample_poly
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice,SynthesisConfig
        need(version("piper-tts")=="1.8.0"
             and version("numpy")=="1.26.4",
             "EXACT_CPU_SYNTHESIS_ENGINE")
        model=p["model"]
        cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"mono8-locked-piper"
        weights=Path(hf_hub_download(repo_id=model["repo"],
                filename=model["file"],revision=model["commit"],
                local_dir=str(cache),token=False))
        cfg=Path(hf_hub_download(repo_id=model["repo"],
                filename=model["config"],revision=model["commit"],
                local_dir=str(cache),token=False))
        need(weights.is_file() and
             weights.stat().st_size==model["file_bytes"]
             and previous.digest(weights)==model["file_sha256"]
             and cfg.is_file() and cfg.stat().st_size==model["config_bytes"]
             and hashlib.md5(cfg.read_bytes()).hexdigest()==model["config_md5"],
             "EXACT_PIPER_ONNX_AND_904_MAP_BYTES")
        config=json.loads(cfg.read_bytes())
        choices=native_speakers(config)
        voice=PiperVoice.load(str(weights))
        need(voice.config.sample_rate==22050 and
             voice.config.num_speakers==904,
             "EXACT_PINNED_EN_US_LIBRITTS_NATIVE_VOICE")
        write_once(out/"MODEL_LOCK.json",{"model":model,
             "source_plan_sha256":source_check()["plan_sha256"],
             "source_speaker_ids":{str(k):v for k,v in choices.items()},
             "training_allowed":False})
        rows=planned(choices)
        for row in rows:
            phrase="[["+row["original_raw_ipa"]+"]]"
            phones=[s for seq in voice.phonemize(phrase) for s in seq]
            need(phones==list(row["original_raw_ipa"])
                 and all(s in voice.config.phoneme_id_map for s in phones
                         if s!=" "),
                 "NATIVE_DIRECT_IPA_COMPLETE_SOURCE")
            started+=1
            write_once(out/"rows"/(row["id"]+"-STARTED.json"),
                 {"id":row["id"],"attempt":1,
                  "source_group_id":row["source_group_id"],
                  "position":started})
            setting=SynthesisConfig(speaker_id=row["speaker_index"],
                length_scale=row["generation_length_scale"],
                noise_scale=0.333,noise_w_scale=0.333,
                normalize_audio=False,volume=1.0)
            chunks=list(voice.synthesize(phrase,setting))
            need(len(chunks)==1 and chunks[0].sample_rate==22050
                 and chunks[0].sample_channels==1
                 and chunks[0].sample_width==2,
                 "WHOLE_NATIVE_ONE_CHUNK")
            native=chunks[0].audio_int16_bytes
            x=np.frombuffer(native,dtype="<i2").astype(np.float32)/32768.0
            y=resample_poly(x,320,441).astype(np.float32)
            need(3200<=len(y)<=240000 and np.isfinite(y).all(),
                 "NEW_COMPLETE_UNCROPPED_16K_AUDIO")
            pcm=np.rint(np.clip(y,-1,32767/32768)*32768).astype("<i2").tobytes()
            wav=out/"wav"/(row["id"]+".wav")
            previous.wav_write(wav,pcm,16000)
            obs=previous.c.inspect_wav(wav)
            need(obs["basic_audio_ok"],"GENERATED_PCM_BASIC_QUALITY")
            doc={**row,**obs,
                 "source_provider":"official-piper-libritts-904",
                 "whole_native_resample":True,
                 "native_pcm_sha256":sha(native),
                 "native_samples":len(native)//2,
                 "actual_native_phones":"".join(phones),
                 "crop_calls":0,
                 "nearphone_is_background":False,
                 "positive_duration_pass":
                     row["requested_class"]!=2 or
                     obs["samples"]<=MAX_POSITIVE_SAMPLES}
            finished.append(doc)
            write_once(out/"rows"/(row["id"]+"-COMPLETED.json"),doc)
        write_once(out/"MANIFEST.json",{"schema":SCHEMA,"run":identity,
            "source_plan_sha256":source_check()["plan_sha256"],
            "speakers_by_index":{str(k):v for k,v in choices.items()},
            "denominator":OUTPUT_COUNT,
            "completed":len(finished),
            "new_tts_calls":started,
            "new_asr_calls":0,"new_phone_calls":0,
            "training_allowed":False,"human_gold":False,
            "rows":finished})
        print(json.dumps(readback(out,identity),sort_keys=True))
    except BaseException as error:
        write_once(out/"STOP.json",{"schema":SCHEMA,"run":identity,
            "started_forwards":started,
            "complete_original_wavs":len(finished),
            "remaining":OUTPUT_COUNT-len(finished),
            "reason_type":type(error).__name__,
            "reason":str(error)[:420],"automatic_retry":False,
            "no_test_or_calibration_model_eval":True})
        raise


def main():
    parser=argparse.ArgumentParser()
    for k in ("--source-check","--guard","--generate","--readback","--preflight-config"):
        parser.add_argument(k,action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--model-config",type=Path)
    a=parser.parse_args()
    need(sum((a.source_check,a.guard,a.generate,a.readback,a.preflight_config))==1,
         "EXACTLY_ONE_REAL_SOURCE_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True))
    elif a.guard:
        print(json.dumps(guard(),sort_keys=True))
    elif a.preflight_config:
        need(a.model_config is not None,"EXACT_PINNED_CONFIG_REQUIRED")
        print(json.dumps(preflight_config(a.model_config),sort_keys=True))
    else:
        need(a.output,"EXACT_OUTPUT_PATH_REQUIRED")
        if a.generate:
            generate(a.output)
        else:
            print(json.dumps(readback(a.output),sort_keys=True))


if __name__=="__main__":
    main()
