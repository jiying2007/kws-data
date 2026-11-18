"""One-shot new18 KModel direct-phone English KWS source research.

Only 18 official previously unexposed Kokoro preset assets and 7 frozen
phonetic requests per synthetic style; no prior WAV replay, human identity
claims, model training or self-issued acoustic labels.
"""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import wave

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("direct18_"+name,HERE/(name+".py"))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod
c=load("english_hifi_micro")
old=load("english_fresh24_generate_once")
native=load("english_kokoro48_generate")
SCHEMA="english-fresh18-new-kokoro-direct-phones-v1"
REPO="jiying2007/kws-data"
BEFORE="75ccd81bb3abe2973e01fb2bf1a74411629e9279"
PLAN="english_fresh18_direct_protocol.json"
FROZEN_CTCSHA="0a688325d0d36e83ef6623ac678511df24a92374"
COUNT=126
ROLES={"train":10,"development":4,"locked-test":4}
MAX_POSITIVE_SAMPLES=28672

def need(ok,reason):
    if not ok:raise ValueError(reason)
def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,
                      separators=(",",":"),allow_nan=False).encode()
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def crc131(name):
    result=0
    for c in name.encode("ascii"):
        result=(result*131+c)&0xffffffff
    return result
def plan():
    doc=json.loads((HERE/PLAN).read_bytes())
    need(doc["schema"]=="english-fresh18-new-source-direct-kokoro-v1"
         and doc["status"]=="FROZEN_BEFORE_NUMERICAL_AUDIO"
         and doc["target_ipa"]=={"hey_mero":"/heɪ ˈmɛroʊ/",
                                 "hey_vosen":"/heɪ ˈvoʊsən/"}
         and doc["groups_by_role"]==ROLES and
         doc["minimum_full_cumulative_6_2_2_2"]==
              {"train":6,"development":2,"calibration":2,"locked-test":2}
         and doc["budgets"]=={"native_tts_forwards_max":126,
                              "three_asr_per_model":126,
                              "two_phone_per_model":126,"retry_count":0}
         and doc["fresh_model_training_allowed"] is False
         and doc["human_gold"] is False
         and doc["source_requests_are_not_gold"] is True
         and doc["locked_rows_are_never_train_dev_calibration"] is True,
         "FROZEN_NEW18_BRAND_AND_BUDGET")
    groups=doc["groups"]
    need(len(groups)==18 and len({g["preset_id"] for g in groups})==18
         and len({g["source_group_id"] for g in groups})==18 and
         Counter(g["role"] for g in groups)==Counter(ROLES) and
         not {g["preset_id"] for g in groups}.intersection(doc["original_source_excluded"]),
         "EIGHTEEN_UNEXPOSED_VOICES_PRESPLIT")
    names=sorted((g["preset_id"] for g in groups),key=lambda x:(crc131(x),x))
    need([g["preset_id"] for g in groups]==names and
         [g["role"] for g in groups]==
         (["train"]*10+["development"]*4+["locked-test"]*4),
         "NON_OUTCOME_DETERMINISTIC_SOURCE_SPLIT")
    raw=(ROOT/"research/english_mero_meh_20261118/KOKORO_VOICES_TREE.json").read_bytes()
    tree=json.loads(raw)
    assets={x.get("path"):x for x in tree}
    need(len(tree)==54,"ORIGINAL_PINNED_FIFTY_FOUR_VOICE_TREE")
    for g in groups:
        meta=assets.get(g["asset_path"],{})
        need(g["asset_path"]=="voices/"+g["preset_id"]+".pt" and
             g["source_group_id"]=="kokoro-v1_0-preset:"+g["preset_id"] and
             meta.get("lfs",{}).get("oid")==g["asset_sha256"] and
             meta.get("lfs",{}).get("size")==g["asset_bytes"] and
             g["source_kind"]=="NEW_UNEXPOSED_SYNTHETIC_PRESET_DIRECT_KMODEL_NOT_HUMAN",
             "EXACT_OFFICIAL_UNUSED_VOICE_ASSET")
    model=doc["model"]
    need(model["repo"]=="hexgrad/Kokoro-82M" and
         model["revision"]=="f3ff3571791e39611d31c381e3a41a3af07b4987" and
         model["weights_sha256"]=="496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4" and
         model["weights_bytes"]==327212226 and
         model["KModel_runtime"]=="kokoro==0.9.4" and
         model["native_rate"]==24000 and model["derived_rate"]==16000,
         "PINNED_ORIGINAL_UNMODIFIED_NATIVE_KMODEL")
    cells=doc["cells"]
    need(len(cells)==7 and len({x["cell_id"] for x in cells})==7 and
         Counter(x["requested_class"] for x in cells)==Counter({0:1,1:2,2:2,None:2})
         and all(type(x["speed"]) is float and 0.85<=x["speed"]<=1.15
                 and len(x["raw_phonemes"])<=128
                 and x["raw_phonemes"].startswith("hˈA ")
                 if x["requested_class"] in (1,2,None)
                 else x["raw_phonemes"].startswith("w")
                 for x in cells),
         "EXACT_PHONETIC_INPUTS_CANONICAL_CONTROL_AND_ORDINARY")
    # Check exact frozen role and prior quality identity, never its locked rows.
    prior=doc["immutable_prior_quality_report"]
    need(prior["run"]==44306842000 and prior["source_artifact"]==14429385634 and
         prior["sha256"]=="5a9b226b8fe475331d9f03d3ca4f91d5bb94c5270113f6d0cf3e8b4c36fe8316" and
         prior["qualification_by_role"]=={"train":4,"development":1,
                                         "calibration":3,"locked-test":0} and
         prior["original_stop_immutable"] is True and
         doc["original_immutable_rule_git_blob_sha1"]==FROZEN_CTCSHA,
         "IMMUTABLE_HISTORICAL_SOURCE_STOP")
    # Do not redefine pre-observed CTC decision policy to inflate margins.
    policy=(HERE/"english_fresh24_ctc_admission.py").read_bytes()
    gitblob=hashlib.sha1(b"blob "+str(len(policy)).encode()+b"\x00"+policy).hexdigest()
    need(gitblob==FROZEN_CTCSHA,"PREAUDIO_ADMISSION_POLICY_UNMODIFIED")
    return doc

def source_check():
    v=plan()
    return {"status":"FROZEN_SOURCE_ONLY","new_audio":0,
            "source_groups":len(v["groups"]),"planned_tts_forwards":COUNT,
            "roles":ROLES,"source_plan_sha256":sha(canonical(v)),
            "native_model_calls":0,"training_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"FIRST_PUBLIC_MAIN_ONLY")
    commit=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",commit) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "ACTUAL_PUBLIC_MAIN_COMMIT_AND_RUN")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==commit and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "ONE_ACTUAL_PREDECLARED_MAIN_SUCCESSOR")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"EXACT_FIRST_PARENT_NO_REPLAY")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":commit,"before":BEFORE}

def rows():
    spec=plan();result=[]
    for g in spec["groups"]:
        for request in spec["cells"]:
            ident="fresh18-"+g["preset_id"]+"-"+request["cell_id"]
            result.append({"id":ident,"source_group_id":g["source_group_id"],
                "source_preset":g["preset_id"],"role":g["role"],
                "voice_pack_sha256":g["asset_sha256"],
                "voice_pack_bytes":g["asset_bytes"],
                "cell_id":request["cell_id"],
                "generation_render_text":request["render_text"],
                "raw_phonemes":request["raw_phonemes"],
                "speed":request["speed"],
                "requested_class":request["requested_class"],
                "requested_target":request["target"],
                "observed_class":None,"label_basis":"NATIVE_REQUEST_NOT_ACOUSTIC_GOLD",
                "training_allowed":False,"human_gold":False})
    need(len(result)==COUNT and len({x["id"] for x in result})==COUNT,
         "FULL_PRESPLIT_126_REQUESTS")
    return result

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda:f.read(1024*1024),b""):h.update(piece)
    return h.hexdigest()

def readback(root):
    root=Path(root)
    manifest=json.loads((root/"MANIFEST.json").read_bytes())
    expected=rows()
    need(manifest.get("schema")==SCHEMA and
         manifest.get("run")==guard() and
         manifest.get("source_plan_sha256")==source_check()["source_plan_sha256"]
         and manifest.get("denominator")==126 and manifest.get("completed")==126
         and manifest.get("native_tts_calls")==126 and
         manifest.get("new_asr_calls")==0 and
         manifest.get("new_phone_calls")==0 and
         manifest.get("training_allowed") is False
         and manifest.get("human_gold") is False and
         len(manifest.get("rows",[]))==126,
         "ACTUAL_NEW126_NATIVE_MODEL_COMPLETION")
    seen=set()
    for a,request in zip(manifest["rows"],expected):
        need(all(a.get(k)==v for k,v in request.items()) and
             a.get("source_provider")=="official-kokoro-v1-direct-phonemes"
             and a["training_allowed"] is False and a["observed_class"] is None,
             "NATIVE_SOURCE_OR_REQUEST_CHANGED")
        wav=c.inspect_wav(root/"wav"/(request["id"]+".wav"))
        need(wav["basic_audio_ok"] and
             all(a[k]==wav[k] for k in ("wav_sha256","pcm_sha256","samples")) and
             wav["pcm_sha256"] not in seen and
             a.get("positive_duration_pass")==
                  (a["requested_class"] not in (1,2) or wav["samples"]<=MAX_POSITIVE_SAMPLES),
             "NATIVE_WAVE_HASH_AND_UNCROPPED_LENGTH")
        seen.add(wav["pcm_sha256"])
    return {"status":"126_NEW_NATIVE_WAVS_AUTHENTICATED",
            "source_groups":18,"new_native_forwards":126,
            "training_allowed":False}

def generate(out):
    run=guard();spec=plan()
    root=Path(out)
    need(root.is_absolute() and not root.exists() and root.parent.is_dir(),
         "FRESH_OUTPUT_NEVER_OVERWRITE")
    root.mkdir()
    write(root/"STARTED.json",{"schema":SCHEMA,"run":run,
          "denominator":COUNT,"max_calls":126,
          "source_plan_sha256":source_check()["source_plan_sha256"],
          "training_allowed":False})
    saved=[];count=0
    try:
        import numpy as np
        import torch
        from scipy.signal import resample_poly
        from huggingface_hub import HfApi,snapshot_download
        from kokoro import KModel
        from importlib.metadata import version
        need(version("kokoro")=="0.9.4" and version("numpy")=="1.26.4",
             "PINNED_KOKORO_NUMERICAL_RUNTIME")
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        model_spec=spec["model"]
        actual=HfApi(token=False).model_info(model_spec["repo"],
                  revision=model_spec["revision"])
        need(actual.sha==model_spec["revision"],"FROZEN_MODEL_COMMIT")
        patterns=["config.json",model_spec["weights_path"]]+[
                  g["asset_path"] for g in spec["groups"]]
        folder=Path(snapshot_download(repo_id=model_spec["repo"],
               revision=actual.sha,allow_patterns=patterns,
               cache_dir=str(Path(os.environ.get("RUNNER_TEMP","/tmp"))/"fresh18-private-assets"),
               max_workers=2,token=False))
        weight=folder/model_spec["weights_path"]
        need(weight.is_file() and
             weight.stat().st_size==model_spec["weights_bytes"] and
             digest(weight)==model_spec["weights_sha256"],
             "EXACT_OFFICIAL_UNMODIFIED_KMODEL_WEIGHTS")
        packs={}
        for g in spec["groups"]:
            file=folder/g["asset_path"]
            need(file.is_file() and file.stat().st_size==g["asset_bytes"]
                 and digest(file)==g["asset_sha256"],
                 "PINNED_NEW_SYNTHETIC_SOURCE_PACK:"+g["preset_id"])
            audio_pack=torch.load(file,map_location="cpu",weights_only=True)
            need(isinstance(audio_pack,torch.Tensor) and
                 audio_pack.dtype==torch.float32 and
                 audio_pack.ndim==3 and tuple(audio_pack.shape[1:])==(1,256)
                 and torch.isfinite(audio_pack).all().item(),
                 "NATIVE_NEW_VOICE_PACK_SHAPE")
            packs[g["preset_id"]]=audio_pack
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        model=KModel(repo_id=model_spec["repo"],config=str(folder/"config.json"),
                     model=str(weight),disable_complex=False).to("cpu").float().eval()
        need(model.context_length==512 and model.vocab,
             "ORIGINAL_EXACT_NATIVE_KMODEL_VOCAB")
        for item in rows():
            native.phoneme_ids(item["raw_phonemes"],model.vocab,512)
            need(len(item["raw_phonemes"])-1<len(packs[item["source_preset"]]),
                 "SOURCE_VOICE_PACK_PHONE_INDEX")
        write(root/"MODEL_LOCK.json",{
            "model":model_spec,"source_plan_sha256":source_check()["source_plan_sha256"],
            "source_voice_packs":{g["source_group_id"]:g["asset_sha256"]
                                for g in spec["groups"]},
            "changed_model_weights":False,"training_allowed":False})
        for index,request in enumerate(rows()):
            seed=int.from_bytes(hashlib.sha256(
                 (SCHEMA+"/"+request["id"]).encode()).digest()[:4],"big")
            torch.manual_seed(seed);np.random.seed(seed)
            with torch.inference_mode():
                out=model(request["raw_phonemes"],
                         packs[request["source_preset"]][len(request["raw_phonemes"])-1],
                         speed=request["speed"],return_output=True)
            need(getattr(out,"audio",None) is not None and
                 getattr(out,"pred_dur",None) is not None,
                 "NATIVE_KMODEL_COMPLETE_OUTPUT")
            native_f32=out.audio.detach().cpu().numpy().reshape(-1).astype(np.float32)
            need(0<len(native_f32)<24000*15 and
                 np.isfinite(native_f32).all(),
                 "FINITE_FULL_NATIVE_24K_PCM")
            audio_16k=resample_poly(native_f32,2,3).astype(np.float32)
            need(len(audio_16k)>=3200 and
                 len(audio_16k)<=240000 and np.isfinite(audio_16k).all(),
                 "WHOLE_RESAMPLED_PCM_BOUNDED")
            pcm=np.rint(np.clip(audio_16k,-1,32767/32768)*32768).astype("<i2").tobytes()
            wav=root/"wav"/(request["id"]+".wav")
            wav.parent.mkdir(exist_ok=True)
            with wave.open(str(wav),"xb") as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
                f.writeframes(pcm)
            observed=c.inspect_wav(wav)
            need(observed["basic_audio_ok"],"NEW_NATIVE_AUDIO_QUALITY_NO_SUBSTITUTE")
            result={**request,**observed,"source_provider":
                  "official-kokoro-v1-direct-phonemes",
                  "source_model_sha256":model_spec["weights_sha256"],
                  "native_sample_rate":24000,"sample_rate":16000,
                  "crop_calls":0,"resample_calls":1,
                  "positive_duration_pass":request["requested_class"] not in (1,2)
                                             or observed["samples"]<=MAX_POSITIVE_SAMPLES,
                  "seed":seed,"original_index":index}
            saved.append(result)
            write(root/"rows"/(request["id"]+".json"),result)
            count+=1
        write(root/"MANIFEST.json",{"schema":SCHEMA,"run":run,
            "denominator":126,"completed":len(saved),
            "native_tts_calls":count,"source_plan_sha256":
            source_check()["source_plan_sha256"],"rows":saved,
            "new_asr_calls":0,"new_phone_calls":0,
            "human_gold":False,"training_allowed":False})
        print(json.dumps(readback(root),sort_keys=True))
    except BaseException as exc:
        write(root/"FAILED.json",{"schema":SCHEMA,"run":run,
            "completed_full_wavs":len(saved),"returned_native_forwards":count,
            "remaining":126-count,"error":type(exc).__name__+":"+str(exc)[:500],
            "retry":False,"training_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    modes=("--source-check","--guard","--generate","--readback")
    for x in modes:ap.add_argument(x,action="store_true")
    ap.add_argument("--output")
    a=ap.parse_args()
    need(sum(bool(getattr(a,x[2:].replace("-","_"))) for x in modes)==1,
         "ONE_EXACT_SOURCE_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.output,"EXACT_FRESH_SOURCE_OUTPUT_REQUIRED")
    if a.generate:generate(a.output)
    else:print(json.dumps(readback(a.output),sort_keys=True))

if __name__=="__main__":
    main()
