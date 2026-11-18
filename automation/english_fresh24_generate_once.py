"""New independent-source English canonical Qwen voice-clone research campaign.

The source roster and CTC contrast policy were frozen before any of these WAVs.
Stage 1: original Kokoro preset neutral reference, no wake in donor audio.
Stage 2: Qwen Base synthetic voice clone, two target renderings + ordinary +
         explicitly non-training old-/i/ and /z/ diagnostic contrasts.
Neither stage admits labels, trains KWS, or calls recognition models.

No inference without the exact first PUBLIC main push of the pre-registered
source commit. No retry, skipped source, resplit, old voice reuse or Work/Codex.
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
import wave

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("fresh24_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
policy=load("english_fresh24_ctc_admission")
c=load("english_hifi_micro")
SCHEMA="english-fresh24-native-donor-and-clone-v1"
REPOSITORY="jiying2007/kws-data"
BEFORE="9899bf3cd505abcc6c363efd4490cf84fcdfc89d"
REFERENCE_SCHEMA="english-fresh24-neutral-source-v1"
CLONE_SCHEMA="english-fresh24-qwen-voice-clone-v1"
KOKORO_MODEL_SHA="496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
NEUTRAL_MAX=48
CLONE_MAX=168
SAMPLE_RATE=16000
SAMPLE_WIDTH=2

def need(ok,why):
    if not ok:raise ValueError(why)
def source_plan():
    plan=json.loads((HERE/"english_fresh24_protocol.json").read_bytes())
    policy.validate_protocol(plan)
    need(plan["upstream"]["model_sha256"]==KOKORO_MODEL_SHA and
         plan["upstream"]["commit"]==
         "f3ff3571791e39611d31c381e3a41a3af07b4987" and
         plan["budgets"]["max_new_neutral_forward_calls"]==NEUTRAL_MAX and
         plan["budgets"]["max_new_clone_forward_calls"]==CLONE_MAX and
         plan["budgets"]["max_retries"]==0 and
         plan["model_for_cloning"]["model"]=="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
         "EXACT_V1_NO_EXPANSION_BUDGET")
    tree=json.loads((ROOT/"research/english_mero_meh_20261118/KOKORO_VOICES_TREE.json").read_bytes())
    actual={row.get("path"):row for row in tree}
    need(len(tree)==54,"EXACT_UPSTREAM_SOURCE_TREE")
    for row in plan["voice_groups"]:
        native=actual.get(row["asset_path"],{})
        need(native.get("lfs",{}).get("oid")==row["asset_sha256"] and
             native.get("lfs",{}).get("size")==row["asset_bytes"],
             "PRE_RECORDED_OFFICIAL_UPSTREAM_ASSET")
    return plan

def source_check():
    p=source_plan()
    return {"status":"SOURCE_ONLY","new_audio":0,"source_groups":len(p["voice_groups"]),
        "planned_neutral_calls":NEUTRAL_MAX,"planned_clone_calls":CLONE_MAX,
        "source_plan_sha256":policy.sha(policy.canonical(p)),
        "training_allowed":False,"human_gold":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPOSITORY,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"PUBLIC_FIRST_MAIN_ONLY")
    sha=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "REQUIRE_ACTUAL_MAIN_SHA_AND_RUN")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==sha and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPOSITORY and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "ONE_EXACT_SOURCE_SUCCESSOR")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                       cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"PINNED_FIRST_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":sha,"before":BEFORE}

def safe_root(path):
    path=Path(path)
    need(path.is_absolute() and not path.exists() and path.parent.is_dir()
         and not path.parent.is_symlink(),"NEW_UNREPEATED_OUTPUT_DIRECTORY")
    path.mkdir()
    return path

def write(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as file:
        json.dump(data,file,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        file.write("\n")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def write_wav(path,pcm):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.exists() and type(pcm) is bytes and len(pcm)%2==0,
         "NEW_COMPLETE_PCM16")
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return c.inspect_wav(path)

def neutral_requests(plan):
    return [{"id":"fresh24-neutral-"+row["name"],
             "source_group_id":row["source_group_id"],
             "role":row["role"],"preset_id":row["name"],
             "voice_pack_sha256":row["asset_sha256"],
             "voice_pack_bytes":row["asset_bytes"],
             "reference_text":plan["reference_text"],
             "source_kind":"SYNTHETIC_NEUTRAL_REFERENCE_NO_WAKE",
             "human_speaker_identity_verified":False,
             "training_allowed":False,
             "human_gold":False} for row in plan["voice_groups"]]

def readback_refs(root):
    root=Path(root)
    plan=source_plan()
    package=json.loads((root/"MANIFEST.json").read_bytes())
    expected=neutral_requests(plan)
    need(package.get("schema")==REFERENCE_SCHEMA and
         package.get("run")==guard() and
         package.get("denominator")==24 and
         package.get("new_neutral_model_calls")==48 and
         package.get("training_allowed") is False and
         package.get("human_gold") is False and
         package.get("frozen_plan_sha256")==source_check()["source_plan_sha256"]
         and len(package.get("rows",[]))==24,
         "FROZEN_NEUTRAL_SOURCE_MANIFEST")
    seen=set()
    for row,request in zip(package["rows"],expected):
        need(all(row.get(k)==v for k,v in request.items()),
             "PRESPLIT_NEUTRAL_REQUEST_CHANGED")
        original=c.inspect_wav(root/"wav"/(row["id"]+".wav"))
        need(original["basic_audio_ok"] and
             all(original[k]==row.get(k) for k in ("wav_sha256","pcm_sha256","samples")) and
             24000<=row["samples"]<=96000 and
             row["pcm_sha256"] not in seen,
             "NEW_COMPLETE_UNIQUE_REFERENCE_PCM")
        seen.add(row["pcm_sha256"])
    return {"status":"24_NEW_SYNTHETIC_DONORS_AUTHENTICATED",
            "new_neutral_model_calls":48,"source_groups":24,
            "training_allowed":False,"human_gold":False}

def generate_refs(out):
    run=guard()
    plan=source_plan()
    root=safe_root(out)
    write(root/"STARTED.json",{"schema":REFERENCE_SCHEMA,"run":run,
        "denominator":24,"max_calls":48,
        "training_allowed":False,"source_plan_sha256":source_check()["source_plan_sha256"]})
    completed=[];count=0
    try:
        import numpy as np
        import torch
        from scipy.signal import resample_poly
        from huggingface_hub import HfApi,snapshot_download
        from kokoro import KModel
        from importlib.metadata import version
        need(version("kokoro")=="0.9.4" and version("numpy")=="1.26.4",
             "ORIGINAL_PINNED_KOKORO_CPU_RUNTIME")
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.manual_seed(2026111818)
        torch.use_deterministic_algorithms(True)
        model_spec=plan["upstream"]
        remote=HfApi(token=False).model_info(model_spec["model"],
                                              revision=model_spec["commit"])
        need(remote.sha==model_spec["commit"],"EXACT_UPSTREAM_MODEL_REVISION")
        patterns=["config.json","kokoro-v1_0.pth"]+[r["asset_path"] for r in plan["voice_groups"]]
        cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"fresh24-kokoro-cache"
        source=Path(snapshot_download(repo_id=model_spec["model"],
            revision=remote.sha,allow_patterns=patterns,cache_dir=str(cache),
            max_workers=2,token=False))
        weight=source/"kokoro-v1_0.pth"
        need(weight.is_file() and weight.stat().st_size==327212226 and
             digest(weight)==model_spec["model_sha256"],
             "ORIGINAL_UNMODIFIED_KOKORO_WEIGHTS")
        packs={}
        for p in plan["voice_groups"]:
            file=source/p["asset_path"]
            need(file.is_file() and file.stat().st_size==p["asset_bytes"]
                 and digest(file)==p["asset_sha256"],
                 "AUTHENTIC_OFFICIAL_VOICE_PACK:"+p["name"])
            tensor=torch.load(file,map_location="cpu",weights_only=True)
            need(isinstance(tensor,torch.Tensor) and tensor.dtype==torch.float32 and
                 tensor.ndim==3 and tuple(tensor.shape[1:])==(1,256) and
                 torch.isfinite(tensor).all().item(),
                 "EXACT_NATIVE_PACK_TENSOR:"+p["name"])
            packs[p["name"]]=tensor
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        model=KModel(repo_id=model_spec["model"],config=str(source/"config.json"),
                     model=str(weight),disable_complex=False).to("cpu").float().eval()
        need(model.context_length==512 and model.vocab is not None,
             "NATIVE_UNMODIFIED_KOKORO_MODEL")
        write(root/"MODEL_LOCK.json",{"model":model_spec["model"],"revision":remote.sha,
            "model_sha256":model_spec["model_sha256"],
            "voice_pack_sha256_by_group":{x["source_group_id"]:x["asset_sha256"]
                for x in plan["voice_groups"]},
            "model_version":"kokoro==0.9.4","training_allowed":False})
        for i,(row,pin) in enumerate(zip(neutral_requests(plan),plan["voice_groups"])):
            pcm_segments=[]
            for j,raw_phones in enumerate(plan["neutral_reference_phones"]):
                need(1<=len(raw_phones)<=512 and
                     len(raw_phones)-1<len(packs[pin["name"]]),
                     "PREDECLARED_NEUTRAL_PHONEMES_AND_PACK_LENGTH")
                seed=int.from_bytes(hashlib.sha256(
                    (SCHEMA+"/"+row["id"]+"/"+str(j)).encode()).digest()[:4],"big")
                torch.manual_seed(seed)
                # Return entire original native 24k waveform, without trimming.
                with torch.inference_mode():
                    result=model(raw_phones,packs[pin["name"]][len(raw_phones)-1],
                                 speed=1.0,return_output=True)
                need(getattr(result,"audio",None) is not None and
                     getattr(result,"pred_dur",None) is not None,
                     "REAL_NATIVE_KMODEL_OUTPUT_AND_DURATION")
                audio=result.audio
                values=audio.detach().cpu().numpy().reshape(-1).astype(np.float32)
                need(values.ndim==1 and len(values)>1000 and len(values)<=240000
                     and np.isfinite(values).all(),"COMPLETE_NATIVE_NEUTRAL_AUDIO")
                converted=resample_poly(values,2,3).astype(np.float32)
                pcm=np.rint(np.clip(converted,-1,32767/32768)*32768).astype("<i2").tobytes()
                pcm_segments.append(pcm)
                count+=1
            neutral=pcm_segments[0]+b"\x00\x00"*plan["neutral_gap_samples_16k"]+pcm_segments[1]
            original=write_wav(root/"wav"/(row["id"]+".wav"),neutral)
            need(original["basic_audio_ok"] and 24000<=original["samples"]<=96000,
                 "AUTHENTIC_SYNTHETIC_REFERENCE_QUALITY")
            completed.append({**row,**original,
                "component_pcm_sha256":[hashlib.sha256(x).hexdigest() for x in pcm_segments],
                "model_revision":remote.sha,"model_sha256":model_spec["model_sha256"],
                "new_neutral_model_calls":2})
            write(root/"rows"/(row["id"]+".json"),completed[-1])
        write(root/"MANIFEST.json",{"schema":REFERENCE_SCHEMA,"run":run,
            "denominator":24,"completed":len(completed),"new_neutral_model_calls":count,
            "rows":completed,"frozen_plan_sha256":source_check()["source_plan_sha256"],
            "human_gold":False,"training_allowed":False})
        print(json.dumps(readback_refs(root),sort_keys=True))
    except BaseException as exc:
        write(root/"FAILED.json",{"schema":REFERENCE_SCHEMA,"run":run,
            "completed_references":len(completed),"native_forward_returns":count,
            "remaining_possible_calls":48-count,
            "failure":type(exc).__name__+":"+str(exc)[:500],
            "retry":False,"training_allowed":False})
        raise

def clone_requests(refs):
    plan=source_plan()
    need(set(refs)=={x["id"] for x in neutral_requests(plan)},
         "EXACT_NEW_REFERENCE_IDS")
    result=[]
    for row in neutral_requests(plan):
        donor=refs[row["id"]]
        for cell in plan["cells"]:
            result.append({"id":"fresh24-"+row["preset_id"]+"-"+cell["cell_id"],
                "source_group_id":row["source_group_id"],
                "role":row["role"],"source_preset":row["preset_id"],
                "clone_reference_id":row["id"],
                "clone_reference_wav_sha256":donor["wav_sha256"],
                "clone_reference_pcm_sha256":donor["pcm_sha256"],
                "clone_reference_text":row["reference_text"],
                "generation_render_text":cell["generation_render_text"],
                "cell_id":cell["cell_id"],
                "requested_class":cell["requested_class"],
                "requested_target":cell["requested_target"],
                "label_basis":"REQUEST_ONLY_NOT_AN_ACOUSTIC_LABEL",
                "training_allowed":False,"observed_class":None,"human_gold":False})
    need(len(result)==168 and len({r["id"] for r in result})==168 and
         Counter(r["role"] for r in result)==
         Counter({k:v*7 for k,v in policy.FROZEN_SOURCE_ROLES.items()}),
         "PREASSIGNED_168_NO_RESPLIT")
    return result

def readback_clone(root,refs):
    root=Path(root)
    plan=source_plan()
    rows=json.loads((root/"MANIFEST.json").read_bytes())
    expected=clone_requests(refs)
    need(rows.get("schema")==CLONE_SCHEMA and rows.get("run")==guard()
         and rows.get("frozen_plan_sha256")==source_check()["source_plan_sha256"]
         and rows.get("denominator")==168 and rows.get("completed")==168
         and rows.get("new_clone_tts_calls")==168
         and rows.get("training_allowed") is False
         and rows.get("human_gold") is False and
         len(rows.get("rows",[]))==168,"COMPLETE_UNLABELLED_168")
    seen=set(r["pcm_sha256"] for r in refs.values())
    for actual,preg in zip(rows["rows"],expected):
        need(all(actual.get(k)==v for k,v in preg.items()),
             "PRESPLIT_168_CLONE_CELL_DRIFT")
        rec=c.inspect_wav(root/"wav"/(preg["id"]+".wav"))
        need(rec["basic_audio_ok"] and
             all(rec[k]==actual.get(k) for k in ("wav_sha256","pcm_sha256","samples"))
             and rec["pcm_sha256"] not in seen,
             "NEW_COMPLETE_UNCROPPED_CLONE_PCM")
        seen.add(rec["pcm_sha256"])
        need(actual.get("positive_duration_pass")==
             (preg["requested_class"] not in (1,2) or rec["samples"]<=28672),
             "NO_CROP_TO_PASS_DURATION")
    return {"status":"168_NEW_CLONES_AUTHENTICATED_UNLABELLED",
            "source_groups":24,"rows":168,"training_allowed":False}

def generate_clone(ref_dir,out):
    identity=guard()
    donor_manifest=readback_refs(ref_dir)
    refs={r["id"]:r for r in json.loads((Path(ref_dir)/"MANIFEST.json").read_bytes())["rows"]}
    rows=clone_requests(refs)
    root=safe_root(out)
    write(root/"STARTED.json",{"schema":CLONE_SCHEMA,"run":identity,
        "donor_status":donor_manifest["status"],"max_new_tts":168,
        "training_allowed":False})
    saved=[]
    try:
        import numpy as np
        import torch
        from importlib.metadata import version
        from qwen_tts import Qwen3TTSModel
        q=load("english_hifi_qwen_once")
        need(version("qwen-tts")=="0.1.1","PINNED_QWEN_PACKAGE")
        q.resources()
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        model_id,revision=c.CLONE_MODEL
        need(model_id==source_plan()["model_for_cloning"]["model"],
             "EXACT_EXISTING_QWEN_BASE_MODEL")
        assets,lock=q.model_lock(model_id,revision,
            Path(os.environ.get("RUNNER_TEMP","/tmp"))/"fresh24-private-qwen")
        write(root/"MODEL_LOCK.json",{**lock,"original_donor_groups":24,
              "donor_commit":identity["commit"],"training_allowed":False})
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        model=Qwen3TTSModel.from_pretrained(str(assets),device_map="cpu",
                            dtype=torch.float32,attn_implementation="eager")
        prompts={}
        for item in neutral_requests(source_plan()):
            prompts[item["preset_id"]]=model.create_voice_clone_prompt(
                ref_audio=str(Path(ref_dir)/"wav"/(item["id"]+".wav")),
                ref_text=item["reference_text"],x_vector_only_mode=False)
        for n,row in enumerate(rows):
            seed=int.from_bytes(hashlib.sha256((SCHEMA+"/"+row["id"]).encode()).digest()[:4],"big")
            torch.manual_seed(seed);np.random.seed(seed)
            with torch.inference_mode():
                waves,sr=model.generate_voice_clone(text=row["generation_render_text"],
                    language="English",voice_clone_prompt=prompts[row["source_preset"]])
            need(len(waves)==1,"EXACT_ONE_NEW_AUDIO_PER_REQUEST")
            audio=q.encode_whole(waves[0],sr,root,{**row,"seed":seed,
                "generated_index":n,"speaker":"synthetic-"+row["source_preset"],
                "text":row["generation_render_text"],
                "requested_kind":row["cell_id"]})
            audio["positive_duration_pass"]=(row["requested_class"] not in (1,2)
                                              or audio["samples"]<=28672)
            saved.append(audio)
            write(root/"rows"/(row["id"]+".json"),audio)
        write(root/"MANIFEST.json",{"schema":CLONE_SCHEMA,"run":identity,
            "frozen_plan_sha256":source_check()["source_plan_sha256"],
            "denominator":168,"completed":len(saved),"rows":saved,
            "new_clone_tts_calls":len(saved),"new_asr_calls":0,
            "new_phone_calls":0,"human_gold":False,"training_allowed":False})
        print(json.dumps(readback_clone(root,refs),sort_keys=True))
    except BaseException as exc:
        write(root/"FAILED.json",{"schema":CLONE_SCHEMA,"run":identity,
            "completed_clones":len(saved),"remaining":168-len(saved),
            "failure":type(exc).__name__+":"+str(exc)[:500],
            "retry":False,"training_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    modes=("--source-check","--guard","--references","--verify-references",
           "--clone","--verify-clones")
    for mode in modes:ap.add_argument(mode,action="store_true")
    ap.add_argument("--output")
    ap.add_argument("--input")
    args=ap.parse_args()
    need(sum(bool(getattr(args,x[2:].replace("-","_"))) for x in modes)==1,
         "ONE_STAGE_ONLY")
    if args.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if args.guard:
        print(json.dumps(guard(),sort_keys=True));return
    if args.references:
        need(args.output,"NEW_REFS_OUTPUT_REQUIRED")
        generate_refs(args.output);return
    need(args.input,"INPUT_BOUND_TO_FROZEN_SOURCE_REQUIRED")
    if args.verify_references:
        guard();print(json.dumps(readback_refs(args.input),sort_keys=True));return
    if args.clone:
        need(args.output,"NEW_CLONE_OUTPUT_REQUIRED")
        generate_clone(args.input,args.output);return
    need(args.output,"CLONE_AND_REF_SOURCE_REQUIRED")
    guard()
    refs={r["id"]:r for r in json.loads((Path(args.input)/"MANIFEST.json").read_bytes())["rows"]}
    print(json.dumps(readback_clone(args.output,refs),sort_keys=True))

if __name__=="__main__":
    main()
