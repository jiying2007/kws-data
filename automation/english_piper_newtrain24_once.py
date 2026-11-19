"""One-shot 24 NEW Piper LibriTTS voices, frozen brand-IPA source only.

The public first main run may generate at most 168 original 16k WAVs. It
does not train KWS, make acoustic-gold labels or rerun historical speakers.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, re, sys, wave
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("newpiper_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
c=load("english_hifi_micro")
old=load("english_synthetic192_recipe")
REPO="jiying2007/kws-data"
BEFORE="7971b74e3c59fe23a5609c2df6e47939427ea984"
SCHEMA="english-piper-newtrain24-original-source-v1"
SOURCE_COUNT=168
MAX_POSITIVE=28672
def need(cond,msg):
    if not cond:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,
                      separators=(",",":"),allow_nan=False).encode()
def base131(s):
    h=0
    for c_ in str(s).encode("ascii"):h=(h*131+c_)&0xffffffff
    return h

def protocol():
    plan=json.loads((HERE/"english_piper_newtrain24_protocol.json").read_bytes())
    need(plan.get("schema")=="english-piper-newtrain24-prospective-20261119-v1" and
         plan.get("status")=="FROZEN_BEFORE_NEW_TRAINING_SOURCE_AUDIO" and
         plan.get("brand_ipa")=={"Mero":"/heɪ ˈmɛroʊ/","Vosen":"/heɪ ˈvoʊsən/"} and
         plan.get("already_qualified_readonly")=={"train":8,"development":4,
              "calibration":3,"locked-test":3} and
         plan.get("minimum_qualified_groups")=={"train":6,"development":2,
              "calibration":2,"locked-test":2} and
         plan.get("budget")=={"max_new_tts":168,"max_asr_per_model":168,
              "max_phone_per_model":168,"retry_count":0} and
         plan.get("training_allowed") is False and
         plan.get("old_wavs_reused") is False and
         plan.get("new_audio_request_is_label") is False and
         plan.get("frozen_phone_contrast_policy_blob")==
              "0a688325d0d36e83ef6623ac678511df24a92374",
         "ORIGINAL_FROZEN_SOURCE_POLICY")
    model=plan["model"]
    need(model["repo"]=="rhasspy/piper-voices" and
         model["commit"]=="de3dcfdf1912bb49726dc4aa11c26017ce2ac62a" and
         model["file"]=="en/en_US/libritts/high/en_US-libritts-high.onnx" and
         model["file_bytes"]==136673811 and
         model["file_sha256"]=="9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f" and
         model["config"]=="en/en_US/libritts/high/en_US-libritts-high.onnx.json" and
         model["config_bytes"]==20163 and model["config_md5"]==
             "cb1564bb779f3b1f87e2de778adeb669" and
         model["piper_tts_version"]=="1.8.0" and
         model["num_speakers"]==904,
         "EXACT_ORIGINAL_PIPER_904_NATIVE_WEIGHTS")
    groups=plan["groups"]
    need(len(groups)==24 and
         [g["speaker_index"] for g in groups]==sorted(range(424,448),
              key=lambda x:(base131(x),x)) and
         [g["role"] for g in groups]==["train"]*18+["development"]*6 and
         all(g["source_group_id"]==f"piper-libritts-index-{g['speaker_index']}"
               for g in groups) and
         not set(range(424,448)).intersection(set(range(58))|set(range(400,424))) and
         not set(range(424,448)).intersection(i for i,_ in old.NEW_VOICES),
         "NEW_UNEXPOSED_24_SOURCE_INDICES_AND_ROLES")
    cells=plan["cells"]
    need(len(cells)==7 and len({r["id"] for r in cells})==7 and
         [r["class"] for r in cells]==[2,2,1,1,0,None,None] and
         [r["raw_ipa"] for r in cells[:4]]==
            ["heɪ mˈɛɹoʊ","heɪ mˈɛɹoʊ","heɪ vˈoʊsən","heɪ vˈoʊsən"] and
         [r["raw_ipa"] for r in cells[-2:]]==
             ["heɪ mˈiːɹoʊ","heɪ vˈoʊzən"] and
         all(type(r["length_scale"]) in (int,float) and
             0.90<=r["length_scale"]<=1.10 for r in cells),
         "ONLY_CANONICAL_WAKES_WITH_OLD_PHONETIC_CONTROLS")
    return plan

def source_check():
    p=protocol()
    return {"status":"FROZEN_SOURCE_ONLY","voice_groups":24,
            "planned_tts_calls":SOURCE_COUNT,
            "source_plan_sha256":sha(canonical(p)),"training_allowed":False}
def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
         "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
         "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
         "GITHUB_RUN_ATTEMPT":"1"}.items()),"PUBLIC_MAIN_FIRST_ATTEMPT_ONLY")
    commit=env.get("GITHUB_SHA","")
    need(re.fullmatch("[a-f0-9]{40}",commit) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "ACTUAL_SOURCE_RUN_ID")
    if event is None:event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==commit and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "FROZEN_FIRST_MAIN_SUCCESSOR_ONLY")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"EXACT_FIRST_PARENT_NO_REPLAY")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":commit,"before":BEFORE}
def write(path,packet):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(packet,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1048576),b""):h.update(part)
    return h.hexdigest()
def native_speakers(config):
    keys=config.get("speaker_id_map",{})
    need(config.get("num_speakers")==904 and type(keys) is dict and
         len(keys)==904 and set(keys.values())==set(range(904)),
         "ACTUAL_ORIGINAL_904_SPEAKER_CONFIG")
    inv={i:label for label,i in keys.items()}
    blocked=set(old.OLD_LIBRI_IDS)|set(old.OLD_PIPER_IDS)|{
           source for idx,source in old.NEW_VOICES}
    blocked|={int(inv[i][1:]) for i in (*range(58),*range(400,424))}
    chosen={i:inv[i] for i in range(424,448)}
    need(len(set(chosen.values()))==24 and
         all(re.fullmatch("p[0-9]+",p) and int(p[1:]) not in blocked
             for p in chosen.values()),
         "NO_REPLAY_OF_OLD_LIBRI_SOURCE_SPEAKER_IDS")
    return chosen
def planned(speakers):
    result=[]
    for g in protocol()["groups"]:
        sid=speakers[g["speaker_index"]]
        for cell in protocol()["cells"]:
            result.append({
              "id":f"newpiper-{g['speaker_index']}-{cell['id']}",
              "speaker_index":g["speaker_index"],
              "source_speaker_id":sid,
              "source_group_id":"piper-libritts-"+sid,"role":g["role"],
              "cell_id":cell["id"],"requested_text":cell["text"],
              "original_raw_ipa":cell["raw_ipa"],
              "generation_length_scale":cell["length_scale"],
              "requested_class":cell["class"],"observed_class":None,
              "human_gold":False,"training_allowed":False,
              "label_basis":"SOURCE_REQUEST_NOT_ACOUSTIC_TRUTH"})
    need(len(result)==SOURCE_COUNT and
         len({r["id"] for r in result})==SOURCE_COUNT and
         len({r["source_group_id"] for r in result})==24,
         "ORIGINAL_24_SOURCE_AND_168_CELL_ROSTER")
    return result
def wav_write(path,pcm,rate):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.exists() and type(pcm) is bytes and len(pcm)%2==0,
         "NEW_UNTRIMMED_PCM16_WAV_ONLY")
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate)
        f.writeframes(pcm)

def source_readback(root):
    root=Path(root);p=protocol()
    packet=json.loads((root/"MANIFEST.json").read_bytes())
    need(packet.get("schema")==SCHEMA and packet.get("run")==guard() and
         packet.get("source_plan_sha256")==source_check()["source_plan_sha256"] and
         packet.get("completed")==SOURCE_COUNT and
         packet.get("new_tts_calls")==SOURCE_COUNT and
         packet.get("new_asr_calls")==packet.get("new_phone_calls")==0 and
         packet.get("training_allowed") is False and
         packet.get("human_gold") is False and
         len(packet.get("rows",[]))==SOURCE_COUNT,
         "AUTHENTICATED_COMPLETE_168_PIPER_SOURCE")
    speakers={int(k):v for k,v in packet["speakers_by_index"].items()}
    expected=planned(speakers)
    seen=set()
    for row,request in zip(packet["rows"],expected):
        need(all(row.get(k)==v for k,v in request.items()) and
             row.get("source_provider")=="official-piper-libritts-904" and
             row.get("crop_calls")==0 and
             row.get("whole_native_resample") is True,
             "ORIGINAL_NAMED_SOURCE_AND_NO_CROP")
        actual=c.inspect_wav(root/"wav"/(request["id"]+".wav"))
        need(actual["basic_audio_ok"] and
             all(row.get(k)==actual[k] for k in
                 ("wav_sha256","pcm_sha256","samples")) and
             row["pcm_sha256"] not in seen and
             row.get("positive_duration_pass")==
                (row["requested_class"] not in (1,2)
                 or row["samples"]<=28672),
             "EXACT_NEW_PCM_HASH_AND_UNCHANGED_LENGTH")
        seen.add(row["pcm_sha256"])
    return {"status":"NEW_168_PIPER_TRAIN_DEV_WAVS_AUTHENTICATED","groups":24,
            "new_tts_calls":168,"training_allowed":False}

def generate(out):
    run=guard();p=protocol()
    root=Path(out)
    need(root.is_absolute() and not root.exists() and root.parent.is_dir(),
         "NO_EXISTING_OUTPUT_OR_REPLAY")
    root.mkdir()
    write(root/"STARTED.json",{"schema":SCHEMA,"run":run,
         "denominator":168,"source_plan_sha256":source_check()["source_plan_sha256"],
         "max_model_forwards":168,"training_allowed":False})
    finished=[];started=0
    try:
        from importlib.metadata import version
        import numpy as np
        from scipy.signal import resample_poly
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice,SynthesisConfig
        need(version("piper-tts")=="1.8.0" and version("numpy")=="1.26.4",
             "PINNED_ORIGINAL_PIPER_CPU_RUNTIME")
        m=p["model"]
        cache=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"newpiper-cpu-cache"
        model=Path(hf_hub_download(repo_id=m["repo"],filename=m["file"],
             revision=m["commit"],local_dir=str(cache),token=False))
        configfile=Path(hf_hub_download(repo_id=m["repo"],filename=m["config"],
             revision=m["commit"],local_dir=str(cache),token=False))
        need(model.is_file() and model.stat().st_size==m["file_bytes"] and
             digest(model)==m["file_sha256"] and
             configfile.is_file() and configfile.stat().st_size==m["config_bytes"] and
             hashlib.md5(configfile.read_bytes()).hexdigest()==m["config_md5"],
             "EXACT_ORIGINAL_PINNED_PIPER_ONNX_AND_CONFIG")
        config=json.loads(configfile.read_bytes())
        choices=native_speakers(config)
        voice=PiperVoice.load(str(model))
        need(voice.config.num_speakers==904 and
             voice.config.sample_rate==22050,
             "PINNED_NATIVE_PIPER_VOICE_GEOMETRY")
        write(root/"MODEL_LOCK.json",{"model":m,
              "source_plan_sha256":source_check()["source_plan_sha256"],
              "speaker_source_ids":{str(k):v for k,v in choices.items()},
              "human_gold":False,"training_allowed":False})
        for idx,row in enumerate(planned(choices)):
            raw="[["+row["original_raw_ipa"]+"]]"
            phonemes=[a for sentence in voice.phonemize(raw) for a in sentence]
            need(phonemes==list(row["original_raw_ipa"]) and
                 all(x in voice.config.phoneme_id_map for x in phonemes if x!=" "),
                 "NATIVE_DIRECT_IPA_NO_G2P_DRIFT")
            write(root/"rows"/(row["id"]+"-STARTED.json"),
                {"id":row["id"],"forward_attempt":idx+1,
                 "source_speaker_id":row["source_speaker_id"],
                 "source_pcm_is_original":True})
            started+=1
            opts=SynthesisConfig(speaker_id=row["speaker_index"],
                length_scale=row["generation_length_scale"],
                noise_scale=0.333,noise_w_scale=0.333,
                normalize_audio=False,volume=1.0)
            chunks=list(voice.synthesize(raw,opts))
            need(len(chunks)==1 and chunks[0].sample_rate==22050 and
                 chunks[0].sample_width==2 and chunks[0].sample_channels==1,
                 "ONE_FULL_NATIVE_CHUNK")
            native=chunks[0].audio_int16_bytes
            samples=np.frombuffer(native,dtype="<i2").astype(np.float32)/32768.0
            derived=resample_poly(samples,320,441).astype(np.float32)
            need(3200<=len(derived)<=240000 and np.isfinite(derived).all(),
                 "COMPLETE_RESAMPLED_NATIVE_AUDIO")
            pcm=np.rint(np.clip(derived,-1,32767/32768)*32768).astype("<i2").tobytes()
            wav=root/"wav"/(row["id"]+".wav")
            wav_write(wav,pcm,16000)
            obs=c.inspect_wav(wav)
            need(obs["basic_audio_ok"],"ORIGINAL_AUDIO_BASIC_QUALITY")
            new={**row,**obs,"source_provider":"official-piper-libritts-904",
                 "whole_native_resample":True,
                 "native_pcm_sha256":sha(native),
                 "native_samples":len(native)//2,
                 "crop_calls":0,"actual_native_phones":"".join(phonemes),
                 "positive_duration_pass":row["requested_class"] not in (1,2)
                     or obs["samples"]<=MAX_POSITIVE}
            finished.append(new)
            write(root/"rows"/(row["id"]+"-COMPLETED.json"),new)
        write(root/"MANIFEST.json",{"schema":SCHEMA,"run":run,
             "source_plan_sha256":source_check()["source_plan_sha256"],
             "speakers_by_index":{str(k):v for k,v in choices.items()},
             "denominator":SOURCE_COUNT,"completed":len(finished),
             "new_tts_calls":started,"new_asr_calls":0,"new_phone_calls":0,
             "training_allowed":False,"human_gold":False,"rows":finished})
        print(json.dumps(source_readback(root),sort_keys=True))
    except BaseException as exc:
        write(root/"FAILED.json",{"schema":SCHEMA,"run":run,
             "attempted_native_forwards":started,
             "completed_full_wavs":len(finished),"remaining":168-len(finished),
             "error":type(exc).__name__+":"+str(exc)[:350],
             "retry_count":0,"training_allowed":False})
        raise
def main():
    args=argparse.ArgumentParser()
    for x in ("source-check","guard","generate","readback"):
        args.add_argument("--"+x,action="store_true")
    args.add_argument("--output")
    a=args.parse_args()
    need(sum(bool(getattr(a,x)) for x in ("source_check","guard","generate","readback"))==1,
         "ONE_EXECUTION_MODE")
    if a.source_check:print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:print(json.dumps(guard(),sort_keys=True));return
    need(a.output,"NEW_OUTPUT_REQUIRED")
    if a.generate:generate(a.output)
    else:print(json.dumps(source_readback(a.output),sort_keys=True))
if __name__=="__main__":main()
