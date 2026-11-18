"""Native English CMU-phoneme-controlled MeloTTS pilot; research only.

Different from the Qwen3 spelling probes: the CMU frontend dictionary is
frozen BEFORE any audio generation. The unmodified upstream VITS acoustic
weights then receive explicit /m ɛ r oʊ/ and /v oʊ s ə n/ phone sequences.
No old Meero positive, human recording, inferred label, or training updates.
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
    spec=importlib.util.spec_from_file_location("melo_contract_"+name,HERE/(name+".py"))
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
c=load("english_hifi_micro")
SCHEMA="english-melo-cmu-native-phone-25-v1"
REPOSITORY="jiying2007/kws-data"
BEFORE="44fecc89d0c1207dee480161d2ef7c7175ac4ac2"
UPSTREAM="myshell-ai/MeloTTS"
UPSTREAM_SHA="209145371cff8fc3bd60d7be902ea69cbdb7965a"
MODEL=("myshell-ai/MeloTTS-English",
       "17f1313104f4ad12d09bffc2fcbe2e72227ac161")
CHECKPOINT_SHA="acd278040eaf9536908e2b965273df5a731c44d8f0da66cc5fed7972772ed23c"
BERT=("google-bert/bert-base-uncased",
      "86b5e0934494bd15c9632b12f734a8a67f723594")
SPEAKERS=("EN-US","EN-BR","EN_INDIA","EN-AU","EN-Default")
# Frozen, model-facing phones: explicitly not inferred from either target text
# or an ASR transcript.
CMU={
    "MERO":[["M","EH1"],["R","OW0"]],
    "VOSEN":[["V","OW1"],["S","AH0","N"]],
    "MEERO":[["M","IY1"],["R","OW0"]],
    "VOZEN":[["V","OW1"],["Z","AH0","N"]]}
CELLS=(
    ("mero","Hey Mero.",2,("m","eh","r","ow")),
    ("vosen","Hey Vosen.",1,("v","ow","s","ah","n")),
    ("meero-control","Hey Meero.",None,("m","iy","r","ow")),
    ("vozen-control","Hey Vozen.",None,("v","ow","z","ah","n")),
    ("ordinary","What time is it?",0,None))
MAX_SYNTH=25

def need(x,code):
    if not x:raise ValueError(code)
def source_check():
    need(len(SPEAKERS)==5 and len(CELLS)==5 and
         len(SPEAKERS)*len(CELLS)==MAX_SYNTH and
         len(CMU)==4 and set(CMU)=={"MERO","VOSEN","MEERO","VOZEN"},
         "FIVE_STYLES_BY_FIVE_CELLS")
    need(CMU["MERO"]==[["M","EH1"],["R","OW0"]] and
         CMU["VOSEN"]==[["V","OW1"],["S","AH0","N"]] and
         CMU["MEERO"]==[["M","IY1"],["R","OW0"]] and
         CMU["VOZEN"]==[["V","OW1"],["Z","AH0","N"]],
         "EXPLICIT_BRAND_AND_CONTRAST_PHONES")
    need(len({x[0] for x in CELLS})==5 and
         len({x[1] for x in CELLS})==5 and
         len(UPSTREAM_SHA)==40 and len(MODEL[1])==40 and len(BERT[1])==40,
         "IMMUTABLE_SOURCE_VERSIONS")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY",
            "native_tts_calls":0,"planned_new_tts_calls":MAX_SYNTH,
            "old_meero_positive":False,"human_gold":False,
            "training_allowed":False,"source_groups_assumed_independent":False}

def guard(env=None,event=None,parent=None):
    env=os.environ if env is None else env
    required={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPOSITORY,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),"FIRST_PUBLIC_MAIN_ONLY")
    sha=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "EXACT_GITHUB_EXECUTION")
    if event is None:event=json.loads(c.read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before")==BEFORE and event.get("after")==sha and
         event.get("repository",{}).get("full_name")==REPOSITORY and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "EXACT_MAIN_SUCCESSOR")
    if parent is None:
        import subprocess
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                       cwd=c.ROOT,text=True).strip()
    need(parent==BEFORE,"EXACT_FIRST_PARENT")
    return {"commit":sha,"run_id":env["GITHUB_RUN_ID"],
            "source_parent":BEFORE}

def planned():
    source_check()
    rows=[]
    for voice in SPEAKERS:
        for key,text,target,critical in CELLS:
            rows.append({"id":"melocmu-"+voice.lower().replace("_","-")+"-"+key,
                         "style_id":"melo1:"+voice,
                         "role":"research-train-only",
                         "speaker":voice,"cell_id":key,"render_text":text,
                         "requested_class":target,
                         "declared_critical_phones":list(critical) if critical else None,
                         "phone_input_controlled":True,
                         "requested_text_is_ground_truth":False,
                         "human_gold":False,"training_allowed":False})
    need(len(rows)==MAX_SYNTH and len({x["id"] for x in rows})==25,
         "EXACT_TWENTY_FIVE_UNIQUE_REQUESTS")
    return rows

def phones_are_locked(english):
    # Must test actual upstream G2P BEFORE model initialization, not just the
    # Python dictionary contents.
    english.eng_dict.update(CMU)
    evidence={}
    for term,expected in (
        ("Mero",("m","eh","r","ow")),
        ("Vosen",("v","ow","s","ah","n")),
        ("Meero",("m","iy","r","ow")),
        ("Vozen",("v","ow","z","ah","n"))):
        phones,tones,_=english.g2p("Hey "+term+".")
        stripped=[str(x).lower() for x in phones]
        start=next((i for i in range(len(stripped))
                    if stripped[i:i+len(expected)]==list(expected)),None)
        need(start is not None and
             all(x not in ("unk","UNK") for x in stripped),
             "UPSTREAM_NATIVE_ENGLISH_G2P_NOT_CMU_CONTROLLED:"+term)
        evidence[term]={"requested_critical":list(expected),
                        "all_phones":stripped,"tones":tones,
                        "matched_start_index":start}
    return evidence

def readback(root):
    root=Path(root)
    manifest=json.loads(c.read(root/"MANIFEST.json"))
    expected=planned()
    need(manifest.get("schema")==SCHEMA and
         manifest.get("model")==MODEL[0] and
         manifest.get("model_revision")==MODEL[1] and
         manifest.get("denominator")==MAX_SYNTH and
         manifest.get("training_allowed") is False and
         len(manifest.get("rows",[]))==MAX_SYNTH,
         "ORIGINAL_TWENTY_FIVE_CMU_ROWS")
    seen=set()
    for actual,request in zip(manifest["rows"],expected):
        need(all(actual.get(k)==v for k,v in request.items()),
             "UNALTERED_SPEAKER_AND_PHONEME_CONTROL_REQUEST")
        measured=c.inspect_wav(root/"wav"/(request["id"]+".wav"))
        need(measured["basic_audio_ok"] and
             all(actual.get(k)==measured[k] for k in
                 ("wav_sha256","pcm_sha256","samples")) and
             measured["pcm_sha256"] not in seen,
             "EXACT_NEW_NATIVE_PHONETIC_WAV")
        seen.add(measured["pcm_sha256"])
        need(actual.get("positive_duration_pass") ==
             (actual["requested_class"] not in (1,2) or
              measured["samples"]<=28672),
             "DURATION_FLAG_NOT_LABEL")
    return {"status":"SOURCE_AUTHENTICATED","rows":25,
            "training_allowed":False,"new_labels":0}

def run(root):
    source_check();identity=guard()
    root=Path(root)
    need(root.is_absolute() and not root.exists(),"CREATE_NEVER_REPLACE")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":SCHEMA,"run":identity,
           "planned_calls":MAX_SYNTH,"training_allowed":False})
    done=[]
    try:
        import hashlib
        from huggingface_hub import HfApi,snapshot_download
        import numpy as np,torch
        import soundfile as sf
        from scipy.signal import resample_poly
        import transformers
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        tmp=Path(os.environ.get("RUNNER_TEMP","/tmp"))
        m_info=HfApi(token=False).model_info(MODEL[0],revision=MODEL[1])
        b_info=HfApi(token=False).model_info(BERT[0],revision=BERT[1])
        need(m_info.sha==MODEL[1] and b_info.sha==BERT[1],
             "FROZEN_OFFICIAL_MODEL_REVISIONS")
        mdir=Path(snapshot_download(repo_id=MODEL[0],revision=MODEL[1],
            allow_patterns=["config.json","checkpoint.pth"],
            cache_dir=str(tmp/"melo-native-cache"),max_workers=2,token=False))
        bdir=Path(snapshot_download(repo_id=BERT[0],revision=BERT[1],
            allow_patterns=["config.json","tokenizer.json","tokenizer_config.json",
                            "vocab.txt","pytorch_model.bin"],
            cache_dir=str(tmp/"melo-bert-cache"),max_workers=2,token=False))
        weights=mdir/"checkpoint.pth"
        need(weights.is_file() and weights.stat().st_size>=190_000_000,
             "COMPLETE_ENGLISH_MELONET_WEIGHTS")
        h=hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1_048_576),b""):
                h.update(chunk)
        need(h.hexdigest()==CHECKPOINT_SHA,"OFFICIAL_MELONET_WEIGHT_SHA256")
        from transformers import AutoTokenizer,AutoModelForMaskedLM
        original_tokenizer=AutoTokenizer.from_pretrained
        original_bert=AutoModelForMaskedLM.from_pretrained
        @classmethod
        def locked_tokenizer(cls,identifier,*args,**kwargs):
            need(identifier=="bert-base-uncased","NO_UNPINNED_BERT_TOKENIZER")
            return original_tokenizer(str(bdir),local_files_only=True)
        @classmethod
        def locked_bert(cls,identifier,*args,**kwargs):
            need(identifier=="bert-base-uncased","NO_UNPINNED_BERT_WEIGHTS")
            return original_bert(str(bdir),local_files_only=True)
        AutoTokenizer.from_pretrained=locked_tokenizer
        AutoModelForMaskedLM.from_pretrained=locked_bert
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        from melo.text import english
        locked=phones_are_locked(english)
        c.record(root/"PHONEME_INPUT_LOCK.json",{"schema":SCHEMA,
              "source_code_sha":UPSTREAM_SHA,
              "official_model_sha":MODEL[1],
              "official_weight_sha256":CHECKPOINT_SHA,
              "bert_sha":BERT[1],"input_phones":locked,
              "requested_text_is_ground_truth":False})
        from melo.api import TTS
        model=TTS(language="EN",device="cpu",use_hf=False,
                  config_path=str(mdir/"config.json"),
                  ckpt_path=str(weights))
        need(set(SPEAKERS)<=set(model.hps.data.spk2id),
             "EXACT_TRAINED_ENGLISH_STYLE_IDS")
        native_sr=int(model.hps.data.sampling_rate)
        need(native_sr==44100,"EXACT_NATIVE_RATE")
        for index,row in enumerate(planned()):
            seed=int.from_bytes(hashlib.sha256((SCHEMA+"/"+row["id"]).encode()).digest()[:4],"big")
            torch.manual_seed(seed);np.random.seed(seed)
            with torch.inference_mode():
                audio=model.tts_to_file(row["render_text"],
                       model.hps.data.spk2id[row["speaker"]],
                       output_path=None,quiet=True,speed=1.0)
            raw=np.asarray(audio,dtype=np.float32)
            need(raw.ndim==1 and 0<len(raw)<native_sr*10
                 and np.isfinite(raw).all(),
                 "COMPLETE_NATIVE_VITS_FLOAT32_WAVE")
            pcm=resample_poly(raw,160,441).astype(np.float32)
            need(3200<=len(pcm)<=160000 and np.isfinite(pcm).all(),
                 "WHOLE_UNCROPPED_16KHZ_WAVE")
            import wave
            import struct
            quant=np.rint(np.clip(pcm,-1,32767/32768)*32768).astype("<i2").tobytes()
            wavdir=root/"wav"
            wavdir.mkdir(exist_ok=True)
            path=wavdir/(row["id"]+".wav")
            need(not path.exists(),"NEVER_REPLACE_SYNTHETIC_PCM")
            with wave.open(str(path),"wb") as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
                f.writeframes(quant)
            check=c.inspect_wav(path)
            need(check["basic_audio_ok"],"LEVELS_OR_RAILS_INVALID")
            record={**row,**check,"seed":seed,
                 "source_provider":"melo-cmu-phones",
                 "model_commit":MODEL[1],"native_sample_rate":native_sr,
                 "positive_duration_pass":row["requested_class"] not in (1,2)
                                          or len(pcm)<=28672}
            done.append(record)
            c.record(root/(row["id"]+".json"),record)
        c.record(root/"MANIFEST.json",{"schema":SCHEMA,
             "model":MODEL[0],"model_revision":MODEL[1],
             "source_revision":UPSTREAM_SHA,"bert_revision":BERT[1],
             "denominator":25,"completed":len(done),"rows":done,
             "training_allowed":False,"human_gold":False,
             "new_tts_calls":len(done),"independent_asr":"NOT_RUN",
             "independent_phone":"NOT_RUN"})
        print(json.dumps(readback(root),sort_keys=True))
    except BaseException as exc:
        c.record(root/"FAILED.json",{"schema":SCHEMA,
             "completed_native_audio":len(done),"remaining":25-len(done),
             "type":type(exc).__name__,"message":str(exc)[:500],
             "retry":False,"training_allowed":False})
        raise

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--guard",action="store_true")
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--readback",action="store_true")
    ap.add_argument("--output")
    a=ap.parse_args()
    need(sum(bool(x) for x in (a.source_check,a.guard,a.execute,a.readback))==1,
         "ONE_MODE_ONLY")
    if a.source_check:print(json.dumps(source_check()));return
    if a.guard:print(json.dumps(guard()));return
    need(a.output,"EXACT_SOURCE_OUTPUT_REQUIRED")
    if a.execute:run(a.output)
    else:print(json.dumps(readback(a.output)))

if __name__=="__main__":main()
