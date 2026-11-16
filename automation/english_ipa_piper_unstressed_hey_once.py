"""One-shot Piper raw-phoneme pronunciation probe; no retries, ASR or training."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, wave
from importlib.metadata import version
from pathlib import Path

POLICY="english-ipa-piper-unstressed-hey-probe-v1"
REPO="rhasspy/piper-voices"
REV="de3dcfdf1912bb49726dc4aa11c26017ce2ac62a"
MODEL_REL="en/en_US/libritts/high/en_US-libritts-high.onnx"
MODEL_SIZE=136_673_811
MODEL_SHA="9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f"
CONFIG_REL="en/en_US/libritts/high/en_US-libritts-high.onnx.json"
CONFIG_SIZE=20_163
CONFIG_MD5="cb1564bb779f3b1f87e2de778adeb669"
PIPER_VERSION="1.8.0"
MAX_MODEL_BYTES=150*1024**2
MAX_CALLS=4
TRIGGER=Path(__file__).resolve().parents[1]/".github/triggers/english-ipa-piper-unstressed-hey-20261116.json"
EXPECTED_CELLS=[
 {"phrase":"Hey Mero","ipa":"/heɪ ˈmɪəroʊ/","raw_espeak_ipa":"heɪ mˈɪəɹoʊ","speaker_id":0,"source_speaker_id":"p3922"},
 {"phrase":"Hey Mero","ipa":"/heɪ ˈmɪəroʊ/","raw_espeak_ipa":"heɪ mˈɪəɹoʊ","speaker_id":1,"source_speaker_id":"p8699"},
 {"phrase":"Hey Vosen","ipa":"/heɪ ˈvoʊsən/","raw_espeak_ipa":"heɪ vˈoʊsən","speaker_id":0,"source_speaker_id":"p3922"},
 {"phrase":"Hey Vosen","ipa":"/heɪ ˈvoʊsən/","raw_espeak_ipa":"heɪ vˈoʊsən","speaker_id":1,"source_speaker_id":"p8699"}
]
def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def md5(raw):
    return hashlib.md5(raw).hexdigest()
def safe(path,limit):
    p=Path(path)
    need(p.is_file() and not p.is_symlink(),"REGULAR_FILE_REQUIRED")
    need(not any(x.is_symlink() for x in p.parents),"SYMLINK_PATH")
    need(p.stat().st_size<=limit,"FILE_SIZE_LIMIT")
    return p.read_bytes()
def verify_trigger():
    expected={
      "schema":POLICY,"model_repo":REPO,"model_revision":REV,
      "model_file":MODEL_REL,"model_bytes":MODEL_SIZE,"model_sha256":MODEL_SHA,
      "config_file":CONFIG_REL,"config_bytes":CONFIG_SIZE,"config_md5":CONFIG_MD5,
      "piper_version":PIPER_VERSION,"speakers":[{"id":0,"source_speaker_id":"p3922"},{"id":1,"source_speaker_id":"p8699"}],
      "cells":EXPECTED_CELLS,"max_calls":MAX_CALLS,"retry_count":0,
      "asr_calls":0,"training_admitted":False,"human_gold":False,
      "pronunciation_verified":False,"voice_dataset_license":"CC-BY-4.0",
      "voice_dataset":"LibriTTS train-clean-360","voice_attribution_required":True
    }
    need(json.loads(TRIGGER.read_text(encoding="utf-8"))==expected,"TRIGGER_POLICY_DRIFT")
    return expected
def load_model(root):
    from huggingface_hub import hf_hub_download
    model=Path(hf_hub_download(repo_id=REPO,filename=MODEL_REL,revision=REV,
                               local_dir=str(root),token=False))
    config=Path(hf_hub_download(repo_id=REPO,filename=CONFIG_REL,revision=REV,
                                local_dir=str(root),token=False))
    mb=safe(model,MAX_MODEL_BYTES)
    cb=safe(config,1024*1024)
    need(len(mb)==MODEL_SIZE and sha(mb)==MODEL_SHA,"PIPER_MODEL_SHA")
    need(len(cb)==CONFIG_SIZE and md5(cb)==CONFIG_MD5,"PIPER_CONFIG_MD5")
    cfg=json.loads(cb)
    need(cfg.get("num_speakers")==904 and cfg.get("sample_rate")==22050 or
         (cfg.get("num_speakers")==904 and cfg.get("audio",{}).get("sample_rate")==22050),
         "PIPER_VOICE_GEOMETRY")
    speakers=cfg.get("speaker_id_map",{})
    need(speakers.get("p3922")==0 and speakers.get("p8699")==1,"PIPER_SPEAKER_MAP")
    return model,config,{"model_sha256":sha(mb),"model_bytes":len(mb),
      "config_md5":md5(cb),"config_bytes":len(cb),"speaker_count":cfg["num_speakers"],
      "speaker_ids":{"p3922":0,"p8699":1},"dataset":"LibriTTS train-clean-360",
      "dataset_license":"CC-BY-4.0","attribution_required":True}
def synthesize(voice,cell):
    import numpy as np
    from piper import SynthesisConfig
    raw="[[ "+cell["raw_espeak_ipa"]+" ]]"
    phonemes=voice.phonemize(raw)
    flattened=[p for sentence in phonemes for p in sentence]
    required=list(cell["raw_espeak_ipa"])
    need(flattened==required,"RAW_PHONEME_ROUNDTRIP")
    need(all(p in voice.config.phoneme_id_map for p in flattened if p!=" "),"PHONEME_ID_MISSING")
    config=SynthesisConfig(speaker_id=cell["speaker_id"],length_scale=1.0,
                           noise_scale=0.333,noise_w_scale=0.333,
                           normalize_audio=False,volume=1.0)
    chunks=list(voice.synthesize(raw,config))
    need(len(chunks)==1,"ONE_SENTENCE_ONE_CHUNK")
    chunk=chunks[0]
    need(chunk.sample_rate==22050 and chunk.sample_width==2 and chunk.sample_channels==1,
         "PIPER_PCM_GEOMETRY")
    pcm=chunk.audio_int16_bytes
    need(len(pcm)>0 and len(pcm)%2==0,"PIPER_PCM_LENGTH")
    samples=np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768.0
    rms=float(np.sqrt(np.mean(samples*samples)))
    peak=float(np.max(np.abs(samples)))
    need(0.003<=rms<=0.6 and 0.01<=peak<=1.0,"PIPER_AUDIO_LEVEL")
    return pcm,chunk.sample_rate,{"native_frames":len(samples),"native_rate":22050,
      "duration_s":len(samples)/22050,"rms":rms,"peak":peak,
      "raw_espeak_ipa":cell["raw_espeak_ipa"],"phoneme_sequence":"".join(flattened)}
def write_audio(folder,pcm,rate):
    import numpy as np
    from scipy.signal import resample_poly
    native=folder/"native-22050.wav"
    with wave.open(str(native),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm)
    x=np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768.0
    y=resample_poly(x,320,441)
    need(0<len(y)<=192000 and np.all(np.isfinite(y)),"RESAMPLE")
    pcm16=np.rint(np.clip(y,-1.0,0.999969)*32768).astype("<i2").tobytes()
    derived=folder/"audio-16k.wav"
    with wave.open(str(derived),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm16)
    nb=native.read_bytes();wb=derived.read_bytes()
    return {"native_sha256":sha(nb),"native_bytes":len(nb),
      "wav_sha256":sha(wb),"wav_bytes":len(wb),
      "pcm_sha256":sha(pcm16),"audio_frames":len(pcm16)//2,
      "sample_rate_hz":16000,"native_rate_hz":rate}
def execute(output):
    need(os.environ.get("GITHUB_REPOSITORY")=="jiying2007/kws-data" and
         os.environ.get("GITHUB_REF")=="refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT")=="1","MAIN_ATTEMPT_ONE_ONLY")
    trigger=verify_trigger()
    output=Path(output).resolve()
    need(output.parent.is_dir() and not output.exists(),"CREATE_ONLY_OUTPUT")
    output.mkdir(mode=0o700)
    runner=output/"private-model";runner.mkdir(mode=0o700)
    result={"schema":POLICY,"status":"FAILED_NO_RETRY","attempts":0,
      "completed":0,"asr_calls":0,"training_admitted":False,
      "human_gold":False,"pronunciation_verified":False}
    try:
        need(version("piper-tts")==PIPER_VERSION,"PIPER_RUNTIME_VERSION")
        model,config,lock=load_model(runner)
        from piper import PiperVoice
        voice=PiperVoice.load(str(model))
        (output/"model-lock.json").write_text(json.dumps({
          "schema":"english-ipa-piper-model-lock-v1","repo":REPO,"revision":REV,
          "license":"CC-BY-4.0","engine":"piper-tts","engine_version":PIPER_VERSION,
          **lock},ensure_ascii=False,indent=2)+"\n")
        plan={"schema":POLICY,"model_repo":REPO,"model_revision":REV,
          "model_sha256":lock["model_sha256"],"config_md5":lock["config_md5"],
          "piper_version":PIPER_VERSION,"cells":EXPECTED_CELLS,
          "training_admitted":False,"human_gold":False,"pronunciation_verified":False}
        (output/"source-plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
        for i,cell in enumerate(EXPECTED_CELLS):
            cid="pipa-"+sha(json.dumps({"i":i,"cell":cell,"revision":REV},
                sort_keys=True,separators=(",",":")).encode())[:24]
            folder=output/"cells"/cid
            folder.mkdir(parents=True,mode=0o700)
            (folder/"started.json").write_text(json.dumps({
              "schema":POLICY,"cell_id":cid,"attempt":1,"retry_count":0,
              "training_admitted":False},indent=2)+"\n")
            result["attempts"]+=1
            try:
                pcm,rate,metrics=synthesize(voice,cell)
                assets=write_audio(folder,pcm,rate)
                (folder/"completed.json").write_text(json.dumps({
                  "schema":POLICY,"cell_id":cid,"phrase":cell["phrase"],
                  "ipa_target":cell["ipa"],"raw_espeak_ipa":cell["raw_espeak_ipa"],
                  "speaker_id":cell["speaker_id"],"source_speaker_id":cell["source_speaker_id"],
                  "technical":metrics,"audio":assets,"pronunciation_verified":False,
                  "human_gold":False,"training_admitted":False},ensure_ascii=False,indent=2)+"\n")
                result["completed"]+=1
            except Exception as exc:
                (folder/"failed.json").write_text(json.dumps({
                  "schema":POLICY,"cell_id":cid,"status":"FAILED_NO_RETRY",
                  "error_type":type(exc).__name__,"training_admitted":False},indent=2)+"\n")
        result.update({"status":"COMPLETE" if result["completed"]==4 else "INCOMPLETE_NO_RETRY",
          "model_lock_sha256":sha((output/"model-lock.json").read_bytes()),
          "source_plan_sha256":sha((output/"source-plan.json").read_bytes()),
          "piper_version":PIPER_VERSION,"model_revision":REV,
          "asr_calls":0,"training_admitted":False,"human_gold":False,
          "pronunciation_verified":False})
    except Exception as exc:
        result["error_type"]=type(exc).__name__
    finally:
        (output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        if runner.exists(): shutil.rmtree(runner)
    return result
def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args(argv)
    if not a.execute:
        verify_trigger()
        print("PLAN_VALID_NO_MODEL_OR_AUDIO_EXECUTION")
        return 0
    result=execute(a.output)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0 if result.get("status")=="COMPLETE" else 1
if __name__=="__main__":
    raise SystemExit(main())
