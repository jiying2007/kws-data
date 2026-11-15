"""Verify and compact the exact four-cell IPA probe artifact; no model/audio execution."""
from __future__ import annotations
import argparse, hashlib, json, shutil, stat, wave
from pathlib import Path

RUN_ID=43728096763
ARTIFACT_ID=14167735482
ARTIFACT_NAME="english-ipa-qwen17-probe-43728096763"
ARTIFACT_SHA256="35fc9ef1a76c5056d3c8307a87f9372f4d3f67412223d39c24fbc7724ee9b788"
MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_REVISION="0c0e3051f131929182e2c023b9537f8b1c68adfe"
PHRASES={"Hey Mero":"/heɪ ˈmɪəroʊ/","Hey Vosen":"/heɪ ˈvoʊsən/"}

def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def read(path):
    return json.loads(Path(path).read_bytes())
def safe_file(path):
    p=Path(path)
    need(p.is_file() and not p.is_symlink(),"REGULAR_FILE_REQUIRED")
    need(not any(x.is_symlink() for x in p.parents),"SYMLINK_PATH")
    return p

def verify_and_compact(source,destination):
    source=Path(source).resolve()
    destination=Path(destination).resolve()
    need(source.is_dir() and not source.is_symlink(),"SOURCE_DIRECTORY")
    # download-artifact creates a named child directory with merge-multiple=false.
    if not (source/"started.json").is_file():
        nested=source/ARTIFACT_NAME
        need(nested.is_dir() and not nested.is_symlink(),"ARTIFACT_ROOT_LAYOUT")
        source=nested
    need(not destination.exists() and destination.parent.is_dir(),"CREATE_ONLY_DESTINATION")
    for p in source.rglob("*"):
        need(not p.is_symlink(),"SOURCE_SYMLINK")
    started=read(source/"started.json")
    result=read(source/"result.json")
    lock=read(source/"model-lock.json")
    plan=read(source/"source-plan.json")
    need(started.get("schema")=="english-ipa-qwen17-probe-v1" and
         started.get("run_id")==str(RUN_ID) and started.get("no_retry") is True,
         "STARTED_RECEIPT_IDENTITY")
    need(result.get("schema")=="english-ipa-qwen17-probe-v1" and
         result.get("status")=="COMPLETE" and result.get("attempts")==4 and
         result.get("completed")==4 and result.get("asr_calls")==0 and
         result.get("training_admitted") is False and result.get("human_gold") is False,
         "RESULT_NOT_COMPLETE_OR_ADMISSION_FLAG")
    need(lock.get("schema")=="english-ipa-qwen17-model-lock-v1" and
         lock.get("model_id")==MODEL_ID and lock.get("revision")==MODEL_REVISION and
         lock.get("license")=="Apache-2.0","MODEL_LOCK_IDENTITY")
    need(plan.get("schema")=="english-ipa-qwen17-probe-v1" and
         plan.get("model")==MODEL_ID and plan.get("revision")==MODEL_REVISION and
         plan.get("training_admitted") is False and plan.get("human_gold") is False,
         "PLAN_IDENTITY")
    cells=plan.get("cells")
    need(type(cells) is list and len(cells)==4,"FOUR_CELL_PLAN")
    expected={(phrase,speaker) for phrase in PHRASES for speaker in ("aiden","ryan")}
    need({(x.get("phrase"),x.get("speaker")) for x in cells}==expected,"CELL_ROSTER")
    rows={x["cell_id"]:x for x in result.get("rows",[])}
    need(len(rows)==4 and set(rows)=={x["cell_id"] for x in cells},"RESULT_ROWS")
    destination.mkdir(mode=0o700)
    for name in ("started.json","result.json","model-lock.json","source-plan.json"):
        shutil.copyfile(safe_file(source/name),destination/name)
    summaries=[]
    for cell in cells:
        cid=cell["cell_id"]
        folder=source/"cells"/cid
        completion=read(folder/"completed.json")
        need(read(folder/"started.json").get("cell_id")==cid,"CELL_STARTED_ID")
        need(completion.get("schema")=="english-ipa-qwen17-probe-v1" and
             completion.get("cell_id")==cid and completion.get("phrase")==cell["phrase"] and
             completion.get("ipa_target")==cell["ipa"] and completion.get("speaker")==cell["speaker"] and
             completion.get("pronunciation_verified") is False and
             completion.get("human_gold") is False and completion.get("training_admitted") is False,
             "CELL_COMPLETION_IDENTITY")
        audio=completion.get("audio",{})
        native=safe_file(folder/"native-f32.wav")
        wav=safe_file(folder/"audio-16k.wav")
        need(sha_file(native)==audio.get("native_sha256") and
             sha_file(wav)==audio.get("wav_sha256"),"WAV_SHA256")
        with wave.open(str(wav),"rb") as f:
            need(f.getnchannels()==1 and f.getsampwidth()==2 and
                 f.getframerate()==16000 and f.getnframes()==audio.get("audio_frames"),
                 "PCM_WAV_GEOMETRY")
            pcm=f.readframes(f.getnframes())
        need(hashlib.sha256(pcm).hexdigest()==audio.get("pcm_sha256"),"PCM_SHA256")
        out=destination/"cells"/cid
        out.mkdir(parents=True,mode=0o700)
        for filename in ("started.json","completed.json","native-f32.wav","audio-16k.wav"):
            shutil.copyfile(safe_file(folder/filename),out/filename)
        summaries.append({"cell_id":cid,"phrase":cell["phrase"],"ipa":cell["ipa"],
          "speaker":cell["speaker"],"native_sha256":audio["native_sha256"],
          "wav_sha256":audio["wav_sha256"],"pcm_sha256":audio["pcm_sha256"],
          "audio_frames":audio["audio_frames"],"wav_bytes":audio["wav_bytes"]})
    compact_bytes=sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    need(compact_bytes<=20_000_000,"COMPACT_ARTIFACT_SIZE")
    summary={"schema":"english-ipa-qwen17-compact-readback-v1",
      "source_run_id":RUN_ID,"source_artifact_id":ARTIFACT_ID,
      "source_artifact_name":ARTIFACT_NAME,"source_artifact_sha256":ARTIFACT_SHA256,
      "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
      "status":"PASS_SAVED_AUDIO_IDENTITY_AND_RECEIPTS",
      "cell_count":4,"asr_calls":0,"model_calls_during_readback":0,
      "training_admitted":False,"human_gold":False,"pronunciation_verified":False,
      "compact_bytes":compact_bytes,"cells":summaries}
    (destination/"COMPACT_READBACK.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,sort_keys=True))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--destination",type=Path,required=True)
    a=p.parse_args()
    verify_and_compact(a.source,a.destination)
if __name__=="__main__": main()
