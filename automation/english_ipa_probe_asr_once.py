"""Three-family blind ASR screen of the four confirmed-IPA TTS probes; no label admission."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,wave
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_blind_asr as blind
import english_seven_voice_asr_once as seven
import english_qwen_capture as english
import multilingual_campaign as market

POLICY="english-ipa-qwen17-three-blind-asr-v1"
SOURCE_RUN_ID=43730318351
SOURCE_ARTIFACT_ID=14168142552
SOURCE_ARTIFACT_NAME="english-ipa-qwen17-probe-compact-43728096763"
SOURCE_ARTIFACT_SHA256="e9c4c5a0b4f4e99a6e5e9e7d4b6c0a1e5e1b4c2b4f9a0d2c7a4f8a0c4e7b6a9d"
PROVIDERS=("qwen","whisper","parakeet")
PHRASES={"Hey Mero":"/heɪ ˈmɪəroʊ/","Hey Vosen":"/heɪ ˈvoʊsən/"}
MAX_CALLS=12

def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def safe_file(path,limit):
    return english.media()._safe_file(Path(path),limit=limit)
def root_for(parent):
    parent=Path(parent).resolve()
    need(parent.is_dir() and not parent.is_symlink(),"COMPACT_PARENT")
    roots=[p.parent for p in parent.rglob("COMPACT_READBACK.json")
           if p.parent.is_dir() and not p.parent.is_symlink()]
    need(len(roots)==1,"COMPACT_ROOT_UNIQUE")
    root=roots[0]
    summary=market.read(root/"COMPACT_READBACK.json")
    need(summary.get("schema")=="english-ipa-qwen17-compact-readback-v1" and
         summary.get("source_run_id")==SOURCE_RUN_ID and
         summary.get("source_artifact_id")==SOURCE_ARTIFACT_ID and
         summary.get("source_artifact_name")==SOURCE_ARTIFACT_NAME and
         summary.get("status")=="PASS_SAVED_AUDIO_IDENTITY_AND_RECEIPTS" and
         summary.get("cell_count")==4 and summary.get("asr_calls")==0 and
         summary.get("pronunciation_verified") is False and
         summary.get("human_gold") is False and
         summary.get("training_admitted") is False,
         "COMPACT_READBACK_IDENTITY")
    plan=market.read(root/"source-plan.json")
    need(plan.get("schema")=="english-ipa-qwen17-probe-v1" and
         plan.get("model_revision")=="0c0e3051f131929182e2c023b9537f8b1c68adfe" and
         plan.get("training_admitted") is False and
         plan.get("human_gold") is False,"SOURCE_PLAN_IDENTITY")
    cells=plan.get("cells")
    expected={(phrase,speaker) for phrase in PHRASES for speaker in ("aiden","ryan")}
    need(type(cells) is list and len(cells)==4 and
         {(c.get("phrase"),c.get("speaker")) for c in cells}==expected,
         "FOUR_CELL_SOURCE_ROSTER")
    verified=[]
    for cell in cells:
        folder=root/"cells"/cell["cell_id"]
        completion=market.read(folder/"completed.json")
        need(completion.get("cell_id")==cell["cell_id"] and
             completion.get("phrase")==cell["phrase"] and
             completion.get("ipa_target")==cell["ipa"] and
             completion.get("pronunciation_verified") is False and
             completion.get("human_gold") is False and
             completion.get("training_admitted") is False,
             "CELL_RECEIPT")
        wav=folder/"audio-16k.wav"
        n,digest=safe_file(wav,2*1024*1024)
        audio=completion["audio"]
        need(digest==audio["wav_sha256"] and n==audio["wav_bytes"],"WAV_HASH")
        with wave.open(str(wav),"rb") as f:
            need(f.getnchannels()==1 and f.getsampwidth()==2 and
                 f.getframerate()==16000 and f.getnframes()==audio["audio_frames"],
                 "WAV_GEOMETRY")
            pcm=f.readframes(f.getnframes())
        need(sha(pcm)==audio["pcm_sha256"],"PCM_HASH")
        verified.append((cell,wav,{"wav_sha256":digest,"frames":len(pcm)//2}))
    return root,plan,verified

def run_provider(provider,source_parent,out):
    need(provider in PROVIDERS,"PROVIDER")
    need(os.environ.get("GITHUB_REPOSITORY")=="jiying2007/kws-data" and
         os.environ.get("GITHUB_REF")=="refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT")=="1","MAIN_ATTEMPT_ONE")
    _,plan,cells=root_for(source_parent)
    out=Path(out).resolve()
    need(out.parent.is_dir() and not out.exists(),"OUTPUT_CREATE_ONLY")
    out.mkdir(mode=0o700)
    provider_id,model_id,revision=seven.PROVENANCE[provider]
    media=english.media()
    media._once(out/"started.json",{"schema":POLICY,"provider":provider,
      "provider_id":provider_id,"model_id":model_id,"model_revision_sha":revision,
      "source_run_id":SOURCE_RUN_ID,"source_artifact_id":SOURCE_ARTIFACT_ID,
      "calls_budget":4,"training_admitted":False,"human_gold":False,"retry_count":0})
    model_root=out/"private-model";model_root.mkdir(mode=0o700)
    rows=[]
    try:
        decode=seven._load(provider,model_root)
        for cell,wav,pcm in cells:
            cid=cell["cell_id"]
            folder=out/"cells"/cid
            folder.mkdir(parents=True,mode=0o700)
            media._once(folder/"started.json",{"schema":POLICY,"cell_id":cid,
              "provider":provider,"wav_sha256":pcm["wav_sha256"],"attempt":1})
            text=decode(wav,folder)
            need(type(text) is str and 0<len(text)<=512 and
                 not any(ord(ch)<32 for ch in text),"ASR_TEXT")
            native={"schema":blind.RAW_SCHEMA,"provider_id":provider_id,
              "model_id":model_id,"model_revision_sha":revision,
              "audio_wav_sha256":pcm["wav_sha256"],"processed_start_frame":0,
              "processed_end_frame":pcm["frames"],"decoder_completed":True,
              "actual_text":text}
            media._once(folder/"native.json",native)
            blind.native_result(folder/"native.json",{"provider_id":provider_id,
              "model_id":model_id,"model_revision_sha":revision},cell,pcm)
            media._once(folder/"completed.json",{"schema":POLICY,"cell_id":cid,
              "provider":provider,"provider_id":provider_id,"wav_sha256":pcm["wav_sha256"],
              "native_sha256":sha((folder/"native.json").read_bytes()),
              "status":"BLIND_ASR_UNADMITTED","training_admitted":False,"human_gold":False})
            rows.append({"cell_id":cid,"wav_sha256":pcm["wav_sha256"],"status":"COMPLETE"})
    except Exception as e:
        media._once(out/"failed.json",{"schema":POLICY,"provider":provider,
          "status":"INCOMPLETE_NO_RETRY","error_type":type(e).__name__,
          "completed_cells":len(rows),"training_admitted":False})
        raise
    finally:
        if model_root.exists(): shutil.rmtree(model_root)
    summary={"schema":POLICY,"status":"COMPLETE" if len(rows)==4 else "INCOMPLETE",
      "provider":provider,"provider_id":provider_id,"model_id":model_id,
      "model_revision_sha":revision,"source_run_id":SOURCE_RUN_ID,
      "source_artifact_id":SOURCE_ARTIFACT_ID,"source_cell_count":4,
      "asr_attempts":len(rows),"rows":rows,"training_admitted":False,
      "human_gold":False,"pronunciation_verified":False}
    media._once(out/"summary.json",summary)
    return summary

def combine(results_parent,out):
    results_parent=Path(results_parent).resolve()
    out=Path(out).resolve()
    need(results_parent.is_dir() and not out.exists(),"COMBINE_PATHS")
    transcripts={}
    for provider in PROVIDERS:
        roots=[p.parent for p in results_parent.rglob("summary.json")
               if p.parent.is_dir() and market.read(p).get("provider")==provider]
        need(len(roots)==1,"PROVIDER_RESULT_ROOT")
        root=roots[0]
        summary=market.read(root/"summary.json")
        need(summary.get("schema")==POLICY and summary.get("status")=="COMPLETE" and
             summary.get("asr_attempts")==4 and summary.get("training_admitted") is False,
             "PROVIDER_RESULT_NOT_COMPLETE")
        for cell_dir in (root/"cells").iterdir():
            native=market.read(cell_dir/"native.json")
            need(native.get("decoder_completed") is True and
                 native.get("provider_id")==summary["provider_id"] and
                 native.get("audio_wav_sha256")==next(x["wav_sha256"] for x in summary["rows"] if x["cell_id"]==native["cell_id"]) if "cell_id" in native else True,
                 "NATIVE_RESULT_PROVENANCE")
            transcripts.setdefault(cell_dir.name,{})[provider]=native["actual_text"]
    source=Path(os.environ.get("COMPACT_SOURCE_ROOT",""))
    need(len(transcripts)==4,"COMBINED_CELL_COUNT")
    out.mkdir(mode=0o700)
    rows=[]
    for cid,values in sorted(transcripts.items()):
        need(set(values)==set(PROVIDERS),"THREE_ASR_REQUIRED")
        folded=[market.normalize(values[p],"en-US") for p in PROVIDERS]
        # Target text is loaded only after all blind recognizers have completed.
        rows.append({"cell_id":cid,"transcripts":values,
          "all_three_normalized_agree":len(set(folded))==1,
          "normalized_transcript":folded[0] if len(set(folded))==1 else None,
          "acoustic_truth":False,"human_gold":False,"training_admitted":False})
    summary={"schema":POLICY,"status":"THREE_ASR_COMPLETE_REVIEW_REQUIRED",
      "source_run_id":SOURCE_RUN_ID,"source_artifact_id":SOURCE_ARTIFACT_ID,
      "providers":list(PROVIDERS),"asr_calls":12,"cells":rows,
      "machine_consensus_is_not_acoustic_gold":True,
      "training_admitted":False,"human_gold":False}
    (out/"three-asr-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,sort_keys=True))
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--provider",choices=PROVIDERS)
    p.add_argument("--source",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--combine",type=Path)
    p.add_argument("--combined-output",type=Path)
    a=p.parse_args()
    if a.combine:
        combine(a.combine,a.combined_output);return
    need(a.provider and a.source and a.output,"ARGS")
    run_provider(a.provider,a.source,a.output)
if __name__=="__main__": main()
