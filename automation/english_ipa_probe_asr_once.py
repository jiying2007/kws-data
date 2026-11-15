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
SOURCE_RUN_ID=43728096763
SOURCE_ARTIFACT_ID=14167735482
SOURCE_ARTIFACT_NAME="english-ipa-qwen17-probe-43728096763"
SOURCE_ARTIFACT_SHA256="35fc9ef1a76c5056d3c8307a87f9372f4d3f67412223d39c24fbc7724ee9b788"
COMPACT_PRODUCER_RUN_ID=43730318351
COMPACT_ARTIFACT_ID=14168142552
COMPACT_ARTIFACT_NAME="english-ipa-qwen17-probe-compact-43728096763"
COMPACT_ARTIFACT_SHA256="717b52d4fef8b17aa3e91baf1e28663c8d337919b4dd49f5cfdbba050f32cf64"
PROVIDERS=("qwen","whisper","parakeet")
PHRASES={"Hey Mero":"/heɪ ˈmɪəroʊ/","Hey Vosen":"/heɪ ˈvoʊsən/"}
MAX_CALLS=12
TRIGGER=Path(__file__).resolve().parents[1]/".github/triggers/english-ipa-probe-three-asr-20261115.json"

def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def safe_file(path,limit):
    return english.media()._safe_file(Path(path),limit=limit)
def verify_trigger():
    expected={"schema":POLICY,"source_run_id":SOURCE_RUN_ID,
      "source_artifact_id":SOURCE_ARTIFACT_ID,
      "source_artifact_name":SOURCE_ARTIFACT_NAME,
      "source_artifact_sha256":SOURCE_ARTIFACT_SHA256,
      "compact_producer_run_id":COMPACT_PRODUCER_RUN_ID,
      "compact_artifact_id":COMPACT_ARTIFACT_ID,
      "compact_artifact_name":COMPACT_ARTIFACT_NAME,
      "compact_artifact_sha256":COMPACT_ARTIFACT_SHA256,
      "providers":list(PROVIDERS),"max_calls":MAX_CALLS,"retry_count":0,
      "training_admitted":False,"human_gold":False,"pronunciation_verified":False}
    need(market.read(TRIGGER)==expected,"TRIGGER_POLICY_DRIFT")
    return expected


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
         summary.get("source_artifact_sha256")==SOURCE_ARTIFACT_SHA256 and
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
    verify_trigger()
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
    model_root=out/"private-model"
    model_root.mkdir(mode=0o700)
    rows=[]
    attempts=0
    model_error=None
    try:
        try:
            decode=seven._load(provider,model_root)
        except Exception as exc:
            decode=None
            model_error=type(exc).__name__
        for cell,wav,pcm in cells:
            cid=cell["cell_id"]
            folder=out/"cells"/cid
            folder.mkdir(parents=True,mode=0o700)
            media._once(folder/"started.json",{"schema":POLICY,"cell_id":cid,
              "provider":provider,"wav_sha256":pcm["wav_sha256"],"attempt":1,
              "training_admitted":False})
            if decode is None:
                media._once(folder/"failed.json",{"schema":POLICY,"cell_id":cid,
                  "provider":provider,"status":"MODEL_LOAD_FAILED_NO_RETRY",
                  "error_type":model_error,"training_admitted":False})
                rows.append({"cell_id":cid,"wav_sha256":pcm["wav_sha256"],
                  "status":"MODEL_LOAD_FAILED_NO_RETRY"})
                continue
            attempts+=1
            try:
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
                status="COMPLETE"
            except Exception as exc:
                media._once(folder/"failed.json",{"schema":POLICY,"cell_id":cid,
                  "provider":provider,"status":"FAILED_NO_RETRY",
                  "error_type":type(exc).__name__,"training_admitted":False})
                status="FAILED_NO_RETRY"
            rows.append({"cell_id":cid,"wav_sha256":pcm["wav_sha256"],"status":status})
    finally:
        if model_root.exists():
            shutil.rmtree(model_root)
    completed=sum(row["status"]=="COMPLETE" for row in rows)
    summary={"schema":POLICY,"status":"COMPLETE" if completed==4 else "INCOMPLETE_NO_RETRY",
      "provider":provider,"provider_id":provider_id,"model_id":model_id,
      "model_revision_sha":revision,"source_run_id":SOURCE_RUN_ID,
      "source_artifact_id":SOURCE_ARTIFACT_ID,"source_artifact_sha256":SOURCE_ARTIFACT_SHA256,
      "source_cell_count":4,"asr_attempts":attempts,"completed_asr":completed,
      "model_load_error_type":model_error,"rows":rows,"training_admitted":False,
      "human_gold":False,"pronunciation_verified":False}
    media._once(out/"summary.json",summary)
    return summary

def combine(results_parent,out,source_parent):
    results_parent=Path(results_parent).resolve()
    out=Path(out).resolve()
    need(results_parent.is_dir() and not out.exists(),"COMBINE_PATHS")
    _,plan,source_cells=root_for(source_parent)
    target_by_id={cell["cell_id"]:cell for cell,_,_ in source_cells}
    transcripts={}
    for provider in PROVIDERS:
        root=results_parent/provider
        need(root.is_dir(),"PROVIDER_RESULT_ROOT")
        summary=market.read(root/"summary.json")
        need(summary.get("schema")==POLICY and summary.get("status")=="COMPLETE" and
             summary.get("provider")==provider and summary.get("asr_attempts")==4 and
             summary.get("training_admitted") is False and summary.get("human_gold") is False,
             "PROVIDER_RESULT_NOT_COMPLETE")
        rows={row["cell_id"]:row for row in summary["rows"]}
        need(set(rows)==set(target_by_id),"PROVIDER_CELL_ROSTER")
        for cid in sorted(target_by_id):
            native=market.read(root/"cells"/cid/"native.json")
            need(native.get("decoder_completed") is True and
                 native.get("provider_id")==summary["provider_id"] and
                 native.get("audio_wav_sha256")==rows[cid]["wav_sha256"] and
                 type(native.get("actual_text")) is str and bool(native["actual_text"]),
                 "NATIVE_RESULT_PROVENANCE")
            transcripts.setdefault(cid,{})[provider]=native["actual_text"]
    need(len(transcripts)==4,"COMBINED_CELL_COUNT")
    out.mkdir(mode=0o700)
    rows=[]
    for cid,values in sorted(transcripts.items()):
        need(set(values)==set(PROVIDERS),"THREE_ASR_REQUIRED")
        folded=[market.normalize(values[p],"en-US") for p in PROVIDERS]
        target=target_by_id[cid]["phrase"]
        agree=len(set(folded))==1
        match=agree and folded[0]==market.normalize(target,"en-US")
        rows.append({"cell_id":cid,"target_phrase":target,
          "target_ipa":target_by_id[cid]["ipa"],"transcripts":values,
          "all_three_normalized_agree":agree,
          "agreed_transcript":folded[0] if agree else None,
          "agrees_with_requested_phrase":match,
          "machine_verdict":"THREE_ASR_AGREE_TARGET_REVIEW_REQUIRED" if match else
             "MISMATCH_OR_DISAGREEMENT_REVIEW_REQUIRED",
          "acoustic_truth":False,"human_gold":False,"training_admitted":False})
    summary={"schema":POLICY,"status":"THREE_ASR_COMPLETE_REVIEW_REQUIRED",
      "source_run_id":SOURCE_RUN_ID,"source_artifact_id":SOURCE_ARTIFACT_ID,
      "providers":list(PROVIDERS),"asr_calls":12,"cells":rows,
      "machine_consensus_is_not_acoustic_gold":True,
      "training_admitted":False,"human_gold":False}
    (out/"three-asr-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\\n")
    print(json.dumps(summary,ensure_ascii=False,sort_keys=True))
    return summary
def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--provider",choices=PROVIDERS)
    p.add_argument("--source",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--combine",type=Path)
    p.add_argument("--compact-source",type=Path)
    p.add_argument("--combined-output",type=Path)
    a=p.parse_args(argv)
    if a.combine:
        need(a.compact_source is not None,"COMBINE_SOURCE_REQUIRED")
        combine(a.combine,a.combined_output,a.compact_source);return
    need(a.provider and a.source and a.output,"ARGS")
    run_provider(a.provider,a.source,a.output)
if __name__=="__main__": main()
