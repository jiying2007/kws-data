"""Three-family blind ASR on exact-IPA Piper probe; no label admission."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,wave
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_blind_asr as blind
import english_seven_voice_asr_once as seven
import english_qwen_capture as english
import multilingual_campaign as market

POLICY="english-ipa-piper-three-blind-asr-v1"
SOURCE_RUN_ID=43779647466
SOURCE_ARTIFACT_ID=14187636805
SOURCE_ARTIFACT_NAME="english-ipa-piper-probe-43779647466"
SOURCE_ARTIFACT_SHA256="0e84ed8724748b7a9891527e44efc079a60514943e41fcdf0fbb585db44ff253"
PIPER_MODEL_REV="de3dcfdf1912bb49726dc4aa11c26017ce2ac62a"
PROVIDERS=("qwen","whisper","parakeet")
MAX_CALLS=12
TRIGGER=Path(__file__).resolve().parents[1]/".github/triggers/english-ipa-piper-three-asr-20261116.json"

def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def verify_trigger():
    expected={"schema":POLICY,"source_run_id":SOURCE_RUN_ID,
      "source_artifact_id":SOURCE_ARTIFACT_ID,"source_artifact_name":SOURCE_ARTIFACT_NAME,
      "source_artifact_sha256":SOURCE_ARTIFACT_SHA256,
      "piper_model_revision":PIPER_MODEL_REV,
      "providers":list(PROVIDERS),"max_calls":MAX_CALLS,"retry_count":0,
      "training_admitted":False,"human_gold":False,"pronunciation_verified":False}
    need(market.read(TRIGGER)==expected,"TRIGGER_POLICY_DRIFT")
    return expected

def source_root(parent):
    parent=Path(parent).resolve()
    need(parent.is_dir() and not parent.is_symlink(),"SOURCE_PARENT")
    roots=[]
    for result_path in parent.rglob("result.json"):
        root=result_path.parent
        if all((root/name).is_file() for name in ("source-plan.json","model-lock.json")):
            roots.append(root)
    need(len(roots)==1,"SOURCE_ROOT_UNIQUE")
    root=roots[0]
    result=market.read(root/"result.json")
    plan=market.read(root/"source-plan.json")
    lock=market.read(root/"model-lock.json")
    need(result.get("schema")=="english-ipa-piper-phoneme-probe-v1" and
         result.get("status")=="COMPLETE" and result.get("attempts")==4 and
         result.get("completed")==4 and result.get("asr_calls")==0 and
         result.get("training_admitted") is False and result.get("human_gold") is False,
         "PIPER_SOURCE_RESULT")
    need(result.get("model_revision")==PIPER_MODEL_REV and
         result.get("source_plan_sha256")==sha((root/"source-plan.json").read_bytes()) and
         result.get("model_lock_sha256")==sha((root/"model-lock.json").read_bytes()),
         "PIPER_SOURCE_HASHES")
    need(plan.get("schema")=="english-ipa-piper-phoneme-probe-v1" and
         plan.get("model_revision")==PIPER_MODEL_REV and
         plan.get("training_admitted") is False and plan.get("human_gold") is False,
         "PIPER_SOURCE_PLAN")
    need(lock.get("schema")=="english-ipa-piper-model-lock-v1" and
         lock.get("repo")=="rhasspy/piper-voices" and
         lock.get("revision")==PIPER_MODEL_REV and
         lock.get("license")=="CC-BY-4.0" and lock.get("dataset_license")=="CC-BY-4.0",
         "PIPER_MODEL_LOCK")
    plan_cells=plan.get("cells")
    expected={(x["phrase"],x["speaker_id"],x["raw_espeak_ipa"]) for x in plan_cells}
    need(len(plan_cells)==4 and expected=={
      ("Hey Mero",0,"hˈeɪ mˈɪəɹoʊ"),("Hey Mero",1,"hˈeɪ mˈɪəɹoʊ"),
      ("Hey Vosen",0,"hˈeɪ vˈoʊsən"),("Hey Vosen",1,"hˈeɪ vˈoʊsən")},
      "PIPER_CELL_ROSTER")
    folders=[p for p in (root/"cells").iterdir() if p.is_dir()]
    need(len(folders)==4,"PIPER_CELL_DIRS")
    cells=[]
    for folder in folders:
        completion=market.read(folder/"completed.json")
        need(completion.get("schema")=="english-ipa-piper-phoneme-probe-v1" and
             completion.get("pronunciation_verified") is False and
             completion.get("human_gold") is False and
             completion.get("training_admitted") is False,"PIPER_CELL_ADMISSION")
        cell=next((x for x in plan_cells if
          x["phrase"]==completion.get("phrase") and
          x["speaker_id"]==completion.get("speaker_id") and
          x["raw_espeak_ipa"]==completion.get("raw_espeak_ipa")),None)
        need(cell is not None,"PIPER_CELL_PLAN_JOIN")
        wav=folder/"audio-16k.wav"
        n,digest=english.media()._safe_file(wav,limit=2*1024*1024)
        audio=completion.get("audio",{})
        need(n==audio.get("wav_bytes") and digest==audio.get("wav_sha256"),
             "PIPER_WAV_SHA")
        with wave.open(str(wav),"rb") as f:
            frames=f.getnframes()
            need(f.getnchannels()==1 and f.getsampwidth()==2 and
                 f.getframerate()==16000 and frames==audio.get("audio_frames"),
                 "PIPER_WAV_GEOMETRY")
            pcm=f.readframes(frames)
        need(sha(pcm)==audio.get("pcm_sha256"),"PIPER_PCM_SHA")
        cells.append({"cell_id":completion["cell_id"],"phrase":cell["phrase"],
          "ipa":cell["ipa"],"raw_espeak_ipa":cell["raw_espeak_ipa"],
          "speaker_id":cell["speaker_id"],"wav":wav,
          "pcm":{"wav_sha256":digest,"frames":frames}})
    need(len(cells)==4,"PIPER_VERIFIED_CELL_COUNT")
    return root,plan,sorted(cells,key=lambda x:(x["phrase"],x["speaker_id"]))

def run_provider(provider,source_parent,out):
    need(provider in PROVIDERS,"PROVIDER")
    need(os.environ.get("GITHUB_REPOSITORY")=="jiying2007/kws-data" and
         os.environ.get("GITHUB_REF")=="refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT")=="1","MAIN_ATTEMPT_ONE")
    verify_trigger()
    _,plan,cells=source_root(source_parent)
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
    rows=[];attempts=0;model_error=None
    try:
        try:
            decode=seven._load(provider,model_root)
        except Exception as exc:
            decode=None;model_error=type(exc).__name__
        for cell in cells:
            cid=cell["cell_id"];wav=cell["wav"];pcm=cell["pcm"]
            folder=out/"cells"/cid;folder.mkdir(parents=True,mode=0o700)
            media._once(folder/"started.json",{"schema":POLICY,"cell_id":cid,
              "provider":provider,"wav_sha256":pcm["wav_sha256"],"attempt":1,
              "training_admitted":False})
            if decode is None:
                media._once(folder/"failed.json",{"schema":POLICY,"cell_id":cid,
                  "status":"MODEL_LOAD_FAILED_NO_RETRY","error_type":model_error,
                  "training_admitted":False})
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
                  "status":"FAILED_NO_RETRY","error_type":type(exc).__name__,
                  "training_admitted":False})
                status="FAILED_NO_RETRY"
            rows.append({"cell_id":cid,"wav_sha256":pcm["wav_sha256"],"status":status})
    finally:
        if model_root.exists(): shutil.rmtree(model_root)
    completed=sum(x["status"]=="COMPLETE" for x in rows)
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
    results_parent=Path(results_parent).resolve();out=Path(out).resolve()
    need(results_parent.is_dir() and not out.exists(),"COMBINE_PATHS")
    _,plan,cells=source_root(source_parent)
    targets={x["cell_id"]:x for x in cells};transcripts={}
    for provider in PROVIDERS:
        root=results_parent/provider
        need(root.is_dir(),"PROVIDER_RESULT_ROOT")
        summary=market.read(root/"summary.json")
        need(summary.get("schema")==POLICY and summary.get("status")=="COMPLETE" and
             summary.get("provider")==provider and summary.get("asr_attempts")==4 and
             summary.get("training_admitted") is False and summary.get("human_gold") is False,
             "PROVIDER_RESULT_NOT_COMPLETE")
        rows={x["cell_id"]:x for x in summary["rows"]}
        need(set(rows)==set(targets),"PROVIDER_CELL_ROSTER")
        for cid in sorted(targets):
            native=market.read(root/"cells"/cid/"native.json")
            need(native.get("decoder_completed") is True and
                 native.get("provider_id")==summary["provider_id"] and
                 native.get("audio_wav_sha256")==rows[cid]["wav_sha256"] and
                 type(native.get("actual_text")) is str and bool(native["actual_text"]),
                 "NATIVE_RESULT_PROVENANCE")
            transcripts.setdefault(cid,{})[provider]=native["actual_text"]
    out.mkdir(mode=0o700);rows=[]
    for cid,values in sorted(transcripts.items()):
        need(set(values)==set(PROVIDERS),"THREE_ASR_REQUIRED")
        cell=targets[cid]
        folded=[market.normalize(values[p],"en-US") for p in PROVIDERS]
        agree=len(set(folded))==1
        match=agree and folded[0]==market.normalize(cell["phrase"],"en-US")
        rows.append({"cell_id":cid,"target_phrase":cell["phrase"],
          "target_ipa":cell["ipa"],"raw_espeak_ipa":cell["raw_espeak_ipa"],
          "speaker_id":cell["speaker_id"],"transcripts":values,
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
    (out/"three-asr-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
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
