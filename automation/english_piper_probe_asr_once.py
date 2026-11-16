"""Three-family blind ASR for the exact, immutable Piper IPA probe; no label admission."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, wave
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
PROVIDERS=("qwen","whisper","parakeet")
PHRASES={"Hey Mero":"/heɪ ˈmɪəroʊ/","Hey Vosen":"/heɪ ˈvoʊsən/"}
SPEAKERS={"p3922":0,"p8699":1}
MAX_CALLS=12
TRIGGER=Path(__file__).resolve().parents[1]/".github/triggers/english-ipa-piper-three-asr-20261116.json"

def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
def verify_trigger():
    expected={"schema":POLICY,"source_run_id":SOURCE_RUN_ID,
      "source_artifact_id":SOURCE_ARTIFACT_ID,"source_artifact_name":SOURCE_ARTIFACT_NAME,
      "source_artifact_sha256":SOURCE_ARTIFACT_SHA256,"providers":list(PROVIDERS),
      "max_calls":MAX_CALLS,"retry_count":0,"training_admitted":False,
      "human_gold":False,"pronunciation_verified":False}
    need(read(TRIGGER)==expected,"TRIGGER_DRIFT")
    return expected

def source_cells(parent):
    parent=Path(parent).resolve()
    need(parent.is_dir() and not parent.is_symlink(),"SOURCE_PARENT")
    matches=[p for p in parent.rglob("result.json") if p.parent.name=="probe"]
    need(len(matches)==1,"SOURCE_ROOT_UNIQUE")
    root=matches[0].parent
    summary=read(root/"result.json")
    need(summary.get("schema")=="english-ipa-piper-phoneme-probe-v1" and
         summary.get("status")=="COMPLETE" and summary.get("attempts")==4 and
         summary.get("completed")==4 and summary.get("asr_calls")==0 and
         summary.get("training_admitted") is False and summary.get("human_gold") is False and
         summary.get("pronunciation_verified") is False,"PIPER_SOURCE_NOT_COMPLETE")
    lock=read(root/"model-lock.json")
    need(lock.get("repo")=="rhasspy/piper-voices" and
         lock.get("revision")=="de3dcfdf1912bb49726dc4aa11c26017ce2ac62a" and
         lock.get("model_sha256")=="9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f" and
         lock.get("engine_version")=="1.8.0" and lock.get("speaker_count")==904,
         "PIPER_MODEL_LOCK")
    plan=read(root/"source-plan.json")
    need(plan.get("schema")=="english-ipa-piper-phoneme-probe-v1" and
         plan.get("model_revision")==lock["revision"] and
         plan.get("model_sha256")==lock["model_sha256"] and
         plan.get("training_admitted") is False and plan.get("human_gold") is False and
         plan.get("pronunciation_verified") is False,"PIPER_PLAN")
    cells=plan.get("cells")
    need(type(cells) is list and len(cells)==4,"PIPER_CELL_COUNT")
    expected={(phrase,speaker) for phrase in PHRASES for speaker in SPEAKERS}
    actual={(c.get("phrase"),c.get("source_speaker_id")) for c in cells}
    need(actual==expected,"PIPER_CELL_ROSTER")
    dirs=list((root/"cells").iterdir())
    need(len(dirs)==4 and all(d.is_dir() and not d.is_symlink() for d in dirs),"PIPER_CELL_DIRS")
    by_pair={}
    rows=[]
    for folder in dirs:
        receipt=read(folder/"completed.json")
        phrase=receipt.get("phrase")
        speaker=receipt.get("source_speaker_id")
        need((phrase,speaker) in expected and (phrase,speaker) not in by_pair,"PIPER_RECEIPT_ROSTER")
        cell=next(c for c in cells if c["phrase"]==phrase and c["source_speaker_id"]==speaker)
        need(receipt.get("schema")=="english-ipa-piper-phoneme-probe-v1" and
             receipt.get("ipa_target")==PHRASES[phrase] and receipt.get("raw_espeak_ipa")==cell["raw_espeak_ipa"] and
             receipt.get("speaker_id")==SPEAKERS[speaker] and
             receipt.get("pronunciation_verified") is False and
             receipt.get("human_gold") is False and receipt.get("training_admitted") is False,
             "PIPER_CELL_RECEIPT")
        wav=folder/"audio-16k.wav"
        size,digest=english.media()._safe_file(wav,limit=2*1024*1024)
        audio=receipt["audio"]
        need(size==audio["wav_bytes"] and digest==audio["wav_sha256"],"PIPER_WAV_SHA")
        with wave.open(str(wav),"rb") as f:
            need(f.getnchannels()==1 and f.getsampwidth()==2 and f.getframerate()==16000 and
                 f.getnframes()==audio["audio_frames"] and f.getcomptype()=="NONE","PIPER_WAV_GEOMETRY")
            pcm=f.readframes(f.getnframes())
        need(sha(pcm)==audio["pcm_sha256"],"PIPER_PCM_SHA")
        cell={**cell,"cell_id":receipt["cell_id"],"source_speaker_id":speaker,
              "phrase":phrase,"ipa":PHRASES[phrase]}
        item=(cell,wav,{"wav_sha256":digest,"frames":len(pcm)//2})
        by_pair[(phrase,speaker)]=item
        rows.append(item)
    need(set(by_pair)==expected,"PIPER_SOURCE_CLOSURE")
    return root,plan,rows

def run_provider(provider,source_parent,out):
    need(provider in PROVIDERS,"PROVIDER")
    need(os.environ.get("GITHUB_REPOSITORY")=="jiying2007/kws-data" and
         os.environ.get("GITHUB_REF")=="refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT")=="1","PROTECTED_MAIN_ATTEMPT_ONE")
    verify_trigger()
    root,plan,cells=source_cells(source_parent)
    out=Path(out).resolve()
    blind.private_path(out,"PRIVATE_OUTPUT")
    need(out.parent.is_dir() and not out.exists(),"CREATE_ONLY_OUTPUT")
    out.mkdir(mode=0o700)
    provider_id,model_id,revision=seven.PROVENANCE[provider]
    media=english.media()
    media._once(out/"started.json",{"schema":POLICY,"provider":provider,
      "provider_id":provider_id,"model_id":model_id,"model_revision_sha":revision,
      "source_run_id":SOURCE_RUN_ID,"source_artifact_id":SOURCE_ARTIFACT_ID,
      "source_artifact_sha256":SOURCE_ARTIFACT_SHA256,"call_budget":4,
      "training_admitted":False,"human_gold":False,"retry_count":0})
    model_root=out/"private-model"
    model_root.mkdir(mode=0o700)
    rows=[]
    decode=None
    load_error=None
    try:
        try: decode=seven._load(provider,model_root)
        except Exception as exc: load_error=type(exc).__name__
        for cell,wav,pcm in cells:
            folder=out/"cells"/cell["cell_id"]
            folder.mkdir(parents=True,mode=0o700)
            media._once(folder/"started.json",{"schema":POLICY,"cell_id":cell["cell_id"],
              "wav_sha256":pcm["wav_sha256"],"provider":provider,"attempt":1,"retry_count":0})
            status="FAILED_NO_RETRY"
            if decode is None:
                media._once(folder/"failed.json",{"schema":POLICY,"cell_id":cell["cell_id"],
                  "status":"MODEL_LOAD_FAILED_NO_RETRY","error_type":load_error,
                  "training_admitted":False})
            else:
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
                    media._once(folder/"completed.json",{"schema":POLICY,"cell_id":cell["cell_id"],
                      "provider":provider,"provider_id":provider_id,"wav_sha256":pcm["wav_sha256"],
                      "native_sha256":sha((folder/"native.json").read_bytes()),
                      "status":"BLIND_ASR_UNADMITTED","human_gold":False,"training_admitted":False})
                    status="COMPLETE"
                except Exception as exc:
                    media._once(folder/"failed.json",{"schema":POLICY,"cell_id":cell["cell_id"],
                      "status":"FAILED_NO_RETRY","error_type":type(exc).__name__,
                      "training_admitted":False})
            rows.append({"cell_id":cell["cell_id"],"phrase":cell["phrase"],
              "source_speaker_id":cell["source_speaker_id"],"wav_sha256":pcm["wav_sha256"],
              "status":status})
    finally:
        if model_root.exists(): shutil.rmtree(model_root)
    complete=sum(x["status"]=="COMPLETE" for x in rows)
    summary={"schema":POLICY,"status":"COMPLETE" if complete==4 else "INCOMPLETE_NO_RETRY",
      "provider":provider,"provider_id":provider_id,"model_id":model_id,
      "model_revision_sha":revision,"source_run_id":SOURCE_RUN_ID,
      "source_artifact_id":SOURCE_ARTIFACT_ID,"source_artifact_sha256":SOURCE_ARTIFACT_SHA256,
      "asr_attempts":len(rows),"completed_asr":complete,"model_load_error_type":load_error,
      "rows":rows,"training_admitted":False,"human_gold":False,"pronunciation_verified":False}
    media._once(out/"summary.json",summary)
    return summary

def combine(results_parent,source_parent,out):
    _,plan,cells=source_cells(source_parent)
    root=Path(results_parent).resolve()
    out=Path(out).resolve()
    need(not out.exists(),"COMBINED_CREATE_ONLY")
    target={c["cell_id"]:c for c,_,_ in cells}
    transcripts={}
    for provider in PROVIDERS:
        folder=root/provider
        summary=read(folder/"summary.json")
        need(summary.get("schema")==POLICY and summary.get("status")=="COMPLETE" and
             summary.get("provider")==provider and summary.get("asr_attempts")==4 and
             summary.get("completed_asr")==4 and summary.get("training_admitted") is False,
             "PROVIDER_INCOMPLETE")
        for cid in target:
            native=read(folder/"cells"/cid/"native.json")
            need(native.get("decoder_completed") is True and
                 native.get("audio_wav_sha256")==next(p[2]["wav_sha256"] for p in cells if p[0]["cell_id"]==cid),
                 "NATIVE_IDENTITY")
            transcripts.setdefault(cid,{})[provider]=native["actual_text"]
    out.mkdir(mode=0o700)
    result=[]
    for cid,providers in sorted(transcripts.items()):
        need(set(providers)==set(PROVIDERS),"ASR_ROSTER")
        c=target[cid]
        folded={p:market.normalize(providers[p],"en-US") for p in PROVIDERS}
        agree=len(set(folded.values()))==1
        target_match=agree and next(iter(folded.values()))==market.normalize(c["phrase"],"en-US")
        result.append({"cell_id":cid,"phrase":c["phrase"],"ipa":c["ipa"],
          "speaker":c["source_speaker_id"],"transcripts":providers,
          "normalized":folded,"all_three_agree":agree,"agrees_with_requested_phrase":target_match,
          "machine_verdict":"THREE_ASR_AGREE_TARGET_REVIEW_REQUIRED" if target_match else
             "MISMATCH_OR_DISAGREEMENT_REVIEW_REQUIRED",
          "acoustic_truth":False,"human_gold":False,"training_admitted":False})
    summary={"schema":POLICY,"status":"THREE_ASR_COMPLETE_REVIEW_REQUIRED",
      "source_run_id":SOURCE_RUN_ID,"source_artifact_id":SOURCE_ARTIFACT_ID,
      "providers":list(PROVIDERS),"asr_calls":12,"cells":result,
      "machine_consensus_is_not_acoustic_gold":True,"human_gold":False,"training_admitted":False}
    (out/"three-asr-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,sort_keys=True))
    return summary

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--provider",choices=PROVIDERS)
    p.add_argument("--source-parent",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--combine",type=Path)
    p.add_argument("--combine-source",type=Path)
    p.add_argument("--combined-output",type=Path)
    a=p.parse_args(argv)
    if a.combine:
        need(a.combine_source is not None and a.combined_output is not None,"COMBINE_ARGS")
        combine(a.combine,a.combine_source,a.combined_output); return
    need(a.provider and a.source_parent and a.output,"PROVIDER_ARGS")
    run_provider(a.provider,a.source_parent,a.output)

if __name__=="__main__": main()
