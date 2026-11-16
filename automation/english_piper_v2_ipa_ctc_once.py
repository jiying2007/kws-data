"""One-shot independent CTC phoneme observation on four exact Piper WAVs.

The phoneme model receives only PCM. It never receives target phrase, IPA,
ASR transcripts, speaker identity or split metadata. Output is machine-only;
it is not acoustic gold or training admission.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, wave
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_qwen_capture as english
import multilingual_campaign as market
import english63_independent_ipa as ipa

POLICY="english-ipa-piper-v2-independent-ctc-observation-v1"
SOURCE_RUN_ID=43789385672
SOURCE_ARTIFACT_ID=14191228436
SOURCE_ARTIFACT_NAME="english-ipa-piper-probe-43789385672"
SOURCE_ARTIFACT_SHA256="8ab7ff273ce5ee98e4c4bd83f43fd80b82c7ae03ac89449ded93a4cd61bedc61"
MODEL_ID=ipa.MODEL_ID
MODEL_REV=ipa.MODEL_REV
MODEL_SHA256=ipa.WEIGHTS_SHA256
PHRASES={"Hey Mero":"/heɪ ˈmiːroʊ/","Hey Vosen":"/heɪ ˈvoʊzən/"}
SPEAKERS={"p3922":0,"p8699":1}
MAX_FORWARDS=4
TRIGGER=Path(__file__).resolve().parents[1]/".github/triggers/english-ipa-piper-v2-independent-ctc-20261116.json"

def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write_once(path,data):
    path=Path(path)
    need(not path.exists(),"CREATE_ONLY_OUTPUT")
    with path.open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2); f.write("\n")
    path.chmod(0o600)

def verify_trigger():
    expected={"schema":POLICY,"source_run_id":SOURCE_RUN_ID,
      "source_artifact_id":SOURCE_ARTIFACT_ID,"source_artifact_name":SOURCE_ARTIFACT_NAME,
      "source_artifact_sha256":SOURCE_ARTIFACT_SHA256,"model_id":MODEL_ID,
      "model_revision":MODEL_REV,"model_safetensors_sha256":MODEL_SHA256,
      "max_new_phoneme_forwards":MAX_FORWARDS,"new_tts_calls":0,
      "new_asr_calls":0,"new_kws_model_forwards":0,"new_optimizer_steps":0,
      "human_gold":False,"training_admitted":False,"no_scheduled_tasks":True}
    need(read(TRIGGER)==expected,"TRIGGER_DRIFT")
    return expected

def source_cells(parent):
    parent=Path(parent).resolve()
    need(parent.is_dir() and not parent.is_symlink(),"SOURCE_PARENT")
    matches=[p for p in parent.rglob("result.json") if p.parent.name=="probe"]
    need(len(matches)==1,"PIPER_ROOT_UNIQUE")
    root=matches[0].parent
    result=read(root/"result.json")
    need(result.get("schema")=="english-ipa-piper-phoneme-probe-v2" and
         result.get("status")=="COMPLETE" and result.get("attempts")==4 and
         result.get("completed")==4 and result.get("asr_calls")==0 and
         result.get("training_admitted") is False and result.get("human_gold") is False and
         result.get("pronunciation_verified") is False,"PIPER_NOT_COMPLETE")
    lock=read(root/"model-lock.json")
    need(lock.get("schema")=="english-ipa-piper-model-lock-v1" and lock.get("repo")=="rhasspy/piper-voices" and
         lock.get("revision")=="de3dcfdf1912bb49726dc4aa11c26017ce2ac62a" and
         lock.get("model_sha256")=="9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f" and
         lock.get("engine_version")=="1.8.0","PIPER_LOCK")
    plan=read(root/"source-plan.json")
    need(plan.get("schema")=="english-ipa-piper-phoneme-probe-v2" and
         all(c.get("raw_espeak_ipa")=={"Hey Mero":"hˈeɪ mˈiːɹoʊ","Hey Vosen":"hˈeɪ vˈoʊzən"}[c.get("phrase")] for c in plan.get("cells",[])) and
         plan.get("model_revision")==lock["revision"] and plan.get("model_sha256")==lock["model_sha256"] and
         plan.get("training_admitted") is False and plan.get("human_gold") is False and
         plan.get("pronunciation_verified") is False,"PIPER_PLAN")
    cells=plan.get("cells")
    expected={(phrase,speaker) for phrase in PHRASES for speaker in SPEAKERS}
    need(type(cells) is list and len(cells)==4 and
         {(c.get("phrase"),c.get("source_speaker_id")) for c in cells}==expected and
         all(c.get("ipa")==PHRASES[c.get("phrase")] and
             c.get("raw_espeak_ipa")=={"Hey Mero":"hˈeɪ mˈiːɹoʊ","Hey Vosen":"hˈeɪ vˈoʊzən"}[c.get("phrase")] and
             c.get("speaker_id")==SPEAKERS[c.get("source_speaker_id")] for c in cells),
         "PIPER_PLAN_ROSTER_OR_FINAL_IPA")
    folders=list((root/"cells").iterdir())
    need(len(folders)==4 and all(x.is_dir() and not x.is_symlink() for x in folders),
         "PIPER_CELL_DIRS")
    rows=[]; seen=set()
    for folder in folders:
        receipt=read(folder/"completed.json")
        phrase=receipt.get("phrase"); speaker=receipt.get("source_speaker_id")
        need((phrase,speaker) in expected and (phrase,speaker) not in seen,"PIPER_RECEIPT_ROSTER")
        seen.add((phrase,speaker))
        cell=next(c for c in cells if c["phrase"]==phrase and c["source_speaker_id"]==speaker)
        need(receipt.get("schema")=="english-ipa-piper-phoneme-probe-v2" and
             receipt.get("ipa_target")==PHRASES[phrase] and
             receipt.get("raw_espeak_ipa")==cell["raw_espeak_ipa"] and
             receipt.get("speaker_id")==SPEAKERS[speaker] and
             receipt.get("training_admitted") is False and receipt.get("human_gold") is False and
             receipt.get("pronunciation_verified") is False,"PIPER_CELL_RECEIPT")
        wav=folder/"audio-16k.wav"
        size,digest=english.media()._safe_file(wav,limit=2*1024*1024)
        audio=receipt["audio"]
        need(size==audio["wav_bytes"] and digest==audio["wav_sha256"],"PIPER_WAV_HASH")
        with wave.open(str(wav),"rb") as f:
            need(f.getframerate()==16000 and f.getnchannels()==1 and f.getsampwidth()==2 and
                 f.getnframes()==audio["audio_frames"] and f.getcomptype()=="NONE","PIPER_WAV_GEOMETRY")
            pcm=f.readframes(f.getnframes())
        need(sha(pcm)==audio["pcm_sha256"],"PIPER_PCM_HASH")
        rows.append({"cell_id":receipt["cell_id"],"phrase":phrase,"ipa":PHRASES[phrase],
          "speaker":speaker,"wav":wav,"wav_sha256":digest,"frames":len(pcm)//2})
    need(len(seen)==4,"PIPER_SOURCE_CLOSURE")
    return root,rows

def execute(source_parent,out):
    need(os.environ.get("GITHUB_REPOSITORY")=="jiying2007/kws-data" and
         os.environ.get("GITHUB_EVENT_NAME")=="push" and
         os.environ.get("GITHUB_REF")=="refs/heads/main" and
         os.environ.get("GITHUB_RUN_ATTEMPT")=="1","PROTECTED_MAIN_ATTEMPT_ONE")
    verify_trigger()
    root,rows=source_cells(source_parent)
    out=Path(out).resolve()
    ipa.protected_main()
    need(out.is_absolute() and out.parent.is_dir() and not out.exists() and
         not any(p.is_symlink() for p in (out,*out.parents)),"PRIVATE_CREATE_ONLY_OUTPUT")
    out.mkdir(mode=0o700)
    model_root=out/"private-model"; model_root.mkdir(mode=0o700)
    result={"schema":POLICY,"status":"FAILED_NO_RETRY","source_run_id":SOURCE_RUN_ID,
      "source_artifact_id":SOURCE_ARTIFACT_ID,"source_artifact_sha256":SOURCE_ARTIFACT_SHA256,
      "model_id":MODEL_ID,"model_revision":MODEL_REV,"model_safetensors_sha256":MODEL_SHA256,
      "attempted_forwards":0,"completed_forwards":0,"human_gold":False,"training_admitted":False}
    observations=[]
    try:
        processor,model,torch,asset=ipa.load_model_from_pinned_assets(model_root)
        write_once(out/"model-lock.json",asset)
        for cell in rows:
            result["attempted_forwards"]+=1
            raw,phones,frames=ipa.predict_phone(cell["wav"],processor,model,torch)
            need(frames==cell["frames"],"PHONEME_FRAME_BINDING")
            observations.append({"cell_id":cell["cell_id"],"source_wav_sha256":cell["wav_sha256"],
              "source_frames":frames,"predicted_raw_ipa_tokens":raw,
              "predicted_stressless_phone_tokens":phones,"target_phrase":cell["phrase"],
              "target_ipa":cell["ipa"],"target_is_not_passed_to_model":True,
              "machine_observation_only":True,"pronunciation_verified":False,
              "human_gold":False,"training_admitted":False})
            result["completed_forwards"]+=1
        result.update({"status":"COMPLETE" if result["completed_forwards"]==4 else "INCOMPLETE_NO_RETRY",
          "model_manifest_sha256":asset["manifest_sha256"],
          "observations":observations,"no_tts_calls":True,"no_asr_calls":True,
          "no_kws_model_forwards":True,"no_optimizer_steps":True})
        write_once(out/"result.json",result)
    except BaseException as exc:
        result.update({"status":"FAILED_NO_RETRY","error_type":type(exc).__name__,
          "completed_forwards":len(observations),"observations":observations})
        write_once(out/"result.json",result)
        raise
    finally:
        if model_root.exists(): shutil.rmtree(model_root)
    return result

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--source-parent",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args(argv)
    verify_trigger()
    if not a.execute:
        print(json.dumps({"schema":POLICY,"max_forwards":MAX_FORWARDS,
          "executed":0,"training_admitted":False}))
        return 0
    need(a.source_parent is not None and a.output is not None,"EXPLICIT_INPUTS_REQUIRED")
    r=execute(a.source_parent,a.output)
    print(json.dumps({"status":r["status"],"attempted_forwards":r["attempted_forwards"],
      "completed_forwards":r["completed_forwards"],"training_admitted":False},sort_keys=True))
    return 0 if r["status"]=="COMPLETE" else 2

if __name__=="__main__": raise SystemExit(main())
