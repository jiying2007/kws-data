"""ASR agreement as machine screening (never human-gold or release authority).

Receipts are from three native inference adapters, independently seeing only an
opaque source audio identity. No intended text, keyword or TTS prompt is passed
to the ASR workers. This module never invokes a recognizer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import wave

import campaign as c
import receipt_contract as receipts
from runner import _safe_file, _once, _hash, validate_reference


SCHEMA="blind-native-asr-observation-v1"
FIELDS={"schema","engine","model_id","model_revision","audio_id",
        "wav_sha256","pcm_sha256","audio_frames","status",
        "decoder_completed","processed_start_frame","processed_end_frame",
        "actual_text","raw_native_sha256","attempts"}
MODEL_IDS={"qwen06":"Qwen/Qwen3-ASR-0.6B",
           "sensevoice":"FunAudioLLM/SenseVoiceSmall",
           "whisper":"openai/whisper-small"}
REVIEW_LIMIT=64


def _audio_metadata(path):
    size,digest=_safe_file(path,limit=3*1024*1024)
    with wave.open(str(path),"rb") as stream:
        c.require((stream.getnchannels(),stream.getsampwidth(),stream.getframerate(),
                   stream.getcomptype())==(1,2,16000,"NONE"),"INVALID_ASR_AUDIO_FORMAT")
        frames=stream.getnframes()
        payload=stream.readframes(frames)
        c.require(not stream.readframes(1),"ASR_AUDIO_TRAILING_BYTES")
    c.require(len(payload)==2*frames and size==len(payload)+44,
              "ASR_AUDIO_NONCANONICAL")
    return {"wav_sha256":digest,"pcm_sha256":_hash(payload),
            "audio_frames":frames,"wav_bytes":size}


def validate_native_observation(obs,cell,generated):
    c.require(type(obs) is dict and set(obs)==FIELDS and
              obs["schema"]==SCHEMA and obs["engine"] in c.ENGINES,
              "ASR_RECEIPT_FIELDS")
    c.require(obs["audio_id"]==cell["cell_id"],"ASR_AUDIO_ID")
    c.require(obs["model_id"]==MODEL_IDS[obs["engine"]] and
              type(obs["model_revision"]) is str and c._REV.fullmatch(obs["model_revision"]),
              "ASR_MODEL_IDENTITY")
    for k in ("wav_sha256","pcm_sha256"):
        c.require(obs[k]==generated[k],"ASR_AUDIO_HASH")
    c.require(type(obs["audio_frames"]) is int and
              obs["audio_frames"]==generated["audio_frames"],
              "ASR_INPUT_LENGTH")
    c.require(obs["status"]=="success" and obs["decoder_completed"] is True and
              type(obs["processed_start_frame"]) is int and
              obs["processed_start_frame"]==0 and
              type(obs["processed_end_frame"]) is int and
              obs["processed_end_frame"]==generated["audio_frames"],
              "INCOMPLETE_ASR_FORWARD")
    c.require(type(obs["attempts"]) is int and obs["attempts"]==1,"ASR_REPLAY")
    c.require(type(obs["raw_native_sha256"]) is str and
              c._SHA.fullmatch(obs["raw_native_sha256"]),"NATIVE_RESULT_DIGEST")
    c.normalize(obs["actual_text"])
    return obs


def screen_cell(plan_cell,output_root,asr_root,*,adapters,expected_plan_sha256):
    models=receipts.check_adapters(adapters)
    expected={"plan_sha256":expected_plan_sha256,
              "adapter_lock_sha256":_hash(c.canonical(adapters))}
    receipts.verify_root(asr_root,expected,"asr-autonomous-lock-v1")
    root=Path(output_root)/"cells"/plan_cell["cell_id"]
    done=root/"completed.json"
    if not done.exists():
        return {"cell_id":plan_cell["cell_id"],"status":"NO_TTS_RESULT_OR_FAILURE",
                "review_required":False}
    complete=c.read_json(done)
    c.require(type(complete) is dict and
              complete.get("cell_id")==plan_cell["cell_id"] and
              complete.get("schema")=="tts-generated-cell-completed-v1" and
              complete.get("status")=="GENERATED_UNADMITTED" and
              complete.get("training_admitted") is False and
              complete.get("qualification_allowed") is False and
              complete.get("plan_sha256")==expected_plan_sha256,"TTS_RECEIPT_INVALID")
    generated=complete["audio"]
    measured=_audio_metadata(root/"audio-16k.wav")
    c.require(all(generated.get(k)==measured.get(k) for k in
                  ("wav_sha256","pcm_sha256","audio_frames","wav_bytes")),
              "TTS_AUDIO_CHANGED")
    observations={}
    for engine in c.ENGINES:
        path=Path(asr_root)/plan_cell["cell_id"]/(engine+".json")
        if not path.exists():
            continue
        obs=validate_native_observation(c.read_json(path,65536),plan_cell,measured)
        c.require(obs["engine"]==engine,"ASR_ENGINE_SIDECAR_DRIFT")
        receipts.verify_native(Path(asr_root)/plan_cell["cell_id"]/engine,
            obs,models[engine],expected,started_schema="asr-native-started-v1")
        observations[engine]=obs
    result={"cell_id":plan_cell["cell_id"],"source_family_id":plan_cell["source_family_id"],
            "role":plan_cell["role"],"mode":plan_cell["mode"],
            "kind":plan_cell["kind"],"keyword_id":plan_cell["keyword_id"],
            "observed_engines":len(observations),"asr_texts":{
                k:observations[k]["actual_text"] for k in sorted(observations)},
            "technical_audio_pass":True,
            "automated_label_is_human_gold":False,
            "training_admitted":False,"qualification_allowed":False,
            "review_required":False}
    if len(observations)<3:
        result["status"]="INCOMPLETE_ASR_HOLD"
        return result
    normalized=[c.normalize(observations[e]["actual_text"]) for e in c.ENGINES]
    expected=plan_cell["target_normalized"]
    hits=sum(t==expected for t in normalized)
    # A genuine 3/3 exact text match is only an automated technical screen.
    if hits==3 and plan_cell["kind"]=="wake":
        result["status"]="MACHINE_SCREENED_POSITIVE_CANDIDATE"
    elif hits==3 and plan_cell["kind"]=="confusable":
        result["status"]="MACHINE_SCREENED_CONFUSABLE_CANDIDATE"
    elif hits==2:
        result["status"]="DISAGREEMENT_REVIEW_OR_NEW_GENERATION"
        result["review_required"]=True
    else:
        result["status"]="REJECT_TTS_PRONUNCIATION_OR_ASR_DISAGREEMENT"
        result["review_required"]=False
    return result


def screen_campaign(plan,generated_root,asr_root,adapters):
    c.validate_plan(plan)
    expected=receipts.binding(plan,adapters)
    receipts.verify_root(asr_root,expected,"asr-autonomous-lock-v1")
    outputs=[screen_cell(cell,generated_root,asr_root,adapters=adapters,
             expected_plan_sha256=expected["plan_sha256"]) for cell in plan["cells"]]
    assert len(outputs)==len(plan["cells"])
    counts={status:sum(x["status"]==status for x in outputs)
            for status in sorted({x["status"] for x in outputs})}
    return {"schema":"autonomous-synthetic-kws-screen-v1",
        "plan_sha256":expected["plan_sha256"],
        "adapter_lock_sha256":expected["adapter_lock_sha256"],
        "rows":outputs,"counts":counts,
        "human_gold":False,"training_admitted":False,
        "qualification_allowed":False,
        "real_human_qualification":False,
        "status":"MACHINE_SCREENING_DEVELOPMENT_ONLY"}


def plan_refinement(plan,report):
    """One bounded positive-only development refinement. Never retry a consumed cell."""
    c.validate_plan(plan)
    c.require(plan["round_id"]==0 and report["schema"]==
              "autonomous-synthetic-kws-screen-v1" and
              report["plan_sha256"]==_hash(c.canonical(plan)) and
              report["training_admitted"] is False,
              "INVALID_REFINEMENT_PARENT")
    voices={v["voice_id"]:v for v in plan["voices"]}
    all_cells={r["cell_id"]:r for r in plan["cells"]}
    row_ids=[r["cell_id"] for r in report["rows"]]
    c.require(len(row_ids)==len(set(row_ids))==len(plan["cells"]) and
              set(row_ids)==set(all_cells),"REFINEMENT_REPORT_ROSTER")
    # Refine a voice only when at least one K1 variant was machine-rejected.
    # No 2/3 majority is made a positive label; all outputs remain R&D only.
    selected=set()
    for row in report["rows"]:
        cell=all_cells[row["cell_id"]]
        if (cell["keyword_id"]==1 and row["status"] in (
                "REJECT_TTS_PRONUNCIATION_OR_ASR_DISAGREEMENT",
                "DISAGREEMENT_REVIEW_OR_NEW_GENERATION") and
                cell["role"]=="train-candidate"):
            selected.add(cell["voice_id"])
    if not selected:return None
    return build_adaptive([voices[n] for n in sorted(selected)],plan["batch_id"])


def build_adaptive(selected,batch):
    # A separate one-shot run ID. Not a replay of an earlier cell/seed.
    # The source-family ancestry remains the same; it is not a new speaker.
    return c.build_plan(selected,round_id=1,batch_id=batch)


def create_clone_reference(cell,screened,generated_root,destination):
    """Create a reference from auto-screened *synthetic* voice; no humans."""
    c.require(cell["kind"]=="wake" and cell["mode"] in ("preset","design") and
              screened["cell_id"]==cell["cell_id"] and
              screened["status"]=="MACHINE_SCREENED_POSITIVE_CANDIDATE" and
              screened["observed_engines"]==3 and
              screened["training_admitted"] is False,
              "NOT_A_SCREENED_SYNTHETIC_REFERENCE")
    source=Path(generated_root)/"cells"/cell["cell_id"]/"audio-16k.wav"
    metadata=_audio_metadata(source)
    dst=Path(destination)
    c.require(not dst.is_symlink() and
              not any(p.is_symlink() for p in dst.parents),
              "REFERENCE_DIRECTORY_ALIAS")
    dst.mkdir(parents=True,exist_ok=True)
    target=dst/(cell["cell_id"]+".wav")
    c.require(not target.exists(),"REFERENCE_AUDIO_ALREADY_EXISTS")
    with source.open("rb") as src,target.open("xb") as out:
        for block in iter(lambda:src.read(1024*1024),b""):
            out.write(block)
    c.require(_audio_metadata(target)==metadata,"COPIED_REFERENCE_DRIFT")
    screening={"schema":"synthetic-clone-reference-screen-v1",
        "status":"MACHINE_SCREENED_REFERENCE_ONLY","audio_id":cell["cell_id"],
        "wav_sha256":metadata["wav_sha256"],
        "pcm_sha256":metadata["pcm_sha256"],
        "source_family_id":cell["source_family_id"],
        "normalized_actual_text":cell["target_normalized"],
        "engines":list(c.ENGINES),"machine_only":True,"training_admitted":False}
    _once(dst/(cell["cell_id"]+".screen.json"),screening)
    receipt={"audio_id":cell["cell_id"],"wav_sha256":metadata["wav_sha256"],
        "pcm_sha256":metadata["pcm_sha256"],"frames":metadata["audio_frames"],
        "sample_rate_hz":16000,"reference_text":cell["target_normalized"],
        "source_family_id":cell["source_family_id"],
        "source_kind":"synthesized_and_machine_screened",
        "screening_sha256":_safe_file(dst/(cell["cell_id"]+".screen.json"),
            limit=65536)[1],
        "rights_basis":"synthetic_model_output_research_only",
        "real_person_identity":None,"training_admitted":False}
    _once(dst/(cell["cell_id"]+".reference.json"),receipt)
    # The permitted clone remains same ancestry, cannot populate a new split.
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True)
    p.add_argument("--generated-root",required=True)
    p.add_argument("--asr-root",required=True)
    p.add_argument("--adapter-lock",required=True)
    p.add_argument("--report",required=True)
    p.add_argument("--refinement")
    args=p.parse_args(argv)
    try:
        plan=c.validate_plan(c.read_json(args.plan))
        result=screen_campaign(plan,args.generated_root,args.asr_root,
                               c.read_json(args.adapter_lock))
        _once(Path(args.report),result)
        if args.refinement:
            next_plan=plan_refinement(plan,result)
            if next_plan is not None:
                _once(Path(args.refinement),next_plan)
    except (ValueError,OSError) as err:
        print("ASR_SCREEN_NOT_ADMITTED: "+type(err).__name__,file=__import__("sys").stderr)
        return 2
    print(json.dumps({"counts":result["counts"],"human_gold":False,
        "training_admitted":False},ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
