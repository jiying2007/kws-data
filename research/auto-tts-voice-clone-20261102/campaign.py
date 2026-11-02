"""Bounded synthetic KWS TTS/voice-clone campaign; no trusted labels.

Native source and sample integrity remain separate from real-human review,
product CTC train-source admission, and sealed device qualification.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

CAMPAIGN="auto-synthetic-kws-20261102-v1"
MODEL_FAMILIES={
    "preset":"Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "design":"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "clone":"Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
PRESET_REVISION="85e237c12c027371202489a0ec509ded67b5e4b5"
PRESETS={"Aiden","Dylan","Eric","Ono_Anna","Ryan","Serena","Sohee","Uncle_Fu","Vivian"}
TARGETS=("你好小窝","小窝小窝")
NEGATIVES=("你好小屋","你好小吴","小屋小屋","小沃小沃")
ROLES=("train-candidate","development-only")
ENGINES=("qwen06","sensevoice","whisper")
MAX_CELLS=64
MAX_BYTES=1024*1024
_ID=re.compile(r"[a-z][a-z0-9_-]{0,47}\Z")
_SHA=re.compile(r"[0-9a-f]{64}\Z")
_REV=re.compile(r"[0-9a-f]{40}\Z")
_PUNCT=re.compile(r'[\s，。！？、,.!?;；：“”"\u2018\u2019]+')


def require(cond,code):
    if not cond:raise ValueError(code)


def canonical(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"),
                      allow_nan=False).encode("utf-8")


def _uniq(items):
    out={}
    for k,v in items:
        require(k not in out,"DUPLICATE_JSON")
        out[k]=v
    return out


def _float(value):
    converted=float(value)
    require(math.isfinite(converted),"NONFINITE_JSON")
    return converted


def read_json(path,max_bytes=MAX_BYTES):
    with Path(path).open("rb") as stream:
        raw=stream.read(max_bytes+1)
    require(len(raw)<=max_bytes,"OVERSIZED_INPUT")
    try:
        return json.loads(raw.decode("utf-8"),object_pairs_hook=_uniq,
                          parse_constant=lambda x:(_ for _ in ()).throw(ValueError("NONFINITE_JSON")),
                          parse_float=_float)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise ValueError("MALFORMED_JSON") from exc


def normalize(text):
    """Typography only: 窝/屋/沃/吴/王 are never considered equivalent."""
    require(type(text) is str and 0<len(text)<=128,"TRANSCRIPT_TYPE")
    return _PUNCT.sub("",unicodedata.normalize("NFKC",text))


def _ident(text,label):
    require(type(text) is str and bool(_ID.fullmatch(text)),label)
    return text


def _sha(text,label):
    require(type(text) is str and bool(_SHA.fullmatch(text)),label)
    return text


def validate_voices(voices):
    require(type(voices) is list and 1<=len(voices)<=16,"VOICES_COUNT")
    allowed={"voice_id","mode","preset","description","reference","role",
             "source_family_id","model_id","model_revision","model_license"}
    seen, families, result=set(),{},[]
    for voice in voices:
        require(type(voice) is dict and set(voice)==allowed,"VOICE_FIELDS")
        vid=_ident(voice["voice_id"],"VOICE_ID")
        family=_ident(voice["source_family_id"],"SOURCE_FAMILY")
        require(vid not in seen,"DUPLICATE_VOICE")
        seen.add(vid)
        mode=voice["mode"]
        require(type(mode) is str and mode in MODEL_FAMILIES,"VOICE_MODE")
        require(voice["model_id"]==MODEL_FAMILIES[mode],"VOICE_MODEL")
        require(type(voice["model_revision"]) is str and
                bool(_REV.fullmatch(voice["model_revision"])),"UNPINNED_MODEL")
        require(voice["model_license"]=="Apache-2.0","MODEL_LICENSE_NOT_REVIEWED")
        role=voice["role"]
        require(type(role) is str and role in ROLES,"VOICE_ROLE")
        require(family not in families or families[family]==role,
                "SAME_ANCESTRY_CROSSES_SPLITS")
        families[family]=role
        if mode=="preset":
            require(voice["preset"] in PRESETS and voice["description"] is None and
                    voice["reference"] is None and
                    family=="qwen3-preset-"+voice["preset"].lower(),
                    "PRESET_IDENTITY_OR_LINEAGE")
        elif mode=="design":
            require(voice["preset"] is None and voice["reference"] is None and
                    type(voice["description"]) is str and
                    12<=len(voice["description"])<=240 and
                    family=="qwen3-design-"+vid,
                    "DESIGNED_VOICE_IDENTITY")
            require("模仿" not in voice["description"] and
                    "声音克隆" not in voice["description"] and
                    "真人" not in voice["description"],"PERSON_IMPERSONATION_DESCRIPTION")
        else:
            ref=voice["reference"]
            require(voice["preset"] is None and voice["description"] is None and
                    type(ref) is dict and set(ref)=={"audio_id","wav_sha256",
                      "pcm_sha256","frames","sample_rate_hz","reference_text",
                      "source_family_id","source_kind","screening_sha256",
                      "rights_basis","real_person_identity"},
                    "SYNTHETIC_REFERENCE_FIELDS")
            _ident(ref["audio_id"],"REFERENCE_ID")
            _sha(ref["wav_sha256"],"REFERENCE_WAV_ID")
            _sha(ref["pcm_sha256"],"REFERENCE_PCM_ID")
            _sha(ref["screening_sha256"],"REFERENCE_SCREEN_ID")
            require(type(ref["frames"]) is int and ref["frames"]>0 and
                    type(ref["sample_rate_hz"]) is int and
                    ref["sample_rate_hz"] in (16000,24000),
                    "REFERENCE_GEOMETRY")
            require(type(ref["reference_text"]) is str and
                    normalize(ref["reference_text"]) in TARGETS and
                    ref["source_kind"]=="synthesized_and_machine_screened" and
                    ref["rights_basis"]=="synthetic_model_output_research_only" and
                    ref["real_person_identity"] is None and
                    ref["source_family_id"]==family,
                    "NON_SYNTHETIC_OR_UNSCREENED_REFERENCE")
        result.append(voice)
    return result


def build_plan(voices, *, round_id=0, batch_id="batch01"):
    validate_voices(voices)
    require(type(round_id) is int and round_id in (0,1),"ROUNDS_ARE_BOUNDED")
    _ident(batch_id,"BATCH_ID")
    # K1 gets two punctuation variants; K2 and phonetically similar negatives
    # remain explicit, rather than assuming a changed prompt is the same sound.
    phrases=[("你好小窝","wake",1,"exact"),("你好，小窝","wake",1,"punctuated"),
             ("小窝小窝","wake",2,"exact"),
             ("你好小屋","confusable",None,"exact"),("你好小吴","confusable",None,"exact"),
             ("小屋小屋","confusable",None,"exact")]
    cells=[]
    for v in voices:
        for index,(prompt,kind,kw,variant) in enumerate(phrases):
            if round_id==1 and kind=="confusable":
                continue  # one bounded refinement is for poorly produced positives
            identity=hashlib.sha256(canonical({"batch_id":batch_id,
                        "round":round_id,"voice_id":v["voice_id"],"prompt":prompt})).hexdigest()
            seed=1000000+int(identity[:8],16)%1000000000
            cells.append({"cell_id":"c-"+identity[:24],"voice_id":v["voice_id"],
                "source_family_id":v["source_family_id"],"role":v["role"],
                "mode":v["mode"],"model_id":v["model_id"],
                "model_revision":v["model_revision"],"seed":seed,
                "text":prompt,"target_normalized":normalize(prompt),"kind":kind,
                "keyword_id":kw,"variant":variant,"round":round_id,
                "status":"NOT_RUN","synthetic":True,"actual_spoken_text":None,
                "model_calls":0,"human_gold":False,"training_admitted":False,
                "qualification_allowed":False})
    require(len(cells)<=MAX_CELLS and
            len({c["cell_id"] for c in cells})==len(cells),"CAMPAIGN_BUDGET")
    cells.sort(key=lambda c:c["cell_id"])
    result={"schema":CAMPAIGN,"batch_id":batch_id,"round_id":round_id,
        "voices":voices,"cells":cells,
        "budget":{"max_tts_calls":len(cells),"max_asr_calls":3*len(cells),
            "max_automatic_rounds":2,"max_retries_per_cell":0,"max_paid_usd":0},
        "source_scope":"synthetic_technical_development_only",
        "blind_asr_targets_prohibited":True,"training_admitted":False,
        "qualification_allowed":False,"real_human_qualification":False,
        "shipping_approved":False}
    return result


def validate_plan(plan):
    require(type(plan) is dict and set(plan)=={
        "schema","batch_id","round_id","voices","cells","budget","source_scope",
        "blind_asr_targets_prohibited","training_admitted","qualification_allowed",
        "real_human_qualification","shipping_approved"} and
        plan["schema"]==CAMPAIGN,"PLAN_SCHEMA")
    expected=build_plan(plan["voices"],round_id=plan["round_id"],batch_id=plan["batch_id"])
    require(canonical(plan)==canonical(expected),"PLAN_BYTES_OR_POLICY_DRIFT")
    return plan


def example_voices():
    # Four official presets not in prior Qwen12 selection. Fixed source family
    # includes the original preset: another seed is NOT another person.
    roles={"Ryan":"train-candidate","Aiden":"train-candidate",
           "Sohee":"train-candidate","Ono_Anna":"development-only"}
    return [{"voice_id":name.lower().replace("_","-"),"mode":"preset",
        "preset":name,"description":None,"reference":None,
        "source_family_id":"qwen3-preset-"+name.lower(),
        "role":role,"model_id":MODEL_FAMILIES["preset"],
        "model_revision":PRESET_REVISION,"model_license":"Apache-2.0"}
        for name,role in roles.items()]


def main(argv=None):
    """Offline plan entry; no model, network, reviewer, or authority calls."""
    import argparse
    import os
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--voices",type=Path)
    parser.add_argument("--batch-id",default="default01")
    parser.add_argument("--round",dest="round_id",type=int,choices=(0,1),default=0)
    args=parser.parse_args(argv)
    try:
        voices=read_json(args.voices) if args.voices else example_voices()
        plan=build_plan(voices,round_id=args.round_id,batch_id=args.batch_id)
        require(args.output.parent.is_dir() and
                not args.output.exists() and not args.output.is_symlink(),
                "PLAN_OUTPUT_MUST_BE_NEW")
        fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(canonical(plan)+b"\n")
            stream.flush();os.fsync(stream.fileno())
    except (OSError,ValueError) as error:
        print("SYNTHETIC_PLAN_NOT_CREATED: "+type(error).__name__,file=sys.stderr)
        return 2
    print(json.dumps({"plan_sha256":hashlib.sha256(canonical(plan)).hexdigest(),
        "planned_tts_calls":len(plan["cells"]),"training_admitted":False},
        sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
