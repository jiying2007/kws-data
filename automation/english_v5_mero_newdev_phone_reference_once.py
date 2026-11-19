"""V5 Hey Mero: frozen pretrained-phone challenger on 4 exposed NEW-DEV PCM.

The newly trained 43,876-param FSMN from run 44495594627 detected only 1/2
source-disjoint synthetic development positives. Compare a DISTINCT official
Zh-En 3M phone-lexicon KWS reference with unchanged fixed decoder settings;
never tune from these already-exposed development results. No original
training/calibration/locked audio, no new WAV, optimizer, TTS, ASR or KWS
locked-test evaluation. This is a bounded acoustic-diagnostic comparison,
not a product qualification or lightweight-model promotion.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(name):
    spec=importlib.util.spec_from_file_location("en_v5_newdev_phone_"+name,HERE/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

f=load("english_v5_mero_fresh24_plus70_fit_once")
ref=load("english_v5_mero_pretrained_reference_once")
SCHEMA="english-v5-mero-fresh96-exposed-newdev-frozen-phone-reference-v1"
REPO="jiying2007/kws-data"
BEFORE="d0b567fe37147aae6ab5d7bdec3721b879a0b22d"
MODEL_RUN=44495594627
MODEL_ZIP=14514228406
MODEL_ZIP_SHA="sha256:91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702"
DEVICE_CLASS="DIAGNOSTIC_ONLY_NOT_SSC305"
DECODER={"score":1.0,"threshold":0.25,"num_trailing_blanks":1,
         "max_active_paths":4,"num_threads":1,"sample_rate":16000,
         "feature_dim":80,"provider":"cpu"}
TEST_COUNT=4
POS_COUNT=2
NEG_COUNT=2
MODE="zh-en"

def need(ok,reason):
    if not ok:
        raise ValueError(reason)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def emit(path,value):
    with Path(path).open("x",encoding="utf-8") as fh:
        json.dump(value,fh,ensure_ascii=False,sort_keys=True,
                  indent=2,allow_nan=False)
        fh.write("\n")

def source_check():
    prior=f.source_check()
    baseline=ref.source_contract()
    spec=ref.pretrained.MODEL_SPECS[MODE]
    need(prior["target"]=="Hey Mero"
         and prior["model_parameters"]==43876
         and prior["new_source_qualified_train_groups"]==11
         and prior["new_source_qualified_dev_groups"]==2
         and prior["calibration_or_locked_kws_audio_opened"] is False
         and baseline["reference_model"]==MODE
         and baseline["keyword_ids"]==["hey_mero"]
         and baseline["calibration_or_locked_access"] is False
         and baseline["model_promotion_allowed"] is False
         and spec["archive_sha256"]==ref.MODEL_SHA256
         and spec["archive_bytes"]==ref.MODEL_BYTES==32885699
         and ref.MODEL_SHA256=="68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
         and ref.KEYWORD=="HEY_MERO"
         and f.FRESH_SOURCE["id"]==14507208152
         and f.FRESH_SCREEN["id"]==14509252921
         and f.FULL_GATE["id"]==14512954554
         and f.FRESH_QUALIFIED=={"train":11,"development":2}
         and TEST_COUNT==POS_COUNT+NEG_COUNT==4
         and DECODER["threshold"]==0.25
         and DECODER["num_trailing_blanks"]==1,
         "EXACT_V5_FROZEN_FOUR_NEWDEV_AND_PRETRAINED_PHONE_REFERENCE")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_KWS_INFERENCE",
            "single_keyword":"Hey Mero","model":"official-zh-en-phone-3m",
            "model_sha256":ref.MODEL_SHA256,
            "target_counts":{"positive":POS_COUNT,"ordinary":NEG_COUNT},
            "new_exposed_dev_source_groups":2,
            "max_new_kws_forwards":4,"new_optimizer_updates":0,
            "new_asr_phone_model_calls":0,"new_tts_calls":0,
            "threshold_frozen":DECODER["threshold"],
            "calibration_or_locked_audio_opened":False,
            "model_promotion_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    expected={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in expected.items()),
         "ONLY_FIRST_PUBLIC_MAIN_VALID_EXPERIMENT")
    head=env.get("GITHUB_SHA","")
    need(bool(re.fullmatch("[0-9a-f]{40}",head))
         and bool(re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID",""))),
         "BOUNDED_PUBLIC_RUN_ID")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "EXACT_SUCCESSOR_NOT_A_REPEAT_OR_OTHER_REPO")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                      cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"NO_OUT_OF_ORDER_MERO_REFERENCE_REPLAY")
    return {"run_id":env["GITHUB_RUN_ID"],"head":head,
            "new_kws_calls_allowed":TEST_COUNT,
            "model_promotion_allowed":False}

def audited_new_dev(newsrc,screen,gate):
    source_check()
    g=f.frozen_gate(gate)
    need(g["status"]=="PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE",
         "ORIGINAL_MACHINE_GATE_ONLY_NOT_HUMAN_QUALIFIED")
    members,proof=f.screen_new(newsrc,screen)
    rows=sorted((x for x in members if x["role"]=="development"),
                key=lambda r:r["id"])
    need(len(rows)==TEST_COUNT
         and Counter(r["mono_class"] for r in rows)=={0:NEG_COUNT,1:POS_COUNT}
         and len({r["source_group_id"] for r in rows})==2
         and all(r["origin"]=="new_frozen_piper_source96"
                 and r["research_dev_only"] is True
                 and r["research_training_bind"] is False
                 and r["tier"] in ("silver","weak")
                 for r in rows)
         and not {r["id"] for r in rows}&
                 {r["id"] for r in members if r["role"]=="train"},
         "EXACT_MACHINE_SCREENED_TWO_NEW_DEV_POS_ORD_NO_LEAK")
    selected=[]
    for row in rows:
        pcm=f.readpcm(row,{},newsrc)
        need(len(pcm)==2*row["samples"]
             and sha(pcm)==row["source_pcm_sha256"],
             "EXACT_ORIGINAL_UNMODIFIED_COMPLETE_DEV_PCM16")
        selected.append({k:row[k] for k in (
            "id","source_group_id","speaker_index","role",
            "mono_class","source_pcm_sha256","tier","samples")})
    return rows,{"source_artifact":f.FRESH_SOURCE,
                 "screen_artifact":f.FRESH_SCREEN,
                 "machine_gate_artifact":f.FULL_GATE,
                 "selected":selected,
                 "no_train_calibration_locked_pcm":True,
                 "source_qa_only_is_not_product_qualification":True}

def audit(newsrc,screen,gate):
    _,result=audited_new_dev(newsrc,screen,gate)
    return {"status":"SAVED_FOUR_EXPOSED_NEWDEV_AUDIO_AUTHENTICATED",
            "originals":result,"numerical_kws_forwards":0,
            "optimizer_updates":0}

def execute(newsrc,screen,gate,model_archive,output):
    import numpy as np
    import sherpa_onnx
    identity=guard()
    rows,proof=audited_new_dev(newsrc,screen,gate)
    need(output.is_absolute() and output.parent.is_dir()
         and not output.exists() and not output.parent.is_symlink(),
         "ONE_CREATE_ONLY_DIAGNOSTIC_DIR")
    output.mkdir()
    emit(output/"STARTED.json",{"schema":SCHEMA,"run":identity,
         "new_kws_forwards_limit":TEST_COUNT,"model":MODE,
         "model_sha256":ref.MODEL_SHA256,
         "never_calibration_or_locked":True})
    try:
        need(model_archive.is_absolute() and model_archive.is_file()
             and model_archive.stat().st_size==ref.MODEL_BYTES
             and ref.hash_file(model_archive)==ref.MODEL_SHA256,
             "EXACT_OFFICIAL_PINNED_PRETRAINED_PHONE_3M_BYTES")
        import tempfile
        with tempfile.TemporaryDirectory(prefix="v5-newdev-phone-") as tmp:
            root=Path(tmp)/"model";root.mkdir()
            parts,digest=ref.pretrained.extract_locked_model(
                model_archive,MODE,root)
            need(digest==ref.MODEL_SHA256,"EXACT_MODEL_ARCHIVE_HASH")
            keyword=ref.make_mero_keyword(parts,Path(tmp))
            spotter=sherpa_onnx.KeywordSpotter(
                tokens=str(parts["tokens"]), encoder=str(parts["encoder"]),
                decoder=str(parts["decoder"]),joiner=str(parts["joiner"]),
                keywords_file=str(keyword),
                num_threads=DECODER["num_threads"],
                sample_rate=DECODER["sample_rate"],
                feature_dim=DECODER["feature_dim"],
                max_active_paths=DECODER["max_active_paths"],
                keywords_score=DECODER["score"],
                keywords_threshold=DECODER["threshold"],
                num_trailing_blanks=DECODER["num_trailing_blanks"],
                provider=DECODER["provider"])
            outcomes=[]
            t0=time.perf_counter()
            for row in rows:
                pcm=f.readpcm(row,{},newsrc)
                samples=np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768
                events,steps=ref.pretrained.stream_events(spotter,samples)
                need(all(ev["mapped_keyword"]==ref.KEYWORD for ev in events),
                     "ONLY_EXACT_CANONICAL_MERO_EVENT")
                outcomes.append({"id":row["id"],"role":row["role"],
                   "source_group_id":row["source_group_id"],
                   "pcm_sha256":row["source_pcm_sha256"],
                   "positive":bool(row["mono_class"]),
                   "detected":bool(events),
                   "event_count":len(events),"decoder_steps":steps,
                   "original_samples":row["samples"],
                   "events":events})
            positives=[x for x in outcomes if x["positive"]]
            ordinary=[x for x in outcomes if not x["positive"]]
            need(len(positives)==POS_COUNT and len(ordinary)==NEG_COUNT,
                 "EXACT_TWO_NEWDEV_POSITIVE_TWO_ORDINARY")
            report={"schema":SCHEMA,
                "status":"REAL_EXPOSED_NEWDEV_PHONE_REFERENCE_DIAGNOSTIC",
                "previous_fsmn_newdev_hits":1,
                "model":"frozen_official_zh_en_phone_zipformer_3M",
                "official_model_sha256":ref.MODEL_SHA256,
                "fixed_decoder":DECODER,
                "newdev_positive":len(positives),
                "pretrained_positive_hits":sum(x["detected"] for x in positives),
                "newdev_ordinary":len(ordinary),
                "pretrained_ordinary_events":sum(x["event_count"] for x in ordinary),
                "original_source_receipt":proof,
                "source_disjoint_from_model_train":True,
                "synthetic_development_already_exposed_to_fsmn":True,
                "dev_used_for_training_or_threshold_selection":False,
                "original_clips":outcomes,
                "elapsed_host_s":round(time.perf_counter()-t0,4),
                "new_kws_forwards":TEST_COUNT,
                "new_asr_phone_tts_forwards":0,
                "optimizer_steps":0,
                "model_weights_created":False,
                "human_far_frr":"NOT_MEASURED",
                "ssc305":"NOT_RUN",
                "full_product_promotion_allowed":False}
            emit(output/"RESULT.json",report)
            print(json.dumps({k:report[k] for k in (
                "status","newdev_positive","pretrained_positive_hits",
                "newdev_ordinary","pretrained_ordinary_events",
                "new_kws_forwards","full_product_promotion_allowed")},
                sort_keys=True))
    except BaseException as e:
        emit(output/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_PHONE_REFERENCE_NO_INDEPENDENT_QUALITY_PROMOTION",
             "error_type":type(e).__name__,"reason":str(e)[:450],
             "no_old_locked_model_inference":True,
             "no_retry":True})
        raise

def main():
    p=argparse.ArgumentParser()
    for x in ("--source-check","--guard","--audit","--execute"):
        p.add_argument(x,action="store_true")
    p.add_argument("--new-source",type=Path)
    p.add_argument("--new-screen",type=Path)
    p.add_argument("--source-gate",type=Path)
    p.add_argument("--model-archive",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.audit,a.execute))==1,
         "ONE_EXACT_PRETRAINED_COMPARISON_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(all((a.new_source,a.new_screen,a.source_gate)),
         "ORIGINAL_NEW_SOURCE_SCREEN_GATE_REQUIRED")
    if a.audit:
        print(json.dumps(audit(a.new_source,a.new_screen,a.source_gate),
                         sort_keys=True));return
    need(a.model_archive and a.output,"PRETRAINED_MODEL_AND_SAVE_DIR_REQUIRED")
    execute(a.new_source,a.new_screen,a.source_gate,a.model_archive,a.output)

if __name__=="__main__":
    main()
