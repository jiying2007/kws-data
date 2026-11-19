"""Engineering-only frozen 43,876-param Hey Mero ONNX/PT numeric parity.

No new optimizer, TTS, ASR or quality tuning. Reuse ten already exposed,
source-authenticated 480..487 prospective DEV original WAVs. Compare exact
PT-vs-ONNX frame logit, greedy CTC and event parity, not a new FAR/FRR test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import importlib.util

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("mero43_parity_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj
prior=load("english_v5_mero_prospective_three_frozen_once")
SCHEMA="english-v5-mero43-original-pcm-onnx-pt-parity-v1"
REPO="jiying2007/kws-data"
BEFORE="1c8b8c616ad912e5c615fae957e89f7d4b0234fd"
CANDIDATE="fsmn43k256"
MAX_CLIPS=10
MAX_ABS_LOGIT=0.001
MAX_MEAN_ABS_LOGIT=0.0001
ONNX_SHA256="10bf30da57291f78f56db196db1137309f2c3b91dfe67bf0aa45ebb8cec05b1c"

def need(x,code):
    if not x:raise ValueError(code)
def save(path,data):
    with Path(path).open("x",encoding="utf-8") as handle:
        json.dump(data,handle,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write("\n")

def source_check():
    c=prior.source_check()
    r=prior.MODELS[CANDIDATE]
    need(c["keyword"]=="Hey Mero" and c["new_dev_positive"]==2
         and c["new_dev_ordinary"]==8
         and c["optimizer_steps"]==0
         and c["product_promotion_allowed"] is False
         and r["id"]==14514228406 and r["run"]==44495594627
         and r["steps"]==256 and r["params"]==43876
         and r["onnx"]==ONNX_SHA256
         and "pt" not in r
         and prior.PHONEMODEL_SHA=="68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
         and MAX_CLIPS==10
         and MAX_ABS_LOGIT==0.001
         and MAX_MEAN_ABS_LOGIT==0.0001,
         "FROZEN_ORIGINAL_43K_ONNX_NUMERIC_COMPARISON_ONLY")
    return {"status":"SOURCE_ONLY_PT_ONNX_PARITY_NOT_RUN",
        "schema":SCHEMA,"keyword":"Hey Mero",
        "source":prior.SOURCE["id"],"saved_machine_screen":prior.SCREEN["id"],
        "original_model_artifact":r["id"],
        "model_onnx_sha256":ONNX_SHA256,
        "clipped_original_wavs":0,"original_dev_audio_count":MAX_CLIPS,
        "max_abs_logit_tolerance":MAX_ABS_LOGIT,
        "max_mean_abs_logit_tolerance":MAX_MEAN_ABS_LOGIT,
        "new_optimizer_calls":0,"new_asr_tts_calls":0,
        "new_dev_heldout_for_selection":False,
        "new_locked_or_calibration_forwards":0,"product_promotion":False}
def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"ONLY_FIRST_PUBLIC_MAIN_PARITY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "EXACT_MAIN_SHA_AND_RUN_ID")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "ONE_NONREPLAYED_MODEL_EXPORT_PARITY")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                     cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"EXACT_MODEL_EXPORT_MAIN_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"new_optimizer_calls":0,
            "locked_or_calibration_opened":False}

def audit(source,screen):
    rows,receipt=prior.select(source,screen)
    need(len(rows)==MAX_CLIPS and sum(r["mono_class"] for r in rows)==2
         and len({r["source_group_id"] for r in rows})==8
         and receipt["no_locked_or_calibration"] is True,
         "ORIGINAL_IDENTICAL_TEN_PROSPECTIVE_DEV_PCM")
    return rows,receipt

def execute(source,screen,weights,output):
    import numpy as np
    import torch
    import onnxruntime as ort
    identity=guard()
    need(output.is_absolute() and output.parent.is_dir()
         and not output.exists(),"CREATE_ONCE_PARITY_OUTPUT")
    output.mkdir()
    save(output/"STARTED.json",{
        "schema":SCHEMA,"run":identity,
        "same_original_audio":True,"no_new_model_weights":True})
    try:
        artifact=prior.provenance(os.environ.get("GH_TOKEN"))
        originals,source_receipt=audit(source,screen)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        model,bank,mean,scale,pins=prior.model(torch,np,weights,CANDIDATE)
        need(pins["onnx"]==ONNX_SHA256,
             "EXACT_EXPORTED_ONNX_SHA_TO_TEST")
        so=ort.SessionOptions()
        so.intra_op_num_threads=1;so.inter_op_num_threads=1
        so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        path=(weights/"models"/(prior.MODELS[CANDIDATE]["nameprefix"]+".onnx"))
        session=ort.InferenceSession(str(path),sess_options=so,
                                     providers=["CPUExecutionProvider"])
        need(len(session.get_inputs())==1 and session.get_inputs()[0].name=="fbank40"
             and len(session.get_outputs())==1
             and session.get_outputs()[0].name=="frame_ctc_logits",
             "EXACT_ONNX_FRAME_LOGIT_ABI")
        records=[]
        begin=time.perf_counter()
        with torch.no_grad():
            for row in originals:
                data=prior.pcm(row,source)
                mel=prior.v5.features(data,row,torch,np,bank)
                x=((mel-mean)/scale).astype(np.float32)[None,:,:]
                pt=model(torch.from_numpy(x)).detach().cpu().numpy()
                ort_result=session.run(["frame_ctc_logits"],{"fbank40":x})[0]
                need(pt.shape==ort_result.shape
                     and pt.shape[0]==1 and pt.shape[2]==4
                     and pt.shape[1]>35
                     and np.isfinite(pt).all() and np.isfinite(ort_result).all(),
                     "IDENTICAL_FINITE_CAUSAL_FSMN_FRAME_SHAPE")
                diff=np.abs(pt.astype(np.float64)-ort_result.astype(np.float64))
                maximum=float(diff.max());average=float(diff.mean())
                tokens_equal=bool(np.array_equal(pt.argmax(-1),ort_result.argmax(-1)))
                pt_ctc=prior.v5.ctc_greedy(torch.from_numpy(pt),torch)
                ort_ctc=prior.v5.ctc_greedy(torch.from_numpy(ort_result),torch)
                events_equal=(pt_ctc["mero_event"]==ort_ctc["mero_event"]
                              and pt_ctc["extra_events"]==ort_ctc["extra_events"])
                need(maximum<=MAX_ABS_LOGIT and average<=MAX_MEAN_ABS_LOGIT
                     and tokens_equal and events_equal,
                     "STOP_MODEL_EXPORT_NOT_NUMERICALLY_PARITY")
                records.append({"id":row["id"],"role":"development",
                    "source_group_id":row["source_group_id"],
                    "original_pcm_sha256":row["source_pcm_sha256"],
                    "positive_machine_only":bool(row["mono_class"]),
                    "frame_count":pt.shape[1],
                    "max_abs_logit":maximum,"mean_abs_logit":average,
                    "argmax_frame_parity":tokens_equal,
                    "complete_ctc_event_parity":events_equal,
                    "pt_event":bool(pt_ctc["mero_event"]),
                    "onnx_event":bool(ort_ctc["mero_event"])})
        need(len(records)==10
             and sum(r["positive_machine_only"] for r in records)==2
             and all(r["argmax_frame_parity"] and r["complete_ctc_event_parity"]
                     for r in records),"FULL_TEN_UNMODIFIED_ORIGINALS_REQUIRED")
        report={"schema":SCHEMA,
            "status":"PASS_FROZEN_43K_PT_ONNX_HOST_NUMERIC_PARITY",
            "models":"one_unchanged_43876_param_fsmn_ctc",
            "model_weights_sha256":pins,
            "original_research_artifacts":artifact,
            "source_audit":source_receipt,
            "original_development_audio_replayed_for_export_parity_only":True,
            "abs_tolerance":MAX_ABS_LOGIT,
            "mean_abs_tolerance":MAX_MEAN_ABS_LOGIT,
            "maximum_abs_observed":max(r["max_abs_logit"] for r in records),
            "all_argmax_and_full_ctc_events_equal":True,
            "observed_original_clips":records,
            "host_seconds":round(time.perf_counter()-begin,3),
            "no_optimizer_or_new_data":True,
            "real_long_negative_far":"NOT_MEASURED",
            "arm32_c_inference":"NOT_RUN",
            "ssc305":"NOT_RUN",
            "product_promotion_allowed":False}
        save(output/"RESULT.json",report)
        print(json.dumps({"status":report["status"],
            "original_wavs":len(records),
            "maximum_abs_logit":report["maximum_abs_observed"],
            "pt_onnx_event_parity":True,
            "product_promotion_allowed":False},sort_keys=True))
    except BaseException as e:
        save(output/"STOP.json",{"schema":SCHEMA,
            "status":"STOP_ONNX_PARITY_NOT_QUALIFIED",
            "exception_type":type(e).__name__,
            "reason":str(e)[:480],"replay_forbidden":True,
            "calibration_locked_access":False})
        raise

def main():
    p=argparse.ArgumentParser()
    for flag in ("--source-check","--guard","--audit","--execute"):
        p.add_argument(flag,action="store_true")
    for name in ("--source","--screen","--weights","--output"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.audit,a.execute))==1,
         "ONLY_ONE_MODEL_PARITY_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(a.source and a.screen,"ORIGINAL_DEV_AUDIT_REQUIRED")
    if a.audit:
        _,r=audit(a.source,a.screen)
        print(json.dumps(r,sort_keys=True));return
    need(a.weights and a.output,"FROZEN_WEIGHT_DIRECTORY_OUTPUT_REQUIRED")
    execute(a.source,a.screen,a.weights,a.output)
if __name__=="__main__":main()
