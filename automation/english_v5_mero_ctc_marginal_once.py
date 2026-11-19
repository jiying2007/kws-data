"""V5 Hey Mero frozen-weight CTC marginal-vs-blank diagnostic, research only.

Uses EXACT previous frozen model, original source-screen ZIPs and the same 17
train/dev samples. One additional bounded set of 17 model forwards; zero
new TTS/ASR/phoneme forwards, optimizer steps, new calibration or locked tests.
This tests only whether full-target CTC posterior separates source-disjoint
development positives from ordinary negatives. No learned threshold or
deployable event decoder is produced by this diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REPO="jiying2007/kws-data"
BEFORE="29cc99a44231b1842ed42833689958abfc8be11e"
SCHEMA="english-v5-mero-frozen-ctc-fullseq-vs-blank-v1"
OLD_RUN=44453447586
FORWARDS=17

import importlib.util


def load(name):
    spec=importlib.util.spec_from_file_location(
        "v5_ctc_marginal_"+name,HERE/(name+".py"))
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


saved=load("english_v5_mero_saved_diagnostic_once")
v5=saved.v5
binding=saved.binding
prior=saved.prior


def need(ok,message):
    if not ok:raise ValueError(message)


def source_check():
    observed=saved.source_check()
    need(observed["original_model_artifact"]==14496732088
         and observed["max_new_KWS_forwards"]==17
         and observed["new_optimizer_updates"]==0
         and observed["old_calibration_or_locked_replay"] is False
         and FORWARDS==17
         and v5.config()["active_keyword_count"]==1
         and v5.config()["target"]["ctc_token_ids"]==[1,2,3],
         "SAME_ACTUAL_V5_SINGLE_MERO_FROZEN_SOURCE_AND_17_PCM")
    return {"status":"PINNED_FROZEN_MERO_FULL_SEQUENCE_DIAG",
            "original_model_run":OLD_RUN,"model_artifact":14496732088,
            "max_new_model_forwards":FORWARDS,
            "new_optimizer_updates":0,"new_generated_audio":0,
            "calibration_or_locked_used":False}


def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),
        "ONE_FIRST_PUBLIC_MAIN_EVENT_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch(r"[a-f0-9]{40}",head) is not None and
         re.fullmatch(r"[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "ACTUAL_PUBLIC_MAIN_RUN_RECEIPT")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False
         and event.get("deleted") is False,
         "EXACT_MAIN_SUCCESSOR_NOT_ORIGINAL_REPLAY")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                       cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"EXPECTED_MAIN_PARENT_ONLY")
    return {"run_id":env["GITHUB_RUN_ID"],"head":head,
            "original_weight_run":OLD_RUN}


def ctc_marginal(logits,torch):
    """Exact complete-clip CTC sequence-vs-blank log likelihood ratio.

    This is NOT a causal online event score, does not use an operating
    threshold, and is computed for already-exposed train/dev only.
    """
    need(logits.ndim==2 and logits.shape[1]==4 and
         logits.shape[0]>3 and bool(torch.isfinite(logits).all()),
         "FINITE_FOUR_OUTPUT_WHOLE_UTTERANCE_CTC")
    logp=logits.log_softmax(-1)
    target=torch.tensor([1,2,3],dtype=torch.long)
    loss=torch.nn.functional.ctc_loss(
        logp.unsqueeze(1),target,
        torch.tensor([len(logp)],dtype=torch.long),
        torch.tensor([len(target)],dtype=torch.long),
        blank=0,reduction="none",zero_infinity=False)[0]
    need(bool(torch.isfinite(loss)),
         "COMPLETE_CTC_TARGET_CANNOT_BE_SCORED")
    blank=logp[:,0].sum()
    margin=(-loss-blank).item()
    need(math.isfinite(margin),"FINITE_TARGET_VS_BLANK_MARGINAL")
    return {"target_log_probability":float(-loss.item()),
            "all_blank_log_probability":float(blank.item()),
            "target_minus_all_blank":float(margin),
            "margin_per_frame":float(margin/logits.shape[0]),
            "max_roh_frame_probability":float(logp[:,3].max().exp().item()),
            "blank_argmax_fraction":float(
                 (logits.argmax(-1)==0).float().mean().item())}


def separate(values):
    pos=[x for x in values if x["mono_class"]==1]
    neg=[x for x in values if x["mono_class"]==0]
    need(pos and neg,"BOTH_CLASSES_PRESENT_WITHOUT_HOLDOUT")
    p=[x["margin"]["target_minus_all_blank"] for x in pos]
    n=[x["margin"]["target_minus_all_blank"] for x in neg]
    pairs=[int(a>b)+0.5*int(a==b) for a in p for b in n]
    return {
       "positive_count":len(p),"ordinary_count":len(n),
       "positive_margin_min":min(p),"positive_margin_mean":sum(p)/len(p),
       "positive_margin_max":max(p),
       "ordinary_margin_min":min(n),"ordinary_margin_mean":sum(n)/len(n),
       "ordinary_margin_max":max(n),
       "strict_margin_separation_observed":min(p)>max(n),
       "exploratory_pairwise_rank_fraction":sum(pairs)/len(pairs),
       "calibration_performed":False,
       "release_threshold_selected":False}


def review(model_zip,sources,out):
    identity=guard()
    out=Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "NEW_ONLY_ONCE_OUTPUT_DIR")
    out.mkdir()
    binding.once(out/"STARTED.json",{
        "schema":SCHEMA,"run":identity,
        "input_model_artifact":saved.MODEL_ARTIFACT,
        "new_model_forward_calls":0,
        "new_optimizer_updates":0})
    archives={}
    z=None
    try:
        saved.check_original_git(os.environ.get("GH_TOKEN"))
        z,result,train_receipt,intake=saved.verified_model_packet(model_zip)
        rows,audit=v5.admission(Path(sources))
        need(rows==intake["rows"] and
             len(rows)==35 and
             Counter(x["role"] for x in rows)==
             {"train":24,"development":11},
             "RECOMPUTED_ORIGINAL_V5_REVIEWED_TRAIN_AND_DEV")
        dev=[x for x in rows if x["role"]=="development"]
        train=saved.choose_train_diagnostics(rows)
        selected=dev+train
        need(len(dev)==11 and len(train)==6 and
             len(selected)==FORWARDS and
             len({x["id"] for x in selected})==FORWARDS,
             "NO_DUPLICATE_OR_LOCKED_PCM_IN_17_REVIEWS")
        import torch
        import numpy as np
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        model=v5.build_model(torch)
        state=torch.load(io.BytesIO(
            z.read("models/mero-fsmn-ctc-v5.pt")),
            weights_only=True,map_location="cpu")
        model.load_state_dict(state,strict=True)
        model.eval()
        with np.load(io.BytesIO(z.read(
                "models/mero-fsmn-ctc-v5.npz")),allow_pickle=False) as features:
            bank=features["mel_filters"].copy()
            mean=features["train_mean"].copy()
            scale=features["train_scale"].copy()
            for name,value in state.items():
                need(name in features.files and np.array_equal(
                    features[name],value.detach().cpu().numpy()),
                    "EXACT_SAVED_PT_NPZ_WEIGHT_ROUNDTRIP")
        need(np.array_equal(bank,v5.make_mel_bank(np))
             and mean.shape==scale.shape==(40,)
             and np.isfinite(mean).all()
             and np.isfinite(scale).all()
             and bool((scale>0).all()),
             "EXACT_TRAIN_ONLY_LOGMEL_NORMALIZER")
        for spec in binding.config()["corpus"]:
            key=spec["key"]
            archives[key]=binding.archive(
                Path(sources)/(key+"-source.zip"),spec["source"])
        archives["newtrain24"]=binding.archive(
            Path(sources)/"newtrain24-source.zip",prior.SOURCE)
        observations=[]
        with torch.no_grad():
            for row in selected:
                pcm=v5.base.load_source_pcm(row,archives)
                arr=v5.features(pcm,row,torch,np,bank)
                norm=((arr-mean)/scale).astype(np.float32)
                logits=model(torch.from_numpy(norm).unsqueeze(0))[0]
                m=ctc_marginal(logits,torch)
                greedy=v5.ctc_greedy(logits,torch)
                observations.append({
                    "id":row["id"],"source_group":row["source_group_id"],
                    "role":row["role"],"mono_class":row["mono_class"],
                    "original_pcm_sha256":row["source_pcm_sha256"],
                    "frames":int(logits.shape[0]),
                    "greedy_event":bool(greedy["mero_event"]),
                    "greedy_tokens":greedy["collapsed_tokens"],
                    "margin":m})
        train_obs=[x for x in observations if x["role"]=="train"]
        dev_obs=[x for x in observations if x["role"]=="development"]
        develop=separate(dev_obs)
        train_result=separate(train_obs)
        status=("DEV_MARGINAL_STRICTLY_SEPARATES_EXPOSED_6_5"
                if develop["strict_margin_separation_observed"]
                else "DEV_MARGINAL_NOT_SEPARATED")
        report={
            "schema":SCHEMA,"status":status,
            "original_model_run":OLD_RUN,
            "original_model_artifact_id":saved.MODEL_ARTIFACT["id"],
            "original_saved_weight_sha256":
                result["artifacts"]["mero-fsmn-ctc-v5.onnx"]["sha256"],
            "model_forward_calls":len(observations),
            "new_optimizer_updates":0,
            "new_generated_audio_calls":0,
            "training_samples_evaluated":len(train_obs),
            "development_samples_evaluated":len(dev_obs),
            "dev":develop,"train_diagnostic":train_result,
            "rows":observations,
            "predeclared_operating_threshold":None,
            "new_model_weights":False,
            "is_streaming_event_detector":False,
            "reused_exposed_dev_for_diagnosis_only":True,
            "old_calibration_or_locked_reopened":False,
            "human_gold":False,
            "real_frr_far_or_board_measurement":False}
        binding.once(out/"RESULT.json",report)
        print(json.dumps({k:v for k,v in report.items() if k!="rows"},
                         sort_keys=True))
    except BaseException as err:
        binding.once(out/"STOP.json",{
            "schema":SCHEMA,"status":"STOP_NO_REPLAY_OR_AUTOPROMOTION",
            "error":type(err).__name__+":"+str(err)[:500],
            "optimizer_updates":0,"old_locked_reopened":False})
        raise
    finally:
        if z is not None:z.close()
        for a in archives.values():a.close()


def main():
    parser=argparse.ArgumentParser()
    for mode in ("--source-check","--guard","--audit-model","--diagnose"):
        parser.add_argument(mode,action="store_true")
    parser.add_argument("--model-zip")
    parser.add_argument("--sources")
    parser.add_argument("--output")
    opts=parser.parse_args()
    need(sum((opts.source_check,opts.guard,opts.audit_model,opts.diagnose))==1,
         "ONE_FROZEN_CTC_MARGINAL_STAGE")
    if opts.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if opts.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(opts.model_zip,"AUTHENTIC_ORIGINAL_MODEL_ZIP_REQUIRED")
    if opts.audit_model:
        print(json.dumps(saved.audit_model(opts.model_zip),sort_keys=True))
        return
    need(opts.sources and opts.output,
         "ORIGINAL_ONLY_TRAIN_AND_EXPOSED_DEV_REQUIRED")
    review(opts.model_zip,opts.sources,opts.output)


if __name__=="__main__":
    main()
