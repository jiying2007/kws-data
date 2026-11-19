"""One bounded train-only gain-invariant refit of the failed mono70 FSMN.

Acoustic diagnosis (EXPOSED dev only): two p1112 Silver Mero clips output only
HEY at original RMS, and full HEY-MEH-ROH under 1.5x diagnostic gain.
Do not normalize those dev clips or tune an inference gain from them. Instead,
modify only ORIGINAL TRAIN logmel power, using a fixed bounded gain grid,
train-only parent clean posterior consistency, and one frozen final checkpoint.

Old model weights remain immutable, new model bytes are separate. No human FAR,
old locked audio, unknown/other keyword labels, TTS/ASR or Work/Codex.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import time
import zipfile

HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location(
    "en_v5_mono70_fixed_parent",HERE/"english_v5_mero_mono70_robust_train_once.py")
parent=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)
v5=parent.v5
REPO="jiying2007/kws-data"
BRANCH="research/english-v5-mero-gain-invariant-refit-20261119"
SCHEMA="english-v5-mero-train-only-gain-invariant-refit-v1"
PARENT_RUN=44471889167
PARENT_ARTIFACT=14504422674
PARENT_ZIP_SHA="1c6c96129fe0223a92e09b6141d70c90ff72f7a3bd8144e29827f0cddaa476d0"
PARENT_PT_SHA="483c7ede52f2308a461202da7823af38d1353e98287aba3f926705b1e0ea9fd3"
PARENT_NPZ_SHA="9d55f12a80566b9ce7926d9059cc7230c9309a53823d4002d1d02e8a663b5c11"
PARENT_ONNX_SHA="a1446d6b67d9e4e78cf97537b95b7b912493a39e9fde042b2cbb50889602aaba"
GAIN_LEVELS=(0.5,0.75,1.0,1.5,2.0)
SEED=20261122
UPDATES=144
LR=0.0001


def need(x,reason):
    if not x:raise ValueError(reason)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def write_once(path,data):
    with path.open("x",encoding="utf-8") as f:
        json.dump(data,f,sort_keys=True,ensure_ascii=False,indent=2,
                  allow_nan=False)
        f.write("\n")

def source_check():
    old=parent.source_check()
    need(old["target"]=="Hey Mero"
         and old["source_audit_run"]==44470829024
         and old["product_release"] is False
         and old["old_locked_audio"]=="NEVER_OPENED"
         and parent.HIDDEN_PARAMS==43876
         and parent.TOTAL_SOURCE==70
         and len(GAIN_LEVELS)==5 and GAIN_LEVELS[2]==1.0
         and PARENT_RUN==44471889167
         and PARENT_ARTIFACT==14504422674
         and all(re.fullmatch("[0-9a-f]{64}",s)
                 for s in (PARENT_ZIP_SHA,PARENT_PT_SHA,
                           PARENT_NPZ_SHA,PARENT_ONNX_SHA))
         and UPDATES==144 and SEED==20261122 and LR==0.0001,
         "PINNED_70PCM_PARENT_AND_GAIN_ONLY_RESEARCH")
    return {"schema":SCHEMA,"status":"TRAIN_ONLY_GAIN_HYPOTHESIS_NOT_EXECUTED",
       "only_keyword":"Hey Mero","frozen_parent_run":PARENT_RUN,
       "frozen_parent_artifact":PARENT_ARTIFACT,
       "parent_zip_sha256":PARENT_ZIP_SHA,
       "gain_levels":GAIN_LEVELS,"new_optimizer_updates":0,
       "new_synthesis_or_asr":0,"dev_audio_in_training":False,
       "calibration_or_locked_opened":False,
       "formal_promotion_allowed":False}

def run_guard(env=None,event=None):
    env=os.environ if env is None else env
    check={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
           "GITHUB_EVENT_NAME":"pull_request",
           "GITHUB_HEAD_REF":BRANCH,"GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in check.items()),"PR_ONLY_FIRST_EXECUTION")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    pr=event.get("pull_request",{})
    need(event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and pr.get("head",{}).get("repo",{}).get("full_name")==REPO
         and pr.get("head",{}).get("ref")==BRANCH
         and pr.get("base",{}).get("ref")=="main"
         and event.get("action") in ("opened","synchronize")
         and re.fullmatch("[0-9a-f]{40}",pr.get("head",{}).get("sha",""))
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "EXACT_FRESH_PUBLIC_MERO_PR_FIRST_ATTEMPT")
    return {"run":env["GITHUB_RUN_ID"],"head":pr["head"]["sha"],
            "branch":BRANCH}


def parent_packet(zip_path):
    need(zip_path.is_absolute() and zip_path.is_file()
         and not zip_path.is_symlink()
         and sha(zip_path.read_bytes())==PARENT_ZIP_SHA,
         "EXACT_REAL_REJECTED_PARENT_ZIP_BYTES")
    archive=zipfile.ZipFile(zip_path)
    try:
        names=set(archive.namelist())
        expected={"STARTED.json","ADMISSION.json","TRAINING.json","RESULT.json",
           "models/mero-mono70-fsmn-ctc.pt",
           "models/mero-mono70-fsmn-ctc.npz",
           "models/mero-mono70-fsmn-ctc.onnx"}
        need(names==expected,"ONLY_ORIGINAL_SEVEN_MODEL_RECEIPTS")
        report=json.loads(archive.read("RESULT.json"))
        need(report["schema"]==parent.SCHEMA
             and report["status"]=="REJECTED_NOT_NEW_MODEL_QUALITY_QUALIFIED"
             and report["parameters"]==43876
             and report["real_optimizer_updates"]==360
             and report["train_original_pcm"]==54
             and report["dev_original_pcm"]==16
             and report["dev"]["positive_original"]==9
             and report["dev"]["positive_misses"]==2
             and report["dev"]["ordinary_false_or_extra"]==0
             and report["product_promotion_allowed"] is False,
             "EXACT_SAVED_FAILED_360_STEP_MODEL_NOT_PROMOTED")
        files={}
        for name,digest in (
          ("models/mero-mono70-fsmn-ctc.pt",PARENT_PT_SHA),
          ("models/mero-mono70-fsmn-ctc.npz",PARENT_NPZ_SHA),
          ("models/mero-mono70-fsmn-ctc.onnx",PARENT_ONNX_SHA)):
            files[name]=archive.read(name)
            need(sha(files[name])==digest,"PARENT_MODEL_HASH_MISMATCH")
        return files,report
    finally:
        archive.close()


def derived_level(log_mel,gain,np):
    """Exact log1p(gain² * mel-power) for a linear PCM-level change.

    No waveform crop, parent identity overwrite or developer-selected gain.
    Used ONLY for original TRAIN features; never on DEVELOPMENT.
    """
    need(gain in GAIN_LEVELS and log_mel.ndim==2
         and log_mel.shape[1]==40 and np.isfinite(log_mel).all(),
         "TRAIN_ONLY_CAUSAL_MEL_AND_BOUNDED_LEVEL")
    power=np.maximum(np.expm1(log_mel.astype(np.float64)),0)
    adjusted=np.log1p(power*(gain*gain)).astype(np.float32)
    need(adjusted.shape==log_mel.shape and np.isfinite(adjusted).all(),
         "FINITE_ORIGINAL_TRAIN_GAIN_MEL")
    return adjusted


def frozen_train_and_eval(rows,source_zips,model_files,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    train=sorted((x for x in rows if x["role"]=="train"),key=lambda x:x["id"])
    dev=sorted((x for x in rows if x["role"]=="development"),key=lambda x:x["id"])
    need(len(train)==54 and len(dev)==16
         and len({x["source_group_id"] for x in train})==24
         and len({x["source_group_id"] for x in dev})==7
         and not ({x["source_group_id"] for x in train} &
                  {x["source_group_id"] for x in dev}),
         "FROZEN_SOURCE_GROUP_DISJOINT_GAIN_REFIT")
    with np.load(io.BytesIO(model_files["models/mero-mono70-fsmn-ctc.npz"]),
                 allow_pickle=False) as a:
        bank=a["mel_filters"].copy()
        mean=a["train_mean"].copy()
        scale=a["train_scale"].copy()
    need(bank.shape==(257,40) and mean.shape==(40,) and
         scale.shape==(40,) and (scale>0).all(),
         "FROZEN_PREVIOUS_TRAIN_ONLY_MEL_NORMALIZATION")
    baseline=v5.build_model(torch)
    state=torch.load(io.BytesIO(
        model_files["models/mero-mono70-fsmn-ctc.pt"]),
        map_location="cpu",weights_only=True)
    baseline.load_state_dict(state,strict=True)
    baseline.eval()
    for param in baseline.parameters():param.requires_grad_(False)
    updated=v5.build_model(torch)
    updated.load_state_dict(state,strict=True)
    groups=parent.groups_by_class(train)
    # Only ORIGINAL TRAIN full PCM is processed into numerical optimizers.
    cache={}
    torch_bank=torch.from_numpy(bank)
    for row in train+dev:
        pcm=v5.base.load_source_pcm(row,source_zips)
        need(sha(pcm)==row["source_pcm_sha256"],"REAL_BYTE_IDENTICAL_SOURCE_PCM")
        cache[row["id"]]=v5.features(
            pcm,row,torch,np,torch_bank)
    teacher={}
    with torch.no_grad():
        for row in train:
            x=torch.from_numpy(((cache[row["id"]]-mean)/scale)
                .astype(np.float32)).unsqueeze(0)
            teacher[row["id"]]=baseline(x).softmax(-1).detach()
    optimizer=torch.optim.AdamW(updated.parameters(),
       lr=LR,weight_decay=0.001)
    trace=[];began=time.perf_counter()
    for step in range(UPDATES):
        updated.train()
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for cls in (0,1):
            names=sorted(groups[cls])
            group=names[step%len(names)]
            candidates=sorted(groups[cls][group],key=lambda r:r["id"])
            row=candidates[(step//len(names))%len(candidates)]
            # Different source groups see a fixed deterministic gain grid.
            # Every class gets low-level, unchanged and boosted TRAIN views.
            idx=(step//len(names) + int(sha(row["source_pcm_sha256"].encode())[:2],16))%len(GAIN_LEVELS)
            gain=GAIN_LEVELS[idx]
            x=derived_level(cache[row["id"]],gain,np)
            normalized=torch.from_numpy(((x-mean)/scale)
                .astype(np.float32)).unsqueeze(0)
            logits=updated(normalized)
            seq,blank,margin=parent.ctc_losses(logits,torch)
            if cls:
                supervised=seq+F.softplus(0.5-4.0*margin)
            else:
                supervised=0.35*blank+F.softplus(0.5+4.0*margin)
            if row["tier"]=="weak":
                supervised=supervised*0.75
            # Train-only consistency: frozen previous student's clean
            # posteriors vs this student's gained view, no dev teacher.
            kl=F.kl_div(logits.log_softmax(-1),teacher[row["id"]],
                        reduction="none").sum(-1).mean()
            objective=supervised+0.2*kl
            need(bool(torch.isfinite(objective)), "FINITE_LEVEL_CONSISTENCY_LOSS")
            losses.append(objective)
        total=sum(losses)/2
        total.backward()
        torch.nn.utils.clip_grad_norm_(updated.parameters(),5.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        if step%12==0 or step==UPDATES-1:
            trace.append({"step":step+1,"loss":float(total.detach()),
                          "ordinary":float(losses[0].detach()),
                          "positive":float(losses[1].detach())})
    updated.eval()
    directory=out/"models";directory.mkdir()
    weights=directory/"mero-mono70-gain-refit.pt"
    torch.save(updated.state_dict(),weights)
    readback=v5.build_model(torch)
    readback.load_state_dict(
        torch.load(weights,map_location="cpu",weights_only=True),strict=True)
    readback.eval()
    with torch.no_grad():
        probe=torch.from_numpy(((cache[dev[0]["id"]]-mean)/scale)
                .astype(np.float32)).unsqueeze(0)
        need(bool(torch.equal(updated(probe),readback(probe))),
             "FROZEN_RELOADED_GAIN_MODEL_NUMERIC_PARITY")
    npz=directory/"mero-mono70-gain-refit.npz"
    np.savez(npz,mel_filters=bank,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy()
                for k,v in readback.state_dict().items()})
    onnx=directory/"mero-mono70-gain-refit.onnx"
    torch.onnx.export(readback,probe,onnx.as_posix(),
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        opset_version=17,dynamo=False,
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(onnx.exists() and onnx.stat().st_size>10000,
         "ACTUAL_FINITE_GAIN_RETRAINED_ONNX")
    # One final dev-only numerical replay of unchanged 16 PCM. Never normalize
    # or boost the exposed dev WAVs; no threshold or model selection from them.
    observations=[]
    with torch.no_grad():
        for row in dev:
            feat=torch.from_numpy(((cache[row["id"]]-mean)/scale)
                 .astype(np.float32)).unsqueeze(0)
            old=v5.ctc_greedy(baseline(feat),torch)
            new=v5.ctc_greedy(readback(feat),torch)
            _,_,margin=parent.ctc_losses(readback(feat),torch)
            observations.append({"id":row["id"],"role":"development",
                "source_group_id":row["source_group_id"],
                "pcm_sha256":row["source_pcm_sha256"],
                "positive":bool(row["mono_class"]),
                "parent_event":old["mero_event"],
                "new_event":new["mero_event"],
                "extra_events":new["extra_events"],
                "full_ctc_margin_per_frame":float(margin.detach())})
    pos=[r for r in observations if r["positive"]]
    neg=[r for r in observations if not r["positive"]]
    missed=sum(not r["new_event"] for r in pos)
    false=sum(r["new_event"]+r["extra_events"] for r in neg)
    need(len(pos)==9 and len(neg)==7
         and sum(not r["parent_event"] for r in pos)==2,
         "PARENT_EXACT_TWO_OF_NINE_DEV_MISSES_REFERENCE")
    provisional=(missed==0 and false==0)
    result={"schema":SCHEMA,
      "status":("PROVISIONAL_EXPOSED_SYNTHETIC_DEV_ALL_HIT"
                if provisional else "GAIN_RESEARCH_NOT_QUALIFIED"),
      "parent_model_artifact_id":PARENT_ARTIFACT,
      "parent_model_status":"REJECTED_ORIGINAL_2_OF_9_MISSED",
      "candidate":"43_876_param_causal_fsmn_ctc",
      "new_optimizer_updates":UPDATES,
      "gain_levels_train_only":GAIN_LEVELS,
      "train_source_groups":24,
      "train_m ero_original_clips".replace(" ",""):30,
      "train_ordinary_original_clips":24,
      "dev_source_groups":7,
      "frozen_parent_weights":True,"train_only_consistency":True,
      "new_TTS_ASR_inference_calls":0,
      "development":{"positive_original":len(pos),"positive_misses":missed,
        "ordinary_original":len(neg),"ordinary_false_or_extra":false,
        "original_pcm_only_no_gain_or_threshold_selection":True,
        "previously_exposed":True,"rows":observations},
      "loss_first":trace[0]["loss"],"loss_last":trace[-1]["loss"],
      "train_cpu_seconds":round(time.perf_counter()-began,3),
      "new_model_weights":{k.name:{"bytes":k.stat().st_size,
                   "sha256":sha(k.read_bytes())} for k in (weights,npz,onnx)},
      "provisional_research_gate_only":provisional,
      "independent_new_source_locked":"NOT_RUN",
      "real_far_frr":"NOT_MEASURED",
      "ssc305":"NOT_RUN","arm32":"NOT_RUN",
      "product_promotion_allowed":False}
    write_once(out/"TRAINING.json",{"schema":SCHEMA,
       "old_weights_sha256":PARENT_PT_SHA,"new_steps":UPDATES,
       "learning_rate":LR,"seed":SEED,
       "train_only_gain":GAIN_LEVELS,
       "train_only_student_teacher_kl":True,
       "trace":trace,"dev_gradients":False,
       "old_other_keyword_gradients":False,
       "reopened_calibration_or_locked_pcm":False})
    write_once(out/"RESULT.json",result)
    return result


def execute(sources,source_audit_zip,parent_zip,out):
    source_check()
    identity=run_guard()
    need(sources.is_absolute() and sources.is_dir()
         and out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "FROZEN_CREATE_ONCE_GAIN_RESULT")
    out.mkdir()
    write_once(out/"STARTED.json",{"schema":SCHEMA,
               "run":identity,"parent_artifact":PARENT_ARTIFACT,
               "new_optimizer_updates":0,"retry":False})
    handles={}
    try:
        rows,summary=parent.authorized_source(source_audit_zip,sources)
        model,report=parent_packet(parent_zip)
        write_once(out/"ADMISSION.json",{"schema":SCHEMA,
            "source_audit_id":parent.AUDIT_ARTIFACT,
            "source_audit_sha":parent.AUDIT_SHA,
            "source_summary":summary,
            "parent_model_run":PARENT_RUN,
            "parent_model_sha256":PARENT_PT_SHA,
            "train_54_dev_16_original_sources_disjoint":True,
            "never_supervise_dev_or_locked":True})
        for spec in v5.binding.config()["corpus"]:
            key=spec["key"]
            handles[key]=v5.binding.archive(
                sources/(key+"-source.zip"),spec["source"])
        handles["newtrain24"]=v5.binding.archive(
            sources/"newtrain24-source.zip",v5.prior.SOURCE)
        result=frozen_train_and_eval(rows,handles,model,out)
        print(json.dumps({"status":result["status"],
             "steps":result["new_optimizer_updates"],
             "dev_missed":result["development"]["positive_misses"],
             "dev_ordinary_events":
                result["development"]["ordinary_false_or_extra"],
             "production_promotion":False},sort_keys=True))
    except BaseException as e:
        write_once(out/"STOP.json",{"schema":SCHEMA,
           "status":"STOP_NO_INDEPENDENT_GAIN_MODEL_QUALITY_PROOF",
           "error_type":type(e).__name__,"error":str(e)[:500],
           "no_automatic_replay":True,
           "old_locked_never_opened":True})
        raise
    finally:
        for z in handles.values():z.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--sources",type=Path)
    p.add_argument("--source-audit-zip",type=Path)
    p.add_argument("--parent-model-zip",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    need(a.source_check!=a.execute,"EXACT_ONE_GAIN_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    need(a.sources and a.source_audit_zip and a.parent_model_zip and a.output,
         "EXACT_SOURCE_AUDIT_PARENT_MODEL_AND_OUTPUT")
    execute(a.sources,a.source_audit_zip,a.parent_model_zip,a.output)


if __name__=="__main__":
    main()
