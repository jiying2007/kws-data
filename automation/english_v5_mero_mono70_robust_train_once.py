"""Hey Mero V5: real robust mono-source FSMN-CTC fit on 70 audited PCM.

The 70 rows come ONLY from unchanged five-native Silver/Weak ordinary/Mero
reviews, proven by exact previous PR audit artifact and all eight original
source/screen ZIP hashes. This is a new, bounded training population, not a
repeat of 240/512-step 35-clip failures and NOT a product/locked-set test.
No legacy other word, UNKNOWN, calibration, locked, fabricated teacher phones,
ASR/TTS or pretrained weight copy. Original source roles never change.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import zipfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "en_v5_mono_audited_population", HERE/"english_v5_mero_mono_source_audit.py")
mono=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mono)
v5=mono.v5
SCHEMA="english-v5-mero-70pcm-robust-fsmn-ctc-v1"
REPO="jiying2007/kws-data"
BRANCH="research/english-v5-mero-mono-source-recovery-20261119"
AUDIT_RUN=44470829024
AUDIT_ARTIFACT=14503604626
AUDIT_SHA="12a1eb39c7a458a466fec2e93ba59e5036319bd9db4b4bdc47e7e6e1c5990f1b"
AUDIT_FILE="english-v5-mero-mono-audit-44470829024.json"
SEED=20261121
UPDATES=360
LR=0.0005
WEIGHT_DECAY=0.002
TRAIN_COUNTS={0:24,1:30}
DEV_COUNTS={0:7,1:9}
TOTAL_SOURCE=70
HIDDEN_PARAMS=43876


def need(ok,reason):
    if not ok: raise ValueError(reason)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def write_once(path,value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,
                  indent=2,allow_nan=False)
        f.write("\n")

def source_check():
    base=mono.source_check()
    need(base["keyword"]=="Hey Mero" and
         base["admitted_original_labels"]==[0,2] and
         base["audited_original_pcm_count"]==630 and
         base["old_calibration_and_exposed_locked_pcm"]=="NOT_OPENED" and
         v5.config()["model"]["hidden"]==96 and
         v5.config()["model"]["ctc_outputs"]==4 and
         AUDIT_RUN==44470829024 and AUDIT_ARTIFACT==14503604626 and
         re.fullmatch("[0-9a-f]{64}",AUDIT_SHA) and
         HIDDEN_PARAMS==43876 and UPDATES==360 and SEED==20261121,
         "PINNED_V5_MONO_SCREENED_70_RESEARCH_INPUT")
    return {"schema":SCHEMA,"status":"PRE_REGISTERED_NOT_TRAINED",
        "target":"Hey Mero","source_audit_run":AUDIT_RUN,
        "source_audit_artifact":AUDIT_ARTIFACT,"source_audit_sha":AUDIT_SHA,
        "old_locked_audio":"NEVER_OPENED",
        "new_TTS_ASR_calls":0,"old_weights_reused":False,
        "optimizer_updates":0,"product_release":False}

def run_guard(env=None,event=None):
    env=os.environ if env is None else env
    must={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
          "GITHUB_EVENT_NAME":"pull_request","GITHUB_RUN_ATTEMPT":"1",
          "GITHUB_HEAD_REF":BRANCH}
    need(all(env.get(k)==v for k,v in must.items()),"PR_ONLY_FIRST_RUN")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    pr=event.get("pull_request",{})
    need(event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         pr.get("head",{}).get("repo",{}).get("full_name")==REPO and
         pr.get("head",{}).get("ref")==BRANCH and
         pr.get("base",{}).get("ref")=="main" and
         event.get("action") in ("opened","synchronize") and
         re.fullmatch("[0-9a-f]{40}",pr.get("head",{}).get("sha",""))
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "EXACT_PUBLIC_V5_RESEARCH_PR_FIRST_ATTEMPT")
    return {"run_id":env["GITHUB_RUN_ID"],"head_sha":pr["head"]["sha"],
            "branch":BRANCH}


def authorized_source(audit_zip,source_root):
    """Reaudit immutable real audio; compare entire exact old source identity."""
    need(audit_zip.is_absolute() and audit_zip.is_file()
         and not audit_zip.is_symlink()
         and sha(audit_zip.read_bytes())==AUDIT_SHA,
         "SAVED_70_SOURCE_COHORT_ZIP_IDENTITY")
    with zipfile.ZipFile(audit_zip) as z:
        need(z.namelist()==[AUDIT_FILE],"SAVED_ONLY_FROZEN_MERO_AUDIT_JSON")
        previous=json.loads(z.read(AUDIT_FILE))
    current=mono.audit(source_root)
    need(previous==current and
         current["schema"]==mono.SCHEMA and
         current["status"]=="NEW_MONO_SOURCE_COVERAGE_FOUND_NOT_PROMOTED" and
         current["old_63_and_new_11_preserved"] is True and
         current["old_calibration_test_pcm_not_opened"] is True and
         current["summary"]["mono_selected_rows"]==TOTAL_SOURCE and
         current["summary"]["newly_recovered_from_original_independent_review"]==35 and
         current["summary"]["mono_source_groups"]==
         {"train":24,"development":7} and
         current["summary"]["additional_source_groups_by_role"]==
         {"train":14,"development":2},
         "COMPLETE_70_PCM_FROZEN_SAME_SCREEN_NO_REPLAY_DRIFT")
    rows=current["source_role_rows"]
    need(len(rows)==TOTAL_SOURCE
         and Counter(r["mono_class"] for r in rows if r["role"]=="train")==TRAIN_COUNTS
         and Counter(r["mono_class"] for r in rows if r["role"]=="development")==DEV_COUNTS
         and all(r["tier"] in ("silver","weak")
                 and r["weak_class"]==(2 if r["mono_class"] else 0)
                 and r["role"] in ("train","development")
                 and r["human_gold"] is False
                 and r["training_allowed_in_original"] is False
                 for r in rows),
         "ONLY_70_AUTHENTIC_MERO_OR_ORDINARY_NO_HOLDOUT")
    return rows, current["summary"]


def groups_by_class(rows):
    groups={0:defaultdict(list),1:defaultdict(list)}
    roles={}
    for r in rows:
        need(r["mono_class"] in (0,1) and r["role"]=="train",
             "NO_DEV_OR_OLD_LOCKED_GRADIENT")
        group=r["source_group_id"]
        need(group not in roles or roles[group]=="train",
             "SOURCE_ROLE_CROSSING")
        roles[group]="train"
        groups[r["mono_class"]][group].append(r)
    need(len(groups[0])==len(groups[1])==24
         and len(rows)==54 and
         Counter(r["mono_class"] for r in rows)==TRAIN_COUNTS,
         "REAL_24_SOURCE_GROUP_BALANCED_SUPERVISION")
    return groups


def augment(feature,mean,scale,row,step,variant,np):
    """Only TRAIN features. Bounded speed, frequency and time masks; no cut."""
    source=(row["source_pcm_sha256"]+"/"+str(step)+"/"+variant).encode()
    rng=np.random.default_rng(int(sha(source)[:16],16))
    x=((feature-mean)/scale).astype(np.float32)
    # Entire time axis interpolated (not cropped); ±5% duration perturbation.
    if step%5:
        ratio=(0.95,1.0,1.05)[int(rng.integers(0,3))]
        t=max(36,int(round(len(x)*ratio)))
        if t!=len(x):
            src=np.arange(len(x),dtype=np.float64)
            dst=np.linspace(0,len(x)-1,t,dtype=np.float64)
            x=np.stack([np.interp(dst,src,x[:,j])
                        for j in range(x.shape[1])],axis=1).astype(np.float32)
    # Masks are train-only feature augmentation; retain all original SHA IDs.
    if step%3:
        f=int(rng.integers(0,38))
        x[:,f:f+2]=0
    if step%4==0 and len(x)>30:
        t=int(rng.integers(4,len(x)-4))
        x[t:t+2,:]=0
    need(x.shape[1]==40 and len(x)>=35 and np.isfinite(x).all(),
         "TRAIN_AUGMENT_FINITE_CAUSAL_40MEL")
    return x


def ctc_losses(logits,torch):
    import torch.nn.functional as F
    logp=logits.log_softmax(-1)
    frames=logp.shape[1]
    target=torch.tensor([1,2,3],dtype=torch.long)
    positive=F.ctc_loss(logp.transpose(0,1),target,
        torch.tensor([frames],dtype=torch.long),
        torch.tensor([3],dtype=torch.long),
        blank=0,reduction="none",zero_infinity=False)[0]/frames
    blank=-logp[0,:,0].mean()
    margin=blank-positive
    need(bool(torch.isfinite(positive) and torch.isfinite(blank) and
              torch.isfinite(margin)),"FINITE_REAL_SEQUENCE_CTC_MARGIN")
    return positive,blank,margin


def execute_training(rows,archives,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    train=sorted([r for r in rows if r["role"]=="train"],key=lambda r:r["id"])
    develop=sorted([r for r in rows if r["role"]=="development"],key=lambda r:r["id"])
    need(len(train)==54 and len(develop)==16 and
         len({r["source_group_id"] for r in train})==24 and
         len({r["source_group_id"] for r in develop})==7 and
         not {r["source_group_id"] for r in train} &
             {r["source_group_id"] for r in develop},
         "IMMUTABLE_SOURCE_DISJOINT_70_COHORT")
    bank=v5.make_mel_bank(np)
    cache={}
    for r in rows:
        pcm=v5.base.load_source_pcm(r,archives)
        need(sha(pcm)==r["source_pcm_sha256"],
             "SAVED_AND_INFERENCE_EXACT_ORIGINAL_PCM_SHA")
        cache[r["id"]]={"clean":v5.features(pcm,r,torch,np,bank)}
        if r["role"]=="train":
            cache[r["id"]]["snr10"]=v5.features(
                 pcm,r,torch,np,bank,noise_db=10)
    moments=np.concatenate([cache[r["id"]]["clean"].astype(np.float64)
                            for r in train])
    mean=moments.mean(axis=0).astype(np.float32)
    scale=np.maximum(moments.std(axis=0),1e-4).astype(np.float32)
    need(np.isfinite(mean).all() and np.isfinite(scale).all() and
         (scale>0).all(),"TRAIN_ONLY_STATS")
    by_class=groups_by_class(train)
    model=v5.build_model(torch)
    need(sum(p.numel() for p in model.parameters())==HIDDEN_PARAMS,
         "EXACT_SMALL_CAUSAL_FSMN_CTC_SIZE")
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,
                                weight_decay=WEIGHT_DECAY)
    trace=[];started=time.perf_counter()
    for step in range(UPDATES):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        class_losses=[]
        for klass in (0,1):
            groups=sorted(by_class[klass])
            group=groups[step%len(groups)]
            members=sorted(by_class[klass][group],key=lambda r:r["id"])
            row=members[(step//len(groups))%len(members)]
            version=("clean" if step%3==0 else "snr10")
            feat=augment(cache[row["id"]][version],mean,scale,row,step,
                         version,np)
            x=torch.from_numpy(feat).unsqueeze(0)
            logits=model(x)
            seq,blank,margin=ctc_losses(logits,torch)
            # Full ordered Hey-Mero phrase, not any partial-token trigger.
            if klass:
                loss=seq+F.softplus(0.5-4.0*margin)
            else:
                loss=0.35*blank+F.softplus(0.5+4.0*margin)
            if row["tier"]=="weak":
                loss=loss*0.75
            need(bool(torch.isfinite(loss)), "FINITE_BALANCED_CLASS_LOSS")
            class_losses.append(loss)
        total=(class_losses[0]+class_losses[1])/2
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,
                                       error_if_nonfinite=True)
        optimizer.step()
        if step%24==0 or step==UPDATES-1:
            trace.append({"step":step+1,"loss":float(total.detach()),
               "positive":float(class_losses[1].detach()),
               "ordinary":float(class_losses[0].detach())})
    need(len(trace)>=15,"TRUE_NUMERIC_GRADIENT_STEPS_NOT_SKIPPED")
    model.eval()
    model_dir=out/"models";model_dir.mkdir()
    pt=model_dir/"mero-mono70-fsmn-ctc.pt"
    torch.save(model.state_dict(),pt)
    check=v5.build_model(torch)
    check.load_state_dict(torch.load(pt,map_location="cpu",weights_only=True),
                          strict=True)
    check.eval()
    with torch.no_grad():
        probe=torch.from_numpy(((cache[develop[0]["id"]]["clean"]-
                  mean)/scale).astype(np.float32)).unsqueeze(0)
        need(bool(torch.equal(model(probe),check(probe))),
             "EXACT_FROZEN_MODEL_RELOAD_OUTPUTS")
    npz=model_dir/"mero-mono70-fsmn-ctc.npz"
    np.savez(npz,mel_filters=bank,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy()
                for k,v in check.state_dict().items()})
    onnx=model_dir/"mero-mono70-fsmn-ctc.onnx"
    torch.onnx.export(check,probe,onnx.as_posix(),
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        opset_version=17,dynamo=False,
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(onnx.exists() and onnx.stat().st_size>10000,
         "ACTUAL_NEW_ONNX_MODEL")
    # One frozen final checkpoint. Never tune on already observed development.
    predictions=[]
    with torch.no_grad():
        for row in develop:
            input_mel=((cache[row["id"]]["clean"]-mean)/scale).astype(np.float32)
            logits=check(torch.from_numpy(input_mel).unsqueeze(0))
            output=v5.ctc_greedy(logits,torch)
            _,_,margin=ctc_losses(logits,torch)
            predictions.append({"id":row["id"],"role":"development",
                "source_group_id":row["source_group_id"],
                "source_pcm_sha256":row["source_pcm_sha256"],
                "positive":row["mono_class"]==1,
                "event":bool(output["mero_event"]),
                "extra_events":output["extra_events"],
                "ctc_full_phrase_blank_margin_per_frame":float(margin.detach())})
    positives=[p for p in predictions if p["positive"]]
    ordinary=[p for p in predictions if not p["positive"]]
    misses=sum(not p["event"] for p in positives)
    false=sum(p["event"]+p["extra_events"] for p in ordinary)
    need(len(positives)==9 and len(ordinary)==7,"SOURCE_GATED_DEV_9_7")
    # All exposed dev: cannot independently certify FAR/FRR or deploy.
    provisional=(misses==0 and false==0)
    result={"schema":SCHEMA,
      "status":("PROVISIONAL_MONO70_SYNTHETIC_RESEARCH_SIGNAL" if provisional
                else "REJECTED_NOT_NEW_MODEL_QUALITY_QUALIFIED"),
      "model":"causal_fsmn_ctc","parameters":HIDDEN_PARAMS,
      "real_optimizer_updates":UPDATES,
      "train_original_pcm":54,"train_source_groups":24,
      "dev_original_pcm":16,"dev_source_groups":7,
      "train_by_class":{"mero":30,"ordinary":24},
      "dev_by_class":{"mero":9,"ordinary":7},
      "original_five_native_admission_only":True,
      "newly_recovered_machine_review_rows":35,
      "no_old_other_keyword_gradients":True,
      "no_calibration_locked_opened":True,
      "train_only_augmentations":["original","snr10","speed_5pct","freq2","time2"],
      "train_only_normalization":True,
      "checkpoint":"fixed_360_not_dev_selected",
      "loss_begin":trace[0]["loss"],"loss_final":trace[-1]["loss"],
      "training_cpu_seconds":round(time.perf_counter()-started,2),
      "dev":{"positive_original":len(positives),
        "positive_misses":misses,"ordinary_original":len(ordinary),
        "ordinary_false_or_extra":false,"rows":predictions,
        "exposed_source_disjoint_development_only":True,
        "dev_gradient_and_threshold_tuning":False},
      "research_provisional_gate":provisional,
      "new_independent_locked":"NOT_AVAILABLE",
      "human_far_frr":"NOT_MEASURED",
      "arm32":"NOT_RUN","ssc305":"NOT_RUN",
      "product_promotion_allowed":False,
      "artifacts":{p.name:{"bytes":p.stat().st_size,
         "sha256":sha(p.read_bytes())} for p in (pt,npz,onnx)}}
    write_once(out/"TRAINING.json",
       {"schema":SCHEMA,"source_role_counts":result["train_by_class"],
        "optimizer_updates":UPDATES,"loss_trace":trace,
        "seed":SEED,"weight_decay":WEIGHT_DECAY,
        "learning_rate":LR,"checkpoint_selection":"fixed_final",
        "old_scratch_model_reused":False,"new_tts_asr_calls":0,
        "teacher_frame_alignment_fabricated":False,
        "calibration_locked_audio_opened":False})
    write_once(out/"RESULT.json",result)
    return result


def execute(sources,audit_zip,out):
    source_check()
    identity=run_guard()
    need(sources.is_absolute() and sources.is_dir() and
         out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "CREATE_ONLY_AUTHENTICATED_NUMERIC_RECEIPT")
    out.mkdir()
    write_once(out/"STARTED.json",{"schema":SCHEMA,
        "run":identity,"model_training_status":"started_not_yet_qualified",
        "old_weights":False,"retry":False})
    opened={}
    try:
        rows,summary=authorized_source(audit_zip,sources)
        write_once(out/"ADMISSION.json",
            {"schema":SCHEMA,"source_audit":summary,
             "original_source_artifact_ids":v5.ARCHIVES,
             "mono_rows":[{"id":r["id"],"role":r["role"],
                 "source_group_id":r["source_group_id"],
                 "pcm_sha256":r["source_pcm_sha256"],
                 "tier":r["tier"],"class":r["mono_class"]} for r in rows],
             "calibration_locked_opened":False})
        for spec in v5.binding.config()["corpus"]:
            key=spec["key"]
            opened[key]=v5.binding.archive(
                sources/(key+"-source.zip"),spec["source"])
        opened["newtrain24"]=v5.binding.archive(
            sources/"newtrain24-source.zip",v5.prior.SOURCE)
        result=execute_training(rows,opened,out)
        print(json.dumps({"status":result["status"],
            "parameters":HIDDEN_PARAMS,
            "updates":result["real_optimizer_updates"],
            "dev":{k:v for k,v in result["dev"].items()
                   if k!="rows"},"product_promotion":False},
            sort_keys=True))
    except BaseException as exc:
        write_once(out/"STOP.json",{"schema":SCHEMA,
            "status":"STOP_NO_MODEL_QUALITY_PROMOTION",
            "error_type":type(exc).__name__,"detail":str(exc)[:500],
            "automatic_retry":False,
            "reopened_locked_audio":False})
        raise
    finally:
        for item in opened.values():item.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--sources",type=Path)
    p.add_argument("--audit-zip",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    need(a.source_check!=a.execute,"SELECT_EXACTLY_ONE_MODE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True))
        return
    need(a.sources and a.audit_zip and a.output,
         "PINNED_70_SOURCE_AND_AUDIT_ZIP_REQUIRED")
    execute(a.sources,a.audit_zip,a.output)


if __name__=="__main__":
    main()
