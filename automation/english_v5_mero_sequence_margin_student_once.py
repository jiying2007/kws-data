"""English V5 Hey Mero: one real teacher-confirmed, sequence-margin causal FSMN fit.

No teacher phoneme timing exists: original Zh-En keyword events have EMPTY tokens
and timestamps. They certify train-only WHOLE-PHRASE positives, not frame labels.
Use auditable existing source-disjoint 24 train / 11 development PCM, train-only
CTC sequence-vs-blank margin, and class-balanced objective. Old 240-step weights
and exposed locked/calibration are neither loaded nor modified. Research only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import zipfile

HERE=Path(__file__).resolve().parent
import importlib.util
_spec=importlib.util.spec_from_file_location(
    "english_v5_student_locked_source", HERE/"english_v5_mero_fsmn_ctc_once.py")
v5=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v5)

REPO="jiying2007/kws-data"
BRANCH="research/english-v5-mero-sequence-margin-student-20261119"
SCHEMA="english-v5-mero-sequence-margin-student-v1"
TEACHER_RUN=44461057432
TEACHER_ARTIFACT=14500132512
TEACHER_ZIP_SHA="86fca579654cbbc17182f710392eb58069952915849c1ecef499da780ec6d7ea"
TEACHER_MODEL_SHA="68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
SEED=20261120
UPDATES=512
LR=0.0006
MARGIN_GAIN=8.0
MARGIN_TARGET=0.2
POSITIVES=14
ORDINARY=10
DEV_POS=6
DEV_ORD=5
PHRASE=[1,2,3]


def need(ok,reason):
    if not ok: raise ValueError(reason)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def save_json(path,obj):
    with path.open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")


def source_check():
    old=v5.source_check()
    c=v5.config()
    need(old["active_target_ids"]==["hey_mero"]
         and old["positive_ctc"]==PHRASE
         and old["old_locked_reopened"] is False
         and c["active_keyword_count"]==1
         and c["model"]["hidden"]==96
         and c["model"]["layers"]==4
         and c["model"]["left_memory"]==4
         and c["model"]["center"] is False
         and len(v5.ARCHIVES)==8 and UPDATES==512 and SEED==20261120
         and TEACHER_RUN==44461057432 and TEACHER_ARTIFACT==14500132512
         and MARGIN_GAIN==8.0 and MARGIN_TARGET==0.2,
         "FROZEN_ONE_MERO_TEACHER_CONFIRMED_CTC_CONTRACT")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NOT_TRAINED",
            "only_keyword":"Hey Mero","positive_ctc":PHRASE,
            "input_source_archives":v5.ARCHIVES,
            "teacher_run":TEACHER_RUN,"teacher_artifact":TEACHER_ARTIFACT,
            "teacher_zip_sha256":TEACHER_ZIP_SHA,"teacher_model_sha256":TEACHER_MODEL_SHA,
            "teacher_time_alignment":"UNAVAILABLE_NO_PHONEME_TIMESTAMPS",
            "new_tts_asr_calls":0,"new_teacher_inference":0,
            "new_optimizer_updates":0,"calibration_locked_opened":False,
            "promotion_allowed":False}


def guard(env=None,event=None):
    env=os.environ if env is None else env
    required={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"pull_request","GITHUB_HEAD_REF":BRANCH,
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),"FIRST_PR_ONLY_NUMERIC")
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
         "EXACT_SAME_REPO_PR_IDENTITY")
    return {"head_sha":pr["head"]["sha"],"run_id":env["GITHUB_RUN_ID"]}


def verified_teacher_train(zip_path,rows):
    """Auth 20 frozen real teacher events, but expose ONLY train roles to fit."""
    need(zip_path.is_file() and not zip_path.is_symlink()
         and sha(zip_path.read_bytes())==TEACHER_ZIP_SHA,
         "IMMUTABLE_REAL_TEACHER_ARTIFACT_HASH")
    with zipfile.ZipFile(zip_path) as z:
        names=set(z.namelist())
        need(names=={"STARTED.json","ADMISSION.json","RESULT.json"},
             "EXACT_THREE_STORED_TEACHER_RECEIPTS")
        teacher=json.loads(z.read("RESULT.json"))
        intake=json.loads(z.read("ADMISSION.json"))
    need(teacher["status"]=="REAL_MERO_PRETRAINED_REFERENCE_REPLAY_COMPLETE"
         and teacher["pretrained_model"]=="zh-en"
         and teacher["model_archive_sha256"]==TEACHER_MODEL_SHA
         and teacher["keyword_ids"]==["hey_mero"]
         and teacher["model_promotion_allowed"] is False
         and teacher["optimizer_updates"]==0
         and teacher["no_calibration_or_locked_test"] is True
         and teacher["summary"]["train"]["mero_positive_detected_clips"]==POSITIVES
         and teacher["summary"]["train"]["ordinary_triggered_clips"]==0
         and teacher["summary"]["development"]["mero_positive_detected_clips"]==DEV_POS,
         "EXACT_REAL_ZH_EN_TEACHER_RECEIPT")
    selected={r["id"]:r for r in rows if r["role"]=="train"}
    reported={r["id"]:r for r in teacher["clips"] if r["role"]=="train"}
    authorized={r["id"]:r for r in intake["selected_audio"] if r["role"]=="train"}
    need(len(selected)==POSITIVES+ORDINARY and set(selected)==set(reported)==set(authorized),
         "NO_ADDED_OR_MISSING_TRAIN_AUDIO")
    expected=Counter()
    confirmed={}
    for ident,row in selected.items():
        source=reported[ident];auth=authorized[ident]
        need(row["source_pcm_sha256"]==source["pcm_sha256"]==auth["pcm_sha256"]
             and row["source_group_id"]==source["source_group_id"]==auth["source_group_id"]
             and row["mono_class"]==source["mono_class"]==auth["mono_class"]
             and len(source["events"]) == (1 if row["mono_class"]==1 else 0),
             "TEACHER_TRAIN_ROW_OR_EVENT_CHANGED")
        for e in source["events"]:
            need(e["mapped_keyword"]=="HEY_MERO" and e["keyword"]=="HEY_MERO"
                 and e["tokens"]==[] and e["token_timestamps_sec"]==[],
                 "TEACHER_EVENT_NOT_PHONE_ALIGNED")
        expected[row["mono_class"]]+=1
        confirmed[ident]=bool(source["events"])
    need(expected==Counter({1:POSITIVES,0:ORDINARY}),
         "TEACHER_EXACT_TRAIN_CLASS_DENOMINATORS")
    # No returned dev teacher predictions, dev timing or scores are used in training.
    return confirmed,{"status":"14_TRAIN_POSITIVES_CONFIRMED_NO_TIMING",
         "train_audio_ids":sorted(selected),"train_sources":
             len({r["source_group_id"] for r in selected.values()}),
         "train_pos":POSITIVES,"train_ordinary":ORDINARY,
         "teacher_artifact_sha256":TEACHER_ZIP_SHA,
         "teacher_zero_gradients":True,
         "dev_teacher_targets_used":False}


def ctc_sequence_margin(logits,torch):
    """Differentiable sequence log-prob contrast, not an exposed-dev threshold."""
    import torch.nn.functional as F
    logp=logits.log_softmax(-1)
    t=logp.shape[1]
    log=logp.transpose(0,1)
    target=torch.tensor(PHRASE,dtype=torch.long)
    # Always compute the same positive phrase marginal also for ordinary rows.
    positive_loss=F.ctc_loss(log,target,torch.tensor([t],dtype=torch.long),
             torch.tensor([3],dtype=torch.long),blank=0,
             reduction="none",zero_infinity=False)[0]
    blank_loss=-logp[0,:,0].sum()
    need(bool(torch.isfinite(positive_loss) and torch.isfinite(blank_loss)),
         "NONFINITE_SEQ_MARGIN")
    # (log P(HEY-MEH-ROH) - log P(blank-only)) per original frame.
    margin=(blank_loss-positive_loss)/t
    return positive_loss/t,blank_loss/t,margin


def source_round_robin(rows):
    groups={0:defaultdict(list),1:defaultdict(list)}
    for row in rows:
        groups[row["mono_class"]][row["source_group_id"]].append(row)
    need(all(len(groups[k])>=6 for k in (0,1)),
         "TRAIN_SIX_SOURCE_GROUPS_EACH_CLASS")
    return groups


def train(rows,teacher_confirmed,archives,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    bank=v5.make_mel_bank(np)
    train_rows=[r for r in rows if r["role"]=="train"]
    dev_rows=[r for r in rows if r["role"]=="development"]
    need(len(train_rows)==24 and len(dev_rows)==11
         and Counter(r["mono_class"] for r in dev_rows)==
             Counter({1:DEV_POS,0:DEV_ORD}),
         "FROZEN_TRAIN_AND_EXPOSED_DEV_COUNTS")
    need(all(teacher_confirmed[r["id"]] == bool(r["mono_class"])
             for r in train_rows),"TEACHER_TRAIN_ONLY_CONFIRMATION")
    cache={}
    for row in rows:
        pcm=v5.base.load_source_pcm(row,archives)
        need(sha(pcm)==row["source_pcm_sha256"],"HASH_BOUND_REAL_PCM")
        cache[row["id"]]={"clean":v5.features(pcm,row,torch,np,bank)}
        if row["role"]=="train":
            cache[row["id"]]["snr10"]=v5.features(
                pcm,row,torch,np,bank,noise_db=10)
    # Training normalization ONLY; never calculate mean/std over dev.
    moments=[cache[r["id"]]["clean"].astype(np.float64)
             for r in train_rows]
    concat=np.concatenate(moments)
    mean=concat.mean(axis=0).astype(np.float32)
    scale=np.maximum(concat.std(axis=0),1e-4).astype(np.float32)
    need(np.isfinite(mean).all() and np.isfinite(scale).all(),
         "FINITE_TRAIN_ONLY_NORMALIZATION")
    groups=source_round_robin(train_rows)
    model=v5.build_model(torch)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    trace=[]
    begin=time.perf_counter()
    for step in range(UPDATES):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        per_class=[]
        for klass in (0,1):
            source_groups=sorted(groups[klass])
            group=source_groups[step%len(source_groups)]
            candidates=sorted(groups[klass][group],key=lambda r:r["id"])
            row=candidates[(step//len(source_groups))%len(candidates)]
            variant="clean" if step%2==0 else "snr10"
            frames=cache[row["id"]][variant]
            x=torch.from_numpy(((frames-mean)/scale).astype(np.float32)).unsqueeze(0)
            logits=model(x)
            sequence,blank,margin=ctc_sequence_margin(logits,torch)
            # Previously blank collapse came from frame-summed ordinary CTC
            # overwhelming the positive event. Class-average, T-normalize,
            # explicitly separate full positive sequence from all-blank.
            if klass==1:
                loss=sequence+2*F.softplus(MARGIN_TARGET-MARGIN_GAIN*margin)
                # Weak clip-level only: no forged teacher phone alignment.
                nonblank=(1-logits.softmax(-1)[0,:,0]).mean()
                loss=loss+0.5*F.relu(0.12-nonblank)
            else:
                loss=0.45*blank+2*F.softplus(MARGIN_TARGET+MARGIN_GAIN*margin)
            need(bool(torch.isfinite(loss)), "FINITE_CLASS_BALANCED_OBJECTIVE")
            per_class.append(loss)
        objective=sum(per_class)/2
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),5.0,error_if_nonfinite=True)
        optimizer.step()
        if step%32==0 or step==UPDATES-1:
            trace.append({"optimizer_step":step+1,"loss":float(objective.detach()),
                         "positive_loss":float(per_class[1].detach()),
                         "ordinary_loss":float(per_class[0].detach())})
    need(len(trace)>=16,"REAL_NUMERIC_UPDATES_REQUIRED")
    modeldir=out/"models";modeldir.mkdir()
    state=modeldir/"mero-fsmn-ctc-sequence-margin.pt"
    torch.save(model.state_dict(),state)
    reloaded=v5.build_model(torch)
    reloaded.load_state_dict(torch.load(state,map_location="cpu",weights_only=True),
                            strict=True)
    reloaded.eval()
    with torch.no_grad():
        probe=torch.from_numpy(
          ((cache[dev_rows[0]["id"]]["clean"]-mean)/scale).astype(np.float32)).unsqueeze(0)
        need(bool(torch.equal(model.eval()(probe),reloaded(probe))),
             "EXACT_SAVED_MODEL_READBACK")
    npz=modeldir/"mero-fsmn-ctc-sequence-margin.npz"
    np.savez(npz,mel_filters=bank,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy() for k,v in
                reloaded.state_dict().items()})
    onnx=modeldir/"mero-fsmn-ctc-sequence-margin.onnx"
    torch.onnx.export(reloaded,probe,onnx.as_posix(),
         input_names=["fbank40"],output_names=["frame_ctc_logits"],
         opset_version=17,dynamo=False,
         dynamic_axes={"fbank40":{1:"frames"},"frame_ctc_logits":{1:"frames"}})
    need(onnx.exists() and onnx.stat().st_size>10000,"ACTUAL_ONNX_EXPORT")
    # No choice of checkpoint or threshold from the exposed development data.
    evaluation=[]
    with torch.no_grad():
        for row in sorted(dev_rows,key=lambda r:r["id"]):
            frames=cache[row["id"]]["clean"]
            inputs=torch.from_numpy(
                ((frames-mean)/scale).astype(np.float32)).unsqueeze(0)
            logits=reloaded(inputs)
            decoded=v5.ctc_greedy(logits,torch)
            _,_,margin=ctc_sequence_margin(logits,torch)
            evaluation.append({"id":row["id"],
                 "source_group_id":row["source_group_id"],
                 "role":"development","positive":bool(row["mono_class"]),
                 "event":bool(decoded["mero_event"]),
                 "extra_events":decoded["extra_events"],
                 "ctc_sequence_margin_per_frame":float(margin.detach())})
    pos=[x for x in evaluation if x["positive"]]
    neg=[x for x in evaluation if not x["positive"]]
    misses=sum(not x["event"] for x in pos)
    false_events=sum(x["event"]+x["extra_events"] for x in neg)
    upgrade=(misses==0 and false_events==0 and
             len({x["source_group_id"] for x in evaluation})>=2)
    actual={"schema":SCHEMA,"status":
            "PROVISIONAL_SYNTHETIC_RESEARCH_QUALITY_UPGRADE" if upgrade else
            "RESEARCH_CANDIDATE_REJECTED_ON_EXPOSED_DEV",
            "keyword_id":"hey_mero","parameter_count":
                sum(p.numel() for p in reloaded.parameters()),
            "real_optimizer_updates":UPDATES,
            "loss_first":trace[0]["loss"],"loss_last":trace[-1]["loss"],
            "cpu_training_elapsed_seconds":round(time.perf_counter()-begin,3),
            "teacher":{"model":"Zh-En 3M","artifact":TEACHER_ARTIFACT,
                "archive_sha256":TEACHER_ZIP_SHA,
                "train_positives_confirmed":POSITIVES,"teacher_target":
                "BINARY_WHOLE_PHRASE_ONLY_NO_PHONE_TIMESTAMPS",
                "teacher_parameters_transferred":0},
            "dataset":{"train_original_wavs":len(train_rows),
                "dev_original_wavs":len(dev_rows),
                "train_source_groups":len({r["source_group_id"] for r in train_rows}),
                "dev_source_groups":len({r["source_group_id"] for r in dev_rows}),
                "train_only_augmentation":["original","snr10"],
                "source_disjoint":True,"dev_not_used_for_gradient_or_checkpoint":True},
            "development":{"source_disjoint":True,"previously_exposed":True,
                "positive_wavs":len(pos),"positive_misses":misses,
                "ordinary_wavs":len(neg),"ordinary_or_extra_events":false_events,
                "rows":evaluation,"threshold_tuned":False,
                "real_far_fr_per_hour":"NOT_MEASURED"},
            "models":{p.name:{"bytes":p.stat().st_size,
                "sha256":sha(p.read_bytes())} for p in (state,npz,onnx)},
            "research_upgrade_gate_met":upgrade,
            "promotion_to_locked_or_arm_or_product":False,
            "independent_new_source_locked":"NOT_AVAILABLE_NOT_ACCESSED",
            "ssc305":"NOT_RUN","human_far_frr":"NOT_MEASURED"}
    save_json(out/"TRAINING.json",{"schema":SCHEMA,"loss_trace":trace,
        "optimizer_updates":UPDATES,"old_scratch_model_reused":False,
        "old_other_keyword_gradients":0,"dev_teacher_events_used":False,
        "new_audio_generation":0,"source_calibration_locked_read":False})
    save_json(out/"RESULT.json",actual)
    return actual


def execute(sources,teacher_zip,out):
    identity=guard()
    source_check()
    need(sources.is_absolute() and sources.is_dir() and out.is_absolute()
         and out.parent.is_dir() and not out.exists(),"IMMUTABLE_CREATE_ONCE_OUTPUT")
    out.mkdir()
    save_json(out/"STARTED.json",{"schema":SCHEMA,"identity":identity,
        "attempt":1,"old_scratch_model_reused":False})
    opened={}
    try:
        rows,audit=v5.admission(sources)
        confirmed,receipt=verified_teacher_train(teacher_zip,rows)
        save_json(out/"ADMISSION.json",{"schema":SCHEMA,
            "summary":audit,"teacher_train":receipt,
            "rows":[{"id":r["id"],"role":r["role"],
                "source_group_id":r["source_group_id"],
                "pcm_sha256":r["source_pcm_sha256"],
                "class":r["mono_class"]} for r in rows],
            "calibration_or_locked":False})
        for entry in v5.binding.config()["corpus"]:
            opened[entry["key"]]=v5.binding.archive(
                sources/(entry["key"]+"-source.zip"),entry["source"])
        opened["newtrain24"]=v5.binding.archive(
            sources/"newtrain24-source.zip",v5.prior.SOURCE)
        result=train(rows,confirmed,opened,out)
        print(json.dumps({"status":result["status"],
            "updates":result["real_optimizer_updates"],
            "parameters":result["parameter_count"],
            "development":{k:v for k,v in result["development"].items()
                 if k!="rows"},"production_promotion":False},sort_keys=True))
    except BaseException as exc:
        save_json(out/"STOP.json",{"schema":SCHEMA,
            "status":"STOP_NO_NEW_QUALITY_PROMOTION",
            "error_type":type(exc).__name__,"error":str(exc)[:500],
            "automatic_retry":False})
        raise
    finally:
        for item in opened.values():item.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--sources",type=Path)
    p.add_argument("--teacher-zip",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    need(args.source_check!=args.execute,"ONE_STAGE_REQUIRED")
    if args.source_check:
        print(json.dumps(source_check(),sort_keys=True))
    else:
        need(args.sources and args.teacher_zip and args.output,
             "EXACT_SOURCE_TEACHER_OUTPUT_REQUIRED")
        execute(args.sources,args.teacher_zip,args.output)


if __name__=="__main__":
    main()
