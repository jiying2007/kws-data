"""One real 43K Hey Mero train-only robustness challenger (no audio regeneration).

Start from the exact pretrained 43,876-parameter 256-update research checkpoint,
train ONLY existing independently reviewed TRAIN PCM in 35 source groups.
Use deterministic gain/noise/echo augmentation on TRAIN; independently reviewed
480..487 prospective DEV is read ONLY after final PT/NPZ/ONNX is frozen.
Old exposed DEV, CAL and LOCKED are never read for acoustic fitting.
This is synthetic research, NOT natural FAR/h, human 3-5m FRR or SSC305 release.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(name):
    spec=importlib.util.spec_from_file_location("mero43_robust_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

strong=load("english_v5_mero_strong_fsmn128_once")
prospective=load("english_v5_mero_prospective_three_frozen_once")
v5=strong.v5
sourcefit=strong.proof
gain=strong.gain
parent=strong.parent
SCHEMA="en-v5-frozen43k-trainonly-gain-noise-reflections-288-v1"
REPO="jiying2007/kws-data"
BEFORE="f2f60076b5524c2c68ac0d279916303559a5f7bf"
MODEL=prospective.MODELS["fsmn43k256"]
STEPS=288
SEED=20261119
LR=0.000055
WEIGHT_DECAY=0.001
SNR=(None,10,5)
GAIN=(0.5,0.75,1.0,1.5,2.0)
DEV_CONDITIONS=("original","attenuated_0p5","snr10")
MAX_DEV=10

def need(ok,msg):
    if not ok:raise ValueError(msg)

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def record(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")

def source_check():
    gate=strong.source_check()
    cmp=prospective.source_check()
    need(gate["status"]=="EXACT_STRONG_FSMN_RESEARCH_SOURCE_REGISTERED"
         if "status" in gate and gate["status"]=="EXACT_STRONG_FSMN_RESEARCH_SOURCE_REGISTERED"
         else gate.get("keyword")=="Hey Mero" or gate.get("model_parameters")==93444,
         "REUSE_AUTHENTICATED_STRONG_TRAIN_ONLY_SOURCE_GATES")
    need(cmp["keyword"]=="Hey Mero" and cmp["new_dev_positive"]==2
         and cmp["new_dev_ordinary"]==8
         and MODEL["params"]==43876 and MODEL["steps"]==256
         and MODEL["id"]==14514228406
         and MODEL["sha"]=="91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702"
         and STEPS==288 and LR==0.000055 and SEED==20261119
         and GAIN==tuple(gain.GAIN_LEVELS) and SNR==(None,10,5)
         and MAX_DEV==10,
         "EXACT_RESEARCH_HYPOTHESIS_PREDECLARED")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_OPTIMIZER_OR_AUDIO_FORWARD",
        "keyword":"Hey Mero","source_train_groups":35,
        "new_prospective_dev_machine_positive":2,
        "new_prospective_dev_ordinary":8,
        "frozen_model_zip_sha256":MODEL["sha"],
        "predeclared_updates":STEPS,
        "predeclared_learning_rate":LR,
        "augmentation_snr_db":SNR,"gain_grid":GAIN,
        "old_exposed_dev_audio_opened":False,
        "calibration_locked_audio_opened":False,
        "product_promotion_allowed":False}

def guard(env=None,event=None,parent_sha=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"ONLY_FIRST_PROTECTED_PUBLIC_MAIN_TRAIN")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "REAL_GITHUB_ACTION_RUN_REQUIRED")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "ONE_FROZEN_MAIN_SUCCESSOR_NO_REPLAY")
    if parent_sha is None:
        parent_sha=subprocess.check_output(
            ["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent_sha==BEFORE,"EXACT_UNREPLAYABLE_MAIN_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"main_sha":head,"parent_sha":BEFORE}

def audit(old_archives,old_audit,fresh_source,fresh_screen,source_gate,
          prospective_source,prospective_screen,model_zip):
    source_check()
    old,new,exposed,old_receipt=strong.audit(
        old_archives,old_audit,fresh_source,fresh_screen,source_gate)
    prospective_rows,dev_receipt=prospective.select(prospective_source,prospective_screen)
    training=old+new
    train_groups={r["source_group_id"] for r in training}
    dev_groups={r["source_group_id"] for r in prospective_rows}
    olddev_groups={r["source_group_id"] for r in exposed}
    trainsha={r["source_pcm_sha256"] for r in training}
    devsha={r["source_pcm_sha256"] for r in prospective_rows}
    need(len(train_groups)==35 and len(dev_groups)==8
         and len(olddev_groups)==2
         and not (train_groups&dev_groups) and not (train_groups&olddev_groups)
         and not (trainsha&devsha)
         and Counter(r["mono_class"] for r in prospective_rows)=={0:8,1:2}
         and all(r["role"]=="train" for r in training)
         and all(r["role"]=="development" for r in prospective_rows),
         "NEVER_TRAIN_ON_DEV_CAL_LOCKED_OR_REUSED_SPEAKER")
    zipped=Path(model_zip)
    need(zipped.is_file() and not zipped.is_symlink()
         and zipped.stat().st_size==501276
         and sha(zipped.read_bytes())==MODEL["sha"],
         "EXACT_SAVED_TRAINED_256_STEP_MODEL_ZIP")
    return training,prospective_rows,{"train_groups":35,
         "train_pcm":len(training),"prospective_dev_groups":8,
         "prospective_dev_originals":10,
         "independently_screened_positive":2,
         "independently_screened_ordinary":8,
         "old_train_source":old_receipt,
         "prospective_source":dev_receipt,
         "no_old_dev_cal_locked_pcm_opened":True}

def train_echo(pcm,np):
    """Train-only finite deterministic direct/75ms/180ms reflection profile."""
    source=np.frombuffer(pcm,dtype="<i2").astype(np.float64)/32768.0
    out=source.copy()
    delays=(1200,2880)
    gains=(0.24,0.13)
    for samples,level in zip(delays,gains):
        if len(source)>samples:
            out[samples:]+=level*source[:-samples]
    out=np.clip(out/1.37,-1.0,32767.0/32768.0)
    return np.rint(out*32768).astype("<i2").tobytes()

def train_model(training,dev,old_archives,fresh_source,dev_source,model_dir,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    teacher,bank,mean,scale,hashes=prospective.model(
        torch,np,model_dir,"fsmn43k256")
    need(hashes["pt"]=="1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
         and hashes["npz"]=="02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696",
         "EXACT_OLD_TRAINED_WEIGHTS_NOT_NEW_INIT")
    teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    student=v5.build_model(torch)
    student.load_state_dict(teacher.state_dict(),strict=True)
    trained=sorted(training,key=lambda r:(r["source_group_id"],r["id"]))
    train_cache={}
    handles={}
    try:
        for spec in v5.binding.config()["corpus"]:
            handles[spec["key"]]=v5.binding.archive(
                old_archives/(spec["key"]+"-source.zip"),spec["source"])
        handles["newtrain24"]=v5.binding.archive(
                old_archives/"newtrain24-source.zip",v5.prior.SOURCE)
        for row in trained:
            raw=sourcefit.readpcm(row,handles,fresh_source)
            need(sha(raw)==row["source_pcm_sha256"],
                 "TRAIN_ORIGINAL_PCM_HASH_UNCHANGED")
            train_cache[row["id"]]={
                "clean":v5.features(raw,row,torch,np,bank),
                "snr10":v5.features(raw,row,torch,np,bank,noise_db=10),
                "snr5":v5.features(raw,row,torch,np,bank,noise_db=5),
                "echo":v5.features(train_echo(raw,np),row,torch,np,bank)}
    finally:
        for h in handles.values():h.close()
    group={0:defaultdict(list),1:defaultdict(list)}
    for row in trained:group[row["mono_class"]][row["source_group_id"]].append(row)
    need(len(group[0])==len(group[1])==35,
         "ALL_35_TRAIN_GROUPS_HAVE_BOTH_ACOUSTIC_CLASSES")
    original_teacher={}
    with torch.no_grad():
        for row in trained:
            feature=train_cache[row["id"]]["clean"]
            x=torch.from_numpy(((feature-mean)/scale).astype(np.float32)).unsqueeze(0)
            original_teacher[row["id"]]=teacher(x).softmax(-1).detach()
    optimizer=torch.optim.AdamW(student.parameters(),lr=LR,
                                weight_decay=WEIGHT_DECAY)
    trace=[]
    started=time.perf_counter()
    lanes=("clean","snr10","snr5","echo")
    for step in range(STEPS):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        scores=[]
        for klass in (0,1):
            family=sorted(group[klass])[step%35]
            choices=sorted(group[klass][family],key=lambda r:r["id"])
            row=choices[(step//35)%len(choices)]
            lane=lanes[(step//35+klass+step%4)%len(lanes)]
            level=GAIN[(step//35+int(sha(row["source_pcm_sha256"].encode())[:2],16))%len(GAIN)]
            feature=gain.derived_level(train_cache[row["id"]][lane],level,np)
            x=torch.from_numpy(((feature-mean)/scale).astype(np.float32)).unsqueeze(0)
            logits=student(x)
            pos,blank,margin=parent.ctc_losses(logits,torch)
            main=(pos+F.softplus(0.5-4.0*margin) if klass else
                  0.35*blank+F.softplus(0.5+4.0*margin))
            if row["tier"]=="weak":main=0.75*main
            clean=train_cache[row["id"]]["clean"]
            cleanx=torch.from_numpy(((clean-mean)/scale).astype(np.float32)).unsqueeze(0)
            clean_logits=student(cleanx)
            regularizer=F.kl_div(clean_logits.log_softmax(-1),
                          original_teacher[row["id"]],reduction="none").sum(-1).mean()
            scores.append(main+0.22*regularizer)
        total=sum(scores)/2
        need(bool(torch.isfinite(total)), "FINITE_ACTUAL_ACOUSTIC_TRAIN_LOSS")
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),5.0,error_if_nonfinite=True)
        optimizer.step()
        if step%16==0 or step==STEPS-1:
            trace.append({"step":step+1,"loss":float(total.detach()),
                           "ordinary":float(scores[0].detach()),
                           "positive":float(scores[1].detach())})
    # Freeze weights BEFORE any prospective DEV PCM read.
    student.eval()
    frozen=out/"models";frozen.mkdir()
    pt=frozen/"mero-fsmn-ctc-trainonly-robust288.pt"
    torch.save(student.state_dict(),pt)
    reloaded=v5.build_model(torch)
    reloaded.load_state_dict(torch.load(pt,map_location="cpu",weights_only=True),strict=True)
    reloaded.eval()
    npz=frozen/"mero-fsmn-ctc-trainonly-robust288.npz"
    np.savez(npz,mel_filters=bank,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy() for k,v in reloaded.state_dict().items()})
    onnx=frozen/"mero-fsmn-ctc-trainonly-robust288.onnx"
    trainrow=trained[0]
    xtrain=torch.from_numpy(((train_cache[trainrow["id"]]["clean"]-mean)/scale)
                             .astype(np.float32)).unsqueeze(0)
    torch.onnx.export(reloaded,xtrain,onnx.as_posix(),
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        opset_version=17,dynamo=False,
        dynamic_axes={"fbank40":{1:"frames"},"frame_ctc_logits":{1:"frames"}})
    need(pt.is_file() and npz.is_file() and onnx.is_file()
         and onnx.stat().st_size>10000
         and all(torch.equal(reloaded.state_dict()[key],student.state_dict()[key])
                 for key in student.state_dict()),
         "FROZEN_NEW_43K_MODEL_AND_EXACT_RELOAD")
    del train_cache
    # One diagnostic replay only: the same ten original, previously exposed
    # synthetic prospective DEV WAVs; no threshold/search/optimization changes.
    observed=[]
    with torch.no_grad():
        for row in sorted(dev,key=lambda r:r["id"]):
            raw=prospective.pcm(row,dev_source)
            item={"id":row["id"],"source_group_id":row["source_group_id"],
                  "original_pcm_sha256":row["source_pcm_sha256"],
                  "machine_positive":bool(row["mono_class"]),"lanes":{}}
            for lane in DEV_CONDITIONS:
                if lane=="original":
                    mel=v5.features(raw,row,torch,np,bank)
                elif lane=="attenuated_0p5":
                    mel=gain.derived_level(v5.features(raw,row,torch,np,bank),0.5,np)
                else:
                    mel=v5.features(raw,row,torch,np,bank,noise_db=10)
                x=torch.from_numpy(((mel-mean)/scale).astype(np.float32)).unsqueeze(0)
                base=v5.ctc_greedy(teacher(x),torch)
                new=v5.ctc_greedy(reloaded(x),torch)
                item["lanes"][lane]={"frozen_baseline_event":bool(base["mero_event"]),
                      "challenger_event":bool(new["mero_event"]),
                      "baseline_extra":int(base["extra_events"]),
                      "challenger_extra":int(new["extra_events"])}
            observed.append(item)
    need(len(observed)==MAX_DEV
         and Counter(int(i["machine_positive"]) for i in observed)=={0:8,1:2},
         "EXACT_TEN_POST_FREEZE_DIAGNOSTIC_DEV_ORIGINALS")
    bylane={}
    for lane in DEV_CONDITIONS:
        pos=[x for x in observed if x["machine_positive"]]
        neg=[x for x in observed if not x["machine_positive"]]
        bylane[lane]={"positive":2,"ordinary":8,
            "baseline_hits":sum(x["lanes"][lane]["frozen_baseline_event"] for x in pos),
            "challenger_hits":sum(x["lanes"][lane]["challenger_event"] for x in pos),
            "baseline_false_or_extra":sum(
                x["lanes"][lane]["frozen_baseline_event"]+
                x["lanes"][lane]["baseline_extra"] for x in neg),
            "challenger_false_or_extra":sum(
                x["lanes"][lane]["challenger_event"]+
                x["lanes"][lane]["challenger_extra"] for x in neg)}
    originals=bylane["original"]
    exploratory=(originals["challenger_hits"]>=originals["baseline_hits"]
                 and originals["challenger_false_or_extra"]<=originals["baseline_false_or_extra"]
                 and any(bylane[k]["challenger_hits"]>bylane[k]["baseline_hits"]
                         for k in DEV_CONDITIONS if k!="original"))
    result={"schema":SCHEMA,
        "status":"RESEARCH_EXPOSED_DEV_DIAGNOSTIC_ONLY_NO_PROMOTION",
        "exploratory_augmented_dev_gain":bool(exploratory),
        "pretrained_original_model_artifact":MODEL["id"],
        "pretrained_original_zip_sha256":MODEL["sha"],
        "model_parameters":43876,"actual_optimizer_updates":STEPS,
        "train_source_groups":35,"train_originals":len(trained),
        "training_augmentation_lanes":list(lanes),
        "gain_levels":GAIN,"real_optimizer_duration_s":round(time.perf_counter()-started,3),
        "teacher_clean_train_only":True,"dev_opened_after_export":True,
        "new_frozen_models":{p.name:{"bytes":p.stat().st_size,
                     "sha256":sha(p.read_bytes())} for p in (pt,npz,onnx)},
        "training_loss_trace":trace,"replay_source":"EXPOSED_PREEXISTING_SYNTHETIC_DEV",
        "diagnostic":bylane,"per_original":observed,
        "new_tts_asr_audio_generated":0,"cal_locked_opened":False,
        "physical_ssc305":"NOT_RUN","natural_far_per_hour":"NOT_MEASURED",
        "real_human_3_5m_frr":"NOT_MEASURED",
        "product_promotion_allowed":False}
    record(out/"RESULT.json",result)
    print(json.dumps({"status":result["status"],
        "optimizer_updates":STEPS,"dev_diagnostic":bylane,
        "new_model_pt_sha256":result["new_frozen_models"][pt.name]["sha256"],
        "product_promotion_allowed":False},sort_keys=True))
    return result

def execute(old_archives,old_audit,fresh_source,fresh_screen,source_gate,
            prospective_source,prospective_screen,model_zip,model_dir,out):
    identity=guard()
    need(out.is_absolute() and out.parent.is_dir()
         and not out.exists(),"NEW_SINGLE_RESEARCH_TRAIN_OUTPUT")
    out.mkdir()
    record(out/"STARTED.json",{"schema":SCHEMA,"run":identity,
         "training_only":True,"optimizer_max":STEPS,
         "model_weights_sha":MODEL["sha"],"release_allowed":False})
    try:
        train,dev,receipt=audit(old_archives,old_audit,fresh_source,fresh_screen,
             source_gate,prospective_source,prospective_screen,model_zip)
        record(out/"ADMISSION.json",{"schema":SCHEMA,"receipt":receipt,
             "dev_and_train_roles_immutable":True,"no_cal_or_locked":True})
        train_model(train,dev,old_archives,fresh_source,prospective_source,model_dir,out)
    except BaseException as error:
        record(out/"STOP.json",{"schema":SCHEMA,
            "status":"STOP_REAL_ROBUST_43K_TRAIN_NOT_QUALIFIED",
            "error_type":type(error).__name__,"reason":str(error)[:600],
            "no_automatic_retry":True,"no_cal_or_locked_opened":True})
        raise

def main():
    p=argparse.ArgumentParser()
    for x in ("--source-check","--guard","--audit","--execute"):
        p.add_argument(x,action="store_true")
    for x in ("--old-archives","--old-audit","--fresh-source","--fresh-screen",
              "--source-gate","--prospective-source","--prospective-screen",
              "--original-model-zip","--model-dir","--output"):
        p.add_argument(x,type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.audit,a.execute))==1,
         "ONLY_ONE_BOUNDED_ACOUSTIC_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    inputs=(a.old_archives,a.old_audit,a.fresh_source,a.fresh_screen,
            a.source_gate,a.prospective_source,a.prospective_screen,
            a.original_model_zip)
    need(all(inputs),"ALL_EIGHT_ORIGINAL_SOURCE_AND_MODEL_ARGUMENTS_REQUIRED")
    if a.audit:
        _,_,receipt=audit(*inputs)
        print(json.dumps({"status":"EXACT_TRAIN_ONLY_AND_PROSPECTIVE_DEV_AUDITED",
              "details":receipt,"new_optimizer_updates":0},sort_keys=True));return
    need(a.model_dir and a.output,"MODEL_WEIGHTS_AND_RESULT_DIR_REQUIRED")
    execute(*inputs,a.model_dir,a.output)

if __name__=="__main__":main()
