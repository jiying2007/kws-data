"""V5 Hey Mero: a DIFFERENT 93,444-parameter 5-block strong causal FSMN.

Source gate 11/2/2/5 passed for NEW synthetic labels but the 43,876-param
warmstarted FSMN remained 1/2 on the two exposed development positives.
The official pretrained phone model was actually 2/2 on those same clips,
0/2 ordinary, so acoustic representation capacity is a justified hypothesis.

Execute one predeclared, source-qualified TRAIN-only from-scratch architecture:
128 hidden, FIVE causal depthwise-memory blocks, each 6 past frames; identical
four blank/HEY/MEH/ROH outputs, same full-utterance 40-Mel causal input.
No teacher frame alignments, old failed checkpoint initialization, reused
calibration, KWS locked data, test threshold or after-the-fact dev tuning.
Only one new TRAIN-only 448-update AdamW and one frozen exposed-DEV replay.
This remains research; no human 3-5m, FAR, FRR or SSC305 claim.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
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
    spec=importlib.util.spec_from_file_location(
        "v5_strong_fsmn_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

proof=load("english_v5_mero_fresh24_plus70_fit_once")
v5=proof.v5
parent=proof.parent
gain=proof.gain
SCHEMA="english-v5-mero-strong-five-block-fsmn128-source35-v1"
REPO="jiying2007/kws-data"
BEFORE="1633bf3b161497f10aa86ff2b3e3b2bbd4f69e55"
PARAMS=93444
HIDDEN=128
BLOCKS=5
MEMORY=6
UPDATES=448
LR=0.00042
WEIGHT_DECAY=0.002
SEED=20261130
GAIN_LEVELS=tuple(gain.GAIN_LEVELS)
OLD_TRAIN=54
OLD_TRAIN_GROUPS=24
NEW_TRAIN_GROUPS=11
NEW_DEV_GROUPS=2

def need(ok,why):
    if not ok:
        raise ValueError(why)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def record(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,sort_keys=True,
                  indent=2,allow_nan=False)
        f.write("\n")

def source_check():
    x=proof.source_check()
    need(x["status"]=="NEW_TRAIN_ONLY_MODEL_NOT_EXECUTED"
         and x["target"]=="Hey Mero"
         and x["model_parameters"]==43876
         and x["new_source_qualified_train_groups"]==11
         and x["new_source_qualified_dev_groups"]==2
         and x["calibration_or_locked_kws_audio_opened"] is False
         and proof.FRESH_QUALIFIED=={"train":11,"development":2}
         and proof.FRESH_SOURCE["id"]==14507208152
         and proof.FRESH_SCREEN["id"]==14509252921
         and proof.FULL_GATE["id"]==14512954554
         and GAIN_LEVELS==gain.GAIN_LEVELS
         and (HIDDEN,BLOCKS,MEMORY)==(128,5,6)
         and PARAMS==93444 and UPDATES==448
         and LR==0.00042 and SEED==20261130,
         "EXACT_NEW_ARCHITECTURE_ONLY_HEY_MERO_ORIGINAL_TRAIN_CONTRACT")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_STRONG_FSMN_NOT_RUN",
            "keyword":"Hey Mero","architecture":"five_causal_memory_blocks",
            "hidden":HIDDEN,"memory_frames_left":MEMORY,
            "parameters":PARAMS,"max_optimizer_updates":UPDATES,
            "old_train_original_pcm":OLD_TRAIN,
            "new_machine_qualified_train_groups":NEW_TRAIN_GROUPS,
            "new_exposed_dev_groups":NEW_DEV_GROUPS,
            "new_tts_asr_kws_model_forwards_on_pr":0,
            "trained_from_scratch":True,
            "prior_failed_256_model_warmstart":False,
            "old_dev_or_locked_train_access":False,
            "calibration_and_locked_kws_access":False,
            "model_promotion_allowed":False}

def guard(env=None,event=None,parentsha=None):
    source_check()
    env=os.environ if env is None else env
    expected={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in expected.items()),
         "ONLY_FIRST_PUBLIC_MAIN_STRONG_FSMN_EXECUTION")
    head=env.get("GITHUB_SHA","")
    need(bool(re.fullmatch("[0-9a-f]{40}",head))
         and bool(re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID",""))),
         "MAIN_EVENT_HEAD_RUN_ID_REQUIRED")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "NO_MOVED_MAIN_OR_REPLAY_OF_STRONG_FSMN")
    if parentsha is None:
        parentsha=subprocess.check_output(["git","rev-parse","HEAD^1"],
                     cwd=ROOT,text=True).strip()
    need(parentsha==BEFORE,"EXACT_PREMODEL_MAIN_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"head":head,
            "optimizer_budget":UPDATES,"repeated_old_model":False}

def audit(old_archives,old_audit,fresh_source,fresh_screen,source_gate):
    source_check()
    gate=proof.frozen_gate(source_gate)
    old,oldsummary=parent.authorized_source(old_audit,old_archives)
    fresh,freshsummary=proof.screen_new(fresh_source,fresh_screen)
    train_old=[x for x in old if x["role"]=="train"]
    train_new=[x for x in fresh if x["role"]=="train"]
    dev=[x for x in fresh if x["role"]=="development"]
    need(gate["status"]=="PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"
         and len(train_old)==OLD_TRAIN
         and len({x["source_group_id"] for x in train_old})==OLD_TRAIN_GROUPS
         and len({x["source_group_id"] for x in train_new})==NEW_TRAIN_GROUPS
         and len({x["source_group_id"] for x in dev})==NEW_DEV_GROUPS
         and Counter(x["mono_class"] for x in dev)=={0:2,1:2}
         and all(x["role"]=="train" for x in train_old+train_new)
         and all(x["role"]=="development" for x in dev)
         and not ({x["source_group_id"] for x in train_old+train_new}&
                  {x["source_group_id"] for x in dev})
         and not ({x["source_pcm_sha256"] for x in train_old}&
                  {x["source_pcm_sha256"] for x in train_new+dev})
         and all(x["mono_class"] in (0,1) and x["tier"] in ("silver","weak")
                 for x in train_old+train_new+dev),
         "EXACT_AUDITED_35_TRAIN_GROUPS_2_EXPOSED_DEV_AND_NO_ROLE_LEAK")
    return train_old,train_new,dev,{
       "gate":gate,"old_source_audit":oldsummary,
       "new_fresh_screen":freshsummary,
       "old_train_pcm":len(train_old),
       "new_train_pcm":len(train_new),
       "new_dev_pcm":len(dev),
       "combined_train_groups":OLD_TRAIN_GROUPS+NEW_TRAIN_GROUPS,
       "new_dev_groups":NEW_DEV_GROUPS,
       "train_dev_original_pcm_sha_disjoint":True,
       "original_locked_audio_opened":False,
       "model_kws_calls":0}

def build_strong_model(torch):
    import torch.nn.functional as F
    class Memory(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear=torch.nn.Linear(HIDDEN,HIDDEN)
            self.memory=torch.nn.Conv1d(HIDDEN,HIDDEN,MEMORY+1,groups=HIDDEN)
        def forward(self,x):
            projected=self.linear(x)
            causal=self.memory(F.pad(projected.transpose(1,2),
                                     (MEMORY,0))).transpose(1,2)
            return F.relu(x+causal)
    class StrongFsmn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input=torch.nn.Linear(40,HIDDEN)
            self.layers=torch.nn.ModuleList(Memory() for _ in range(BLOCKS))
            self.output=torch.nn.Linear(HIDDEN,4)
        def forward(self,features):
            x=F.relu(self.input(features))
            for layer in self.layers:
                x=layer(x)
            return self.output(x)
    model=StrongFsmn()
    need(sum(p.numel() for p in model.parameters())==PARAMS,
         "EXACT_93444_STRONG_CAUSAL_FSMN_PARAMETER_COUNT")
    return model

def run_train(train_old,train_new,dev,handles,new_source,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    train=sorted(train_old+train_new,key=lambda x:(x["source_group_id"],x["id"]))
    need(len({r["source_group_id"] for r in train})==35
         and len({r["source_group_id"] for r in dev})==2,
         "EXACT_TRAIN_ONLY_35_GROUPS_AND_SEALED_DEV_REPLAY")
    bank=v5.make_mel_bank(np)
    cache={}
    for row in train:
        pcm=proof.readpcm(row,handles,new_source)
        need(sha(pcm)==row["source_pcm_sha256"],"TRAIN_ORIGINAL_PCM_HASH")
        cache[row["id"]]={
           "clean":v5.features(pcm,row,torch,np,bank),
           "snr10":v5.features(pcm,row,torch,np,bank,noise_db=10)}
    moments=np.concatenate(
        [cache[r["id"]]["clean"].astype(np.float64) for r in train])
    mean=moments.mean(axis=0).astype(np.float32)
    std=np.maximum(moments.std(axis=0),1e-4).astype(np.float32)
    need(mean.shape==(40,) and std.shape==(40,)
         and np.isfinite(mean).all() and np.isfinite(std).all()
         and (std>0).all(),"NEW_TRAIN_ONLY_FROZEN_NORMALIZATION")
    groups={0:defaultdict(list),1:defaultdict(list)}
    for row in train:
        groups[row["mono_class"]][row["source_group_id"]].append(row)
    need(len(groups[0])==len(groups[1])==35,
         "BOTH_CLASSES_IN_EACH_FROZEN_TRAIN_GROUP")
    model=build_strong_model(torch)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,
                                weight_decay=WEIGHT_DECAY)
    trace=[]
    began=time.perf_counter()
    for step in range(UPDATES):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for klass in (0,1):
            family=sorted(groups[klass])[step%35]
            candidates=sorted(groups[klass][family],key=lambda r:r["id"])
            row=candidates[(step//35)%len(candidates)]
            variant="snr10" if step%3 else "clean"
            grid_index=(step//35 + int(sha(row["source_pcm_sha256"].encode())[:2],16))%len(GAIN_LEVELS)
            adjusted=gain.derived_level(
                cache[row["id"]][variant],GAIN_LEVELS[grid_index],np)
            augmented=parent.augment(adjusted,mean,std,row,step,variant,np)
            x=torch.from_numpy(augmented).unsqueeze(0)
            logits=model(x)
            pos,blank,margin=parent.ctc_losses(logits,torch)
            objective=(pos+F.softplus(0.5-4.0*margin)
                       if klass else
                       0.35*blank+F.softplus(0.5+4.0*margin))
            if row["tier"]=="weak":
                objective=objective*0.75
            need(bool(torch.isfinite(objective)),
                 "FINITE_STRONG_FSMN_CTC_CLASS_LOSS")
            losses.append(objective)
        loss=sum(losses)/2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),5.0,error_if_nonfinite=True)
        optimizer.step()
        if step%16==0 or step==UPDATES-1:
            trace.append({"step":step+1,
             "loss":float(loss.detach()),
             "ordinary":float(losses[0].detach()),
             "mero":float(losses[1].detach())})
    model.eval()
    directory=out/"models";directory.mkdir()
    pt=directory/"mero-strong-fsmn128x5-448.pt"
    torch.save(model.state_dict(),pt)
    restored=build_strong_model(torch)
    restored.load_state_dict(torch.load(pt,map_location="cpu",
                                        weights_only=True),strict=True)
    restored.eval()
    # AFTER optimizer freeze, first access to the already-exposed NEW DEV PCM.
    dev_features={}
    for row in dev:
        pcm=proof.readpcm(row,handles,new_source)
        dev_features[row["id"]]=v5.features(pcm,row,torch,np,bank)
    with torch.no_grad():
        probe=torch.from_numpy((
            (dev_features[dev[0]["id"]]-mean)/std).astype(np.float32)
        ).unsqueeze(0)
        need(bool(torch.equal(model(probe),restored(probe))),
             "EXACT_STRONG_MODEL_PT_RELOAD_PARITY")
    npz=directory/"mero-strong-fsmn128x5-448.npz"
    np.savez(npz,mel_filters=bank,train_mean=mean,train_scale=std,
             **{k:v.detach().cpu().numpy()
                for k,v in restored.state_dict().items()})
    onnx=directory/"mero-strong-fsmn128x5-448.onnx"
    torch.onnx.export(restored,probe,onnx.as_posix(),
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        opset_version=17,dynamo=False,
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(onnx.is_file() and onnx.stat().st_size>10000,
         "REAL_STRONG_FSMN_EXPORT_REQUIRED")
    observed=[]
    with torch.no_grad():
        for row in dev:
            x=torch.from_numpy(((
                dev_features[row["id"]]-mean)/std).astype(np.float32)
            ).unsqueeze(0)
            out_logits=restored(x)
            event=v5.ctc_greedy(out_logits,torch)
            _,_,margin=parent.ctc_losses(out_logits,torch)
            observed.append({"id":row["id"],"source_group_id":row["source_group_id"],
                "pcm_sha256":row["source_pcm_sha256"],
                "positive":bool(row["mono_class"]),
                "full_ctc_event":bool(event["mero_event"]),
                "extra_events":event["extra_events"],
                "full_sequence_minus_blank_per_frame":float(margin.detach())})
    positives=[x for x in observed if x["positive"]]
    ordinary=[x for x in observed if not x["positive"]]
    need(len(positives)==2 and len(ordinary)==2,
         "EXACT_EXPOSED_SYNTHETIC_DEV_GROUPS_TWO_BY_TWO")
    found=sum(x["full_ctc_event"] for x in positives)
    false=sum(x["full_ctc_event"]+x["extra_events"] for x in ordinary)
    result={"schema":SCHEMA,
       "status":("RESEARCH_EXPOSED_DEV_ALL_HIT_NOT_QUALIFIED"
                 if found==2 and false==0 else
                 "RESEARCH_EXPOSED_DEV_NOT_QUALIFIED"),
       "candidate":"strong_128x5_causal_fsmn_ctc",
       "model_parameters":PARAMS,
       "model_from_scratch":True,"optimizer_updates":UPDATES,
       "real_train_source_groups":35,
       "old_train_original_pcm":len(train_old),
       "new_train_original_pcm":len(train_new),
       "train_only_frozen_feature_stats":True,
       "train_gain_levels":GAIN_LEVELS,
       "train_only_10db_noise_and_full_duration_perturb":True,
       "new_exposed_dev":{"positive":len(positives),
             "hits":found,"ordinary":len(ordinary),
             "false_or_extra":false,"source_groups":2,
             "model_chosen_from_dev":False,"rows":observed},
       "previous_256_student_hits_on_same_dev":1,
       "pretrained_phone_hits_on_same_dev":2,
       "architecture_hypothesis_only_no_independent_qualification":True,
       "new_kws_locked_audio_forwards":0,
       "new_tts_asr_phone_forwards":0,
       "old_exposed_dev_reused":False,
       "dev_gradients":False,
       "test_pcm_opened":False,"calibration_pcm_opened":False,
       "train_wall_seconds":round(time.perf_counter()-began,3),
       "loss_trace":trace,
       "model_files":{p.name:{"bytes":p.stat().st_size,
                              "sha256":sha(p.read_bytes())}
                    for p in (pt,npz,onnx)},
       "far_frr_real":"NOT_MEASURED",
       "physical_ssc305":"NOT_RUN",
       "product_promotion_allowed":False}
    record(out/"RESULT.json",result)
    print(json.dumps({k:result[k] for k in (
         "status","model_parameters","optimizer_updates",
         "real_train_source_groups","new_exposed_dev",
         "product_promotion_allowed") if k!="new_exposed_dev"},
         sort_keys=True))
    print(json.dumps({"new_exposed_dev_summary":{
       k:result["new_exposed_dev"][k] for k in (
         "positive","hits","ordinary","false_or_extra","source_groups")},
       "new_weights":[x.name for x in (pt,npz,onnx)]},sort_keys=True))

def execute(old_archives,old_audit,fresh_source,fresh_screen,source_gate,out):
    identity=guard()
    need(out.is_absolute() and out.parent.is_dir()
         and not out.exists(),"NEW_FROZEN_STRONG_MODEL_OUTPUT_REQUIRED")
    out.mkdir()
    record(out/"STARTED.json",{"schema":SCHEMA,
         "run":identity,"source_baseline":"old54+new11_groups",
         "optimizer_max":UPDATES,"new_model":True,
         "never_locked_kws_test":True})
    handles={}
    try:
        old_train,new_train,dev,receipt=audit(
            old_archives,old_audit,fresh_source,fresh_screen,source_gate)
        record(out/"ADMISSION.json",{
            "schema":SCHEMA,"source":receipt,
            "train_class_counts":dict(Counter(x["mono_class"] for x in
                                             old_train+new_train)),
            "new_dev_class_counts":dict(Counter(x["mono_class"] for x in dev)),
            "old_dev_never_read":True,"old_locked_never_read":True,
            "model_promotion_allowed":False})
        for spec in v5.binding.config()["corpus"]:
            key=spec["key"]
            handles[key]=v5.binding.archive(
                old_archives/(key+"-source.zip"),spec["source"])
        handles["newtrain24"]=v5.binding.archive(
            old_archives/"newtrain24-source.zip",v5.prior.SOURCE)
        run_train(old_train,new_train,dev,handles,fresh_source,out)
    except BaseException as e:
        record(out/"STOP.json",{"schema":SCHEMA,
            "status":"STOP_STRONG_128X5_ARCHITECTURE_NOT_QUALIFIED",
            "error_type":type(e).__name__,"reason":str(e)[:480],
            "no_automatic_retry":True,
            "no_calibration_or_locked_inference":True})
        raise
    finally:
        for h in handles.values():h.close()

def main():
    p=argparse.ArgumentParser()
    for flag in ("--source-check","--guard","--audit","--execute"):
        p.add_argument(flag,action="store_true")
    p.add_argument("--old-archives",type=Path)
    p.add_argument("--old-audit",type=Path)
    p.add_argument("--fresh-source",type=Path)
    p.add_argument("--fresh-screen",type=Path)
    p.add_argument("--source-gate",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.audit,a.execute))==1,
         "ONE_EXACT_STRONG_MODEL_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    need(all((a.old_archives,a.old_audit,a.fresh_source,
              a.fresh_screen,a.source_gate)),
         "ORIGINAL_ARCHIVE_AUDIT_AND_FRESH_SCREEN_REQUIRED")
    if a.audit:
        old,new,dev,receipt=audit(a.old_archives,a.old_audit,
                                  a.fresh_source,a.fresh_screen,a.source_gate)
        print(json.dumps({"status":"PASS_ORIGINAL_35_GROUP_TRAIN_ONLY_SOURCE_AUDIT",
           "old_train":len(old),"new_train":len(new),"dev":len(dev),
           "source":receipt["gate"],"new_model_forwards":0},sort_keys=True))
    else:
        need(a.output is not None,"NEW_WEIGHTS_ARCHIVE_REQUIRED")
        execute(a.old_archives,a.old_audit,
                a.fresh_source,a.fresh_screen,a.source_gate,a.output)

if __name__=="__main__":
    main()
