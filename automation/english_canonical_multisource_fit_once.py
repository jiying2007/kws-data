"""Run exactly one new seeded 49,283-parameter causal English research head.

Consumes ONLY the separately bound 63 machine-reviewed synthetic audio clips;
all original 462 source waveforms have exact byte/role/lineage verification.
Weights are new (no Meero warmstart), fixed after 96 train-only updates.
Calibrates a fixed threshold grid on calibration AFTER weights are frozen;
opens locked-test PCM ONLY AFTER weights AND threshold are saved.
This is a reproducible research model, not a commercial/SSC305 qualification.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import traceback
import zipfile
import wave

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location(
    "canon_source_binding",HERE/"english_canonical_multisource_bind_once.py")
binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
SCHEMA="english-canonical-multisource-fresh-causal-head-v1"
MAGIC=b"EHKWS01\x00"
ORDER=("mean96","scale96","w1","b1","w2","b2")
SHAPES={"mean96":(96,),"scale96":(96,),"w1":(32,1536),
        "b1":(32,),"w2":(3,32),"b2":(3,)}
MEL_HISTORY_END=31840
MEL_LAST_LAG=128
def need(x,reason):
    if not x:raise ValueError(reason)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def saved(path,value):binding.once(path,value)

def verify_assets(folder):
    cfg=binding.config()
    from pathlib import Path
    base=Path(folder)
    need(base.is_dir() and not base.is_symlink(),"ONLY_ORIGINAL_PINNED_FEATURE_ASSETS")
    for name,expected in cfg["frozen_backbone"]["assets"].items():
        path=base/name
        need(path.is_file() and not path.is_symlink() and
             path.stat().st_size==expected["bytes"] and
             binding.sha(path.read_bytes())==expected["sha256"],
             "EXACT_OPENWAKEWORD_BACKBONE_ASSET:"+name)
    return cfg["frozen_backbone"]["assets"]

def make_frontend(folder):
    import numpy as np
    import onnx
    import onnxruntime as ort
    verify_assets(folder)
    path=Path(folder)/"melspectrogram.onnx"
    graph=onnx.load(str(path),load_external_data=False)
    conv=[n for n in graph.graph.node if n.op_type=="Conv"]
    need(len(conv)==2 and not any(n.op_type in ("Pad","Slice","Resize",
                "AveragePool","MaxPool") for n in graph.graph.node),
         "FROZEN_FIXED_STFT_CONV_GEOMETRY")
    init={x.name:x for x in graph.graph.initializer}
    for node in conv:
        attrib={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        weight=init.get(node.input[1])
        need(weight is not None and list(weight.dims)==[257,1,512] and
             list(attrib.get("strides",[1]))==[160] and
             list(attrib.get("pads",[0,0]))==[0,0] and
             list(attrib.get("kernel_shape",[512]))==[512],
             "ORIGINAL_MEL_97_OUTPUT_AND_ACTUAL_HISTORY_COVERAGE")
    opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
    mel=ort.InferenceSession(str(path),sess_options=opt,
                             providers=["CPUExecutionProvider"])
    embed=ort.InferenceSession(str(Path(folder)/"embedding_model.onnx"),
                               sess_options=opt,providers=["CPUExecutionProvider"])
    need([(x.name,x.type) for x in mel.get_inputs()]==
         [("input","tensor(float)")] and
         [(x.name,x.type) for x in embed.get_inputs()]==
         [("input_1","tensor(float)")],
         "PINNED_CAUSAL_FROZEN_ONNX_IO_GEOMETRY")
    def forward(pcm):
        raw=pcm.astype(np.float32,copy=False).reshape(1,16000)
        features=mel.run(None,{"input":raw})[0]
        need(features.shape==(1,1,97,32) and features.dtype==np.float32,
             "MEL_97X32_GEOMETRY")
        proc=((features[0,0,-76:,:]/np.float32(10))+
              np.float32(2)).reshape(1,76,32,1).astype(np.float32)
        vec=embed.run(None,{"input_1":proc})[0]
        need(vec.shape==(1,1,1,96) and vec.dtype==np.float32 and
             np.isfinite(vec).all(),"ORIGINAL_FROZEN_EMBEDDING96")
        return vec.reshape(96)
    return forward

def noise_context(original,row,condition):
    import numpy as np
    cfg=binding.config()["preprocessing"]
    need(condition in cfg["noise_conditions"] and
         type(original) is bytes and len(original)==row["samples"]*2,
         "FULL_CANONICAL_AUDIO_ORIGINAL_PCM")
    samples=np.frombuffer(original,dtype="<i2").astype(np.float32)
    pre=cfg["prefix_noise_samples"];post=cfg["suffix_noise_samples"]
    signal=np.concatenate((np.zeros(pre,dtype=np.float32),samples,
                           np.zeros(post,dtype=np.float32)))
    salt=sha((row["id"]+"/"+condition+"/canon-v1").encode())
    seed=int(salt[:16],16)
    rng=np.random.default_rng(seed)
    rand=rng.standard_normal(len(signal)).astype(np.float32)
    rand-=rand.mean()
    rms=float(np.sqrt(np.mean(samples.astype(np.float64)**2)))
    noise_rms=(32.0 if condition=="clean" else
               rms/10**((10 if condition=="snr10" else 5)/20))
    value=signal+rand*(noise_rms/float(np.sqrt(np.mean(rand**2))))
    if condition=="clean":
        value[pre:pre+len(samples)]=samples
    peak=float(np.max(np.abs(value)))
    gain=min(1.0,30000.0/peak) if peak else 1.0
    out=np.rint(value*gain).clip(-32768,32767).astype("<i2")
    need(len(out)==pre+row["samples"]+post and
         np.isfinite(out.astype(np.float32)).all(),
         "FINITE_WHOLE_SYNTHETIC_CONTEXT")
    return out,{"original_pcm_sha256":row["source_pcm_sha256"],
        "condition":condition,"clip_start":pre,
        "clip_end":pre+row["samples"],"full_samples":len(out),
        "gain":gain,"synthetic_noise_not_true_recording":True}

def embeddings(pcm,frontend,progress):
    import numpy as np
    need(pcm.ndim==1 and pcm.dtype==np.int16 and len(pcm)>=35200,
         "COMPLETE_CAUSAL_HISTORY_IN_SYNTHETIC_CONTEXT")
    rows=[];clock=[]
    for end in range(16000,len(pcm)+1,1280):
        progress["actual_frontend_forward_calls"]+=1
        need(progress["actual_frontend_forward_calls"]<=
             binding.config()["preprocessing"]["max_model_forward_calls"],
             "BOUND_ALL_FROZEN_ONNX_FEATURE_FORWARD_CALLS")
        rows.append(frontend(pcm[end-16000:end]))
        clock.append(end)
    values=np.asarray(rows,dtype=np.float32)
    need(values.shape==(len(clock),96) and
         np.isfinite(values).all(),"FROZEN_CAUSAL_EMBEDDING_STREAM")
    heads=[];ends=[]
    for n in range(15,len(clock)):
        heads.append(values[n-15:n+1].copy())
        ends.append(clock[n])
    need(heads and all(ends[i]-ends[i-1]==1280 for i in range(1,len(ends))),
         "CONTIGUOUS_CAUSAL_HEAD_HISTORIES")
    return np.stack(heads).astype(np.float32),ends

def build_bag(row,original,condition,frontend,progress):
    import numpy as np
    pcm,meta=noise_context(original,row,condition)
    heads,ends=embeddings(pcm,frontend,progress)
    label=row["weak_class"]
    if label in (1,2):
        include=[i for i,end in enumerate(ends)
                 if end-MEL_HISTORY_END<=meta["clip_start"] and
                 end-MEL_LAST_LAG>=meta["clip_end"]]
    else:
        include=[i for i,end in enumerate(ends)
                 if end>=meta["clip_end"]]
    if not include:
        return None,{"id":row["id"],"condition":condition,
                     "reason":"NO_SUPPORTED_WHOLE_ORIGINAL_CLIP_CAUSAL_HISTORY",
                     "original_samples":row["samples"]}
    subset=heads[include]
    need(len(subset)<=64 and np.isfinite(subset).all(),
         "NO_CROPPED_OR_GENERATED_POSITIVE_SOURCE_LABEL")
    return {"id":row["id"]+"#"+condition,
         "row_id":row["id"],"role":row["role"],
         "source_group_id":row["source_group_id"],
         "weak_class":label,"tier":row["tier"],
         "tier_weight":binding.config()["accepted_machine_tiers"][row["tier"]],
         "condition":condition,"windows":subset,"all_windows":heads,
         "head_ends":ends,"clip_start":meta["clip_start"],
         "clip_end":meta["clip_end"],
         "original_pcm_sha256":meta["original_pcm_sha256"],
         "synthetic_context_only":True},None

def score(model,normalizer,bag,threshold):
    import numpy as np
    mean,scale=normalizer
    W=model
    allx=((bag["all_windows"]-mean)/scale).reshape(-1,1536)
    v=np.maximum(allx@W["w1"].T+W["b1"],0)@W["w2"].T+W["b2"]
    need(np.isfinite(v).all(),"FINITE_SAVED_WEIGHT_HEAD_EVALUATION")
    e=np.exp((v-v.max(axis=-1,keepdims=True)).astype(np.float64))
    p=e/e.sum(axis=-1,keepdims=True)
    events=[]
    streak=0;last=0;blocked=0
    confirm=binding.config()["head"]["confirm_embeddings"]
    cooldown=binding.config()["head"]["cooldown_embeddings"]*1280
    for i,(prob,at) in enumerate(zip(p,bag["head_ends"])):
        winner=int(prob.argmax())
        if winner==0 or float(prob[winner])<threshold or at<blocked:
            streak=last=0
        else:
            streak=streak+1 if winner==last else 1
            last=winner
            if streak>=confirm:
                events.append({"class":winner,"end_sample":at,
                    "score":float(prob[winner])})
                blocked=at+cooldown
                streak=last=0
    expected=bag["weak_class"]
    good=sum(ev["class"]==expected and
          bag["clip_end"]<=ev["end_sample"]<=bag["clip_end"]+8000
          for ev in events) if expected else 0
    false=sum(ev["class"]!=expected or expected==0 or
          ev["end_sample"]<bag["clip_end"] or
          ev["end_sample"]>bag["clip_end"]+8000 for ev in events)
    return {"id":bag["id"],"role":bag["role"],
        "condition":bag["condition"],"source_group_id":bag["source_group_id"],
        "weak_class":expected,"matched_intent_after_clip_end":good>0,
        "false_or_extra_events":false,"events":events,
        "clip_end_surrogate_not_human_word_end":True}

def aggregate(predictions):
    labels={}
    for clazz in (0,1,2):
        data=[r for r in predictions if r["weak_class"]==clazz]
        need(data,"NO_EVALUATION_CLASS")
        hit=sum(x["matched_intent_after_clip_end"] for x in data)
        labels[str(clazz)]={"cases":len(data),
             "matched":hit if clazz else None,
             "missed":len(data)-hit if clazz else None,
             "miss_rate":1-hit/len(data) if clazz else None,
             "false_or_extra_events":sum(x["false_or_extra_events"] for x in data)}
    return {"cases":len(predictions),
        "source_groups":len({r["source_group_id"] for r in predictions}),
        "classes":labels,"total_false_or_extra_events":
                sum(x["false_or_extra_events"] for x in predictions),
        "scope":"SYNTHETIC_MACHINE_SILVER_WEAK_ONLY_NO_REAL_WORLD_FAR_FRR"}

def load_source_pcm(row,sources):
    key=row["corpus"]
    archive=sources[key]
    name="wav/"+row["id"]+".wav"
    need(name in archive.namelist(),"REVIEWED_SOURCE_WAV_MISSING")
    raw=archive.read(name)
    return binding.read_wav(raw,{"wav_sha256":row["source_wav_sha256"],
             "pcm_sha256":row["source_pcm_sha256"],"samples":row["samples"]})

def training_loop(selected,sources,frontend,out):
    import numpy as np
    import torch
    cfg=binding.config();spec=cfg["head"]
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(spec["seed"]);torch.use_deterministic_algorithms(True)
    progress={"actual_frontend_forward_calls":0,"optimizer_updates_started":0,
              "optimizer_updates_completed":0,
              "calibration_feature_calls_started":False,
              "locked_test_feature_calls_started":False}
    train_bags=[];dev_bags=[];excluded=[]
    for row in sorted(selected,key=lambda x:x["id"]):
        if row["role"] not in ("train","development"):
            continue
        pcm=load_source_pcm(row,sources)
        if row["role"]=="train":
            conditions=("clean","snr10","snr5")
        else:
            conditions=("clean","snr5")
        for condition in conditions:
            bag,error=build_bag(row,pcm,condition,frontend,progress)
            if error:excluded.append(error)
            else:(train_bags if row["role"]=="train" else dev_bags).append(bag)
    valid=defaultdict(set)
    for b in train_bags+dev_bags:
        valid[b["role"],b["source_group_id"]].add(b["weak_class"])
    eligible={key for key,labels in valid.items() if labels=={0,1,2}}
    counts=Counter(k[0] for k in eligible)
    need(counts["train"]>=cfg["full_qualified_role_minimum"]["train"] and
         counts["development"]>=cfg["full_qualified_role_minimum"]["development"],
         "POST_MEL_CAUSAL_WINDOW_SUPPORT_MINIMUM_SOURCE_GROUPS")
    train_bags=[b for b in train_bags if
                (b["role"],b["source_group_id"]) in eligible]
    dev_bags=[b for b in dev_bags if
              (b["role"],b["source_group_id"]) in eligible]
    need(len(train_bags)>=45 and len(dev_bags)>=12,
         "TRAIN_DEV_CAUSAL_FEATURE_BAGS_TOO_FEW")
    # Absolutely no calibration or locked audio has been opened so far.
    count=sum(len(b["windows"])*16 for b in train_bags)
    values=sum((b["windows"].astype(np.float64).sum((0,1)) for b in train_bags))
    squares=sum(((b["windows"].astype(np.float64)**2).sum((0,1)) for b in train_bags))
    mean=(values/count).astype(np.float32)
    scale=np.maximum(np.sqrt(np.maximum(squares/count-(values/count)**2,0)),1e-5).astype(np.float32)
    need(np.isfinite(mean).all() and np.isfinite(scale).all() and
         (scale>0).all(),"TRAIN_ONLY_FROZEN_NORMALIZATION")
    model=torch.nn.Sequential(torch.nn.Linear(1536,32),torch.nn.ReLU(),
                              torch.nn.Linear(32,3))
    need(sum(p.numel() for p in model.parameters())==49283,
         "ORIGINAL_SMALL_CAUSAL_HEAD_49283_PARAMETERS")
    optimizer=torch.optim.AdamW(model.parameters(),
           lr=spec["learning_rate"],weight_decay=spec["weight_decay"])
    grouped=defaultdict(lambda:defaultdict(list))
    for bag in train_bags:
        grouped[bag["weak_class"]][bag["source_group_id"]].append(bag)
    need(set(grouped)=={0,1,2} and
         all(len(grouped[c])>=6 for c in (0,1,2)),
         "REAL_SIX_SOURCE_TRAINED_CLASSES")
    transformed={b["id"]:torch.from_numpy(
        ((b["windows"]-mean)/scale).reshape(len(b["windows"]),1536).astype(np.float32))
        for b in train_bags}
    losses=[];draws=[]
    for update in range(spec["max_updates"]):
        optimizer.zero_grad(set_to_none=True)
        terms=[];weights=[];records=[]
        progress["optimizer_updates_started"]+=1
        for cl in (0,1,2):
            group_ids=sorted(grouped[cl])
            group=group_ids[update%len(group_ids)]
            choices=sorted(grouped[cl][group],key=lambda r:r["id"])
            b=choices[(update//len(group_ids))%len(choices)]
            pred=torch.log_softmax(model(transformed[b["id"]]),dim=-1)
            loss=(-pred[:,0].mean() if cl==0
                  else torch.log(torch.tensor(len(pred),dtype=torch.float32))-
                       torch.logsumexp(pred[:,cl],dim=0))
            terms.append(loss*b["tier_weight"])
            weights.append(b["tier_weight"])
            records.append({"id":b["id"],"class":cl,
                "source_group_id":group,"weight":b["tier_weight"]})
        obj=torch.stack(terms).sum()/sum(weights)
        need(bool(torch.isfinite(obj)),"FINITE_STRICT_TRAIN_LOSS")
        obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),spec["clip_grad_norm"],
                                       error_if_nonfinite=True)
        optimizer.step()
        progress["optimizer_updates_completed"]+=1
        losses.append(float(obj.detach()))
        draws.append(records)
    arrays={"mean96":mean,"scale96":scale,
            "w1":model[0].weight.detach().cpu().numpy().astype(np.float32),
            "b1":model[0].bias.detach().cpu().numpy().astype(np.float32),
            "w2":model[2].weight.detach().cpu().numpy().astype(np.float32),
            "b2":model[2].bias.detach().cpu().numpy().astype(np.float32)}
    need(set(arrays)==set(SHAPES) and all(
         arrays[key].shape==shape and arrays[key].dtype==np.float32
         and np.isfinite(arrays[key]).all() for key,shape in SHAPES.items()),
         "FULL_ORIGINAL_SMALL_HEAD_FINITE_WEIGHT_ARRAYS")
    modeldir=Path(out)/"models";modeldir.mkdir()
    np.savez(modeldir/"fresh-canonical-head.npz",**arrays)
    with np.load(modeldir/"fresh-canonical-head.npz",allow_pickle=False) as z:
        restored={key:z[key].copy() for key in z.files}
    need(set(restored)==set(SHAPES) and
         all(np.array_equal(restored[k],arrays[k]) for k in SHAPES),
         "SAVED_NEW_HEAD_NUMERICAL_READBACK")
    data=MAGIC+b"".join(restored[k].astype("<f4").tobytes() for k in ORDER)
    need(len(data)==197908,"C_AND_ARM_COMPATIBLE_49283_HEAD_BLOB")
    (modeldir/"fresh-canonical-head.bin").write_bytes(data)
    npz_pin=sha((modeldir/"fresh-canonical-head.npz").read_bytes())
    binary_pin=sha(data)
    saved(Path(out)/"WEIGHTS_FROZEN.json",{
       "schema":SCHEMA,"status":"NEW_CANONICAL_WEIGHTS_FROZEN",
       "npz_sha256":npz_pin,"bin_sha256":binary_pin,
       "optimizer_updates":len(losses),"source_qualified_after_mel":{
            role:counts[role] for role in ("train","development")},
       "train_bags":len(train_bags),"development_bags":len(dev_bags),
       "old_meero_model_warmstart":False,"old_meero_positive_bags":0,
       "calibration_audio_read_before_freeze":False,
       "locked_audio_read_before_freeze":False})
    saved(Path(out)/"TRAINING.json",{
       "schema":SCHEMA,"optimizer":"AdamW","seed":spec["seed"],
       "real_optimizer_updates":len(losses),"training_losses":losses,
       "training_draws":draws,"rejected_mel_geometry":excluded,
       "training_audio_bags":len(train_bags),"development_audio_bags":len(dev_bags),
       "actual_frontend_calls_through_freeze":progress["actual_frontend_forward_calls"],
       "new_head_binary_sha256":binary_pin,
       "all_optimizer_sources_train_role_only":True,
       "normalization_is_train_only":True,"human_gold":False})
    # Development was read before weights freeze but cannot select a different
    # checkpoint; single fixed 96th update is the only candidate.
    dev_scores=[score(restored,(mean,scale),b,0.8) for b in dev_bags]
    saved(Path(out)/"DEVELOPMENT.json",{
       "schema":SCHEMA,"status":"FIXED_FROZEN_HEAD_DEV_DIAGNOSTIC",
       "weights_sha256":binary_pin,
       "working_threshold_predeclared":0.8,
       "summary":aggregate(dev_scores),
       "checkpoint_selection_from_dev":False})
    # Calibrate ONLY from newly opened calibration PCM now weights are saved.
    progress["calibration_feature_calls_started"]=True
    cal_bags=[]
    for row in sorted(selected,key=lambda x:x["id"]):
        if row["role"]!="calibration":continue
        pcm=load_source_pcm(row,sources)
        for mode in ("clean","snr5"):
            bag,error=build_bag(row,pcm,mode,frontend,progress)
            need(error is None,"CALIBRATION_GEOMETRY_MUST_BE_COMPLETE")
            cal_bags.append(bag)
    choices=[]
    for threshold in spec["threshold_choices"]:
        report=aggregate([score(restored,(mean,scale),b,threshold)
                          for b in cal_bags])
        hits=sum(report["classes"][str(cl)]["matched"] for cl in (1,2))
        choices.append({"threshold":threshold,"report":report,
             "selection_key":[report["total_false_or_extra_events"],-hits,
                               spec["threshold_choices"].index(threshold)]})
    best=min(choices,key=lambda z:z["selection_key"])
    cal_sha=sha(binding.canon({
        "model":binary_pin,"threshold":best["threshold"],
        "predeclared_thresholds":spec["threshold_choices"]}))
    saved(Path(out)/"CALIBRATION_FROZEN.json",{
       "schema":SCHEMA,"status":"THRESHOLD_FROZEN_BEFORE_LOCKED_PCM",
       "model_binary_sha256":binary_pin,
       "threshold":best["threshold"],
       "selection_scope":"CALIBRATION_ONLY_11_ORIGINAL_MACHINE_REVIEWED_CLIPS",
       "calibration_cases":len(cal_bags),"all_fixed_candidates":choices,
       "calibration_binding_sha256":cal_sha,
       "locked_pcm_or_scores_used_in_choice":False})
    # Opening locked PCM is permitted only after both weight and threshold
    # freeze receipts are already present on disk.
    need((Path(out)/"WEIGHTS_FROZEN.json").is_file() and
         (Path(out)/"CALIBRATION_FROZEN.json").is_file(),
         "CLOSED_SEQUENTIAL_LOCKED_TEST_GATE")
    progress["locked_test_feature_calls_started"]=True
    locked_bags=[]
    for row in sorted(selected,key=lambda x:x["id"]):
        if row["role"]!="locked-test":continue
        pcm=load_source_pcm(row,sources)
        for mode in ("clean","snr5"):
            bag,error=build_bag(row,pcm,mode,frontend,progress)
            need(error is None,"LOCKED_CASE_GEOMETRY_MUST_BE_COMPLETE")
            locked_bags.append(bag)
    held=aggregate([score(restored,(mean,scale),b,best["threshold"])
                    for b in locked_bags])
    saved(Path(out)/"LOCKED_RESEARCH_EVALUATION.json",{
        "schema":SCHEMA,"status":"LOCKED_NEW_SYNTHETIC_SOURCE_TESTED_ONCE",
        "model_binary_sha256":binary_pin,
        "calibration_binding_sha256":cal_sha,
        "threshold_from_calibration_only":best["threshold"],
        "result":held,
        "threshold_search_using_locked":False,
        "locked_data_seen_during_optimizer":False,
        "independent_human_far_frr":"NOT_RUN",
        "physical_ssc305":"NOT_RUN"})
    final={"schema":SCHEMA,"status":"EXECUTED_NEW_CANONICAL_RESEARCH_HEAD",
       "model":{"npz_sha256":npz_pin,"bin_sha256":binary_pin,
                "parameters":49283,"architecture":spec["architecture"]},
       "source_groups_qualified":binding.config()["exact_expected_qualified"],
       "actual_3_way_training_updates":96,
       "feature_model_forwards":progress["actual_frontend_forward_calls"],
       "training_bags":len(train_bags),"development_bags":len(dev_bags),
       "calibration_bags":len(cal_bags),"locked_bags":len(locked_bags),
       "development":aggregate(dev_scores),
       "calibration":best["report"],"calibration_threshold":best["threshold"],
       "locked_research":held,"no_legacy_meero_bags":True,
       "old_head_weights_reused":False,"training_only_machine_silver_weak":True,
       "training_locked_source_leak":False,
       "human_gold":False,"independent_real_far_frr":"NOT_RUN",
       "ssc305":"NOT_RUN","commercial_model_promotion_allowed":False}
    saved(Path(out)/"RESULT.json",final)
    return final

def execute(input_dir,assets,out):
    run=binding.guard()
    bound=binding.load(Path(out)/"ADMISSION.json")
    verified=binding.admission(input_dir)
    need(binding.canon(bound)==binding.canon(verified) and
         bound["status"]=="MACHINE_SILVER_WEAK_RESEARCH_TRAIN_INPUT_BOUND" and
         bound["source_group_counts"]==binding.config()["exact_expected_qualified"],
         "ABSOLUTELY_SAME_PINNED_BOUND_TRAINING_INPUT")
    verify_assets(assets)
    folders={}
    try:
        for p in binding.config()["corpus"]:
            folders[p["key"]]=binding.archive(Path(input_dir)/(p["key"]+"-source.zip"),p["source"])
        saved(Path(out)/"NUMERIC_STARTED.json",{
           "schema":SCHEMA,"run":run,
           "input_binding_sha256":sha(binding.canon(bound)),
           "actual_optimizer_updates":0,"legacy_head_warmstart":False,
           "locked_audio_opened":False})
        frontend=make_frontend(assets)
        result=training_loop(bound["rows"],folders,frontend,out)
    except BaseException as exc:
        saved(Path(out)/"FAILED.json",{
          "schema":SCHEMA,"run":run,"error":type(exc).__name__+":"+str(exc)[:500],
          "optimizer_or_feature_progress":"READ_EXISTING_RECEIPTS_NO_AUTOMATIC_RETRY",
          "preserve_partial_weights":(Path(out)/"WEIGHTS_FROZEN.json").exists(),
          "retry_count":0,"no_false_model_promotion":True})
        raise
    finally:
        for z in folders.values():z.close()
    print(json.dumps({k:v for k,v in result.items()
              if k not in ("development","calibration","locked_research")},
              sort_keys=True))
    return result

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--source-check",action="store_true")
    cli.add_argument("--execute",action="store_true")
    cli.add_argument("--sources")
    cli.add_argument("--assets")
    cli.add_argument("--output")
    args=cli.parse_args()
    need(args.source_check!=args.execute,"ONE_NUMERIC_RESEARCH_STAGE")
    if args.source_check:
        c=binding.config()
        print(json.dumps({"schema":SCHEMA,"head_parameters":49283,
            "max_updates":c["head"]["max_updates"],
            "source_qualified":c["exact_expected_qualified"],
            "calibration_and_locked_after_weight_freeze":True,
            "old_meero_warmstart":False,"new_optimizer_updates":0},
            sort_keys=True))
        return
    need(args.sources and args.assets and args.output,
         "SIX_REVIEWED_SOURCES_TWO_PINNED_ONNX_AND_BOUND_OUTPUT")
    execute(args.sources,args.assets,args.output)

if __name__=="__main__":
    main()
