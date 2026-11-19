"""V5 Hey Mero: one real causal FSMN-CTC refit with NEW source-qualified TRAIN.

Existing mono70: 54 original TRAIN, 16 exposed DEV (the latter NEVER opened).
Fresh24: 11 original new TRAIN voice groups and 2 new held-out DEVELOPMENT
groups, independently reviewed by Qwen, Whisper, Parakeet, XLSR and LV60.
Reuse all source/PCM SHA and Silver/Weak machine tier decisions; UNKNOWN,
old pronunciation, nearphone, calibration and locked are NEVER supervised.

Exactly one preregistered TRAIN-only 256-step AdamW refit warmstarted from
the actual 144-step gain-robust 43,876-param research model. Freeze final
weights before ONE original NEW-DEV-only KWS test and compare to the parent.
No model selection/threshold tuning from DEV or prior opened tests. ONNX/PT/
NPZ full source hashes and all failed receipts are retained. This is ONLY
synthetic acoustic research, not human 3-5m, FAR/FRR or SSC305 acceptance.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import wave
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(name):
    spec=importlib.util.spec_from_file_location(
        "v5_expanded_mono_"+name,HERE/(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

parent=load("english_v5_mero_mono70_robust_train_once")
gain=load("english_v5_mero_gain_invariant_refit_once")
review=load("english_v5_mero_fresh24_saved_screen_once")
full=load("english_v5_mero_locked8_saved_machine_gate_once")
v5=parent.v5
SCHEMA="english-v5-mero-expanded-original-mono70-plus-fresh24-fsmn-v1"
REPO="jiying2007/kws-data"
BEFORE="b90c9b2533899b7b1bd3cbed2cf7ad23a7189c5c"
UPDATES=256
LR=0.00012
SEED=20261126
PARAMETERS=43876
LEVELS=(0.5,0.75,1.0,1.5,2.0)
OLD_COUNT={"train":54,"development":16}
FRESH_QUALIFIED={"train":11,"development":2}
AUDIT={"run":44470829024,"id":14503604626,
       "digest":"sha256:12a1eb39c7a458a466fec2e93ba59e5036319bd9db4b4bdc47e7e6e1c5990f1b"}
PARENT={"run":44475190841,"id":14505594074,
        "bytes":500934,
        "digest":"sha256:876b3a1ccc3c2c00cefabb5a3273a81dd719fb161700ef557f52093ff640d41c",
        "pt":"23d60d56484e687b24ee84f23624aefdf8d612e7900169976efc1f387d4e4ce9",
        "npz":"94086577f345221d4ee5124aed33edb415e46a22732ad5bff065a7f7fa471e8a",
        "onnx":"8a87b530ec236904aa5631f5995beeaaa01cec253f29ecbdb6d73109b57e5332"}
FRESH_SOURCE={"run":44478554596,"id":14507208152,"bytes":1961038,
              "digest":"sha256:2fbc26386136cfd12dddb01074f3c9c4556c8fc7c49819663f2d11a2499eaa1c"}
FRESH_SCREEN={"run":44483128198,"id":14509252921,"bytes":9282,
              "digest":"sha256:e8bcdf32a0ac138de8bdd90c4f6cd6e419d4ccae42bbd393e75b5daac9502158"}
FULL_GATE={"run":44493479564,"id":14512954554,"bytes":5102,
           "digest":"sha256:3b3166cab0d62570f5c68bf1389af9b40fdea2941b985794471621db96a05293"}

def need(value,reason):
    if not value:
        raise ValueError(reason)

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def write_once(path,doc):
    p=Path(path)
    with p.open("x",encoding="utf-8") as out:
        json.dump(doc,out,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        out.write("\n")

def source_check():
    old=parent.source_check()
    g=gain.source_check()
    s=review.source_check()
    f=full.source_check()
    need(old["target"]=="Hey Mero"
         and old["source_audit_artifact"]==AUDIT["id"]
         and old["old_locked_audio"]=="NEVER_OPENED"
         and g["only_keyword"]=="Hey Mero"
         and g["frozen_parent_artifact"]==14504422674
         and g["formal_promotion_allowed"] is False
         and s["status"]=="READBACK_SOURCE_PINNED"
         and s["saved_source_rows"]==96 and s["review_rows"]==64
         and s["training_allowed"] is False
         and f["status"]=="FROZEN_OLD_GATE_AND_FIVE_REAL_NEW_LOCKED_NATIVE_ZIPS"
         and f["prior_groups"]=={"train":11,"development":2,
                                 "calibration":2,"locked-test":1}
         and f["new_native_or_kws_forward_calls"]==0
         and PARAMETERS==43876 and UPDATES==256 and LR==0.00012
         and LEVELS==gain.GAIN_LEVELS
         and PARENT["run"]==44475190841
         and PARENT["id"]==14505594074
         and FRESH_SOURCE["id"]==14507208152
         and FRESH_SCREEN["id"]==14509252921
         and FULL_GATE["id"]==14512954554,
         "EXACT_SINGLE_MERO_OLD70_AND_NEW32_TRAIN_DEV_SOURCE_RESEARCH")
    return {"status":"NEW_TRAIN_ONLY_MODEL_NOT_EXECUTED",
            "target":"Hey Mero","model_parameters":PARAMETERS,
            "max_real_optimizer_updates":UPDATES,
            "old_audited_train_pcm":OLD_COUNT["train"],
            "new_source_qualified_train_groups":FRESH_QUALIFIED["train"],
            "new_source_qualified_dev_groups":FRESH_QUALIFIED["development"],
            "new_observer_model_calls":0,
            "new_tts_calls":0,
            "calibration_or_locked_kws_audio_opened":False,
            "old_exposed_dev_in_gradients":False,
            "production_promotion_allowed":False}

def guard(env=None,event=None,parent_commit=None):
    source_check()
    env=os.environ if env is None else env
    required={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
    need(all(env.get(k)==v for k,v in required.items()),
         "PROTECTED_PUBLIC_MAIN_FIRST_ATTEMPT_NUMERIC_ONLY")
    head=env.get("GITHUB_SHA","")
    need(bool(re.fullmatch("[0-9a-f]{40}",head))
         and bool(re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID",""))),
         "VALID_GITHUB_MAIN_RUN_ID")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "NO_REPLAY_OR_MOVING_UNSEALED_SOURCE")
    if parent_commit is None:
        parent_commit=subprocess.check_output(
            ["git","rev-parse","HEAD^1"],cwd=ROOT,text=True).strip()
    need(parent_commit==BEFORE,"ONE_EXACT_V5_MODEL_MAIN_SUCCESSOR")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":head,
            "parent":BEFORE,"new_tts_asr_calls":0,"old_locked_pcm":False}

def frozen_gate(folder):
    data=json.loads((Path(folder)/"RESULT.json").read_bytes())
    need(data.get("schema")==full.SCHEMA
         and data.get("status")=="PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"
         and data.get("full_6_2_2_2_synthetic_machine_gate") is True
         and data.get("cumulative_machine_group_counts")=={
             "train":11,"development":2,"calibration":2,"locked-test":5}
         and data.get("new_locked_source_groups_machine_qualified")==4
         and data.get("new_tts_asr_phone_model_calls")==0
         and data.get("kws_locked_evaluation")=="NOT_RUN"
         and data.get("new_optimizer_updates")==0
         and data.get("new_model_weights") is False
         and data.get("training_snapshot_allowed") is False
         and data.get("product_promotion_allowed") is False,
         "ORIGINAL_FULL_MACHINE_SOURCE_GATE_PASSED_NOT_MODEL_GATE")
    return {"run":FULL_GATE["run"],
            "status":data["status"],
            "machine_groups":data["cumulative_machine_group_counts"],
            "research_only":True}

def screen_new(source,packet):
    """Re-bind strictly original TRAIN/DEV Silver/Weak to a NEW research snapshot."""
    root=Path(source)
    original=review.observer.source_rows(root)
    by_id={row["id"]:row for row in original}
    record=json.loads((Path(packet)/"SCREEN.json").read_bytes())
    need(record.get("schema")==review.SCHEMA
         and record.get("status")=="PASS_NEW_TRAIN_DEV_MACHINE_SCREEN_ONLY"
         and record.get("source_run")==FRESH_SOURCE["run"]
         and record.get("original_observer_run")==review.ORIGINAL_RUN
         and record.get("source_group_counts")==FRESH_QUALIFIED
         and record.get("tier_counts")=={"silver":25,"weak":7,"unknown":32}
         and record.get("observed_label_counts")=={"0":16,"2":16}
         and record.get("full_6_2_2_2_gate_passed") is False
         and record.get("training_snapshot_allowed") is False
         and record.get("locked_kws_test_opened") is False
         and len(record.get("rows",[]))==64
         and len(by_id)==64,
         "FROZEN_FIVE_MODEL_SCREEN_NOT_RELABELED")
    seen=set();groups=defaultdict(set);rows=[];roles={}
    for d in record["rows"]:
        sid=d.get("id")
        need(sid in by_id and sid not in seen,
             "EXACT_64_ORIGINAL_SOURCE_SCREEN_IDS")
        seen.add(sid)
        o=by_id[sid]
        need(d["source_group_id"]==o["source_group_id"]
             and d["role"]==o["role"]
             and d["requested_class"]==o["requested_class"]
             and d["pcm_sha256"]==o["pcm_sha256"]
             and d["wav_sha256"]==o["wav_sha256"]
             and d["speaker_index"]==o["speaker_index"]
             and d["human_gold"] is False
             and d["training_allowed"] is False
             and d["role"] in ("train","development"),
             "SOURCE_ROLE_AND_WHOLE_PCM_IMMUTABLE")
        group=d["source_group_id"]
        need(group not in roles or roles[group]==d["role"],
             "NO_SYNTHETIC_SPEAKER_ROLE_CROSSOVER")
        roles[group]=d["role"]
        if d["tier"] in ("silver","weak"):
            need(d["observed_class"] in (0,2)
                 and d["observed_class"]==d["requested_class"]
                 and d["cell_id"]!="nearphone-meero-challenge",
                 "NEVER_USE_OLD_MEERO_OR_REQUESTED_UNVERIFIED_LABEL")
            groups[group].add(d["observed_class"])
            rows.append({**o,"tier":d["tier"],
                         "mono_class":int(d["observed_class"]==2),
                         "source_pcm_sha256":o["pcm_sha256"],
                         "research_training_bind":d["role"]=="train",
                         "research_dev_only":d["role"]=="development",
                         "origin":"new_frozen_piper_source96"})
        else:
            need(d["tier"]=="unknown" and d["observed_class"] is None,
                 "UNKNOWN_NEVER_SUPERVISED_OR_NEGATIVE")
    need(seen==set(by_id) and len(roles)==16,
         "WHOLE_EXACT_ORIGINAL_64_SOURCE_COHORT")
    qualified={g for g,classes in groups.items() if classes=={0,2}}
    selected=sorted((r for r in rows if r["source_group_id"] in qualified),
                    key=lambda x:x["id"])
    qcounts=Counter(roles[g] for g in qualified)
    need({"train":qcounts["train"],"development":qcounts["development"]}
          ==FRESH_QUALIFIED
         and all(r["source_group_id"] in qualified for r in selected)
         and all(r["mono_class"] in (0,1) for r in selected)
         and all(r["role"] in ("train","development") for r in selected),
         "EXACT_FRESH_MACHINE_11_TRAIN_2_NEW_DEV_SOURCE")
    return selected,{
        "original_source_clips":96,
        "machine_reviewed_train_dev_clips":64,
        "qualified_groups":FRESH_QUALIFIED,
        "selected_by_role":dict(Counter(r["role"] for r in selected)),
        "selected_by_label":dict(Counter(r["mono_class"] for r in selected)),
        "old_audio_labels_modified":False,
        "calibration_locked_supervised":False,
        "nearphone_ever_negative":False}

def parent_model(zip_path):
    p=Path(zip_path)
    need(p.is_absolute() and p.is_file() and not p.is_symlink()
         and sha(p.read_bytes())==PARENT["digest"][7:],
         "EXACT_FROZEN_GAIN_PARENT_ARCHIVE_SHA")
    with zipfile.ZipFile(p) as z:
        paths={
          "pt":"models/mero-mono70-gain-refit.pt",
          "npz":"models/mero-mono70-gain-refit.npz",
          "onnx":"models/mero-mono70-gain-refit.onnx"}
        need(set(paths.values())<=set(z.namelist())
             and {"RESULT.json","TRAINING.json","ADMISSION.json"}<=set(z.namelist()),
             "ACTUAL_GAIN_RESEARCH_RESULT_AND_THREE_WEIGHT_FILES")
        report=json.loads(z.read("RESULT.json"))
        need(report.get("schema")==gain.SCHEMA
             and report.get("status")=="GAIN_RESEARCH_NOT_QUALIFIED"
             and report.get("new_optimizer_updates")==144
             and report.get("parent_model_artifact_id")==14504422674
             and report.get("product_promotion_allowed") is False
             and report.get("development",{}).get("positive_original")==9
             and report.get("development",{}).get("positive_misses")==1
             and report.get("development",{}).get("ordinary_false_or_extra")==0,
             "ONE_REAL_SAVED_144_STEP_PARENT_AND_FAILED_DEV")
        blobs={k:z.read(name) for k,name in paths.items()}
        need(all(sha(blobs[k])==PARENT[k] for k in paths),
             "PARENT_PT_NPZ_ONNX_BYTE_AUTHENTICATION")
    return blobs,report

def readpcm(row,old_archives,newdir):
    if row.get("origin")=="new_frozen_piper_source96":
        path=Path(newdir)/"wav"/(row["id"]+".wav")
        need(path.is_file() and not path.is_symlink()
             and sha(path.read_bytes())==row["wav_sha256"],
             "ORIGINAL_NEW_FULL_WAV_BYTES")
        with wave.open(str(path),"rb") as f:
            need(f.getnchannels()==1 and f.getframerate()==16000
                 and f.getsampwidth()==2 and f.getnframes()==row["samples"]
                 and f.getcomptype()=="NONE",
                 "ORIGINAL_COMPLETE_16K_PCM16_NO_CROP")
            data=f.readframes(f.getnframes())
        need(len(data)==2*row["samples"]
             and sha(data)==row["source_pcm_sha256"],
             "ORIGINAL_NEW_SOURCE_PCM_SHA")
        return data
    data=v5.base.load_source_pcm(row,old_archives)
    need(sha(data)==row["source_pcm_sha256"],
         "OLD_70_ORIGINAL_UNCHANGED_TRAIN_PCM")
    return data

def train_and_eval(old_train,new_train,new_dev,old_archives,newdir,weights,out):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    alltrain=sorted(old_train+new_train,key=lambda x:(x["source_group_id"],x["id"]))
    need(len(old_train)==54
         and len({r["source_group_id"] for r in old_train})==24
         and len({r["source_group_id"] for r in new_train})==11
         and len({r["source_group_id"] for r in new_dev})==2
         and not ({r["source_group_id"] for r in alltrain}&
                  {r["source_group_id"] for r in new_dev}),
         "GROW_ONLY_TRAIN_AND_NEW_SOURCE_DISJOINT_DEV")
    with np.load(io.BytesIO(weights["npz"]),allow_pickle=False) as original:
        bank=original["mel_filters"].copy()
        mean=original["train_mean"].copy()
        scale=original["train_scale"].copy()
    need(bank.shape==(257,40) and mean.shape==(40,)
         and scale.shape==(40,) and np.isfinite(bank).all()
         and np.isfinite(mean).all() and np.isfinite(scale).all()
         and (scale>0).all(),
         "FROZEN_PARENT_TRAIN_ONLY_NONDEV_FRONTEND")
    parent_net=v5.build_model(torch)
    state=torch.load(io.BytesIO(weights["pt"]),
                     map_location="cpu",weights_only=True)
    parent_net.load_state_dict(state,strict=True)
    parent_net.eval()
    for param in parent_net.parameters():param.requires_grad_(False)
    student=v5.build_model(torch)
    student.load_state_dict(state,strict=True)
    need(sum(p.numel() for p in student.parameters())==PARAMETERS,
         "EXACT_SMALL_CAUSAL_FSMN_CTC_PARAM_COUNT")
    cache={}
    for row in alltrain+new_dev:
        data=readpcm(row,old_archives,newdir)
        cache[row["id"]]=v5.features(data,row,torch,np,bank)
    teachers={}
    with torch.no_grad():
        for row in old_train:
            x=torch.from_numpy(((cache[row["id"]]-mean)/scale)
                               .astype(np.float32)).unsqueeze(0)
            teachers[row["id"]]=parent_net(x).softmax(-1).detach()
    grouped={0:defaultdict(list),1:defaultdict(list)}
    for row in alltrain:
        grouped[row["mono_class"]][row["source_group_id"]].append(row)
    need(len(grouped[0])==35 and len(grouped[1])==35,
         "BOTH_CLASSES_IN_ALL_35_REAL_TRAIN_SPEAKER_GROUPS")
    optimizer=torch.optim.AdamW(student.parameters(),
                                lr=LR,weight_decay=0.001)
    trace=[];began=time.perf_counter()
    for step in range(UPDATES):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for cls in (0,1):
            families=sorted(grouped[cls])
            gid=families[step%len(families)]
            members=sorted(grouped[cls][gid],key=lambda r:r["id"])
            row=members[(step//len(families))%len(members)]
            gainid=(step//len(families)+
                    int(sha(row["source_pcm_sha256"].encode())[:2],16))%len(LEVELS)
            scaled=gain.derived_level(cache[row["id"]],LEVELS[gainid],np)
            x=torch.from_numpy(((scaled-mean)/scale).astype(np.float32)).unsqueeze(0)
            logits=student(x)
            pos,blank,margin=parent.ctc_losses(logits,torch)
            loss=(pos+F.softplus(0.5-4.0*margin) if cls else
                  0.35*blank+F.softplus(0.5+4.0*margin))
            if row["tier"]=="weak":
                loss=loss*0.75
            if row["id"] in teachers:
                loss=loss+0.12*F.kl_div(
                    logits.log_softmax(-1),teachers[row["id"]],
                    reduction="none").sum(-1).mean()
            need(bool(torch.isfinite(loss)),"FINITE_SOURCE_BALANCED_TRAIN_LOSS")
            losses.append(loss)
        total=sum(losses)/2
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),5.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        if step%16==0 or step==UPDATES-1:
            trace.append({"step":step+1,
                          "loss":float(total.detach()),
                          "ordinary":float(losses[0].detach()),
                          "positive":float(losses[1].detach())})
    student.eval()
    model_dir=out/"models";model_dir.mkdir()
    pt=model_dir/"mero-fsmn-ctc-expanded256.pt"
    torch.save(student.state_dict(),pt)
    reloaded=v5.build_model(torch)
    reloaded.load_state_dict(torch.load(pt,map_location="cpu",
                                       weights_only=True),strict=True)
    reloaded.eval()
    with torch.no_grad():
        probe=torch.from_numpy(
            ((cache[new_dev[0]["id"]]-mean)/scale).astype(np.float32)).unsqueeze(0)
        need(bool(torch.equal(student(probe),reloaded(probe))),
             "EXACT_EXPANDED_MODEL_RELOAD_NUMERIC_PARITY")
    npz=model_dir/"mero-fsmn-ctc-expanded256.npz"
    np.savez(npz,mel_filters=bank,train_mean=mean,train_scale=scale,
             **{k:v.detach().cpu().numpy()
                for k,v in reloaded.state_dict().items()})
    onnx=model_dir/"mero-fsmn-ctc-expanded256.onnx"
    torch.onnx.export(reloaded,probe,onnx.as_posix(),
        input_names=["fbank40"],output_names=["frame_ctc_logits"],
        opset_version=17,dynamo=False,
        dynamic_axes={"fbank40":{1:"frames"},
                      "frame_ctc_logits":{1:"frames"}})
    need(onnx.exists() and onnx.stat().st_size>10000,
         "ACTUAL_EXPANDED_FSMN_CTC_ONNX_MODEL_SAVED")
    # New source-disjoint DEV is evaluated once AFTER frozen final weights,
    # unchanged PCM and unchanged parent normalization; no fitting decisions.
    outcomes=[]
    with torch.no_grad():
        for row in new_dev:
            x=torch.from_numpy(((cache[row["id"]]-mean)/scale)
                               .astype(np.float32)).unsqueeze(0)
            before=v5.ctc_greedy(parent_net(x),torch)
            after=v5.ctc_greedy(reloaded(x),torch)
            _,_,margin=parent.ctc_losses(reloaded(x),torch)
            outcomes.append({
               "id":row["id"],"source_group_id":row["source_group_id"],
               "pcm_sha256":row["source_pcm_sha256"],"role":"development",
               "positive":bool(row["mono_class"]),
               "parent_event":before["mero_event"],
               "new_event":after["mero_event"],
               "new_extra_events":after["extra_events"],
               "new_full_sequence_minus_blank_per_frame":float(margin.detach())})
    positives=[o for o in outcomes if o["positive"]]
    negatives=[o for o in outcomes if not o["positive"]]
    need(len(positives)>=2 and len(negatives)>=2,
         "AT_LEAST_TWO_NEW_SOURCE_DEV_POSITIVES_AND_ORDINARY")
    beforehits=sum(r["parent_event"] for r in positives)
    afterhits=sum(r["new_event"] for r in positives)
    newfalse=sum(r["new_event"]+r["new_extra_events"] for r in negatives)
    status=("RESEARCH_NEW_SOURCE_DEV_GAIN_WITHOUT_ORDINARY_EVENTS"
            if afterhits>beforehits and newfalse==0
            else "RESEARCH_NEW_SOURCE_DEV_NOT_QUALIFIED")
    result={"schema":SCHEMA,"status":status,
       "candidate":"43_876_param_causal_fsmn_ctc_expanded_train",
       "model_parameters":PARAMETERS,
       "parent_gain_model_artifact_id":PARENT["id"],
       "new_actual_optimizer_updates":UPDATES,
       "learning_rate":LR,"seed":SEED,
       "train_gain_grid":LEVELS,
       "old_train_original_pcm":len(old_train),
       "new_train_original_pcm":len(new_train),
       "new_train_source_groups":11,
       "combined_train_source_groups":35,
       "new_unexposed_dev_source_groups":2,
       "new_dev_original_pcm":len(new_dev),
       "development":{"positive":len(positives),
         "parent_positive_hits":beforehits,
         "student_positive_hits":afterhits,
         "ordinary":len(negatives),
         "student_false_or_extra":newfalse,
         "source_disjoint_and_no_dev_gradients":True,
         "zero_gain_or_threshold_tuning":True,
         "rows":outcomes},
       "train_only_parent_posterior_consistency":True,
       "new_tts_asr_phone_model_calls":0,
       "old_exposed_dev_pcm_opened":False,
       "calibration_or_locked_pcm_opened":False,
       "kws_locked_evaluation":"NOT_RUN",
       "new_model_weights":{p.name:{"bytes":p.stat().st_size,
           "sha256":sha(p.read_bytes())} for p in (pt,npz,onnx)},
       "train_cpu_seconds":round(time.perf_counter()-began,3),
       "loss_trace":trace,
       "real_far_frr":"NOT_MEASURED",
       "human_3_5m":"NOT_RUN","ssc305":"NOT_RUN",
       "product_promotion_allowed":False}
    write_once(out/"TRAINING.json",{
       "schema":SCHEMA,"parent_pt_sha256":PARENT["pt"],
       "new_optimizer_updates":UPDATES,
       "frozen_train_only_norm":True,
       "predeclared_loss_and_gain":True,
       "old_dev_training":False,
       "fresh_dev_training":False,
       "calibration_locked_training":False,
       "new_train_and_old_train_source_groups":35,
       "loss_trace":trace})
    write_once(out/"RESULT.json",result)
    return result

def execute(old_archives,old_audit,old_model,
            fresh_source,fresh_screen,gate_dir,out):
    source_check()
    run=guard()
    out=Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "EXACT_NEW_EXPANDED_RESEARCH_MODEL_OUTPUT_DIR")
    out.mkdir()
    write_once(out/"STARTED.json",{
        "schema":SCHEMA,"run":run,"new_optimizer_budget":UPDATES,
        "prior_gain_artifact_id":PARENT["id"],
        "fresh_source_artifact_id":FRESH_SOURCE["id"],
        "fresh_screen_artifact_id":FRESH_SCREEN["id"],
        "locked_kws_opened":False,"retry":False})
    handles={}
    try:
        gate=frozen_gate(gate_dir)
        old,old_summary=parent.authorized_source(old_audit,old_archives)
        fresh,fresh_summary=screen_new(fresh_source,fresh_screen)
        oldtrain=[r for r in old if r["role"]=="train"]
        newtrain=[r for r in fresh if r["role"]=="train"]
        newdev=[r for r in fresh if r["role"]=="development"]
        need(len(oldtrain)==54
             and len({r["source_group_id"] for r in oldtrain})==24
             and len({r["source_group_id"] for r in newtrain})==11
             and len({r["source_group_id"] for r in newdev})==2
             and not ({r["source_group_id"] for r in old}&
                      {r["source_group_id"] for r in fresh})
             and not ({r["source_pcm_sha256"] for r in old}&
                      {r["source_pcm_sha256"] for r in fresh}),
             "OLD_NEW_MACHINE_SOURCE_DISJOINT_BY_SPEAKER_AND_PCM")
        blobs,parent_report=parent_model(old_model)
        write_once(out/"ADMISSION.json",{
            "schema":SCHEMA,"machine_gate":gate,
            "old_source_audit":old_summary,
            "new_frozen_screen":fresh_summary,
            "old_train_rows":len(oldtrain),
            "new_train_rows":len(newtrain),
            "new_dev_rows":len(newdev),
            "research_train_machine_silver_weak_only":True,
            "source_roles_not_reassigned":True,
            "original_training_allowed_flags_unmodified":True,
            "calibration_locked_pcm_never_opened":True,
            "parent_model_rejected_reused_as_research_initialization":True,
            "research_only_no_product_release":True})
        for spec in v5.binding.config()["corpus"]:
            key=spec["key"]
            handles[key]=v5.binding.archive(
                Path(old_archives)/(key+"-source.zip"),spec["source"])
        handles["newtrain24"]=v5.binding.archive(
            Path(old_archives)/"newtrain24-source.zip",v5.prior.SOURCE)
        result=train_and_eval(oldtrain,newtrain,newdev,handles,fresh_source,
                              blobs,out)
        print(json.dumps({"status":result["status"],
           "optimizer_updates":result["new_actual_optimizer_updates"],
           "new_train_groups":11,
           "new_dev_results":{k:v for k,v in result["development"].items()
                              if k!="rows"},
           "model_sha256":result["new_model_weights"][
                    "mero-fsmn-ctc-expanded256.onnx"]["sha256"],
           "product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        write_once(out/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_EXPANDED_FSMN_NOT_QUALIFIED",
             "exception_type":type(exc).__name__,"reason":str(exc)[:520],
             "no_automatic_retry":True,
             "no_calibration_or_locked_training":True})
        raise
    finally:
        for z in handles.values():z.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--guard",action="store_true")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--old-archives",type=Path)
    p.add_argument("--old-audit",type=Path)
    p.add_argument("--old-model",type=Path)
    p.add_argument("--fresh-source",type=Path)
    p.add_argument("--fresh-screen",type=Path)
    p.add_argument("--source-gate",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.execute))==1,
         "EXACT_ONE_REAL_EXPANDED_FSMN_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True))
    elif a.guard:
        print(json.dumps(guard(),sort_keys=True))
    else:
        need(all((a.old_archives,a.old_audit,a.old_model,
                  a.fresh_source,a.fresh_screen,a.source_gate,a.output)),
             "EXACT_OLD_AND_NEW_SOURCE_ARCHIVES_REQUIRED")
        execute(a.old_archives,a.old_audit,a.old_model,
                a.fresh_source,a.fresh_screen,a.source_gate,a.output)

if __name__=="__main__":
    main()
