#!/usr/bin/env python3
"""B3 Hey Mero: exact trained PT/NPZ/ONNX -> real C11 and Cortex-A32 software parity.

Original 14 source-reviewed DEV PCM are REUSED ONLY for numerical/transport checks,
not model selection, independent FRR/FAR, training, threshold tuning or product
release. One immutable B3 43,876-parameter model, same existing C11 runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent

def imported(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

v6=imported(HERE/"english_v6_mero_frozen_b0_b1_paired_once.py","v6_b3_originals")
REPO="jiying2007/kws-data"
PIPELINE_REPO="jiying2007/kws-pipeline"
PIPELINE_SHA="e3261e976965c52bcd0989591abab606d9cdc0e0"
BEFORE="0ff1255055277b82c523addd3411b6152ace9da6"
SCHEMA="english-v6-b3-43k-c11-cortexa32-original-software-parity-v1"
SOURCE=v6.SOURCE
SCREEN=v6.SCREEN
B3={
 "id":14590093798,"run":44651053897,"head":BEFORE,
 "name":"english-v6-mero43-suffixctc256-r4-44651053897",
 "bytes":503345,
 "sha":"59373f0e7e5f9d4ed35d6b116cdd6f8ac8eb5abcfa1acb5311ef658c59a20e04",
 "stem":"mero43-onset256",
 "pt":"682e2689e82aa91d850516c06b2be61efc64731cb1dee3b70a7d2a8e01925a46",
 "npz":"cf3022ea7dd29e7e830b73ce613485a18d1d300467605cb330447cdce6e3ee89",
 "onnx":"5ca791d495f6c57a3608bc412effca15924e9e32e088d45d6eb96e8f97092031",
}
MAX_ABS=0.005
MAX_MEAN_ABS=0.0005
ORIGINAL_COUNT=14
CONDITIONS=("original","one_second_zero_prefix")
COMPILERS=("gcc","clang","cortex-a32-qemu")

def need(value,why):
    if not value: raise ValueError(why)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def store(path,data):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(data,f,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)
        f.write("\n")

def source_check():
    x=v6.source_check()
    need(x["status"]=="SOURCE_ONLY_NO_B0_B1_AUDIO_FORWARDS"
         and x["keyword"]=="Hey Mero"
         and x["new_reviewed_positive"]==6
         and x["new_reviewed_ordinary"]==8
         and x["unknown_excluded"]==18
         and v6.SOURCE["id"]==14543254675
         and v6.SCREEN["id"]==14545584039
         and PIPELINE_SHA=="e3261e976965c52bcd0989591abab606d9cdc0e0"
         and B3["run"]==44651053897 and B3["head"]==BEFORE
         and B3["stem"]=="mero43-onset256"  # old internal filename is intentional
         and len(B3["pt"])==64 and len(B3["npz"])==64
         and len(B3["onnx"])==64
         and ORIGINAL_COUNT==14 and len(CONDITIONS)==2
         and MAX_ABS==0.005 and MAX_MEAN_ABS==0.0005,
         "B3_STANDALONE_MODEL_AND_FROZEN_DEV_SOFTWARE_ONLY")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_ACOUSTIC_FORWARDS",
        "model_artifact":B3["id"],"model_zip_sha256":B3["sha"],
        "model_parameters":43876,"original_reviewed_dev_positive":6,
        "original_reviewed_dev_ordinary":8,"unknown_excluded":18,
        "expected_native_and_arm_conditions":list(CONDITIONS),
        "C11_source_sha":PIPELINE_SHA,
        "max_logit_tolerance":MAX_ABS,"mean_tolerance":MAX_MEAN_ABS,
        "new_training":False,"model_selection_allowed":False,
        "locked_calibration_opened":False,"physical_ssc305":False,
        "product_promotion":False}

def provenance(token):
    import urllib.request
    need(type(token) is str and len(token)>10,
         "GITHUB_READ_ONLY_PROVENANCE_REQUIRED")
    base="https://api.github.com/repos/"+REPO+"/actions"
    def get(suffix):
        req=urllib.request.Request(base+suffix,headers={
            "Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "User-Agent":"kws-v6-b3-parity-provenance"})
        with urllib.request.urlopen(req,timeout=35) as f:
            data=f.read(1048577)
        need(len(data)<=1048576,"PROVENANCE_BOUNDED_RESPONSE")
        return json.loads(data)
    r=get("/runs/"+str(B3["run"]))
    need(r.get("id")==B3["run"] and r.get("head_sha")==B3["head"]
         and r.get("head_branch")=="main" and r.get("event")=="push"
         and r.get("run_attempt")==1
         and r.get("status")=="completed" and r.get("conclusion")=="success",
         "B3_ACTUAL_PROTECTED_MAIN_TRAINING_PASSED")
    for label,item in (("B3",B3),("SOURCE",SOURCE),("SCREEN",SCREEN)):
        record=get("/artifacts/"+str(item["id"]))
        name=item.get("name")
        digest=item.get("sha")
        need(record.get("id")==item["id"]
             and record.get("name")==name
             and record.get("digest")=="sha256:"+digest
             and record.get("expired") is False
             and record.get("workflow_run",{}).get("id")==item["run"],
             "REAL_ORIGINAL_ARTIFACT_BYTES_"+label)
        if label=="B3":
            need(record.get("size_in_bytes")==B3["bytes"]
                 and record.get("workflow_run",{}).get("head_sha")==B3["head"],
                 "B3_SOURCE_RUN_SHA_AND_ZIP_LENGTH")
    return {"B3_run":B3["run"],"B3_artifact":B3["id"],
            "source_artifact":SOURCE["id"],"saved_source_screen":SCREEN["id"],
            "no_new_model_or_wav":True}

def guard():
    source_check()
    env=os.environ
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"FIRST_MAIN_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[a-f0-9]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "GITHUB_EXECUTION_CONTEXT")
    e=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(e.get("before")==BEFORE and e.get("after")==head
         and e.get("ref")=="refs/heads/main"
         and e.get("repository",{}).get("full_name")==REPO
         and e.get("repository",{}).get("private") is False
         and e.get("forced") is False and e.get("deleted") is False
         and subprocess.check_output(["git","rev-parse","HEAD^1"],
                    cwd=ROOT,text=True).strip()==BEFORE,
         "B3_C_PARITY_UNIQUE_FIRST_MAIN_SUCCESSOR")
    return {"run_id":env["GITHUB_RUN_ID"],"main_sha":head,
            "model_pt_sha256":B3["pt"],"pipeline_sha":PIPELINE_SHA}

def audit(source,screen,model_zip):
    source_check()
    rows,receipt=v6.select(source,screen)
    need(len(rows)==ORIGINAL_COUNT
         and sum(x["mono_class"] for x in rows)==6
         and len({r["source_group_id"] for r in rows})==8
         and receipt["no_locked_or_calibration"] is True,
         "ONLY_14_NEW_SYNTHETIC_DEV_ROLE_NO_HUMAN_GOLD")
    z=Path(model_zip)
    need(z.is_file() and not z.is_symlink()
         and z.stat().st_size==B3["bytes"]
         and sha(z.read_bytes())==B3["sha"],
         "B3_ORIGINAL_MODEL_ARCHIVE_IMMUTABLE_SHA")
    with zipfile.ZipFile(z) as archive:
        expected={"STARTED.json","SOURCE.json","RESULT.json"}
        expected|={f"models/{B3['stem']}.{ext}" for ext in ("pt","npz","onnx")}
        need(set(archive.namelist())==expected,
             "B3_ARCHIVE_NO_UNKNOWN_OR_DUPLICATE_MEMBERS")
        result=json.loads(archive.read("RESULT.json"))
        need(result.get("status")=="RESEARCH_TRAIN_ONLY_SUFFIX_CTC_MODEL_NOT_PROMOTABLE"
             and result.get("schema")=="english-v6-mero43-trainonly-suffix-ctc256-r4-recovery-v1"
             and result.get("actual_optimizer_steps")==256
             and result.get("model_parameter_count")==43876
             and result.get("source_train_groups")==35
             and result.get("only_changed_training_factor")=="CTC_ON_KNOWN_SPEECH_SUFFIX"
             and result.get("old_dev_model_forwards")==0
             and result.get("new_dev_model_forwards")==0
             and result.get("cal_locked_access") is False
             and result.get("product_promotion_allowed") is False,
             "REAL_SUFFIX_ONLY_256_UPDATE_B3_NO_TEST_LEAKAGE")
        for ext in ("pt","npz","onnx"):
            name=f"{B3['stem']}.{ext}"
            data=archive.read("models/"+name)
            expected_sha=B3[ext]
            reported=result["new_model_hashes"][name]
            need(sha(data)==expected_sha
                 and reported["sha256"]==expected_sha
                 and reported["bytes"]==len(data),
                 "B3_HASHED_MEMBER_"+ext)
    return rows,receipt

def pipeline_files(pipeline):
    p=Path(pipeline)
    directory=p/"research"/"english_v5_mero"
    need(directory.is_dir() and all((directory/name).is_file() for name in (
        "pack_mero43.py","mero43_stream.c","mero_ctc.c",
        "mero43_host_cli.c","mero43_stream.h","mero_ctc.h",
        "test_mero43_stream.c")), "PINNED_COMPLETE_C11_SOURCES_REQUIRED")
    need(subprocess.check_output(["git","-C",str(p),"rev-parse","HEAD"],
                                text=True).strip()==PIPELINE_SHA,
         "EXACT_PINNED_C11_SOURCE_REVISION")
    return directory

def load_b3(torch,np,archive_path,packer):
    with zipfile.ZipFile(archive_path) as z:
        pt=z.read(f"models/{B3['stem']}.pt")
        npz=z.read(f"models/{B3['stem']}.npz")
    state=torch.load(io.BytesIO(pt),weights_only=True,map_location="cpu")
    model=v6.v5.build_model(torch)
    model.load_state_dict(state,strict=True)
    model.eval()
    packed=[]
    with np.load(io.BytesIO(npz),allow_pickle=False) as arrays:
        shape_spec=packer.geometry()
        need(set(arrays.files)=={name for name,_ in shape_spec},
             "EXACT_FLOAT32_TENSOR_KEYSET")
        for name,shape in shape_spec:
            value=arrays[name]
            need(value.shape==shape and value.dtype==np.dtype("float32")
                 and np.isfinite(value).all(),"B3_FULL_MODEL_TENSOR_SHAPE_"+name)
            if name in state:
                need(np.array_equal(value,state[name].detach().cpu().numpy()),
                     "B3_PT_NPZ_IDENTICAL_"+name)
            if name=="train_scale":need(bool(np.all(value>0)), "TRAIN_NORMALIZER_SCALE")
            if name=="mel_filters":
                need(bool(np.all(value>=0) and np.all(value<=1)
                          and np.all(value.sum(axis=0)>0)),"VALID_C11_MEL_BANK")
            packed.append(np.ascontiguousarray(value,dtype="<f4").tobytes())
        mel=arrays["mel_filters"].copy()
        mean=arrays["train_mean"].copy()
        scale=arrays["train_scale"].copy()
    buf=struct.pack("<8sII",b"MERO43V1",1,54236)+b"".join(packed)
    need(len(buf)==216960 and sha(buf)!=B3["sha"],
         "B3_PORTABLE_C11_PACK_SIZE_AND_DISTINCT_SHA")
    return model,mel,mean,scale,buf

def compile_c(directory,variant,out):
    if variant=="cortex-a32-qemu":
        cc="arm-linux-gnueabihf-gcc"
        flags=["-O2","-fno-fast-math","-ffp-contract=off",
              "-fexcess-precision=standard","-mcpu=cortex-a32",
              "-mfpu=neon-vfpv4","-mfloat-abi=hard","-static"]
    else:
        cc=variant
        flags=["-O2","-ffp-contract=off"]
    cmd=[cc,"-std=c11","-Wall","-Wextra","-Werror","-pedantic",
          *flags,"-I",str(directory),
          str(directory/"mero43_stream.c"),str(directory/"mero_ctc.c"),
          str(directory/"mero43_host_cli.c"),"-lm","-o",str(out)]
    done=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=125)
    need(not done.stderr.strip() and out.is_file() and out.stat().st_size>10000,
         "C11_B3_COMPILATION_"+variant)
    if variant=="cortex-a32-qemu":
        elf=subprocess.check_output(["readelf","-h",str(out)],text=True,timeout=20)
        need("ELF32" in elf and "ARM" in elf
             and "little endian" in elf and "hard-float ABI" in elf,
             "REQUIRE_REAL_ARM_HARDFLOAT_ELF")
    return out

def c_events(file,frames):
    lines=file.read_text(encoding="ascii").splitlines()
    need(len(lines)==frames,"EXACT_FRAME_COUNT_EVENT_CLOCK")
    results=[]
    for i,line in enumerate(lines):
        columns=line.split(",")
        need(len(columns)==3 and
             all(re.fullmatch("[0-9]+",v) for v in columns),
             "C_CTC_CSV_CLOCK")
        frame,sample,ev=map(int,columns)
        need(frame==i and sample==512+160*i and ev in (0,1),
             "C_CTC_MONOTONIC_SAMPLE_CLOCK")
        if ev:results.append(sample)
    return results

def execute(source,screen,b3zip,pipeline,output):
    import numpy as np
    import torch
    token=os.environ.get("GH_TOKEN")
    identity=guard()
    need(output.is_absolute() and output.parent.is_dir() and not output.exists(),
         "NEW_IMMUTABLE_PARITY_OUTPUT")
    output.mkdir()
    store(output/"STARTED.json",{"schema":SCHEMA,"run":identity,
        "training":False,"model_selection":False,"old_test_audio_opened":False})
    completed=[]
    try:
        pins=provenance(token)
        original,source_receipt=audit(source,screen,b3zip)
        native=pipeline_files(pipeline)
        packer=imported(native/"pack_mero43.py","v6_b3_c11_frozen_geometry")
        need(packer.PACK_BYTES==216960 and
             packer.PACK_FLOAT_COUNT==54236
             and packer.MAGIC==b"MERO43V1",
             "SAME_C11_PACK_ABI_AS_B0_NO_SOURCE_MODIFICATION")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        model,mel,mean,scale,packed=load_b3(torch,np,b3zip,packer)
        pack=output/"b3-immutable-temp.pack"
        pack.write_bytes(packed)
        binaries={kind:compile_c(native,kind,output/("b3-"+kind))
                  for kind in COMPILERS}
        store(output/"ADMISSION.json",{
           "model":pins,"source":source_receipt,
           "zip_sha256":B3["sha"],"model_pt_sha256":B3["pt"],
           "pack_sha256":sha(packed),"pack_bytes":len(packed),
           "pipeline_revision":PIPELINE_SHA,"no_fresh_heldout_assessment":True})
        started=time.perf_counter()
        matrix=[]
        for row in original:
            raw=v6.pcm(row,source)
            for condition in CONDITIONS:
                pcm=(b"\0"*32000+raw) if condition=="one_second_zero_prefix" else raw
                derived={**row,"samples":len(pcm)//2}
                feature=v6.v5.features(pcm,derived,torch,np,mel)
                x=torch.from_numpy(((feature-mean)/scale).astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    pt=model(x).squeeze(0).detach().cpu().numpy()
                need(pt.ndim==2 and pt.shape[1]==4 and pt.shape[0]>35 and
                     np.isfinite(pt).all(),"B3_REAL_ORIGINAL_AUDIO_TORCH_FRAME_LOGITS")
                path=output/("scratch-"+re.sub("[^0-9a-zA-Z_-]","_",row["id"])+
                            "-"+condition)
                path.mkdir()
                pcmfile=path/"input.pcm"
                pcmfile.write_bytes(pcm)
                py_ctc=v6.v5.ctc_greedy(torch.from_numpy(pt).unsqueeze(0),torch)
                record={"id":row["id"],"condition":condition,
                    "pcm_sha256":sha(pcm),"original_pcm_sha256":row["source_pcm_sha256"],
                    "positive_machine_only":bool(row["mono_class"]),
                    "source_group_id":row["source_group_id"],
                    "frame_count":int(pt.shape[0]),
                    "torch_full_word_ctc_events":
                        int(py_ctc["mero_event"])+int(py_ctc["extra_events"]),
                    "torch_first_argmax":int(pt[0].argmax()),"runtimes":{}}
                for kind,exe in binaries.items():
                    outlog=path/(kind+".f32")
                    eventlog=path/(kind+".csv")
                    command=([str(exe)] if kind!="cortex-a32-qemu"
                        else ["qemu-arm",str(exe)])
                    command += [str(pack),str(pcmfile),str(outlog),str(eventlog)]
                    done=subprocess.run(command,check=True,text=True,
                               capture_output=True,timeout=130)
                    need("MERO43_C_PCM_STREAM_OK" in done.stderr,
                         "ACTUAL_C11_ORIGINAL_PCM_EXECUTION_"+kind)
                    rawlog=outlog.read_bytes()
                    need(len(rawlog)==pt.shape[0]*16,
                         "REAL_C11_LOGITS_BYTE_SHAPE_"+kind)
                    candidate=np.frombuffer(rawlog,dtype="<f4").reshape(-1,4)
                    need(candidate.shape==pt.shape and np.isfinite(candidate).all(),
                         "FINITE_SAME_FRAME_C11_"+kind)
                    errors=np.abs(candidate.astype(np.float64)-pt.astype(np.float64))
                    frame_equal=bool(np.array_equal(
                        candidate.argmax(-1),pt.argmax(-1)))
                    ctc=v6.v5.ctc_greedy(torch.from_numpy(
                              candidate.copy()).unsqueeze(0),torch)
                    events=c_events(eventlog,len(pt))
                    full=int(ctc["mero_event"])+int(ctc["extra_events"])
                    same=(frame_equal and
                          ctc["collapsed_tokens"]==py_ctc["collapsed_tokens"]
                          and full==len(events)==record["torch_full_word_ctc_events"])
                    record["runtimes"][kind]={
                       "max_abs":float(errors.max()),
                       "mean_abs":float(errors.mean()),
                       "ctc_token_event_equal":bool(same),
                       "C_events":len(events),
                       "event_sample_clocks":events,
                       "C_frame_sha256":sha(rawlog)}
                    need(float(errors.max())<=MAX_ABS and
                         float(errors.mean())<=MAX_MEAN_ABS and same,
                         "STOP_B3_ACTUAL_PYTORCH_C11_ARM32_LOGIT_OR_EVENT_MISMATCH_"+kind)
                completed.append(record)
                shutil.rmtree(path)
        need(len(completed)==ORIGINAL_COUNT*len(CONDITIONS),
             "TWENTY_EIGHT_ACTUAL_PAIRED_B3_PCM_CASES")
        summary={"schema":SCHEMA,
           "status":"PASS_B3_PYTORCH_C11_ARM32_QEMU_ORIGINAL_PCM_PARITY",
           "run":identity,"B3_artifact_id":B3["id"],
           "B3_pt_sha256":B3["pt"],"pack_sha256":sha(packed),
           "pipeline_sha":PIPELINE_SHA,"input_source":source_receipt,
           "original_audio_files":ORIGINAL_COUNT,"input_conditions":list(CONDITIONS),
           "cases":len(completed),"compilers":list(COMPILERS),
           "max_abs_logit_tolerance":MAX_ABS,
           "mean_abs_logit_tolerance":MAX_MEAN_ABS,
           "observed_max_abs":max(v["max_abs"] for r in completed
                    for v in r["runtimes"].values()),
           "observed_max_mean_abs":max(v["mean_abs"] for r in completed
                    for v in r["runtimes"].values()),
           "full_ctc_event_sample_clock_parity":True,
           "elapsed_seconds":round(time.perf_counter()-started,3),
           "records":completed,"new_optimizer_updates":0,
           "new_tts_asr_calls":0,"reused_exposed_development_for_software_only":True,
           "new_independent_development_quality":"NOT_RUN",
           "physical_ssc305":"NOT_RUN",
           "natural_far_per_hour":"NOT_MEASURED",
           "real_human_3_5m_frr":"NOT_MEASURED",
           "word_end_p95_latency":"NOT_MEASURED",
           "model_selection_allowed":False,
           "product_promotion_allowed":False}
        store(output/"RESULT.json",summary)
        print(json.dumps({"status":summary["status"],
           "cases":len(completed),"observed_max_abs":summary["observed_max_abs"],
           "product_promotion_allowed":False},sort_keys=True))
    except BaseException as exc:
        store(output/"STOP.json",{"schema":SCHEMA,
             "status":"STOP_B3_NUMERIC_PARITY_NO_RETRY",
             "type":type(exc).__name__,"reason":str(exc)[:450],
             "completed_cases":completed,
             "new_cal_locked_reads":0,"product_promotion_allowed":False})
        raise
    finally:
        for f in output.glob("scratch-*"):
            if f.is_dir():shutil.rmtree(f)
        for f in output.glob("b3-*"):
            if f.is_file():f.unlink()
        if (output/"b3-immutable-temp.pack").is_file():
            (output/"b3-immutable-temp.pack").unlink()

def main():
    ap=argparse.ArgumentParser()
    for flag in ("--source-check","--guard","--audit","--execute"):
        ap.add_argument(flag,action="store_true")
    for arg in ("--source","--screen","--b3-zip","--pipeline-dir","--output"):
        ap.add_argument(arg,type=Path)
    args=ap.parse_args()
    need(sum((args.source_check,args.guard,args.audit,args.execute))==1,
         "EXACT_ONE_EXECUTION_LANE")
    if args.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if args.guard:print(json.dumps(guard(),sort_keys=True));return
    need(args.source and args.screen and args.b3_zip,
         "SOURCE_SCREEN_AND_TRAINED_MODEL_REQUIRED")
    if args.audit:
        proof=provenance(os.environ.get("GH_TOKEN"))
        rows,receipt=audit(args.source,args.screen,args.b3_zip)
        print(json.dumps({"status":"B3_SOURCE_AUDIO_AND_ACTUAL_PT_NPZ_ONNX_PINNED",
            "source":receipt,"model":proof,"rows":len(rows)},sort_keys=True))
        return
    need(args.pipeline_dir and args.output,
         "ACTUAL_FROZEN_C11_SOURCE_AND_RESULT_ROOT")
    execute(args.source,args.screen,args.b3_zip,args.pipeline_dir,args.output)

if __name__=="__main__":
    main()
