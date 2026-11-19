"""Real original-audio Hey Mero V5 PyTorch-vs-native C stream parity ONCE.

Only the frozen trained 43,876-param 256-step FSMN source; no new model,
optimizer, ASR, TTS, speaker selection, threshold or public product claim.
Uses 2 independently screened synthetic DEV positives and 8 ordinary clips
from never-used Piper speaker families 480..487. This is software engineering
parity, not fresh acoustic qualification, real-human FRR or continuous FAR/h.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    spec=importlib.util.spec_from_file_location("mero43_c_native_"+name,HERE/(name+".py"))
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj
proof=load("english_v5_mero_prospective_three_frozen_once")
SCHEMA="en-v5-mero43-pcm-fft40mel-fsmn-ctc-c-original-parity-v1"
REPO="jiying2007/kws-data"
BEFORE="57b8d6419e88e95f600a84eac4d0f2b25d1c4165"
PIPELINE_REPO="jiying2007/kws-pipeline"
PIPELINE_COMMIT="e3261e976965c52bcd0989591abab606d9cdc0e0"
MODEL={"id":14514228406,"run":44495594627,
       "size":501276,"sha":"91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702",
       "pt":"1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7",
       "npz":"02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696",
       "onnx":"10bf30da57291f78f56db196db1137309f2c3b91dfe67bf0aa45ebb8cec05b1c"}
SOURCE=proof.SOURCE
SCREEN=proof.SCREEN
MAX_ABS_LOGIT=0.005
MAX_MEAN_ABS_LOGIT=0.0005
CLIPS=10

def need(cond,why):
    if not cond:raise ValueError(why)
def sha(data):
    return hashlib.sha256(data).hexdigest()
def record(path,data):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(data,f,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")

def source_check():
    p=proof.source_check()
    need(p["keyword"]=="Hey Mero"
         and p["new_dev_positive"]==2
         and p["new_dev_ordinary"]==8
         and p["optimizer_steps"]==0
         and p["product_promotion_allowed"] is False
         and SOURCE["id"]==14524071469 and SCREEN["id"]==14525005967
         and proof.MODELS["fsmn43k256"]["id"]==MODEL["id"]
         and proof.MODELS["fsmn43k256"]["steps"]==256
         and proof.MODELS["fsmn43k256"]["params"]==43876
         and proof.MODELS["fsmn43k256"]["onnx"]==MODEL["onnx"]
         and proof.MODELS["fsmn43k256"]["sha"]==MODEL["sha"]
         and "pt" not in proof.MODELS["fsmn43k256"]
         and MODEL["npz"]=="02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696"
         and MAX_ABS_LOGIT==0.005 and MAX_MEAN_ABS_LOGIT==0.0005
         and CLIPS==10,
         "PREREGISTERED_FROZEN_43K_FULL_C_STREAM_PARITY_ONLY")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NATIVE_PARITY_NOT_RUN",
            "model_artifact":MODEL["id"],"source_artifact":SOURCE["id"],
            "machine_screen_artifact":SCREEN["id"],
            "source_repo_sha":PIPELINE_COMMIT,
            "original_audited_positive":2,"original_audited_ordinary":8,
            "maximum_C_forwards":CLIPS,
            "absolute_logit_tolerance":MAX_ABS_LOGIT,
            "mean_logit_tolerance":MAX_MEAN_ABS_LOGIT,
            "new_train_or_ASR_TTS_forwards":0,
            "existing_test_audio_reuse_for_software_parity_only":True,
            "calibration_or_locked_wavs_opened":False,
            "product_promotion_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),
        "ONLY_FIRST_PUBLIC_PROTECTED_MAIN_NUMERIC")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "GITHUB_MAIN_HEAD_RUN_REQUIRED")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "NO_REPLAY_OR_MOVING_MAIN_PARITY")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                                       cwd=ROOT,text=True).strip()
    need(parent==BEFORE,"EXACT_FIRST_MAIN_PARITY_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"model_sha":MODEL["sha"],
            "pipeline_source_commit":PIPELINE_COMMIT}

def audit(source,screen,original_zip):
    source_check()
    rows,receipt=proof.select(source,screen)
    need(len(rows)==CLIPS
         and sum(x["mono_class"] for x in rows)==2
         and len({x["source_group_id"] for x in rows})==8
         and receipt["no_locked_or_calibration"] is True,
         "ORIGINAL_TEN_DEV_MACHINE_LABELS_UNMODIFIED")
    sourcefile=Path(original_zip)
    need(sourcefile.is_file() and not sourcefile.is_symlink()
         and sourcefile.stat().st_size==MODEL["size"]
         and sha(sourcefile.read_bytes())==MODEL["sha"],
         "AUTHENTIC_ORIGINAL_43K_TRAINED_MODEL_ZIP")
    return rows,receipt

def pipeline_files(pipeline):
    root=Path(pipeline).resolve()
    sub=root/"research"/"english_v5_mero"
    need(sub.is_dir() and not sub.is_symlink(),
         "CHECKOUT_EXACT_PINNED_PIPELINE_SOURCE_PATH")
    required=("mero_ctc.h","mero_ctc.c","mero43_stream.h",
              "mero43_stream.c","mero43_host_cli.c","pack_mero43.py",
              "test_mero43_stream.c")
    need(all((sub/name).is_file() and not (sub/name).is_symlink()
             for name in required),
         "ALL_FROZEN_NATIVE_C_SOURCE_FILES_EXIST")
    return sub

def compile_c(source,compiler,destination):
    binary=Path(destination)
    cmd=[compiler,"-std=c11","-Wall","-Wextra","-Werror",
         "-pedantic","-O2","-ffp-contract=off",
         "-I",str(source),
         str(source/"mero43_stream.c"),str(source/"mero_ctc.c"),
         str(source/"mero43_host_cli.c"),
         "-lm","-o",str(binary)]
    result=subprocess.run(cmd,check=True,capture_output=True,
                          text=True,timeout=70)
    need(binary.is_file() and binary.stat().st_size>10000
         and not result.stderr.strip(),
         "NATIVE_COMPILER_FINITE_C11_CLI")
    return binary

def load_events(path,total_frames):
    events=[]
    data=Path(path).read_text(encoding="ascii").splitlines()
    need(len(data)==total_frames,"STREAM_EMITS_ONE_ROW_PER_FRAME")
    for index,line in enumerate(data):
        parts=line.split(",")
        need(len(parts)==3
             and all(re.fullmatch("[0-9]+",s) is not None for s in parts),
             "ACTUAL_C_EVENT_CLOCK_TOKENS")
        frame,sample,detected=map(int,parts)
        need(frame==index and sample==512+160*index
             and detected in (0,1),
             "C_STREAM_FRAME_CLOCK_AND_EVENT_ABI")
        events.append(detected)
    return events

def execute(source,screen,original_zip,model_dir,pipeline,output):
    import numpy as np
    import torch
    identity=guard()
    need(output.is_absolute() and output.parent.is_dir()
         and not output.exists(),"NEW_FROZEN_STAGEWISE_EVIDENCE_DIR")
    output.mkdir()
    record(output/"STARTED.json",{"schema":SCHEMA,"run":identity,
           "pipeline_source_sha":PIPELINE_COMMIT,
           "original_model_sha":MODEL["sha"],
           "fresh_model_or_optimizer":False,
           "repeat_old_acoustic_quality_selection":False})
    observed=[]
    try:
        rows,admission=audit(source,screen,original_zip)
        native=pipeline_files(pipeline)
        model,npz_bank,mean,scale,hashes=proof.model(
                                      torch,np,model_dir,"fsmn43k256")
        need(hashes["pt"]==MODEL["pt"]
             and hashes["npz"]==MODEL["npz"]
             and hashes["onnx"]==MODEL["onnx"],
             "FROZEN_ORIGINAL_PT_NPZ_ONNX_CONTENT_HASHES")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled=False
        model.eval()
        bundle=output/"mero43-unchanged.pack"
        result=subprocess.run([sys.executable,str(native/"pack_mero43.py"),
              "--source-zip",str(original_zip),"--output",str(bundle)],
              check=True,capture_output=True,text=True,timeout=90)
        pack=json.loads(result.stdout.strip().splitlines()[-1])
        need(pack["status"]=="PASS_BYTE_FROZEN_MODEL_REPACKAGE_NO_TRAIN"
             and pack["model_parameters"]==43876
             and pack["model_pack_bytes"]==216960
             and pack["source_zip_sha256"]==MODEL["sha"]
             and sha(bundle.read_bytes())==pack["model_pack_sha256"],
             "EXACT_MODEL_BINARY_PACK_SHA_FROM_ORIGINAL_NPZ")
        record(output/"ADMISSION.json",{
             "schema":SCHEMA,"run":identity,"source_audit":admission,
             "model_zip_sha256":MODEL["sha"],
             "model_pack_sha256":pack["model_pack_sha256"],
             "original_model_hashes":hashes})
        compiled={cc:compile_c(native,cc,output/("mero43-compiled-"+cc))
                  for cc in ("gcc","clang")}
        began=time.perf_counter()
        for row in rows:
            pcm=proof.pcm(row,source)
            features=proof.v5.features(pcm,row,torch,np,npz_bank)
            x=((features-mean)/scale).astype(np.float32)
            with torch.no_grad():
                pt=model(torch.from_numpy(x).unsqueeze(0))
                py=pt.squeeze(0).detach().cpu().numpy()
            need(py.shape[1]==4 and np.isfinite(py).all(),
                 "REAL_FROZEN_TORCH_FRAME_LOGITS_FINITE")
            py_greedy=proof.v5.ctc_greedy(pt,torch)
            item={"id":row["id"],"source_group_id":row["source_group_id"],
                  "original_pcm_sha256":row["source_pcm_sha256"],
                  "machine_positive":bool(row["mono_class"]),
                  "frame_count":int(py.shape[0]),"compilers":{}}
            for cc,executable in compiled.items():
                root=output/"temp"/cc;root.mkdir(parents=True,exist_ok=True)
                datafile=root/(row["id"]+".pcm")
                rawlog=root/(row["id"]+".f32")
                eventfile=root/(row["id"]+".csv")
                datafile.write_bytes(pcm)
                call=subprocess.run([str(executable),str(bundle),
                    str(datafile),str(rawlog),str(eventfile)],
                    check=True,capture_output=True,text=True,timeout=70)
                need("MERO43_C_PCM_STREAM_OK" in call.stderr,
                     "REAL_C_STREAM_CLI_SUCCESS_EVENT_RECEIPT")
                raw=rawlog.read_bytes()
                need(len(raw)==len(py)*4*4,
                     "C_FRAME_LENGTH_EQUALS_FROZEN_TORCH")
                c=np.frombuffer(raw,dtype="<f4").reshape(-1,4)
                need(c.shape==py.shape and np.isfinite(c).all(),
                     "REAL_NATIVE_F32_LOGITS_FRAME_SHAPE")
                absdiff=np.abs(py.astype(np.float64)-c.astype(np.float64))
                maximum=float(absdiff.max())
                meanerr=float(absdiff.mean())
                frames_equal=bool(np.array_equal(py.argmax(-1),c.argmax(-1)))
                c_events=load_events(eventfile,len(py))
                c_total=sum(c_events)
                true_total=int(py_greedy["mero_event"])+int(py_greedy["extra_events"])
                expected_ctc=proof.v5.ctc_greedy(
                                  torch.from_numpy(c.copy()).unsqueeze(0),torch)
                ctc_equal=(c_total==true_total
                     and expected_ctc["mero_event"]==py_greedy["mero_event"]
                     and expected_ctc["extra_events"]==py_greedy["extra_events"])
                entry={"max_abs":maximum,"mean_abs":meanerr,
                       "frame_argmax_equal":frames_equal,
                       "c_full_ctc_events":c_total,
                       "torch_full_ctc_events":true_total,
                       "whole_phrase_equal":ctc_equal,
                       "expected_frames":len(py)}
                item["compilers"][cc]=entry
                need(maximum<=MAX_ABS_LOGIT and meanerr<=MAX_MEAN_ABS_LOGIT
                     and frames_equal and ctc_equal,
                     "STOP_TRUE_C_VS_TORCH_LOGITS_OR_CTC_PARITY")
            observed.append(item)
        need(len(observed)==CLIPS and sum(x["machine_positive"] for x in observed)==2,
             "ALL_TEN_ORIGINALS_HAVE_COMPLETE_COMPARISON")
        record(output/"RESULT.json",{
            "schema":SCHEMA,"status":"PASS_ORIGINAL_43K_FSMN_C_VS_TORCH_FRAME_PARITY",
            "source_evidence":admission,"pipeline_source_sha":PIPELINE_COMMIT,
            "original_weight_zip_sha256":MODEL["sha"],
            "model_binary_pack_sha256":pack["model_pack_sha256"],
            "max_abs_logit_tolerance":MAX_ABS_LOGIT,
            "mean_abs_logit_tolerance":MAX_MEAN_ABS_LOGIT,
            "maximum_abs_observed":max(
                x["compilers"][cc]["max_abs"] for x in observed
                for cc in ("gcc","clang")),
            "original_clips":observed,
            "total_original_wavs":len(observed),
            "full_ctc_and_frame_argmax_parity_all":True,
            "host_elapsed_seconds":round(time.perf_counter()-began,3),
            "real_arm32_qemu":"NOT_RUN",
            "physical_ssc305":"NOT_RUN",
            "natural_far_per_hour":"NOT_MEASURED",
            "human_3_5m_frr":"NOT_MEASURED",
            "optimizer_updates":0,"new_tts_asr_forwards":0,
            "model_promotion_allowed":False})
        print(json.dumps({"status":"PASS_ORIGINAL_43K_FSMN_C_VS_TORCH_FRAME_PARITY",
            "original_wavs":len(observed),
            "max_abs_logit":max(
                x["compilers"][cc]["max_abs"] for x in observed
                for cc in ("gcc","clang")),
            "product_promotion_allowed":False},sort_keys=True))
    except BaseException as e:
        record(output/"STOP.json",{"schema":SCHEMA,
            "status":"STOP_REAL_NUMERIC_PARITY_NOT_QUALIFIED",
            "reason":str(e)[:480],"error_type":type(e).__name__,
            "completed_original_rows":len(observed),
            "observed":observed,
            "no_automatic_retry":True,"no_calibration_or_locked_opened":True})
        raise
    finally:
        # Do not archive full original WAVs, private keys or duplicate model;
        # their immutable original GitHub artifacts remain the authority.
        for p in (output/"temp",):
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
        if (output/"mero43-unchanged.pack").is_file():
            (output/"mero43-unchanged.pack").unlink()
        for cc in ("gcc","clang"):
            exe=output/("mero43-compiled-"+cc)
            if exe.exists():exe.unlink()


# Follow-up to the actual protected main host proof: compare the unchanged
# original full-stream C inference on Cortex-A32/ARM32 QEMU against x86 C.
# This is a software numerical bridge, NOT a real SSC305 board measurement.
ARM_BEFORE="e65986618d77dbd9053bafd27b95d2a8c85dce43"
HOST_PROOF={"run":44541193521,"head":ARM_BEFORE,"artifact":14538062948,
            "name":"english-v5-mero43-c-vs-torch-original-parity-44541193521",
            "digest":"sha256:050c0466a4abd588aa7778350db770e461b28e8f0ae641ffb3b6394d7442719b"}
ARM_SCHEMA="en-v5-mero43-cortex-a32-qemu-original-10pcm-parity-v1"

def arm_guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"ARM32_ONLY_FIRST_PUBLIC_MAIN")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",head) is not None
         and re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")) is not None,
         "ARM32_EXACT_MAIN_RUN_REQUIRED")
    if event is None:
        event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==ARM_BEFORE and event.get("after")==head
         and event.get("ref")=="refs/heads/main"
         and event.get("repository",{}).get("full_name")==REPO
         and event.get("repository",{}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False,
         "ARM32_EXACT_UNREPLAYED_SUCCESSOR")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                cwd=ROOT,text=True).strip()
    need(parent==ARM_BEFORE,"ARM32_EXACT_UNREPLAYED_PARENT")
    return {"main_sha":head,"run_id":env["GITHUB_RUN_ID"],
            "pipeline_sha":PIPELINE_COMMIT,"host_run":HOST_PROOF["run"]}

def prior_host_proof(token):
    import urllib.request
    need(isinstance(token,str) and len(token)>=10,
         "EXACT_PRIOR_HOST_SUCCESS_REQUIRES_ACTION_TOKEN")
    root="https://api.github.com/repos/"+REPO+"/actions/"
    def read_json(url):
        request=urllib.request.Request(url,headers={
            "Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "User-Agent":"mero43-immutable-arm32-parity"})
        with urllib.request.urlopen(request,timeout=35) as response:
            data=response.read(200001)
        need(len(data)<200001,"BOUNDED_HOST_PROOF_METADATA")
        return json.loads(data)
    run=read_json(root+"runs/"+str(HOST_PROOF["run"]))
    artifact=read_json(root+"artifacts/"+str(HOST_PROOF["artifact"]))
    need(run.get("id")==HOST_PROOF["run"]
         and run.get("event")=="push"
         and run.get("head_branch")=="main"
         and run.get("head_sha")==HOST_PROOF["head"]
         and run.get("run_attempt")==1 and run.get("status")=="completed"
         and run.get("conclusion")=="success",
         "HOST_C_VS_TORCH_FULL_ORIGINAL_PARITY_MUST_BE_GREEN")
    need(artifact.get("id")==HOST_PROOF["artifact"]
         and artifact.get("name")==HOST_PROOF["name"]
         and artifact.get("digest")==HOST_PROOF["digest"]
         and artifact.get("expired") is False
         and artifact.get("workflow_run",{}).get("id")==HOST_PROOF["run"]
         and artifact.get("workflow_run",{}).get("head_sha")==ARM_BEFORE,
         "PIN_PRIOR_HOST_PARITY_ORIGINAL_EVIDENCE")
    return {"run":HOST_PROOF["run"],"artifact":HOST_PROOF["artifact"],
            "digest":HOST_PROOF["digest"],"model_zip":MODEL["sha"]}

def compile_arm(native,out):
    binary=Path(out)
    cmd=["arm-linux-gnueabihf-gcc","-std=c11","-Wall","-Wextra",
         "-Werror","-pedantic","-O2","-fno-fast-math",
         "-ffp-contract=off","-fexcess-precision=standard",
         "-mcpu=cortex-a32","-mfpu=neon-vfpv4",
         "-mfloat-abi=hard","-static","-I",str(native),
         str(native/"mero43_stream.c"),str(native/"mero_ctc.c"),
         str(native/"mero43_host_cli.c"),"-lm","-o",str(binary)]
    completed=subprocess.run(cmd,check=True,text=True,
                             capture_output=True,timeout=110)
    need(binary.is_file() and binary.stat().st_size>10000
         and not completed.stderr.strip(),"REAL_CORTEX_A32_C11_BINARY")
    elf=subprocess.check_output(["readelf","-h",str(binary)],
                                text=True,timeout=15)
    need("ELF32" in elf and "ARM" in elf
         and "little endian" in elf and "hard-float ABI" in elf,
         "ELF32_ARM_HARDFLOAT_NOT_FAKE_HOST_BINARY")
    return binary

def execute_arm(source,screen,original_zip,pipeline,output):
    import numpy as np
    identity=arm_guard()
    need(output.is_absolute() and output.parent.is_dir()
         and not output.exists(),"FRESH_ARM32_EVIDENCE_ONLY")
    output.mkdir()
    record(output/"STARTED.json",{"schema":ARM_SCHEMA,
           "identity":identity,"original_model_sha":MODEL["sha"],
           "host_torch_proof":HOST_PROOF,"not_an_ssc305_device":True,
           "no_training_or_new_audio":True})
    completed=[]
    try:
        verified_host=prior_host_proof(os.environ.get("GH_TOKEN"))
        rows,receipt=audit(source,screen,original_zip)
        native=pipeline_files(pipeline)
        bundle=output/"temp"/"frozen43k.pack"
        bundle.parent.mkdir()
        proc=subprocess.run([sys.executable,str(native/"pack_mero43.py"),
               "--source-zip",str(original_zip),"--output",str(bundle)],
               check=True,capture_output=True,text=True,timeout=110)
        pack=json.loads(proc.stdout.strip().splitlines()[-1])
        need(pack["status"]=="PASS_BYTE_FROZEN_MODEL_REPACKAGE_NO_TRAIN"
             and pack["source_zip_sha256"]==MODEL["sha"]
             and pack["model_pack_bytes"]==216960
             and sha(bundle.read_bytes())==pack["model_pack_sha256"],
             "UNCHANGED_BYTE_AUTHENTICATED_CROSS_ARCH_MODEL")
        host=compile_c(native,"gcc",output/"temp"/"mero43-host")
        arm=compile_arm(native,output/"temp"/"mero43-arm32")
        for row in rows:
            raw=proof.pcm(row,source)
            need(len(raw)==2*row["samples"] and row["samples"]>=512,
                 "ORIGINAL_PCM16_LENGTH_AND_FRAME_GEOMETRY")
            expected_frames=1+(row["samples"]-512)//160
            point={"source_id":row["id"],
                   "source_group":row["source_group_id"],
                   "source_pcm_sha256":row["source_pcm_sha256"],
                   "machine_positive":bool(row["mono_class"]),
                   "samples":row["samples"],"frames":expected_frames}
            inputs=output/"temp"/(row["id"]+".pcm")
            inputs.write_bytes(raw)
            readings={}
            for target,argv in (("host",[str(host)]),
                                ("arm32",["qemu-arm",str(arm)])):
                log=output/"temp"/(row["id"]+"-"+target+".f32")
                events=output/"temp"/(row["id"]+"-"+target+".csv")
                result=subprocess.run(argv+[str(bundle),str(inputs),
                        str(log),str(events)],check=True,
                        capture_output=True,text=True,timeout=120)
                need("MERO43_C_PCM_STREAM_OK" in result.stderr,
                     "ACTUAL_CLI_FINISHED_"+target)
                data=np.fromfile(log,dtype="<f4")
                need(data.shape==(expected_frames*4,)
                     and np.isfinite(data).all(),
                     "ACTUAL_FULL_FRAME_FLOAT_LOGITS_"+target)
                frames=data.reshape(expected_frames,4)
                marks=load_events(events,expected_frames)
                readings[target]=(frames,marks)
            x,xt=readings["host"]
            y,yt=readings["arm32"]
            differences=np.abs(x.astype(np.float64)-y.astype(np.float64))
            maximum=float(differences.max())
            meanerr=float(differences.mean())
            equal_tokens=bool(np.array_equal(x.argmax(-1),y.argmax(-1)))
            equal_events=xt==yt
            point.update({"max_abs_logit":maximum,"mean_abs_logit":meanerr,
                          "all_frame_argmax_equal":equal_tokens,
                          "all_full_word_event_frames_equal":equal_events,
                          "host_events":sum(xt),"arm32_events":sum(yt)})
            completed.append(point)
            need(maximum<=MAX_ABS_LOGIT
                 and meanerr<=MAX_MEAN_ABS_LOGIT
                 and equal_tokens and equal_events,
                 "STOP_ARM32_VS_HOST_LOGITS_OR_EVENT_FRAME_MISMATCH")
        need(len(completed)==CLIPS
             and sum(p["machine_positive"] for p in completed)==2
             and sum(not p["machine_positive"] for p in completed)==8,
             "EVERY_ORIGINAL_DEVELOPMENT_CLIP_EXECUTED_ON_REAL_ARM_ELF")
        maximum=max(p["max_abs_logit"] for p in completed)
        record(output/"RESULT.json",{
            "schema":ARM_SCHEMA,
            "status":"PASS_CORTEX_A32_QEMU_VS_HOST_REAL_ORIGINAL_PCM_PARITY",
            "host_torch_receipt":verified_host,
            "identity":identity,"original_source_receipt":receipt,
            "original_frozen_model_zip_sha256":MODEL["sha"],
            "same_model_pack_sha256":pack["model_pack_sha256"],
            "arm32_elf_verified":True,
            "actual_arm32_qemu_executed":True,
            "all_original_pcm_count":len(completed),
            "maximum_abs_logit":maximum,
            "max_abs_bound":MAX_ABS_LOGIT,
            "mean_abs_bound":MAX_MEAN_ABS_LOGIT,
            "all_frame_tokens_equal":True,
            "all_event_frames_equal":True,
            "clips":completed,
            "physical_ssc305":"NOT_RUN",
            "device_cpu_ram":"NOT_MEASURED",
            "real_human_far_frr":"NOT_MEASURED",
            "product_promotion_allowed":False,
            "optimizer_updates":0,"new_audio_asr_tts":0})
        print(json.dumps({"status":"PASS_CORTEX_A32_QEMU_VS_HOST_REAL_ORIGINAL_PCM_PARITY",
              "original_pcm":len(completed),"maximum_abs_logit":maximum,
              "actual_arm_elf_executed":True,
              "physical_ssc305":False},sort_keys=True))
    except BaseException as error:
        record(output/"STOP.json",{
            "schema":ARM_SCHEMA,"status":"STOP_ARM32_NUMERIC_NOT_QUALIFIED",
            "reason":str(error)[:450],"error_type":type(error).__name__,
            "completed_original_pcm":len(completed),"clips":completed,
            "no_automatic_retry":True,"product_promotion_allowed":False})
        raise
    finally:
        temp=output/"temp"
        if temp.is_dir():
            import shutil
            shutil.rmtree(temp)


def main():
    p=argparse.ArgumentParser()
    for flag in ("--source-check","--guard","--audit","--execute","--arm-guard","--arm-execute"):
        p.add_argument(flag,action="store_true")
    for name in ("--source","--screen","--original-model-zip",
                 "--model-dir","--pipeline-dir","--output"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    need(sum((a.source_check,a.guard,a.audit,a.execute,a.arm_guard,a.arm_execute))==1,
         "ONE_REGISTERED_C_NUMERIC_STAGE")
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    if a.arm_guard:
        print(json.dumps(arm_guard(),sort_keys=True));return
    need(a.source and a.screen and a.original_model_zip,
         "ORIGINAL_MODEL_AND_SOURCE_ARTIFACTS_REQUIRED")
    if a.audit:
        _,receipt=audit(a.source,a.screen,a.original_model_zip)
        print(json.dumps(receipt,sort_keys=True));return
    if a.arm_execute:
        need(a.pipeline_dir and a.output,
             "FROZEN_ARM32_PIPELINE_AND_OUTPUT_REQUIRED")
        execute_arm(a.source,a.screen,a.original_model_zip,
                    a.pipeline_dir,a.output)
        return
    need(a.model_dir and a.pipeline_dir and a.output,
         "ORIGINAL_MODEL_DIR_AND_VERIFIED_C_CODE_AND_OUTPUT_REQUIRED")
    execute(a.source,a.screen,a.original_model_zip,
            a.model_dir,a.pipeline_dir,a.output)

if __name__=="__main__":main()
