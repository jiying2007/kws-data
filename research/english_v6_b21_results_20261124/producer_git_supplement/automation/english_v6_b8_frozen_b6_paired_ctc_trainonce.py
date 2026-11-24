#!/usr/bin/env python3
"""B8: two same-job TRAIN-only continuations from the exact frozen B6 PT.

A new, separately preregistered experiment; NEVER recover, relabel, or bypass
the failed B7 exact historical replay. Both arms start from the same successful,
byte-authenticated B6 parent. Only ordinary-negative CTC weight differs:
control 0.60 versus challenger 0.85. Exact independent source TRAIN only.
No DEV512 model score, CAL/LOCKED access, new audio, or model promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location(
    "b8_reuse_exact_frozen_b7_loss", HERE /
    "english_v6_b7_ordinary_ctc_weight_trainonly_once.py")
b7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b7)
b6 = b7.b6
REPO = "jiying2007/kws-data"
BEFORE = "73b7003c58fd8ec1f01a25c8a30c1d46efc93bea"
SCHEMA = "english-v6-b8-frozen-b6-parent-paired-ordinary-ctc-v1"
B6_BLOB = "7731f0292fc0b30dd1d6dbb7e4b92275e255f711"
B7_BLOB = "f0997e1b3d16665a41b7dd09310a2e54bd13404f"
STEPS = 192
TRAIN_GROUPS = 35
PARAMS = 43876
B6 = b7.B6_MODEL
B7_STOP = {
    "run": 44721025043,
    "sha": "f9f2f56eece785da0d0c79d24dca12cdbcc29b29",
    "conclusion": "failure",
}
B7_DRIFT = {
    "run": 44738689144,
    "sha": BEFORE,
    "id": 14629527557,
    "name": "english-v6-b7-saved-control-drift-44738689144",
    "bytes": 2269,
    "zip_sha256": "f72785e3013b4d159dd3e134b29e5c261615b074ec603173112a652712fb702a",
}

def need(value, code):
    if not value:
        raise ValueError(code)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def file_blob(path):
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") +
                        b"\0" + raw).hexdigest()

def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True,
                  ensure_ascii=False, allow_nan=False)
        stream.write("\n")

def source_check():
    prior = b7.source_check()
    need(prior["status"] == "SOURCE_ONLY_NO_TRAIN_OR_MODEL_FORWARD"
         and b7.STEPS == b6.STEPS == STEPS
         and b7.TRAIN_GROUPS == b6.SOURCE_GROUPS == TRAIN_GROUPS
         and b7.PARAMETERS == b6.PARAMS == PARAMS
         and b7.SEED == b6.SEED == 20261120
         and b7.PREFIXES == b6.PREFIXES == (6400, 12800, 19200)
         and b7.GAINS == b6.GAINS == (1.0, 0.5)
         and b7.KL_WEIGHT == b6.WEIGHT == 0.03
         and b7.KL_CLIP == b6.KL_CLIP == 8.0
         and b7.OLD_ORDINARY_WEIGHT == 0.60
         and b7.NEW_ORDINARY_WEIGHT == 0.85
         and b6.LR == 0.00006
         and B6["id"] == 14615025178
         and B6["pt_sha256"] ==
             "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04"
         and file_blob(HERE/"english_v6_b6_gain_posterior_consistency_trainonly_once.py") == B6_BLOB
         and file_blob(HERE/"english_v6_b7_ordinary_ctc_weight_trainonly_once.py") == B7_BLOB
         and B7_STOP["conclusion"] == "failure"
         and B7_DRIFT["id"] == 14629527557,
         "EXACT_UNMODIFIED_B6_B7_SOURCE_AND_FROZEN_B6_PARENT")
    return {
        "schema": SCHEMA, "status": "SOURCE_ONLY_NO_OPTIMIZER_OR_MODEL_FORWARD",
        "active_keyword": "Hey Mero",
        "experiment": "B8_NEW_FROZEN_B6_PARENT_PAIRED_CONTINUATION",
        "not_a_B7_replay_or_recovery": True,
        "b7_strict_historical_replay_stays_failed": True,
        "exact_immutable_initial_model_zip": B6["id"],
        "exact_initial_pt_sha256": B6["pt_sha256"],
        "source_groups": TRAIN_GROUPS,
        "per_arm_real_optimizer_updates": STEPS,
        "maximum_new_optimizer_updates": 2*STEPS,
        "same_job_parent_tensor_bitwise_equal": True,
        "unchanged_b6_gain_KL_weight": 0.03,
        "control_ordinary_ctc_weight": 0.60,
        "challenger_ordinary_ctc_weight": 0.85,
        "one_changed_factor": "TRAIN_ONLY_ORDINARY_NEGATIVE_CTC_060_TO_085",
        "unchanged_seed_prefixes_gains_loss_architecture": True,
        "unmodified_B6_B7_training_implementations": True,
        "dev_gradients": 0, "dev_forwards": 0, "new_tts_asr_phone": 0,
        "cal_locked_opened": False, "model_promotion_allowed": False,
        "private_repo": False, "work_codex": False, "schedule": False,
    }

def provenance():
    # Includes verified immutable B3, B4, B6, and exposed DEV512 receipts.
    previous = b7.provenance()
    import urllib.request
    token = os.environ.get("GH_TOKEN", "")
    need(len(token)>10, "READ_ONLY_GITHUB_TOKEN_REQUIRED")
    def get(suffix):
        request = urllib.request.Request(
            "https://api.github.com/repos/"+REPO+"/actions/"+suffix,
            headers={"Authorization": "Bearer "+token,
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "kws-v6-b8-frozen-b6-paired-train"})
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read(1000001)
        need(len(raw) <= 1000000, "BOUND_REMOTE_METADATA")
        return json.loads(raw)
    for record, status in ((B7_STOP, "failure"), (B7_DRIFT, "success")):
        run = get("runs/"+str(record["run"]))
        need(run.get("id") == record["run"]
             and run.get("head_sha") == record["sha"]
             and run.get("head_branch") == "main"
             and run.get("event") == "push"
             and run.get("run_attempt") == 1
             and run.get("status") == "completed"
             and run.get("conclusion") == status,
             "B7_FAILED_AND_SAVED_DRIFT_ACTUAL_STATUS")
    art = get("artifacts/"+str(B7_DRIFT["id"]))
    need(art.get("id") == B7_DRIFT["id"]
         and art.get("name") == B7_DRIFT["name"]
         and art.get("size_in_bytes") == B7_DRIFT["bytes"]
         and art.get("digest") == "sha256:"+B7_DRIFT["zip_sha256"]
         and art.get("expired") is False
         and art.get("workflow_run", {}).get("id") == B7_DRIFT["run"]
         and art.get("workflow_run", {}).get("head_sha") == BEFORE,
         "ACTUAL_PROVEN_B7_STOP_DRIFT_ARTIFACT")
    return {"B6_and_train_source": previous,
            "B7_real_failed_control": B7_STOP["run"],
            "B7_saved_drift_not_promotable": B7_DRIFT["id"]}

def validate_main_event(env, event, parent):
    need(all(env.get(k)==v for k,v in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ATTEMPT": "1"}.items()),
        "FIRST_PUBLIC_MAIN_ONLY_NO_PR_OR_RERUN")
    head = env.get("GITHUB_SHA", "")
    need(re.fullmatch(r"[0-9a-f]{40}", head) is not None
         and re.fullmatch(r"[1-9][0-9]*",env.get("GITHUB_RUN_ID", "")) is not None
         and event.get("before") == BEFORE
         and event.get("after") == head
         and event.get("ref") == "refs/heads/main"
         and event.get("deleted") is False
         and event.get("forced") is False
         and event.get("repository",{}).get("full_name") == REPO
         and event.get("repository",{}).get("private") is False
         and parent == BEFORE,
         "EXACT_NEW_B8_FIRST_PROTECTED_MAIN_SUCCESSOR")
    return {"run": int(env["GITHUB_RUN_ID"]), "head": head,
            "parent": BEFORE}

def guard():
    source_check()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"],
                                      cwd=ROOT, text=True).strip()
    result = validate_main_event(os.environ, event, parent)
    result["provenance"] = provenance()
    return result

def load_original_B6(path, torch):
    path = Path(path)
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size == B6["bytes"]
         and sha(path.read_bytes()) == B6["zip_sha256"],
         "BYTE_AUTHENTIC_SUCCESSFUL_B6_ZIP_REQUIRED")
    with zipfile.ZipFile(path) as arc:
        need(arc.testzip() is None, "WHOLE_B6_ORIGINAL_ZIP_CRC")
        receipt = json.loads(arc.read("RESULT.json"))
        source = json.loads(arc.read("SOURCE.json"))
        need(receipt.get("schema") == b6.SCHEMA
             and receipt.get("status") ==
                 "RESEARCH_B6_TRAIN_ONLY_GAIN_POSTERIOR_KL_NO_PROMOTION"
             and receipt.get("actual_optimizer_updates") == 384
             and receipt.get("train_source_groups") == TRAIN_GROUPS
             and receipt.get("model_parameters") == PARAMS
             and receipt.get("dev_model_forwards") == 0
             and receipt.get("cal_locked_opened") is False
             and receipt.get("product_promotion_allowed") is False
             and receipt.get("B4_half_baseline_replay",{}).get("status") ==
                 "PASS_EXACT_B4_HALF_TENSOR_REPLAY"
             and source.get("dev_pcm_in_optimizer") is False
             and len(source.get("train_shas",[]))>0,
             "REAL_SUCCESSFUL_B6_TRAIN_SOURCE_AND_NO_PROMOTION")
        pt = arc.read(B6["member"])
        need(sha(pt) == B6["pt_sha256"], "EXACT_FROZEN_B6_PARENT_PT_HASH")
        name = B6["member"].replace(".pt", ".npz")
        features = arc.read(name)
    state = torch.load(io.BytesIO(pt), map_location="cpu", weights_only=True)
    need(sum(t.numel() for t in state.values()) == PARAMS
         and all(bool(torch.isfinite(t).all()) for t in state.values()),
         "REAL_FINITE_FROZEN_43876_PARAMETER_PARENT")
    return state, source["train_shas"], features

def environment_receipt(torch, np):
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "aten_cpu_capability_env": os.environ.get("ATEN_CPU_CAPABILITY"),
        "runner_image": os.environ.get("ImageVersion"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
    }

def execute(a):
    identity = guard()
    out = a.output
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(),
         "ONE_B8_REAL_TRAIN_OUTPUT_ONLY")
    out.mkdir()
    save(out/"STARTED.json", {
        "schema": SCHEMA, "identity": identity,
        "maximum_optimizer_updates": STEPS*2,
        "B7_replay_still_failed": True, "new_audio": 0,
        "dev_gradients": 0, "dev_inference": 0,
        "cal_locked_access": False, "product_promotion_allowed": False})
    try:
        rows, dev, sources, groups = b6.source_audit(a)
        train_hashes = sorted(r["source_pcm_sha256"] for r in rows)
        need(len(groups[0]) == len(groups[1]) == TRAIN_GROUPS
             and all(r["role"] == "train" for r in rows)
             and all(r["role"] == "development" for r in dev),
             "B8_SOURCE_TRAIN_ONLY_AND_DEV_METADATA_ONLY")
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        parent, prior_hashes, npz_bytes = load_original_B6(a.b6_model_zip, torch)
        need(prior_hashes == train_hashes,
             "EXACT_B6_B8_TRAIN_PCM_SHA_LIST_NO_LEAKAGE")
        _b3, mel, mean, scale = b6.b4.authenticate_b3(
            a.b3_model_zip,torch,np)
        with np.load(io.BytesIO(npz_bytes),allow_pickle=False) as frozen:
            need(np.array_equal(frozen["mel_filters"],mel)
                 and np.array_equal(frozen["train_mean"],mean)
                 and np.array_equal(frozen["train_scale"],scale),
                 "EXACT_B6_B3_FROZEN_LOGMEL_NORMALIZATION")
        need(set(parent)==set(_b3)
             and all(parent[k].shape==_b3[k].shape for k in parent),
             "EXACT_B3_B6_MODEL_GEOMETRY")
        del _b3
        save(out/"SOURCE.json", {
            "schema": SCHEMA, "train_pcm_sha256": train_hashes,
            "source_evidence": sources,
            "B6_parent_artifact": B6["id"],
            "B6_parent_pt_sha256": B6["pt_sha256"],
            "dev_rows_metadata_only": len(dev),
            "dev_pcm_in_gradients": False})
        save(out/"ENVIRONMENT.json", environment_receipt(torch,np))
        cache = b6.b4.cache_train(
            rows,a.old_archives,a.fresh_source,mel,mean,scale,torch,np)
        control_dir = out/"control_ordinary060"
        challenger_dir = out/"challenger_ordinary085"
        control_dir.mkdir()
        challenger_dir.mkdir()
        control = b6.fit("consistency",parent,rows,groups,cache,
                         mel,mean,scale,control_dir,torch,np,F)
        need(control["gain_consistency_enabled"] is True
             and control["optimizer_updates"]==STEPS,
             "GENUINE_B8_060_CONTROL")
        # Read back after the control before starting the challenger.
        # This is an independent B8 experiment, NOT an old B7 tensor replay.
        challenger = b7.fit_challenger(
            parent,rows,groups,cache,mel,mean,scale,
            challenger_dir,torch,np,F)
        need(challenger["optimizer_updates"]==STEPS
             and challenger["ordinary_ctc_weight"]==0.85
             and challenger["gain_kl_weight"]==0.03,
             "GENUINE_B8_085_SINGLE_FACTOR_CHALLENGER")
        cpath = control_dir/"models"/(b6.ARMS["consistency"]+".pt")
        qpath = challenger_dir/"models"/(b7.STEM+".pt")
        c = torch.load(cpath,map_location="cpu",weights_only=True)
        q = torch.load(qpath,map_location="cpu",weights_only=True)
        need(set(c)==set(q)==set(parent)
             and all(c[k].shape==q[k].shape==parent[k].shape for k in c),
             "SAME_43K_GEOMETRY_BOTH_TRAINED_ARMS")
        changed = sum(int(torch.count_nonzero(c[k]!=q[k]).item()) for k in c)
        need(changed>0, "MUST_HAVE_ACTUAL_DIFFERENT_TRAINED_CTC_ARMS")
        trace_c = [(r["step"],r["prefix_samples"]) for r in control["trace"]]
        trace_q = [(r["step"],r["prefix_samples"]) for r in challenger["trace"]]
        need(trace_c==trace_q and len(trace_c)>8,
             "EXACT_MATCHED_PER_STEP_SOURCE_POLICY")
        need(all(bool(torch.isfinite(t).all()) for t in list(c.values())+list(q.values())),
             "FINITE_43K_PAIRED_B8_WEIGHTS")
        result = {
            "schema": SCHEMA,
            "status": "RESEARCH_B8_TRAINED_PAIRED_FROZEN_B6_PARENT_NO_PROMOTION",
            "experiment_start": "ORIGINAL_BYTE_AUTHENTIC_B6_CONSISTENCY_PT",
            "B7_historical_exact_replay": "FAILED_UNCHANGED",
            "same_parent_pt_sha256": B6["pt_sha256"],
            "same_source_groups": TRAIN_GROUPS,
            "only_different_optimization_factor": "ORDINARY_CTC_060_VS_085",
            "unchanged_KL_weight": 0.03,
            "unchanged_training_seed": b6.SEED,
            "unchanged_prefix_samples": list(b6.PREFIXES),
            "unchanged_gain_pair": list(b6.GAINS),
            "same_job_two_arms": True,
            "control": control, "challenger": challenger,
            "actual_optimizer_updates": STEPS*2,
            "trained_tensor_values_different": changed,
            "new_tts_asr_phone": 0, "dev_model_forwards": 0,
            "dev_pcm_in_gradients": False, "cal_locked_opened": False,
            "independent_after_B8_DEV_qualified": "NOT_RUN",
            "new_C11_ARM_parity": "NOT_RUN",
            "physical_SSC305": "NOT_RUN",
            "human_3_5m_frr": "NOT_MEASURED",
            "natural_far_per_hour": "NOT_MEASURED",
            "model_selection_allowed": False,
            "product_promotion_allowed": False
        }
        save(out/"RESULT.json",result)
        print(json.dumps({
            "status": result["status"], "actual_optimizer_updates": STEPS*2,
            "control_pt_sha256": control["files"][b6.ARMS["consistency"]+".pt"]["sha256"],
            "challenger_pt_sha256": challenger["files"][b7.STEM+".pt"]["sha256"],
            "trained_values_different": changed,
            "B7_historical_exact_replay": "FAILED_UNCHANGED",
            "model_selection_allowed": False},sort_keys=True))
    except BaseException as error:
        save(out/"STOP.json",{
            "schema": SCHEMA, "status": "STOP_B8_NO_QUALIFIED_CHALLENGER",
            "error": type(error).__name__, "reason": str(error)[:450],
            "B7_historical_exact_replay": "FAILED_UNCHANGED",
            "model_selection_allowed": False,
            "product_promotion_allowed": False})
        raise

def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    for key in ("--source-check","--provenance","--guard","--audit","--execute"):
        mode.add_argument(key,action="store_true")
    for name in ("old-archives","old-audit","fresh-source","fresh-screen",
                 "source-gate","prospective-source","prospective-screen",
                 "original-model-zip","b3-model-zip","b6-model-zip","output"):
        p.add_argument("--"+name,type=Path)
    a=p.parse_args()
    if a.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    if a.provenance:
        print(json.dumps(provenance(),sort_keys=True));return
    if a.guard:
        print(json.dumps(guard(),sort_keys=True));return
    required=("old_archives","old_audit","fresh_source","fresh_screen",
              "source_gate","prospective_source","prospective_screen",
              "original_model_zip")
    need(all(getattr(a,k) is not None for k in required),
         "EIGHT_EXISTING_BYTE_AUTHENTICATED_SOURCE_ASSETS_REQUIRED")
    if a.audit:
        rows,dev,receipt,groups = b6.source_audit(a)
        print(json.dumps({"status":"SOURCE_AUDITED_NO_NEW_TRAIN",
              "train_rows":len(rows),"train_groups":len(groups[0]),
              "dev_rows_metadata_only":len(dev),
              "dev_gradients":0,"new_optimizer_updates":0},sort_keys=True))
        return
    need(a.b3_model_zip is not None and a.b6_model_zip is not None
         and a.output is not None, "FROZEN_B3_FEATURES_AND_B6_PARENT_REQUIRED")
    execute(a)

if __name__=="__main__":
    main()
