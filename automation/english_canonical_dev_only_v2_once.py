"""English canonical KWS dev-only causal head v2: real numeric candidate training.

A new fresh-seed 49,283-parameter research head using ONLY the already
byte-authenticated 63 machine Silver/Weak research rows, 8 train + 4 dev
qualified source groups. Cal/locked *PCM is not opened*, scores are not
loaded, and the previous opened 22-bag locked test is not revisited.
Select from exactly [96,192,384] seeded updates by fixed development
event metrics at a fixed 0.8 working threshold, before any fresh heldout test.
Weights are saved and C-compatible but are not release-qualified.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,importlib.util,json,os,urllib.request
from pathlib import Path
import re,subprocess,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def load(name):
    path=HERE/(name+".py")
    spec=importlib.util.spec_from_file_location("v2_"+name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value
binding=load("english_canonical_multisource_bind_once")
base=load("english_canonical_multisource_fit_once")
REPO="jiying2007/kws-data"
BEFORE="edf154f3b7d2e977fe78b567d023a289e7054ba3"
PRIOR_RUN=44425553398
PRIOR_SHA=BEFORE
PRIOR_NUMERIC_JOB=135229803469
PRIOR_FAILED_STEP="Bind exact public main first parent and preregistered dev-only no-locked training"
SCHEMA="english-canonical-development-only-head-v2-20261119"
UPDATES=(96,192,384)
SEED=20261119
WORKING_THRESHOLD=0.8
SOURCE_QUALIFIED={"train":8,"development":4,"calibration":3,"locked-test":3}
MODEL_PARAMETERS=49283
MAX_INPUTS=6
def need(v,reason):
    if not v:raise ValueError(reason)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def source_check():
    cfg=binding.config()
    need(cfg["exact_expected_qualified"]==SOURCE_QUALIFIED and
         cfg["head"]["architecture"]==
            "frozen_96d_embedding_causal_16_history_relu_1536x32x3" and
         cfg["head"]["parameters"]==MODEL_PARAMETERS and
         cfg["head"]["max_updates"]==96 and
         cfg["head"]["learning_rate"]==0.001 and
         cfg["head"]["weight_decay"]==0.0001 and
         cfg["head"]["clip_grad_norm"]==1 and
         cfg["preprocessing"]["feature_width"]==96 and
         cfg["preprocessing"]["history_vectors"]==16 and
         cfg["no_legacy_meero_weights"] is True and
         cfg["no_legacy_meero_positive_audio"] is True and
         len(cfg["corpus"])==3 and
         all(x["source"]["id"]>0 and x["screen"]["id"]>0
             for x in cfg["corpus"]) and
         UPDATES==(96,192,384) and SEED==20261119 and
         WORKING_THRESHOLD==0.8 and
         base.SHAPES["w1"]==(32,1536) and
         base.MAGIC==b"EHKWS01\x00",
         "EXACT_PREVIOUS_6_2_2_2_AND_NEW_DEV_ONLY_RECIPE")
    return {"status":"FROZEN_DEV_ONLY_NO_NUMERICS",
       "source_groups":SOURCE_QUALIFIED,"max_real_updates":UPDATES[-1],
       "dev_selection_steps":list(UPDATES),
       "dev_working_threshold":WORKING_THRESHOLD,
       "new_audio_calls":0,"old_locked_kws_test_reuse":False,
       "calibration_audio_opened":False,"locked_audio_opened":False,
       "new_model_weights":False,"training_allowed":False}

def guard(env=None,event=None,parent=None):
    source_check()
    env=os.environ if env is None else env
    need(all(env.get(k)==value for k,value in {
        "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
        "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
        "GITHUB_RUN_ATTEMPT":"1"}.items()),"EXACT_PUBLIC_MAIN_FIRST_ATTEMPT")
    sha_=env.get("GITHUB_SHA","")
    need(re.fullmatch("[0-9a-f]{40}",sha_) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "ACTUAL_PROTECTED_MAIN_SHA_AND_RUN")
    if event is None:event=json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before")==BEFORE and event.get("after")==sha_ and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "EXACT_FIRST_MAIN_SUCCESSOR_WITHOUT_REPLAY")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
                text=True,cwd=ROOT).strip()
    need(parent==BEFORE,"ORIGINAL_FROZEN_MAIN_FIRST_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"commit":sha_,
            "before":BEFORE,"source_model":"fresh_uninitialized"}

def verify_prior_zero(token):
    """Bounded GitHub REST attestation: original failure before NUMERIC steps.

    This is not an authorize-general-retry mechanism. It requires the
    exact original first public-main run, job, and step conclusions.
    """
    guard()
    need(type(token) is str and len(token)>=10,
         "READONLY_GITHUB_ACTIONS_TOKEN")
    def github(path):
        url=f"https://api.github.com/repos/{REPO}/actions/{path}"
        req=urllib.request.Request(url,headers={
            "Authorization":"Bearer "+token,
            "Accept":"application/vnd.github+json",
            "User-Agent":"kws-exact-zero-forward-dev-only-v2"})
        with urllib.request.urlopen(req,timeout=40) as response:
            raw=response.read(2_000_001)
        need(len(raw)<=2_000_000,"SMALL_GITHUB_ORIGINAL_RECEIPT")
        return json.loads(raw)
    run=github(f"runs/{PRIOR_RUN}")
    need(run.get("id")==PRIOR_RUN and run.get("event")=="push" and
         run.get("status")=="completed" and
         run.get("conclusion")=="failure" and
         run.get("run_attempt")==1 and
         run.get("head_sha")==PRIOR_SHA and
         run.get("head_branch")=="main" and
         run.get("repository",{}).get("full_name")==REPO and
         run.get("repository",{}).get("private") is False,
         "EXACT_PRIOR_PUBLIC_MAIN_ZERO_FORWARD_FAILURE")
    receipt=github(f"runs/{PRIOR_RUN}/jobs?per_page=100")
    jobs=receipt.get("jobs",[])
    need(len(jobs)==3 and set(j["name"] for j in jobs)==
         {"source-contract","exact-six-artifact-audit","dev-only-numeric-fit"},
         "ORIGINAL_EXACT_THREE_STAGE_WORKFLOW")
    previous=next(j for j in jobs if j["name"]=="dev-only-numeric-fit")
    steps=previous.get("steps",[])
    need(previous.get("id")==PRIOR_NUMERIC_JOB and
         previous.get("conclusion")=="failure" and
         sum(x["conclusion"]=="failure" for x in steps)==1 and
         any(x["name"]==PRIOR_FAILED_STEP and x["conclusion"]=="failure"
             for x in steps) and
         all(x["conclusion"]=="skipped" for x in steps
             if x["name"] in (
               "Read six original archives under a fresh numeric job identity",
               "Verify original 63 Silver/Weak research rows and disjoint 8/4/3/3 roles without numerical inference",
               "Install exactly one original CPU frozen-ONNX plus fresh PyTorch training stack",
               "Fetch only the two original 0.5.1 frozen feature models, validate bytes before training",
               "Run real fresh-seed 384-update train/dev-only candidate selection, no calibration or locked PCM"
             )) and
         all(j["conclusion"]=="success" for j in jobs
             if j["name"]!="dev-only-numeric-fit"),
         "NO_PRIOR_MODEL_FORWARDS_OR_UPDATE_AFTER_FAILED_ARGPARSE")
    artifacts=github(f"runs/{PRIOR_RUN}/artifacts?per_page=100")
    need(artifacts.get("total_count")==0 and
         artifacts.get("artifacts")==[],
         "NO_PARTIAL_NEW_V2_MODEL_WEIGHTS_REPLAY")
    return {"status":"EXACT_FIRST_DEV_V2_RUN_ZERO_MODEL_FORWARDS",
            "original_run":PRIOR_RUN,"source_group_gate":"PASS",
            "original_numeric_updates":0,"old_locked_reused":False}

def choose(candidates):
    """Exactly pre-registered development score; does not see cal/locked."""
    need(len(candidates)==3 and [x["update"] for x in candidates]==
         list(UPDATES),"EXACT_THREE_NEW_DEV_ONLY_CANDIDATES")
    for x in candidates:
        metrics=x["development"]
        need(metrics["source_groups"]==4 and
             all(metrics["classes"][str(k)]["cases"]>0 for k in (0,1,2)),
             "ALL_THREE_DEV_CLASSES_WITH_DISTINCT_SOURCES")
        need(x["threshold"]==WORKING_THRESHOLD,"PREDECLARED_DEV_WORKING_POINT")
        v=x["selection_key"]
        need(type(v) is list and len(v)==4 and
             all(type(y) in (int,float) for y in v),
             "FOUR_DIMENSION_FIXED_DEV_SELECTION")
    return min(candidates,key=lambda x:tuple(x["selection_key"]))

def _model_arrays(model,mean,scale):
    import numpy as np
    a={"mean96":mean.copy(),"scale96":scale.copy(),
       "w1":model[0].weight.detach().cpu().numpy().astype(np.float32).copy(),
       "b1":model[0].bias.detach().cpu().numpy().astype(np.float32).copy(),
       "w2":model[2].weight.detach().cpu().numpy().astype(np.float32).copy(),
       "b2":model[2].bias.detach().cpu().numpy().astype(np.float32).copy()}
    need(set(a)==set(base.SHAPES) and
         all(a[k].shape==base.SHAPES[k] and
             a[k].dtype==np.float32 and np.isfinite(a[k]).all()
             for k in a),"EXACT_FINITE_CAUSAL_C_HEAD_ARRAYS")
    return a

def training_loop(selected,sources,frontend,output):
    """Open only train and dev original WAVs; never cal/locked PCM."""
    import numpy as np
    import torch
    cfg=binding.config()
    spec=cfg["head"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    progress={"actual_frontend_forward_calls":0,
              "optimizer_updates_started":0,"optimizer_updates_completed":0,
              "calibration_feature_calls_started":False,
              "locked_test_feature_calls_started":False}
    need(len(selected)==63 and
         Counter(r["role"] for r in selected)==
           Counter({"train":28,"development":13,"calibration":11,
                    "locked-test":11}),
         "ORIGINAL_BOUND_EXACT_63_WEAK_RESEARCH_ROWS")
    train_bags=[];dev_bags=[];excluded=[]
    for row in sorted(selected,key=lambda x:x["id"]):
        if row["role"] not in ("train","development"):
            continue
        original=base.load_source_pcm(row,sources)
        conditions=("clean","snr10","snr5") if row["role"]=="train" else ("clean","snr5")
        for condition in conditions:
            bag,error=base.build_bag(row,original,condition,frontend,progress)
            if error:excluded.append(error)
            else:(train_bags if row["role"]=="train" else dev_bags).append(bag)
    groups=defaultdict(set)
    for bag in train_bags+dev_bags:
        groups[bag["role"],bag["source_group_id"]].add(bag["weak_class"])
    good={key for key,classes in groups.items() if classes=={0,1,2}}
    counts=Counter(k[0] for k in good)
    need(counts["train"]>=6 and counts["development"]>=2,
         "AT_LEAST_6_TRAIN_2_DEV_POST_MEL_ACOUSTIC_GROUPS")
    train_bags=[b for b in train_bags if
                (b["role"],b["source_group_id"]) in good]
    dev_bags=[b for b in dev_bags if
              (b["role"],b["source_group_id"]) in good]
    need(len(train_bags)>=45 and len(dev_bags)>=12,
         "ACTUAL_SUPPORTED_NEW_CANONICAL_FEATURE_BAGS")
    # Normalization is taken exclusively from source-qualified train bags.
    n=sum(len(b["windows"])*16 for b in train_bags)
    total=sum((b["windows"].astype(np.float64).sum((0,1))
               for b in train_bags))
    squares=sum(((b["windows"].astype(np.float64)**2).sum((0,1))
                 for b in train_bags))
    mean=(total/n).astype(np.float32)
    scale=np.maximum(np.sqrt(np.maximum(squares/n-(total/n)**2,0)),
                     1e-5).astype(np.float32)
    need(np.isfinite(mean).all() and np.isfinite(scale).all() and
         (scale>0).all(),"TRAIN_ONLY_FINITE_NORMALIZATION")
    model=torch.nn.Sequential(torch.nn.Linear(1536,32),
            torch.nn.ReLU(),torch.nn.Linear(32,3))
    need(sum(x.numel() for x in model.parameters())==MODEL_PARAMETERS,
         "EXACT_49283_PARAMETER_MODEL_NO_WARMSTART")
    optimizer=torch.optim.AdamW(model.parameters(),
        lr=spec["learning_rate"],weight_decay=spec["weight_decay"])
    grouped=defaultdict(lambda:defaultdict(list))
    for b in train_bags:grouped[b["weak_class"]][b["source_group_id"]].append(b)
    need(set(grouped)=={0,1,2} and
         all(len(grouped[k])>=6 for k in (0,1,2)),
         "ALL_TRAIN_CLASSES_SIX_DISJOINT_SOURCES")
    data={b["id"]:torch.from_numpy(
        ((b["windows"]-mean)/scale).reshape(len(b["windows"]),1536).astype(np.float32))
          for b in train_bags}
    losses=[];draws=[];candidates=[];snapshots=[]
    for index in range(UPDATES[-1]):
        optimizer.zero_grad(set_to_none=True)
        terms=[];weights=[];audit=[]
        progress["optimizer_updates_started"]+=1
        for cl in (0,1,2):
            identities=sorted(grouped[cl])
            source=identities[index%len(identities)]
            choices=sorted(grouped[cl][source],key=lambda b:b["id"])
            bag=choices[(index//len(identities))%len(choices)]
            prediction=torch.log_softmax(model(data[bag["id"]]),dim=-1)
            # Identical train-only MIL bag semantics to the first 96-step study.
            loss=(-prediction[:,0].mean() if cl==0 else
                  torch.log(torch.tensor(len(prediction),dtype=torch.float32))-
                  torch.logsumexp(prediction[:,cl],dim=0))
            terms.append(loss*bag["tier_weight"])
            weights.append(bag["tier_weight"])
            audit.append({"bag":bag["id"],"source":source,"class":cl,
                          "tier_weight":bag["tier_weight"]})
        objective=torch.stack(terms).sum()/sum(weights)
        need(bool(torch.isfinite(objective)),"FINITE_REAL_TRAINING_LOSS")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
           spec["clip_grad_norm"],error_if_nonfinite=True)
        optimizer.step()
        progress["optimizer_updates_completed"]+=1
        losses.append(float(objective.detach()))
        draws.append(audit)
        steps=index+1
        if steps in UPDATES:
            arrays=_model_arrays(model,mean,scale)
            # The only permitted selection signal is new dev audio at the
            # original preregistered 0.8 working threshold.
            preds=[base.score(arrays,(mean,scale),b,WORKING_THRESHOLD)
                   for b in dev_bags]
            summary=base.aggregate(preds)
            misses=[summary["classes"][str(cl)]["miss_rate"] for cl in (1,2)]
            key=[max(misses),summary["total_false_or_extra_events"],
                 sum(misses)/2,steps]
            candidates.append({"update":steps,
                "development":summary,"selection_key":key,
                "threshold":WORKING_THRESHOLD})
            snapshots.append(arrays)
    choice=choose(candidates)
    index=list(UPDATES).index(choice["update"])
    arrays=snapshots[index]
    modeldir=Path(output)/"models"
    modeldir.mkdir(exist_ok=False)
    np.savez(modeldir/"canonical-dev-selected.npz",**arrays)
    with np.load(modeldir/"canonical-dev-selected.npz",allow_pickle=False) as z:
        restored={name:z[name].copy() for name in z.files}
    need(set(restored)==set(base.SHAPES) and
         all(np.array_equal(restored[k],arrays[k]) for k in arrays),
         "BYTE_ROUNDTRIP_NEW_SELECTED_MODEL")
    raw=base.MAGIC+b"".join(restored[k].astype("<f4").tobytes()
                            for k in base.ORDER)
    need(len(raw)==197908,"ORIGINAL_C_HEAD_BIN_SIZE_NO_MODEL_INFLATION")
    binary=modeldir/"canonical-dev-selected.bin"
    binary.write_bytes(raw)
    npz_pin=sha((modeldir/"canonical-dev-selected.npz").read_bytes())
    bin_pin=sha(raw)
    saved=base.saved
    saved(Path(output)/"TRAINING.json",{
       "schema":SCHEMA,"status":"EXECUTED_REAL_DEV_ONLY_OPTIMIZATION",
       "optimizer":"AdamW","fresh_seed":SEED,
       "updates_completed":progress["optimizer_updates_completed"],
       "training_draws":draws,"training_losses":losses,
       "train_bags":len(train_bags),"development_bags":len(dev_bags),
       "rejected_mel_geometry":excluded,
       "actual_frozen_backbone_forward_calls":
          progress["actual_frontend_forward_calls"],
       "all_optimizer_source_roles_train":True,
       "normalization_train_role_only":True,
       "legacy_meero_warmstart":False,
       "original_calibration_pcm_opened":False,
       "original_locked_pcm_opened":False,
       "new_calibration_pcm_opened":False,
       "new_locked_pcm_opened":False,"human_gold":False})
    saved(Path(output)/"DEV_SELECTION.json",{
       "schema":SCHEMA,"status":"SELECTED_FROM_DEV_ONLY",
       "predeclared_candidate_updates":list(UPDATES),
       "predeclared_dev_threshold":WORKING_THRESHOLD,
       "predeclared_selection_key":
          ["worst_word_miss_rate","false_and_extra_events",
           "macro_word_miss_rate","training_update"],
       "candidates":candidates,"selected":choice,
       "source_groups":{"train":counts["train"],
                        "development":counts["development"]},
       "original_calibration_or_locked_scored":False,
       "new_locked_test_used":False,
       "selected_head_bin_sha256":bin_pin,
       "human_gold":False})
    saved(Path(output)/"WEIGHTS_FROZEN.json",{
       "schema":SCHEMA,"status":"NEW_DEV_ONLY_CANONICAL_WEIGHTS_FROZEN",
       "selected_update":choice["update"],"npz_sha256":npz_pin,
       "bin_sha256":bin_pin,"parameter_count":MODEL_PARAMETERS,
       "before_any_new_calibration_or_locked_audio":True,
       "old_meero_weights_reused":False,
       "training_source_groups":counts["train"],
       "dev_source_groups":counts["development"]})
    final={"schema":SCHEMA,"status":"EXECUTED_DEV_ONLY_RESEARCH_HEAD_V2",
       "model":{"parameters":MODEL_PARAMETERS,
                "bin_sha256":bin_pin,"npz_sha256":npz_pin},
       "training_updates":UPDATES[-1],
       "selected_update":choice["update"],
       "dev_selection_key":choice["selection_key"],
       "development":choice["development"],
       "old_96_step_locked_evaluation_is_excluded_from_selection":True,
       "calibration_status":"NOT_RUN_REQUIRE_FRESH_UNEXPOSED_SOURCE",
       "locked_test_status":"NOT_RUN_REQUIRE_FRESH_UNEXPOSED_SOURCE",
       "independent_real_far_frr":"NOT_RUN",
       "physical_ssc305":"NOT_RUN",
       "research_synthetic_only":True,
       "production_model_promotion_allowed":False}
    saved(Path(output)/"RESULT.json",final)
    return final

def execute(sources,assets,output):
    identity=guard()
    out=Path(output)
    need(out.is_absolute() and out.parent.is_dir() and
         not out.exists() and not out.is_symlink(),
         "NEW_NUMERIC_RESEARCH_OUTPUT_ONCE")
    out.mkdir()
    base.saved(out/"STARTED.json",{
       "schema":SCHEMA,"run":identity,
       "seed":SEED,"max_updates":UPDATES[-1],
       "calibration_locked_pcm_access":"DENIED",
       "new_tts_calls":0,"new_asr_calls":0,"new_phone_calls":0})
    folders={}
    try:
        original=binding.admission(sources)
        need(original["status"]==
             "MACHINE_SILVER_WEAK_RESEARCH_TRAIN_INPUT_BOUND" and
             original["source_group_counts"]==SOURCE_QUALIFIED and
             len(original["rows"])==63,
             "SIX_ORIGINAL_SIGNED_CORPORA_REVIEW_AGREEMENT")
        base.saved(out/"ADMISSION.json",original)
        base.verify_assets(assets)
        for spec in binding.config()["corpus"]:
            name=spec["key"]
            folders[name]=binding.archive(
                Path(sources)/(name+"-source.zip"),spec["source"])
        base.saved(out/"NUMERIC_STARTED.json",{
           "schema":SCHEMA,"run":identity,
           "training_input_sha256":sha(binding.canon(original)),
           "original_historical_locked_source_already_opened":True,
           "new_training_uses_only_train_and_dev_pcm":True,
           "numeric_forward_calls":0,
           "new_locked_evaluation":"NOT_RUN"})
        frontend=base.make_frontend(assets)
        result=training_loop(original["rows"],folders,frontend,out)
        print(json.dumps({k:v for k,v in result.items() if k!="development"},
                         sort_keys=True))
        return result
    except BaseException as err:
        base.saved(out/"FAILED.json",{
            "schema":SCHEMA,"run":identity,
            "error":type(err).__name__+":"+str(err)[:500],
            "automatic_retry":False,
            "old_locked_kws_evaluation_replay":False,
            "no_premature_head_promotion":True})
        raise
    finally:
        for a in folders.values():a.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-check",action="store_true")
    ap.add_argument("--guard",action="store_true")
    ap.add_argument("--verify-prior-zero",action="store_true")
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--sources")
    ap.add_argument("--assets")
    ap.add_argument("--output")
    p=ap.parse_args()
    need(sum(bool(v) for v in (p.source_check,p.guard,
                                 p.verify_prior_zero,p.execute))==1,
         "ONE_NUMERIC_RESEARCH_STAGE")
    if p.source_check:
        print(json.dumps(source_check(),sort_keys=True))
        return
    if p.guard:
        print(json.dumps(guard(),sort_keys=True))
        return
    if p.verify_prior_zero:
        print(json.dumps(verify_prior_zero(os.environ.get("GH_TOKEN")),
                         sort_keys=True))
        return
    need(p.sources and p.assets and p.output,
         "EXACT_SIX_SIGNED_INPUTS_AND_TWO_FROZEN_ONNX")
    execute(p.sources,p.assets,p.output)

if __name__=="__main__":
    main()
