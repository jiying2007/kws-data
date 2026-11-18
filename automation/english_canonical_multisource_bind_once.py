"""Bind THREE genuinely screened source campaigns to an immutable train-only snapshot.

The original source AUDIT decisions remain immutable. This independent binder
never infers ASR, changes UNKNOWN, allocates speaker roles, or promotes KWS.
All six transport ZIPs must be byte-matched to exact public Actions artifacts.
Model training MUST NOT consume locked-test audio before weights and calibration.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import wave
import zipfile

ROOT=Path(__file__).resolve().parent.parent
CONFIG_PATH=ROOT/"automation/english_canonical_multisource_train_config.json"
REPO="jiying2007/kws-data"
ROLES=("train","development","calibration","locked-test")
CATEGORIES=(0,1,2)
SCHEMA="english-canonical-multisource-reviewed-pcm-v1"
LIMIT=16*1024*1024
def need(x,reason):
    if not x:raise ValueError(reason)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canon(v):
    return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
def once(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:f.write(canon(data)+b"\n")
def load(path):
    return json.loads(Path(path).read_bytes())
def config():
    cfg=load(CONFIG_PATH)
    need(cfg.get("schema")=="english-canonical-multisource-frozen-head-train-v1" and
         cfg.get("project")==REPO and
         cfg.get("before")=="4992666d4356d76301ad8ccbd71fad12db3d7caa" and
         cfg.get("brand_ipa")=={"Hey Mero":"/heɪ ˈmɛroʊ/","Hey Vosen":"/heɪ ˈvoʊsən/"} and
         cfg.get("full_qualified_role_minimum")==dict(zip(ROLES,(6,2,2,2))) and
         cfg.get("exact_expected_qualified")==dict(zip(ROLES,(8,4,3,3))) and
         cfg.get("accepted_machine_tiers")=={"silver":1.0,"weak":0.5} and
         cfg.get("no_legacy_meero_weights") is True and
         cfg.get("no_legacy_meero_positive_audio") is True and
         cfg.get("research_training_enabled") is True and
         cfg.get("automatic_model_promotion_allowed") is False and
         cfg.get("human_far_frr") is False and
         cfg.get("ssc305_measurements") is False and
         len(cfg.get("corpus",[]))==3 and
         [x["key"] for x in cfg["corpus"]]==["fresh24","fresh18","piper24"],
         "FROZEN_BRAND_SOURCE_SELECTION_NO_LEGACY_REPLAY")
    unique=set()
    counts=Counter()
    for entry in cfg["corpus"]:
        for kind in ("source","screen"):
            part=entry[kind]
            need(type(part["id"]) is int and part["id"]>0 and
                 type(part["run"]) is int and part["run"]>0 and
                 re.fullmatch("[0-9a-f]{40}",part["sha"]) is not None and
                 type(part["bytes"]) is int and 0<part["bytes"]<=LIMIT and
                 re.fullmatch("[0-9a-f]{64}",part["sha256"]) is not None and
                 type(part["name"]) is str and part["name"].endswith(str(part["run"])),
                 "EXACT_SIX_GITHUB_ARTIFACT_PINS")
            need(part["id"] not in unique,"DUPLICATE_ARTIFACT_ID")
            unique.add(part["id"])
        need(set(entry["qualified_by_role"])==set(ROLES) and
             all(type(v) is int and v>=0 for v in entry["qualified_by_role"].values()),
             "FROZEN_SYNTHETIC_SOURCE_ROLE_COUNTS")
        counts.update(entry["qualified_by_role"])
    need(dict(counts)==cfg["exact_expected_qualified"] and
         cfg["corpus"][-1]["screen"]["status"]=="PASS_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE",
         "ALL_THREE_ORIGINAL_MACHINE_EVIDENCE_GATE")
    assets=cfg["frozen_backbone"]["assets"]
    need(set(assets)=={"melspectrogram.onnx","embedding_model.onnx"} and
         assets["melspectrogram.onnx"]["sha256"]==
         "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f" and
         assets["embedding_model.onnx"]["sha256"]==
         "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
         "PINNED_OPENWAKEWORD_RESEARCH_FEATURE_GRAPHS")
    recipe=cfg["head"];frontend=cfg["preprocessing"]
    need(recipe["architecture"]=="frozen_96d_embedding_causal_16_history_relu_1536x32x3" and
         recipe["parameters"]==49283 and recipe["max_updates"]==96 and
         recipe["seed"]==20261118 and
         recipe["checkpoint_selection"]=="last_fixed_96th_update_no_dev_sweep" and
         recipe["calibration"]=="one_fixed_grid_using_calibration_only_after_weight_freeze" and
         recipe["locked_eval"]=="once_after_fixed_calibration_and_weight_sha_no_model_selection" and
         frontend["feature_width"]==96 and
         frontend["history_vectors"]==16 and
         frontend["embedding_sample_hop"]==1280 and
         frontend["minimum_history_samples"]==35200 and
         frontend["positive_max_original_samples"]==28672,
         "FROZEN_CAUSAL_HEAD_GEOMETRY_AND_ONE_PASS_SELECTION")
    need(cfg["new_tts_calls"]==cfg["new_asr_calls"]==cfg["new_phone_calls"]==0,
         "NO_DATA_GENERATION_OR_REVIEW_REPLAY")
    return cfg

def guard(env=None,event=None,parent=None):
    cfg=config();env=os.environ if env is None else env
    for name,val in {"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":REPO,
           "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
           "GITHUB_RUN_ATTEMPT":"1"}.items():
        need(env.get(name)==val,"ONE_PUBLIC_MAIN_ATTEMPT_ONLY")
    head=env.get("GITHUB_SHA","")
    need(re.fullmatch("[a-f0-9]{40}",head) and
         re.fullmatch("[1-9][0-9]*",env.get("GITHUB_RUN_ID","")),
         "ACTUAL_FIRST_MAIN_RUN_AND_HEAD")
    if event is None:
        event=load(env["GITHUB_EVENT_PATH"])
    need(event.get("before")==cfg["before"] and event.get("after")==head and
         event.get("ref")=="refs/heads/main" and
         event.get("repository",{}).get("full_name")==REPO and
         event.get("repository",{}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "FROZEN_SOURCE_FIRST_MAIN_SUCCESSOR")
    if parent is None:
        parent=subprocess.check_output(["git","rev-parse","HEAD^1"],
              text=True,cwd=ROOT).strip()
    need(parent==cfg["before"],"EXACT_PROTECTED_MAIN_PARENT")
    return {"run_id":env["GITHUB_RUN_ID"],"head":head,
            "before":cfg["before"],"config_sha256":sha(canon(cfg))}

def archive(path,pin):
    path=Path(path)
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size==pin["bytes"]
         and sha(path.read_bytes())==pin["sha256"],
         "EXACT_IMMUTABLE_SAVED_ARTIFACT_SHA:"+pin["name"])
    z=zipfile.ZipFile(path)
    names=z.infolist()
    need(names and len({i.filename for i in names})==len(names)
         and len(names)<=1000 and sum(i.file_size for i in names)<=150*1024**2,
         "COMPLETE_ARCHIVE_MEMBER_IDENTITY")
    for i in names:
        p=PurePosixPath(i.filename)
        need(not i.is_dir() and not p.is_absolute() and
             p.as_posix()==i.filename and
             len(p.parts)<=5 and ".." not in p.parts and
             i.file_size<=8*1024**2 and
             ((i.external_attr>>16)&0o170000)!=0o120000,
             "SOURCE_ARCHIVE_PATH_OR_SYMLINK")
    return z

def read_wav(raw,row):
    need(sha(raw)==row["wav_sha256"],"ORIGINAL_WAV_SHA256")
    with wave.open(io.BytesIO(raw),"rb") as w:
        need((w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())
             ==(16000,1,2,"NONE") and
             w.getnframes()==row["samples"] and
             3200<=row["samples"]<=240000,
             "UNCROPPED_NATIVE_PCM16_16KHZ")
        pcm=w.readframes(w.getnframes())
        need(w.readframes(1)==b"","NO_TRAILING_AUDIO_PCM")
    need(sha(pcm)==row["pcm_sha256"] and len(pcm)==2*row["samples"],
         "ORIGINAL_FULL_PCM16_SHA256")
    return pcm

def admission(paths,*,verify_audio=True):
    """Returns validated eligible research labels without changing source verdict."""
    cfg=config()
    all_source_groups={}
    seen_pcm=set();seen_ids=set()
    candidate=[];reports=[]
    for spec in cfg["corpus"]:
        key=spec["key"]
        source=archive(Path(paths)/(key+"-source.zip"),spec["source"])
        screen=archive(Path(paths)/(key+"-screen.zip"),spec["screen"])
        try:
            need("MANIFEST.json" in source.namelist()
                 and "SCREEN.json" in screen.namelist(),
                 "ORIGINAL_SOURCE_AND_REVIEW_MANIFEST")
            s=json.loads(source.read("MANIFEST.json"))
            c=json.loads(screen.read("SCREEN.json"))
            m=c.get("machine_review",{})
            expected=spec["source"];outcome=spec["screen"]
            need(s.get("schema")==expected["source_schema"] and
                 s.get("run",{}).get("run_id")==str(expected["run"]) and
                 s.get("run",{}).get("commit")==expected["sha"] and
                 s.get("human_gold") is False and
                 s.get("training_allowed") is False and
                 len(s.get("rows",[]))==expected["denominator"] and
                 c.get("status")==outcome["status"] and
                 c.get("source_artifact_id")==expected["id"] and
                 c.get("all_five_native_observers_complete") is True and
                 c.get("training_snapshot_allowed") is False and
                 m.get("denominator")==expected["denominator"] and
                 m.get("training_snapshot_allowed") is False and
                 len(m.get("rows",[]))==expected["denominator"],
                 "EXACT_HISTORICAL_SOURCE_SCREEN_ROSTER_AND_NO_AUTOPROMOTION")
            original={r["id"]:r for r in s["rows"]}
            reviews={r["id"]:r for r in m["rows"]}
            need(len(original)==len(reviews)==expected["denominator"]
                 and set(original)==set(reviews),
                 "SAME_ORIGINAL_ID_DENOMINATOR")
            observed_group=defaultdict(set)
            group_role={}
            for id,row in original.items():
                n=reviews[id]
                need(id not in seen_ids and row.get("human_gold") is False
                     and row.get("training_allowed") is False and
                     row.get("observed_class") is None and
                     n.get("training_allowed") is False and
                     n.get("human_gold") is False and
                     row["role"]==n["role"] and
                     row["source_group_id"]==n["source_group_id"] and
                     row["pcm_sha256"]==n["pcm_sha256"] and
                     row["wav_sha256"]==n["wav_sha256"] and
                     row["samples"]==n["samples"] and
                     row["cell_id"]==n["cell_id"] and
                     row["requested_class"]==n["requested_class"] and
                     row["role"] in ROLES,
                     "FULL_ORIGINAL_SOURCE_TO_INDEPENDENT_REVIEW_IDENTITY")
                seen_ids.add(id)
                pcmsha=row["pcm_sha256"]
                need(re.fullmatch("[0-9a-f]{64}",pcmsha) and
                     pcmsha not in seen_pcm,
                     "DUPLICATE_PCM_ACROSS_ANY_SOURCE_OR_ROLE")
                seen_pcm.add(pcmsha)
                if verify_audio:
                    read_wav(source.read("wav/"+id+".wav"),row)
                group=row["source_group_id"];role=row["role"]
                need(group_role.get(group,role)==role,
                     "ONE_IMMUTABLE_ROLE_PER_SOURCE")
                group_role[group]=role
                need(group not in all_source_groups,
                     "INTER_CORPUS_SOURCE_GROUP_COLLISION")
                t=n["tier"]
                need(t in ("silver","weak","unknown"),"EXACT_MACHINE_SILVER_WEAK_UNKNOWN")
                if t=="unknown":
                    need(n["observed_class"] is None,
                         "UNKNOWN_NOT_A_BACKGROUND_NEGATIVE")
                else:
                    label=n["observed_class"]
                    need(type(label) is int and label in CATEGORIES and
                         label==row["requested_class"] and
                         n.get("positive_duration_pass")==row["positive_duration_pass"],
                         "MACHINE_CLASS_MUST_MATCH_ORIGINAL_REQUEST_AND_DURATION")
                    if label in (1,2):
                        need(row["positive_duration_pass"] is True and
                             row["samples"]<=cfg["preprocessing"]["positive_max_original_samples"],
                             "UNCROPPED_POSITIVE_TOO_LONG")
                    observed_group[group].add(label)
                if row["requested_class"] is None:
                    need(t=="unknown",
                         "LEGACY_OR_VOICED_CONTROL_NEVER_SUPERVISED")
                candidate.append({
                   "id":id,"corpus":key,"source_group_id":group,"role":role,
                   "cell_id":row["cell_id"],"weak_class":n["observed_class"],
                   "tier":t,"machine_reason":n["reason"],
                   "source_pcm_sha256":pcmsha,
                   "source_wav_sha256":row["wav_sha256"],
                   "samples":row["samples"],"requested_class":row["requested_class"],
                   "human_gold":False,"training_allowed_in_original":False})
            qualified={k for k,classes in observed_group.items() if classes=={0,1,2}}
            observed=Counter(group_role[g] for g in qualified)
            byrole={r:observed[r] for r in ROLES}
            need(byrole==spec["qualified_by_role"],
                 "RECOMPUTED_REAL_SOURCE_QUALIFICATION_COUNTS")
            for group,role in group_role.items():
                all_source_groups[group]={"role":role,"corpus":key,
                                         "qualified":group in qualified}
            reports.append({"corpus":key,"source_run":expected["run"],
                            "source_artifact_id":expected["id"],
                            "review_run":outcome["run"],
                            "review_artifact_id":outcome["id"],
                            "total_original_rows":len(original),
                            "new_qualified_by_role":byrole,
                            "all_native_168_126_168_screen_bytes_preserved":True})
        finally:
            source.close();screen.close()
    expected=cfg["exact_expected_qualified"]
    approved=Counter(v["role"] for v in all_source_groups.values() if v["qualified"])
    counts={role:approved[role] for role in ROLES}
    need(counts==expected and
         all(counts[r]>=cfg["full_qualified_role_minimum"][r] for r in ROLES),
         "ACTUAL_MACHINE_SCREEN_6_2_2_2_NOT_PASSED")
    selected=[x for x in candidate if
              all_source_groups[x["source_group_id"]]["qualified"] and
              x["tier"] in ("silver","weak")]
    need(all(x["weak_class"] in CATEGORIES and
             x["requested_class"]==x["weak_class"]
             for x in selected) and
         {(r,x["weak_class"]) for r in ("train","development") for x in selected if x["role"]==r}
         =={(r,k) for r in ("train","development") for k in CATEGORIES},
         "TRAIN_DEV_MUST_HAVE_ALL_THREE_CANONICAL_CLASSES")
    counts_by_role_class=Counter((x["role"],x["weak_class"]) for x in selected)
    need(len(candidate)==462 and len(selected)>=50 and
         sum(x["role"]=="train" for x in selected)>=24 and
         sum(x["role"]=="development" for x in selected)>=8,
         "ACTUAL_REVIEWED_SMALL_DATA_COVERAGE")
    output={"schema":SCHEMA,"status":"MACHINE_SILVER_WEAK_RESEARCH_TRAIN_INPUT_BOUND",
         "source_corpus_reports":reports,"source_artifacts":[x["source"] for x in cfg["corpus"]],
         "review_artifacts":[x["screen"] for x in cfg["corpus"]],
         "source_group_counts":counts,"minimum":cfg["full_qualified_role_minimum"],
         "qualified_source_groups":[{"source_group_id":k,**v}
             for k,v in sorted(all_source_groups.items()) if v["qualified"]],
         "all_462_original_pcm_hashed":verify_audio,
         "original_candidate_denominator":len(candidate),
         "selected_machine_reviewed_rows":len(selected),
         "selected_counts_by_role_class":[{"role":k[0],"class":k[1],"count":n}
            for k,n in sorted(counts_by_role_class.items())],
         "rows":selected,
         "calibration_and_locked_audio_not_training":True,
         "human_gold":False,"no_old_meero_warmstart":True,
         "old_screen_original_training_snapshot_allowed_unchanged":False,
         "fresh_separate_research_training_input_allowed":True,
         "independent_real_far_frr":"NOT_RUN","ssc305":"NOT_RUN"}
    return output

def prepare(paths,output):
    identity=guard()
    result=admission(paths)
    target=Path(output)
    need(target.is_absolute() and not target.exists() and target.parent.is_dir(),
         "CREATE_ONLY_NEW_TRAINING_OUTPUT")
    target.mkdir()
    once(target/"STARTED.json",{"schema":SCHEMA,"run":identity,
       "original_source_artifact_count":6,"new_asr_calls":0,
       "new_tts_calls":0,"optimizer_updates":0,"training_allowed":False})
    once(target/"ADMISSION.json",result)
    return {"status":result["status"],"selected":result["selected_machine_reviewed_rows"],
            "by_role":result["source_group_counts"],"training_allowed":True,
            "human_gold":False,"new_model_weights":False}

def main():
    ap=argparse.ArgumentParser()
    modes=("--source-check","--guard","--audit-inputs","--bind")
    for key in modes:ap.add_argument(key,action="store_true")
    ap.add_argument("--sources")
    ap.add_argument("--output")
    a=ap.parse_args()
    need(sum(bool(getattr(a,x[2:].replace("-","_"))) for x in modes)==1,
         "ONE_CLOSED_SOURCE_STAGE")
    if a.source_check:
        c=config()
        print(json.dumps({"status":"FROZEN_SIX_ARTIFACT_TRAIN_DESIGN",
             "source_artifacts":6,"expected_roles":c["exact_expected_qualified"],
             "human_gold":False,"new_optimizer_calls":0},sort_keys=True));return
    if a.guard:print(json.dumps(guard(),sort_keys=True));return
    if a.audit_inputs:
        need(a.sources,"SIX_PINNED_RESEARCH_SOURCES_REQUIRED")
        data=admission(a.sources)
        print(json.dumps({"status":data["status"],
             "selected":data["selected_machine_reviewed_rows"],
             "roles":data["source_group_counts"],
             "source_rows":data["original_candidate_denominator"],
             "new_model_calls":0,"human_gold":False},sort_keys=True))
        return
    need(a.sources and a.output,"ALL_SIX_PINNED_SOURCES_AND_NEW_OUTPUT_REQUIRED")
    print(json.dumps(prepare(a.sources,a.output),sort_keys=True))

if __name__=="__main__":
    main()
