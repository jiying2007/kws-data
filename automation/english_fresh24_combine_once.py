"""Byte-verified, model-free new fresh24 review of all 168 saved complete WAVs.

Requires 3 unprompted ASR models plus two original full-logit independent CTC
models on the SAME new PCM. Source artifact/voice roles and frozen prospective
contrast policy must exactly match pre-audio protected-main bytes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("fresh24_combine_"+name,HERE/(name+".py"))
    x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)
    return x
c=load("english_hifi_micro")
phones=load("english_hifi_phones_once")
worker=load("english_fresh24_five_native_once")
policy=load("english_fresh24_ctc_admission")
SCHEMA="english-fresh24-prospective-blind-source-screen-v1"
POLICY_ORIGINAL_BLOB_SHA1="0a688325d0d36e83ef6623ac678511df24a92374"
PLAN_ORIGINAL_BLOB_SHA1="dbe911fbcf8a697492ad29c7ca11858e707a72ff"

def need(ok,reason):
    if not ok:raise ValueError(reason)

def git_blob(path):
    raw=c.read(path)
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()

def source_check():
    check=worker.source_check()
    need(git_blob(HERE/"english_fresh24_ctc_admission.py")==
         POLICY_ORIGINAL_BLOB_SHA1 and
         git_blob(HERE/"english_fresh24_protocol.json")==
         PLAN_ORIGINAL_BLOB_SHA1,
         "UNMODIFIED_PREAUDIO_NATIVE_CTC_RULE_AND_SOURCE_ROLES")
    need(tuple(worker.PROVIDERS)==tuple(policy.TEXT_MODELS)+tuple(policy.PHONE_MODELS)
         and policy.FROZEN_SOURCE_ROLES=={"train":12,"development":4,
               "calibration":4,"locked-test":4} and
         policy.MIN_QUALIFIED=={"train":6,"development":2,
               "calibration":2,"locked-test":2},
         "EXACT_FIRST_168_PROSPECTIVE_FIVE_OBSERVER_POLICY")
    return {**check, "policy_git_blob_sha1":POLICY_ORIGINAL_BLOB_SHA1,
            "frozen_source_plan_git_blob_sha1":PLAN_ORIGINAL_BLOB_SHA1}

def packet(root,provider,rows,run):
    root=Path(root)/("fresh24-observer-"+provider)
    all_ids={row["id"] for row in rows}
    if not (root/"RESULT.json").is_file():
        return {sid:None for sid in all_ids}, {
            "provider":provider,"status":"MISSING_OBSERVER_ARTIFACT",
            "complete":0,"denominator":168}
    outcome=json.loads(c.read(root/"RESULT.json"))
    need(outcome.get("schema")==worker.SCHEMA and
         outcome.get("provider")==provider and
         outcome.get("run")==run and
         outcome.get("source_artifact_id")==worker.SOURCE["id"] and
         outcome.get("denominator")==168 and
         outcome.get("new_tts_calls")==0 and
         outcome.get("target_conditioned") is False and
         outcome.get("training_allowed") is False and
         outcome.get("human_gold") is False and
         type(outcome.get("rows")) is list,
         "SOURCE_BOUND_ORIGINAL_NATIVE_OBSERVER_PACKET")
    saved=outcome["rows"]
    need(len(saved)<=168 and
         len({x.get("id") for x in saved})==len(saved) and
         {x.get("id") for x in saved}<=all_ids,
         "ONLY_ORIGINAL_NEW_168_WAV_IDS")
    mapping={x["id"]:x for x in saved}
    source_by_id={r["id"]:r for r in rows}
    model_data=(json.loads(c.read(root/"MODEL.json"))
                if (root/"MODEL.json").is_file() else None)
    original_vocab=None
    if model_data is not None:
        expected=worker.identity(provider)
        if provider in ("xlsr","lv60"):
            name,commit=expected
            need(model_data.get("model")==name and
                 model_data.get("revision","").startswith(commit) and
                 model_data.get("weights_sha256")==
                    phones.WEIGHT_SHA[0 if provider=="xlsr" else 1] and
                 model_data.get("target_conditioned") is False,
                 "FROZEN_NATIVE_PHONE_WEIGHT_SOURCE")
            v=json.loads(c.read(root/"VOCAB.json"))
            need(v.get("model")==name and
                 v.get("revision")==model_data["revision"] and
                 v.get("pad_token_id")==0 and
                 type(v.get("vocab")) is dict and
                 len(v["vocab"])==392 and
                 len(set(v["vocab"].values()))==392,
                 "FROZEN_ESPEAK_NATIVE_392_PHONE_VOCAB")
            original_vocab=v["vocab"]
        else:
            need(model_data.get("observer")==expected and
                 model_data.get("target_conditioned") is False and
                 model_data.get("initialization_model_calls")==0,
                 "FROZEN_TRIPLE_BLIND_TEXT_ADAPTER")
    results={};complete=0
    for sid in all_ids:
        item=mapping.get(sid)
        if item is None or item.get("status")!="success":
            results[sid]=None
            continue
        original=source_by_id[sid]
        need(item.get("provider")==provider and
             item.get("source_wav_sha256")==original["wav_sha256"] and
             item.get("source_pcm_sha256")==original["pcm_sha256"] and
             item.get("source_samples")==original["samples"] and
             item.get("target_conditioned") is False and
             item.get("input_coverage_complete") is True and
             item.get("training_allowed") is False,
             "ACTUAL_COMPLETE_ORIGINAL_PCM_WAS_OBSERVED")
        recorded=json.loads(c.read(root/"rows"/sid/"CAPTURE.json"))
        need(recorded==item,"SAVED_MODEL_CAPTURE_MATCHES_RESULT")
        if provider in ("xlsr","lv60"):
            need(original_vocab is not None,
                 "NATIVE_PHONE_MODEL_PROVENANCE_REQUIRED")
            raw=c.read(root/"logits"/(sid+".f32"),40*1024*1024)
            n=item.get("logit_frames")
            need(type(n) is int and 0<n<=2000 and
                 item.get("vocab_size")==392 and
                 len(raw)==n*392*4 and
                 c.digest(raw)==item.get("logits_sha256"),
                 "SAVED_COMPLETE_392_NATIVE_CTC_FLOAT32_LOGITS")
            ids=[]
            for frame in struct.iter_unpack("<392f",raw):
                need(all(math.isfinite(x) for x in frame),
                     "FINITE_COMPLETE_ORIGINAL_NATIVE_CTC")
                ids.append(max(range(392),key=frame.__getitem__))
            recomputed=phones.ctc(ids,original_vocab,0)
            need(recomputed==item.get("raw_phone_tokens"),
                 "RECONSTRUCT_ORIGINAL_NATIVE_ARGMAX_CTC")
            results[sid]={"logits":raw,"vocab":original_vocab,
                "sha256":item["logits_sha256"],"pad_token_id":0}
        else:
            transcription=item.get("raw_transcript")
            need(type(transcription) is str and
                 0<len(transcription)<=4096 and
                 not any(ord(ch)<32 for ch in transcription),
                 "UNPROMPTED_ACTUAL_NATIVE_TRANSCRIPT")
            results[sid]=transcription
        complete+=1
    need(complete==outcome.get("complete") and
         complete<=outcome.get("attempted",0)<=168 and
         outcome.get("status")==
            ("COMPLETE" if complete==168 else "INCOMPLETE_NO_RETRY"),
         "NO_MISSING_INFERENCE_MASQUERADING_AS_ZERO")
    return results,{"provider":provider,"status":outcome["status"],
        "complete":complete,"denominator":168,
        "reported_attempted":outcome["attempted"]}

def evaluate(rows,texts_by_id,phones_by_id):
    need(len(rows)==168 and set(texts_by_id)==set(phones_by_id)==
         {r["id"] for r in rows},
         "COMPLETE_FROZEN_168_ROSTER")
    group_rows=defaultdict(list)
    decisions=[]
    for row in rows:
        sid=row["id"]
        observed=policy.decide({**row,"pcm_samples":row["samples"]},
                     texts_by_id[sid],phones_by_id[sid])
        # Source intent is never proof; an acoustic class must also agree
        # with the immutable requested kind to be supervised.
        if (observed["tier"]!="unknown" and
            observed["class"]!=row["requested_class"]):
            observed={**observed,"tier":"unknown","class":None,
                "reason":"INDEPENDENT_ACOUSTIC_LABEL_CONTRADICTS_SOURCE_INTENT",
                "training_allowed":False}
        record={
            "id":sid,"source_group_id":row["source_group_id"],
            "role":row["role"],"cell_id":row["cell_id"],
            "generation_render_text":row["generation_render_text"],
            "requested_class":row["requested_class"],
            "tier":observed["tier"],"observed_class":observed["class"],
            "reason":observed["reason"],
            "phone_canonical_contrast_log_margins":observed["margins"],
            "wav_sha256":row["wav_sha256"],"pcm_sha256":row["pcm_sha256"],
            "samples":row["samples"],"positive_duration_pass":row["positive_duration_pass"],
            "training_allowed":False,"human_gold":False,
            "independent_far_frr":False}
        decisions.append(record)
        group_rows[row["source_group_id"]].append(record)
    need(len(group_rows)==24 and all(len(r)==7 for r in group_rows.values()),
         "TWENTY_FOUR_SEVEN_CELL_GROUPS")
    role_map={r["source_group_id"]:r["role"] for r in worker.source_worker.source_plan()["voice_groups"]}
    qualified=set()
    for group,group_items in group_rows.items():
        need({x["role"] for x in group_items}=={role_map[group]} and
             {x["cell_id"] for x in group_items}==
              {x["cell_id"] for x in worker.source_worker.source_plan()["cells"]},
             "SPLIT_AND_SEVEN_REQUESTS_UNCHANGED")
        accepted={item["observed_class"] for item in group_items
                  if item["tier"] in ("silver","weak")}
        if accepted=={0,1,2}:
            qualified.add(group)
    counts=Counter(role_map[g] for g in qualified)
    complete={role:counts[role] for role in policy.FROZEN_SOURCE_ROLES}
    passed=all(complete[role]>=policy.MIN_QUALIFIED[role]
               for role in policy.FROZEN_SOURCE_ROLES)
    prospective_training_rows=sum(x["role"]=="train" and
         x["source_group_id"] in qualified and x["tier"] in ("silver","weak")
         for x in decisions)
    return {"schema":SCHEMA,
        "status":"PASS_CANONICAL_6_2_2_2_SOURCE_COVERAGE" if passed
             else "STOP_CANONICAL_6_2_2_2_SOURCE_COVERAGE",
        "denominator":168,
        "minimum_qualified_groups":policy.MIN_QUALIFIED,
        "qualified_groups_by_role":complete,
        "total_qualified_synthetic_source_groups":len(qualified),
        "source_groups_are_verified_distinct_synthetic_presets_not_humans":True,
        "prospective_training_rows_available_if_full_admission_passes":
            prospective_training_rows if passed else 0,
        "tier_counts":dict(Counter(x["tier"] for x in decisions)),
        "reason_counts":dict(Counter(x["reason"] for x in decisions)),
        "observed_label_counts":dict(Counter(str(x["observed_class"])
                    for x in decisions if x["tier"] in ("silver","weak"))),
        "machine_source_qualification_passed":passed,
        "training_snapshot_allowed":passed,
        "new_optimizer_updates":0,"new_model_weights":False,
        "locked_kws_evaluation_consumed":False,
        "human_gold":False,
        "independent_human_far_frr":False,"ssc305_measured":False,
        "rows":decisions}

def combine(source,neutral,observer_root,out):
    frozen=source_check()
    need(frozen["status"]=="SOURCE_PINNED",
         "NO_INFERENCE_RESULT_UNTIL_ACTUAL_SOURCE_PIN")
    run=worker.guard()
    inputs=worker.source_rows(source,neutral)
    root=Path(out)
    need(root.is_absolute() and not root.exists(),"NEW_SINGLE_PROSPECTIVE_SCREEN")
    root.mkdir(parents=True)
    c.record(root/"STARTED.json",{"schema":SCHEMA,"run":run,
        "source_run":worker.SOURCE_RUN,
        "source_artifact_id":worker.SOURCE["id"],
        "source_artifact_digest":worker.SOURCE["digest"],
        "policy_original_git_blob_sha1":POLICY_ORIGINAL_BLOB_SHA1,
        "actual_model_forwards_this_stage":0,
        "training_allowed":False,"human_gold":False})
    texts={r["id"]:{} for r in inputs}
    phones_data={r["id"]:{} for r in inputs}
    reports=[]
    for provider in worker.PROVIDERS:
        observations,fact=packet(observer_root,provider,inputs,run)
        reports.append(fact)
        for sid,value in observations.items():
            (phones_data if provider in ("xlsr","lv60") else texts)[sid][provider]=value
    complete=all(fact["complete"]==168 and fact["status"]=="COMPLETE"
                 for fact in reports)
    research=evaluate(inputs,texts,phones_data)
    allowed=bool(complete and research["machine_source_qualification_passed"])
    c.record(root/"SCREEN.json",{
        "schema":SCHEMA,
        "status":research["status"] if complete else "STOP_INCOMPLETE_FIVE_NATIVE_OBSERVERS",
        "source_run":worker.SOURCE_RUN,"source_commit":worker.SOURCE_COMMIT,
        "source_artifact_id":worker.SOURCE["id"],
        "source_artifact_digest":worker.SOURCE["digest"],
        "review_run":run,
        "frozen_policy_git_blob_sha1":POLICY_ORIGINAL_BLOB_SHA1,
        "frozen_source_plan_git_blob_sha1":PLAN_ORIGINAL_BLOB_SHA1,
        "observers":reports,
        "all_five_native_observers_complete":complete,
        "machine_review":research,
        "training_snapshot_allowed":allowed,
        "prior_exposed_melo25_and_clone60_unmodified":True,
        "new_tts_calls":0,"new_native_observer_calls_this_combine":0,
        "new_optimizer_updates":0,"new_model_weights":False,
        "locked_kws_evaluation_consumed":False,
        "human_gold":False,"independent_human_far_frr":False,
        "ssc305_measured":False})
    print(json.dumps({"status":research["status"] if complete else
         "STOP_INCOMPLETE_FIVE_NATIVE_OBSERVERS",
        "source_rows":168,"observer_completion":reports,
        "qualified_groups_by_role":research["qualified_groups_by_role"],
        "tier_counts":research["tier_counts"],
        "training_snapshot_allowed":allowed,
        "optimizer_updates":0},sort_keys=True))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-check",action="store_true")
    parser.add_argument("--combine",action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--reference")
    parser.add_argument("--observers")
    parser.add_argument("--output")
    opts=parser.parse_args()
    need(opts.source_check!=opts.combine,"ONE_CLOSED_RESEARCH_STAGE")
    if opts.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    need(all((opts.source,opts.reference,opts.observers,opts.output)),
         "ORIGINAL_SOURCE_AND_ALL_FIVE_OBSERVERS_REQUIRED")
    combine(opts.source,opts.reference,opts.observers,opts.output)

if __name__=="__main__":
    main()
