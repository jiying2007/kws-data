"""Byte-verified, model-free new fresh24 review of all 126 saved complete WAVs.

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
worker=load("english_fresh18_five_native_once")
policy=load("english_fresh24_ctc_admission")
SCHEMA="english-fresh18-joined-prospective-blind-source-screen-v1"
POLICY_ORIGINAL_BLOB_SHA1="0a688325d0d36e83ef6623ac678511df24a92374"
PLAN_ORIGINAL_BLOB_SHA1="c2ee2f7bf2fbdeb8821920d6cb5e6890d0df1f76"

def need(ok,reason):
    if not ok:raise ValueError(reason)

def git_blob(path):
    raw=c.read(path)
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()

PRIOR={"run_id":44306842000,"artifact_id":14429385634,
       "name":"english-fresh24-prospective-screen-44306842000",
       "bytes":19501,
       "zip_sha256":"5a9b226b8fe475331d9f03d3ca4f91d5bb94c5270113f6d0cf3e8b4c36fe8316",
       "original_source_run":44289263966}
FROZEN_PRIOR_GROUPS={"train":4,"development":1,
                     "calibration":3,"locked-test":0}

def source_check():
    check=worker.source_check()
    need(git_blob(HERE/"english_fresh24_ctc_admission.py")==
         POLICY_ORIGINAL_BLOB_SHA1 and
         git_blob(HERE/"english_fresh18_direct_protocol.json")==
         PLAN_ORIGINAL_BLOB_SHA1,
         "FROZEN_PREAUDIO_POLICY_AND_NEW18_ROLE_ROSTER")
    need(tuple(worker.PROVIDERS)==tuple(policy.TEXT_MODELS)+tuple(policy.PHONE_MODELS) and
         worker.source_worker.plan()["groups_by_role"]==
              {"train":10,"development":4,"locked-test":4} and
         policy.MIN_QUALIFIED=={"train":6,"development":2,
                               "calibration":2,"locked-test":2} and
         worker.SOURCE_COUNT==126 and worker.SOURCE["id"]==14429578824,
         "NATIVE_FIVE_OBSERVERS_AND_EXACT_126_NEW_GROUPS")
    return {**check,"original_policy_git_blob_sha1":POLICY_ORIGINAL_BLOB_SHA1,
            "new18_plan_git_blob_sha1":PLAN_ORIGINAL_BLOB_SHA1,
            "prior_run":PRIOR["run_id"],"training_allowed":False}

def readback_prior(root):
    """Use only immutable prior source-qualification counts; never inspect locked rows."""
    root=Path(root)
    saved=json.loads(c.read(root/"SCREEN.json"))
    review=saved.get("machine_review",{})
    need(saved.get("schema")=="english-fresh24-prospective-blind-source-screen-v1" and
         saved.get("status")=="STOP_CANONICAL_6_2_2_2_SOURCE_COVERAGE" and
         saved.get("source_run")==PRIOR["original_source_run"] and
         saved.get("source_artifact_id")==14422271973 and
         saved.get("all_five_native_observers_complete") is True and
         saved.get("training_snapshot_allowed") is False and
         saved.get("new_optimizer_updates")==0 and
         review.get("qualified_groups_by_role")==FROZEN_PRIOR_GROUPS and
         review.get("total_qualified_synthetic_source_groups")==8 and
         review.get("denominator")==126 and
         review.get("training_snapshot_allowed") is False,
         "PREVIOUS_ORIGINAL_IMMUTABLE_FIVE_OBSERVER_STOP")
    return FROZEN_PRIOR_GROUPS.copy()

def packet(root,provider,rows,run):
    root=Path(root)/("fresh18-observer-"+provider)
    all_ids={row["id"] for row in rows}
    if not (root/"RESULT.json").is_file():
        return {sid:None for sid in all_ids}, {
            "provider":provider,"status":"MISSING_OBSERVER_ARTIFACT",
            "complete":0,"denominator":126}
    outcome=json.loads(c.read(root/"RESULT.json"))
    need(outcome.get("schema")==worker.SCHEMA and
         outcome.get("provider")==provider and
         outcome.get("run")==run and
         outcome.get("source_artifact_id")==worker.SOURCE["id"] and
         outcome.get("denominator")==126 and
         outcome.get("new_tts_calls")==0 and
         outcome.get("target_conditioned") is False and
         outcome.get("training_allowed") is False and
         outcome.get("human_gold") is False and
         type(outcome.get("rows")) is list,
         "SOURCE_BOUND_ORIGINAL_NATIVE_OBSERVER_PACKET")
    saved=outcome["rows"]
    need(len(saved)<=126 and
         len({x.get("id") for x in saved})==len(saved) and
         {x.get("id") for x in saved}<=all_ids,
         "ONLY_ORIGINAL_NEW_126_WAV_IDS")
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
         complete<=outcome.get("attempted",0)<=126 and
         outcome.get("status")==
            ("COMPLETE" if complete==126 else "INCOMPLETE_NO_RETRY"),
         "NO_MISSING_INFERENCE_MASQUERADING_AS_ZERO")
    return results,{"provider":provider,"status":outcome["status"],
        "complete":complete,"denominator":126,
        "reported_attempted":outcome["attempted"]}

def evaluate(rows,texts_by_id,phones_by_id,prior_counts):
    need(len(rows)==126 and set(texts_by_id)==set(phones_by_id)==
         {r["id"] for r in rows},
         "COMPLETE_ORIGINAL_126_NEW_NATIVE_WAV_ROSTER")
    group_rows=defaultdict(list)
    decisions=[]
    for row in rows:
        sid=row["id"]
        outcome=policy.decide({**row,"pcm_samples":row["samples"]},
                      texts_by_id[sid],phones_by_id[sid])
        if outcome["tier"]!="unknown" and outcome["class"]!=row["requested_class"]:
            outcome={**outcome,"tier":"unknown","class":None,
                "reason":"BLIND_OBSERVED_LABEL_CONTRADICTS_REQUEST_INTENT",
                "margins":{},"training_allowed":False}
        data={"id":sid,"source_group_id":row["source_group_id"],
            "role":row["role"],"cell_id":row["cell_id"],
            "generation_render_text":row["generation_render_text"],
            "requested_class":row["requested_class"],
            "tier":outcome["tier"],"observed_class":outcome["class"],
            "reason":outcome["reason"],"phone_margins":outcome["margins"],
            "wav_sha256":row["wav_sha256"],"pcm_sha256":row["pcm_sha256"],
            "samples":row["samples"],"positive_duration_pass":row["positive_duration_pass"],
            "human_gold":False,"training_allowed":False}
        decisions.append(data);group_rows[row["source_group_id"]].append(data)
    design=worker.source_worker.plan()
    group_map={x["source_group_id"]:x["role"] for x in design["groups"]}
    need(len(group_rows)==18 and set(group_rows)==set(group_map) and
         all(len(items)==7 for items in group_rows.values()),
         "ALL_EIGHTEEN_NEW_GROUPS_SEVEN_REQUESTS")
    eligible=set()
    for group,items in group_rows.items():
        need({x["role"] for x in items}=={group_map[group]} and
             {x["cell_id"] for x in items}==
               {x["cell_id"] for x in design["cells"]},
             "PRESPLIT_ROLE_AND_CONTROL_SET")
        observed={x["observed_class"] for x in items
                  if x["tier"] in ("silver","weak")}
        if observed=={0,1,2}:eligible.add(group)
    new=Counter(group_map[x] for x in eligible)
    new_counts={role:new[role] for role in ("train","development","locked-test")}
    need(prior_counts==FROZEN_PRIOR_GROUPS and
         not set(group_map).intersection(
           {"kokoro-v1_0-preset:"+x for x in design["original_source_excluded"]}),
         "NO_OLD_PREVIOUS_SOURCE_LEAK_OR_ROLE_DRIFT")
    combined={role:prior_counts[role]+new[role] for role in policy.MIN_QUALIFIED}
    passed=all(combined[r]>=policy.MIN_QUALIFIED[r]
               for r in policy.MIN_QUALIFIED)
    return {"schema":SCHEMA,
       "status":"PASS_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE" if passed
           else "STOP_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE",
       "denominator":126,
       "new_qualified_groups_by_role":new_counts,
       "immutable_prior_qualified_groups_by_role":prior_counts,
       "cumulative_qualified_groups_by_role":combined,
       "required_qualified_groups":policy.MIN_QUALIFIED,
       "cumulative_machine_source_qualification_passed":passed,
       "source_lineage_distinct_synthetic_presets_not_verified_humans":True,
       "tier_counts":dict(Counter(x["tier"] for x in decisions)),
       "reason_counts":dict(Counter(x["reason"] for x in decisions)),
       "observed_label_counts":dict(Counter(str(x["observed_class"])
                 for x in decisions if x["tier"] in ("silver","weak"))),
       "training_snapshot_allowed":False,
       "requires_authenticated_cross_corpus_train_binding":True,
       "locked_kws_evaluation_consumed":False,
       "new_optimizer_updates":0,"new_model_weights":False,
       "human_gold":False,"ssc305_measured":False,"rows":decisions}

def combine(source,prior,observer_root,out):
    frozen=source_check()
    need(frozen["status"]=="SOURCE_PINNED",
         "EXACT_NATIVE_SOURCE_AUTHENTICATION_BEFORE_REVIEW")
    run=worker.guard()
    previous=readback_prior(prior)
    new=worker.source_rows(source)
    output=Path(out)
    need(output.is_absolute() and not output.exists(),"NEW_PROSPECTIVE_EVIDENCE_ONLY")
    output.mkdir(parents=True)
    c.record(output/"STARTED.json",{
       "schema":SCHEMA,"run":run,"source_run":worker.SOURCE_RUN,
       "source_artifact_id":worker.SOURCE["id"],
       "source_artifact_digest":worker.SOURCE["digest"],
       "original_frozen_policy_git_sha1":POLICY_ORIGINAL_BLOB_SHA1,
       "original_new18_plan_sha1":PLAN_ORIGINAL_BLOB_SHA1,
       "prior_saved_review_artifact":PRIOR,
       "new_model_calls":0,"training_allowed":False})
    transcripts={r["id"]:{} for r in new}
    native={r["id"]:{} for r in new}
    reports=[]
    for provider in worker.PROVIDERS:
        seen,facts=packet(observer_root,provider,new,run)
        reports.append(facts)
        for sid,value in seen.items():
            (native if provider in ("xlsr","lv60") else transcripts)[sid][provider]=value
    complete=all(x["complete"]==126 and x["status"]=="COMPLETE"
                 for x in reports)
    result=evaluate(new,transcripts,native,previous)
    passed=bool(complete and result["cumulative_machine_source_qualification_passed"])
    c.record(output/"SCREEN.json",{
       "schema":SCHEMA,
       "status":result["status"] if complete else "STOP_INCOMPLETE_FIVE_NATIVE_SOURCE_126",
       "run":run,"original_source_run":worker.SOURCE_RUN,
       "source_artifact_id":worker.SOURCE["id"],
       "source_artifact_digest":worker.SOURCE["digest"],
       "original_prior_168_screen_artifact":PRIOR,
       "frozen_policy_git_blob_sha1":POLICY_ORIGINAL_BLOB_SHA1,
       "frozen_new18_plan_git_blob_sha1":PLAN_ORIGINAL_BLOB_SHA1,
       "observer_reports":reports,
       "all_five_native_observers_complete":complete,
       "cumulative_machine_source_qualification_passed":passed,
       "training_snapshot_allowed":False,
       "training_binding_required_even_if_coverage_passes":True,
       "machine_review":result,
       "new_tts_calls":0,"new_model_calls_in_combine":0,
       "new_optimizer_updates":0,"new_model_weights":False,
       "locked_kws_evaluation_consumed":False,
       "human_gold":False,"ssc305_measured":False})
    print(json.dumps({"status":result["status"] if complete else
        "STOP_INCOMPLETE_FIVE_NATIVE_SOURCE_126",
        "new_source_rows":126,"all_five_observers_complete":complete,
        "new_qualified_groups_by_role":result["new_qualified_groups_by_role"],
        "cumulative_qualified_groups_by_role":result["cumulative_qualified_groups_by_role"],
        "tier_counts":result["tier_counts"],
        "cumulative_source_qualification_passed":passed,
        "training_snapshot_allowed":False},sort_keys=True))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-check",action="store_true")
    parser.add_argument("--combine",action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--prior")
    parser.add_argument("--observers")
    parser.add_argument("--output")
    opts=parser.parse_args()
    need(opts.source_check!=opts.combine,"ONE_CLOSED_RESEARCH_STAGE")
    if opts.source_check:
        print(json.dumps(source_check(),sort_keys=True));return
    need(all((opts.source,opts.prior,opts.observers,opts.output)),
         "ORIGINAL_SOURCE_AND_ALL_FIVE_OBSERVERS_REQUIRED")
    combine(opts.source,opts.prior,opts.observers,opts.output)

if __name__=="__main__":
    main()
