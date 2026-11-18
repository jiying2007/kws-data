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
worker=load("english_piper_fresh24_five_native_once")
policy=load("english_fresh24_ctc_admission")
SCHEMA="english-piper-fresh24-prospective-native-cumulative-screen-v1"
POLICY_ORIGINAL_BLOB_SHA1="0a688325d0d36e83ef6623ac678511df24a92374"
PLAN_ORIGINAL_BLOB_SHA1="d477ecec2a07910fc1a91063752aff1acc9006ef"

def need(ok,reason):
    if not ok:raise ValueError(reason)

def git_blob(path):
    raw=c.read(path)
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()

PRIOR={"run_id":44312719394,"artifact_id":14429884666,
       "name":"english-fresh18-cumulative-screen-44312719394",
       "bytes":15068,"sha256":"4a685afa1d8e8c35767900290a7fd590f2153fb69f3b2e5ae68d5dada58b11e1",
       "original_source_run":44310115097,
       "head_sha":"6fc03ff706b8949b9a6c066f4478890db8de28aa"}
FROZEN_PRIOR_GROUPS={"train":4,"development":1,
                     "calibration":3,"locked-test":1}
def source_check():
    check=worker.source_check()
    need(git_blob(HERE/"english_fresh24_ctc_admission.py")==
         POLICY_ORIGINAL_BLOB_SHA1 and
         git_blob(HERE/"english_piper_fresh24_protocol.json")==
         PLAN_ORIGINAL_BLOB_SHA1 and
         worker.SOURCE_COUNT==168 and
         worker.SOURCE["id"]==14434542792 and
         tuple(worker.PROVIDERS)==tuple(policy.TEXT_MODELS)+tuple(policy.PHONE_MODELS) and
         policy.MIN_QUALIFIED=={"train":6,"development":2,
                               "calibration":2,"locked-test":2} and
         worker.source_worker.protocol()["already_qualified_readonly"]==
              FROZEN_PRIOR_GROUPS,
         "ORIGINAL_168_DIRECT_PIPER_AND_PREAUDIO_CTC_RULE")
    return {**check,"frozen_policy_git_blob_sha1":POLICY_ORIGINAL_BLOB_SHA1,
            "source_plan_git_blob_sha1":PLAN_ORIGINAL_BLOB_SHA1,
            "prior_evidence_artifact_id":PRIOR["artifact_id"],
            "training_allowed":False}
def verify_prior_remote(token):
    """Independent immutable GitHub Actions artifact identity, not a filename trust."""
    import urllib.request
    need(type(token) is str and len(token)>=10,"READONLY_ACTIONS_TOKEN")
    request=urllib.request.Request(
         f"https://api.github.com/repos/{worker.REPO}/actions/artifacts/{PRIOR['artifact_id']}",
         headers={"Authorization":"Bearer "+token,
                  "Accept":"application/vnd.github+json",
                  "User-Agent":"piper-fresh24-immutable-prior-evidence"})
    with urllib.request.urlopen(request,timeout=30) as response:
        value=json.loads(response.read(1000001))
    need(value.get("id")==PRIOR["artifact_id"] and
         value.get("name")==PRIOR["name"] and
         value.get("size_in_bytes")==PRIOR["bytes"] and
         value.get("digest")=="sha256:"+PRIOR["sha256"] and
         value.get("expired") is False and
         value.get("workflow_run",{}).get("id")==PRIOR["run_id"] and
         value.get("workflow_run",{}).get("head_sha")==PRIOR["head_sha"],
         "EXACT_UNMODIFIED_PRIOR_126_SOURCE_QUALIFICATION_STOP")
    return {"prior_artifact":PRIOR["artifact_id"],"verified":True}
def readback_prior(root):
    """Read ONLY previous immutable aggregate, not the exposed locked rows."""
    screen=json.loads(c.read(Path(root)/"SCREEN.json"))
    review=screen.get("machine_review",{})
    need(screen.get("schema")==
              "english-fresh18-joined-prospective-blind-source-screen-v1" and
         screen.get("status")=="STOP_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE" and
         screen.get("original_source_run")==PRIOR["original_source_run"] and
         screen.get("source_artifact_id")==14429578824 and
         screen.get("all_five_native_observers_complete") is True and
         screen.get("cumulative_machine_source_qualification_passed") is False and
         screen.get("training_snapshot_allowed") is False and
         screen.get("new_optimizer_updates")==0 and
         review.get("cumulative_qualified_groups_by_role")==FROZEN_PRIOR_GROUPS and
         review.get("denominator")==168 and
         review.get("new_model_weights") is False,
         "PREVIOUS_EXACT_ORIGINAL_126_IMMUTABLE_STOP")
    return FROZEN_PRIOR_GROUPS.copy()

def packet(root,provider,rows,run):
    root=Path(root)/("pipernew-observer-"+provider)
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
         complete<=outcome.get("attempted",0)<=168 and
         outcome.get("status")==
            ("COMPLETE" if complete==168 else "INCOMPLETE_NO_RETRY"),
         "NO_MISSING_INFERENCE_MASQUERADING_AS_ZERO")
    return results,{"provider":provider,"status":outcome["status"],
        "complete":complete,"denominator":168,
        "reported_attempted":outcome["attempted"]}

def evaluate(rows,texts_by_id,phones_by_id,prior_counts):
    need(len(rows)==168 and set(texts_by_id)==set(phones_by_id)==
         {r["id"] for r in rows},
         "COMPLETE_NEW_168_PIPER_VOICE_AND_REQUEST_ROSTER")
    decisions=[];by_group=defaultdict(list)
    for row in rows:
        sid=row["id"]
        outcome=policy.decide({**row,"pcm_samples":row["samples"]},
                   texts_by_id[sid],phones_by_id[sid])
        if outcome["tier"]!="unknown" and outcome["class"]!=row["requested_class"]:
            outcome={**outcome,"tier":"unknown","class":None,
                     "reason":"INDEPENDENT_ACOUSTIC_REQUEST_CONTRADICTION",
                     "margins":{},"training_allowed":False}
        data={"id":sid,"source_group_id":row["source_group_id"],
              "source_speaker_id":row["source_speaker_id"],
              "speaker_index":row["speaker_index"],
              "role":row["role"],"cell_id":row["cell_id"],
              "requested_class":row["requested_class"],
              "tier":outcome["tier"],"observed_class":outcome["class"],
              "reason":outcome["reason"],"phone_margins":outcome["margins"],
              "wav_sha256":row["wav_sha256"],
              "pcm_sha256":row["pcm_sha256"],
              "samples":row["samples"],
              "positive_duration_pass":row["positive_duration_pass"],
              "training_allowed":False,"human_gold":False}
        decisions.append(data);by_group[row["source_group_id"]].append(data)
    need(len(by_group)==24 and all(len(v)==7 for v in by_group.values()),
         "EXACT_24_NEW_SPEAKER_GROUPS_BY_SEVEN_CELLS")
    source_roles={}
    for row in rows:
        key=row["source_group_id"]
        need(key=="piper-libritts-"+row["source_speaker_id"] and
             key.startswith("piper-libritts-p"),
             "PIPER_SOURCE_FAMILY_AND_NUMERIC_ID")
        if key in source_roles:
            need(source_roles[key]==row["role"],"SOURCE_ROLE_MIXED")
        source_roles[key]=row["role"]
    complete=set()
    for group,items in by_group.items():
        observed={x["observed_class"] for x in items
                  if x["tier"] in ("silver","weak")}
        if observed=={0,1,2}:complete.add(group)
    latest=Counter(source_roles[g] for g in complete)
    extra={role:latest[role] for role in ("train","development","locked-test")}
    need(prior_counts==FROZEN_PRIOR_GROUPS and
         not set(source_roles).intersection(
            {"kokoro-v1_0-preset:"+x for x in worker.source_worker.protocol()["groups"]}),
         "NO_REASSIGNMENT_OR_PREVIOUS_SOURCE_MIX")
    total={r:prior_counts[r]+latest[r] for r in policy.MIN_QUALIFIED}
    allowed=all(total[r]>=policy.MIN_QUALIFIED[r] for r in policy.MIN_QUALIFIED)
    return {"schema":SCHEMA,
      "status":"PASS_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE" if allowed
               else "STOP_CUMULATIVE_6_2_2_2_SOURCE_COVERAGE",
      "denominator":168,
      "new_qualified_groups_by_role":extra,
      "immutable_prior_qualified_groups_by_role":prior_counts,
      "cumulative_qualified_groups_by_role":total,
      "minimum_qualified_groups":policy.MIN_QUALIFIED,
      "cumulative_machine_source_qualification_passed":allowed,
      "source_identity":"distinct new LibriTTS pNNN speaker source IDs; synthetic embeddings not independent real human recording test",
      "tier_counts":dict(Counter(x["tier"] for x in decisions)),
      "reason_counts":dict(Counter(x["reason"] for x in decisions)),
      "observed_label_counts":dict(Counter(str(x["observed_class"]) for x in decisions
                                  if x["tier"] in ("silver","weak"))),
      "training_snapshot_allowed":False,
      "training_binding_required":True,
      "new_optimizer_updates":0,"new_model_weights":False,
      "human_gold":False,"locked_kws_evaluation_consumed":False,
      "ssc305_measured":False,
      "rows":decisions}

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
       "original_piper_source_plan_sha1":PLAN_ORIGINAL_BLOB_SHA1,
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
    complete=all(x["complete"]==168 and x["status"]=="COMPLETE"
                 for x in reports)
    result=evaluate(new,transcripts,native,previous)
    passed=bool(complete and result["cumulative_machine_source_qualification_passed"])
    c.record(output/"SCREEN.json",{
       "schema":SCHEMA,
       "status":result["status"] if complete else "STOP_INCOMPLETE_FIVE_NATIVE_SOURCE_126",
       "run":run,"original_source_run":worker.SOURCE_RUN,
       "source_artifact_id":worker.SOURCE["id"],
       "source_artifact_digest":worker.SOURCE["digest"],
       "original_prior_126_screen_artifact":PRIOR,
       "frozen_policy_git_blob_sha1":POLICY_ORIGINAL_BLOB_SHA1,
       "frozen_piper_new_plan_git_blob_sha1":PLAN_ORIGINAL_BLOB_SHA1,
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
        "new_source_rows":168,"all_five_observers_complete":complete,
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
