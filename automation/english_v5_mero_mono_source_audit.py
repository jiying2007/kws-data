"""Read-only Hey Mero mono-target admission on EXISTING five-model-reviewed PCM.

The historical three-target classifier required ordinary+Mero+Vosen from each
source group. V5 has exactly ONE English target, so we recompute eligibility
from unchanged independent silver/weak labels for ordinary(0)+Mero(2).
We NEVER rewrite screening tiers, repair unknowns, treat the retired keyword as
ordinary, resplit source groups, open old calibration/locked audio, or grant
unobserved voices. Prior selected 35 original research rows must be preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location(
    "en_v5_mono_original_auth", HERE/"english_v5_mero_fsmn_ctc_once.py")
v5=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v5)

SCHEMA="english-v5-mero-original-independent-mono-source-audit-v1"
SOURCE_GROUP_LABELS=(0,2)
SPLITS=("train","development")
EXPECTED_COUNTS={"fresh24":168,"fresh18":126,"piper24":168,"newtrain24":168}
EXPECTED_TOTAL=630


def need(ok,code):
    if not ok: raise ValueError(code)


def source_check():
    v=v5.source_check()
    need(v["status"]=="FROZEN_MONO_MERO_RESEARCH_SOURCE"
         and v["active_target_ids"]==["hey_mero"]
         and v["old_locked_reopened"] is False
         and v5.config()["target"]["ipa"]=="/heɪ ˈmɛroʊ/"
         and len(v5.ARCHIVES)==8
         and EXPECTED_TOTAL==sum(EXPECTED_COUNTS.values()),
         "EXACT_OLD_INDEPENDENT_SCREENS_SINGLE_V5_MERO_ONLY")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_AUDIO_OR_MODEL_CALLED",
            "keyword":"Hey Mero","brand_ipa":"/heɪ ˈmɛroʊ/",
            "admitted_original_labels":[0,2],
            "historical_other_label":1,
            "audited_original_pcm_count":EXPECTED_TOTAL,
            "roles_read":["train","development"],
            "old_calibration_and_exposed_locked_pcm":"NOT_OPENED",
            "training":False,"new_tts_asr_ctc_calls":0,
            "ssc305":"NOT_RUN","quality_promotion":False}


def select_mono(observations,baseline):
    """Pure function: preserve old rows, never promote unknown or other word."""
    role_by_group={};accepted=[];all_ids=set();all_hashes=set()
    for row in observations:
        need(row["role"] in SPLITS and row["tier"] in ("silver","weak","unknown")
             and row["source_group_id"] and
             row["requested_class"] in (0,1,2,None)
             and row["id"] not in all_ids
             and re.fullmatch(r"[0-9a-f]{64}",row["source_pcm_sha256"])
             and row["source_pcm_sha256"] not in all_hashes
             and row["human_gold"] is False
             and row["training_allowed_in_original"] is False,
             "UNIQUE_ORIGINAL_SOURCE_ROW_ROLE_AND_AUDIT")
        all_ids.add(row["id"])
        all_hashes.add(row["source_pcm_sha256"])
        group=row["source_group_id"]
        need(group not in role_by_group or role_by_group[group]==row["role"],
             "IMMUTABLE_SOURCE_GROUP_ROLE")
        role_by_group[group]=row["role"]
        if row["tier"]=="unknown":
            need(row["weak_class"] is None,
                 "UNKNOWN_NEVER_CLASSIFIED_OR_SUPERVISED")
            continue
        need(row["weak_class"] in (0,1,2)
             and row["weak_class"]==row["requested_class"],
             "EXACT_PRESERVED_NATIVE_REVIEW_CLASS")
        if row["weak_class"] in SOURCE_GROUP_LABELS:
            accepted.append({**row,
                "mono_class":1 if row["weak_class"]==2 else 0,
                "ctc_targets":[1,2,3] if row["weak_class"]==2 else [],
                "mono_keyword_id":"hey_mero" if row["weak_class"]==2 else None})
    by_group=defaultdict(set)
    for row in accepted:
        by_group[row["source_group_id"]].add(row["weak_class"])
    qualified={g for g,c in by_group.items() if c==set(SOURCE_GROUP_LABELS)}
    selected=sorted((r for r in accepted if r["source_group_id"] in qualified),
                    key=lambda r:(r["role"],r["corpus"],r["source_group_id"],r["id"]))
    selected_by_id={r["id"]:r for r in selected}
    need(len(selected_by_id)==len(selected),"DUPLICATE_SELECTED_ID")
    for old in baseline:
        now=selected_by_id.get(old["id"])
        need(now is not None and
             all(now[k]==old[k] for k in
                 ("source_group_id","role","source_pcm_sha256","mono_class",
                  "ctc_targets","tier","weak_class","requested_class")),
             "PREVIOUS_V5_35_ORIGINAL_TRAIN_DEV_ROWS_MUST_SURVIVE")
    counts=Counter((r["role"],r["mono_class"]) for r in selected)
    groups=Counter(role_by_group[g] for g in qualified)
    need(len(baseline)==35 and
         Counter(x["role"] for x in baseline)=={"train":24,"development":11}
         and groups["train"]>=10 and groups["development"]>=5
         and counts[("train",0)]>=10 and counts[("train",1)]>=14
         and counts[("development",0)]>=5 and counts[("development",1)]>=6,
         "DO_NOT_LOSE_PREVIOUS_35_OR_SOURCE_ROLES")
    extra=[r for r in selected if r["id"] not in {b["id"] for b in baseline}]
    return selected,{
        "all_reviewed_original_rows":len(observations),
        "previously_selected_original_rows":len(baseline),
        "mono_selected_rows":len(selected),
        "newly_recovered_from_original_independent_review":len(extra),
        "mono_source_groups":{role:groups[role] for role in SPLITS},
        "mono_rows_by_role_class":[{"role":role,"class":label,
              "count":counts[(role,label)]}
             for role in SPLITS for label in (0,1)],
        "additional_source_groups_by_role":
            {role:len({r["source_group_id"] for r in extra
                  if r["role"]==role}-
                {r["source_group_id"] for r in baseline
                  if r["role"]==role})
             for role in SPLITS},
        "additional_m ero_positives_by_role".replace(" ",""):
            {role:sum(r["role"]==role and r["mono_class"]==1 for r in extra)
             for role in SPLITS},
        "previous_selection_exactly_preserved":True,
        "no_old_other_word_becomes_negative":True,
        "no_unknown_relabel":True,
        "all_labels_original_machine_silver_weak":True,
        "no_human_gold":True,
        "no_calibration_or_locked_audio_read":True,
        "new_tts_asr_phone_calls":0,
        "release_promotion":False}


def _source_rows(original,screen,corpus):
    """Metadata is read for all roles, WAV bytes only for train/dev accepted."""
    manifest=json.loads(original.read("MANIFEST.json"))
    result=json.loads(screen.read("SCREEN.json"))
    reviews=result.get("machine_review",{})
    candidates=reviews.get("rows",[])
    rows=manifest.get("rows",[])
    need(len(rows)==len(candidates)==EXPECTED_COUNTS[corpus] and
         result.get("all_five_native_observers_complete") is True and
         result.get("training_snapshot_allowed") is False and
         reviews.get("denominator",len(candidates))==EXPECTED_COUNTS[corpus],
         "FULL_ORIGINAL_SOURCE_VS_FIVE_INDEPENDENT_SCREEN")
    lookup={r["id"]:r for r in candidates}
    need(len(lookup)==len(candidates) and
         len({r["id"] for r in rows})==len(rows) and
         set(lookup)=={r["id"] for r in rows},
         "EXACT_FULL_ORIGINAL_SCREEN_IDS")
    retained=[]
    for raw in rows:
        review=lookup[raw["id"]]
        for key in ("id","role","source_group_id","requested_class",
                    "wav_sha256","pcm_sha256","samples"):
            need(raw[key]==review[key],
                 "ORIGINAL_SOURCE_AND_MACHINE_VERDICT_MISMATCH")
        need(raw["human_gold"] is False and raw["training_allowed"] is False
             and raw.get("observed_class") is None and
             review["human_gold"] is False and
             review["training_allowed"] is False,
             "NO_SOURCE_OR_VERDICT_AUTO_PROMOTION")
        if raw["role"] not in SPLITS:
            need(raw["role"] in ("calibration","locked-test") and
                 corpus!="newtrain24",
                 "UNKNOWN_HISTORICAL_HOLDOUT_ROLE")
            # Critical: no old calibration/locked WAV is ever read here.
            continue
        tier=review["tier"]
        cls=review["observed_class"]
        need(tier in ("silver","weak","unknown")
             and (cls is None if tier=="unknown" else
                  cls in (0,1,2) and cls==raw["requested_class"]),
             "REVIEW_TIERS_NOT_ADJUSTED")
        if tier in ("silver","weak") and cls in SOURCE_GROUP_LABELS:
            v5.binding.read_wav(original.read("wav/"+raw["id"]+".wav"),raw)
        retained.append({"id":raw["id"],
           "corpus":corpus,"role":raw["role"],
           "source_group_id":raw["source_group_id"],
           "source_pcm_sha256":raw["pcm_sha256"],
           "source_wav_sha256":raw["wav_sha256"],
           "samples":raw["samples"],"tier":tier,
           "weak_class":cls,"requested_class":raw["requested_class"],
           "human_gold":False,"training_allowed_in_original":False,
           "machine_reason":review["reason"]})
    return retained


def audit(path):
    source_check()
    path=Path(path)
    need(path.is_absolute() and path.is_dir() and not path.is_symlink(),
         "PINNED_SOURCE_ZIP_ROOT")
    # First reuse original independent six-artifact and newtrain 168 verifiers.
    baseline,original_receipt=v5.admission(path)
    recorded=v5.binding.admission(path,verify_audio=False)
    need(len(recorded["rows"])==63
         and recorded["source_group_counts"]==
          {"train":8,"development":4,"calibration":3,"locked-test":3},
         "ORIGINAL_THREE_WORD_63_RETAINED_NOT_EDITED")
    new,receipt=v5.prior.bind_new(path/"newtrain24-source.zip",
                                 path/"newtrain24-screen.zip")
    need(len(new)==11 and receipt["new_source_groups"]==
         {"train":2,"development":1},
         "ORIGINAL_NEWTRAIN168_FROZEN_REVIEW_UNCHANGED")
    all_rows=[]
    for spec in v5.binding.config()["corpus"]:
        key=spec["key"]
        source=v5.binding.archive(path/(key+"-source.zip"),spec["source"])
        screen=v5.binding.archive(path/(key+"-screen.zip"),spec["screen"])
        try:
            all_rows.extend(_source_rows(source,screen,key))
        finally:
            source.close();screen.close()
    source=v5.binding.archive(path/"newtrain24-source.zip",v5.prior.SOURCE)
    screen=v5.binding.archive(path/"newtrain24-screen.zip",v5.prior.SCREEN)
    try:
        all_rows.extend(_source_rows(source,screen,"newtrain24"))
    finally:
        source.close();screen.close()
    selected,summary=select_mono(all_rows,baseline)
    # All original train/dev PCM selected from machine-admitted screen,
    # after exactly the original role/speaker/source checks, are recoverable.
    output={"schema":SCHEMA,"status":
          "NEW_MONO_SOURCE_COVERAGE_FOUND_NOT_PROMOTED"
          if summary["newly_recovered_from_original_independent_review"] else
          "NO_EXTRA_MONO_SOURCE_COVERAGE",
        "base_main":"2d5554a9b8bb6c12c0e932cd8257927298179694",
        "archived_source_artifacts":dict(v5.ARCHIVES),
        "old_63_and_new_11_preserved":True,
        "summary":summary,
        "source_role_rows":selected,
        "old_calibration_test_pcm_not_opened":True,
        "independent_human_3_5m":"NOT_MEASURED",
        "new_training_model":"NOT_RUN",
        "source_review_not_a_fresh_locked_test":True,
        "production_release_allowed":False}
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--audit",action="store_true")
    p.add_argument("--sources",type=Path)
    p.add_argument("--output",type=Path)
    opts=p.parse_args()
    need(opts.source_check!=opts.audit,"EXACTLY_ONE_AUDIT_STAGE")
    if opts.source_check:
        print(json.dumps(source_check(),sort_keys=True))
        return
    need(opts.sources and opts.output and opts.output.is_absolute()
         and opts.output.parent.is_dir() and not opts.output.exists(),
         "EXACT_PINNED_ARCHIVES_AND_CREATE_ONLY_RECEIPT")
    result=audit(opts.sources)
    with opts.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    print(json.dumps({"status":result["status"],"summary":result["summary"],
           "training":False,"quality_promotion":False},sort_keys=True))


if __name__=="__main__":
    main()
