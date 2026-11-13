"""Rebuild existing English 63-source pronunciation review queue (zero model calls).

Read only the three SHA256-pinned original GitHub Actions ZIPs:
three-ASR source-linked review, Log-Mel/DTW, and blind IPA observations.
Cross-bind each cell by original WAV SHA256 and source/split identity.
This cannot authenticate acoustic speech, promote labels or train a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

POLICY = "english63-independent-evidence-joined-review-priority-v1"
SOURCES = {
    "asr": (
        "3786baa9fd3575437a05e97956fafdf3298eac23de2538b485b8342ef4d25133",
        "60054a3532491631a47b094ea7ff2bc86ef16530ff510f968bc9edcdd253a9ad",
        "english-63-machine-review.json",
    ),
    "dtw": (
        "64efe60c4b8c45cf130b2b55b4be6674f665dbcf7d667548c463dda15d951e18",
        "6dfe169bb37a8264e281270c3eb6aaa3d49135a79a2eb6761cffb69b1b5681f4",
        "english63-paired-acoustic-result.json",
    ),
    "ipa": (
        "a70ee89c76bf25f8a0a6b82a5a72c46128782e7f2020fb7655c06845de131be0",
        "093e18093a52bffa68aa60082cc490fdc3f9129eb1d60a209c2fef1d29e9e151",
        "english63-ipa-result/summary.json",
    ),
}
PROVIDERS = ("qwen", "whisper", "parakeet")
TRAIN_VOICES = {"dylan", "eric", "ono_anna", "sohee", "vivian"}
DEV_VOICES = {"serena", "uncle_fu"}
REQUESTS = {
    "Hey Vosen": "wake", "Hey, Vosen!": "wake",
    "Hey Mero": "wake", "Hey, Mero!": "wake",
    "Hey Wosen": "near_miss", "Hey Miro": "near_miss",
    "Hey Rosen": "near_miss",
    "What time is it?": "ordinary_negative",
    "Hey, can you hear me?": "ordinary_negative",
}
WAKES = (
    ("vosen", "Hey Vosen", "Hey, Vosen!",
     ("vosen_vs_wosen", "vosen_vs_rosen")),
    ("mero", "Hey Mero", "Hey, Mero!", ("mero_vs_miro",)),
)


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_original_archive(path: Path, name: str) -> dict:
    need(name in SOURCES and path.is_absolute() and path.is_file()
         and not path.is_symlink() and path.stat().st_size < 512*1024,
         "ENGLISH63_ARCHIVE_INPUT_INVALID")
    expected_zip, expected_json, member = SOURCES[name]
    need(sha(path.read_bytes()) == expected_zip, "ENGLISH63_SOURCE_ZIP_DRIFT")
    with zipfile.ZipFile(path) as z:
        need(z.namelist().count(member) == 1 and
             0 < z.getinfo(member).file_size <= 2*1024*1024,
             "ENGLISH63_ARCHIVE_MEMBER_INVALID")
        with z.open(member) as stream:
            content = stream.read(2*1024*1024 + 1)
    need(sha(content) == expected_json, "ENGLISH63_SOURCE_JSON_DRIFT")
    return json.loads(content)


def _by_id(records: list[dict], purpose: str) -> dict:
    need(type(records) is list and len(records) == 63 and
         all(type(row) is dict and type(row.get("cell_id")) is str
             for row in records),
         "ENGLISH63_ROW_SHAPE_" + purpose)
    out = {r["cell_id"]:r for r in records}
    need(len(out) == 63, "ENGLISH63_DUPLICATED_CELL_" + purpose)
    return out


def join(asr: dict, dtw: dict, ipa: dict) -> dict:
    """Build machine attention queue; no inference, threshold or admission."""
    need(
        asr.get("schema") == "english-63-three-asr-source-linked-review-v1" and
        dtw.get("schema") == "english-63-paired-logmel-dtw-acoustic-contrast-v1" and
        ipa.get("schema") == "english63-independent-ipa-ctc-machine-review-v1" and
        asr.get("training_admitted") is False and
        dtw.get("training_admitted") is False and
        ipa.get("training_admitted") is False and
        asr.get("human_gold") is False and
        dtw.get("human_gold") is False and
        ipa.get("human_gold") is False and
        asr.get("counts",{}).get("three_way_completed_calls") == 189 and
        ipa.get("new_real_phoneme_forwards") == 63 and
        dtw.get("sample_count") == 63 and
        asr.get("source_tts_run_id") == dtw.get("tts_run_id") == 43242278962 and
        ipa.get("reused_prior_three_asr_calls") == 189,
        "ENGLISH63_THREE_EVIDENCE_POLICY_MISMATCH")
    by_asr = _by_id(asr.get("rows"),"asr")
    by_dtw = _by_id(dtw.get("sample_receipts"),"dtw")
    by_ipa = _by_id(ipa.get("rows"),"ipa")
    need(set(by_asr) == set(by_dtw) == set(by_ipa),
         "ENGLISH63_THREE_EVIDENCE_CELL_MISMATCH")
    rows, waves, pcms, voice_prompts = [],set(),set(),set()
    for cid,a in by_asr.items():
        d,i = by_dtw[cid],by_ipa[cid]
        need(
            a.get("wav_sha256") == d.get("wav_sha256") ==
            i.get("wav_sha256") and
            a.get("voice_id") == d.get("voice_id") == i.get("voice_id") and
            a.get("requested_text") == d.get("text_requested") ==
            i.get("text_requested") and
            a.get("role") == d.get("role") == i.get("role") and
            a.get("source_family_id") == d.get("source_family_id") ==
            i.get("source_family_id") and
            a.get("requested_intent") == d.get("requested_intent") ==
            i.get("requested_intent") and
            all(x.get("human_gold") is False and
                x.get("training_admitted") is False for x in (a,d,i)) and
            a.get("acoustic_pronunciation_verified") is False and
            a.get("trusted_negative") is False and
            i.get("phonetic_label_authenticated") is False and
            i.get("trusted_negative") is False,
            "ENGLISH63_SOURCE_ROLE_HASH_OR_GOLD_DRIFT")
        voice = a["voice_id"]
        need(a.get("requested_text") in REQUESTS and
             a.get("requested_intent") == REQUESTS[a["requested_text"]],
             "ENGLISH63_PROMPT_INTENT_DRIFT")
        known = TRAIN_VOICES | DEV_VOICES
        need(voice in known and a["source_family_id"] ==
             "qwen3-preset-" + voice and
             a["role"] == ("train-candidate" if voice in TRAIN_VOICES
                           else "development-only"),
             "ENGLISH63_VOICE_ROLE_DRIFT")
        need(a["wav_sha256"] not in waves and
             d["pcm_sha256"] not in pcms and
             (voice,a["requested_text"]) not in voice_prompts,
             "ENGLISH63_DUPLICATE_PCM_WAV_OR_PROMPT")
        waves.add(a["wav_sha256"])
        pcms.add(d["pcm_sha256"])
        voice_prompts.add((voice,a["requested_text"]))
        observations = a.get("asr")
        need(type(observations) is dict and set(observations) ==
             set(PROVIDERS) and
             all(type(observations[p].get("text")) is str and
                 type(observations[p].get("normalized")) is str and
                 type(observations[p].get("native_sha256")) is str
                 for p in PROVIDERS) and
             type(i.get("stressless_phones")) is list and
             len(i["stressless_phones"]) >= 2,
             "ENGLISH63_ASR_OR_IPA_NOT_COMPLETE")
        rows.append({
            "cell_id":cid,"source_family_id":a["source_family_id"],
            "voice_id":voice,"role":a["role"],
            "wav_sha256":a["wav_sha256"],"pcm_sha256":d["pcm_sha256"],
            "requested_text":a["requested_text"],
            "requested_intent":a["requested_intent"],
            "machine_agreement":a["machine_agreement"],
            "asr_observations":{
                p:{"text":o["text"],"normalized":o["normalized"],
                   "native_sha256":o["native_sha256"]}
                for p,o in observations.items()},
            "ipa_tokens":i["stressless_phones"],
            "ipa_raw_tokens":i["raw_ipa_tokens"],
            "prioritized_for_human_review_only":
                a["requested_intent"] in ("wake","near_miss"),
            "phonetic_label_authenticated":False,
            "trusted_negative":False,"human_gold":False,
            "training_admitted":False,"independent_evaluation":False,
        })
    prompts = set(voice_prompts)
    need(prompts == {(v, p) for v in TRAIN_VOICES | DEV_VOICES
                     for p in REQUESTS},
         "ENGLISH63_PROMPT_ROSTER_DRIFT")
    by_word = {(r["voice_id"],r["requested_text"]):r for r in rows}
    dtw_comparisons = {v["voice_id"]:v["comparisons"]
                       for v in dtw["by_voice"]}
    ipa_comparisons = {v["voice_id"]:v["contrasts"]
                       for v in ipa["by_voice"]}
    need(set(dtw_comparisons) == set(ipa_comparisons) ==
         TRAIN_VOICES | DEV_VOICES,
         "ENGLISH63_COMPARISON_VOICE_DRIFT")
    pairs = []
    for voice in sorted(TRAIN_VOICES | DEV_VOICES):
        for label,plain,marked,contrasts in WAKES:
            a,b = by_word[voice,plain],by_word[voice,marked]
            differ = sum(a["asr_observations"][p]["normalized"] !=
                         b["asr_observations"][p]["normalized"]
                         for p in PROVIDERS)
            same = ipa_comparisons[voice][contrasts[0]][
                "same_wake_variant_phone_edit"]
            need(all(ipa_comparisons[voice][c][
                 "same_wake_variant_phone_edit"] == same
                 for c in contrasts),
                 "ENGLISH63_SAME_WAKE_REFERENCE_DRIFT")
            comparisons = {c:{
                "dtw_near_to_same_ratio":dtw_comparisons[voice][c][
                    "nearphone_to_same_variant_ratio"],
                "dtw_near_not_farther":dtw_comparisons[voice][c][
                    "nearphone_not_farther_than_same"],
                "ipa_near_phone_edit":ipa_comparisons[voice][c][
                    "nearest_nearphone_phone_edit"],
                "ipa_near_not_farther":ipa_comparisons[voice][c][
                    "nearphone_not_farther_than_wake_variant"],
                } for c in contrasts}
            pairs.append({
                "voice_id":voice,"source_family_id":a["source_family_id"],
                "role":a["role"],"wake_candidate":label,
                "requested_texts":[plain,marked],
                "cell_ids":[a["cell_id"],b["cell_id"]],
                "same_wake_variants_ipa_edit":same,
                "three_asr_provider_pairs_differ_across_variants":differ,
                "same_wake_variant_ipa":{
                    "plain":a["ipa_tokens"],"punctuated":b["ipa_tokens"]},
                "asr_pair_transcripts":{p:{
                    "plain":a["asr_observations"][p]["text"],
                    "punctuated":b["asr_observations"][p]["text"]}
                    for p in PROVIDERS},
                "nearphone_comparisons":comparisons,
                "machine_only_priority":True,"human_gold":False,
                "phonetic_label_authenticated":False,
                "trusted_negative":False,"training_admitted":False,
            })
    pairs.sort(key=lambda r:(
        r["role"] != "train-candidate",
        -r["same_wake_variants_ipa_edit"],
        -r["three_asr_provider_pairs_differ_across_variants"],
        -sum(x["ipa_near_not_farther"] for x in
             r["nearphone_comparisons"].values()),
        r["voice_id"],r["wake_candidate"]))
    for index,row in enumerate(pairs,start=1):
        row["diagnostic_review_order"] = index
    need(len(pairs) == 14 and
         sum(r["role"] == "train-candidate" for r in pairs) == 10 and
         sum(r["role"] == "development-only" for r in pairs) == 4,
         "ENGLISH63_REVIEW_PAIR_SPLIT_DRIFT")
    return {
        "schema":POLICY,
        "status":"MACHINE_EVIDENCE_JOINED_REVIEW_NOT_GOLD",
        "source_archive_sha256":{k:SOURCES[k][0] for k in SOURCES},
        "source_payload_sha256":{k:SOURCES[k][1] for k in SOURCES},
        "source_tts_run_id":43242278962,
        "original_three_asr_run_id":43246104418,
        "source_ipa_run_id":43340235981,
        "audio_rows_reused":63,"asr_observations_reused":189,
        "new_model_calls":0,"voice_count_synthetic":7,
        "review_wake_pairs":14,"nearphone_challenges":21,
        "wake_pairs":pairs,"rows":sorted(rows,key=lambda r:r["cell_id"]),
        "role_separation":"First 10 train-candidate pairs are diagnostic review candidates; final 4 development-only pairs are exposed diagnostics, never training or model tuning.",
        "training_admitted":False,"human_gold":False,
        "independent_human_validation":"NOT_RUN",
        "note":"Sorting is exploratory attention prioritization, not an acceptance threshold. No audio label or role is changed.",
    }


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in SOURCES:
        parser.add_argument("--"+name+"-zip",type=Path)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    try:
        sources={k:getattr(args,k+"_zip") for k in SOURCES}
        if all(v is None for v in sources.values()) and args.output_dir is None:
            print(json.dumps({"schema":POLICY,"new_model_calls":0,
                              "training_admitted":False},sort_keys=True))
            return 0
        need(all(v is not None for v in sources.values()) and
             args.output_dir is not None and
             args.output_dir.is_absolute() and
             args.output_dir.parent.is_dir() and
             not args.output_dir.exists(),
             "ENGLISH63_EXPLICIT_CREATE_ONLY_INPUTS_REQUIRED")
        report=join(*(load_original_archive(sources[k],k)
                      for k in SOURCES))
        dest=args.output_dir
        dest.mkdir(mode=0o700)
        report_path=dest/"english63-three-evidence-review.json"
        raw=(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)
             + "\n").encode()
        report_path.write_bytes(raw)
        report_path.chmod(0o600)
        csv_path=dest/"english63-wake-pairs.csv"
        fields=["diagnostic_review_order","voice_id","role",
                "wake_candidate","same_wake_variants_ipa_edit",
                "three_asr_provider_pairs_differ_across_variants","cell_ids",
                "human_gold","training_admitted"]
        with csv_path.open("x",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields)
            w.writeheader()
            for item in report["wake_pairs"]:
                w.writerow({k:";".join(item[k]) if k=="cell_ids" else
                            item[k] for k in fields})
        csv_path.chmod(0o600)
        print(json.dumps({"status":report["status"],"report_sha256":sha(raw),
                          "rows":63,"review_pairs":14,
                          "training_admitted":False},sort_keys=True))
        return 0
    except (ValueError,OSError,json.JSONDecodeError,zipfile.BadZipFile) as exc:
        print("ENGLISH63_TRIAGE_STOP: "+type(exc).__name__+": "+str(exc),
              file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
