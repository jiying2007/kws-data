#!/usr/bin/env python3
"""Read-only, one-shot B4/B5 original-PCM paired result pathology.

No model inference, no audio generation, no optimizer, no score tuning.
The original B5 117-forward result remains in its original Actions artifact.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

REPO = "jiying2007/kws-data"
SCHEMA = "english-v6-b5-dev504-frozen-saved-lowgain-mechanism-v1"
PARENT = "1668301a2116cabb0e78ca247ae1102b1ab25d98"
ORIGINAL = {
    "run": 44700786677,
    "head": PARENT,
    "id": 14611963808,
    "name": "english-v6-b5-dev504-three-frozen-paired-44700786677",
    "size": 39379,
    "digest": "67267fcdf78535a75854645de60e7ad662aaa5016788b62e9901c1f073e9b016",
}
MODELS = {
    "B4_half": "c795f8b5f6bf8e9c37d195126fc25006dbde92fb71591473c05854e3d3698f15",
    "B5_control": "0b0f09f4db86e9b10a05c194bba56071be46d5031e573a22183b785a3b65659a",
    "B5_early_hey": "784f82b552c411f04631d5adc0664b4f07f305ca45d551a1de8352e844ee440d",
}
GAINS = ("1.0", "0.75", "0.5")
HITS = {
    "1.0": {"B4_half": 5, "B5_control": 5, "B5_early_hey": 5},
    "0.75": {"B4_half": 5, "B5_control": 5, "B5_early_hey": 5},
    "0.5": {"B4_half": 3, "B5_control": 3, "B5_early_hey": 2},
}
MISSING_MEH_HALF = {"B4_half": 0, "B5_control": 0, "B5_early_hey": 1}
RESULT_SCHEMA = "english-v6-b5-dev504-three-frozen-new-source-original-paired-v1"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def source_check():
    need(len(ORIGINAL["digest"]) == 64
         and all(re.fullmatch("[0-9a-f]{64}", value) for value in MODELS.values())
         and ORIGINAL["run"] == 44700786677
         and ORIGINAL["head"] == PARENT
         and sum(HITS["0.5"].values()) == 8
         and MISSING_MEH_HALF["B5_early_hey"] == 1,
         "ORIGINAL_B5_SAVED_RESULT_AND_WEIGHT_LOCK")
    return {
        "status": "ONLY_PREVIOUSLY_SAVED_B5_RESULT",
        "original_result_id": ORIGINAL["id"],
        "run_id": ORIGINAL["run"],
        "original_forward_cases": 117,
        "original_positive": 5,
        "original_ordinary": 8,
        "excluded_unknown": 19,
        "independent_qualified_dev_groups": 3,
        "new_model_forwards": 0,
        "new_audio_calls": 0,
        "new_optimizer_updates": 0,
        "cal_locked_access": False,
        "threshold_selection_allowed": False,
        "model_selection_allowed": False,
        "product_promotion_allowed": False,
    }


def guard(env=None, event=None, parent=None):
    source_check()
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ATTEMPT": "1"}.items()),
        "FIRST_PUBLIC_MAIN_SAVED_ONLY")
    head = e.get("GITHUB_SHA", "")
    need(re.fullmatch("[0-9a-f]{40}", head) is not None
         and re.fullmatch("[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")) is not None,
         "EXACT_NEW_SAVED_RESEARCH_RUN")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(
            ["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == PARENT and event.get("after") == head
         and event.get("ref") == "refs/heads/main"
         and event.get("repository", {}).get("full_name") == REPO
         and event.get("repository", {}).get("private") is False
         and event.get("forced") is False and event.get("deleted") is False
         and parent == PARENT,
         "PREVIOUS_SUCCESSFUL_117_FORWARD_MAIN_SUCCESSOR_ONLY")
    return {"run": e["GITHUB_RUN_ID"], "head": head, "before": PARENT}


def verify_remote(token):
    need(type(token) is str and len(token) >= 10, "READ_ONLY_ACTIONS_TOKEN")
    def get(path):
        req = urllib.request.Request(
            "https://api.github.com/repos/" + REPO + path,
            headers={"Authorization": "Bearer " + token,
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "kws-b5-saved-only-source-audit"})
        with urllib.request.urlopen(req, timeout=40) as f:
            raw = f.read(1000001)
        need(len(raw) <= 1000000, "BOUNDED_GITHUB_RUN_METADATA")
        return json.loads(raw)
    run = get("/actions/runs/" + str(ORIGINAL["run"]))
    need(run.get("id") == ORIGINAL["run"]
         and run.get("head_sha") == ORIGINAL["head"]
         and run.get("event") == "push"
         and run.get("run_attempt") == 1
         and run.get("status") == "completed"
         and run.get("conclusion") == "success",
         "REAL_FROZEN_THREE_WAY_INFERENCE_PASSED")
    a = get("/actions/artifacts/" + str(ORIGINAL["id"]))
    need(a.get("id") == ORIGINAL["id"]
         and a.get("name") == ORIGINAL["name"]
         and a.get("size_in_bytes") == ORIGINAL["size"]
         and a.get("digest") == "sha256:" + ORIGINAL["digest"]
         and a.get("expired") is False
         and a.get("workflow_run", {}).get("id") == ORIGINAL["run"]
         and a.get("workflow_run", {}).get("head_sha") == ORIGINAL["head"],
         "EXACT_IMMUTABLE_117_FORWARD_ARTIFACT")
    return {"run": ORIGINAL["run"], "artifact": ORIGINAL["id"],
            "zip_sha256": ORIGINAL["digest"]}


def analyze(result):
    need(result.get("schema") == RESULT_SCHEMA
         and result.get("status") == "B5_NEW_SOURCE_FROZEN_THREE_WAY_DEV_DIAGNOSTIC_ONLY"
         and result.get("model_pt_sha256") == MODELS
         and result.get("B4_half_B5_control_identical_tensors") is True
         and result.get("qualified_positive_source_groups") == 3
         and result.get("same_tts_engine_not_human_gold") is True
         and result.get("cal_locked_opened") is False
         and result.get("model_selection_allowed") is False
         and result.get("product_promotion_allowed") is False
         and result.get("new_optimizer_steps") == 0
         and result.get("new_tts_asr_calls") == 0
         and result.get("derived_conditions_do_not_increase_original_denominator") is True,
         "ORIGINAL_NONPROMOTABLE_RESEARCH_117_RUN_CONTRACT")
    rows = result.get("per_original")
    need(type(rows) is list and len(rows) == 13
         and Counter(x.get("positive") for x in rows) == {True: 5, False: 8}
         and len({x.get("id") for x in rows}) == 13
         and len({x.get("pcm_sha256") for x in rows}) == 13
         and all(x.get("role") == "development"
                 and x.get("machine_tier") in ("silver", "weak")
                 and re.fullmatch("[0-9a-f]{64}", x.get("pcm_sha256", ""))
                 for x in rows),
         "EXACT_13_NONLEAKING_INDEPENDENT_ORIGINAL_DEV_PCM")
    summary = result.get("summary")
    need(type(summary) is dict and set(summary) == set(GAINS),
         "THREE_FROZEN_GAIN_CONDITIONS_ONLY")
    positives = [x for x in rows if x["positive"]]
    ordinary = [x for x in rows if not x["positive"]]
    failures = []
    by_gain = {}
    for gain in GAINS:
        need(set(summary[gain]) == set(MODELS), "EXACT_THREE_TRAINED_MODEL_LABELS")
        for row in rows:
            cond = row.get("conditions", {}).get(gain)
            need(type(cond) is dict and set(cond) == set(MODELS),
                 "EVERY_ORIGINAL_FULL_THREE_MODEL_GAIN_CONDITION")
            for name, entry in cond.items():
                n = entry.get("full_phrase_event_count")
                need(type(n) is int and n >= 0
                     and entry.get("any_complete_event") is (n > 0)
                     and type(entry.get("any_meh_argmax")) is bool
                     and entry.get("first_frame_argmax") in (0, 1, 2, 3)
                     and type(entry.get("collapsed_tokens")) is list
                     and type(entry.get("frames")) is int and entry["frames"] > 0,
                     "REAL_CTC_GREEDY_PATH_FROM_ORIGINAL_FRAME_FORWARDS")
            need(cond["B4_half"] == cond["B5_control"],
                 "EXACT_B4_HALF_B5_CONTROL_PER_ORIGINAL_TENSOR_REPLAY")
        by_gain[gain] = {}
        for model in MODELS:
            p = [x for x in positives
                 if x["conditions"][gain][model]["any_complete_event"]]
            neg_events = sum(x["conditions"][gain][model]["full_phrase_event_count"]
                             for x in ordinary)
            missing_meh = sum(
                not x["conditions"][gain][model]["any_meh_argmax"]
                for x in positives)
            record = summary[gain][model]
            need(record["positive_hits"] == HITS[gain][model] == len(p)
                 and record["positive_originals"] == 5
                 and record["ordinary_originals"] == 8
                 and record["ordinary_complete_event_clips"] == 0
                 and record["ordinary_false_or_extra_events"] == neg_events == 0
                 and record["positive_missing_meh_argmax"] == missing_meh,
                 "ORIGINAL_NUMERIC_SUMMARY_RECALCULATION")
            if gain == "0.5":
                need(missing_meh == MISSING_MEH_HALF[model],
                     "HALF_GAIN_MEH_CTC_PATH_CONTRACT")
            by_gain[gain][model] = {
                "positive_hits": len(p), "positive_originals": 5,
                "ordinary_false_or_extra_events": neg_events,
                "positive_missing_meh_argmax": missing_meh,
            }
    for row in positives:
        b4 = row["conditions"]["0.5"]["B4_half"]
        b5 = row["conditions"]["0.5"]["B5_early_hey"]
        if b4["any_complete_event"] != b5["any_complete_event"]:
            failures.append({
                "source_id": row["id"],
                "source_group_id": row["source_group_id"],
                "pcm_sha256": row["pcm_sha256"],
                "B4_half_hit": b4["any_complete_event"],
                "B5_early_hey_hit": b5["any_complete_event"],
                "B4_half_ctc": b4["collapsed_tokens"],
                "B5_early_hey_ctc": b5["collapsed_tokens"],
                "B5_has_meh_argmax": b5["any_meh_argmax"],
            })
    need(sum(int(not x["B5_early_hey_hit"]) - int(not x["B4_half_hit"])
             for x in failures) == 1,
         "B5_HALFGAIN_NET_ONE_MORE_POSITIVE_MISS")
    return {
        "schema": SCHEMA,
        "status": "PASS_IMMUTABLE_B5_LOWGAIN_CTC_PATHOLOGY_DIAGNOSTIC_ONLY",
        "source_artifact": ORIGINAL["id"], "original_saved_inference": 117,
        "model_names": list(MODELS), "by_gain": by_gain,
        "half_gain_discordant_positive_origins": failures,
        "half_gain_B5_net_new_positive_misses": 1,
        "original_positive_pcm": 5, "original_ordinary_pcm": 8,
        "original_qualified_development_groups": 3,
        "gain_variants_not_independent_samples": True,
        "half_gain_frame_logits_saved": False,
        "startup_or_phone_timing_cause_proven": False,
        "actual_3_5m_human_frr_measured": False,
        "natural_far_per_hour_measured": False,
        "physical_ssc305_measured": False,
        "new_model_forwards": 0, "new_asr_phone_tts_calls": 0,
        "new_optimizer_updates": 0, "training_allowed": False,
        "model_selection_allowed": False,
        "product_promotion_allowed": False,
    }


def evaluate(archive, out):
    source_check()
    receipt = guard()
    output = Path(out)
    need(output.is_absolute() and not output.exists()
         and output.parent.is_dir(), "ONE_NEW_SAVED_PROOF_DIRECTORY")
    output.mkdir()
    save(output / "STARTED.json", {"schema": SCHEMA, "run": receipt,
          "old_artifact": ORIGINAL["id"], "new_model_forwards": 0})
    try:
        verified = verify_remote(os.environ.get("GH_TOKEN"))
        archive = Path(archive)
        need(archive.is_file() and not archive.is_symlink()
             and archive.stat().st_size == ORIGINAL["size"]
             and sha(archive.read_bytes()) == ORIGINAL["digest"],
             "EXACT_FROZEN_117_FORWARD_ZIP_DIGEST")
        with zipfile.ZipFile(archive) as z:
            need(z.testzip() is None, "ORIGINAL_ZIP_INTEGRITY")
            result = json.loads(z.read("RESULT.json"))
            logits = z.read("ORIGINAL_FRAME_LOGITS.npz")
            need(sha(logits) == result.get("original_frame_logits_sha256"),
                 "ORIGINAL_FULL_FRAME_GAIN1_LOGIT_SHA")
        derived = analyze(result)
        derived["authenticated_input"] = verified
        derived["readback_run"] = receipt
        save(output / "FINDINGS.json", derived)
        print(json.dumps({
            "status": derived["status"],
            "half_gain_regressed_originals": len(derived["half_gain_discordant_positive_origins"]),
            "half_gain_net_new_positive_misses": derived["half_gain_B5_net_new_positive_misses"],
            "by_gain": derived["by_gain"],
            "new_model_forwards": 0}, sort_keys=True))
    except BaseException as exc:
        save(output / "STOP.json", {"schema": SCHEMA,
             "status": "STOP_AUTHENTIC_B5_SAVED_RESULT",
             "type": type(exc).__name__, "reason": str(exc)[:450],
             "new_model_forwards": 0, "product_promotion_allowed": False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-check", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(args.source_check != args.evaluate, "EXACT_ONE_SOURCE_OR_SAVED_READBACK")
    if args.source_check:
        print(json.dumps(source_check(), sort_keys=True)); return
    need(args.archive is not None and args.output is not None,
         "SAVED_RESULT_ARCHIVE_AND_OUTPUT_REQUIRED")
    evaluate(args.archive, args.output)


if __name__ == "__main__":
    main()
