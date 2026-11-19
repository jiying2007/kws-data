"""One-shot evidence join for saved 2026-11-19 English p424..p447 PCM.

No TTS, ASR, phone, KWS optimizer, calibration, locked-test replay, or
device inference. The old native reviewer remains byte-for-byte immutable.
Only independently observed original full-PCM evidence creates machine tiers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "english_newtrain_readback_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observer = load("english_newtrain24_five_native_once")
prior_join = load("english_piper_fresh24_combine_once")
policy = load("english_fresh24_ctc_admission")
c = load("english_hifi_micro")

# Reuse the previously tested immutable complete-PCM packet reader; its
# identity checks are parameterized here only by the new pinned observer.
# The original module on disk is NOT edited, including its old source gate.
prior_join.worker = observer
SCHEMA = "english-newtrain24-saved-five-source-screen-v1"
BEFORE = "9fd28b29ff099156ed544da862f66782ee4bcb2f"
OBSERVER_RUN = 44433435970
OBSERVER_SHA = BEFORE
PRIOR_PACKET_BLOB = "d3cd2e5779aba7ad271c83eb612e2aed9133e538"
FROZEN_POLICY_BLOB = "0a688325d0d36e83ef6623ac678511df24a92374"
ARTIFACTS = {
    "qwen": (14488111130, 159193),
    "whisper": (14488575666, 160197),
    "parakeet": (14488110800, 159512),
    "xlsr": (14486314438, 8311269),
    "lv60": (14486424306, 8312308),
}
PRIOR_COUNTS = {"train": 8, "development": 4, "calibration": 3,
                "locked-test": 3}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def source_check():
    result = observer.source_check()
    plan = observer.source_worker.protocol()
    need(prior_join.git_blob(HERE / "english_piper_fresh24_combine_once.py")
         == PRIOR_PACKET_BLOB and
         prior_join.git_blob(HERE / "english_fresh24_ctc_admission.py")
         == FROZEN_POLICY_BLOB and
         result["status"] == "SOURCE_PINNED" and
         observer.SOURCE_RUN == 44431769603 and
         observer.SOURCE["id"] == 14486048125 and
         observer.SOURCE_COUNT == 168 and
         tuple(observer.PROVIDERS) ==
         tuple(policy.TEXT_MODELS) + tuple(policy.PHONE_MODELS) and
         len(ARTIFACTS) == len(observer.PROVIDERS) == 5 and
         len({x[0] for x in ARTIFACTS.values()}) == 5 and
         [g["role"] for g in plan["groups"]] ==
         ["train"] * 18 + ["development"] * 6 and
         set(g["speaker_index"] for g in plan["groups"]) ==
         set(range(424, 448)) and
         plan["already_qualified_readonly"] == PRIOR_COUNTS and
         plan["frozen_phone_contrast_policy_blob"] ==
         FROZEN_POLICY_BLOB and
         policy.MIN_QUALIFIED == {"train": 6, "development": 2,
                                  "calibration": 2, "locked-test": 2},
         "EXACT_PINNED_NEWTRAIN_SOURCE_FIVE_MODELS_AND_FROZEN_PHONE_POLICY")
    return {"status": "PINNED_NO_NEW_MODEL_CALLS",
            "source_run": observer.SOURCE_RUN,
            "five_observer_run": OBSERVER_RUN,
            "source_rows": 168, "speaker_roles": {"train": 18, "development": 6},
            "model_forwards_in_this_stage": 0,
            "training_allowed": False}


def guard(env=None, event=None, parent=None):
    source_check()
    env = os.environ if env is None else env
    need(all(env.get(k) == v for k, v in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": observer.REPO,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ATTEMPT": "1",
    }.items()), "ONE_PUBLIC_MAIN_EVENT_ONLY")
    sha = env.get("GITHUB_SHA", "")
    need(bool(re.fullmatch(r"[a-f0-9]{40}", sha)) and
         bool(re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", ""))),
         "SAVED_REVIEW_MAIN_RUN_IDENTITY_REQUIRED")
    if event is None:
        event = json.loads(c.read(env["GITHUB_EVENT_PATH"]))
    need(event.get("before") == BEFORE and event.get("after") == sha and
         event.get("ref") == "refs/heads/main" and
         event.get("repository", {}).get("full_name") == observer.REPO and
         event.get("repository", {}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "EXACT_UNREPEATED_READBACK_MAIN_SUCCESSOR")
    if parent is None:
        parent = subprocess.check_output(
            ["git", "rev-parse", "HEAD^1"], cwd=ROOT, text=True).strip()
    need(parent == BEFORE, "NO_HISTORY_REWRITE_OR_RERUN")
    return {"run_id": env["GITHUB_RUN_ID"], "commit": sha,
            "observer_run": OBSERVER_RUN, "new_model_forwards": 0}


def original_observer_run():
    return {"run_id": str(OBSERVER_RUN), "commit": OBSERVER_SHA,
            "source_run": observer.SOURCE_RUN,
            "source_commit": observer.SOURCE_COMMIT}


def verify_github(token):
    """Bind every stored observer ZIP to the original protected-main run."""
    need(type(token) is str and len(token) >= 10, "ACTIONS_READONLY_TOKEN")
    def get(path):
        request = urllib.request.Request(
            "https://api.github.com/repos/" + observer.REPO + path,
            headers={"Authorization": "Bearer " + token,
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "kws-english-newtrain24-saved-readback"})
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read(1000001)
        need(len(raw) <= 1000000, "BOUNDED_GITHUB_METADATA")
        return json.loads(raw)
    run = get("/actions/runs/" + str(OBSERVER_RUN))
    need(run.get("id") == OBSERVER_RUN and
         run.get("head_sha") == OBSERVER_SHA and
         run.get("head_branch") == "main" and
         run.get("event") == "push" and
         run.get("run_attempt") == 1 and
         run.get("status") == "completed" and
         run.get("conclusion") == "success",
         "ORIGINAL_FIVE_NATIVE_PUBLIC_MAIN_RUN_NOT_SUCCESSFUL")
    source = get("/actions/artifacts/" + str(observer.SOURCE["id"]))
    need(source.get("id") == observer.SOURCE["id"] and
         source.get("name") == observer.SOURCE["name"] and
         source.get("size_in_bytes") == observer.SOURCE["size"] and
         source.get("digest") == observer.SOURCE["digest"] and
         source.get("expired") is False and
         source.get("workflow_run", {}).get("id") == observer.SOURCE_RUN and
         source.get("workflow_run", {}).get("head_sha") ==
         observer.SOURCE_COMMIT, "EXACT_ORIGINAL_SOURCE_ZIP_IDENTITY")
    archived = {}
    for provider in observer.PROVIDERS:
        artifact_id, archive_bytes = ARTIFACTS[provider]
        record = get("/actions/artifacts/" + str(artifact_id))
        digest = record.get("digest")
        need(record.get("id") == artifact_id and
             record.get("name") == "newtrain24-observer-" + provider and
             record.get("size_in_bytes") == archive_bytes and
             record.get("expired") is False and
             record.get("workflow_run", {}).get("id") == OBSERVER_RUN and
             record.get("workflow_run", {}).get("head_sha") == OBSERVER_SHA and
             isinstance(digest, str) and
             bool(re.fullmatch(r"sha256:[a-f0-9]{64}", digest)),
             "SAVED_NATIVE_PROVIDER_ARTIFACT_NOT_IMMUTABLE")
        archived[provider] = {"id": artifact_id, "bytes": archive_bytes,
                              "digest": digest}
    return archived


def evaluate(rows, transcripts, phone_evidence):
    expected = {x["id"] for x in rows}
    need(len(rows) == 168 and
         set(transcripts) == set(phone_evidence) == expected,
         "EXACT_SOURCE_DENOMINATOR_AND_FIVE_MODEL_KEYS")
    groups = defaultdict(list)
    decisions = []
    roles = {}
    for row in rows:
        gid = row["source_group_id"]
        need(gid == "piper-libritts-" + row["source_speaker_id"] and
             row["role"] in ("train", "development") and
             (gid not in roles or roles[gid] == row["role"]),
             "ORIGINAL_SOURCE_GROUP_AND_PREAUDIO_SPLIT_ONLY")
        roles[gid] = row["role"]
        verdict = policy.decide({**row, "pcm_samples": row["samples"]},
                                transcripts[row["id"]],
                                phone_evidence[row["id"]])
        if verdict["tier"] in ("silver", "weak") and (
                verdict["class"] != row["requested_class"]):
            verdict = {"tier": "unknown", "class": None,
                       "reason": "REQUEST_ACOUSTIC_CONTRADICTION",
                       "margins": {}, "training_allowed": False}
        d = {"id": row["id"], "source_group_id": gid,
             "source_speaker_id": row["source_speaker_id"],
             "speaker_index": row["speaker_index"],
             "role": row["role"], "cell_id": row["cell_id"],
             "requested_class": row["requested_class"],
             "observed_class": verdict["class"], "tier": verdict["tier"],
             "reason": verdict["reason"],
             "phone_margins": verdict["margins"],
             "pcm_sha256": row["pcm_sha256"],
             "wav_sha256": row["wav_sha256"],
             "samples": row["samples"],
             "training_allowed": False, "human_gold": False}
        decisions.append(d)
        groups[gid].append(d)
    need(len(groups) == 24 and all(len(v) == 7 for v in groups.values()) and
         len({x["source_speaker_id"] for x in rows}) == 24 and
         {x["speaker_index"] for x in rows} == set(range(424, 448)) and
         Counter(roles.values()) == {"train": 18, "development": 6},
         "EXACT_NEW_UNEXPOSED_18_TRAIN_6_DEV_GROUPS")
    qualified = {gid for gid, items in groups.items()
                 if {x["observed_class"] for x in items
                     if x["tier"] in ("silver", "weak")} == {0, 1, 2}}
    new_groups = {role: sum(roles[gid] == role for gid in qualified)
                  for role in ("train", "development")}
    eligible = [x for x in decisions
                if x["source_group_id"] in qualified and
                x["tier"] in ("silver", "weak") and
                x["observed_class"] in (0, 1, 2)]
    by_role = Counter(x["role"] for x in eligible)
    status = ("PASS_NEWTRAIN_MACHINE_TRAIN_DEV_SOURCE_SCREEN"
              if new_groups["train"] > 0 and new_groups["development"] > 0
              else "STOP_NEWTRAIN_INSUFFICIENT_INDEPENDENT_GROUPS")
    return {"schema": SCHEMA, "status": status,
            "new_groups": new_groups,
            "historical_group_counts_readonly": PRIOR_COUNTS,
            "previous_locked_test_already_exposed": True,
            "new_calibration_and_locked_sources_required": True,
            "source_groups_disjoint": True,
            "tier_counts": dict(Counter(x["tier"] for x in decisions)),
            "reason_counts": dict(Counter(x["reason"] for x in decisions)),
            "eligible_train_dev_counts": {role: by_role[role]
                                          for role in ("train", "development")},
            "candidate_rows": eligible, "rows": decisions,
            "training_snapshot_allowed": False,
            "new_kws_optimizer_updates": 0,
            "new_tts_asr_ctc_forwards": 0,
            "human_gold": False, "ssc305_measured": False}


def combine(source, observer_root, output):
    gate = source_check()
    need(gate["status"] == "PINNED_NO_NEW_MODEL_CALLS",
         "FROZEN_OBSERVER_SOURCE_REQUIRED")
    current = guard()
    dest = Path(output)
    need(dest.is_absolute() and not dest.exists(), "NEW_IMMUTABLE_REVIEW_OUTPUT")
    dest.mkdir(parents=True)
    c.record(dest / "STARTED.json", {
        "schema": SCHEMA, "current_run": current,
        "source_artifact": observer.SOURCE, "original_observer_run": OBSERVER_RUN,
        "observer_artifacts": ARTIFACTS, "training_allowed": False})
    try:
        archived = verify_github(os.environ.get("GH_TOKEN"))
        source_rows = observer.source_rows(source)
        texts = {x["id"]: {} for x in source_rows}
        phones = {x["id"]: {} for x in source_rows}
        reports = []
        for provider in observer.PROVIDERS:
            saved, report = prior_join.packet(observer_root, provider,
                                             source_rows, original_observer_run())
            reports.append(report)
            for sid, evidence in saved.items():
                if provider in policy.PHONE_MODELS:
                    phones[sid][provider] = evidence
                else:
                    texts[sid][provider] = evidence
        all_complete = all(r["complete"] == 168 and r["status"] == "COMPLETE"
                           for r in reports)
        result = evaluate(source_rows, texts, phones)
        status = result["status"] if all_complete else "STOP_NATIVE_SOURCE_INCOMPLETE"
        c.record(dest / "SCREEN.json", {
            "schema": SCHEMA, "status": status,
            "current_run": current, "original_observer_run": OBSERVER_RUN,
            "source_artifact": observer.SOURCE,
            "observer_artifacts": archived, "observer_reports": reports,
            "all_five_native_observers_complete": all_complete,
            "frozen_policy_git_blob_sha1": FROZEN_POLICY_BLOB,
            "reused_original_packet_reader_blob": PRIOR_PACKET_BLOB,
            "machine_review": result,
            "training_snapshot_allowed": False,
            "locked_kws_evaluation_consumed": False,
            "new_tts_asr_ctc_forwards": 0, "new_optimizer_updates": 0,
            "human_gold": False, "ssc305_measured": False})
        print(json.dumps({
            "status": status, "all_five_native_observers_complete": all_complete,
            "new_groups": result["new_groups"],
            "eligible_train_dev_counts": result["eligible_train_dev_counts"],
            "tier_counts": result["tier_counts"], "training_allowed": False,
        }, sort_keys=True))
    except BaseException as exc:
        c.record(dest / "STOP.json", {
            "schema": SCHEMA, "status": "STOP_READBACK_FAILED",
            "exception": type(exc).__name__,
            "reason": str(exc)[:450], "original_observer_run": OBSERVER_RUN,
            "new_tts_asr_ctc_forwards": 0, "new_optimizer_updates": 0})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-check", action="store_true")
    p.add_argument("--guard", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--source")
    p.add_argument("--observers")
    p.add_argument("--output")
    args = p.parse_args()
    need(sum((args.source_check, args.guard, args.combine)) == 1,
         "ONE_READBACK_STAGE_ONLY")
    if args.source_check:
        print(json.dumps(source_check(), sort_keys=True))
    elif args.guard:
        print(json.dumps(guard(), sort_keys=True))
    else:
        need(all((args.source, args.observers, args.output)),
             "ORIGINAL_SOURCE_AND_SAVED_FIVE_ARTIFACTS_REQUIRED")
        combine(args.source, args.observers, args.output)


if __name__ == "__main__":
    main()
