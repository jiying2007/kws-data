"""One new actual English canonical research head on authenticated 8+3 new rows.

Original historical 63 Silver/Weak records remain immutable (28 train,
13 development, 11 old calibration, 11 OLD EXPOSED locked). Open ONLY the
historical train/development WAVs and the new 8/3 train/development WAVs.
Never infer on a historical held-out, select a threshold from its scores,
rerun TTS/ASR/CTC, or claim a field/SSC305 qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "english_newtrain_real_head_" + name, HERE / (name + ".py"))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


binding = load("english_canonical_multisource_bind_once")
base = load("english_canonical_multisource_fit_once")
v2 = load("english_canonical_dev_only_v2_once")
readback = load("english_newtrain24_evidence_readback")
source_worker = readback.observer.source_worker
SCHEMA = "english-canonical-newtrain24-source-extended-dev-only-v1"
REPO = "jiying2007/kws-data"
BEFORE = "cc746197ce470209ae9dadfae47c7f8c7c01e38f"
SOURCE_ID = 14486048125
SOURCE_RUN = 44431769603
SOURCE_HEAD = "9113713ad4fcb87e111f373db35922f61775de1e"
SCREEN_ID = 14489594100
SCREEN_RUN = 44439791964
SCREEN_HEAD = BEFORE
SOURCE = {"id": SOURCE_ID,
          "name": "english-newtrain-piper-source-44431769603",
          "bytes": 3353181,
          "sha256": "3db5e99caf33e04ae920df1c93205f38f98cbc0f326705e2367dc8c67dc02341"}
SCREEN = {"id": SCREEN_ID,
          "name": "english-newtrain24-saved-native-screen-44439791964",
          "bytes": 20256,
          "sha256": "cebb45ffa01a9be5b57353c25473d99bb2d5c9ea2166037d85a44b43ca432689"}
SEED = 20261123
UPDATES = 96
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
THRESHOLD = 0.8  # fixed *research development* operating point, not calibrated
NEW_TRAIN = 8
NEW_DEV = 3
OLD_TRAIN = 28
OLD_DEV = 13


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_check():
    cfg = binding.config()
    old = v2.source_check()
    new = readback.source_check()
    need(cfg["exact_expected_qualified"] ==
         {"train": 8, "development": 4, "calibration": 3, "locked-test": 3}
         and old["source_groups"] == cfg["exact_expected_qualified"]
         and new["source_run"] == SOURCE_RUN
         and new["five_observer_run"] == 44433435970
         and cfg["no_legacy_meero_weights"] is True
         and cfg["no_legacy_meero_positive_audio"] is True
         and base.SHAPES["w1"] == (32, 1536)
         and base.MAGIC == b"EHKWS01\x00"
         and source_worker.SCHEMA == "english-piper-newtrain24-original-source-v1"
         and source_worker.source_check()["voice_groups"] == 24
         and source_worker.protocol()["brand_ipa"] ==
         {"Mero": "/heɪ ˈmɛroʊ/", "Vosen": "/heɪ ˈvoʊsən/"}
         and readback.ARTIFACTS["xlsr"][0] == 14486314438
         and readback.ARTIFACTS["lv60"][0] == 14486424306
         and len({SOURCE_ID, SCREEN_ID}) == 2
         and UPDATES == 96 and SEED == 20261123
         and LEARNING_RATE == cfg["head"]["learning_rate"]
         and WEIGHT_DECAY == cfg["head"]["weight_decay"]
         and THRESHOLD == 0.8
         and (OLD_TRAIN, OLD_DEV, NEW_TRAIN, NEW_DEV) ==
         (28, 13, 8, 3),
         "REAL_FIRST_96_STEP_NEW_SOURCE_ONLY_NO_LOCKED_REUSE")
    return {"status": "READY_ONLY_SOURCE_VALIDATION",
            "old_train_dev": [OLD_TRAIN, OLD_DEV],
            "new_train_dev": [NEW_TRAIN, NEW_DEV],
            "new_piper_source_artifact": SOURCE_ID,
            "new_five_native_screen_artifact": SCREEN_ID,
            "seed": SEED, "updates": UPDATES,
            "predefined_dev_threshold": THRESHOLD,
            "full_locked_evaluation": "DENIED",
            "new_model_forwards_in_pr": 0}


def guard(env=None, event=None, parent=None):
    source_check()
    env = os.environ if env is None else env
    need(all(env.get(k) == v for k, v in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
        "GITHUB_RUN_ATTEMPT": "1"}.items()), "ONLY_ONE_PROTECTED_PUBLIC_MAIN_RUN")
    head = env.get("GITHUB_SHA", "")
    need(bool(re.fullmatch(r"[a-f0-9]{40}", head)) and
         bool(re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", ""))),
         "FIRST_RUN_ID_AND_ACTUAL_SHA")
    if event is None:
        event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    need(event.get("before") == BEFORE and event.get("after") == head and
         event.get("ref") == "refs/heads/main" and
         event.get("repository", {}).get("full_name") == REPO and
         event.get("repository", {}).get("private") is False and
         event.get("forced") is False and event.get("deleted") is False,
         "EXPECTED_FIRST_NEW_SOURCE_NUMERIC_MAIN_SUCCESSOR")
    if parent is None:
        parent = subprocess.check_output(
            ["git", "rev-parse", "HEAD^1"], cwd=ROOT, text=True).strip()
    need(parent == BEFORE, "NO_MERGE_PARENT_DRIFT_OR_RUN_REPLAY")
    return {"run_id": env["GITHUB_RUN_ID"], "head_sha": head,
            "before": BEFORE, "newtrain_screen": SCREEN_ID}


def verified_remote(token):
    """Authenticate source and 47-tier source-screen ZIPs to actual Actions."""
    need(type(token) is str and len(token) >= 10, "READ_ONLY_GITHUB_TOKEN")
    def github(path):
        request = urllib.request.Request(
            "https://api.github.com/repos/" + REPO + "/actions/" + path,
            headers={"Authorization": "Bearer " + token,
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "kws-first-newtrain24-real-96-fit"})
        with urllib.request.urlopen(request, timeout=40) as handle:
            raw = handle.read(1_000_001)
        need(len(raw) <= 1_000_000, "SMALL_GITHUB_IDENTITY_RECEIPT")
        return json.loads(raw)
    for record, run_id, head in (
        (SOURCE, SOURCE_RUN, SOURCE_HEAD),
        (SCREEN, SCREEN_RUN, SCREEN_HEAD),
    ):
        x = github("artifacts/" + str(record["id"]))
        need(x.get("id") == record["id"] and
             x.get("name") == record["name"] and
             x.get("size_in_bytes") == record["bytes"] and
             x.get("digest") == "sha256:" + record["sha256"] and
             x.get("expired") is False and
             x.get("workflow_run", {}).get("id") == run_id and
             x.get("workflow_run", {}).get("head_sha") == head,
             "ORIGINAL_PIPER_SOURCE_AND_SCREEN_TRUST_ANCHORS")
        run = github("runs/" + str(run_id))
        need(run.get("id") == run_id and
             run.get("head_sha") == head and
             run.get("run_attempt") == 1 and
             run.get("event") == "push" and
             run.get("head_branch") == "main" and
             run.get("status") == "completed" and
             run.get("conclusion") == "success",
             "ONLY_ORIGINAL_FINISHED_PROTECTED_MAIN_ARTIFACTS")
    return {"source_artifact_id": SOURCE_ID, "screen_artifact_id": SCREEN_ID,
            "all_previous_numeric_observers_ran_once": True}


def bind_new(source_zip, screen_zip):
    """Read all 168 new WAV hashes and original 47 tiers; no new inference."""
    z = binding.archive(source_zip, SOURCE)
    screen = binding.archive(screen_zip, SCREEN)
    try:
        manifest = json.loads(z.read("MANIFEST.json"))
        lock = json.loads(z.read("MODEL_LOCK.json"))
        report = json.loads(screen.read("SCREEN.json"))
        review = report.get("machine_review", {})
        speakers = {int(k): v for k, v in
                    manifest.get("speakers_by_index", {}).items()}
        expected = source_worker.planned(speakers)
        need(manifest.get("schema") == source_worker.SCHEMA and
             manifest.get("run") == readback.observer.original_identity() and
             manifest.get("completed") == 168 and
             manifest.get("denominator") == 168 and
             manifest.get("human_gold") is False and
             manifest.get("training_allowed") is False and
             lock.get("model") == source_worker.protocol()["model"] and
             len(manifest.get("rows", [])) == 168 and
             set(speakers) == set(range(424, 448)) and
             len(set(speakers.values())) == 24 and
             len(expected) == 168,
             "ORIGINAL_NEW_PIPER168_PCM_AND_SOURCE_ANCHOR")
        need(report.get("schema") == readback.SCHEMA and
             report.get("status") == "PASS_NEWTRAIN_MACHINE_TRAIN_DEV_SOURCE_SCREEN" and
             report.get("original_observer_run") == 44433435970 and
             report.get("current_run", {}).get("commit") == SCREEN_HEAD and
             report.get("source_artifact") == readback.observer.SOURCE and
             report.get("all_five_native_observers_complete") is True and
             report.get("training_snapshot_allowed") is False and
             report.get("locked_kws_evaluation_consumed") is False and
             report.get("new_tts_asr_ctc_forwards") == 0 and
             review.get("status") == report["status"] and
             review.get("new_groups") == {"train": 2, "development": 1} and
             review.get("tier_counts") ==
             {"silver": 43, "weak": 4, "unknown": 121} and
             review.get("eligible_train_dev_counts") ==
             {"train": NEW_TRAIN, "development": NEW_DEV} and
             review.get("training_snapshot_allowed") is False and
             len(review.get("rows", [])) == 168 and
             len(review.get("candidate_rows", [])) == NEW_TRAIN + NEW_DEV,
             "IMMUTABLE_REAL_FIVE_MODEL_47_TIER_REVIEW_AND_8_3_SCREEN")
        originals = {r["id"]: r for r in manifest["rows"]}
        verdicts = {r["id"]: r for r in review["rows"]}
        chosen = {r["id"]: r for r in review["candidate_rows"]}
        need(len(originals) == len(verdicts) == 168 and
             set(originals) == set(verdicts) and
             len(chosen) == 11 and set(chosen) <= set(originals),
             "EXACT_COMPLETE_NEWTRAIN_SCREEN_NO_DUPLICATE_IDS")
        used_pcm = set()
        roles = {}
        observed_groups = defaultdict(set)
        selected = []
        for actual, request in zip(manifest["rows"], expected):
            ident = request["id"]
            seen = verdicts[ident]
            need(actual["id"] == ident and
                 all(actual.get(k) == value
                     for k, value in request.items()) and
                 actual.get("observed_class") is None and
                 actual.get("training_allowed") is False and
                 actual.get("human_gold") is False and
                 actual["source_group_id"] ==
                 "piper-libritts-" + actual["source_speaker_id"] and
                 actual["role"] in ("train", "development") and
                 (actual["source_group_id"] not in roles or
                  roles[actual["source_group_id"]] == actual["role"]) and
                 seen["source_group_id"] == actual["source_group_id"] and
                 seen["role"] == actual["role"] and
                 seen["cell_id"] == actual["cell_id"] and
                 seen["requested_class"] == actual["requested_class"] and
                 seen["samples"] == actual["samples"] and
                 seen["pcm_sha256"] == actual["pcm_sha256"] and
                 seen["wav_sha256"] == actual["wav_sha256"] and
                 seen["training_allowed"] is False and
                 seen["human_gold"] is False,
                 "ORIGINAL_SOURCE_REQUEST_VS_INDEPENDENT_SCREEN_IDENTITY")
            binding.read_wav(z.read("wav/" + ident + ".wav"), actual)
            need(actual["pcm_sha256"] not in used_pcm,
                 "NO_REPEATED_COMPLETE_PCM")
            used_pcm.add(actual["pcm_sha256"])
            group = actual["source_group_id"]
            roles[group] = actual["role"]
            tier = seen["tier"]
            clazz = seen["observed_class"]
            need(tier in ("silver", "weak", "unknown") and
                 (clazz is None if tier == "unknown" else
                  clazz == actual["requested_class"] and
                  clazz in (0, 1, 2)),
                 "UNKNOWN_AND_CONTRAST_CANNOT_BECOME_TRAINING_CLASSES")
            if tier != "unknown":
                observed_groups[group].add(clazz)
            if ident in chosen:
                need(chosen[ident] == seen and
                     tier in ("silver", "weak") and
                     clazz in (0, 1, 2) and
                     group.startswith("piper-libritts-p"),
                     "ONLY_UNMODIFIED_MACHINE_REVIEWED_CANDIDATE_ROWS")
                selected.append({
                    "id": ident, "corpus": "newtrain24",
                    "source_group_id": group, "role": actual["role"],
                    "cell_id": actual["cell_id"], "weak_class": clazz,
                    "tier": tier, "machine_reason": seen["reason"],
                    "source_pcm_sha256": actual["pcm_sha256"],
                    "source_wav_sha256": actual["wav_sha256"],
                    "samples": actual["samples"],
                    "requested_class": actual["requested_class"],
                    "human_gold": False, "training_allowed_in_original": False,
                })
        approved = {g for g, cl in observed_groups.items() if cl == {0, 1, 2}}
        need(len(roles) == 24 and
             Counter(roles.values()) == {"train": 18, "development": 6} and
             Counter(roles[g] for g in approved) ==
             {"train": 2, "development": 1} and
             all(s["source_group_id"] in approved for s in selected) and
             Counter(x["role"] for x in selected) ==
             {"train": NEW_TRAIN, "development": NEW_DEV} and
             {x["weak_class"] for x in selected} == {0, 1, 2},
             "EXACT_NEW_SOURCE_GROUP_ADMISSION_8_TRAIN_3_DEV")
        return selected, {"new_source_groups": {"train": 2, "development": 1},
                          "original_new_source_wavs_hashed": 168,
                          "machine_labels": review["tier_counts"],
                          "new_candidate_rows": len(selected)}
    finally:
        screen.close()
        z.close()


def all_train_dev(sources):
    original = binding.admission(sources, verify_audio=False)
    need(original["status"] == "MACHINE_SILVER_WEAK_RESEARCH_TRAIN_INPUT_BOUND" and
         original["source_group_counts"] ==
         {"train": 8, "development": 4, "calibration": 3, "locked-test": 3} and
         len(original["rows"]) == 63 and
         Counter(x["role"] for x in original["rows"]) ==
         {"train": OLD_TRAIN, "development": OLD_DEV,
          "calibration": 11, "locked-test": 11},
         "AUTHENTICATED_63_OLD_MACHINE_ROWS_BUT_ONLY_TRAIN_DEV_CAN_SCORE")
    new, receipt = bind_new(Path(sources) / "newtrain24-source.zip",
                            Path(sources) / "newtrain24-screen.zip")
    past = [r for r in original["rows"] if r["role"] in ("train", "development")]
    ids = {r["id"] for r in past}
    groups = {r["source_group_id"] for r in past}
    need(not any(x["id"] in ids or x["source_group_id"] in groups
                 for x in new) and
         len({r["source_pcm_sha256"] for r in past + new}) ==
         len(past) + len(new) and
         len(past) + len(new) == 52,
         "NEVER_JOIN_SAME_PCM_OR_SOURCE_GROUP_BETWEEN_OLD_AND_NEW")
    return past + new, receipt


def numeric_train(rows, sources, frontend, out):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    cfg = binding.config()
    spec = cfg["head"]
    progress = {"actual_frontend_forward_calls": 0,
                "optimizer_updates_started": 0,
                "optimizer_updates_completed": 0}
    train_bags, dev_bags, excluded = [], [], []
    for row in sorted(rows, key=lambda x: x["id"]):
        need(row["role"] in ("train", "development"),
             "NO_OLD_CALIBRATION_OR_LOCKED_WAV_EVER_OPENED")
        pcm = base.load_source_pcm(row, sources)
        conditions = ("clean", "snr10", "snr5") if row["role"] == "train" \
                     else ("clean", "snr5")
        for condition in conditions:
            bag, error = base.build_bag(row, pcm, condition, frontend, progress)
            if error:
                excluded.append(error)
            else:
                (train_bags if row["role"] == "train" else dev_bags).append(bag)
    supported = defaultdict(set)
    for b in train_bags + dev_bags:
        supported[b["role"], b["source_group_id"]].add(b["weak_class"])
    approved = {key for key, classes in supported.items()
                if classes == {0, 1, 2}}
    old_new_groups = {"train": 0, "development": 0}
    for role, group in approved:
        if group in {r["source_group_id"] for r in rows
                     if r["corpus"] == "newtrain24"}:
            old_new_groups[role] += 1
    count = Counter(r for r, _ in approved)
    need(count["train"] >= 6 and count["development"] >= 2 and
         old_new_groups["train"] >= 1 and
         old_new_groups["development"] >= 1,
         "TRUE_NEW_SOURCE_FEATURE_EVIDENCE_REQUIRED_NOT_OLD_RETRAIN")
    train_bags = [x for x in train_bags if
                  (x["role"], x["source_group_id"]) in approved]
    dev_bags = [x for x in dev_bags if
                (x["role"], x["source_group_id"]) in approved]
    need(len(train_bags) >= 60 and len(dev_bags) >= 16 and
         all(x["synthetic_context_only"] for x in train_bags + dev_bags),
         "ENOUGH_REAL_CANONICAL_SOURCE_WINDOWS_BEFORE_OPTIMIZER")
    total = sum(len(b["windows"]) * 16 for b in train_bags)
    sum_x = sum(b["windows"].astype(np.float64).sum((0, 1))
                for b in train_bags)
    sum_x2 = sum((b["windows"].astype(np.float64) ** 2).sum((0, 1))
                 for b in train_bags)
    mean = (sum_x / total).astype(np.float32)
    scale = np.maximum(np.sqrt(np.maximum(sum_x2 / total -
                                           (sum_x / total)**2, 0)),
                       1e-5).astype(np.float32)
    need(np.isfinite(mean).all() and np.isfinite(scale).all()
         and (scale > 0).all(), "TRAIN_ONLY_NORMALIZATION_NOT_DEV_FIT")
    model = torch.nn.Sequential(torch.nn.Linear(1536, 32),
                                torch.nn.ReLU(),
                                torch.nn.Linear(32, 3))
    need(sum(p.numel() for p in model.parameters()) == 49283,
         "ONLY_FRESH_49283_PARAMETER_HEAD")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    grouped = defaultdict(lambda: defaultdict(list))
    for bag in train_bags:
        grouped[bag["weak_class"]][bag["source_group_id"]].append(bag)
    need(set(grouped) == {0, 1, 2} and
         all(len(grouped[cl]) >= 6 for cl in (0, 1, 2)),
         "ALL_THREE_CLASSES_WITH_AT_LEAST_6_INDEPENDENT_TRAIN_GROUPS")
    tensors = {b["id"]: torch.from_numpy(
        ((b["windows"] - mean) / scale).reshape(-1, 1536).astype(np.float32))
        for b in train_bags}
    losslog, drawlog = [], []
    for step in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        losses, weights, drawn = [], [], []
        progress["optimizer_updates_started"] += 1
        for cl in (0, 1, 2):
            speakers = sorted(grouped[cl])
            speaker = speakers[step % len(speakers)]
            choices = sorted(grouped[cl][speaker], key=lambda b: b["id"])
            bag = choices[(step // len(speakers)) % len(choices)]
            scores = torch.log_softmax(model(tensors[bag["id"]]), dim=-1)
            value = (-scores[:, 0].mean() if cl == 0 else
                     torch.log(torch.tensor(len(scores), dtype=torch.float32)) -
                     torch.logsumexp(scores[:, cl], dim=0))
            losses.append(value * bag["tier_weight"])
            weights.append(bag["tier_weight"])
            drawn.append({"source_group_id": speaker, "id": bag["id"],
                          "class": cl, "tier_weight": bag["tier_weight"]})
        objective = torch.stack(losses).sum() / sum(weights)
        need(bool(torch.isfinite(objective)), "FINITE_ADAMW_OBJECTIVE")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), spec["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        progress["optimizer_updates_completed"] += 1
        losslog.append(float(objective.detach()))
        drawlog.append(drawn)
    arrays = v2._model_arrays(model, mean, scale)
    modeldir = Path(out) / "models"
    modeldir.mkdir()
    npz = modeldir / "newtrain24-dev-only.npz"
    np.savez(npz, **arrays)
    with np.load(npz, allow_pickle=False) as npz_read:
        frozen = {k: npz_read[k].copy() for k in npz_read.files}
    need(set(frozen) == set(base.SHAPES) and
         all(np.array_equal(frozen[k], arrays[k]) for k in frozen),
         "ACTUAL_MODEL_WEIGHT_BINARY_ROUNDTRIP")
    raw = base.MAGIC + b"".join(
        frozen[name].astype("<f4").tobytes() for name in base.ORDER)
    need(len(raw) == 197908, "ORIGINAL_C_NATIVE_WEIGHT_BINARY_SHAPE")
    (modeldir / "newtrain24-dev-only.bin").write_bytes(raw)
    model_sha = sha(raw)
    predictions = [base.score(frozen, (mean, scale), b, THRESHOLD)
                   for b in dev_bags]
    overall = base.aggregate(predictions)
    old_dev = [p for p in predictions if
               p["source_group_id"] not in
               {r["source_group_id"] for r in rows if r["corpus"] == "newtrain24"}]
    new_dev = [p for p in predictions if
               p["source_group_id"] in
               {r["source_group_id"] for r in rows if r["corpus"] == "newtrain24"}]
    need(len(old_dev) and len(new_dev) and
         {p["weak_class"] for p in old_dev} == {0, 1, 2} and
         {p["weak_class"] for p in new_dev} == {0, 1, 2},
         "SEPARATE_OLD_AND_NEW_DEV_SIGNAL_NO_LEAKAGE")
    original_part = base.aggregate(old_dev)
    new_part = base.aggregate(new_dev)
    def receipt(path, value):
        base.saved(Path(out) / path, value)
    receipt("TRAINING.json", {
        "schema": SCHEMA, "status": "REAL_96_FRESH_SEED_ADAMW_UPDATES",
        "fresh_seed": SEED, "optimizer": "AdamW",
        "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "train_bags": len(train_bags), "dev_bags": len(dev_bags),
        "new_groups_post_feature": old_new_groups,
        "total_groups_post_feature": dict(count),
        "losses": losslog, "train_only_draws": drawlog,
        "rejected_feature_geometry": excluded,
        "real_frozen_onxx_frontend_calls": progress["actual_frontend_forward_calls"],
        "optimizer_updates_started": progress["optimizer_updates_started"],
        "optimizer_updates_completed": progress["optimizer_updates_completed"],
        "old_calibration_pcm_opened": False,
        "old_locked_pcm_opened": False,
        "new_calibration_pcm_opened": False,
        "new_locked_pcm_opened": False,
        "new_tts_asr_phone_forwards": 0, "human_gold": False})
    receipt("WEIGHTS_FROZEN.json", {
        "schema": SCHEMA, "status": "NEWTRAIN24_96_STEP_RESEARCH_WEIGHTS_FROZEN",
        "bin_sha256": model_sha, "npz_sha256": sha(npz.read_bytes()),
        "param_count": 49283, "bin_size": len(raw),
        "development_threshold_predeclared": THRESHOLD,
        "old_meero_weight_warmstart": False,
        "calibration_or_locked_audio_in_model_selection": False})
    result = {
        "schema": SCHEMA, "status": "RESEARCH_NEWTRAIN24_DEV_ONLY_WEIGHT_COMPLETE",
        "new_model": {"parameters": 49283, "sha256": model_sha,
                      "raw_bytes": len(raw)},
        "fresh_real_optimizer_updates": UPDATES,
        "training_groups": count["train"], "development_groups": count["development"],
        "new_source_groups_used": old_new_groups,
        "development": overall,
        "old_source_development": original_part,
        "new_source_development": new_part,
        "selected_checkpoint": "fixed_step96_no_heldout_or_dev_sweep",
        "old_locked_test_reopened": False,
        "fresh_calibration_and_locked_sources_still_required": True,
        "automatic_model_promotion": False,
        "real_3_5m_far_frr": "NOT_MEASURED",
        "physical_ssc305": "NOT_MEASURED"}
    receipt("RESULT.json", result)
    return result


def execute(sources, assets, output):
    identity = guard()
    out = Path(output)
    need(out.is_absolute() and not out.exists() and out.parent.is_dir(),
         "NEW_UNREPEATED_NUMERIC_OUTPUT_DIRECTORY")
    out.mkdir()
    base.saved(out / "STARTED.json", {
        "schema": SCHEMA, "run": identity,
        "model_updates_planned": UPDATES,
        "source_run": SOURCE_RUN, "screen_run": SCREEN_RUN,
        "no_old_calibration_or_locked_inference": True,
        "new_model_forwards": 0, "new_audio_generation": 0})
    archives = {}
    try:
        remote = verified_remote(os.environ.get("GH_TOKEN"))
        selected, audit = all_train_dev(sources)
        base.saved(out / "ADMISSION.json", {
            "schema": SCHEMA, "status": "52_REAL_TRAIN_DEV_SOURCE_ROW_BIND",
            "actual_train_dev_counts": dict(Counter(x["role"] for x in selected)),
            "original_63_source_qualified": True,
            "new_review": audit, "remote_artifacts": remote,
            "no_old_locked_test_selection": True,
            "selected_sha256": sha(binding.canon(selected))})
        base.verify_assets(assets)
        for spec in binding.config()["corpus"]:
            key = spec["key"]
            archives[key] = binding.archive(
                Path(sources) / (key + "-source.zip"), spec["source"])
        archives["newtrain24"] = binding.archive(
            Path(sources) / "newtrain24-source.zip", SOURCE)
        frontend = base.make_frontend(assets)
        result = numeric_train(selected, archives, frontend, out)
        print(json.dumps({
            "status": result["status"],
            "new_model_sha256": result["new_model"]["sha256"],
            "fresh_real_optimizer_updates": result["fresh_real_optimizer_updates"],
            "training_groups": result["training_groups"],
            "development_groups": result["development_groups"],
            "new_source_groups_used": result["new_source_groups_used"],
            "development": result["development"],
            "new_source_development": result["new_source_development"],
            "promotion": False}, sort_keys=True))
    except BaseException as exc:
        base.saved(out / "STOP.json", {
            "schema": SCHEMA, "status": "STOP_NO_MODEL_PROMOTION",
            "exception": type(exc).__name__, "reason": str(exc)[:600],
            "automatic_retry": False,
            "original_locked_test_reopened": False})
        raise
    finally:
        for archive in archives.values():
            archive.close()


def main():
    p = argparse.ArgumentParser()
    flags = ("--source-check", "--guard", "--audit-inputs", "--execute")
    for flag in flags:
        p.add_argument(flag, action="store_true")
    p.add_argument("--sources")
    p.add_argument("--assets")
    p.add_argument("--output")
    a = p.parse_args()
    need(sum(bool(getattr(a, f[2:].replace("-", "_"))) for f in flags) == 1,
         "EXACT_ONE_NUMERIC_STAGE")
    if a.source_check:
        print(json.dumps(source_check(), sort_keys=True))
    elif a.guard:
        print(json.dumps(guard(), sort_keys=True))
    elif a.audit_inputs:
        need(a.sources, "EIGHT_SAVED_ZIPS_BEFORE_NUMERIC_TRAINING")
        rows, report = all_train_dev(a.sources)
        print(json.dumps({"status": "AUTHENTICATED_52_TRAIN_DEV_AUDIO",
                          "roles": dict(Counter(x["role"] for x in rows)),
                          "new_review": report,
                          "numeric_model_forwards": 0}, sort_keys=True))
    else:
        need(all((a.sources, a.assets, a.output)),
             "ORIGINAL_EIGHT_SAVED_SOURCES_AND_TWO_PINNED_ONNX_REQUIRED")
        execute(a.sources, a.assets, a.output)


if __name__ == "__main__":
    main()
