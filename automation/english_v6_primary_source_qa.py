"""Blind source admission for the predeclared primary synthetic DEV panel."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_fresh24_ctc_admission as policy
import english_v6_b11_dev_panel as panel
import english_v6_b11_supplement_observe as observers
from english_v6_b11_supplement import need, pcm_tools, record
from english_v6_b11_supplement_screen import load_observations, sha

SCHEMA = "english-v6-primary-blind-source-qa-v1"
PARENT = "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655"
PLAN_SHA = panel.b10.sha(panel.b10.canon(panel.source_plan("primary")))


def read_panel(source, panel_name, recipe=panel.source_plan):
    manifest = json.loads((source / "MANIFEST.json").read_bytes())
    plan = recipe(panel_name)
    planned = plan["rows"]
    plan_sha = panel.b10.sha(panel.b10.canon(plan))
    rows = manifest["rows"]
    need(manifest["schema"] == plan["schema"] and manifest["panel"] == panel_name
         and manifest["plan_sha256"] == plan_sha and manifest["new_kws_forwards"] == 0
         and manifest["tts_calls"] == len(rows) == 48 and manifest["human_gold"] is False
         and len({r["pcm_sha256"] for r in rows}) == 48, "complete original primary cohort")
    for row, expected in zip(rows, planned, strict=True):
        need(all(row[k] == v for k, v in expected.items()) and row["whole_native_resample"]
             and row["crop_calls"] == 0, "frozen primary source identity")
        facts = pcm_tools.inspect_wav(source / "wav" / (row["id"] + ".wav"))
        need(facts["basic_audio_ok"] and all(row[k] == v for k, v in facts.items()),
             "original WAV and PCM digests")
    return rows


def read_primary(source):
    return read_panel(source, "primary")


def admit(row, texts, posteriors):
    reject = {"tier": "unknown", "class": None, "margins": {}, "training_allowed": False}
    if row["requested_class"] == 2:
        # The three ASRs and both phone encoders ran without target hints. The
        # frozen label rule then contrasts each native phone posterior with the
        # alternate pronunciation and rejects contradictions and out-of-range durations.
        return policy.decide(dict(row, pcm_samples=row["samples"]), texts, posteriors)
    allowed = dict(panel.ORDINARY)
    need(row["requested_class"] == 0 and allowed[row["cell_id"]] == row["requested_text"],
         "predeclared ordinary sentence")
    if set(texts) != set(policy.TEXT_MODELS) or set(posteriors) != set(policy.PHONE_MODELS):
        return dict(reject, reason="MISSING_FIVE_MODEL_OBSERVATIONS")
    heard = [policy.norm(texts[name]) for name in policy.TEXT_MODELS]
    if any(t in policy.OLD_ALIASES or re.search(r"\b(?:hey|hay|mero|meero|miro|marrow|merrow)\b", t)
           for t in heard):
        return dict(reject, reason="CONTRADICTORY_WAKE_TRANSCRIPT")
    for p in policy.PHONE_MODELS:
        # load_observations authenticates the buffer SHA, native vocabulary and
        # full finite tensor and checks the saved greedy tokens for each row.
        value = posteriors[p]
        if policy.greedy_full_wake(value["logits"], value["vocab"]):
            return dict(reject, reason="NATIVE_FULL_WAKE_VETO")
    votes = heard.count(policy.norm(row["requested_text"]))
    if votes < 2:
        return dict(reject, reason="INSUFFICIENT_EXACT_SENTENCE_AGREEMENT")
    return dict(reject, tier="silver" if votes == 3 else "weak", **{"class": 0},
                reason="BLIND_EXACT_SENTENCE_NO_WAKE")


def screen(source, evidence, output, *, panel_name="primary", schema=SCHEMA, recipe=panel.source_plan):
    rows = read_panel(source, panel_name, recipe)
    texts, posteriors, coverage = load_observations(source, evidence, rows, schema)
    decisions = []
    qualified = defaultdict(set)
    for row in rows:
        sid = row["id"]
        result = admit(row, texts[sid], posteriors[sid])
        observed = result["class"]
        need(result["training_allowed"] is False and
             (observed is None or observed == row["requested_class"]), "DEV role and class")
        if observed is not None:
            qualified[row["source_group_id"]].add(observed)
        decisions.append(dict(row, observed_class=observed, tier=result["tier"],
                              reason=result["reason"], phone_margins=result["margins"]))
    paired = sorted(g for g, classes in qualified.items() if classes == {0, 2})
    metadata = {"primary_originals": 48} if panel_name == "primary" else {"replication_originals": 48}
    record(output, dict(schema=schema, source_manifest_sha256=sha(source / "MANIFEST.json"),
                        plan_sha256=panel.b10.sha(panel.b10.canon(recipe(panel_name))),
                        rows=decisions, coverage=coverage, reviewed_source_groups=12, **metadata,
                        paired_source_groups=paired, meets_four_pair_minimum=len(paired) >= 4,
                        tier_counts=dict(Counter(r["tier"] for r in decisions)),
                        frozen_before_kws=True, kws_forwards=0, optimizer_updates=0,
                        human_gold=False, replication_opened=panel_name == "replication", promotion_allowed=False))


def check_train_gate(archive, expected_digest):
    need(sha(archive) == expected_digest, "exact original TRAIN ZIP digest")
    with zipfile.ZipFile(archive) as z:
        need(z.testzip() is None, "TRAIN archive CRC")
        result = json.loads(z.read("score/RESULT.json"))
        labels = z.read("SCREEN.json")
        need(hashlib.sha256(labels).hexdigest() == result["screen_sha256"], "TRAIN label binding")
        logits = z.read("score/FRAME_LOGITS.npz")
        need(hashlib.sha256(logits).hexdigest() == result["logits_sha256"], "TRAIN logits binding")
    rows = result["scored"]
    hard = {r["group"] for r in rows if r["gains"]["1.0"]["events"] > 0}
    need(result["batch"] == "b11-diverse-train-supplement-v1" and
         result["parent_pt_sha256"] == PARENT and result["original_hard_groups"] == []
         and len(rows) == result["originals"] and len(rows) + result["unknown"] == 24
         and 3 * len(rows) == result["kws_forwards"] and result["optimizer_updates"] == 0 and
         sorted(hard) == result["combined_hard_groups"] == result["new_hard_groups"] and
         len(hard) >= 2 and result["train_hardness_gate"] == "PASS",
         "B11 requires measured original-gain whole-word false wakes in two TRAIN families")


def check_b12_gate(archive, expected_digest):
    need(sha(archive) == expected_digest, "exact original independent B12 TRAIN diagnostic ZIP")
    with zipfile.ZipFile(archive) as z:
        need(z.testzip() is None, "original B12 TRAIN archive CRC")
        r = json.loads(z.read("RESULT.json"))
        raw = z.read("TRAIN_POSITIVE_FRAME_LOGITS.npz")
        need(hashlib.sha256(raw).hexdigest() == r["logits_npz_sha256"], "frozen logit archive")
    records = r["records"]
    hard = {x["group"] for x in records if x["delivery"]["verified_faster"] and
            x["gains"]["1.0"]["events"] > 0 and x["gains"]["0.5"]["events"] == 0 and
            1 not in x["gains"]["0.5"]["tokens"]}
    need(r["schema"] == "english-v6-b12-frozen-train-positive-diagnostic-v1" and
         r["run"] == "45094053219" and r["head"] == "6129dcc7e9c7fc4675f049892f3dcea4a8bb812c" and
         r["model_pt_sha256"] == PARENT and r["originals"] == len(records) == 44 and
         r["source_groups"] == 35 and r["actual_forwards"] == 132 and
         r["verified_faster_clips"] == sum(x["delivery"]["verified_faster"] for x in records) == 14 and
         sorted(hard) == r["verified_faster_clean_hit_half_miss_hey_absent_groups"] and
         len(hard) >= 2 and r["two_group_trigger"] is True and
         r["optimizer_updates"] == r["dev_cal_locked_forwards"] == 0, "B12 TRAIN trigger")


def generate(output, *, panel_name="primary"):
    from huggingface_hub import hf_hub_download
    model = panel.source_plan(panel_name)["model"]
    paths = [Path(hf_hub_download(repo_id=model["repo"], filename=model[k],
                                 revision=model["commit"], token=False)) for k in ("file", "config")]
    panel.generate_panel(panel_name, *paths, output)
    read_panel(output, panel_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="mode", required=True)
    for choice in ("b11", "b12"):
        p = cmd.add_parser(choice)
        p.add_argument("evidence", type=Path)
        p.add_argument("sha256")
    cmd.add_parser("generate").add_argument("out", type=Path)
    p = cmd.add_parser("observe")
    p.add_argument("provider", choices=policy.TEXT_MODELS + policy.PHONE_MODELS)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p = cmd.add_parser("screen")
    for arg in ("source", "evidence", "output"):
        p.add_argument(arg, type=Path)
    args = parser.parse_args()
    if args.mode == "b11":
        check_train_gate(args.evidence, args.sha256)
    elif args.mode == "b12":
        check_b12_gate(args.evidence, args.sha256)
    elif args.mode == "generate":
        generate(args.out)
    elif args.mode == "observe":
        observers.observe(args.provider, args.source, args.output, read_rows=read_primary, schema=SCHEMA)
    else:
        screen(args.source, args.evidence, args.output)
