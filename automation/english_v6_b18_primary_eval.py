"""One new prospective source assessment of the already frozen B17 models.

B17's incomplete synthetic source result remains STOP. No B17 weights,
training assessment, original natural source labels, or quality thresholds are
changed by this separate B18 evaluation. All model work is inert until exact
new source/screen artifacts and the successful original natural panel are bound.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_primary_eval as old
import english_v6_b17_replication_eval as prior_replica
import english_v6_b18_source as synth

SCHEMA = "english-v6-b18-b17-frozen-models-primary-v1"
SPEC = Path(__file__).with_name("english_v6_b18_primary_inputs.json")
REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b18-primary-scoring-once-20261124"
GAINS = old.GAINS
GATE = old.GATE
need = old.common.need
sha = old.common.sha
ARTIFACT_KEYS = {"B0", "B9", "pair", "synth_source", "synth_screen", "natural_source", "natural_screen"}


def protocol():
    prior = old.protocol()
    need(prior["scoring_allowed"] is False,
         "B17 incomplete synthetic source remains permanently closed")
    doc = json.loads(SPEC.read_bytes())
    raw = synth.ASSESSMENT.read_bytes()
    need(sha(raw) == synth.ASSESSMENT_SHA, "frozen B18 source assessment")
    assessment = json.loads(raw)
    need(doc["schema"] == SCHEMA and type(doc["scoring_allowed"]) is bool
         and doc["assessment_sha256"] == synth.ASSESSMENT_SHA
         and doc["training_assessment_sha256"] == old.ASSESSMENT_SHA
         and doc["training_design_sha256"] == old.DESIGN_SHA
         and doc["parent_pt_sha256"] == old.B0_PT
         and doc["pair_schema"] == old.PAIR_SCHEMA and doc["pair_stems"] == old.STEMS
         and doc["source_gate"] == GATE and doc["gains"] == list(GAINS)
         and set(doc["inputs"]) == ARTIFACT_KEYS
         and doc["model_pt_sha256"] == assessment["frozen_model_pt_sha256"]
         and doc["maximum_model_forwards"] == 1404
         and all(doc[k] is False for k in ("training_allowed", "threshold_search",
                  "promotion_allowed", "product_released", "cal_locked_access",
                  "B17_failed_synthetic_used_for_scoring", "replication_opened")),
         "new data-only assessment cannot rewrite B17 training, source failure or quality criteria")
    for key in ("B0", "B9", "pair"):
        need(doc["inputs"][key] == assessment["frozen_model_artifacts"][key],
             "same authenticated frozen checkpoint archive " + key)
    reuse = assessment["reused_natural_primary"]
    need(doc["inputs"]["natural_source"] == reuse["source"]
         and doc["inputs"]["natural_screen"] == reuse["screen"]
         and reuse["admitted_originals"] == 21 and reuse["admitted_speakers"] == 11
         and reuse["original_slots"] == 24 and reuse["original_readers"] == 12,
         "reuse the original unscored B17 natural panel without relabeling or re-observation")
    if doc["scoring_allowed"]:
        for lock in doc["inputs"].values():
            prior_replica.opening.valid_artifact(lock)
        need(all(doc["inputs"]["synth_source"][k] == doc["inputs"]["synth_screen"][k]
                 for k in ("run_id", "run_head")),
             "new source and its strict readback belong to the same one-shot run")
    return doc


def guard(prepared_parent, env=None, event=None, parent=None):
    e = os.environ if env is None else env
    need(re.fullmatch(r"[0-9a-f]{40}", prepared_parent or "") is not None,
         "actual reviewed predecessor required")
    need(all(e.get(k) == value for k, value in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH,
        "GITHUB_RUN_ATTEMPT": "1"}.items())
         and re.fullmatch(r"[0-9a-f]{40}", e.get("GITHUB_SHA", "")) is not None
         and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")) is not None,
         "first designated public B18 score event only")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == e["GITHUB_SHA"]
         and event.get("ref") == BRANCH and event.get("forced") is False
         and event.get("deleted") is False and event.get("repository", {}).get("private") is False
         and event.get("repository", {}).get("full_name") == REPO,
         "one reviewed parent, no force push, replay or private execution")
    return dict(run_id=e["GITHUB_RUN_ID"], run_head=e["GITHUB_SHA"], prepared_parent=parent)


def fetch(inputs, prepared_parent):
    guard(prepared_parent)
    meta = protocol()
    need(meta["scoring_allowed"] is True, "source qualification and all original inputs required")
    inputs.mkdir()
    for key, pin in meta["inputs"].items():
        art = json.loads(prior_replica.opening.gh("actions/artifacts/" + str(pin["id"])))
        run = json.loads(prior_replica.opening.gh("actions/runs/" + str(pin["run_id"])))
        need(art["id"] == pin["id"] and art["expired"] is False
             and art["size_in_bytes"] == pin["bytes"] and art["digest"] == "sha256:" + pin["sha256"]
             and art["workflow_run"]["id"] == pin["run_id"]
             and art["workflow_run"]["head_sha"] == pin["run_head"]
             and run["head_sha"] == pin["run_head"] and run["run_attempt"] == 1
             and run["status"] == "completed" and run["conclusion"] == "success"
             and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False,
             "original successful public input run and artifact " + key)
        raw = prior_replica.opening.gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "original input ZIP bytes " + key)
        (inputs / (key + ".zip")).write_bytes(raw)


def verified_synthetic(source, screen, source_pin=None):
    """Read full original PCM before selecting the fixed blind source labels."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "source"
        prior_replica.extract_source(source, directory)
        originals = synth.read_source(directory)
        if source_pin is not None:
            actual = synth.lineage.require_authorization(directory)["source_execution"]
            need(str(actual["run_id"]) == str(source_pin["run_id"])
                 and actual["run_head"] == source_pin["run_head"] and actual["run_attempt"] == 1
                 and actual["run_branch"] == synth.lineage.BRANCH,
                 "original source authorization matches its actual artifact run identity")
    raw = source.read("MANIFEST.json")
    manifest = json.loads(raw)
    lab = json.loads(screen.read("SCREEN.json"))
    plan = synth.source_plan("primary")
    assessment = synth.lineage.read_plan()
    synth.code_contract()
    need(manifest["schema"] == synth.SCHEMA and manifest["panel"] == "primary"
         and lab["schema"] == synth.BATCH and lab["panel"] == "primary"
         and lab["source_manifest_sha256"] == sha(raw)
         and lab["assessment_sha256"] == synth.ASSESSMENT_SHA
         and lab["registered_plan_file_sha256"] == sha((synth.PLANS / "primary.json").read_bytes())
         and lab["coverage"] == dict.fromkeys(synth.qa.observers.PROVIDERS, 96)
         and lab["source_ancestry_sha256"] == synth.lineage.BASE_ANCESTRY_SHA
         and lab["parser_contract_sha256"] == assessment["parser_contract_sha256"]
         and lab["observer_code_contract"] == assessment["observer_code_contract"]
         and lab["strict_raw_receipt_readback"] is True
         and lab["B17_source_stop_retained"] is True and lab["all_prior_replication_sealed"] is True
         and lab["original_B17_source_proof"] == assessment["original_B17_source_proof"]["source_inputs"]
         and lab["new_model_forwards"] == lab["optimizer_updates"] == lab["tts_calls_at_screen"] == 0
         and lab["training_allowed"] is lab["promotion_allowed"] is lab["replication_opened"] is False
         and len(originals) == len(lab["rows"]) == len(plan["rows"]) == 96,
         "complete original five-by-96 source qualification under the new frozen contract")
    chosen, groups, unknown = [], {}, 0
    for row, label, planned in zip(originals, lab["rows"], plan["rows"], strict=True):
        need(all(row[k] == value for k, value in planned.items())
             and all(label[k] == value for k, value in row.items() if k != "observed_class"),
             "no missing, altered, reordered or replaced original source record")
        cls = label["observed_class"]
        if cls is None:
            need(label["tier"] == "unknown", "unqualified source stays unknown")
            unknown += 1
            continue
        need(type(cls) is int and cls == row["requested_class"] and cls in (0, 2)
             and label["tier"] in ("silver", "weak"), "unchanged conservative source label")
        groups.setdefault(row["source_group_id"], set()).add(cls)
        chosen.append(dict(row, observed_class=cls, tier=label["tier"]))
    paired = sorted(key for key, classes in groups.items() if classes == {0, 2})
    counts = dict(synthetic_positive=sum(r["observed_class"] == 2 for r in chosen),
                  synthetic_ordinary=sum(r["observed_class"] == 0 for r in chosen),
                  synthetic_paired=len(paired))
    need(counts == dict(synthetic_positive=lab["source_coverage"]["positive_originals"],
                       synthetic_ordinary=lab["source_coverage"]["ordinary_originals"],
                       synthetic_paired=lab["source_coverage"]["paired_source_groups"])
         and paired == lab["paired_source_groups"] and unknown == lab["unknown_count"],
         "recomputed true source denominators")
    minima = dict(positive_originals=12, ordinary_originals=24, paired_source_groups=8)
    missing = {key: {"observed": lab["source_coverage"][key], "min": value}
               for key, value in minima.items() if lab["source_coverage"][key] < value}
    need(lab["frozen_minima"] == minima and lab["source_shortfall"] == missing
         and lab["source_gate"] == ("INSUFFICIENT_SOURCE" if missing else "SOURCE_GATE_PASS"),
         "source report cannot alter the frozen original-count gate")
    return chosen, counts, originals


def verified_natural(source, screen):
    # This is deliberately the old source verifier and old source assessment.
    # The new parser must never be applied to these original successful captures.
    selected, counts = old.verified_human(source, screen)
    need(counts == {"natural_originals": 21, "natural_speakers": 11},
         "original successful natural screen retains its exact 21/11 denominator")
    original = json.loads(source.read("MANIFEST.json"))["rows"]
    need(len(original) == 24, "all original human slots retained, including three UNKNOWN")
    # Audit every original PCM, including UNKNOWN, before any model is loaded.
    for row in original:
        raw = source.read("wav/" + row["id"] + ".wav")
        with wave.open(io.BytesIO(raw)) as audio:
            need(audio.getparams()[:3] == (1, 2, 16000)
                 and audio.getnframes() == row["samples"], "whole original natural PCM")
            pcm = audio.readframes(audio.getnframes())
        need(sha(raw) == row["wav_sha256"] and sha(pcm) == row["pcm_sha256"],
             "all 24 original natural identities, including unchanged UNKNOWN")
    return selected, counts, original


def validate_originals(synthetic, natural, training_sources):
    rows = synthetic + natural
    need(len(synthetic) == 96 and len(natural) == 24
         and len({r["id"] for r in rows}) == len({r["pcm_sha256"] for r in rows}) == 120,
         "all original identities and PCM remain distinct before admission filtering")
    need(not ({prior_replica.canonical_person(r) for r in synthetic}
              & {prior_replica.canonical_person(r) for r in natural}),
         "new synthetic and original natural use distinct canonical people")
    old.reject_train_overlap(rows, training_sources)


def score(inputs, out, prepared_parent):
    identity = guard(prepared_parent)
    meta = protocol()
    need(meta["scoring_allowed"] is True, "inert until all seven real inputs are qualified")
    out.mkdir()
    report = dict(schema=SCHEMA, **identity, assessment_sha256=synth.ASSESSMENT_SHA,
        training_assessment_sha256=old.ASSESSMENT_SHA, training_design_sha256=old.DESIGN_SHA,
        source_gate="NOT_RUN", panel_gate="NOT_RUN", actual_forwards=0, saved_array_count=0,
        optimizer_updates=0, threshold_search=False, training_allowed=False,
        promotion_allowed=False, product_released=False, replication_open_allowed=False,
        B17_failed_synthetic_used_for_scoring=False, B17_source_failure_preserved=True,
        B13_replication_opened=False, B14_replication_opened=False,
        B15_replication_opened=False, B16_replication_opened=False, B17_replication_opened=False,
        cal_locked_access=False, ssc305_physical=False,
        installed_software_baseline="B0", natural_field_far_per_hour=None, field_3_5m_frr=None,
        original_natural_reused_without_reobservation=True,
        registered_inputs_sha256=sha(SPEC.read_bytes()), implementation_sha256=sha(Path(__file__).read_bytes()))
    arrays, calls, hooks = {}, dict.fromkeys(old.common.NAMES, 0), []
    np = None
    try:
        with ExitStack() as stack:
            arcs = {key: stack.enter_context(old.common.opened(inputs / (key + ".zip"), meta["inputs"][key]))
                    for key in ("synth_source", "synth_screen", "natural_source", "natural_screen")}
            sr, sc, all_synth = verified_synthetic(arcs["synth_source"], arcs["synth_screen"],
                                                 meta["inputs"]["synth_source"])
            nr, nc, all_natural = verified_natural(arcs["natural_source"], arcs["natural_screen"])
            counts = {**sc, **nc}
            gate, missing = old.source_gate(counts)
            report.update(source_coverage=counts, source_gate=gate, source_shortfall=missing,
                          panel_gate="INSUFFICIENT_SOURCE")
            if missing:
                return
            with old.common.opened(inputs / "pair.zip", meta["inputs"]["pair"]) as archive:
                old.validate_pair_receipt(json.loads(archive.read("RESULT.json")))
                training_sources = json.loads(archive.read("SOURCE.json"))
            validate_originals(all_synth, all_natural, training_sources)
            import numpy as np
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.mkldnn.enabled = False
            models, loaded_training_sources = old.snapshots(inputs, meta, torch, np)
            need({key: value[-1] for key, value in models.items()} == meta["model_pt_sha256"],
                 "B18 cannot change any of the four frozen B17 model roles")
            need(training_sources == loaded_training_sources, "same authenticated TRAIN inventory")
            for name, values in models.items():
                def before(_module, _args, role=name):
                    calls[role] += 1
                hooks.append(values[0].register_forward_pre_hook(before))
            records = []
            for key, cohort, admitted in (("synth_source", "synthetic", sr),
                                          ("natural_source", "natural", nr)):
                for row in admitted:
                    gains = old.common.infer(arcs[key].read("wav/" + row["id"] + ".wav"),
                                             row, models, arrays, torch, np)
                    records.append(dict(id=row["id"], group=row["source_group_id"], cohort=cohort,
                        cell=row["cell_id"], positive=row["observed_class"] == 2, tier=row["tier"],
                        pcm_sha256=row["pcm_sha256"], gains=gains))
            need(len(records) == len({r["pcm_sha256"] for r in records})
                 and sum(calls.values()) == len(arrays) == 12 * len(records)
                 and sum(calls.values()) <= meta["maximum_model_forwards"],
                 "complete bounded original four-model/three-gain assessment")
            decision = old.old_decision.decide(records)
            report.update(decision, records=records,
                model_pt_sha256={key: value[-1] for key, value in models.items()},
                replication_open_allowed=decision["panel_gate"] == "PASS")
    except Exception as exc:
        report.update(panel_gate="STOP", error=str(exc), replication_open_allowed=False)
        raise
    finally:
        for handle in hooks:
            handle.remove()
        report.update(actual_forwards=sum(calls.values()), model_forward_calls=calls,
                      saved_array_count=len(arrays))
        if arrays:
            target = out / "ALL_FRAME_LOGITS.npz"
            np.savez_compressed(target, **arrays)
            report["logits_sha256"] = sha(target.read_bytes())
        (out / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "fetch", "score"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-parent")
    args = parser.parse_args()
    if args.mode == "check":
        doc = protocol()
        print(json.dumps({"schema": SCHEMA, "scoring_allowed": doc["scoring_allowed"],
                          "promotion_allowed": False, "model_forwards": 0}, sort_keys=True))
    else:
        need(args.inputs is not None, "exact input directory required")
        if args.mode == "fetch":
            fetch(args.inputs, args.prepared_parent)
        else:
            need(args.output is not None, "new result directory required")
            score(args.inputs, args.output, args.prepared_parent)
