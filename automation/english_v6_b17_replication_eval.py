"""Inert-by-default B17 replication: same frozen models and original B14 decision.

Opening requires the byte-authenticated successful primary and all saved logits.
Replication independently verifies the original history, new source bytes, blind
source labels and same-model identities; it cannot deploy or train any model.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b17_primary_eval as primary
import english_v6_b17_replication_gate as opening
import english_v6_b17_replication_source as synthetic
import english_v6_b17_replication_natural_sources as natural

common = primary.common
need, sha = common.need, common.sha
SCHEMA = "english-v6-b17-unseen-replication-four-frozen-models-v1"
REGISTERED = Path(__file__).with_name("english_v6_b17_replication_inputs.json")
RUN_BRANCH = "refs/heads/research/en-v6-b17-replication-scoring-once-20261124"
GATE = dict(primary.GATE)
INPUT_KEYS = set(primary.INPUT_KEYS)


def protocol():
    doc = json.loads(REGISTERED.read_bytes())
    assessment, ancestry = primary.synth.lineage.read_plan()
    need(doc["schema"] == SCHEMA and type(doc["scoring_allowed"]) is bool
         and doc["source_gate"] == GATE and doc["pair_schema"] == primary.PAIR_SCHEMA
         and doc["pair_stems"] == primary.STEMS and set(doc["inputs"]) == INPUT_KEYS
         and doc["parent_pt_sha256"] == primary.B0_PT
         and doc["assessment_sha256"] == primary.ASSESSMENT_SHA
         and doc["training_design_sha256"] == primary.DESIGN_SHA
         and doc["source_ancestry_sha256"] == primary.synth.lineage.ANCESTRY_SHA
         and all(doc[k] is False for k in ("promotion_allowed", "product_released",
                   "physical_ssc305", "cal_locked_access", "old_replication_opened",
                   "training_allowed", "threshold_search")),
         "replication preserves frozen model/source/quality roles and cannot deploy")
    need(assessment["replication"]["synthetic"]["speaker_indices"] == list(range(742, 766))
         and assessment["replication"]["natural"]["corpus_split"] == "dev-other"
         and assessment["replication"]["natural"]["requested_speakers"] == 9
         and sha(Path(primary.old_decision.__file__).read_bytes()) == opening.DECISION_SHA
         and sha(primary.old_decision.ASSESSMENT.read_bytes()) == opening.QUALITY_SHA,
         "pre-optimizer replica source identities and original B14 decision unchanged")
    return doc


def guard(prepared_parent, env=None, event=None, parent=None):
    e = os.environ if env is None else env
    need(re.fullmatch(r"[0-9a-f]{40}", prepared_parent or "") is not None,
         "reviewed replication predecessor required; placeholder cannot execute")
    need(all(e.get(k) == v for k, v in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "jiying2007/kws-data",
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": RUN_BRANCH,
        "GITHUB_RUN_ATTEMPT": "1"}.items()), "first designated public replica event only")
    need(re.fullmatch(r"[0-9a-f]{40}", e.get("GITHUB_SHA", "")) is not None
         and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")) is not None,
         "actual replica run identity")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == e["GITHUB_SHA"]
         and event.get("ref") == RUN_BRANCH and event.get("forced") is False
         and event.get("deleted") is False and event.get("repository", {}).get("private") is False
         and event.get("repository", {}).get("full_name") == "jiying2007/kws-data",
         "single reviewed predecessor, public repository, no forced push or replay")
    return dict(run_id=e["GITHUB_RUN_ID"], head=e["GITHUB_SHA"], parent=parent)


def bind_primary(spec, authorization):
    """Recompute primary CTC/decision before opening any new replica archive."""
    need(spec["scoring_allowed"] is True, "replication scoring remains inert until source QA is bound")
    permit = opening.require(authorization)
    need(spec["opening_lock_sha256"] == permit["opening_lock_sha256"]
         and spec["primary_result_sha256"] == permit["primary_result_sha256"]
         and spec["model_pt_sha256"] == permit["model_pt_sha256"],
         "only this authentic primary PASS and its exact four weights authorize replication")
    primary_spec = json.loads((authorization / "english_v6_b17_primary_inputs.json").read_bytes())
    for key in ("B0", "B9", "pair"):
        need(spec["inputs"][key] == primary_spec["inputs"][key],
             "replication cannot replace the actual primary checkpoint archive " + key)
    for lock in spec["inputs"].values():
        opening.valid_artifact(lock)
    return permit


def fetch(specfile, inputs, history, authorization, prepared_parent):
    guard(prepared_parent)
    spec = json.loads(specfile.read_bytes())
    need(spec == protocol(), "checked-in exact replica artifact specification")
    permit = bind_primary(spec, authorization)
    inputs.mkdir()
    history.mkdir()
    for key, locked in spec["inputs"].items():
        artifact = json.loads(opening.gh("actions/artifacts/" + str(locked["id"])))
        run = json.loads(opening.gh("actions/runs/" + str(locked["run_id"])))
        need(artifact["id"] == locked["id"] and artifact["expired"] is False
             and artifact["size_in_bytes"] == locked["bytes"]
             and artifact["digest"] == "sha256:" + locked["sha256"]
             and artifact["workflow_run"]["id"] == locked["run_id"]
             and artifact["workflow_run"]["head_sha"] == locked["run_head"]
             and run["status"] == "completed" and run["conclusion"] == "success"
             and run["run_attempt"] == 1 and run["head_sha"] == locked["run_head"]
             and run["repository"]["full_name"] == "jiying2007/kws-data"
             and run["repository"]["private"] is False,
             "completed original public source/checkpoint artifact " + key)
        raw = opening.gh("actions/artifacts/" + str(locked["id"]) + "/zip")
        need(len(raw) == locked["bytes"] and sha(raw) == locked["sha256"], "original ZIP " + key)
        (inputs / (key + ".zip")).write_bytes(raw)
    # These are old source archives only. No historical model-selection scoring
    # is repeated; history() verifies the immutable original source metadata.
    ids = synthetic.required_archives(authorization)
    need(len(ids) == len(set(ids)) and all(re.fullmatch(r"[1-9][0-9]*", str(i)) for i in ids),
         "explicit bounded historical source archive identifiers")
    for aid in ids:
        raw = opening.gh("actions/artifacts/" + str(aid) + "/zip")
        (history / (str(aid) + ".zip")).write_bytes(raw)
    synthetic.history(history, authorization)
    return permit


def extract_source(archive, target):
    """Extract only bounded ordinary files; source validators verify their bytes."""
    infos = archive.infolist()
    need(len(infos) <= 4096 and len({x.filename for x in infos}) == len(infos)
         and sum(x.file_size for x in infos) <= 128_000_000, "bounded unique source archive members")
    target.mkdir()
    for item in infos:
        relative = PurePosixPath(item.filename)
        need(not relative.is_absolute() and relative.parts and ".." not in relative.parts
             and "\\" not in item.filename and item.file_size <= 16_000_000
             and not stat.S_ISLNK(item.external_attr >> 16), "safe original source member")
        dest = target.joinpath(*relative.parts)
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as handle:
                handle.write(archive.read(item))


def permission_fields(doc, permit):
    need(doc["panel"] == "replication"
         and doc["assessment_sha256"] == primary.ASSESSMENT_SHA
         and doc["source_ancestry_sha256"] == primary.synth.lineage.ANCESTRY_SHA
         and doc["primary_result_sha256"] == permit["primary_result_sha256"]
         and doc["opening_lock_sha256"] == permit["opening_lock_sha256"]
         and doc["primary_source_inputs"] == permit["primary_source_inputs"]
         and doc["old_replication_opened"] is False,
         "source or screen must bind the same primary authorization and unopened old replicas")


def verified_synthetic(source, screen, authorization, permit):
    original = synthetic.read_source(source, authorization)
    manifest_raw = (source / "MANIFEST.json").read_bytes()
    manifest, lab = json.loads(manifest_raw), json.loads(screen.read("SCREEN.json"))
    permission_fields(manifest, permit)
    permission_fields(lab, permit)
    plan = primary.synth.source_plan("replication")
    need(manifest["schema"] == synthetic.SCHEMA == plan["schema"]
         and lab["schema"] == synthetic.BATCH and lab["source_manifest_sha256"] == sha(manifest_raw)
         and manifest["registered_plan_file_sha256"] == primary.synth.FILES["replication"]
         and lab["registered_plan_file_sha256"] == manifest["registered_plan_file_sha256"]
         and lab["coverage"] == dict.fromkeys(primary.synth.qa.observers.PROVIDERS, 96)
         and lab["tts_calls_at_screen"] == lab["new_model_forwards"] == lab["optimizer_updates"] == 0
         and lab["training_allowed"] is lab["promotion_allowed"] is False,
         "new 96-source replication labels use all five observers and original source bytes")
    decisions = {r["id"]: r for r in lab["rows"]}
    need(len(decisions) == len(lab["rows"]) == len(original) == 96
         and set(decisions) == {r["id"] for r in original}, "complete replica source/screen join")
    selected, groups = [], {}
    for row, frozen in zip(original, plan["rows"], strict=True):
        need(all(row[k] == value for k, value in frozen.items()), "frozen replication voice/cell/source role")
        label = decisions[row["id"]]
        need(all(label[k] == value for k, value in row.items() if k != "observed_class"),
             "blind observers cannot mutate original source identity")
        if label["observed_class"] is None:
            need(label["tier"] == "unknown", "unqualified source excluded")
            continue
        need(label["tier"] in ("silver", "weak") and label["observed_class"] == row["requested_class"],
             "same blind canonical source admission rules")
        selected.append(label)
        groups.setdefault(row["source_group_id"], set()).add(label["observed_class"])
    paired = sorted(g for g, classes in groups.items() if classes == {0, 2})
    counts = dict(synthetic_positive=sum(r["observed_class"] == 2 for r in selected),
                  synthetic_ordinary=sum(r["observed_class"] == 0 for r in selected),
                  synthetic_paired=len(paired))
    need(paired == lab["paired_source_groups"] and counts == {
        "synthetic_positive": lab["source_coverage"]["positive_originals"],
        "synthetic_ordinary": lab["source_coverage"]["ordinary_originals"],
        "synthetic_paired": lab["source_coverage"]["paired_source_groups"]},
        "recomputed replication positive/ordinary/source-family denominators")
    return selected, counts, original


def verified_human(source, screen, authorization, permit):
    original = natural.read_source(source, authorization)
    raw = (source / "MANIFEST.json").read_bytes()
    manifest, lab = json.loads(raw), json.loads(screen.read("SCREEN.json"))
    permission_fields(manifest, permit)
    permission_fields(lab, permit)
    assessment, ancestry = primary.synth.lineage.read_plan()
    plan = assessment["replication"]["natural"]
    roster_raw = (source / "ROSTERS.json").read_bytes()
    rosters = json.loads(roster_raw)
    need(manifest["schema"] == lab["schema"] == natural.SCHEMA
         and manifest["corpus_split"] == natural.CORPUS_SPLIT == "dev-other"
         and manifest["corpus_sha256"] == plan["upstream"]["sha256"]
         and manifest["rosters_sha256"] == sha(roster_raw)
         and lab["source_manifest_sha256"] == sha(raw)
         and set(rosters) == {"replication"} and len(rosters["replication"]) == 18
         and [r["speaker"] for r in rosters["replication"]] == [p for p in plan["speakers"] for _ in range(2)]
         and manifest["selected_original_slots"] == 18 and manifest["selected_speakers"] == 9
         and lab["coverage"] == dict.fromkeys(primary.natural.qa.observers.PROVIDERS, len(original))
         and manifest["kws_forwards"] == lab["kws_forwards"] == 0
         and lab["training_allowed"] is False,
         "nine fixed new readers, eighteen slots and five source observers")
    labels = {r["id"]: r for r in lab["decisions"]}
    need(len(original) == len(labels) == len(lab["decisions"])
         and set(labels) == {r["id"] for r in original}, "complete natural replica source/screen join")
    admitted = []
    for row in original:
        label = labels[row["id"]]
        need(str(row["speaker"]) in plan["speakers"] and str(row["speaker"]) not in ancestry["natural_excluded_speakers"]
             and row["source_group_id"] == "librispeech-dev-other-" + str(row["speaker"])
             and row["role"] == "development" and row["training_allowed"] is False
             and label["source_id"] == row["source_id"] and label["pcm_sha256"] == row["pcm_sha256"]
             and label["group"] == row["source_group_id"], "same independent human reader and original waveform")
        if label["tier"] == "unknown":
            continue
        need(label["tier"] in ("silver", "weak") and label["full_wake_veto"] is False
             and type(label["exact_votes"]) is int and label["exact_votes"] >= 2,
             "blindly admitted ordinary natural speech only")
        admitted.append(dict(row, cell_id="read-speech", observed_class=0, tier=label["tier"]))
    counts = dict(natural_originals=len(admitted), natural_speakers=len({r["speaker"] for r in admitted}))
    need(counts["natural_originals"] == lab["admitted"] and counts["natural_speakers"] == lab["admitted_speakers"],
         "recomputed natural source denominator")
    return admitted, counts, original


def canonical_person(row):
    if "source_speaker_id" in row:
        name = row["source_speaker_id"]
        need(type(name) is str and re.fullmatch(r"p[0-9]+", name) is not None, "canonical Piper reader identity")
        return name[1:]
    value = str(row["speaker"])
    need(re.fullmatch(r"[0-9]+", value) is not None, "canonical natural reader identity")
    return value


def reject_ancestry(rows, old_pcm, old_groups, old_people):
    need(len({r["id"] for r in rows}) == len(rows)
         and len({r["pcm_sha256"] for r in rows}) == len(rows), "original replica identities are unique")
    need(not ({r["pcm_sha256"] for r in rows} & old_pcm)
         and not ({r["source_group_id"] for r in rows} & old_groups)
         and not ({canonical_person(r) for r in rows} & old_people),
         "raw history/primary source speaker, family or PCM overlap forbids replication")


def run(specfile, inputs, history, authorization, out, prepared_parent):
    identity = guard(prepared_parent)
    meta = json.loads(specfile.read_bytes())
    need(meta == protocol(), "unchanged checked-in replication input specification")
    permit = bind_primary(meta, authorization)  # Before every replica/source/model read.
    old_pcm, old_groups, old_people = synthetic.history(history, authorization)
    out.mkdir()
    report = dict(schema=SCHEMA, **identity, panel="replication",
        assessment_sha256=primary.ASSESSMENT_SHA, training_design_sha256=primary.DESIGN_SHA,
        source_ancestry_sha256=primary.synth.lineage.ANCESTRY_SHA,
        primary_result_sha256=permit["primary_result_sha256"], opening_lock_sha256=permit["opening_lock_sha256"],
        primary_model_pt_sha256=permit["model_pt_sha256"], source_gate="NOT_RUN", panel_gate="NOT_RUN",
        actual_forwards=0, optimizer_updates=0, threshold_search=False, training_allowed=False,
        promotion_allowed=False, product_released=False, old_replication_opened=False,
        B17_replication_opened=True, cal_locked_access=False, ssc305_physical=False,
        natural_field_far_per_hour=None, field_3_5m_frr=None, installed_software_baseline="B0",
        research_replication_passed=False, candidate_engineering_validation_allowed=False)
    arrays = {}
    np = None
    forward_calls = dict.fromkeys(common.NAMES, 0)
    forward_handles = []
    try:
        with ExitStack() as stack, tempfile.TemporaryDirectory() as tmp:
            archives = {key: stack.enter_context(common.opened(inputs / (key + ".zip"), meta["inputs"][key]))
                        for key in ("synth_source", "synth_screen", "natural_source", "natural_screen")}
            synthdir, humandir = Path(tmp) / "synthetic", Path(tmp) / "natural"
            extract_source(archives["synth_source"], synthdir)
            extract_source(archives["natural_source"], humandir)
            synthrows, synthcounts, all_synth = verified_synthetic(synthdir, archives["synth_screen"], authorization, permit)
            humans, humancounts, all_human = verified_human(humandir, archives["natural_screen"], authorization, permit)
            all_rows = all_synth + all_human
            reject_ancestry(all_rows, old_pcm, old_groups, old_people)
            need(not ({canonical_person(r) for r in all_synth} & {canonical_person(r) for r in all_human}),
                 "synthetic and natural replication use distinct canonical readers")
            # Validate every original WAV retained by the source jobs, including
            # a natural audio-quality reject that cannot enter model scoring.
            all_wav_sha = set()
            for directory in (synthdir, humandir):
                if directory == synthdir:
                    allowed_ids = {r["id"] for r in all_synth}
                else:
                    roster = json.loads((directory / "ROSTERS.json").read_bytes())["replication"]
                    allowed_ids = {"b17-replication-natural-" + r["source_id"] for r in roster}
                for path in sorted((directory / "wav").glob("*.wav")):
                    need(path.stem in allowed_ids, "no unregistered original WAV is allowed in the source archive")
                    with wave.open(str(path), "rb") as audio:
                        need((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype())
                             == (1, 2, 16000, "NONE"), "whole original replica PCM format")
                        digest = sha(audio.readframes(audio.getnframes()))
                    need(digest not in old_pcm and digest not in all_wav_sha,
                         "all original replica WAVs are disjoint from historical/primary PCM")
                    all_wav_sha.add(digest)
            counts = {**synthcounts, **humancounts}
            gate, missing = primary.source_gate(counts)
            report.update(source_coverage=counts, source_gate=gate, source_shortfall=missing,
                          panel_gate="INSUFFICIENT_SOURCE", ancestry_audit=dict(
                              old_pcm_count=len(old_pcm), old_family_count=len(old_groups),
                              old_canonical_people_count=len(old_people),
                              original_replica_rows=len(all_rows), retained_original_wavs=len(all_wav_sha),
                              source_identity_sha256=sha(json.dumps(sorted((r["id"], r["pcm_sha256"],
                                  canonical_person(r)) for r in all_rows), separators=(",", ":")).encode()),
                              source_primary_disjoint=True))
            if missing:
                return
            import numpy as np
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            torch.backends.mkldnn.enabled = False
            models, training_sources = primary.snapshots(inputs, meta, torch, np)
            need({k: value[-1] for k, value in models.items()} == permit["model_pt_sha256"],
                 "replication uses the identical four authenticated primary model tensors")
            primary.reject_train_overlap(all_rows, training_sources)
            # Track real calls independently of arrays, including an inference
            # that raises before its output can be saved. A STOP must not report
            # zero forwards merely because an incomplete call produced no array.
            for name, values in models.items():
                def before_forward(_model, _args, role=name):
                    forward_calls[role] += 1
                forward_handles.append(values[0].register_forward_pre_hook(before_forward))
            rows = []
            for key, cohort, admitted in (("synth_source", "synthetic", synthrows),
                                         ("natural_source", "natural", humans)):
                for row in admitted:
                    scores = common.infer(archives[key].read("wav/" + row["id"] + ".wav"),
                                          row, models, arrays, torch, np)
                    rows.append(dict(id=row["id"], group=row["source_group_id"], cohort=cohort,
                        cell=row["cell_id"], positive=row["observed_class"] == 2,
                        tier=row["tier"], pcm_sha256=row["pcm_sha256"], gains=scores))
                    report["actual_forwards"] = sum(forward_calls.values())
            need(len(rows) == len({r["pcm_sha256"] for r in rows})
                 and sum(forward_calls.values()) == len(arrays) == 12 * len(rows),
                 "complete four-model three-gain original replication calls and saved arrays")
            decision = primary.old_decision.decide(rows)
            report.update(decision, records=rows, model_pt_sha256={k: value[-1] for k, value in models.items()},
                          research_replication_passed=decision["panel_gate"] == "PASS",
                          candidate_engineering_validation_allowed=decision["panel_gate"] == "PASS")
    except Exception as exc:
        report.update(panel_gate="STOP", error=str(exc), actual_forwards=sum(forward_calls.values()),
                      research_replication_passed=False, candidate_engineering_validation_allowed=False)
        raise
    finally:
        for handle in forward_handles:
            handle.remove()
        report.update(actual_forwards=sum(forward_calls.values()), saved_array_count=len(arrays),
                      model_forward_calls=forward_calls)
        if arrays:
            target = out / "ALL_FRAME_LOGITS.npz"
            np.savez_compressed(target, **arrays)
            report["logits_sha256"] = sha(target.read_bytes())
        (out / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "score"))
    parser.add_argument("spec", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("history", type=Path)
    parser.add_argument("authorization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-parent", required=True)
    args = parser.parse_args()
    if args.mode == "fetch":
        fetch(args.spec, args.inputs, args.history, args.authorization, args.prepared_parent)
    else:
        need(args.output is not None, "new output directory required")
        run(args.spec, args.inputs, args.history, args.authorization, args.output, args.prepared_parent)
