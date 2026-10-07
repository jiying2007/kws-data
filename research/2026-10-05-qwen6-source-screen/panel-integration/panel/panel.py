"""Deterministic, preparation-only panel compiler. No synthesis or model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from vendor.quality_gates import coverage_admission, human_truth, identity_audit, normalized_actual, require

ROOT = Path(__file__).resolve().parent
SOURCES = (("Ryan", "train"), ("Aiden", "train"), ("Ono_Anna", "dev"), ("Sohee", "heldout"))
CELLS = (("K1", "你好小窝"), ("K2", "小窝小窝"),
         ("near_K1", "你好小屋"), ("near_K2", "小屋小屋"),
         ("repetition", "你好你好"), ("ordinary_OOV", "今天天气很好"))
MODEL = {"repository": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
         "revision": "85e237c12c027371202489a0ec509ded67b5e4b5",
         "language": "Chinese", "reference_cloning": False}
POLICY = {"policy_id": "prospective-six-cell-coverage-v1", "frozen": True,
          "minima": {"K1": 1, "K2": 1},
          "nonwake_minima": {"你好小屋": 1, "小屋小屋": 1, "你好你好": 1,
                             "out_of_vocabulary": 1},
          "vocabulary": "你好小窝屋"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_history(path=ROOT / "declared-history.json"):
    return json.loads(Path(path).read_text())


def compile_panel():
    """Freeze source roles and every attempt ID before any audio is generated."""
    history = load_history()
    plan = {"schema": "prospective-voice-panel-v1", "generator": MODEL,
            "execution_status": "NOT_EXECUTED", "purpose": "preparation_only",
            "future_endpoint_intent": "balanced small KWS dataset after independent admission",
            "history_sha256": digest(history), "policy": POLICY,
            "sources": [], "rows": []}
    for voice, role in SOURCES:
        identity = "Qwen3-stock:" + voice
        plan["sources"].append({"voice_identity": identity, "prospective_role": role,
                                "lineage_status": "declared_stock_preset"})
        for stage, cells in (("screen", CELLS[:2] if role != "heldout" else ()), ("coverage", CELLS)):
            for cell, text in cells:
                plan["rows"].append({"id": f"{stage}-{voice.lower()}-{cell}",
                    "stage": stage, "voice_identity": identity, "source_group": identity,
                    "prospective_role": role, "cell": cell, "intended_text": text,
                    "actual_text": None, "wav_sha256": None, "pcm_sha256": None,
                    "review": {"status": "pending", "independent_human": False,
                               "complete": False}, "asr_results": [None, None],
                    "generation_status": "NOT_GENERATED", "actual_use": "none",
                    "train_use": False, "ctc_target": None,
                    "exposure": "UNKNOWN", "reference_sha256": None,
                    "future_use": "exposed_pronunciation_source_screen" if stage == "screen"
                                  else "independent_actual_label_coverage_candidate"})
    return {"plan": plan, "frozen_plan_sha256": digest(plan)}


def scoped_identity(rows, history):
    """Never promote missing bounded history into project novelty evidence.

    Upstream's documented contract requires complete history. This adapter has a
    known-incomplete inventory and renames that relative result explicitly.
    Its declared lineage attestations are not acoustic identity authentication.
    """
    def omit_empty_reference(row):
        # The inherited restriction checker represents no reference by absence.
        return {k: v for k, v in row.items() if not (k == "reference_sha256" and v is None)}
    audit = identity_audit([omit_empty_reference(r) for r in rows],
                           [omit_empty_reference(r) for r in history.get("records", [])])
    scoped = []
    for row in audit["rows"]:
        item = {k: v for k, v in row.items()
                if k not in ("unseen_voice", "strong_independence")}
        item["absent_from_declared_history"] = row["unseen_voice"]
        item["project_novelty"] = "UNKNOWN"
        item["donor_pretraining_overlap"] = "UNKNOWN"
        scoped.append(item)
    return {"status": audit["status"], "conflicts": audit["conflicts"], "rows": scoped,
            "history_scope": history.get("scope", "MISSING"),
            "identity_evidence_scope": "declared catalog metadata or supplied attestations; not audio-authenticated",
            "complete_project_history_verified": False,
            "project_unseen_qualified": False,
            "generated_screen_exposure_policy": "EXPOSED"}


def _identity_row(row, split=None):
    return {**row, "split": split or row["prospective_role"],
            "lineage_status": "verified", "exposure": row.get("exposure", "UNKNOWN"),
            "use": "preparation"}


def assess(panel, observations=None, history=None, require_data=False,
           require_project_unseen=False):
    """Exit 0 means plan accepted only; 1 unmet evidence; 2 malformed input.

    Observations are explicit evidence records joined by frozen attempt ID. They
    never overwrite the plan. No output of this tool authorizes any execution.
    """
    scope = {"training_authorized": False, "generation_authorized": False,
             "asr_authorized": False, "kws_authorized": False,
             "executed_dataset_qualified": False, "fresh_validation_qualified": False,
             "project_unseen_qualified": False, "donor_pretraining_overlap": "UNKNOWN"}
    try:
        history = load_history() if history is None else history
        plan = panel["plan"]
        require(digest(plan) == panel["frozen_plan_sha256"], "frozen plan digest mismatch")
        require(plan["history_sha256"] == digest(history), "declared history digest mismatch")
        require(plan["schema"] == "prospective-voice-panel-v1", "unknown schema")
        require(plan["generator"] == MODEL, "this panel requires pinned Qwen stock Chinese route")
        require(plan["execution_status"] == "NOT_EXECUTED" and plan["purpose"] == "preparation_only",
                "plan is preparation only; store execution evidence separately")
        require(plan["policy"] == POLICY, "coverage policy changed")
        sources = plan["sources"]
        require(len(sources) == 4, "four source identities required")
        roles = {s["voice_identity"]: s["prospective_role"] for s in sources}
        require(len(roles) == 4 and set(roles.values()) == {"train", "dev", "heldout"},
                "distinct train/dev/heldout source roles required")
        require(roles == {"Qwen3-stock:" + voice: role for voice, role in SOURCES},
                "fixed source roles changed; do not select or swap after generation")
        rows = plan["rows"]
        require(len({r["id"] for r in rows}) == len(rows), "duplicate attempt ID")
        for r in rows:
            require(r["stage"] in ("screen", "coverage"), "unknown stage")
            require(r["voice_identity"] in roles and r["source_group"] == r["voice_identity"]
                    and r["prospective_role"] == roles[r["voice_identity"]], "source role mismatch")
            require(r["reference_sha256"] is None, "no reference cloning in this route")
            require(r["actual_text"] is None and not r["wav_sha256"] and not r["pcm_sha256"]
                    and r["generation_status"] == "NOT_GENERATED" and r["actual_use"] == "none"
                    and r["review"] == {"status": "pending", "independent_human": False,
                                         "complete": False} and r["asr_results"] == [None, None]
                    and r["exposure"] == "UNKNOWN" and r["train_use"] is False
                    and r["ctc_target"] is None, "plan must not invent executed evidence")
        missing = []
        for identity in roles:
            for stage, cells in (("screen", CELLS[:2] if roles[identity] != "heldout" else ()), ("coverage", CELLS)):
                found = [(r["cell"], r["intended_text"]) for r in rows
                         if r["voice_identity"] == identity and r["stage"] == stage]
                if sorted(found) != sorted(cells):
                    missing.append({"source": identity, "stage": stage,
                                    "required": list(cells), "found": found})
        observations = [] if observations is None else observations
        require(type(observations) is list, "observations must be a list")
        require(len({r["id"] for r in observations}) == len(observations), "duplicate observation ID")
        lookup = {r["id"]: r for r in rows}
        require(all(o["id"] in lookup for o in observations), "observation absent from frozen plan")
        evidence = {o["id"]: o for o in observations}
        materialized, screened, screen_checks = [], [], []
        for row in rows:
            observation = evidence.get(row["id"], {})
            allowed = {"id", "actual_text", "wav_sha256", "pcm_sha256", "review", "exposure",
                       "reference_sha256", "lineage_ids", "generation_receipt_sha256",
                       "asr_results", "evidence_basis", "generator", "voice_identity"}
            require(not (set(observation) - allowed), "observation contains plan-changing fields")
            if observation:
                require(observation.get("generator") == MODEL, "observation generator/pin mismatch")
                require(observation.get("voice_identity") == row["voice_identity"],
                        "observed identity differs from frozen source")
                require(observation.get("reference_sha256") is None, "stock-only observation required")
                for field in ("wav_sha256", "pcm_sha256", "generation_receipt_sha256"):
                    value = observation.get(field)
                    require(value is None or (type(value) is str and len(value) == 64
                            and all(c in "0123456789abcdef" for c in value)), "invalid " + field)
            merged = {**row, **observation}
            if not observation.get("generation_receipt_sha256") or observation.get("evidence_basis") != "observed":
                merged["review"] = {"status": "pending", "complete": False, "independent_human": False}
            if row["stage"] == "screen":
                reviewed = (human_truth(merged) is not None
                    and normalized_actual(merged["actual_text"]) == normalized_actual(row["intended_text"])
                    and merged["review"].get("continuous_phrase") is True
                    and merged["review"].get("last_syllable_complete") is True
                    and bool(merged.get("wav_sha256") or merged.get("pcm_sha256")))
                screen_checks.append({"id": row["id"], "actual_text": merged.get("actual_text"),
                    "intended_text": row["intended_text"],
                    "status": "HUMAN_SCREEN_ATTESTATION_COMPLETE" if reviewed else "PENDING_OR_REJECTED",
                    "train_use": False, "ctc_target": None})
                if (observation.get("generation_receipt_sha256")
                        and (observation.get("wav_sha256") or observation.get("pcm_sha256"))
                        and observation.get("evidence_basis") == "observed"):
                    # Screen audio is exposed preparation, never a coverage row.
                    screened.append({**_identity_row(merged, "regression"), "exposure": "EXPOSED",
                                     "use": "regression", "restriction": "regression_only"})
            else:
                materialized.append(_identity_row(merged))
        declarations = [{"source_group": identity, "split": role, "role": "balanced"}
                        for identity, role in roles.items()]
        coverage = coverage_admission(materialized, declarations, POLICY, digest(POLICY))
        identity = scoped_identity(materialized, {**history, "records": history.get("records", []) + screened})
        reasons = []
        split_conflicts = [c for c in identity["conflicts"] if c["kind"] == "SHARED_IDENTITY_ACROSS_SPLITS"]
        if split_conflicts:
            reasons.append("DECLARED_IDENTITY_SPLIT_CONFLICT")
        if missing:
            reasons.append("PLANNED_PER_SOURCE_COVERAGE_INCOMPLETE")
        if require_data and not coverage["balanced_admission"]:
            reasons.append("ACTUAL_PER_SOURCE_COVERAGE_INCOMPLETE")
        if require_data and identity["status"] == "FAIL":
            reasons.append("IDENTITY_OR_EXPOSURE_CONFLICT")
        if require_data:
            reasons.append("PREPARATION_ONLY_CANNOT_GRANT_EXECUTED_DATASET_QUALIFICATION")
        if require_project_unseen:
            reasons.append("COMPLETE_PROJECT_HISTORY_NOT_VERIFIED")
        # Split assignment is a plan property, not a claim about donor weights.
        known_d20 = history.get("a20_supervised_adaptation", {}).get("source_counts", {})
        evaluation_sources = {k for k, role in roles.items() if role != "train"}
        return {"status": "PLAN_ACCEPTED_PREPARATION_ONLY" if not reasons else "REJECTED",
                "plan_valid": not missing and not split_conflicts,
                "reasons": reasons, "missing_planned_cells": missing,
                "counts": {"screen": sum(r["stage"] == "screen" for r in rows),
                           "coverage": sum(r["stage"] == "coverage" for r in rows),
                           "observations": len(observations)},
                "coverage": coverage, "identity": identity,
                "screen": {"rows": screen_checks,
                    "all_six_attestations_complete": len(screen_checks) == 6 and all(
                        r["status"] == "HUMAN_SCREEN_ATTESTATION_COMPLETE" for r in screen_checks),
                    "scope": "independent human pronunciation and endpoint review attestations; no acoustic execution"},
                "planned_adaptation_source_disjoint": not bool(split_conflicts),
                "heldout_screen_attempts": sum(r["stage"] == "screen" and r["prospective_role"] == "heldout" for r in rows),
                "known_A20_D20_adaptation_disjoint": not bool(evaluation_sources & set(known_d20))
                    if known_d20 else None,
                "full_candidate_adaptation_lineage": "NOT_ESTABLISHED_UNTIL_CHECKPOINT_PINNED",
                "ordinary_OOV_scope": "outside declared comparison vocabulary only; no CTC target emitted",
                "plan_pin_scope": "integrity reference; not proof of external preregistration",
                **scope}, 1 if reasons else 0
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return {"status": "INVALID_INPUT", "plan_valid": False, "error": str(error), **scope}, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_command = commands.add_parser("compile")
    compile_command.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("panel", type=Path)
    check.add_argument("--observations", type=Path)
    check.add_argument("--require-data", action="store_true")
    check.add_argument("--require-project-unseen", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            args.output.write_text(json.dumps(compile_panel(), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            return 0
        panel = json.loads(args.panel.read_text())
        observations = json.loads(args.observations.read_text()) if args.observations else None
        report, code = assess(panel, observations, require_data=args.require_data,
                              require_project_unseen=args.require_project_unseen)
    except (OSError, ValueError) as error:
        report, code = {"status": "INVALID_INPUT", "error": str(error)}, 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
