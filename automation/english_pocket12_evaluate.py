"""Pure saved-byte Pocket12 feasibility evaluation; never admits training data."""
from __future__ import annotations

import ast
from collections import Counter
import importlib.util
from pathlib import Path
import re
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_pocket12_observe as observe

HERE = Path(__file__).resolve().parent
SCHEMA = "english-pocket12-evaluation-v1"
FLAGS = {"training_allowed": False, "permanently_excluded_from_training": True,
         "diagnostic_only": True, "human_gold": False, "final_native_us_pronunciation_verified": False,
         "independent_real_quality_claims_allowed": False, "automatic_expansion_allowed": False}
need, sha, regular = observe.need, observe.sha, observe.regular


def frozen_reference():
    path = HERE / "english_context64_policy_v2_reference.py"
    need(sha(regular(path)) == observe.HELPER_PINS["automation/" + path.name], "FROZEN_REFERENCE_POLICY")
    spec = importlib.util.spec_from_file_location("pocket12_frozen_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE = frozen_reference()
RULES = REFERENCE.RULES
normalized_text, normalized_phones = REFERENCE.normalized_text, REFERENCE.normalized_phones
CONTRAST_PATTERNS = {"hey marrow": r"h?eɪmær(?:oʊ|ou|o)",
                     "hey bosen": r"h?eɪb(?:oʊ|ou|o)s[əɐʌ]n"}


def source_check(plan=None):
    checked = observe.source_check(plan)
    if plan is not None:
        need(plan.get("evaluate_code_sha256") == sha(regular(Path(__file__))), "EVALUATOR_SOURCE_PIN")
    return checked | {"evaluator_source_check": "PASS", "rules_sha256": REFERENCE.RULES_SHA256,
                      "permanently_excluded_from_training": True}


def whole_support_geometry(samples):
    need(type(samples) is int and 0 < samples <= 480000, "WHOLE_SHORT_INTEGER_SAMPLES")
    path = HERE / "english_synthetic192_training_core.py"
    raw = regular(path)
    need(sha(raw) == observe.HELPER_PINS["automation/" + path.name], "FROZEN_INTEGER_GEOMETRY")
    selected = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)
                and n.name in ("frame_ends", "positive_ends")]
    need([n.name for n in selected] == ["frame_ends", "positive_ends"], "FROZEN_GEOMETRY_FUNCTIONS")
    scope = {"need": need}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "pocket12_integer_geometry", "exec"), scope)
    ends = scope["positive_ends"](38400, samples, 38400 + samples + 16000)
    return {"prefix_samples": 38400, "suffix_samples": 16000, "whole_support_ends": ends,
        "whole_support_windows": len(ends), "geometry_usable_positive": len(ends) >= 3,
        "synthetic_geometry_only": True, "new_audio_generated": False, "feature_or_model_executed": False}


def candidate_from_evidence(texts, phone_tokens, whole_support_windows):
    """The classifier receives observations and integer geometry, never intent."""
    need(type(texts) is dict and set(texts) == set(observe.TEXT_PROVIDERS), "EXACT_THREE_ASR_SLOTS")
    normalized = {p: None if text is None else normalized_text(text) for p, text in texts.items()}
    phone = None if phone_tokens is None else normalized_phones(phone_tokens)
    hypotheses = [] if phone is None else [h for h, (_, pattern) in RULES["phone_patterns"].items()
                                           if re.fullmatch(pattern, phone)]
    votes, vetoes = {}, {}
    for kind in (1, 2):
        votes[str(kind)] = sum(t in RULES["aliases"][str(kind)] for t in normalized.values())
        conflicts = RULES["aliases"][str(3 - kind)] + RULES["contradictory_core_phrases"][str(kind)]
        vetoes[str(kind)] = [p for p, text in normalized.items() if text is not None
            and any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text) for a in conflicts)]
    result = {"candidate_positive": False, "candidate_class": None, "candidate_hypothesis": None,
        "reason": "MISSING_OBSERVATION", "normalized_texts": normalized, "normalized_phones": phone,
        "whole_phone_hypotheses": hypotheses, "class_votes": votes, "contradictory_providers": vetoes,
        "target_phone_pattern_anywhere": phone is not None and any(re.search(p, phone)
            for _, p in RULES["phone_patterns"].values()), "whole_support_windows": whole_support_windows,
        "candidate_evidence_scope": "SAME_ENTIRE_UNCROPPED_SHORT_PCM", **FLAGS}
    if any(t is None for t in normalized.values()) or phone is None:
        return result
    if len(hypotheses) != 1:
        result["reason"] = "NO_UNIQUE_WHOLE_PHONE_HYPOTHESIS"
        return result
    hypothesis = hypotheses[0]
    kind = RULES["phone_patterns"][hypothesis][0]
    if vetoes[str(kind)]:
        result["reason"] = "FROZEN_CONTRADICTORY_WAKE_VETO"
    elif votes[str(kind)] != 3:
        result["reason"] = "THREE_FROZEN_ALIAS_VOTES_REQUIRED"
    elif type(whole_support_windows) is not int or whole_support_windows < 3:
        result["reason"] = "FEWER_THAN_THREE_WHOLE_SUPPORT_WINDOWS"
    else:
        result.update(candidate_positive=True, candidate_class=kind, candidate_hypothesis=hypothesis,
                      reason="THREE_ASR_AND_SAME_INPUT_WHOLE_PHONE_SUPPORT")
    return result


def _counts(source, packets):
    stages = {"tts": {"denominator": 12, "actual_requests": sum(r["tts_attempts"] for r in source["rows"]),
        "returned": sum(r["tts_returned"] for r in source["rows"]),
        "completed": sum(r["generation_status"] == "success" for r in source["rows"]),
        "failed_count": sum(r["generation_status"] == "failed" for r in source["rows"]),
        "not_started_count": sum(r["generation_status"] == "not_started" for r in source["rows"]), "unknown_count": 0}}
    for provider, packet in packets.items():
        s = packet["summary"]
        stages[provider] = {"denominator": 12, "actual_requests": s["actual_forward_attempts"],
            "returned": s["native_forward_returns"], "completed": s["completed_count"],
            **{k: s.get(k) for k in ("failed_count", "not_started_count", "unknown_count")},
            "attempts_lower_bound": s["attempts_lower_bound"], "attempts_upper_bound": s["attempts_upper_bound"]}
    actual = [s["actual_requests"] for s in stages.values()]
    return {"planned_tts_requests": 12, "planned_observer_requests": 48, "planned_total_requests": 60,
        "stages": stages, "actual_total_requests": sum(actual) if all(n is not None for n in actual) else None,
        "actual_total_lower_bound": sum(s.get("attempts_lower_bound", s["actual_requests"]) for s in stages.values()),
        "actual_total_upper_bound": sum(s.get("attempts_upper_bound", s["actual_requests"]) for s in stages.values()),
        "complete_observations": sum(p["summary"]["completed_count"] for p in packets.values()),
        "call_unit": "one high-level TTS generation, ASR decode or XLSR native forward; not internal tensor operations"}


def incomplete_without_source(run_identity):
    import english_pocket12_generate as generate
    return {"schema": SCHEMA, "run": run_identity, "status": "INCOMPLETE", "outcome": "INCOMPLETE",
        "reason": "SOURCE_TERMINAL_NOT_AVAILABLE", "denominator": 12,
        "counts": {"planned_tts_requests": 12, "planned_observer_requests": 48, "planned_total_requests": 60,
            "actual_total_requests": None, "actual_total_lower_bound": 0, "actual_total_upper_bound": 60,
            "complete_observations": 0,
            "stages": {p: {"denominator": 12, "actual_requests": None, "returned": None, "completed": 0,
                "attempts_lower_bound": 0, "attempts_upper_bound": 12, "failed_count": 0,
                "not_started_count": None, "unknown_count": 12} for p in ("tts", *observe.PROVIDERS)}},
        "rows": [{**r, "observation_status": "unknown", "candidate": None, "gate_passed": False, **FLAGS}
                 for r in generate.recipe()["rows"]], "gates": {"source_and_observations_complete": False}, **FLAGS}


def evaluate(source_root, observer_roots, plan, run_identity):
    source_check(plan)
    import english_pocket12_generate as generate
    if not (Path(source_root) / "SOURCE.json").is_file():
        return incomplete_without_source(run_identity)
    source, successful = generate.validate_source(source_root, run_identity)
    packets = {}
    for provider in observe.PROVIDERS:
        if len(successful) == 12:
            packets[provider] = observe.readback(provider=provider, source_root=source_root,
                observer_root=observer_roots.get(provider), plan=plan, run_identity=run_identity)
        else:
            packets[provider] = {"summary": {"actual_forward_attempts": None, "native_forward_returns": None,
                "completed_count": 0, "attempts_lower_bound": 0, "attempts_upper_bound": 12,
                "failed_count": 0, "not_started_count": None, "unknown_count": 12},
                "rows": [{"id": r["id"], "status": "missing", "raw_transcript": None, "phone_tokens": None}
                         for r in source["rows"]]}
    counts, evaluated = _counts(source, packets), []
    for index, row in enumerate(source["rows"]):
        texts = {p: packets[p]["rows"][index]["raw_transcript"] for p in observe.TEXT_PROVIDERS}
        tokens = packets["phoneme"]["rows"][index]["phone_tokens"]
        geometry = whole_support_geometry(row["pcm_samples"]) if row["generation_status"] == "success" else None
        candidate = candidate_from_evidence(texts, tokens, geometry["whole_support_windows"] if geometry else None)
        # Intent is consulted only after the blind candidate above is fixed.
        all_complete = all(packets[p]["rows"][index]["status"] == "success" for p in observe.PROVIDERS)
        control = {"machine_corroborated": False}
        if row["kind"] == "positive":
            passed = candidate["candidate_positive"] and candidate["candidate_class"] == row["requested_class"]
        elif row["kind"] == "contrast":
            pattern = CONTRAST_PATTERNS[normalized_text(row["requested_text"])]
            phone_match = tokens is not None and bool(re.fullmatch(pattern, candidate["normalized_phones"]))
            control = {"machine_corroborated": all_complete and phone_match,
                "expected_whole_phone_pattern": pattern, "expected_whole_phone_match": phone_match,
                "all_three_blind_asr_captures_complete": all_complete,
                "asr_may_spell_positive_aliases": True}
            passed = control["machine_corroborated"] and not candidate["candidate_positive"]
        else:
            exact = all(t == normalized_text(row["requested_text"]) for t in candidate["normalized_texts"].values())
            absent = tokens is not None and not candidate["target_phone_pattern_anywhere"]
            control = {"machine_corroborated": all_complete and exact and absent,
                "all_three_exact_text": exact, "no_target_phone_pattern": absent}
            passed = control["machine_corroborated"] and not candidate["candidate_positive"]
        evaluated.append({**row, "observations": {p: packets[p]["rows"][index] for p in observe.PROVIDERS},
            "observation_status": "complete" if all_complete else "incomplete", "geometry": geometry,
            "candidate": candidate, "control": control, "gate_passed": bool(all_complete and passed), **FLAGS})
    positives = [r for r in evaluated if r["kind"] == "positive"]
    contrasts = [r for r in evaluated if r["kind"] == "contrast"]
    ordinary = [r for r in evaluated if r["kind"] == "ordinary"]
    voice_support = {voice: sorted({r["candidate"]["candidate_class"] for r in positives
        if r["voice_id"] == voice and r["gate_passed"]}) for voice in ("george", "mary")}
    gates = {"source_and_observations_complete": len(successful) == 12 and counts["complete_observations"] == 48
                and counts["actual_total_requests"] == 60,
        "all_four_positives_supported": len(positives) == 4 and all(r["gate_passed"] for r in positives),
        "both_voices_support_both_classes": all(v == [1, 2] for v in voice_support.values()),
        "all_four_contrasts_corroborated_and_zero_positive": len(contrasts) == 4 and all(r["gate_passed"] for r in contrasts),
        "all_four_ordinary_exact_and_no_target_phone": len(ordinary) == 4 and all(r["gate_passed"] for r in ordinary)}
    complete, passed = gates["source_and_observations_complete"], all(gates.values())
    status = "PASS_POCKET12_SOURCE_FEASIBILITY" if passed else "STOP_POCKET12_SOURCE_FEASIBILITY_FAILED" if complete else "INCOMPLETE"
    return {"schema": SCHEMA, "run": run_identity, "status": status, "outcome": status,
        "denominator": 12, "counts": counts, "gates": gates, "voice_supported_classes": voice_support,
        "group_counts": {kind: {"denominator": 4, "passed": sum(r["gate_passed"] for r in group),
            "candidate_positive": sum(r["candidate"]["candidate_positive"] for r in group),
            "machine_corroborated": None if kind == "positive" else sum(r["control"]["machine_corroborated"] for r in group)}
            for kind, group in (("positive", positives), ("contrast", contrasts), ("ordinary", ordinary))},
        "positive_hypothesis_counts": {h: sum(r["candidate"]["candidate_hypothesis"] == h for r in positives)
                                       for h in RULES["phone_patterns"]},
        "observed_whole_phone_hypothesis_counts": dict(Counter(h for r in evaluated
            for h in r["candidate"]["whole_phone_hypotheses"])),
        "reference_policy_sha256": observe.HELPER_PINS["automation/english_context64_policy_v2_reference.py"],
        "rules_sha256": REFERENCE.RULES_SHA256, "rows": evaluated,
        "source_sha256": sha(regular(Path(source_root) / "SOURCE.json")),
        "next_action": "REVIEW_NEW_DISJOINT_TRAINING_DESIGN" if passed else "STOP_NO_RETRY_OR_EXPANSION",
        "model_trained": False, "model_quality_improved_claimed": False, "new_model_calls": 0, **FLAGS}


if __name__ == "__main__":
    need(sys.argv[1:] == ["--source-check"], "PURE_EVALUATOR_SOURCE_CHECK_ONLY")
    print(json.dumps(source_check(), ensure_ascii=False, sort_keys=True))
