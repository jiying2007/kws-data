"""Read-only evaluation of the fixed exact-child context controls.

Candidate classification is blind to requested class, cell and hypothesis.
Parent-conditioned phones and child-only ASRs have distinct input identities.
Every item in this diagnostic probe is permanently excluded from training.
Only standard-library modules and pinned pure readers are used here.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.util
from itertools import groupby
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
SCHEMA = "english-kokoro-context64-evaluation-v1"
CAMPAIGN = "kokoro-exact-child-context-controls64-v1"
DESIGN_SHA256 = "d0460da69b64aed66aec8526d253ed6b8a4762eebff234a862331207dd5b99f6"
REFERENCE_SHA256 = "fde516d9f1f2457cdb0047e8d8aee6a667c68fa6ea21f83d22f135e4bc9069da"
GEOMETRY_SHA256 = "0896d375f47318adbc42f63835cb85fc77441ae2472d9df8212dfca4dd7e62c0"
TEXT_PROVIDERS = ("qwen", "whisper", "parakeet")
PHONE_PROVIDERS = ("phoneme", "phoneme_lv60")
KINDS = ("positive", "contrast", "ordinary", "outside")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def regular(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and 0 < path.stat().st_size <= 2 * 1024**2, "REGULAR_SOURCE_FILE")
    return path.read_bytes()


def module_at(name, path, expected_sha):
    path = Path(path)
    need(sha(regular(path)) == expected_sha, "PINNED_PURE_SOURCE: " + path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reference = module_at("context64_frozen_reference", HERE / "english_context64_policy_v2_reference.py", REFERENCE_SHA256)
# Never call the old same-PCM verify_observation, decide_label or admission entry.
RULES = _reference.RULES
normalized_text = _reference.normalized_text
normalized_phones = _reference.normalized_phones


def geometry_functions():
    raw = regular(HERE / "english_synthetic192_training_core.py")
    need(sha(raw) == GEOMETRY_SHA256, "FROZEN_INTEGER_GEOMETRY_SOURCE")
    tree = ast.parse(raw)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("frame_ends", "positive_ends")]
    need([n.name for n in selected] == ["frame_ends", "positive_ends"], "EXACT_GEOMETRY_FUNCTIONS")
    scope = {"need": need}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "frozen_integer_geometry", "exec"), scope)
    return scope["positive_ends"]


_positive_ends = geometry_functions()


def whole_support_geometry(samples):
    need(type(samples) is int and 0 < samples <= 192000, "CHILD_INTEGER_SAMPLE_COUNT")
    ends = _positive_ends(38400, samples, 38400 + samples + 16000)
    return {"prefix_samples": 38400, "suffix_samples": 16000, "whole_support_ends": ends,
            "whole_support_windows": len(ends), "geometry_usable_positive": len(ends) >= 3,
            "new_audio_generated": False, "feature_or_model_executed": False}


def whole_matches(value):
    if value is None:
        return []
    return [name for name, (_, pattern) in RULES["phone_patterns"].items() if re.fullmatch(pattern, value)]


def terminal_matches(value):
    if value is None:
        return []
    return [name for name, (_, pattern) in RULES["phone_patterns"].items() if re.search(pattern + r"\Z", value)]


def any_target(value):
    return value is not None and any(re.search(pattern, value) for _, pattern in RULES["phone_patterns"].values())


def candidate_from_evidence(texts, fixed_region_tokens, full_parent_tokens, whole_support_windows):
    """No request metadata enters this classifier; all evidence is blind."""
    need(type(texts) is dict and set(texts) == set(TEXT_PROVIDERS), "EXACT_THREE_TEXT_OBSERVATIONS")
    normalized = {p: None if texts[p] is None else normalized_text(texts[p]) for p in TEXT_PROVIDERS}
    fixed = None if fixed_region_tokens is None else normalized_phones(fixed_region_tokens)
    full = None if full_parent_tokens is None else normalized_phones(full_parent_tokens)
    matches, suffixes = whole_matches(fixed), terminal_matches(full)
    votes, vetoes = {}, {}
    for kind in (1, 2):
        votes[str(kind)] = sum(value in RULES["aliases"][str(kind)] for value in normalized.values())
        conflicts = RULES["aliases"][str(3 - kind)] + RULES["contradictory_core_phrases"][str(kind)]
        vetoes[str(kind)] = [p for p, text in normalized.items() if text is not None
            and any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text) for alias in conflicts)]
    result = {"candidate_positive": False, "candidate_class": None, "candidate_hypothesis": None,
        "candidate_tier": "unknown", "reason": "MISSING_OR_INCOMPLETE_OBSERVATION",
        "normalized_child_texts": normalized, "class_votes": votes, "contradictory_providers": vetoes,
        "reference_fixed_region_normalized": fixed, "reference_full_parent_normalized": full,
        "reference_whole_hypotheses": matches, "reference_terminal_hypotheses": suffixes,
        "whole_support_windows": whole_support_windows,
        "phonetic_evidence_scope": "PARENT_FULL_INPUT_FIXED_CHILD_REGION",
        "text_evidence_scope": "EXACT_CHILD_INPUT_ONLY", "child_independent_phone_silver": False,
        "training_allowed": False, "training_weight": 0.0}
    if any(value is None for value in normalized.values()) or fixed is None or full is None:
        return result
    if len(matches) != 1 or suffixes != matches:
        result["reason"] = "NO_UNIQUE_SAME_REFERENCE_REGION_AND_TERMINAL_HYPOTHESIS"
        return result
    hypothesis = matches[0]
    kind = RULES["phone_patterns"][hypothesis][0]
    if vetoes[str(kind)]:
        result["reason"] = "FROZEN_CONTRADICTORY_WAKE_VETO"
    elif votes[str(kind)] != 3:
        result["reason"] = "THREE_CHILD_ALIAS_VOTES_REQUIRED"
    elif type(whole_support_windows) is not int or whole_support_windows < 3:
        result["reason"] = "FEWER_THAN_THREE_WHOLE_SUPPORT_WINDOWS"
    else:
        result.update(candidate_positive=True, candidate_class=kind, candidate_hypothesis=hypothesis,
            candidate_tier="context_supported_weak", reason="THREE_CHILD_TEXTS_AND_FIXED_PARENT_REGION_SUPPORT")
    return result


def fixed_child_region(frame_ids, by_id, start, end, parent_samples):
    need(type(parent_samples) is int and type(start) is int and type(end) is int
         and 0 <= start < end <= parent_samples, "FIXED_CHILD_PARENT_SAMPLE_BOUNDS")
    frames = parent_samples
    for kernel, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        frames = (frames - kernel) // stride + 1
    need(type(frame_ids) is list and len(frame_ids) == frames and 0 < frames <= 2000
         and all(type(i) is int and 0 <= i < 392 for i in frame_ids), "COMPLETE_PARENT_FRAME_IDS")
    need(type(by_id) is dict and set(by_id) == set(range(392))
         and all(type(i) is int and type(v) is str and v for i, v in by_id.items()), "FIXED392_REVERSE_VOCAB")
    selected = [t for t in range(frames) if 320 * t >= start and 320 * t + 400 <= end]
    tokens = [by_id[number] for number, _ in groupby(frame_ids[t] for t in selected) if number not in (0, 1, 2, 3)]
    normalized = normalized_phones(tokens)
    return {"input_scope": "PARENT_FULL_16K_PCM", "selection_rule": "320*t >= child_start AND320*t+400 <= child_end",
        "child_start_parent16k": start, "child_end_parent16k": end,
        "frame_start": selected[0] if selected else None,
        "frame_stop_exclusive": selected[-1] + 1 if selected else None,
        "selected_frame_count": len(selected), "phone_tokens": tokens, "normalized_phones": normalized,
        "whole_pattern_hypotheses": whole_matches(normalized), "target_pattern_anywhere": any_target(normalized),
        "nominal_convolution_support_only": True, "transformer_context_is_parent_global": True,
        "independent_child_forward": False, "duration_boundary_is_acoustic_gold": False}


def checked_design(plan):
    design = plan.get("design")
    need(type(design) is dict and sha(canonical(design)) == DESIGN_SHA256, "FROZEN_BEFORE_CALLS_CONTROL_DESIGN")
    need(design["campaign_id"] == CAMPAIGN and design["reference_provider"] == "phoneme"
         and design["diagnostic_provider"] == "phoneme_lv60"
         and plan.get("reference_policy_sha256") == REFERENCE_SHA256, "FIXED_REFERENCE_AND_DIAGNOSTIC_ROLES")
    return design


def check_roster(manifest, design, run):
    need(manifest.get("run") == run, "ACTUAL_SOURCE_RUN_BINDING")
    parents, children = manifest.get("parents"), manifest.get("children")
    need(type(parents) is list and len(parents) == 48 and type(children) is list and len(children) == 64,
         "FULL48_PARENT64_CHILD_DENOMINATOR")
    need(len({p["id"] for p in parents}) == 48 and len({c["id"] for c in children}) == 64,
         "UNIQUE_PARENT_CHILD_IDS")
    by_parent = {p["id"]: p for p in parents}
    need(Counter(p["kind"] for p in parents) == {"positive": 16, "contrast": 16, "ordinary": 16}
         and Counter(c["kind"] for c in children) == {k: 16 for k in KINDS}, "PREDECLARED_CELL_COUNTS")
    for child in children:
        parent = by_parent[child["parent_id"]]
        need(child["role"] == parent["role"] == "train" and child["preset_id"] == parent["preset_id"]
             and child["preset_id"] in design["presets"]
             and child["preset_component"] == parent["preset_component"] == "kokoro-v1_0-preset:" + child["preset_id"]
             and child["source_group_id"] == parent["source_group_id"], "ONE_EXPOSED_TYPED_COMPONENT")
        need(child["kind"] == parent["kind"] or (child["kind"] == "outside" and parent["kind"] == "positive"), "CHILD_KIND_RELATION")
        if child["generation_status"] == "success":
            need(parent["generation_status"] == "success" and child.get("exact_parent_pcm_slice") is True
                 and child["child_end_parent16k"] - child["child_start_parent16k"] == child["pcm_samples"], "EXACT_CHILD_SAMPLE_RELATION")
            actual = whole_support_geometry(child["pcm_samples"])
            need(type(child.get("whole_support_windows")) is int and child["whole_support_windows"] == actual["whole_support_windows"]
                 and child.get("geometry_usable_positive") is actual["geometry_usable_positive"], "ACTUAL_WHOLE_SUPPORT_INTEGER_GEOMETRY")
    return parents, children, by_parent


def observation_maps(text_results, phone_results, children, parents):
    need(set(text_results) == set(TEXT_PROVIDERS) and set(phone_results) == set(PHONE_PROVIDERS), "FIXED_FIVE_OBSERVER_DENOMINATOR")
    text_maps = {}
    for provider in TEXT_PROVIDERS:
        rows = text_results[provider]["rows"]
        need(type(rows) is list and len(rows) == 64 and [r["id"] for r in rows] == [c["id"] for c in children], "COMPLETE64_TEXT_ROWS")
        text_maps[provider] = {r["id"]: r for r in rows}
        need(all(text_maps[provider][c["id"]]["parent_id"] == c["parent_id"] for c in children), "TEXT_CHILD_PARENT_IDENTITY")
    for provider in PHONE_PROVIDERS:
        need(type(phone_results[provider]["rows"]) is dict
             and set(phone_results[provider]["rows"]) == {p["id"] for p in parents}, "COMPLETE48_PHONE_ROWS")
    return text_maps


def combined_counts(manifest, text_results, phone_results):
    stages = {"generation": manifest["generation_summary"]}
    stages.update({p: phone_results[p]["summary"] for p in PHONE_PROVIDERS})
    stages.update({p: {k: v for k, v in text_results[p].items() if k != "rows"} for p in TEXT_PROVIDERS})
    for name, maximum in (("generation", 48), ("phoneme", 48), ("phoneme_lv60", 48), ("qwen", 64), ("whisper", 64), ("parakeet", 64)):
        info = stages[name]
        lower, upper = info["forward_attempts_lower_bound"], info["forward_attempts_upper_bound"]
        need(type(lower) is int and type(upper) is int and 0 <= lower <= upper <= maximum, "FINITE_STAGE_BOUNDS: " + name)
        if info["actual_forward_attempts"] is None:
            need(info["native_forward_returns"] is None and info["completed"] is None and info["complete"] is False, "UNKNOWN_IS_NOT_ZERO: " + name)
        else:
            attempts, returns, completed = (info[k] for k in ("actual_forward_attempts", "native_forward_returns", "completed"))
            need(all(type(v) is int for v in (attempts, returns, completed)) and lower == attempts == upper
                 and 0 <= completed <= returns <= attempts <= maximum, "HONEST_STAGE_COUNTS: " + name)
            need(type(info.get("complete")) is bool and (not info["complete"] or attempts == returns == completed == maximum),
                 "COMPLETE_STAGE_COUNT: " + name)
    known = all(v["actual_forward_attempts"] is not None for v in stages.values())
    return {"counter_scope": "TTS requests plus one-input text/phone observation calls, not all internal neural decoder steps",
        "maximum_high_level_calls": 336,
        "actual_high_level_calls": sum(v["actual_forward_attempts"] for v in stages.values()) if known else None,
        "high_level_returns": sum(v["native_forward_returns"] for v in stages.values()) if known else None,
        "complete_saved_outputs": sum(v["completed"] for v in stages.values()) if known else None,
        "calls_lower_bound": sum(v["forward_attempts_lower_bound"] for v in stages.values()),
        "calls_upper_bound": sum(v["forward_attempts_upper_bound"] for v in stages.values()),
        "all_stages_complete": all(v["complete"] for v in stages.values()), "stages": stages,
        "retry_count": 0, "new_model_calls_during_evaluation": 0}


def evaluate_evidence(manifest, text_results, phone_results, plan, run_identity):
    """Pure decision layer. Inputs are already byte-verified by pinned readers."""
    design = checked_design(plan)
    parents, children, by_parent = check_roster(manifest, design, run_identity)
    text_maps = observation_maps(text_results, phone_results, children, parents)
    counts = combined_counts(manifest, text_results, phone_results)
    contrasts = {c["cell_id"]: c for c in design["contrast_cells"]}
    positive_cells = {c["cell_id"]: c for c in design["positive_cells"]}
    ordinary_cells = {c["cell_id"]: c for c in design["ordinary_cells"]}
    results = []
    for child in children:
        parent = by_parent[child["parent_id"]]
        texts = {p: text_maps[p][child["id"]]["raw_text"] for p in TEXT_PROVIDERS}
        child_ok = child["generation_status"] == "success"
        geometry = whole_support_geometry(child["pcm_samples"]) if child_ok else None
        phones = {}
        for provider in PHONE_PROVIDERS:
            observed = phone_results[provider]["rows"][parent["id"]]
            region = None
            if child_ok and observed["capture"] is not None:
                region = fixed_child_region(observed["frame_ids"], observed["by_id"], child["child_start_parent16k"],
                    child["child_end_parent16k"], parent["pcm_samples"])
            full = None if observed["phone_tokens"] is None else normalized_phones(observed["phone_tokens"])
            phones[provider] = {"parent_id": parent["id"], "parent_input_pcm_sha256": parent.get("pcm_sha256"),
                "child_input_pcm_sha256": child.get("pcm_sha256"), "parent_capture_available": observed["capture"] is not None,
                "full_parent_phone_tokens": observed["phone_tokens"], "full_parent_normalized": full,
                "full_parent_terminal_hypotheses": terminal_matches(full), "fixed_child_region": region,
                "reference_for_candidate": provider == "phoneme", "diagnostic_only": provider != "phoneme"}
        reference = phones["phoneme"]
        region = reference["fixed_child_region"]
        candidate = candidate_from_evidence(texts, None if region is None else region["phone_tokens"],
            reference["full_parent_phone_tokens"], None if geometry is None else geometry["whole_support_windows"])
        # Expected metadata is first consulted below, after blind classification.
        machine = None
        corroboration = {"kind": child["kind"], "machine_corroborated": None, "reason": "MISSING_COMPLETE_CONTROL_OBSERVATION"}
        ready = child_ok and all(t is not None for t in texts.values()) and region is not None and region["selected_frame_count"] > 0
        if child["kind"] == "contrast" and ready:
            expected = contrasts[child["cell_id"]]
            pattern = expected["contrast_pattern"]
            fixed = region["normalized_phones"]
            full = reference["full_parent_normalized"]
            machine = bool(re.fullmatch(pattern, fixed) and re.search(pattern + r"\Z", full)
                and not whole_matches(fixed) and not terminal_matches(full))
            corroboration.update(machine_corroborated=machine, reason="CONTRAST_REFERENCE_CORROBORATED" if machine else "CONTRAST_REFERENCE_NOT_CORROBORATED",
                expected_contrast_pattern=pattern, all_three_child_captures_complete=True,
                child_transcripts_may_spell_positive_aliases=True)
        elif child["kind"] in ("ordinary", "outside") and ready:
            text = design["carrier"]["boundary_control"]["text"] if child["kind"] == "outside" else ordinary_cells[child["cell_id"]]["requested_text"]
            exact_text = all(normalized_text(t) == normalized_text(text) for t in texts.values())
            no_target = not region["target_pattern_anywhere"]
            before_target = True
            if child["kind"] == "outside":
                need(type(parent.get("target_start_parent16k")) is int, "PARENT_TARGET_PREDICTED_SAMPLE")
                before_target = child["child_end_parent16k"] <= parent["target_start_parent16k"]
            machine = exact_text and no_target and before_target
            corroboration.update(machine_corroborated=machine, reason="ORDINARY_CONTROL_CORROBORATED" if machine else "ORDINARY_CONTROL_NOT_CORROBORATED",
                exact_child_text_all_three=exact_text, no_target_pattern_in_child_region=no_target,
                child_ends_before_parent_terminal_target=before_target, expected_ordinary_text=text)
        if child["kind"] == "positive":
            expected = positive_cells[child["cell_id"]]
            expected_match = candidate["candidate_positive"] and candidate["candidate_class"] == expected["requested_class"]
            hypothesis_match = expected_match and candidate["candidate_hypothesis"] == expected["pronunciation_hypothesis"]
        else:
            expected_match = hypothesis_match = None
        parent_hypothesis = parent["pronunciation_hypothesis"]
        parent_supported = parent_hypothesis in reference["full_parent_terminal_hypotheses"] if parent["kind"] == "positive" else None
        results.append({"id": child["id"], "parent_id": parent["id"], "preset_id": child["preset_id"],
            "preset_component": child["preset_component"], "source_group_id": child["source_group_id"], "role": "train",
            "kind": child["kind"], "cell_id": child["cell_id"], "generation_status": child["generation_status"],
            "expected_class_for_comparison_only": child["requested_class"],
            "expected_hypothesis_for_comparison_only": child["pronunciation_hypothesis"],
            "child_pcm_sha256": child.get("pcm_sha256"), "parent_pcm_sha256": parent.get("pcm_sha256"),
            "raw_child_texts": texts, "text_capture_sha256": {p: text_maps[p][child["id"]].get("capture_sha256") for p in TEXT_PROVIDERS},
            "phone_evidence": phones, "geometry": geometry, "candidate": candidate, "control": corroboration,
            "candidate_matches_expected_class": expected_match, "candidate_matches_expected_hypothesis": hypothesis_match,
            "parent_terminal_supports_its_declared_positive_hypothesis": parent_supported,
            "training_allowed": False, "training_weight": 0.0, "permanently_excluded_from_training": True})
    controls = [r for r in results if r["kind"] != "positive"]
    positive_coverage, outside_coverage = [], []
    for cell in design["positive_cells"]:
        hypothesis = cell["pronunciation_hypothesis"]
        positives = [r for r in results if r["kind"] == "positive" and r["cell_id"] == cell["cell_id"]]
        outside = [r for r in results if r["kind"] == "outside" and by_parent[r["parent_id"]]["pronunciation_hypothesis"] == hypothesis]
        components = sorted({r["preset_component"] for r in positives if r["candidate_matches_expected_hypothesis"]})
        supported = sorted({r["preset_component"] for r in outside if r["parent_terminal_supports_its_declared_positive_hypothesis"]})
        positive_coverage.append({"hypothesis": hypothesis, "declared_children": len(positives), "supported_preset_components": components,
            "supported_count": len(components), "minimum": design["validation_gates"]["minimum_distinct_preset_components_per_positive_hypothesis"]})
        outside_coverage.append({"hypothesis": hypothesis, "declared_outside_children": len(outside), "parent_supported_preset_components": supported,
            "supported_count": len(supported), "minimum": design["validation_gates"]["outside_control_parent_target_support_per_hypothesis_minimum"]})
    gates = {"all48_generation_complete": counts["stages"]["generation"]["complete"] and all(p["generation_status"] == "success" for p in parents)
             and all(c["generation_status"] == "success" for c in children),
        "all96_parent_phone_captures_complete": all(phone_results[p]["summary"]["complete"] for p in PHONE_PROVIDERS),
        "all192_child_text_captures_complete": all(text_results[p]["complete"] for p in TEXT_PROVIDERS),
        "all48_controls_candidate_positive_count_zero": sum(r["candidate"]["candidate_positive"] for r in controls) == 0,
        "all16_contrast_controls_reference_pattern_corroborated": all(r["control"]["machine_corroborated"] is True for r in controls if r["kind"] == "contrast"),
        "all16_ordinary_controls_all3_exact_text_and_no_target_in_fixed_region": all(r["control"]["machine_corroborated"] is True for r in controls if r["kind"] == "ordinary"),
        "all16_outside_controls_all3_exact_text_no_target_and_before_terminal": all(r["control"]["machine_corroborated"] is True for r in controls if r["kind"] == "outside"),
        "positive_each_hypothesis_distinct_preset_minimum": all(r["supported_count"] >= r["minimum"] for r in positive_coverage),
        "outside_each_parent_hypothesis_distinct_preset_minimum": all(r["supported_count"] >= r["minimum"] for r in outside_coverage)}
    complete = all(gates[k] for k in ("all48_generation_complete", "all96_parent_phone_captures_complete", "all192_child_text_captures_complete"))
    passed = complete and all(gates.values())
    categories = [{"kind": kind, "declared_count": 16,
        "successful_generation": sum(r["generation_status"] == "success" for r in results if r["kind"] == kind),
        "candidate_positive_count": sum(r["candidate"]["candidate_positive"] for r in results if r["kind"] == kind),
        "machine_corroborated_controls": sum(r["control"]["machine_corroborated"] is True for r in results if r["kind"] == kind),
        "unresolved_controls": sum(r["control"]["machine_corroborated"] is None for r in results if r["kind"] == kind) if kind != "positive" else None}
        for kind in KINDS]
    return {"schema": SCHEMA, "campaign_id": CAMPAIGN, "run": run_identity, "design_sha256": DESIGN_SHA256,
        "reference_policy_sha256": REFERENCE_SHA256, "reference_rules_sha256": _reference.RULES_SHA256,
        "status": "INCOMPLETE_DIAGNOSTIC_NO_EXPANSION" if not complete else ("PASS_CONTEXT_CONTRACT_FEASIBILITY_ONLY" if passed else "STOP_CONTEXT_CONTRACT_VALIDATION_FAILED"),
        "all_calls_complete": complete, "context_contract_control_test_passed": passed,
        "counts": counts, "gates": gates, "failed_gates": [k for k, value in gates.items() if not value],
        "declared_parents": 48, "declared_children": 64, "declared_controls": 48,
        "by_kind": categories, "positive_hypothesis_coverage": positive_coverage, "outside_parent_support": outside_coverage,
        "rows": results, "reference_provider": "phoneme", "diagnostic_provider": "phoneme_lv60",
        "candidate_classifier_uses_expected_metadata": False, "new_tier": "context_supported_weak",
        "old_labels_modified": False, "old_data_re_admitted": False, "training_allowed": False,
        "training_snapshot_allowed": False, "training_weight": 0.0, "all_probe_inputs_permanently_excluded_from_training": True,
        "automatic_expansion": False, "high_quality_model_passed": False, "new_model_calls_during_evaluation": 0,
        "whole_child_pronunciation_gold": False, "result_is_contract_validation_not_model_quality": True}


def evaluate(source_root, observer_roots, plan, run_identity):
    checked_design(plan)
    need(set(observer_roots) == set(TEXT_PROVIDERS + PHONE_PROVIDERS), "ALL_FIVE_PROVIDER_ROOTS")
    need(plan.get("evaluate_code_sha256") == sha(regular(Path(__file__))), "EVALUATOR_SOURCE_PIN")
    worker = module_at("context64_pure_source_readback", HERE / "english_kokoro_context64_once.py", plan["worker_sha256"])
    phone = module_at("context64_pure_phone_readback", HERE / "english_kokoro_context64_phone.py", plan["phone_code_sha256"])
    manifest = worker.source_readback(Path(source_root), plan, allow_partial=True)
    text_results = {provider: worker.read_text_provider(Path(observer_roots[provider]), manifest["children"], plan, run_identity, provider=provider)
        for provider in TEXT_PROVIDERS}
    phone_results = {provider: phone.readback(provider=provider, observer_root=Path(observer_roots[provider]),
        source_root=Path(source_root), plan=plan, run_identity=run_identity, parent_rows=manifest["parents"])
        for provider in PHONE_PROVIDERS}
    return evaluate_evidence(manifest, text_results, phone_results, plan, run_identity)


def source_preflight():
    need(sha(regular(HERE / "english_context64_policy_v2_reference.py")) == REFERENCE_SHA256, "REFERENCE_EXACT_BYTES")
    need(normalized_text("Hay Mero.") == "hay mero" and normalized_phones(["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"]) == "heɪmɛroʊ", "FROZEN_NORMALIZERS")
    return {"schema": SCHEMA, "source_check": "PASS", "reference_policy_sha256": REFERENCE_SHA256,
        "design_sha256": DESIGN_SHA256, "new_model_calls": 0, "new_processor_calls": 0, "training_allowed": False}
