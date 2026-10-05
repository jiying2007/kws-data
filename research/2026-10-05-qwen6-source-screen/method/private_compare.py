"""Private, post-freeze comparison of six synthetic clips; no ASR or audio I/O.

The runner must verify and freeze both raw recognizer results before importing
the private plan and calling ``summarize``. This helper neither proves that
ordering nor supplies any plan fields to a recognizer.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
from types import ModuleType


AUDIO_REVIEW_SHA256 = "9c5c1e4d31238f3e5e2daa6a5d102638f916183daee56d63735a11a37a60e891"
MODELS = ("FunAudioLLM/SenseVoiceSmall", "Qwen/Qwen3-ASR-0.6B")
TARGETS = {"K1": "你好小窝", "K2": "小窝小窝"}
PRESET_ROLES = {"Ryan": "train", "Aiden": "train", "Ono_Anna": "dev"}
STATUSES = ("complete", "incomplete", "error", "missing", "quarantine")


def _load_audio_review():
    here = Path(__file__).resolve().parent
    source = here / "vendor/audio_review.py"
    if not source.is_file():
        raise ImportError("Pinned audio_review.py is required")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != AUDIO_REVIEW_SHA256:
        raise ImportError("audio_review.py source hash mismatch")
    module = ModuleType("_asr6_pinned_audio_review")
    module.__file__ = str(source)
    exec(compile(data, str(source), "exec"), module.__dict__)
    return module


_audio = _load_audio_review()
normalize = _audio.normalize
repetition = _audio.repetition
ComparisonError = _audio.ReviewError
_require = _audio.require


def _checked_plan(plan):
    planned = _audio.indexed(plan)
    _require(len(planned) == 6, "Private plan must contain exactly six unique IDs")
    cells = set()
    for row in planned.values():
        _audio.exact_keys(row, ("audio_id", "role", "preset", "keyword", "intended_text"))
        _require(type(row["preset"]) is str and row["preset"] in PRESET_ROLES,
                 "Plan preset is outside the six-clip panel")
        _require(row["role"] == PRESET_ROLES[row["preset"]], "Wrong role for panel preset")
        _require(type(row["keyword"]) is str and row["keyword"] in TARGETS,
                 "Unknown panel keyword")
        _require(row["intended_text"] == TARGETS[row["keyword"]],
                 "Panel requires the exact intended K1/K2 text")
        cell = (row["preset"], row["keyword"])
        _require(cell not in cells, "Duplicate panel cell")
        cells.add(cell)
    _require(cells == {(preset, keyword) for preset in PRESET_ROLES for keyword in TARGETS},
             "Plan must cover each required preset/keyword pair once")
    return planned


def _checked_results(results_by_model, ids):
    _require(type(results_by_model) is dict and set(results_by_model) == set(MODELS),
             "Results must name exactly FunAudioLLM/SenseVoiceSmall and Qwen/Qwen3-ASR-0.6B")
    indexed = {}
    for model in MODELS:
        rows = _audio.indexed(results_by_model[model])
        _require(set(rows) <= ids, "ASR contains an unrequested audio ID")
        for row in rows.values():
            _audio.exact_keys(row, ("audio_id", "status"),
                              ("raw_text", "quality_flags", "raw_output", "wav_sha256"))
            _require(type(row.get("status")) is str and row["status"] in STATUSES,
                     "Invalid ASR status")
            _require("raw_text" in row or row["status"] in ("error", "missing", "quarantine"),
                     "Complete/incomplete ASR outcome requires raw_text")
            _audio.valid_text(row.get("raw_text", ""))
            flags = row.get("quality_flags", [])
            _require(type(flags) is list and all(type(flag) is str for flag in flags),
                     "Invalid ASR quality flags")
        indexed[model] = rows
    return indexed


def summarize(plan, results_by_model, metrics_by_id):
    """Return a JSON-ready private report without mutating inputs or doing I/O.

    ``plan`` is a projected list of exactly six rows with audio_id, role,
    preset, keyword, intended_text. ``results_by_model`` maps each exact model
    name to raw outcome rows with audio_id, status, raw_text and optional
    quality_flags/raw_output/wav_sha256. Execution details remain in raw_output.
    Missing rows are explicit missing outcomes. ``metrics_by_id`` maps IDs to
    unchanged wav_metrics results, optionally augmented with ``generation_flags``
    from the private generation receipt. Absent measurements quarantine a clip.
    Unknown IDs fail closed.
    """
    planned = _checked_plan(plan)
    ids = set(planned)
    results = _checked_results(results_by_model, ids)
    _require(type(metrics_by_id) is dict and set(metrics_by_id) <= ids,
             "Metrics contain an unrequested audio ID or are not an object")
    # Validate finite, UTF-8-safe JSON before retaining raw evidence. No model,
    # filesystem, network, threshold selection, or audio processing occurs here.
    _audio.encode_json([plan, results_by_model, metrics_by_id])
    clips, outcomes = [], []
    for audio_id, row in planned.items():
        metrics = metrics_by_id.get(audio_id)
        reasons = []
        if metrics is None:
            reasons.append("missing_signal_metrics")
        else:
            _require(type(metrics) is dict and type(metrics.get("flags")) is list
                     and all(type(flag) is str for flag in metrics["flags"]),
                     "Invalid measurement flags")
            _require(type(metrics.get("digital_all_zero")) is bool,
                     "Metrics require a digital_all_zero boolean")
            generation_flags = metrics.get("generation_flags", [])
            _require(type(generation_flags) is list
                     and all(type(flag) is str for flag in generation_flags),
                     "Invalid generation flags")
            reasons.extend("signal:" + flag for flag in metrics["flags"])
            reasons.extend("generation:" + flag for flag in generation_flags)
            if metrics["digital_all_zero"] and "signal:digital_all_zero" not in reasons:
                reasons.append("signal:digital_all_zero")
        evidence = []
        for model in MODELS:
            raw = results[model].get(audio_id)
            status = raw["status"] if raw is not None else "missing"
            text = normalize(raw.get("raw_text", "")) if raw is not None else None
            flags = raw.get("quality_flags", []) if raw is not None else []
            if status != "complete":
                reasons.append(f"asr:{model}:{status}")
            if not text:
                reasons.append(f"asr:{model}:empty_or_missing_text")
            reasons.extend(f"asr:{model}:{flag}" for flag in flags)
            evidence.append({"audio_id": audio_id, "model": model, "status": status,
                             "raw_record": deepcopy(raw), "normalized_text": text,
                             "character_count": len(text) if text is not None else None,
                             "repetition": repetition(text) if text is not None else None})
        valid = all(item["status"] == "complete" and item["normalized_text"] for item in evidence)
        texts = [item["normalized_text"] for item in evidence]
        intent = normalize(row["intended_text"])
        if not valid:
            state = "machine_unresolved"
        elif texts[0] != texts[1]:
            state = "machine_dispute"
            reasons.append("asr_disagreement")
        elif texts[0] != intent:
            state = "machine_agreement_mismatch"
            reasons.append("both_asrs_differ_from_intended_text")
        else:
            state = "machine_agreement_match"
        candidate = state == "machine_agreement_match" and not reasons
        handling = "weak_machine_supported_candidate" if candidate else "quarantine"
        for item in evidence:
            item["handling"] = handling
        outcomes.extend(evidence)
        clips.append({**deepcopy(row), "intended_normalized_text": intent,
                      "intended_repetition": repetition(intent),
                      "signal_measurements": deepcopy(metrics), "machine_state": state,
                      "handling": handling, "quarantine_reasons": sorted(set(reasons)),
                      "weak_machine_supported_candidate": candidate,
                      "tail_completeness_truth": "UNKNOWN",
                      "actual_positive_established": False, "human_truth_established": False,
                      "training_admitted": False, "automatic_retry": False,
                      "listening_requested": False})
    status_counts = Counter(item["status"] for item in outcomes)
    candidate_count = sum(row["weak_machine_supported_candidate"] for row in clips)
    return {
        "schema": "private-asr6-comparison-v1", "clips": clips, "outcomes": outcomes,
        "counts": {"clips": 6, "expected_asr_outcomes": 12, "asr_outcomes": len(outcomes),
                   "asr_status": {status: status_counts[status] for status in STATUSES},
                   "weak_machine_supported_candidates": candidate_count,
                   "quarantined_clips": 6 - candidate_count,
                   "actual_positives_established": 0, "training_admitted": 0},
        "models": list(MODELS), "audio_review_sha256": AUDIO_REVIEW_SHA256,
        "normalization": "Unchanged audio_review.normalize: remove whitespace and Unicode P* only",
        "evidence_level": "weak_machine_evidence_only_no_human_or_gold",
        "raw_freeze_verification": "Caller prerequisite; not performed or established by this pure helper",
        "decoder_complete_means": "Execution completed, not proof of acoustic completeness",
        "signal_interpretation": "Digital all-zero is integrity evidence only; tail metrics do not prove acoustic completeness",
        "thresholds_added_by_comparison": False, "statistical_independence_assumed": False,
        "asr_executed": False, "automatic_retry": False, "listening_requested": False,
        "future24": {
            "coverage_proven": False, "heldout_preset_excluded_from_this_panel": "Sohee",
            "required_actual_positive_cells": [
                {"role": role, "keyword": keyword, "actual_positive_established": False}
                for role in ("train", "dev", "heldout") for keyword in TARGETS],
            "note": "Future24 still requires actual K1 and K2 positives in each role; intentions and this machine screen do not establish them",
        },
    }
