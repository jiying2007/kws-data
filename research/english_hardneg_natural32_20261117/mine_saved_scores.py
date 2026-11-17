"""Turn saved native-C ready scores into an unadmitted acoustic-review ledger.

This module performs no model inference, decoding, feature extraction, ASR,
training, or negative-label admission. Raw C trace parsing/authentication belongs
to the calling scan worker; each original trace digest is required here.

API: build_ledger(roster, pcm_by_id, ready_rows_by_id, *, model_sha256,
                  config_sha256, source_evidence)
The two mappings use roster.rows[*].recording_id as keys. Each score row has
sample (absolute availability count within its original utterance), three
probabilities [background, Vosen, Mero], and keyword (the emitted event ID,
not argmax). Only ready rows are supplied, with no missing/extra ready clocks.
source_evidence requires trace_sha256_by_id; other execution metadata is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "english-hardneg-saved-score-review-ledger-v1"
RATE = 16000
FIRST_READY = 35200
HOP = 1280
PROPOSAL_THRESHOLD = 0.5
PRE_SAMPLES = 38400
POST_SAMPLES = 16000
MAX_PER_SOURCE = 2
MAX_TOTAL = 32
MAX_SOURCES = 32
MAX_SOURCE_SAMPLES = 240000
KEYWORDS = {1: "Hey Vosen", 2: "Hey Mero"}
FALSE_FLAGS = {
    "human_gold": False,
    "trusted_negative": False,
    "training_admitted": False,
    "formal_training_admitted": False,
    "pronunciation_verified": False,
    "model_promotion_allowed": False,
    "qualification_allowed": False,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_value(value, label):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
         label + ": exact lowercase SHA256 required")
    return value


def exact_int(value, minimum, maximum, label):
    need(type(value) is int and minimum <= value <= maximum, label + ": integer bounds")
    return value


def finite_probability(value):
    need(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
         "finite probability in [0,1] required")
    return float(value)


def normalized_rows(rows, frames):
    """Check the complete observed ready clock; no probability or event repair."""
    need(type(rows) is list, "ready scores must be a list")
    expected = list(range(FIRST_READY, frames + 1, HOP))
    need(len(rows) == len(expected), "complete ready-score count required")
    result = []
    for row, sample in zip(rows, expected):
        need(type(row) is dict and {"sample", "probabilities", "keyword"} <= set(row),
             "ready-score fields missing")
        need(type(row["sample"]) is int and row["sample"] == sample,
             "missing, repeated, reordered, or out-of-bounds ready clock")
        if "ready" in row:
            need(row["ready"] is True, "non-ready row in ready-score input")
        probability = row["probabilities"]
        need(type(probability) is list and len(probability) == 3, "three-class probability vector required")
        probability = [finite_probability(p) for p in probability]
        need(abs(sum(probability) - 1) <= 1e-5, "probabilities must sum to one")
        keyword = exact_int(row["keyword"], 0, 2, "emitted keyword")
        if "logits" in row:
            need(type(row["logits"]) is list and len(row["logits"]) == 3 and
                 all(type(x) in (int, float) and math.isfinite(x) for x in row["logits"]),
                 "finite three-class optional logits required")
        result.append({"sample": sample, "probabilities": probability, "keyword": keyword})
    return result


def keyword_summary(rows, keyword):
    peak = max(rows, key=lambda r: (r["probabilities"][keyword], -r["sample"]), default=None)
    return {
        "keyword_id": keyword,
        "keyword_name": KEYWORDS[keyword],
        "peak_probability": peak["probabilities"][keyword] if peak else None,
        "peak_sample": peak["sample"] if peak else None,
        "proposal_threshold_samples": [r["sample"] for r in rows if r["probabilities"][keyword] >= PROPOSAL_THRESHOLD],
        "event_samples": [r["sample"] for r in rows if r["keyword"] == keyword],
    }


def merged_windows(rows, frames):
    seeds = []
    for row in rows:
        threshold_ids = [k for k in KEYWORDS if row["probabilities"][k] >= PROPOSAL_THRESHOLD]
        if not threshold_ids and row["keyword"] == 0:
            continue
        sample = row["sample"]
        seeds.append({
            "start_sample": max(0, sample - PRE_SAMPLES),
            "end_sample_exclusive": min(frames, sample + POST_SAMPLES),
            "seed": {"sample": sample, "proposal_threshold_keywords": threshold_ids,
                     "emitted_keyword": row["keyword"]},
        })
    merged = []
    for seed in seeds:
        start, end = seed["start_sample"], seed["end_sample_exclusive"]
        # PCM intervals are half-open. Merely adjacent intervals are not overlap.
        if merged and start < merged[-1]["end_sample_exclusive"]:
            merged[-1]["end_sample_exclusive"] = max(end, merged[-1]["end_sample_exclusive"])
            merged[-1]["seeds"].append(seed["seed"])
        else:
            merged.append({"start_sample": start, "end_sample_exclusive": end, "seeds": [seed["seed"]]})
    return merged


def priority(candidate):
    """Frozen before scan: events first, then score, then stable source/clock."""
    return (0 if candidate["event_count"] else 1,
            -candidate["maximum_keyword_probability"], candidate["recording_id"],
            candidate["start_sample"], candidate["end_sample_exclusive"])


def build_ledger(roster, pcm_by_id, ready_rows_by_id, *, model_sha256,
                 config_sha256, source_evidence):
    """Return an UNREVIEWED queue and every source's saved-score peak evidence."""
    hash_value(model_sha256, "model")
    hash_value(config_sha256, "config")
    need(type(roster) is dict and type(roster.get("rows")) is list and
         1 <= len(roster["rows"]) <= MAX_SOURCES, "bounded source roster required")
    need(roster.get("coverage_head_binary_sha256") == model_sha256, "roster/model identity mismatch")
    need(type(pcm_by_id) is dict and type(ready_rows_by_id) is dict and type(source_evidence) is dict,
         "input mappings required")
    need(all(type(r) is dict for r in roster["rows"]), "source rows must be objects")
    ids = [r.get("recording_id") for r in roster["rows"]]
    need(all(type(k) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", k) for k in ids) and
         len(ids) == len(set(ids)), "unique safe recording IDs")
    need(set(pcm_by_id) == set(ready_rows_by_id) == set(ids), "exact source/PCM/score roster required")
    traces = source_evidence.get("trace_sha256_by_id")
    need(type(traces) is dict and set(traces) == set(ids), "every original C trace digest required")
    for recording_id, value in traces.items():
        hash_value(value, recording_id + " C trace")
    # Also rejects unserializable/nonfinite extra execution metadata.
    source_evidence_sha = digest(canonical(source_evidence))
    roster_sha = digest(canonical(roster))
    all_windows, source_summaries = [], []
    for source in roster["rows"]:
        recording_id = source["recording_id"]
        need(source.get("source_role") == "train", "only original train-role sources may be mined")
        for role_field in ("old_bg48_role", "current_coverage_role"):
            if role_field in source:
                need(source[role_field] == "train", "historical development/holdout excluded")
        need(all(source.get(key) is False for key in FALSE_FLAGS), "source admission flags must remain false")
        need(source.get("label_status") == "UNKNOWN_REVIEW_REQUIRED" and source.get("expected_events") is None,
             "source must remain unknown, without invented complete event annotation")
        need(all(type(source.get(k)) is int for k in ("sample_rate", "channels", "sample_width")) and
             (source.get("sample_rate"), source.get("channels"), source.get("sample_width")) == (RATE, 1, 2),
             "original mono 16k PCM16 geometry")
        frames = exact_int(source.get("frames"), 1, MAX_SOURCE_SAMPLES, "source frames")
        pcm = pcm_by_id[recording_id]
        need(type(pcm) is bytes and len(pcm) == 2 * frames and
             digest(pcm) == hash_value(source.get("pcm_sha256"), "source PCM"),
             "original PCM bytes/length/digest mismatch")
        scores = normalized_rows(ready_rows_by_id[recording_id], frames)
        if "expected_ready_rows" in source:
            need(source["expected_ready_rows"] == len(scores), "roster ready count mismatch")
        input_rows_sha = digest(canonical(ready_rows_by_id[recording_id]))
        normalized_sha = digest(canonical(scores))
        by_keyword = {str(k): keyword_summary(scores, k) for k in KEYWORDS}
        common = {
            "recording_id": recording_id,
            "source_member": source.get("source_member"),
            "speaker_id": source.get("speaker_id"),
            "source_family_id": source.get("source_family_id"),
            "source_role": "train",
            "source_utterance_continuity": "natural_individual",
            "source_pcm_sha256": source["pcm_sha256"],
            "source_samples": frames,
            "model_sha256": model_sha256,
            "config_sha256": config_sha256,
            "roster_sha256": roster_sha,
            "original_c_trace_sha256": traces[recording_id],
            "input_ready_rows_sha256": input_rows_sha,
            "normalized_ready_rows_sha256": normalized_sha,
            "review": "UNREVIEWED",
            "label_status": "UNKNOWN_REVIEW_REQUIRED",
            **FALSE_FLAGS,
        }
        source_windows = []
        for interval in merged_windows(scores, frames):
            start, end = interval["start_sample"], interval["end_sample_exclusive"]
            # A score at sample N observes PCM through N-1; it belongs to (start,end].
            context_scores = [r for r in scores if start < r["sample"] <= end]
            detail = {str(k): keyword_summary(context_scores, k) for k in KEYWORDS}
            triggering = sorted({k for seed in interval["seeds"]
                                 for k in seed["proposal_threshold_keywords"] +
                                 ([seed["emitted_keyword"]] if seed["emitted_keyword"] else [])})
            peak = max(d["peak_probability"] for d in detail.values())
            identity = {"source_pcm_sha256": source["pcm_sha256"], "model_sha256": model_sha256,
                        "config_sha256": config_sha256, "recording_id": recording_id,
                        "start_sample": start, "end_sample_exclusive": end}
            candidate = {
                **common, **interval,
                "candidate_id": "hardneg-review-" + digest(canonical(identity))[:24],
                "start_seconds": start / RATE,
                "end_seconds_exclusive": end / RATE,
                "window_samples": end - start,
                "window_pcm_sha256": digest(pcm[2 * start:2 * end]),
                "pre_context_clipped_at_source_start": any(s["sample"] < PRE_SAMPLES for s in interval["seeds"]),
                "post_context_clipped_at_source_end": any(s["sample"] + POST_SAMPLES > frames for s in interval["seeds"]),
                "triggering_keywords": triggering,
                "maximum_keyword_probability": peak,
                "event_count": sum(len(d["event_samples"]) for d in detail.values()),
                "by_keyword": detail,
                "selection": "PENDING_CAPS",
                "training_tsv_emitted": False,
                "word_end_annotation": None,
            }
            source_windows.append(candidate)
        source_windows.sort(key=priority)
        for rank, candidate in enumerate(source_windows, 1):
            candidate["source_priority_rank"] = rank
            candidate["selection"] = "ELIGIBLE_FOR_GLOBAL_CAP" if rank <= MAX_PER_SOURCE else "EXCLUDED_PER_SOURCE_CAP"
        all_windows.extend(source_windows)
        source_summaries.append({
            **common,
            "source_seconds": frames / RATE,
            "ready_rows": len(scores),
            "by_keyword": by_keyword,
            "actual_event_count": sum(len(d["event_samples"]) for d in by_keyword.values()),
            "merged_candidate_windows": len(source_windows),
            "windows_after_per_source_cap": min(len(source_windows), MAX_PER_SOURCE),
            "selected_candidate_windows": 0,
        })
    eligible = sorted([c for c in all_windows if c["selection"] == "ELIGIBLE_FOR_GLOBAL_CAP"], key=priority)
    for rank, candidate in enumerate(eligible, 1):
        candidate["global_priority_rank"] = rank
        candidate["selection"] = "SELECTED_UNREVIEWED" if rank <= MAX_TOTAL else "EXCLUDED_GLOBAL_CAP"
    selected = [c for c in eligible if c["selection"] == "SELECTED_UNREVIEWED"]
    excluded = sorted([c for c in all_windows if c["selection"].startswith("EXCLUDED_")], key=priority)
    for summary in source_summaries:
        summary["selected_candidate_windows"] = sum(c["recording_id"] == summary["recording_id"] for c in selected)
    source_summaries.sort(key=lambda r: r["recording_id"])
    return {
        "schema": SCHEMA,
        "status": "UNREVIEWED_CANDIDATES_SAVED" if selected else "NO_CANDIDATE_IN_SAVED_OBSERVATION",
        "policy": {
            "candidate_seed": "emitted keyword event OR p(class1)>=0.5 OR p(class2)>=0.5",
            "proposal_threshold": PROPOSAL_THRESHOLD,
            "pre_samples": PRE_SAMPLES,
            "post_samples": POST_SAMPLES,
            "sample_rate": RATE,
            "merge": "strictly overlapping half-open source PCM windows; merge across keywords without duplicating PCM",
            "maximum_windows_per_source": MAX_PER_SOURCE,
            "maximum_total_windows": MAX_TOTAL,
            "priority": "has_event descending; maximum_keyword_probability descending; recording_id/start/end ascending",
            "same_priority_used_before_each_cap": True,
            "source_utterance_continuity": "natural_individual",
            "state_policy": "reset only between original utterances; no concatenation or gap filling",
            "eof_padding_samples": 0,
            "first_ready_sample": FIRST_READY,
            "ready_hop_samples": HOP,
            "event_clock": "input availability sample count; not acoustic word end",
            "no_candidate_means": "empty review queue, not a proven negative or FAR result",
        },
        "model_sha256": model_sha256,
        "config_sha256": config_sha256,
        "roster_sha256": roster_sha,
        "source_evidence": source_evidence,
        "source_evidence_sha256": source_evidence_sha,
        "counts": {
            "sources": len(source_summaries),
            "source_samples": sum(r["source_samples"] for r in source_summaries),
            "source_seconds": sum(r["source_samples"] for r in source_summaries) / RATE,
            "ready_rows": sum(r["ready_rows"] for r in source_summaries),
            "actual_events": sum(r["actual_event_count"] for r in source_summaries),
            "merged_candidates": len(all_windows),
            "selected_candidates": len(selected),
            "excluded_by_per_source_cap": sum(c["selection"] == "EXCLUDED_PER_SOURCE_CAP" for c in excluded),
            "excluded_by_global_cap": sum(c["selection"] == "EXCLUDED_GLOBAL_CAP" for c in excluded),
        },
        "source_summaries": source_summaries,
        "candidates": selected,
        "excluded_candidates": excluded,
        "new_model_forwards": 0,
        "new_tts_asr_phone_calls": 0,
        "optimizer_updates": 0,
        "training_tsv_emitted": False,
        "verified_negative_seconds": 0,
        "independent_far": "NOT_MEASURED",
        "independent_frr": "NOT_MEASURED",
        "review": "UNREVIEWED",
        **FALSE_FLAGS,
    }


def unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True,
                        help="JSON with model_sha256, config_sha256, source_evidence and ready_rows_by_id")
    parser.add_argument("--pcm-root", type=Path, required=True,
                        help="original PCM files named <recording_id>.pcm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roster, scan = read_json(args.roster), read_json(args.scan)
    pcm = {}
    for row in roster["rows"]:
        rid = row["recording_id"]
        need(type(rid) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", rid), "safe recording ID required")
        path = (args.pcm_root / (rid + ".pcm")).absolute()
        need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "regular original PCM required")
        pcm[rid] = path.read_bytes()
    ledger = build_ledger(roster, pcm, scan["ready_rows_by_id"], model_sha256=scan["model_sha256"],
                          config_sha256=scan["config_sha256"], source_evidence=scan["source_evidence"])
    output = args.output.absolute()
    need(not output.exists() and output.parent.is_dir() and
         not any(p.is_symlink() for p in (output, *output.parents)), "create-only ledger path required")
    raw = canonical(ledger)
    with output.open("xb") as target:
        target.write(raw)
    print(json.dumps({"status": ledger["status"], "ledger_sha256": digest(raw),
                      "counts": ledger["counts"], "new_model_forwards": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
