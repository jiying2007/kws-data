"""Saved-byte V7 candidate-relative pilot evaluation; no KWS training or tuning."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import english_kws_v7_voxcpm_pilot as pilot
import english_kws_v7_voxcpm_observer as observer
import english_kws_v7_voxcpm_policy as policy

PROVIDERS = ("qwen", "whisper", "parakeet", "phoneme")
CANDIDATES = ("VOSEN", "MERO")
SCHEMA = "english-kws-v7-voxcpm-pilot-evaluation-v1"


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return pilot.sha(raw)


def read_json(path):
    return pilot.read_json(path)


def write_once(path, value):
    return pilot.write_once(path, value)


def source_check():
    plan, _ = pilot.load_plan()
    observer.source_check()
    need(plan.get("admission_policy") == {
        "repository": policy.UPSTREAM["repository"],
        "commit": policy.UPSTREAM["commit"],
        "path": policy.UPSTREAM["path"],
        "git_blob_sha": policy.UPSTREAM["blob_sha"],
        "rule": "exact V7 candidate-relative triple-ASR plus independent phoneme; unknown is never background",
    }, "PINNED_V7_ADMISSION_POLICY")
    return {
        "source_check": "PASS",
        "policy_upstream": policy.UPSTREAM,
        "providers": list(PROVIDERS),
        "new_model_calls": 0,
        "new_kws_forwards": 0,
        "optimizer_updates": 0,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def _read_observers(source_root, observer_root):
    results = {}
    summaries = {}
    for provider in PROVIDERS:
        root = Path(observer_root) / provider
        summaries[provider] = observer.readback(provider, source_root, root)
        result = read_json(root / "RESULT.json")
        need(result.get("provider") == provider and result.get("denominator") == 12
             and result.get("training_allowed") is False
             and result.get("retry_count") == 0,
             "OBSERVER_PACKET_IDENTITY:" + provider)
        results[provider] = {item["id"]: item for item in result["rows"]}
        need(len(results[provider]) == 12, "OBSERVER_TWELVE_ROW_DENOMINATOR:" + provider)
    return results, summaries


def _audio_gate(rows):
    failures = []
    seen_pcm = {}
    for row in rows:
        if row.get("generation_status") != "success":
            failures.append({"id": row["id"], "reason": "SOURCE_NOT_SUCCESS"})
            continue
        if not (6400 <= row.get("pcm_samples", 0) <= 28672):
            failures.append({"id": row["id"], "reason": "MODEL_INPUT_GEOMETRY"})
        if row.get("clip_ratio", 1.0) > 0.001:
            failures.append({"id": row["id"], "reason": "CLIPPING_RATIO"})
        if row.get("rms_dbfs", -120.0) <= -55.0:
            failures.append({"id": row["id"], "reason": "LOW_SIGNAL_RMS"})
        value = row.get("pcm_sha256")
        if value in seen_pcm:
            failures.append({"id": row["id"], "reason": "DUPLICATE_PCM",
                             "duplicate_of": seen_pcm[value]})
        else:
            seen_pcm[value] = row["id"]
    return {"pass": not failures and len(rows) == 12,
            "denominator": len(rows), "failures": failures,
            "all_clips_untrimmed": True, "all_clips_full_pcm": True}


def _candidate_results(source_rows, observer_packets, observer_root):
    all_results = []
    by_candidate = {candidate: [] for candidate in CANDIDATES}
    for row in source_rows:
        texts = []
        evidence_refs = []
        for provider in ("qwen", "whisper", "parakeet"):
            item = observer_packets[provider][row["id"]]
            if item.get("status") != "success":
                texts.append(None)
                evidence_refs.append(None)
                continue
            text = policy.verify_observation(
                item, provider, row, Path(observer_root) / provider, phones=False)
            texts.append(text)
            evidence_refs.append({
                "provider": provider,
                "capture_sha256": item["capture_sha256"],
            })
        phone_item = observer_packets["phoneme"][row["id"]]
        if phone_item.get("status") == "success":
            phones = policy.verify_observation(
                phone_item, "phoneme", row, Path(observer_root) / "phoneme", phones=True)
            phone_ref = {"provider": "phoneme", "capture_sha256": phone_item["capture_sha256"]}
        else:
            phones = None
            phone_ref = None
        for candidate in CANDIDATES:
            tier, label, reason, observed = policy.decide_candidate(
                row, candidate, texts if all(isinstance(text, str) and text for text in texts) else None,
                phones)
            item = {
                "id": row["id"], "source_group_id": row["source_group_id"],
                "role": row["role"], "candidate": candidate,
                "candidate_label": row["candidate_labels"][candidate],
                "tier": tier, "research_class": label, "reason": reason,
                "observed_phrase": observed,
                "asr_evidence": evidence_refs, "phone_evidence": phone_ref,
                "training_allowed": False, "human_gold": False,
            }
            by_candidate[candidate].append(item)
            all_results.append(item)
    return all_results, by_candidate


def _support_gate(source_rows, by_candidate):
    groups = sorted({row["source_group_id"] for row in source_rows})
    detail = {}
    passes = True
    for candidate in CANDIDATES:
        detail[candidate] = {}
        for group_id in groups:
            items = [item for item in by_candidate[candidate]
                     if item["source_group_id"] == group_id]
            positives = [item for item in items if item["candidate_label"] == 1]
            cross_negatives = [item for item in items if item["candidate_label"] == 0]
            positive_silver = sum(item["tier"] == "silver" for item in positives)
            cross_silver = sum(item["tier"] == "silver" for item in cross_negatives)
            passed = positive_silver >= 2 and cross_silver >= 2
            passes = passes and passed
            detail[candidate][group_id] = {
                "positive_rows": len(positives),
                "positive_silver": positive_silver,
                "cross_trigger_rows": len(cross_negatives),
                "cross_trigger_silver": cross_silver,
                "pass": passed,
            }
    return {"pass": passes, "minimum_silver_positive_per_group": 2,
            "minimum_silver_cross_trigger_per_group": 2, "groups": detail}


def _write_blind_review(source_root, source_rows, blind_output, evaluation_output):
    blind_root = pilot.create_output_root(blind_output)
    audio_dir = blind_root / "audio"
    audio_dir.mkdir(mode=0o700)
    ordered = sorted(source_rows, key=lambda row: (row.get("wav_sha256", ""), row["id"]))
    items, key = [], []
    for index, row in enumerate(ordered, 1):
        need(row.get("generation_status") == "success", "BLIND_REVIEW_REQUIRES_ALL_TWELVE_CLIPS")
        blind_id = f"blind-{index:02d}"
        raw = pilot.regular(pilot.within(source_root, row["wav_path"]), pilot.MAX_WAV_BYTES)
        need(sha(raw) == row["wav_sha256"], "BLIND_REVIEW_AUDIO_HASH")
        filename = blind_id + ".wav"
        target = audio_dir / filename
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        items.append({
            "blind_id": blind_id, "audio_file": "audio/" + filename,
            "wav_sha256": row["wav_sha256"],
            "naturalness_score_1_to_5": None,
            "pronunciation_score_1_to_5": None,
            "audible_clipping_or_artifacts": None,
            "notes": "",
        })
        key.append({
            "blind_id": blind_id, "source_row_id": row["id"],
            "donor_id": row["donor_id"], "role": row["role"],
            "candidate": row["candidate"], "cell_id": row["cell_id"],
            "tempo": row["tempo"], "requested_text": row["requested_text"],
        })
    form = {
        "schema": "english-kws-v7-voxcpm-blind-listening-review-v1",
        "status": "PENDING_HUMAN_REVIEW",
        "instructions": [
            "Listen to each clip without consulting the evaluation key.",
            "Score naturalness and pronunciation independently from 1 (poor) to 5 (excellent).",
            "Mark audible clipping, dropouts or artifacts.",
            "Do not infer speaker identity; all donors are synthetic references.",
        ],
        "reviewer_id": "",
        "reviewed_at_utc": "",
        "items": items,
        "training_allowed": False,
    }
    write_once(blind_root / "REVIEW_FORM.json", form)
    write_once(blind_root / "FILES.json", {
        "schema": "english-kws-v7-voxcpm-blind-review-files-v1",
        "files": pilot.file_inventory(blind_root),
    })
    write_once(Path(evaluation_output) / "BLIND_REVIEW_KEY.json", {
        "schema": "english-kws-v7-voxcpm-blind-review-key-v1",
        "items": key,
        "keep_separate_until_blind_review_is_submitted": True,
    })
    return blind_root


def evaluate(source_root, observer_root, output_root, blind_output):
    source_root = Path(source_root)
    source, source_sha = observer.validate_source(source_root)
    observer_packets, observer_summaries = _read_observers(source_root, observer_root)
    output = pilot.create_output_root(output_root)
    source_rows = source["rows"]
    audio_gate = _audio_gate(source_rows)
    candidate_results, by_candidate = _candidate_results(
        source_rows, observer_packets, observer_root)
    support_gate = _support_gate(source_rows, by_candidate)
    observer_gate = all(
        summaries["completed_count"] == 12 and summaries["status"] == "PASS"
        for summaries in observer_summaries.values())
    source_gate = source.get("status") == "COMPLETE" and len(source_rows) == 12
    machine_pass = source_gate and audio_gate["pass"] and observer_gate and support_gate["pass"]
    outcome = ("PILOT_MACHINE_GATES_PASS_HUMAN_REVIEW_REQUIRED"
               if machine_pass else "PILOT_FAIL_NO_SCALE_UP")
    write_once(output / "MACHINE_EVIDENCE.json", {
        "schema": "english-kws-v7-voxcpm-machine-evidence-v1",
        "source_sha256": source_sha,
        "observers": observer_summaries,
        "training_allowed": False,
        "optimizer_updates": 0,
    })
    write_once(output / "CANDIDATE_RESULTS.json", {
        "schema": "english-kws-v7-voxcpm-candidate-results-v1",
        "rows": candidate_results,
        "by_candidate": by_candidate,
        "training_allowed": False,
    })
    evaluation = {
        "schema": SCHEMA, "outcome": outcome,
        "source_sha256": source_sha,
        "source_status": source["status"],
        "source_denominator": 12,
        "source_successful_clips": source["successful_final_clips"],
        "source_base_tts_attempts": source["actual_base_tts_attempts"],
        "source_base_tts_returns": source["base_tts_returns"],
        "audio_geometry_clipping_gate": audio_gate,
        "observer_gate": {
            "pass": observer_gate, "providers": observer_summaries,
            "expected_calls_per_provider": 12, "retries": 0,
        },
        "candidate_joint_support_gate": support_gate,
        "machine_gates_pass": machine_pass,
        "naturalness_review": "PENDING_BLIND_HUMAN_LISTENING",
        "blind_human_review_required_before_scale_up": True,
        "scale_up_allowed": False,
        "training_allowed": False,
        "human_gold": False,
        "independent_human_speaker_identity_claimed": False,
        "real_human_far_field_quality_claimed": False,
        "calibration_locked_test_opened": False,
        "new_kws_forwards": 0, "optimizer_updates": 0,
        "product_promotion_allowed": False,
    }
    write_once(output / "PILOT_EVALUATION.json", evaluation)
    if machine_pass:
        _write_blind_review(source_root, source_rows, blind_output, output)
    else:
        write_once(output / "BLIND_REVIEW_NOT_ISSUED.json", {
            "reason": "MACHINE_GATES_NOT_PASSED",
            "scale_up_allowed": False,
            "training_allowed": False,
        })
    write_once(output / "FILES.json", {
        "schema": SCHEMA + "/files", "self_excluded": "FILES.json",
        "files": pilot.file_inventory(output),
    })
    return evaluation


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("source-check")
    run_parser = sub.add_parser("evaluate")
    run_parser.add_argument("--source", type=Path, required=True)
    run_parser.add_argument("--observers", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--blind-output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "source-check":
        print(json.dumps(source_check(), sort_keys=True))
    else:
        result = evaluate(args.source, args.observers, args.output, args.blind_output)
        print(json.dumps({
            "outcome": result["outcome"],
            "machine_gates_pass": result["machine_gates_pass"],
            "scale_up_allowed": result["scale_up_allowed"],
            "training_allowed": False,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
