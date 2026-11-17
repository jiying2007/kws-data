#!/usr/bin/env python3
"""Compare saved dense-CE outputs with aug96; failed temporal is secondary.

Standard-library data inspection only.  The shared descriptive reader is pinned
by SHA and never changed.  No model, optimizer, probability calculation, PCM
mixture, threshold search, or fresh frontend evaluation runs in this program.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path


SHARED_REVIEW_SHA = "5dfb3d4eac4de9f8b3b108bb275655792dfe2bf8af10ff4fd330bec9c22553eb"
TEMPORAL_IDENTITY = {"bytes": 17980402,
    "sha256": "6a1e38febf22b76a7cf2227362e51f783d5ad524b2d81b5677c22edc6dccc631"}
CONDITIONS = ("clean", "snr10", "snr5")


def shared_review(path=None):
    if path is None:
        candidates = [Path(__file__).resolve().parents[2] / "english_temporal/review/compare_temporal.py",
                      Path(__file__).resolve().parents[2] / "english_temporal_quality_20261117/review/compare_temporal.py",
                      Path(__file__).resolve().with_name("compare_temporal.py")]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise ValueError("provide --temporal-review-module or include the pinned compare_temporal.py")
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHARED_REVIEW_SHA:
        raise ValueError("shared descriptive reader identity does not match")
    spec = importlib.util.spec_from_file_location("saved_temporal_descriptive_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"bytes": len(raw), "sha256": SHARED_REVIEW_SHA}


def historical_context(base, previous_path, temporal_path, training_path=None):
    """Read existing control/failed-arm traces without a fabricated dense result."""
    old = base.read_previous(previous_path)
    temporal = base.Archive(Path(temporal_path).read_bytes())
    base.need(temporal.identity == TEMPORAL_IDENTITY, "authenticated failed temporal artifact")
    temporal_result = temporal.json("evidence/RESULT.json")
    base.need(temporal_result["status"] == "COMPLETED_WEAK_TEMPORAL_QUALITY_RESEARCH"
              and temporal_result["research_retention"]["passed"] is False,
              "failed temporal arm is secondary evidence")
    roster, noisy = old["roster"], old["snr10_roster"]
    base.need(temporal.read("sources/STREAM_ROSTER.json") == old["archive"].read("sources/STREAM_ROSTER.json")
              and temporal.read("cache/snr10-roster.json") == old["archive"].read("cache/mixed-roster.json"),
              "unchanged historical source/10 dB rosters")
    snr5 = temporal.json("cache/snr5-roster.json")
    receipts = temporal.json("evidence/SNR5_RECEIPTS.json")
    base.need(len(snr5) == 71 and len(receipts) == 77, "fixed previously evaluated 5 dB denominator")
    rosters = {"clean": roster, "snr10": noisy, "snr5": snr5}
    data = {}
    for condition, rows in rosters.items():
        control = old["data"][condition] if condition != "snr5" else base.read_predictions(temporal, "aug96-snr5", rows)
        data[condition] = {"aug96": control,
            "failed_temporal": base.read_predictions(temporal, "temporal-" + condition, rows)}
    data["snr5"]["baseline"] = base.read_predictions(temporal, "baseline-snr5", snr5)
    features, original_source, training_identity = None, None, None
    if training_path is not None:
        training = base.Archive(Path(training_path).read_bytes())
        base.need(training.identity["sha256"] == base.TRAINING_SHA, "optional original training identity")
        feature_raw = training.read("FEATURE_EVIDENCE.json")
        base.need(feature_raw == temporal.read("sources/FEATURE_EVIDENCE.json"), "original supported-output metadata")
        features = {row["id"]: row for row in base.parse_json(feature_raw)}
        snapshot = base.Archive(training.read("SOURCE_SNAPSHOT.zip")).json("SNAPSHOT.json")
        original_source = {row["cell_id"]: row for row in snapshot["source_rows"]}
        training_identity = training.identity
    return {"old": old, "temporal": temporal, "temporal_result": temporal_result,
        "rosters": rosters, "receipts": receipts, "data": data, "features": features,
        "original_source": original_source, "training_identity": training_identity}


def paired_cases(base, condition, aliases, reference_name, reference, candidate):
    rows = []
    for case_id, old in reference["metrics"].items():
        pair = base.compare_case(condition, aliases[case_id], old, candidate["metrics"][case_id])
        pair["reference"] = pair.pop("incumbent")
        pair["candidate"] = pair.pop("temporal")
        pair["reference_model"] = reference_name
        pair["candidate_model"] = "dense"
        rows.append(pair)
    return rows


def summarize_pairs(rows):
    shifts = [r["single_matching_event_shift_ms"] for r in rows if r["single_matching_event_shift_ms"] is not None]
    return {"cases": len(rows), "rescued": sum(r["rescued"] for r in rows),
        "regressed": sum(r["regressed"] for r in rows),
        "development_misses_rescued": sum(r["development_miss_rescued"] for r in rows),
        "cases_with_event_error_increase": sum(bool(r["event_error_categories_increased"]) for r in rows),
        "single_matching_event_clock_shift_ms_histogram": dict(sorted(collections.Counter(str(v) for v in shifts).items())),
        "event_clock_is_not_word_end_latency": True}


def retention_from_primary(pairs, snr5_rows):
    criteria = {condition + "_no_previously_correct_case_regression": not any(r["regressed"] for r in pairs[condition])
                for condition in CONDITIONS}
    criteria["snr10_at_least_one_rescued_development_miss"] = any(r["development_miss_rescued"] for r in pairs["snr10"])
    criteria["all_conditions_no_per_case_event_error_increase"] = not any(
        r["event_error_categories_increased"] for rows in pairs.values() for r in rows)
    criteria["snr5_both_development_keywords_present"] = all(
        any(r["role"] == "development" and r["weak_class"] == label for r in snr5_rows) for label in (1, 2))
    return {"criteria": criteria, "passed": all(criteria.values()), "reference": "AUG96_ONLY",
            "formal_promotion_allowed": False}


def render_text(report):
    lines = ["英文最后一个 dense-CE96 arm：已保存结果比较", "",
        "主对照：aug96。失败 temporal 仅作旁证；相对它改善不替代相对 aug96 的保留门。",
        "全部读取保存的 frames/events，没有模型执行、概率重算、PCM混音、前端或阈值搜索。",
        "5dB已在上一轮暴露，本轮仅复用缓存评估，不能称本轮未见过的测试条件。", "",
        "一、主对照结果"]
    for condition, summary in report["primary_aug96_to_dense_summary"].items():
        lines.append(condition + ": " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    lines.append("相对aug96保留门：" + json.dumps(report["derived_primary_retention"], ensure_ascii=False, sort_keys=True))
    lines.extend(["", "二、相对失败temporal的旁证"])
    for condition, summary in report["secondary_temporal_to_dense_summary"].items():
        lines.append(condition + ": " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    lines.extend(["", "三、逐条件 / 角色 / 类别汇总"])
    for r in report["aggregates"]:
        lines.append("{condition:5} {model:15} {role:11} {keyword:29} cases={cases} matched={matched} errors={research_error} prefix={prefix_events} wrong={wrong_keyword_events} extra={extra_same_keyword_events} neg_events={weak_negative_intent_events}".format(**r))
    lines.extend(["", "四、全部11个固定旧alias的开发正意图", "",
                  "每行：condition/model/event-correct/peak/达门总帧/最长连帧/原clipEOF相对事件ms。",
                  "三帧min→max是保存概率的描述统计，不执行带冷却状态的确认门。"])
    for row in report["development_positives"]:
        lines.append("\n" + row["case_alias"] + " " + row["keyword"])
        for condition, item in row["conditions"].items():
            for model, summary in item["models"].items():
                lines.append("  %s/%s/%s/%.9f/%d/%d/%s" % (condition, model, not summary["research_error"],
                    summary["max_expected_probability"], summary["target_threshold_frame_count"],
                    summary["longest_eligible_streak"], summary["event_minus_clip_eof_ms"]))
                for scope, triple in summary["triplet_diagnostics"].items():
                    if triple is not None:
                        lines.append("    %s max(min3)=%s minus_threshold=%s" % (scope,
                            triple["best_consecutive_triplet_min_probability"], triple["best_triplet_min_minus_threshold"]))
    lines.extend(["", "五、上一轮新增朗读事件5338的旁证"])
    for model, row in report["previous_read_speech_regression_case"]["models"].items():
        lines.append(model + ": " + json.dumps(row, ensure_ascii=False, sort_keys=True))
    lines.extend(["", "六、边界", *report["limitations"], "",
                  "全部主/次对照逐例事件、来源身份、分母与三帧统计见DENSE_COMPARISON.json。"])
    return "\n".join(lines) + "\n"


def compare(quality_path, previous_path, temporal_path, output_dir, training_path=None, shared_path=None):
    base, shared_identity = shared_review(shared_path)
    history = historical_context(base, previous_path, temporal_path, training_path)
    old, temporal, data, rosters = history["old"], history["temporal"], history["data"], history["rosters"]
    quality = base.Archive(Path(quality_path).read_bytes())
    result = quality.json("evidence/RESULT.json")
    base.need(result["status"] == "COMPLETED_WEAK_DENSE_QUALITY_RESEARCH"
        and result["schema"] == "english-dense-quality-once-v1"
        and result["training_source"]["sha256"] == base.TRAINING_SHA
        and result["incumbent_source"]["sha256"] == base.PREVIOUS_SHA
        and result["temporal_source"]["sha256"] == TEMPORAL_IDENTITY["sha256"], "completed fixed dense sources")
    base.need(result["formal_training_admitted"] is False and result["model_promotion_allowed"] is False,
              "research-only interpretation")
    plan = quality.json("evidence/PLAN.json")
    base.need(plan["threshold"] == 0.8 and plan["confirm_embeddings"] == 3 and plan["cooldown_embeddings"] == 13
        and plan["positive_loss"] == "MEAN_WINDOW_TARGET_CROSS_ENTROPY"
        and plan["snr5_training_allowed"] is False and plan["maximum_embedding_calls"] == 0
        and plan["maximum_onnx_forwards"] == 0 and plan["last_additional_objective_arm"] is True,
        "fixed final cache-only dense protocol")
    counts = result["new_counts"]
    base.need(all(counts[key] == 0 for key in ("new_embedding_calls_started", "new_embedding_calls_completed",
        "onnx_forward_started", "onnx_forward_completed", "new_training_frontend_calls", "new_tts_asr_phone_calls")),
        "no new frontend or voice-observer execution")
    copied = [("sources/STREAM_ROSTER.json", old["archive"], "sources/STREAM_ROSTER.json"),
        ("cache/snr10-roster.json", old["archive"], "cache/mixed-roster.json"),
        ("cache/clean-original.npz", old["archive"], "cache/clean-original.npz"),
        ("cache/mixed-snr10.npz", old["archive"], "cache/mixed-embeddings.npz")]
    copied.extend((name, temporal, name) for name in ("cache/snr5-roster.json", "cache/snr5-embeddings.npz",
                  "evidence/SNR5_SUMMARY.json", "evidence/SNR5_RECEIPTS.json"))
    for condition in ("clean", "snr10"):
        for suffix in ("-frames.jsonl", "-events.json"):
            filename = "candidate-" + condition + suffix
            copied.append(("predictions/reused/" + filename, old["archive"], "predictions/" + filename))
    for name in ("baseline-snr5", "aug96-snr5"):
        for suffix in ("-frames.jsonl", "-events.json"):
            copied.append(("predictions/reused/" + name + suffix, temporal, "predictions/" + name + suffix))
    for destination, archive, source in copied:
        base.need(quality.read(destination) == archive.read(source), "immutable reused data/predictions: " + destination)
    if history["features"] is not None:
        base.need(quality.read("sources/FEATURE_EVIDENCE.json") == temporal.read("sources/FEATURE_EVIDENCE.json"),
                  "original supported-output evidence unchanged")
    aliases, bindings = old["aliases"], old["bindings"]
    aggregates, primary, secondary = [], {}, {}
    for condition, rows in rosters.items():
        data[condition]["dense"] = base.read_predictions(quality, "dense-" + condition, rows)
        for model, values in data[condition].items():
            aggregates.extend(base.aggregate(condition, model, values))
        primary[condition] = paired_cases(base, condition, aliases, "aug96",
            data[condition]["aug96"], data[condition]["dense"])
        secondary[condition] = paired_cases(base, condition, aliases, "failed_temporal",
            data[condition]["failed_temporal"], data[condition]["dense"])
    derived = retention_from_primary(primary, rosters["snr5"])
    base.need(derived["criteria"] == result["research_retention"]["criteria"]
              and derived["passed"] == result["research_retention"]["passed"], "reported gate agrees with aug96 pairing")
    development, source_details = [], {}
    triplet_changes = {scope: {} for scope in ("full_observed_ready", "original_clip_supported_head_outputs")}
    for row in rosters["clean"]:
        if row["role"] != "development" or row["weak_class"] == 0:
            continue
        case_id, alias = row["id"], aliases[row["id"]]
        ends = history["features"][case_id]["head_instance_ends"] if history["features"] is not None else None
        conditions = {}
        for condition, models in data.items():
            base.need(case_id in models["dense"]["metrics"], "all eleven development positive conditions retained")
            conditions[condition] = {"models": {model: base.probability_summary(row,
                values["frames"][case_id], values["metrics"][case_id], ends) for model, values in models.items()}}
        development.append({"case_alias": alias, "id": case_id, "keyword": base.KEYWORDS[row["weak_class"]],
            "weak_class": row["weak_class"], "source_binding": bindings[alias], "conditions": conditions})
        if history["original_source"] is not None:
            source = history["original_source"][case_id]
            base.need(source["requested_keyword_index"] == row["weak_class"]
                      and source["pcm_sha256"] == bindings[alias]["source_pcm_sha256"], "original source canonical binding")
            source_details[alias] = {key: source.get(key) for key in ("canonical_phrase", "target_ipa", "role",
                "source_family_id", "variant", "family_exposed_in_rendering_pilot")}
    for scope in triplet_changes:
        for condition in CONDITIONS:
            deltas = []
            for row in development:
                models = row["conditions"][condition]["models"]
                a = models["aug96"]["triplet_diagnostics"][scope]
                b = models["dense"]["triplet_diagnostics"][scope]
                if a is not None and b is not None:
                    deltas.append(b["best_consecutive_triplet_min_probability"] - a["best_consecutive_triplet_min_probability"])
            triplet_changes[scope][condition] = {"increased": sum(v > 0 for v in deltas),
                "decreased": sum(v < 0 for v in deltas), "equal": sum(v == 0 for v in deltas), "cases": len(deltas)}
    regression_id, prior_triplet = "libri-speaker-5338", [228480, 229760, 231040]
    regression = {"case_alias": aliases[regression_id], "id": regression_id, "role": "development",
        "source_kind": "exposed-read-speech-splice", "trusted_negative": False,
        "prior_temporal_event_keyword": {"index": 2, "name": "Hey Mero"},
        "prior_event_sample": 231040, "prior_event_stream_seconds": 14.44,
        "prior_trigger_triplet_samples": prior_triplet, "models": {}}
    for model, values in data["clean"].items():
        records = values["frames"][regression_id]
        at = {r["sample"]: r for r in records}
        regression["models"][model] = {"events": values["metrics"][regression_id]["events"],
            "prior_trigger_triplet_mero_probabilities": [at[t]["probabilities"][2] for t in prior_triplet],
            "mero_full_ready_triplet": base.triplet_diagnostic([r for r in records if r["ready"]], 2, "FULL_OBSERVED_READY_OUTPUTS")}
    report = {"schema": "english-dense-saved-predictions-comparison-v1",
        "status": "SAVED_DENSE_PREDICTIONS_COMPARED_NO_MODEL_EXECUTION", "run_id": result["run_id"],
        "head_sha": result["head_sha"], "quality_artifact": quality.identity,
        "previous_aug96_artifact": old["archive"].identity, "failed_temporal_artifact": temporal.identity,
        "optional_training_artifact": history["training_identity"], "shared_descriptive_reader": shared_identity,
        "primary_control": "AUG96", "failed_temporal_is_primary_control": False,
        "class_mapping": base.KEYWORDS, "source_bindings_by_alias": bindings,
        "alias_basis": {"prior_paired_review_sha256": base.ALIAS_REPORT_SHA,
                        "all_eleven_original_development_aliases_checked": True},
        "denominators": {"clean_cases": 117, "snr10_cases": 76, "planned_snr5_cases": 77,
            "eligible_snr5_cases": 71, "excluded_snr5_cases": 6, "evaluated_case_conditions": 264,
            "development_positive_source_cases": 11, "independent_source_count_increment": 0,
            "new_frontend_calls": 0, "snr5_previously_exposed": True},
        "primary_aug96_to_dense_summary": {c: summarize_pairs(rows) for c, rows in primary.items()},
        "secondary_temporal_to_dense_summary": {c: summarize_pairs(rows) for c, rows in secondary.items()},
        "primary_aug96_to_dense_pairs": primary, "secondary_temporal_to_dense_pairs": secondary,
        "derived_primary_retention": derived, "reported_primary_retention": result["research_retention"],
        "aggregates": aggregates, "development_positives": development,
        "development_triplet_changes_aug96_to_dense": triplet_changes,
        "optional_original_source_details": source_details or None,
        "previous_read_speech_regression_case": regression,
        "excluded_snr5_cases": [{**r, "case_alias": aliases[r["id"]], "scored_as_success_or_failure": False}
                                for r in history["receipts"] if r["status"] != "MIXED"],
        "quality_member_identities": quality.member_identities,
        "previous_member_identities": old["archive"].member_identities,
        "temporal_member_identities": temporal.member_identities,
        "new_model_calls_by_this_review": 0, "new_probability_computations_by_this_review": 0,
        "new_audio_decodes_or_mixes_by_this_review": 0, "threshold_searches_by_this_review": 0,
        "limitations": ["主对照固定为aug96；相对失败temporal的修复只作旁证。",
            "同一批原来源、开发模式和5dB结果已经暴露；这不是新的独立测试集。",
            "11个开发正意图仍是原弱标签；三条件不计为33个独立样本。",
            "没有词尾标注；事件仅相对原clipEOF，未出现事件保持观察窗口删失。",
            "三帧min→max只描述保存概率，不执行确认门或推断冷却、未来帧与音素根因。",
            "背景和普通负意图trusted_negative=false；计数不代表独立自然FAR。",
            "正式训练准入、模型晋级和产品SDK替换不由本报告授予。",
            "本轮是最后一个同池目标函数比较；不据其结果继续搜索loss、seed或阈值。"],
        "formal_training_admitted": False, "model_promotion_allowed": False,
        "word_end_latency": "NOT_MEASURED", "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN"}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "DENSE_COMPARISON.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "DENSE_COMPARISON.txt").open("x", encoding="utf-8") as stream:
        stream.write(render_text(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--temporal-review-module", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.quality, args.previous, args.temporal, args.output_dir,
                     args.training, args.temporal_review_module)
    print(json.dumps({"status": report["status"], "run_id": report["run_id"],
        "primary": report["primary_aug96_to_dense_summary"], "retention": report["derived_primary_retention"],
        "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
