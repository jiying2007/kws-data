#!/usr/bin/env python3
"""Explain saved temporal-quality predictions; standard library, read-only.

Reads probabilities and events already stored by the bounded worker.  It does
not evaluate a model, calculate logits/probabilities, mix/decode PCM, change a
threshold, select a checkpoint, or assign acoustic truth to weak-intent labels.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile


PREVIOUS_BYTES = 32878537
PREVIOUS_SHA = "acc6def807b1dd5728111beba4801352d8de9cb453e530f2270005254f44fa15"
TRAINING_SHA = "2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1"
ALIAS_REPORT_SHA = "1cd425a924ddff0ef6881c0ec8206c4d508f52334fb9159be642a7a1654996b6"
KEYWORDS = {0: "background_or_negative_intent", 1: "Hey Vosen", 2: "Hey Mero"}
THRESHOLD = struct.unpack("<f", struct.pack("<f", 0.8))[0]
HOP = 1280
SAMPLE_RATE = 16000

# Copied from the authenticated prior paired-quality-review source bindings.
# These aliases must survive later sorting, changed conditions, and exclusions.
DEV_BINDINGS = {
    "DEV-VOSEN-01": (90, "enfinal144-141349b6a5ed3f6bae6a8c64", "0937a6e81fadce8ff06a72e3f37bc15e05dfdff2a306e90638d127fd55e91652"),
    "DEV-VOSEN-02": (91, "enfinal144-899f0b4f1eb91bc7d265b647", "986cc3d2f8a9d06d6dcf34d4f401d13fb5c16201c0d0a4b1ce71630fca453568"),
    "DEV-VOSEN-03": (92, "enfinal144-c16ecd5a55ad6e4e9ccff5c7", "703013cebd1293edea58240d32901c01a42ba423a2014373bcf3c2a4f923f488"),
    "DEV-MERO-01": (87, "enfinal144-8c88701e96689dfa5325d59a", "7348be432dbd8e13b7797aba09341de090aa818e4aa9c9abd1d73b6fb8bc67f2"),
    "DEV-MERO-02": (88, "enfinal144-66d2ba1c908c2398f4f00e04", "9a0d5b42c50e1ffa7e4f318b7ea2b34a524ada59b7dee727af0b8e0ff827412d"),
    "DEV-MERO-03": (89, "enfinal144-8af3e72998c16cf0c2378e6f", "2a72b18b0bee4e267cc0e03fd028827c0be1e93330fa62230628e6ec67c780ec"),
    "DEV-MERO-04": (101, "enfinal144-a0976f1f5f7b334993a7b4a6", "0241affd23716a91271ff98b5d09cdda26fc2704f3fd7e857187f3bd113183bf"),
    "DEV-MERO-05": (102, "enfinal144-5602dd573b0afb64e36b51b2", "3204b743b78360f8b2b743db64c51c6c421aeaf2fab255bfff2a143fe075e3b5"),
    "DEV-MERO-06": (103, "enfinal144-4ea8ef8e5ef11eadcfc863b9", "078f5ab975df1d9cbc34b7e5427623f6f30828b51319221e01bb8b932550451c"),
    "DEV-MERO-07": (104, "enfinal144-a5a700f0030bd649e0d17ae0", "f86ae491992879a19b390fb248adbdab0bd268d7b712a5b2229544cd527383ab"),
    "DEV-MERO-08": (105, "enfinal144-d692d0ef9dc69703364b9640", "2a67746ad789824a06346010e38b723c8fa33abb034ed4e221561502ff332bd7"),
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


def parse_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


class Archive:
    def __init__(self, raw):
        self.identity = identity(raw)
        self.zip = zipfile.ZipFile(io.BytesIO(raw))
        infos = self.zip.infolist()
        need(len(infos) == len({i.filename for i in infos}) <= 2000,
             "unique bounded ZIP members")
        need(sum(i.file_size for i in infos) <= 512 * 1024**2, "bounded ZIP expansion")
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = stat.S_IFMT(info.external_attr >> 16)
            need(not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
                 and path.as_posix() == info.filename and not info.is_dir()
                 and mode in (0, stat.S_IFREG), "safe regular ZIP member")
        self.member_identities = {}

    def read(self, name):
        raw = self.zip.read(name)
        self.member_identities[name] = identity(raw)
        return raw

    def json(self, name):
        return parse_json(self.read(name))


def build_aliases(roster):
    aliases, bindings, counters = {}, {}, collections.Counter()
    for index, row in enumerate(roster):
        role = {"train": "TRAIN", "development": "DEV"}[row["role"]]
        if row["weak_class"]:
            group = "VOSEN" if row["weak_class"] == 1 else "MERO"
        else:
            group = "READ-SPEECH" if row["id"].startswith("libri-") else "NEG-INTENT"
        prefix = role + "-" + group
        counters[prefix] += 1
        alias = prefix + "-%02d" % counters[prefix]
        need(row["id"] not in aliases and alias not in bindings, "unique stable case aliases")
        aliases[row["id"]] = alias
        bindings[alias] = {"id": row["id"], "case_id_sha256": sha(row["id"].encode()),
            "stream_roster_index": index, "role": row["role"], "weak_class": row["weak_class"],
            "source_pcm_sha256": row["source"].get("source_pcm_sha256", row["source"].get("pcm_sha256"))}
    for alias, (index, case_id, pcm_sha) in DEV_BINDINGS.items():
        need(aliases.get(case_id) == alias and bindings[alias]["stream_roster_index"] == index
             and bindings[alias]["source_pcm_sha256"] == pcm_sha,
             "prior fixed development alias binding: " + alias)
    need(sum(r["role"] == "development" and r["weak_class"] > 0 for r in roster) == 11,
         "all eleven prior development positive intents")
    return aliases, bindings


def event_metrics(row, events):
    source, label = row["source"], row["weak_class"]
    start, eof = source.get("clip_start_sample"), source.get("original_clip_eof_sample")
    need(type(label) is int and label in KEYWORDS and row["role"] in ("train", "development"),
         "fixed weak class and role")
    need((label == 0 and start is None and eof is None) or
         (type(start) is int and type(eof) is int and 0 < start < eof <= row["observed_samples"]),
         "original clip/observed geometry")
    need(all(type(e["sample"]) is int and 0 < e["sample"] <= row["observed_samples"]
             and type(e["keyword"]) is int and e["keyword"] in (1, 2) for e in events)
         and all(a["sample"] < b["sample"] for a, b in zip(events, events[1:])), "observed event clocks")
    prefix = [e for e in events if start is not None and e["sample"] <= start]
    after = [e for e in events if start is None or e["sample"] > start]
    matching = [e for e in after if label and e["keyword"] == label]
    wrong = [e for e in after if label and e["keyword"] != label]
    extra, missing = max(0, len(matching) - 1), bool(label and not matching)
    negative = len(events) if label == 0 else 0
    return {"id": row["id"], "role": row["role"], "weak_class": label,
        "keyword": KEYWORDS[label], "event_count": len(events), "events": events,
        "matched": bool(matching), "prefix_events": len(prefix), "wrong_keyword_events": len(wrong),
        "extra_same_keyword_events": extra, "weak_negative_intent_events": negative,
        "no_matching_event_in_observed_context": missing,
        "research_error": bool(prefix or wrong or extra or missing or negative),
        "matching_event_samples": [e["sample"] for e in matching],
        "event_minus_clip_eof_ms": [(e["sample"] - eof) / 16 for e in matching] if eof is not None else [],
        "clip_start_sample": start, "clip_eof_sample": eof, "observed_samples": row["observed_samples"],
        "observation_end_minus_clip_eof_ms": (row["observed_samples"] - eof) / 16 if eof is not None else None,
        "no_matching_event_status": "CENSORED_AT_OBSERVED_END_NOT_FRR" if missing else None,
        "word_end_latency": "NOT_MEASURED"}


def read_predictions(archive, name, roster):
    """Validate saved frame/event agreement without recomputing probabilities."""
    raw_events = archive.json("predictions/" + name + "-events.json")
    need([r["id"] for r in raw_events] == [r["id"] for r in roster], "prediction event roster order: " + name)
    frames = {row["id"]: [] for row in roster}
    sequence = []
    for line in archive.read("predictions/" + name + "-frames.jsonl").splitlines():
        record = parse_json(line)
        case_id = record["id"]
        need(case_id in frames and type(record["sample"]) is int and type(record["ready"]) is bool
             and type(record["keyword"]) is int and record["keyword"] in (0, 1, 2),
             "saved frame fields")
        probabilities = record["probabilities"]
        need(isinstance(probabilities, list) and len(probabilities) == 3
             and all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities),
             "finite stored three-class probabilities")
        if record["ready"]:
            need(abs(sum(probabilities) - 1) <= 2e-6, "stored ready probabilities sum to one")
        else:
            need(probabilities == [0.0, 0.0, 0.0] and record["keyword"] == 0, "unready frame emits no event")
        frames[case_id].append(record)
        sequence.append(case_id)
    metrics = {}
    cursor = 0
    for row, reported in zip(roster, raw_events):
        records = frames[row["id"]]
        expected_samples = list(range(16000, row["observed_samples"] + 1, HOP))
        need([r["sample"] for r in records] == expected_samples
             and sequence[cursor:cursor + len(records)] == [row["id"]] * len(records),
             "complete ordered stored frame clocks: " + name)
        need([r["ready"] for r in records] == [i >= 15 for i in range(len(records))],
             "original head warm-up and readiness")
        cursor += len(records)
        events = [{"sample": r["sample"], "keyword": r["keyword"]} for r in records if r["keyword"]]
        need(events == reported["events"] and reported["role"] == row["role"]
             and reported["weak_class"] == row["weak_class"]
             and reported["observed_samples"] == row["observed_samples"], "saved frame/event identity")
        calculated = event_metrics(row, events)
        checks = {"matched_intent_after_clip_start": calculated["matched"],
            "extra_same_keyword_events": calculated["extra_same_keyword_events"],
            "weak_negative_intent_events": calculated["weak_negative_intent_events"],
            "no_matching_event_in_observed_context": calculated["no_matching_event_in_observed_context"],
            "research_error": calculated["research_error"],
            "clip_start_sample": calculated["clip_start_sample"],
            "original_clip_eof_sample": calculated["clip_eof_sample"], "word_end_latency": "NOT_MEASURED"}
        need(all(reported[key] == value for key, value in checks.items())
             and len(reported["prefix_events"]) == calculated["prefix_events"]
             and len(reported["wrong_keyword_after_clip_start"]) == calculated["wrong_keyword_events"],
             "saved event summaries agree with observed events")
        metrics[row["id"]] = calculated
    need(cursor == len(sequence), "no unbound frames")
    return {"metrics": metrics, "frames": frames, "frame_count": len(sequence)}


def triplet_diagnostic(records, label, scope):
    """Min/max over stored probabilities, without model or event-gate execution."""
    triplets, best, above = 0, None, 0
    for index in range(len(records) - 2):
        triple = records[index:index + 3]
        if not (all(r["ready"] for r in triple)
                and len({r["id"] for r in triple}) == 1
                and triple[1]["sample"] - triple[0]["sample"] == HOP
                and triple[2]["sample"] - triple[1]["sample"] == HOP):
            continue
        probabilities = [r["probabilities"][label] for r in triple]
        score = min(probabilities)
        triplets += 1
        above += int(score >= THRESHOLD)
        if best is None or score > best["score"]:
            best = {"score": score, "samples": [r["sample"] for r in triple],
                    "stored_target_probabilities": probabilities}
    return {"scope": scope, "consecutive_triplets_observed": triplets,
        "best_consecutive_triplet_min_probability": best["score"] if best is not None else None,
        "best_triplet_min_minus_threshold": best["score"] - THRESHOLD if best is not None else None,
        "best_triplet_samples": best["samples"] if best is not None else [],
        "best_triplet_stored_target_probabilities": best["stored_target_probabilities"] if best is not None else [],
        "consecutive_triplets_at_threshold": above,
        "interpretation": "DESCRIPTIVE_MIN_MAX_OF_STORED_PROBABILITIES_NOT_A_MODEL_OR_EVENT_GATE_REEXECUTION",
        "does_not_account_for_cooldown_or_prior_events": True,
        "cross_case_or_missing_frame_triplets_allowed": False}


def probability_summary(row, frames, metrics, supported_ends=None):
    """Describe saved target probabilities after clip start, never future frames."""
    label, start, eof = row["weak_class"], row["source"]["clip_start_sample"], row["source"]["original_clip_eof_sample"]
    need(label in (1, 2), "positive-intent probability summary")
    observed = [r for r in frames if r["ready"] and r["sample"] > start]
    need(observed, "observed ready outputs after clip start")
    peak = max(observed, key=lambda r: r["probabilities"][label])
    threshold_frames, eligible_frames, runs, run = 0, 0, [], []
    previous_sample = None
    for record in observed:
        above = record["probabilities"][label] >= THRESHOLD
        target_wins = record["probabilities"][label] >= record["probabilities"][3 - label]
        threshold_frames += int(above)
        eligible = above and target_wins
        eligible_frames += int(eligible)
        if not eligible or (previous_sample is not None and record["sample"] - previous_sample != HOP):
            if run:
                runs.append(run)
                run = []
        if eligible:
            run.append(record["sample"])
        previous_sample = record["sample"]
    ending = len(run)
    if run:
        runs.append(run)
    longest = max((len(r) for r in runs), default=0)
    explanation = None
    if metrics["no_matching_event_in_observed_context"]:
        explanation = ("NO_THREE_CONSECUTIVE_TARGET_THRESHOLD_OUTPUTS_OBSERVED" if longest < 3 else
                       "THRESHOLD_RUN_PRESENT_WITHOUT_MATCHING_RECORDED_EVENT_GATE_STATE_NOT_INFERRED")
    full_triplets = triplet_diagnostic([r for r in frames if r["ready"]], label, "FULL_OBSERVED_READY_OUTPUTS")
    triplets = {"full_observed_ready": full_triplets,
        "after_clip_start_observed_ready": triplet_diagnostic(observed, label, "READY_OUTPUTS_AFTER_CLIP_START"),
        "original_clip_supported_head_outputs": None}
    if supported_ends is not None:
        need(supported_ends and len(set(supported_ends)) == len(supported_ends)
             and all(type(t) is int for t in supported_ends), "original support timestamps")
        by_sample = {r["sample"]: r for r in frames}
        need(all(t in by_sample and by_sample[t]["ready"] and t - 31840 <= start
                 and t - 128 >= eof and t <= row["observed_samples"] for t in supported_ends),
             "original whole-clip mel-supported head outputs")
        triplets["original_clip_supported_head_outputs"] = triplet_diagnostic(
            [by_sample[t] for t in sorted(supported_ends)], label, "ORIGINAL_FEATURE_EVIDENCE_HEAD_INSTANCE_ENDS")
    return {"single_frame_and_streak_scope": "STORED_READY_OUTPUTS_AFTER_CLIP_START_THROUGH_OBSERVED_END",
        "observed_ready_frames": len(observed), "max_expected_probability": peak["probabilities"][label],
        "peak_sample": peak["sample"], "peak_minus_clip_eof_ms": (peak["sample"] - eof) / 16,
        "target_threshold_frame_count": threshold_frames, "eligible_frame_count": eligible_frames,
        "longest_eligible_streak": longest, "ending_eligible_streak": ending,
        "eligible_streaks": [{"frames": len(r), "start_sample": r[0], "end_sample": r[-1],
            "start_minus_clip_eof_ms": (r[0] - eof) / 16, "end_minus_clip_eof_ms": (r[-1] - eof) / 16} for r in runs],
        "last_expected_probability": observed[-1]["probabilities"][label],
        "last_recorded_sample": observed[-1]["sample"],
        "last_six_frames": [{"sample": r["sample"], "minus_clip_eof_ms": (r["sample"] - eof) / 16,
            "probabilities_background_vosen_mero": r["probabilities"], "recorded_event_keyword": r["keyword"]}
            for r in observed[-6:]],
        "observed_miss_trace_description": explanation,
        "best_consecutive_triplet_min_probability": full_triplets["best_consecutive_triplet_min_probability"],
        "best_consecutive_triplet_scope": "FULL_OBSERVED_READY_OUTPUTS",
        "triplet_diagnostics": triplets,
        "unobserved_future_frames": "NOT_INFERRED", "acoustic_or_pronunciation_root_cause": "NOT_ESTABLISHED",
        "matched": metrics["matched"], "research_error": metrics["research_error"],
        "events": metrics["events"], "event_minus_clip_eof_ms": metrics["event_minus_clip_eof_ms"],
        "no_matching_event_status": metrics["no_matching_event_status"], "word_end_latency": "NOT_MEASURED"}


def read_previous(path):
    raw = Path(path).read_bytes()
    need(len(raw) == PREVIOUS_BYTES and sha(raw) == PREVIOUS_SHA, "authenticated prior quality ZIP")
    archive = Archive(raw)
    roster = archive.json("sources/STREAM_ROSTER.json")
    noisy = archive.json("cache/mixed-roster.json")
    need(len(roster) == 117 and len(noisy) == 76, "original clean and noisy denominators")
    aliases, bindings = build_aliases(roster)
    data = {condition: read_predictions(archive, "candidate-" + condition, rows)
            for condition, rows in (("clean", roster), ("snr10", noisy))}
    return {"archive": archive, "roster": roster, "snr10_roster": noisy,
            "aliases": aliases, "bindings": bindings, "data": data}


def aggregate(condition, model, data):
    groups = {}
    count_fields = ("matched", "research_error", "event_count", "prefix_events", "wrong_keyword_events",
                    "extra_same_keyword_events", "weak_negative_intent_events", "no_matching_event_in_observed_context")
    for row in data["metrics"].values():
        key = (row["role"], row["weak_class"])
        group = groups.setdefault(key, {"condition": condition, "model": model, "role": key[0],
            "weak_class": key[1], "keyword": KEYWORDS[key[1]], "cases": 0,
            **{field: 0 for field in count_fields}})
        group["cases"] += 1
        for field in count_fields:
            group[field] += int(row[field])
    return [groups[key] for key in sorted(groups)]


def compare_case(condition, alias, old, new):
    need(all(old[key] == new[key] for key in ("id", "role", "weak_class", "observed_samples",
                                             "clip_start_sample", "clip_eof_sample")), "paired case identity")
    error_fields = ("prefix_events", "wrong_keyword_events", "extra_same_keyword_events", "weak_negative_intent_events")
    one_each = len(old["matching_event_samples"]) == len(new["matching_event_samples"]) == 1
    shift = (new["matching_event_samples"][0] - old["matching_event_samples"][0]) / 16 if one_each else None
    return {"condition": condition, "case_alias": alias, "id": old["id"], "role": old["role"],
        "weak_class": old["weak_class"], "keyword": old["keyword"],
        "incumbent": old, "temporal": new,
        "rescued": old["research_error"] and not new["research_error"],
        "regressed": not old["research_error"] and new["research_error"],
        "development_miss_rescued": old["role"] == "development" and old["weak_class"] > 0
            and old["no_matching_event_in_observed_context"] and not new["research_error"],
        "event_error_categories_increased": [field for field in error_fields if new[field] > old[field]],
        "observed_event_sequence_changed": old["events"] != new["events"],
        "single_matching_event_shift_ms": shift,
        "event_shift_status": "ONE_MATCHING_EVENT_EACH" if one_each else "NOT_COMPARABLE_SINGLE_EVENT",
        "event_clock_interpretation": "SAME_CONSTRUCTED_STREAM_EVENT_CLOCK_NOT_WORD_END_LATENCY"}


def text_report(report):
    lines = ["英文 temporal loss：已保存结果逐例解释", "",
        "状态：仅读取既有 probabilities/events；未执行模型、概率计算、PCM 混音、训练或阈值搜索。",
        "本轮对照是旧 aug96；baseline 仅在 5 dB 提供附加参照。", "",
        "一、分母与数据边界", json.dumps(report["denominators"], ensure_ascii=False, sort_keys=True), "",
        "同源条件不是独立新录音。所有匹配率均为弱意图诊断；开发集此前已暴露。",
        "时间统一相对原 clip EOF，不是词尾；未观察的后续帧保持 UNKNOWN。", "",
        "二、逐条件 / 角色 / 类别"]
    for item in report["aggregates"]:
        lines.append("{condition:5} {model:8} {role:11} {keyword:29} cases={cases} matched={matched} errors={research_error} prefix={prefix_events} wrong={wrong_keyword_events} extra={extra_same_keyword_events} neg_events={weak_negative_intent_events}".format(**item))
    lines.extend(["", "三、incumbent → temporal 的事件质量变化"])
    for condition, summary in report["paired_summary"].items():
        lines.append(condition + ": " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    changed = [row for row in report["paired_cases"] if row["rescued"] or row["regressed"]
               or row["event_error_categories_increased"]]
    if not changed:
        lines.append("没有事件正确性修复、回退或错误类别增加；时间变化另列。")
    for row in changed:
        lines.append("{condition} {case_alias} rescued={rescued} regressed={regressed} increases={event_error_categories_increased} incumbent={old} temporal={new}".format(
            **row, old=row["incumbent"]["events"], new=row["temporal"]["events"]))
    lines.extend(["", "四、全部 11 个开发正意图：固定旧 alias", "",
                  "列：condition / model / event-correct / peak / 达阈值帧 / 最长连续帧 / EOF相对事件ms",
                  "阈值为 float32(0.8)，输出步长 80 ms；峰值与连续帧仅统计 clip start 后已观察 ready 输出。"])
    for row in report["development_positives"]:
        lines.append("\n" + row["case_alias"] + " " + row["keyword"])
        for condition in ("clean", "snr10", "snr5"):
            data = row["conditions"][condition]
            if not data["available"]:
                lines.append(condition + " EXCLUDED " + data["disposition"])
                continue
            for model, summary in data["models"].items():
                lines.append("  %s / %s / %s / %.9f / %d / %d / %s" % (
                    condition, model, not summary["research_error"], summary["max_expected_probability"],
                    summary["target_threshold_frame_count"], summary["longest_eligible_streak"],
                    summary["event_minus_clip_eof_ms"]))
                if summary["no_matching_event_status"]:
                    lines.append("    " + summary["no_matching_event_status"] + "; " + summary["observed_miss_trace_description"])
                for scope, diagnostic in summary["triplet_diagnostics"].items():
                    if diagnostic is None:
                        continue
                    lines.append("    triplet %s: max(min3)=%s, minus_threshold=%s, windows=%d" % (
                        scope, diagnostic["best_consecutive_triplet_min_probability"],
                        diagnostic["best_triplet_min_minus_threshold"], diagnostic["consecutive_triplets_observed"]))
            lines.append("  aug96→temporal 单一匹配事件时间差 ms：" + str(data["paired"]["single_matching_event_shift_ms"]))
    lines.extend(["", "五、5 dB 开发稳定性（同源条件比较）"])
    for row in report["development_condition_stability"]:
        lines.append(row["case_alias"] + ": " + json.dumps(row["models"], ensure_ascii=False, sort_keys=True))
    lines.extend(["", "六、边界", *report["limitations"], "",
                  "完整逐例事件、帧统计、原成员身份与别名绑定见同目录 TEMPORAL_COMPARISON.json。"])
    return "\n".join(lines) + "\n"


def compare(quality_path, previous_path, output_dir, training_path=None):
    previous = read_previous(previous_path)
    quality = Archive(Path(quality_path).read_bytes())
    roster, snr10 = previous["roster"], previous["snr10_roster"]
    need(quality.read("sources/STREAM_ROSTER.json") == previous["archive"].read("sources/STREAM_ROSTER.json")
         and quality.read("cache/snr10-roster.json") == previous["archive"].read("cache/mixed-roster.json"),
         "new artifact preserves original clean and 10 dB rosters")
    for condition in ("clean", "snr10"):
        for extension in ("events.json", "frames.jsonl"):
            name = "candidate-" + condition + "-" + extension
            need(quality.read("predictions/reused/" + name) == previous["archive"].read("predictions/" + name),
                 "incumbent saved prediction bytes reused exactly")
    result = quality.json("evidence/RESULT.json")
    need(result["status"] == "COMPLETED_WEAK_TEMPORAL_QUALITY_RESEARCH"
         and result["incumbent_source"]["sha256"] == PREVIOUS_SHA
         and result["training_source"]["sha256"] == TRAINING_SHA, "completed fixed research result source binding")
    need(result["formal_training_admitted"] is False and result["model_promotion_allowed"] is False,
         "research admission boundary unchanged")
    plan = quality.json("evidence/PLAN.json")
    need(plan["threshold"] == 0.8 and plan["confirm_embeddings"] == 3 and plan["cooldown_embeddings"] == 13
         and plan["snr_db"] == 5.0 and plan["snr5_training_allowed"] is False,
         "fixed event protocol and evaluation-only 5 dB")
    receipts = quality.json("evidence/SNR5_RECEIPTS.json")
    snr5 = quality.json("cache/snr5-roster.json")
    selected = [row for row in roster if not row["id"].startswith("libri-")]
    need([r["id"] for r in receipts] == [r["id"] for r in selected]
         and [r["id"] for r in snr5] == [r["id"] for r in receipts if r["status"] == "MIXED"]
         and len(receipts) == 77, "complete planned denominator and ordered eligible 5 dB subset")
    original_by_id = {row["id"]: row for row in roster}
    for row in snr5:
        need(all(row[key] == value for key, value in original_by_id[row["id"]].items()),
             "5 dB retains original role, class, and source metadata")
    aliases, bindings = previous["aliases"], previous["bindings"]
    data = {"clean": {"aug96": previous["data"]["clean"],
                       "temporal": read_predictions(quality, "temporal-clean", roster)},
            "snr10": {"aug96": previous["data"]["snr10"],
                       "temporal": read_predictions(quality, "temporal-snr10", snr10)},
            "snr5": {model: read_predictions(quality, model + "-snr5", snr5)
                     for model in ("baseline", "aug96", "temporal")}}
    aggregates, pairs, paired_summary, pair_index = [], [], {}, {}
    for condition, models in data.items():
        for model, predictions in models.items():
            aggregates.extend(aggregate(condition, model, predictions))
        condition_pairs = []
        for case_id, old in models["aug96"]["metrics"].items():
            pair = compare_case(condition, aliases[case_id], old, models["temporal"]["metrics"][case_id])
            condition_pairs.append(pair)
            pair_index[(condition, case_id)] = pair
        pairs.extend(condition_pairs)
        shifts = [r["single_matching_event_shift_ms"] for r in condition_pairs if r["single_matching_event_shift_ms"] is not None]
        paired_summary[condition] = {"cases": len(condition_pairs),
            "rescued": sum(r["rescued"] for r in condition_pairs),
            "regressed": sum(r["regressed"] for r in condition_pairs),
            "development_misses_rescued": sum(r["development_miss_rescued"] for r in condition_pairs),
            "cases_with_event_error_increase": sum(bool(r["event_error_categories_increased"]) for r in condition_pairs),
            "single_matching_event_clock_shift_ms_histogram": dict(sorted(collections.Counter(str(v) for v in shifts).items())),
            "event_clock_is_not_word_end_latency": True}
    receipt_by_id = {r["id"]: r for r in receipts}
    training_identity, source_rows, original_features = None, None, None
    if training_path is not None:
        training = Archive(Path(training_path).read_bytes())
        need(training.identity["sha256"] == TRAINING_SHA, "optional original training ZIP identity")
        snapshot = Archive(training.read("SOURCE_SNAPSHOT.zip")).json("SNAPSHOT.json")
        source_rows = {r["cell_id"]: r for r in snapshot["source_rows"]}
        raw_features = training.read("FEATURE_EVIDENCE.json")
        need(raw_features == quality.read("sources/FEATURE_EVIDENCE.json"), "original supported-output metadata reused")
        original_features = {r["id"]: r for r in parse_json(raw_features)}
        need(set(original_features) == set(original_by_id), "original feature evidence covers full roster")
        training_identity = training.identity
    dev_rows, stability = [], []
    for row in roster:
        if row["role"] != "development" or row["weak_class"] == 0:
            continue
        case_id = row["id"]
        conditions = {}
        for condition, models in data.items():
            if case_id not in models["temporal"]["metrics"]:
                conditions[condition] = {"available": False, "disposition": receipt_by_id[case_id]["status"],
                                         "scored_as_success_or_failure": False}
                continue
            conditions[condition] = {"available": True,
                "models": {model: probability_summary(row, values["frames"][case_id], values["metrics"][case_id],
                    original_features[case_id]["head_instance_ends"] if original_features is not None else None)
                           for model, values in models.items()},
                "paired": pair_index[(condition, case_id)]}
        dev_rows.append({"case_alias": aliases[case_id], "id": case_id, "keyword": KEYWORDS[row["weak_class"]],
            "weak_class": row["weak_class"], "source_binding": bindings[aliases[case_id]], "conditions": conditions})
        models = {}
        for model in ("aug96", "temporal"):
            states = {condition: not values[model]["metrics"][case_id]["research_error"]
                      if case_id in values[model]["metrics"] else None for condition, values in data.items()}
            models[model] = {"event_correct_by_condition": states,
                "correct_snr10_but_error_snr5": states["snr10"] is True and states["snr5"] is False,
                "error_snr10_but_correct_snr5": states["snr10"] is False and states["snr5"] is True,
                "all_three_conditions_event_correct": all(value is True for value in states.values())}
        stability.append({"case_alias": aliases[case_id], "id": case_id, "models": models,
                          "independent_source_count_increment": 0})
    original_source_details = None
    if source_rows is not None:
        original_source_details = {}
        for row in dev_rows:
            source = source_rows[row["id"]]
            need(source["requested_keyword_index"] == row["weak_class"]
                 and source["canonical_phrase"] == row["keyword"]
                 and source["pcm_sha256"] == row["source_binding"]["source_pcm_sha256"],
                 "optional original source canonical class/PCM binding")
            original_source_details[row["case_alias"]] = {key: source.get(key) for key in (
                "canonical_phrase", "requested_keyword_index", "target_ipa", "source_family_id",
                "role", "variant", "synthesis_text", "family_exposed_in_rendering_pilot")}
    excluded = [{**r, "case_alias": aliases[r["id"]], "scored_as_success_or_failure": False}
                for r in receipts if r["status"] != "MIXED"]
    report = {"schema": "english-temporal-saved-predictions-comparison-v1",
        "status": "SAVED_PREDICTIONS_COMPARED_NO_MODEL_EXECUTION", "run_id": result["run_id"],
        "head_sha": result["head_sha"], "quality_artifact": quality.identity,
        "previous_artifact": previous["archive"].identity, "optional_training_artifact": training_identity,
        "class_mapping": KEYWORDS, "alias_basis": {"source": "prior paired-quality-review.json",
            "sha256": ALIAS_REPORT_SHA, "ordering": "ORIGINAL_STREAM_ROSTER_ORDER_WITHIN_ROLE_AND_CLASS",
            "all_eleven_development_positive_bindings_checked": True},
        "denominators": {"original_synthetic_sources": 77, "original_synthetic_train": 50,
            "original_synthetic_development": 27, "original_read_speech_tracks": 40,
            "clean_cases": len(roster), "snr10_cases": len(snr10), "planned_snr5_cases": len(receipts),
            "eligible_snr5_cases": len(snr5), "excluded_snr5_cases": len(excluded),
            "development_positive_source_cases": 11, "independent_source_count_increment": 0},
        "protocol": {"threshold_nominal": 0.8, "threshold_float32": THRESHOLD, "confirm_embeddings": 3,
            "cooldown_embeddings": 13, "embedding_hop_ms": 80,
            "comparison": "AUG96_VS_TEMPORAL_EQUAL_DATA_DRAWS_AND_UPDATE_BUDGET",
            "baseline_snr5": "ADDITIONAL_REFERENCE_ONLY", "snr5_training_allowed": False},
        "aggregates": aggregates, "paired_summary": paired_summary, "paired_cases": pairs,
        "development_positives": dev_rows, "development_condition_stability": stability,
        "source_bindings_by_alias": bindings, "optional_original_source_details": original_source_details,
        "excluded_snr5_cases": excluded, "reported_research_retention": result["research_retention"],
        "artifact_member_identities": quality.member_identities,
        "previous_member_identities": previous["archive"].member_identities,
        "new_model_calls_by_this_review": 0, "new_probability_computations_by_this_review": 0,
        "new_audio_decodes_or_mixes_by_this_review": 0, "threshold_searches_by_this_review": 0,
        "limitations": [
            "这是已保存预测的解释工具，不替代完整来源/训练/数值独立审计。",
            "同一原录音的 clean/10 dB/5 dB 条件不是独立样本；不能据此宣称真实 FAR/FRR。",
            "development 已暴露，且此前失败模式参与提出研究假设；结果属于开发诊断。",
            "所有语音标签仍为原弱意图，没有新增发音真值、可信负例或正式训练准入。",
            "事件时间相对原 clip EOF；没有词尾标注，词尾延迟保持 NOT_MEASURED。",
            "无匹配事件仅表示已观察窗口内没有匹配；不推断窗口外未来帧或发音根因。",
            "三帧 min→max 仅对保存概率作描述统计；不执行模型、不改变阈值，也不代替带冷却和历史状态的实际事件门。",
            "5 dB 溢出排除保留完整计划分母，不计作模型成功或失败，未补选。",
            "模型晋级、正式资格验证与产品 SDK 替换不由本报告授予。"],
        "human_gold": False, "pronunciation_verified": False, "trusted_negative": False,
        "formal_training_admitted": False, "model_promotion_allowed": False,
        "independent_far": "NOT_RUN", "independent_frr": "NOT_RUN", "word_end_latency": "NOT_MEASURED"}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "TEMPORAL_COMPARISON.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with (destination / "TEMPORAL_COMPARISON.txt").open("x", encoding="utf-8") as stream:
        stream.write(text_report(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.quality, args.previous, args.output_dir, args.training)
    print(json.dumps({"status": report["status"], "run_id": report["run_id"],
        "denominators": report["denominators"], "paired_summary": report["paired_summary"],
        "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
