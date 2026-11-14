"""Project verified existing Chinese/English evidence into keyword-level rows.

This is a source-preserving consumer, not a new ASR runner or admission gate.
English orthographic observations never become acoustic labels. Chinese weak
targets/masks/weights are copied only after the existing native verifier passes.
No model calls, audio generation, changed splits, acoustic timestamps or gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "automation"))
POLICY = "kws-v4-existing-keyword-evidence-v1"
KEYWORDS = {"zh": (("zh-k1", "你好小窝"), ("zh-k2", "小窝小窝")),
            "en": (("en-k1", "Hey Vosen"), ("en-k2", "Hey Mero"))}
ENGLISH_SOURCES = {
    "3786baa9fd3575437a05e97956fafdf3298eac23de2538b485b8342ef4d25133": (
        "english-63-machine-review.json",
        "60054a3532491631a47b094ea7ff2bc86ef16530ff510f968bc9edcdd253a9ad",
        "english-63-three-asr-source-linked-review-v1"),
    "f83d6a2f399c20af5bbc920ef6c38c42ac1e44b516d2b7291f4510557491cec8": (
        "english63-three-evidence-review.json",
        "89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12",
        "english63-independent-evidence-joined-review-priority-v1"),
}
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_BYTES = 32 * 1024 * 1024
ENGLISH_USAGE = {
    "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY": "weak_diagnostic_only",
    "EXPOSED_DEVELOPMENT_NO_TUNE": "development_no_tune",
    "QUARANTINE_AMBIGUOUS_NEARPHONE": "nearphone_quarantine",
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, "DUPLICATE_JSON_KEY")
        out[key] = value
    return out


def load(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def read(path, maximum=MAX_BYTES):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
         and path.stat().st_size <= maximum, "INVALID_INPUT_FILE")
    return path.read_bytes()


def normalize(text, language):
    need(type(text) is str, "OBSERVATION_TEXT_REQUIRED")
    clean = unicodedata.normalize("NFKC", text).casefold()
    if language == "en":
        return re.findall(r"[^\W_]+(?:'[^\W_]+)?", clean)
    return "".join(c for c in clean if not c.isspace()
                   and not unicodedata.category(c).startswith("P"))


def positions(text, phrase, language):
    """Offsets in normalized text/tokens only; never acoustic event times."""
    haystack, needle = normalize(text, language), normalize(phrase, language)
    return [i for i in range(len(haystack) - len(needle) + 1)
            if haystack[i:i + len(needle)] == needle]


def keyword_row(identity, observations, phrase, keyword_id, language,
                target=None, mask=False, weight=0.0, label_tier="unknown"):
    need(type(mask) is bool and type(weight) in (int, float)
         and math.isfinite(weight) and 0 <= weight <= 1, "INVALID_WEAK_MASK_WEIGHT")
    need((mask and type(target) is int and target in (0, 1) and weight > 0)
         or (not mask and target is None and weight == 0), "WEAK_TARGET_MASK_CONFLICT")
    providers = [o["provider"] for o in observations]
    need(len(providers) == len(set(providers)) and len(providers) <= 3,
         "DUPLICATED_ASR_PROVIDER")
    evidence = []
    for observed in observations:
        text = observed.get("text")
        complete = observed.get("complete") is True
        usable = observed.get("usable") is True
        hits = positions(text, phrase, language) if type(text) is str else []
        evidence.append({**observed, "literal_present": bool(hits),
                         "normalized_match_positions": hits,
                         "position_unit": "token" if language == "en" else "character",
                         "usable_for_observation": complete and usable and bool(text)})
    positive_votes = sum(o["usable_for_observation"] and o["literal_present"] for o in evidence)
    usable_count = sum(o["usable_for_observation"] for o in evidence)
    agreement = ("three_literal_present" if positive_votes == 3 else
                 "three_literal_not_observed" if usable_count == 3 and positive_votes == 0 else
                 "literal_presence_disagreement" if usable_count == 3 else "incomplete_observations")
    return {**identity, "keyword_id": keyword_id, "keyword": phrase,
            "language": language, "label_state": "present" if target == 1 else
            "absent" if target == 0 else "unknown",
            "existing_weak_target": target, "existing_weak_mask": mask,
            "existing_weak_weight": weight, "label_tier": label_tier,
            "observations": evidence, "positive_provider_count": positive_votes,
            "usable_provider_count": usable_count, "literal_agreement": agreement,
            "orthographic_positive_candidate": positive_votes == 3,
            "not_observed_is_not_negative": True,
            "reason": "preserved_verified_existing_weak_decision" if mask else
            "pronunciation_not_authenticated" if language == "en" else
            "preserved_existing_unknown_mask",
            "acoustic_event_times": None, "human_gold": False,
            "training_admitted": False, "independent_test_qualified": False,
            "trusted_negative": False}


def summarize(rows, source, batch_id, language):
    need(bool(rows), "EMPTY_EVIDENCE")
    need(type(batch_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", batch_id),
         "EXPLICIT_BATCH_ID_REQUIRED")
    seen, roles, audio, hashes = set(), {}, set(), {}
    for row in rows:
        key = (row["source_id"], row["keyword_id"])
        need(key not in seen, "DUPLICATE_SOURCE_KEYWORD")
        seen.add(key)
        family = row["source_family_id"]
        role = row["source_role"]
        need(roles.setdefault(family, role) == role, "SOURCE_FAMILY_SPLIT_LEAK")
        need(row["source_split"] in ("train", "development"), "NEW_SPLIT_NOT_ALLOWED")
        for field in ("wav_sha256", "pcm_sha256"):
            value = row.get(field)
            if value is not None:
                need(type(value) is str and SHA.fullmatch(value), "INVALID_AUDIO_SHA")
                need(hashes.setdefault((field, value), row["source_id"]) == row["source_id"],
                     "DUPLICATED_AUDIO_SOURCE")
        audio.add(row["source_id"])
    need(len(rows) == len(audio) * 2, "INCOMPLETE_KEYWORD_PROJECTION")
    counts = {}
    for key, _ in KEYWORDS[language]:
        subset = [r for r in rows if r["keyword_id"] == key]
        counts[key] = {"labels": dict(Counter(r["label_state"] for r in subset)),
                       "literal_agreement": dict(Counter(r["literal_agreement"] for r in subset)),
                       "preserved_weak_mask_true": sum(r["existing_weak_mask"] for r in subset),
                       "orthographic_positive_candidates": sum(r["orthographic_positive_candidate"] for r in subset)}
    return {"schema": POLICY, "batch_id": batch_id, "language": language,
            "source": source, "audio_rows": len(audio), "keyword_rows": len(rows),
            "source_families": len(roles), "new_model_calls": 0,
            "keyword_counts": counts, "rows": rows,
            "keyword_spec": {"scope": "literal-observation-only-not-pronunciation-approval",
                             "keywords": [{"id": key, "text": value, "accepted_aliases": []}
                                          for key, value in KEYWORDS[language]]},
            "training_admitted": False, "human_gold": False,
            "independent_evaluation": False, "accepted_negative_seconds": 0,
            "consumer_note": "Use original verified weak package and the existing pipeline admission chain for training; this view grants no new permission or labels."}


def project_english(report, source):
    need(report.get("training_admitted") is False and report.get("human_gold") is False,
         "ENGLISH_SOURCE_BOUNDARY_CHANGED")
    original = report.get("rows")
    need(type(original) is list and len(original) == 63, "ENGLISH63_ROWS_REQUIRED")
    rows = []
    for item in original:
        need(item.get("training_admitted") is False and item.get("human_gold") is False
             and item.get("trusted_negative") is False, "ENGLISH_ROW_BOUNDARY_CHANGED")
        role = item["role"]
        need(role in ("train-candidate", "development-only"), "ENGLISH_ROLE_CHANGED")
        intent = item["requested_intent"]
        need(intent in ("wake", "ordinary_negative", "near_miss"), "ENGLISH_INTENT_CHANGED")
        derived_usage = ("nearphone_quarantine" if intent == "near_miss" else
                         "development_no_tune" if role == "development-only" else
                         "weak_diagnostic_only")
        original_action = item.get("action")
        usage = (ENGLISH_USAGE.get(original_action, original_action)
                 if original_action is not None else derived_usage)
        need(usage in ENGLISH_USAGE.values(), "ENGLISH_RESEARCH_USAGE_CHANGED")
        need((intent != "near_miss" or usage == "nearphone_quarantine")
             and (role != "development-only" or usage != "weak_diagnostic_only"),
             "ENGLISH_RESEARCH_USAGE_PROMOTION")
        observations = item.get("asr", item.get("asr_observations"))
        need(type(observations) is dict and set(observations) == {"qwen", "whisper", "parakeet"},
             "ENGLISH_THREE_FAMILIES_REQUIRED")
        observed = []
        for provider, value in sorted(observations.items()):
            need(SHA.fullmatch(value.get("native_sha256", "")) is not None,
                 "ENGLISH_NATIVE_SHA_REQUIRED")
            observed.append({"provider": provider, "text": value["text"],
                             "native_sha256": value["native_sha256"],
                             "complete": True, "usable": True})
        identity = {"source_id": item["cell_id"], "source_family_id": item["source_family_id"],
                    "voice_id": item["voice_id"], "source_role": role,
                    "source_split": "train" if role == "train-candidate" else "development",
                    "wav_sha256": item["wav_sha256"], "pcm_sha256": item.get("pcm_sha256"),
                    "requested_intent_diagnostic_only": intent,
                    "original_research_usage": usage,
                    "original_research_action": original_action,
                    "research_usage_basis": "preserved_original_action" if original_action is not None
                    else "fixed_requested_intent_and_source_role"}
        for key, phrase in KEYWORDS["en"]:
            rows.append(keyword_row(identity, observed, phrase, key, "en"))
    result = summarize(rows, source, "en-qwen-seven-voice-63-20261112", "en")
    result["asr_observations_reused"] = 189
    return result


def english_from_archive(path):
    raw = read(path, 2 * 1024 * 1024)
    archive_sha = digest(raw)
    need(archive_sha in ENGLISH_SOURCES, "ENGLISH_ORIGINAL_ARCHIVE_REQUIRED")
    member, payload_sha, schema = ENGLISH_SOURCES[archive_sha]
    with zipfile.ZipFile(path) as archive:
        # Locate by payload digest as the joined original uses a nested output path.
        candidates = [x for x in archive.infolist() if not x.is_dir()
                      and x.filename.endswith(member) and 0 < x.file_size < MAX_BYTES]
        need(len(candidates) == 1, "ENGLISH_SOURCE_MEMBER_REQUIRED")
        raw_json = archive.read(candidates[0])
    need(digest(raw_json) == payload_sha, "ENGLISH_PAYLOAD_DRIFT")
    report = load(raw_json)
    need(report.get("schema") == schema, "ENGLISH_SOURCE_SCHEMA")
    return project_english(report, {"archive_sha256": archive_sha, "payload_sha256": payload_sha,
                                    "schema": schema, "verification": "pinned-existing-original-artifact"})


def project_chinese(report, source, batch_id):
    original = report.get("items", report.get("rows"))
    need(type(original) is list, "CHINESE_WEAK_ROWS_REQUIRED")
    rows = []
    for item in original:
        src, decision = item.get("source", item), item["decision"]
        need(decision.get("human_gold") is False, "CHINESE_NOT_HUMAN_GOLD")
        targets, masks, weights = (decision[n] for n in ("keyword_targets", "keyword_mask", "keyword_weight"))
        need(all(type(v) is list and len(v) == 2 for v in (targets, masks, weights)),
             "CHINESE_TWO_KEYWORDS_REQUIRED")
        observed = [{"provider": x["model"], "text": x.get("text"),
                     "complete": x.get("status") == "complete", "usable": x.get("usable") is True}
                    for x in item.get("asr", item.get("observations", []))]
        if "input" in item:
            wav_sha, pcm_sha = (item["input"][name]["sha256"] for name in ("original_wav", "original_pcm"))
        else:
            wav_sha, pcm_sha = item.get("wav_sha256"), item.get("pcm_sha256")
        identity = {"source_id": src["source_id"], "source_family_id": src["source_family_id"],
                    "source_role": src.get("role", src["split"]), "source_split": src["split"],
                    "wav_sha256": wav_sha, "pcm_sha256": pcm_sha,
                    "audio": item.get("audio"), "source_item_id": item.get("item_id", item.get("clip_id")),
                    "original_batch_id": item.get("batch_id"),
                    "source_plan_sha256": src.get("source_plan_sha256")}
        for i, (key, phrase) in enumerate(KEYWORDS["zh"]):
            rows.append(keyword_row(identity, observed, phrase, key, "zh", targets[i], masks[i], weights[i],
                                    decision["label_tier"]))
    return summarize(rows, source, batch_id, "zh")


def chinese_from_package(path, manifest_sha, batch_id):
    path = Path(path).absolute()
    need(SHA.fullmatch(manifest_sha) is not None and
         digest(read(path / "MANIFEST.json")) == manifest_sha, "CHINESE_MANIFEST_PIN_REQUIRED")
    raw = read(path / "report.json")
    report = load(raw)
    schema = report.get("schema")
    modules = {"kws-active-asr-research-weak-view-v1": "active_asr_weak_view",
               "kws-active-asr-research-weak-view-v2": "active_asr_weak_view",
               "kws-active-context-v3-research-weak-view-v1": "active_context_v3_weak_view",
               "kws-active-context-v3-metadata-research-weak-view-v1": "active_context_v3_metadata_weak_view",
               "kws-natural-asr-evidence-view-v1": "active_natural_asr_weak_view"}
    need(schema in modules, "UNSUPPORTED_EXISTING_CHINESE_PACKAGE")
    module = importlib.import_module(modules[schema])
    if modules[schema] == "active_asr_weak_view":
        # Exhaust both existing generators; they replay all source/audio/ASR rows.
        list(module.load_training_rows(path, consumer="rnn-four-token"))
        list(module.load_development_rows(path, consumer="rnn-four-token"))
    else:
        rebuilt = module.verify_view(path, expected_manifest_sha256=manifest_sha)
        need(rebuilt == report, "CHINESE_NATIVE_REPORT_CHANGED")
    need(read(path / "report.json") == raw, "CHINESE_REPORT_CHANGED_DURING_REPLAY")
    return project_chinese(report, {"manifest_sha256": manifest_sha, "payload_sha256": digest(raw),
                                    "schema": schema, "verification": "existing-native-source-audio-asr-verifier"}, batch_id)


def write_result(report, output):
    output = Path(output).absolute()
    need(not output.exists() and output.parent.is_dir()
         and not any(p.is_symlink() for p in output.parents), "OUTPUT_CREATE_ONLY")
    output.mkdir(mode=0o700)
    for name, value in (("report.json", report), ("rows.jsonl", report["rows"])):
        raw = ("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in value)
               if name.endswith("jsonl") else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        path = output / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(raw)
        path.chmod(0o600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    english = commands.add_parser("english63")
    english.add_argument("--evidence-zip", type=Path, required=True)
    chinese = commands.add_parser("zh-weak-view")
    chinese.add_argument("--package", type=Path, required=True)
    chinese.add_argument("--manifest-sha256", required=True)
    chinese.add_argument("--batch-id", required=True)
    for command in (english, chinese):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = (english_from_archive(args.evidence_zip) if args.command == "english63" else
                  chinese_from_package(args.package, args.manifest_sha256, args.batch_id))
        write_result(report, args.output_dir)
        print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "keyword_spec")},
                         ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print("V4_KEYWORD_EVIDENCE_STOP: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
