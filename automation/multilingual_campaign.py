"""Multilingual source-first KWS data campaign planner (stdlib, zero model calls).

Creates an immutable, non-admitted PRIVATE request matrix over legal-candidate
TTS backends, distinct pinned ASR families and authorized original sources.
It does not replace the canonical zh data producer or run arbitrary adapters.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(__file__).with_name("providers.v1.json")
POLICY = "multilingual-kws-offline-campaign-plan-v1"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
IDENT = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
LANGUAGES = ("zh-CN", "en-US")
ROLES = ("train-candidate", "development-only")
KINDS = ("wake", "near_miss", "ordinary_negative")
CHALLENGES = {
    "wake": ("complete_keyword",),
    "near_miss": ("homophone", "partial_prefix", "partial_suffix",
                  "cross_sentence", "repetition", "other_confusable"),
    "ordinary_negative": ("conversation", "media", "music", "environment"),
}
MAX_INPUT_BYTES = 1024 * 1024
MAX_GENERATION_CELLS = 256
MAX_VOICES = 16
MAX_UTTERANCES = 48
CANONICAL_CHINESE = ("你好小窝", "小窝小窝")
PLANNING_ONLY = {
    "human_gold": False, "source_rights_authenticated": False,
    "asr_source_authenticated": False, "training_admitted": False,
    "qualification_allowed": False, "shipping_approved": False,
    "model_calls_executed": 0, "next_campaign_authorized": False,
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for k, v in pairs:
        need(k not in result, "DUPLICATE_JSON_KEY")
        result[k] = v
    return result


def read(path: Path) -> dict:
    need(path.is_file() and not path.is_symlink() and
         stat.S_ISREG(path.stat().st_mode) and path.stat().st_nlink == 1 and
         0 < path.stat().st_size <= MAX_INPUT_BYTES, "INPUT_FILE_UNSAFE_OR_OVERSIZE")
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ValueError("NONFINITE_JSON")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("MALFORMED_JSON") from exc
    need(type(value) is dict, "INPUT_EXPECTED_OBJECT")
    return value


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def opaque(value: object, label: str) -> str:
    need(type(value) is str and IDENT.fullmatch(value) is not None, label)
    return value


def hex_sha(value: object, n: int, label: str) -> str:
    need(type(value) is str and (HEX40 if n == 40 else HEX64).fullmatch(value) is not None,
         label)
    return value


def normalize(text: str, locale: str) -> str:
    need(type(text) is str and 0 < len(text) <= 96 and
         not any(ord(c) < 32 for c in text), "INVALID_TEXT")
    clean = unicodedata.normalize("NFKC", text).strip()
    if locale == "zh-CN":
        return re.sub(r'[\s，。！？、,.!?;；：“”"‘’]+', "", clean)
    # For English, only orthographic differences are folded; different
    # phonemes, word order, missing words and accents remain separate.
    need(all(ord(c) < 128 for c in clean), "ENGLISH_REQUEST_MUST_BE_ASCII")
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", clean.casefold()))


def catalog() -> dict[str, dict]:
    data = read(CATALOG)
    need(data.get("schema_version") == 1 and
         data.get("policy") == "multilingual-kws-provider-research-v1",
         "PROVIDER_CATALOG_POLICY_CHANGED")
    rows = data.get("providers")
    need(type(rows) is list and 1 <= len(rows) <= 64, "PROVIDER_CATALOG_COUNT")
    result = {}
    for row in rows:
        need(type(row) is dict and set(row) == {
            "id", "role", "languages", "family", "upstream_model",
            "code_license", "checkpoint_license", "license_status",
            "mode", "adapter"}, "PROVIDER_CATALOG_FIELDS")
        key = opaque(row["id"], "PROVIDER_CATALOG_ID")
        need(key not in result and row["role"] in ("tts", "asr", "voice-conversion", "kws") and
             type(row["languages"]) is list and row["languages"] and
             all(locale in LANGUAGES for locale in row["languages"]) and
             len(set(row["languages"])) == len(row["languages"]) and
             row["license_status"] in
             ("candidate-permissive", "source-review-required", "blocked-noncommercial") and
             type(row["family"]) is str and row["family"] and
             type(row["upstream_model"]) is str and row["upstream_model"] and
             type(row["checkpoint_license"]) is str and row["checkpoint_license"],
             "PROVIDER_CATALOG_INVALID")
        result[key] = row
    return result


def choose(value: object, role: str, locale: str, providers: dict) -> list[dict]:
    need(type(value) is list and value and len(value) <= 8,
         "PINNED_PROVIDER_ROSTER_REQUIRED")
    result, ids, families, weights = [], set(), set(), set()
    for row in value:
        need(type(row) is dict and set(row) == {
            "provider_id", "revision_sha", "model_assets_sha256",
            "license_review_sha256"}, "PROVIDER_PIN_FIELDS")
        key = opaque(row["provider_id"], "UNKNOWN_PROVIDER")
        need(key in providers, "UNKNOWN_PROVIDER")
        source = providers[key]
        need(source["role"] == role and locale in source["languages"],
             "PROVIDER_LANGUAGE_OR_ROLE_UNSUPPORTED")
        need(source["license_status"] != "blocked-noncommercial",
             "NONCOMMERCIAL_MODEL_NOT_PERMITTED")
        need(key not in ids and source["family"] not in families,
             "MODEL_FAMILY_NOT_INDEPENDENT")
        revision = hex_sha(row["revision_sha"], 40, "MODEL_REVISION_NOT_PINNED")
        assets = hex_sha(row["model_assets_sha256"], 64, "MODEL_ASSETS_NOT_PINNED")
        license_sha = hex_sha(row["license_review_sha256"], 64, "MODEL_LICENSE_REVIEW_MISSING")
        need(assets not in weights, "MODEL_ASSETS_NOT_DISTINCT")
        ids.add(key)
        families.add(source["family"])
        weights.add(assets)
        result.append({"provider_id": key, "model_id": source["upstream_model"],
                       "family": source["family"], "mode": source["mode"],
                       "adapter": source["adapter"], "model_revision_sha": revision,
                       "model_assets_sha256": assets,
                       "license_review_sha256": license_sha,
                       "license_status": source["license_status"]})
    if role == "asr":
        need(len(result) == 3, "THREE_DISTINCT_ASR_FAMILIES_REQUIRED")
    return sorted(result, key=lambda x: x["provider_id"])


def plan(request: dict) -> dict:
    need(request.get("schema_version") == 1 and request.get("policy") == POLICY and
         set(request) == {"schema_version", "policy", "locale", "wake_phrases",
                          "utterances", "voices", "tts_providers", "asr_providers",
                          "repetitions", "max_generation_cells", "max_paid_usd",
                          "model_calls_authorized"} and
         request.get("locale") in LANGUAGES and
         request.get("max_paid_usd") == 0 and type(request.get("max_paid_usd")) is int and
         request.get("model_calls_authorized") is False,
         "CAMPAIGN_REQUEST_POLICY_OR_BUDGET")
    locale = request["locale"]
    wakes = request["wake_phrases"]
    need(type(wakes) is list and 1 <= len(wakes) <= 4 and
         all(type(x) is str for x in wakes), "WAKE_PHRASE_ROSTER")
    normalized = [normalize(word, locale) for word in wakes]
    need(all(normalized) and len(set(normalized)) == len(normalized),
         "DUPLICATE_OR_EMPTY_WAKE")
    if locale == "zh-CN":
        need(tuple(normalized) == CANONICAL_CHINESE,
             "CHINESE_PRODUCT_WAKE_LIST_MUST_REMAIN_FROZEN")
    else:
        need(all(len(w.split()) >= 2 for w in normalized),
             "ENGLISH_WAKE_MUST_BE_EXPLICIT_MULTIWORD")
    prompts = request["utterances"]
    need(type(prompts) is list and 1 <= len(prompts) <= MAX_UTTERANCES,
         "UTTERANCE_BUDGET")
    seen, entries, counts = set(), [], Counter()
    for row in prompts:
        need(type(row) is dict and set(row) == {
            "id", "text", "intent", "challenge", "keyword_index"},
             "UTTERANCE_FIELDS")
        uid = opaque(row["id"], "UTTERANCE_ID")
        intent = row["intent"]
        need(intent in KINDS and row["challenge"] in CHALLENGES[intent],
             "CHALLENGE_INTENT_CONFLICT")
        cleaned = normalize(row["text"], locale)
        need(cleaned and uid not in seen, "UTTERANCE_DUPLICATE_OR_EMPTY")
        seen.add(uid)
        if intent == "wake":
            index = row["keyword_index"]
            need(type(index) is int and 1 <= index <= len(normalized) and
                 cleaned == normalized[index - 1], "POSITIVE_MUST_MATCH_FULL_KEYWORD")
        else:
            need(row["keyword_index"] is None and cleaned not in normalized,
                 "NEGATIVE_MUST_NOT_MATCH_WAKE")
        entries.append({"id": uid, "text_requested": row["text"],
                        "normalized_text": cleaned, "intent": intent,
                        "challenge": row["challenge"],
                        "requested_keyword_index": row["keyword_index"]})
        counts[intent] += 1
    need(counts["wake"] >= len(wakes) and
         all(any(row["intent"] == "wake" and
                 row["requested_keyword_index"] == index + 1
                 for row in entries) for index in range(len(wakes))),
         "BOTH_WAKE_KEYWORDS_MUST_BE_REQUESTED")
    need(counts["near_miss"] and counts["ordinary_negative"],
         "NEGATIVE_COVERAGE_REQUIRED")

    voices = request["voices"]
    need(type(voices) is list and 1 <= len(voices) <= MAX_VOICES,
         "VOICE_ROSTER_BUDGET")
    rights, split_owners, voice_ids = [], {}, set()
    for row in voices:
        need(type(row) is dict and set(row) == {
            "id", "source_family_id", "role", "origin_kind",
            "reference_pcm_sha256", "rights_evidence_sha256"},
             "VOICE_FIELDS")
        ident = opaque(row["id"], "VOICE_ID")
        family = opaque(row["source_family_id"], "VOICE_FAMILY")
        role = row["role"]
        origin = row["origin_kind"]
        need(ident not in voice_ids and role in ROLES and
             origin in ("owned-synthetic", "consented-human", "licensed-public"),
             "VOICE_ORIGIN_OR_ROLE")
        voice_ids.add(ident)
        need(family not in split_owners or split_owners[family] == role,
             "VOICE_FAMILY_LEAKAGE")
        split_owners[family] = role
        pcm, proof = row["reference_pcm_sha256"], row["rights_evidence_sha256"]
        if origin != "owned-synthetic":
            hex_sha(pcm, 64, "REFERENCE_PCM_NOT_PINNED")
            hex_sha(proof, 64, "VOICE_CLONE_RIGHTS_NOT_PINNED")
        elif pcm is not None or proof is not None:
            hex_sha(pcm, 64, "SYNTHETIC_REFERENCE_PCM_NOT_PINNED")
            hex_sha(proof, 64, "SYNTHETIC_REFERENCE_RIGHTS_NOT_PINNED")
        rights.append(dict(row))
    provider_list = catalog()
    tts = choose(request["tts_providers"], "tts", locale, provider_list)
    asr = choose(request["asr_providers"], "asr", locale, provider_list)
    if any(t["mode"] == "authorized-reference-clone" for t in tts):
        need(all(v["reference_pcm_sha256"] is not None and
                 v["rights_evidence_sha256"] is not None for v in rights),
             "CLONING_REQUIRES_EACH_ORIGINAL_REFERENCE_AND_RIGHTS")
    repetitions = request["repetitions"]
    max_cells = request["max_generation_cells"]
    need(type(repetitions) is int and 1 <= repetitions <= 4 and
         type(max_cells) is int and 1 <= max_cells <= MAX_GENERATION_CELLS,
         "GENERATION_BUDGET_INVALID")
    needed = len(voices) * len(entries) * len(tts) * repetitions
    need(needed <= max_cells, "REQUESTED_CELLS_EXCEED_FIXED_BUDGET")
    rows = []
    for voice in sorted(rights, key=lambda x: x["id"]):
        for entry in sorted(entries, key=lambda x: x["id"]):
            for model in tts:
                for rep in range(1, repetitions + 1):
                    envelope = {"locale": locale, "voice_id": voice["id"],
                                "source_family_id": voice["source_family_id"],
                                "utterance_id": entry["id"],
                                "provider_id": model["provider_id"],
                                "provider_revision_sha": model["model_revision_sha"],
                                "repetition": rep}
                    rows.append({"cell_id": "cell-" + sha(envelope)[:24],
                                 **envelope, "role": voice["role"],
                                 "intent": entry["intent"], "challenge": entry["challenge"],
                                 "requested_keyword_index": entry["requested_keyword_index"],
                                 "text_requested": entry["text_requested"],
                                 "source_origin_kind": voice["origin_kind"],
                                 "synthetic": True, "actual_text_verified": False,
                                 "training_admitted": False,
                                 "qualification_allowed": False})
    rows.sort(key=lambda x: x["cell_id"])
    need(len({x["cell_id"] for x in rows}) == len(rows), "DUPLICATE_CAMPAIGN_CELL")
    return {
        "schema_version": 1, "policy": POLICY, "locale": locale,
        "wake_phrases_requested": list(wakes), "source_request_sha256": sha(request),
        "provider_catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "tts_pins": tts, "asr_pins": asr,
        "source_families_declared": len(split_owners),
        "voice_sources": rights, "requested_intents": dict(sorted(counts.items())),
        "asr_observations_planned": len(rows) * len(asr),
        "generation_cells_planned": len(rows), "max_paid_usd": 0,
        "planned_cells_sha256": sha(rows), "cells": rows,
        "note": "ASR models must receive audio ONLY, without intended text/hotword hints.",
        **PLANNING_ONLY,
    }


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--request", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args(argv)
    result = plan(read(args.request))
    dest = args.output.resolve()
    need(dest != ROOT and ROOT not in dest.parents and
         dest != args.request.resolve() and
         dest != CATALOG.resolve() and dest.parent.is_dir(),
         "PRIVATE_CREATE_ONLY_OUTPUT_REQUIRED")
    raw = (json.dumps(result, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(dest, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({"policy": POLICY, "locale": result["locale"],
                      "planned_cells": result["generation_cells_planned"],
                      "asr_observations_planned": result["asr_observations_planned"],
                      "model_calls_executed": 0, "training_admitted": False},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
