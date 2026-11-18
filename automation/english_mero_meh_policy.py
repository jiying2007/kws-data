"""Prospective saved-evidence admission for the selected MEH-roh corpus.

Only the Python standard library is used. The producer authenticates source and
capture bytes before this pure JSON decision layer; the trainer repeats that
authentication and recomputes this policy. No TTS, recognizer or KWS is invoked.
Unknown is masked, never background. Historical probes retain their old rules.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import unicodedata

SCHEMA = "english-mero-meh-admission-v1"
RULE_ID = "english-mero-meh-machine-policy-v1"
RECIPE_SHA256 = "e4f3381da7678eb45581d0fa5513128b2ca54e8127f70a34eb15af73e0ea55a7"
TEXT_PROVIDERS = ("qwen", "whisper", "parakeet")
PROVIDERS = TEXT_PROVIDERS + ("phoneme",)
HEX64 = re.compile(r"[0-9a-f]{64}")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _recipe():
    path = Path(__file__).with_name("english_mero_meh_recipe.py")
    need(path.is_file() and not path.is_symlink()
         and hashlib.sha256(path.read_bytes()).hexdigest() == RECIPE_SHA256,
         "PINNED_FRESH_RECIPE_BEFORE_IMPORT")
    spec = importlib.util.spec_from_file_location("mero_meh_frozen_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recipe = _recipe()
CAMPAIGN_ID = recipe.CAMPAIGN_ID


def _hey_aliases(words):
    return [prefix + " " + word for prefix in ("hey", "hay") for word in words]


# These are spelling families, not extra accepted acoustic pronunciations.
# In particular, miro/meero spellings do not authorize the MEER vowel family.
ALIASES = {
    "mero": _hey_aliases(("mero", "merrow", "marrow", "miro", "meero")),
    "vosen": _hey_aliases(("vosen", "vosin", "vohsen", "vozen", "vozin", "vohzen", "vozan")),
    "nero": _hey_aliases(("nero", "narrow", "nerrow")),
    "memo": _hey_aliases(("memo",)),
    "bosen": _hey_aliases(("bosen", "bozen", "bosun", "bozin", "bohsen", "bohzen", "bowson")),
}
PHONE_HYPOTHESES = {
    "mero-meh-us-v1": {"class": 2, "kind": "positive", "family": "mero",
                        "pattern": r"h?eɪmɛr(?:oʊ|ou|o)"},
    "vosen-us-s-v1": {"class": 1, "kind": "positive", "family": "vosen",
                       "pattern": r"h?eɪv(?:oʊ|ou|o)s[əɐʌ]n"},
    "nero-eh-us-v1": {"class": 0, "kind": "contrast", "family": "nero",
                      "pattern": r"h?eɪnɛr(?:oʊ|ou|o)"},
    "memo-us-v1": {"class": 0, "kind": "contrast", "family": "memo",
                   "pattern": r"h?eɪmɛm(?:oʊ|ou|o)"},
    "bosen-us-s-v1": {"class": 0, "kind": "contrast", "family": "bosen",
                      "pattern": r"h?eɪb(?:oʊ|ou|o)s[əɐʌ]n"},
    "bosen-us-z-v1": {"class": 0, "kind": "contrast", "family": "bosen",
                      "pattern": r"h?eɪb(?:oʊ|ou|o)z[əɐʌ]n"},
}
CONTRADICTORY_PHRASES = _hey_aliases((
    "frozen", "chosen", "rosen", "wosen", "hero", "zero",
    "meryl", "merrill", "merle", "meadow", "mirror"))
ORDINARY = tuple(x["requested_text"] for x in recipe.cells() if x["kind"] == "ordinary")
RULES = {
    "rule_id": RULE_ID, "campaign_id": CAMPAIGN_ID, "recipe_sha256": RECIPE_SHA256,
    "profile": copy.deepcopy(recipe.PROFILE), "aliases": ALIASES,
    "phone_hypotheses": PHONE_HYPOTHESES, "contradictory_phrases": CONTRADICTORY_PHRASES,
    "ordinary": list(ORDINARY), "silver_votes": 3, "weak_votes": 2,
    "tier_weights": {"silver": 1.0, "weak": 0.5},
    "minimum_balanced_groups": copy.deepcopy(recipe.MINIMUM_BALANCED_GROUPS),
    "pcm": {"sample_rate": 16000, "channels": 1, "sample_width": 2,
            "minimum_samples": 6400, "maximum_samples": 160000,
            "positive_maximum_samples": 28672, "minimum_rms": 0.0001,
            "maximum_rail_fraction": 0.01},
    "phones": {"scope": "ENTIRE_UNCROPPED_SHORT_PCM", "strip_stress_length_ties": True,
               "rhotics": {"ɹ": "r", "ɾ": "r"},
               "missing_h_requires_all_three_texts": True,
               "missing_h_scope": "Predeclared CTC omission tolerance; not a separately generated wake pronunciation.",
               "vosen_unstressed_vowel_observer_family": ["ə", "ɐ", "ʌ"],
               "vosen_scope": "The existing unstressed-vowel observer family is retained; medial z is excluded.",
               "e_notation_is_not_xlsr_token_substitution": True,
               "stress_independently_verified": False},
    "collision_policy": "No strict marrow/merrow negative; spelling alone cannot distinguish a homophone.",
    "request_mismatch": "Observed candidate precedes requested class/kind check; mismatch is unknown.",
    "ordinary_phone_rule": "Complete phone output, no target pattern and no hey/hay phrase evidence.",
    "unknown_is_background": False, "parent_context_weak_policy": False,
    "historical_result_changes": False, "automatic_expansion": False,
}
RULES_SHA256 = digest(RULES)


def normalized_text(text):
    need(isinstance(text, str), "TRANSCRIPT_MUST_BE_STRING")
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c
                            for c in value).split())


def normalized_phones(tokens):
    need(type(tokens) is list and tokens and all(type(t) is str and t for t in tokens),
         "NONEMPTY_PHONE_TOKEN_LIST")
    value = unicodedata.normalize("NFKC", "".join(tokens))
    value = re.sub(r"[\sˈˌːˑ͜͡]", "", value)
    return value.replace("ɹ", "r").replace("ɾ", "r")


def whole_support_ends(samples):
    """Exact existing 16k integer support; no audio or model is evaluated."""
    need(type(samples) is int and samples > 0, "INTEGER_PCM_SAMPLES")
    start, observed = 38400, 38400 + samples + 16000
    return [end for end in range(16000, observed + 1, 1280)
            if end >= 35200 and end - 31840 <= start and end - 128 >= start + samples]


def _contains(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def blind_candidate(texts, phone_tokens):
    """Only observations enter this function: no row, intent, TTS or KWS label."""
    need(type(texts) is dict and set(texts) == set(TEXT_PROVIDERS), "EXACT_THREE_ASR_SLOTS")
    normalized = {p: normalized_text(v) if isinstance(v, str) else None for p, v in texts.items()}
    result = {"tier": "unknown", "observed_class": None, "observed_kind": None,
              "matched_pronunciation_hypothesis": None, "reason": "MISSING_OR_INCOMPLETE_OBSERVATION",
              "normalized_texts": normalized, "normalized_phones": None,
              "alias_votes": {}, "contradictory_providers": [], "phone_matches": []}
    if any(not v for v in normalized.values()) or not phone_tokens:
        return result
    phone = normalized_phones(phone_tokens)
    result["normalized_phones"] = phone
    matches = [name for name, rule in PHONE_HYPOTHESES.items() if re.fullmatch(rule["pattern"], phone)]
    result["phone_matches"] = matches
    result["alias_votes"] = {family: sum(t in aliases for t in normalized.values())
                             for family, aliases in ALIASES.items()}
    if len(matches) == 1:
        hypothesis = matches[0]
        rule = PHONE_HYPOTHESES[hypothesis]
        family = rule["family"]
        conflicts = [a for other, aliases in ALIASES.items() if other != family for a in aliases]
        conflicts += CONTRADICTORY_PHRASES
        vetoes = [p for p, text in normalized.items() if any(_contains(text, a) for a in conflicts)]
        votes = result["alias_votes"][family]
        result["contradictory_providers"] = vetoes
        if vetoes:
            result["reason"] = "ASR_PHONE_COMPETING_WORD_DISAGREEMENT"
        elif votes < 2:
            result["reason"] = "INSUFFICIENT_SPELLING_FAMILY_SUPPORT"
        elif not phone.startswith("h") and votes != 3:
            result["reason"] = "MISSING_H_REQUIRES_THREE_TEXT_SUPPORTERS"
        else:
            result.update(tier="silver" if votes == 3 else "weak",
                          observed_class=rule["class"], observed_kind=rule["kind"],
                          matched_pronunciation_hypothesis=hypothesis,
                          reason="COMPLETE_PHONE_AND_BLIND_TEXT_SUPPORT")
        return result
    if len(matches) > 1:
        result["reason"] = "NONUNIQUE_PHONE_HYPOTHESIS"
        return result
    # Ordinary speech must be observed as an entire registered sentence, not
    # inferred merely from a failed wake-word pattern or generation intent.
    ordinary_votes = {sentence: sum(t == normalized_text(sentence) for t in normalized.values())
                      for sentence in ORDINARY}
    supported = [s for s, count in ordinary_votes.items() if count >= 2]
    wake_in_text = any(re.search(r"\b(?:hey|hay)\b", t) for t in normalized.values())
    wake_in_phones = any(re.search(rule["pattern"], phone) for rule in PHONE_HYPOTHESES.values()
                         if rule["class"] in (1, 2))
    if len(supported) == 1 and not wake_in_text and not wake_in_phones:
        result.update(tier="silver" if ordinary_votes[supported[0]] == 3 else "weak",
                      observed_class=0, observed_kind="ordinary",
                      matched_pronunciation_hypothesis="ordinary-negative-v1",
                      reason="COMPLETE_ORDINARY_TEXT_WITH_NO_WAKE_EVIDENCE",
                      observed_ordinary_sentence=supported[0])
    else:
        result["reason"] = "NO_PREDECLARED_COMPLETE_ACOUSTIC_CONTENT_SUPPORT"
    return result


def _safe_capture(path):
    if type(path) is not str or not path or "\\" in path:
        return False
    p = PurePosixPath(path)
    return not p.is_absolute() and p.as_posix() == path and all(part not in ("", ".", "..") for part in p.parts)


def _observations(row, packet):
    need(type(packet) is dict and packet.get("id") == row["id"], "EVIDENCE_ROW_IDENTITY")
    obs = packet.get("observations")
    need(type(obs) is dict and set(obs) == set(PROVIDERS), "FOUR_EXPLICIT_OBSERVATION_SLOTS")
    texts, phones, refs, missing = {}, None, [], []
    for provider in PROVIDERS:
        observation = obs[provider]
        need(type(observation) is dict and observation.get("status") in
             ("success", "missing", "failed", "not_started", "unknown"), "EXPLICIT_OBSERVATION_STATUS")
        value = None
        if observation["status"] == "success":
            need(observation.get("input_pcm_sha256") == row.get("pcm_sha256")
                 and observation.get("input_samples") == row.get("pcm_samples"), "SAME_COMPLETE_PCM_OBSERVATION")
            need(_safe_capture(observation.get("capture_path"))
                 and HEX64.fullmatch(observation.get("capture_sha256", "")), "RETAINED_CAPTURE_IDENTITY")
            refs.append({"provider": provider, "capture_path": observation["capture_path"],
                         "capture_sha256": observation["capture_sha256"]})
            complete = (observation.get("target_conditioned") is False
                        and observation.get("input_coverage_complete") is True
                        and observation.get("decoder_output_complete") is True)
            if complete:
                field = "phone_tokens" if provider == "phoneme" else "raw_transcript"
                value = observation.get(field)
                if provider == "phoneme":
                    need(type(value) is list and all(type(t) is str and t for t in value), "RAW_PHONE_CAPTURE_TYPE")
                else:
                    need(type(value) is str and len(value) <= 16384, "RAW_TEXT_CAPTURE_TYPE")
        if value is None or value == "" or value == []:
            missing.append(provider)
        if provider == "phoneme":
            phones = value
        else:
            texts[provider] = value
    return texts, phones, refs, missing


def _quality(row):
    if row.get("generation_status") != "success":
        return False, "GENERATION_UNAVAILABLE", []
    for field in ("pcm_sha256", "wav_sha256", "parent_pcm_sha256"):
        need(type(row.get(field)) is str and HEX64.fullmatch(row[field]), "VALID_SOURCE_DIGEST:" + field)
    need(row.get("sample_rate") == 16000 and row.get("channels") == 1 and row.get("sample_width") == 2,
         "SAME_INT16_MONO_16K_CORPUS_FORMAT")
    need(type(row.get("pcm_samples")) is int and row["pcm_samples"] > 0, "EXACT_PCM_SAMPLE_COUNT")
    rms, rail = row.get("pcm_rms"), row.get("pcm_rail_fraction")
    need(type(rms) in (int, float) and type(rail) in (int, float)
         and math.isfinite(rms) and math.isfinite(rail) and rms >= 0 and 0 <= rail <= 1,
         "FINITE_RECOMPUTED_PCM_MEASUREMENTS")
    need(type(row.get("pcm_quality_passed")) is bool, "EXPLICIT_PCM_QUALITY_MEASUREMENT")
    limits, n = RULES["pcm"], row["pcm_samples"]
    quality = (limits["minimum_samples"] <= n <= limits["maximum_samples"]
               and rms >= limits["minimum_rms"] and rail <= limits["maximum_rail_fraction"])
    need(row["pcm_quality_passed"] == quality, "PCM_QUALITY_FLAG_MUST_MATCH_MEASUREMENTS")
    return quality, "PCM_DURATION_ENERGY_OR_RAIL" if not quality else "PCM_QUALITY_SUPPORTED", whole_support_ends(n)


def build_admission(corpus, machine_evidence, identity_contract):
    """Recomputable 144-row decision, including every failed or unknown request."""
    need(type(corpus) is dict and corpus.get("campaign_id") == CAMPAIGN_ID
         and corpus.get("source_design_id") == CAMPAIGN_ID, "EXACT_FRESH_CAMPAIGN")
    need(corpus.get("diagnostic_only") is False and corpus.get("training_allowed") is True
         and corpus.get("permanently_excluded_from_training") is not True,
         "FRESH_TRAINING_PURPOSE_NOT_HISTORICAL_PROBE")
    need(canonical(identity_contract) == canonical(recipe.identity_contract()), "FROZEN_12_PRESET_IDENTITY_CONTRACT")
    contract_sha = digest(identity_contract)
    need(corpus.get("source_identity_contract_sha256") == contract_sha, "CORPUS_IDENTITY_CONTRACT_BINDING")
    need(type(machine_evidence) is dict and machine_evidence.get("schema") == "english-mero-meh-machine-evidence-v1"
         and machine_evidence.get("campaign_id") == CAMPAIGN_ID, "FROZEN_MACHINE_EVIDENCE_SCHEMA")
    rows, packets, expected = corpus.get("rows"), machine_evidence.get("rows"), recipe.rows()
    need(type(rows) is list and type(packets) is list and len(rows) == len(packets) == 144,
         "ALL_144_PLANNED_ROWS_REQUIRED")
    expected_ids = [r["id"] for r in expected]
    need([r.get("id") for r in rows] == [r.get("id") for r in packets] == expected_ids,
         "FROZEN_COMPLETE_ORDERED_ROSTER")
    result, pcm_seen = [], set()
    for row, packet, planned in zip(rows, packets, expected):
        need(canonical({key: row.get(key) for key in planned}) == canonical(planned),
             "FROZEN_REQUEST_ROLE_IDENTITY_BINDING")
        quality, quality_reason, ends = _quality(row)
        if row.get("generation_status") == "success":
            need(row["pcm_sha256"] not in pcm_seen, "DUPLICATE_PCM_NOT_INDEPENDENT_NEW_SAMPLE")
            pcm_seen.add(row["pcm_sha256"])
        texts, phones, refs, missing = _observations(row, packet)
        candidate = blind_candidate(texts, phones)
        # Intent and geometry are consulted only after the blind decision.
        reason, tier, label = candidate["reason"], candidate["tier"], candidate["observed_class"]
        if not quality:
            reason, tier, label = quality_reason, "unknown", None
        elif candidate["tier"] != "unknown":
            if candidate["observed_class"] != row["requested_class"] or candidate["observed_kind"] != row["kind"]:
                reason, tier, label = "OBSERVED_CONTENT_DISAGREES_WITH_REQUESTED_CLASS_OR_KIND", "unknown", None
            elif label in (1, 2) and (row["pcm_samples"] > RULES["pcm"]["positive_maximum_samples"] or len(ends) < 3):
                reason, tier, label = "FEWER_THAN_THREE_COMPLETE_POSITIVE_SUPPORT_WINDOWS", "unknown", None
        result.append({**{key: copy.deepcopy(row[key]) for key in planned},
            "pcm_sha256": row.get("pcm_sha256"), "wav_sha256": row.get("wav_sha256"),
            "parent_pcm_sha256": row.get("parent_pcm_sha256"), "pcm_samples": row.get("pcm_samples"),
            "tier": tier, "research_class": label, "reason": reason,
            "evidence_tier": candidate["tier"], "observed_class": candidate["observed_class"],
            "observed_kind": candidate["observed_kind"], "candidate": candidate,
            "matched_pronunciation_hypothesis": candidate["matched_pronunciation_hypothesis"],
            "evidence_refs": refs, "missing_observers": missing,
            "phonetic_evidence_scope": "ENTIRE_UNCROPPED_SHORT_PCM",
            "child_independent_phone_silver": tier == "silver",
            "whole_support_windows": len(ends), "whole_support_ends": ends,
            "supervision_allowed": False, "training_allowed": False,
            "training_class": None, "training_weight": 0.0,
            "human_gold": False, "final_native_pronunciation_verified": False,
            "independent_real_evaluation_allowed": False})
    supported = defaultdict(set)
    for row in result:
        if row["tier"] in ("silver", "weak"):
            supported[row["source_group_id"]].add(row["research_class"])
    balanced = {group for group, labels in supported.items() if labels == {0, 1, 2}}
    roles = {row["source_group_id"]: row["role"] for row in rows}
    counts = Counter(roles[group] for group in balanced)
    for row in result:
        if row["tier"] in ("silver", "weak") and row["source_group_id"] in balanced:
            row["supervision_allowed"] = True
            if row["role"] == "train":
                row.update(training_allowed=True, training_class=row["research_class"],
                           training_weight=RULES["tier_weights"][row["tier"]])
        elif row["tier"] != "unknown":
            row.update(tier="unknown", research_class=None, reason="SOURCE_GROUP_MISSING_AN_ADMITTED_CLASS",
                       child_independent_phone_silver=False)
        row["disposition"] = "UNKNOWN" if row["tier"] == "unknown" else row["tier"].upper() + "_SYNTHETIC_RESEARCH"
    passed = all(counts[role] >= count for role, count in recipe.MINIMUM_BALANCED_GROUPS.items())
    return {"schema": SCHEMA, "campaign_id": CAMPAIGN_ID, "rule_id": RULE_ID,
        "status": "PASS_FRESH_CORPUS_ADMISSION" if passed else "STOP_FRESH_CORPUS_COVERAGE",
        "rules_sha256": RULES_SHA256, "recipe_sha256": RECIPE_SHA256,
        "profile_sha256": recipe.PROFILE["sha256"], "source_identity_contract_sha256": contract_sha,
        "corpus_sha256": digest(corpus), "machine_evidence_sha256": digest(machine_evidence),
        "rows": result, "denominator": 144, "tier_counts": dict(Counter(r["tier"] for r in result)),
        "balanced_groups_by_role": {role: counts[role] for role in recipe.MINIMUM_BALANCED_GROUPS},
        "minimum_balanced_groups": copy.deepcopy(recipe.MINIMUM_BALANCED_GROUPS),
        "training_snapshot_allowed": passed, "training_allowed": passed,
        "diagnostic_only": False, "permanently_excluded_from_training": False,
        "parent_context_weak_policy": False, "unknown_is_background": False,
        "historical_probe_reclassified": False, "automatic_expansion_allowed": False,
        "human_gold": False, "independent_real_quality_claims_allowed": False,
        "physical_ssc305_claims_allowed": False, "new_model_calls": 0}
