"""V7 candidate-relative blind-observation policy mirrored from kws-pipeline.

Upstream source is pinned by repository commit and Git blob SHA in the pilot
plan. This module intentionally contains only the frozen normalization,
observer identity and decision functions required by the 12-clip pilot.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata

SCHEMA = "english-kws-v7-voxcpm-pilot-policy-v1"
CAPTURE_SCHEMA = "english-kws-dual-track-v7-voice-clone-observer-capture-v1"
UPSTREAM = {
    "repository": "jiying2007/kws-pipeline",
    "commit": "ec414dc89c52fb5756e9560e79517bcd41784248",
    "path": "training/english_kws_dual_track_v7_silver.py",
    "blob_sha": "d7c0d68b7ad7ed7e79f67b0e15037cd42ab74ffb",
}
MODEL_PINS = {
    "qwen": ("qwen3-asr", "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea",
             "6ef4ab0c3a329ac618980f42a00ebff601c08a768b8f341256b85ffaffb3bca1"),
    "whisper": ("whisper", "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
                "439b51d2e7e5f7ce4ade748a87c3d3e6d5b6b66d2acaec893bfc3bf673af04a7"),
    "parakeet": ("parakeet-tdt", "e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e",
                 "c83bcf97f4c3f718264df1ecac8f013a276bdf7024f3ed39b255f85ad4bb9b6d"),
    "phoneme": ("wav2vec2-xlsr", "04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551",
                "2b37c89e02afc6b44b8109a9ae38303eca75411564f53b3d61d86985de5af441"),
}
ALIASES = {
    "VOSEN": ("hey vosen", "hey vohsen"),
    "MERO": ("hey mero", "hey merrow"),
}
EXCLUDED_TRANSCRIPTS = {
    "hey meero", "hay meero", "hey miro", "hay miro",
    "hey vozen", "hay vozen", "hey vohzen", "hay vohzen",
    "hey vozan", "hay vozan", "hey vozin", "hay vozin",
}
PHONE_PATTERNS = {
    "VOSEN": re.compile(r"h?eɪv(?:oʊ|ou|o)s[əɐʌ]n$"),
    "MERO": re.compile(r"h?eɪmɛr(?:oʊ|ou|o)$"),
}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def normalized_text(text):
    need(isinstance(text, str), "TRANSCRIPT_MUST_BE_TEXT")
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(char).startswith("P") else char
                            for char in text).split())


def normalized_phones(tokens):
    need(isinstance(tokens, list) and all(isinstance(token, str) and token for token in tokens),
         "PHONE_TOKENS_REQUIRED")
    value = unicodedata.normalize("NFKC", "".join(tokens))
    value = "".join(value.split())
    for marker in ("ˈ", "ˌ", "ː", "ˑ", "͡", "͜"):
        value = value.replace(marker, "")
    return value.replace("ɹ", "r").replace("ɾ", "r")


def phone_candidate(tokens):
    value = normalized_phones(tokens)
    matches = [name for name, pattern in PHONE_PATTERNS.items() if pattern.fullmatch(value)]
    return matches[0] if len(matches) == 1 else ("AMBIGUOUS" if matches else None)


def observer_identity(provider):
    need(provider in MODEL_PINS, "FIXED_FOUR_OBSERVERS")
    family, model_sha, decode_sha = MODEL_PINS[provider]
    return {
        "recognizer_id": provider,
        "model_family": family,
        "model_sha256": model_sha,
        "decode_config_sha256": decode_sha,
    }


def safe_relative(name):
    need(type(name) is str and name and chr(92) not in name, "SAFE_CAPTURE_PATH")
    path = PurePosixPath(name)
    need(not path.is_absolute() and path.as_posix() == name
         and all(part not in ("", ".", "..", ".git") for part in path.parts),
         "SAFE_CAPTURE_PATH")
    return name


def read_member(root, relative, expected_sha):
    path = Path(root)
    for part in PurePosixPath(safe_relative(relative)).parts:
        path = path / part
        need(not path.is_symlink(), "NO_CAPTURE_SYMLINK")
    need(path.is_file() and not path.is_symlink()
         and 0 < path.stat().st_size <= 1024 * 1024, "BOUNDED_CAPTURE_FILE")
    raw = path.read_bytes()
    need(sha(raw) == expected_sha, "CAPTURE_SHA256")
    return raw


def verify_observation(observation, provider, row, root, *, phones=False):
    if not isinstance(observation, dict):
        return None
    identity = observer_identity(provider)
    need(all(observation.get(key) == value for key, value in identity.items()),
         "OBSERVER_IDENTITY:" + provider)
    need(observation.get("id") == row["id"]
         and observation.get("schema") == CAPTURE_SCHEMA
         and observation.get("status") == "success"
         and observation.get("source_pcm_sha256") == row["pcm_sha256"]
         and observation.get("source_wav_sha256") == row["wav_sha256"]
         and observation.get("source_samples") == row["pcm_samples"]
         and observation.get("target_conditioned") is False
         and observation.get("input_coverage_complete") is True
         and observation.get("decoder_output_complete") is True,
         "OBSERVER_SOURCE_AND_BLIND_BINDING:" + provider)
    raw = read_member(root, observation["capture_path"], observation["capture_sha256"])
    capture = json.loads(raw, object_pairs_hook=lambda pairs: _unique_pairs(pairs))
    need(capture == {key: value for key, value in observation.items()
                     if key not in ("capture_path", "capture_sha256")},
         "CAPTURE_BYTES_MATCH_INDEX:" + provider)
    if phones:
        tokens = capture.get("phone_tokens")
        need(isinstance(tokens, list) and 1 <= len(tokens) <= 300
             and all(isinstance(token, str) and 0 < len(token) <= 40 for token in tokens),
             "PHONE_CAPTURE_CONTENT")
        return tokens
    text = capture.get("raw_text")
    need(isinstance(text, str) and 0 < len(text.strip()) <= 512
         and not any(ord(char) < 32 for char in text.strip()),
         "ASR_CAPTURE_CONTENT:" + provider)
    return normalized_text(text)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_CAPTURE_JSON_KEY")
        result[key] = value
    return result


def decide_candidate(row, candidate, texts, phones):
    """Same V7 candidate-relative rules as the pinned pipeline source."""
    need(candidate in ("VOSEN", "MERO"), "KNOWN_CANDIDATE")
    if texts is None or len(texts) != 3 or any(not isinstance(text, str) or not text for text in texts):
        return "unknown", None, "MISSING_OR_INCOMPLETE_ASR", None
    if phones is None or not phones:
        return "unknown", None, "MISSING_OR_INCOMPLETE_PHONE", None
    normalized = [normalized_text(text) for text in texts]
    if any(text in EXCLUDED_TRANSCRIPTS for text in normalized):
        return "unknown", None, "SUPERSEDED_PRONUNCIATION_TRANSCRIPT", None
    votes = {"VOSEN": 0, "MERO": 0}
    for text in normalized:
        for name, aliases in ALIASES.items():
            if text in aliases:
                votes[name] += 1
    other = "MERO" if candidate == "VOSEN" else "VOSEN"
    phone = phone_candidate(phones)
    if phone == "AMBIGUOUS":
        return "unknown", None, "AMBIGUOUS_PHONE_CLASS", None
    target_votes, other_votes = votes[candidate], votes[other]
    expected_class = row["candidate_labels"][candidate]
    if expected_class == 1:
        if target_votes < 2 or other_votes != 0 or phone != candidate:
            return "unknown", None, "NO_JOINT_CANONICAL_TARGET_SUPPORT", phone
        return ("silver" if target_votes == 3 else "weak"), 1, "JOINT_ASR_AND_PHONE_TARGET_SUPPORT", candidate
    if row["kind"] == "target":
        if other_votes < 2 or target_votes != 0 or phone != other:
            return "unknown", None, "NO_JOINT_CROSS_TRIGGER_NEGATIVE_SUPPORT", phone
        return ("silver" if other_votes == 3 else "weak"), 0, "OTHER_BRAND_CROSS_TRIGGER_NEGATIVE", other
    if target_votes or other_votes or phone is not None:
        return "unknown", None, "WAKE_PHRASE_OR_PHONE_PRESENT_IN_NEGATIVE", phone
    expected = normalized_text(row["requested_text"])
    exact_votes = sum(text == expected for text in normalized)
    if exact_votes < 2:
        return "unknown", None, "NEGATIVE_NOT_EXPLICITLY_RECOGNIZED", None
    if row["kind"] not in ("contrast-negative", "ordinary-negative"):
        return "unknown", None, "UNREGISTERED_NEGATIVE_KIND", None
    return ("silver" if exact_votes == 3 else "weak"), 0, "EXPLICIT_NEGATIVE_TRANSCRIPT_AND_NO_WAKE_PHONE", None


def source_check():
    return {
        "status": "PASS",
        "upstream": UPSTREAM,
        "providers": list(MODEL_PINS),
        "new_model_calls": 0,
        "training_allowed": False,
        "promotion_allowed": False,
    }
