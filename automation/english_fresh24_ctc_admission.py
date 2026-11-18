"""Prospectively frozen two-phone CTC canonical-vs-contrast discriminator.

This file is committed BEFORE ANY new fresh24 source WAV exists. Historic
Melo25 and clone60 are EXPOSED development evidence only, never admitted or
re-labelled. No target words reach inference. No phone posterior alone
creates a human-gold label, real FAR/FRR or source-group independence.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import unicodedata

SCHEMA = "english-fresh24-ctc-contrast-admission-v1"
PLAN_PATH = Path(__file__).with_name("english_fresh24_protocol.json")
FROZEN_SOURCE_ROLES = {"train": 12, "development": 4, "calibration": 4,
                       "locked-test": 4}
MIN_QUALIFIED = {"train": 6, "development": 2, "calibration": 2,
                 "locked-test": 2}
TEXT_MODELS = ("qwen", "whisper", "parakeet")
PHONE_MODELS = ("xlsr", "lv60")
OLD_PRESETS = frozenset(("af_bella", "af_sarah", "am_michael", "am_puck",
    "af_nicole", "am_fenrir", "af_aoede", "am_echo",
    "af_kore", "af_alloy", "af_nova", "am_onyx"))
VOWELS = ("oʊ", "o", "ɔ")
SCHWAS = ("ə", "ɐ", "ʌ")
# Frozen ONCE, before new source generation. Every alternate occurs equally
# in positive and contrast lexica to prevent tail-variant cherry picking.
# A missing required token in a recognizer is UNKNOWN, never automatic PASS.
MERO_CORE = (("m", "ɛ", "r"),)
MERO_CONTRAST_CORE = (("m", "iː", "r"), ("m", "i", "r"), ("m", "ɪ", "r"))
VOSEN_CORE = (("v",),)
VOSEN_CONTRAST_CORE = (("b",), ("w",))
ASR_ALIASES = {
    2: frozenset(("hey mero", "hey marrow", "hey merrow",
                  "hay mero", "hay marrow", "hay merrow")),
    1: frozenset(("hey vosen", "hey vohsen", "hey vosin",
                  "hay vosen", "hay vohsen", "hay vosin")),
}
OLD_ALIASES = frozenset(("hey meero", "hay meero", "hey miro",
    "hay miro", "hey vozen", "hay vozen", "hey vozin",
    "hay vozin", "hey vohzen", "hay vohzen"))
ORDINARY = frozenset(("what time is it",))
MAX_POSITIVE_SAMPLES = 28672
MIN_POSITIVE_SAMPLES = 6400
NEGATIVE_NOT_FROM_ASR_ABSENCE = True


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(text):
    need(type(text) is str and 0 < len(text) <= 4096, "COMPLETE_RAW_ASR_REQUIRED")
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c
        for c in value).split())


def paths(kind):
    """Fixed phone paths including optional weak initial H and GOAT variants."""
    need(kind in (1, 2), "EXACT_TWO_CANONICAL_BRANDS")
    prefixes = (("h", "eɪ"), ("eɪ",))
    if kind == 2:
        return (
            tuple(prefix + core + (tail,)
                for prefix in prefixes for core in MERO_CORE for tail in VOWELS),
            tuple(prefix + core + (tail,)
                for prefix in prefixes for core in MERO_CONTRAST_CORE
                for tail in VOWELS))
    return (
        tuple(prefix + ("v", goat, "s", schwa, "n")
            for prefix in prefixes for goat in VOWELS for schwa in SCHWAS),
        tuple(prefix + lead + (goat, medial, schwa, "n")
            for prefix in prefixes for goat in VOWELS for schwa in SCHWAS
            for lead, medial in ((("v",), "z"), (("b",), "s"), (("w",), "s"))))


def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if b > a:
        a, b = b, a
    return a + math.log1p(math.exp(b-a))


def verified_phone_input(raw, vocab, *, pad_token_id=0):
    """Read a complete original CTC float32 logits tensor, no frame cropping."""
    need(type(raw) is bytes and len(raw) > 0 and len(raw) <= 40 * 1024 * 1024,
         "BOUNDED_ORIGINAL_NATIVE_LOGITS")
    need(type(vocab) is dict and len(vocab) == 392 and
         len(set(vocab.values())) == 392 and
         all(type(v) is int and 0 <= v < 392 for v in vocab.values()),
         "EXACT_NATIVE_PHONE_VOCAB")
    need(type(pad_token_id) is int and pad_token_id == 0,
         "EXACT_CTC_BLANK_ID")
    size = len(vocab)
    need(len(raw) % (size * 4) == 0, "COMPLETE_FLOAT32_FRAME_SHAPE")
    frames = len(raw) // (size * 4)
    need(1 <= frames <= 2000, "FULL_INPUT_FRAME_BOUND")
    labels = {pad_token_id}
    for kind in (1, 2):
        for language in paths(kind):
            for tokens in language:
                for token in tokens:
                    if token in vocab:
                        labels.add(vocab[token])
    result = []
    for frame in struct.iter_unpack("<" + "f" * size, raw):
        need(all(math.isfinite(x) for x in frame), "NONFINITE_NATIVE_CTC_LOGIT")
        vmax = max(frame)
        normlog = vmax + math.log(math.fsum(math.exp(x-vmax) for x in frame))
        result.append({i:frame[i]-normlog for i in labels})
    return result


def ctc_sequence_logp(frames, vocab, phones, *, pad_token_id=0):
    """Log-semiring forward CTC probability for EXACT whole-phrase sequence."""
    if any(token not in vocab for token in phones):
        return None
    ids = [vocab[token] for token in phones]
    states = [pad_token_id]
    for ident in ids:
        states.extend((ident, pad_token_id))
    dp = [-math.inf] * len(states)
    dp[0] = 0.0
    for frame in frames:
        next_dp = [-math.inf] * len(states)
        for idx, symbol in enumerate(states):
            score = dp[idx]
            if idx:
                score = _logadd(score, dp[idx-1])
            if idx > 1 and symbol != pad_token_id and symbol != states[idx-2]:
                score = _logadd(score, dp[idx-2])
            next_dp[idx] = score + frame[symbol] if score != -math.inf else -math.inf
        dp = next_dp
    return _logadd(dp[-1], dp[-2] if len(dp) > 1 else -math.inf)


def greedy_full_wake(raw, vocab, *, pad_token_id=0):
    """Conservative whole-utterance phoneme veto for ordinary negative labels."""
    size=len(vocab)
    need(type(raw) is bytes and len(raw)%(4*size)==0 and
         len(raw)>0 and len(raw)<=40*1024*1024 and
         type(pad_token_id) is int and pad_token_id==0,
         "FULL_PCM_PHONE_VETO_INPUT")
    inverse={v:k for k,v in vocab.items()}
    need(len(inverse)==size,"UNIQUE_NATIVE_PHONE_VOCAB")
    last=None
    decoded=[]
    for frame in struct.iter_unpack("<"+"f"*size,raw):
        need(all(math.isfinite(x) for x in frame),
             "FINITE_PHONE_VETO_LOGITS")
        ident=max(range(size),key=frame.__getitem__)
        if ident!=last and ident!=pad_token_id:
            decoded.append(inverse[ident])
        last=ident
    phones="".join(decoded)
    phones=re.sub(r"[\sˈˌːˑ͜͡]","",phones)
    phones=phones.replace("ɹ","r").replace("ɾ","r")
    mero=re.fullmatch(r"h?eɪmɛr(?:oʊ|o|ɔ)",phones)
    vosen=re.fullmatch(r"h?eɪv(?:oʊ|o|oː)s[əɐʌ]n",phones)
    return bool(mero or vosen)


def margin(frames, vocab, kind):
    positive, contrast = paths(kind)
    def highest(candidates):
        scores = [ctc_sequence_logp(frames, vocab, seq) for seq in candidates]
        present = [s for s in scores if s is not None]
        return max(present) if present else None
    yes, no = highest(positive), highest(contrast)
    if yes is None or no is None or not all(math.isfinite(x) for x in (yes, no)):
        return None
    return yes - no


def decide(row, transcripts, phone_evidence):
    """Separate acoustic evidence from requested text and synthetic voice identity.

    phone_evidence[model] = (byte-complete logits, original vocab, sha256).
    An UNKNOWN row NEVER turns into a background example.
    """
    need(type(row) is dict and type(transcripts) is dict and
         type(phone_evidence) is dict, "EXACT_REVIEW_INPUT_SHAPES")
    request = row.get("requested_class")
    need(request in (0, 1, 2, None), "UNSUPPORTED_REQUEST_CLASS")
    if request is None:
        return {"tier":"unknown", "class":None,
                "reason":"CONTRAST_DIAGNOSTIC_NEVER_TRAIN_NEGATIVE",
                "margins":{}, "training_allowed":False}
    if set(transcripts) != set(TEXT_MODELS) or set(phone_evidence) != set(PHONE_MODELS):
        return {"tier":"unknown","class":None,"reason":"MISSING_FIVE_MODEL_OBSERVATIONS",
                "margins":{}, "training_allowed":False}
    try:
        normalized = {key:norm(transcripts[key]) for key in TEXT_MODELS}
    except (TypeError,ValueError,KeyError):
        return {"tier":"unknown","class":None,"reason":"INCOMPLETE_BLIND_ASR",
                "margins":{}, "training_allowed":False}
    if any(text in OLD_ALIASES for text in normalized.values()):
        return {"tier":"unknown","class":None,"reason":"OLD_OR_VOICED_PRONUNCIATION_OBSERVED",
                "margins":{}, "training_allowed":False}
    if request in (1, 2) and not (
        type(row.get("pcm_samples")) is int and
        MIN_POSITIVE_SAMPLES <= row["pcm_samples"] <= MAX_POSITIVE_SAMPLES and
        row.get("positive_duration_pass") is True):
        return {"tier":"unknown","class":None,"reason":"UNCROPPED_POSITIVE_DURATION_GATE",
                "margins":{}, "training_allowed":False}
    try:
        parsed = {}
        for model in PHONE_MODELS:
            record = phone_evidence[model]
            raw, vocab, digest = record["logits"], record["vocab"], record["sha256"]
            need(type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest)
                 and sha(raw)==digest, "FULL_NATIVE_LOGIT_HASH_NOT_AUTHENTICATED")
            parsed[model] = verified_phone_input(raw,vocab,
                pad_token_id=record.get("pad_token_id",0))
    except (ValueError, KeyError, TypeError, struct.error, OverflowError):
        return {"tier":"unknown","class":None,"reason":"INVALID_OR_INCOMPLETE_NATIVE_PHONES",
                "margins":{}, "training_allowed":False}
    if request == 0:
        any_wake = any(text in ASR_ALIASES[1] | ASR_ALIASES[2] or
            text.startswith(("hey ", "hay ")) for text in normalized.values())
        votes = sum(x in ORDINARY for x in normalized.values())
        if any_wake or votes < 2 or any(
            greedy_full_wake(phone_evidence[name]["logits"],
                             phone_evidence[name]["vocab"])
            for name in PHONE_MODELS):
            return {"tier":"unknown","class":None,
                    "reason":"ORDINARY_NEGATIVE_NOT_INDEPENDENTLY_CONFIRMED",
                    "margins":{}, "training_allowed":False}
        return {"tier":"silver" if votes==3 else "weak", "class":0,
                "reason":"BLIND_ORDINARY_NO_WAKE", "margins":{},
                "training_allowed":False}
    competing = ASR_ALIASES[3-request]
    if any(x in competing for x in normalized.values()) or any(
        text.startswith(("hey ", "hay ")) and text not in ASR_ALIASES[request]
        for text in normalized.values()):
        return {"tier":"unknown","class":None,"reason":"CONTRADICTORY_ASR_NAME",
                "margins":{}, "training_allowed":False}
    votes = sum(x in ASR_ALIASES[request] for x in normalized.values())
    if votes < 2:
        return {"tier":"unknown","class":None,"reason":"INSUFFICIENT_CANONICAL_ASR",
                "margins":{}, "training_allowed":False}
    margins = {model:margin(parsed[model],phone_evidence[model]["vocab"],request)
               for model in PHONE_MODELS}
    if any(v is None for v in margins.values()):
        return {"tier":"unknown","class":None,"reason":"CANONICAL_CONTRAST_PHONE_UNSCORABLE",
                "margins":margins, "training_allowed":False}
    if any(v <= 0 for v in margins.values()):
        return {"tier":"unknown","class":None,"reason":"DUAL_PHONE_CONTRAST_NOT_POSITIVE",
                "margins":margins, "training_allowed":False}
    return {"tier":"silver" if votes==3 else "weak", "class":request,
            "reason":"TWO_PHONE_CTC_MARGINS_PLUS_BLIND_ASR",
            "margins":margins, "training_allowed":False}


def validate_protocol(plan):
    need(plan.get("schema")=="english-fresh24-preregistered-source-and-contrast-v1"
         and plan.get("status")=="FROZEN_BEFORE_NEW_AUDIO_NO_TRAINING_ADMISSION"
         and plan.get("target_phrases")=={
             "hey_mero":"/heɪ ˈmɛroʊ/","hey_vosen":"/heɪ ˈvoʊsən/"}
         and plan.get("role_counts")==FROZEN_SOURCE_ROLES
         and plan.get("minimum_qualified_groups")==MIN_QUALIFIED
         and plan.get("training_allowed") is False
         and plan.get("human_gold") is False
         and plan.get("legacy_positive_reuse") is False, "EXACT_PRE_AUDIO_BRAND_CONTRACT")
    rows=plan.get("voice_groups",[])
    need(len(rows)==24 and len({x.get("name") for x in rows})==24 and
         len({x.get("source_group_id") for x in rows})==24 and
         Counter(x.get("role") for x in rows)==Counter(FROZEN_SOURCE_ROLES)
         and not any(x.get("name") in OLD_PRESETS for x in rows),
         "FRESH_24_SOURCE_ROLE_OR_EXPOSURE_DRIFT")
    need(all(type(x.get("asset_sha256")) is str and
         re.fullmatch("[0-9a-f]{64}",x["asset_sha256"]) and
         type(x.get("asset_bytes")) is int and x["asset_bytes"]>500000 and
         x.get("source_group_id")=="kokoro-v1_0-preset:"+x["name"] and
         x.get("asset_path")=="voices/"+x["name"]+".pt" and
         x.get("source_kind")=="frozen_synthetic_voice_preset_not_human_speaker"
         for x in rows),"SOURCE_FAMILY_ANCESTRY_AND_PIN")
    cells=plan.get("cells",[])
    need(len(cells)==7 and len({x.get("cell_id") for x in cells})==7
         and Counter(x.get("requested_class") for x in cells)==
              Counter({2:2,1:2,0:1,None:2}) and
         all(x.get("label_is_not_truth") is True for x in cells),
         "SEVEN_PROSPECTIVE_CELLS_AND_CONTRAST")
    return {"status":"FROZEN_SOURCE_ONLY","groups":24,
            "roles":FROZEN_SOURCE_ROLES,"min_groups":MIN_QUALIFIED,
            "planned_new_clone_calls":168,"new_audio":0,
            "training_allowed":False, "policy_id":SCHEMA,
            "source_plan_sha256":sha(canonical(plan))}


def main():
    import argparse
    args=argparse.ArgumentParser()
    args.add_argument("--source-check",action="store_true")
    flags=args.parse_args()
    need(flags.source_check,"SOURCE_ONLY_NO_RESULT_UNTIL_AUTHENTICATED_AUDIO")
    print(json.dumps(validate_protocol(json.loads(PLAN_PATH.read_bytes())),
                     ensure_ascii=False,sort_keys=True))


if __name__=="__main__":
    main()
