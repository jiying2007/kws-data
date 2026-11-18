"""Prospectively frozen dual-phone + triple-blind-ASR admission for cloned KWS.

This is a *new* study rule, not a retroactive reinterpretation of the old
Kokoro/Piper STOP gates. The two independent native phone observers must agree
on the brand-critical vowel/consonant, and the second confirms the GOAT tail.
Requested text never generates an observed label. Locked test is never used
to select thresholds or phoneme patterns.
"""
from collections import Counter,defaultdict
import json
import re
import unicodedata

SCHEMA="english-synthetic-clone12-dualphone-tripleasr-admission-v1"
TEXT_MODELS=("qwen","whisper","parakeet")
PHONE_MODELS=("xlsr","lv60")
ROLES={"train":6,"development":2,"calibration":2,"locked-test":2}
POSITIVE_MAX_SAMPLES=28672
M_ALIASES={"hey mero","hey marrow","hey merrow",
           "hay mero","hay marrow","hay merrow"}
V_ALIASES={"hey vosen","hey vohsen","hey vosin",
           "hay vosen","hay vohsen","hay vosin"}
OLD_ALIAS={"hey meero","hay meero","hey miro","hay miro",
           "hey vozen","hay vozen","hey vozin","hay vozin",
           "hey vohzen","hay vohzen","hey vohzin","hay vohzin"}
ORDINARY={"what time is it"}
# Pre-registered from independent observations 44253051549 / 44254394313,
# before seeing ANY 60-source outputs. XLSR often emits /ɔ/ for GOAT, while
# LV60 can show /oʊ/. Do not replace or rewrite v1 exact XLSR policy.
M_XLSR=re.compile(r"h?eɪmɛr(?:ɔ|o|oʊ)\Z")
M_LV60=re.compile(r"h?eɪmɛr(?:oʊ|o)\Z")
V_XLSR=re.compile(r"h?eɪv(?:oʊ|o|oː)s[əɐʌ]n\Z")
V_LV60=re.compile(r"h?eɪv(?:oʊ|o)s[əɐʌ]n\Z")
WAKE_PREFIX=re.compile(r"^(?:hey|hay)\b")

def need(value,code):
    if not value:raise ValueError(code)

def norm(txt):
    need(type(txt) is str and len(txt)<=4096,"RAW_ASR_TEXT")
    txt=unicodedata.normalize("NFKC",txt).casefold()
    return " ".join("".join(" " if unicodedata.category(x).startswith("P")
                            else x for x in txt).split())

def flattened(tokens):
    need(type(tokens) is list and 0<len(tokens)<=512 and
         all(type(x) is str and len(x)<32 for x in tokens),
         "COMPLETE_PHONE_TOKENS")
    x=re.sub(r"[\sˈˌːˑ͜͡]","",unicodedata.normalize("NFKC","".join(tokens)))
    return x.replace("ɹ","r").replace("ɾ","r")

def blind(texts,phones):
    """Independent observers only; no request label, role, donor or threshold."""
    if not isinstance(texts,dict) or set(texts)!=set(TEXT_MODELS) or \
       not isinstance(phones,dict) or set(phones)!=set(PHONE_MODELS):
        return ("unknown",None,"MISSING_INDEPENDENT_OBSERVATION")
    if any(not isinstance(texts[k],str) or not texts[k].strip()
           for k in TEXT_MODELS):
        return ("unknown",None,"MISSING_ASR_TEXT")
    try:
        readings={k:flattened(phones[k]) for k in PHONE_MODELS}
        words={k:norm(texts[k]) for k in TEXT_MODELS}
    except (TypeError,ValueError):
        return ("unknown",None,"BAD_OBSERVER_OUTPUT")
    if any(v in OLD_ALIAS for v in words.values()):
        return ("unknown",None,"LEGACY_OR_VOICED_CONTRADICTION")
    mero=M_XLSR.fullmatch(readings["xlsr"]) and M_LV60.fullmatch(readings["lv60"])
    vosen=V_XLSR.fullmatch(readings["xlsr"]) and V_LV60.fullmatch(readings["lv60"])
    if mero or vosen:
        label,aliases,competitor=(2,M_ALIASES,V_ALIASES) if mero else (1,V_ALIASES,M_ALIASES)
        if any(v in competitor for v in words.values()):
            return ("unknown",None,"OTHER_WAKE_OBSERVED")
        n=sum(v in aliases for v in words.values())
        # Third ASR may be silent/ordinary, but it cannot assert a different
        # wake phrase. This keeps the two-of-three weak tier fail-closed.
        if any(WAKE_PREFIX.search(v) and v not in aliases for v in words.values()):
            return ("unknown",None,"CONTRADICTING_THIRD_ASR")
        if n>=2:
            return ("silver" if n==3 else "weak",label,
                    "TWO_INDEPENDENT_PHONES_AND_BLIND_ASR")
        return ("unknown",None,"TWO_OF_THREE_ASR_NOT_MET")
    # One-sided phoneme resemblance is NOT a trusted negative.
    if (M_XLSR.fullmatch(readings["xlsr"]) or M_LV60.fullmatch(readings["lv60"])
         or V_XLSR.fullmatch(readings["xlsr"]) or V_LV60.fullmatch(readings["lv60"])):
        return ("unknown",None,"ONE_SIDED_WAKE_PHONE")
    if any(WAKE_PREFIX.search(v) for v in words.values()):
        return ("unknown",None,"POSSIBLE_WAKE_NOT_BACKGROUND")
    n=max((sum(v==phrase for v in words.values()) for phrase in ORDINARY),default=0)
    if n>=2:
        return ("silver" if n==3 else "weak",0,"BLIND_ORDINARY_WITH_NO_WAKE")
    return ("unknown",None,"NO_TRUSTED_ORDINARY_LABEL")

def evaluate(source_rows,asr_by_id,phones_by_id):
    """Return explicit every-row decisions; never silently drop UNKNOWN rows."""
    need(type(source_rows) is list and len(source_rows)==60,
         "FIXED_SIXTY_SOURCE_DENOMINATOR")
    need(len({x["id"] for x in source_rows})==60,"UNIQUE_ROW_IDS")
    need(set(asr_by_id)==set(phones_by_id)=={x["id"] for x in source_rows},
         "EXACT_SIXTY_OBSERVER_ROSTER")
    results=[]
    source_groups=defaultdict(list)
    for row in source_rows:
        sid=row["id"]
        tier,label,reason=blind(asr_by_id[sid],phones_by_id[sid])
        if tier!="unknown" and label!=row["requested_brand_class"]:
            tier,label,reason="unknown",None,"REQUEST_CONTRADICTED_BY_OBSERVERS"
        if tier!="unknown" and label in (1,2) and (
            not row.get("positive_duration_pass") or
            row.get("samples",100000000)>POSITIVE_MAX_SAMPLES):
            tier,label,reason="unknown",None,"POSITIVE_DURATION_TOO_LONG_NO_CROP"
        key=row["source_group_id"]
        need(row.get("role") in ROLES and type(key) is str and key,
             "FROZEN_SOURCE_GROUP_AND_ROLE")
        item={"id":sid,"source_group_id":key,"role":row["role"],
              "cell_id":row["cell_id"],"tier":tier,"observed_class":label,
              "reason":reason,"wav_sha256":row["wav_sha256"],
              "pcm_sha256":row["pcm_sha256"],
              "samples":row["samples"],"label_basis":
              "BLIND_THREE_ASR_AND_DUAL_COMPLETE_PHONES" if tier!="unknown" else None,
              "training_allowed":False,"human_gold":False}
        results.append(item)
        source_groups[key].append(item)
    need(len(source_groups)==12 and all(len(x)==5 for x in source_groups.values()),
         "EXACT_FIVE_ROWS_PER_TWELVE_SOURCE_GROUPS")
    complete=set()
    for key,items in source_groups.items():
        need(len({x["role"] for x in items})==1,"GROUP_ROLE_LEAKAGE")
        classes={x["observed_class"] for x in items if x["tier"]!="unknown"}
        if classes=={0,1,2}:complete.add(key)
    role_counts=Counter(source_groups[k][0]["role"] for k in complete)
    passes=all(role_counts[r]>=ROLES[r] for r in ROLES)
    # Result rows themselves remain research-only. A separately authenticated
    # source+model manifest must be written to enable the fresh optimizer.
    return {"schema":SCHEMA,
            "status":"PASS_CANONICAL_SOURCE_COVERAGE" if passes
                    else "STOP_CANONICAL_SOURCE_COVERAGE",
            "denominator":60,
            "complete_groups_by_role":{r:role_counts[r] for r in ROLES},
            "minimum_groups_by_role":ROLES,
            "tier_counts":dict(Counter(x["tier"] for x in results)),
            "reason_counts":dict(Counter(x["reason"] for x in results)),
            "source_group_count":len(source_groups),
            "training_snapshot_allowed":passes,
            "new_optimizer_updates":0,
            "new_model_weights":False,
            "human_gold":False,"independent_real_far_frr":False,
            "rows":results}
