"""Authenticate all previously exposed natural identities before any B17 source call."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

PLANS = Path(__file__).with_name("english_v6_b17_panels")
ASSESSMENT = PLANS / "assessment.json"
ASSESSMENT_SHA = "7fdfe9cd47b18f64429fece944468af8adfde6c6ba012a57236a8a33aae1a948"
ANCESTRY_SHA = "41c45e6bc5dcbf691551cf0565d1d3e1618a1b96280e6e39a8f2b0c1e9c15cf9"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_plan():
    raw = ASSESSMENT.read_bytes()
    need(digest(raw) == ASSESSMENT_SHA, "frozen B17 assessment before source/model exposure")
    assessment = json.loads(raw)
    raw = (PLANS / "source-ancestry.json").read_bytes()
    need(digest(raw) == ANCESTRY_SHA == assessment["source_ancestry_sha256"],
         "complete frozen B17 source ancestry")
    lineage = json.loads(raw)
    need(assessment["schema"] == "english-v6-b17-prospective-b0-ceiling-assessment-v1"
         and lineage["schema"] == "english-v6-b17-complete-source-ancestry-v1",
         "versioned source-independent study")
    old = set(lineage["legacy_natural"]["all_exposed_speakers"])
    excluded = set(lineage["natural_excluded_speakers"])
    need(len(old) == 40 and old <= excluded, "all 40 historical natural readers excluded")
    for panel in ("primary", "replication"):
        spec = assessment[panel]["natural"]
        census = lineage["metadata_census"]["complete_split_speakers"][spec["corpus_split"]]
        available = set(census) - excluded
        need(set(spec["available_census"]) == available
             and len(spec["speakers"]) == len(set(spec["speakers"])) == spec["requested_speakers"]
             and set(spec["speakers"]) <= available,
             "fixed unexposed natural reader roster")
        ranked = sorted(available, key=lambda sp: (digest((spec["salt"] + "/speaker/" + sp).encode()), sp))
        need(spec["speakers"] == ranked[:spec["requested_speakers"]], "prospective reader salt order")
    return assessment, lineage


def required_archives():
    _, lineage = read_plan()
    return tuple(str(lineage[k]["artifact_id"]) for k in
                 ("legacy_natural", "b13_natural_source", "b15_natural_source"))


def member(directory, lock, name):
    path = Path(directory) / (str(lock["artifact_id"]) + ".zip")
    raw = path.read_bytes()
    need(digest(raw) == lock["sha256"] and ("bytes" not in lock or len(raw) == lock["bytes"]),
         "original historical archive identity " + path.name)
    with zipfile.ZipFile(path) as archive:
        need(archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist())),
             "historical archive CRC and unique names")
        return archive.read(name)


def verify_history(directory):
    """Read only lineage/source metadata, never archived checkpoints or model scores."""
    assessment, lineage = read_plan()
    legacy = lineage["legacy_natural"]
    raw = member(directory, legacy, legacy["member"])
    need(digest(raw) == legacy["member_sha256"], "original legacy natural source member")
    old = json.loads(raw)
    people = {str(row["speaker"]) for row in old["actual_research_tracks"]}
    need(people == set(legacy["all_exposed_speakers"]) and len(people) == 40,
         "actual archived 40 readers, not an incomplete historical constant")
    people.update(str(row["speaker"]) for row in old["decoded_originals"])
    seen = {row["pcm_sha256"] for row in old["decoded_originals"]}
    for label in ("b13", "b15"):
        lock = lineage[label + "_natural_source"]
        raw = member(directory, lock, "ROSTERS.json")
        need(digest(raw) == lock["rosters_sha256"] == lineage["historical_rosters"][label + "_rosters"]["sha256"],
             "both primary and reserved historical reader rosters")
        rows = json.loads(raw)
        recorded = {str(row["speaker"]) for panel in rows.values() for row in panel}
        need(recorded == set(lineage["historical_rosters"][label + "_rosters"]["speakers"]),
             "original source roster ancestry " + label)
        people.update(recorded)
        source = json.loads(member(directory, lock, "MANIFEST.json"))
        seen.update(row["pcm_sha256"] for row in source["rows"])
    for key, relative in (("b14_rosters", "english_v6_b14_panels/natural-rosters.json"),
                          ("unadmitted_train_roster", "english_v6_train_context24_roster.json")):
        raw = (PLANS.parent / relative).read_bytes()
        lock = lineage["historical_rosters"][key]
        need(digest(raw) == lock["sha256"], "frozen earlier source role " + key)
        item = json.loads(raw)
        rows = item["source_rows"] if key == "unadmitted_train_roster" else [r for v in item.values() for r in v]
        recorded = {str(row["speaker"]) for row in rows}
        need(recorded == set(lock["speakers"]), "complete historical reader set " + key)
        people.update(recorded)
    excluded = set(lineage["natural_excluded_speakers"])
    need(people <= excluded, "no recorded historical identity omitted from new natural exclusions")
    for panel in ("primary", "replication"):
        need(not (people & set(assessment[panel]["natural"]["speakers"])),
             "old TRAIN/DEV/reserved speaker cannot become new B17 DEV")
    return seen, people
