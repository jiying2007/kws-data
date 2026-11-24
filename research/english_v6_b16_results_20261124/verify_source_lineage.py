"""Audit frozen speaker/source metadata only; never run a model or decode audio."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
LOCKS = {
    "legacy": {
        "run_id": 43867764366,
        "head": "d8be1c24b12cac0ba554e9a4a41e4b6ee0db18db",
        "artifact_id": 14225893574,
        "bytes": 61290521,
        "sha256": "2aa398cd11b787342d91c056fed188d8e3607929da9fdaf873451b5cf05b07f1",
        "members": {
            "LIBRISPEECH_SOURCES.json": "8a21f832bd847b8a169f41de0fb40c5cc9b683dffc607d9e100ebc5958f4b15d",
            "STREAM_ROSTER.json": "53646ce22780e8647e5654334c633a3f4dc8c1dd9a78d0856a1877b99d75b12a",
        },
    },
    "source": {
        "run_id": 45297962939,
        "head": "3da0d5a35d8dcd7d1c79bbf870a1df776c9d7c8d",
        "artifact_id": 14878351529,
        "bytes": 3983738,
        "sha256": "a5b0e07ebd583bce0fd561a738fcd540619541a22534bbdf3f20b2550d2eda97",
        "members": {
            "MANIFEST.json": "5a3fb7683dae90fb4c73791e71ee30c1818a79a096617cea8027b6f9c0e48340",
            "ROSTERS.json": "68704e3e8d85836a4fa426a19a023bb7d01869ae0f3a4a6f75a84f354d2a7697",
        },
    },
    "scores": {
        "run_id": 45314679545,
        "head": "737fc6ccf129b82e74d595ff7ea985f60f2d136b",
        "artifact_id": 14884732009,
        "bytes": 2425605,
        "sha256": "aa57b2dae6859f12f1e939a1815c58678a1db270e8bb21b66378783be6fe3f2b",
        "members": {
            "RESULT.json": "034a02f77afd53fb0cca50d5ffc54088b6ba745dcd3fc243a957aaaa9fc4b7cf",
        },
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_person(speaker):
    value = str(speaker)
    require(bool(re.fullmatch(r"[1-9][0-9]*", value)), "numeric LibriSpeech speaker identity")
    return "librispeech:" + value


def read_locked(path, lock):
    raw = path.read_bytes()
    require(len(raw) == lock["bytes"] and sha256(raw).hexdigest() == lock["sha256"],
            "exact original archive identity")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None,
                "unique archive members and complete CRC")
        documents = {}
        for member, expected in lock["members"].items():
            data = archive.read(member)
            require(sha256(data).hexdigest() == expected, "exact metadata member " + member)
            documents[member] = json.loads(data)
    return documents


def cross_check(legacy, prior_roster, current, current_roster, scores):
    """Join numeric corpus identities, never namespace strings or decoder outcomes."""
    old_tracks = {canonical_person(row["speaker"]): row for row in legacy["actual_research_tracks"]}
    require(len(old_tracks) == len(legacy["actual_research_tracks"]), "unique historical readers")
    prior = {row["id"]: row for row in prior_roster}
    for row in old_tracks.values():
        used = prior[row["id"]]
        require(used["role"] == row["role"] == used["source"]["role"]
                and str(used["source"]["speaker"]) == str(row["speaker"])
                and used["source"]["pcm_sha256"] == row["pcm_sha256"]
                and used["observed_samples"] == row["frames"] and used["records"] > 0,
                "historical reader was in actual feature stream, not just a proposed constant")
    decoded = {row["id"]: row for row in legacy["decoded_originals"]}
    require(len(decoded) == len(legacy["decoded_originals"]), "unique historical decoded sources")
    source_rows = {row["id"]: row for row in current["rows"]}
    roster = {row["source_id"]: row for row in current_roster["primary"]}
    require(len(source_rows) == len(current["rows"]), "unique current source rows")
    selected = [row for row in scores["records"] if row["cohort"] == "natural"]
    require(len({row["id"] for row in selected}) == len(selected), "unique scored human originals")
    overlaps = []
    for record in selected:
        row = source_rows[record["id"]]
        person = canonical_person(row["speaker"])
        expected_member = "LibriSpeech/dev-clean/{speaker}/{chapter}/{source_id}.flac".format(**row)
        require(row["source_member"] == expected_member
                and row["source_id"].split("-")[0] == str(row["speaker"])
                and all(row[key] == value for key, value in roster[row["source_id"]].items())
                and record["group"] == row["source_group_id"]
                and row["pcm_sha256"] == record["pcm_sha256"]
                and row["role"] == "development" and record["positive"] is False,
                "scored original bound to full source/roster/canonical reader")
        if person not in old_tracks:
            continue
        old = old_tracks[person]
        original = decoded.get(row["source_member"])
        if original is not None:
            require(original["pcm_sha256"] == row["pcm_sha256"]
                    and original["flac_sha256"] == row["flac_sha256"],
                    "identical historical decoded original bytes")
        overlaps.append({
            "canonical_person": person,
            "historical_track_id": old["id"],
            "historical_role": old["role"],
            "historical_track_pcm_sha256": old["pcm_sha256"],
            "current_id": row["id"],
            "current_source_group_id": row["source_group_id"],
            "current_source_member": row["source_member"],
            "current_pcm_sha256": row["pcm_sha256"],
            "current_flac_sha256": row["flac_sha256"],
            "same_historical_decoded_original": original is not None,
            "present_in_historical_splice_fragments": any(
                fragment["source_id"] == row["source_member"] for fragment in old["fragments"]
            ),
        })
    overlaps.sort(key=lambda row: (row["canonical_person"], row["current_id"]))
    old_people = {row["canonical_person"] for row in overlaps}
    current_people = {canonical_person(source_rows[row["id"]]["speaker"]) for row in selected}
    return {
        "status": "SOURCE_LEAKAGE" if overlaps else "NO_KNOWN_OVERLAP_IN_AUDITED_ARCHIVE",
        "historical_actual_reader_count": len(old_tracks),
        "historical_roles": dict(sorted(Counter(row["role"] for row in old_tracks.values()).items())),
        "observed_natural_originals": len(selected),
        "observed_natural_readers": len(current_people),
        "overlapping_originals": len(overlaps),
        "overlapping_readers": len(old_people),
        "overlapping_readers_by_historical_role": dict(sorted(Counter(
            old_tracks[person]["role"] for person in old_people).items())),
        "overlapping_originals_by_historical_role": dict(sorted(Counter(
            row["historical_role"] for row in overlaps).items())),
        "known_disjoint_natural_originals": len(selected) - len(overlaps),
        "known_disjoint_natural_readers": len(current_people - old_people),
        "same_historical_decoded_originals": sum(row["same_historical_decoded_original"] for row in overlaps),
        "same_historical_splice_fragment_originals": sum(row["present_in_historical_splice_fragments"] for row in overlaps),
        "identity_rows": overlaps,
    }


def verify(legacy_path, source_path, score_path):
    old = read_locked(legacy_path, LOCKS["legacy"])
    new = read_locked(source_path, LOCKS["source"])
    scored = read_locked(score_path, LOCKS["scores"])["RESULT.json"]
    result = cross_check(old["LIBRISPEECH_SOURCES.json"], old["STREAM_ROSTER.json"],
                         new["MANIFEST.json"], new["ROSTERS.json"], scored)
    require(result["status"] == "SOURCE_LEAKAGE"
            and result["overlapping_originals"] == 18 and result["overlapping_readers"] == 12
            and result["known_disjoint_natural_readers"] == 0
            and result["same_historical_decoded_originals"] == 2,
            "frozen observed lineage correction")
    require(scored["actual_forwards"] == 912 and scored["source_gate"] == "SOURCE_GATE_PASS"
            and scored["panel_gate"] == "NO_PROMOTION" and not scored["promotion_allowed"],
            "preserve original numerical outcome; do not rerun or select a new result")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("scores", type=Path)
    args = parser.parse_args()
    observed = verify(args.legacy, args.source, args.scores)
    expected = json.loads((ROOT / "SOURCE_LINEAGE_AUDIT.json").read_bytes())["observed"]
    require(observed == expected, "persisted lineage audit is reproducible from original metadata")
    print(json.dumps({key: value for key, value in observed.items() if key != "identity_rows"}, sort_keys=True))
