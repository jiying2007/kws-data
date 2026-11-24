"""B21 source-only identity and byte-authenticated complete prior exposure.

The original B20 source archive includes its authenticated full historical
receipt, original canary proof and every synthetic source, including UNKNOWNs.
It is a fixed evidence checkpoint, never new training or evaluation material.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

import english_v6_b20_source as previous

qa, need = previous.qa, previous.qa.need
PLANS = Path(__file__).with_name("english_v6_b21_panels")
ASSESSMENT = PLANS / "assessment.json"
ASSESSMENT_SHA = "65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc"
ANCESTRY_SHA = "26cf48813b5e712bc8820a2429fc0917db9f6e3eaa809a008d98c533714be2b0"
REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b21-primary-sources-once-20261124"
SCHEMA = "english-v6-b21-complete-source-history-v1"
SOURCE_PIN = dict(id=14892539914, run_id=45334413855,
    run_head="249449c8a572c536dfb4e990584b451247d7da17", bytes=8974682,
    sha256="de5c580f7c6cdd99abd3268f5667b60d748d21f15ab5e2451123962b26775505")
B20_MANIFEST_SHA = "8acc2cca1eb775174eb8b4e3d18a26ab7a68e9dfd35b9dd811d428eb0a57d4b8"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def read_ancestry():
    raw = (PLANS / "ancestry.json").read_bytes()
    need(digest(raw) == ANCESTRY_SHA, "frozen B21 complete previous exposure")
    value = json.loads(raw)
    need(value["schema"] == SCHEMA and value["B20_original_source"] == SOURCE_PIN
         and value["B20_manifest_sha256"] == B20_MANIFEST_SHA
         and value["all_prior_replication_sealed"] is True,
         "source-only checkpoint identity and sealed historical reserves")
    for key in ("all_pcm_sha256", "all_source_groups", "canonical_people"):
        values = value[key]
        need(values == sorted(set(values)), "unique canonically ordered history: " + key)
    return value


def read_plan():
    raw = ASSESSMENT.read_bytes()
    need(re.fullmatch(r"[0-9a-f]{64}", ASSESSMENT_SHA) is not None and digest(raw) == ASSESSMENT_SHA,
         "separately frozen B21 assessment before source or training execution")
    design = json.loads(raw)
    ancestry = read_ancestry()
    need(design["schema"] == "english-v6-b21-prospective-blank-inclusive-retention-assessment-v1"
         and design["ancestry_sha256"] == ANCESTRY_SHA
         and design["source_gate"] == previous.lineage.read_plan()["source_gate"]
         and design["quality_criteria"] == previous.lineage.read_plan()["quality_criteria"]
         and design["quality_assessment_sha256"] == previous.lineage.read_plan()["quality_assessment_sha256"]
         and design["gains"] == ["1.0", "0.75", "0.5"]
         and design["maximum_public_Piper_synthesize_calls"] == 96
         and design["maximum_natural_original_decodes"] == 24
         and design["maximum_native_observer_calls"] == 600
         and design["training_allowed_for_sources"] is design["promotion_allowed"] is design["replication_opened"] is False,
         "unchanged independent quality gates, bounded fresh source stage only")
    synthetic_people = set()
    config = json.loads((previous.lineage.canary.previous.PLANS / "piper-config.json").read_bytes())
    for role, first in (("primary", 846), ("replication", 870)):
        family = design[role]["synthetic"]
        need(family["speaker_indices"] == list(range(first, first + 24)), "fixed disjoint new 24-voice block")
        for index, person in family["speaker_id_map"].items():
            digits = person.removeprefix("p")
            need(config["speaker_id_map"].get(person) == int(index)
                 and digits not in ancestry["synthetic_excluded_people"] and digits not in synthetic_people,
                 "new canonical voice excludes actual exposure and every sealed reserve")
            synthetic_people.add(digits)
    need(len(synthetic_people) == 48, "all 48 new primary/reserve voices are independent")
    raw_census = (PLANS / "SPEAKERS.TXT").read_bytes()
    need(digest(raw_census) == design["metadata_speakers_sha256"], "official source census bytes")
    pool = {x.split("|")[0].strip() for x in raw_census.decode().splitlines()
            if x.strip() and not x.startswith(";") and x.split("|")[2].strip() == "train-clean-100"}
    excluded = set(ancestry["natural_excluded_people"]) | synthetic_people
    chosen = set()
    for role in ("primary", "replication"):
        natural = design[role]["natural"]
        available = pool - excluded - chosen
        ranked = sorted(available, key=lambda p: (digest((natural["salt"] + "/speaker/" + p).encode()), p))
        need(natural["speakers"] == ranked[:12] and natural["source_slots"] == 24
             and natural["requested_speakers"] == 12 and natural["utterances_per_speaker"] == 2
             and natural["project_role"] == "development" and natural["training_allowed"] is False
             and natural["corpus_split"] == "train-clean-100"
             and natural["upstream"] == previous.lineage.read_plan()["replication"]["natural"]["upstream"],
             "metadata-only fixed natural readers; upstream TRAIN means project DEV here")
        chosen.update(natural["speakers"])
    need(len(chosen) == 24 and not chosen & synthetic_people, "fresh primary and replication reader separation")
    return design


def validate_identity(identity):
    need(set(identity) == {"run_id", "run_head", "run_attempt", "run_branch", "prepared_parent"}
         and re.fullmatch(r"[1-9][0-9]*", identity["run_id"] or "") is not None
         and all(re.fullmatch(r"[0-9a-f]{40}", identity[k] or "") is not None for k in ("run_head", "prepared_parent"))
         and identity["run_head"] != identity["prepared_parent"] and identity["run_attempt"] == 1
         and identity["run_branch"] == BRANCH, "actual first public B21 source identity")
    return identity


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    read_plan()
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_REF": BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1"}.items()),
         "first separately registered public B21 source event only")
    identity = validate_identity(dict(run_id=e.get("GITHUB_RUN_ID", ""), run_head=e.get("GITHUB_SHA", ""),
        run_attempt=1, run_branch=BRANCH, prepared_parent=prepared_parent))
    if event is None: event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None: parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == identity["run_head"]
         and event.get("ref") == BRANCH and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event["repository"].get("private") is False,
         "exact reviewed predecessor, no retry, private execution or forced/multiple-commit push")
    return identity


def validate_history(value):
    frozen = read_ancestry()
    need(value == frozen, "every prior raw PCM/group/person, including UNKNOWN and new B20 source slots")
    return tuple(set(value[k]) for k in ("all_pcm_sha256", "all_source_groups", "canonical_people"))


def authenticate_previous(directory):
    directory = Path(directory)
    raw = (directory / "B20_SOURCE.zip").read_bytes()
    need(len(raw) == SOURCE_PIN["bytes"] and digest(raw) == SOURCE_PIN["sha256"], "original B20 source archive bytes")
    # A fixed archive SHA authenticates its exact member paths before extraction.
    with zipfile.ZipFile(__import__("io").BytesIO(raw)) as archive:
        need(archive.testzip() is None, "original B20 source ZIP CRC")
        for name in archive.namelist():
            path = Path(name)
            need(not path.is_absolute() and ".." not in path.parts, "safe original source archive members")
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "B20_SOURCE"
        source.mkdir()
        with zipfile.ZipFile(__import__("io").BytesIO(raw)) as archive: archive.extractall(source)
        rows = previous.read_source(source)
        need(digest((source / "MANIFEST.json").read_bytes()) == B20_MANIFEST_SHA and len(rows) == 96,
             "all 96 original B20 source rows, including rejected labels")
        old = json.loads((source / "HISTORY.json").read_bytes())
        previous.lineage.validate_history(old)
        combined = (set(old["all_pcm_sha256"]) | {r["pcm_sha256"] for r in rows},
                    set(old["all_source_groups"]) | {r["source_group_id"] for r in rows},
                    set(old["canonical_people"]) | {r["source_speaker_id"].removeprefix("p") for r in rows})
        frozen = read_ancestry()
        need(all(values == set(frozen[k]) for k, values in zip(("all_pcm_sha256", "all_source_groups", "canonical_people"), combined)),
             "authenticated history checkpoint plus all actual B20 originals, no missing unknowns")
        return frozen


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint])


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), "fresh source authorization destination")
    run = json.loads(gh("actions/runs/" + str(SOURCE_PIN["run_id"])))
    artifact = json.loads(gh("actions/artifacts/" + str(SOURCE_PIN["id"])))
    need(run["id"] == SOURCE_PIN["run_id"] and run["head_sha"] == SOURCE_PIN["run_head"]
         and run["run_attempt"] == 1 and run["conclusion"] == "success" and run["status"] == "completed"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and artifact["id"] == SOURCE_PIN["id"] and artifact["size_in_bytes"] == SOURCE_PIN["bytes"]
         and artifact["digest"] == "sha256:" + SOURCE_PIN["sha256"] and artifact["expired"] is False
         and artifact["workflow_run"]["id"] == SOURCE_PIN["run_id"]
         and artifact["workflow_run"]["head_sha"] == SOURCE_PIN["run_head"], "authentic completed previous public source producer")
    raw = gh("actions/artifacts/" + str(SOURCE_PIN["id"]) + "/zip")
    need(len(raw) == SOURCE_PIN["bytes"] and digest(raw) == SOURCE_PIN["sha256"], "original source checkpoint download")
    out.mkdir()
    (out / "B20_SOURCE.zip").write_bytes(raw)
    history = authenticate_previous(out)
    qa.record(out / "HISTORY.json", history)
    qa.record(out / "AUTHORIZATION.json", dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA,
        ancestry_sha256=ANCESTRY_SHA, source_execution=identity, B20_original_source=SOURCE_PIN,
        history_sha256=qa.sha(out / "HISTORY.json"), previous_source_run=run, previous_source_artifact=artifact,
        all_prior_replication_sealed=True, new_KWS_forwards=0, promotion_allowed=False))


def require_authorization(directory):
    read_plan()
    directory = Path(directory)
    permission = json.loads((directory / "AUTHORIZATION.json").read_bytes())
    validate_identity(permission["source_execution"])
    need(permission["schema"] == SCHEMA and permission["assessment_sha256"] == ASSESSMENT_SHA
         and permission["ancestry_sha256"] == ANCESTRY_SHA and permission["B20_original_source"] == SOURCE_PIN
         and permission["all_prior_replication_sealed"] is True and permission["new_KWS_forwards"] == 0
         and permission["promotion_allowed"] is False
         and permission["history_sha256"] == qa.sha(directory / "HISTORY.json"), "original source-only authorization")
    run, artifact = permission["previous_source_run"], permission["previous_source_artifact"]
    need(run["id"] == SOURCE_PIN["run_id"] and run["head_sha"] == SOURCE_PIN["run_head"]
         and run["run_attempt"] == 1 and run["status"] == "completed" and run["conclusion"] == "success"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and artifact["id"] == SOURCE_PIN["id"] and artifact["size_in_bytes"] == SOURCE_PIN["bytes"]
         and artifact["digest"] == "sha256:" + SOURCE_PIN["sha256"]
         and artifact["workflow_run"]["id"] == SOURCE_PIN["run_id"]
         and artifact["workflow_run"]["head_sha"] == SOURCE_PIN["run_head"], "retained exact previous producer receipt")
    validate_history(json.loads((directory / "HISTORY.json").read_bytes()))
    authenticate_previous(directory)
    return permission


def retain_proof(directory, output):
    output = Path(output)
    for name in ("B20_SOURCE.zip", "AUTHORIZATION.json", "HISTORY.json"):
        shutil.copyfile(Path(directory) / name, output / name)
