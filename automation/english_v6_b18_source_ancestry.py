"""Authenticate original B17 source STOP/natural PASS before a new B18 source call."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile

import english_v6_b17_source as previous
import english_v6_b17_natural_sources as natural
import english_v6_b17_primary_eval as old_eval

PLANS = Path(__file__).with_name("english_v6_b18_panels")
ASSESSMENT = PLANS / "assessment.json"
ASSESSMENT_SHA = "4020b29b55efc8199a2fc6eed964b72fa842bb581504cf2c37e17ea8027e2490"
BASE_ANCESTRY_SHA = previous.lineage.ANCESTRY_SHA
SCHEMA = "english-v6-b18-complete-original-source-history-v1"
BRANCH = "refs/heads/research/en-v6-b18-synthetic-primary-once-20261124"
REPO = "jiying2007/kws-data"
sha, need = previous.lineage.digest, previous.qa.need


def read_plan():
    raw = ASSESSMENT.read_bytes()
    need(sha(raw) == ASSESSMENT_SHA, "prospective B18 evaluation and source contract")
    design = json.loads(raw)
    prior, _ = previous.lineage.read_plan()
    need(design["schema"] == "english-v6-b18-prospective-frozen-model-source-assessment-v1"
         and design["training_assessment_sha256"] == previous.lineage.ASSESSMENT_SHA
         and design["training_design_sha256"] == old_eval.DESIGN_SHA
         and design["base_source_ancestry_sha256"] == BASE_ANCESTRY_SHA
         and design["primary"]["synthetic"]["speaker_indices"] == list(range(766, 790))
         and design["replication"]["synthetic"]["speaker_indices"] == list(range(742, 766))
         and design["replication"]["natural"] == prior["replication"]["natural"]
         and design["primary"]["natural"] == prior["primary"]["natural"]
         and design["source_gate"] == old_eval.GATE
         and design["gains"] == ["1.0", "0.75", "0.5"]
         and design["new_training_allowed"] is False
         and design["new_natural_observer_calls"] == design["new_natural_source_calls"] == 0
         and design["maximum_source_calls"] == 96 and design["maximum_native_observer_calls"] == 480
         and design["B17_source_stop_retained"] is True and design["B17_scoring_allowed"] is False
         and design["all_prior_replication_sealed"] is True and design["replication_opened"] is False
         and design["promotion_allowed"] is False and design["threshold_search"] is False,
         "same trained weights and quality; only prospective new synthetic originals")
    for relative, expected in design["unchanged_code_sha256"].items():
        need(sha((PLANS.parent / relative).read_bytes()) == expected, "unchanged source/quality implementation " + relative)
    need(sha(old_eval.old_decision.ASSESSMENT.read_bytes()) == design["quality_assessment_sha256"],
         "original unchanged B14 acoustic criteria")
    need(sha((PLANS / "piper-config.json").read_bytes()) == design["primary"]["synthetic"]["config_sha256"],
         "retained official source mapping before any new source downloads")
    return design


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    e = os.environ if env is None else env
    need(re.fullmatch(r"[0-9a-f]{40}", prepared_parent or "") is not None,
         "exact reviewed predecessor required; inert placeholder cannot execute")
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH, "GITHUB_RUN_ATTEMPT": "1"}.items())
         and re.fullmatch(r"[0-9a-f]{40}", e.get("GITHUB_SHA", "")) is not None
         and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")) is not None,
         "first designated public source execution only")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == e["GITHUB_SHA"]
         and parent != e["GITHUB_SHA"]
         and event.get("ref") == BRANCH and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event.get("repository", {}).get("private") is False,
         "one public successor, no force push, multi-commit jump or retry")
    return dict(run_id=e["GITHUB_RUN_ID"], run_head=e["GITHUB_SHA"], run_attempt=1,
                run_branch=BRANCH, prepared_parent=parent)


def proof_locks():
    original = read_plan()["original_B17_source_proof"]
    locks = {k: v for k, v in original["source_inputs"].items() if v is not None}
    for cohort, providers in original["observer_inputs"].items():
        locks.update({cohort + "_" + p: pin for p, pin in providers.items()})
    need(len(locks) == 13 and "synth_screen" not in locks, "thirteen original archives; no fabricated synthetic SCREEN")
    return locks


def required_archives():
    return tuple(dict.fromkeys((*previous.required_archives(), *(str(p["id"]) for p in proof_locks().values()))))


def opened(directory, pin):
    raw = (Path(directory) / (str(pin["id"]) + ".zip")).read_bytes()
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "exact retained original source archive")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    need(archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist())),
         "original archive CRC and unique members")
    return archive


def extract_original(archive, target):
    target.mkdir()
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        need(not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
             and not stat.S_ISLNK(item.external_attr >> 16) and item.file_size < 100_000_000,
             "bounded original source member")
    archive.extractall(target)


def verify_original_b17(directory):
    design, pins = read_plan(), proof_locks()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for role, module in (("synth_source", previous), ("natural_source", natural)):
            with opened(directory, pins[role]) as archive:
                extract_original(archive, root / role)
        synth = previous.read_source(root / "synth_source")
        human = natural.read_source(root / "natural_source")
        all_rows = synth + human
        need(len(synth) == 96 and len(human) == 24 and len({r["pcm_sha256"] for r in all_rows}) == 120,
             "all B17 originals, including failed and unqualified sources")
        coverage = {}
        for cohort, rows, source, schema in (("synthetic", synth, "synth_source", previous.BATCH),
                                             ("natural", human, "natural_source", natural.SCHEMA)):
            providers = root / cohort
            providers.mkdir()
            for name in previous.qa.observers.PROVIDERS:
                with opened(directory, pins[cohort + "_" + name]) as archive:
                    extract_original(archive, providers / name)
            _, _, coverage[cohort] = previous.qa.load_observations(root / source, providers, rows, schema)
        need(coverage["synthetic"] == dict(qwen=96, whisper=96, parakeet=94, xlsr=96, lv60=96)
             and coverage["natural"] == dict.fromkeys(previous.qa.observers.PROVIDERS, 24),
             "retain the original B17 source STOP and fully observed natural panel")
        report = json.loads((root / "synthetic/parakeet/RESULT.json").read_bytes())
        failed = {r["id"]: r["error"] for r in report["rows"] if r["status"] == "failed"}
        need(failed == dict.fromkeys(design["original_B17_source_proof"]["failed_original_ids"],
                                     "ValueError:RAW_UNPROMPTED_COMPLETE_ASR_TEXT"),
             "original missing raw captures are not converted into blank/abstention results")
        with opened(directory, pins["natural_source"]) as source, opened(directory, pins["natural_screen"]) as screen:
            selected, counts = old_eval.verified_human(source, screen)
            native_screen = json.loads(screen.read("SCREEN.json"))
        need(counts == dict(natural_originals=21, natural_speakers=11)
             and native_screen["source_gate"] == "SOURCE_GATE_PASS" and native_screen["kws_forwards"] == 0,
             "unchanged actual original natural labels, no new natural source review")
        seen = {r["pcm_sha256"] for r in all_rows}
        groups = {r["source_group_id"] for r in all_rows}
        people = {r["source_speaker_id"].removeprefix("p") for r in synth} | {r["speaker"] for r in human}
        need(seen == set(design["all_B17_primary_exclusions"]["pcm_sha256"])
             and people == set(design["all_B17_primary_exclusions"]["canonical_people"])
             and len(people) == 36, "full frozen 120-original/36-person B17 increment")
        return seen, groups, people


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint])


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    design = read_plan()
    need(out.is_absolute() and not out.exists() and out.parent.is_dir(),
         "fresh absolute original-proof directory before any downloads")
    # This reads only original B17 proof and metadata; no B18 corpus/voice access.
    runs = {}
    for name, conclusion in (("synthetic", "failure"), ("natural", "success")):
        pin = design["original_B17_source_proof"]["runs"][name]
        run = json.loads(gh("actions/runs/" + str(pin["id"])))
        need(run["status"] == "completed" and run["conclusion"] == conclusion and run["run_attempt"] == 1
             and run["head_sha"] == pin["head"] and run["head_branch"] == pin["branch"]
             and run["event"] == "push" and run["repository"]["private"] is False
             and run["repository"]["full_name"] == REPO, "original public first-attempt B17 source outcome")
        runs[name] = {k: run[k] for k in ("id", "head_sha", "status", "conclusion", "run_attempt", "head_branch")}
    zero = json.loads(gh("actions/runs?branch=research%2Fen-v6-b17-primary-scoring-once-20261124&per_page=100"))
    need(zero["total_count"] == 0 and zero["workflow_runs"] == [], "B17 natural must still have zero KWS assessment exposure")
    out.mkdir()
    for name, pin in proof_locks().items():
        meta = json.loads(gh("actions/artifacts/" + str(pin["id"])))
        need(meta["expired"] is False and meta["size_in_bytes"] == pin["bytes"]
             and meta["digest"] == "sha256:" + pin["sha256"]
             and meta["workflow_run"]["id"] == pin["run_id"] and meta["workflow_run"]["head_sha"] == pin["run_head"],
             "original B17 source/native artifact identity " + name)
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "exact original B17 ZIP " + name)
        (out / (str(pin["id"]) + ".zip")).write_bytes(raw)
    verify_original_b17(out)
    receipt = dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, base_source_ancestry_sha256=BASE_ANCESTRY_SHA,
                   source_execution=identity, original_runs=runs, B17_designated_KWS_runs=zero,
                   original_proof_locks=proof_locks(), new_natural_calls=0, old_replication_opened=False,
                   new_training_allowed=False, new_KWS_forwards=0)
    (out / "AUTHORIZATION.json").write_text(json.dumps(receipt, indent=2) + "\n")


def require_authorization(directory):
    design = read_plan()
    receipt = json.loads((Path(directory) / "AUTHORIZATION.json").read_bytes())
    need(receipt["schema"] == SCHEMA and receipt["assessment_sha256"] == ASSESSMENT_SHA
         and receipt["base_source_ancestry_sha256"] == BASE_ANCESTRY_SHA
         and receipt["original_proof_locks"] == proof_locks()
         and receipt["B17_designated_KWS_runs"] == {"total_count": 0, "workflow_runs": []}
         and receipt["new_natural_calls"] == receipt["new_KWS_forwards"] == 0
         and receipt["old_replication_opened"] is False and receipt["new_training_allowed"] is False,
         "same actual B17 source STOP/natural PASS authorization")
    identity = receipt["source_execution"]
    need(set(identity) == {"run_id", "run_head", "run_attempt", "run_branch", "prepared_parent"}
         and isinstance(identity["run_id"], str) and re.fullmatch(r"[1-9][0-9]*", identity["run_id"]) is not None
         and re.fullmatch(r"[0-9a-f]{40}", identity["run_head"]) is not None
         and re.fullmatch(r"[0-9a-f]{40}", identity["prepared_parent"]) is not None
         and identity["run_head"] != identity["prepared_parent"]
         and type(identity["run_attempt"]) is int and identity["run_attempt"] == 1
         and identity["run_branch"] == BRANCH,
         "exact public source execution identity retained in immutable authorization")
    for name, expected in design["original_B17_source_proof"]["runs"].items():
        run = receipt["original_runs"][name]
        need(run["id"] == expected["id"] and run["head_sha"] == expected["head"]
             and run["run_attempt"] == 1 and run["status"] == "completed"
             and run["head_branch"] == expected["branch"]
             and run["conclusion"] == ("failure" if name == "synthetic" else "success"),
             "unchanged terminal original outcomes")
    return receipt


def history(directory):
    read_plan()
    seen, groups, people = previous.history(Path(directory))
    new_pcm, new_groups, new_people = verify_original_b17(directory)
    need(not (new_pcm & seen) and not (new_people & people), "actual B17 originals were independent of prior sources")
    seen.update(new_pcm); groups.update(new_groups); people.update(new_people)
    return seen, groups, people


def history_receipt(directory):
    seen, groups, people = history(directory)
    archives = {}
    for ident in required_archives():
        raw = (Path(directory) / (ident + ".zip")).read_bytes()
        archives[ident] = dict(bytes=len(raw), sha256=sha(raw))
    return dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, base_source_ancestry_sha256=BASE_ANCESTRY_SHA,
                original_archives=archives, all_pcm_sha256=sorted(seen), all_source_groups=sorted(groups),
                canonical_people=sorted(people, key=int), old_replication_opened=False,
                B17_source_STOP_retained=True, B17_all_original_slots_excluded=True)


def validate_history(receipt):
    design = read_plan()
    need(receipt["schema"] == SCHEMA and receipt["assessment_sha256"] == ASSESSMENT_SHA
         and receipt["base_source_ancestry_sha256"] == BASE_ANCESTRY_SHA
         and set(receipt["original_archives"]) == set(required_archives())
         and receipt["old_replication_opened"] is False and receipt["B17_source_STOP_retained"] is True
         and receipt["B17_all_original_slots_excluded"] is True,
         "complete original historical source receipt")
    seen, groups, people = map(set, (receipt["all_pcm_sha256"], receipt["all_source_groups"], receipt["canonical_people"]))
    _, original_ancestry = previous.lineage.read_plan()
    old_natural = set(original_ancestry["legacy_natural"]["all_exposed_speakers"])
    for roster in original_ancestry["historical_rosters"].values():
        old_natural.update(roster["speakers"])
    need(len(seen) == len(receipt["all_pcm_sha256"]) and len(groups) == len(receipt["all_source_groups"])
         and len(people) == len(receipt["canonical_people"])
         and old_natural <= people
         and set(design["all_B17_primary_exclusions"]["pcm_sha256"]) <= seen
         and set(design["all_B17_primary_exclusions"]["source_groups"]) <= groups
         and set(design["all_B17_primary_exclusions"]["canonical_people"]) <= people
         and all(re.fullmatch(r"[0-9a-f]{64}", value) is not None for value in seen)
         and all(re.fullmatch(r"[0-9]+", value) is not None for value in people)
         and len(seen) >= 1492 and len(groups) >= 312,
         "B17 complete original denominators cannot be reduced to admitted/successful rows")
    for pin in proof_locks().values():
        need(receipt["original_archives"][str(pin["id"])]["sha256"] == pin["sha256"]
             and receipt["original_archives"][str(pin["id"])]["bytes"] == pin["bytes"],
             "actual B17 archive membership in complete history")
    for ident, expected in design["base_original_archive_sha256"].items():
        need(receipt["original_archives"][ident]["sha256"] == expected
             and type(receipt["original_archives"][ident]["bytes"]) is int
             and receipt["original_archives"][ident]["bytes"] > 0,
             "every original pre-B17 archive remains SHA-locked")
    return seen, groups, people
