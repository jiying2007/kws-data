"""Explicit audited continuation of a B21 transport stop with zero source calls.

This is a separate first execution, never a rerun of the failed workflow. The
original frozen history, source requests, assessment and observers stay intact.
"""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import english_v6_b21_source_ancestry as frozen
import english_v6_b21_continuation_source_ancestry as previous
import english_v6_b21_continuation_origin as original_origin
import english_v6_b21_fixed_mirror_origin as origin

qa, need, digest, canonical = frozen.qa, frozen.need, frozen.digest, frozen.canonical
PLANS, ASSESSMENT, ASSESSMENT_SHA = frozen.PLANS, frozen.ASSESSMENT, frozen.ASSESSMENT_SHA
ANCESTRY_SHA, SOURCE_PIN, REPO = frozen.ANCESTRY_SHA, frozen.SOURCE_PIN, frozen.REPO
read_plan, read_ancestry, validate_history = frozen.read_plan, frozen.read_ancestry, frozen.validate_history
BRANCH = "refs/heads/research/en-v6-b21-primary-source-fixed-mirror-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-primary-source-fixed-mirror-once.yml"
SCHEMA = "english-v6-b21-audited-fixed-mirror-authorization-v1"
MARKER = "english-v6-b21-source-fixed-mirror-consumed"
CORPUS_BYTES = 6387309499
CORPUS_SHA = "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2"


def gh(endpoint):
    return subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint])


def verify_frozen_producer(root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    origin.verify()
    first = previous.verify_frozen_producer(root)
    second = origin.verify_current_producer(root)
    return sorted(set(first) | set(second))


def validate_identity(identity):
    need(set(identity) == {"run_id", "run_head", "run_attempt", "run_branch", "prepared_parent"}
         and re.fullmatch(r"[1-9][0-9]*", identity["run_id"] or "") is not None
         and all(re.fullmatch(r"[0-9a-f]{40}", identity[k] or "") is not None for k in ("run_head", "prepared_parent"))
         and identity["run_head"] != identity["prepared_parent"] and identity["run_attempt"] == 1
         and identity["run_branch"] == BRANCH, "one new public continuation identity")
    return identity


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    read_plan(); origin.verify(); verify_frozen_producer()
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_REF": BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1"}.items()),
         "first separately reviewed public source continuation only")
    identity = validate_identity(dict(run_id=e.get("GITHUB_RUN_ID", ""), run_head=e.get("GITHUB_SHA", ""),
        run_attempt=1, run_branch=BRANCH, prepared_parent=prepared_parent))
    if event is None: event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None: parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == identity["run_head"]
         and event.get("ref") == BRANCH and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event["repository"].get("private") is False,
         "reviewed single predecessor and new nonforced first push; old failed run is immutable")
    return identity


def one_execution(identity, marker_present):
    branch = BRANCH.removeprefix("refs/heads/")
    runs = json.loads(gh("actions/workflows/" + Path(WORKFLOW).name + "/runs?branch=" + branch + "&per_page=100"))
    need(runs["total_count"] == len(runs["workflow_runs"]) == 1
         and str(runs["workflow_runs"][0]["id"]) == identity["run_id"]
         and runs["workflow_runs"][0]["head_sha"] == identity["run_head"]
         and runs["workflow_runs"][0]["run_attempt"] == 1, "exactly one continuation run, including expired/failed history")
    markers = json.loads(gh("actions/artifacts?name=" + MARKER + "&per_page=100"))
    need(markers["total_count"] == len(markers["artifacts"]) == int(marker_present), "one independent continuation consumption marker")
    if marker_present:
        marker = markers["artifacts"][0]
        need(marker["name"] == MARKER and marker["expired"] is False
             and str(marker["workflow_run"]["id"]) == identity["run_id"]
             and marker["workflow_run"]["head_sha"] == identity["run_head"], "this continuation owns the generation marker")
        return marker
    return None


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    zero = origin.verify()
    one_execution(identity, False)
    run_id = str(zero["original_run_id"])
    live = {name: json.loads(gh(endpoint)) for name, endpoint in {
        "run": "actions/runs/" + run_id,
        "jobs": "actions/runs/" + run_id + "/jobs?filter=all&per_page=100",
        "artifacts": "actions/runs/" + run_id + "/artifacts?per_page=100"}.items()}
    original_origin.validate_remote(live["run"], live["jobs"], live["artifacts"])
    failed_id = str(zero["failed_continuation_run_id"])
    failed_live = {name: json.loads(gh(endpoint)) for name, endpoint in {
        "run": "actions/runs/" + failed_id,
        "jobs": "actions/runs/" + failed_id + "/jobs?filter=all&per_page=100",
        "artifacts": "actions/runs/" + failed_id + "/artifacts?per_page=100"}.items()}
    origin.validate_remote(failed_live["run"], failed_live["jobs"], failed_live["artifacts"])
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), "fresh continuation authorization directory")
    raw = gh("actions/artifacts/" + str(SOURCE_PIN["id"]) + "/zip")
    need(len(raw) == SOURCE_PIN["bytes"] and digest(raw) == SOURCE_PIN["sha256"], "unchanged already-exposed B20 history checkpoint")
    out.mkdir(); prior = out / "PRIOR"; prior.mkdir()
    shutil.copytree(original_origin.FAILURE_DIR, out / "TRANSPORT_ORIGIN")
    shutil.copytree(origin.FAILURE_DIR, out / "FIXED_MIRROR_ORIGIN")
    with zipfile.ZipFile(original_origin.FAILURE_DIR / "opened-original.zip") as archive:
        (prior / "AUTHORIZATION.json").write_bytes(archive.read("AUTHORIZATION.json"))
    (prior / "B20_SOURCE.zip").write_bytes(raw)
    qa.record(prior / "HISTORY.json", frozen.authenticate_previous(prior))
    frozen.require_authorization(prior)
    shutil.copyfile(prior / "HISTORY.json", out / "HISTORY.json")
    qa.record(out / "ORIGINAL_RUN_LIVE.json", live)
    qa.record(out / "FAILED_CONTINUATION_LIVE.json", failed_live)
    permission = dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, ancestry_sha256=ANCESTRY_SHA,
        source_execution=identity, transport_origin=zero, B20_original_source=SOURCE_PIN,
        history_sha256=qa.sha(out / "HISTORY.json"), original_run_live_sha256=qa.sha(out / "ORIGINAL_RUN_LIVE.json"),
        failed_continuation_live_sha256=qa.sha(out / "FAILED_CONTINUATION_LIVE.json"),
        original_authorization_sha256=qa.sha(prior / "AUTHORIZATION.json"), all_prior_replication_sealed=True,
        new_KWS_forwards=0, promotion_allowed=False, training_allowed=False)
    qa.record(out / "AUTHORIZATION.json", permission)
    return require_authorization(out)


def require_authorization(directory):
    read_plan()
    directory = Path(directory)
    zero = origin.verify(directory / "FIXED_MIRROR_ORIGIN")
    need(zero == origin.verify(), "retained exact two zero-call transport proofs")
    need(original_origin.verify(directory / "TRANSPORT_ORIGIN") == original_origin.verify(),
         "earliest original failed-run evidence is unchanged")
    with zipfile.ZipFile(directory / "TRANSPORT_ORIGIN/opened-original.zip") as archive:
        need((directory / "PRIOR/AUTHORIZATION.json").read_bytes() == archive.read("AUTHORIZATION.json"),
             "retained prior authorization equals the exact original opened-marker member")
    prior = frozen.require_authorization(directory / "PRIOR")
    need((directory / "HISTORY.json").read_bytes() == (directory / "PRIOR/HISTORY.json").read_bytes(),
         "unchanged complete prior exposure, including all UNKNOWN original PCMs")
    live = json.loads((directory / "ORIGINAL_RUN_LIVE.json").read_bytes())
    original_origin.validate_remote(live["run"], live["jobs"], live["artifacts"])
    failed_live = json.loads((directory / "FAILED_CONTINUATION_LIVE.json").read_bytes())
    origin.validate_remote(failed_live["run"], failed_live["jobs"], failed_live["artifacts"])
    permission = json.loads((directory / "AUTHORIZATION.json").read_bytes())
    identity = validate_identity(permission["source_execution"])
    need(permission == dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, ancestry_sha256=ANCESTRY_SHA,
        source_execution=identity, transport_origin=zero, B20_original_source=SOURCE_PIN,
        history_sha256=qa.sha(directory / "HISTORY.json"), original_run_live_sha256=qa.sha(directory / "ORIGINAL_RUN_LIVE.json"),
        failed_continuation_live_sha256=qa.sha(directory / "FAILED_CONTINUATION_LIVE.json"),
        original_authorization_sha256=qa.sha(directory / "PRIOR/AUTHORIZATION.json"), all_prior_replication_sealed=True,
        new_KWS_forwards=0, promotion_allowed=False, training_allowed=False), "one complete audited continuation authorization")
    need(prior["source_execution"]["run_id"] == str(zero["original_run_id"]), "the original opened marker is retained, never replaced")
    return permission


def consumption(directory, corpus, prepared_parent):
    from english_v6_b21_fixed_mirror_transport import verify_transport
    identity = execution_guard(prepared_parent)
    permission = require_authorization(directory)
    need(permission["source_execution"] == identity, "same continuation before full verified corpus consumption")
    receipt = verify_transport(corpus, directory)
    one_execution(identity, False)
    value = dict(schema=SCHEMA + "-consumption", marker_name=MARKER, source_execution=identity,
        transport_origin=permission["transport_origin"], authorization_sha256=qa.sha(Path(directory) / "AUTHORIZATION.json"),
        corpus_bytes=CORPUS_BYTES, corpus_sha256=CORPUS_SHA,
        transport_receipt_sha256=qa.sha(Path(directory) / "CORPUS_TRANSPORT.json"),
        actual_tts_calls=0, actual_natural_original_decodes=0, actual_observer_calls=0, actual_kws_forwards=0)
    qa.record(Path(directory) / "CONSUMPTION.json", value)
    return value


def require_generation_permit(directory, corpus, prepared_parent):
    from english_v6_b21_fixed_mirror_transport import verify_transport
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    permission = require_authorization(directory)
    need(permission["source_execution"] == identity, "same actual continuation before the first real generation")
    verify_transport(corpus, directory)
    expected = dict(schema=SCHEMA + "-consumption", marker_name=MARKER, source_execution=identity,
        transport_origin=permission["transport_origin"], authorization_sha256=qa.sha(directory / "AUTHORIZATION.json"),
        corpus_bytes=CORPUS_BYTES, corpus_sha256=CORPUS_SHA,
        transport_receipt_sha256=qa.sha(directory / "CORPUS_TRANSPORT.json"),
        actual_tts_calls=0, actual_natural_original_decodes=0, actual_observer_calls=0, actual_kws_forwards=0)
    raw = (directory / "CONSUMPTION.json").read_bytes()
    need(json.loads(raw) == expected, "exact completed corpus transport before any source call")
    require_saved_consumption(directory)
    marker = one_execution(identity, True)
    marker_raw = gh("actions/artifacts/" + str(marker["id"]) + "/zip")
    need(len(marker_raw) == marker["size_in_bytes"] and digest(marker_raw) == marker["digest"].removeprefix("sha256:"),
         "actual uploaded independent consumption marker")
    with zipfile.ZipFile(io.BytesIO(marker_raw)) as archive:
        need(archive.namelist() == ["CONSUMPTION.json"] and archive.read("CONSUMPTION.json") == raw,
             "generation begins only after the exact marker is durably uploaded")
    return expected


def require_saved_consumption(directory):
    from english_v6_b21_fixed_mirror_transport import read_receipt, PLAN
    directory = Path(directory)
    permission = require_authorization(directory)
    identity = permission["source_execution"]
    receipt = read_receipt(directory)
    started = json.loads((directory / "TRANSPORT/STARTED.json").read_bytes())
    from english_v6_b21_fixed_mirror_transport import ranges
    need(started == dict(source_execution=identity, transport_plan_sha256=digest(PLAN.read_bytes()),
        transport_origin=permission["transport_origin"], ranges=ranges(CORPUS_BYTES, 8),
        range_retries=0, source_calls=0, model_calls=0), "all transport requests belong to this authorized continuation")
    value = json.loads((directory / "CONSUMPTION.json").read_bytes())
    need(value == dict(schema=SCHEMA + "-consumption", marker_name=MARKER, source_execution=identity,
        transport_origin=permission["transport_origin"], authorization_sha256=qa.sha(directory / "AUTHORIZATION.json"),
        corpus_bytes=CORPUS_BYTES, corpus_sha256=CORPUS_SHA,
        transport_receipt_sha256=qa.sha(directory / "CORPUS_TRANSPORT.json"),
        actual_tts_calls=0, actual_natural_original_decodes=0, actual_observer_calls=0, actual_kws_forwards=0),
        "durable exact corpus verification and before-generation consumption receipt")
    return value


def retain_proof(directory, output):
    directory, output = Path(directory), Path(output)
    require_authorization(directory)
    for name in ("AUTHORIZATION.json", "HISTORY.json", "ORIGINAL_RUN_LIVE.json", "FAILED_CONTINUATION_LIVE.json", "CONSUMPTION.json", "CORPUS_TRANSPORT.json"):
        shutil.copyfile(directory / name, output / name)
    for name in ("PRIOR", "TRANSPORT_ORIGIN", "FIXED_MIRROR_ORIGIN", "TRANSPORT"):
        shutil.copytree(directory / name, output / name)
