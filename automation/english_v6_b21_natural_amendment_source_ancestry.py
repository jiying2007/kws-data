"""Joint provenance for reused96 originals and a separately registered natural24.

The old failed execution stays STOP. New authorization never grants TTS calls;
one natural decode budget and five observer budgets are consumed independently.
"""
from __future__ import annotations
import base64
import copy
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import english_v6_b21_source_ancestry as frozen
import english_v6_b21_fixed_mirror_source_ancestry as fixed
import english_v6_b21_natural_amendment as selection
import english_v6_b21_reuse_synthetic as reuse
import english_v6_b21_natural_amendment_auth_origin as auth_origin

qa, need, digest, canonical = frozen.qa, frozen.need, frozen.digest, frozen.canonical
PLANS, ASSESSMENT, ASSESSMENT_SHA = frozen.PLANS, frozen.ASSESSMENT, frozen.ASSESSMENT_SHA
ANCESTRY_SHA, REPO, SOURCE_PIN = frozen.ANCESTRY_SHA, frozen.REPO, frozen.SOURCE_PIN
read_ancestry = frozen.read_ancestry
origin = fixed.origin
CORPUS_BYTES, CORPUS_SHA = fixed.CORPUS_BYTES, fixed.CORPUS_SHA
BRANCH = "refs/heads/research/en-v6-b21-natural-amendment-sources-auth-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-natural-amendment-sources-auth-once.yml"
SCHEMA = "english-v6-b21-reused96-natural24-amendment-authorization-v1"
MARKER = "english-v6-b21-natural-amendment-auth-source-consumed"
PROVIDERS = ("qwen", "whisper", "parakeet", "xlsr", "lv60")
REUSED_SOURCE_PIN = dict(id=14907672424, run_id=45367984097,
    run_head="c262e4b5cf06de8a6950c2c797c7aefcf43498d8", bytes=5726331,
    sha256="e7a9cb7c22c701222806ec53cdd241b8d1fdd9e2d5ec38fea03dc46d90980bfd")
REUSED_MANIFEST_SHA = "d09c3c7032d199a5d4a953545106e9242d87c5da37ddcc366bb676f0f20af0f8"


def read_plan():
    value = copy.deepcopy(frozen.read_plan())
    registration = selection.verify_registration()
    if registration["bound"]:
        for panel in ("primary", "replication"):
            value[panel]["natural"] = registration[panel + "_design"]
    return value


def registered():
    value = selection.verify_registration()
    need(value["bound"] is True and value["primary_roster"] is not None
         and value["replication_roster"] is not None, "committed exact metadata-qualified rosters before source access")
    return value


def amendment_binding():
    value = registered()
    return dict(rule_sha256=value["rule_sha256"],
        inputs_sha256=qa.sha(Path(selection.__file__).with_name("english_v6_b21_natural_amendment_inputs.json")),
        descriptor_sha256=value["descriptor_sha256"],
        catalog_sha256=value["catalog_sha256"], metadata_artifact_pin=value["metadata_artifact_pin"],
        metadata_producer_pin=value["metadata_producer_pin"],
        metadata_consumed_pin=value["metadata_consumed_pin"], metadata_execution=value["metadata_execution"],
        primary_roster_sha256=digest(canonical(value["primary_roster"])),
        replication_roster_sha256=digest(canonical(value["replication_roster"])),
        reused_source_pin=REUSED_SOURCE_PIN, reused_synthetic_manifest_sha256=REUSED_MANIFEST_SHA,
        original_failed_source_status="STOP", retained_original_synthetic_sources=96,
        original_public_tts_calls=96, new_public_tts_calls=0, maximum_new_natural_decodes=24,
        maximum_new_observer_calls=600, primary_natural_slots=24,
        replication_audio_opened=False, training_allowed=False, promotion_allowed=False,
        authorization_continuation=True, authorization_origin=auth_origin.verify())


def validate_identity(identity):
    need(set(identity) == {"run_id", "run_head", "run_attempt", "run_branch", "prepared_parent"}
         and re.fullmatch(r"[1-9][0-9]*", identity["run_id"] or "") is not None
         and all(re.fullmatch(r"[0-9a-f]{40}", identity[k] or "") is not None for k in ("run_head", "prepared_parent"))
         and identity["run_head"] != identity["prepared_parent"] and identity["run_attempt"] == 1
         and identity["run_branch"] == BRANCH, "one first public natural amendment execution")
    return identity


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    read_plan()
    registered()
    origin.verify()
    auth_origin.verify()
    auth_origin.verify_current_producer()
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_REF": BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1"}.items()),
         "first separately registered public amendment, never replay the source STOP")
    identity = validate_identity(dict(run_id=e.get("GITHUB_RUN_ID", ""), run_head=e.get("GITHUB_SHA", ""),
        run_attempt=1, run_branch=BRANCH, prepared_parent=prepared_parent))
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == identity["run_head"]
         and event.get("ref") == BRANCH and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event["repository"].get("private") is False,
         "reviewed exact predecessor, public single activation push and no retry")
    return identity


def gh(endpoint):
    path = endpoint if endpoint.startswith("repos/" + REPO + "/") else "repos/" + REPO + "/" + endpoint
    return subprocess.run(["gh", "api", path, "--allow-escape-sequences"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS, "registered native provider only")
    return MARKER if provider is None else "english-v6-b21-natural-amendment-auth-observer-" + provider + "-consumed"


def prior_budget_names():
    return ("english-v6-b21-natural-amendment-source-consumed", *(
        "english-v6-b21-natural-amendment-observer-" + provider + "-consumed" for provider in PROVIDERS))


def validate_prior_unconsumed(document):
    need(set(document) == set(prior_budget_names())
         and all(value["total_count"] == 0 and value["artifacts"] == [] for value in document.values()),
         "every old source/provider budget is absent; a fresh name never bypasses consumption")
    return document


def one_execution(identity, marker_present, provider=None):
    branch = BRANCH.removeprefix("refs/heads/")
    runs = json.loads(gh("actions/workflows/" + Path(WORKFLOW).name + "/runs?branch=" + branch + "&per_page=100"))
    need(runs["total_count"] == len(runs["workflow_runs"]) == 1
         and str(runs["workflow_runs"][0]["id"]) == identity["run_id"]
         and runs["workflow_runs"][0]["head_sha"] == identity["run_head"]
         and runs["workflow_runs"][0]["run_attempt"] == 1,
         "one amendment execution, including all failed or expired runs")
    name = marker_name(provider)
    artifacts = json.loads(gh("actions/artifacts?name=" + name + "&per_page=100"))
    need(artifacts["total_count"] == len(artifacts["artifacts"]) == int(marker_present), "single independent consumed budget " + name)
    if not marker_present:
        return None
    artifact = artifacts["artifacts"][0]
    need(artifact["name"] == name and artifact["expired"] is False
         and str(artifact["workflow_run"]["id"]) == identity["run_id"]
         and artifact["workflow_run"]["head_sha"] == identity["run_head"], "this actual run owns the consumed budget")
    return artifact


def fetch_reuse_bundle(output):
    output = Path(output)
    need(output.is_absolute() and not output.exists() and output.parent.is_dir(), "fresh original source proof destination")
    output.mkdir()
    for name, endpoint in reuse.API_ENDPOINTS.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gh(endpoint))
    for name, pin in reuse.BUNDLE_ARTIFACTS.items():
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"], "exact original source evidence " + name)
        (output / name).write_bytes(raw)
    return reuse.verify(output)


def validate_metadata_live(live):
    registration = registered()
    identity = registration["metadata_execution"]
    run, artifacts = live["run"], live["artifacts"]
    need(str(run["id"]) == identity["run_id"] and run["head_sha"] == identity["run_head"]
         and run["run_attempt"] == identity["run_attempt"] == 1 and run["event"] == "push"
         and run["head_branch"] == identity["run_branch"].removeprefix("refs/heads/")
         and run["path"] == ".github/workflows/english-v6-b21-natural-metadata-once.yml"
         and run["status"] == "completed" and run["conclusion"] == "success"
         and all(run[key]["full_name"] == REPO and run[key]["private"] is False for key in ("repository", "head_repository")),
         "same actual first public successful metadata diagnostic at source authorization")
    pins = [registration[key] for key in ("metadata_artifact_pin", "metadata_producer_pin", "metadata_consumed_pin")]
    need(artifacts["total_count"] == len(artifacts["artifacts"]) == 3
         and {item["id"] for item in artifacts["artifacts"]} == {pin["id"] for pin in pins},
         "exact complete original metadata/producer/consumed artifact set")
    by_id = {item["id"]: item for item in artifacts["artifacts"]}
    for pin in pins:
        reuse.validate_artifact(by_id[pin["id"]], pin)
    return live


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    one_execution(identity, False)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), "fresh joint authorization")
    out.mkdir()
    failed = auth_origin.verify()
    failed_run = str(failed["failed_run_id"])
    failed_live = {name: json.loads(gh(endpoint)) for name, endpoint in {
        "run": "actions/runs/" + failed_run,
        "jobs": "actions/runs/" + failed_run + "/jobs?filter=all&per_page=100",
        "artifacts": "actions/runs/" + failed_run + "/artifacts?per_page=100"}.items()}
    auth_origin.validate_remote(failed_live["run"], failed_live["jobs"], failed_live["artifacts"])
    qa.record(out / "AUTHORIZATION_STOP_LIVE.json", failed_live)
    old_budgets = validate_prior_unconsumed({name: json.loads(gh("actions/artifacts?name=" + name + "&per_page=100"))
        for name in prior_budget_names()})
    qa.record(out / "PRIOR_BUDGETS_LIVE.json", old_budgets)
    metadata_run = registered()["metadata_execution"]["run_id"]
    metadata_live = validate_metadata_live(dict(run=json.loads(gh("actions/runs/" + metadata_run)),
        artifacts=json.loads(gh("actions/runs/" + metadata_run + "/artifacts?per_page=100"))))
    qa.record(out / "METADATA_LIVE.json", metadata_live)
    proof = fetch_reuse_bundle(out / "REUSE")
    need(proof["source_pin"] == REUSED_SOURCE_PIN and proof["synthetic_manifest_sha256"] == REUSED_MANIFEST_SHA,
         "only the audited96 originals from the preserved failed execution")
    frozen.validate_history(proof["prior_history"])
    validate_history(proof["history"])
    qa.record(out / "HISTORY.json", proof["prior_history"])
    qa.record(out / "COMPLETE_HISTORY.json", proof["history"])
    qa.record(out / "AUTHORIZATION.json", dict(schema=SCHEMA, source_execution=identity,
        assessment_sha256=ASSESSMENT_SHA, ancestry_sha256=ANCESTRY_SHA, transport_origin=origin.verify(),
        natural_amendment=amendment_binding(), original_synthetic_execution=proof["original_identity"],
        history_sha256=qa.sha(out / "HISTORY.json"), complete_history_sha256=qa.sha(out / "COMPLETE_HISTORY.json"),
        metadata_live_sha256=qa.sha(out / "METADATA_LIVE.json"),
        authorization_stop_live_sha256=qa.sha(out / "AUTHORIZATION_STOP_LIVE.json"),
        prior_budgets_live_sha256=qa.sha(out / "PRIOR_BUDGETS_LIVE.json"),
        new_tts_calls_allowed=0, maximum_natural_decodes=24, maximum_native_observer_calls=600,
        training_allowed=False, promotion_allowed=False, replication_audio_opened=False))
    return require_authorization(out)


def validate_history(value):
    reuse.validate_complete_history(value)
    return tuple(set(value[k]) for k in ("all_pcm_sha256", "all_source_groups", "canonical_people"))


def require_authorization(directory):
    directory = Path(directory)
    proof = reuse.verify(directory / "REUSE")
    need(proof["source_pin"] == REUSED_SOURCE_PIN and proof["synthetic_manifest_sha256"] == REUSED_MANIFEST_SHA,
         "original96 exact archive, manifest and authenticated failed execution")
    prior = json.loads((directory / "HISTORY.json").read_bytes())
    complete = json.loads((directory / "COMPLETE_HISTORY.json").read_bytes())
    need(prior == proof["prior_history"] and complete == proof["history"], "baseline plus separate complete96 exposure history")
    frozen.validate_history(prior)
    validate_history(complete)
    validate_metadata_live(json.loads((directory / "METADATA_LIVE.json").read_bytes()))
    failed_live = json.loads((directory / "AUTHORIZATION_STOP_LIVE.json").read_bytes())
    auth_origin.validate_remote(failed_live["run"], failed_live["jobs"], failed_live["artifacts"])
    validate_prior_unconsumed(json.loads((directory / "PRIOR_BUDGETS_LIVE.json").read_bytes()))
    value = json.loads((directory / "AUTHORIZATION.json").read_bytes())
    identity = validate_identity(value["source_execution"])
    need(value == dict(schema=SCHEMA, source_execution=identity, assessment_sha256=ASSESSMENT_SHA,
        ancestry_sha256=ANCESTRY_SHA, transport_origin=origin.verify(), natural_amendment=amendment_binding(),
        original_synthetic_execution=proof["original_identity"], history_sha256=qa.sha(directory / "HISTORY.json"),
        complete_history_sha256=qa.sha(directory / "COMPLETE_HISTORY.json"),
        metadata_live_sha256=qa.sha(directory / "METADATA_LIVE.json"),
        authorization_stop_live_sha256=qa.sha(directory / "AUTHORIZATION_STOP_LIVE.json"),
        prior_budgets_live_sha256=qa.sha(directory / "PRIOR_BUDGETS_LIVE.json"), new_tts_calls_allowed=0,
        maximum_natural_decodes=24, maximum_native_observer_calls=600, training_allowed=False,
        promotion_allowed=False, replication_audio_opened=False), "exact joint provenance, no synthetic regeneration grant")
    return value


def consumption_document(directory):
    from english_v6_b21_natural_amendment_transport import read_receipt
    directory = Path(directory)
    permission = require_authorization(directory)
    read_receipt(directory)
    return dict(schema=SCHEMA + "-consumption", marker_name=MARKER, source_execution=permission["source_execution"],
        authorization_sha256=qa.sha(directory / "AUTHORIZATION.json"), natural_amendment=permission["natural_amendment"],
        corpus_bytes=CORPUS_BYTES, corpus_sha256=CORPUS_SHA,
        transport_receipt_sha256=qa.sha(directory / "CORPUS_TRANSPORT.json"),
        status="NATURAL24_OBSERVER600_CONSUMED_UNKNOWN", maximum_new_tts_calls=0,
        maximum_new_natural_decodes=24, maximum_new_observer_calls=600, new_kws_forwards=0)


def consumption(directory, corpus, prepared_parent):
    from english_v6_b21_natural_amendment_transport import verify_transport
    identity = execution_guard(prepared_parent)
    permission = require_authorization(directory)
    need(permission["source_execution"] == identity, "same amendment before consuming natural24")
    verify_transport(corpus, directory)
    one_execution(identity, False)
    value = consumption_document(directory)
    path = Path(directory) / "CONSUMPTION.json"
    need(not path.exists(), "one natural decode budget")
    qa.record(path, value)
    return value


def uploaded_marker(artifact, expected, member):
    raw = gh("actions/artifacts/" + str(artifact["id"]) + "/zip")
    result = dict(artifact=artifact, original_zip_base64=base64.b64encode(raw).decode(), receipt=expected)
    validate_uploaded_marker(result, expected, member)
    return result


def validate_uploaded_marker(value, expected, member):
    artifact = value["artifact"]
    raw = base64.b64decode(value["original_zip_base64"], validate=True)
    identity = expected["source_execution"]
    need(artifact["id"] > 0 and artifact["name"] == expected["marker_name"] and artifact["expired"] is False
         and str(artifact["workflow_run"]["id"]) == identity["run_id"]
         and artifact["workflow_run"]["head_sha"] == identity["run_head"]
         and len(raw) == artifact["size_in_bytes"] and "sha256:" + digest(raw) == artifact["digest"]
         and value["receipt"] == expected, "exact original durably uploaded budget ZIP")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        need(archive.namelist() == [member] and archive.testzip() is None
             and json.loads(archive.read(member)) == expected, "uploaded budget contains exactly the expected receipt")
    return expected


def require_natural_permit(directory, corpus, prepared_parent):
    from english_v6_b21_natural_amendment_transport import verify_transport
    directory = Path(directory)
    identity = execution_guard(prepared_parent)
    expected = consumption_document(directory)
    need(expected["source_execution"] == identity
         and json.loads((directory / "CONSUMPTION.json").read_bytes()) == expected, "same exact natural24 budget")
    verify_transport(corpus, directory)
    marker = one_execution(identity, True)
    receipt = uploaded_marker(marker, expected, "CONSUMPTION.json")
    need(not (directory / "CONSUMPTION_UPLOAD.json").exists(), "one natural decode entry, no retry")
    qa.record(directory / "CONSUMPTION_UPLOAD.json", receipt)
    return expected


def require_saved_consumption(directory):
    directory = Path(directory)
    expected = consumption_document(directory)
    need(json.loads((directory / "CONSUMPTION.json").read_bytes()) == expected, "retained natural24 budget")
    validate_uploaded_marker(json.loads((directory / "CONSUMPTION_UPLOAD.json").read_bytes()), expected, "CONSUMPTION.json")
    return expected


def observer_document(provider, registration):
    need(provider in PROVIDERS, "registered provider")
    identity = validate_identity({key: registration[key] for key in ("run_id", "run_head", "run_attempt", "run_branch", "prepared_parent")})
    need(96 <= registration["source_rows"] == registration["max_calls"] <= 120, "all and only eligible originals")
    return dict(schema=SCHEMA + "-observer-consumption", marker_name=marker_name(provider), provider=provider,
        source_execution=identity, source_manifest_sha256=registration["source_manifest_sha256"],
        registration=registration, status="OBSERVER_BUDGET_CONSUMED_UNKNOWN", maximum_native_calls=registration["source_rows"],
        new_tts_calls=0, new_natural_decodes=0, new_kws_forwards=0, retry_count=0)


def consume_observer(provider, source, out, prepared_parent):
    import english_v6_b21_natural_amendment_source as owner
    identity = execution_guard(prepared_parent)
    registration = owner.registration(source)
    need(all(registration[k] == value for k, value in identity.items()), "same new source before observer consumption")
    one_execution(identity, True)
    one_execution(identity, False, provider)
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), "fresh individual observer budget")
    out.mkdir()
    value = observer_document(provider, registration)
    qa.record(out / "OBSERVER_CONSUMPTION.json", value)
    return value


def require_observer_permit(provider, registration):
    identity = execution_guard(registration["prepared_parent"])
    need(all(registration[k] == value for k, value in identity.items()), "same source observer identity")
    one_execution(identity, True)
    marker = one_execution(identity, True, provider)
    return uploaded_marker(marker, observer_document(provider, registration), "OBSERVER_CONSUMPTION.json")


def validate_observer_budget(value, provider, registration):
    return validate_uploaded_marker(value, observer_document(provider, registration), "OBSERVER_CONSUMPTION.json")


def retain_proof(directory, output):
    directory, output = Path(directory), Path(output)
    require_saved_consumption(directory)
    for name in ("AUTHORIZATION.json", "HISTORY.json", "COMPLETE_HISTORY.json", "CONSUMPTION.json",
                 "CONSUMPTION_UPLOAD.json", "CORPUS_TRANSPORT.json", "METADATA_LIVE.json",
                 "AUTHORIZATION_STOP_LIVE.json", "PRIOR_BUDGETS_LIVE.json"):
        need(not (output / name).exists(), "preserve original synthetic members")
        shutil.copyfile(directory / name, output / name)
    for name in ("REUSE", "TRANSPORT"):
        shutil.copytree(directory / name, output / name)
