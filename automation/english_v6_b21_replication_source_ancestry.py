"""B21 replication opening bound to complete original primary PASS evidence.

Frozen primary scripts stay unchanged. This module authenticates historical
bytes only; it performs no KWS, TTS, ASR or optimizer operation.
"""
from __future__ import annotations
import base64
import io
import zipfile
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import english_v6_b21_source_ancestry as previous
import english_v6_b21_natural_amendment_source_ancestry as amended_primary

qa, need = previous.qa, previous.need
PLANS, ASSESSMENT, ASSESSMENT_SHA = previous.PLANS, previous.ASSESSMENT, previous.ASSESSMENT_SHA
ANCESTRY_SHA = previous.ANCESTRY_SHA
REPO = previous.REPO
BRANCH = "refs/heads/research/en-v6-b21-replication-sources-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-replication-sources-once.yml"
SCHEMA = "english-v6-b21-replication-source-history-v1"
OPENING = Path(__file__).with_name("english_v6_b21_replication_opening.json")
READER = Path(__file__).with_name("english_v6_b21_fixed_mirror_quality_readback.py")
DESIGN_SHA = "76a1fd1353de00f712023a2b20256a8e622557551823cacd8b808d3f0549b0a1"
PAIR = dict(id=14900878022, run_id=45352494611, run_head="eb386f2b8e489bbd8e69c6a7972708dd667aa685",
    bytes=1801215, sha256="35fa747026589d3db83a9b24508a3d480d88392080b5c07a910c109b5912ba81")
MODEL_PT = dict(B0="1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7",
    B9="7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655",
    control="5f8863d4ef3f10fd35edd8278ac9c01bd48138184e18e08bff9d6f44b4699b05",
    challenger="4947d017142f3b01f5dd4cf7706cc3c6978cf1a072ca022c26ea847ec0dc2f04")


def digest(raw): return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


MARKER = "en-v6-b21-replication-panel-opened"
PROVIDERS = ("qwen", "whisper", "parakeet", "xlsr", "lv60")
SOURCE_BUDGET = dict(maximum_tts_calls=96, maximum_natural_decodes=24,
    maximum_native_observer_calls=600, new_kws_forwards=0, retry_count=0)


def registered():
    return amended_primary.registered()


def read_plan():
    return amended_primary.read_plan()


def amendment_binding():
    value = registered()
    return dict(rule_sha256=value["rule_sha256"], descriptor_sha256=value["descriptor_sha256"],
        catalog_sha256=value["catalog_sha256"], metadata_artifact_pin=value["metadata_artifact_pin"],
        metadata_producer_pin=value["metadata_producer_pin"], metadata_consumed_pin=value["metadata_consumed_pin"],
        metadata_execution=value["metadata_execution"],
        primary_roster_sha256=digest(canonical(value["primary_roster"])),
        replication_roster_sha256=digest(canonical(value["replication_roster"])),
        original_failed_primary12_excluded=True, original_sealed_replication12_excluded=True,
        exact_replication_slots=24, original_replication_synthetic_slots=96,
        original_primary_pass_required=True, training_allowed=False, promotion_allowed=False)


def gh(endpoint):
    return subprocess.run(["gh", "api", "repos/" + REPO + "/" + endpoint,
        "--allow-escape-sequences"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def read_opening(require_bound=True):
    value = json.loads(OPENING.read_bytes())
    need(value["schema"] == "english-v6-b21-replication-original-primary-opening-v1"
         and value["panel"] == "replication" and value["assessment_sha256"] == ASSESSMENT_SHA
         and value["training_design_sha256"] == DESIGN_SHA and value["pair_pin"] == PAIR
         and value["model_pt_sha256"] == MODEL_PT
         and value["source_schema"] == "english-v6-b21-fresh-replication-source-v1"
         and value["screen_schema"] == "english-v6-b21-120-original-replication-screen-v1"
         and value["natural_panel_amendment"] is True
         and value["natural_amendment"] == amendment_binding()
         and value["original_primary_pass_required"] is True
         and value["training_allowed"] is value["promotion_allowed"] is False,
         "fixed replication protocol, original trained pair and unchanged model roles")
    if not require_bound:
        need((value["status"] == "SEALED_UNTIL_ORIGINAL_PRIMARY_PASS"
              and value["quality_readback_sha256"] is value["primary_quality_lock"] is None)
             or value["status"] == "BOUND_ORIGINAL_PRIMARY_PASS",
             "explicit inert registration or separately bound original PASS")
        if value["status"] == "BOUND_ORIGINAL_PRIMARY_PASS":
            return read_opening(require_bound=True)
        return value
    lock = value["primary_quality_lock"]
    need(value["status"] == "BOUND_ORIGINAL_PRIMARY_PASS" and type(lock) is dict
         and re.fullmatch(r"[0-9a-f]{64}", value["quality_readback_sha256"] or "") is not None,
         "STOP: replication remains sealed until original primary PASS is byte-bound")
    need(set(lock) == {"schema", "panel", "result", "producer", "consumed", "evaluator_sha256",
                      "inputs_sha256", "producer_manifest_sha256", "original_result_sha256"}
         and lock["schema"] == "english-v6-b21-original-quality-lock-v1" and lock["panel"] == "primary"
         and all(re.fullmatch(r"[0-9a-f]{64}", lock[k] or "") is not None for k in
                 ("evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256")),
         "exact independently reviewed original primary evidence lock")
    pins = [lock[k] for k in ("result", "producer", "consumed")]
    for pin in pins:
        need(set(pin) == {"id", "run_id", "run_head", "bytes", "sha256"}
             and all(type(pin[k]) is int and pin[k] > 0 for k in ("id", "run_id", "bytes"))
             and re.fullmatch(r"[0-9a-f]{40}", pin["run_head"] or "") is not None
             and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"] or "") is not None,
             "complete actual original result/producer/consumed artifact pins")
    need(len({p["id"] for p in pins}) == 3 and len({(p["run_id"], p["run_head"]) for p in pins}) == 1,
         "three distinct artifacts from one original primary execution")
    need(READER.is_file() and digest(READER.read_bytes()) == value["quality_readback_sha256"],
         "exact reviewed saved-quality reader before historical or new source access")
    return value


def opening_sha256():
    read_opening()
    return digest(OPENING.read_bytes())


def quality_reader():
    opening = read_opening()
    import english_v6_b21_fixed_mirror_quality_readback as reader
    need(Path(reader.__file__).resolve() == READER.resolve()
         and digest(Path(reader.__file__).read_bytes()) == opening["quality_readback_sha256"],
         "executing saved-quality reader is the exact registered implementation")
    return reader


def validate_primary(summary):
    need(summary["panel_gate"] == "PASS" and summary["model_pt_sha256"] == MODEL_PT
         and summary["assessment_sha256"] == ASSESSMENT_SHA
         and summary["training_design_sha256"] == DESIGN_SHA and summary["pair_pin"] == PAIR
         and summary["original_result_sha256"] == read_opening()["primary_quality_lock"]["original_result_sha256"],
         "actual primary PASS of this exact frozen pair and prospective assessment")
    identity = summary["original_identity"]
    pin = read_opening()["primary_quality_lock"]["result"]
    need(identity["run_id"] == str(pin["run_id"]) and identity["run_head"] == pin["run_head"]
         and identity["run_attempt"] == 1
         and re.fullmatch(r"[0-9a-f]{40}", identity["prepared_parent"] or "") is not None
         and identity["prepared_parent"] != identity["run_head"], "same original completed primary identity")
    inv = summary["source_inventory"]
    need(set(inv) == {"pcm_sha256", "source_group_id", "canonical_people", "original_count"}
         and inv["original_count"] == 120 and len(inv["pcm_sha256"]) == 120
         and len(inv["source_group_id"]) == len(inv["canonical_people"]) == 36,
         "all120 actual primary raw originals, including unknowns and quality rejects")
    for k in ("pcm_sha256", "source_group_id", "canonical_people"):
        need(inv[k] == sorted(set(inv[k])) and all(type(v) is str and v for v in inv[k]),
             "unique primary raw source inventory " + k)
    need(all(re.fullmatch(r"[0-9a-f]{64}", h) is not None for h in inv["pcm_sha256"]), "actual raw primary PCM hashes")
    expected_primary = {person.removeprefix("p") for person in read_plan()["primary"]["synthetic"]["speaker_id_map"].values()}
    expected_primary.update(registered()["primary_design"]["speakers"])
    need(set(inv["canonical_people"]) == expected_primary
         and not expected_primary & set(registered()["replication_design"]["speakers"]),
         "actual amended-primary36 people and independently frozen new replica12")
    return summary


def make_history(summary):
    validate_primary(summary)
    old, inv = previous.read_ancestry(), summary["source_inventory"]
    mapping = {"all_pcm_sha256": "pcm_sha256", "all_source_groups": "source_group_id", "canonical_people": "canonical_people"}
    for old_key, new_key in mapping.items():
        need(not set(old[old_key]) & set(inv[new_key]), "original primary is independent of full historical exposure")
    combined = {k: sorted(set(old[k]) | set(inv[v])) for k, v in mapping.items()}
    return dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA, previous_ancestry_sha256=ANCESTRY_SHA,
        opening_sha256=opening_sha256(), primary_quality_lock_sha256=digest(canonical(read_opening()["primary_quality_lock"])),
        primary_quality=summary, **combined, all_prior_replication_sealed=True, b21_replication_opened=True,
        training_allowed=False, new_KWS_forwards=0, promotion_allowed=False)


def validate_history(value):
    need(value == make_history(value["primary_quality"]), "complete old history plus every actual primary original")
    return tuple(set(value[k]) for k in ("all_pcm_sha256", "all_source_groups", "canonical_people"))


def validate_identity(identity):
    need(set(identity) == {"run_id", "run_head", "run_attempt", "run_branch", "prepared_parent"}
         and re.fullmatch(r"[1-9][0-9]*", identity["run_id"] or "") is not None
         and all(re.fullmatch(r"[0-9a-f]{40}", identity[k] or "") is not None for k in ("run_head", "prepared_parent"))
         and identity["run_head"] != identity["prepared_parent"] and identity["run_attempt"] == 1
         and identity["run_branch"] == BRANCH, "actual first public B21 replication source identity")
    return identity


def execution_guard(prepared_parent, env=None, event=None, parent=None):
    read_plan(); read_opening()
    e = os.environ if env is None else env
    need(all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
         "GITHUB_REF": BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1"}.items()),
         "first separately registered public B21 replication source event only")
    identity = validate_identity(dict(run_id=e.get("GITHUB_RUN_ID", ""), run_head=e.get("GITHUB_SHA", ""),
        run_attempt=1, run_branch=BRANCH, prepared_parent=prepared_parent))
    if event is None: event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None: parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == identity["run_head"]
         and event.get("ref") == BRANCH and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event["repository"].get("private") is False,
         "exact reviewed predecessor, no retry, private execution or forced/multiple-commit push")
    return identity


def authorize(out, prepared_parent):
    identity = execution_guard(prepared_parent)
    opening = read_opening()
    out = Path(out)
    need(out.is_absolute() and out.parent.is_dir() and not out.exists(), "one fresh original-PASS authorization destination")
    out.mkdir()
    reader = quality_reader()
    reader.fetch_quality_original(out / "PRIMARY_QUALITY", opening["primary_quality_lock"])
    summary = validate_primary(reader.verify_quality_original(out / "PRIMARY_QUALITY", opening["primary_quality_lock"]))
    one_execution(identity, False)
    metadata_run = registered()["metadata_execution"]["run_id"]
    live = amended_primary.validate_metadata_live(dict(run=json.loads(gh("actions/runs/" + metadata_run)),
        artifacts=json.loads(gh("actions/runs/" + metadata_run + "/artifacts?per_page=100"))))
    qa.record(out / "METADATA_LIVE.json", live)
    history = make_history(summary)
    qa.record(out / "HISTORY.json", history)
    (out / "OPENING.json").write_bytes(OPENING.read_bytes())
    qa.record(out / "AUTHORIZATION.json", dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA,
        opening_sha256=opening_sha256(), quality_readback_sha256=opening["quality_readback_sha256"],
        primary_quality_lock_sha256=digest(canonical(opening["primary_quality_lock"])),
        source_execution=identity, primary_quality=summary, history_sha256=qa.sha(out / "HISTORY.json"),
        natural_amendment=amendment_binding(), metadata_live_sha256=qa.sha(out / "METADATA_LIVE.json"),
        marker_name=MARKER, status="REPLICATION_SOURCE_BUDGET_CONSUMED_UNKNOWN", source_budget=SOURCE_BUDGET,
        all_prior_replication_sealed=True, b21_replication_opened=True,
        new_KWS_forwards=0, training_allowed=False, promotion_allowed=False))
    return require_authorization(out)


def require_authorization(directory):
    opening = read_opening()
    directory = Path(directory)
    need((directory / "OPENING.json").read_bytes() == OPENING.read_bytes(), "original bound opening bytes retained unchanged")
    permission = json.loads((directory / "AUTHORIZATION.json").read_bytes())
    identity = validate_identity(permission["source_execution"])
    summary = validate_primary(quality_reader().verify_quality_original(directory / "PRIMARY_QUALITY", opening["primary_quality_lock"]))
    history = json.loads((directory / "HISTORY.json").read_bytes())
    need(history == make_history(summary), "original quality proof binds the complete primary source exclusion history")
    amended_primary.validate_metadata_live(json.loads((directory / "METADATA_LIVE.json").read_bytes()))
    need(permission == dict(schema=SCHEMA, assessment_sha256=ASSESSMENT_SHA,
        opening_sha256=opening_sha256(), quality_readback_sha256=opening["quality_readback_sha256"],
        primary_quality_lock_sha256=digest(canonical(opening["primary_quality_lock"])),
        source_execution=identity, primary_quality=summary, history_sha256=qa.sha(directory / "HISTORY.json"),
        natural_amendment=amendment_binding(), metadata_live_sha256=qa.sha(directory / "METADATA_LIVE.json"),
        marker_name=MARKER, status="REPLICATION_SOURCE_BUDGET_CONSUMED_UNKNOWN", source_budget=SOURCE_BUDGET,
        all_prior_replication_sealed=True, b21_replication_opened=True,
        new_KWS_forwards=0, training_allowed=False, promotion_allowed=False), "unchanged original-primary PASS authorization")
    return permission


def retain_proof(directory, output):
    directory, output = Path(directory), Path(output)
    require_saved_generation(directory)
    for name in ("AUTHORIZATION.json", "HISTORY.json", "OPENING.json", "METADATA_LIVE.json",
                 "OPENING_UPLOAD.json", "GENERATION_STARTED.json", "CORPUS_TRANSPORT.json"):
        shutil.copyfile(directory / name, output / name)
    shutil.copytree(directory / "PRIMARY_QUALITY", output / "PRIMARY_QUALITY")
    shutil.copytree(directory / "TRANSPORT", output / "TRANSPORT")


def marker_name(provider=None):
    need(provider is None or provider in PROVIDERS, "registered native provider only")
    return MARKER if provider is None else "english-v6-b21-replication-observer-" + provider + "-consumed"


def one_execution(identity, marker_present, provider=None):
    branch = BRANCH.removeprefix("refs/heads/")
    runs = json.loads(gh("actions/workflows/" + Path(WORKFLOW).name + "/runs?branch=" + branch + "&per_page=100"))
    need(runs["total_count"] == len(runs["workflow_runs"]) == 1
         and str(runs["workflow_runs"][0]["id"]) == identity["run_id"]
         and runs["workflow_runs"][0]["head_sha"] == identity["run_head"]
         and runs["workflow_runs"][0]["run_attempt"] == 1,
         "one replication execution, including all failed or expired runs")
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


def observer_document(provider, registration):
    need(provider in PROVIDERS, "registered provider")
    identity = validate_identity({key: registration[key] for key in ("run_id", "run_head", "run_attempt", "run_branch", "prepared_parent")})
    need(96 <= registration["source_rows"] == registration["max_calls"] <= 120, "all and only eligible originals")
    return dict(schema=SCHEMA + "-observer-consumption", marker_name=marker_name(provider), provider=provider,
        source_execution=identity, source_manifest_sha256=registration["source_manifest_sha256"],
        registration=registration, status="OBSERVER_BUDGET_CONSUMED_UNKNOWN", maximum_native_calls=registration["source_rows"],
        new_tts_calls=0, new_natural_decodes=0, new_kws_forwards=0, retry_count=0)


def consume_observer(provider, source, out, prepared_parent):
    import english_v6_b21_replication_source as owner
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


def require_uploaded_opening(directory, prepared_parent):
    identity = execution_guard(prepared_parent)
    permission = require_authorization(directory)
    need(permission["source_execution"] == identity, "same actual original-PASS replication execution")
    artifact = one_execution(identity, True)
    value = uploaded_marker(artifact, permission, "AUTHORIZATION.json")
    path = Path(directory) / "OPENING_UPLOAD.json"
    if path.exists():
        need(json.loads(path.read_bytes()) == value, "identical durable original source budget")
    else:
        qa.record(path, value)
    return permission


def require_saved_opening(directory):
    directory = Path(directory)
    permission = require_authorization(directory)
    validate_uploaded_marker(json.loads((directory / "OPENING_UPLOAD.json").read_bytes()),
                             permission, "AUTHORIZATION.json")
    return permission


def require_generation_permit(directory, prepared_parent):
    directory = Path(directory)
    permission = require_uploaded_opening(directory, prepared_parent)
    path = directory / "GENERATION_STARTED.json"
    need(not path.exists(), "only one source generation entry; never retry into another output")
    qa.record(path, dict(source_execution=permission["source_execution"],
        authorization_sha256=qa.sha(directory / "AUTHORIZATION.json"),
        natural_amendment=amendment_binding(), source_budget=SOURCE_BUDGET,
        status="REPLICATION_SOURCE_BUDGET_CONSUMED_UNKNOWN"))
    return permission


def require_saved_generation(directory):
    directory = Path(directory)
    permission = require_saved_opening(directory)
    need(json.loads((directory / "GENERATION_STARTED.json").read_bytes()) ==
         dict(source_execution=permission["source_execution"],
              authorization_sha256=qa.sha(directory / "AUTHORIZATION.json"),
              natural_amendment=amendment_binding(), source_budget=SOURCE_BUDGET,
              status="REPLICATION_SOURCE_BUDGET_CONSUMED_UNKNOWN"),
         "same exact source budget and generation entry retained in source artifact")
    return permission
