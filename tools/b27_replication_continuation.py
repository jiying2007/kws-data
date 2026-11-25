"""Read-only B27 replica continuation, inert until its exact run is reviewed.

NO_PROMOTION invokes the frozen replica quality reader once at top level.
PASS invokes only the frozen engineering input binder once; all its original
PRIMARY, replica, recursive TRAIN and original-pair verifications remain.
No source, observer, model, optimizer, compiler, pack or marker is executed.
The 8 MB export bound is observation metadata only, never a scientific cap.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_suffix(".json")
sys.path.insert(0, str(ROOT / "automation"))
import bind_b27_engineering as engineering_binder
import english_v6_b27_original_proof as original_proof
import english_v6_b27_replication_eval as evaluator
import english_v6_b27_profile as profile
import english_v6_b27_quality_readback as quality

REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b27-replication-continuation-ci-once-20261125"
WORKFLOW = ".github/workflows/b27-replication-continuation.yml"
PARENT_TOKEN = "__B27_REVIEWED_REPLICATION_CONTINUATION_PARENT__"
CONFIG_SCHEMA = "english-v6-b27-replication-continuation-config-v1"
HASH_FILES = {
    "evaluator_sha256": "automation/english_v6_b27_replication_eval.py",
    "inputs_sha256": "automation/english_v6_b27_replication_inputs.json",
    "producer_manifest_sha256": "automation/english_v6_b27_replication_producer.json",
    "external_proof_binding_sha256": "automation/english_v6_b27_replication_proof_binding.json",
}
CONFIG_PINS = ("replication_run_id", "replication_run_head", "replication_prepared_parent", *HASH_FILES)
FIXED_FILES = {
    "automation/bind_b27_engineering.py":
        "a3ee15f0a9433f9b8fece207dfdc50b3044b3ed714fe71597514ae946e87d886",
    "automation/english_v6_b27_engineering.py":
        "e9d1c59742c50e175650d94be78f8d4a7f697e3f613c94b3ff25c77cf9b10a2c",
    "automation/english_v6_b27_engineering_readback.py":
        "98848744d966f35855c3408adec7e67601917009b565c6c8a978fd914ec2de79",
    "automation/english_v6_b27_quality_readback.py":
        "87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa",
    "automation/english_v6_b27_engineering_inputs.json":
        "5983a7bf1f2686f2c5a3e29276be00703200f973a440569678ba3bdf3b0e26a5",
}
ENGINEERING_SPEC = "automation/english_v6_b27_engineering_inputs.json"
QUALITY_SHA = FIXED_FILES["automation/english_v6_b27_quality_readback.py"]
TRAIN_REPORT_SHA = "cea6637193b1071710ab7ec3676b7226fc00f017403b78a5f051a2650a2ccc93"
PROVIDERS = ("qwen", "whisper", "parakeet", "xlsr", "lv60")
INPUT_ROLES = frozenset(("B0", "B9", "pair", "source", "screen",
    *("native_" + p for p in PROVIDERS), "train_producer", "source_producer"))
OUTPUT_ROLES = ("result", "producer", "consumed")
QUALITY_FILES = frozenset((
    *("api/" + name + ".json" for name in ("run", "commit", "artifacts")),
    *("originals/" + name + ".zip" for name in OUTPUT_ROLES),
    *("inputs/" + name + ".zip" for name in INPUT_ROLES),
))
EVIDENCE = "research/english_v6_b27_replication_continuation_20261125"
LOCK_PATH = EVIDENCE + "/REPLICATION_LOCK.json"
REPORT_PATH = EVIDENCE + "/REPLICATION_READBACK.json"
RESULT_PATH = EVIDENCE + "/ORIGINAL_RESULT.json"
RECEIPT_PATH = EVIDENCE + "/CONTINUATION.json"
INVENTORY_PATH = EVIDENCE + "/ORIGINAL_INVENTORY.json"
PRIMARY_REPORT_PATH = EVIDENCE + "/PRIMARY_READBACK.json"
BINDING_PATH = EVIDENCE + "/ENGINEERING_BINDING.json"
COMMON_PATHS = (LOCK_PATH, REPORT_PATH, RESULT_PATH, RECEIPT_PATH, INVENTORY_PATH)
PASS_PATHS = (ENGINEERING_SPEC, PRIMARY_REPORT_PATH, BINDING_PATH)
MAX_EXPORT_BYTES = 8_000_000
CHUNK_CHARS = 32768
CHUNK_PREFIX = "B27_REPLICATION_CONTINUATION_CHUNK "
COMPLETE_PREFIX = "B27_REPLICATION_CONTINUATION_COMPLETE "
need = profile.need


def validate_config(config):
    need(type(config) is dict and set(config) == {"schema", "enabled", *CONFIG_PINS}
         and config["schema"] == CONFIG_SCHEMA and type(config["enabled"]) is bool,
         "exact replica continuation configuration")
    if not config["enabled"]:
        need(all(config[key] is None for key in CONFIG_PINS),
             "disabled configuration keeps all future pins null")
        raise ValueError("INERT: no exact completed replica result is configured")
    need(type(config["replication_run_id"]) is int and config["replication_run_id"] > 0,
         "one actual replica run")
    for key in ("replication_run_head", "replication_prepared_parent"):
        need(type(config[key]) is str and re.fullmatch("[0-9a-f]{40}", config[key]),
             "one exact original Git identity " + key)
    need(config["replication_run_head"] != config["replication_prepared_parent"]
         and all(profile.valid_hash(config[key]) for key in HASH_FILES),
         "actual child and four reviewed original file digests")
    return config


def validate_entry(env, event, head, parent, changed):
    need(all(env.get(key) == value for key, value in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH, "GITHUB_RUN_ATTEMPT": "1",
    }.items()), "first public push on the separate read-only continuation branch")
    need(type(env.get("PREPARED_PARENT")) is str
         and re.fullmatch("[0-9a-f]{40}", env["PREPARED_PARENT"])
         and head == env.get("GITHUB_SHA") and parent == env["PREPARED_PARENT"]
         and head != parent and changed == [WORKFLOW],
         "one workflow-only child of the exact reviewed continuation parent")
    need(event.get("before") == parent and event.get("after") == head
         and event.get("ref") == BRANCH and event.get("forced") is False
         and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event["repository"].get("private") is False,
         "actual nonforced public activation")


def validate_workflow_activation(before, current, parent):
    need(type(before) is bytes and type(current) is bytes
         and type(parent) is str and re.fullmatch("[0-9a-f]{40}", parent)
         and before.count(PARENT_TOKEN.encode("ascii")) == 2
         and current == before.replace(PARENT_TOKEN.encode("ascii"), parent.encode("ascii")),
         "only the two reviewed workflow parent substitutions")


def check_checkout(config):
    raw = {key: (ROOT / path).read_bytes() for key, path in HASH_FILES.items()}
    need(all(profile.sha(raw[key]) == config[key] for key in HASH_FILES),
         "all four current replica registrations match reviewed original hashes")
    need(all(profile.sha((ROOT / path).read_bytes()) == digest
             for path, digest in FIXED_FILES.items()),
         "unchanged engineering binder, driver, readers and inert registration")
    doc = profile.document((ROOT / ENGINEERING_SPEC).read_bytes())
    need(doc["enabled"] is False and doc["candidate_files"] is None
         and doc["inputs"]["pair"] is None
         and all(value is None for value in doc["quality_locks"].values()),
         "engineering remains wholly unbound and cannot be retargeted")
    return raw, doc


def validate_binding_links(config, raw, bound, primary, primary_raw):
    need(set(raw) == set(HASH_FILES)
         and all(profile.sha(raw[key]) == config[key] for key in HASH_FILES)
         and profile.document(raw["external_proof_binding_sha256"]) == bound
         and set(primary_raw) == set(HASH_FILES)
         and profile.document(primary_raw["external_proof_binding_sha256"]) == primary,
         "exact current replica and PRIMARY original proof bytes")
    need(bound["panel"] == "replication" and primary["panel"] == "primary"
         and bound["status"] == primary["status"] == "BOUND_COMPLETE_ORIGINAL_TRAIN"
         and bound["scientific_execution_authorized"] is primary["scientific_execution_authorized"] is False
         and all(bound[key] == primary[key] for key in (
             "training_lock", "files", "transport", "training_reader_sha256",
             "training_report_sha256", "assessment_sha256"))
         and bound["training_report_sha256"] == TRAIN_REPORT_SHA
         and len(bound["files"]) == 41 and set(bound["primary_quality_files"]) == QUALITY_FILES
         and bound["primary_quality_reader_sha256"] == QUALITY_SHA
         and profile.valid_hash(bound["primary_quality_report_sha256"]),
         "same complete TRAIN41 and frozen original passed PRIMARY18")
    primary_lock = bound["primary_quality_lock"]
    quality.validate_lock(primary_lock)
    need(primary_lock["panel"] == "primary"
         and all(profile.sha(primary_raw[key]) == primary_lock[key] for key in HASH_FILES),
         "frozen PRIMARY lock matches every actual original PRIMARY registration")
    for panel, members, lock in (("primary", primary_raw, primary_lock),
                                 ("replication", raw, config)):
        spec = profile.document(members["inputs_sha256"])
        producer = profile.document(members["producer_manifest_sha256"])
        need(spec["panel"] == panel
             and spec["schema"] == quality.SCHEMAS[panel] and spec["scoring_allowed"] is True
             and spec["producer_manifest_sha256"] == lock["producer_manifest_sha256"]
             and spec["external_proof_binding_sha256"] == lock["external_proof_binding_sha256"]
             and spec["model_pt_sha256"] == bound["training_lock"]["training"]["model_pt_sha256"]
             and spec["assessment_sha256"] == bound["assessment_sha256"]
             and producer["schema"] == "english-v6-b27-" + panel + "-producer-v1"
             and producer["frozen"] is True,
             "same frozen candidate and reviewed original registration " + panel)
    replica_spec = profile.document(raw["inputs_sha256"])
    need(replica_spec["primary_quality_lock"] == primary_lock
         and replica_spec["original_primary_pass_required"] is True
         and replica_spec["replication_opened"] is True,
         "replica depends on exactly this complete original PRIMARY PASS")
    profile.file_pins(bound["files"], profile.LIMITS["primary_external_proof"])
    profile.file_pins(bound["primary_quality_files"], profile.LIMITS["replication_external_proof"])
    profile.validate_transport(bound["transport"], bound["files"])
    profile.validate_transport(bound["primary_quality_transport"], bound["primary_quality_files"])
    need(sum(pin["bytes"] for pin in bound["files"].values())
         + sum(pin["bytes"] for pin in bound["primary_quality_files"].values())
         <= profile.LIMITS["replication_external_proof"], "unchanged complete external proof cap")


def load_bound(config, raw):
    bound, primary = profile.binding("replication"), profile.binding("primary")
    primary_raw = {key: (ROOT / path.replace("_replication_", "_primary_")).read_bytes()
                   for key, path in HASH_FILES.items()}
    validate_binding_links(config, raw, bound, primary, primary_raw)
    return bound


def validate_run(config, run):
    need(type(run.get("id")) is int and run["id"] == config["replication_run_id"]
         and run.get("head_sha") == config["replication_run_head"]
         and type(run.get("run_attempt")) is int and run["run_attempt"] == 1
         and run.get("head_branch") == evaluator.BRANCH.removeprefix("refs/heads/")
         and run.get("path") == evaluator.WORKFLOW and run.get("event") == "push"
         and run.get("status") == "completed" and run.get("conclusion") == "success"
         and run.get("repository", {}).get("full_name") == REPO
         and run["repository"].get("private") is False,
         "exact first completed successful public replica; never wait or retry")


def validate_api(config, run, commit, inventory):
    validate_run(config, run)
    identity = dict(run_id=str(config["replication_run_id"]),
        run_head=config["replication_run_head"], run_attempt=1,
        prepared_parent=config["replication_prepared_parent"])
    evaluator.validate_execution_metadata(identity, run, commit)
    need(type(commit.get("files")) is list and len(commit["files"]) < 100,
         "complete untruncated original commit")
    need(type(inventory.get("total_count")) is int and inventory["total_count"] == 3
         and type(inventory.get("artifacts")) is list and len(inventory["artifacts"]) == 3,
         "all and only three complete original final replica artifacts")
    names = quality.artifact_names("replication", run["id"])
    need({item.get("name") for item in inventory["artifacts"]} == set(names.values()),
         "three exact output names, without missing or extra artifacts")
    pins = {}
    for role, name in names.items():
        matches = [item for item in inventory["artifacts"] if item["name"] == name]
        need(len(matches) == 1, "one original output " + role)
        item = matches[0]
        need(type(item.get("digest")) is str
             and re.fullmatch("sha256:[0-9a-f]{64}", item["digest"]),
             "actual complete original artifact SHA256")
        pin = dict(id=item["id"], run_id=run["id"], run_head=run["head_sha"],
                   bytes=item["size_in_bytes"], sha256=item["digest"][7:])
        evaluator.validate_provenance(pin, dict(run=run, commit=commit, artifact=item), role=role)
        pins[role] = pin
    need(len({pin["id"] for pin in pins.values()}) == 3, "three distinct original output IDs")
    return pins


def build_lock(config, pins, members, result_raw):
    need(set(pins) == set(OUTPUT_ROLES) and set(members) == set(HASH_FILES)
         and type(result_raw) is bytes and result_raw,
         "complete original lock inputs")
    need(all(type(raw) is bytes and profile.sha(raw) == config[key]
             for key, raw in members.items()),
         "every captured original member matches the explicit reviewed hash")
    lock = dict(schema=quality.LOCK_SCHEMA, panel="replication", **pins,
        **{key: profile.sha(raw) for key, raw in members.items()},
        original_result_sha256=profile.sha(result_raw))
    need(set(lock) == quality.LOCK_FIELDS, "exact ten-field replica lock")
    return lock


def preliminary_gate(config, result, spec, lock, bound):
    need(spec.get("panel") == "replication" and spec.get("schema") == evaluator.SCHEMA
         and spec.get("scoring_allowed") is True
         and spec.get("producer_manifest_sha256") == lock["producer_manifest_sha256"]
         and spec.get("external_proof_binding_sha256") == lock["external_proof_binding_sha256"]
         and spec.get("model_pt_sha256") == bound["training_lock"]["training"]["model_pt_sha256"]
         and spec.get("assessment_sha256") == bound["assessment_sha256"]
         and spec.get("primary_quality_lock") == bound["primary_quality_lock"],
         "same actual B27 candidate, complete PRIMARY dependency and original registration")
    need(result.get("schema") == evaluator.SCHEMA
         and result.get("run_id") == str(config["replication_run_id"])
         and result.get("run_head") == config["replication_run_head"]
         and type(result.get("run_attempt")) is int and result["run_attempt"] == 1
         and result.get("prepared_parent") == config["replication_prepared_parent"]
         and result.get("registered_inputs_sha256") == lock["inputs_sha256"]
         and result.get("producer_manifest_sha256") == lock["producer_manifest_sha256"]
         and result.get("implementation_sha256") == lock["evaluator_sha256"]
         and result.get("model_pt_sha256") == spec["model_pt_sha256"]
         and result.get("assessment_sha256") == spec["assessment_sha256"]
         and result.get("source_gate") == "SOURCE_GATE_PASS"
         and result.get("source_shortfall") == {}
         and result.get("replay_allowed") is False and result.get("consumed_unknown") == [],
         "raw original RESULT identity and complete-consumption flags before routing")
    gate = result.get("panel_gate")
    need(gate in ("PASS", "NO_PROMOTION"),
         "STOP, partial, unknown or missing outcomes never become complete results")
    return gate  # Routing only; the unchanged full reader certifies the outcome.


def input_pins(spec, bound):
    need(set(spec["inputs"]) == {"B0", "B9", "pair", "source", "screen"}
         and set(spec["native_inputs"]) == set(PROVIDERS)
         and set(spec["upstream_producers"]) == {"train", "source"},
         "complete original twelve-role registration")
    pins = evaluator.archive_pins(spec)
    need(set(pins) == INPUT_ROLES and len({pin["id"] for pin in pins.values()}) == 12,
         "twelve distinct original input IDs")
    for role, pin in pins.items():
        quality.valid_artifact(pin, panel="replication", role=role)
    need(all(pins[role] == pin for role, pin in evaluator.BASELINES.items()),
         "same original B0 and B9")
    training = bound["training_lock"]["training"]
    for role, original_role in (("pair", "models"), ("train_producer", "producer")):
        original = training["artifacts"][original_role]
        expected = {key: original[key] for key in ("id", "bytes", "sha256")}
        expected.update(run_id=training["run_id"], run_head=training["run_head"])
        need(pins[role] == expected, "same complete original B27 TRAIN input " + role)
    return pins


def quality_layout(api_raw, outputs, inputs):
    need(set(api_raw) == {"run", "commit", "artifacts"}
         and set(outputs) == set(OUTPUT_ROLES) and set(inputs) == INPUT_ROLES
         and len({pin["id"] for pin in (*outputs.values(), *inputs.values())}) == 15,
         "complete distinct original API, output and input closure")
    files, transport = {}, {}
    for name, raw in api_raw.items():
        path = "api/" + name + ".json"
        files[path] = dict(bytes=len(raw), sha256=profile.sha(raw))
        transport[path] = dict(original_api_base64=base64.b64encode(raw).decode("ascii"))
    for directory, pins in (("originals", outputs), ("inputs", inputs)):
        for name, pin in pins.items():
            path = directory + "/" + name + ".zip"
            files[path] = {key: pin[key] for key in ("bytes", "sha256")}
            transport[path] = dict(artifact_id=pin["id"])
    need(set(files) == QUALITY_FILES, "all eighteen exact original quality files")
    profile.validate_transport(transport, files)
    return files, transport


def check_report(report, panel, gate, lock, bound):
    need(report.get("schema") == quality.REPORT_SCHEMA and report.get("panel") == panel
         and report.get("panel_gate") == gate
         and report.get("original_lock_sha256") == profile.sha(profile.canonical(lock))
         and report.get("original_result_sha256") == lock["original_result_sha256"]
         and report.get("external_proof_binding_sha256") == lock["external_proof_binding_sha256"]
         and report.get("original_training_report_sha256") == bound["training_report_sha256"]
         and report.get("model_pt_sha256") == bound["training_lock"]["training"]["model_pt_sha256"]
         and report.get("assessment_sha256") == bound["assessment_sha256"],
         "same fully certified original outcome, lock, candidate and TRAIN identity")
    need(all(report.get(key) is True for key in (
        "all_original_arrays_verified", "original_source_admission_recomputed",
        "original_producer_git_verified", "all_original_training_proof_verified",
        "complete_original_Q1_failure_verified", "old_stops_and_reserves_preserved"))
         and report.get("replication_open_allowed") is (panel == "primary" and gate == "PASS")
         and report.get("candidate_engineering_validation_allowed")
             is (panel == "replication" and gate == "PASS")
         and all(report.get(key) is False for key in (
             "automatic_next_assessment_allowed", "promotion_allowed", "product_released"))
         and all(type(report.get(key)) is int and report[key] == 0
                 for key in ("new_model_calls", "new_source_calls", "new_observer_calls")),
         "all original proofs verified without any new scientific execution")


def certify_once(gate, replication, lock, locks, proof, output, bound, inert_doc):
    need(gate in ("PASS", "NO_PROMOTION"), "exactly one complete-result route")
    if gate == "NO_PROMOTION":
        report = quality.verify_quality_original(replication, lock, proof)
        check_report(report, "replication", gate, lock, bound)
        return profile.canonical(report), {}

    # Never add a preliminary full replica reader. The frozen binder keeps
    # both top-level readers, recursive proofs and original-pair authentication.
    audit = engineering_binder.prepare_inputs(
        primary=proof / "primary-quality", primary_lock=locks / "PRIMARY_LOCK.json",
        primary_proof=proof, replication=replication,
        replication_lock=locks / "REPLICATION_LOCK.json", replication_proof=proof,
        output=output)
    names = ("engineering_inputs.json", "PRIMARY_READBACK.json", "REPLICATION_READBACK.json")
    raw = {name: (output / name).read_bytes() for name in (*names, "BINDING.json")}
    report = profile.document(raw["REPLICATION_READBACK.json"])
    primary = profile.document(raw["PRIMARY_READBACK.json"])
    check_report(report, "replication", "PASS", lock, bound)
    check_report(primary, "primary", "PASS", bound["primary_quality_lock"], bound)
    need(profile.sha(raw["PRIMARY_READBACK.json"]) == bound["primary_quality_report_sha256"],
         "same exact already frozen complete original PRIMARY report")
    expected_pins = {name: dict(bytes=len(raw[name]), sha256=profile.sha(raw[name])) for name in names}
    need(profile.document(raw["BINDING.json"]) == audit
         and audit["schema"] == "english-v6-b27-reviewed-engineering-binding-v1"
         and audit["status"] == "READY_FOR_REVIEW_ONLY" and audit["files"] == expected_pins
         and all(audit[key] is False for key in (
             "actual_registration_changed", "engineering_executed", "selection_changed"))
         and all(type(audit[key]) is int and audit[key] == 0 for key in (
             "new_model_calls", "new_source_calls", "compiler_calls", "packs_created"))
         and profile.inventory(output) == {
             name: dict(bytes=len(body), sha256=profile.sha(body)) for name, body in raw.items()},
         "four complete original binder outputs and no engineering execution")
    doc = profile.document(raw["engineering_inputs.json"])
    expected = profile.document(profile.canonical(inert_doc))
    expected.update(enabled=True, quality_reader_sha256=QUALITY_SHA,
        model_pt_sha256=primary["model_pt_sha256"],
        original_training_report_sha256=bound["training_report_sha256"],
        candidate_files=doc["candidate_files"])
    expected["quality_locks"] = dict(primary=bound["primary_quality_lock"], replication=lock)
    expected["inputs"]["pair"] = primary["pair_pin"]
    stem = engineering_binder.eng.STEM
    need(doc == expected and doc["inputs"]["pair"] == report["pair_pin"]
         and type(doc["candidate_files"]) is dict
         and set(doc["candidate_files"]) == {stem + "." + suffix for suffix in ("pt", "npz", "onnx")}
         and all(set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int
                 and pin["bytes"] > 10000 and profile.valid_hash(pin["sha256"])
                 for pin in doc["candidate_files"].values())
         and doc["candidate_files"][stem + ".pt"]["sha256"] == doc["model_pt_sha256"]["challenger"],
         "only original dual locks, actual TRAIN exports and documented inert-to-bound fields change")
    return raw["REPLICATION_READBACK.json"], {
        ENGINEERING_SPEC: raw["engineering_inputs.json"],
        PRIMARY_REPORT_PATH: raw["PRIMARY_READBACK.json"], BINDING_PATH: raw["BINDING.json"]}


def make_export(files, outcome):
    need(outcome in ("PASS", "NO_PROMOTION"), "only certified complete outcomes")
    paths = COMMON_PATHS + (PASS_PATHS if outcome == "PASS" else ())
    need(set(files) == set(paths) and all(type(raw) is bytes and raw for raw in files.values()),
         "exact five or eight nonempty original-byte output files")
    total = sum(map(len, files.values()))
    need(total <= MAX_EXPORT_BYTES, "observation metadata export exceeds 8,000,000 bytes")
    records, entries = [], []
    for path in paths:
        raw = files[path]
        encoded = base64.b64encode(raw).decode("ascii")
        count = (len(encoded) + CHUNK_CHARS - 1) // CHUNK_CHARS
        pin = dict(path=path, bytes=len(raw), sha256=profile.sha(raw), chunks=count)
        entries.append(pin)
        records.extend(dict(pin, index=index,
            base64=encoded[index * CHUNK_CHARS:(index + 1) * CHUNK_CHARS])
            for index in range(count))
    return records, dict(outcome=outcome, file_count=len(paths), total_bytes=total, files=entries)


def prepare(config, work):
    config = validate_config(config)  # Refuse inert before env, network or mkdir.
    work = Path(work)
    need(work.is_absolute() and work == work.resolve() and work.parent.is_dir()
         and not work.exists() and all(not path.is_symlink() for path in work.parents),
         "fresh absolute regular read-only continuation directory")
    raw, inert_doc = check_checkout(config)
    api_raw = {"run": evaluator.gh("actions/runs/" + str(config["replication_run_id"]))}
    validate_run(config, profile.document(api_raw["run"]))
    bound = load_bound(config, raw)

    # Preserve every obtained raw API even if later metadata or certification fails.
    work.mkdir()
    raw_dir = work / "review/raw"
    (raw_dir / "api").mkdir(parents=True)
    (raw_dir / "api/run.json").write_bytes(api_raw["run"])
    for name, endpoint in (
        ("commit", "commits/" + config["replication_run_head"] + "?per_page=100"),
        ("artifacts", "actions/runs/" + str(config["replication_run_id"]) + "/artifacts?per_page=100"),
    ):
        api_raw[name] = evaluator.gh(endpoint)
        (raw_dir / "api" / (name + ".json")).write_bytes(api_raw[name])
    outputs = validate_api(config, *(profile.document(api_raw[name])
                                    for name in ("run", "commit", "artifacts")))
    replication = work / "replication-quality"
    for name in ("api", "originals", "inputs"):
        (replication / name).mkdir(parents=True)
    for name, body in api_raw.items():
        (replication / "api" / (name + ".json")).write_bytes(body)

    cache = {}

    def obtain(target, pin):
        key = (pin["id"], pin["bytes"], pin["sha256"])
        body = cache[key].read_bytes() if key in cache else evaluator.gh(
            "actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(body) == pin["bytes"] and profile.sha(body) == pin["sha256"],
             "whole original artifact bytes " + str(pin["id"]))
        need(not target.exists(), "never overwrite an acquired original")
        target.write_bytes(body)
        cache[key] = target

    for role in OUTPUT_ROLES:
        obtain(replication / "originals" / (role + ".zip"), outputs[role])
    with quality.original_archive(replication / "originals/producer.zip",
                                  outputs["producer"], "replication", "producer") as archive:
        members = {key: archive.read("replication_producer/" + path)
                   for key, path in HASH_FILES.items()}
    with quality.original_archive(replication / "originals/result.zip",
                                  outputs["result"], "replication", "result") as archive:
        result_raw = archive.read("RESULT.json")
    for key, body in members.items():
        target = raw_dir / "captured" / HASH_FILES[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    (raw_dir / "RESULT.json").write_bytes(result_raw)
    lock = build_lock(config, outputs, members, result_raw)
    quality.validate_lock(lock)
    spec = profile.document(members["inputs_sha256"])
    gate = preliminary_gate(config, profile.document(result_raw), spec, lock, bound)
    inputs = input_pins(spec, bound)
    files18, transport18 = quality_layout(api_raw, outputs, inputs)
    locks = work / "locks"
    locks.mkdir()
    lock_raw = profile.canonical(lock)
    primary_lock_raw = profile.canonical(bound["primary_quality_lock"])
    for name, body in (("PRIMARY_LOCK.json", primary_lock_raw), ("REPLICATION_LOCK.json", lock_raw)):
        need(len(body) <= engineering_binder.readback.REPORT_LIMIT, "unchanged original lock report cap")
        (locks / name).write_bytes(body)
        (raw_dir / name).write_bytes(body)

    proof = work / "proof"
    proof.mkdir()
    # Fetch/hash only; the fetch()/quality-fetch CLIs would add full readers.
    for name, files, transport in (
        ("training-origin", bound["files"], bound["transport"]),
        ("primary-quality", bound["primary_quality_files"], bound["primary_quality_transport"]),
    ):
        original_proof.fetch_files(proof / name, files, transport)
        for path, location in transport.items():
            if "artifact_id" in location:
                pin = files[path]
                cache[(location["artifact_id"], pin["bytes"], pin["sha256"])] = proof / name / path
    for role, pin in inputs.items():
        obtain(replication / "inputs" / (role + ".zip"), pin)
    expected = {replication: files18, proof / "training-origin": bound["files"],
                proof / "primary-quality": bound["primary_quality_files"]}
    need(all(profile.inventory(directory) == files for directory, files in expected.items()),
         "all and only original TRAIN41, PRIMARY18 and REPLICATION18 bytes")
    report_raw, extra = certify_once(gate, replication, lock, locks, proof, work / "bound", bound, inert_doc)
    report = profile.document(report_raw)
    outcome = report["panel_gate"]
    need(outcome == gate and all(profile.inventory(directory) == files
                                for directory, files in expected.items()),
         "complete certified outcome and every acquired original remain unchanged")
    need(not {"torch", "numpy", "onnxruntime", "piper", "transformers"} & set(sys.modules),
         "no scientific runtime during original continuation")
    receipt = dict(schema="english-v6-b27-replication-continuation-v1",
        status=("DUAL_PASS_ENGINEERING_INPUTS_READY_FOR_REVIEW" if outcome == "PASS"
                else "B27_REPLICATION_NO_PROMOTION_TERMINAL"),
        replication_outcome=outcome, replication_run_id=config["replication_run_id"],
        replication_run_head=config["replication_run_head"],
        original_lock_sha256=profile.sha(lock_raw),
        certification_route=("frozen_engineering_input_binder" if outcome == "PASS"
                             else "frozen_replication_quality_reader"),
        additional_preverification_calls=0, candidate_terminal=outcome == "NO_PROMOTION",
        engineering_inputs_ready_for_review=outcome == "PASS",
        actual_registration_changed=False, selection_changed=False, engineering_started=False,
        scientific_execution_authorized=False, new_model_calls=0, new_source_calls=0,
        new_observer_calls=0, new_optimizer_updates=0, compiler_calls=0, packs_created=0,
        new_marker_consumptions=0, remote_mutations=0, promotion_allowed=False, product_released=False,
        automatic_next_assessment_allowed=False, old_stops_and_reserves_preserved=True)
    exported = {LOCK_PATH: lock_raw, REPORT_PATH: report_raw, RESULT_PATH: result_raw,
                RECEIPT_PATH: profile.canonical(receipt), **extra}
    inventory = dict(schema="english-v6-b27-replication-complete-original-inventory-v1",
        replication_quality_files=files18, replication_quality_transport=transport18,
        quality_output_artifacts=outputs, quality_input_artifacts=inputs,
        replication_proof_binding_sha256=config["external_proof_binding_sha256"],
        training_origin_files=bound["files"], primary_quality_files=bound["primary_quality_files"],
        primary_quality_transport=bound["primary_quality_transport"],
        primary_quality_lock_sha256=profile.sha(primary_lock_raw),
        primary_quality_report_sha256=bound["primary_quality_report_sha256"],
        original_training_report_sha256=bound["training_report_sha256"],
        original_lock_sha256=profile.sha(lock_raw), original_result_sha256=profile.sha(result_raw),
        review_files={path: dict(bytes=len(body), sha256=profile.sha(body))
                      for path, body in exported.items()},
        original_input_archives_repackaged=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0,
        compiler_calls=0, packs_created=0, new_marker_consumptions=0, remote_mutations=0)
    exported[INVENTORY_PATH] = profile.canonical(inventory)
    records, manifest = make_export(exported, outcome)
    export = work / "review/export"
    export.mkdir()
    for path, body in exported.items():
        target = export / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    need(profile.inventory(export) == {
        path: dict(bytes=len(body), sha256=profile.sha(body)) for path, body in exported.items()},
        "exact regular five or eight exported files")
    for record in records:
        print(CHUNK_PREFIX + profile.canonical(record).decode("ascii"), flush=True)
    print(COMPLETE_PREFIX + profile.canonical(manifest).decode("ascii"), flush=True)


def main():
    config = validate_config(profile.document(CONFIG.read_bytes()))
    need(len(sys.argv) == 2, "usage: helper ABSOLUTE_FRESH_WORK_DIRECTORY")
    env = os.environ
    event = profile.document(Path(env["GITHUB_EVENT_PATH"]).read_bytes())

    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True, text=True).stdout.strip()

    lineage = git("rev-list", "--parents", "-n", "1", "HEAD").split()
    need(len(lineage) == 2, "exactly one continuation Git parent")
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    validate_entry(env, event, lineage[0], lineage[1], changed)
    before = subprocess.run(["git", "-C", str(ROOT), "show", lineage[1] + ":" + WORKFLOW],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    validate_workflow_activation(before, (ROOT / WORKFLOW).read_bytes(), lineage[1])
    prepare(config, Path(sys.argv[1]))


if __name__ == "__main__":
    main()

