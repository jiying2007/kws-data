"""Inert-by-default, read-only B27 PRIMARY original-result continuation.

Only existing API bodies and original artifacts are fetched. No model, audio
generation, observer, optimizer, remote write, or workflow activation is called.
PASS invokes the frozen replication binder once, without extra preverification.
That frozen binder retains its two internal TRAIN verifications and one PRIMARY
verification. NO_PROMOTION invokes only the frozen quality reader once.

The 8,000,000-byte export bound applies solely to observation/review metadata.
All original scientific, archive and proof limits remain untouched.
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
import bind_b27_source as source_binder
import english_v6_b27_original_proof as original_proof
import english_v6_b27_primary_eval as evaluator
import english_v6_b27_profile as profile
import english_v6_b27_quality_readback as quality

REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b27-primary-continuation-ci-once-20261125"
WORKFLOW = ".github/workflows/b27-primary-continuation.yml"
PARENT_TOKEN = "__B27_REVIEWED_PRIMARY_CONTINUATION_PARENT__"
CONFIG_SCHEMA = "english-v6-b27-primary-continuation-config-v1"
HASH_FILES = {
    "evaluator_sha256": "automation/english_v6_b27_primary_eval.py",
    "inputs_sha256": "automation/english_v6_b27_primary_inputs.json",
    "producer_manifest_sha256": "automation/english_v6_b27_primary_producer.json",
    "external_proof_binding_sha256": "automation/english_v6_b27_primary_proof_binding.json",
}
CONFIG_PINS = (
    "primary_run_id", "primary_run_head", "primary_prepared_parent", *HASH_FILES,
)
PROVIDERS = ("qwen", "whisper", "parakeet", "xlsr", "lv60")
INPUT_ROLES = frozenset((
    "B0", "B9", "pair", "source", "screen",
    *("native_" + name for name in PROVIDERS), "train_producer", "source_producer",
))
OUTPUT_ROLES = ("result", "producer", "consumed")
EVIDENCE = "research/english_v6_b27_primary_results_20261125"
LOCK_PATH = EVIDENCE + "/PRIMARY_LOCK.json"
REPORT_PATH = EVIDENCE + "/PRIMARY_READBACK.json"
RESULT_PATH = EVIDENCE + "/ORIGINAL_RESULT.json"
INVENTORY_PATH = EVIDENCE + "/ORIGINAL_INVENTORY.json"
RECEIPT_PATH = EVIDENCE + "/CONTINUATION.json"
REPLICA_PATH = "automation/english_v6_b27_replication_proof_binding.json"
TRAIN_REPORT_PATH = EVIDENCE + "/replication_binding/TRAINING_READBACK.json"
AUDIT_PATH = EVIDENCE + "/replication_binding/BINDING_AUDIT.json"
COMMON_PATHS = (LOCK_PATH, REPORT_PATH, RESULT_PATH, INVENTORY_PATH, RECEIPT_PATH)
PASS_PATHS = (REPLICA_PATH, TRAIN_REPORT_PATH, AUDIT_PATH)
MAX_EXPORT_BYTES = 8_000_000
CHUNK_CHARS = 32768
CHUNK_PREFIX = "B27_PRIMARY_CONTINUATION_CHUNK "
COMPLETE_PREFIX = "B27_PRIMARY_CONTINUATION_COMPLETE "
need = profile.need


def validate_config(config):
    need(type(config) is dict
         and set(config) == {"schema", "enabled", *CONFIG_PINS}
         and config["schema"] == CONFIG_SCHEMA
         and type(config["enabled"]) is bool, "exact continuation configuration")
    if not config["enabled"]:
        need(all(config[key] is None for key in CONFIG_PINS),
             "disabled configuration retains every future identity as null")
        raise ValueError("INERT: no reviewed actual PRIMARY identity is bound")
    need(type(config["primary_run_id"]) is int and config["primary_run_id"] > 0,
         "one explicit original PRIMARY run")
    for key in ("primary_run_head", "primary_prepared_parent"):
        need(type(config[key]) is str and re.fullmatch("[0-9a-f]{40}", config[key]),
             "exact original PRIMARY Git identity " + key)
    need(config["primary_run_head"] != config["primary_prepared_parent"],
         "original PRIMARY must be a new single-parent child")
    need(all(profile.valid_hash(config[key]) for key in HASH_FILES),
         "four explicitly reviewed original file digests")
    return config


def validate_entry(env, event, head, parent, changed):
    need(all(env.get(key) == value for key, value in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH,
        "GITHUB_RUN_ATTEMPT": "1",
    }.items()), "first push on the separate read-only continuation branch")
    prepared = env.get("PREPARED_PARENT", "")
    need(re.fullmatch("[0-9a-f]{40}", prepared) is not None
         and head == env.get("GITHUB_SHA") and parent == prepared
         and head != parent and changed == [WORKFLOW],
         "one workflow-only child of the reviewed continuation parent")
    need(event.get("before") == parent and event.get("after") == head
         and event.get("ref") == BRANCH
         and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event["repository"].get("private") is False,
         "actual nonforced public workflow-only activation")


def validate_workflow_activation(before_raw, current_raw, parent):
    need(type(before_raw) is bytes and type(current_raw) is bytes
         and type(parent) is str and re.fullmatch("[0-9a-f]{40}", parent)
         and before_raw.count(PARENT_TOKEN.encode("ascii")) == 2
         and current_raw == before_raw.replace(PARENT_TOKEN.encode("ascii"), parent.encode("ascii")),
         "only the two reviewed continuation parent substitutions")


def validate_run(config, run):
    need(type(run.get("id")) is int and run["id"] == config["primary_run_id"]
         and run.get("head_sha") == config["primary_run_head"]
         and type(run.get("run_attempt")) is int and run["run_attempt"] == 1
         and run.get("head_branch") == evaluator.BRANCH.removeprefix("refs/heads/")
         and run.get("path") == evaluator.WORKFLOW and run.get("event") == "push"
         and run.get("status") == "completed" and run.get("conclusion") == "success"
         and run.get("repository", {}).get("full_name") == REPO
         and run["repository"].get("private") is False,
         "exact first completed successful public PRIMARY; never wait or retry it")


def validate_primary_api(config, run, commit, inventory):
    validate_run(config, run)
    identity = dict(run_id=str(config["primary_run_id"]),
        run_head=config["primary_run_head"], run_attempt=1,
        prepared_parent=config["primary_prepared_parent"])
    evaluator.validate_execution_metadata(identity, run, commit)
    need(type(inventory.get("total_count")) is int and inventory["total_count"] == 3
         and type(inventory.get("artifacts")) is list and len(inventory["artifacts"]) == 3,
         "all and only the three original final PRIMARY artifacts")
    names = quality.artifact_names("primary", run["id"])
    need({item.get("name") for item in inventory["artifacts"]} == set(names.values()),
         "exact three frozen output names; no missing or extra artifact")
    pins = {}
    for role, name in names.items():
        items = [item for item in inventory["artifacts"] if item.get("name") == name]
        need(len(items) == 1, "one original named output " + role)
        item = items[0]
        need(type(item.get("digest")) is str
             and re.fullmatch("sha256:[0-9a-f]{64}", item["digest"]),
             "actual original artifact SHA256")
        pin = dict(id=item["id"], run_id=run["id"], run_head=run["head_sha"],
            bytes=item["size_in_bytes"], sha256=item["digest"][7:])
        evaluator.validate_provenance(pin, dict(run=run, commit=commit, artifact=item), role=role)
        pins[role] = pin
    need(len({pin["id"] for pin in pins.values()}) == 3, "three distinct output IDs")
    return pins


def build_lock(config, pins, members, result_raw):
    need(set(pins) == set(OUTPUT_ROLES) and set(members) == set(HASH_FILES)
         and type(result_raw) is bytes and result_raw, "complete raw lock inputs")
    for key, raw in members.items():
        need(type(raw) is bytes and profile.sha(raw) == config[key],
             "captured original bytes match reviewed digest " + key)
    lock = dict(schema=quality.LOCK_SCHEMA, panel="primary", **pins,
        **{key: profile.sha(raw) for key, raw in members.items()},
        original_result_sha256=profile.sha(result_raw))
    need(set(lock) == quality.LOCK_FIELDS, "exact ten-field PRIMARY lock")
    return lock


def preliminary_gate(config, result, spec, lock, bound):
    need(spec.get("schema") == evaluator.SCHEMA and spec.get("panel") == "primary"
         and spec.get("scoring_allowed") is True
         and spec.get("producer_manifest_sha256") == lock["producer_manifest_sha256"]
         and spec.get("external_proof_binding_sha256") == lock["external_proof_binding_sha256"]
         and spec.get("model_pt_sha256") == bound["training_lock"]["training"]["model_pt_sha256"]
         and spec.get("assessment_sha256") == bound["assessment_sha256"],
         "original registration belongs to the same frozen B27 candidate")
    need(result.get("schema") == evaluator.SCHEMA
         and result.get("run_id") == str(config["primary_run_id"])
         and result.get("run_head") == config["primary_run_head"]
         and type(result.get("run_attempt")) is int and result["run_attempt"] == 1
         and result.get("prepared_parent") == config["primary_prepared_parent"]
         and result.get("registered_inputs_sha256") == lock["inputs_sha256"]
         and result.get("producer_manifest_sha256") == lock["producer_manifest_sha256"]
         and result.get("implementation_sha256") == lock["evaluator_sha256"]
         and result.get("model_pt_sha256") == spec["model_pt_sha256"]
         and result.get("assessment_sha256") == spec["assessment_sha256"]
         and result.get("source_gate") == "SOURCE_GATE_PASS"
         and result.get("source_shortfall") == {},
         "raw RESULT identity is consistent before choosing a certification path")
    gate = result.get("panel_gate")
    need(gate in ("PASS", "NO_PROMOTION"),
         "STOP, partial or unknown results cannot enter a complete-result reader")
    return gate  # Routing only; never a certified quality verdict.


def input_pins(spec, bound):
    need(set(spec["inputs"]) == {"B0", "B9", "pair", "source", "screen"}
         and set(spec["native_inputs"]) == set(PROVIDERS)
         and set(spec["upstream_producers"]) == {"train", "source"},
         "complete original twelve-role registration")
    pins = evaluator.archive_pins(spec)
    need(set(pins) == INPUT_ROLES and len({pin["id"] for pin in pins.values()}) == 12,
         "twelve distinct original input IDs")
    for role, pin in pins.items():
        quality.valid_artifact(pin, panel="primary", role=role)
    need(all(pins[role] == pin for role, pin in evaluator.BASELINES.items()),
         "original B0/B9 input pins unchanged")
    training = bound["training_lock"]["training"]
    for role, original_role in (("pair", "models"), ("train_producer", "producer")):
        original = training["artifacts"][original_role]
        expected = {key: original[key] for key in ("id", "bytes", "sha256")}
        expected.update(run_id=training["run_id"], run_head=training["run_head"])
        need(pins[role] == expected, "same exact original B27 TRAIN artifact")
    return pins


def quality_layout(api_raw, outputs, inputs):
    need(set(api_raw) == {"run", "commit", "artifacts"}
         and set(outputs) == set(OUTPUT_ROLES) and set(inputs) == INPUT_ROLES,
         "exact API/output/input role closure")
    need(len({pin["id"] for pin in (*outputs.values(), *inputs.values())}) == 15,
         "fifteen distinct original output and input artifacts")
    files, transport = {}, {}
    for name, raw in api_raw.items():
        path = "api/" + name + ".json"
        files[path] = dict(bytes=len(raw), sha256=profile.sha(raw))
        transport[path] = dict(original_api_base64=base64.b64encode(raw).decode("ascii"))
    for directory, pins in (("originals", outputs), ("inputs", inputs)):
        for role, pin in pins.items():
            path = directory + "/" + role + ".zip"
            files[path] = {key: pin[key] for key in ("bytes", "sha256")}
            transport[path] = dict(artifact_id=pin["id"])
    need(len(files) == 18, "exact eighteen original quality files")
    profile.validate_transport(transport, files)
    return files, transport


def require_quality_inventory(directory, expected):
    need(len(expected) == 18 and profile.inventory(directory) == expected,
         "all and only the eighteen exact original quality files")


def check_report(report, gate, lock, bound):
    need(report.get("schema") == quality.REPORT_SCHEMA and report.get("panel") == "primary"
         and report.get("panel_gate") == gate
         and report.get("original_lock_sha256") == profile.sha(profile.canonical(lock))
         and report.get("original_result_sha256") == lock["original_result_sha256"]
         and report.get("external_proof_binding_sha256") == lock["external_proof_binding_sha256"]
         and report.get("original_training_report_sha256") == bound["training_report_sha256"]
         and report.get("model_pt_sha256") == bound["training_lock"]["training"]["model_pt_sha256"]
         and report.get("assessment_sha256") == bound["assessment_sha256"],
         "same complete original PRIMARY verdict, lock, candidate and TRAIN proof")
    need(all(report.get(key) is True for key in (
        "all_original_arrays_verified", "original_source_admission_recomputed",
        "original_producer_git_verified", "all_original_training_proof_verified",
        "complete_original_Q1_failure_verified", "old_stops_and_reserves_preserved"))
         and report.get("replication_open_allowed") is (gate == "PASS")
         and all(report.get(key) is False for key in (
             "candidate_engineering_validation_allowed", "automatic_next_assessment_allowed",
             "promotion_allowed", "product_released"))
         and all(type(report.get(key)) is int and report[key] == 0
                 for key in ("new_model_calls", "new_source_calls", "new_observer_calls")),
         "fully verified original result and no newly executed scientific operation")


def certify_once(gate, lock, proof, output, bound, primary_files, primary_transport):
    need(gate in ("PASS", "NO_PROMOTION"), "exactly one complete-result route")
    if gate == "NO_PROMOTION":
        report = quality.verify_quality_original(proof / "primary-quality", lock, proof)
        check_report(report, gate, lock, bound)
        return profile.canonical(report), {}

    # Retain the original binder's two internal TRAIN checks and one PRIMARY
    # check. Do not add a preliminary quality reader or alter its caching.
    audit = source_binder.bind("replication", proof, bound["training_lock"],
                              output, primary_lock=lock)
    primary_raw = (output / "PRIMARY_READBACK.json").read_bytes()
    training_raw = (output / "TRAINING_READBACK.json").read_bytes()
    audit_raw = (output / "BINDING_AUDIT.json").read_bytes()
    binding_raw = (output / REPLICA_PATH).read_bytes()
    report, replica = profile.document(primary_raw), profile.document(binding_raw)
    check_report(report, gate, lock, bound)
    need(profile.document(audit_raw) == audit
         and audit["binding_sha256"] == profile.sha(binding_raw)
         and audit["original_primary_pass_required"] is True
         and audit["scientific_execution_authorized"] is False
         and all(type(audit[key]) is int and audit[key] == 0
                 for key in ("new_model_calls", "new_source_calls", "new_observer_calls", "remote_mutations"))
         and replica["panel"] == "replication"
         and replica["status"] == "BOUND_COMPLETE_ORIGINAL_TRAIN"
         and replica["scientific_execution_authorized"] is False
         and replica["training_lock"] == bound["training_lock"]
         and replica["files"] == bound["files"]
         and replica["transport"] == bound["transport"]
         and replica["training_report_sha256"] == bound["training_report_sha256"]
             == profile.sha(training_raw)
         and replica["primary_quality_lock"] == lock
         and replica["primary_quality_files"] == primary_files
         and replica["primary_quality_transport"] == primary_transport
         and replica["primary_quality_report_sha256"] == profile.sha(primary_raw),
         "exact frozen binder output, whole original proofs and no execution authorization")
    return primary_raw, {REPLICA_PATH: binding_raw,
                         TRAIN_REPORT_PATH: training_raw, AUDIT_PATH: audit_raw}


def make_export(files, outcome):
    need(outcome in ("PASS", "NO_PROMOTION"), "certified export outcome only")
    paths = COMMON_PATHS + (PASS_PATHS if outcome == "PASS" else ())
    need(set(files) == set(paths)
         and all(type(raw) is bytes and raw for raw in files.values()),
         "exact five or eight nonempty original-byte export files")
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
    config = validate_config(config)  # Must precede network or directory creation.
    work = Path(work)
    need(work.is_absolute() and work == work.resolve() and work.parent.is_dir()
         and not work.exists() and all(not p.is_symlink() for p in work.parents),
         "fresh absolute regular continuation workspace")
    for key, path in HASH_FILES.items():
        need(profile.sha((ROOT / path).read_bytes()) == config[key],
             "checkout retains reviewed original bytes " + path)
    bound = profile.binding("primary")
    need(profile.binding("replication", False)["status"] == "INERT_UNTIL_COMPLETE_ORIGINAL_B27_TRAIN",
         "replica remains unopened and cannot be rebound")

    api_raw = {"run": evaluator.gh("actions/runs/" + str(config["primary_run_id"]))}
    run = profile.document(api_raw["run"])
    validate_run(config, run)  # Refuse nonterminal runs before any archive transport.
    api_raw["commit"] = evaluator.gh("commits/" + config["primary_run_head"])
    api_raw["artifacts"] = evaluator.gh(
        "actions/runs/" + str(config["primary_run_id"]) + "/artifacts?per_page=100")
    outputs = validate_primary_api(config, run, profile.document(api_raw["commit"]),
                                   profile.document(api_raw["artifacts"]))

    work.mkdir()
    proof = work / "proof"
    primary = proof / "primary-quality"
    for name in ("api", "originals", "inputs"):
        (primary / name).mkdir(parents=True, exist_ok=True)
    raw_dir = work / "review" / "raw"
    (raw_dir / "api").mkdir(parents=True)
    for name, raw in api_raw.items():
        (primary / "api" / (name + ".json")).write_bytes(raw)
        (raw_dir / "api" / (name + ".json")).write_bytes(raw)

    cache = {}

    def obtain(target, pin):
        key = (pin["id"], pin["bytes"], pin["sha256"])
        raw = cache[key].read_bytes() if key in cache else evaluator.gh(
            "actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and profile.sha(raw) == pin["sha256"],
             "entire exact original artifact " + str(pin["id"]))
        need(not target.exists(), "never overwrite an original acquisition")
        target.write_bytes(raw)
        cache[key] = target

    for role in OUTPUT_ROLES:
        obtain(primary / "originals" / (role + ".zip"), outputs[role])
    with quality.original_archive(primary / "originals/producer.zip",
                                  outputs["producer"], "primary", "producer") as archive:
        members = {key: archive.read("primary_producer/" + path)
                   for key, path in HASH_FILES.items()}
    with quality.original_archive(primary / "originals/result.zip",
                                  outputs["result"], "primary", "result") as archive:
        result_raw = archive.read("RESULT.json")
    for key, raw in members.items():
        target = raw_dir / "captured" / HASH_FILES[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
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
    (locks / "PRIMARY_LOCK.json").write_bytes(lock_raw)
    (locks / "TRAINING_LOCK.json").write_bytes(profile.canonical(bound["training_lock"]))

    # Fetch/hash only. Its fetch()/CLI would perform an unwanted extra readback.
    origin = proof / "training-origin"
    original_proof.fetch_files(origin, bound["files"], bound["transport"])
    for name, location in bound["transport"].items():
        if "artifact_id" in location:
            pin = bound["files"][name]
            cache[(location["artifact_id"], pin["bytes"], pin["sha256"])] = origin / name
    for role, pin in inputs.items():
        obtain(primary / "inputs" / (role + ".zip"), pin)
    require_quality_inventory(primary, files18)

    primary_raw, extra = certify_once(
        gate, lock, proof, work / "bound", bound, files18, transport18)
    report = profile.document(primary_raw)
    outcome = report["panel_gate"]  # Export only the fully certified outcome.
    require_quality_inventory(primary, files18)
    need(profile.inventory(origin) == bound["files"],
         "all complete original TRAIN proof bytes remain unchanged")
    need(not {"torch", "numpy", "onnxruntime", "piper", "transformers"} & set(sys.modules),
         "no scientific runtime in continuation")

    receipt = dict(
        schema="english-v6-b27-primary-continuation-v1",
        status=("PRIMARY_PASS_REPLICA_PROOF_READY_FOR_REVIEW" if outcome == "PASS"
                else "B27_PRIMARY_NO_PROMOTION_TERMINAL"),
        primary_outcome=outcome, primary_run_id=config["primary_run_id"],
        primary_run_head=config["primary_run_head"],
        original_lock_sha256=profile.sha(lock_raw),
        certification_route=("frozen_replication_binder" if outcome == "PASS"
                             else "frozen_primary_quality_reader"),
        additional_preverification_calls=0, candidate_terminal=outcome == "NO_PROMOTION",
        replica_proof_ready_for_review=outcome == "PASS",
        replication_source_started=False, engineering_started=False,
        scientific_execution_authorized=False, new_model_calls=0, new_source_calls=0,
        new_observer_calls=0, new_optimizer_updates=0, remote_mutations=0,
        promotion_allowed=False, product_released=False,
        automatic_next_assessment_allowed=False, old_stops_and_reserves_preserved=True,
    )
    exported = {LOCK_PATH: lock_raw, REPORT_PATH: primary_raw, RESULT_PATH: result_raw,
                RECEIPT_PATH: profile.canonical(receipt), **extra}
    inventory = dict(
        schema="english-v6-b27-primary-complete-original-inventory-v1",
        primary_quality_files=files18, primary_quality_transport=transport18,
        quality_output_artifacts=outputs, quality_input_artifacts=inputs,
        primary_proof_binding_sha256=config["external_proof_binding_sha256"],
        training_origin_files=bound["files"],
        original_training_report_sha256=bound["training_report_sha256"],
        original_lock_sha256=profile.sha(lock_raw),
        original_result_sha256=profile.sha(result_raw),
        review_files={path: dict(bytes=len(raw), sha256=profile.sha(raw))
                      for path, raw in exported.items()},
        original_input_archives_repackaged=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0,
    )
    exported[INVENTORY_PATH] = profile.canonical(inventory)
    records, manifest = make_export(exported, outcome)
    export = work / "review" / "export"
    export.mkdir()
    for path, raw in exported.items():
        target = export / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    need(profile.inventory(export) ==
         {path: dict(bytes=len(raw), sha256=profile.sha(raw))
          for path, raw in exported.items()}, "exact regular export closure")
    for record in records:
        print(CHUNK_PREFIX + profile.canonical(record).decode("ascii"), flush=True)
    print(COMPLETE_PREFIX + profile.canonical(manifest).decode("ascii"), flush=True)


def main():
    need(len(sys.argv) == 2, "usage: helper ABSOLUTE_FRESH_WORK_DIRECTORY")
    config = validate_config(profile.document(CONFIG.read_bytes()))
    env = os.environ
    event = profile.document(Path(env["GITHUB_EVENT_PATH"]).read_bytes())

    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, text=True).stdout.strip()

    lineage = git("rev-list", "--parents", "-n", "1", "HEAD").split()
    need(len(lineage) == 2, "exactly one continuation Git parent")
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    validate_entry(env, event, lineage[0], lineage[1], changed)
    before_raw = subprocess.run(
        ["git", "-C", str(ROOT), "show", lineage[1] + ":" + WORKFLOW],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    validate_workflow_activation(before_raw, (ROOT / WORKFLOW).read_bytes(), lineage[1])
    prepare(config, Path(sys.argv[1]))


if __name__ == "__main__":
    main()
