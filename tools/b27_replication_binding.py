"""Inert-by-default transport of complete B27 replica quality inputs.

Only existing original artifacts and API bodies are acquired. The frozen
replication quality binder is called once, with no preliminary full reader.
Its internal original-proof verification remains unchanged. SOURCE_GATE_PASS
does not certify replica model quality or authorize promotion.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_suffix(".json")
sys.path.insert(0, str(ROOT / "automation"))
import bind_b27_quality as binder
import english_v6_b27_original_proof as original_proof
import english_v6_b27_profile as profile
import english_v6_b27_quality_readback as quality
import english_v6_b27_replication_eval as evaluator

REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b27-replication-binding-ci-once-20261125"
WORKFLOW = ".github/workflows/b27-replication-binding.yml"
PARENT_TOKEN = "__B27_REVIEWED_REPLICATION_BINDING_PARENT__"
CONFIG_SCHEMA = "english-v6-b27-replication-binding-config-v1"
HASH_FILES = {
    "replication_evaluator_sha256": "automation/english_v6_b27_replication_eval.py",
    "replication_proof_binding_sha256": "automation/english_v6_b27_replication_proof_binding.json",
    "primary_inputs_sha256": "automation/english_v6_b27_primary_inputs.json",
    "primary_producer_manifest_sha256": "automation/english_v6_b27_primary_producer.json",
}
CONFIG_PINS = ("source_run_id", "source_run_head", "source_prepared_parent", *HASH_FILES)
TRAIN_REPORT_SHA = "cea6637193b1071710ab7ec3676b7226fc00f017403b78a5f051a2650a2ccc93"
PRIMARY_FILES = frozenset((
    *("api/" + name + ".json" for name in ("run", "commit", "artifacts")),
    *("originals/" + role + ".zip" for role in ("result", "producer", "consumed")),
    *("inputs/" + role + ".zip" for role in binder.ROLES),
))
EVIDENCE = "research/english_v6_b27_replication_binding_20261125"
EXPORT_PATHS = (
    "automation/english_v6_b27_replication_inputs.json",
    "automation/english_v6_b27_replication_producer.json",
    EVIDENCE + "/BINDING_AUDIT.json",
    EVIDENCE + "/SCREEN.json",
    EVIDENCE + "/ORIGINAL_INVENTORY.json",
)
MAX_EXPORT_BYTES = 2_000_000  # Observation metadata only; scientific limits are unchanged.
CHUNK_CHARS = 32768
CHUNK_PREFIX = "B27_REPLICATION_BINDING_CHUNK "
COMPLETE_PREFIX = "B27_REPLICATION_BINDING_COMPLETE "
need = profile.need


def validate_config(config):
    need(type(config) is dict and set(config) == {"schema", "enabled", *CONFIG_PINS}
         and config["schema"] == CONFIG_SCHEMA and type(config["enabled"]) is bool,
         "exact replica binding configuration")
    if not config["enabled"]:
        need(all(config[key] is None for key in CONFIG_PINS),
             "disabled configuration keeps every future pin null")
        raise ValueError("INERT: no reviewed completed replica SOURCE is configured")
    need(type(config["source_run_id"]) is int and config["source_run_id"] > 0,
         "one actual replica SOURCE run")
    for key in ("source_run_head", "source_prepared_parent"):
        need(type(config[key]) is str and re.fullmatch("[0-9a-f]{40}", config[key]),
             "exact original SOURCE Git identity " + key)
    need(config["source_run_head"] != config["source_prepared_parent"],
         "SOURCE is a new one-parent execution")
    need(all(profile.valid_hash(config[key]) for key in HASH_FILES),
         "four explicitly reviewed original file hashes")
    return config


def validate_entry(env, event, head, parent, changed):
    need(all(env.get(key) == value for key, value in {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH,
        "GITHUB_RUN_ATTEMPT": "1",
    }.items()), "first public push on the separate read-only replica binding branch")
    need(type(env.get("PREPARED_PARENT")) is str
         and re.fullmatch("[0-9a-f]{40}", env["PREPARED_PARENT"])
         and head == env.get("GITHUB_SHA") and parent == env["PREPARED_PARENT"]
         and head != parent and changed == [WORKFLOW],
         "one workflow-only child of the reviewed binding parent")
    need(event.get("before") == parent and event.get("after") == head
         and event.get("ref") == BRANCH
         and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event["repository"].get("private") is False,
         "actual nonforced public first activation")


def validate_workflow_activation(before, current, parent):
    need(type(before) is bytes and type(current) is bytes
         and type(parent) is str and re.fullmatch("[0-9a-f]{40}", parent)
         and before.count(PARENT_TOKEN.encode("ascii")) == 2
         and current == before.replace(PARENT_TOKEN.encode("ascii"), parent.encode("ascii")),
         "only the two reviewed workflow parent substitutions")


def validate_source_run(config, run):
    need(type(run.get("id")) is int and run["id"] == config["source_run_id"]
         and run.get("head_sha") == config["source_run_head"]
         and type(run.get("run_attempt")) is int and run["run_attempt"] == 1
         and run.get("head_branch") == evaluator.SOURCE_BRANCH
         and run.get("path") == evaluator.SOURCE_WORKFLOW
         and run.get("event") == "push"
         and run.get("status") == "completed" and run.get("conclusion") == "success"
         and run.get("repository", {}).get("full_name") == REPO
         and run["repository"].get("private") is False,
         "exact first completed successful public replica SOURCE; no wait or retry")


def validate_source_commit(config, commit):
    need(commit.get("sha") == config["source_run_head"]
         and [item["sha"] for item in commit.get("parents", [])]
             == [config["source_prepared_parent"]]
         and [item["filename"] for item in commit.get("files", [])]
             == [evaluator.SOURCE_WORKFLOW],
         "actual replica SOURCE workflow-only child of its separately reviewed parent")


def validate_history(config, history):
    need(type(history.get("total_count")) is int and history["total_count"] == 1
         and type(history.get("workflow_runs")) is list and len(history["workflow_runs"]) == 1,
         "complete SOURCE history contains only its first dedicated push")
    validate_source_run(config, history["workflow_runs"][0])


def validate_api_group(name, run, commit, inventory, expected_run, expected_head, tree=None):
    need(name in binder.GROUPS
         and type(run.get("id")) is int and run["id"] == expected_run
         and run.get("head_sha") == expected_head == commit.get("sha")
         and type(run.get("run_attempt")) is int and run["run_attempt"] == 1
         and run.get("status") == "completed" and run.get("conclusion") == "success"
         and run.get("repository", {}).get("full_name") == REPO
         and run["repository"].get("private") is False,
         "fixed final public original API group " + name)
    need(type(inventory.get("total_count")) is int
         and type(inventory.get("artifacts")) is list
         and inventory["total_count"] == len(inventory["artifacts"])
         and type(commit.get("files")) is list and len(commit["files"]) < 100,
         "untruncated whole original artifact and commit inventories")
    if name in ("train", "source"):
        tree_id = commit["commit"]["tree"]["sha"]
        need(type(tree_id) is str and re.fullmatch("[0-9a-f]{40}", tree_id)
             and type(tree) is dict and tree.get("sha") == tree_id
             and tree.get("truncated") is False and type(tree.get("tree")) is list,
             "complete recursive tree from the actual original tree ID")
    else:
        need(tree is None, "trees apply only to original TRAIN and SOURCE producers")


def check_checkout(config):
    raw = {key: (ROOT / path).read_bytes() for key, path in HASH_FILES.items()}
    need(all(profile.sha(raw[key]) == config[key] for key in HASH_FILES),
         "checkout retains every explicitly reviewed original file byte")
    need(profile.document(evaluator.SPEC.read_bytes()).get("scoring_allowed") is False
         and profile.document(evaluator.PRODUCER.read_bytes()).get("frozen") is False,
         "replica input registration remains inert and cannot be retargeted")
    return raw


def validate_binding_links(config, raw, bound, primary, primary_eval_raw, primary_binding_raw):
    need(set(raw) == set(HASH_FILES)
         and all(profile.sha(raw[key]) == config[key] for key in HASH_FILES)
         and profile.document(raw["replication_proof_binding_sha256"]) == bound
         and profile.document(primary_binding_raw) == primary,
         "same exact configured replica proof and original PRIMARY binding bytes")
    need(bound["panel"] == "replication" and primary["panel"] == "primary"
         and bound["status"] == primary["status"] == "BOUND_COMPLETE_ORIGINAL_TRAIN"
         and bound["scientific_execution_authorized"] is False
         and primary["scientific_execution_authorized"] is False
         and all(bound[key] == primary[key] for key in (
             "training_lock", "files", "transport", "training_report_sha256", "assessment_sha256"))
         and bound["training_report_sha256"] == TRAIN_REPORT_SHA
         and len(bound["files"]) == 41
         and set(bound["primary_quality_files"]) == PRIMARY_FILES,
         "complete same-candidate TRAIN41 and exact original PRIMARY18")
    lock = bound["primary_quality_lock"]
    need(type(lock) is dict and set(lock) == quality.LOCK_FIELDS
         and lock["schema"] == quality.LOCK_SCHEMA and lock["panel"] == "primary"
         and lock["inputs_sha256"] == config["primary_inputs_sha256"]
         and lock["producer_manifest_sha256"] == config["primary_producer_manifest_sha256"]
         and lock["evaluator_sha256"] == profile.sha(primary_eval_raw)
         and lock["external_proof_binding_sha256"] == profile.sha(primary_binding_raw)
         and profile.valid_hash(bound["primary_quality_report_sha256"]),
         "actual original PRIMARY lock, evaluator, inputs, producer and external proof")
    for role in ("result", "producer", "consumed"):
        quality.valid_artifact(lock[role], panel="primary", role=role)
    need(len({lock[role]["id"] for role in ("result", "producer", "consumed")}) == 3
         and all(lock[role][field] == lock["result"][field]
                 for role in ("producer", "consumed") for field in ("run_id", "run_head")),
         "three distinct original PRIMARY outputs from the same first run")
    spec = profile.document(raw["primary_inputs_sha256"])
    producer = profile.document(raw["primary_producer_manifest_sha256"])
    need(spec["panel"] == "primary"
         and spec["schema"] == "english-v6-b27-fresh-primary-four-frozen-models-v1"
         and spec["scoring_allowed"] is True
         and spec["producer_manifest_sha256"] == config["primary_producer_manifest_sha256"]
         and spec["external_proof_binding_sha256"] == lock["external_proof_binding_sha256"]
         and spec["model_pt_sha256"] == bound["training_lock"]["training"]["model_pt_sha256"]
         and spec["assessment_sha256"] == bound["assessment_sha256"]
         and producer["schema"] == "english-v6-b27-primary-producer-v1"
         and producer["frozen"] is True,
         "already registered same-candidate original PRIMARY files")
    profile.file_pins(bound["files"], profile.LIMITS["primary_external_proof"])
    profile.file_pins(bound["primary_quality_files"], profile.LIMITS["replication_external_proof"])
    profile.validate_transport(bound["transport"], bound["files"])
    profile.validate_transport(bound["primary_quality_transport"], bound["primary_quality_files"])
    need(sum(pin["bytes"] for pin in bound["files"].values())
         + sum(pin["bytes"] for pin in bound["primary_quality_files"].values())
         <= profile.LIMITS["replication_external_proof"],
         "unchanged total replica external-proof capacity")


def load_bound(config, raw):
    bound = profile.binding("replication")  # Metadata validation only, never require_proof().
    primary = profile.binding("primary")
    validate_binding_links(config, raw, bound, primary,
        (ROOT / "automation/english_v6_b27_primary_eval.py").read_bytes(),
        (ROOT / "automation/english_v6_b27_primary_proof_binding.json").read_bytes())
    return bound


def validate_screen(config, raw, bound):
    screen = profile.document(raw)
    identity = screen.get("source_execution", {})
    need(screen.get("schema") == evaluator.SCREEN_SCHEMA
         and screen.get("panel") == "replication"
         and screen.get("source_gate") == "SOURCE_GATE_PASS"
         and screen.get("source_shortfall") == {}
         and screen.get("assessment_sha256") == bound["assessment_sha256"]
         and identity.get("run_id") == str(config["source_run_id"])
         and identity.get("run_head") == config["source_run_head"]
         and identity.get("prepared_parent") == config["source_prepared_parent"]
         and identity.get("run_branch") == "refs/heads/" + evaluator.SOURCE_BRANCH
         and type(identity.get("run_attempt")) is int and identity["run_attempt"] == 1,
         "raw original replica SCREEN gate and same actual SOURCE identity")
    return screen  # Transport gate only; the frozen binder recomputes every source receipt.


def fetch_proofs(proof, bound):
    proof.mkdir()
    original_proof.fetch_files(proof / "training-origin", bound["files"], bound["transport"])
    original_proof.fetch_files(proof / "primary-quality",
                               bound["primary_quality_files"], bound["primary_quality_transport"])


def verify_outputs(config, generated, audit, screen_raw, bound, pins):
    spec_raw = (generated / EXPORT_PATHS[0]).read_bytes()
    producer_raw = (generated / EXPORT_PATHS[1]).read_bytes()
    audit_raw = (generated / "BINDING_AUDIT.json").read_bytes()
    spec = profile.document(spec_raw)
    need(profile.document(audit_raw) == audit
         and audit["schema"] == "english-v6-b27-original-quality-binding-audit-v1"
         and audit["panel"] == "replication"
         and audit["status"] == "BOUND_ORIGINAL_INPUTS_FOR_REVIEWED_ONCE_QUALITY"
         and audit["evaluator_sha256"] == config["replication_evaluator_sha256"]
         and profile.sha(spec_raw) == audit["registration_sha256"]
         and profile.sha(producer_raw) == spec["producer_manifest_sha256"]
             == audit["producer_manifest_sha256"]
         and profile.sha(screen_raw) == audit["source_audit"]["screen_sha256"]
         and audit["full_original_training_report_sha256"]
             == bound["training_report_sha256"] == TRAIN_REPORT_SHA,
         "exact frozen binder output bytes and complete original TRAIN identity")
    flattened = evaluator.archive_pins(spec)
    need(flattened == audit["input_artifacts"] == pins
         and spec["schema"] == evaluator.SCHEMA and spec["panel"] == "replication"
         and spec["scoring_allowed"] is True
         and spec["primary_quality_lock"] == bound["primary_quality_lock"]
         and spec["original_primary_pass_required"] is True and spec["replication_opened"] is True
         and spec["external_proof_binding_sha256"] == config["replication_proof_binding_sha256"]
         and spec["model_pt_sha256"] == audit["model_pt_sha256"]
             == bound["training_lock"]["training"]["model_pt_sha256"]
         and audit["all_original_screens_recomputed"] is True
         and audit["every_actual_upstream_producer_member_preserved"] is True
         and audit["execution_started"] is False
         and audit["promotion_allowed"] is audit["product_released"] is False
         and all(type(audit[key]) is int and audit[key] == 0 for key in (
             "new_model_calls", "new_source_calls", "new_observer_calls", "remote_mutations")),
         "review-only replica registration from one complete original binder invocation")
    return dict(zip(EXPORT_PATHS[:4], (spec_raw, producer_raw, audit_raw, screen_raw)))


def make_export(files):
    need(set(files) == set(EXPORT_PATHS)
         and all(type(raw) is bytes and raw for raw in files.values()),
         "exact five nonempty original-byte export files")
    total = sum(map(len, files.values()))
    need(total <= MAX_EXPORT_BYTES, "observation export exceeds 2,000,000 bytes")
    records, entries = [], []
    for path in EXPORT_PATHS:
        raw = files[path]
        encoded = base64.b64encode(raw).decode("ascii")
        count = (len(encoded) + CHUNK_CHARS - 1) // CHUNK_CHARS
        pin = dict(path=path, bytes=len(raw), sha256=profile.sha(raw), chunks=count)
        entries.append(pin)
        records.extend(dict(pin, index=index,
            base64=encoded[index * CHUNK_CHARS:(index + 1) * CHUNK_CHARS])
            for index in range(count))
    return records, dict(file_count=5, total_bytes=total, files=entries)


def prepare(config, work):
    config = validate_config(config)  # Before network, environment access or mkdir.
    work = Path(work)
    need(work.is_absolute() and work == work.resolve() and work.parent.is_dir()
         and not work.exists() and all(not path.is_symlink() for path in work.parents),
         "fresh absolute regular read-only binding workspace")
    raw = check_checkout(config)
    source_raw = evaluator.gh("actions/runs/" + str(config["source_run_id"]))
    validate_source_run(config, profile.document(source_raw))
    bound = load_bound(config, raw)
    training = bound["training_lock"]["training"]

    work.mkdir()
    bundle_root = work / "bundle"
    (bundle_root / "archives").mkdir(parents=True)
    groups = {}

    def api(group, key, endpoint=None, raw=None):
        body = evaluator.gh(endpoint) if raw is None else raw
        value = profile.document(body)
        relative = "api/" + group + "/" + key + ".json"
        target = bundle_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        groups.setdefault(group, {})[key] = relative
        return value

    origins = {
        **{name: (evaluator.BASELINES[name]["run_id"], evaluator.BASELINES[name]["run_head"])
           for name in ("B0", "B9")},
        "train": (training["run_id"], training["run_head"]),
        "source": (config["source_run_id"], config["source_run_head"]),
    }
    for name, (run_id, head) in origins.items():
        run = api(name, "run", "actions/runs/" + str(run_id),
                  source_raw if name == "source" else None)
        commit = api(name, "commit", "commits/" + head + "?per_page=100")
        inventory = api(name, "artifacts",
                        "actions/runs/" + str(run_id) + "/artifacts?per_page=100")
        tree = None
        if name in ("train", "source"):
            tree_id = commit["commit"]["tree"]["sha"]
            need(type(tree_id) is str and re.fullmatch("[0-9a-f]{40}", tree_id),
                 "actual original tree identity before transport")
            tree = api(name, "tree", "git/trees/" + tree_id + "?recursive=1")
        validate_api_group(name, run, commit, inventory, run_id, head, tree)
        if name == "source":
            validate_source_commit(config, commit)
    query = urlencode(dict(branch=evaluator.SOURCE_BRANCH, event="push", per_page=100))
    history = api("source", "workflow_runs",
        "actions/workflows/" + Path(evaluator.SOURCE_WORKFLOW).name + "/runs?" + query)
    validate_history(config, history)

    bundle = dict(schema="english-v6-b27-replication-original-quality-binding-inputs-v1",
        groups=groups, archives={role: "archives/" + role + ".zip" for role in binder.ROLES})
    bundle_path = bundle_root / "BUNDLE.json"
    bundle_path.write_bytes(profile.canonical(bundle))
    _, pins, _ = binder.metadata(bundle, bundle_root, evaluator, "replication")
    for role, original_role in (("pair", "models"), ("train_producer", "producer")):
        old = training["artifacts"][original_role]
        expected = {key: old[key] for key in ("id", "bytes", "sha256")}
        expected.update(run_id=training["run_id"], run_head=training["run_head"])
        need(pins[role] == expected, "same original B27 TRAIN artifact " + role)

    review = work / "review"
    raw_dir = review / "raw"
    raw_dir.mkdir(parents=True)
    shutil.copytree(bundle_root / "api", raw_dir / "api")
    shutil.copyfile(bundle_path, raw_dir / "BUNDLE.json")
    need(profile.inventory(raw_dir / "api") == profile.inventory(bundle_root / "api"),
         "every whole original API and recursive tree byte preserved")

    cache = {}

    def obtain(role):
        pin = pins[role]
        target = bundle_root / bundle["archives"][role]
        key = (pin["id"], pin["bytes"], pin["sha256"])
        if target.exists():
            body = target.read_bytes()
        elif key in cache:
            body = cache[key].read_bytes()
        else:
            body = evaluator.gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(body) == pin["bytes"] and profile.sha(body) == pin["sha256"],
             "whole original artifact bytes " + role)
        if not target.exists():
            target.write_bytes(body)
        cache[key] = target
        return target

    # The small whole original SCREEN gates the large proof acquisition.
    with evaluator.original_zip(obtain("screen"), pins["screen"], role="screen") as archive:
        screen_raw = archive.read("SCREEN.json")
    (raw_dir / "SCREEN.json").write_bytes(screen_raw)
    validate_screen(config, screen_raw, bound)
    proof = work / "proof"
    fetch_proofs(proof, bound)  # Only fetch_files; no fetch CLI or preliminary reader.
    for directory, files, transport in (
        ("training-origin", bound["files"], bound["transport"]),
        ("primary-quality", bound["primary_quality_files"], bound["primary_quality_transport"]),
    ):
        for name, location in transport.items():
            if "artifact_id" in location:
                pin = files[name]
                cache[(location["artifact_id"], pin["bytes"], pin["sha256"])] = proof / directory / name
    for role in binder.ROLES:
        obtain(role)
    archive_files = {role + ".zip": {key: pin[key] for key in ("bytes", "sha256")}
                     for role, pin in pins.items()}
    need(profile.inventory(bundle_root / "archives") == archive_files,
         "all and only the twelve complete original input ZIPs")

    # One invocation; retain all original internal TRAIN/PRIMARY verification.
    generated = work / "bound"
    audit = binder.bind("replication", bundle_path, proof, generated)
    files = verify_outputs(config, generated, audit, screen_raw, bound, pins)
    need(profile.inventory(bundle_root / "archives") == archive_files
         and profile.inventory(proof / "training-origin") == bound["files"]
         and profile.inventory(proof / "primary-quality") == bound["primary_quality_files"],
         "every original source, TRAIN and PRIMARY byte preserved after binding")
    need(not {"torch", "numpy", "onnxruntime", "piper", "transformers"} & set(sys.modules),
         "no scientific runtime during read-only binding")

    inventory = dict(schema="english-v6-b27-replication-binding-original-inventory-v1",
        source_run_id=config["source_run_id"], source_run_head=config["source_run_head"],
        source_prepared_parent=config["source_prepared_parent"],
        proof_binding_sha256=config["replication_proof_binding_sha256"],
        evaluator_sha256=config["replication_evaluator_sha256"],
        primary_inputs_sha256=config["primary_inputs_sha256"],
        primary_producer_manifest_sha256=config["primary_producer_manifest_sha256"],
        training_report_sha256=TRAIN_REPORT_SHA, training_origin_files=bound["files"],
        primary_quality_lock_sha256=profile.sha(profile.canonical(bound["primary_quality_lock"])),
        primary_quality_report_sha256=bound["primary_quality_report_sha256"],
        primary_quality_files=bound["primary_quality_files"],
        quality_input_artifacts=pins, raw_api_files=profile.inventory(raw_dir / "api"),
        bundle_sha256=profile.sha(bundle_path.read_bytes()),
        review_files={path: dict(bytes=len(body), sha256=profile.sha(body))
                      for path, body in files.items()},
        original_input_archives_repackaged=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0,
        remote_mutations=0, execution_started=False, scientific_execution_authorized=False,
        replication_quality_passed=False, promotion_allowed=False, product_released=False)
    files[EXPORT_PATHS[4]] = profile.canonical(inventory)
    records, manifest = make_export(files)
    export = review / "export"
    export.mkdir()
    for path, body in files.items():
        target = export / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    need(profile.inventory(export) ==
         {path: dict(bytes=len(body), sha256=profile.sha(body)) for path, body in files.items()},
         "exact five regular exported original-byte files")
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
    need(len(lineage) == 2, "exactly one binding Git parent")
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    validate_entry(env, event, lineage[0], lineage[1], changed)
    before = subprocess.run(["git", "-C", str(ROOT), "show", lineage[1] + ":" + WORKFLOW],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    validate_workflow_activation(before, (ROOT / WORKFLOW).read_bytes(), lineage[1])
    prepare(config, Path(sys.argv[1]))


if __name__ == "__main__":
    main()
