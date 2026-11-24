"""Bind the actual B22 candidate and independently admitted original primary panel.

Reads twelve authenticated saved ZIPs and API/Git metadata. Captures the union
of both complete original producer inventories, so dynamic dependencies are
never dropped. No source, model or remote call; no matched-control claim.
"""
from __future__ import annotations

import argparse

import builtins

from contextlib import ExitStack

import hashlib

import importlib.util

import json

from pathlib import Path

import sys

import tempfile

from unittest import mock

DRIVER_SHA = "a679d9a2410d55315c4ed93e4f93b3b632e79fe826b53788424004db5c5aeed8"

GROUPS = ("B0", "B9", "train", "source")

PROVIDERS = ("qwen", "whisper", "parakeet", "xlsr", "lv60")

ROLES = ("B0", "B9", "pair", "source", "screen", *("native_" + p for p in PROVIDERS),
         "train_producer", "source_producer")

GROUP_FOR = {**dict.fromkeys(("source", "screen", "source_producer", *("native_" + p for p in PROVIDERS)), "source"),
             "B0": "B0", "B9": "B9", "pair": "train", "train_producer": "train"}

def need(value, message):
    if not value:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(value, base):
    return value if isinstance(value, dict) else json.loads((base / value).read_bytes())

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def serialized(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

def metadata(bundle, base, primary):
    need(bundle["schema"] == "b22-original-historical-control-primary-binding-inputs-v1"
         and set(bundle["groups"]) == set(GROUPS) and set(bundle["archives"]) == set(ROLES),
         "four complete saved API groups and twelve exact original ZIP paths")
    groups = {name: {key: read(value, base) for key, value in item.items()}
              for name, item in bundle["groups"].items()}
    pins, evidence = {}, {}
    for role in ROLES:
        group = groups[GROUP_FOR[role]]
        run, commit = group["run"], group["commit"]
        items = group["artifacts"]["artifacts"]
        if role in primary.BASELINES:
            selected = [item for item in items if item["id"] == primary.BASELINES[role]["id"]]
        else:
            selected = [item for item in items if item["name"] == artifact_name(role, run["id"])]
        need(len(selected) == 1, "exactly one original artifact for " + role)
        item = selected[0]
        need(type(item.get("digest")) is str and item["digest"].startswith("sha256:"), "real GitHub artifact digest")
        pin = dict(id=item["id"], run_id=run["id"], run_head=run["head_sha"],
                   bytes=item["size_in_bytes"], sha256=item["digest"].removeprefix("sha256:"))
        primary.validate_provenance(pin, dict(run=run, commit=commit, artifact=item))
        if role in primary.BASELINES:
            need(pin == primary.BASELINES[role], "unchanged original baseline " + role)
        pins[role] = pin
        evidence[role] = {**group, "artifact": item}
    need(len({pin["id"] for pin in pins.values()}) == 12, "twelve distinct original artifacts")
    need(groups["train"]["run"]["event"] == groups["source"]["run"]["event"] == "push"
         and groups["train"]["run"]["head_branch"] == primary.TRAIN_BRANCH
         and groups["source"]["run"]["head_branch"] == primary.SOURCE_BRANCH,
         "actual separately registered train/source events")
    return groups, pins, evidence

def source_protection():
    stack = ExitStack()
    old_import = builtins.__import__
    denied = {"torch", "numpy", "torchaudio", "transformers", "piper", "soundfile", "onnxruntime", "requests", "httpx"}
    def guard(name, *args, **kwargs):
        need(name.split(".")[0] not in denied, "no runtime/network import while binding saved evidence: " + name)
        return old_import(name, *args, **kwargs)
    stack.enter_context(mock.patch("builtins.__import__", side_effect=guard))
    stack.enter_context(mock.patch("subprocess.Popen", side_effect=AssertionError("no process or network during saved binding")))
    return stack



def artifact_name(role, run_id):
    if role == "pair": return "english-v6-b22-union-retention-models-" + str(run_id)
    if role == "source": return "english-v6-b22-primary-source-" + str(run_id)
    if role == "screen": return "english-v6-b22-primary-frozen-screen-" + str(run_id)
    if role.startswith("native_"): return "b22-primary-dev-observer-" + role.removeprefix("native_")
    if role == "source_producer": return "english-v6-b22-primary-source-producer-" + str(run_id)
    if role == "train_producer": return "english-v6-b22-train-producer-" + str(run_id)
    raise ValueError("B0/B9 artifact selection is by its immutable original ID")


def bind(repository, bundle_path, output):
    repository, bundle_path, output = repository.resolve(), bundle_path.resolve(), output.resolve()
    need(not output.exists() and output.parent.is_dir(), "new output directory; original evidence and code stay unchanged")
    driver = repository / "automation/english_v6_b22_primary_eval.py"
    need(sha(driver.read_bytes()) == DRIVER_SHA, "exact independently reviewed B22 evaluator")
    primary = load(driver, "b22_original_binding_primary")
    spec = json.loads(primary.SPEC.read_bytes())
    need(spec["scoring_allowed"] is False and spec["producer_manifest_sha256"] is None,
         "only the inert registration may be bound")
    need(sha(primary.ASSESSMENT.read_bytes()) == primary.TRAINING_ASSESSMENT
         and sha(primary.TRAIN_DESIGN.read_bytes()) == primary.TRAINING_DESIGN, "prospective assessment and training design remain fixed")
    bundle = json.loads(bundle_path.read_bytes())
    groups, pins, meta = metadata(bundle, bundle_path.parent, primary)
    with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
        archives = {role: stack.enter_context(primary.original_zip(bundle_path.parent / bundle["archives"][role], pin))
                    for role, pin in pins.items()}
        snapshots = {role: primary.snapshot_contract(archives[role + "_producer"], role, meta[role + "_producer"], repository)
                     for role in ("train", "source")}
        spec["inputs"] = {key: pins[key] for key in primary.INPUT_KEYS}
        spec["native_inputs"] = {provider: pins["native_" + provider] for provider in PROVIDERS}
        spec["upstream_producers"] = {role: pins[role + "_producer"] for role in ("train", "source")}
        receipt = json.loads(archives["pair"].read("RESULT.json"))
        for arm in ("control", "challenger"):
            name = primary.STEMS[arm] + ".pt"
            spec["model_pt_sha256"][arm] = sha(archives["pair"].read(name))
        train = primary.validate_training_archive(archives["pair"], spec, meta["pair"])
        train_identity = snapshots["train"].get("identity", snapshots["train"])
        need(receipt["identity"]["prepared_parent"] == train_identity["prepared_parent"], "actual first TRAIN producer and receipt agree")
        source_dir, providers = Path(temporary) / "source", Path(temporary) / "providers"
        source_dir.mkdir()
        archives["source"].extractall(source_dir)
        for provider in PROVIDERS:
            directory = providers / provider
            directory.mkdir(parents=True)
            archives["native_" + provider].extractall(directory)
        screen = json.loads(archives["screen"].read("SCREEN.json"))
        with source_protection():
            _, _, source, plan = primary.helpers(spec)
            owner = source.evaluation_preflight(source_dir)
            source.verify_screen_contract(source_dir, screen, providers)
        execution = owner["source_execution"]
        source_identity = snapshots["source"].get("identity", snapshots["source"])
        need(str(execution["run_id"]) == str(pins["source"]["run_id"])
             and execution["run_head"] == pins["source"]["run_head"]
             and execution["prepared_parent"] == source_identity["prepared_parent"], "exact original SOURCE identity before binding")
        selected = primary.exact_admitted_join(owner["rows"], screen["rows"])
        reserve = {primary.person(row) for row in source.source_plan("replication")["rows"]}
        reserve.update(str(int(value)) for value in plan["replication"]["natural"]["speakers"])
        primary.reject_overlap(owner["all_original_rows"], train["base_train"] + train["natural_train"], owner["history"], reserve)
        counts = primary.source_counts(selected)
        need(counts == screen["source_coverage"] and all(counts[key] >= value for key, value in primary.GATE.items())
             and screen["source_gate"] == "SOURCE_GATE_PASS" and screen["source_shortfall"] == {}
             and 12 * len(selected) <= primary.MAX_FORWARDS, "original full population satisfies unchanged source minima")
        # The complete original producer inventories, not a guessed AST closure,
        # are the authority for every dynamically loaded module and resource.
        source_python = {name for name in snapshots["source"]["files"] if name.endswith(".py")}
        source_resources = {name for name in snapshots["source"]["files"]
                            if not name.endswith(".py") and name != primary.SOURCE_WORKFLOW}
        primary_python = primary.static_closure(repository / "automation", primary.PRODUCER_ROOTS)
        resources = set(source_resources)
        for role in ("train", "source"):
            primary_python |= {name for name in snapshots[role]["files"] if name.endswith(".py")}
            resources |= {name for name in snapshots[role]["files"]
                          if not name.endswith(".py") and name != snapshots[role]["runtime_workflow"]}
        need(primary.static_closure(repository / "automation", (primary.SOURCE_MODULE,)) <= source_python,
             "original source snapshot already contains every static and declared dynamic dependency")
        self_paths = {"automation/" + primary.SPEC.name, "automation/" + primary.PRODUCER.name,
                      "automation/" + primary.WORKFLOW_TEMPLATE.name, primary.WORKFLOW, primary.SOURCE_WORKFLOW}
        resources -= self_paths
        resources |= {"automation/english_v6_b14_panels/assessment.json", "automation/english_v6_b22_panels/assessment.json",
                      "automation/english_v6_b22_union_retention_design.json"}
        def hashes(names): return {name: sha((repository / name).read_bytes()) for name in sorted(names)}
        producer = dict(schema="english-v6-b22-primary-producer-v1", frozen=True,
            primary=dict(files=hashes(primary_python), workflow_path=primary.WORKFLOW,
                         workflow_template_sha256=sha(primary.WORKFLOW_TEMPLATE.read_bytes())),
            source=dict(files=hashes(source_python), resources=hashes(source_resources), workflow_path=primary.SOURCE_WORKFLOW,
                        workflow_sha256=snapshots["source"]["files"][primary.SOURCE_WORKFLOW]["sha256"]),
            resources=hashes(resources), upstream_manifests=snapshots)
        producer_raw = serialized(producer)
        spec.update(scoring_allowed=True, producer_manifest_sha256=sha(producer_raw))
        spec_file = Path(temporary) / primary.SPEC.name
        spec_file.write_bytes(serialized(spec))
        with mock.patch.object(primary, "SPEC", spec_file): primary.protocol()
        need("torch" not in sys.modules and "numpy" not in sys.modules, "no runtime imported during original evidence binding")
        output.mkdir()
        target = output / "automation"
        target.mkdir()
        (target / primary.SPEC.name).write_bytes(serialized(spec))
        (target / primary.PRODUCER.name).write_bytes(producer_raw)
        report = dict(schema="b22-original-historical-control-primary-binding-v1", status="BOUND_ORIGINAL_INPUTS_READY_FOR_REVIEWED_PRIMARY",
            evaluator_sha256=DRIVER_SHA, registration_sha256=sha(serialized(spec)), producer_manifest_sha256=sha(producer_raw),
            input_artifacts=pins, upstream_producer_manifests={role: sha(archives[role + "_producer"].read("PRODUCER.json")) for role in snapshots},
            original_source_rows=120, observer_eligible_rows=len(owner["rows"]), admitted_originals=len(selected), source_coverage=counts,
            planned_model_forwards=12 * len(selected), all_original_screens_recomputed=True,
            primary_python_files=len(primary_python), source_python_files=len(source_python), metadata_resources=len(resources),
            source_metadata_resources=len(source_resources), no_self_referential_metadata_hashes=True,
            entire_actual_upstream_manifests_inherited=True, same_job_matched_control=False, historical_control_retrained=False,
            model_comparison=primary.MODEL_COMPARISON, driver_modified=False, remote_mutations=0, new_model_calls=0,
            new_source_calls=0, new_observer_calls=0, primary_execution_started=False, replication_opened=False,
            promotion_allowed=False, product_released=False)
        (output / "BINDING_AUDIT.json").write_bytes(serialized(report))
        print(json.dumps(report, sort_keys=True))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bind(args.repository, args.bundle, args.output)
