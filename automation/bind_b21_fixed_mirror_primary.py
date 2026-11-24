"""Bind original completed B21 evidence to the unchanged primary evaluator.

Offline only. Reads saved GitHub API documents and twelve original ZIPs,
including an explicitly amended natural source panel and all96 exact unscored
synthetic originals from a consumed metadata STOP. Both prior zero-call
transport failures and the later consumed STOP remain required and preserved.
Recomputes source admission, and writes two metadata files plus an audit.
No remote call, model import, inference, workflow activation or driver edit.
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

DRIVER_SHA = "ca14214dbfe3946e737806aec6cfc099e01e7985cb7d38b9031fd9b641200434"
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


def artifact_name(role, run_id):
    if role == "pair": return "english-v6-b21-competing-margin-weights-" + str(run_id)
    if role == "source": return "english-v6-b21-natural-amendment-source-" + str(run_id)
    if role == "screen": return "english-v6-b21-natural-amendment-frozen-screen-" + str(run_id)
    if role.startswith("native_"): return "b21-natural-amendment-dev-observer-" + role.removeprefix("native_")
    if role == "source_producer": return "english-v6-b21-natural-amendment-source-producer-" + str(run_id)
    if role.endswith("_producer"): return "english-v6-b21-" + role.removesuffix("_producer") + "-producer-" + str(run_id)
    raise ValueError("B0/B9 artifact selection is by its fixed original ID")


def metadata(bundle, base, primary):
    need(bundle["schema"] == "b21-original-natural-amendment-primary-binding-inputs-v1"
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


def snapshot(archive, role, meta, primary, repository):
    doc = json.loads(archive.read("PRODUCER.json"))
    run, commit = meta["run"], meta["commit"]
    workflow = ".github/workflows/english-v6-b21-" + ("blank-retention-train-once.yml" if role == "train" else "natural-amendment-sources-once.yml")
    branch = primary.TRAIN_BRANCH if role == "train" else primary.SOURCE_BRANCH
    tree = meta["tree"]
    commit_tree = commit.get("commit", commit)["tree"]["sha"]
    if "files" in commit:
        changes = [item["filename"] for item in commit["files"]]
    else:
        comparison = meta["comparison"]
        comparison_head = comparison.get("head") or comparison["commits"][-1]["sha"]
        need(comparison["base_commit"]["sha"] == doc["prepared_parent"]
             and comparison_head == run["head_sha"]
             and comparison["total_commits"] == comparison["ahead_by"] == 1
             and comparison["behind_by"] == 0, "original parent-to-child Git comparison")
        changes = [item["filename"] for item in comparison["files"]]
    need(doc["schema"] == "english-v6-b21-original-git-producer-snapshot-v1" and doc["role"] == ("source_natural_amendment" if role == "source" else role)
         and str(doc["run_id"]) == str(run["id"]) and doc["run_head"] == run["head_sha"] == commit["sha"]
         and doc["run_attempt"] == 1 and doc["run_branch"] == branch and doc["runtime_workflow"] == workflow
         and doc["git_tree"] == commit_tree == tree["sha"] and tree["truncated"] is False
         and len(commit["parents"]) == 1 and doc["prepared_parent"] == commit["parents"][0]["sha"]
         and changes == [workflow] and run["path"] == workflow
         and doc["runtime_workflow_preserved_hidden"] is doc["source_code_reads_only"] is True
         and doc["network_calls"] == doc["new_source_calls"] == doc["new_model_forwards"] == doc["optimizer_updates"] == 0,
         "actual pre-execution Git-bound producer snapshot " + role)
    expected = {"PRODUCER.json", *("repository/" + name for name in doc["files"])}
    need({item.filename for item in archive.infolist() if not item.is_dir()} == expected
         and set(doc["files"]) == set(doc["python_files"]) | set(doc["resource_files"]) | {workflow},
         "complete original producer upload including every hidden file")
    for name, pin in doc["files"].items():
        raw = archive.read("repository/" + name)
        entries = [item for item in tree["tree"] if item["path"] == name]
        need(pin["mode"] in ("100644", "100755") and len(raw) == pin["bytes"]
             and sha(raw) == pin["sha256"] and primary.git_blob(raw) == pin["git_blob_sha1"],
             "original producer content/Git blob identity " + name)
        need(len(entries) == 1 and entries[0]["type"] == "blob"
             and entries[0]["mode"] == pin["mode"] and entries[0]["size"] == pin["bytes"]
             and entries[0]["sha"] == pin["git_blob_sha1"], "actual authenticated Git tree member " + name)
        if name != workflow:
            need((repository / name).is_file() and (repository / name).read_bytes() == raw,
                 "prepared scorer retains identical original producer source/resource " + name)
    actual_workflow = archive.read("repository/" + workflow)
    template_path = repository / ("automation/english_v6_b21_natural_amendment_source_auth_workflow_template.yml" if role == "source" else workflow)
    template = template_path.read_bytes()
    token = b"__B21_NATURAL_AMENDMENT_AUTH_MAIN_SHA__" if role == "source" else b"__B21_REVIEWED_TRAIN_PARENT__"
    need(template.count(token) == 2 and template.replace(token, doc["prepared_parent"].encode()) == actual_workflow,
         "exact reviewed upstream workflow, only its two parent substitutions")
    return doc


def validate_transport_origin(origin, authenticated):
    """Require the complete newly verified nested proof, never the first stop alone."""
    need(type(origin) is dict and type(authenticated) is dict and origin == authenticated,
         "source owner matches both independently authenticated original failure proofs")
    required = {"original_transport_origin", "failed_continuation_run_id", "failed_continuation_run_head",
                "failed_continuation_prepared_parent", "failed_continuation_job_id",
                "failed_continuation_manifest_sha256", "failed_continuation_producer_files",
                "failed_continuation_actual_tts_calls", "failed_continuation_natural_original_decodes",
                "failed_continuation_observer_calls", "failed_continuation_kws_forwards",
                "failed_continuation_consumption_started", "failed_continuation_source_generation_started",
                "failed_continuation_preserved", "live_refresh_required_for_both_failed_runs"}
    need(required <= set(origin), "both original failure identities and zero-call proofs are mandatory")
    need(origin["original_run_id"] == 45352507569
         and origin["original_run_head"] == "d6db2375632adc8a627d094fcf23c5e87b22eddd"
         and origin["original_prepared_parent"] == "60f1e6507216ae605d601f3f09ab15177b46aae2"
         and origin["original_job_id"] == 138302172086 and origin["original_run_attempt"] == 1
         and origin["source_job_log_sha256"] == "c9ecd7399ef7b7b243ea243becfa9b07fbd1ecba402195fc61019581c7db62ac"
         and all(type(origin[key]) is int and origin[key] == 0 for key in
                 ("actual_tts_calls", "natural_original_decodes", "observer_calls", "kws_forwards"))
         and origin["source_generation_started"] is False
         and origin["transport_continuation"] is origin["original_opened_marker_preserved"] is True,
         "exact first zero-call failure remains preserved")
    first = origin["original_transport_origin"]
    need(type(first) is dict and first
         and all(origin.get(key) == value for key, value in first.items())
         and origin["failed_continuation_run_id"] == 45365292922
         and origin["failed_continuation_run_head"] == "2da5cbe0d616f96b21a1bb5665613b9ce8d1c788"
         and origin["failed_continuation_prepared_parent"] == "3144028b5db5a7f1314654858b8d4f1343b8cf40"
         and origin["failed_continuation_job_id"] == 138345323477
         and origin["failed_continuation_manifest_sha256"] == "a1a0d2acf630fd8b23e773e1e7906401787a8a75cbca81fca2de08dd1d54c4ca"
         and origin["failed_continuation_producer_files"] == 246
         and all(type(origin[key]) is int and origin[key] == 0 for key in
                 ("failed_continuation_actual_tts_calls", "failed_continuation_natural_original_decodes",
                  "failed_continuation_observer_calls", "failed_continuation_kws_forwards"))
         and origin["failed_continuation_consumption_started"] is False
         and origin["failed_continuation_source_generation_started"] is False
         and origin["failed_continuation_preserved"] is origin["live_refresh_required_for_both_failed_runs"] is True,
         "latest zero-call continuation is separately authenticated; first origin alone cannot authorize")
    return origin


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


def bind(repository, bundle_path, output):
    repository, bundle_path, output = repository.resolve(), bundle_path.resolve(), output.resolve()
    need(not output.exists() and output.parent.is_dir(), "new output directory only; leave repository and driver untouched")
    driver = repository / "automation/english_v6_b21_fixed_mirror_primary_eval.py"
    need(sha(driver.read_bytes()) == DRIVER_SHA, "unchanged independently reviewed final evaluator")
    primary = load(driver, "b21_final_binding_primary")
    helper = load(repository / "automation/english_v6_b21_producer_snapshot.py", "b21_binding_closure")
    spec = json.loads(primary.SPEC.read_bytes())
    need(spec["scoring_allowed"] is False and spec["producer_manifest_sha256"] is None,
         "only the original inert registration may be bound")
    need(sha(primary.ASSESSMENT.read_bytes()) == primary.TRAINING_ASSESSMENT
         and sha(primary.TRAIN_DESIGN.read_bytes()) == primary.TRAINING_DESIGN, "final shared preexecution registrations")
    bundle = json.loads(bundle_path.read_bytes())
    groups, pins, meta = metadata(bundle, bundle_path.parent, primary)
    with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
        archives = {role: stack.enter_context(primary.original_zip(bundle_path.parent / bundle["archives"][role], pin))
                    for role, pin in pins.items()}
        snapshots = {role: snapshot(archives[role + "_producer"], role, meta[role + "_producer"], primary, repository)
                     for role in ("train", "source")}
        spec["inputs"] = {key: pins[key] for key in primary.INPUT_KEYS}
        spec["native_inputs"] = {provider: pins["native_" + provider] for provider in PROVIDERS}
        receipt = json.loads(archives["pair"].read("RESULT.json"))
        need(str(receipt["run_id"]) == str(pins["pair"]["run_id"]) and receipt["head"] == pins["pair"]["run_head"]
             and receipt["parent"] == snapshots["train"]["prepared_parent"], "original completed pair execution")
        for arm in ("control", "challenger"):
            for extension in ("pt", "npz", "onnx"):
                name = primary.STEMS[arm] + "." + extension
                raw = archives["pair"].read(name)
                need(receipt[arm]["files"][Path(name).name] == dict(bytes=len(raw), sha256=sha(raw)), "actual trained model file " + name)
                if extension == "pt": spec["model_pt_sha256"][arm] = sha(raw)
        primary.validate_pair_receipt(receipt, spec)
        train = json.loads(archives["pair"].read("SOURCE.json"))
        need(train["design_sha256"] == primary.TRAINING_DESIGN
             and train["independent_assessment_sha256"] == primary.TRAINING_ASSESSMENT
             and train["parent_pt_sha256"] == primary.B0_PT
             and train["forbidden_sources_opened"] is False, "actual original TRAIN-only source registration")
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
        origin = owner["transport_origin"]
        with source_protection():
            failure = load(repository / "automation/english_v6_b21_fixed_mirror_origin.py",
                           "b21_binding_fixed_mirror_origin")
            authenticated_origin = failure.verify()
        validate_transport_origin(origin, authenticated_origin)
        spec["transport_continuation"]["verified_transport_origin"] = origin
        execution = owner["source_execution"]
        need(str(execution["run_id"]) == str(pins["source"]["run_id"])
             and execution["run_head"] == pins["source"]["run_head"]
             and execution["prepared_parent"] == snapshots["source"]["prepared_parent"], "exact source origin before binding")
        selected = primary.exact_admitted_join(owner["rows"], screen["rows"])
        reserve = {primary.person(row) for row in source.source_plan("replication")["rows"]}
        reserve.update(str(int(value)) for value in plan["replication"]["natural"]["speakers"])
        primary.reject_overlap(owner["all_original_rows"], train["base_train"] + train["natural_train"], owner["history"], reserve)
        counts = primary.source_counts(selected)
        need(counts == screen["source_coverage"] and all(counts[key] >= value for key, value in primary.GATE.items())
             and 12 * len(selected) <= primary.MAX_FORWARDS, "original source population meets unchanged preregistered gate")
        source_python = primary.static_closure(repository / "automation", (primary.SOURCE_MODULE,))
        primary_python = primary.static_closure(repository / "automation", primary.PRODUCER_ROOTS)
        need(source_python <= set(snapshots["source"]["files"]), "original source producer contains complete scoring dependency closure")
        self_paths = {"automation/" + primary.SPEC.name, "automation/" + primary.PRODUCER.name,
                      "automation/" + primary.WORKFLOW_TEMPLATE.name, primary.WORKFLOW, primary.SOURCE_WORKFLOW}
        source_resources = set(snapshots["source"]["resource_files"]) - self_paths
        origin_directory = repository / "automation/english_v6_b21_continuation_failure"
        origin_resources = {path.relative_to(repository).as_posix() for path in origin_directory.rglob("*") if path.is_file()}
        need(len(origin_resources) == 10 and origin_resources <= source_resources,
             "actual source producer retains all nine original failure payloads and manifest, including both original ZIPs")
        probe_directory = repository / "automation/english_v6_b21_continuation_probe"
        probe_resources = {path.relative_to(repository).as_posix() for path in probe_directory.rglob("*") if path.is_file()}
        need(len(probe_resources) == 4 and probe_resources <= source_resources
             and "automation/english_v6_b21_continuation_transport.json" in source_resources,
             "original official-mirror probe ZIP/receipts/result and exact transport registration remain in both producer closures")
        fixed_failure_directory = repository / "automation/english_v6_b21_fixed_mirror_failure"
        fixed_failure_resources = {path.relative_to(repository).as_posix()
                                   for path in fixed_failure_directory.rglob("*") if path.is_file()}
        need(len(fixed_failure_resources) == 10 and fixed_failure_resources <= source_resources
             and "automation/english_v6_b21_fixed_mirror_transport.json" in source_resources,
             "complete second failure proof, original ZIPs and fixed mirror registration remain in producer")
        resources = (helper.resource_paths(repository, primary_python) | source_resources) - self_paths
        def hashes(paths): return {name: sha((repository / name).read_bytes()) for name in sorted(paths)}
        producer = dict(schema="english-v6-b21-primary-producer-v1", frozen=True,
            primary=dict(files=hashes(primary_python), workflow_path=primary.WORKFLOW,
                         workflow_template_sha256=sha(primary.WORKFLOW_TEMPLATE.read_bytes())),
            source=dict(files=hashes(source_python), resources=hashes(source_resources), workflow_path=primary.SOURCE_WORKFLOW,
                        workflow_sha256=snapshots["source"]["files"][primary.SOURCE_WORKFLOW]["sha256"]), resources=hashes(resources))
        producer_raw = serialized(producer)
        spec.update(scoring_allowed=True, producer_manifest_sha256=sha(producer_raw))
        # Pure protocol validation against temporary metadata; no guard or runtime is called.
        spec_file = Path(temporary) / primary.SPEC.name
        spec_file.write_bytes(serialized(spec))
        with mock.patch.object(primary, "SPEC", spec_file): primary.protocol()
        need("torch" not in sys.modules and "numpy" not in sys.modules, "saved binding imported no model/array runtime")
        output.mkdir()
        target = output / "automation"
        target.mkdir()
        (target / primary.SPEC.name).write_bytes(serialized(spec))
        (target / primary.PRODUCER.name).write_bytes(producer_raw)
        report = dict(schema="b21-completed-original-natural-amendment-primary-binding-v1", status="BOUND_ORIGINAL_INPUTS_READY_FOR_REVIEWED_PRIMARY",
            evaluator_sha256=DRIVER_SHA, registration_sha256=sha(serialized(spec)), producer_manifest_sha256=sha(producer_raw),
            input_artifacts=pins, upstream_producer_manifests={role: sha(archives[role + "_producer"].read("PRODUCER.json")) for role in snapshots},
            original_source_rows=120, observer_eligible_rows=len(owner["rows"]), admitted_originals=len(selected), source_coverage=counts,
            planned_model_forwards=12 * len(selected), all_original_screens_recomputed=True,
            transport_origin=origin, natural_amendment=owner["natural_amendment"],
            consumed_natural_metadata_stop_preserved=True, exact_synthetic_originals_reused=96,
            original_failed_source_attempt_preserved=True,
            failed_continuation_attempt_preserved=True, both_zero_call_failures_authenticated=True,
            primary_python_files=len(primary_python), source_python_files=len(source_python), metadata_resources=len(resources),
            source_metadata_resources=len(source_resources), no_self_referential_metadata_hashes=True,
            driver_modified=False, remote_mutations=0, new_model_calls=0, new_source_calls=0, new_observer_calls=0,
            primary_execution_started=False, replication_opened=False, promotion_allowed=False, product_released=False)
        (output / "BINDING_AUDIT.json").write_bytes(serialized(report))
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bind(args.repository, args.bundle, args.output)
