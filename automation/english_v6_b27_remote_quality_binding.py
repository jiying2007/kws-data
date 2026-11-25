"""Read-only B27 PRIMARY input binding; no source/model execution or remote writes."""
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
sys.path.insert(0, str(ROOT / "automation"))
import bind_b27_quality as binder
import english_v6_b27_original_proof as original_proof
import english_v6_b27_primary_eval as evaluator
import english_v6_b27_profile as profile

REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-b27-primary-binding-ci-once-20261125"
WORKFLOW = ".github/workflows/english-v6-b27-remote-quality-binding.yml"
SOURCE_RUN = 45597047117
SOURCE_HEAD = "7429fee92456bb04ada5187be04aadbb5b18d826"
TRAIN_REPORT_SHA = "cea6637193b1071710ab7ec3676b7226fc00f017403b78a5f051a2650a2ccc93"
EVIDENCE = "research/english_v6_b27_primary_binding_20261125"
EXPORT_PATHS = (
    "automation/english_v6_b27_primary_inputs.json",
    "automation/english_v6_b27_primary_producer.json",
    EVIDENCE + "/BINDING_AUDIT.json",
    EVIDENCE + "/SCREEN.json",
    EVIDENCE + "/ORIGINAL_INVENTORY.json",
)
MAX_EXPORT_BYTES = 2_000_000
CHUNK_CHARS = 32768
need = profile.need


def validate_entry(env, event, head, parent, changed):
    need(all(env.get(k) == v for k, v in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": BRANCH,
        "GITHUB_RUN_ATTEMPT": "1",
    }.items()), "only the first public push on the read-only binding branch")
    prepared = env.get("PREPARED_PARENT", "")
    need(re.fullmatch("[0-9a-f]{40}", prepared) is not None
         and head == env.get("GITHUB_SHA")
         and parent == prepared and head != parent
         and changed == [WORKFLOW],
         "one workflow-only child of the reviewed prepared parent")
    need(event.get("before") == parent and event.get("after") == head
         and event.get("ref") == BRANCH
         and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event["repository"].get("private") is False,
         "actual nonforced first public workflow-only activation")


def validate_source_run(run):
    need(run.get("id") == SOURCE_RUN
         and run.get("head_sha") == SOURCE_HEAD
         and type(run.get("run_attempt")) is int and run["run_attempt"] == 1
         and run.get("head_branch") == evaluator.SOURCE_BRANCH
         and run.get("path") == evaluator.SOURCE_WORKFLOW
         and run.get("event") == "push"
         and run.get("status") == "completed"
         and run.get("conclusion") == "success"
         and run.get("repository", {}).get("full_name") == REPO
         and run["repository"].get("private") is False,
         "SOURCE is not the exact first completed successful public original")


def make_export(files):
    """Pure, bounded transport encoder; called only after every check succeeds."""
    need(set(files) == set(EXPORT_PATHS)
         and all(type(raw) is bytes and raw for raw in files.values()),
         "exact five nonempty original-byte export files")
    total = sum(map(len, files.values()))
    need(total <= MAX_EXPORT_BYTES, "five-file export exceeds 2,000,000 bytes")
    records, entries = [], []
    for path in EXPORT_PATHS:
        raw = files[path]
        encoded = base64.b64encode(raw).decode("ascii")
        count = (len(encoded) + CHUNK_CHARS - 1) // CHUNK_CHARS
        pin = dict(path=path, bytes=len(raw), sha256=profile.sha(raw), chunks=count)
        entries.append(pin)
        for index in range(count):
            records.append(dict(pin, index=index,
                base64=encoded[index * CHUNK_CHARS:(index + 1) * CHUNK_CHARS]))
    return records, dict(file_count=5, total_bytes=total, files=entries)


def prepare(work):
    work = Path(work)
    need(work.is_absolute() and work == work.resolve()
         and work.parent.is_dir() and not work.exists()
         and all(not p.is_symlink() for p in work.parents),
         "fresh absolute regular binding workspace")

    # The first network read is the final SOURCE run; no wait or scientific retry.
    source_raw = evaluator.gh("actions/runs/" + str(SOURCE_RUN))
    validate_source_run(profile.document(source_raw))

    bound = profile.binding("primary")  # Metadata/hash validation, not readback.
    need(bound["training_report_sha256"] == TRAIN_REPORT_SHA,
         "same already frozen complete B27 TRAIN report")
    training = bound["training_lock"]["training"]

    work.mkdir()
    bundle_root = work / "bundle"
    (bundle_root / "archives").mkdir(parents=True)
    groups = {}

    def api(group, key, endpoint=None, raw=None):
        body = evaluator.gh(endpoint) if raw is None else raw
        value = profile.document(body)
        relative = "api/" + group + "/" + key + ".json"
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)  # Preserve original API bytes.
        groups.setdefault(group, {})[key] = relative
        return value

    origins = {
        "B0": (evaluator.BASELINES["B0"]["run_id"], evaluator.BASELINES["B0"]["run_head"]),
        "B9": (evaluator.BASELINES["B9"]["run_id"], evaluator.BASELINES["B9"]["run_head"]),
        "train": (training["run_id"], training["run_head"]),
        "source": (SOURCE_RUN, SOURCE_HEAD),
    }
    for name, (run_id, head) in origins.items():
        run = api(name, "run", "actions/runs/" + str(run_id),
                  source_raw if name == "source" else None)
        need(run["id"] == run_id and run["head_sha"] == head,
             "fixed original API group identity")
        commit = api(name, "commit", "commits/" + head + "?per_page=100")
        inventory = api(name, "artifacts",
                        "actions/runs/" + str(run_id) + "/artifacts?per_page=100")
        need(inventory["total_count"] == len(inventory["artifacts"]),
             "complete unfiltered original artifact inventory")
        need(len(commit.get("files", [])) < 100,
             "commit file inventory must not be silently truncated")
        if name in ("train", "source"):
            tree_id = commit["commit"]["tree"]["sha"]
            tree = api(name, "tree", "git/trees/" + tree_id + "?recursive=1")
            need(tree["sha"] == tree_id and tree["truncated"] is False,
                 "complete real original producer tree")

    query = urlencode(dict(branch=evaluator.SOURCE_BRANCH, event="push", per_page=100))
    runs = api("source", "workflow_runs",
               "actions/workflows/" + Path(evaluator.SOURCE_WORKFLOW).name + "/runs?" + query)
    need(runs["total_count"] == len(runs["workflow_runs"]) == 1,
         "only the original SOURCE push on its dedicated branch")
    validate_source_run(runs["workflow_runs"][0])

    bundle = dict(
        schema="english-v6-b27-primary-original-quality-binding-inputs-v1",
        groups=groups,
        archives={role: "archives/" + role + ".zip" for role in binder.ROLES},
    )
    bundle_path = bundle_root / "BUNDLE.json"
    bundle_path.write_bytes(profile.canonical(bundle))
    _, pins, _ = binder.metadata(bundle, bundle_root, evaluator, "primary")
    for role, original_role in (("pair", "models"), ("train_producer", "producer")):
        original = training["artifacts"][original_role]
        expected = {key: original[key] for key in ("id", "bytes", "sha256")}
        expected.update(run_id=training["run_id"], run_head=training["run_head"])
        need(pins[role] == expected, "exact frozen B27 TRAIN artifact")

    reusable = {}

    def obtain(role):
        pin = pins[role]
        target = bundle_root / bundle["archives"][role]
        if target.exists():
            raw = target.read_bytes()
        elif pin["id"] in reusable:
            raw = reusable[pin["id"]].read_bytes()
        else:
            raw = evaluator.gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and profile.sha(raw) == pin["sha256"],
             "complete exact original artifact " + role)
        if not target.exists():
            target.write_bytes(raw)
        return target

    # Cheap gate before the large historical proof download. It does not replace
    # the binder's complete independent source/native SCREEN reconstruction.
    screen_path = obtain("screen")
    with evaluator.original_zip(screen_path, pins["screen"], role="screen") as archive:
        screen_raw = archive.read("SCREEN.json")
    screen = profile.document(screen_raw)
    need(screen["schema"] == evaluator.SCREEN_SCHEMA
         and screen["source_gate"] == "SOURCE_GATE_PASS"
         and screen["source_shortfall"] == {},
         "original SOURCE SCREEN has not passed its unchanged gate")

    proof = work / "proof"
    proof.mkdir()
    original_proof.fetch_files(
        proof / "training-origin", bound["files"], bound["transport"])
    for name, location in bound["transport"].items():
        if "artifact_id" in location:
            reusable[location["artifact_id"]] = proof / "training-origin" / name
    for role in binder.ROLES:
        obtain(role)

    # The single complete authentication. Do not call fetch(), require_proof(),
    # another preflight, or a separate training reader before/after this call.
    generated = work / "bound"
    audit = binder.bind("primary", bundle_path, proof, generated)

    spec_raw = (generated / EXPORT_PATHS[0]).read_bytes()
    producer_raw = (generated / EXPORT_PATHS[1]).read_bytes()
    audit_raw = (generated / "BINDING_AUDIT.json").read_bytes()
    spec = profile.document(spec_raw)
    need(profile.document(audit_raw) == audit
         and profile.sha(spec_raw) == audit["registration_sha256"]
         and profile.sha(producer_raw) == spec["producer_manifest_sha256"]
             == audit["producer_manifest_sha256"]
         and profile.sha(screen_raw) == audit["source_audit"]["screen_sha256"]
         and audit["full_original_training_report_sha256"]
             == bound["training_report_sha256"] == TRAIN_REPORT_SHA,
         "exact generated bytes agree with the original binding audit")
    flattened = {
        **spec["inputs"],
        **{"native_" + k: v for k, v in spec["native_inputs"].items()},
        **{k + "_producer": v for k, v in spec["upstream_producers"].items()},
    }
    need(flattened == audit["input_artifacts"] == pins
         and spec["scoring_allowed"] is True
         and audit["execution_started"] is False
         and audit["promotion_allowed"] is audit["product_released"] is False
         and all(audit[key] == 0 for key in (
             "new_model_calls", "new_source_calls", "new_observer_calls", "remote_mutations")),
         "review binding only, with all twelve authentic input pins")
    need(not {"torch", "numpy", "onnxruntime", "piper", "transformers"} & set(sys.modules),
         "no scientific runtime during remote binding")

    review = work / "review"
    export = review / "export"
    raw_dir = review / "raw"
    export.mkdir(parents=True)
    raw_dir.mkdir()
    shutil.copytree(bundle_root / "api", raw_dir / "api")
    shutil.copyfile(bundle_path, raw_dir / "BUNDLE.json")
    need(profile.inventory(raw_dir / "api") == profile.inventory(bundle_root / "api"),
         "every raw API and tree byte preserved in review artifact")

    files = dict(zip(EXPORT_PATHS[:4],
                     (spec_raw, producer_raw, audit_raw, screen_raw)))
    inventory = dict(
        schema="english-v6-b27-primary-binding-original-inventory-v1",
        source_run_id=SOURCE_RUN,
        source_run_head=SOURCE_HEAD,
        proof_binding_sha256=profile.sha(
            (profile.HERE / "english_v6_b27_primary_proof_binding.json").read_bytes()),
        training_report_sha256=TRAIN_REPORT_SHA,
        training_origin_files=bound["files"],
        quality_input_artifacts=pins,
        raw_api_files=profile.inventory(raw_dir / "api"),
        bundle_sha256=profile.sha(bundle_path.read_bytes()),
        review_files={path: dict(bytes=len(body), sha256=profile.sha(body))
                      for path, body in files.items()},
        original_input_archives_repackaged=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0,
        new_optimizer_updates=0, execution_started=False,
        promotion_allowed=False, product_released=False,
    )
    files[EXPORT_PATHS[4]] = profile.canonical(inventory)
    records, manifest = make_export(files)  # Check count/size before printing.
    for path, body in files.items():
        target = export / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    need(profile.inventory(export) ==
         {path: dict(bytes=len(body), sha256=profile.sha(body))
          for path, body in files.items()}, "exact five regular export files")

    for record in records:
        print("B27_REVIEW_CHUNK " + profile.canonical(record).decode(), flush=True)
    print("B27_REVIEW_EXPORT_COMPLETE " + profile.canonical(manifest).decode(), flush=True)


def main():
    need(len(sys.argv) == 2, "usage: helper ABSOLUTE_FRESH_WORK_DIRECTORY")
    env = os.environ
    event = profile.document(Path(env["GITHUB_EVENT_PATH"]).read_bytes())

    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, text=True).stdout.strip()

    lineage = git("rev-list", "--parents", "-n", "1", "HEAD").split()
    need(len(lineage) == 2, "exactly one Git parent")
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
    validate_entry(env, event, lineage[0], lineage[1], changed)
    prepare(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
