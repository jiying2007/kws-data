"""Capture the exact Git-bound replication source producer; no source access."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_producer_snapshot as common
import english_v6_b21_replication_source_ancestry as lineage

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("english_v6_b21_replication_source", "test_english_v6_b21_replication_sources",
         "english_v6_b21_replication_source_producer", "english_v6_b21_fixed_mirror_quality_readback",
         "english_v6_b21_fixed_mirror_primary_eval", "english_v6_b21_replication_transport",
         "test_english_v6_b21_replication_transport",
         "test_english_v6_b21_natural_amendment_replication",
         "english_v6_b21_natural_amendment_source")


def validate_event(prepared, head, parents, changed, env, event):
    common.need(re.fullmatch(r"[0-9a-f]{40}", prepared or "")
         and re.fullmatch(r"[0-9a-f]{40}", head or "") and parents == [prepared]
         and head != prepared and changed == [lineage.WORKFLOW], "one reviewed predecessor and workflow-only activation")
    common.need(all(env.get(k) == v for k, v in dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=lineage.REPO,
         GITHUB_EVENT_NAME="push", GITHUB_REF=lineage.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA=head).items())
         and re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", "")), "actual first public replication source checkout")
    common.need(event.get("before") == prepared and event.get("after") == head and event.get("ref") == lineage.BRANCH
         and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == lineage.REPO and event["repository"].get("private") is False,
         "exact nonforced public push before producer capture")
    return dict(role="replication_source", run_id=env["GITHUB_RUN_ID"], run_head=head, run_attempt=1,
        run_branch=lineage.BRANCH.removeprefix("refs/heads/"), prepared_parent=prepared, runtime_workflow=lineage.WORKFLOW)


def capture(output, prepared, root=ROOT):
    # The sealed/default opening fails before Git payload capture or archive access.
    lineage.execution_guard(prepared)
    root, output = Path(root).resolve(), Path(output)
    head = common.git(root, "rev-parse", "HEAD").decode().strip()
    parents = common.git(root, "rev-list", "--parents", "-n", "1", "HEAD").decode().split()[1:]
    changed = common.git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").decode().splitlines()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
    identity = validate_event(prepared, head, parents, changed, os.environ, event)
    python = common.prior.static_closure(root / "automation", ROOTS)
    resources = common.resource_paths(root, python) | {
        "automation/english_v6_b21_replication_opening.json",
        "automation/english_v6_b21_fixed_mirror_transport.json",
        "automation/english_v6_b21_replication_source_workflow_template.yml"}
    resources |= set(lineage.amended_primary.selection.resource_paths(root))
    resources |= set(lineage.amended_primary.selection.FROZEN_PATHS.values())
    resources |= set(lineage.amended_primary.auth_origin.resource_paths(root))
    template = (root / "automation/english_v6_b21_replication_source_workflow_template.yml").read_bytes()
    common.need((root / lineage.WORKFLOW).read_bytes() == template.replace(
        b"__B21_REPLICATION_SOURCE_MAIN_SHA__", prepared.encode()),
        "exact reviewed replication source workflow with only parent substitutions")
    # The actual primary source now carries a byte-frozen transport-stop proof.
    # Its ZIP/log payloads are required resources, outside the old text suffix scan.
    failure = root / "automation/english_v6_b21_continuation_failure"
    failure_files = {str(p.relative_to(root)) for p in failure.rglob("*") if p.is_file()}
    common.need(len(failure_files) == 10
         and str((failure / "MANIFEST.json").relative_to(root)) in failure_files,
         "complete explicit original zero-source transport proof resources")
    resources |= failure_files
    probe = root / "automation/english_v6_b21_continuation_probe"
    probe_files = {str(p.relative_to(root)) for p in probe.rglob("*") if p.is_file()}
    common.need(probe_files == {str((probe / name).relative_to(root)) for name in
         ("original.zip", "api/run.json", "api/artifact.json", "original/RESULT.json")},
         "complete explicit original official-mirror probe resources")
    resources |= probe_files
    second_failure = root / "automation/english_v6_b21_fixed_mirror_failure"
    second_files = {str(p.relative_to(root)) for p in second_failure.rglob("*") if p.is_file()}
    common.need(len(second_files) == 10
         and str((second_failure / "MANIFEST.json").relative_to(root)) in second_files,
         "complete actual fixed-mirror transport provenance resources")
    resources |= second_files
    paths = python | resources | {lineage.WORKFLOW}
    tracked = common.tree_entries(common.git(root, "ls-tree", "-rz", "--full-tree", "HEAD"), paths)
    common.need(output.is_absolute() and output.parent.is_dir() and not output.exists(), "fresh producer output, no overwrite")
    files, payloads = {}, {}
    for name in sorted(paths):
        path = root / name
        common.need(name in tracked and path.is_file() and not path.is_symlink()
             and path.resolve().is_relative_to(root), "regular tracked producer source " + name)
        raw = path.read_bytes()
        common.need(common.blob(raw) == tracked[name]["git_blob_sha1"], "working source differs from actual Git HEAD " + name)
        files[name] = dict(**tracked[name], bytes=len(raw), sha256=common.sha(raw)); payloads[name] = raw
    output.mkdir()
    for name, raw in payloads.items():
        dest = output / "repository" / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
    manifest = dict(schema="english-v6-b21-original-git-producer-snapshot-v1", **identity,
        git_tree=common.git(root, "rev-parse", "HEAD^{tree}").decode().strip(), files=files,
        python_files=sorted(python), resource_files=sorted(resources), runtime_workflow_preserved_hidden=True,
        source_code_reads_only=True, network_calls=0, new_source_calls=0, new_model_forwards=0,
        optimizer_updates=0, promotion_allowed=False)
    raw = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    (output / "PRODUCER.json").write_bytes(raw)
    print(json.dumps(dict(role=identity["role"], files=len(files), bytes=sum(v["bytes"] for v in files.values()),
        manifest_sha256=common.sha(raw), run_head=head, prepared_parent=prepared, new_model_forwards=0)))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path); parser.add_argument("--prepared-parent", required=True)
    args = parser.parse_args(); capture(args.output, args.prepared_parent)
