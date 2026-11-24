"""Capture joint amended source producer and every frozen source dependency."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_producer_snapshot as common
import english_v6_b21_natural_amendment_source_ancestry as lineage
import english_v6_b21_fixed_mirror_source_producer as frozen_producer

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("english_v6_b21_natural_amendment_source", "english_v6_b21_natural_amendment_source_producer",
         "english_v6_b21_natural_amendment_transport", "test_english_v6_b21_natural_amendment_sources",
         "test_english_v6_b21_natural_amendment", "test_english_v6_b21_reuse_synthetic",
         "test_english_v6_b21_natural_amendment_auth_origin", "test_english_v6_b21_natural_amendment_auth_continuation")


def validate_event(prepared, head, parents, changed, env, event):
    common.need(re.fullmatch(r"[0-9a-f]{40}", prepared or "")
         and re.fullmatch(r"[0-9a-f]{40}", head or "") and parents == [prepared]
         and head != prepared and changed == [lineage.WORKFLOW], "one reviewed predecessor and workflow-only activation")
    common.need(all(env.get(k) == v for k, v in dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=lineage.REPO,
         GITHUB_EVENT_NAME="push", GITHUB_REF=lineage.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA=head).items())
         and re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", "")), "actual first public continuation source checkout")
    common.need(event.get("before") == prepared and event.get("after") == head and event.get("ref") == lineage.BRANCH
         and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == lineage.REPO and event["repository"].get("private") is False,
         "exact nonforced public push before producer capture")
    return dict(role="source_natural_amendment", run_id=env["GITHUB_RUN_ID"], run_head=head, run_attempt=1,
        run_branch=lineage.BRANCH.removeprefix("refs/heads/"), prepared_parent=prepared, runtime_workflow=lineage.WORKFLOW)


def capture(output, prepared, root=ROOT):
    # Exact original zero-call failure is authenticated before new producer capture.
    lineage.execution_guard(prepared)
    root, output = Path(root).resolve(), Path(output)
    head = common.git(root, "rev-parse", "HEAD").decode().strip()
    parents = common.git(root, "rev-list", "--parents", "-n", "1", "HEAD").decode().split()[1:]
    changed = common.git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").decode().splitlines()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
    identity = validate_event(prepared, head, parents, changed, os.environ, event)
    python = common.prior.static_closure(root / "automation", ROOTS + frozen_producer.ROOTS)
    resources = common.resource_paths(root, python) | {
        "automation/english_v6_b21_fixed_mirror_transport.json",
        "automation/english_v6_b21_fixed_mirror_source_workflow_template.yml",
        "automation/english_v6_b21_natural_amendment_source_workflow_template.yml",
        "automation/english_v6_b21_natural_amendment_source_auth_workflow_template.yml",
        "automation/english_v6_b21_natural_amendment_escape_fixture.bin",
        lineage.fixed.WORKFLOW}
    resources |= set(lineage.selection.resource_paths(root))
    resources |= set(lineage.selection.FROZEN_PATHS.values()) - python
    resources |= set(lineage.auth_origin.resource_paths(root))
    for name in lineage.fixed.verify_frozen_producer(root):
        (python if name.endswith(".py") else resources).add(name)
    for folder in ("english_v6_b21_continuation_failure", "english_v6_b21_continuation_probe",
                   "english_v6_b21_fixed_mirror_failure"):
        retained = {p.relative_to(root).as_posix() for p in (root / "automation" / folder).rglob("*") if p.is_file()}
        common.need(bool(retained), "every original ZIP/log/metadata proof resource " + folder)
        resources |= retained
    template = (root / "automation/english_v6_b21_natural_amendment_source_auth_workflow_template.yml").read_bytes()
    common.need((root / lineage.WORKFLOW).read_bytes() == template.replace(b"__B21_NATURAL_AMENDMENT_AUTH_MAIN_SHA__", prepared.encode()),
        "actual continuation workflow differs from reviewed template only by exact parent substitution")
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
