"""Register only this new source batch on its development branch; no model runs.

The existing closed-file verifier stays unchanged. A successful source-test job
may submit the generated contents-API payload to this one unprotected branch.
The actual source HEAD, not an invented/future commit, owns each new file.
"""
from __future__ import annotations
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from intake_contract import CAMPAIGN, REPOSITORY, canonical, require

BRANCH = "research/" + CAMPAIGN
MANIFEST = "docs/BRANCH_RETENTION_20261007.json"
OWNED = {"research/" + CAMPAIGN + "/" + name for name in
         ("intake_contract.py", "plan.json", "run.py", "worker.py", "test_contract.py",
          "register_retention.py", "source-freeze.json", "README.md")}
OWNED |= {".github/workflows/qwen16-intake-20261030-v1.yml",
          ".github/workflows/qwen16-intake-contracts.yml"}


def register(value, tree, commit, previous_sha256):
    require(type(value) is dict and value.get("schema") == "kws-public-branch-retention-v1",
            "existing retention schema required")
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "real source commit required")
    rows = {row["path"]: row for row in value["retained_files"]}
    require(len(rows) == len(value["retained_files"]), "duplicate original retained path")
    integration = set(value["integration_files"])
    require(len(integration) == len(value["integration_files"]) and not integration & rows.keys(),
            "original inventory overlap")
    require(len(rows) == value["expected_retained_files"] and
            len(rows) + len(integration) == value["expected_tracked_files"], "original counts drift")
    for name, row in rows.items():
        require(tree.get(name) == (row["mode"], row["git_blob_sha1"], row["bytes"]),
                "original retained bytes changed: " + name)
    require((set(rows) | integration) <= set(tree), "existing tracked path removed")
    new_paths = set(tree) - set(rows) - integration
    require(new_paths == OWNED or (not new_paths and OWNED <= set(rows)),
            "only the exact new campaign source may be registered")
    if not new_paths:
        return copy.deepcopy(value), False
    result = copy.deepcopy(value)
    for name in sorted(new_paths):
        mode, blob, size = tree[name]
        require(mode in ("100644", "100755") and re.fullmatch(r"[0-9a-f]{40}", blob) and
                type(size) is int and size >= 0, "new Git blob identity")
        result["retained_files"].append({"path": name, "mode": mode,
            "git_blob_sha1": blob, "bytes": size, "source_commit": commit})
    result["expected_retained_files"] = len(result["retained_files"])
    result["expected_tracked_files"] = len(tree)
    result.setdefault("source_registration_history", []).append({
        "campaign": CAMPAIGN, "source_commit": commit, "paths": sorted(new_paths),
        "previous_manifest_sha256": previous_sha256,
        "scope": "new execution source only; zero generated or admitted recordings"})
    # Preserve all previous rows, provenance and integration exemptions exactly.
    require(result["retained_files"][:len(value["retained_files"])] == value["retained_files"],
            "original retained records changed")
    return result, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY and
            os.environ.get("GITHUB_EVENT_NAME") == "push" and
            os.environ.get("GITHUB_REF") == "refs/heads/" + BRANCH,
            "source registration only on the fixed development branch")
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(ROOT), *argv], timeout=30)
    commit = git("rev-parse", "HEAD").decode().strip()
    require(commit == os.environ.get("GITHUB_SHA") and not git("status", "--porcelain"),
            "exact clean source checkout required")
    tree = {}
    for item in git("ls-tree", "-r", "-l", "-z", "HEAD").split(b"\0"):
        if not item:
            continue
        metadata, name = item.split(b"\t", 1)
        mode, kind, blob, size = metadata.decode("ascii").split()
        require(kind == "blob", "no submodule registration")
        tree[name.decode("utf-8")] = (mode, blob, int(size))
    original = (ROOT / MANIFEST).read_bytes()
    value = json.loads(original)
    updated, changed = register(value, tree, commit, hashlib.sha256(original).hexdigest())
    if not changed:
        print("Source already registered; no write requested")
        return
    blob = git("rev-parse", "HEAD:" + MANIFEST).decode().strip()
    payload = {"message": "chore: retain exact Qwen16 source identities before integration",
               "branch": BRANCH, "sha": blob,
               "content": base64.b64encode((json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode()).decode()}
    args.payload.write_bytes(canonical(payload) + b"\n")
    print(json.dumps({"source_commit": commit, "new_paths": len(OWNED),
                      "retained_before": len(value["retained_files"]),
                      "retained_after": len(updated["retained_files"]), "model_calls": 0}))


if __name__ == "__main__":
    main()
