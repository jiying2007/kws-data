"""Fixed public-hosted launcher for one new, unadmitted 16-recording batch.

Run only after protected source integration and offline contract acceptance.
It stages retained loader/containment code, not any old one-shot runner.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from intake_contract import (BUDGET, CAMPAIGN, DONOR_COMMIT, canonical, expected_plan,
                      file_sha256, read_json, require, save_json, terminal_ledger,
                      validate_launch, validate_plan, verify_completed_outputs)


# Only these public, bounded outputs may be uploaded. Runtime, model and setup
# directories, arbitrary exception messages and unrecognized files are excluded.
OUTPUT_JSON = {"generation-receipt.json", "kernel-scope.json", "progress.json",
               "runtime-observations.json"}
HOST_JSON = {"host-receipt.json", "image-transfer.json", "container-setup.json",
             "container-tts.json", "plan.json", "ledger.json"}


def git(directory: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(directory), *args], check=True,
                          capture_output=True, text=True, timeout=60).stdout.strip()


def verify_source(donor: Path, expected_sha: str) -> dict:
    require(git(ROOT, "rev-parse", "HEAD") == expected_sha, "source checkout mismatch")
    current_main = git(ROOT, "ls-remote", "https://github.com/jiying2007/kws-data.git", "refs/heads/main")
    require(current_main == expected_sha + "\trefs/heads/main", "execution must equal current reviewed main")
    # The donor is a second sibling checkout, not an untracked directory in ROOT.
    require(not git(ROOT, "status", "--porcelain", "--untracked-files=all"),
            "source checkout must be clean")
    require(git(donor, "rev-parse", "HEAD") == DONOR_COMMIT, "retained donor commit mismatch")
    require(not git(donor, "status", "--porcelain", "--untracked-files=all"),
            "retained donor checkout must be clean")
    freeze = read_json(HERE / "source-freeze.json")
    require(freeze.get("schema") == "qwen16-intake-source-freeze-v1" and
            freeze.get("campaign") == CAMPAIGN, "source freeze identity")
    require(freeze.get("donor_commit") == DONOR_COMMIT, "donor freeze identity")
    expected_names = {"intake_contract.py", "run.py", "worker.py", "test_contract.py", "plan.json", "register_retention.py"}
    require(type(freeze.get("files")) is dict and set(freeze["files"]) == expected_names,
            "complete fixed source membership")
    for name, digest in freeze["files"].items():
        path = HERE / name
        require(not path.is_symlink() and path.is_file() and file_sha256(path) == digest,
                "campaign source drift")
    validate_plan(read_json(HERE / "plan.json"))
    return freeze


def pack_artifact(work: Path, output: Path, host_receipt: dict) -> Path:
    receipt_file = output / "generation-receipt.json"
    try:
        receipt = read_json(receipt_file) if receipt_file.exists() else {"outcomes": []}
        verify_completed_outputs(output, receipt)
        ledger = terminal_ledger(output, receipt)
    except (ValueError, OSError, TypeError, KeyError) as error:
        # Keep a failure-only artifact even when receipt/audio integrity fails.
        # No unverified output is reclassified as successfully generated.
        host_receipt.update(status="failed_evidence_validation", artifact_complete=False,
                            evidence_error_type=type(error).__name__)
        save_json(work / "host-receipt.json", host_receipt)
        destination = work / "artifact"
        destination.mkdir(exist_ok=False)
        shutil.copyfile(work / "host-receipt.json", destination / "host-receipt.json")
        save_json(destination / "INCOMPLETE.json", {
            "campaign": CAMPAIGN, "status": "UNVERIFIED_OUTPUTS_NOT_PUBLISHED",
            "source_commit": host_receipt["source_commit"], "run_id": host_receipt["run_id"],
            "attempts_unknown": True, "retry_allowed": False, "training_admitted": False})
        raise
    save_json(work / "ledger.json", {"campaign": CAMPAIGN, "cells": ledger,
                                     "training_admitted": False})
    generated = sum(row["status"] == "GENERATED_PENDING_REVIEW" for row in ledger)
    host_receipt["generated_pending_review"] = generated
    host_receipt["consumed_cells"] = sum(row["attempts"] for row in ledger)
    host_receipt["not_run_cells"] = sum(row["status"] == "NOT_RUN" for row in ledger)
    if host_receipt["status"] == "complete_pending_review":
        require(generated == 16 and receipt.get("status") == "generated_16_pending_review",
                "completed stage without complete batch")
    save_json(work / "host-receipt.json", host_receipt)
    destination = work / "artifact"
    destination.mkdir(exist_ok=False)
    sources: dict[str, Path] = {}
    for name in HOST_JSON:
        path = work / name
        if path.exists():
            sources[name] = path
    names = set(OUTPUT_JSON)
    for row in expected_plan():
        names.update(row["cell_id"] + suffix for suffix in
                     (".claim", ".raw-float.wav", ".wav"))
    for name in names:
        path = output / name
        if path.exists() or path.is_symlink():
            sources["output/" + name] = path
    total = 0
    manifest = {}
    for relative, source in sorted(sources.items()):
        require(not source.is_symlink() and source.is_file() and source.stat().st_nlink == 1,
                "unaliased artifact file required")
        size = source.stat().st_size
        require(size <= 8 * 1024**2, "per-file evidence cap")
        total += size
        require(total <= BUDGET["artifact_bytes"] - 65536, "whole evidence cap")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[relative] = {"bytes": target.stat().st_size, "sha256": file_sha256(target)}
    save_json(destination / "artifact-freeze.json", {
        "schema": "qwen16-intake-artifact-freeze-v1", "campaign": CAMPAIGN,
        "files": manifest, "source_commit": host_receipt["source_commit"],
        "source_freeze_sha256": host_receipt["source_freeze_sha256"],
        "run_id": host_receipt["run_id"], "run_attempt": 1,
        "labels_verified": False, "training_admitted": False,
        "qualification_allowed": False})
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", required=True, type=Path)
    parser.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    validate_launch(env, read_json(Path(env["GITHUB_EVENT_PATH"])), args.expected_source_sha)
    donor = args.donor.resolve()
    freeze = verify_source(donor, args.expected_source_sha)
    work = Path(env["RUNNER_TEMP"]) / CAMPAIGN
    work.mkdir(exist_ok=False)  # Never resume or erase an interrupted attempt.
    runtime, code, output = work / "runtime", work / "code", work / "output"
    setup_output = work / "setup-output"
    for path in (runtime, code, output, setup_output):
        path.mkdir()
    started = time.monotonic()
    host_receipt = {"schema": "qwen16-intake-host-v1", "campaign": CAMPAIGN,
                    "source_commit": args.expected_source_sha, "donor_commit": DONOR_COMMIT,
                    "source_freeze_sha256": file_sha256(HERE / "source-freeze.json"),
                    "run_id": env["GITHUB_RUN_ID"], "run_attempt": 1,
                    "status": "started", "budget": BUDGET,
                    "training_admitted": False, "asr_calls": 0, "training_runs": 0,
                    "stages": {"setup": "NOT_RUN", "tts": "NOT_RUN"}}
    save_json(work / "host-receipt.json", host_receipt)
    save_json(work / "plan.json", expected_plan())
    primary = None
    try:
        require(shutil.disk_usage(work).free >= 16 * 1024**3, "16 GiB setup headroom required")
        # Read-only import of retained host containment helpers; no old main().
        sys.path.insert(0, str(donor / "research/source-screen32-v1"))
        import hosted_run as retained
        host_receipt["capacity"] = retained.host_capacity()
        image_lock = read_json(donor / "research/source-screen32-v1/container-lock.json")
        retained.pull_image(image_lock, work / "image-transfer.json")
        research = code / "research"
        research.mkdir()
        ignored = shutil.ignore_patterns("__pycache__", "evidence", "history", "runtime")
        for name in ("source-screen32-v1", "tts-quality-pilot-v1", "qwen6_tts", "qwen6_asr"):
            shutil.copytree(donor / "research" / name, research / name, ignore=ignored)
        shutil.copytree(HERE, research / CAMPAIGN, ignore=ignored)
        adapter = research / "tts-quality-pilot-v1"
        shutil.copyfile(adapter / "qwen06-model-lock.json", adapter / "qwen-model-lock.json")
        for stage in ("setup", "tts"):
            retained.host_capacity()  # Recheck before each actual scope creation.
            name = CAMPAIGN + "-" + env["GITHUB_RUN_ID"] + "-" + stage
            stage_output = setup_output if stage == "setup" else output
            argv = retained.container_args(name, image_lock["image"], code,
                                           runtime, stage_output, "tts", stage)
            image_index = argv.index(image_lock["image"])
            argv[image_index:image_index] = ["--env", "TQP_ENGINE=qwen06"]
            tail_index = argv.index(image_lock["image"]) + 1
            if stage == "setup":
                argv[tail_index:] = ["python", "-I", "-B",
                    "/code/research/tts-quality-pilot-v1/setup_adapter.py",
                    "--profile", "tts", "--stage", "setup", "--runtime", "/runtime"]
            else:
                argv[tail_index:] = ["/runtime/venv/bin/python", "-I", "-B",
                                    "/code/research/" + CAMPAIGN + "/worker.py"]
            host_receipt["stages"][stage] = "STARTED"
            save_json(work / "host-receipt.json", host_receipt)
            cap = BUDGET["setup_seconds"] if stage == "setup" else BUDGET["generation_stage_seconds"]
            host_receipt["stages"][stage] = retained.supervise_container(
                argv, name, stage_output, stage, min(started + 6000, time.monotonic() + cap))
        host_receipt["status"] = "complete_pending_review"
    except BaseException as error:
        primary = error
        host_receipt.update(status="stopped_no_retry", error_type=type(error).__name__)
        for stage, state in host_receipt["stages"].items():
            if state == "STARTED":
                host_receipt["stages"][stage] = "FAILED_NO_RETRY"
    finally:
        host_receipt["wall_seconds"] = time.monotonic() - started
        try:
            artifact = pack_artifact(work, output, host_receipt)
            if env.get("GITHUB_OUTPUT"):
                with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
                    stream.write("artifact_root=" + str(artifact) + "\n")
        except BaseException as error:
            if primary is None:
                primary = error
            # Never conceal a stage failure with a later packaging exception.
            print(canonical({"artifact_packaging_error": type(error).__name__}).decode())
    if primary is not None:
        raise primary


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(canonical({"status": "stopped_no_retry", "error_type": type(error).__name__}).decode())
        raise SystemExit(1)
