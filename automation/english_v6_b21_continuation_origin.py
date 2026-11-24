"""Exact stdlib proof of the one B21 failure before any original was generated.

This authorizes only separate, explicitly reviewed transport continuation of
run 45352507569. It never retries a run or opens a model, corpus or source.
An absent source artifact alone is insufficient: the original Git-bound shell,
exact curl28 log, complete job graph, producer and opened marker must all agree.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

FAILURE_DIR = Path(__file__).with_name("english_v6_b21_continuation_failure")
MANIFEST_SHA256 = "8e4745c640a6729361d70f73f20b5ad18c0972ebcd108fb1d5ca258f22479aeb"
RUN = 45352507569
HEAD = "d6db2375632adc8a627d094fcf23c5e87b22eddd"
PREPARED = "60f1e6507216ae605d601f3f09ab15177b46aae2"
TREE = "e92016adacc41e7b856763522ac71dfe0368b785"
JOB = 138302172086
REPO = "jiying2007/kws-data"
BRANCH = "research/en-v6-b21-primary-sources-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-primary-sources-once.yml"
WORKFLOW_SHA = "2df77efb41f847eb393110e82121fee7132885f871e44f9e000e775faf5e8809"
ASSESSMENT_SHA = "65ad9c7204755f1cd0b639added163387af39604818f9bd26b9da9fe8bd098fc"
ANCESTRY_SHA = "26cf48813b5e712bc8820a2429fc0917db9f6e3eaa809a008d98c533714be2b0"
LOG_SHA = "c9ecd7399ef7b7b243ea243becfa9b07fbd1ecba402195fc61019581c7db62ac"
PRODUCER = dict(id=14900722688, run_id=RUN, run_head=HEAD, bytes=1260477,
    sha256="36cc252052a0f31240866312a1cae13dee5b90cebe719bbbb4922d9134b0fc4e")
OPENED = dict(id=14901610526, run_id=RUN, run_head=HEAD, bytes=2504,
    sha256="69d64b227e0b2ca611006a16fff7419accfe833a2981114446c9756cc0ff897b")
FILES = {"api/run-final.json", "api/jobs-final.json", "api/artifacts-final.json", "api/commit.json",
    "api/prepared_main_comparison.json", "api/tree.json", "api/source-job.log",
    "producer-original.zip", "opened-original.zip"}


def need(ok, reason):
    if not ok: raise ValueError(reason)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def blob(raw): return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def execution_records(run, jobs, artifacts):
    """Same immutable evidence constraints for frozen snapshots and raw live APIs."""
    need(run["id"] == RUN and run["head_sha"] == HEAD and run["run_attempt"] == 1
         and run["status"] == "completed" and run["conclusion"] == "failure" and run["event"] == "push"
         and run["head_branch"] == BRANCH and run["path"] == WORKFLOW
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and run["head_repository"]["full_name"] == REPO and run["head_repository"]["private"] is False
         and run["head_commit"]["id"] == HEAD and run["head_commit"]["tree_id"] == TREE,
         "only the exact completed original public first-attempt failure is eligible")
    by_name = {j["name"]: j for j in jobs}
    need(len(jobs) == len(by_name) == 3 and set(by_name) == {"sources", "observers", "readback"}
         and all(j["status"] == "completed" and j["run_id"] == RUN for j in jobs),
         "complete three-job source graph; no running or unknown downstream job")
    source = by_name["sources"]
    need(source["id"] == JOB and source["conclusion"] == "failure", "exact original source job failed")
    for name, ident in (("observers", 138313787118), ("readback", 138313787042)):
        job = by_name[name]
        need(job["id"] == ident and job["conclusion"] == "skipped" and not job.get("steps"),
             "no native observer or source readback started")
    expected = [
        (1, "Set up job", "success"),
        (2, "Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "success"),
        (3, "Run actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1", "success"),
        (4, "Capture exact Git-bound scientific producer", "success"),
        (5, "Run actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", "success"),
        (6, "Authenticate complete previous source history before new original access", "success"),
        (7, "Run actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", "success"),
        (8, "Generate96synthetic and retain24whole natural originals exactly once", "failure"),
        (9, "Run actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", "success"),
        (17, "Post Run actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1", "skipped"),
        (18, "Post Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "success"),
        (19, "Complete job", "success")]
    need([(s["number"], s["name"], s["conclusion"]) for s in source["steps"]] == expected
         and all(s["status"] == "completed" for s in source["steps"]),
         "exact source step8 stop and successful no-file upload; no unknown execution path")
    pins = {p["id"]: p for p in (PRODUCER, OPENED)}
    need(len(artifacts) == 2 and {a["id"] for a in artifacts} == set(pins),
         "only original producer and preserved opened marker; any source/native/screen artifact forbids continuation")
    for item in artifacts:
        pin = pins[item["id"]]
        expected_name = "en-v6-b21-primary-panel-opened" if item["id"] == OPENED["id"] else "english-v6-b21-source-producer-45352507569"
        need(item["name"] == expected_name and item["expired"] is False
             and item["size_in_bytes"] == pin["bytes"] and item["digest"] == "sha256:" + pin["sha256"]
             and item["workflow_run"]["id"] == RUN and item["workflow_run"]["head_sha"] == HEAD
             and item["workflow_run"]["head_branch"] == BRANCH,
             "exact original producer and opened-marker API identity")


def validate_remote(run, jobs, artifacts):
    """Require complete raw GitHub API pages; this helper performs no network I/O."""
    need(jobs.get("total_count") == len(jobs.get("jobs", [])) == 3
         and artifacts.get("total_count") == len(artifacts.get("artifacts", [])) == 2,
         "complete live original-run inventories; no first-page or summary-only inference")
    execution_records(run, jobs["jobs"], artifacts["artifacts"])
    return dict(original_run_id=RUN, original_run_head=HEAD, original_job_id=JOB,
        original_run_attempt=1, complete_jobs=3, complete_artifacts=2,
        original_opened_marker_preserved=True, no_source_or_native_artifact=True)


def original_archive(raw, pin):
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "exact original ZIP bytes")
    archive = zipfile.ZipFile(io.BytesIO(raw)); items = archive.infolist()
    need(len(items) == len({i.filename for i in items}) and archive.testzip() is None,
         "complete unique original ZIP")
    for item in items:
        path = PurePosixPath(item.filename)
        need(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
             and not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), "regular safe original ZIP members")
    return archive


def verify(directory=FAILURE_DIR):
    directory = Path(directory)
    manifest_raw = (directory / "MANIFEST.json").read_bytes()
    need(len(manifest_raw) == 1693 and sha(manifest_raw) == MANIFEST_SHA256, "reviewed immutable nine-file original failure proof")
    manifest = json.loads(manifest_raw)
    need(manifest["schema"] == "english-v6-b21-fixed-zero-source-transport-failure-proof-v1"
         and manifest["original_run_id"] == RUN and manifest["original_run_head"] == HEAD
         and manifest["original_prepared_parent"] == PREPARED and manifest["original_job_id"] == JOB
         and set(manifest["files"]) == FILES, "one explicitly identified original failure; no generalized retry gate")
    found = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    need(found == FILES | {"MANIFEST.json"} and not directory.is_symlink()
         and not any(p.is_symlink() for p in directory.rglob("*")), "all and only complete original proof payloads")
    payloads = {}
    for name, pin in manifest["files"].items():
        raw = (directory / name).read_bytes()
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "unchanged original evidence " + name)
        payloads[name] = raw
    doc = lambda name: json.loads(payloads["api/" + name + ".json"])
    run, jobs, artifacts, commit, comparison, tree = (doc(n) for n in
        ("run-final", "jobs-final", "artifacts-final", "commit", "prepared_main_comparison", "tree"))
    # These exact connector snapshots normalize away total_count; their entire
    # fixed bytes and full known graph are checked. Live refresh uses the stricter
    # raw-API validate_remote entry point above.
    execution_records(run, jobs["jobs"], artifacts["artifacts"])
    need(commit["sha"] == HEAD and [p["sha"] for p in commit["parents"]] == [PREPARED]
         and commit["tree"]["sha"] == tree["sha"] == TREE and tree["truncated"] is False,
         "actual original single reviewed parent and complete exact Git tree")
    need(comparison["status"] == "ahead" and comparison["ahead_by"] == comparison["total_commits"] == 1
         and comparison["behind_by"] == 0 and comparison["base_commit"]["sha"] == PREPARED
         and comparison["merge_base_commit"]["sha"] == PREPARED
         and [c["sha"] for c in comparison["commits"]] == [HEAD]
         and [(f["filename"], f["status"]) for f in comparison["files"]] == [(WORKFLOW, "modified")],
         "original activation changed only the frozen runtime workflow")
    entries = {item["path"]: item for item in tree["tree"]}
    need(len(entries) == len(tree["tree"]), "unique original full Git tree")
    with original_archive(payloads["producer-original.zip"], PRODUCER) as archive:
        proof_raw = archive.read("PRODUCER.json"); proof = json.loads(proof_raw)
        need(proof["schema"] == "english-v6-b21-original-git-producer-snapshot-v1"
             and proof["role"] == "source" and proof["run_id"] == str(RUN) and proof["run_head"] == HEAD
             and proof["prepared_parent"] == PREPARED and proof["run_attempt"] == 1
             and proof["run_branch"] == BRANCH and proof["runtime_workflow"] == WORKFLOW and proof["git_tree"] == TREE
             and proof["runtime_workflow_preserved_hidden"] is proof["source_code_reads_only"] is True
             and proof["network_calls"] == proof["new_source_calls"] == proof["new_model_forwards"] == proof["optimizer_updates"] == 0
             and proof["promotion_allowed"] is False and len(proof["files"]) == 221,
             "exact original scientific producer, including hidden actual workflow")
        need(set(proof["files"]) == set(proof["python_files"]) | set(proof["resource_files"]) | {WORKFLOW}
             and set(archive.namelist()) == {"PRODUCER.json"} | {"repository/" + n for n in proof["files"]},
             "complete original runtime dependency and resource snapshot")
        for name, pin in proof["files"].items():
            raw = archive.read("repository/" + name); entry = entries[name]
            need(entry["type"] == "blob" and entry["mode"] == pin["mode"] in ("100644", "100755")
                 and entry["sha"] == pin["git_blob_sha1"] == blob(raw)
                 and pin["bytes"] == len(raw) and pin["sha256"] == sha(raw), "actual original Git bytes " + name)
        workflow = archive.read("repository/" + WORKFLOW)
        template = archive.read("repository/automation/english_v6_b21_source_workflow_template.yml")
        need(sha(workflow) == WORKFLOW_SHA
             and workflow == template.replace(b"__B21_SOURCE_MAIN_SHA__", PREPARED.encode()),
             "exact original source shell and guarded workflow, never reconstructed after failure")
        assessment = archive.read("repository/automation/english_v6_b21_panels/assessment.json")
        ancestry = archive.read("repository/automation/english_v6_b21_panels/ancestry.json")
        need(sha(assessment) == ASSESSMENT_SHA and sha(ancestry) == ANCESTRY_SHA,
             "same prospective scientific assessment and original source history")
    with original_archive(payloads["opened-original.zip"], OPENED) as archive:
        need(archive.namelist() == ["AUTHORIZATION.json"], "opened marker preserves only its original authorization")
        authorization = json.loads(archive.read("AUTHORIZATION.json"))
        need(authorization["schema"] == "english-v6-b21-complete-source-history-v1"
             and authorization["assessment_sha256"] == ASSESSMENT_SHA and authorization["ancestry_sha256"] == ANCESTRY_SHA
             and authorization["source_execution"] == dict(run_id=str(RUN), run_head=HEAD, run_attempt=1,
                 run_branch="refs/heads/" + BRANCH, prepared_parent=PREPARED)
             and authorization["history_sha256"] == sha((json.dumps(json.loads(ancestry), indent=2, sort_keys=True) + "\n").encode())
             and authorization["all_prior_replication_sealed"] is True and authorization["new_KWS_forwards"] == 0
             and authorization["promotion_allowed"] is False,
             "preserved original opened marker keeps the failed run identity and frozen history")
    log_raw = payloads["api/source-job.log"]
    need(len(log_raw) == 175274 and sha(log_raw) == LOG_SHA, "exact original connector-decoded log bytes, BOM preserved")
    log = log_raw.decode("utf-8", errors="strict")
    stop = "curl: (28) Operation timed out after 1200000 milliseconds with 1363383941 out of 6387309499 bytes received"
    no_source = "No files were found with the provided path: /home/runner/work/_temp/source/. No artifacts will be uploaded."
    need(log.count(stop) == 1 and log.count("##[error]Process completed with exit code 28.") == 1
         and log.count(no_source) == 1 and log.index(stop) < log.index(no_source),
         "original transport failure before absent source output; no ambiguous source exception")
    shell = workflow.decode("utf-8")
    command = 'curl --fail --location --connect-timeout 30 --max-time 1200 --retry 0 https://www.openslr.org/resources/12/train-clean-100.tar.gz --output "$RUNNER_TEMP/train-clean-100.tar.gz"'
    verify_command = 'python -IB automation/english_v6_b21_source.py verify-corpus "$RUNNER_TEMP/train-clean-100.tar.gz"'
    generate_command = 'python -IB automation/english_v6_b21_source.py generate "$RUNNER_TEMP/source" "$RUNNER_TEMP/prior-source" "$RUNNER_TEMP/train-clean-100.tar.gz" "$PREPARED_PARENT"'
    at = shell.index(command)
    need(shell.count(command) == shell.count(verify_command) == shell.count(generate_command) == 1
         and shell.rfind("set -euo pipefail", 0, at) > shell.rfind("Generate96synthetic", 0, at)
         and at < shell.index(verify_command) < shell.index(generate_command)
         and "shell: /usr/bin/bash --noprofile --norc -e -o pipefail {0}" in log,
         "reviewed fail-fast shell places full-corpus transport before verification and generation")
    # The log prints all planned shell commands before executing the block.
    # Its printed generate line is not execution evidence. The byte-frozen
    # set-e block aborts at curl28, and the no-file upload/skipped jobs agree.
    return dict(original_run_id=RUN, original_run_head=HEAD, original_prepared_parent=PREPARED,
        original_job_id=JOB, original_run_attempt=1, original_producer_pin=PRODUCER,
        original_opened_pin=OPENED, source_job_log_sha256=LOG_SHA,
        failure_proof_manifest_sha256=MANIFEST_SHA256, original_workflow_sha256=WORKFLOW_SHA,
        actual_tts_calls=0, natural_original_decodes=0, observer_calls=0, kws_forwards=0,
        source_generation_started=False, transport_continuation=True, original_opened_marker_preserved=True)


if __name__ == "__main__": print(json.dumps(verify(), indent=2, sort_keys=True))
