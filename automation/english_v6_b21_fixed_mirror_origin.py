"""Exact proof of B21's second transport failure before any source invocation.

Only run45365292922 is recognized. The first curl28 proof remains nested and
unchanged. Fixed-mirror authorization must freshly validate the live APIs for
both failed runs; this module itself never performs network or source calls.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_continuation_origin as original

need, sha, blob = original.need, original.sha, original.blob
FAILURE_DIR = Path(__file__).with_name("english_v6_b21_fixed_mirror_failure")
MANIFEST_SHA256 = "a1a0d2acf630fd8b23e773e1e7906401787a8a75cbca81fca2de08dd1d54c4ca"
RUN = 45365292922
HEAD = "2da5cbe0d616f96b21a1bb5665613b9ce8d1c788"
PREPARED = "3144028b5db5a7f1314654858b8d4f1343b8cf40"
TREE = "a7b9b0da0ad8d0991f011952194e014b14bfcf29"
JOB = 138345323477
REPO = original.REPO
BRANCH = "research/en-v6-b21-primary-source-continuation-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-primary-source-continuation-once.yml"
WORKFLOW_SHA = "0e2833cb9d4396a88b51a3309fee2d0095a2bd9fca549c823f49afda75aeb07a"
LOG_SHA = "5684761885422d2b1ad72849971cd73fdf10d3d81457c086cadb241e6281af13"
CORPUS_BYTES = 6387309499
CORPUS_SHA = "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2"
TRANSPORT_PLAN_SHA = "2848fefaf7d0c23f23b364be558b4f7a34792f64754ca0004d05a93e6823aef1"
PRODUCER = dict(id=14907020666, run_id=RUN, run_head=HEAD, bytes=2752314,
    sha256="6e811000e29489e75f819b56cf18100fffd522483177d2bc54223a1c533b6830")
TRANSPORT = dict(id=14907235352, run_id=RUN, run_head=HEAD, bytes=11985,
    sha256="be9b9a9feb9e0f9c4d69683d6e5eeac875b099fcf2c1075f862682ccc9781744")
FILES = {"api/run-final.json", "api/jobs-final.json", "api/artifacts-final.json", "api/commit.json",
    "api/prepared_main_comparison.json", "api/tree.json", "api/source-job.log",
    "producer-original.zip", "transport-original.zip"}


def execution_records(run, jobs, artifacts):
    need(run["id"] == RUN and run["head_sha"] == HEAD and run["run_attempt"] == 1
         and run["status"] == "completed" and run["conclusion"] == "failure" and run["event"] == "push"
         and run["head_branch"] == BRANCH and run["path"] == WORKFLOW
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False
         and run["head_repository"]["full_name"] == REPO and run["head_repository"]["private"] is False
         and run["head_commit"]["id"] == HEAD and run["head_commit"]["tree_id"] == TREE,
         "only the exact completed first-attempt continuation failure is eligible")
    by_name = {job["name"]: job for job in jobs}
    need(len(jobs) == len(by_name) == 3 and set(by_name) == {"sources", "observers", "readback"}
         and all(job["run_id"] == RUN and job["status"] == "completed" for job in jobs),
         "complete terminal three-job continuation graph")
    for name, ident in (("observers", 138346524180), ("readback", 138346525282)):
        job = by_name[name]
        need(job["id"] == ident and job["conclusion"] == "skipped" and not job.get("steps"),
             "no observer or source readback execution")
    source = by_name["sources"]
    need(source["id"] == JOB and source["conclusion"] == "failure", "exact failed source job")
    expected = [
        (1, "Set up job", "success"),
        (2, "Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "success"),
        (3, "Run actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1", "success"),
        (4, "Capture exact Git-bound scientific producer", "success"),
        (5, "Run actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", "success"),
        (6, "Authenticate original curl-stop and complete unchanged history before continuation", "success"),
        (7, "Fetch the complete identical corpus over eight audited official ranges", "failure"),
        (8, "Run actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", "success"),
        (9, "Freeze before-generation consumption after complete whole-corpus verification", "skipped"),
        (10, "Run actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", "skipped"),
        (11, "Generate the original96synthetic and24whole natural sources exactly once", "skipped"),
        (12, "Run actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", "success"),
        (23, "Post Run actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1", "skipped"),
        (24, "Post Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "success"),
        (25, "Complete job", "success")]
    need([(step["number"], step["name"], step["conclusion"]) for step in source["steps"]] == expected
         and all(step["status"] == "completed" for step in source["steps"]),
         "exact transport stop; consume, consumption upload and generation all skipped")
    pins = {pin["id"]: pin for pin in (PRODUCER, TRANSPORT)}
    need(len(artifacts) == 2 and {item["id"] for item in artifacts} == set(pins),
         "only producer and STOP transport receipts; any consumption/source/native artifact forbids continuation")
    for item in artifacts:
        pin = pins[item["id"]]
        name = ("english-v6-b21-primary-continuation-source-producer-" if pin == PRODUCER
                else "english-v6-b21-continuation-transport-") + str(RUN)
        need(item["name"] == name and item["expired"] is False
             and item["size_in_bytes"] == pin["bytes"] and item["digest"] == "sha256:" + pin["sha256"]
             and item["workflow_run"]["id"] == RUN and item["workflow_run"]["head_sha"] == HEAD
             and item["workflow_run"]["head_branch"] == BRANCH, "exact original two artifact identities")


def validate_remote(run, jobs, artifacts):
    """Require complete freshly read raw API inventories before new transport."""
    need(jobs.get("total_count") == len(jobs.get("jobs", [])) == 3
         and artifacts.get("total_count") == len(artifacts.get("artifacts", [])) == 2,
         "complete live failed-continuation inventories, no abbreviated first page")
    execution_records(run, jobs["jobs"], artifacts["artifacts"])
    return dict(failed_continuation_run_id=RUN, failed_continuation_run_head=HEAD,
        failed_continuation_job_id=JOB, run_attempt=1, complete_jobs=3, complete_artifacts=2,
        no_consumption_or_source_artifact=True, consumption_and_generation_skipped=True)


def verify(directory=FAILURE_DIR):
    first = original.verify()
    directory = Path(directory)
    raw = (directory / "MANIFEST.json").read_bytes()
    need(len(raw) == 1595 and sha(raw) == MANIFEST_SHA256, "exact reviewed second-failure manifest")
    manifest = json.loads(raw)
    need(manifest["schema"] == "english-v6-b21-fixed-mirror-zero-source-failure-proof-v1"
         and manifest["failed_continuation_run_id"] == RUN and manifest["failed_continuation_run_head"] == HEAD
         and manifest["prepared_parent"] == PREPARED and manifest["source_job_id"] == JOB
         and set(manifest["files"]) == FILES, "one explicitly identified zero-call transport failure")
    found = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    need(found == FILES | {"MANIFEST.json"} and not directory.is_symlink()
         and not any(p.is_symlink() for p in directory.rglob("*")), "all and only complete regular original evidence files")
    payloads = {}
    for name, pin in manifest["files"].items():
        body = (directory / name).read_bytes()
        need(len(body) == pin["bytes"] and sha(body) == pin["sha256"], "unchanged original evidence " + name)
        payloads[name] = body
    doc = lambda name: json.loads(payloads["api/" + name + ".json"])
    run, jobs, artifacts, commit, comparison, tree = (doc(name) for name in
        ("run-final", "jobs-final", "artifacts-final", "commit", "prepared_main_comparison", "tree"))
    validate_remote(run, jobs, artifacts)
    need(commit["sha"] == HEAD and [p["sha"] for p in commit["parents"]] == [PREPARED]
         and commit["tree"]["sha"] == tree["sha"] == TREE and tree["truncated"] is False,
         "actual original single parent and complete source Git tree")
    need(comparison["status"] == "ahead" and comparison["ahead_by"] == comparison["total_commits"] == 1
         and comparison["behind_by"] == 0 and comparison["base_commit"]["sha"] == PREPARED
         and comparison["merge_base_commit"]["sha"] == PREPARED
         and [c["sha"] for c in comparison["commits"]] == [HEAD]
         and [(f["filename"], f["status"]) for f in comparison["files"]] == [(WORKFLOW, "modified")],
         "original continuation activation changed only its frozen workflow")
    entries = {item["path"]: item for item in tree["tree"]}
    need(len(entries) == len(tree["tree"]), "unique complete original Git tree")
    with original.original_archive(payloads["producer-original.zip"], PRODUCER) as archive:
        proof = json.loads(archive.read("PRODUCER.json"))
        need(proof["schema"] == "english-v6-b21-original-git-producer-snapshot-v1"
             and proof["role"] == "source_continuation" and proof["run_id"] == str(RUN)
             and proof["run_head"] == HEAD and proof["run_attempt"] == 1 and proof["run_branch"] == BRANCH
             and proof["prepared_parent"] == PREPARED and proof["runtime_workflow"] == WORKFLOW
             and proof["git_tree"] == TREE and len(proof["files"]) == 246
             and proof["runtime_workflow_preserved_hidden"] is proof["source_code_reads_only"] is True
             and proof["network_calls"] == proof["new_source_calls"] == proof["new_model_forwards"] == proof["optimizer_updates"] == 0
             and proof["promotion_allowed"] is False, "complete original 246-file Git-bound source producer")
        need(set(proof["files"]) == set(proof["python_files"]) | set(proof["resource_files"]) | {WORKFLOW}
             and set(archive.namelist()) == {"PRODUCER.json"} | {"repository/" + name for name in proof["files"]},
             "all original producer dependencies and hidden workflow retained")
        for name, pin in proof["files"].items():
            body = archive.read("repository/" + name); entry = entries[name]
            need(entry["type"] == "blob" and entry["mode"] == pin["mode"] in ("100644", "100755")
                 and entry["sha"] == pin["git_blob_sha1"] == blob(body)
                 and len(body) == pin["bytes"] and sha(body) == pin["sha256"], "original Git producer bytes " + name)
        workflow = archive.read("repository/" + WORKFLOW)
        template = archive.read("repository/automation/english_v6_b21_continuation_source_workflow_template.yml")
        need(sha(workflow) == WORKFLOW_SHA
             and workflow == template.replace(b"__B21_SOURCE_CONTINUATION_MAIN_SHA__", PREPARED.encode()),
             "exact original closed-stage workflow and authentic activation")
        plan_raw = archive.read("repository/automation/english_v6_b21_continuation_transport.json")
        need(sha(plan_raw) == TRANSPORT_PLAN_SHA, "unchanged original transport plan")
        plan = json.loads(plan_raw)
        for name in original.FILES | {"MANIFEST.json"}:
            need(archive.read("repository/automation/english_v6_b21_continuation_failure/" + name)
                 == (original.FAILURE_DIR / name).read_bytes(), "original first failure proof remains byte-identical " + name)
    identity = dict(run_id=str(RUN), run_head=HEAD, run_attempt=1,
        run_branch="refs/heads/" + BRANCH, prepared_parent=PREPARED)
    parts = [dict(index=i, start=CORPUS_BYTES * i // 8, end=CORPUS_BYTES * (i + 1) // 8 - 1) for i in range(8)]
    with original.original_archive(payloads["transport-original.zip"], TRANSPORT) as archive:
        names = {"TRANSPORT/RESULT.json", "TRANSPORT/STARTED.json", "AUTHORIZATION.json", "ORIGINAL_RUN_LIVE.json"}
        names |= {"TRANSPORT/ranges/" + str(i) + "-STARTED.json" for i in range(8)}
        names |= {"TRANSPORT/ranges/" + str(i) + "-RETURNED.json" for i in (3, 7)}
        need(set(archive.namelist()) == names, "only exact STOP transport receipts; no complete corpus or consumption")
        get = lambda name: json.loads(archive.read(name))
        result = get("TRANSPORT/RESULT.json"); started = get("TRANSPORT/STARTED.json")
        permission = get("AUTHORIZATION.json"); live = get("ORIGINAL_RUN_LIVE.json")
        original.validate_remote(live["run"], live["jobs"], live["artifacts"])
        need(permission["schema"] == "english-v6-b21-audited-transport-continuation-authorization-v1"
             and permission["source_execution"] == identity and permission["transport_origin"] == first
             and permission["assessment_sha256"] == original.ASSESSMENT_SHA
             and permission["ancestry_sha256"] == original.ANCESTRY_SHA
             and permission["original_run_live_sha256"] == sha(archive.read("ORIGINAL_RUN_LIVE.json"))
             and permission["all_prior_replication_sealed"] is True
             and permission["new_KWS_forwards"] == 0 and permission["training_allowed"] is permission["promotion_allowed"] is False,
             "failed continuation retained exact original failure and authorization identity")
        need(started == dict(source_execution=identity, transport_plan_sha256=TRANSPORT_PLAN_SHA,
             transport_origin=first, ranges=parts, range_retries=0, source_calls=0, model_calls=0),
             "all eight transport requests belong to this exact source-free continuation")
        need(result["schema"] == plan["schema"] and result["status"] == "STOP"
             and result["bytes"] == CORPUS_BYTES and result["sha256"] == CORPUS_SHA
             and result["transport_plan_sha256"] == TRANSPORT_PLAN_SHA and result["urls"] == plan["urls"]
             and result["range_retries"] == result["source_calls"] == result["model_calls"] == 0
             and result["error_type"] == "URLError"
             and result["error"] == "<urlopen error _ssl.c:994: The handshake operation timed out>"
             and result["elapsed_seconds"] == 61.645063718000074
             and [part["index"] for part in result["ranges"]] == [7, 3],
             "exact original incomplete TLS-stop; source/model calls remain zero")
        for part in parts:
            need(get("TRANSPORT/ranges/" + str(part["index"]) + "-STARTED.json")
                 == dict(part, url=plan["urls"][part["index"] % 4], attempt=1), "every original range started once")
        range_hashes = {3: "1f39ad4bcbc4ff6e8f0248da509b44c02381a9502e51d790a4ed18d49a374f5a",
                        7: "ddae4b6faa7747866037710a822bda8ef7a53273445a0cab25fc133e8ab6abf2"}
        for returned in result["ranges"]:
            part = parts[returned["index"]]
            need(returned == get("TRANSPORT/ranges/" + str(part["index"]) + "-RETURNED.json")
                 and all(returned[k] == v for k, v in part.items())
                 and returned["attempt"] == 1 and returned["status"] == "COMPLETE"
                 and returned["url"] == plan["urls"][3] and returned["sha256"] == range_hashes[part["index"]]
                 and returned["received_bytes"] == part["end"] - part["start"] + 1
                 and returned["content_range"] == "bytes " + str(part["start"]) + "-" + str(part["end"]) + "/" + str(CORPUS_BYTES),
                 "two exact successful fixed-official-mirror ranges, not whole-corpus completion")
    raw_log = payloads["api/source-job.log"]
    need(len(raw_log) == 41875 and sha(raw_log) == LOG_SHA, "exact connector-decoded original log bytes")
    log = raw_log.decode("utf-8", errors="strict")
    stop = "urllib.error.URLError: <urlopen error _ssl.c:994: The handshake operation timed out>"
    absent = "No files were found with the provided path: /home/runner/work/_temp/source/. No artifacts will be uploaded."
    need(log.count(stop) == log.count("##[error]Process completed with exit code 1.") == log.count(absent) == 1
         and log.index(stop) < log.index(absent), "exact original transport exception and absent source output")
    shell = workflow.decode("utf-8")
    transport_cmd = 'python -IB automation/english_v6_b21_continuation_source.py transport "$RUNNER_TEMP/train-clean-100.tar.gz" "$RUNNER_TEMP/prior-source" "$PREPARED_PARENT"'
    consume_cmd = 'python -IB automation/english_v6_b21_continuation_source.py consume "$RUNNER_TEMP/train-clean-100.tar.gz" "$RUNNER_TEMP/prior-source" "$PREPARED_PARENT"'
    generate_cmd = 'python -IB automation/english_v6_b21_continuation_source.py generate "$RUNNER_TEMP/source" "$RUNNER_TEMP/prior-source" "$RUNNER_TEMP/train-clean-100.tar.gz" "$PREPARED_PARENT"'
    need(shell.count(transport_cmd) == shell.count(consume_cmd) == shell.count(generate_cmd) == 1
         and shell.index(transport_cmd) < shell.index(consume_cmd) < shell.index(generate_cmd),
         "frozen workflow separates failed transport before skipped consumption and generation")
    return dict(first, original_transport_origin=first,
        failed_continuation_run_id=RUN, failed_continuation_run_head=HEAD,
        failed_continuation_prepared_parent=PREPARED, failed_continuation_job_id=JOB,
        failed_continuation_producer_pin=PRODUCER, failed_continuation_transport_pin=TRANSPORT,
        failed_continuation_manifest_sha256=MANIFEST_SHA256, failed_continuation_log_sha256=LOG_SHA,
        failed_continuation_workflow_sha256=WORKFLOW_SHA, failed_continuation_producer_files=246,
        failed_continuation_actual_tts_calls=0, failed_continuation_natural_original_decodes=0,
        failed_continuation_observer_calls=0, failed_continuation_kws_forwards=0,
        failed_continuation_consumption_started=False, failed_continuation_source_generation_started=False,
        failed_continuation_preserved=True, live_refresh_required_for_both_failed_runs=True)


def verify_current_producer(root=None):
    """Every prior producer file except its separate activation workflow stays exact."""
    verify()
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    with zipfile.ZipFile(FAILURE_DIR / "producer-original.zip") as archive:
        proof = json.loads(archive.read("PRODUCER.json"))
        names = sorted(set(proof["files"]) - {WORKFLOW})
        need(len(names) == 245, "all 245 retained original continuation code/resources")
        for name in names:
            path = root / name
            need(path.is_file() and not path.is_symlink()
                 and path.read_bytes() == archive.read("repository/" + name), "unchanged prior source producer " + name)
    return names


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
