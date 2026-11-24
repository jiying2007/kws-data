"""Authenticate exactly one pre-consumption CLI failure; grant no scientific calls.

The original 45367984097 execution still consumed 96 public TTS calls and stays
STOP. This later run failed while fetching its log, before source authorization,
corpus transfer, natural consumption, decoding or observation. No retry is
authorized here. A separately registered first execution must verify this proof
and refresh validate_remote before its existing independent consumption gates.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAILURE_DIR = HERE / "english_v6_b21_natural_amendment_auth_failure"
SCHEMA = "english-v6-b21-natural-amendment-auth-origin-v1"
REPO = "jiying2007/kws-data"
RUN = 45375018737
HEAD = "c662f81c8005e0d1108f5d5c8eb250b0a26398b2"
PARENT = "4901f60f5c01deafb032d43ee839953b9240eb82"
TREE = "383b5667c7c6aed6e6a1ce7cd7d447d4749f2456"
JOB = 138378800173
BRANCH = "research/en-v6-b21-natural-amendment-sources-once-20261124"
WORKFLOW = ".github/workflows/english-v6-b21-natural-amendment-sources-once.yml"
TEMPLATE = "automation/english_v6_b21_natural_amendment_source_workflow_template.yml"
TOKEN = "__B21_NATURAL_AMENDMENT_MAIN_SHA__"
EXTRA_METADATA_WORKFLOW = ".github/workflows/english-v6-b21-natural-metadata-once.yml"
EXTRA_METADATA_SHA256 = "2bbfd312b33e7597d3e852777f2928d85f0f0b3c685555687e82111309b523a0"
EXTRA_METADATA_BLOB = "066415090f439244362cb11eb89ba9ea53905d8b"
MANIFEST_SHA256 = 'd3509217683a66d8b5aa8d946f750d5dc02e119461451897a75649cc913d004f'
FILE_PINS = {'artifacts.json': {'bytes': 1589, 'sha256': 'f4d3a8f1bbb472f1aebb65824ceafe7f9a4024a5c9d633a47df1ffdc6c5d0b9c'},
 'commit.json': {'bytes': 5248, 'sha256': 'a479dbf2876a05a22944e515f22cf33d7cfbabd2a5ccddb71050862430ff34c1'},
 'comparison.json': {'bytes': 16050, 'sha256': '95637d74271ae9f9a388247a2096f50f0b7cf3faa62312fb973e00affa0ab6e2'},
 'jobs.json': {'bytes': 5959, 'sha256': 'b55e130672f25bca258dc7aae7a7b17094193027e93dd14b736a4599ad13b989'},
 'original-source-ancestry.py': {'bytes': 18660,
                                 'sha256': 'c2505f9cd47b384ff432b228373f12a3c16f7406f5c28786af4a4ba1b294e938'},
 'original-source-producer.py': {'bytes': 6208,
                                 'sha256': '05e33a256f24617ac7f2928b4070cb1c8794edf65a3519c5622069c1b2962423'},
 'original-source-template.yml': {'bytes': 10836,
                                  'sha256': 'e78bc143b0f7d5670cb0e953e45b9315b9c04e42f36d85a4c1467bb34d0925b6'},
 'original-source-workflow.yml': {'bytes': 10848,
                                  'sha256': '1220f20216ad0311d2032400e231a16ee6478bf0f906f516b840d5a9e884012b'},
 'original-test-sources.py': {'bytes': 13054,
                              'sha256': 'ba5c7a9c4468d8e78671d2ed44e4416ac8daec7787f51eb591ac50afa98415c9'},
 'producer-manifest.json': {'bytes': 95710,
                            'sha256': 'b9f7fbf9b318e0791295a723cb8cdfa333d6321fef0bbf9b9affd0a867226140'},
 'reviewed-inventory.json': {'bytes': 68222,
                             'sha256': 'f66411b5740cd2d40c3755671ef779977728c7604b32b2da17e9487d8e455ed4'},
 'run.json': {'bytes': 12335, 'sha256': 'ae3afc38d57c79de1a57c83972fd8387ba43c0e81538f93ecdfcbf8e049ad30b'},
 'source-job.log': {'bytes': 38335, 'sha256': '2e9cf5938d5288e87fcdf4611130ce657731166fb1fcd6d52b79e0fed3580887'},
 'transport-original.zip': {'bytes': 181224,
                            'sha256': 'b1b8a688f39f3e6bedad542c59e92b9c56472a62bffe941da4e915e47f5852d0'},
 'tree.json': {'bytes': 1131789, 'sha256': '61ee8c4504d4ad84f3392eeab81bb0f254f600188e428b917d5669a5dbdd3fe6'}}
PRODUCER_PIN = dict(id=14910193423, run_id=RUN, run_head=HEAD, bytes=8575842,
    sha256="53362c4f108bd3902895713c6c688807ec239d00e0a697592fdb0fb247734485")
TRANSPORT_PIN = dict(id=14910527807, run_id=RUN, run_head=HEAD, **FILE_PINS["transport-original.zip"])
OPERATIONAL_CHANGES = frozenset({
    "automation/english_v6_b21_natural_amendment_source_ancestry.py",
    "automation/english_v6_b21_natural_amendment_source_producer.py",
    "automation/test_english_v6_b21_natural_amendment_sources.py",
})
PARTIAL_MEMBERS = frozenset({"METADATA_LIVE.json", *(
    "REUSE/api/" + name for name in ("run-final.json", "jobs-final.json", "artifacts-final.json",
        "commit.json", "tree.json", "prepared_main_comparison.json"))})
SNAPSHOTS = {
    "original-source-ancestry.py": "automation/english_v6_b21_natural_amendment_source_ancestry.py",
    "original-source-producer.py": "automation/english_v6_b21_natural_amendment_source_producer.py",
    "original-test-sources.py": "automation/test_english_v6_b21_natural_amendment_sources.py",
    "original-source-workflow.yml": WORKFLOW,
    "original-source-template.yml": TEMPLATE,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def validate_remote(run, jobs, artifacts):
    """Validate fresh complete raw API objects; never accept a new/retried run."""
    need(run["id"] == RUN and run["head_sha"] == HEAD and run["head_branch"] == BRANCH
         and run["path"] == WORKFLOW and run["event"] == "push" and run["run_attempt"] == 1
         and run["status"] == "completed" and run["conclusion"] == "failure"
         and all(run[key]["full_name"] == REPO and run[key]["private"] is False
                 for key in ("repository", "head_repository")), "exact first public failed authorization run")
    expected_jobs = {"sources": (JOB, "failure"), "observers": (138379362638, "skipped"),
                     "readback": (138379365585, "skipped")}
    values = jobs["jobs"]
    need(jobs["total_count"] == len(values) == 3 and {job["name"] for job in values} == set(expected_jobs),
         "complete three-job failure, no observer matrix entered")
    for job in values:
        expected_id, conclusion = expected_jobs[job["name"]]
        need(job["id"] == expected_id and job["run_id"] == RUN and job["head_sha"] == HEAD
             and job["status"] == "completed" and job["conclusion"] == conclusion,
             "exact original job identity and conclusion")
        if job["name"] != "sources":
            need(job["steps"] == [], "no observer/readback step was entered")
            continue
        steps = {step["number"]: step for step in job["steps"]}
        expected = {1: "success", 2: "success", 3: "success", 4: "success", 5: "success",
            6: "failure", 7: "skipped", 8: "success", 9: "skipped", 10: "skipped", 11: "skipped",
            12: "success", 23: "skipped", 24: "success", 25: "success"}
        need(len(steps) == len(job["steps"]) and set(steps) == set(expected)
             and all(steps[number]["status"] == "completed" and steps[number]["conclusion"] == state
                     for number, state in expected.items()), "failure precedes transfer, consumption and all scientific steps")
        names = {
            4: "Capture exact Git-bound scientific producer",
            6: "Authenticate the preserved failed96 source and committed metadata-qualified natural24",
            7: "Fetch the complete identical corpus over eight fixed official magicdatatech ranges",
            9: "Consume natural24 and observer600 budgets after whole-corpus verification",
            11: "Copy the saved96 originals and decode only the committed24 natural originals",
        }
        need(all(steps[number]["name"] == name for number, name in names.items()), "exact known failure boundary")
    pins = {
        "english-v6-b21-natural-amendment-source-producer-" + str(RUN): PRODUCER_PIN,
        "english-v6-b21-natural-amendment-transport-" + str(RUN): TRANSPORT_PIN,
    }
    values = artifacts["artifacts"]
    need(artifacts["total_count"] == len(values) == 2 and {item["name"] for item in values} == set(pins)
         and len({item["id"] for item in values}) == 2, "only producer and partial metadata; no consumed/source/native artifact")
    for item in values:
        pin = pins[item["name"]]
        need(item["id"] == pin["id"] and item["size_in_bytes"] == pin["bytes"]
             and item["digest"] == "sha256:" + pin["sha256"] and item["expired"] is False
             and item["workflow_run"]["id"] == RUN and item["workflow_run"]["head_sha"] == HEAD
             and item["workflow_run"]["head_branch"] == BRANCH
             and item["workflow_run"]["repository_id"] == item["workflow_run"]["head_repository_id"] == 1397319075,
             "same two complete unexpired original artifacts")
    return dict(failed_run_id=RUN, failed_run_head=HEAD, failure_stage="authorize_fetch_prior_job_log",
                source_budget_uploaded=False, new_source_calls=0, new_natural_decodes=0, new_observer_calls=0, new_kws_forwards=0)


def _files(directory):
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink()
         and {path.name for path in directory.iterdir()} == set(FILE_PINS) | {"MANIFEST.json"}, "exact complete failure proof file set")
    raw = {}
    for name, pin in FILE_PINS.items():
        path = directory / name
        need(path.is_file() and not path.is_symlink(), "regular failure evidence file")
        raw[name] = path.read_bytes()
        need(len(raw[name]) == pin["bytes"] and digest(raw[name]) == pin["sha256"], "immutable failure evidence: " + name)
    manifest = directory / "MANIFEST.json"
    need(manifest.is_file() and not manifest.is_symlink() and digest(manifest.read_bytes()) == MANIFEST_SHA256
         and json.loads(manifest.read_bytes()) == dict(schema="english-v6-b21-natural-amendment-auth-failure-files-v1", files=FILE_PINS),
         "complete exact content-addressed failure proof manifest")
    return raw


def _archive(raw):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    items = archive.infolist()
    need(items and len(items) == len({item.filename for item in items}) and archive.testzip() is None,
         "full original ZIP CRC with unique members")
    for item in items:
        path = PurePosixPath(item.filename)
        need(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in item.filename
             and "\0" not in item.filename and not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16),
             "safe original regular ZIP member")
    return archive


def _producer(raw):
    commit, tree, comparison, reviewed = (json.loads(raw[name]) for name in
        ("commit.json", "tree.json", "comparison.json", "reviewed-inventory.json"))
    commit_tree = commit["tree"] if "tree" in commit else commit["commit"]["tree"]
    need(commit["sha"] == HEAD and commit_tree["sha"] == TREE and [p["sha"] for p in commit["parents"]] == [PARENT]
         and tree["sha"] == TREE and tree["truncated"] is False, "exact actual commit, tree and sole prepared parent")
    need(comparison["status"] == "ahead" and comparison["ahead_by"] == comparison["total_commits"] == 1
         and comparison["behind_by"] == 0 and comparison["base_commit"]["sha"] == PARENT
         and [item["sha"] for item in comparison["commits"]] == [HEAD]
         and [(item["filename"], item["status"]) for item in comparison["files"]] == [(WORKFLOW, "modified")],
         "single workflow-only activation with no scientific producer edit")
    tree_files = {entry["path"]: entry for entry in tree["tree"]}
    document = json.loads(raw["producer-manifest.json"])
    files = document["files"]
    need(document["schema"] == "english-v6-b21-original-git-producer-snapshot-v1"
         and document["role"] == "source_natural_amendment" and document["run_id"] == str(RUN)
         and document["run_head"] == HEAD and document["run_attempt"] == 1 and document["run_branch"] == BRANCH
         and document["prepared_parent"] == PARENT and document["git_tree"] == TREE
         and document["runtime_workflow"] == WORKFLOW, "original failed execution's producer identity")
    need(reviewed["total_files"] == len(reviewed["files"]) == 291 and len(files) == 292
         and set(files) == set(reviewed["files"]) | {EXTRA_METADATA_WORKFLOW}
         and document["python_files"] == reviewed["python_files"] and len(document["python_files"]) == 151
         and set(document["resource_files"]) == set(reviewed["resource_files"]) | {EXTRA_METADATA_WORKFLOW}
         and len(document["resource_files"]) == 141, "reviewed291 plus exactly one unchanged metadata workflow captured from Git")
    for name, pin in files.items():
        entry = tree_files.get(name, {})
        need(pin["git_blob_sha1"] == entry.get("sha") and entry.get("type") == "blob"
             and entry.get("size") == pin["bytes"] and entry.get("mode") == pin["mode"]
             and pin["mode"] in ("100644", "100755"), "original producer manifest matches actual Git entry: " + name)
        if name == EXTRA_METADATA_WORKFLOW:
            need(pin["bytes"] == 2702 and pin["sha256"] == EXTRA_METADATA_SHA256 and pin["git_blob_sha1"] == EXTRA_METADATA_BLOB,
                 "sole additional resource is the immutable historical metadata workflow")
        elif name != WORKFLOW:
            need(all(pin[key] == reviewed["files"][name][key] for key in ("bytes", "sha256")), "all reviewed scientific bytes unchanged")
    for snapshot, name in SNAPSHOTS.items():
        content, pin = raw[snapshot], files[name]
        need(len(content) == pin["bytes"] and digest(content) == pin["sha256"] and blob(content) == pin["git_blob_sha1"],
             "original operational/workflow bytes preserved: " + name)
    template = raw["original-source-template.yml"]
    need(template.count(TOKEN.encode()) == 2
         and raw["original-source-workflow.yml"] == template.replace(TOKEN.encode(), PARENT.encode()),
         "original activated workflow remains exact with only two substitutions")
    need(document["source_code_reads_only"] is document["runtime_workflow_preserved_hidden"] is True
         and document["promotion_allowed"] is False and all(document[key] == 0 for key in
             ("network_calls", "new_source_calls", "new_model_forwards", "optimizer_updates")), "producer capture is a read-only byte snapshot")
    return document


def verify(directory=FAILURE_DIR):
    raw = _files(directory)
    state = validate_remote(*(json.loads(raw[name]) for name in ("run.json", "jobs.json", "artifacts.json")))
    producer = _producer(raw)
    with _archive(raw["transport-original.zip"]) as archive:
        need(set(archive.namelist()) == PARTIAL_MEMBERS, "exact seven JSON files, no authorization/transport/source/consumption")
        partial = {name: json.loads(archive.read(name)) for name in archive.namelist()}
        prior = partial["REUSE/api/run-final.json"]
        need(prior["id"] == 45367984097 and prior["head_sha"] == "c262e4b5cf06de8a6950c2c797c7aefcf43498d8"
             and prior["status"] == "completed" and prior["conclusion"] == "failure" and prior["run_attempt"] == 1,
             "earlier96-call failed source retained, never reclassified as zero-call")
        metadata = partial["METADATA_LIVE.json"]
        need(metadata["run"]["id"] == 45371439279 and metadata["run"]["head_sha"] == "6e44e650c88ddab732eacc0605c371fe804df956"
             and metadata["run"]["conclusion"] == "success" and metadata["artifacts"]["total_count"] == len(metadata["artifacts"]["artifacts"]) == 3,
             "only committed metadata evidence fetched before the log failed")
    log = raw["source-job.log"]
    need(all(value in log for value in (
        b"the response contains terminal escape sequences; pass --allow-escape-sequences to output it anyway",
        b'proof = fetch_reuse_bundle(out / "REUSE")',
        b'return subprocess.check_output(["gh", "api", path])',
        b"actions/jobs/138354602305/logs']' returned non-zero exit status 1.",
        b'"new_natural_decodes": 0, "new_observer_calls": 0, "source_opened": false',
        b"No files were found with the provided path: /home/runner/work/_temp/source/. No artifacts will be uploaded.",
    )), "exact CLI raw-log refusal before source authorization, transfer or science")
    return dict(schema=SCHEMA, **state, failed_prepared_parent=PARENT, failed_tree=TREE, failed_source_job_id=JOB,
        producer_pin=dict(PRODUCER_PIN), transport_pin=dict(TRANSPORT_PIN), original_producer_files=len(producer["files"]),
        reviewed_producer_files=291, additional_unchanged_metadata_workflow=EXTRA_METADATA_WORKFLOW,
        producer_manifest_sha256=digest(raw["producer-manifest.json"]), proof_manifest_sha256=MANIFEST_SHA256,
        original_source_status="STOP", prior_original_public_tts_calls=96, new_public_tts_calls=0,
        authorization_completed=False, corpus_transfer_started=False, natural_budget_consumed=False,
        observer_budgets_consumed=False, source_artifact_created=False, execution_authorized=False,
        training_allowed=False, promotion_allowed=False)


def verify_original_producer_archive(path, directory=FAILURE_DIR):
    """One offline preparation check of the separately retained original ZIP.

    Runtime proof capture retains its exact manifest and original changed-file
    snapshots instead of recursively embedding this 8.5 MB archive. Current
    unchanged files are authenticated separately by verify_current_producer.
    """
    raw = _files(directory)
    document = _producer(raw)
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "regular separately retained original producer archive")
    packed = path.read_bytes()
    need(len(packed) == PRODUCER_PIN["bytes"] and digest(packed) == PRODUCER_PIN["sha256"], "exact original producer ZIP size and SHA256")
    with _archive(packed) as archive:
        need(set(archive.namelist()) == {"PRODUCER.json", *("repository/" + name for name in document["files"])}
             and archive.read("PRODUCER.json") == raw["producer-manifest.json"], "complete original producer archive and exact extracted manifest")
        for name, pin in document["files"].items():
            content = archive.read("repository/" + name)
            need(len(content) == pin["bytes"] and digest(content) == pin["sha256"] and blob(content) == pin["git_blob_sha1"],
                 "all original archive member bytes: " + name)
    return dict(producer_pin=dict(PRODUCER_PIN), original_files=292, original_zip_members=293,
                compact_manifest_sha256=digest(raw["producer-manifest.json"]))


def verify_current_producer(root=None):
    root = ROOT if root is None else Path(root)
    directory = root / "automation" / FAILURE_DIR.name
    verify(directory)
    raw = _files(directory)
    producer = _producer(raw)
    excluded = OPERATIONAL_CHANGES | {WORKFLOW}
    need(excluded <= set(producer["files"]), "only explicit operational files are excluded from current equality")
    checked = 0
    for name, pin in producer["files"].items():
        if name in excluded:
            continue
        path = root / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == pin["bytes"]
             and digest(path.read_bytes()) == pin["sha256"] and blob(path.read_bytes()) == pin["git_blob_sha1"],
             "unchanged prior producer outside reviewed operational changes: " + name)
        checked += 1
    need(checked == 288, "all288 untouched current producer files authenticated")
    return dict(original_producer_files=292, preserved_current_files=checked,
                explicit_operational_exceptions=sorted(OPERATIONAL_CHANGES), original_workflow_preserved_in_proof=True)


def resource_paths(root=None):
    root = ROOT if root is None else Path(root)
    directory = root / "automation" / FAILURE_DIR.name
    verify(directory)
    return {str((directory / name).relative_to(root)) for name in set(FILE_PINS) | {"MANIFEST.json"}}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
