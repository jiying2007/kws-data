"""Pure readback of original B20 source/native archives. No model initialization.

This script never fills missing native observations or creates a source SCREEN.
When the original five-provider evidence arrives, the frozen loader/screen
reader can be called on it without replaying any observation.
"""
from __future__ import annotations
import argparse
import builtins
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from unittest import mock
import zipfile

RUN = 45334413855
HEAD = "249449c8a572c536dfb4e990584b451247d7da17"
PARENT = "5dcce63a5ffced40e4b5dae76d06eb99ece5da3a"
SOURCE_PIN = dict(id=14892539914, run_id=RUN, run_head=HEAD, bytes=8974682,
    sha256="de5c580f7c6cdd99abd3268f5667b60d748d21f15ab5e2451123962b26775505")
MANIFEST_SHA = "8acc2cca1eb775174eb8b4e3d18a26ab7a68e9dfd35b9dd811d428eb0a57d4b8"
SCREEN_SHA = "d125b512340fb540b9e0e32fb6077420813579d816ccebfce8d5d982e6831646"
CODE = {
    "automation/english_v6_b20_source.py": "f6a3a1e53d3307b3369561ae7f8c25346a43ff6424ecc87d37dd71bebcb5893e",
    "automation/english_v6_b20_source_ancestry.py": "3778dc315d5015aa0dc576171d7b79a1ae73d4682d2a5a14dc785d87a971c0fa",
    "automation/english_v6_b20_source_observe.py": "9b8171ad3b6286202a23823e91c9f8075862e339c73fb1a93dfcbef9e9779fd2",
    "automation/english_v6_b20_panels/assessment.json": "22a238a7c773912bd4bc1429e8b3efbd18147a793a68a798fa064417713afd87",
    "automation/english_v6_b20_panels/primary.json": "7e197c8cd15d8a738fb559db2858202089f3428d25e1a4529d1b0211b0958727",
}


def need(ok, reason):
    if not ok: raise ValueError(reason)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def unpack(path, pin, destination):
    raw = Path(path).read_bytes()
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "original archive exact length and SHA")
    destination.mkdir()
    with zipfile.ZipFile(path) as archive:
        need(archive.testzip() is None, "original ZIP CRC")
        names = archive.namelist()
        need(len(names) == len(set(names)), "unique ZIP member names")
        for member in archive.infolist():
            p = PurePosixPath(member.filename)
            need(not p.is_absolute() and not any(v in ("", ".", "..") for v in p.parts)
                 and "\\" not in member.filename and not stat.S_ISLNK(member.external_attr >> 16),
                 "safe complete original ZIP member")
            output = destination.joinpath(*p.parts)
            if member.is_dir(): output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.read(member))
    return len(names)


def protected(source):
    stack = ExitStack()
    forbidden = {"torch", "torchaudio", "transformers", "piper", "soundfile", "onnxruntime", "qwen_asr", "whisper"}
    original_import = builtins.__import__
    def import_guard(name, *args, **kwargs):
        need(name.split(".")[0] not in forbidden, "NO_MODEL_OR_AUDIO_DECODER_IMPORT_DURING_READBACK: " + name)
        return original_import(name, *args, **kwargs)
    stack.enter_context(mock.patch("builtins.__import__", side_effect=import_guard))
    for obj, name in ((source.B13, "render_panel"), (source.observers.raw_asr, "_load"),
                      (source.observers, "_load_phone"), (source.observers.native, "_load_phone")):
        stack.enter_context(mock.patch.object(obj, name, side_effect=AssertionError("NO_NEW_SOURCE_OR_MODEL_CALLS")))
    stack.enter_context(mock.patch.object(source.lineage, "gh", side_effect=AssertionError("NO_NETWORK_DURING_SAVED_READBACK")))
    return stack


def source_readback(actual, evidence, repository):
    for path, digest in CODE.items(): need(sha((repository / path).read_bytes()) == digest, "frozen scientific code: " + path)
    sys.path.insert(0, str(repository / "automation"))
    import english_v6_b20_source as source
    run, jobs, item = (read(evidence / name) for name in ("SOURCE_RUN_API.json", "SOURCE_JOBS_API.json", "SOURCE_ARTIFACT_API.json"))
    commit = read(evidence / "COMMIT_API.json")
    need(run["id"] == RUN and run["head_sha"] == HEAD and run["run_attempt"] == 1 and run["event"] == "push"
         and run["repository"]["private"] is False and run["repository"]["full_name"] == source.lineage.REPO,
         "actual first public B20 source run")
    source_jobs = [job for job in jobs["jobs"] if job["name"] == "sources"]
    need(len(source_jobs) == 1 and source_jobs[0]["status"] == "completed" and source_jobs[0]["conclusion"] == "success"
         and source_jobs[0]["run_id"] == RUN and source_jobs[0]["head_sha"] == HEAD and source_jobs[0]["run_attempt"] == 1,
         "original source producer completed, independently of downstream observers")
    need(commit["sha"] == HEAD and len(commit["parents"]) == 1 and commit["parents"][0]["sha"] == PARENT
         and [row["filename"] for row in commit["files"]] == [".github/workflows/english-v6-b20-synthetic-primary-once.yml"],
         "only the reviewed one-shot workflow changed at its exact parent")
    need(item["id"] == SOURCE_PIN["id"] and item["size_in_bytes"] == SOURCE_PIN["bytes"]
         and item["digest"] == "sha256:" + SOURCE_PIN["sha256"] and item["expired"] is False
         and item["workflow_run"]["id"] == RUN and item["workflow_run"]["head_sha"] == HEAD,
         "actual original source artifact metadata")
    with tempfile.TemporaryDirectory() as temporary, protected(source):
        directory = Path(temporary) / "source"
        members = unpack(actual / "artifacts/source-original.zip", SOURCE_PIN, directory)
        need(sha((directory / "MANIFEST.json").read_bytes()) == MANIFEST_SHA, "original source manifest")
        rows = source.read_source(directory)
        preflight = source.evaluation_preflight(directory)
        reg = source.registration(directory)
        meta, history = read(directory / "MANIFEST.json"), preflight["history"]
        need(rows == preflight["rows"] and len(rows) == len({r["pcm_sha256"] for r in rows}) == 96
             and len({r["source_group_id"] for r in rows}) == len({r["source_speaker_id"] for r in rows}) == 24,
             "all 96 original PCM and 24 independent families")
        need(reg["run_id"] == str(RUN) and reg["run_head"] == HEAD and reg["prepared_parent"] == PARENT,
             "real artifact run equals original generation and authorization")
        positive = [r for r in rows if r["requested_class"] == 2]
        need(len(positive) == 48 and all(6400 <= r["samples"] <= 28672 for r in positive), "all positive whole originals fit preregistered model input limits")
        result = dict(schema="english-v6-b20-independent-original-source-readback-v1", status="PASS_SOURCE_BYTES",
            source_artifact=SOURCE_PIN, archive_members=members, source_manifest_sha256=MANIFEST_SHA,
            history_sha256=sha((directory / "HISTORY.json").read_bytes()), assessment_sha256=source.ASSESSMENT_SHA,
            source_execution=meta["source_execution"], source_rows=len(rows), unique_PCM=96, source_groups=24,
            positive_originals=48, positive_samples_min=min(r["samples"] for r in positive), positive_samples_max=max(r["samples"] for r in positive),
            public_Piper_synthesize_calls=meta["tts_public_synthesize_api_invocations"], native_output_chunks=meta["native_output_chunks"],
            internal_TTS_model_forwards=meta["internal_tts_model_forwards"],
            original_history_counts={k: len(history[k]) for k in ("all_pcm_sha256", "all_source_groups", "canonical_people")},
            full_raw_history_receipt_verified=True, original_B19_three_ZIPs_recomputed=True,
            all_48_old_archive_bodies_reopened_in_this_local_readback=False,
            source_admission_performed=False, five_observer_QA_not_asserted_by_this_component=True,
            new_source_calls_during_readback=0, new_model_calls_during_readback=0,
            old_replication_opened=False, promotion_allowed=False)
    (evidence / "SOURCE_INDEPENDENT_READBACK.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def producer_readback(evidence, repository):
    commit, tree, inventory = (read(evidence / name) for name in
        ("COMMIT_API.json", "ACTUAL_PRODUCER_TREE_API.json", "SOURCE_PRODUCER_INVENTORY.json"))
    need(tree["truncated"] is False and tree["sha"] == commit["commit"]["tree"]["sha"], "actual complete producer tree")
    members = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
    hashes = {**inventory["files"], **inventory["resources"]}
    measured = {}
    for path, digest in hashes.items():
        raw = (repository / path).read_bytes()
        git_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        need(sha(raw) == digest and members[path]["sha"] == git_hash and members[path]["mode"] in ("100644", "100755"),
             "actual producer scientific file/resource differs: " + path)
        measured[path] = dict(bytes=len(raw), sha256=digest, git_blob_sha=git_hash)
    raw = (evidence / "ACTUAL_WORKFLOW.yml").read_bytes()
    path = inventory["workflow_path"]
    need(members[path]["sha"] == hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
         and raw.count(PARENT.encode()) == 2
         and sha(raw.replace(PARENT.encode(), b"__B20_SYNTHETIC_MAIN_SHA__")) == inventory["reviewed_workflow_template_sha256"],
         "actual producer workflow has precisely the two reviewed parent substitutions")
    need(inventory["workflow_sha256"] == sha(raw), "expanded inventory pins the exact actual source workflow")
    original = read(evidence / "SOURCE_PRODUCER_ORIGINAL_INVENTORY.json")
    need(all(inventory[k].get(path) == value for k in ("files", "resources") for path, value in original[k].items()),
         "static closure expansion preserves every original registered file pin")
    expansion = dict(python_files=sorted(set(inventory["files"]) - set(original["files"])),
        resources=sorted(set(inventory["resources"]) - set(original["resources"])),
        literal_filename_dynamic_imports_and_repository_packages_included=True,
        new_audio_or_model_evidence_required=False)
    result = dict(schema="english-v6-b20-original-source-producer-readback-v1", run_head=HEAD,
        actual_tree_sha=tree["sha"], prepared_parent=PARENT, python_files=len(inventory["files"]),
        resources=len(inventory["resources"]), files=measured, actual_workflow_sha256=sha(raw),
        reviewed_workflow_template_sha256=inventory["reviewed_workflow_template_sha256"],
        registered_producer_inventory_verified=True, dependency_inventory_exhaustiveness_claimed=False,
        expanded_static_closure_verified=True, expansion=expansion,
        new_network_or_model_calls=0)
    (evidence / "PRODUCER_READBACK.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def native_readback(actual, evidence, repository):
    sys.path.insert(0, str(repository / "automation"))
    import english_v6_b20_source as source
    run, jobs, artifacts = (read(evidence / name) for name in ("RUN_FINAL.json", "JOBS_FINAL.json", "ARTIFACTS_FINAL.json"))
    need(run["id"] == RUN and run["head_sha"] == HEAD and run["status"] == "completed" and run["conclusion"] == "success"
         and run["run_attempt"] == 1 and run["event"] == "push" and run["repository"]["private"] is False,
         "actual terminal first public B20 source run")
    expected_jobs = {"sources", "readback", *("observers (" + p + ")" for p in source.observers.PROVIDERS)}
    need(jobs["total_count"] == len(jobs["jobs"]) == len(expected_jobs) == 7
         and {j["name"] for j in jobs["jobs"]} == expected_jobs
         and all(j["run_id"] == RUN and j["head_sha"] == HEAD and j["run_attempt"] == 1
                 and j["status"] == "completed" and j["conclusion"] == "success" for j in jobs["jobs"]),
         "all seven original source/native/readback producers succeeded once")
    names = {"source": "english-v6-b20-primary-source-" + str(RUN),
        "screen": "english-v6-b20-primary-frozen-screen-" + str(RUN),
        **{p: "b20-primary-dev-observer-" + p for p in source.observers.PROVIDERS}}
    locks = {}
    for role, name in names.items():
        found = [a for a in artifacts["artifacts"] if a["name"] == name]
        need(len(found) == 1, "one original artifact for " + role)
        item = found[0]
        need(item["workflow_run"]["id"] == RUN and item["workflow_run"]["head_sha"] == HEAD and item["expired"] is False
             and item["digest"].startswith("sha256:"), "actual immutable original artifact provenance")
        locks[role] = dict(id=item["id"], run_id=RUN, run_head=HEAD, bytes=item["size_in_bytes"], sha256=item["digest"].removeprefix("sha256:"))
    need(locks["source"] == SOURCE_PIN, "same already audited original source")
    with tempfile.TemporaryDirectory() as temporary, protected(source):
        root = Path(temporary)
        unpack(actual / "artifacts/source-original.zip", locks["source"], root / "source")
        unpack(actual / "artifacts/screen-original.zip", locks["screen"], root / "screen")
        providers = root / "providers"; providers.mkdir()
        for provider in source.observers.PROVIDERS:
            unpack(actual / "artifacts" / (provider + "-original.zip"), locks[provider], providers / provider)
        official_raw = (root / "screen/SCREEN.json").read_bytes()
        need(sha(official_raw) == SCREEN_SHA, "original official source SCREEN bytes")
        official = json.loads(official_raw)
        # This reads every FILES closure, original call/return/valid ledger,
        # raw ASR payload, native phone grid/CTC and source decision. No model.
        recomputed = source.verify_screen_contract(root / "source", official, providers)
        rows = source.read_source(root / "source")
        counts = {}
        for provider in source.observers.PROVIDERS:
            ledger = source.observers.row_ledger(providers / provider, rows)
            value = source.observers.ledger_counts(ledger)
            need(source.observers.complete_ledger(ledger, 96), "all original distinct valid native payloads")
            counts[provider] = value
        totals = {key: sum(value[key] for value in counts.values()) for key in next(iter(counts.values()))}
        need(totals == dict(input_started=480, forward_attempted=480, native_returned=480,
             raw_payload_saved=480, payload_valid=480, capture_saved=480, failed=0), "exact complete 480 original native returns")
        need(recomputed["source_gate"] == "SOURCE_GATE_PASS" and recomputed["source_coverage"] ==
             dict(positive_originals=15, ordinary_originals=48, paired_source_groups=12)
             and recomputed["unknown_count"] == 33 and recomputed["tier_counts"] == {"silver": 52, "weak": 11, "unknown": 33},
             "original unchanged source-admission result, including all UNKNOWN slots")
        result = dict(schema="english-v6-b20-independent-original-native-readback-v1", status="SOURCE_GATE_PASS",
            source_artifact=locks["source"], native_artifacts={p: locks[p] for p in source.observers.PROVIDERS}, screen_artifact=locks["screen"],
            run_id=RUN, run_head=HEAD, prepared_parent=PARENT, run_attempt=1, source_manifest_sha256=MANIFEST_SHA,
            screen_sha256=SCREEN_SHA, provider_counts=counts, total_native_counts=totals,
            source_coverage=recomputed["source_coverage"], unknown_count=33, tier_counts=recomputed["tier_counts"],
            original_slots=96, admitted_originals=63, source_families=24,
            all_original_ZIPs_reauthenticated=True, all_raw_native_payloads_verified=True,
            official_SCREEN_equal_to_pure_recomputation=True, admission_function_unchanged=True,
            new_source_calls_during_readback=0, new_model_calls_during_readback=0,
            KWS_evaluation_performed_by_this_source_run=False, numerical_model_quality_claimed=False,
            old_replication_opened=False, promotion_allowed=False)
    (evidence / "NATIVE_INDEPENDENT_READBACK.json").write_text(json.dumps(result, indent=2) + "\n")
    (evidence / "ARTIFACT_LOCKS.json").write_text(json.dumps(locks, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path); parser.add_argument("repository", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--all-native", action="store_true")
    args = parser.parse_args()
    result = source_readback(args.actual.resolve(), args.evidence.resolve(), args.repository.resolve())
    if args.all_native:
        producer_readback(args.evidence.resolve(), args.repository.resolve())
        result = native_readback(args.actual.resolve(), args.evidence.resolve(), args.repository.resolve())
    print(json.dumps(result, indent=2))
