"""One independent LV60 diagnostic of 24 saved full PCMs; no TTS or training.

Default/PR checks are stdlib-only. An exact first main push may authenticate the
old source packet, install one fixed runtime, and make at most 24 new forwards.
The original STOP packet is retained byte-for-byte and is never relabelled.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import english_pocket12_once as utility

need, digest, read_json, write_json, within = (
    utility.need, utility.digest, utility.read_json, utility.write_json, utility.within)
PLAN = HERE / "english_mero_meh_lv60_plan.json"
REPOSITORY = "jiying2007/kws-data"
WORKFLOW = ".github/workflows/english-mero-meh-lv60-once.yml"
CAMPAIGN = "english-mero-meh-lv60-diagnostic-v1"
CONTROLS = "research/english_mero_meh_20261118/LV60_DIAGNOSTIC_CONTRACT.md"
SOURCE_WORKFLOW = ".github/workflows/english-mero-meh-once.yml"
SOURCE_PLAN = "automation/english_mero_meh_plan.json"
SOURCE_PLAN_SHA = "43a98e1125a44d6599e46aa91efa593bd113613d004f612a730f681a18f2b335"
SOURCE = {"run_id": 44218734553, "artifact_id": 14389757284,
    "name": "english-mero-meh-corpus-44218734553", "bytes": 74476997,
    "head_sha": "7e9797789cba7d1476d02f14e21af91dc3cd0712",
    "sha256": "712e083b1bc7e1451c8eb43c2f99c94549d34f8d71db81d8cf2ef965e1df72da"}
SOURCE_RUN = {"repository": REPOSITORY, "run_id": str(SOURCE["run_id"]),
    "commit": SOURCE["head_sha"], "run_attempt": "1", "event": "push"}
SOURCE_FILES_SHA = "b9884f8d19c545f5027ad73b71d9176da6fd94e6cf592a97acb6f14ecf953d4b"
BUDGETS = {"initialization": 1200, "observation": 1800, "readback": 300, "download": 300}
LIMITS = {"maximum_expanded_bytes": 536870912, "maximum_member_bytes": 67108864,
          "maximum_files": 20000}
FLAGS = {"diagnostic_only": True, "training_allowed": False,
    "admission_mutation_allowed": False, "source_training_eligibility_changed": False,
    "weights_mutation_allowed": False, "threshold_mutation_allowed": False,
    "automatic_expansion": False, "native_tts_request_budget": 0, "optimizer_updates": 0,
    "retry_count": 0, "warmup_count": 0, "maximum_calls_per_observer": 24,
    "maximum_total_model_calls": 24, "normalization_match_required": True,
    "original_stop_preserved": "STOP_FRESH_CORPUS_COVERAGE"}
DOCUMENTS = {"CORPUS_MANIFEST.json", "MACHINE_EVIDENCE.json", "ADMISSION.json",
    "SOURCE_IDENTITY_CONTRACT.json", "EXECUTION_PLAN.json", "EXCLUSION_EVIDENCE.json",
    "FILES.json", "RESULT.json", "SOURCE_RESULT.json"}


def observer():
    return importlib.import_module("english_mero_meh_lv60_observe")


def source_preflight(plan=None):
    plan = read_json(PLAN) if plan is None else plan
    fixed = {"schema": "english-mero-meh-lv60-plan-v1", "campaign_id": CAMPAIGN,
        "repository": REPOSITORY, "workflow_path": WORKFLOW, "provider": "phoneme_lv60",
        "source_workflow_path": SOURCE_WORKFLOW, "source_artifact": SOURCE, "source_run": SOURCE_RUN,
        "source_files_manifest_sha256": SOURCE_FILES_SHA, "time_budget_seconds": BUDGETS,
        "source_limits": LIMITS, "controls_document": CONTROLS, **FLAGS}
    need(all(type(plan.get(k)) is type(v) and plan[k] == v for k, v in fixed.items()), "FIXED_LV60_AUDIT_PLAN")
    need(type(plan.get("enabled")) is bool, "BOOLEAN_ENABLED")
    before = plan.get("main_before")
    need(before is None or (type(before) is str and re.fullmatch(r"[0-9a-f]{40}", before)), "MAIN_PREDECESSOR")
    need(not plan["enabled"] or before is not None, "ENABLED_REQUIRES_PREDECESSOR")
    need(digest(within(ROOT, SOURCE_PLAN)) == SOURCE_PLAN_SHA, "IMMUTABLE_ORIGINAL_EXECUTION_PLAN")
    original_plan = read_json(within(ROOT, SOURCE_PLAN))
    pins = plan.get("source_files_sha256")
    required = set(original_plan["source_files_sha256"]) | {SOURCE_PLAN, WORKFLOW, CONTROLS}
    required |= {"automation/english_mero_meh_lv60_" + name + ".py" for name in ("once", "observe")}
    required |= {"automation/test_english_mero_meh_lv60_" + name + ".py" for name in ("once", "observe")}
    need(type(pins) is dict and required <= pins.keys(), "COMPLETE_OLD_AND_NEW_SOURCE_CLOSURE")
    need("automation/english_mero_meh_lv60_plan.json" not in pins, "PLAN_CANNOT_SELF_HASH")
    need(all(pins.get(n) == h for n, h in original_plan["source_files_sha256"].items()), "OLD_SOURCE_BYTES_UNCHANGED")
    for name, expected in pins.items():
        path = within(ROOT, name)
        need(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected)
             and path.is_file() and digest(path) == expected, "SOURCE_BYTES: " + name)
    documents = plan.get("source_document_pins")
    need(type(documents) is dict and set(documents) == DOCUMENTS, "NINE_ORIGINAL_DOCUMENT_PINS")
    for name, pin in documents.items():
        need(type(pin) is dict and set(pin) == {"bytes", "sha256"}
             and type(pin["bytes"]) is int and 0 < pin["bytes"] <= LIMITS["maximum_member_bytes"]
             and type(pin["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]),
             "BOUNDED_SOURCE_DOCUMENT_PIN: " + name)
    need(documents["FILES.json"]["sha256"] == SOURCE_FILES_SHA
         and documents["EXECUTION_PLAN.json"]["sha256"] == SOURCE_PLAN_SHA, "SOURCE_DOCUMENT_BINDING")
    old = importlib.import_module("english_mero_meh_once")
    old.source_preflight(original_plan)
    module = observer()
    module.source_check(plan)
    recipe = importlib.import_module("english_mero_meh_recipe")
    original_rows = {r["id"]: r for r in recipe.rows()}
    for row in plan["panel_rows"]:
        expected = original_rows[row["id"]]
        need(all(row.get(k) == expected[k] for k in ("id", "preset_id", "cell_id", "speed", "role",
             "requested_text", "requested_class", "kind", "request_sha256")), "ORIGINAL_PREREGISTERED_REQUEST")
        need(row["role"] == "train" and type(row["pcm_samples"]) is int and 6400 <= row["pcm_samples"] <= 160000,
             "EXISTING_FULL_PCM_PANEL")
        need(row["pcm_path"] == "pcm/" + row["id"] + ".pcm" and row["wav_path"] == "wav/" + row["id"] + ".wav",
             "ORIGINAL_PCM_AND_WAV_PATHS")
        need(all(type(row[k]) is str and re.fullmatch(r"[0-9a-f]{64}", row[k]) for k in
             ("pcm_sha256", "wav_sha256", "parent_pcm_sha256", "source_row_sha256")), "EXACT_PANEL_BYTE_PINS")
    return {"source_check": "PASS", "enabled": plan["enabled"], "pinned_files": len(pins),
        "denominator": 24, "new_model_calls": 0, "new_network_calls": 0, "diagnostic_only": True,
        "training_allowed": False}


def execution_identity(plan, *, environ=None, event=None, head=None, parent=None):
    env = os.environ if environ is None else environ
    need(plan.get("enabled") is True, "DISABLED_DIAGNOSTIC")
    expected = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_WORKFLOW_REF": REPOSITORY + "/" + WORKFLOW + "@refs/heads/main"}
    need(all(env.get(k) == v for k, v in expected.items()), "FIRST_PUBLIC_MAIN_PUSH_ONLY")
    run_id, commit = env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_SHA", "")
    need(re.fullmatch(r"[1-9][0-9]{0,19}", run_id) and re.fullmatch(r"[0-9a-f]{40}", commit)
         and run_id != SOURCE_RUN["run_id"], "DISTINCT_NEW_AUDIT_RUN")
    before = plan.get("main_before")
    need(type(before) is str and re.fullmatch(r"[0-9a-f]{40}", before), "BOUND_MAIN_PREDECESSOR")
    event = read_json(env["GITHUB_EVENT_PATH"]) if event is None else event
    repo = event.get("repository", {})
    need(repo.get("full_name") == REPOSITORY and repo.get("private") is False
         and event.get("ref") == "refs/heads/main" and event.get("before") == before
         and event.get("after") == commit and event.get("forced") is False and event.get("deleted") is False,
         "EXACT_PUBLIC_MAIN_SUCCESSOR")
    if head is None:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], cwd=ROOT, text=True, timeout=10).strip()
    need(head == commit and parent == before, "CHECKOUT_AND_PARENT")
    return {"repository": REPOSITORY, "run_id": run_id, "commit": commit, "run_attempt": "1", "event": "push"}


def validate_source_api(run, artifact, plan):
    pin = plan["source_artifact"]
    expected = {"id": pin["run_id"], "head_sha": pin["head_sha"], "head_branch": "main",
        "event": "push", "run_attempt": 1, "path": plan["source_workflow_path"],
        "status": "completed", "conclusion": "success"}
    need(all(type(run.get(k)) is type(v) and run[k] == v for k, v in expected.items()), "ORIGINAL_COMPLETED_PRODUCER_RUN")
    for name in ("repository", "head_repository"):
        need(run.get(name, {}).get("full_name") == REPOSITORY and run[name].get("private") is False,
             "ORIGINAL_PUBLIC_REPOSITORY")
    expected = {"id": pin["artifact_id"], "name": pin["name"], "size_in_bytes": pin["bytes"],
        "expired": False, "digest": "sha256:" + pin["sha256"]}
    need(all(type(artifact.get(k)) is type(v) and artifact[k] == v for k, v in expected.items()), "EXACT_ORIGINAL_ARTIFACT_API")
    producer = artifact.get("workflow_run", {})
    need(producer.get("id") == pin["run_id"] and producer.get("head_sha") == pin["head_sha"]
         and producer.get("head_branch") == "main"
         and producer.get("repository_id") == producer.get("head_repository_id") == run["repository"]["id"],
         "ARTIFACT_BELONGS_TO_ORIGINAL_PRODUCER")
    return {"source_api_verified": True, "source_artifact": pin, "source_run": plan["source_run"], "new_model_calls": 0}


def extract_archive(archive, destination, plan):
    archive, destination = Path(archive), Path(destination)
    pin, limits = plan["source_artifact"], plan["source_limits"]
    need(archive.is_file() and not archive.is_symlink() and archive.stat().st_size == pin["bytes"]
         and digest(archive) == pin["sha256"], "EXACT_ORIGINAL_ZIP_BYTES")
    need(not destination.exists() and not any(p.is_symlink() for p in destination.parents), "FRESH_EXTRACT_DIRECTORY")
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        names = [r.filename for r in members]
        need(len(names) == len(set(names)) and 0 < len(names) <= limits["maximum_files"], "UNIQUE_BOUNDED_ZIP_MEMBERS")
        need(sum(r.file_size for r in members) <= limits["maximum_expanded_bytes"], "ZIP_EXPANDED_BOUND")
        for item in members:
            within(destination, item.filename)
            mode = (item.external_attr >> 16) & 0xffff
            need(not item.is_dir() and not item.flag_bits & 1 and item.file_size <= limits["maximum_member_bytes"]
                 and not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0, stat.S_IFREG), "REGULAR_BOUNDED_ZIP_MEMBER")
        index_raw = source.read("FILES.json")
        need(observer().sha(index_raw) == plan["source_files_manifest_sha256"], "ORIGINAL_ARCHIVE_FILES_PIN")
        index = json.loads(index_raw, object_pairs_hook=utility.unique_object)
        need(set(index) == {"schema", "files"} and index["schema"] == "english-mero-meh-corpus-files-v1"
             and type(index["files"]) is list, "ORIGINAL_ARCHIVE_FILES_SCHEMA")
        pins = {r["path"]: {k: r[k] for k in ("bytes", "sha256")} for r in index["files"]}
        need(len(pins) == len(index["files"]) and set(names) == set(pins) | {"FILES.json"}, "EXACT_SOURCE_ZIP_CLOSURE")
        for item in members:
            raw = source.read(item)
            if item.filename != "FILES.json":
                need({"bytes": len(raw), "sha256": observer().sha(raw)} == pins[item.filename], "SOURCE_PAYLOAD: " + item.filename)
            output = within(destination, item.filename)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(raw)
    verify_documents(destination, plan)
    return destination


def verify_documents(source_root, plan):
    for name, pin in plan["source_document_pins"].items():
        path = within(source_root, name)
        need(path.is_file() and path.stat().st_size == pin["bytes"] and digest(path) == pin["sha256"],
             "ORIGINAL_DOCUMENT_BYTES: " + name)
    admission = read_json(within(source_root, "ADMISSION.json"))
    result = read_json(within(source_root, "RESULT.json"))
    need(admission["status"] == plan["original_stop_preserved"] and admission["training_allowed"] is False
         and admission["permanently_excluded_from_training"] is False, "ORIGINAL_STOP_AND_ELIGIBILITY_UNCHANGED")
    need(result["stage"] == "combine" and result["status"] == "MACHINE_EVIDENCE_PACKET_RETAINED"
         and result["run"] == plan["source_run"] and result["actual_total_model_calls"] == 720,
         "ORIGINAL_COMBINED_RECEIPT")


def authenticated_source(source_root, plan):
    verify_documents(source_root, plan)
    return observer().source_readback(source_root, plan)


def fetch_source(plan, output):
    root = Path(output).absolute()
    need(not root.exists() and root.parent.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
         "FRESH_SOURCE_DOWNLOAD_DIRECTORY")
    root.mkdir(mode=0o700)
    deadline = time.monotonic() + BUDGETS["download"]
    receipt = {"status": "STARTED", "source_artifact": plan["source_artifact"], "commands": [],
        "new_model_calls": 0, "retry_count": 0}
    def command(name, endpoint):
        record = {"name": name, "endpoint": endpoint, "status": "STARTED"}
        receipt["commands"].append(record)
        write_json(root / "DOWNLOAD.json", receipt)
        env = utility.clean_environment()
        env["GH_TOKEN"] = os.environ.get("GH_TOKEN", "")
        need(bool(env["GH_TOKEN"]), "READ_ONLY_ACTIONS_TOKEN")
        env["GH_HOST"] = "github.com"
        with (root / name).open("xb") as stdout, (root / (name + ".stderr.txt")).open("xb") as stderr:
            process = subprocess.run(["gh", "api", endpoint], env=env, stdout=stdout, stderr=stderr,
                timeout=utility.remaining(deadline), check=False)
        record.update(status="RETURNED", returncode=process.returncode)
        write_json(root / "DOWNLOAD.json", receipt)
        need(process.returncode == 0, "ONE_SHOT_SOURCE_DOWNLOAD_FAILED: " + name)
        return root / name
    try:
        prefix = "repos/" + REPOSITORY + "/actions/"
        source = plan["source_artifact"]
        run = read_json(command("RUN_API.json", prefix + "runs/" + str(source["run_id"])))
        artifact = read_json(command("ARTIFACT_API.json", prefix + "artifacts/" + str(source["artifact_id"])))
        validate_source_api(run, artifact, plan)
        archive = command("source.zip", prefix + "artifacts/" + str(source["artifact_id"]) + "/zip")
        destination = extract_archive(archive, root / "corpus", plan)
        proof = authenticated_source(destination, plan)
        write_json(root / "SOURCE_READBACK.json", proof)
        receipt.update(status="COMPLETE", archive_bytes=archive.stat().st_size, archive_sha256=digest(archive),
                       source_ready=True, actual_source_model_calls=proof["actual_source_model_calls"])
        return receipt
    except Exception as exc:
        receipt.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc), source_ready=False)
        raise
    finally:
        write_json(root / "DOWNLOAD.json", receipt)


def paired_comparison(source_root, plan, packet):
    original = {r["id"]: r["observations"] for r in read_json(Path(source_root) / "MACHINE_EVIDENCE.json")["rows"]}
    current = {r["id"]: r for r in packet["rows"]}
    need(len(current) == 24 and list(current) == [r["id"] for r in plan["panel_rows"]], "ALL_24_COMPARISON_ROWS")
    rows = []
    for row in plan["panel_rows"]:
        old, new = original[row["id"]], current[row["id"]]
        success = new["status"] == "success"
        rows.append({**{k: row[k] for k in ("id", "preset_id", "cell_id", "speed", "requested_text", "pcm_sha256", "pcm_samples")},
            "source_run": plan["source_run"], "run": packet["summary"]["run"],
            "status": new["status"], "xlsr_phone_tokens": old["phoneme"]["phone_tokens"],
            "lv60_phone_tokens": new.get("phone_tokens"),
            "exact_token_sequence_equal": old["phoneme"]["phone_tokens"] == new["phone_tokens"] if success else None,
            "original_asr": {p: old[p]["raw_transcript"] for p in ("qwen", "whisper", "parakeet")},
            "xlsr_capture": {k: old["phoneme"][k] for k in ("capture_path", "capture_sha256")},
            "lv60_capture": {k: new.get(k) for k in ("capture_path", "capture_sha256", "normalized_input_sha256")},
            "xlsr_reference": new.get("xlsr_reference"), "diagnostic_only": True, "training_allowed": False})
    return {"schema": "english-mero-meh-lv60-comparison-v1", "status": "DIAGNOSTIC_ONLY", "denominator": 24,
        "rows": rows, "complete_pairs": sum(r["status"] == "success" for r in rows),
        "exact_equal_pairs": sum(r["exact_token_sequence_equal"] is True for r in rows),
        "phone_normalization_applied": False, "observer_disagreement_is_gold": False,
        "admission_recomputed": False, "original_stop_preserved": plan["original_stop_preserved"], "new_model_calls": 0}


def finalize(plan, source_root, observer_root, output, run, outcome):
    output = Path(output).absolute()
    need(not output.exists() and output.parent.is_dir(), "FRESH_FINAL_DIAGNOSTIC_DIRECTORY")
    output.mkdir()
    result = {"schema": "english-mero-meh-lv60-result-v1", "status": "DIAGNOSTIC_ONLY", "stage": "readback",
        "campaign_id": CAMPAIGN, "run": run, "source_run": plan["source_run"],
        "source_artifact": plan["source_artifact"], "denominator": 24,
        "workflow_observation_outcome": outcome, "evidence_verified": False, "complete": False,
        "actual_forward_attempts": None, "attempts_lower_bound": 0, "attempts_upper_bound": 24,
        "missing_terminal_is_zero": False, "new_model_calls": 0, "new_network_calls": 0, **FLAGS}
    error = None
    try:
        proof = authenticated_source(source_root, plan)
        write_json(output / "SOURCE_READBACK.json", proof)
        packet = observer().readback(source_root, observer_root, plan, expected_run=run)
        write_json(output / "OBSERVER_READBACK.json", packet)
        write_json(output / "COMPARISON.json", paired_comparison(source_root, plan, packet))
        summary = packet["summary"]
        result.update(evidence_verified=True, complete=summary["status"] == "COMPLETE", observer_status=summary["status"],
            **{k: summary[k] for k in ("actual_forward_attempts", "native_forward_returns", "completed_count", "failed_count",
                "not_started_count", "attempts_lower_bound", "attempts_upper_bound", "unknown_count", "terminal_receipt_present")})
        verify_documents(source_root, plan)
    except Exception as exc:
        error = exc
        result.update(readback_error=type(exc).__name__ + ": " + str(exc), observer_status="UNVERIFIED_OR_ABSENT")
    finally:
        for name in sorted(plan["source_files_sha256"]):
            path = within(output / "frozen_source", name)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(within(ROOT, name), path)
        write_json(output / "EXECUTION_PLAN.json", plan)
        write_json(output / "frozen_source/automation/english_mero_meh_lv60_plan.json", plan)
        write_json(output / "RESULT.json", result)
        write_json(output / "FILES.json", {"schema": "english-mero-meh-lv60-diagnostic-files-v1",
            "files": observer().file_records(output)})
    if error is not None:
        raise error
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for flag in ("source-check", "guard", "fetch-source", "install", "observe", "finalize"):
        modes.add_argument("--" + flag, action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--observer")
    parser.add_argument("--output")
    parser.add_argument("--outcome", default="unknown")
    args = parser.parse_args(argv)
    plan = read_json(PLAN)
    checked = source_preflight(plan)
    if not any((args.guard, args.fetch_source, args.install, args.observe, args.finalize)):
        print(json.dumps(checked, sort_keys=True))
        return 0
    run = execution_identity(plan)
    if args.guard:
        print(json.dumps({**checked, "run": run, "source_run": plan["source_run"]}, sort_keys=True))
        return 0
    need(args.output is not None, "OUTPUT_REQUIRED")
    if args.fetch_source:
        report = fetch_source(plan, args.output)
    elif args.install:
        need(args.source is not None, "SOURCE_REQUIRED")
        proof = authenticated_source(args.source, plan)
        report = observer().install(plan, args.output)
        write_json(Path(args.output) / "SOURCE_READBACK.json", proof)
    elif args.observe:
        need(args.source is not None, "SOURCE_REQUIRED")
        authenticated_source(args.source, plan)
        report = observer().observe(plan, args.source, args.output, run)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "COMPLETE" else 1
    else:
        need(args.source is not None and args.observer is not None, "SOURCE_AND_OBSERVER_REQUIRED")
        report = finalize(plan, args.source, args.observer, args.output, run, args.outcome)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
