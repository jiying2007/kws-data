"""Read-only original SOURCE proof for B22's three never-started text observers.

The original run, all consumed markers, every original source byte, and both
complete phone returns remain immutable. This module performs no synthesis,
decoding, observer inference, KWS inference, model import, or remote mutation.
"""
from __future__ import annotations

import argparse
import ast
import base64
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile
import wave
import zipfile

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "automation/english_v6_b22_source_origin.json"
EVIDENCE = ROOT / "research/english_v6_b22_source_20261124/initial_failure"
REPO = "jiying2007/kws-data"
ORIGINAL_RUN_ID = 45402179335
ORIGINAL_HEAD = ORIGINAL_RUN_HEAD = "54c5f514d0aed0bf276927c8b6631245d833167c"
ORIGINAL_BRANCH = "research/en-v6-b22-primary-sources-once-20261124"
ORIGINAL_WORKFLOW = ".github/workflows/english-v6-b22-primary-sources-once.yml"
ORIGINAL_PARENT = "b13379a895e5229dd2a3503fa246011357041fa2"
ORIGINAL_SOURCE_EXECUTION = dict(run_id=str(ORIGINAL_RUN_ID), run_head=ORIGINAL_HEAD,
    run_branch="refs/heads/" + ORIGINAL_BRANCH, run_attempt=1, prepared_parent=ORIGINAL_PARENT)
TEXT_PROVIDERS = ("qwen", "whisper", "parakeet")
PHONE_PROVIDERS = ("xlsr", "lv60")
PROVIDERS = TEXT_PROVIDERS + PHONE_PROVIDERS
DOWNLOAD_ROLES = ("source", "producer") + PROVIDERS
SCHEMA = "english-v6-b22-verified-source-zero-text-origin-v1"
_REGISTERED = json.loads(SPEC.read_bytes())
ARTIFACTS = {role: {k: v for k, v in _REGISTERED["artifacts"][role].items() if k != "name"}
             for role in DOWNLOAD_ROLES}
COUNTS = ("input_started", "forward_attempted", "native_returned", "raw_payload_saved",
          "payload_valid", "capture_saved", "failed")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def read(path):
    path = Path(path)
    need(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
         "regular original evidence file: " + str(path))
    return path.read_bytes()


def document(path):
    return json.loads(read(path))


def put(path, raw):
    path = Path(path)
    need(type(raw) is bytes and path.parent.is_dir()
         and not any(p.is_symlink() for p in (path, *path.parents)), "new regular origin evidence")
    with path.open("xb") as handle:
        handle.write(raw)


def local():
    proof = document(SPEC)
    need(proof == _REGISTERED and proof["schema"] == "english-v6-b22-original-source-zero-text-continuation-proof-v1"
         and proof["original_run_id"] == ORIGINAL_RUN_ID and proof["original_head"] == ORIGINAL_HEAD
         and proof["original_branch"] == ORIGINAL_BRANCH and proof["original_workflow"] == ORIGINAL_WORKFLOW
         and proof["prepared_parent"] == ORIGINAL_PARENT and proof["producer_file_count"] == 1172
         and proof["original_source_count"] == 120 and proof["original_tts_calls"] == 96
         and proof["original_natural_decodes"] == 24
         and proof["text_forward_counts"] == dict.fromkeys(TEXT_PROVIDERS, 0)
         and proof["phone_forward_counts"] == dict.fromkeys(PHONE_PROVIDERS, 120)
         and proof["remaining_text_forward_calls"] == 360 and proof["original_budget_reset"] is False
         and all(proof[k] == 0 for k in ("new_tts_calls_allowed", "new_phone_calls_allowed",
             "new_natural_decodes_allowed", "new_kws_forwards_allowed")), "unchanged original consumption and exact remainder")
    need(set(proof["artifacts"]) == set(DOWNLOAD_ROLES) | {p + "_consumed" for p in PROVIDERS}
         | {"source_consumed", "transport"} and tuple(proof["download_roles"]) == DOWNLOAD_ROLES,
         "all fourteen original artifacts and only seven required original ZIPs")
    expected = set()
    for name, digest in proof["evidence_files"].items():
        path = ROOT / name
        need(path.is_relative_to(EVIDENCE) and sha(read(path)) == digest, "unchanged frozen origin evidence " + name)
        expected.add(path)
    need({p for p in EVIDENCE.rglob("*") if p.is_file()} == expected,
         "complete frozen origin evidence inventory, including every original consumed marker")
    return proof


def producer_manifest():
    proof = local()
    raw = read(EVIDENCE / "PRODUCER.json")
    need(sha(raw) == proof["producer_sha256"], "original source producer manifest")
    return json.loads(raw)


def validate_live(directory, proof):
    directory = Path(directory)
    run, jobs, artifacts, commit = (document(directory / (name + ".json"))
                                   for name in ("RUN", "JOBS", "ARTIFACTS", "COMMIT"))
    need(run["id"] == ORIGINAL_RUN_ID and run["head_sha"] == ORIGINAL_HEAD
         and run["run_attempt"] == 1 and run["status"] == "completed" and run["conclusion"] == "failure"
         and run["event"] == "push" and run["head_branch"] == ORIGINAL_BRANCH and run["path"] == ORIGINAL_WORKFLOW
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False,
         "original public SOURCE failure remains final, attempt one, and unrerun")
    need(commit["sha"] == ORIGINAL_HEAD and commit["commit"]["tree"]["sha"] == proof["git_tree"]
         and [p["sha"] for p in commit["parents"]] == [ORIGINAL_PARENT]
         and [f["filename"] for f in commit["files"]] == [ORIGINAL_WORKFLOW],
         "the original single-parent workflow-only SOURCE activation")
    need(jobs["total_count"] == len(jobs["jobs"]) == len(proof["jobs"]) == 7
         and {str(j["id"]) for j in jobs["jobs"]} == set(proof["jobs"]), "all seven original jobs, including failed readback")
    for job in jobs["jobs"]:
        pin = proof["jobs"][str(job["id"])]
        need(job["run_id"] == ORIGINAL_RUN_ID and job["head_sha"] == ORIGINAL_HEAD
             and job["run_attempt"] == 1 and job["status"] == "completed"
             and job["name"] == pin["name"] and job["conclusion"] == pin["conclusion"]
             and [s["name"] for s in job["steps"] if s["conclusion"] == "failure"] == pin["failed_steps"],
             "exact original job and failing-step boundary")
        raw = read(directory / "logs" / (str(job["id"]) + ".log"))
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "complete unchanged original live job log")
    pins = {p["id"]: p for p in proof["artifacts"].values()}
    need(artifacts["total_count"] == len(artifacts["artifacts"]) == len(pins) == 14
         and {a["id"] for a in artifacts["artifacts"]} == set(pins), "all original artifacts remain visible, no extra attempt")
    for value in artifacts["artifacts"]:
        pin = pins[value["id"]]
        need(value["name"] == pin["name"] and value["size_in_bytes"] == pin["bytes"]
             and value["digest"] == "sha256:" + pin["sha256"] and value["expired"] is False
             and value["workflow_run"]["id"] == ORIGINAL_RUN_ID
             and value["workflow_run"]["head_sha"] == ORIGINAL_HEAD,
             "original artifact digest, identity, and consumed marker are unchanged")
    return dict(original_run_id=ORIGINAL_RUN_ID, original_head=ORIGINAL_HEAD,
                original_attempt=1, original_artifact_count=14, original_job_count=7)


def original_zip(path, pin):
    raw = read(path)
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "full original ZIP bytes " + str(pin["id"]))
    archive = zipfile.ZipFile(io.BytesIO(raw))
    try:
        infos = archive.infolist()
        need(len(infos) <= 8192 and len(infos) == len({i.filename for i in infos})
             and sum(i.file_size for i in infos) <= 128_000_000, "bounded complete original ZIP")
        for info in infos:
            name = PurePosixPath(info.filename)
            need(name.parts and not name.is_absolute() and ".." not in name.parts
                 and "\\" not in info.filename and not info.is_dir()
                 and not stat.S_ISLNK(info.external_attr >> 16), "safe regular original archive member")
        need(archive.testzip() is None, "original ZIP CRC")
        return archive
    except BaseException:
        archive.close()
        raise


def check_archive(role, archive, proof, inventory):
    need(set(archive.namelist()) == set(inventory[role]), "complete original archive member set " + role)
    for name, pin in inventory[role].items():
        raw = archive.read(name)
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "complete original member " + role + "/" + name)
    for member, path in proof["saved_archive_members"].get(role, {}).items():
        need(archive.read(member) == read(ROOT / path), "archive equals preserved original evidence " + role + "/" + member)


def verify_producer(archive, proof):
    producer = json.loads(archive.read("PRODUCER.json"))
    tree = document(EVIDENCE / "TREE.json")
    need(tree["sha"] == proof["git_tree"] and tree["truncated"] is False
         and producer["git_tree"] == tree["sha"] and len(producer["files"]) == 1172
         and producer["runtime_workflow"] == ORIGINAL_WORKFLOW
         and producer["run_branch"] == ORIGINAL_BRANCH
         and all(producer[k] == v for k, v in ORIGINAL_SOURCE_EXECUTION.items() if k != "run_branch"),
         "complete original producer Git identity")
    entries = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    need(set(archive.namelist()) == {"PRODUCER.json"} | {"repository/" + n for n in producer["files"]},
         "all actual original producer files retained, with no invented runtime files")
    for name, pin in producer["files"].items():
        raw = archive.read("repository/" + name)
        need(len(raw) == pin["bytes"] == entries[name]["size"] and sha(raw) == pin["sha256"]
             and blob(raw) == pin["git_blob_sha1"] == entries[name]["sha"]
             and pin["mode"] == entries[name]["mode"] == "100644", "original compiled/read source was an actual Git blob " + name)
    helper_tree = ast.parse(archive.read("repository/automation/english_pocket12_observe.py"))
    helpers = next(ast.literal_eval(n.value) for n in helper_tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "HELPER_PINS" for t in n.targets))
    missing = {name for name in helpers if name not in producer["files"]}
    need(len(helpers) == 19 and missing == set(proof["missing_helpers"])
         == {"research/auto-tts-voice-clone-20261102/campaign.py", "research/auto-tts-voice-clone-20261102/runner.py"},
         "exact two omitted dynamic helper files; no scientific change")
    for name, digest in helpers.items():
        if name in missing:
            pin = proof["missing_helpers"][name]
            need(pin["sha256"] == digest and pin["git_blob_sha1"] == entries[name]["sha"]
                 and pin["bytes"] == entries[name]["size"], "omitted helper was already immutable original Git source")
        else:
            need(producer["files"][name]["sha256"] == digest, "all seventeen present helper hashes already match")
    adapter_tree = ast.parse(archive.read("repository/automation/english_v6_b18_asr_adapter.py"))
    loader = next(n for n in adapter_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load")
    calls = [n for n in ast.walk(loader) if isinstance(n, ast.Call)]
    check_line = next(n.lineno for n in calls if isinstance(n.func, ast.Attribute)
                      and isinstance(n.func.value, ast.Name) and n.func.value.id == "old" and n.func.attr == "source_check")
    need(all(check_line < n.lineno for n in calls if isinstance(n.func, ast.Attribute)
             and n.func.attr in {"version", "from_pretrained", "load_model", "pinned_model", "lock_model", "verify_runtime", "verify_gguf"}),
         "the missing-file guard precedes every text runtime or model initialization")
    return producer


def source_rows(archive, proof):
    manifest_raw = archive.read("MANIFEST.json")
    manifest = json.loads(manifest_raw)
    rows = manifest["rows"]
    need(sha(manifest_raw) == proof["source_manifest_sha256"]
         and manifest["source_execution"] == ORIGINAL_SOURCE_EXECUTION
         and manifest["selected_original_slots"] == manifest["observer_eligible_originals"] == len(rows) == 120
         and len({r["id"] for r in rows}) == len({r["pcm_sha256"] for r in rows}) == 120
         and manifest["tts_calls"] == manifest["tts_public_synthesize_api_invocations"] == 96
         and manifest["natural_original_decodes"] == 24 and manifest["new_kws_forwards"] == 0
         and manifest["replication_opened"] is manifest["training_allowed"] is manifest["promotion_allowed"] is False,
         "the exact existing 96 synthetic plus 24 whole natural originals remain the sole source")
    need(sum(r["cohort"] == "synthetic" for r in rows) == 96
         and sum(r["cohort"] == "natural" for r in rows) == 24, "complete original source cohorts")
    for row in rows:
        raw = archive.read("wav/" + row["id"] + ".wav")
        need(sha(raw) == row["wav_sha256"], "whole original WAV unchanged")
        with wave.open(io.BytesIO(raw), "rb") as stream:
            need(stream.getnchannels() == 1 and stream.getsampwidth() == 2 and stream.getframerate() == 16000
                 and stream.getcomptype() == "NONE" and stream.getnframes() == row["samples"], "unchanged original mono PCM geometry")
            pcm = stream.readframes(stream.getnframes())
            need(not stream.readframes(1) and sha(pcm) == row["pcm_sha256"], "whole original PCM bytes retained")
    need(archive.read("CONSUMPTION.json") == read(EVIDENCE / "consumed/source/CONSUMPTION.json"),
         "original source consumption is preserved, never reset")
    return rows


def binding(provider, row, proof):
    return dict(id=row["id"], provider=provider, source_manifest_sha256=proof["source_manifest_sha256"],
                source_wav_sha256=row["wav_sha256"], source_pcm_sha256=row["pcm_sha256"], source_samples=row["samples"],
                target_conditioned=False, training_allowed=False)


def observer_receipt(provider, archive, rows, proof):
    report, ledger, started, budget, files = (json.loads(archive.read(name + ".json"))
        for name in ("RESULT", "LEDGER", "STARTED", "OBSERVER_BUDGET", "FILES"))
    receipt = document(EVIDENCE / "consumed" / provider / "OBSERVER_CONSUMPTION.json")
    consumed_raw = read(EVIDENCE / "consumed" / provider / "ORIGINAL.zip")
    consumed_pin = proof["artifacts"][provider + "_consumed"]
    need(len(consumed_raw) == consumed_pin["bytes"] and sha(consumed_raw) == consumed_pin["sha256"]
         and base64.b64decode(budget["original_zip_base64"], validate=True) == consumed_raw
         and budget["receipt"] == receipt and started["registration"] == receipt["registration"]
         and receipt["source_execution"] == ORIGINAL_SOURCE_EXECUTION and receipt["maximum_native_calls"] == 120
         and receipt["provider"] == provider and receipt["status"] == "OBSERVER_BUDGET_CONSUMED_UNKNOWN",
         "independently uploaded original observer budget remains consumed")
    need({p["path"] for p in files["files"]} == set(archive.namelist()) - {"FILES.json"}
         and len(files["files"]) == len(archive.namelist()) - 1, "entire observer archive, no hidden returned payload")
    for pin in files["files"]:
        raw = archive.read(pin["path"])
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "exact original observer FILES closure")
    need(report["provider"] == ledger["provider"] == provider and report["denominator"] == len(rows) == 120
         and report["source_manifest_sha256"] == ledger["source_manifest_sha256"] == proof["source_manifest_sha256"]
         and report["run"] == started["run"] == {"run": str(ORIGINAL_RUN_ID), "sha": ORIGINAL_HEAD}
         and [r["id"] for r in report["rows"]] == [r["id"] for r in ledger["rows"]] == [r["id"] for r in rows]
         and all(report[k] == 0 for k in ("new_tts_calls", "kws_forward_calls", "retry_count"))
         and report["target_conditioned"] is report["human_gold"] is report["training_allowed"] is report["promotion_allowed"] is False,
         "every native slot belongs to the exact original source and first run")
    count = 0 if provider in TEXT_PROVIDERS else 120
    need(all(type(report[k]) is int and report[k] == count for k in
             ("actual_native_forward_attempts", "attempted", "native_forward_returns", "valid_native_payloads", "complete"))
         and report["missing_or_failed"] == 120 - count
         and ledger["counts"] == {k: (0 if k == "failed" else count) for k in COUNTS},
         "separate original attempted, returned, payload-valid, and captured counts")
    expected_ids = [] if count == 0 else [r["id"] for r in rows]
    need(all(ledger[key] == expected_ids for key in ("attempted_ids", "returned_ids", "payload_valid_ids")),
         "exact original per-call ledger, no unaccounted attempt")
    if count == 0:
        need(set(archive.namelist()) == {"FILES.json", "LEDGER.json", "MODEL_LOAD_STARTED.json", "OBSERVER_BUDGET.json", "RESULT.json", "STARTED.json"}
             and json.loads(archive.read("MODEL_LOAD_STARTED.json")) == dict(provider=provider, actual_forward_attempts=0)
             and report["initialization_error"] == proof["initialization_error"] and report["row_error"] is None
             and report["status"] == "STOP_OBSERVER_FAILURE" and report["whole_panel_stop"] is True
             and report["new_asr_calls"] == report["new_phone_calls"] == 0,
             "initialization stopped before model load and before any text source was entered")
        for row, state, result in zip(rows, ledger["rows"], report["rows"], strict=True):
            need(state == dict(id=row["id"], **dict.fromkeys(COUNTS, False))
                 and result == {**binding(provider, row, proof), "status": "not_started"},
                 "all120 text rows were never started, including no ambiguous consumed call")
    else:
        need(report["initialization_error"] is report["row_error"] is None and report["status"] == "COMPLETE"
             and report["whole_panel_stop"] is False and report["new_asr_calls"] == 0 and report["new_phone_calls"] == 120,
             "both original phone observers completed exactly120 calls")
        model = json.loads(archive.read("MODEL.json"))
        for row, state, capture in zip(rows, ledger["rows"], report["rows"], strict=True):
            start = binding(provider, row, proof)
            folder = "rows/" + row["id"] + "/"
            need(capture == json.loads(archive.read(folder + "CAPTURE.json"))
                 and all(capture[k] == v for k, v in start.items()) and capture["status"] == "success"
                 and capture["input_coverage_complete"] is True
                 and all(state[k] is (False if k == "failed" else True) for k in COUNTS),
                 "original phone capture and ledger bind every whole source")
            for name, field, expected in (("FORWARD_STARTED.json", "attempt", 1), ("FORWARD_RETURNED.json", "native_return_observed", True)):
                value = json.loads(archive.read(folder + name))
                need(all(value[k] == v for k, v in start.items()) and value[field] == expected,
                     "original phone call was attempted once and returned")
            evidence = capture["raw_evidence"]
            for key in ("payload", "validation"):
                pin = evidence[key]; raw = archive.read(folder + pin["path"])
                need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "raw phone payload pin")
            raw_phone = json.loads(archive.read(folder + "RAW_PHONE.json"))
            valid = json.loads(archive.read(folder + "PAYLOAD_VALID.json"))
            frames = row["samples"]
            for kernel, stride in zip(model["conv_kernel"], model["conv_stride"], strict=True):
                frames = (frames - kernel) // stride + 1
            logits = archive.read("logits/" + row["id"] + ".f32")
            pin = dict(path="logits/" + row["id"] + ".f32", bytes=len(logits), sha256=sha(logits))
            need(raw_phone == {**start, "schema": "english-v6-b18-native-raw-return-v1", "logits": pin,
                 "shape": [frames, 392], "dtype": "float32-little-endian", "order": "C"}
                 and valid == {**start, "schema": "english-v6-b18-native-raw-return-v1", "payload_valid": True,
                 "raw_phone": evidence["payload"], "logit_frames": frames, "vocab_size": 392, "logits_sha256": pin["sha256"]}
                 and capture["logits_sha256"] == pin["sha256"] and capture["logit_frames"] == frames
                 and capture["vocab_size"] == 392 and len(logits) == 4 * frames * 392,
                 "all original phone arrays are present in their full frame geometry")
    return count


def verify_origin(directory):
    """Offline verification of saved original ZIPs, live receipts, and frozen proof."""
    proof = local()
    directory = Path(directory).resolve()
    validate_live(directory / "LIVE", proof)
    frozen = directory / "FROZEN"
    need(read(frozen / "ORIGIN.json") == read(SPEC), "same frozen original-source registration")
    expected_frozen = {"ORIGIN.json"}
    for name, digest in proof["evidence_files"].items():
        relative = (ROOT / name).relative_to(EVIDENCE)
        need(sha(read(frozen / relative)) == digest, "self-contained preserved original failure evidence")
        expected_frozen.add(relative.as_posix())
    need({p.relative_to(frozen).as_posix() for p in frozen.rglob("*") if p.is_file()} == expected_frozen,
         "complete original proof travels with the continuation source")
    inventory = document(EVIDENCE / "ARCHIVE_FILES.json")
    with ExitStack() as stack:
        archives = {role: stack.enter_context(original_zip(directory / ("source-" + str(pin["id"]) + ".zip"), pin))
                    for role, pin in ARTIFACTS.items()}
        for role, archive in archives.items():
            check_archive(role, archive, proof, inventory)
        verify_producer(archives["producer"], proof)
        rows = source_rows(archives["source"], proof)
        counts = {p: observer_receipt(p, archives[p], rows, proof) for p in PROVIDERS}
    return dict(schema=SCHEMA, status="VERIFIED_ORIGINAL_ZERO_TEXT_CALL_STOP_WITH240_PHONE_RETURNS",
        original_source_execution=ORIGINAL_SOURCE_EXECUTION, source_pin=ARTIFACTS["source"],
        source_sha256=ARTIFACTS["source"]["sha256"], source_manifest_sha256=proof["source_manifest_sha256"],
        original_source_count=120, original_tts_calls=96, original_natural_decodes=24,
        phone_forward_counts={p: counts[p] for p in PHONE_PROVIDERS},
        text_forward_counts={p: counts[p] for p in TEXT_PROVIDERS}, three_text_new_model_forwards=0,
        remaining_text_forward_calls=360, original_artifact_count=14, original_producer_files=1172,
        original_consumption_preserved=True, original_failure_preserved=True, original_budget_reset=False,
        new_tts_calls=0, new_natural_decodes=0, new_phone_calls=0, new_kws_forwards=0,
        original_zip_pins=ARTIFACTS, origin_registration_sha256=sha(read(SPEC)))


def gh(endpoint):
    return subprocess.run(["gh", "api", "repos/" + REPO + "/" + endpoint, "--allow-escape-sequences"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=180).stdout


def _fetch_live(directory, proof):
    directory.mkdir()
    (directory / "logs").mkdir()
    for name, endpoint in (("RUN", "actions/runs/" + str(ORIGINAL_RUN_ID)),
        ("JOBS", "actions/runs/" + str(ORIGINAL_RUN_ID) + "/jobs?filter=all&per_page=100"),
        ("ARTIFACTS", "actions/runs/" + str(ORIGINAL_RUN_ID) + "/artifacts?per_page=100"),
        ("COMMIT", "commits/" + ORIGINAL_HEAD)):
        put(directory / (name + ".json"), gh(endpoint))
    for job_id in proof["jobs"]:
        put(directory / "logs" / (job_id + ".log"), gh("actions/jobs/" + job_id + "/logs"))
    return validate_live(directory, proof)


def verify_live():
    proof = local()
    with tempfile.TemporaryDirectory(prefix="b22-source-origin-live-") as temporary:
        return _fetch_live(Path(temporary) / "LIVE", proof)


def fetch_origin(directory):
    """GET the seven immutable ZIPs and full live receipts; create no new call."""
    proof = local()
    directory = Path(directory).resolve()
    need(not directory.exists() and directory.parent.is_dir(), "fresh original evidence destination")
    directory.mkdir()
    _fetch_live(directory / "LIVE", proof)
    frozen = directory / "FROZEN"
    frozen.mkdir()
    put(frozen / "ORIGIN.json", read(SPEC))
    for name in proof["evidence_files"]:
        relative = (ROOT / name).relative_to(EVIDENCE)
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        put(target, read(ROOT / name))
    for role, pin in ARTIFACTS.items():
        raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "downloaded exact original " + role)
        put(directory / ("source-" + str(pin["id"]) + ".zip"), raw)
    return verify_origin(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "live", "fetch", "verify"))
    parser.add_argument("directory", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "check":
        result = local()
        print(json.dumps(dict(schema=result["schema"], evidence_files=len(result["evidence_files"]),
                             text_forward_counts=result["text_forward_counts"], phone_forward_counts=result["phone_forward_counts"])))
    else:
        if args.mode == "live":
            result = verify_live()
        else:
            need(args.directory is not None, "origin directory required")
            result = fetch_origin(args.directory) if args.mode == "fetch" else verify_origin(args.directory)
        print(json.dumps(result, sort_keys=True))
