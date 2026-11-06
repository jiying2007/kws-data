#!/usr/bin/env python3
"""Offline recovery from confirmed ranges of one genuinely failed bounded fetch."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import wave
import run_newsource8_intake as common

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    started = time.monotonic_ns()
    deadline = time.monotonic() + 300
    if common.pin(args.request)["sha256"] != args.request_sha256:
        raise ValueError("RECOVERY_REQUEST_CHANGED")
    req = common.read(args.request)
    if (req["schema"] != "kws-newsource8-offline-recovery-v1" or req["worker_seconds"] != 300
            or req["max_new_speakers"] != 8 or req["network_calls_allowed"] != 0
            or req["model_calls_allowed"] != 0):
        raise ValueError("RECOVERY_FIXED_BUDGET")
    root, source, original = map(Path, (req["runtime_root"], req["source_root"], req["original_runtime"]))
    for item in req["source_and_input_pins"]:
        if common.pin(item["path"]) != item:
            raise ValueError("RECOVERY_PIN_CHANGED: " + item["path"])
    pinned = {x["path"]: x for x in req["source_and_input_pins"]}
    if len(pinned) != len(req["source_and_input_pins"]):
        raise ValueError("DUPLICATE_SOURCE_PIN")
    for item in req["confirmed_ranges"] + req["http_request_pins"] + req["http_response_pins"]:
        if pinned.get(item["path"]) != item:
            raise ValueError("UNPINNED_ORIGINAL_RANGE_EVIDENCE")
    for name in ("terminal.json", "driver/fetch/receipt.json"):
        path = original / name
        if pinned.get(str(path)) != common.pin(path):
            raise ValueError("UNPINNED_ORIGINAL_TERMINAL")
    if pinned.get(req["original_launch_receipt"]) != common.pin(req["original_launch_receipt"]):
        raise ValueError("UNPINNED_ORIGINAL_LAUNCH_RECEIPT")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != req["source_commit"]:
        raise ValueError("RECOVERY_SOURCE_COMMIT_CHANGED")
    old_terminal = common.read(original / "terminal.json")
    old_fetch = common.read(original / "driver/fetch/receipt.json")
    old_launch = common.read(req["original_launch_receipt"])
    if (old_terminal["status"] != "failed" or old_launch["returncode"] != 1
            or (old_fetch["returncode"] == 0 and not old_fetch["timed_out"])):
        raise ValueError("ONLY_ACTUAL_FAILED_FETCH_CAN_RECOVER")
    if old_terminal["new_model_calls"] != 0:
        raise ValueError("UNEXPECTED_MODEL_EXECUTION")
    root.mkdir(parents=True, exist_ok=True)
    common.put(root / "execution-claim.json", {"request": common.pin(args.request), "started_at": common.utc(),
        "source_commit": req["source_commit"], "worker_seconds": 300, "network_calls_allowed": 0,
        "model_calls_allowed": 0, "original_failed_fetch_preserved": True, "automatic_retries": 0})
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = req["python_library_path"]
    base = [req["python"], "-E", "-S", "-B"]
    commands, count, error, status = [], 0, None, "failed"

    def run(name, argv, cap):
        limit = min(cap, deadline - time.monotonic() - 15)
        if limit <= 0:
            raise TimeoutError("OFFLINE_RECOVERY_DEADLINE")
        folder = root / "driver" / name
        folder.mkdir(parents=True, exist_ok=False)
        common.put(folder / "command.json", {"argv": argv, "cwd": str(source),
            "started_at": common.utc(), "timeout_seconds": limit, "automatic_retry": False,
            "environment_override": {"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]}})
        began, timeout = time.monotonic_ns(), False
        with (folder / "stdout.log").open("xb") as out, (folder / "stderr.log").open("xb") as err:
            child = subprocess.Popen(argv, cwd=source, env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                rc = child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timeout = True
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    rc = child.wait()
        receipt = {"returncode": rc, "timed_out": timeout, "elapsed_ns": time.monotonic_ns() - began,
            "command": common.pin(folder / "command.json"), "stdout": common.pin(folder / "stdout.log"),
            "stderr": common.pin(folder / "stderr.log"), "finished_at": common.utc()}
        common.put(folder / "receipt.json", receipt)
        commands.append(common.pin(folder / "receipt.json"))
        if rc or timeout:
            raise RuntimeError("OFFLINE_CHILD_FAILURE: " + name)

    try:
        prefix = Path(req["original_prefix"]["path"])
        if common.pin(prefix) != req["original_prefix"]:
            raise ValueError("ORIGINAL_PARTIAL_BYTES_CHANGED")
        receipts = [common.read(x["path"]) for x in req["confirmed_ranges"]]
        if not 1 <= len(receipts) <= 128:
            raise ValueError("CONFIRMED_RANGE_COUNT")
        if {x["path"] for x in req["confirmed_ranges"]} != {str(x) for x in original.glob("aishell-prefix128m.tgz.partial.range-*.json")}:
            raise ValueError("CONFIRMED_RANGE_ROSTER_CHANGED")
        verified, next_byte = [], 0
        requests_by_range = {}
        for claim_pin in req["http_request_pins"]:
            claim = common.read(claim_pin["path"])
            headers = {k.lower(): v for k, v in claim["request_headers"].items()}
            if claim["url"] != req["url"] or claim["method"] != "GET" or headers.get("if-range") != req["etag"]:
                raise ValueError("ORIGINAL_HTTP_IDENTITY")
            requests_by_range.setdefault(headers.get("range"), []).append(Path(claim_pin["path"]).parent)
        if any(len(x) > 2 for x in requests_by_range.values()):
            raise ValueError("ORIGINAL_RANGE_ATTEMPT_BOUND")
        with prefix.open("rb") as stream:
            for record in receipts:
                if (record["url"] != req["url"] or record["etag"] != req["etag"]
                        or record["archive_bytes"] != req["archive_bytes"] or record["start"] != next_byte
                        or record["bytes"] != 1048576 or record["end"] != next_byte + 1048575):
                    raise ValueError("RANGE_IDENTITY_OR_CONTIGUITY")
                raw = stream.read(record["bytes"])
                if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                    raise ValueError("CONFIRMED_RANGE_BODY_CHANGED")
                valid_headers = []
                key = f"bytes={record['start']}-{record['end']}"
                for attempt in requests_by_range.get(key, []):
                    response_path = attempt / "response.json"
                    if response_path.is_file():
                        if pinned.get(str(response_path)) != common.pin(response_path):
                            raise ValueError("UNPINNED_HTTP_RESPONSE")
                        response = common.read(response_path)
                        hdr = response.get("response_headers", {})
                        if (response.get("http_status") == 206 and
                                response.get("status") == "headers_returned_body_not_yet_verified"
                                and hdr.get("ETag") == req["etag"] and
                                hdr.get("Content-Range") == key.replace("=", " ") + "/" + str(req["archive_bytes"])):
                            valid_headers.append(common.pin(response_path))
                if len(valid_headers) != 1:
                    raise ValueError("CONFIRMED_RANGE_NEEDS_ONE_ACTUAL_206_RESPONSE")
                verified.append({"range": record, "actual_response": valid_headers[0]})
                next_byte += record["bytes"]
        if next_byte != req["confirmed_prefix_bytes"] or next_byte > req["original_prefix"]["bytes"] or next_byte > 134217728:
            raise ValueError("CONFIRMED_LENGTH_CHANGED")
        destination = root / "verified-prefix.tgz.partial"
        with prefix.open("rb") as src, destination.open("xb") as dst:
            left = next_byte
            while left:
                part = src.read(min(left, 1048576))
                if not part:
                    raise ValueError("COPY_SOURCE_TRUNCATED")
                dst.write(part)
                left -= len(part)
            dst.flush()
            os.fsync(dst.fileno())
        if common.pin(prefix) != req["original_prefix"]:
            raise ValueError("ORIGINAL_CHANGED_DURING_RECOVERY")
        common.put(root / "verified-prefix-provenance.json", {"original_prefix": req["original_prefix"],
            "derived_prefix": common.pin(destination), "confirmed_prefix_bytes": next_byte,
            "unconfirmed_original_tail_bytes_preserved": req["original_prefix"]["bytes"] - next_byte,
            "verified_complete_ranges": verified, "original_fetch_status": "failed",
            "original_fetch_success_claimed": False, "new_network_calls": 0,
            "full_archive_verified": False, "not_a_retrieval_success_receipt": True})
        for name in ("exclusion32-registry.json", "exclusion32-source-identities.json",
                     "terms-snapshot.txt", "research-scope.json", "research-review.txt"):
            common.copy_once(original / name, root / name)
        before = common.read(root / "exclusion32-registry.json")
        old_sources = common.read(root / "exclusion32-source-identities.json")["rows"]
        excluded = {x["speaker_key"] for x in before["entries"]}
        if len(excluded) != 32 or len(before["entries"]) != 32 or len(old_sources) != 32:
            raise ValueError("ORIGINAL32_COVERAGE")
        subset = root / "subset"
        run("select", base + [str(source / "automation/fetch_aishell_subset.py"), "select", "--prefix",
            str(destination), "--output", str(subset), "--speaker-registry", str(root / "exclusion32-registry.json"),
            "--max-speakers", "8"], 120)
        refs = common.read(subset / "source-subset.json")["references"]
        keys = {x["speaker_key"] for x in refs}
        count = len(refs)
        if len(keys) != count or count > 8 or keys & excluded:
            raise ValueError("RECOVERY_MUST_ONLY_SELECT_NEW_SOURCES")
        scope = common.read(root / "research-scope.json")
        common.put(root / "selection.json", {"schema": "kws-private-open-corpus-wav-selection-v1",
            "dataset_id": scope["dataset_id"], "dataset_release": scope["dataset_release"],
            "license_snapshot_sha256": scope["source_terms_sha256"], "clips":
            [{"speaker_key": x["speaker_key"], "original_path": x["original_path"],
              "original_sha256": x["sha256"]} for x in refs]})
        if count:
            prepared = root / "prepared"
            run("normalize", base + [str(source / "automation/prepare_open_reference_pilot.py"), "--selection",
                str(root / "selection.json"), "--original-root", str(subset / "original"),
                "--license-snapshot", str(root / "terms-snapshot.txt"), "--output-root", str(prepared),
                "--max-speakers", "8", "--ffmpeg", req["ffmpeg"]["path"],
                "--ffmpeg-sha256", req["ffmpeg"]["sha256"], "--execute"], 180)
            for item in common.read(prepared / "source-index.json")["rows"]:
                with wave.open(str(prepared / "normalized" / item["normalized_path"]), "rb") as stream:
                    pcm_sha = hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest()
                if (item["original_sha256"] in {x["original_sha256"] for x in old_sources}
                        or item["normalized_sha256"] in {x["wav_sha256"] for x in old_sources}
                        or pcm_sha in {x["pcm_sha256"] for x in old_sources}):
                    raise ValueError("RECOVERY_AUDIO_OVERLAPS_ORIGINAL32")
            common.copy_once(root / "exclusion32-registry.json", root / "speakers.json")
            run("registry", base + ["-m", "kws_data_contract.source_registry", "--registry",
                str(root / "speakers.json"), "--index", str(prepared / "source-index.json"), "--execute"], 20)
            after = common.read(root / "speakers.json")
            delta = after["entries"][32:]
            if (after["registry_id"] != before["registry_id"] or after["split_policy"] != before["split_policy"]
                    or after["entries"][:32] != before["entries"] or len(delta) != count
                    or {x["speaker_key"] for x in delta} != keys
                    or {x["source_family_id"] for x in delta} & {x["source_family_id"] for x in before["entries"]}):
                raise ValueError("RECOVERY_REGISTRY_ANCESTRY")
            common.put(root / "registry-delta.json", {"before": common.pin(root / "exclusion32-registry.json"),
                "after": common.pin(root / "speakers.json"), "new_entries": delta,
                "new_roles": dict(Counter(x["role"] for x in delta)), "old_entries_unchanged": True})
            parts = sorted({x["original_partition"] for x in refs})
            transcript = subset / "original" / parts[0] / "content.txt" if len(parts) == 1 and parts[0] in ("train", "test") else None
            if transcript is not None and not transcript.is_file():
                transcript = None
            common.put(root / "reference-text-decision.json", {"source_partitions": parts,
                "official_transcript": common.pin(transcript) if transcript is not None else None,
                "reference_text_status": "dataset-provided" if transcript is not None else "unknown",
                "content_checked": False, "human_gold": False})
            argv = base + [str(source / "automation/research_only_bulk_ingest.py"), "--index",
                str(prepared / "source-index.json"), "--original-root", str(subset / "original"),
                "--audio-root", str(prepared / "normalized"), "--conversion-root", str(prepared / "conversions"),
                "--license-snapshot", str(root / "terms-snapshot.txt"), "--research-scope",
                str(root / "research-scope.json"), "--research-review-evidence", str(root / "research-review.txt"),
                "--speaker-registry", str(root / "speakers.json"), "--output", str(root / "sources.json"),
                "--max-refs", "8", "--execute"]
            if transcript is not None:
                argv += ["--transcript-source", str(transcript), "--transcript-sha256",
                         common.pin(transcript)["sha256"], "--transcript-format", "aishell3-content-v1"]
            run("ingest", argv, 45)
        status = "completed_eight_new_sources" if count == 8 else "completed_partial_new_sources" if count else "bounded_no_new_sources"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    files = [common.pin(p) for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
             and not p.name.endswith(".lock") and p.name != "terminal.json"]
    result = {"schema": "kws-newsource8-offline-terminal-v1", "status": status, "finished_at": common.utc(),
        "elapsed_ns": time.monotonic_ns() - started, "request": common.pin(args.request), "driver": common.pin(__file__),
        "actual_selected_new_sources": count, "original_fetch_status": "failed", "new_network_calls": 0,
        "new_model_calls": 0, "new_training_steps": 0, "queue_initialized": False,
        "training_admitted": False, "reference_content_checked": False, "human_gold": False,
        "commands": commands, "error": error, "files": files}
    common.put(root / "terminal.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "error")}, sort_keys=True))
    return 0 if status != "failed" else 1

if __name__ == "__main__":
    raise SystemExit(main())

