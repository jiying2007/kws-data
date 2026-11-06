#!/usr/bin/env python3
"""One bounded, zero-model new-source intake; compose existing CLI implementations."""
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

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def pin(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("REGULAR_FILE_REQUIRED: " + str(path))
    value = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            value.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": value.hexdigest()}

def read(path):
    return json.loads(Path(path).read_text())

def put(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

def copy_once(source, target):
    with Path(target).open("xb") as f:
        f.write(Path(source).read_bytes())
        f.flush()
        os.fsync(f.fileno())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    if pin(args.request)["sha256"] != args.request_sha256:
        raise ValueError("REQUEST_CHANGED")
    request = read(args.request)
    if (request["schema"] != "kws-newsource8-input-only-v1" or request["prefix_bytes"] != 134217728
            or request["max_new_speakers"] != 8 or request["model_calls_allowed"] != 0
            or request["stage_seconds"] != 1200 or request["fetch_seconds"] != 600):
        raise ValueError("FROZEN_TRANCHE_CONTRACT")
    root, source = Path(request["runtime_root"]), Path(request["source_root"])
    if not root.is_dir() or root.is_symlink() or not source.is_dir():
        raise ValueError("EXISTING_PRIVATE_REQUEST_ROOT_REQUIRED")
    for item in request["source_and_input_pins"]:
        if pin(item["path"]) != item:
            raise ValueError("PINNED_INPUT_CHANGED: " + item["path"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != request["source_commit"]:
        raise ValueError("SOURCE_COMMIT_CHANGED")
    before = read(root / "exclusion32-registry.json")
    old_sources = read(root / "exclusion32-source-identities.json")["rows"]
    excluded = sorted(x["speaker_key"] for x in before["entries"] if x["dataset_id"] == "aishell3-slr93")
    if len(before["entries"]) != 32 or len(set(excluded)) != 32 or excluded != request["excluded_speaker_keys"]:
        raise ValueError("EXCLUSION_32_CHANGED")
    if len(old_sources) != 32 or {x["source_family_id"] for x in old_sources} != {x["source_family_id"] for x in before["entries"]}:
        raise ValueError("EXCLUSION_SOURCE_FAMILY_COVERAGE")
    started = time.monotonic_ns()
    deadline = time.monotonic() + request["stage_seconds"]
    put(root / "execution-claim.json", {"started_at": utc(), "request": pin(args.request),
        "driver": pin(__file__), "maximum_wall_seconds": request["stage_seconds"],
        "model_calls_allowed": 0, "whole_fetch_invocations_allowed": 1,
        "max_range_bytes": 1048576, "range_count_if_complete": 128,
        "range_attempts_maximum": 2, "automatic_whole_fetch_retries": 0})
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = request["python_library_path"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    python = request["python"]
    base = [python, "-E", "-S", "-B"]
    commands = []
    status, error, chosen = "failed", None, 0

    def run(name, argv, seconds):
        allowance = min(float(seconds), deadline - time.monotonic() - 30)
        if allowance <= 0:
            raise TimeoutError("GLOBAL_PREPARATION_BUDGET_EXHAUSTED")
        folder = root / "driver" / name
        folder.mkdir(parents=True, exist_ok=False)
        put(folder / "command.json", {"argv": argv, "cwd": str(source), "started_at": utc(),
            "timeout_seconds": allowance, "environment_override":
            {"LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
            "request": pin(args.request), "automatic_retry": False})
        began, timed_out = time.monotonic_ns(), False
        code = None
        with (folder / "stdout.log").open("xb") as out, (folder / "stderr.log").open("xb") as err:
            process = subprocess.Popen(argv, cwd=source, env=env, stdout=out, stderr=err,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=allowance)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    try:
                        code = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                finally:
                    # A direct child can exit while a descendant ignores TERM.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = process.wait()
        receipt = {"returncode": code, "timed_out": timed_out, "finished_at": utc(),
                   "elapsed_ns": time.monotonic_ns() - began, "command": pin(folder / "command.json"),
                   "stdout": pin(folder / "stdout.log"), "stderr": pin(folder / "stderr.log")}
        put(folder / "receipt.json", receipt)
        commands.append(pin(folder / "receipt.json"))
        if code != 0 or timed_out:
            raise RuntimeError("CHILD_TERMINAL_FAILURE: " + name)

    try:
        prefix, subset = root / "aishell-prefix128m.tgz.partial", root / "subset"
        run("fetch", base + [str(source / "automation/recorded_aishell_fetch.py"), "--output", str(prefix),
            "--etag", request["etag"], "--bytes", str(request["prefix_bytes"]),
            "--attempts", str(root / "http-attempts")], request["fetch_seconds"])
        retrieval = read(prefix.with_name(prefix.name + ".retrieval.json"))
        if retrieval["archive_bytes"] != request["archive_bytes"] or retrieval["prefix_bytes"] != request["prefix_bytes"]:
            raise ValueError("ARCHIVE_GEOMETRY_CHANGED")
        run("select", base + [str(source / "automation/fetch_aishell_subset.py"), "select",
            "--prefix", str(prefix), "--output", str(subset), "--speaker-registry",
            str(root / "exclusion32-registry.json"), "--max-speakers", "8"], 150)
        selected = read(subset / "source-subset.json")
        refs = selected["references"]
        keys = [r["speaker_key"] for r in refs]
        chosen = len(refs)
        if len(set(keys)) != chosen or chosen > 8 or set(keys) & set(excluded):
            raise ValueError("SOURCE_SELECTION_NOT_NEW_AND_UNIQUE")
        scope = read(root / "research-scope.json")
        selection = {"schema": "kws-private-open-corpus-wav-selection-v1", "dataset_id": scope["dataset_id"],
            "dataset_release": scope["dataset_release"], "license_snapshot_sha256": scope["source_terms_sha256"],
            "clips": [{"speaker_key": r["speaker_key"], "original_path": r["original_path"],
                       "original_sha256": r["sha256"]} for r in refs]}
        put(root / "selection.json", selection)
        if not chosen:
            status = "bounded_no_new_sources"
        else:
            prepared = root / "prepared"
            run("normalize", base + [str(source / "automation/prepare_open_reference_pilot.py"),
                "--selection", str(root / "selection.json"), "--original-root", str(subset / "original"),
                "--license-snapshot", str(root / "terms-snapshot.txt"), "--output-root", str(prepared),
                "--max-speakers", "8", "--ffmpeg", request["ffmpeg"]["path"],
                "--ffmpeg-sha256", request["ffmpeg"]["sha256"], "--execute"], 300)
            for item in read(prepared / "source-index.json")["rows"]:
                wav = prepared / "normalized" / item["normalized_path"]
                with wave.open(str(wav), "rb") as stream:
                    pcm_sha = hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest()
                if (item["original_sha256"] in {x["original_sha256"] for x in old_sources}
                        or item["normalized_sha256"] in {x["wav_sha256"] for x in old_sources}
                        or pcm_sha in {x["pcm_sha256"] for x in old_sources}):
                    raise ValueError("NEW_SOURCE_AUDIO_OVERLAPS_OLD32")
            copy_once(root / "exclusion32-registry.json", root / "speakers.json")
            run("registry", base + ["-m", "kws_data_contract.source_registry", "--registry",
                str(root / "speakers.json"), "--index", str(prepared / "source-index.json"), "--execute"], 30)
            after = read(root / "speakers.json")
            delta = after["entries"][32:]
            if (after["registry_id"] != before["registry_id"] or after["split_policy"] != before["split_policy"]
                    or after["entries"][:32] != before["entries"] or len(delta) != chosen
                    or {x["speaker_key"] for x in delta} != set(keys)):
                raise ValueError("OLD_IDENTITIES_OR_ASSIGNMENTS_CHANGED")
            if {x["source_family_id"] for x in delta} & {x["source_family_id"] for x in before["entries"]}:
                raise ValueError("NEW_SOURCE_FAMILY_OVERLAPS_OLD32")
            put(root / "registry-delta.json", {"before": pin(root / "exclusion32-registry.json"),
                "after": pin(root / "speakers.json"), "new_entries": delta,
                "new_roles": dict(Counter(x["role"] for x in delta)), "old_entries_unchanged": True})
            partitions = sorted({r["original_partition"] for r in refs})
            transcript = (subset / "original" / partitions[0] / "content.txt"
                          if len(partitions) == 1 and partitions[0] in ("train", "test") else None)
            if transcript is not None and not transcript.is_file():
                transcript = None
            put(root / "reference-text-decision.json", {"source_partitions": partitions,
                "official_transcript": pin(transcript) if transcript is not None else None,
                "reference_text_status": "dataset-provided" if transcript is not None else "unknown",
                "rule": "Use one matching original partition document only. Mixed/missing text stays unknown.",
                "content_checked": False, "human_gold": False})
            ingest = base + [str(source / "automation/research_only_bulk_ingest.py"),
                "--index", str(prepared / "source-index.json"), "--original-root", str(subset / "original"),
                "--audio-root", str(prepared / "normalized"), "--conversion-root", str(prepared / "conversions"),
                "--license-snapshot", str(root / "terms-snapshot.txt"), "--research-scope",
                str(root / "research-scope.json"), "--research-review-evidence", str(root / "research-review.txt"),
                "--speaker-registry", str(root / "speakers.json"), "--output", str(root / "sources.json"),
                "--max-refs", "8", "--execute"]
            if transcript is not None:
                ingest += ["--transcript-source", str(transcript), "--transcript-sha256",
                           pin(transcript)["sha256"], "--transcript-format", "aishell3-content-v1"]
            run("ingest", ingest, 60)
            status = "completed_eight_new_sources" if chosen == 8 else "completed_partial_new_sources"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        files = [pin(p) for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
                 and p.name != "terminal.json" and not p.name.endswith(".lock")]
        result = {"schema": "kws-newsource8-input-terminal-v1", "status": status, "finished_at": utc(),
            "elapsed_ns": time.monotonic_ns() - started, "request": pin(args.request), "driver": pin(__file__),
            "actual_selected_new_sources": chosen, "new_model_calls": 0, "new_training_steps": 0,
            "queue_initialized": False, "old_d2_resubmitted": False, "reference_content_checked": False,
            "training_admitted": False, "human_gold": False, "independent_test": False,
            "whole_fetch_automatically_retried": False, "download_expanded": False,
            "commands": commands, "error": error, "files": files}
        put(root / "terminal.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("files", "error")}, sort_keys=True))
    return 0 if status != "failed" else 1

if __name__ == "__main__":
    raise SystemExit(main())

