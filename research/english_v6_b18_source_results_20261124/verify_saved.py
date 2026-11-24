"""Audit retained B18 native receipts only; never initialize or call a model."""
import array
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
REPOSITORY = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "automation"))
import english_v6_b18_source_observe as native
import english_v6_b18_asr_adapter as adapter

need, sha = adapter.need, adapter.sha
read = lambda path: json.loads(path.read_bytes())


def verify():
    artifacts = read(BASE / "evidence/artifacts-final-45324181095.json")["artifacts"]
    meta = read(BASE / "MANIFEST.json")
    rows = meta["rows"]
    source_sha = sha((BASE / "MANIFEST.json").read_bytes())
    need(source_sha == "a75645e42cda5fe4dbf824bc2a72478b5b62086d2c6b3a7ea23bc8728c62224c", "original source manifest")
    code = native.source_check()
    expected_registration = dict(assessment_sha256=meta["assessment_sha256"], source_schema=meta["schema"],
        batch_schema=native.BATCH, source_manifest_sha256=source_sha, source_rows=96, max_calls=96,
        **meta["source_execution"], parser_contract_sha256=code["parser_contract_sha256"],
        observer_sha256=code["observer_sha256"], asr_adapter_sha256=code["asr_adapter_sha256"])
    need(meta["source_execution"] == read(BASE / "AUTHORIZATION.json")["source_execution"], "original run authorization")
    run = read(BASE / "run-final-45324181095.json")
    need(run["id"] == 45324181095 and run["head_sha"] == meta["source_execution"]["run_head"]
         and run["status"] == "completed" and run["conclusion"] == "failure" and run["run_attempt"] == 1
         and run["event"] == "push" and run["repository"]["private"] is False, "terminal public first attempt")
    commit = read(BASE / "evidence/commit-7664b4554308fbf2d874fc8873e77a1095707d21.json")
    need(commit["sha"] == run["head_sha"] and len(commit["parents"]) == 1
         and commit["parents"][0]["sha"] == meta["source_execution"]["prepared_parent"]
         and [row["filename"] for row in commit["files"]] ==
             [".github/workflows/english-v6-b18-synthetic-primary-once.yml"], "unchanged preregistered code")
    locks = {}
    for role, artifact_name, filename in [("source", "english-v6-b18-primary-source-45324181095", "synth_source.zip")] + [
            (provider, "b18-primary-dev-observer-" + provider, "synthetic-" + provider + ".zip") for provider in native.PROVIDERS]:
        pin = next(item for item in artifacts if item["name"] == artifact_name)
        raw = (BASE / "artifacts" / filename).read_bytes()
        need(len(raw) == pin["size_in_bytes"] and "sha256:" + sha(raw) == pin["digest"]
             and pin["expired"] is False and pin["workflow_run"]["id"] == run["id"]
             and pin["workflow_run"]["head_sha"] == run["head_sha"], "exact original GitHub artifact " + role)
        with zipfile.ZipFile(BASE / "artifacts" / filename) as archive:
            need(archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist())), "original ZIP CRC")
            if role == "source":
                need(archive.read("MANIFEST.json") == (BASE / "MANIFEST.json").read_bytes(), "root-audited source unchanged")
        locks[role] = dict(id=pin["id"], run_id=run["id"], run_head=run["head_sha"], bytes=len(raw), sha256=sha(raw))
    need(not any(item["name"].startswith("english-v6-b18-primary-frozen-screen-") for item in artifacts), "no synthetic SCREEN exists")
    results = {}
    for provider in native.PROVIDERS:
        root = BASE / "observations" / provider
        closure = read(root / "FILES.json")
        need(closure == {"schema": "english-v6-b18-observer-file-closure-v1", "self_excluded": "FILES.json",
                         "files": native.file_closure(root)}, "original complete byte closure")
        started, report, model = (read(root / name) for name in ("STARTED.json", "RESULT.json", "MODEL.json"))
        need(started["registration"] == expected_registration and started["input_manifest"] == source_sha
             and started["provider"] == provider and started["old_admission_function_sha256"] == native.POLICY_FUNCTIONS
             and report["source_manifest_sha256"] == source_sha and report["denominator"] == 96
             and report["observer_code_sha256"] == code["observer_sha256"]
             and report["asr_adapter_sha256"] == code["asr_adapter_sha256"]
             and report["kws_forward_calls"] == report["new_tts_calls"] == report["retry_count"] == 0,
             "original code, source, model-call and no-replay binding")
        ledger = native.row_ledger(root, rows)
        counts = native.ledger_counts(ledger)
        need(read(root / "LEDGER.json") == {"schema": "english-v6-b18-native-attempt-return-valid-ledger-v1",
             "provider": provider, "source_manifest_sha256": source_sha, "rows": ledger, "counts": counts,
             "attempted_ids": [r["id"] for r in ledger if r["forward_attempted"]],
             "returned_ids": [r["id"] for r in ledger if r["native_returned"]],
             "payload_valid_ids": [r["id"] for r in ledger if r["payload_valid"]]}, "actual ledger reconstructed")
        need(report["actual_native_forward_attempts"] == report["attempted"] == counts["forward_attempted"]
             and report["native_forward_returns"] == counts["native_returned"]
             and report["valid_native_payloads"] == counts["payload_valid"]
             and report["complete"] == counts["capture_saved"], "attempts, returns and validity are distinct")
        by_id = {row["id"]: row for row in report["rows"]}
        need(len(by_id) == len(report["rows"]) == 96 and set(by_id) == {row["id"] for row in rows}, "all original slots accounted")
        for row in rows:
            captured = by_id[row["id"]]
            directory = root / "rows" / row["id"]
            if captured["status"] != "success":
                continue
            start = native.binding(provider, row, source_sha)
            need(read(directory / "STARTED.json") == start and read(directory / "CAPTURE.json") == captured
                 and all(captured[k] == value for k, value in start.items()), "unchanged native input identity")
            if provider in adapter.TEXT_PROVIDERS:
                need(model["observer"] == code["asr_models"][provider]
                     and model["runtime_versions"] == code["asr_runtime_versions"][provider]
                     and model["adapter_sha256"] == code["asr_adapter_sha256"], "same pinned ASR backend")
                adapter.validate_capture(provider, directory, start, captured)
            else:
                native._verify_phone(root, directory, start, captured, model)
                values = array.array("f", (root / "logits" / (row["id"] + ".f32")).read_bytes())
                if sys.byteorder != "little": values.byteswap()
                need(all(math.isfinite(value) for value in values), "original finite native phone array")
        if provider == "parakeet":
            need(counts == dict(input_started=6, forward_attempted=6, native_returned=6,
                 raw_payload_saved=6, payload_valid=5, capture_saved=5, failed=1), "exact original six-attempt failure")
            need(report["status"] == "STOP_OBSERVER_FAILURE" and report["whole_panel_stop"] is True
                 and report["initialization_error"] is None and report["row_error"] == "ValueError: EMPTY_NATIVE_TRANSCRIPT"
                 and sum(row["status"] == "not_started" for row in by_id.values()) == 90,
                 "invalid sixth response stopped all remaining native calls")
        else:
            need(native.complete_ledger(ledger) and report["status"] == "COMPLETE"
                 and report["whole_panel_stop"] is False, "all 96 complete valid originals for " + provider)
        results[provider] = dict(attempted=counts["forward_attempted"], returned=counts["native_returned"],
             valid=counts["payload_valid"], failed=counts["failed"], not_started=96-counts["input_started"])
    failed_id = "b18-primary-767-mero-slow"
    directory = BASE / "observations/parakeet/rows" / failed_id
    row = next(row for row in rows if row["id"] == failed_id)
    start = native.binding("parakeet", row, source_sha)
    attempted, returned, process, failed = (read(directory / name) for name in
        ("FORWARD_STARTED.json", "FORWARD_RETURNED.json", "RAW_PROCESS.json", "FAILED.json"))
    need(all(all(receipt[k] == value for k, value in start.items()) for receipt in (attempted, returned, process, failed)), "failed return source binding")
    stdout = adapter._read_pinned(directory, process["stdout"])
    stderr = adapter._read_pinned(directory, process["stderr"])
    command_raw = adapter._read_pinned(directory, process["command"])
    command = json.loads(command_raw)
    need(attempted["attempt"] == 1 and attempted["native_command_sha256"] == sha(command_raw)
         and command["command_sha256"] == sha(adapter.canonical(command["command"]))
         and returned["native_return_observed"] is process["native_return_observed"] is True
         and returned["native_exit_code"] == process["native_exit_code"] == 0
         and process["process_started"] is True and process["timed_out"] is False and process["exception_type"] is None
         and stdout == b"\n" and len(stderr) == 203 and not (directory / "PAYLOAD_VALID.json").exists()
         and not (directory / "CAPTURE.json").exists(), "exact retained empty native return, never a valid capture")
    try: adapter.normalize_raw(stdout)
    except ValueError as error: need(str(error) == "EMPTY_NATIVE_TRANSCRIPT", "original parser rejection")
    else: raise ValueError("empty payload cannot become valid")
    totals = {key: sum(value[key] for value in results.values()) for key in ("attempted", "returned", "valid", "failed", "not_started")}
    need(totals == dict(attempted=390, returned=390, valid=389, failed=1, not_started=90), "honest complete panel accounting")
    need(read(BASE / "B18_KWS_ASSESSMENT_RUNS.json") == {"total_count": 0, "workflow_runs": []}, "zero B18 KWS runs")
    output = dict(schema="english-v6-b18-original-native-stop-readback-v1", source_gate="STOP_INCOMPLETE_VALID_NATIVE_COVERAGE",
        run_id=run["id"], run_head=run["head_sha"], run_attempt=1, artifacts=locks, synthetic_screen=None,
        provider_counts=results, total_native_observers=totals, original_slots=96, source_families=24,
        source_manifest_sha256=source_sha, source_logical_TTS_calls=96, source_native_output_chunks=96,
        internal_TTS_model_forwards=None, failed_original=dict(id=failed_id, source_samples=row["samples"],
          pcm_sha256=row["pcm_sha256"], native_exit_code=0, raw_stdout_bytes=len(stdout), raw_stdout_sha256=sha(stdout),
          raw_stdout_exact_hex=stdout.hex(), raw_stderr_bytes=len(stderr), raw_stderr_sha256=sha(stderr),
          valid_native_capture=False, error="EMPTY_NATIVE_TRANSCRIPT"),
        source_admission_performed=False, KWS_assessment_runs=0, new_audit_model_calls=0,
        B17_missing_capture_cause_inferred=False, old_replication_opened=False, promotion_allowed=False)
    (BASE / "SOURCE_READBACK.json").write_text(json.dumps(output, indent=2) + "\n")
    (BASE / "ARTIFACT_LOCKS.json").write_text(json.dumps(dict(schema="english-v6-b18-original-source-artifact-locks-v1",
        source=locks["source"], observers={p: locks[p] for p in native.PROVIDERS}, synthetic_screen=None), indent=2) + "\n")
    print(json.dumps(dict(total_native_observers=totals, failed_original=output["failed_original"],
                         source_gate=output["source_gate"], new_audit_model_calls=0), indent=2))


if __name__ == "__main__": verify()
