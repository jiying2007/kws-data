"""A bounded source-format diagnostic, permanently ineligible for KWS evaluation.

The frozen Parakeet backend sees 48 new whole originals once. An authenticated
exit-zero empty return is an invalid diagnostic observation, never a valid ASR
capture or an UNKNOWN source label. Other errors stop further native calls.
No result from this experiment can authorize a model score or promotion.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b18_source_observe as raw_observer
import english_v6_b18_asr_adapter as adapter

need, sha, record = adapter.need, adapter.sha, adapter.record
SCHEMA = "english-v6-b19-parakeet-source-feasibility-v1"
SOURCE_SCHEMA = "english-v6-b19-source-feasibility-originals-v1"
BRANCH = "refs/heads/research/en-v6-b19-source-feasibility-once-20261124"
REPO = "jiying2007/kws-data"
COUNT = 48
BASE_OBSERVER_SHA = "7681815398c4f597556e30fb5b69e46bf4e2a42faf4f8a4116ab4d2bbf80f26e"
RAW_ADAPTER_SHA = "5f1070c1443742a69953c8376c4f044471224a535bf9ee4bed89ce007f2efef4"
PARSER_SHA = "7bf7fa315b016f034cfa5e0917d57db2c9d7d25ee7791c24e84ce2d2acff8591"
CELLS = {"baseline-fast": ("baseline", .92, 2),
         "baseline-slow": ("baseline", 1.08, 2),
         "prospective-fast": ("prospective", 1.4, 2),
         "prospective-slow": ("prospective", 1.8, 2),
         "question": ("ordinary", 1.0, 0), "request": ("ordinary", 1.08, 0)}


def code_contract():
    frozen = raw_observer.source_check()
    need(frozen["observer_sha256"] == BASE_OBSERVER_SHA
         and frozen["asr_adapter_sha256"] == RAW_ADAPTER_SHA
         and frozen["parser_contract_sha256"] == PARSER_SHA,
         "original raw capture, parser, native runtime and model remain unchanged")
    return dict(schema="english-v6-b19-source-diagnostic-code-contract-v1",
                observer_sha256=sha(Path(__file__).read_bytes()),
                raw_observer_sha256=BASE_OBSERVER_SHA, asr_adapter_sha256=RAW_ADAPTER_SHA,
                parser_contract_sha256=PARSER_SHA,
                native_model=frozen["asr_models"]["parakeet"],
                native_runtime=frozen["asr_runtime_versions"]["parakeet"],
                old_file_sha256=frozen["old_file_sha256"],
                source_indices=list(range(790, 798)), original_slots=COUNT,
                maximum_native_calls=COUNT, new_kws_calls=0, source_admission_eligible=False)


def guard(registration, env=None, event=None, parent=None):
    e = os.environ if env is None else env
    need(registration["run_branch"] == BRANCH
         and re.fullmatch(r"[0-9a-f]{40}", registration["prepared_parent"]) is not None
         and all(e.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
                    "GITHUB_EVENT_NAME": "push", "GITHUB_REF": BRANCH, "GITHUB_RUN_ATTEMPT": "1"}.items())
         and registration["run_id"] == e.get("GITHUB_RUN_ID")
         and registration["run_head"] == e.get("GITHUB_SHA") and registration["run_attempt"] == 1
         and re.fullmatch(r"[1-9][0-9]*", e.get("GITHUB_RUN_ID", "")) is not None
         and re.fullmatch(r"[0-9a-f]{40}", e.get("GITHUB_SHA", "")) is not None,
         "only the registered first public B19 diagnostic event")
    if event is None:
        event = json.loads(Path(e["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], text=True).strip()
    need(parent == event.get("before") == registration["prepared_parent"]
         and event.get("after") == e["GITHUB_SHA"] and event.get("ref") == BRANCH
         and event.get("forced") is False and event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO
         and event.get("repository", {}).get("private") is False,
         "one reviewed predecessor, no replay or private diagnostic run")
    validate_registration(registration)


def validate_registration(registration):
    contract = code_contract()
    need(registration["source_schema"] == SOURCE_SCHEMA and registration["batch_schema"] == SCHEMA
         and registration["source_rows"] == registration["max_calls"] == COUNT
         and registration["diagnostic_only"] is True and registration["source_admission_eligible"] is False
         and all(registration[k] == contract[k] for k in
                 ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256"))
         and all(re.fullmatch(r"[0-9a-f]{64}", registration[k]) is not None
                 for k in ("assessment_sha256", "source_manifest_sha256")),
         "the frozen diagnostic protocol never becomes a source admission")


def validate_rows(rows):
    need(len(rows) == len({r["id"] for r in rows}) == len({r["pcm_sha256"] for r in rows}) == COUNT
         and {r["speaker_index"] for r in rows} == set(range(790, 798))
         and all(Path(r["id"]).name == r["id"] and r["id"] not in ("", ".", "..")
                 and r["role"] == "source-feasibility" and r["observed_class"] is None
                 and all(r[k] is False for k in ("training_allowed", "evaluation_allowed", "admission_eligible"))
                 for r in rows), "48 new, non-admissible, whole diagnostic originals")
    for speaker in range(790, 798):
        family = [r for r in rows if r["speaker_index"] == speaker]
        need(len(family) == 6 and {r["cell_id"] for r in family} == set(CELLS),
             "every original family retains all six preregistered cells")
        for row in family:
            arm, scale, requested = CELLS[row["cell_id"]]
            need((row["feasibility_arm"], row["length_scale"], row["requested_class"]) == (arm, scale, requested),
                 "no post-observation source-format choice or cell replacement")


def verify_empty(directory, start):
    """Prove an invalid empty response from original bytes, not an inferred blank."""
    directory = Path(directory)
    attempted = json.loads((directory / "FORWARD_STARTED.json").read_bytes())
    returned = json.loads((directory / "FORWARD_RETURNED.json").read_bytes())
    process = json.loads((directory / "RAW_PROCESS.json").read_bytes())
    need(all(attempted.get(k) == returned.get(k) == process.get(k) == v for k, v in start.items())
         and attempted["attempt"] == 1 and returned["native_return_observed"] is True
         and returned["native_exit_code"] == 0 and process["native_exit_code"] == 0
         and process["process_started"] is True and process["native_return_observed"] is True
         and process["timed_out"] is False and process["exception_type"] is None
         and process["schema"] == adapter.RAW_SCHEMA,
         "only an actual exit-zero original native return can be a diagnostic empty")
    command_raw = adapter._read_pinned(directory, process["command"])
    command = json.loads(command_raw)
    argv = command["command"]
    need(all(command.get(k) == v for k, v in start.items())
         and attempted["native_command_sha256"] == sha(command_raw)
         and command["schema"] == adapter.RAW_SCHEMA and len(argv) == 7
         and Path(argv[0]).name == "nemo-speech" and argv[1] == "transcribe"
         and Path(argv[2]).name == start["id"] + ".wav" and argv[3] == "--model"
         and Path(argv[4]).name == "parakeet-tdt-0.6b-v3.q8_0.gguf"
         and argv[5:] == ["--device", "cpu"]
         and command["command_sha256"] == sha(adapter.canonical(argv))
         and command["timeout_seconds"] == 180
         and command["stdout_transform"] is command["stderr_transform"] is None,
         "unchanged complete-input native command without text filters or hints")
    need(command["environment"] == {"PATH": "/usr/bin:/bin", "HOME": command["cwd"], "LANG": "C.UTF-8",
         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
         "NEMO_SPEECH_MODEL_DIR": str(Path(argv[4]).parent), "OMP_NUM_THREADS": "4"},
         "same pinned unprompted native process environment")
    raw = adapter._read_pinned(directory, process["stdout"])
    adapter._read_pinned(directory, process["stderr"])
    text = raw.decode("utf-8", errors="strict")
    need(len(text) <= 4096 and all(ord(c) in adapter.WHITE_SPACE for c in text)
         and not (directory / "PAYLOAD_VALID.json").exists() and not (directory / "CAPTURE.json").exists(),
         "empty is retained as invalid, never admitted or reconstructed")
    return dict(raw_stdout_sha256=sha(raw), raw_stdout_bytes=len(raw),
                raw_process_sha256=sha((directory / "RAW_PROCESS.json").read_bytes()))


def capture_original(backend, wav, directory, row, source_sha):
    start = raw_observer.binding("parakeet", row, source_sha)
    record(directory / "STARTED.json", start)
    try:
        value = backend(wav, directory, start)
        captured = dict(start, **value, status="success", input_coverage_complete=True)
        record(directory / "CAPTURE.json", captured)
        return captured
    except BaseException as exc:
        empty = type(exc) is ValueError and str(exc) == "EMPTY_NATIVE_TRANSCRIPT"
        evidence, proof_error = {}, None
        if empty:
            try:
                evidence = verify_empty(directory, start)
            except BaseException as proof_exc:
                empty = False
                proof_error = type(proof_exc).__name__ + ": " + str(proof_exc)[:500]
        failed = dict(start, status="invalid_empty" if empty else "failed",
                      error=type(exc).__name__ + ": " + str(exc)[:500], retry_count=0,
                      payload_valid=False, diagnostic_only=True, source_admission_eligible=False,
                      empty_proof_error=proof_error, **evidence)
        record(directory / "FAILED.json", failed)
        if not empty:
            if proof_error is not None:
                raise ValueError("UNPROVEN_EMPTY_NATIVE_TRANSCRIPT: " + proof_error) from exc
            raise
        return failed


def feasibility(rows, captures):
    """Fixed feasibility rule only; no model gate, label or release authority."""
    validate_rows(rows)
    need(set(captures) == {r["id"] for r in rows}, "all planned observations accounted for")
    groups = {}
    for arm in ("baseline", "prospective", "ordinary"):
        originals = [r for r in rows if r["feasibility_arm"] == arm]
        need(len(originals) == 16, "sixteen paired original slots per diagnostic arm")
        status = [captures[r["id"]]["status"] for r in originals]
        groups[arm] = dict(originals=16, valid_nonempty=status.count("success"),
                           invalid_empty=status.count("invalid_empty"),
                           other_or_unstarted=sum(s not in ("success", "invalid_empty") for s in status),
                           whole_audio_bounds_ok=all(r["basic_audio_ok"]
                               and 3200 <= r["samples"] <= 240000
                               and (r["requested_class"] != 2 or 6400 <= r["samples"] <= 28672) for r in originals))
    complete = all(v["other_or_unstarted"] == 0 for v in groups.values())
    feasible = (complete and all(v["whole_audio_bounds_ok"] for v in groups.values())
                and all(groups[k]["valid_nonempty"] == 16 for k in ("prospective", "ordinary")))
    return dict(arms=groups, diagnostic_complete=complete,
                prospective_pair_feasible=feasible,
                result="SOURCE_FORMAT_FEASIBLE_REQUIRES_NEW_PREREGISTRATION" if feasible else "STOP_SOURCE_FORMAT_NOT_QUALIFIED",
                source_admission_eligible=False, source_gate_passed=False, KWS_evaluation_allowed=False,
                training_allowed=False, promotion_allowed=False, replication_opened=False,
                causal_fix_proven=False, old_B17_B18_failures_preserved=True)


def observe(source, out, *, read_rows, registration):
    guard(registration)
    source, out = Path(source), Path(out)
    import english_v6_b19_source as source_owner
    need(registration == source_owner.registration(source),
         "actual frozen source assessment is authenticated before loading the native model")
    rows = read_rows(source)
    validate_rows(rows)
    source_sha = sha((source / "MANIFEST.json").read_bytes())
    need(source_sha == registration["source_manifest_sha256"] and out.is_absolute()
         and not out.exists() and out.parent.is_dir(), "fresh diagnostic output and exact original source")
    out.mkdir(mode=0o700)
    contract = code_contract()
    record(out / "STARTED.json", dict(schema=SCHEMA, registration=registration,
           source_manifest_sha256=source_sha, diagnostic_only=True, native_call_budget=COUNT,
           source_admission_eligible=False, code_contract=contract))
    captures, fatal = {}, None
    try:
        assets = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "b19-parakeet-native-assets"
        backend, model = adapter._load("parakeet", assets, out)
        record(out / "MODEL.json", model)
        import english_hifi_micro as pcm_tools
        for row in rows:
            directory = out / "rows" / row["id"]
            directory.mkdir(parents=True, exist_ok=False)
            wav = source / "wav" / (row["id"] + ".wav")
            facts = pcm_tools.inspect_wav(wav)
            need(all(facts[k] == row[k] for k in ("wav_sha256", "pcm_sha256", "samples")),
                 "the same complete original waveform immediately before one native call")
            captures[row["id"]] = capture_original(backend, wav, directory, row, source_sha)
    except BaseException as exc:
        fatal = type(exc).__name__ + ": " + str(exc)[:500]
    ledger = raw_observer.row_ledger(out, rows)
    counts = raw_observer.ledger_counts(ledger)
    record(out / "LEDGER.json", dict(schema=SCHEMA, rows=ledger, counts=counts))
    rows_result = {}
    for row in rows:
        failed = out / "rows" / row["id"] / "FAILED.json"
        rows_result[row["id"]] = captures.get(row["id"],
            json.loads(failed.read_bytes()) if failed.is_file() else dict(id=row["id"], status="not_started"))
    verdict = feasibility(rows, rows_result)
    result = dict(schema=SCHEMA, source_manifest_sha256=source_sha, registration=registration,
                  actual_native_attempts=counts["forward_attempted"], native_returns=counts["native_returned"],
                  valid_nonempty_payloads=counts["payload_valid"], invalid_empty_payloads=sum(r["status"] == "invalid_empty" for r in rows_result.values()),
                  native_call_budget=COUNT, new_TTS_calls=0, new_KWS_forwards=0, optimizer_updates=0,
                  retry_count=0, threshold_search=False, fatal_error=fatal, code_contract=contract,
                  rows=[rows_result[r["id"]] for r in rows], **verdict)
    record(out / "RESULT.json", result)
    record(out / "FILES.json", dict(schema=SCHEMA, self_excluded="FILES.json", files=raw_observer.file_closure(out)))
    need(fatal is None and counts["forward_attempted"] == counts["native_returned"] == COUNT,
         "incomplete native diagnostic stops; original partial evidence retained")
    return result


def readback(source, evidence, read_rows, registration):
    source, evidence = Path(source), Path(evidence)
    import english_v6_b19_source as source_owner
    need(registration == source_owner.registration(source),
         "actual frozen source registration cannot be replaced by matching self-reported metadata")
    validate_registration(registration)
    rows = read_rows(source)
    validate_rows(rows)
    source_sha = sha((source / "MANIFEST.json").read_bytes())
    contract = code_contract()
    start = json.loads((evidence / "STARTED.json").read_bytes())
    need(start == dict(schema=SCHEMA, registration=registration, source_manifest_sha256=source_sha,
         diagnostic_only=True, native_call_budget=COUNT, source_admission_eligible=False, code_contract=contract),
         "actual source generation and frozen diagnostic code registration")
    need(json.loads((evidence / "FILES.json").read_bytes()) == dict(schema=SCHEMA,
         self_excluded="FILES.json", files=raw_observer.file_closure(evidence)), "all original raw diagnostic bytes retained")
    model = json.loads((evidence / "MODEL.json").read_bytes())
    need(model["observer"] == contract["native_model"] and model["runtime_versions"] == contract["native_runtime"]
         and model["adapter_sha256"] == RAW_ADAPTER_SHA and model["parser_contract_sha256"] == PARSER_SHA
         and model["raw_before_validation"] is True and model["decode_config_unchanged"] is True,
         "the same unprompted native ASR model and runtime")
    result = json.loads((evidence / "RESULT.json").read_bytes())
    captures = {r["id"]: r for r in result["rows"]}
    need(len(captures) == len(result["rows"]) == COUNT and result["fatal_error"] is None,
         "complete original diagnostic rows without repeated identity")
    for row in rows:
        directory = evidence / "rows" / row["id"]
        binding = raw_observer.binding("parakeet", row, source_sha)
        need(json.loads((directory / "STARTED.json").read_bytes()) == binding, "original diagnostic waveform identity")
        capture = captures[row["id"]]
        if capture["status"] == "success":
            need(capture == json.loads((directory / "CAPTURE.json").read_bytes())
                 and all(capture.get(k) == v for k, v in binding.items()), "exact saved native capture")
            adapter.validate_capture("parakeet", directory, binding, capture)
        else:
            proof = verify_empty(directory, binding)
            need(capture["status"] == "invalid_empty" and capture == json.loads((directory / "FAILED.json").read_bytes())
                 and all(capture.get(k) == v for k, v in {**binding, **proof}.items())
                 and capture["payload_valid"] is False and capture["source_admission_eligible"] is False
                 and capture["empty_proof_error"] is None,
                 "an authenticated empty remains invalid diagnostic evidence")
    ledger = raw_observer.row_ledger(evidence, rows)
    counts = raw_observer.ledger_counts(ledger)
    need(json.loads((evidence / "LEDGER.json").read_bytes()) == dict(schema=SCHEMA, rows=ledger, counts=counts)
         and counts["forward_attempted"] == counts["native_returned"] == counts["raw_payload_saved"] == COUNT
         and result["actual_native_attempts"] == result["native_returns"] == COUNT
         and result["valid_nonempty_payloads"] == counts["payload_valid"] == counts["capture_saved"]
         and result["invalid_empty_payloads"] == COUNT - counts["payload_valid"] == counts["failed"]
         and result["new_TTS_calls"] == result["new_KWS_forwards"] == result["optimizer_updates"] == result["retry_count"] == 0
         and result["schema"] == SCHEMA and result["native_call_budget"] == COUNT
         and result["threshold_search"] is False
         and result["registration"] == registration and result["code_contract"] == contract
         and result["source_manifest_sha256"] == source_sha,
         "truthful distinct attempts, returns, valid payloads and invalid empty outcomes")
    verdict = feasibility(rows, captures)
    need(all(result[k] == v for k, v in verdict.items()), "unchanged prospective source-format feasibility rule")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-contract", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(code_contract(), indent=2, sort_keys=True))
