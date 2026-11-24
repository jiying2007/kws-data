"""Future B21 strict five-observer wrapper; closed until complete new preregistration.

The unchanged B18 native backends and raw validators are reused. Every missing,
empty or invalid native payload still stops the whole panel. The B19 diagnostic
exception is deliberately absent from this full source protocol.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b18_source_observe as native
import english_v6_b21_source_ancestry as lineage

raw_asr = native.raw_asr
need, sha, record = native.need, native.sha, native.record
PROVIDERS, POLICY_FUNCTIONS = native.PROVIDERS, native.POLICY_FUNCTIONS
binding, row_ledger, ledger_counts = native.binding, native.row_ledger, native.ledger_counts
complete_ledger, file_closure = native.complete_ledger, native.file_closure
_load_phone, _verify_phone = native._load_phone, native._verify_phone
SOURCE_SCHEMA = "english-v6-b21-fresh-combined-source-v1"
BATCH = "english-v6-b21-120-original-strict-source-screen-v1"
MAX_COUNT = 120
FROZEN_NATIVE_SHA = "7681815398c4f597556e30fb5b69e46bf4e2a42faf4f8a4116ab4d2bbf80f26e"


def source_check(repository=None):
    prior = native.source_check(repository)
    need(prior["observer_sha256"] == FROZEN_NATIVE_SHA
         and prior["asr_adapter_sha256"] == "5f1070c1443742a69953c8376c4f044471224a535bf9ee4bed89ce007f2efef4"
         and prior["parser_contract_sha256"] == "7bf7fa315b016f034cfa5e0917d57db2c9d7d25ee7791c24e84ce2d2acff8591",
         "unchanged native backends and strict raw parser")
    return dict(schema="english-v6-b21-five-observer-contract-v1",
        **{k: prior[k] for k in ("asr_adapter_sha256", "parser_contract_sha256", "old_file_sha256",
             "source_admission_function_sha256", "asr_models", "asr_runtime_versions", "providers")},
        observer_sha256=sha(Path(__file__).read_bytes()), frozen_native_observer_sha256=FROZEN_NATIVE_SHA,
        maximum_source_rows=120, maximum_calls_per_provider=120, maximum_native_calls=600,
        source_indices=list(range(846, 870)), fixed_positive_rate_pair=[1.4, 1.8],
        strict_valid_five_per_source=True, diagnostic_empty_exception=False,
        no_retry=True, whole_originals_only=True, new_model_calls=0, new_network_calls=0,
        training_allowed=False, promotion_allowed=False)


def guard(registration, *, env=None, event=None, parent=None):
    lineage.read_plan()
    need(type(registration) is dict, "actual authenticated B21 registration required")
    identity = lineage.execution_guard(registration["prepared_parent"], env, event, parent)
    contract = source_check()
    need(all(registration[k] == value for k, value in identity.items())
         and all(registration[k] == contract[k] for k in
                 ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256"))
         and registration["assessment_sha256"] == lineage.ASSESSMENT_SHA
         and registration["source_schema"] == SOURCE_SCHEMA and registration["batch_schema"] == BATCH
         and registration["source_rows"] == registration["max_calls"]
         and type(registration["source_rows"]) is int and 96 <= registration["source_rows"] <= MAX_COUNT,
         "actual first source run and every retained original in the fixed 120 slots")
    return {"run": identity["run_id"], "sha": identity["run_head"]}


def validate_rows(rows, source_sha, registration):
    count = registration["source_rows"]
    need(source_sha == registration["source_manifest_sha256"] and 96 <= count <= MAX_COUNT
         and len(rows) == len({r["id"] for r in rows}) == len({r["pcm_sha256"] for r in rows}) == count
         and all(Path(r["id"]).name == r["id"] and r["id"].startswith("b21-primary-")
                 and r["role"] == "development" and r["training_allowed"] is False
                 and r["observed_class"] is None for r in rows), "every original eligible source exactly once")
    synthetic = [r for r in rows if r["cohort"] == "synthetic"]
    natural = [r for r in rows if r["cohort"] == "natural"]
    need(len(synthetic) == 96 and len(natural) == count - 96
         and {r["speaker_index"] for r in synthetic} == set(range(846, 870)), "all96 synthetic and only retained natural originals")
    for index in range(846, 870):
        family = [r for r in synthetic if r["speaker_index"] == index]
        need(len(family) == 4 and {r["cell_id"]: r["length_scale"] for r in family}
             == {"mero-fast": 1.4, "mero-slow": 1.8, "question": 1.0, "request": 1.08}
             and all(r["requested_class"] == (2 if r["cell_id"].startswith("mero-") else 0) for r in family),
             "unchanged positive rate pair and ordinary sentences")
    need(all(r["requested_class"] == 0 and r["whole_original"] is True for r in natural), "whole unmodified natural negatives")


def expected_registration(source, supplied=None):
    lineage.read_plan()
    import english_v6_b21_source as owner
    expected = owner.registration(Path(source))
    need(supplied is None or supplied == expected, "source-authorized registration, never provider self-report")
    return expected


def require_original_rows(source, rows):
    import english_v6_b21_source as owner
    need(rows == owner.read_source(Path(source)),
         "every supplied row equals the authenticated full original source, in registered order")


def observe(provider, source, out, *, read_rows, schema, registration):
    lineage.read_plan()
    need(provider in PROVIDERS and schema == BATCH, "ONLY_REGISTERED_B21_FIVE_OBSERVER_SOURCE")
    run = guard(registration)
    source, root = Path(source), Path(out)
    rows = read_rows(source)
    count = len(rows)
    source_sha = sha((source / "MANIFEST.json").read_bytes())
    registration = expected_registration(source, registration)
    require_original_rows(source, rows)
    validate_rows(rows, source_sha, registration)
    need(root.is_absolute() and not root.exists() and root.parent.is_dir()
         and not any(p.is_symlink() for p in (root, *root.parents)), "FRESH_OBSERVER_OUTPUT_ONLY")
    root.mkdir(mode=0o700)
    record(root / "STARTED.json", {"run": run, "batch": schema, "provider": provider,
            "input_manifest": source_sha, "registration": registration,
            "parser_contract_sha256": raw_asr.PARSER_CONTRACT_SHA256,
            "old_admission_function_sha256": POLICY_FUNCTIONS,
            "denominator": count, "whole_panel_stop_on_failure": True})
    captures, initialization_error, row_error = {}, None, None
    try:
        import english_hifi_micro as c
        record(root / "MODEL_LOAD_STARTED.json", {"provider": provider, "actual_forward_attempts": 0})
        if provider in ("xlsr", "lv60"):
            backend, lock = _load_phone(provider, root)
        else:
            assets = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / ("b21-asr-" + provider)
            backend, lock = raw_asr._load(provider, assets, root)
        record(root / "MODEL.json", lock)
    except BaseException as exc:
        initialization_error = type(exc).__name__ + ": " + str(exc)[:500]
    if initialization_error is None:
        for row in rows:
            directory = root / "rows" / row["id"]
            directory.mkdir(parents=True, exist_ok=False)
            start = binding(provider, row, source_sha)
            try:
                record(directory / "STARTED.json", start)
                wav = source / "wav" / (row["id"] + ".wav")
                measured = c.inspect_wav(wav)
                need(all(measured[k] == row[k] for k in ("wav_sha256", "pcm_sha256", "samples")),
                     "UNCHANGED_FULL_ORIGINAL_BEFORE_NATIVE_CALL")
                captured = backend(wav, directory, start)
                result = {**start, **captured, "status": "success", "input_coverage_complete": True}
                record(directory / "CAPTURE.json", result)
                captures[row["id"]] = result
            except BaseException as exc:
                row_error = type(exc).__name__ + ": " + str(exc)[:500]
                failed = {**start, "status": "failed", "error": row_error,
                          "retry_count": 0, "whole_panel_stop": True}
                record(directory / "FAILED.json", failed)
                captures[row["id"]] = failed
                break  # A missing/invalid return cannot become an abstention or a retry.
    ledger = row_ledger(root, rows)
    counts = ledger_counts(ledger)
    ledger_doc = {"schema": "english-v6-b18-native-attempt-return-valid-ledger-v1", "provider": provider,
                  "source_manifest_sha256": source_sha, "rows": ledger, "counts": counts,
                  "attempted_ids": [r["id"] for r in ledger if r["forward_attempted"]],
                  "returned_ids": [r["id"] for r in ledger if r["native_returned"]],
                  "payload_valid_ids": [r["id"] for r in ledger if r["payload_valid"]]}
    record(root / "LEDGER.json", ledger_doc)
    complete = complete_ledger(ledger, count) and initialization_error is None and row_error is None
    summary = {"schema": schema, "provider": provider, "run": run,
        "source_manifest_sha256": source_sha, "denominator": count,
        "attempted": counts["forward_attempted"], "actual_native_forward_attempts": counts["forward_attempted"],
        "native_forward_returns": counts["native_returned"], "valid_native_payloads": counts["payload_valid"],
        "complete": counts["capture_saved"], "missing_or_failed": count - counts["capture_saved"],
        "status": "COMPLETE" if complete else "STOP_OBSERVER_FAILURE",
        "whole_panel_stop": not complete, "initialization_error": initialization_error, "row_error": row_error,
        "rows": [captures.get(r["id"], {**binding(provider, r, source_sha), "status": "not_started"}) for r in rows],
        "new_tts_calls": 0, "new_asr_calls": counts["forward_attempted"] if provider in raw_asr.TEXT_PROVIDERS else 0,
        "new_phone_calls": counts["forward_attempted"] if provider in ("xlsr", "lv60") else 0,
        "target_conditioned": False, "human_gold": False, "training_allowed": False,
        "retry_count": 0, "kws_forward_calls": 0, "promotion_allowed": False,
        "parser_contract_sha256": raw_asr.PARSER_CONTRACT_SHA256,
        "observer_code_sha256": sha(Path(__file__).read_bytes()),
        "asr_adapter_sha256": sha((HERE / "english_v6_b18_asr_adapter.py").read_bytes())}
    record(root / "RESULT.json", summary)
    record(root / "FILES.json", {"schema": "english-v6-b18-observer-file-closure-v1",
                                 "self_excluded": "FILES.json", "files": file_closure(root)})
    need(complete, "WHOLE_PANEL_STOP_NATIVE_RETURN_OR_PAYLOAD_FAILURE_NO_RETRY")
    return summary

def load_observations(source, evidence, rows, schema, *, registration=None):
    """Mandatory new evidence gate, followed by the unchanged five-model loader."""
    lineage.read_plan()
    contract = source_check()
    count = len(rows)
    need(schema == BATCH and 96 <= count <= MAX_COUNT and len({r["id"] for r in rows}) == count,
         "STRICT_B21_FIXED_ORIGINAL_PANEL")
    source, evidence = Path(source), Path(evidence)
    registration = expected_registration(source, registration)
    source_sha = sha((source / "MANIFEST.json").read_bytes())
    require_original_rows(source, rows)
    validate_rows(rows, source_sha, registration)
    registrations = []
    for provider in PROVIDERS:
        root = evidence / provider
        closure = json.loads((root / "FILES.json").read_bytes())
        need(closure == {"schema": "english-v6-b18-observer-file-closure-v1", "self_excluded": "FILES.json",
                         "files": file_closure(root)}, "FULL_RAW_OBSERVER_FILE_CLOSURE")
        report = json.loads((root / "RESULT.json").read_bytes())
        started = json.loads((root / "STARTED.json").read_bytes())
        need(started["input_manifest"] == source_sha and started["provider"] == provider and started["batch"] == schema
             and started["old_admission_function_sha256"] == POLICY_FUNCTIONS
             and started["parser_contract_sha256"] == raw_asr.PARSER_CONTRACT_SHA256
             and started["denominator"] == count and started["whole_panel_stop_on_failure"] is True,
             "FROZEN_SOURCE_ADMISSION_AND_NEW_PARSER_START")
        reg = started["registration"]
        need(reg["source_schema"] == SOURCE_SCHEMA and reg["batch_schema"] == BATCH
             and reg["source_manifest_sha256"] == source_sha and reg["source_rows"] == reg["max_calls"] == count
             and all(reg[k] == contract[k] for k in ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256"))
             and reg == registration, "PROSPECTIVE_REGISTRATION_BINDING")
        registrations.append(reg)
        ledger = row_ledger(root, rows)
        counts = ledger_counts(ledger)
        stored_ledger = json.loads((root / "LEDGER.json").read_bytes())
        need(stored_ledger == {"schema": "english-v6-b18-native-attempt-return-valid-ledger-v1", "provider": provider,
              "source_manifest_sha256": source_sha, "rows": ledger, "counts": counts,
              "attempted_ids": [r["id"] for r in ledger if r["forward_attempted"]],
              "returned_ids": [r["id"] for r in ledger if r["native_returned"]],
              "payload_valid_ids": [r["id"] for r in ledger if r["payload_valid"]]}
              and complete_ledger(ledger, count), "ALL_DISTINCT_ATTEMPTS_RETURNS_VALID_PAYLOADS_AND_CAPTURES")
        need(report["schema"] == schema and report["provider"] == provider
             and report["source_manifest_sha256"] == source_sha and report["denominator"] == count
             and report["parser_contract_sha256"] == contract["parser_contract_sha256"]
             and report["target_conditioned"] is False
             and report["status"] == "COMPLETE" and report["whole_panel_stop"] is False
             and report["run"] == started["run"] == {"run": reg["run_id"], "sha": reg["run_head"]}
             and report["initialization_error"] is report["row_error"] is None
             and report["actual_native_forward_attempts"] == report["native_forward_returns"]
             == report["valid_native_payloads"] == report["attempted"] == report["complete"] == count
             and report["missing_or_failed"] == report["retry_count"] == report["new_tts_calls"] == report["kws_forward_calls"] == 0
             and report["new_asr_calls"] == (count if provider in raw_asr.TEXT_PROVIDERS else 0)
             and report["new_phone_calls"] == (count if provider in ("xlsr", "lv60") else 0)
             and report["promotion_allowed"] is report["training_allowed"] is report["human_gold"] is False
             and report["observer_code_sha256"] == contract["observer_sha256"]
             and report["asr_adapter_sha256"] == contract["asr_adapter_sha256"],
             "NO_FAILED_EMPTY_OR_LOST_RETURN_CAN_COMPLETE_PANEL")
        model = json.loads((root / "MODEL.json").read_bytes())
        by_id = {r["id"]: r for r in report["rows"]}
        need(len(by_id) == len(report["rows"]) == count, "ALL_UNIQUE_CAPTURE_RECORDS")
        for row in rows:
            directory = root / "rows" / row["id"]
            start = binding(provider, row, source_sha)
            capture = by_id[row["id"]]
            need(json.loads((directory / "STARTED.json").read_bytes()) == start
                 and capture == json.loads((directory / "CAPTURE.json").read_bytes())
                 and capture["status"] == "success" and capture["input_coverage_complete"] is True
                 and all(capture.get(k) == v for k, v in start.items()), "COMPLETE_ORIGINAL_CAPTURE_IDENTITY")
            if provider in raw_asr.TEXT_PROVIDERS:
                need(model["adapter_sha256"] == contract["asr_adapter_sha256"]
                     and model["parser_contract_sha256"] == raw_asr.PARSER_CONTRACT_SHA256
                     and model["runtime_versions"] == contract["asr_runtime_versions"][provider]
                     and model["observer"] == contract["asr_models"][provider]
                     and model["raw_before_validation"] is True and model["decode_config_unchanged"] is True,
                     "RAW_PRESERVING_FIXED_ASR_ADAPTER")
                raw_asr.validate_capture(provider, directory, start, capture)
            else:
                _verify_phone(root, directory, start, capture, model)
    need(all(r == registrations[0] for r in registrations), "SAME_FROZEN_SOURCE_REGISTRATION_ALL_FIVE")
    import english_v6_primary_source_qa as qa
    texts, phones, coverage = qa.load_observations(source, evidence, rows, schema)
    need(coverage == dict.fromkeys(PROVIDERS, count), "ALL_FIVE_VALID_NATIVE_PAYLOADS_REQUIRED")
    return texts, phones, coverage

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(source_check(), indent=2, sort_keys=True))
