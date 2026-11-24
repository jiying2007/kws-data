"""Authenticate original B23 quality evidence without any model/runtime call.

Both public entry points require a complete original lock. Verification uses an
isolated stdlib process, the original Git-bound producer, every original source
and native observation, every complete FP32 return, and the unchanged B14 rule.
Only an exactly reproduced PASS can return a report. Fixtures are not permits.
"""
from __future__ import annotations
import argparse
import ast
import builtins
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from unittest import mock

REPO = "jiying2007/kws-data"
PRIMARY_CODE_SHA = "6635730ab7a02351a8b60e97f8f8c40ecec8631e95ca538c4bde5b2d994b4406"
EVALUATOR_STEMS = {"primary": "english_v6_b23_primary", "replication": "english_v6_b23_replication"}
QUALITY_CODE_SHA = "6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023"
QUALITY_ASSESSMENT_SHA = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"
ASSESSMENT_SHA = "a5923fa64969ffa5d550fcdd320fd1c8d8aa68a9430f6de03fe5eab41b24e8ae"
DESIGN_SHA = "8485e152f0ba93e22c510731dc4acd7823b36df033856b4145aa971aa96ebe56"
SCHEMAS = {panel: "english-v6-b23-fresh-" + panel + "-four-frozen-models-v1" for panel in ("primary", "replication")}
ARMS = ("B0", "B9", "control", "challenger")
GAINS = ("1.0", "0.75", "0.5")
STATE_FIELDS = {"array", "sha256", "frames", "tokens", "event_frames", "event_samples", "events"}
IDENTITY_FIELDS = ("run_id", "run_head", "run_attempt", "prepared_parent")
FALSE_FIELDS = ("training_allowed", "threshold_search", "promotion_allowed", "product_released",
                "cal_locked_access", "old_replication_opened")
LOCK_FIELDS = {"schema", "panel", "result", "producer", "consumed", "evaluator_sha256", "inputs_sha256",
               "producer_manifest_sha256", "original_result_sha256"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_hash(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def equal(left, right):
    return canonical(left) == canonical(right)


def document(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "no duplicate JSON field")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON number " + value)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    need(type(value) is dict, "original JSON object")
    return value


def archive_limit(panel="primary", role=None):
    need(panel in SCHEMAS and role in {None, "result", "producer", "consumed", "B0", "B9", "pair", "source", "screen",
         "train_producer", "source_producer", *("native_" + provider for provider in ("qwen", "whisper", "parakeet", "xlsr", "lv60"))},
         "explicit existing panel and artifact role")
    return 512_000_000 if panel == "replication" and role in {"source", "producer"} else 128_000_000


def valid_artifact(pin, *, panel="primary", role=None):
    need(type(pin) is dict and set(pin) == {"id", "run_id", "run_head", "bytes", "sha256"}
         and all(type(pin[key]) is int and pin[key] > 0 for key in ("id", "run_id", "bytes"))
         and pin["bytes"] <= archive_limit(panel, role) and valid_hash(pin["run_head"], 40) and valid_hash(pin["sha256"]),
         "complete original artifact identity; unset locks remain closed")


def validate_lock(lock):
    need(type(lock) is dict and set(lock) == LOCK_FIELDS
         and lock["schema"] == "english-v6-b23-original-quality-lock-v1" and lock["panel"] in SCHEMAS,
         "explicit original primary/replication lock")
    for key in ("result", "producer", "consumed"):
        valid_artifact(lock[key], panel=lock['panel'], role=key)
    need(len({lock[key]["id"] for key in ("result", "producer", "consumed")}) == 3
         and all(lock[key][field] == lock["result"][field] for key in ("producer", "consumed") for field in ("run_id", "run_head")),
         "three separate originals from the same exact quality execution")
    for key in ("evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256"):
        need(valid_hash(lock[key]), "actual immutable quality digest " + key)
    if lock["panel"] == "primary":
        need(lock["evaluator_sha256"] == PRIMARY_CODE_SHA, "unchanged preregistered primary evaluator")


def opened(raw, maximum=128_000_000):
    need(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded complete original archive")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    try:
        entries = archive.infolist()
        need(len(entries) <= 8192 and len(entries) == len({entry.filename for entry in entries})
             and sum(entry.file_size for entry in entries) <= maximum, "unique bounded archive members")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            need(path.parts and not path.is_absolute() and ".." not in path.parts
                 and "\\" not in entry.filename and not stat.S_ISLNK(entry.external_attr >> 16)
                 and not (entry.flag_bits & 1), "safe unencrypted original member")
        need(archive.testzip() is None, "complete original archive CRC")
        return archive
    except BaseException:
        archive.close()
        raise


def original(path, pin, *, panel="primary", role=None):
    valid_artifact(pin, panel=panel, role=role)
    raw = Path(path).read_bytes()
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "exact original archive bytes " + str(path))
    return opened(raw, maximum=archive_limit(panel, role))


def identity(value):
    need(type(value) is dict and set(value) == set(IDENTITY_FIELDS)
         and type(value["run_id"]) is str and re.fullmatch(r"[1-9][0-9]*", value["run_id"])
         and type(value["run_attempt"]) is int and value["run_attempt"] == 1
         and valid_hash(value["run_head"], 40) and valid_hash(value["prepared_parent"], 40)
         and value["run_head"] != value["prepared_parent"], "actual first-run/head/unique-parent identity")


def read_npy(raw, expected_shape):
    """Read every finite FP32 NPY frame without NumPy, Torch or pickle."""
    need(type(raw) is bytes and raw[:6] == b"\x93NUMPY" and len(raw) >= 10, "real complete NPY member")
    version = tuple(raw[6:8])
    need(version in ((1, 0), (2, 0)), "supported immutable NPY header")
    width = 2 if version == (1, 0) else 4
    size = int.from_bytes(raw[8:8 + width], "little")
    need(0 < size <= 8192 and len(raw) >= 8 + width + size, "complete bounded NPY header")
    header = ast.literal_eval(raw[8 + width:8 + width + size].decode("latin1").strip())
    shape = header.get("shape") if type(header) is dict else None
    need(type(shape) is tuple and all(type(value) is int for value in shape)
         and shape == expected_shape and expected_shape[-1] == 4
         and len(expected_shape) in (2, 3) and 1 <= expected_shape[-2] <= 1500
         and (len(expected_shape) == 2 or expected_shape[0] == 1)
         and set(header) == {"descr", "fortran_order", "shape"}
         and header["descr"] == "<f4" and header["fortran_order"] is False,
         "exact complete float32 NPY shape before any batch selection")
    body = raw[8 + width + size:]
    need(len(body) == expected_shape[-2] * 16, "every returned NPY frame byte is present")
    labels = []
    for frame in struct.iter_unpack("<4f", body):
        need(all(math.isfinite(value) for value in frame), "finite original frame output")
        labels.append(max(range(4), key=frame.__getitem__))
    return body, labels


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def offline_only():
    stack = ExitStack()
    old_import, old_popen = builtins.__import__, subprocess.Popen
    forbidden = {"numpy", "torch", "torchaudio", "onnxruntime", "soundfile", "transformers", "piper", "requests", "httpx"}
    def import_guard(name, *args, **kwargs):
        need(name.split(".")[0] not in forbidden, "no acoustic/network runtime during original quality verification")
        return old_import(name, *args, **kwargs)
    def popen_guard(command, *args, **kwargs):
        need(type(command) is list and len(command) == 8 and command[:4] == [sys.executable, "-I", "-S", "-B"]
             and command[5] == "--verify-worker" and sha(Path(command[4]).read_bytes()) == sha(Path(__file__).read_bytes()),
             "only an isolated recursive saved-quality verifier may start during readback")
        return old_popen(command, *args, **kwargs)
    stack.enter_context(mock.patch("builtins.__import__", side_effect=import_guard))
    stack.enter_context(mock.patch("subprocess.Popen", side_effect=popen_guard))
    stack.enter_context(mock.patch("socket.socket", side_effect=ValueError("no network during original quality verification")))
    return stack


def source_admission(primary, inputs, spec, identity, workflow, fetched, panel, primary_quality=None):
    """Repeat original pure admission; never call/emulate the execution guard."""
    original_zip = primary.original_zip
    validate_pair_receipt = primary.validate_pair_receipt
    exact_admitted_join, person = primary.exact_admitted_join, primary.person
    reject_overlap, source_counts = primary.reject_overlap, primary.source_counts
    score_identity = primary.score_identity
    TRAIN_BRANCH, B0_PT, B0_NPZ, B9_PT = primary.TRAIN_BRANCH, primary.B0_PT, primary.B0_NPZ, primary.B9_PT
    SOURCE_BRANCH, SOURCE_WORKFLOW = primary.SOURCE_BRANCH, primary.SOURCE_WORKFLOW
    SCREEN_SCHEMA, PROVIDERS, GATE = primary.SCREEN_SCHEMA, primary.PROVIDERS, primary.GATE
    MAX_FORWARDS, STEMS, SCHEMA, SPEC = primary.MAX_FORWARDS, primary.STEMS, primary.SCHEMA, primary.SPEC
    pins = primary.archive_pins(spec)
    common, decision, source, plan = primary.helpers(spec)
    with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
        archives = {key: stack.enter_context(original_zip(inputs / (key + ".zip"), pin, role=key)
                    if panel == 'replication' else original_zip(inputs / (key + ".zip"), pin)) for key, pin in pins.items()}
        pair = archives["pair"]
        producer = json.loads(primary.PRODUCER.read_bytes())
        primary.verify_upstream_archives(archives, spec, fetched, producer)
        train = primary.validate_training_archive(pair, spec, fetched["source_metadata"]["pair"])
        need(sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.pt")) == B0_PT
             and sha(archives["B0"].read("models/mero-fsmn-ctc-expanded256.npz")) == B0_NPZ
             and sha(archives["B9"].read(common.b9.MODELS["B9_control"]["path"] + ".pt")) == B9_PT,
             "original B0/B9 baseline identities before Torch")
        directory = Path(temporary) / "source"
        directory.mkdir()
        archives["source"].extractall(directory)
        owner = source.evaluation_preflight(directory)
        rows, originals = owner["rows"], owner["all_original_rows"]
        execution, source_meta = owner["source_execution"], fetched["source_metadata"]["source"]
        need(owner["assessment_sha256"] == spec["assessment_sha256"]
             and str(execution["run_id"]) == str(spec["inputs"]["source"]["run_id"])
             and execution["run_head"] == spec["inputs"]["source"]["run_head"]
             and execution["run_attempt"] == 1 and execution["run_branch"] == "refs/heads/" + SOURCE_BRANCH
             and source_meta["run"]["head_branch"] == SOURCE_BRANCH and source_meta["run"]["path"] == SOURCE_WORKFLOW
             and source_meta["run"]["event"] == "push" and len(source_meta["commit"]["parents"]) == 1
             and source_meta["commit"]["parents"][0]["sha"] == execution["prepared_parent"]
             and [entry["filename"] for entry in source_meta["commit"]["files"]] == [SOURCE_WORKFLOW],
             "complete real first public B23 source manifest and workflow-only opening")
        providers = Path(temporary) / "providers"
        for provider in PROVIDERS:
            target = providers / provider
            target.mkdir(parents=True)
            archives["native_" + provider].extractall(target)
        screen = json.loads(archives["screen"].read("SCREEN.json"))
        need(screen["schema"] == SCREEN_SCHEMA and screen["source_execution"] == execution
             and screen["source_manifest_sha256"] == sha(archives["source"].read("MANIFEST.json"))
             and screen["coverage"] == dict.fromkeys(PROVIDERS, len(rows))
             and screen["selected_original_slots"] == 120, "all original native observations and unchanged SCREEN source identity")
        source.verify_screen_contract(directory, screen, providers)
        selected = exact_admitted_join(rows, screen["rows"])
        reserve_people = set()
        if panel == "primary":
            reserve_people = {person(row) for row in source.source_plan("replication")["rows"]}
            reserve_people.update(str(int(value)) for value in plan["replication"]["natural"]["speakers"])
        else:
            need(primary_quality is not None and equal(owner["primary_quality"], primary_quality),
                 "source and scorer reauthenticate the exact same original primary PASS")
            prior = primary_quality["source_inventory"]
            need(not set(prior["pcm_sha256"]) & {row["pcm_sha256"] for row in originals}
                 and not set(prior["source_group_id"]) & {row["source_group_id"] for row in originals}
                 and not set(prior["canonical_people"]) & {person(row) for row in originals},
                 "all 120 replica originals independent of every primary original including UNKNOWN")
        reject_overlap(originals, train["base_train"] + train["natural_train"], owner["history"], reserve_people)
        counts = source_counts(selected)
        need(counts == screen["source_coverage"] and screen["frozen_minima"] == GATE
             and all(counts[key] >= minimum for key, minimum in GATE.items())
             and screen["source_gate"] == "SOURCE_GATE_PASS" and not screen["source_shortfall"],
             "source coverage gate before any KWS work")
        need(12 * len(selected) <= MAX_FORWARDS, "complete source population fits preregistered1440 forward ceiling")
    need("torch" not in sys.modules and "numpy" not in sys.modules,
         "all source and original TRAIN admission completes before model/array runtimes are imported")
    chosen = [(row, row["cohort"]) for row in selected]
    admission = dict(schema=SCHEMA, identity=identity, assessment_sha256=spec["assessment_sha256"],
        training_design_sha256=spec["training_design_sha256"], registered_inputs_sha256=sha(SPEC.read_bytes()),
        producer_manifest_sha256=spec["producer_manifest_sha256"], workflow_sha256=sha(workflow),
        source_gate="SOURCE_GATE_PASS", source_shortfall={}, source_coverage=counts,
        original_source_rows=120, observer_eligible_rows=len(rows),
        admitted_rows=[score_identity(row, cohort) for row, cohort in chosen],
        training_source_sha256=sha(json.dumps(train, sort_keys=True, separators=(",", ":")).encode()),
        history_sha256=owner["history_sha256"], model_calls=0, old_development_reused=False,
        replication_opened=panel == "replication", promotion_allowed=False,
        same_job_matched_control=False, historical_control_retrained=False, model_comparison=primary.MODEL_COMPARISON)
    if panel == "replication":
        admission.update(primary_result_sha256=primary_quality["original_result_sha256"],
                         primary_quality_lock_sha256=sha(canonical(spec["primary_quality_lock"])))
    inventory = dict(original_count=len(originals),
        pcm_sha256=sorted({row["pcm_sha256"] for row in originals}),
        source_group_id=sorted({row["source_group_id"] for row in originals}),
        canonical_people=sorted({person(row) for row in originals}))
    need(inventory["original_count"] == 120 and len(inventory["pcm_sha256"]) == 120,
         "full unique raw source inventory, including every rejected source")
    return dict(admission=admission, selected_rows=selected, source_inventory=inventory, decision=decision)


def saved_arrays(raw_zip, spec, panel, admission, selected_rows, primary, decision):
    """Require an exact PASS from every original raw return and unchanged B14."""
    need(spec["schema"] == SCHEMAS[panel] and spec["scoring_allowed"] is True
         and spec["assessment_sha256"] == spec["training_assessment_sha256"] == ASSESSMENT_SHA
         and spec["training_design_sha256"] == DESIGN_SHA and spec["gains"] == list(GAINS)
         and type(spec["maximum_model_forwards"]) is int and spec["maximum_model_forwards"] == 1440
         and spec["same_job_matched_control"] is spec["historical_control_retrained"] is False
         and spec["model_comparison"] == primary.MODEL_COMPARISON,
         "same preregistered assessment, models, gains and full forward ceiling")
    fixed_models = spec["model_pt_sha256"]
    need(set(fixed_models) == set(ARMS) and all(valid_hash(value) for value in fixed_models.values())
         and len(set(fixed_models.values())) == 4, "four exact original fixed model roles")
    need(type(selected_rows) is list and bool(selected_rows), "complete exact source-admitted originals")
    for row in selected_rows:
        need(type(row["id"]) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", row["id"])
             and valid_hash(row["pcm_sha256"]) and valid_hash(row["wav_sha256"])
             and type(row["samples"]) is int and 3200 <= row["samples"] <= 240000
             and row["cohort"] in ("synthetic", "natural") and row["tier"] in ("silver", "weak")
             and type(row["observed_class"]) is int and row["observed_class"] in (0, 2)
             and (row["cohort"] != "natural" or row["observed_class"] == 0),
             "whole exact admitted original with its frozen source label")
    need(len({row["id"] for row in selected_rows}) == len({row["pcm_sha256"] for row in selected_rows}) == len(selected_rows),
         "every admitted original is present once")
    expected_rows = [primary.score_identity(row, row["cohort"]) for row in selected_rows]
    measured = primary.source_counts(selected_rows)
    need(admission["schema"] == SCHEMAS[panel] and admission["assessment_sha256"] == ASSESSMENT_SHA
         and admission["training_design_sha256"] == DESIGN_SHA
         and equal(admission["admitted_rows"], expected_rows) and equal(admission["source_coverage"], measured)
         and equal(spec["source_gate"], primary.GATE) and all(measured[name] >= minimum for name, minimum in primary.GATE.items())
         and admission["source_gate"] == "SOURCE_GATE_PASS" and admission["source_shortfall"] == {}
         and type(admission["model_calls"]) is int and admission["model_calls"] == 0
         and type(admission["original_source_rows"]) is int and admission["original_source_rows"] == 120
         and admission["old_development_reused"] is admission["promotion_allowed"] is False
         and admission["replication_opened"] is (panel == "replication")
         and admission["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
         and admission["same_job_matched_control"] is admission["historical_control_retrained"] is False
         and admission["model_comparison"] == primary.MODEL_COMPARISON,
         "exact independently reauthenticated whole-source admission")
    identity(admission["identity"])
    calls = len(selected_rows) * 12
    need(calls <= 1440, "unchanged maximum 1440 actual forwards")
    keys = {row["id"] + "__" + gain + "__" + role for row in selected_rows for gain in GAINS for role in ARMS}
    required = {"RESULT.json", "STARTED.json", "ADMISSION.json", "ALL_FRAME_LOGITS.npz"}
    required |= {"attempts/" + key + ".json" for key in keys} | {"raw_outputs/" + key + ".npy" for key in keys}
    with opened(raw_zip) as archive:
        files = {item.filename for item in archive.infolist() if not item.is_dir()}
        dirs = {item.filename for item in archive.infolist() if item.is_dir()}
        need(files == required and dirs <= {"attempts/", "raw_outputs/"}, "every original output, journal and metadata; no missing or orphan member")
        stored_admission = document(archive.read("ADMISSION.json"))
        result = document(archive.read("RESULT.json"))
        started = document(archive.read("STARTED.json"))
        need(equal(stored_admission, admission), "original ADMISSION exactly equals independent raw-source recomputation")
        for report in (started, result):
            need(report["schema"] == SCHEMAS[panel] and equal({key: report[key] for key in IDENTITY_FIELDS}, admission["identity"])
                 and report["assessment_sha256"] == report["training_assessment_sha256"] == ASSESSMENT_SHA
                 and report["training_design_sha256"] == DESIGN_SHA
                 and report["registered_inputs_sha256"] == admission["registered_inputs_sha256"]
                 and report["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
                 and report["workflow_sha256"] == admission["workflow_sha256"]
                 and report["implementation_sha256"] == sha(Path(primary.__file__).read_bytes())
                 and report["source_gate"] == "SOURCE_GATE_PASS" and report["source_shortfall"] == {}
                 and equal(report["source_coverage"], measured) and equal(report["model_pt_sha256"], fixed_models)
                 and report["installed_software_baseline"] == "B0" and report["B20_failed_primary_preserved"] is report["B21_failed_primary_preserved"] is report["B22_failed_primary_preserved"] is True
                 and report["same_job_matched_control"] is report["historical_control_retrained"] is False
                 and report["model_comparison"] == primary.MODEL_COMPARISON
                 and report["fresh_source_originals"] is True and report["original_natural_reused_without_reobservation"] is False
                 and type(report["optimizer_updates"]) is int and report["optimizer_updates"] == 0
                 and all(report[key] is False for key in FALSE_FIELDS)
                 and report["replication_opened"] is (panel == "replication") and report["ssc305_physical"] is False
                 and report["natural_field_far_per_hour"] is report["field_3_5m_frr"] is None,
                 "same actual producer, exact original models and honest unreleased scope")
            if panel == "replication":
                need(report["panel"] == "replication"
                     and report["primary_result_sha256"] == admission["primary_result_sha256"]
                     and report["primary_quality_lock_sha256"] == admission["primary_quality_lock_sha256"],
                     "replica opening binds the exact original primary PASS")
        need(started["panel_gate"] == "STARTED" and started["records"] == []
             and all(type(started[key]) is int and started[key] == 0 for key in ("actual_forwards", "saved_array_count", "raw_output_count"))
             and started["candidate_engineering_validation_allowed"] is started["replication_open_allowed"] is False,
             "honest before-forward STARTED record")
        need(result["panel_gate"] == "PASS" and result["replay_allowed"] is False and result["consumed_unknown"] == []
             and "error" not in result and "error_type" not in result
             and all(type(result[key]) is int and result[key] == calls for key in ("actual_forwards", "saved_array_count", "raw_output_count"))
             and equal(result["model_forward_calls"], dict.fromkeys(ARMS, len(selected_rows) * 3)),
             "only a complete actual PASS; STOP, partial return, unknown consumption or NO_PROMOTION stays closed")
        need(result["replication_open_allowed"] is (panel == "primary")
             and result["candidate_engineering_validation_allowed"] is (panel == "replication"), "next-stage flags exactly match this original PASS")
        if panel == "replication":
            need(result["research_replication_passed"] is True and started["research_replication_passed"] is False,
                 "only completed original replication admits engineering")
        records = result["records"]
        primary.validate_scored_join(records, expected_rows)
        npz = archive.read("ALL_FRAME_LOGITS.npz")
        need(sha(npz) == result["logits_sha256"], "exact original full-frame NPZ hash")
        with opened(npz) as arrays:
            need(set(arrays.namelist()) == {key + ".npy" for key in keys}, "every exact original/gain/model array")
            for source, record in zip(selected_rows, records, strict=True):
                need(set(record) == set(primary.IDENTITY_FIELDS) | {"gains"} and set(record["gains"]) == set(GAINS),
                     "only exact admitted row identity and three fixed gains")
                frames = (source["samples"] - 512) // 160 + 1
                for gain in GAINS:
                    need(set(record["gains"][gain]) == set(ARMS), "all four frozen model roles")
                    for role in ARMS:
                        key = source["id"] + "__" + gain + "__" + role
                        state = record["gains"][gain][role]
                        need(set(state) == STATE_FIELDS and state["array"] == key
                             and type(state["frames"]) is int and state["frames"] == frames
                             and type(state["events"]) is int and state["events"] >= 0, "whole uncropped original frame count")
                        journal = document(archive.read("attempts/" + key + ".json"))
                        need(equal(journal, dict(key=key, role=role, source_id=source["id"], gain=gain, status="consumed_unknown")),
                             "one durable actual-forward-entry journal; no replay")
                        body, labels = read_npy(arrays.read(key + ".npy"), (frames, 4))
                        returned, returned_labels = read_npy(archive.read("raw_outputs/" + key + ".npy"), (1, frames, 4))
                        need(body == returned and labels == returned_labels and sha(body) == state["sha256"],
                             "every retained tensor byte matches the complete actual returned batch")
                        events = primary.ctc_events(labels)
                        need(equal({name: state[name] for name in events}, events), "tokens, events and sample clocks recompute from raw frames")
        replay = decision.decide(records)
        need(replay["panel_gate"] == "PASS" and all(equal(result[key], value) for key, value in replay.items()),
             "every unchanged original B14 condition independently recomputes to PASS")
        return dict(result=result, original_result_sha256=sha(archive.read("RESULT.json")), logits_sha256=sha(npz))


def verify_worker(directory, lock):
    validate_lock(lock)
    directory = Path(directory).resolve()
    panel = lock["panel"]
    run = document((directory / "api/run.json").read_bytes())
    commit = document((directory / "api/commit.json").read_bytes())
    artifacts = document((directory / "api/artifacts.json").read_bytes())
    need(type(artifacts["artifacts"]) is list, "saved original public artifact inventory")
    artifact_names = dict(result="english-v6-b23-historical-control-" + panel + "-logits-" + str(run["id"]),
        producer="english-v6-b23-" + panel + "-producer-evidence-" + str(run["id"]),
        consumed="english-v6-b23-" + panel + "-consumed")
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix="b23-original-quality-") as temporary:
        archives = {key: stack.enter_context(original(directory / "originals" / (key + ".zip"), lock[key], panel=panel, role=key))
                    for key in ("result", "producer", "consumed")}
        producer_directory = Path(temporary) / "producer"
        producer_directory.mkdir()
        archives["producer"].extractall(producer_directory)
        root = producer_directory / (panel + "_producer")
        stem = EVALUATOR_STEMS[panel]
        driver = root / "automation" / (stem + "_eval.py")
        spec_path = root / "automation" / (stem + "_inputs.json")
        producer_path = root / "automation" / (stem + "_producer.json")
        need(sha(driver.read_bytes()) == lock["evaluator_sha256"] and sha(spec_path.read_bytes()) == lock["inputs_sha256"]
             and sha(producer_path.read_bytes()) == lock["producer_manifest_sha256"], "original reviewed code, input registration and producer inventory")
        primary = load(driver, stem + "_eval")
        for key, pin in ((key, lock[key]) for key in ("result", "producer", "consumed")):
            selected = [item for item in artifacts["artifacts"] if item["id"] == pin["id"]]
            need(len(selected) == 1 and selected[0]["name"] == artifact_names[key], "exact named original quality artifact " + key)
            primary.validate_provenance(pin, dict(run=run, commit=commit, artifact=selected[0]))
        spec = primary.protocol()
        need(spec["schema"] == SCHEMAS[panel] and spec["producer_manifest_sha256"] == lock["producer_manifest_sha256"],
             "actual authenticated registration of this quality panel")
        fetched = document((producer_directory / "FETCH.json").read_bytes())
        original_identity = fetched["identity"]
        identity(original_identity)
        need(original_identity["run_id"] == str(lock["result"]["run_id"])
             and original_identity["run_head"] == lock["result"]["run_head"], "quality lock binds the actual original execution")
        producer, workflow = primary.producer_contract(spec, original_identity["prepared_parent"], root=root)
        need(fetched["schema"] == SCHEMAS[panel] and fetched["inputs_sha256"] == lock["inputs_sha256"]
             and fetched["producer_manifest_sha256"] == lock["producer_manifest_sha256"]
             and fetched["workflow_sha256"] == sha(workflow) and fetched["prior_consumed_markers"]["total_count"] == 0,
             "original acquisition, reviewed workflow and no prior consumption")
        bound_commit = fetched[panel + "_commit"]
        primary.validate_execution_metadata(original_identity, fetched[panel + "_run"], bound_commit)
        primary.validate_execution_metadata(original_identity, run, bound_commit)
        need(commit["sha"] == bound_commit["sha"]
             and [item["sha"] for item in commit["parents"]] == [item["sha"] for item in bound_commit["parents"]]
             and commit.get("commit", commit)["tree"]["sha"] == bound_commit["commit"]["tree"]["sha"],
             "independently saved original commit agrees with original acquisition")
        prior_runs = fetched["original_workflow_run_inventory"]
        need(prior_runs["total_count"] == len(prior_runs["workflow_runs"]) == 1
             and prior_runs["workflow_runs"][0]["id"] == int(original_identity["run_id"]), "one first original scoring run only")
        pins = primary.archive_pins(spec)
        need(set(fetched["source_metadata"]) == set(pins), "all twelve original input and complete producer metadata receipts")
        for key, pin in pins.items():
            primary.validate_provenance(pin, fetched["source_metadata"][key])
        expected_files = {"FETCH.json", "CONSUMPTION.json"}
        for role, source_commit in ((panel, bound_commit), ("source", fetched["source_metadata"]["source"]["commit"])):
            tree = fetched["producer_trees"][role]
            need(tree["sha"] == source_commit["commit"]["tree"]["sha"] and tree["truncated"] is False,
                 "actual complete original producer Git tree " + role)
            expected = primary.producer_files(spec, producer, workflow, role)
            target = producer_directory / (role + "_producer")
            need(set(expected) == {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()},
                 "all and only original producer code/resources including hidden workflow " + role)
            for path, digest in expected.items():
                raw = (target / path).read_bytes()
                need(sha(raw) == digest and primary.git_blob(raw) == primary.tree_member(tree, path), "actual original Git-bound producer file " + path)
                expected_files.add(role + "_producer/" + path)
        primary_quality = None
        if panel == "replication":
            proof = producer_directory / "primary_quality"
            primary_quality = verify_quality_original(proof, spec["primary_quality_lock"])
            need(equal(primary_quality["model_pt_sha256"], spec["model_pt_sha256"])
                 and equal(primary_quality["pair_pin"], spec["inputs"]["pair"])
                 and primary_quality["assessment_sha256"] == spec["assessment_sha256"]
                 and primary_quality["training_design_sha256"] == spec["training_design_sha256"],
                 "replication uses the exact already-passed original pair and assessment")
            expected_files |= {"primary_quality/" + path.relative_to(proof).as_posix() for path in proof.rglob("*") if path.is_file()}
        actual_files = {item.filename for item in archives["producer"].infolist() if not item.is_dir()}
        need(actual_files == expected_files, "complete original producer upload; no reconstructed supplement or missing original member")
        evidence = source_admission(primary, directory / "inputs", spec, original_identity, workflow, fetched, panel, primary_quality)
        admission = evidence["admission"]
        consumed_raw = (producer_directory / "CONSUMPTION.json").read_bytes()
        need({item.filename for item in archives["consumed"].infolist() if not item.is_dir()} == {"CONSUMPTION.json"}
             and archives["consumed"].read("CONSUMPTION.json") == consumed_raw, "same independently uploaded original no-replay marker")
        consumed = document(consumed_raw)
        need(equal(consumed, dict(identity=original_identity, marker_name=primary.MARKER, admission_sha256=sha(canonical(admission)))),
             "durable original consumption binds this exact independently repeated admission")
        checked = saved_arrays((directory / "originals/result.zip").read_bytes(), spec, panel, admission,
                              evidence["selected_rows"], primary, evidence["decision"])
        need(checked["original_result_sha256"] == lock["original_result_sha256"], "exact original RESULT member independently locked")
        return dict(schema="english-v6-b23-verified-original-quality-v1", panel=panel, panel_gate="PASS",
            model_pt_sha256=spec["model_pt_sha256"], assessment_sha256=spec["assessment_sha256"],
            training_design_sha256=spec["training_design_sha256"], original_result_sha256=checked["original_result_sha256"],
            pair_pin=spec["inputs"]["pair"], admission=admission, source_inventory=evidence["source_inventory"],
            original_identity=original_identity, original_lock_sha256=sha(canonical(lock)),
            logits_sha256=checked["logits_sha256"], actual_forwards=checked["result"]["actual_forwards"],
            all_original_arrays_verified=True, original_source_admission_recomputed=True, original_producer_git_verified=True,
            new_model_calls=0, new_source_calls=0, new_observer_calls=0, promotion_allowed=False, product_released=False)


def verify_quality_original(directory: Path, lock: dict) -> dict:
    """Return a fully reauthenticated PASS or raise; never load an acoustic runtime."""
    validate_lock(lock)
    need("torch" not in sys.modules and "numpy" not in sys.modules and "onnxruntime" not in sys.modules,
         "original quality authorization must precede acoustic/array runtime imports")
    with tempfile.TemporaryDirectory(prefix="b23-quality-lock-") as temporary:
        path = Path(temporary) / "lock.json"
        path.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                                       "--verify-worker", str(Path(directory).resolve()), str(path)])
    report = document(raw)
    need(report["schema"] == "english-v6-b23-verified-original-quality-v1" and report["panel"] == lock["panel"]
         and report["panel_gate"] == "PASS" and report["original_result_sha256"] == lock["original_result_sha256"]
         and report["original_lock_sha256"] == sha(canonical(lock)), "actual isolated original quality readback")
    return report


def fetch_quality_original(directory: Path, lock: dict) -> dict:
    """Collect the locked original archives with read-only GitHub calls, then verify.

    This function is future execution code. A complete reviewed lock is checked
    before any network or archive access; it never creates source/model calls.
    """
    validate_lock(lock)
    directory = Path(directory).resolve()
    need(not directory.exists() and directory.parent.is_dir(), "new original quality directory only")
    def gh(endpoint):
        return subprocess.run(["gh", "api", "repos/" + REPO + "/" + endpoint, "--allow-escape-sequences"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    directory.mkdir()
    for name in ("api", "originals", "inputs"):
        (directory / name).mkdir()
    run_id, head = lock["result"]["run_id"], lock["result"]["run_head"]
    for name, endpoint in (("run", "actions/runs/" + str(run_id)), ("commit", "commits/" + head),
                           ("artifacts", "actions/runs/" + str(run_id) + "/artifacts?per_page=100")):
        (directory / "api" / (name + ".json")).write_bytes(gh(endpoint))
    for key in ("result", "producer", "consumed"):
        raw = gh("actions/artifacts/" + str(lock[key]["id"]) + "/zip")
        need(len(raw) == lock[key]["bytes"] and sha(raw) == lock[key]["sha256"], "original fetched quality archive " + key)
        (directory / "originals" / (key + ".zip")).write_bytes(raw)
    with original(directory / "originals/producer.zip", lock["producer"], panel=lock['panel'], role='producer') as archive:
        member = lock["panel"] + "_producer/automation/" + EVALUATOR_STEMS[lock["panel"]] + "_inputs.json"
        raw = archive.read(member)
        need(sha(raw) == lock["inputs_sha256"], "exact original input registration before following its pins")
        spec = document(raw)
        need(set(spec["inputs"]) == {"B0", "B9", "pair", "source", "screen"}
             and set(spec["native_inputs"]) == {"qwen", "whisper", "parakeet", "xlsr", "lv60"}, "exact original scientific input roles")
        need(set(spec["upstream_producers"]) == {"train", "source"}, "both complete original upstream producers")
        pins = {**spec["inputs"], **{"native_" + key: value for key, value in spec["native_inputs"].items()},
                **{key + "_producer": value for key, value in spec["upstream_producers"].items()}}
        for key, pin in pins.items():
            valid_artifact(pin, panel=lock['panel'], role=key)
            raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
            need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "complete fetched original scientific input " + key)
            (directory / "inputs" / (key + ".zip")).write_bytes(raw)
    return verify_quality_original(directory, lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-worker", action="store_true")
    parser.add_argument("directory", type=Path)
    parser.add_argument("lock", type=Path)
    args = parser.parse_args()
    bound = document(args.lock.read_bytes())
    if args.verify_worker:
        with offline_only():
            report = verify_worker(args.directory, bound)
    else:
        report = verify_quality_original(args.directory, bound)
    print(json.dumps(report, sort_keys=True, allow_nan=False))

