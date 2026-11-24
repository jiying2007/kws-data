"""Pure original-array verification for B20; no source/model/runtime execution.

The caller must authenticate the original ZIP pin, actual public producer run,
scientific code/resources and exact raw-source admission before this function.
This module independently verifies every retained output and repeats unchanged
B14 quality. A structurally valid fixture is never an artifact authorization.
"""
from __future__ import annotations
import ast
import hashlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile

PRIMARY_MODULE = "english_v6_b20_primary_eval"
PRIMARY_CODE_SHA = "87cb205d5bbd5e5ba17bd0a5929b7e123c5b5688894e753d34406535273ca802"
QUALITY_MODULE = "english_v6_b14_decision"
QUALITY_CODE_SHA = "6aa5f999598e47b41eb0a946acd82fdc5b47d30aea8990871c1c1a01065e4023"
QUALITY_ASSESSMENT_SHA = "944f5df3c7bc3244977a8a33f7e1c393bffff9f7f5c32d2d8cc79e2628e265d9"
ASSESSMENT_SHA = "22a238a7c773912bd4bc1429e8b3efbd18147a793a68a798fa064417713afd87"
SCHEMAS = {panel: "english-v6-b20-b17-frozen-models-" + panel + "-v1"
           for panel in ("primary", "replication")}
MAXIMUM = {"primary": 1404, "replication": 1368}
ARMS = ("B0", "B9", "control", "challenger")
GAINS = ("1.0", "0.75", "0.5")
STATE_FIELDS = {"array", "sha256", "frames", "tokens", "event_frames", "event_samples", "events"}
FALSE_FIELDS = ("training_allowed", "threshold_search", "promotion_allowed", "product_released",
                "cal_locked_access", "old_replication_opened", "B17_failed_synthetic_used_for_scoring",
                "B18_failed_synthetic_used_for_scoring", "B19_canary_used_for_scoring")
IDENTITY_FIELDS = ("run_id", "run_head", "run_attempt", "prepared_parent")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def valid_hash(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def equal(left, right):
    # JSON distinguishes true/false from 1/0 even though Python equality does not.
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


def dependencies():
    loaded = []
    for name, expected in ((PRIMARY_MODULE, PRIMARY_CODE_SHA), (QUALITY_MODULE, QUALITY_CODE_SHA)):
        found = importlib.util.find_spec(name)
        need(found is not None and found.origin and sha(Path(found.origin).read_bytes()) == expected,
             "unchanged frozen scientific implementation " + name)
        loaded.append(importlib.import_module(name))
    primary, decision = loaded
    need(sha(decision.ASSESSMENT.read_bytes()) == QUALITY_ASSESSMENT_SHA,
         "unchanged original B14 quality assessment")
    return primary, decision


def opened(raw, maximum=128_000_000):
    need(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded complete original score archive")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    try:
        entries = archive.infolist()
        need(len(entries) <= 8192 and len(entries) == len({entry.filename for entry in entries})
             and sum(entry.file_size for entry in entries) <= maximum, "unique bounded archive members")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            need(path.parts and not path.is_absolute() and ".." not in path.parts
                 and "\\" not in entry.filename and not stat.S_ISLNK(entry.external_attr >> 16)
                 and not (entry.flag_bits & 1), "safe unencrypted original score member")
        need(archive.testzip() is None, "complete original score archive CRC")
        return archive
    except BaseException:
        archive.close()
        raise


def read_npy(raw, expected_shape):
    """Read exact finite FP32 NPY bytes with no NumPy, Torch or pickle loader."""
    need(type(raw) is bytes and raw[:6] == b"\x93NUMPY" and len(raw) >= 10, "real bounded NPY member")
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


def identity(value):
    need(type(value) is dict and set(value) == set(IDENTITY_FIELDS)
         and type(value["run_id"]) is str and re.fullmatch(r"[1-9][0-9]*", value["run_id"])
         and type(value["run_attempt"]) is int and value["run_attempt"] == 1
         and valid_hash(value["run_head"], 40) and valid_hash(value["prepared_parent"], 40)
         and value["run_head"] != value["prepared_parent"], "actual first-run/head/unique-parent identity")


def coverage(rows):
    groups = {}
    for row in rows:
        if row["cohort"] == "synthetic":
            groups.setdefault(row["source_group_id"], set()).add(row["observed_class"])
    return dict(synthetic_positive=sum(row["cohort"] == "synthetic" and row["observed_class"] == 2 for row in rows),
                synthetic_ordinary=sum(row["cohort"] == "synthetic" and row["observed_class"] == 0 for row in rows),
                synthetic_paired=sum(values == {0, 2} for values in groups.values()),
                natural_originals=sum(row["cohort"] == "natural" for row in rows),
                natural_speakers=len({row["source_group_id"] for row in rows if row["cohort"] == "natural"}))


def _read_saved_result(raw_zip, spec, panel, admission, selected_rows, *, require_pass):
    """Shared complete-array validation; callers select the public boundary.

    `selected_rows` is a flat list of exact raw source dictionaries with a
    `cohort` field. It must come from the caller's original source/native SCREEN
    readback, including the special fixed B17 natural reuse authorization.
    """
    need(panel in SCHEMAS, "only new B20 primary or replication outputs")
    primary, decision = dependencies()
    fixed_models = primary.fixed()["model_pt_sha256"]
    need(spec["schema"] == SCHEMAS[panel] and spec["scoring_allowed"] is True
         and spec["assessment_sha256"] == ASSESSMENT_SHA
         and spec["training_assessment_sha256"] == primary.TRAINING_ASSESSMENT
         and spec["training_design_sha256"] == primary.TRAINING_DESIGN
         and equal(spec["model_pt_sha256"], fixed_models) and spec["gains"] == list(GAINS)
         and type(spec["maximum_model_forwards"]) is int and spec["maximum_model_forwards"] == MAXIMUM[panel]
         and valid_hash(spec["producer_manifest_sha256"]), "same four original weights and fixed B20 assessment")
    need(type(selected_rows) is list and bool(selected_rows) and all(type(row) is dict for row in selected_rows),
         "flat exact admitted raw-source dictionaries")
    for row in selected_rows:
        need(type(row["id"]) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", row["id"])
             and valid_hash(row["pcm_sha256"]) and valid_hash(row["wav_sha256"])
             and type(row["samples"]) is int and 3200 <= row["samples"] <= 240000
             and row["cohort"] in ("synthetic", "natural") and row["tier"] in ("silver", "weak")
             and type(row["observed_class"]) is int and row["observed_class"] in (0, 2)
             and (row["cohort"] != "natural" or row["observed_class"] == 0),
             "only exact whole admitted silver/weak originals on the frozen feature frame grid")
    need(len({row["id"] for row in selected_rows}) == len({row["pcm_sha256"] for row in selected_rows}) == len(selected_rows),
         "each admitted original appears exactly once")
    expected_rows = [primary.score_identity(row, row["cohort"]) for row in selected_rows]
    measured = coverage(selected_rows)
    need(admission["schema"] == SCHEMAS[panel] and admission["assessment_sha256"] == ASSESSMENT_SHA
         and equal(admission["admitted_rows"], expected_rows) and equal(admission["source_coverage"], measured)
         and equal(spec["source_gate"], primary.GATE)
         and all(measured[name] >= threshold for name, threshold in primary.GATE.items())
         and admission["source_gate"] == "SOURCE_GATE_PASS" and admission["source_shortfall"] == {}
         and type(admission["model_calls"]) is int and admission["model_calls"] == 0
         and admission["promotion_allowed"] is False and admission["B19_canary_excluded"] is True
         and valid_hash(admission["registered_inputs_sha256"])
         and admission["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
         and valid_hash(admission["workflow_sha256"]) and valid_hash(admission["training_source_sha256"]),
         "externally authenticated exact source admission and unchanged source minima")
    identity(admission["identity"])
    if panel == "primary":
        need(type(admission["original_source_rows"]) is int and admission["original_source_rows"] == 120,
             "all 96 plus fixed original 24 natural sources")
    else:
        need(type(admission["original_source_rows"]) is int and admission["original_source_rows"] == 114
             and valid_hash(admission["primary_result_sha256"]) and valid_hash(admission["opening_lock_sha256"]),
             "complete new replica original denominator and authentic primary authorization")
    calls = len(selected_rows) * 12
    need(calls <= MAXIMUM[panel], "unchanged total forward ceiling")
    expected_keys = {row["id"] + "__" + gain + "__" + role for row in selected_rows for gain in GAINS for role in ARMS}
    required = {"RESULT.json", "STARTED.json", "ADMISSION.json", "ALL_FRAME_LOGITS.npz"}
    required |= {"attempts/" + key + ".json" for key in expected_keys}
    required |= {"raw_outputs/" + key + ".npy" for key in expected_keys}
    with opened(raw_zip) as archive:
        files = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
        dirs = {entry.filename for entry in archive.infolist() if entry.is_dir()}
        need(files == required and dirs <= {"attempts/", "raw_outputs/"},
             "all and only original score metadata, actual attempt journals and every raw return")
        stored_admission = document(archive.read("ADMISSION.json"))
        result = document(archive.read("RESULT.json"))
        started = document(archive.read("STARTED.json"))
        need(equal(stored_admission, admission), "saved ADMISSION is the exact original-source readback")
        for report in (started, result):
            need(report["schema"] == SCHEMAS[panel]
                 and equal({name: report[name] for name in IDENTITY_FIELDS}, admission["identity"])
                 and report["assessment_sha256"] == ASSESSMENT_SHA
                 and report["training_assessment_sha256"] == primary.TRAINING_ASSESSMENT
                 and report["training_design_sha256"] == primary.TRAINING_DESIGN
                 and report["registered_inputs_sha256"] == admission["registered_inputs_sha256"]
                 and report["producer_manifest_sha256"] == spec["producer_manifest_sha256"]
                 and report["workflow_sha256"] == admission["workflow_sha256"]
                 and report["source_gate"] == "SOURCE_GATE_PASS" and report["source_shortfall"] == {}
                 and equal(report["source_coverage"], measured) and equal(report["model_pt_sha256"], fixed_models)
                 and report["installed_software_baseline"] == "B0"
                 and report["B17_source_failure_preserved"] is True and report["B18_source_failure_preserved"] is True
                 and type(report["optimizer_updates"]) is int and report["optimizer_updates"] == 0
                 and all(report[name] is False for name in FALSE_FIELDS)
                 and report["ssc305_physical"] is False and report["natural_field_far_per_hour"] is None
                 and report["field_3_5m_frr"] is None and valid_hash(report["implementation_sha256"]),
                 "same actual producer, original models, source admission and honest unreleased scope")
            if panel == "primary":
                need(report["implementation_sha256"] == PRIMARY_CODE_SHA
                     and report["original_natural_reused_without_reobservation"] is True
                     and report["replication_opened"] is False,
                     "original frozen primary implementation and precise natural reuse only")
            else:
                need(report["panel"] == "replication" and report["replication_opened"] is True
                     and report["B20_replication_opened"] is True and report["replication_open_allowed"] is False
                     and report["primary_result_sha256"] == admission["primary_result_sha256"]
                     and report["opening_lock_sha256"] == admission["opening_lock_sha256"],
                     "only the new authenticated B20 replica was opened")
        need(started["implementation_sha256"] == result["implementation_sha256"]
             and started["panel_gate"] == "STARTED" and started["records"] == []
             and all(type(started[name]) is int and started[name] == 0 for name in
                     ("actual_forwards", "saved_array_count", "raw_output_count"))
             and started["candidate_engineering_validation_allowed"] is False
             and started["replication_open_allowed"] is False,
             "honest before-forward STARTED evidence")
        need(result["panel_gate"] in (("PASS",) if require_pass else ("PASS", "NO_PROMOTION"))
             and result["replay_allowed"] is False
             and result["consumed_unknown"] == [] and "error" not in result and "error_type" not in result
             and all(type(result[name]) is int and result[name] == calls for name in
                     ("actual_forwards", "saved_array_count", "raw_output_count"))
             and equal(result["model_forward_calls"], dict.fromkeys(ARMS, len(selected_rows) * 3)),
             "genuine complete one-shot outputs, no unknown consumption or fake counters")
        passed = result["panel_gate"] == "PASS"
        need((panel == "primary" and result["replication_open_allowed"] is passed
              and result["candidate_engineering_validation_allowed"] is False)
             or (panel == "replication" and result["research_replication_passed"] is passed
                 and result["candidate_engineering_validation_allowed"] is passed
                 and started["research_replication_passed"] is False),
             "original next-gate flags must match the actual completed decision")
        records = result["records"]
        primary.validate_scored_join(records, expected_rows)
        npz_bytes = archive.read("ALL_FRAME_LOGITS.npz")
        need(sha(npz_bytes) == result["logits_sha256"], "exact whole original frame archive hash")
        with opened(npz_bytes) as arrays:
            need(set(arrays.namelist()) == {key + ".npy" for key in expected_keys},
                 "every original/gain/model array with no orphan, duplicate or missing entry")
            for source, record in zip(selected_rows, records, strict=True):
                need(set(record) == set(primary.IDENTITY_FIELDS) | {"gains"}
                     and set(record["gains"]) == set(GAINS), "exact saved row identity and gain slots")
                frames = (source["samples"] - 512) // 160 + 1
                for gain in GAINS:
                    need(set(record["gains"][gain]) == set(ARMS), "all four fixed model roles")
                    for role in ARMS:
                        key = source["id"] + "__" + gain + "__" + role
                        state = record["gains"][gain][role]
                        need(set(state) == STATE_FIELDS and state["array"] == key
                             and type(state["frames"]) is int and state["frames"] == frames
                             and type(state["events"]) is int and state["events"] >= 0,
                             "exact original/gain/role and uncropped frame count")
                        journal = document(archive.read("attempts/" + key + ".json"))
                        need(equal(journal, dict(key=key, role=role, source_id=source["id"], gain=gain, status="consumed_unknown")),
                             "one honest actual forward-entry journal; no successful/unknown replay")
                        body, labels = read_npy(arrays.read(key + ".npy"), (frames, 4))
                        raw_body, raw_labels = read_npy(archive.read("raw_outputs/" + key + ".npy"), (1, frames, 4))
                        need(body == raw_body and labels == raw_labels and sha(body) == state["sha256"],
                             "whole returned tensor is byte-identical to every retained frame array")
                        decoded = primary.ctc_events(labels)
                        need(equal({name: state[name] for name in decoded}, decoded),
                             "all tokens/events/frames/sample clocks recompute from original frame bytes")
        replay = decision.decide(records)
        need((not require_pass or replay["panel_gate"] == "PASS")
             and all(equal(result[name], value) for name, value in replay.items()),
             "all unchanged B14 quality conditions must recompute exactly" + (" to PASS" if require_pass else ""))
    return dict(result=result, admission=stored_admission, records=records, npz_bytes=npz_bytes)


def verify_saved_result(raw_zip, spec, panel, admission, selected_rows):
    """PASS-only full-array check for an externally authorized future gate.

    A completed NO_PROMOTION, STOP, partial return or consumed-unknown outcome
    cannot pass this entry. Provenance and exact source authorization remain
    the caller's responsibility; this function never manufactures a permit.
    """
    return _read_saved_result(raw_zip, spec, panel, admission, selected_rows, require_pass=True)


def audit_saved_result(raw_zip, spec, panel, admission, selected_rows):
    """Audit a complete PASS or NO_PROMOTION without granting any permission.

    Uses exactly the same raw tensor, source-row, identity, attempt, counter and
    B14 checks as the gate reader. STOP/partial or unknown-consumption outputs
    still fail. The distinct audit shape intentionally has no `result` or
    `permit` key; original flags are retained only inside `original_result`.
    """
    checked = _read_saved_result(raw_zip, spec, panel, admission, selected_rows, require_pass=False)
    return dict(schema="english-v6-b20-completed-result-array-audit-v1", panel=panel,
        audit_only=True, authorization_granted=False, replication_open_allowed=False,
        candidate_engineering_validation_allowed=False, training_allowed=False,
        promotion_allowed=False, product_released=False, model_calls=0,
        audit_status="COMPLETE_ARRAYS_AND_B14_RECOMPUTED", panel_gate=checked["result"]["panel_gate"],
        original_archive_sha256=sha(raw_zip), original_result=checked["result"],
        original_admission=checked["admission"], records=checked["records"], npz_bytes=checked["npz_bytes"])
