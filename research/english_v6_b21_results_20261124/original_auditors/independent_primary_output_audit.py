"""Read complete original B21 PRIMARY outputs without authorizing a next stage.

This standalone audit accepts an observed PASS or NO_PROMOTION. It neither edits
nor calls the production PASS-only authorization entry points. Every successful
audit explicitly returns authorization_granted=False. No model is executed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

SCHEMA = "english-v6-b21-independent-primary-output-audit-bundle-v1"
READER_SHA = "5258da3d6428a8bef5b7eee939806137edf5a95ef4aee50a160e8588d1d24cee"
EVALUATOR_SHA = "ca14214dbfe3946e737806aec6cfc099e01e7985cb7d38b9031fd9b641200434"
INPUTS_SHA = "c369c2f104ea468f5c13cd962e9aedffd5ddfff9d07be6a885f88f4d8d558e3b"
PRODUCER_SHA = "dfabcd419cab4031df70c1e6094fb11fd318b45051e01c67f23836bcd19bc55a"
ARMS = ("B0", "B9", "control", "challenger")
GAINS = ("1.0", "0.75", "0.5")
ROLES = ("result", "producer", "consumed")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def independent_npy(raw, expected_shape):
    """Parse all FP32 bytes and perform first-index argmax using stdlib only."""
    need(raw[:6] == b"\x93NUMPY" and len(raw) >= 10, "complete NPY signature")
    version = tuple(raw[6:8])
    need(version in ((1, 0), (2, 0)), "original NPY version")
    width = 2 if version == (1, 0) else 4
    length = int.from_bytes(raw[8:8 + width], "little")
    need(0 < length <= 8192 and len(raw) >= 8 + width + length, "bounded full NPY header")
    header = ast.literal_eval(raw[8 + width:8 + width + length].decode("latin1").strip())
    need(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}
         and header["descr"] == "<f4" and header["fortran_order"] is False
         and type(header["shape"]) is tuple and all(type(v) is int for v in header["shape"])
         and header["shape"] == expected_shape, "exact uncropped contiguous FP32 tensor")
    need(len(expected_shape) in (2, 3) and expected_shape[-1] == 4
         and 1 <= expected_shape[-2] <= 1500
         and (len(expected_shape) == 2 or expected_shape[0] == 1), "full original frame dimensions")
    payload = raw[8 + width + length:]
    need(len(payload) == expected_shape[-2] * 16, "every FP32 output byte present")
    labels = []
    for values in struct.iter_unpack("<4f", payload):
        need(all(math.isfinite(value) for value in values), "finite original tensor")
        index, best = 0, values[0]
        for candidate in range(1, 4):
            if values[candidate] > best:
                index, best = candidate, values[candidate]
        labels.append(index)
    return payload, labels


def independent_decode(labels):
    """Collapse repeats before removing blanks; retain sample-availability time."""
    tokens, starts, last = [], [], None
    for frame, token in enumerate(labels):
        need(type(token) is int and 0 <= token < 4, "four actual frame classes")
        if token != last and token != 0:
            tokens.append(token)
            starts.append(frame)
        last = token
    events = [starts[end] for end in range(2, len(tokens)) if tokens[end - 2:end + 1] == [1, 2, 3]]
    return dict(tokens=tokens, event_frames=events, event_samples=[512 + 160 * frame for frame in events], events=len(events))


def prepare(directory, q):
    run = q.document((directory / "api/run.json").read_bytes())
    items = q.document((directory / "api/artifacts.json").read_bytes())["artifacts"]
    names = dict(result="english-v6-b21-paired-primary-logits-" + str(run["id"]),
                 producer="english-v6-b21-primary-producer-evidence-" + str(run["id"]),
                 consumed="english-v6-b21-primary-consumed")
    pins = {}
    for role, name in names.items():
        candidates = [item for item in items if item["name"] == name]
        need(len(candidates) == 1, "unique original artifact " + role)
        item = candidates[0]
        need(item["digest"].startswith("sha256:"), "native GitHub artifact digest")
        pin = dict(id=item["id"], run_id=run["id"], run_head=run["head_sha"],
                   bytes=item["size_in_bytes"], sha256=item["digest"].removeprefix("sha256:"))
        with q.original(directory / "originals" / (role + ".zip"), pin):
            pass
        pins[role] = pin
    with q.original(directory / "originals/result.zip", pins["result"]) as archive:
        result_sha = sha(archive.read("RESULT.json"))
    return dict(schema=SCHEMA, panel="primary", artifacts=pins,
                evaluator_sha256=EVALUATOR_SHA, inputs_sha256=INPUTS_SHA,
                producer_manifest_sha256=PRODUCER_SHA, original_result_sha256=result_sha)


def verify_outputs(raw, q, primary, spec, evidence, bundle):
    admission, selected, decision = (evidence[key] for key in ("admission", "selected_rows", "decision"))
    need(len(selected) == 90 and admission["source_coverage"] == dict(
         synthetic_positive=22, synthetic_ordinary=48, synthetic_paired=15,
         natural_originals=20, natural_speakers=12), "the exact fully qualified 90-original panel")
    identities = [primary.score_identity(row, row["cohort"]) for row in selected]
    keys = {row["id"] + "__" + gain + "__" + arm for row in selected for gain in GAINS for arm in ARMS}
    required = {"RESULT.json", "ADMISSION.json", "STARTED.json", "ALL_FRAME_LOGITS.npz"}
    required |= {"attempts/" + key + ".json" for key in keys} | {"raw_outputs/" + key + ".npy" for key in keys}
    count, total_frames, recomputed = Counter(), 0, []
    with q.opened(raw) as archive:
        need({item.filename for item in archive.infolist() if not item.is_dir()} == required
             and {item.filename for item in archive.infolist() if item.is_dir()} <= {"attempts/", "raw_outputs/"},
             "all and only original output, entry journal and metadata members")
        result_raw = archive.read("RESULT.json")
        need(sha(result_raw) == bundle["original_result_sha256"], "original RESULT member digest")
        result, started = q.document(result_raw), q.document(archive.read("STARTED.json"))
        need(q.equal(q.document(archive.read("ADMISSION.json")), admission), "admission exactly rederived from original sources")
        for report in (started, result):
            need(report["schema"] == spec["schema"] and q.equal({k: report[k] for k in q.IDENTITY_FIELDS}, admission["identity"])
                 and report["assessment_sha256"] == report["training_assessment_sha256"] == q.ASSESSMENT_SHA
                 and report["training_design_sha256"] == q.DESIGN_SHA
                 and report["registered_inputs_sha256"] == INPUTS_SHA
                 and report["producer_manifest_sha256"] == PRODUCER_SHA
                 and report["implementation_sha256"] == EVALUATOR_SHA
                 and report["workflow_sha256"] == admission["workflow_sha256"]
                 and report["source_gate"] == "SOURCE_GATE_PASS" and report["source_shortfall"] == {}
                 and q.equal(report["source_coverage"], admission["source_coverage"])
                 and q.equal(report["model_pt_sha256"], spec["model_pt_sha256"])
                 and all(report[key] is False for key in q.FALSE_FIELDS)
                 and report["installed_software_baseline"] == "B0" and report["B20_failed_primary_preserved"] is True
                 and report["fresh_source_originals"] is True and report["original_natural_reused_without_reobservation"] is False
                 and type(report["optimizer_updates"]) is int and report["optimizer_updates"] == 0
                 and report["replication_opened"] is report["candidate_engineering_validation_allowed"] is report["ssc305_physical"] is False
                 and report["natural_field_far_per_hour"] is report["field_3_5m_frr"] is None,
                 "original frozen source/model identity and truthful unreleased scope")
        need(started["panel_gate"] == "STARTED" and started["records"] == []
             and started["replication_open_allowed"] is False
             and all(type(started[k]) is int and started[k] == 0 for k in ("actual_forwards", "saved_array_count", "raw_output_count")),
             "pre-forward STARTED record")
        need(result["panel_gate"] in ("PASS", "NO_PROMOTION")
             and result["replication_open_allowed"] is (result["panel_gate"] == "PASS")
             and result["replay_allowed"] is False and result["consumed_unknown"] == []
             and "error" not in result and "error_type" not in result
             and all(type(result[k]) is int and result[k] == 1080 for k in ("actual_forwards", "saved_array_count", "raw_output_count"))
             and q.equal(result["model_forward_calls"], dict.fromkeys(ARMS, 270)),
             "complete original attempt; observing a verdict grants no permission")
        primary.validate_scored_join(result["records"], identities)
        npz = archive.read("ALL_FRAME_LOGITS.npz")
        need(sha(npz) == result["logits_sha256"], "complete original NPZ digest")
        with q.opened(npz) as arrays:
            need(set(arrays.namelist()) == {key + ".npy" for key in keys}, "all 1080 exact output keys")
            for source, identity, record in zip(selected, identities, result["records"], strict=True):
                need(set(record) == set(primary.IDENTITY_FIELDS) | {"gains"}
                     and set(record["gains"]) == set(GAINS), "exact source record and registered gains")
                output = dict(identity, gains={})
                frames = (source["samples"] - 512) // 160 + 1
                for gain in GAINS:
                    need(set(record["gains"][gain]) == set(ARMS), "all four frozen models")
                    output["gains"][gain] = {}
                    for arm in ARMS:
                        key = source["id"] + "__" + gain + "__" + arm
                        saved = record["gains"][gain][arm]
                        need(set(saved) == q.STATE_FIELDS and saved["array"] == key
                             and type(saved["frames"]) is int and saved["frames"] == frames
                             and type(saved["events"]) is int and saved["events"] >= 0, "whole original frame count and state")
                        journal = q.document(archive.read("attempts/" + key + ".json"))
                        need(q.equal(journal, dict(key=key, role=arm, source_id=source["id"], gain=gain, status="consumed_unknown")),
                             "durable original write-before-forward journal")
                        body, labels = independent_npy(arrays.read(key + ".npy"), (frames, 4))
                        returned, returned_labels = independent_npy(archive.read("raw_outputs/" + key + ".npy"), (1, frames, 4))
                        need(body == returned and labels == returned_labels and sha(body) == saved["sha256"],
                             "actual raw tensor and saved array are byte-identical")
                        decoded = independent_decode(labels)
                        need(q.equal({name: saved[name] for name in decoded}, decoded), "independent argmax, CTC tokens/events and sample clocks")
                        output["gains"][gain][arm] = decoded
                        count[arm] += 1
                        total_frames += frames
                recomputed.append(output)
        verdict = decision.decide(recomputed)
        need(all(q.equal(result[k], v) for k, v in verdict.items()), "entire unchanged B14 decision from independent decoded arrays")
    return dict(observed_panel_gate=verdict["panel_gate"], decision=verdict,
                exact_raw_and_saved_arrays=sum(count.values()), total_verified_frames=total_frames,
                model_forward_calls=dict(count), logits_sha256=sha(npz),
                original_result_sha256=sha(result_raw), original_attempt_journals_verified=len(keys))


def verify(directory, bundle, q):
    need(set(bundle) == {"schema", "panel", "artifacts", "evaluator_sha256", "inputs_sha256", "producer_manifest_sha256", "original_result_sha256"}
         and bundle["schema"] == SCHEMA and bundle["panel"] == "primary"
         and bundle["evaluator_sha256"] == EVALUATOR_SHA and bundle["inputs_sha256"] == INPUTS_SHA
         and bundle["producer_manifest_sha256"] == PRODUCER_SHA and q.valid_hash(bundle["original_result_sha256"])
         and set(bundle["artifacts"]) == set(ROLES), "separate observation-only audit bundle")
    pins = bundle["artifacts"]
    for pin in pins.values():
        q.valid_artifact(pin)
    need(len({p["id"] for p in pins.values()}) == 3
         and all(pin[k] == pins["result"][k] for pin in pins.values() for k in ("run_id", "run_head")), "three originals from the same attempt")
    run = q.document((directory / "api/run.json").read_bytes())
    commit = q.document((directory / "api/commit.json").read_bytes())
    artifacts = q.document((directory / "api/artifacts.json").read_bytes())
    names = dict(result="english-v6-b21-paired-primary-logits-" + str(run["id"]),
                 producer="english-v6-b21-primary-producer-evidence-" + str(run["id"]),
                 consumed="english-v6-b21-primary-consumed")
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix="b21-independent-output-audit-") as temporary:
        archives = {role: stack.enter_context(q.original(directory / "originals" / (role + ".zip"), pins[role])) for role in ROLES}
        producer_root = Path(temporary) / "producer"; producer_root.mkdir()
        archives["producer"].extractall(producer_root)
        root = producer_root / "primary_producer"
        driver = root / "automation/english_v6_b21_fixed_mirror_primary_eval.py"
        need(sha(driver.read_bytes()) == EVALUATOR_SHA
             and sha((root / "automation/english_v6_b21_fixed_mirror_primary_inputs.json").read_bytes()) == INPUTS_SHA
             and sha((root / "automation/english_v6_b21_fixed_mirror_primary_producer.json").read_bytes()) == PRODUCER_SHA,
             "original Git-bound evaluator, registration and producer")
        primary = load(driver, "b21_independent_original_primary")
        for role, pin in pins.items():
            matches = [item for item in artifacts["artifacts"] if item["id"] == pin["id"]]
            need(len(matches) == 1 and matches[0]["name"] == names[role], "named original artifact")
            primary.validate_provenance(pin, dict(run=run, commit=commit, artifact=matches[0]))
        spec = primary.protocol()
        fetched = q.document((producer_root / "FETCH.json").read_bytes())
        ident = fetched["identity"]; q.identity(ident)
        need(ident["run_id"] == str(pins["result"]["run_id"]) and ident["run_head"] == pins["result"]["run_head"], "exact original run identity")
        producer, workflow = primary.producer_contract(spec, ident["prepared_parent"], root=root)
        need(fetched["schema"] == spec["schema"] and fetched["inputs_sha256"] == INPUTS_SHA
             and fetched["producer_manifest_sha256"] == PRODUCER_SHA and fetched["workflow_sha256"] == sha(workflow)
             and fetched["prior_consumed_markers"]["total_count"] == 0, "original unused acquisition and reviewed workflow")
        bound_commit = fetched["primary_commit"]
        primary.validate_execution_metadata(ident, fetched["primary_run"], bound_commit)
        primary.validate_execution_metadata(ident, run, bound_commit)
        need(commit["sha"] == bound_commit["sha"]
             and [p["sha"] for p in commit["parents"]] == [p["sha"] for p in bound_commit["parents"]]
             and commit.get("commit", commit)["tree"]["sha"] == bound_commit["commit"]["tree"]["sha"], "independent final commit metadata agrees")
        inventory = fetched["original_workflow_run_inventory"]
        need(inventory["total_count"] == len(inventory["workflow_runs"]) == 1
             and inventory["workflow_runs"][0]["id"] == int(ident["run_id"]), "unique first original execution")
        input_pins = primary.archive_pins(spec)
        need(set(fetched["source_metadata"]) == set(input_pins), "all ten scientific input receipts")
        for role, pin in input_pins.items():
            primary.validate_provenance(pin, fetched["source_metadata"][role])
        required_files = {"FETCH.json", "CONSUMPTION.json"}
        producer_counts = {}
        for role, origin in (("primary", bound_commit), ("source", fetched["source_metadata"]["source"]["commit"])):
            tree = fetched["producer_trees"][role]
            need(tree["sha"] == origin["commit"]["tree"]["sha"] and tree["truncated"] is False, "complete original producer Git tree")
            expected = primary.producer_files(spec, producer, workflow, role)
            target = producer_root / (role + "_producer")
            need(set(expected) == {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}, "all and only original producer members")
            for path, digest in expected.items():
                raw = (target / path).read_bytes()
                need(sha(raw) == digest and primary.git_blob(raw) == primary.tree_member(tree, path), "actual original Git-bound bytes " + path)
                required_files.add(role + "_producer/" + path)
            producer_counts[role] = len(expected)
        need({item.filename for item in archives["producer"].infolist() if not item.is_dir()} == required_files, "no producer supplement or omission")
        evidence = q.source_admission(primary, directory / "inputs", spec, ident, workflow, fetched, "primary")
        admission = evidence["admission"]
        consumed_raw = (producer_root / "CONSUMPTION.json").read_bytes()
        need({item.filename for item in archives["consumed"].infolist() if not item.is_dir()} == {"CONSUMPTION.json"}
             and archives["consumed"].read("CONSUMPTION.json") == consumed_raw, "independently uploaded original consumption marker")
        need(q.equal(q.document(consumed_raw), dict(identity=ident, marker_name=primary.MARKER, admission_sha256=sha(q.canonical(admission)))), "consumption binds independently repeated source admission")
        outputs = verify_outputs((directory / "originals/result.zip").read_bytes(), q, primary, spec, evidence, bundle)
        need(not {"torch", "numpy", "onnxruntime"} & set(sys.modules), "no acoustic or array runtime imported")
        return dict(schema="english-v6-b21-independent-primary-observation-v1", status="COMPLETE_ORIGINAL_OUTPUTS_AND_PROVENANCE_REPRODUCED",
                    **outputs, original_identity=ident, artifacts=pins, original_source_denominator=120,
                    admitted_originals=len(evidence["selected_rows"]), source_coverage=admission["source_coverage"],
                    source_inventory=evidence["source_inventory"], producer_files_verified=producer_counts,
                    original_source_admission_recomputed=True, original_producer_git_verified=True,
                    all_original_arrays_verified=True, model_pt_sha256=spec["model_pt_sha256"],
                    authorization_granted=False, replication_open_allowed=False,
                    candidate_engineering_validation_allowed=False, promotion_allowed=False, product_released=False,
                    installed_software_baseline="B0", ssc305_physical=False,
                    new_model_forwards=0, new_source_calls=0, new_observer_calls=0, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    need(sha(args.reader.read_bytes()) == READER_SHA, "exact independently reviewed pure source/provenance reader")
    q = load(args.reader.resolve(), "b21_original_readonly_helpers")
    with q.offline_only():
        if args.mode == "prepare":
            need(not args.bundle.exists(), "new audit bundle only")
            report = prepare(args.directory.resolve(), q)
            with args.bundle.open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True); handle.write("\n")
        else:
            need(args.output is not None and not args.output.exists(), "new audit report only")
            report = verify(args.directory.resolve(), q.document(args.bundle.read_bytes()), q)
            with args.output.open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps({key: report[key] for key in ("schema", "status", "observed_panel_gate", "exact_raw_and_saved_arrays", "new_model_forwards", "authorization_granted") if key in report}))


if __name__ == "__main__":
    main()
