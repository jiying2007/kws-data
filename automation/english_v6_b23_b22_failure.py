"""Authenticate B22 NO_PROMOTION before a distinct TRAIN experiment may start.

This fail-only reader reuses the original stdlib provenance/source primitives.
The copied numerical readback functions differ only at the required failed
verdict and downstream-closed flags. Original B22 PASS-only readers are intact.
A partial/STOP result, unreadable original, or successful model cannot open B23.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from english_v6_b22_quality_readback import (REPO, PRIMARY_CODE_SHA, EVALUATOR_STEMS,
    SCHEMAS, ARMS, GAINS, ASSESSMENT_SHA, DESIGN_SHA, IDENTITY_FIELDS, FALSE_FIELDS,
    STATE_FIELDS, need, sha, valid_hash, canonical, equal, document, valid_artifact,
    opened, original, identity, read_npy, load, offline_only, source_admission,
    validate_lock as original_validate_lock)


BASE_READER_PATH = Path(__file__).with_name('english_v6_b22_quality_readback.py')
BASE_READER_SHA = '63201dd5479c9a7a86eb318ec871747a46ae2750f21bf7beaa6f2fd2d562818c'

def validate_lock(lock):
    need(sha(BASE_READER_PATH.read_bytes()) == BASE_READER_SHA, 'unchanged original B22 stdlib provenance and source verifier')
    original_validate_lock(lock)
    need(lock['panel'] == 'primary', 'only actual B22 primary NO_PROMOTION opens this single new TRAIN experiment')

def saved_failure_arrays(raw_zip, spec, panel, admission, selected_rows, primary, decision):
    """Require exact NO_PROMOTION from every original raw return and unchanged B14."""
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
                 and report["installed_software_baseline"] == "B0" and report["B20_failed_primary_preserved"] is report["B21_failed_primary_preserved"] is True
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
        need(result["panel_gate"] == "NO_PROMOTION" and result["replay_allowed"] is False and result["consumed_unknown"] == []
             and "error" not in result and "error_type" not in result
             and all(type(result[key]) is int and result[key] == calls for key in ("actual_forwards", "saved_array_count", "raw_output_count"))
             and equal(result["model_forward_calls"], dict.fromkeys(ARMS, len(selected_rows) * 3)),
             "only complete actual NO_PROMOTION; STOP, partial return, unknown consumption and PASS stay closed")
        need(result["replication_open_allowed"] is False
             and result["candidate_engineering_validation_allowed"] is False, "failed candidate cannot open any next-quality or engineering stage")
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
        need(replay["panel_gate"] == "NO_PROMOTION" and all(equal(result[key], value) for key, value in replay.items()),
             "unchanged B14 independently recomputes the original NO_PROMOTION")
        return dict(result=result, original_result_sha256=sha(archive.read("RESULT.json")), logits_sha256=sha(npz))

def verify_failure_worker(directory, lock):
    validate_lock(lock)
    directory = Path(directory).resolve()
    panel = lock["panel"]
    run = document((directory / "api/run.json").read_bytes())
    commit = document((directory / "api/commit.json").read_bytes())
    artifacts = document((directory / "api/artifacts.json").read_bytes())
    need(type(artifacts["artifacts"]) is list, "saved original public artifact inventory")
    artifact_names = dict(result="english-v6-b22-historical-control-" + panel + "-logits-" + str(run["id"]),
        producer="english-v6-b22-" + panel + "-producer-evidence-" + str(run["id"]),
        consumed="english-v6-b22-" + panel + "-consumed")
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix="b22-original-quality-") as temporary:
        archives = {key: stack.enter_context(original(directory / "originals" / (key + ".zip"), lock[key]))
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
        checked = saved_failure_arrays((directory / "originals/result.zip").read_bytes(), spec, panel, admission,
                              evidence["selected_rows"], primary, evidence["decision"])
        need(checked["original_result_sha256"] == lock["original_result_sha256"], "exact original RESULT member independently locked")
        return dict(schema="english-v6-b23-original-b22-no-promotion-v1", panel=panel, panel_gate="NO_PROMOTION",
            model_pt_sha256=spec["model_pt_sha256"], assessment_sha256=spec["assessment_sha256"],
            training_design_sha256=spec["training_design_sha256"], original_result_sha256=checked["original_result_sha256"],
            pair_pin=spec["inputs"]["pair"], admission=admission, source_inventory=evidence["source_inventory"],
            original_identity=original_identity, original_lock_sha256=sha(canonical(lock)),
            logits_sha256=checked["logits_sha256"], actual_forwards=checked["result"]["actual_forwards"],
            all_original_arrays_verified=True, original_source_admission_recomputed=True, original_producer_git_verified=True,
            authorization_granted=False, replication_open_allowed=False, candidate_engineering_validation_allowed=False,
            decision=evidence['decision'].decide(checked['result']['records']),
            model_forward_calls=checked['result']['model_forward_calls'],
            verified_frame_count=sum(state['frames'] for row in checked['result']['records']
                for roles in row['gains'].values() for state in roles.values()),
            new_model_calls=0, new_source_calls=0, new_observer_calls=0, promotion_allowed=False, product_released=False)

def verify_failure_original(directory: Path, lock: dict) -> dict:
    """Return a fully reauthenticated NO_PROMOTION or raise; never load an acoustic runtime."""
    validate_lock(lock)
    need("torch" not in sys.modules and "numpy" not in sys.modules and "onnxruntime" not in sys.modules,
         "original quality authorization must precede acoustic/array runtime imports")
    with tempfile.TemporaryDirectory(prefix="b22-quality-lock-") as temporary:
        path = Path(temporary) / "lock.json"
        path.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                                       "--verify-worker", str(Path(directory).resolve()), str(path)])
    report = document(raw)
    need(report["schema"] == "english-v6-b23-original-b22-no-promotion-v1" and report["panel"] == lock["panel"]
         and report["panel_gate"] == "NO_PROMOTION" and report["original_result_sha256"] == lock["original_result_sha256"]
         and report["original_lock_sha256"] == sha(canonical(lock)), "actual isolated original quality readback")
    return report

def fetch_failure_original(directory: Path, lock: dict) -> dict:
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
    with original(directory / "originals/producer.zip", lock["producer"]) as archive:
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
            valid_artifact(pin)
            raw = gh("actions/artifacts/" + str(pin["id"]) + "/zip")
            need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "complete fetched original scientific input " + key)
            (directory / "inputs" / (key + ".zip")).write_bytes(raw)
    return verify_failure_original(directory, lock)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-worker', action='store_true')
    parser.add_argument('directory', type=Path)
    parser.add_argument('lock', type=Path)
    args = parser.parse_args()
    bound = document(args.lock.read_bytes())
    if args.verify_worker:
        with offline_only():
            report = verify_failure_worker(args.directory, bound)
    else:
        report = verify_failure_original(args.directory, bound)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
