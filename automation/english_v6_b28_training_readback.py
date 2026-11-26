"""Complete original B28 TRAIN readback without models or optimizer replay.

The new schema requires seven original observation files and independent
384-return no-event reconstruction. The frozen B27 lock, live-Git and producer
validators retain their code objects in private B28 namespaces; old files and
old module globals are never changed. Full B27 failure and its TRAIN41 proof
remain external originals, not recursively nested copies inside a new ZIP.

PT/ONNX are authenticated as exact original serialized bytes. The stdlib reader
verifies all candidate NPZ geometry and its unchanged initializer frontend,
plus the two saved checkpoint-reload outputs, but does not deserialize PT,
infer ONNX, recompute gradients or claim a new quality/promotion decision.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import FunctionType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _name, _digest in (
    ("english_v6_b27_training_readback.py", "9b952dc1787f5b88afd038b29e881b006fde5a3b64c7829074f765f26b829195"),
    ("english_v6_b24r1_engineering_readback.py", "29277d95565660c52b7a71d2d5013321280b5780aaa774abd4c133430bbcde36"),
):
    if hashlib.sha256((HERE / _name).read_bytes()).hexdigest() != _digest:
        raise ValueError("unchanged original readback helper before import: " + _name)
import english_v6_b27_training_readback as legacy
import english_v6_b24r1_engineering_readback as binary
import english_v6_b28_objective_readback as objective

need, sha, document, canonical = legacy.need, legacy.sha, legacy.document, legacy.canonical
LOCK_SCHEMA = "english-v6-b28-original-training-lock-v1"
REPORT_SCHEMA = "english-v6-b28-original-training-readback-v1"
ARTIFACT_ROLES = legacy.ARTIFACT_ROLES
OBSERVATIONS = legacy.OBSERVATIONS + ("NATURAL_OBJECTIVE_TRACE.json",)
SOURCE_SCHEMA = "english-v6-b28-authenticated-train-source-v1"
_files, sequence = legacy._files, legacy.sequence


def modules():
    # Late imports also keep contract tests usable while the inert registration
    # is being assembled. Both files must be captured explicitly by producer.
    return (importlib.import_module("english_v6_b28_train"),
            importlib.import_module("english_v6_b28_producer"))


def artifact_names(run):
    return dict(models="english-v6-b28-natural-event-models-" + str(run),
        producer="english-v6-b28-train-producer-" + str(run),
        receipts="english-v6-b28-original-input-receipts-" + str(run),
        consumed="english-v6-b28-natural-event-train-consumed")


def _preserved(function, *args):
    train, producer = modules()
    private = dict(function.__globals__, train=train, producer=producer,
        LOCK_SCHEMA=LOCK_SCHEMA, REPORT_SCHEMA=REPORT_SCHEMA,
        artifact_names=artifact_names, validate_lock=validate_lock)
    return FunctionType(function.__code__, private, function.__name__,
                        function.__defaults__, function.__closure__)(*args)


def validate_lock(lock):
    return _preserved(legacy.validate_lock, lock)


def validate_live(live, lock):
    return _preserved(legacy.validate_live, live, lock)


def verify_producer(archive, live, lock, identity):
    manifest = _preserved(legacy.verify_producer, archive, live, lock, identity)
    train, _producer = modules()
    contract = train.fit.contract()
    need(all(manifest.get(key) == value for key, value in contract.items()),
         "original B28 producer binds the fixed objective, independent reference and new fit")
    return manifest


def saved_training_arrays(models, result, source):
    """Keep all original returns and add the independent natural objective."""
    train, _producer = modules()
    pins = result["saved_training_observations"]
    need(type(pins) is dict and set(pins) == set(OBSERVATIONS),
         "all seven complete original B28 observation files, including the new objective")
    data = {}
    for name, pin in pins.items():
        need(type(pin) is dict and set(pin) == {"bytes", "sha256"}
             and type(pin["bytes"]) is int and 0 < pin["bytes"] <= train.capacity.CORE_LIMIT
             and train.failure.valid_hash(pin["sha256"]), "bounded original observation pin " + name)
        raw = models.read("challenger/" + name)
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"],
             "every byte of the original saved training observation " + name)
        data[name] = raw
    draws = sequence(models.read("COMMON_DRAW.json"))
    expected = objective.expected_forward_schedule(draws, source)
    index = document(data["TRAIN_FORWARD_INDEX.json"])
    need(index["count"] == len(index["records"]) == 1844
         and index["pending_unreturned_forward"] is None and index["logits_are_raw"] is True
         and index["arrays_sha256"] == sha(data["TRAIN_FORWARD_LOGITS.npz"]),
         "the complete original optimization NPZ and ledger, without unknown returns")
    payloads = {}
    with train.failure.opened(data["TRAIN_FORWARD_LOGITS.npz"]) as arrays:
        wanted = {"forward_" + str(i).zfill(5) + ".npy" for i in range(1844)}
        need(len(arrays.namelist()) == 1844 and set(arrays.namelist()) == wanted,
             "all and only 1844 unique full-grid original NPY arrays")
        for i, (row, meta) in enumerate(zip(index["records"], expected, strict=True)):
            key = "forward_" + str(i).zfill(5)
            need(set(row) == set(meta) | {"array", "sha256"}
                 and all(row[name] == value for name, value in meta.items()) and row["array"] == key,
                 "same original complete forward schedule before NPY parsing")
            payloads[key] = train.failure.read_npy(arrays.read(key + ".npy"), (meta["frames"], 4))[0]
            need(sha(payloads[key]) == row["sha256"], "original finite array payload SHA256")
    optimizer = document(data["OPTIMIZER_TRACE.json"])
    natural = document(data["NATURAL_OBJECTIVE_TRACE.json"])
    recomputed = objective.verify_natural_objective(index, payloads, natural, optimizer, source, draws)
    positives = sequence(data["POSITIVE_RETENTION_TRACE.json"])
    ordinary = sequence(data["ORDINARY_MARGIN_TRACE.json"])
    need(len(positives) == 692 and len(ordinary) == 768,
         "all unchanged original positive and ordinary margin observations")
    retained = [row for row in expected if row["kind"] == "positive_retention"]
    for row, original in zip(positives, retained, strict=True):
        need(all(row[key] == original[key] for key in ("step", "id", "group", "teacher", "newly_protected"))
             and float(row["gain"]) == original["gain"] and type(row["unweighted_loss"]) in (int, float)
             and math.isfinite(row["unweighted_loss"]) and row["unweighted_loss"] >= 0,
             "all 692 original positive path losses with exact source identities")
    expected_ordinary = [row for row in expected if row["kind"] in ("synthetic_ordinary", "natural_ordinary")]
    groups = {slot[kind]["id"]: slot[kind]["source_group_id"] for slot in draws for kind in ("ordinary", "natural")}
    for row, original in zip(ordinary, expected_ordinary, strict=True):
        need(all(row[key] == original[key] for key in ("step", "id", "gain", "prefix", "frames"))
             and row["group"] == groups[row["id"]]
             and all(type(row[key]) in (int, float) and math.isfinite(row[key])
                     for key in ("mean", "maximum", "active_fraction"))
             and type(row["active_cells"]) is int and type(row["cells"]) is int
             and 0 <= row["active_cells"] <= row["cells"] and row["cells"] == row["frames"] * 3,
             "all 768 complete original ordinary-margin observations")
    with train.failure.opened(data["CHECKPOINT_RELOAD_LOGITS.npz"]) as reload:
        need(len(reload.namelist()) == 2 and set(reload.namelist()) == {"saved.npy", "reloaded.npy"},
             "only the two already-budgeted original checkpoint reload outputs")
        shape = (index["records"][0]["frames"], 4)
        saved = train.failure.read_npy(reload.read("saved.npy"), shape)[0]
        rebuilt = train.failure.read_npy(reload.read("reloaded.npy"), shape)[0]
        need(saved == rebuilt, "identical original before/after-checkpoint-reload outputs")
    return dict(optimization_arrays=1844, checkpoint_reload_arrays=2, optimizer_trace_entries=192,
        positive_trace_entries=692, ordinary_trace_entries=768,
        verified_optimization_frames=recomputed["verified_optimization_frames"],
        exported_onnx_forward_count_from_original_producer=1,
        original_observation_files=pins, natural_objective=recomputed,
        natural_objective_returns_verified=384, natural_no_event_terms_recomputed=True,
        gradients_recomputed=False, optimizer_updates_replayed=False)


def _npy_shape(raw):
    need(raw[:6] == b"\x93NUMPY" and tuple(raw[6:8]) in ((1, 0), (2, 0)), "original NPY tensor header")
    width = 2 if raw[6] == 1 else 4
    size = int.from_bytes(raw[8:8 + width], "little")
    need(0 < size <= 8192 and len(raw) >= 8 + width + size, "bounded original tensor header")
    header = ast.literal_eval(raw[8 + width:8 + width + size].decode("latin1").strip())
    need(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}
         and header["descr"] == "<f4" and header["fortran_order"] is False
         and type(header["shape"]) is tuple and len(header["shape"]) in (1, 2, 3)
         and all(type(v) is int and 0 < v <= 1024 for v in header["shape"]),
         "bounded unchanged FP32 tensor shape")
    return header["shape"]


def candidate_npz(candidate, initializer):
    """Compare saved NPZ geometry/frontend; do not deserialize PT or infer."""
    train, _producer = modules()
    frontend = {"mel_filters.npy", "train_mean.npy", "train_scale.npy"}
    with train.failure.opened(candidate) as learned, train.failure.opened(initializer) as parent:
        names = parent.namelist()
        need(len(names) == len(set(names)) and set(learned.namelist()) == set(names)
             and len(learned.namelist()) == len(names) and frontend <= set(names)
             and all(name.endswith(".npy") and "/" not in name for name in names),
             "all and only the original initializer NPZ tensor/frontend names")
        parameters, shapes = 0, {}
        for name in sorted(names):
            before = parent.read(name)
            shape = _npy_shape(before)
            original_body = binary.npy_body(before, shape)
            body = binary.npy_body(learned.read(name), shape)
            shapes[name[:-4]] = list(shape)
            if name in frontend:
                need(body == original_body, "unchanged original initializer frontend " + name)
            else:
                parameters += math.prod(shape)
        need(parameters == 43876, "all 43876 finite candidate NPZ parameters")
    return dict(parameter_count=43876, shapes=shapes, complete_npz_geometry_verified=True,
        initializer_frontend_bytes_unchanged=True, serialized_pt_tensors_deserialized=False,
        pt_npz_elementwise_equivalence_independently_recomputed=False)


def validate_failure_certification(*documents):
    train, _producer = modules()
    expected = train.failure_certification()
    need(bool(documents) and all(value.get("original_B27_failure_certification") == expected for value in documents),
         "same exact reviewed B27 import-root adapter and complete original report binding")
    return expected


def validate_result(result, source, identity, training):
    train, _producer = modules()
    validate_failure_certification(result, source)
    contract = train.fit.contract()
    need(result["schema"] == train.SCHEMA and result["status"] == train.RESULT_STATUS
         and result["identity"] == identity and result["recipe_sha256"] == result["design_sha256"] == train.DESIGN_SHA
         and result["model_pt_sha256"] == training["model_pt_sha256"] and result["natural_weight"] == 0.16
         and result["assessment_sha256"] == train.assessment_binding()["assessment"]["sha256"]
         and result["actual_optimizer_updates"] == 192 and result["actual_new_model_forwards"] == 1847
         and result["reused_ordinary_teacher_arrays"] == 264 and result["original_retention_conditions"] == 120
         and result["union_retention_conditions"] == 122 and result["prefix_blank_reduction"] == "amax"
         and result["phase_initialization_pt_sha256"] == train.INITIAL_PT
         and result["optimizer_initialization"] == "fresh_AdamW"
         and result["original_B27_NO_PROMOTION_reauthenticated"] is True
         and result["natural_objective_returns"] == 384
         and result["natural_objective_schema"] == objective.OBJECTIVE_SCHEMA
         and all(result[key] is False for key in ("restored_optimizer_state", "exact_optimizer_continuation",
             "historical_numerical_replay_claimed", "same_job_matched_control", "promotion_allowed",
             "physical_ssc305", "B27_metrics_used_to_choose_parameters", "source_execution_allowed",
             "new_primary_allowed", "replication_allowed", "candidate_engineering_allowed", "product_released"))
         and all(type(result[key]) is int and result[key] == 0 for key in
             ("old_B23_optimizer_updates_reexecuted", "old_B24_optimizer_updates_reexecuted",
              "new_control_optimizer_updates", "new_control_forwards", "new_teacher_forwards",
              "dev_cal_locked_forwards", "new_tts_asr_calls"))
         and result["field_far_per_hour"] is None and result["architecture"] == train.b24.baseline.architecture_contract()
         and all(result.get(key) == source.get(key) == value for key, value in contract.items()),
         "actual bounded B28 natural-event phase with unchanged other science and no quality claim")


def _verify_worker(directory, lock, inputs, proof):
    train, _producer = modules()
    directory, inputs = Path(directory), Path(inputs)
    train.design(); train.assessment_binding(); train.opening()
    need({p.name for p in directory.iterdir()} == {"TRAIN_LIVE.json"} |
         {"TRAIN_" + role + ".zip" for role in ARTIFACT_ROLES}
         and all(p.is_file() and not p.is_symlink() for p in directory.iterdir()),
         "exact complete original TRAIN directory without supplementary files")
    live = document((directory / "TRAIN_LIVE.json").read_bytes())
    identity = validate_live(live, lock)
    t = lock["training"]
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix="b28-original-training-") as temporary:
        archives = {}
        for role, pin in t["artifacts"].items():
            raw = (directory / ("TRAIN_" + role + ".zip")).read_bytes()
            need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "complete original B28 artifact " + role)
            archives[role] = stack.enter_context(train.failure.opened(raw))
        verify_producer(archives["producer"], live, lock, identity)
        models, receipts, consumed = archives["models"], archives["receipts"], archives["consumed"]
        expected_receipts = {"FETCH.json", "SOURCE_PREFLIGHT.json", "CONSUMPTION.json", "EXTERNAL_INPUTS.json",
                             "original_b24/TRAIN_LIVE.json"}
        expected_receipts |= {"original_b24/TRAIN_" + role + ".zip" for role in ARTIFACT_ROLES}
        need(_files(receipts) == expected_receipts,
             "nine complete B28 receipt members; original B27/41+18 proofs stay external")
        retained = Path(temporary)
        receipts.extractall(retained)
        train.retain_original_b24(inputs)
        for name in train.OLD_B24_FILES:
            need(receipts.read("original_b24/" + name) == (inputs / "original_b24" / name).read_bytes(),
                 "same entire B24 original inside B27 proof, external input and new receipt " + name)
        old_result = train.old_proof.verify(retained / "original_b24")
        original_control = train.historical_control(retained / "original_b24")
        external = train.external_inputs(inputs)
        need(receipts.read("EXTERNAL_INPUTS.json") == train.encoded(external),
             "complete unchanged external proof and TRAIN input byte inventory")
        raw = models.read("RESULT.json")
        result, source = document(raw), document(models.read("SOURCE.json"))
        need(sha(raw) == t["result_sha256"], "exact final original RESULT bytes")
        validate_result(result, source, identity, t)
        for field, name in (("source_sha256", "SOURCE.json"), ("common_draw_sha256", "COMMON_DRAW.json"),
                ("union_evidence_sha256", "UNION_EVIDENCE.json"), ("old120_references_sha256", "OLD120_POSITIVE_REFERENCES.json"),
                ("opening_sha256", "OPENING.json"), ("assessment_binding_sha256", "ASSESSMENT_BINDING.json")):
            need(sha(models.read(name)) == result[field], "original result dependency " + name)
        for name, path in (("DESIGN.json", train.DESIGN), ("OPENING.json", train.OPENING), ("ASSESSMENT_BINDING.json", train.BINDING)):
            need(models.read(name) == path.read_bytes(), "same frozen before-training " + name)
        need(sha(models.read("COMMON_DRAW.json")) == train.design()["common_draw_sha256"] == objective.COMMON_DRAW_SHA,
             "same original 192 common draws")
        need(_files(consumed) == {"CONSUMPTION.json"}
             and consumed.read("CONSUMPTION.json") == receipts.read("CONSUMPTION.json")
             and receipts.read("SOURCE_PREFLIGHT.json") == models.read("SOURCE.json"),
             "same original durable one-shot marker and complete before-fit source")
        marker, fetched = document(consumed.read("CONSUMPTION.json")), document(receipts.read("FETCH.json"))
        validate_failure_certification(fetched, source, result)
        need(marker["schema"] == train.SCHEMA and marker["status"] == "TRAINING_BUDGET_CONSUMED_UNKNOWN"
             and marker["identity"] == fetched["identity"] == identity and marker["budget"] == train.design()["budget"]
             and marker["promotion_allowed"] is False and marker["source_preflight_sha256"] == result["source_sha256"]
             and all(marker[key] == fetched[key] == result[key]
                     for key in ("recipe_sha256", "assessment_binding_sha256", "opening_sha256"))
             and fetched["previous_failure"] == source["original_B27_failed_primary"] == proof
             and fetched["external_inputs_sha256"] == sha(receipts.read("EXTERNAL_INPUTS.json"))
             and source["complete_external_inputs"] == external and source["original_B24_train"] == old_result
             and source["historical_control"] == original_control,
             "same complete actual B27 failure, original inputs and durable one-shot consumption")
        need(source["receipt_capacity"] == fetched["receipt_capacity"] == train.receipt_capacity(inputs)
             and sum(len(receipts.read(name)) for name in
                     ("FETCH.json", "SOURCE_PREFLIGHT.json", "CONSUMPTION.json", "EXTERNAL_INPUTS.json"))
                 <= train.design()["archive_caps"]["new_receipt_metadata_reserve"],
             "all original retained bytes and new receipts fit the prospectively fixed envelope")
        history = train.source_history(inputs)
        need(source["original_B27_source_exclusion"] == fetched["original_B27_source_exclusion"] == history
             and all(fetched.get(key) == value for key, value in train.fit.contract().items())
             and type(fetched["new_model_forwards"]) is int and fetched["new_model_forwards"] == 0,
             "complete original catalog and B27 exclusions independently rebuilt from external originals")
        need(source["prospective_assessment_sha256"] == result["assessment_sha256"] and source["schema"] == SOURCE_SCHEMA
             and source["initialization_pt_sha256"] == train.INITIAL_PT and source["natural_weight"] == 0.16
             and source["all_original_train_sources_and_teachers_unchanged"] is True
             and source["B27_metrics_used_to_choose_parameters"] is False
             and all(type(source[key]) is int and source[key] == 0 for key in
                     ("dev_rows_used_for_training", "new_model_forwards", "new_source_calls", "new_dev_model_forwards")),
             "new source authorization adds no TRAIN/DEV source or model call")
        with train.original_models(retained / "original_b24") as (old_models, _old):
            original_source = document(old_models.read("SOURCE.json"))
            for key in ("base_train", "natural_train", "train_originals", "positive_retention", "retention_additions",
                        "union", "input_archives", "registered_positive_retention_forwards", "registered_base_ctc_forwards"):
                need(source[key] == original_source[key], "all original TRAIN/source/protection identities " + key)
            need(models.read("OLD120_POSITIVE_REFERENCES.json") == old_models.read("OLD120_POSITIVE_REFERENCES.json")
                 and models.read("UNION_EVIDENCE.json") == old_models.read("UNION_EVIDENCE.json"),
                 "the original 120 references and same two additions")
            npz_report = candidate_npz(models.read("challenger/models/" + train.STEM + ".npz"),
                                      old_models.read("control/models/mero43-b23-challenger192.npz"))
        need(len(source["base_train"]) == 79 and len(source["natural_train"]) == 9 and len(source["train_originals"]) == 88,
             "all 88 original training sources for later overlap certification")
        expected_members = {"RESULT.json", "STARTED.json", "SOURCE.json", "COMMON_DRAW.json", "DESIGN.json",
            "OPENING.json", "ASSESSMENT_BINDING.json", "UNION_EVIDENCE.json", "OLD120_POSITIVE_REFERENCES.json"}
        expected_members |= {"challenger/" + name for name in OBSERVATIONS}
        for role, stem in (("control", "mero43-b24-challenger192"), ("challenger", train.STEM)):
            need(set(result[role]["files"]) == {stem + suffix for suffix in (".pt", ".npz", ".onnx")},
                 "all three exact original export formats " + role)
            for name, pin in result[role]["files"].items():
                member = role + "/models/" + name
                expected_members.add(member)
                body = models.read(member)
                need(len(body) == pin["bytes"] and sha(body) == pin["sha256"], "complete unchanged original export " + member)
            need(result[role]["files"][stem + ".pt"]["sha256"] == t["model_pt_sha256"][role], "exact PT role " + role)
        need(_files(models) == expected_members and result["control"]["files"] == original_control["files"]
             and result["control"]["output_stem"] == "control/models/mero43-b24-challenger192"
             and result["challenger"]["stem"] == "challenger/models/" + train.STEM
             and result["challenger"]["parent_pt_sha256"] == train.INITIAL_PT
             and result["challenger"]["historical_control_pt_sha256"] == train.CONTROL_PT
             and result["challenger"]["natural_weight"] == 0.16
             and result["challenger"]["optimization_forwards"] == 1844
             and result["challenger"]["checkpoint_reload_forwards"] == 2
             and result["challenger"]["onnx_export_forwards"] == 1
             and all(result["challenger"].get(key) == value for key, value in train.fit.contract().items()),
             "one complete B28 final candidate and immutable historical B24 control, with no checkpoint selection")
        need(document(models.read("STARTED.json")) == dict(schema=train.SCHEMA, identity=identity,
             budget=train.design()["budget"], recipe_sha256=train.DESIGN_SHA), "exact original before-fit STARTED marker")
        observed = saved_training_arrays(models, result, source)
    return dict(schema=REPORT_SCHEMA, result=result, training_source=source,
        original_lock_sha256=sha(canonical(lock)), result_sha256=t["result_sha256"], source_sha256=result["source_sha256"],
        producer_manifest_sha256=lock["training_producer_manifest_sha256"], saved_training_observations=observed,
        candidate_parameter_archive=npz_report, all_original_producer_git_verified=True,
        all_original_inputs_verified=True, all_saved_training_arrays_verified=True,
        natural_no_event_terms_recomputed=True, complete_original_B27_NO_PROMOTION_verified=True,
        original_B24_control_unchanged=True, gradients_recomputed=False, optimizer_updates_replayed=False,
        new_model_calls=0, new_source_calls=0, new_optimizer_updates=0, promotion_allowed=False, product_released=False)


def validate_report(report, lock):
    need(report["schema"] == REPORT_SCHEMA and report["original_lock_sha256"] == sha(canonical(lock))
         and report["result_sha256"] == lock["training"]["result_sha256"]
         and all(report[key] is True for key in ("all_original_producer_git_verified", "all_original_inputs_verified",
            "all_saved_training_arrays_verified", "natural_no_event_terms_recomputed",
            "complete_original_B27_NO_PROMOTION_verified", "original_B24_control_unchanged"))
         and report["saved_training_observations"]["natural_objective_returns_verified"] == 384
         and all(type(report[key]) is int and report[key] == 0 for key in
                 ("new_model_calls", "new_source_calls", "new_optimizer_updates"))
         and all(report[key] is False for key in ("promotion_allowed", "product_released",
                                               "gradients_recomputed", "optimizer_updates_replayed")),
         "complete isolated original B28 evidence and new objective without execution or quality claims")
    return report


def verify_training_original(directory, lock, inputs):
    validate_lock(lock)
    need(not {"numpy", "torch", "onnxruntime"} & set(sys.modules), "independent TRAIN readback before tensor imports")
    with tempfile.TemporaryDirectory(prefix="b28-train-lock-") as temporary:
        pin = Path(temporary) / "lock.json"
        pin.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
            "--verify-worker", str(Path(directory).resolve()), str(pin), str(Path(inputs).resolve())])
    return validate_report(document(raw), lock)


def fetch_training_original(directory, lock, inputs):
    train, _producer = modules()
    # Evidence acquisition shares the inert preparation boundary: a plausible
    # future lock must not reach GitHub before the reviewed opening is bound.
    train.design()
    train.opening()
    train.assessment_binding()
    t = validate_lock(lock)
    directory = Path(directory)
    need(directory.is_absolute() and directory == directory.resolve() and directory.parent.is_dir()
         and not directory.exists(), "fresh complete immutable B28 TRAIN original acquisition")
    commit = document(train.gh("commits/" + t["run_head"]))
    live = dict(run=document(train.gh("actions/runs/" + str(t["run_id"]))), commit=commit,
        artifacts=document(train.gh("actions/runs/" + str(t["run_id"]) + "/artifacts?per_page=100")),
        tree=document(train.gh("git/trees/" + commit["commit"]["tree"]["sha"] + "?recursive=1")))
    validate_live(live, lock)
    directory.mkdir()
    train.write(directory / "TRAIN_LIVE.json", live)
    for role, pin in t["artifacts"].items():
        raw = train.gh("actions/artifacts/" + str(pin["id"]) + "/zip")
        need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "complete exact original fetched artifact " + role)
        (directory / ("TRAIN_" + role + ".zip")).write_bytes(raw)
    return verify_training_original(directory, lock, inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-worker", action="store_true")
    parser.add_argument("directory", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    bound = document(args.lock.read_bytes())
    if args.verify_worker:
        train, _producer = modules()
        validate_lock(bound)
        proof = train.previous_failure(args.inputs)
        # Full frozen B27 recursive readback completes before this stdlib-only
        # phase. From here no network, scientific imports or subprocess remain.
        with train.failure.offline_only():
            report = _verify_worker(args.directory, bound, args.inputs, proof)
    else:
        report = verify_training_original(args.directory, bound, args.inputs)
    sys.stdout.buffer.write(canonical(report))


if __name__ == "__main__":
    main()
