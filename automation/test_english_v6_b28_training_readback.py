"""Fictional complete-array and original-provenance contracts for B28."""
from __future__ import annotations

import copy
import io
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_training_readback as reader
import test_english_v6_b28_objective_readback as objective_tests


def fake_train():
    values = dict(vars(reader.legacy.train))
    values.update(BRANCH="refs/heads/research/en-v6-b28-natural-event-train-once-20261125",
        WORKFLOW=".github/workflows/english-v6-b28-natural-event-train-once.yml",
        MARKER="english-v6-b28-natural-event-train-consumed", STEM="mero43-b28-challenger192",
        SCHEMA="english-v6-b28-train-natural-event-v1")
    return SimpleNamespace(**values)


def fictional_failure_certification():
    return dict(schema="english-v6-b28-original-b27-failure-certification-binding-v1",
        original_reader_path="automation/english_v6_b27_quality_readback.py", original_reader_sha256="1" * 64,
        original_report_sha256="2" * 64, adapter_path="automation/fictional_import_adapter.py",
        adapter_sha256="3" * 64, adapter_report_schema="fictional-original-import-certification-v1",
        adapter_report_sha256="4" * 64, original_standalone_continuation_succeeded=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0)


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return stream.getvalue()


def npy(shape, payload, dtype="<f4", fortran=False):
    header = repr(dict(descr=dtype, fortran_order=fortran, shape=shape)).encode()
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + payload


def repin(members, result):
    result["saved_training_observations"] = {name: dict(bytes=len(members["challenger/" + name]),
        sha256=reader.sha(members["challenger/" + name])) for name in reader.OBSERVATIONS}


def fixture():
    (index, payloads, natural, optimizer, source, draws), draw_hash = objective_tests.fixtures()
    arrays = {row["array"] + ".npy": npy((row["frames"], 4), payloads[row["array"]]) for row in index["records"]}
    archive = zip_bytes(arrays)
    index["arrays_sha256"] = reader.sha(archive)
    positives, ordinary = [], []
    for row in index["records"]:
        if row["kind"] == "positive_retention":
            positives.append(dict(**{key: row[key] for key in ("step", "id", "group", "teacher", "newly_protected")},
                gain=str(row["gain"]), unweighted_loss=0.0))
        if row["kind"] in ("natural_ordinary", "synthetic_ordinary"):
            original = draws[row["step"] - 1]["natural" if row["kind"] == "natural_ordinary" else "ordinary"]
            ordinary.append(dict(**{key: row[key] for key in ("step", "id", "gain", "prefix", "frames")},
                group=original["source_group_id"], mean=0.0, maximum=0.0, active_fraction=0.0,
                active_cells=0, cells=row["frames"] * 3))
    enc = reader.objective.encoded
    reload_raw = arrays["forward_00000.npy"]
    members = {"COMMON_DRAW.json": enc(draws),
        "challenger/TRAIN_FORWARD_INDEX.json": enc(index),
        "challenger/TRAIN_FORWARD_LOGITS.npz": archive,
        "challenger/OPTIMIZER_TRACE.json": enc(optimizer),
        "challenger/NATURAL_OBJECTIVE_TRACE.json": enc(natural),
        "challenger/POSITIVE_RETENTION_TRACE.json": enc(positives),
        "challenger/ORDINARY_MARGIN_TRACE.json": enc(ordinary),
        "challenger/CHECKPOINT_RELOAD_LOGITS.npz": zip_bytes({"saved.npy": reload_raw, "reloaded.npy": reload_raw})}
    result = {}
    repin(members, result)
    return members, result, source, draw_hash


def lock():
    train = fake_train()
    run = 123
    training = dict(run_id=run, run_head="b" * 40, prepared_parent="a" * 40,
        run_branch=train.BRANCH.removeprefix("refs/heads/"), runtime_workflow=train.WORKFLOW,
        result_sha256="c" * 64,
        model_pt_sha256=dict(B0=train.b24.old.B0_PT, B9=train.b24.old.b10.b9.MODELS["B9_control"]["pt"],
            control=train.CONTROL_PT, challenger="d" * 64),
        artifacts={role: dict(id=i + 1, bytes=100, sha256=str(i + 1) * 64, name=reader.artifact_names(run)[role])
                   for i, role in enumerate(reader.ARTIFACT_ROLES)})
    return dict(schema=reader.LOCK_SCHEMA, training=training, training_producer_manifest_sha256="e" * 64)


def live_fixture(bound):
    train = fake_train()
    t = bound["training"]
    return dict(run=dict(id=t["run_id"], head_sha=t["run_head"], run_attempt=1,
        event="push", head_branch=t["run_branch"], path=train.WORKFLOW,
        status="completed", conclusion="success", repository=dict(full_name=train.REPO, private=False),
        head_repository=dict(full_name=train.REPO, private=False)),
        commit=dict(sha=t["run_head"], parents=[dict(sha=t["prepared_parent"])],
                    files=[dict(filename=train.WORKFLOW)], commit=dict(tree=dict(sha="f" * 40))),
        tree=dict(sha="f" * 40, truncated=False),
        artifacts=dict(total_count=4, artifacts=[dict(id=pin["id"], name=pin["name"], expired=False,
            size_in_bytes=pin["bytes"], digest="sha256:" + pin["sha256"],
            workflow_run=dict(id=t["run_id"], head_sha=t["run_head"])) for pin in t["artifacts"].values()]))


class B28SavedObservationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()

    def verify(self, members=None, result=None, source=None):
        if members is None:
            members, result, source, _hash = self.original
        with patch.object(reader, "modules", return_value=(fake_train(), SimpleNamespace())), \
             patch.object(reader.objective, "COMMON_DRAW_SHA", self.original[3]), \
             zipfile.ZipFile(io.BytesIO(zip_bytes(members))) as archive:
            return reader.saved_training_arrays(archive, result, source)

    def change_json(self, path, alter):
        members, result, source, _hash = copy.deepcopy(self.original)
        raw = members["challenger/" + path]
        value = reader.document(raw) if raw.startswith(b"{") else reader.sequence(raw)
        alter(value)
        members["challenger/" + path] = reader.objective.encoded(value)
        repin(members, result)
        return members, result, source

    def test_all_seven_files_and_original192_updates_with_new384_recomputation(self):
        with patch.object(reader.objective, "verify_natural_objective", wraps=reader.objective.verify_natural_objective) as verify:
            report = self.verify()
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(report["optimization_arrays"], 1844)
        self.assertEqual(report["checkpoint_reload_arrays"], 2)
        self.assertEqual(report["natural_objective_returns_verified"], 384)
        self.assertEqual(report["positive_trace_entries"], 692)
        self.assertEqual(report["ordinary_trace_entries"], 768)
        self.assertFalse(report["gradients_recomputed"])
        self.assertEqual(set(report["original_observation_files"]), set(reader.OBSERVATIONS))

    def test_six_file_legacy_observations_or_hash_only_substitution_cannot_pass(self):
        members, result, source, _hash = copy.deepcopy(self.original)
        result["saved_training_observations"].pop("NATURAL_OBJECTIVE_TRACE.json")
        with self.assertRaisesRegex(ValueError, "seven"):
            self.verify(members, result, source)
        members, result, source, _hash = copy.deepcopy(self.original)
        members["challenger/NATURAL_OBJECTIVE_TRACE.json"] += b" "
        with self.assertRaisesRegex(ValueError, "every byte"):
            self.verify(members, result, source)

    def test_raw_dtype_and_missing_array_fail_even_with_new_container_hash(self):
        for action in ("dtype", "missing"):
            members, result, source, _hash = copy.deepcopy(self.original)
            with zipfile.ZipFile(io.BytesIO(members["challenger/TRAIN_FORWARD_LOGITS.npz"])) as z:
                arrays = {name: z.read(name) for name in z.namelist()}
            if action == "missing":
                arrays.pop("forward_00004.npy")
            else:
                arrays["forward_00004.npy"] = arrays["forward_00004.npy"].replace(b"<f4", b"<f8")
            members["challenger/TRAIN_FORWARD_LOGITS.npz"] = zip_bytes(arrays)
            index = reader.document(members["challenger/TRAIN_FORWARD_INDEX.json"])
            index["arrays_sha256"] = reader.sha(members["challenger/TRAIN_FORWARD_LOGITS.npz"])
            members["challenger/TRAIN_FORWARD_INDEX.json"] = reader.objective.encoded(index)
            repin(members, result)
            with self.assertRaises(ValueError):
                self.verify(members, result, source)

    def test_new_objective_is_verified_not_renamed_legacy_arithmetic(self):
        def inject(value):
            value["records"][0]["natural_no_event"] += 0.1
            value["records"][0]["loss"] += 0.1
        with self.assertRaisesRegex(ValueError, "per-step natural"):
            self.verify(*self.change_json("OPTIMIZER_TRACE.json", inject))
        with self.assertRaisesRegex(ValueError, "per-gain"):
            self.verify(*self.change_json("NATURAL_OBJECTIVE_TRACE.json", lambda value: value["records"][0].update(weighted_loss=100.0)))

    def test_all_retention_and_margin_rows_remain_mandatory(self):
        for path, change in (("POSITIVE_RETENTION_TRACE.json", lambda rows: rows[0].update(id="different")),
                             ("ORDINARY_MARGIN_TRACE.json", lambda rows: rows[0].update(group="different")),
                             ("ORDINARY_MARGIN_TRACE.json", lambda rows: rows.pop())):
            with self.assertRaises(ValueError):
                self.verify(*self.change_json(path, change))

    def test_original_reload_arrays_must_be_equal_and_complete(self):
        members, result, source, _hash = copy.deepcopy(self.original)
        with zipfile.ZipFile(io.BytesIO(members["challenger/CHECKPOINT_RELOAD_LOGITS.npz"])) as z:
            old = z.read("saved.npy")
        for reload_files in ({"saved.npy": old}, {"saved.npy": old, "reloaded.npy": old[:-4] + struct.pack("<f", 2.0)}):
            members["challenger/CHECKPOINT_RELOAD_LOGITS.npz"] = zip_bytes(reload_files)
            repin(members, result)
            with self.assertRaises(ValueError):
                self.verify(members, result, source)


class B28ProvenanceContracts(unittest.TestCase):
    def setUp(self):
        self.context = patch.object(reader, "modules", return_value=(fake_train(), SimpleNamespace()))
        self.context.start()
        self.addCleanup(self.context.stop)

    def test_new_lock_requires_distinct_actual_originals_and_b28_identity(self):
        bound = lock()
        self.assertEqual(reader.validate_lock(bound), bound["training"])
        for alter in (lambda value: value.update(schema=reader.legacy.LOCK_SCHEMA),
                      lambda value: value["training"].update(run_branch="research/old-b27"),
                      lambda value: value["training"]["artifacts"]["models"].update(name="old-models"),
                      lambda value: value["training"]["artifacts"]["consumed"].update(id=1),
                      lambda value: value["training"]["model_pt_sha256"].update(challenger=fake_train().INITIAL_PT)):
            changed = copy.deepcopy(bound)
            alter(changed)
            with self.assertRaises(ValueError):
                reader.validate_lock(changed)

    def test_unchanged_live_provenance_guards_reject_retries_private_and_extra_changes(self):
        bound = lock()
        original = live_fixture(bound)
        self.assertEqual(reader.validate_live(original, bound)["run_id"], "123")
        for alter in (lambda x: x["run"].update(run_attempt=2),
                      lambda x: x["run"]["repository"].update(private=True),
                      lambda x: x["tree"].update(truncated=True),
                      lambda x: x["commit"]["files"].append(dict(filename="science.py")),
                      lambda x: x["artifacts"]["artifacts"][0].update(digest="sha256:" + "0" * 64),
                      lambda x: x["artifacts"].update(total_count=3)):
            changed = copy.deepcopy(original)
            alter(changed)
            with self.assertRaises(ValueError):
                reader.validate_live(changed, bound)

    def test_private_rebinding_does_not_modify_old_reader_or_old_train(self):
        before = (reader.legacy.LOCK_SCHEMA, reader.legacy.REPORT_SCHEMA,
                  reader.legacy.train.BRANCH, reader.legacy.artifact_names(123))
        reader.validate_lock(lock())
        after = (reader.legacy.LOCK_SCHEMA, reader.legacy.REPORT_SCHEMA,
                 reader.legacy.train.BRANCH, reader.legacy.artifact_names(123))
        self.assertEqual(before, after)

    def test_actual_producer_checks_every_git_blob_and_new_objective_binding(self):
        train = fake_train()
        contract = dict(objective_schema=reader.objective.OBJECTIVE_SCHEMA,
            objective_sha256=reader.objective.OBJECTIVE_SHA, reference_sha256=reader.objective.REFERENCE_SHA,
            fit_adapter_sha256="f" * 64, common_draw_sha256=reader.objective.COMMON_DRAW_SHA)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train.ROOT = root
            train.DESIGN_SHA = "d" * 64
            train.BINDING = root / "binding.json"; train.BINDING.write_bytes(b"fictional binding")
            train.OPENING = root / "opening.json"; train.OPENING.write_bytes(b"fictional opening")
            train.fit = SimpleNamespace(contract=lambda: contract)
            payloads = {"automation/fictional.py": b"fictional captured scientific bytes\n",
                        train.WORKFLOW: b"fictional workflow activated from its exact parent\n"}
            for name, raw in payloads.items():
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            def capacity(files):
                return dict(bytes=sum(pin["bytes"] for pin in files.values()))
            producer = SimpleNamespace(SCHEMA="english-v6-b28-complete-git-producer-v1",
                paths=lambda: set(payloads), workflow=lambda parent: payloads[train.WORKFLOW], capacity_receipt=capacity)
            files = {name: dict(bytes=len(raw), sha256=reader.sha(raw), mode="100644",
                git_blob_sha1=reader.binary.blob(raw)) for name, raw in payloads.items()}
            identity = dict(run_id="123", prepared_parent="b" * 40)
            manifest = dict(schema=producer.SCHEMA, identity=identity, git_tree="a" * 40,
                runtime_workflow=train.WORKFLOW, original_b24_dependency_files=338,
                source_code_reads_only=True, include_hidden_files=True, recipe_sha256=train.DESIGN_SHA,
                assessment_binding_sha256=reader.sha(train.BINDING.read_bytes()),
                opening_sha256=reader.sha(train.OPENING.read_bytes()), new_model_forwards=0,
                optimizer_updates=0, new_source_calls=0, files=files, capacity=capacity(files), **contract)
            members = {"repository/" + name: raw for name, raw in payloads.items()}
            members["PRODUCER.json"] = reader.objective.encoded(manifest)
            bound = dict(training_producer_manifest_sha256=reader.sha(members["PRODUCER.json"]))
            live = dict(tree=dict(sha="a" * 40, tree=[dict(path=name, sha=pin["git_blob_sha1"],
                        type="blob", mode=pin["mode"]) for name, pin in files.items()]))
            with patch.object(reader, "modules", return_value=(train, producer)):
                with zipfile.ZipFile(io.BytesIO(zip_bytes(members))) as archive:
                    self.assertEqual(reader.verify_producer(archive, live, bound, identity), manifest)
                changed = copy.deepcopy(live); changed["tree"]["tree"][0]["sha"] = "0" * 40
                with zipfile.ZipFile(io.BytesIO(zip_bytes(members))) as archive, self.assertRaises(ValueError):
                    reader.verify_producer(archive, changed, bound, identity)
                bad = dict(manifest, objective_sha256="0" * 64)
                members["PRODUCER.json"] = reader.objective.encoded(bad)
                bound["training_producer_manifest_sha256"] = reader.sha(members["PRODUCER.json"])
                with zipfile.ZipFile(io.BytesIO(zip_bytes(members))) as archive, self.assertRaisesRegex(ValueError, "fixed objective"):
                    reader.verify_producer(archive, live, bound, identity)

    def test_result_retains_budget_and_does_not_open_downstream_or_legacy_objective(self):
        train = fake_train()
        train.DESIGN_SHA = "e" * 64
        train.RESULT_STATUS = "TRAINED_NATURAL_EVENT_CANDIDATE_REQUIRES_NEW_INDEPENDENT_PRIMARY"
        contract = dict(objective_schema=reader.objective.OBJECTIVE_SCHEMA,
            objective_sha256=reader.objective.OBJECTIVE_SHA, reference_sha256=reader.objective.REFERENCE_SHA,
            fit_adapter_sha256="f" * 64, common_draw_sha256=reader.objective.COMMON_DRAW_SHA)
        train.fit = SimpleNamespace(contract=lambda: contract)
        train.failure_certification = fictional_failure_certification
        train.assessment_binding = lambda: dict(assessment=dict(sha256="a" * 64))
        bound = lock()["training"]
        identity = dict(run_id="123")
        result = dict(schema=train.SCHEMA, status=train.RESULT_STATUS, identity=identity,
            recipe_sha256=train.DESIGN_SHA, design_sha256=train.DESIGN_SHA,
            model_pt_sha256=bound["model_pt_sha256"], natural_weight=0.16,
            assessment_sha256="a" * 64, actual_optimizer_updates=192, actual_new_model_forwards=1847,
            reused_ordinary_teacher_arrays=264, original_retention_conditions=120, union_retention_conditions=122,
            prefix_blank_reduction="amax", phase_initialization_pt_sha256=train.INITIAL_PT,
            optimizer_initialization="fresh_AdamW", original_B27_NO_PROMOTION_reauthenticated=True,
            natural_objective_returns=384, natural_objective_schema=reader.objective.OBJECTIVE_SCHEMA,
            field_far_per_hour=None, architecture=train.b24.baseline.architecture_contract(),
            original_B27_failure_certification=fictional_failure_certification(), **contract)
        source = dict(contract, original_B27_failure_certification=fictional_failure_certification())
        result.update({name: False for name in ("restored_optimizer_state", "exact_optimizer_continuation",
            "historical_numerical_replay_claimed", "same_job_matched_control", "promotion_allowed", "physical_ssc305",
            "B27_metrics_used_to_choose_parameters", "source_execution_allowed", "new_primary_allowed",
            "replication_allowed", "candidate_engineering_allowed", "product_released")})
        result.update({name: 0 for name in ("old_B23_optimizer_updates_reexecuted", "old_B24_optimizer_updates_reexecuted",
            "new_control_optimizer_updates", "new_control_forwards", "new_teacher_forwards", "dev_cal_locked_forwards", "new_tts_asr_calls")})
        with patch.object(reader, "modules", return_value=(train, SimpleNamespace())):
            reader.validate_result(result, source, identity, bound)
            for key, value in (("actual_new_model_forwards", 1848), ("natural_objective_returns", 383),
                               ("phase_initialization_pt_sha256", train.CONTROL_PT),
                               ("new_primary_allowed", True), ("B27_metrics_used_to_choose_parameters", True),
                               ("objective_sha256", "0" * 64), ("natural_objective_schema", "old-empty-ctc")):
                with self.assertRaises(ValueError):
                    reader.validate_result(dict(result, **{key: value}), source, identity, bound)

    def test_all_source_fetch_result_adapter_bindings_are_exact(self):
        train = fake_train()
        train.failure_certification = fictional_failure_certification
        documents = [dict(original_B27_failure_certification=fictional_failure_certification()) for _ in range(3)]
        with patch.object(reader, "modules", return_value=(train, SimpleNamespace())):
            self.assertEqual(reader.validate_failure_certification(*documents), fictional_failure_certification())
            for index in range(3):
                for field, wrong in (("adapter_sha256", "0" * 64), ("adapter_report_sha256", "0" * 64),
                    ("original_report_sha256", "0" * 64), ("original_standalone_continuation_succeeded", True),
                    ("new_model_calls", 1)):
                    changed = copy.deepcopy(documents)
                    changed[index]["original_B27_failure_certification"][field] = wrong
                    with self.assertRaisesRegex(ValueError, "reviewed B27"):
                        reader.validate_failure_certification(*changed)
                changed = copy.deepcopy(documents)
                del changed[index]["original_B27_failure_certification"]
                with self.assertRaisesRegex(ValueError, "reviewed B27"):
                    reader.validate_failure_certification(*changed)

    def test_tensor_runtime_rejected_before_new_worker_or_tempfile(self):
        with patch.dict(sys.modules, {"torch": object()}), \
             patch.object(reader.tempfile, "TemporaryDirectory", side_effect=AssertionError("no temporary write")), \
             patch.object(reader.subprocess, "check_output", side_effect=AssertionError("no process")):
            with self.assertRaisesRegex(ValueError, "tensor imports"):
                reader.verify_training_original(Path("/invented"), lock(), Path("/invented-inputs"))

    def test_inert_fetch_rejected_before_network_or_output_directory(self):
        self.context.stop()  # This regression exercises the actual inert B28 files.
        train, _producer = reader.modules()
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(train, "gh", side_effect=AssertionError("no network before binding")) as network:
            output = Path(temporary) / "original-training"
            with self.assertRaises(ValueError):
                reader.fetch_training_original(output, lock(), Path(temporary) / "inputs")
            network.assert_not_called()
            self.assertFalse(output.exists())

        # Each preparation gate must run before structural lock validation or
        # acquisition, including future states where earlier gates are bound.
        for stopped_at in ("design", "opening", "assessment_binding"):
            checked = []
            def gate(name):
                def check():
                    checked.append(name)
                    if name == stopped_at:
                        raise ValueError("fictional inactive " + name)
                return check
            with self.subTest(gate=stopped_at), tempfile.TemporaryDirectory() as temporary, \
                 patch.object(train, "design", side_effect=gate("design")), \
                 patch.object(train, "opening", side_effect=gate("opening")), \
                 patch.object(train, "assessment_binding", side_effect=gate("assessment_binding")), \
                 patch.object(train, "gh", side_effect=AssertionError("no network")) as network, \
                 patch.object(reader, "validate_lock", side_effect=AssertionError("gates before lock")):
                output = Path(temporary) / "original-training"
                with self.assertRaisesRegex(ValueError, "fictional inactive"):
                    reader.fetch_training_original(output, lock(), Path(temporary) / "inputs")
                self.assertEqual(checked, ["design", "opening", "assessment_binding"][:checked.index(stopped_at) + 1])
                network.assert_not_called()
                self.assertFalse(output.exists())

    def test_report_cannot_claim_gradients_or_omit_new_objective(self):
        bound = lock()
        report = dict(schema=reader.REPORT_SCHEMA, original_lock_sha256=reader.sha(reader.canonical(bound)),
            result_sha256=bound["training"]["result_sha256"], all_original_producer_git_verified=True,
            all_original_inputs_verified=True, all_saved_training_arrays_verified=True,
            natural_no_event_terms_recomputed=True, complete_original_B27_NO_PROMOTION_verified=True,
            original_B24_control_unchanged=True, saved_training_observations=dict(natural_objective_returns_verified=384),
            new_model_calls=0, new_source_calls=0, new_optimizer_updates=0, promotion_allowed=False,
            product_released=False, gradients_recomputed=False, optimizer_updates_replayed=False)
        self.assertIs(reader.validate_report(report, bound), report)
        for key, value in (("gradients_recomputed", True), ("optimizer_updates_replayed", True),
                           ("natural_no_event_terms_recomputed", False), ("schema", reader.legacy.REPORT_SCHEMA)):
            changed = dict(report, **{key: value})
            with self.assertRaises(ValueError):
                reader.validate_report(changed, bound)

    def test_finite_complete_npz_and_frontend_without_pickle_claim(self):
        geometry = {"weight": (438, 100), "bias": (76,), "mel_filters": (257, 40),
                    "train_mean": (40,), "train_scale": (40,)}
        original = {name + ".npy": npy(shape, struct.pack("<f", 1.0 if name == "train_scale" else 0.0) * math.prod(shape))
                    for name, shape in geometry.items()}
        learned = dict(original)
        learned["weight.npy"] = npy(geometry["weight"], struct.pack("<f", 0.5) * 43800)
        report = reader.candidate_npz(zip_bytes(learned), zip_bytes(original))
        self.assertEqual(report["parameter_count"], 43876)
        self.assertTrue(report["initializer_frontend_bytes_unchanged"])
        self.assertFalse(report["serialized_pt_tensors_deserialized"])
        self.assertFalse(report["pt_npz_elementwise_equivalence_independently_recomputed"])
        changed = dict(learned)
        changed["train_scale.npy"] = npy((40,), struct.pack("<f", 2.0) * 40)
        with self.assertRaisesRegex(ValueError, "frontend"):
            reader.candidate_npz(zip_bytes(changed), zip_bytes(original))
        changed = dict(learned)
        changed["weight.npy"] = npy(geometry["weight"], struct.pack("<f", math.nan) * 43800)
        with self.assertRaisesRegex(ValueError, "finite"):
            reader.candidate_npz(zip_bytes(changed), zip_bytes(original))


if __name__ == "__main__":
    unittest.main()
