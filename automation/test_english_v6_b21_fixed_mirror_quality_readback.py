"""Stdlib fixtures only: these outputs never authorize any real model gate."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
REPOSITORY = Path(os.environ.get("B21_TEST_REPOSITORY", HERE.parent))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


reader = load(HERE / "english_v6_b21_fixed_mirror_quality_readback.py", "readback_fixture")
primary = load(REPOSITORY / "automation" / (reader.EVALUATOR_STEMS["primary"] + "_eval.py"), "original_primary_fixture")
decision = load(REPOSITORY / "automation/english_v6_b14_decision.py", "original_b14_fixture")


def npy(labels, batch=False):
    shape = (1, len(labels), 4) if batch else (len(labels), 4)
    header = repr(dict(descr="<f4", fortran_order=False, shape=shape)).encode("ascii")
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    body = b"".join(struct.pack("<4f", *(1.0 if index == label else 0.0 for index in range(4))) for label in labels)
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + body, body


def zip_bytes(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in members.items():
            archive.writestr(key, value)
    return output.getvalue()


def fixture(panel="primary"):
    rows = []
    for index in range(48):
        cohort = "synthetic" if index < 36 else "natural"
        positive = index < 12
        group = "family-" + str(index % 12) if cohort == "synthetic" else "reader-" + str(index % 8)
        rows.append(dict(id="fixture-" + str(index), source_group_id=group, cohort=cohort,
            cell_id="positive" if positive else "ordinary", observed_class=2 if positive else 0,
            tier="silver", pcm_sha256=f"{index + 1:064x}", wav_sha256=f"{index + 1000:064x}", samples=3200))
    model_pins = dict(B0=primary.B0_PT, B9=primary.B9_PT, control="c" * 64, challenger="d" * 64)
    spec = dict(schema=reader.SCHEMAS[panel], scoring_allowed=True, assessment_sha256=reader.ASSESSMENT_SHA,
        training_assessment_sha256=reader.ASSESSMENT_SHA, training_design_sha256=reader.DESIGN_SHA,
        gains=list(reader.GAINS), maximum_model_forwards=1440, model_pt_sha256=model_pins,
        producer_manifest_sha256="e" * 64, source_gate=primary.GATE)
    execution = dict(run_id="123", run_head="a" * 40, run_attempt=1, prepared_parent="b" * 40)
    admission = dict(schema=spec["schema"], identity=execution, assessment_sha256=reader.ASSESSMENT_SHA,
        training_design_sha256=reader.DESIGN_SHA, admitted_rows=[primary.score_identity(row, row["cohort"]) for row in rows],
        source_coverage=primary.source_counts(rows), source_gate="SOURCE_GATE_PASS", source_shortfall={}, model_calls=0,
        original_source_rows=120, old_development_reused=False, promotion_allowed=False,
        replication_opened=panel == "replication", producer_manifest_sha256=spec["producer_manifest_sha256"],
        registered_inputs_sha256="f" * 64, workflow_sha256="0" * 64)
    base = dict(schema=spec["schema"], **execution, assessment_sha256=reader.ASSESSMENT_SHA,
        training_assessment_sha256=reader.ASSESSMENT_SHA, training_design_sha256=reader.DESIGN_SHA,
        registered_inputs_sha256=admission["registered_inputs_sha256"], producer_manifest_sha256=spec["producer_manifest_sha256"],
        workflow_sha256=admission["workflow_sha256"], implementation_sha256=reader.sha(Path(primary.__file__).read_bytes()),
        source_gate="SOURCE_GATE_PASS", source_shortfall={}, source_coverage=admission["source_coverage"], model_pt_sha256=model_pins,
        installed_software_baseline="B0", B20_failed_primary_preserved=True, fresh_source_originals=True,
        original_natural_reused_without_reobservation=False, optimizer_updates=0, replication_opened=panel == "replication",
        ssc305_physical=False, natural_field_far_per_hour=None, field_3_5m_frr=None, **dict.fromkeys(reader.FALSE_FIELDS, False))
    if panel == "replication":
        admission.update(primary_result_sha256="1" * 64, primary_quality_lock_sha256="2" * 64)
        base.update(panel="replication", primary_result_sha256="1" * 64, primary_quality_lock_sha256="2" * 64)
    started = dict(base, panel_gate="STARTED", records=[], actual_forwards=0, saved_array_count=0, raw_output_count=0,
        candidate_engineering_validation_allowed=False, replication_open_allowed=False)
    if panel == "replication": started["research_replication_passed"] = False
    arrays, members, records = {}, {}, []
    for index, row in enumerate(rows):
        record = primary.score_identity(row, row["cohort"])
        record["gains"] = {}
        for gain in reader.GAINS:
            record["gains"][gain] = {}
            for arm in reader.ARMS:
                event = (index < 12 and (arm != "control" or index > 0)) or (index >= 36 and arm in ("B9", "control"))
                labels = [1, 2, 3] + [0] * 14 if event else [0] * 17
                key = row["id"] + "__" + gain + "__" + arm
                raw, body = npy(labels)
                arrays[key + ".npy"] = raw
                members["raw_outputs/" + key + ".npy"] = npy(labels, True)[0]
                members["attempts/" + key + ".json"] = reader.canonical(dict(key=key, role=arm, source_id=row["id"], gain=gain, status="consumed_unknown"))
                record["gains"][gain][arm] = dict(array=key, frames=17, sha256=reader.sha(body), **primary.ctc_events(labels))
        records.append(record)
    npz = zip_bytes(arrays)
    result = dict(base, records=records, actual_forwards=576, saved_array_count=576, raw_output_count=576,
        model_forward_calls=dict.fromkeys(reader.ARMS, 144), consumed_unknown=[], replay_allowed=False,
        candidate_engineering_validation_allowed=panel == "replication", replication_open_allowed=panel == "primary",
        logits_sha256=reader.sha(npz), **decision.decide(records))
    if panel == "replication": result["research_replication_passed"] = True
    members.update({"RESULT.json": reader.canonical(result), "STARTED.json": reader.canonical(started),
                    "ADMISSION.json": reader.canonical(admission), "ALL_FRAME_LOGITS.npz": npz})
    return members, spec, admission, rows


class ReadbackTests(unittest.TestCase):
    def test_fetch_saves_original_control_bytes_and_never_renders_them(self):
        raw = b"PK\x03\x04\x1b[31m\xff\x00\x80\r\n"
        def pin(number):
            return dict(id=number, run_id=123, run_head="a" * 40,
                        bytes=len(raw), sha256=reader.sha(raw))
        lock = dict(schema="english-v6-b21-original-quality-lock-v1", panel="primary",
                    result=pin(1), producer=pin(2), consumed=pin(3),
                    evaluator_sha256=reader.PRIMARY_CODE_SHA, inputs_sha256="b" * 64,
                    producer_manifest_sha256="c" * 64, original_result_sha256="d" * 64)
        def captured(argv, **kwargs):
            self.assertEqual(argv[:2], ["gh", "api"])
            self.assertEqual(argv[-1], "--allow-escape-sequences")
            self.assertEqual(kwargs, dict(stdout=reader.subprocess.PIPE, stderr=reader.subprocess.PIPE, check=True))
            return reader.subprocess.CompletedProcess(argv, 0, raw, b"")
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "originals"
            with patch.object(reader.subprocess, "run", side_effect=captured) as run, \
                 patch.object(reader, "original", side_effect=ValueError("stop before archive interpretation")):
                with self.assertRaisesRegex(ValueError, "stop before archive interpretation"):
                    reader.fetch_quality_original(destination, lock)
                self.assertEqual(run.call_count, 6)
            for key in ("result", "producer", "consumed"):
                self.assertEqual((destination / "originals" / (key + ".zip")).read_bytes(), raw)

    @classmethod
    def setUpClass(cls):
        cls.base = fixture()

    def verify(self, members=None, panel="primary"):
        original, spec, admission, rows = self.base if panel == "primary" else fixture(panel)
        return reader.saved_arrays(zip_bytes(members or original), spec, panel, admission, rows, primary, decision)

    def test_complete_raw_array_fixtures_for_both_panels(self):
        for panel in ("primary", "replication"):
            self.assertEqual(self.verify(panel=panel)["result"]["panel_gate"], "PASS")
        self.assertEqual(reader.sha(Path(primary.__file__).read_bytes()), reader.PRIMARY_CODE_SHA)
        self.assertEqual(reader.sha(Path(decision.__file__).read_bytes()), reader.QUALITY_CODE_SHA)
        self.assertEqual(reader.sha(decision.ASSESSMENT.read_bytes()), reader.QUALITY_ASSESSMENT_SHA)

    def test_raw_return_tampering_is_rejected(self):
        members = dict(self.base[0])
        key = next(key for key in members if key.startswith("raw_outputs/"))
        members[key] = npy([0] * 17, True)[0]
        with self.assertRaisesRegex(ValueError, "tensor byte"):
            self.verify(members)

    def test_missing_consumption_journal_is_rejected(self):
        members = dict(self.base[0])
        del members[next(key for key in members if key.startswith("attempts/"))]
        with self.assertRaises(ValueError): self.verify(members)

    def test_no_promotion_result_cannot_authorize(self):
        members = dict(self.base[0]); report = reader.document(members["RESULT.json"])
        report["panel_gate"] = "NO_PROMOTION"; members["RESULT.json"] = reader.canonical(report)
        with self.assertRaisesRegex(ValueError, "complete actual PASS"): self.verify(members)

    def test_partial_or_unknown_consumption_cannot_authorize(self):
        for field, value in (("consumed_unknown", ["one"]), ("actual_forwards", 575)):
            members = dict(self.base[0]); report = reader.document(members["RESULT.json"])
            report[field] = value; members["RESULT.json"] = reader.canonical(report)
            with self.assertRaises(ValueError): self.verify(members)

    def test_wrong_saved_clock_cannot_authorize(self):
        members = dict(self.base[0]); report = reader.document(members["RESULT.json"])
        report["records"][0]["gains"]["1.0"]["B0"]["event_samples"] = [999]
        members["RESULT.json"] = reader.canonical(report)
        with self.assertRaisesRegex(ValueError, "sample clocks"): self.verify(members)

    def test_wrong_model_and_changed_b14_summary_are_rejected(self):
        for field, value in (("model_pt_sha256", dict.fromkeys(reader.ARMS, "7" * 64)), ("restored_positives", [])):
            members = dict(self.base[0]); report = reader.document(members["RESULT.json"])
            report[field] = value; members["RESULT.json"] = reader.canonical(report)
            with self.assertRaises(ValueError): self.verify(members)

    def test_unbound_lock_rejects_before_io_process_or_network(self):
        with patch.object(reader.subprocess, "check_output", side_effect=AssertionError("process/network forbidden")), \
             patch.object(reader.subprocess, "run", side_effect=AssertionError("process/network forbidden")):
            for call in (reader.verify_quality_original, reader.fetch_quality_original):
                with self.assertRaises(ValueError): call(Path("/not-created"), {})

    def test_full_npy_batch_and_finiteness_are_mandatory(self):
        raw, _ = npy([0] * 17)
        with self.assertRaises(ValueError): reader.read_npy(raw, (1, 17, 4))
        bad = raw[:-4] + struct.pack("<f", float("nan"))
        with self.assertRaises(ValueError): reader.read_npy(bad, (17, 4))

    def test_offline_worker_forbids_model_and_network_imports(self):
        with reader.offline_only():
            with self.assertRaises(ValueError): __import__("torch")
            with self.assertRaises(ValueError): reader.subprocess.check_output(["gh", "api", "anything"])
            with self.assertRaises(ValueError): reader.socket.socket()


if __name__ == "__main__":
    unittest.main()
