"""Fail-only saved-evidence contracts; invented arrays never authorize a run."""
from __future__ import annotations
import ast
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24q1_primary_failure as failure
import english_v6_b24q1_quality_readback as original
import test_english_v6_b24q1_quality_readback as fixtures


def function(module, name):
    return next(node for node in ast.parse(Path(module.__file__).read_bytes()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def syntax(node):
    return ast.dump(node, include_attributes=False)


def lock_fixture():
    def pin(number):
        return dict(id=number, run_id=123, run_head="a" * 40, bytes=100, sha256="b" * 64)
    return dict(schema="english-v6-b24q1-original-quality-lock-v1", panel="primary",
        result=pin(1), producer=pin(2), consumed=pin(3), evaluator_sha256=original.PRIMARY_CODE_SHA,
        inputs_sha256="c" * 64, producer_manifest_sha256="d" * 64, original_result_sha256="e" * 64)


def failed_fixture():
    members, spec, admission, rows = fixtures.fixture()
    result = original.document(members["RESULT.json"])
    with zipfile.ZipFile(io.BytesIO(members["ALL_FRAME_LOGITS.npz"])) as archive:
        arrays = {name: archive.read(name) for name in archive.namelist()}
    record = result["records"][0]
    key = record["id"] + "__1.0__challenger"
    labels = [0] * 17
    raw, body = fixtures.npy(labels)
    arrays[key + ".npy"] = raw
    members["raw_outputs/" + key + ".npy"] = fixtures.npy(labels, True)[0]
    record["gains"]["1.0"]["challenger"] = dict(array=key, frames=17, sha256=original.sha(body),
                                               **fixtures.primary.ctc_events(labels))
    members["ALL_FRAME_LOGITS.npz"] = fixtures.zip_bytes(arrays)
    result.update(fixtures.decision.decide(result["records"]))
    result.update(logits_sha256=original.sha(members["ALL_FRAME_LOGITS.npz"]),
                  replication_open_allowed=False, candidate_engineering_validation_allowed=False)
    members["RESULT.json"] = original.canonical(result)
    return members, spec, admission, rows


class FailureReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = failed_fixture()

    def verify(self, members=None, spec=None, admission=None, rows=None):
        originals, original_spec, original_admission, original_rows = self.base
        return failure.saved_failure_arrays(fixtures.zip_bytes(originals if members is None else members),
            original_spec if spec is None else spec, "primary", original_admission if admission is None else admission,
            original_rows if rows is None else rows, fixtures.primary, fixtures.decision)

    def test_immutable_source_and_parsers_are_the_original_implementations(self):
        self.assertEqual(original.sha(failure.BASE_READER_PATH.read_bytes()), failure.BASE_READER_SHA)
        for name in ("source_admission", "read_npy", "original", "identity", "offline_only", "valid_artifact"):
            self.assertIs(getattr(failure, name), getattr(original, name))
        self.assertEqual(original.sha(Path(fixtures.primary.__file__).read_bytes()), original.PRIMARY_CODE_SHA)
        self.assertEqual(original.sha(Path(fixtures.decision.__file__).read_bytes()), original.QUALITY_CODE_SHA)
        self.assertEqual(original.sha(fixtures.decision.ASSESSMENT.read_bytes()), original.QUALITY_ASSESSMENT_SHA)

    def test_only_terminal_verdict_and_downstream_flags_differ_in_array_validation(self):
        before = ast.get_source_segment(Path(original.__file__).read_text(), function(original, "saved_arrays"))
        after = ast.get_source_segment(Path(failure.__file__).read_text(), function(failure, "saved_failure_arrays"))
        changes = (
            ("def saved_failure_arrays(", "def saved_arrays("),
            ("Require exact NO_PROMOTION from every original raw return and unchanged B14.", "Require an exact PASS from every original raw return and unchanged B14."),
            ('result["panel_gate"] == "NO_PROMOTION"', 'result["panel_gate"] == "PASS"'),
            ("only complete actual NO_PROMOTION; STOP, partial return, unknown consumption and PASS stay closed", "only a complete actual PASS; STOP, partial return, unknown consumption or NO_PROMOTION stays closed"),
            ('result["replication_open_allowed"] is False', 'result["replication_open_allowed"] is (panel == "primary")'),
            ('result["candidate_engineering_validation_allowed"] is False', 'result["candidate_engineering_validation_allowed"] is (panel == "replication")'),
            ("failed candidate cannot open any next-quality or engineering stage", "next-stage flags exactly match this original PASS"),
            ('replay["panel_gate"] == "NO_PROMOTION"', 'replay["panel_gate"] == "PASS"'),
            ("unchanged B14 independently recomputes the original NO_PROMOTION", "every unchanged original B14 condition independently recomputes to PASS"),
        )
        for left, right in changes:
            self.assertEqual(after.count(left), 1)
            after = after.replace(left, right)
        self.assertEqual(syntax(ast.parse(before)), syntax(ast.parse(after)))

    def test_all_producer_api_input_and_consumption_checks_precede_only_changed_report(self):
        before = copy.deepcopy(function(original, "_verify_worker"))
        after = copy.deepcopy(function(failure, "_verify_failure_worker"))
        after.name = before.name
        for node in ast.walk(after):
            if isinstance(node, ast.Name) and node.id == "saved_failure_arrays":
                node.id = "saved_arrays"
        before.body[-1].body[-1] = after.body[-1].body[-1] = ast.Return(value=ast.Constant(value="verified-originals"))
        self.assertEqual(syntax(before), syntax(after))

    def test_full_failure_requires_every_raw_return_and_unchanged_b14_recomputation(self):
        checked = self.verify()
        self.assertEqual(checked["result"]["panel_gate"], "NO_PROMOTION")
        self.assertEqual(checked["result"]["actual_forwards"], 576)
        self.assertTrue(checked["result"]["lost_positives"])
        self.assertFalse(checked["result"]["replication_open_allowed"])
        self.assertFalse(checked["result"]["candidate_engineering_validation_allowed"])

    def test_original_pass_and_summary_only_forged_failure_both_reject(self):
        members, spec, admission, rows = fixtures.fixture()
        for forged in (False, True):
            copied = dict(members)
            if forged:
                report = original.document(copied["RESULT.json"])
                report.update(panel_gate="NO_PROMOTION", replication_open_allowed=False, candidate_engineering_validation_allowed=False)
                copied["RESULT.json"] = original.canonical(report)
            with self.assertRaises(ValueError):
                self.verify(copied, spec, admission, rows)

    def test_failed_result_cannot_pass_the_original_pass_reader(self):
        members, spec, admission, rows = self.base
        with self.assertRaisesRegex(ValueError, "complete actual PASS"):
            original.saved_arrays(fixtures.zip_bytes(members), spec, "primary", admission, rows,
                                  fixtures.primary, fixtures.decision)

    def test_raw_journal_clocks_counts_unknown_or_stop_tampering_rejects(self):
        for kind in ("journal", "raw", "clock", "count", "unknown", "stop", "orphan", "extra_event"):
            members = dict(self.base[0])
            report = original.document(members["RESULT.json"])
            if kind == "journal":
                del members[next(key for key in members if key.startswith("attempts/"))]
            elif kind == "raw":
                name = "raw_outputs/fixture-1__1.0__B0.npy"
                members[name] = fixtures.npy([0] * 17, True)[0]
            elif kind == "clock":
                report["records"][1]["gains"]["1.0"]["B0"]["event_samples"] = [999]
            elif kind == "count":
                report["actual_forwards"] -= 1
            elif kind == "unknown":
                report["consumed_unknown"] = ["one"]
            elif kind == "stop":
                report["panel_gate"] = "STOP"
            elif kind == "orphan":
                members["raw_outputs/unregistered.npy"] = fixtures.npy([0] * 17, True)[0]
            elif kind == "extra_event":
                report["added_ordinary_events"] = [["not-recomputed", "1.0", "B9"]]
            members["RESULT.json"] = original.canonical(report)
            with self.assertRaises(ValueError, msg=kind):
                self.verify(members)

    def test_no_downstream_flag_or_source_audit_mutation_can_hide_in_failure(self):
        for field, value in (("replication_open_allowed", True), ("candidate_engineering_validation_allowed", True),
                             ("promotion_allowed", True), ("source_audit", {}), ("B24R1_original_source_stop_preserved", False)):
            members = dict(self.base[0])
            report = original.document(members["RESULT.json"])
            report[field] = value
            members["RESULT.json"] = original.canonical(report)
            with self.assertRaises(ValueError, msg=field):
                self.verify(members)

    def test_quarantined_original_or_different_model_is_not_a_valid_failure_sample(self):
        admission = copy.deepcopy(self.base[2])
        admission["source_audit"]["quarantined_ids"].append("fixture-0")
        with self.assertRaises(ValueError):
            self.verify(admission=admission)
        spec = copy.deepcopy(self.base[1])
        spec["model_pt_sha256"]["challenger"] = "f" * 64
        with self.assertRaises(ValueError):
            self.verify(spec=spec)

    def test_incomplete_or_replica_lock_rejects_before_network(self):
        valid = lock_fixture()
        failure.validate_lock(valid)
        for lock in (dict(valid, panel="replication"), dict(valid, result=None), dict(valid, evaluator_sha256="f" * 64), {}):
            with tempfile.TemporaryDirectory() as temporary, patch.object(failure.subprocess, "run") as launch:
                target = Path(temporary) / "untouched"
                with self.assertRaises((ValueError, KeyError)):
                    failure.fetch_failure_original(target, lock)
                self.assertFalse(target.exists())
                launch.assert_not_called()

    def test_changed_base_reader_rejects_before_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / failure.BASE_READER_PATH.name
            path.write_bytes(failure.BASE_READER_PATH.read_bytes() + b"\n")
            with patch.object(failure, "BASE_READER_PATH", path), patch.object(failure.subprocess, "check_output") as launch:
                with self.assertRaisesRegex(ValueError, "preregistered"):
                    failure.verify_failure_original(Path(temporary), lock_fixture())
                launch.assert_not_called()

    def test_public_wrapper_rejects_pass_or_any_authority_in_worker_response(self):
        lock = lock_fixture()
        report = dict(schema=failure.REPORT_SCHEMA, panel="primary", panel_gate="NO_PROMOTION",
            original_result_sha256=lock["original_result_sha256"], original_lock_sha256=original.sha(original.canonical(lock)),
            old_B25_and_B26_openers_unchanged=True, new_model_calls=0, new_source_calls=0, new_observer_calls=0,
            **dict.fromkeys(failure.CLOSED, False))
        cases = [("panel_gate", "PASS"), ("old_B25_and_B26_openers_unchanged", False), ("new_model_calls", 1)]
        cases += [(field, True) for field in failure.CLOSED]
        with tempfile.TemporaryDirectory() as temporary:
            for field, value in cases:
                changed = dict(report); changed[field] = value
                with patch.object(failure.subprocess, "check_output", return_value=original.canonical(changed)):
                    with self.assertRaisesRegex(ValueError, "no downstream authority", msg=field):
                        failure.verify_failure_original(Path(temporary), lock)


if __name__ == "__main__":
    unittest.main()
