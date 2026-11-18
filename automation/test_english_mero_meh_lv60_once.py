"""Offline orchestration contracts. No model runtime, downloads or forwards."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_mero_meh_lv60_once as m


def plan_fixture(*, enabled=False):
    plan = m.read_json(m.PLAN)
    plan["enabled"] = enabled
    plan["main_before"] = m.SOURCE["head_sha"]
    return plan


def new_identity(plan):
    run = {"repository": m.REPOSITORY, "run_id": "99999999999", "commit": "a" * 40,
           "event": "push", "run_attempt": "1"}
    env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": m.REPOSITORY,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": run["run_id"], "GITHUB_SHA": run["commit"],
        "GITHUB_WORKFLOW_REF": m.REPOSITORY + "/" + m.WORKFLOW + "@refs/heads/main"}
    event = {"repository": {"full_name": m.REPOSITORY, "private": False}, "ref": "refs/heads/main",
        "before": plan["main_before"], "after": run["commit"], "forced": False, "deleted": False}
    return run, env, event


def api_fixture(plan):
    pin = plan["source_artifact"]
    repository = {"id": 1397319075, "full_name": m.REPOSITORY, "private": False}
    run = {"id": pin["run_id"], "head_sha": pin["head_sha"], "head_branch": "main", "event": "push",
        "run_attempt": 1, "path": plan["source_workflow_path"], "status": "completed", "conclusion": "success",
        "repository": repository, "head_repository": copy.deepcopy(repository)}
    artifact = {"id": pin["artifact_id"], "name": pin["name"], "size_in_bytes": pin["bytes"],
        "expired": False, "digest": "sha256:" + pin["sha256"],
        "workflow_run": {"id": pin["run_id"], "head_sha": pin["head_sha"], "head_branch": "main",
            "repository_id": repository["id"], "head_repository_id": repository["id"]}}
    return run, artifact


def miniature_archive(directory, extra=None):
    """A small closed packet for archive parsing, never a valid inference source."""
    plan = plan_fixture()
    source = directory / "source"
    source.mkdir()
    for name in m.DOCUMENTS - {"FILES.json"}:
        value = {"fixture": name}
        if name == "ADMISSION.json":
            value = {"status": plan["original_stop_preserved"], "training_allowed": False,
                     "permanently_excluded_from_training": False}
        if name == "RESULT.json":
            value = {"stage": "combine", "status": "MACHINE_EVIDENCE_PACKET_RETAINED",
                     "run": plan["source_run"], "actual_total_model_calls": 720}
        m.write_json(source / name, value)
    m.write_json(source / "FILES.json", {"schema": "english-mero-meh-corpus-files-v1",
        "files": m.observer().file_records(source)})
    plan["source_document_pins"] = {name: {"bytes": (source / name).stat().st_size, "sha256": m.digest(source / name)}
        for name in m.DOCUMENTS}
    plan["source_files_manifest_sha256"] = plan["source_document_pins"]["FILES.json"]["sha256"]
    archive = directory / "source.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        for path in sorted(source.iterdir()):
            writer.write(path, path.name)
        if extra:
            extra(writer)
    plan["source_artifact"].update(bytes=archive.stat().st_size, sha256=m.digest(archive))
    return archive, plan


def partial_packet(plan, run):
    rows = []
    for i, row in enumerate(plan["panel_rows"]):
        value = {"id": row["id"], "status": "success" if i == 0 else "missing"}
        if i == 0:
            value.update(phone_tokens=["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"], capture_path="rows/" + row["id"] + "/CAPTURE.json",
                capture_sha256="e" * 64, normalized_input_sha256="f" * 64,
                xlsr_reference={"normalized_input_sha256": "f" * 64})
        rows.append(value)
    return {"summary": {"run": run, "status": "INCOMPLETE_UNKNOWN_COUNTERS", "actual_forward_attempts": None,
        "native_forward_returns": None, "completed_count": 1, "failed_count": 0, "not_started_count": None,
        "attempts_lower_bound": 2, "attempts_upper_bound": 24, "unknown_count": 23, "terminal_receipt_present": False},
        "rows": rows, "new_model_calls": 0}


def old_machine_fixture(source, plan):
    machine = []
    for row in plan["panel_rows"]:
        observations = {p: {"raw_transcript": "Unchanged original transcript."} for p in ("qwen", "whisper", "parakeet")}
        observations["phoneme"] = {"phone_tokens": ["h", "eɪ", "m", "ɛ", "ɹ", "ɔ"],
            "capture_path": "observers/phoneme/rows/" + row["id"] + "/CAPTURE.json", "capture_sha256": "d" * 64}
        machine.append({"id": row["id"], "observations": observations})
    m.write_json(source / "MACHINE_EVIDENCE.json", {"rows": machine})
    m.write_json(source / "ADMISSION.json", {"status": "STOP_FRESH_CORPUS_COVERAGE", "training_allowed": False})


class OrchestrationTests(unittest.TestCase):
    def test_both_disabled_and_enabled_source_contract_are_model_free(self):
        with mock.patch.object(m.subprocess, "run", side_effect=AssertionError("No process")), \
             mock.patch.object(m.subprocess, "check_output", side_effect=AssertionError("No process")):
            for enabled in (False, True):
                plan = plan_fixture(enabled=enabled)
                report = m.source_preflight(plan)
                self.assertEqual(report["enabled"], enabled)
                self.assertEqual(report["denominator"], 24)
                self.assertEqual(report["new_model_calls"], 0)
        self.assertFalse({"torch", "numpy", "transformers"} & sys.modules.keys())

    def test_default_cli_uses_fixture_activation_and_returns_explicit_zero(self):
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "plan.json"
                m.write_json(path, plan_fixture(enabled=enabled))
                stdout = io.StringIO()
                with mock.patch.object(m, "PLAN", path), redirect_stdout(stdout):
                    result = m.main([])
                self.assertIs(type(result), int)
                self.assertEqual(result, 0)
                self.assertEqual(json.loads(stdout.getvalue())["enabled"], enabled)

    def test_guard_exact_new_main_attempt_and_integer_cli_result(self):
        plan = plan_fixture(enabled=True)
        run, env, event = new_identity(plan)
        self.assertEqual(m.execution_identity(plan, environ=env, event=event, head=run["commit"], parent=plan["main_before"]), run)
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                           ("GITHUB_RUN_ID", plan["source_run"]["run_id"])):
            with self.assertRaises(ValueError):
                m.execution_identity(plan, environ=env | {key: value}, event=event, head=run["commit"], parent=plan["main_before"])
        with self.assertRaisesRegex(ValueError, "CHECKOUT_AND_PARENT"):
            m.execution_identity(plan, environ=env, event=event, head=run["commit"], parent="b" * 40)
        with self.assertRaises(ValueError):
            m.execution_identity(plan, environ=env, event=event | {"repository": {"full_name": m.REPOSITORY, "private": True}},
                head=run["commit"], parent=plan["main_before"])
        with mock.patch.object(m, "source_preflight", return_value={"enabled": True}), \
             mock.patch.object(m, "execution_identity", return_value=run), redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(["--guard"]), 0)

    def test_source_api_binds_original_run_artifact_and_public_head(self):
        plan = plan_fixture()
        run, artifact = api_fixture(plan)
        self.assertTrue(m.validate_source_api(run, artifact, plan)["source_api_verified"])
        for wrong in (artifact | {"id": artifact["id"] + 1}, artifact | {"expired": True},
                      artifact | {"digest": "sha256:" + "0" * 64},
                      artifact | {"workflow_run": artifact["workflow_run"] | {"head_sha": "a" * 40}}):
            with self.assertRaises(ValueError):
                m.validate_source_api(run, wrong, plan)
        with self.assertRaises(ValueError):
            m.validate_source_api(run | {"run_attempt": 2}, artifact, plan)

    def test_archive_exact_closed_bytes_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, plan = miniature_archive(root)
            target = root / "extract"
            self.assertEqual(m.extract_archive(archive, target, plan), target)
            self.assertEqual(m.digest(target / "ADMISSION.json"), plan["source_document_pins"]["ADMISSION.json"]["sha256"])
            archive.write_bytes(archive.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "ZIP_BYTES"):
                m.extract_archive(archive, root / "another", plan)

    def test_archive_traversal_symlink_and_unlisted_payload_rejected(self):
        def symlink(writer):
            item = zipfile.ZipInfo("symlink")
            item.create_system = 3
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
            writer.writestr(item, "outside")
        attacks = (lambda writer: writer.writestr("../outside", "bad"), symlink,
                   lambda writer: writer.writestr("extra.txt", "unlisted"))
        for attack in attacks:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive, plan = miniature_archive(root, attack)
                with self.assertRaises(ValueError):
                    m.extract_archive(archive, root / "extract", plan)
                self.assertFalse((root / "outside").exists())

    def test_install_authenticates_before_fresh_runtime_and_observe_returns_status(self):
        plan = plan_fixture(enabled=True)
        run, _, _ = new_identity(plan)
        module = m.observer()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            def authenticate(*args):
                calls.append("source")
                return {"source_ready": True}
            def install(bound, output):
                self.assertEqual(calls, ["source"])
                self.assertFalse(Path(output).exists())
                Path(output).mkdir()
                calls.append("install")
                return {"status": "COMPLETE"}
            with mock.patch.object(m, "source_preflight", return_value={"enabled": True}), \
                 mock.patch.object(m, "execution_identity", return_value=run), \
                 mock.patch.object(m, "authenticated_source", side_effect=authenticate), \
                 mock.patch.object(module, "install", side_effect=install), redirect_stdout(io.StringIO()):
                self.assertEqual(m.main(["--install", "--source", str(root / "source"), "--output", str(root / "runtime")]), 0)
            self.assertEqual(calls, ["source", "install"])
            for status, expected in (("COMPLETE", 0), ("INCOMPLETE", 1)):
                with mock.patch.object(m, "source_preflight", return_value={"enabled": True}), \
                     mock.patch.object(m, "execution_identity", return_value=run), \
                     mock.patch.object(m, "authenticated_source", return_value={"source_ready": True}), \
                     mock.patch.object(module, "observe", return_value={"status": status}), redirect_stdout(io.StringIO()):
                    self.assertEqual(m.main(["--observe", "--source", str(root / "source"), "--output", str(root / "observer")]), expected)

    def test_partial_finalization_preserves_24_unknowns_original_stop_and_frozen_workflow(self):
        plan = plan_fixture()
        run, _, _ = new_identity(plan)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            old_machine_fixture(source, plan)
            before = (source / "ADMISSION.json").read_bytes()
            packet = partial_packet(plan, run)
            with mock.patch.object(m, "authenticated_source", return_value={"source_ready": True}), \
                 mock.patch.object(m, "verify_documents"), mock.patch.object(m.observer(), "readback", return_value=packet):
                result = m.finalize(plan, source, root / "observer", root / "diagnostic", run, "failure")
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
            self.assertIsNone(result["actual_forward_attempts"])
            self.assertEqual((result["attempts_lower_bound"], result["attempts_upper_bound"]), (2, 24))
            self.assertEqual((result["denominator"], result["unknown_count"]), (24, 23))
            self.assertFalse(result["complete"])
            comparison = m.read_json(root / "diagnostic/COMPARISON.json")
            self.assertEqual(len(comparison["rows"]), 24)
            self.assertFalse(comparison["rows"][0]["exact_token_sequence_equal"])
            self.assertFalse(comparison["phone_normalization_applied"])
            self.assertFalse(comparison["admission_recomputed"])
            self.assertEqual((source / "ADMISSION.json").read_bytes(), before)
            frozen = root / "diagnostic/frozen_source"
            self.assertEqual(m.digest(frozen / m.WORKFLOW), plan["source_files_sha256"][m.WORKFLOW])
            self.assertEqual(m.read_json(frozen / "automation/english_mero_meh_lv60_plan.json"), plan)
            index = m.read_json(root / "diagnostic/FILES.json")
            self.assertEqual(index["files"], m.observer().file_records(root / "diagnostic"))

    def test_absent_source_writes_unknown_diagnostic_without_false_zero(self):
        plan = plan_fixture()
        run, _, _ = new_identity(plan)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(m, "authenticated_source", side_effect=ValueError("Missing source")):
                with self.assertRaises(ValueError):
                    m.finalize(plan, root / "missing", root / "absent", root / "diagnostic", run, "skipped")
            result = m.read_json(root / "diagnostic/RESULT.json")
            self.assertEqual(result["denominator"], 24)
            self.assertIsNone(result["actual_forward_attempts"])
            self.assertFalse(result["evidence_verified"])
            self.assertFalse(result["missing_terminal_is_zero"])

    def test_workflow_retains_captures_without_runtime_or_private_assets(self):
        workflow = (m.ROOT / m.WORKFLOW).read_text()
        self.assertIn("24 unchanged full PCMs", workflow)
        self.assertIn("1800s", workflow)
        self.assertIn("300s", workflow)
        self.assertIn("!${{ runner.temp }}/mero-meh-lv60-audit/runtime/venv/**", workflow)
        self.assertIn("!${{ runner.temp }}/mero-meh-lv60-audit/observer-private-assets/**", workflow)
        self.assertIn("include-hidden-files: true", workflow)
        self.assertNotIn("workflow_dispatch:", workflow)
        self.assertNotIn("schedule:", workflow)


if __name__ == "__main__":
    unittest.main()
