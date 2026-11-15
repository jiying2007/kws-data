#!/usr/bin/env python3
"""Invented request/receipt fixtures only: no model, speech or network execution."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

SPEC = importlib.util.spec_from_file_location("background_consumer_test",
                                            Path(__file__).with_name("research_v4_background_continuation.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class BackgroundBoundaries(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / (C.REQUEST_ID + ".json")
        self.request = copy.deepcopy(C.FIXED)
        self.request.update(pipeline_commit=C.PIPELINE_COMMIT, reason="Invented evidence; no audio or models.")
        self.write(self.path, self.request)
        self.digest = C.core.sha256(self.path.read_bytes())
        self.event = self.root / "event.json"
        self.write(self.event, {"repository": {"private": False}})

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    @staticmethod
    def archive(files):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, raw in files.items():
                archive.writestr(name, raw)
        return output.getvalue()

    def environment(self, **changes):
        values = {"GITHUB_REPOSITORY": C.core.REPOSITORY, "GITHUB_EVENT_NAME": "push",
                  "GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_PATH": str(self.event),
                  "GITHUB_ACTIONS": "true", "GITHUB_RUN_ID": "999", "GITHUB_RUN_ATTEMPT": "1",
                  "GITHUB_SHA": "b" * 40, "GH_TOKEN": "invented-secret"}
        values.update(changes)
        return mock.patch.dict(os.environ, values, clear=True)

    def execution_fixture(self):
        state, pipeline = self.root / "state", self.root / "pipeline"
        (state / "result").mkdir(parents=True)
        (state / "claim").mkdir()
        (state / "result/REQUEST.json").write_bytes(self.path.read_bytes())
        script = pipeline / C.ENTRYPOINT
        script.parent.mkdir(parents=True)
        script.write_text("# invented source; never executed\n")
        self.write(state / "claim/CLAIM.json", C.claim_record(self.request, self.digest,
                   {"run_id": "999", "run_attempt": "1"}))
        return state, pipeline, script

    def source_fixture(self):
        files = {C.ENTRYPOINT: b"# invented source, not executed\n", "fixed/helper.py": b"# invented helper\n"}
        return files, {name: C.core.sha256(raw) for name, raw in files.items()}

    def test_new_training_contract_is_exact_and_separate_from_native_no_training(self):
        self.assertEqual(C.load_request(self.path)[2], self.digest)
        self.assertIsNone(C.core.native_consumer(self.path))
        self.assertEqual(C.core.fixed_consumer(self.path).ENTRYPOINT, C.ENTRYPOINT)
        self.assertEqual(C.FIXED["profile"]["optimizer_updates"], 192)
        self.assertEqual(C.FIXED["profile"]["native_eval_frames"], 431790)
        self.assertEqual(C.FIXED["profile"]["steps_per_arm"], 48)
        self.assertEqual(C.FIXED["profile"]["training_arms"], 4)
        for key, value in (("pipeline_commit", "a" * 40), ("pipeline_commit", "main"),
                           ("max_execution_seconds", 1201), ("max_verification_seconds", 181),
                           ("entrypoint", "arbitrary.py"), ("new_audio_generated", True),
                           ("human_gold", True), ("reason", "")):
            bad = copy.deepcopy(self.request)
            bad[key] = value
            self.write(self.path, bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.load_request(self.path)
        for key, value in (("optimizer_updates", False), ("optimizer_updates", 193),
                           ("training_arms", 3), ("pool_time_major", 1), ("threshold", 0.79),
                           ("natural_loss_weight", 1), ("train_microbatch_natural", 128),
                           ("native_eval_frames", 431790.0), ("training_arm_seconds", 181)):
            bad = copy.deepcopy(self.request)
            bad["profile"][key] = value
            self.write(self.path, bad)
            with self.subTest(profile=key), self.assertRaises(ValueError):
                C.load_request(self.path)

    def test_extra_duplicate_and_wrong_request_filename_are_rejected(self):
        self.path.write_text('{"schema":"first","schema":"duplicate"}')
        with self.assertRaises(ValueError):
            C.load_request(self.path)
        self.write(self.path, dict(self.request, shell="arbitrary"))
        with self.assertRaises(ValueError):
            C.load_request(self.path)
        other = self.root / "old-consumed-request.json"
        self.write(other, self.request)
        with self.assertRaises(ValueError):
            C.load_request(other)

    def test_claim_distinguishes_future_budget_from_already_completed_work(self):
        claim = C.claim_record(self.request, self.digest, {"run_id": "999", "run_attempt": "1"})
        self.assertEqual(claim["optimizer_updates"], 0)
        self.assertEqual(claim["optimizer_update_budget"], 192)
        self.assertEqual(claim["source_checkpoint_artifact_id"], 14114695344)
        self.assertEqual(claim["source_negative_artifact_id"], 14090383839)
        self.assertEqual(claim["status"], "CONSUMED_BEFORE_BACKGROUND_CONTINUATION")

    def test_claim_or_ambiguity_or_unproven_rerun_stops_before_inputs(self):
        cases = (([[], [{"id": 1}]], "1"), ([[{}, {}], []], "1"), ([[], []], "2"))
        for index, (artifacts, attempt) in enumerate(cases):
            with self.subTest(index=index), self.environment(GITHUB_RUN_ATTEMPT=attempt), \
                 mock.patch.object(C.core, "named_artifacts", side_effect=artifacts), \
                 mock.patch.object(C.core, "github_get", side_effect=AssertionError("must not load inputs")), \
                 mock.patch.object(C, "download_corpus", side_effect=AssertionError("must not download")), \
                 mock.patch.object(C.native, "run_bounded", side_effect=AssertionError("must not execute")):
                with self.assertRaises(ValueError):
                    C.prepare(self.path, self.root / ("state" + str(index)))

    def test_reuse_is_saved_only_and_failed_terminal_never_retrains(self):
        archive = self.archive({"TERMINAL.json": b'{"invented": true}'})
        for failed in (False, True):
            with self.subTest(failed=failed), self.environment(GITHUB_RUN_ATTEMPT="2"), \
                 mock.patch.object(C.core, "named_artifacts", side_effect=[[{"id": 77,
                     "workflow_run": {"id": 999, "head_sha": "b" * 40}}], []]), \
                 mock.patch.object(C.core, "artifact_bytes", return_value=archive), \
                 mock.patch.object(C, "validate_terminal", return_value={"run_id": "999", "data_commit": "b" * 40},
                                   side_effect=ValueError("failed consumed terminal") if failed else None) as saved, \
                 mock.patch.object(C, "download_corpus", side_effect=AssertionError("must not download")), \
                 mock.patch.object(C.core, "github_get", side_effect=AssertionError("must not fetch inputs")), \
                 mock.patch.object(C.native, "run_bounded", side_effect=AssertionError("must not run models")), \
                 mock.patch.object(C.core, "output") as output:
                state = self.root / ("failed" if failed else "success")
                if failed:
                    with self.assertRaisesRegex(ValueError, "failed consumed"):
                        C.prepare(self.path, state)
                    output.assert_not_called()
                else:
                    C.prepare(self.path, state)
                    self.assertIs(output.call_args.kwargs["execute"], False)
                    self.assertEqual(output.call_args.kwargs["prior_artifact_id"], 77)
                    self.assertTrue(output.call_args.kwargs["result_artifact"].startswith("research-v4-background-result-"))
                saved.assert_called_once()

    def test_wrong_repository_event_branch_or_private_stop_before_lookup(self):
        for index, change in enumerate(({"GITHUB_EVENT_NAME": "pull_request"}, {"GITHUB_REF": "refs/heads/feature"},
                                         {"GITHUB_REPOSITORY": "another/repo"})):
            with self.environment(**change), mock.patch.object(C.core, "named_artifacts", side_effect=AssertionError("lookup")):
                with self.assertRaises(ValueError):
                    C.prepare(self.path, self.root / ("state" + str(index)))
        self.write(self.event, {"repository": {"private": True}})
        with self.environment(), self.assertRaises(ValueError):
            C.prepare(self.path, self.root / "private")

    def test_prepare_verifies_four_original_artifacts_and_corpus_before_claim(self):
        review = b'{"invented":"review"}'
        archives = {name: self.archive({"fixture.json": b"{}"}) for name in C.DOWNLOADS}
        archives["review"] = self.archive({C.core.INPUTS["review"]["member"]: review})
        downloads = {name: dict(value, zip_bytes=len(archives[name]), zip_sha256=C.core.sha256(archives[name]))
                     for name, value in C.DOWNLOADS.items()}
        runs = [{"status": "completed", "conclusion": "success", "run_attempt": 1,
                 "head_sha": row["data_commit"], "event": row["event"]} for row in downloads.values()]
        metadata = [{"id": row["artifact_id"], "size_in_bytes": row["zip_bytes"],
                     "workflow_run": {"id": row["run_id"], "head_sha": row["data_commit"]}}
                    for row in downloads.values()]
        state = self.root / "prepared"
        def source_download(target):
            self.assertFalse((state / "claim/CLAIM.json").exists())
            target.write_bytes(b"invented source archive, not audio")
        with self.environment(), mock.patch.dict(C.DOWNLOADS, downloads, clear=True), \
             mock.patch.dict(C.core.INPUTS["review"], {"member_sha256": C.core.sha256(review)}), \
             mock.patch.object(C.core, "named_artifacts", side_effect=[[], []]), \
             mock.patch.object(C.core, "github_get", side_effect=runs) as original_runs, \
             mock.patch.object(C.core, "artifact_metadata", side_effect=metadata) as artifacts, \
             mock.patch.object(C.core, "artifact_bytes", side_effect=list(archives.values())), \
             mock.patch.object(C.native, "validate_checkpoint") as checkpoint, \
             mock.patch.object(C, "validate_negative"), \
             mock.patch.object(C, "download_corpus", side_effect=source_download) as corpus, \
             mock.patch.object(C, "validate_inputs") as inputs, \
             mock.patch.object(C.core, "output") as output:
            C.prepare(self.path, state)
        self.assertEqual(original_runs.call_count, 4)
        self.assertEqual(artifacts.call_count, 4)
        self.assertEqual([call.args[0] for call in artifacts.call_args_list],
                         [14114695344, 13954744587, 14001747172, 14090383839])
        checkpoint.assert_called_once_with(state / "inputs/trained-result")
        corpus.assert_called_once()
        inputs.assert_called_once()
        self.assertEqual((state / "inputs/review.json").read_bytes(), review)
        self.assertEqual(C.core.read_json(state / "claim/CLAIM.json")["optimizer_update_budget"], 192)
        self.assertIs(output.call_args.kwargs["execute"], True)

    def test_execution_runs_one_fixed_entrypoint_then_zero_model_verification(self):
        state, pipeline, script = self.execution_fixture()
        def actual(command, **kwargs):
            self.assertEqual(kwargs["seconds"], 1200)
            self.assertNotIn("GH_TOKEN", kwargs["environment"])
            self.assertEqual(kwargs["environment"]["GITHUB_RUN_ATTEMPT"], "1")
            self.assertEqual(command[-2:], ["--output-dir", str(state / "result" / C.RESULT_DIR)])
            for flag in ("--trained-result-dir", "--tts-zip", "--review-json", "--negative-evidence-dir", "--source-tar"):
                self.assertIn(flag, command)
            self.write(state / "result" / C.RESULT_DIR / "SUMMARY.json",
                       dict(C.WORK_COUNTERS))
            return 0
        with self.environment(), mock.patch.object(C.subprocess, "check_output", side_effect=[C.PIPELINE_COMMIT, "", script.read_bytes()]), \
             mock.patch.object(C, "validate_checkout_sources"), mock.patch.object(C, "validate_inputs") as inputs, \
             mock.patch.object(C.native, "run_bounded", side_effect=actual) as execute, \
             mock.patch.object(C, "verified_source", return_value={"invented": b"x"}), \
             mock.patch.object(C, "run_saved_verifier", return_value={"model_forward_calls": 0}) as saved:
            C.execute(self.path, state, pipeline, Path("/fake/python"))
        execute.assert_called_once()
        saved.assert_called_once()
        self.assertEqual(inputs.call_count, 2)
        execution = C.core.read_json(state / "result/EXECUTION.json")
        self.assertEqual(execution["optimizer_updates"], 192)
        self.assertEqual(execution["native_forward_frames"], 431790)
        C.execution_ok(execution, self.digest)

    def test_failed_started_update_keeps_partial_progress_and_unknown_count_without_retry(self):
        state, pipeline, script = self.execution_fixture()
        partial = state / "result" / C.RESULT_DIR
        def failed(*args, **kwargs):
            self.write(partial / "TRAINING.json", {"optimizer_updates_started": 49, "optimizer_updates_completed": 48,
                                                  "active_training": "invented"})
            (partial / "WEIGHTS.zip").write_bytes(self.archive({"control-gru.npz": b"invented completed-arm checkpoint"}))
            return 1
        with self.environment(), mock.patch.object(C.subprocess, "check_output", side_effect=[C.PIPELINE_COMMIT, "", script.read_bytes()]), \
             mock.patch.object(C, "validate_checkout_sources"), mock.patch.object(C, "validate_inputs"), \
             mock.patch.object(C.native, "run_bounded", side_effect=failed) as execute, \
             mock.patch.object(C, "run_saved_verifier", side_effect=AssertionError("failure must not verify")), \
             mock.patch.object(C, "validate_result", side_effect=AssertionError("partial failure must not require success closure")) as complete:
            with self.assertRaises(ValueError):
                C.execute(self.path, state, pipeline, Path("/fake/python"))
            C.seal(self.path, state)
        execute.assert_called_once()
        complete.assert_not_called()
        terminal = C.core.read_json(state / "result/TERMINAL.json")
        self.assertEqual(terminal["status"], "FAILED")
        self.assertIsNone(terminal["optimizer_updates"])
        self.assertEqual(terminal["optimizer_update_budget"], 192)
        self.assertEqual(C.core.read_json(partial / "TRAINING.json")["optimizer_updates_started"], 49)
        self.assertEqual(C.core.read_json(partial / "TRAINING.json")["optimizer_updates_completed"], 48)
        with zipfile.ZipFile(partial / "WEIGHTS.zip") as archive:
            self.assertEqual(archive.namelist(), ["control-gru.npz"])
            self.assertEqual(archive.read("control-gru.npz"), b"invented completed-arm checkpoint")

    def test_execution_receipt_requires_actual_integer_counts_and_complete_source_verification(self):
        execution = {"schema": "kws-research-v4-background-continuation-execution-v1", "request_sha256": self.digest,
                     "execute_exit_code": 0, "verify_exit_code": 0, **C.WORK_COUNTERS,
                     "optimizer_update_budget": 192,
                     "source_inputs_before_after_verified": True}
        C.execution_ok(execution, self.digest)
        for key, value in (("optimizer_updates", 0), ("optimizer_updates", None), ("optimizer_updates", 192.0),
                           ("native_forward_frames", 431791), ("native_initializations", 77),
                           ("native_finishes", 77), ("native_feed_calls", 431945),
                           ("native_supplied_samples", 69110905), ("verify_exit_code", None),
                           ("source_inputs_before_after_verified", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.execution_ok(dict(execution, **{key: value}), self.digest)

    def test_all_six_trace_files_have_exact_independent_segment_geometry(self):
        path = self.root / "invented.trace"
        with path.open("wb") as stream:
            stream.write(b"EN63ABT1" + struct.pack("<II", 1, 13))
            for index, samples in enumerate(C.SEGMENT_SAMPLES):
                frames = (samples - 400) // 160 + 1
                stream.write(struct.pack("<IIQQ", index, 32, samples, frames))
                stream.write(b"\0" * (frames * 32))
        original = path.read_bytes()
        self.assertEqual(len(original), 2303208)
        self.assertEqual(sum((n - 400) // 160 + 1 for n in C.SEGMENT_SAMPLES) * 6, 431790)
        C.validate_trace_geometry(path)
        for raw in (original[:-1], original + b"x", b"changed!" + original[8:],
                    original[:16] + struct.pack("<IIQQ", 1, 32, 11124244, 69525) + original[40:]):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                C.validate_trace_geometry(path)

    def test_weight_runtime_archives_are_closed_and_never_extract_arbitrary_members(self):
        path = self.root / "WEIGHTS.zip"
        files = {name: b"PK\x03\x04invented" for name in C.WEIGHT_FILES}
        path.write_bytes(self.archive(files))
        self.assertEqual(C.closed_archive(path, C.WEIGHT_FILES, 500000), files)
        for name in ("extra.npz", "../escape.py"):
            path.write_bytes(self.archive(dict(files, **{name: b"x"})))
            with self.assertRaisesRegex(ValueError, "file closure"):
                C.closed_archive(path, C.WEIGHT_FILES, 500000)
        first = next(iter(files))
        path.write_bytes(self.archive(dict(files, **{first: b""})))
        with self.assertRaisesRegex(ValueError, "file closure"):
            C.closed_archive(path, C.WEIGHT_FILES, 500000)

    def test_scientific_no_benefit_is_success_and_does_not_authorize_more_model_work(self):
        state, _, _ = self.execution_fixture()
        result = state / "result"
        self.write(result / "EXECUTION.json", {
            "schema": "kws-research-v4-background-continuation-execution-v1", "request_sha256": self.digest,
            "execute_exit_code": 0, "verify_exit_code": 0, **C.WORK_COUNTERS,
            "optimizer_update_budget": 192,
            "source_inputs_before_after_verified": True})
        self.write(result / C.RESULT_DIR / "SUMMARY.json", {"benefit": "NO_BENEFIT", "public_run": {
            "repository": C.core.REPOSITORY, "head": "b" * 40, "run_id": "999", "run_attempt": "1"}})
        with self.environment(), mock.patch.object(C, "validate_result") as saved, \
             mock.patch.object(C.native, "run_bounded", side_effect=AssertionError("must not execute")):
            C.seal(self.path, state)
            terminal = C.validate_terminal(result, self.request, self.digest)
        self.assertEqual(terminal["status"], "SUCCEEDED")
        self.assertEqual(terminal["optimizer_updates"], 192)
        self.assertEqual(saved.call_count, 2)

    def test_source_replacement_with_rehashed_zip_rejected_before_marker_executes(self):
        files, allowlist = self.source_fixture()
        path, marker = self.root / "SOURCE.zip", self.root / "marker"
        path.write_bytes(self.archive(files))
        with mock.patch.dict(C.SOURCE_SHA256, allowlist, clear=True):
            self.assertEqual(C.verified_source(path), files)
            replacement = dict(files)
            replacement[C.ENTRYPOINT] = ("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n").encode()
            path.write_bytes(self.archive(replacement))
            rewritten_outer_hash = C.core.sha256(path.read_bytes())
            self.assertEqual(len(rewritten_outer_hash), 64)
            with mock.patch.object(C, "run_saved_verifier", side_effect=AssertionError("unverified source must not execute")), \
                 self.assertRaisesRegex(ValueError, "reviewed allowlist"):
                C.run_saved_verifier(self.root, C.verified_source(path))
        self.assertFalse(marker.exists())

    def test_source_closed_paths_reject_missing_extra_duplicate_traversal_and_symlink(self):
        files, allowlist = self.source_fixture()
        path = self.root / "SOURCE.zip"
        for mutation in ("missing", "extra", "duplicate", "traversal", "symlink"):
            selected = dict(files)
            if mutation == "missing":
                selected.pop(C.ENTRYPOINT)
            if mutation in ("extra", "traversal"):
                selected["../escape.py" if mutation == "traversal" else "extra.py"] = b"x"
            with zipfile.ZipFile(path, "w") as archive:
                for name, raw in selected.items():
                    if mutation == "symlink" and name == C.ENTRYPOINT:
                        info = zipfile.ZipInfo(name)
                        info.external_attr = 0o120777 << 16
                        archive.writestr(info, raw)
                    else:
                        archive.writestr(name, raw)
                if mutation == "duplicate":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        archive.writestr(C.ENTRYPOINT, files[C.ENTRYPOINT])
            with self.subTest(case=mutation), mock.patch.dict(C.SOURCE_SHA256, allowlist, clear=True), \
                 self.assertRaisesRegex(ValueError, "path closure"):
                C.verified_source(path)

    def test_source_identity_is_checked_in_checkout_before_any_model_child(self):
        files, allowlist = self.source_fixture()
        tree = self.root / "source"
        for name, raw in files.items():
            path = tree / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        with mock.patch.dict(C.SOURCE_SHA256, allowlist, clear=True):
            C.validate_checkout_sources(tree)
            (tree / C.ENTRYPOINT).write_text("raise RuntimeError('different source')\n")
            with self.assertRaisesRegex(ValueError, "reviewed source"):
                C.validate_checkout_sources(tree)

    def test_saved_reader_is_fixed_stdlib_credential_free_and_hard_bounded(self):
        files, _ = self.source_fixture()
        root = self.root / "result"
        root.mkdir()
        receipt = {"model_forward_calls": 0, "optimizer_updates": 0}
        process = subprocess.CompletedProcess([], 0, json.dumps(receipt), "")
        with self.environment(), mock.patch.object(C.subprocess, "run", return_value=process) as child:
            self.assertEqual(C.run_saved_verifier(root, files), receipt)
        child.assert_called_once()
        command = child.call_args.args[0]
        self.assertEqual(command[1:4], ["-I", "-S", "-B"])
        self.assertEqual(command[-2:], ["--verify-result-dir", str(root.resolve())])
        self.assertEqual(len(command), 7)
        self.assertEqual(child.call_args.kwargs["timeout"], 180)
        for name in ("GH_TOKEN", "GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_REPOSITORY", "AWS_ACCESS_KEY_ID"):
            self.assertNotIn(name, child.call_args.kwargs["env"])

    def test_saved_reader_failure_timeout_and_unexpected_output_fail(self):
        files, _ = self.source_fixture()
        for process in (subprocess.CompletedProcess([], 1, "{}", "error"),
                        subprocess.CompletedProcess([], 0, "{}", "warning"),
                        subprocess.CompletedProcess([], 0, "x" * 65536, "")):
            with mock.patch.object(C.subprocess, "run", return_value=process), self.assertRaises(ValueError):
                C.run_saved_verifier(self.root, files)
        with mock.patch.object(C.subprocess, "run", side_effect=subprocess.TimeoutExpired("saved", 180)) as child:
            with self.assertRaises(subprocess.TimeoutExpired):
                C.run_saved_verifier(self.root, files)
        child.assert_called_once()


if __name__ == "__main__":
    unittest.main()
