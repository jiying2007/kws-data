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

SPEC = importlib.util.spec_from_file_location("projection_consumer_test",
                                            Path(__file__).with_name("research_v4_projection_continuation.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class ProjectionBoundaries(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(C, "PIPELINE_COMMIT", "e" * 40).start()
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
        self.assertEqual(C.FIXED["profile"]["optimizer_updates"], 96)
        self.assertEqual(C.FIXED["profile"]["native_eval_frames"], 575720)
        self.assertEqual(C.FIXED["profile"]["steps_per_arm"], 48)
        self.assertEqual(C.FIXED["profile"]["training_arms"], 2)
        for key, value in (("pipeline_commit", "a" * 40), ("pipeline_commit", "main"),
                           ("max_execution_seconds", 1201), ("max_verification_seconds", 181),
                           ("entrypoint", "arbitrary.py"), ("new_audio_generated", True),
                           ("human_gold", True), ("reason", "")):
            bad = copy.deepcopy(self.request)
            bad[key] = value
            self.write(self.path, bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.load_request(self.path)
        for key, value in (("optimizer_updates", False), ("optimizer_updates", 97),
                           ("training_arms", 3), ("pool_time_major", 1), ("threshold", 0.79),
                           ("natural_loss_weight", 1), ("train_microbatch_natural", 128),
                           ("native_eval_frames", 575720.0), ("training_arm_seconds", 181)):
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
        self.assertEqual(claim["optimizer_update_budget"], 96)
        self.assertEqual(claim["source_checkpoint_artifact_id"], 14114695344)
        self.assertEqual(claim["source_negative_artifact_id"], 14090383839)
        self.assertEqual(claim["status"], "CONSUMED_BEFORE_GRADIENT_PROJECTION")

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
                    self.assertTrue(output.call_args.kwargs["result_artifact"].startswith("research-v4-projection-result-"))
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

    def test_prepare_verifies_five_original_artifacts_and_corpus_before_claim(self):
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
             mock.patch.object(C, "download_corpus", side_effect=source_download) as corpus, \
             mock.patch.object(C, "validate_inputs") as inputs, \
             mock.patch.object(C.core, "output") as output:
            C.prepare(self.path, state)
        self.assertEqual(original_runs.call_count, 5)
        self.assertEqual(artifacts.call_count, 5)
        self.assertEqual([call.args[0] for call in artifacts.call_args_list],
                         [14114695344, 13954744587, 14001747172, 14090383839, 14149230666])
        corpus.assert_called_once()
        inputs.assert_called_once()
        self.assertEqual((state / "inputs/review.json").read_bytes(), review)
        self.assertEqual(C.core.read_json(state / "claim/CLAIM.json")["optimizer_update_budget"], 96)
        self.assertIs(output.call_args.kwargs["execute"], True)

    def test_execution_runs_one_fixed_entrypoint_then_zero_model_verification(self):
        state, pipeline, script = self.execution_fixture()
        def actual(command, **kwargs):
            self.assertEqual(kwargs["seconds"], 1200)
            self.assertNotIn("GH_TOKEN", kwargs["environment"])
            self.assertEqual(kwargs["environment"]["GITHUB_RUN_ATTEMPT"], "1")
            self.assertEqual(command[-2:], ["--output-dir", str(state / "result" / C.RESULT_DIR)])
            for flag in ("--trained-result-dir", "--tts-zip", "--review-json", "--negative-evidence-dir", "--source-tar", "--background-result-dir"):
                self.assertIn(flag, command)
            prior_index = command.index("--background-result-dir")
            self.assertEqual(command[prior_index + 1],
                             str(state / "inputs/background-result" / C.background.RESULT_DIR))
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
        self.assertEqual(execution["optimizer_updates"], 96)
        self.assertEqual(execution["native_forward_frames"], 575720)
        C.execution_ok(execution, self.digest)

    def test_failed_started_update_keeps_partial_progress_and_unknown_count_without_retry(self):
        state, pipeline, script = self.execution_fixture()
        partial = state / "result" / C.RESULT_DIR
        def failed(*args, **kwargs):
            self.write(partial / "TRAINING.json", {"optimizer_updates_started": 49, "optimizer_updates_completed": 48,
                                                  "active_training": "invented"})
            (partial / "WEIGHTS.zip").write_bytes(self.archive({"projected-gru.npz": b"invented completed-arm checkpoint"}))
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
        self.assertEqual(terminal["optimizer_update_budget"], 96)
        self.assertEqual(C.core.read_json(partial / "TRAINING.json")["optimizer_updates_started"], 49)
        self.assertEqual(C.core.read_json(partial / "TRAINING.json")["optimizer_updates_completed"], 48)
        with zipfile.ZipFile(partial / "WEIGHTS.zip") as archive:
            self.assertEqual(archive.namelist(), ["projected-gru.npz"])
            self.assertEqual(archive.read("projected-gru.npz"), b"invented completed-arm checkpoint")

    def test_execution_receipt_requires_actual_integer_counts_and_complete_source_verification(self):
        execution = {"schema": "kws-research-v4-projection-continuation-execution-v1", "request_sha256": self.digest,
                     "execute_exit_code": 0, "verify_exit_code": 0, **C.WORK_COUNTERS,
                     "optimizer_update_budget": 96,
                     "source_inputs_before_after_verified": True}
        C.execution_ok(execution, self.digest)
        for key, value in (("optimizer_updates", 0), ("optimizer_updates", None), ("optimizer_updates", 96.0),
                           ("training_forward_calls", 479), ("training_padded_forward_frames", 4669823),
                           ("native_forward_frames", 575721), ("native_initializations", 103),
                           ("native_finishes", 103), ("native_feed_calls", 575927),
                           ("native_supplied_samples", 92147873), ("verify_exit_code", None),
                           ("source_inputs_before_after_verified", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.execution_ok(dict(execution, **{key: value}), self.digest)

    def test_all_eight_trace_members_have_exact_independent_segment_geometry(self):
        stream = io.BytesIO()
        stream.write(b"EN63ABT1" + struct.pack("<II", 1, 13))
        for index, samples in enumerate(C.background.SEGMENT_SAMPLES):
            frames = (samples - 400) // 160 + 1
            stream.write(struct.pack("<IIQQ", index, 32, samples, frames))
            stream.write(b"\0" * (frames * 32))
        raw = stream.getvalue()
        path = self.root / "TRACES.zip"
        files = dict.fromkeys(C.TRACE_FILES, raw)
        path.write_bytes(self.archive(files))
        self.assertEqual(len(raw), 2303208)
        self.assertEqual(len(files), 8)
        self.assertEqual(sum((n - 400) // 160 + 1 for n in C.background.SEGMENT_SAMPLES) * 8, 575720)
        C.validate_traces(path)
        first = sorted(files)[0]
        for bad in (raw[:-1], raw + b"x", b"changed!" + raw[8:],
                    raw[:16] + struct.pack("<IIQQ", 1, 32, 11124244, 69525) + raw[40:]):
            path.write_bytes(self.archive(dict(files, **{first: bad})))
            with self.assertRaises(ValueError):
                C.validate_traces(path)
        for name in ("extra.trace", "../escape.trace"):
            path.write_bytes(self.archive(dict(files, **{name: raw})))
            with self.assertRaises(ValueError):
                C.validate_traces(path)

    def test_weight_runtime_gradient_archives_are_closed_and_never_extract_arbitrary_members(self):
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
            "schema": "kws-research-v4-projection-continuation-execution-v1", "request_sha256": self.digest,
            "execute_exit_code": 0, "verify_exit_code": 0, **C.WORK_COUNTERS,
            "optimizer_update_budget": 96,
            "source_inputs_before_after_verified": True})
        self.write(result / C.RESULT_DIR / "SUMMARY.json", {"benefit": "NO_BENEFIT", "public_run": {
            "repository": C.core.REPOSITORY, "head": "b" * 40, "run_id": "999", "run_attempt": "1"}})
        with self.environment(), mock.patch.object(C, "validate_result") as saved, \
             mock.patch.object(C.native, "run_bounded", side_effect=AssertionError("must not execute")):
            C.seal(self.path, state)
            terminal = C.validate_terminal(result, self.request, self.digest)
        self.assertEqual(terminal["status"], "SUCCEEDED")
        self.assertEqual(terminal["optimizer_updates"], 96)
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

    def test_old_request_and_source_contract_remain_immutable(self):
        old = copy.deepcopy(C.background.FIXED)
        old_source = dict(C.background.SOURCE_SHA256)
        self.assertEqual(C.FIXED["profile"]["variants"], ["baseline", "control", "treatment", "projected"])
        self.assertEqual(C.FIXED["profile"]["training_group_seconds"], 360)
        self.assertEqual(C.FIXED["profile"]["training_forward_calls"], 480)
        self.assertEqual(len(C.SOURCE_SHA256), 27)
        self.assertEqual(len(old_source), 24)
        self.assertEqual({name: C.SOURCE_SHA256[name] for name in old_source}, old_source)
        C.load_request(self.path)
        self.assertEqual(C.background.FIXED, old)
        self.assertEqual(C.background.SOURCE_SHA256, old_source)
        old_request = copy.deepcopy(old)
        old_request.update(pipeline_commit=C.background.PIPELINE_COMMIT, reason="Invented prior fixture.")
        C.background.request_value(json.dumps(old_request).encode())
        with self.assertRaises(ValueError):
            C.request_value(json.dumps(old_request).encode())
        self.assertIs(C.FIXED["formal_training_admitted"], False)
        self.assertEqual(len(C.RESULT_FILES), 13)

    def test_frozen_background_members_and_unchanged_old_validator_are_both_required(self):
        result = self.root / "prior-background"
        inner = result / C.background.RESULT_DIR
        inner.mkdir(parents=True)
        paths = {
            "request_sha256": result / "REQUEST.json", "terminal_sha256": result / "TERMINAL.json",
            "summary_sha256": inner / "SUMMARY.json", "inputs_sha256": inner / "INPUTS.json",
            "training_sha256": inner / "TRAINING.json", "replay_sha256": inner / "REPLAY.json"}
        for key, path in paths.items():
            path.write_bytes(("invented " + key).encode())
        weights = {name: ("invented " + name).encode() for name in C.PRIOR["weights_sha256"]}
        archive = inner / "WEIGHTS.zip"
        archive.write_bytes(self.archive(weights))
        pins = {key: C.core.sha256(path.read_bytes()) for key, path in paths.items()}
        pins.update(weights_zip_sha256=C.core.sha256(archive.read_bytes()),
                    weights_sha256={name: C.core.sha256(raw) for name, raw in weights.items()})
        terminal = {"run_id": str(C.PRIOR["run_id"]), "run_attempt": "1",
                    "data_commit": C.PRIOR["data_commit"], "pipeline_commit": C.PRIOR["pipeline_commit"]}
        with mock.patch.dict(C.PRIOR, pins), \
             mock.patch.object(C.background, "request_value", return_value={"invented": True}), \
             mock.patch.object(C.background, "validate_terminal", return_value=terminal) as frozen:
            C.validate_background(result)
            frozen.assert_called_once_with(result, {"invented": True}, pins["request_sha256"])
            for path in paths.values():
                original = path.read_bytes()
                path.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "artifact member changed"):
                    C.validate_background(result)
                path.write_bytes(original)
            first = sorted(weights)[0]
            archive.write_bytes(self.archive(dict(weights, **{first: b"replaced frozen checkpoint"})))
            with mock.patch.dict(C.PRIOR, {"weights_zip_sha256": C.core.sha256(archive.read_bytes())}), \
                 self.assertRaisesRegex(ValueError, "frozen comparator checkpoints changed"):
                C.validate_background(result)
            archive.write_bytes(self.archive(weights))
            frozen.side_effect = ValueError("prior terminal was not successful")
            with self.assertRaisesRegex(ValueError, "prior terminal"):
                C.validate_background(result)

    def test_existing_inputs_and_prior_background_are_rechecked_independently(self):
        state = self.root / "inputs-state"
        with mock.patch.object(C.background, "validate_inputs") as original, \
             mock.patch.object(C, "validate_file") as archive, \
             mock.patch.object(C, "validate_background") as prior:
            C.validate_inputs(state)
        original.assert_called_once_with(state)
        archive.assert_called_once_with(state / "inputs/background.zip", C.PRIOR["zip_bytes"],
                                        C.PRIOR["zip_sha256"], "frozen background result archive")
        prior.assert_called_once_with(state / "inputs/background-result")

    def test_gradient_archive_closes_two_bounded_npz_members(self):
        self.assertEqual(C.GRADIENT_FILES, {"gru-gradients.npz", "dsc_tcn-gradients.npz"})
        path = self.root / "GRADIENTS.zip"
        members = dict.fromkeys(C.GRADIENT_FILES, b"PK\x03\x04invented tape")
        path.write_bytes(self.archive(members))
        self.assertEqual(C.closed_archive(path, C.GRADIENT_FILES, 8 * 1024 * 1024), members)
        for bad in (dict(members, **{"extra.npz": b"x"}),
                    {name: b"" if name == "gru-gradients.npz" else raw for name, raw in members.items()}):
            path.write_bytes(self.archive(bad))
            with self.assertRaises(ValueError):
                C.closed_archive(path, C.GRADIENT_FILES, 8 * 1024 * 1024)

    def test_post_execution_checkout_mutation_fails_before_success(self):
        state, pipeline, script = self.execution_fixture()
        def actual(*args, **kwargs):
            self.write(state / "result" / C.RESULT_DIR / "SUMMARY.json", dict(C.WORK_COUNTERS))
            return 0
        with self.environment(), \
             mock.patch.object(C.subprocess, "check_output", side_effect=[C.PIPELINE_COMMIT, "", script.read_bytes()]), \
             mock.patch.object(C, "validate_checkout_sources", side_effect=[None, ValueError("source changed after run")]) as source, \
             mock.patch.object(C, "validate_inputs"), \
             mock.patch.object(C.native, "run_bounded", side_effect=actual), \
             mock.patch.object(C, "verified_source", return_value={"invented": b"x"}), \
             mock.patch.object(C, "run_saved_verifier", return_value={"model_forward_calls": 0}):
            with self.assertRaises(ValueError):
                C.execute(self.path, state, pipeline, Path("/fake/python"))
        self.assertEqual(source.call_count, 2)
        execution = C.core.read_json(state / "result/EXECUTION.json")
        self.assertIs(execution["source_inputs_before_after_verified"], False)
        self.assertIn("source changed after run", execution["error"])

    def test_complete_invented_result_binds_packed_evidence_and_replay(self):
        profile = copy.deepcopy(C.FIXED["profile"])
        baseline = b"invented fixed baseline header"
        profile["baseline_header_sha256"] = C.core.sha256(baseline)
        self.request["profile"] = profile
        self.write(self.path, self.request)
        self.digest = C.core.sha256(self.path.read_bytes())
        state, _, _ = self.execution_fixture()
        result, root = state / "result", state / "result" / C.RESULT_DIR
        root.mkdir()
        for name in ("EXECUTION.json", "execute.log", "verify.log", "runtime.txt"):
            (result / name).write_bytes(b"invented outer evidence")
        source, allowlist = self.source_fixture()
        (root / "SOURCE.zip").write_bytes(self.archive(source))
        trace = io.BytesIO()
        trace.write(b"EN63ABT1" + struct.pack("<II", 1, 13))
        for index, samples in enumerate(C.background.SEGMENT_SAMPLES):
            frames = (samples - 400) // 160 + 1
            trace.write(struct.pack("<IIQQ", index, 32, samples, frames))
            trace.write(b"\0" * (frames * 32))
        (root / "TRACES.zip").write_bytes(self.archive(dict.fromkeys(C.TRACE_FILES, trace.getvalue())))
        (root / "GRADIENTS.zip").write_bytes(self.archive(dict.fromkeys(C.GRADIENT_FILES, b"PK\x03\x04invented tape")))
        weights = {name: b"PK\x03\x04invented " + name.encode() for name in C.WEIGHT_FILES}
        (root / "WEIGHTS.zip").write_bytes(self.archive(weights))
        fixed_weights = {name: C.core.sha256(weights[name]) for name in C.PRIOR["weights_sha256"]}
        runtime = {name: b"\x7fELFinvented" if name.endswith(".bin") else b"invented header"
                   for name in C.RUNTIME_FILES}
        runtime["baseline.h"] = baseline
        (root / "RUNTIME.zip").write_bytes(self.archive(runtime))
        run = {"repository": C.core.REPOSITORY, "head": "b" * 40,
               "run_id": "999", "run_attempt": "1"}
        claim = json.dumps(C.claim_record(self.request, self.digest, run))
        self.write(root / "INPUTS.json", {"request_raw_utf8": self.path.read_text(),
                   "request_sha256": self.digest, "claim_raw_utf8": claim,
                   "claim_sha256": C.core.sha256(claim.encode())})
        self.write(root / "BUILD.json", {"public_run": run, "pipeline_head": C.PIPELINE_COMMIT,
                                        "source_sha256": allowlist})
        self.write(root / "TRAINING.json", {"optimizer_updates_started": 96,
                   "optimizer_updates_completed": 96, "active_training": None})
        self.write(root / "SUMMARY.json", {"schema": "english63-projection-ab-result-v1",
                   "experiment": C.EXPERIMENT, "request_sha256": self.digest,
                   "public_run": run, "pipeline_head": C.PIPELINE_COMMIT, **C.WORK_COUNTERS,
                   "new_independent_audio_seconds": 0, "new_tts_asr_ipa_calls": 0,
                   **dict.fromkeys(C.FALSE_FIELDS, False)})
        for name in ("PROTOCOL.json", "MEASUREMENTS.json"):
            self.write(root / name, {"invented": True})
        (root / "native.stderr").write_bytes(b"")
        hashes = C.core.file_manifest(root)
        replay = {"schema": "english63-projection-ab-replay-v1", "status": "VERIFIED_SAVED_PROJECTION_AB",
                  "files": {name: {"bytes": (root / name).stat().st_size, "sha256": digest}
                            for name, digest in hashes.items()},
                  "files_sha256": hashes, "summary_sha256": hashes["SUMMARY.json"],
                  "gradient_tape_sha256": hashes["GRADIENTS.zip"], "model_forward_calls": 0,
                  "optimizer_updates": 0, "new_audio_asr_calls": 0,
                  "formal_training_admitted": False, "model_promotion_allowed": False}
        self.write(root / "REPLAY.json", replay)
        with mock.patch.dict(C.FIXED, {"profile": profile}), \
             mock.patch.dict(C.PRIOR, {"weights_sha256": fixed_weights}), \
             mock.patch.dict(C.SOURCE_SHA256, allowlist, clear=True), \
             mock.patch.object(C, "run_saved_verifier", return_value=replay) as saved:
            C.validate_result(result)
            saved.assert_called_once_with(root, source)
            self.assertEqual(len(C.core.file_manifest(result)), 18)
            changed = dict(replay, gradient_tape_sha256="0" * 64)
            self.write(root / "REPLAY.json", changed)
            with self.assertRaisesRegex(ValueError, "receipt manifest"):
                C.validate_result(result)
            self.write(root / "REPLAY.json", replay)
            first = sorted(fixed_weights)[0]
            (root / "WEIGHTS.zip").write_bytes(self.archive(dict(weights, **{first: b"PK\x03\x04changed"})))
            with self.assertRaisesRegex(ValueError, "changed a frozen comparator"):
                C.validate_result(result)


if __name__ == "__main__":
    unittest.main()
