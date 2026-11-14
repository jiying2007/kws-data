#!/usr/bin/env python3
"""Stdlib-only invented evidence: never compile or forward project models here."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

SPEC = importlib.util.spec_from_file_location("native_c", Path(__file__).with_name("research_v4_native_c.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class NativeConsumerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = copy.deepcopy(C.FIXED) | {
            "request_id": "native-fixture-v1", "pipeline_commit": "a" * 40,
            "reason": "Invented source-only fixture; no C compilation or model calls.",
        }
        self.path = self.root / "native-fixture-v1.json"
        self.write_request()

    def write_request(self):
        self.path.write_text(json.dumps(self.request) + "\n")
        self.digest = C.core.sha256(self.path.read_bytes())

    def write(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

    def refresh_verification(self, result):
        comparison = result / "comparison"
        manifest = C.core.file_manifest(comparison)
        manifest.pop("REPLAY.json", None)
        self.write(comparison / "REPLAY.json", {
            "schema": "english63-native-c-saved-evidence-verification-v1",
            "status": "VERIFIED_SAVED_C_COMPARISON_ARTIFACTS",
            "summary_sha256": manifest["SUMMARY.json"], "files_sha256": manifest,
            "optimizer_updates": 0, "model_forward_calls": 0, "new_audio_asr_calls": 0,
            "formal_training_admitted": False, "model_promotion_allowed": False,
        })

    def comparison_fixture(self):
        state = self.root / "comparison-state"
        result = state / "result"
        comparison = result / "comparison"
        comparison.mkdir(parents=True)
        for name in C.COMPARISON_FILES - {"SUMMARY.json", "REPLAY.json"}:
            path = comparison / name
            path.parent.mkdir(parents=True, exist_ok=True)
            prefix = b"EN63TRC1" if name.endswith(".trace") else b"\x7fELF" if name == "en63_replay" else b""
            path.write_bytes(prefix + b"invented offline fixture; never executed")
        models = []
        for kind, digest in C.CHECKPOINT["weights_sha256"].items():
            models.append({
                "kind": kind, "source_weights_sha256": digest, "isolated_clips": 63,
                "isolated_frames": 13010, "continuous_frames": 13134,
                "chunk_plans": ["160", "mixed"], "reference_forward_sequences": 64,
                "chunk_probabilities_bit_identical": True, "chunk_events_identical": True,
                "python_c_events_identical": True, "eof_padding_samples": 0, "eof_events": 0,
                "continuous_state_resets": 0, "resource_profile_scope": "native-linux-host-not-ssc305",
                "max_feature_abs_error": 0.00001, "max_logit_abs_error": 0.00001,
                "max_probability_abs_error": 0.00001,
            })
        summary = {
            "schema": "english63-saved-fp32-native-c-stream-parity-v1",
            "status": "MEASURED_NATIVE_C_STREAM_PARITY_NOT_KWS_QUALIFICATION",
            "source_audio_clips": 63, "source_pcm_samples": 2101760,
            "source_training_run_id": C.CHECKPOINT["run_id"], "optimizer_updates": 0,
            "new_tts_asr_ipa_calls": 0, "formal_training_admitted": False,
            "human_gold": False, "model_promotion_allowed": False,
            "model_selection_made": False, "threshold_selection": False,
            "numeric_tolerances": {"features": 0.0002, "logits": 0.0002, "probabilities": 0.0002},
            "models": models,
        }
        summary.update({key: "NOT_RUN" for key in (
            "independent_english_frr", "continuous_negative_far_per_hour", "word_end_latency", "ssc305_board_profile")})
        self.write(comparison / "SUMMARY.json", summary)
        self.refresh_verification(result)
        (result / "REQUEST.json").write_bytes(self.path.read_bytes())
        self.write(result / "EXECUTION.json", {
            "request_sha256": self.digest, "compare_exit_code": 0, "verify_exit_code": 0,
            "source_checkpoint_before_after_verified": True, "optimizer_updates": 0,
        })
        (result / "runtime.txt").write_text("Invented fixture runtime.\n")
        return state, result

    def seal(self, state):
        with contextlib.redirect_stdout(io.StringIO()):
            C.seal(self.path, state)

    def environment(self, attempt="1"):
        event = self.root / "event.json"
        self.write(event, {"repository": {"private": False}})
        return mock.patch.dict(os.environ, {
            "GITHUB_EVENT_PATH": str(event), "GITHUB_EVENT_NAME": "push",
            "GITHUB_REPOSITORY": C.core.REPOSITORY, "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "fixture", "GITHUB_RUN_ATTEMPT": attempt,
        })

    def archive(self, result):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            for path in result.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(result).as_posix())
        return content.getvalue()

    def execution_fixture(self):
        state = self.root / "execution"
        (state / "claim").mkdir(parents=True)
        (state / "result").mkdir()
        (state / "result/REQUEST.json").write_bytes(self.path.read_bytes())
        self.write(state / "claim/CLAIM.json", {"request_sha256": self.digest})
        pipeline = self.root / "pipeline"
        script = pipeline / C.ENTRYPOINT
        script.parent.mkdir(parents=True)
        script.write_text("# invented fixture, never executed\n")
        return state, pipeline

    def test_request_rejects_training_free_commands_and_changed_checkpoints(self):
        self.assertEqual(C.load_request(self.path)[2], self.digest)
        mutations = (("optimizer_updates", 48), ("entrypoint", C.core.ENTRYPOINT),
                     ("pipeline_repository", "someone/else"), ("cpu_threads", 2),
                     ("chunk_plans", ["mixed"]), ("numeric_abs_tolerance", 1.0),
                     ("pipeline_commit", "main"), ("trained_checkpoint", {}),
                     ("max_execution_seconds", 36000), ("model_promotion_allowed", True))
        for key, value in mutations:
            with self.subTest(key=key):
                original = self.request[key]
                self.request[key] = value
                self.write_request()
                with self.assertRaises(ValueError):
                    C.load_request(self.path)
                self.request[key] = original

    def test_native_template_cannot_be_executed_with_placeholder_sha(self):
        template = Path(__file__).parents[1] / ".github/research-v4/native-c-request.template.json"
        value = json.loads(template.read_bytes())
        path = self.root / (value["request_id"] + ".json")
        path.write_bytes(template.read_bytes())
        with self.assertRaises(ValueError):
            C.load_request(path)

    def test_consumed_online_request_is_unchanged_and_dispatch_cannot_train(self):
        self.assertEqual(C.core.sha256(C.SOURCE_REQUEST.read_bytes()), C.CHECKPOINT["request_sha256"])
        self.assertIsNone(C.core.native_consumer(C.SOURCE_REQUEST))
        self.assertEqual(C.core.native_consumer(self.path).ENTRYPOINT, C.ENTRYPOINT)
        consumer = mock.Mock()
        argv = ["executor", "execute", "--request", str(self.path), "--state", str(self.root / "state")]
        with mock.patch.object(C.core.sys, "argv", argv), \
             mock.patch.object(C.core, "native_consumer", return_value=consumer), \
             mock.patch.object(C.core, "execute", side_effect=AssertionError("must never train")) as training:
            C.core.main()
        consumer.dispatch.assert_called_once()
        training.assert_not_called()

    def test_complete_evidence_can_seal_and_altered_trace_cannot_reuse(self):
        state, result = self.comparison_fixture()
        self.seal(state)
        self.assertEqual(C.validate_terminal(result, self.request, self.digest)["status"], "SUCCEEDED")
        (result / "comparison/gru-continuous-mixed.trace").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "manifest"):
            C.validate_terminal(result, self.request, self.digest)

    def test_original_success_validator_and_exact_checkpoint_pins_are_both_required(self):
        result = self.root / "original-result"
        (result / "model").mkdir(parents=True)
        (result / "TERMINAL.json").write_bytes(b"invented original terminal")
        weights = {}
        for kind in ("gru", "dsc_tcn"):
            path = result / "model" / (kind + "-weights.npz")
            path.write_bytes(("invented original checkpoint " + kind).encode())
            weights[kind] = C.core.sha256(path.read_bytes())
        terminal = {"run_id": str(C.CHECKPOINT["run_id"]), "run_attempt": "1",
                    "data_commit": C.CHECKPOINT["data_commit"]}
        fixture_pins = {"terminal_sha256": C.core.sha256((result / "TERMINAL.json").read_bytes()),
                        "weights_sha256": weights}
        with mock.patch.dict(C.CHECKPOINT, fixture_pins), \
             mock.patch.object(C.core, "validate_terminal", return_value=terminal) as original:
            C.validate_checkpoint(result)
            self.assertEqual(original.call_args.args[0], result)
            self.assertEqual(original.call_args.args[2], C.CHECKPOINT["request_sha256"])
            (result / "model/gru-weights.npz").write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "existing checkpoint changed"):
                C.validate_checkpoint(result)
        with mock.patch.object(C.core, "validate_terminal", side_effect=ValueError("prior execution failed")), \
             self.assertRaisesRegex(ValueError, "prior execution failed"):
            C.validate_checkpoint(result)

    def test_missing_or_fake_native_evidence_cannot_succeed(self):
        state, result = self.comparison_fixture()
        binary = result / "comparison/en63_replay"
        binary.write_bytes(b"not a compiled ELF")
        self.refresh_verification(result)
        with self.assertRaisesRegex(ValueError, "executable"):
            C.validate_comparison(result)
        binary.write_bytes(b"\x7fELFfixture")
        (result / "comparison/gru-isolated-160.trace").unlink()
        self.refresh_verification(result)
        self.seal(state)
        self.assertEqual(C.core.read_json(result / "TERMINAL.json")["status"], "FAILED")

    def test_numeric_nonfinite_padding_and_false_event_parity_are_rejected(self):
        _, result = self.comparison_fixture()
        path = result / "comparison/SUMMARY.json"
        original = C.core.read_json(path)
        mutations = (("max_feature_abs_error", float("nan")), ("max_logit_abs_error", -1.0),
                     ("max_probability_abs_error", 0.00021), ("eof_padding_samples", 160),
                     ("eof_events", 1), ("python_c_events_identical", False),
                     ("continuous_state_resets", 63), ("source_weights_sha256", "0" * 64))
        for key, value in mutations:
            with self.subTest(key=key):
                summary = copy.deepcopy(original)
                summary["models"][0][key] = value
                self.write(path, summary)
                self.refresh_verification(result)
                with self.assertRaises(ValueError):
                    C.validate_comparison(result)

    def test_new_optimizer_work_or_board_qualification_cannot_be_reported(self):
        _, result = self.comparison_fixture()
        path = result / "comparison/SUMMARY.json"
        original = C.core.read_json(path)
        for key, value in (("optimizer_updates", 1), ("model_selection_made", True),
                           ("threshold_selection", True), ("ssc305_board_profile", "PASS")):
            with self.subTest(key=key):
                summary = copy.deepcopy(original)
                summary[key] = value
                self.write(path, summary)
                self.refresh_verification(result)
                with self.assertRaises(ValueError):
                    C.validate_comparison(result)

    def test_success_cache_has_no_input_download_compile_or_forward(self):
        state, result = self.comparison_fixture()
        self.seal(state)
        with self.environment(attempt="2"), \
             mock.patch.object(C.core, "named_artifacts", side_effect=[[{"id": 123}], [{"id": 122}]]), \
             mock.patch.object(C.core, "artifact_bytes", return_value=self.archive(result)), \
             mock.patch.object(C.core, "github_get", side_effect=AssertionError("no new input API")), \
             mock.patch.object(C, "execute", side_effect=AssertionError("no model execution")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            C.prepare(self.path, self.root / "reused-state")
        self.assertIn("execute=false", output.getvalue())
        self.assertFalse((self.root / "reused-state/inputs").exists())

    def test_failed_or_unfinished_attempt_cannot_repeat(self):
        for suffix, attempt, receipts in (("claimed", "1", [[], [{"id": 1}]]), ("lost", "2", [[], []])):
            with self.subTest(suffix=suffix), self.environment(attempt), \
                 mock.patch.object(C.core, "named_artifacts", side_effect=receipts), \
                 mock.patch.object(C.core, "github_get", side_effect=AssertionError("no input download")), \
                 self.assertRaises(ValueError):
                C.prepare(self.path, self.root / suffix)
        state, result = self.comparison_fixture()
        execution = C.core.read_json(result / "EXECUTION.json")
        execution["compare_exit_code"] = 1
        self.write(result / "EXECUTION.json", execution)
        self.seal(state)
        with self.environment(attempt="2"), \
             mock.patch.object(C.core, "named_artifacts", side_effect=[[{"id": 123}], []]), \
             mock.patch.object(C.core, "artifact_bytes", return_value=self.archive(result)), \
             self.assertRaisesRegex(ValueError, "prior native C attempt failed"):
            C.prepare(self.path, self.root / "failed-cache")

    def test_timeout_kills_compiler_and_native_process_group(self):
        child = mock.MagicMock()
        child.__enter__.return_value = child
        child.pid = 1234
        child.wait.side_effect = [subprocess.TimeoutExpired("fixture", 1), 0]
        with mock.patch.object(C.subprocess, "Popen", return_value=child) as launch, \
             mock.patch.object(C.os, "killpg") as kill, self.assertRaises(subprocess.TimeoutExpired):
            C.run_bounded(["invented"], pipeline=self.root, environment={}, log=io.BytesIO(), seconds=1)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        kill.assert_called_once_with(1234, signal.SIGKILL)

    def test_execution_only_uses_new_fixed_cli_and_checks_source_before_after(self):
        state, pipeline = self.execution_fixture()
        with mock.patch.object(C.subprocess, "check_output", side_effect=["a" * 40, ""]), \
             mock.patch.object(C, "validate_checkpoint") as checkpoint, \
             mock.patch.object(C, "run_bounded", return_value=0) as run, \
             mock.patch.dict(os.environ, {"GH_TOKEN": "fixture-secret", "ACTIONS_RUNTIME_TOKEN": "fixture-runtime"}):
            C.execute(self.path, state, pipeline, Path("/fake/venv/bin/python"))
        self.assertEqual(checkpoint.call_count, 2)
        self.assertEqual(run.call_count, 2)
        first, second = run.call_args_list
        self.assertEqual(first.args[0][3], str(pipeline / C.ENTRYPOINT))
        self.assertIn("--output-dir", first.args[0])
        self.assertIn("--verify-result-dir", second.args[0])
        self.assertIn("--trained-result-dir", first.args[0])
        self.assertNotIn(str(pipeline / C.core.ENTRYPOINT), first.args[0])
        self.assertEqual(first.kwargs["seconds"], 1200)
        self.assertEqual(second.kwargs["seconds"], 300)
        self.assertNotIn("GH_TOKEN", first.kwargs["environment"])
        self.assertNotIn("ACTIONS_RUNTIME_TOKEN", first.kwargs["environment"])
        self.assertTrue(C.core.read_json(state / "result/EXECUTION.json")["source_checkpoint_before_after_verified"])

    def test_post_execution_source_change_is_failure_even_after_zero_exit_codes(self):
        state, pipeline = self.execution_fixture()
        with mock.patch.object(C.subprocess, "check_output", side_effect=["a" * 40, ""]), \
             mock.patch.object(C, "validate_checkpoint", side_effect=[None, ValueError("source weights changed")]), \
             mock.patch.object(C, "run_bounded", return_value=0), self.assertRaises(ValueError):
            C.execute(self.path, state, pipeline, Path("/fake/venv/bin/python"))
        execution = C.core.read_json(state / "result/EXECUTION.json")
        self.assertFalse(execution["source_checkpoint_before_after_verified"])
        self.assertEqual(execution["optimizer_updates"], 0)
        self.assertIn("source weights changed", execution["error"])
        self.seal(state)
        self.assertEqual(C.core.read_json(state / "result/TERMINAL.json")["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
