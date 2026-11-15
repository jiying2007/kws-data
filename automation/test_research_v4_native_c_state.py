#!/usr/bin/env python3
"""Invented JSON and mocked execution only; no original audio or model calls."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

SPEC = importlib.util.spec_from_file_location("state_consumer", Path(__file__).with_name("research_v4_native_c_state.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class StateConsumerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.header = b"invented analytic header; no project weights\n"
        patcher = mock.patch.dict(C.BASELINE, {"header_sha256": C.core.sha256(self.header)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.request = copy.deepcopy(C.FIXED) | {
            "request_id": "state-fixture-v1", "pipeline_commit": "a" * 40,
            "reason": "Invented JSON fixture; no audio decoding or native model execution.",
        }
        self.path = self.root / "state-fixture-v1.json"
        self.write(self.path, self.request)
        self.digest = C.core.sha256(self.path.read_bytes())

    def write(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

    def fixture(self):
        state = self.root / "state"
        root = state / "result/state-contracts"
        root.mkdir(parents=True)
        models = [{"kind": kind, "all_contracts_passed": True, "baseline_trace_bit_identical": True,
                   "source_weights_sha256": digest,
                   "cases": [{"name": name, "passed": True, "observed": "invented fixture"} for name in C.CASES]}
                  for kind, digest in C.native.CHECKPOINT["weights_sha256"].items()]
        summary = {
            "schema": "english63-native-c-state-contracts-v1",
            "status": "MEASURED_NATIVE_C_STATE_BOUNDARIES_NOT_KWS_QUALIFICATION",
            "source_audio_clips": 63, "source_pcm_samples": 2101760,
            "test_unique_pcm_samples": 128246,
            "source_native_run_id": C.BASELINE["run_id"], "source_header_sha256": C.BASELINE["header_sha256"],
            "optimizer_updates": 0, "new_tts_asr_ipa_calls": 0, "new_independent_audio_seconds": 0,
            "formal_training_admitted": False, "human_gold": False,
            "model_promotion_allowed": False, "model_selection_made": False, "threshold_selection": False,
            "models": models,
        }
        summary.update({key: "NOT_RUN" for key in (
            "independent_english_frr", "continuous_negative_far_per_hour", "word_end_latency", "ssc305_board_profile")})
        self.write(root / "SUMMARY.json", summary)
        for name in C.RESULT_FILES - {"SUMMARY.json", "REPLAY.json"}:
            body = b"invented fixture; never executed"
            if name == "en63_generated.h":
                body = self.header
            elif name == "en63_state_cases":
                body = b"\x7fELF" + body
            elif name.endswith("-baseline-prefix.trace"):
                body = b"EN63TRC1" + bytes(67048 - 8)
            elif name.endswith(".trace"):
                body = b"EN63BND1" + bytes(859156 - 8)
            elif name.endswith(".stderr"):
                body = b""
            (root / name).write_bytes(body)
        self.refresh(state)
        result = state / "result"
        (result / "REQUEST.json").write_bytes(self.path.read_bytes())
        self.write(result / "EXECUTION.json", {
            "request_sha256": self.digest, "execute_exit_code": 0, "verify_exit_code": 0,
            "source_inputs_before_after_verified": True, "optimizer_updates": 0,
        })
        return state

    def refresh(self, state):
        root = state / "result/state-contracts"
        manifest = C.core.file_manifest(root)
        manifest.pop("REPLAY.json", None)
        self.write(root / "REPLAY.json", {
            "schema": "english63-native-c-state-saved-verification-v1",
            "status": "VERIFIED_SAVED_NATIVE_C_STATE_CONTRACTS",
            "summary_sha256": manifest["SUMMARY.json"], "files_sha256": manifest,
            "model_forward_calls": 0, "optimizer_updates": 0, "new_audio_asr_calls": 0,
            "formal_training_admitted": False, "model_promotion_allowed": False,
        })

    def seal(self, state):
        with contextlib.redirect_stdout(io.StringIO()):
            C.seal(self.path, state)

    def environment(self, attempt="1", private=False, event_name="push"):
        event = self.root / "event.json"
        self.write(event, {"repository": {"private": private}})
        return mock.patch.dict(os.environ, {
            "GITHUB_EVENT_PATH": str(event), "GITHUB_EVENT_NAME": event_name,
            "GITHUB_REPOSITORY": C.core.REPOSITORY, "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456", "GITHUB_RUN_ATTEMPT": attempt,
            "GITHUB_ACTIONS": "true", "GITHUB_SHA": "b" * 40,
        })

    def archive(self, root):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as handle:
            for path in root.rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(root).as_posix())
        return out.getvalue()

    def test_only_fixed_new_state_request_is_accepted(self):
        self.assertEqual(C.load_request(self.path)[2], self.digest)
        mutations = (("optimizer_updates", 1), ("optimizer_updates", False),
                     ("entrypoint", C.core.ENTRYPOINT), ("pipeline_commit", "main"),
                     ("native_baseline", {}), ("source_artifacts", {}),
                     ("cpu_threads", 2), ("max_execution_seconds", 3600),
                     ("new_independent_audio_seconds", 131.36), ("human_gold", True))
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                request = copy.deepcopy(self.request)
                request[key] = value
                self.write(self.path, request)
                with self.assertRaises(ValueError):
                    C.load_request(self.path)

    def test_duplicate_keys_and_extra_commands_fail(self):
        self.path.write_text('{"experiment":"x","experiment":"y"}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            C.load_request(self.path)
        self.write(self.path, self.request | {"command": "arbitrary"})
        with self.assertRaises(ValueError):
            C.load_request(self.path)

    def test_old_request_is_immutable_and_dispatch_does_not_train(self):
        self.assertEqual(C.core.sha256(C.SOURCE_REQUEST.read_bytes()), C.BASELINE["request_sha256"])
        self.assertEqual(C.core.native_consumer(self.path).ENTRYPOINT, C.ENTRYPOINT)
        consumer = mock.Mock()
        with mock.patch.object(C.core.sys, "argv", ["executor", "execute", "--request", str(self.path),
                                                     "--state", str(self.root / "dispatch")]), \
             mock.patch.object(C.core, "native_consumer", return_value=consumer), \
             mock.patch.object(C.core, "execute", side_effect=AssertionError("must not train")) as training:
            C.core.main()
        consumer.dispatch.assert_called_once()
        training.assert_not_called()

    def test_complete_terminal_binds_saved_files_and_request(self):
        state = self.fixture()
        self.seal(state)
        self.assertEqual(C.validate_terminal(state / "result", self.request, self.digest)["status"], "SUCCEEDED")
        (state / "result/state-contracts/gru.trace").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "manifest"):
            C.validate_terminal(state / "result", self.request, self.digest)

    def test_missing_duplicate_failed_case_or_false_qualification_cannot_seal(self):
        state = self.fixture()
        path = state / "result/state-contracts/SUMMARY.json"
        original = C.core.read_json(path)
        for mutation in ("missing", "duplicate", "failed", "promotion", "baseline", "weights", "latency"):
            with self.subTest(mutation=mutation):
                summary = copy.deepcopy(original)
                cases = summary["models"][0]["cases"]
                if mutation == "missing":
                    cases.pop()
                elif mutation == "duplicate":
                    cases[-1] = copy.deepcopy(cases[0])
                elif mutation == "failed":
                    cases[-1]["passed"] = False
                elif mutation == "promotion":
                    summary["model_promotion_allowed"] = True
                elif mutation == "baseline":
                    summary["models"][0]["baseline_trace_bit_identical"] = False
                elif mutation == "weights":
                    summary["models"][0]["source_weights_sha256"] = "0" * 64
                else:
                    summary["word_end_latency"] = 0
                self.write(path, summary)
                self.refresh(state)
                with self.assertRaises(ValueError):
                    C.validate_result(state / "result")

    def test_partial_execution_stays_failed(self):
        state = self.fixture()
        execution = C.core.read_json(state / "result/EXECUTION.json")
        execution["verify_exit_code"] = None
        self.write(state / "result/EXECUTION.json", execution)
        self.seal(state)
        self.assertEqual(C.core.read_json(state / "result/TERMINAL.json")["status"], "FAILED")
        with self.assertRaisesRegex(ValueError, "prior state attempt failed"):
            C.validate_terminal(state / "result", self.request, self.digest)

    def test_saved_verifier_cannot_claim_forward_calls(self):
        state = self.fixture()
        path = state / "result/state-contracts/REPLAY.json"
        value = C.core.read_json(path)
        value["model_forward_calls"] = 1
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "must not execute models"):
            C.validate_result(state / "result")

    def test_false_and_float_are_not_successful_process_exit_codes(self):
        for field in ("execute_exit_code", "verify_exit_code"):
            for bad in (False, 0.0, None):
                with self.subTest(field=field, bad=bad):
                    execution = {"request_sha256": self.digest, "execute_exit_code": 0,
                                 "verify_exit_code": 0, "source_inputs_before_after_verified": True,
                                 "optimizer_updates": 0}
                    execution[field] = bad
                    with self.assertRaises(ValueError):
                        C.execution_ok(execution, self.digest)

    def test_rehashed_missing_added_truncated_and_nested_terminal_evidence_fails(self):
        state = self.fixture()
        root = state / "result/state-contracts"
        path = root / "gru.trace"
        original = path.read_bytes()
        path.unlink()
        self.refresh(state)
        with self.assertRaisesRegex(ValueError, "closure"):
            C.validate_result(state / "result")
        path.write_bytes(original[:-1])
        self.refresh(state)
        with self.assertRaisesRegex(ValueError, "trace"):
            C.validate_result(state / "result")
        path.write_bytes(original)
        (root / "EXTRA.json").write_text("{}")
        self.refresh(state)
        with self.assertRaisesRegex(ValueError, "closure"):
            C.validate_result(state / "result")
        (root / "EXTRA.json").unlink()
        (root / "TERMINAL.json").write_text("{}")
        self.refresh(state)
        with self.assertRaisesRegex(ValueError, "nested state terminal"):
            C.validate_result(state / "result")

    def test_only_explicit_nonsecret_actions_identity_reaches_child(self):
        state = self.root / "child"
        state.mkdir()
        with self.environment(), mock.patch.dict(os.environ, {
            "GH_TOKEN": "fixture-secret", "GITHUB_TOKEN": "fixture-secret",
            "ACTIONS_RUNTIME_TOKEN": "fixture-secret", "GITHUB_ARBITRARY": "not-forwarded",
        }):
            environment = C.state_environment(Path("/invented/python"), state)
        self.assertEqual({k for k in environment if k.startswith("GITHUB_")}, {
            "GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_REPOSITORY", "GITHUB_SHA"})
        self.assertEqual(environment["GITHUB_REPOSITORY"], C.core.REPOSITORY)
        self.assertFalse({"GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"} & set(environment))
        self.assertNotIn("GITHUB_ARBITRARY", environment)
        with self.environment(attempt="2"):
            with self.assertRaisesRegex(ValueError, "public Actions run identity"):
                C.state_environment(Path("/invented/python"), state)

    def test_prior_success_reuses_terminal_before_inputs_or_models(self):
        state = self.fixture()
        self.seal(state)
        raw = self.archive(state / "result")
        with self.environment(attempt="2"), contextlib.redirect_stdout(io.StringIO()) as output, \
             mock.patch.object(C.core, "named_artifacts", side_effect=[[{"id": 7}], []]), \
             mock.patch.object(C.core, "artifact_bytes", return_value=raw), \
             mock.patch.object(C.core, "github_get", side_effect=AssertionError("must not download inputs")):
            C.prepare(self.path, self.root / "reuse")
        self.assertIn("execute=false", output.getvalue())

    def test_failed_terminal_and_orphan_claim_cannot_repeat(self):
        state = self.fixture()
        (state / "result/EXECUTION.json").unlink()
        self.seal(state)
        raw = self.archive(state / "result")
        with self.environment(), mock.patch.object(C.core, "named_artifacts", side_effect=[[{"id": 7}], []]), \
             mock.patch.object(C.core, "artifact_bytes", return_value=raw), \
             mock.patch.object(C.core, "github_get", side_effect=AssertionError("must not download inputs")):
            with self.assertRaisesRegex(ValueError, "prior state attempt failed"):
                C.prepare(self.path, self.root / "failed-reuse")
        with self.environment(), mock.patch.object(C.core, "named_artifacts", side_effect=[[], [{"id": 8}]]):
            with self.assertRaisesRegex(ValueError, "prior state claim"):
                C.prepare(self.path, self.root / "claim-only")

    def test_missing_receipts_do_not_authorize_rerun(self):
        with self.environment(attempt="2"), mock.patch.object(C.core, "named_artifacts", return_value=[]):
            with self.assertRaisesRegex(ValueError, "rerun"):
                C.prepare(self.path, self.root / "second-attempt")

    def test_private_repo_and_pr_cannot_execute(self):
        for private, name in ((True, "push"), (False, "pull_request")):
            with self.subTest(private=private, event=name), self.environment(private=private, event_name=name):
                with self.assertRaisesRegex(ValueError, "public main"):
                    C.prepare(self.path, self.root / "blocked")

    def test_execution_failure_retains_receipt_and_does_not_start_verifier(self):
        state = self.root / "execute"
        (state / "result").mkdir(parents=True)
        (state / "claim").mkdir()
        self.write(state / "claim/CLAIM.json", {"request_sha256": self.digest})
        pipeline = self.root / "pipeline"
        script = pipeline / C.ENTRYPOINT
        script.parent.mkdir(parents=True)
        script.write_text("# inert fixture; never executed\n")
        with self.environment(), mock.patch.object(C.subprocess, "check_output", side_effect=["a" * 40, ""]), \
             mock.patch.object(C, "validate_inputs"), \
             mock.patch.object(C.native, "run_bounded", return_value=1) as child:
            with self.assertRaisesRegex(ValueError, "execution"):
                C.execute(self.path, state, pipeline, Path("/invented/python"))
        self.assertEqual(child.call_count, 1)
        receipt = C.core.read_json(state / "result/EXECUTION.json")
        self.assertEqual(receipt["execute_exit_code"], 1)
        self.assertIsNone(receipt["verify_exit_code"])
        self.assertFalse(receipt["source_inputs_before_after_verified"])


if __name__ == "__main__":
    unittest.main()
