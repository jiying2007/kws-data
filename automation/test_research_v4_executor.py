#!/usr/bin/env python3
"""Offline failure/reuse boundaries; invented checkpoint bytes, no model calls."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.error
import zipfile

SPEC = importlib.util.spec_from_file_location("executor", Path(__file__).with_name("research_v4_executor.py"))
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.request = copy.deepcopy(E.FIXED) | {
            "request_id": "fixture-v1", "pipeline_commit": "a" * 40,
            "reason": "Invented offline fixture; no actual model computation.",
        }
        self.path = self.root / "fixture-v1.json"
        self.save_request()

    def save_request(self):
        self.path.write_text(json.dumps(self.request) + "\n")
        self.digest = E.sha256(self.path.read_bytes())

    def success_fixture(self):
        state = self.root / "completed"
        result = state / "result"
        model = result / "model"
        model.mkdir(parents=True)
        (result / "REQUEST.json").write_bytes(self.path.read_bytes())
        (model / "events.jsonl").write_bytes(b"")  # zero proposals are valid
        rows = []
        replay_rows = []
        for kind in ("gru", "dsc_tcn"):
            filename = kind + "-weights.npz"
            (model / filename).write_bytes(("invented-checkpoint-" + kind).encode())
            digest = E.sha256((model / filename).read_bytes())
            rows.append({"kind": kind, "weight_file": filename, "weights_sha256": digest,
                         "warm_start_sha256": "b" * 64, "optimizer_updates": 48,
                         "train_rows": 30, "train_nearphone_rows": 0,
                         "train_development_rows": 0, "model_promotion_allowed": False})
            replay_rows.append({"kind": kind, "weights_sha256": digest,
                                "saved_predictions_reproduced": True,
                                "clip_comparisons": 63, "continuous_comparisons": 1})
        summary = {
            "schema": "english63-bounded-online-weak-mil-v1",
            "status": "EXECUTED_WEAK_MIL_ONLINE_PROPOSALS_NOT_KWS_QUALIFICATION",
            "formal_training_admitted": False, "human_gold": False,
            "model_promotion_allowed": False, "source_audio_clips": 63,
            "source_pcm_samples": 2101760, "role_counts": E.ROLES,
            "models": rows,
        }
        summary.update({key: "NOT_RUN" for key in (
            "independent_english_frr", "continuous_negative_far_per_hour",
            "word_end_latency", "ssc305_board_profile")})
        E.write_json(model / "SUMMARY.json", summary)
        E.write_json(model / "REPLAY.json", {
            "schema": "english63-online-saved-weight-replay-v1",
            "status": "REPRODUCED_SAVED_WEAK_PROPOSALS", "optimizer_updates": 0,
            "formal_training_admitted": False, "model_promotion_allowed": False,
            "summary_sha256": E.sha256((model / "SUMMARY.json").read_bytes()),
            "events_sha256": E.sha256(b""), "models": replay_rows,
        })
        evidence = result / "keyword-evidence"
        evidence.mkdir()
        E.write_json(evidence / "report.json", {
            "schema": "kws-v4-existing-keyword-evidence-v1", "audio_rows": 63,
            "keyword_rows": 126, "asr_observations_reused": 189,
            "new_model_calls": 0, "training_admitted": False,
            "human_gold": False, "accepted_negative_seconds": 0,
        })
        (evidence / "rows.jsonl").write_text('{"fixture": true}\n')
        E.write_json(result / "EXECUTION.json", {"request_sha256": self.digest, "train_exit_code": 0, "replay_exit_code": 0})
        (result / "runtime.txt").write_text("Invented source-only fixture.\n")
        return state, result

    def seal_fixture(self, state):
        with contextlib.redirect_stdout(io.StringIO()):
            E.seal(self.path, state)

    def prepare_environment(self, attempt="1"):
        event = self.root / "event.json"
        event.write_text('{"repository": {"private": false}}')
        return mock.patch.dict(os.environ, {
            "GITHUB_EVENT_PATH": str(event), "GITHUB_EVENT_NAME": "push",
            "GITHUB_REPOSITORY": E.REPOSITORY, "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": attempt,
        })

    def test_request_is_fixed_and_template_cannot_execute(self):
        self.assertEqual(E.load_request(self.path)[2], self.digest)
        for key, value in (("pipeline_repository", "someone/else"), ("entrypoint", "arbitrary.py"),
                           ("optimizer_updates_per_model", 49), ("new_asr_calls", False),
                           ("pipeline_commit", "main"), ("formal_training_admitted", True)):
            with self.subTest(key=key):
                original = self.request[key]
                self.request[key] = value
                self.save_request()
                with self.assertRaises(ValueError):
                    E.load_request(self.path)
                self.request[key] = original
        template = Path(__file__).parents[1] / ".github/research-v4/request.template.json"
        value = json.loads(template.read_text())
        target = self.root / (value["request_id"] + ".json")
        target.write_bytes(template.read_bytes())
        with self.assertRaises(ValueError):
            E.load_request(target)

    def test_request_duplicate_keys_and_wrong_filename_rejected(self):
        self.path.write_text('{"schema": "a", "schema": "b"}')
        with self.assertRaises(ValueError):
            E.load_request(self.path)
        self.save_request()
        target = self.root / "renamed.json"
        target.write_bytes(self.path.read_bytes())
        with self.assertRaises(ValueError):
            E.load_request(target)

    def test_only_one_new_versioned_request_can_trigger(self):
        path = E.REQUEST_DIR + "/fixture-v1.json"
        self.assertEqual(E.changed_request("A\t" + path, False), path)
        self.assertIsNone(E.changed_request("", True))
        for changes in ("", "M\t" + path, "D\t" + path, "A\t" + path + "\nA\t" + path,
                        "R100\told\t" + path, "A\t" + E.REQUEST_DIR + "/nested/request.json"):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                E.changed_request(changes, False)

    def test_model_environment_has_no_credential_or_python_injection(self):
        with mock.patch.dict(os.environ, {"GH_TOKEN": "fake-secret", "ACTIONS_RUNTIME_TOKEN": "fake-runtime",
                                         "PYTHONPATH": "/untrusted", "GITHUB_TOKEN": "fake-token"}):
            environment = E.child_environment("/fake/venv/bin/python", self.root)
        self.assertEqual(environment["PATH"], "/fake/venv/bin:/usr/bin:/bin")
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN", "PYTHONPATH"):
            self.assertNotIn(name, environment)
        self.assertNotIn("fake-secret", json.dumps(environment))

    def test_blob_redirect_does_not_forward_authorization(self):
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError("https://api.github.com/example", 302,
                                                        "found", {"Location": "https://blob.example/fixture"}, None)
        with mock.patch.dict(os.environ, {"GH_TOKEN": "fixture-token"}), \
             mock.patch.object(E.urllib.request, "build_opener", return_value=opener), \
             mock.patch.object(E.urllib.request, "urlopen", return_value=io.BytesIO(b"fixture")) as redirect:
            self.assertEqual(E.github_get("/actions/artifacts/1/zip", archive=True), b"fixture")
        original = opener.open.call_args.args[0]
        signed = redirect.call_args.args[0]
        self.assertEqual(original.get_header("Authorization"), "Bearer fixture-token")
        self.assertIsNone(signed.get_header("Authorization"))

    def test_actual_checkpoints_replay_and_zero_proposals_are_accepted(self):
        state, result = self.success_fixture()
        self.seal_fixture(state)
        terminal = E.validate_terminal(result, self.request, self.digest)
        self.assertEqual(terminal["status"], "SUCCEEDED")
        (result / "model/gru-weights.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "manifest"):
            E.validate_terminal(result, self.request, self.digest)

    def test_missing_checkpoint_is_terminal_failure_not_success(self):
        state, result = self.success_fixture()
        (result / "model/dsc_tcn-weights.npz").unlink()
        self.seal_fixture(state)
        self.assertEqual(E.read_json(result / "TERMINAL.json")["status"], "FAILED")
        with self.assertRaisesRegex(ValueError, "prior execution failed"):
            E.validate_terminal(result, self.request, self.digest)

    def test_exposed_training_and_fake_replay_are_rejected(self):
        _, result = self.success_fixture()
        summary_path = result / "model/SUMMARY.json"
        summary = E.read_json(summary_path)
        summary["models"][0]["train_development_rows"] = 1
        summary_path.write_text(json.dumps(summary))
        replay_path = result / "model/REPLAY.json"
        replay = E.read_json(replay_path)
        replay["summary_sha256"] = E.sha256(summary_path.read_bytes())
        replay_path.write_text(json.dumps(replay))
        with self.assertRaisesRegex(ValueError, "role isolation"):
            E.validate_model_result(result)
        summary["models"][0]["train_development_rows"] = 0
        summary_path.write_text(json.dumps(summary))
        replay["summary_sha256"] = E.sha256(summary_path.read_bytes())
        replay["optimizer_updates"] = 1
        replay_path.write_text(json.dumps(replay))
        with self.assertRaisesRegex(ValueError, "replay must not train"):
            E.validate_model_result(result)

    def test_prior_success_reuses_checkpoints_without_inputs_or_models(self):
        state, result = self.success_fixture()
        self.seal_fixture(state)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as handle:
            for file in result.rglob("*"):
                if file.is_file():
                    handle.write(file, file.relative_to(result).as_posix())
        item = {"id": 99, "expired": False}
        fresh = self.root / "rerun"
        with self.prepare_environment(attempt="2"), \
             mock.patch.object(E, "named_artifacts", side_effect=[[item], [{"id": 98}]]), \
             mock.patch.object(E, "artifact_bytes", return_value=archive.getvalue()), \
             mock.patch.object(E, "artifact_metadata", side_effect=AssertionError("must not download original inputs")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            E.prepare(self.path, fresh)
        self.assertIn("execute=false", output.getvalue())
        self.assertTrue((fresh / "reused/model/gru-weights.npz").is_file())
        self.assertFalse((fresh / "inputs").exists())

    def test_started_or_receiptless_rerun_cannot_repeat_training(self):
        for attempt, receipts in (("1", [[], [{"id": 1}]]), ("2", [[], []])):
            with self.subTest(attempt=attempt), self.prepare_environment(attempt), \
                 mock.patch.object(E, "named_artifacts", side_effect=receipts), \
                 mock.patch.object(E, "artifact_metadata", side_effect=AssertionError("must not download")), \
                 self.assertRaises(ValueError):
                E.prepare(self.path, self.root / ("attempt-" + attempt))

    def test_expired_receipt_and_unsafe_zip_cannot_start_over(self):
        with mock.patch.object(E, "github_get") as network, self.assertRaisesRegex(ValueError, "consumed"):
            E.artifact_bytes({"expired": True, "id": 1})
        network.assert_not_called()
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../escape", b"fixture")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            E.restore_archive(archive, self.root / "restored")
        self.assertFalse((self.root / "escape").exists())

    def test_timeout_retains_failed_execution_and_partial_checkpoint(self):
        state = self.root / "running"
        (state / "claim").mkdir(parents=True)
        (state / "result/model").mkdir(parents=True)
        (state / "result/REQUEST.json").write_bytes(self.path.read_bytes())
        (state / "result/model/gru-weights.npz").write_bytes(b"partial-fixture")
        E.write_json(state / "claim/CLAIM.json", {"request_sha256": self.digest})
        pipeline = self.root / "pipeline"
        script = pipeline / E.ENTRYPOINT
        script.parent.mkdir(parents=True)
        script.write_text("# never executed\n")
        with mock.patch.object(E.subprocess, "check_output", return_value="a" * 40 + "\n"), \
             mock.patch.object(E.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 1200)), \
             self.assertRaisesRegex(ValueError, "failed"):
            E.execute(self.path, state, pipeline, Path("/fake/venv/bin/python"))
        execution = E.read_json(state / "result/EXECUTION.json")
        self.assertIn("TimeoutExpired", execution["error"])
        self.assertIsNone(execution["replay_exit_code"])
        self.seal_fixture(state)
        self.assertEqual(E.read_json(state / "result/TERMINAL.json")["status"], "FAILED")
        self.assertEqual((state / "result/model/gru-weights.npz").read_bytes(), b"partial-fixture")


if __name__ == "__main__":
    unittest.main()
