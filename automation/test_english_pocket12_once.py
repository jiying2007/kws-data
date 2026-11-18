"""Source-only guards plus one complete twelve-row cross-module mock packet."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_pocket12_once as w


def execution_fixture():
    plan = copy.deepcopy(w.read_json(w.PLAN))
    plan.update(enabled=True, main_before="a" * 40)
    env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": w.REPOSITORY,
           "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1",
           "GITHUB_RUN_ID": "123", "GITHUB_SHA": "b" * 40,
           "GITHUB_WORKFLOW_REF": w.REPOSITORY + "/" + w.WORKFLOW + "@refs/heads/main"}
    event = {"repository": {"full_name": w.REPOSITORY, "private": False},
             "ref": "refs/heads/main", "before": "a" * 40, "after": "b" * 40,
             "deleted": False, "forced": False}
    identity = {"repository": w.REPOSITORY, "run_id": "123", "commit": "b" * 40,
                "run_attempt": "1", "event": "push"}
    return plan, env, event, identity


class OnceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_frozen_source_check_imports_no_numeric_runtime(self):
        before = set(sys.modules)
        result = w.source_preflight()
        self.assertEqual((result["source_check"], result["new_model_calls"], result["new_network_calls"]),
                         ("PASS", 0, 0))
        self.assertFalse({"torch", "numpy", "scipy", "transformers", "pocket_tts"} & (set(sys.modules) - before))

    def test_only_bound_public_first_main_event_is_executable(self):
        plan, env, event, identity = execution_fixture()
        kwargs = {"environ": env, "event": event, "head": env["GITHUB_SHA"], "parent": plan["main_before"]}
        self.assertEqual(w.execution_identity(plan, **kwargs), identity)
        cases = [("plan", "enabled", False), ("env", "GITHUB_RUN_ATTEMPT", "2"),
                 ("env", "GITHUB_EVENT_NAME", "pull_request"), ("env", "GITHUB_REF", "refs/heads/topic"),
                 ("env", "GITHUB_WORKFLOW_REF", w.REPOSITORY + "/other.yml@refs/heads/main"),
                 ("event", "before", "c" * 40), ("event", "after", "c" * 40),
                 ("event", "forced", True), ("event", "deleted", True),
                 ("event", "repository", {"full_name": w.REPOSITORY, "private": True})]
        for scope, key, value in cases:
            with self.subTest(scope=scope, key=key):
                p, e, ev = copy.deepcopy(plan), dict(env), copy.deepcopy(event)
                {"plan": p, "env": e, "event": ev}[scope][key] = value
                with self.assertRaises(ValueError):
                    w.execution_identity(p, environ=e, event=ev, head=env["GITHUB_SHA"], parent=plan["main_before"])
        with self.assertRaisesRegex(ValueError, "FIRST_PARENT"):
            w.execution_identity(plan, environ=env, event=event, head=env["GITHUB_SHA"], parent="c" * 40)

    def test_missing_source_is_unknown_and_blocks_installation(self):
        plan, _, _, identity = execution_fixture()
        value = w.source_readback(self.root / "missing", plan, identity)
        self.assertFalse(value["source_ready"])
        self.assertIsNone(value["tts_attempts"])
        self.assertIsNone(value["failed_count"])
        with mock.patch.object(w.subprocess, "run") as process, mock.patch.object(w.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(ValueError, "TWELVE_VERIFIED"):
                w.install_runtime("qwen", self.root / "venv", self.root / "runtime", None,
                    source_root=self.root / "missing", plan=plan, identity=identity)
        process.assert_not_called()
        network.assert_not_called()
        self.assertEqual(w.read_json(self.root / "runtime/INSTALLATION.json")["status"], "FAILED")

    def test_missing_stage_receipt_does_not_claim_zero_or_modify_sealed_packet(self):
        identity = execution_fixture()[3]
        path, runtime = self.root / "packet", self.root / "runtime"
        w.stage_status("qwen", path, runtime, identity, "skipped")
        self.assertIsNone(w.read_json(path / "UNOBSERVED.json")["actual_model_calls"])
        (path / "UNOBSERVED.json").unlink()
        (path / "RESULT.json").write_text('{"status":"COMPLETE"}\n')
        original = (path / "RESULT.json").read_bytes()
        w.stage_status("qwen", path, runtime, identity, "success")
        self.assertEqual([p.name for p in path.iterdir()], ["RESULT.json"])
        self.assertEqual((path / "RESULT.json").read_bytes(), original)

    def test_installer_has_no_inherited_credentials_or_private_indexes(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "do-not-copy", "GITHUB_TOKEN": "do-not-copy",
                "ACTIONS_RUNTIME_TOKEN": "do-not-copy", "PIP_INDEX_URL": "https://private.invalid"}):
            env = w.clean_environment()
            self.assertFalse({"HF_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN", "PIP_INDEX_URL"} & set(env))
            w.strip_credentials()
            self.assertFalse({"HF_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"} & set(os.environ))
            self.assertEqual(os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")

    def test_asset_failure_has_one_attempt_and_retained_partial_receipt(self):
        pin = {"path": "fake.bin", "url": "https://example.invalid/frozen", "bytes": 3, "sha256": "0" * 64}
        source = w.component("source")
        with mock.patch.object(source, "recipe", return_value={"assets": [pin]}), \
                mock.patch.object(w.urllib.request, "urlopen", return_value=io.BytesIO(b"abc")) as request:
            with self.assertRaisesRegex(ValueError, "EXACT_ASSET_BYTES"):
                w.download_assets(self.root / "assets", self.root / "runtime", time.monotonic() + 10)
        self.assertEqual(request.call_count, 1)
        receipt = w.read_json(self.root / "runtime/DOWNLOADS.json")
        self.assertEqual((len(receipt), receipt[0]["status"], receipt[0]["received_bytes"]), (1, "FAILED", 3))

    def test_duplicate_json_and_symlink_packet_fail_closed(self):
        path = self.root / "bad.json"
        path.write_text('{"enabled":false,"enabled":true}')
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            w.read_json(path)
        packet = self.root / "packet"
        packet.mkdir()
        (packet / "link").symlink_to(path)
        with self.assertRaisesRegex(ValueError, "SYMLINK"):
            w.copy_packet(packet, self.root / "copy")

    def test_combination_failure_preserves_raw_partial_and_unknown_counts(self):
        plan, _, _, identity = execution_fixture()
        source = self.root / "partial"
        source.mkdir()
        (source / "CHECKPOINT.json").write_text('{"partial":true}\n')
        output = self.root / "combined"
        observers = {p: self.root / p for p in w.PROVIDERS}
        with mock.patch.object(w.component("evaluate"), "evaluate", side_effect=ValueError("incomplete fixture")):
            result = w.combine(source, observers, output, plan, identity)
        evaluation = w.read_json(output / "EVALUATION.json")
        self.assertEqual(result["evaluation_status"], "INCOMPLETE")
        self.assertIsNone(evaluation["actual_total_model_calls"])
        self.assertEqual((output / "source/CHECKPOINT.json").read_bytes(), (source / "CHECKPOINT.json").read_bytes())
        self.assertIn("source/CHECKPOINT.json", {r["path"] for r in w.read_json(output / "FILES.json")["files"]})

    def test_workflow_has_no_dispatch_and_fixed_four_observers(self):
        workflow = (w.ROOT / w.WORKFLOW).read_text()
        self.assertNotIn("workflow_dispatch", workflow)
        self.assertNotIn("secrets.", workflow)
        self.assertIn("provider: [qwen, whisper, parakeet, phoneme]", workflow)
        self.assertIn("fetch-depth: 2", workflow)
        for artifact in ("pocket12-source", "pocket12-observer-", "pocket12-runtime-", "english-pocket12-probe-"):
            self.assertIn(artifact, workflow)
        self.assertIn("1140s", workflow)
        self.assertIn("2040s", workflow)

    def test_complete_packet_crosses_real_cli_and_readback_boundaries(self):
        from test_english_pocket12_observe import MockSourceBackend as MockBackend, MockObserverFactory
        source, observer = w.component("source"), w.component("observer")
        plan, env, event, identity = execution_fixture()
        plan_path, event_path = self.root / "plan.json", self.root / "event.json"
        w.write_json(plan_path, plan)
        w.write_json(event_path, event)
        env["GITHUB_EVENT_PATH"] = str(event_path)
        source_root, observer_parent = self.root / "source", self.root / "observers"
        factory = None
        MockBackend.calls, MockBackend.failing = [], None

        def git_read(args, **kwargs):
            self.assertEqual(args[:2], ["git", "rev-parse"])
            return (env["GITHUB_SHA"] if args[2] == "HEAD" else plan["main_before"]) + "\n"

        def cli(*args):
            with contextlib.redirect_stdout(io.StringIO()):
                code = w.main(["--plan", str(plan_path), *map(str, args)])
            self.assertEqual(code, 0)

        with mock.patch.dict(os.environ, env), mock.patch.object(w, "PLAN", plan_path), \
                mock.patch.object(w.subprocess, "check_output", side_effect=git_read), \
                mock.patch.object(source, "NativePocket", MockBackend), \
                mock.patch.object(observer, "_load", side_effect=lambda *args: factory(*args)), \
                mock.patch.object(w.urllib.request, "urlopen", side_effect=AssertionError("no network in mock integration")):
            cli("--generate", "--output", source_root, "--assets", self.root / "unused-assets")
            factory = MockObserverFactory(source_root)
            cli("--read-source", "--source", source_root, "--output", self.root / "source-readback.json")
            readback = w.read_json(self.root / "source-readback.json")
            self.assertEqual((readback["source_ready"], readback["tts_attempts"], readback["tts_returned"]), (True, 12, 12))
            for provider in w.PROVIDERS:
                folder = observer_parent / ("pocket12-observer-" + provider)
                cli("--observe", provider, "--source", source_root, "--output", folder,
                    "--assets", self.root / ("private-" + provider))
                seen = observer.readback(provider=provider, source_root=source_root,
                    observer_root=folder, plan=plan, run_identity=identity)
                self.assertEqual(len(seen["rows"]), 12)
                for row in source.recipe()["rows"]:
                    captured = w.read_json(folder / "rows" / row["id"] / "CAPTURE.json")
                    source_row = next(r for r in readback["rows"] if r["id"] == row["id"])
                    self.assertEqual(captured["input_pcm_sha256"], source_row["pcm_sha256"])
                    self.assertEqual(captured["run"], identity)
            cli("--combine", "--source", source_root, "--observers", observer_parent,
                "--output", self.root / "combined")
        self.assertEqual(len(MockBackend.calls), 12)
        self.assertEqual(len(set(MockBackend.calls)), 12)
        self.assertEqual(len(factory.calls), 48)
        self.assertEqual(len(set((p, row) for p, row, _ in factory.calls)), 48)
        result = w.read_json(self.root / "combined/RESULT.json")
        evaluation = w.read_json(self.root / "combined/EVALUATION.json")
        self.assertIsNone(result["combination_error"])
        self.assertEqual(result["evaluation_status"], "PASS_POCKET12_SOURCE_FEASIBILITY")
        self.assertEqual(len(evaluation["rows"]), 12)
        self.assertEqual(evaluation["counts"]["actual_total_requests"], 60)
        self.assertFalse(evaluation["training_allowed"])
        self.assertFalse({"torch", "numpy", "scipy", "transformers", "pocket_tts"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main()
