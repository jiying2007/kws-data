#!/usr/bin/env python3
"""Source-only process-group tests; never starts a TTS or ASR model."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import supervise_bulk_autopilot as supervisor  # noqa: E402


class HardDeadlineSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def child(self, code):
        script = self.root / "fake-child.py"
        script.write_text(code, encoding="utf-8")
        return [sys.executable, "-B", str(script)]

    def cli_args(self):
        args = []
        for key in ("plan", "sources", "audio-root", "workspace",
                    "adapter-lock", "pipeline-root"):
            value = self.root if key == "workspace" else self.root / (key + ".json")
            args += ["--" + key, str(value)]
        return args + ["--max-tts", "1", "--max-asr", "0"]

    def test_pinned_existing_canonical_controller_source(self):
        controller = supervisor.checked_controller(ROOT)
        self.assertEqual(controller.name, "bulk_autopilot.py")
        other = self.root / supervisor.CONTROLLER_RELATIVE
        other.parent.mkdir(parents=True)
        other.write_bytes(controller.read_bytes() + b"\n# modified fixture\n")
        with self.assertRaisesRegex(ValueError, "CONTROLLER_SOURCE_DRIFT"):
            supervisor.checked_controller(self.root)

    def test_child_environment_removes_secrets_and_proxy(self):
        passed = supervisor.child_environment({
            "HOME": "/tmp/not-a-secret", "PATH": "/usr/bin",
            "GITHUB_TOKEN": "private", "HF_TOKEN": "private",
            "AWS_SECRET_ACCESS_KEY": "private", "HTTPS_PROXY": "proxy",
            "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
        self.assertEqual(passed["HF_HUB_OFFLINE"], "1")
        self.assertEqual(passed["TRANSFORMERS_OFFLINE"], "1")
        self.assertNotIn("GITHUB_TOKEN", passed)
        self.assertNotIn("HF_TOKEN", passed)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", passed)
        self.assertNotIn("HTTPS_PROXY", passed)

    def test_dry_run_and_missing_model_never_start_child(self):
        args = self.cli_args()
        with patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("spawn")):
            with contextlib.redirect_stdout(io.StringIO()) as dry:
                self.assertEqual(supervisor.main(args), 0)
            self.assertIn("NOT_RUN_EXECUTE_REQUIRED", dry.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as missing:
                self.assertEqual(supervisor.main(args + ["--execute"]), 0)
            self.assertIn("NOT_RUN_MODEL_ASSETS_MISSING", missing.getvalue())

    def test_budget_fails_before_first_child(self):
        args = self.cli_args() + ["--model-folder", str(self.root),
                                  "--model-lock", str(self.root / "lock.json"),
                                  "--execute", "--hard-seconds", "0"]
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
            with patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("spawn")):
                with self.assertRaisesRegex(ValueError, "HARD_BOUND"):
                    supervisor.main(args)

    def test_successful_fake_child_is_not_retried(self):
        marker = self.root / "invocations"
        cmd = self.child("from pathlib import Path\n"
                         "p=Path(" + repr(str(marker)) + ")\n"
                         "p.write_text((p.read_text() if p.exists() else '')+'1')\n")
        result = supervisor.supervise(cmd, hard_seconds=5, grace_seconds=1, env=os.environ)
        self.assertEqual(result["status"], "CHILD_SUCCESS")
        self.assertEqual(marker.read_text(), "1")
        self.assertFalse(result["training_admitted"])

    def test_nonzero_fake_child_preserved_without_auto_restart(self):
        marker = self.root / "invocations"
        cmd = self.child("from pathlib import Path\nimport sys\n"
                         "p=Path(" + repr(str(marker)) + ")\n"
                         "p.write_text((p.read_text() if p.exists() else '')+'1')\n"
                         "sys.exit(7)\n")
        result = supervisor.supervise(cmd, hard_seconds=5, grace_seconds=1, env=os.environ)
        self.assertEqual(result["status"], "CHILD_FAILED_NO_AUTO_RETRY")
        self.assertEqual(result["child_exit_code"], 7)
        self.assertEqual(marker.read_text(), "1")

    def test_hard_timeout_kills_process_group_not_just_parent(self):
        started = self.root / "started"
        descendant_written = self.root / "orphan-output"
        grandchild = ("import time\nfrom pathlib import Path\n"
                      "time.sleep(2)\n"
                      "Path(" + repr(str(descendant_written)) + ").write_text('escaped')\n")
        code = ("import subprocess,sys,time\nfrom pathlib import Path\n"
                "Path(" + repr(str(started)) + ").write_text('once')\n"
                "subprocess.Popen([sys.executable, '-c', " + repr(grandchild) + "])\n"
                "time.sleep(20)\n")
        result = supervisor.supervise(self.child(code), hard_seconds=1,
                                      grace_seconds=1, env=os.environ)
        self.assertEqual(result["status"], "HARD_TIMEOUT_HOLD_NO_REPLAY")
        self.assertTrue(started.exists())
        time.sleep(2.2)
        self.assertFalse(descendant_written.exists())
        self.assertFalse(result["qualification_allowed"])

    def test_timeout_parameter_bounds_and_boolean_rejected(self):
        with self.assertRaisesRegex(ValueError, "HARD_TIMEOUT_BUDGET"):
            supervisor.supervise(["not-started"], hard_seconds=True,
                                 grace_seconds=1, env=os.environ)
        with self.assertRaisesRegex(ValueError, "TERMINATION_GRACE_BUDGET"):
            supervisor.supervise(["not-started"], hard_seconds=5,
                                 grace_seconds=0, env=os.environ)


if __name__ == "__main__":
    unittest.main()
