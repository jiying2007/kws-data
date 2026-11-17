"""Source guards and process supervision: invented data, no numeric model."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quality_snr10_guard_subject", HERE / "english_quality_snr10_continue.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class Guards(unittest.TestCase):
    def plan(self):
        return json.loads((HERE / "english_quality_snr10_continue_plan.json").read_text())

    def test_fixed_budget_rejects_expansion_and_authority(self):
        plan = self.plan()
        subject.validate_plan(plan)
        for field, value in [("maximum_optimizer_updates", 97), ("maximum_embedding_calls", 4001),
                             ("snr_db", 10), ("snr_db", 9.0), ("retry_count", 1),
                             ("maximum_execution_seconds", 901), ("human_gold", True),
                             ("development_tuning", True), ("threshold", 0.79)]:
            changed = copy.deepcopy(plan)
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                subject.validate_plan(changed)

    def test_only_exact_first_public_push_is_executable(self):
        plan = self.plan()
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": subject.REPO,
               "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
               "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "a" * 40}
        event = {"repository": {"full_name": subject.REPO, "private": False},
                 "before": subject.MAIN_BEFORE, "after": env["GITHUB_SHA"]}
        subject.execution_guard(plan, env, event)
        for field, value in [("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "pull_request"),
                             ("GITHUB_ACTIONS", "false"), ("GITHUB_REPOSITORY", "someone/else")]:
            changed = dict(env, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.execution_guard(plan, changed, event)
        for update in ({"before": "b" * 40}, {"repository": {"full_name": subject.REPO, "private": True}}):
            with self.assertRaises(ValueError):
                subject.execution_guard(plan, env, {**event, **update})

    def test_default_mode_never_imports_numeric_runtime(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("onnxruntime", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_successful_process_exit_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject.supervised_command([sys.executable, "-c", "print('source fixture')"],
                                       root / "probe.stdout", root / "probe.stderr", 5)
            receipt = json.loads((root / "probe.exit.json").read_text())
            self.assertEqual(receipt["returncode"], 0)
            self.assertFalse(receipt["process_group_terminated"])

    @unittest.skipUnless(os.name == "posix", "process group supervision is a Linux worker contract")
    def test_timeout_terminates_child_group_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(subprocess.TimeoutExpired):
                subject.supervised_command([sys.executable, "-c", "import time; time.sleep(30)"],
                                           root / "probe.stdout", root / "probe.stderr", 0.1)
            receipt = json.loads((root / "probe.exit.json").read_text())
            self.assertTrue(receipt["process_group_terminated"])
            self.assertIsNotNone(receipt["returncode"])
            self.assertEqual(receipt["error"], "TimeoutExpired")


if __name__ == "__main__":
    unittest.main()
