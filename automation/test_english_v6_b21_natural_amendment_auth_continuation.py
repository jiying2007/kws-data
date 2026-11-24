"""Original-byte GH regression and zero-call operational continuation guards."""
from __future__ import annotations
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_natural_amendment_source_ancestry as lineage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class AuthorizationContinuation(unittest.TestCase):
    def test_original_twelve_escape_bytes_roundtrip_without_rendering(self):
        raw = (HERE / "english_v6_b21_natural_amendment_escape_fixture.bin").read_bytes()
        self.assertEqual(raw, bytes.fromhex("1b5b33316d4f4b1b5b306d0a"))
        self.assertEqual(len(raw), 12)
        stdout, stderr = io.StringIO(), io.StringIO()
        result = subprocess.CompletedProcess([], 0, stdout=raw, stderr=b"")
        with redirect_stdout(stdout), redirect_stderr(stderr), mock.patch.object(lineage.subprocess, "run", return_value=result) as run:
            self.assertEqual(lineage.gh("actions/jobs/138354602305/logs"), raw)
        run.assert_called_once_with(["gh", "api", "repos/jiying2007/kws-data/actions/jobs/138354602305/logs", "--allow-escape-sequences"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))

    def test_absolute_repo_endpoint_is_not_double_prefixed(self):
        with mock.patch.object(lineage.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"{}", b"")) as run:
            self.assertEqual(lineage.gh("repos/jiying2007/kws-data/actions/runs/123"), b"{}")
        self.assertEqual(run.call_args.args[0][2], "repos/jiying2007/kws-data/actions/runs/123")

    def test_failed_gh_retains_raw_output_in_exception_without_printing(self):
        raw = (HERE / "english_v6_b21_natural_amendment_escape_fixture.bin").read_bytes()
        stdout, stderr = io.StringIO(), io.StringIO()
        error = subprocess.CalledProcessError(1, ["gh", "api"], output=raw, stderr=raw)
        with redirect_stdout(stdout), redirect_stderr(stderr), mock.patch.object(lineage.subprocess, "run", side_effect=error):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                lineage.gh("actions/jobs/1/logs")
        self.assertEqual(caught.exception.output, raw)
        self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))

    def test_fresh_names_do_not_bypass_any_old_consumed_budget(self):
        empty = {name: dict(total_count=0, artifacts=[]) for name in lineage.prior_budget_names()}
        self.assertEqual(len(empty), 6)
        self.assertEqual(lineage.validate_prior_unconsumed(empty), empty)
        for name in empty:
            changed = dict(empty, **{name: dict(total_count=1, artifacts=[dict(expired=True)])})
            with self.assertRaises(Exception):
                lineage.validate_prior_unconsumed(changed)
        with self.assertRaises(Exception):
            lineage.validate_prior_unconsumed({})

    def test_old_consumption_blocks_before_reuse_download(self):
        def gh(endpoint):
            if "actions/artifacts?name=" in endpoint:
                return json.dumps(dict(total_count=1, artifacts=[dict(expired=False)])).encode()
            return b"{}"
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(lineage, "execution_guard", return_value={}), \
             mock.patch.object(lineage, "one_execution"), mock.patch.object(lineage.auth_origin, "verify", return_value=dict(failed_run_id=45375018737)), \
             mock.patch.object(lineage.auth_origin, "validate_remote"), mock.patch.object(lineage, "gh", side_effect=gh), \
             mock.patch.object(lineage, "fetch_reuse_bundle", side_effect=AssertionError("no old budget bypass")) as fetch:
            with self.assertRaisesRegex(Exception, "every old source/provider budget"):
                lineage.authorize(Path(temp) / "authorization", "a"*40)
            fetch.assert_not_called()

    def test_original_failed_execution_and_scientific_files_are_retained(self):
        proof = lineage.auth_origin.verify()
        self.assertEqual(proof["failed_run_id"], 45375018737)
        self.assertNotEqual(lineage.BRANCH, "refs/heads/research/en-v6-b21-natural-amendment-sources-once-20261124")
        self.assertNotEqual(lineage.MARKER, "english-v6-b21-natural-amendment-source-consumed")
        lineage.auth_origin.verify_current_producer(ROOT)
        source = (HERE / "english_v6_b21_natural_amendment_source.py").read_bytes()
        self.assertEqual(lineage.digest(source), "de10ce7f2652d9bed4044db7138256dad48b8b63eb6bcab6a40ffb933ed28d44")
        natural = (HERE / "english_v6_b21_natural_amendment_natural_sources.py").read_bytes()
        self.assertEqual(lineage.digest(natural), "1d70cdaf6f16acdfa029930c68dbe74fa8ea0716ed0c365d34de4fe0080e0fd7")
        observer = (HERE / "english_v6_b21_natural_amendment_source_observe.py").read_bytes()
        self.assertEqual(lineage.digest(observer), "1afc5553d3ca372f761cfaafec0a4dcd080b34e163d3bfc9f5d50a88c7eb51c9")

    def test_original_and_new_workflows_remain_distinct(self):
        old = ROOT / ".github/workflows/english-v6-b21-natural-amendment-sources-once.yml"
        new = ROOT / lineage.WORKFLOW
        self.assertTrue(old.is_file() and new.is_file())
        self.assertIn("research/en-v6-b21-natural-amendment-sources-once-20261124", old.read_text())
        self.assertIn("research/en-v6-b21-natural-amendment-sources-auth-once-20261124", new.read_text())
        template = (HERE / "english_v6_b21_natural_amendment_source_auth_workflow_template.yml").read_text()
        self.assertEqual(template.count("__B21_NATURAL_AMENDMENT_AUTH_MAIN_SHA__"), 2)
        self.assertIn("AUTHORIZATION_STOP_LIVE.json", template)
        self.assertIn("PRIOR_BUDGETS_LIVE.json", template)


if __name__ == "__main__":
    unittest.main()
