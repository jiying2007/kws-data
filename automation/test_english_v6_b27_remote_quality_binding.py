"""Pure-data rejection tests; no network, original readback, or scientific calls."""
from __future__ import annotations
import base64
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_remote_quality_binding as helper


class Contracts(unittest.TestCase):
    def source(self):
        return dict(
            id=helper.SOURCE_RUN, head_sha=helper.SOURCE_HEAD, run_attempt=1,
            head_branch=helper.evaluator.SOURCE_BRANCH,
            path=helper.evaluator.SOURCE_WORKFLOW,
            event="push", status="completed", conclusion="success",
            repository=dict(full_name=helper.REPO, private=False))

    def test_source_rejects_nonfinal_failure_replay_and_substitution(self):
        helper.validate_source_run(self.source())
        for key, value in (
            ("status", "in_progress"), ("conclusion", None),
            ("conclusion", "failure"), ("run_attempt", 2), ("run_attempt", True),
            ("id", helper.SOURCE_RUN + 1), ("head_sha", "0" * 40),
            ("head_branch", "main"), ("event", "workflow_dispatch"),
            ("path", ".github/workflows/other.yml"),
        ):
            run = self.source()
            run[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                helper.validate_source_run(run)
        run = self.source()
        run["repository"]["private"] = True
        with self.assertRaises(ValueError):
            helper.validate_source_run(run)

    def test_nonfinal_source_stops_before_proof_download_or_binding(self):
        run = self.source()
        run["status"], run["conclusion"] = "in_progress", None
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "fresh"
            with mock.patch.object(helper.evaluator, "gh", return_value=helper.profile.canonical(run)) as gh, \
                 mock.patch.object(helper.profile, "binding") as binding, \
                 mock.patch.object(helper.original_proof, "fetch_files") as fetch, \
                 mock.patch.object(helper.binder, "bind") as bind:
                with self.assertRaises(ValueError):
                    helper.prepare(target)
                gh.assert_called_once_with("actions/runs/" + str(helper.SOURCE_RUN))
                binding.assert_not_called()
                fetch.assert_not_called()
                bind.assert_not_called()
                self.assertFalse(target.exists())

    def test_workflow_only_first_push_entry(self):
        parent, head = "1" * 40, "2" * 40
        env = dict(
            GITHUB_ACTIONS="true", GITHUB_REPOSITORY=helper.REPO,
            GITHUB_EVENT_NAME="push", GITHUB_REF=helper.BRANCH,
            GITHUB_RUN_ATTEMPT="1", GITHUB_SHA=head, PREPARED_PARENT=parent)
        event = dict(
            before=parent, after=head, ref=helper.BRANCH,
            forced=False, deleted=False,
            repository=dict(full_name=helper.REPO, private=False))
        helper.validate_entry(env, event, head, parent, [helper.WORKFLOW])
        for key, value in (
            ("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", "refs/heads/main"),
            ("GITHUB_EVENT_NAME", "pull_request"), ("PREPARED_PARENT", "3" * 40),
        ):
            bad = dict(env, **{key: value})
            with self.assertRaises(ValueError):
                helper.validate_entry(bad, event, head, parent, [helper.WORKFLOW])
        for changed in ([], [helper.WORKFLOW, "automation/changed.py"]):
            with self.assertRaises(ValueError):
                helper.validate_entry(env, event, head, parent, changed)
        for key, value in (("forced", True), ("deleted", True), ("before", "3" * 40)):
            bad_event = copy.deepcopy(event)
            bad_event[key] = value
            with self.assertRaises(ValueError):
                helper.validate_entry(env, bad_event, head, parent, [helper.WORKFLOW])

    def test_chunk_roundtrip_and_exact_five_file_manifest(self):
        files = {path: (bytes(range(256)) * 180 + b"\x00\xff\n" + str(i).encode())
                 for i, path in enumerate(helper.EXPORT_PATHS)}
        records, manifest = helper.make_export(files)
        self.assertEqual(manifest["file_count"], 5)
        self.assertEqual(manifest["total_bytes"], sum(map(len, files.values())))
        for pin in manifest["files"]:
            chunks = [r for r in records if r["path"] == pin["path"]]
            self.assertEqual([r["index"] for r in chunks], list(range(pin["chunks"])))
            self.assertTrue(all(len(r["base64"]) <= 32768 for r in chunks))
            self.assertTrue(all({k: r[k] for k in pin} == pin for r in chunks))
            raw = base64.b64decode("".join(r["base64"] for r in chunks), validate=True)
            self.assertEqual(raw, files[pin["path"]])
            self.assertEqual(len(raw), pin["bytes"])
            self.assertEqual(helper.profile.sha(raw), pin["sha256"])

    def test_export_rejects_extra_missing_empty_and_oversized_files(self):
        files = {path: b"x" for path in helper.EXPORT_PATHS}
        bad_cases = [
            {**files, "unexpected.json": b"x"},
            {k: v for k, v in files.items() if k != helper.EXPORT_PATHS[0]},
            {**files, helper.EXPORT_PATHS[0]: b""},
            {**files, helper.EXPORT_PATHS[0]: b"x" * helper.MAX_EXPORT_BYTES},
        ]
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                helper.make_export(bad)


if __name__ == "__main__":
    unittest.main()
