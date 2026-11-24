"""Failure tests for exact reuse; optional original-archive readback stays offline."""
from __future__ import annotations

import builtins
from contextlib import ExitStack
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_reuse_synthetic as reuse


def remote_fixture():
    public = dict(full_name=reuse.REPO, private=False)
    run = dict(id=reuse.RUN, head_sha=reuse.HEAD, run_attempt=1, status="completed", conclusion="failure",
        event="push", path=reuse.WORKFLOW, head_branch=reuse.BRANCH, repository=public, head_repository=public)
    states = [(n, "success") for n in range(1, 11)] + [
        (11, "failure"), (12, "success"), (23, "skipped"), (24, "success"), (25, "success")]
    jobs = dict(total_count=3, jobs=[dict(id=138354602305, name="sources", run_id=reuse.RUN,
        status="completed", conclusion="failure", steps=[dict(number=n, conclusion=status, status="completed",
        name="Generate the original96synthetic and24whole natural sources exactly once" if n == 11 else "fixture")
        for n, status in states])])
    jobs["jobs"] += [dict(id=ident, name=name, run_id=reuse.RUN, status="completed", conclusion="skipped", steps=[])
                      for name, ident in (("observers", 138356964398), ("readback", 138356964588))]
    artifacts = dict(total_count=4, artifacts=[dict(id=p["id"], size_in_bytes=p["bytes"],
        digest="sha256:" + p["sha256"], expired=False,
        workflow_run=dict(id=reuse.RUN, head_sha=reuse.HEAD, head_branch=reuse.BRANCH))
        for p in reuse.BUNDLE_ARTIFACTS.values() if p["run_id"] == reuse.RUN])
    return run, jobs, artifacts


def offline_only():
    stack, original = ExitStack(), builtins.__import__
    forbidden = {"numpy", "torch", "torchaudio", "onnxruntime", "soundfile", "transformers", "piper", "requests", "httpx"}
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError("no scientific/network runtime during original readback: " + name)
        return original(name, *args, **kwargs)
    stack.enter_context(mock.patch("builtins.__import__", side_effect=guarded))
    stack.enter_context(mock.patch("subprocess.Popen", side_effect=AssertionError("no external execution")))
    stack.enter_context(mock.patch("socket.socket", side_effect=AssertionError("no network")))
    return stack


class RemoteConsumptionTests(unittest.TestCase):
    def test_exact_failure_is_accepted_without_claiming_zero_generation(self):
        reuse.validate_remote(*remote_fixture())

    def test_retry_success_private_or_foreign_head_fails(self):
        for key, value in (("run_attempt", 2), ("conclusion", "success"), ("head_sha", "f" * 40)):
            with self.subTest(key=key):
                run, jobs, artifacts = remote_fixture()
                run[key] = value
                with self.assertRaises(ValueError):
                    reuse.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = remote_fixture()
        run["repository"]["private"] = True
        with self.assertRaises(ValueError):
            reuse.validate_remote(run, jobs, artifacts)

    def test_generation_skipped_or_source_upload_failed_is_not_this_origin(self):
        for number, state in ((9, "skipped"), (11, "skipped"), (11, "success"), (12, "failure")):
            with self.subTest(number=number, state=state):
                run, jobs, artifacts = remote_fixture()
                next(s for s in jobs["jobs"][0]["steps"] if s["number"] == number)["conclusion"] = state
                with self.assertRaises(ValueError):
                    reuse.validate_remote(run, jobs, artifacts)

    def test_started_observer_or_readback_fails(self):
        for index in (1, 2):
            run, jobs, artifacts = remote_fixture()
            jobs["jobs"][index].update(conclusion="failure", steps=[dict(number=1, conclusion="failure")])
            with self.assertRaises(ValueError):
                reuse.validate_remote(run, jobs, artifacts)

    def test_missing_marker_or_extra_observer_artifact_fails(self):
        run, jobs, artifacts = remote_fixture()
        artifacts["artifacts"] = [a for a in artifacts["artifacts"] if a["id"] != 14908181309]
        artifacts["total_count"] = 3
        with self.assertRaises(ValueError):
            reuse.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = remote_fixture()
        artifacts["artifacts"].append(dict(artifacts["artifacts"][0], id=1))
        artifacts["total_count"] = 5
        with self.assertRaises(ValueError):
            reuse.validate_remote(run, jobs, artifacts)

    def test_truncated_inventory_or_wrong_artifact_digest_fails(self):
        run, jobs, artifacts = remote_fixture()
        jobs["total_count"] = 4
        with self.assertRaises(ValueError):
            reuse.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = remote_fixture()
        artifacts["artifacts"][0]["digest"] = "sha256:" + "0" * 64
        with self.assertRaises(ValueError):
            reuse.validate_remote(run, jobs, artifacts)

    def test_unpinned_valid_zip_is_rejected_before_member_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source-original.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("SYNTHETIC_MANIFEST.json", "{}")
            with self.assertRaises(ValueError):
                reuse.opened(temporary, path.name)

    def test_empty_history_and_prefix_reuse_exemption_fail(self):
        with self.assertRaises(ValueError):
            reuse.validate_complete_history(dict(all_pcm_sha256=[], all_source_groups=[], canonical_people=[]))
        proof = dict(schema=reuse.SCHEMA, source_pin=reuse.SOURCE_PIN, original_identity=reuse.IDENTITY,
                     synthetic_manifest_sha256=reuse.MANIFEST_SHA256, rows=[dict(id="b21-primary-846-mero-fast")])
        with self.assertRaises(ValueError):
            reuse.validate_reused_rows(proof["rows"], proof)

    def test_canonical_person_normalizes_leading_zeros(self):
        self.assertEqual(reuse.canonical_person("p000210"), "210")
        self.assertEqual(reuse.canonical_person("000210"), "210")
        for invalid in (210, "reader-210", "p210-other", ""):
            with self.assertRaises(ValueError):
                reuse.canonical_person(invalid)


class OriginalArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("B21_REUSE_TEST_BUNDLE")
        if not configured:
            raise unittest.SkipTest("set B21_REUSE_TEST_BUNDLE for the original cached-archive integration readback")
        cls.bundle = Path(configured)
        with offline_only():
            cls.proof = reuse.verify(cls.bundle)

    def test_all_originals_and_complete_cumulative_history(self):
        proof = self.proof
        self.assertEqual(len(proof["rows"]), 96)
        self.assertEqual(proof["original_members"], 290)
        self.assertEqual(len(proof["prior_history"]["all_pcm_sha256"]), 2343)
        self.assertEqual(len(proof["history"]["all_pcm_sha256"]), 2439)
        self.assertEqual(proof["consumed_source_stop"]["actual_public_piper_synthesize_calls"], 96)
        self.assertEqual(proof["consumed_source_stop"]["actual_natural_original_decodes"], 0)
        self.assertFalse(proof["consumed_source_stop"]["complete_source_gate_pass"])
        self.assertFalse(proof["regeneration_allowed"])
        self.assertTrue(all(value == 0 for value in proof["zero_new_calls"].values()))

    def test_history_cannot_omit_old_or_consumed_rows(self):
        for key in ("all_pcm_sha256", "all_source_groups", "canonical_people", "natural_excluded_people"):
            history = deepcopy(self.proof["history"])
            history[key].pop()
            with self.assertRaises(ValueError):
                reuse.validate_complete_history(history)

    def test_reuse_whitelist_requires_exact_full_ordered_rows(self):
        self.assertEqual(reuse.validate_reused_rows(self.proof["rows"], self.proof), self.proof["rows"])
        for rows in (self.proof["rows"][:-1], list(reversed(self.proof["rows"]))):
            with self.assertRaises(ValueError):
                reuse.validate_reused_rows(rows, self.proof)
        forged = deepcopy(self.proof)
        forged["rows"][0]["pcm_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            reuse.validate_reused_rows(forged["rows"], forged)

    def test_new_natural_has_no_pcm_group_or_person_exemption(self):
        row = dict(id="new-natural", cohort="natural", speaker="999999999", pcm_sha256="a" * 64,
                   source_group_id="new-natural-group")
        reuse.require_new_natural_disjoint([row], self.proof)
        original = self.proof["rows"][0]
        for change in (dict(pcm_sha256=original["pcm_sha256"]), dict(source_group_id=original["source_group_id"]),
                       dict(speaker=original["source_speaker_id"]),
                       dict(speaker="000" + original["source_speaker_id"].removeprefix("p"))):
            with self.assertRaises(ValueError):
                reuse.require_new_natural_disjoint([dict(row, **change)], self.proof)
        with self.assertRaises(ValueError):
            reuse.require_new_natural_disjoint([row, dict(row, id="other-id")], self.proof)

    def test_materialize_preserves_all_original_bytes_and_refuses_resume(self):
        with tempfile.TemporaryDirectory() as temporary, offline_only():
            output = Path(temporary) / "source"
            proof = reuse.materialize(self.bundle, output)
            with reuse.opened(self.bundle, "source-original.zip") as archive:
                self.assertEqual({str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}, set(archive.namelist()))
                for name in archive.namelist():
                    self.assertEqual((output / name).read_bytes(), archive.read(name))
            (output / "AMENDMENT_STARTED.json").write_text("{}")
            self.assertEqual(reuse.validate_reused_source(output, proof), proof["rows"])
            with self.assertRaises(ValueError):
                reuse.materialize(self.bundle, output)
            started = output / "STARTED.json"
            original_started = started.read_bytes()
            started.write_bytes(original_started + b"\n")
            with self.assertRaises(ValueError):
                reuse.validate_reused_source(output, proof)
            started.write_bytes(original_started)
            path = output / "SYNTHETIC_MANIFEST.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                reuse.validate_reused_source(output, proof)


if __name__ == "__main__":
    unittest.main()
