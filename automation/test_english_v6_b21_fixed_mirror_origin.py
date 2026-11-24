"""Pure original-data checks for the exact second zero-source transport stop."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_fixed_mirror_origin as origin


class OriginTests(unittest.TestCase):
    def records(self):
        return [json.loads((origin.FAILURE_DIR / "api" / (name + ".json")).read_bytes())
                for name in ("run-final", "jobs-final", "artifacts-final")]

    def test_actual_two_original_failures_and_all245_unchanged_producer_files(self):
        value = origin.verify()
        self.assertEqual(value["original_run_id"], 45352507569)
        self.assertEqual(value["original_transport_origin"], origin.original.verify())
        self.assertEqual(value["failed_continuation_run_id"], 45365292922)
        self.assertEqual(value["failed_continuation_producer_files"], 246)
        self.assertFalse(value["failed_continuation_consumption_started"])
        self.assertFalse(value["failed_continuation_source_generation_started"])
        self.assertTrue(value["live_refresh_required_for_both_failed_runs"])
        self.assertEqual(len(origin.verify_current_producer()), 245)
        self.assertTrue(origin.validate_remote(*self.records())["no_consumption_or_source_artifact"])

    def test_any_consumption_or_source_artifact_forbids_new_transport(self):
        for name in ("english-v6-b21-source-continuation-consumed", "english-v6-b21-primary-continuation-source-45365292922"):
            with self.subTest(name=name):
                run, jobs, artifacts = self.records()
                item = copy.deepcopy(artifacts["artifacts"][0]); item.update(id=9, name=name)
                artifacts["artifacts"].append(item); artifacts["total_count"] += 1
                with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)

    def test_started_or_completed_consumption_or_generation_is_not_zero_call_proof(self):
        for number in (9, 10, 11):
            for conclusion in ("success", "failure"):
                with self.subTest(step=number, conclusion=conclusion):
                    run, jobs, artifacts = self.records()
                    step = next(s for s in jobs["jobs"][0]["steps"] if s["number"] == number)
                    step["conclusion"] = conclusion
                    with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)

    def test_native_or_readback_job_cannot_have_started(self):
        for index in (1, 2):
            run, jobs, artifacts = self.records()
            jobs["jobs"][index]["conclusion"] = "success"
            with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)
            run, jobs, artifacts = self.records()
            jobs["jobs"][index]["steps"] = [{"name": "Any started operation"}]
            with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)

    def test_rerun_changed_head_or_unfinished_run_stays_closed(self):
        for key, value in (("run_attempt", 2), ("id", 45352507569), ("head_sha", "0" * 40),
                           ("status", "in_progress"), ("conclusion", "cancelled"), ("event", "workflow_dispatch")):
            with self.subTest(key=key):
                run, jobs, artifacts = self.records(); run[key] = value
                with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)

    def test_private_or_foreign_repository_is_rejected(self):
        for field in ("repository", "head_repository"):
            run, jobs, artifacts = self.records(); run[field]["private"] = True
            with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)
            run, jobs, artifacts = self.records(); run[field]["full_name"] = "other/kws-data"
            with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)

    def test_full_inventory_required_and_duplicate_jobs_rejected(self):
        run, jobs, artifacts = self.records(); jobs["total_count"] = 4
        with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = self.records(); artifacts.pop("total_count")
        with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = self.records(); jobs["jobs"][1] = copy.deepcopy(jobs["jobs"][2])
        with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)

    def test_expired_replaced_or_wrong_run_artifact_rejected(self):
        for mutation in ("expired", "digest", "run"):
            run, jobs, artifacts = self.records(); item = artifacts["artifacts"][0]
            if mutation == "expired": item["expired"] = True
            elif mutation == "digest": item["digest"] = "sha256:" + "0" * 64
            else: item["workflow_run"]["id"] = 1
            with self.assertRaises(ValueError): origin.validate_remote(run, jobs, artifacts)

    def test_modified_or_added_local_proof_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "proof"; shutil.copytree(origin.FAILURE_DIR, target)
            log = target / "api/source-job.log"; log.write_bytes(log.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "unchanged original evidence"):
                origin.verify(target)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "proof"; shutil.copytree(origin.FAILURE_DIR, target)
            (target / "CONSUMPTION.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "all and only"):
                origin.verify(target)

    def test_first_original_failure_cannot_be_omitted_or_reclassified(self):
        with patch.object(origin.original, "verify", side_effect=ValueError("first original proof unavailable")):
            with self.assertRaisesRegex(ValueError, "first original proof unavailable"):
                origin.verify()


if __name__ == "__main__":
    unittest.main()
