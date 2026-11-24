"""Fixed original stop proof regressions; stdlib only, no source/model/network."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_continuation_origin as origin


def recorded_api_fixture():
    root = origin.FAILURE_DIR / "api"
    run = json.loads((root / "run-final.json").read_bytes())
    jobs = json.loads((root / "jobs-final.json").read_bytes())
    artifacts = json.loads((root / "artifacts-final.json").read_bytes())
    # Invented full-page envelopes are test inputs, not live API receipts.
    jobs["total_count"] = len(jobs["jobs"])
    artifacts["total_count"] = len(artifacts["artifacts"])
    return run, jobs, artifacts


class OriginalStop(unittest.TestCase):
    def test_exact_original_payloads_verify_zero_source_with_preserved_identity(self):
        result = origin.verify()
        self.assertEqual(result["original_run_id"], 45352507569)
        self.assertEqual(result["original_run_head"], origin.HEAD)
        self.assertEqual(result["original_producer_pin"]["id"], 14900722688)
        self.assertEqual(result["original_opened_pin"]["id"], 14901610526)
        self.assertEqual(result["failure_proof_manifest_sha256"], origin.MANIFEST_SHA256)
        self.assertTrue(result["original_opened_marker_preserved"])
        self.assertFalse(result["source_generation_started"])
        for key in ("actual_tts_calls", "natural_original_decodes", "observer_calls", "kws_forwards"):
            self.assertEqual(result[key], 0)
        self.assertNotIn("torch", sys.modules); self.assertNotIn("numpy", sys.modules)

    def test_live_validation_requires_complete_unambiguous_api_pages(self):
        run, jobs, artifacts = recorded_api_fixture()
        result = origin.validate_remote(run, jobs, artifacts)
        self.assertEqual((result["complete_jobs"], result["complete_artifacts"]), (3, 2))
        for value in (None, 4):
            bad = copy.deepcopy(jobs)
            if value is None: bad.pop("total_count")
            else: bad["total_count"] = value
            with self.assertRaises(ValueError): origin.validate_remote(run, bad, artifacts)
        bad = copy.deepcopy(artifacts); bad["total_count"] = 3
        with self.assertRaises(ValueError): origin.validate_remote(run, jobs, bad)

    def test_any_source_artifact_missing_marker_or_unknown_native_job_is_ineligible(self):
        run, jobs, artifacts = recorded_api_fixture()
        bad = copy.deepcopy(artifacts); item = copy.deepcopy(bad["artifacts"][0])
        item.update(id=1, name="english-v6-b21-primary-source-45352507569")
        bad["artifacts"].append(item); bad["total_count"] = 3
        with self.assertRaises(ValueError): origin.validate_remote(run, jobs, bad)
        bad = copy.deepcopy(artifacts); bad["artifacts"] = bad["artifacts"][1:]; bad["total_count"] = 1
        with self.assertRaises(ValueError): origin.validate_remote(run, jobs, bad)
        bad = copy.deepcopy(jobs)
        for job in bad["jobs"]:
            if job["name"] == "observers": job.update(conclusion="failure", steps=[dict(number=1, name="MODEL_LOAD_STARTED", status="completed", conclusion="failure")])
        with self.assertRaises(ValueError): origin.validate_remote(run, bad, artifacts)
        bad = copy.deepcopy(jobs)
        for job in bad["jobs"]:
            if job["name"] == "sources": job["steps"][7]["conclusion"] = "success"
        with self.assertRaises(ValueError): origin.validate_remote(run, bad, artifacts)

    def test_rerun_success_wrong_head_or_replaced_artifact_cannot_authorize(self):
        run, jobs, artifacts = recorded_api_fixture()
        for field, value in (("run_attempt", 2), ("conclusion", "success"), ("head_sha", "a" * 40)):
            bad = copy.deepcopy(run); bad[field] = value
            with self.assertRaises(ValueError): origin.validate_remote(bad, jobs, artifacts)
        bad = copy.deepcopy(artifacts); bad["artifacts"][0]["digest"] = "sha256:" + "b" * 64
        with self.assertRaises(ValueError): origin.validate_remote(run, jobs, bad)

    def test_tampered_log_manifest_or_extra_evidence_is_not_a_zero_call_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "proof"; shutil.copytree(origin.FAILURE_DIR, root)
            path = root / "api/source-job.log"; raw = path.read_bytes(); path.write_bytes(raw + b"invented no-call claim")
            with self.assertRaisesRegex(ValueError, "unchanged original evidence"):
                origin.verify(root)
            path.write_bytes(raw)
            extra = root / "unreported-source.wav"; extra.write_bytes(b"unknown generation attempt")
            with self.assertRaisesRegex(ValueError, "all and only"):
                origin.verify(root)
            extra.unlink()
            manifest = root / "MANIFEST.json"; manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "immutable nine-file"):
                origin.verify(root)


if __name__ == "__main__": unittest.main()
