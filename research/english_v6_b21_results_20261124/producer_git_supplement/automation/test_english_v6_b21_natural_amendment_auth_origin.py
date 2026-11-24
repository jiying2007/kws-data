"""Offline adversarial checks for the one exact pre-consumption failure."""
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_natural_amendment_auth_origin as origin


def remote():
    return [json.loads((origin.FAILURE_DIR / name).read_bytes())
            for name in ("run.json", "jobs.json", "artifacts.json")]


class ExactFailureTests(unittest.TestCase):
    def test_exact_real_failure_preserves_earlier96_consumption(self):
        proof = origin.verify()
        self.assertEqual(proof["failed_run_id"], 45375018737)
        self.assertEqual(proof["prior_original_public_tts_calls"], 96)
        self.assertEqual(proof["original_source_status"], "STOP")
        self.assertEqual(proof["original_producer_files"], 292)
        self.assertEqual(proof["new_public_tts_calls"], 0)
        self.assertEqual(proof["new_natural_decodes"], 0)
        self.assertEqual(proof["new_observer_calls"], 0)
        self.assertFalse(proof["execution_authorized"])
        self.assertFalse(proof["source_budget_uploaded"])

    def test_live_same_exact_complete_failure_is_accepted(self):
        self.assertEqual(origin.validate_remote(*remote())["failed_run_id"], origin.RUN)

    def test_retry_success_running_wrong_head_and_private_are_rejected(self):
        changes = [dict(run_attempt=2), dict(conclusion="success"), dict(status="in_progress"),
                   dict(head_sha="0" * 40), dict(head_branch="main"), dict(event="workflow_dispatch")]
        for change in changes:
            values = remote(); values[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                origin.validate_remote(*values)
        for key in ("repository", "head_repository"):
            values = remote(); values[0][key]["private"] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                origin.validate_remote(*values)

    def test_entered_transport_consumption_or_generation_cannot_be_zero_call(self):
        for number in (7, 9, 10, 11):
            values = remote()
            step = next(s for s in values[1]["jobs"][0]["steps"] if s["number"] == number)
            step["conclusion"] = "success"
            with self.subTest(step=number), self.assertRaises(ValueError):
                origin.validate_remote(*values)

    def test_entered_observer_or_incomplete_job_listing_is_rejected(self):
        values = remote(); values[1]["jobs"][1]["conclusion"] = "success"
        with self.assertRaises(ValueError): origin.validate_remote(*values)
        values = remote(); values[1]["jobs"][1]["steps"] = [dict(name="observe", number=1)]
        with self.assertRaises(ValueError): origin.validate_remote(*values)
        values = remote(); values[1]["jobs"].pop()
        with self.assertRaises(ValueError): origin.validate_remote(*values)

    def test_source_consumed_marker_or_source_artifact_is_rejected(self):
        for name in ("english-v6-b21-natural-amendment-source-consumed",
                     "english-v6-b21-natural-amendment-source-45375018737"):
            values = remote(); extra = copy.deepcopy(values[2]["artifacts"][0])
            extra.update(id=123, name=name); values[2]["artifacts"].append(extra); values[2]["total_count"] = 3
            with self.subTest(name=name), self.assertRaises(ValueError):
                origin.validate_remote(*values)

    def test_wrong_expired_or_foreign_original_artifact_is_rejected(self):
        for change in (dict(expired=True), dict(size_in_bytes=1), dict(digest="sha256:" + "0" * 64)):
            values = remote(); values[2]["artifacts"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                origin.validate_remote(*values)
        values = remote(); values[2]["artifacts"][0]["workflow_run"]["id"] = 45367984097
        with self.assertRaises(ValueError): origin.validate_remote(*values)

    def test_all_compact_evidence_files_are_content_addressed(self):
        paths = origin.resource_paths()
        self.assertEqual(len(paths), 16)
        self.assertTrue(all(name.startswith("automation/" + origin.FAILURE_DIR.name + "/") for name in paths))
        self.assertFalse(any(name.endswith("/producer-original.zip") for name in paths))

    def test_truncated_log_modified_manifest_missing_and_extra_evidence_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "proof"
            for mode in ("log", "manifest", "missing", "extra"):
                if folder.exists(): shutil.rmtree(folder)
                shutil.copytree(origin.FAILURE_DIR, folder)
                if mode == "log": (folder / "source-job.log").write_bytes(b"new log")
                elif mode == "manifest": (folder / "MANIFEST.json").write_text("{}")
                elif mode == "missing": (folder / "transport-original.zip").unlink()
                else: (folder / "CONSUMPTION.json").write_text("{}")
                with self.subTest(mode=mode), self.assertRaises(ValueError): origin.verify(folder)

    def test_original_producer_manifest_and_raw_operational_snapshots_match_git(self):
        raw = origin._files(origin.FAILURE_DIR)
        producer = origin._producer(raw)
        self.assertEqual(len(producer["files"]), 292)
        for target in ("tree", "manifest", "snapshot"):
            changed = dict(raw)
            if target == "tree":
                tree = json.loads(changed["tree.json"])
                next(entry for entry in tree["tree"] if entry["path"] == origin.EXTRA_METADATA_WORKFLOW)["sha"] = "0" * 40
                changed["tree.json"] = json.dumps(tree).encode()
            elif target == "manifest":
                doc = json.loads(changed["producer-manifest.json"])
                doc["files"][origin.TEMPLATE]["sha256"] = "0" * 64
                changed["producer-manifest.json"] = json.dumps(doc).encode()
            else: changed["original-source-ancestry.py"] += b"\n# altered\n"
            with self.subTest(target=target), self.assertRaises(ValueError): origin._producer(changed)

    def test_only_three_named_operational_files_and_old_activation_are_exempt(self):
        self.assertEqual(origin.OPERATIONAL_CHANGES, {
            "automation/english_v6_b21_natural_amendment_source_ancestry.py",
            "automation/english_v6_b21_natural_amendment_source_producer.py",
            "automation/test_english_v6_b21_natural_amendment_sources.py"})
        value = origin.verify_current_producer()
        self.assertEqual(value["preserved_current_files"], 288)

    def test_modifying_old_consumed_science_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            document = json.loads((origin.FAILURE_DIR / "producer-manifest.json").read_bytes())
            for name in document["files"]:
                destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin.ROOT / name, destination)
            shutil.copytree(origin.FAILURE_DIR, root / "automation" / origin.FAILURE_DIR.name)
            self.assertEqual(origin.verify_current_producer(root)["preserved_current_files"], 288)
            target = root / "automation/english_v6_b21_reuse_synthetic.py"
            target.write_bytes(target.read_bytes() + b"\n# mutation\n")
            with self.assertRaises(ValueError): origin.verify_current_producer(root)

    def test_proof_requires_no_network_subprocess_or_scientific_runtime(self):
        with mock.patch.object(socket, "socket", side_effect=AssertionError("network forbidden")), \
             mock.patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")), \
             mock.patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess forbidden")), \
             mock.patch.object(os, "system", side_effect=AssertionError("execution forbidden")):
            self.assertEqual(origin.verify()["new_source_calls"], 0)
        self.assertFalse(any(name.split(".")[0] in {"torch", "numpy", "onnxruntime", "soundfile", "piper", "transformers"}
                             for name in sys.modules))

    def test_separately_retained_original_archive_when_supplied(self):
        value = os.environ.get("B21_AUTH_ORIGINAL_PRODUCER")
        if value is None: self.skipTest("original ZIP is retained externally, not recursively embedded")
        self.assertEqual(origin.verify_original_producer_archive(value)["original_zip_members"], 293)


if __name__ == "__main__":
    unittest.main()
