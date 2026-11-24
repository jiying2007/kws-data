"""Original-evidence checks for the B22 zero-text-call SOURCE continuation."""
from __future__ import annotations
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b22_source_origin as origin


def zero_archive(provider, change=None):
    directory = origin.EVIDENCE / "observers" / provider
    files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    if change:
        change(files)
        closure = json.loads(files["FILES.json"])
        closure["files"] = [dict(path=name, bytes=len(raw), sha256=origin.sha(raw))
                            for name, raw in sorted(files.items()) if name != "FILES.json"]
        files["FILES.json"] = json.dumps(closure).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))


class OriginalSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = origin.local()
        cls.rows = json.loads((origin.EVIDENCE / "source/MANIFEST.json").read_bytes())["rows"]

    def test_full_original_failure_and_consumption_are_retained(self):
        self.assertEqual(len(self.proof["artifacts"]), 14)
        self.assertEqual(len(self.proof["jobs"]), 7)
        self.assertEqual(len(origin.producer_manifest()["files"]), 1172)
        self.assertEqual(self.proof["remaining_text_forward_calls"], 360)
        self.assertEqual(self.proof["phone_forward_counts"], {"xlsr": 120, "lv60": 120})
        self.assertEqual(set(self.proof["missing_helpers"]), {
            "research/auto-tts-voice-clone-20261102/campaign.py",
            "research/auto-tts-voice-clone-20261102/runner.py"})
        origin.validate_live(origin.EVIDENCE, self.proof)

    def test_every_original_text_slot_was_never_entered(self):
        for provider in origin.TEXT_PROVIDERS:
            with self.subTest(provider=provider), zero_archive(provider) as archive:
                self.assertEqual(origin.observer_receipt(provider, archive, self.rows, self.proof), 0)

    def test_any_hidden_forward_or_reported_attempt_refuses_continuation(self):
        def hidden(files):
            files["rows/" + self.rows[0]["id"] + "/FORWARD_STARTED.json"] = b'{"attempt":1}'
        def attempted(files):
            value = json.loads(files["RESULT.json"])
            value["actual_native_forward_attempts"] = 1
            files["RESULT.json"] = json.dumps(value).encode()
        for change in (hidden, attempted):
            with self.subTest(change=change.__name__), zero_archive("qwen", change) as archive:
                with self.assertRaises(ValueError):
                    origin.observer_receipt("qwen", archive, self.rows, self.proof)

    def test_original_marker_cannot_be_reset_or_replaced(self):
        def reset(files):
            budget = json.loads(files["OBSERVER_BUDGET.json"])
            budget["receipt"]["status"] = "UNCONSUMED"
            files["OBSERVER_BUDGET.json"] = json.dumps(budget).encode()
        with zero_archive("whisper", reset) as archive:
            with self.assertRaises(ValueError):
                origin.observer_receipt("whisper", archive, self.rows, self.proof)

    def test_rerun_inventory_change_or_log_normalization_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="b22-origin-contract-") as temporary:
            directory = Path(temporary)
            for name in ("RUN", "JOBS", "ARTIFACTS", "COMMIT"):
                shutil.copyfile(origin.EVIDENCE / (name + ".json"), directory / (name + ".json"))
            shutil.copytree(origin.EVIDENCE / "logs", directory / "logs")
            run_path = directory / "RUN.json"
            original = run_path.read_bytes()
            value = json.loads(original); value["run_attempt"] = 2
            run_path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                origin.validate_live(directory, self.proof)
            run_path.write_bytes(original)
            artifacts_path = directory / "ARTIFACTS.json"
            original = artifacts_path.read_bytes()
            value = json.loads(original); value["artifacts"] = value["artifacts"][:-1]; value["total_count"] -= 1
            artifacts_path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                origin.validate_live(directory, self.proof)
            artifacts_path.write_bytes(original)
            log_path = directory / "logs/138475905417.log"
            log_path.write_bytes(log_path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                origin.validate_live(directory, self.proof)

    def test_fetch_never_mutates_remote_or_accepts_existing_destination(self):
        with tempfile.TemporaryDirectory(prefix="b22-origin-existing-") as temporary:
            with patch.object(origin, "gh", side_effect=AssertionError("network forbidden")):
                with self.assertRaises(ValueError):
                    origin.fetch_origin(Path(temporary))
        self.assertFalse({"torch", "numpy", "onnxruntime"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main()
