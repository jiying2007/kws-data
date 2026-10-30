"""Offline contracts only; no dependencies, model execution or network required.

Fixtures are invented; these tests never download or run a real model.
"""
from __future__ import annotations

import copy
import json
import hashlib
import struct
from unittest import mock
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_contract as c


class CampaignContractTests(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.env = {"GITHUB_REPOSITORY": c.REPOSITORY, "GITHUB_REF": "refs/heads/main",
                    "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_RUN_NUMBER": "1", "GITHUB_SHA": self.sha, "GITHUB_RUN_ID": "1234"}
        self.event = {"repository": {"full_name": c.REPOSITORY, "private": False,
                                     "default_branch": "main"},
                      "inputs": {"expected_source_sha": self.sha}}

    def test_fixed_matrix(self):
        rows = c.validate_plan(c.expected_plan())
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r["cell_id"] for r in rows}), 16)
        self.assertEqual(sum(r["proposed_split"] == "train" for r in rows), 12)
        self.assertEqual(sum(r["proposed_split"] == "development_a" for r in rows), 4)

    def test_serena_cannot_move_to_training(self):
        rows = c.expected_plan()
        next(r for r in rows if r["speaker"] == "Serena")["proposed_split"] = "train"
        with self.assertRaises(ValueError):
            c.validate_plan(rows)

    def test_seed_cannot_change(self):
        rows = c.expected_plan()
        rows[0]["seed"] += 1
        with self.assertRaises(ValueError):
            c.validate_plan(rows)

    def test_intended_text_not_a_label(self):
        for row in c.expected_plan():
            self.assertIsNone(row["actual_text"])
            self.assertIsNone(row["actual_tokens"])
            self.assertIs(row["ctc_training_allowed"], False)

    def test_admission_flag_mutation_rejected(self):
        for value in (True, "false", 0, None):
            with self.subTest(value=value):
                rows = c.expected_plan()
                rows[0]["ctc_training_allowed"] = value
                with self.assertRaises(ValueError):
                    c.validate_plan(rows)

    def test_family_not_seed_local(self):
        for voice in c.VOICES:
            rows = [r for r in c.expected_plan() if r["speaker"] == voice]
            self.assertEqual(len({r["source_family_id"] for r in rows}), 1)
            self.assertNotIn("20261030", rows[0]["source_family_id"])

    def test_valid_first_dispatch(self):
        c.validate_launch(self.env, self.event, self.sha)

    def test_rerun_rejected(self):
        self.env["GITHUB_RUN_ATTEMPT"] = "2"
        with self.assertRaises(ValueError):
            c.validate_launch(self.env, self.event, self.sha)

    def test_second_dispatch_rejected(self):
        self.env["GITHUB_RUN_NUMBER"] = "2"
        with self.assertRaises(ValueError):
            c.validate_launch(self.env, self.event, self.sha)

    def test_private_repository_rejected(self):
        for value in (True, "false", 0, None):
            with self.subTest(value=value):
                event = copy.deepcopy(self.event)
                event["repository"]["private"] = value
                with self.assertRaises(ValueError):
                    c.validate_launch(self.env, event, self.sha)

    def test_wrong_branch_repository_or_event_rejected(self):
        for key, value in (("GITHUB_REF", "refs/heads/topic"),
                           ("GITHUB_REPOSITORY", "other/repo"),
                           ("GITHUB_EVENT_NAME", "pull_request")):
            with self.subTest(key=key):
                env = dict(self.env)
                env[key] = value
                with self.assertRaises(ValueError):
                    c.validate_launch(env, self.event, self.sha)

    def test_source_sha_mismatch_rejected(self):
        self.event["inputs"]["expected_source_sha"] = "b" * 40
        with self.assertRaises(ValueError):
            c.validate_launch(self.env, self.event, self.sha)

    def test_noncanonical_sha_rejected(self):
        for sha in ("main", "A" * 40, "a" * 39, "a" * 41):
            with self.subTest(sha=sha), self.assertRaises(ValueError):
                c.validate_launch(self.env, self.event, sha)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"result":false,"result":true}', encoding="utf-8")
            with self.assertRaises(ValueError):
                c.read_json(path)

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                c.read_json(path)

    def test_size_bound_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "big.json"
            path.write_text('"' + "x" * 100 + '"', encoding="utf-8")
            with self.assertRaises(ValueError):
                c.read_json(path, 10)

    def test_audio_technical_valid_not_admitted(self):
        result = c.signal_metrics([0.1] * 2400, 24000, eos=True)
        self.assertTrue(result["technically_valid"])
        self.assertIs(result["training_admitted"], False)

    def test_native_eos_required(self):
        for eos in (None, False, 1, "true"):
            with self.subTest(eos=eos):
                self.assertFalse(c.signal_metrics([0.1] * 100, 24000, eos=eos)["technically_valid"])

    def test_silence_and_clipping_not_repaired(self):
        for values in ([0.0] * 10, [1.0] * 10, [-1.1] * 10):
            with self.subTest(values=values):
                self.assertFalse(c.signal_metrics(values, 24000, eos=True)["technically_valid"])

    def test_invalid_audio_rejected(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                c.signal_metrics(values, 24000, eos=True)

    def test_empty_ledger_is_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = c.terminal_ledger(Path(directory), {"outcomes": []})
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r["status"] == "NOT_RUN" for r in rows))

    def test_interrupted_claim_consumes_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cell = c.expected_plan()[0]["cell_id"]
            (output / (cell + ".claim")).write_bytes(b"")
            rows = c.terminal_ledger(output, {"outcomes": []})
            self.assertEqual((rows[0]["status"], rows[0]["attempts"]), ("FAILED_NO_RETRY", 1))

    def test_outcome_without_claim_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = {"outcomes": [{"cell_id": c.expected_plan()[0]["cell_id"],
                                      "status": "generated_pending_review"}]}
            with self.assertRaises(ValueError):
                c.terminal_ledger(Path(directory), receipt)

    def test_duplicate_outcome_rejected(self):
        row = {"cell_id": c.expected_plan()[0]["cell_id"]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                c.terminal_ledger(Path(directory), {"outcomes": [row, row]})

    def test_unknown_claim_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "unknown.claim").write_bytes(b"")
            with self.assertRaises(ValueError):
                c.terminal_ledger(output, {"outcomes": []})

    def test_nonprefix_claim_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cell = c.expected_plan()[1]["cell_id"]
            (output / (cell + ".claim")).write_text(cell + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                c.terminal_ledger(output, {"outcomes": []})

    def test_missing_completed_audio_rejected(self):
        row = dict(c.expected_plan()[0], status="generated_pending_review", attempts=1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                c.verify_completed_outputs(Path(directory), {"outcomes": [row]})

    def test_completed_label_drift_rejected(self):
        row = dict(c.expected_plan()[0], status="generated_pending_review", attempts=1,
                   actual_text="你好小窝")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                c.verify_completed_outputs(Path(directory), {"outcomes": [row]})

    def test_source_freeze_matches_delivered_bytes(self):
        root = Path(__file__).resolve().parent
        freeze = c.read_json(root / "source-freeze.json")
        self.assertEqual(freeze["donor_commit"], c.DONOR_COMMIT)
        for name, digest in freeze["files"].items():
            self.assertEqual(c.file_sha256(root / name), digest)

    def test_atomic_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            c.save_json(path, {"status": "started"})
            c.save_json(path, {"status": "pending_review"})
            self.assertEqual(c.read_json(path), {"status": "pending_review"})


    def test_valid_first_execution_branch_creation(self):
        env = dict(self.env, GITHUB_EVENT_NAME="push", GITHUB_REF="refs/heads/" + c.EXECUTION_BRANCH)
        event = dict(self.event, ref=env["GITHUB_REF"], created=True, deleted=False,
                     forced=False, before="0" * 40, after=self.sha)
        c.validate_launch(env, event, self.sha)
        for key, value in (("created", False), ("deleted", True), ("forced", True),
                           ("before", "b" * 40), ("after", "b" * 40), ("created", 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                c.validate_launch(env, dict(event, **{key: value}), self.sha)

    def test_json_exponent_overflow_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overflow.json"
            path.write_text('{"value":1e400}', encoding="utf-8")
            with self.assertRaises(ValueError):
                c.read_json(path)

    def test_stale_temporary_does_not_block_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            stale = path.with_name(path.name + ".tmp")
            stale.write_bytes(b"interrupted")
            c.save_json(path, {"status": "failed_no_retry"})
            self.assertEqual(c.read_json(path)["status"], "failed_no_retry")
            self.assertEqual(stale.read_bytes(), b"interrupted")

    def test_replace_failure_keeps_original_and_cleans_only_owned_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            c.save_json(path, {"status": "started"})
            before = path.read_bytes()
            with mock.patch.object(c.os, "replace", side_effect=OSError("fixture")):
                with self.assertRaises(OSError):
                    c.save_json(path, {"status": "failed_no_retry"})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["receipt.json"])

    def make_audio_pair(self, output):
        row = dict(c.expected_plan()[0], status="generated_pending_review", attempts=1)
        def wav(payload, tag, rate, bits):
            width = bits // 8
            fmt = struct.pack("<HHIIHH", tag, 1, rate, rate * width, width, bits)
            body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
            return b"RIFF" + struct.pack("<I", len(body)) + body
        native = struct.pack("<fff", 0.125, 0.125, 0.125)
        pcm = struct.pack("<hh", 4096, 4096)
        raw_path = output / (row["cell_id"] + ".raw-float.wav")
        pcm_path = output / (row["cell_id"] + ".wav")
        raw_path.write_bytes(wav(native, 3, 24000, 32))
        pcm_path.write_bytes(wav(pcm, 1, 16000, 16))
        (output / (row["cell_id"] + ".claim")).write_text(row["cell_id"] + "\n")
        row.update(raw_wav_sha256=c.file_sha256(raw_path), wav_sha256=c.file_sha256(pcm_path),
                   native_float_values_sha256=hashlib.sha256(native).hexdigest(),
                   pcm_sha256=hashlib.sha256(pcm).hexdigest(), frames=2,
                   native_termination={"ended_with_eos": True, "eos_token_id": 9, "last_token_ids": [9]},
                   native_metrics=c.signal_metrics([0.125] * 3, 24000, eos=True),
                   derived_metrics=c.signal_metrics([0.125] * 2, 16000),
                   resample_policy="scipy-resample-poly-f64-2-over-3-no-gain-no-trim-v1")
        return row

    def test_complete_float_and_pcm_pair_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.make_audio_pair(output)
            c.verify_completed_outputs(output, {"outcomes": [row]})
            ledger = c.terminal_ledger(output, {"outcomes": [row]})
            self.assertEqual(ledger[0]["status"], "GENERATED_PENDING_REVIEW")
            self.assertEqual(sum(r["attempts"] for r in ledger), 1)

    def test_raw_values_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.make_audio_pair(output)
            row["native_float_values_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                c.verify_completed_outputs(output, {"outcomes": [row]})

    def test_rehashed_invalid_raw_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.make_audio_pair(output)
            raw = output / (row["cell_id"] + ".raw-float.wav")
            value = bytearray(raw.read_bytes())
            struct.pack_into("<I", value, 24, 22050)
            raw.write_bytes(value)
            row["raw_wav_sha256"] = c.file_sha256(raw)
            with self.assertRaises(ValueError):
                c.verify_completed_outputs(output, {"outcomes": [row]})

    def test_rehashed_trailing_pcm_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = self.make_audio_pair(output)
            path = output / (row["cell_id"] + ".wav")
            path.write_bytes(path.read_bytes() + b"x")
            row["wav_sha256"] = c.file_sha256(path)
            with self.assertRaises(ValueError):
                c.verify_completed_outputs(output, {"outcomes": [row]})

    def test_native_metrics_and_eos_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = self.make_audio_pair(output)
            for field, value in (("native_metrics", {}),
                                 ("native_termination", {"ended_with_eos": True}),
                                 ("resample_policy", "gain-repaired"),
                                 ("derived_metrics", {"technically_valid": "true"}),
                                 ("frames", True)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    c.verify_completed_outputs(output, {"outcomes": [dict(original, **{field: value})]})

    def test_successful_subset_artifact_retains_denominator(self):
        import run as launcher
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            output = work / "output"; output.mkdir()
            row = self.make_audio_pair(output)
            c.save_json(output / "generation-receipt.json", {"outcomes": [row], "status": "stopped_no_retry"})
            host = {"status": "stopped_no_retry", "source_commit": self.sha,
                    "source_freeze_sha256": "b" * 64, "run_id": "1234"}
            destination = launcher.pack_artifact(work, output, host)
            self.assertEqual(host["generated_pending_review"], 1)
            self.assertEqual(host["not_run_cells"], 15)
            self.assertIs(c.read_json(destination / "artifact-freeze.json")["training_admitted"], False)

    def test_invalid_output_still_retains_failure_only_artifact(self):
        import run as launcher
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            output = work / "output"; output.mkdir()
            row = self.make_audio_pair(output)
            row["native_float_values_sha256"] = "0" * 64
            c.save_json(output / "generation-receipt.json", {"outcomes": [row]})
            host = {"status": "complete_pending_review", "source_commit": self.sha,
                    "source_freeze_sha256": "b" * 64, "run_id": "1234"}
            with self.assertRaises(ValueError):
                launcher.pack_artifact(work, output, host)
            self.assertEqual(host["status"], "failed_evidence_validation")
            self.assertFalse(host["artifact_complete"])
            self.assertTrue((work / "artifact/INCOMPLETE.json").exists())
            self.assertFalse(list((work / "artifact").rglob("*.wav")))

    def test_unreviewed_source_tip_rejected_before_any_donor_use(self):
        import run as launcher
        with mock.patch.object(launcher, "git", side_effect=[self.sha, "b" * 40 + "\trefs/heads/main"]):
            with self.assertRaises(ValueError):
                launcher.verify_source(Path("/invented-donor"), self.sha)



    def registration_fixture(self):
        import register_retention as r
        original = {"schema": "kws-public-branch-retention-v1", "retained_files": [
            {"path": "old.py", "mode": "100644", "git_blob_sha1": "1" * 40,
             "bytes": 7, "source_commit": "2" * 40}],
             "integration_files": [r.MANIFEST], "expected_retained_files": 1,
             "expected_tracked_files": 2, "unchanged_provenance": {"sentinel": 1}}
        tree = {"old.py": ("100644", "1" * 40, 7), r.MANIFEST: ("100644", "3" * 40, 99)}
        tree.update({name: ("100644", "4" * 40, 5) for name in r.OWNED})
        return r, original, tree

    def test_registration_preserves_all_original_records(self):
        r, original, tree = self.registration_fixture()
        updated, changed = r.register(original, tree, self.sha, "5" * 64)
        self.assertTrue(changed)
        self.assertEqual(updated["retained_files"][:1], original["retained_files"])
        self.assertEqual(updated["unchanged_provenance"], original["unchanged_provenance"])
        self.assertEqual(updated["expected_retained_files"], 11)
        self.assertEqual(updated["expected_tracked_files"], 12)
        self.assertTrue(all(row["source_commit"] == self.sha for row in updated["retained_files"][1:]))
        second, changed = r.register(updated, tree, "6" * 40, "7" * 64)
        self.assertFalse(changed)
        self.assertEqual(updated, second)

    def test_registration_rejects_unrelated_or_changed_files(self):
        r, original, tree = self.registration_fixture()
        for mutation in (dict(tree, **{"other.py": ("100644", "6" * 40, 1)}),
                         dict(tree, **{"old.py": ("100644", "7" * 40, 7)}),
                         {k: v for k, v in tree.items() if k != next(iter(r.OWNED))}):
            with self.assertRaises(ValueError):
                r.register(original, mutation, self.sha, "5" * 64)

    def test_malformed_receipt_retains_failure_only_artifact(self):
        import run as launcher
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); output = work / "output"; output.mkdir()
            (output / "generation-receipt.json").write_text("{", encoding="utf-8")
            host = {"status": "stopped_no_retry", "source_commit": self.sha, "run_id": "1234"}
            with self.assertRaises(ValueError):
                launcher.pack_artifact(work, output, host)
            self.assertTrue((work / "artifact/INCOMPLETE.json").is_file())


if __name__ == "__main__":
    unittest.main()
