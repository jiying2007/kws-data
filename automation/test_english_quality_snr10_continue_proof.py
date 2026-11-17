"""Independent continuation admission fixtures using the authenticated failed ZIP.

No network, model, tensor runtime, real audio decoding or training is invoked.
The embedded public evidence is the exact consumed preflight-failure artifact;
negative cases alter only in-memory copies, never either historical source.
"""
from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("continuation_proof_review_subject", HERE / "english_quality_snr10_continue.py")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)
FAILURE_SHA = "43ad533ed64eb25421fd24478c69e1cb26f5412f864229f902c0641b498b47e3"
FAILURE_ZIP = base64.b64decode("""
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
EgQAAKIUAAAAAA==
""")


def original_members():
    with zipfile.ZipFile(io.BytesIO(FAILURE_ZIP)) as archive:
        return {item.filename: archive.read(item.filename) for item in archive.infolist()}


def change_record(members, name, changes=None, delete=()):
    key = "evidence/" + name
    value = json.loads(members[key])
    value.update(changes or {})
    for field in delete:
        value.pop(field, None)
    members[key] = json.dumps(value, sort_keys=True).encode()


class CompleteFailureProof(unittest.TestCase):
    def test_exact_consumed_failure_proves_zero_numerical_execution(self):
        self.assertEqual(len(FAILURE_ZIP), 6346)
        self.assertEqual(hashlib.sha256(FAILURE_ZIP).hexdigest(), FAILURE_SHA)
        proof = subject.verify_failed_proof(FAILURE_ZIP)
        self.assertEqual(proof["closed_member_count"], 14)
        self.assertEqual(proof["failure_stage"], "CORPUS_TLS_CONNECTION")
        self.assertEqual(proof["first_failure"]["run_id"], 44019143948)
        self.assertEqual(proof["first_failure"]["artifact_id"], 14300895359)
        self.assertEqual(proof["numeric_worker_invocations_in_first_run"], 0)
        self.assertEqual(proof["first_run_new_embedding_calls"], 0)
        self.assertEqual(proof["first_run_optimizer_updates"], 0)
        self.assertEqual((proof["source_acquisition_attempt"], proof["numeric_attempt"]), (2, 1))
        self.assertIs(proof["first_failure_consumed"], True)

    def test_complete_archive_identity_rejects_substitution_or_truncation(self):
        mutated = bytearray(FAILURE_ZIP)
        mutated[len(mutated) // 2] ^= 1
        for changed in (bytes(mutated), FAILURE_ZIP[:-1], FAILURE_ZIP + b"x", bytearray(FAILURE_ZIP)):
            with self.subTest(kind=type(changed).__name__, length=len(changed)), self.assertRaises(ValueError):
                subject.verify_failed_proof(changed)

    def test_semantically_equal_repacked_archive_is_not_the_consumed_artifact(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, value in original_members().items():
                archive.writestr(name, value)
        with self.assertRaises(ValueError):
            subject.verify_failed_proof(out.getvalue())

    def test_closed_roster_rejects_every_missing_member(self):
        for missing in original_members():
            changed = original_members()
            del changed[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                subject.verify_failed_layout(changed)

    def test_closed_roster_rejects_any_worker_or_extra_evidence(self):
        names = ("evidence/WORKER_STARTED.json", "evidence/PROGRESS.json",
                 "evidence/model-worker.stdout", "evidence/model-worker.stderr",
                 "evidence/model-worker.exit.json", "cache/mixed-embeddings.npz",
                 "evidence/unexplained.json")
        for name in names:
            changed = original_members()
            changed[name] = b"{}"
            with self.subTest(extra=name), self.assertRaises(ValueError):
                subject.verify_failed_layout(changed)

    def test_original_plan_requires_exact_bytes_not_just_matching_parameters(self):
        changed = original_members()
        change_record(changed, "PLAN.json")
        with self.assertRaises(ValueError):
            subject.verify_failed_layout(changed)
        changed = original_members()
        change_record(changed, "PLAN.json", {"worker_sha256": "0" * 64})
        with self.assertRaises(ValueError):
            subject.verify_failed_layout(changed)

    def test_missing_or_nonnull_progress_cannot_claim_zero_calls(self):
        for progress in ({}, {"new_embedding_calls_started": 0}, {"new_embedding_calls_started": 1}, False):
            changed = original_members()
            change_record(changed, "FAILED_NO_RETRY.json", {"partial_progress": progress})
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                subject.verify_failed_layout(changed)
        changed = original_members()
        change_record(changed, "FAILED_NO_RETRY.json", delete=("partial_progress",))
        with self.assertRaises(ValueError):
            subject.verify_failed_layout(changed)

    def test_failed_receipt_cannot_be_reclassified_as_completion_or_retry(self):
        for changes in ({"status": "COMPLETED"}, {"retry_count": 1}, {"error_type": "TimeoutExpired"},
                        {"error": "model command failed"}):
            changed = original_members()
            change_record(changed, "FAILED_NO_RETRY.json", changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                subject.verify_failed_layout(changed)

    def test_first_run_binding_cannot_move_to_another_run_head_or_attempt(self):
        record = json.loads(original_members()["evidence/STARTED.json"])
        altered_training = dict(record["training"], artifact_id=0)
        for changes in ({"head_sha": "0" * 40}, {"run_id": "44019143949"},
                        {"run_attempt": 2}, {"training": altered_training}):
            changed = original_members()
            change_record(changed, "STARTED.json", changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                subject.verify_failed_layout(changed)

    def test_failure_never_grants_formal_authority(self):
        fields = ("formal_training_admitted", "training_admitted", "human_gold", "trusted_negative",
                  "pronunciation_verified", "model_promotion_allowed", "qualification_allowed")
        for field in fields:
            for value in (True, 0, None):
                changed = original_members()
                change_record(changed, "FAILED_NO_RETRY.json", {field: value})
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    subject.verify_failed_layout(changed)

    def test_curl_timeout_is_the_exact_before_worker_failure(self):
        original = json.loads(original_members()["evidence/corpus-download.exit.json"])
        argv = list(original["argv"])
        argv[-1] = "https://example.invalid/other.tar.gz"
        for changes in ({"returncode": 0}, {"returncode": 22}, {"error": None},
                        {"group_cleanup_checked_after_leader_exit": False}, {"argv": argv}):
            changed = original_members()
            change_record(changed, "corpus-download.exit.json", changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                subject.verify_failed_layout(changed)
        changed = original_members()
        changed["evidence/corpus-download.stderr"] = b"curl: (28) transfer timed out\n"
        with self.assertRaises(ValueError):
            subject.verify_failed_layout(changed)

    def test_hidden_command_output_or_failed_source_metadata_is_rejected(self):
        for path in ("evidence/corpus-download.stdout", "evidence/training-zip.stderr",
                     "evidence/source-artifact.stderr", "evidence/source-run.stderr"):
            changed = original_members()
            changed[path] = b"unexpected output\n"
            with self.subTest(path=path), self.assertRaises(ValueError):
                subject.verify_failed_layout(changed)
        for name in ("source-artifact", "source-run"):
            changed = original_members()
            change_record(changed, name + ".exit.json", {"returncode": 1})
            with self.subTest(name=name), self.assertRaises(ValueError):
                subject.verify_failed_layout(changed)


class ContinuationSourceBoundary(unittest.TestCase):
    def test_original_sources_and_all_numerical_context_remain_unchanged(self):
        checked = subject.validate_original_sources()
        self.assertEqual(checked["original_worker_sha256"], "76f2ecd9a1269a587d4ff35971e338533e0d70d588780cc47e84910f8bb70cd9")
        old = (HERE / "english_quality_snr10_once.py").read_text()
        new = (HERE / "english_quality_snr10_continue.py").read_text()
        def objects(text):
            tree = ast.parse(text)
            functions = {node.name: ast.get_source_segment(text, node) for node in tree.body if isinstance(node, ast.FunctionDef)}
            constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                         if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
            return functions, constants
        old_functions, old_constants = objects(old)
        new_functions, new_constants = objects(new)
        # Includes supervision, which is intentionally outside the ten scientific functions.
        names = ("pcm_values", "pcm_bytes", "mix_pcm16", "select_noise_track", "summarize_case",
                 "candidate_decision", "validate_zip", "archive_read", "aggregate", "worker", "supervised_command")
        for name in names:
            self.assertEqual(old_functions[name], new_functions[name], name)
        for name in ("REPO", "TRAINING", "PINS", "BUNDLE", "FALSE_FLAGS", "RUNTIME"):
            self.assertEqual(old_constants[name], new_constants[name], name)
        self.assertEqual(old_constants["CORPUS"]["bytes"], new_constants["CORPUS"]["bytes"])
        self.assertEqual(old_constants["CORPUS"]["sha256"], new_constants["CORPUS"]["sha256"])
        self.assertEqual(new_constants["CORPUS"]["url"], "https://openslr.elda.org/resources/12/dev-clean.tar.gz")

    def test_new_plan_cannot_turn_continuation_into_another_numerical_attempt(self):
        plan = json.loads((HERE / "english_quality_snr10_continue_plan.json").read_text())
        subject.validate_plan(plan)
        subject.assert_same_research_parameters(plan)
        for field, value in (("numeric_attempt", 2), ("numeric_attempt", True),
                             ("source_acquisition_attempt", 3), ("first_failure_consumed", False),
                             ("original_plan_sha256", "0" * 64), ("original_worker_sha256", "0" * 64)):
            altered = copy.deepcopy(plan)
            altered[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                subject.validate_plan(altered)

    def invoke_main_with_stub(self, mode, directory, gate):
        root = Path(directory)
        (root / "downloads").mkdir()
        (root / "evidence").mkdir()
        (root / "downloads/failed-first-run.zip").write_bytes(FAILURE_ZIP)
        (root / "evidence/SOURCE_RECOVERY_GATE.json").write_text(json.dumps(gate))
        with mock.patch.object(subject, "worker") as worker, mock.patch.object(subject, "prepare_noise") as prepare, \
             mock.patch.object(subject.signal, "signal"):
            try:
                subject.main(["--plan", str(HERE / "english_quality_snr10_continue_plan.json"),
                              mode, "--output", str(root)])
            except BaseException:
                worker.assert_not_called()
                prepare.assert_not_called()
                raise
            (worker if mode == "--worker" else prepare).assert_called_once()
            (prepare if mode == "--worker" else worker).assert_not_called()

    def test_both_worker_entries_require_matching_zero_call_proof_before_dispatch(self):
        valid = {**subject.verify_failed_proof(FAILURE_ZIP), "quality_parameters_unchanged": True}
        for mode in ("--worker", "--prepare-noise"):
            for field, value in (("numeric_worker_invocations_in_first_run", 1),
                                 ("first_run_optimizer_updates", 1),
                                 ("numeric_attempt", 2), ("quality_parameters_unchanged", False)):
                with self.subTest(mode=mode, field=field), tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                    self.invoke_main_with_stub(mode, directory, {**valid, field: value})

    def test_valid_proof_reaches_only_selected_stub_no_model(self):
        valid = {**subject.verify_failed_proof(FAILURE_ZIP), "quality_parameters_unchanged": True}
        for mode in ("--worker", "--prepare-noise"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.invoke_main_with_stub(mode, directory, valid)


if __name__ == "__main__":
    unittest.main()
