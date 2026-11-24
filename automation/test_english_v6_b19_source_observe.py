"""Offline source-diagnostic fixtures; no audio generation or native ASR process."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b19_source_observe as subject


def rows_fixture():
    rows = []
    for speaker in range(790, 798):
        for cell, (arm, scale, cls) in subject.CELLS.items():
            ident = f"b19-fixture-{speaker}-{cell}"
            rows.append(dict(id=ident, pcm_sha256=subject.sha(ident.encode()),
                wav_sha256=subject.sha(("wav:" + ident).encode()), samples=16000,
                speaker_index=speaker, source_speaker_id="p" + str(speaker), cell_id=cell,
                feasibility_arm=arm, length_scale=scale, requested_class=cls,
                role="source-feasibility", observed_class=None, training_allowed=False,
                evaluation_allowed=False, admission_eligible=False, basic_audio_ok=True))
    return rows


def registration(source_sha="c" * 64):
    contract = subject.code_contract()
    return dict(run_id="123", run_head="a" * 40, run_attempt=1,
                run_branch=subject.BRANCH, prepared_parent="b" * 40,
                source_schema=subject.SOURCE_SCHEMA, batch_schema=subject.SCHEMA,
                source_rows=48, max_calls=48, diagnostic_only=True,
                source_admission_eligible=False, assessment_sha256="d" * 64,
                source_manifest_sha256=source_sha,
                **{k: contract[k] for k in ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256")})


def native_fixture(outcome, calls):
    def backend(wav, directory, start):
        chosen = outcome(start["id"]) if callable(outcome) else outcome
        stdout, code = chosen
        command = ["/fixture/assets/nemo-speech", "transcribe", str(wav), "--model",
                   "/fixture/assets/parakeet-tdt-0.6b-v3.q8_0.gguf", "--device", "cpu"]
        env = {"PATH": "/usr/bin:/bin", "HOME": str(directory), "LANG": "C.UTF-8",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "NEMO_SPEECH_MODEL_DIR": "/fixture/assets", "OMP_NUM_THREADS": "4"}
        class Process:
            def __init__(self, argv, **kwargs):
                calls.append(start["id"])
                kwargs["stdout"].write(stdout)
                kwargs["stderr"].write(b"fixture diagnostic only\n")
            def wait(self, timeout=None):
                return code
            def poll(self):
                return code
            def kill(self):
                raise AssertionError("fixture should not be killed")
        return subject.adapter.capture_process(command, env, directory, start, popen=Process)
    return backend


class DiagnosticSemantics(unittest.TestCase):
    def test_same_raw_backend_and_fixed_cells(self):
        contract = subject.code_contract()
        self.assertEqual(contract["asr_adapter_sha256"], subject.RAW_ADAPTER_SHA)
        self.assertEqual(contract["maximum_native_calls"], 48)
        subject.validate_rows(rows_fixture())
        changed = rows_fixture()
        changed[2]["length_scale"] = 1.5
        with self.assertRaises(ValueError):
            subject.validate_rows(changed)
        for field, value in (("role", "development"), ("evaluation_allowed", True),
                             ("admission_eligible", True), ("observed_class", 2)):
            changed = rows_fixture()
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.validate_rows(changed)

    def test_baseline_empty_is_invalid_and_cannot_open_a_model_gate(self):
        rows = rows_fixture()
        captures = {r["id"]: dict(status="invalid_empty" if r["feasibility_arm"] == "baseline" else "success") for r in rows}
        verdict = subject.feasibility(rows, captures)
        self.assertTrue(verdict["prospective_pair_feasible"])
        self.assertEqual(verdict["arms"]["baseline"]["valid_nonempty"], 0)
        self.assertEqual(verdict["arms"]["baseline"]["invalid_empty"], 16)
        for field in ("source_admission_eligible", "source_gate_passed", "KWS_evaluation_allowed",
                      "training_allowed", "promotion_allowed", "replication_opened", "causal_fix_proven"):
            self.assertIs(verdict[field], False)

    def test_no_prospective_or_ordinary_empty_can_pass_feasibility(self):
        rows = rows_fixture()
        for arm in ("prospective", "ordinary"):
            captures = {r["id"]: dict(status="success") for r in rows}
            ident = next(r["id"] for r in rows if r["feasibility_arm"] == arm)
            captures[ident]["status"] = "invalid_empty"
            verdict = subject.feasibility(rows, captures)
            self.assertFalse(verdict["prospective_pair_feasible"])
            self.assertTrue(verdict["diagnostic_complete"])

    def test_missing_original_and_invalid_native_call_are_not_valid_empty(self):
        rows = rows_fixture()
        captures = {r["id"]: dict(status="success") for r in rows}
        captures[rows[0]["id"]]["status"] = "failed"
        self.assertFalse(subject.feasibility(rows, captures)["diagnostic_complete"])
        del captures[rows[-1]["id"]]
        with self.assertRaises(ValueError):
            subject.feasibility(rows, captures)

    def test_original_duration_bounds_cannot_be_relaxed(self):
        for index in (0, 2):
            for bad in (6399, 28673):
                rows = rows_fixture()
                rows[index]["samples"] = bad
                captures = {r["id"]: dict(status="success") for r in rows}
                self.assertFalse(subject.feasibility(rows, captures)["prospective_pair_feasible"])
        rows = rows_fixture()
        rows[0]["basic_audio_ok"] = False
        self.assertFalse(subject.feasibility(rows, {r["id"]: dict(status="success") for r in rows})["prospective_pair_feasible"])

    def test_first_public_execution_only_and_no_source_admission_flag(self):
        reg = registration()
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=subject.REPO, GITHUB_EVENT_NAME="push",
                   GITHUB_REF=subject.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID="123", GITHUB_SHA="a" * 40)
        event = dict(before="b" * 40, after="a" * 40, ref=subject.BRANCH, forced=False, deleted=False,
                     repository=dict(private=False, full_name=subject.REPO))
        subject.guard(reg, env, event, "b" * 40)
        for key, bad in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", "refs/heads/main")):
            with self.assertRaises(ValueError):
                subject.guard(reg, dict(env, **{key: bad}), event, "b" * 40)
        with self.assertRaises(ValueError):
            subject.guard(reg, env, dict(event, forced=True), "b" * 40)
        with self.assertRaises(ValueError):
            subject.validate_registration(dict(reg, source_admission_eligible=True))


class OriginalRawDiagnostic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.row = rows_fixture()[0]
        self.directory = self.root / "row"
        self.directory.mkdir()
        self.wav = self.root / (self.row["id"] + ".wav")

    def tearDown(self):
        self.tmp.cleanup()

    def test_exit_zero_blank_keeps_exact_raw_and_never_gets_payload_valid(self):
        calls = []
        result = subject.capture_original(native_fixture((b"\n", 0), calls), self.wav,
                                          self.directory, self.row, "c" * 64)
        self.assertEqual(calls, [self.row["id"]])
        self.assertEqual(result["status"], "invalid_empty")
        self.assertEqual((self.directory / "RAW_STDOUT.bin").read_bytes(), b"\n")
        self.assertFalse((self.directory / "PAYLOAD_VALID.json").exists())
        self.assertFalse((self.directory / "CAPTURE.json").exists())
        self.assertFalse(result["payload_valid"])
        self.assertEqual(result, json.loads((self.directory / "FAILED.json").read_bytes()))

    def test_nonempty_native_text_uses_original_capture_validator(self):
        calls = []
        result = subject.capture_original(native_fixture((b" Hey Mero.\n", 0), calls), self.wav,
                                          self.directory, self.row, "c" * 64)
        start = subject.raw_observer.binding("parakeet", self.row, "c" * 64)
        self.assertEqual(subject.adapter.validate_capture("parakeet", self.directory, start, result), "Hey Mero.")
        self.assertEqual(calls, [self.row["id"]])

    def test_nonzero_native_exit_cannot_be_counted_as_diagnostic_empty(self):
        calls = []
        with self.assertRaises(ValueError):
            subject.capture_original(native_fixture((b"\n", 3), calls), self.wav, self.directory, self.row, "c" * 64)
        self.assertEqual(calls, [self.row["id"]])
        result = json.loads((self.directory / "FAILED.json").read_bytes())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["payload_valid"])

    def test_lost_raw_return_and_control_bytes_cannot_be_reconstructed_as_empty(self):
        calls = []
        subject.capture_original(native_fixture((b"\n", 0), calls), self.wav,
                                 self.directory, self.row, "c" * 64)
        start = subject.raw_observer.binding("parakeet", self.row, "c" * 64)
        (self.directory / "RAW_STDOUT.bin").write_bytes(b"\x1c")
        with self.assertRaises(ValueError):
            subject.verify_empty(self.directory, start)
        (self.directory / "RAW_STDOUT.bin").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            subject.verify_empty(self.directory, start)


class BoundedRun(unittest.TestCase):
    def simulate(self, outcome, expect_fatal=False, lose_raw=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            source.mkdir(); (source / "MANIFEST.json").write_text("fixture")
            reg = registration(subject.sha(b"fixture"))
            rows = rows_fixture(); by_id = {r["id"]: r for r in rows}
            contract = subject.code_contract()
            model = dict(observer=contract["native_model"], runtime_versions=contract["native_runtime"],
                         adapter_sha256=subject.RAW_ADAPTER_SHA, parser_contract_sha256=subject.PARSER_SHA,
                         raw_before_validation=True, decode_config_unchanged=True)
            calls = []
            native = native_fixture(outcome, calls)
            def backend(wav, directory, start):
                try:
                    return native(wav, directory, start)
                except ValueError:
                    if lose_raw:
                        (directory / "RAW_STDOUT.bin").unlink()
                    raise
            import english_hifi_micro as pcm_tools
            with patch.object(subject, "guard"), \
                 patch.dict(sys.modules, {"english_v6_b19_source": SimpleNamespace(registration=lambda _: reg)}), \
                 patch.object(subject.adapter, "_load", return_value=(backend, model)), \
                 patch.object(pcm_tools, "inspect_wav", side_effect=lambda wav: by_id[wav.stem]):
                if expect_fatal:
                    with self.assertRaises(ValueError):
                        subject.observe(source, output, read_rows=lambda _: rows, registration=reg)
                else:
                    subject.observe(source, output, read_rows=lambda _: rows, registration=reg)
                    subject.readback(source, output, lambda _: rows, reg)
            result = json.loads((output / "RESULT.json").read_bytes())
            return calls, result

    def test_diagnostic_continues_after_proven_empty_without_replaying_it(self):
        calls, result = self.simulate(lambda ident: (b"\n", 0) if "baseline" in ident else (b"Hey Mero.\n", 0))
        self.assertEqual(len(calls), 48)
        self.assertEqual(len(set(calls)), 48)
        self.assertEqual((result["actual_native_attempts"], result["native_returns"], result["valid_nonempty_payloads"],
                          result["invalid_empty_payloads"]), (48, 48, 32, 16))
        self.assertTrue(result["prospective_pair_feasible"])
        self.assertFalse(result["source_gate_passed"])
        self.assertEqual(result["new_KWS_forwards"], 0)

    def test_other_native_failure_stops_every_remaining_call_with_true_counter(self):
        calls, result = self.simulate((b"\n", 2), expect_fatal=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["actual_native_attempts"], 1)
        self.assertEqual(result["native_returns"], 1)
        self.assertEqual(result["valid_nonempty_payloads"], 0)
        self.assertEqual(result["invalid_empty_payloads"], 0)
        self.assertEqual(result["rows"][0]["status"], "failed")
        self.assertFalse(result["prospective_pair_feasible"])

    def test_self_reported_registration_cannot_trigger_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = registration()
            with patch.object(subject, "guard"), \
                 patch.dict(sys.modules, {"english_v6_b19_source": SimpleNamespace(registration=lambda _: dict(reg, assessment_sha256="e" * 64))}), \
                 patch.object(subject.adapter, "_load", side_effect=AssertionError("native model must not load")) as load_model:
                with self.assertRaises(ValueError):
                    subject.observe(root, root / "out", read_rows=lambda _: rows_fixture(), registration=reg)
                load_model.assert_not_called()
                self.assertFalse((root / "out").exists())

    def test_empty_with_missing_raw_proof_is_recorded_failed_and_stops(self):
        calls, result = self.simulate((b"\n", 0), expect_fatal=True, lose_raw=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["actual_native_attempts"], 1)
        self.assertEqual(result["native_returns"], 1)
        self.assertEqual(result["invalid_empty_payloads"], 0)
        self.assertEqual(result["rows"][0]["status"], "failed")
        self.assertIn("REGULAR_ORIGINAL_EVIDENCE_FILE", result["rows"][0]["empty_proof_error"])
        self.assertIn("UNPROVEN_EMPTY_NATIVE_TRANSCRIPT", result["fatal_error"])
        self.assertTrue(all(row["status"] == "not_started" for row in result["rows"][1:]))


if __name__ == "__main__":
    unittest.main()
