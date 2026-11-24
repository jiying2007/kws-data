"""Future-only raw capture/ledger fixtures: no audio, model, network, or real process."""
from __future__ import annotations
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b18_asr_adapter as adapter
import english_v6_b18_source_observe as observer

REPOSITORY = HERE.parent


class RawCapture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.row = {"id": "fixture", "wav_sha256": "a" * 64,
                    "pcm_sha256": "b" * 64, "samples": 16000}
        self.directory = self.root / "rows" / "fixture"
        self.directory.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, provider):
        value = observer.binding(provider, self.row, "c" * 64)
        adapter.record(self.directory / "STARTED.json", value)
        return value

    def capture_python(self, text, provider="whisper"):
        binding = self.start(provider)
        native = {"text": text} if provider == "whisper" else [SimpleNamespace(text=text)]
        result = adapter.capture_python(provider, self.directory, binding, lambda: native)
        capture = {**binding, **result, "status": "success", "input_coverage_complete": True}
        adapter.record(self.directory / "CAPTURE.json", capture)
        return binding, capture

    def process(self, stdout, stderr=b"native diagnostic\n", exit_code=0, timeout=False):
        binding = self.start("parakeet")
        command = ["/private/assets/nemo-speech", "transcribe", "/no-audio/fixture.wav", "--model",
                   "/private/assets/parakeet-tdt-0.6b-v3.q8_0.gguf", "--device", "cpu"]
        environment = {"PATH": "/usr/bin:/bin", "HOME": str(self.directory), "LANG": "C.UTF-8",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "NEMO_SPEECH_MODEL_DIR": "/private/assets", "OMP_NUM_THREADS": "4"}
        calls = []
        class FakeProcess:
            def __init__(self):
                self.killed = False
            def wait(self, timeout=None):
                if timeout is not None and timeout_flag:
                    raise subprocess.TimeoutExpired(command, timeout)
                return -9 if self.killed else exit_code
            def kill(self):
                self.killed = True
            def poll(self):
                return None if not self.killed else -9
        timeout_flag = timeout
        def fake_popen(argv, **kwargs):
            calls.append((argv, kwargs))
            self.assertNotIn("text", kwargs)
            self.assertNotEqual(kwargs["stderr"], subprocess.DEVNULL)
            kwargs["stdout"].write(stdout)
            kwargs["stderr"].write(stderr)
            return FakeProcess()
        result = adapter.capture_process(command, environment, self.directory, binding, popen=fake_popen)
        self.assertEqual(len(calls), 1)
        capture = {**binding, **result, "status": "success", "input_coverage_complete": True}
        adapter.record(self.directory / "CAPTURE.json", capture)
        return binding, capture

    def test_exact_white_space_set_and_no_lexical_changes(self):
        self.assertEqual(len(adapter.WHITE_SPACE), 25)
        self.assertNotIn(0x001C, adapter.WHITE_SPACE)
        self.assertEqual(adapter.normalize_raw(" \tAlpha,\u00a0\u2028BETa!\r\n".encode()), "Alpha, BETa!")
        self.assertEqual(adapter.normalize_raw(" \u00e9 \uff21 ".encode()), "é Ａ")

    def test_no_log_removal_or_line_extraction(self):
        self.assertEqual(adapter.normalize_raw(b"INFO: ready\nActual text.\n"), "INFO: ready Actual text.")

    def test_u001c_to_u001f_and_other_controls_rejected(self):
        for codepoint in (*range(0x001C, 0x0020), 0, 1, 0x7F, 0x80, 0x200B, 0x202E, 0xFEFF):
            with self.subTest(codepoint=codepoint):
                with self.assertRaisesRegex(ValueError, "CONTROL"):
                    adapter.normalize_raw(("A" + chr(codepoint) + "B").encode())

    def test_utf8_strict_and_raw_length_before_normalization(self):
        with self.assertRaises(UnicodeDecodeError):
            adapter.normalize_raw(b"a\xffb")
        with self.assertRaisesRegex(ValueError, "OVERSIZE_BEFORE_NORMALIZATION"):
            adapter.normalize_raw(b" " * 4096 + b"A")
        self.assertEqual(adapter.normalize_raw(b" " * 4095 + b"A"), "A")

    def test_nonempty_normalized_required(self):
        for raw in (b"", b" \t\r\n", "\u3000\u0085".encode()):
            with self.assertRaisesRegex(ValueError, "EMPTY_NATIVE"):
                adapter.normalize_raw(raw)

    def test_valid_python_raw_is_lossless_and_legacy_capture_compatible(self):
        text = " \tAlpha,\u00a0BETa!\n"
        binding, capture = self.capture_python(text, "qwen")
        raw = json.loads((self.directory / "RAW_RETURN.json").read_bytes())
        self.assertEqual(raw["returned_text_fields"][0]["value"], text)
        self.assertEqual((self.directory / "RAW_TEXT.utf8").read_bytes(), text.encode())
        self.assertEqual(adapter.validate_capture("qwen", self.directory, binding, capture), "Alpha, BETa!")
        self.assertTrue(observer.complete_ledger(observer.row_ledger(self.root, [self.row]), 1))

    def test_empty_return_is_returned_but_never_valid_or_complete(self):
        binding = self.start("whisper")
        with self.assertRaisesRegex(ValueError, "EMPTY_NATIVE"):
            adapter.capture_python("whisper", self.directory, binding, lambda: {"text": "\t\n"})
        self.assertEqual((self.directory / "RAW_TEXT.utf8").read_bytes(), b"\t\n")
        counts = observer.ledger_counts(observer.row_ledger(self.root, [self.row]))
        self.assertEqual((counts["forward_attempted"], counts["native_returned"], counts["raw_payload_saved"], counts["payload_valid"]), (1, 1, 1, 0))
        self.assertFalse(observer.complete_ledger(observer.row_ledger(self.root, [self.row]), 1))

    def test_oversize_text_saved_fully_before_failure(self):
        binding = self.start("whisper")
        raw = " " * 5000 + "A"
        with self.assertRaisesRegex(ValueError, "OVERSIZE"):
            adapter.capture_python("whisper", self.directory, binding, lambda: {"text": raw})
        self.assertEqual((self.directory / "RAW_TEXT.utf8").read_bytes(), raw.encode())
        self.assertFalse((self.directory / "PAYLOAD_VALID.json").exists())

    def test_invalid_surrogate_still_retained_losslessly(self):
        binding = self.start("whisper")
        raw = "A\ud800B"
        with self.assertRaises(UnicodeEncodeError):
            adapter.capture_python("whisper", self.directory, binding, lambda: {"text": raw})
        saved = json.loads((self.directory / "RAW_RETURN.json").read_bytes())
        self.assertEqual(saved["returned_text_field"]["value"], raw)
        self.assertTrue((self.directory / "FORWARD_RETURNED.json").exists())
        self.assertFalse((self.directory / "PAYLOAD_VALID.json").exists())

    def test_native_shape_failure_keeps_all_returned_text_fields(self):
        binding = self.start("qwen")
        with self.assertRaisesRegex(ValueError, "NATIVE_TEXT_RESULT_SHAPE"):
            adapter.capture_python("qwen", self.directory, binding,
                                   lambda: [SimpleNamespace(text="first"), SimpleNamespace(text="second")])
        raw = json.loads((self.directory / "RAW_RETURN.json").read_bytes())
        self.assertEqual([r["value"] for r in raw["returned_text_fields"]], ["first", "second"])
        self.assertFalse(observer.complete_ledger(observer.row_ledger(self.root, [self.row]), 1))

    def test_raising_native_call_is_attempted_not_returned(self):
        binding = self.start("whisper")
        def invoke():
            raise RuntimeError("fixture native call did not return")
        with self.assertRaises(RuntimeError):
            adapter.capture_python("whisper", self.directory, binding, invoke)
        counts = observer.ledger_counts(observer.row_ledger(self.root, [self.row]))
        self.assertEqual((counts["forward_attempted"], counts["native_returned"], counts["payload_valid"]), (1, 0, 0))

    def test_lost_payload_cannot_be_recovered_from_capture_or_counted(self):
        binding, capture = self.capture_python("Original output.")
        (self.directory / "RAW_TEXT.utf8").unlink()
        with self.assertRaisesRegex(ValueError, "REGULAR_ORIGINAL_EVIDENCE_FILE"):
            adapter.validate_capture("whisper", self.directory, binding, capture)

    def test_parakeet_preserves_all_stdout_stderr_bytes_and_command(self):
        stdout, stderr = b"\tAlpha,\nBETa!\r\n", b"diagnostic\x00bytes\n"
        binding, capture = self.process(stdout, stderr)
        self.assertEqual((self.directory / "RAW_STDOUT.bin").read_bytes(), stdout)
        self.assertEqual((self.directory / "RAW_STDERR.bin").read_bytes(), stderr)
        self.assertEqual(adapter.validate_capture("parakeet", self.directory, binding, capture), "Alpha, BETa!")

    def test_parakeet_invalid_utf8_preserves_payload_then_fails(self):
        with self.assertRaises(UnicodeDecodeError):
            self.process(b"raw\xffoutput")
        self.assertEqual((self.directory / "RAW_STDOUT.bin").read_bytes(), b"raw\xffoutput")
        self.assertTrue((self.directory / "RAW_PROCESS.json").is_file())
        self.assertTrue((self.directory / "FORWARD_RETURNED.json").is_file())
        self.assertFalse((self.directory / "PAYLOAD_VALID.json").exists())

    def test_parakeet_nonzero_exit_is_returned_not_valid(self):
        with self.assertRaisesRegex(ValueError, "PARAKEET_NATIVE_PROCESS_FAILED"):
            self.process(b"unadmitted text\n", b"failure detail\n", exit_code=7)
        raw = json.loads((self.directory / "RAW_PROCESS.json").read_bytes())
        self.assertEqual(raw["native_exit_code"], 7)
        self.assertTrue(raw["native_return_observed"])
        self.assertEqual((self.directory / "RAW_STDERR.bin").read_bytes(), b"failure detail\n")
        self.assertFalse((self.directory / "PAYLOAD_VALID.json").exists())

    def test_parakeet_timeout_keeps_partial_raw_and_observed_kill_exit(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.process(b"partial\n", b"timeout diagnostic\n", timeout=True)
        raw = json.loads((self.directory / "RAW_PROCESS.json").read_bytes())
        self.assertEqual(raw["native_exit_code"], -9)
        self.assertTrue(raw["timed_out"])
        self.assertFalse(raw["native_return_observed"])
        self.assertFalse((self.directory / "FORWARD_RETURNED.json").exists())
        self.assertEqual((self.directory / "RAW_STDOUT.bin").read_bytes(), b"partial\n")


class ProspectiveContract(unittest.TestCase):
    def test_exact_existing_policy_and_asr_decode_keyword_parity(self):
        contract = observer.source_check(REPOSITORY)
        self.assertEqual(contract["source_admission_function_sha256"], observer.POLICY_FUNCTIONS)
        self.assertEqual(contract["maximum_native_calls"], 480)
        self.assertEqual(contract["asr_runtime_versions"]["qwen"]["torch"], "2.9.1+cpu")
        def calls(path):
            tree = ast.parse(path.read_text())
            result = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "transcribe":
                    result.append(tuple((kw.arg, ast.dump(kw.value, include_attributes=False)) for kw in node.keywords))
            return sorted(result)
        self.assertEqual(calls(REPOSITORY / "automation/english_seven_voice_asr_once.py"),
                         calls(HERE / "english_v6_b18_asr_adapter.py"))

    def test_old_panel_or_train_rows_refused(self):
        rows = [{"id": f"fixture-{i}", "speaker_index": 766 + i // 4,
                 "role": "development", "training_allowed": False} for i in range(96)]
        registration = {"source_manifest_sha256": "d" * 64}
        observer.validate_rows(rows, "d" * 64, registration)
        rows[0]["speaker_index"] = 718
        with self.assertRaisesRegex(ValueError, "766_TO_789"):
            observer.validate_rows(rows, "d" * 64, registration)
        rows[0]["speaker_index"] = 766
        rows[0]["role"] = "train"
        with self.assertRaises(ValueError):
            observer.validate_rows(rows, "d" * 64, registration)

    def test_distinct_96_valid_payloads_required_not_96_attempts(self):
        rows = [{"id": f"r{i}", **dict.fromkeys(("input_started", "forward_attempted", "native_returned",
                    "raw_payload_saved", "payload_valid", "capture_saved"), True), "failed": False} for i in range(96)]
        self.assertTrue(observer.complete_ledger(rows))
        rows[17].update(payload_valid=False, capture_saved=False, failed=True)
        self.assertFalse(observer.complete_ledger(rows))
        self.assertEqual(observer.ledger_counts(rows)["native_returned"], 96)
        self.assertEqual(observer.ledger_counts(rows)["payload_valid"], 95)
        rows[17]["capture_saved"] = True
        with self.assertRaisesRegex(ValueError, "COUNTER_ORDER"):
            observer.ledger_counts(rows)

    def test_no_placeholder_or_unregistered_execution(self):
        with self.assertRaisesRegex(ValueError, "PREREGISTERED"):
            observer.guard(None, env={})
        with self.assertRaisesRegex(ValueError, "PREDECESSOR"):
            observer.guard({"prepared_parent": "UNSET", "run_branch": "refs/heads/research/en-v6-b18-source-once"}, env={})

    def test_provider_agreement_cannot_replace_frozen_source_registration(self):
        expected = {"assessment_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
                    "run_id": "123", "run_head": "c" * 40}
        calls = []
        def authenticate(source):
            calls.append(source)
            return expected.copy()
        with patch.dict(sys.modules, {"english_v6_b18_source": SimpleNamespace(registration=authenticate)}):
            self.assertEqual(observer.expected_registration(Path("fixture-source")), expected)
            self.assertEqual(calls, [Path("fixture-source")])
            with self.assertRaisesRegex(ValueError, "AUTHENTICATED_SOURCE_AND_ASSESSMENT"):
                observer.expected_registration(Path("fixture-source"), {**expected, "assessment_sha256": "d" * 64})

    def test_exact_public_first_push_guard(self):
        contract = observer.source_check(REPOSITORY)
        branch = "refs/heads/research/en-v6-b18-synthetic-primary-once-20261124"
        registration = {"prepared_parent": "b" * 40, "run_branch": branch, "run_id": "123", "run_head": "a" * 40,
            "run_attempt": 1, "source_schema": observer.SOURCE_SCHEMA, "batch_schema": observer.BATCH,
            "source_rows": 96, "max_calls": 96, "assessment_sha256": "d" * 64,
            "source_manifest_sha256": "e" * 64,
            **{k: contract[k] for k in ("observer_sha256", "asr_adapter_sha256", "parser_contract_sha256")}}
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "jiying2007/kws-data", "GITHUB_EVENT_NAME": "push",
               "GITHUB_RUN_ATTEMPT": "1", "GITHUB_REF": branch, "GITHUB_RUN_ID": "123", "GITHUB_SHA": "a" * 40}
        event = {"before": "b" * 40, "after": "a" * 40, "ref": branch, "forced": False, "deleted": False,
                 "repository": {"full_name": "jiying2007/kws-data", "private": False}}
        with patch.object(observer, "source_check", return_value=contract):
            self.assertEqual(observer.guard(registration, env=env, event=event, parent="b" * 40), {"run": "123", "sha": "a" * 40})
            for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", "refs/heads/main")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    observer.guard(registration, env={**env, key: value}, event=event, parent="b" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    options, remaining = parser.parse_known_args()
    REPOSITORY = options.repository.resolve()
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
