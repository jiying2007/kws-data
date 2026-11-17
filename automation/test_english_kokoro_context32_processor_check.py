"""Stdlib/mock tests only. Actual native processor execution belongs to public CI."""
from __future__ import annotations
import ast
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


c = module("processor_check_fixture", HERE / "english_kokoro_context32_processor_check.py")
f = module("processor_mock_values", HERE / "test_english_kokoro_context32_mask_observe.py")


class FakeFactory:
    def __init__(self, *, dtype="torch.int32", create_failure=False, call_failure=False):
        self.dtype = dtype
        self.create_failure = create_failure
        self.call_failure = call_failure
        self.instances = []
        self.requests = []

    def from_pretrained(self, path, *, local_files_only):
        self.requests.append((path, local_files_only))
        if self.create_failure:
            raise ValueError("MOCK_PROCESSOR_CONSTRUCTOR_FAILURE")
        extractor = f.MockExtractor(dtype=self.dtype)
        extractor.sampling_rate = 16000
        extractor.do_normalize = True
        extractor.return_attention_mask = True
        self.instances.append(extractor)
        if self.call_failure:
            def fail(*args, **kwargs):
                raise ValueError("MOCK_PROCESSOR_CALL_FAILURE")
            return type("MockFailedProcessor", (), {"sampling_rate": 16000,
                "do_normalize": True, "return_attention_mask": True, "__call__": fail})()
        return extractor


class ProcessorCheckFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="context32-processor-mock-only-")
        self.root = Path(self.temp.name)
        self.output = self.root / "processor-output"
        self.event_path = self.root / "event.json"
        repo = {"full_name": "jiying2007/kws-data", "private": False}
        self.event = {"repository": repo, "number": 999,
            "pull_request": {"base": {"ref": "main", "repo": repo}, "head": {"repo": repo}}}
        self.env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "jiying2007/kws-data",
            "GITHUB_RUN_ATTEMPT": "1", "GITHUB_JOB": c.JOB, "KWS_CONTEXT32_PROCESSOR_TASK": c.TASK,
            "GITHUB_WORKFLOW_REF": c.WORKFLOW + "@refs/pull/999/merge", "GITHUB_SHA": "1" * 40,
            "GITHUB_RUN_ID": "999", "GITHUB_EVENT_PATH": str(self.event_path),
            "GITHUB_EVENT_NAME": "pull_request", "GITHUB_REF": "refs/pull/999/merge"}
        self.write_event()

    def tearDown(self):
        self.temp.cleanup()

    def write_event(self):
        self.event_path.write_text(json.dumps(self.event))

    def run_mock(self, factory=None):
        factory = FakeFactory() if factory is None else factory
        native = (f.MockNumpy, f.MockTorch, factory, c.observer.RUNTIME_VERSIONS,
            c.observer.PROCESSOR_SOURCE_FILES)
        with patch.object(c, "_load_runtime", return_value=native):
            result = c.run(self.output, env=self.env)
        return result, factory

    def test_import_and_source_check_use_only_stdlib(self):
        for name in ("torch", "transformers", "numpy"):
            self.assertNotIn(name, sys.modules)
        result = c.source_preflight()
        self.assertEqual(result["preprocessor_config"], c.observer.COMMON_FILES["preprocessor_config.json"])
        self.assertEqual(result["processor_calls"], 0)
        self.assertEqual(result["model_forwards"], 0)

    def test_fixed_fixture_is_exact_non_corpus_integer_pcm(self):
        raw = c.fixture_pcm()
        self.assertEqual(len(raw), 1280)
        self.assertEqual(struct.unpack("<640h", raw), tuple(((i * 73 + 19) % 16001) - 8000 for i in range(640)))
        self.assertFalse(c.FIXTURE["campaign_audio"])
        self.assertFalse(c.FIXTURE["model_input_task"])

    def test_same_public_pr_guard(self):
        self.assertEqual(c.public_guard(self.env)["event"], "pull_request")

    def test_public_main_source_guard(self):
        self.env.update(GITHUB_EVENT_NAME="push", GITHUB_REF="refs/heads/main",
            GITHUB_WORKFLOW_REF=c.WORKFLOW + "@refs/heads/main")
        self.event = {"repository": {"full_name": "jiying2007/kws-data", "private": False},
            "ref": "refs/heads/main", "after": self.env["GITHUB_SHA"]}
        self.write_event()
        self.assertEqual(c.public_guard(self.env)["event"], "push")

    def test_private_or_fork_pr_rejected_before_runtime(self):
        for side, private in (("head", False), ("head", True), ("base", True)):
            with self.subTest(side=side, private=private):
                self.event["pull_request"][side]["repo"] = {
                    "full_name": "jiying2007/kws-data" if private else "fixture/fork", "private": private}
                self.write_event()
                with patch.object(c, "_load_runtime", side_effect=AssertionError("must not load")) as load:
                    with self.assertRaisesRegex(ValueError, "PUBLIC"):
                        c.run(self.output, env=self.env)
                    load.assert_not_called()

    def test_wrong_task_job_workflow_rerun_and_dispatch_rejected(self):
        for key, bad in (("GITHUB_ACTIONS", "false"), ("GITHUB_RUN_ATTEMPT", "2"),
            ("GITHUB_JOB", "private"), ("KWS_CONTEXT32_PROCESSOR_TASK", "private"),
            ("GITHUB_WORKFLOW_REF", "different/repo/workflow@refs/heads/main"),
            ("GITHUB_EVENT_NAME", "workflow_dispatch")):
            env = self.env | {key: bad}
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.public_guard(env)

    def test_shared_prepare_saves_native_i32_and_zero_model_evidence(self):
        with patch.object(c.observer, "prepare_native_pcm", wraps=c.observer.prepare_native_pcm) as prepare:
            result, factory = self.run_mock()
            self.assertEqual(prepare.call_count, 1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "PASSED_NATIVE_INT32_PROCESSOR_ONLY")
        self.assertEqual((result["processor_construction_attempts"], result["processor_instances"],
            result["processor_call_attempts"], result["processor_native_returns"]), (1, 1, 1, 1))
        self.assertEqual((result["model_constructions"], result["model_forwards"]), (0, 0))
        self.assertFalse(result["weights_downloaded"])
        self.assertFalse(result["campaign_audio_consumed"])
        self.assertEqual(factory.requests, [(str(self.output), True)])
        self.assertEqual(factory.instances[0].calls, [(16000, "pt", 640)])
        capture = json.loads((self.output / "CAPTURE.json").read_bytes())
        self.assertEqual(capture["processor"]["attention_mask_dtype"], "torch.int32")
        self.assertEqual(capture["attention_mask"]["bytes"], 2560)
        self.assertEqual((self.output / "attention-mask.i32").read_bytes(), struct.pack("<640i", *([1] * 640)))
        self.assertEqual(capture["native_processor_source_files"], c.observer.PROCESSOR_SOURCE_FILES)

    def test_old_int64_native_return_is_retained_and_rejected(self):
        result, factory = self.run_mock(FakeFactory(dtype="torch.int64"))
        self.assertFalse(result["passed"])
        self.assertEqual((result["processor_call_attempts"], result["processor_native_returns"]), (1, 1))
        returned = json.loads((self.output / "PROCESSOR_RETURNED.json").read_bytes())
        self.assertEqual(returned["attention_mask_dtype"], "torch.int64")
        self.assertIn("ACTUAL_UNPADDED_ATTENTION_MASK", result["failure"])
        self.assertFalse((self.output / "CAPTURE.json").exists())

    def test_processor_constructor_failure_has_no_call(self):
        result, _ = self.run_mock(FakeFactory(create_failure=True))
        self.assertFalse(result["passed"])
        self.assertEqual((result["processor_construction_attempts"], result["processor_instances"],
            result["processor_call_attempts"], result["processor_native_returns"]), (1, 0, 0, 0))

    def test_processor_call_failure_is_not_retried(self):
        result, _ = self.run_mock(FakeFactory(call_failure=True))
        self.assertFalse(result["passed"])
        self.assertEqual((result["processor_call_attempts"], result["processor_native_returns"]), (1, 0))
        self.assertEqual(result["retry_count"], 0)

    def test_capture_write_failure_keeps_processor_return(self):
        original = c.observer.put
        def bad_put(path, raw):
            if Path(path).name == "attention-mask.i32":
                raise OSError("MOCK_AFTER_PROCESSOR_RETURN")
            return original(path, raw)
        with patch.object(c.observer, "put", bad_put):
            result, _ = self.run_mock()
        self.assertFalse(result["passed"])
        self.assertEqual((result["processor_call_attempts"], result["processor_native_returns"]), (1, 1))

    def test_initial_runtime_failure_retains_zero_instances(self):
        with patch.object(c, "_load_runtime", side_effect=ValueError("MOCK_BAD_RUNTIME")):
            result = c.run(self.output, env=self.env)
        self.assertFalse(result["passed"])
        self.assertEqual((result["processor_construction_attempts"], result["processor_instances"],
            result["processor_call_attempts"], result["processor_native_returns"]), (0, 0, 0, 0))

    def test_existing_output_rejects_before_runtime(self):
        self.run_mock()
        with patch.object(c, "_load_runtime", side_effect=AssertionError("must not load")) as load:
            with self.assertRaisesRegex(ValueError, "CREATE_ONLY"):
                c.run(self.output, env=self.env)
            load.assert_not_called()

    def test_module_has_no_model_load_forward_or_audio_input_cli(self):
        tree = ast.parse((HERE / "english_kokoro_context32_processor_check.py").read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn("Wav2Vec2ForCTC", names)
        self.assertNotIn("Wav2Vec2Config", names)
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertNotIn("_load_native", attrs)
        self.assertNotIn("native_forward", attrs)
        self.assertNotIn("snapshot_download", attrs)
        self.assertNotIn("load_state_dict", attrs)


if __name__ == "__main__":
    unittest.main()
