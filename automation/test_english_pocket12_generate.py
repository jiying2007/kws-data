"""Stdlib boundary tests only. The backend below never imports or calls a model."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_pocket12_generate as g


RUN = {"repository": "jiying2007/kws-data", "commit": "1" * 40,
       "run_id": "123", "run_attempt": "1", "event": "push"}
PLAN = {"training_allowed": False, "native_tts_request_budget": 12, "retry_count": 0}


class MockBackend:
    """Mock byte fixtures test bookkeeping, not resampling or native compatibility."""
    calls = []
    failing = None

    def __init__(self, assets, output, lock):
        self.output = Path(output)
        lock.update({"status": "MOCK_TEST_BACKEND_NO_MODEL", "test_only": True})
        g.write_json(self.output / "MODEL_LOCK.json", lock, replace=True)

    def generate(self, row):
        self.calls.append(row["id"])
        g.mark_native_started(self.output, row)
        if row["id"] == self.failing:
            raise RuntimeError("mock native request failure")
        g.mark_native_returned(self.output, row)
        r = g.recipe()
        request = {"id": row["id"], "requested_text": row["requested_text"],
            "requested_text_is_observation": False, "model_revision": r["model_revision"],
            "native_parameters": r["native_parameters"], "request_parameters": r["request_parameters"],
            "native_tts_attempts": 1, "native_returned": True, "precomputed_state_unchanged": True,
            "actual_inputs": [{"formatted_text": row["requested_text"], "token_ids": [[1, 2, 3]],
                               "shape": [1, 3], "dtype": "torch.int64"}],
            "actual_short_calls": [{"text_to_generate": row["requested_text"],
                                    "frames_after_eos": 5, "copy_state": True}],
            "mock_test_only": True}
        g.write_json(self.output / "rows" / row["id"] / "NATIVE_REQUEST.json", request)
        return b"mock-result"

    def materialize(self, audio, row, directory):
        native = struct.pack("<2400f", *((0.0, 0.1, -0.1) * 800))
        pcm = struct.pack("<1600h", *((123, -123) * 800))
        (directory / "native_24k.f32").write_bytes(native)
        (directory / "audio.pcm").write_bytes(pcm)
        with wave.open(str(directory / "audio.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 1600, "NONE", "not compressed"))
            stream.writeframes(pcm)
        prefix = "rows/" + row["id"] + "/"
        return {"native_f32_path": prefix + "native_24k.f32", "native_f32_sha256": g.sha(native),
            "native_samples": 2400, "native_sample_rate": 24000, "pcm_path": prefix + "audio.pcm",
            "pcm_sha256": g.sha(pcm), "pcm_samples": 1600, "wav_path": prefix + "audio.wav",
            "wav_sha256": g.digest(directory / "audio.wav"), "resample_calls": 1, "quantization_calls": 1,
            "actual_input_record_path": prefix + "NATIVE_REQUEST.json"}

    def close(self):
        pass


class SourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        MockBackend.calls = []
        MockBackend.failing = None

    def tearDown(self):
        self.temp.cleanup()

    def make(self):
        output = self.root / "source"
        result = g.generate(PLAN, output, self.root / "assets", RUN, backend_factory=MockBackend)
        return output, result

    def test_source_checks_are_stdlib_and_bad_assets_precede_import(self):
        g.source_preflight(PLAN)
        before = set(sys.modules)
        with self.assertRaisesRegex(ValueError, "ASSET_BYTES"):
            g.NativePocket(self.root, self.root, {})
        self.assertFalse({"torch", "numpy", "scipy", "pocket_tts"} & (set(sys.modules) - before))

    def test_initialization_failure_keeps_all_twelve_unstarted(self):
        def failed_factory(*args):
            raise RuntimeError("mock initialization failure")
        output = self.root / "source"
        result = g.generate(PLAN, output, self.root, RUN, backend_factory=failed_factory)
        source, successes = g.validate_source(output, RUN)
        self.assertEqual((result["tts_attempts"], result["tts_returned"], result["not_started_count"]), (0, 0, 12))
        self.assertEqual(len(source["rows"]), 12)
        self.assertEqual(successes, [])
        self.assertFalse(g.source_readback(output)["source_ready"])

    def test_one_failed_request_keeps_denominator_and_never_retries(self):
        MockBackend.failing = "george-marrow"
        output, result = self.make()
        source, successes = g.validate_source(output, RUN)
        self.assertEqual((result["tts_attempts"], result["tts_returned"], result["failed_count"]), (12, 11, 1))
        self.assertEqual(len(source["rows"]), 12)
        self.assertEqual(len(successes), 11)
        self.assertEqual(len(MockBackend.calls), len(set(MockBackend.calls)))
        self.assertEqual(source["rows"][-1]["generation_status"], "success")

    def test_reexecution_rejected_before_backend(self):
        output, _ = self.make()
        with self.assertRaisesRegex(ValueError, "FRESH_OUTPUT_NO_RETRY"):
            g.generate(PLAN, output, self.root, RUN, backend_factory=MockBackend)
        self.assertEqual(len(MockBackend.calls), 12)

    def test_waveform_tamper_and_unlisted_file_rejected(self):
        output, _ = self.make()
        g.validate_source(output, RUN)
        target = output / "rows/george-mero/native_24k.f32"
        target.write_bytes(target.read_bytes()[:-1] + b"\x00")
        with self.assertRaisesRegex(ValueError, "FILES_BYTES"):
            g.validate_source(output, RUN)
        (output / "unlisted.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "CLOSED_FILES_INVENTORY"):
            g.validate_source(output, RUN)

    def test_safetensors_overlap_rejected_without_numeric_import(self):
        header = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                  "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        raw = json.dumps(header).encode()
        path = self.root / "bad.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)
        with self.assertRaisesRegex(ValueError, "TENSOR_STORAGE"):
            g.safetensors_header(path)


if __name__ == "__main__":
    unittest.main()
