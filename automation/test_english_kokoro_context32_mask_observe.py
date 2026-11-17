"""Pure stdlib/mock observer fixtures: no networks, model loading or inference."""
from __future__ import annotations
import ast
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import wave

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("context32_observer_fixture", HERE / "english_kokoro_context32_mask_observe.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def mock_inputs(root):
    rows = []
    old = [m.sha(("mock-old-" + str(i)).encode()) for i in range(48)]
    for index, (preset, cell) in enumerate((p, c) for p in m.PRESETS for c in m.CELLS):
        ident = "enkctx32-" + m.sha(("mock-new-" + str(index)).encode())[:24]
        folder = root / "source/audio" / ident
        folder.mkdir(parents=True)
        pcm = struct.pack("<h", index + 1) * 2400
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
        wav_bytes = stream.getvalue()
        (folder / "full-16k.pcm").write_bytes(pcm)
        (folder / "full-16k.wav").write_bytes(wav_bytes)
        row = {"id": ident, "source_id": "enk48-" + m.sha(("mock-source-" + str(index)).encode())[:24],
            "preset_id": preset, "mode": "full-carrier", "cell_id": cell,
            "requested_class": 2 if cell.startswith("mero-") else 1,
            "pronunciation_hypothesis": {"mero-direct-eh": "mero-us-eh-v1", "mero-direct-i": "mero-us-i-v1",
                "vosen-direct-s": "vosen-us-s-v1", "vosen-direct-z": "vosen-us-z-v1"}[cell],
            "native_full_f32_sha256": m.sha(("mock-native-" + str(index)).encode()),
            "native_samples": 3600, "crop_start_native_sample": 1200, "crop_end_native_sample": 3600,
            "old_pcm_sha256": old[index], "old_pcm_samples": 1600,
            "full_to_crop_offset_16k": 800, "conv_grid_offset_mod_320": 160,
            "wav_path": "source/audio/" + ident + "/full-16k.wav",
            "pcm_path": "source/audio/" + ident + "/full-16k.pcm",
            "wav_sha256": m.sha(wav_bytes), "pcm_sha256": m.sha(pcm),
            "pcm_samples": 2400, "sample_rate": 16000, "channels": 1, "sample_width": 2,
            "role": "train", "fixture_only": True}
        rows.append(row)
    plan = {"input_roster": [{k: row[k] for k in m.FROZEN_INPUT_FIELDS} for row in rows], "old_observed_pcm_sha256": old}
    save_manifest(root, rows)
    plan["input_manifest_sha256"] = m.sha((root / "INPUT_MANIFEST.json").read_bytes())
    return rows, plan


def save_manifest(root, rows):
    (root / "INPUT_MANIFEST.json").write_text(json.dumps({"schema": m.INPUT_SCHEMA, "fixture_only": True, "rows": rows}))


class Interrupted(BaseException):
    pass


class FakeNative:
    def __init__(self, *, prepare_failure=None, forward_failure=None, materialize_failure=None, interrupt=None, wrong_shape=None):
        self.calls = []
        self.prepared = []
        self.prepare_failure = prepare_failure
        self.forward_failure = forward_failure
        self.materialize_failure = materialize_failure
        self.interrupt = interrupt
        self.wrong_shape = wrong_shape

    def prepare(self, pcm, row):
        self.prepared.append(row["id"])
        if row["id"] == self.prepare_failure:
            raise ValueError("MOCK_BEFORE_FORWARD")
        samples = row["pcm_samples"]
        facts = {"fixture_only": True, "valid_input_samples": samples, "padded_input_samples": samples,
                 "right_padding_samples": 0, "normalized_shape": [1, samples], "normalized_dtype": "torch.float32",
                 "attention_mask_shape": [1, samples], "attention_mask_dtype": "torch.int32",
                 "attention_mask_sum": samples, "attention_mask_unique_values": [1]}
        return row["id"], facts, struct.pack("<f", 0.0) * samples, struct.pack("<i", 1) * samples

    def forward(self, ident):
        self.calls.append(ident)
        if ident == self.interrupt:
            raise Interrupted("MOCK_TIMEOUT")
        if ident == self.forward_failure:
            raise ValueError("MOCK_DURING_FORWARD")
        return {"fixture_native_return": ident}

    def materialize(self, output, row):
        if row["id"] == self.materialize_failure:
            raise ValueError("MOCK_AFTER_NATIVE_RETURN")
        frames = m.frame_geometry(row["pcm_samples"], row["full_to_crop_offset_16k"])["frames"]
        frame = [0.0] * 392
        frame[39] = 1.0
        raw = struct.pack("<392f", *frame) * frames
        return {"shape": [1, frames, 392] if row["id"] != self.wrong_shape else [1, frames - 1, 392],
                "dtype": "float32-little-endian", "order": "C", "frame_ids": [39] * frames,
                "phone_tokens": ["h"], "argmax_special_frame_counts": {"0": 0, "1": 0, "2": 0, "3": 0}}, raw

    def functions(self):
        return self.prepare, self.forward, self.materialize


class MockArray:
    """Tiny stdlib value carrier, never NumPy or a native processor."""
    def __init__(self, values, dtype="float32", shape=None):
        self.values = list(values)
        self.dtype = dtype
        self.shape = shape or (len(self.values),)

    def astype(self, dtype, copy=True):
        return MockArray(self.values, dtype, self.shape)

    def __truediv__(self, divisor):
        return MockArray([x / divisor for x in self.values], self.dtype, self.shape)

    def mean(self, dtype=None):
        return sum(self.values) / len(self.values)

    def std(self, dtype=None):
        mean = self.mean()
        return math.sqrt(sum((x - mean) ** 2 for x in self.values) / len(self.values))

    def tobytes(self, order="C"):
        code = {"<f4": "f", "float32": "f", "<i4": "i", "int32": "i", "<i8": "q"}[self.dtype]
        return struct.pack("<" + code * len(self.values), *self.values)


class MockTensor:
    def __init__(self, values, dtype, *, shape=None, device="cpu"):
        self.values = list(values)
        self.dtype = dtype
        self.shape = shape or (1, len(self.values))
        self.device = SimpleNamespace(type=device)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return MockArray(self.values, self.dtype.removeprefix("torch."), self.shape)

    def __eq__(self, value):
        return SimpleNamespace(all=lambda: all(x == value for x in self.values))

    def sum(self):
        return SimpleNamespace(item=lambda: sum(self.values))


class MockNumpy:
    float64 = "float64"

    @staticmethod
    def frombuffer(raw, dtype):
        if dtype != "<i2":
            raise AssertionError("unexpected mock dtype")
        return MockArray(struct.unpack("<" + "h" * (len(raw) // 2), raw), "int16")

    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda: all(math.isfinite(x) for x in value.values))


class MockTorch:
    float32 = "torch.float32"
    int32 = "torch.int32"

    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda: all(math.isfinite(x) for x in value.values))


class MockExtractor:
    def __init__(self, *, dtype="torch.int32", mask_value=1, mask_shape=None, mask_device="cpu", normalized_value=0.25):
        self.calls = []
        self.dtype = dtype
        self.mask_value = mask_value
        self.mask_shape = mask_shape
        self.mask_device = mask_device
        self.normalized_value = normalized_value
        self.batch = None

    def __call__(self, values, *, sampling_rate, return_tensors):
        self.calls.append((sampling_rate, return_tensors, len(values.values)))
        self.batch = {"input_values": MockTensor([self.normalized_value] * len(values.values), "torch.float32"),
            "attention_mask": MockTensor([self.mask_value] * len(values.values), self.dtype,
                shape=self.mask_shape, device=self.mask_device)}
        return self.batch


class NativeMaskContractFixtures(unittest.TestCase):
    def prepare(self, extractor):
        return m.prepare_native_pcm(MockNumpy, MockTorch, extractor, struct.pack("<4h", -3, 0, 2, 7), {"pcm_samples": 4})

    def test_int32_preserves_batch_and_native_mask_bytes(self):
        extractor = MockExtractor()
        batch, facts, normalized, mask = self.prepare(extractor)
        self.assertIs(batch, extractor.batch)
        self.assertIs(batch["attention_mask"], extractor.batch["attention_mask"])
        self.assertEqual(batch["attention_mask"].dtype, "torch.int32")
        self.assertEqual(facts["attention_mask_dtype"], "torch.int32")
        self.assertEqual(mask, struct.pack("<4i", 1, 1, 1, 1))
        self.assertEqual(normalized, struct.pack("<4f", 0.25, 0.25, 0.25, 0.25))
        self.assertEqual(extractor.calls, [(16000, "pt", 4)])

    def test_old_int64_assumption_is_rejected_without_cast(self):
        extractor = MockExtractor(dtype="torch.int64")
        with self.assertRaisesRegex(ValueError, "ACTUAL_UNPADDED_ATTENTION_MASK"):
            self.prepare(extractor)
        self.assertEqual(extractor.batch["attention_mask"].dtype, "torch.int64")

    def test_mask_shape_device_and_values_remain_strict(self):
        for kwargs in ({"mask_shape": (4,)}, {"mask_device": "cuda"}, {"mask_value": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "ACTUAL_UNPADDED_ATTENTION_MASK"):
                self.prepare(MockExtractor(**kwargs))

    def test_nonfinite_normalized_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ACTUAL_NORMALIZED_INPUT"):
            self.prepare(MockExtractor(normalized_value=float("nan")))

    def test_exact_installed_processor_source_bytes_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b"fixture-only-source\n"
            (root / "processor.py").write_bytes(raw)
            pins = {"processor.py": {"bytes": len(raw), "sha256": m.sha(raw)}}
            with patch.object(m, "PROCESSOR_SOURCE_FILES", pins):
                self.assertEqual(m.verify_processor_sources(root), pins)
                (root / "processor.py").write_bytes(raw + b"#changed\n")
                with self.assertRaisesRegex(ValueError, "EXACT_NATIVE_PROCESSOR_SOURCE"):
                    m.verify_processor_sources(root)


class ObserverFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="context32-mock-only-")
        self.root = Path(self.tmp.name)
        self.source = self.root / "source-packet"
        self.source.mkdir()
        self.rows, self.plan = mock_inputs(self.source)
        self.checked, self.manifest_sha = m.validate_inputs(self.source, self.rows, self.plan)
        self.out = self.root / "provider-output"
        self.out.mkdir()
        self.run = {"repository": "fixture-only", "commit": "fixture-only", "run_id": "fixture-only", "run_attempt": "fixture-only", "event": "fixture-only"}

    def tearDown(self):
        self.tmp.cleanup()

    def execute(self, fake):
        ledger = m.initial_rows(self.checked)
        failure = None
        try:
            m._execute_rows(self.checked, self.out, "phoneme", self.run, fake.functions(), ledger)
        except BaseException as exc:
            failure = type(exc).__name__ + ": " + str(exc)
        result = m.finish(self.out, "phoneme", self.run, ledger, failure, self.manifest_sha)
        return result

    def test_source_import_is_stdlib_only(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_disabled_guard_rejects_before_runtime(self):
        with self.assertRaisesRegex(ValueError, "ENABLED_PUBLIC"):
            m.public_guard({"enabled": False}, "phoneme", {}, env={})

    def test_exact_two_original_observers_and_shared_vocab(self):
        self.assertEqual(set(m.OBSERVERS), {"phoneme", "phoneme_lv60"})
        for field in m.COMMON_FILES:
            self.assertEqual(m.OBSERVER_FILES["phoneme"][field], m.OBSERVER_FILES["phoneme_lv60"][field])
        self.assertEqual(m.observer_identity("phoneme")["decode_config_sha256"], "2b37c89e02afc6b44b8109a9ae38303eca75411564f53b3d61d86985de5af441")

    def test_no_auxiliary_or_legacy_provider(self):
        for provider in ("qwen", "whisper", "parakeet", "replacement"):
            with self.assertRaisesRegex(ValueError, "EXACT_TWO"):
                m.observer_identity(provider)

    def test_exact_sixteen_full_rows_and_no_hidden_data(self):
        self.assertEqual(len(self.checked), 16)
        self.assertEqual(len({r[0]["pcm_sha256"] for r in self.checked}), 16)

    def test_old_pcm_reuse_is_rejected(self):
        self.plan["old_observed_pcm_sha256"][-1] = self.rows[0]["pcm_sha256"]
        with self.assertRaisesRegex(ValueError, "NEW_UNOBSERVED"):
            m.validate_inputs(self.source, self.rows, self.plan)

    def test_duplicate_new_pcm_is_rejected(self):
        a, b = self.rows[:2]
        for kind in ("wav", "pcm"):
            (self.source / b[kind + "_path"]).write_bytes((self.source / a[kind + "_path"]).read_bytes())
            b[kind + "_sha256"] = a[kind + "_sha256"]
        save_manifest(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "NEW_UNOBSERVED"):
            m.validate_inputs(self.source, self.rows, self.plan)

    def test_native_identity_drift_is_rejected(self):
        self.rows[0]["native_full_f32_sha256"] = "1" * 64
        save_manifest(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "FROZEN_NATIVE"):
            m.validate_inputs(self.source, self.rows, self.plan)

    def test_swapped_row_order_is_rejected(self):
        self.rows.reverse()
        save_manifest(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "FROZEN_NATIVE"):
            m.validate_inputs(self.source, self.rows, self.plan)

    def test_missing_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "EXACT_SIXTEEN"):
            m.validate_inputs(self.source, self.rows[:-1], self.plan)

    def test_grid_phase_is_preserved_without_equivalence_claim(self):
        for phase in (0, 80, 160, 240):
            g = m.frame_geometry(121600, 102400 + phase)
            self.assertEqual(g["frames"], (121600 - 400) // 320 + 1)
            self.assertEqual(g["crop_offset_mod_stride"], phase)
            self.assertEqual(g["nominal_support_interval"], "[320*t,320*t+400)")
            self.assertFalse(g["old_crop_frame_index_equivalence_claimed"])

    def test_exact_one_forward_each_and_full_arrays(self):
        fake = FakeNative()
        result = self.execute(fake)
        self.assertEqual(result["status"], "COMPLETE_DIAGNOSTIC")
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed"]), (16, 16, 16))
        self.assertEqual(fake.calls, [r["id"] for r in self.rows])
        for row in self.rows:
            parent = self.out / row["id"]
            capture = json.loads((parent / "CAPTURE.json").read_bytes())
            ids = json.loads((parent / "frame-ids.json").read_bytes())["frame_ids"]
            raw = (parent / "logits.f32").read_bytes()
            values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
            recalculated = [max(range(392), key=lambda j: values[i * 392 + j]) for i in range(len(ids))]
            self.assertEqual(ids, recalculated)
            self.assertEqual(m.sha(raw), capture["logits"]["sha256"])
            self.assertEqual((parent / "normalized-input.f32").stat().st_size, row["pcm_samples"] * 4)
            self.assertEqual((parent / "attention-mask.i32").stat().st_size, row["pcm_samples"] * 4)
            self.assertTrue((parent / "FORWARD_STARTED.json").is_file())
            self.assertTrue((parent / "FORWARD_RETURNED.json").is_file())

    def test_before_forward_failure_consumes_zero_for_that_input(self):
        ident = self.rows[3]["id"]
        fake = FakeNative(prepare_failure=ident)
        result = self.execute(fake)
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed"]), (15, 15, 15))
        self.assertNotIn(ident, fake.calls)
        self.assertFalse((self.out / ident / "FORWARD_STARTED.json").exists())

    def test_forward_failure_counts_once_without_retry(self):
        ident = self.rows[3]["id"]
        fake = FakeNative(forward_failure=ident)
        result = self.execute(fake)
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed"]), (16, 15, 15))
        self.assertEqual(fake.calls.count(ident), 1)
        self.assertFalse((self.out / ident / "FORWARD_RETURNED.json").exists())

    def test_materialization_failure_preserves_native_return(self):
        ident = self.rows[3]["id"]
        result = self.execute(FakeNative(materialize_failure=ident))
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed"]), (16, 16, 15))
        failure = json.loads((self.out / ident / "FAILED.json").read_bytes())
        self.assertEqual(failure["forward_attempts"], 1)
        self.assertTrue(failure["forward_returned"])

    def test_write_failure_preserves_native_return(self):
        ident = self.rows[3]["id"]
        original = m.put
        def bad_put(path, raw):
            if Path(path).name == "logits.f32" and Path(path).parent.name == ident:
                raise OSError("MOCK_DISK_FULL_AFTER_RETURN")
            return original(path, raw)
        with patch.object(m, "put", bad_put):
            result = self.execute(FakeNative())
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed"]), (16, 16, 15))
        self.assertTrue(json.loads((self.out / ident / "FAILED.json").read_bytes())["forward_returned"])

    def test_interrupt_stops_remaining_inputs_with_full_denominator(self):
        result = self.execute(FakeNative(interrupt=self.rows[3]["id"]))
        self.assertEqual((result["declared_count"], result["actual_forward_attempts"], result["native_forward_returns"], result["completed"], result["not_started"]), (16, 4, 3, 3, 12))
        self.assertEqual(len(result["rows"]), 16)
        self.assertEqual(result["retry_count"], 0)
        self.assertFalse(result["training_allowed"])

    def test_bad_materialized_geometry_does_not_erase_call(self):
        result = self.execute(FakeNative(wrong_shape=self.rows[3]["id"]))
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed"]), (16, 16, 15))

    def test_return_without_attempt_is_rejected(self):
        ledger = m.initial_rows(self.checked)
        ledger[0]["forward_returned"] = True
        with self.assertRaisesRegex(ValueError, "ACTUAL_COUNTER"):
            m.finish(self.out, "phoneme", self.run, ledger, None, self.manifest_sha)

    def test_not_started_initialization_failure_retains_all_rows(self):
        result = m.finish(self.out, "phoneme", self.run, m.initial_rows(self.checked), "MOCK_INITIALIZATION_FAILURE", self.manifest_sha)
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed"], result["not_started"]), (0, 0, 0, 16))

    def test_output_existing_prevents_second_execution(self):
        fake = FakeNative()
        self.execute(fake)
        with self.assertRaises(FileExistsError):
            m._execute_rows(self.checked, self.out, "phoneme", self.run, fake.functions(), m.initial_rows(self.checked))
        self.assertEqual(len(fake.calls), 16)

    def test_native_forward_returns_before_logits_materialization(self):
        tree = ast.parse((HERE / "english_kokoro_context32_mask_observe.py").read_text())
        loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_native")
        forward = next(n for n in loader.body if isinstance(n, ast.FunctionDef) and n.name == "native_forward")
        calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "model"]
        self.assertEqual(len(calls), 1)
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "logits" for n in ast.walk(forward)))


if __name__ == "__main__":
    unittest.main()
