"""Pure standard-library fixtures: no real processor, resampler or model."""
from __future__ import annotations
import copy
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_kokoro_context64_phone as phone

RUN = {"repository": "jiying2007/kws-data", "commit": "a" * 40,
       "run_id": "7654321", "run_attempt": "1", "event": "push"}
VOCAB = {"token" + str(i): i for i in range(392)}
VOCAB.pop("token39")
VOCAB["h"] = 39


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def wav(raw):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(raw)
    return stream.getvalue()


def close_fixture_packet(root):
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "FILES.json":
            raw = p.read_bytes()
            files.append({"path":p.relative_to(root).as_posix(), "bytes":len(raw), "sha256":phone.sha(raw)})
    dump(root / "FILES.json", {"schema":"english-kokoro-context64-once-v1/files", "files":files})


def plan():
    return {"enabled": True, "campaign_id": phone.CAMPAIGN_ID, "main_before": phone.MAIN_BEFORE,
        "parent_phone_calls_per_provider": 48, "child_text_calls_per_provider": 64,
        "native_tts_request_budget": 48, "total_high_level_model_calls_max": 336,
        "retry_count": 0, "training_allowed": False, "automatic_expansion": False,
        "phone_code_sha256": phone.sha(Path(phone.__file__).read_bytes()),
        "helper_sha256": {phone.NATIVE_PATH: phone.NATIVE_SHA, phone.MATH_PATH: phone.MATH_SHA},
        "observers": copy.deepcopy(phone.OBSERVERS), "observer_files": copy.deepcopy(phone.OBSERVER_FILES)}


class Fixture:
    def __init__(self, root, valid=1):
        self.root = Path(root)
        self.source = self.root / "source-packet"
        self.output = self.root / "observer"
        self.source.mkdir()
        self.output.mkdir()
        self.rows = []
        for i in range(48):
            row = {"id": "parent" + str(i), "generation_status": "success" if i < valid else "not_started"}
            if i < valid:
                raw = struct.pack("<640h", *[((j * 73 + 19 + i) % 16001) - 8000 for j in range(640)])
                row.update(pcm_path="source/audio/" + row["id"] + ".pcm",
                           wav_path="source/audio/" + row["id"] + ".wav",
                           pcm_samples=640, sample_rate=16000, channels=1, sample_width=2,
                           pcm_sha256=phone.sha(raw), wav_sha256=phone.sha(wav(raw)))
                for key, b in (("pcm_path", raw), ("wav_path", wav(raw))):
                    p = self.source / row[key];p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(b)
            self.rows.append(row)
        dump(self.source / "source/RESULT.json", {"run": RUN, "parents": self.rows})
        self.checked, self.source_sha = phone.validate_inputs(self.source, self.rows, RUN)
        self.ledger = phone.initial_rows(self.rows)
        self.plan = plan()
        self.calls = []
        self.start_provider()

    def start_provider(self):
        dump(self.output / "STARTED.json", {"schema": phone.SCHEMA, "run": RUN, "provider": "phoneme",
            "input_scope": phone.INPUT_SCOPE, "source_result_sha256": self.source_sha,
            "declared_count": 48, "valid_source_count": len(self.checked), "maximum_calls": 48,
            "retry_count": 0, "phone_code_sha256": self.plan["phone_code_sha256"],
            "native_helper_sha256": phone.NATIVE_SHA})
        # This marker is an explicitly mocked model lock, replaced only in the
        # saved-byte fixture reader. No native construction occurs in tests.
        dump(self.output / "MODEL_LOCK.json", {"fixture_only": True})

    def prepare(self, raw, row):
        if set(row) != {"pcm_samples", "full_to_crop_offset_16k"} or row["full_to_crop_offset_16k"] != 0:
            raise ValueError("TARGET_METADATA_LEAKED_TO_BACKEND")
        values = [v[0] / 32768 for v in struct.iter_unpack("<h", raw)]
        mean = math.fsum(values) / len(values)
        var = math.fsum((v - mean) ** 2 for v in values) / len(values)
        normalized = struct.pack("<640f", *[(v - mean) / math.sqrt(var + 1e-7) for v in values])
        ns = list(v[0] for v in struct.iter_unpack("<f", normalized))
        nm = math.fsum(ns) / len(ns)
        nstd = math.sqrt(math.fsum((v - nm) ** 2 for v in ns) / len(ns))
        facts = {"sample_rate": 16000, "pcm_samples": 640, "pcm_divisor": 32768.0,
            "do_normalize": True, "pcm_float_mean": mean, "pcm_float_std_population": math.sqrt(var),
            "normalized_mean": nm, "normalized_std_population": nstd,
            "normalized_shape": [1, 640], "normalized_dtype": "torch.float32",
            "attention_mask_shape": [1, 640], "attention_mask_dtype": "torch.int32",
            "attention_mask_sum": 640, "attention_mask_unique_values": [1],
            "valid_input_samples": 640, "padded_input_samples": 640, "right_padding_samples": 0}
        return {"mock_batch": True}, facts, normalized, struct.pack("<640i", *([1] * 640))

    def forward(self, batch):
        self.calls.append(batch)
        return {"mock_return": True}

    def materialize(self, returned, row):
        if returned != {"mock_return": True}:
            raise ValueError("MOCK_RETURN")
        values = [-5.0] * 392
        values[39] = 4.0
        return {"shape": [1, 1, 392], "dtype": "float32-little-endian", "order": "C",
            "frame_ids": [39], "phone_tokens": ["h"],
            "argmax_special_frame_counts": {str(k): 0 for k in range(4)}}, struct.pack("<392f", *values)

    def execute(self, prepare=None, forward=None, materialize=None):
        phone.execute_rows(self.checked, self.output, "phoneme", RUN, self.source_sha,
            (prepare or self.prepare, forward or self.forward, materialize or self.materialize), self.ledger)

    def finish(self, failure=None):
        result = phone.finish(self.output, "phoneme", RUN, self.ledger, failure, self.source_sha)
        close_fixture_packet(self.output)
        return result

    def read(self):
        with patch.object(phone, "_verify_lock", return_value=VOCAB):
            return phone.readback(provider="phoneme", observer_root=self.output,
                source_root=self.source, plan=self.plan, run_identity=RUN, parent_rows=self.rows)


class ParentPhoneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def fixture(self, valid=1):
        return Fixture(self.root, valid)
    def mutate_json(self, path, mutation):
        value = json.loads(path.read_bytes());mutation(value);dump(path, value)
        for directory in path.parents:
            if (directory / "FILES.json").is_file():
                close_fixture_packet(directory)
                break
    def test_frozen_real_helper_source_preflight_is_standard_library_only(self):
        self.assertEqual(phone.source_preflight(plan())["new_model_calls"], 0)
    def test_one_success_47_unavailable_sources_have_complete_ledger(self):
        f = self.fixture();f.execute();result = f.finish();read = f.read()
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed"]), (1, 1, 1))
        self.assertEqual(result["not_started"], 47)
        self.assertFalse(read["summary"]["complete"])
        self.assertEqual(read["rows"]["parent0"]["phone_tokens"], ["h"])
        self.assertEqual(len(read["rows"]), 48)
    def test_all48_counters_equal_no_hidden_extra_forward(self):
        f = self.fixture(48);f.execute();r = f.finish();read = f.read()
        self.assertEqual(len(f.calls), 48)
        self.assertEqual((r["actual_forward_attempts"], r["native_forward_returns"], r["completed"]), (48, 48, 48))
        self.assertTrue(read["summary"]["complete"])
    def test_preprocess_failure_consumes_no_forward(self):
        f = self.fixture()
        def bad(*args):raise ValueError("PREPROCESS_FIXTURE")
        f.execute(prepare=bad);r = f.finish();f.read()
        self.assertEqual((r["actual_forward_attempts"], r["native_forward_returns"], r["failed"]), (0, 0, 1))
        self.assertEqual(f.calls, [])
    def test_forward_failure_not_retried(self):
        f = self.fixture()
        def bad(*args):f.calls.append("failed");raise ValueError("FORWARD_FIXTURE")
        f.execute(forward=bad);r = f.finish();f.read()
        self.assertEqual((r["actual_forward_attempts"], r["native_forward_returns"], r["failed"]), (1, 0, 1))
        self.assertEqual(len(f.calls), 1)
    def test_bad_materialization_retains_native_return(self):
        f = self.fixture()
        def bad(*args):raise ValueError("CAPTURE_FIXTURE")
        f.execute(materialize=bad);r = f.finish();f.read()
        self.assertEqual((r["actual_forward_attempts"], r["native_forward_returns"], r["completed"]), (1, 1, 0))
    def test_failed_return_marker_write_retains_native_return_with_failed_receipt(self):
        f = self.fixture();original_once = phone.once
        def fail_return_marker(path, value):
            if path.name == "FORWARD_RETURNED.json":
                raise OSError("RETURN_MARKER_WRITE_FIXTURE")
            return original_once(path, value)
        with patch.object(phone, "once", side_effect=fail_return_marker):f.execute()
        r = f.finish();read = f.read()
        self.assertFalse((f.output / "parent0/FORWARD_RETURNED.json").exists())
        self.assertTrue((f.output / "parent0/FAILED.json").exists())
        self.assertEqual(len(f.calls), 1)
        self.assertEqual((r["actual_forward_attempts"], r["native_forward_returns"], r["completed"]), (1, 1, 0))
        self.assertEqual(read["summary"]["native_forward_returns"], 1)
        self.assertFalse(read["summary"]["complete"])
    def test_return_marker_requires_failed_receipt_to_retain_true_return(self):
        f = self.fixture()
        def bad(*args):raise ValueError("CAPTURE_FIXTURE")
        f.execute(materialize=bad);f.finish()
        self.mutate_json(f.output / "parent0/FAILED.json", lambda c:c.update(forward_returned=False))
        with self.assertRaisesRegex(ValueError, "FAILED_RECEIPT_MATCHES"):f.read()
    def test_terminal_cannot_invent_native_return_without_marker_or_failed_evidence(self):
        f = self.fixture()
        def bad(*args):raise ValueError("FORWARD_FIXTURE")
        f.execute(forward=bad);f.finish()
        def forged(c):
            c["rows"][0]["forward_returned"] = True
            c["native_forward_returns"] = 1
        self.mutate_json(f.output / "RESULT.json", forged)
        with self.assertRaisesRegex(ValueError, "LEDGER_MATCHES"):f.read()
    def test_interruption_marks_failed_and_does_not_restart(self):
        f = self.fixture(2)
        def interrupted(*args):raise KeyboardInterrupt("fixture")
        with self.assertRaises(KeyboardInterrupt):f.execute(forward=interrupted)
        r = f.finish("KeyboardInterrupt")
        f.read()
        self.assertEqual((r["actual_forward_attempts"], r["failed"], r["not_started"]), (1, 1, 47))
    def test_missing_terminal_keeps_unknown_actual_counts(self):
        f = self.fixture();f.execute();r = f.read()["summary"]
        self.assertIsNone(r["actual_forward_attempts"])
        self.assertEqual((r["forward_attempts_lower_bound"], r["forward_attempts_upper_bound"]), (1, 1))
        self.assertFalse(r["complete"])
    def test_source_run_cannot_be_substituted(self):
        f = self.fixture()
        with self.assertRaises(ValueError):
            phone.validate_inputs(f.source, f.rows, RUN | {"commit": "b" * 40})
    def test_duplicate_parent_pcm_is_rejected_before_model(self):
        f = self.fixture(2)
        for key in ("pcm_path", "wav_path", "pcm_sha256", "wav_sha256"):
            f.rows[1][key] = f.rows[0][key]
        dump(f.source / "source/RESULT.json", {"run": RUN, "parents": f.rows})
        with self.assertRaisesRegex(ValueError, "UNIQUE_EXACT_PARENT"):
            phone.validate_inputs(f.source, f.rows, RUN)
    def test_parent_capture_cannot_claim_child_input_scope(self):
        f = self.fixture();f.execute();f.finish()
        self.mutate_json(f.output / "parent0/CAPTURE.json", lambda c:c.update(input_scope="child_pcm"))
        with self.assertRaises(ValueError):f.read()
    def test_child_pcm_sha_cannot_replace_actual_parent_sha(self):
        f = self.fixture();f.execute();f.finish()
        self.mutate_json(f.output / "parent0/CAPTURE.json", lambda c:c.update(parent_pcm_sha256="b" * 64))
        with self.assertRaises(ValueError):f.read()
    def test_wrong_int64_mask_descriptor_rejected(self):
        f = self.fixture();f.execute();f.finish()
        for name in ("CAPTURE.json", "INPUT.json"):
            self.mutate_json(f.output / ("parent0/" + name),
                lambda c:c["attention_mask"].update(dtype="int64-little-endian"))
        with self.assertRaises(ValueError):f.read()
    def test_normalized_descriptor_tampering_rejected_even_when_join_matches(self):
        f = self.fixture();f.execute();f.finish()
        original = {name:json.loads((f.output / ("parent0/" + name)).read_bytes())
                    for name in ("CAPTURE.json", "INPUT.json")}
        for alteration in ({"dtype":"float64-little-endian"}, {"order":"F"}, {"shape":[640, 1]}):
            with self.subTest(alteration=alteration):
                for name, value in original.items():
                    changed = copy.deepcopy(value);changed["normalized_input"].update(alteration)
                    dump(f.output / ("parent0/" + name), changed)
                close_fixture_packet(f.output)
                with self.assertRaisesRegex(ValueError, "SAVED_NORMALIZED_INPUT_FORMAT"):f.read()
    def test_normalization_sha_valid_but_values_wrong_rejected(self):
        f = self.fixture();f.execute();f.finish()
        p = f.output / "parent0/normalized-input.f32";p.write_bytes(struct.pack("<640f", *([0.0] * 640)))
        for name in ("CAPTURE.json", "INPUT.json"):
            self.mutate_json(f.output / ("parent0/" + name),
                lambda c:c["normalized_input"].update(sha256=phone.sha(p.read_bytes())))
        with self.assertRaises(ValueError):f.read()
    def test_nonfinite_logits_even_with_updated_digest_rejected(self):
        f = self.fixture();f.execute();f.finish()
        p = f.output / "parent0/logits.f32";b=bytearray(p.read_bytes());b[:4]=struct.pack("<f", math.nan);p.write_bytes(b)
        self.mutate_json(f.output / "parent0/CAPTURE.json",
            lambda c:c["logits"].update(sha256=phone.sha(p.read_bytes())))
        with self.assertRaises(ValueError):f.read()
    def test_forged_terminal_counter_rejected(self):
        f = self.fixture();f.execute();f.finish()
        self.mutate_json(f.output / "RESULT.json", lambda c:c.update(actual_forward_attempts=0))
        with self.assertRaises(ValueError):f.read()
    def test_terminal_requires_closed_phone_packet(self):
        f = self.fixture();f.execute();f.finish()
        (f.output / "FILES.json").unlink()
        with self.assertRaisesRegex(ValueError, "PHONE_TERMINAL_REQUIRES_CLOSED_FILES"):f.read()
    def test_phone_packet_manifest_tampering_rejected(self):
        f = self.fixture();f.execute();f.finish()
        p = f.output / "FILES.json";value = json.loads(p.read_bytes())
        value["files"][0]["sha256"] = "0" * 64;dump(p, value)
        with self.assertRaisesRegex(ValueError, "EXACT_CLOSED_PHONE_PACKET_FILES"):f.read()
    def test_missing_forward_marker_not_silently_counted_zero(self):
        f = self.fixture();f.execute();f.finish()
        (f.output / "parent0/FORWARD_STARTED.json").unlink()
        with self.assertRaises(ValueError):f.read()
    def test_extra_parent_directory_rejected(self):
        f = self.fixture();f.execute();f.finish()
        (f.output / "unplanned-parent").mkdir()
        with self.assertRaisesRegex(ValueError, "EXTRA_PARENT"):f.read()
    def test_phone_helper_pin_drift_rejected(self):
        p = plan();p["helper_sha256"][phone.NATIVE_PATH] = "0" * 64
        with self.assertRaises(ValueError):phone.source_preflight(p)
    def test_public_guard_exact_first_successor_only(self):
        path = self.root / "event.json"
        dump(path, {"before": phone.MAIN_BEFORE, "after": RUN["commit"],
             "repository": {"full_name": RUN["repository"], "private": False}})
        env = {"GITHUB_ACTIONS": "true", "GITHUB_WORKFLOW_REF": phone.WORKFLOW,
            "GITHUB_REPOSITORY": RUN["repository"], "GITHUB_SHA": RUN["commit"],
            "GITHUB_RUN_ID": RUN["run_id"], "GITHUB_RUN_ATTEMPT": "1", "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_PATH": str(path)}
        self.assertEqual(phone.public_guard(plan(), "phoneme", RUN, env), RUN)
        for changed in ({"GITHUB_RUN_ATTEMPT": "2"}, {"GITHUB_EVENT_NAME": "pull_request"},
                        {"GITHUB_ACTIONS": "false"}, {"GITHUB_REF": "refs/heads/other"},
                        {"GITHUB_WORKFLOW_REF": "other"}):
            with self.assertRaises(ValueError):phone.public_guard(plan(), "phoneme", RUN, env | changed)
        dump(path, {"before": "b" * 40, "after": RUN["commit"],
             "repository": {"full_name": RUN["repository"], "private": False}})
        with self.assertRaises(ValueError):phone.public_guard(plan(), "phoneme", RUN, env)


if __name__ == "__main__":
    unittest.main()
