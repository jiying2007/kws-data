"""Saved-byte comparator fixtures; all PCM/logits are mock data, no model calls.

Only source.readback/original_probe are mocked for the complete comparison;
the actual new writer, byte hashes, metadata, counter and logit readers execute.
The official small metadata fixture contains no weights or audio.
"""
from __future__ import annotations
import base64
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
from unittest.mock import patch
import wave
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_kokoro_context32_compare as c
import english_kokoro_context32_observe as o
import test_english_kokoro_context32_observe as writer_fixture

SMALL_METADATA_BASE64 = "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"
SMALL = json.loads(zlib.decompress(base64.b64decode(SMALL_METADATA_BASE64)))
VOCAB_BYTES = SMALL["common/vocab.json"].encode()
VOCAB = c.vocabulary(VOCAB_BYTES)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def descriptor(path):
    raw = path.read_bytes()
    return {"path": path.name, "bytes": len(raw), "sha256": o.sha(raw)}


def target_tokens(row):
    return {"mero-direct-eh": ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"],
            "mero-direct-i": ["h", "eɪ", "m", "iː", "ɹ", "oʊ"],
            "vosen-direct-s": ["h", "eɪ", "v", "oʊ", "s", "ə", "n"],
            "vosen-direct-z": ["h", "eɪ", "v", "oʊ", "z", "ə", "n"]}[row["cell_id"]]


class MockFullNative(writer_fixture.FakeNative):
    def __init__(self, *, replacements=None, **kwargs):
        super().__init__(**kwargs)
        self.replacements = replacements or {}

    def prepare(self, pcm, row):
        batch, facts, norm, mask = super().prepare(pcm, row)
        values = [value[0] / 32768.0 for value in struct.iter_unpack("<h", pcm)]
        mean = math.fsum(values) / len(values)
        variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
        # Each mock input is a constant integer PCM, hence its exact normalized
        # input is zero. This is a fixture, not a real acoustic observation.
        if variance != 0:
            raise ValueError("FIXTURE_CONSTANT_PCM_REQUIRED")
        facts.update(sample_rate=16000, pcm_samples=len(values), pcm_divisor=32768.0,
            do_normalize=True, normalization_scope="Entire new full-carrier input, independently of the old crop.",
            pcm_float_mean=mean, pcm_float_std_population=0.0, normalized_mean=0.0,
            normalized_std_population=0.0)
        return batch, facts, norm, mask

    def materialize(self, output, row):
        if row["id"] == self.materialize_failure:
            raise ValueError("MOCK_AFTER_NATIVE_RETURN")
        frames = o.frame_geometry(row["pcm_samples"], row["full_to_crop_offset_16k"])["frames"]
        wanted = self.replacements.get(row["id"], target_tokens(row))
        ids = [0] * frames
        ids[:3] = [VOCAB[x] for x in ["ð", "ə", "m"]]
        ids[-len(wanted):] = [VOCAB[x] for x in wanted]
        raw = bytearray()
        for index in ids:
            logits = [-1.0] * 392
            logits[index] = 2.0
            raw.extend(struct.pack("<392f", *logits))
        tokens = [x["token"] for x in c.ctc_runs(ids, VOCAB)]
        return {"shape": [1, frames, 392], "dtype": "float32-little-endian", "order": "C",
                "frame_ids": ids, "phone_tokens": tokens,
                "argmax_special_frame_counts": {str(k): ids.count(k) for k in (0, 1, 2, 3)}}, bytes(raw)


class ComparatorFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="context32-compare-mock-")
        self.root = Path(self.temp.name)
        self.source_root = self.root / "prepared"
        self.source_root.mkdir()
        self.rows, self.plan = writer_fixture.mock_inputs(self.source_root)
        for index, row in enumerate(self.rows):
            phase = (index % 4) * 80
            row.update(native_samples=9600, crop_start_native_sample=4800 + phase * 3 // 2,
                crop_end_native_sample=9600, full_to_crop_offset_16k=3200 + phase,
                conv_grid_offset_mod_320=phase, old_pcm_samples=3200 - phase,
                pcm_samples=6400, preparation_status="COMPLETE",
                derivation_path="source/audio/" + row["id"] + "/DERIVATION.json")
            pcm = struct.pack("<h", index + 1) * row["pcm_samples"]
            stream = io.BytesIO()
            with wave.open(stream, "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
            wav_bytes = stream.getvalue()
            (self.source_root / row["pcm_path"]).write_bytes(pcm)
            (self.source_root / row["wav_path"]).write_bytes(wav_bytes)
            row.update(pcm_sha256=o.sha(pcm), wav_sha256=o.sha(wav_bytes))
            write_json(self.source_root / row["derivation_path"], {"fixture_only": True,
                "old_crop_comparison": {"fixture_only": True, "compared_samples": row["old_pcm_samples"]}})
        self.plan["input_roster"] = [{k: r[k] for k in o.FROZEN_INPUT_FIELDS} for r in self.rows]
        self.plan.update(observer_code_sha256=o.sha(Path(o.__file__).read_bytes()),
            comparator_code_sha256=c.sha(Path(c.__file__).read_bytes()),
            source_code_sha256=c.sha(Path(c.source.__file__).read_bytes()))
        self.run = {"repository": "fixture-only", "commit": "fixture-only", "run_id": "fixture-only",
                    "run_attempt": "fixture-only", "event": "fixture-only"}
        self.manifest = {"schema": o.INPUT_SCHEMA, "fixture_only": True, "rows": self.rows, "run": self.run}
        write_json(self.source_root / "INPUT_MANIFEST.json", self.manifest)
        self.checked, self.manifest_sha = o.validate_inputs(self.source_root, self.rows, self.plan)
        self.observer_roots = {p: self.root / p for p in o.PROVIDERS}
        self.old_files, self.old_rows = self.old_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def old_fixture(self):
        files, old_rows = {}, {}
        for row in self.rows:
            old = {"id": row["source_id"], "reference_tier": "unknown", "reference_supported": False,
                   "reference_observed_hypothesis": None, "reason": "MOCK_ORIGINAL_UNKNOWN_NOT_RELABELLED"}
            for provider in o.PROVIDERS:
                tokens = target_tokens(row)
                if row["requested_class"] == 2:
                    tokens[-1] = "ɔ"
                value = {"schema": "english-kokoro48-native-capture-v1", "id": row["source_id"],
                    **o.observer_identity(provider), "source_pcm_sha256": row["old_pcm_sha256"],
                    "source_wav_sha256": o.sha(("mock-old-wav-" + row["id"]).encode()),
                    "source_samples": row["old_pcm_samples"], "status": "success", "input_coverage_complete": True,
                    "decoder_output_complete": True, "target_conditioned": False,
                    "diagnostic_only": provider == "phoneme_lv60", "phone_tokens": tokens}
                path = "evidence/" + provider + "/" + row["source_id"] + "/CAPTURE.json"
                files[path] = json.dumps(value, ensure_ascii=False).encode()
                old["xlsr_normalized_phones" if provider == "phoneme" else "lv60_normalized_phones"] = c.normalized_phones(tokens)
            old_rows[row["source_id"]] = old
        return files, old_rows

    def provider(self, provider, *, fake=None, initial_failure=False):
        root = self.observer_roots[provider]
        root.mkdir()
        o.once(root / "STARTED.json", {"schema": o.SCHEMA, "provider": provider, "run": self.run,
            "input_manifest_sha256": self.manifest_sha, "maximum_calls": 16, "retry_count": 0,
            "observer_identity": o.observer_identity(provider), "diagnostic_only": True})
        ledger = o.initial_rows(self.checked)
        if initial_failure:
            return o.finish(root, provider, self.run, ledger, "MOCK_LOAD_FAILURE_NO_MODEL", self.manifest_sha)
        metadata = root / "model-metadata"
        metadata.mkdir()
        for name in o.OBSERVER_FILES[provider]:
            if name == "pytorch_model.bin":
                continue
            key = provider + "/tokenizer_config.json" if name == "tokenizer_config.json" else "common/" + name
            (metadata / name).write_bytes(SMALL[key].encode())
        _, function_sha = o.ctc_function((HERE / "english_final_ipa_xlsr_once.py").read_bytes())
        lock = {"schema": o.SCHEMA, "model": o.OBSERVERS[provider], "files": o.OBSERVER_FILES[provider],
            "runtime_versions": o.RUNTIME_VERSIONS, "preprocessor_config": json.loads(SMALL["common/preprocessor_config.json"]),
            "ctc_helper_sha256": o.CTC_HELPER_SHA256, "ctc_function_sha256": function_sha,
            "blank_id": 0, "skipped_special_ids": [0, 1, 2, 3], "collapse_before_blank_deletion": True,
            "weights_only": True, "strict_state_dict_loading": True, "eval": True, "model_parameters_modified": False,
            "trust_remote_code": False, "phonemizer_used": False, "target_conditioned": False,
            "new_model_constructions": 1, "native_forward_attempts_at_initialization": 0,
            "observer_identity": o.observer_identity(provider), "observer_code_sha256": self.plan["observer_code_sha256"],
            "input_manifest_sha256": self.manifest_sha, "fixture_only": True}
        write_json(root / "MODEL_LOCK.json", lock)
        failure = None
        try:
            o._execute_rows(self.checked, root, provider, self.run, (fake or MockFullNative()).functions(), ledger)
        except BaseException as exc:
            failure = type(exc).__name__ + ": " + str(exc)
        return o.finish(root, provider, self.run, ledger, failure, self.manifest_sha)

    def read_provider(self, provider="phoneme"):
        return c.read_provider(self.observer_roots[provider], self.rows, self.manifest_sha,
            self.run, provider, self.plan, source_root=self.source_root)

    def compare(self):
        with (patch.object(c.source, "readback", return_value=self.manifest) as source_readback,
              patch.object(c, "original_probe", return_value=(self.old_files, self.old_rows))):
            result = c.compare(self.source_root, self.observer_roots, self.plan)
        source_readback.assert_called_once_with(self.source_root, self.plan, allow_partial=True)
        return result

    def capture(self, provider="phoneme", index=0):
        directory = self.observer_roots[provider] / self.rows[index]["id"]
        return directory, json.loads((directory / "CAPTURE.json").read_bytes())

    def reseal(self, directory, capture, *, processor=False, provider="phoneme"):
        write_json(directory / "CAPTURE.json", capture)
        if processor:
            item = json.loads((directory / "INPUT.json").read_bytes())
            for key in ("processor", "normalized_input", "attention_mask"):
                item[key] = capture[key]
            write_json(directory / "INPUT.json", item)
        path = self.observer_roots[provider] / "RESULT.json"
        ledger = json.loads(path.read_bytes())
        for item in ledger["rows"]:
            if item["id"] == directory.name:
                item["capture_sha256"] = o.sha((directory / "CAPTURE.json").read_bytes())
        write_json(path, ledger)

    def test_argmax_first_index_ties_and_negative_logits(self):
        a, b = [-9.0] * 392, [-4.0] * 392
        a[19] = a[49] = 2.0
        b[40] = -1.0
        self.assertEqual(c.argmax_ids(struct.pack("<784f", *(a + b)), 2), [19, 40])

    def test_argmax_rejects_truncated_shape_nan_and_infinity(self):
        with self.assertRaisesRegex(ValueError, "COMPLETE_LOGIT_SHAPE"):
            c.argmax_ids(b"\0" * (391 * 4), 1)
        for bad in (float("nan"), float("inf"), -float("inf")):
            values = [0.0] * 392
            values[-1] = bad
            with self.assertRaisesRegex(ValueError, "FINITE_ALL_LOGITS"):
                c.argmax_ids(struct.pack("<392f", *values), 1)

    def test_ctc_blank_separated_repeat_and_absolute_support(self):
        runs = c.ctc_runs([VOCAB["h"], VOCAB["h"], 0, VOCAB["h"], 1, VOCAB["eɪ"]], VOCAB, offset_frame=7)
        self.assertEqual([x["token"] for x in runs], ["h", "h", "eɪ"])
        self.assertEqual(runs[0]["nominal_conv_support_union_samples"], [2240, 2960])
        self.assertEqual(runs[1]["frame_start"], 10)

    def test_whole_and_suffix_report_every_hypothesis_separately(self):
        for cell, expected in zip(o.CELLS, ["mero-us-eh-v1", "mero-us-i-v1", "vosen-us-s-v1", "vosen-us-z-v1"]):
            tokens = target_tokens({"cell_id": cell})
            whole = c.phone_diagnostics(tokens)
            suffix = c.phone_diagnostics(["ð", "ə"] + tokens)
            self.assertEqual(whole["whole_pattern_hypotheses"], [expected])
            self.assertEqual(suffix["whole_pattern_hypotheses"], [])
            self.assertEqual([x["hypothesis"] for x in suffix["terminal_suffix_matches"]], [expected])
            self.assertIsNone(suffix["admission_decision"])
            self.assertFalse(suffix["acoustic_gold"])

    def test_rejected_vowels_and_i5_are_not_normalized_into_targets(self):
        for tail in ("ɔ", "aʊ", "ɑ"):
            value = c.phone_diagnostics(["h", "eɪ", "m", "ɛ", "ɹ", tail])
            self.assertEqual(value["terminal_suffix_matches"], [])
            self.assertEqual(value["mero_terminal_after_r"], tail)
        self.assertEqual(c.phone_diagnostics(["h", "eɪ", "m", "i5", "ɹ", "oʊ"])["terminal_suffix_matches"], [])

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":{"x":1,"x":2}}'):
            with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
                c.loads(raw)
        with self.assertRaisesRegex(ValueError, "NONFINITE_JSON"):
            c.loads(b'{"a":NaN}')

    def test_actual_writer_packet_roundtrip_all32_and_readonly(self):
        for provider in o.PROVIDERS:
            self.provider(provider)
        before = {str(p.relative_to(self.root)): o.sha(p.read_bytes()) for p in self.root.rglob("*") if p.is_file()}
        result = self.compare()
        after = {str(p.relative_to(self.root)): o.sha(p.read_bytes()) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["status"], "COMPLETE_DIAGNOSTIC_NO_ADMISSION")
        self.assertEqual((result["source_input_count"], result["declared_observer_pairs"], result["actual_forward_attempts"], result["native_forward_returns"], result["completed_new_captures"]), (16, 32, 32, 32, 32))
        self.assertEqual(len(result["rows"]), 16)
        self.assertEqual(len(result["by_cell_and_provider"]), 8)
        self.assertTrue(all(r["declared_count"] == 4 and r["completed_new_captures"] == 4 for r in result["by_cell_and_provider"]))
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["high_quality_model_passed"])
        self.assertIsNone(result["selected_observer"])
        self.assertEqual(result["old_probe_decision"], "STOP_DIAGNOSTIC_NO_EXPANSION")

    def test_cross_variant_gain_is_not_counted_for_requested_variant(self):
        ident = self.rows[0]["id"]
        self.provider("phoneme", fake=MockFullNative(replacements={ident: target_tokens({"cell_id": "mero-direct-i"})}))
        self.provider("phoneme_lv60")
        result = self.compare()
        value = result["rows"][0]["observers"]["phoneme"]
        self.assertFalse(value["new_full_terminal_suffix_matches_request"])
        suffixes = value["new_full"]["capture"]["full_greedy"]["terminal_suffix_matches"]
        self.assertEqual([s["hypothesis"] for s in suffixes], ["mero-us-i-v1"])
        self.assertTrue(value["old_reference_decision_preserved"])

    def test_after_return_failure_keeps32_attempts_31_captures(self):
        self.provider("phoneme", fake=MockFullNative(materialize_failure=self.rows[3]["id"]))
        self.provider("phoneme_lv60")
        result = self.compare()
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed_new_captures"]), (32, 32, 31))
        self.assertEqual(result["status"], "INCOMPLETE_DIAGNOSTIC_NO_ADMISSION")
        self.assertEqual(result["providers"]["phoneme"]["failed"], 1)
        self.assertEqual(len(result["rows"]), 16)

    def test_initialization_failure_is_zero_calls_with16_rows(self):
        self.provider("phoneme", initial_failure=True)
        self.provider("phoneme_lv60")
        result = self.compare()
        self.assertEqual(result["actual_forward_attempts"], 16)
        self.assertEqual(result["providers"]["phoneme"]["not_started"], 16)
        self.assertEqual(result["completed_new_captures"], 16)

    def test_missing_final_ledger_remains_unknown_with_bounds(self):
        self.provider("phoneme", fake=MockFullNative(interrupt=self.rows[3]["id"]))
        self.provider("phoneme_lv60")
        (self.observer_roots["phoneme"] / "RESULT.json").unlink()
        result = self.compare()
        self.assertIsNone(result["actual_forward_attempts"])
        self.assertEqual((result["forward_attempts_lower_bound"], result["forward_attempts_upper_bound"]), (20, 32))
        self.assertEqual(result["providers"]["phoneme"]["declared_count"], 16)
        self.assertEqual(len(result["rows"]), 16)

    def test_missing_provider_retains_all_pairs_and_unknown_calls(self):
        self.provider("phoneme")
        result = self.compare()
        self.assertIsNone(result["actual_forward_attempts"])
        self.assertEqual((result["forward_attempts_lower_bound"], result["forward_attempts_upper_bound"]), (16, 32))
        self.assertEqual(result["declared_observer_pairs"], 32)
        self.assertEqual(result["completed_new_captures"], 16)

    def test_resealed_frame_id_tamper_fails_independent_argmax(self):
        self.provider("phoneme")
        directory, capture = self.capture()
        value = json.loads((directory / "frame-ids.json").read_bytes())
        value["frame_ids"][0] = (value["frame_ids"][0] + 1) % 392
        write_json(directory / "frame-ids.json", value)
        capture["frame_ids"] = descriptor(directory / "frame-ids.json")
        self.reseal(directory, capture)
        with self.assertRaisesRegex(ValueError, "INDEPENDENT_ARGMAX_FRAME_IDS"):
            self.read_provider()

    def test_resealed_ctc_token_tamper_fails_independent_decode(self):
        self.provider("phoneme")
        directory, capture = self.capture()
        capture["phone_tokens"][-1] = "ɔ"
        self.reseal(directory, capture)
        with self.assertRaisesRegex(ValueError, "INDEPENDENT_COLLAPSE"):
            self.read_provider()

    def test_empty_pure_ctc_allowed_but_success_capture_must_match_producer(self):
        self.assertEqual(c.ctc_runs([0, 0, 0], VOCAB), [])
        self.assertEqual(c.phone_diagnostics([])["whole_pattern_hypotheses"], [])
        self.provider("phoneme")
        directory, capture = self.capture()
        frames = capture["logits"]["shape"][1]
        (directory / "logits.f32").write_bytes(b"\0" * (frames * 392 * 4))
        capture["logits"].update(descriptor(directory / "logits.f32"))
        write_json(directory / "frame-ids.json", {"schema": o.SCHEMA, "id": self.rows[0]["id"], "frame_ids": [0] * frames})
        capture["frame_ids"] = descriptor(directory / "frame-ids.json")
        capture["phone_tokens"] = []
        capture["argmax_special_frame_counts"] = {"0": frames, "1": 0, "2": 0, "3": 0}
        self.reseal(directory, capture)
        with self.assertRaisesRegex(ValueError, "INDEPENDENT_COLLAPSE"):
            self.read_provider()

    def test_nominal_grid_alignment_is_distinct_from_output_equivalence(self):
        self.provider("phoneme")
        result = self.read_provider()
        for index, phase in enumerate((0, 80, 160, 240)):
            region = result["rows"][index]["capture"]["fixed_region"]
            self.assertEqual(region["nominal_grid_phase_aligned"], phase == 0)
            self.assertFalse(region["old_crop_output_equivalence_claimed"])
            self.assertEqual(region["first_frame_offset_from_crop_samples"], (320 - phase) % 320)
            self.assertFalse(region["whole_phrase_admission"])

    def test_resealed_nan_in_logits_is_rejected(self):
        self.provider("phoneme")
        directory, capture = self.capture()
        raw = bytearray((directory / "logits.f32").read_bytes())
        struct.pack_into("<f", raw, 0, float("nan"))
        (directory / "logits.f32").write_bytes(raw)
        capture["logits"].update(descriptor(directory / "logits.f32"))
        self.reseal(directory, capture)
        with self.assertRaisesRegex(ValueError, "FINITE_ALL_LOGITS"):
            self.read_provider()

    def test_shape_claim_cannot_truncate_saved_logits(self):
        self.provider("phoneme")
        directory, capture = self.capture()
        capture["logits"]["shape"][1] -= 1
        self.reseal(directory, capture)
        with self.assertRaisesRegex(ValueError, "LOGIT_FORMAT_NO_TRUNCATION"):
            self.read_provider()

    def test_changed_mask_is_rejected_after_reseal(self):
        self.provider("phoneme")
        directory, capture = self.capture()
        raw = bytearray((directory / "attention-mask.i64").read_bytes())
        struct.pack_into("<q", raw, 0, 0)
        (directory / "attention-mask.i64").write_bytes(raw)
        capture["attention_mask"] = descriptor(directory / "attention-mask.i64")
        self.reseal(directory, capture, processor=True)
        with self.assertRaisesRegex(ValueError, "UNPADDED_PROCESSOR_INPUT_BYTES"):
            self.read_provider()

    def test_changed_normalization_is_not_hidden_by_matching_stats(self):
        self.provider("phoneme")
        directory, capture = self.capture()
        count = self.rows[0]["pcm_samples"]
        values = [0.0] * count
        values[0] = 1.0
        (directory / "normalized-input.f32").write_bytes(struct.pack("<" + "f" * count, *values))
        mean = 1.0 / count
        variance = math.fsum((x - mean) ** 2 for x in values) / count
        capture["normalized_input"] = descriptor(directory / "normalized-input.f32")
        capture["processor"].update(normalized_mean=mean, normalized_std_population=math.sqrt(variance))
        self.reseal(directory, capture, processor=True)
        with self.assertRaisesRegex(ValueError, "NORMALIZED_INPUT_PCM_RELATIONSHIP"):
            self.read_provider()

    def test_duplicate_key_inside_actual_capture_is_rejected(self):
        self.provider("phoneme")
        directory, _ = self.capture()
        path = directory / "CAPTURE.json"
        raw = path.read_text()
        path.write_text(raw.replace('"status": "success"', '"status": "success", "status": "success"'))
        with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
            self.read_provider()

    def test_counter_tamper_cannot_erase_consumed_call(self):
        self.provider("phoneme")
        path = self.observer_roots["phoneme"] / "RESULT.json"
        result = json.loads(path.read_bytes())
        result["actual_forward_attempts"] -= 1
        write_json(path, result)
        with self.assertRaisesRegex(ValueError, "EXACT_OBSERVER_COUNTERS"):
            self.read_provider()

    def test_drop_or_duplicate_ledger_row_is_rejected(self):
        self.provider("phoneme")
        path = self.observer_roots["phoneme"] / "RESULT.json"
        result = json.loads(path.read_bytes())
        result["rows"][-1] = copy.deepcopy(result["rows"][0])
        write_json(path, result)
        with self.assertRaisesRegex(ValueError, "COMPLETE_ORDERED_PROVIDER_LEDGER"):
            self.read_provider()

    def test_marker_attempt_tamper_is_rejected(self):
        self.provider("phoneme")
        directory, _ = self.capture()
        path = directory / "FORWARD_STARTED.json"
        value = json.loads(path.read_bytes())
        value["forward_attempts"] = 0
        write_json(path, value)
        with self.assertRaises(ValueError):
            self.read_provider()

    def test_marker_return_tamper_is_rejected_even_without_final_ledger(self):
        self.provider("phoneme")
        directory, _ = self.capture()
        path = directory / "FORWARD_RETURNED.json"
        value = json.loads(path.read_bytes())
        value["forward_returned"] = False
        write_json(path, value)
        (self.observer_roots["phoneme"] / "RESULT.json").unlink()
        with self.assertRaises(ValueError):
            self.read_provider()

    def test_old_reference_capture_identity_tamper_is_rejected(self):
        row = self.rows[0]
        key = "evidence/phoneme/" + row["source_id"] + "/CAPTURE.json"
        old = json.loads(self.old_files[key])
        old["source_pcm_sha256"] = "f" * 64
        self.old_files[key] = json.dumps(old).encode()
        with self.assertRaisesRegex(ValueError, "ORIGINAL_CROP_CAPTURE_IDENTITY"):
            c.old_capture(self.old_files, self.old_rows, row, "phoneme")

    def test_source_pin_change_prevents_comparison(self):
        self.plan["comparator_code_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "FROZEN_COMPARISON_SOURCE"):
            c.compare(self.source_root, self.observer_roots, self.plan)


if __name__ == "__main__":
    unittest.main()
