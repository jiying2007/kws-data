"""Offline full-packet tests; artificial audio/outputs, no numerical libraries."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_mero_meh_lv60_observe as observe
import test_english_mero_meh_once as source_fixture
from test_english_mero_meh_generate import MOCK_VOCAB

NEW_RUN = {"repository": "jiying2007/kws-data", "run_id": "99002", "run_attempt": "1",
           "commit": "b" * 40, "event": "push"}
# Exact 321-byte public LV60 tokenizer metadata; no weights are test fixtures.
LV60_TOKENIZER = b'{"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "phonemizer_lang": "en-us", "phonemizer_backend": "espeak", "do_lower_case": false, "special_tokens_map_file": "./special_tokens_map.json", "tokenizer_file": null, "name_or_path": "./", "tokenizer_class": "Wav2Vec2PhonemeCTCTokenizer"}'


def fixture_plan(source):
    corpus = observe.read(source / "CORPUS_MANIFEST.json")
    by_cell = {(r["preset_id"], r["cell_id"]): r for r in corpus["rows"]}
    fields = ("id", "preset_id", "cell_id", "speed", "pcm_path", "pcm_sha256", "pcm_samples", "wav_path", "wav_sha256")
    panel = []
    for preset in observe.PRESETS:
        for cell in observe.CELLS:
            row = by_cell[preset, cell]
            panel.append({k: row[k] for k in fields} | {"source_row_sha256": observe.sha(observe.canonical(row))})
    return {"campaign_id": observe.CAMPAIGN, "provider": observe.PROVIDER, "diagnostic_only": True,
        "training_allowed": False, "retry_count": 0, "warmup_count": 0, "enabled": True,
        "maximum_calls_per_observer": 24, "maximum_total_model_calls": 24, "normalization_match_required": True,
        "source_run": corpus["run"], "source_files_manifest_sha256": observe.sha(observe.regular(source / "FILES.json")),
        "panel_rows": panel, "model": observe.native.OBSERVERS[observe.PROVIDER],
        "model_assets": observe.native.OBSERVER_FILES[observe.PROVIDER], "runtime_versions": observe.native.RUNTIME_VERSIONS,
        "time_budget_seconds": {"initialization": 840, "observation": 1800, "readback": 300, "download": 300}}


@contextmanager
def artificial_native_source():
    """Only replace the artificial source fixture's unavailable weight proof."""
    original = observe._load_source_component
    def load(name):
        module = original(name)
        if name == "generate":
            module.verify_native_lock = lambda *a, **k: MOCK_VOCAB
        return module
    with patch.object(observe, "_load_source_component", side_effect=load):
        yield


def environment():
    return {"GITHUB_REPOSITORY": NEW_RUN["repository"], "GITHUB_RUN_ID": NEW_RUN["run_id"],
        "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": NEW_RUN["commit"], "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main"}


class MockLV60:
    def __init__(self, source, plan, fail=None, changed_normalization=False):
        self.source, self.rows, self.calls = source, plan["panel_rows"], []
        self.fail, self.changed_normalization = fail, changed_normalization

    def __call__(self, provider, assets, output):
        if provider != "phoneme_lv60":
            raise AssertionError("LV60_ONLY")
        assets.mkdir()
        metadata = output / "model-metadata"
        metadata.mkdir()
        for name, payload in source_fixture.phone_metadata().items():
            (metadata / name).write_bytes(base64.b64decode(payload))
        (metadata / "tokenizer_config.json").write_bytes(LV60_TOKENIZER)
        vocab = observe.read(metadata / "vocab.json")
        native, cursor = observe.native, [0]
        _, ctc_sha = native.ctc_function(observe.regular(HERE / "english_final_ipa_xlsr_once.py"))
        lock = {"schema": native.SCHEMA, "model": native.OBSERVERS[provider], "files": native.OBSERVER_FILES[provider],
            "runtime_versions": native.RUNTIME_VERSIONS, "native_processor_source_files": native.PROCESSOR_SOURCE_FILES,
            "preprocessor_config": observe.read(metadata / "preprocessor_config.json"),
            "ctc_helper_sha256": native.CTC_HELPER_SHA256, "ctc_function_sha256": ctc_sha,
            "blank_id": 0, "skipped_special_ids": [0, 1, 2, 3], "collapse_before_blank_deletion": True,
            "weights_only": True, "strict_state_dict_loading": True, "eval": True, "model_parameters_modified": False,
            "trust_remote_code": False, "phonemizer_used": False, "target_conditioned": False,
            "new_model_constructions": 1, "native_forward_attempts_at_initialization": 0}
        def prepare(pcm, geometry):
            row = self.rows[cursor[0]]
            cursor[0] += 1
            if geometry != {"pcm_samples": row["pcm_samples"], "full_to_crop_offset_16k": 0}:
                raise AssertionError("ONLY_FULL_GEOMETRY_NO_TARGET")
            if observe.sha(pcm) != row["pcm_sha256"] or len(pcm) != row["pcm_samples"] * 2:
                raise AssertionError("SAME_ENTIRE_PCM_REQUIRED")
            n = row["pcm_samples"]
            values = [x[0] / 32768 for x in struct.iter_unpack("<h", pcm)]
            mean = math.fsum(values) / n
            scale = math.sqrt(math.fsum((x - mean)**2 for x in values) / n + 1e-7)
            normalized = struct.pack("<" + "f" * n, *((x - mean) / scale for x in values))
            if self.changed_normalization and cursor[0] == 1:
                normalized = struct.pack("<f", 99.0) + normalized[4:]
            facts = {"sample_rate": 16000, "pcm_samples": n, "normalized_shape": [1, n],
                "normalized_dtype": "torch.float32", "attention_mask_shape": [1, n], "attention_mask_dtype": "torch.int32",
                "attention_mask_sum": n, "attention_mask_unique_values": [1], "valid_input_samples": n,
                "padded_input_samples": n, "right_padding_samples": 0,
                "normalization_scope": "Historical helper wording deliberately replaced by new scope."}
            return row, facts, normalized, struct.pack("<i", 1) * n
        def forward(row):
            self.calls.append((row["id"], row["pcm_sha256"], row["pcm_samples"]))
            if row["id"] == self.fail:
                raise RuntimeError("ARTIFICIAL_SINGLE_FORWARD_FAILURE")
            return row
        def materialize(row, geometry):
            tokens = ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"]
            frames = native.frame_geometry(geometry["pcm_samples"], 0)["frames"]
            ids = [vocab[t] for t in tokens] + [0] * (frames - len(tokens))
            raw = b"".join(struct.pack("<392f", *(1.0 if i == peak else -1.0 for i in range(392))) for peak in ids)
            return {"shape": [1, frames, 392], "dtype": "float32-little-endian", "order": "C", "frame_ids": ids,
                "phone_tokens": tokens, "argmax_special_frame_counts": {str(k): ids.count(k) for k in (0, 1, 2, 3)}}, raw
        return prepare, forward, materialize, lock


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.original = source_fixture.mock_pipeline(cls.root / "original")
        cls.source = cls.original["output"]
        cls.plan = fixture_plan(cls.source)
        cls.output = cls.root / "complete"
        cls.factory = MockLV60(cls.source, cls.plan)
        with artificial_native_source(), patch.dict(os.environ, environment()):
            cls.result = observe.observe(cls.plan, cls.source, cls.output, NEW_RUN, cls.factory)
            if cls.result["status"] != "COMPLETE":
                raise AssertionError(cls.result)
            cls.packet = observe.readback(cls.source, cls.output, cls.plan, NEW_RUN)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def read(self, root, plan=None):
        with artificial_native_source():
            return observe.readback(self.source, root, plan or self.plan, NEW_RUN)

    def copied(self):
        root = self.root / self._testMethodName
        shutil.copytree(self.output, root)
        return root

    def reseal(self, root):
        (root / "FILES.json").unlink()
        observe.write(root / "FILES.json", {"schema": observe.SCHEMA + "/files", "files": observe.file_records(root)})

    def test_full_packet_original_readers_and_24_blind_forwards(self):
        self.assertEqual(len(self.factory.calls), 24)
        self.assertEqual(len(set(x[0] for x in self.factory.calls)), 24)
        self.assertEqual(self.result["actual_forward_attempts"], 24)
        self.assertEqual(self.packet["summary"]["completed_count"], 24)
        for observation, row in zip(self.packet["rows"], self.plan["panel_rows"]):
            self.assertEqual(observation["input_samples"], row["pcm_samples"])
            self.assertEqual(observation["normalized_input_sha256"], observation["xlsr_reference"]["normalized_input_sha256"])
            self.assertNotEqual(observation["run"], observation["source_run"])
            self.assertFalse(observation["training_allowed"])
        self.assertFalse({"numpy", "torch", "scipy", "transformers"} & set(sys.modules))

    def test_single_failed_forward_is_one_call_no_retry(self):
        factory = MockLV60(self.source, self.plan, fail=self.plan["panel_rows"][0]["id"])
        output = self.root / "single-failure"
        with artificial_native_source(), patch.dict(os.environ, environment()):
            result = observe.observe(self.plan, self.source, output, NEW_RUN, factory)
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["failed_count"]), (24, 23, 1))
        self.assertEqual(len(set(x[0] for x in factory.calls)), 24)
        self.assertEqual(self.read(output)["summary"]["status"], "INCOMPLETE")
        # A failed native call still requires authenticated whole prepared input.
        path = output / "rows" / self.plan["panel_rows"][0]["id"] / "normalized-input.f32"
        path.write_bytes(struct.pack("<f", 99.0) + path.read_bytes()[4:])
        self.reseal(output)
        with self.assertRaisesRegex(ValueError, "SAVED_PREPARED_ARRAY_BYTES"):
            self.read(output)

    def test_normalization_difference_blocks_forward_for_that_row(self):
        factory = MockLV60(self.source, self.plan, changed_normalization=True)
        output = self.root / "different-normalization"
        with artificial_native_source(), patch.dict(os.environ, environment()):
            result = observe.observe(self.plan, self.source, output, NEW_RUN, factory)
        self.assertEqual((result["actual_forward_attempts"], result["completed_count"], result["failed_count"]), (23, 23, 1))
        self.assertNotIn(self.plan["panel_rows"][0]["id"], [x[0] for x in factory.calls])
        self.assertEqual(self.read(output)["summary"]["actual_forward_attempts"], 23)

    def test_argmax_tamper_rejected_after_file_reseal(self):
        root = self.copied()
        path = root / "rows" / self.plan["panel_rows"][0]["id"] / "CAPTURE.json"
        value = observe.read(path)
        value["values"]["phone_tokens"] = ["oʊ"]
        path.write_text(json.dumps(value))
        self.reseal(root)
        with self.assertRaisesRegex(ValueError, "SAVED_LOGITS_ARGMAX_AND_CTC"):
            self.read(root)

    def test_native_int32_mask_cannot_be_cropped(self):
        root = self.copied()
        directory = root / "rows" / self.plan["panel_rows"][0]["id"]
        path = directory / "attention-mask.i32"
        path.write_bytes(path.read_bytes()[:-4])
        value = observe.read(directory / "CAPTURE.json")
        value["attention_mask"].update(bytes=path.stat().st_size, sha256=observe.sha(path.read_bytes()))
        (directory / "CAPTURE.json").write_text(json.dumps(value))
        self.reseal(root)
        with self.assertRaisesRegex(ValueError, "SAVED_PREPARED_ARRAY_BYTES"):
            self.read(root)

    def test_provider_identity_cannot_be_relabelled_xlsr(self):
        root = self.copied()
        path = root / "MODEL_LOCK.json"
        value = observe.read(path)
        value["model"] = observe.native.OBSERVERS["phoneme"]
        path.write_text(json.dumps(value))
        self.reseal(root)
        with self.assertRaisesRegex(ValueError, "FROZEN_LV60_NATIVE_LOCK"):
            self.read(root)

    def test_missing_terminal_keeps_24_denominator_and_unknown_count(self):
        root = self.copied()
        (root / "RESULT.json").unlink()
        (root / "FILES.json").unlink()
        summary = self.read(root)["summary"]
        self.assertEqual(summary["denominator"], 24)
        self.assertIsNone(summary["actual_forward_attempts"])
        self.assertEqual(summary["attempts_lower_bound"], 24)
        self.assertEqual(summary["attempts_upper_bound"], 24)
        missing = self.read(self.root / "missing")["summary"]
        self.assertEqual(missing["unknown_count"], 24)
        self.assertIsNone(missing["actual_forward_attempts"])

    def test_call_cap_and_immutable_panel_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["maximum_total_model_calls"] = 25
        with self.assertRaisesRegex(ValueError, "FIXED_LV60_DIAGNOSTIC"):
            observe.source_check(plan)
        plan = copy.deepcopy(self.plan)
        plan["panel_rows"][-1] = plan["panel_rows"][0]
        with self.assertRaisesRegex(ValueError, "FIXED_24_ROW_PANEL"):
            observe.source_check(plan)
        plan = copy.deepcopy(self.plan)
        plan["panel_rows"][0]["pcm_samples"] -= 1
        with artificial_native_source(), self.assertRaisesRegex(ValueError, "EXACT_ORIGINAL_PANEL_ROW"):
            observe.source_readback(self.source, plan)

    def test_original_artifact_tampering_rejected_before_import(self):
        plan = copy.deepcopy(self.plan)
        plan["source_files_manifest_sha256"] = "0" * 64
        with patch.object(observe, "_load_source_component", side_effect=AssertionError("NO_UNAUTHENTICATED_IMPORT")):
            with self.assertRaisesRegex(ValueError, "ORIGINAL_FILES_MANIFEST_PIN"):
                observe.source_readback(self.source, plan)

    def test_install_commands_are_pinned_and_no_model_calls(self):
        result = type("Process", (), {"returncode": 0})()
        with patch.object(observe.subprocess, "run", return_value=result) as execute:
            installation = observe.install(self.plan, self.root / "mock-runtime")
        self.assertEqual(installation["status"], "COMPLETE")
        self.assertEqual(installation["new_model_calls"], 0)
        self.assertEqual(execute.call_count, 4)
        all_args = [a for call in execute.call_args_list for a in call.args[0]]
        self.assertIn("torch==2.9.1+cpu", all_args)
        self.assertIn("transformers==4.46.3", all_args)


if __name__ == "__main__":
    unittest.main()
