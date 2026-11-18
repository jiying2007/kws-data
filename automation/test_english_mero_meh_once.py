"""Offline cross-module fixtures; all waveforms and model outputs are artificial."""
from __future__ import annotations

import ast
import base64
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_mero_meh_generate as generate
import english_mero_meh_observe as observe
import english_mero_meh_once as once
from test_english_mero_meh_generate import MockBackend, MOCK_RUN, MOCK_VOCAB


def fixture_plan():
    plan = once.read_json(once.PLAN)
    names = set(observe.native.HELPER_PINS) | set(generate.LEGACY_PINS)
    names |= {"automation/english_pocket12_once.py", "automation/english_pocket12_observe.py",
              "automation/test_english_pocket12_observe.py", "EXCLUSION_EVIDENCE.json", once.WORKFLOW,
              "research/english_mero_meh_20261118/PRONUNCIATION.md",
              "automation/english_mero_meh_profile_v1.json"}
    names |= {"automation/english_mero_meh_" + x + ".py" for x in ("once", "generate", "observe", "recipe", "policy")}
    names |= {"automation/test_english_mero_meh_" + x + ".py" for x in ("once", "generate", "policy")}
    plan.update(enabled=False, main_before=None,
                source_files_sha256={name: once.digest(once.ROOT / name) for name in sorted(names)})
    return plan


def phone_metadata():
    # Reuse only the five immutable small model metadata fixtures, by parsing
    # their literal. No historical test, observation, audio or model is run.
    tree = ast.parse((HERE / "test_english_pocket12_observe.py").read_text())
    value = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "FIXED_METADATA_B64" for target in node.targets))
    return json.loads(zlib.decompress(base64.b64decode(value)))


class MockObservers:
    def __init__(self, source, fail=None):
        self.source, self.fail, self.calls = Path(source), fail, []
        self.rows = generate.read(self.source / "CORPUS_MANIFEST.json")["rows"]
        self.paths = {(self.source / row["wav_path"]).absolute(): row for row in self.rows}
        self.tokens = {}
        for row in self.rows:
            cell = row["cell_id"]
            if cell.startswith("mero-"):
                phones = ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"]
            elif cell.startswith("vosen-"):
                phones = ["h", "eɪ", "v", "oʊ", "s", "ə", "n"]
            elif cell == "nero":
                phones = ["h", "eɪ", "n", "ɛ", "ɹ", "oʊ"]
            elif cell == "memo":
                phones = ["h", "eɪ", "m", "ɛ", "m", "oʊ"]
            elif cell.startswith("bosen-"):
                phones = ["h", "eɪ", "b", "oʊ", cell[-1], "ə", "n"]
            else:
                phones = ["h", "ə", "l", "oʊ"]
            self.tokens[row["id"]] = phones

    def _forward(self, provider, row):
        self.calls.append((provider, row["id"], row["pcm_sha256"]))
        if self.fail == (provider, row["id"]):
            raise RuntimeError("ARTIFICIAL_SINGLE_FORWARD_FAILURE")
        return row

    def __call__(self, provider, assets, output):
        Path(assets).mkdir()
        if provider != "phoneme":
            def decode(wav, directory):
                return self._forward(provider, self.paths[Path(wav).absolute()])["requested_text"]
            return decode, {"observer": observe.identity(provider),
                "runtime_versions": observe.native.TEXT_RUNTIME[provider],
                "adapter_sha256": observe.native.HELPER_PINS["automation/english_seven_voice_asr_once.py"],
                "initialization_model_calls": 0, "target_conditioned": False}
        metadata = Path(output) / "model-metadata"
        metadata.mkdir()
        for name, encoded in phone_metadata().items():
            (metadata / name).write_bytes(base64.b64decode(encoded))
        vocab = json.loads((metadata / "vocab.json").read_bytes())
        helper, cursor = observe.native.native_module(), [0]
        def prepare(pcm, geometry):
            row = self.rows[cursor[0]]
            cursor[0] += 1
            if set(geometry) != {"pcm_samples", "full_to_crop_offset_16k"} or observe.sha(pcm) != row["pcm_sha256"]:
                raise ValueError("BLIND_COMPLETE_PCM_ONLY")
            n = geometry["pcm_samples"]
            values = [v[0] / 32768 for v in struct.iter_unpack("<h", pcm)]
            mean = math.fsum(values) / n
            scale = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / n + 1e-7)
            normalized = struct.pack("<" + "f" * n, *((v - mean) / scale for v in values))
            facts = {"sample_rate": 16000, "pcm_samples": n, "normalized_shape": [1, n],
                "normalized_dtype": "torch.float32", "attention_mask_shape": [1, n],
                "attention_mask_dtype": "torch.int32", "attention_mask_sum": n, "attention_mask_unique_values": [1],
                "valid_input_samples": n, "padded_input_samples": n, "right_padding_samples": 0}
            return row, facts, normalized, struct.pack("<i", 1) * n
        def forward(row):
            return self._forward(provider, row)
        def materialize(row, geometry):
            frames = helper.frame_geometry(geometry["pcm_samples"], 0)["frames"]
            tokens = self.tokens[row["id"]]
            ids = [vocab[token] for token in tokens]
            ids += [0] * (frames - len(ids))
            raw = b"".join(struct.pack("<392f", *(1.0 if i == peak else -1.0 for i in range(392))) for peak in ids)
            return {"shape": [1, frames, 392], "dtype": "float32-little-endian", "order": "C",
                    "frame_ids": ids, "phone_tokens": tokens,
                    "argmax_special_frame_counts": {str(k): ids.count(k) for k in (0, 1, 2, 3)}}, raw
        lock = {"model": helper.OBSERVERS[provider], "files": helper.OBSERVER_FILES[provider],
                "runtime_versions": helper.RUNTIME_VERSIONS, "native_forward_attempts_at_initialization": 0,
                "model_parameters_modified": False, "target_conditioned": False}
        return prepare, forward, materialize, lock


def mock_pipeline(root, plan=None):
    """Return a complete artificial packet for downstream saved-byte tests."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = fixture_plan() if plan is None else plan
    source, backends = root / "source", []
    def factory(output, assets):
        backend = MockBackend(output, assets)
        backends.append(backend)
        return backend
    source_result = generate.generate(plan, source, root / "tts-assets", MOCK_RUN, backend_factory=factory)
    observers, output = {}, root / "combined"
    fake = MockObservers(source)
    with patch.object(generate, "verify_native_lock", return_value=MOCK_VOCAB), \
         patch.object(once, "execution_identity", return_value=MOCK_RUN), \
         patch.object(observe, "_load", side_effect=fake), \
         patch("urllib.request.urlopen", side_effect=AssertionError("NO_NETWORK_IN_MOCK")):
        for provider in observe.PROVIDERS:
            observers[provider] = root / provider
            result = observe.run(provider=provider, source_root=source, output_root=observers[provider],
                asset_root=root / (provider + "-assets"), plan=plan, run_identity=MOCK_RUN)
            if result["status"] != "COMPLETE":
                raise AssertionError(result)
        combined_result = once.combine(source, observers, output, plan, MOCK_RUN)
    return {"output": output, "source": source, "observers": observers, "plan": plan,
            "source_result": source_result, "combined_result": combined_result,
            "tts_calls": list(backends[0].calls), "observer_calls": list(fake.calls)}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fixture = mock_pipeline(cls.root)
        cls.plan = cls.fixture["plan"]
        cls.native_patch = patch.object(generate, "verify_native_lock", return_value=MOCK_VOCAB)
        cls.native_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.native_patch.stop()
        cls.temp.cleanup()

    def test_complete_144_pipeline_and_720_calls(self):
        f, output = self.fixture, self.fixture["output"]
        self.assertEqual(len(f["tts_calls"]), 144)
        self.assertEqual(len(f["observer_calls"]), 576)
        self.assertEqual(len(set((p, row) for p, row, _ in f["observer_calls"])), 576)
        self.assertEqual(f["combined_result"]["actual_total_model_calls"], 720)
        corpus, valid = generate.validate_source(output, MOCK_RUN)
        self.assertEqual((len(corpus["rows"]), len(valid)), (144, 144))
        by_id = {r["id"]: r["pcm_sha256"] for r in valid}
        self.assertTrue(all(pcm == by_id[row] for _, row, pcm in f["observer_calls"]))
        evidence = once.read_json(output / "MACHINE_EVIDENCE.json")
        self.assertEqual(len(evidence["rows"]), 144)
        self.assertTrue(all(set(row["observations"]) == set(observe.PROVIDERS) for row in evidence["rows"]))
        admission = once.read_json(output / "ADMISSION.json")
        self.assertEqual(admission["status"], "PASS_FRESH_CORPUS_ADMISSION")
        self.assertEqual(admission["balanced_groups_by_role"], {"train": 6, "development": 2, "calibration": 2, "locked-test": 2})
        self.assertEqual(once.read_json(output / "FILES.json"),
                         {"schema": generate.FILES_SCHEMA, "files": generate.file_records(output)})
        self.assertTrue((output / "frozen_source" / once.WORKFLOW).is_file())
        self.assertFalse({"torch", "numpy", "scipy", "transformers", "kokoro", "qwen_asr"} & set(sys.modules))

    def test_missing_observer_stays_unknown_with_all_144_rows(self):
        roots = self.fixture["observers"] | {"qwen": self.root / "missing-qwen"}
        output = self.root / "missing-combined"
        result = once.combine(self.fixture["source"], roots, output, self.plan, MOCK_RUN)
        evidence = once.read_json(output / "MACHINE_EVIDENCE.json")
        self.assertIsNone(result["actual_total_model_calls"])
        self.assertEqual(evidence["actual_total_model_calls_lower_bound"], 576)
        self.assertEqual(evidence["observer_summaries"]["qwen"]["unknown_count"], 144)
        self.assertEqual(len(evidence["rows"]), 144)
        admission = once.read_json(output / "ADMISSION.json")
        self.assertFalse(admission["training_allowed"])
        self.assertEqual(admission["tier_counts"], {"unknown": 144})

    def test_missing_source_keeps_full_denominator_and_stops_admission(self):
        output = self.root / "missing-source-combined"
        roots = {p: self.root / ("absent-" + p) for p in observe.PROVIDERS}
        result = once.combine(self.root / "absent-source", roots, output, self.plan, MOCK_RUN)
        self.assertIsNone(result["actual_total_model_calls"])
        self.assertFalse(result["training_allowed"])
        self.assertEqual(len(once.read_json(output / "MACHINE_EVIDENCE.json")["rows"]), 144)
        self.assertEqual(len(once.read_json(output / "ADMISSION.json")["rows"]), 144)
        self.assertEqual(once.read_json(output / "CORPUS_MANIFEST.json")["status"], "UNVERIFIED_SOURCE")

    def test_missing_terminal_does_not_claim_zero_calls(self):
        root = self.root / "partial-qwen"
        shutil.copytree(self.fixture["observers"]["qwen"], root)
        (root / "RESULT.json").unlink()
        (root / "FILES.json").unlink()
        result = observe.readback(provider="qwen", source_root=self.fixture["source"], observer_root=root,
                                  plan=self.plan, run_identity=MOCK_RUN)
        self.assertIsNone(result["summary"]["actual_forward_attempts"])
        self.assertEqual(result["summary"]["attempts_lower_bound"], 144)
        self.assertEqual(result["summary"]["denominator"], 144)

    def test_failed_forward_is_one_attempt_without_retry(self):
        row_id = generate.recipe.rows()[0]["id"]
        fake = MockObservers(self.fixture["source"], fail=("qwen", row_id))
        output = self.root / "failed-qwen"
        with patch.object(once, "execution_identity", return_value=MOCK_RUN), patch.object(observe, "_load", side_effect=fake):
            result = observe.run(provider="qwen", source_root=self.fixture["source"], output_root=output,
                asset_root=self.root / "failed-assets", plan=self.plan, run_identity=MOCK_RUN)
        readback = observe.readback(provider="qwen", source_root=self.fixture["source"], observer_root=output,
                                    plan=self.plan, run_identity=MOCK_RUN)
        self.assertEqual((result["actual_forward_attempts"], result["native_forward_returns"], result["completed_count"]),
                         (144, 143, 143))
        self.assertEqual(readback["summary"]["failed_count"], 1)
        self.assertEqual(len(fake.calls), 144)

    def test_native_logit_claim_tamper_rejected_after_reseal(self):
        root = self.root / "tampered-phoneme"
        shutil.copytree(self.fixture["observers"]["phoneme"], root)
        row_id = generate.recipe.rows()[0]["id"]
        path = root / "rows" / row_id / "CAPTURE.json"
        capture = once.read_json(path)
        capture["values"]["frame_ids"][0] = 0
        once.write_json(path, capture)
        once.write_json(root / "FILES.json", {"schema": observe.SCHEMA + "/files",
                        "files": observe.file_records(root), "self_excluded": "FILES.json"})
        with self.assertRaisesRegex(ValueError, "SAVED_LOGITS_ARGMAX_AND_CTC"):
            observe.readback(provider="phoneme", source_root=self.fixture["source"], observer_root=root,
                             plan=self.plan, run_identity=MOCK_RUN)


class GuardTests(unittest.TestCase):
    def test_exact_main_first_attempt_and_disabled_defaults(self):
        plan = fixture_plan()
        self.assertFalse(plan["enabled"])
        with self.assertRaisesRegex(ValueError, "DISABLED"):
            once.execution_identity(plan)
        plan.update(enabled=True, main_before="b" * 40)
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": once.REPOSITORY, "GITHUB_EVENT_NAME": "push",
               "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": MOCK_RUN["run_id"],
               "GITHUB_SHA": MOCK_RUN["commit"],
               "GITHUB_WORKFLOW_REF": once.REPOSITORY + "/" + once.WORKFLOW + "@refs/heads/main"}
        event = {"repository": {"full_name": once.REPOSITORY, "private": False}, "ref": "refs/heads/main",
                 "before": plan["main_before"], "after": MOCK_RUN["commit"], "forced": False, "deleted": False}
        kwargs = {"environ": env, "event": event, "head": MOCK_RUN["commit"], "parent": plan["main_before"]}
        self.assertEqual(once.execution_identity(plan, **kwargs), MOCK_RUN)
        with self.assertRaisesRegex(ValueError, "FIRST_PUBLIC"):
            once.execution_identity(plan, **(kwargs | {"environ": env | {"GITHUB_RUN_ATTEMPT": "2"}}))
        with self.assertRaisesRegex(ValueError, "CHECKOUT_AND_PARENT"):
            once.execution_identity(plan, **(kwargs | {"parent": "c" * 40}))
        workflow = (once.ROOT / once.WORKFLOW).read_text()
        self.assertNotIn("workflow_dispatch", workflow)
        self.assertIn("include-hidden-files: true", workflow)


if __name__ == "__main__":
    unittest.main()
