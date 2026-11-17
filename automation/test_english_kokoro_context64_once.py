"""Stdlib/mock integration fixtures; no model, processor or resampler calls."""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_kokoro_context64_once as w
import english_kokoro_context64_generate as g
from test_english_kokoro_context64_generate import fake_rows, fixture_duration, fake_parent_bytes, VOCAB

MOCK_RUN = {"repository": "jiying2007/kws-data", "commit": "a" * 40,
    "run_id": "999999", "run_attempt": "1", "event": "push"}


def plan_fixture():
    plan = w.read(HERE / "english_kokoro_context64_plan.json")
    # Final source files can be reviewed before the coordinator refreshes pins.
    # The separate --source-check invocation validates the actual shipped pins.
    for key, path in w.CODE_PATHS.items():
        plan[key] = w.sha((w.ROOT / path).read_bytes())
    plan["recipe_sha256"] = w.digest(w.recipe_module().recipe())
    return plan


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def refresh_fixture_manifest(root):
    (root / "FILES.json").unlink(missing_ok=True)
    w.once(root / "FILES.json", w.file_manifest(root))


def text_fixture(root, *, complete_sources=True):
    source = root / "source-packet"
    source.mkdir()
    w.once(source / "source/RESULT.json", {"fixture_only": True, "run": MOCK_RUN})
    children = copy.deepcopy(g.recipe.recipe()["children"])
    for index, child in enumerate(children):
        child["generation_status"] = "success" if complete_sources or index < 2 else "not_started"
        child.update(pcm_path=None, wav_path=None, pcm_sha256=None, wav_sha256=None, pcm_samples=None)
        if child["generation_status"] == "success":
            raw = struct.pack("<" + "h" * 640, *[index * 17 + j for j in range(640)])
            child["pcm_path"] = "fixture/" + child["id"] + ".pcm"
            child["wav_path"] = "fixture/" + child["id"] + ".wav"
            wav = g.native._wav(raw, 16000)
            child.update(pcm_sha256=w.sha(raw), wav_sha256=w.sha(wav), pcm_samples=640)
            w.put(source / child["pcm_path"], raw)
            w.put(source / child["wav_path"], wav)
    return source, {"run": MOCK_RUN, "children": children}


def run_text_fixture(root, *, decode=None, once=None, complete_sources=True):
    plan = plan_fixture()
    source, manifest = text_fixture(root, complete_sources=complete_sources)
    output, assets = root / "observer", root / "private-model"
    calls = []
    def fake_decode(wav, folder):
        calls.append((wav, folder))
        return "  Hey Mero.  " if decode is None else decode(len(calls), wav, folder)
    lock = {"verified_pinned_loader": "english_seven_voice_asr_once._load", "provider": "qwen",
        "source_sha256": plan["helper_sha256"]["automation/english_seven_voice_asr_once.py"]}
    fake = SimpleNamespace(load_observer=lambda provider, model_root, supplied_plan: (fake_decode, lock))
    patch_once = mock.patch.object(w, "once", side_effect=once) if once else mock.patch.object(w, "once", wraps=w.once)
    with mock.patch.object(w, "legacy", return_value=fake), patch_once:
        result = w.observe_text("qwen", source, output, assets, plan, MOCK_RUN, manifest)
    refresh_fixture_manifest(output)
    return plan, source, manifest, output, result, calls


def full_parent_fixture(root):
    """Artificial float buffers test the format contract, never a resampler."""
    parents, children = fake_rows()
    parent = parents[0]
    duration = fixture_duration(parent)
    n = 600 * sum(duration)
    m = n * 2 // 3
    native = struct.pack("<" + "f" * n, *[((i % 31) - 15) / 32768 for i in range(n)])
    derived = struct.pack("<" + "f" * m, *[((i % 37) - 18) / 32768 for i in range(m)])
    pcm = struct.pack("<" + "h" * m, *[(i % 37) - 18 for i in range(m)])
    prefix = "source/parents/" + parent["id"] + "/"
    parent.update(w.duration_coordinates(parent, duration, n))
    values = {"native_full_f32": ("native-full.f32", native),
        "native_full_wav": ("native-full.wav", g.native._wav(w.float_pcm(native), 24000)),
        "resampled_f32": ("resampled-16k.f32", derived),
        "pcm": ("parent-16k.pcm", pcm), "wav": ("parent-16k.wav", g.native._wav(pcm, 16000))}
    for key, (name, raw) in values.items():
        parent[key + "_path"] = prefix + name
        parent[key + "_sha256"] = w.put(root / (prefix + name), raw)["sha256"]
    parent.update(pcm_samples=m, sample_rate=16000, channels=1, sample_width=2, native_sample_rate=24000,
        native_float_format="float32-little-endian", resampled_float_format="float32-little-endian",
        resampler=g.recipe.GEOMETRY["resampler"], normalization_applied=False,
        parent_16k_quantized_once=True, synthetic_silence_added=False, observed_target_pronunciation=False,
        human_gold=False, duration_path=prefix + "pred_dur.json")
    doc = {"schema": "english-kokoro-context64-generation-v1/native-duration", "campaign_id": w.CAMPAIGN_ID,
        "id": parent["id"], "phonemes": parent["raw_phonemes"], "input_ids": [0, *[VOCAB[c] for c in parent["raw_phonemes"]], 0],
        "includes_bos_eos": True, "pred_dur": duration, "samples_per_frame": 600, "sample_rate": 24000,
        "native_samples": n, "boundary_is_acoustic_gold": False, "boundary_config": g.recipe.BOUNDARIES,
        "target_start_char": 125, "target_end_char_exclusive": len(parent["raw_phonemes"])}
    parent["duration_sha256"] = w.once(root / parent["duration_path"], doc)["sha256"]
    return parent, children, duration, pcm


class SourceAndGuard(unittest.TestCase):
    def test_both_disabled_and_actual_activation_shape_source_check(self):
        plan = plan_fixture()
        plan.update(enabled=False, main_before=None)
        self.assertEqual(w.source_check(plan)["source_check"], "PASS")
        plan.update(enabled=True, main_before=w.MAIN_BEFORE)
        self.assertEqual(w.source_check(plan)["source_check"], "PASS")

    def test_fixed_design_and_budget_cannot_drift(self):
        plan = plan_fixture()
        for key, value in (("retry_count", 1), ("training_allowed", True), ("total_high_level_model_calls_max", 337)):
            changed = copy.deepcopy(plan)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed["design"]["validation_gates"]["all48_controls_candidate_positive_count"] = 1
        with self.assertRaisesRegex(ValueError, "FROZEN_PROSPECTIVE_DESIGN"):
            w.validate_plan(changed)

    def test_old_pcm_exclusion_cannot_be_removed(self):
        plan = plan_fixture()
        plan["source_exposure_exclusion"]["old_pcm_sha256"].pop()
        with self.assertRaisesRegex(ValueError, "EXACT_KNOWN_PREVIOUS_EXPOSURE"):
            w.validate_plan(plan)

    def test_exact_public_main_guard_and_no_rerun(self):
        plan = plan_fixture()
        plan.update(enabled=True, main_before=w.MAIN_BEFORE)
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": MOCK_RUN["repository"],
            "GITHUB_SHA": MOCK_RUN["commit"], "GITHUB_RUN_ID": MOCK_RUN["run_id"],
            "GITHUB_RUN_ATTEMPT": "1", "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
            "GITHUB_WORKFLOW_REF": "jiying2007/kws-data/.github/workflows/english-kokoro-context64-once.yml@refs/heads/main"}
        event = {"before": w.MAIN_BEFORE, "after": MOCK_RUN["commit"],
            "repository": {"full_name": MOCK_RUN["repository"], "private": False}}
        self.assertEqual(w.execution_guard(plan, env, event), MOCK_RUN)
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "pull_request"),
                           ("GITHUB_WORKFLOW_REF", "old-workflow"), ("GITHUB_ACTIONS", "false")):
            changed = dict(env)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.execution_guard(plan, changed, event)
        event["before"] = "b" * 40
        with self.assertRaises(ValueError):
            w.execution_guard(plan, env, event)

    def test_missing_source_retains_unknown_status_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            result, valid = w.source_status(Path(temp), plan_fixture())
        self.assertFalse(valid)
        self.assertFalse(result["source_ready"])
        self.assertFalse(result["observer_ready"])
        self.assertIsNone(result["generation_summary"]["actual_forward_attempts"])
        self.assertEqual(result["generation_summary"]["forward_attempts_upper_bound"], 48)
        self.assertIsNone(result["valid_child_count"])

    def test_missing_source_cli_writes_requested_unknown_receipt_before_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_path = root / "mock-plan.json"
            write_json(plan_path, plan_fixture())
            output = root / "SOURCE_READBACK.json"
            child = subprocess.run([sys.executable, "-I", "-S", "-B", str(HERE / "english_kokoro_context64_once.py"),
                "--plan", str(plan_path), "--read-source", "--source", str(root / "missing-source"),
                "--output", str(output)], capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 1)
            receipt = w.read(output)
            self.assertEqual(receipt["status"], "SOURCE_READBACK_FAILED_COUNTS_UNKNOWN")
            self.assertIsNone(receipt["generation_summary"]["actual_forward_attempts"])
            self.assertFalse(receipt["observer_ready"])

    def test_actual_generation_writer_failed_initialization_reads_all112_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = plan_fixture(); plan.update(enabled=True, main_before=w.MAIN_BEFORE)
            with mock.patch.object(g, "public_guard", return_value=MOCK_RUN), \
                 mock.patch.object(g, "_load_native", side_effect=RuntimeError("MOCK_NO_MODEL_INIT")):
                result = g.generate(plan, root, run=MOCK_RUN, asset_root=root / "fixture-private")
            checked = w.read_generation(root, plan, allow_partial=True)
            self.assertEqual((len(checked["parents"]), len(checked["children"])), (48, 64))
            self.assertEqual(checked["generation_summary"]["actual_forward_attempts"], 0)
            self.assertFalse(checked["source_integrity_passed"])
            self.assertEqual(result["completed_parents"], result["completed_children"])
            self.assertFalse(any(root.rglob("*.pcm")))

    def test_closed_manifest_rejects_extra_or_changed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            w.once(root / "a.json", {"fixture": True})
            w.once(root / "FILES.json", w.file_manifest(root))
            w.closed_readback(root)
            w.once(root / "extra.json", {})
            with self.assertRaisesRegex(ValueError, "EXACT_CLOSED_PACKET_FILES"):
                w.closed_readback(root)


class ByteContracts(unittest.TestCase):
    def test_pcm_quantizer_ties_clipping_and_nonfinite(self):
        values = [-2.0, -1.0, -1.5 / 32768, -0.5 / 32768, 0.5 / 32768, 1.5 / 32768, 1.0, 2.0]
        self.assertEqual(struct.unpack("<8h", w.float_pcm(struct.pack("<8f", *values))),
            (-32768, -32768, -2, 0, 0, 2, 32767, 32767))
        with self.assertRaisesRegex(ValueError, "FINITE_FLOAT32"):
            w.float_pcm(struct.pack("<f", float("nan")))

    def test_parent_format_readback_no_resampler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, _, duration, pcm = full_parent_fixture(root)
            got = w.parent_audio_readback(root, parent, VOCAB)
            self.assertEqual(got["pcm"], pcm)
            self.assertEqual(got["duration"], duration)
            self.assertFalse(any(name in sys.modules for name in ("numpy", "scipy", "torch", "kokoro")))

    def test_parent_saved_quantization_tamper_rejected_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, _, _, _ = full_parent_fixture(root)
            raw = bytearray((root / parent["pcm_path"]).read_bytes())
            raw[4] ^= 1
            (root / parent["pcm_path"]).write_bytes(raw)
            parent["pcm_sha256"] = w.sha(bytes(raw))
            with self.assertRaisesRegex(ValueError, "PARENT_INT16_SINGLE_QUANTIZATION"):
                w.parent_audio_readback(root, parent, VOCAB)

    def test_actual_child_writer_to_independent_reader_main_and_outside(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, children, duration, pcm = full_parent_fixture(root)
            parent_media = w.parent_audio_readback(root, parent, VOCAB)
            seen = {"pcm": {parent["pcm_sha256"]}}
            exposure = g.exclusion_sets(plan_fixture())
            for child in children:
                g.write_child_audio(root, parent, child, duration, pcm, exposure, seen)
                proof = w.child_audio_readback(root, child, parent, parent_media)
                self.assertEqual(proof["pcm_sha256"], child["pcm_sha256"])
                if child["kind"] == "outside":
                    self.assertLessEqual(child["child_end_parent16k"], parent["target_start_parent16k"])

    def test_exact_child_byte_or_boundary_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, children, duration, pcm = full_parent_fixture(root)
            parent_media = w.parent_audio_readback(root, parent, VOCAB)
            child = children[0]
            g.write_child_audio(root, parent, child, duration, pcm, g.exclusion_sets(plan_fixture()), {"pcm": {parent["pcm_sha256"]}})
            child["child_start_parent16k"] += 1
            with self.assertRaisesRegex(ValueError, "UNMODIFIED_CHILD_BOUNDARY_RULE"):
                w.child_audio_readback(root, child, parent, parent_media)
            child["child_start_parent16k"] -= 1
            raw = bytearray((root / child["pcm_path"]).read_bytes()); raw[0] ^= 1
            (root / child["pcm_path"]).write_bytes(raw); child["pcm_sha256"] = w.sha(bytes(raw))
            with self.assertRaisesRegex(ValueError, "CHILD_EXACT_INT16_PARENT_SUBSTRING"):
                w.child_audio_readback(root, child, parent, parent_media)


class TextProducerConsumer(unittest.TestCase):
    def test_all64_blind_calls_and_raw_text_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, source, manifest, output, result, calls = run_text_fixture(Path(temp))
            checked = w.read_text_provider(output, manifest["children"], plan, MOCK_RUN, provider="qwen")
            self.assertEqual(len(calls), 64)
            self.assertTrue(checked["complete"])
            self.assertEqual((checked["actual_forward_attempts"], checked["native_forward_returns"], checked["completed"]), (64, 64, 64))
            self.assertEqual(checked["rows"][0]["raw_text"], "  Hey Mero.  ")
            self.assertTrue(all(len(args) == 2 for args in calls))
            self.assertFalse((Path(temp) / "private-model").exists())

    def test_one_native_error_has_one_attempt_no_retry(self):
        def decode(number, wav, folder):
            if number == 1:
                raise ValueError("MOCK_NATIVE_ERROR")
            return "Hey Mero."
        with tempfile.TemporaryDirectory() as temp:
            plan, _, manifest, output, result, calls = run_text_fixture(Path(temp), decode=decode)
            checked = w.read_text_provider(output, manifest["children"], plan, MOCK_RUN, provider="qwen")
            self.assertEqual(len(calls), 64)
            self.assertEqual((checked["actual_forward_attempts"], checked["native_forward_returns"], checked["completed"]), (64, 63, 63))
            self.assertIsNone(checked["rows"][0]["raw_text"])

    def test_return_receipt_write_failure_keeps_true_native_return(self):
        real_once = w.once
        consumed = False
        def save(path, value):
            nonlocal consumed
            if Path(path).name == "FORWARD_RETURNED.json" and not consumed:
                consumed = True
                raise OSError("MOCK_RETURN_RECEIPT_WRITE")
            return real_once(path, value)
        with tempfile.TemporaryDirectory() as temp:
            plan, _, manifest, output, result, calls = run_text_fixture(Path(temp), once=save)
            checked = w.read_text_provider(output, manifest["children"], plan, MOCK_RUN, provider="qwen")
            self.assertEqual((checked["actual_forward_attempts"], checked["native_forward_returns"], checked["completed"]), (64, 64, 63))
            self.assertEqual(checked["retained_native_forward_returns_lower_bound"], 63)
            self.assertIsNone(checked["rows"][0]["raw_text"])

    def test_global_stop_preserves_all64_denominator_without_continuation(self):
        def decode(number, wav, folder):
            raise w.Deadline("MOCK_STOP")
        with tempfile.TemporaryDirectory() as temp:
            plan, _, manifest, output, result, calls = run_text_fixture(Path(temp), decode=decode)
            checked = w.read_text_provider(output, manifest["children"], plan, MOCK_RUN, provider="qwen")
            self.assertEqual(len(calls), 1)
            self.assertEqual((checked["actual_forward_attempts"], checked["native_forward_returns"], checked["completed"]), (1, 0, 0))
            self.assertEqual(len(checked["rows"]), 64)

    def test_failed_source_children_never_call_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _, manifest, output, result, calls = run_text_fixture(Path(temp), complete_sources=False)
            checked = w.read_text_provider(output, manifest["children"], plan, MOCK_RUN, provider="qwen")
            self.assertEqual(len(calls), 2)
            self.assertEqual(checked["actual_forward_attempts"], 2)
            self.assertEqual(len(checked["rows"]), 64)
            self.assertIsNone(checked["rows"][2]["raw_text"])

    def test_missing_terminal_is_bounded_unknown_even_with_retained_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _, manifest, output, result, calls = run_text_fixture(Path(temp))
            (output / "RESULT.json").unlink(); (output / "FILES.json").unlink()
            checked = w.read_text_provider(output, manifest["children"], plan, MOCK_RUN, provider="qwen")
            self.assertIsNone(checked["actual_forward_attempts"])
            self.assertIsNone(checked["native_forward_returns"])
            self.assertEqual((checked["forward_attempts_lower_bound"], checked["forward_attempts_upper_bound"]), (64, 64))
            self.assertTrue(all(row["raw_text"] is None for row in checked["rows"]))

    def test_capture_cannot_switch_to_parent_scope_or_another_child(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _, manifest, output, result, calls = run_text_fixture(Path(temp))
            path = output / manifest["children"][0]["id"] / "CAPTURE.json"
            original = w.read(path)
            for key, value in (("input_scope", "parent_full_16k_pcm"), ("parent_id", "not-the-parent"),
                               ("source_pcm_sha256", "0" * 64), ("target_conditioned", True)):
                changed = dict(original); changed[key] = value; write_json(path, changed)
                refresh_fixture_manifest(output)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "COMPLETE_CHILD_TEXT_CAPTURE"):
                    w.read_text_provider(output, manifest["children"], plan, MOCK_RUN, provider="qwen")
            write_json(path, original)

    def test_extra_input_slot_cannot_hide_behind_closed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, _, manifest, output, result, calls = run_text_fixture(Path(temp))
            w.once(output / "unregistered-child/CAPTURE.json", {"fixture": True})
            refresh_fixture_manifest(output)
            with self.assertRaisesRegex(ValueError, "ONLY_REGISTERED_CHILD_TEXT_EVIDENCE"):
                w.read_text_provider(output, manifest["children"], plan, MOCK_RUN, provider="qwen")

    def test_missing_provider_preserves_unknown64(self):
        with tempfile.TemporaryDirectory() as temp:
            source, manifest = text_fixture(Path(temp))
            checked = w.read_text_provider(Path(temp) / "absent", manifest["children"], plan_fixture(), MOCK_RUN, provider="parakeet")
            self.assertEqual(checked["provider"], "parakeet")
            self.assertIsNone(checked["actual_forward_attempts"])
            self.assertEqual((checked["forward_attempts_lower_bound"], checked["forward_attempts_upper_bound"]), (0, 64))
            self.assertEqual(len(checked["rows"]), 64)


if __name__ == "__main__":
    unittest.main()
