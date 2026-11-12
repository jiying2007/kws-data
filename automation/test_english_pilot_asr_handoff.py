"""No-model handoff tests: one-shot English WAVs -> independently pinned blind ASR.

All audio and native transcripts are invented fixtures, not scientific evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_blind_asr as blind
import english_qwen_capture as english
import english_qwen_real_once as pilot
import multilingual_campaign as market
from test_multilingual_campaign import (
    english_preset_request, fake_asr_assets, FakeEnglishBackend, fixture_wave_writer,
)


class PilotToNativeAsrTests(unittest.TestCase):
    def setup(self, root: Path):
        source = pilot.pilot_plan("a" * 64)
        source_file = root / "source-plan.json"
        source_file.write_bytes(market.canonical(source) + b"\n")
        input_request, adapter_file, model_root = fake_asr_assets(
            english_preset_request(), root)
        asr_pins_path = root / "asr-pins.json"
        asr_pins_path.write_bytes(market.canonical({
            "schema": blind.PILOT_PINS_SCHEMA,
            "providers": input_request["asr_providers"]}) + b"\n")
        return source, source_file, asr_pins_path, adapter_file, model_root

    def test_explicit_immutable_pilot_source_is_accepted_without_forged_asr_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, source, pins, adapters, model_root = self.setup(root)
            self.assertEqual(blind.load_pilot_plan(source), plan)
            self.assertNotIn("asr_pins", plan)
            selected = blind.load_pilot_asr_pins(pins)
            self.assertEqual({x["provider_id"] for x in selected}, set(blind.NAMES))
            adapter_set = blind.verify_adapters(
                plan, adapters, model_root, asr_pins=selected)
            self.assertEqual(set(adapter_set), set(blind.NAMES))
            self.assertFalse(plan["training_admitted"])

    def test_changed_original_pilot_source_is_rejected_before_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, source, pins, adapters, model_root = self.setup(root)
            cases = [
                ("text", lambda row: row["cells"][0].update(text_requested="Hey Nero")),
                ("word-list", lambda row: row.update(wake_phrases_requested=["Hey Mero", "Hey Vosen"])),
                ("source-family", lambda row: row["cells"][0].update(source_family_id="fake-parent")),
                ("train-role", lambda row: row["cells"][0].update(role="development-only")),
            ]
            for description, change in cases:
                with self.subTest(description=description):
                    altered = copy.deepcopy(plan)
                    change(altered)
                    source.write_bytes(market.canonical(altered) + b"\n")
                    with self.assertRaisesRegex(ValueError, "PILOT_SOURCE_PLAN_DRIFT"):
                        blind.load_pilot_plan(source)
            source.write_bytes(market.canonical(plan) + b"\n")
            bad = market.read(pins)
            bad["providers"][0]["revision_sha"] = "main"
            pins.write_bytes(market.canonical(bad) + b"\n")
            with self.assertRaisesRegex(ValueError, "MODEL_REVISION_NOT_PINNED"):
                blind.load_pilot_asr_pins(pins)

    def test_genuine_source_hash_and_blind_three_model_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, source, pins, adapter_path, model_root = self.setup(root)
            selected = blind.load_pilot_asr_pins(pins)
            adapters = blind.verify_adapters(plan, adapter_path,
                                             model_root, asr_pins=selected)
            capture_root = root / "real-capture-fixture"
            english.capture(plan, capture_root, FakeEnglishBackend(), max_calls=1,
                            writer=fixture_wave_writer)
            item = plan["cells"][0]
            real_wav = capture_root / "cells" / item["cell_id"] / "audio-16k.wav"
            self.assertIsNotNone(blind.verify_capture(plan, capture_root, item))
            by_name = {r["provider_id"]: r for r in selected}
            seen = []

            def native_stub(argv, **kwargs):
                self.assertEqual([argv[1], argv[3], argv[5]],
                                 ["--audio", "--model-dir", "--output"])
                self.assertNotIn("Hey", " ".join(argv))
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                provider = Path(argv[0]).name
                with wave.open(argv[2], "rb") as stream:
                    frames = stream.getnframes()
                with open(argv[-1], "w", encoding="utf-8") as stream:
                    json.dump({
                        "schema": blind.RAW_SCHEMA, "provider_id": provider,
                        "model_id": by_name[provider]["model_id"],
                        "model_revision_sha": by_name[provider]["model_revision_sha"],
                        "audio_wav_sha256": hashlib.sha256(Path(argv[2]).read_bytes()).hexdigest(),
                        "processed_start_frame": 0, "processed_end_frame": frames,
                        "decoder_completed": True, "actual_text": "Hey Vosen",
                    }, stream)
                seen.append(provider)
                return type("FakeNoModelReturn", (), {"returncode": 0})()

            out = root / "asr-results"
            adapter_sha = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
            result = blind.dispatch(plan, capture_root, out, adapters,
                                    adapter_sha, max_calls=3,
                                    asr_pins=selected, executor=native_stub)
            self.assertEqual(result["asr_attempts_this_call"], 3)
            self.assertEqual(set(seen), set(blind.NAMES))
            screened = blind.machine_screen(plan, out, item)
            self.assertTrue(screened["three_models_responded"])
            self.assertFalse(screened["training_admitted"])
            self.assertFalse(screened["actual_label_authenticated"])
            again = blind.dispatch(plan, capture_root, out, adapters,
                                   adapter_sha, max_calls=3,
                                   asr_pins=selected, executor=native_stub)
            self.assertEqual(again["asr_attempts_this_call"], 0)
            with open(real_wav, "ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "CAPTURE_WAV_SHA_DRIFT"):
                blind.dispatch(plan, capture_root, out, adapters,
                               adapter_sha, max_calls=1,
                               asr_pins=selected, executor=native_stub)
            self.assertEqual(len(seen), 3)

    def test_cli_requires_disjoint_generic_or_pilot_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, source, pins, adapters, model_root = self.setup(root)
            self.assertEqual(blind.main(["--pilot-source-plan", str(source),
                                         "--pilot-asr-pins", str(pins)]), 0)
            self.assertEqual(blind.main(["--pilot-source-plan", str(source)]), 2)
            self.assertEqual(blind.main([]), 2)
            self.assertEqual(blind.main(["--request", str(source),
                                         "--pilot-source-plan", str(source),
                                         "--pilot-asr-pins", str(pins)]), 2)


if __name__ == "__main__":
    unittest.main()
