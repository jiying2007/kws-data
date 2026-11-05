#!/usr/bin/env python3
"""Real bulk file formats with fictional PCM/ASR fixtures; zero model calls."""
from __future__ import annotations

from collections import Counter
import copy
import io
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_weak_view as weak
import research_weak_sampling as sampling
import test_active_asr_weak_view as weak_fixture

sys.path.insert(0, str(weak.ROOT / "research/auto-tts-voice-clone-20261102"))
import bulk_pool as pool
import bulk_runner
import test_bulk as bulk_fixture


class FictionalClone:
    def synthesize(self, cell, ref, root):
        frequency = 0.01 + int(cell["cell_id"][-6:], 16) / 16777216 * .03
        return [0.2 * math.sin(index * frequency) for index in range(24000)], 24000


def fixture_writer(folder, values, rate, metrics):
    native = weak_fixture.float_wav(struct.pack("<24000f", *values))
    # Fixture interpolation only, never represented as actual model resampling.
    pcm = struct.pack("<16000h", *(round(values[index * 3 // 2] * 32767) for index in range(16000)))
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(pcm)
    wav = target.getvalue()
    (folder / "native-f32.wav").write_bytes(native)
    (folder / "audio-16k.wav").write_bytes(wav)
    return {"native_sha256": weak.sha(native), "native_bytes": len(native),
            "wav_sha256": weak.sha(wav), "wav_bytes": len(wav), "pcm_sha256": weak.sha(pcm),
            "audio_frames": 16000, "sample_rate_hz": 16000}


class D1WeakViewTests(unittest.TestCase):
    asr_fixture = weak_fixture.WeakPackageTests.asr_fixture

    def setUp(self):
        fixture = bulk_fixture.Fixture()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.sources = fixture.d1_sources()
        self.plan = pool.plan(self.sources, self.root, repetitions=2, batch_id="fictional-d1",
                              revision="b" * 40, layout=pool.D1_LAYOUT)
        self.plan_sha = weak.sha(weak.canonical(self.plan))
        self.plan_path = self.root / "frozen-plan.json"
        weak.save_json(self.root, self.plan_path.name, self.plan)
        self.generation = self.root / "generation"
        self.result = bulk_runner.run(self.plan, self.sources, self.root, self.generation,
                                      FictionalClone(), max_new_calls=24, writer=fixture_writer)
        self.inputs, self.asr = self.root / "inputs", self.root / "active-asr"
        self.inputs.mkdir()
        self.asr.mkdir()
        self.archive = self.root / "native-only.zip"
        clips = []
        with zipfile.ZipFile(self.archive, "w") as archive:
            for number, cell in enumerate(self.plan["cells"][:24]):
                identifier = f"clip-{number:03d}"
                member = "cells/" + cell["cell_id"] + "/native-f32.wav"
                raw = (self.generation / member).read_bytes()
                pcm = weak.float_wav_pcm(raw)
                values = [v[0] for v in struct.iter_unpack("<f", pcm)]
                derived = struct.pack("<16000f", *(values[index * 3 // 2] for index in range(16000)))
                clip = {"id": identifier, "original_member": member,
                        "original_wav": {"path": "original/" + identifier + ".wav", "sha256": weak.sha(raw), "bytes": len(raw)},
                        "original_pcm": {"sample_rate": 24000, "channels": 1, "frames": 24000,
                                         "dtype": "<f4", "sha256": weak.sha(pcm)},
                        "derived_pcm": {"path": "pcm16k/" + identifier + ".f32", "sample_rate": 16000,
                                        "channels": 1, "frames": 16000, "dtype": "<f4", "sha256": weak.sha(derived)}}
                clips.append(clip)
                archive.writestr(member, raw)
                weak.save(self.inputs, clip["original_wav"]["path"], raw)
                weak.save(self.inputs, clip["derived_pcm"]["path"], derived)
        self.manifest = {"schema": "active-asr-inputs-v1", "source_archive_sha256": weak.sha(self.archive.read_bytes()),
                         "derivation": {"fixture_only": True}, "clips": clips}
        weak.save_json(self.inputs, "inputs.json", self.manifest)
        self.input_sha = weak.sha((self.inputs / "inputs.json").read_bytes())
        for clip, cell in zip(clips, self.plan["cells"]):
            for model in weak.MODELS:
                self.asr_fixture(clip, model, cell["prompt"])
        chars = list(dict.fromkeys("".join(cell["prompt"] for cell in self.plan["cells"])))
        chars += [chr(0x3400 + i) for i in range(2598 - len(chars))]
        raw = ("<blk> 1\n" + "".join(f"{char} {index}\n" for index, char in enumerate(chars, 2))).encode()
        self.tokens = self.root / "tokens.txt"
        self.tokens.write_bytes(raw)
        patcher = patch.object(weak, "FSMN_TOKENS_SHA", weak.sha(raw))
        patcher.start()
        self.addCleanup(patcher.stop)

    def join(self, *, manifest=None, plan=None, plan_sha=None):
        return weak.d1_source_rows(self.archive.read_bytes(), manifest or self.manifest,
                                   weak.canonical(plan or self.plan), plan_sha or self.plan_sha, self.generation)

    def create(self):
        output = self.root / "view"
        report = weak.create_view(self.inputs / "inputs.json", self.input_sha, self.archive,
                                  self.asr, output, fsmn_tokens=self.tokens,
                                  d1_plan=self.plan_path, d1_plan_sha256=self.plan_sha,
                                  d1_generation_root=self.generation)
        return output, report

    def completed(self, number=0):
        return self.generation / "cells" / self.plan["cells"][number]["cell_id"] / "completed.json"

    def rehash_package(self, output):
        manifest = {path.relative_to(output).as_posix(): weak.sha(path.read_bytes())
                    for path in output.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
        (output / "MANIFEST.json").write_bytes(weak.canonical(manifest) + b"\n")

    def test_real_runner_format_join_separates_roles_and_connects_sampling(self):
        before = {path: path.read_bytes() for path in self.generation.rglob("*") if path.is_file()}
        output, report = self.create()
        self.assertEqual(self.result["total_completed"], 24)
        self.assertEqual(report["statistics"]["train_only"], 18)
        self.assertEqual(report["statistics"]["development"], 6)
        self.assertEqual(report["statistics"]["machine_candidates"], 24)
        train = list(weak.load_training_rows(output))
        development = list(weak.load_development_rows(output))
        self.assertEqual((len(train), len(development)), (18, 6))
        self.assertEqual(len({r["source_family_id"] for r in train}), 6)
        self.assertEqual(len({r["source_family_id"] for r in development}), 2)
        self.assertTrue({r["source_family_id"] for r in train}.isdisjoint(
            r["source_family_id"] for r in development))
        self.assertTrue(all(r["source_plan_sha256"] == self.plan_sha for r in [*train, *development]))
        self.assertEqual(Counter(r["source"]["generation_category"] for r in report["items"]),
                         {"full-positive": 12, "ordinary-negative": 6, "partial-negative": 3, "near-negative": 3})
        rnn = list(weak.load_training_rows(output, "rnn-four-token"))
        self.assertTrue(all(r["generation_category"] in ("full-positive", "partial-negative") for r in rnn))
        partial = [r for r in rnn if r["generation_category"] == "partial-negative"]
        self.assertTrue(partial)
        self.assertTrue(all(r["target_ids"] == [3, 4] and r["keyword_targets"] == [0, 0] for r in partial))
        positive_draws = []
        for percent in (0, 5, 10):
            sample = sampling.exposure_plan(train, total_exposures=100, full_positive_exposures=50, near_percent=percent)
            self.assertEqual(sample["counts"]["near-negative"], percent)
            positive_draws.append([r for r in sample["draws"] if r["bucket"] == "full-positive"])
        self.assertEqual(positive_draws[0], positive_draws[1])
        self.assertEqual(positive_draws[1], positive_draws[2])
        with self.assertRaisesRegex(ValueError, "RESEARCH_TRAIN_ROWS_REQUIRED"):
            sampling.exposure_plan(development, total_exposures=100, full_positive_exposures=50, near_percent=0)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_plan_pin_changed_or_missing_category_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "D1_FROZEN_PLAN_CHANGED"):
            self.join(plan_sha="a" * 64)
        changed = copy.deepcopy(self.plan)
        del changed["cells"][0]["generation_category"]
        with self.assertRaisesRegex(ValueError, "CELL_PLAN_CHANGED"):
            self.join(plan=changed, plan_sha=weak.sha(weak.canonical(changed)))

    def test_missing_started_and_wrong_reference_binding_are_rejected(self):
        path = self.completed().with_name("started.json")
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.join()
        changed = weak.loads(original)
        changed["reference_wav_sha256"] = "a" * 64
        path.write_bytes(weak.canonical(changed))
        with self.assertRaisesRegex(ValueError, "D1_STARTED_PLAN_JOIN"):
            self.join()

    def test_duplicate_input_or_changed_source_pcm_is_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed["clips"][1] = copy.deepcopy(changed["clips"][0])
        with self.assertRaisesRegex(ValueError, "D1_SOURCE_AUDIO_ROSTER"):
            self.join(manifest=changed)
        changed = copy.deepcopy(self.manifest)
        changed["clips"][0]["original_pcm"]["sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "D1_NATIVE_WAV_PCM_JOIN"):
            self.join(manifest=changed)

    def test_completed_native_or_derived_audio_drift_is_rejected(self):
        path = self.completed()
        original = path.read_bytes()
        changed = weak.loads(original)
        changed["audio"]["native_sha256"] = "a" * 64
        path.write_bytes(weak.canonical(changed))
        with self.assertRaisesRegex(ValueError, "D1_NATIVE_WAV_PCM_JOIN"):
            self.join()
        path.write_bytes(original)
        wav = path.with_name("audio-16k.wav")
        raw = wav.read_bytes()
        wav.write_bytes(raw[:-2] + b"\0\0")
        with self.assertRaisesRegex(ValueError, "D1_GENERATED_WAV_PCM_JOIN"):
            self.join()

    def test_source_ancestor_cannot_cross_roles(self):
        output, _ = self.create()
        ref = self.plan["references"][0]
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK"):
            list(weak.load_training_rows(output, other_split_sources=[
                {"role": "development-only", "wav_sha256": ref["parent_source_sha256"]}]))

    def test_rehashed_category_forgery_is_rejected(self):
        output, _ = self.create()
        path = output / "fsmn-fulltext-train.jsonl"
        rows = [weak.loads(line) for line in path.read_bytes().splitlines()]
        partial = next(row for row in rows if row["generation_category"] == "partial-negative")
        partial["generation_category"] = "near-negative"
        path.write_bytes(b"".join(weak.canonical(row) + b"\n" for row in rows))
        self.rehash_package(output)
        with self.assertRaisesRegex(ValueError, "WEAK_TRAIN_ROWS_CHANGED"):
            list(weak.load_training_rows(output))

    def test_unknown_category_is_not_guessed_as_negative(self):
        output, _ = self.create()
        train = list(weak.load_training_rows(output))
        next(row for row in train if row["generation_category"] == "partial-negative")["generation_category"] = "unknown"
        with self.assertRaisesRegex(ValueError, "UNKNOWN_GENERATION_CATEGORY"):
            sampling.exposure_plan(train, total_exposures=100, full_positive_exposures=50, near_percent=0)


if __name__ == "__main__":
    unittest.main()
