#!/usr/bin/env python3
"""Fictional source/native fixtures only; no model or network calls."""
from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_weak_view as weak


def votes(texts):
    return [{"model": model, "usable": True, "text": text,
             "normalized_text": weak.normalize(text)} for model, text in zip(weak.MODELS, texts)]


def source(kind="positive-intent", keyword=1):
    return {"intended_kind": kind, "intended_keyword_id": keyword}


def float_wav(pcm):
    fmt = struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


class WeakDecisionTests(unittest.TestCase):
    def test_complete_accented_wake_survives_third_decoder_near_character(self):
        result = weak.decide(source(), votes(["你好，小窝。", "你好小窝", "你好小屋"]))
        self.assertEqual(result["status"], "machine_candidate")
        self.assertEqual(result["keyword_targets"], [1, 0])
        self.assertEqual(result["full_text"], "你好小窝")
        self.assertEqual(result["selected_raw_text"], "你好，小窝。")
        self.assertEqual(result["ctc_weight"], 0.5)
        self.assertTrue(result["disagreement"])
        self.assertFalse(result["human_gold"])
        self.assertIsNone(result["event_times"])

    def test_positive_intent_never_becomes_negative_from_near_character_agreement(self):
        result = weak.decide(source(), votes(["你好小屋", "你好小屋", "你好小吴"]))
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["full_text"], "你好小屋")
        self.assertFalse(result["ctc_mask"])
        self.assertEqual(result["keyword_targets"], [None, None])
        self.assertEqual(result["keyword_mask"], [False, False])

    def test_partial_xiaowo_is_negative_and_keeps_nonblank_full_transcript(self):
        result = weak.decide(source("partial-intent", None), votes(["小窝", "小窝。", "小窝"]))
        self.assertEqual(result["keyword_targets"], [0, 0])
        self.assertEqual(result["full_text"], "小窝")
        self.assertEqual(result["ctc_weight"], 1.0)
        targets = weak.consumer_targets(result["full_text"], weak.FOUR_CHAR_IDS, 16000, frame_stride=2)
        self.assertEqual(targets["target_ids"], [3, 4])

    def test_keyword_subsequence_cannot_enter_negative_loss(self):
        for text in ("你好今天小窝", "你好小窝"):
            with self.subTest(text=text):
                result = weak.decide(source("near-miss-intent", None), votes(["你好小屋", "你好小屋", text]))
                self.assertEqual(result["status"], "uncertain")
                self.assertFalse(any(result["keyword_mask"]))

    def test_incomplete_third_wake_observation_still_protects_negative_label(self):
        observed = votes(["小窝", "小窝", "你好小窝"])
        observed[2].update(usable=False, status="complete", quality_flags=["incomplete"])
        result = weak.decide(source("partial-intent", None), observed)
        self.assertEqual(result["status"], "uncertain")
        self.assertTrue(result["disagreement"])
        self.assertEqual(result["keyword_mask"], [False, False])

    def test_entire_sentence_is_required_and_retained(self):
        text = "你好小窝，请打开客厅的灯。"
        result = weak.decide(source(), votes([text, text, "你好小窝"]))
        self.assertEqual(result["full_text"], "你好小窝请打开客厅的灯")
        unsupported = weak.consumer_targets(result["full_text"], weak.FOUR_CHAR_IDS, 16000, frame_stride=2)
        self.assertFalse(unsupported["supported"])
        self.assertIsNone(unsupported["target_ids"])
        divergent = weak.decide(source(), votes([text, "你好小窝，请关灯。", "你好小窝，请开门。"]))
        self.assertEqual(divergent["status"], "uncertain")

    def test_empty_incomplete_and_bad_signal_are_not_negative_labels(self):
        observations = votes(["你好小窝", "你好小窝", ""])
        observations[1]["usable"] = False
        observations[2]["usable"] = False
        self.assertFalse(weak.decide(source(), observations)["ctc_mask"])
        self.assertFalse(weak.decide(source(), votes(["你好小窝"] * 3), ["clipped"])["ctc_mask"])
        self.assertEqual(weak.decide(source("speech-negative-intent", None), votes([""] * 3))["status"],
                         "uncertain")

    def test_same_engine_cannot_vote_twice(self):
        rows = votes(["你好小窝"] * 3)
        rows[1]["model"] = rows[0]["model"]
        with self.assertRaisesRegex(ValueError, "ASR_VOTE_MODEL_ROSTER"):
            weak.decide(source(), rows)

    def test_english_boundaries_and_non_target_phonemes_are_not_erased(self):
        self.assertNotEqual(weak.normalize("the rapist"), weak.normalize("therapist"))
        self.assertNotEqual(weak.normalize("we're ready"), weak.normalize("were ready"))
        self.assertNotEqual(weak.normalize("小屋"), weak.normalize("小窝"))
        self.assertIsNone(weak.consumer_targets("小屋", weak.FOUR_CHAR_IDS, 16000, frame_stride=2)["target_ids"])

    def test_quantization_records_clipping_and_rejects_nonfinite_audio(self):
        wav, receipt = weak.quantize_pcm16(struct.pack("<400f", *([0.25] * 399 + [1.25])), 400)
        self.assertEqual(receipt["clipped_samples"], 1)
        self.assertEqual(receipt["quality_flags"], ["quantization_saturation"])
        self.assertEqual(receipt["frames"], 400)
        self.assertEqual(weak.sha(wav), receipt["file_sha256"])
        with self.assertRaisesRegex(ValueError, "NONFINITE_DERIVED_AUDIO"):
            weak.quantize_pcm16(struct.pack("<400f", *([float("nan")] * 400)), 400)


class WeakPackageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.asr = self.root / "asr"
        self.asr.mkdir()
        self.specs = [
            ("你好小窝", "positive-intent", 1, ["你好，小窝。", "你好小窝", "你好小屋"]),
            ("小窝", "partial-intent", None, ["小窝", "小窝", "小窝"]),
            ("请打开屋里的灯", "speech-negative-intent", None, ["请打开屋里的灯"] * 3),
            ("你好小窝", "positive-intent", 1, ["你好小屋", "你好小屋", "你好小吴"]),
        ]
        clips, planned, outcomes, wavs = [], [], [], {}
        for number, (text, kind, keyword, _) in enumerate(self.specs):
            identifier = f"clip-{number:03d}"
            cell_id = f"fictional-{number:03d}"
            member = "lane-0/output/" + cell_id + ".raw-float.wav"
            original = struct.pack("<24000f", *(0.2 * math.sin(i * (0.01 + number * .001)) for i in range(24000)))
            derived = struct.pack("<16000f", *(0.2 * math.sin(i * (0.015 + number * .0015)) for i in range(16000)))
            raw = float_wav(original)
            wavs[member] = raw
            clip = {"id": identifier, "original_member": member,
                    "original_wav": {"path": f"original/{identifier}.wav", "sha256": weak.sha(raw), "bytes": len(raw)},
                    "original_pcm": {"sample_rate": 24000, "channels": 1, "frames": 24000, "dtype": "<f4", "sha256": weak.sha(original)},
                    "derived_pcm": {"path": f"pcm16k/{identifier}.f32", "sample_rate": 16000,
                                    "channels": 1, "frames": 16000, "dtype": "<f4", "sha256": weak.sha(derived)}}
            clips.append(clip)
            weak.save(self.inputs, clip["original_wav"]["path"], raw)
            weak.save(self.inputs, clip["derived_pcm"]["path"], derived)
            row = {"cell_id": cell_id, "text": text, "speaker": "Vivian", "intended_kind": kind,
                   "intended_keyword_id": keyword, "intended_split": "train_only",
                   "label_authority": "requested_text_only", "intended_label_tier": "machine_weak_label",
                   "training_admitted": False}
            planned.append(row)
            outcomes.append({**row, "status": "generated", "raw_wav_sha256": weak.sha(raw),
                             "native": {"source_values_sha256": weak.sha(original), "frames": 24000},
                             "metrics": {"quality_flags": [], "technically_valid": True}})
        generation = {"campaign": "fictional-test-only", "source_kind": "synthetic-public-preset",
                      "human_gold": False, "training_admitted": False, "planned": planned, "outcomes": outcomes}
        runtime = {"campaign": generation["campaign"], "training_admitted": False,
                   "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "model_revision": "b" * 40}
        self.archive = self.root / "original.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, raw in wavs.items():
                archive.writestr(name, raw)
            archive.writestr("lane-0/output/generation.json", weak.canonical(generation))
            archive.writestr("lane-0/output/runtime-verification.json", weak.canonical(runtime))
        self.manifest = {"schema": "active-asr-inputs-v1", "source_archive_sha256": weak.sha(self.archive.read_bytes()),
                         "derivation": {"fixture_only": True}, "clips": clips}
        weak.save_json(self.inputs, "inputs.json", self.manifest)
        self.input_sha = weak.sha((self.inputs / "inputs.json").read_bytes())
        for clip, (_, _, _, texts) in zip(clips, self.specs):
            for model, text in zip(weak.MODELS, texts):
                self.asr_fixture(clip, model, text)
        # A complete fictitious 2599-class vocabulary, not a copied model artifact.
        chars = list(dict.fromkeys("你好小窝屋请打开里的灯吴"))
        chars += [chr(0x3400 + i) for i in range(2598 - len(chars))]
        self.tokens = self.root / "tokens.txt"
        raw = ("<blk> 1\n" + "".join(f"{char} {index}\n" for index, char in enumerate(chars, 2))).encode()
        self.tokens.write_bytes(raw)
        patcher = patch.object(weak, "FSMN_TOKENS_SHA", weak.sha(raw))
        patcher.start()
        self.addCleanup(patcher.stop)

    def asr_fixture(self, clip, model, text):
        config = {"qwen06": {"context": "", "force_language": None},
                  "sensevoice": {"language": "auto", "use_itn": False},
                  "whisper": {"prompt": None, "prefix": None, "language": "zh"}}[model]
        identity = {"model": model, "decoder_config": config, "input_manifest_sha256": self.input_sha,
                    "recipe": "active-asr-native-v1", "model_assets": {"fixture_only": True},
                    "runtime_lock_sha256": {"fictional": "a" * 64}, "runtime_packages_sha256": "c" * 64}
        identity_sha = weak.sha(weak.canonical({"model_identity": identity, "input": clip}))
        common = {"model": model, "clip_id": clip["id"], "identity_sha256": identity_sha}
        if model == "qwen06":
            payload = {"decoded_skip_special_tokens": "language Chinese<asr_text>" + text,
                       "metadata": {"eos_terminated": True}, "token_ids": [1000, 151645]}
        elif model == "sensevoice":
            payload = {"result": [{"key": clip["id"], "text": "<|zh|>" + text}]}
        else:
            payload = {"text": text, "decoder_terminated": True, "pre_finalize_greedy_trace": [{"completed": True}]}
        native = {"schema": "active-asr-native-output-v1", **common, "native": payload}
        path = "native/" + model + "/" + clip["id"] + ".json"
        weak.save_json(self.asr, path, native)
        claim = {"schema": "active-asr-call-claim-v1", **common, "input": clip,
                 "model_identity": identity, "attempt": "fixture-001"}
        weak.save_json(self.asr, "claims/" + model + "/" + clip["id"] + ".json", claim)
        result = {"schema": "active-asr-result-v1", **common, "input": clip, "model_identity": identity,
                  "status": "complete", "attempt": "fixture-001", "text": text, "quality_flags": [],
                  "decoder_terminated": True, "native_output_sha256": weak.sha((self.asr / path).read_bytes()),
                  "decoder_source_sha256": "d" * 64,
                  "boundary": {**dict.fromkeys(("human_gold", "training_admitted", "shipping_approved",
                                                "target_text_received", "peer_transcript_received"), False),
                               "acoustic_verbatim_completeness": "unknown"}}
        weak.save_json(self.asr, "results/" + model + "/" + clip["id"] + ".json", result)

    def create(self, name="view", **kwargs):
        output = self.root / name
        report = weak.create_view(self.inputs / "inputs.json", self.input_sha, self.archive,
                                  self.asr, output, fsmn_tokens=kwargs.get("tokens", self.tokens))
        return output, report

    def test_private_end_to_end_fulltext_rnn_and_split_boundary(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        output, report = self.create()
        self.assertEqual(report["statistics"]["source_items"], 4)
        self.assertEqual(report["statistics"]["completed_asr_results"], 12)
        self.assertEqual(report["statistics"]["machine_candidates"], 3)
        self.assertEqual(report["statistics"]["development"], 0)
        rnn = list(weak.load_training_rows(output, "rnn-four-token"))
        full = list(weak.load_training_rows(output, "fsmn-fulltext"))
        self.assertEqual(len(rnn), 2)
        self.assertEqual(len(full), 3)
        self.assertTrue(all(row["audio"] == row["audio_path"] and Path(row["audio"]).is_absolute()
                            for row in [*rnn, *full]))
        self.assertEqual(full[2]["full_text"], "请打开屋里的灯")
        self.assertTrue(all(row["target_ids"] and all(t > 0 for t in row["target_ids"]) for row in full))
        self.assertEqual(rnn[1]["keyword_targets"], [0, 0])
        self.assertEqual(rnn[1]["target_ids"], [3, 4])
        self.assertEqual(rnn[0]["ctc_weight"], .5)
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)
        self.assertEqual((output / "evidence/source-original.zip").read_bytes(), self.archive.read_bytes())
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK"):
            list(weak.load_training_rows(output, other_split_sources=[
                {"role": "development-only", "source_family_id": "qwen3-preset-vivian"}]))
        with self.assertRaisesRegex(ValueError, "NEW_PRIVATE_VIEW_REQUIRED"):
            self.create()

    def test_missing_result_preserves_other_evidence_and_does_not_block_ready_rows(self):
        (self.asr / "results/whisper/clip-001.json").unlink()
        output, report = self.create()
        self.assertEqual(report["items"][1]["asr"][2]["status"], "missing")
        self.assertTrue((output / "evidence/asr/native/whisper/clip-001.json").is_file())
        self.assertEqual(len(list(weak.load_training_rows(output))), 3)

    def test_all_uncertain_view_has_valid_empty_training_manifests(self):
        for path in (self.asr / "results").rglob("*.json"):
            path.unlink()
        output, report = self.create()
        self.assertEqual(report["statistics"]["machine_candidates"], 0)
        self.assertEqual(report["statistics"]["uncertain"], 4)
        self.assertEqual(list(weak.load_training_rows(output)), [])
        self.assertEqual(list(weak.load_training_rows(output, "rnn-four-token")), [])

    def test_native_hash_drift_is_retained_and_rejected_per_observation(self):
        path = self.asr / "native/sensevoice/clip-000.json"
        path.write_bytes(path.read_bytes() + b"\n")
        output, report = self.create()
        self.assertEqual(report["items"][0]["decision"]["status"], "uncertain")
        self.assertEqual(report["items"][0]["asr"][1]["status"], "rejected")
        self.assertEqual((output / "evidence/asr/native/sensevoice/clip-000.json").read_bytes(), path.read_bytes())
        self.assertEqual(len(list(weak.load_training_rows(output))), 2)

    def test_report_edit_is_rejected_even_if_package_hash_is_recalculated(self):
        output, report = self.create()
        report["items"][0]["decision"]["ctc_weight"] = 1.0
        path = output / "report.json"
        path.write_bytes(weak.canonical(report) + b"\n")
        manifest = json.loads((output / "MANIFEST.json").read_bytes())
        manifest["report.json"] = weak.sha(path.read_bytes())
        (output / "MANIFEST.json").write_bytes(weak.canonical(manifest))
        with self.assertRaisesRegex(ValueError, "WEAK_VIEW_DIFFERS_FROM_RETAINED_EVIDENCE"):
            list(weak.load_training_rows(output))

    def test_unavailable_fulltext_vocabulary_is_explicit(self):
        output, report = self.create(tokens=None)
        self.assertFalse(report["items"][0]["consumers"]["fsmn-fulltext"]["supported"])
        self.assertEqual(len(list(weak.load_training_rows(output, "rnn-four-token"))), 2)
        with self.assertRaisesRegex(ValueError, "FSMN_VOCABULARY_NOT_SUPPLIED"):
            list(weak.load_training_rows(output))


if __name__ == "__main__":
    unittest.main()
