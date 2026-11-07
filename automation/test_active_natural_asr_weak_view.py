#!/usr/bin/env python3
"""Natural-view counterexamples using fictional PCM/native receipts only."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_natural_asr_weak_view as natural
from test_selective_asr_weak_view import policy_bytes

weak = natural.weak


def encoded(value):
    return weak.canonical(value) + b"\n"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def waveform(values):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(16000)
        writer.writeframes(struct.pack("<" + "h" * len(values), *values))
    return stream.getvalue()


def observations(texts, *, bad_model=None, quality=False, incomplete=False):
    return [{"model": model, "status": "complete",
             "usable": model != bad_model,
             "text": text, "normalized_text": weak.normalize(text),
             "decoder_terminated": not (incomplete and model == bad_model),
             "quality_flags": ["fixture-warning"] if quality and model == bad_model else []}
            for model, text in zip(weak.MODELS, texts)]


class NaturalFixture:
    """Small format-level fixture; no model/audio corpus or pipeline checkout."""

    def __init__(self, root, *, batch_id="batch-a", family="family-a",
                 session="session-a", split="development", shift=0, texts=None):
        self.root = Path(root); self.root.mkdir(parents=True)
        self.bundle, self.asr = self.root / "bundle", self.root / "asr"
        self.bundle.mkdir(); self.asr.mkdir()
        self.policy = self.root / "policy.json"; self.policy.write_bytes(policy_bytes())
        values = [100 + shift, -100, 300, -300, 50, -50, 1, -1] * 4
        parent_wav = waveform(values)
        self.parent = {"parent_id": session, "audio_path": session + ".wav",
            "wav_sha256": weak.sha(parent_wav),
            "pcm_sha256": weak.sha(struct.pack("<32h", *values)),
            "sample_rate_hz": 16000, "sample_count": 32, "dataset_id": "fictional",
            "source_partition": "eval" if split == "development" else "train",
            "session_id": session, "source_family_id": family, "split": split,
            "provenance": {"source_archive_sha256": None, "source_member": None,
                "source_wav_sha256": None, "channel_index": None, "source_audit_sha256": None,
                "annotation_sha256": None}}
        windows = [{"window_id": session + "-left", "parent_id": session, "start_sample": 0, "end_sample": 16},
                   {"window_id": session + "-right", "parent_id": session, "start_sample": 8, "end_sample": 24}]
        self.declaration = {"schema": "kws-natural-asr-window-declaration-v1",
            "status": "declared_not_materialized", "selection_reason": "Fictional deterministic sample ranges",
            "parents": [self.parent], "windows": windows}
        declaration_raw = encoded(self.declaration)
        self.plan_sha = weak.sha(declaration_raw)
        (self.bundle / "declaration.json").write_bytes(declaration_raw)
        prep_raw = b"# Fictional preparation identity; no model or real corpus.\n"
        (self.bundle / "preparation-source.py").write_bytes(prep_raw)
        clips, map_rows, bindings = [], [], []
        with zipfile.ZipFile(self.bundle / "window-audio.zip", "w") as archive:
            for index, window in enumerate(windows):
                clip_id = f"clip-{index:03d}"
                samples = values[window["start_sample"]:window["end_sample"]]
                wav = waveform(samples)
                pcm = struct.pack("<16h", *samples)
                floating = struct.pack("<16f", *(value / 32768 for value in samples))
                original, derived = f"original/{clip_id}.wav", f"pcm16k/{clip_id}.f32"
                for name, body in ((original, wav), (derived, floating)):
                    path = self.bundle / name; path.parent.mkdir(exist_ok=True); path.write_bytes(body)
                archive.writestr(original, wav)
                clips.append({"id": clip_id, "original_member": original,
                    "original_wav": {"path": original, **natural.record(wav)},
                    "original_pcm": {"sha256": weak.sha(pcm), "sample_rate": 16000,
                                     "channels": 1, "frames": 16, "dtype": "<i2"},
                    "derived_pcm": {"path": derived, "sha256": weak.sha(floating),
                                    "sample_rate": 16000, "channels": 1, "frames": 16, "dtype": "<f4"}})
                source_id = "natural-window-" + weak.sha(encoded({
                    "parent_pcm_sha256": self.parent["pcm_sha256"], "sample_rate_hz": 16000,
                    "start_sample": window["start_sample"], "end_sample": window["end_sample"]}))
                map_rows.append({"clip_id": clip_id, "source_id": source_id,
                    "original_wav_sha256": weak.sha(wav), "original_pcm_sha256": weak.sha(pcm),
                    "derived_pcm_sha256": weak.sha(floating), "quality_uncertain": True,
                    "quality_reasons": ["unreviewed-natural-audio-window"]})
                bindings.append({"clip_id": clip_id, "source_id": source_id, **window,
                    **{key: self.parent[key] for key in ("dataset_id", "session_id", "source_family_id", "split")},
                    "parent_wav_sha256": self.parent["wav_sha256"], "parent_pcm_sha256": self.parent["pcm_sha256"],
                    "window_wav_sha256": weak.sha(wav), "window_pcm_sha256": weak.sha(pcm),
                    "derived_pcm_sha256": weak.sha(floating)})
        archive_sha = weak.sha((self.bundle / "window-audio.zip").read_bytes())
        self.inputs = {"schema": "active-asr-inputs-v1", "created_at": "2026-11-07T00:00:00+00:00",
            "source_archive_sha256": archive_sha, "preparation_source_sha256": weak.sha(prep_raw),
            "derivation": natural.DERIVATION, "clips": clips}
        write_json(self.bundle / "inputs.json", self.inputs)
        self.inputs_sha = weak.sha((self.bundle / "inputs.json").read_bytes())
        self.mapping = {"schema": "active-asr-source-map-v2", "input_manifest_sha256": self.inputs_sha,
            "source_archive_sha256": archive_sha, "plan_sha256": self.plan_sha, "rows": map_rows}
        write_json(self.bundle / "source-map.json", self.mapping)
        coverage = {"sample_rate_hz": 16000, "window_input_samples": 32,
            "unique_parent_samples": 24, "overlap_samples": 8,
            "window_input_seconds": 32/16000, "unique_parent_seconds": 24/16000,
            "overlap_seconds": 8/16000, "by_parent": [{
                **{key: self.parent[key] for key in ("parent_id", "dataset_id", "session_id", "source_family_id", "split")},
                "union_sample_ranges": [[0, 24]], "window_input_samples": 32,
                "unique_parent_samples": 24, "overlap_samples": 8}],
            "semantics": "union within each parent PCM; not independent or verified negative exposure"}
        binding = {"schema": "kws-natural-asr-window-bindings-v1", "declaration_sha256": self.plan_sha,
            "inputs_manifest_sha256": self.inputs_sha,
            "source_map_sha256": weak.sha((self.bundle / "source-map.json").read_bytes()),
            "window_archive_sha256": archive_sha,
            "preparation_sources": {"tools/prepare_natural_asr_windows.py": natural.record(prep_raw)},
            "not_decoder_input": True, "target_text_included": False,
            "parent_identity_scope": "full parent WAV and decoded PCM checked against declaration",
            "upstream_provenance_scope": "retained declaration only; upstream archives and annotations not read",
            "parents": [self.parent], "rows": bindings, "coverage": coverage, "boundary": natural.INPUT_BOUNDARY}
        write_json(self.bundle / "natural-window-bindings.json", binding)
        write_json(self.bundle / "summary.json", {"schema": "kws-natural-asr-window-summary-v1",
            "status": "materialized_unlabelled_inputs", "declaration_sha256": self.plan_sha,
            "parents": 1, "windows": 2, "coverage": coverage,
            "bindings_sha256": weak.sha((self.bundle / "natural-window-bindings.json").read_bytes()),
            "boundary": natural.INPUT_BOUNDARY})
        self.boundary = {**dict.fromkeys(("human_gold", "training_admitted", "shipping_approved",
                                        "target_text_received", "peer_transcript_received"), False),
                         "acoustic_verbatim_completeness": "unknown"}
        self.texts = texts or [("你好，小窝。",) * 3,
                              ("请打开客厅的灯", "请打开客厅的灯", "请打开客厅灯")]
        self.worker_sha = "d" * 64
        identities = {}
        for clip, utterances in zip(clips, self.texts):
            for model, text in zip(weak.MODELS, utterances):
                identities[model] = self.native(clip, model, text)
        self.request = {"schema": "active-asr-selective-request-v2", "created_at": "2026-11-07T00:00:00+00:00",
            "runtime": "/fictional/no-model-runtime", "output": str(self.asr),
            "inputs": {"path": str(self.bundle / "inputs.json"), "sha256": self.inputs_sha},
            "source_map": {"path": str(self.bundle / "source-map.json"),
                           "sha256": weak.sha((self.bundle / "source-map.json").read_bytes())},
            "policy": {"path": str(self.policy), "sha256": weak.SELECTIVE_POLICY_SHA},
            "threads": 4, "budget_seconds": 600, "max_calls": 6,
            "worker_sha256": self.worker_sha, "source_files": {"tools/active_asr_worker.py": self.worker_sha},
            "model_identities": identities, "boundary": self.boundary}
        self.write("request.json", self.request)
        self.write("job.json", {"schema": "active-asr-selective-job-v2",
            "deadline_unix": 9999999999,
            "identity": {"request_sha256": weak.sha(encoded(self.request)), "max_calls": 6,
                         "heavy_model_concurrency": 1, "threads": 4},
            "supervision": {"process_id": 123, "process_group": 123, "inherited_lease": True, "supervised": True}})
        self.write("network-guard.json", {"seccomp_installed": True, "no_new_privileges": True, "fixture_only": True})
        self.write("attempts/selective-001/summary.json", {
            "status": "complete", "finished_at": "2026-11-07T00:00:01+00:00"})
        for model in weak.MODELS:
            self.write("attempts/selective-001/" + model + ".json", {
                "schema": "active-asr-worker-v1", "model": model, "status": "complete",
                "attempt": "selective-001", "identity": identities[model], "source_sha256": self.worker_sha,
                "boundary": self.boundary, "selected": [c["id"] for c in clips],
                "new_calls": 2, "reused": 0, "unresolved_prior_calls": [],
                "heavy_lock": {"inherited": True, "supervised": True, "process_group": 123}})
            self.write("attempts/selective-001/" + model + "-load.json", {"fixture_only": True})
        (self.asr / "attempts/selective-001/quiet.log").write_bytes(b"")
        self.batch_id = batch_id
        self.seal_fixture()

    def write(self, name, value):
        write_json(self.asr / name, value)

    def native(self, clip, model, text, *, quality=None, terminated=True):
        config = {"qwen06": {"context": "", "force_language": None},
                  "sensevoice": {"language": "auto", "use_itn": False},
                  "whisper": {"prompt": None, "prefix": None, "language": "zh"}}[model]
        identity = {"model": model, "decoder_config": config, "input_manifest_sha256": self.inputs_sha,
            "recipe": "active-asr-native-v1", "model_assets": {"fixture_only": True},
            "runtime_lock_sha256": {"fictional": "a" * 64}, "runtime_packages_sha256": "c" * 64}
        common = {"model": model, "clip_id": clip["id"],
            "identity_sha256": weak.sha(weak.canonical({"model_identity": identity, "input": clip}))}
        if model == "qwen06":
            payload = {"decoded_skip_special_tokens": "language Chinese<asr_text>" + text,
                "metadata": {"eos_terminated": terminated}, "token_ids": [1000, 151645 if terminated else 42]}
        elif model == "sensevoice":
            payload = {"result": [{"key": clip["id"], "text": "<|zh|>" + text}]}
        else:
            payload = {"text": text, "decoder_terminated": terminated,
                       "pre_finalize_greedy_trace": [{"completed": terminated}]}
        self.write(f"native/{model}/{clip['id']}.json",
                   {"schema": "active-asr-native-output-v1", **common, "native": payload})
        self.write(f"claims/{model}/{clip['id']}.json", {"schema": "active-asr-call-claim-v1",
            **common, "input": clip, "model_identity": identity, "attempt": "selective-001"})
        self.write(f"results/{model}/{clip['id']}.json", {"schema": "active-asr-result-v1",
            **common, "input": clip, "model_identity": identity, "status": "complete",
            "attempt": "selective-001", "text": text, "quality_flags": quality or [],
            "decoder_terminated": terminated,
            "native_output_sha256": weak.sha((self.asr / f"native/{model}/{clip['id']}.json").read_bytes()),
            "decoder_source_sha256": self.worker_sha, "boundary": self.boundary})
        return identity

    def seal_fixture(self):
        """Only fictional fixture writes use this helper; production has no seal callback."""
        routes = []
        for clip, source in zip(self.inputs["clips"], self.mapping["rows"]):
            primary = {model: weak.observation(self.asr, model, clip, self.inputs_sha, {})
                       for model in weak.MODELS[:2]}
            routes.append(weak.selective_route(source, self.plan_sha, primary))
        routing = {"schema": "active-asr-selective-routing-v2", "input_manifest_sha256": self.inputs_sha,
            "source_map_sha256": self.request["source_map"]["sha256"], "policy_sha256": weak.SELECTIVE_POLICY_SHA,
            "plan_sha256": self.plan_sha, "decoder_source_sha256": self.worker_sha, "items": routes}
        self.write("routing.json", routing)
        files = {"output/" + p.relative_to(self.asr).as_posix(): natural.record(p.read_bytes())
                 for p in self.asr.rglob("*") if p.is_file() and "audit" not in p.relative_to(self.asr).parts}
        files.update({"inputs/" + p.relative_to(self.bundle).as_posix(): natural.record(p.read_bytes())
                      for p in self.bundle.rglob("*") if p.is_file()})
        files.update({"metadata/source_map.json": natural.record((self.bundle / "source-map.json").read_bytes()),
                      "metadata/policy.json": natural.record(self.policy.read_bytes())})
        self.write("audit/file-hashes.json", {"schema": "active-asr-selective-inventory-v2",
            "files": files, "file_count": len(files), "files_sha256": weak.sha(weak.canonical(files))})
        pairs = sorted((model, clip["id"]) for model in weak.MODELS for clip in self.inputs["clips"])
        self.audit = {"schema": "active-asr-selective-final-audit-v2", "created_at": "2026-11-07T00:00:02+00:00",
            "request_sha256": weak.sha((self.asr / "request.json").read_bytes()),
            "inventory_sha256": weak.sha((self.asr / "audit/file-hashes.json").read_bytes()),
            "state": {"status": "complete", "input_manifest_sha256": self.inputs_sha,
                "source_map_sha256": self.request["source_map"]["sha256"], "policy_sha256": weak.SELECTIVE_POLICY_SHA,
                "routing_sha256": weak.sha((self.asr / "routing.json").read_bytes()),
                "decoder_source_sha256": self.worker_sha, "inputs": 2,
                "requested_matrix": [list(p) for p in pairs], "not_requested_matrix": [],
                "expected_matrix_cells": 6, "verified_matrix_cells": 6, "original_recognition_calls": 6,
                "call_budget": 6, "duplicate_model_clip_pairs": 0,
                "models": {model: {"requested": 2, "not_requested": 0, "verified": 2,
                                  "claims": 2, "worker_new_calls": 2} for model in weak.MODELS},
                "errors": [], "finished_before_deadline": True, "boundary": self.boundary,
                "unrequested_is_not_a_negative_or_agreement": True}}
        self.write("audit/final-summary.json", self.audit)

    def entry(self):
        digest = lambda p: weak.sha(p.read_bytes())
        return {"batch_id": self.batch_id, "bundle": str(self.bundle),
            "declaration_sha256": self.plan_sha,
            "bindings_sha256": digest(self.bundle / "natural-window-bindings.json"),
            "asr_output": str(self.asr), "asr_request_sha256": digest(self.asr / "request.json"),
            "final_audit_sha256": digest(self.asr / "audit/final-summary.json"),
            "policy": {"path": str(self.policy), **natural.record(self.policy.read_bytes())}}


class NaturalDecisionTests(unittest.TestCase):
    def test_three_whole_sentences_preserve_nonempty_text(self):
        decision, _ = natural.decide(observations(["请打开灯。", "请打开灯", "请 打 开 灯"]))
        self.assertEqual((decision["ctc_mask"], decision["ctc_weight"], decision["full_text"]),
                         (True, 1.0, "请打开灯"))
        self.assertEqual(decision["keyword_targets"], [0, 0])
        self.assertIsNone(decision["event_times"])

    def test_two_full_votes_require_three_keyword_presence_vectors(self):
        decision, _ = natural.decide(observations(["你好小窝请开灯", "你好小窝请开灯", "你好小窝打开灯"]))
        self.assertEqual(decision["ctc_weight"], 0.5)
        self.assertEqual(decision["keyword_targets"], [1, 0])
        decision, _ = natural.decide(observations(["你好小窝请开灯", "你好小窝请开灯", "你好小屋请开灯"]))
        self.assertEqual(decision["ctc_weight"], 0)
        self.assertEqual(decision["keyword_targets"], [None, None])

    def test_three_absent_targets_without_fulltext_quorum_are_not_negatives(self):
        decision, evidence = natural.decide(observations(["请开灯", "今天很好", "我们回家"]))
        self.assertTrue(all(row["three_usable_literal_presence_agree"] for row in evidence))
        self.assertFalse(decision["full_text_quorum"])
        self.assertEqual(decision["keyword_mask"], [False, False])

    def test_ordered_and_near_paths_remain_diagnostics_under_frozen_rule(self):
        texts = ["你好这个小窝真好", "你好这个小窝真好", "你好那个小窝挺好"]
        decision, evidence = natural.decide(observations(texts))
        self.assertTrue(any(row["any_ordered_nonliteral_path"] for row in evidence))
        self.assertEqual(decision["ctc_weight"], 0.5)
        self.assertEqual(decision["keyword_targets"], [0, 0])
        decision, _ = natural.decide(observations([texts[0]] * 3))
        self.assertEqual(decision["ctc_weight"], 1)
        # Near words alone are not an unregistered general rejection gate.
        decision, _ = natural.decide(observations(["门口有个小屋", "门口有个小屋", "门前有个小屋"]))
        self.assertEqual(decision["ctc_weight"], 0.5)
        self.assertIn("屋", decision["full_text"])

    def test_quality_incomplete_and_empty_third_never_become_two_clean_votes(self):
        for changes in ({"quality": True}, {"incomplete": True}, {}):
            with self.subTest(changes=changes):
                decision, _ = natural.decide(observations(["你好小窝"] * 3, bad_model="whisper", **changes))
                self.assertFalse(decision["ctc_mask"])
        decision, _ = natural.decide(observations(["你好小窝", "你好小窝", " 。"]))
        self.assertFalse(decision["ctc_mask"])


class NaturalViewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.fixture = NaturalFixture(self.root / "original")
        self.output = self.root / "view"

    def create(self, entries=None):
        request = self.root / "request.json"
        write_json(request, {"schema": natural.REQUEST_SCHEMA, "batches": entries or [self.fixture.entry()]})
        return natural.create_view(request, weak.sha(request.read_bytes()), self.output)

    def verify(self):
        return natural.verify_view(self.output,
            expected_manifest_sha256=weak.sha((self.output / "MANIFEST.json").read_bytes()))

    def rehash(self):
        files = {p.relative_to(self.output).as_posix(): natural.record(p.read_bytes())
                 for p in self.output.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
        write_json(self.output / "MANIFEST.json", {"schema": natural.MANIFEST_SCHEMA, "files": files,
            "file_count": len(files), "files_sha256": weak.sha(weak.canonical(files))})

    def test_real_format_replay_without_original_paths_preserves_all_observations(self):
        before = {p: p.read_bytes() for p in self.fixture.root.rglob("*") if p.is_file()}
        report = self.create()
        self.assertEqual(report["statistics"]["machine_weak_three"], 1)
        self.assertEqual(report["statistics"]["machine_weak_two"], 1)
        self.assertTrue(all(r["split"] == "development" for r in report["rows"]))
        self.assertEqual(report["rows"][1]["observations"][2]["text"], "请打开客厅灯")
        self.assertEqual(report["rows"][1]["decision"]["full_text"], "请打开客厅的灯")
        self.assertEqual(report["boundary"]["accepted_negative_seconds"], 0)
        self.assertEqual(report["decision_rules"], natural.DECISION_RULES)
        self.assertFalse(report["boundary"]["training_admitted"])
        self.assertFalse(any(self.output.glob("*train*.jsonl")))
        self.assertEqual(before, {p: p.read_bytes() for p in self.fixture.root.rglob("*") if p.is_file()})
        shutil.rmtree(self.fixture.root)
        saved = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(report, self.verify())
        self.assertEqual(saved, {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()})

    def test_missing_final_seal_is_not_written_by_create(self):
        entry = self.fixture.entry()
        seal = self.fixture.asr / "audit/final-summary.json"
        seal.unlink()
        with self.assertRaises(FileNotFoundError):
            self.create([entry])
        self.assertFalse(seal.exists()); self.assertFalse(self.output.exists())

    def test_output_inside_input_evidence_is_rejected_before_any_write(self):
        before = {p: p.read_bytes() for p in self.fixture.root.rglob("*") if p.is_file()}
        for output in (self.fixture.bundle / "mistaken-output", self.fixture.asr / "mistaken-output",
                       self.fixture.asr / "../asr/mistaken-output"):
            with self.subTest(output=output):
                self.output = output
                with self.assertRaisesRegex(ValueError, "OUTPUT_INSIDE_READ_ONLY_EVIDENCE"):
                    self.create()
                self.assertFalse(self.output.exists())
                self.assertEqual(before, {p: p.read_bytes() for p in self.fixture.root.rglob("*") if p.is_file()})

    def test_recipe_and_external_manifest_pins_are_required(self):
        entry = self.fixture.entry(); entry["asr_request_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "EXISTING_SEAL_AND_RECIPE"):
            self.create([entry])
        self.assertFalse(self.output.exists())
        self.create()
        with self.assertRaisesRegex(ValueError, "MANIFEST_PIN"):
            natural.verify_view(self.output, expected_manifest_sha256="0" * 64)

    def test_forged_report_even_with_new_manifest_is_rejected_by_native_replay(self):
        report = self.create()
        report["rows"][1]["decision"]["ctc_weight"] = 1.0
        write_json(self.output / "report.json", report); self.rehash()
        with self.assertRaisesRegex(ValueError, "REPORT_DIFFERS_FROM_NATIVE_EVIDENCE"):
            self.verify()

    def test_missing_requested_native_is_not_unknown_or_auto_resealed(self):
        (self.fixture.asr / "native/whisper/clip-000.json").unlink()
        self.fixture.seal_fixture()
        with self.assertRaisesRegex(ValueError, "SELECTIVE_REQUESTED_NATIVE_RESULT"):
            self.create()

    def test_native_text_mismatch_survives_inventory_reseal(self):
        path = self.fixture.asr / "native/qwen06/clip-000.json"
        value = weak.loads(path.read_bytes())
        value["native"]["decoded_skip_special_tokens"] = "language Chinese<asr_text>不同内容"
        write_json(path, value)
        result_path = self.fixture.asr / "results/qwen06/clip-000.json"
        result = weak.loads(result_path.read_bytes()); result["native_output_sha256"] = weak.sha(path.read_bytes())
        write_json(result_path, result)
        # Keep original routing and explicitly rehash inventory only; invalid native projection still rejects.
        files = weak.loads((self.fixture.asr / "audit/file-hashes.json").read_bytes())
        for relative in ("native/qwen06/clip-000.json", "results/qwen06/clip-000.json"):
            files["files"]["output/" + relative] = natural.record((self.fixture.asr / relative).read_bytes())
        files["files_sha256"] = weak.sha(weak.canonical(files["files"]))
        self.fixture.write("audit/file-hashes.json", files)
        self.fixture.audit["inventory_sha256"] = weak.sha((self.fixture.asr / "audit/file-hashes.json").read_bytes())
        self.fixture.write("audit/final-summary.json", self.fixture.audit)
        with self.assertRaises(ValueError):
            self.create()

    def test_valid_low_quality_native_record_is_retained_with_zero_loss(self):
        clip = self.fixture.inputs["clips"][0]
        self.fixture.native(clip, "whisper", "你好，小窝。", quality=["fixture-warning"])
        self.fixture.seal_fixture()
        report = self.create()
        row = report["rows"][0]
        self.assertFalse(row["decision"]["ctc_mask"])
        self.assertEqual(row["observations"][2]["text"], "你好，小窝。")
        self.assertEqual(report["statistics"]["verified_native_observations"], 6)
        self.assertEqual(report["statistics"]["usable_native_observations"], 5)

    def test_resealed_call_count_and_source_map_forgery_rejected(self):
        self.fixture.audit["state"]["original_recognition_calls"] += 1
        self.fixture.write("audit/final-summary.json", self.fixture.audit)
        with self.assertRaisesRegex(ValueError, "SELECTIVE_AUDIT_DIFFERS_FROM_REPLAY"):
            self.create()

    def test_duplicate_source_across_batch_names_does_not_inflate_observations(self):
        other = copy.deepcopy(self.fixture.entry()); other["batch_id"] = "renamed-batch"
        with self.assertRaisesRegex(ValueError, "DUPLICATE_GLOBAL_SOURCE"):
            self.create([self.fixture.entry(), other])

    def test_source_family_split_conflict_across_batches_is_rejected(self):
        second = NaturalFixture(self.root / "second", batch_id="batch-b", family="family-a",
                                session="session-b", split="train", shift=1)
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK"):
            self.create([self.fixture.entry(), second.entry()])

    def test_development_partition_cannot_be_relabelled_as_train(self):
        declaration = copy.deepcopy(self.fixture.declaration)
        declaration["parents"][0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT_PARTITION_NOT_TRAIN"):
            natural.validate_declaration(declaration)

    def test_pcm_gain_or_binding_edits_do_not_pass_via_resealed_inventory(self):
        path = self.fixture.bundle / "pcm16k/clip-000.f32"
        values = struct.unpack("<16f", path.read_bytes())
        path.write_bytes(struct.pack("<16f", *(2 * value for value in values)))
        self.fixture.seal_fixture()
        with self.assertRaisesRegex(ValueError, "WINDOW_AUDIO_BINDING"):
            self.create()


class NaturalImportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.copy_root = Path(tmp.name) / "other-checkout"
        for name in natural.PRODUCER_FILES:
            target = self.copy_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((natural.ROOT / name).read_bytes())

    def run_python(self, source):
        flags = ["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []
        result = subprocess.run([sys.executable, "-I", "-S", "-B", *flags, "-c", source],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cached_module_from_other_checkout_is_rejected(self):
        self.run_python(f"""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, {str(natural.ROOT)!r})
from automation import active_asr_weak_view
path = Path({str(self.copy_root)!r}) / 'automation/active_natural_asr_weak_view.py'
spec = importlib.util.spec_from_file_location('fixture_natural_mixed', path)
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except ValueError as error:
    if 'MIXED_PRODUCER_MODULE' not in str(error):
        raise
else:
    raise RuntimeError('cached cross-checkout module was accepted')
""")

    def test_source_change_after_import_requires_fresh_process_and_code_pin(self):
        self.run_python(f"""
from pathlib import Path
import sys
sys.path.insert(0, {str(self.copy_root)!r})
from automation import active_natural_asr_weak_view as view
path = Path({str(self.copy_root)!r}) / 'automation/active_asr_weak_view.py'
path.write_bytes(path.read_bytes() + b'\\n# deliberate local fixture source edit\\n')
try:
    view.producer_files()
except ValueError as error:
    if 'LOADED_SOURCE_CHANGED' not in str(error):
        raise
else:
    raise RuntimeError('stale imported producer code was accepted')
""")


if __name__ == "__main__":
    unittest.main()
