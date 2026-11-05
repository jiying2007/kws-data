#!/usr/bin/env python3
"""D2 raw source/native/selective-ASR replays; fictional PCM, no model imports."""
from __future__ import annotations

from pathlib import Path
import shutil
import struct
import sys
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_d2_weak_view as d2
import active_asr_weak_view as weak
import test_active_d1_weak_view as audio_fixture
import test_active_d1_outcomes as native_fixture
import test_selective_asr_weak_view as selective_fixture
import test_d2 as plan_fixture
import bulk_runner


class D2WeakViewTests(unittest.TestCase):
    asr_fixture = audio_fixture.D1WeakViewTests.asr_fixture
    rehash_package = audio_fixture.D1WeakViewTests.rehash_package
    record = staticmethod(native_fixture.D1OutcomesTests.record)
    write = selective_fixture.SelectiveViewTests.write
    seal = selective_fixture.SelectiveViewTests.seal

    def setUp(self):
        fixture = plan_fixture.D2Tests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.plan_fixture, self.root, self.plan = fixture, fixture.root, fixture.plan
        sources = fixture.old_sources
        if self._testMethodName == "test_new_twenty_four_uses_same_source_and_loss_contract":
            sources = fixture.new_sources(); self.plan = fixture.new_plan(sources)
        self.f = self
        self.plan_sha = weak.sha(weak.canonical(self.plan))
        self.plan_path = self.root / "d2-plan.json"
        weak.save_json(self.root, self.plan_path.name, self.plan)
        self.generation, self.native = self.root / "generation", self.root / "native"
        bulk_runner.run(self.plan, sources, self.root, self.generation, audio_fixture.FictionalClone(),
                        max_new_calls=8, writer=audio_fixture.fixture_writer)
        self.inputs, self.asr = self.root / "inputs", self.root / "asr"
        self.inputs.mkdir(); self.asr.mkdir()
        self.archive = self.root / "native-only.zip"
        refs = {ref["reference_id"]: ref for ref in self.plan["references"]}
        clips = []
        generation = {"fixture_only": True, "max_new_tokens": 256}
        with zipfile.ZipFile(self.archive, "w") as archive:
            for number, cell in enumerate(self.plan["cells"][:8]):
                name, ref = cell["cell_id"], refs[cell["reference_id"]]
                generated, native = self.generation / "cells" / name, self.native / name
                raw = (generated / "native-f32.wav").read_bytes()
                pcm = weak.float_wav_pcm(raw)
                values = [v[0] for v in struct.iter_unpack("<f", pcm)]
                derived = struct.pack("<16000f", *(values[i * 3 // 2] for i in range(16000)))
                checked = ref["reference_text_status"] in ("verified", "dataset-provided-content-checked")
                weak.save(native, "returned-array.npy", native_fixture.npy(pcm))
                weak.save(native, "returned-native.wav", raw)
                weak.save_json(native, "call.json", {"schema": "active-tts-generation-call-v1", "cell_id": name,
                    "reference_id": ref["reference_id"], "source_family_id": ref["source_family_id"],
                    "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
                    "reference_text_status": ref["reference_text_status"], "reference_mode": "ICL" if checked else "x-vector-only",
                    "reference_text_passed": checked, "seed": cell["seed"], "intended_text": cell["prompt"],
                    "generation": generation, "reference_prompt_seconds": 0.0, "attempt": 1})
                weak.save_json(native, "native.json", {"schema": "active-tts-native-return-v1",
                    "array": self.record(native / "returned-array.npy"), "wav": self.record(native / "returned-native.wav"),
                    "dtype": "<f4", "shape": [24000], "sample_rate_hz": 24000,
                    "samples_sha256": weak.sha(pcm), "all_finite": True, "technical_gate_applied": False})
                weak.save_json(native, "returned.json", {"generation_seconds": .1, "total_seconds": .2,
                    "native_stop_reason": "not_exposed_by_voice_clone_API"})
                if number == 0:
                    for path in ("completed.json", "native-f32.wav", "audio-16k.wav"):
                        (generated / path).unlink()
                    weak.save_json(generated, "failed.json", {"schema": "kws-bulk-clone-failure-v1", "cell_id": name,
                        "status": "FAILED_NO_RETRY", "error_type": "ValueError", "training_admitted": False})
                identifier = f"clip-{number:03d}"
                member = "cells/" + name + "/native-f32.wav"
                clip = {"id": identifier, "original_member": member,
                    "original_wav": {"path": "original/" + identifier + ".wav", "sha256": weak.sha(raw), "bytes": len(raw)},
                    "original_pcm": {"sample_rate": 24000, "channels": 1, "frames": 24000, "dtype": "<f4", "sha256": weak.sha(pcm)},
                    "derived_pcm": {"path": "pcm16k/" + identifier + ".f32", "sample_rate": 16000,
                                    "channels": 1, "frames": 16000, "dtype": "<f4", "sha256": weak.sha(derived)}}
                clips.append(clip); archive.writestr(member, raw)
                weak.save(self.inputs, clip["original_wav"]["path"], raw)
                weak.save(self.inputs, clip["derived_pcm"]["path"], derived)
        self.manifest = {"schema": "active-asr-inputs-v1", "source_archive_sha256": weak.sha(self.archive.read_bytes()),
                         "derivation": {"fixture_only": True}, "clips": clips}
        weak.save_json(self.inputs, "inputs.json", self.manifest)
        self.input_sha = weak.sha((self.inputs / "inputs.json").read_bytes())
        self.stage_path, self.recipe_path = self.root / "stage.json", self.root / "recipe.json"
        chunks = [[c["cell_id"] for c in self.plan["cells"][i:i + 8]] for i in range(0, len(self.plan["cells"]), 8)]
        stage = {"schema": "kws-d2-generation-stage-v1", "plan": self.record(self.plan_path), "generation": generation,
                 "chunks": chunks, "max_new_calls": len(self.plan["cells"]), "automatic_retries": 0,
                 "model_concurrency": 1, "d1_cells_generated_again": 0, "human_gold": False}
        weak.save_json(self.root, self.stage_path.name, stage)
        recipe = {"schema": "kws-d2-generation-job-v1", "stage": {"path": str(self.stage_path), **self.record(self.stage_path)},
                  "chunk_index": 0, "job_id": "generate-001", "selected_cells": chunks[0], "max_calls": 8, "generation": generation}
        weak.save_json(self.root, self.recipe_path.name, recipe)
        self.mapping = d2.build_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **self.options())
        self.map_path, self.policy_path = self.root / "source-map.json", self.root / "policy.json"
        weak.save_json(self.root, self.map_path.name, self.mapping)
        self.policy_path.write_bytes(selective_fixture.policy_bytes())
        self.observed_text = "请打开客厅的灯"
        ordinary = [i for i, c in enumerate(self.plan["cells"][:8])
                    if i != 0 and c["generation_category"] == "ordinary-negative"]
        self.changed_text_index = ordinary[0]
        for number, (clip, cell) in enumerate(zip(clips, self.plan["cells"])):
            for model in weak.MODELS:
                self.asr_fixture(clip, model, self.observed_text if number == self.changed_text_index else cell["prompt"])
        chars = list(dict.fromkeys(self.observed_text + "".join(c["prompt"] for c in self.plan["cells"])))
        chars += [chr(0x3400 + i) for i in range(2598 - len(chars))]
        tokens = ("<blk> 1\n" + "".join(f"{char} {index}\n" for index, char in enumerate(chars, 2))).encode()
        self.tokens = self.root / "tokens.txt"; self.tokens.write_bytes(tokens)
        pin = patch.object(weak, "FSMN_TOKENS_SHA", weak.sha(tokens)); pin.start(); self.addCleanup(pin.stop)
        self.selective()

    def options(self):
        return {"plan_path": self.plan_path, "plan_sha256": self.plan_sha,
                "generation_root": self.generation, "native_root": self.native,
                "job_recipe_path": self.recipe_path, "job_recipe_sha256": weak.sha(self.recipe_path.read_bytes()),
                "stage_path": self.stage_path}

    def selective(self):
        self.worker_sha = "d" * 64
        self.boundary = {"human_gold": False, "training_admitted": False, "shipping_approved": False,
                        "target_text_received": False, "peer_transcript_received": False,
                        "acoustic_verbatim_completeness": "unknown"}
        identities = {}
        for clip in self.manifest["clips"]:
            for model in weak.MODELS:
                paths = [self.asr / kind / model / (clip["id"] + ".json") for kind in ("results", "claims")]
                result, claim = [weak.loads(p.read_bytes()) for p in paths]
                result["attempt"] = claim["attempt"] = "selective-001"
                result["boundary"] = self.boundary
                for path, row in zip(paths, (result, claim)):
                    path.write_bytes(native_fixture.encoded(row))
                identities[model] = result["model_identity"]
        self.request = {"schema": "active-asr-selective-request-v2", "inputs": {"sha256": self.input_sha},
            "source_map": {"sha256": weak.sha(self.map_path.read_bytes())}, "policy": {"sha256": weak.SELECTIVE_POLICY_SHA},
            "threads": 4, "budget_seconds": 600, "max_calls": 3 * len(self.manifest["clips"]),
            "worker_sha256": self.worker_sha, "source_files": {"tools/active_asr_worker.py": self.worker_sha},
            "model_identities": identities, "boundary": self.boundary}
        routes = []
        for clip, source in zip(self.manifest["clips"], self.mapping["rows"]):
            primary = {model: weak.observation(self.asr, model, clip, self.input_sha, {}) for model in weak.MODELS[:2]}
            route = weak.selective_route(source, self.plan_sha, primary); routes.append(route)
            if route["whisper_state"] == "not_requested":
                for kind in ("claims", "native", "results"):
                    (self.asr / kind / "whisper" / (clip["id"] + ".json")).unlink()
        self.routing = {"schema": "active-asr-selective-routing-v2", "input_manifest_sha256": self.input_sha,
            "source_map_sha256": self.request["source_map"]["sha256"], "policy_sha256": weak.SELECTIVE_POLICY_SHA,
            "plan_sha256": self.plan_sha, "decoder_source_sha256": self.worker_sha, "items": routes}
        self.write("request.json", self.request); self.write("routing.json", self.routing)
        self.write("job.json", {"schema": "active-asr-selective-job-v2", "deadline_unix": 9999999999,
            "identity": {"request_sha256": weak.sha(native_fixture.encoded(self.request)), "max_calls": self.request["max_calls"],
                         "heavy_model_concurrency": 1, "threads": 4}, "supervision": {"process_group": 100}})
        self.write("network-guard.json", {"seccomp_installed": True, "no_new_privileges": True, "fixture_only": True})
        self.write("attempts/selective-001/summary.json", {"status": "complete", "finished_at": "2026-11-05T00:00:01+00:00"})
        for model in weak.MODELS:
            ids = [r["clip_id"] for r in routes if model != "whisper" or r["whisper_state"] == "requested"]
            if ids:
                self.write("attempts/selective-001/" + model + ".json", {"schema": "active-asr-worker-v1", "model": model,
                    "status": "complete", "attempt": "selective-001", "identity": identities[model],
                    "source_sha256": self.worker_sha, "boundary": self.boundary, "selected": ids,
                    "new_calls": len(ids), "reused": 0, "unresolved_prior_calls": [],
                    "heavy_lock": {"inherited": True, "supervised": True, "process_group": 100}})
                self.write("attempts/selective-001/" + model + "-load.json", {"fixture_only": True})
        (self.asr / "attempts/selective-001/quiet.log").write_bytes(b"")
        self.seal()

    def create(self):
        output = self.root / "d2-view"
        report = d2.create_view(self.inputs / "inputs.json", self.input_sha, self.archive, self.asr, output,
            fsmn_tokens=self.tokens, selective_source_map=self.map_path, selective_policy=self.policy_path,
            selective_audit_sha256=weak.sha((self.asr / "audit/final-summary.json").read_bytes()), **self.options())
        return output, report

    def test_replay_offline_preserves_all_observations_and_fixed_roles(self):
        output, report = self.create()
        self.assertEqual(report["schema"], d2.SCHEMA)
        self.assertEqual(report["statistics"]["source_items"], 8)
        self.assertEqual(report["statistics"]["machine_candidates"], 7)
        self.assertEqual(report["statistics"]["uncertain"], 1)
        self.assertEqual((report["statistics"]["train"], report["statistics"]["development"]), (6, 2))
        self.assertEqual(report["items"][self.changed_text_index]["decision"]["full_text"], self.observed_text)
        self.assertNotEqual(report["items"][self.changed_text_index]["source"]["intended_text"], self.observed_text)
        failed = report["items"][0]["decision"]
        self.assertFalse(failed["ctc_mask"]); self.assertEqual(failed["ctc_weight"], 0)
        self.assertEqual(failed["keyword_weight"], [0.0, 0.0])
        self.assertEqual(len((output / "all-observed.jsonl").read_text().splitlines()), 8)
        self.assertEqual(report["source_adapter"]["job_owner"]["selected_cell_ids"], [c["cell_id"] for c in self.plan["cells"][:8]])
        for root in (self.generation, self.native, self.inputs, self.asr):
            shutil.rmtree(root)
        self.plan_path.unlink(); self.recipe_path.unlink(); self.stage_path.unlink()
        verified = d2.verify_view(output, expected_manifest_sha256=weak.sha((output / "MANIFEST.json").read_bytes()))
        self.assertEqual(verified, report)
        train, dev = list(d2.load_training_rows(output)), list(d2.load_development_rows(output))
        self.assertEqual(len(train) + len(dev), 7)
        self.assertTrue({r["speaker_id"] for r in train}.isdisjoint(r["speaker_id"] for r in dev))
        self.assertTrue(all(Path(r["audio_path"]).is_absolute() and r["ctc_ids"] for r in [*train, *dev]))

    def test_new_twenty_four_uses_same_source_and_loss_contract(self):
        output, report = self.create()
        self.assertEqual(d2.verify_view(output), report)
        self.assertEqual(report["source_adapter"]["stage"], "new-twenty-four")
        parents = {r["speaker_group_id"] for r in self.plan["d1_parent"]["plan"]["references"]}
        self.assertTrue(parents.isdisjoint(r["source"]["speaker_id"] for r in report["items"]))

    def test_plan_pin_and_owner_scope_are_required_when_supplied(self):
        options = self.options(); options["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "D2_FROZEN_PLAN_CHANGED"):
            d2.build_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **options)
        options = self.options(); options["job_recipe_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "D2_JOB_RECIPE_PIN_CHANGED"):
            d2.build_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **options)
        recipe = weak.loads(self.recipe_path.read_bytes()); recipe["selected_cells"] = recipe["selected_cells"][1:]
        self.recipe_path.write_bytes(native_fixture.encoded(recipe))
        with self.assertRaisesRegex(ValueError, "D2_JOB_OWNER_SCOPE"):
            d2.build_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **self.options())

    def test_archive_sorting_does_not_reassign_source_ownership(self):
        manifest = {**self.manifest, "clips": sorted(self.manifest["clips"], key=lambda row: row["original_member"])}
        self.assertNotEqual(manifest["clips"], self.manifest["clips"])
        raw = native_fixture.encoded(manifest)
        (self.inputs / "inputs.json").write_bytes(raw)
        mapping = d2.build_source_map(self.inputs / "inputs.json", weak.sha(raw), self.archive, **self.options())
        self.assertEqual([r["source_id"] for r in mapping["rows"]],
                         [r["original_member"].split("/")[1] for r in manifest["clips"]])

    def test_reference_call_mode_and_unknown_returns_cannot_be_fabricated(self):
        path = self.native / self.plan["cells"][0]["cell_id"] / "call.json"
        original = path.read_bytes(); call = weak.loads(original)
        call["reference_mode"] = "wrong"; path.write_bytes(native_fixture.encoded(call))
        with self.assertRaisesRegex(ValueError, "D2_NATIVE_CALL_REFERENCE_MODE_JOIN"):
            self.create()
        path.write_bytes(original); path.with_name("returned-native.wav").unlink()
        with self.assertRaisesRegex(ValueError, "D2_NATIVE_FILE_ROSTER"):
            self.create()

    def test_rehashed_report_cannot_turn_failed_native_into_a_target(self):
        output, report = self.create()
        report["items"][0]["decision"].update(ctc_mask=True, ctc_weight=1, full_text="你好小窝")
        (output / "report.json").write_bytes(native_fixture.encoded(report)); self.rehash_package(output)
        with self.assertRaisesRegex(ValueError, "WEAK_VIEW_DIFFERS_FROM_RETAINED_EVIDENCE"):
            list(d2.load_training_rows(output))

    def test_rehashed_native_array_must_still_match_returned_wav(self):
        output, _ = self.create()
        folder = output / "evidence/source/d2/native" / self.plan["cells"][0]["cell_id"]
        array = folder / "returned-array.npy"
        raw = array.read_bytes(); array.write_bytes(raw[:-4] + struct.pack("<f", .123))
        native = weak.loads((folder / "native.json").read_bytes())
        pcm = weak.float_wav_pcm((folder / "returned-native.wav").read_bytes())
        native.update(array=self.record(array), samples_sha256=weak.sha(pcm[:-4] + struct.pack("<f", .123)))
        (folder / "native.json").write_bytes(native_fixture.encoded(native)); self.rehash_package(output)
        with self.assertRaisesRegex(ValueError, "D1_NATIVE_ARRAY_PCM_JOIN"):
            d2.verify_view(output)

    def test_retained_asr_native_change_is_rejected_even_if_outer_manifest_rehashed(self):
        output, _ = self.create()
        path = output / "evidence/asr/native/qwen06/clip-000.json"
        body = weak.loads(path.read_bytes()); body["native"]["decoded_skip_special_tokens"] = "你好小窝"
        path.write_bytes(native_fixture.encoded(body)); self.rehash_package(output)
        with self.assertRaisesRegex(ValueError, "SELECTIVE_INVENTORY_FILE_CHANGED"):
            d2.verify_view(output)

    def test_cross_package_split_leak_and_modified_exports_are_rejected(self):
        output, report = self.create()
        row = next(row for row in report["items"] if row["source"]["split"] == "train")
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK"):
            d2.verify_view(output, other_split_sources=[{"role": "development-only", "speaker_id": row["source"]["speaker_id"]}])
        (output / "uncertain.jsonl").write_bytes(b""); self.rehash_package(output)
        with self.assertRaisesRegex(ValueError, "REGISTRY_FILE|D2_WEAK_EXPORT_CHANGED"):
            d2.verify_view(output)


if __name__ == "__main__":
    unittest.main()
