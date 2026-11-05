#!/usr/bin/env python3
"""Context native/selective-ASR replays; fictional PCM, no model imports."""
from __future__ import annotations

from pathlib import Path
import copy
import shutil
import struct
import sys
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_context_weak_view as context
import active_asr_weak_view as weak
import test_active_d1_weak_view as audio_fixture
import test_active_d1_outcomes as native_fixture
import test_selective_asr_weak_view as selective_fixture
import test_d2 as plan_fixture
import bulk_runner


class ContextWeakViewTests(unittest.TestCase):
    asr_fixture = audio_fixture.D1WeakViewTests.asr_fixture
    rehash_package = audio_fixture.D1WeakViewTests.rehash_package
    record = staticmethod(native_fixture.D1OutcomesTests.record)
    write = selective_fixture.SelectiveViewTests.write
    seal = selective_fixture.SelectiveViewTests.seal

    def setUp(self):
        fixture = plan_fixture.D2Tests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.plan_fixture, self.root = fixture, fixture.root
        sources = fixture.old_sources
        self.plan = context.pool.plan(fixture.plan, context.pool.digest(fixture.plan), sources, self.root,
                                     count=16, batch_id="context-fixture")
        self.f = self
        self.plan_sha = weak.sha(weak.canonical(self.plan))
        self.plan_path = self.root / "context-plan.json"
        weak.save_json(self.root, self.plan_path.name, self.plan)
        self.generation, self.native = self.root / "generation", self.root / "native"
        backend = audio_fixture.FictionalClone()
        backend.recipe = context.pool.runner_recipe(self.plan, [c["cell_id"] for c in self.plan["cells"][:8]])
        context.pool.run(self.plan, sources, self.root, self.generation, backend,
                         max_new_calls=8, writer=audio_fixture.fixture_writer)
        self.inputs, self.asr = self.root / "inputs", self.root / "asr"
        self.inputs.mkdir(); self.asr.mkdir()
        self.archive = self.root / "native-only.zip"
        refs = {ref["reference_id"]: ref for ref in self.plan["references"]}
        clips = []
        generation = context.pool.GENERATION
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
        stage = {"schema": "kws-context-generation-stage-v1", "plan": self.record(self.plan_path), "generation": generation,
                 "chunks": chunks, "max_new_calls": len(self.plan["cells"]), "automatic_retries": 0,
                 "model_concurrency": 1, "d1_cells_generated_again": 0, "d2_cells_generated_again": 0,
                 "human_gold": False}
        weak.save_json(self.root, self.stage_path.name, stage)
        recipe = {"schema": "kws-context-generation-job-v1", "stage": {"path": str(self.stage_path), **self.record(self.stage_path)},
                  "chunk_index": 0, "job_id": "generate-001", "selected_cells": chunks[0], "max_calls": 8, "generation": generation}
        weak.save_json(self.root, self.recipe_path.name, recipe)
        self.mapping = context.build_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **self.options())
        self.map_path, self.policy_path = self.root / "source-map.json", self.root / "policy.json"
        weak.save_json(self.root, self.map_path.name, self.mapping)
        self.policy_path.write_bytes(selective_fixture.policy_bytes())
        self.changed_text_index, self.missing_keyword_index = 1, 2
        keyword = weak.KEYWORDS[self.plan["cells"][self.changed_text_index]["intended_keyword_id"] - 1]
        self.observed_text = "我听见" + keyword + "请回应"
        self.missing_keyword_text = "今天晚上吃什么"
        for number, (clip, cell) in enumerate(zip(clips, self.plan["cells"])):
            for model in weak.MODELS:
                text = (self.observed_text if number == self.changed_text_index else
                        self.missing_keyword_text if number == self.missing_keyword_index else cell["prompt"])
                self.asr_fixture(clip, model, text)
        chars = list(dict.fromkeys(self.observed_text + self.missing_keyword_text +
                                  "".join(c["prompt"] for c in self.plan["cells"])))
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
        output = self.root / "context-view"
        report = context.create_view(self.inputs / "inputs.json", self.input_sha, self.archive, self.asr, output,
            fsmn_tokens=self.tokens, selective_source_map=self.map_path, selective_policy=self.policy_path,
            selective_audit_sha256=weak.sha((self.asr / "audit/final-summary.json").read_bytes()), **self.options())
        return output, report

    def test_context_native_asr_package_replays_offline_without_changing_source_roles(self):
        output, report = self.create()
        self.assertEqual(report["schema"], context.SCHEMA)
        self.assertEqual(report["source_adapter"]["schema"], context.SOURCE_ADAPTER)
        self.assertEqual(report["statistics"]["source_items"], 8)
        self.assertEqual(report["statistics"]["machine_candidates"], 6)
        self.assertEqual(report["statistics"]["uncertain"], 2)
        self.assertEqual((report["statistics"]["train"], report["statistics"]["development"]), (6, 2))
        self.assertEqual(len((output / "all-observed.jsonl").read_text().splitlines()), 8)
        changed = report["items"][self.changed_text_index]
        self.assertEqual(changed["decision"]["full_text"], self.observed_text)
        self.assertNotEqual(changed["source"]["intended_text"], self.observed_text)
        self.assertEqual(changed["context_observation"]["confirmed_context_class"], "middle")
        self.assertFalse(changed["context_observation"]["requested_context_confirmed"])
        for root in (self.generation, self.native, self.inputs, self.asr):
            shutil.rmtree(root)
        self.plan_path.unlink(); self.recipe_path.unlink(); self.stage_path.unlink()
        self.assertEqual(context.verify_view(output), report)
        train, dev = list(context.load_training_rows(output)), list(context.load_development_rows(output))
        self.assertEqual((len(train), len(dev)), (4, 2))
        self.assertTrue({r["speaker_id"] for r in train}.isdisjoint(r["speaker_id"] for r in dev))
        self.assertTrue(all(r["ctc_text"] == r["full_text"] and Path(r["audio_path"]).is_absolute() for r in train + dev))
        self.assertTrue(report["consumer_contracts"]["fsmn-fulltext"]["legacy"])
        self.assertFalse(report["consumer_contracts"]["fsmn-fulltext"]["full2601_projection"])

    def test_failed_native_and_unconfirmed_positive_keep_all_transcripts_with_zero_loss(self):
        _, report = self.create()
        failed = report["items"][0]
        missing = report["items"][self.missing_keyword_index]
        for row in (failed, missing):
            self.assertEqual(row["decision"]["status"], "uncertain")
            self.assertFalse(row["decision"]["ctc_mask"])
            self.assertEqual(row["decision"]["ctc_weight"], 0)
            self.assertEqual(row["decision"]["keyword_targets"], [None, None])
            self.assertEqual(row["decision"]["keyword_weight"], [0, 0])
            self.assertFalse(row["context_observation"]["confirmed_positive"])
            self.assertIsNone(row["context_observation"]["confirmed_context_class"])
            self.assertTrue(all(item["text"] is not None for item in row["asr"][:2]))
        self.assertIsNone(failed["decision"]["full_text"])
        self.assertEqual(missing["decision"]["full_text"], self.missing_keyword_text)
        self.assertEqual(missing["decision"]["reason"], "intended_full_wake_not_confirmed_never_auto_negative")

    def test_d1_or_d2_plan_cannot_masquerade_as_context(self):
        for value in (self.plan_fixture.plan, self.plan_fixture.parent):
            raw = weak.canonical(value)
            with self.assertRaisesRegex(ValueError, "CONTEXT_PLAN_SCHEMA_REQUIRED"):
                context._plan(raw, weak.sha(raw))
        changed = copy.deepcopy(self.plan)
        changed["references"][0]["role"] = "development-only"
        raw = weak.canonical(changed)
        with self.assertRaises(ValueError):
            context._plan(raw, weak.sha(raw))

    def test_owner_pin_and_historical_zero_reissue_fields_are_required(self):
        options = self.options(); options["job_recipe_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "CONTEXT_JOB_RECIPE_PIN_CHANGED"):
            context.build_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **options)
        plan_raw = self.plan_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "CONTEXT_JOB_RECIPE_PIN_AND_STAGE_REQUIRED"):
            context._owner(self.plan, plan_raw, [], None, None, None, None)
        original_stage, original_recipe = self.stage_path.read_bytes(), self.recipe_path.read_bytes()
        for field in ("d1_cells_generated_again", "d2_cells_generated_again"):
            stage = weak.loads(original_stage); stage[field] = 1
            self.stage_path.write_bytes(native_fixture.encoded(stage))
            recipe = weak.loads(original_recipe)
            recipe["stage"] = {"path": str(self.stage_path), **self.record(self.stage_path)}
            self.recipe_path.write_bytes(native_fixture.encoded(recipe))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "CONTEXT_STAGE_REQUEST_BOUNDARY"):
                context.build_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **self.options())
        self.stage_path.write_bytes(original_stage); self.recipe_path.write_bytes(original_recipe)

    def test_owner_cannot_claim_another_chunk_or_generation_recipe(self):
        original = self.recipe_path.read_bytes()
        for key, value in (("selected_cells", [c["cell_id"] for c in self.plan["cells"][8:]]),
                           ("generation", {**context.pool.GENERATION, "temperature": .2}),
                           ("chunk_index", False), ("job_id", "generate-099")):
            recipe = weak.loads(original); recipe[key] = value
            self.recipe_path.write_bytes(native_fixture.encoded(recipe))
            with self.subTest(key=key), self.assertRaises(ValueError):
                context.build_source_map(self.inputs / "inputs.json", self.input_sha, self.archive, **self.options())
        self.recipe_path.write_bytes(original)

    def test_native_reference_text_array_and_unknown_terminal_must_authenticate(self):
        native = self.native / self.plan["cells"][0]["cell_id"]
        call = native / "call.json"
        original = call.read_bytes(); changed = weak.loads(original)
        changed["intended_text"] = "invented observation"
        call.write_bytes(native_fixture.encoded(changed))
        with self.assertRaisesRegex(ValueError, "CONTEXT_NATIVE_CALL_REFERENCE_MODE_JOIN"):
            self.create()
        call.write_bytes(original)
        array = native / "returned-array.npy"
        array.write_bytes(array.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "CONTEXT_NATIVE_RETURN_JOIN"):
            self.create()
        # A claim without a terminal remains unknown even if native files exist.
        (self.generation / "cells" / self.plan["cells"][0]["cell_id"] / "failed.json").unlink()
        with self.assertRaisesRegex(ValueError, "CONTEXT_NATIVE_FILE_ROSTER"):
            self.create()

    def test_actual_context_statistics_cannot_be_replaced_by_requested_class_counts(self):
        output, report = self.create()
        report["items"][self.changed_text_index]["context_observation"]["confirmed_context_class"] = "prefix"
        (output / "report.json").write_bytes(native_fixture.encoded(report))
        self.rehash_package(output)
        with self.assertRaisesRegex(ValueError, "WEAK_VIEW_DIFFERS_FROM_RETAINED_EVIDENCE"):
            context.verify_view(output)

    def test_opposite_split_reference_identity_blocks_even_zero_weight_rows(self):
        output, report = self.create()
        source = report["items"][0]["source"]
        with self.assertRaises(ValueError):
            context.verify_view(output, other_split_sources=[{"role": "development-only",
                "source_family_id": source["source_family_id"], "speaker_id": source["speaker_id"]}])


class ContextObservationTests(unittest.TestCase):
    def test_position_and_nonoverlapping_counts_come_from_full_observed_text(self):
        cases = (("你好小窝，请把灯打开", 1, "prefix", 1),
                 ("我听见你好小窝请回应", 1, "middle", 1),
                 ("准备好了就回应你好小窝", 1, "suffix", 1),
                 ("小窝小窝，小窝小窝", 2, "repeated-call", 2),
                 ("你好小窝", 1, "isolated", 1), ("没有唤醒词", 1, "absent", 0))
        for text, keyword_id, expected, count in cases:
            row = context.observed_context(text)["keywords"][keyword_id - 1]
            with self.subTest(text=text):
                self.assertEqual((row["observed_context_class"], row["occurrences"]), (expected, count))
        repeated = context.observed_context("小窝小窝小窝小窝")["keywords"][1]
        self.assertEqual(repeated["start_offsets"], [0, 4])

    def test_unchanged_full_transcript_policy_admits_context_without_using_requested_text(self):
        for text in ("你好小窝请把灯打开", "准备好你好小窝再回应", "请回应你好小窝", "你好小窝你好小窝"):
            observations = [{"model": model, "status": "complete", "usable": True,
                             "text": text, "normalized_text": weak.normalize(text)} for model in weak.MODELS[:2]]
            source = {"generation_outcome": "completed", "intended_kind": "positive-intent", "intended_keyword_id": 1}
            label = weak.decide(source, observations)
            self.assertEqual(label["full_text"], text)
            self.assertEqual(label["keyword_targets"], [1, 0])
            self.assertEqual(label["ctc_weight"], .5)
            self.assertTrue(label["ctc_mask"])


if __name__ == "__main__":
    unittest.main()

