#!/usr/bin/env python3
"""Versioned sparse evidence fixtures, with no model or network execution."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_weak_view as weak
import test_active_d1_outcomes as d1_fixture


def policy_bytes():
    value = {"schema": "active-asr-selective-policy-v2", "primary_models": ["qwen06", "sensevoice"],
        "selective_model": "whisper", "normalization": "reference-text-NFKC-casefold-punctuation-CJK-spaces-v1",
        "mandatory_reasons": ["primary_incomplete", "primary_quality_uncertain", "primary_disagreement", "source_quality_uncertain"],
        "agreement_audit": {"namespace": "kws-d1-whisper-agreement-audit-v2", "seed": 20261105,
            "hash_input": "UTF8(namespace|seed|plan_sha256|source_id)", "hash_algorithm": "sha256",
            "selection": "int(hash,16)*5 < 2**256", "probability_numerator": 1, "probability_denominator": 5},
        "no_target_or_peer_text_to_native_worker": True, "missing_requested_result_blocks_complete_audit": True,
        "quality_retries": 0, "human_gold": False}
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


class SelectiveRoutingTests(unittest.TestCase):
    def test_fixed_hash_and_two_vote_state(self):
        self.assertEqual(weak.sha(policy_bytes()), weak.SELECTIVE_POLICY_SHA)
        source = {"clip_id": "clip-000", "source_id": "bulk-000000000000000000000000", "quality_uncertain": False}
        primary = {model: {"status": "complete", "text": "你好，小窝。", "result_sha256": "b" * 64,
                           "decoder_terminated": True, "quality_flags": []} for model in weak.MODELS[:2]}
        route = weak.selective_route(source, "a" * 64, primary)
        self.assertEqual(route["sample_hash"], "b617ae80346ac9c1e5fb2e1b63d1a0a2fd479814c537019d035d772e49b3c807")
        self.assertEqual(route["whisper_state"], "not_requested")
        self.assertEqual(route["sample_hash"], weak.selective_route({**source, "clip_id": "clip-027"}, "a" * 64, primary)["sample_hash"])
        selected = weak.selective_route({**source, "source_id": "bulk-000000000000000000000001"}, "a" * 64, primary)
        self.assertEqual(selected["sample_hash"], "054a426f42f8adb4e233ac2316f69243a4e4eefbabd976b6513325a169324422")
        self.assertEqual(selected["reasons"], ["deterministic_20pct"])

    def test_each_mandatory_reason_and_word_boundaries(self):
        source = {"clip_id": "clip-000", "source_id": "bulk-000000000000000000000000", "quality_uncertain": False}
        base = {"status": "complete", "text": "你好小窝", "result_sha256": "a" * 64,
                "decoder_terminated": True, "quality_flags": []}
        cases = [({"decoder_terminated": False}, "primary_incomplete"), ({"text": " 。"}, "primary_incomplete"),
                 ({"quality_flags": ["warning"]}, "primary_quality_uncertain"), ({"text": "你好小屋"}, "primary_disagreement")]
        for changes, reason in cases:
            route = weak.selective_route(source, "a" * 64, {"qwen06": {**base, **changes}, "sensevoice": base})
            self.assertIn(reason, route["reasons"]); self.assertEqual(route["whisper_state"], "requested")
        route = weak.selective_route({**source, "quality_uncertain": True}, "a" * 64, dict.fromkeys(weak.MODELS[:2], base))
        self.assertEqual(route["reasons"], ["source_quality_uncertain"])
        route = weak.selective_route(source, "a" * 64, {"qwen06": {**base, "text": "the rapist"},
                                                       "sensevoice": {**base, "text": "therapist"}})
        self.assertIn("primary_disagreement", route["reasons"])


class SelectiveViewTests(unittest.TestCase):
    def setUp(self):
        fixture = d1_fixture.D1OutcomesTests("test_mixed_outcomes_remain_replayable_without_original_roots")
        fixture.setUp(); self.addCleanup(fixture.doCleanups); self.f = fixture
        self.mapping = weak.build_d1_source_map(fixture.inputs / "inputs.json", fixture.input_sha, fixture.archive, **fixture.options())
        self.map_path, self.policy_path = fixture.root / "source-map.json", fixture.root / "policy.json"
        self.map_path.write_bytes(d1_fixture.encoded(self.mapping)); self.policy_path.write_bytes(policy_bytes())
        self.worker_sha = "d" * 64
        self.boundary = {"human_gold": False, "training_admitted": False, "shipping_approved": False,
            "target_text_received": False, "peer_transcript_received": False, "acoustic_verbatim_completeness": "unknown",
            "language_independent_discovery": False, "whisper_language_condition": "Chinese (zh), explicitly supplied, not detected"}
        identities = {}
        for clip in fixture.manifest["clips"]:
            for model in weak.MODELS:
                result_path = fixture.asr / "results" / model / (clip["id"] + ".json")
                claim_path = fixture.asr / "claims" / model / (clip["id"] + ".json")
                result = weak.loads(result_path.read_bytes()); claim = weak.loads(claim_path.read_bytes())
                result["attempt"] = claim["attempt"] = "selective-001"
                result["boundary"] = self.boundary
                result_path.write_bytes(d1_fixture.encoded(result)); claim_path.write_bytes(d1_fixture.encoded(claim))
                identities[model] = result["model_identity"]
        self.request = {"schema": "active-asr-selective-request-v2", "created_at": "2026-11-05T00:00:00+00:00",
            "runtime": "/fictional/no-model-runtime", "output": str(fixture.asr),
            "inputs": {"path": str(fixture.inputs / "inputs.json"), "sha256": fixture.input_sha},
            "source_map": {"path": str(self.map_path), "sha256": weak.sha(self.map_path.read_bytes())},
            "policy": {"path": str(self.policy_path), "sha256": weak.SELECTIVE_POLICY_SHA},
            "threads": 4, "budget_seconds": 600, "max_calls": 3 * len(fixture.manifest["clips"]),
            "worker_sha256": self.worker_sha, "source_files": {"tools/active_asr_worker.py": self.worker_sha},
            "model_identities": identities, "boundary": self.boundary}
        routes = []
        for clip, source in zip(fixture.manifest["clips"], self.mapping["rows"]):
            # Fixture primaries all have the same complete full sentence. The
            # native output is independently projected by the tested consumer.
            sample = weak.sha(("kws-d1-whisper-agreement-audit-v2|20261105|" + fixture.plan_sha + "|" + source["source_id"]).encode())
            reasons = (["source_quality_uncertain"] if source["quality_uncertain"] else
                       ["deterministic_20pct"] if int(sample, 16) * 5 < 2**256 else ["two_primary_agree_not_sampled"])
            requested = reasons != ["two_primary_agree_not_sampled"]
            routes.append({"clip_id": clip["id"], "source_id": source["source_id"], "sample_hash": sample,
                "primary_result_sha256": {model: weak.sha((fixture.asr / "results" / model / (clip["id"] + ".json")).read_bytes())
                                           for model in weak.MODELS[:2]},
                "whisper_state": "requested" if requested else "not_requested", "reasons": reasons})
            if not requested:
                for kind in ("claims", "native", "results"):
                    (fixture.asr / kind / "whisper" / (clip["id"] + ".json")).unlink()
        self.routing = {"schema": "active-asr-selective-routing-v2", "input_manifest_sha256": fixture.input_sha,
                        "source_map_sha256": self.request["source_map"]["sha256"], "policy_sha256": weak.SELECTIVE_POLICY_SHA,
                        "plan_sha256": fixture.plan_sha, "decoder_source_sha256": self.worker_sha, "items": routes}
        self.write("request.json", self.request); self.write("routing.json", self.routing)
        self.write("job.json", {"schema": "active-asr-selective-job-v2", "created_at": "2026-11-05T00:00:00+00:00",
            "deadline_unix": 9999999999, "identity": {"request_sha256": weak.sha(d1_fixture.encoded(self.request)),
            "max_calls": self.request["max_calls"], "heavy_model_concurrency": 1, "threads": 4},
            "supervision": {"process_id": 100, "process_group": 100, "inherited_lease": True, "supervised": True}})
        self.write("network-guard.json", {"seccomp_installed": True, "no_new_privileges": True, "fixture_only": True})
        self.write("attempts/selective-001/summary.json", {"schema": "active-asr-selective-attempt-v2", "status": "complete",
                                                        "finished_at": "2026-11-05T00:00:01+00:00"})
        for model in weak.MODELS:
            ids = [r["clip_id"] for r in routes if model != "whisper" or r["whisper_state"] == "requested"]
            if ids:
                self.write("attempts/selective-001/" + model + ".json", {"schema": "active-asr-worker-v1", "model": model,
                    "status": "complete", "attempt": "selective-001", "identity": identities[model],
                    "source_sha256": self.worker_sha, "boundary": self.boundary, "selected": ids,
                    "new_calls": len(ids), "reused": 0, "unresolved_prior_calls": [],
                    "heavy_lock": {"inherited": True, "supervised": True, "process_group": 100}})
                self.write("attempts/selective-001/" + model + "-load.json", {"fictional_upstream_load_audit": True})
        (fixture.asr / "attempts/selective-001/quiet.log").write_bytes(b"")
        self.seal()

    def write(self, name, value):
        path = self.f.asr / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(d1_fixture.encoded(value))

    def seal(self):
        f = self.f
        files = {"output/" + p.relative_to(f.asr).as_posix(): f.record(p) for p in f.asr.rglob("*")
                 if p.is_file() and not p.relative_to(f.asr).as_posix().startswith("audit/")}
        files.update({"inputs/" + p.relative_to(f.inputs).as_posix(): f.record(p) for p in f.inputs.rglob("*") if p.is_file()})
        files.update({"metadata/source_map.json": f.record(self.map_path), "metadata/policy.json": f.record(self.policy_path)})
        self.write("audit/file-hashes.json", {"schema": "active-asr-selective-inventory-v2", "files": files,
                                            "file_count": len(files), "files_sha256": weak.sha(weak.canonical(files))})
        requested = {(model, row["clip_id"]) for row in self.routing["items"] for model in weak.MODELS
                     if model != "whisper" or row["whisper_state"] == "requested"}
        skipped = {("whisper", row["clip_id"]) for row in self.routing["items"] if row["whisper_state"] == "not_requested"}
        self.audit = {"schema": "active-asr-selective-final-audit-v2", "created_at": "2026-11-05T00:00:01+00:00",
            "request_sha256": weak.sha((f.asr / "request.json").read_bytes()),
            "inventory_sha256": weak.sha((f.asr / "audit/file-hashes.json").read_bytes()), "state": {
                "status": "complete", "input_manifest_sha256": f.input_sha,
                "source_map_sha256": weak.sha(self.map_path.read_bytes()), "policy_sha256": weak.SELECTIVE_POLICY_SHA,
                "routing_sha256": weak.sha((f.asr / "routing.json").read_bytes()), "decoder_source_sha256": self.worker_sha,
                "inputs": len(f.manifest["clips"]), "requested_matrix": [list(x) for x in sorted(requested)],
                "not_requested_matrix": [list(x) for x in sorted(skipped)], "expected_matrix_cells": len(requested),
                "verified_matrix_cells": len(requested), "original_recognition_calls": len(requested),
                "call_budget": self.request["max_calls"], "duplicate_model_clip_pairs": 0,
                "models": {m: {"requested": sum(x == m for x, _ in requested), "not_requested": sum(x == m for x, _ in skipped),
                               "verified": sum(x == m for x, _ in requested), "claims": sum(x == m for x, _ in requested),
                               "worker_new_calls": sum(x == m for x, _ in requested)} for m in weak.MODELS},
                "errors": [], "finished_before_deadline": True, "boundary": self.boundary,
                "unrequested_is_not_a_negative_or_agreement": True}}
        self.write("audit/final-summary.json", self.audit)

    def create(self):
        f = self.f; output = f.root / "sparse-view"
        report = weak.create_view(f.inputs / "inputs.json", f.input_sha, f.archive, f.asr, output,
            fsmn_tokens=f.tokens, **f.options(), selective_source_map=self.map_path, selective_policy=self.policy_path,
            selective_audit_sha256=weak.sha((f.asr / "audit/final-summary.json").read_bytes()))
        return output, report

    def test_explicit_not_requested_is_only_two_votes_and_replays_without_origins(self):
        path, report = self.create()
        self.assertEqual(report["schema"], weak.SCHEMA_V2)
        skipped = [r for r in report["items"] if r["asr"][2]["status"] == "not_requested"]
        self.assertTrue(skipped)
        self.assertEqual(report["statistics"]["not_requested_asr_results"], len(skipped))
        self.assertEqual(report["statistics"]["completed_asr_results"], 48 - len(skipped))
        for row in skipped:
            self.assertIsNone(row["asr"][2]["text"])
            self.assertFalse(row["asr"][2]["usable"])
            self.assertEqual(row["decision"]["ctc_weight"], .5)
            self.assertEqual(row["decision"]["label_tier"], "machine-weak-two")
        failed = report["items"][0]
        self.assertEqual(failed["source"]["generation_outcome"], "failed-native")
        self.assertTrue(all(o["status"] == "complete" for o in failed["asr"]))
        self.assertIsNone(failed["decision"]["full_text"])
        for root in (self.f.output, self.f.prepared, self.f.asr, self.f.inputs):
            shutil.rmtree(root)
        self.map_path.unlink(); self.policy_path.unlink()
        rows = [*weak.load_training_rows(path), *weak.load_development_rows(path)]
        self.assertEqual(len(rows), 15)

    def test_unrequested_native_call_rejected_even_with_resealed_inventory(self):
        row = next(r for r in self.routing["items"] if r["whisper_state"] == "not_requested")
        clip = next(c for c in self.f.manifest["clips"] if c["id"] == row["clip_id"])
        self.f.asr_fixture(clip, "whisper", "你好小窝")
        self.seal()
        with self.assertRaisesRegex(ValueError, "SELECTIVE_REQUESTED_MATRIX_CHANGED"):
            self.create()

    def test_missing_requested_output_is_not_not_requested(self):
        first = self.f.manifest["clips"][0]["id"]
        (self.f.asr / "native/whisper" / (first + ".json")).unlink(); self.seal()
        with self.assertRaisesRegex(ValueError, "SELECTIVE_REQUESTED_NATIVE_RESULT"):
            self.create()

    def test_resealed_sampling_hash_and_call_count_forgery_rejected(self):
        self.routing["items"][1]["sample_hash"] = "0" * 64
        self.write("routing.json", self.routing); self.seal()
        with self.assertRaisesRegex(ValueError, "SELECTIVE_ROUTING_DIFFERS_FROM_EVIDENCE"):
            self.create()

    def test_final_counts_recomputed(self):
        self.audit["state"]["original_recognition_calls"] += 1
        self.write("audit/final-summary.json", self.audit)
        with self.assertRaisesRegex(ValueError, "SELECTIVE_AUDIT_DIFFERS_FROM_REPLAY"):
            self.create()

    def test_failed_source_quality_flag_cannot_be_erased(self):
        self.mapping["rows"][0].update(quality_uncertain=False, quality_reasons=[])
        self.map_path.write_bytes(d1_fixture.encoded(self.mapping)); self.seal()
        with self.assertRaisesRegex(ValueError, "SELECTIVE_SOURCE_MAP_CHANGED"):
            self.create()

    def test_retained_report_cannot_make_unrequested_a_third_vote(self):
        path, report = self.create()
        row = next(r for r in report["items"] if r["asr"][2]["status"] == "not_requested")
        row["decision"].update(ctc_weight=1.0, label_tier="machine-weak-three")
        (path / "report.json").write_bytes(d1_fixture.encoded(report)); self.f.rehash_package(path)
        with self.assertRaisesRegex(ValueError, "WEAK_VIEW_DIFFERS_FROM_RETAINED_EVIDENCE"):
            list(weak.load_training_rows(path))


if __name__ == "__main__":
    unittest.main()
