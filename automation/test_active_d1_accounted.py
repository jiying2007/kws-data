#!/usr/bin/env python3
"""Fictional timeout/completed receipts and numeric media; no model execution."""
from __future__ import annotations

import copy
import io
from pathlib import Path
import shutil
import struct
import sys
import unittest
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import active_asr_weak_view as w
import test_active_d1_weak_view as base
from test_active_d1_outcomes import npy
from test_active_asr_weak_view import float_wav


def encoded(value):
    return w.canonical(value) + b"\n"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else encoded(value))
    return pin(path)


def pin(path):
    raw = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": w.sha(raw)}


def record(path):
    return {key: value for key, value in pin(path).items() if key != "path"}


def roster(path):
    return {p.relative_to(path).as_posix(): record(p) for p in sorted(path.rglob("*")) if p.is_file()}


def dense_view(testcase, fixture, rows, name, *, generation=None):
    folder = testcase.root / name; folder.mkdir()
    testcase.inputs, testcase.asr = folder / "inputs", folder / "asr"
    testcase.inputs.mkdir(); testcase.asr.mkdir()
    archive, clips = folder / "native.zip", []
    with zipfile.ZipFile(archive, "w") as stream:
        for index, row in enumerate(rows):
            value, identifier = row["record"], "clip-" + str(index).zfill(3)
            wav = (Path(value["output_root"]) / value["native_wav"]).read_bytes()
            pcm = w.float_wav_pcm(wav); samples = [x[0] for x in struct.iter_unpack("<f", pcm)]
            derived = struct.pack("<1600f", *(samples[j * 3 // 2] for j in range(1600)))
            member = "cells/" + row["cell_id"] + "/native-f32.wav"
            clip = {"id": identifier, "original_member": member,
                "original_wav": {"path": "original/" + identifier + ".wav", "sha256": w.sha(wav), "bytes": len(wav)},
                "original_pcm": {"sample_rate": 24000, "channels": 1, "frames": 2400,
                    "dtype": "<f4", "sha256": w.sha(pcm)},
                "derived_pcm": {"path": "pcm16k/" + identifier + ".f32", "sample_rate": 16000,
                    "channels": 1, "frames": 1600, "dtype": "<f4", "sha256": w.sha(derived)}}
            write(testcase.inputs / clip["original_wav"]["path"], wav)
            write(testcase.inputs / clip["derived_pcm"]["path"], derived)
            stream.writestr(member, wav); clips.append(clip)
    inputs = {"schema": "active-asr-inputs-v1", "source_archive_sha256": w.sha(archive.read_bytes()),
              "derivation": {"fixture_only": True}, "clips": clips}
    write(testcase.inputs / "inputs.json", inputs); testcase.input_sha = w.sha(encoded(inputs))
    for clip, row in zip(clips, rows):
        text = fixture.plan["cells"][fixture.ids.index(row["cell_id"])]["prompt"]
        for model in w.MODELS:
            testcase.asr_fixture(clip, model, text)
    options = ({"d1_plan": testcase.plan_path, "d1_plan_sha256": fixture.plan_sha, "d1_generation_root": generation}
               if generation else {"d1_checkpoint": fixture.checkpoint_path,
                                   "d1_checkpoint_sha256": pin(fixture.checkpoint_path)["sha256"]})
    output = folder / "view"
    report = w.create_view(testcase.inputs / "inputs.json", testcase.input_sha, archive, testcase.asr, output,
                           fsmn_tokens=testcase.tokens, **options)
    return output, report


class AccountedFixture:
    """All objects in this fixture are explicitly synthetic protocol examples."""

    def __init__(self, testcase):
        base.D1WeakViewTests.setUp(testcase)
        self.root = testcase.root / "accounted-fixture"
        self.root.mkdir()
        self.plan, self.plan_sha = testcase.plan, testcase.plan_sha
        self.exclude_return = testcase._testMethodName == "test_unrecognizable_return_is_preserved_outside_observations"
        self.plan_pin = pin(testcase.plan_path)
        self.ids = [c["cell_id"] for c in self.plan["cells"]]
        self.refs = {r["reference_id"]: r for r in self.plan["references"]}
        self.ledger = self.root / "ledger"
        self.source = self.root / "source"
        names = ["tools/" + name for name in ("active_tts_expand.py", "active_tts_smoke.py",
                 "research_job_lock.py", "research_no_network.py", "research_runtime.py", "research_queue.py")]
        for name in names + ["tools/active_tts_continuation.py", "tools/research_tts_partial.py",
                             "configs/research/active-plan-20261105.json"]:
            write(self.source / name, b"# synthetic source pin, never imported\n")
        self.old_sources = {name: record(self.source / name) for name in names}
        self.new_sources = roster(self.source)
        self.binding = {"plan_canonical_sha256": self.plan_sha, "plan_file_sha256": self.plan_pin["sha256"],
            **{key: "a" * 64 for key in ("model_assets_sha256", "source_manifest_sha256",
                "producer_source_sha256", "runtime_setup_sha256")}, "generation": {"fixture_only": True}}
        self.old = self.root / "old"
        self.old_stage = {"schema": "active-tts-extension-stage-v1", "prepared": str(self.old),
            "source_revision": "43ffea59304121691a84424236543c00a56de588", "target_count": 232,
            "target_cells": self.ids[:232], "initial_reused_cells": self.ids[:176],
            "chunks": [self.ids[176:204], self.ids[204:232]], "maximum_new_calls": 56,
            "stage_budget_minutes": 20, "maximum_job_seconds": 600, "max_retries": 0,
            "training_admitted": False, "paired_cases_included": False, "execution_root": str(self.source),
            "execution_source": {key: value for key, value in self.old_sources.items() if key != "tools/research_queue.py"},
            "binding": self.binding, "ledger": str(self.ledger), "queue_root": str(self.old / "queue"),
            "heavy_lock": str(self.root / "heavy.lock")}
        self.old_stage_pin = write(self.old / "stage.json", self.old_stage)
        jobs = []
        for index in range(2):
            name = "chunk-" + str(index + 1).zfill(3)
            recipe = {"schema": "active-tts-extension-job-v1", "stage": self.old_stage_pin["path"],
                "stage_sha256": self.old_stage_pin["sha256"], "job_id": name, "chunk_index": index,
                "cells": self.old_stage["chunks"][index], "maximum_new_calls": 28, "timeout_seconds": 600}
            recipe_pin = write(self.old / "jobs" / (name + ".json"), recipe)
            jobs.append(self.definition(name, recipe_pin, str(self.source / "tools/active_tts_expand.py"),
                                        [] if index == 0 else ["chunk-001"], []))
        self.old_queue, self.old_request = self.queue(self.old / "queue", jobs, 20)
        prior = []
        for cell in self.plan["cells"][:176]:
            identity = self.identity(cell)
            rp = write(self.ledger / "cells" / (cell["cell_id"] + ".json"), {
                "cell_id": cell["cell_id"], "identity_sha256": identity, "training_admitted": False,
                "status": "GENERATED_PENDING_ASR"})
            prior.append({"cell_id": cell["cell_id"], "record_pin": rp})
        old_recipe_pin = pin(self.old / "queue/recipes/chunk-001.json")
        old_rows = [self.returned(cell, self.old / "queue/jobs/chunk-001/outputs",
                    self.old_stage_pin, old_recipe_pin, "original-232-chunk-001", "original-timed-out",
                    failed=index in (1, 7, 12))
                    for index, cell in enumerate(self.plan["cells"][176:197])]
        unknown = self.unknown(self.plan["cells"][197], self.old_stage_pin, old_recipe_pin)
        self.terminal(self.old / "queue", jobs[0], 0, {}, old=True)
        old_owner = self.owner(self.old / "queue", "chunk-001", self.old_stage_pin,
                               self.old_sources, "original-timed-out")
        audit_pin = write(self.root / "partial-audit.json", {
            "schema": "active-tts-timed-out-partial-independent-readback-v1", "stage": self.old_stage_pin,
            "new_terminal_cells": 21, "unknown_cell_ids": [self.ids[197]], "native_arrays_read_back": 21,
            "derived_pcm_read_back": 18, "model_calls_during_audit": 0, "original_stage_files": roster(self.old)})
        other = write(self.root / "other-declaration.json", {"fixture_only": True})
        self.chunks = [self.ids[198:210], self.ids[210:222], self.ids[222:232],
                       self.ids[232:244], self.ids[244:256]]
        self.protocol = {"schema": "kws-d1-post-timeout-continuation-predeclaration-v1",
            "new_generation_maximum_calls": 58, "pause_after_jobs": 3, "known_returned_cells": 197,
            "old_stage_returned_cells": 21, "old_stage_consumed_calls": 22,
            "first_pause_requires_ASR_accounted232_before_last_two_jobs": True,
            "unknown_receives_asr_or_native_call": False, "unknown_becomes_failed_native": False,
            "unknown_has_observed_text_or_pcm": False, "unknown_is_negative": False,
            "never_claimed_ids": self.ids[198:], "consumed_unknown_ids": [self.ids[197]], "new_chunks": self.chunks,
            "old_tts_stage": self.old_stage_pin, "old_queue_terminal": old_owner["result"],
            "partial_audit": audit_pin, "recovery_contract_review": other, "original_training_protocol": other,
            "budget": {"prior_actual_elapsed_ns": 600186647856, "new_queue_global_minutes": 29.9,
                "new_queue_global_ns": 1794000000000, "combined_allocation_ns": 2400000000000,
                "per_job_maximum_ns": 600000000000, "combined_declared_actual_ceiling_ns": 2394186647856,
                "cleanup_reserve_ns": 5813352144}}
        protocol_pin = write(self.root / "protocol.json", self.protocol)
        self.proof = {"schema": "active-tts-original-partial-proof-v2", "protocol_declaration": protocol_pin,
            "old_stage": self.old_stage_pin, "old_terminal": old_owner["result"], "prior_returned": prior,
            "returned_rows": old_rows, "unobservable_consumed": [unknown], "owner": old_owner,
            "never_claimed_ids_at_registration": self.ids[198:], "binding": self.binding, "plan": self.plan_pin,
            "ledger": str(self.ledger), "old_source_root": str(self.source), "old_source_files": self.old_sources}
        proof_pin = write(self.root / "old-proof.json", self.proof)
        prior_view = self.root / "prior-view"
        prior_report = {"source_adapter": {"plan_sha256": self.plan_sha},
            "items": [{"source": {"source_id": name, "source_plan_sha256": self.plan_sha}} for name in self.ids[:176]]}
        write(prior_view / "report.json", prior_report)
        prior_manifest = write(prior_view / "MANIFEST.json", {"report.json": pin(prior_view / "report.json")["sha256"]})
        entries = [{"path": str(prior_view), "manifest_sha256": prior_manifest["sha256"]}]
        if testcase._testMethodName == "test_accounted_prior_replays_observed_and_keeps_unknown_outside_rows":
            original_ledger = self.ledger
            self.ledger = self.root / "prior-fixture-generation-ledger"
            prior_output = self.root / "prior-fixture-output"
            previous_rows = [self.returned(cell, prior_output, self.old_stage_pin, old_recipe_pin,
                             "prior-fixture", "original-timed-out") for cell in self.plan["cells"][:176]]
            self.ledger = original_ledger
            entries = []
            for offset in range(0, 176, 64):
                package, _ = dense_view(testcase, self, previous_rows[offset:offset + 64],
                                       "prior-batch-" + str(offset), generation=prior_output / "generation")
                entries.append({"path": str(package), "manifest_sha256": pin(package / "MANIFEST.json")["sha256"]})
        prior_pin = write(self.root / "prior176.json", {"schema": "d1-sealed-prior-evidence-v1",
            "prefix_count": 176, "plan_sha256": self.plan_sha, "excluded_exports": [],
            "views": entries})
        self.prepared = self.root / "new"
        self.d = {"schema": "active-tts-unclaimed-continuation-v2", "prepared": str(self.prepared),
            "queue_root": str(self.prepared / "queue"), "execution_root": str(self.source),
            "execution_source": self.new_sources, "protocol_declaration": protocol_pin, "old_partial_proof": proof_pin,
            "old_stage": self.old_stage_pin, "old_terminal": old_owner["result"], "prior_evidence": prior_pin,
            "binding": self.binding, "ledger": str(self.ledger), "heavy_lock": self.old_stage["heavy_lock"],
            "chunks": self.chunks, "consumed_unknown_ids": [self.ids[197]], "maximum_new_calls": 58,
            "new_queue_global_minutes": 29.9, "per_job_maximum_seconds": 600, "combined_allocation_ns": 2400000000000,
            "prior_actual_elapsed_ns": 600186647856, "pause_after_jobs": 3, "max_retries": 0,
            "further_continuation_allowed": False, "paired_cases_included": False, "training_admitted": False,
            "registration_path": str(self.ledger / "continuations" / (self.old_stage_pin["sha256"] + ".json")),
            "continuation_id": "fictional-continuation", "runtime_python": "/fixture/venv/bin/python"}
        for name in self.new_sources:
            write(self.prepared / "execution-source" / name, (self.source / name).read_bytes())
        declaration_pin = write(self.prepared / "declaration.json", self.d)
        new_jobs = []
        for index, chunk in enumerate(self.chunks):
            name = "continuation-" + str(index + 1).zfill(3)
            recipe = {"schema": "active-tts-unclaimed-job-v2", "declaration": declaration_pin["path"],
                "declaration_sha256": declaration_pin["sha256"], "job_id": name, "chunk_index": index,
                "cells": chunk, "maximum_new_calls": len(chunk), "timeout_seconds": 600,
                "requires_asr232_handoff": index >= 3}
            recipe_pin = write(self.prepared / "jobs" / (name + ".json"), recipe)
            outputs = ["summary.json", "inventory.json", "invocation.json", "outcomes.json",
                       "network-guard.json", "model-loaded.json", "generation/plan-lock.json"]
            new_jobs.append(self.definition(name, recipe_pin, str(self.source / "tools/active_tts_continuation.py"),
                            [] if index == 0 else ["continuation-" + str(index).zfill(3)], outputs))
        new_queue, new_request = self.queue(self.prepared / "queue", new_jobs, 29.9)
        new_request_pin = write(self.prepared / "queue-request.json", new_request)
        registration = {"schema": "active-tts-single-continuation-registration-v2",
            "declaration": declaration_pin, "queue_request": new_request_pin, "protocol_declaration": protocol_pin,
            "original_stage_sha256": self.old_stage_pin["sha256"],
            "original_terminal_sha256": old_owner["result"]["sha256"], "never_claimed_cell_ids": self.ids[198:],
            "permanently_consumed_unknown_ids": [self.ids[197]], "maximum_new_calls": 58,
            "automatic_retry": False, "further_continuation_allowed": False,
            "new_queue_budget_ns": 1794000000000, "old_actual_ns": 600186647856, "combined_allocation_ns": 2400000000000}
        registration_pin = write(Path(self.d["registration_path"]), registration)
        write(self.prepared / "registration.json", registration)
        owners, rows, states = {"original-232-chunk-001": old_owner}, list(old_rows), {}
        for index, job in enumerate(new_jobs[:3]):
            name = job["job_id"]; output = self.prepared / "queue/jobs" / name / "outputs"
            rp = pin(self.prepared / "queue/recipes" / (name + ".json"))
            new_rows = [self.returned(self.plan["cells"][self.ids.index(cell_id)], output, declaration_pin, rp,
                name, "continuation-completed", failed=cell_id == self.ids[201]) for cell_id in self.chunks[index]]
            for leaf in ("invocation.json", "network-guard.json", "model-loaded.json"):
                write(output / leaf, {"fixture_only": True})
            write(output / "outcomes.json", {"schema": "active-tts-continuation-outcomes-v2",
                "declaration": declaration_pin, "registration": registration_pin, "job_id": name,
                "job_recipe": pin(self.prepared / "jobs" / (name + ".json")),
                "plan_canonical_sha256": self.plan_sha, "binding": self.binding,
                "selected_cells": self.chunks[index], "new_call_ids": self.chunks[index], "reused_cell_ids": [],
                "rows": new_rows})
            write(output / "inventory.json", {"files": roster(output)})
            write(output / "summary.json", {"schema": "active-tts-continuation-result-v2", "status": "completed",
                "job_recipe_sha256": rp["sha256"], "declaration_sha256": declaration_pin["sha256"],
                "new_generation_calls": len(new_rows), "new_call_ids": self.chunks[index], "reused_cell_ids": [],
                "inventory": record(output / "inventory.json"), "outcomes": record(output / "outcomes.json")})
            deps = {} if index == 0 else {new_jobs[index - 1]["job_id"]: states[new_jobs[index - 1]["job_id"]]}
            self.terminal(self.prepared / "queue", job, index * 1000000, deps)
            owner = self.owner(self.prepared / "queue", name, declaration_pin, self.new_sources, "continuation-completed")
            states[name] = {"receipt": owner["result"]["path"], "receipt_sha256": owner["result"]["sha256"],
                "outputs_root": owner["outputs_root"], "outputs": w.loads(Path(owner["result"]["path"]).read_bytes())["outputs"]}
            owners[name] = owner; rows.extend(new_rows)
        self.checkpoint = {"schema": "active-tts-accounted-checkpoint-v2", "declaration": declaration_pin,
            "registration": registration_pin, "protocol_declaration": protocol_pin, "prior_evidence": prior_pin,
            "old_partial_proof": proof_pin, "plan": self.plan_pin, "plan_canonical_sha256": self.plan_sha,
            "binding": self.binding, "prefix_count": 232, "prior_count": 176, "planned_cell_ids": self.ids[:232],
            "prior_cell_ids": self.ids[:176], "owners": owners, "returned_rows": rows,
            "unobservable_consumed": [unknown], "eligible_cell_ids": [row["cell_id"] for row in rows if row["eligibility"]["eligible"]],
            "excluded_native_rows": [{"cell_id": row["cell_id"], "reason": row["eligibility"]["reason"],
                "ledger_record_sha256": row["ledger_record_sha256"]} for row in rows if not row["eligibility"]["eligible"]],
            "accounted_cell_ids": self.ids[:232], "previous_checkpoint": None,
            "new_model_calls_during_checkpoint": 0, "training_admitted": False,
            "budget": {"old_actual_elapsed_ns": 600186647856, "new_actual_known_ns": 3000000,
                "combined_actual_known_ns": 600189647856, "new_actual_total_complete": True,
                "new_queue_allocation_ns": 1794000000000, "combined_allocation_ns": 2400000000000,
                "combined_declared_ceiling_ns": 2394186647856, "cleanup_reserve_ns": 5813352144,
                "new_queue_overrun_ns": 0, "observed_combined_overrun_ns": 0, "selected_job_count": 3,
                "completed_job_count": 3, "within_new_queue_allocation": True, "within_combined_allocation": True,
                "historical_original_smoke_actual_total_ns": None, "historical_original_smoke_actual_total_complete": False}}
        self.checkpoint_path = self.root / "checkpoint.json"
        write(self.checkpoint_path, self.checkpoint)

    def identity(self, cell):
        return w.sha(w.canonical({"binding": self.binding, "cell": cell, "reference": self.refs[cell["reference_id"]]}))

    def definition(self, name, recipe, tool, deps, outputs):
        return {"job_id": name, "recipe": recipe["path"], "recipe_sha256": recipe["sha256"],
            "argv": ["/fixture/venv/bin/python", "-I", "-B", tool, "run", "--recipe", "{recipe}",
                     "--output", "{output_dir}", "--heavy-lock", str(self.root / "heavy.lock")],
            "cwd": str(self.source), "depends_on": deps, "timeout_minutes": 10, "outputs": outputs}

    def queue(self, root, jobs, budget):
        request = {"schema": "kws-local-research-queue-v1", "queue_id": "fictional-continuation",
                   "global_budget_minutes": budget, "jobs": jobs}
        request_pin = write(root / "request.json", request)
        active = write(root / "active-plan.json", {"fixture_only": True})
        queue = {"schema": request["schema"], "request_sha256": request_pin["sha256"],
            "active_plan_sha256": active["sha256"], "automatic_retries": 0, "historical_jobs_imported": 0,
            "heavy_lock": str(self.root / "heavy.lock"),
            "source_sha256": {name: self.old_sources["tools/" + name]["sha256"]
                              for name in ("research_queue.py", "research_job_lock.py")}}
        write(root / "queue.json", queue)
        for job in jobs:
            write(root / "recipes" / (job["job_id"] + ".json"), Path(job["recipe"]).read_bytes())
        return queue, request

    def terminal(self, root, job, before, deps, *, old=False):
        folder = root / "jobs" / job["job_id"]
        write(folder / "dependencies.json", deps)
        write(folder / "stdout.log", b"fictional queue fixture\n"); write(folder / "stderr.log", b"")
        binding = {"job_id": job["job_id"], "queue_sha256": pin(root / "queue.json")["sha256"],
                   "job_definition_sha256": w.sha(encoded(job)), "recipe_sha256": job["recipe_sha256"]}
        claim = {"schema": "kws-local-research-job-claim-v1", **binding, "attempt": 1, "automatic_retry": False,
            "dependencies_sha256": pin(folder / "dependencies.json")["sha256"],
            "allowed_runner_nanoseconds": 600000000000, "started_monotonic_ns": 1,
            "deadline_monotonic_ns": 600000000001, "global_actual_before_nanoseconds": before}
        cp = write(folder / "claim.json", claim)
        for name in ("worker-started.json", "command-started.json"):
            write(folder / name, {"claim_sha256": cp["sha256"]})
        write(folder / "result.json", {"schema": "kws-local-research-job-result-v1", **binding,
            "claim_sha256": cp["sha256"], "automatic_retry": False, "actual_duration_known": True,
            "actual_elapsed_ns": 600186647856 if old else 1000000, "allowed_runner_nanoseconds": 600000000000,
            "cwd": str(self.source), "argv": [s.replace("{recipe}", str(root / "recipes" / (job["job_id"] + ".json")))
                .replace("{output_dir}", str(folder / "outputs")) for s in job["argv"]],
            "logs": {name: record(folder / name) for name in ("stdout.log", "stderr.log")},
            "outputs": {} if old else {name: record(folder / "outputs" / name) for name in job["outputs"]},
            "status": "timed_out" if old else "completed", "exit_code": -9 if old else 0,
            "reason": "actual_runner_deadline" if old else "command_exit"})

    def owner(self, root, name, stage, sources, origin):
        folder = root / "jobs" / name
        return {"origin": origin, "job_id": name, "stage": stage,
            "job_recipe": pin(root / "recipes" / (name + ".json")), "queue_root": str(root),
            "queue_files": {leaf: pin(root / leaf) for leaf in ("queue.json", "request.json", "active-plan.json")},
            "job_root": str(folder), "job_files": roster(folder), "result": pin(folder / "result.json"),
            "outputs_root": str(folder / "outputs"), "source_root": str(self.source), "source_files": sources}

    def claim_and_start(self, cell, output, stage, recipe):
        name, ref = cell["cell_id"], self.refs[cell["reference_id"]]
        job_id = Path(recipe["path"]).stem
        claim = {"schema": "active-tts-plan-call-claim-v1", "cell_id": name, "identity_sha256": self.identity(cell),
            "attempt": 1, "owner": {"stage_sha256": stage["sha256"], "job_id": job_id,
            "job_recipe_sha256": recipe["sha256"]}, "started_at": "2026-11-05T00:00:00+00:00"}
        cp = write(self.ledger / "claims" / (name + ".json"), claim)
        checked = ref["reference_text_status"] in ("verified", "dataset-provided-content-checked")
        write(output / "native" / name / "call.json", {"schema": "active-tts-generation-call-v1", "cell_id": name,
            "reference_id": ref["reference_id"], "source_family_id": ref["source_family_id"],
            "reference_wav_sha256": ref["wav_sha256"], "reference_pcm_sha256": ref["pcm_sha256"],
            "reference_text_status": ref["reference_text_status"], "reference_mode": "ICL" if checked else "x-vector-only",
            "reference_text_passed": checked, "seed": cell["seed"], "intended_text": cell["prompt"],
            "generation": self.binding["generation"], "reference_prompt_seconds": 0.0, "attempt": 1})
        write(output / "generation/cells" / name / "started.json", {"schema": "kws-bulk-clone-start-v1",
            "cell_id": name, "reference_id": ref["reference_id"], "reference_wav_sha256": ref["wav_sha256"],
            "plan_sha256": self.plan_sha, "source_family_id": ref["source_family_id"], "seed": cell["seed"],
            "attempts": 1, "training_admitted": False})
        write(output / "generation/plan-lock.json", {"schema": "kws-bulk-clone-execution-lock-v1",
            "plan_sha256": self.plan_sha, "training_admitted": False, "model_id": self.plan["model_id"]})
        return claim, cp

    def returned(self, cell, output, stage, recipe, owner_id, origin, *, failed=False):
        name = cell["cell_id"]
        claim, cp = self.claim_and_start(cell, output, stage, recipe)
        index = self.ids.index(name)
        excluded = self.exclude_return and index == 202
        failed = failed or excluded
        samples = [((j % 31) - 15) * (index + 1) / 100000 for j in range(2400)]
        if excluded:
            samples[0] = float("nan")
        pcm = struct.pack("<2400f", *samples); wav = float_wav(pcm)
        native, generated = output / "native" / name, output / "generation/cells" / name
        write(native / "returned-array.npy", npy(pcm))
        if not excluded:
            write(native / "returned-native.wav", wav)
        write(native / "native.json", {"schema": "active-tts-native-return-v1",
            "array": record(native / "returned-array.npy"), "wav": None if excluded else record(native / "returned-native.wav"),
            "dtype": "<f4", "shape": [2400], "sample_rate_hz": 24000, "samples_sha256": w.sha(pcm),
            "all_finite": not excluded, "technical_gate_applied": False,
            **({"wav_unavailable_reason": "NONFINITE_ARRAY"} if excluded else {})})
        write(native / "returned.json", {"generation_seconds": .001, "total_seconds": .002,
            "native_stop_reason": "not_exposed_by_voice_clone_API"})
        if failed:
            write(generated / "failed.json", {"schema": "kws-bulk-clone-failure-v1", "cell_id": name,
                "status": "FAILED_NO_RETRY", "error_type": "ValueError", "training_admitted": False})
        else:
            write(generated / "native-f32.wav", wav)
            derived = struct.pack("<1600f", *(samples[j * 3 // 2] for j in range(1600)))
            old_wav, measured = w.quantize_pcm16(derived, 1600)
            write(generated / "audio-16k.wav", old_wav)
            write(generated / "completed.json", {"schema": "kws-bulk-clone-complete-v1", "cell_id": name,
                "status": "GENERATED_PENDING_ASR", "source_family_id": cell["source_family_id"],
                "intended_kind": cell["intended_kind"], "actual_text": None,
                "training_admitted": False, "qualification_allowed": False,
                "technical": {"native_frames": 2400, "native_rate": 24000, "duration_s": .1,
                    "technical_audio_pass": True},
                "audio": {"native_sha256": w.sha(wav), "native_bytes": len(wav),
                    "audio_frames": 1600, "sample_rate_hz": 16000, "wav_sha256": w.sha(old_wav),
                    "wav_bytes": len(old_wav), "pcm_sha256": measured["pcm_sha256"]}})
        value = {"schema": "active-tts-plan-cell-v1", "origin": "extension-call", "cell_id": name,
            "identity_sha256": self.identity(cell), "claim_sha256": cp["sha256"], "new_generation_calls": 1,
            "training_admitted": False, "status": "FAILED_NO_RETRY" if failed else "GENERATED_PENDING_ASR",
            "technical_failure_reason": "FICTIONAL_GATE_FAILURE" if failed else None,
            "output_root": str(output), "generation_root": str(output / "generation"),
            "native_wav": (None if excluded else "native/" + name + "/returned-native.wav" if failed else
                          "generation/cells/" + name + "/native-f32.wav"),
            "files": {p.relative_to(output).as_posix(): record(p) for folder in (native, generated)
                      for p in folder.rglob("*") if p.is_file()}}
        rp = write(self.ledger / "cells" / (name + ".json"), value)
        return {"cell_id": name, "origin": origin, "owner_id": owner_id, "ledger_record_sha256": rp["sha256"],
            "record_pin": rp, "claim_pin": cp, "record": value, "claim": claim,
            "eligibility": ({"eligible": False, "reason": "NO_DECODABLE_NATIVE_WAV"} if excluded else
                            {"eligible": True, "reason": None, "frames": 2400, "native_wav_sha256": w.sha(wav)})}

    def unknown(self, cell, stage, recipe):
        output = self.old / "queue/jobs/chunk-001/outputs"; name = cell["cell_id"]
        claim, cp = self.claim_and_start(cell, output, stage, recipe)
        artifacts = {p.relative_to(output).as_posix(): record(p)
                     for folder in (output / "native" / name, output / "generation/cells" / name)
                     for p in folder.rglob("*") if p.is_file()}
        absent = [str(self.ledger / "cells" / (name + ".json"))] + [
            str(output / "native" / name / leaf) for leaf in
            ("returned-array.npy", "returned-native.wav", "native.json", "returned.json")] + [
            str(output / "generation/cells" / name / leaf) for leaf in
            ("completed.json", "failed.json", "native-f32.wav", "audio-16k.wav")]
        return {"cell_id": name, "origin": "original-timed-out", "status": "unobservable-consumed",
            "owner_id": "original-232-chunk-001", "identity_sha256": self.identity(cell),
            "claim_pin": cp, "claim": claim, "output_root": str(output), "artifacts": artifacts,
            "absent_paths": absent, "model_return_known": False, "may_retry": False, "asr_calls": 0,
            "native_system_calls": 0, "observed_text": None, "training_admitted": False}


class AccountedTests(unittest.TestCase):
    asr_fixture = base.D1WeakViewTests.asr_fixture

    def setUp(self):
        self.fixture = AccountedFixture(self)

    def replay(self, value=None):
        raw = encoded(value or self.fixture.checkpoint)
        return w._d1_accounted_checkpoint(raw, w.sha(raw), w._D1CheckpointEvidence())

    def test_mixed_owners_real_formats_and_unknown_are_distinct(self):
        value, plan, refs, sources, prior = self.replay()
        self.assertEqual(len(sources), 55)
        self.assertEqual(value["prefix_count"], 232)
        self.assertEqual(len(prior["views"]), 1)
        self.assertNotIn(self.fixture.ids[197], sources)
        self.assertEqual(sum(row["generation_outcome"] == "failed-native" for row in sources.values()), 4)
        self.assertEqual(value["owners"]["original-232-chunk-001"]["origin"], "original-timed-out")

    def test_wrong_owner_and_duplicate_return_rejected(self):
        changed = copy.deepcopy(self.fixture.checkpoint)
        changed["returned_rows"][0]["owner_id"] = "continuation-001"
        with self.assertRaisesRegex(ValueError, "OLD_ROWS_UNCHANGED"):
            self.replay(changed)
        changed = copy.deepcopy(self.fixture.checkpoint)
        changed["returned_rows"][-1] = changed["returned_rows"][0]
        with self.assertRaisesRegex(ValueError, "RETURN_ROSTER"):
            self.replay(changed)

    def test_unknown_return_or_promotion_is_rejected(self):
        unknown = self.fixture.checkpoint["unobservable_consumed"][0]
        write(Path(unknown["absent_paths"][1]), b"an illicit returned array")
        with self.assertRaisesRegex(ValueError, "EXACT_OWNER_FILE_ROSTER"):
            self.replay()

    def test_old_timeout_cannot_be_relabelled_completed(self):
        changed = copy.deepcopy(self.fixture.checkpoint)
        changed["owners"]["original-232-chunk-001"]["origin"] = "continuation-completed"
        with self.assertRaisesRegex(ValueError, "OLD_ROWS_UNCHANGED"):
            self.replay(changed)

    def test_cumulative_budget_is_not_a_static_summary(self):
        changed = copy.deepcopy(self.fixture.checkpoint)
        changed["budget"]["new_actual_known_ns"] += 1
        with self.assertRaisesRegex(ValueError, "ACTUAL_BUDGET"):
            self.replay(changed)

    def test_unknown_cannot_be_substituted_with_another_audio_for_asr(self):
        f = self.fixture
        returned = f.checkpoint["returned_rows"][0]["record"]
        raw = (Path(returned["output_root"]) / returned["native_wav"]).read_bytes()
        pcm = w.float_wav_pcm(raw)
        archive = io.BytesIO()
        member = "cells/" + f.ids[197] + "/native-f32.wav"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr(member, raw)
        body = archive.getvalue()
        inputs = {"source_archive_sha256": w.sha(body), "clips": [{"id": "clip-000",
            "original_member": member, "original_wav": {"sha256": w.sha(raw), "bytes": len(raw)},
            "original_pcm": {"sha256": w.sha(pcm), "frames": 2400}}]}
        checkpoint = encoded(f.checkpoint)
        with self.assertRaisesRegex(ValueError, "NEW_OBSERVABLE_ONLY"):
            w.d1_source_rows_v3(body, inputs, checkpoint, w.sha(checkpoint))

    def test_unrecognizable_return_is_preserved_outside_observations(self):
        value, _, _, sources, _ = self.replay()
        self.assertEqual(len(value["returned_rows"]), 55)
        self.assertEqual(len(sources), 54)
        self.assertEqual(value["excluded_native_rows"], [{"cell_id": self.fixture.ids[202],
            "reason": "NO_DECODABLE_NATIVE_WAV",
            "ledger_record_sha256": next(row["ledger_record_sha256"] for row in value["returned_rows"]
                                        if row["cell_id"] == self.fixture.ids[202])}])
        self.assertNotIn(self.fixture.ids[202], sources)
        self.assertEqual(value["unobservable_consumed"][0]["cell_id"], self.fixture.ids[197])

    def test_accounted_prior_replays_observed_and_keeps_unknown_outside_rows(self):
        f = self.fixture
        previous = w.loads(Path(f.checkpoint["prior_evidence"]["path"]).read_bytes())
        views = list(previous["views"])
        for offset in (0, 28):
            package, _ = dense_view(self, f, f.checkpoint["returned_rows"][offset:offset + 28],
                                    "current-batch-" + str(offset))
            views.append({"path": str(package), "manifest_sha256": pin(package / "MANIFEST.json")["sha256"]})
        prior = {"schema": "d1-accounted-prior-evidence-v2", "plan_sha256": f.plan_sha,
            "plan_file_sha256": f.plan_pin["sha256"], "accounted_prefix_count": 232, "observed_source_count": 231,
            "views": views, "parent_evidence": f.checkpoint["prior_evidence"],
            "continuation_checkpoint": pin(f.checkpoint_path),
            "unobservable_consumed": f.checkpoint["unobservable_consumed"], "excluded_returned": []}
        path = f.root / "accounted-prior.json"; frozen = write(path, prior)
        result = w.verify_accounted_prior(path, frozen["sha256"], expected_count=256)
        self.assertEqual(result["schema"], "kws-d1-accounted-prior-verification-v2")
        self.assertEqual(result["observed_ids"], f.ids[:197] + f.ids[198:232])
        self.assertEqual(result["pending_ids"], f.ids[232:])
        self.assertFalse(result["accounted_complete"])
        self.assertFalse(result["complete_terminal_coverage"])
        self.assertEqual(result["unobservable_consumed"][0]["status"], "unobservable-consumed")
        self.assertEqual(result["unobservable_consumed"][0]["original_proof"], f.proof["unobservable_consumed"][0])
        self.assertEqual(result["counts"], {"observed": 231, "excluded_returned": 0, "unobservable_consumed": 1, "pending": 24})
        self.assertEqual(result["new_model_calls"], 0)
        self.assertTrue(set(w.ACCOUNTED_SOURCE_PATHS) <= {
            str(Path(row["path"]).relative_to(w.ROOT)) for row in result["evidence_files"]
            if Path(row["path"]).is_relative_to(w.ROOT)})
        complete = w.verify_accounted_prior(path, frozen["sha256"], expected_count=232)
        self.assertTrue(complete["accounted_complete"])
        self.assertFalse(complete["complete_terminal_coverage"])
        self.assertEqual(complete["pending_ids"], [])

    def test_retained_v3_full_view_replays_after_original_roots_removed(self):
        f = self.fixture
        self.inputs = self.root / "new-inputs"; self.asr = self.root / "new-asr"
        self.inputs.mkdir(); self.asr.mkdir()
        archive = self.root / "new-native.zip"; clips = []
        chosen = f.checkpoint["returned_rows"][:28]
        with zipfile.ZipFile(archive, "w") as stream:
            for index, row in enumerate(chosen):
                name = row["cell_id"]; value = row["record"]
                wav = (Path(value["output_root"]) / value["native_wav"]).read_bytes()
                pcm = w.float_wav_pcm(wav)
                samples = [r[0] for r in struct.iter_unpack("<f", pcm)]
                derived = struct.pack("<1600f", *(samples[j * 3 // 2] for j in range(1600)))
                identifier = "clip-" + str(index).zfill(3)
                member = "cells/" + name + "/native-f32.wav"
                clip = {"id": identifier, "original_member": member,
                    "original_wav": {"path": "original/" + identifier + ".wav", "sha256": w.sha(wav), "bytes": len(wav)},
                    "original_pcm": {"sample_rate": 24000, "channels": 1, "frames": 2400,
                        "dtype": "<f4", "sha256": w.sha(pcm)},
                    "derived_pcm": {"path": "pcm16k/" + identifier + ".f32", "sample_rate": 16000,
                        "channels": 1, "frames": 1600, "dtype": "<f4", "sha256": w.sha(derived)}}
                write(self.inputs / clip["original_wav"]["path"], wav)
                write(self.inputs / clip["derived_pcm"]["path"], derived)
                stream.writestr(member, wav); clips.append(clip)
        inputs = {"schema": "active-asr-inputs-v1", "source_archive_sha256": w.sha(archive.read_bytes()),
                  "derivation": {"fixture_only": True}, "clips": clips}
        write(self.inputs / "inputs.json", inputs)
        self.input_sha = w.sha(encoded(inputs))
        for clip, row in zip(clips, chosen):
            text = f.plan["cells"][f.ids.index(row["cell_id"])]["prompt"]
            for model in w.MODELS:
                self.asr_fixture(clip, model, text)
        options = {"d1_checkpoint": f.checkpoint_path, "d1_checkpoint_sha256": pin(f.checkpoint_path)["sha256"]}
        mapping = w.build_d1_source_map(self.inputs / "inputs.json", self.input_sha, archive, **options)
        self.assertEqual(len(mapping["rows"]), 28)
        self.assertNotIn(f.ids[197], [r["source_id"] for r in mapping["rows"]])
        package = self.root / "view-accounted"
        report = w.create_view(self.inputs / "inputs.json", self.input_sha, archive, self.asr, package,
                               fsmn_tokens=self.tokens, **options)
        self.assertEqual(report["source_adapter"]["schema"], w.D1_ADAPTER_V3)
        self.assertEqual(set(report["producer"]["source_files_sha256"]), set(w.ACCOUNTED_SOURCE_PATHS))
        before = {p: p.read_bytes() for p in package.rglob("*") if p.is_file()}
        shutil.rmtree(f.root)
        train = list(w.load_training_rows(package))
        dev = list(w.load_development_rows(package))
        self.assertTrue(all(r["source_id"] != f.ids[197] for r in train + dev))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        failed = [r for r in report["items"] if r["source"]["generation_outcome"] == "failed-native"]
        self.assertEqual(len(failed), 4)
        self.assertTrue(all(r["decision"]["ctc_weight"] == 0 and
                            r["decision"]["keyword_mask"] == [False, False] for r in failed))


if __name__ == "__main__":
    unittest.main()
