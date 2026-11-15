#!/usr/bin/env python3
"""Invented files and mocked native processes only; no project model/audio."""

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "pool_consumer", Path(__file__).with_name("research_v4_native_c_pool.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class PoolConsumerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.header = b"invented analytic fixture; no trained weights\n"
        self.trace = b"EN63TRC1" + struct.pack("<II", 1, 168) + bytes(2206528 - 16)
        for target, changes in (
            (C.ANCESTOR, {"header_sha256": C.core.sha256(self.header)}),
            (C.BASELINE_TRACE_SHA, {kind: C.core.sha256(self.trace) for kind in ("gru", "dsc_tcn")}),
        ):
            patcher = mock.patch.dict(target, changes)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.request = copy.deepcopy(C.FIXED) | {
            "pipeline_commit": "a" * 40,
            "reason": "Invented JSON fixture; no audio decoding or native model execution.",
        }
        self.path = self.root / (C.REQUEST_ID + ".json")
        self.write(self.path, self.request)
        self.digest = C.core.sha256(self.path.read_bytes())

    def write(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def environment(self, attempt="1", private=False, event_name="push"):
        event = self.root / "event.json"
        self.write(event, {"repository": {"private": private}})
        return mock.patch.dict(os.environ, {
            "GITHUB_EVENT_PATH": str(event), "GITHUB_EVENT_NAME": event_name,
            "GITHUB_REPOSITORY": C.core.REPOSITORY, "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456", "GITHUB_RUN_ATTEMPT": attempt,
            "GITHUB_ACTIONS": "true", "GITHUB_SHA": "b" * 40,
        })

    def archive(self, members):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as handle:
            for name, body in members.items():
                handle.writestr(name, body)
        return out.getvalue()

    def execution_fixture(self):
        state = self.root / "execute"
        (state / "result").mkdir(parents=True)
        (state / "claim").mkdir()
        with self.environment():
            self.write(state / "claim/CLAIM.json", C.claim_record(self.request, self.digest))
        (state / "result/REQUEST.json").write_bytes(self.path.read_bytes())
        pipeline = self.root / "pipeline"
        script = pipeline / C.ENTRYPOINT
        script.parent.mkdir(parents=True)
        script.write_text("# inert fixture; never executed\n")
        return state, pipeline, script.read_bytes()

    def result_fixture(self, *, benefit=False):
        state = self.root / ("benefit" if benefit else "no-benefit")
        root = state / "result/pool-profile"
        root.mkdir(parents=True)
        outer = root.parent
        (outer / "REQUEST.json").write_bytes(self.path.read_bytes())
        for name in ("execute.log", "verify.log", "runtime.txt"):
            (outer / name).write_text("invented fixture; no execution\n")
        self.write(outer / "EXECUTION.json", {
            "schema": "kws-research-v4-native-c-pool-execution-v1", "request_sha256": self.digest,
            "execute_exit_code": 0, "verify_exit_code": 0, "optimizer_updates": 0,
            "source_inputs_before_after_verified": True,
        })
        (root / "en63_generated.h").write_bytes(self.header)
        (root / "native.stderr").write_text("")
        for variant in ("baseline", "candidate"):
            (root / ("en63_pool_" + variant)).write_bytes(b"\x7fELFinvented-never-executed-" + variant.encode())
            for kind in ("gru", "dsc_tcn"):
                (root / (kind + "-" + variant + ".trace")).write_bytes(self.trace)
        source_bodies = {name: ("# invented source fixture: " + name + "\n").encode() for name in C.SOURCE_FILES}
        (root / "SOURCE.zip").write_bytes(self.archive(source_bodies))
        source = {name: C.core.sha256(raw) for name, raw in source_bodies.items()}
        binaries = {variant: C.core.sha256((root / ("en63_pool_" + variant)).read_bytes())
                    for variant in ("baseline", "candidate")}
        run = {"repository": C.core.REPOSITORY, "head": "b" * 40, "run_id": "123456", "run_attempt": "1"}
        build = {"schema": "english63-pool-native-build-v1", "compiler": "/usr/bin/gcc",
                 "compiler_realpath": "/invented/compiler", "compiler_sha256": "d" * 64,
                 "compiler_version": "invented fixture compiler", "source_root": "/invented/pipeline",
                 "output_root": "/invented/result/pool-profile", "source_sha256": source,
                 "source_header_sha256": C.ANCESTOR["header_sha256"], "public_run": run,
                 "pipeline_head": "a" * 40, "compile_flags": C.FLAGS,
                 "machine": "x86_64", "libc": ["invented", "0"], "physical_ssc305": False, "compiles": []}
        for variant, macro in (("baseline", 0), ("candidate", 1)):
            here = Path(build["source_root"]) / "research/english63_online_c"
            command = [build["compiler"], *C.FLAGS, "-DEN63_POOL_TIME_MAJOR=" + str(macro),
                       "-I", str(here), "-I", build["output_root"], str(here / "en63_stream.c"),
                       str(here / "pool_profile.c"), "-lm", "-o", build["output_root"] + "/en63_pool_" + variant]
            build["compiles"].append({"variant_name": variant, "variant": macro, "command": command,
                                      "exit_code": 0, "stdout": "", "stderr": "", "binary_sha256": binaries[variant]})
        self.write(root / "BUILD.json", build)
        rows = []
        for index in range(63):
            action = ("TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY" if index < 30 else
                      "EXPOSED_DEVELOPMENT_NO_TUNE" if index < 42 else "QUARANTINE_AMBIGUOUS_NEARPHONE")
            rows.append({"cell_id": "invented-%03d" % index, "source_samples": 32000 if index < 62 else 117760,
                         "test_samples_used": 0, "action": action, "human_gold": False, "training_admitted": False})
        parent_json = {"TERMINAL.json": {"files_sha256": {"invented": "c" * 64}},
                       "REQUEST.json": {"fixture": "invented parent"},
                       "state-contracts/REPLAY.json": {"fixture": "invented saved replay"},
                       "state-contracts/INPUTS.json": {"rows": rows}}
        parent = {}
        for name, value in parent_json.items():
            raw = (json.dumps(value, sort_keys=True) + "\n").encode()
            parent[name] = {"raw_utf8": raw.decode(), "bytes": len(raw), "sha256": C.core.sha256(raw)}
        patcher = mock.patch.dict(C.PARENT_EVIDENCE, {name: value["sha256"] for name, value in parent.items()})
        patcher.start()
        self.addCleanup(patcher.stop)
        claim = {"schema": "kws-research-v4-native-c-pool-start-claim-v1", "request_sha256": self.digest,
                 "pipeline_commit": "a" * 40, "source_state_artifact_id": C.ANCESTOR["artifact_id"],
                 "run_id": "123456", "run_attempt": "1", "optimizer_updates": 0,
                 "status": "CONSUMED_BEFORE_PAIRED_NATIVE_C_POOL_EXPERIMENT"}
        claim_raw = json.dumps(claim, sort_keys=True) + "\n"
        common = {"native_state_ancestor": C.ANCESTOR, "source_header_sha256": C.ANCESTOR["header_sha256"],
                  "source_tts_zip_sha256": C.core.INPUTS["tts"]["zip_sha256"],
                  "source_review_json_sha256": C.core.INPUTS["review"]["member_sha256"],
                  "source_weights_sha256": C.native.CHECKPOINT["weights_sha256"]}
        inputs = common | {"schema": "english63-pool-profile-inputs-v1", "source_audio_clips": 63,
                 "source_pcm_samples": 2101760, "unique_input_samples": 2101760, "test_pcm_bytes": 4203520,
                 "test_pcm_sha256": "e" * 64, "pcm_operation": "byte-preserving-full63-concatenation",
                 "new_audio_generation_or_resampling": False, "parent_evidence": parent,
                 "rows": [{**row, "test_samples_used": row["source_samples"]} for row in rows],
                 "request_raw_utf8": self.path.read_text(), "request_sha256": self.digest,
                 "claim_raw_utf8": claim_raw, "claim_sha256": C.core.sha256(claim_raw.encode())}
        self.write(root / "INPUTS.json", inputs)
        scope = {**dict.fromkeys(C.FALSE_FIELDS, False), **dict.fromkeys(C.UNMEASURED, "NOT_RUN"),
                 "new_independent_audio_samples": 0, "new_independent_audio_seconds": 0,
                 "optimizer_updates": 0, "new_tts_asr_ipa_calls": 0}
        resource_scope = "Linux-host-full-feed-chain-with-constant-signature-not-SSC305-or-acoustic-quality"
        timed_region = "preloaded-pcm-en63_feed-and-constant-signature-callback"
        protocol = common | scope | {"schema": "english63-pool-profile-protocol-v1", "experiment": C.EXPERIMENT,
                    "profile": C.PROFILE, "pass_order": C.schedule(), "baseline_trace_sha256": C.BASELINE_TRACE_SHA,
                    "compile_flags": C.FLAGS, "no_benefit_is_valid_completed_experiment": True,
                    "no_result_driven_extra_passes_or_retries": True, "timed_region": timed_region,
                    "per_feed_quantile": "nearest-rank-ceil(p*N)-1-on-sorted-positive-wall-nanoseconds",
                    "resource_scope": resource_scope}
        self.write(root / "PROTOCOL.json", protocol)
        integrity = {"source_before": source, "source_after": source,
                     "header_before": C.ANCESTOR["header_sha256"], "header_after": C.ANCESTOR["header_sha256"],
                     "tts_before": common["source_tts_zip_sha256"], "tts_after": common["source_tts_zip_sha256"],
                     "review_before": common["source_review_json_sha256"], "review_after": common["source_review_json_sha256"],
                     "pcm_before": "e" * 64, "pcm_after": "e" * 64,
                     "parent_terminal_before": C.ANCESTOR["terminal_sha256"], "parent_terminal_after": C.ANCESTOR["terminal_sha256"],
                     "parent_files_before": parent_json["TERMINAL.json"]["files_sha256"],
                     "parent_files_after": parent_json["TERMINAL.json"]["files_sha256"],
                     "binaries_before": binaries, "binaries_after": binaries,
                     "request_before": self.digest, "request_after": self.digest,
                     "claim_before": inputs["claim_sha256"], "claim_after": inputs["claim_sha256"]}
        measurements = {"schema": "english63-pool-profile-measurements-v1", "public_run": run,
                        "pipeline_head": "a" * 40, "request_sha256": self.digest,
                        "initial_allowed_cpus": [2, 3], "selected_cpu": 2, "final_allowed_cpus": [2, 3],
                        "native_group_elapsed_seconds": 20.0, "source_integrity": integrity, "passes": []}
        signature = 14695981039346656037
        for byte in self.trace[16:]:
            signature = ((signature ^ byte) * 1099511628211) & ((1 << 64) - 1)
        signature = format(signature, "016x")
        candidate_cpu = 180000000 if benefit else 196000000
        timed = {kind: [] for kind in ("gru", "dsc_tcn")}
        for plan in C.schedule():
            is_timed = plan["phase"] == "timed"
            feed = 10000 if plan["variant"] == 0 else 10100
            cpu = 200000000 if plan["variant"] == 0 else candidate_cpu
            body = {"schema": "english63-pool-native-pass-v1", "kind": plan["kind"], "variant": plan["variant"],
                    "phase": plan["phase"], "affinity_cpu": 2, "affinity_single_cpu": True,
                    "pcm_samples": 2101760, "frames": 13134, "feed_calls": 13136, "proposal_count": 0,
                    "signature64": signature, "state_bytes": 38192,
                    "weight_parameter_bytes": (6531 if plan["kind"] == "gru" else 4803) * 4,
                    "frontend_table_bytes": 3894, "input_pcm_bytes": 4203520,
                    "cpu_ns": cpu if is_timed else 0, "wall_ns": 300000000 if is_timed else 0,
                    "feed_ns": [feed] * 13136 if is_timed else [], "trace_io_enabled": plan["phase"] == "trace",
                    "eof_padding_samples": 0, "eof_events": 0, "inference_end_high_water_rss_kib": 1000,
                    "timed_region": timed_region}
            lifetime = {"schema": "english63-pool-child-lifetime-v1", "child_exit_code": 0,
                        "child_signaled": False, "process_lifetime_peak_rss_kib": 1100,
                        "child_user_us": 300000, "child_system_us": 1000}
            command = [build["output_root"] + "/en63_pool_" + plan["variant_name"], "--supervise", plan["kind"],
                       plan["phase"], "/invented/full-original.pcm", build["output_root"] + "/" + plan["kind"] + "-" +
                       plan["variant_name"] + ".trace" if plan["phase"] == "trace" else "-", "2"]
            record = plan | {"status": "COMPLETED", "exit_code": 0, "timed_out": False, "command": command,
                            "raw_stdout": json.dumps(body) + "\n" + json.dumps(lifetime) + "\n", "raw_stderr": "",
                            "worker": body, "lifetime": lifetime, "binary_sha256": binaries[plan["variant_name"]]}
            measurements["passes"].append(record)
            if is_timed:
                timed[plan["kind"]].append({"run_index": plan["run_index"], "pair": plan["pair"], "variant": plan["variant"],
                    "variant_name": plan["variant_name"], "cpu_ns": cpu, "wall_ns": 300000000,
                    "feed_p50_ns": feed, "feed_p95_ns": feed, "feed_p99_ns": feed, "feed_max_ns": feed,
                    "state_bytes": 38192, "inference_end_high_water_rss_kib": 1000,
                    "process_lifetime_peak_rss_kib": 1100})
        self.write(root / "MEASUREMENTS.json", measurements)
        per_kind = []
        reduction = 1 - candidate_cpu / 200000000
        for kind in ("gru", "dsc_tcn"):
            pairs = [{"pair": pair, "order": C.PROFILE["timed_pair_orders"][pair], "baseline_cpu_ns": 200000000,
                      "candidate_cpu_ns": candidate_cpu, "baseline_p99_ns": 10000, "candidate_p99_ns": 10100,
                      "cpu_reduction": reduction, "p99_ratio": 1.01} for pair in range(6)]
            per_kind.append({"kind": kind, "source_weights_sha256": C.native.CHECKPOINT["weights_sha256"][kind],
                "bit_identity": True, "baseline_trace_sha256": C.BASELINE_TRACE_SHA[kind],
                "candidate_trace_sha256": C.BASELINE_TRACE_SHA[kind], "signature64": signature, "proposal_count": 0,
                "state_bytes": 38192, "state_bytes_increase": 0, "timed_passes": timed[kind], "pairs": pairs,
                "paired_cpu_reductions": [reduction] * 6, "paired_cpu_reduction_median": reduction,
                "paired_cpu_reduction_range": [reduction, reduction], "positive_cpu_pairs": 6,
                "paired_p99_ratios": [1.01] * 6, "paired_p99_ratio_median": 1.01,
                "decision": "BENEFIT" if benefit else "NO_BENEFIT", "gate_observations": {
                    "median_cpu_reduction": benefit, "positive_cpu_pairs": True, "paired_p99": True,
                    "state_not_increased": True, "bit_identity": True}})
        summary = common | scope | {"schema": "english63-pool-profile-result-v1", "status": "SUCCEEDED",
            "experiment": C.EXPERIMENT, "decision": "BENEFIT" if benefit else "NO_BENEFIT", "per_kind": per_kind,
            "source_training_run_id": 43595338774, "source_audio_clips": 63, "source_pcm_samples": 2101760,
            "unique_input_samples": 2101760, "total_supplied_samples": 67256320, "total_frames": 420288,
            "passes": 32, "source_before_after_verified": True, "source_sha256": source, "public_run": run,
            "pipeline_head": "a" * 40, "request_sha256": self.digest, "selected_cpu": 2,
            "initial_allowed_cpus": [2, 3], "native_group_elapsed_seconds": 20.0, "source_integrity": integrity,
            "resource_scope": resource_scope, "timed_region": timed_region,
            "clock_scope": "supplied-PCM-input-dependency-not-acoustic-word-end-or-wall-clock-latency",
            "repeated_diagnostic_audio_is_not_independent_exposure": True,
            "old_wholeclip_models_or_INT8_chunk8_verdicts_changed": False}
        self.write(root / "SUMMARY.json", summary)
        self.refresh_result(state)
        return state

    def refresh_result(self, state):
        root = state / "result/pool-profile"
        summary = C.core.read_json(root / "SUMMARY.json")
        summary["input_bindings_sha256"] = C.core.sha256((root / "INPUTS.json").read_bytes())
        summary["measurements_sha256"] = C.core.sha256((root / "MEASUREMENTS.json").read_bytes())
        self.write(root / "SUMMARY.json", summary)
        manifest = C.core.file_manifest(root)
        manifest.pop("REPLAY.json", None)
        self.write(root / "REPLAY.json", {"schema": "english63-pool-profile-replay-v1",
            "status": "VERIFIED_SAVED_POOL_PROFILE", "files_sha256": manifest,
            "files": {name: {"bytes": (root / name).stat().st_size, "sha256": value} for name, value in manifest.items()},
            "summary_sha256": manifest["SUMMARY.json"], "model_forward_calls": 0, "optimizer_updates": 0,
            "new_audio_asr_calls": 0, "formal_training_admitted": False, "model_promotion_allowed": False,
            "runtime_default_changed": False})

    def seal_fixture(self, state):
        with self.environment(), contextlib.redirect_stdout(io.StringIO()):
            C.seal(self.path, state)

    def test_complete_no_benefit_is_successful_and_prior_terminal_reuses_without_execution(self):
        state = self.result_fixture()
        C.validate_result(state / "result")
        self.seal_fixture(state)
        terminal = C.validate_terminal(state / "result", self.request, self.digest)
        self.assertEqual(terminal["status"], "SUCCEEDED")
        self.assertEqual(C.core.read_json(state / "result/pool-profile/SUMMARY.json")["decision"], "NO_BENEFIT")
        raw = self.archive({p.relative_to(state / "result").as_posix(): p.read_bytes()
                            for p in (state / "result").rglob("*") if p.is_file()})
        with self.environment(attempt="2"), contextlib.redirect_stdout(io.StringIO()) as output, \
             mock.patch.object(C.core, "named_artifacts", side_effect=[[{"id": 77}], []]), \
             mock.patch.object(C.core, "artifact_bytes", return_value=raw), \
             mock.patch.object(C.core, "github_get", side_effect=AssertionError("must not redownload inputs")), \
             mock.patch.object(C.native, "run_bounded", side_effect=AssertionError("must not execute models")):
            C.prepare(self.path, self.root / "reused")
        self.assertIn("execute=false", output.getvalue())

    def test_complete_benefit_and_typed_summary_forgery(self):
        state = self.result_fixture(benefit=True)
        C.validate_result(state / "result")
        path = state / "result/pool-profile/SUMMARY.json"
        original = C.core.read_json(path)
        for key, bad in (("optimizer_updates", False), ("passes", 32.0), ("runtime_default_changed", 0),
                         ("total_frames", 420289), ("decision", "NO_BENEFIT")):
            with self.subTest(key=key):
                self.write(path, original | {key: bad})
                self.refresh_result(state)
                with self.assertRaises(ValueError):
                    C.validate_result(state / "result")

    def test_missing_added_or_truncated_trace_fails_even_with_new_manifest(self):
        state = self.result_fixture()
        root = state / "result/pool-profile"
        path = root / "gru-candidate.trace"
        path.unlink()
        self.refresh_result(state)
        with self.assertRaisesRegex(ValueError, "closure"):
            C.validate_result(state / "result")
        path.write_bytes(self.trace[:-1])
        self.refresh_result(state)
        with self.assertRaisesRegex(ValueError, "trace"):
            C.validate_result(state / "result")
        path.write_bytes(self.trace)
        (root / "EXTRA.json").write_text("{}")
        self.refresh_result(state)
        with self.assertRaisesRegex(ValueError, "closure"):
            C.validate_result(state / "result")

    def test_saved_raw_types_schedule_and_real_frontend_bytes_are_enforced(self):
        state = self.result_fixture()
        path = state / "result/pool-profile/MEASUREMENTS.json"
        original = C.core.read_json(path)
        for mutation in ("raw-bool", "exit-bool", "missing-pass", "order", "frontend", "timed-zero", "affinity"):
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(original)
                row = value["passes"][0]
                if mutation == "raw-bool":
                    parsed = json.loads(row["raw_stdout"].splitlines()[1])
                    parsed["child_signaled"] = 0
                    row["raw_stdout"] = json.dumps(row["worker"]) + "\n" + json.dumps(parsed) + "\n"
                elif mutation == "exit-bool":
                    row["exit_code"] = False
                elif mutation == "missing-pass":
                    value["passes"].pop()
                elif mutation == "order":
                    value["passes"][4], value["passes"][5] = value["passes"][5], value["passes"][4]
                elif mutation == "frontend":
                    row["worker"]["frontend_table_bytes"] = 2178
                    row["raw_stdout"] = json.dumps(row["worker"]) + "\n" + json.dumps(row["lifetime"]) + "\n"
                elif mutation == "timed-zero":
                    row = value["passes"][4]
                    row["worker"]["feed_ns"][0] = 0
                    row["raw_stdout"] = json.dumps(row["worker"]) + "\n" + json.dumps(row["lifetime"]) + "\n"
                else:
                    value["selected_cpu"] = 3
                self.write(path, value)
                self.refresh_result(state)
                with self.assertRaises(ValueError):
                    C.validate_result(state / "result")

    def test_rehashed_input_claim_source_and_compiler_changes_fail(self):
        state = self.result_fixture()
        root = state / "result/pool-profile"
        for filename, mutation in (("INPUTS.json", "role"), ("INPUTS.json", "claim"),
                                   ("INPUTS.json", "parent"), ("BUILD.json", "macro"), ("BUILD.json", "source")):
            path = root / filename
            original = C.core.read_json(path)
            changed = copy.deepcopy(original)
            if mutation == "role":
                changed["rows"][30]["action"] = "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY"
            elif mutation == "claim":
                claim = json.loads(changed["claim_raw_utf8"])
                claim["run_id"] = "99999"
                changed["claim_raw_utf8"] = json.dumps(claim)
                changed["claim_sha256"] = C.core.sha256(changed["claim_raw_utf8"].encode())
            elif mutation == "parent":
                changed["parent_evidence"]["REQUEST.json"]["raw_utf8"] += " "
            elif mutation == "macro":
                changed["compiles"][1]["command"][9] = "-DEN63_POOL_TIME_MAJOR=0"
            else:
                changed["source_sha256"][C.ENTRYPOINT] = "f" * 64
            self.write(path, changed)
            self.refresh_result(state)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                C.validate_result(state / "result")
            self.write(path, original)
        self.refresh_result(state)
        replay = C.core.read_json(root / "REPLAY.json")
        replay["model_forward_calls"] = False
        self.write(root / "REPLAY.json", replay)
        with self.assertRaisesRegex(ValueError, "replay"):
            C.validate_result(state / "result")

    def test_failed_terminal_and_wrong_run_cannot_be_reused(self):
        state = self.result_fixture()
        execution_path = state / "result/EXECUTION.json"
        execution = C.core.read_json(execution_path)
        execution["verify_exit_code"] = None
        self.write(execution_path, execution)
        self.seal_fixture(state)
        with self.assertRaisesRegex(ValueError, "prior pool attempt failed"):
            C.validate_terminal(state / "result", self.request, self.digest)
        (state / "result/TERMINAL.json").unlink()
        execution["verify_exit_code"] = 0
        self.write(execution_path, execution)
        self.seal_fixture(state)
        path = state / "result/TERMINAL.json"
        terminal = C.core.read_json(path)
        terminal["run_id"] = "99999"
        self.write(path, terminal)
        with self.assertRaisesRegex(ValueError, "public run binding"):
            C.validate_terminal(state / "result", self.request, self.digest)

    def test_exact_fixed_request_and_nested_types(self):
        self.assertEqual(C.load_request(self.path)[2], self.digest)
        mutations = [
            ("request_id", "en63-native-c-pool-another-v1"),
            ("optimizer_updates", False), ("optimizer_updates", 1),
            ("cpu_threads", 2), ("max_execution_seconds", 601),
            ("entrypoint", C.state_consumer.ENTRYPOINT),
            ("pipeline_commit", "main"), ("native_state_ancestor", {}),
            ("source_artifacts", {}), ("human_gold", True),
            ("new_independent_audio_seconds", 131.36),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                request = copy.deepcopy(self.request)
                request[key] = value
                self.write(self.path, request)
                with self.assertRaises(ValueError):
                    C.load_request(self.path)
        for location, key, value in (
            ("native_state_ancestor", "run_id", float(C.ANCESTOR["run_id"])),
            ("profile", "full_passes", 32.0),
            ("profile", "cpu_affinity", "any"),
        ):
            with self.subTest(location=location, key=key):
                request = copy.deepcopy(self.request)
                request[location][key] = value
                self.write(self.path, request)
                with self.assertRaisesRegex(ValueError, "unsupported pool request field"):
                    C.load_request(self.path)
        request = copy.deepcopy(self.request)
        request["profile"]["variants"]["baseline"] = False
        self.write(self.path, request)
        with self.assertRaises(ValueError):
            C.load_request(self.path)

    def test_request_budget_order_and_thresholds_cannot_be_changed(self):
        for field, value in (("timed_pair_orders", [["baseline", "candidate"]] * 6),
                             ("trace_pair_order", ["candidate", "baseline"]),
                             ("native_group_seconds", 181), ("native_child_seconds", 31),
                             ("state_bytes_max", 38193), ("supplied_samples_including_replays", 1)):
            request = copy.deepcopy(self.request)
            request["profile"][field] = value
            self.write(self.path, request)
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.load_request(self.path)
        request = copy.deepcopy(self.request)
        request["profile"]["acceptance"]["median_paired_cpu_reduction_min"] = 0.01
        self.write(self.path, request)
        with self.assertRaises(ValueError):
            C.load_request(self.path)

    def test_duplicate_extra_command_and_alternate_filename_fail(self):
        self.path.write_text('{"profile":{"x":1,"x":2}}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            C.load_request(self.path)
        self.write(self.path, self.request | {"command": "arbitrary"})
        with self.assertRaises(ValueError):
            C.load_request(self.path)
        other = self.root / "other-pool-request.json"
        self.write(other, self.request)
        with self.assertRaisesRegex(ValueError, "filename"):
            C.load_request(other)

    def test_old_state_request_is_immutable_and_dispatch_does_not_train(self):
        self.assertEqual(C.core.sha256(C.SOURCE_REQUEST.read_bytes()), C.ANCESTOR["request_sha256"])
        self.assertEqual(C.core.native_consumer(self.path).ENTRYPOINT, C.ENTRYPOINT)
        consumer = mock.Mock()
        with mock.patch.object(C.core.sys, "argv", ["executor", "execute", "--request", str(self.path),
                                                     "--state", str(self.root / "dispatch")]), \
             mock.patch.object(C.core, "native_consumer", return_value=consumer), \
             mock.patch.object(C.core, "execute", side_effect=AssertionError("must not train")) as training:
            C.core.main()
        consumer.dispatch.assert_called_once()
        training.assert_not_called()

    def test_source_event_still_requires_one_added_immutable_request(self):
        path = C.core.REQUEST_DIR + "/" + C.REQUEST_ID + ".json"
        self.assertEqual(C.core.changed_request("A\t" + path, False), path)
        for line in ("M\t" + path, "D\t" + path, "A\t" + path + "\nA\t" + path):
            with self.subTest(line=line), self.assertRaises(ValueError):
                C.core.changed_request(line, False)

    def test_public_first_attempt_and_existing_claim_boundaries(self):
        for private, event in ((True, "push"), (False, "pull_request")):
            with self.environment(private=private, event_name=event):
                with self.assertRaisesRegex(ValueError, "public main"):
                    C.prepare(self.path, self.root / "blocked")
        with self.environment(attempt="2"), mock.patch.object(C.core, "named_artifacts", return_value=[]):
            with self.assertRaisesRegex(ValueError, "rerun"):
                C.prepare(self.path, self.root / "rerun")
        with self.environment(), mock.patch.object(C.core, "named_artifacts", side_effect=[[], [{"id": 8}]]):
            with self.assertRaisesRegex(ValueError, "prior pool claim"):
                C.prepare(self.path, self.root / "orphan")
        with self.environment(), mock.patch.object(C.core, "named_artifacts", side_effect=[[{}, {}], []]):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                C.prepare(self.path, self.root / "ambiguous")

    def test_prepare_checks_original_bytes_before_creating_claim(self):
        ancestor_raw = self.archive({"INVENTED.json": b'{"fixture":true}\n'})
        review = b'{"fixture":"invented review"}\n'
        review_raw = self.archive({"english63-three-evidence-review.json": review})
        tts = b"invented opaque archive fixture, no audio"
        inputs = {
            "tts": {"artifact_id": 11, "zip_sha256": C.core.sha256(tts)},
            "review": {"artifact_id": 12, "zip_sha256": C.core.sha256(review_raw),
                       "member": "english63-three-evidence-review.json", "member_sha256": C.core.sha256(review)},
        }
        ancestor_metadata = {"id": C.ANCESTOR["artifact_id"], "size_in_bytes": len(ancestor_raw),
                             "workflow_run": {"id": C.ANCESTOR["run_id"], "head_sha": C.ANCESTOR["data_commit"]}}
        tts_metadata = {"id": 11, "workflow_run": {"id": 43242278962,
                                                  "head_sha": "bec1a1421d1db2de17afb9c6e3155343becdb980"}}
        run = {"status": "completed", "conclusion": "success", "run_attempt": 1,
               "event": "push", "head_sha": C.ANCESTOR["data_commit"]}
        state = self.root / "prepared"
        with mock.patch.dict(C.ANCESTOR, {"zip_bytes": len(ancestor_raw), "zip_sha256": C.core.sha256(ancestor_raw)}), \
             mock.patch.dict(C.core.INPUTS, inputs, clear=True):
            self.write(self.path, copy.deepcopy(C.FIXED) | {"pipeline_commit": "a" * 40, "reason": "invented"})
            with self.environment(), contextlib.redirect_stdout(io.StringIO()), \
                 mock.patch.object(C.core, "named_artifacts", return_value=[]), \
                 mock.patch.object(C.core, "github_get", return_value=run), \
                 mock.patch.object(C.core, "artifact_metadata", side_effect=[ancestor_metadata, tts_metadata, {"id": 12}]), \
                 mock.patch.object(C.core, "artifact_bytes", side_effect=[ancestor_raw, tts, review_raw]), \
                 mock.patch.object(C, "validate_ancestor") as ancestor_check:
                C.prepare(self.path, state)
            self.assertEqual(ancestor_check.call_count, 2)
            self.assertTrue((state / "claim/CLAIM.json").is_file())
            rejected = self.root / "bad-original-before-claim"
            with self.environment(), contextlib.redirect_stdout(io.StringIO()), \
                 mock.patch.object(C.core, "named_artifacts", return_value=[]), \
                 mock.patch.object(C.core, "github_get", return_value=run), \
                 mock.patch.object(C.core, "artifact_metadata", side_effect=[ancestor_metadata, tts_metadata, {"id": 12}]), \
                 mock.patch.object(C.core, "artifact_bytes", side_effect=[ancestor_raw, b"changed original", review_raw]), \
                 mock.patch.object(C, "validate_ancestor"):
                with self.assertRaisesRegex(ValueError, "original pool input changed"):
                    C.prepare(self.path, rejected)
            self.assertFalse((rejected / "claim/CLAIM.json").exists())
            with mock.patch.object(C, "validate_ancestor"):
                (state / "inputs/tts.zip").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "original pool input changed"):
                    C.validate_inputs(state)

    def test_only_public_actions_identity_and_no_tokens_reach_child(self):
        state = self.root / "environment"
        state.mkdir()
        with self.environment(), mock.patch.dict(os.environ, {
            "GH_TOKEN": "fixture-secret", "GITHUB_TOKEN": "fixture-secret",
            "ACTIONS_RUNTIME_TOKEN": "fixture-secret", "GITHUB_ARBITRARY": "not-forwarded",
        }):
            environment = C.state_consumer.state_environment(Path("/invented/python"), state)
        self.assertEqual({k for k in environment if k.startswith("GITHUB_")}, {
            "GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_REPOSITORY", "GITHUB_SHA"})
        self.assertFalse({"GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN", "GITHUB_ARBITRARY"} & set(environment))
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")

    def test_successful_execute_uses_saved_only_verifier_without_input_arguments(self):
        state, pipeline, script_raw = self.execution_fixture()
        with self.environment(), \
             mock.patch.object(C.subprocess, "check_output", side_effect=["a" * 40, "", script_raw]), \
             mock.patch.object(C, "validate_inputs") as inputs, \
             mock.patch.object(C.native, "run_bounded", return_value=0) as child:
            C.execute(self.path, state, pipeline, Path("/invented/python"))
        self.assertEqual(inputs.call_count, 2)
        self.assertEqual(child.call_count, 2)
        self.assertIn("--state-result-dir", child.call_args_list[0].args[0])
        self.assertEqual(child.call_args_list[0].kwargs["seconds"], 600)
        verification = child.call_args_list[1].args[0]
        self.assertIn("--verify-result-dir", verification)
        self.assertNotIn("--state-result-dir", verification)
        self.assertNotIn("--tts-zip", verification)
        self.assertNotIn("--review-json", verification)
        self.assertEqual(child.call_args_list[1].kwargs["seconds"], 180)
        self.assertTrue(C.core.read_json(state / "result/EXECUTION.json")["source_inputs_before_after_verified"])

    def test_execution_failure_retains_receipt_without_starting_verifier(self):
        state, pipeline, script_raw = self.execution_fixture()
        with self.environment(), \
             mock.patch.object(C.subprocess, "check_output", side_effect=["a" * 40, "", script_raw]), \
             mock.patch.object(C, "validate_inputs"), \
             mock.patch.object(C.native, "run_bounded", return_value=1) as child:
            with self.assertRaisesRegex(ValueError, "execution"):
                C.execute(self.path, state, pipeline, Path("/invented/python"))
        self.assertEqual(child.call_count, 1)
        receipt = C.core.read_json(state / "result/EXECUTION.json")
        self.assertEqual(receipt["execute_exit_code"], 1)
        self.assertIsNone(receipt["verify_exit_code"])
        self.assertFalse(receipt["source_inputs_before_after_verified"])

    def test_timeout_and_input_change_do_not_become_success(self):
        for mode in ("timeout", "input_change"):
            with self.subTest(mode=mode):
                temporary = tempfile.TemporaryDirectory(dir=self.root)
                self.addCleanup(temporary.cleanup)
                saved_root = self.root
                self.root = Path(temporary.name)
                state, pipeline, script_raw = self.execution_fixture()
                self.root = saved_root
                with self.environment(), \
                     mock.patch.object(C.subprocess, "check_output", side_effect=["a" * 40, "", script_raw]), \
                     mock.patch.object(C, "validate_inputs", side_effect=[None, ValueError("changed input")] if mode == "input_change" else None), \
                     mock.patch.object(C.native, "run_bounded", side_effect=subprocess.TimeoutExpired("invented", 600) if mode == "timeout" else [0, 0]):
                    with self.assertRaises(ValueError):
                        C.execute(self.path, state, pipeline, Path("/invented/python"))
                self.assertFalse(C.core.read_json(state / "result/EXECUTION.json")["source_inputs_before_after_verified"])

    def test_boolean_float_and_partial_exit_codes_are_not_success(self):
        good = {"schema": "kws-research-v4-native-c-pool-execution-v1",
                "request_sha256": self.digest, "execute_exit_code": 0, "verify_exit_code": 0,
                "source_inputs_before_after_verified": True, "optimizer_updates": 0}
        C.execution_ok(good, self.digest)
        for field in ("execute_exit_code", "verify_exit_code", "optimizer_updates"):
            for bad in (False, 0.0, None, 1):
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    C.execution_ok(good | {field: bad}, self.digest)


if __name__ == "__main__":
    unittest.main()
