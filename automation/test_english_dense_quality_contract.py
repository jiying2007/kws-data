#!/usr/bin/env python3
"""Source-only protocol fixtures; no numerical/model library is imported."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


WORKER = Path(__file__).with_name("english_dense_quality_once.py")
SPEC = importlib.util.spec_from_file_location("dense_worker_source_contract", WORKER)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def plan_fixture():
    plan = worker.expected_plan()
    plan.update({key: "0" * 64 for key in worker.SOURCE_HASH_KEYS})
    plan["common_worker_sha256"] = worker.COMMON_WORKER_SHA
    return plan


def environment_fixture():
    env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": worker.REPO,
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "12345"}
    event = {"before": worker.MAIN_BEFORE, "after": "a" * 40,
        "repository": {"full_name": worker.REPO, "private": False}}
    return env, event


def metric(case_id, *, role="development", label=1, matched=True, prefix=None):
    row = {"id": case_id, "role": role, "weak_class": label, "observed_samples": 60000,
        "source": {"clip_start_sample": 35200, "original_clip_eof_sample": 48000}}
    events = ([{"sample": 52000, "keyword": label}] if matched and label else [])
    if prefix:
        events.insert(0, {"sample": 35200, "keyword": prefix})
    return worker.summarize_case(row, events)


class PlanAndGuardTests(unittest.TestCase):
    def test_plan_closed_and_fixed(self):
        worker.validate_plan(plan_fixture())
        for key, value in (("maximum_optimizer_updates", 97), ("maximum_embedding_calls", 4001),
                           ("threshold", 0.79), ("confirm_embeddings", 2), ("retry_count", 1),
                           ("snr5_training_allowed", True), ("formal_training_admitted", True),
                           ("maximum_training_bag_forwards", True)):
            with self.subTest(key=key):
                plan = plan_fixture(); plan[key] = value
                with self.assertRaises(ValueError):
                    worker.validate_plan(plan)
        plan = plan_fixture(); plan["unregistered_override"] = True
        with self.assertRaises(ValueError):
            worker.validate_plan(plan)

    def test_public_first_attempt_exact_successor_guard(self):
        env, event = environment_fixture()
        with patch.object(worker, "source_contract", return_value={}):
            worker.execution_guard(plan_fixture(), env, event)
            for key, value in (("GITHUB_ACTIONS", "false"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                               ("GITHUB_REF", "refs/heads/research"), ("GITHUB_RUN_ATTEMPT", "2"),
                               ("GITHUB_REPOSITORY", "private/other"), ("GITHUB_SHA", "bad")):
                altered = dict(env); altered[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    worker.execution_guard(plan_fixture(), altered, event)
            for mutation in ("private", "before", "after"):
                altered = copy.deepcopy(event)
                if mutation == "private":
                    altered["repository"]["private"] = True
                else:
                    altered[mutation] = "b" * 40
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    worker.execution_guard(plan_fixture(), env, altered)

    def test_remote_artifact_binds_successful_public_run_and_bytes(self):
        expected = worker.TEMPORAL
        run = {"status": "completed", "conclusion": "success", "id": expected["run_id"],
            "head_sha": expected["head_sha"], "run_attempt": 1, "event": "push", "head_branch": "main",
            "repository": {"full_name": worker.REPO, "private": False}}
        artifact = {"id": expected["artifact_id"], "name": expected["name"], "size_in_bytes": expected["bytes"],
            "expired": False, "workflow_run": {"id": expected["run_id"], "head_sha": expected["head_sha"]},
            "digest": "sha256:" + expected["sha256"]}
        worker.verify_remote_source(run, artifact, expected)
        for target, key, value in ((run, "conclusion", "failure"), (run, "run_attempt", 2),
                                   (artifact, "size_in_bytes", 1), (artifact, "digest", "sha256:" + "0" * 64)):
            altered = copy.deepcopy(target); altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.verify_remote_source(altered if target is run else run,
                                            altered if target is artifact else artifact, expected)

    def test_import_and_default_cli_need_no_numerical_modules(self):
        tree = ast.parse(WORKER.read_text())
        forbidden = {"numpy", "torch", "onnx", "onnxruntime", "soundfile", "portable_head", "causal_embedding"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(forbidden & {alias.name.split(".")[0] for alias in node.names})
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)
        plan = worker.expected_plan()
        plan.update(worker_sha256=worker.identity(WORKER)["sha256"],
            common_worker_sha256=worker.identity(WORKER.with_name("english_temporal_quality_once.py"))["sha256"],
            workflow_sha256=worker.identity(WORKER.parent.parent / ".github/workflows/english-dense-quality-once.yml")["sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"; path.write_text(json.dumps(plan))
            # -S makes this independent of installed third-party packages.
            result = subprocess.run([sys.executable, "-S", "-B", str(WORKER), "--plan", str(path)],
                                    capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(json.loads(result.stdout)["status"], "SOURCE_PLAN_VALIDATED_NO_MODEL_EXECUTION")

    def test_final_arm_plan_forbids_frontend_restarts_and_more_search(self):
        plan = plan_fixture()
        for field, value in (("maximum_embedding_calls", 1), ("maximum_onnx_forwards", 1),
                             ("maximum_head_case_passes", 265), ("last_additional_objective_arm", False),
                             ("subsequent_same_pool_loss_seed_threshold_search_allowed", True),
                             ("failed_temporal_head_used_as_warmstart", True)):
            changed = dict(plan); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                worker.validate_plan(changed)
        changed = dict(plan); changed["common_worker_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            worker.validate_plan(changed)

    def test_source_has_no_frontend_or_previous_worker_call_path(self):
        tree = ast.parse(WORKER.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({"onnx", "onnxruntime", "soundfile", "causal_embedding"} &
                                 {alias.name.split(".")[0] for alias in node.names})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"FrozenOnnxEmbedding", "build_snr5_conditions"})
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "common":
                    self.assertNotIn(node.func.attr, {"worker", "execute", "main", "prepare_noise"})
        evaluations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "evaluate"]
        self.assertEqual([node.args[0].value for node in evaluations], ["dense-clean", "dense-snr10", "dense-snr5"])
        self.assertTrue(all(isinstance(node.args[1], ast.Name) and node.args[1].id == "candidate" for node in evaluations))

    def test_source_uses_dense_positive_and_unchanged_background_bag_mean(self):
        tree = ast.parse(WORKER.read_text())
        branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "label == 0"]
        self.assertEqual(len(branches), 1)
        self.assertEqual(ast.unparse(branches[0].body[0]), "terms.append(-logp[:, 0].mean())")
        self.assertEqual(ast.unparse(branches[0].orelse[0]), "terms.append(-logp[:, label].mean())")
        means = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "loss" for target in node.targets)]
        self.assertEqual(len(means), 1)
        self.assertEqual(ast.unparse(means[0].value), "torch.stack(terms).mean()")


class TemporalGeometryTests(unittest.TestCase):
    def setUp(self):
        self.row = {"weak_class": 1, "observed_samples": 53120,
            "source": {"clip_start_sample": 35200, "original_clip_eof_sample": 48000}}
        self.samples = list(range(16000, 53121, 1280))
        self.ends = [49280, 50560, 51840, 53120]

    def test_exact_original_contiguous_three_and_four_windows(self):
        worker.validate_head_ends(self.row, self.ends, self.samples)
        worker.validate_head_ends(self.row, self.ends[:3], self.samples)

    def test_reject_sparse_triplets_history_gaps_and_incomplete_clip_support(self):
        fixtures = [([49280, 51840, 53120], self.samples), (self.ends[:2], self.samples),
                    (self.ends, [t for t in self.samples if t != 36480]),
                    ([48000, 49280, 50560], self.samples)]
        for ends, samples in fixtures:
            with self.subTest(ends=ends, n=len(samples)), self.assertRaises(ValueError):
                worker.validate_head_ends(self.row, ends, samples)


class PairedRetentionTests(unittest.TestCase):
    def test_regression_cannot_be_hidden_by_aggregate_rescue(self):
        old = [metric("rescued", matched=False), metric("regressed", matched=True)]
        new = [metric("rescued", matched=True), metric("regressed", matched=False)]
        pairs = worker.paired_comparison(old, new)
        self.assertTrue(pairs[0]["development_miss_rescued"])
        self.assertTrue(pairs[1]["previously_correct_case_regressed"])
        gate = worker.research_retention({"clean": [], "snr10": pairs, "snr5": []},
            [{"role": "development", "weak_class": label} for label in (1, 2)])
        self.assertFalse(gate["passed"])

    def test_error_category_increase_fails_even_on_preexisting_miss(self):
        pairs = worker.paired_comparison([metric("still_missed", matched=False)],
                                        [metric("still_missed", matched=False, prefix=1)])
        self.assertFalse(pairs[0]["previously_correct_case_regressed"])
        self.assertEqual(pairs[0]["event_error_categories_increased"], ["prefix_events"])
        clean = worker.paired_comparison([metric("clean")], [metric("clean")])
        rescued = worker.paired_comparison([metric("rescue", matched=False)], [metric("rescue")])
        gate = worker.research_retention({"clean": clean, "snr10": rescued, "snr5": pairs},
            [{"role": "development", "weak_class": label} for label in (1, 2)])
        self.assertFalse(gate["passed"])

    def test_exact_retention_pass_and_missing_keyword_fail(self):
        unchanged = worker.paired_comparison([metric("same")], [metric("same")])
        rescued = worker.paired_comparison([metric("rescue", matched=False)], [metric("rescue")])
        paired = {"clean": unchanged, "snr10": rescued, "snr5": unchanged}
        rows = [{"role": "development", "weak_class": label} for label in (1, 2)]
        self.assertTrue(worker.research_retention(paired, rows)["passed"])
        self.assertFalse(worker.research_retention(paired, rows[:1])["passed"])


class SupervisorCleanupTests(unittest.TestCase):
    def test_descendant_group_is_cleaned_after_successful_leader_exit(self):
        class FakeLeader:
            pid, returncode = 123456789, 0

            def wait(self, timeout):
                return self.returncode

            def poll(self):
                return self.returncode

        alive = [True]
        signals = []

        def killpg(pid, sig):
            self.assertEqual(pid, FakeLeader.pid)
            if not alive[0]:
                raise ProcessLookupError
            if sig:
                signals.append(sig)
                alive[0] = False

        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory) / "child.stdout", Path(directory) / "child.stderr"
            with patch.object(worker.common.subprocess, "Popen", return_value=FakeLeader()) as popen, \
                 patch.object(worker.os, "killpg", side_effect=killpg):
                worker.supervised_command(["fake-source-fixture"], out, err, 1, {})
            receipt = json.loads(out.with_suffix(".exit.json").read_text())
        self.assertEqual(signals, [signal.SIGTERM])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertTrue(receipt["process_group_terminated"])
        self.assertTrue(receipt["group_cleanup_checked_after_leader_exit"])
        self.assertEqual(receipt["returncode"], 0)

    def test_timeout_always_cleans_group_and_records_failure(self):
        class FakeLeader:
            pid, returncode = 123456789, -15

            def wait(self, timeout):
                raise subprocess.TimeoutExpired("fake-source-fixture", timeout)

            def poll(self):
                return self.returncode

        alive = [True]

        def killpg(_pid, sig):
            if not alive[0]:
                raise ProcessLookupError
            if sig:
                alive[0] = False

        with tempfile.TemporaryDirectory() as directory:
            out, err = Path(directory) / "child.stdout", Path(directory) / "child.stderr"
            with patch.object(worker.common.subprocess, "Popen", return_value=FakeLeader()), \
                 patch.object(worker.os, "killpg", side_effect=killpg), \
                 self.assertRaises(subprocess.TimeoutExpired):
                worker.supervised_command(["fake-source-fixture"], out, err, 1, {})
            receipt = json.loads(out.with_suffix(".exit.json").read_text())
        self.assertTrue(receipt["process_group_terminated"])
        self.assertEqual(receipt["error"], "TimeoutExpired")


if __name__ == "__main__":
    unittest.main()
