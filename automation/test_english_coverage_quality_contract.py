#!/usr/bin/env python3
"""Pure source and invented-receipt fixtures. Never execute a numerical model."""
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


WORKER = Path(__file__).with_name("english_coverage_quality_once.py")
SPEC = importlib.util.spec_from_file_location("coverage_worker_contract", WORKER)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def plan_fixture():
    plan = worker.expected_plan()
    plan.update({key: "0" * 64 for key in worker.SOURCE_HASH_KEYS})
    plan["common_worker_sha256"] = worker.COMMON_WORKER_SHA
    return plan


def metric(case_id, *, label=1, matched=True, prefix=False):
    row = {"id": case_id, "role": "development", "weak_class": label, "observed_samples": 60000,
        "source": {"clip_start_sample": 35200, "original_clip_eof_sample": 48000}}
    events = [{"sample": 52000, "keyword": label}] if matched and label else []
    if prefix:
        events.insert(0, {"sample": 35200, "keyword": 1})
    return worker.summarize_case(row, events)


def new8_roster():
    return [{"id": "new8-" + str(i), "parent_id": "parent-" + str(i),
             "role": "development", "weak_class": 0 if i < 16 else 1 if i < 19 else 2}
            for i in range(27)]


class PlanAndGuardTests(unittest.TestCase):
    def test_closed_plan_preserves_fixed_scope(self):
        worker.validate_plan(plan_fixture())
        for key, value in (("maximum_optimizer_updates", 97), ("maximum_embedding_calls", 2636),
                ("maximum_onnx_forwards", 5271), ("maximum_head_case_passes", 319),
                ("maximum_host_c_head_processes", 2), ("maximum_arm_c_head_processes", 2),
                ("threshold", 0.79), ("confirm_embeddings", 2), ("retry_count", 1),
                ("snr5_training_allowed", True), ("formal_training_admitted", True),
                ("new8_preparation_after_candidate_frozen", False),
                ("subsequent_same_pool_loss_seed_threshold_search_allowed", True),
                ("same_pool_epoch_search_allowed", True), ("maximum_training_bag_forwards", True)):
            with self.subTest(key=key):
                plan = plan_fixture(); plan[key] = value
                with self.assertRaises(ValueError):
                    worker.validate_plan(plan)
        plan = plan_fixture(); plan["optional_retry"] = False
        with self.assertRaises(ValueError):
            worker.validate_plan(plan)

    def test_exact_first_public_main_successor(self):
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": worker.REPO,
            "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123"}
        event = {"before": worker.MAIN_BEFORE, "after": "a" * 40,
            "repository": {"full_name": worker.REPO, "private": False}}
        with patch.object(worker, "source_contract", return_value={}):
            worker.execution_guard(plan_fixture(), env, event)
            for key, value in (("GITHUB_ACTIONS", "false"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                    ("GITHUB_REF", "refs/heads/test"), ("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_SHA", "bad")):
                altered = dict(env); altered[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    worker.execution_guard(plan_fixture(), altered, event)
            for key in ("before", "after", "private"):
                altered = copy.deepcopy(event)
                if key == "private":
                    altered["repository"]["private"] = True
                else:
                    altered[key] = "b" * 40
                with self.subTest(key=key), self.assertRaises(ValueError):
                    worker.execution_guard(plan_fixture(), env, altered)

    def test_default_cli_is_source_only_without_site_packages(self):
        tree = ast.parse(WORKER.read_text())
        forbidden = {"numpy", "torch", "onnx", "onnxruntime", "portable_head", "causal_embedding"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(forbidden & {x.name.split(".")[0] for x in node.names})
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)
        plan = worker.expected_plan()
        paths = {"worker_sha256": WORKER,
            "workflow_sha256": WORKER.parent.parent / ".github/workflows/english-coverage-quality-once.yml",
            "common_worker_sha256": WORKER.with_name("english_temporal_quality_once.py"),
            "data_helper_sha256": WORKER.with_name("english_coverage_quality_data.py")}
        plan.update({key: worker.identity(path)["sha256"] for key, path in paths.items()})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "plan.json"; target.write_text(json.dumps(plan))
            result = subprocess.run([sys.executable, "-S", "-B", str(WORKER), "--plan", str(target)],
                check=True, capture_output=True, text=True, timeout=10)
        self.assertEqual(json.loads(result.stdout)["status"], "SOURCE_PLAN_VALIDATED_NO_MODEL_EXECUTION")

    def test_original_three_input_pins_are_immutable(self):
        self.assertEqual(worker.TRAINING["run_id"], 43867764366)
        self.assertEqual(worker.INCUMBENT["run_id"], 44021084918)
        self.assertEqual(worker.TEMPORAL["run_id"], 44031666420)
        self.assertEqual(worker.TEMPORAL["head_sha"], "b4982d7a2f39c4b2647a86cbf96d3701a1123b6a")
        changed = plan_fixture(); changed["incumbent_source"] = dict(worker.INCUMBENT, bytes=1)
        with self.assertRaises(ValueError):
            worker.validate_plan(changed)


class DrawAndSourceTests(unittest.TestCase):
    def setUp(self):
        self.groups = {(condition, label): [(condition, condition + "/" + str(label) + "/" + str(i))
                for i in range(32 if label == 0 else 8 if label == 1 else 10)]
            for condition in ("clean", "snr10", "new5", "new15") for label in range(3)}
        self.old = [[{"condition": condition, "id": self.groups[(condition, label)][step % len(self.groups[(condition, label)])][1],
                      "weak_class": label, "role": "train"}
                     for condition in ("clean", "snr10") for label in range(3)] for step in range(96)]

    def test_six_balanced_bags_and_complete_negative_condition_coverage(self):
        draws = [worker.training_draws(self.groups, step, self.old) for step in range(96)]
        for step, used in enumerate(draws):
            self.assertEqual(used[:3], self.old[step][:3])
            self.assertEqual(len(used), 6)
            self.assertEqual([row["weak_class"] for row in used], [0, 1, 2, 0, 1, 2])
            self.assertEqual({row["condition"] for row in used[3:]}, {( "snr10", "new5", "new15")[step % 3]})
            self.assertEqual({row["condition_occurrence"] for row in used[3:]}, {step // 3})
        for condition in ("snr10", "new5", "new15"):
            selected = [row["id"] for used in draws for row in used if row["condition"] == condition and row["weak_class"] == 0]
            self.assertEqual(selected, [key[1] for key in self.groups[(condition, 0)]])

    def test_sparse_eligible_groups_cycle_without_replacement_generation(self):
        self.groups[("new5", 1)] = self.groups[("new5", 1)][:6]
        self.assertEqual(worker.training_draws(self.groups, 94, self.old)[4]["id"], self.groups[("new5", 1)][31 % 6][1])
        self.groups[("new5", 1)] = []
        with self.assertRaises(ValueError):
            worker.training_draws(self.groups, 1, self.old)
        broken = copy.deepcopy(self.old); broken[0][0]["id"] = "wrong"
        with self.assertRaises(ValueError):
            worker.training_draws(self.groups, 0, broken)

    def test_only_original_objective_and_unchanged_optimizer(self):
        tree = ast.parse(WORKER.read_text())
        branches = [node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == "label == 0"]
        self.assertEqual(len(branches), 1)
        self.assertEqual(ast.unparse(branches[0].body[0]), "terms.append(-logp[:, 0].mean())")
        self.assertEqual(ast.unparse(branches[0].orelse[0]),
            "terms.append(math.log(len(logp)) - torch.logsumexp(logp[:, label], dim=0))")
        losses = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "loss" for target in node.targets)]
        self.assertEqual([ast.unparse(node.value) for node in losses], ["torch.stack(terms).mean()"])
        calls = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertIn("torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)", calls)
        self.assertIn("torch.manual_seed(20261117)", calls)
        self.assertIn("torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)", calls)

    def test_no_old_worker_invocation_and_no_extra_python_parity_model_pass(self):
        tree = ast.parse(WORKER.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "common":
                    self.assertNotIn(node.func.attr, {"worker", "execute", "main"})
                self.assertNotIn(node.func.attr, {"check_parity", "build_snr5_conditions"})
        evaluations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "evaluate"]
        self.assertEqual([(node.args[0].value, node.args[1].id) for node in evaluations],
            [("coverage-clean", "candidate"), ("coverage-snr10", "candidate"), ("coverage-snr5", "candidate"),
             ("aug96-new8", "control"), ("coverage-new8", "candidate")])
        references = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "reference"]
        self.assertEqual(len(references), 1)

    def test_new8_construction_follows_the_frozen_candidate(self):
        source = WORKER.read_text()
        frozen = source.index('phase("CANDIDATE_FROZEN")')
        challenge = source.index('prepare_new("development_challenge")')
        self.assertLess(frozen, challenge)
        self.assertLess(source.index('phase("TRAINING_STARTED")'), frozen)
        self.assertLess(challenge, source.index('phase("NEW8_EVALUATION_STARTED")'))


class RetentionAndGeometryTests(unittest.TestCase):
    def gate(self, new8=None, rows=None):
        unchanged = worker.paired_comparison([metric("same")], [metric("same")])
        rescued = worker.paired_comparison([metric("rescue", matched=False)], [metric("rescue")])
        paired = {"clean": unchanged, "snr10": rescued, "snr5": unchanged,
                  "new8": unchanged if new8 is None else new8}
        return worker.retention_record(paired, [{"role": "development", "weak_class": 1},
            {"role": "development", "weak_class": 2}], new8_roster() if rows is None else rows)

    def test_complete_new8_and_real_old10_rescue_pass(self):
        self.assertTrue(self.gate()["passed"])

    def test_missing_any_new8_case_rejects_without_repair(self):
        gate = self.gate(rows=new8_roster()[:-1])
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["criteria"]["new8_all_27_planned_development_cases_eligible"])
        duplicate = new8_roster(); duplicate[-1]["parent_id"] = duplicate[-2]["parent_id"]
        self.assertFalse(self.gate(rows=duplicate)["passed"])

    def test_event_error_or_case_regression_cannot_hide_behind_aggregate_gain(self):
        for new8 in (worker.paired_comparison([metric("old-hit")], [metric("old-hit", matched=False)]),
                worker.paired_comparison([metric("old-miss", matched=False)], [metric("old-miss", matched=False, prefix=True)])):
            self.assertFalse(self.gate(new8=new8)["passed"])

    def test_original_positive_ends_must_be_consecutive_and_fully_supported(self):
        row = {"weak_class": 1, "observed_samples": 53120,
               "source": {"clip_start_sample": 35200, "original_clip_eof_sample": 48000}}
        samples = list(range(16000, 53121, 1280))
        worker.validate_head_ends(row, [49280, 50560, 51840, 53120], samples)
        for ends in ([49280, 51840, 53120], [49280, 50560], [48000, 49280, 50560]):
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                worker.validate_head_ends(row, ends, samples)


class CachedCReceiptTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"sample": 16000, "ready": False, "keyword": 0,
                      "logits": [0.0, 0.0, 0.0], "probabilities": [0.0, 0.0, 0.0]},
                     {"sample": 37760, "ready": True, "keyword": 1,
                      "logits": [-1.0, 2.0, 0.0], "probabilities": [0.1, 0.8, 0.1]}]
        self.raw = b"16000 0 0 0 0 0 0 0 0\n37760 1 1 -1 2 0 0.1 0.8 0.1\n"

    def test_parse_saved_c_output_without_model_work(self):
        report = worker.compare_c_rows(self.raw, self.rows)
        self.assertTrue(report["passed"])
        self.assertEqual(report["input_embeddings"], 2)
        self.assertEqual(report["head_outputs"], 1)
        self.assertEqual(report["events"], [{"sample": 37760, "keyword": 1}])

    def test_clock_event_ready_nonfinite_and_tolerance_errors_reject(self):
        variants = [self.raw.replace(b"37760", b"39040"), self.raw.replace(b"37760 1 1", b"37760 1 0"),
            self.raw.replace(b"16000 0 0", b"16000 1 0"), self.raw.replace(b"-1 2 0", b"nan 2 0"),
            self.raw.replace(b"-1 2 0", b"-1.001 2 0"), self.raw.replace(b"0.8 0.1", b"0.80001 0.1"),
            self.raw.splitlines(keepends=True)[0], self.raw + self.raw]
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                worker.compare_c_rows(raw, self.rows)


class SupervisorCleanupTests(unittest.TestCase):
    def test_owned_descendants_cleaned_after_successful_leader_exit(self):
        class FakeLeader:
            pid, returncode = 123456789, 0
            def wait(self, timeout):
                return self.returncode
            def poll(self):
                return self.returncode
        alive, signals = [True], []
        def killpg(pid, sig):
            self.assertEqual(pid, FakeLeader.pid)
            if not alive[0]:
                raise ProcessLookupError
            if sig:
                signals.append(sig); alive[0] = False
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "fake.stdout"
            with patch.object(worker.common.subprocess, "Popen", return_value=FakeLeader()), \
                    patch.object(worker.common.os, "killpg", side_effect=killpg):
                worker.supervised_command(["invented-fixture-command"], out, Path(directory) / "fake.stderr", 1)
            receipt = json.loads(out.with_suffix(".exit.json").read_text())
        self.assertEqual(signals, [signal.SIGTERM])
        self.assertTrue(receipt["group_cleanup_checked_after_leader_exit"])
        self.assertTrue(receipt["process_group_terminated"])
        self.assertEqual(receipt["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
