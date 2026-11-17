"""Pure-Python guard/build/metric tests: no subprocess, kernel, model or network."""
import ast
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_stream_3x1_ab_once as ab

TREE = ast.parse(Path(ab.__file__).read_text(encoding="utf-8"))


def plan_fixture():
    return {"schema": "english-stream-3x1-ab-once-v1", "main_before": ab.RESOURCE["head_sha"],
            "pipeline_commit": "a" * 40, "planned_full_model_passes": 8,
            "maximum_execution_seconds": 720, "new_audio_or_model_training": False, "retry_count": 0}


def metric_fixture(case_id="libri-speaker-1272"):
    measured = {"schema": "english-stream-performance-v1", "passed": True,
                "measurement_completed": True, "performance_budget_evaluated": False,
                "sample_rate_hz": 16000, "chunk_samples": 320, "warmup_samples": 35200,
                "input_samples": 256000, "input_repetitions": 1, "score_observer_enabled": False,
                "embeddings": 188, "ready_outputs": 173, "events": 0,
                "event_digest_fnv1a64": "cbf29ce484222325", "resets": 0, "eof_padding_samples": 0,
                "warmup_output_counts": {"embeddings": 16, "ready_outputs": 1, "events": 0},
                "steady_output_counts": {"embeddings": 172, "ready_outputs": 172, "events": 0},
                "steady": {"samples": 220800, "audio_seconds": 13.8, "calls": 690,
                           "accept_wall_seconds": 27.6, "accept_process_cpu_seconds": 26.91,
                           "wall_rtf": 2.0, "cpu_seconds_per_audio_second": 1.95}}
    return {"schema": "english-stream-performance-run-v1", "passed": True,
            "measurement_completed": True, "actual_input_seconds": 16,
            "cases": [{"input": {"id": case_id, "sha256": ab.TRACKS[case_id], "samples": 256000},
                       "status": "completed", "exit_code": 0, "pcm_sha256_after": ab.TRACKS[case_id],
                       "measurements": measured}]}


def named_command(label):
    matches = [node for node in ast.walk(TREE)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "command" and node.args
               and isinstance(node.args[0], ast.Constant) and node.args[0].value == label]
    if len(matches) != 1:
        raise AssertionError("expected one build command: " + label)
    return matches[0]


def evaluate_argv(node, **values):
    # Evaluate only the trusted worker's argv expression, with no I/O functions
    # or import mechanism available. This never calls command() or a process.
    namespace = {"__builtins__": {}, "str": str, "FLAGS": ab.FLAGS, "MEMBERS": ab.MEMBERS, **values}
    expression = ast.fix_missing_locations(ast.Expression(body=node))
    return eval(compile(expression, "<synthetic-build-argv>", "eval"), namespace)


class PlanAndExecutionGuardTests(unittest.TestCase):
    def test_fixed_plan_and_exact_integer_budget(self):
        ab.validate_plan(plan_fixture())
        actual = Path(ab.__file__).with_name("english_stream_3x1_ab_inputs.json")
        ab.validate_plan(json.loads(actual.read_text(encoding="utf-8")))
        for key, values in {"planned_full_model_passes": [True, 8.0, 7, 9],
                            "maximum_execution_seconds": [True, 720.0, 721],
                            "retry_count": [False, 0.0, 1]}.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    plan = plan_fixture(); plan[key] = value
                    with self.assertRaises(ValueError):
                        ab.validate_plan(plan)

    def test_plan_rejects_changed_source_or_training(self):
        for key, value in [("schema", "other"), ("main_before", "0" * 40),
                           ("pipeline_commit", "main"), ("pipeline_commit", "a" * 39),
                           ("new_audio_or_model_training", True)]:
            with self.subTest(key=key):
                plan = plan_fixture(); plan[key] = value
                with self.assertRaises(ValueError):
                    ab.validate_plan(plan)

    def test_public_first_main_successor_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "event.json"
            event = {"before": ab.RESOURCE["head_sha"], "after": "b" * 40,
                     "repository": {"full_name": ab.saved.REPO, "private": False}}
            path.write_text(json.dumps(event), encoding="utf-8")
            env = {"GITHUB_EVENT_PATH": str(path), "GITHUB_ACTIONS": "true",
                   "GITHUB_REPOSITORY": ab.saved.REPO, "GITHUB_EVENT_NAME": "push",
                   "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "b" * 40}
            with mock.patch.dict(ab.os.environ, env, clear=True):
                ab.saved.guard(plan_fixture())
            for key, value in [("GITHUB_ACTIONS", "false"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                               ("GITHUB_REF", "refs/heads/experiment"), ("GITHUB_RUN_ATTEMPT", "2"),
                               ("GITHUB_REPOSITORY", "someone/other"), ("GITHUB_SHA", "c" * 40)]:
                with self.subTest(key=key), mock.patch.dict(ab.os.environ, {**env, key: value}, clear=True):
                    with self.assertRaises(ValueError):
                        ab.saved.guard(plan_fixture())
            for mutation in ("private", "before"):
                altered = copy.deepcopy(event)
                if mutation == "private": altered["repository"]["private"] = True
                else: altered["before"] = "0" * 40
                path.write_text(json.dumps(altered), encoding="utf-8")
                with self.subTest(mutation=mutation), mock.patch.dict(ab.os.environ, env, clear=True):
                    with self.assertRaises(ValueError):
                        ab.saved.guard(plan_fixture())


class MeasurementTests(unittest.TestCase):
    def test_both_fixed_inputs_accept_valid_synthetic_metrics(self):
        for case_id in ab.TRACKS:
            report = metric_fixture(case_id)
            self.assertIs(ab.verify_measurement(report, case_id), report["cases"][0]["measurements"])

    def test_rejects_failed_unverified_or_wrong_input(self):
        for location, field, value in [("report", "passed", False), ("report", "measurement_completed", False),
                                       ("report", "actual_input_seconds", 32), ("case", "exit_code", 1),
                                       ("case", "status", "started"), ("case", "pcm_sha256_after", "0" * 64)]:
            with self.subTest(field=field):
                report = metric_fixture()
                target = report if location == "report" else report["cases"][0]
                target[field] = value
                with self.assertRaises(ValueError):
                    ab.verify_measurement(report, "libri-speaker-1272")
        with self.assertRaises(ValueError):
            ab.verify_measurement(metric_fixture("libri-speaker-1462"), "libri-speaker-1272")

    def test_rejects_wrong_stream_geometry(self):
        for field, value in [("score_observer_enabled", True), ("input_samples", 16000),
                             ("embeddings", 0), ("ready_outputs", 172), ("resets", 1),
                             ("eof_padding_samples", 1)]:
            with self.subTest(field=field):
                report = metric_fixture(); report["cases"][0]["measurements"][field] = value
                with self.assertRaises(ValueError):
                    ab.verify_measurement(report, "libri-speaker-1272")
        for field, value in [("audio_seconds", 16), ("calls", 800)]:
            with self.subTest(field=field):
                report = metric_fixture(); report["cases"][0]["measurements"]["steady"][field] = value
                with self.assertRaises(ValueError):
                    ab.verify_measurement(report, "libri-speaker-1272")
        for field in ("embeddings", "ready_outputs"):
            report = metric_fixture(); report["cases"][0]["measurements"]["steady_output_counts"][field] = 171
            with self.subTest(field=field), self.assertRaises(ValueError):
                ab.verify_measurement(report, "libri-speaker-1272")

    def test_rejects_nonfinite_or_nonpositive_timing(self):
        for field in ("accept_wall_seconds", "accept_process_cpu_seconds", "wall_rtf", "cpu_seconds_per_audio_second"):
            for bad in (0, -1, float("nan"), float("inf")):
                with self.subTest(field=field, bad=bad):
                    report = metric_fixture(); report["cases"][0]["measurements"]["steady"][field] = bad
                    with self.assertRaises(ValueError):
                        ab.verify_measurement(report, "libri-speaker-1272")


class BuildBoundaryTests(unittest.TestCase):
    def test_frontend_is_only_compiled_archive_member(self):
        candidate, objects = Path("/fixture/candidate"), Path("/fixture/candidate/objects")
        argv = evaluate_argv(named_command("build-candidate-frontend").args[1],
                             includes=["-I/fixture/header"], candidate=candidate, objects=objects)
        self.assertEqual(argv[0], "gcc")
        self.assertEqual(argv[argv.index("-c") + 1], candidate / "frontend.c")
        self.assertEqual(argv[argv.index("-o") + 1], objects / "fixed_frontend_mel.c.o")
        for flag in ("-O2", "-fno-fast-math", "-ffp-contract=off"):
            self.assertIn(flag, argv)
        self.assertFalse(any(str(x).startswith(("-march", "-mcpu", "-mfpu")) or
                             x in ("-Ofast", "-ffast-math", "-funsafe-math-optimizations") for x in argv))
        self.assertEqual(ab.MEMBERS, ["ekws_research_stream.c.o", "fixed_frontend_mel.c.o",
                                     "english_causal_head.c.o", "fixed_models.c.o"])

    def test_original_archive_is_extracted_readonly_and_new_archive_is_distinct(self):
        original, objects = Path("/fixture/original/lib.a"), Path("/fixture/candidate/objects")
        call = named_command("extract-original-objects")
        self.assertEqual(evaluate_argv(call.args[1], library=original), ["ar", "x", original])
        cwd = next(k.value for k in call.keywords if k.arg == "cwd")
        self.assertEqual(evaluate_argv(cwd, objects=objects), objects)
        destination = Path("/fixture/candidate/sdk-native/lib/libekws_research_stream.a")
        argv = evaluate_argv(named_command("build-candidate-archive").args[1],
                             candidate_library=destination, objects=objects)
        self.assertEqual(argv, ["ar", "rcs", destination, *[objects / n for n in ab.MEMBERS]])
        self.assertNotIn(original, argv)

    def test_interleaved_orders_have_eight_passes_and_fixed_denominators(self):
        order_nodes = [n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "orders" for t in n.targets)]
        self.assertEqual(len(order_nodes), 1)
        orders = ast.literal_eval(order_nodes[0])
        self.assertEqual(orders, [("libri-speaker-1272", ("baseline", "candidate")),
                                  ("libri-speaker-1462", ("candidate", "baseline"))])
        performance_passes = sum(len(variants) for _, variants in orders)
        self.assertEqual(performance_passes * 2, 8)
        self.assertEqual(performance_passes * 2 * 16, 128)
        self.assertEqual(len({case for case, _ in orders}) * 16, 32)
        self.assertAlmostEqual(len(orders) * 13.8, 27.6)


class CommandBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.evidence = Path(self.temporary.name)
        self.calls, self.waits, self.kills = [], [], []
        self.behavior = {"returncode": 0, "timeout": False}
        self.clock = {"now": 100.0}
        self.state = {"model_runner_invocations_started": 0}
        owner = self

        class FakeProcess:
            pid = 54321
            def __init__(self, argv, **kwargs):
                owner.calls.append((argv, kwargs))
                kwargs["stdout"].write(b"synthetic stdout; no process executed\n")
                kwargs["stderr"].write(b"synthetic stderr\n")
                self.wait_count = 0
            def wait(self, timeout=None):
                owner.waits.append(timeout); self.wait_count += 1
                if owner.behavior["timeout"] and self.wait_count == 1:
                    raise subprocess.TimeoutExpired(["synthetic-only"], timeout)
                return owner.behavior["returncode"]

        execute = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        function = next(n for n in execute.body if isinstance(n, ast.FunctionDef) and n.name == "command")
        isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {"time": types.SimpleNamespace(monotonic=lambda: self.clock["now"]), "started": 100.0,
                     "need": ab.need, "state": self.state, "evidence": self.evidence,
                     "checkpoint": lambda: None, "write_json": ab.write_json, "signal": signal,
                     "os": types.SimpleNamespace(environ={"GH_TOKEN": "synthetic-token", "GITHUB_TOKEN": "synthetic-token",
                                                           "ACTIONS_RUNTIME_TOKEN": "synthetic-token", "KEEP": "yes"},
                                                 killpg=lambda pid, sig: self.kills.append((pid, sig))),
                     "subprocess": types.SimpleNamespace(Popen=FakeProcess, TimeoutExpired=subprocess.TimeoutExpired)}
        exec(compile(isolated, "<isolated-real-command-with-fake-process>", "exec"), namespace)
        self.command = namespace["command"]

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_command_strips_tokens_and_starts_one_process_group(self):
        self.command("normal", ["synthetic-only"], model=True)
        self.assertEqual(self.state["model_runner_invocations_started"], 1)
        self.assertTrue(self.calls[0][1]["start_new_session"])
        self.assertEqual(self.calls[0][1]["env"], {"KEEP": "yes"})
        self.assertEqual(self.waits, [180])
        self.assertEqual(json.loads((self.evidence / "normal-exit.json").read_text())["returncode"], 0)

    def test_deadline_refuses_to_start_a_process(self):
        self.clock["now"] = 820.0
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.command("expired", ["synthetic-only"], model=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state["model_runner_invocations_started"], 0)

    def test_ninth_full_model_invocation_never_starts(self):
        self.state["model_runner_invocations_started"] = 8
        with self.assertRaisesRegex(ValueError, "pass budget"):
            self.command("ninth", ["synthetic-only"], model=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state["model_runner_invocations_started"], 8)

    def test_timeout_uses_remaining_deadline_and_kills_whole_group(self):
        self.clock["now"] = 800.0
        self.behavior["timeout"] = True
        with self.assertRaises(subprocess.TimeoutExpired):
            self.command("timeout", ["synthetic-only"], timeout=180, model=True)
        self.assertEqual(self.waits, [20.0, None])
        self.assertEqual(self.kills, [(54321, signal.SIGKILL)])
        self.assertIn("synthetic stdout", (self.evidence / "timeout.stdout").read_text())
        self.assertIn("synthetic stderr", (self.evidence / "timeout.stderr").read_text())
        result = json.loads((self.evidence / "timeout-exit.json").read_text())
        self.assertEqual(result["error_type"], "TimeoutExpired")
        self.assertTrue(result["full_model_process"])

    def test_nonzero_command_fails_once_with_retained_exit_record(self):
        self.behavior["returncode"] = 7
        with self.assertRaisesRegex(ValueError, "command failed"):
            self.command("failed", ["synthetic-only"], model=True)
        self.assertEqual(len(self.calls), 1)
        result = json.loads((self.evidence / "failed-exit.json").read_text())
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(result["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
