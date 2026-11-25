"""UNEXECUTED draft: explicit, asset-free CPU Torch mathematics only.

Authoring/static inspection does not import Torch. Actual execution requires
--run-pure-math and the independently installed pinned CPU runtime. No model,
weights, audio, optimizer, source/observer, Git, or network operation is part of
this suite. The frozen original objective is copied byte-for-byte, not edited.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import copy
from decimal import Decimal, localcontext
import functools
import hashlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parent
PINS_SHA = "16c76a1c990fa8419fb396b745b1d810748ea78aee438ad22d793189c8890faf"
OBJECTIVE_SHA = "1321eef19cf02d8ce8009e307d4fc384eeac7f59a57e2e07c05dbf79d273350a"
DECODER_SHA = "0ff27034c441b12b69f8b0bf670262116801cd72e0e134fa04a6b09f13029975"
PRIOR_STDLIB_REPORT_SHA = "dbb0acacdc87dfdd9a62f3cefae79c94d1bf87ae4a957d4a42785da2339adeb1"
EXPECTED_TORCH = "2.9.1+cpu"
EXPECTED_TEST_CASES = 10
RUNTIME = None
DECODER = None
OBSERVATIONS = {}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_originals():
    raw = (ROOT / "ORIGINAL_PINS.json").read_bytes()
    need(sha(raw) == PINS_SHA, "exact copied original-pin manifest")
    pins = json.loads(raw)
    for item in pins["entries"]:
        path = ROOT / item["path"]
        need(path.is_file() and not path.is_symlink(), "ordinary frozen source/report file")
        content = path.read_bytes()
        need(len(content) == item["bytes"] and sha(content) == item["sha256"],
             "original source bytes must never be silently fixed or substituted")
    need(sha((ROOT / "frozen/torch_objective_draft.py").read_bytes()) == OBJECTIVE_SHA,
         "exact preexisting objective source")
    need(sha((ROOT / "frozen/STDLIB_VALIDATION.json").read_bytes()) == PRIOR_STDLIB_REPORT_SHA,
         "retain the already executed stdlib proof without rerunning its exhaustive census")
    return pins


def load_actual_decoder_ast():
    path = ROOT / "frozen/originals/english_v6_b24q1_primary_eval.py"
    raw = path.read_bytes()
    need(sha(raw) == DECODER_SHA, "actual preserved event-decoder source pin")
    nodes = [node for node in ast.parse(raw.decode()).body
             if isinstance(node, ast.FunctionDef) and node.name == "ctc_events"]
    need(len(nodes) == 1 and not nodes[0].decorator_list, "one exact actual decoder function")
    namespace = {"need": need, "__builtins__": {
        "all": all, "dict": dict, "enumerate": enumerate, "int": int,
        "len": len, "list": list, "max": max, "range": range, "type": type,
    }}
    selected = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(nodes[0])], type_ignores=[]))
    exec(compile(selected, "<frozen-ctc-events-only>", "exec"), namespace)
    return namespace["ctc_events"]


@functools.lru_cache(maxsize=8)
def classified_paths(frames):
    need(DECODER is not None and 0 <= frames <= 6, "small invented-path enumeration only")
    live, hit = [], []
    for path in itertools.product(range(4), repeat=frames):
        (live if DECODER(list(path))["events"] == 0 else hit).append(path)
    return tuple(live), tuple(hit)


def probabilities(rows):
    result = []
    for row in rows:
        maximum = max(row)
        weights = [math.exp(value - maximum) for value in row]
        total = math.fsum(weights)
        result.append([value / total for value in weights])
    return result


def enumerated_oracle(rows):
    """Independent exact-path sum and conditional occupancies; no DFA code."""
    p = probabilities(rows)
    paths, hits = classified_paths(len(rows))
    weights = [math.prod(p[frame][token] for frame, token in enumerate(path)) for path in paths]
    z = math.fsum(weights)
    h = math.fsum(math.prod(p[frame][token] for frame, token in enumerate(path)) for path in hits)
    need(z > 0, "moderate oracle inputs retain representable live mass")
    loss = math.log1p(h / z)
    occupancy = [[math.fsum(weight for path, weight in zip(paths, weights) if path[frame] == token) / z
                  for token in range(4)] for frame in range(len(rows))]
    gradient = [[p[frame][token] - occupancy[frame][token] for token in range(4)] for frame in range(len(rows))]
    return loss, gradient


def decimal_oracle(rows):
    """120-digit path enumeration, avoiding subtraction from one."""
    with localcontext() as ctx:
        ctx.prec = 120
        p = []
        for row in rows:
            values = [Decimal(str(value)) for value in row]
            maximum = max(values)
            weights = [(value - maximum).exp() for value in values]
            total = sum(weights, Decimal(0))
            p.append([value / total for value in weights])
        masses = []
        for group in classified_paths(len(rows)):
            total = Decimal(0)
            for path in group:
                weight = Decimal(1)
                for frame, token in enumerate(path):
                    weight *= p[frame][token]
                total += weight
            masses.append(total)
        z, h = masses
        return float(((z + h) / z).ln())


def invented_rows(frames, seed=20261125):
    generator = random.Random(seed)
    return [[generator.uniform(-1.5, 1.5) for _ in range(4)] for _ in range(frames)]


class PureMathGuard:
    def __init__(self):
        self.restore = []
        self.blocked = []
        self.autograd_attempts = 0
        self.autograd_completed = 0

    def patch(self, obj, name, label):
        original = getattr(obj, name)
        def forbidden(*args, **kwargs):
            self.blocked.append(label)
            raise RuntimeError("Forbidden in a pure invented-logit test: " + label)
        self.restore.append((obj, name, original))
        setattr(obj, name, forbidden)

    def __enter__(self):
        for name in ("connect", "connect_ex", "bind", "listen", "sendto"):
            self.patch(socket.socket, name, "socket." + name)
        for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
            self.patch(socket, name, "socket." + name)
        return self

    def after_torch_import(self, torch):
        self.patch(subprocess, "Popen", "subprocess.Popen")
        self.patch(torch, "load", "torch.load")
        self.patch(torch.jit, "load", "torch.jit.load")
        self.patch(torch.hub, "load", "torch.hub.load")
        self.patch(torch.nn.Module, "__call__", "nn.Module.__call__")
        self.patch(torch.optim.Optimizer, "__init__", "Optimizer construction")
        original_grad = torch.autograd.grad
        @functools.wraps(original_grad)
        def counted_grad(*args, **kwargs):
            self.autograd_attempts += 1
            result = original_grad(*args, **kwargs)
            self.autograd_completed += 1
            return result
        self.restore.append((torch.autograd, "grad", original_grad))
        torch.autograd.grad = counted_grad
        # Runtime source/report reads remain allowed; model/audio artifacts do
        # not. No test needs any of these file formats.
        forbidden_suffixes = {".pt", ".pth", ".onnx", ".safetensors", ".ckpt",
                              ".flac", ".wav", ".pcm", ".mp3", ".npy", ".npz"}
        # Path.open/read_bytes use io.open rather than builtins.open.
        for obj, name in ((builtins, "open"), (io, "open")):
            original = getattr(obj, name)
            def guarded_open(file, *args, _original=original, **kwargs):
                if isinstance(file, (str, bytes, Path)):
                    text = file.decode() if isinstance(file, bytes) else str(file)
                    if Path(text).suffix.lower() in forbidden_suffixes:
                        self.blocked.append("model/audio artifact open")
                        raise RuntimeError("Model/audio artifacts are forbidden in these tests")
                return _original(file, *args, **kwargs)
            self.restore.append((obj, name, original))
            setattr(obj, name, guarded_open)

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.restore):
            setattr(obj, name, original)


class FixedInventedLogitSource:
    """An ordinary callable returning supplied toy logits; never an nn.Module."""
    total_calls = 0
    def __init__(self, raw, sentinel):
        self.raw, self.sentinel, self.calls = raw, sentinel, 0

    def __call__(self, value):
        need(value is self.sentinel, "only the invented non-audio sentinel is accepted")
        self.calls += 1
        type(self).total_calls += 1
        return self.raw


class ObjectiveMathematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need(RUNTIME is not None, "use the explicit --run-pure-math entry point")
        cls.torch, cls.F, cls.objective = RUNTIME

    def tensor(self, rows, *, dtype=None, gradients=True):
        return self.torch.tensor(rows, dtype=dtype or self.torch.float64, device="cpu", requires_grad=gradients)

    def close(self, actual, expected, *, atol=1e-12, rtol=1e-10):
        self.torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_small_enumeration_and_conditional_gradient(self):
        errors = []
        for frames in (3, 4, 5):
            rows = invented_rows(frames, 20261125 + frames)
            expected, gradient = enumerated_oracle(rows)
            logits = self.tensor(rows)
            result = self.objective.no_event_nll_torch(logits, self.torch)
            self.assertEqual(result.ndim, 0)
            self.assertEqual(result.dtype, self.torch.float64)
            self.close(result.detach(), self.tensor(expected, gradients=False))
            actual = self.torch.autograd.grad(result, logits)[0]
            oracle = self.tensor(gradient, gradients=False)
            self.close(actual, oracle, atol=2e-12, rtol=2e-10)
            errors.append(float((actual - oracle).abs().max()))
            self.close(actual.sum(-1), self.torch.zeros(frames, dtype=self.torch.float64))
        OBSERVATIONS["small_path_gradient_max_abs_error"] = max(errors)

    def test_double_gradcheck_scalar_and_batch(self):
        torch = self.torch
        scalar = self.tensor(invented_rows(4, 17))
        batch = self.tensor([invented_rows(4, 18), invented_rows(4, 19)])
        function = lambda value: self.objective.no_event_nll_torch(value, torch)
        for value in (scalar, batch):
            self.assertTrue(torch.autograd.gradcheck(function, (value,), eps=1e-6,
                atol=2e-6, rtol=2e-4, raise_exception=True, fast_mode=False))
        OBSERVATIONS["double_gradcheck_shapes"] = [list(scalar.shape), list(batch.shape)]

    def test_independent_finite_differences(self):
        rows = invented_rows(4, 20261126)
        logits = self.tensor(rows)
        result = self.objective.no_event_nll_torch(logits, self.torch)
        actual = self.torch.autograd.grad(result, logits)[0].tolist()
        maximum, epsilon = 0.0, 1e-5
        for frame in range(4):
            for token in range(4):
                plus, minus = copy.deepcopy(rows), copy.deepcopy(rows)
                plus[frame][token] += epsilon
                minus[frame][token] -= epsilon
                # Numerically differentiate the independent enumerated oracle,
                # not the objective implementation being checked.
                derivative = (enumerated_oracle(plus)[0] - enumerated_oracle(minus)[0]) / (2 * epsilon)
                maximum = max(maximum, abs(actual[frame][token] - derivative))
        self.assertLess(maximum, 2e-9)
        OBSERVATIONS["independent_enumerated_finite_difference_max_error"] = maximum

    def test_extreme_values_and_rare_event_gradient(self):
        torch = self.torch
        cases = [
            [[0.0] * 4 for _ in range(3)],
            [[0.0, -30.0, -100.0, -100.0], [0.0, -100.0, -30.0, -100.0], [0.0, -100.0, -100.0, -30.0]],
            [[-1000.0, 0.0, -1000.0, -1000.0], [-1000.0, -1000.0, 0.0, -1000.0], [-1000.0, -1000.0, -1000.0, 0.0]],
            [[-10000.0, 10000.0, -10000.0, -10000.0], [-10000.0, -10000.0, 10000.0, -10000.0], [-10000.0, -10000.0, -10000.0, 10000.0]],
        ]
        values = []
        for index, rows in enumerate(cases):
            logits = self.tensor(rows)
            loss = self.objective.no_event_nll_torch(logits, torch)
            expected = decimal_oracle(rows)
            self.assertTrue(math.isclose(float(loss.detach()), expected, rel_tol=5e-12, abs_tol=0.0))
            gradient = torch.autograd.grad(loss, logits)[0]
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertLessEqual(float(gradient.abs().max()), 1.0 + 1e-12)
            self.close(gradient.sum(-1), torch.zeros(3, dtype=torch.float64), atol=2e-10)
            if index == 1:
                self.assertGreater(float(loss.detach()), 0.0)
                p = probabilities(rows)
                event = p[0][1] * p[1][2] * p[2][3]
                factor = event / (1.0 - event)
                analytic = [[factor * ((1.0 if token == frame + 1 else 0.0) - p[frame][token])
                             for token in range(4)] for frame in range(3)]
                self.close(gradient, self.tensor(analytic, gradients=False),
                           atol=expected * 1e-14, rtol=2e-10)
                self.assertGreater(float(gradient.abs().max()), expected * 0.5)
            values.append({"case": index, "nll": float(loss.detach()), "decimal_nll": expected})
        tiny = self.tensor([[0.0, -1000.0, -1000.0, -1000.0] for _ in range(3)])
        tiny_loss = self.objective.no_event_nll_torch(tiny, torch)
        self.assertEqual(float(tiny_loss.detach()), 0.0)
        tiny_gradient = torch.autograd.grad(tiny_loss, tiny)[0]
        self.assertTrue(bool(torch.isfinite(tiny_gradient).all()))
        self.assertTrue(torch.equal(tiny_gradient, torch.zeros_like(tiny)))
        OBSERVATIONS["extreme_values"] = values
        OBSERVATIONS["below_float64_range"] = "Expected final NLL/gradient underflow; no epsilon or clamp introduced."

    def test_short_sequences_have_connected_exact_zero(self):
        torch = self.torch
        checks = []
        for frames in (0, 1, 2):
            for batch in (None, 2):
                shape = (frames, 4) if batch is None else (batch, frames, 4)
                logits = torch.zeros(shape, dtype=torch.float64, device="cpu", requires_grad=True)
                loss = self.objective.no_event_nll_torch(logits, torch)
                self.assertEqual(tuple(loss.shape), () if batch is None else (batch,))
                self.assertTrue(loss.requires_grad)
                self.assertTrue(torch.equal(loss, torch.zeros_like(loss)))
                gradient = torch.autograd.grad(loss.sum(), logits, allow_unused=False)[0]
                self.assertTrue(torch.equal(gradient, torch.zeros_like(logits)))
                checks.append(list(shape))
        OBSERVATIONS["connected_zero_shapes"] = checks

    def test_batch_independence_shift_and_fp32_boundary(self):
        torch = self.torch
        rows = [invented_rows(4, 31), invented_rows(4, 32)]
        logits = self.tensor(rows)
        loss = self.objective.no_event_nll_torch(logits, torch)
        self.assertEqual(tuple(loss.shape), (2,))
        first = torch.autograd.grad(loss[0], logits, retain_graph=True)[0]
        second = torch.autograd.grad(loss[1], logits)[0]
        self.assertTrue(torch.equal(first[1], torch.zeros_like(first[1])))
        self.assertTrue(torch.equal(second[0], torch.zeros_like(second[0])))
        shifts = self.tensor([[[1000.0], [-1000.0], [3000.0], [-3000.0]]], gradients=False)
        shifted = self.objective.no_event_nll_torch(logits.detach() + shifts, torch)
        self.close(shifted, loss.detach(), atol=2e-12, rtol=2e-10)
        fp32 = self.tensor(rows[0], dtype=torch.float32)
        fp32_loss = self.objective.no_event_nll_torch(fp32, torch)
        self.assertEqual(fp32_loss.dtype, torch.float64)
        self.assertEqual(torch.autograd.grad(fp32_loss, fp32)[0].dtype, torch.float32)

    def test_long_uniform_log_domain_and_gradients(self):
        torch = self.torch
        frames = 256
        a, b, c, d, h = 1, 0, 0, 0, 0
        for _ in range(frames):
            a, b, c, d, h = 3*a+b+d, a+2*b+c+d, b+c, c+d, 4*h+c+d
        self.assertEqual(a+b+c+d+h, 4**frames)
        expected = frames * math.log(4.0) - math.log(a+b+c+d)
        logits = torch.zeros((frames, 4), dtype=torch.float64, device="cpu", requires_grad=True)
        loss = self.objective.no_event_nll_torch(logits, torch)
        self.assertAlmostEqual(float(loss.detach()), expected, delta=2e-10)
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.close(gradient.sum(-1), torch.zeros(frames, dtype=torch.float64), atol=2e-11)
        OBSERVATIONS["long_uniform"] = {"frames": frames, "nll": float(loss.detach()), "exact_count_nll": expected}

    def test_invalid_input_rejections(self):
        torch = self.torch
        bad = [[], torch.zeros((3, 3), dtype=torch.float64), torch.zeros((1, 2, 3, 4), dtype=torch.float64),
               torch.zeros((0, 3, 4), dtype=torch.float64), torch.zeros((3, 4), dtype=torch.int64)]
        for value in (math.nan, math.inf, -math.inf):
            rows = [[0.0] * 4 for _ in range(3)]
            rows[1][2] = value
            bad.append(self.tensor(rows, gradients=False))
        bad.append(self.tensor([[1e308, -1e308, 0.0, 0.0]] * 3, gradients=False))
        # This softmax is finite but an accumulated path can exceed the
        # representability bound, so it exercises the distinct range guard.
        bad.append(self.tensor([[0.0, -1e308, 0.0, 0.0]] * 3, gradients=False))
        for logits in bad:
            with self.assertRaises((TypeError, ValueError)):
                self.objective.no_event_nll_torch(logits, torch)
        OBSERVATIONS["invalid_Torch_inputs_rejected"] = len(bad)

    def test_natural_wrapper_single_callable_value_and_gradient(self):
        torch = self.torch
        # prefix=1600 -> blank=2, start=7; T=13 leaves a six-frame suffix.
        # These are invented dimensions, not PCM, features or a model input.
        rows = invented_rows(13, 20261127)
        rows[0], rows[1] = [2.0, -2.0, -2.0, -2.0], [-1.0, 0.4, 0.2, -0.3]
        observations = []
        for tier, factor in (("silver", 1.0), ("bronze", 0.75)):
            raw = self.tensor([rows], dtype=torch.float32)
            rounded = raw.detach().tolist()[0]
            suffix_value, suffix_gradient = enumerated_oracle(rounded[7:])
            prefix_probabilities = probabilities(rounded[:2])
            prefix_losses = [-math.log(row[0]) for row in prefix_probabilities]
            self.assertGreater(prefix_losses[1] - prefix_losses[0], 0.1)
            expected_value = factor * (0.6 * suffix_value + 0.5 * prefix_losses[1])
            sentinel = object()
            source = FixedInventedLogitSource(raw, sentinel)
            loss, returned_logp = self.objective.natural_no_event_suffix_loss(
                source, sentinel, 1600, tier, torch, self.F, True)
            self.assertEqual(source.calls, 1)
            self.assertEqual(returned_logp.dtype, torch.float32)
            self.assertTrue(torch.equal(returned_logp, raw.log_softmax(-1)))
            self.assertEqual(loss.dtype, torch.float64)
            self.assertAlmostEqual(float(loss.detach()), expected_value, delta=2e-6)
            actual_gradient = torch.autograd.grad(loss, raw)[0]
            expected_gradient = [[0.0] * 4 for _ in range(13)]
            expected_gradient[1] = [factor * 0.5 * (value - (1.0 if token == 0 else 0.0))
                                    for token, value in enumerate(prefix_probabilities[1])]
            for frame in range(6):
                expected_gradient[7 + frame] = [factor * 0.6 * value for value in suffix_gradient[frame]]
            self.close(actual_gradient, self.tensor([expected_gradient], dtype=torch.float32, gradients=False),
                       atol=3e-7, rtol=3e-5)
            self.assertTrue(torch.equal(actual_gradient[:, 2:7], torch.zeros_like(actual_gradient[:, 2:7])))
            self.assertEqual(source.calls, 1, "autograd must not cause another callable invocation")
            direct = FixedInventedLogitSource(raw.detach(), sentinel)
            without_logp = self.objective.natural_no_event_suffix_loss(
                direct, sentinel, 1600, tier, torch, self.F, False)
            self.assertEqual(direct.calls, 1)
            self.close(without_logp, loss.detach())
            observations.append({"tier": tier, "calls_per_loss_invocation": 1,
                                 "nll": float(loss.detach()), "independent_expected_loss": expected_value})
        OBSERVATIONS["natural_only_interface"] = observations

    def test_unchanged_sources_and_no_complete_fit_substitution(self):
        verify_originals()
        tree = ast.parse((ROOT / "frozen/torch_objective_draft.py").read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("fit", functions)
        self.assertNotIn("prefix_peak_suffix_loss", functions)
        wrapper = functions["natural_no_event_suffix_loss"]
        calls = [node for node in ast.walk(wrapper) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "model"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0]), "model(x)")
        old_tree = ast.parse((ROOT / "frozen/originals/english_v6_b23_prefix_peak.py").read_text())
        old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
        shared = old_functions["prefix_peak_suffix_loss"]
        self.assertEqual([arg.arg for arg in wrapper.args.args],
                         [arg.arg for arg in shared.args.args if arg.arg != "klass"])
        old_calls = [node for node in ast.walk(old_functions["fit"]) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "prefix_peak_suffix_loss"]
        self.assertEqual(len(old_calls), 2)
        natural_calls = [node for node in old_calls
                         if any(isinstance(item, ast.Name) and item.id == "natural_cache"
                                for item in ast.walk(node.args[1]))]
        self.assertEqual(len(natural_calls), 1)
        self.assertEqual(ast.unparse(natural_calls[0].args[2]), "0")
        self.assertEqual(ast.unparse(natural_calls[0].args[-1]), "True")
        self.assertEqual([ast.unparse(arg) for arg in natural_calls[0].args[3:-1]],
                         ["prefix", "row['tier']", "torch", "F"])
        synthetic_calls = [node for node in old_calls if node not in natural_calls]
        self.assertEqual(ast.unparse(synthetic_calls[0].args[2]), "klass")
        weight_tree = ast.parse((ROOT / "frozen/originals/english_v6_b27_natural_weight.py").read_text())
        weights = [node.value.value for node in weight_tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "NATURAL_WEIGHT" for target in node.targets)
                   and isinstance(node.value, ast.Constant)]
        self.assertEqual(weights, [0.16])
        OBSERVATIONS["frozen_fit_interface_inspected_only"] = {
            "original_shared_loss_call_sites": 2, "original_natural_call_sites": 1,
            "natural_wrapper_omits_only_klass_argument": True,
            "outer_natural_weight_unchanged": 0.16, "complete_fit_execution_or_rewrite": False,
        }
        self.assertEqual(DECODER([1, 2, 0, 2, 3])["events"], 0)
        self.assertEqual(DECODER([1, 0, 2, 0, 3])["events"], 1)


def main():
    global RUNTIME, DECODER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pure-math", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "CI_MATH_RESULT.json")
    args = parser.parse_args()
    if not args.run_pure_math:
        parser.error("Torch is not imported unless --run-pure-math is explicitly supplied")
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    need(report_path.parent.resolve() == ROOT and report_path.name == "CI_MATH_RESULT.json",
         "only this independent draft directory's CI_MATH_RESULT.json may be written")
    need(not report_path.exists() and not report_path.is_symlink(), "a fresh result path is required")
    sys.dont_write_bytecode = True
    pins = verify_originals()
    DECODER = load_actual_decoder_ast()
    report = {"schema": "b28-invented-logit-math-test-result-v1", "preparation_only": True,
              "original_objective_sha256": OBJECTIVE_SHA, "original_source_changed": False,
              "original_files_verified": len(pins["entries"]),
              "candidate_registration": False, "training_or_promotion_authority": False,
              "prior_exhaustive_stdlib_report_sha256": PRIOR_STDLIB_REPORT_SHA}
    guard = PureMathGuard()
    successful = False
    try:
        with guard:
            import torch
            import torch.nn.functional as F
            need(str(torch.__version__) == EXPECTED_TORCH, "reviewed exact Torch CPU runtime")
            torch.set_num_threads(1)
            guard.after_torch_import(torch)
            spec = importlib.util.spec_from_file_location("frozen_b28_objective_under_test", ROOT / "frozen/torch_objective_draft.py")
            objective = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(objective)
            RUNTIME = (torch, F, objective)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObjectiveMathematics)
            expected_tests = EXPECTED_TEST_CASES
            need(suite.countTestCases() == expected_tests, "all ten reviewed mathematical test methods are present")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            successful = result.wasSuccessful() and result.testsRun == expected_tests and not result.skipped and not guard.blocked
            report.update(status="PASS" if successful else "FAIL", torch_version=str(torch.__version__),
                          python_version=sys.version.split()[0], device="cpu", torch_threads=torch.get_num_threads(),
                          tests_run=result.testsRun, expected_test_cases=expected_tests,
                          failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                          observations=OBSERVATIONS)
    except BaseException as error:
        report.update(status="ERROR", error_type=type(error).__name__, error=str(error))
    report.update(blocked_forbidden_attempts=guard.blocked,
                  toy_autograd_executed=guard.autograd_completed > 0,
                  toy_autograd_grad_attempts=guard.autograd_attempts,
                  toy_autograd_grad_completed_calls=guard.autograd_completed,
                  invented_logit_callable_invocations=FixedInventedLogitSource.total_calls,
                  real_model_forwards=0, real_weight_loads=0, audio_decodes=0, optimizer_updates=0,
                  real_training_runs=0, source_or_observer_calls=0, network_operations_allowed=False,
                  repository_refs_modified=False,
                  operation_scope_basis="Reviewed pinned test code; runtime guards; invented tensors and an ordinary fixed-logit callable only.",
                  test_source_sha256=sha(Path(__file__).read_bytes()), original_pins_sha256=PINS_SHA)
    output = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with report_path.open("xb") as stream:
        stream.write(output)
    print(json.dumps({"status": report["status"], "report": str(report_path), "bytes": len(output), "sha256": sha(output)}, sort_keys=True))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
