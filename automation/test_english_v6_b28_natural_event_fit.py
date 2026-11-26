"""B28 integration contracts using source ASTs and invented byte metadata only.

No actual model, audio, Torch tensor, objective recurrence or optimizer runs.
The previously executed mathematical validation is not repeated here.
"""
from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b28_natural_event_fit as fit


def function_node(callable_value):
    return ast.parse(inspect.getsource(callable_value)).body[0]


def fixture_return():
    return dict(step=1, kind='natural_ordinary', id='invented-natural', gain=1.0,
                prefix=6400, array='forward_00004', frames=100, sha256='a' * 64)


def fixture_row():
    return dict(id='invented-natural', source_group_id='invented-reader', tier='silver')


class FitContracts(unittest.TestCase):
    def test_frozen_math_is_unchanged_and_no_numeric_runtime_was_imported(self):
        provenance = fit.contract()
        self.assertEqual(provenance['objective_sha256'], fit.OBJECTIVE_SHA)
        self.assertEqual(provenance['reference_sha256'], fit.REFERENCE_SHA)
        for name, digest in fit.DEPENDENCIES.items():
            self.assertEqual(fit.sha((fit.HERE / name).read_bytes()), digest, name)
        self.assertTrue({'torch', 'numpy', 'onnx', 'onnxruntime'}.isdisjoint(sys.modules))

    def test_private_natural_weight_and_shared_helpers_cannot_change_old_recipe(self):
        self.assertEqual((fit.old.NATURAL_WEIGHT, fit.b24.old.NATURAL_WEIGHT), (0.16, 0.08))
        self.assertEqual(set(vars(fit.old)), set(vars(fit.b24.old)))
        for name, value in vars(fit.b24.old).items():
            if name != 'NATURAL_WEIGHT':
                self.assertIs(getattr(fit.old, name), value, name)
        self.assertIs(fit.prefix_peak_suffix_loss, fit.b24.b23.prefix_peak_suffix_loss)
        self.assertIs(fit.positive_loss, fit.b24.b23.positive_loss)
        self.assertIs(fit.margin_statistics, fit.b24.b23.margin_statistics)
        self.assertIs(fit.Recorder, fit.b24.b23.Recorder)
        self.assertIs(fit.fit_engine(), fit.fit_loop)
        self.assertIsNot(fit.fit_loop.__code__, fit.b24.b23.fit.__code__)

    def test_exact_original_scientific_ast_except_one_natural_call_and_observations(self):
        original = function_node(fit.b24.b23.fit)
        proposed = function_node(fit.fit_loop)
        original.name = proposed.name
        counts = dict(contract=0, draw=0, initialize=0, append=0,
                      natural_write=0, optimizer_write=0, cardinality=0,
                      natural_call=0, natural_field=0)

        class Normalize(ast.NodeTransformer):
            def visit_Expr(self, node):
                call = node.value
                if isinstance(call, ast.Call):
                    if isinstance(call.func, ast.Name) and call.func.id == 'contract':
                        self.assert_empty(call)
                        counts['contract'] += 1
                        return None
                    if isinstance(call.func, ast.Name) and call.func.id == 'need':
                        expression = ast.unparse(call.args[0])
                        if expression == 'sha(encoded(slots)) == COMMON_DRAW_SHA':
                            counts['draw'] += 1
                            return None
                        if expression == 'len(natural_trace) == 384':
                            counts['cardinality'] += 1
                            return None
                    if isinstance(call.func, ast.Attribute) and ast.unparse(call.func) == 'natural_trace.append':
                        counts['append'] += 1
                        expected = "natural_observation(step + 1, row, gain, prefix, recorder.rows[-1], float(loss.detach()))"
                        if len(call.args) != 1 or ast.unparse(call.args[0]) != expected or call.keywords:
                            raise AssertionError('new trace cannot add an objective/model recomputation')
                        return None
                    if isinstance(call.func, ast.Name) and call.func.id == 'write':
                        destination = ast.unparse(call.args[0])
                        if destination == "output / 'NATURAL_OBJECTIVE_TRACE.json'":
                            if ast.unparse(call.args[1]) != 'natural_trace_document(natural_trace)':
                                raise AssertionError('exact new natural trace writer')
                            counts['natural_write'] += 1
                            return None
                        if destination == "output / 'OPTIMIZER_TRACE.json'":
                            expected = 'dict(schema=OPTIMIZER_SCHEMA, count=len(trace), records=trace)'
                            if ast.unparse(call.args[1]) != expected:
                                raise AssertionError('new optimizer schema must wrap unchanged records')
                            counts['optimizer_write'] += 1
                            call.args[1] = ast.Name(id='trace', ctx=ast.Load())
                return self.generic_visit(node)

            @staticmethod
            def assert_empty(call):
                if call.args or call.keywords:
                    raise AssertionError('contract is a source-only read')

            def visit_Assign(self, node):
                if len(node.targets) == 1 and ast.unparse(node.targets[0]) == 'natural_trace':
                    if not isinstance(node.value, ast.List) or node.value.elts:
                        raise AssertionError('new trace starts empty')
                    counts['initialize'] += 1
                    return None
                return self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'natural_no_event_suffix_loss':
                    if len(node.args) != 7 or ast.unparse(node.args[1]) != "natural_cache[row['id']][gain][prefix]":
                        raise AssertionError('only the exact natural_cache call site changes')
                    counts['natural_call'] += 1
                    node.func.id = 'prefix_peak_suffix_loss'
                    node.args.insert(2, ast.Constant(value=0))
                return self.generic_visit(node)

            def visit_keyword(self, node):
                if node.arg == 'natural_no_event':
                    counts['natural_field'] += 1
                    node.arg = 'natural_ctc'
                return self.generic_visit(node)

        normalized = Normalize().visit(proposed)
        self.assertEqual(counts, dict.fromkeys(counts, 1))
        self.assertEqual(ast.dump(normalized, include_attributes=False), ast.dump(original, include_attributes=False))

    def test_existing_forward_is_linked_without_recomputing_a_scalar(self):
        result = fit.natural_observation(1, fixture_row(), 1.0, 6400, fixture_return(), 0.625)
        self.assertEqual(result, dict(step=1, id='invented-natural', group='invented-reader', gain=1.0,
            prefix=6400, tier='silver', array='forward_00004', raw_logits_sha256='a' * 64,
            frames=100, suffix_start=37, prefix_blank_end=32, weighted_loss=0.625))
        self.assertEqual(fit.natural_trace_document([result]), dict(schema=fit.NATURAL_TRACE_SCHEMA,
            objective_schema=fit.OBJECTIVE_SCHEMA, objective_sha256=fit.OBJECTIVE_SHA,
            reference_sha256=fit.REFERENCE_SHA, common_draw_sha256=fit.COMMON_DRAW_SHA,
            count=1, records=[result]))
        self.assertEqual(fit.natural_trace_document([])['count'], 0)

    def test_natural_observation_rejects_mismatched_or_partial_returns(self):
        mutations = [('step', 2), ('kind', 'synthetic_ordinary'), ('id', 'other'), ('gain', 0.5),
                     ('prefix', 12800), ('array', 'forward_4'), ('sha256', 'x' * 64), ('frames', 42)]
        for key, value in mutations:
            with self.subTest(key=key):
                returned = dict(fixture_return(), **{key: value})
                with self.assertRaises(ValueError):
                    fit.natural_observation(1, fixture_row(), 1.0, 6400, returned, 0.625)
        for value in (math.nan, math.inf, -0.5):
            with self.subTest(loss=value), self.assertRaises(ValueError):
                fit.natural_observation(1, fixture_row(), 1.0, 6400, fixture_return(), value)

    def test_draw_mismatch_stops_before_any_runtime_call(self):
        class NoRuntime:
            def __getattr__(self, name):
                raise AssertionError('no model or tensor runtime may be touched: ' + name)
        with self.assertRaisesRegex(ValueError, 'COMMON_DRAW'):
            fit.fit_loop(None, [], None, None, None, None, None, None,
                         NoRuntime(), NoRuntime(), NoRuntime())

    def test_original_export_wrapper_preserves_bytes_and_corrects_parent(self):
        initial, rest, runtime, calls = object(), [object() for _ in range(6)], [object() for _ in range(3)], []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            def fake(*args):
                calls.append(args)
                self.assertEqual(args, (initial, *rest, output, *runtime))
                (output / 'models').mkdir()
                files = {}
                for suffix in ('.pt', '.npz', '.onnx'):
                    name = 'mero43-b23-challenger192' + suffix
                    raw = ('INVENTED EXPORT BYTES: ' + suffix).encode()
                    (output / 'models' / name).write_bytes(raw)
                    files[name] = dict(bytes=len(raw), sha256=fit.sha(raw))
                return dict(parent_pt_sha256=fit.b24.old.B0_PT, optimizer_updates=192,
                            actual_new_model_forwards=1847, files=files)
            with patch.object(fit, 'fit_engine', return_value=fake):
                result = fit.fit(initial, *rest, output, *runtime)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['parent_pt_sha256'], fit.INITIAL_PT)
            self.assertEqual(result['historical_control_pt_sha256'], fit.CONTROL_PT)
            self.assertEqual(result['natural_weight'], 0.16)
            self.assertEqual(result['natural_objective_returns'], 384)
            self.assertFalse(result['restored_optimizer_state'])
            self.assertTrue(result['natural_scalar_promotes_total_loss'])
            self.assertEqual(set(result['files']), {fit.STEM + ext for ext in ('.pt', '.npz', '.onnx')})
            for name, pin in result['files'].items():
                self.assertEqual(fit.sha((output / 'models' / name).read_bytes()), pin['sha256'])
            self.assertFalse(any((output / 'models').glob('mero43-b23-*')))

    def test_objective_source_has_only_one_model_return_and_one_suffix_term(self):
        node = function_node(fit.objective.natural_no_event_suffix_loss)
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertEqual(sum(n.func.id == 'model' for n in calls), 1)
        self.assertEqual(sum(n.func.id == 'no_event_nll_torch' for n in calls), 1)
        self.assertNotIn('ctc_loss', ast.unparse(node))
        tree = ast.parse((fit.HERE / 'english_v6_b28_no_event_objective.py').read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual([ast.unparse(n) for n in imports], ['from __future__ import annotations'])


if __name__ == '__main__':
    unittest.main()
