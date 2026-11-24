"""Bounded B22 source/objective fixtures; never load an acoustic checkpoint."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b22_union_retention as train
import english_v6_b22_producer as producer

NUMERIC = '--numeric' in sys.argv
if NUMERIC:
    sys.argv.remove('--numeric')


class Contract(unittest.TestCase):
    def test_single_new_challenger_and_immutable_original_control(self):
        d = train.design()
        self.assertEqual(d['updates'], 192)
        self.assertEqual(d['historical_control']['pt_sha256'], '4947d017142f3b01f5dd4cf7706cc3c6978cf1a072ca022c26ea847ec0dc2f04')
        self.assertEqual(d['budget']['new_historical_control_forwards'], 0)
        self.assertEqual(d['budget']['new_teacher_forwards'], 0)
        self.assertEqual(d['budget']['total_new_model_forwards'], 1847)
        self.assertFalse(d['historical_numerical_replay_claimed'])

    def test_only_two_named_original_train_successes_added(self):
        d = train.design()
        got = {(r['id'], r['gain'], r['teacher']) for r in d['retention_additions']}
        self.assertEqual(got, {('fresh18-am_santa-mero-fast', '1.0', 'B9'),
                               ('fresh24-bf_alice-mero-merrow', '0.75', 'B0')})
        self.assertEqual((d['existing_retention_conditions'], d['union_retention_conditions']), (120, 122))
        self.assertEqual((d['retention_groups'], d['retention_originals']), (34, 43))
        self.assertEqual(d['ordinary_ceiling_weight'], .08)
        self.assertEqual(d['positive_retention_weight'], .08)

    def test_original_complete_producer_used_instead_of_ast_only(self):
        manifest = json.loads(train.OLD_PRODUCER.read_bytes())
        self.assertEqual(len(manifest['files']), 221)
        self.assertEqual(train.sha(train.OLD_PRODUCER.read_bytes()), train.design()['original_b21_producer_manifest_sha256'])
        self.assertIn('automation/english_v6_b21_competing_margin_pair.py', manifest['files'])
        self.assertIn('automation/english_v6_b22_union_retention.py', producer.NEW_FILES)

    def test_inert_assessment_stops_before_network_or_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            obj = dict(schema='english-v6-b22-prospective-assessment-binding-v1', frozen=False,
                recipe_path='automation/english_v6_b22_union_retention_design.json', recipe_sha256=train.DESIGN_SHA,
                assessment=None, files={}, optimizer_allowed=False)
            binding = root / 'binding.json'
            binding.write_text(json.dumps(obj))
            with patch.object(train, 'BINDING', binding), patch.object(train, 'gh', side_effect=AssertionError('no network')):
                with self.assertRaisesRegex(ValueError, 'new independent assessment'):
                    train.fetch(root / 'inputs', '1' * 40)
            self.assertFalse((root / 'inputs').exists())

    def test_fixed_architecture_has_no_stochastic_forward(self):
        value = train.architecture_contract()
        self.assertFalse(value['dropout'])
        self.assertFalse(value['batchnorm'])
        self.assertFalse(value['random_forward'])

    def test_old_objective_formula_and_old_ceiling_call_are_preserved(self):
        tree = ast.parse(Path(train.__file__).read_bytes())
        fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        calls = [n for n in ast.walk(fit) if isinstance(n, ast.Call)]
        names = [ast.unparse(n.func) for n in calls]
        self.assertIn('old.b17.ceiling_loss', names)
        self.assertIn('old.b10.b6.suffix_loss', names)
        self.assertIn('torch.nn.utils.clip_grad_norm_', names)
        self.assertNotIn('old.fit', names)
        self.assertNotIn('old.b17.make_ordinary_teachers', names)

    def test_exact_parent_and_no_replay_identity(self):
        parent, head = '1' * 40, '2' * 40
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=train.REPO, GITHUB_EVENT_NAME='push',
                   GITHUB_REF=train.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=head, GITHUB_RUN_ID='123')
        event = dict(before=parent, after=head, ref=train.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=train.REPO, private=False))
        with patch.object(train, 'assessment_binding', return_value={}):
            self.assertEqual(train.guard(parent, env, event, parent)['run_id'], '123')
            with self.assertRaises(ValueError):
                train.guard(parent, dict(env, GITHUB_RUN_ATTEMPT='2'), event, parent)
            with self.assertRaises(ValueError):
                train.guard(parent, env, dict(event, forced=True), parent)

    def test_template_consumes_before_torch_and_preserves_two_tokens(self):
        text = Path(train.__file__).with_name('english_v6_b22_union_retention_workflow_template.yml').read_text()
        self.assertEqual(text.count('__B22_REVIEWED_TRAIN_PARENT__'), 2)
        self.assertLess(text.index('union_retention.py consume '), text.index('torch==2.9.1+cpu'))
        self.assertLess(text.index('name: english-v6-b22-union-retention-train-consumed'), text.index('torch==2.9.1+cpu'))
        self.assertNotIn('workflow_dispatch', text)


@unittest.skipUnless(NUMERIC, 'explicit numeric fixture mode only')
class Numeric(unittest.TestCase):
    def test_saved_margin_statistics_and_original_loss_agree(self):
        import torch
        student = torch.tensor([[0., 2., -1., 1.], [1., 0., 2., -1.]], requires_grad=True)
        teacher = torch.zeros_like(student)
        stats = train.margin_statistics(student, teacher, torch)
        self.assertAlmostEqual(stats['mean'], 4 / 6)
        self.assertEqual(stats['maximum'], 2.)
        self.assertEqual(stats['active_cells'], 3)
        row = dict(id='fixture', role='train', mono_class=0, source_group_id='fixture', source_pcm_sha256='1' * 64)
        penalty = train.old.b17.ordinary_margin_loss(student, teacher, row, torch)
        penalty.backward()
        self.assertLess(float(student.grad[0, 0]), 0.)
        self.assertGreater(float(student.grad[0, 1]), 0.)

    def test_full_competitor_retention_includes_blank_with_frozen_reference(self):
        import torch
        matrix = torch.tensor([[3., 1., 0., 0.], [3., 0., 1., 0.], [3., 0., 0., 1.]], requires_grad=True)
        reference = torch.tensor([2.], requires_grad=True)
        spans = [(token, token - 1, token, [other for other in range(4) if other != token], reference)
                 for token in (1, 2, 3)]
        class TensorOnly:
            def __call__(self, unused):
                return matrix[None]
        class ReceiptOnly:
            def before(self, **metadata):
                self.metadata = metadata
        meta = dict(gain='0.75', teacher='B0', newly_protected=True)
        trace = []
        loss = train.positive_loss(0, TensorOnly(), {'group': {'fixture': [(None, spans, meta)]}},
                                   ReceiptOnly(), trace, torch)
        loss.backward()
        self.assertAlmostEqual(float(loss), .32)
        self.assertIsNone(reference.grad)
        self.assertLess(float(matrix.grad[0, 1]), 0.)
        self.assertGreater(float(matrix.grad[0, 0]), 0.)
        self.assertTrue(trace[0]['newly_protected'])
        self.assertEqual(trace[0]['teacher'], 'B0')


if __name__ == '__main__':
    unittest.main()
