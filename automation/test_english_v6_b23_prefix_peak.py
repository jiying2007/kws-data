"""B23 single-factor and closed-until-failure contracts; no acoustic models."""
from __future__ import annotations
import ast
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b23_prefix_peak as train
import english_v6_b23_b22_failure as failure
import english_v6_b23_producer as producer
import english_v6_b22_quality_readback as old_reader
import test_english_v6_b22_quality_readback as fixtures

NUMERIC = '--numeric' in sys.argv
if NUMERIC:
    sys.argv.remove('--numeric')


def function(module, name):
    return next(n for n in ast.parse(Path(module.__file__).read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def syntax(node):
    return ast.dump(node, include_attributes=False)


def failed_fixture():
    members, spec, admission, rows = fixtures.fixture()
    result = fixtures.reader.document(members['RESULT.json'])
    with zipfile.ZipFile(io.BytesIO(members['ALL_FRAME_LOGITS.npz'])) as archive:
        arrays = {name: archive.read(name) for name in archive.namelist()}
    # Lose one protected positive in one gain, preserving the complete denominator.
    row = result['records'][0]
    key = row['id'] + '__1.0__challenger'
    labels = [0] * 17
    raw, body = fixtures.npy(labels)
    arrays[key + '.npy'] = raw
    members['raw_outputs/' + key + '.npy'] = fixtures.npy(labels, True)[0]
    row['gains']['1.0']['challenger'] = dict(array=key, frames=17, sha256=train.sha(body), **fixtures.primary.ctc_events(labels))
    npz = fixtures.zip_bytes(arrays)
    result.update(fixtures.decision.decide(result['records']))
    result.update(logits_sha256=train.sha(npz), replication_open_allowed=False, candidate_engineering_validation_allowed=False)
    members['ALL_FRAME_LOGITS.npz'] = npz
    members['RESULT.json'] = fixtures.reader.canonical(result)
    return members, spec, admission, rows


class Contracts(unittest.TestCase):
    def test_recipe_keeps_all_original_scientific_fields_and_control_bytes(self):
        value = train.design()
        self.assertEqual(value['budget'], train.baseline.design()['budget'])
        self.assertEqual(value['historical_control']['pt_sha256'], '9b1900312430c5c92dee7f47608fc2be866778387d4ed3baedb11f4bcda6a655')
        self.assertEqual(value['opening_condition'], 'only_after_actual_B22_NO_PROMOTION')
        self.assertFalse(value['historical_numerical_replay_claimed'])

    def test_exact_single_prefix_reduction_in_suffix_function(self):
        before = function(train.old.b10.b6, 'suffix_loss')
        after = copy.deepcopy(function(train, 'prefix_peak_suffix_loss'))
        after.name = 'suffix_loss'
        changed = 0
        for node in ast.walk(after):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'F.softplus(blank_margin).amax':
                node.func.attr = 'mean'
                changed += 1
        self.assertEqual(changed, 1)
        self.assertEqual(syntax(before), syntax(after))

    def test_entire_optimizer_and_1844_forward_loop_are_otherwise_identical(self):
        before = function(train.baseline, 'fit')
        after = copy.deepcopy(function(train, 'fit'))
        calls, stems = 0, 0
        for node in ast.walk(after):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'prefix_peak_suffix_loss':
                node.func = ast.parse('old.b10.b6.suffix_loss', mode='eval').body
                calls += 1
            elif isinstance(node, ast.Constant) and node.value == 'mero43-b23-challenger192':
                node.value = 'mero43-b22-challenger192'
                stems += 1
        self.assertEqual((calls, stems), (2, 1))
        self.assertEqual(syntax(before), syntax(after))
        self.assertIs(train.positive_loss, train.baseline.positive_loss)
        self.assertIs(train.Recorder, train.baseline.Recorder)

    def test_failure_saved_array_parser_has_only_opposite_required_verdict(self):
        old = ast.get_source_segment(Path(old_reader.__file__).read_text(), function(old_reader, 'saved_arrays'))
        new = ast.get_source_segment(Path(failure.__file__).read_text(), function(failure, 'saved_failure_arrays'))
        replacements = (
            ('def saved_failure_arrays(', 'def saved_arrays('),
            ('Require exact NO_PROMOTION from every original raw return and unchanged B14.', 'Require an exact PASS from every original raw return and unchanged B14.'),
            ('result["panel_gate"] == "NO_PROMOTION"', 'result["panel_gate"] == "PASS"'),
            ('only complete actual NO_PROMOTION; STOP, partial return, unknown consumption and PASS stay closed', 'only a complete actual PASS; STOP, partial return, unknown consumption or NO_PROMOTION stays closed'),
            ('result["replication_open_allowed"] is False', 'result["replication_open_allowed"] is (panel == "primary")'),
            ('result["candidate_engineering_validation_allowed"] is False', 'result["candidate_engineering_validation_allowed"] is (panel == "replication")'),
            ('failed candidate cannot open any next-quality or engineering stage', 'next-stage flags exactly match this original PASS'),
            ('replay["panel_gate"] == "NO_PROMOTION"', 'replay["panel_gate"] == "PASS"'),
            ('unchanged B14 independently recomputes the original NO_PROMOTION', 'every unchanged original B14 condition independently recomputes to PASS'),
        )
        for source, target in replacements:
            self.assertEqual(new.count(source), 1)
            new = new.replace(source, target)
        self.assertEqual(syntax(ast.parse(old)), syntax(ast.parse(new)))

    def test_valid_failure_recomputes_every_original_array(self):
        members, spec, admission, rows = failed_fixture()
        value = failure.saved_failure_arrays(fixtures.zip_bytes(members), spec, 'primary', admission, rows, fixtures.primary, fixtures.decision)
        self.assertEqual(value['result']['panel_gate'], 'NO_PROMOTION')
        self.assertEqual(value['result']['actual_forwards'], 576)
        self.assertFalse(value['result']['replication_open_allowed'])

    def test_pass_or_forged_no_promotion_cannot_open_new_training(self):
        members, spec, admission, rows = fixtures.fixture()
        for fake_verdict in (False, True):
            copied = dict(members)
            if fake_verdict:
                report = fixtures.reader.document(copied['RESULT.json'])
                report.update(panel_gate='NO_PROMOTION', replication_open_allowed=False, candidate_engineering_validation_allowed=False)
                copied['RESULT.json'] = fixtures.reader.canonical(report)
            with self.assertRaises(ValueError):
                failure.saved_failure_arrays(fixtures.zip_bytes(copied), spec, 'primary', admission, rows, fixtures.primary, fixtures.decision)

    def test_partial_raw_output_and_clock_tampering_reject(self):
        original, spec, admission, rows = failed_fixture()
        for kind in ('missing', 'clock', 'raw', 'unknown'):
            members = dict(original)
            report = fixtures.reader.document(members['RESULT.json'])
            if kind == 'missing': del members[next(k for k in members if k.startswith('attempts/'))]
            if kind == 'raw': members[next(k for k in members if k.startswith('raw_outputs/'))] = fixtures.npy([0] * 17, True)[0]
            if kind == 'clock': report['records'][1]['gains']['1.0']['B0']['event_samples'] = [999]
            if kind == 'unknown': report['consumed_unknown'] = ['one']
            members['RESULT.json'] = fixtures.reader.canonical(report)
            with self.assertRaises(ValueError):
                failure.saved_failure_arrays(fixtures.zip_bytes(members), spec, 'primary', admission, rows, fixtures.primary, fixtures.decision)

    def test_sealed_opening_blocks_before_any_directory_or_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = json.loads(train.OPENING.read_bytes())
            value.update(enabled=False, quality_lock=None)
            opening = root / 'closed.json'
            opening.write_text(json.dumps(value))
            with patch.object(train, 'OPENING', opening), \
                 patch.object(train, 'assessment_binding', return_value={}), \
                 patch.object(train, 'gh', side_effect=AssertionError('no network')):
                with self.assertRaisesRegex(ValueError, 'sealed'):
                    train.fetch(root / 'inputs', '1' * 40)
            self.assertFalse((root / 'inputs').exists())

    def test_execution_identity_cannot_retry_or_force(self):
        parent, head = '1' * 40, '2' * 40
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=train.REPO, GITHUB_EVENT_NAME='push',
            GITHUB_REF=train.BRANCH, GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=head, GITHUB_RUN_ID='123')
        event = dict(before=parent, after=head, ref=train.BRANCH, forced=False, deleted=False,
            repository=dict(full_name=train.REPO, private=False))
        with patch.object(train, 'assessment_binding', return_value={}), patch.object(train, 'opening', return_value={}):
            self.assertEqual(train.guard(parent, env, event, parent)['run_id'], '123')
            for bad_env, bad_event in ((dict(env, GITHUB_RUN_ATTEMPT='2'), event), (env, dict(event, forced=True)),
                (env, dict(event, repository=dict(full_name=train.REPO, private=True)))):
                with self.assertRaises(ValueError): train.guard(parent, bad_env, bad_event, parent)

    def test_original_files_preserved_and_closed_stage_sources_captured(self):
        names = producer.paths()
        original = json.loads((train.ROOT / train.design()['original_b22_producer_manifest_path']).read_bytes())
        self.assertTrue(set(original['files']) <= names)
        self.assertIn('automation/english_v6_b23_b22_failure.py', names)
        self.assertIn('automation/english_v6_b22_quality_readback.py', names)
        self.assertIn('automation/english_v6_b23_opening.json', names)
        self.assertTrue(set(train.assessment_binding()['files']) <= names)

    def test_workflow_durable_budget_precedes_model_install_and_is_inert(self):
        text = (train.ROOT / producer.TEMPLATE).read_text()
        self.assertEqual(text.count(producer.TOKEN.decode()), 2)
        self.assertIn(train.BRANCH.removeprefix('refs/heads/'), text)
        self.assertNotIn('en-v6-b22-union-retention-train-continuation-once', text)
        self.assertLess(text.index('prefix_peak.py consume '), text.index('torch==2.9.1+cpu'))
        self.assertLess(text.index('name: english-v6-b23-prefix-peak-train-consumed'), text.index('torch==2.9.1+cpu'))
        self.assertNotIn('workflow_dispatch', text)
        raw = (train.ROOT / train.WORKFLOW).read_text()
        if raw != text:
            parent = os.environ.get('PREPARED_PARENT', '')
            self.assertRegex(parent, r'^[0-9a-f]{40}$')
            self.assertEqual(raw, text.replace(producer.TOKEN.decode(), parent))


@unittest.skipUnless(NUMERIC, 'explicit numeric tensors only; no checkpoint or acoustic model')
class Numeric(unittest.TestCase):
    def test_known_silent_worst_frame_is_not_diluted_by_prefix_length(self):
        import torch
        import torch.nn.functional as F
        for prefix in (6400, 12800, 19200):
            frames = prefix // 160 + 24
            x = torch.full((1, frames, 4), -12., dtype=torch.float64)
            x[..., 0] = 12.
            x[0, 0] = torch.tensor([0., 3., -12., -12.], dtype=torch.float64)
            original = x.clone().requires_grad_()
            candidate = x.clone().requires_grad_()
            class TensorOnly:
                def __call__(self, value): return value
            before = train.old.b10.b6.suffix_loss(TensorOnly(), original, 0, prefix, 'silver', torch, F)
            after = train.prefix_peak_suffix_loss(TensorOnly(), candidate, 0, prefix, 'silver', torch, F)
            before.backward(); after.backward()
            self.assertGreater(float(after), float(before))
            blank = prefix // 160 - 8
            self.assertAlmostEqual(float(candidate.grad[0, 0, 1] / original.grad[0, 0, 1]), blank, places=9)
            self.assertEqual(float(candidate.grad[0, 1:blank].abs().sum()), 0.)
            self.assertTrue(torch.equal(candidate.grad[:, prefix // 160 - 3:], original.grad[:, prefix // 160 - 3:]))
            self.assertEqual(float(candidate.grad[0, blank:prefix // 160 - 3].abs().sum()), 0.)

    def test_uniform_prefix_keeps_original_loss_and_ctc_for_both_classes(self):
        import torch
        import torch.nn.functional as F
        x = torch.zeros((1, 66, 4), dtype=torch.float64)
        class TensorOnly:
            def __call__(self, value): return value
        for klass in (0, 1):
            for tier in ('silver', 'weak'):
                before, logp = train.old.b10.b6.suffix_loss(TensorOnly(), x, klass, 6400, tier, torch, F, True)
                after, peak_logp = train.prefix_peak_suffix_loss(TensorOnly(), x, klass, 6400, tier, torch, F, True)
                self.assertTrue(torch.equal(logp, peak_logp))
                self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
