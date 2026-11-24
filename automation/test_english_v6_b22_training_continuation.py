"""Regression coverage for the original tuple/list STOP; no model operations."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b22_training_origin as origin
import english_v6_b22_training_continuation_producer as producer
import english_v6_b22_union_retention as original
import english_v6_b22_union_retention_continuation as continuation


class Contract(unittest.TestCase):
    def test_original_failure_and_all243_files_remain_authentic(self):
        proof = origin.local()
        self.assertEqual(proof['optimizer_updates'], 0)
        self.assertEqual(proof['new_acoustic_model_forwards'], 0)

    def test_complete_actual120_tuple_list_failure_and_fix(self):
        saved = json.loads((origin.EVIDENCE / 'partial/SOURCE.json').read_bytes())['positive_retention']
        audit = copy.deepcopy(saved)
        for row in audit:
            for key in ('b9_registered_spans', 'matched_complete_spans'):
                row[key] = [tuple(span) for span in row[key]]
        self.assertEqual(len(saved), 120)
        self.assertNotEqual(audit, saved)
        self.assertEqual(json.loads(continuation.encoded(audit)), saved)
        for key in ('id', 'gain', 'pcm_sha256', 'b9_array_sha256'):
            changed = copy.deepcopy(audit)
            changed[0][key] = 'incorrect'
            self.assertNotEqual(json.loads(continuation.encoded(changed)), saved)
        changed = copy.deepcopy(audit)
        changed[0]['b9_registered_spans'][0] = (1, 0, 1)
        self.assertNotEqual(json.loads(continuation.encoded(changed)), saved)

    def test_one_exact_serialization_expression_only(self):
        before = ast.parse(Path(original.__file__).read_bytes())
        after = ast.parse(Path(continuation.__file__).read_bytes())
        old = {n.name: n for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new = {n.name: n for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        changed = [name for name in old if ast.dump(old[name], include_attributes=False)
                   != ast.dump(new[name], include_attributes=False)]
        self.assertEqual(changed, ['guard', 'prepare_union_paths'])
        fixed = copy.deepcopy(new['prepare_union_paths'])
        matches = 0
        for node in ast.walk(fixed):
            if isinstance(node, ast.Compare) and ast.unparse(node.left) == 'json.loads(encoded(audit))':
                node.left = ast.Name(id='audit', ctx=ast.Load())
                matches += 1
        self.assertEqual(matches, 1)
        self.assertEqual(ast.dump(old['prepare_union_paths'], include_attributes=False),
                         ast.dump(fixed, include_attributes=False))

    def test_new_budget_preserves_original_consumption(self):
        self.assertNotEqual(original.MARKER, continuation.MARKER)
        self.assertNotEqual(original.BRANCH, continuation.BRANCH)
        self.assertEqual(continuation.design(), original.design())
        self.assertEqual(continuation.assessment_binding(), original.assessment_binding())
        tree = ast.parse(Path(origin.__file__).read_bytes())
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in {'unlink', 'rmtree', 'remove'}
                             for n in ast.walk(tree)))

    def test_every_initial_file_and_origin_resource_captured(self):
        names = producer.paths()
        previous = json.loads((origin.EVIDENCE / 'PRODUCER.json').read_bytes())['files']
        self.assertTrue(set(previous) <= names)
        self.assertTrue(set(origin.local()['evidence_files']) <= names)
        self.assertIn('automation/english_v6_b22_training_origin.py', names)
        self.assertIn(continuation.WORKFLOW, names)
        self.assertGreater(len(names), 243)

    def test_workflow_matches_inert_template(self):
        root = Path(__file__).resolve().parent.parent
        raw = (root / continuation.WORKFLOW).read_bytes()
        template = (root / 'automation/english_v6_b22_training_continuation_workflow_template.yml').read_bytes()
        token = b'__B22_REVIEWED_TRAIN_CONTINUATION_PARENT__'
        self.assertEqual(template.count(token), 2)
        if raw != template:
            parent = os.environ.get('PREPARED_PARENT', '')
            self.assertRegex(parent, r'^[0-9a-f]{40}$')
            self.assertEqual(raw, template.replace(token, parent.encode()))
        self.assertIn(b"env: {GH_TOKEN: '${{ github.token }}'}", raw)
        self.assertIn(b'english-v6-b22-union-retention-train-continuation-consumed', raw)


if __name__ == '__main__':
    unittest.main()
