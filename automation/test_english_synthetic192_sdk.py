"""Source/negative-path checks only; no model execution or fake successful SDK."""
import copy
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sdk_assembly_under_test', ROOT/'package_english_synthetic192_sdk.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
v = p.v


class OnlyResult:
    def __init__(self, value): self.value = value
    def json(self, path):
        if path != 'evidence/RESULT.json': raise RuntimeError('Validation passed the expected rejection boundary')
        return self.value


def public_api(kind='training'):
    source = {'run_id': 1, 'artifact_id': 2, 'head_sha': 'a'*40,
        'name': 'english-synthetic192-' + kind + '-1', 'bytes': 123, 'sha256': 'b'*64}
    run = {'id': 1, 'head_sha': 'a'*40, 'status': 'completed', 'conclusion': 'success',
        'run_attempt': 1, 'head_branch': 'main', 'event': 'push',
        'repository': {'private': False, 'full_name': v.REPO}}
    artifact = {'id': 2, 'name': source['name'], 'expired': False, 'size_in_bytes': 123,
        'digest': 'sha256:' + 'b'*64, 'workflow_run': {'id': 1, 'head_sha': 'a'*40}}
    return source, run, artifact


class SDKContract(unittest.TestCase):
    def test_disabled_source_has_no_future_artifact_identity(self):
        plan = p.fixed_plan(); p.source_preflight(plan)
        self.assertFalse(plan['enabled'])
        self.assertIsNone(plan['api']); self.assertIsNone(plan['candidate_head'])

    def test_plan_must_be_exact_and_disabled_cannot_claim_success(self):
        for key, value in [('enabled', True), ('fullpcm_processes', 10), ('new_model_calls', 1)]:
            plan = p.fixed_plan(); plan[key] = value
            with self.assertRaises(ValueError): p.source_preflight(plan)

    def test_all_ten_fixed_gates_include_actual_quality_gain(self):
        self.assertEqual(len(set(v.GATES)), 10)
        self.assertIn('locked_test_strict_positive_gain', v.GATES)
        self.assertIn('source_balanced_reviewed_ce_decreased', v.GATES)

    def test_legacy_teacher_success_status_is_never_new_synthetic_success(self):
        result = {'schema': 'english-train-teacher-retention-once-v1',
            'status': 'RETAINED_RESEARCH_TEACHER_RETENTION_HEAD', 'passed': True,
            'research_retention_allowed': True, 'numerical_execution_completed': True}
        with self.assertRaisesRegex(ValueError, 'RETAINED_NEW_SYNTHETIC'):
            v.audit_training(OnlyResult(result), {}, {}, {}, None, None)

    def test_failed_new_candidate_is_rejected_before_any_weight_access(self):
        result = {'schema': v.TRAIN_SCHEMA, 'status': 'REJECTED_SYNTHETIC192_QUALITY_GATE',
            'passed': False, 'research_retention_allowed': False, 'numerical_execution_completed': True}
        with self.assertRaisesRegex(ValueError, 'RETAINED_NEW_SYNTHETIC'):
            v.audit_training(OnlyResult(result), {}, {}, {}, None, None)

    def test_one_false_or_missing_gate_blocks_retention(self):
        source, _, _ = public_api()
        result = {'schema': v.TRAIN_SCHEMA, 'status': 'RETAINED_RESEARCH_SYNTHETIC192_HEAD',
            'passed': True, 'research_retention_allowed': True, 'numerical_execution_completed': True,
            'locked_test_opened': True, 'repository': v.REPO, 'public_repository': True,
            'run_id': '1', 'head_sha': 'a'*40, 'run_attempt': 1,
            'gates': dict.fromkeys(v.GATES, True)}
        for gate in v.GATES:
            bad = copy.deepcopy(result); bad['gates'][gate] = False
            with self.assertRaisesRegex(ValueError, 'ALL_TEN_TRAINING_GATES'):
                v.audit_training(OnlyResult(bad), source, {}, {}, None, None)
        del result['gates']['locked_test_strict_positive_gain']
        with self.assertRaisesRegex(ValueError, 'ALL_TEN_TRAINING_GATES'):
            v.audit_training(OnlyResult(result), source, {}, {}, None, None)

    def test_old_coverage_fullpcm_cannot_certify_new_head(self):
        result = {'schema': 'english-coverage-fullpcm-once-v1/result',
            'status': 'VERIFIED_NEW_COVERAGE_INSTALLED_C_FULLPCM', 'passed': True}
        with self.assertRaisesRegex(ValueError, 'SUCCESSFUL_SAME_NEW_MODEL_FULLPCM'):
            v.audit_fullpcm(OnlyResult(result), {}, {}, None, {}, {}, None)

    def test_api_raw_bytes_and_semantics_both_required(self):
        source, run, artifact = public_api()
        rr, ar = v.canonical(run), v.canonical(artifact)
        binding = {'source': source, 'run_json': v.pin(rr), 'artifact_json': v.pin(ar)}
        v.audit_api(rr, ar, binding, 'training')
        with self.assertRaisesRegex(ValueError, 'EXACT_API_RESPONSE_BYTES'):
            v.audit_api(rr + b' ', ar, binding, 'training')
        for key, value in [('conclusion', 'failure'), ('run_attempt', 2), ('event', 'workflow_dispatch')]:
            bad = dict(run, **{key: value}); raw = v.canonical(bad)
            with self.assertRaisesRegex(ValueError, 'ACTUAL_SUCCESSFUL_PUBLIC_RUN'):
                v.audit_api(raw, ar, dict(binding, run_json=v.pin(raw)), 'training')

    def test_failed_or_mismatched_artifact_rejected_even_with_updated_response_pin(self):
        source, run, artifact = public_api(); rr = v.canonical(run)
        for field, value in [('expired', True), ('digest', 'sha256:' + 'c'*64), ('size_in_bytes', 124), ('name', 'english-teacher-retention-1')]:
            ar = v.canonical(dict(artifact, **{field: value}))
            binding = {'source': source, 'run_json': v.pin(rr), 'artifact_json': v.pin(ar)}
            with self.assertRaisesRegex(ValueError, 'ACTUAL_SOURCE_ARTIFACT'):
                v.audit_api(rr, ar, binding, 'training')

    def test_locked_selection_is_class_condition_then_id_not_event_outcome(self):
        rows = []
        for label in (0, 1, 2):
            for condition in ('clean', 'snr10', 'snr5'):
                for suffix in ('z', 'a'):
                    rows.append({'id': f'{label}-{condition}-{suffix}', 'role': 'locked-test',
                        'usable': True, 'candidate_reference_available': True, 'weak_class': label,
                        'condition': condition, 'events': [] if suffix == 'a' else [[label, 1]]})
        doc = {'schema': v.TRAIN_SCHEMA + '/fresh-cases', 'candidate': {'bin': 'example'},
            'rows': rows, 'references_bound_to_selected_weights': True, 'locked_test_opened': True}
        selected = v.selected_fresh(doc, {'bin': 'example'})
        self.assertEqual(len(selected), 9); self.assertTrue(all(row['id'].endswith('-a') for row in selected))
        doc['locked_test_opened'] = False
        with self.assertRaises(ValueError): v.selected_fresh(doc, {'bin': 'example'})

    def test_archive_traversal_symlink_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, name in enumerate(('../outside', '/absolute', 'a\\b')):
                path = root / (str(index) + '.zip')
                with zipfile.ZipFile(path, 'w') as archive: archive.writestr(name, b'bad')
                with self.assertRaises(ValueError): v.Archive(path, v.file_pin(path))
            path = root/'link.zip'; item = zipfile.ZipInfo('link'); item.create_system = 3
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(path, 'w') as archive: archive.writestr(item, b'../outside')
            with self.assertRaises(ValueError): v.Archive(path, v.file_pin(path))

    def test_duplicate_json_keys_and_nonfinite_values_rejected(self):
        for raw in (b'{"passed":false,"passed":true}', b'{"metric":NaN}', b'{"metric":Infinity}'):
            with self.assertRaises(ValueError): v.loads(raw)

    def test_supervisor_missing_cleanup_or_interrupted_child_not_success(self):
        source, _, _ = public_api()
        doc = {'schema': v.TRAIN_SCHEMA + '/supervisor', 'passed': True, 'returncode': 0,
            'timed_out': False, 'error': None, 'cleanup': {'descendants_found': False},
            'retry_count': 0, 'partial_receipt_errors': [], 'repository': v.REPO,
            'run_attempt': 1, 'run_id': '1', 'head_sha': 'a'*40}
        v.audit_supervisor(doc, source, v.TRAIN_SCHEMA)
        for key, value in [('cleanup', {}), ('partial_receipt_errors', ['interrupted']), ('timed_out', True), ('returncode', 1)]:
            with self.assertRaises(ValueError): v.audit_supervisor(dict(doc, **{key: value}), source, v.TRAIN_SCHEMA)


if __name__ == '__main__':
    unittest.main()
