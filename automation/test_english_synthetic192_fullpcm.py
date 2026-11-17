"""Saved-byte/static and mock tests. Never execute a model or subprocess."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('teacher_fullpcm_tested', HERE / 'english_synthetic192_fullpcm_once.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
h, inc = m.helpers()


def fixture():
    source = dict(m.PREVIOUS, run_id=123, artifact_id=456, head_sha='a'*40, name='english-synthetic192-training-123')
    plan = m.fixed_plan(h.sha((HERE / 'english_synthetic192_fullpcm_once.py').read_bytes()), source, 'b'*40, h.HEAD, h.NPZ)
    doc = {'schema': 'english-synthetic192-training-once-v1', 'status': 'RETAINED_RESEARCH_SYNTHETIC192_HEAD',
        'passed': True, 'research_retention_allowed': True, 'numerical_execution_completed': True,
        'current_default_model_replaced': False, 'run_id': 123, 'head_sha': 'a'*40, 'run_attempt': 1,
        'repository': m.REPO, 'public_repository': True, 'strong_english_fsmn_ctc': False, 'ssc305_qualification': False,
        'candidate': {'bin': h.HEAD, 'npz': h.NPZ}, 'teacher_train_bag_forwards': 225,
        'teacher_train_head_windows': 8370, 'teacher_development_forwards': 0, 'teacher_reviewed_forwards': 0,
        'gates': dict.fromkeys(m.GATES, True),
        **dict.fromkeys(m.FALSE_FLAGS[:5], False)}
    return plan, doc


def roster_doc(rows):
    return {'schema': 'english-synthetic192-training-once-v1/fresh-cases',
        'rows': rows, 'references_bound_to_selected_weights': True, 'locked_test_opened': True}


class FullPcmContracts(unittest.TestCase):
    def test_disabled_plan_is_source_only(self):
        p = m.fixed_plan(h.sha((HERE / 'english_synthetic192_fullpcm_once.py').read_bytes()))
        m.source_preflight(p, h)
        self.assertFalse(p['enabled'])
        self.assertEqual(p['maximum_fullpcm_processes'], 28)
        self.assertEqual(p['new_compilations'], 0)

    def test_only_actual_synthetic_retained_result_is_eligible(self):
        p, doc = fixture(); m.candidate_gate(doc, p)
        for values in ({'passed': False}, {'status': 'REJECTED_RESEARCH_REGRESSION_GATE'},
                       {'status': 'RETAINED_RESEARCH_INCREMENTAL_HEAD'}, {'schema': 'other'},
                       {'head_sha': 'b'*40}, {'ssc305_qualification': True}, {'strong_english_fsmn_ctc': True}):
            with self.assertRaises(ValueError): m.candidate_gate(dict(doc, **values), p)

    def test_all_candidate_gates_and_exact_weights_required(self):
        p, doc = fixture()
        for key in doc['gates']:
            bad = copy.deepcopy(doc); bad['gates'][key] = False
            with self.assertRaises(ValueError): m.candidate_gate(bad, p)
        bad = copy.deepcopy(doc); bad['candidate']['bin']['sha256'] = '0'*64
        with self.assertRaises(ValueError): m.candidate_gate(bad, p)

    def test_source_plan_rejects_old_candidate_and_extra_compute(self):
        p, _ = fixture(); m.source_preflight(p, h)
        for change in ({'new_compilations': 1}, {'maximum_fullpcm_processes': 29}, {'enabled': False}):
            with self.assertRaises(ValueError): m.source_preflight(dict(p, **change), h)
        bad = copy.deepcopy(p); bad['candidate_source']['name'] = 'english-hardneg-increment-123'
        with self.assertRaises(ValueError): m.source_preflight(bad, h)

    def test_qemu_present_does_not_install(self):
        r = mock.Mock()
        with mock.patch.object(m.shutil, 'which', return_value='/usr/bin/qemu-arm'):
            m.prepare_qemu(r)
        self.assertEqual(r.command.call_count, 1)
        self.assertEqual(r.command.call_args.args[0], 'qemu-version')

    def test_qemu_missing_has_one_bounded_install_no_compile(self):
        r = mock.Mock()
        with mock.patch.object(m.shutil, 'which', side_effect=[None, '/usr/bin/qemu-arm']):
            m.prepare_qemu(r)
        self.assertEqual([c.args[0] for c in r.command.call_args_list], ['qemu-apt-update','qemu-apt-install','qemu-version'])
        for call in r.command.call_args_list[:2]:
            self.assertIn('Acquire::Retries=0', call.args[1]); self.assertEqual(call.kwargs['timeout'], 180)
            self.assertNotIn('gcc', call.args[1]); self.assertNotIn('ffmpeg', call.args[1])

    def test_install_failure_does_not_retry_or_run_model(self):
        r = mock.Mock(); r.command.side_effect = ValueError('install failed')
        with mock.patch.object(m.shutil, 'which', return_value=None):
            with self.assertRaises(ValueError): m.prepare_qemu(r)
        self.assertEqual(r.command.call_count, 1)

    def test_outside_actions_fails_before_any_command(self):
        p, _ = fixture()
        with mock.patch.dict(m.os.environ, {}, clear=True), mock.patch.object(m.subprocess, 'check_output', side_effect=AssertionError('NO_PROCESS')):
            with self.assertRaisesRegex(ValueError, 'NEW_PUBLIC_MAIN_FIRST_PUSH_ONLY'): m.execution_identity(p, h)

    def test_reference_shape_must_be_complete_before_use(self):
        with self.assertRaisesRegex(ValueError, 'COMPLETE_CANDIDATE_REFERENCE_GEOMETRY'):
            m.select_references([], [], b'', h)


    def test_locked_case_choice_ignores_scores_and_input_order(self):
        rows = []
        for label in (0, 1, 2):
            for condition in m.CONDITIONS:
                common = dict(role='locked-test', weak_class=label, condition=condition,
                              usable=True, candidate_reference_available=True)
                rows.extend([dict(common, id=f'z-{label}-{condition}', events=[[label, 90000]]),
                             dict(common, id=f'a-{label}-{condition}', events=[])])
        first = m.selected_fresh(roster_doc(rows))
        second = m.selected_fresh(roster_doc(list(reversed(rows))))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertTrue(all(x['id'].startswith('a-') for x in first))
        for row in rows:
            if row['id'].startswith('a-'):
                row['usable'] = False
        self.assertTrue(all(x['id'].startswith('z-') for x in m.selected_fresh(roster_doc(rows))))

    def test_missing_or_unopened_locked_cases_fail(self):
        rows = [dict(id=f'{label}-{condition}', role='locked-test', weak_class=label,
                     condition=condition, usable=True, candidate_reference_available=True)
                for label in (0, 1, 2) for condition in m.CONDITIONS]
        for key, value in (('role', 'development'), ('usable', False), ('candidate_reference_available', False)):
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.assertRaisesRegex(ValueError, 'LOCKED_CASE_FOR_EVERY_CLASS_AND_CONDITION'):
                m.selected_fresh(roster_doc(bad))
        with self.assertRaises(ValueError): m.selected_fresh(roster_doc(rows + [rows[0]]))

    def test_bounded_runner_rejects_29th_model_before_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            r = m.ReplayRunner(Path(directory), m.time.monotonic() + 20, h)
            r.passes = 28
            with mock.patch.object(m.subprocess, 'Popen', side_effect=AssertionError('NO_PROCESS')):
                with self.assertRaisesRegex(ValueError, 'TWENTYEIGHT_FULLPCM_PROCESSES_MAXIMUM'):
                    r.command('extra', ['never-execute'], model=True)



if __name__ == '__main__':
    unittest.main()
