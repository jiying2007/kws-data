"""Source, invented-trace and process-control tests; no audio/model/C execution."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

SOURCE = Path(__file__).with_name('english_coverage_fullpcm_once.py')
spec = importlib.util.spec_from_file_location('coverage_fullpcm_contract', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def invented_case(event=True):
    """Independent format fixture: 2.2 seconds, one ready output, optional event."""
    return {'id': 'invented', 'kind': 'synthetic_format_only', 'samples': 35200,
            'records': 16, 'ready': 1, 'events': [[1, 35200]] if event else [], 'pcm_sha256': '0' * 64}


def invented_trace(event=True):
    case = invented_case(event)
    raw = bytearray(b'EKWSTRC1' + struct.pack('<IIIIIIQ', 1, 1, 16000, 320, 96, 3, 35200))
    refs, embeddings, offsets = [], [], []
    for i in range(16):
        sample = 16000 + i * 1280
        vector = [float(i + k) / 128 for k in range(96)]
        payload = struct.pack('<QQ96f', i, sample, *vector)
        offsets.append(len(raw))
        raw += struct.pack('<BI', 1, 400) + payload
        logits = [0.0, 1.0, -1.0] if i == 15 else [0.0] * 3
        probabilities = [0.125, 0.75, 0.125] if i == 15 else [0.0] * 3
        keyword = int(i == 15 and event)
        payload = struct.pack('<QIIQ6f', i, int(i == 15), keyword, sample, *logits, *probabilities)
        raw += struct.pack('<BI', 2, 48) + payload
        if keyword:
            raw += struct.pack('<BIQQIQf', 3, 32, 0, i, keyword, sample, probabilities[1])
        refs.append({'sample': sample, 'ready': i == 15, 'keyword': keyword,
                     'logits': logits, 'probabilities': probabilities})
        embeddings.append(vector)
    raw += struct.pack('<BI6Q', 255, 48, 35200, 16, 16, 1, int(event), 110)
    metadata = {'schema': 'english-stream-numeric-trace-v1', 'trace_completed': True,
        'format_magic': 'EKWSTRC1', 'format_version': 1, 'abi': 1, 'sample_rate_hz': 16000,
        'encoding': 's16le_mono', 'chunk_samples': 320, 'input_samples': 35200,
        'input_bytes': 70400, 'input_repetitions': 1, 'resets': 0, 'eof_padding_samples': 0,
        'feed_calls': 110, 'embedding_records': 16, 'real_head_calls': 16, 'score_records': 16,
        'ready_outputs': 1, 'event_records': int(event), 'trace_records': 34 if event else 33,
        'trace_bytes': len(raw), 'end_record_written': True, 'score_observer_enabled': True,
        'original_head_link_wrap': True, 'performance_measured': False}
    return bytes(raw), {'case': case, 'reference': refs, 'embeddings': embeddings}, metadata, offsets


class SourceAndAdmission(unittest.TestCase):
    def test_default_does_not_start_process_or_read_model(self):
        with (mock.patch.object(m.subprocess, 'Popen', side_effect=AssertionError('no subprocess')),
             mock.patch.object(m, 'prepare_inputs', side_effect=AssertionError('no artifact')),
             mock.patch('sys.stdout', new_callable=io.StringIO) as out):
            self.assertEqual(m.main([]), 0)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc['status'], 'SOURCE_CONTRACT_ONLY')
        self.assertEqual(doc['model_calls'], 0)
        self.assertFalse(doc['execution_enabled'])

    def test_plan_cannot_change_case_tolerance_or_budget(self):
        p = m.expected_plan(m.sha(m.regular(SOURCE)))
        self.assertIs(m.validate_plan(p), p)
        for key, value in [('maximum_execution_seconds', 1201), ('retry_count', 1),
                           ('threshold', 0.79), ('cpu_threads', 2), ('main_before', '0' * 40)]:
            q = copy.deepcopy(p)
            q[key] = value
            with self.assertRaises(ValueError):
                m.validate_plan(q)
        q = copy.deepcopy(p)
        q['cases'][-1]['pcm_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            m.validate_plan(q)
        q = copy.deepcopy(p)
        q['limits']['probabilities']['atol'] = 0.001
        with self.assertRaises(ValueError):
            m.validate_plan(q)

    def test_exact_successor_and_public_push_required(self):
        plan = m.expected_plan('0' * 64)
        env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': m.REPO,
               'GITHUB_EVENT_NAME': 'push', 'GITHUB_REF': 'refs/heads/main',
               'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '123'}
        event = {'before': m.BEFORE, 'after': 'a' * 40, 'ref': 'refs/heads/main',
                 'repository': {'private': False, 'full_name': m.REPO}, 'deleted': False}
        self.assertTrue(m.execution_identity(plan, env, event, 'a' * 40)['public_repository'])
        for key, value in [('GITHUB_ACTIONS', 'false'), ('GITHUB_EVENT_NAME', 'workflow_dispatch'),
                           ('GITHUB_REF', 'refs/heads/main-other'), ('GITHUB_RUN_ATTEMPT', '2'),
                           ('GITHUB_REPOSITORY', 'elsewhere/private')]:
            changed = dict(env, **{key: value})
            with self.assertRaises(ValueError):
                m.execution_identity(plan, changed, event, 'a' * 40)
        for key, value in [('before', 'b' * 40), ('after', 'b' * 40), ('deleted', True)]:
            changed = dict(event, **{key: value})
            with self.assertRaises(ValueError):
                m.execution_identity(plan, env, changed, 'a' * 40)
        changed = copy.deepcopy(event)
        changed['repository']['private'] = True
        with self.assertRaises(ValueError):
            m.execution_identity(plan, env, changed, 'a' * 40)
        with self.assertRaises(ValueError):
            m.execution_identity(plan, env, event, 'b' * 40)

    def test_execute_is_denied_before_any_process_outside_actions(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(m.subprocess, 'Popen') as start:
            with self.assertRaises(ValueError):
                m.execution_identity(m.expected_plan('0' * 64))
            start.assert_not_called()

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                m.loads(raw)

    def test_archive_paths_and_identity_are_checked(self):
        for name in ('../x', '/x', 'a//b', 'a/./b', 'a/../b', 'a\\b', 'a\x00b', ''):
            self.assertFalse(m.safe_member(name))
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'x.zip'
            with zipfile.ZipFile(p, 'w') as z:
                z.writestr('../outside', b'x')
            pin = m.identity(p.read_bytes())
            with self.assertRaises(ValueError):
                m.Archive(p, pin)
            with self.assertRaises(ValueError):
                m.Archive(p, {'bytes': pin['bytes'], 'sha256': '0' * 64})

    def test_model_subprocesses_receive_no_credentials_and_one_thread(self):
        with mock.patch.dict(os.environ, {'GH_TOKEN': 'fixture', 'GITHUB_TOKEN': 'fixture',
                             'ACTIONS_RUNTIME_TOKEN': 'fixture', 'OTHER_TOKEN': 'fixture'}, clear=True):
            env = m.command_env(False)
            self.assertFalse(any('TOKEN' in key for key in env))
            self.assertEqual(env['OMP_NUM_THREADS'], '1')
            self.assertEqual(env['OPENBLAS_NUM_THREADS'], '1')
            self.assertIn('GH_TOKEN', m.command_env(True))


class CompleteTrace(unittest.TestCase):
    def test_eof_event_and_every_numeric_element_are_compared(self):
        raw, item, meta, _ = invented_trace()
        parsed = m.parse_trace(raw, item['case'])
        report = m.compare_trace(item, parsed, meta)
        self.assertTrue(report['passed'])
        self.assertEqual(report['actual_events'], [[1, 35200]])
        self.assertEqual(report['numeric']['embedding']['elements'], 16 * 96)
        self.assertEqual(report['numeric']['logits']['elements'], 16 * 3)
        self.assertEqual(report['numeric']['probabilities']['elements'], 16 * 3)

    def test_no_event_still_requires_full_scores_and_embeddings(self):
        raw, item, meta, _ = invented_trace(False)
        parsed = m.parse_trace(raw, item['case'])
        self.assertTrue(m.compare_trace(item, parsed, meta)['passed'])
        changed = copy.deepcopy(parsed)
        changed['embeddings'][8][63] += 1.0
        result = m.compare_trace(item, changed, meta)
        self.assertFalse(result['passed'])
        self.assertEqual(result['numeric']['embedding']['violations'], 1)

    def test_missing_end_trailing_bytes_and_wrong_end_counter_rejected(self):
        raw, item, _, _ = invented_trace()
        malformed = [raw[:-1], raw[:-53], raw + b'\0']
        altered = bytearray(raw)
        struct.pack_into('<Q', altered, len(altered) - 48, 35201)
        malformed.append(bytes(altered))
        for bad in malformed:
            with self.assertRaises(ValueError):
                m.parse_trace(bad, item['case'])

    def test_nonfinite_embedding_and_score_rejected(self):
        raw, item, _, positions = invented_trace()
        for pos in (positions[0] + 5 + 16, positions[15] + 405 + 5 + 24):
            bad = bytearray(raw)
            struct.pack_into('<f', bad, pos, float('nan'))
            with self.assertRaises(ValueError):
                m.parse_trace(bytes(bad), item['case'])

    def test_reordered_record_wrong_clock_and_early_ready_rejected(self):
        raw, item, _, positions = invented_trace()
        bad = bytearray(raw)
        bad[positions[0]] = 2
        with self.assertRaises(ValueError):
            m.parse_trace(bytes(bad), item['case'])
        bad = bytearray(raw)
        struct.pack_into('<Q', bad, positions[3] + 5 + 8, 19841)
        with self.assertRaises(ValueError):
            m.parse_trace(bytes(bad), item['case'])
        bad = bytearray(raw)
        struct.pack_into('<I', bad, positions[0] + 405 + 5 + 8, 1)
        with self.assertRaises(ValueError):
            m.parse_trace(bytes(bad), item['case'])

    def test_event_confidence_and_keyword_are_bound_to_score(self):
        raw, item, _, _ = invented_trace()
        event_start = len(raw) - 53 - 37
        for pos, fmt, value in [(event_start + 5 + 28, '<f', 0.5),
                                (event_start + 5 + 16, '<I', 2)]:
            bad = bytearray(raw)
            struct.pack_into(fmt, bad, pos, value)
            with self.assertRaises(ValueError):
                m.parse_trace(bytes(bad), item['case'])

    def test_exact_clock_cannot_be_replaced_by_tolerant_numeric_match(self):
        raw, item, meta, _ = invented_trace()
        parsed = m.parse_trace(raw, item['case'])
        changed = copy.deepcopy(item)
        changed['reference'][-1]['sample'] += 1
        report = m.compare_trace(changed, parsed, meta)
        self.assertFalse(report['passed'])
        self.assertFalse(report['exact_clock_ready_keyword'])

    def test_metadata_no_padding_one_pass_and_bytes_required(self):
        raw, item, meta, _ = invented_trace()
        parsed = m.parse_trace(raw, item['case'])
        for key, value in [('eof_padding_samples', 1), ('input_repetitions', 2),
                           ('real_head_calls', 17), ('trace_bytes', len(raw) - 1),
                           ('end_record_written', False), ('performance_measured', True)]:
            with self.assertRaises(ValueError):
                m.compare_trace(item, parsed, dict(meta, **{key: value}))

    def test_fixed_tolerance_near_zero_is_not_relative_only(self):
        inside = m.compare_numbers([[0.0]], [[0.0005]], m.LIMITS['probabilities'])
        outside = m.compare_numbers([[0.0]], [[0.00050001]], m.LIMITS['probabilities'])
        self.assertTrue(inside['passed'])
        self.assertFalse(outside['passed'])
        with self.assertRaises(ValueError):
            m.compare_numbers([[0.0]], [[float('inf')]], m.LIMITS['probabilities'])


class BudgetAndCleanup(unittest.TestCase):
    def test_budget_rejection_occurs_before_launch(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(m.subprocess, 'Popen') as start:
            r = m.Runner(Path(t), float('inf'))
            r.passes = 10
            with self.assertRaises(ValueError):
                r.command('eleventh', ['never'], model=True)
            r.builds = 2
            with self.assertRaises(ValueError):
                r.command('third', ['never'], build=True)
            start.assert_not_called()

    def test_no_process_after_global_deadline(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(m.subprocess, 'Popen') as start:
            with self.assertRaises(ValueError):
                m.Runner(Path(t), -1).command('expired', ['never'])
            start.assert_not_called()

    def test_command_started_receipt_precedes_process_launch(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            process = mock.Mock(returncode=0)
            process.poll.return_value = 0
            def fake_start(*args, **kwargs):
                self.assertTrue((p / 'evidence/fixture.started.json').is_file())
                self.assertFalse(kwargs['start_new_session'])
                return process
            with mock.patch.object(m.subprocess, 'Popen', side_effect=fake_start):
                m.Runner(p, m.time.monotonic() + 60).command('fixture', ['invented-only'])
            receipt = json.loads((p / 'evidence/fixture.exit.json').read_text())
            self.assertEqual(receipt['returncode'], 0)
            self.assertIsNone(receipt['error'])

    def test_nonzero_command_is_recorded_without_retry(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            process = mock.Mock(returncode=7)
            process.poll.return_value = 7
            with mock.patch.object(m.subprocess, 'Popen', return_value=process) as start:
                with self.assertRaises(ValueError):
                    m.Runner(p, m.time.monotonic() + 60).command('failure', ['invented-only'])
                self.assertEqual(start.call_count, 1)
            receipt = json.loads((p / 'evidence/failure.exit.json').read_text())
            self.assertEqual(receipt['returncode'], 7)
            self.assertEqual(receipt['retry_count'], 0)

    def test_cleanup_finds_descendants_even_after_leader_exited(self):
        with (mock.patch.object(m, 'group_exists', side_effect=[True, False, False]),
             mock.patch.object(m.os, 'killpg') as kill):
            result = m.cleanup_group(12345)
        self.assertTrue(result['descendants_found'])
        self.assertTrue(result['terminated'])
        self.assertFalse(result['killed'])
        kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_cleanup_kills_stubborn_descendants(self):
        with (mock.patch.object(m, 'group_exists', return_value=True),
             mock.patch.object(m.time, 'monotonic', side_effect=[0.0, 3.0]),
             mock.patch.object(m.os, 'killpg') as kill):
            result = m.cleanup_group(12345)
        self.assertTrue(result['killed'])
        self.assertEqual(kill.call_args_list, [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)])

    def test_cleanup_handles_group_exit_race(self):
        with (mock.patch.object(m, 'group_exists', return_value=True),
             mock.patch.object(m.os, 'killpg', side_effect=ProcessLookupError)):
            result = m.cleanup_group(12345)
        self.assertFalse(result['descendants_found'])

    def test_arm_consumer_must_be_static(self):
        raw = bytearray(84)
        raw[:6] = b'\x7fELF\x01\x01'
        struct.pack_into('<H', raw, 18, 40)
        struct.pack_into('<I', raw, 28, 52)
        struct.pack_into('<HH', raw, 42, 32, 1)
        struct.pack_into('<I', raw, 52, 1)
        m.validate_elf(bytes(raw), 'arm')
        struct.pack_into('<I', raw, 52, 3)
        with self.assertRaises(ValueError):
            m.validate_elf(bytes(raw), 'arm')


if __name__ == '__main__':
    unittest.main()
