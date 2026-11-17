"""Stdlib/synthetic/mock contracts only; never compile, decode, or run a model."""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import signal
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKER = HERE / 'english_hardneg_natural32_once.py'
if not WORKER.exists():
    WORKER = HERE / 'natural32_once.py'
spec = importlib.util.spec_from_file_location('natural32_worker_under_test', WORKER)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def helper_source(relative):
    staged = HERE / Path(relative).name
    if relative.startswith(w.RESEARCH + '/') and staged.exists():
        return staged
    roots = [HERE.parent, HERE.parent / 'kws-data']
    for root in roots:
        path = root / relative
        if path.exists():
            return path
    raise AssertionError('missing pinned source fixture: ' + relative)


def assemble_source_tree(root):
    for path in w.SOURCE_PINS:
        w.put(root / path, helper_source(path).read_bytes())
    w.put(root / w.WORKER_PATH, WORKER.read_bytes())
    plan = w.expected_plan(w.sha(WORKER.read_bytes()))
    w.once(root / w.PLAN_PATH, plan)
    return plan


def score_rows(probabilities, emitted=None):
    emitted = {} if emitted is None else emitted
    return [{'sample': 35200 + i * 1280, 'ready': True, 'keyword': emitted.get(i, 0),
             'logits': [0.0, 0.0, 0.0], 'probabilities': list(p)} for i, p in enumerate(probabilities)]


def flac_header(frames=40000, rate=16000, channels=1, bits=16):
    block = bytearray(34)
    value = (rate << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | frames
    block[10:18] = value.to_bytes(8, 'big')
    return b'fLaC' + bytes([128, 0, 0, 34]) + bytes(block) + b'not-decoded-fixture'


def tar_fixture(path, members):
    with tarfile.open(path, 'w:gz') as archive:
        for name, raw, kind in members:
            item = tarfile.TarInfo(name)
            item.size = len(raw) if kind == tarfile.REGTYPE else 0
            item.type = kind
            if kind == tarfile.SYMTYPE:
                item.linkname = '/tmp/outside'
            archive.addfile(item, io.BytesIO(raw) if item.size else None)
    return w.file_identity(path)


class ContractTests(unittest.TestCase):
    def test_default_and_preflight_have_no_subprocess_or_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = assemble_source_tree(root)
            with mock.patch.object(w, 'ROOT', root), mock.patch.object(w.subprocess, 'Popen', side_effect=AssertionError('no process')), \
                 mock.patch.object(w.subprocess, 'check_output', side_effect=AssertionError('no process')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.main([]), 0)
                self.assertEqual(w.main(['--preflight']), 0)
                self.assertEqual(len(w.source_preflight(root, plan)['rows']), 32)
                with self.assertRaisesRegex(ValueError, 'SOURCE_ONLY_HAS_NO_OUTPUT'):
                    w.main(['--output', str(root / 'forbidden')])
            self.assertFalse((root / 'forbidden').exists())

    def test_source_plan_and_helper_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = assemble_source_tree(root)
            changed = copy.deepcopy(plan)
            changed['maximum_model_processes'] = 33
            with self.assertRaisesRegex(ValueError, 'EXACT_FROZEN_PLAN'):
                w.source_preflight(root, changed)
            helper = root / w.RESEARCH / 'mine_saved_scores.py'
            helper.write_bytes(helper.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'SOURCE_PIN'):
                w.source_preflight(root, plan)

    def test_source_has_no_unbounded_or_model_framework_execution(self):
        raw = WORKER.read_text()
        for bad in ('import numpy', 'import torch', 'import onnxruntime', 'extractall(', 'shell=True', 'apt-get', 'workflow_dispatch'):
            if bad == 'workflow_dispatch':
                continue  # It appears only in the descriptive trigger contract.
            self.assertNotIn(bad, raw)
        self.assertEqual(w.expected_plan('a' * 64)['maximum_model_processes'], 32)
        self.assertEqual(w.expected_plan('a' * 64)['maximum_decoder_processes'], 32)

    def valid_identity(self):
        env = {'GITHUB_ACTIONS': 'true', 'GITHUB_REPOSITORY': w.REPO, 'GITHUB_EVENT_NAME': 'push',
               'GITHUB_REF': 'refs/heads/main', 'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': 'a' * 40, 'GITHUB_RUN_ID': '123'}
        event = {'before': w.BEFORE, 'after': env['GITHUB_SHA'], 'ref': 'refs/heads/main',
                 'repository': {'private': False, 'full_name': w.REPO}, 'deleted': False, 'forced': False}
        return env, event

    def test_execution_only_public_exact_successor_push_attempt_one(self):
        env, event = self.valid_identity()
        plan = w.expected_plan('b' * 64)
        result = w.execution_identity(plan, env, event, head=env['GITHUB_SHA'], parent=w.BEFORE)
        self.assertEqual(result['run_id'], '123')
        variants = [('GITHUB_RUN_ATTEMPT', '2'), ('GITHUB_EVENT_NAME', 'workflow_dispatch'),
                    ('GITHUB_REF', 'refs/heads/topic'), ('GITHUB_ACTIONS', 'false'), ('GITHUB_REPOSITORY', 'x/y')]
        for key, value in variants:
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.execution_identity(plan, dict(env, **{key: value}), event, head=env['GITHUB_SHA'], parent=w.BEFORE)
        for field, value in [('before', 'b' * 40), ('after', 'b' * 40), ('forced', True), ('deleted', True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                w.execution_identity(plan, env, dict(event, **{field: value}), head=env['GITHUB_SHA'], parent=w.BEFORE)
        private = copy.deepcopy(event)
        private['repository']['private'] = True
        with self.assertRaises(ValueError):
            w.execution_identity(plan, env, private, head=env['GITHUB_SHA'], parent=w.BEFORE)
        for head, parent in [('b' * 40, w.BEFORE), (env['GITHUB_SHA'], 'b' * 40)]:
            with self.assertRaisesRegex(ValueError, 'WORKTREE_EXACT_SUCCESSOR'):
                w.execution_identity(plan, env, event, head=head, parent=parent)

    def test_float32_threshold_and_three_confirmation_eof(self):
        threshold = w.CONFIG['threshold_float32']
        self.assertEqual(threshold, 0.800000011920929)
        below = struct.unpack('<f', struct.pack('<I', struct.unpack('<I', struct.pack('<f', threshold))[0] - 1))[0]
        for p in (below, 0.8):
            self.assertEqual(w.replay_events(score_rows([[1 - p, p, 0]] * 3), 37760), [])
        rows = score_rows([[1 - threshold, threshold, 0]] * 3, {2: 1})
        self.assertEqual(w.replay_events(rows, 37760), [[1, 37760]])
        self.assertEqual(w.replay_events(rows[:2], 36480), [])
        with self.assertRaisesRegex(ValueError, 'EMITTED_KEYWORD'):
            w.replay_events(score_rows([[0.1, 0.9, 0]] * 3), 37760)

    def test_event_switching_and_cooldown_boundary(self):
        rows = score_rows([[0.1, 0.9, 0]] * 2 + [[0.1, 0, 0.9]] * 3, {4: 2})
        self.assertEqual(w.replay_events(rows, 40320), [[2, 40320]])
        rows = score_rows([[0.1, 0.9, 0]] * 18, {2: 1, 17: 1})
        self.assertEqual(w.replay_events(rows, 56960), [[1, 37760], [1, 56960]])
        bad = copy.deepcopy(rows)
        bad[15]['keyword'] = 1
        with self.assertRaisesRegex(ValueError, 'EMITTED_KEYWORD'):
            w.replay_events(bad, 56960)

    def test_event_clock_and_nonfinite_or_invalid_probability_rejected(self):
        rows = score_rows([[1, 0, 0]] * 3)
        for mutate in ('clock', 'nonfinite', 'sum', 'extra', 'missing'):
            bad = copy.deepcopy(rows)
            if mutate == 'clock':
                bad[1]['sample'] += 1
            elif mutate == 'nonfinite':
                bad[1]['probabilities'][1] = float('nan')
            elif mutate == 'sum':
                bad[1]['probabilities'] = [1, 1, 1]
            elif mutate == 'extra':
                bad.append(copy.deepcopy(rows[-1]))
            else:
                bad.pop()
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                w.replay_events(bad, 37760)

    def test_flac_streaminfo_geometry_is_checked_without_decoder(self):
        self.assertEqual(w.flac_streaminfo(flac_header())['frames'], 40000)
        for options in ({'rate': 48000}, {'channels': 2}, {'bits': 24}, {'frames': 0}, {'frames': 240001}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                w.flac_streaminfo(flac_header(**options))
        for data in (b'bad', b'RIFF' + flac_header()[4:], b'fLaC\x81\x00\x00\x22' + bytes(34)):
            with self.assertRaisesRegex(ValueError, 'FLAC_FIRST_STREAMINFO'):
                w.flac_streaminfo(data)

    def test_selected_tar_and_transcript_are_exact_without_extractall(self):
        raw = flac_header()
        name = 'LibriSpeech/dev-clean/12/34/12-34-0001.flac'
        text = b'EXACT ORIGINAL TRANSCRIPT'
        row = {'source_member': name, 'recording_id': 'case1', 'frames': 40000,
               'flac_sha256': w.sha(raw), 'transcript_sha256': w.sha(text)}
        members = [(name, raw, tarfile.REGTYPE),
                   ('LibriSpeech/dev-clean/12/34/12-34.trans.txt', b'12-34-0001 ' + text + b'\n', tarfile.REGTYPE),
                   ('LibriSpeech/dev-clean/ignored.txt', b'not extracted', tarfile.REGTYPE)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / 'a.tar.gz'
            pin = tar_fixture(archive, members)
            with mock.patch.object(tarfile.TarFile, 'extractall', side_effect=AssertionError('no extraction')):
                selected, report = w.select_corpus(archive, {'rows': [row]}, root / 'out', pin)
            self.assertEqual(list(selected), [name])
            self.assertEqual(report['archive_paths_extracted'], 0)
            self.assertEqual((root / 'out/selectedinputs/transcripts/case1.txt').read_bytes(), text)
            self.assertFalse((root / 'out/LibriSpeech').exists())
            changed = dict(row, transcript_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'SOURCE_TRANSCRIPT_SHA256'):
                w.select_corpus(archive, {'rows': [changed]}, root / 'bad', pin)

    def test_tar_path_links_duplicates_and_bad_pins_fail(self):
        row = {'source_member': 'LibriSpeech/dev-clean/12/34/12-34-0001.flac', 'recording_id': 'case1',
               'frames': 40000, 'flac_sha256': w.sha(flac_header()), 'transcript_sha256': w.sha(b'text')}
        for members in [[('../outside', b'x', tarfile.REGTYPE)],
                        [('link', b'', tarfile.SYMTYPE)],
                        [('a', b'x', tarfile.REGTYPE), ('a', b'x', tarfile.REGTYPE)]]:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                pin = tar_fixture(root / 'a.tar.gz', members)
                with self.assertRaises(ValueError):
                    w.select_corpus(root / 'a.tar.gz', {'rows': [row]}, root / 'out', pin)
                self.assertFalse((root / 'outside').exists())
                with self.assertRaisesRegex(ValueError, 'CORPUS_EXACT_TAR'):
                    w.select_corpus(root / 'a.tar.gz', {'rows': [row]}, root / 'out', dict(pin, sha256='0' * 64))

    def test_decode_command_forbids_audio_transformation(self):
        args = w.decode_argv('/tmp/existing.flac')
        for option in ('-ar', '-ac', '-af', '-filter:a', '-filter_complex', '-t', '-ss', '-stream_loop', '-y'):
            self.assertNotIn(option, args)
        self.assertIn('pcm_s16le', args)
        self.assertEqual(args[-3:], ['-f', 's16le', 'pipe:1'])

    def test_runner_nonzero_retains_receipts_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'evidence').mkdir()
            proc = mock.Mock(returncode=7)
            proc.poll.return_value = 7
            with mock.patch.object(w.subprocess, 'Popen', return_value=proc) as popen:
                runner = w.Runner(root, w.time.monotonic() + 100)
                with self.assertRaisesRegex(ValueError, 'COMMAND_NONZERO_EXIT'):
                    runner.command('only', ['fake-native'], kind='model')
            self.assertEqual(popen.call_count, 1)
            self.assertFalse(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(runner.counts['model'], 1)
            receipt = w.loads((root / 'evidence/only.exit.json').read_bytes())
            self.assertEqual(receipt['returncode'], 7)
            self.assertEqual(receipt['retry_count'], 0)

    def test_runner_deadline_kills_process_and_preserves_actual_exit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'evidence').mkdir()
            proc = mock.Mock(returncode=None)
            proc.poll.side_effect = lambda: proc.returncode
            proc.kill.side_effect = lambda: setattr(proc, 'returncode', -9)
            with mock.patch.object(w.subprocess, 'Popen', return_value=proc) as popen, \
                 mock.patch.object(w.time, 'monotonic', side_effect=[0, 0, 2, 2]):
                runner = w.Runner(root, 100)
                with self.assertRaisesRegex(ValueError, 'COMMAND_DEADLINE'):
                    runner.command('timeout', ['fake-decoder'], kind='decode', timeout=1)
            self.assertEqual(popen.call_count, 1)
            proc.kill.assert_called_once()
            proc.wait.assert_called_once()
            receipt = w.loads((root / 'evidence/timeout.exit.json').read_bytes())
            self.assertTrue(receipt['timed_out'])
            self.assertEqual(receipt['returncode'], -9)

    def test_runner_enforces_single_build_and_32_passes_before_popen(self):
        with tempfile.TemporaryDirectory() as d:
            runner = w.Runner(Path(d), w.time.monotonic() + 100)
            runner.counts = {'build': 1, 'decode': 32, 'model': 32}
            with mock.patch.object(w.subprocess, 'Popen', side_effect=AssertionError('budget must stop process')):
                for kind in runner.counts:
                    with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'COMMAND_BUDGET'):
                        runner.command('excess-' + kind, ['forbidden'], kind=kind)

    def test_group_cleanup_handles_exit_race_and_sigkill(self):
        with mock.patch.object(w, 'group_exists', return_value=True), \
             mock.patch.object(w.os, 'killpg', side_effect=ProcessLookupError):
            self.assertEqual(w.cleanup_group(123), {'descendants_found': False, 'terminated': False, 'killed': False})
        with mock.patch.object(w, 'group_exists', side_effect=[True, True, True]), \
             mock.patch.object(w.os, 'killpg') as killpg, mock.patch.object(w.time, 'monotonic', side_effect=[0, 3]):
            result = w.cleanup_group(123)
            self.assertTrue(result['killed'])
            self.assertEqual(killpg.call_args_list, [mock.call(123, signal.SIGTERM), mock.call(123, signal.SIGKILL)])

    def test_supervisor_launch_failure_keeps_started_result_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'output'
            binding = {'run_id': '1', 'head_sha': 'a' * 40}
            def failed_launch(*_args, **_kwargs):
                self.assertTrue((output / 'STARTED.json').is_file())
                raise OSError('fixture launch failure')
            with mock.patch.object(w, 'execution_identity', return_value=binding), \
                 mock.patch.object(w.subprocess, 'Popen', side_effect=failed_launch) as popen, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.supervise(argparse.Namespace(output=output), w.expected_plan('b' * 64)), 1)
            self.assertEqual(popen.call_count, 1)
            result = w.loads((output / 'evidence/RESULT.json').read_bytes())
            self.assertFalse(result['passed'])
            self.assertEqual(result['model_process_attempts'], 0)
            self.assertTrue((output / 'MANIFEST.json').is_file())

    def test_supervisor_zero_exit_without_result_is_failure_and_partial_steps_close(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'output'
            binding = {'run_id': '1', 'head_sha': 'a' * 40}
            def launch(*_args, **kwargs):
                self.assertTrue(kwargs['start_new_session'])
                w.once(output / 'evidence/scan-0.started.json', {'kind': 'model', 'argv': ['fake']})
                w.put(output / 'downloads/partial.zip', b'do not upload')
                proc = mock.Mock(pid=123, returncode=0)
                proc.poll.return_value = 0
                return proc
            with mock.patch.object(w, 'execution_identity', return_value=binding), \
                 mock.patch.object(w.subprocess, 'Popen', side_effect=launch), \
                 mock.patch.object(w, 'cleanup_group', return_value={'descendants_found': False}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.supervise(argparse.Namespace(output=output), w.expected_plan('b' * 64)), 1)
            closed = w.loads((output / 'evidence/scan-0.exit.json').read_bytes())
            self.assertIsNone(closed['returncode'])
            self.assertFalse(closed['completed'])
            manifest = w.loads((output / 'MANIFEST.json').read_bytes())
            self.assertNotIn('downloads/partial.zip', manifest['files'])
            self.assertIn('evidence/scan-0.exit.json', manifest['files'])

    def test_truncated_started_and_result_do_not_prevent_failure_sealing(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'output'
            binding = {'run_id': '1', 'head_sha': 'a' * 40}
            def launch(*_args, **_kwargs):
                w.put(output / 'evidence/scan-0.started.json', b'{"kind":')
                w.put(output / 'evidence/scan-0.exit.json.pending-child', b'{"interrupted":')
                w.put(output / 'evidence/RESULT.json', b'{')
                proc = mock.Mock(pid=123, returncode=-9)
                proc.poll.return_value = -9
                return proc
            with mock.patch.object(w, 'execution_identity', return_value=binding), \
                 mock.patch.object(w.subprocess, 'Popen', side_effect=launch), \
                 mock.patch.object(w, 'cleanup_group', return_value={'descendants_found': False}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(w.supervise(argparse.Namespace(output=output), w.expected_plan('b' * 64)), 1)
            self.assertEqual((output / 'evidence/RESULT.json').read_bytes(), b'{')
            receipt = w.loads((output / 'evidence/SUPERVISOR_EXIT.json').read_bytes())
            self.assertFalse(receipt['passed'])
            self.assertEqual(len(receipt['partial_receipt_errors']), 1)
            fallback = w.loads((output / 'evidence/RESULT_SUPERVISOR_FAILURE.json').read_bytes())
            self.assertFalse(fallback['passed'])
            self.assertIsNone(fallback['model_process_attempts'])
            self.assertTrue((output / 'MANIFEST.json').is_file())

    def test_json_receipts_are_complete_and_create_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'receipt.json'
            w.once(path, {'value': 1})
            with self.assertRaises(FileExistsError):
                w.once(path, {'value': 2})
            self.assertEqual(w.loads(path.read_bytes()), {'value': 1})
            self.assertEqual(list(Path(d).glob('receipt.json.pending-*')), [])

    def test_false_research_boundary_cannot_claim_improvement_or_far(self):
        result = w.result_boundary({})
        self.assertTrue(all(result[k] is False for k in w.FALSE_FLAGS))
        self.assertFalse(result['quality_improvement_claim'])
        self.assertFalse(result['kws_model_delivered'])
        self.assertEqual(result['verified_negative_count'], 0)
        self.assertEqual(result['new_independent_speaker_count'], 0)
        self.assertEqual(result['independent_far'], 'NOT_MEASURED')
        self.assertEqual(result['independent_frr'], 'NOT_MEASURED')


if __name__ == '__main__':
    unittest.main()
