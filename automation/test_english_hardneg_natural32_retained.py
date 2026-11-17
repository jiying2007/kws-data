"""Retained-source and dependency tests: stdlib/mock only, no actual processes."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKER = HERE / 'english_hardneg_natural32_retained_once.py'
if not WORKER.exists():
    WORKER = HERE / 'natural32_retained_once.py'
spec = importlib.util.spec_from_file_location('retained_under_test', WORKER)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
REPO_ROOT = HERE.parent if (HERE.parent / w.ORIGINAL_WORKER).exists() else HERE.parent / 'kws-data'


def fixture_sources():
    rows, files, chapters = [], {}, {}
    packed = (16000 << 44) | (15 << 36) | 40000
    block = bytearray(34)
    block[10:18] = packed.to_bytes(8, 'big')
    flac = b'fLaC' + bytes([128, 0, 0, 34]) + bytes(block)
    for i in range(32):
        chapter, rid = str(i % 27), 'fixture-' + str(i)
        uid = '1-' + chapter + '-' + str(i).zfill(4)
        text = ('ORIGINAL WORDS ' + str(i)).encode()
        row = {'recording_id': rid, 'source_member': 'LibriSpeech/dev-clean/1/' + chapter + '/' + uid + '.flac',
               'flac_sha256': w.sha(flac), 'transcript_sha256': w.sha(text), 'frames': 40000}
        rows.append(row)
        files['selectedinputs/flac/' + rid + '.flac'] = flac
        files['selectedinputs/transcripts/' + rid + '.txt'] = text
        chapters.setdefault(chapter, []).append(uid.encode() + b' ' + text + b'\n')
    for chapter, lines in chapters.items():
        files['selectedinputs/transcript-sources/1-' + chapter + '.trans.txt'] = b''.join(lines)
    archive = SimpleNamespace(read=lambda name: files[name], z=SimpleNamespace(namelist=lambda: list(files)))
    return {'rows': rows}, files, archive


def fixture_failure():
    binding = {'repository': w.REPO, 'run_id': str(w.RETAINED['run_id']), 'head_sha': w.BEFORE,
               'main_before': w.FAILURE['head_sha'], 'run_attempt': 1, 'event': 'push', 'public_repository': True}
    error = "ValueError:FileNotFoundError:[Errno 2] No such file or directory: 'ffmpeg'"
    result = {'status': 'FAILED_NO_RETRY', 'passed': False, 'reports': [], 'error': error,
              'candidate_head': w.HEAD, 'native_library': w.LIBRARY,
              'configuration_sha256': w.sha(w.canonical(w.CONFIG)), 'prior_failed_acquisition': w.FAILURE,
              **binding, **dict.fromkeys(w.FALSE_FLAGS, False)}
    for key in ('consumer_build_attempts', 'decoder_process_attempts', 'model_process_attempts', 'completed_validated_traces',
                'prior_compute_attempts', 'new_onnx_calls', 'python_head_forwards', 'optimizer_updates',
                'new_tts_asr_phone_calls', 'new_mixes', 'retry_count'):
        result[key] = 0
    supervisor = {'passed': False, 'returncode': 1, 'timed_out': False, 'error': None, 'partial_receipt_errors': [],
                  'cleanup': {'descendants_found': False, 'terminated': False, 'killed': False},
                  'wall_seconds': 537.4946303590001, **binding}
    start = {'binding': binding, 'network_calls_at_start': 0, 'maximum_seconds': 1200,
             'monotonic_started': 0, 'monotonic_deadline': 1200,
             'worker_sha256': w.RECOVERY_FILES['automation/english_hardneg_natural32_recovery_once.py']['sha256']}
    names = {'current-run', 'corpus-download', 'ffmpeg-version'} | {
        a + '-' + b for a in ('prior-failure', 'training', 'quality', 'sdk') for b in ('run', 'artifact', 'zip')}
    commands = {}
    for name in names:
        common = {'argv': ['ffmpeg', '-version'] if name == 'ffmpeg-version' else ['fixture-metadata'],
                  'kind': None, 'attempt': 1, 'retry_count': 0}
        begun = {**common, 'new_session': False, 'cpu_threads': 1, 'global_deadline': 1200, 'monotonic_started': 10}
        ended = {**common, 'timed_out': False, 'returncode': None if name == 'ffmpeg-version' else 0,
                 'error': "FileNotFoundError:[Errno 2] No such file or directory: 'ffmpeg'" if name == 'ffmpeg-version' else None}
        commands[name] = begun, ended
    stages = {}
    for name in ('source-binding', 'verify-prior-acquisition-failure', 'authenticate-artifacts', 'select-original-corpus', 'decode-originals'):
        stages['stage-' + name] = {'stage': name}, {'stage': name, 'error': error if name == 'decode-originals' else None}
    return result, supervisor, start, commands, stages


class RetainedTests(unittest.TestCase):
    def test_all_eight_previously_consumed_files_are_exact(self):
        for name, pin in {**w.ORIGINAL_FILES, **w.RECOVERY_FILES}.items():
            self.assertEqual(w.identity((REPO_ROOT / name).read_bytes()), pin)

    def test_no_corpus_download_or_tar_reader_remains(self):
        self.assertFalse(hasattr(w, 'select_corpus'))
        self.assertFalse(hasattr(w.Runner, 'download_corpus'))
        self.assertNotIn('import tarfile', WORKER.read_text())
        self.assertNotIn('tarfile.open', WORKER.read_text())
        plan = w.expected_plan(w.sha(WORKER.read_bytes()))
        self.assertEqual(plan['new_corpus_downloads'], 0)
        self.assertEqual(plan['new_tar_reads'], 0)
        self.assertEqual(plan['source_archive_acquisitions'], 2)
        self.assertEqual(plan['workflow_execution_sequence'], 3)
        self.assertEqual(plan['aggregate_maximum_model_processes'], 32)
        self.assertEqual(plan['maximum_execution_seconds'], 1200)

    def test_compute_frontend_event_decoder_and_cleanup_stay_unchanged(self):
        old = ast.parse((REPO_ROOT / 'automation/english_hardneg_natural32_recovery_once.py').read_text())
        new = ast.parse(WORKER.read_text())
        a = {n.name: ast.dump(n, include_attributes=False) for n in old.body if isinstance(n, ast.FunctionDef)}
        b = {n.name: ast.dump(n, include_attributes=False) for n in new.body if isinstance(n, ast.FunctionDef)}
        for name in ('decode_argv', 'flac_streaminfo', 'replay_events', 'validate_trace_metadata', 'supervise', 'cleanup_group'):
            self.assertEqual(a[name], b[name], name)
        self.assertEqual(w.sha(w.canonical(w.CONFIG)), '13ff76c46b35d8d1c427bc237cfc3153771f281379ad86df8203f685aa2e7af4')

    def test_dependency_setup_is_bounded_explicit_and_before_downloads(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            runner = SimpleNamespace(output=Path(d), command=lambda *a, **k: calls.append((a, k)))
            with mock.patch.object(w.shutil, 'which', side_effect=lambda name: '/usr/bin/' + name):
                w.prepare_dependencies(runner)
            self.assertEqual(len(calls), 7)
            for _, kwargs in calls[:2]:
                self.assertEqual(kwargs['timeout'], 180)
                self.assertIsNone(kwargs.get('kind'))
            self.assertIn('Acquire::Retries=0', calls[0][0][1])
            self.assertEqual(calls[0][0][1][:5], ['sudo', '-n', 'env', 'DEBIAN_FRONTEND=noninteractive', 'NEEDRESTART_MODE=a'])
            self.assertEqual(calls[0][0][1][-1], 'update')
            self.assertEqual(calls[1][0][1][-6:], ['install', '--no-install-recommends', '-y', 'ffmpeg', 'gcc', 'binutils'])
            self.assertEqual(calls[2][0][1], ['ffmpeg', '-version'])
        body = WORKER.read_text().split('def child(', 1)[1].split('def group_exists(', 1)[0]
        self.assertLess(body.index("stage('dependency-preflight')"), body.index("stage('source-binding')"))
        self.assertLess(body.index('prepare_retained_source('), body.index("stage('decode-originals')"))

    def test_missing_base_tool_stops_before_install_or_download(self):
        with tempfile.TemporaryDirectory() as d:
            command = mock.Mock()
            runner = SimpleNamespace(output=Path(d), command=command)
            with mock.patch.object(w.shutil, 'which', side_effect=lambda name: None if name == 'gh' else '/bin/' + name), \
                 self.assertRaisesRegex(ValueError, 'REQUIRED_BASE_TOOLS_MISSING'):
                w.prepare_dependencies(runner)
            command.assert_not_called()

    def test_failed_dependency_install_has_no_retry_or_later_version(self):
        with tempfile.TemporaryDirectory() as d:
            command = mock.Mock(side_effect=ValueError('fixture apt failed'))
            runner = SimpleNamespace(output=Path(d), command=command)
            with mock.patch.object(w.shutil, 'which', return_value='/usr/bin/tool'), \
                 self.assertRaisesRegex(ValueError, 'fixture apt failed'):
                w.prepare_dependencies(runner)
            self.assertEqual(command.call_count, 1)

    def test_missing_ffmpeg_after_install_stops_before_any_decode(self):
        with tempfile.TemporaryDirectory() as d:
            command = mock.Mock()
            runner = SimpleNamespace(output=Path(d), command=command)
            with mock.patch.object(w.shutil, 'which', side_effect=lambda name: None if name == 'ffmpeg' else '/bin/' + name), \
                 self.assertRaisesRegex(ValueError, 'REQUIRED_TOOLS_MISSING_AFTER_SETUP'):
                w.prepare_dependencies(runner)
            self.assertEqual(command.call_count, 2)

    def test_retained_32_sources_are_checked_without_pcm_or_decoder(self):
        roster, files, archive = fixture_sources()
        with mock.patch.object(w.subprocess, 'Popen', side_effect=AssertionError('no decoder')):
            selected, paths = w.retained_members(archive, roster)
        self.assertEqual(len(selected), 32)
        self.assertEqual(len(paths), 91)
        self.assertFalse(any(name.endswith('.pcm') for name in paths))

    def test_changed_flac_text_geometry_or_extra_pcm_rejected(self):
        for failure in ('flac', 'text', 'frames', 'pcm'):
            roster, files, archive = fixture_sources()
            if failure == 'flac':
                files['selectedinputs/flac/fixture-0.flac'] += b'changed'
            elif failure == 'text':
                files['selectedinputs/transcripts/fixture-0.txt'] += b'changed'
            elif failure == 'frames':
                roster['rows'][0]['frames'] += 1
            else:
                files['selectedinputs/pcm/fixture-0.pcm'] = b'not authorized'
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                w.retained_members(archive, roster)

    def test_second_failure_has_zero_compute_and_exact_no_launch_cause(self):
        w.verify_retained_failure_semantics(*fixture_failure())
        for field, value in [('decoder_process_attempts', 1), ('model_process_attempts', None),
                             ('consumer_build_attempts', False), ('prior_compute_attempts', 1)]:
            args = fixture_failure()
            args[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'ZERO_COMPUTE'):
                w.verify_retained_failure_semantics(*args)

    def test_second_uncertain_exit_extra_command_or_residue_rejected(self):
        args = fixture_failure()
        args[3]['ffmpeg-version'][1]['returncode'] = 1
        with self.assertRaisesRegex(ValueError, 'FFMPEG_NOT_LAUNCHED'):
            w.verify_retained_failure_semantics(*args)
        args = fixture_failure()
        args[3]['decode-0'] = copy.deepcopy(args[3]['current-run'])
        with self.assertRaisesRegex(ValueError, 'PRECOMPUTE_COMMAND_SET'):
            w.verify_retained_failure_semantics(*args)
        args = fixture_failure()
        args[1]['cleanup']['descendants_found'] = True
        with self.assertRaisesRegex(ValueError, 'CLEAN_SUPERVISOR'):
            w.verify_retained_failure_semantics(*args)


if __name__ == '__main__':
    unittest.main()
