#!/usr/bin/env python3
"""Synthetic intake-control regressions; no network, models or real source data."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import wave

ROOT = Path(os.environ.get('KWS_NEWSOURCE8_TEST_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'automation'))


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


driver = module('intake_control_under_test', 'automation/run_newsource8_intake.py')
fetch = module('recorded_fetch_under_test', 'automation/recorded_aishell_fetch.py')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def pin(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class IntakeControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='newsource8-synthetic-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.runtime = self.home / 'runtime'
        self.source = self.home / 'source'
        self.runtime.mkdir()
        self.source.mkdir()
        self.entries = [dict(dataset_id='aishell3-slr93', speaker_key=f'SSB{i:04d}',
                             source_family_id=f'fam-fixture-{i}', speaker_group_id=f'spk-fixture-{i}',
                             role='development-only' if i % 4 == 3 else 'train-candidate')
                        for i in range(32)]
        self.registry = dict(schema='kws-private-speaker-registry-v1', registry_id='registry-fixture',
                             split_policy='append-three-train-one-development-v1', entries=self.entries)
        self.old_sources = [dict(source_family_id=row['source_family_id'], original_sha256=f'{i+1:064x}',
                                 wav_sha256=f'{i+101:064x}', pcm_sha256=f'{i+201:064x}')
                            for i, row in enumerate(self.entries)]
        put(self.runtime / 'exclusion32-registry.json', self.registry)
        put(self.runtime / 'exclusion32-source-identities.json', {'rows': self.old_sources})
        put(self.runtime / 'research-scope.json', dict(dataset_id='aishell3-slr93',
            dataset_release='synthetic-unit-fixture', source_terms_sha256='a' * 64))
        (self.runtime / 'terms-snapshot.txt').write_text('synthetic fixture')
        (self.runtime / 'research-review.txt').write_text('synthetic fixture')
        self.original_registry = (self.runtime / 'exclusion32-registry.json').read_bytes()
        self.request = dict(schema='kws-newsource8-input-only-v1', prefix_bytes=134217728,
            max_new_speakers=8, model_calls_allowed=0, stage_seconds=1200, fetch_seconds=600,
            runtime_root=str(self.runtime), source_root=str(self.source), source_commit='a' * 40,
            excluded_speaker_keys=sorted(row['speaker_key'] for row in self.entries),
            source_and_input_pins=[pin(p) for p in sorted(self.runtime.iterdir())],
            python=sys.executable, python_library_path='/fixture/no-library',
            etag='"synthetic-release"', archive_bytes=19057141777,
            ffmpeg={'path': '/fixture/no-ffmpeg', 'sha256': 'b' * 64})
        self.partitions = ['test', 'test']
        self.keep_text = True
        self.failure = None
        self.timeout = False
        self.overlap = False
        self.change_old_role = False
        self.calls, self.kills, self.stdout = [], [], io.StringIO()

    def child(self, argv, **kwargs):
        # These are process fixtures, not replacement source-selection algorithms.
        script = Path(argv[4]).name
        phase = {'recorded_aishell_fetch.py': 'fetch', 'fetch_aishell_subset.py': 'select',
                 'prepare_open_reference_pilot.py': 'normalize', '-m': 'registry',
                 'research_only_bulk_ingest.py': 'ingest'}[script]
        self.assertTrue((self.runtime / 'execution-claim.json').is_file())
        self.assertTrue(kwargs['start_new_session'])
        self.calls.append((phase, argv))
        kwargs['stdout'].write(('synthetic child ' + phase + '\n').encode())
        kwargs['stderr'].write(b'synthetic stderr retained\n')
        value = lambda flag: Path(argv[argv.index(flag) + 1])
        if phase == 'fetch':
            prefix = value('--output')
            prefix.write_bytes(b'partial synthetic source bytes')
            if not self.failure and not self.timeout:
                put(prefix.with_name(prefix.name + '.retrieval.json'),
                    dict(archive_bytes=self.request['archive_bytes'], prefix_bytes=self.request['prefix_bytes']))
        elif phase == 'select':
            subset = value('--output')
            refs = [dict(speaker_key=f'SSB{1000+i:04d}', original_path=f'{part}/wav/fixture-{i}.wav',
                         original_partition=part, sha256=hashlib.sha256(f'new-original-{i}'.encode()).hexdigest())
                    for i, part in enumerate(self.partitions)]
            put(subset / 'source-subset.json', {'references': refs})
            if self.keep_text:
                for part in set(self.partitions):
                    text = subset / 'original' / part / 'content.txt'
                    text.parent.mkdir(parents=True, exist_ok=True)
                    text.write_text('synthetic official partition fixture\n')
        elif phase == 'normalize':
            prepared = value('--output-root')
            (prepared / 'normalized').mkdir(parents=True)
            (prepared / 'conversions').mkdir()
            rows = []
            for i, row in enumerate(json.loads((self.runtime / 'selection.json').read_text())['clips']):
                target = prepared / 'normalized' / f'fixture-{i}.wav'
                with wave.open(str(target), 'wb') as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16000)
                    stream.writeframes(bytes([i + 1, 0]) * 16000)
                rows.append(dict(row, normalized_path=target.name, normalized_sha256=pin(target)['sha256']))
            if self.overlap:
                rows[0]['original_sha256'] = self.old_sources[0]['original_sha256']
            put(prepared / 'source-index.json', {'rows': rows})
        elif phase == 'registry':
            target = value('--registry')
            registry = json.loads(target.read_text())
            for row in json.loads(value('--index').read_text())['rows']:
                i = len(registry['entries'])
                registry['entries'].append(dict(dataset_id='aishell3-slr93', speaker_key=row['speaker_key'],
                    source_family_id=f'fam-new-{i}', speaker_group_id=f'spk-new-{i}',
                    role='development-only' if i % 4 == 3 else 'train-candidate'))
            if self.change_old_role:
                registry['entries'][0]['role'] = 'development-only'
            put(target, registry)
        else:
            put(value('--output'), {'synthetic_ingest_fixture': True})
        controller = self
        class Process:
            pid = 123456789
            waits = 0
            def wait(self, timeout=None):
                self.waits += 1
                if controller.timeout and phase == 'fetch':
                    if self.waits == 1:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    return -signal.SIGTERM  # Parent exited; descendant still needs KILL.
                return 23 if controller.failure == phase else 0
        return Process()

    def invoke(self):
        path = self.runtime / 'fixture-request.json'
        put(path, self.request)
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['driver', '--request', str(path),
                                                         '--request-sha256', pin(path)['sha256']]))
            stack.enter_context(patch.object(driver.subprocess, 'check_output', return_value='a' * 40 + '\n'))
            stack.enter_context(patch.object(driver.subprocess, 'Popen', side_effect=self.child))
            stack.enter_context(patch.object(driver.os, 'killpg', side_effect=lambda pid, sig: self.kills.append((pid, sig))))
            stack.enter_context(patch('socket.socket', side_effect=AssertionError('network forbidden in tests')))
            stack.enter_context(redirect_stdout(self.stdout))
            return driver.main()

    def terminal(self):
        return json.loads((self.runtime / 'terminal.json').read_text())

    def phases(self):
        return [name for name, _ in self.calls]

    def test_failed_fetch_retains_partial_receipt_and_claim_blocks_second_execution(self):
        self.failure = 'fetch'
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.phases(), ['fetch'])
        partial = (self.runtime / 'aishell-prefix128m.tgz.partial').read_bytes()
        receipt = json.loads((self.runtime / 'driver/fetch/receipt.json').read_text())
        self.assertEqual(receipt['returncode'], 23)
        self.assertFalse(receipt['timed_out'])
        for field in ['command', 'stdout', 'stderr']:
            self.assertEqual(receipt[field], pin(Path(receipt[field]['path'])))
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.phases(), ['fetch'])
        self.assertEqual((self.runtime / 'aishell-prefix128m.tgz.partial').read_bytes(), partial)

    def test_timeout_kills_descendant_group_even_after_direct_child_exits(self):
        self.timeout = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.phases(), ['fetch'])
        self.assertEqual(self.kills, [(123456789, signal.SIGTERM), (123456789, signal.SIGKILL)])
        receipt = json.loads((self.runtime / 'driver/fetch/receipt.json').read_text())
        self.assertTrue(receipt['timed_out'])
        self.assertEqual(receipt['returncode'], -signal.SIGTERM)
        self.assertEqual(self.terminal()['status'], 'failed')

    def test_zero_new_sources_is_terminal_without_normalization_or_ingest(self):
        self.partitions = []
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.phases(), ['fetch', 'select'])
        self.assertEqual(self.terminal()['status'], 'bounded_no_new_sources')
        self.assertEqual(self.terminal()['actual_selected_new_sources'], 0)

    def test_partial_train_partition_uses_its_original_text_and_keeps_old32(self):
        self.partitions = ['train', 'train']
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.terminal()['status'], 'completed_partial_new_sources')
        self.assertEqual(self.terminal()['actual_selected_new_sources'], 2)
        self.assertEqual((self.runtime / 'exclusion32-registry.json').read_bytes(), self.original_registry)
        after = json.loads((self.runtime / 'speakers.json').read_text())
        self.assertEqual(after['entries'][:32], self.entries)
        self.assertEqual(len(after['entries']), 34)
        decision = json.loads((self.runtime / 'reference-text-decision.json').read_text())
        text = self.runtime / 'subset/original/train/content.txt'
        self.assertEqual(decision['official_transcript'], pin(text))
        self.assertEqual(decision['reference_text_status'], 'dataset-provided')
        argv = self.calls[-1][1]
        self.assertEqual(argv[argv.index('--transcript-source') + 1], str(text))
        self.assertFalse(self.terminal()['reference_content_checked'])
        self.assertFalse(self.terminal()['training_admitted'])

    def unknown_text(self):
        self.assertEqual(self.invoke(), 0)
        decision = json.loads((self.runtime / 'reference-text-decision.json').read_text())
        self.assertEqual(decision['reference_text_status'], 'unknown')
        self.assertIsNone(decision['official_transcript'])
        for flag in ['--transcript-source', '--transcript-sha256', '--transcript-format']:
            self.assertNotIn(flag, self.calls[-1][1])

    def test_mixed_partitions_keep_text_unknown(self):
        self.partitions = ['test', 'train']
        self.unknown_text()

    def test_missing_partition_text_stays_unknown(self):
        self.keep_text = False
        self.unknown_text()

    def test_old_audio_overlap_stops_before_registry_or_ingest(self):
        self.overlap = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.phases(), ['fetch', 'select', 'normalize'])
        self.assertIn('OVERLAPS_OLD32', self.terminal()['error']['message'])

    def test_changed_old_assignment_stops_before_ingest(self):
        self.change_old_role = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.phases(), ['fetch', 'select', 'normalize', 'registry'])
        self.assertIn('OLD_IDENTITIES_OR_ASSIGNMENTS_CHANGED', self.terminal()['error']['message'])
        self.assertEqual((self.runtime / 'exclusion32-registry.json').read_bytes(), self.original_registry)

    def test_budget_or_pinned_input_drift_stops_before_any_claim_or_child(self):
        self.request['prefix_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'FROZEN_TRANCHE'):
            self.invoke()
        self.request['prefix_bytes'] -= 1
        (self.runtime / 'terms-snapshot.txt').write_text('changed fixture')
        with self.assertRaisesRegex(ValueError, 'PINNED_INPUT_CHANGED'):
            self.invoke()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.runtime / 'execution-claim.json').exists())

    def test_symlink_pinned_interpreter_is_rejected_before_execution_claim(self):
        regular = self.home / 'regular-interpreter-fixture'
        regular.write_bytes(b'synthetic interpreter identity; never executed')
        alias = self.home / 'interpreter-alias'
        alias.symlink_to(regular)
        self.request['python'] = str(alias)
        # A content hash follows the link; the driver's stronger type guard must reject it.
        self.request['source_and_input_pins'].append(pin(alias))
        with self.assertRaisesRegex(ValueError, 'REGULAR_FILE_REQUIRED'):
            self.invoke()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.runtime / 'execution-claim.json').exists())
        self.assertFalse((self.runtime / 'driver').exists())


class RecordedFetchTests(unittest.TestCase):
    def test_actual_helper_body_check_and_attempt_receipts_without_network(self):
        with tempfile.TemporaryDirectory(prefix='newsource8-http-fixture-') as folder:
            root = Path(folder)
            output, attempts = root / 'prefix', root / 'attempts'
            class Response(io.BytesIO):
                status = 206
                headers = {'ETag': '"fixture"', 'Content-Range': 'bytes 0-7/1000'}
                def geturl(self):
                    return fetch.fetcher.URL
            opens = []
            def opened(request, timeout):
                self.assertTrue((attempts / 'attempt-0001/request.json').is_file())
                opens.append(request.full_url)
                return Response(b'12345678')
            with patch.object(sys, 'argv', ['fetch', '--output', str(output), '--etag', '"fixture"',
                                          '--bytes', '8', '--attempts', str(attempts)]), \
                 patch.object(fetch.urllib.request, 'build_opener') as build, \
                 patch('socket.socket', side_effect=AssertionError('network forbidden in tests')), \
                 redirect_stdout(io.StringIO()):
                build.return_value.open.side_effect = opened
                self.assertEqual(fetch.main(), 0)
            self.assertEqual(opens, [fetch.fetcher.URL])
            self.assertEqual(output.read_bytes(), b'12345678')
            receipt = json.loads((root / 'prefix.range-000000000000.json').read_text())
            self.assertEqual(receipt['bytes'], 8)
            self.assertEqual(receipt['sha256'], hashlib.sha256(b'12345678').hexdigest())
            response = json.loads((attempts / 'attempt-0001/response.json').read_text())
            self.assertEqual(response['status'], 'headers_returned_body_not_yet_verified')
            self.assertFalse(response['redirects_allowed'])
            self.assertIsNone(fetch.NoRedirect().redirect_request(None, None, 302, 'fixture', {}, 'https://invalid'))


if __name__ == '__main__':
    unittest.main()
