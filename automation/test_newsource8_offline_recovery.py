#!/usr/bin/env python3
"""Offline-recovery control regressions with invented bytes and fake local children."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('KWS_NEWSOURCE8_TEST_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'automation')]
import test_newsource8_intake as fixtures

recovery = fixtures.module('offline_recovery_under_test', 'automation/recover_newsource8_partial.py')
MIB = 1048576


class RecoveryControlTests(unittest.TestCase):
    def setUp(self):
        # Reuse the already adopted local-child fixtures, not a second ingestion model.
        self.f = fixtures.IntakeControlTests(methodName='runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.original = self.f.runtime
        self.f.runtime = self.f.home / 'recovery'
        self.f.runtime.mkdir()
        self.prefix = self.original / 'aishell-prefix128m.tgz.partial'
        self.blocks = [b'A' * MIB, b'B' * MIB]
        self.tail = b'unconfirmed-original-tail-must-remain'
        self.prefix.write_bytes(b''.join(self.blocks) + self.tail)
        self.original_prefix_bytes = self.prefix.read_bytes()
        self.range_paths, self.http_requests, self.http_responses = [], [], []
        self.url, self.etag, self.archive_bytes = 'https://invalid.example/fixture.tgz', '"fixture-release"', 19057141777
        for i, block in enumerate(self.blocks):
            first, last = i * MIB, (i + 1) * MIB - 1
            record = self.prefix.with_name(self.prefix.name + f'.range-{i:06d}.json')
            fixtures.put(record, dict(url=self.url, etag=self.etag, archive_bytes=self.archive_bytes,
                start=first, end=last, bytes=MIB, sha256=hashlib.sha256(block).hexdigest()))
            self.range_paths.append(record)
            request = self.original / 'http-attempts' / f'attempt-{i+1:04d}' / 'request.json'
            response = request.with_name('response.json')
            fixtures.put(request, dict(url=self.url, method='GET', request_headers={
                'Range': f'bytes={first}-{last}', 'If-range': self.etag}))
            fixtures.put(response, dict(http_status=206, status='headers_returned_body_not_yet_verified',
                response_url=self.url, redirects_allowed=False, response_headers={
                'ETag': self.etag, 'Content-Range': f'bytes {first}-{last}/{self.archive_bytes}',
                'Content-Length': str(MIB), 'Content-Encoding': None}))
            self.http_requests.append(request)
            self.http_responses.append(response)
        self.launch = self.f.home / 'original-launch/receipt.json'
        fixtures.put(self.launch, {'returncode': 1})
        fixtures.put(self.original / 'terminal.json', {'status': 'failed', 'new_model_calls': 0})
        fixtures.put(self.original / 'driver/fetch/receipt.json', {'returncode': -15, 'timed_out': True})
        self.original_terminal_bytes = (self.original / 'terminal.json').read_bytes()
        self.request = dict(schema='kws-newsource8-offline-recovery-v1', worker_seconds=300,
            max_new_speakers=8, network_calls_allowed=0, model_calls_allowed=0,
            runtime_root=str(self.f.runtime), source_root=str(self.f.source), original_runtime=str(self.original),
            original_launch_receipt=str(self.launch), source_commit='a' * 40,
            python=sys.executable, python_library_path='/fixture/no-library',
            url=self.url, etag=self.etag, archive_bytes=self.archive_bytes, confirmed_prefix_bytes=2*MIB,
            ffmpeg={'path': '/fixture/no-ffmpeg', 'sha256': 'b' * 64})

    def bind(self):
        self.request['original_prefix'] = fixtures.pin(self.prefix)
        self.request['confirmed_ranges'] = [fixtures.pin(p) for p in self.range_paths]
        self.request['http_request_pins'] = [fixtures.pin(p) for p in self.http_requests]
        self.request['http_response_pins'] = [fixtures.pin(p) for p in self.http_responses]
        files = sorted(p for p in self.original.rglob('*') if p.is_file())
        self.request['source_and_input_pins'] = [fixtures.pin(p) for p in [*files, self.launch]]

    def invoke(self):
        self.bind()
        path = self.f.runtime / 'fixture-request.json'
        fixtures.put(path, self.request)
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['recovery', '--request', str(path),
                '--request-sha256', fixtures.pin(path)['sha256']]))
            stack.enter_context(patch.object(recovery.subprocess, 'check_output', return_value='a' * 40 + '\n'))
            stack.enter_context(patch.object(recovery.subprocess, 'Popen', side_effect=self.f.child))
            stack.enter_context(patch('socket.socket', side_effect=AssertionError('network forbidden')))
            stack.enter_context(patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')))
            stack.enter_context(redirect_stdout(self.f.stdout))
            return recovery.main()

    def terminal(self):
        return json.loads((self.f.runtime / 'terminal.json').read_text())

    def change(self, path, **values):
        record = json.loads(path.read_text())
        record.update(values)
        fixtures.put(path, record)

    def rejected_before_copy(self, reason):
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.f.phases(), [])
        self.assertEqual(self.terminal()['status'], 'failed')
        self.assertIn(reason, self.terminal()['error']['message'])
        self.assertFalse((self.f.runtime / 'verified-prefix.tgz.partial').exists())
        self.assertEqual(self.prefix.read_bytes(), self.original_prefix_bytes)

    def test_exact_confirmed_copy_preserves_tail_failed_origin_and_old32_without_network(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.f.phases(), ['select', 'normalize', 'registry', 'ingest'])
        self.assertEqual((self.f.runtime / 'verified-prefix.tgz.partial').read_bytes(), b''.join(self.blocks))
        self.assertEqual(self.prefix.read_bytes(), self.original_prefix_bytes)
        self.assertEqual((self.original / 'terminal.json').read_bytes(), self.original_terminal_bytes)
        self.assertEqual((self.original / 'exclusion32-registry.json').read_bytes(), self.f.original_registry)
        registry = json.loads((self.f.runtime / 'speakers.json').read_text())
        self.assertEqual(registry['entries'][:32], self.f.entries)
        self.assertEqual(len(registry['entries']), 34)
        provenance = json.loads((self.f.runtime / 'verified-prefix-provenance.json').read_text())
        self.assertEqual(provenance['confirmed_prefix_bytes'], 2*MIB)
        self.assertEqual(provenance['unconfirmed_original_tail_bytes_preserved'], len(self.tail))
        self.assertFalse(provenance['original_fetch_success_claimed'])
        self.assertTrue(provenance['not_a_retrieval_success_receipt'])
        self.assertEqual(list(self.f.runtime.rglob('*.retrieval.json')), [])
        result = self.terminal()
        self.assertEqual(result['status'], 'completed_partial_new_sources')
        self.assertEqual(result['actual_selected_new_sources'], 2)
        self.assertEqual(result['new_network_calls'], 0)
        self.assertEqual(result['new_model_calls'], 0)
        self.assertFalse(result['training_admitted'])
        self.assertNotIn('torch', sys.modules)

    def test_range_digest_mismatch_rejects_before_downstream_or_copy(self):
        self.change(self.range_paths[0], sha256='0' * 64)
        self.rejected_before_copy('CONFIRMED_RANGE_BODY_CHANGED')

    def test_gap_in_complete_ranges_rejects_before_downstream_or_copy(self):
        self.change(self.range_paths[1], start=2*MIB, end=3*MIB-1)
        self.rejected_before_copy('RANGE_IDENTITY_OR_CONTIGUITY')

    def test_omitted_existing_complete_range_rejects_roster(self):
        self.range_paths = self.range_paths[:1]
        self.rejected_before_copy('CONFIRMED_RANGE_ROSTER_CHANGED')

    def test_missing_actual_206_rejects_even_with_valid_body_digest(self):
        self.change(self.http_responses[1], http_status=200)
        self.rejected_before_copy('CONFIRMED_RANGE_NEEDS_ONE_ACTUAL_206_RESPONSE')

    def test_confirmed_length_cannot_include_unconfirmed_tail(self):
        self.request['confirmed_prefix_bytes'] += len(self.tail)
        self.rejected_before_copy('CONFIRMED_LENGTH_CHANGED')

    def test_successful_original_fetch_or_terminal_rejects_before_claim(self):
        for terminal, fetch, launch in (
            ({'status': 'completed_eight_new_sources', 'new_model_calls': 0}, {'returncode': 0, 'timed_out': False}, 0),
            ({'status': 'failed', 'new_model_calls': 0}, {'returncode': 0, 'timed_out': False}, 1)):
            with self.subTest(terminal=terminal['status'], fetch_returncode=fetch['returncode']):
                fixtures.put(self.original / 'terminal.json', terminal)
                fixtures.put(self.original / 'driver/fetch/receipt.json', fetch)
                fixtures.put(self.launch, {'returncode': launch})
                with self.assertRaisesRegex(ValueError, 'ONLY_ACTUAL_FAILED_FETCH'):
                    self.invoke()
                self.assertFalse((self.f.runtime / 'execution-claim.json').exists())
                self.assertEqual(self.f.phases(), [])

    def test_duplicate_execution_claim_blocks_second_recovery(self):
        self.assertEqual(self.invoke(), 0)
        phases = self.f.phases()
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(self.f.phases(), phases)
        self.assertEqual(self.prefix.read_bytes(), self.original_prefix_bytes)

    def test_zero_new_sources_is_truthful_without_normalize_or_ingest(self):
        self.f.partitions = []
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.f.phases(), ['select'])
        self.assertEqual(self.terminal()['status'], 'bounded_no_new_sources')
        self.assertEqual(self.terminal()['actual_selected_new_sources'], 0)

    def test_old_audio_identity_collision_blocks_registry_and_ingest(self):
        self.f.overlap = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.f.phases(), ['select', 'normalize'])
        self.assertIn('RECOVERY_AUDIO_OVERLAPS_ORIGINAL32', self.terminal()['error']['message'])

    def test_old32_assignment_change_blocks_ingest(self):
        self.f.change_old_role = True
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.f.phases(), ['select', 'normalize', 'registry'])
        self.assertIn('RECOVERY_REGISTRY_ANCESTRY', self.terminal()['error']['message'])
        self.assertEqual((self.original / 'exclusion32-registry.json').read_bytes(), self.f.original_registry)

    def test_failed_local_child_retains_actual_receipt_and_stops(self):
        self.f.failure = 'select'
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.f.phases(), ['select'])
        receipt = json.loads((self.f.runtime / 'driver/select/receipt.json').read_text())
        self.assertEqual(receipt['returncode'], 23)
        self.assertFalse(receipt['timed_out'])
        for key in ('command', 'stdout', 'stderr'):
            self.assertEqual(receipt[key], fixtures.pin(Path(receipt[key]['path'])))
        self.assertEqual(self.terminal()['status'], 'failed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
