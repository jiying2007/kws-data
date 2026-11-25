"""Pure rejection tests for the one approved Q1 opening; no original download."""
from __future__ import annotations
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24q1_r1_source_failure as failure


class Opening(unittest.TestCase):
    def inert(self):
        value = failure.document(failure.BINDING.read_bytes())
        value.update(enabled=False, original_source_lock=None, expected_failure_report_sha256=None, assessment=None,
                     historical_git_snapshot_transport=None)
        return value

    def read(self, value, require_enabled=False):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'opening.json'
            path.write_bytes(failure.encoded(value))
            return failure.read_binding(require_enabled, path)

    def test_approved_resources_and_immutable_corrected_reader(self):
        approval, design = failure.approval_resources()
        self.assertEqual(approval['user_statement'], '批准提案，继续一次全新评估（推荐）')
        self.assertIs(approval['user_message_sent_at_utc'], None)
        self.assertTrue(design['original_whole_panel_empty_return_policy_changed'])
        self.assertTrue(design['original_native_validity_and_numerical_admission_unchanged'])
        self.assertEqual(failure.sha(failure.READER_PATH.read_bytes()), failure.READER_SHA)
        self.assertEqual(failure.READER_PATH.name, 'english_v6_b24r1_source_stop_readback.py')

    def test_enabled_only_switch_and_forged_report_do_not_open_a_panel(self):
        inert = self.inert()
        self.assertFalse(self.read(inert)['enabled'])
        with self.assertRaises(ValueError):
            self.read(inert, True)
        for change in ({'enabled': True}, {'expected_failure_report_sha256': failure.REPORT_SHA},
                       {'assessment': dict(path=failure.ASSESSMENT_PATH, sha256=failure.ASSESSMENT_SHA)},
                       {'original_source_lock': {'fixture': True}}, {'enabled': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read(dict(inert, **change))

    def test_no_scope_expansion_old_reserve_training_or_second_assessment(self):
        inert = self.inert()
        for key in failure.DENIED:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read(dict(inert, **{key: True}))
        for change in ({'maximum_new_assessments': 2}, {'maximum_new_assessments': True},
                       {'candidate_pt_sha256': 'a' * 64}, {'original_train_assessment_sha256': failure.ASSESSMENT_SHA},
                       {'immutable_original_reader_sha256': 'a' * 64}, {'condition': 'resume_old_R1'},
                       {'unexpected_permission': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read(dict(inert, **change))

    def test_exact_approval_design_proposal_and_contract_are_not_replaceable(self):
        inert = self.inert()
        for key in ('approval', 'approved_proposal', 'evaluation_design', 'orchestration_contract'):
            changed = copy.deepcopy(inert)
            changed[key]['sha256'] = 'a' * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.read(changed)

    def test_old_lock_cannot_be_rebound_even_to_structurally_valid_other_evidence(self):
        path = failure.ROOT / 'research/english_v6_b24r1_source_stop_20261125/ORIGINAL_LOCK.json'
        value = failure.document(path.read_bytes())
        self.assertEqual(failure.validate_lock(value), value)
        changed = copy.deepcopy(value)
        changed['artifacts']['source']['sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            failure.validate_lock(changed)

    def test_historical_raw_bytes_cannot_be_reserialized_or_normalized(self):
        raw = b'{"fixture":"unchanged"}'
        expected = failure.pin(raw)
        self.assertEqual(failure.require_pin(raw, expected, 'invented'), raw)
        for changed in (raw + b'\n', b'{ "fixture": "unchanged" }', raw.decode(), raw[:-1], b''):
            with self.subTest(value=changed), self.assertRaises(ValueError):
                failure.require_pin(changed, expected, 'invented')

    def test_metadata_transport_preflight_is_seven_reads_without_artifact_calls(self):
        # Invented HTTP bodies isolate the transport contract from any network.
        commit = failure.canonical(dict(sha=failure.HEAD, commit=dict(tree=dict(sha='a' * 40))))
        raws = dict(run=b'run', jobs=b'jobs', artifacts=b'artifacts', commit=commit, tree=b'tree')
        lock = dict(api={k: failure.pin(v) for k, v in raws.items()}, logs={
            'parakeet': dict(path='PARAKEET_JOB.log', job_id=1, **failure.pin(b'parakeet-log')),
            'readback': dict(path='READBACK_JOB.log', job_id=2, **failure.pin(b'readback-log'))})
        endpoints = {
            'actions/runs/' + str(failure.RUN): raws['run'],
            'actions/runs/' + str(failure.RUN) + '/jobs?per_page=100': raws['jobs'],
            'actions/runs/' + str(failure.RUN) + '/artifacts?per_page=100': raws['artifacts'],
            'commits/' + failure.HEAD: commit,
            'git/trees/' + 'a' * 40 + '?recursive=1': raws['tree'],
            'actions/jobs/1/logs': b'parakeet-log', 'actions/jobs/2/logs': b'readback-log'}
        with patch.object(failure, 'validate_lock', return_value=lock), \
             patch.object(failure, 'gh', side_effect=endpoints.__getitem__) as fetch:
            result = failure.live_metadata(lock, git_pins={k: lock['api'][k] for k in ('commit', 'tree')})
        self.assertEqual(fetch.call_count, 7)
        self.assertEqual(len(result), 7)
        self.assertFalse(any('/zip' in call.args[0] for call in fetch.call_args_list))
        endpoints['actions/jobs/1/logs'] += b'\n'
        with patch.object(failure, 'validate_lock', return_value=lock), \
             patch.object(failure, 'gh', side_effect=endpoints.__getitem__), self.assertRaises(ValueError):
            failure.live_metadata(lock, git_pins={k: lock['api'][k] for k in ('commit', 'tree')})

    def test_duplicate_keys_and_nonfinite_values_are_rejected(self):
        for raw in (b'{"enabled":false,"enabled":true}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                failure.document(raw)

    def test_actual_cli_preflight_reopens_separate_metadata_transport_before_sources(self):
        import english_v6_b24q1_r1_git_transport as transport
        raw = {'metadata/source_' + role + '.json': role.encode() for role in ('commit', 'tree')}
        opening = dict(assessment={'sha256': failure.ASSESSMENT_SHA}, original_source_lock={'fixture': True})
        output = io.StringIO()
        with patch.object(sys, 'argv', ['preflight', 'check-live-metadata']), \
             patch.object(failure, 'read_binding', return_value=opening), \
             patch.object(failure, 'live_metadata', return_value=raw), \
             patch.object(transport, 'fetch', return_value={'commit': b'old-commit', 'tree': b'old-tree'}) as fetch, \
             patch.object(failure, 'fetch', side_effect=AssertionError('no original scientific archives')), \
             contextlib.redirect_stdout(output):
            failure.main()
        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.args[1], {'commit': b'commit', 'tree': b'tree'})
        result = json.loads(output.getvalue())
        self.assertTrue(result['additional_metadata_transport_independently_verified'])
        self.assertEqual(result['additional_metadata_snapshot_archives_reopened'], 1)
        self.assertFalse(result['all14_original_archives_reopened'])
        self.assertFalse(result['scientific_execution_authorized'])
        self.assertEqual(result['new_source_calls'], 0)


if __name__ == '__main__':
    unittest.main()
