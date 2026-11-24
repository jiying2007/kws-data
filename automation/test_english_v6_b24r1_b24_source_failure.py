"""Fail-closed tests for one separate evaluation after an immutable SOURCE STOP."""
import copy
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b24r1_b24_source_failure as failure


def lock_fixture():
    return dict(schema=failure.LOCK_SCHEMA, repo=failure.REPO, run_id=failure.RUN,
        run_head=failure.HEAD, prepared_parent=failure.PARENT, branch=failure.BRANCH,
        workflow=failure.WORKFLOW, artifacts={role: dict(id=i+1, name=name, bytes=200,
        sha256='1'*64) for i, (role, name) in enumerate(failure.NAMES.items())},
        source_producer_manifest_sha256='2'*64, source_manifest_sha256='3'*64,
        history_sha256='4'*64, source_assessment_sha256=failure.TRAIN_ASSESSMENT,
        candidate_pt_sha256=failure.CANDIDATE_PT, train_result_sha256='5'*64,
        logs=dict(parakeet=dict(path='PARAKEET_JOB.log', job_id=138672328824, bytes=200, sha256='6'*64),
                  readback=dict(path='READBACK_JOB.log', job_id=138676788255, bytes=200, sha256='7'*64)))


def counts_fixture():
    values = {}
    for provider in failure.PROVIDERS:
        values[provider] = dict(status='COMPLETE', input_started=120, forward_attempted=120,
            native_returned=120, raw_payload_saved=120, payload_valid=120, capture_saved=120,
            failed=0, not_started=0, entered_without_return=0, entered_without_raw=0)
    values['parakeet'].update(status='STOP_OBSERVER_FAILURE', input_started=49,
        forward_attempted=49, native_returned=49, raw_payload_saved=49, payload_valid=48,
        capture_saved=48, failed=1, not_started=71)
    return values


class SourceStopContracts(unittest.TestCase):
    def test_original_lock_shape_has_no_scientific_authority(self):
        value = lock_fixture()
        self.assertEqual(failure.validate_lock(value), value)

    def test_missing_any_original_or_added_screen_is_rejected(self):
        for role in failure.NAMES:
            value = lock_fixture(); del value['artifacts'][role]
            with self.subTest(role=role), self.assertRaises(ValueError):
                failure.validate_lock(value)
        value = lock_fixture(); value['artifacts']['screen'] = dict(value['artifacts']['source'])
        with self.assertRaises(ValueError): failure.validate_lock(value)

    def test_different_run_head_candidate_or_original_assessment_rejected(self):
        for field in ('run_head', 'prepared_parent', 'candidate_pt_sha256', 'source_assessment_sha256'):
            value = lock_fixture(); value[field] = 'a' * len(value[field])
            with self.subTest(field=field), self.assertRaises(ValueError):
                failure.validate_lock(value)

    def test_disabled_registration_and_generic_enable_are_closed(self):
        value = failure.document(failure.BINDING.read_bytes())
        value.update(enabled=False, original_source_lock=None, expected_failure_report_sha256=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'binding.json'; path.write_bytes(failure.encoded(value))
            self.assertFalse(failure.read_binding(False, path)['enabled'])
            with self.assertRaises(ValueError): failure.read_binding(True, path)
            value['enabled'] = True; path.write_bytes(failure.encoded(value))
            with self.assertRaises(ValueError): failure.read_binding(True, path)

    def test_no_second_replacement_training_retry_or_old_reserve_switch(self):
        original = failure.document(failure.BINDING.read_bytes())
        original.update(enabled=False, original_source_lock=None, expected_failure_report_sha256=None)
        for field in ('new_training_allowed', 'old_source_retry_allowed', 'old_source_continuation_allowed',
                      'open_old_sealed_reserve', 'second_replacement_allowed', 'maximum_new_source_registrations'):
            value = copy.deepcopy(original); value[field] = 2 if field == 'maximum_new_source_registrations' else True
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'binding.json'; path.write_bytes(failure.encoded(value))
                with self.subTest(field=field), self.assertRaises(ValueError): failure.read_binding(False, path)

    def test_only_exact_observed_failure_counts_match(self):
        value = counts_fixture()
        self.assertEqual(failure.validate_failure_pattern(value), value)

    def test_complete_panel_or_only_changed_failure_label_cannot_open(self):
        values = counts_fixture(); values['parakeet'] = dict(values['qwen'])
        with self.assertRaises(ValueError): failure.validate_failure_pattern(values)
        values['parakeet']['status'] = 'STOP_OBSERVER_FAILURE'
        with self.assertRaises(ValueError): failure.validate_failure_pattern(values)

    def test_lost_return_lost_raw_or_extra_forward_remain_closed(self):
        for changes in [dict(native_returned=48, entered_without_return=1),
                        dict(raw_payload_saved=48, entered_without_raw=1),
                        dict(forward_attempted=50, input_started=50, not_started=70),
                        dict(payload_valid=49, capture_saved=49), dict(not_started=70)]:
            values = counts_fixture(); values['parakeet'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                failure.validate_failure_pattern(values)

    def test_other_provider_failure_cannot_be_hidden_by_aggregate_totals(self):
        values = counts_fixture(); values['qwen'].update(payload_valid=119, capture_saved=119, failed=1)
        with self.assertRaises(ValueError): failure.validate_failure_pattern(values)

    def test_unsafe_zip_path_is_rejected_after_exact_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'original.zip'
            with zipfile.ZipFile(path, 'w') as z: z.writestr('../outside', b'data')
            with self.assertRaises(ValueError): failure.original(path, failure.pin(path.read_bytes()))

    def test_json_duplicates_nonfinite_and_noncanonical_report_changes(self):
        for raw in (b'{"same":1,"same":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError): failure.document(raw)
        self.assertEqual(failure.canonical({'b': 2, 'a': 1}), b'{"a":1,"b":2}')
        self.assertNotEqual(failure.sha(failure.canonical({'allowed': False})),
                            failure.sha(failure.canonical({'allowed': True})))

    def test_pure_worker_blocks_runtime_network_and_process(self):
        with failure.offline_only():
            with self.assertRaises(ValueError): __import__('torch')
            with self.assertRaises(RuntimeError): socket.socket()
            with self.assertRaises(RuntimeError): subprocess.Popen(['true'])


if __name__ == '__main__':
    unittest.main()
