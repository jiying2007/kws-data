"""The dependency repair cannot create another scientific assessment or budget."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_source_ancestry as lineage
import english_v6_b27_replication_source_ancestry as replica
import english_v6_b27_primary_eval as primary
import english_v6_b27_replication_eval as replica_eval
import english_v6_b27_quality_readback as reader
import english_v6_b27_source_preconsumption_stop as stop
import english_v6_b27_profile as profile
import english_v6_b27_producer as training

HERE = Path(__file__).resolve().parent
PARENT, HEAD = 'a' * 40, 'b' * 40
IDENTITY = dict(run_id='45599999999', run_head=HEAD, run_attempt=1,
                run_branch=lineage.BRANCH, prepared_parent=PARENT)


class ContinuationContracts(unittest.TestCase):
    def test_only_new_first_push_and_unchanged_original_marker(self):
        env = dict(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=lineage.REPO,
            GITHUB_REF=lineage.BRANCH, GITHUB_EVENT_NAME='push', GITHUB_RUN_ATTEMPT='1',
            GITHUB_RUN_ID=IDENTITY['run_id'], GITHUB_SHA=HEAD)
        event = dict(before=PARENT, after=HEAD, ref=lineage.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=lineage.REPO, private=False))
        self.assertEqual(lineage.execution_guard(PARENT, env, event, PARENT), IDENTITY)
        for key, value in (('GITHUB_REF', 'refs/heads/' + stop.BRANCH), ('GITHUB_RUN_ATTEMPT', '2')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                lineage.execution_guard(PARENT, dict(env, **{key: value}), event, PARENT)
        self.assertEqual(primary.SOURCE_BRANCH, lineage.BRANCH.removeprefix('refs/heads/'))
        self.assertEqual(replica_eval.SOURCE_BRANCH, replica.BRANCH.removeprefix('refs/heads/'))
        self.assertNotEqual(lineage.BRANCH, replica.BRANCH)
        self.assertEqual(lineage.MARKER, 'english-v6-b27-primary-source-consumed')
        for provider in lineage.PROVIDERS:
            self.assertEqual(lineage.marker_name(provider), 'english-v6-b27-primary-observer-' + provider + '-consumed')

    def test_any_existing_source_or_observer_budget_blocks_authorization(self):
        run = dict(total_count=1, workflow_runs=[dict(id=int(IDENTITY['run_id']), head_sha=HEAD, run_attempt=1)])
        for provider in (None, *lineage.PROVIDERS):
            with self.subTest(provider=provider), patch.object(lineage, 'gh', side_effect=[
                    json.dumps(run).encode(), json.dumps(dict(total_count=1, artifacts=[{}])).encode()]):
                with self.assertRaises(ValueError):
                    lineage.one_execution(IDENTITY, False, provider)

    def test_authorize_rechecks_original_terminal_run_before_training_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary).resolve() / 'authorization'
            with patch.object(lineage, 'execution_guard', return_value=IDENTITY), \
                 patch.object(lineage, 'gh', return_value=b'{}') as remote, \
                 patch.object(lineage, 'verify_training') as training_check:
                with self.assertRaises(ValueError):
                    lineage.authorize(out, PARENT)
                self.assertEqual(remote.call_count, 3)
                training_check.assert_not_called()
                self.assertFalse(out.exists())

    def test_authorize_checks_all_six_original_budget_names_before_output(self):
        originals = [json.loads((stop.FAILURE_DIR / name).read_bytes())
                     for name in ('run.json', 'jobs.json', 'artifacts.json')]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary).resolve() / 'authorization'
            calls = []
            def budget(identity, marker_present, provider=None):
                self.assertEqual(identity, IDENTITY)
                self.assertFalse(marker_present)
                calls.append(provider)
                if provider == lineage.PROVIDERS[-1]:
                    raise ValueError('final observer marker exists')
            with patch.object(lineage, 'execution_guard', return_value=IDENTITY), \
                 patch.object(lineage, 'gh', side_effect=[json.dumps(value).encode() for value in originals]), \
                 patch.object(lineage, 'verify_training'), patch.object(lineage, 'one_execution', side_effect=budget):
                with self.assertRaises(ValueError):
                    lineage.authorize(out, PARENT)
            self.assertEqual(calls, [None, *lineage.PROVIDERS])
            self.assertFalse(out.exists())

    def test_full_stop_proof_remains_usable_inside_original_offline_reader(self):
        with tempfile.TemporaryDirectory() as temporary, reader.offline_only(Path(temporary).resolve(), 'primary'):
            report = stop.verify()
            self.assertEqual(profile.sha(stop.canonical(report)),
                             '9a7e6ccdc2598c2503e291ad23a4da0c10a6bdfa3a0fdde85437a23104efa577')
            lineage.read_plan()

    def test_original_training_producer_and_scientific_profile_are_unchanged(self):
        paths = training.paths()
        self.assertEqual(len(paths), 407)
        self.assertFalse(paths & {'automation/english_v6_b27_source_ancestry.py',
            'automation/english_v6_b27_source_producer.py', 'automation/english_v6_b27_primary_eval.py',
            'automation/english_v6_b27_source_workflow_template.yml', lineage.WORKFLOW})
        self.assertEqual(profile.sha(profile.PROFILE.read_bytes()), stop.PROFILE_SHA)
        self.assertEqual(profile.sha(lineage.INPUTS.read_bytes()), stop.PROOF_BINDING_SHA)


if __name__ == '__main__':
    unittest.main()
