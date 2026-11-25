"""Pure saved-evidence rejection contracts for the original B27 engineering stop."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_source_preconsumption_stop as stop


class OriginalStopContracts(unittest.TestCase):
    def records(self):
        """Minimum API-shaped test fixtures; these are never published evidence."""
        run = dict(id=stop.RUN, head_sha=stop.HEAD, run_attempt=1,
            status='completed', conclusion='failure', event='push',
            head_branch=stop.BRANCH, path=stop.WORKFLOW,
            repository=dict(full_name=stop.REPO, private=False),
            head_repository=dict(full_name=stop.REPO, private=False),
            head_commit=dict(id=stop.HEAD, tree_id=stop.TREE))
        jobs = dict(total_count=4, jobs=[dict(name=name, id=ident, run_id=stop.RUN,
            run_attempt=1, head_sha=stop.HEAD, head_branch=stop.BRANCH,
            status='completed', conclusion=conclusion,
            steps=[dict(number=number, name=label, conclusion=result, status='completed')
                   for number, label, result in steps])
            for name, (ident, conclusion, steps) in stop.JOB_IDENTITIES.items()])
        artifacts = dict(total_count=2, artifacts=[dict(id=pin['id'], name=pin['name'], expired=False,
            size_in_bytes=pin['bytes'], digest='sha256:' + pin['sha256'],
            workflow_run=dict(id=stop.RUN, head_sha=stop.HEAD, head_branch=stop.BRANCH))
            for pin in (stop.PRODUCER, stop.TRANSPORT)])
        return run, jobs, artifacts

    def test_exact_terminal_metadata_shape_is_accepted_without_io(self):
        value = stop.validate_remote(*self.records())
        self.assertEqual(value['original_run_id'], stop.RUN)
        self.assertEqual(value['complete_jobs'], 4)
        self.assertEqual(value['complete_artifacts'], 2)
        self.assertTrue(value['no_scientific_consumption_or_source_artifact'])

    def test_any_entered_transport_consume_upload_or_generation_is_rejected(self):
        for number in (8, 10, 11, 12):
            for result in ('success', 'failure', 'cancelled', None):
                with self.subTest(number=number, result=result):
                    run, jobs, artifacts = self.records()
                    source = next(job for job in jobs['jobs'] if job['name'] == 'sources')
                    next(step for step in source['steps'] if step['number'] == number)['conclusion'] = result
                    with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)

    def test_no_observer_or_readback_execution_is_permitted(self):
        for name in ('observers', 'readback'):
            run, jobs, artifacts = self.records()
            next(job for job in jobs['jobs'] if job['name'] == name)['conclusion'] = 'success'
            with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)
            run, jobs, artifacts = self.records()
            next(job for job in jobs['jobs'] if job['name'] == name)['steps'] = [dict(number=1, name='entered')]
            with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)

    def test_foreign_rerun_changed_head_or_unfinished_run_is_rejected(self):
        for key, value in (('id', 1), ('run_attempt', 2), ('run_attempt', True),
                ('head_sha', '0' * 40), ('head_branch', 'other'), ('path', 'other.yml'),
                ('status', 'in_progress'), ('conclusion', 'cancelled'), ('event', 'workflow_dispatch')):
            with self.subTest(key=key, value=value):
                run, jobs, artifacts = self.records(); run[key] = value
                with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)
        for field in ('repository', 'head_repository'):
            run, jobs, artifacts = self.records(); run[field]['private'] = True
            with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)
            run, jobs, artifacts = self.records(); run[field]['full_name'] = 'other/kws-data'
            with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)

    def test_complete_unique_job_and_artifact_inventories_are_required(self):
        for target in ('jobs', 'artifacts'):
            run, jobs, artifacts = self.records(); (jobs if target == 'jobs' else artifacts)['total_count'] += 1
            with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = self.records(); jobs['jobs'][0] = copy.deepcopy(jobs['jobs'][1])
        with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = self.records(); jobs['jobs'][0]['head_sha'] = '0' * 40
        with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = self.records(); artifacts['artifacts'][1] = copy.deepcopy(artifacts['artifacts'][0])
        with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)

    def test_consumption_source_or_observer_artifact_closes_the_gate(self):
        for name in ('english-v6-b27-primary-source-consumed',
                     'english-v6-b27-primary-source-45589429827',
                     'english-v6-b27-primary-observer-qwen-consumed'):
            run, jobs, artifacts = self.records()
            item = copy.deepcopy(artifacts['artifacts'][0]); item.update(id=9, name=name)
            artifacts['artifacts'].append(item); artifacts['total_count'] += 1
            with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)

    def test_expired_replaced_or_foreign_archive_is_rejected(self):
        for key, value in (('expired', True), ('digest', 'sha256:' + '0' * 64),
                           ('size_in_bytes', 1), ('name', 'replacement')):
            run, jobs, artifacts = self.records(); artifacts['artifacts'][0][key] = value
            with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)
        run, jobs, artifacts = self.records(); artifacts['artifacts'][0]['workflow_run']['id'] = 1
        with self.assertRaises(ValueError): stop.validate_remote(run, jobs, artifacts)

    def test_truncated_or_false_tree_identity_is_rejected(self):
        for tree in (dict(sha=stop.TREE, truncated=True, tree=[]),
                     dict(sha=stop.TREE, truncated=False, tree=[]),
                     dict(sha='0' * 40, truncated=False, tree=[])):
            with self.assertRaises(ValueError): stop.validate_tree(tree)

    def test_json_duplicates_nonfinite_and_unsafe_paths_are_rejected(self):
        for raw in (b'{"same":1,"same":2}', b'{"value":NaN}', b'[]'):
            with self.assertRaises(ValueError): stop.document(raw)
        for name in ('../a', '/a', 'a//b', 'a/./b', 'a/../b', '.git/config', 'a\\b', 'a\0b'):
            with self.assertRaises(ValueError): stop.safe_path(name)

    def test_archive_duplicates_unsafe_paths_or_changed_whole_bytes_are_rejected(self):
        for names in (['../a'], ['a', 'a'], ['dir/']):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                for name in names: archive.writestr(name, b'original')
            raw = stream.getvalue(); pin = dict(bytes=len(raw), sha256=stop.sha(raw))
            with self.assertRaises(ValueError): stop.original_archive(raw, pin)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive: archive.writestr('original.json', b'{}')
        raw = stream.getvalue(); pin = dict(bytes=len(raw), sha256=stop.sha(raw))
        with stop.original_archive(raw, pin) as archive:
            self.assertEqual(archive.read('original.json'), b'{}')
        with self.assertRaises(ValueError): stop.original_archive(raw + b'changed', pin)

    @unittest.skipUnless(stop.FAILURE_DIR.is_dir(), 'original external evidence not staged in this checkout')
    def test_complete_actual_original_proof_and_rejection_of_mutation(self):
        report = stop.verify()
        self.assertEqual(report['original_producer_files'], 439)
        self.assertEqual(report['actual_new_tts_calls'], 0)
        self.assertEqual(report['actual_new_natural_original_decodes'], 0)
        self.assertEqual(report['actual_new_observer_calls'], 0)
        self.assertEqual(report['actual_new_kws_forwards'], 0)
        self.assertFalse(report['new_scientific_execution_authorized'])
        self.assertFalse(report['original_train_proof_reverified_by_this_reader'])
        self.assertEqual(stop.canonical(report), stop.canonical(stop.verify()))
        tree = json.loads((stop.FAILURE_DIR / 'tree.json').read_bytes())
        tree['tree'][0]['sha'] = '0' * 40
        with self.assertRaises(ValueError): stop.validate_tree(tree)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / 'proof'; shutil.copytree(stop.FAILURE_DIR, root)
            path = root / 'source-job.log'; path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'source log'): stop.verify(root)
            shutil.copyfile(stop.FAILURE_DIR / 'source-job.log', path)
            (root / 'CONSUMPTION.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'nine-file'): stop.verify(root)
            (root / 'CONSUMPTION.json').unlink()
            (root / 'extra').symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'regular files'): stop.verify(root)


if __name__ == '__main__':
    unittest.main()
