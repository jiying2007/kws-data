"""Real captured-only transport resource checks; no plan or proof mocks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_source_producer as producer

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'automation/english_v6_b27_source_producer.py'


class CapturedResourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='b27-original-source-resources-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name).resolve() / 'captured'
        cls.base.mkdir()
        names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
        tracked = {name.decode('utf-8'): {} for name in names.split(b'\0') if name}
        cls.tracked = tracked
        cls.paths = producer.select_paths(tracked, ROOT, 'primary')
        cls.resources = producer.transport_resource_paths(ROOT)
        for name in cls.paths:
            target = cls.base / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='b27-captured-resource-case-')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.captured = self.directory / 'captured'
        shutil.copytree(self.base, self.captured)

    def run_preflight(self):
        flags = ['-I', '-S', '-B']
        if sys.flags.optimize:
            flags.append('-' + 'O' * sys.flags.optimize)
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'automation'))
        return subprocess.run([sys.executable, *flags, str(self.captured / SCRIPT), '--resource-preflight'],
                              cwd=ROOT, env=environment, capture_output=True, timeout=90)

    def test_real_captured_plan_reopens_every_original_without_scientific_calls(self):
        self.assertEqual(len(self.resources), 25)
        self.assertTrue(self.resources <= self.paths)
        before = {name: hashlib.sha256((self.captured / name).read_bytes()).hexdigest() for name in self.paths}
        result = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8'))
        self.assertEqual(result.stderr, b'')
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'PASS_ORIGINAL_TRANSPORT_AND_PROOFS')
        self.assertTrue(self.resources <= set(report['resource_files']))
        self.assertEqual(report['original_plan_sha256'], producer.TRANSPORT_PLAN_SHA256)
        self.assertEqual(report['original_plan_executions'], 1)
        self.assertEqual(report['corpus_bytes'], 6387309499)
        for name in ('new_source_calls', 'new_model_calls', 'new_network_calls', 'new_audio_reads', 'optimizer_updates'):
            self.assertEqual(report[name], 0)
        self.assertFalse(report['acoustic_evidence'])
        self.assertTrue(report['original_helpers_unchanged'])
        self.assertGreater(report['authenticated_original_helper_files'], 0)
        self.assertEqual(report['reviewed_metadata_adapters'], producer.REVIEWED_METADATA_ADAPTERS)
        for name, pin in report['resource_files'].items():
            self.assertEqual(pin['sha256'], before[name])
        after = {name: hashlib.sha256((self.captured / name).read_bytes()).hexdigest() for name in self.paths}
        self.assertEqual(after, before)
        self.assertFalse(any(self.captured.rglob('__pycache__')))

    def test_missing_original_plan_cannot_fall_back_to_checkout_or_pythonpath(self):
        (self.captured / producer.TRANSPORT_PLAN).unlink()
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')
        self.assertIn(b'complete regular original transport resource', result.stderr)

    def test_both_panels_keep_all_resources_and_truthful_archive_capacity(self):
        resources = self.resources | producer.preconsumption_resource_paths(ROOT)
        self.assertEqual(len(resources), 34)
        for panel, paths in (('primary', self.paths),
                             ('replication', producer.select_paths(self.tracked, ROOT, 'replication'))):
            with self.subTest(panel=panel):
                self.assertTrue(resources <= paths)
                files = {name: dict(bytes=(ROOT / name).stat().st_size) for name in paths}
                envelope = producer.envelope(files)
                self.assertEqual(envelope['training_and_quality_input_archives_nested'], 0)
                self.assertEqual(envelope['retained_source_preconsumption_artifacts'], 2)
                self.assertEqual(envelope['retained_transport_original_archives'], 5)
                self.assertLessEqual(envelope['maximum_zip_bytes'], 128000000)
                score = self.directory / ('score-' + panel)
                source_root = score / 'source_producer'
                for name in paths:
                    target = source_root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, target)
                shutil.copytree(source_root, score / (panel + '_producer'))
                # Only a byte-envelope fixture, never an execution/fetch claim.
                (score / 'FETCH.json').write_bytes(b'{}')
                scoring = producer.capacity.producer_members(score, panel)
                archive_paths = sorted(prefix + '/' + name for prefix in ('source_producer', panel + '_producer')
                                       for name in paths if name.endswith('.zip'))
                self.assertEqual(scoring['historical_archive_paths'], archive_paths)
                self.assertEqual(scoring['historical_archives_nested'], 14)
                self.assertEqual(scoring['limit_bytes'], 128000000)
                self.assertLessEqual(scoring['maximum_zip_bytes'], scoring['limit_bytes'])

    def test_missing_nested_original_code_archive_is_rejected(self):
        path = self.captured / 'automation/english_v6_b21_fixed_mirror_failure/producer-original.zip'
        path.unlink()
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')
        self.assertIn(b'complete regular original transport resource', result.stderr)

    def test_mutated_original_probe_is_rejected(self):
        path = self.captured / (producer.TRANSPORT_PROBE + '/original/RESULT.json')
        path.write_bytes(path.read_bytes().replace(b'16777216', b'16777217', 1))
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')
        self.assertIn(b'unchanged original transport resource bytes', result.stderr)

    def test_symlink_to_uncaptured_original_is_rejected(self):
        path = self.captured / producer.TRANSPORT_PLAN
        path.unlink()
        path.symlink_to(ROOT / producer.TRANSPORT_PLAN)
        result = self.run_preflight()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')
        self.assertIn(b'complete regular original transport resource', result.stderr)

    def test_preflight_permanently_denies_scientific_import_network_process_and_writes(self):
        path = self.captured / 'automation/english_v6_b21_fixed_mirror_transport.py'
        original = path.read_bytes()
        sentinel = self.directory / 'must-not-exist'
        attempts = (
            'import numpy',
            'import socket; socket.socket()',
            'import subprocess; subprocess.run(["false"], check=False)',
            'open(' + repr(str(sentinel)) + ', "wb").write(b"forbidden")',
        )
        for statement in attempts:
            with self.subTest(statement=statement):
                path.write_bytes(original + ('\n' + statement + '\n').encode('utf-8'))
                result = self.run_preflight()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b'')
                self.assertIn(b'ValueError:', result.stderr)
                self.assertFalse(sentinel.exists())
        path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
