"""Invented-byte rejection contracts and an optional exact saved-code fixture.

No original audio/model call is performed. Set B27_ORIGINAL_PRODUCER_ZIP to the
already-downloaded original artifact for the isolated two-extraction fixture.
The production adapter authenticates the entire original ZIP itself.
"""
from __future__ import annotations

import copy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_primary_no_promotion_readback as subject


def original_lock():
    return dict(schema='english-v6-b27-original-quality-lock-v1', panel='primary',
        **{role: dict(id=pin[0], bytes=pin[1], sha256=pin[2], run_id=subject.RUN_ID,
                     run_head=subject.RUN_HEAD) for role, pin in subject.ARTIFACTS.items()},
        **subject.ORIGINAL_HASHES)


def invented_report():
    """A shape fixture only; never published or treated as original evidence."""
    lock = original_lock()
    report = dict(schema='english-v6-b27-verified-original-quality-v1', panel='primary', panel_gate='NO_PROMOTION',
        original_lock_sha256=subject.sha(subject.canonical(lock)),
        original_result_sha256=subject.ORIGINAL_HASHES['original_result_sha256'],
        external_proof_binding_sha256=subject.ORIGINAL_HASHES['external_proof_binding_sha256'],
        model_pt_sha256={'challenger': subject.CANDIDATE_PT}, actual_forwards=996, verified_frame_count=178776,
        all_original_arrays_verified=True, original_source_admission_recomputed=True,
        original_producer_git_verified=True, all_original_training_proof_verified=True,
        complete_original_Q1_failure_verified=True, old_stops_and_reserves_preserved=True,
        replication_open_allowed=False, candidate_engineering_validation_allowed=False,
        automatic_next_assessment_allowed=False, promotion_allowed=False, product_released=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0)
    return lock, report


class CapturedImports(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='b27-rooted-contract-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'captured'
        (self.root / 'automation').mkdir(parents=True)
        self.name = 'english_v6_root_scope_fixture'
        self.path = self.root / 'automation' / (self.name + '.py')
        self.body = b'ORIGIN = "captured-only"\n'
        self.path.write_bytes(self.body)
        self.pins = {'automation/' + self.name + '.py': dict(bytes=len(self.body), sha256=subject.sha(self.body))}
        self.addCleanup(sys.modules.pop, self.name, None)

    def test_exact_origin_precedes_a_second_root_and_scope_restores_namespace(self):
        fallback = Path(self.temporary.name) / 'checkout'
        fallback.mkdir()
        (fallback / (self.name + '.py')).write_bytes(b'raise RuntimeError("fallback executed")\n')
        before_path, before_meta = list(sys.path), list(sys.meta_path)
        with subject.captured_imports(self.root, self.pins) as finder:
            sys.path.insert(0, str(fallback))
            module = importlib.import_module(self.name)
            self.assertEqual(module.ORIGIN, 'captured-only')
            self.assertEqual(Path(module.__file__).resolve(), self.path)
            self.assertEqual(finder.loaded[self.name], self.pins['automation/' + self.name + '.py'])
        self.assertNotIn(self.name, sys.modules)
        self.assertEqual(sys.path, before_path)
        self.assertEqual(sys.meta_path, before_meta)

    def test_missing_captured_dependency_never_falls_back_to_same_bytes(self):
        fallback = Path(self.temporary.name) / 'checkout'
        fallback.mkdir()
        (fallback / (self.name + '.py')).write_bytes(self.body)
        with subject.captured_imports(self.root, self.pins):
            sys.path.insert(0, str(fallback))
            self.path.unlink()
            with self.assertRaises(ValueError):
                importlib.import_module(self.name)
        self.assertNotIn(self.name, sys.modules)

    def test_mutated_captured_dependency_is_rejected_before_execution(self):
        with subject.captured_imports(self.root, self.pins):
            self.path.write_bytes(b'raise RuntimeError("mutation executed")\n')
            with self.assertRaisesRegex(ValueError, 'unchanged complete captured'):
                importlib.import_module(self.name)

    def test_cached_other_root_is_rejected_without_transferring_context(self):
        module = ModuleType(self.name)
        module.__file__ = '/untrusted/checkout/' + self.name + '.py'
        sys.modules[self.name] = module
        with self.assertRaisesRegex(ValueError, 'fresh isolated process'):
            with subject.captured_imports(self.root, self.pins):
                self.fail('mixed namespace accepted')
        self.assertIs(sys.modules[self.name], module)

    def test_symlinked_module_is_rejected(self):
        target = Path(self.temporary.name) / 'outside.py'
        target.write_bytes(self.body)
        self.path.unlink()
        self.path.symlink_to(target)
        with self.assertRaises(ValueError):
            with subject.captured_imports(self.root, self.pins):
                self.fail('symlink accepted')

    def test_uncaptured_project_name_has_no_checkout_fallback(self):
        fallback = Path(self.temporary.name) / 'checkout'
        fallback.mkdir()
        missing = 'english_v6_not_in_original_snapshot'
        (fallback / (missing + '.py')).write_bytes(self.body)
        with subject.captured_imports(self.root, self.pins):
            sys.path.insert(0, str(fallback))
            with self.assertRaisesRegex(ModuleNotFoundError, 'no fallback'):
                importlib.import_module(missing)
        self.assertNotIn(missing, sys.modules)

    def package(self):
        bodies = {'kws_data_contract/__init__.py': b'"""Captured package fixture."""\n',
                  'kws_data_contract/identity.py': b'ORIGIN = "captured-package"\n',
                  'kws_data_contract/reference_text.py': b'from .identity import ORIGIN\n'}
        for relative, body in bodies.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            self.pins[relative] = dict(bytes=len(body), sha256=subject.sha(body))

    def test_package_and_relative_import_origins_are_captured_and_recorded(self):
        self.package()
        fallback = Path(self.temporary.name) / 'checkout'
        package = fallback / 'kws_data_contract'
        package.mkdir(parents=True)
        (package / '__init__.py').write_bytes(b'raise RuntimeError("ambient package executed")\n')
        with subject.captured_imports(self.root, self.pins) as finder:
            sys.path.insert(0, str(fallback))
            module = importlib.import_module('kws_data_contract.reference_text')
            self.assertEqual(module.ORIGIN, 'captured-package')
            imported, paths = finder.verified_imports()
            for name in ('kws_data_contract', 'kws_data_contract.identity', 'kws_data_contract.reference_text'):
                self.assertEqual(imported[name], self.pins[paths[name]])
                self.assertEqual(Path(sys.modules[name].__file__).resolve(), self.root / paths[name])
            sys.modules['kws_data_contract'].__path__.append(str(package))
            with self.assertRaisesRegex(ValueError, 'one exact original search root'):
                finder.verified_imports()
        self.assertFalse(any(name.startswith('kws_data_contract') for name in sys.modules))

    def test_missing_package_member_and_uncaptured_member_cannot_fall_back(self):
        self.package()
        fallback = Path(self.temporary.name) / 'checkout'
        fallback.mkdir()
        (fallback / 'identity.py').write_bytes((self.root / 'kws_data_contract/identity.py').read_bytes())
        (fallback / 'uncaptured.py').write_bytes(b'raise RuntimeError("fallback executed")\n')
        with subject.captured_imports(self.root, self.pins):
            package = importlib.import_module('kws_data_contract')
            package.__path__.insert(0, str(fallback))
            (self.root / 'kws_data_contract/identity.py').unlink()
            with self.assertRaises(ValueError):
                importlib.import_module('kws_data_contract.identity')
            with self.assertRaisesRegex(ModuleNotFoundError, 'no fallback'):
                importlib.import_module('kws_data_contract.uncaptured')

    def test_cached_package_descendant_is_rejected_before_entering_scope(self):
        self.package()
        name = 'kws_data_contract.uncaptured'
        module = ModuleType(name)
        self.addCleanup(sys.modules.pop, name, None)
        sys.modules[name] = module
        with self.assertRaisesRegex(ValueError, 'fresh isolated process'):
            with subject.captured_imports(self.root, self.pins):
                self.fail('cached package namespace accepted')
        self.assertIs(sys.modules[name], module)

    def test_both_original_research_script_import_names_are_captured(self):
        bodies = {'campaign': b'ORIGIN = "captured-research"\n',
                  'runner': b'import campaign as c\n'}
        for name, body in bodies.items():
            relative = subject.RESEARCH_MODULES[name]
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            self.pins[relative] = dict(bytes=len(body), sha256=subject.sha(body))
        with subject.captured_imports(self.root, self.pins) as finder:
            runner = importlib.import_module('runner')
            self.assertEqual(runner.c.ORIGIN, 'captured-research')
            imported, paths = finder.verified_imports()
            self.assertEqual(paths, subject.RESEARCH_MODULES)
            self.assertEqual(set(imported), {'campaign', 'runner'})
        self.assertNotIn('campaign', sys.modules)
        self.assertNotIn('runner', sys.modules)

    def test_nested_module_requires_complete_captured_package_ancestors(self):
        path = self.root / 'kws_data_contract/identity.py'
        path.parent.mkdir()
        path.write_bytes(self.body)
        self.pins['kws_data_contract/identity.py'] = dict(bytes=len(self.body), sha256=subject.sha(self.body))
        with self.assertRaisesRegex(ValueError, 'authenticated initializer'):
            subject.CapturedModuleFinder(self.root, self.pins)

    def test_bootstrap_blocks_model_network_and_children(self):
        import socket
        with subject.bootstrap_offline():
            for name in ('torch', 'numpy', 'requests'):
                with self.assertRaises(ValueError):
                    __import__(name)
            with self.assertRaises(ValueError):
                socket.socket()
            with self.assertRaises(ValueError):
                subprocess.Popen([sys.executable, '-c', 'raise SystemExit(0)'])


class FixedFailureEvidence(unittest.TestCase):
    def test_lock_only_accepts_the_exact_completed_original(self):
        self.assertEqual(subject.exact_lock(original_lock()), original_lock())
        for path, value in ((('panel',), 'replication'), (('result', 'id'), 15036520815),
                (('result', 'id'), 15036520814.0),
                (('producer', 'sha256'), '0' * 64), (('result', 'run_head'), 'a' * 40),
                (('original_result_sha256',), '0' * 64)):
            lock = original_lock()
            target = lock
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                subject.exact_lock(lock)

    def test_summary_or_pass_cannot_substitute_for_full_original_failure(self):
        lock, report = invented_report()
        subject.check_original_report(report, lock)
        for key, value in (('panel_gate', 'PASS'), ('all_original_training_proof_verified', False),
                ('all_original_arrays_verified', False), ('original_source_admission_recomputed', False),
                ('complete_original_Q1_failure_verified', False), ('replication_open_allowed', True),
                ('candidate_engineering_validation_allowed', True), ('new_model_calls', 1),
                ('actual_forwards', 995), ('verified_frame_count', 178775)):
            changed = copy.deepcopy(report)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.check_original_report(changed, lock)

    def test_wrong_whole_artifact_fails_before_extraction_or_project_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / 'producer.zip'
            raw.write_bytes(b'invented-not-an-original-zip')
            target = Path(temporary) / 'captured'
            with self.assertRaisesRegex(ValueError, 'whole original artifact'):
                subject.extract_original_root(raw, target)
            self.assertFalse(target.exists())

    def test_invalid_lock_fails_before_child_or_original_file_access(self):
        lock = original_lock()
        lock['schema'] = 'invented'
        with mock.patch.object(subprocess, 'check_output', side_effect=AssertionError('child started')):
            with self.assertRaises(ValueError):
                subject.verify_original(Path('/does-not-exist'), lock, Path('/does-not-exist'))

    @unittest.skipUnless(os.environ.get('B27_ORIGINAL_PRODUCER_ZIP'), 'exact downloaded original fixture not supplied')
    def test_actual_captured_original_reproduces_then_repairs_without_new_science(self):
        path = Path(os.environ['B27_ORIGINAL_PRODUCER_ZIP']).resolve()
        flags = ['-' + ('O' * sys.flags.optimize)] if sys.flags.optimize else []
        outputs = []
        for mode in ('--probe-original', '--probe-repaired'):
            raw = subprocess.check_output([sys.executable, *flags, '-I', '-S', '-B', str(Path(subject.__file__).resolve()), mode, str(path)])
            outputs.append(json.loads(raw))
        self.assertEqual(outputs[0]['reproduced_original_failure'], 'exact local original implementation')
        self.assertFalse(outputs[0]['repaired'])
        repaired = outputs[1]
        self.assertTrue(repaired['repaired'])
        self.assertEqual(repaired['captured_files'], 479)
        paths, modules = repaired['imported_original_module_paths'], repaired['imported_original_modules']
        self.assertEqual(set(paths), set(modules))
        self.assertEqual(paths['kws_data_contract'], 'kws_data_contract/__init__.py')
        self.assertEqual(paths['kws_data_contract.identity'], 'kws_data_contract/identity.py')
        self.assertEqual(modules['kws_data_contract']['sha256'],
                         '3b3c0ef178e7115b2206e895c19128603272c0d3ad8a22eade280d372f884608')
        self.assertEqual(modules['kws_data_contract.identity']['sha256'],
                         '7cb39db2c8fe07e6fada96735bae526c2f442775fc8e4786940add80abbba3e8')
        self.assertTrue(repaired['original_protocol_passed'])
        self.assertTrue(repaired['original_producer_contract_passed'])
        self.assertTrue(repaired['original_helpers_and_source_policy_passed'])
        self.assertTrue(repaired['single_original_profile'])
        self.assertFalse(repaired['proof_context_created'])
        self.assertEqual(repaired['original_reader_sha256'], subject.READER_SHA)
        for report in outputs:
            self.assertEqual([report[key] for key in ('new_model_calls', 'new_source_calls', 'new_observer_calls')], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
