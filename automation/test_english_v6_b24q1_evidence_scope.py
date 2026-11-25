"""Small invented proofs only: no audio, network, model or native process."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).with_name('english_v6_b24q1_evidence_scope.py')
SPEC = importlib.util.spec_from_file_location('q1_scope_test_target', PATH)
t = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


class EvidenceScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'proof'
        (self.directory / 'api').mkdir(parents=True)
        self.data = {'original.zip': b'invented complete archive bytes',
                     'api/run.json': b'{"fixture_only":true}', 'empty.txt': b''}
        for name, raw in self.data.items():
            (self.directory / name).write_bytes(raw)
        self.files = {name: pin(raw) for name, raw in self.data.items()}
        self.reader = self.root / 'reader.py'
        self.reader.write_bytes(b'# invented immutable reader; never executed\n')
        self.reader_sha = sha(self.reader.read_bytes())
        self.lock = canonical({'fixture_only': True, 'run': 1})
        self.report = canonical({'fixture_only': True, 'full_original_proof': True,
                                 'scientific_execution_authorized': False})
        self.report_sha = sha(self.report)
        self.calls = 0

    def tearDown(self):
        self.assertIsNone(t._ACTIVE.get())
        self.tmp.cleanup()

    def execute(self):
        self.calls += 1
        return self.report

    def verify(self, **overrides):
        args = dict(kind='r1_source_stop', directory=self.directory,
                    files=self.files, lock_raw=self.lock, reader_path=self.reader,
                    reader_sha256=self.reader_sha, report_sha256=self.report_sha,
                    execute=self.execute)
        args.update(overrides)
        return t.verify_once(**args)

    def test_nested_scope_executes_once_and_destroys_results_on_exit(self):
        with t.scope():
            self.assertEqual(self.verify(), self.report)
            with t.scope():
                self.assertEqual(self.verify(), self.report)
            self.assertEqual(self.verify(), self.report)
            self.assertEqual(self.calls, 1)
        with t.scope():
            self.assertEqual(self.verify(), self.report)
        self.assertEqual(self.calls, 2)

    def test_unscoped_calls_always_run_complete_original_verifier(self):
        self.verify()
        self.verify()
        self.assertEqual(self.calls, 2)

    def test_scope_exception_clears_complete_results(self):
        with self.assertRaises(RuntimeError):
            with t.scope():
                self.verify()
                raise RuntimeError('fixture interruption')
        with t.scope():
            self.verify()
        self.assertEqual(self.calls, 2)

    def test_cross_directory_reuse_requires_every_file_and_reader_byte(self):
        other = self.root / 'identical_proof'
        shutil.copytree(self.directory, other)
        other_reader = self.root / 'identical_reader.py'
        shutil.copyfile(self.reader, other_reader)
        with t.scope():
            self.verify()
            self.verify(directory=other, reader_path=other_reader)
            self.assertEqual(self.calls, 1)
            (other / 'api/run.json').write_bytes(b'{"fixture_only":null}')
            with self.assertRaises(ValueError):
                self.verify(directory=other)
            self.assertEqual(self.calls, 1)

    def test_missing_file_rejected_before_callback_even_after_cache_hit(self):
        with t.scope():
            self.verify()
            (self.directory / 'api/run.json').unlink()
            with self.assertRaises(ValueError):
                self.verify()
        self.assertEqual(self.calls, 1)

    def test_extra_file_and_empty_directory_rejected(self):
        with t.scope():
            self.verify()
            for name, is_dir in (('extra.json', False), ('unregistered', True)):
                path = self.directory / name
                path.mkdir() if is_dir else path.write_bytes(b'extra')
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.verify()
                path.rmdir() if is_dir else path.unlink()

    def test_changed_bytes_fail_even_with_same_size_and_restored_mtime(self):
        path = self.directory / 'original.zip'
        old_stat = path.stat()
        with t.scope():
            self.verify()
            path.write_bytes(bytes([self.data['original.zip'][0] ^ 1]) + self.data['original.zip'][1:])
            os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                self.verify()
            path.write_bytes(self.data['original.zip'])
            self.verify()
        self.assertEqual(self.calls, 2, 'a failed reauthentication evicts the old result')

    def test_symlink_file_and_directory_rejected_even_for_identical_target(self):
        with t.scope():
            self.verify()
            path = self.directory / 'original.zip'
            target = self.root / 'archive_target'
            path.rename(target)
            path.symlink_to(target)
            with self.assertRaises(ValueError):
                self.verify()
            path.unlink()
            target.rename(path)
            target = self.root / 'api_target'
            (self.directory / 'api').rename(target)
            (self.directory / 'api').symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.verify()

    def test_symlink_root_and_ancestor_rejected_without_resolving_them_away(self):
        root_link = self.root / 'proof_link'
        root_link.symlink_to(self.directory, target_is_directory=True)
        ancestor_link = self.root / 'ancestor_link'
        ancestor_link.symlink_to(self.root, target_is_directory=True)
        for directory in (root_link, ancestor_link / 'proof'):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                self.verify(directory=directory)
        with self.assertRaises(ValueError):
            self.verify(directory=self.directory / '..' / 'proof')
        self.assertEqual(self.calls, 0)

    def test_reader_byte_mutation_symlink_and_ancestor_rejected(self):
        raw = self.reader.read_bytes()
        with t.scope():
            self.verify()
            self.reader.write_bytes(raw.replace(b'invented', b'modified'))
            with self.assertRaises(ValueError):
                self.verify()
            self.reader.write_bytes(raw)
            target = self.root / 'reader_target'
            self.reader.rename(target)
            self.reader.symlink_to(target)
            with self.assertRaises(ValueError):
                self.verify()
            ancestor = self.root / 'reader_parent_link'
            ancestor.symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.verify(reader_path=ancestor / 'reader_target')
        self.assertEqual(self.calls, 1)

    def test_known_kind_lock_reader_and_report_pins_are_distinct_identities(self):
        with t.scope():
            self.verify()
            self.verify(kind='q1_primary_quality')
            self.verify(lock_raw=canonical({'fixture_only': True, 'run': 2}))
            different = self.root / 'other_reader.py'
            different.write_bytes(b'# other original fixture reader\n')
            self.verify(reader_path=different, reader_sha256=sha(different.read_bytes()))
            report = canonical({'fixture_only': True, 'second_original_report': True})
            self.verify(report_sha256=sha(report), execute=lambda: report)
            self.assertEqual(self.calls, 4)
        with self.assertRaises(ValueError):
            self.verify(kind='model_execution')

    def test_repinned_evidence_bytes_require_a_new_original_verifier(self):
        with t.scope():
            self.verify()
            raw = b'new fully pinned fixture bytes'
            (self.directory / 'original.zip').write_bytes(raw)
            files = copy.deepcopy(self.files)
            files['original.zip'] = pin(raw)
            self.verify(files=files)
        self.assertEqual(self.calls, 2)

    def test_wrong_report_hash_and_callback_exception_never_cache(self):
        with t.scope():
            with self.assertRaisesRegex(ValueError, 'report SHA-256'):
                self.verify(report_sha256='0' * 64)
            self.verify()
            changed_lock = canonical({'fixture_only': True, 'run': 3})
            with self.assertRaises(RuntimeError):
                self.verify(lock_raw=changed_lock,
                            execute=mock.Mock(side_effect=RuntimeError('incomplete proof')))
            self.verify(lock_raw=changed_lock)
        self.assertEqual(self.calls, 3)

    def test_callback_mutating_inputs_or_reader_is_rejected_and_not_cached(self):
        raw_reader = self.reader.read_bytes()
        for member in ('original.zip', 'reader', 'extra'):
            with self.subTest(member=member), t.scope():
                def mutate():
                    self.calls += 1
                    if member == 'reader':
                        self.reader.write_bytes(b'changed reader')
                    elif member == 'extra':
                        (self.directory / 'extra').write_bytes(b'extra')
                    else:
                        (self.directory / member).write_bytes(b'changed')
                    return self.report
                with self.assertRaises(ValueError):
                    self.verify(execute=mutate)
                self.reader.write_bytes(raw_reader)
                (self.directory / 'original.zip').write_bytes(self.data['original.zip'])
                if (self.directory / 'extra').exists():
                    (self.directory / 'extra').unlink()
                self.verify()
        self.assertEqual(self.calls, 6)

    def test_mutable_caller_pin_map_cannot_change_snapshot_during_callback(self):
        with t.scope():
            def mutate():
                self.files['original.zip']['sha256'] = '0' * 64
                return self.report
            self.assertEqual(self.verify(execute=mutate), self.report)
            with self.assertRaises(ValueError):
                self.verify()

    def test_noncanonical_or_invalid_json_lock_and_report_rejected(self):
        invalid = (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}',
                   b'{"x":-Infinity}', b'{"x":1e10000}', b'{"x":-1e10000}',
                   b'{"x": 1}', b'{"x":1}\n', b'[]', b'null', b'\xff', b'')
        for raw in invalid:
            with self.subTest(role='lock', raw=raw), self.assertRaises(ValueError):
                self.verify(lock_raw=raw)
            with self.subTest(role='report', raw=raw), self.assertRaises(ValueError):
                self.verify(execute=lambda: raw, report_sha256=sha(raw))

    def test_mutable_report_buffers_and_lock_buffers_rejected(self):
        for raw in (bytearray(self.report), memoryview(self.report), self.report.decode()):
            with self.subTest(raw=type(raw)), self.assertRaises(ValueError):
                self.verify(execute=lambda: raw)
        with self.assertRaises(ValueError):
            self.verify(lock_raw=bytearray(self.lock))

    def test_bad_paths_pins_and_file_directory_collisions_rejected(self):
        for path in ('../escape', '/absolute', 'api//run.json', './original.zip',
                     'api/../original.zip', 'back\\slash', 'nul\x00', 'trailing/'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.verify(files={path: pin(b'')})
        for bad_pin in ({'bytes': True, 'sha256': '0' * 64},
                        {'bytes': -1, 'sha256': '0' * 64},
                        {'bytes': 0, 'sha256': 'A' * 64},
                        {'bytes': 0, 'sha256': '0' * 64, 'extra': False}):
            with self.subTest(pin=bad_pin), self.assertRaises(ValueError):
                self.verify(files={'file': bad_pin})
        with self.assertRaises(ValueError):
            self.verify(files={'a': pin(b''), 'a/b': pin(b'')})

    def test_special_member_rejected_without_opening_or_blocking(self):
        path = self.directory / 'fifo'
        os.mkfifo(path)
        files = dict(self.files, fifo=pin(b''))
        with self.assertRaises(ValueError):
            self.verify(files=files)

    def test_every_cache_lookup_reopens_and_hashes_all_whole_files_and_reader(self):
        original = t._file_hash
        with mock.patch.object(t, '_file_hash', wraps=original) as hashed, t.scope():
            self.verify()
            self.assertEqual(hashed.call_count, 2 * (len(self.files) + 1))
            self.verify()
            self.assertEqual(hashed.call_count, 3 * (len(self.files) + 1))

    def test_pure_byte_reuse_has_no_network_process_or_persistent_write(self):
        before = {p.relative_to(self.root).as_posix(): p.read_bytes()
                  for p in self.root.rglob('*') if p.is_file()}
        with mock.patch.object(socket, 'socket', side_effect=AssertionError('no network')), \
             mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('no process')), \
             t.scope():
            report = self.verify()
            self.verify()
        after = {p.relative_to(self.root).as_posix(): p.read_bytes()
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertIs(type(report), bytes)
        self.assertIs(json.loads(report)['scientific_execution_authorized'], False)


if __name__ == '__main__':
    unittest.main()
