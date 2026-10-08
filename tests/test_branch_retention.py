"""Invented-byte tests only; never import or execute archived experiments."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('branch_retention', Path(__file__).resolve().parents[1] / 'tools/verify_branch_retention.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'archive').mkdir()
        raw = b'invented saved evidence\n'
        (self.root / 'archive/item.txt').write_bytes(raw)
        (self.root / 'README.md').write_text('new index\n')
        self.sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        row = {'path': 'archive/item.txt', 'mode': '100644', 'bytes': len(raw),
               'git_blob_sha1': self.sha, 'source_commit': 'a' * 40}
        self.manifest = {'schema': 'kws-public-branch-retention-v1', 'base_commit': 'a' * 40,
                         'retained_files': [row], 'baseline_files': [copy.deepcopy(row)],
                         'baseline_changes': [], 'integration_files': ['README.md'],
                         'expected_retained_files': 1, 'expected_tracked_files': 2}
        self.tree = {'archive/item.txt': ('100644', self.sha), 'README.md': ('100644', 'b' * 40)}

    def verify(self):
        return V.verify(self.root, self.manifest, self.tree)

    def test_valid(self):
        self.assertEqual(self.verify()['retained_files'], 1)

    def test_tampered_same_size(self):
        p = self.root / 'archive/item.txt'; p.write_bytes(b'x' * p.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'working byte identity'):
            self.verify()

    def test_missing(self):
        (self.root / 'archive/item.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'missing regular'):
            self.verify()

    def test_extra_git_file(self):
        self.tree['secret.bin'] = ('100644', 'c' * 40)
        with self.assertRaisesRegex(ValueError, 'closed tracked'):
            self.verify()

    def test_git_mode(self):
        self.tree['archive/item.txt'] = ('100755', self.sha)
        with self.assertRaisesRegex(ValueError, 'Git identity'):
            self.verify()

    def test_working_mode(self):
        (self.root / 'archive/item.txt').chmod(0o755)
        with self.assertRaisesRegex(ValueError, 'working mode'):
            self.verify()

    def test_blob_identity(self):
        self.tree['archive/item.txt'] = ('100644', 'c' * 40)
        with self.assertRaisesRegex(ValueError, 'Git identity'):
            self.verify()

    def test_symlink(self):
        p = self.root / 'archive/item.txt'; p.rename(self.root / 'outside.txt'); p.symlink_to('../outside.txt')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.verify()

    def test_symlink_parent(self):
        (self.root / 'archive').rename(self.root / 'other'); (self.root / 'archive').symlink_to('other')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.verify()

    def test_integration_symlink(self):
        p = self.root / 'README.md'; p.unlink(); p.symlink_to('archive/item.txt')
        with self.assertRaisesRegex(ValueError, 'integration symlink'):
            self.verify()

    def test_duplicate_path(self):
        self.manifest['retained_files'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate file'):
            self.verify()

    def test_bool_bytes(self):
        self.manifest['retained_files'][0]['bytes'] = True
        with self.assertRaisesRegex(ValueError, 'invalid byte'):
            self.verify()

    def test_path_traversal(self):
        for name in ('../escape', '/absolute', './dot', 'x//y', 'x/../y', '.git/config', 'x\\y', 'x\0y'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                V.safe_path(name)

    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            json.loads('{"x":1,"x":2}', object_pairs_hook=V.unique)

    def test_baseline_loss(self):
        self.manifest['baseline_files'][0]['git_blob_sha1'] = 'd' * 40
        with self.assertRaisesRegex(ValueError, 'baseline retention'):
            self.verify()

    def test_real_git_tree(self):
        def git(*args):
            return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.DEVNULL)
        git('init'); git('add', '.')
        git('-c', 'user.name=Offline test', '-c', 'user.email=offline@example.invalid', 'commit', '-m', 'invented fixture')
        self.assertEqual(V.git_tree(self.root)['archive/item.txt'], ('100644', self.sha))
        self.assertEqual(V.verify(self.root, self.manifest)['tracked_files'], 2)
        (self.root / 'private.txt').write_text('invented unapproved payload')
        git('add', '.'); git('-c', 'user.name=Offline test', '-c', 'user.email=offline@example.invalid', 'commit', '-m', 'extra fixture')
        with self.assertRaisesRegex(ValueError, 'closed tracked'):
            V.verify(self.root, self.manifest)


class AtomicExtensionTests(unittest.TestCase):
    """Only invented bytes, including invented maintenance source contents."""
    def setUp(self):
        RetentionTests.setUp(self)
        for name in sorted(V.ATOMIC_SOURCES):
            self.write_source(name, ('invented public fixture for ' + name + '\n').encode())
        self.manifest['integration_files'] += sorted(V.ATOMIC_EXISTING_SOURCES)
        self.manifest['expected_tracked_files'] += len(V.ATOMIC_EXISTING_SOURCES)
        self.sidecar = {'schema': 'kws-git-atomic-prune-public-source-v1',
                        'repository': 'jiying2007/kws-data', 'files': []}
        for name in sorted(V.ATOMIC_SOURCES):
            raw = (self.root / name).read_bytes()
            self.sidecar['files'].append({'path': name, 'mode': '100644',
                                          'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        self.save_sidecar()
        self.source = sorted(V.ATOMIC_NEW_SOURCES)[0]

    @staticmethod
    def blob(raw):
        return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()

    def write_source(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        self.tree[name] = ('100644', self.blob(raw))

    def save_sidecar(self, raw=None):
        if raw is None:
            raw = (json.dumps(self.sidecar, sort_keys=True) + '\n').encode()
        self.write_source(V.ATOMIC_SIDECAR, raw)

    def verify(self):
        return V.verify(self.root, self.manifest, self.tree)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.DEVNULL)

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'user.name=Offline test', '-c', 'user.email=offline@example.invalid',
                 'commit', '-m', 'invented fixture')

    def test_exact_extension_valid(self):
        result = self.verify()
        self.assertEqual(len(V.ATOMIC_ADDITIONS), 8)
        self.assertEqual(len(V.ATOMIC_SOURCES), 12)
        self.assertEqual(result['tracked_files'], self.manifest['expected_tracked_files'] + 8)
        self.assertEqual(result['retained_files'], 1)
        self.assertEqual(result['retained_bytes'], self.manifest['retained_files'][0]['bytes'])

    def test_extension_all_absent(self):
        for name in V.ATOMIC_ADDITIONS:
            del self.tree[name]
            (self.root / name).unlink()
        self.assertEqual(self.verify()['tracked_files'], self.manifest['expected_tracked_files'])

    def test_each_missing_addition_rejected(self):
        for name in V.ATOMIC_ADDITIONS:
            with self.subTest(name=name):
                identity = self.tree.pop(name)
                try:
                    with self.assertRaisesRegex(ValueError, 'incomplete atomic extension'):
                        self.verify()
                finally:
                    self.tree[name] = identity

    def test_sidecar_only_rejected(self):
        for name in V.ATOMIC_NEW_SOURCES:
            del self.tree[name]
        with self.assertRaisesRegex(ValueError, 'incomplete atomic extension'):
            self.verify()

    def test_unlisted_tracked_path_rejected(self):
        self.write_source('tools/git_atomic_unapproved.py', b'invented unapproved bytes')
        with self.assertRaisesRegex(ValueError, 'closed tracked'):
            self.verify()

    def test_self_declared_source_addition_rejected(self):
        raw = b'invented new source'
        self.write_source('tools/unreviewed.py', raw)
        self.sidecar['files'].append({'path': 'tools/unreviewed.py', 'mode': '100644',
                                     'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        self.save_sidecar()
        with self.assertRaisesRegex(ValueError, 'closed atomic source'):
            self.verify()

    def test_sidecar_cannot_reclassify_baseline_file(self):
        raw = (self.root / 'archive/item.txt').read_bytes()
        self.sidecar['files'][0] = {'path': 'archive/item.txt', 'mode': '100644',
                                    'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        self.save_sidecar()
        with self.assertRaisesRegex(ValueError, 'closed atomic source'):
            self.verify()

    def test_missing_source_row_rejected(self):
        self.sidecar['files'].pop()
        self.save_sidecar()
        with self.assertRaisesRegex(ValueError, 'closed atomic source'):
            self.verify()

    def test_duplicate_source_path_rejected(self):
        self.sidecar['files'].append(copy.deepcopy(self.sidecar['files'][0]))
        self.save_sidecar()
        with self.assertRaisesRegex(ValueError, 'duplicate atomic source'):
            self.verify()

    def test_duplicate_json_key_rejected(self):
        raw = json.dumps(self.sidecar).replace('"files":', '"files": [], "files":', 1).encode()
        self.save_sidecar(raw)
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            self.verify()

    def test_nonfinite_json_rejected(self):
        raw = json.dumps(self.sidecar).replace('"files":', '"unused": NaN, "files":', 1).encode()
        self.save_sidecar(raw)
        with self.assertRaisesRegex(ValueError, 'nonfinite JSON'):
            self.verify()

    def test_source_row_fields_rejected(self):
        for field in ('mode', 'sha256', 'bytes', 'path'):
            with self.subTest(field=field):
                row = self.sidecar['files'][0]
                old = row.pop(field)
                self.save_sidecar()
                try:
                    with self.assertRaisesRegex(ValueError, 'atomic source row fields'):
                        self.verify()
                finally:
                    row[field] = old
        self.sidecar['files'][0]['allow_extra'] = True
        self.save_sidecar()
        with self.assertRaisesRegex(ValueError, 'atomic source row fields'):
            self.verify()

    def test_invalid_source_row_values_rejected(self):
        for field, bad in (('bytes', True), ('bytes', -1), ('bytes', '12'),
                           ('sha256', 'g' * 64), ('sha256', 'A' * 64), ('sha256', None),
                           ('mode', '100755'), ('mode', '120000')):
            with self.subTest(field=field, value=bad):
                old = self.sidecar['files'][0][field]
                self.sidecar['files'][0][field] = bad
                self.save_sidecar()
                try:
                    with self.assertRaises(ValueError):
                        self.verify()
                finally:
                    self.sidecar['files'][0][field] = old

    def test_sidecar_wrong_identity_rejected(self):
        for field in ('schema', 'repository'):
            with self.subTest(field=field):
                old = self.sidecar[field]
                self.sidecar[field] = 'unreviewed'
                self.save_sidecar()
                try:
                    with self.assertRaisesRegex(ValueError, 'atomic sidecar identity'):
                        self.verify()
                finally:
                    self.sidecar[field] = old

    def test_sidecar_invalid_inventory_rejected(self):
        self.sidecar['files'] = {}
        self.save_sidecar()
        with self.assertRaisesRegex(ValueError, 'sidecar file inventory'):
            self.verify()

    def test_traversal_row_rejected(self):
        for name in ('../escape', '/absolute', './dot', 'x//y', 'x/../y', '.git/config', 'x\\y', 'x\0y'):
            with self.subTest(name=name):
                self.sidecar['files'][0]['path'] = name
                self.save_sidecar()
                with self.assertRaisesRegex(ValueError, 'unsafe path'):
                    self.verify()

    def test_extension_retained_overlap_rejected(self):
        row = copy.deepcopy(self.manifest['retained_files'][0])
        row['path'] = self.source
        self.manifest['retained_files'].append(row)
        self.manifest['expected_retained_files'] += 1
        self.manifest['expected_tracked_files'] += 1
        with self.assertRaisesRegex(ValueError, 'extension overlaps original'):
            self.verify()

    def test_extension_baseline_overlap_rejected(self):
        row = copy.deepcopy(self.manifest['retained_files'][0])
        row['path'] = self.source
        self.manifest['baseline_files'].append(row)
        self.manifest['baseline_changes'].append(self.source)
        with self.assertRaisesRegex(ValueError, 'extension overlaps original'):
            self.verify()

    def test_extension_integration_overlap_rejected(self):
        self.manifest['integration_files'].append(self.source)
        self.manifest['expected_tracked_files'] += 1
        with self.assertRaisesRegex(ValueError, 'extension overlaps original'):
            self.verify()

    def test_source_baseline_overlap_rejected(self):
        name = sorted(V.ATOMIC_EXISTING_SOURCES)[0]
        row = copy.deepcopy(self.manifest['retained_files'][0])
        row['path'] = name
        self.manifest['baseline_files'].append(row)
        self.manifest['baseline_changes'].append(name)
        with self.assertRaisesRegex(ValueError, 'source overlaps retained or baseline'):
            self.verify()

    def test_missing_original_integration_membership_rejected(self):
        self.manifest['integration_files'].remove(sorted(V.ATOMIC_EXISTING_SOURCES)[0])
        self.manifest['expected_tracked_files'] -= 1
        with self.assertRaisesRegex(ValueError, 'not original integration'):
            self.verify()

    def test_changed_retained_baseline_rejected(self):
        raw = b'invented changed retained baseline'
        self.write_source('archive/item.txt', raw)
        self.manifest['retained_files'][0].update(bytes=len(raw), git_blob_sha1=self.blob(raw))
        with self.assertRaisesRegex(ValueError, 'baseline retention mismatch'):
            self.verify()

    def test_all_twelve_source_hashes_enforced(self):
        for row in self.sidecar['files']:
            with self.subTest(name=row['path']):
                old = row['sha256']
                row['sha256'] = '0' * 64
                self.save_sidecar()
                try:
                    with self.assertRaisesRegex(ValueError, 'extension SHA-256 identity'):
                        self.verify()
                finally:
                    row['sha256'] = old

    def test_all_twelve_source_sizes_enforced(self):
        for row in self.sidecar['files']:
            with self.subTest(name=row['path']):
                row['bytes'] += 1
                self.save_sidecar()
                try:
                    with self.assertRaisesRegex(ValueError, 'extension byte count'):
                        self.verify()
                finally:
                    row['bytes'] -= 1

    def test_source_git_mode_rejected(self):
        for name in V.ATOMIC_SOURCES | {V.ATOMIC_SIDECAR}:
            with self.subTest(name=name):
                old = self.tree[name]
                self.tree[name] = ('100755', old[1])
                try:
                    with self.assertRaisesRegex(ValueError, 'extension Git mode'):
                        self.verify()
                finally:
                    self.tree[name] = old

    def test_source_working_mode_rejected(self):
        for name in V.ATOMIC_SOURCES | {V.ATOMIC_SIDECAR}:
            with self.subTest(name=name):
                path = self.root / name
                path.chmod(0o755)
                try:
                    with self.assertRaisesRegex(ValueError, 'extension working mode'):
                        self.verify()
                finally:
                    path.chmod(0o644)

    def test_git_blob_identity_rejected(self):
        for name in (self.source, V.ATOMIC_SIDECAR):
            with self.subTest(name=name):
                old = self.tree[name]
                self.tree[name] = ('100644', 'e' * 40)
                try:
                    with self.assertRaisesRegex(ValueError, 'extension committed byte identity'):
                        self.verify()
                finally:
                    self.tree[name] = old

    def test_same_size_working_tamper_rejected(self):
        for name in (self.source, V.ATOMIC_SIDECAR):
            with self.subTest(name=name):
                path = self.root / name
                raw = path.read_bytes()
                path.write_bytes(b'x' * len(raw))
                try:
                    with self.assertRaisesRegex(ValueError, 'extension committed byte identity'):
                        self.verify()
                finally:
                    path.write_bytes(raw)

    def test_sidecar_size_tamper_rejected(self):
        path = self.root / V.ATOMIC_SIDECAR
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'extension committed byte identity'):
            self.verify()

    def test_oversized_committed_sidecar_rejected(self):
        raw = json.dumps(self.sidecar).encode().ljust(V.SIDECAR_MAX_BYTES + 1, b' ')
        self.save_sidecar(raw)
        self.assertEqual(self.tree[V.ATOMIC_SIDECAR][1], self.blob(raw))
        with self.assertRaisesRegex(ValueError, 'extension size limit'):
            self.verify()

    def test_missing_working_files_rejected(self):
        for name in (self.source, V.ATOMIC_SIDECAR):
            with self.subTest(name=name):
                path = self.root / name
                raw = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaisesRegex(ValueError, 'missing extension regular'):
                        self.verify()
                finally:
                    path.write_bytes(raw)

    def test_symlink_files_rejected(self):
        for name in (self.source, V.ATOMIC_SIDECAR):
            with self.subTest(name=name):
                path = self.root / name
                raw = path.read_bytes()
                path.unlink()
                path.symlink_to(self.root / 'archive/item.txt')
                try:
                    with self.assertRaisesRegex(ValueError, 'extension symlink'):
                        self.verify()
                finally:
                    path.unlink()
                    path.write_bytes(raw)

    def test_symlink_parent_rejected(self):
        parent = self.root / 'research'
        parent.rename(self.root / 'research-real')
        parent.symlink_to('research-real')
        with self.assertRaisesRegex(ValueError, 'extension symlink'):
            self.verify()

    def test_real_git_tree_with_extension(self):
        self.git('init')
        self.commit()
        result = V.verify(self.root, self.manifest)
        self.assertEqual(result['tracked_files'], self.manifest['expected_tracked_files'] + 8)
        self.assertEqual(V.git_tree(self.root)[self.source], self.tree[self.source])
        self.write_source('unreviewed.txt', b'invented extra')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'closed tracked'):
            V.verify(self.root, self.manifest)

    def test_real_git_retained_mode_rejected(self):
        self.git('init')
        self.commit()
        self.git('update-index', '--chmod=+x', 'archive/item.txt')
        self.git('-c', 'user.name=Offline test', '-c', 'user.email=offline@example.invalid',
                 'commit', '-m', 'invented mode change')
        with self.assertRaisesRegex(ValueError, 'Git identity mismatch'):
            V.verify(self.root, self.manifest)

    def test_frozen_manifest_identity_rejected(self):
        raw = json.dumps(self.manifest).encode()
        self.write_source(V.MANIFEST, raw)
        with self.assertRaisesRegex(ValueError, 'frozen retention manifest Git identity'):
            V.frozen_manifest(self.root, self.tree)

    def test_frozen_manifest_invented_pinned_fixture(self):
        raw = json.dumps(self.manifest).encode()
        self.write_source(V.MANIFEST, raw)
        # No real retained evidence is read; patch only the immutable fixture pin.
        with mock.patch.object(V, 'MANIFEST_BLOB', self.blob(raw)), \
             mock.patch.object(V, 'MANIFEST_BYTES', len(raw)):
            self.assertEqual(V.frozen_manifest(self.root, self.tree), self.manifest)
            (self.root / V.MANIFEST).write_bytes(b'x' * len(raw))
            with self.assertRaisesRegex(ValueError, 'committed byte identity'):
                V.frozen_manifest(self.root, self.tree)


if __name__ == '__main__':
    unittest.main()
