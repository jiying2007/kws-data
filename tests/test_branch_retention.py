"""Invented-byte tests only; never import or execute archived experiments."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
