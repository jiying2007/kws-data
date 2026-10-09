import io
import json
import os
import stat
from unittest import mock
from pathlib import Path
import tempfile
import unittest
import zipfile
from verify_archive import digest, verify, checked_path, checked_inventory, stream_identity, MAX_OBJECT_BYTES

class TestArchive(unittest.TestCase):
    def fixture(self, root, member='evidence/data.bin'):
        payload = b'synthetic fixture, no model'
        key = 'objects/' + digest(payload)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            z.writestr(key, payload)
        blob = buf.getvalue()
        (root / 'part').write_bytes(blob)
        archive = {'parts': [{'path': 'part', 'bytes': len(blob), 'sha256': digest(blob)}], 'archive_bytes': len(blob), 'archive_sha256': digest(blob), 'members': 1, 'unique_objects': 1}
        row = {'path': member, 'object': key, 'published_bytes': len(payload), 'published_sha256': digest(payload), 'original_bytes': len(payload), 'original_sha256': digest(payload), 'transformations': []}
        (root / 'ARCHIVE.json').write_text(json.dumps(archive))
        (root / 'CATALOG.json').write_text(json.dumps({'members': [row]}))
        return row
    def test_valid_and_restore(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root)
            self.assertEqual(verify(root)['members'], 1)
            target = root / 'restored'; verify(root, target)
            self.assertEqual((target / 'evidence/data.bin').read_bytes(), b'synthetic fixture, no model')
            with self.assertRaises(FileExistsError): verify(root, target)
    def test_corrupted_part(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root); (root / 'part').write_bytes(b'bad')
            with self.assertRaises(ValueError): verify(root)
    def test_traversal(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root, '../escape')
            with self.assertRaises(ValueError): verify(root, root / 'restored')
            self.assertFalse((root / 'restored').exists())
    def test_undeclared_change(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); row = self.fixture(root); row['original_sha256'] = '0' * 64
            (root / 'CATALOG.json').write_text(json.dumps({'members': [row]}))
            with self.assertRaises(ValueError): verify(root)
    def test_duplicate_members(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); row = self.fixture(root)
            (root / 'CATALOG.json').write_text(json.dumps({'members': [row, row]}))
            with self.assertRaises(ValueError): verify(root)

class TestArchiveHardening(unittest.TestCase):
    fixture = TestArchive.fixture

    def rewrite(self, root, name, edit):
        path = root / name
        data = json.loads(path.read_text())
        edit(data)
        path.write_text(json.dumps(data))

    def replace_zip(self, root, writer):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            writer(z)
        data = buf.getvalue()
        (root / 'part').write_bytes(data)
        def edit(a):
            a['parts'][0].update(bytes=len(data), sha256=digest(data))
            a.update(archive_bytes=len(data), archive_sha256=digest(data))
        self.rewrite(root, 'ARCHIVE.json', edit)

    def test_noncanonical_and_windows_paths(self):
        bad = ['', '.', './a', 'a//b', 'a/./b', 'a/../b', '/a', 'a/',
               'C:/a', 'C:a', '//server/share', 'a\\b', 'a:b', 'a\x00b',
               'a?', 'a*', 'a<', 'a>', 'a|', 'a"', 'a\nb', 'a.', 'a ',
               'CON', 'aux.txt', 'dir/NUL.log', 'LPT1', 'com9.dat', 'COM¹.txt',
               'cafe\u0301/file']
        for name in bad:
            with self.subTest(name=name), self.assertRaises(ValueError):
                checked_path(name)

    def test_portable_alias_and_prefix_inventory(self):
        for names in [('a', 'a'), ('A', 'a'), ('A/x', 'a/y'), ('a', 'a/b'),
                      ('a/b', 'a'), ('a/B/c', 'a/b')]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                checked_inventory(names)
        checked_inventory(['a/b', 'a/c', 'z'])

    def test_alias_rejected_before_destination_creation(self):
        for other in ['evidence/./data.bin', 'evidence//data.bin',
                      'EVIDENCE/data.bin', 'evidence/data.bin/child', 'evidence']:
            with self.subTest(other=other), tempfile.TemporaryDirectory() as t:
                root = Path(t); row = self.fixture(root)
                row2 = dict(row, path=other)
                self.rewrite(root, 'CATALOG.json', lambda c: c.update(members=[row, row2]))
                self.rewrite(root, 'ARCHIVE.json', lambda a: a.update(members=2))
                with self.assertRaises(ValueError): verify(root, root / 'restored')
                self.assertFalse((root / 'restored').exists())

    def test_duplicate_parts(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root)
            self.rewrite(root, 'ARCHIVE.json', lambda a: a['parts'].append(a['parts'][0]))
            with self.assertRaises(ValueError): verify(root)

    def test_archive_hash(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root)
            self.rewrite(root, 'ARCHIVE.json', lambda a: a.update(archive_sha256='0' * 64))
            with self.assertRaises(ValueError): verify(root)

    def test_object_inventory_extra_missing_and_duplicate(self):
        for variant in ['extra', 'missing', 'duplicate']:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as t:
                root = Path(t); row = self.fixture(root)
                def writer(z):
                    if variant != 'missing': z.writestr(row['object'], b'synthetic fixture, no model')
                    if variant == 'extra': z.writestr('objects/' + '0' * 64, b'other')
                    if variant == 'duplicate':
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', UserWarning)
                            z.writestr(row['object'], b'synthetic fixture, no model')
                self.replace_zip(root, writer)
                with self.assertRaises(ValueError): verify(root)

    def test_object_corruption_with_rehashed_outer_archive(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); row = self.fixture(root)
            self.replace_zip(root, lambda z: z.writestr(row['object'], b'x' * row['published_bytes']))
            with self.assertRaises(ValueError): verify(root)

    def test_object_symlink_and_special_modes(self):
        for mode in [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFDIR]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as t:
                root = Path(t); row = self.fixture(root)
                info = zipfile.ZipInfo(row['object']); info.create_system = 3
                info.external_attr = (mode | 0o600) << 16
                self.replace_zip(root, lambda z: z.writestr(info, b'synthetic fixture, no model'))
                with self.assertRaises(ValueError): verify(root)

    def test_byte_limits_and_invalid_types(self):
        for value in [-1, True, 1.5, '1', MAX_OBJECT_BYTES + 1]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as t:
                root = Path(t); self.fixture(root)
                self.rewrite(root, 'CATALOG.json', lambda c: c['members'][0].update(published_bytes=value))
                with self.assertRaises(ValueError): verify(root)
        with self.assertRaises(ValueError): stream_identity(io.BytesIO(b'oversized'), 3)

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root)
            (root / 'ARCHIVE.json').write_text('{"parts":[],"parts":[]}')
            with self.assertRaises(ValueError): verify(root)

    def test_same_object_conflicting_sizes(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); row = self.fixture(root)
            second = dict(row, path='other.bin', published_bytes=row['published_bytes'] + 1)
            self.rewrite(root, 'CATALOG.json', lambda c: c.update(members=[row, second]))
            self.rewrite(root, 'ARCHIVE.json', lambda a: a.update(members=2))
            with self.assertRaises(ValueError): verify(root)

    def test_existing_file_and_symlink_destination(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root)
            file = root / 'file'; file.write_bytes(b'keep')
            link = root / 'link'; link.symlink_to(file)
            for target in [file, link]:
                with self.assertRaises(FileExistsError): verify(root, target)
            self.assertEqual(file.read_bytes(), b'keep')

    def test_restore_exclusive_leaf(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root)
            real_open = os.open
            def inject(path, flags, mode=0o777, *, dir_fd=None):
                if path == 'data.bin' and flags & os.O_CREAT:
                    fd = real_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                   0o600, dir_fd=dir_fd)
                    os.write(fd, b'keep'); os.close(fd)
                return real_open(path, flags, mode, dir_fd=dir_fd)
            with mock.patch('verify_archive.os.open', side_effect=inject):
                with self.assertRaises(FileExistsError): verify(root, root / 'restored')
            self.assertEqual((root / 'restored/evidence/data.bin').read_bytes(), b'keep')

    def test_restore_symlink_directory_injection(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); self.fixture(root)
            outside = root / 'outside'; outside.mkdir()
            real_mkdir = os.mkdir
            def inject(path, mode=0o777, *, dir_fd=None):
                if path == 'evidence':
                    os.symlink(str(outside), path, dir_fd=dir_fd)
                    raise FileExistsError(path)
                return real_mkdir(path, mode, dir_fd=dir_fd)
            with mock.patch('verify_archive.os.mkdir', side_effect=inject):
                with self.assertRaises(OSError): verify(root, root / 'restored')
            self.assertEqual(list(outside.iterdir()), [])

    def test_reused_object_restores_both_members(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); row = self.fixture(root)
            self.rewrite(root, 'CATALOG.json', lambda c: c.update(members=[row, dict(row, path='other.bin')]))
            self.rewrite(root, 'ARCHIVE.json', lambda a: a.update(members=2))
            verify(root, root / 'restored')
            self.assertEqual((root / 'restored/other.bin').read_bytes(), b'synthetic fixture, no model')

if __name__ == '__main__':
    unittest.main()
