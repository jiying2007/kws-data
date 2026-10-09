import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from verify_archive import digest, verify

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

if __name__ == '__main__':
    unittest.main()
