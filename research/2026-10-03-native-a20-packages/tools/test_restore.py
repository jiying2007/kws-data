"""Invented byte-only negative tests for the outer transport. No scientific workload."""
import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

import restore as m


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'
        (self.bundle / 'chunks').mkdir(parents=True)
        self.limit = 256
        self.entries = [('public/kws-data/alpha.txt', b'invented alpha\n', stat.S_IFREG | 0o644),
                        ('public/kws-data/nested/beta.bin', bytes(range(127)), stat.S_IFREG | 0o644)]
        self.files = {n[len(m.PREFIX):]: {'path': n[len(m.PREFIX):], 'bytes': len(b), 'sha256': m.sha(b)}
                      for n, b, _ in self.entries}
        self.zip_bytes = self.make_zip(self.entries)
        self.parts = [self.make_part(self.zip_bytes)]

    def tearDown(self):
        self.temp.cleanup()

    def make_zip(self, entries):
        out = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED) as z:
                for name, data, mode in entries:
                    info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = mode << 16
                    z.writestr(info, data)
        return out.getvalue()

    def make_part(self, b, name='part.zip', count=None, payload=None):
        for old in (self.bundle / 'chunks').iterdir():
            old.unlink()
        chunks = []
        for start in range(0, len(b), self.limit):
            x = b[start:start + self.limit]
            h = m.sha(x)
            rel = 'chunks/' + h + '.bin'
            (self.bundle / rel).write_bytes(x)
            chunks.append({'ordinal': len(chunks), 'path': rel, 'size_bytes': len(x), 'sha256': h})
        return {'filename': name, 'size_bytes': len(b), 'sha256': m.sha(b),
                'member_count': len(self.entries) if count is None else count,
                'payload_bytes': sum(len(x[1]) for x in self.entries) if payload is None else payload,
                'chunks': chunks}

    def extract(self, parts=None, files=None):
        parts = self.parts if parts is None else parts
        m.reconstruct(self.bundle, self.root / 'packages', parts, self.limit)
        return m.extract_public(self.root / 'packages', self.root / 'out',
                                {'parts': parts, 'files': self.files if files is None else files})

    def test_exact_round_trip(self):
        result = self.extract()
        self.assertEqual(result['public_files'], 2)
        self.assertEqual((self.root / 'out/nested/beta.bin').read_bytes(), bytes(range(127)))

    def test_missing_chunk(self):
        (self.bundle / self.parts[0]['chunks'][0]['path']).unlink()
        with self.assertRaises(m.Stop): self.extract()

    def test_extra_chunk(self):
        (self.bundle / 'chunks/extra.bin').write_bytes(b'invented extra')
        with self.assertRaises(m.Stop): self.extract()

    def test_extra_chunk_directory(self):
        (self.bundle / 'chunks/extra').mkdir()
        with self.assertRaises(m.Stop): self.extract()

    def test_tampered_chunk(self):
        p = self.bundle / self.parts[0]['chunks'][0]['path']
        b = p.read_bytes(); p.write_bytes(bytes([b[0] ^ 1]) + b[1:])
        with self.assertRaises(m.Stop): self.extract()

    def test_wrong_chunk_size(self):
        self.parts[0]['chunks'][0]['size_bytes'] -= 1
        with self.assertRaises(m.Stop): self.extract()

    def test_reordered_chunks(self):
        self.parts[0]['chunks'].reverse()
        with self.assertRaises(m.Stop): self.extract()

    def test_reordered_and_renumbered(self):
        self.parts[0]['chunks'].reverse()
        for i, x in enumerate(self.parts[0]['chunks']): x['ordinal'] = i
        with self.assertRaises(m.Stop): self.extract()

    def test_duplicate_chunk(self):
        self.parts[0]['chunks'].append(copy.deepcopy(self.parts[0]['chunks'][0]))
        self.parts[0]['chunks'][-1]['ordinal'] = len(self.parts[0]['chunks']) - 1
        with self.assertRaises(m.Stop): self.extract()

    def test_non_content_addressed_chunk(self):
        self.parts[0]['chunks'][0]['path'] = 'chunks/wrong.bin'
        with self.assertRaises(m.Stop): self.extract()

    def test_wrong_package_hash(self):
        self.parts[0]['sha256'] = '0' * 64
        with self.assertRaises(m.Stop): self.extract()

    def test_wrong_package_size(self):
        self.parts[0]['size_bytes'] += 1
        with self.assertRaises(m.Stop): self.extract()

    def test_unsafe_chunk_path(self):
        self.parts[0]['chunks'][0]['path'] = '../escape'
        with self.assertRaises(m.Stop): self.extract()

    def test_existing_package_directory(self):
        p = self.root / 'packages'; p.mkdir(); (p / 'keep').write_text('untouched')
        with self.assertRaises(m.Stop): self.extract()
        self.assertEqual((p / 'keep').read_text(), 'untouched')

    def test_existing_extract_directory(self):
        p = self.root / 'out'; p.mkdir(); (p / 'keep').write_text('untouched')
        with self.assertRaises(m.Stop): self.extract()
        self.assertEqual((p / 'keep').read_text(), 'untouched')

    def test_symlink_chunk(self):
        p = self.bundle / self.parts[0]['chunks'][0]['path']; target = self.root / 'actual'
        p.rename(target); p.symlink_to(target)
        with self.assertRaises(m.Stop): self.extract()

    def test_symlink_ancestry(self):
        link = self.root / 'linked'; link.symlink_to(self.bundle, target_is_directory=True)
        with self.assertRaises(m.Stop): m.reconstruct(link, self.root / 'packages', self.parts, self.limit)

    def test_zip_path_traversal(self):
        bad = self.entries + [('public/kws-data/../escape', b'x', stat.S_IFREG | 0o644)]
        p = self.make_part(self.make_zip(bad), count=3, payload=sum(len(x[1]) for x in bad))
        with self.assertRaises(m.Stop): self.extract([p])

    def test_private_extra_member(self):
        bad = self.entries + [('private/notes.txt', b'x', stat.S_IFREG | 0o644)]
        p = self.make_part(self.make_zip(bad), count=3, payload=sum(len(x[1]) for x in bad))
        with self.assertRaises(m.Stop): self.extract([p])

    def test_duplicate_zip_member(self):
        bad = self.entries + [self.entries[0]]
        p = self.make_part(self.make_zip(bad), count=3, payload=sum(len(x[1]) for x in bad))
        with self.assertRaises(m.Stop): self.extract([p])

    def test_case_collision(self):
        bad = self.entries + [('public/kws-data/ALPHA.txt', b'x', stat.S_IFREG | 0o644)]
        p = self.make_part(self.make_zip(bad), count=3, payload=sum(len(x[1]) for x in bad))
        with self.assertRaises(m.Stop): self.extract([p])

    def test_zip_symlink(self):
        bad = [(self.entries[0][0], b'target', stat.S_IFLNK | 0o777)]
        p = self.make_part(self.make_zip(bad), count=1, payload=6)
        with self.assertRaises(m.Stop): self.extract([p])

    def test_missing_zip_member(self):
        p = self.make_part(self.make_zip(self.entries[:1]), count=1, payload=len(self.entries[0][1]))
        with self.assertRaises(m.Stop): self.extract([p])

    def test_member_hash_mismatch(self):
        fs = copy.deepcopy(self.files); fs['alpha.txt']['sha256'] = '0' * 64
        with self.assertRaises(m.Stop): self.extract(files=fs)

    def test_crc_failure(self):
        b = self.zip_bytes.replace(b'invented alpha\n', b'invented AlphA\n', 1)
        p = self.make_part(b)
        with self.assertRaises(zipfile.BadZipFile): self.extract([p])

    def test_duplicate_json(self):
        with self.assertRaises(m.Stop): m.strict_json(b'{"a":1,"a":2}')

    def test_nonfinite_json(self):
        with self.assertRaises(m.Stop): m.strict_json(b'{"a":NaN}')

    def test_bool_is_not_integer(self):
        with self.assertRaises(m.Stop): m.uint(True, 10)

    def test_windows_paths(self):
        for name in ['A:stream', 'dir\\x', 'AUX.txt', 'dir/COM1', 'x.', 'x ', '/root', 'a//b', '.git/a']:
            with self.subTest(name=name), self.assertRaises(m.Stop): m.path_name(name)

    def test_untrusted_root_manifest(self):
        (self.bundle / 'transport.json').write_text('{}')
        with self.assertRaises(m.Stop): m.policy(self.bundle)

    def test_exact_production_metadata(self):
        bundle = Path(m.__file__).resolve().parents[1]
        p = m.policy(bundle)
        self.assertEqual(len(p['files']), 2243)
        self.assertEqual(sum(len(r['chunks']) for r in p['parts']), 111)


if __name__ == '__main__':
    unittest.main()
