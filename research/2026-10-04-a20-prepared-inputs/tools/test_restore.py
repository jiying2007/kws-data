"""Invented byte-only rejection tests; no scientific workload is executed."""
import copy
import gzip
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
import zlib
from unittest import mock

import restore as m


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bundle = self.root / 'bundle'
        (self.bundle / 'success/chunks').mkdir(parents=True)
        self.entries = [('tiny.f32le', struct.pack('<fff', 0.0, 1.25, -3.0), None)]
        self.data, self.archive = self.make(self.entries)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, entries, chunk_size=96):
        records = [{'path': n, 'bytes': len(b), 'sha256': m.sha(b)} for n, b, _ in entries]
        manifest = json.dumps({'members': sorted(records, key=lambda x: x['path'])}).encode()
        all_entries = sorted(entries + [('artifact-manifest.json', manifest, None)])
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode='w', format=tarfile.USTAR_FORMAT) as t:
            for name, payload, kind in all_entries:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                if kind:
                    info.type = kind
                    info.linkname = '../../escape'
                t.addfile(info, io.BytesIO(payload))
        raw = out.getvalue()
        data = gzip.compress(raw, mtime=0)
        archive = {'id': 'success', 'bytes': len(data), 'sha256': m.sha(data),
                   'tar_bytes': len(raw), 'tar_sha256': m.sha(raw),
                   'member_count': len(all_entries), 'payload_bytes': sum(len(b) for _, b, _ in all_entries),
                   'members': [{'path': n, 'bytes': len(b), 'sha256': m.sha(b)} for n, b, _ in all_entries]}
        self.chunk(data, archive, chunk_size)
        return data, archive

    def chunk(self, data, archive, size=96):
        for p in (self.bundle / 'success/chunks').iterdir():
            p.unlink()
        archive['chunks'] = []
        for offset in range(0, len(data), size):
            b = data[offset:offset+size]
            path = 'success/chunks/' + m.sha(b) + '.bin'
            (self.bundle / path).write_bytes(b)
            archive['chunks'].append({'ordinal': len(archive['chunks']), 'offset': offset,
                                      'path': path, 'bytes': len(b), 'sha256': m.sha(b)})

    def run_restore(self):
        data = m.reconstruct(self.bundle, self.archive)
        return m.inspect_archive(data, self.archive)

    def test_exact_reassembly(self):
        result = self.run_restore()
        self.assertEqual(result['tiny.f32le'], self.entries[0][1])

    def test_missing_chunk(self):
        (self.bundle / self.archive['chunks'][0]['path']).unlink()
        with self.assertRaises(m.Stop): self.run_restore()

    def test_tampered_chunk(self):
        p = self.bundle / self.archive['chunks'][0]['path']
        b = p.read_bytes(); p.write_bytes(bytes([b[0] ^ 1]) + b[1:])
        with self.assertRaises(m.Stop): self.run_restore()

    def test_extra_chunk(self):
        (self.bundle / 'success/chunks/extra.bin').write_bytes(b'invented')
        with self.assertRaises(m.Stop): self.run_restore()

    def test_reordered_chunk(self):
        self.archive['chunks'].reverse()
        with self.assertRaises(m.Stop): self.run_restore()

    def test_reordered_and_renumbered(self):
        self.archive['chunks'].reverse()
        offset = 0
        for i, c in enumerate(self.archive['chunks']):
            c['ordinal'] = i; c['offset'] = offset; offset += c['bytes']
        with self.assertRaises(m.Stop): self.run_restore()

    def test_duplicate_chunk(self):
        self.archive['chunks'].append(copy.deepcopy(self.archive['chunks'][0]))
        with self.assertRaises(m.Stop): self.run_restore()

    def test_oversize_chunk(self):
        self.archive['chunks'][0]['bytes'] = m.CHUNK_LIMIT + 1
        with self.assertRaises(m.Stop): self.run_restore()

    def test_chunk_path_traversal(self):
        self.archive['chunks'][0]['path'] = '../escape.bin'
        with self.assertRaises(m.Stop): self.run_restore()

    def test_chunk_symlink(self):
        p = self.bundle / self.archive['chunks'][0]['path']
        b = p.read_bytes(); p.unlink(); target = self.root / 'outside'; target.write_bytes(b)
        p.symlink_to(target)
        with self.assertRaises(m.Stop): self.run_restore()

    def test_chunk_parent_symlink(self):
        q = self.bundle / 'success/chunks'; q.rename(self.root / 'outside')
        q.symlink_to(self.root / 'outside', target_is_directory=True)
        with self.assertRaises(m.Stop): self.run_restore()

    def test_wrong_archive_hash(self):
        self.archive['sha256'] = '0' * 64
        with self.assertRaises(m.Stop): self.run_restore()

    def test_wrong_tar_hash(self):
        self.archive['tar_sha256'] = '0' * 64
        with self.assertRaises(m.Stop): self.run_restore()

    def test_missing_member(self):
        self.archive['members'].append({'path': 'missing.json', 'bytes': 2, 'sha256': m.sha(b'{}')})
        self.archive['member_count'] += 1
        with self.assertRaises(m.Stop): self.run_restore()

    def test_oversize_member(self):
        self.archive['members'][0]['bytes'] = m.CHUNK_LIMIT + 1
        with self.assertRaises(m.Stop): self.run_restore()

    def test_member_path_traversal(self):
        for name in ['../escape.json', '/escape.json', 'a/../../escape.json', 'a\\b.json', 'C:escape.json', '.git/config', 'NUL.json']:
            with self.subTest(name=name):
                data, archive = self.make([(name, b'{}', None)])
                with self.assertRaises(m.Stop): m.inspect_archive(data, archive)
        self.assertFalse((self.root / 'escape.json').exists())

    def test_member_symlink_and_hardlink(self):
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            data, archive = self.make([('link.json', b'{}', kind)])
            with self.assertRaises(m.Stop): m.inspect_archive(data, archive)

    def test_duplicate_member(self):
        data, archive = self.make([('same.json', b'{}', None), ('same.json', b'{}', None)])
        with self.assertRaises(m.Stop): m.inspect_archive(data, archive)

    def test_casefold_alias(self):
        data, archive = self.make([('same.json', b'{}', None), ('SAME.json', b'{}', None)])
        with self.assertRaises(m.Stop): m.inspect_archive(data, archive)

    def test_nonfinite_float(self):
        data, archive = self.make([('tiny.f32le', struct.pack('<f', float('nan')), None)])
        with self.assertRaises(m.Stop): m.inspect_archive(data, archive)

    def test_bad_float_length(self):
        data, archive = self.make([('tiny.f32le', b'123', None)])
        with self.assertRaises(m.Stop): m.inspect_archive(data, archive)

    def test_duplicate_json(self):
        with self.assertRaises(m.Stop): m.read_json(b'{"x":1,"x":2}')

    def test_nonfinite_json(self):
        with self.assertRaises(m.Stop): m.read_json(b'{"x":NaN}')

    def test_json_exponent_overflow(self):
        with self.assertRaises(m.Stop): m.read_json(b'{"x":1e400}')

    def test_nested_json_exponent_overflow(self):
        for data in [b'{"nested":[{"value":1e999}]}', b'{"nested":[{"value":-1e999}]}']:
            with self.subTest(data=data):
                with self.assertRaises(m.Stop): m.read_json(data)

    def test_oversize_expansion(self):
        self.archive['tar_bytes'] -= 1
        with self.assertRaises(m.Stop): self.run_restore()

    def test_oversize_tar_bound(self):
        self.archive['tar_bytes'] = m.TAR_LIMIT + 1
        with self.assertRaises(m.Stop): self.run_restore()

    def test_truncated_gzip(self):
        data = self.data[:-5]; a = copy.deepcopy(self.archive)
        a['bytes'] = len(data); a['sha256'] = m.sha(data)
        with self.assertRaises(m.Stop): m.inspect_archive(data, a)

    def test_concatenated_gzip(self):
        data = self.data + gzip.compress(b'extra'); a = copy.deepcopy(self.archive)
        a['bytes'] = len(data); a['sha256'] = m.sha(data)
        with self.assertRaises(m.Stop): m.inspect_archive(data, a)

    def test_unknown_file_type(self):
        data, archive = self.make([('program.exe', b'invented', None)])
        with self.assertRaises(m.Stop): m.inspect_archive(data, archive)

    def test_manifest_pin(self):
        (self.bundle / 'TRANSPORT.json').write_text('{}')
        with self.assertRaises(m.Stop): m.verify(self.bundle)

    def test_restore_existing_output(self):
        out = self.root / 'out'; out.mkdir()
        with self.assertRaises(m.Stop): m.restore(self.bundle, out)

    def test_restore_validates_before_output(self):
        out = self.root / 'out'
        with self.assertRaises(m.Stop): m.restore(self.bundle, out)
        self.assertFalse(out.exists())

    def test_commit_requires_full_sha(self):
        with self.assertRaises(m.Stop): m.check_commit(self.root, 'main')

    def test_commit_mismatch(self):
        with mock.patch.object(m.subprocess, 'check_output', return_value='0' * 40):
            with self.assertRaises(m.Stop): m.check_commit(self.root, '1' * 40)

    def test_commit_dirty(self):
        with mock.patch.object(m.subprocess, 'check_output', side_effect=['1' * 40, ' M file']):
            with self.assertRaises(m.Stop): m.check_commit(self.root, '1' * 40)


if __name__ == '__main__':
    unittest.main()
