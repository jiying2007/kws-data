"""Pure storage tests: invented bytes only, no model/audio inference."""
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
import wave

import archive as a


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "archive"
        self.plan = self.base / "plan.json"

    def tearDown(self):
        self.tmp.cleanup()

    def entry(self, name, data, **metadata):
        path = self.base / ("input-" + str(len(list(self.base.glob("input-*")))))
        path.write_bytes(data)
        return {"logical_path": name, "source_path": str(path), "sha256": a.digest(data),
                "size_bytes": len(data), "mime_type": "application/octet-stream", **metadata}

    def build(self, entries):
        self.plan.write_bytes(a.canonical({"assets": entries}))
        return a.pack(self.plan, self.root)

    def wav(self, channels=1, rate=16000, width=2):
        buf = io.BytesIO()
        pcm = b"\x00\x00\x01\x00\xff\x7f\x00\x80" * channels
        with wave.open(buf, "wb") as f:
            f.setnchannels(channels)
            f.setsampwidth(width)
            f.setframerate(rate)
            f.writeframes(pcm)
        return buf.getvalue(), pcm

    def test_roundtrip_binary_empty_unicode_alias(self):
        data = b"\x00\xff123\n" * 200
        result = self.build([self.entry("bytes/original.bin", data), self.entry("copy.bin", data),
                             self.entry("空/empty", b"")])
        self.assertEqual(result["stored_unique_objects"], 2)
        self.assertEqual(result["logical_asset_count"], 3)
        self.assertEqual(a.verify(self.root, result["archive_index_sha256"])["status"], "PASS")
        out = self.base / "restored"
        a.verify(self.root, result["archive_index_sha256"], out)
        self.assertEqual((out / "copy.bin").read_bytes(), data)
        self.assertEqual((out / "空/empty").read_bytes(), b"")

    def test_deterministic_and_bounded(self):
        data = random.Random(7).randbytes(220000)
        entries = [self.entry("random.bin", data)]
        result = self.build(entries)
        other = self.base / "other"
        self.assertEqual(a.pack(self.plan, other), result)
        self.assertEqual(a.pack(self.plan, self.root), result)
        files = sorted(p.relative_to(self.root) for p in self.root.rglob("*") if p.is_file())
        self.assertTrue(all((self.root / p).read_bytes() == (other / p).read_bytes() for p in files))
        self.assertTrue(all((self.root / p).stat().st_size <= 65536 for p in files))
        self.assertGreater(result["stored_unique_shards"], 1)
        self.assertEqual(result["max_shard_bytes"], 65536)

    def test_manifest_paging(self):
        entries = [self.entry("item%03d" % i, b"one", provenance="字" * 700) for i in range(80)]
        self.build(entries)
        index = json.loads((self.root / "archive-index.json").read_bytes())
        self.assertGreater(len(index["assets"]), 1)
        self.assertEqual(a.verify(self.root)["logical_asset_count"], 80)

    def test_pcm_derivation(self):
        wav, pcm = self.wav()
        entries = [self.entry("audio/fixed12/synthetic.wav", wav),
                   {"logical_path": "benchmark/release/pcm/synthetic.raw",
                    "source_logical_path": "audio/fixed12/synthetic.wav",
                    "transformation": "wav-pcm-s16le", "sha256": a.digest(pcm), "size_bytes": len(pcm)}]
        result = self.build(entries)
        self.assertEqual(result["stored_unique_objects"], 1)
        self.assertEqual(result["derived_asset_count"], 1)
        out = self.base / "out"
        a.verify(self.root, output=out)
        self.assertEqual((out / entries[1]["logical_path"]).read_bytes(), pcm)

    def test_reject_bad_pcm_identity(self):
        wav, pcm = self.wav()
        with self.assertRaises(a.ArchiveError):
            self.build([self.entry("audio.wav", wav),
                        {"logical_path": "pcm.raw", "source_logical_path": "audio.wav",
                         "transformation": "wav-pcm-s16le", "sha256": "0" * 64, "size_bytes": len(pcm)}])

    def test_reject_noncanonical_wav(self):
        for settings in ({"channels": 2}, {"rate": 8000}, {"width": 1}):
            with self.subTest(settings=settings), self.assertRaises(a.ArchiveError):
                a.wav_pcm(self.wav(**settings)[0])

    def test_reject_missing_or_cyclic_derivation(self):
        entry = {"logical_path": "a.raw", "source_logical_path": "a.raw",
                 "transformation": "wav-pcm-s16le", "sha256": a.digest(b""), "size_bytes": 0}
        with self.assertRaises(a.ArchiveError):
            self.build([entry])
        entry["source_logical_path"] = "missing.wav"
        with self.assertRaises(a.ArchiveError):
            self.build([entry])

    def test_reject_unsupported_derivation(self):
        entry = {"logical_path": "out", "source_logical_path": "parent",
                 "transformation": "execute-pickle", "sha256": a.digest(b""), "size_bytes": 0}
        with self.assertRaises(a.ArchiveError):
            self.build([entry])

    def test_reject_source_mismatch(self):
        entry = self.entry("test", b"abc")
        entry["sha256"] = "0" * 64
        with self.assertRaises(a.ArchiveError):
            self.build([entry])
        entry["size_bytes"] = 99
        with self.assertRaises(a.ArchiveError):
            self.build([entry])

    def test_reject_duplicate_paths(self):
        with self.assertRaises(a.ArchiveError):
            self.build([self.entry("test", b"a"), self.entry("test", b"b")])

    def test_reject_file_directory_collision(self):
        with self.assertRaises(a.ArchiveError):
            self.build([self.entry("test", b"a"), self.entry("test/child", b"b")])

    def test_reject_paths(self):
        for path in ("/absolute", "../parent", "a/../escape", "a//b", "./b", "a\\b", "", "a/", "a\x00b",
                     "C:/escape", "C:escape", "file:stream", "a/CON.txt", "NUL", "a/COM1",
                     "a/LPT²", "a/trailing.", "trailing ", "a\nb", "a*b", "a?b"):
            with self.subTest(path=path), self.assertRaises(a.ArchiveError):
                a.safe_name(path)

    def test_reject_symlink_source(self):
        entry = self.entry("a", b"abc")
        link = self.base / "link"
        link.symlink_to(entry["source_path"])
        entry["source_path"] = str(link)
        with self.assertRaises(a.ArchiveError):
            self.build([entry])

    def test_reject_root_symlink(self):
        real = self.base / "real"
        real.mkdir()
        self.root.symlink_to(real)
        with self.assertRaises(a.ArchiveError):
            self.build([self.entry("test", b"a")])

    def test_reject_symlink_root_ancestor(self):
        real = self.base / "real"
        real.mkdir()
        link = self.base / "parent-link"
        link.symlink_to(real)
        self.root = link / "nested"
        with self.assertRaises(a.ArchiveError):
            self.build([self.entry("test", b"a")])
        self.assertFalse((real / "nested").exists())

    def test_reject_nested_symlink(self):
        self.build([self.entry("a", b"abc")])
        part = next((self.root / "archive/shards").iterdir())
        moved = self.base / "moved"
        part.rename(moved)
        part.symlink_to(moved)
        with self.assertRaises(a.ArchiveError):
            a.verify(self.root)

    def test_reject_tampered_shard_and_no_partial_output(self):
        self.build([self.entry("a", b"abc")])
        part = next((self.root / "archive/shards").iterdir())
        data = bytearray(part.read_bytes())
        data[-1] ^= 1
        part.write_bytes(data)
        output = self.base / "out"
        with self.assertRaises(a.ArchiveError):
            a.verify(self.root, output=output)
        self.assertFalse(output.exists())

    def test_reject_tampered_manifest_and_descriptor(self):
        self.build([self.entry("a", b"abc")])
        for subdir in ("assets", "objects", "object-index"):
            path = next((self.root / "archive" / subdir).iterdir())
            data = path.read_bytes()
            path.write_bytes(data.replace(b"sha256", b"sha257", 1))
            with self.subTest(subdir=subdir), self.assertRaises(a.ArchiveError):
                a.verify(self.root)
            path.write_bytes(data)

    def test_reject_wrong_root_pin(self):
        self.build([self.entry("a", b"abc")])
        with self.assertRaises(a.ArchiveError):
            a.verify(self.root, "0" * 64)

    def test_reject_statistics_before_output(self):
        self.build([self.entry("a", b"abc")])
        path = self.root / "archive-index.json"
        index = json.loads(path.read_bytes())
        index["statistics"]["logical_asset_count"] += 1
        path.write_bytes(a.canonical(index))
        out = self.base / "out"
        with self.assertRaises(a.ArchiveError):
            a.verify(self.root, output=out)
        self.assertFalse(out.exists())

    def test_reject_missing_shard(self):
        self.build([self.entry("a", b"abc")])
        next((self.root / "archive/shards").iterdir()).unlink()
        with self.assertRaises(a.ArchiveError):
            a.verify(self.root)

    def test_reject_nonempty_materialization(self):
        self.build([self.entry("a", b"abc")])
        out = self.base / "out"
        out.mkdir()
        (out / "keep").write_bytes(b"keep")
        with self.assertRaises(a.ArchiveError):
            a.verify(self.root, output=out)
        self.assertEqual((out / "keep").read_bytes(), b"keep")

    def test_reject_oversize_gzip_trailing_and_truncated(self):
        data = a.deterministic_gzip(b"abc" * 10000)
        for raw, size in ((data, 1), (data + b"x", 30000), (data + data, 30000), (data[:-1], 30000)):
            with self.subTest(size=size, length=len(raw)), self.assertRaises(a.ArchiveError):
                a.unpack_gzip(raw, size)

    def test_reject_hash_and_size_formats(self):
        for bad in ("0" * 63, "z" * 64, 123, None):
            with self.assertRaises(a.ArchiveError):
                a.check_hash(bad)
        for bad in (-1, True, 1.0, a.MAX_OBJECT + 1):
            with self.assertRaises(a.ArchiveError):
                a.check_size(bad)

    def test_strip_private_source_path(self):
        self.build([self.entry("a", b"abc", provenance={"original_private_sha256": "f" * 64})])
        public_metadata = b"".join(p.read_bytes() for p in self.root.rglob("*.json"))
        self.assertNotIn(str(self.base).encode(), public_metadata)
        self.assertNotIn(b'"source_path"', public_metadata)

    def test_reject_oversized_single_metadata(self):
        with self.assertRaises(a.ArchiveError):
            self.build([self.entry("a", b"abc", provenance="x" * 65536)])


if __name__ == "__main__":
    unittest.main()
