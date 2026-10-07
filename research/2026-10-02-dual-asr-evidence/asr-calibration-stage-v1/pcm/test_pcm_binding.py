"""All unit inputs are stdlib-generated synthetic samples; no model dependencies."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcm_binding as p


def wav(samples=(-32768, -1, 0, 1, 32767)):
    data = b"".join(struct.pack("<h", s) for s in samples)
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(data), b"WAVE",
                       b"fmt ", 16, 1, 1, 16000, 32000, 2, 16, b"data", len(data)) + data


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = self.tmp.name
        self.path = Path(self.root) / "sample.wav"
        self.good = wav()
        self.path.write_bytes(self.good)
        self.expected = self.expect(self.good)

    def tearDown(self):
        self.tmp.cleanup()

    def expect(self, data, **kwargs):
        values = dict(opaque_id="clip-0001", basename="sample.wav", file_sha256=p.sha256(data),
                      file_bytes=len(data), frame_count=(len(data) - 44) // 2)
        values.update(kwargs)
        return p.WaveExpectation(**values)

    def reject(self, data):
        self.path.write_bytes(data)
        with self.assertRaises(p.BindingError):
            p.bind_wave(self.root, self.expect(data))

    def test_known_boundary_sign_bytes(self):
        bound = p.bind_wave(self.root, self.expected)
        self.assertEqual(bound.pcm16_le.hex(), "0080ffff00000100ff7f")
        self.assertEqual(bound.pcm_float32_le.hex(), "000080bf000000b8000000000000003800fe7f3f")
        self.assertEqual(struct.unpack("<5f", bound.pcm_float32_le),
                         (-1.0, -1 / 32768, 0.0, 1 / 32768, 32767 / 32768))

    def test_every_signed_int16_exact_and_order_preserved(self):
        data = wav(range(-32768, 32768))
        self.path.write_bytes(data)
        bound = p.bind_wave(self.root, self.expect(data))
        values = tuple(x[0] for x in struct.iter_unpack("<f", bound.pcm_float32_le))
        self.assertEqual(values, tuple(math.ldexp(float(i), -15) for i in range(-32768, 32768)))
        self.assertEqual(bound.pcm16_le, data[44:])
        self.assertEqual(bound.descriptor["float32_pcm"]["frame_count"], 65536)

    def test_immutable_return(self):
        bound = p.bind_wave(self.root, self.expected)
        self.assertIs(type(bound.pcm16_le), bytes)
        self.assertIs(type(bound.pcm_float32_le), bytes)
        self.assertTrue(memoryview(bound.pcm_float32_le).readonly)
        with self.assertRaises(FrozenInstanceError):
            bound.opaque_id = "clip-0002"
        modified = bound.descriptor
        modified["float32_pcm"]["sha256"] = "bad"
        self.assertNotEqual(bound.descriptor["float32_pcm"]["sha256"], "bad")

    def test_descriptor_hashes_and_no_labels(self):
        bound = p.bind_wave(self.root, self.expected)
        desc = bound.descriptor
        self.assertEqual(bound.descriptor_sha256, p.sha256(bound.descriptor_json))
        for key in ("raw_pcm16", "float32_pcm"):
            canonical = json.dumps(desc[key], sort_keys=True, separators=(",", ":")).encode("ascii")
            self.assertEqual(desc[key + "_descriptor_sha256"], hashlib.sha256(canonical).hexdigest())
        manifest = p.decoder_manifest((bound,))
        encoded = json.dumps(manifest)
        for forbidden in (self.root, "sample.wav", "transcript", "ground_truth", "Serena"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(manifest["order"], ["clip-0001"])

    def test_emitted_descriptor_validates_without_changes(self):
        descriptor = p.bind_wave(self.root, self.expected).descriptor
        before = p._canonical(descriptor)
        p.validate_pcm_descriptor(descriptor)
        self.assertEqual(p._canonical(descriptor), before)

    def test_repeated_binding_deterministic(self):
        self.assertEqual(p.bind_wave(self.root, self.expected), p.bind_wave(self.root, self.expected))

    def test_file_hash_mismatch(self):
        with self.assertRaisesRegex(p.BindingError, "SHA256"):
            p.bind_wave(self.root, replace(self.expected, file_sha256="0" * 64))

    def test_all_truncation_positions(self):
        for n in range(len(self.good)):
            with self.subTest(length=n):
                self.path.write_bytes(self.good[:n])
                with self.assertRaises(p.BindingError):
                    p.bind_wave(self.root, self.expected)

    def test_extra_file_bytes(self):
        self.path.write_bytes(self.good + b"extra")
        with self.assertRaisesRegex(p.BindingError, "size"):
            p.bind_wave(self.root, self.expected)

    def test_extra_bytes_even_with_updated_size_hash(self):
        self.reject(self.good + b"\x00\x00")

    def test_fully_updated_extra_chunk(self):
        altered = bytearray(self.good + b"JUNK\x00\x00\x00\x00")
        struct.pack_into("<I", altered, 4, len(altered) - 8)
        self.reject(bytes(altered))

    def test_duplicate_data_chunk(self):
        altered = bytearray(self.good + b"data\x02\x00\x00\x00\x00\x00")
        struct.pack_into("<I", altered, 4, len(altered) - 8)
        self.reject(bytes(altered))

    def test_fmt_extension_chunk_rejected(self):
        altered = bytearray(self.good[:36] + b"\x00\x00" + self.good[36:])
        struct.pack_into("<I", altered, 4, len(altered) - 8)
        struct.pack_into("<I", altered, 16, 18)
        self.reject(bytes(altered))

    def test_junk_before_fmt_rejected(self):
        altered = bytearray(self.good[:12] + b"JUNK\x00\x00\x00\x00" + self.good[12:])
        struct.pack_into("<I", altered, 4, len(altered) - 8)
        self.reject(bytes(altered))

    def test_header_tags(self):
        for offset, replacement in ((0, b"RIFX"), (0, b"RF64"), (8, b"AVI "),
                                    (12, b"data"), (36, b"fmt ")):
            with self.subTest(offset=offset, tag=replacement):
                altered = bytearray(self.good)
                altered[offset:offset + 4] = replacement
                self.reject(bytes(altered))

    def test_header_integer_mismatches(self):
        fields = ((4, "I", (0, 45, 47, 0xFFFFFFFF)), (16, "I", (0, 18, 40, 0xFFFFFFFF)),
                  (20, "H", (0, 3, 0xFFFE)), (22, "H", (0, 2, 65535)),
                  (24, "I", (0, 8000, 48000)), (28, "I", (0, 16000, 64000)),
                  (32, "H", (0, 1, 4)), (34, "H", (0, 8, 24, 32)),
                  (40, "I", (0, 1, 8, 9, 12, 0xFFFFFFFF)))
        for offset, fmt, values in fields:
            for value in values:
                with self.subTest(offset=offset, value=value):
                    altered = bytearray(self.good)
                    struct.pack_into("<" + fmt, altered, offset, value)
                    self.reject(bytes(altered))

    def test_expectation_types_and_boundaries(self):
        bad = {"file_bytes": (True, 54.0, "54", 0, 44, 45, p.MAX_WAV_BYTES + 1),
               "frame_count": (True, 5.0, "5", 0, -1, p.MAX_WAV_BYTES),
               "sample_rate_hz": (True, 16000.0, 8000), "channels": (True, 2, 0),
               "bits_per_sample": (True, 8, 32),
               "file_sha256": (None, b"0" * 64, "g" * 64, "a" * 63, "A" * 64),
               "opaque_id": (None, "sample.wav", "clip-1", "clip-0001/", "clip-0001\n")}
        for key, values in bad.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(p.BindingError):
                        replace(self.expected, **{key: value})

    def test_predeclared_frames_bytes_mismatch(self):
        with self.assertRaisesRegex(p.BindingError, "frames"):
            replace(self.expected, frame_count=6)

    def test_name_path_aliases(self):
        for name in ("../sample.wav", "./sample.wav", "/sample.wav", "a/sample.wav",
                     "a\\sample.wav", "sample.wav/", "sample.wav\x00", "sample.wav\n",
                     "sample.wav ", "sämple.wav", "", Path("sample.wav")):
            with self.subTest(name=repr(name)):
                with self.assertRaises(p.BindingError):
                    replace(self.expected, basename=name)

    def test_root_path_aliases(self):
        for root in (self.root + "/", self.root + "/.", self.root + "/../" + Path(self.root).name,
                     self.root.replace("/", "//", 1), "relative", Path(self.root), "\x00"):
            with self.subTest(root=repr(root)):
                with self.assertRaises(p.BindingError):
                    p.bind_wave(root, self.expected)

    def test_symlink_final_file(self):
        self.path.rename(Path(self.root) / "target.wav")
        self.path.symlink_to("target.wav")
        with self.assertRaises(p.BindingError):
            p.bind_wave(self.root, self.expected)

    def test_symlink_root_component(self):
        directory = Path(self.root) / "real"
        directory.mkdir()
        (directory / "sample.wav").write_bytes(self.good)
        link = Path(self.root) / "alias"
        link.symlink_to("real", target_is_directory=True)
        for root in (str(link), str(link / "child")):
            with self.subTest(root=root):
                with self.assertRaises(p.BindingError):
                    p.bind_wave(root, self.expected)

    def test_hardlink_alias(self):
        os.link(self.path, Path(self.root) / "alias.wav")
        with self.assertRaisesRegex(p.BindingError, "hard-link"):
            p.bind_wave(self.root, self.expected)

    def test_directory_not_regular(self):
        self.path.unlink()
        self.path.mkdir()
        with self.assertRaisesRegex(p.BindingError, "regular"):
            p.bind_wave(self.root, self.expected)

    def test_fifo_not_regular_does_not_block(self):
        self.path.unlink()
        os.mkfifo(self.path)
        with self.assertRaisesRegex(p.BindingError, "regular"):
            p.bind_wave(self.root, self.expected)

    def test_missing_file(self):
        self.path.unlink()
        with self.assertRaises(p.BindingError):
            p.bind_wave(self.root, self.expected)

    def test_wrong_expectation_type(self):
        with self.assertRaises(p.BindingError):
            p.bind_wave(self.root, {})

    def test_short_read(self):
        with patch.object(os, "read", side_effect=[self.good[:-2], b""]):
            with self.assertRaisesRegex(p.BindingError, "short read"):
                p.bind_wave(self.root, self.expected)

    def test_read_extra_byte_is_bounded(self):
        with patch.object(os, "read", return_value=self.good + b"\x00") as reader:
            with self.assertRaisesRegex(p.BindingError, "extra bytes"):
                p.bind_wave(self.root, self.expected)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(reader.call_args.args[1], len(self.good) + 1)

    def test_read_chunk_bound(self):
        data = wav(range(-32768, 32768))
        self.path.write_bytes(data)
        original = os.read
        sizes = []
        def tracked(fd, size):
            sizes.append(size)
            return original(fd, size)
        with patch.object(os, "read", side_effect=tracked):
            p.bind_wave(self.root, self.expect(data))
        self.assertTrue(all(0 < n <= 65536 for n in sizes))
        self.assertGreater(len(sizes), 2)

    def test_replacement_during_read(self):
        original = os.read
        replaced = False
        def altered(fd, size):
            nonlocal replaced
            result = original(fd, size)
            if not replaced:
                other = Path(self.root) / "new.wav"
                other.write_bytes(self.good)
                os.replace(other, self.path)
                replaced = True
            return result
        with patch.object(os, "read", side_effect=altered):
            with self.assertRaisesRegex(p.BindingError, "changed"):
                p.bind_wave(self.root, self.expected)

    def test_inplace_mutation_during_read(self):
        original = os.read
        changed = False
        def altered(fd, size):
            nonlocal changed
            result = original(fd, size)
            if not changed:
                self.path.write_bytes(self.good[:-2] + b"\xff\xff")
                changed = True
            return result
        with patch.object(os, "read", side_effect=altered):
            with self.assertRaisesRegex(p.BindingError, "changed"):
                p.bind_wave(self.root, self.expected)

    def test_unexpected_read_error_normalized(self):
        with patch.object(os, "read", side_effect=OSError(5, "synthetic error")):
            with self.assertRaisesRegex(p.BindingError, "errno 5"):
                p.bind_wave(self.root, self.expected)

    def test_batch_preserves_order(self):
        (Path(self.root) / "other.wav").write_bytes(wav((2, 1)))
        other = self.expect(wav((2, 1)), opaque_id="clip-0002", basename="other.wav")
        bound = p.bind_batch(self.root, (other, self.expected))
        self.assertEqual([b.opaque_id for b in bound], ["clip-0002", "clip-0001"])
        self.assertEqual(p.decoder_manifest(bound)["order"], ["clip-0002", "clip-0001"])

    def test_duplicate_id_and_path_rejected(self):
        for declarations in ((self.expected, self.expected),
                             (self.expected, replace(self.expected, opaque_id="clip-0002")),
                             (self.expected, replace(self.expected, basename="other.wav"))):
            with self.subTest(declarations=declarations):
                with self.assertRaises(p.BindingError):
                    p.bind_batch(self.root, declarations)

    def test_invalid_batch(self):
        for declarations in ((), ({},), (self.expected,) * 1001):
            with self.assertRaises(p.BindingError):
                p.bind_batch(self.root, declarations)

    def test_infinite_batch_is_bounded(self):
        consumed = 0
        def forever():
            nonlocal consumed
            while True:
                consumed += 1
                yield self.expected
        with self.assertRaises(p.BindingError):
            p.bind_batch(self.root, forever())
        self.assertEqual(consumed, 1001)

    def test_batch_noniterable_rejected(self):
        for function in (lambda: p.bind_batch(self.root, None), lambda: p.decoder_manifest(None)):
            with self.assertRaises(p.BindingError):
                function()

    def test_manifest_tamper_rejected(self):
        bound = p.bind_wave(self.root, self.expected)
        for tampered in (replace(bound, pcm16_le=b"bad"),
                         replace(bound, pcm_float32_le=b"bad"),
                         replace(bound, descriptor_sha256="0" * 64),
                         replace(bound, opaque_id="clip-0002")):
            with self.assertRaises(p.BindingError):
                p.decoder_manifest((tampered,))

    def test_duplicate_or_empty_manifest_rejected(self):
        bound = p.bind_wave(self.root, self.expected)
        for values in ((), (bound, bound), ({},)):
            with self.assertRaises(p.BindingError):
                p.decoder_manifest(values)


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        self.descriptor = json.loads((Path(__file__).resolve().parent / 'decoder-inputs.json').read_bytes())['clips'][0]['descriptor']

    def rehash_samples(self):
        # Matching fixture hashes must not conceal semantic schema errors.
        for name in ('raw_pcm16', 'float32_pcm'):
            if name in self.descriptor:
                self.descriptor[name + '_descriptor_sha256'] = p.sha256(p._canonical(self.descriptor[name]))

    def test_valid_descriptor_does_not_require_claiming_sample_authenticity(self):
        self.descriptor['raw_pcm16']['sha256'] = '0' * 64
        self.rehash_samples()
        # This helper validates declarations; actual sample hashes need bytes.
        p.validate_pcm_descriptor(self.descriptor)

    def test_differing_raw_float_frame_counts_rejected(self):
        pcm = self.descriptor['float32_pcm']
        pcm['frame_count'] += 1; pcm['bytes'] += 4; pcm['shape'][0] += 1
        self.rehash_samples()
        with self.assertRaises(p.BindingError):p.validate_pcm_descriptor(self.descriptor)

    def test_wav_pcm_byte_relationship_rejected(self):
        self.descriptor['wav']['bytes'] += 2
        with self.assertRaises(p.BindingError):p.validate_pcm_descriptor(self.descriptor)

    def test_sample_descriptor_digest_is_checked_without_repair(self):
        self.descriptor['raw_pcm16']['sha256'] = '0' * 64
        before = p._canonical(self.descriptor)
        with self.assertRaises(p.BindingError):p.validate_pcm_descriptor(self.descriptor)
        self.assertEqual(p._canonical(self.descriptor), before)


for name, change in [
    ('top_missing', lambda d:d.pop('float32_pcm')),
    ('top_extra', lambda d:d.update(extra=True)),
    ('schema', lambda d:d.update(schema='incorrect')),
    ('opaque_id', lambda d:d.update(opaque_id='clip-１２３４')),
    ('wav_missing', lambda d:d['wav'].pop('header_bytes')),
    ('wav_extra', lambda d:d['wav'].update(extra=True)),
    ('wav_hash', lambda d:d['wav'].update(sha256='A'*64)),
    ('wav_size_bool', lambda d:d['wav'].update(bytes=True)),
    ('wav_size_max', lambda d:d['wav'].update(bytes=p.MAX_WAV_BYTES+1)),
    ('wav_header_float', lambda d:d['wav'].update(header_bytes=44.0)),
    ('wav_format', lambda d:d['wav'].update(format='WAVE')),
    ('operations_missing', lambda d:d['operations'].pop('trim')),
    ('operations_extra', lambda d:d['operations'].update(gain=False)),
    ('operations_true', lambda d:d['operations'].update(trim=True)),
    ('operations_integer', lambda d:d['operations'].update(trim=0)),
]:
    def test(self, change=change):
        change(self.descriptor); self.rehash_samples()
        with self.assertRaises(p.BindingError):p.validate_pcm_descriptor(self.descriptor)
    setattr(DescriptorTests, 'test_reject_descriptor_' + name, test)

for kind in ('raw_pcm16', 'float32_pcm'):
    for name, change in [
        ('missing', lambda d:d.pop('encoding')),
        ('extra', lambda d:d.update(extra=True)),
        ('frames_bool', lambda d:d.update(frame_count=True)),
        ('frames_float', lambda d:d.update(frame_count=56320.0)),
        ('frames_zero', lambda d:d.update(frame_count=0)),
        ('frames_oversize', lambda d:d.update(frame_count=p.MAX_WAV_BYTES)),
        ('sample_rate', lambda d:d.update(sample_rate_hz=8000)),
        ('sample_rate_float', lambda d:d.update(sample_rate_hz=16000.0)),
        ('channels_bool', lambda d:d.update(channels=True)),
        ('bytes_mismatch', lambda d:d.update(bytes=d['bytes']+1)),
        ('bytes_float', lambda d:d.update(bytes=float(d['bytes']))),
        ('shape_type', lambda d:d.update(shape=tuple(d['shape']))),
        ('shape_rank', lambda d:d.update(shape=d['shape']+[1])),
        ('shape_float', lambda d:d.update(shape=[float(d['frame_count'])])),
        ('shape_mismatch', lambda d:d.update(shape=[d['frame_count']+1])),
        ('encoding', lambda d:d.update(encoding='incorrect')),
        ('dtype', lambda d:d.update(dtype='>f4')),
        ('sample_order', lambda d:d.update(sample_order='reversed')),
        ('hash', lambda d:d.update(sha256='g'*64)),
    ]:
        def test(self, kind=kind, change=change):
            change(self.descriptor[kind]); self.rehash_samples()
            with self.assertRaises(p.BindingError):p.validate_pcm_descriptor(self.descriptor)
        setattr(DescriptorTests, 'test_reject_' + kind + '_' + name, test)

for name, change in [
    ('missing', lambda d:d['float32_pcm'].pop('normalization')),
    ('wrong', lambda d:d['float32_pcm'].update(normalization='signed_int16 / 32767.0')),
    ('raw_extra', lambda d:d['raw_pcm16'].update(normalization='signed_int16 / 32768.0')),
]:
    def test(self, change=change):
        change(self.descriptor); self.rehash_samples()
        with self.assertRaises(p.BindingError):p.validate_pcm_descriptor(self.descriptor)
    setattr(DescriptorTests, 'test_reject_normalization_' + name, test)


if __name__ == "__main__":
    unittest.main(verbosity=2)
