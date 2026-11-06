import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import zipfile
from unittest.mock import patch

import acoustic_assets as a


class Response(io.BytesIO):
    def __init__(self, body, status=206, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class AcousticAssetsTest(unittest.TestCase):
    def request(self, members):
        return {"schema": a.SCHEMA, "source": {"url": a.OFFICIAL_URL,
            "homepage": "https://www.openslr.org/28/", "bytes": 12345, "etag": '"fixed"',
            "license_basis": "fixture"}, "purpose": "technical-research-only",
            "human_gold": False, "accepted_natural_negative_hours": 0,
            "max_download_bytes": 1000000, "max_elapsed_seconds": 60, "members": members}

    def test_refuses_same_room_across_splits(self):
        request = self.request([{"member": "RIRS_NOISES/real_rirs_isotropic_noises/a.wav",
            "kind": "measured-rir", "source_group": "same-room", "split": "train"},
            {"member": "RIRS_NOISES/real_rirs_isotropic_noises/b.wav",
            "kind": "measured-rir", "source_group": "same-room", "split": "development"}])
        with self.assertRaisesRegex(ValueError, "CROSSES_SPLIT"):
            a.validate_request(request)

    def test_rejects_ignored_range_and_changed_etag(self):
        for status, etag in [(200, '"fixed"'), (206, '"different"')]:
            with self.subTest(status=status, etag=etag), tempfile.TemporaryDirectory() as tmp:
                source = {"url": a.OFFICIAL_URL, "bytes": 6, "etag": '"fixed"'}
                def opener(*args, **kwargs):
                    return Response(b"abc", status, {"ETag": etag, "Content-Range": "bytes 0-2/6"})
                reader = a.RangeReader(source, tmp, 100, 10, opener=opener)
                with self.assertRaisesRegex(ValueError, "IDENTITY"):
                    reader.read(3)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_bounded_read_refuses_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            def opener(*args, **kwargs):
                self.fail("network should not start")
            reader = a.RangeReader({"url": a.OFFICIAL_URL, "bytes": 100, "etag": '"x"'},
                                   tmp, 2, 10, opener=opener)
            with self.assertRaisesRegex(ValueError, "BYTE_BUDGET"):
                reader.read(3)

    def test_zip_member_collection_and_tamper(self):
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"\\x01\\x00" * 320)
        member = "RIRS_NOISES/pointsource_noises/fixture.wav"
        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(member, wav.getvalue())
        blob = zip_bytes.getvalue()
        request = self.request([{"member": member, "kind": "recorded-noise",
            "source_group": "fixture-source", "split": "train"}])
        request["source"]["bytes"] = len(blob)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p = tmp / "request.json"; p.write_bytes(a.canonical(request))
            original_reader = a.RangeReader
            def opener(req, **kwargs):
                bounds = req.headers["Range"].removeprefix("bytes=").split("-")
                start, end = map(int, bounds)
                return Response(blob[start:end+1], headers={"ETag": '"fixed"',
                    "Content-Range": f"bytes {start}-{end}/{len(blob)}"})
            def factory(*args, **kwargs):
                return original_reader(*args, **kwargs, opener=opener)
            with patch.object(a, "RangeReader", factory):
                a.collect(p, tmp / "output")
            self.assertEqual(a.verify(tmp / "output")["status"], "PASS")
            (tmp / "output/original" / member).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "MEMBER_HASH"):
                a.verify(tmp / "output")


if __name__ == "__main__":
    unittest.main()
