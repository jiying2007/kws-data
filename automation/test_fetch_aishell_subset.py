import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_aishell_subset as subject


def wav(seconds=2, rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x01\x00" * int(rate * seconds))
    return output.getvalue()


def archive(path, rows):
    with tarfile.open(path, "w:gz") as stream:
        for name, body, link in rows:
            info = tarfile.TarInfo(name)
            if link:
                info.type, info.linkname = tarfile.SYMTYPE, link
                stream.addfile(info)
            else:
                info.size = len(body)
                stream.addfile(info, io.BytesIO(body))


class Response(io.BytesIO):
    status = 206

    def __init__(self, body, **headers):
        super().__init__(body)
        self.headers = headers


class SourceIntakeTests(unittest.TestCase):
    def test_real_wav_selection_keeps_existing_speakers_out(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive(root / "source.tgz", [
                ("test/content.txt", b"dataset transcript", None),
                ("test/wav/SSB0001/SSB00010001.wav", wav(), None),
                ("test/wav/SSB0002/SSB00020001.wav", wav(2), None),
                ("test/wav/SSB0002/SSB00020002.wav", wav(5), None),
                ("test/wav/SSB0003/SSB00030001.wav", wav(), None)])
            result = subject.select_members(root / "source.tgz", root / "out",
                exclude_speakers=["SSB0001"], max_speakers=1)
            self.assertEqual(result["selected_speakers"], 1)
            self.assertEqual(result["references"][0]["speaker_key"], "SSB0002")
            self.assertEqual(result["references"][0]["duration_seconds"], 5)
            self.assertFalse(result["training_admitted"])
            self.assertFalse(result["full_archive_verified"])
            self.assertFalse((root / "out/original/test/wav/SSB0001").exists())

    def test_tar_link_cannot_become_a_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive(root / "source.tgz", [("test/wav/SSB0001/SSB00010001.wav", b"", "/etc/passwd")])
            with self.assertRaisesRegex(ValueError, "NOT_REGULAR"):
                subject.select_members(root / "source.tgz", root / "out")

    def test_no_path_traversal_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive(root / "source.tgz", [("../escape", b"bad", None)])
            result = subject.select_members(root / "source.tgz", root / "out")
            self.assertEqual(result["selected_speakers"], 0)
            self.assertFalse((root / "escape").exists())

    def test_exact_seed_and_range_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "seed").write_bytes(b"1234")
            def opener(request, timeout):
                self.assertEqual(request.headers["Range"], "bytes=4-7")
                self.assertEqual(request.headers["If-range"], '"version"')
                return Response(b"5678", ETag='"version"', **{"Content-Range": "bytes 4-7/100"})
            result = subject.fetch_prefix(root / "prefix", etag='"version"', target_bytes=8,
                seed=root / "seed", seed_sha256=subject.digest(root / "seed"), opener=opener)
            self.assertEqual((root / "prefix").read_bytes(), b"12345678")
            self.assertEqual(result["seed_bytes"], 4)
            self.assertEqual(result["retrieved_ranges"][0]["bytes"], 4)
            self.assertFalse(result["complete_archive"])

    def test_changed_etag_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "RELEASE_CHANGED"):
                subject.fetch_prefix(Path(folder) / "prefix", etag='"old"', target_bytes=4,
                    opener=lambda *a, **k: Response(b"1234", ETag='"new"',
                                                  **{"Content-Range": "bytes 0-3/100"}))

    def test_range_ignoring_server_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            def opener(*args, **kwargs):
                response = Response(b"1234", ETag='"v"', **{"Content-Range": "bytes 0-3/100"})
                response.status = 200
                return response
            with self.assertRaisesRegex(ValueError, "EXACT_RANGE"):
                subject.fetch_prefix(Path(folder) / "prefix", etag='"v"', target_bytes=4, opener=opener)

    def test_incomplete_http_body_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "INCOMPLETE_HTTP"):
                subject.fetch_prefix(root / "prefix", etag='"v"', target_bytes=4,
                    opener=lambda *a, **k: Response(b"12", ETag='"v"',
                                                  **{"Content-Range": "bytes 0-3/100"}))
            self.assertFalse((root / "prefix.retrieval.json").exists())

    def test_truncated_pcm_is_not_a_complete_reference(self):
        with self.assertRaisesRegex(ValueError, "WAV_TRUNCATED"):
            subject.wav_info(wav()[:-4])


if __name__ == "__main__":
    unittest.main()
