"""检查错误数据不能因更新 catalog 哈希而绕过语义校验。"""
import json
import pathlib
import shutil
import tempfile
import unittest
from tools.codex_assets import __main__ as archive


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.original = archive.ROOT
        self.temp = tempfile.TemporaryDirectory(prefix="kws-data-integrity-")
        self.root = pathlib.Path(self.temp.name) / "repo"
        shutil.copytree(self.original, self.root,
                        ignore=shutil.ignore_patterns(".git", "__pycache__"))
        archive.ROOT = self.root

    def tearDown(self):
        archive.ROOT = self.original
        self.temp.cleanup()

    def rebind(self, index, filename, key):
        catalog = json.loads((self.root / "catalog.json").read_text())
        entry = catalog["datasets"][index]
        entry[key] = archive.sha(self.root / entry["path"] / filename)
        archive.write_json(self.root / "catalog.json", catalog)

    def test_clean(self):
        result = archive.verify()
        self.assertEqual(result["recordings"], 42)

    def test_corrupted_audio(self):
        audio = next((self.root / "datasets/qwen3-reviewed-v1/audio").glob("*.wav"))
        payload = bytearray(audio.read_bytes())
        payload[-1] ^= 1
        audio.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "file SHA mismatch"):
            archive.verify()

    def test_relabel_with_rebound_manifest(self):
        manifest = self.root / "datasets/qwen3-reviewed-v1/manifest.json"
        data = json.loads(manifest.read_text())
        data["recordings"][0]["intended_text"] = "你好小温"
        archive.write_json(manifest, data)
        self.rebind(0, "manifest.json", "manifest_sha256")
        with self.assertRaisesRegex(ValueError, "review label mismatch"):
            archive.verify()

    def test_speaker_split_reassignment(self):
        path = self.root / "datasets/qwen3-reviewed-v1/splits.json"
        data = json.loads(path.read_text())
        data["recordings"][0]["split"] = "development_a"
        archive.write_json(path, data)
        self.rebind(0, "splits.json", "splits_sha256")
        with self.assertRaisesRegex(ValueError, "speaker split mismatch"):
            archive.verify()

    def test_asr_identity_reassignment(self):
        path = self.root / "datasets/qwen3-train-asr-v1/asr-review.jsonl"
        rows = archive.read_rows(path)
        rows[0]["asr_model_sha256"] = "0" * 64
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        self.rebind(1, "asr-review.jsonl", "review_sha256")
        with self.assertRaisesRegex(ValueError, "ASR asset identity mismatch"):
            archive.verify()


if __name__ == "__main__":
    unittest.main()
