"""Actual isolated import-path fixtures; no assets, audio or model execution."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b22_source_dynamic_preflight as dynamic


class RuntimeDependencies(unittest.TestCase):
    def isolated_root(self, directory):
        paths = set(dynamic.required_paths()) | {dynamic.POCKET,
            "automation/english_v6_b18_asr_adapter.py",
            "automation/english_v6_b22_source_dynamic_preflight.py"}
        root = Path(directory) / "captured"
        for name in paths:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dynamic.ROOT / name, target)
        return root, paths

    def run_check(self, root):
        return subprocess.run([sys.executable, "-I", "-S", "-B",
            str(root / "automation/english_v6_b22_source_dynamic_preflight.py")],
            cwd=root.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_complete_original_helper_authority_includes_both_research_files(self):
        pins = dynamic.required_paths()
        self.assertEqual(len(pins), 19)
        self.assertEqual({name for name in pins if name.startswith("research/")},
                         {dynamic.CAMPAIGN, dynamic.RUNNER})
        self.assertEqual(pins[dynamic.CAMPAIGN], "55f07e3604ac72a6ff912f25dc86f706678c6bb7661714b3dec3c1646e8d766f")
        self.assertEqual(pins[dynamic.RUNNER], "df3e8858e25f041ecbc1ef7f10e7b7715722ad7ad8a2f5e53f120c172d3d40c1")

    def test_real_loader_routes_and_dynamic_runner_work_inside_only22_captured_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root, paths = self.isolated_root(directory)
            self.assertEqual(len(paths), 22)
            process = self.run_check(root)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["helper_pins"], dynamic.required_paths())
            self.assertTrue(set(result["loaded_repository_modules"]) <= paths)
            self.assertTrue({dynamic.CAMPAIGN, dynamic.RUNNER} <= set(result["loaded_repository_modules"]))
            self.assertEqual(set(result["provider_boundaries"]), {"qwen", "whisper", "parakeet"})
            for provider in result["provider_boundaries"].values():
                self.assertTrue(provider["original_source_check"])
                self.assertTrue(all(provider[key] == 0 for key in
                    ("assets_acquired", "models_constructed", "audio_reads", "forwards")))
            self.assertTrue(result["code_reads_only"] and result["no_model"] and result["no_network"])
            self.assertFalse(result["model_construction_verified"] or result["acoustic_evidence"])

    def test_missing_either_dynamic_research_file_stops_before_loader_boundary(self):
        for missing in (dynamic.CAMPAIGN, dynamic.RUNNER):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root, _ = self.isolated_root(directory)
                (root / missing).unlink()
                process = self.run_check(root)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("complete actual runtime helper: " + missing, process.stderr)
                self.assertFalse(process.stdout)

    def test_changed_research_code_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.isolated_root(directory)
            (root / dynamic.CAMPAIGN).write_text("raise RuntimeError('replacement')\n")
            process = self.run_check(root)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("complete actual runtime helper: " + dynamic.CAMPAIGN, process.stderr)
            self.assertFalse(process.stdout)


if __name__ == "__main__":
    unittest.main()
