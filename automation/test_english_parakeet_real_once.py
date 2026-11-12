"""No-model regression for NVIDIA Parakeet Q8 GGUF on original English WAVs."""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_parakeet_real_once as p
from test_english_qwen_asr_sourcefix_once import EnglishAsrSourcefixTests


class ParakeetOnceTests(unittest.TestCase):
    def test_pinned_distinct_model_gguf_and_cpu_binary(self):
        c = p.request()
        self.assertEqual(c["original_audio_artifact_id"], 13942568634)
        self.assertEqual(c["max_new_asr_calls"], 10)
        self.assertEqual(c["model_id"], "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(c["model_revision"], p.MODEL_COMMIT)
        self.assertEqual(c["model_gguf_sha256"], p.MODEL_SHA256)
        self.assertEqual(c["runtime_release_sha256"], p.RUNTIME_SHA256)
        self.assertEqual(p.RUNTIME_BYTES, 9605101)
        self.assertFalse(c["human_gold"])
        self.assertFalse(c["training_admitted"])
        self.assertFalse(c["three_asr_completed"])

    def test_original_audio_is_reused_without_other_recognizer_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            EnglishAsrSourcefixTests().populate(parent)
            plan, sources = p.source_inputs(parent)
            self.assertEqual(len(sources), 10)
            self.assertEqual(plan["locale"], "en-US")
            self.assertEqual({x["role"] for x, _, _ in sources},
                             {"train-candidate", "development-only"})
            self.assertEqual(p.main(["--artifact-parent",str(parent),
                                     "--verify-source"]),0)
            self.assertEqual(p.main(["--artifact-parent",str(parent)]),2)
            audio = sources[0][1]
            audio.write_bytes(audio.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError,"CAPTURE_WAV_SHA_DRIFT"):
                p.source_inputs(parent)

    def test_release_archive_extraction_rejects_aliases_and_works_with_leading_dot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "fixture.tar.gz"
            with tarfile.open(archive,"w:gz") as tar:
                d = tarfile.TarInfo("./")
                d.type = tarfile.DIRTYPE
                tar.addfile(d)
                d = tarfile.TarInfo("./bin/")
                d.type = tarfile.DIRTYPE
                tar.addfile(d)
                content = b"ELF!" + b"x" * 2048
                file = tarfile.TarInfo("./bin/nemo-speech")
                file.size = len(content)
                tar.addfile(file, io.BytesIO(content))
            def fake_fetch(url, target, maximum, expected_sha, expected_length):
                self.assertEqual(url,p.RUNTIME_URL)
                target.write_bytes(archive.read_bytes())
                return {"bytes":len(archive.read_bytes()),"sha256":"invented"}
            with mock.patch.object(p,"fetch_bounded",side_effect=fake_fetch):
                # fixture archive lives at root, but fetch writes a different fixed path.
                # source archive fixture must be separate from target.
                exe, receipt = p.verify_runtime(root)
            self.assertEqual(exe.read_bytes(),content)
            self.assertEqual(exe.stat().st_mode & 0o700,0o700)

    def test_execution_is_blind_bounded_and_replay_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            EnglishAsrSourcefixTests().populate(parent)
            output = parent / "new-real-results"
            calls = []
            def fake_runtime(root):
                binary = root / "nemo-speech"
                binary.write_bytes(b"invented")
                return binary, {"bytes":7,"sha256":"mock"}
            def fake_model(root):
                model = root / "private-gguf"
                model.write_bytes(b"invented")
                return model, {"bytes":8,"sha256":"mock"}
            def fake_executor(cmd, **kwargs):
                self.assertEqual(cmd[1],"transcribe")
                self.assertEqual(cmd[3],"--model")
                self.assertEqual(cmd[5:7],["--device","cpu"])
                self.assertNotIn("Hey Mero"," ".join(cmd))
                self.assertNotIn("Hey Vosen"," ".join(cmd))
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"],"1")
                calls.append(str(cmd[2]))
                return type("NoModelProcess",(),{"returncode":0,"stdout":"Hey, Vosen"})()
            with mock.patch.dict("os.environ",{
                "GITHUB_REPOSITORY":"jiying2007/kws-data",
                "GITHUB_REF":"refs/heads/main",
                "GITHUB_RUN_ATTEMPT":"1"},clear=False), \
                mock.patch.object(p,"verify_runtime",side_effect=fake_runtime), \
                mock.patch.object(p,"verify_gguf",side_effect=fake_model):
                result=p.run(parent,output,executor=fake_executor)
                self.assertEqual(result["completed_asr"],10)
                self.assertEqual(result["asr_attempts"],10)
                self.assertEqual(result["status"],"COMPLETE")
                self.assertFalse(result["human_gold"])
                self.assertFalse(result["training_admitted"])
                self.assertEqual(len(calls),10)
                self.assertFalse((output/"private-models").exists())
                with self.assertRaisesRegex(ValueError,"PARAKEET_OUTPUT_CREATE_ONCE"):
                    p.run(parent,output,executor=fake_executor)

    def test_wrong_main_scope_cannot_create_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"unwritten"
            with mock.patch.dict("os.environ",{
                "GITHUB_REPOSITORY":"unrelated/runner",
                "GITHUB_REF":"refs/heads/main",
                "GITHUB_RUN_ATTEMPT":"1"},clear=False):
                with self.assertRaisesRegex(ValueError,"PARAKEET_ONCE_MAIN_ONLY"):
                    p.run(Path(tmp),out)
            self.assertFalse(out.exists())


if __name__=="__main__":
    unittest.main()
