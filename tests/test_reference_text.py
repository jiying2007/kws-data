"""Publisher source parsing and content-evidence counterexamples."""
import json
from pathlib import Path
import tempfile
import unittest

from kws_data_contract import reference_text as text


class ReferenceTextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / "content.txt"
        self.file.write_text("SSB00050001.wav\t今 jin1 天 tian1\n", encoding="utf-8")
        self.index = {"dataset_id": "aishell3-slr93", "dataset_release": "fake-test"}
        self.row = {"original_path": "train/wav/SSB0005/SSB00050001.wav",
                    "original_sha256": "a" * 64, "speaker_key": "SSB0005"}
        self.meta = {"wav_sha256": "b" * 64, "pcm_sha256": "c" * 64,
                     "frames": 80000, "sample_rate_hz": 16000}

    def source(self):
        return text.document(self.file, text.sha(self.file.read_bytes()), "aishell3-content-v1")

    def reference(self):
        actual, status, provenance = text.attach(self.index, self.row, self.meta,
                                                 self.source(), (None, {}))
        return {**self.meta, "dataset_id": self.index["dataset_id"],
                "parent_source_sha256": self.row["original_sha256"],
                "reference_text": actual, "reference_text_status": status,
                "reference_text_provenance": provenance}

    def test_aishell_pinyin_and_full_source_record_remain_evidence(self):
        ref = self.reference()
        self.assertEqual(ref["reference_text"], "今天")
        self.assertEqual(ref["reference_text_status"], text.PROVIDED)
        self.assertIn("jin1", ref["reference_text_provenance"]["source_record"])
        text.validate_reference(ref)

    def test_wrong_original_speaker_and_unmatched_filename_are_not_relabelled(self):
        row = dict(self.row, speaker_key="SSB9999")
        with self.assertRaisesRegex(ValueError, "TRANSCRIPT_ORIGINAL_SPEAKER_MISMATCH"):
            text.attach(self.index, row, self.meta, self.source(), (None, {}))
        row = dict(self.row, original_path="train/wav/SSB0005/SSB00059999.wav")
        self.assertEqual(text.attach(self.index, row, self.meta, self.source(), (None, {})),
                         (None, "unknown", None))

    def test_source_drift_and_duplicate_record_rejected(self):
        with self.assertRaisesRegex(ValueError, "TRANSCRIPT_SOURCE_CHANGED"):
            text.document(self.file, "0" * 64, "aishell3-content-v1")
        self.file.write_bytes(self.file.read_bytes() * 2)
        with self.assertRaisesRegex(ValueError, "DUPLICATE_TRANSCRIPT_RECORD"):
            self.source()

    def test_wrong_parent_or_unsupported_checked_claim_rejected(self):
        ref = self.reference()
        ref["parent_source_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "TRANSCRIPT_ORIGINAL_BINDING"):
            text.validate_reference(ref)
        ref = self.reference()
        ref["reference_text_status"] = text.CHECKED
        with self.assertRaisesRegex(ValueError, "REFERENCE_CONTENT_STATUS_NOT_EVIDENCED"):
            text.validate_reference(ref)

    def test_text_hints_and_wrong_audio_geometry_cannot_enable_icl(self):
        observation = {"engine": "sensevoice", "model_id": "iic/SenseVoiceSmall",
                       "model_revision": "e" * 40, "model_assets_sha256": "f" * 64,
                       "wav_sha256": self.meta["wav_sha256"],
                       "pcm_sha256": self.meta["pcm_sha256"], "sample_rate_hz": 16000,
                       "audio_frames": 80000, "actual_text": "今天", "input_mode": "audio-only",
                       "decoder_completed": True}
        hinted = dict(observation, input_mode="target-text-hinted")
        with self.assertRaisesRegex(ValueError, "REFERENCE_CONTENT_NOT_AUDIO_ONLY_COMPLETE"):
            text.validate_observation(hinted)
        observation["audio_frames"] = 79999
        with self.assertRaisesRegex(ValueError, "REFERENCE_CONTENT_CHECK_AUDIO_MISMATCH"):
            text.attach(self.index, self.row, self.meta, self.source(),
                        ("a" * 64, {self.meta["wav_sha256"]: [observation]}))

    def test_normalization_preserves_english_boundaries_and_contractions(self):
        self.assertEqual(text.normalize("今 天，天气！"), text.normalize("今天天气"))
        self.assertEqual(text.normalize("Hello, world!"), text.normalize("hello world"))
        self.assertEqual(text.normalize("We're ready"), text.normalize("we’re ready"))
        for publisher, observed in (("the rapist", "therapist"),
                                    ("we're ready", "were ready")):
            with self.subTest(publisher=publisher):
                self.assertNotEqual(text.normalize(publisher), text.normalize(observed))
                observation = {"engine": "fixture", "model_id": "fixture-model",
                    "model_revision": "e" * 40, "model_assets_sha256": "f" * 64,
                    "wav_sha256": self.meta["wav_sha256"],
                    "pcm_sha256": self.meta["pcm_sha256"], "sample_rate_hz": 16000,
                    "audio_frames": 80000, "actual_text": observed,
                    "input_mode": "audio-only", "decoder_completed": True}
                self.assertFalse(text._checked(publisher, [observation], self.meta))

    def test_content_file_keeps_disagreement_and_rejects_duplicate_engine(self):
        observation = {"engine": "fixture", "model_id": "fixture-model",
            "model_revision": "e" * 40, "model_assets_sha256": "f" * 64,
            "wav_sha256": self.meta["wav_sha256"], "pcm_sha256": self.meta["pcm_sha256"],
            "sample_rate_hz": 16000, "audio_frames": 80000, "actual_text": "今天",
            "input_mode": "audio-only", "decoder_completed": True}
        path = self.root / "observations.json"
        observations = [observation, dict(observation, engine="second", actual_text="明天")]
        path.write_text(json.dumps({"schema": text.CHECK_SCHEMA, "observations": observations}))
        actual, status, proof = text.attach(self.index, self.row, self.meta,
                                            self.source(), text.content_checks(path))
        self.assertEqual((actual, status), ("今天", text.PROVIDED))
        self.assertEqual(proof["content_observations"], observations)
        for invalid, error in (([observation, observation], "DUPLICATE_REFERENCE_CONTENT_CHECK"),
                              ([dict(observation, engine=str(n)) for n in range(4)],
                               "REFERENCE_CONTENT_COUNT")):
            with self.subTest(error=error):
                path.write_text(json.dumps({"schema": text.CHECK_SCHEMA, "observations": invalid}))
                with self.assertRaisesRegex(ValueError, error):
                    text.content_checks(path)


if __name__ == "__main__":
    unittest.main()
