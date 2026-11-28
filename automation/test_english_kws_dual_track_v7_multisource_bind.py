"""Pure source-contract tests for V7 multisource candidate snapshot binding."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, (ROOT / "automation").as_posix())
import english_canonical_multisource_bind_once as source_binder
import english_kws_dual_track_v7_multisource_bind as binder


class MultisourceV7BindTests(unittest.TestCase):
    def test_exact_six_artifact_bundle_and_qualified_role_counts(self):
        cfg = source_binder.config()
        bundle = binder.bundle_from_config(cfg)
        self.assertEqual(bundle["schema"], "english-canonical-multisource-reviewed-pcm-v1")
        self.assertEqual(len(bundle["artifacts"]), 6)
        self.assertEqual(bundle["qualified_groups_by_role"],
                         {"train": 8, "development": 4, "calibration": 3, "locked-test": 3})
        self.assertEqual(bundle["minimum_groups_by_role"],
                         {"train": 6, "development": 2, "calibration": 2, "locked-test": 2})
        self.assertEqual(bundle["selected_machine_reviewed_rows"], 63)
        self.assertEqual(bundle["original_rows_authenticated"], 462)
        self.assertEqual((bundle["new_tts_calls"], bundle["new_asr_calls"],
                          bundle["new_phone_calls"]), (0, 0, 0))
        self.assertFalse(bundle["audio_regenerated"])
        self.assertFalse(bundle["human_gold"])
        self.assertFalse(bundle["original_screen_training_snapshot_allowed"])

    def test_candidate_relative_cross_trigger_mapping(self):
        for observed in (0, 1, 2):
            self.assertIn(binder.candidate_class("VOSEN", observed), (0, 1))
            self.assertIn(binder.candidate_class("MERO", observed), (0, 1))
        self.assertEqual(binder.candidate_class("VOSEN", 1), 1)
        self.assertEqual(binder.candidate_class("VOSEN", 2), 0)
        self.assertEqual(binder.candidate_class("VOSEN", 0), 0)
        self.assertEqual(binder.candidate_class("MERO", 2), 1)
        self.assertEqual(binder.candidate_class("MERO", 1), 0)
        self.assertEqual(binder.candidate_class("MERO", 0), 0)
        with self.assertRaisesRegex(ValueError, "KNOWN_MACHINE_OBSERVED_CLASS"):
            binder.candidate_class("MERO", None)

    def test_candidate_snapshot_contains_only_train_dev_and_preserves_cross_trigger_negatives(self):
        groups = []
        rows = []
        for role, count in (("train", 8), ("development", 4)):
            for index in range(count):
                group_id = f"{role}-group-{index}"
                groups.append({"source_group_id": group_id, "role": role, "qualified": True})
                for observed, suffix in ((2, "mero"), (1, "vosen"), (0, "ordinary")):
                    rows.append({
                        "id": f"{group_id}-{suffix}", "corpus": "piper24",
                        "role": role, "source_group_id": group_id,
                        "weak_class": observed, "requested_class": observed,
                        "tier": "silver", "machine_reason": "TWO_PHONE_CTC_MARGINS_PLUS_BLIND_ASR",
                        "source_pcm_sha256": "a" * 64, "source_wav_sha256": "b" * 64,
                        "samples": 6400,
                    })
        candidate_plan = {
            "VOSEN": {"candidate_id": "hey-vosen-v7", "phrase": "Hey Vosen",
                      "ipa": "/heɪ ˈvoʊ.sən/", "token_labels": ["HEY", "VOH", "SEN"],
                      "token_ids": [1, 2, 3]},
            "MERO": {"candidate_id": "hey-mero-v7", "phrase": "Hey Mero",
                     "ipa": "/heɪ ˈmɛroʊ/", "token_labels": ["HEY", "MEH", "ROH"],
                     "token_ids": [1, 2, 3]},
        }
        plan = {"candidates": candidate_plan, "source_plan_sha256": "c" * 64}
        identity = {"repository": "jiying2007/kws-data", "run_id": 123,
                    "commit": "d" * 40, "parent": "e" * 40, "attempt": 1,
                    "source_bundle_sha256": "c" * 64}
        pcm = b"\x00\x00" * 6400
        wav = b"RIFF" + b"\x00" * 40
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "VOSEN"
            with patch.object(binder, "read_source_audio", return_value=(pcm, wav)):
                result = binder.write_candidate_snapshot(
                    root, "VOSEN", rows, groups, {}, {}, plan, identity)
            manifest = json.loads((root / "MANIFEST.json").read_bytes())
            snapshot = json.loads((root / "SNAPSHOT.json").read_bytes())
            self.assertEqual(result["rows"], 36)
            self.assertEqual(snapshot["role_counts"], {"train": 8, "development": 4})
            self.assertEqual(snapshot["training_rows"], 24)
            self.assertEqual(snapshot["development_rows"], 12)
            self.assertTrue(all(row["role"] in ("train", "development")
                                for row in manifest["rows"]))
            self.assertTrue(all(row["optimizer_allowed"] is False
                                for row in manifest["rows"]))
            self.assertTrue(all(row["training_class"] is None
                                for row in manifest["rows"] if row["role"] == "development"))
            labels = {row["id"].rsplit("-", 1)[-1]: row["research_class"]
                      for row in manifest["rows"] if row["role"] == "train"}
            self.assertEqual(labels["vosen"], 1)
            self.assertEqual(labels["mero"], 0)
            self.assertEqual(labels["ordinary"], 0)
            self.assertFalse(snapshot["calibration_audio_included"])
            self.assertFalse(snapshot["locked_test_audio_included"])
            self.assertFalse(snapshot["locked_test_access_in_training_job"])
            self.assertTrue((root / "FILES.json").is_file())

    def test_evaluation_packet_is_separate_from_training_snapshot(self):
        self.assertIn("def write_evaluation_packet", Path(binder.__file__).read_text())
        self.assertIn("calibration", Path(binder.__file__).read_text())
        self.assertIn("locked-test", Path(binder.__file__).read_text())


if __name__ == "__main__":
    unittest.main()
