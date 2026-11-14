"""No-network regressions for evidence projection and preserved label boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automation import v4_keyword_evidence as view


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def english_report():
    rows = []
    for i in range(63):
        rows.append({"cell_id": "cell-" + str(i), "source_family_id": "voice-" + str(i // 9),
                     "voice_id": "voice-" + str(i // 9),
                     "role": "train-candidate" if i < 45 else "development-only",
                     "wav_sha256": sha(i), "requested_text": "Hey Mero",
                     "requested_intent": "wake" if i % 9 < 4 else
                     "near_miss" if i % 9 < 7 else "ordinary_negative", "training_admitted": False,
                     "human_gold": False, "trusted_negative": False,
                     "asr": {p: {"text": "Good morning", "native_sha256": sha((p, i))}
                             for p in ("qwen", "whisper", "parakeet")}})
    return {"rows": rows, "training_admitted": False, "human_gold": False}


def chinese_report():
    items = []
    for i, (targets, masks, weights) in enumerate((([1, 0], [True, True], [0.5, 0.5]),
                                                  ([None, None], [False, False], [0.0, 0.0]))):
        items.append({"item_id": "clip-" + str(i),
                      "source": {"source_id": "source-" + str(i), "source_family_id": "family-" + str(i),
                                 "role": "train-candidate" if i == 0 else "development-only",
                                 "split": "train" if i == 0 else "development"},
                      "input": {"original_wav": {"sha256": sha(i)},
                                "original_pcm": {"sha256": sha((i, "pcm"))}},
                      "asr": [{"model": p, "text": "你好小窝" if i == 0 else "你好小屋",
                               "status": "complete", "usable": True}
                              for p in ("qwen3_asr", "whisper", "sensevoice")],
                      "decision": {"human_gold": False, "keyword_targets": targets,
                                   "keyword_mask": masks, "keyword_weight": weights,
                                   "label_tier": "machine-weak-two" if i == 0 else "machine-weak-uncertain"}})
    return {"schema": "kws-active-context-v3-research-weak-view-v1", "items": items}


class ProjectionTests(unittest.TestCase):
    def test_english_full_token_and_punctuation_boundaries(self):
        self.assertEqual(view.positions("HEY, Mero!", "Hey Mero", "en"), [0])
        self.assertEqual(view.positions("hey merona", "Hey Mero", "en"), [])
        self.assertEqual(view.positions("they mero", "Hey Mero", "en"), [])
        self.assertEqual(view.positions("hey mirror", "Hey Mero", "en"), [])

    def test_chinese_complete_repeated_word_not_half_word(self):
        self.assertEqual(view.positions("小窝", "小窝小窝", "zh"), [])
        self.assertEqual(view.positions("小窝，小窝", "小窝小窝", "zh"), [0])
        self.assertEqual(view.positions("你好小屋", "你好小窝", "zh"), [])

    def test_three_asr_positive_remains_unadmitted_unknown(self):
        report = english_report()
        for row in report["rows"][0]["asr"].values():
            row["text"] = "Hey, Mero!"
        out = view.project_english(report, {"pin": "synthetic"})
        selected = [r for r in out["rows"] if r["source_id"] == "cell-0" and r["keyword_id"] == "en-k2"][0]
        self.assertEqual(selected["positive_provider_count"], 3)
        self.assertTrue(selected["orthographic_positive_candidate"])
        self.assertEqual(selected["label_state"], "unknown")
        self.assertFalse(selected["existing_weak_mask"])
        self.assertFalse(selected["training_admitted"])
        self.assertFalse(selected["human_gold"])
        self.assertIsNone(selected["acoustic_event_times"])
        self.assertEqual(out["audio_rows"], 63)
        self.assertEqual(out["keyword_rows"], 126)
        self.assertEqual(out["asr_observations_reused"], 189)

    def test_absence_votes_and_requested_text_do_not_create_negative(self):
        report = english_report()
        report["rows"][0]["requested_text"] = "Hey Vosen"
        out = view.project_english(report, {})
        self.assertTrue(all(r["label_state"] == "unknown" for r in out["rows"]))
        self.assertTrue(all(not r["trusted_negative"] for r in out["rows"]))
        self.assertEqual(out["accepted_negative_seconds"], 0)

    def test_21_nearphones_stay_quarantined_separate_from_family_split(self):
        out = view.project_english(english_report(), {})
        near = [r for r in out["rows"] if r["requested_intent_diagnostic_only"] == "near_miss"]
        self.assertEqual(len(near), 42)
        self.assertEqual({r["original_research_usage"] for r in near}, {"nearphone_quarantine"})
        self.assertEqual({r["source_split"] for r in near}, {"train", "development"})
        usages = [r["original_research_usage"] for r in out["rows"]]
        self.assertEqual(usages.count("weak_diagnostic_only"), 60)
        self.assertEqual(usages.count("development_no_tune"), 24)

    def test_original_research_action_preserved_without_role_promotion(self):
        report = english_report()
        report["rows"][0]["action"] = "EXPOSED_DEVELOPMENT_NO_TUNE"
        out = view.project_english(report, {})
        self.assertEqual(out["rows"][0]["original_research_usage"], "development_no_tune")
        self.assertEqual(out["rows"][0]["source_split"], "train")
        self.assertEqual(out["rows"][0]["research_usage_basis"], "preserved_original_action")
        for i in (4, 45):
            changed = english_report()
            changed["rows"][i]["action"] = "TRAIN_ROLE_WEAK_INTENT_DIAGNOSTIC_ONLY"
            with self.assertRaisesRegex(ValueError, "USAGE_PROMOTION"):
                view.project_english(changed, {})

    def test_false_gold_or_training_claims_rejected(self):
        for field in ("training_admitted", "human_gold", "trusted_negative"):
            report = english_report()
            report["rows"][0][field] = True
            with self.assertRaisesRegex(ValueError, "BOUNDARY_CHANGED"):
                view.project_english(report, {})

    def test_missing_provider_rejected(self):
        report = english_report()
        del report["rows"][0]["asr"]["qwen"]
        with self.assertRaisesRegex(ValueError, "THREE_FAMILIES"):
            view.project_english(report, {})

    def test_source_family_cannot_move_between_splits(self):
        report = english_report()
        report["rows"][45]["source_family_id"] = "voice-0"
        with self.assertRaisesRegex(ValueError, "SPLIT_LEAK"):
            view.project_english(report, {})

    def test_new_audio_identity_cannot_duplicate_old_hash(self):
        report = english_report()
        report["rows"][1]["wav_sha256"] = report["rows"][0]["wav_sha256"]
        with self.assertRaisesRegex(ValueError, "DUPLICATED_AUDIO_SOURCE"):
            view.project_english(report, {})

    def test_chinese_masks_weights_roles_and_unknown_preserved(self):
        report = chinese_report()
        before = copy.deepcopy(report)
        out = view.project_chinese(report, {}, "zh-newsource8-synthetic64-20261106")
        self.assertEqual(report, before)
        self.assertEqual([r["label_state"] for r in out["rows"]], ["present", "absent", "unknown", "unknown"])
        self.assertEqual([r["existing_weak_weight"] for r in out["rows"]], [0.5, 0.5, 0.0, 0.0])
        self.assertEqual([r["source_split"] for r in out["rows"]], ["train", "train", "development", "development"])
        self.assertTrue(all(not r["human_gold"] and not r["training_admitted"] for r in out["rows"]))

    def test_mask_target_inconsistency_and_nonfinite_weight_rejected(self):
        for target, mask, weight in ((0, False, 0.0), (None, True, 0.5), (True, True, 1.0), (1, True, float("nan"))):
            with self.assertRaises(ValueError):
                view.keyword_row({}, [], "你好小窝", "zh-k1", "zh", target, mask, weight)

    def test_native_chinese_verifier_failure_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = b'{"fixture": "synthetic"}'
            (root / "MANIFEST.json").write_bytes(manifest)
            (root / "report.json").write_text(json.dumps(chinese_report()))
            module = types.SimpleNamespace(verify_view=lambda *a, **k: (_ for _ in ()).throw(ValueError("NATIVE_FAILURE")))
            with patch.object(view.importlib, "import_module", return_value=module):
                with self.assertRaisesRegex(ValueError, "NATIVE_FAILURE"):
                    view.chinese_from_package(root, view.digest(manifest), "synthetic-only-test")

    def test_wrong_manifest_or_repacked_english_zip_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MANIFEST.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "MANIFEST_PIN"):
                view.chinese_from_package(root, "0" * 64, "synthetic-test")
            path = root / "fake.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("english-63-machine-review.json", json.dumps(english_report()))
            with self.assertRaisesRegex(ValueError, "ORIGINAL_ARCHIVE_REQUIRED"):
                view.english_from_archive(path)

    def test_json_duplicate_keys_and_existing_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
            view.load(b'{"human_gold":false,"human_gold":true}')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            report = view.project_english(english_report(), {})
            view.write_result(report, output)
            with self.assertRaisesRegex(ValueError, "CREATE_ONLY"):
                view.write_result(report, output)


if __name__ == "__main__":
    unittest.main()
