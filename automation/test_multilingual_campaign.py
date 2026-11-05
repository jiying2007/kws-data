"""Offline/no-model regression tests for the bilingual source-first planner."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multilingual_campaign as campaign


def fake_sha(letter: str, length: int = 64) -> str:
    return letter * length


def pin(name: str, digit: int) -> dict:
    return {"provider_id": name, "revision_sha": f"{digit:040x}",
            "model_assets_sha256": f"{digit:064x}",
            "license_review_sha256": f"{digit + 500:064x}"}


def request(locale: str = "zh-CN") -> dict:
    if locale == "zh-CN":
        words = ["你好小窝", "小窝小窝"]
        prompts = [
            ("wake-1", "你好小窝", "wake", "complete_keyword", 1),
            ("wake-2", "小窝小窝", "wake", "complete_keyword", 2),
            ("near-1", "你好小屋", "near_miss", "homophone", None),
            ("other-1", "今天晚上吃什么", "ordinary_negative", "conversation", None),
        ]
        tts = [pin("qwen3-custom-zh", 1)]
        asr = [pin("qwen3-asr-zh", 11), pin("sensevoice-small-zh", 12),
               pin("whisper-small-zh", 13)]
    else:
        words = ["hello orbit"]
        prompts = [
            ("wake-1", "HELLO orbit!", "wake", "complete_keyword", 1),
            ("near-1", "hello orphan", "near_miss", "other_confusable", None),
            ("other-1", "turn off the light", "ordinary_negative", "conversation", None),
        ]
        tts = [pin("kokoro-82m-en", 21), pin("qwen3-custom-en", 22)]
        asr = [pin("qwen3-asr-en", 31), pin("whisper-small-en", 32),
               pin("parakeet-tdt-v3-en", 33)]
    return {
        "schema_version": 1, "policy": campaign.POLICY, "locale": locale,
        "wake_phrases": words,
        "utterances": [
            {"id": name, "text": text, "intent": kind,
             "challenge": challenge, "keyword_index": keyword_id}
            for name, text, kind, challenge, keyword_id in prompts
        ],
        "voices": [
            {"id": "voice-01", "source_family_id": "orig-01", "role": "train-candidate",
             "origin_kind": "owned-synthetic", "reference_pcm_sha256": None,
             "rights_evidence_sha256": None},
            {"id": "voice-02", "source_family_id": "orig-02", "role": "development-only",
             "origin_kind": "owned-synthetic", "reference_pcm_sha256": None,
             "rights_evidence_sha256": None},
        ],
        "tts_providers": tts, "asr_providers": asr,
        "repetitions": 2, "max_generation_cells": 32, "max_paid_usd": 0,
        "model_calls_authorized": False,
    }


class SourceFirstLanguageTests(unittest.TestCase):
    def test_mandarin_wake_and_homophone_are_distinct(self) -> None:
        p = campaign.plan(request())
        self.assertEqual(p["locale"], "zh-CN")
        self.assertEqual(p["generation_cells_planned"], 16)
        self.assertEqual(p["asr_observations_planned"], 48)
        self.assertEqual(p["source_families_declared"], 2)
        self.assertEqual(len({row["cell_id"] for row in p["cells"]}), 16)
        self.assertEqual({row["role"] for row in p["cells"]},
                         {"train-candidate", "development-only"})
        self.assertFalse(p["training_admitted"])
        self.assertFalse(p["source_rights_authenticated"])
        self.assertFalse(p["independent_acceptance_evidence"])
        self.assertNotIn("shipping_approved", p)
        self.assertNotIn("qualification_allowed", p)
        self.assertTrue(all("qualification_allowed" not in c for c in p["cells"]))
        self.assertEqual(p["model_calls_executed"], 0)
        self.assertEqual(p["requested_intents"], {
            "near_miss": 1, "ordinary_negative": 1, "wake": 2})
        self.assertEqual(
            {x["normalized_text"] for x in [dict(x, normalized_text=campaign.normalize(x["text_requested"],"zh-CN")) for x in p["cells"] if x["intent"] == "wake"]},
            {"你好小窝", "小窝小窝"})

    def test_explicit_english_multiword_and_cross_family_models(self) -> None:
        first = campaign.plan(request("en-US"))
        again = campaign.plan(request("en-US"))
        self.assertEqual(first, again)
        self.assertEqual(first["generation_cells_planned"], 24)
        self.assertEqual(first["asr_observations_planned"], 72)
        self.assertEqual({x["family"] for x in first["asr_pins"]},
                         {"qwen3-asr", "whisper", "nvidia-parakeet"})
        self.assertEqual({x["family"] for x in first["tts_pins"]},
                         {"qwen3-tts", "kokoro"})
        self.assertEqual(first["wake_phrases_requested"], ["hello orbit"])
        self.assertFalse(first["next_campaign_authorized"])

    def test_research_licensed_tts_is_plannable_only_with_source_provenance(self) -> None:
        raw = request()
        raw["tts_providers"] = [pin("f5-tts-official-weights", 46)]
        with self.assertRaisesRegex(ValueError, "CLONING_REQUIRES_EACH_ORIGINAL_REFERENCE_AND_RIGHTS"):
            campaign.plan(raw)
        for idx, voice in enumerate(raw["voices"]):
            voice["reference_pcm_sha256"] = f"{400 + idx:064x}"
            voice["rights_evidence_sha256"] = f"{500 + idx:064x}"
        planned = campaign.plan(raw)
        self.assertEqual(planned["tts_pins"][0]["license_status"],
                         "research-terms-review-required")
        self.assertIsNone(planned["tts_pins"][0]["adapter"])
        self.assertFalse(planned["source_rights_authenticated"])
        self.assertFalse(planned["training_admitted"])
        self.assertFalse(planned["independent_acceptance_evidence"])
        raw["tts_providers"][0]["license_review_sha256"] = "unreviewed"
        with self.assertRaisesRegex(ValueError, "MODEL_LICENSE_REVIEW_MISSING"):
            campaign.plan(raw)

    def test_requires_three_distinct_asr_families(self) -> None:
        raw = request()
        raw["asr_providers"][1] = pin("qwen3-asr-zh", 101)
        with self.assertRaisesRegex(ValueError, "MODEL_FAMILY_NOT_INDEPENDENT"):
            campaign.plan(raw)
        raw = request()
        raw["asr_providers"] = raw["asr_providers"][:2]
        with self.assertRaisesRegex(ValueError, "THREE_DISTINCT_ASR_FAMILIES_REQUIRED"):
            campaign.plan(raw)

    def test_pinned_revisions_and_license_review_receipts_are_mandatory(self) -> None:
        raw = request()
        raw["tts_providers"][0]["revision_sha"] = "main"
        with self.assertRaisesRegex(ValueError, "MODEL_REVISION_NOT_PINNED"):
            campaign.plan(raw)
        raw = request()
        raw["asr_providers"][1]["license_review_sha256"] = None
        with self.assertRaisesRegex(ValueError, "MODEL_LICENSE_REVIEW_MISSING"):
            campaign.plan(raw)

    def test_clone_requires_reference_and_evidence_for_every_voice(self) -> None:
        raw = request()
        raw["tts_providers"] = [pin("qwen3-base-clone-zh", 50)]
        with self.assertRaisesRegex(ValueError, "CLONING_REQUIRES_EACH_ORIGINAL_REFERENCE_AND_RIGHTS"):
            campaign.plan(raw)
        for idx, voice in enumerate(raw["voices"]):
            voice["reference_pcm_sha256"] = f"{400 + idx:064x}"
            voice["rights_evidence_sha256"] = f"{500 + idx:064x}"
        planned = campaign.plan(raw)
        self.assertEqual(planned["generation_cells_planned"], 16)
        self.assertFalse(planned["source_rights_authenticated"])

    def test_intended_negative_never_relabels_exact_wake(self) -> None:
        raw = request()
        raw["utterances"][2]["text"] = "你好小窝"
        with self.assertRaisesRegex(ValueError, "NEGATIVE_MUST_NOT_MATCH_WAKE"):
            campaign.plan(raw)
        raw = request()
        raw["utterances"][0]["text"] = "你好小屋"
        with self.assertRaisesRegex(ValueError, "POSITIVE_MUST_MATCH_FULL_KEYWORD"):
            campaign.plan(raw)
        raw = request()
        raw["utterances"][2]["keyword_index"] = 1
        with self.assertRaisesRegex(ValueError, "NEGATIVE_MUST_NOT_MATCH_WAKE"):
            campaign.plan(raw)

    def test_english_does_not_inherit_mandarin_research_words(self) -> None:
        raw = request("en-US")
        raw["wake_phrases"] = ["Orbit"]
        with self.assertRaisesRegex(ValueError, "ENGLISH_WAKE_MUST_BE_EXPLICIT_MULTIWORD"):
            campaign.plan(raw)
        raw = request("en-US")
        raw["tts_providers"] = [pin("qwen3-custom-zh", 1)]
        with self.assertRaisesRegex(ValueError, "PROVIDER_LANGUAGE_OR_ROLE_UNSUPPORTED"):
            campaign.plan(raw)

    def test_synthetic_voices_do_not_masquerade_as_distinct_train_split(self) -> None:
        raw = request()
        raw["voices"][1]["source_family_id"] = raw["voices"][0]["source_family_id"]
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK:family"):
            campaign.plan(raw)
        raw = request()
        raw["voices"][0]["role"] = "sealed-evaluation"
        with self.assertRaisesRegex(ValueError, "VOICE_ORIGIN_OR_ROLE"):
            campaign.plan(raw)

    def test_same_reference_pcm_cannot_cross_splits_under_different_names(self):
        raw = request()
        raw["tts_providers"] = [pin("qwen3-base-clone-zh", 1)]
        for voice in raw["voices"]:
            voice.update(origin_kind="licensed-public",
                         reference_pcm_sha256="a" * 64,
                         rights_evidence_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY_SPLIT_LEAK:pcm"):
            campaign.plan(raw)

    def test_budget_exposure_and_unknown_provider(self) -> None:
        raw = request()
        raw["max_generation_cells"] = 15
        with self.assertRaisesRegex(ValueError, "REQUESTED_CELLS_EXCEED_FIXED_BUDGET"):
            campaign.plan(raw)
        raw = request()
        raw["max_paid_usd"] = 1
        with self.assertRaisesRegex(ValueError, "CAMPAIGN_REQUEST_POLICY_OR_BUDGET"):
            campaign.plan(raw)
        raw = request()
        raw["tts_providers"][0]["provider_id"] = "fake-paid"
        with self.assertRaisesRegex(ValueError, "UNKNOWN_PROVIDER"):
            campaign.plan(raw)

    def test_duplicate_source_json_fields_and_private_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source, output = work / "request.json", work / "plan.json"
            source.write_text(json.dumps(request(), ensure_ascii=False))
            self.assertEqual(campaign.main(["--request", str(source), "--output", str(output)]), 0)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("shipping_approved", json.loads(output.read_text()))
            with self.assertRaises(FileExistsError):
                campaign.main(["--request", str(source), "--output", str(output)])
            source.write_text('{"policy":"a","policy":"b"}')
            with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
                campaign.read(source)

    def test_source_doc_and_nonmodel_behavior(self) -> None:
        data = campaign.catalog()
        self.assertIn("f5-tts-official-weights", data)
        self.assertEqual(data["openwakeword-framework-en"]["license_status"],
                         "research-terms-review-required")
        self.assertEqual(len(data), 18)
        plan = campaign.plan(request())
        self.assertTrue(all(row["synthetic"] for row in plan["cells"]))
        self.assertTrue(all(not row["actual_text_verified"] for row in plan["cells"]))


if __name__ == "__main__":
    unittest.main()
