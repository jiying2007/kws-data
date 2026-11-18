"""Model-free tests of admission boundaries and actual recipe declarations."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("meh_policy_under_test", HERE / "english_mero_meh_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
recipe = policy.recipe


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def texts(value):
    return {provider: value for provider in policy.TEXT_PROVIDERS}


def fixture():
    """Explicit synthetic JSON fixture, not an audio or model-quality result."""
    rows = recipe.rows()
    phones = {
        "mero-slow": "heɪmɛroʊ", "mero-fast": "heɪmɛroʊ",
        "vosen-slow": "heɪvoʊsən", "vosen-fast": "heɪvoʊsən",
        "nero": "heɪnɛroʊ", "memo": "heɪmɛmoʊ", "bosen-s": "heɪboʊsən", "bosen-z": "heɪboʊzən",
        "hello": "həloʊðɛr", "morning": "ɡʊdmɔrnɪŋ", "turn-off": "tɜrnɪtɔf", "open-door": "oʊpənðədɔr",
    }
    packets = []
    for row in rows:
        row.update(generation_status="success", sample_rate=16000, channels=1, sample_width=2,
                   pcm_samples=16000, pcm_rms=0.1, pcm_rail_fraction=0.0, pcm_quality_passed=True,
                   pcm_sha256=sha(row["id"] + "/mock-pcm"), wav_sha256=sha(row["id"] + "/mock-wav"),
                   parent_pcm_sha256=sha(row["id"] + "/mock-native"), parent_sample_rate=24000,
                   parent_dtype="float32-little-endian", pcm_path="pcm/" + row["id"] + ".pcm",
                   wav_path="wav/" + row["id"] + ".wav")
        observations = {}
        for provider in policy.PROVIDERS:
            observation = {"status": "success", "input_pcm_sha256": row["pcm_sha256"],
                "input_samples": row["pcm_samples"], "target_conditioned": False,
                "input_coverage_complete": True, "decoder_output_complete": True,
                "capture_path": "observers/" + provider + "/rows/" + row["id"] + "/CAPTURE.json",
                "capture_sha256": sha(row["id"] + "/mock-" + provider)}
            if provider == "phoneme":
                observation["phone_tokens"] = [phones[row["cell_id"]]]
            else:
                observation["raw_transcript"] = row["requested_text"]
            observations[provider] = observation
        packets.append({"id": row["id"], "observations": observations})
    corpus = {"schema": "english-mero-meh-corpus-v1", "campaign_id": recipe.CAMPAIGN_ID,
        "source_design_id": recipe.CAMPAIGN_ID, "diagnostic_only": False, "training_allowed": True,
        "auto_admission_allowed": True, "permanently_excluded_from_training": False,
        "source_identity_contract_sha256": recipe.digest(recipe.identity_contract()), "rows": rows}
    evidence = {"schema": "english-mero-meh-machine-evidence-v1", "campaign_id": recipe.CAMPAIGN_ID,
                "rows": packets}
    return corpus, evidence, recipe.identity_contract()


class BlindDecisionTests(unittest.TestCase):
    def decide(self, speech, phones):
        return policy.blind_candidate(texts(speech), [phones])

    def test_transcript_homophones_support_the_same_observed_target(self):
        result = policy.blind_candidate({"qwen": "Hey Mero.", "whisper": "Hay Marrow!",
                                         "parakeet": "Hey Merrow"}, ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"])
        self.assertEqual((result["tier"], result["observed_class"]), ("silver", 2))

    def test_two_supporters_are_weak_not_acoustic_failure(self):
        value = texts("Hey Mero")
        value["parakeet"] = "Hey Mera"
        result = policy.blind_candidate(value, ["heɪmɛroʊ"])
        self.assertEqual((result["tier"], result["observed_class"]), ("weak", 2))

    def test_competing_word_is_unknown_not_relabelled(self):
        value = texts("Hey Mero")
        value["whisper"] = "Hey Memo"
        result = policy.blind_candidate(value, ["heɪmɛroʊ"])
        self.assertEqual(result["tier"], "unknown")
        self.assertEqual(result["contradictory_providers"], ["whisper"])

    def test_meryl_disagreement_is_not_gold_pronunciation_judgment(self):
        value = texts("Hey Mero")
        value["parakeet"] = "Hey Meryl"
        result = policy.blind_candidate(value, ["heɪmɛroʊ"])
        self.assertEqual(result["reason"], "ASR_PHONE_COMPETING_WORD_DISAGREEMENT")
        self.assertIsNone(result["observed_class"])

    def test_old_failed_final_vowels_do_not_become_new_positives(self):
        for final in ("ɔ", "ə", "ʌ", "ɚ"):
            with self.subTest(final=final):
                self.assertEqual(self.decide("Hey Mero", "heɪmɛr" + final)["tier"], "unknown")

    def test_excluded_first_vowels_do_not_enter_primary_class(self):
        for vowel in ("i", "ɪ", "ɪə", "æ", "e", "eɪ"):
            with self.subTest(vowel=vowel):
                self.assertEqual(self.decide("Hey Mero", "heɪm" + vowel + "roʊ")["tier"], "unknown")

    def test_vosen_z_is_not_an_added_positive_target(self):
        self.assertEqual(self.decide("Hey Vosen", "heɪvoʊzən")["tier"], "unknown")
        self.assertEqual(self.decide("Hey Vosin", "heɪvoʊsən")["observed_class"], 1)

    def test_bosen_s_and_z_are_independently_supported_negatives(self):
        for sibilant in ("s", "z"):
            result = self.decide("Hey Bosun", "heɪboʊ" + sibilant + "ən")
            self.assertEqual((result["tier"], result["observed_class"]), ("silver", 0))
            self.assertEqual(result["matched_pronunciation_hypothesis"], "bosen-us-" + sibilant + "-v1")

    def test_v_cannot_impersonate_b_negative(self):
        result = self.decide("Hey Bosen", "heɪvoʊsən")
        self.assertEqual(result["tier"], "unknown")
        self.assertIsNone(result["observed_class"])

    def test_missing_h_requires_three_text_supporters(self):
        self.assertEqual(self.decide("Hey Mero", "eɪmɛroʊ")["tier"], "silver")
        value = texts("Hey Mero"); value["whisper"] = "Hey Mera"
        self.assertEqual(policy.blind_candidate(value, ["eɪmɛroʊ"])["tier"], "unknown")

    def test_arbitrary_content_is_not_background(self):
        self.assertEqual(self.decide("Unregistered sentence", "arbitrary")["tier"], "unknown")

    def test_wake_anywhere_blocks_ordinary_negative(self):
        result = self.decide("Hello there", "həloʊðɛrheɪmɛroʊ")
        self.assertEqual(result["tier"], "unknown")
        value = texts("Hello there"); value["parakeet"] = "Hey Mero"
        self.assertEqual(policy.blind_candidate(value, ["həloʊðɛr"])["tier"], "unknown")


class CorpusDecisionTests(unittest.TestCase):
    def test_complete_mock_roster_can_enter_training_without_probe_flags(self):
        result = policy.build_admission(*fixture())
        self.assertTrue(result["training_snapshot_allowed"])
        self.assertEqual(result["balanced_groups_by_role"], recipe.MINIMUM_BALANCED_GROUPS)
        self.assertEqual(result["tier_counts"], {"silver": 144})
        self.assertEqual(sum(r["training_allowed"] for r in result["rows"]), 72)
        for row in result["rows"]:
            if row["role"] != "train":
                self.assertFalse(row["training_allowed"])
                self.assertIsNone(row["training_class"])
                self.assertEqual(row["training_weight"], 0)

    def test_unknown_row_is_retained_without_poisoning_other_supported_rows(self):
        corpus, evidence, contract = fixture()
        evidence["rows"][0]["observations"]["whisper"] = {"status": "missing"}
        result = policy.build_admission(corpus, evidence, contract)
        row = result["rows"][0]
        self.assertTrue(result["training_snapshot_allowed"])
        self.assertEqual(len(result["rows"]), 144)
        self.assertEqual(row["tier"], "unknown")
        self.assertIsNone(row["research_class"])
        self.assertFalse(row["training_allowed"])
        self.assertEqual(row["training_weight"], 0)

    def test_entire_unbalanced_group_masks_supported_evidence_and_stops(self):
        corpus, evidence, contract = fixture()
        for packet in evidence["rows"][:2]:
            packet["observations"]["phoneme"] = {"status": "failed"}
        result = policy.build_admission(corpus, evidence, contract)
        self.assertFalse(result["training_snapshot_allowed"])
        self.assertEqual(result["balanced_groups_by_role"]["train"], 5)
        for row in result["rows"][:12]:
            self.assertEqual(row["tier"], "unknown")
            self.assertIsNone(row["research_class"])
            self.assertEqual(row["training_weight"], 0)
        self.assertEqual(result["rows"][2]["evidence_tier"], "silver")

    def test_failed_intended_positive_never_becomes_background(self):
        corpus, evidence, contract = fixture()
        for provider in policy.TEXT_PROVIDERS:
            evidence["rows"][0]["observations"][provider]["raw_transcript"] = "Hey Memo"
        evidence["rows"][0]["observations"]["phoneme"]["phone_tokens"] = ["heɪmɛmoʊ"]
        row = policy.build_admission(corpus, evidence, contract)["rows"][0]
        self.assertEqual(row["observed_class"], 0)
        self.assertIsNone(row["research_class"])
        self.assertEqual(row["reason"], "OBSERVED_CONTENT_DISAGREES_WITH_REQUESTED_CLASS_OR_KIND")

    def test_geometry_bound_28672_passes_and_28673_is_masked(self):
        self.assertEqual(len(policy.whole_support_ends(28672)), 3)
        self.assertEqual(len(policy.whole_support_ends(28673)), 2)
        for n, tier in ((28672, "silver"), (28673, "unknown")):
            corpus, evidence, contract = fixture()
            corpus["rows"][0]["pcm_samples"] = n
            for observation in evidence["rows"][0]["observations"].values():
                observation["input_samples"] = n
            self.assertEqual(policy.build_admission(corpus, evidence, contract)["rows"][0]["tier"], tier)

    def test_same_pcm_binding_cannot_be_changed(self):
        corpus, evidence, contract = fixture()
        evidence["rows"][0]["observations"]["phoneme"]["input_pcm_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SAME_COMPLETE_PCM"):
            policy.build_admission(corpus, evidence, contract)

    def test_partial_or_target_conditioned_observation_is_unknown(self):
        for flag, value in (("input_coverage_complete", False), ("decoder_output_complete", False),
                            ("target_conditioned", True)):
            corpus, evidence, contract = fixture()
            evidence["rows"][0]["observations"]["phoneme"][flag] = value
            self.assertEqual(policy.build_admission(corpus, evidence, contract)["rows"][0]["tier"], "unknown")

    def test_missing_planned_row_and_role_substitution_are_rejected(self):
        corpus, evidence, contract = fixture()
        corpus["rows"].pop()
        with self.assertRaisesRegex(ValueError, "ALL_144"):
            policy.build_admission(corpus, evidence, contract)
        corpus, evidence, contract = fixture()
        corpus["rows"][0]["role"] = "locked-test"
        with self.assertRaisesRegex(ValueError, "FROZEN_REQUEST_ROLE"):
            policy.build_admission(corpus, evidence, contract)

    def test_wrong_pcm_quality_summary_is_not_a_silent_pass(self):
        corpus, evidence, contract = fixture()
        corpus["rows"][0]["pcm_rail_fraction"] = 0.5
        with self.assertRaisesRegex(ValueError, "PCM_QUALITY_FLAG"):
            policy.build_admission(corpus, evidence, contract)

    def test_af_heart_and_probe_promotion_are_not_available(self):
        self.assertNotIn("af_heart", recipe.PRESETS)
        self.assertEqual(len(recipe.PRESETS), 12)
        self.assertTrue(all(recipe.PRESET_ROLES[x] == "train" for x in
                            ("af_bella", "af_sarah", "am_michael", "am_puck")))
        corpus, evidence, contract = fixture()
        corpus["permanently_excluded_from_training"] = True
        with self.assertRaisesRegex(ValueError, "FRESH_TRAINING_PURPOSE"):
            policy.build_admission(corpus, evidence, contract)


if __name__ == "__main__":
    unittest.main()
