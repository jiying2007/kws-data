"""Frozen candidate and complete-denominator policy tests; no model execution."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_pocket12_evaluate as evaluate
import english_pocket12_generate as generate
import english_pocket12_observe as observe
from test_english_pocket12_observe import MockObserverFactory, MockSourceBackend, RUN, test_plan


def three(text):
    return dict.fromkeys(observe.TEXT_PROVIDERS, text)


class CandidateTests(unittest.TestCase):
    def test_both_predeclared_mero_and_vosen_hypotheses(self):
        cases = [("Hey Mero.", ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"], "mero-us-eh-v1"),
            ("Hey Mero.", ["h", "eɪ", "m", "iː", "ɹ", "oʊ"], "mero-us-i-v1"),
            ("Hey Vosen.", ["h", "eɪ", "v", "oʊ", "s", "ə", "n"], "vosen-us-s-v1"),
            ("Hey Vosen.", ["h", "eɪ", "v", "oʊ", "z", "ə", "n"], "vosen-us-z-v1")]
        for text, phones, hypothesis in cases:
            result = evaluate.candidate_from_evidence(three(text), phones, 3)
            self.assertTrue(result["candidate_positive"])
            self.assertEqual(result["candidate_hypothesis"], hypothesis)
            self.assertFalse(result["training_allowed"])
            self.assertFalse(result["final_native_us_pronunciation_verified"])

    def test_marrow_alias_cannot_override_actual_ae_phone(self):
        result = evaluate.candidate_from_evidence(three("Hey Marrow."), ["h", "eɪ", "m", "æ", "ɹ", "oʊ"], 3)
        self.assertEqual(result["class_votes"]["2"], 3)
        self.assertFalse(result["candidate_positive"])

    def test_no_new_alias_no_two_vote_and_no_substring_positive(self):
        phone = ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"]
        for texts, tokens in ((three("Hey Maro."), phone), (three("Hey Mero.") | {"qwen": "Hey Hero."}, phone),
                              (three("Hey Mero."), phone + ["p", "l"])):
            self.assertFalse(evaluate.candidate_from_evidence(texts, tokens, 3)["candidate_positive"])
        result = evaluate.candidate_from_evidence(three("Hey Mero.") | {"qwen": "Hey Hero."}, phone, 3)
        self.assertEqual(result["reason"], "FROZEN_CONTRADICTORY_WAKE_VETO")

    def test_three_whole_support_windows_are_required(self):
        self.assertTrue(evaluate.whole_support_geometry(16000)["geometry_usable_positive"])
        self.assertFalse(evaluate.whole_support_geometry(40000)["geometry_usable_positive"])
        result = evaluate.candidate_from_evidence(three("Hey Mero."), ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"], 2)
        self.assertEqual(result["reason"], "FEWER_THAN_THREE_WHOLE_SUPPORT_WINDOWS")

    def test_missing_source_keeps_unknown_sixty_and_twelve_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate.evaluate(Path(directory), {}, test_plan(), RUN)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(len(result["rows"]), 12)
        self.assertIsNone(result["counts"]["actual_total_requests"])
        self.assertEqual(result["counts"]["actual_total_upper_bound"], 60)

    def test_policy_join_compares_intent_only_after_blind_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            source, plan = Path(directory) / "source", test_plan()
            MockSourceBackend.calls, MockSourceBackend.failing = [], None
            generate.generate(plan, source, Path(directory) / "unused", RUN, backend_factory=MockSourceBackend)
            fixture = MockObserverFactory(source)
            packets = {p: {"summary": {"actual_forward_attempts": 12, "native_forward_returns": 12,
                "completed_count": 12, "attempts_lower_bound": 12, "attempts_upper_bound": 12},
                "rows": [{"id": r["id"], "status": "success", "raw_transcript": fixture.texts[r["id"]] if p != "phoneme" else None,
                    "phone_tokens": fixture.phone_tokens[r["id"]] if p == "phoneme" else None} for r in fixture.rows]}
                for p in observe.PROVIDERS}
            def run(current):
                with patch.object(observe, "readback", side_effect=lambda **kw: current[kw["provider"]]):
                    return evaluate.evaluate(source, {}, plan, RUN)
            good = run(packets)
            self.assertEqual(good["status"], "PASS_POCKET12_SOURCE_FEASIBILITY")
            self.assertEqual(good["counts"]["actual_total_requests"], 60)
            self.assertEqual(good["voice_supported_classes"], {"george": [1, 2], "mary": [1, 2]})
            self.assertTrue(all(n == 1 for n in good["positive_hypothesis_counts"].values()))
            wrong = deepcopy(packets)
            wrong["phoneme"]["rows"][2]["phone_tokens"] = ["h", "eɪ", "m", "ɛ", "ɹ", "oʊ"]
            bad = run(wrong)
            self.assertTrue(bad["rows"][2]["candidate"]["candidate_positive"])
            self.assertFalse(bad["gates"]["all_four_contrasts_corroborated_and_zero_positive"])
            self.assertEqual(bad["status"], "STOP_POCKET12_SOURCE_FEASIBILITY_FAILED")
            wrong = deepcopy(packets)
            wrong["qwen"]["rows"][4]["raw_transcript"] = "Hello there please."
            self.assertFalse(run(wrong)["gates"]["all_four_ordinary_exact_and_no_target_phone"])
            wrong = deepcopy(packets)
            wrong["qwen"]["summary"]["actual_forward_attempts"] = None
            incomplete = run(wrong)
            self.assertIsNone(incomplete["counts"]["actual_total_requests"])
            self.assertEqual(incomplete["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
