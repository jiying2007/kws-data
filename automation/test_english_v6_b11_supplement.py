import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b11_supplement as s


class SupplementPolicy(unittest.TestCase):
    def test_fixed_train_panel(self):
        rows = s.planned()
        self.assertEqual(len(rows), 24)
        self.assertEqual({r["role"] for r in rows}, {"train"})
        self.assertEqual(len({r["source_group_id"] for r in rows}), 6)
        for _, text in s.SENTENCES:
            cases = [r for r in rows if r["text"] == text]
            self.assertEqual(len(cases), 6)
            self.assertEqual({r["rate"] for r in cases}, set(s.RATES))

    def test_votes_must_match_this_sentence_and_both_observers_must_be_present(self):
        row = s.planned()[0]
        raw = b"phone evidence fixture"
        phones = {p: {"logits": raw, "sha256": hashlib.sha256(raw).hexdigest(), "vocab": {}}
                  for p in s.policy.PHONE_MODELS}
        texts = dict.fromkeys(s.policy.TEXT_MODELS, row["text"])
        with patch.object(s.policy, "verified_phone_input"), patch.object(s.policy, "greedy_full_wake", return_value=False):
            self.assertEqual(s.decision(row, texts, phones), "silver")
            self.assertEqual(s.decision(row, texts, {}), "unknown")
            texts["qwen"] = "Different sentence."
            self.assertEqual(s.decision(row, texts, phones), "weak")
            texts["whisper"] = s.SENTENCES[1][1]
            self.assertEqual(s.decision(row, texts, phones), "unknown")
            texts = dict.fromkeys(s.policy.TEXT_MODELS, row["text"])
            texts["parakeet"] = "Can you hear me, Mero?"
            self.assertEqual(s.decision(row, texts, phones), "unknown")
        with patch.object(s.policy, "verified_phone_input"), patch.object(s.policy, "greedy_full_wake", return_value=True):
            self.assertEqual(s.decision(row, dict.fromkeys(texts, row["text"]), phones), "unknown")


if __name__ == "__main__":
    unittest.main()
