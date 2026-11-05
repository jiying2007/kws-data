#!/usr/bin/env python3
"""Synthetic sampler fixtures, without any training execution."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_weak_sampling as sampling


class ExposureTests(unittest.TestCase):
    def rows(self):
        rows = []
        for speaker in range(6):
            for kind, label in (("positive-intent", [1, 0]), ("positive-intent", [0, 1]),
                                ("speech-negative-intent", [0, 0]), ("partial-intent", [0, 0]),
                                ("near-miss-intent", [0, 0])):
                rows.append({"source_id": f"source-{len(rows)}", "source_family_id": f"speaker-{speaker}",
                             "split": "train", "human_gold": False, "independent_test": False,
                             "ctc_mask": True, "ctc_weight": 0.5 if speaker % 2 else 1.0,
                             "keyword_mask": [True, True], "keyword_targets": label,
                             "keyword_weight": [.5, .5] if speaker % 2 else [1.0, 1.0], "intended_kind": kind})
        return rows

    def test_near_ablation_preserves_total_positive_exposure_and_loss_weights(self):
        rows = self.rows()
        positives = []
        for percent in (0, 5, 10):
            plan = sampling.exposure_plan(rows, total_exposures=200, full_positive_exposures=100,
                                          near_percent=percent)
            draws = plan["draws"]
            self.assertEqual(len(draws), 200)
            self.assertEqual(Counter(d["bucket"] for d in draws)["near-negative"], 2 * percent)
            actual = [d for d in draws if d["bucket"] == "full-positive"]
            self.assertEqual(len(actual), 100)
            positives.append(actual)
            for draw in draws:
                self.assertEqual(draw["ctc_weight"], rows[draw["row_index"]]["ctc_weight"])
                self.assertEqual(draw["keyword_weight"], rows[draw["row_index"]]["keyword_weight"])
            self.assertEqual(len({d["source_family_id"] for d in actual[:6]}), 6)
        self.assertEqual(positives[0], positives[1])
        self.assertEqual(positives[1], positives[2])

    def test_d0_near_only_negatives_cannot_masquerade_as_fixed_exposure_ablation(self):
        rows = [r for r in self.rows() if r["intended_kind"] not in ("speech-negative-intent", "partial-intent")]
        with self.assertRaisesRegex(ValueError, "MISSING_SAMPLING_POOL:ordinary-or-partial"):
            sampling.exposure_plan(rows, total_exposures=100, full_positive_exposures=50, near_percent=0)

    def test_development_rows_and_fractional_budget_are_rejected(self):
        rows = self.rows()
        rows[0]["split"] = "development"
        with self.assertRaisesRegex(ValueError, "RESEARCH_TRAIN_ROWS_REQUIRED"):
            sampling.exposure_plan(rows, total_exposures=100, full_positive_exposures=50, near_percent=10)
        with self.assertRaisesRegex(ValueError, "EXPOSURE_BUDGET"):
            sampling.exposure_plan(self.rows(), total_exposures=101, full_positive_exposures=50, near_percent=5)


if __name__ == "__main__":
    unittest.main()
