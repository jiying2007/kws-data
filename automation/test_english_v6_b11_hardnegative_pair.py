"""Independent source-only regression for the measured B11 auxiliary TRAIN draw."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b11_hardnegative_pair as pair


# Direct readback of original run 45094102574, artifact 14783505472, score/RESULT.json.
# Each tuple is one frozen B9 original-gain error family, with its unique original cells.
FAMILIES = (
    ("p2992", 451, ("command", "question")),
    ("p30", 452, ("command", "question", "statement")),
    ("p3905", 449, ("command",)),
    ("p7384", 450, ("command", "question", "request", "statement")),
    ("p7498", 448, ("command", "request", "statement")),
)
DRAW_SHA = "5b26c6982090c7db20789c526f70b181774e9e4415d3fefb7349ec980d1139f0"


def measured_rows():
    return [{"source_group_id": "piper-libritts-" + source, "id": f"b11-train-{idx}-{cell}"}
            for source, idx, cells in FAMILIES for cell in cells]


class MeasuredHardDraw(unittest.TestCase):
    def test_frozen_192_steps_and_unique_source_coverage(self):
        originals = measured_rows()
        group_order = ["piper-libritts-" + source for source, _, _ in FAMILIES]
        draws = pair.freeze_draw(list(reversed(originals)), 192)
        self.assertEqual(draws, pair.freeze_draw(originals, 192))
        self.assertEqual(len(draws), 192)
        self.assertEqual(set(r["id"] for r in draws), set(r["id"] for r in originals))
        counts = Counter(r["source_group_id"] for r in draws)
        self.assertEqual([counts[g] for g in group_order], [39, 39, 38, 38, 38])
        encoded = "".join(r["id"] + "\n" for r in draws).encode("utf-8")
        self.assertEqual(sha256(encoded).hexdigest(), DRAW_SHA)

    def test_duplicate_or_single_family_is_ineligible(self):
        originals = measured_rows()
        for invalid in ([originals[0]], originals + [originals[2]]):
            with self.subTest(size=len(invalid)), self.assertRaises(ValueError):
                pair.freeze_draw(invalid, 192)


if __name__ == "__main__":
    unittest.main()
