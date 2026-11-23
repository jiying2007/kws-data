import hashlib
from pathlib import Path
import sys
from collections import Counter
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b11_dev_panel as source


class FrozenPanelTest(unittest.TestCase):
    def test_design(self):
        primary, repeat = [source.planned(p) for p in ("primary", "replication")]
        self.assertEqual(len({r["id"] for r in primary + repeat}), 96)
        self.assertEqual(len({r["source_group_id"] for r in primary + repeat}), 24)
        for rows in (primary, repeat):
            self.assertEqual(set(Counter(r["source_group_id"] for r in rows).values()), {4})
            self.assertTrue(all(r["role"] == "development" and r["observed_class"] is None
                            and r["training_allowed"] is False for r in rows))
            positives = [r for r in rows if r["requested_class"] == 2]
            negatives = [r for r in rows if r["requested_class"] == 0]
            self.assertEqual(Counter(r["length_scale"] for r in positives), {0.92: 12, 1.08: 12})
            self.assertEqual(Counter(r["length_scale"] for r in negatives), {0.92: 12, 1.08: 12})
            for name, _ in source.ORDINARY:
                self.assertEqual(Counter(r["length_scale"] for r in negatives if r["cell_id"] == name),
                                 {0.92: 3, 1.08: 3})

    def test_panel_pins(self):
        for panel, expected in PINS.items():
            digest = hashlib.sha256(source.b10.canon(source.source_plan(panel))).hexdigest()
            self.assertEqual(digest, expected)

    def test_speaker_edges(self):
        mapping = {"test-" + str(i): i for i in range(904)}
        for idx, speaker in source.GROUPS["primary"] + source.GROUPS["replication"]:
            mapping.pop("test-" + str(idx))
            mapping[speaker] = idx
        source.check_mapping(mapping)
        speaker = source.GROUPS["primary"][0][1]
        mapping[speaker], mapping["test-0"] = 0, mapping[speaker]
        with self.assertRaises(ValueError):
            source.check_mapping(mapping)


PINS = {'primary': '5a51f8c33c6250ed1b33166833970e43b64d3339e4081ea669b3d332aae7d1a0', 'replication': '183580b7300f3322ddd6ddbf99b9532bf9e97c74a850fe7ab77dfd41e50c0eb0'}


if __name__ == "__main__":
    unittest.main()
