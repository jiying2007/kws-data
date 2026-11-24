"""B16 source-only proof: all roles, source IDs and gate frozen before 96 TTS calls."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b16_source as src


class FrozenOriginalSource(unittest.TestCase):
    def test_two_independent_voice_roles(self):
        src.self_test()
        one=src.source_plan("primary")
        two=src.source_plan("replication")
        self.assertEqual(len(one["rows"]),96)
        self.assertEqual(len(two["rows"]),96)
        self.assertEqual({r["speaker_index"] for r in one["rows"]},set(range(670,694)))
        self.assertEqual({r["speaker_index"] for r in two["rows"]},set(range(694,718)))
        self.assertEqual(len({r["source_speaker_id"] for r in one["rows"]}),24)
        self.assertFalse({r["source_speaker_id"] for r in one["rows"]}
                         & {r["source_speaker_id"] for r in two["rows"]})
        self.assertFalse(one["training_allowed"])
        self.assertFalse(one["promotion_allowed"])
    def test_fixed_canonical_phonetic_inputs_and_ordinary(self):
        one=src.source_plan("primary")
        for row in one["rows"]:
            if row["requested_class"]==2:
                self.assertEqual(row["native_input"],"[[heɪ mˈɛɹoʊ]]")
            else:
                self.assertIn(row["cell_id"],("question","request"))
        self.assertEqual(one["frozen_source_gate"],
                         {"positive_originals_min":12,"ordinary_originals_min":24,
                          "paired_source_groups_min":8})
    def test_fail_closed_on_modified_original_roster(self):
        with patch.object(src,"blob_sha",return_value="0"*40):
            with self.assertRaises(ValueError):
                src.source_plan("primary")
    def test_original_frozen_gate_cli_is_available(self):
        from unittest.mock import patch
        with patch.object(src.b14,"gate",return_value=None) as check:
            src.gate(Path("/nonexistent-original-fixture.zip"))
            check.assert_called_once()
    def test_all_original_history_artifacts_are_enumerated(self):
        ids=src.required_archives()
        self.assertEqual(len(ids),len(set(ids)))
        self.assertIn("14789951699",ids)
        self.assertIn("14878153098",ids)
        self.assertLess(ids.index("14789951699"),ids.index("14878153098"))

    def test_historical_chain_includes_b14_primary_before_b15(self):
        # The failed first source run omitted this inherited set; no audio synthesis occurred.
        from unittest.mock import patch
        with patch.object(src.b15,"history",side_effect=RuntimeError("parent history invoked")) as parent:
            with self.assertRaisesRegex(RuntimeError,"parent history invoked"):
                src.history(Path("/nonexistent-prior-source"))
            parent.assert_called_once()
    def test_previous_model_and_audio_are_not_admitted_for_training(self):
        p=src.source_plan("primary")
        self.assertEqual(p["independent_model_training"]["run"],45296300913)
        self.assertFalse(p["independent_model_training"]["model_selected"])
        self.assertFalse(p["threshold_search"])
        self.assertIn("unopened",p["human_natural_dev_role"].replace("zero-KWS-forward","unopened")
                      if "zero-KWS-forward" in p["human_natural_dev_role"] else "unopened")
if __name__=="__main__":
    unittest.main()
