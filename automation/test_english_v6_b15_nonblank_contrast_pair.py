"""Source-only B15 protocol and nonblank competitor regression (no Torch, audio or KWS)."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b15_nonblank_contrast_pair as b15

class B15ProtocolTests(unittest.TestCase):
    def test_immutable_study_and_new_assessment(self):
        design=b15.read_design()
        assessment=json.loads(b15.B15_ASSESSMENT.read_bytes())
        self.assertEqual(assessment["schema"],"english-v6-b15-prospective-nonblank-contrast-assessment-v1")
        self.assertTrue(assessment["freeze_before_optimizer"])
        self.assertEqual(design["pair"]["updates_per_arm"],192)
        self.assertEqual(design["pair"]["challenger_aux_weight"],0.08)
        self.assertEqual(design["pair"]["natural_negative_ctc_weight"],0.08)
        self.assertEqual(design["retention"]["eligible_originals"],43)
        self.assertEqual(design["retention"]["eligible_groups"],34)
        self.assertFalse(design["promotion_allowed"])
        self.assertTrue(assessment["constraints"]["retained_b14_replication_sealed"])
        self.assertFalse(assessment["constraints"]["training_uses_dev_cal_locked"])
    def test_only_two_other_nonblank_competitors(self):
        for token,expected in ((1,[2,3]),(2,[1,3]),(3,[1,2])):
            self.assertEqual(b15.nonblank_competitors(token),expected)
            self.assertNotIn(0,b15.nonblank_competitors(token))
        with self.assertRaises(ValueError):
            b15.nonblank_competitors(0)
    def test_primary_replication_source_reservations_are_distinct(self):
        panel=json.loads(b15.B15_ASSESSMENT.read_bytes())
        p=set(panel["primary"]["synthetic"]["speaker_indices"])
        r=set(panel["replication"]["synthetic"]["speaker_indices"])
        self.assertEqual(len(p),len(r),12)
        self.assertFalse(p & r)
        self.assertTrue(all(x>645 for x in p|r))
        self.assertIn("natural",panel["primary_gate"]["ordinary"])

if __name__=="__main__":
    unittest.main()
