"""B16/source-only independent comparison gate: zero models or audio."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b16_b15_primary_eval as v

class B16Gate(unittest.TestCase):
    def test_prequalified_protocol_has_no_model_scorer(self):
        x=v.protocol()
        self.assertFalse(x["scoring_allowed"])
        self.assertIsNone(x["inputs"]["synth_screen"])
        self.assertFalse(x["promotion_allowed"])
        self.assertFalse(x["B14_replication_opened"])
        self.assertFalse(x["B16_replication_opened"])
        self.assertEqual(v.GATE["synthetic_positive"],12)
        self.assertEqual(v.GATE["synthetic_paired"],8)
    def test_source_shortfall_before_model_weights(self):
        passed,missing=v.source_gate(v.GATE)
        self.assertEqual(passed,"SOURCE_GATE_PASS")
        self.assertFalse(missing)
        c=dict(v.GATE);c["synthetic_positive"]=11;c["synthetic_paired"]=7
        blocked,short=v.source_gate(c)
        self.assertEqual(blocked,"INSUFFICIENT_SOURCE")
        self.assertEqual(set(short),{"synthetic_positive","synthetic_paired"})
        with self.assertRaises(ValueError):
            v.source_gate({"synthetic_positive":99})
    def test_original_three_gain_research_decision_contract(self):
        self.assertIn("0.75",v.old_decision.GAINS)
        self.assertEqual(set(v.GAINS),set(v.old_decision.GAINS))
        self.assertEqual(v.old_decision.ARMS,("B0","B9","control","challenger"))

if __name__=="__main__":unittest.main()
