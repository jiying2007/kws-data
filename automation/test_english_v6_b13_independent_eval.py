"""Regression fixtures for the frozen speaker, gain and per-original decision."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_v6_b13_independent_eval as ev


def sample(ident, group, cohort, counts, positive=False):
    return dict(id=ident,group=group,cohort=cohort,positive=positive,
                gains={g:{m:{"events":n} for m,n in zip(ev.ARMS,counts,strict=True)} for g in ev.GAINS})


class Gate(unittest.TestCase):
    def setUp(self):
        self.rows=[sample("a","reader1","natural",[0,1,1,0]),
                   sample("b","reader2","natural",[0,1,1,0]),
                   sample("ordinary","synthetic1","synthetic",[0,0,0,0]),
                   sample("positive","synthetic1","synthetic",[1,1,1,1],True)]

    def outcome(self):
        return ev.decision(self.rows)["panel_gate"]

    def test_two_distinct_natural_readers(self):
        self.assertEqual(self.outcome(),"PASS")
        self.rows[1]["group"]="reader1"
        self.assertEqual(self.outcome(),"NO_PROMOTION")

    def test_original_gain_improvement_versus_both(self):
        self.rows[1]["gains"]["1.0"]["B9"]["events"]=0
        self.assertEqual(self.outcome(),"NO_PROMOTION")

    def test_repeat_event_on_synthetic_or_real_stops(self):
        for i in (0,2):
            with self.subTest(row=i):
                block=self.rows[i]["gains"]["0.75"]
                block["challenger"]["events"]=2
                self.assertEqual(self.outcome(),"NO_PROMOTION")
                block["challenger"]["events"]=0

    def test_detection_loss(self):
        self.rows[-1]["gains"]["0.5"]["challenger"]["events"]=0
        self.assertEqual(self.outcome(),"NO_PROMOTION")

    def test_absolute_baseline(self):
        for row in self.rows[:2]:
            for g in ev.GAINS:
                for m in ("B9","control"):
                    row["gains"][g][m]["events"]=2
                row["gains"][g]["challenger"]["events"]=1
        self.assertEqual(self.outcome(),"NO_PROMOTION")


if __name__=="__main__":
    unittest.main()
