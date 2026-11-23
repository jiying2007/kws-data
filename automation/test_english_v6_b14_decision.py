"""Small fixtures for individual blockers in the frozen B14 assessment."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from english_v6_b14_decision import ARMS, GAINS, decide


def example(name, positive, counts):
    return {"id":name, "group":name, "positive":positive, "cohort":"synthetic" if positive else "natural",
            "gains":{g:{a:{"events":v} for a,v in zip(ARMS,counts,strict=True)} for g in GAINS}}


class Decision(unittest.TestCase):
    def setUp(self):
        self.rows = [example("wake",True,[1,1,0,1]), example("reader1",False,[1,1,0,0]), example("reader2",False,[1,1,0,0])]
    def test_all_conditions(self):
        self.assertEqual(decide(self.rows)["panel_gate"],"PASS")
    def test_guardrails(self):
        for index, gain, arm, value in ((0,"1.0","challenger",0),(0,"0.5","challenger",2),(1,"0.75","challenger",1),(1,"1.0","B9",0)):
            rows=deepcopy(self.rows)
            rows[index]["gains"][gain][arm]["events"]=value
            self.assertEqual(decide(rows)["panel_gate"],"NO_PROMOTION")
    def test_no_recovery(self):
        self.assertEqual(decide(self.rows[1:])["panel_gate"],"NO_PROMOTION")
    def test_increment_over_b0_alone(self):
        for arm in ("B9","control","challenger"):
            self.rows[1]["gains"]["0.5"][arm]["events"]=3
        self.assertEqual(decide(self.rows)["natural_above_B0_gains"],["0.5"])


if __name__ == "__main__":
    unittest.main()
