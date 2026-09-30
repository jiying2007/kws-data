import importlib.util
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('reference_gate.py'));d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
t=d.torch
class AggregationTests(unittest.TestCase):
 def test_preserves_mass_without_target_renormalization(self):
  p=t.zeros(2,2599);ids=[0,1027,1401,1462,2093];p[:,ids]=.1;p[:,100]=.3;p[:,101]=.2
  q=d.aggregate(p,ids);self.assertTrue(t.allclose(q[:,:5],t.full((2,5),.1)));self.assertTrue(t.allclose(q[:,5],t.full((2,),.5)))
  self.assertTrue(t.allclose(q.sum(-1),t.ones(2)));self.assertTrue(t.allclose(q.log().softmax(-1),q))
 def test_other_is_not_blank(self):
  p=t.zeros(1,2599);p[0,100]=1;q=d.aggregate(p,[0,1027,1401,1462,2093]);self.assertEqual(q.argmax(-1).item(),5);self.assertEqual(q[0,0].item(),0)
 def test_invalid_probability_rejected(self):
  for p in (t.zeros(1,2599),t.full((1,2599),float('nan'))):
   with self.assertRaises(AssertionError):d.aggregate(p,[0,1027,1401,1462,2093])
if __name__=='__main__':unittest.main()
