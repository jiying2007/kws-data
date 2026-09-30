import importlib.util
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('repair',Path(__file__).with_name('repair_statistics.py'));d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
t=d.torch
class StableKLTests(unittest.TestCase):
 def test_zero_q_terms(self):
  q=t.tensor([[1.,0.,0.,0.,0.,0.]]);logits=t.tensor([[0.,-1000.,-1000.,-1000.,-1000.,-1000.]])
  self.assertEqual(float(d.stable_kl(q,logits)[0]),0.)
 def test_positive_q_underflow_p(self):
  q=t.tensor([[0.,1.,0.,0.,0.,0.]]);logits=t.tensor([[0.,-1000.,-1000.,-1000.,-1000.,-1000.]])
  self.assertEqual(float(logits.softmax(-1)[0,1]),0.)
  self.assertEqual(float(d.stable_kl(q,logits)[0]),1000.)
 def test_distribution_self_match(self):
  x=t.tensor([[1.,2.,3.,4.,5.,6.]]);q=x.softmax(-1)
  self.assertLess(abs(float(d.stable_kl(q,x)[0])),1e-6)
 def test_nonfinite_and_negative_rejected(self):
  with self.assertRaises(AssertionError):d.stable_kl(t.full((1,6),1/6),t.full((1,6),float('inf')))
  with self.assertRaises(AssertionError):d.stable_kl(t.tensor([[2.,-1.,0.,0.,0.,0.]]),t.zeros(1,6))
if __name__=='__main__':unittest.main()
