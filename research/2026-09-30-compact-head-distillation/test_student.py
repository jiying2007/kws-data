import importlib.util
from pathlib import Path
import math
import unittest
s=importlib.util.spec_from_file_location('student',Path(__file__).with_name('student.py'));d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
t=d.torch
class StudentTests(unittest.TestCase):
 def test_clip_equal_weight_and_kl_direction(self):
  logits=t.zeros(100,6);q=t.zeros_like(logits);q[:,5]=1
  a=d.kl_clip(logits,q);b=d.kl_clip(t.zeros(1,6),t.full((1,6),1/6))
  self.assertAlmostEqual(float(a),math.log(6),places=6)
  self.assertAlmostEqual(float(t.stack([a,b]).mean()),math.log(6)/2,places=6)
 def test_head_seed_and_synthetic_gradient(self):
  a=d.new_head();t.randn(97);b=d.new_head();self.assertEqual(d.r.tensor_hash(a.state_dict()),d.r.tensor_hash(b.state_dict()))
  q=t.full((10,6),1/6);loss=d.kl_clip(a(t.zeros(10,140)),q);loss.backward()
  self.assertTrue(all(p.grad is not None and t.isfinite(p.grad).all() for p in a.parameters()))
  self.assertFalse(q.requires_grad)
 def test_invalid_teacher_mass_rejected(self):
  with self.assertRaises(AssertionError):d.kl_clip(t.zeros(3,6),t.zeros(3,6))
if __name__=='__main__':unittest.main()
