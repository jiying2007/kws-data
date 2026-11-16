"""Source-only tests for four-cell pronunciation probe; no model/audio/network."""
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_ipa_qwen17_final_ipa_probe_once as probe

class ProbeTests(unittest.TestCase):
 def test_fixed_readings_and_four_cells(self):
  x=probe.read_trigger()
  self.assertEqual(len(x["cells"]),4)
  self.assertEqual({c["ipa"] for c in x["cells"]},{"/heɪ ˈmiːroʊ/","/heɪ ˈvoʊzən/"})
  self.assertEqual(x["asr_calls"],0)
  self.assertFalse(x["training_admitted"])
  self.assertFalse(x["human_gold"])
 def test_distinct_cell_identity(self):
  x=probe.read_trigger()
  ids=[probe.sha(json.dumps({"i":i,"cell":c,"revision":probe.REV},
      sort_keys=True,separators=(",",":")).encode())[:24] for i,c in enumerate(x["cells"])]
  self.assertEqual(len(ids),len(set(ids)))
 def test_wrong_model_revision_rejected(self):
  original=probe.REV
  self.assertEqual(original,"0c0e3051f131929182e2c023b9537f8b1c68adfe")
 def test_source_plan_is_only_four_calls(self):
  x=probe.read_trigger()
  self.assertEqual(x["max_calls"],4)
  self.assertTrue(x["no_retry"])
if __name__=="__main__": unittest.main()
