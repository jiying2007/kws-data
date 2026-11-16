"""Source-only contracts for the 16-cell stress-aligned Piper v4 speaker screen."""
import sys,unittest,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_ipa_piper_probe_v4_once as p

class PiperV4ProbeTests(unittest.TestCase):
 def test_exact_fixed_roster(self):
  t=p.verify_trigger()
  self.assertEqual(len(p.EXPECTED_CELLS),16)
  self.assertEqual(len(p.SPEAKERS),8)
  self.assertEqual({x["id"] for x in p.SPEAKERS},{2,3,4,5,6,7,8,9})
  self.assertEqual({x["source_speaker_id"] for x in p.SPEAKERS},
   {"p4535","p6701","p3638","p922","p2531","p1638","p8848","p6544"})
  self.assertEqual({(x["phrase"],x["speaker_id"]) for x in p.EXPECTED_CELLS},
   {(phrase,sid) for phrase in ("Hey Mero","Hey Vosen") for sid in range(2,10)})
  self.assertEqual(t["max_calls"],16)
 def test_exact_owner_confirmed_ipa_and_raw_espeak(self):
  for x in p.EXPECTED_CELLS:
   if x["phrase"]=="Hey Mero":
    self.assertEqual(x["ipa"],"/heɪ ˈmiːroʊ/")
    self.assertEqual(x["raw_espeak_ipa"],"heɪ mˈiːɹoʊ")
   else:
    self.assertEqual(x["ipa"],"/heɪ ˈvoʊzən/")
    self.assertEqual(x["raw_espeak_ipa"],"heɪ vˈoʊzən")
 def test_no_old_speaker_reuse_and_no_training(self):
  self.assertFalse({"p3922","p8699"} & {x["source_speaker_id"] for x in p.SPEAKERS})
  self.assertEqual(p.MAX_CALLS,16)
  self.assertEqual(p.PIPER_VERSION,"1.8.0")
  self.assertEqual(p.REV,"de3dcfdf1912bb49726dc4aa11c26017ce2ac62a")
  self.assertEqual(p.verify_trigger()["retry_count"],0)
 def test_model_loaded_once_outside_cell_loop(self):
  tree=ast.parse(Path(p.__file__).read_text(encoding="utf-8"))
  functions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
  synth=functions["synthesize"]
  self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=="load" and isinstance(n.value,ast.Name) and n.value.id=="PiperVoice" for n in ast.walk(synth)))
  execute=functions["execute"]
  loads=[n for n in ast.walk(execute) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="load" and isinstance(n.func.value,ast.Name) and n.func.value.id=="PiperVoice"]
  self.assertEqual(len(loads),1)
  calls=[n for n in ast.walk(execute) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="synthesize"]
  self.assertEqual(len(calls),1)
  self.assertIsInstance(calls[0].args[0],ast.Name)
  self.assertEqual(calls[0].args[0].id,"voice")
 def test_cli_parser(self):
  with self.assertRaises(SystemExit) as caught:p.main(["--help"])
  self.assertEqual(caught.exception.code,0)
if __name__=="__main__":unittest.main()
