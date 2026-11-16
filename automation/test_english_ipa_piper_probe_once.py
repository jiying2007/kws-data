"""Source-only contract tests for Piper raw-phoneme probe."""
import sys,unittest,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_ipa_piper_probe_once as p

class PiperProbeTests(unittest.TestCase):
 def test_exact_fixed_roster(self):
  p.verify_trigger()
  self.assertEqual(len(p.EXPECTED_CELLS),4)
  self.assertEqual({(x["phrase"],x["speaker_id"]) for x in p.EXPECTED_CELLS},
    {("Hey Mero",0),("Hey Mero",1),("Hey Vosen",0),("Hey Vosen",1)})
 def test_exact_raw_ipa_strings(self):
  self.assertEqual(p.EXPECTED_CELLS[0]["raw_espeak_ipa"],"hˈeɪ mˈɪəɹoʊ")
  self.assertEqual(p.EXPECTED_CELLS[2]["raw_espeak_ipa"],"hˈeɪ vˈoʊsən")
 def test_no_training_or_asr(self):
  self.assertEqual(p.MAX_CALLS,4)
  self.assertEqual(p.PIPER_VERSION,"1.8.0")
  self.assertEqual(p.REV,"de3dcfdf1912bb49726dc4aa11c26017ce2ac62a")
 def test_voice_model_is_loaded_once_outside_cell_loop(self):
  tree=ast.parse(Path(p.__file__).read_text(encoding="utf-8"))
  functions={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
  synth=functions["synthesize"]
  self.assertFalse(any(isinstance(node,ast.Attribute) and node.attr=="load" and
    isinstance(node.value,ast.Name) and node.value.id=="PiperVoice"
    for node in ast.walk(synth)))
  execute=functions["execute"]
  loads=[node for node in ast.walk(execute) if isinstance(node,ast.Call) and
    isinstance(node.func,ast.Attribute) and node.func.attr=="load" and
    isinstance(node.func.value,ast.Name) and node.func.value.id=="PiperVoice"]
  self.assertEqual(len(loads),1)
  calls=[node for node in ast.walk(execute) if isinstance(node,ast.Call) and
    isinstance(node.func,ast.Name) and node.func.id=="synthesize"]
  self.assertEqual(len(calls),1)
  self.assertIsInstance(calls[0].args[0],ast.Name)
  self.assertEqual(calls[0].args[0].id,"voice")
 def test_cli_parser(self):
  with self.assertRaises(SystemExit) as caught:
   p.main(["--help"])
  self.assertEqual(caught.exception.code,0)
if __name__=="__main__": unittest.main()
