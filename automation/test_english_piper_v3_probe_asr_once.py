"""Source-only fixed-contract checks; no ASR model or audio execution."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_piper_v3_probe_asr_once as a

class PiperAsrContract(unittest.TestCase):
 def test_trigger(self):
  x=a.verify_trigger()
  self.assertEqual(x["source_artifact_id"],14235001433)
 def test_provider_roster(self):
  self.assertEqual(a.PROVIDERS,("qwen","whisper","parakeet"))
  self.assertEqual(a.MAX_CALLS,12)
 def test_phrase_profile(self):
  self.assertEqual(a.PHRASES["Hey Mero"],"/heɪ ˈmiːroʊ/")
  self.assertEqual(a.PHRASES["Hey Vosen"],"/heɪ ˈvoʊzən/")
 def test_exact_raw_ipa_profile(self):
  self.assertEqual(a.RAW_IPA["Hey Mero"],"heɪ mˈiːɹoʊ")
  self.assertEqual(a.RAW_IPA["Hey Vosen"],"heɪ vˈoʊzən")
 def test_source_does_not_admit(self):
  x=a.verify_trigger()
  self.assertIs(x["training_admitted"],False)
  self.assertIs(x["human_gold"],False)
 def test_cli_parser(self):
  with self.assertRaises(SystemExit) as e: a.main(["--help"])
  self.assertEqual(e.exception.code,0)

if __name__=="__main__": unittest.main()
