"""Source-only contract tests for blind ASR on the Piper phoneme probe."""
import sys,unittest,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_ipa_piper_probe_asr_once as a
class PiperBlindAsrTests(unittest.TestCase):
 def test_trigger_is_fixed(self):
  x=a.verify_trigger()
  self.assertEqual(x["source_artifact_id"],14187636805)
  self.assertEqual(x["source_artifact_sha256"],"0e84ed8724748b7a9891527e44efc079a60514943e41fcdf0fbb585db44ff253")
 def test_exact_provider_roster_and_budget(self):
  self.assertEqual(a.PROVIDERS,("qwen","whisper","parakeet"))
  self.assertEqual(a.MAX_CALLS,12)
 def test_no_training_or_gold(self):
  x=a.verify_trigger()
  self.assertFalse(x["training_admitted"])
  self.assertFalse(x["human_gold"])
 def test_actual_cli_parser(self):
  with self.assertRaises(SystemExit) as caught:a.main(["--help"])
  self.assertEqual(caught.exception.code,0)
if __name__=="__main__":unittest.main()
