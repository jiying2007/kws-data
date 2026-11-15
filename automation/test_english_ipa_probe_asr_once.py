"""Source-only contract tests for blind IPA probe ASR; no model, network or audio execution."""
import sys,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_ipa_probe_asr_once as a

class ProbeAsrTests(__import__("unittest").TestCase):
 def test_fixed_provider_roster(self):
  self.assertEqual(a.PROVIDERS,("qwen","whisper","parakeet"))
  self.assertEqual(a.MAX_CALLS,12)
 def test_pronunciation_roles_remain_unadmitted(self):
  self.assertEqual(a.PHRASES,{"Hey Mero":"/heɪ ˈmɪəroʊ/","Hey Vosen":"/heɪ ˈvoʊsən/"})
 def test_actual_cli_has_no_duplicate_flags(self):
  with self.assertRaises(SystemExit) as caught:
   a.main(["--help"])
  self.assertEqual(caught.exception.code,0)
 def test_trigger_is_exactly_pinned(self):
  self.assertEqual(a.verify_trigger()["source_artifact_sha256"],
    "35fc9ef1a76c5056d3c8307a87f9372f4d3f67412223d39c24fbc7724ee9b788")
 def test_three_asr_is_not_gold(self):
  self.assertEqual(a.SOURCE_ARTIFACT_SHA256,
    "35fc9ef1a76c5056d3c8307a87f9372f4d3f67412223d39c24fbc7724ee9b788")
  self.assertEqual(a.COMPACT_ARTIFACT_ID,14168142552)
  self.assertEqual(a.COMPACT_ARTIFACT_SHA256,
    "717b52d4fef8b17aa3e91baf1e28663c8d337919b4dd49f5cfdbba050f32cf64")
  self.assertEqual(a.PROVIDERS,("qwen","whisper","parakeet"))
  self.assertEqual(a.MAX_CALLS,12)
if __name__=="__main__":
 import unittest;unittest.main()
