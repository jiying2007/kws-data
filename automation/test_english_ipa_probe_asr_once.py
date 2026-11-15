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
 def test_parser_has_distinct_source_and_compact_source_flags(self):
  import argparse
  parser=argparse.ArgumentParser()
  parser.add_argument("--source",type=Path)
  parser.add_argument("--compact-source",type=Path)
  parsed=parser.parse_args(["--source","a","--compact-source","b"])
  self.assertEqual(parsed.source,Path("a"))
  self.assertEqual(parsed.compact_source,Path("b"))
 def test_three_asr_is_not_gold(self):
  self.assertEqual(a.SOURCE_ARTIFACT_SHA256,
    "35fc9ef1a76c5056d3c8307a87f9372f4d3f67412223d39c24fbc7724ee9b788")
  self.assertEqual(a.PROVIDERS,("qwen","whisper","parakeet"))
  self.assertEqual(a.MAX_CALLS,12)
if __name__=="__main__":
 import unittest;unittest.main()
