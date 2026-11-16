"""Source-only checks for 16-cell blind ASR against exact Piper v4 artifact."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_piper_v4_probe_asr_once as a

class PiperV4AsrContract(unittest.TestCase):
 def test_exact_source_trigger(self):
  x=a.verify_trigger()
  self.assertEqual(x["source_run_id"],43898728142)
  self.assertEqual(x["source_artifact_id"],14240252131)
  self.assertEqual(x["max_calls"],48)
  self.assertEqual(x["cells_per_provider"],16)
  self.assertEqual(a.SOURCE_ARTIFACT_SHA256,"9a611676829195c8ac07c07e50705cd8e98822703c60ab66e21ee5be03567193")
 def test_provider_roster(self):
  self.assertEqual(a.PROVIDERS,("qwen","whisper","parakeet"))
  self.assertEqual(a.MAX_CALLS,48)
  self.assertEqual(a.CELLS_PER_PROVIDER,16)
 def test_speaker_and_phrase_roster(self):
  self.assertEqual(a.PHRASES,{"Hey Mero":"/heɪ ˈmiːroʊ/","Hey Vosen":"/heɪ ˈvoʊzən/"})
  self.assertEqual(a.SPEAKERS,{"p4535":2,"p6701":3,"p3638":4,"p922":5,"p2531":6,"p1638":7,"p8848":8,"p6544":9})
 def test_raw_phoneme_targets(self):
  self.assertEqual(a.RAW_IPA,{"Hey Mero":"heɪ mˈiːɹoʊ","Hey Vosen":"heɪ vˈoʊzən"})
 def test_source_aggregate_mismatch_requires_complete_cells(self):
  src=Path(__file__).resolve().parents[1]/".github/triggers/english-ipa-piper-v4-three-asr-20261116.json"
  import json
  d=json.loads(src.read_text())
  self.assertEqual(d["retry_count"],0)
  self.assertIs(d["training_admitted"],False)
  self.assertIs(d["human_gold"],False)
  self.assertIs(d["pronunciation_verified"],False)
  source=Path(a.__file__).read_text(encoding="utf-8")
  self.assertIn('summary.get("status") in ("COMPLETE","INCOMPLETE_NO_RETRY")',source)
  self.assertIn("source_cell_closure_complete",source)
  self.assertIn("source_aggregate_status_mismatch",source)
 def test_cli_parser(self):
  with self.assertRaises(SystemExit) as e:a.main(["--help"])
  self.assertEqual(e.exception.code,0)
if __name__=="__main__":unittest.main()
