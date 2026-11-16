"""Source-only contracts for final-IPA Piper v2 CTC phone observations."""
import sys,unittest,tempfile,json,wave,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_piper_v2_ipa_ctc_once as p
def sha(b): return hashlib.sha256(b).hexdigest()
def fixture(root):
 probe=root/"piper-probe-parent"/"probe"; probe.mkdir(parents=True)
 lock={"schema":"english-ipa-piper-model-lock-v1","repo":"rhasspy/piper-voices","revision":"de3dcfdf1912bb49726dc4aa11c26017ce2ac62a","model_sha256":"9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f","engine_version":"1.8.0","speaker_count":904}
 plan={"schema":"english-ipa-piper-phoneme-probe-v2","model_revision":lock["revision"],"model_sha256":lock["model_sha256"],"training_admitted":False,"human_gold":False,"pronunciation_verified":False,"cells":[]}
 for phrase,ipa,raw in [("Hey Mero","/heɪ ˈmiːroʊ/","hˈeɪ mˈiːɹoʊ"),("Hey Vosen","/heɪ ˈvoʊzən/","hˈeɪ vˈoʊzən")]:
  for speaker,sid in [("p3922",0),("p8699",1)]:
   cell={"phrase":phrase,"ipa":ipa,"raw_espeak_ipa":raw,"source_speaker_id":speaker,"speaker_id":sid};plan["cells"].append(cell)
   cid="cell-"+str(len(plan["cells"]));folder=probe/"cells"/cid;folder.mkdir(parents=True)
   pcm=b"\0\0"*800;wav=folder/"audio-16k.wav"
   with wave.open(str(wav),"wb") as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(pcm)
   wb=wav.read_bytes()
   receipt={"schema":"english-ipa-piper-phoneme-probe-v2","cell_id":cid,"phrase":phrase,"ipa_target":ipa,"raw_espeak_ipa":raw,"source_speaker_id":speaker,"speaker_id":sid,"training_admitted":False,"human_gold":False,"pronunciation_verified":False,"audio":{"wav_bytes":len(wb),"wav_sha256":sha(wb),"pcm_sha256":sha(pcm),"audio_frames":800}}
   (folder/"completed.json").write_text(json.dumps(receipt))
 (probe/"model-lock.json").write_text(json.dumps(lock));(probe/"source-plan.json").write_text(json.dumps(plan))
 (probe/"result.json").write_text(json.dumps({"schema":"english-ipa-piper-phoneme-probe-v2","status":"COMPLETE","attempts":4,"completed":4,"asr_calls":0,"training_admitted":False,"human_gold":False,"pronunciation_verified":False}))
 return root
class PiperV2IpaCtcTests(unittest.TestCase):
 def test_exact_trigger(self):
  t=p.verify_trigger();self.assertEqual(t["source_artifact_id"],14191228436);self.assertEqual(t["max_new_phoneme_forwards"],4)
  self.assertEqual(p.PHRASES["Hey Mero"],"/heɪ ˈmiːroʊ/");self.assertEqual(p.PHRASES["Hey Vosen"],"/heɪ ˈvoʊzən/")
 def test_source_cells_verify_hashes_and_roster(self):
  with tempfile.TemporaryDirectory() as tmp:
   _,rows=p.source_cells(fixture(Path(tmp)));self.assertEqual(len(rows),4)
   self.assertEqual({x["phrase"] for x in rows},{"Hey Mero","Hey Vosen"})
 def test_modified_wav_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=fixture(Path(tmp));wav=next((root/"piper-probe-parent"/"probe"/"cells").glob("*/audio-16k.wav"));wav.write_bytes(wav.read_bytes()+b"x")
   with self.assertRaises(ValueError):p.source_cells(root)
 def test_old_pronunciation_schema_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=fixture(Path(tmp));path=root/"piper-probe-parent"/"probe"/"source-plan.json";plan=json.loads(path.read_text());plan["cells"][0]["ipa"]="/heɪ ˈmɪəroʊ/";path.write_text(json.dumps(plan))
   with self.assertRaises(ValueError):p.source_cells(root)
 def test_training_never_admitted(self):
  t=p.verify_trigger();self.assertIs(t["training_admitted"],False);self.assertIs(t["human_gold"],False)
 def test_cli_parser(self):
  with self.assertRaises(SystemExit) as e:p.main(["--help"])
  self.assertEqual(e.exception.code,0)
if __name__=="__main__":unittest.main()
