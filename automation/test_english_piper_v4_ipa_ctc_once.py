"""Source-only contracts for the 16-cell Piper v4 independent phone observer."""
import sys,unittest,tempfile,json,wave,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_piper_v4_ipa_ctc_once as p

def sha(b): return hashlib.sha256(b).hexdigest()
def fixture(root):
 probe=root/"piper-probe-parent"/"probe"
 probe.mkdir(parents=True)
 lock={"repo":"rhasspy/piper-voices","revision":"de3dcfdf1912bb49726dc4aa11c26017ce2ac62a",
       "model_sha256":"9127a559e11603f10b366d1a20ac7426826081dbc521de4c2420c57728d73f0f",
       "engine_version":"1.8.0","speaker_count":904}
 plan={"schema":"english-ipa-piper-phoneme-probe-v4","model_revision":lock["revision"],
       "model_sha256":lock["model_sha256"],"training_admitted":False,"human_gold":False,
       "pronunciation_verified":False,"cells":[]}
 folders=[]
 for phrase,ipa,raw in [("Hey Mero","/heɪ ˈmiːroʊ/","heɪ mˈiːɹoʊ"),
                         ("Hey Vosen","/heɪ ˈvoʊzən/","heɪ vˈoʊzən")]:
  for speaker,sid in p.SPEAKERS.items():
   cell={"phrase":phrase,"ipa":ipa,"raw_espeak_ipa":raw,"source_speaker_id":speaker,"speaker_id":sid}
   plan["cells"].append(cell)
   cid="cell-"+str(len(plan["cells"]))
   folder=probe/"cells"/cid;folder.mkdir(parents=True)
   pcm=b"\0\0"*800
   wav=folder/"audio-16k.wav"
   with wave.open(str(wav),"wb") as f:
    f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(pcm)
   wb=wav.read_bytes()
   receipt={"schema":"english-ipa-piper-phoneme-probe-v4","cell_id":cid,
    "phrase":phrase,"ipa_target":ipa,"raw_espeak_ipa":raw,"source_speaker_id":speaker,
    "speaker_id":sid,"training_admitted":False,"human_gold":False,"pronunciation_verified":False,
    "audio":{"wav_bytes":len(wb),"wav_sha256":sha(wb),"pcm_sha256":sha(pcm),"audio_frames":800}}
   (folder/"completed.json").write_text(json.dumps(receipt))
   folders.append(folder)
 (probe/"model-lock.json").write_text(json.dumps(lock))
 (probe/"source-plan.json").write_text(json.dumps(plan))
 summary={"schema":"english-ipa-piper-phoneme-probe-v4","status":"INCOMPLETE_NO_RETRY",
  "attempts":16,"completed":16,"asr_calls":0,"training_admitted":False,
  "human_gold":False,"pronunciation_verified":False,
  "model_lock_sha256":sha((probe/"model-lock.json").read_bytes()),
  "source_plan_sha256":sha((probe/"source-plan.json").read_bytes())}
 (probe/"result.json").write_text(json.dumps(summary))
 return root

class PiperV4IpaCtcTests(unittest.TestCase):
 def test_exact_trigger_and_source(self):
  t=p.verify_trigger()
  self.assertEqual(t["source_artifact_id"],14240252131)
  self.assertEqual(t["max_new_phoneme_forwards"],16)
 def test_sixteen_source_cells_verify_hashes_and_roster(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=fixture(Path(tmp))
   _,rows=p.source_cells(root)
   self.assertEqual(len(rows),16)
   self.assertEqual({x["phrase"] for x in rows},{"Hey Mero","Hey Vosen"})
   self.assertEqual({x["speaker"] for x in rows},set(p.SPEAKERS))
 def test_modified_wav_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=fixture(Path(tmp))
   wav=next((root/"piper-probe-parent"/"probe"/"cells").glob("*/audio-16k.wav"))
   wav.write_bytes(wav.read_bytes()+b"x")
   with self.assertRaises(ValueError):p.source_cells(root)
 def test_training_never_admitted(self):
  t=p.verify_trigger()
  self.assertIs(t["training_admitted"],False)
  self.assertIs(t["human_gold"],False)
  self.assertEqual(t["new_tts_calls"],0)
  self.assertEqual(t["new_asr_calls"],0)
 def test_cli_parser(self):
  with self.assertRaises(SystemExit) as e:p.main(["--help"])
  self.assertEqual(e.exception.code,0)
if __name__=="__main__":unittest.main()
