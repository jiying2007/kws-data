"""Fixture-only tests for the compact readback verifier; no model/audio generation."""
import hashlib,json,sys,tempfile,wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_ipa_probe_compact_readback as r

def put(p,b): p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def wav(p,rate=16000):
 p.parent.mkdir(parents=True,exist_ok=True)
 with wave.open(str(p),"wb") as w:
  w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(b"\0\0"*160)

class ReadbackTests(__import__("unittest").TestCase):
 def fixture(self,root):
  cells=[]
  rows=[]
  for i,(phrase,ipa) in enumerate((("Hey Mero","/heɪ ˈmɪəroʊ/"),("Hey Vosen","/heɪ ˈvoʊsən/"))):
   for speaker in ("aiden","ryan"):
    cid=f"cell-{i}-{speaker}"
    folder=root/"cells"/cid
    put(folder/"started.json",json.dumps({"cell_id":cid}).encode())
    wav(folder/"native-f32.wav");wav(folder/"audio-16k.wav")
    raw=(folder/"audio-16k.wav").read_bytes()
    pcm=b"\0\0"*160
    audio={"native_sha256":r.sha_file(folder/"native-f32.wav"),
      "wav_sha256":r.sha_file(folder/"audio-16k.wav"),
      "pcm_sha256":hashlib.sha256(pcm).hexdigest(),"audio_frames":160,"wav_bytes":len(raw)}
    completion={"schema":"english-ipa-qwen17-probe-v1","cell_id":cid,
      "phrase":phrase,"ipa_target":ipa,"speaker":speaker,"audio":audio,
      "pronunciation_verified":False,"human_gold":False,"training_admitted":False}
    put(folder/"completed.json",json.dumps(completion).encode())
    cells.append({"cell_id":cid,"phrase":phrase,"ipa":ipa,"speaker":speaker})
    rows.append({"cell_id":cid,"status":"GENERATED_UNADMITTED"})
  put(root/"started.json",json.dumps({"schema":"english-ipa-qwen17-probe-v1","run_id":str(r.RUN_ID),"no_retry":True}).encode())
  put(root/"result.json",json.dumps({"schema":"english-ipa-qwen17-probe-v1","status":"COMPLETE","attempts":4,"completed":4,"asr_calls":0,"training_admitted":False,"human_gold":False,"rows":rows}).encode())
  put(root/"model-lock.json",json.dumps({"schema":"english-ipa-qwen17-model-lock-v1","model_id":r.MODEL_ID,"revision":r.MODEL_REVISION,"license":"Apache-2.0"}).encode())
  put(root/"source-plan.json",json.dumps({"schema":"english-ipa-qwen17-probe-v1","model":r.MODEL_ID,"revision":r.MODEL_REVISION,"training_admitted":False,"human_gold":False,"cells":cells}).encode())
 def test_compacts_four_unadmitted_audio_cells(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/"source";root.mkdir();self.fixture(root)
   out=Path(td)/"compact"
   summary=r.verify_and_compact(root,out)
   self.assertEqual(summary["cell_count"],4)
   self.assertFalse((out/"model").exists())
   self.assertTrue((out/"COMPACT_READBACK.json").is_file())
 def test_preserved_common_parent_directory_is_accepted(self):
  with tempfile.TemporaryDirectory() as td:
   parent=Path(td)/"download";root=parent/"en-ipa-probe";root.mkdir(parents=True)
   self.fixture(root)
   summary=r.verify_and_compact(parent,Path(td)/"compact")
   self.assertEqual(summary["cell_count"],4)
 def test_modified_wav_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/"source";root.mkdir();self.fixture(root)
   cell=next((root/"cells").iterdir())
   with (cell/"audio-16k.wav").open("ab") as f:f.write(b"x")
   with self.assertRaises(ValueError):r.verify_and_compact(root,Path(td)/"compact")
 def test_incomplete_run_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/"source";root.mkdir();self.fixture(root)
   p=root/"result.json";x=json.loads(p.read_text());x["completed"]=3;p.write_text(json.dumps(x))
   with self.assertRaises(ValueError):r.verify_and_compact(root,Path(td)/"compact")
if __name__=="__main__":
 import unittest;unittest.main()
