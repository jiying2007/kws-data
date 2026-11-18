"""Pure-stdlib synthetic fixture for immutable 12-clone-reference preparation."""
import collections
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
p=Path(__file__).with_name("english_clone_reference12_once.py")
spec=importlib.util.spec_from_file_location("clone_references",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CELLS=("mero-slow","mero-fast","vosen-slow","vosen-fast","nero","memo",
       "bosen-s","bosen-z","hello","morning","turn-off","open-door")
def dummy():
    rows=[];files={}
    for idx,(preset,role) in enumerate(m.EXPECTED_ROLES.items()):
        for j,cell in enumerate(CELLS):
            sid="test-"+preset+"-"+cell
            pcm=(int(idx*10+j+1)).to_bytes(2,"little",signed=True)*8000
            stream=io.BytesIO()
            with wave.open(stream,"wb") as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
                w.writeframes(pcm)
            wav=stream.getvalue()
            row={"id":sid,"preset_id":preset,"cell_id":cell,"role":role,
                 "requested_text":m.NEUTRAL.get(cell,"control."),
                 "requested_class":0 if cell in m.NEUTRAL else 1,
                 "generation_status":"success","human_gold":False,
                 "source_group_id":"kokoro-v1_0-preset:"+preset,
                 "pcm_quality_passed":True,
                 "pcm_path":"pcm/"+sid+".pcm","wav_path":"wav/"+sid+".wav",
                 "pcm_sha256":m.sha(pcm),"wav_sha256":m.sha(wav)}
            rows.append(row)
            if cell in m.NEUTRAL:
                files[row["pcm_path"]]=pcm
                files[row["wav_path"]]=wav
    files["CORPUS_MANIFEST.json"]=json.dumps({
        "campaign_id":"english-mero-meh-corpus-v1","training_allowed":False,
        "rows":rows}).encode()
    return files

class NeutralReference(unittest.TestCase):
    def test_preflight_preserves_original_roles(self):
        v=m.check_contract()
        self.assertEqual(v["voices"],12)
        self.assertEqual(v["new_tts_calls"],0)
    def test_exact_old_synthetic_audio_reference_composition(self):
        selected=m.select(dummy())
        self.assertEqual(len(selected),12)
        self.assertEqual(collections.Counter(x["role"] for x in selected),
                         collections.Counter(m.EXPECTED_COUNTS))
        self.assertTrue(all(x["reference_text"]==m.REF_TEXT for x in selected))
        self.assertTrue(all(not x["training_allowed"] for x in selected))
        self.assertEqual(len({x["joined_pcm_sha256"] for x in selected}),12)
        with tempfile.TemporaryDirectory() as temp:
            packet=m.write(str(Path(temp)/"refs"),selected)
            self.assertEqual(packet["reference_audio_count"],12)
            self.assertFalse(packet["training_allowed"])
            self.assertEqual(packet["new_model_calls"],0)
            for row in packet["rows"]:
                path=Path(temp)/"refs"/"wav"/(row["id"]+".wav")
                self.assertEqual(m.sha(path.read_bytes()),row["wav_sha256"])
    def test_no_positive_audio_or_role_substitution(self):
        packet=dummy()
        manifest=json.loads(packet["CORPUS_MANIFEST.json"])
        item=next(x for x in manifest["rows"] if x["cell_id"]=="hello")
        item["requested_text"]="Hey Mero."
        packet["CORPUS_MANIFEST.json"]=json.dumps(manifest).encode()
        with self.assertRaises(ValueError):m.select(packet)
        packet=dummy()
        manifest=json.loads(packet["CORPUS_MANIFEST.json"])
        item=next(x for x in manifest["rows"] if x["cell_id"]=="hello")
        item["role"]="locked-test"
        packet["CORPUS_MANIFEST.json"]=json.dumps(manifest).encode()
        with self.assertRaises(ValueError):m.select(packet)

if __name__=="__main__":unittest.main()
