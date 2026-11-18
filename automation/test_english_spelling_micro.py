"""Offline source/guard/manifest boundaries for new phonetic rendering research."""
from array import array
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import wave
p=Path(__file__).with_name("english_spelling_micro_once.py")
sp=importlib.util.spec_from_file_location("spelling_contract",p)
s=importlib.util.module_from_spec(sp)
sp.loader.exec_module(s)

class Spelling(unittest.TestCase):
    def test_source_locked_without_old_replay(self):
        x=s.source_check()
        self.assertEqual((x["planned_tts"],x["planned_phone"]),(10,20))
        self.assertFalse(x["training_allowed"])
        self.assertEqual(len(set(y[0] for y in s.CUSTOM+s.CLONE)),10)
        self.assertEqual({s.item(y,"custom")["source_group"] for y in s.CUSTOM},
                          {"synthetic-aiden","synthetic-ryan"})
        self.assertEqual(s.item(s.CLONE[0],"clone")["source_group"],"synthetic-aiden")
        self.assertIsNone(s.item(s.CUSTOM[3],"custom")["brand_target"])
    def test_one_main_parent_not_replay(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":s.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"420",
             "GITHUB_SHA":"a"*40}
        ev={"before":s.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
            "repository":{"full_name":s.REPO,"private":False},
            "forced":False,"deleted":False}
        self.assertEqual(s.guard(env,ev,parent=s.BEFORE)["source_predecessor"],s.BEFORE)
        for key,value in (("GITHUB_RUN_ATTEMPT","2"),("GITHUB_REF","refs/heads/other")):
            bad=env.copy();bad[key]=value
            with self.assertRaises(ValueError):s.guard(bad,ev,parent=s.BEFORE)
        ev["before"]="b"*40
        with self.assertRaises(ValueError):s.guard(env,ev,parent=s.BEFORE)
    def test_closing_quality_barrier_with_short_WAV(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"wav").mkdir()
            sig=array("h",(int(5000*math.sin(i/9)) for i in range(8000))).tobytes()
            rows=[]
            for x in s.CUSTOM:
                meta=s.item(x,"custom")
                path=root/"wav"/(meta["id"]+".wav")
                with wave.open(str(path),"wb") as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
                    w.writeframes(sig)
                # Each distinct TTS output must have a distinct PCM identity.
                if len(rows):
                    with wave.open(str(path),"wb") as w:
                        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
                        w.writeframes(sig[:-2]+bytes([len(rows),0]))
                meta.update(s.c.inspect_wav(path))
                meta["source_provider"]="qwen3-tts"
                rows.append(meta)
            s.c.record(root/"MANIFEST.json",{"schema":s.SCHEMA,"stage":"custom",
                        "planned":len(s.CUSTOM),"rows":rows})
            self.assertEqual(s.verify(root,"custom")["rows"],8)
            rows[0]["training_allowed"]=True
            (root/"MANIFEST.json").write_text(json.dumps({"schema":s.SCHEMA,
                      "stage":"custom","planned":8,"rows":rows}))
            with self.assertRaises(ValueError):s.verify(root,"custom")
    def test_unknown_not_promoted(self):
        self.assertFalse(s.item(s.CUSTOM[0],"custom")["training_allowed"])
        self.assertIsNone(s.item(s.CUSTOM[3],"custom")["brand_target"])

if __name__=="__main__":unittest.main()
