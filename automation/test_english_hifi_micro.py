"""Model-free boundary tests; never download, synthesize or observe audio."""
import importlib.util
from pathlib import Path
from array import array
import hashlib
import json
import math
import tempfile
import unittest
import wave

HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location("test_contract_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

c=load("english_hifi_micro")
q=load("english_hifi_qwen_once")
ph=load("english_hifi_phones_once")

class Micro(unittest.TestCase):
    def test_immutable_small_roster(self):
        x=c.contract()
        self.assertEqual((len(x["custom"]),len(x["clone"])),(6,2))
        self.assertEqual(x["model_calls"],0)
        self.assertFalse(x["training_allowed"])
        self.assertNotIn("Mero",c.REFERENCE)
        self.assertNotIn("Vosen",c.REFERENCE)
        self.assertEqual(c.CUSTOM[2][-1],"legacy-pronunciation-control")

    def test_guard_rejects_future_replay_or_drift(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":c.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"499001",
             "GITHUB_SHA":"a"*40}
        event={"before":c.PREDECESSOR,"after":"a"*40,
               "ref":"refs/heads/main","forced":False,"deleted":False,
               "repository":{"full_name":c.REPO,"private":False}}
        self.assertEqual(c.guard(env,event,parent=c.PREDECESSOR)["before"],c.PREDECESSOR)
        for change in (
            lambda e,j:e.update(GITHUB_RUN_ATTEMPT="2"),
            lambda e,j:j.update(before="b"*40),
            lambda e,j:j["repository"].update(private=True),
            lambda e,j:j.update(forced=True)
        ):
            e,j=env.copy(),json.loads(json.dumps(event))
            change(e,j)
            with self.assertRaises(ValueError):
                c.guard(e,j,parent=c.PREDECESSOR)

    def test_hf_cache_symlink_readonly_inside_cache_and_reject_escape(self):
        with tempfile.TemporaryDirectory() as t:
            cache=Path(t)/"cache"
            blobs=cache/"blobs"
            snapshot=cache/"snapshots"/("a"*40)
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (blobs/"vocab-content").write_text('{"<pad>":0,"ɛ":1}',encoding="utf8")
            (snapshot/"vocab.json").symlink_to(blobs/"vocab-content")
            self.assertEqual(ph.read_verified_hf_metadata(snapshot,cache,"vocab.json")["ɛ"],1)
            with self.assertRaises(ValueError):
                ph.read_verified_hf_metadata(snapshot,cache,"config.json")
            (snapshot/"vocab.json").unlink()
            elsewhere=Path(t)/"untrusted.json"
            elsewhere.write_text('{"id":9}',encoding="utf8")
            (snapshot/"vocab.json").symlink_to(elsewhere)
            with self.assertRaises(ValueError):
                ph.read_verified_hf_metadata(snapshot,cache,"vocab.json")

    def test_recovery_guard_keeps_first_parent_bound_without_replaying_source(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":c.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"499002",
             "GITHUB_SHA":"a"*40}
        event={"before":c.RECOVERY_PREDECESSOR,"after":"a"*40,
               "ref":"refs/heads/main","forced":False,"deleted":False,
               "repository":{"full_name":c.REPO,"private":False}}
        value=c.guard(env,event,parent=c.RECOVERY_PREDECESSOR,recovery=True)
        self.assertTrue(value["recovery_only"])
        self.assertEqual(value["before"],c.RECOVERY_PREDECESSOR)
        with self.assertRaises(ValueError):
            c.guard(env,event,parent=c.RECOVERY_PREDECESSOR)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            c.guard(env,event,parent=c.RECOVERY_PREDECESSOR,recovery=True)

    def test_ctc_collapses_without_inventing_phones(self):
        vocab={"<pad>":0,"h":1,"eɪ":2,"m":3,"ɛ":4,"r":5,"oʊ":6}
        ids=[1,1,0,2,2,0,3,3,0,4,4,0,5,5,0,6,6]
        self.assertEqual(ph.ctc(ids,vocab,0),["h","eɪ","m","ɛ","r","oʊ"])

    def test_complete_wav_source_and_nonreplacement(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            sig=array("h",(round(9000*math.sin(2*math.pi*i/80)) for i in range(8000)))
            root.joinpath("wav").mkdir()
            entries=[]
            for sid,speaker,text,kind in c.CUSTOM:
                path=root/"wav"/(sid+".wav")
                with wave.open(str(path),"wb") as out:
                    out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000)
                    out.writeframes(sig.tobytes())
                facts=c.inspect_wav(path)
                entries.append({"id":sid,"speaker":speaker,"text":text,
                                "requested_kind":kind,"source_provider":"qwen3-tts",
                                "human_gold":False,"training_allowed":False,**facts})
            c.record(root/"MANIFEST.json",{"schema":c.SCHEMA,
                     "stage":"custom","planned":6,"rows":entries})
            self.assertEqual(len(c.verify_stage(root,"custom")["rows"]),6)
            bad=json.loads((root/"MANIFEST.json").read_text())
            bad["rows"][2]["requested_kind"]="canonical-mero-request"
            (root/"MANIFEST.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):c.verify_stage(root,"custom")

    def test_zero_audio_missing_is_not_success(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            c.record(root/"MANIFEST.json",{"schema":c.SCHEMA,
                     "stage":"custom","planned":6,"rows":[]})
            with self.assertRaises(ValueError):c.verify_stage(root,"custom")

if __name__=="__main__":unittest.main()
