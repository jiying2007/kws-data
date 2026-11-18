"""Offline exact CMU pronunciation + provenance tests, no numerical calls."""
import importlib.util
from pathlib import Path
import unittest, tempfile, wave, math, json
from array import array
p=Path(__file__).with_name("english_melo_cmu_once.py")
spec=importlib.util.spec_from_file_location("melo_pilot",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FakeEnglish:
    eng_dict={}
    @classmethod
    def g2p(cls,text):
        phones=["hh","ey"]
        if "Mero" in text:phones+=["m","eh","r","ow"]
        elif "Vosen" in text:phones+=["v","ow","s","ah","n"]
        elif "Meero" in text:phones+=["m","iy","r","ow"]
        elif "Vozen" in text:phones+=["v","ow","z","ah","n"]
        return phones,[1]*len(phones),[1]*len(phones)

class CMUContract(unittest.TestCase):
    def test_cmu_stress_and_contrast(self):
        x=m.source_check()
        self.assertEqual(x["planned_new_tts_calls"],25)
        self.assertFalse(x["old_meero_positive"])
        self.assertEqual(m.CMU["MERO"],[["M","EH1"],["R","OW0"]])
        self.assertEqual(m.CMU["VOSEN"],[["V","OW1"],["S","AH0","N"]])
        self.assertNotEqual(m.CMU["VOSEN"],m.CMU["VOZEN"])
        self.assertNotEqual(m.CMU["MERO"],m.CMU["MEERO"])
    def test_all_25_are_unlabeled_and_old_controls_cannot_be_positives(self):
        rows=m.planned()
        self.assertEqual(len(rows),25)
        self.assertEqual(len({x["id"] for x in rows}),25)
        self.assertEqual(len({x["style_id"] for x in rows}),5)
        self.assertEqual(sum(x["requested_class"] is None for x in rows),10)
        self.assertTrue(all(x["training_allowed"] is False and
                            x["human_gold"] is False for x in rows))
    def test_native_g2p_canonical_phone_path_is_checked(self):
        proof=m.phones_are_locked(FakeEnglish)
        self.assertEqual(proof["Mero"]["requested_critical"],["m","eh","r","ow"])
        self.assertEqual(proof["Vosen"]["requested_critical"],["v","ow","s","ah","n"])
        self.assertEqual(len(FakeEnglish.eng_dict),4)
    def test_pinned_native_hparams_style_lookup_and_fail_closed(self):
        from types import SimpleNamespace

        class NativeHParams:
            # Exact public Melo utils.HParams iteration semantics: keys()
            # exists but __iter__() does not.
            def __init__(self, values):
                self.__dict__.update(values)
            def keys(self):
                return self.__dict__.keys()
            def __getitem__(self, key):
                return getattr(self, key)

        roster=dict(zip(m.SPEAKERS,range(5)))
        hps=SimpleNamespace(data=SimpleNamespace(
            spk2id=NativeHParams(roster),n_speakers=5))
        self.assertEqual(m.native_speaker_ids(hps),roster)
        with self.assertRaises(TypeError):
            set(hps.data.spk2id)
        hps.data.spk2id=NativeHParams(dict(list(roster.items())[:-1]))
        with self.assertRaisesRegex(ValueError,"EXACT_TRAINED_ENGLISH_STYLE_IDS"):
            m.native_speaker_ids(hps)
        hps.data.spk2id=NativeHParams({**roster,"EN-US":1})
        with self.assertRaisesRegex(ValueError,"DISTINCT_NATIVE_ENGLISH_STYLE_IDS"):
            m.native_speaker_ids(hps)

    def test_main_guard_never_replays_or_accepts_other_owner(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPOSITORY,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"55001",
             "GITHUB_SHA":"a"*40}
        event={"before":m.BEFORE,"after":"a"*40,
               "repository":{"full_name":m.REPOSITORY,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(m.guard(env,event,m.BEFORE)["source_parent"],m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):m.guard(env,event,m.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1";event["before"]="b"*40
        with self.assertRaises(ValueError):m.guard(env,event,m.BEFORE)
    def test_full_pcm_source_readback_refuses_promoted_synthesis_request(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"wav").mkdir()
            rows=[]
            for i,requested in enumerate(m.planned()):
                p=root/"wav"/(requested["id"]+".wav")
                pcm=array("h",(int(5000*math.sin(j/8))+(i*3) for j in range(16000))).tobytes()
                with wave.open(str(p),"wb") as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
                    w.writeframes(pcm)
                row={**requested,**m.c.inspect_wav(p),
                     "positive_duration_pass":True}
                rows.append(row)
            packet={"schema":m.SCHEMA,"model":m.MODEL[0],
                    "model_revision":m.MODEL[1],"denominator":25,
                    "training_allowed":False,"rows":rows}
            m.c.record(root/"MANIFEST.json",packet)
            self.assertEqual(m.readback(root)["rows"],25)
            rows[0]["training_allowed"]=True
            (root/"MANIFEST.json").write_text(json.dumps(packet))
            with self.assertRaises(ValueError):m.readback(root)

if __name__=="__main__":unittest.main()
