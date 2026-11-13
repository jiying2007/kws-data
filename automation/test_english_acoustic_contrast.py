"""Source-only and invented-audio regressions for real 63-WAV DTW review."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_acoustic_contrast as acoustics
import english_seven_voice_tts_once as source_tts
import multilingual_campaign as market


class AcousticContrastTests(unittest.TestCase):
    def test_frozen_source_contract_and_no_model_calls(self):
        pin = acoustics.claim()
        self.assertEqual(pin["source_audio_count"],63)
        self.assertEqual(pin["source_asr_observations"],189)
        self.assertEqual(pin["source_review_artifact_id"],13965259534)
        self.assertEqual(pin["new_tts_calls"],0)
        self.assertEqual(pin["new_asr_calls"],0)
        self.assertEqual(pin["new_neural_model_calls"],0)
        self.assertFalse(pin["training_admitted"])
        self.assertFalse(pin["human_gold"])
        self.assertEqual(acoustics.main([]),0)

    def test_main_only_no_retry_or_untrusted_branch(self):
        env={"GITHUB_REPOSITORY":"jiying2007/kws-data",
             "GITHUB_EVENT_NAME":"push",
             "GITHUB_REF":"refs/heads/main","GITHUB_RUN_ATTEMPT":"1"}
        with mock.patch.dict(os.environ,env,clear=True):
            acoustics.gate()
            for key,value in [
                ("GITHUB_REPOSITORY","attacker/kws-data"),
                ("GITHUB_EVENT_NAME","pull_request"),
                ("GITHUB_REF","refs/heads/feature"),
                ("GITHUB_RUN_ATTEMPT","2"),
            ]:
                with mock.patch.dict(os.environ,{**env,key:value},clear=True):
                    with self.assertRaisesRegex(ValueError,
                         "ACOUSTIC_CONTRAST_PROTECTED_MAIN_ONCE_ONLY"):
                        acoustics.gate()

    def test_mutated_one_time_trigger_refused(self):
        base=acoustics.claim()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"tampered-trigger.json"
            with mock.patch.object(acoustics,"TRIGGER",path):
                path.write_bytes(market.canonical(base)+b"\n")
                self.assertEqual(acoustics.claim(),base)
                for key,value in [
                    ("training_admitted",True),
                    ("source_review_zip_sha256","f"*64),
                    ("new_tts_calls",1),
                    ("source_audio_count",64),
                ]:
                    changed=copy.deepcopy(base)
                    changed[key]=value
                    path.write_bytes(market.canonical(changed)+b"\n")
                    with self.assertRaisesRegex(ValueError,
                         "ACOUSTIC_CONTRAST_TRIGGER_DRIFT"):
                        acoustics.claim()

    def test_dtw_is_symmetric_and_bounded(self):
        import numpy as np
        wave_a=(np.arange(70,dtype=float)[:,None]/70 *
                np.arange(1,41,dtype=float)[None,:] / 40)
        wave_b=np.roll(wave_a,2,axis=0)
        self.assertEqual(acoustics.dtw(wave_a,wave_a),0.0)
        self.assertGreater(acoustics.dtw(wave_a,wave_b),0)
        self.assertEqual(acoustics.dtw(wave_a,wave_b),
                         acoustics.dtw(wave_b,wave_a))
        with self.assertRaisesRegex(ValueError,"ACOUSTIC_DTW_SHAPE"):
            acoustics.dtw(np.zeros((0,40)),wave_a)

    def test_real_geometry_from_invented_pcm_not_asr(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"invented.wav"
            pcm=np.asarray(
                [int(9000*math.sin(2*math.pi*210*i/16000)) for i in range(16000)],
                dtype="<i2")
            with wave.open(str(path),"wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            feats,meta=acoustics.features(path)
            self.assertEqual(feats.shape[1],40)
            self.assertGreater(meta["active_feature_frames"],50)
            self.assertEqual(meta["audio_frames"],16000)
            self.assertEqual(meta["pcm_sha256"],
                             hashlib.sha256(pcm.tobytes()).hexdigest())
            self.assertFalse((Path(tmp)/"any-asr-result").exists())
            silent=Path(tmp)/"silent.wav"
            with wave.open(str(silent),"wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(b"\0\0"*16000)
            with self.assertRaisesRegex(ValueError,"ACOUSTIC_NO_ACTIVE_SPEECH"):
                acoustics.features(silent)

    def test_seven_synthetic_voices_are_not_seven_humans_or_training(self):
        import numpy as np
        by_voice={}
        ids={}
        for i,(voice,_) in enumerate(source_tts.VOICES):
            t=np.linspace(0,4,65)
            b={}
            for j,prompt in enumerate(acoustics.PROMPTS):
                noise=np.random.default_rng(i*20+j).normal(0,.02,size=(65,40))
                base=(np.sin((t[:,None]+j/60)*(1.05+j*.022)) *
                      (np.arange(1,41)[None,:]/40))
                b[prompt]=base+noise
            by_voice[voice]=b
            ids[voice]={p:"fixture-"+voice+"-"+str(j)
                        for j,p in enumerate(acoustics.PROMPTS)}
        report=acoustics.contrast(by_voice,ids)
        self.assertEqual(len(report["by_voice"]),7)
        self.assertEqual(set(report["by_contrast"]),
                         {c[0] for c in acoustics.CONTRASTS})
        for name,r in report["by_contrast"].items():
            self.assertEqual(r["source_families"],7)
            self.assertEqual(r["independent_human_speaker_count"],0)
            self.assertLessEqual(r["nearphone_not_farther_than_same"],7)
        for voice in report["by_voice"]:
            self.assertFalse(voice["human_gold"])
            self.assertFalse(voice["training_admitted"])
            for comparison in voice["comparisons"].values():
                self.assertFalse(comparison["trusted_negative"])
                self.assertFalse(comparison["training_admitted"])
                self.assertTrue(comparison["diagnostic_only"])

    def test_wrong_zip_identity_rejected_before_any_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            corrupt=root/"incorrect.zip"
            corrupt.write_bytes(b"not a real retained ZIP")
            with self.assertRaisesRegex(ValueError,
                    "ASR_REVIEW_ARTIFACT_DIGEST_DRIFT"):
                acoustics.load_exact_sources(corrupt,corrupt,root/"unpacked")

    def test_exact_isolated_cli_needs_no_numpy_or_download(self):
        result=subprocess.run(
            [sys.executable,"-I","-S","-B",str(Path(acoustics.__file__))],
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            check=False,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)["new_neural_model_calls"],0)
        result=subprocess.run(
            [sys.executable,"-I","-S","-B",str(Path(acoustics.__file__)),
             "--execute"],
            text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            check=False,timeout=20)
        self.assertEqual(result.returncode,2,result.stderr)
        self.assertIn("ACOUSTIC_EXPLICIT_INPUT_OUTPUT_REQUIRED",result.stderr)


if __name__=="__main__":
    unittest.main()
