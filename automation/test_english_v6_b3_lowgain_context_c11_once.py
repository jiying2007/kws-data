"""V6 B3 low-gain / predecessor experiment: deterministic source-only regression."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
mod=importlib.util.spec_from_file_location(
    "v6_b3_gain",HERE/"english_v6_b3_lowgain_context_c11_once.py")
m=importlib.util.module_from_spec(mod)
mod.loader.exec_module(m)

def fixture():
    raw=struct.pack("<1200h",*[(-20 if i%2 else 21) for i in range(1200)])
    source_sha=hashlib.sha256(raw).hexdigest()
    rows=[{"id":f"merov6dev8-{i}-ordinary-time",
           "mono_class":0,"role":"development",
           "source_group_id":f"src-{i}","source_pcm_sha256":source_sha}
          for i in range(488,496)]
    rows.extend([
        {"id":(m.TARGET if i==0 else f"mero-synthetic-valid-{i}"),
         "mono_class":1,"role":"development",
         "source_group_id":f"src-{488+i}",
         "source_pcm_sha256":source_sha}
        for i in range(6)])
    return rows,raw

class TestB3LowGainSource(unittest.TestCase):
    def test_stable_hypothesis_and_no_promotion(self):
        c=m.source_check()
        self.assertEqual(c["status"],"SOURCE_ONLY_NO_KWS_AUDIO_FORWARD")
        self.assertEqual(c["model_artifact"],14590093798)
        self.assertEqual(c["prior_completed_stream_run"],44658362723)
        self.assertEqual(c["max_acoustic_cases"],46)
        self.assertEqual(c["gain_values"],[1.0,.75,.5])
        self.assertEqual(c["input_qualified_positive"],6)
        self.assertEqual(c["input_qualified_ordinary"],8)
        self.assertEqual(c["excluded_unknown"],18)
        self.assertTrue(c["one_changed_factor_per_case"])
        self.assertFalse(c["model_selection_allowed"])
        self.assertFalse(c["new_training"])
        self.assertFalse(c["new_audio_tts_asr"])
        self.assertFalse(c["cal_locked_read"])
        self.assertFalse(c["physical_ssc305"])
        self.assertFalse(c["product_promotion"])
        self.assertEqual(m.BEFORE,"4995bfe0813f4bd6b65fa75a9c6764b14d64bd4e")

    def test_pcm_gain_is_deterministic_le_and_never_exceeds_original(self):
        raw=struct.pack("<8h",-32768,-7,-3,-1,0,1,3,32767)
        self.assertEqual(m.gain_pcm(raw,1),raw)
        expected=struct.pack("<8h",-16384,-4,-2,0,0,0,2,16384)
        self.assertEqual(m.gain_pcm(raw,.5),expected)
        for gain in (.5,.75,1):
            data=struct.pack("<1200h",*([32767,-32768]*600))
            out=m.gain_pcm(data,gain)
            orig=struct.unpack("<1200h",data)
            derived=struct.unpack("<1200h",out)
            self.assertTrue(all(abs(y)<=abs(x) for x,y in zip(orig,derived)))
        with self.assertRaisesRegex(ValueError,"BOUNDED_PCM16_GAIN_INPUT"):
            m.gain_pcm(struct.pack("<1200h",*([1]*1200)),.6)
        with self.assertRaisesRegex(ValueError,"BOUNDED_PCM16_GAIN_INPUT"):
            m.gain_pcm(b"\0",.5)

    def test_exact_490_and_two_ordinary_predecessors(self):
        rows,raw=fixture()
        groups,by=m.source_rows(rows)
        self.assertEqual(len(groups["mixed"]),14)
        self.assertIn(m.TARGET,by)
        self.assertEqual(len(by),14)
        condition_mixed,start,n=m.context_pcm(by,lambda r:raw,"exact_prev488")
        self.assertEqual(start,16000+len(raw)//2+16000)
        self.assertEqual(n,len(raw)//2)
        self.assertEqual(len(condition_mixed)//2,start+n+m.TRAILING)
        positions={}
        for case in m.CONTEXTS:
            pcm,s,length=m.context_pcm(by,lambda r:raw,case)
            positions[case]=s
            self.assertEqual(length,1200)
            self.assertEqual(len(pcm)//2,s+length+m.TRAILING)
        self.assertEqual(positions["aligned_silence"],48000)
        self.assertEqual(positions["aligned_prev488"],48000)
        self.assertEqual(positions["aligned_prev489"],48000)
        self.assertEqual(positions["exact_prev488"],33200)
        raw_bad=struct.pack("<1200h",*([10]*1200))
        with self.assertRaisesRegex(ValueError,"EXACT_490_ORIGINAL_SOURCE_SHA"):
            m.context_pcm(by,lambda r:raw_bad,"aligned_silence")
        with self.assertRaisesRegex(ValueError,"DECLARED_490_CONTEXT_CASE"):
            m.context_pcm(by,lambda r:raw,"unknown")

    def test_excluded_unknown_fails_closed(self):
        rows,_=fixture()
        rows[-1]["mono_class"]=None
        with self.assertRaises(ValueError):
            m.source_rows(rows)
        rows,_=fixture()
        rows[0]["role"]="locked-test"
        with self.assertRaises(ValueError):
            m.source_rows(rows)
        rows,_=fixture()
        del rows[0]
        with self.assertRaises(ValueError):
            m.source_rows(rows)

    def test_guard_blocks_pr_without_network(self):
        with patch.dict("os.environ",{"GITHUB_ACTIONS":"true",
               "GITHUB_REPOSITORY":"jiying2007/kws-data",
               "GITHUB_EVENT_NAME":"pull_request",
               "GITHUB_REF":"refs/pull/279/merge",
               "GITHUB_RUN_ATTEMPT":"1"},clear=True):
            with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_ONLY"):
                m.guard()
        with self.assertRaisesRegex(ValueError,"GITHUB_READ_ONLY_PROVENANCE_REQUIRED"):
            m.verify_prior("")

if __name__=="__main__":
    unittest.main()
