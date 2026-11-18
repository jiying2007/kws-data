"""Independent source-policy tests; no audio download, TTS, ASR or model call."""
from collections import Counter
import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).with_name("english_clone12_admission_policy.py")
spec=importlib.util.spec_from_file_location("future_dualphone_gate",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

M={"xlsr":["h","eɪ","m","ɛ","ɹ","ɔ"],
   "lv60":["h","eɪ","m","ɛ","ɹ","oʊ"]}
V={"xlsr":["h","eɪ","v","oː","s","ə","n"],
   "lv60":["h","eɪ","v","oʊ","s","ə","n"]}
N={"xlsr":["w","ʌ","t","t","aɪ","m"],
   "lv60":["w","ʌ","t","t","aɪ","m"]}
def texts(s):return {x:s for x in m.TEXT_MODELS}

def roster():
    groups={"train":6,"development":2,"calibration":2,"locked-test":2}
    rows=[];asr={};phone={}
    for role,n in groups.items():
        for i in range(n):
            group=f"synthetic:{role}:{i}"
            for cell,klass,txt,ph in [
                ("mero-marrow",2,"Hey marrow.",M),
                ("mero-merrow",2,"Hey merrow.",M),
                ("vosen-vohsen",1,"Hey Vohsen.",V),
                ("vosen-name",1,"Hey Vosen.",V),
                ("ordinary-time",0,"What time is it?",N)]:
                sid=f"{group}/{cell}"
                rows.append({"id":sid,"source_group_id":group,"role":role,
                    "cell_id":cell,"requested_brand_class":klass,
                    "positive_duration_pass":True,
                    "samples":24000,"pcm_sha256":sid,"wav_sha256":sid})
                asr[sid]=texts(txt);phone[sid]=ph
    return rows,asr,phone

class StrictBlindSource(unittest.TestCase):
    def test_two_real_phone_families_support_target_only_with_three_asr(self):
        self.assertEqual(m.blind(texts("Hey marrow."),M)[:2],("silver",2))
        self.assertEqual(m.blind(texts("Hey Vohsen."),V)[:2],("silver",1))
        partial=texts("Hey Vohsen.");partial["whisper"]="Nothing"
        self.assertEqual(m.blind(partial,V)[:2],("weak",1))
    def test_old_meero_and_voiced_vosen_rejected(self):
        old=dict(M);old["lv60"]=["h","eɪ","m","iː","ɹ","oʊ"]
        self.assertEqual(m.blind(texts("Hey marrow."),old)[0],"unknown")
        voiced=dict(V);voiced["xlsr"]=["h","eɪ","v","oː","z","ə","n"]
        self.assertEqual(m.blind(texts("Hey Vohsen."),voiced)[0],"unknown")
        self.assertEqual(m.blind(texts("Hey Meero."),M)[0],"unknown")
        self.assertEqual(m.blind(texts("Hey Vozen."),V)[0],"unknown")
    def test_contradictory_third_asr_does_not_permit_weak_label(self):
        t=texts("Hey marrow.");t["parakeet"]="Hey Nero!"
        self.assertEqual(m.blind(t,M)[0],"unknown")
        t=texts("Hey Vohsen.");t["parakeet"]="Hey Vozen!"
        self.assertEqual(m.blind(t,V)[0],"unknown")
    def test_unknown_never_becomes_background(self):
        self.assertEqual(m.blind(texts("Hey Nero."),N)[0],"unknown")
        self.assertEqual(m.blind({"qwen":"hello"},N)[0],"unknown")
        self.assertEqual(m.blind(texts("What time is it?"),N)[:2],("silver",0))
    def test_all_twelve_groups_cannot_be_inferred_from_requested_labels(self):
        rows,asr,phones=roster()
        complete=m.evaluate(rows,asr,phones)
        self.assertEqual(complete["status"],"PASS_CANONICAL_SOURCE_COVERAGE")
        self.assertEqual(complete["complete_groups_by_role"],m.ROLES)
        self.assertTrue(complete["training_snapshot_allowed"])
        self.assertTrue(all(not r["training_allowed"] for r in complete["rows"]))
        # One locked-test group missing an independent phone veto => STOP,
        # even if its generation requests still claim canonical classes.
        one=next(r for r in rows if r["role"]=="locked-test" and
                 r["cell_id"]=="vosen-vohsen")
        two=next(r for r in rows if r["source_group_id"]==one["source_group_id"] and
                 r["cell_id"]=="vosen-name")
        phones[one["id"]]={**V,"lv60":["h","eɪ","v","oʊ","z","ə","n"]}
        phones[two["id"]]={**V,"lv60":["h","eɪ","v","oʊ","z","ə","n"]}
        missing=m.evaluate(rows,asr,phones)
        self.assertEqual(missing["status"],"STOP_CANONICAL_SOURCE_COVERAGE")
        self.assertEqual(missing["complete_groups_by_role"]["locked-test"],1)
        self.assertFalse(missing["training_snapshot_allowed"])
        self.assertEqual(missing["denominator"],60)
    def test_duration_is_not_fixed_by_cropping(self):
        rows,asr,phones=roster()
        sid=rows[0]["id"]
        rows[0]["positive_duration_pass"]=False
        rows[0]["samples"]=50000
        x=m.evaluate(rows,asr,phones)
        self.assertEqual(next(r for r in x["rows"] if r["id"]==sid)["reason"],
                         "POSITIVE_DURATION_TOO_LONG_NO_CROP")

if __name__=="__main__":unittest.main()
