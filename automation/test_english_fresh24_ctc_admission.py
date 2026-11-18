"""Offline prospective-rule tests; never import or run TTS / ASR models."""
import importlib.util
import json
from pathlib import Path
import struct
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("fresh24",HERE/"english_fresh24_ctc_admission.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_vocab():
    names=["_","h","eɪ","m","ɛ","r","oʊ","iː","i","ɪ",
           "v","s","ə","n","z","b","w","o","ɔ","ɐ","ʌ","oː"]
    names.extend("fake_"+str(i) for i in range(392-len(names)))
    return {name:i for i,name in enumerate(names)}


VOCAB=make_vocab()


def logits(phones):
    rows=[]
    for token in phones:
        x=[-8.0]*392
        x[VOCAB[token]]=8.0
        rows.extend(x)
    return struct.pack("<"+str(len(rows))+"f",*rows)


def capture(phones):
    raw=logits(phones)
    return {"logits":raw,"vocab":VOCAB,"sha256":m.sha(raw),"pad_token_id":0}


def evidence(phones):
    return {"xlsr":capture(phones),"lv60":capture(phones)}


def texts(*values):
    return dict(zip(m.TEXT_MODELS,values))


class Fresh24(unittest.TestCase):
    def test_source_24_is_not_recycled_and_roles_frozen(self):
        plan=json.loads(m.PLAN_PATH.read_bytes())
        value=m.validate_protocol(plan)
        self.assertEqual(value["groups"],24)
        self.assertEqual(value["roles"],m.FROZEN_SOURCE_ROLES)
        self.assertEqual(value["min_groups"],m.MIN_QUALIFIED)
        self.assertFalse(value["training_allowed"])
        self.assertEqual(value["planned_new_clone_calls"],168)
        self.assertEqual(len(set(v["source_group_id"] for v in plan["voice_groups"])),24)
        plan["voice_groups"][0]["role"]="locked-test"
        with self.assertRaises(ValueError):
            m.validate_protocol(plan)
    def test_exact_lexicon_and_old_vowel_are_contrasts(self):
        positive,contrast=m.paths(2)
        self.assertIn(("h","eɪ","m","ɛ","r","oʊ"),positive)
        self.assertIn(("h","eɪ","m","iː","r","oʊ"),contrast)
        self.assertNotIn(("h","eɪ","m","iː","r","oʊ"),positive)
        positive,contrast=m.paths(1)
        self.assertIn(("h","eɪ","v","oʊ","s","ə","n"),positive)
        self.assertIn(("h","eɪ","v","oʊ","z","ə","n"),contrast)
        self.assertNotIn(("h","eɪ","v","oʊ","z","ə","n"),positive)
    def test_ctc_blank_and_duplicate_contract(self):
        frames=m.verified_phone_input(logits(("h","_","h")),VOCAB)
        self.assertGreater(m.ctc_sequence_logp(frames,VOCAB,("h","h")),
                           m.ctc_sequence_logp(frames,VOCAB,("h",)))
        self.assertIsNone(m.ctc_sequence_logp(frames,VOCAB,("nonexistent",)))
        with self.assertRaises(ValueError):
            m.verified_phone_input(b"\x00"*7,VOCAB)
    def test_observed_phone_margins_are_not_labels_alone(self):
        mero=("h","eɪ","m","ɛ","r","oʊ")
        vosen=("h","eɪ","v","oʊ","s","ə","n")
        self.assertGreater(m.margin(m.verified_phone_input(logits(mero),VOCAB),VOCAB,2),0)
        self.assertGreater(m.margin(m.verified_phone_input(logits(vosen),VOCAB),VOCAB,1),0)
        row={"requested_class":2,"pcm_samples":16000,"positive_duration_pass":True}
        decision=m.decide(row,texts("Hey Mero","Hey Merrow","Hey Mero"),
                          evidence(mero))
        self.assertEqual((decision["tier"],decision["class"]),("silver",2))
        self.assertFalse(decision["training_allowed"])
        row["pcm_samples"]=30000
        self.assertEqual(m.decide(row,texts("Hey Mero","Hey Mero","Hey Mero"),
                         evidence(mero))["tier"],"unknown")
    def test_old_ipa_and_voiced_contrasts_fail_closed(self):
        row={"requested_class":2,"pcm_samples":16000,"positive_duration_pass":True}
        mero_old=("h","eɪ","m","iː","r","oʊ")
        self.assertEqual(m.decide(row,texts("Hey Mero","Hey Mero","Hey Mero"),
                    evidence(mero_old))["reason"],"DUAL_PHONE_CONTRAST_NOT_POSITIVE")
        self.assertEqual(m.decide(row,texts("Hey Meero","Hey Mero","Hey Mero"),
                    evidence(mero_old))["reason"],"OLD_OR_VOICED_PRONUNCIATION_OBSERVED")
        row["requested_class"]=1
        voiced=("h","eɪ","v","oʊ","z","ə","n")
        self.assertEqual(m.decide(row,texts("Hey Vosen","Hey Vosen","Hey Vosen"),
                    evidence(voiced))["reason"],"DUAL_PHONE_CONTRAST_NOT_POSITIVE")
        self.assertEqual(m.decide(row,texts("Hey Vosen","Hey Vosen","Hey Bosen"),
                    evidence(("h","eɪ","v","oʊ","s","ə","n")))["reason"],
                    "CONTRADICTORY_ASR_NAME")
    def test_nearword_controls_are_not_negative_examples(self):
        row={"requested_class":None,"pcm_samples":16000}
        self.assertEqual(m.decide(row,texts("Hey Meero","Hey Meero","Hey Meero"),
                    evidence(("h","eɪ","m","iː","r","oʊ")))["reason"],
                    "CONTRAST_DIAGNOSTIC_NEVER_TRAIN_NEGATIVE")
        row["requested_class"]=0
        self.assertEqual(m.decide(row,texts("Hey Meero","What time is it?","What time is it?"),
                    evidence(("w","o","s")))["tier"],"unknown")
        self.assertEqual(m.decide(row,texts("What time is it?","What time is it?","What time is it?"),
                    evidence(("w","o","s")))["tier"],"silver")
    def test_partial_missing_posterior_never_promotes(self):
        row={"requested_class":2,"pcm_samples":16000,"positive_duration_pass":True}
        raw=logits(("h","eɪ","m","ɛ","r","oʊ"))
        bad={"logits":raw,"vocab":VOCAB,"sha256":"0"*64,"pad_token_id":0}
        packets={"xlsr":bad,"lv60":capture(("h","eɪ","m","ɛ","r","oʊ"))}
        self.assertEqual(m.decide(row,texts("Hey Mero","Hey Mero","Hey Mero"),
            packets)["reason"],"INVALID_OR_INCOMPLETE_NATIVE_PHONES")
        self.assertEqual(m.decide(row,{"qwen":"Hey Mero"},
            evidence(("h","eɪ","m","ɛ","r","oʊ")))["tier"],"unknown")


if __name__=="__main__":
    unittest.main()
