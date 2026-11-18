"""Offline five-blind Melo source/label tests; no network, audio or model calls."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location("melo_five_test_"+name,HERE/(name+".py"))
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
worker=load("english_melo_cmu_observe_once")
screen=load("english_melo_cmu_combine_once")

def texts(text):
    return {name:text for name in ("qwen","whisper","parakeet")}
def phones(tokens):
    return {name:list(tokens) for name in ("xlsr","lv60")}

M_PHONES=("h","eɪ","m","ɛ","r","oʊ")
V_PHONES=("h","eɪ","v","oʊ","s","ə","n")
ORD_PHONES=("w","ɒ","t","t","aɪ","m","ɪ","z","ɪ","t")
def rows():
    return [{**x,"wav_sha256":"a"*64,"pcm_sha256":"b"*64,
             "samples":16000,"positive_duration_pass":True}
            for x in worker.m.planned()]

class FiveNativeReview(unittest.TestCase):
    def test_source_pins_no_numeric_preflight(self):
        self.assertEqual(worker.source_check()["source_rows"],25)
        self.assertFalse(worker.source_check()["training_allowed"])
        self.assertEqual(screen.source_check()["screen_rule"],
                         "english-synthetic-clone12-dualphone-tripleasr-admission-v1")
        self.assertEqual(worker.ARTIFACT_ID,14411564380)
        self.assertEqual(worker.SOURCE_RUN,44271169127)

    def test_guard_exact_main_only_and_no_retry(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":worker.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"555001",
             "GITHUB_SHA":"a"*40}
        event={"before":worker.BEFORE,"after":"a"*40,
               "ref":"refs/heads/main",
               "repository":{"full_name":worker.REPO,"private":False},
               "forced":False,"deleted":False}
        self.assertEqual(worker.guard(env,event,worker.BEFORE)["source_run"],
                         worker.SOURCE_RUN)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            worker.guard(env,event,worker.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1";event["repository"]["private"]=True
        with self.assertRaises(ValueError):
            worker.guard(env,event,worker.BEFORE)
        event["repository"]["private"]=False
        with self.assertRaises(ValueError):
            worker.guard(env,event,"b"*40)

    def test_canonical_positive_and_ordinary_machine_tiers(self):
        self.assertEqual(screen.one_decision(
            {"requested_class":2,"positive_duration_pass":True,"samples":16000},
            texts("Hey Mero"),phones(M_PHONES))[:2],("silver",2))
        self.assertEqual(screen.one_decision(
            {"requested_class":1,"positive_duration_pass":True,"samples":16000},
            texts("Hey Vosen"),phones(V_PHONES))[:2],("silver",1))
        self.assertEqual(screen.one_decision(
            {"requested_class":0,"positive_duration_pass":True,"samples":16000},
            texts("What time is it?"),phones(ORD_PHONES))[:2],("silver",0))

    def test_negative_controls_never_turn_positive_or_background(self):
        for requested,heard in ((None,"Hey Meero"),(None,"Hey Vozen")):
            tier,label,why=screen.one_decision(
                {"requested_class":requested,"positive_duration_pass":True,"samples":16000},
                texts(heard),phones(M_PHONES))
            self.assertEqual((tier,label,why),
                ("unknown",None,"NONCANONICAL_CONTRAST_NOT_TRAINING_NEGATIVE"))
        tier,label,why=screen.one_decision(
            {"requested_class":2,"positive_duration_pass":True,"samples":16000},
            texts("Hey Meero"),phones(M_PHONES))
        self.assertEqual(tier,"unknown")
        self.assertIsNone(label)

    def test_uncropped_duration_veto_and_two_of_three_weak(self):
        row={"requested_class":2,"positive_duration_pass":False,"samples":32000}
        self.assertEqual(screen.one_decision(row,texts("Hey Mero"),
                         phones(M_PHONES))[:2],("unknown",None))
        weak={"qwen":"Hey Mero","whisper":"Hey Mero",
              "parakeet":"Mero"}
        self.assertEqual(screen.one_decision(
            {"requested_class":2,"positive_duration_pass":True,"samples":16000},
            weak,phones(M_PHONES))[:2],("weak",2))
        self.assertEqual(screen.one_decision(
            {"requested_class":2,"positive_duration_pass":True,"samples":16000},
            {"qwen":"Hey Mero","whisper":"Hey Mero",
             "parakeet":"Hey Nero"},phones(M_PHONES))[:2],("unknown",None))

    def test_all_five_synthetic_styles_cannot_claim_6_2_2_2(self):
        roster=rows()
        asrs={};phonetics={}
        for row in roster:
            target=row["requested_class"]
            if target==2:
                asrs[row["id"]]=texts("Hey Mero")
                phonetics[row["id"]]=phones(M_PHONES)
            elif target==1:
                asrs[row["id"]]=texts("Hey Vosen")
                phonetics[row["id"]]=phones(V_PHONES)
            elif target==0:
                asrs[row["id"]]=texts("What time is it?")
                phonetics[row["id"]]=phones(ORD_PHONES)
            else:
                asrs[row["id"]]=texts("Hey Meero")
                phonetics[row["id"]]=phones(M_PHONES)
        outcome=screen.evaluate(roster,asrs,phonetics)
        self.assertEqual(outcome["complete_synthetic_styles"],5)
        self.assertEqual(outcome["tier_counts"],{"silver":15,"unknown":10})
        self.assertEqual(outcome["status"],"PILOT_ALL_FIVE_STYLES_SUPPORTED")
        self.assertFalse(outcome["training_snapshot_allowed"])
        self.assertFalse(outcome["twelve_independent_source_groups_satisfied"])
        self.assertEqual(outcome["role_coverage"],
                         {"train":0,"development":0,"calibration":0,"locked-test":0})
        change=next(r for r in roster if r["requested_class"]==2)
        asrs[change["id"]]["whisper"]=None
        outcome=screen.evaluate(roster,asrs,phonetics)
        self.assertEqual(outcome["complete_synthetic_styles"],4)
        self.assertEqual(outcome["status"],"PILOT_SOURCE_QUALITY_INCOMPLETE")

if __name__=="__main__":unittest.main()
