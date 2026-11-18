"""Pure contract tests for frozen original source & five future blind observers."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
def load(n):
    spec=importlib.util.spec_from_file_location("observer_test_"+n,HERE/(n+".py"))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod
obs=load("english_clone12_observer_once")
comb=load("english_clone12_combine_once")

class ObserverIntegrity(unittest.TestCase):
    def test_no_numeric_call_before_source_binding(self):
        self.assertEqual(obs.source_check()["candidate_source_rows"],60)
        self.assertFalse(obs.source_check()["enabled"])
        self.assertEqual(obs.source_check()["new_numeric_calls"],0)
        with self.assertRaises(ValueError):obs.guard()
    def test_all_five_models_one_provider_each(self):
        self.assertEqual(set(obs.PROVIDERS),
                          {"qwen","whisper","parakeet","xlsr","lv60"})
        self.assertEqual(set(obs.PROVIDERS),set(comb.PROVIDERS))
        self.assertEqual(comb.worker.SCHEMA,obs.SCHEMA)
        self.assertEqual(obs.model_id("xlsr")[0],
                         "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        with self.assertRaises(ValueError):obs.model_id("alleged-gold")
    def test_missing_observer_packet_is_unknown_not_success(self):
        source=[{"id":"clip-"+str(i)} for i in range(60)]
        with tempfile.TemporaryDirectory() as t:
            values,facts=comb.read_provider(t,"qwen",source)
        self.assertEqual(len(values),60)
        self.assertTrue(all(x is None for x in values.values()))
        self.assertEqual(facts["status"],"MISSING_OBSERVER_ARTIFACT")
    def test_no_unpinned_artifact_identity(self):
        plan=obs.read_plan()
        self.assertEqual(plan["main_before"],obs.cohort.PREDECESSOR)
        self.assertEqual(plan["source"]["run_id"],44257296925)
        self.assertFalse(plan["requested_text_is_truth"])
        self.assertFalse(plan["training_allowed"])
        self.assertFalse(plan["human_gold"])
        self.assertEqual(plan["retry_count"],0)

if __name__=="__main__":unittest.main()
