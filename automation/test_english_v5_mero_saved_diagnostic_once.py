"""No-model, no-inference tests for the original V5 240-step saved-weight diagnosis."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "english_v5_saved_diag_test",
    HERE/"english_v5_mero_saved_diagnostic_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SavedV5Study(unittest.TestCase):
    def test_original_frozen_model_is_not_retrained(self):
        d=m.source_check()
        self.assertEqual(d["original_model_artifact"],14496732088)
        self.assertEqual(d["max_new_KWS_forwards"],17)
        self.assertEqual(d["new_optimizer_updates"],0)
        self.assertEqual(d["new_TTS_ASR_CTC_phone_forwards"],0)
        self.assertFalse(d["old_calibration_or_locked_replay"])
        self.assertEqual(m.ORIGINAL_RUN,44453447586)
        self.assertEqual(m.MODEL_ARTIFACT["size"],505111)

    def test_exact_previous_successor_first_attempt(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"444700001"}
        event={"before":m.BEFORE,"after":"a"*40,
               "ref":"refs/heads/main","forced":False,
               "deleted":False,"repository":{"full_name":m.REPO,
                                             "private":False}}
        self.assertEqual(m.guard(env,event,m.BEFORE)["original_model_run"],
                         m.ORIGINAL_RUN)
        with self.assertRaises(ValueError):
            m.guard({**env,"GITHUB_RUN_ATTEMPT":"2"},event,m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env,{**event,"before":"b"*40},m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env,event,"b"*40)

    def test_diagnostic_train_subset_uses_distinct_source_groups(self):
        rows=[]
        for clazz in (0,1):
            for group in range(8):
                rows.append({"id":f"{clazz}-{group}",
                   "source_group_id":f"{clazz}/speaker-{group}",
                   "role":"train","mono_class":clazz})
        chosen=m.choose_train_diagnostics(rows)
        self.assertEqual(len(chosen),6)
        self.assertEqual(len(set(x["source_group_id"] for x in chosen)),6)
        self.assertEqual(sum(x["mono_class"]==1 for x in chosen),3)
        rows=[x for x in rows if x["mono_class"]==0]
        with self.assertRaises(ValueError):
            m.choose_train_diagnostics(rows)

    def test_missing_zip_cannot_invent_weights_or_dev_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=Path(tmp)/"model.zip"
            with self.assertRaises(ValueError):
                m.audit_model(x)
            x.write_bytes(b"FORGED_MODEL_NOT_SIGNED")
            with self.assertRaises(ValueError):
                m.audit_model(x)

    def test_source_contains_no_new_training_or_old_locked_inference(self):
        t=(HERE/"english_v5_mero_saved_diagnostic_once.py").read_text()
        self.assertNotIn("optimizer.step(",t)
        self.assertNotIn("torch.onnx.export(",t)
        self.assertIn("torch.load(",t)


if __name__=="__main__":unittest.main()
