"""No model calls. Enforce immutability and single-keyword source screening."""
from __future__ import annotations
from collections import Counter
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
   "english_v5_mono_audit_test", HERE/"english_v5_mero_mono_source_audit.py")
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def example():
    baseline=[]
    seq=0
    for role,n,extra in (("train",10,4),("development",5,1)):
        for idx in range(n):
            for label in ([0,2,2] if idx<extra else [0,2]):
                seq+=1
                baseline.append({
                    "id":f"a{seq}","corpus":"fresh24",
                    "role":role,"source_group_id":f"{role}_{idx}",
                    "source_pcm_sha256":f"{seq:064x}",
                    "source_wav_sha256":f"{seq:064x}",
                    "samples":16000,"weak_class":label,
                    "requested_class":label,"tier":"silver",
                    "mono_class":1 if label==2 else 0,
                    "ctc_targets":[1,2,3] if label==2 else [],
                    "human_gold":False,"training_allowed_in_original":False})
    assert len(baseline)==35
    return baseline


class MonoScreenAudit(unittest.TestCase):
    def test_source_contract_preserves_solo_target_and_prior_artifacts(self):
        state=s.source_check()
        self.assertEqual(state["keyword"],"Hey Mero")
        self.assertEqual(state["admitted_original_labels"],[0,2])
        self.assertEqual(state["audited_original_pcm_count"],630)
        self.assertEqual(state["old_calibration_and_exposed_locked_pcm"],
                         "NOT_OPENED")
        self.assertFalse(state["training"])
        self.assertFalse(state["quality_promotion"])

    def test_existing_35_are_unchanged_and_new_mono_only_group_recovers(self):
        base=example()
        new=[]
        for role in ("train","development"):
            for klass in (0,2):
                seq=len(base)+len(new)+1
                new.append({"id":f"added_{seq}","corpus":"newtrain24",
                            "role":role,"source_group_id":role+"_mono",
                            "source_pcm_sha256":f"{seq:064x}",
                            "source_wav_sha256":f"{seq:064x}",
                            "samples":16000,"weak_class":klass,
                            "requested_class":klass,"tier":"weak",
                            "human_gold":False,
                            "training_allowed_in_original":False})
        result,info=s.select_mono(base+new,base)
        self.assertEqual(len(result),39)
        self.assertEqual(info["previously_selected_original_rows"],35)
        self.assertEqual(info["newly_recovered_from_original_independent_review"],4)
        self.assertEqual(info["mono_source_groups"],{"train":11,"development":6})
        self.assertEqual(info["additional_source_groups_by_role"],
                         {"train":1,"development":1})
        self.assertTrue(info["no_unknown_relabel"])
        self.assertTrue(info["no_old_other_word_becomes_negative"])
        self.assertFalse(info["release_promotion"])

    def test_retired_keyword_and_unknown_never_become_mono_negatives(self):
        base=example()
        ref=base[0]
        added=[]
        for label in (1,None):
            row={**ref,"id":"new"+str(label),
                 "source_pcm_sha256":"e"*64 if label is None else "f"*64,
                 "source_group_id":"train_nonmono",
                 "requested_class":label,"weak_class":label,
                 "tier":"unknown" if label is None else "weak"}
            row.pop("mono_class",None)
            row.pop("ctc_targets",None)
            added.append(row)
        chosen,info=s.select_mono(base+added,base)
        self.assertEqual(len(chosen),35)
        self.assertEqual(info["newly_recovered_from_original_independent_review"],0)
        self.assertEqual(Counter(x["mono_class"] for x in chosen),
                         Counter({1:20,0:15}))

    def test_do_not_rewrite_previous_silver_or_role_or_pcm(self):
        base=example()
        for key,bad in (("tier","unknown"),("role","development"),
                        ("source_pcm_sha256","e"*64),
                        ("requested_class",2)):
            changed=[dict(x) for x in base]
            changed[0][key]=bad
            if key=="tier":
                changed[0]["weak_class"]=None
            with self.subTest(field=key):
                with self.assertRaises(ValueError):
                    s.select_mono(changed,base)

    def test_never_let_holdout_or_duplicate_audio_enter(self):
        base=example()
        for key,value in (("role","locked-test"),
                          ("source_pcm_sha256",base[2]["source_pcm_sha256"]),
                          ("id",base[2]["id"])):
            other=[dict(x) for x in base]
            other[0][key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    s.select_mono(other,base)

    def test_code_only_reads_current_train_dev_wav_and_makes_no_models(self):
        code=(HERE/"english_v5_mero_mono_source_audit.py").read_text()
        self.assertIn('if raw["role"] not in SPLITS:',code)
        self.assertIn('continue',code)
        self.assertIn('if tier in ("silver","weak")',code)
        self.assertIn('v5.binding.read_wav',code)
        self.assertNotIn('torch.optim',code)
        self.assertNotIn('model.forward(',code)
        self.assertNotIn('old_calibration_or_locked_audio_for_training',code)


if __name__=="__main__":
    unittest.main()
