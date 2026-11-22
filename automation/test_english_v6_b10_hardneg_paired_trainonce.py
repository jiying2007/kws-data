"""B10 TRAIN-only same-parent experiment source/role and optimizer safety regressions."""
from __future__ import annotations
from collections import Counter
import importlib.util
import os
from pathlib import Path
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("b10_hardneg_source",
                  HERE/"english_v6_b10_hardneg_paired_trainonce.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

TRAIN_GROUPS=("piper-libritts-p7498","piper-libritts-p3905",
             "piper-libritts-p7384","piper-libritts-p2992",
             "piper-libritts-p30","piper-libritts-p497")
def fabricated_data():
    all_groups=TRAIN_GROUPS+tuple("train-old-group-"+str(i) for i in range(29))
    group={kind:{g:[] for g in all_groups} for kind in (0,1)}
    old=[]
    for i in range(79):
        row={"id":"old-"+str(i),"source_pcm_sha256":f"{i+1:064x}",
             "role":"train"}
        old.append(row)
        group[i%2][all_groups[i%35]].append(row)
    source=[];screen=[]
    for i,g in enumerate(TRAIN_GROUPS):
        row={"id":f"b10-train-{448+i}-ordinary-time-hard",
             "role":"train","cell_id":"ordinary-time-hard",
             "source_group_id":g,"speaker_index":448+i,
             "pcm_sha256":f"{200+i:064x}",
             "wav_sha256":f"{300+i:064x}",
             "samples":12000,"training_allowed":False,
             "observed_class":None}
        source.append(row)
        screen.append({**row,"tier":"silver","observed_class":0,
                       "reason":"BLIND_ORDINARY_NO_WAKE","human_gold":False})
    for i in range(6):
        for c in ("mero-fast","ordinary-time"):
            idx=210+i*2+(c=="ordinary-time")
            row={"id":f"b10-dev-{540+i}-{c}",
                 "role":"development","cell_id":c,
                 "source_group_id":"dev-"+str(i),"speaker_index":540+i,
                 "pcm_sha256":f"{idx:064x}",
                 "wav_sha256":f"{idx+100:064x}","samples":10000,
                 "training_allowed":False,"observed_class":None}
            source.append(row)
            observed=(0 if c=="ordinary-time" else
                      2 if i in (2,4) else None)
            screen.append({**row,"observed_class":observed,
                           "tier":"silver" if observed==0 else
                                  "weak" if observed==2 else "unknown",
                           "reason":"FROZEN_SYNTHETIC_FIXTURE",
                           "human_gold":False})
    admission=m.screen.admission(source,screen)
    packet={"schema":m.observer.source_worker.SCHEMA,
            "completed":18,"denominator":18,"rows":source}
    receipt={"schema":m.screen.SCHEMA,
             "status":"PASS_B10_TRAIN_DEV_MACHINE_RESEARCH_GATE_ONLY",
             "source_run":44906200481,"observer_run":44908125673,
             "complete_native_attempts":90,
             "admission":admission,"new_optimizer_updates":0,
             "product_promotion_allowed":False,"rows":screen}
    return old,group,packet,receipt

class B10HardNegativeMatchedTests(TestCase):
    def test_exact_immutable_parent_no_new_model_forward(self):
        p=m.source_check()
        self.assertEqual(p["status"],"SOURCE_ONLY_NO_OPTIMIZER_OR_MODEL_FORWARD")
        self.assertEqual(p["frozen_b9_control_pt_sha256"],
            "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655")
        self.assertEqual(p["original_old_train_rows"],79)
        self.assertEqual(p["new_train_ordinary_hard_rows"],6)
        self.assertEqual(p["same_old_train_groups"],35)
        self.assertEqual(p["max_optimizer_updates"],384)
        self.assertEqual((p["control_hard_negative_loss_weight"],
                          p["challenger_hard_negative_loss_weight"]),(0.0,0.08))
        self.assertEqual(p["new_model_forwards_in_pr"],0)
        self.assertTrue(p["no_old_DEV520_DEV528_training"])
        self.assertFalse(p["model_selection_allowed"])
        self.assertFalse(p["product_promotion_allowed"])

    def test_six_silver_train_and_two_source_disjoint_dev_gate(self):
        old,group,source,receipt=fabricated_data()
        self.assertTrue(receipt["admission"]["B10_research_optimizer_allowed"])
        def read(path):
            return source if str(path).endswith("MANIFEST.json") else receipt
        with mock.patch.object(m,"read",side_effect=read):
            selected,qa=m.training_rows("/fake/source","/fake/screen",old,group)
        self.assertEqual(len(selected),6)
        self.assertEqual({r["source_group_id"] for r in selected},set(TRAIN_GROUPS))
        self.assertEqual(qa["dev_source_metadata_count"],12)
        self.assertEqual(len(qa["qualified_dev_groups"]),2)
        self.assertFalse(qa["dev_pcm_in_gradients"])
        self.assertTrue(all(r["mono_class"]==0 and r["tier"]=="silver" for r in selected))

    def test_no_unknown_weak_or_unqualified_train_new_loss(self):
        old,group,source,receipt=fabricated_data()
        receipt["rows"][0]["tier"]="unknown"
        receipt["rows"][0]["observed_class"]=None
        def read(path):
            return source if str(path).endswith("MANIFEST.json") else receipt
        with mock.patch.object(m,"read",side_effect=read):
            with self.assertRaisesRegex(ValueError,"RECOMPUTED_B10"):
                m.training_rows("/fake/source","/fake/screen",old,group)
        receipt["admission"]=m.screen.admission(source["rows"],receipt["rows"])
        self.assertFalse(receipt["admission"]["B10_research_optimizer_allowed"])
        with mock.patch.object(m,"read",side_effect=read):
            with self.assertRaisesRegex(ValueError,"ACTUAL_PASS_B10"):
                m.training_rows("/fake/source","/fake/screen",old,group)

    def test_fabricated_dev_promotion_detected(self):
        old,group,source,receipt=fabricated_data()
        dev=next(r for r in receipt["rows"] if r["id"]=="b10-dev-545-mero-fast")
        dev["tier"]="weak";dev["observed_class"]=2
        def read(path):
            return source if str(path).endswith("MANIFEST.json") else receipt
        with mock.patch.object(m,"read",side_effect=read):
            with self.assertRaisesRegex(ValueError,"RECOMPUTED_B10"):
                m.training_rows("/fake/source","/fake/screen",old,group)

    def test_pr_or_retry_cannot_reach_optimizer_guard(self):
        with mock.patch.dict(os.environ,{"GITHUB_ACTIONS":"false"},clear=True):
            with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_ONLY_NO_REPLAY"):
                m.guard()

    def test_no_dev_cache_or_model_promotion_calls_in_source_stage(self):
        script=(HERE/"english_v6_b10_hardneg_paired_trainonce.py").read_text()
        self.assertIn("rows,dev,source,groups=b6.source_audit",script)
        self.assertIn('extra_cache=cache_new_train(',script)
        self.assertIn('if weight:',script)
        self.assertNotIn("retrain_on_dev",script)
        self.assertIn('"dev_model_forwards":0',script)
        self.assertIn('"product_promotion_allowed":False',script)

if __name__=="__main__":
    main()
