"""Strict source-only B10 72-forward post-train frozen DEV comparison contracts."""
from __future__ import annotations
from collections import Counter
import importlib.util
from pathlib import Path
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "frozen_b10_dev540_source_check",HERE/"english_v6_b10_frozen_dev540_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def fixtures():
    originals=[];decisions=[]
    for index in range(448,454):
        row={"id":f"b10-train-{index}-ordinary-time-hard",
             "role":"train","cell_id":"ordinary-time-hard",
             "source_group_id":f"piper-libritts-training-{index}",
             "speaker_index":index,"pcm_sha256":f"{index:064x}",
             "wav_sha256":f"{index+2000:064x}",
             "samples":12000,"human_gold":False,
             "training_allowed":False,"observed_class":None,
             "requested_class":0}
        originals.append(row)
        decisions.append({**row,"tier":"silver","observed_class":0})
    groups={540:"piper-libritts-p3221",541:"piper-libritts-p8687",
            542:"piper-libritts-p7030",543:"piper-libritts-p476",
            544:"piper-libritts-p4257",545:"piper-libritts-p5918"}
    for index,g in groups.items():
        for cell in ("mero-fast","ordinary-time"):
            row={"id":f"b10-dev-{index}-{cell}","role":"development",
                 "cell_id":cell,"source_group_id":g,"speaker_index":index,
                 "pcm_sha256":f"{index+(1000 if cell=='mero-fast' else 2000):064x}",
                 "wav_sha256":f"{index+(3000 if cell=='mero-fast' else 4000):064x}",
                 "samples":9500,"human_gold":False,
                 "training_allowed":False,"observed_class":None,
                 "requested_class":2 if cell=="mero-fast" else 0}
            observed=(2 if cell=="mero-fast" and index in (542,544)
                      else 0 if cell=="ordinary-time" else None)
            originals.append(row)
            decisions.append({**row,"observed_class":observed,
                              "tier":"weak" if observed==2 else
                                     "silver" if observed==0 else "unknown"})
    status={"status":"PASS_B10_TRAIN_DEV_MACHINE_RESEARCH_GATE_ONLY",
            "schema":m.screen.SCHEMA,"source_run":44906200481,
            "observer_run":44908125673,"complete_native_attempts":90,
            "admission":{
                 "train_hard_negative_qualified":6,
                 "development_qualified_groups":2,
                 "development_qualified_source_groups":
                    sorted(m.QUALIFIED_DEV_GROUPS),
                 "B10_research_optimizer_allowed":True},
            "no_old_DEV520_528_training":True,
            "cal_locked_opened":False,"new_kws_model_forwards":0,
            "product_promotion_allowed":False,"rows":decisions}
    return originals,status

class B10FrozenDEVTests(TestCase):
    def test_exact_model_weights_and_72_forwards_are_research_only(self):
        r=m.source_check()
        self.assertEqual(r["status"],"SOURCE_ONLY_ZERO_FROZEN_MODEL_FORWARDS")
        self.assertEqual(r["run"],44987190871)
        self.assertEqual(r["model_archive_id"],14740303244)
        self.assertEqual(r["planned_actual_model_forwards"],72)
        self.assertEqual(r["machine_positive"],2)
        self.assertEqual(r["machine_ordinary"],6)
        self.assertEqual(r["unknown_excluded"],4)
        self.assertEqual(r["fixed_gains"],[1.0,0.75,0.5])
        self.assertFalse(r["threshold_sweep"])
        self.assertFalse(r["model_selection_allowed"])
        self.assertFalse(r["cal_locked_opened"])
        self.assertEqual(set(r["b10_model_pt_sha256"]),
                         {"B10_control","B10_hard_negative"})

    def test_source_roles_and_unknown_remain_separate(self):
        originals,doc=fixtures()
        with mock.patch.object(m.observer,"source_rows",return_value=originals), \
             mock.patch.object(m,"read",return_value=doc), \
             mock.patch.object(m.screen,"admission",
                return_value=doc["admission"]):
            scored,proof=m.select("/fake-src","/fake-train","/fake-screen",
                                  "/fake-result")
        self.assertEqual(len(scored),8)
        self.assertEqual(Counter(x["positive"] for x in scored),
                         {True:2,False:6})
        self.assertEqual(len(proof["train_ids_excluded"]),6)
        self.assertEqual(len(proof["dev_unknown_excluded"]),4)
        self.assertEqual({x["speaker_index"] for x in scored if x["positive"]},
                         {542,544})
        self.assertTrue(all(x["role"]=="development" for x in scored))

    def test_reject_fabricated_dev_positives(self):
        originals,doc=fixtures()
        target=next(r for r in doc["rows"] if r["id"]=="b10-dev-540-mero-fast")
        target["tier"]="weak";target["observed_class"]=2
        with mock.patch.object(m.observer,"source_rows",return_value=originals), \
             mock.patch.object(m,"read",return_value=doc), \
             mock.patch.object(m.screen,"admission",
                return_value=doc["admission"]):
            with self.assertRaisesRegex(ValueError,"FRESH_EIGHT_APPROVED"):
                m.select("/fake-src","/fake-train","/fake-screen","/fake-result")

    def test_reject_dev_role_drift(self):
        originals,doc=fixtures()
        doc["rows"][-1]["role"]="train"
        with mock.patch.object(m.observer,"source_rows",return_value=originals), \
             mock.patch.object(m,"read",return_value=doc), \
             mock.patch.object(m.screen,"admission",
                return_value=doc["admission"]):
            with self.assertRaisesRegex(ValueError,"NO_ORIGINAL_ROLE_LABEL"):
                m.select("/fake-src","/fake-train","/fake-screen","/fake-result")

    def test_reject_non_sha_model_archive(self):
        with self.assertRaisesRegex(ValueError,"ACTUAL_B10_TRAIN_ZIP"):
            m.verify_models("/missing/B10-model.zip",None,None,None,None,None)

    def test_original_b10_run_and_model_hashes_pinned(self):
        self.assertEqual(m.TRAIN["bytes"],1000738)
        self.assertEqual(m.TRAIN["sha"],
            "3c7247e301aa57ef442b116ad28dc3356e80b88942cea687fcebb8af8270b33d")
        self.assertEqual(m.B10_PTS["B10_control"],
            "43e8dc8e1e4150421e55d44b4ad6c5813201947f1560408c0a47d83439608352")
        self.assertEqual(m.B10_PTS["B10_hard_negative"],
            "9dc4d8696c16adde295f32b287653165e17fd3e0b650865679474d67616c4ae5")
        script=(HERE/"english_v6_b10_frozen_dev540_once.py").read_text()
        self.assertIn("with torch.no_grad()",script)
        self.assertNotIn("optimizer.step()",script)
        self.assertIn('"product_promotion_allowed":False',script)

if __name__=="__main__":
    main()
