"""B3 vs B4 exact post-training DEV496 three-model source-only review guards."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b4_paired_frozen_source",HERE/"english_v6_b4_dev496_three_frozen_paired_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def dataset():
    originals=[];decisions=[]
    cells=("mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge")
    for i in range(8):
        for cell in cells:
            cls=0 if cell=="ordinary-time" else (None if cell=="nearphone-meero-challenge" else 2)
            label=(2 if cell=="mero-slow" and i<5 else
                   0 if cell=="ordinary-time" else None)
            tier=("silver" if label==0 or (label==2 and i<3) else
                  "weak" if label==2 else "unknown")
            row={"id":f"source{i}-{cell}","source_group_id":f"g{i}",
                 "speaker_index":496+i,"role":"development","cell_id":cell,
                 "requested_class":cls,"pcm_sha256":f"{i*4+cells.index(cell):064x}",
                 "wav_sha256":f"{i*4+cells.index(cell)+50:064x}",
                 "samples":11200,"human_gold":False,"training_allowed":False}
            originals.append(row)
            decisions.append({**row,"tier":tier,"observed_class":label})
    screen={"schema":m.saved.SCHEMA,
       "status":"PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY",
       "source_run":m.SOURCE["run"],"original_observer_run":44672571583,
       "source_group_counts":{"development":5},
       "tier_counts":{"silver":11,"weak":2,"unknown":19},
       "observed_label_counts":{"0":8,"2":5},
       "source_group_identity_members":[f"g{i}" for i in range(5)],
       "full_6_2_2_2_gate_passed":False,
       "nearphone_as_ordinary_negative":False,
       "machine_only_no_human_gold":True,
       "new_audio_or_model_forwards":0,"new_optimizer_updates":0,
       "new_kws_model_forwards":0,"training_snapshot_allowed":False,
       "locked_kws_test_opened":False,"product_promotion":False,
       "rows":decisions}
    return originals,screen

class SourceOnly(TestCase):
    def test_exact_new_qa_and_model_hashes(self):
        q=m.source_check()
        self.assertEqual(q["status"],"SOURCE_ONLY_NO_NEURAL_FORWARD")
        self.assertEqual(q["keyword"],"Hey Mero")
        self.assertEqual(q["new_source_run"],44667050336)
        self.assertEqual(q["saved_machine_screen_run"],44673975940)
        self.assertEqual(q["machine_qualified_positive"],5)
        self.assertEqual(q["machine_qualified_ordinary"],8)
        self.assertEqual(q["excluded_unknown"],19)
        self.assertEqual(q["qualified_development_groups"],5)
        self.assertEqual(q["frozen_gains"],[1.0,0.75,0.5])
        self.assertEqual(set(q["models"]),{"B3","B4_control","B4_half"})
        self.assertEqual(q["models"]["B3"]["artifact"],14590093798)
        self.assertEqual(q["models"]["B4_control"]["artifact"],14597862669)
        self.assertEqual(q["models"]["B4_half"]["artifact"],14597862669)
        self.assertEqual(m.ARCHIVES["B4"]["sha"],
                         "6eba8013da5ef06aa5046eab3f8d48b912af5f80941d98f94d0affae23764cb2")
        self.assertEqual(m.SCREEN["sha"],
                         "99f85da0562ee380fd5ae42c252ecb4f9e1b12d51c896b5169f59c772158a050")
        self.assertEqual(m.SCREEN["bytes"],5595)
        self.assertEqual(q["new_model_forwards_in_PR"],0)
        self.assertFalse(q["cal_locked_audio_access"])
        self.assertTrue(q["no_old_exposed_dev_reuse"])
        self.assertFalse(q["model_selection_allowed"])
        self.assertFalse(q["product_promotion_allowed"])
        self.assertEqual(m.BEFORE,"b6524b0cb5dcb384236cadfcd2d3c9ac164b59a5")

    def test_only_qualified_originals_and_nearphone_unknown(self):
        original,screen=dataset()
        with TemporaryDirectory() as root:
            p=Path(root)/"SCREEN.json"
            p.write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=original):
                selected,receipt=m.select(root,root)
            self.assertEqual(len(selected),13)
            self.assertEqual(sum(r["mono_class"]==1 for r in selected),5)
            self.assertEqual(sum(r["mono_class"]==0 for r in selected),8)
            self.assertEqual(receipt["source_groups"],8)
            self.assertEqual(len(receipt["qualified_positive_source_groups"]),5)
            self.assertTrue(all(r["role"]=="development" for r in selected))
            self.assertFalse(any(r["cell_id"]=="nearphone-meero-challenge"
                                 for r in selected))
            screen["rows"][-1]["tier"]="silver"
            screen["rows"][-1]["observed_class"]=0
            p.write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=original):
                with self.assertRaisesRegex(ValueError,"NEARPHONE_MUST_REMAIN_UNKNOWN"):
                    m.select(root,root)

    def test_fresh_source_only_not_training_or_locked(self):
        original,screen=dataset()
        original[0]["role"]="locked-test"
        with TemporaryDirectory() as root:
            (Path(root)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=original):
                with self.assertRaisesRegex(ValueError,"BYTE_IDENTITY_ASR_SOURCE_BINDING"):
                    m.select(root,root)
        original,screen=dataset()
        screen["tier_counts"]["unknown"]=18
        with TemporaryDirectory() as root:
            (Path(root)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=original):
                with self.assertRaisesRegex(ValueError,"FROZEN_INDEPENDENT_MACHINE_QA_ONLY"):
                    m.select(root,root)

    def test_no_pr_replay(self):
        with mock.patch.dict("os.environ",{
            "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
            "GITHUB_EVENT_NAME":"pull_request",
            "GITHUB_REF":"refs/pull/285/merge",
            "GITHUB_RUN_ATTEMPT":"1"},clear=True):
            with self.assertRaisesRegex(ValueError,"FIRST_PROTECTED_PUBLIC_MAIN_ONLY"):
                m.guard()
        with self.assertRaisesRegex(ValueError,"GITHUB_READ_ONLY_TOKEN_REQUIRED"):
            m.provenance("")

if __name__=="__main__":main()
