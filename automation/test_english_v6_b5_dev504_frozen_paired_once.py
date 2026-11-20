"""Read-only source contracts for frozen B4/B5 on new DEV504 originals."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v6_b5_frozen_compare_test",HERE/"english_v6_b5_dev504_frozen_paired_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def fixture():
    cells=("mero-fast","mero-slow","ordinary-time","nearphone-meero-challenge")
    originals=[];decisions=[]
    for i in range(8):
        for n,cell in enumerate(cells):
            requested=0 if n==2 else (None if n==3 else 2)
            actual=(2 if (i in (0,1) and n in (0,1)) or (i==2 and n==0)
                    else (0 if n==2 else None))
            tier=("silver" if actual==0 or (actual==2 and ((i in (0,1) and n==0) or i==2))
                  else ("weak" if actual==2 else "unknown"))
            row={"id":f"n{i}-{cell}","source_group_id":f"g{i}",
                 "speaker_index":504+i,"role":"development","cell_id":cell,
                 "requested_class":requested,"pcm_sha256":f"{i*4+n:064x}",
                 "wav_sha256":f"{i*4+n+48:064x}","samples":10000,
                 "human_gold":False,"training_allowed":False}
            originals.append(row)
            decisions.append({**row,"tier":tier,"observed_class":actual})
    screen={"schema":m.saved.SCHEMA,
      "status":"PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY",
      "source_run":m.SOURCE["run"],"original_observer_run":44694776784,
      "source_group_counts":{"development":3},
      "tier_counts":{"silver":11,"weak":2,"unknown":19},
      "observed_label_counts":{"0":8,"2":5},
      "full_6_2_2_2_gate_passed":False,
      "nearphone_as_ordinary_negative":False,"machine_only_no_human_gold":True,
      "new_audio_or_model_forwards":0,"new_optimizer_updates":0,
      "new_kws_model_forwards":0,"training_snapshot_allowed":False,
      "locked_kws_test_opened":False,"product_promotion":False,
      "source_group_identity_members":["g0","g1","g2"],"rows":decisions}
    return originals,screen

class B5SourceOnly(TestCase):
    def test_frozen_training_source_and_result(self):
        p=m.source_check()
        self.assertEqual(p["status"],"SOURCE_ONLY_NO_NEURAL_FORWARD")
        self.assertEqual(p["machine_qualified_positive"],5)
        self.assertEqual(p["machine_qualified_ordinary"],8)
        self.assertEqual(p["excluded_unknown"],19)
        self.assertEqual(p["qualified_development_groups"],3)
        self.assertEqual(p["frozen_gains"],[1.0,0.75,0.5])
        self.assertEqual(p["new_model_forwards_in_PR"],0)
        self.assertFalse(p["product_promotion_allowed"])
        self.assertEqual(set(p["models"]),{"B4_half","B5_control","B5_early_hey"})
        self.assertEqual(m.ARCHIVES["B5"]["id"],14608546071)
        self.assertEqual(m.ARCHIVES["B5"]["sha"],
             "e945b156d52199569aa36963b786424df7940f0b9285a26af9509284024e081a")
        self.assertEqual(m.MODELS["B5_early_hey"]["pt"],
             "784f82b552c411f04631d5adc0664b4f07f305ca45d551a1de8352e844ee440d")
        self.assertEqual(m.BEFORE,"72a8309188ab95ac3c35e32ea347d0240bcd2881")

    def test_only_new_13_machine_qualified_originals(self):
        originals,screen=fixture()
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                x,receipt=m.select(tmp,tmp)
        self.assertEqual(len(x),13)
        self.assertEqual(sum(a["mono_class"]==1 for a in x),5)
        self.assertEqual(sum(a["mono_class"]==0 for a in x),8)
        self.assertEqual(len(receipt["qualified_positive_source_groups"]),3)
        self.assertFalse(any(a["cell_id"]=="nearphone-meero-challenge" for a in x))

    def test_nearphone_never_ordinary(self):
        originals,screen=fixture()
        screen["rows"][-1]["tier"]="silver"
        screen["rows"][-1]["observed_class"]=0
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"NEARPHONE"):
                    m.select(tmp,tmp)

    def test_no_nondev_source_and_no_machine_gate_drift(self):
        originals,screen=fixture()
        originals[0]["role"]="locked-test"
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"BYTE_IDENTITY_ASR_SOURCE_BINDING"):
                    m.select(tmp,tmp)
        originals,screen=fixture()
        screen["source_group_counts"]["development"]=2
        with TemporaryDirectory() as tmp:
            (Path(tmp)/"SCREEN.json").write_text(json.dumps(screen))
            with mock.patch.object(m.saved,"source_rows",return_value=originals):
                with self.assertRaisesRegex(ValueError,"FROZEN_INDEPENDENT_MACHINE_QA_ONLY"):
                    m.select(tmp,tmp)

    def test_recovery_is_only_prior_preinference_failure(self):
        self.assertEqual(m.BEFORE,"72a8309188ab95ac3c35e32ea347d0240bcd2881")
        self.assertEqual(m.SCREEN["head"],"027621f5f908961b3133ae15d26485821cb07b43")
        self.assertEqual(m.RECOVERY["failed_run"],44698504736)
        self.assertEqual(m.RECOVERY["failed_job"],136161797047)
        self.assertEqual(m.RECOVERY["head"],m.BEFORE)
        workflow=(m.ROOT/".github/workflows/english-v6-b5-dev504-frozen-paired-once.yml").read_text()
        self.assertEqual(len(m.ARCHIVES["B5"]["sha"]),64)
        self.assertEqual(workflow.count("'"+m.ARCHIVES["B5"]["sha"]+"'"),1)
        self.assertEqual(workflow.count("'"+m.ARCHIVES["B4"]["sha"]+"'"),1)
        self.assertIn("github.event.before == '"+m.BEFORE+"'",workflow)

    def test_exact_first_main_successor_still_allowed(self):
        with TemporaryDirectory() as tmp:
            event={"before":m.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
                   "repository":{"full_name":m.REPO,"private":False},
                   "forced":False,"deleted":False}
            event_path=Path(tmp)/"event.json"
            event_path.write_text(json.dumps(event))
            env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
                 "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
                 "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
                 "GITHUB_RUN_ID":"999","GITHUB_EVENT_PATH":str(event_path)}
            with mock.patch.dict("os.environ",env,clear=True),mock.patch.object(
                    m.subprocess,"check_output",return_value=m.BEFORE):
                self.assertEqual(m.guard()["parent"],m.BEFORE)

    def test_first_main_only_no_pr_or_rerun(self):
        with mock.patch.dict("os.environ",{
            "GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":m.REPO,
            "GITHUB_EVENT_NAME":"pull_request",
            "GITHUB_REF":"refs/pull/290/merge","GITHUB_RUN_ATTEMPT":"1"},clear=True):
            with self.assertRaisesRegex(ValueError,"FIRST_PROTECTED_PUBLIC_MAIN_ONLY"):
                m.guard()
        with self.assertRaisesRegex(ValueError,"GITHUB_READ_ONLY_TOKEN_REQUIRED"):
            m.provenance("")
if __name__=="__main__":main()
