"""Source-only B9 B6/B8/B9 four frozen DEV528 192 real-forward gate tests."""
from __future__ import annotations
from collections import Counter
import importlib.util
import io
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b9_four_frozen_source_contract",
    HERE/"english_v6_b9_dev528_four_frozen_once.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CELLS=("mero-fast","mero-slow","ordinary-time",
       "hard-hey-mary","nearphone-meero-challenge")
POSITIVES={530:"mero-slow",532:"mero-fast",534:"mero-slow",536:"mero-fast"}
GROUPS={530:"piper-libritts-p1513",532:"piper-libritts-p5154",
        534:"piper-libritts-p6308",536:"piper-libritts-p664"}

def synthetic():
    originals=[];screen=[];num=0
    for speaker in range(528,540):
        for cell in CELLS:
            num+=1
            requested=None if cell==CELLS[4] else (0 if cell in CELLS[2:4] else 2)
            key=GROUPS.get(speaker,f"piper-new-{speaker}")
            row={"id":(m.PARAKEET_FAILED_ID if speaker==531 and cell=="mero-slow"
                       else f"dev528-{speaker}-{cell}"),
                 "source_group_id":key,"speaker_index":speaker,
                 "role":"development","cell_id":cell,
                 "requested_class":requested,"pcm_sha256":f"{num:064x}",
                 "wav_sha256":f"{num+100:064x}","samples":12000,
                 "human_gold":False,"training_allowed":False}
            originals.append(row)
            observed=(0 if cell=="ordinary-time"
                      else 2 if POSITIVES.get(speaker)==cell else None)
            tier=("weak" if observed==2 and speaker==530
                  else "silver" if observed is not None else "unknown")
            reason=("ONE_AUTHENTIC_PAR-AKEET_CAPTURE_FAILED"
                    if speaker==531 and cell=="mero-slow"
                    else "SYNTHETIC_TEST_REASON")
            screen.append({**row,"observed_class":observed,"tier":tier,
                           "reason":reason})
    doc={"schema":m.saved.SCHEMA,
         "status":"PASS_PROSPECTIVE_DEV_SYNTHETIC_MACHINE_QA_ONLY",
         "source_run":m.SOURCE["run"],"original_observer_run":44870958716,
         "source_group_counts":{"development":4},
         "tier_counts":{"silver":15,"weak":1,"unknown":44},
         "observed_label_counts":{"0":12,"2":4},
         "native_attempts":300,"native_successful_captures":299,
         "parakeet_failed_source_id":m.PARAKEET_FAILED_ID,
         "originals_with_five_complete_native":59,
         "hard_hey_mary_requested_is_not_gold":True,
         "full_6_2_2_2_gate_passed":False,
         "nearphone_as_ordinary_negative":False,
         "machine_only_no_human_gold":True,
         "new_audio_or_model_forwards":0,"new_optimizer_updates":0,
         "new_kws_model_forwards":0,"training_snapshot_allowed":False,
         "locked_kws_test_opened":False,"product_promotion":False,
         "source_group_identity_members":sorted(GROUPS.values()),
         "rows":screen}
    return originals,doc

class B9FourFrozenTests(TestCase):
    def test_source_provenance_and_no_PR_neural_score(self):
        p=m.source_check()
        self.assertEqual(p["status"],"SOURCE_ONLY_NO_NEURAL_FORWARD")
        self.assertEqual(p["keyword"],"Hey Mero")
        self.assertEqual(p["model_parameters"],43876)
        self.assertEqual((p["source_originals"],p["positive_originals"],
                          p["ordinary_originals"],p["unknown_excluded"]),
                         (60,4,12,44))
        self.assertEqual(p["qualified_groups"],4)
        self.assertEqual(p["fixed_gains"],[1.0,0.75,0.5])
        self.assertEqual(p["max_actual_frozen_forwards"],192)
        self.assertEqual(p["new_training_updates"],0)
        self.assertEqual(p["new_audio_asr_phone_calls"],0)
        self.assertEqual(p["new_model_forwards_in_PR"],0)
        self.assertFalse(p["cal_locked_opened"])
        self.assertFalse(p["model_selection_allowed"])
        self.assertFalse(p["product_promotion_allowed"])
        self.assertTrue(p["no_old_DEV520_reuse"])
        self.assertEqual(m.BEFORE,"1b47e4a5d0c30ae59bc48e754772852c90b28a55")
        self.assertEqual(m.RECOVERY_BEFORE,"820e3db36806991faf368b5197dfa6c5f97c4157")
        self.assertEqual(m.FAILED_FIRST_RUN,44872555787)
        self.assertEqual(p["original_failed_no_forward_run"],44872555787)
        self.assertEqual(m.SOURCE["id"],14693632800)
        self.assertEqual(m.SCREEN["id"],14694540474)
        self.assertEqual(m.SCREEN["sha"],
            "ae3062102541dbfc6f18a3d8db6a170fad529b0e97095b31bbc0ee6b8b6d026b")
        self.assertEqual(m.B9["id"],14689218171)
        self.assertEqual(m.B9["sha"],
            "1a8b270200e0c72ab1025f139d16abde43c209f142d72e71a7201b847e06a612")
        self.assertEqual(m.PINNED_B8_EVALUATOR_BLOB,
                         "d4c08b504945b1928b01e73ac9b11ef1dd13e29b")
        self.assertEqual(len(m.MODELS),4)
        self.assertEqual({z["pt"] for z in m.MODELS.values()},{
            "0461ce14ff5ed6490e88801b23c9036ca63b2c302aee7e8d96cea26c3eaa9f04",
            "cc16b81f84e4f21c5347ca153f405ac41ed116e3eedfaeed5677d1a4bc86e22f",
            "7dba86a3fcda6e5f05cb80b414dc372c70a5a03b3aee783f168441a4871e7655",
            "d672dcc87ca9e5d2c6dcf5952251a3c40fe8e7956d3b7c8753667c3093761cea"})

    def test_exact_prospective_new_screen_only_16_source_rows(self):
        source,doc=synthetic()
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"SCREEN.json").write_text(json.dumps(doc))
            with mock.patch.object(m.saved,"source_rows",return_value=source):
                rows,receipt=m.select(root,root)
        self.assertEqual(len(rows),16)
        self.assertEqual(Counter(r["mono_class"] for r in rows),{0:12,1:4})
        self.assertEqual(receipt["qualified_development_groups"],
                         sorted(m.QUALIFIED_FAMILY_IDS))
        self.assertEqual(receipt["unqualified_hardneg_count"],12)
        self.assertFalse(receipt["human_gold"])
        self.assertTrue(all(r["cell_id"] not in
            ("hard-hey-mary","nearphone-meero-challenge") for r in rows))
        self.assertTrue(all(r["role"]=="development" for r in rows))

    def test_fabricated_hard_or_failed_native_cannot_enter_scoring(self):
        source,doc=synthetic()
        bad=next(r for r in doc["rows"] if r["cell_id"]=="hard-hey-mary")
        bad["tier"]="silver";bad["observed_class"]=0
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"SCREEN.json").write_text(json.dumps(doc))
            with mock.patch.object(m.saved,"source_rows",return_value=source):
                with self.assertRaisesRegex(ValueError,"NO_UNQUALIFIED"):
                    m.select(root,root)
        source,doc=synthetic()
        bad=next(r for r in doc["rows"] if
                 r["speaker_index"]==531 and r["cell_id"]=="mero-slow")
        bad["tier"]="silver";bad["observed_class"]=2
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"SCREEN.json").write_text(json.dumps(doc))
            with mock.patch.object(m.saved,"source_rows",return_value=source):
                with self.assertRaisesRegex(ValueError,"NO_FAILED_PAR_AKEET"):
                    m.select(root,root)

    def test_reject_old_data_or_missing_model_archive(self):
        source,doc=synthetic()
        doc["native_successful_captures"]=300
        with TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"SCREEN.json").write_text(json.dumps(doc))
            with mock.patch.object(m.saved,"source_rows",return_value=source):
                with self.assertRaisesRegex(ValueError,
                                            "FROZEN_INDEPENDENT_B9_NEW_SOURCE_QA"):
                    m.select(root,root)
        with self.assertRaisesRegex(ValueError,"EXACT_ORIGINAL_B9_TRAINING_ZIP_HASH"):
            m.verify_b9_archive("/missing/b9-real-weight.zip",None,None)

    def test_actual_frozen_CTC_research_not_release(self):
        script=(HERE/"english_v6_b9_dev528_four_frozen_once.py").read_text()
        self.assertIn("ref.verify_model_archive(b6_zip,\"B6\"",script)
        self.assertIn("ref.verify_model_archive(b8_zip,\"B8\"",script)
        self.assertIn("verify_b9_archive(b9_zip,torch,np)",script)
        self.assertIn("torch.no_grad()",script)
        self.assertIn('"natural_far_per_hour":"NOT_MEASURED"',script)
        self.assertIn('"product_promotion_allowed":False',script)
        self.assertIn('"real_frozen_model_forward_conditions"]==192',script)
        self.assertNotIn("optimizer.step()",script)
        self.assertNotIn("retrain_on_dev",script)

    def test_original_zero_forward_failed_run_required_for_recovery(self):
        run={"id":m.FAILED_FIRST_RUN,"head_sha":m.RECOVERY_BEFORE,
             "head_branch":"main","event":"push","run_attempt":1,
             "status":"completed","conclusion":"failure"}
        jobs={"total_count":3,"jobs":[
            {"name":"source-contract","conclusion":"failure"},
            {"name":"original-source-screen-audit","conclusion":"skipped"},
            {"name":"real-four-frozen-models","conclusion":"skipped"}]}
        artifacts={"total_count":0,"artifacts":[]}
        def get(req,timeout):
            uri=req.full_url
            val=(jobs if uri.endswith("/jobs?per_page=100")
                 else artifacts if uri.endswith("/artifacts?per_page=100")
                 else run)
            return io.BytesIO(json.dumps(val).encode())
        with mock.patch.object(m.urllib.request,"urlopen",side_effect=get):
            good=m.failed_first_run_proof("synthetic-test-token-long-enough")
            self.assertEqual(good["artifact_count"],0)
            self.assertEqual(good["frozen_model_job"],"skipped")
            jobs["jobs"][2]["conclusion"]="success"
            with self.assertRaisesRegex(ValueError,"FAILED_BEFORE_SOURCE_AUDIT"):
                m.failed_first_run_proof("synthetic-test-token-long-enough")
            jobs["jobs"][2]["conclusion"]="skipped"
            artifacts["total_count"]=1
            with self.assertRaisesRegex(ValueError,"NO_MODEL_OUTPUT_ARCHIVE"):
                m.failed_first_run_proof("synthetic-test-token-long-enough")

    def test_only_first_exact_main_successor_no_rerun(self):
        with mock.patch.dict(os.environ,{"GITHUB_ACTIONS":"false"}):
            with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_PROTECTED_MAIN_ONLY"):
                m.guard()

if __name__=="__main__":
    main()
