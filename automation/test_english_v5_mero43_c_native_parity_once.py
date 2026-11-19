"""Source-only regression contracts for frozen real 43K C-vs-PyTorch test."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "nero43_real_c_test",HERE/"english_v5_mero43_c_native_parity_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

class NativeParitySourceTests(TestCase):
    def test_exact_original_trained_weights_and_new_machine_sources(self):
        state=v.source_check()
        self.assertEqual(state["status"],"SOURCE_ONLY_NATIVE_PARITY_NOT_RUN")
        self.assertEqual(state["model_artifact"],14514228406)
        self.assertEqual(state["source_artifact"],14524071469)
        self.assertEqual(state["machine_screen_artifact"],14525005967)
        self.assertEqual(state["original_audited_positive"],2)
        self.assertEqual(state["original_audited_ordinary"],8)
        self.assertEqual(state["maximum_C_forwards"],10)
        self.assertEqual(state["absolute_logit_tolerance"],0.005)
        self.assertEqual(state["mean_logit_tolerance"],0.0005)
        self.assertEqual(state["new_train_or_ASR_TTS_forwards"],0)
        self.assertFalse(state["product_promotion_allowed"])
        self.assertFalse(state["calibration_or_locked_wavs_opened"])
        self.assertEqual(state["source_repo_sha"],v.PIPELINE_COMMIT)

    def test_frame_clocks_and_decoder_event_receipts(self):
        with TemporaryDirectory() as td:
            file=Path(td)/"clocks.csv"
            file.write_text("0,512,0\n1,672,0\n2,832,1\n")
            self.assertEqual(v.load_events(file,3),[0,0,1])
            file.write_text("0,512,0\n1,671,0\n2,832,1\n")
            with self.assertRaisesRegex(ValueError,"C_STREAM_FRAME_CLOCK"):
                v.load_events(file,3)
            file.write_text("0,512,0\n1,672,0\n")
            with self.assertRaisesRegex(ValueError,"STREAM_EMITS"):
                v.load_events(file,3)
            file.write_text("0,512,2\n1,672,0\n2,832,1\n")
            with self.assertRaises(ValueError):
                v.load_events(file,3)

    def test_complete_exact_pipeline_C_source_required(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            target=root/"research"/"english_v5_mero"
            target.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,"ALL_FROZEN_NATIVE_C_SOURCE"):
                v.pipeline_files(root)
            for name in ("mero_ctc.h","mero_ctc.c","mero43_stream.h",
                         "mero43_stream.c","mero43_host_cli.c",
                         "pack_mero43.py","test_mero43_stream.c"):
                (target/name).write_text("source")
            self.assertEqual(v.pipeline_files(root),target)

    def test_one_first_public_main_only_without_replay(self):
        event={"before":v.BEFORE,"after":"a"*40,
               "ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"9999"}
        self.assertEqual(v.guard(env=env,event=event,parent=v.BEFORE)
                         ["pipeline_source_commit"],v.PIPELINE_COMMIT)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):
            v.guard(env=env,event=event,parent=v.BEFORE)

    def test_bad_source_count_fails_without_model_audio(self):
        original=[{"mono_class":1,"source_group_id":"a"}]
        with mock.patch.object(v.proof,"select",return_value=(original,{
                "no_locked_or_calibration":True})):
            with self.assertRaisesRegex(ValueError,"ORIGINAL_TEN_DEV"):
                v.audit(Path("/tmp/old"),Path("/tmp/screen"),Path("/tmp/model"))


    def test_arm32_succession_no_replay_and_not_product_promotion(self):
        self.assertEqual(v.HOST_PROOF["run"],44541193521)
        self.assertEqual(v.HOST_PROOF["head"],v.ARM_BEFORE)
        self.assertEqual(v.HOST_PROOF["artifact"],14538062948)
        self.assertTrue(v.HOST_PROOF["digest"].startswith("sha256:"))
        self.assertFalse(v.source_check()["product_promotion_allowed"])
        event={"before":v.ARM_BEFORE,"after":"b"*40,
               "ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"b"*40,
             "GITHUB_RUN_ID":"44599999999"}
        receipt=v.arm_guard(env=env,event=event,parent=v.ARM_BEFORE)
        self.assertEqual(receipt["host_run"],44541193521)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaisesRegex(ValueError,"ARM32_ONLY_FIRST_PUBLIC_MAIN"):
            v.arm_guard(env=env,event=event,parent=v.ARM_BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        event["before"]="0"*40
        with self.assertRaisesRegex(ValueError,"ARM32_EXACT_UNREPLAYED_SUCCESSOR"):
            v.arm_guard(env=env,event=event,parent=v.ARM_BEFORE)
        event["before"]=v.ARM_BEFORE
        with self.assertRaisesRegex(ValueError,"ARM32_EXACT_UNREPLAYED_PARENT"):
            v.arm_guard(env=env,event=event,parent="a"*40)


if __name__=="__main__": main()
