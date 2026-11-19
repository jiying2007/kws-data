"""No-model contract tests for V5 frozen phone pretrained NEW-DEV probe."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "v5_newdev_phone_test",HERE/"english_v5_mero_newdev_phone_reference_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

class V5NewDevPretrainedRefTests(TestCase):
    def test_only_one_keyword_and_original_model_pins(self):
        a=v.source_check()
        self.assertEqual(a["status"],"SOURCE_ONLY_NO_KWS_INFERENCE")
        self.assertEqual(a["single_keyword"],"Hey Mero")
        self.assertEqual(a["model"],"official-zh-en-phone-3m")
        self.assertEqual(a["target_counts"],{"positive":2,"ordinary":2})
        self.assertEqual(a["max_new_kws_forwards"],4)
        self.assertEqual(a["new_optimizer_updates"],0)
        self.assertEqual(a["new_tts_calls"],0)
        self.assertEqual(a["new_asr_phone_model_calls"],0)
        self.assertEqual(a["threshold_frozen"],0.25)
        self.assertFalse(a["model_promotion_allowed"])
        self.assertFalse(a["calibration_or_locked_audio_opened"])

    def test_successful_first_main_and_reject_replay_or_pr(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
               "repository":{"full_name":v.REPO,"private":False},
               "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,
             "GITHUB_RUN_ID":"777"}
        result=v.guard(env=env,event=event,parent=v.BEFORE)
        self.assertEqual(result["new_kws_calls_allowed"],4)
        self.assertFalse(result["model_promotion_allowed"])
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):v.guard(env=env,event=event,parent=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):v.guard(env=env,event=event,parent=v.BEFORE)

    def candidate_rows(self):
        pcm=b"\x00\x00"*8000
        digest=hashlib.sha256(pcm).hexdigest()
        return ([{"id":f"new-dev-{speaker}-{label}",
                  "source_group_id":f"piper-libritts-p{speaker}",
                  "speaker_index":speaker,"role":"development",
                  "mono_class":label,"tier":"silver",
                  "source_pcm_sha256":digest,
                  "research_dev_only":True,
                  "research_training_bind":False,
                  "origin":"new_frozen_piper_source96",
                  "samples":8000}
                 for speaker in (80001,80002) for label in (0,1)],pcm)

    def test_audit_reuses_only_four_source_disjoint_original_pcm(self):
        rows,pcm=self.candidate_rows()
        with mock.patch.object(v.f,"frozen_gate",return_value={
                "status":"PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"}),\
             mock.patch.object(v.f,"screen_new",return_value=(rows,{})),\
             mock.patch.object(v.f,"readpcm",return_value=pcm) as getter:
            result=v.audit(Path("/tmp/only-source"),Path("/tmp/only-screen"),
                           Path("/tmp/only-gate"))
        self.assertEqual(result["status"],
                         "SAVED_FOUR_EXPOSED_NEWDEV_AUDIO_AUTHENTICATED")
        self.assertEqual(len(result["originals"]["selected"]),4)
        self.assertEqual({x["mono_class"] for x in result["originals"]["selected"]},
                         {0,1})
        self.assertEqual(result["numerical_kws_forwards"],0)
        self.assertTrue(result["originals"]["no_train_calibration_locked_pcm"])
        self.assertEqual(getter.call_count,4)

    def test_train_role_or_extra_dev_source_is_not_accepted(self):
        rows,pcm=self.candidate_rows()
        rows[0]["research_training_bind"]=True
        with mock.patch.object(v.f,"frozen_gate",return_value={
               "status":"PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"}),\
             mock.patch.object(v.f,"screen_new",return_value=(rows,{})),\
             mock.patch.object(v.f,"readpcm",return_value=pcm):
            with self.assertRaisesRegex(ValueError,"EXACT_MACHINE_SCREENED"):
                v.audited_new_dev(Path("/tmp/a"),Path("/tmp/b"),Path("/tmp/c"))

    def test_old_locked_and_unknown_requests_never_become_labels(self):
        rows,pcm=self.candidate_rows()
        rows[0]["role"]="locked-test"
        with mock.patch.object(v.f,"frozen_gate",return_value={
               "status":"PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"}),\
             mock.patch.object(v.f,"screen_new",return_value=(rows,{})),\
             mock.patch.object(v.f,"readpcm",return_value=pcm):
            with self.assertRaises(ValueError):
                v.audited_new_dev(Path("/tmp/a"),Path("/tmp/b"),Path("/tmp/c"))

if __name__=="__main__":
    main()
