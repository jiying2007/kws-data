"""No-model regression contracts for strong V5 Hey Mero causal FSMN research."""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
 "v5_strong_128_tests",HERE/"english_v5_mero_strong_fsmn128_once.py")
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def row(role,g,klass,num,corpus="old"):
    name=f"source-{corpus}-{role}-{g}-{klass}-{num}"
    return {"id":name,"role":role,"source_group_id":f"group-{corpus}-{role}-{g}",
            "source_pcm_sha256":hashlib.sha256(name.encode()).hexdigest(),
            "mono_class":klass,"tier":"silver"}

def samples():
    old=[row("train",i,klass,0) for i in range(24) for klass in (0,1)]
    old += [row("train",i,1,1) for i in range(6)]
    fresh=[row("train",i,klass,0,"fresh") for i in range(11) for klass in (0,1)]
    dev=[row("development",i,klass,0,"fresh") for i in range(2) for klass in (0,1)]
    return old,fresh+dev

class V5StrongFsmn128SourceContracts(TestCase):
    def test_architecture_resource_contract_pure_python(self):
        a=v.source_check()
        self.assertEqual(a["keyword"],"Hey Mero")
        self.assertEqual(a["parameters"],93444)
        self.assertEqual(a["hidden"],128)
        self.assertEqual(a["memory_frames_left"],6)
        self.assertEqual(a["max_optimizer_updates"],448)
        self.assertEqual(a["old_train_original_pcm"],54)
        self.assertEqual(a["new_machine_qualified_train_groups"],11)
        self.assertEqual(a["new_exposed_dev_groups"],2)
        self.assertTrue(a["trained_from_scratch"])
        self.assertFalse(a["prior_failed_256_model_warmstart"])
        self.assertFalse(a["model_promotion_allowed"])
        # Exact parameter count for input, 5 linear+causal-depthwise blocks,
        # and a single blank/HEY/MEH/ROH output layer.
        params=(40*128+128)+5*((128*128+128)+(128*(6+1)+128))+(128*4+4)
        self.assertEqual(params,v.PARAMS)

    def test_valid_main_guard_without_work_or_replay(self):
        event={"before":v.BEFORE,"after":"a"*40,"ref":"refs/heads/main",
            "repository":{"full_name":v.REPO,"private":False},
            "forced":False,"deleted":False}
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_SHA":"a"*40,"GITHUB_RUN_ID":"449"}
        good=v.guard(env=env,event=event,parentsha=v.BEFORE)
        self.assertEqual(good["optimizer_budget"],448)
        env["GITHUB_RUN_ATTEMPT"]="2"
        with self.assertRaises(ValueError):v.guard(env=env,event=event,parentsha=v.BEFORE)
        env["GITHUB_RUN_ATTEMPT"]="1"
        env["GITHUB_EVENT_NAME"]="pull_request"
        with self.assertRaises(ValueError):v.guard(env=env,event=event,parentsha=v.BEFORE)

    def test_original_35_group_train_and_two_dev_audit_without_models(self):
        old,fresh=samples()
        with mock.patch.object(v.proof,"frozen_gate",return_value={
               "status":"PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"}),\
             mock.patch.object(v.parent,"authorized_source",return_value=(old,{})),\
             mock.patch.object(v.proof,"screen_new",return_value=(fresh,{})):
            a,b,dev,summary=v.audit(Path("/tmp/old"),Path("/tmp/audit"),
                 Path("/tmp/fresh"),Path("/tmp/screen"),Path("/tmp/gate"))
        self.assertEqual(len(a),54)
        self.assertEqual(len(b),22)
        self.assertEqual(len(dev),4)
        self.assertEqual(summary["combined_train_groups"],35)
        self.assertEqual(summary["new_dev_groups"],2)
        self.assertTrue(summary["train_dev_original_pcm_sha_disjoint"])
        self.assertFalse(summary["original_locked_audio_opened"])
        self.assertEqual(summary["model_kws_calls"],0)

    def test_same_source_in_train_and_dev_fails_closed(self):
        old,fresh=samples()
        devrow=next(x for x in fresh if x["role"]=="development")
        devrow["source_group_id"]=next(x for x in fresh if x["role"]=="train")["source_group_id"]
        with mock.patch.object(v.proof,"frozen_gate",return_value={
               "status":"PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"}),\
             mock.patch.object(v.parent,"authorized_source",return_value=(old,{})),\
             mock.patch.object(v.proof,"screen_new",return_value=(fresh,{})):
            with self.assertRaises(ValueError):
                v.audit(Path("/tmp/old"),Path("/tmp/audit"),
                        Path("/tmp/fresh"),Path("/tmp/screen"),Path("/tmp/gate"))

    def test_locked_and_nearphone_can_never_enter_train(self):
        old,fresh=samples()
        fresh[0]["role"]="locked-test"
        with mock.patch.object(v.proof,"frozen_gate",return_value={
               "status":"PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"}),\
             mock.patch.object(v.parent,"authorized_source",return_value=(old,{})),\
             mock.patch.object(v.proof,"screen_new",return_value=(fresh,{})):
            with self.assertRaises(ValueError):
                v.audit(Path("/tmp/old"),Path("/tmp/audit"),
                        Path("/tmp/fresh"),Path("/tmp/screen"),Path("/tmp/gate"))
        old,fresh=samples()
        fresh[0]["tier"]="unknown"
        with mock.patch.object(v.proof,"frozen_gate",return_value={
               "status":"PASS_SYNTHETIC_MACHINE_6_2_2_2_SOURCE_GATE"}),\
             mock.patch.object(v.parent,"authorized_source",return_value=(old,{})),\
             mock.patch.object(v.proof,"screen_new",return_value=(fresh,{})):
            with self.assertRaises(ValueError):
                v.audit(Path("/tmp/old"),Path("/tmp/audit"),
                        Path("/tmp/fresh"),Path("/tmp/screen"),Path("/tmp/gate"))

if __name__=="__main__":
    main()
