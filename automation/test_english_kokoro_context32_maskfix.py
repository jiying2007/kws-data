"""Stdlib recovery-boundary fixtures; no model or processor execution."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_kokoro_context32_maskfix_once as worker
source=worker.source


def plan():
    return json.loads((HERE/"english_kokoro_context32_maskfix_plan.json").read_bytes())


def encoded(value):
    return json.dumps(value,sort_keys=True).encode()


def zero_fixture():
    manifest={"rows":plan()["input_roster"]}
    terminal={"run":source.PRIOR_RUN,"counter_scope":"complete_terminal_receipts","declared_inputs":16,
        "declared_observation_slots":32,"new_derivative_inputs":16,"training_allowed":False}
    for key in ("actual_forward_attempts","native_forward_returns","completed","forward_attempts_upper_bound",
        "retained_forward_attempts_lower_bound","retained_native_forward_returns_lower_bound","retained_capture_files_lower_bound",
        "new_tts_calls","new_text_asr_calls","new_embedding_calls","new_training_calls","retry_count"):terminal[key]=0
    files={"RESULT.json":encoded(terminal)}
    comparison={"status":"INCOMPLETE_DIAGNOSTIC_NO_ADMISSION","actual_forward_attempts":0,
        "native_forward_returns":0,"completed_new_captures":0,"forward_attempts_upper_bound":0}
    for provider in ("phoneme","phoneme_lv60"):
        prefix="observers/"+provider+"/"
        rows=[]
        for row in manifest["rows"]:
            item={"id":row["id"],"source_id":row["source_id"],"status":"failed","input_started":True,
                "forward_attempts":0,"forward_returned":False,"capture_saved":False,"failure":source.RECOVERY["prior_error"],"retry_count":0}
            rows.append(item)
            base={"id":row["id"],"source_id":row["source_id"],"run":source.PRIOR_RUN,
                "input_pcm_sha256":row["pcm_sha256"],"input_wav_sha256":row["wav_sha256"],"input_samples":row["pcm_samples"],
                "forward_attempts":0,"forward_returned":False,"retry_count":0}
            files[prefix+row["id"]+"/STARTED.json"]=encoded(base)
            files[prefix+row["id"]+"/FAILED.json"]=encoded(base|item)
        files[prefix+"MODEL_LOAD_STARTED.json"]=encoded({"model_stage_started":True})
        files[prefix+"MODEL_LOCK.json"]=encoded({"new_model_constructions":1,"native_forward_attempts_at_initialization":0,
            "input_manifest_sha256":source.INPUT_MANIFEST_SHA256})
        files[prefix+"RESULT.json"]=encoded({"run":source.PRIOR_RUN,"input_manifest_sha256":source.INPUT_MANIFEST_SHA256,
            "status":"INCOMPLETE_DIAGNOSTIC_NO_RETRY","input_started":16,"failed":16,"declared_count":16,
            "not_started":0,"actual_forward_attempts":0,"native_forward_returns":0,"completed":0,
            "failure":None,"retry_count":0,"rows":rows})
    return files,manifest,comparison


class RecoverySourceTests(unittest.TestCase):
    def test_source_pins_disabled_and_actual_single_bound_state(self):
        frozen=plan();worker.validate_plan(frozen)
        frozen.update(enabled=False,main_before=None);worker.validate_plan(frozen)
        frozen.update(enabled=True,main_before=source.MAIN_BEFORE);worker.validate_plan(frozen)

    def test_other_predecessor_cannot_consume_same_recovery(self):
        frozen=plan();frozen.update(enabled=True,main_before="a"*40)
        with self.assertRaisesRegex(ValueError,"ONE_FIXED_NEW_PREDECESSOR"):worker.validate_plan(frozen)

    def test_actual_input_manifest_and_rows_cannot_change(self):
        for mutate in (lambda p:p.__setitem__("input_manifest_sha256","0"*64),
                       lambda p:p["input_roster"][0].__setitem__("pcm_sha256","0"*64),
                       lambda p:p["input_roster"].reverse()):
            frozen=plan();mutate(frozen)
            with self.assertRaises(ValueError):worker.validate_plan(frozen)

    def test_original_run_and_new_run_never_conflated(self):
        frozen=plan();frozen["prior_run_identity"]["commit"]="1"*40
        with self.assertRaisesRegex(ValueError,"FIXED_ACTUAL_ZERO"):worker.validate_plan(frozen)

    def test_prior_consumption_cannot_be_erased(self):
        for key in ("prior_processor_attempts","prior_model_constructions"):
            frozen=plan();frozen[key]=0
            with self.assertRaisesRegex(ValueError,"FINITE_FIRST"):worker.validate_plan(frozen)

    def test_no_new_derivation_or_replay_budget(self):
        for key in ("new_derivations","new_tts_calls","forward_replays","retry_count","new_training_calls"):
            frozen=plan();frozen[key]=1
            with self.assertRaises(ValueError):worker.validate_plan(frozen)

    def test_model_and_source_helpers_are_exact(self):
        frozen=plan();self.assertEqual(len(frozen["helper_sha256"]),28)
        self.assertNotIn("automation/english_kokoro_context32_plan.json",frozen["helper_sha256"])
        self.assertNotIn(".github/workflows/english-kokoro-context32-once.yml",frozen["helper_sha256"])
        frozen["observers"]["phoneme"]["model_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"UNCHANGED_TWO_MODELS"):worker.validate_plan(frozen)

    def test_only_first_public_successor_guard(self):
        frozen=plan();frozen.update(enabled=True,main_before=source.MAIN_BEFORE)
        env={"GITHUB_REPOSITORY":"jiying2007/kws-data","GITHUB_SHA":"2"*40,"GITHUB_RUN_ID":"123",
            "GITHUB_RUN_ATTEMPT":"1","GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main"}
        event={"before":source.MAIN_BEFORE,"after":env["GITHUB_SHA"],"repository":{"full_name":env["GITHUB_REPOSITORY"],"private":False}}
        self.assertEqual(worker.execution_guard(frozen,env,event)["run_id"],"123")
        for key,value in (("GITHUB_RUN_ATTEMPT","2"),("GITHUB_EVENT_NAME","workflow_dispatch"),("GITHUB_REF","refs/heads/other")):
            with self.assertRaises(ValueError):worker.execution_guard(frozen,env|{key:value},event)
        with self.assertRaises(ValueError):worker.execution_guard(frozen,env,event|{"before":"3"*40})

    def test_disabled_plan_cannot_start(self):
        frozen=plan();frozen.update(enabled=False,main_before=None)
        with self.assertRaises(ValueError):worker.execution_guard(frozen,{}, {})

    def test_prior_zip_requires_actual_original_api_bytes(self):
        with self.assertRaisesRegex(ValueError,"EXACT_ACTUAL_PRIOR_FINAL_ZIP"):source.prior_archive(b"repacked fixture is not the raw API ZIP")

    def test_complete_zero_forward_means_started_preprocessing_not_unstarted(self):
        files,manifest,comparison=zero_fixture()
        proof=source.zero_forward_evidence(files,manifest,comparison)
        self.assertEqual(sum(p["prior_model_constructions"] for p in proof.values()),2)
        self.assertEqual(sum(p["prior_input_started"] for p in proof.values()),32)
        self.assertEqual(sum(p["prior_failed_preprocessing"] for p in proof.values()),32)
        self.assertEqual(sum(p["prior_forward_attempts"] for p in proof.values()),0)

    def test_any_consumed_forward_blocks_recovery(self):
        files,manifest,comparison=zero_fixture()
        old=json.loads(files["RESULT.json"]);old["actual_forward_attempts"]=1;files["RESULT.json"]=encoded(old)
        with self.assertRaisesRegex(ValueError,"PRIOR_COMPLETE_ZERO"):source.zero_forward_evidence(files,manifest,comparison)

    def test_forward_marker_cannot_hide_behind_zero_count(self):
        for name in ("FORWARD_STARTED.json","FORWARD_RETURNED.json","CAPTURE.json","logits.f32","frame-ids.json"):
            files,manifest,comparison=zero_fixture()
            files["observers/phoneme/"+manifest["rows"][0]["id"]+"/"+name]=b"{}"
            with self.assertRaises(ValueError):source.zero_forward_evidence(files,manifest,comparison)

    def test_actual_model_construction_is_required_and_not_renamed_zero(self):
        files,manifest,comparison=zero_fixture()
        lock=json.loads(files["observers/phoneme/MODEL_LOCK.json"]);lock["new_model_constructions"]=0
        files["observers/phoneme/MODEL_LOCK.json"]=encoded(lock)
        with self.assertRaisesRegex(ValueError,"PRIOR_MODEL_CONSTRUCTION"):source.zero_forward_evidence(files,manifest,comparison)

    def test_missing_failed_receipt_does_not_prove_no_forward(self):
        files,manifest,comparison=zero_fixture()
        del files["observers/phoneme/"+manifest["rows"][0]["id"]+"/FAILED.json"]
        with self.assertRaisesRegex(ValueError,"EXACT_OLD_INPUT_STARTED"):source.zero_forward_evidence(files,manifest,comparison)

    def test_other_exception_does_not_use_this_recovery(self):
        files,manifest,comparison=zero_fixture()
        terminal=json.loads(files["observers/phoneme/RESULT.json"]);terminal["rows"][0]["failure"]="some other error"
        files["observers/phoneme/RESULT.json"]=encoded(terminal)
        with self.assertRaisesRegex(ValueError,"EXACT_PREPROCESSOR_FAILURE"):source.zero_forward_evidence(files,manifest,comparison)

    def test_zero_forward_receipt_must_bind_same_actual_full_pcm(self):
        files,manifest,comparison=zero_fixture()
        path="observers/phoneme/"+manifest["rows"][0]["id"]+"/FAILED.json"
        failed=json.loads(files[path]);failed["input_pcm_sha256"]="0"*64;files[path]=encoded(failed)
        with self.assertRaisesRegex(ValueError,"PRIOR_ZERO_FORWARD_INPUT_BINDING"):source.zero_forward_evidence(files,manifest,comparison)

    def test_create_only_and_closed_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source.put(root/"a",b"kept");source.once(root/"FILES.json",worker.file_manifest(root));worker.closed_readback(root)
            with self.assertRaises(FileExistsError):source.put(root/"a",b"overwrite")
            (root/"a").write_bytes(b"mutated")
            with self.assertRaises(ValueError):worker.closed_readback(root)

    def test_output_limit_accepts_raw_prior_and_new_final_capacity(self):
        self.assertLess(source.RECOVERY["prior_artifact"]["bytes"],source.MAX_FILE)
        self.assertEqual(source.MAX_FILE,256*1024**2)
        self.assertEqual(source.MAX_EXPANDED,1024*1024**2)

    def test_no_numerical_or_model_import_in_source_fixtures(self):
        self.assertFalse(any(name in sys.modules for name in ("numpy","scipy","torch","kokoro","transformers")))


if __name__=="__main__":unittest.main()
