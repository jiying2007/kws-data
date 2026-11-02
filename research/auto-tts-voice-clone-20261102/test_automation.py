"""Contract tests use artificial audio and mock ASR; no neural models are run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import campaign as c
import runner
import screen
import asr_dispatch as dispatch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def example_plan():
    return c.build_plan(c.example_voices(),batch_id="pilot001")


def wave_writer(folder,output,rate,metrics):
    # A fake writer for the fixture only; not the actual 24k -> 16k production path.
    pcm=struct.pack("<h",3200)*16000
    path=folder/"audio-16k.wav"
    with wave.open(str(path),"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(pcm)
    body=path.read_bytes()
    return {"wav_sha256":sha(body),"wav_bytes":len(body),
        "pcm_sha256":sha(pcm),"audio_frames":16000,"sample_rate_hz":16000,
        "native_sha256":"a"*64,"native_bytes":5}


class FakeTTS:
    def __init__(self,invalid=False):
        self.calls=[]
        self.invalid=invalid

    def synthesize(self,cell,voice):
        self.calls.append((cell["cell_id"],voice["voice_id"]))
        return ([1.0] if self.invalid else [0.1])*24000,24000


def synthesize_one(plan,root,writer=wave_writer):
    backend=FakeTTS()
    result=runner.run_batch(plan,root,backend,max_calls=1,writer=writer)
    cell=next(cel for cel in plan["cells"]
              if (root/"cells"/cel["cell_id"]/"completed.json").exists())
    return backend,result,cell


def native_obs(cell,metadata,engine,text):
    return {"schema":screen.SCHEMA,"engine":engine,
       "model_id":screen.MODEL_IDS[engine],"model_revision":"a"*40,
       "audio_id":cell["cell_id"],"wav_sha256":metadata["wav_sha256"],
       "pcm_sha256":metadata["pcm_sha256"],"audio_frames":metadata["audio_frames"],
       "status":"success","decoder_completed":True,
       "processed_start_frame":0,"processed_end_frame":metadata["audio_frames"],
       "actual_text":text,"raw_native_sha256":"a"*64,"attempts":1}


def fake_adapters():
    return {"schema":dispatch.LOCK_SCHEMA,"adapters":[{
        "engine":engine,"model_id":screen.MODEL_IDS[engine],"model_revision":"a"*40,
        "model_assets_sha256":"a"*64,"executable":"/fixture/asr",
        "executable_sha256":"b"*64,"argv":["--audio","{audio}","--output","{output}"],
        "timeout_seconds":5} for engine in c.ENGINES]}


def add_asr(asr_root,cell,metadata,texts):
    adapters=fake_adapters()
    expected={"plan_sha256":sha(c.canonical(example_plan())),
              "adapter_lock_sha256":sha(c.canonical(adapters))}
    directory=asr_root/cell["cell_id"];directory.mkdir(parents=True,exist_ok=True)
    (asr_root/"plan-lock.json").write_bytes(c.canonical({
        "schema":"asr-autonomous-lock-v1",**expected,"training_admitted":False}))
    for engine,text in zip(c.ENGINES,texts):
        folder=directory/engine;folder.mkdir(exist_ok=True)
        observation=native_obs(cell,metadata,engine,text)
        native={key:observation[key] for key in ("actual_text","decoder_completed",
            "processed_start_frame","processed_end_frame","audio_id",
            "model_id","model_revision","engine")}
        raw=c.canonical(native);(folder/"native.json").write_bytes(raw)
        observation["raw_native_sha256"]=sha(raw)
        (directory/(engine+".json")).write_bytes(c.canonical(observation))
        (folder/"started.json").write_bytes(c.canonical({
            "schema":"asr-native-started-v1",**expected,"engine":engine,
            "model_id":screen.MODEL_IDS[engine],"model_revision":"a"*40,
            "model_assets_sha256":"a"*64,"audio_id":cell["cell_id"],
            "wav_sha256":metadata["wav_sha256"],"pcm_sha256":metadata["pcm_sha256"],
            "audio_frames":metadata["audio_frames"],"attempts":1,"training_admitted":False}))
        (folder/"completed.json").write_bytes(c.canonical({
            **expected,"status":"NATIVE_ASR_COMPLETE_UNADMITTED",
            "native_bytes":len(raw),"native_sha256":sha(raw),
            "observation_sha256":sha(c.canonical(observation)),"training_admitted":False}))


class CampaignTests(unittest.TestCase):
    def test_bounded_matrix_and_automatic_model_calls(self):
        plan=example_plan()
        self.assertEqual(len(plan["cells"]),24)
        self.assertEqual(plan["budget"]["max_tts_calls"],24)
        self.assertEqual(plan["budget"]["max_asr_calls"],72)
        self.assertEqual({p["role"] for p in plan["voices"]},set(c.ROLES))
        self.assertEqual(sum(row["keyword_id"]==1 for row in plan["cells"]),8)
        self.assertEqual(sum(row["keyword_id"]==2 for row in plan["cells"]),4)
        self.assertTrue(all(row["training_admitted"] is False and
            row["actual_spoken_text"] is None for row in plan["cells"]))
        self.assertEqual(c.validate_plan(plan),plan)

    def test_defaults_pin_existing_qwen_custom_voice_revision(self):
        for voice in c.example_voices():
            self.assertEqual(voice["model_id"],c.MODEL_FAMILIES["preset"])
            self.assertEqual(voice["model_revision"],c.PRESET_REVISION)
            self.assertEqual(voice["model_license"],"Apache-2.0")

    def test_deterministic_plan_and_seed_change_by_round(self):
        a=example_plan()
        self.assertEqual(c.canonical(a),c.canonical(example_plan()))
        b=c.build_plan(c.example_voices(),round_id=1,batch_id="pilot001")
        self.assertEqual(len(b["cells"]),12)
        self.assertFalse({r["cell_id"] for r in a["cells"]}&
                         {r["cell_id"] for r in b["cells"]})
        self.assertTrue(all(row["kind"]=="wake" for row in b["cells"]))
        with self.assertRaises(ValueError):
            c.build_plan(c.example_voices(),round_id=2)

    def test_same_voice_ancestry_never_crosses_roles(self):
        voices=c.example_voices()
        candidate=copy.deepcopy(voices[0])
        candidate["voice_id"]="another-id"
        candidate["role"]="development-only"
        with self.assertRaises(ValueError):
            c.validate_voices([voices[0],candidate])

    def test_unknown_preset_model_license_and_persona_rejected(self):
        for key,value in (("preset","nonexistent"),("model_license","Unknown"),
                          ("model_id","unknown"),("model_revision","main"),
                          ("role","qualification")):
            voices=c.example_voices()
            voices[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                c.build_plan(voices)

    def test_design_generation_allowed_only_new_synthetic_persona(self):
        voice={"voice_id":"studio-adult-a","mode":"design","preset":None,
           "description":"成年女性普通话，轻声柔和、语速自然，没有模仿具体人声的要求",
           "reference":None,"source_family_id":"qwen3-design-studio-adult-a",
           "role":"train-candidate","model_id":c.MODEL_FAMILIES["design"],
           "model_revision":"c"*40,"model_license":"Apache-2.0"}
        # Refuse impersonation language; a synthetically constructed persona
        # cannot claim a real person's independent identity.
        voice["description"]="成年女性普通话，稳定自然的轻声和清晰吐字"
        self.assertEqual(len(c.build_plan([voice])["cells"]),6)
        voice["description"]="请模仿一位真人播音员的声音效果"
        with self.assertRaises(ValueError):c.build_plan([voice])

    def test_clone_requires_synthetic_reference_and_same_ancestry(self):
        ref={"audio_id":"c-"+"1"*24,"wav_sha256":"a"*64,
             "pcm_sha256":"b"*64,"frames":16000,"sample_rate_hz":16000,
             "reference_text":"你好小窝","source_family_id":"qwen3-preset-ryan",
             "source_kind":"synthesized_and_machine_screened",
             "screening_sha256":"c"*64,
             "rights_basis":"synthetic_model_output_research_only",
             "real_person_identity":None,"training_admitted":False}
        voice={"voice_id":"ryan-clone","mode":"clone","preset":None,
          "description":None,"reference":ref,
          "source_family_id":ref["source_family_id"],
          "role":"train-candidate",
          "model_id":c.MODEL_FAMILIES["clone"],
          "model_revision":"d"*40,"model_license":"Apache-2.0"}
        self.assertEqual(len(c.build_plan([voice])["cells"]),6)
        for key,val in (("source_kind","human"),("real_person_identity","someone"),
                        ("reference_text","你好小屋"),
                        ("rights_basis","unknown")):
            candidate=copy.deepcopy(voice)
            candidate["reference"][key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):
                c.build_plan([candidate])
        candidate=copy.deepcopy(voice)
        candidate["source_family_id"]="another-source"
        with self.assertRaises(ValueError):c.build_plan([candidate])

    def test_typography_only_and_no_homophone_invention(self):
        self.assertEqual(c.normalize("你 好，小窝。"),"你好小窝")
        self.assertNotEqual(c.normalize("你好小屋"),"你好小窝")
        self.assertNotEqual(c.normalize("你好小沃"),"你好小窝")
        self.assertNotEqual(c.normalize("你好小吴"),"你好小窝")

    def test_plan_tamper_fails_closed(self):
        x=example_plan()
        for mutation in ("seed","status","trained","kind","budget","removed"):
            bad=copy.deepcopy(x)
            if mutation=="seed":bad["cells"][0]["seed"]+=1
            elif mutation=="status":bad["cells"][0]["training_admitted"]=True
            elif mutation=="trained":bad["training_admitted"]=True
            elif mutation=="kind":bad["cells"][0]["kind"]="wake"
            elif mutation=="budget":bad["budget"]["max_asr_calls"]+=1
            else:bad["cells"].pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                c.validate_plan(bad)

    def test_nonfinite_and_duplicate_json_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data.json"
            for body in (b'{"a":1,"a":2}',b'{"n":NaN}',b'{"n":1e999}',b'\xff'):
                p.write_bytes(body)
                with self.subTest(body=body),self.assertRaises(ValueError):
                    c.read_json(p)

    def test_no_cost_escalation_or_unbounded_campaign(self):
        with self.assertRaises(ValueError):c.build_plan(c.example_voices(),round_id=-1)
        with self.assertRaises(ValueError):c.build_plan(c.example_voices()*30)
        self.assertEqual(example_plan()["budget"]["max_paid_usd"],0)


class WorkerTests(unittest.TestCase):
    def test_technical_wave_checks(self):
        row=runner.quality([0.12]*24000,24000)
        self.assertTrue(row["technical_audio_pass"])
        for samples,rate in (([0.0]*24000,24000),
                             ([1.0]*24000,24000),
                             ([float("nan")]*24000,24000),
                             ([0.1]*24000,16000),
                             ([0.1]*100,24000),
                             ([0.1]*300000,24000)):
            with self.subTest(n=len(samples),rate=rate),self.assertRaises(ValueError):
                runner.quality(samples,rate)

    def test_generated_receipt_and_resume_zero_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=example_plan();root=Path(tmp)/"tts"
            engine,first,cell=synthesize_one(plan,root)
            self.assertEqual(first["attempts_this_call"],1)
            self.assertEqual(first["total_completed"],1)
            self.assertEqual(len(engine.calls),1)
            second=runner.run_batch(plan,root,engine,max_calls=0,writer=wave_writer)
            self.assertEqual(second["attempts_this_call"],0)
            done=c.read_json(root/"cells"/cell["cell_id"]/"completed.json")
            self.assertFalse(done["training_admitted"])
            self.assertIsNone(done["actual_text"])
            plan["batch_id"]="bad"
            with self.assertRaises(ValueError):
                runner.run_batch(plan,root,engine,max_calls=1,writer=wave_writer)

    def test_failed_first_attempt_never_rerun_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=example_plan();root=Path(tmp)/"tts"
            engine=FakeTTS(invalid=True)
            r=runner.run_batch(plan,root,engine,max_calls=1,writer=wave_writer)
            self.assertEqual(r["total_completed"],0)
            self.assertEqual(r["total_started"],1)
            r2=runner.run_batch(plan,root,engine,max_calls=0,writer=wave_writer)
            self.assertEqual(r2["attempts_this_call"],0)
            self.assertEqual(len(engine.calls),1)

    def test_generator_stages_do_not_admit_without_human(self):
        x=example_plan()
        with tempfile.TemporaryDirectory() as tmp:
            _,_,cell=synthesize_one(x,Path(tmp)/"gen")
            p=Path(tmp)/"gen/cells"/cell["cell_id"]/"started.json"
            self.assertEqual(c.read_json(p)["retry_count"],0)
            self.assertNotIn("actual_text",c.read_json(p))

    def test_mock_model_lock_verified_without_loading_weights(self):
        plan=example_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"assets";mdl=root/"model";mdl.mkdir(parents=True)
            b=b"fake-model-test-fixture";(mdl/"test.safetensors").write_bytes(b)
            lock={"schema":runner.MODEL_LOCK_SCHEMA,"models":[{
                "model_id":c.MODEL_FAMILIES["preset"],"revision":c.PRESET_REVISION,
                "relative_dir":"model",
                "files":[{"path":"test.safetensors","bytes":len(b),"sha256":sha(b)}]}]}
            result=runner.validate_models(plan,lock,root)
            self.assertEqual(len(result),1)
            (mdl/"unexpected").write_bytes(b"additional")
            with self.assertRaises(ValueError):
                runner.validate_models(plan,lock,root)

    def test_undeclared_model_symlink_and_tampering_blocked(self):
        plan=example_plan()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"model";folder.mkdir()
            original=Path(tmp)/"original";original.write_bytes(b"fake")
            (folder/"file").symlink_to(original)
            lock={"schema":runner.MODEL_LOCK_SCHEMA,"models":[{
               "model_id":c.MODEL_FAMILIES["preset"],"revision":c.PRESET_REVISION,
               "relative_dir":"model",
               "files":[{"path":"file","bytes":4,"sha256":sha(b"fake")}]}]}
            with self.assertRaises(ValueError):
                runner.validate_models(plan,lock,tmp)


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plan=example_plan()
        self.generation=self.root/"tts"
        _,_,self.cell=synthesize_one(self.plan,self.generation)
        self.media=screen._audio_metadata(
            self.generation/"cells"/self.cell["cell_id"]/"audio-16k.wav")
        self.asr=self.root/"asr"
        self.config=fake_adapters()

    def test_strict_three_engine_target_match_candidate_only(self):
        add_asr(self.asr,self.cell,self.media,[self.cell["text"]]*3)
        row=screen.screen_cell(self.cell,self.generation,self.asr,adapters=self.config,
            expected_plan_sha256=sha(c.canonical(self.plan)))
        expected=("MACHINE_SCREENED_POSITIVE_CANDIDATE" if
                  self.cell["kind"]=="wake" else "MACHINE_SCREENED_CONFUSABLE_CANDIDATE")
        self.assertEqual(row["status"],expected)
        self.assertIs(row["training_admitted"],False)
        self.assertIs(row["automated_label_is_human_gold"],False)

    def test_asr_missing_engine_never_passes(self):
        add_asr(self.asr,self.cell,self.media,[self.cell["text"]])
        row=screen.screen_cell(self.cell,self.generation,self.asr,adapters=self.config,
            expected_plan_sha256=sha(c.canonical(self.plan)))
        self.assertEqual(row["status"],"INCOMPLETE_ASR_HOLD")
        self.assertFalse(row["training_admitted"])

    def test_two_of_three_does_not_auto_label(self):
        add_asr(self.asr,self.cell,self.media,[self.cell["text"],self.cell["text"],"你好小屋"])
        result=screen.screen_cell(self.cell,self.generation,self.asr,adapters=self.config,
            expected_plan_sha256=sha(c.canonical(self.plan)))
        self.assertTrue(result["review_required"])
        self.assertEqual(result["status"],"DISAGREEMENT_REVIEW_OR_NEW_GENERATION")
        self.assertFalse(result["qualification_allowed"])

    def test_wrong_homophones_are_rejected_without_automatic_substitutions(self):
        add_asr(self.asr,self.cell,self.media,["你好小屋","你好小吴","你好小沃"])
        x=screen.screen_cell(self.cell,self.generation,self.asr,adapters=self.config,
            expected_plan_sha256=sha(c.canonical(self.plan)))
        self.assertEqual(x["status"],"REJECT_TTS_PRONUNCIATION_OR_ASR_DISAGREEMENT")

    def test_bad_full_frame_or_fake_asr_model_rejected(self):
        for key,val in (("processed_end_frame",self.media["audio_frames"]-1),
                        ("model_id","other"),("attempts",2),("wav_sha256","0"*64),
                        ("decoder_completed",False)):
            entry=native_obs(self.cell,self.media,"qwen06",self.cell["text"])
            entry[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):
                screen.validate_native_observation(entry,self.cell,self.media)

    def test_audio_drift_is_detected_before_trusting_asr(self):
        p=self.generation/"cells"/self.cell["cell_id"]/"audio-16k.wav"
        raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaises(ValueError):
            screen.screen_cell(self.cell,self.generation,self.asr,adapters=self.config,
            expected_plan_sha256=sha(c.canonical(self.plan)))

    def test_one_bounded_refinement_only_for_failed_k1_train(self):
        rows=[]
        for cell in self.plan["cells"]:
            status="NO_TTS_RESULT_OR_FAILURE"
            if cell["keyword_id"]==1 and cell["role"]=="train-candidate":
                status="REJECT_TTS_PRONUNCIATION_OR_ASR_DISAGREEMENT"
            rows.append({"cell_id":cell["cell_id"],"status":status})
        report={"schema":"autonomous-synthetic-kws-screen-v1",
                "plan_sha256":sha(c.canonical(self.plan)),"rows":rows,
                "training_admitted":False}
        next_plan=screen.plan_refinement(self.plan,report)
        self.assertEqual(next_plan["round_id"],1)
        self.assertEqual(len(next_plan["voices"]),3)
        self.assertEqual(len(next_plan["cells"]),9)
        with self.assertRaises(ValueError):
            screen.plan_refinement(next_plan,report)

    def test_clone_reference_from_genuine_machine_screen_only(self):
        if self.cell["kind"]!="wake":
            # The primary test is only meaningful for a wake slot.
            cell=next(row for row in self.plan["cells"] if row["kind"]=="wake")
            # We must create its genuinely matched generated receipt, not relabel a different slot.
            backend=FakeTTS()
            runner.run_batch(self.plan,self.generation,backend,max_calls=24,writer=wave_writer)
            self.cell=cell
            self.media=screen._audio_metadata(self.generation/"cells"/cell["cell_id"]/"audio-16k.wav")
        add_asr(self.asr,self.cell,self.media,[self.cell["text"]]*3)
        result=screen.screen_cell(self.cell,self.generation,self.asr,adapters=self.config,
            expected_plan_sha256=sha(c.canonical(self.plan)))
        self.assertEqual(result["status"],"MACHINE_SCREENED_POSITIVE_CANDIDATE")
        out=self.root/"reference"
        receipt=screen.create_clone_reference(self.cell,result,self.generation,out)
        raw,rate=runner.validate_reference(receipt,out)
        self.assertEqual(len(raw),32000)
        self.assertEqual(rate,16000)
        self.assertIsNone(receipt["real_person_identity"])
        # No new independent speaker or qualification authority.
        clone={"voice_id":"example-clone","mode":"clone","preset":None,
            "description":None,"reference":receipt,
            "source_family_id":self.cell["source_family_id"],
            "role":self.cell["role"],"model_id":c.MODEL_FAMILIES["clone"],
            "model_revision":"a"*40,"model_license":"Apache-2.0"}
        self.assertEqual(len(c.build_plan([clone])["cells"]),6)
        (out/(receipt["audio_id"]+".screen.json")).write_bytes(b"{}")
        with self.assertRaises(ValueError):
            runner.validate_reference(receipt,out)



    def _screen_bound(self, adapters=None):
        return screen.screen_cell(self.cell,self.generation,self.asr,
            adapters=self.config if adapters is None else adapters,
            expected_plan_sha256=sha(c.canonical(self.plan)))

    def test_native_receipt_chain_mutations_are_rejected(self):
        add_asr(self.asr,self.cell,self.media,[self.cell["text"]]*3)
        folder=self.asr/self.cell["cell_id"]/"qwen06"
        cases=[
            (self.asr/self.cell["cell_id"]/"qwen06.json","actual_text","different"),
            (folder/"native.json","actual_text","different"),
            (folder/"completed.json","native_bytes",1),
            (folder/"completed.json","observation_sha256","0"*64),
            (folder/"started.json","audio_id","wrong"),
            (folder/"started.json","wav_sha256","0"*64),
            (folder/"started.json","pcm_sha256","0"*64),
            (folder/"started.json","audio_frames",1),
            (folder/"started.json","attempts",True),
            (folder/"started.json","adapter_lock_sha256","0"*64),
            (folder/"completed.json","plan_sha256","0"*64)]
        for path,key,value in cases:
            raw=path.read_bytes()
            try:
                body=c.read_json(path);body[key]=value;path.write_bytes(c.canonical(body))
                with self.subTest(key=key),self.assertRaises(ValueError):
                    self._screen_bound()
            finally:path.write_bytes(raw)
        for path in (folder/"started.json",folder/"native.json",folder/"completed.json",
                     self.asr/"plan-lock.json"):
            raw=path.read_bytes();path.unlink()
            try:
                with self.subTest(missing=path.name),self.assertRaises((ValueError,OSError)):
                    self._screen_bound()
            finally:path.write_bytes(raw)

    def test_receipt_json_files_reject_links_and_nonregular_paths(self):
        add_asr(self.asr,self.cell,self.media,[self.cell["text"]]*3)
        folder=self.asr/self.cell["cell_id"]/"qwen06"
        for path in (self.asr/"plan-lock.json",folder/"started.json",
                     folder/"completed.json"):
            for kind in ("symlink","hardlink","directory"):
                raw=path.read_bytes()
                backup=path.with_name(path.name+".original")
                path.rename(backup)
                try:
                    if kind=="symlink":
                        path.symlink_to(backup)
                    elif kind=="hardlink":
                        os.link(backup,path)
                    else:
                        path.mkdir()
                    with self.subTest(receipt=path.name,kind=kind),self.assertRaises((ValueError,OSError)):
                        self._screen_bound()
                    self.assertEqual(backup.read_bytes(),raw)
                finally:
                    if path.is_dir() and not path.is_symlink():
                        path.rmdir()
                    else:
                        path.unlink()
                    backup.rename(path)
        self.assertEqual(self._screen_bound()["observed_engines"],3)

    def test_all_adapter_identity_fields_bound_for_report(self):
        add_asr(self.asr,self.cell,self.media,[self.cell["text"]]*3)
        changes={"model_revision":"b"*40,"model_assets_sha256":"b"*64,
                 "executable":"/different/asr","executable_sha256":"c"*64,
                 "argv":["--wav","{audio}","--native","{output}"],"timeout_seconds":6}
        for key,value in changes.items():
            lock=copy.deepcopy(self.config);lock["adapters"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                self._screen_bound(lock)

    def test_generator_plan_and_missing_root_lock_fail_closed(self):
        add_asr(self.asr,self.cell,self.media,[self.cell["text"]]*3)
        path=self.generation/"cells"/self.cell["cell_id"]/"completed.json"
        original=path.read_bytes();body=c.read_json(path);body["plan_sha256"]="0"*64
        path.write_bytes(c.canonical(body))
        try:
            with self.assertRaises(ValueError):self._screen_bound()
        finally:path.write_bytes(original)
        lock=self.asr/"plan-lock.json";body=c.read_json(lock)
        del body["adapter_lock_sha256"];lock.write_bytes(c.canonical(body))
        with self.assertRaises(ValueError):self._screen_bound()


class ASRDispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plan=example_plan()
        self.generation=self.root/"tts"
        _,_,self.cell=synthesize_one(self.plan,self.generation)
        self.adapters=self.root/"asr-app"
        self.adapters.write_bytes(b"#!/usr/bin/env python3\n# synthetic test stub\n")
        self.adapters.chmod(0o700)
        self.config={"schema":dispatch.LOCK_SCHEMA,
            "adapters":[{
                "engine":engine,"model_id":screen.MODEL_IDS[engine],
                "model_revision":"b"*40,"model_assets_sha256":"c"*64,
                "executable":str(self.adapters),"executable_sha256":sha(self.adapters.read_bytes()),
                "argv":["--wav","{audio}","--native-output","{output}"],
                "timeout_seconds":30}
                for engine in c.ENGINES]}

    def test_dispatch_dry_run_uses_no_models(self):
        result=dispatch.dispatch(self.plan,self.generation,self.root/"asr",self.config,
                                 max_calls=72,execute=False)
        self.assertEqual(result["actual_calls"],0)
        self.assertFalse((self.root/"asr").exists())

    def test_missing_one_pinned_engine_rejected(self):
        bad=copy.deepcopy(self.config);bad["adapters"].pop()
        with self.assertRaises(ValueError):dispatch.check_adapters(bad)

    def test_reject_target_text_in_adapter_argv(self):
        bad=copy.deepcopy(self.config)
        bad["adapters"][0]["argv"].append("你好小窝")
        with self.assertRaises(ValueError):dispatch.check_adapters(bad)

    def test_anonymous_asr_native_receipts_auto_screen_and_no_retries(self):
        calls=[]
        def executor(cmd,**kwargs):
            self.assertNotIn("你好小窝"," ".join(cmd))
            self.assertNotIn("小窝小窝"," ".join(cmd))
            calls.append(cmd)
            raw=Path(cmd[-1])
            item=next(r for r in self.config["adapters"] if r["executable"]==cmd[0]
                and r["engine"]==c.ENGINES[len(calls)-1])
            entry={"actual_text":self.cell["text"],"decoder_completed":True,
                   "processed_start_frame":0,"processed_end_frame":16000,
                   "audio_id":self.cell["cell_id"],"model_id":item["model_id"],
                   "model_revision":item["model_revision"],"engine":item["engine"]}
            raw.write_bytes(c.canonical(entry))
            return SimpleNamespace(returncode=0)
        root=self.root/"asr"
        with patch.dict(os.environ,{"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}):
            r=dispatch.dispatch(self.plan,self.generation,root,self.config,
                                 max_calls=3,execute=True,executor=executor)
            r2=dispatch.dispatch(self.plan,self.generation,root,self.config,
                                 max_calls=3,execute=True,executor=executor)
        self.assertEqual(r["actual_calls"],3)
        self.assertEqual(r2["actual_calls"],0)
        self.assertEqual(len(calls),3)
        screened=screen.screen_cell(self.cell,self.generation,root,adapters=self.config,
            expected_plan_sha256=sha(c.canonical(self.plan)))
        self.assertEqual(screened["observed_engines"],3)
        self.assertFalse(screened["training_admitted"])

    def test_failed_native_asr_is_consumed_not_replayed(self):
        def fail(cmd,**kwargs):return SimpleNamespace(returncode=2)
        root=self.root/"asr"
        with patch.dict(os.environ,{"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}):
            first=dispatch.dispatch(self.plan,self.generation,root,self.config,
                                    max_calls=1,execute=True,executor=fail)
            second=dispatch.dispatch(self.plan,self.generation,root,self.config,
                                     max_calls=0,execute=True,executor=fail)
        self.assertEqual(first["actual_calls"],1)
        self.assertEqual(second["actual_calls"],0)
        self.assertTrue((root/self.cell["cell_id"]/"qwen06"/"failed.json").exists())



class ControllerTests(unittest.TestCase):
    """Real orchestration functions, mock generation/recognition only."""

    def setUp(self):
        import autopilot
        import prepare_models
        self.autopilot=autopilot
        self.prepare_models=prepare_models
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        voice=c.example_voices()[0]
        self.plan=c.build_plan([voice],batch_id="ci-autopilot")
        self.lock={"schema":runner.MODEL_LOCK_SCHEMA,"models":[]}
        self.asset_root=self.root/"models"
        self.asset_root.mkdir()
        for kind,mid in (("preset",c.MODEL_FAMILIES["preset"]),
                         ("clone",c.MODEL_FAMILIES["clone"])):
            rev=c.PRESET_REVISION if kind=="preset" else "b"*40
            folder=self.asset_root/(kind+"-model")
            folder.mkdir()
            blob=(kind+"-fixture").encode()
            (folder/"model.safetensors").write_bytes(blob)
            self.lock["models"].append({"model_id":mid,"revision":rev,
                "relative_dir":kind+"-model",
                "files":[{"path":"model.safetensors","bytes":len(blob),"sha256":sha(blob)}]})
        self.adapter_exe=self.root/"fake-asr-executable"
        self.adapter_exe.write_bytes(b"#!/bin/sh\n# test fixture only\n")
        self.adapter_exe.chmod(0o700)
        self.adapters={"schema":dispatch.LOCK_SCHEMA,"adapters":[{
            "engine":name,"model_id":screen.MODEL_IDS[name],"model_revision":"b"*40,
            "model_assets_sha256":"c"*64,
            "executable":str(self.adapter_exe),
            "executable_sha256":sha(self.adapter_exe.read_bytes()),
            "argv":["--input","{audio}","--output","{output}"],
            "timeout_seconds":30} for name in c.ENGINES]}
        self.tts_backend=FakeTTS()
        self.asr_calls=[]

    def fake_executor(self,cmd,**kwargs):
        self.assertFalse(any(text in " ".join(cmd) for text in c.TARGETS))
        self.asr_calls.append(cmd)
        audio=Path(cmd[-3])
        path=Path(cmd[-1])
        engine=path.parent.name
        cell_id=path.parent.parent.name
        entry=next(r for r in self.adapters["adapters"] if r["engine"]==engine)
        with wave.open(str(audio),"rb") as stream:
            frames=stream.getnframes()
        raw={"actual_text":"你好小窝","decoder_completed":True,
            "processed_start_frame":0,"processed_end_frame":frames,
            "audio_id":cell_id,"engine":engine,
            "model_id":entry["model_id"],"model_revision":entry["model_revision"]}
        path.write_bytes(c.canonical(raw))
        return SimpleNamespace(returncode=0)

    def launch(self,*,tts,asr,clone_revision=None):
        with patch.dict(os.environ,{"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}):
            return self.autopilot.execute(self.plan,self.root/"workspace",
                self.lock,self.asset_root,self.adapters,
                max_tts=tts,max_asr=asr,rounds=1,
                clone_revision=clone_revision,
                tts_backend_factory=lambda plan:self.tts_backend,
                asr_executor=self.fake_executor,writer=wave_writer)

    def test_single_command_uses_three_blinded_asr_and_never_approves_training(self):
        first=self.launch(tts=1,asr=3)
        self.assertEqual(first["ttl_tts_started"],1)
        self.assertEqual(first["ttl_asr_started"],3)
        self.assertEqual(len(self.tts_backend.calls),1)
        self.assertEqual(len(self.asr_calls),3)
        for k in ("training_admitted","qualification_allowed","shipping_approved"):
            self.assertIs(first[k],False)
        repeated=self.launch(tts=1,asr=3)
        self.assertEqual(repeated,first)
        self.assertEqual(len(self.tts_backend.calls),1)
        self.assertEqual(len(self.asr_calls),3)

    def test_single_command_auto_clones_only_screened_synthetic_reference(self):
        result=self.launch(tts=7,asr=21,clone_revision="b"*40)
        self.assertEqual(result["ttl_tts_started"],7)
        self.assertEqual(result["ttl_asr_started"],21)
        self.assertIn("clone",result["stages"])
        references=list((self.root/"workspace/references").glob("*.reference.json"))
        self.assertEqual(len(references),1)
        ref=c.read_json(references[0])
        self.assertEqual(ref["source_family_id"],self.plan["voices"][0]["source_family_id"])
        self.assertIsNone(ref["real_person_identity"])
        self.assertTrue(all(not x["training_admitted"] for x in
             (ref,)))
        repeated=self.launch(tts=7,asr=21,clone_revision="b"*40)
        self.assertEqual(repeated,result)
        self.assertEqual(len(self.tts_backend.calls),7)
        self.assertEqual(len(self.asr_calls),21)

    def test_exhausted_controller_budget_still_rejects_adapter_drift(self):
        self.launch(tts=1,asr=3)
        before=(len(self.tts_backend.calls),len(self.asr_calls))
        self.adapters=copy.deepcopy(self.adapters)
        self.adapters["adapters"][0]["timeout_seconds"]=31
        with self.assertRaises(ValueError):
            self.launch(tts=1,asr=3)
        self.assertEqual((len(self.tts_backend.calls),len(self.asr_calls)),before)

    def test_controller_does_not_accept_changed_budget_or_source_plan_on_resume(self):
        self.launch(tts=1,asr=3)
        with self.assertRaises(ValueError):
            self.launch(tts=2,asr=3)
        original=self.plan
        self.plan=copy.deepcopy(original)
        self.plan["batch_id"]="different"
        with self.assertRaises(ValueError):
            self.launch(tts=1,asr=3)

    def test_offline_public_model_asset_manifest_is_generated_from_real_bytes(self):
        source=self.root/"locked"
        source.mkdir()
        name=self.prepare_models.destination_name(c.MODEL_FAMILIES["preset"],c.PRESET_REVISION)
        folder=source/name
        folder.mkdir()
        data=b"actually hashed fixture not a neural model"
        (folder/"model.safetensors").write_bytes(data)
        lock=self.prepare_models.lock_existing(self.plan,source)
        self.assertEqual(lock["models"][0]["files"][0]["sha256"],sha(data))
        self.assertEqual(len(runner.validate_models(self.plan,lock,source)),1)
        (folder/"extra").write_bytes(b"something")
        with self.assertRaises(ValueError):
            runner.validate_models(self.plan,lock,source)

    def test_no_models_on_automatic_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            planfile=Path(tmp)/"plan.json"
            assetfile=Path(tmp)/"asset.json"
            adapterfile=Path(tmp)/"adapter.json"
            planfile.write_bytes(c.canonical(self.plan))
            assetfile.write_bytes(c.canonical(self.lock))
            adapterfile.write_bytes(c.canonical(self.adapters))
            result=self.autopilot.main(["--plan",str(planfile),
                "--workspace",str(Path(tmp)/"nothing"),
                "--model-lock",str(assetfile),"--model-root",str(self.asset_root),
                "--adapter-lock",str(adapterfile)])
            self.assertEqual(result,0)
            self.assertFalse((Path(tmp)/"nothing").exists())


if __name__=="__main__":
    unittest.main()
