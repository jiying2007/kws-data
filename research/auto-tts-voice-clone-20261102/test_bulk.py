"""Deterministic synthetic-only fixtures for many-speaker cloning and ASR.

No human voices, public dataset downloads, neural imports, paid calls,
real ASR or production labels are used in these tests.
"""
from __future__ import annotations
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

import bulk_pool as p
import bulk_ingest as ingest
import bulk_runner as worker
import bulk_asr as a
import asr_dispatch
import runner
import screen
import campaign


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_ref(path, seed):
    raw=struct.pack("<h", seed)*16000
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(raw)
    return p.measure_reference(path.parent,path.name)


class FakeClone:
    def __init__(self):
        self.calls=[]
        self.prepared=set()
    def synthesize(self,cell,ref,root):
        self.prepared.add(ref["reference_id"])
        self.calls.append(cell["cell_id"])
        return [0.125]*24000,24000


def writer(folder,audio,rate,metrics):
    pcm=struct.pack("<h",4000)*16000
    path=folder/"audio-16k.wav"
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
        f.writeframes(pcm)
    data=path.read_bytes()
    return {"wav_sha256":hash_bytes(data),"wav_bytes":len(data),
       "pcm_sha256":hash_bytes(pcm),"audio_frames":16000,
       "sample_rate_hz":16000,"native_sha256":"a"*64,"native_bytes":5}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.references=[]
        for idx in range(2):
            name=f"ref{idx}.wav"
            meta=write_ref(self.root/name,3000+idx*1000)
            id=f"ref-{idx:04d}"
            role="train-candidate" if idx==0 else "development-only"
            rights={
                "schema":"kws-clone-source-rights-review-v1",
                "reference_id":id,
                "wav_sha256":meta["wav_sha256"],
                "source_terms_sha256":"f"*64,
                "usage":"synthetic-kws-voice-cloning",
                "basis":"dataset-terms-and-voice-rights-reviewed",
                "prohibited_uses":[],
                "status":"source-terms-reviewed",
                "rights_record_origin":"independently-reviewed-declaration"
            }
            rights_name=f"proof{idx}.json"
            body=p.canonical(rights)
            (self.root/rights_name).write_bytes(body)
            ref={"reference_id":id,
                 "speaker_group_id":f"speaker-{idx:04d}",
                 "source_family_id":f"family-{idx:04d}",
                 "dataset_id":"test-open-audio",
                 "source_type":"reviewed-open-audio",
                 "role":role,"audio_path":name,
                 "wav_sha256":meta["wav_sha256"],
                 "pcm_sha256":meta["pcm_sha256"],
                 "frames":meta["frames"],
                 "sample_rate_hz":meta["sample_rate_hz"],
                 "reference_text":None,
                 "reference_text_status":"unknown",
                 "source_license":"CC0-1.0",
                 "source_terms_sha256":"f"*64,
                 "rights_record_path":rights_name,
                 "rights_record_sha256":hash_bytes(body),
                 "parent_source_sha256":meta["wav_sha256"]}
            self.references.append(ref)
        self.sources={"schema":p.SCHEMA,
                      "references":self.references,
                      "training_admitted":False,
                      "qualification_allowed":False}
        self.plan=p.plan(self.sources,self.root,repetitions=2,
                         batch_id="pilot-01",revision="b"*40)
        self.tts=self.root/"generated"
        self.asr=self.root/"asr"
        self.backend=FakeClone()
        self.exe=self.root/"asr-adapter"
        self.exe.write_bytes(b"#!/usr/bin/env python3\n# no real model\n")
        self.exe.chmod(0o700)
        self.adapters={"schema":asr_dispatch.LOCK_SCHEMA,"adapters":[{
            "engine":engine,"model_id":screen.MODEL_IDS[engine],
            "model_revision":"a"*40,"model_assets_sha256":"a"*64,
            "executable":str(self.exe),
            "executable_sha256":hash_bytes(self.exe.read_bytes()),
            "argv":["--audio","{audio}","--out","{output}"],
            "timeout_seconds":30
        } for engine in campaign.ENGINES]}

    def gen(self,amount):
        return worker.run(self.plan,self.sources,self.root,self.tts,
                          self.backend,max_new_calls=amount,writer=writer)

    def simulate_asr(self,cmd,**kwargs):
        wav=Path(cmd[-3])
        native=Path(cmd[-1])
        cid=native.parent.parent.name
        engine=native.parent.name
        cell=next(x for x in self.plan["cells"] if x["cell_id"]==cid)
        with wave.open(str(wav),"rb") as f:
            frames=f.getnframes()
        config=next(x for x in self.adapters["adapters"] if x["engine"]==engine)
        text=cell["prompt"]
        raw={"audio_id":cid,"engine":engine,
             "model_id":config["model_id"],
             "model_revision":config["model_revision"],
             "decoder_completed":True,"processed_start_frame":0,
             "processed_end_frame":frames,"actual_text":text}
        native.write_bytes(p.canonical(raw))
        return SimpleNamespace(returncode=0)


class BulkPlanTests(Fixture):
    def test_many_speakers_many_positive_negative(self):
        self.assertEqual(len(self.plan["cells"]),64)
        count={kind:sum(x["intended_kind"]==kind for x in self.plan["cells"])
               for kind in ("positive-intent","hard-negative-intent",
                            "speech-negative-intent")}
        self.assertEqual(count,{"positive-intent":16,
                                "hard-negative-intent":32,
                                "speech-negative-intent":16})
        self.assertEqual(len({x["speaker_group_id"] for x in self.plan["cells"]}),2)
        self.assertFalse(self.plan["training_admitted"])
        self.assertIs(self.plan["source_rights_authenticated"],False)

    def test_plan_is_stable_reproducible_and_nonadmitting(self):
        p.verify_plan(self.plan)
        second=p.plan(self.sources,self.root,repetitions=2,
                      batch_id="pilot-01",revision="b"*40)
        self.assertEqual(p.canonical(self.plan),p.canonical(second))
        self.assertEqual(self.plan["budgets"]["max_retries"],0)
        self.assertFalse(self.plan["qualification_allowed"])

    def test_group_speaker_never_leaks_to_development(self):
        for field in ("source_family_id","speaker_group_id"):
            bad=copy.deepcopy(self.sources)
            bad["references"][1][field]=bad["references"][0][field]
            with self.subTest(field=field),self.assertRaises(ValueError):
                p.validate_sources(bad,self.root)

    def test_license_name_alone_never_approves_clone_rights(self):
        for mutation in ("missing","changed","forbidden","not-reviewed"):
            old=self.root/"proof0.json"
            content=old.read_bytes()
            try:
                if mutation=="missing":old.unlink()
                else:
                    record=json.loads(content)
                    if mutation=="changed":record["wav_sha256"]="0"*64
                    elif mutation=="forbidden":record["prohibited_uses"]=["voice-cloning"]
                    else:record["status"]="unreviewed"
                    newbody=p.canonical(record)
                    old.write_bytes(newbody)
                    self.sources["references"][0]["rights_record_sha256"]=hash_bytes(newbody)
                with self.subTest(mutation=mutation),self.assertRaises(
                    (ValueError,OSError)):
                    p.validate_sources(self.sources,self.root)
            finally:
                old.write_bytes(content)
                self.sources["references"][0]["rights_record_sha256"]=hash_bytes(content)

    def test_invalid_source_license_is_rejected(self):
        bad=copy.deepcopy(self.sources)
        bad["references"][0]["source_license"]="unknown"
        with self.assertRaises(ValueError):p.validate_sources(bad,self.root)

    def test_synthetic_real_voice_category_requires_distinct_basis(self):
        bad=copy.deepcopy(self.sources)
        bad["references"][0]["source_type"]="human-with-consent"
        with self.assertRaises(ValueError):p.validate_sources(bad,self.root)

    def test_wav_sha_file_mutation_and_symlink_rejected(self):
        bad=copy.deepcopy(self.sources)
        bad["references"][0]["wav_sha256"]="0"*64
        with self.assertRaises(ValueError):p.validate_sources(bad,self.root)
        (self.root/"ref0.wav").write_bytes(b"not wav")
        with self.assertRaises(ValueError):p.validate_sources(self.sources,self.root)

    def test_reference_text_unknown_cannot_invent_transcript(self):
        bad=copy.deepcopy(self.sources)
        bad["references"][0]["reference_text"]="你好小窝"
        with self.assertRaises(ValueError):p.validate_sources(bad,self.root)

    def test_plan_tamper_seed_prompt_and_budget_detected(self):
        for name in ("seed","prompt","role","status","budget","approval"):
            altered=copy.deepcopy(self.plan)
            if name=="seed":altered["cells"][0]["seed"]+=1
            elif name=="prompt":altered["cells"][0]["prompt"]="你好小屋"
            elif name=="role":altered["cells"][0]["role"]="sealed"
            elif name=="status":altered["cells"][0]["status"]="DONE"
            elif name=="budget":altered["budgets"]["max_total_calls"]+=1
            else:altered["training_admitted"]=True
            with self.subTest(name=name),self.assertRaises(ValueError):
                p.verify_plan(altered)

    def test_plan_bounds_are_bounded(self):
        for repetitions in (0,9,True):
            with self.subTest(repetitions=repetitions),self.assertRaises(ValueError):
                p.plan(self.sources,self.root,repetitions=repetitions,
                       batch_id="pilot-01",revision="b"*40)
        with self.assertRaises(ValueError):
            p.plan(self.sources,self.root,batch_id="pilot-01",
                   revision="main")
        with self.assertRaises(ValueError):
            p.plan(self.sources,self.root,batch_id="pilot-01",
                   revision="b"*40,max_calls=65)


class BulkIngestTests(Fixture):
    def test_bulk_import_anonymizes_and_automatically_creates_proofs(self):
        self.assertEqual(len(ingest._catalog({"schema":ingest.CATALOG_SCHEMA,
             "rows":[{"speaker_key":r["speaker_group_id"],"audio_path":r["audio_path"],
                "wav_sha256":r["wav_sha256"],
                "source_original_sha256":r["parent_source_sha256"],
                "reference_text":None,"reference_text_status":"unknown"}
                for r in self.references]})),2)
        catalog={"schema":ingest.CATALOG_SCHEMA,"rows":[{
            "speaker_key":r["speaker_group_id"],"audio_path":r["audio_path"],
            "wav_sha256":r["wav_sha256"],
            "source_original_sha256":r["parent_source_sha256"],
            "reference_text":None,"reference_text_status":"unknown"
            } for r in self.references]}
        rights=b"independently reviewed dataset terms fixture"
        permit={"schema":ingest.PERMIT_SCHEMA,
                "dataset_id":"synthetic-test-only",
                "source_type":"reviewed-open-audio",
                "source_license":"CC0-1.0",
                "source_terms_sha256":"c"*64,
                "rights_review_evidence_sha256":hash_bytes(rights),
                "voice_cloning_explicitly_allowed":True,
                "commercial_synthetic_kws_allowed":True,
                "forbidden_uses":[],
                "review_status":"independently-reviewed"}
        values,records=ingest.prepare(catalog,permit,rights,self.root,
                                     b"f"*32,max_refs=2)
        self.assertEqual(len(values["references"]),2)
        self.assertFalse(values["training_admitted"])
        self.assertTrue(all("speaker-000" not in row["speaker_group_id"]
                            for row in values["references"]))
        ingested=self.root/"output.json"
        ingest.publish_private(values,records,self.root,ingested)
        self.assertEqual(len(p.validate_sources(p.read_json(ingested),self.root)),2)
        with self.assertRaises(ValueError):
            ingest.publish_private(values,records,self.root,ingested)

    def test_dataset_ban_overrides_cc0_license(self):
        permit={"schema":ingest.PERMIT_SCHEMA,"dataset_id":"test-dataset",
                "source_type":"reviewed-open-audio","source_license":"CC0-1.0",
                "source_terms_sha256":"c"*64,
                "rights_review_evidence_sha256":hash_bytes(b"review"),
                "voice_cloning_explicitly_allowed":True,
                "commercial_synthetic_kws_allowed":True,
                "forbidden_uses":["voice-cloning"],
                "review_status":"independently-reviewed"}
        with self.assertRaises(ValueError):
            ingest._validate_permit(permit,hash_bytes(b"review"))

    def test_invalid_rights_and_short_pseudonym_key_rejected(self):
        with self.assertRaises(ValueError):
            ingest._validate_permit({"schema":ingest.PERMIT_SCHEMA},hash_bytes(b""))
        catalog={"schema":ingest.CATALOG_SCHEMA,"rows":[{
            "speaker_key":"one","audio_path":"ref0.wav",
            "wav_sha256":self.references[0]["wav_sha256"],
            "source_original_sha256":"a"*64,"reference_text":None,
            "reference_text_status":"unknown"}]}
        permit={"schema":ingest.PERMIT_SCHEMA,"dataset_id":"dataset-test",
            "source_type":"reviewed-open-audio","source_license":"CC0-1.0",
            "source_terms_sha256":"c"*64,
            "rights_review_evidence_sha256":hash_bytes(b"review"),
            "voice_cloning_explicitly_allowed":True,
            "commercial_synthetic_kws_allowed":True,
            "forbidden_uses":[],"review_status":"independently-reviewed"}
        with self.assertRaises(ValueError):
            ingest.prepare(catalog,permit,b"review",self.root,b"123",max_refs=2)


class BulkRuntimeTests(Fixture):
    def test_resume_never_replays_started_generation(self):
        first=self.gen(3)
        self.assertEqual(first["attempts_this_call"],3)
        self.assertEqual(first["total_started"],3)
        self.assertEqual(first["total_completed"],3)
        second=self.gen(0)
        self.assertEqual(second["attempts_this_call"],0)
        self.assertEqual(len(self.backend.calls),3)
        third=self.gen(2)
        self.assertEqual(third["total_started"],5)
        self.assertEqual(len(self.backend.calls),5)

    def test_wrong_plan_or_generation_source_is_rejected(self):
        self.gen(1)
        bad=copy.deepcopy(self.sources)
        bad["references"][0]["source_family_id"]="wrong-family"
        with self.assertRaises(ValueError):
            worker.run(self.plan,bad,self.root,self.tts,self.backend,
                       max_new_calls=1,writer=writer)
        plan=copy.deepcopy(self.plan)
        plan["cells"][0]["seed"]+=1
        with self.assertRaises(ValueError):
            worker.run(plan,self.sources,self.root,self.tts,self.backend,
                       max_new_calls=1,writer=writer)

    def test_worker_rejects_wrong_model_asset_hash(self):
        folder=self.root/"model"
        folder.mkdir()
        (folder/"config.json").write_bytes(b"synthetic file fixture")
        lock={"schema":worker.LOCK_SCHEMA,"model_id":p.MODEL,
              "revision":"b"*40,
              "files":[{"path":"config.json","sha256":"0"*64,"bytes":22}]}
        with self.assertRaises(ValueError):
            worker.asset_lock(folder,lock,"b"*40)

    def test_prompt_embedding_cached_once_per_ref_and_xvector_only(self):
        class FakeModel:
            def __init__(self):
                self.args=[]
            def create_voice_clone_prompt(self,**kwargs):
                self.args.append(kwargs)
                return ["fake-prompt"]
        m=FakeModel()
        backend=worker.QwenBulkClone(self.root)
        backend._model=lambda:m
        backend._reference=lambda ref,root:([0.2]*32000,16000)
        ref=self.references[0]
        self.assertEqual(backend._prompt(ref,self.root),["fake-prompt"])
        self.assertEqual(backend._prompt(ref,self.root),["fake-prompt"])
        self.assertEqual(len(m.args),1)
        self.assertTrue(m.args[0]["x_vector_only_mode"])
        self.assertIsNone(m.args[0]["ref_text"])
        verified=dict(ref,reference_text_status="verified",
                      reference_text="今天的天气不错")
        backend._prompt(verified,self.root)
        self.assertEqual(len(m.args),2)
        self.assertFalse(m.args[1]["x_vector_only_mode"])

    def test_blinded_asr_three_engines_and_machine_only_screen(self):
        self.gen(1)
        seen=[]
        def fake(cmd,**kw):
            self.assertFalse(any(wake in " ".join(cmd) for wake in
                ("你好小窝","小窝小窝","你好小屋")))
            seen.append(cmd)
            return self.simulate_asr(cmd,**kw)
        with patch.dict(os.environ,{"HF_HUB_OFFLINE":"1",
                                  "TRANSFORMERS_OFFLINE":"1"}):
            first=a.dispatch(self.plan,self.tts,self.asr,self.adapters,
                              max_new_calls=3,execute=True,executor=fake)
            second=a.dispatch(self.plan,self.tts,self.asr,self.adapters,
                               max_new_calls=0,execute=True,executor=fake)
        self.assertEqual(first["actual_calls"],3)
        self.assertEqual(second["actual_calls"],0)
        self.assertEqual(len(seen),3)
        screened=a.screen_results(self.plan,self.tts,self.asr)
        self.assertFalse(screened["training_admitted"])
        self.assertFalse(screened["human_gold"])
        candidates=[x for x in screened["rows"] if x["status"].startswith("MACHINE_")]
        self.assertEqual(len(candidates),1)

    def simulate_asr(self,cmd,**kwargs):
        native=Path(cmd[-1])
        cid=native.parent.parent.name
        engine=native.parent.name
        cell=next(x for x in self.plan["cells"] if x["cell_id"]==cid)
        with wave.open(str(Path(cmd[-3])),"rb") as w:
            frames=w.getnframes()
        adapter=next(x for x in self.adapters["adapters"] if x["engine"]==engine)
        result={"audio_id":cid,"engine":engine,"model_id":adapter["model_id"],
                "model_revision":adapter["model_revision"],
                "decoder_completed":True,"processed_start_frame":0,
                "processed_end_frame":frames,"actual_text":cell["prompt"]}
        native.write_bytes(p.canonical(result))
        return SimpleNamespace(returncode=0)

    def test_asr_failed_attempt_consumed_not_retried(self):
        self.gen(1)
        calls=[]
        def fail(cmd,**kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=2)
        with patch.dict(os.environ,{"HF_HUB_OFFLINE":"1",
                                  "TRANSFORMERS_OFFLINE":"1"}):
            r=a.dispatch(self.plan,self.tts,self.asr,self.adapters,
                         max_new_calls=1,execute=True,executor=fail)
            nxt=a.dispatch(self.plan,self.tts,self.asr,self.adapters,
                           max_new_calls=0,execute=True,executor=fail)
        self.assertEqual(r["actual_calls"],1)
        self.assertEqual(nxt["actual_calls"],0)
        self.assertEqual(len(calls),1)


if __name__=="__main__":
    unittest.main()
