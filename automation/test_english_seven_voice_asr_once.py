"""Invented-file only checks for event-driven 63-cell independent English ASR."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_seven_voice_asr_once as asr
import english_seven_voice_tts_once as seven
import english_qwen_capture as english
import multilingual_campaign as market
from test_multilingual_campaign import FakeEnglishBackend, fixture_wave_writer


def invented_source(parent: Path) -> tuple[dict, Path]:
    root = parent / asr.ARTIFACT_ROOT
    root.mkdir(mode=0o700)
    fake_lock = {"schema": english.MODEL_SCHEMA,
                 "models": [{"model_id":seven.pilot.MODEL,
                             "revision":seven.pilot.REV,
                             "relative_dir":"invented-source-only",
                             "files":[{"path":"invented.txt","bytes":3,
                                       "sha256":hashlib.sha256(b"abc").hexdigest()}]}]}
    (root / "model-lock.json").write_bytes(market.canonical(fake_lock) + b"\n")
    plan = seven.plan(hashlib.sha256((root / "model-lock.json").read_bytes()).hexdigest())
    (root / "source-plan.json").write_bytes(market.canonical(plan) + b"\n")
    fake = FakeEnglishBackend()
    for budget in seven.STAGES:
        english.capture(plan, root / "capture", fake,
                        max_calls=budget, writer=fixture_wave_writer)
    assert len(fake.calls) == 63
    rows=[]
    for cell in plan["cells"]:
        r=market.read(root / "capture" / "cells" / cell["cell_id"] / "completed.json")
        rows.append({"cell_id":cell["cell_id"],
                     "source_family_id":cell["source_family_id"],
                     "role":cell["role"],"intent":cell["intent"],
                     "status":"GENERATED_UNADMITTED",
                     "wav_sha256":r["audio"]["wav_sha256"],
                     "pcm_sha256":r["audio"]["pcm_sha256"],
                     "training_admitted":False})
    summary={
        "schema":seven.POLICY,"status":"COMPLETE",
        "planned_audio_count":63, "attempted_audio_count":63,
        "generated_audio_count":63,"previous_audio_regenerated":0,
        "asr_calls":0,"training_admitted":False,"human_gold":False,
        "source_rows":rows}
    (root / "summary.json").write_bytes(market.canonical(summary) + b"\n")
    return plan,root


class EventDrivenEnglishAsrTests(unittest.TestCase):
    def test_three_independent_pinned_providers_and_no_training_grant(self):
        self.assertEqual(set(asr.PROVIDERS), {"qwen","whisper","parakeet"})
        self.assertEqual(asr.MAX_CALLS, 63)
        self.assertEqual(asr.SOURCE_RUN_ID, 43242278962)
        self.assertEqual(asr.SOURCE_HEAD_SHA,
                         "bec1a1421d1db2de17afb9c6e3155343becdb980")
        self.assertEqual(len({id for id,_,_ in asr.PROVENANCE.values()}),3)
        self.assertEqual(len({rev for _,_,rev in asr.PROVENANCE.values()}),3)

    def test_source_identity_for_63_and_real_launcher_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            plan,root=invented_source(parent)
            self.assertEqual(asr.source_from_artifact(parent)[0],plan)
            self.assertEqual(len(asr.source_from_artifact(parent)[1]),63)
            run=subprocess.run(
                [sys.executable,"-B",str(Path(asr.__file__)),
                 "--provider","qwen","--artifact-parent",str(parent)],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads(run.stdout)
            self.assertEqual(report["new_asr_calls"],0)
            self.assertEqual(report["source_audio_count"],63)
            self.assertFalse(report["training_admitted"])

    def test_invented_three_provider_native_output_no_target_or_peer_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            plan,_=invented_source(parent)
            items=asr.source_from_artifact(parent)[1]
            first=set(x["cell_id"] for x,_,_ in items)
            called=[]
            def fake_decode(wav,folder):
                self.assertEqual(wav.name,"audio-16k.wav")
                self.assertFalse((folder / "peer-output.json").exists())
                called.append(str(wav))
                return "Hey, ambiguous pronunciation."
            for provider in ("qwen","whisper","parakeet"):
                output=parent/("result-"+provider)
                result=asr.capture(provider,parent,output,decoder=fake_decode)
                self.assertEqual(result["asr_attempts"],63)
                self.assertEqual(result["completed_asr"],63)
                self.assertEqual(result["status"],"COMPLETE")
                self.assertFalse(result["human_gold"])
                self.assertFalse(result["training_admitted"])
                self.assertFalse(result["three_asr_completed"])
                self.assertEqual(set(x["cell_id"] for x in result["rows"]),first)
                self.assertEqual(output.stat().st_mode & 0o777,0o700)
                with self.assertRaisesRegex(ValueError,"SEVEN_ASR_OUTPUT_CREATE_ONLY"):
                    asr.capture(provider,parent,output,decoder=fake_decode)
            self.assertEqual(len(called),189)
            self.assertEqual(len({x for x in called}),63)

    def test_source_changes_are_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            plan,root=invented_source(parent)
            wave_path=asr.source_from_artifact(parent)[1][0][1]
            wave_path.write_bytes(wave_path.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError,"CAPTURE_WAV_SHA_DRIFT"):
                asr.source_from_artifact(parent)
            self.assertFalse((parent/"no-asr").exists())

    def test_event_requires_one_exact_completed_source_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"event.json"
            event={"action":"completed","workflow_run":{
                "id":asr.SOURCE_RUN_ID,"name":asr.SOURCE_WORKFLOW,
                "head_sha":asr.SOURCE_HEAD_SHA,"head_branch":"main",
                "status":"completed","conclusion":"success",
                "event":"push","run_attempt":1}}
            path.write_bytes(market.canonical(event)+b"\n")
            self.assertEqual(asr.validate_trigger_event(path)["id"],
                             asr.SOURCE_RUN_ID)
            for key,bad in (("conclusion","failure"),
                            ("head_branch","untrusted"),
                            ("head_sha","a"*40),
                            ("run_attempt",2)):
                copy_event=copy.deepcopy(event)
                copy_event["workflow_run"][key]=bad
                path.write_bytes(market.canonical(copy_event)+b"\n")
                with self.assertRaisesRegex(ValueError,"SEVEN_ASR_SOURCE_WORKFLOW_EVENT_INVALID"):
                    asr.validate_trigger_event(path)


if __name__=="__main__":
    unittest.main()
