"""Artificial WAV/fake backends only; exercises real producer and consumer code."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import wave

from test_bulk import Fixture, FakeClone
import bulk_pool as p
import bulk_autopilot as a
import campaign

PIPELINE = Path(os.environ.get("KWS_BULK_PIPELINE_ROOT",
                              str(Path(__file__).resolve().parents[1] / "pipeline")))

def unique_writer(folder, audio, rate, metrics):
    # Each artificial source has distinct PCM; these are not human voices.
    sample = int(hashlib.sha256(folder.name.encode()).hexdigest()[:4], 16) % 5000 + 1000
    pcm = struct.pack("<h", sample) * 8000
    path = folder / "audio-16k.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(pcm)
    raw = path.read_bytes()
    return {"wav_sha256": hashlib.sha256(raw).hexdigest(), "wav_bytes": len(raw),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "audio_frames": 8000,
            "sample_rate_hz": 16000, "native_sha256": "a" * 64, "native_bytes": 5}

class Interrupted(BaseException):
    pass

class BulkAutopilotTests(Fixture):
    def setUp(self):
        super().setUp()
        self.plan = p.plan(self.sources, self.root, repetitions=1,
                          batch_id="pilot-01", revision="b" * 40)
        self.work = self.root / "controller"
        self.consumer = a.PipelineConsumer(PIPELINE)
        self.asr_calls = []
        self.offline = patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.offline.start()
        self.addCleanup(self.offline.stop)

    def asr_execute(self, argv, **kwargs):
        self.asr_calls.append((str(argv), kwargs["timeout"]))
        return self.simulate_asr(argv, **kwargs)

    def go(self, tts=2, asr=6, **kwargs):
        defaults = dict(max_tts=tts, max_asr=asr, wall_seconds=60,
                        backend_identity={'model_id': self.plan['model_id'],
                            'model_revision': self.plan['model_revision'],
                            'asset_lock_sha256': 'a' * 64, 'provider_kind': 'synthetic-test-double'},
                        consumer=self.consumer, backend=self.backend,
                        executor=self.asr_execute, writer=unique_writer)
        defaults.update(kwargs)
        return a.execute(self.plan, self.sources, self.root, self.work,
                         self.adapters, **defaults)

    def checkpoint(self, result):
        return p.read_json(self.work / "checkpoints" / (result["checkpoint_sha256"] + ".json"))

    def test_success_two_references_real_pipeline_readback_resume_zero_calls(self):
        result = self.go(17, 51)
        self.assertEqual((result["new_tts_calls"], result["new_asr_calls"]), (17, 51))
        self.assertEqual(len(self.backend.prepared), 2)
        checkpoint = self.checkpoint(result)
        self.assertEqual(checkpoint["effect"]["total_generated_audio_verified"], 17)
        self.assertEqual(checkpoint["effect"]["policy"], "kws-data-bulk-effect-consumer-v1")
        self.assertEqual(checkpoint["feedback"]["policy"], "kws-effect-feedback-priorities-v1")
        self.assertFalse(result["training_admitted"])
        resumed = self.go(17, 51)
        self.assertEqual((resumed["new_tts_calls"], resumed["new_asr_calls"]), (0, 0))
        self.assertEqual(resumed["checkpoint_sha256"], result["checkpoint_sha256"])
        self.assertEqual(len(self.backend.calls), 17)
        self.assertEqual(len(self.asr_calls), 51)

    def test_missing_backend_not_run_and_no_started(self):
        result = self.go(backend=None)
        self.assertEqual(result["status"], "NOT_RUN_BACKEND_MISSING")
        self.assertEqual(result["tts_started"], 0)
        self.assertEqual(len(self.asr_calls), 0)

    def test_missing_asr_executor_does_not_claim_asr(self):
        result = self.go(executor=None)
        self.assertEqual(result["status"], "NOT_RUN_ASR_EXECUTOR_MISSING")
        self.assertEqual(result["tts_started"], 1)
        self.assertEqual(result["asr_started"], 0)

    def test_started_unknown_is_consumed_and_remaining_cells_continue(self):
        class Crash:
            def synthesize(self, *args):
                raise Interrupted("synthetic interruption after durable start")
        with self.assertRaises(Interrupted):
            self.go(backend=Crash())
        result = self.go()
        self.assertEqual((result["tts_started"], result["consumed_unknown"]), (2, 1))
        self.assertEqual(result["new_tts_calls"], 1)
        self.assertEqual(result["asr_started"], 3)
        self.assertNotIn(self.plan["cells"][0]["cell_id"], self.backend.calls)

    def test_asr_unknown_not_replayed(self):
        def crash(*args, **kwargs):
            raise Interrupted("synthetic ASR child uncertainty")
        with self.assertRaises(Interrupted):
            self.go(executor=crash)
        result = self.go()
        self.assertEqual(result["consumed_unknown"], 1)
        self.assertEqual(result["asr_started"], 6)
        self.assertEqual(result["new_asr_calls"], 5)
        checkpoint = self.checkpoint(result)
        self.assertGreater(checkpoint["screen"]["counts"].get("ASR_INCOMPLETE_HOLD", 0), 0)

    def test_failed_call_consumes_budget_without_retry(self):
        original = self.backend.synthesize
        def fail_one(cell, ref, root):
            if cell["cell_id"] == self.plan["cells"][0]["cell_id"]:
                raise ValueError("synthetic failure")
            return original(cell, ref, root)
        self.backend.synthesize = fail_one
        result = self.go()
        self.assertEqual(result["consumed_failed"], 1)
        self.assertEqual(result["tts_started"], 2)
        self.assertEqual(result["asr_started"], 3)
        self.assertEqual(self.go()["new_tts_calls"], 0)

    def test_deadline_stops_new_dispatch_and_resume_total_is_bounded(self):
        now = [0.0]
        original = self.backend.synthesize
        def advancing(cell, ref, root):
            now[0] = 61.0
            return original(cell, ref, root)
        self.backend.synthesize = advancing
        result = self.go(clock=lambda: now[0])
        self.assertEqual(result["status"], "DEADLINE_HOLD")
        self.assertEqual((result["tts_started"], result["asr_started"]), (1, 0))
        self.backend.synthesize = original
        result = self.go()
        self.assertEqual((result["tts_started"], result["asr_started"]), (2, 6))
        self.assertEqual(self.go()["new_tts_calls"], 0)

    def test_total_budget_cannot_be_raised_on_resume(self):
        self.go()
        with self.assertRaisesRegex(ValueError, "CONTROLLER_IDENTITY_DRIFT"):
            self.go(3, 9)
        self.assertEqual(len(self.backend.calls), 2)

    def test_resumed_deadline_checkpoint_matches_final_counters(self):
        now = [0.0]
        original = self.backend.synthesize
        def advancing(cell, ref, root):
            now[0] += 61.0
            return original(cell, ref, root)
        self.backend.synthesize = advancing
        first = self.go(clock=lambda: now[0])
        self.assertEqual(first['tts_started'], 1)
        second = self.go(clock=lambda: now[0])
        self.assertEqual(second['tts_started'], 2)
        checkpoint = self.checkpoint(second)
        self.assertEqual(checkpoint['counts']['tts_started'], second['tts_started'])
        self.assertEqual(checkpoint['counts']['asr_started'], second['asr_started'])
        self.assertEqual(checkpoint['effect']['total_generated_audio_verified'], 2)

    def test_backend_assets_changed_on_resume_are_rejected(self):
        self.go()
        changed = {'model_id': self.plan['model_id'],
                   'model_revision': self.plan['model_revision'],
                   'asset_lock_sha256': 'b' * 64, 'provider_kind': 'synthetic-test-double'}
        with self.assertRaisesRegex(ValueError, 'CONTROLLER_IDENTITY_DRIFT'):
            self.go(backend_identity=changed)
        self.assertEqual(len(self.backend.calls), 2)

    def test_changed_adapter_rejected_before_any_new_call(self):
        self.go()
        self.adapters["adapters"][0]["model_revision"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "CONTROLLER_IDENTITY_DRIFT"):
            self.go()
        self.assertEqual(len(self.backend.calls), 2)

    def test_plan_or_rights_drift_prevents_backend(self):
        self.sources["references"][0]["source_license"] = "changed"
        with self.assertRaises(ValueError):
            self.go()
        self.assertEqual(self.backend.calls, [])

    def test_orphan_directory_refuses_resume(self):
        self.go()
        folder = self.work / "tts/cells/foreign-cell"
        folder.mkdir()
        with self.assertRaisesRegex(ValueError, "FOREIGN_ATTEMPT_DIRECTORY"):
            self.go()

    def test_malformed_start_rejected_without_replay(self):
        self.go()
        path = self.work / "tts/cells" / self.plan["cells"][0]["cell_id"] / "started.json"
        value = p.read_json(path)
        value["attempts"] = True
        path.write_bytes(p.canonical(value))
        with self.assertRaisesRegex(ValueError, "START_RECEIPT_IDENTITY"):
            self.go()
        self.assertEqual(len(self.backend.calls), 2)

    def test_changed_native_bytes_rejected_by_real_screen(self):
        self.go()
        path = self.work / "asr/clips" / self.plan["cells"][0]["cell_id"] / "qwen06/native.json"
        value = p.read_json(path)
        value["actual_text"] = "different"
        path.write_bytes(p.canonical(value))
        with self.assertRaisesRegex(ValueError, "ASR_NATIVE_BYTES_DRIFT"):
            self.go()
        self.assertEqual(len(self.asr_calls), 6)

    def test_confusion_stops_following_calls(self):
        # A hard-negative intent among the first five cells.
        def confused(argv, **kwargs):
            result = self.asr_execute(argv, **kwargs)
            output = Path(argv[-1])
            value = p.read_json(output)
            cid = value["audio_id"]
            cell = next(x for x in self.plan["cells"] if x["cell_id"] == cid)
            if cell["intended_kind"] != "positive-intent":
                value["actual_text"] = "你好小窝"
                output.write_bytes(p.canonical(value))
            return result
        result = self.go(8, 24, executor=confused)
        self.assertEqual(result["status"], "CONFUSION_HOLD")
        self.assertLess(result["asr_started"], 24)
        calls = (len(self.backend.calls), len(self.asr_calls))
        resumed = self.go(8, 24, executor=confused)
        self.assertEqual(resumed['status'], 'CONFUSION_HOLD')
        self.assertEqual((resumed['new_tts_calls'], resumed['new_asr_calls']), (0, 0))
        self.assertEqual((len(self.backend.calls), len(self.asr_calls)), calls)

    def test_concurrent_controller_refused(self):
        self.work.mkdir()
        with a.exclusive(self.work):
            with self.assertRaisesRegex(ValueError, "CONTROLLER_ALREADY_RUNNING"):
                self.go()
        self.assertEqual(self.backend.calls, [])

    def test_invalid_numeric_budget_rejected(self):
        for value in (True, -1, 33):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.go(tts=value)
        for value in (True, 0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.go(wall_seconds=value)
        self.assertEqual(self.backend.calls, [])

    def test_pipeline_pin_drift_rejected_before_execution(self):
        other = self.root / "wrong-pipeline/tools"
        other.mkdir(parents=True)
        for file in (PIPELINE / "tools").glob("*.py"):
            (other / file.name).write_bytes(file.read_bytes())
        target = other / "corpus_identity.py"
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "PIPELINE_CONSUMER_SOURCE_DRIFT"):
            a.PipelineConsumer(other.parent)

if __name__ == "__main__":
    unittest.main()
