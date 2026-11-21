"""Source-only B10 five-native observer contract tests; no ASR/phone model loaded."""
from __future__ import annotations
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main,mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location(
    "b10_native_source_contract",HERE/"english_v6_b10_hardneg_five_native_once.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fake_manifest():
    choices={g["speaker_index"]:g["source_speaker_id"] for g in m.source_worker.plan()["train_groups"]}
    choices.update({i:f"p{20000+i}" for i in range(540,546)})
    rows=m.source_worker.planned(choices)
    for i,row in enumerate(rows):
        row.update({"observed_class":None,"training_allowed":False,
                    "human_gold":False,"pcm_sha256":f"{i+1:064x}",
                    "wav_sha256":f"{i+101:064x}","samples":8000})
    return {"schema":m.source_worker.SCHEMA,"run":m.original_identity(),
        "source_plan_sha256":m.source_worker.source_check()["plan_sha256"],
        "denominator":18,"completed":18,"new_tts_calls":18,
        "new_asr_calls":0,"new_phone_calls":0,
        "training_allowed":False,"human_gold":False,"rows":rows}

class B10NativeTests(TestCase):
    def test_frozen_source_and_provider_roster(self):
        x=m.source_check()
        self.assertEqual(x["status"],"B10_SOURCE_PINNED")
        self.assertEqual((x["source_rows"],x["review_rows"]),(18,18))
        self.assertEqual(x["source_roles"],{"train":6,"development":12})
        self.assertEqual(x["observer_roster"],["qwen","whisper","parakeet","xlsr","lv60"])
        self.assertEqual(x["max_new_native_forwards_per_observer"],18)
        self.assertEqual(x["new_tts_calls"],0)
        self.assertEqual(x["new_optimizer_updates"],0)
        self.assertEqual(x["kws_model_forwards"],0)
        self.assertFalse(x["training_allowed"])
        self.assertFalse(x["product_promotion_allowed"])

    def test_source_rows_preserve_exact_train_dev_roles(self):
        packet=fake_manifest()
        with TemporaryDirectory() as folder:
            root=Path(folder);src=root/"source";train=root/"train";screen=root/"screen"
            src.mkdir();train.mkdir();screen.mkdir()
            (src/"MANIFEST.json").write_text(json.dumps(packet))
            with mock.patch.object(m.source_worker,"readback",return_value={"status":"ok"}):
                rows=m.source_rows(src,train,screen)
        self.assertEqual(len(rows),18)
        self.assertEqual(Counter(r["role"] for r in rows),{"train":6,"development":12})
        self.assertEqual(Counter(r["cell_id"] for r in rows),
                         {"ordinary-time-hard":6,"mero-fast":6,"ordinary-time":6})
        bad=json.loads(json.dumps(packet))
        bad["rows"][0]["role"]="development"
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/"MANIFEST.json").write_text(json.dumps(bad))
            with mock.patch.object(m.source_worker,"readback",return_value={"status":"ok"}):
                with self.assertRaisesRegex(ValueError,"ONLY_SIX_TRAIN_HARDNEG"):
                    m.source_rows(root,root,root)

    def test_duplicate_pcm_and_prelabelled_rows_are_rejected(self):
        packet=fake_manifest()
        packet["rows"][1]["pcm_sha256"]=packet["rows"][0]["pcm_sha256"]
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/"MANIFEST.json").write_text(json.dumps(packet))
            with mock.patch.object(m.source_worker,"readback",return_value={"status":"ok"}):
                with self.assertRaisesRegex(ValueError,"NO_DUPLICATED"):
                    m.source_rows(root,root,root)
        packet=fake_manifest();packet["rows"][0]["observed_class"]=0
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/"MANIFEST.json").write_text(json.dumps(packet))
            with mock.patch.object(m.source_worker,"readback",return_value={"status":"ok"}):
                with self.assertRaisesRegex(ValueError,"NO_DUPLICATED"):
                    m.source_rows(root,root,root)

    def test_guard_fails_closed_outside_first_public_main(self):
        with mock.patch.dict(os.environ,{"GITHUB_ACTIONS":"false"},clear=True):
            with self.assertRaisesRegex(ValueError,"FIRST_PUBLIC_MAIN_ONLY"):
                m.guard()

if __name__=="__main__":
    main()
