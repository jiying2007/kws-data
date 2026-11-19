"""Source-only prospective independent Hey Mero Piper24 contract tests."""
from __future__ import annotations
from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location(
    "mero24_test",HERE/"english_v5_mero_fresh24_source_once.py")
v=importlib.util.module_from_spec(s)
s.loader.exec_module(v)


def fake_map():
    return {"num_speakers":904,"speaker_id_map":
            {f"p{10000+i}":i for i in range(904)}}


class Mero24SourceContract(unittest.TestCase):
    def test_exact_predeclared_one_keyword_12_4_4_4_groups(self):
        p=v.plan();c=v.source_check()
        self.assertEqual(c["only_english_keyword"],"Hey Mero")
        self.assertEqual(c["new_speaker_indices"],[448,471])
        self.assertEqual(c["new_speech_wav_budget"],96)
        self.assertEqual(c["source_roles"],
                         {"train":12,"development":4,"calibration":4,
                          "locked-test":4})
        self.assertEqual(Counter(g["role"] for g in p["source_groups"]),
                         c["source_roles"])
        self.assertEqual([r["speaker_index"] for r in p["source_groups"]],
                         list(range(448,472)))
        self.assertEqual([r["requested_class"] for r in p["cells"]],
                         [2,2,0,None])
        self.assertEqual([r["length_scale"] for r in p["cells"]],
                         [0.92,1.08,1,1])
        self.assertFalse(c["nearphone_is_eligible_ordinary"])
        self.assertFalse(c["independent_human_gold"])
        self.assertEqual(c["new_optimizer_updates"],0)

    def test_fresh_904_map_has_24_new_unrepeated_real_ids(self):
        voices=v.native_speakers(fake_map())
        self.assertEqual(len(voices),24)
        self.assertEqual(voices[448],"p10448")
        rows=v.planned(voices)
        self.assertEqual(len(rows),96)
        self.assertEqual(len({x["source_group_id"] for x in rows}),24)
        self.assertEqual(Counter(x["role"] for x in rows),
                         {"train":48,"development":16,
                          "calibration":16,"locked-test":16})
        self.assertTrue(all(x["observed_class"] is None
                            and x["training_allowed"] is False
                            and x["human_gold"] is False for x in rows))
        self.assertTrue(all(x["requested_class"] is None
                            for x in rows if
                            x["cell_id"]=="nearphone-meero-challenge"))

    def test_never_reuse_old_human_original_speaker_ids(self):
        info=fake_map()
        del info["speaker_id_map"]["p10448"]
        info["speaker_id_map"]["p84"]=448  # old Libri speaker
        with self.assertRaises(ValueError):
            v.native_speakers(info)
        info=fake_map()
        info["speaker_id_map"]["p10449"]=448
        with self.assertRaises(ValueError):
            v.native_speakers(info)

    def test_original_config_must_have_exact_size_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"spoof.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                v.preflight_config(path)

    def test_first_public_main_successor_only_no_replay(self):
        env={"GITHUB_ACTIONS":"true","GITHUB_REPOSITORY":v.REPO,
             "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
             "GITHUB_RUN_ATTEMPT":"1","GITHUB_RUN_ID":"44488800111",
             "GITHUB_SHA":"a"*40}
        event={"before":v.MAIN_BEFORE,"after":"a"*40,
               "ref":"refs/heads/main","forced":False,"deleted":False,
               "repository":{"full_name":v.REPO,"private":False}}
        self.assertEqual(v.guard(env,event,v.MAIN_BEFORE)["before"],
                         v.MAIN_BEFORE)
        for e,x,parent in (
            ({**env,"GITHUB_RUN_ATTEMPT":"2"},event,v.MAIN_BEFORE),
            ({**env,"GITHUB_EVENT_NAME":"pull_request"},event,v.MAIN_BEFORE),
            (env,{**event,"before":"b"*40},v.MAIN_BEFORE),
            (env,{**event,"repository":{"full_name":v.REPO,
                                        "private":True}},v.MAIN_BEFORE),
            (env,event,"c"*40),
        ):
            with self.subTest(e=e,x=x,parent=parent):
                with self.assertRaises(ValueError):
                    v.guard(e,x,parent)

    def test_source_only_no_model_inference_or_pseudo_gold(self):
        code=(HERE/"english_v5_mero_fresh24_source_once.py").read_text()
        self.assertIn("native_speakers(config)",code)
        self.assertIn("PiperVoice.load",code)
        self.assertIn("previous.wav_write",code)
        self.assertIn("previous.c.inspect_wav",code)
        self.assertIn("source_plan_sha256",code)
        self.assertNotIn("torch.optim",code)
        self.assertNotIn("ASR.decode",code)
        self.assertNotIn("Hey Vosen",code)
        self.assertNotIn("quality_promotion_allowed=True",code)


if __name__=="__main__":
    unittest.main()
