"""Source-only proof for the post-success 63-WAV ASR one-off activation."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_seven_voice_asr_once as asr
import multilingual_campaign as market
from test_english_seven_voice_asr_once import invented_source


class PostSuccessEnglishAsrTests(unittest.TestCase):
    def test_complete_source_and_no_extra_tts_on_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan, source=invented_source(root)
            loaded,files=asr.source_from_artifact(root)
            self.assertEqual(len(files),63)
            self.assertEqual(loaded,plan)
            self.assertEqual(len({cell["cell_id"] for cell,_,_ in files}),63)
            self.assertEqual({cell["role"] for cell,_,_ in files},
                             {"train-candidate","development-only"})
            self.assertFalse(plan["training_admitted"])
            self.assertFalse(plan["human_gold"])

    def test_push_claim_is_exact_and_refuses_unreviewed_hosts(self):
        known={
            "GITHUB_REPOSITORY":"jiying2007/kws-data",
            "GITHUB_EVENT_NAME":"push",
            "GITHUB_REF":"refs/heads/main",
            "GITHUB_RUN_ATTEMPT":"1",
        }
        with mock.patch.dict(os.environ,known,clear=False):
            result=asr.validate_postsuccess_push()
            self.assertEqual(result["source_tts_run_id"],43242278962)
            self.assertEqual(result["source_tts_artifact_id"],13954744587)
            self.assertFalse(result["training_admitted"])
            self.assertTrue(result["no_scheduled_tasks"])
            self.assertEqual(result["original_10_regenerated"],0)
            for bad_env in [
                {"GITHUB_EVENT_NAME":"pull_request"},
                {"GITHUB_REF":"refs/heads/feature/one"},
                {"GITHUB_RUN_ATTEMPT":"2"},
                {"GITHUB_REPOSITORY":"someone/other"},
            ]:
                with mock.patch.dict(os.environ,bad_env,clear=False):
                    with self.assertRaisesRegex(ValueError,
                        "SEVEN_ASR_POSTSUCCESS_PROTECTED_MAIN_ONLY"):
                        asr.validate_postsuccess_push()

    def test_mutated_trigger_would_fail_even_on_protected_main(self):
        value=market.read(asr.POSTSUCCESS_TRIGGER)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"claim.json"
            with mock.patch.object(asr,"POSTSUCCESS_TRIGGER",path):
                with mock.patch.dict(os.environ,{
                    "GITHUB_REPOSITORY":"jiying2007/kws-data",
                    "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
                    "GITHUB_RUN_ATTEMPT":"1"},clear=False):
                    path.write_bytes(market.canonical(value)+b"\n")
                    self.assertEqual(asr.validate_postsuccess_push(),value)
                    changed=copy.deepcopy(value)
                    changed["source_tts_artifact_sha256"]="f"*64
                    path.write_bytes(market.canonical(changed)+b"\n")
                    with self.assertRaisesRegex(ValueError,
                        "SEVEN_ASR_POSTSUCCESS_CLAIM_CHANGED"):
                        asr.validate_postsuccess_push()
                    changed=copy.deepcopy(value)
                    changed["max_new_asr_calls_per_provider"]=64
                    path.write_bytes(market.canonical(changed)+b"\n")
                    with self.assertRaisesRegex(ValueError,
                        "SEVEN_ASR_POSTSUCCESS_CLAIM_CHANGED"):
                        asr.validate_postsuccess_push()


if __name__=="__main__":
    unittest.main()
