"""No-model proof for independently authorized Parakeet runtime recovery."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english_parakeet_runtimefix_once as fix
import english_parakeet_real_once as native
import multilingual_campaign as market
from test_english_qwen_asr_sourcefix_once import EnglishAsrSourcefixTests


class ParakeetRuntimefixTests(unittest.TestCase):
    def fake_prior(self, root: Path) -> Path:
        output=root/"prior-failure.json"
        content={
            "schema":native.POLICY, "status":"INCOMPLETE_NO_RETRY",
            "asr_attempts":0,"completed_asr":0,
            "runtime_receipt":None,"gguf_receipt":None,
            "error_type":"ValueError",
            "model_weight_sha256":native.MODEL_SHA256,
            "source_tts_run_id":43214159818,
            "training_admitted":False}
        output.write_bytes(market.canonical(content)+b"\n")
        return output

    def test_frozen_recovery_requires_exact_consumed_prior_failure(self):
        claim=fix.claim()
        self.assertEqual(claim["prior_failure_run_id"],43231581918)
        self.assertEqual(claim["prior_asr_attempts"],0)
        self.assertEqual(claim["max_new_asr_calls"],10)
        self.assertEqual(claim["source_tts_artifact_id"],13942568634)
        self.assertFalse(claim["training_admitted"])
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fake_prior(Path(tmp))
            self.assertEqual(fix.verify_prior_receipt(path)["asr_attempts"],0)
            altered=market.read(path)
            for key,value in (("asr_attempts",1),("gguf_receipt",{}),
                              ("runtime_receipt",{}),("status","COMPLETE")):
                with self.subTest(key=key):
                    new=copy.deepcopy(altered)
                    new[key]=value
                    path.write_bytes(market.canonical(new)+b"\n")
                    with self.assertRaisesRegex(ValueError,
                        "ORIGINAL_PARAKEET_ZERO_CALL_FAILURE_UNPROVEN"):
                        fix.verify_prior_receipt(path)
            self.assertEqual(fix.main(["--artifact-parent",str(Path(tmp)),
                                      "--prior-summary",str(path),
                                      "--verify-source"]),2)

    def test_real_cli_checks_ten_original_wavs_and_prior_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            EnglishAsrSourcefixTests().populate(root)
            prior=self.fake_prior(root)
            run=subprocess.run([
                sys.executable,"-B",str(Path(fix.__file__)),
                "--artifact-parent",str(root),"--prior-summary",str(prior),
                "--verify-source"],
                capture_output=True,text=True,timeout=30,check=False)
            self.assertEqual(run.returncode,0,run.stderr)
            receipt=json.loads(run.stdout)
            self.assertEqual(receipt["status"],
                "SOURCE_AND_PREVIOUS_FAILURE_VERIFIED")
            self.assertEqual(receipt["original_audio_count"],10)
            self.assertEqual(receipt["asr_calls"],0)
            self.assertFalse(receipt["training_admitted"])

    def test_execution_never_replays_old_claim_and_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            EnglishAsrSourcefixTests().populate(root)
            prior=self.fake_prior(root)
            output=root/"new-distinct-parakeet-results"
            with mock.patch.object(fix.native,"run",
                return_value={"status":"COMPLETE","asr_attempts":10,
                              "completed_asr":10,"training_admitted":False}) as worker:
                self.assertEqual(fix.main([
                    "--artifact-parent",str(root),"--prior-summary",str(prior),
                    "--output",str(output),"--execute"]),0)
                worker.assert_called_once_with(root,output)
                self.assertFalse(output.exists())
            self.assertEqual(fix.main([
                "--artifact-parent",str(root),"--prior-summary",str(prior)]),2)


if __name__=="__main__":
    unittest.main()
