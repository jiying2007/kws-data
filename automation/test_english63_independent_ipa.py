"""Source-only regressions for blinded original-63-WAV independent IPA runner."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english63_independent_ipa as ipa
import english_acoustic_contrast as acoustic
import multilingual_campaign as market


class Tokenizer:
    all_special_ids = [0,1]
    word_delimiter_token_id = 9
    def convert_ids_to_tokens(self,ids):
        dictionary={2:"ˈv",3:"oʊ",4:"z",5:"ə",6:"n",7:"m"}
        return [dictionary.get(x,"") for x in ids]


def invented_rows():
    rows=[]
    for voice, role in acoustic.source_tts.VOICES:
        for i, prompt in enumerate(acoustic.PROMPTS):
            rows.append({
                "cell_id":"invented-"+voice+"-"+str(i),
                "voice_id":voice,"role":role,"text_requested":prompt,
                "stressless_phones":["h","e",voice,prompt],
                "training_admitted":False,"human_gold":False})
    return rows


class EnglishIpaResearchTests(unittest.TestCase):
    def test_frozen_source_and_one_model_63_budget(self):
        p=ipa.policy()
        self.assertEqual(p["schema"],ipa.POLICY)
        self.assertEqual(p["max_new_phoneme_forwards"],63)
        self.assertEqual(p["source_pcm_count"],63)
        self.assertEqual(p["source_three_asr_observations"],189)
        self.assertEqual(p["model_revision"],ipa.MODEL_REV)
        self.assertEqual(p["model_safetensors_sha256"],ipa.WEIGHTS_SHA256)
        self.assertEqual(len(p["model_files"]),7)
        self.assertFalse(p["training_admitted"])
        self.assertFalse(p["human_gold"])
        self.assertTrue(p["no_scheduled_tasks"])
        self.assertEqual(ipa.main([]),0)

    def test_one_shot_main_protection(self):
        good={"GITHUB_REPOSITORY":"jiying2007/kws-data",
              "GITHUB_EVENT_NAME":"push","GITHUB_REF":"refs/heads/main",
              "GITHUB_RUN_ATTEMPT":"1"}
        with mock.patch.dict(os.environ,good,clear=True):
            ipa.protected_main()
        for key,bad in (("GITHUB_REPOSITORY","wrong/repo"),
                        ("GITHUB_EVENT_NAME","pull_request"),
                        ("GITHUB_REF","refs/heads/mirror"),
                        ("GITHUB_RUN_ATTEMPT","2")):
            with mock.patch.dict(os.environ,{**good,key:bad},clear=True):
                with self.assertRaisesRegex(ValueError,"IPA_PROTECTED_MAIN_ONCE_REQUIRED"):
                    ipa.protected_main()

    def test_trigger_must_match_entire_frozen_claim(self):
        base=ipa.policy()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"config.json"
            with mock.patch.object(ipa,"TRIGGER",path):
                path.write_bytes(market.canonical(base)+b"\n")
                self.assertEqual(ipa.policy(),base)
                for key,value in (("max_new_phoneme_forwards",64),
                                  ("training_admitted",True),
                                  ("model_revision","f"*40),
                                  ("source_review_json_sha256","0"*64),
                                  ("new_tts_calls",1)):
                    altered=copy.deepcopy(base)
                    altered[key]=value
                    path.write_bytes(market.canonical(altered)+b"\n")
                    with self.assertRaisesRegex(ValueError,"IPA_TRIGGER_DRIFT"):
                        ipa.policy()

    def test_ctc_collapse_stress_token_and_blank(self):
        raw,phones=ipa.phone_tokens(
            [0,2,2,0,3,3,0,4,4,0,9,0,5,5,0,6,6,0],Tokenizer())
        self.assertEqual(raw,["ˈv","oʊ","z","ə","n"])
        self.assertEqual(phones,["v","oʊ","z","ə","n"])
        self.assertNotIn("|",phones)
        with self.assertRaisesRegex(ValueError,"IPA_PREDICTED_IDS_INVALID"):
            ipa.phone_tokens([1],Tokenizer())
        with self.assertRaisesRegex(ValueError,"IPA_TOKEN_SHAPE"):
            ipa.phone_tokens([0,77,0,77,0],Tokenizer())

    def test_same_and_nearphone_edit_is_descriptive_only(self):
        self.assertEqual(ipa.edit_distance(["h","ɛ","l"],["h","ɛ","l"]),0)
        self.assertEqual(ipa.edit_distance(["v","oʊ","z"],["w","oʊ","z"]),1)
        self.assertEqual(ipa.edit_distance(["v","oʊ","z"],["oʊ"]),2)
        rows=invented_rows()
        summary=ipa.compare_rows(rows)
        self.assertEqual(len(summary["by_voice"]),7)
        self.assertEqual(set(summary["by_contrast"]),
                         {name for name,_,_,_ in ipa.CONTRASTS})
        for info in summary["by_voice"]:
            self.assertFalse(info["training_admitted"])
            for c in info["contrasts"].values():
                self.assertFalse(c["training_admitted"])
                self.assertFalse(c["trusted_negative"])
                self.assertTrue(c["diagnostic_only"])
        self.assertEqual(summary["independent_human_count"],0)

    def test_duplicate_and_role_leak_fail_closed(self):
        data=invented_rows()
        dup=copy.deepcopy(data)
        dup[1]["cell_id"]=dup[0]["cell_id"]
        with self.assertRaisesRegex(ValueError,"IPA_ROW_ROSTER"):
            ipa.compare_rows(dup)
        leak=copy.deepcopy(data)
        leak[0]["role"]="development-only"
        with self.assertRaisesRegex(ValueError,"IPA_PROMPT_OR_SPLIT_LEAK"):
            ipa.compare_rows(leak)

    def test_weight_hash_not_a_self_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ipa.MODEL_FILES:
                (root/name).write_bytes(b"not authorized weights")
            with self.assertRaisesRegex(ValueError,"IPA_PINNED_WEIGHTS_MISMATCH"):
                ipa.manifest(root)
        self.assertEqual(ipa.WEIGHTS_SHA256,
           "9bc8cd20e1348fc384ba4b9cd60a5c761dbb3f0b7071d1bdc33331ed1b358f29")

    def test_read_only_exact_isolated_launcher(self):
        script=Path(ipa.__file__)
        proc=subprocess.run(
            [sys.executable,"-I","-S","-B",str(script)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            check=False,timeout=20)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["executed"],0)
        fail=subprocess.run(
            [sys.executable,"-I","-S","-B",str(script),"--execute"],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            check=False,timeout=20)
        self.assertEqual(fail.returncode,2,fail.stderr)
        self.assertIn("IPA_EXPLICIT_SOURCES_AND_OUTPUT_REQUIRED",fail.stderr)


if __name__=="__main__":
    unittest.main()
