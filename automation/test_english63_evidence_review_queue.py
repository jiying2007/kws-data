"""Invented metadata only; no raw WAVs or pretrained weights."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import english63_evidence_review_queue as queue


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def invented_three_sources():
    voices = sorted(queue.TRAIN_VOICES | queue.DEV_VOICES)
    asr_rows, dtw_rows, ipa_rows = [], [], []
    dtw_voice, ipa_voice = [], []
    for voice in voices:
        r = "train-candidate" if voice in queue.TRAIN_VOICES else "development-only"
        comparisons = {contrast:{
            "nearphone_to_same_variant_ratio":0.95,
            "nearphone_not_farther_than_same":True,
            } for _,_,_,names in queue.WAKES for contrast in names}
        ipa_contrasts = {contrast:{
            "same_wake_variant_phone_edit":2,
            "nearest_nearphone_phone_edit":1,
            "nearphone_not_farther_than_wake_variant":True,
            } for _,_,_,names in queue.WAKES for contrast in names}
        dtw_voice.append({"voice_id":voice,"comparisons":comparisons})
        ipa_voice.append({"voice_id":voice,"contrasts":ipa_contrasts})
        for text,intent in queue.REQUESTS.items():
            ident="en-"+digest(voice+":"+text)[:24]
            wav=digest("wav:"+voice+":"+text)
            fam="qwen3-preset-"+voice
            common={"cell_id":ident,"voice_id":voice,"source_family_id":fam,
                    "role":r,"wav_sha256":wav,
                    "human_gold":False,"training_admitted":False}
            asr_rows.append({
                **common,"requested_text":text,"requested_intent":intent,
                "machine_agreement":"THREE_DISAGREE",
                "phonetic_label_authenticated":False,
                "trusted_negative":False,
                "asr":{p:{"text":p+" "+text,"normalized":p+" "+text,
                           "native_sha256":digest(p+ident)}
                       for p in queue.PROVIDERS},
            })
            dtw_rows.append({
                **common,"pcm_sha256":digest("pcm:"+ident),
                "text_requested":text,"requested_intent":intent,
            })
            ipa_rows.append({
                **common,"text_requested":text,"requested_intent":intent,
                "phonetic_label_authenticated":False,
                "trusted_negative":False,
                "stressless_phones":["h","eɪ",voice],
                "raw_ipa_tokens":["h","eɪ",voice],
            })
    a={"schema":"english-63-three-asr-source-linked-review-v1",
       "source_tts_run_id":43242278962,
       "human_gold":False,"training_admitted":False,
       "counts":{"three_way_completed_calls":189},
       "rows":asr_rows}
    d={"schema":"english-63-paired-logmel-dtw-acoustic-contrast-v1",
       "tts_run_id":43242278962,"sample_count":63,
       "human_gold":False,"training_admitted":False,
       "sample_receipts":dtw_rows,"by_voice":dtw_voice}
    i={"schema":"english63-independent-ipa-ctc-machine-review-v1",
       "new_real_phoneme_forwards":63,
       "reused_prior_three_asr_calls":189,
       "human_gold":False,"training_admitted":False,
       "rows":ipa_rows,"by_voice":ipa_voice}
    return a,d,i


class ThreeEvidenceQueueTests(unittest.TestCase):
    def test_exact_roster_split_and_machine_only(self):
        a,d,i=invented_three_sources()
        out=queue.join(a,d,i)
        self.assertEqual(out["status"],"MACHINE_EVIDENCE_JOINED_REVIEW_NOT_GOLD")
        self.assertEqual(out["audio_rows_reused"],63)
        self.assertEqual(out["review_wake_pairs"],14)
        self.assertEqual(len(out["rows"]),63)
        self.assertEqual(len(out["wake_pairs"]),14)
        self.assertEqual(sum(p["role"]=="train-candidate"
                             for p in out["wake_pairs"]),10)
        self.assertEqual(sum(p["role"]=="development-only"
                             for p in out["wake_pairs"]),4)
        self.assertFalse(out["human_gold"])
        self.assertFalse(out["training_admitted"])
        self.assertEqual(out["new_model_calls"],0)
        self.assertTrue(all(r["training_admitted"] is False and
                            r["phonetic_label_authenticated"] is False
                            for r in out["rows"]))
        self.assertEqual([p["diagnostic_review_order"]
                          for p in out["wake_pairs"]], list(range(1,15)))
        self.assertEqual(set(out["source_payload_sha256"]),
                         {"asr","dtw","ipa"})

    def test_corrupted_split_and_missing_source_rejected(self):
        for change,reason in [
            (lambda a,d,i: a["rows"][0].update(role="development-only"),
             "ENGLISH63_SOURCE_ROLE_HASH_OR_GOLD_DRIFT"),
            (lambda a,d,i: d["sample_receipts"][0].update(
                pcm_sha256=d["sample_receipts"][1]["pcm_sha256"]),
             "ENGLISH63_DUPLICATE_PCM_WAV_OR_PROMPT"),
            (lambda a,d,i: i["rows"].pop(),
             "ENGLISH63_ROW_SHAPE_ipa"),
            (lambda a,d,i: a["rows"][0]["asr"].pop("qwen"),
             "ENGLISH63_ASR_OR_IPA_NOT_COMPLETE"),
            (lambda a,d,i: a["rows"][0].update(training_admitted=True),
             "ENGLISH63_SOURCE_ROLE_HASH_OR_GOLD_DRIFT"),
            (lambda a,d,i: a["rows"][0].update(requested_intent="near_miss"),
             "ENGLISH63_SOURCE_ROLE_HASH_OR_GOLD_DRIFT"),
        ]:
            a,d,i=invented_three_sources()
            change(a,d,i)
            with self.assertRaisesRegex(ValueError,reason):
                queue.join(a,d,i)

    def test_word_swap_same_voice_and_voice_roster_rejected(self):
        a,d,i=invented_three_sources()
        d["sample_receipts"][0]["text_requested"]="Hey Wosen"
        with self.assertRaisesRegex(ValueError,"ENGLISH63_SOURCE_ROLE_HASH_OR_GOLD_DRIFT"):
            queue.join(a,d,i)
        a,d,i=invented_three_sources()
        a["rows"][0]["voice_id"]="new_voice"
        with self.assertRaisesRegex(ValueError,"ENGLISH63_SOURCE_ROLE_HASH_OR_GOLD_DRIFT"):
            queue.join(a,d,i)

    def test_archive_integrity_drift_rejected_before_unpack(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"invented.zip"
            p.write_bytes(b"not-the-pinned-artifact")
            with self.assertRaisesRegex(ValueError,"ENGLISH63_SOURCE_ZIP_DRIFT"):
                queue.load_original_archive(p,"asr")
            with self.assertRaisesRegex(ValueError,"ENGLISH63_ARCHIVE_INPUT_INVALID"):
                queue.load_original_archive(p,"not-a-source")

    def test_exact_isolated_cli_no_training_side_effects(self):
        path=Path(queue.__file__)
        for opts in ([],["-O"],["-OO"]):
            r=subprocess.run([sys.executable,"-I","-S","-B",*opts,str(path)],
                text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertFalse(json.loads(r.stdout)["training_admitted"])
        with tempfile.TemporaryDirectory() as temp:
            r=subprocess.run([sys.executable,"-I","-S","-B",str(path),
                  "--asr-zip",str(Path(temp)/"fake.zip")],
                text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
            self.assertEqual(r.returncode,2)
            self.assertIn("ENGLISH63_EXPLICIT_CREATE_ONLY_INPUTS_REQUIRED",r.stderr)


if __name__=="__main__":
    unittest.main()
