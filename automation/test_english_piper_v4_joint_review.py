"""Regression checks for evidence mismatch and accidental label promotion."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_piper_v4_joint_review as review


def fixture():
    source, asr_rows, phone_rows = {}, [], []
    natives = {p: {} for p in review.PROVIDERS}
    for i, (phrase, voice) in enumerate((p, v) for p in review.TARGETS for v in review.SPEAKERS):
        cid = "cell-%02d" % i
        target = review.TARGETS[phrase]
        wav_hash = "%064x" % (i + 1)
        source[cid] = {"cell_id": cid, "phrase": phrase, "speaker": voice,
            "ipa": target["ipa"], "wav_sha256": wav_hash, "frames": 16000}
        asr_rows.append({"cell_id": cid, "phrase": phrase, "speaker": voice,
            "ipa": target["ipa"], "transcripts": {p: phrase for p in review.PROVIDERS},
            "human_gold": False, "training_admitted": False})
        phone_rows.append({"cell_id": cid, "target_phrase": phrase, "target_ipa": target["ipa"],
            "source_wav_sha256": wav_hash, "source_frames": 16000,
            "predicted_stressless_phone_tokens": target["phones"],
            "predicted_raw_ipa_tokens": target["phones"], "human_gold": False,
            "training_admitted": False, "pronunciation_verified": False,
            "machine_observation_only": True, "target_is_not_passed_to_model": True})
        for provider in review.PROVIDERS:
            natives[provider][provider + "/cells/" + cid + "/native.json"] = json.dumps({
                "decoder_completed": True, "audio_wav_sha256": wav_hash,
                "processed_start_frame": 0, "processed_end_frame": 16000,
                "actual_text": phrase}).encode()
    shared = {"human_gold": False, "training_admitted": False,
              "source_run_id": review.SOURCE_RUN, "source_artifact_id": review.SOURCE_ARTIFACT}
    asr = {**shared, "schema": "english-ipa-piper-v4-three-blind-asr-v1", "asr_calls": 48,
           "providers": list(review.PROVIDERS), "source_cell_closure_complete": True, "cells": asr_rows}
    phones = {**shared, "schema": "english-ipa-piper-v4-independent-ctc-observation-v1",
              "status": "COMPLETE", "attempted_forwards": 16, "completed_forwards": 16,
              "source_artifact_sha256": review.ARCHIVES["source"][2], "observations": phone_rows}
    return source, {"status": "INCOMPLETE_NO_RETRY"}, asr, phones, natives


class EvidenceJoinTests(unittest.TestCase):
    def test_perfect_machine_agreement_cannot_admit_or_invent_split(self):
        result = review.join(*fixture())
        self.assertEqual(len(result["rows"]), 16)
        self.assertTrue(all(r["phone_segments_match_ignoring_stress_length"] for r in result["rows"]))
        for row in result["rows"]:
            self.assertIsNone(row["source_split"])
            for flag in ("formal_usable_train", "formal_usable_dev", "human_gold", "trusted_negative",
                         "pronunciation_verified", "formal_training_admitted", "word_end_latency_evaluable"):
                self.assertIs(row[flag], False)

    def test_phone_from_wrong_pcm_is_rejected(self):
        values = fixture()
        values[3]["observations"][0]["source_wav_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "JOIN_IDENTITY_MISMATCH"):
            review.join(*values)

    def test_native_transcript_from_other_cell_is_rejected(self):
        values = fixture()
        native = values[4]["qwen"]
        names = list(native)
        native[names[0]] = native[names[1]]
        with self.assertRaisesRegex(ValueError, "NATIVE_ASR_BINDING"):
            review.join(*values)

    def test_same_count_with_duplicate_cell_is_rejected(self):
        values = fixture()
        values[2]["cells"][-1] = copy.deepcopy(values[2]["cells"][0])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_CELL"):
            review.join(*values)

    def test_source_run_substitution_is_rejected(self):
        values = fixture()
        values[2]["source_run_id"] += 1
        with self.assertRaisesRegex(ValueError, "OBSERVATION_SOURCE_ID"):
            review.join(*values)

    def test_unknown_spelling_remains_uncertain_not_negative(self):
        values = fixture()
        cid = values[2]["cells"][0]["cell_id"]
        for provider in review.PROVIDERS:
            values[2]["cells"][0]["transcripts"][provider] = "a mural"
            name = provider + "/cells/" + cid + "/native.json"
            obj = json.loads(values[4][provider][name])
            obj["actual_text"] = "a mural"
            values[4][provider][name] = json.dumps(obj).encode()
        row = review.join(*values)["rows"][0]
        self.assertEqual(row["asr_exact_spelling_providers"], [])
        self.assertEqual(row["label_status"], "UNCERTAIN_REVIEW_REQUIRED")
        self.assertIs(row["trusted_negative"], False)


if __name__ == "__main__":
    unittest.main()
