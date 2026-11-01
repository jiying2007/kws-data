"""No-model regression for source-bound, NOT-admitted Qwen12 auditory observations."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import fresh_contract as fresh
import review_intake as intake

HERE = Path(__file__).resolve().parent


class ObservationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handoff = Path(os.environ["KWS_FRESH_BLIND_DIR"]).parent
        cls.maprows = intake.verify_handoff(cls.handoff)

    def specimen(self):
        return {
            "schema": intake.REVIEW_SCHEMA,
            "reviewer_entered_identity": "self-declared-fixture-not-verified",
            "reviewer_identity_verified": False,
            "reviewer_saw_transcripts_before_listening": False,
            "created_at_claimed_by_browser": "2026-11-01T05:00:00.000Z",
            "source": "qwen12-blind-main-41055537577",
            "independent_review_verified": False,
            "training_admitted": False,
            "qualification_allowed": False,
            "rows": [{
                "audio_id": m["audio_id"], "wav_sha256": m["file_sha256"],
                "heard_text": "", "verdict": "not_reviewed", "notes": "",
            } for m in self.maprows],
        }

    def validate(self, record):
        return intake.audit_observation(record, self.maprows)

    def test_actual_original_handoff_audio_and_roles(self):
        self.assertEqual(len(self.maprows), 12)
        self.assertEqual(sum(x["audio_frames"] for x in self.maprows), 481280)
        self.assertEqual({r["proposed_split"] for r in self.maprows}, {"train"})
        self.assertEqual(intake.verify_handoff(self.handoff), self.maprows)

    def test_blank_manual_export_is_not_training(self):
        result = self.validate(self.specimen())
        self.assertEqual(result["status"], "OBSERVATIONS_BOUND_TO_SOURCE_NOT_ADMITTED")
        self.assertEqual(result["quality_counts"]["not_reviewed"], 12)
        for k in ("training_admitted", "human_gold", "qualification_allowed",
                  "release_authorized", "blind_independence_authenticated",
                  "reviewer_identity_authenticated"):
            self.assertIs(result[k], False)
        self.assertEqual(result["rights_policy_review"], "PENDING")

    def test_homophones_are_not_automatically_corrected(self):
        observations = ("你好 小屋。", "小沃，小沃", "你好小吴", "你好小王", "小窝，小窝。")
        for word in observations:
            x = self.specimen()
            x["rows"][0]["verdict"] = "complete"
            x["rows"][0]["heard_text"] = word
            record = self.validate(x)["rows"][0]
            self.assertEqual(record["target_text_observed"], word == "小窝，小窝。")
            self.assertEqual(record["normalized_text"], intake.normalize_heard(word))
            self.assertIs(record["training_admitted"], False)

    def test_complete_requires_nonempty_heard_words(self):
        x = self.specimen()
        x["rows"][0]["verdict"] = "complete"
        with self.assertRaises(ValueError): self.validate(x)

    def test_blindness_checkbox_does_not_authenticate_independence(self):
        x = self.specimen()
        x["reviewer_saw_transcripts_before_listening"] = True
        result = self.validate(x)
        self.assertIs(result["reviewer_saw_transcripts_before_listening_claim"], True)
        self.assertIs(result["blind_independence_authenticated"], False)

    def test_no_audio_and_not_reviewed_may_not_have_text(self):
        for verdict in ("no_audio", "not_reviewed"):
            x = self.specimen()
            x["rows"][0].update(verdict=verdict,heard_text="你好小窝")
            with self.subTest(verdict=verdict), self.assertRaises(ValueError): self.validate(x)

    def test_reject_duplicate_missing_foreign_reordered_audio(self):
        for mutation in ("duplicate", "missing", "foreign", "wrong_sha", "extra", "reverse"):
            x = self.specimen()
            if mutation == "duplicate": x["rows"][1]["audio_id"] = x["rows"][0]["audio_id"]
            elif mutation == "missing": x["rows"].pop()
            elif mutation == "foreign": x["rows"][0]["audio_id"] = "clip-999999"
            elif mutation == "wrong_sha": x["rows"][0]["wav_sha256"] = "0"*64
            elif mutation == "extra": x["rows"][0]["expected_text"] = "你好小窝"
            else:
                # Order is not semantically meaningful: entries must match identity.
                x["rows"].reverse()
                self.assertEqual(len(self.validate(x)["rows"]), 12)
                continue
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.validate(x)

    def test_reject_permission_or_reviewer_authentication_forgery(self):
        for name in ("reviewer_identity_verified", "independent_review_verified",
                     "training_admitted", "qualification_allowed"):
            for value in (True, 1, "true"):
                x = self.specimen()
                x[name] = value
                with self.subTest(field=name,value=value), self.assertRaises(ValueError):
                    self.validate(x)

    def test_reject_wrong_bool_and_extra_top_level(self):
        for key, value in (("reviewer_saw_transcripts_before_listening", 0),
                           ("source", "other-archive"),
                           ("schema", "speech-like-audio-review-v2"),
                           ("reviewer_entered_identity", ""),
                           ("created_at_claimed_by_browser", "")):
            x = self.specimen()
            x[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(x)
        x=self.specimen()
        x["rights_policy_review"] = "APPROVED"
        with self.assertRaises(ValueError): self.validate(x)

    def test_reject_malformed_verdict_text_or_control(self):
        for key, value in (("verdict", "accepted"), ("verdict", True),
                           ("heard_text", "你"*97),
                           ("heard_text", "\x00hello"), ("notes", "x"*257)):
            x = self.specimen()
            x["rows"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(x)

    def test_strict_json_rejects_duplicate_key_and_nonfinite(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', b"\xff"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"review.json"
                path.write_bytes(raw if type(raw) is bytes else raw.encode("utf-8"))
                with self.subTest(raw=repr(raw)[:20]), self.assertRaises(ValueError):
                    intake.read_json(path, 1000)

    def test_cli_roundtrip_and_create_only_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/"review.json"
            dst=Path(tmp)/"audit.json"
            src.write_bytes(fresh.canonical(self.specimen()))
            self.assertEqual(intake.main(["--handoff-root",str(self.handoff),"--review",str(src),
                                          "--output",str(dst)]),0)
            value=json.loads(dst.read_bytes())
            self.assertEqual(value["observed_clip_count"],12)
            self.assertIs(value["training_admitted"],False)
            original=dst.read_bytes()
            self.assertEqual(intake.main(["--handoff-root",str(self.handoff),"--review",str(src),
                                          "--output",str(dst)]),2)
            self.assertEqual(dst.read_bytes(),original)

    def test_reject_source_map_modified_without_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"handoff"
            shutil.copytree(self.handoff,root)
            p=root/"control/source-map.json"
            data=p.read_bytes()
            p.write_bytes(data[:-1] + bytes([data[-1]^1]))
            with self.assertRaises(ValueError): intake.verify_handoff(root)

    def test_reject_audio_drift_even_with_matching_review_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"handoff"
            shutil.copytree(self.handoff,root)
            p=next((root/"blind/audio").iterdir())
            data=p.read_bytes()
            p.write_bytes(data[:-1] + bytes([data[-1]^1]))
            with self.assertRaises(ValueError): intake.verify_handoff(root)

    def test_reject_review_file_symlink_and_unknown_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/"original.json"
            alias=Path(tmp)/"alias.json"
            original.write_bytes(fresh.canonical(self.specimen()))
            alias.symlink_to(original)
            with self.assertRaises(ValueError): intake.read_json(alias,98304)
            with self.assertRaises(ValueError):
                intake.audit_observation(self.specimen(), self.maprows[:11])


if __name__ == "__main__":
    unittest.main()
