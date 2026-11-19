"""No-network, no-TTS, no-model regression for saved 168-WAV evidence join."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "english_saved_newtrain24_test",
    HERE / "english_newtrain24_evidence_readback.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fake_rows():
    sources = {idx: "p" + str(12000 + idx) for idx in range(424, 448)}
    rows = m.observer.source_worker.planned(sources)
    for idx, row in enumerate(rows):
        row.update(samples=16000, positive_duration_pass=True,
                   wav_sha256=f"{idx+1:064x}",
                   pcm_sha256=f"{idx+1000:064x}")
    return rows


class SavedFiveNativeReadback(unittest.TestCase):
    def test_pinned_original_model_sources_not_rerun(self):
        result = m.source_check()
        self.assertEqual(result["source_run"], 44431769603)
        self.assertEqual(result["five_observer_run"], 44433435970)
        self.assertEqual(result["speaker_roles"], {"train": 18, "development": 6})
        self.assertEqual(m.FROZEN_POLICY_BLOB,
                         "0a688325d0d36e83ef6623ac678511df24a92374")
        self.assertEqual(set(m.ARTIFACTS), set(m.observer.PROVIDERS))
        self.assertFalse(result["training_allowed"])
        self.assertEqual(result["model_forwards_in_this_stage"], 0)

    def test_reject_unexpected_branch_private_and_replay(self):
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": m.observer.REPO,
               "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
               "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "a" * 40,
               "GITHUB_RUN_ID": "444500001"}
        event = {"before": m.BEFORE, "after": "a" * 40,
                 "ref": "refs/heads/main", "forced": False,
                 "deleted": False,
                 "repository": {"full_name": m.observer.REPO,
                                "private": False}}
        self.assertEqual(m.guard(env, event, m.BEFORE)["observer_run"],
                         44433435970)
        for field, wrong in (("GITHUB_RUN_ATTEMPT", "2"),
                             ("GITHUB_EVENT_NAME", "pull_request"),
                             ("GITHUB_REF", "refs/heads/feature")):
            env2 = {**env, field: wrong}
            with self.assertRaises(ValueError):
                m.guard(env2, event, m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env, {**event, "before": "b" * 40}, m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env, {**event, "repository":
                          {**event["repository"], "private": True}},
                    m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env, event, "b" * 40)

    def test_missing_five_observers_is_unknown_not_train(self):
        rows = fake_rows()
        texts = {r["id"]: {} for r in rows}
        phones = {r["id"]: {} for r in rows}
        result = m.evaluate(rows, texts, phones)
        self.assertEqual(result["tier_counts"], {"unknown": 168})
        self.assertEqual(result["new_groups"],
                         {"train": 0, "development": 0})
        self.assertEqual(result["status"],
                         "STOP_NEWTRAIN_INSUFFICIENT_INDEPENDENT_GROUPS")
        self.assertFalse(result["training_snapshot_allowed"])
        self.assertEqual(result["candidate_rows"], [])
        self.assertTrue(result["new_calibration_and_locked_sources_required"])

    def test_only_full_three_class_source_groups_enter_candidate_snapshot(self):
        rows = fake_rows()
        texts = {r["id"]: {} for r in rows}
        phones = {r["id"]: {} for r in rows}

        def mock_independent_decision(row, _asr, _phone):
            if row["requested_class"] is None:
                return {"tier": "unknown", "class": None,
                        "reason": "DIAGNOSTIC_ONLY", "margins": {},
                        "training_allowed": False}
            return {"tier": "silver", "class": row["requested_class"],
                    "reason": "MOCK_ONLY_NOT_ACTUAL_REVIEW", "margins": {},
                    "training_allowed": False}

        with patch.object(m.policy, "decide", mock_independent_decision):
            result = m.evaluate(rows, texts, phones)
        self.assertEqual(result["new_groups"],
                         {"train": 18, "development": 6})
        self.assertEqual(result["eligible_train_dev_counts"],
                         {"train": 90, "development": 30})
        self.assertEqual(result["tier_counts"],
                         {"silver": 120, "unknown": 48})
        self.assertFalse(result["training_snapshot_allowed"])
        self.assertTrue(all(x["requested_class"] in (0, 1, 2)
                            for x in result["candidate_rows"]))

    def test_no_silver_when_request_conflicts_with_acoustics(self):
        rows = fake_rows()
        texts = {r["id"]: {} for r in rows}
        phones = {r["id"]: {} for r in rows}

        def wrong_acoustic_class(row, _asr, _phone):
            return {"tier": "silver", "class": 2,
                    "reason": "TEST_FORGED_DISAGREEMENT",
                    "margins": {}, "training_allowed": False}

        with patch.object(m.policy, "decide", wrong_acoustic_class):
            result = m.evaluate(rows, texts, phones)
        self.assertEqual(result["new_groups"],
                         {"train": 0, "development": 0})
        self.assertEqual(result["candidate_rows"], [])
        self.assertTrue(any(x["reason"] == "REQUEST_ACOUSTIC_CONTRADICTION"
                            for x in result["rows"]))

    def test_historical_full_pcm_reader_fails_closed_on_missing_capture(self):
        with tempfile.TemporaryDirectory() as td:
            rows = fake_rows()
            record, facts = m.prior_join.packet(
                td, "qwen", rows, m.original_observer_run())
            self.assertEqual(facts["status"], "MISSING_OBSERVER_ARTIFACT")
            self.assertEqual(set(record), {r["id"] for r in rows})
            self.assertTrue(all(x is None for x in record.values()))


if __name__ == "__main__":
    unittest.main()
