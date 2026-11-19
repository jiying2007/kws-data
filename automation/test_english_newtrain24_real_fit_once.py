"""No-network contract tests for a real but not automatically promotable new model."""
from __future__ import annotations
from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "newtrain24_numeric_test",
    HERE / "english_newtrain24_real_fit_once.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def samples(new=False):
    roles = ((["train"] * 8 + ["development"] * 3) if new else
             (["train"] * 28 + ["development"] * 13 +
              ["calibration"] * 11 + ["locked-test"] * 11))
    return [{
        "id": ("new" if new else "old") + str(i),
        "corpus": "newtrain24" if new else "piper24",
        "role": role,
        "source_group_id": ("N-" if new else "O-") + str(i//4),
        "source_pcm_sha256": f"{i+(1000 if new else 1):064x}",
    } for i, role in enumerate(roles)]


class ActualNewtrainDataAndModelContract(unittest.TestCase):
    def test_exact_five_model_8_3_readback_and_fresh_only(self):
        data = m.source_check()
        self.assertEqual(data["old_train_dev"], [28, 13])
        self.assertEqual(data["new_train_dev"], [8, 3])
        self.assertEqual(data["updates"], 96)
        self.assertEqual(data["seed"], 20261123)
        self.assertEqual(data["predefined_dev_threshold"], 0.8)
        self.assertEqual(data["full_locked_evaluation"], "DENIED")
        self.assertEqual(data["new_model_forwards_in_pr"], 0)
        self.assertEqual(m.SCREEN_ID, 14489594100)
        self.assertEqual(m.SOURCE_ID, 14486048125)

    def test_only_first_real_protected_main_successor(self):
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": m.REPO,
               "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main",
               "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": "444800001",
               "GITHUB_SHA": "a" * 40}
        event = {"before": m.BEFORE, "after": "a" * 40,
                 "ref": "refs/heads/main", "forced": False,
                 "deleted": False,
                 "repository": {"full_name": m.REPO, "private": False}}
        self.assertEqual(m.guard(env, event, m.BEFORE)["newtrain_screen"],
                         m.SCREEN_ID)
        with self.assertRaises(ValueError):
            m.guard({**env, "GITHUB_RUN_ATTEMPT": "2"}, event, m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard({**env, "GITHUB_EVENT_NAME": "pull_request"}, event, m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env, {**event, "before": "b" * 40}, m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env, {**event, "repository":
                          {"full_name": m.REPO, "private": True}}, m.BEFORE)
        with self.assertRaises(ValueError):
            m.guard(env, event, "b" * 40)

    def test_exact_63_prior_rows_but_never_train_old_cal_or_locked(self):
        old = samples()
        new = samples(new=True)
        prior = {"status": "MACHINE_SILVER_WEAK_RESEARCH_TRAIN_INPUT_BOUND",
                 "source_group_counts": {"train": 8, "development": 4,
                                         "calibration": 3, "locked-test": 3},
                 "rows": old}
        with patch.object(m.binding, "admission", return_value=prior) as old_fn:
            with patch.object(m, "bind_new",
                              return_value=(new, {"new_candidate_rows": 11})):
                joined, meta = m.all_train_dev("/no-original-audio-is-read")
        self.assertEqual(Counter(row["role"] for row in joined),
                         {"train": 36, "development": 16})
        self.assertEqual(len(joined), 52)
        self.assertTrue(all(x["role"] in ("train", "development")
                            for x in joined))
        old_fn.assert_called_once_with("/no-original-audio-is-read",
                                       verify_audio=False)
        self.assertEqual(meta["new_candidate_rows"], 11)

    def test_duplicates_between_old_and_new_are_hard_stop(self):
        old = samples()
        new = samples(new=True)
        new[0]["source_pcm_sha256"] = old[0]["source_pcm_sha256"]
        prior = {"status": "MACHINE_SILVER_WEAK_RESEARCH_TRAIN_INPUT_BOUND",
                 "source_group_counts": {"train": 8, "development": 4,
                                         "calibration": 3, "locked-test": 3},
                 "rows": old}
        with patch.object(m.binding, "admission", return_value=prior):
            with patch.object(m, "bind_new",
                              return_value=(new, {})):
                with self.assertRaises(ValueError):
                    m.all_train_dev("/no-replay")

    def test_wrong_old_locked_source_count_rejected(self):
        old = samples()
        old[0]["role"] = "locked-test"
        prior = {"status": "MACHINE_SILVER_WEAK_RESEARCH_TRAIN_INPUT_BOUND",
                 "source_group_counts": {"train": 8, "development": 4,
                                         "calibration": 3, "locked-test": 3},
                 "rows": old}
        with patch.object(m.binding, "admission", return_value=prior):
            with patch.object(m, "bind_new",
                              return_value=(samples(new=True), {})):
                with self.assertRaises(ValueError):
                    m.all_train_dev("/untrusted-prior")

    def test_any_missing_source_or_review_zip_is_stop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises((ValueError, FileNotFoundError)):
                m.bind_new(root / "absent-source.zip",
                           root / "absent-screen.zip")

    def test_source_and_screen_artifacts_have_distinct_sha_and_runs(self):
        self.assertNotEqual(m.SOURCE["sha256"], m.SCREEN["sha256"])
        self.assertNotEqual(m.SOURCE_RUN, m.SCREEN_RUN)
        self.assertEqual(m.SOURCE_HEAD,
                         "9113713ad4fcb87e111f373db35922f61775de1e")
        self.assertEqual(m.SCREEN_HEAD, m.BEFORE)
        self.assertEqual(m.SOURCE["bytes"], 3353181)
        self.assertEqual(m.SCREEN["bytes"], 20256)


if __name__ == "__main__":
    unittest.main()
