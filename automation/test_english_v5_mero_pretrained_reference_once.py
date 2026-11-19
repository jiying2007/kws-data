"""Stdlib-only tests for the single Mero, read-only pretrained comparison."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "mero_pretrained_check", HERE / "english_v5_mero_pretrained_reference_once.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def toy_observations():
    rows = []
    for role, count in (("train", 6), ("development", 2)):
        for index in range(count):
            for clazz in (0, 1):
                rows.append({"id": f"{role}_{index}_{clazz}",
                    "source_group_id": f"{role}_{index}", "role": role,
                    "mono_class": clazz,
                    "events": ([{"mapped_keyword": "HEY_MERO"}]
                               if clazz == 1 and index % 2 else [])})
    return rows


class SingleMeroPretrainedReference(unittest.TestCase):
    def test_reuses_authentic_single_target_not_failed_model(self):
        state = m.source_contract()
        self.assertEqual(state["keyword_ids"], ["hey_mero"])
        self.assertEqual(state["reference_model"], "zh-en")
        self.assertEqual(len(state["original_source_artifact_ids"]), 8)
        self.assertEqual(state["new_audio_generation"], 0)
        self.assertEqual(state["optimizer_updates"], 0)
        self.assertFalse(state["calibration_or_locked_access"])
        self.assertFalse(state["model_promotion_allowed"])
        self.assertEqual(len(state["model_sha256"]), 64)
        self.assertEqual(m.MODEL_BYTES, 32885699)
        self.assertEqual(m.KEYWORD, "HEY_MERO")

    def test_one_official_phone_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/"tokens.txt").write_text("HH 1\nEY 2\nM 3\nEH 4\nR 5\nOW 6\n")
            (p/"en.phone").write_text("HEY HH EY\nMERO M EH R OW\n")
            def fake(texts, **kw):
                self.assertEqual(texts, ["HEY MERO"])
                self.assertEqual(kw["tokens_type"], "phone+ppinyin")
                self.assertIsNone(kw["bpe_model"])
                self.assertTrue(kw["lexicon"].endswith("en.phone"))
                return [["HH", "EY", "M", "EH", "R", "OW"]]
            f = m.make_mero_keyword({"tokens":p/"tokens.txt",
                "lexicon":p/"en.phone"}, p, tokenizer=fake)
            self.assertEqual(f.read_text(), "HH EY M EH R OW @HEY_MERO\n")

    def test_missing_lexicon_or_unknown_phone_is_rejected(self):
        for lexicon, output in (
            ("HEY HH EY\n", [["HH", "EY"]]),
            ("MERO M EH R OW\n", [["HH", "EY"]]),
            ("HEY HH EY\nMERO M EH R OW\n", []),
            ("HEY HH EY\nMERO M EH R OW\n", [["HH"]]),
            ("HEY HH EY\nMERO M EH R OW\n", [["HH", "BAD"]]),
            ("HEY HH EY\nMERO M EH R OW\n", [["HH", "EY"], ["M", "EH"]]),
        ):
            with self.subTest(lexicon=lexicon,output=output):
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)
                    (p/"tokens").write_text("HH 1\nEY 2\nM 3\nEH 4\nR 5\nOW 6\n")
                    (p/"en.phone").write_text(lexicon)
                    with self.assertRaises(ValueError):
                        m.make_mero_keyword({"tokens":p/"tokens",
                            "lexicon":p/"en.phone"},p,
                            tokenizer=lambda *_args,**_kw:output)

    def test_source_disjoint_reference_summary(self):
        summary = m.summarize(toy_observations())
        self.assertEqual(summary["train"]["source_groups"], 6)
        self.assertEqual(summary["train"]["mero_positive_original_clips"], 6)
        self.assertEqual(summary["train"]["mero_positive_detected_clips"], 3)
        self.assertEqual(summary["development"]["source_groups"], 2)
        self.assertEqual(summary["development"]["ordinary_triggered_clips"], 0)
        self.assertEqual(summary["development"]["mero_positive_detected_clips"], 1)

    def test_leaks_and_incomplete_roles_are_rejected(self):
        source = toy_observations()
        for bad in ("leak", "duplicate", "insufficient"):
            rows = [dict(r) for r in source]
            if bad == "leak":
                rows[-1]["source_group_id"] = "train_0"
            elif bad == "duplicate":
                rows[-1]["id"] = rows[0]["id"]
            else:
                rows = rows[:12]
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    m.summarize(rows)

    def test_only_exact_public_pr_first_attempt_can_execute(self):
        env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": m.REPO,
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_HEAD_REF": m.BRANCH,
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "44460001000"}
        event = {"action": "opened", "repository":
                    {"full_name": m.REPO, "private": False},
                 "pull_request": {
                    "head": {"sha": "a" * 40, "ref": m.BRANCH,
                             "repo": {"full_name": m.REPO}},
                    "base": {"ref": "main"}}}
        self.assertEqual(m.run_guard(env, event)["head_sha"], "a" * 40)
        for mutate in (
            ({**env, "GITHUB_RUN_ATTEMPT": "2"}, event),
            ({**env, "GITHUB_HEAD_REF": "not-mero"}, event),
            (env, {**event, "repository":
                   {"full_name": m.REPO, "private": True}}),
            (env, {**event, "action": "closed"}),
        ):
            with self.assertRaises(ValueError):
                m.run_guard(*mutate)

    def test_source_does_not_rewrite_review_or_model_weights(self):
        raw = (HERE / "english_v5_mero_pretrained_reference_once.py").read_text()
        self.assertNotIn("Hey Vosen", raw)
        self.assertNotIn("VOSEN", raw)
        self.assertNotIn("torch.optim", raw)
        self.assertIn("v5.admission(sources)", raw)
        self.assertIn("v5.base.load_source_pcm", raw)
        self.assertIn("pretrained.stream_events", raw)
        self.assertIn("development_is_exposed_and_synthetic", raw)


if __name__ == "__main__":
    unittest.main()
