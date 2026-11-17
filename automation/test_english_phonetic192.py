"""Source-only protocol tests: no model, network, compiler or real audio."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_phonetic192_recipe as r
import english_phonetic192_once as w


class Phonetic192Tests(unittest.TestCase):
    def plan(self):
        return json.loads(Path(__file__).with_name("english_phonetic192_plan.json").read_text())

    def test_fresh_balanced_source_design(self):
        recipe = r.recipe()
        self.assertEqual(len(recipe["rows"]), 192)
        self.assertEqual(len({x["id"] for x in recipe["rows"]}), 192)
        self.assertEqual(len({x["request_sha256"] for x in recipe["rows"]}), 192)
        for group in r.source_groups():
            rows = [x for x in recipe["rows"] if x["source_group_id"] == group["source_group_id"]]
            self.assertEqual(sorted(x["requested_class"] for x in rows), [0, 0, 0, 0, 1, 1, 2, 2])
            self.assertEqual({x["role"] for x in rows}, {group["role"]})
            self.assertNotIn(group["source_speaker_id"], recipe["exclusion_speakers"])
        self.assertEqual([sum(g["role"] == x for g in r.source_groups()) for x in r.ROLES], [12, 4, 4, 4])

    def test_lexicon_is_explicit_hypothesis(self):
        self.assertIn("mero M EH1 R OW0", r.LEXICON["entries"])
        self.assertFalse(r.recipe()["final_owner_ipa_declared"])
        self.assertEqual({c["pronunciation_hypothesis"] for c in r.CELLS if c["requested_class"] == 2},
            {"mero-us-eh-v1", "mero-us-i-v1"})

    def test_real_source_pins(self):
        self.assertEqual(w.validate_plan(self.plan())["source_count"], 192)

    def test_recipe_or_budget_drift_rejected(self):
        for key, value in (("recipe_sha256", "0" * 64), ("maximum_tts_calls", 193),
                           ("retry_count", 1), ("observation_deadline_seconds", 99999)):
            p = self.plan(); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.validate_plan(p)

    def test_helper_and_worker_tamper_rejected(self):
        p = self.plan(); p["worker_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "WORKER_SOURCE_PIN"):
            w.validate_plan(p)
        p = self.plan(); first = next(iter(p["helper_sha256"])); p["helper_sha256"][first] = "0" * 64
        with self.assertRaisesRegex(ValueError, "HELPER_SOURCE_PIN"):
            w.validate_plan(p)

    def test_disabled_and_second_attempt_cannot_execute(self):
        p = self.plan(); p["enabled"] = False
        with self.assertRaisesRegex(ValueError, "PROTECTED_FIRST_MAIN_ONLY"):
            w.execution_guard(p, {}, {})
        # The guard, independently of file pin validation, rejects every replay
        # and every private or changed-predecessor execution context.
        p["enabled"] = True; p["next_main_before"] = "a" * 40
        env = {"GITHUB_REPOSITORY": "jiying2007/kws-data", "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/main", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "b" * 40}
        event = {"repository": {"private": False}, "before": "a" * 40, "after": "b" * 40}
        with patch.object(w, "validate_plan", return_value=r.recipe()):
            w.execution_guard(p, env, event)
            for changed in ({"GITHUB_RUN_ATTEMPT": "2"}, {"GITHUB_EVENT_NAME": "workflow_dispatch"},
                            {"GITHUB_REF": "refs/heads/other"}):
                with self.assertRaises(ValueError):
                    w.execution_guard(p, env | changed, event)
            with self.assertRaises(ValueError):
                w.execution_guard(p, env, event | {"repository": {"private": True}})
            with self.assertRaises(ValueError):
                w.execution_guard(p, env, event | {"before": "c" * 40})

    def test_all_four_wake_cells_use_exact_direct_ids_without_g2p(self):
        wake = [c for c in r.CELLS if c["requested_class"]]
        self.assertEqual(len(wake), 4)
        self.assertEqual({c["generation_mode"] for c in wake}, {"direct-phoneme-ids"})
        self.assertEqual([c["raw_phonemes"] for c in wake],
            ["hˈeɪ mˈɛɹoʊ.", "hˈeɪ mˈiːɹoʊ.", "hˈeɪ vˈoʊsən.", "hˈeɪ vˈoʊzən."])
        self.assertEqual([len(r.DIRECT_PHONE_IDS[c["cell_id"]]) for c in wake], [27, 29, 29, 29])
        class Config:
            phoneme_id_map = {x: [1] for cell in wake for x in cell["raw_phonemes"]}
        class Voice:
            config = Config()
            def phonemize(self, text):
                raise AssertionError("WAKE_CELL_G2P_MUST_NOT_RUN")
            def phonemes_to_ids(self, phones):
                return next(r.DIRECT_PHONE_IDS[cell["cell_id"]] for cell in wake if list(cell["raw_phonemes"]) == phones)
        for row in wake:
            phones, ids = w.request_phonemes(Voice(), row)
            self.assertEqual(phones, list(row["raw_phonemes"]))
            self.assertEqual(ids, r.DIRECT_PHONE_IDS[row["cell_id"]])

    def test_all_prior_piper_voices_excluded_and_roles_frozen(self):
        self.assertEqual({g["source_speaker_index"] for g in r.source_groups()}, set(range(34, 58)))
        self.assertEqual(len(r.OLD_PIPER_IDS), 34)
        self.assertEqual(len(r.recipe()["exclusion_speakers"]), 74)
        self.assertEqual([g["role"] for g in r.source_groups()], ["train"]*12+["development"]*4+["calibration"]*4+["locked-test"]*4)
        self.assertTrue(all(row["id"].startswith("enp192-") for row in r.recipe()["rows"]))

    def test_new_campaign_and_actual_synthesis_configuration_are_bound(self):
        self.assertEqual(r.CAMPAIGN_ID, "piper-libritts-direct-34-57-v1")
        self.assertEqual(r.SYNTHESIS, {"length_scale": 1.2, "noise_scale": 0.333,
            "noise_w_scale": 0.333, "normalize_audio": False, "volume": 1.0})
        for field, value in (("campaign_id", "old-campaign"), ("synthesis_config", dict(r.SYNTHESIS, length_scale=1.0))):
            plan = self.plan(); plan[field] = value
            with self.assertRaises(ValueError): w.validate_plan(plan)
        plan = self.plan(); plan["silver_admission"]["source_design_id"] = "old-design"
        with self.assertRaises(ValueError): w.validate_plan(plan)

    def test_partial_source_keeps_full_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan = self.plan()
            w.once(root / "RECIPE.json", r.recipe())
            w.once(root / "EXECUTION_PLAN.json", plan)
            w.once(root / "STARTED.json", {"schema": w.SCHEMA, "stage": "generate",
                "maximum_calls": 192, "retry_count": 0, "run": {"test_fixture": True}})
            first = r.recipe()["rows"][0]
            folder = root / "audio" / first["id"]; folder.mkdir(parents=True)
            w.once(folder / "STARTED.json", {"id": first["id"], "test_fixture": True})
            result = w.source_readback(root, plan)
            self.assertEqual(result["valid_count"], 0)
            self.assertEqual(len(result["rows"]), 192)
            self.assertEqual(result["rows"][0]["generation_status"], "UNKNOWN_AFTER_START_NO_RETRY")
            self.assertTrue(all(x["generation_status"] == "NOT_STARTED" for x in result["rows"][1:]))

    def test_source_split_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); recipe = r.recipe(); recipe["rows"][0]["role"] = "locked-test"
            w.once(root / "RECIPE.json", recipe); w.once(root / "EXECUTION_PLAN.json", self.plan())
            with self.assertRaisesRegex(ValueError, "SOURCE_RECIPE_OR_PLAN"):
                w.source_readback(root, self.plan())

    def test_create_once_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); w.once(root / "receipt.json", {"status": "started"})
            with self.assertRaisesRegex(ValueError, "CREATE_ONCE_FILE"):
                w.once(root / "receipt.json", {"status": "retry"})
            for name in ("../outside", "/absolute", "a/../b", "a//b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    w.within(root, name)

    def test_wrong_capture_pcm_never_joins(self):
        rows = [dict(x, generation_status="success", pcm_sha256="a" * 64,
                     wav_sha256="b" * 64, pcm_samples=16000) for x in r.recipe()["rows"]]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); row = rows[0]
            folder = root / "evidence" / "qwen" / row["id"]; folder.mkdir(parents=True)
            w.once(folder / "CAPTURE.json", r.observer_identity("qwen") | {
                "id": row["id"], "source_pcm_sha256": "c" * 64,
                "source_wav_sha256": row["wav_sha256"], "source_samples": 16000})
            with self.assertRaisesRegex(ValueError, "CAPTURE_SOURCE_BINDING"):
                w.capture_evidence(root, {"rows": rows})


if __name__ == "__main__":
    unittest.main()
