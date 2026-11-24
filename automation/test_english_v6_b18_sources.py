"""Offline B18 provenance/fail-closed fixtures; no real source or model calls."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b18_source as source
import english_v6_b18_source_ancestry as lineage


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity():
    return dict(run_id="1234567", run_head="a" * 40, run_attempt=1,
                run_branch=lineage.BRANCH, prepared_parent="b" * 40)


def authorization():
    design = lineage.read_plan()
    return dict(schema=lineage.SCHEMA, assessment_sha256=lineage.ASSESSMENT_SHA,
                base_source_ancestry_sha256=lineage.BASE_ANCESTRY_SHA,
                source_execution=identity(), original_runs={
                    name: dict(id=pin["id"], head_sha=pin["head"], head_branch=pin["branch"],
                               status="completed", conclusion=pin["conclusion"], run_attempt=1)
                    for name, pin in design["original_B17_source_proof"]["runs"].items()},
                B17_designated_KWS_runs={"total_count": 0, "workflow_runs": []},
                original_proof_locks=lineage.proof_locks(), new_natural_calls=0,
                old_replication_opened=False, new_training_allowed=False, new_KWS_forwards=0)


def history_fixture():
    """Synthetic receipt shape only; never written as an actual source result."""
    design = lineage.read_plan()
    seen = set(design["all_B17_primary_exclusions"]["pcm_sha256"])
    seen.update(digest("offline-history-fixture-" + str(i)) for i in range(1492 - len(seen)))
    groups = set(design["all_B17_primary_exclusions"]["source_groups"])
    groups.update("offline-history-fixture-" + str(i) for i in range(312 - len(groups)))
    _, old = source.previous.lineage.read_plan()
    people = set(design["all_B17_primary_exclusions"]["canonical_people"])
    people.update(old["legacy_natural"]["all_exposed_speakers"])
    for roster in old["historical_rosters"].values():
        people.update(roster["speakers"])
    archives = {key: dict(bytes=1, sha256=value) for key, value in design["base_original_archive_sha256"].items()}
    archives.update({str(pin["id"]): dict(bytes=pin["bytes"], sha256=pin["sha256"])
                     for pin in lineage.proof_locks().values()})
    return dict(schema=lineage.SCHEMA, assessment_sha256=lineage.ASSESSMENT_SHA,
                base_source_ancestry_sha256=lineage.BASE_ANCESTRY_SHA, original_archives=archives,
                all_pcm_sha256=sorted(seen), all_source_groups=sorted(groups), canonical_people=sorted(people),
                old_replication_opened=False, B17_source_STOP_retained=True,
                B17_all_original_slots_excluded=True)


def manifest_fixture():
    plan = source.source_plan("primary")
    chunk = bytes(2)
    rows = [dict(row, pcm_sha256=digest(row["id"]), wav_sha256=digest("wav-" + row["id"]),
                 samples=1, basic_audio_ok=True, whole_native_resample=True, crop_calls=0,
                 native_pcm_sha256=lineage.sha(chunk),
                 native_chunks=[dict(file="000.s16le", samples=1, native_pcm_sha256=lineage.sha(chunk))])
            for row in plan["rows"]]
    return dict(schema=source.SCHEMA, panel="primary", rows=rows,
                registered_plan_file_sha256=source.FILES["primary"],
                plan_sha256=source.B13.b10.sha(source.B13.b10.canon(plan)),
                assessment_sha256=source.ASSESSMENT_SHA, source_ancestry_sha256=lineage.BASE_ANCESTRY_SHA,
                tts_calls=96, tts_logical_originals=96, tts_public_synthesize_api_invocations=96,
                new_kws_forwards=0, new_natural_source_calls=0, new_natural_observer_calls=0,
                human_gold=False, B17_source_stop_retained=True, B18_replication_opened=False,
                all_prior_replication_sealed=True,
                original_B17_source_proof=lineage.read_plan()["original_B17_source_proof"]["source_inputs"],
                native_output_chunks=96, internal_tts_model_forwards=None,
                internal_onnx_session_run_calls=None, internal_tts_forwards_instrumented=False,
                source_execution=identity())


def saved_fixture(root):
    manifest = manifest_fixture()
    write(root / "AUTHORIZATION.json", authorization())
    write(root / "HISTORY.json", history_fixture())
    write(root / "STARTED.json", {"plan_sha256": source.FILES["primary"], "one_shot": True})
    for row in manifest["rows"]:
        write(root / "starts" / (row["id"] + ".json"), {"id": row["id"], "attempt": 1, "seed": row["seed"]})
        directory = root / "native" / row["id"]
        directory.mkdir(parents=True)
        (directory / "000.s16le").write_bytes(bytes(2))
    manifest.update(history_receipt_sha256=source.qa.sha(root / "HISTORY.json"),
                    source_authorization_sha256=source.qa.sha(root / "AUTHORIZATION.json"))
    write(root / "MANIFEST.json", manifest)
    return manifest


def inspect_fixture(path):
    return dict(pcm_sha256=digest(path.stem), wav_sha256=digest("wav-" + path.stem), samples=1, basic_audio_ok=True)


class B18Sources(unittest.TestCase):
    def test_frozen_models_quality_and_source_budget(self):
        design = lineage.read_plan()
        self.assertEqual(design["training_assessment_sha256"], source.previous.lineage.ASSESSMENT_SHA)
        self.assertEqual(design["frozen_model_pt_sha256"]["control"],
                         "144370e96fcf58da50114280d728867781934e3c6538d2b2d769babb7934c81e")
        self.assertEqual(design["source_gate"], lineage.old_eval.GATE)
        self.assertEqual(design["maximum_source_calls"], 96)
        self.assertIsNone(design["internal_tts_model_forwards"])
        self.assertFalse(design["new_training_allowed"])
        self.assertEqual(design["new_natural_observer_calls"], 0)

    def test_fresh_primary_and_unchanged_inherited_requests(self):
        primary, replica = (source.source_plan(panel) for panel in ("primary", "replication"))
        self.assertEqual({row["speaker_index"] for row in primary["rows"]}, set(range(766, 790)))
        self.assertEqual({row["seed"] for row in primary["rows"]}, set(range(20262100, 20262124)))
        self.assertEqual(replica["rows"], source.previous.source_plan("replication")["rows"])
        self.assertEqual(len(primary["rows"]), 96)
        self.assertTrue(all(row["observed_class"] is None for row in primary["rows"]))
        self.assertEqual(lineage.read_plan()["replication"]["natural"],
                         source.previous.lineage.read_plan()[0]["replication"]["natural"])

    def test_real_config_and_reserved_natural_canonical_exclusion(self):
        history = history_fixture()
        source.validate_config(source.PLANS / "piper-config.json", set(), set(history["canonical_people"]))
        design = copy.deepcopy(lineage.read_plan())
        design["replication"]["natural"]["speakers"][0] = "6620"
        with patch.object(lineage, "read_plan", return_value=design):
            with self.assertRaisesRegex(ValueError, "speaker ancestry"):
                source.validate_config(source.PLANS / "piper-config.json", set(), set())

    def test_complete_raw_B17_increment_includes_failed_and_unknown_sources(self):
        design = lineage.read_plan()
        original = design["all_B17_primary_exclusions"]
        self.assertEqual(len(original["pcm_sha256"]), 120)
        self.assertEqual(len(original["canonical_people"]), 36)
        self.assertEqual(original["natural_original_slots"], 24)
        self.assertIsNone(design["original_B17_source_proof"]["source_inputs"]["synth_screen"])
        self.assertEqual(len(design["original_B17_source_proof"]["failed_original_ids"]), 2)
        self.assertEqual(design["reused_natural_primary"]["unknown_originals"], 3)
        self.assertFalse(design["B17_scoring_allowed"])

    def test_all_38_original_archives_remain_required(self):
        ids = lineage.required_archives()
        self.assertEqual(len(ids), 38)
        self.assertEqual(len(set(ids)), 38)
        self.assertTrue(set(source.previous.required_archives()) < set(ids))
        self.assertEqual(len(lineage.proof_locks()), 13)
        self.assertNotIn("synth_screen", lineage.proof_locks())

    def test_observer_and_parser_hashes_are_real_frozen_implementations(self):
        contract = source.code_contract()
        self.assertEqual(contract["parser_contract_sha256"],
                         "7bf7fa315b016f034cfa5e0917d57db2c9d7d25ee7791c24e84ce2d2acff8591")
        self.assertEqual(contract["maximum_native_calls"], 480)

    def event(self):
        run = identity()
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=lineage.REPO, GITHUB_EVENT_NAME="push",
                   GITHUB_REF=lineage.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA=run["run_head"],
                   GITHUB_RUN_ID=run["run_id"])
        event = dict(before=run["prepared_parent"], after=run["run_head"], ref=lineage.BRANCH,
                     forced=False, deleted=False, repository=dict(full_name=lineage.REPO, private=False))
        return env, event

    def test_first_public_exact_parent_guard_rejects_replay_and_private(self):
        env, event = self.event()
        self.assertEqual(lineage.execution_guard("b" * 40, env, event, "b" * 40), identity())
        for change in (dict(GITHUB_RUN_ATTEMPT="2"), dict(GITHUB_RUN_ID=""), dict(GITHUB_REF="refs/heads/main")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                lineage.execution_guard("b" * 40, dict(env, **change), event, "b" * 40)
        for key in ("forced", "deleted"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                lineage.execution_guard("b" * 40, env, dict(event, **{key: True}), "b" * 40)
        private = copy.deepcopy(event); private["repository"]["private"] = True
        with self.assertRaises(ValueError):
            lineage.execution_guard("b" * 40, env, private, "b" * 40)
        with self.assertRaises(ValueError):
            lineage.execution_guard("c" * 40, env, event, "b" * 40)

    def test_failed_execution_guard_does_not_download_originals(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(lineage.os.environ, {}, clear=True), \
                patch.object(lineage, "gh") as network:
            with self.assertRaisesRegex(ValueError, "first designated public"):
                lineage.authorize(Path(tmp) / "new", "b" * 40)
            network.assert_not_called()

    def test_stale_authorization_cannot_consume_any_TTS(self):
        receipt = authorization(); receipt["source_execution"]["run_head"] = "c" * 40
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(lineage, "execution_guard", return_value=identity()), \
                patch.object(source, "code_contract"), \
                patch.object(lineage, "require_authorization", return_value=receipt), \
                patch.object(lineage, "history_receipt") as old_reads, \
                patch.object(source.B13, "render_panel") as renderer:
            with self.assertRaisesRegex(ValueError, "before TTS"):
                source.generate(Path(tmp) / "new", Path(tmp) / "old", "b" * 40)
            renderer.assert_not_called()
            old_reads.assert_not_called()

    def test_nonzero_prior_KWS_exposure_rejects_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = authorization()
            receipt["B17_designated_KWS_runs"] = {"total_count": 1, "workflow_runs": [{"id": 1}]}
            write(Path(tmp) / "AUTHORIZATION.json", receipt)
            with self.assertRaisesRegex(ValueError, "source STOP/natural PASS"):
                lineage.require_authorization(Path(tmp))

    def test_history_cannot_drop_original_slots_or_change_old_archives(self):
        original = history_fixture()
        lineage.validate_history(original)
        removed = copy.deepcopy(original)
        removed["all_pcm_sha256"].remove(lineage.read_plan()["all_B17_primary_exclusions"]["pcm_sha256"][0])
        with self.assertRaisesRegex(ValueError, "denominators"):
            lineage.validate_history(removed)
        old_hash = copy.deepcopy(original)
        key = next(iter(lineage.read_plan()["base_original_archive_sha256"]))
        old_hash["original_archives"][key]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pre-B17 archive"):
            lineage.validate_history(old_hash)

    def test_canonical_old_speaker_cannot_hide_behind_group_prefix(self):
        history = history_fixture()
        history["canonical_people"].append("6620")
        with self.assertRaisesRegex(ValueError, "canonical identities excluded"):
            source.validate_manifest(manifest_fixture(), history, authorization())

    def test_manifest_cannot_claim_internal_forward_count_or_other_run(self):
        for key, value in (("internal_tts_model_forwards", 96), ("tts_calls", 95),
                           ("source_execution", dict(identity(), run_head="c" * 40))):
            with self.subTest(key=key):
                meta = manifest_fixture(); meta[key] = value
                with self.assertRaisesRegex(ValueError, "honest source-call accounting"):
                    source.validate_manifest(meta, history_fixture(), authorization())

    def test_saved_source_requires_original_start_receipts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(source.qa.pcm_tools, "inspect_wav", side_effect=inspect_fixture):
            root = Path(tmp); manifest = saved_fixture(root)
            self.assertEqual(len(source.read_source(root)), 96)
            first = manifest["rows"][0]
            write(root / "starts" / (first["id"] + ".json"), {"id": first["id"], "attempt": 2, "seed": first["seed"]})
            with self.assertRaisesRegex(ValueError, "one fixed original TTS request"):
                source.read_source(root)

    def test_registration_is_rebuilt_from_authenticated_source_receipts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(source.qa.pcm_tools, "inspect_wav", side_effect=inspect_fixture):
            root = Path(tmp); saved_fixture(root)
            expected = source.registration(root)
            self.assertEqual({k: expected[k] for k in identity()}, identity())
            self.assertEqual(expected["source_manifest_sha256"], source.qa.sha(root / "MANIFEST.json"))
            receipt = json.loads((root / "AUTHORIZATION.json").read_bytes())
            receipt["source_execution"]["run_head"] = "c" * 40
            write(root / "AUTHORIZATION.json", receipt)
            with self.assertRaisesRegex(ValueError, "source-history and opening receipts"):
                source.registration(root)

    def test_incomplete_observation_stops_before_any_admission(self):
        rows = source.source_plan("primary")["rows"]
        coverage = dict.fromkeys(source.qa.observers.PROVIDERS, 96); coverage["parakeet"] = 95
        with tempfile.TemporaryDirectory() as tmp, patch.object(source, "read_source", return_value=rows), \
                patch.object(source, "registration", return_value={}), \
                patch.object(source.observers, "load_observations", return_value=({}, {}, coverage)), \
                patch.object(source.qa, "admit") as admit:
            with self.assertRaisesRegex(ValueError, "five valid complete"):
                source.screen(Path(tmp), Path(tmp), Path(tmp) / "SCREEN.json")
            admit.assert_not_called()
            self.assertFalse((Path(tmp) / "SCREEN.json").exists())

    def test_unknown_sources_keep_all_96_original_slots(self):
        rows = source.source_plan("primary")["rows"]
        coverage = dict.fromkeys(source.qa.observers.PROVIDERS, 96)
        empty = {row["id"]: {} for row in rows}
        outcome = dict(training_allowed=False, **{"class": None}, tier="unknown", reason="offline fixture", margins={})
        with tempfile.TemporaryDirectory() as tmp, patch.object(source, "read_source", return_value=rows), \
                patch.object(source, "registration", return_value={}), \
                patch.object(source.observers, "load_observations", return_value=(empty, empty, coverage)), \
                patch.object(source.qa, "admit", return_value=outcome):
            root = Path(tmp); (root / "MANIFEST.json").write_text("offline fixture only")
            source.screen(root, root, root / "SCREEN.json")
            result = json.loads((root / "SCREEN.json").read_bytes())
            self.assertEqual(len(result["rows"]), 96)
            self.assertEqual(result["unknown_count"], 96)
            self.assertEqual(result["source_gate"], "INSUFFICIENT_SOURCE")
            self.assertEqual(result["frozen_minima"], dict(positive_originals=12, ordinary_originals=24, paired_source_groups=8))
            self.assertFalse(result["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
