"""Offline B20 provenance and first-source safety fixtures; no native calls."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b20_source as source
import english_v6_b20_source_ancestry as lineage


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def identity():
    return dict(run_id="1234567", run_head="a" * 40, prepared_parent="b" * 40,
                run_attempt=1, run_branch=lineage.BRANCH)


def history_fixture():
    """Assemble frozen metadata only; never asserts real source execution."""
    design = lineage.read_plan()
    seen, groups, people = lineage.previous.registered_history()
    supplement, canary = design["historical_context24_supplement"], design["B19_all_source_exclusions"]
    seen.update(supplement["all_pcm_sha256"]); seen.update(canary["all_PCM_sha256"])
    groups.update(supplement["all_source_groups"]); groups.update(canary["source_groups"])
    people.update(supplement["canonical_people"]); people.update(canary["canonical_people"])
    old_proof = lineage.previous.read_plan()["original_B18_source_proof"]["durable_evidence"]["HISTORY.json"]
    archives = json.loads((lineage.PLANS.parent.parent / old_proof["path"]).read_bytes())["original_archives"]
    for pin in [*lineage.previous.proof_locks().values(), lineage.CONTEXT24, *lineage.proof_locks().values()]:
        archives[str(pin["id"])] = dict(bytes=pin["bytes"], sha256=pin["sha256"])
    return dict(schema=lineage.SCHEMA, assessment_sha256=lineage.ASSESSMENT_SHA,
        original_archives=archives, all_pcm_sha256=sorted(seen), all_source_groups=sorted(groups),
        canonical_people=sorted(people), B19_complete_canary_excluded=True,
        full_context24_originals_added=True, B17_B18_source_STOP_preserved=True, all_old_replication_sealed=True)


def authorization_fixture():
    return dict(schema=lineage.SCHEMA, assessment_sha256=lineage.ASSESSMENT_SHA,
        source_execution=identity(), B19_original_artifacts=lineage.proof_locks(),
        B19_feasibility_result_sha256=lineage.read_plan()["B19_original_proof"]["feasibility_result_sha256"],
        B19_format_feasibility_only=True, B19_producer_file_sha256={},
        B19_canary_admission_allowed=False, old_replication_opened=False, new_KWS_forwards=0)


def manifest_fixture():
    plan = source.source_plan()
    rows = [dict(row, pcm_sha256=digest("offline-pcm-" + row["id"]),
                 wav_sha256=digest("offline-wav-" + row["id"]), samples=8000, rms=.01, rail_fraction=0,
                 basic_audio_ok=True, whole_native_resample=True, crop_calls=0,
                 native_pcm_sha256=digest("offline-native-" + row["id"]),
                 native_chunks=[dict(file="000.s16le", samples=11025,
                                     native_pcm_sha256=digest("offline-native-" + row["id"]))]) for row in plan["rows"]]
    return dict(schema=source.SCHEMA, panel="primary", **source.source_binding(), rows=rows,
        registered_plan_file_sha256=source.FILES["primary"], plan_sha256=source.B13.b10.sha(source.B13.b10.canon(plan)),
        source_execution=identity(), tts_calls=96, tts_logical_originals=96, tts_public_synthesize_api_invocations=96,
        native_output_chunks=96, internal_tts_model_forwards=None, internal_onnx_session_run_calls=None,
        internal_tts_forwards_instrumented=False, new_kws_forwards=0, human_gold=False,
        new_natural_source_calls=0, new_natural_observer_calls=0)


def saved_fixture(root):
    """Artificial bytes only; saved canary verification is explicitly mocked by callers."""
    meta, auth = manifest_fixture(), authorization_fixture()
    proof = root / "B19_PROOF"
    for name in lineage.PROOF_FILES: write(proof / name, {"offline_fixture_only": name})
    auth["B19_producer_file_sha256"] = {name: source.qa.sha(proof / name) for name in lineage.PROOF_FILES}
    write(root / "AUTHORIZATION.json", auth)
    write(root / "HISTORY.json", history_fixture())
    write(root / "STARTED.json", dict(plan_sha256=source.FILES["primary"], one_shot=True))
    for number, row in enumerate(meta["rows"]):
        write(root / "starts" / (row["id"] + ".json"), dict(id=row["id"], attempt=1, seed=row["seed"]))
        path = root / "wav" / (row["id"] + ".wav"); path.parent.mkdir(exist_ok=True)
        sample = struct.pack("<h", 1000 + number)
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 8000, "NONE", "not compressed")); wav.writeframes(sample * 8000)
        row.update(source.qa.pcm_tools.inspect_wav(path))
        native = root / "native" / row["id"] / "000.s16le"; native.parent.mkdir(parents=True)
        native.write_bytes(sample * 11025)
        row["native_pcm_sha256"] = source.qa.sha(native)
        row["native_chunks"][0]["native_pcm_sha256"] = source.qa.sha(native)
    meta.update(history_receipt_sha256=source.qa.sha(root / "HISTORY.json"),
                source_authorization_sha256=source.qa.sha(root / "AUTHORIZATION.json"))
    write(root / "MANIFEST.json", meta)
    return meta


class B20Sources(unittest.TestCase):
    def test_exact_new_format_rosters_and_old_frozen_models(self):
        design = lineage.read_plan()
        for role, first, last in (("primary", 798, 822), ("replication", 822, 846)):
            plan = source.source_plan(role)
            self.assertEqual(len(plan["rows"]), 96)
            self.assertEqual({r["speaker_index"] for r in plan["rows"]}, set(range(first, last)))
            self.assertEqual({r["length_scale"] for r in plan["rows"] if r["requested_class"] == 2}, {1.4, 1.8})
            self.assertTrue(all(r["observed_class"] is None and r["training_allowed"] is False for r in plan["rows"]))
        old = lineage.previous.previous.lineage.read_plan()
        self.assertEqual(design["frozen_model_pt_sha256"], old["frozen_model_pt_sha256"])
        self.assertEqual(design["source_gate"], old["source_gate"])
        self.assertEqual(design["quality_criteria"], old["quality_criteria"])
        self.assertFalse(design["B19_causal_fix_proven"])
        self.assertEqual(source.code_contract(), design["observer_code_contract"])

    def test_natural_new_nine_are_metadata_only_and_old_nine_stay_sealed(self):
        design = lineage.read_plan(); reserve = design["replication"]["natural"]
        lineage.validate_natural_reserve(design)
        self.assertEqual(reserve["speakers"], ["3982", "1116", "4137", "3374", "5192", "40", "1898", "6019", "2391"])
        self.assertEqual(len(reserve["available_census"]), 239)
        self.assertFalse(set(reserve["speakers"]) & set(reserve["old_dev_other_nine_stay_sealed"]))
        self.assertIsNone(reserve["selected_utterance_ids"])
        self.assertFalse(reserve["training_allowed"] or reserve["opened"])
        self.assertEqual((reserve["source_slots"], reserve["minimum_approved_utterances"], reserve["minimum_approved_speakers"]), (18, 12, 8))
        changed = copy.deepcopy(design); changed["replication"]["natural"]["speakers"][0] = "5849"
        with self.assertRaisesRegex(ValueError, "new fixed reserve"): lineage.validate_natural_reserve(changed)

    def test_exact_48_original_archives_and_raw_context24_supplement(self):
        receipt = history_fixture()
        self.assertEqual(len(lineage.required_archives()), 48)
        self.assertEqual(tuple(map(len, lineage.validate_history(receipt))), (2247, 356, 276))
        context = lineage.read_plan()["historical_context24_supplement"]
        self.assertEqual(context["additional_PCM"], 24)
        self.assertEqual(context["additional_groups"], 12)
        self.assertEqual(context["additional_canonical_people"], 0)
        self.assertTrue(set(context["all_pcm_sha256"]) <= set(receipt["all_pcm_sha256"]))
        self.assertEqual(receipt["original_archives"]["14790239936"]["sha256"], lineage.CONTEXT24["sha256"])

    def test_history_cannot_drop_unknown_canary_or_context_originals(self):
        design = lineage.read_plan()
        for key, removed in (("all_pcm_sha256", design["historical_context24_supplement"]["all_pcm_sha256"][0]),
                             ("all_pcm_sha256", design["B19_all_source_exclusions"]["all_PCM_sha256"][0]),
                             ("all_source_groups", design["historical_context24_supplement"]["all_source_groups"][0]),
                             ("canonical_people", design["B19_all_source_exclusions"]["canonical_people"][0])):
            receipt = history_fixture(); receipt[key].remove(removed)
            with self.subTest(key=key, removed=removed), self.assertRaisesRegex(ValueError, "complete B20 original ancestry"):
                lineage.validate_history(receipt)

    def test_history_archive_identity_and_duplicates_are_rejected(self):
        receipt = history_fixture(); receipt["original_archives"]["14790239936"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): lineage.validate_history(receipt)
        receipt = history_fixture(); receipt["all_pcm_sha256"].append(receipt["all_pcm_sha256"][0])
        with self.assertRaisesRegex(ValueError, "duplicated ancestry"): lineage.validate_history(receipt)

    def test_unfinalized_assessment_stops_all_runtime_before_remote_or_source_access(self):
        with patch.object(lineage, "ASSESSMENT_SHA", None), patch.object(lineage, "gh") as remote, \
             patch.object(source.B13, "render_panel") as render, patch.object(source.observers.raw_asr, "_load") as backend:
            calls = [lambda: lineage.authorize(Path("/unused"), "b" * 40),
                     lambda: lineage.require_authorization(Path("/unused")),
                     lambda: lineage.history(Path("/unused")), lambda: source.read_source(Path("/unused")),
                     lambda: source.generate(Path("/unused"), Path("/unused-prior"), "b" * 40)]
            for call in calls:
                with self.assertRaisesRegex(ValueError, "B20_INERT"): call()
            remote.assert_not_called(); render.assert_not_called(); backend.assert_not_called()

    def test_actual_public_first_parent_guard_and_replay_rejection(self):
        current = identity()
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=lineage.REPO, GITHUB_EVENT_NAME="push",
                   GITHUB_REF=lineage.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID=current["run_id"], GITHUB_SHA=current["run_head"])
        event = dict(before=current["prepared_parent"], after=current["run_head"], ref=lineage.BRANCH,
                     forced=False, deleted=False, repository=dict(full_name=lineage.REPO, private=False))
        self.assertEqual(lineage.execution_guard(current["prepared_parent"], env, event, current["prepared_parent"]), current)
        for mutation in (dict(GITHUB_RUN_ATTEMPT="2"), dict(GITHUB_EVENT_NAME="workflow_dispatch"), dict(GITHUB_REF="refs/heads/main")):
            with self.assertRaises(ValueError): lineage.execution_guard(current["prepared_parent"], dict(env, **mutation), event, current["prepared_parent"])
        for mutation in (dict(forced=True), dict(before="c" * 40), dict(repository=dict(full_name=lineage.REPO, private=True))):
            with self.assertRaises(ValueError): lineage.execution_guard(current["prepared_parent"], env, dict(event, **mutation), current["prepared_parent"])

    def test_wrong_run_authorization_cannot_consume_new_source(self):
        current, wrong = identity(), authorization_fixture(); wrong["source_execution"]["run_head"] = "c" * 40
        with patch.object(lineage, "execution_guard", return_value=current), \
             patch.object(lineage, "require_authorization", return_value=wrong), \
             patch.object(lineage, "history_receipt") as history, patch.object(source.B13, "render_panel") as render:
            with self.assertRaisesRegex(ValueError, "before TTS"):
                source.generate(Path("/unused"), Path("/unused-prior"), current["prepared_parent"])
            history.assert_not_called(); render.assert_not_called()

    def test_new_voice_map_excludes_canonical_history_under_changed_prefix(self):
        _, groups, people = lineage.validate_history(history_fixture())
        config = lineage.canary.previous.PLANS / "piper-config.json"
        source.validate_config(config, groups, people)
        with self.assertRaisesRegex(ValueError, "all real old exposure"):
            source.validate_config(config, groups | {"renamed-librispeech-2581"}, people)
        with self.assertRaises(ValueError): source.validate_config(config, groups, people | {"2581"})

    def test_changed_row_label_rate_order_and_source_counts_fail(self):
        receipt, auth = history_fixture(), authorization_fixture()
        self.assertEqual(len(source.validate_manifest(manifest_fixture(), receipt, auth)), 96)
        for mutation in (lambda m: m["rows"].pop(), lambda m: m["rows"].reverse(),
                         lambda m: m["rows"][0].update(observed_class=2),
                         lambda m: m["rows"][0].update(length_scale=.92),
                         lambda m: m.update(tts_public_synthesize_api_invocations=95),
                         lambda m: m.update(internal_tts_model_forwards=96),
                         lambda m: m.update(B19_canary_source_admission_allowed=True)):
            meta = manifest_fixture(); mutation(meta)
            with self.assertRaises(ValueError): source.validate_manifest(meta, receipt, auth)

    def test_saved_originals_exact_registration_and_evaluation_preflight(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(lineage, "verify_b19") as proof:
            root = Path(tmp); saved_fixture(root)
            rows = source.read_source(root); reg = source.registration(root); preflight = source.evaluation_preflight(root)
            self.assertEqual(len(rows), 96); self.assertEqual(preflight["rows"], rows)
            self.assertEqual(reg["source_rows"], 96); self.assertEqual(reg["max_calls"], 96)
            self.assertEqual(reg["source_manifest_sha256"], source.qa.sha(root / "MANIFEST.json"))
            self.assertEqual(preflight["source_execution"], identity())
            self.assertEqual((len(preflight["b19_canary_pcm_sha256"]), len(preflight["b19_canary_people"])), (48, 8))
            proof.assert_called_with(root / "B19_PROOF")

    def test_missing_original_canary_proof_stops_before_current_wav_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); saved_fixture(root)
            with patch.object(lineage, "verify_b19", side_effect=ValueError("ORIGINAL_CANARY_ZIP_MISSING")), \
                 patch.object(source.qa.pcm_tools, "inspect_wav") as wave_reader:
                with self.assertRaisesRegex(ValueError, "ORIGINAL_CANARY_ZIP_MISSING"): source.read_source(root)
                wave_reader.assert_not_called()

    def test_original_pcm_native_chunks_and_source_attempts_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(lineage, "verify_b19"):
            root = Path(tmp); meta = saved_fixture(root); row = meta["rows"][0]
            path = root / "native" / row["id"] / "000.s16le"; raw = path.read_bytes(); path.write_bytes(bytes(len(raw)))
            with self.assertRaisesRegex(ValueError, "exact native chunk bytes"): source.read_source(root)
            path.write_bytes(raw)
            write(root / "starts" / (row["id"] + ".json"), dict(id=row["id"], attempt=2, seed=row["seed"]))
            with self.assertRaisesRegex(ValueError, "exactly one fixed source request"): source.read_source(root)
        with tempfile.TemporaryDirectory() as tmp, patch.object(lineage, "verify_b19"):
            root = Path(tmp); meta = saved_fixture(root)
            path = root / "wav" / (meta["rows"][0]["id"] + ".wav"); raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "whole original WAV/PCM16 bytes"): source.read_source(root)

    def test_screen_must_match_actual_raw_recomputation_and_minima(self):
        expected = dict(source_gate="SOURCE_GATE_PASS", coverage=dict.fromkeys(source.observers.PROVIDERS, 96),
                        source_coverage=dict(positive_originals=12, ordinary_originals=24, paired_source_groups=8))
        with patch.object(source, "screen_document", return_value=expected) as raw:
            self.assertEqual(source.verify_screen_contract(Path("/src"), expected, Path("/providers")), expected)
            raw.assert_called_once_with(Path("/src"), Path("/providers"))
            changed = copy.deepcopy(expected); changed["coverage"]["parakeet"] = 95
            with self.assertRaisesRegex(ValueError, "exact recomputed"): source.verify_screen_contract(Path("/src"), changed, Path("/providers"))
        expected["source_gate"] = "INSUFFICIENT_SOURCE"
        with patch.object(source, "screen_document", return_value=expected), self.assertRaisesRegex(ValueError, "minima"):
            source.verify_screen_contract(Path("/src"), expected, Path("/providers"))

    def test_workflow_keeps_exact_first_public_parent_and_five_native_jobs(self):
        path = Path(__file__).resolve().parents[1] / ".github/workflows/english-v6-b20-synthetic-primary-once.yml"
        text = path.read_text()
        pins = {"checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1", "setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",
                "download-artifact": "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c", "upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02"}
        for action, commit in re.findall(r"uses: actions/([a-z-]+)@([^\s]+)", text):
            self.assertRegex(commit, r"\A[0-9a-f]{40}\Z"); self.assertEqual(commit, pins[action])
        self.assertIn("github.run_attempt == 1", text); self.assertIn("github.event.repository.private == false", text)
        self.assertEqual(text.count("__B20_SYNTHETIC_MAIN_SHA__"), 2)
        self.assertIn("provider: [qwen, parakeet, xlsr, lv60, whisper]", text)
        self.assertIn("fail-fast: false", text); self.assertNotIn("workflow_dispatch", text)
        section = text.split("  observers:\n", 1)[1].split("  readback:\n", 1)[0]
        self.assertLess(section.index("verify-source"), section.index("python -m venv"))
        self.assertNotIn("train-clean-100.tar.gz", text)


if __name__ == "__main__":
    unittest.main()
