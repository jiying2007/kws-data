"""Offline canary authorization, original-byte and never-admit fixtures."""
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
import english_v6_b19_source as source
import english_v6_b19_source_ancestry as lineage


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity():
    return dict(run_id="1234567", run_head="a" * 40, run_attempt=1,
                run_branch=lineage.BRANCH, prepared_parent="b" * 40)


def authorization():
    proof = lineage.read_plan()["original_B18_source_proof"]
    return dict(schema=lineage.SCHEMA, assessment_sha256=lineage.ASSESSMENT_SHA,
        source_execution=identity(), original_B18_source_proof=lineage.proof_locks(),
        original_B18_prepared_parent=proof["prepared_parent"],
        original_B18_run=dict(id=proof["run_id"], head_sha=proof["run_head"], head_branch=proof["run_branch"],
                             status="completed", conclusion="failure", run_attempt=1),
        B18_KWS_runs={"total_count": 0, "workflow_runs": []}, old_source_failures_preserved=True,
        all_prior_replication_sealed=True, diagnostic_only=True, source_admission_eligible=False, new_KWS_forwards=0)


def history_fixture():
    """Only fixture assembly; never emits a source-generation receipt."""
    seen, groups, people = lineage.registered_history()
    pin = lineage.read_plan()["original_B18_source_proof"]["durable_evidence"]["HISTORY.json"]
    archived = json.loads((lineage.PLANS.parent.parent / pin["path"]).read_bytes())["original_archives"]
    archives = copy.deepcopy(archived)
    archives.update({str(pin["id"]): {"bytes": pin["bytes"], "sha256": pin["sha256"]}
                     for pin in lineage.proof_locks().values()})
    return dict(schema=lineage.SCHEMA, assessment_sha256=lineage.ASSESSMENT_SHA,
        original_archives=archives, all_pcm_sha256=sorted(seen), all_source_groups=sorted(groups),
        canonical_people=sorted(people), old_B17_B18_source_failures_preserved=True, all_prior_replication_sealed=True)


def manifest_fixture():
    plan = source.source_plan()
    rows = [dict(row, pcm_sha256=digest("offline-pcm-" + row["id"]),
                 wav_sha256=digest("offline-wav-" + row["id"]), samples=8000, rms=.01,
                 rail_fraction=0, basic_audio_ok=True, whole_native_resample=True, crop_calls=0,
                 native_pcm_sha256=digest("offline-native-" + row["id"]),
                 native_chunks=[dict(file="000.s16le", samples=11025,
                                     native_pcm_sha256=digest("offline-native-" + row["id"]))])
            for row in plan["rows"]]
    return dict(schema=source.SCHEMA, panel="feasibility", rows=rows,
        registered_plan_file_sha256=source.PLAN_SHA,
        plan_sha256=source.B13.b10.sha(source.B13.b10.canon(plan)),
        assessment_sha256=source.ASSESSMENT_SHA, source_ancestry_sha256=lineage.BASE_ANCESTRY_SHA,
        source_execution=identity(), tts_calls=48, tts_logical_originals=48,
        tts_public_synthesize_api_invocations=48, native_output_chunks=48,
        internal_tts_model_forwards=None, internal_onnx_session_run_calls=None, internal_tts_forwards_instrumented=False,
        new_kws_forwards=0, human_gold=False, diagnostic_only=True, source_admission_eligible=False,
        evaluation_allowed=False, training_allowed=False, promotion_allowed=False, future_source_execution_allowed=False,
        all_prior_replication_sealed=True, B17_source_failure_preserved=True, B18_source_STOP_preserved=True,
        planned_permanent_canonical_exclusions=sorted(p.removeprefix("p") for p in
              lineage.read_plan()["source_maps"]["feasibility"].values()),
        generated_permanent_PCM_exclusions=sorted(row["pcm_sha256"] for row in rows))


def saved_fixture(root):
    """Deterministic artificial byte fixtures; no TTS, ASR or speech source."""
    meta = manifest_fixture()
    write(root / "AUTHORIZATION.json", authorization())
    write(root / "HISTORY.json", history_fixture())
    write(root / "STARTED.json", {"plan_sha256": source.PLAN_SHA, "one_shot": True})
    for number, row in enumerate(meta["rows"]):
        write(root / "starts" / (row["id"] + ".json"), {"id": row["id"], "attempt": 1, "seed": row["seed"]})
        path = root / "wav" / (row["id"] + ".wav")
        path.parent.mkdir(exist_ok=True)
        sample = struct.pack("<h", 1000 + number)
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 8000, "NONE", "not compressed"))
            wav.writeframes(sample * 8000)
        row.update(source.qa.pcm_tools.inspect_wav(path))
        native = root / "native" / row["id"] / "000.s16le"
        native.parent.mkdir(parents=True)
        native.write_bytes(sample * 11025)
        row["native_pcm_sha256"] = source.qa.sha(native)
        row["native_chunks"][0]["native_pcm_sha256"] = source.qa.sha(native)
    meta["generated_permanent_PCM_exclusions"] = sorted(row["pcm_sha256"] for row in meta["rows"])
    meta["history_receipt_sha256"] = source.qa.sha(root / "HISTORY.json")
    meta["source_authorization_sha256"] = source.qa.sha(root / "AUTHORIZATION.json")
    write(root / "MANIFEST.json", meta)
    return meta


def guarded_inputs():
    current = identity()
    env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": lineage.REPO,
           "GITHUB_EVENT_NAME": "push", "GITHUB_REF": lineage.BRANCH, "GITHUB_RUN_ATTEMPT": "1",
           "GITHUB_RUN_ID": current["run_id"], "GITHUB_SHA": current["run_head"]}
    event = dict(before=current["prepared_parent"], after=current["run_head"], ref=lineage.BRANCH,
                 forced=False, deleted=False, repository=dict(full_name=lineage.REPO, private=False))
    return current, env, event


class B19Sources(unittest.TestCase):
    def test_exact_fixed_cells_and_diagnostic_authority(self):
        plan, design = source.source_plan(), lineage.read_plan()
        self.assertEqual(len(plan["rows"]), 48)
        self.assertEqual({r["speaker_index"] for r in plan["rows"]}, set(range(790, 798)))
        self.assertEqual(sum(r["requested_class"] == 2 for r in plan["rows"]), 32)
        self.assertEqual({r["length_scale"] for r in plan["rows"] if r["feasibility_arm"] == "prospective"}, {1.4, 1.8})
        self.assertEqual(source.code_contract(), design["observer_code_contract"])
        self.assertTrue(design["all_8_planned_identities_excluded_even_if_never_called"])
        self.assertIsNone(design["internal_tts_model_forwards"])
        for key in ("source_admission_eligible", "new_training_allowed", "model_evaluation_allowed", "promotion_allowed"):
            self.assertIs(design[key], False)

    def test_only_feasibility_plan_can_be_read_for_execution(self):
        for panel in ("primary", "replication", "future_primary", "future_replication"):
            with self.subTest(panel=panel), self.assertRaisesRegex(ValueError, "remain sealed"):
                source.source_plan(panel)

    def test_all_44_original_archives_and_exact_full_exposure(self):
        receipt = history_fixture()
        self.assertEqual(len(lineage.required_archives()), 44)
        self.assertEqual(tuple(len(v) for v in lineage.validate_history(receipt)), (2175, 336, 268))
        proof = lineage.read_plan()["original_B18_source_proof"]
        self.assertEqual(proof["total_native_observers"], dict(attempted=390, returned=390, valid=389, failed=1, not_started=90))
        self.assertIsNone(proof["synthetic_screen"])
        self.assertIn(proof["failed_original"]["pcm_sha256"], receipt["all_pcm_sha256"])

    def test_missing_duplicate_or_changed_historical_identity_is_rejected(self):
        for key in ("all_pcm_sha256", "all_source_groups", "canonical_people"):
            for mode in ("missing", "duplicate", "substitute"):
                receipt = history_fixture()
                if mode == "missing": receipt[key].pop()
                elif mode == "duplicate": receipt[key].append(receipt[key][0])
                else: receipt[key][0] = "offline-changed-history"
                with self.subTest(key=key, mode=mode), self.assertRaises(ValueError):
                    lineage.validate_history(receipt)

    def test_historical_zip_digest_or_original_B18_length_cannot_change(self):
        receipt = history_fixture()
        first = next(iter(receipt["original_archives"]))
        receipt["original_archives"][first]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): lineage.validate_history(receipt)
        receipt = history_fixture()
        receipt["original_archives"][str(lineage.proof_locks()["source"]["id"])]["bytes"] += 1
        with self.assertRaises(ValueError): lineage.validate_history(receipt)

    def test_actual_config_all_56_canonical_people_and_reserved_natural(self):
        _, groups, people = lineage.registered_history()
        config = source.previous.PLANS / "piper-config.json"
        source.validate_config(config, groups, people)
        with self.assertRaisesRegex(ValueError, "canonical source"):
            source.validate_config(config, groups | {"changed-prefix-librispeech-reader-2254"}, people)
        design = copy.deepcopy(source.previous.lineage.read_plan())
        design["replication"]["natural"]["speakers"][0] = "2254"
        with patch.object(source.previous.lineage, "read_plan", return_value=design), self.assertRaisesRegex(ValueError, "canonical source"):
            source.validate_config(config, set(), set())

    def test_original_public_parent_guard_and_replay_rejection(self):
        current, env, event = guarded_inputs()
        self.assertEqual(lineage.execution_guard(current["prepared_parent"], env, event, current["prepared_parent"]), current)
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
                           ("GITHUB_REF", "refs/heads/main"), ("GITHUB_REPOSITORY", "another/repo")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                lineage.execution_guard(current["prepared_parent"], dict(env, **{key: value}), event, current["prepared_parent"])
        for mutation in (dict(forced=True), dict(deleted=True), dict(before="c" * 40),
                         dict(repository=dict(full_name=lineage.REPO, private=True))):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                lineage.execution_guard(current["prepared_parent"], env, dict(event, **mutation), current["prepared_parent"])

    def test_inert_placeholder_fails_before_any_remote_proof_access(self):
        with patch.object(lineage, "gh") as remote, self.assertRaisesRegex(ValueError, "inert placeholder"):
            lineage.authorize(Path("/unused-fixture"), "__B19_SOURCE_FEASIBILITY_MAIN_SHA__")
        remote.assert_not_called()

    def test_mismatched_authorization_stops_before_history_or_renderer(self):
        current, wrong = identity(), authorization()
        wrong["source_execution"]["run_head"] = "c" * 40
        with patch.object(lineage, "execution_guard", return_value=current), \
             patch.object(lineage, "require_authorization", return_value=wrong), \
             patch.object(lineage, "history_receipt") as history, patch.object(source.B13, "render_panel") as render:
            with self.assertRaisesRegex(ValueError, "before TTS"):
                source.generate(Path("/unused-output"), Path("/unused-prior"), current["prepared_parent"])
            history.assert_not_called(); render.assert_not_called()

    def test_all_48_bounds_include_baseline_and_prospective(self):
        receipt, permission = history_fixture(), authorization()
        self.assertEqual(len(source.validate_manifest(manifest_fixture(), receipt, permission)), 48)
        for cell, samples in (("baseline-fast", 6399), ("baseline-slow", 28673),
                              ("prospective-fast", 6399), ("prospective-slow", 28673),
                              ("question", 3199), ("request", 240001)):
            meta = manifest_fixture()
            next(row for row in meta["rows"] if row["cell_id"] == cell)["samples"] = samples
            with self.subTest(cell=cell), self.assertRaisesRegex(ValueError, "all 48 original bounds"):
                source.validate_manifest(meta, receipt, permission)

    def test_no_row_loss_replacement_reorder_label_or_admission(self):
        receipt, permission = history_fixture(), authorization()
        cases = (lambda d: d["rows"].pop(), lambda d: d["rows"].reverse(),
                 lambda d: d["rows"][0].update(observed_class=2),
                 lambda d: d["rows"][0].update(length_scale=1.6),
                 lambda d: d.update(source_admission_eligible=True),
                 lambda d: d.update(internal_tts_model_forwards=48),
                 lambda d: d["planned_permanent_canonical_exclusions"].pop())
        for number, mutation in enumerate(cases):
            meta = manifest_fixture(); mutation(meta)
            with self.subTest(case=number), self.assertRaises(ValueError):
                source.validate_manifest(meta, receipt, permission)

    def test_saved_bytes_registration_and_beforeload_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); meta = saved_fixture(root)
            rows = source.read_source(root)
            self.assertEqual(len(rows), 48)
            reg = source.registration(root)
            self.assertEqual({k: reg[k] for k in identity()}, identity())
            self.assertEqual(reg["source_manifest_sha256"], source.qa.sha(root / "MANIFEST.json"))
            self.assertEqual(reg["source_rows"], reg["max_calls"], 48)
            current = dict(identity(), run_head="d" * 40)
            with patch.object(lineage, "execution_guard", return_value=current), \
                 patch.object(source.observer, "observe") as backend, self.assertRaisesRegex(ValueError, "before model load"):
                source.observe(root, root / "observer", current["prepared_parent"])
            backend.assert_not_called()
            self.assertEqual(meta["planned_permanent_canonical_exclusions"],
                             ["1100", "122", "1323", "2093", "2254", "323", "81", "815"])

    def test_native_chunk_tamper_and_changed_source_start_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); meta = saved_fixture(root)
            row = meta["rows"][0]
            path = root / "native" / row["id"] / "000.s16le"
            original = path.read_bytes(); path.write_bytes(bytes(len(original)))
            with self.assertRaisesRegex(ValueError, "native TTS bytes"): source.read_source(root)
            path.write_bytes(original)
            write(root / "starts" / (row["id"] + ".json"), dict(id=row["id"], attempt=2, seed=row["seed"]))
            with self.assertRaisesRegex(ValueError, "one fixed diagnostic source request"): source.read_source(root)

    def test_wave_and_manifest_binding_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); meta = saved_fixture(root)
            path = root / "wav" / (meta["rows"][0]["id"] + ".wav")
            raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "unchanged complete original PCM16"): source.read_source(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); saved_fixture(root)
            auth = authorization(); auth["source_execution"]["run_head"] = "e" * 40
            write(root / "AUTHORIZATION.json", auth)
            with self.assertRaisesRegex(ValueError, "unchanged complete raw history"): source.read_source(root)

    def test_failed_diagnostic_is_saved_without_admission_authority(self):
        result = dict(source_admission_eligible=False, KWS_evaluation_allowed=False,
                      promotion_allowed=False, prospective_pair_feasible=False,
                      result="STOP_SOURCE_FORMAT_NOT_QUALIFIED")
        with tempfile.TemporaryDirectory() as tmp, patch.object(source, "registration", return_value={}), \
             patch.object(source.observer, "readback", return_value=result):
            path = Path(tmp) / "FEASIBILITY.json"
            with self.assertRaisesRegex(ValueError, "not qualified"):
                source.readback(Path(tmp), Path(tmp), path)
            self.assertEqual(json.loads(path.read_bytes()), result)

    def test_workflow_exact_action_pins_guards_and_preinstall_readback(self):
        path = Path(__file__).resolve().parents[1] / ".github/workflows/english-v6-b19-source-feasibility-once.yml"
        text = path.read_text()
        allowed = dict(checkout="3d3c42e5aac5ba805825da76410c181273ba90b1",
                       **{"setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",
                          "download-artifact": "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
                          "upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02"})
        refs = re.findall(r"uses: actions/([a-z-]+)@([^\s]+)", text)
        self.assertGreaterEqual(len(refs), 10)
        for action, commit in refs:
            self.assertRegex(commit, r"\A[0-9a-f]{40}\Z")
            self.assertEqual(commit, allowed[action])
        self.assertIn("github.run_attempt == 1", text)
        self.assertIn("github.event.repository.private == false", text)
        self.assertEqual(text.count("__B19_SOURCE_FEASIBILITY_MAIN_SHA__"), 2)
        self.assertNotIn("workflow_dispatch", text)
        self.assertNotIn("matrix:", text)
        section = text.split("  observer:\n", 1)[1].split("  readback:\n", 1)[0]
        self.assertLess(section.index("verify-source"), section.index("python -m venv"))
        self.assertLess(section.index("source.py guard"), section.index("verify-source"))
        self.assertEqual(text.count("source.py observe "), 1)
        self.assertIn("needs: sources", section)


if __name__ == "__main__":
    unittest.main()
