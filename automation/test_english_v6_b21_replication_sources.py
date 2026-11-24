"""Focused replication opening, history and source wiring checks; no models."""
from __future__ import annotations
import copy
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_replication_source as source
import english_v6_b21_replication_source_ancestry as lineage
import english_v6_b21_natural_amendment_replication_natural_sources as natural
import english_v6_b21_replication_source_producer as producer


@contextmanager
def sealed_opening():
    value = json.loads(lineage.OPENING.read_bytes())
    value.update(status="SEALED_UNTIL_ORIGINAL_PRIMARY_PASS", quality_readback_sha256=None, primary_quality_lock=None)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "opening.json"; path.write_text(json.dumps(value))
        with patch.object(lineage, "OPENING", path): yield Path(tmp)


def primary_fixture():
    result_sha = "a" * 64
    inventory = dict(original_count=120,
        pcm_sha256=sorted(hashlib.sha256(("unit-primary-" + str(i)).encode()).hexdigest() for i in range(120)),
        source_group_id=[f"unit-primary-group-{i:02d}" for i in range(36)],
        canonical_people=sorted({p.removeprefix("p") for p in lineage.read_plan()["primary"]["synthetic"]["speaker_id_map"].values()}
                                | set(lineage.registered()["primary_design"]["speakers"])))
    summary = dict(panel_gate="PASS", model_pt_sha256=lineage.MODEL_PT,
        assessment_sha256=lineage.ASSESSMENT_SHA, training_design_sha256=lineage.DESIGN_SHA,
        original_result_sha256=result_sha, pair_pin=lineage.PAIR, admission={}, source_inventory=inventory,
        original_identity=dict(run_id="1234", run_head="b" * 40, run_attempt=1, prepared_parent="c" * 40))
    opening = dict(primary_quality_lock=dict(original_result_sha256=result_sha,
        result=dict(run_id=1234, run_head="b" * 40)))
    return summary, opening


class ReplicationSources(unittest.TestCase):
    def test_frozen_reader_and_source_metadata(self):
        source.check()
        rows = source.source_plan("replication")["rows"]
        design = lineage.read_plan()
        self.assertEqual(len(rows), 96)
        self.assertEqual({r["speaker_index"] for r in rows}, set(range(870, 894)))
        self.assertEqual({r["panel"] for r in rows}, {"replication"})
        self.assertFalse(set(design["replication"]["natural"]["speakers"]) & set(design["primary"]["natural"]["speakers"]))
        self.assertEqual((len(natural.plan()["speakers"]), natural.plan()["source_slots"]), (12, 24))
        with self.assertRaises(ValueError): source.source_plan("primary")
        self.assertNotIn("torch", sys.modules)

    def test_sealed_opening_stops_before_directory_fetch_corpus_or_native(self):
        with sealed_opening() as tmp:
            with patch.object(lineage, "quality_reader", side_effect=AssertionError("must not load reader")) as reader, \
                 patch.object(natural, "verify_corpus", side_effect=AssertionError("must not open corpus")) as corpus, \
                 patch.object(source.B13, "render_panel", side_effect=AssertionError("must not synthesize")) as render, \
                 patch.object(source.observers, "observe", side_effect=AssertionError("must not observe")) as observe:
                with self.assertRaisesRegex(ValueError, "replication remains sealed"):
                    lineage.authorize(tmp / "authorization", "1" * 40)
                with self.assertRaisesRegex(ValueError, "replication remains sealed"):
                    source.generate(tmp / "output", tmp / "authorization", tmp / "corpus", "1" * 40)
                with self.assertRaisesRegex(ValueError, "replication remains sealed"):
                    source.observe("qwen", tmp / "source", tmp / "native", "1" * 40)
                with self.assertRaisesRegex(ValueError, "replication remains sealed"):
                    source.read_source(tmp / "source")
                self.assertFalse((tmp / "authorization").exists()); self.assertFalse((tmp / "output").exists())
                for fn in (reader, corpus, render, observe): fn.assert_not_called()

    def test_event_and_producer_reject_replay_private_wrong_parent_or_extra_changes(self):
        prepared, head = "1" * 40, "2" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=lineage.REPO, GITHUB_REF=lineage.BRANCH,
            GITHUB_EVENT_NAME="push", GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID="123", GITHUB_SHA=head)
        event = dict(before=prepared, after=head, ref=lineage.BRANCH, forced=False, deleted=False,
            repository=dict(full_name=lineage.REPO, private=False))
        with patch.object(lineage, "read_opening", return_value={}):
            self.assertEqual(lineage.execution_guard(prepared, env, event, prepared)["run_head"], head)
            for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", source.primary.lineage.BRANCH)):
                with self.assertRaises(ValueError): lineage.execution_guard(prepared, dict(env, **{key: value}), event, prepared)
            with self.assertRaises(ValueError): lineage.execution_guard(prepared, env, event, "3" * 40)
        self.assertEqual(producer.validate_event(prepared, head, [prepared], [lineage.WORKFLOW], env, event)["role"], "replication_source")
        for paths in ([lineage.WORKFLOW, "automation/changed.py"], []):
            with self.assertRaises(ValueError): producer.validate_event(prepared, head, [prepared], paths, env, event)
        bad = copy.deepcopy(event); bad["repository"]["private"] = True
        with self.assertRaises(ValueError): producer.validate_event(prepared, head, [prepared], [lineage.WORKFLOW], env, bad)

    def test_complete_primary_inventory_added_without_original_or_unknown_omission(self):
        summary, opening = primary_fixture()
        with patch.object(lineage, "read_opening", return_value=opening), patch.object(lineage, "opening_sha256", return_value="f" * 64):
            history = lineage.make_history(summary)
            pcm, groups, people = lineage.validate_history(history)
            self.assertEqual((len(pcm), len(groups), len(people)), (2463, 416, 336))
            bad = copy.deepcopy(history); bad["all_pcm_sha256"].pop()
            with self.assertRaises(ValueError): lineage.validate_history(bad)
            bad = copy.deepcopy(summary); bad["source_inventory"]["pcm_sha256"].pop()
            with self.assertRaises(ValueError): lineage.make_history(bad)
            bad = copy.deepcopy(summary); bad["source_inventory"]["pcm_sha256"][0] = lineage.previous.read_ancestry()["all_pcm_sha256"][0]
            bad["source_inventory"]["pcm_sha256"].sort()
            with self.assertRaises(ValueError): lineage.make_history(bad)

    def test_failed_wrong_model_or_wrong_original_primary_cannot_open(self):
        summary, opening = primary_fixture()
        with patch.object(lineage, "read_opening", return_value=opening):
            for field, bad_value in (("panel_gate", "FAIL"), ("original_result_sha256", "d" * 64),
                                     ("model_pt_sha256", dict(lineage.MODEL_PT, challenger="e" * 64))):
                bad = copy.deepcopy(summary); bad[field] = bad_value
                with self.assertRaises(ValueError): lineage.validate_primary(bad)
            bad = copy.deepcopy(summary); bad["original_identity"]["run_attempt"] = 2
            with self.assertRaises(ValueError): lineage.validate_primary(bad)

    def test_fixed_natural_metadata_ranking_and_no_substitution(self):
        design = natural.plan(); transcripts = []
        for sp in design["speakers"]:
            text = "".join(f"{sp}-1-{i:04d} PLEASE BRING THE PACKAGE INSIDE\n" for i in range(5))
            transcripts.append(dict(path=f"LibriSpeech/train-clean-100/{sp}/1/{sp}-1.trans.txt", text=text,
                sha256=lineage.digest(text.encode())))
        catalog = dict(schema="english-v6-b21-replication-natural-original-metadata-v1",
            speakers_text=(lineage.PLANS / "SPEAKERS.TXT").read_text(), transcripts=sorted(transcripts, key=lambda r:r["path"]))
        rows = natural.original.rank_metadata(catalog, design)
        self.assertEqual([r["speaker"] for r in rows], [sp for sp in design["speakers"] for _ in range(2)])
        self.assertEqual(set(json.loads(natural.roster_bytes(rows))), {"replication"})
        bad = copy.deepcopy(catalog); bad["transcripts"].pop()
        with self.assertRaises(ValueError): natural.original.rank_metadata(bad, design)

    def test_generator_preserves_frozen_renderer_manifest_before_combined_receipt(self):
        identity = dict(run_id="123", run_head="2" * 40, prepared_parent="1" * 40,
            run_attempt=1, run_branch=lineage.BRANCH)
        def render(out, plan, plan_sha, seen, validate_config):
            self.assertEqual(plan["panel"], "replication"); self.assertEqual(plan_sha, source.FILES["replication"])
            out.mkdir()
            source.qa.record(out / "MANIFEST.json", dict(schema=plan["schema"], panel=plan["panel"], rows=plan["rows"]))
        def retain(archives, out):
            for name in ("HISTORY.json", "AUTHORIZATION.json"): (out / name).write_bytes((archives / name).read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); archives = tmp / "prior"; archives.mkdir()
            source.qa.record(archives / "HISTORY.json", {})
            source.qa.record(archives / "AUTHORIZATION.json", {"source_execution": identity})
            with patch.object(lineage, "execution_guard", return_value=identity), \
                 patch.object(lineage, "require_authorization", return_value={"source_execution": identity}), \
                 patch.object(lineage, "validate_history", return_value=(set(), set(), set())), \
                 patch.object(lineage, "opening_sha256", return_value="f" * 64), \
                 patch.object(lineage, "retain_proof", side_effect=retain), \
                 patch.object(lineage, "require_generation_permit", return_value={}), \
                 patch.object(natural, "verify_corpus", return_value="a" * 64), \
                 patch.object(natural, "extract", return_value={"rows": [{"id": "natural-" + str(i)} for i in range(24)]}), \
                 patch.object(source.B13, "render_panel", side_effect=render), \
                 patch.object(source, "read_source", return_value=[]) as readback:
                source.generate(tmp / "output", archives, tmp / "corpus", "1" * 40)
            out = tmp / "output"; meta = json.loads((out / "MANIFEST.json").read_bytes())
            self.assertEqual(meta["schema"], source.SCHEMA); self.assertEqual(meta["panel"], "replication")
            self.assertEqual(meta["synthetic"]["schema"], source.source_plan()["schema"])
            self.assertEqual(meta["synthetic_manifest_sha256"], source.qa.sha(out / "SYNTHETIC_MANIFEST.json"))
            self.assertEqual((len(meta["rows"]), meta["selected_original_slots"], meta["observer_eligible_originals"]), (120, 120, 120))
            self.assertTrue(meta["replication_opened"]); readback.assert_called_once_with(out)

    @contextmanager
    def observer_source_fixture(self):
        rows = copy.deepcopy(source.source_plan()["rows"])
        for i, row in enumerate(rows):
            row["pcm_sha256"] = hashlib.sha256(("replica-observer-" + str(i)).encode()).hexdigest()
        for i in range(24):
            rows.append(dict(id=f"b21-replication-natural-fixture{i}",
                pcm_sha256=hashlib.sha256(("replica-natural-" + str(i)).encode()).hexdigest(),
                cohort="natural", role="development", training_allowed=False, observed_class=None,
                requested_class=0, whole_original=True))
        for row in rows: row.update(wav_sha256=row["pcm_sha256"], samples=16000)
        identity = dict(run_id="123", run_head="2" * 40, prepared_parent="1" * 40,
            run_attempt=1, run_branch=lineage.BRANCH)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); src = root / "source"; src.mkdir(); (src / "MANIFEST.json").write_text("{}")
            stack.enter_context(patch.object(lineage, "execution_guard", return_value=identity))
            stack.enter_context(patch.object(lineage, "require_observer_permit", return_value={"fixture": True}))
            stack.enter_context(patch.object(lineage, "validate_observer_budget", return_value={}))
            stack.enter_context(patch.object(source, "_read", return_value=(rows, rows, {}, {"source_execution": identity})))
            # Keep real replica owner registration, guard, source identity and
            # schema validation. Only the authenticated source content is a fixture.
            registration = source.registration(src)
            yield root, src, rows, registration

    def test_replica_observe_reaches_native_init_only_after_actual_owner_and_row_guards(self):
        observers = source.observers
        with self.observer_source_fixture() as (root, src, rows, registration):
            with patch.object(observers, "_load_phone", side_effect=RuntimeError("FIXTURE_NATIVE_INIT_BOUNDARY")) as load:
                with self.assertRaisesRegex(ValueError, "WHOLE_PANEL_STOP"):
                    source.observe("xlsr", src, root / "native", "1" * 40)
                load.assert_called_once()
            report = json.loads((root / "native/RESULT.json").read_bytes())
            self.assertIn("FIXTURE_NATIVE_INIT_BOUNDARY", report["initialization_error"])
            self.assertEqual(report["schema"], source.BATCH)
            self.assertEqual((report["denominator"], report["actual_native_forward_attempts"]), (120, 0))
            self.assertEqual(registration["observer_sha256"], observers.sha(Path(observers.__file__).read_bytes()))
            contract = source.code_contract()
            self.assertEqual(contract["source_indices"], list(range(870, 894)))
            self.assertEqual(contract["frozen_primary_observer_sha256"], source.primary.code_contract()["observer_sha256"])

    def test_replica_observe_records_all120_fake_returns_and_requires_all120(self):
        observers = source.observers
        import english_hifi_micro as micro
        with self.observer_source_fixture() as (root, src, rows, registration):
            by_id = {r["id"]: r for r in rows}; calls = []
            def backend(wav, directory, start):
                calls.append(wav.stem)
                for name in ("FORWARD_STARTED.json", "FORWARD_RETURNED.json", "RAW_PHONE.json", "PAYLOAD_VALID.json"):
                    source.qa.record(directory / name, {"fixture": True})
                return {"fixture": True}
            with patch.object(observers, "_load_phone", return_value=(backend, {"fixture": True})), \
                 patch.object(micro, "inspect_wav", side_effect=lambda path: by_id[path.stem]):
                report = source.observe("xlsr", src, root / "native", "1" * 40)
            self.assertEqual((len(calls), report["valid_native_payloads"], report["status"]), (120, 120, "COMPLETE"))
            ledger = observers.row_ledger(root / "native", rows)
            self.assertTrue(observers.complete_ledger(ledger, 120))
            self.assertFalse(observers.complete_ledger(ledger[:-1], 120))

    def test_replica_readback_front_guards_accept_only_actual_replica_identity(self):
        observers = source.observers
        with self.observer_source_fixture() as (root, src, rows, registration):
            evidence = root / "providers"; evidence.mkdir()
            # Real registration/owner/row checks must pass and reach the first
            # missing raw receipt; a primary hardcode would fail earlier.
            with self.assertRaises(FileNotFoundError) as caught:
                observers.load_observations(src, evidence, rows, source.BATCH, registration=registration)
            self.assertTrue(str(caught.exception.filename).endswith("OBSERVER_BUDGET.json"))
            bad = copy.deepcopy(rows); bad[0]["id"] = "b21-primary-wrong-panel"
            with self.assertRaises(ValueError): observers.validate_rows(bad, registration["source_manifest_sha256"], registration)
            with self.assertRaises(ValueError):
                observers.load_observations(src, evidence, rows, source.primary.BATCH, registration=registration)

    def test_workflow_is_closed_and_preserves_original_proofs(self):
        template = Path(__file__).with_name("english_v6_b21_replication_source_workflow_template.yml").read_text()
        self.assertIn("__B21_REPLICATION_SOURCE_MAIN_SHA__", template)
        self.assertIn("github.run_attempt == 1", template)
        self.assertIn("include-hidden-files: true", template)
        self.assertIn("en-v6-b21-replication-panel-opened", template)
        self.assertIn('verify-corpus "$RUNNER_TEMP/train-clean-100.tar.gz" "$RUNNER_TEMP/prior-source"', template)
        self.assertLess(template.index(' authorize '), template.index('english_v6_b21_replication_transport.py download '))
        self.assertLess(template.index('verify-source'), template.index('Install one pinned native observer runtime'))


if __name__ == "__main__": unittest.main()
